#!/usr/bin/env python3
"""Phase 3: gamepad -> joint-space teleop for the SO-101 MuJoCo env.

Direct in-process pygame polling, no ROS/topic layer -- unlike the earlier
Gazebo project's joint_velocity_teleop_node.py (so101_sim2real/so101_teleop),
which needed a deadman switch and a staleness check specifically because
/joy could silently stop publishing out from under it. That failure mode
doesn't exist here: this script reads the gamepad itself, every tick, in
the same process that's driving the sim, so there's no separate stream to
go stale.

Both sticks map to 4 joints proportionally (shoulder_pan, shoulder_lift,
wrist_flex, elbow_flex); wrist_roll uses the bumpers (digital -- no analog
axis is left once both sticks are claimed); the gripper uses the analog
triggers (squeeze R2 to close, L2 to open); press Y to reset the episode
(env.reset()) without restarting the script -- a press, not a hold, so it
fires once per button-down edge, not once per tick while held. See
config/gamepad.config.yaml for the exact axis/button numbering and how to
recalibrate it for a different pad -- indices are driver/OS-specific and
NOT portable as-is.

Verification (PLAN.md Phase 3): run with --print-only first to confirm the
gamepad produces sensible per-joint deltas as you move it, before wiring
those deltas into the running sim.

Run inside the `lerobot_smolvla` conda env:
    python -m so101_mujoco_env.joint_teleop --print-only   # verify mapping
    python -m so101_mujoco_env.joint_teleop                # drive the sim, free 3D viewer
    python -m so101_mujoco_env.joint_teleop --cameras       # same, plus a front+wrist feed window

--cameras opens an extra window alongside the free 3D viewer showing what
the policy (and a recorded episode) will actually see -- the free viewer's
god's-eye view makes maneuvers look easy that are much harder blind through
the two real cameras, so keeping both up while recording lets you use the
3D view for orientation but judge the actual approach by the camera feeds.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from .pen_pickplace_env import ARM_JOINTS, SO101PenPickPlaceEnv

CONFIG_PATH = Path(__file__).resolve().parent / "config" / "gamepad.config.yaml"


class GamepadJointTeleop:
    """Reads a connected gamepad and produces a normalized 6-dim action (one
    entry per ARM_JOINTS, in [-1, 1]) matching SO101PenPickPlaceEnv's
    action_space directly. Plain pygame is the only dependency -- lerobot's
    own GamepadTeleop/Teleoperator classes are built around a Cartesian
    delta_x/delta_y/delta_z action scheme (see teleop_gamepad.py), which
    doesn't fit a joint-space env, so this reimplements just the gamepad
    reading, not that class hierarchy.
    """

    def __init__(self, config_path: Path = CONFIG_PATH):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.deadzone = float(self.config["deadzone"])
        self.joystick = None
        self.profile: dict | None = None
        self._prev_button_state: dict[str, bool] = {}

    def connect(self) -> bool:
        import pygame

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            return False
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        name = self.joystick.get_name()
        known = self.config["controllers"].get(name)
        self.profile = known if known is not None else self.config["controllers"]["default"]
        print(f"Connected: {name!r} (profile: {name if known is not None else 'default'})")
        return True

    def _raw_axis(self, name: str) -> float:
        idx = self.profile["axes"].get(name)
        if idx is None or idx >= self.joystick.get_numaxes():
            return -1.0  # matches the trigger axes' own "released" rest value
        return self.joystick.get_axis(idx)

    def _axis(self, name: str) -> float:
        val = self._raw_axis(name) if name in self.profile["axes"] else 0.0
        if self.profile.get("axis_inversion", {}).get(name, False):
            val = -val
        return 0.0 if abs(val) < self.deadzone else val

    def _button(self, name: str) -> bool:
        idx = self.profile["buttons"].get(name)
        if idx is None or idx >= self.joystick.get_numbuttons():
            return False
        return bool(self.joystick.get_button(idx))

    def button_pressed_edge(self, name: str) -> bool:
        """True only on the tick `name` transitions from released to pressed,
        not for every tick it's held -- so a reset trigger fires once per
        press, not once per control tick while held."""
        cur = self._button(name)
        prev = self._prev_button_state.get(name, False)
        self._prev_button_state[name] = cur
        return cur and not prev

    def reset_requested(self) -> bool:
        return self.button_pressed_edge(self.config["reset_button"])

    def get_action(self) -> np.ndarray:
        """Normalized [-1, 1] action, one entry per ARM_JOINTS, ready to pass
        straight to SO101PenPickPlaceEnv.step()."""
        import pygame

        pygame.event.pump()
        action = dict.fromkeys(ARM_JOINTS, 0.0)

        for joint, axis_name in self.config["joint_axis_map"].items():
            action[joint] = self._axis(axis_name)

        neg = self._button(self.config["wrist_roll_neg_button"])
        pos = self._button(self.config["wrist_roll_pos_button"])
        action["wrist_roll"] = float(pos) - float(neg)

        # Each trigger's own range is [-1 (released), +1 (fully pressed)];
        # remap to [0, 1] before combining so a released trigger contributes 0.
        close_amt = (self._raw_axis(self.config["gripper_close_axis"]) + 1.0) / 2.0
        open_amt = (self._raw_axis(self.config["gripper_open_axis"]) + 1.0) / 2.0
        action["gripper"] = close_amt - open_amt

        return np.array([action[j] for j in ARM_JOINTS], dtype=np.float32)

    def disconnect(self) -> None:
        import pygame

        if self.joystick is not None:
            self.joystick.quit()
        pygame.joystick.quit()
        pygame.quit()


class CameraFeedWindow:
    """A cv2 window showing the front + wrist camera feeds side by side,
    alongside (not instead of) the free-roam passive 3D viewer -- i.e. an
    extra window showing exactly what an eventual policy (or a recorded
    episode) would see, so you can sanity-check the task from that
    perspective while still having the full-scene view for orientation."""

    def __init__(self, model, cam_size: int):
        import mujoco

        self._renderer = mujoco.Renderer(model, height=cam_size, width=cam_size)
        self.window = "SO-101 teleop: front | wrist"

    def update(self, data) -> bool:
        """Renders one frame and returns False if the window was closed."""
        import cv2

        frames = []
        for cam_name in ("front", "wrist"):
            self._renderer.update_scene(data, camera=cam_name)
            rgb = self._renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(bgr, cam_name, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frames.append(bgr)
        cv2.imshow(self.window, cv2.hconcat(frames))
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 27 == Esc
            return False
        return cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) >= 1

    def close(self) -> None:
        import cv2

        cv2.destroyWindow(self.window)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--print-only", action="store_true", help="Print joint deltas without touching the sim")
    parser.add_argument(
        "--cameras",
        action="store_true",
        help="Also open a front+wrist camera-feed window (what the policy will see) alongside the free 3D viewer",
    )
    parser.add_argument("--cam-size", type=int, default=480, help="Per-camera render resolution for --cameras")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    args = parser.parse_args()

    teleop = GamepadJointTeleop(args.config)
    if not teleop.connect():
        print("No gamepad detected -- connect one and try again.")
        return

    control_rate_hz = float(teleop.config["control_rate_hz"])
    dt = 1.0 / control_rate_hz

    if args.print_only:
        print("Printing joint deltas -- move the sticks/triggers/bumpers. Ctrl+C to stop.")
        try:
            while True:
                action = teleop.get_action()
                reset = teleop.reset_requested()
                line = " ".join(f"{j}={v:+.2f}" for j, v in zip(ARM_JOINTS, action))
                if reset:
                    line += "  [RESET pressed]"
                print(line)
                time.sleep(dt)
        except KeyboardInterrupt:
            pass
        finally:
            teleop.disconnect()
        return

    env = SO101PenPickPlaceEnv(control_dt=dt)
    env.reset()

    import mujoco.viewer

    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    cam_window = CameraFeedWindow(env.model, args.cam_size) if args.cameras else None
    reset_button = teleop.config["reset_button"]
    msg = (
        f"Driving the sim -- move the sticks/triggers/bumpers, press {reset_button.upper()} to reset. "
        "Close the viewer window to stop."
    )
    if cam_window is not None:
        msg += f" (also see '{cam_window.window}' for the front+wrist feeds)"
    print(msg)
    try:
        while viewer.is_running():
            tick_start = time.monotonic()
            action = teleop.get_action()
            if teleop.reset_requested():
                env.reset()
                print("Episode reset.")
            else:
                env.step(action)
            viewer.sync()
            if cam_window is not None and not cam_window.update(env.data):
                break
            elapsed = time.monotonic() - tick_start
            if elapsed < dt:
                time.sleep(dt - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
        if cam_window is not None:
            cam_window.close()
        viewer.close()
        env.close()


if __name__ == "__main__":
    main()
