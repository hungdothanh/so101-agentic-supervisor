"""SO-101 MuJoCo gamepad teleoperator.

Reuses so101_mujoco_env.joint_teleop.GamepadJointTeleop for the actual gamepad reading
(deadzone handling, per-controller-pad profile lookup, button-edge detection) instead of
reimplementing it here. This is a deliberate exception to this package's usual
"stay self-contained" rule (see so101_mujoco.py's module docstring, which duplicates a
few numeric constants rather than importing so101_mujoco_env) -- duplicating actual
STATEFUL LOGIC like gamepad reading would create a second implementation that can drift
out of sync with the one Phase 3 already built and verified against real hardware,
which is exactly the failure mode the earlier so101_gazebo_teleop.py's own docstring
calls out (there solved by making ROS the single source of truth instead; here solved
by importing the one existing implementation instead of forking it).

DELTA INTEGRATION LIVES HERE, NOT IN THE ROBOT: GamepadJointTeleop.get_action() returns
a normalized per-tick *velocity* in [-1, 1] per joint (the same interface
SO101PenPickPlaceEnv.step() consumes for its own RL action space) -- not an absolute
goal position. But SO101Mujoco.send_action() expects an already-absolute goal position
(degrees / [0, 100] gripper), matching SOFollower's real-hardware contract (see
so101_mujoco.py's module docstring). Reconciling those two shapes is this class's job:
get_action() integrates the raw gamepad delta onto an internally-tracked absolute
target every tick (same per-joint max-speed constants as
SO101PenPickPlaceEnv.step()/MAX_JOINT_SPEED_RAD_S, reproduced below since that env's
own integration is in a different unit space and driven by a different fixed-dt
assumption -- see the constants section), then reports the running target in degrees.

This mirrors how a real SO-101 *leader arm* teleoperator (so_leader.py) works only
superficially: there, get_action() just reads the leader's own physical joint encoders
directly (whatever pose a human is holding it at IS the target, no integration needed).
A gamepad has no such physical "current pose" of its own, so an integrator is
unavoidable somewhere; this class is where PLAN.md's design puts it, keeping
SO101Mujoco's send_action() a simple, hardware-realistic "write this absolute goal
and step" -- structurally identical to SOFollower.send_action().

STARTING-POSE ASSUMPTION: the integrator seeds its internal target from this scene's
own 'home' keyframe (HOME_CTRL_RAD below, copied verbatim from
assets/scenes/pen_pickplace_scene.xml's `<key name="home" .../>` ctrl vector) rather
than reading it from the robot, because the Teleoperator and Robot are constructed and
connected independently by lerobot-record (see record_loop() in the lerobot repo) --
there's no reference from one to the other. This is only a safe assumption because
SO101Mujoco.connect() (and reset_scene()) always return to this exact same keyframe;
it would NOT be a safe assumption for a general-purpose teleoperator paired with an
arbitrary robot.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np

from lerobot.teleoperators.teleoperator import Teleoperator
from lerobot.utils.errors import DeviceNotConnectedError

from .config_so101_mujoco_teleop import SO101MujocoTeleopConfig

# lerobot_robot_so101_mujoco/lerobot_robot_so101_mujoco/so101_mujoco_teleop.py ->
# parents[2] is the so101_mujoco_sim2real project root.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from so101_mujoco_env.joint_teleop import CONFIG_PATH as _DEFAULT_GAMEPAD_CONFIG_PATH  # noqa: E402
from so101_mujoco_env.joint_teleop import GamepadJointTeleop  # noqa: E402
from so101_mujoco_env.pen_pickplace_env import ARM_JOINTS as ENV_JOINT_ORDER  # noqa: E402

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"
ALL_JOINTS = ARM_JOINTS + (GRIPPER_JOINT,)

# Duplicated from so101_mujoco_env/pen_pickplace_env.py: that module's integration runs
# in radians against a fixed assumed control_dt; this one measures actual wall-clock dt
# per get_action() call instead (robust to whatever --dataset.fps a recording session
# picks), so the constant is reused but the integration loop itself is not shared code.
MAX_JOINT_SPEED_RAD_S = np.deg2rad(60.0)
MAX_GRIPPER_SPEED_RAD_S = np.deg2rad(120.0)
GRIPPER_RANGE_RAD = (-0.174533, 1.74533)

# Per-joint ctrlrange, copied verbatim from assets/so101/so101.xml's <position> actuators.
CTRL_RANGE_RAD = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": GRIPPER_RANGE_RAD,
}

# Copied verbatim from assets/scenes/pen_pickplace_scene.xml's home keyframe's ctrl
# vector -- see this module's docstring for why seeding from here (rather than reading
# the robot) is safe in this specific integrated system.
HOME_CTRL_RAD = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -1.75,
    "elbow_flex": 1.62,
    "wrist_flex": 1.18,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}


def _gripper_rad_to_pct(angle_rad: float) -> float:
    lo, hi = GRIPPER_RANGE_RAD
    return float(np.clip((angle_rad - lo) / (hi - lo) * 100.0, 0.0, 100.0))


class SO101MujocoTeleop(Teleoperator):
    config_class = SO101MujocoTeleopConfig
    name = "so101_mujoco_teleop"

    def __init__(self, config: SO101MujocoTeleopConfig):
        super().__init__(config)
        self.config = config

        gamepad_config_path = (
            Path(config.gamepad_config_path) if config.gamepad_config_path else _DEFAULT_GAMEPAD_CONFIG_PATH
        )
        self._gamepad = None if config.mock else GamepadJointTeleop(gamepad_config_path)

        self._target_rad: dict[str, float] = {}
        self._last_tick_t: float | None = None
        self._seed_home_target()

        self._mock_start_t: float | None = None
        self._is_connected = False

    def _seed_home_target(self) -> None:
        """(Re)seed the internally-tracked absolute joint target from the scene's own
        home keyframe (HOME_CTRL_RAD) and clear the wall-clock dt tracker, so the very
        next get_action() reports dt=0 (no motion) instead of integrating over however
        long has elapsed since the last tick. Used both at __init__ and by
        reset_target()."""
        self._target_rad = dict(zip(HOME_CTRL_RAD.keys(), HOME_CTRL_RAD.values(), strict=True))
        for joint in ALL_JOINTS:
            lo, hi = CTRL_RANGE_RAD[joint]
            self._target_rad[joint] = float(np.clip(self._target_rad[joint], lo, hi))
        self._last_tick_t = None

    def reset_target(self) -> None:
        """Reset the internally-tracked absolute joint target back to the scene's home
        pose. NOT part of the Teleoperator ABC -- a recording driver must call this
        every time it also calls SO101Mujoco.reset_scene(), since this teleoperator has
        no other way to learn that the robot's state just jumped: Teleoperator and
        Robot are connected independently by lerobot-record's record_loop() (see this
        module's "STARTING-POSE ASSUMPTION" docstring section above). Without this
        call, the next get_action() would keep integrating from wherever the arm was
        at the end of the previous episode and immediately fight the robot's own
        reset -- the sent target would jump from home back toward that stale pose."""
        self._seed_home_target()

    @property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys((f"{j}.pos" for j in ALL_JOINTS), float)

    @property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True  # a gamepad has no calibration concept

    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        if self._gamepad is not None:
            if not self._gamepad.connect():
                raise DeviceNotConnectedError(
                    "No gamepad detected. Connect one, or pass --teleop.mock=true to "
                    "smoke-test the recording plumbing without one."
                )
        self._last_tick_t = None
        self._mock_start_t = time.monotonic()
        self._is_connected = True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _dt(self) -> float:
        now = time.monotonic()
        if self._last_tick_t is None:
            dt = 0.0
        else:
            # Clamp against a stalled first tick / a long pause between episodes --
            # without this, one slow tick would otherwise integrate a huge, unintended
            # jump in target position.
            dt = min(now - self._last_tick_t, 0.5)
        self._last_tick_t = now
        return dt

    def _raw_gamepad_action(self) -> np.ndarray:
        """Normalized [-1, 1] per-joint delta, in ENV_JOINT_ORDER (6-wide, incl.
        gripper) -- either the real gamepad reading, or (if config.mock) a
        deterministic sine sweep used purely to exercise this plumbing headlessly."""
        if self._gamepad is not None:
            return self._gamepad.get_action()

        t = time.monotonic() - self._mock_start_t
        period = self.config.mock_period_s
        amp = self.config.mock_amplitude
        return np.array(
            [
                amp * math.sin(2.0 * math.pi * t / period + i * (2.0 * math.pi / len(ENV_JOINT_ORDER)))
                for i in range(len(ENV_JOINT_ORDER))
            ],
            dtype=np.float32,
        )

    def get_action(self) -> dict:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        dt = self._dt()
        raw = dict(zip(ENV_JOINT_ORDER, self._raw_gamepad_action(), strict=True))

        for joint in ARM_JOINTS:
            lo, hi = CTRL_RANGE_RAD[joint]
            target = self._target_rad[joint] + float(raw[joint]) * MAX_JOINT_SPEED_RAD_S * dt
            self._target_rad[joint] = float(np.clip(target, lo, hi))

        lo, hi = GRIPPER_RANGE_RAD
        gripper_target = self._target_rad[GRIPPER_JOINT] + float(raw[GRIPPER_JOINT]) * MAX_GRIPPER_SPEED_RAD_S * dt
        self._target_rad[GRIPPER_JOINT] = float(np.clip(gripper_target, lo, hi))

        action = {f"{j}.pos": float(np.rad2deg(self._target_rad[j])) for j in ARM_JOINTS}
        action[f"{GRIPPER_JOINT}.pos"] = _gripper_rad_to_pct(self._target_rad[GRIPPER_JOINT])
        return action

    def send_feedback(self, feedback: dict) -> None:
        pass  # no force/vibration feedback path

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._gamepad is not None:
            self._gamepad.disconnect()
        self._is_connected = False
