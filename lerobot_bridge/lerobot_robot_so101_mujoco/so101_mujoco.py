"""SO-101 MuJoCo digital twin, exposed to LeRobot as a Robot.

Unlike SO101Gazebo (a ZMQ client to a separate ROS 2 process -- see the earlier
so101_sim2real project's lerobot_robot_so101_gazebo/so101_gazebo.py module docstring
for why: Gazebo + ros2_control lives in a Humble/Python-3.10 rclpy process that can't be
imported from whatever env LeRobot itself runs in), this Robot owns the MuJoCo
mjModel/mjData directly and steps physics itself, in the SAME process as LeRobot.
MuJoCo has no ROS/rclpy-style Python-version constraint, so there's no reason to pay
for a second process and a wire protocol here.

action_features/observation_features intentionally reproduce
lerobot.robots.so_follower.SOFollower's contract exactly (same 6 `.pos` keys, same
units: degrees for the 5 arm joints, [0, 100] for the gripper) -- the same
feature-contract pattern SO101Gazebo already established, so a dataset recorded here
uses the same feature names/semantics as one recorded on so101_gazebo or eventually the
real so101_follower (modulo the calibration-offset caveat every SO-100/101 sim carries
-- see examples/smolvla_gym_lowcostrobot_finetuned.py in the lerobot repo for the
general version of that problem; Phase 7 in PLAN.md is where this project resolves it
for real, once hardware arrives).

GRIPPER 0/100 CONVENTION: mapped linearly from the gripper joint's own MJCF range
(-10deg .. 100deg -- see assets/so101/so101.xml's `gripper` joint/`gripper_act`
actuator). 0% = that range's minimum angle = CLOSED, matching
scripts/grasp_trigger_test.py's own `_gripper_closed_qpos = env._ctrlrange["gripper"][0]`
(the one place already in this repo that commits to a closed/open direction); 100% =
the range's maximum angle = OPEN. This is a project-internal convention, not yet tied to
any real SO-101's own calibration direction.

send_action() writes straight to `data.ctrl` and steps physics forward by one
`control_dt` of sim time -- unlike SO101PenPickPlaceEnv.step(), which integrates a
normalized per-tick *delta* (its own bespoke RL action interface) into `data.ctrl`
itself. Here the incoming action is already an absolute goal position (matching
SOFollower.send_action(), which just writes `Goal_Position` straight to the motor bus)
-- so the delta-integration duty (turning smooth gamepad motion into successive
absolute joint targets) belongs to SO101MujocoTeleop instead. See that module's
docstring for the follow-on half of this split.

Physics constants below (can spawn zone, can geometry) are intentionally duplicated
from so101_mujoco_env/pen_pickplace_env.py rather than imported: this package is meant
to stay self-contained and pip-installable on its own (mirroring
lerobot_robot_so101_gazebo, which duplicates its own ARM_JOINTS/GRIPPER_JOINT tuple
rather than importing ROS-side code) instead of depending on a sibling directory that
isn't itself an installed package. If the scene's spawn zone changes, update both
places.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path

import mujoco
import numpy as np

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.robots.robot import Robot
from lerobot.utils.errors import DeviceNotConnectedError

from .config_so101_mujoco import SO101MujocoConfig

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER_JOINT = "gripper"
ALL_JOINTS = ARM_JOINTS + (GRIPPER_JOINT,)

# lerobot_robot_so101_mujoco/lerobot_robot_so101_mujoco/so101_mujoco.py -> parents[2] is
# the so101_mujoco_sim2real project root (see pyproject.toml's location one level up).
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SCENE_XML = _PROJECT_ROOT / "assets" / "scenes" / "pen_pickplace_scene.xml"

# Duplicated from so101_mujoco_env/pen_pickplace_env.py -- see module docstring.
GRIPPER_RANGE_RAD = (-0.174533, 1.74533)  # matches assets/so101/so101.xml's gripper joint/actuator range
CAN_SAMPLING_XY_LOW = np.array([0.16, -0.18])
CAN_SAMPLING_XY_HIGH = np.array([0.26, -0.02])
CAN_HALF_HEIGHT = 0.038
CAN_UPRIGHT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


class SO101Mujoco(Robot):
    config_class = SO101MujocoConfig
    name = "so101_mujoco"

    def __init__(self, config: SO101MujocoConfig):
        super().__init__(config)
        self.config = config

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._renderer: mujoco.Renderer | None = None
        self._n_substeps = 1
        self._home_key_id = -1
        self._joint_qpos_addr: dict[str, int] = {}
        self._actuator_id: dict[str, int] = {}
        self._ctrlrange: dict[str, tuple[float, float]] = {}
        self._can_qpos_addr = 0
        self._np_random = np.random.RandomState()

        # Not part of the documented Robot ABC -- lerobot_record.py reaches into
        # `robot.cameras` directly (not through get_observation()) to size its
        # image-writer thread pool (`num_image_writer_threads_per_camera *
        # len(robot.cameras)`). This robot doesn't use LeRobot's CameraConfig/Camera
        # device abstraction at all (frames come from mujoco.Renderer, not a local
        # opencv/realsense device), so this exists purely so `len(robot.cameras)`
        # resolves correctly -- same trick as SO101Gazebo's `self.cameras`.
        self.cameras: dict[str, None] = dict.fromkeys(self.config.camera_shapes)

        self._is_connected = False

    @cached_property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys((f"{j}.pos" for j in ALL_JOINTS), float)

    @cached_property
    def _cameras_ft(self) -> dict[str, tuple[int, int, int]]:
        return dict(self.config.camera_shapes)

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        # No calibration concept for a digital twin: sim joint zero IS the MJCF zero,
        # always. Same rationale as SO101Gazebo.is_calibrated. Real calibration only
        # matters once so_follower/so101_hardware is driving the actual arm (Phase 7).
        return True

    def connect(self, calibrate: bool = True) -> None:
        del calibrate  # no-op; see is_calibrated
        scene_xml = Path(self.config.scene_xml) if self.config.scene_xml else _DEFAULT_SCENE_XML
        self.model = mujoco.MjModel.from_xml_path(str(scene_xml))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = self.config.physics_dt
        self._n_substeps = max(1, round(self.config.control_dt / self.config.physics_dt))

        self._home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self._home_key_id < 0:
            raise RuntimeError(f"{scene_xml} is missing its 'home' keyframe")

        self._joint_qpos_addr = {j: self.model.joint(j).qposadr[0] for j in ALL_JOINTS}
        self._actuator_id = {j: self.model.actuator(f"{j}_act").id for j in ALL_JOINTS}
        self._ctrlrange = {j: tuple(self.model.actuator(f"{j}_act").ctrlrange) for j in ALL_JOINTS}

        can_body_id = self.model.body("can").id
        # The can's <freejoint/> has no explicit name, so it isn't indexable by name --
        # look it up via the body's own joint address instead (same trick
        # pen_pickplace_env.py uses).
        can_joint_id = self.model.body_jntadr[can_body_id]
        self._can_qpos_addr = self.model.jnt_qposadr[can_joint_id]

        self.reset_scene(randomize_can_pose=self.config.randomize_can_pose_on_connect)
        self._is_connected = True

    def calibrate(self) -> None:
        pass  # see is_calibrated

    def configure(self) -> None:
        pass  # nothing to configure; there's no motor firmware in a digital twin

    def reset_scene(
        self,
        *,
        randomize_can_pose: bool = True,
        can_xy: np.ndarray | tuple[float, float] | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Reset the sim to the scene's 'home' keyframe, then place the can according to
        (in priority order): an explicit `can_xy` override, a fresh random draw if
        `randomize_can_pose`, or -- if neither applies -- wherever `mj_resetDataKeyframe`
        itself put it (the scene's fixed keyframe default, NOT any previously-randomized
        pose; earlier code here called this with randomize_can_pose=False expecting it to
        implicitly preserve "whatever pose was active before reset", which is wrong -- the
        keyframe reset always overwrites the can's qpos first, so the previous pose is
        already gone by the time this function could "leave it alone"). NOT part of the
        Robot ABC: there is no hardware equivalent to call here -- on the real robot,
        "resetting the environment" between episodes means a human physically
        repositioning the can by hand during lerobot-record's own `--dataset.reset_time_s`
        window (see record_loop's docstring in the lerobot repo). A recording driver for
        this MuJoCo twin should call this explicitly during that same window instead (e.g.
        from a `--robot.type=so101_mujoco`-aware wrapper around lerobot-record), since
        nothing will do it automatically.

        Returns the can's resulting (x, y), so a caller that wants to retry an episode
        under the identical setup can capture it right after the reset that started that
        episode, then pass it back in as `can_xy` on the retry -- see
        scripts/record_dataset.py's retry path for exactly that pattern.

        Mirrors SO101PenPickPlaceEnv.reset()'s own can-pose sampling exactly (same
        bounds, same axially-symmetric position-only randomization -- see that module's
        docstring for why no orientation randomization is needed for a can).
        """
        if seed is not None:
            self._np_random = np.random.RandomState(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)

        if can_xy is not None:
            xy = np.asarray(can_xy, dtype=float)
        elif randomize_can_pose:
            xy = self._np_random.uniform(CAN_SAMPLING_XY_LOW, CAN_SAMPLING_XY_HIGH)
        else:
            xy = None

        if xy is not None:
            addr = self._can_qpos_addr
            self.data.qpos[addr : addr + 3] = (*xy, CAN_HALF_HEIGHT)
            self.data.qpos[addr + 3 : addr + 7] = CAN_UPRIGHT_QUAT

        mujoco.mj_forward(self.model, self.data)
        return self.data.qpos[self._can_qpos_addr : self._can_qpos_addr + 2].copy()

    @property
    def can_xy(self) -> np.ndarray:
        """The can's current (x, y). Lets a caller capture the pose right after
        connect()'s own initial reset_scene() call too, not just after one it made
        itself -- see reset_scene()'s docstring for the retry pattern this supports."""
        return self.data.qpos[self._can_qpos_addr : self._can_qpos_addr + 2].copy()

    @staticmethod
    def _gripper_rad_to_pct(angle_rad: float) -> float:
        lo, hi = GRIPPER_RANGE_RAD
        return float(np.clip((angle_rad - lo) / (hi - lo) * 100.0, 0.0, 100.0))

    @staticmethod
    def _gripper_pct_to_rad(pct: float) -> float:
        lo, hi = GRIPPER_RANGE_RAD
        return float(lo + np.clip(pct, 0.0, 100.0) / 100.0 * (hi - lo))

    def _render_camera(self, name: str) -> np.ndarray:
        if self._renderer is None:
            # All configured cameras share one renderer resolution -- same constraint
            # joint_teleop.py's CameraFeedWindow already imposes via its single
            # --cam-size flag. Per-camera resolutions aren't supported.
            h, w, _ = next(iter(self.config.camera_shapes.values()))
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        self._renderer.update_scene(self.data, camera=name)
        return self._renderer.render()

    def get_observation(self) -> RobotObservation:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        obs: dict = {}
        for joint in ARM_JOINTS:
            obs[f"{joint}.pos"] = float(np.rad2deg(self.data.qpos[self._joint_qpos_addr[joint]]))
        obs[f"{GRIPPER_JOINT}.pos"] = self._gripper_rad_to_pct(
            self.data.qpos[self._joint_qpos_addr[GRIPPER_JOINT]]
        )
        for cam_name in self.config.camera_shapes:
            obs[cam_name] = self._render_camera(cam_name)
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        sent: dict[str, float] = {}
        for joint in ARM_JOINTS:
            key = f"{joint}.pos"
            if key not in action:
                continue
            lo, hi = self._ctrlrange[joint]
            target_rad = float(np.clip(np.deg2rad(float(action[key])), lo, hi))
            self.data.ctrl[self._actuator_id[joint]] = target_rad
            sent[key] = float(np.rad2deg(target_rad))

        gripper_key = f"{GRIPPER_JOINT}.pos"
        if gripper_key in action:
            lo, hi = self._ctrlrange[GRIPPER_JOINT]
            target_rad = float(np.clip(self._gripper_pct_to_rad(float(action[gripper_key])), lo, hi))
            self.data.ctrl[self._actuator_id[GRIPPER_JOINT]] = target_rad
            sent[gripper_key] = self._gripper_rad_to_pct(target_rad)

        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        return sent

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        self.model = None
        self.data = None
        self._is_connected = False
