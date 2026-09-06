"""Phase 2: custom gym.Env for the SO-101 can -> bin pick-and-place task.

Joint-space position control throughout (matches the real Feetech
position-mode servos -- see PLAN.md's "no IK in the online control path"
decision): action[i] is a normalized per-joint velocity command in [-1, 1]
for (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll,
gripper); step() integrates it into data.ctrl (each actuator's own position
target) at a fixed per-joint max speed, clamped to that actuator's
ctrlrange. This is the one action interface shared by the scripted
grasp-trigger test (scripts/grasp_trigger_test.py), the gamepad teleop
(joint_teleop.py), and eventually policy rollouts -- so "how a normalized
action becomes joint motion" only needs to be right in one place.

Task is can-into-bin, not the originally planned pen-into-pen-box: Phase 2's
scripted grasp-trigger test scored 5% (1/20) on the pen, well under
PLAN.md's 80% keep-the-pen threshold (a pen lying flush on the table sits
right where SO-101's gripper has a fixed palm structure, so a top-down
approach shoves it out of position before the moving jaw can close -- see
assets/scenes/pen_pickplace_scene.xml's header comment). Switched to
PLAN.md's own can+bin fallback, reusing desk_cleanup_env.py's two-geom can
pattern.

Object-pose randomization here is task-necessary (the can must start
somewhere on the table for "pick it up" to mean anything), not Strategy-1
domain randomization (visual/dynamics DR is a later, separate phase -- see
PLAN.md Phase 6). The can is axially symmetric, so unlike the pen this needs
no orientation randomization -- position alone (matching
desk_cleanup_env.py's own can, which is likewise position-only).
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE_XML = Path(__file__).resolve().parent.parent / "assets" / "scenes" / "pen_pickplace_scene.xml"

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# Max commanded joint speed at |action|=1.0. Arm figure matches the earlier
# ROS/Gazebo teleop's max_joint_speed_deg_s=30 (so101_sim2real/so101_teleop),
# doubled here since this is a faster, more direct control loop with no
# network hop; the gripper gets its own (it's a much lower-inertia joint --
# see fetch_so101_urdf.py's per-joint forcerange tuning).
MAX_JOINT_SPEED_RAD_S = np.deg2rad(60.0)
MAX_GRIPPER_SPEED_RAD_S = np.deg2rad(120.0)

# Where the can may spawn: near the arm's reachable zone, clear of the bin's
# footprint (bin center (0.30, 0.15), see BIN_INTERIOR_HALF below). Distances
# are measured from the arm's own mount point (0, 0.15, 0) -- see
# ARM_MOUNT_POS in scripts/fetch_so101_urdf.py -- not world origin: the
# farthest corner here sits ~0.42m from that mount point, under the measured
# 0.44m reach. A wider box that reused the pre-mount-move bounds put one
# corner at 0.476m, outside reach (caught by checking, not by assuming the
# old bounds still applied after moving the arm).
CAN_SAMPLING_XY_LOW = np.array([0.16, -0.18])
CAN_SAMPLING_XY_HIGH = np.array([0.26, -0.02])
CAN_HALF_HEIGHT = 0.038  # small cola can (~38mm diameter x ~76mm tall), not desk_cleanup's 330ml-can size
CAN_RADIUS = 0.019  # assets/scenes/pen_pickplace_scene.xml can_collision geom: size="0.019 0.038"
                     # (radius, half-length) -- the resting z when lying on its side, NOT CAN_HALF_HEIGHT
CAN_UPRIGHT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


def _sample_fallen_quat(rng: np.random.RandomState, heading: float | None = None) -> np.ndarray:
    """A quat for the can lying on its side: tip 90deg about the local X axis (long
    axis goes from vertical to horizontal), then compose a world-Z yaw so the fallen
    heading varies -- matches this project's existing quaternion-construction
    convention (mujoco.mju_euler2Quat/mju_mat2Quat, see scripts/fetch_so101_urdf.py)
    rather than pulling in a new quaternion-math dependency.

    `heading=None` (default): yaw is drawn randomly, as before. `heading=<float>`:
    use this exact yaw instead -- needed by scripts/mimicgen_augment.py to spawn a
    candidate/replay at a specific target heading rather than a random one. Note this
    `heading` parameter is the internal yaw used to CONSTRUCT the quat, not the same
    number `heading_from_quat()` below extracts from the result (that extracted value
    is offset from this one by a fixed 90deg, a consequence of the tilt-then-yaw
    construction order -- verified directly, not a bug: what matters is
    `heading_from_quat(_sample_fallen_quat(rng, h))` is self-consistent for any given
    `h`, which it is)."""
    tilt_quat = np.zeros(4)
    mujoco.mju_euler2Quat(tilt_quat, np.array([np.pi / 2, 0.0, 0.0]), "XYZ")
    yaw = rng.uniform(-np.pi, np.pi) if heading is None else heading
    yaw_quat = np.zeros(4)
    mujoco.mju_euler2Quat(yaw_quat, np.array([0.0, 0.0, yaw]), "XYZ")
    fallen_quat = np.zeros(4)
    mujoco.mju_mulQuat(fallen_quat, yaw_quat, tilt_quat)
    return fallen_quat


def heading_from_quat(quat: np.ndarray) -> float:
    """Extracts the can's heading (direction of its long axis, projected onto the
    table) from an arbitrary quaternion -- not just ones built by
    _sample_fallen_quat() above, this also works on whatever quat real physics
    settles a knocked-over can into. A cylinder lying on its side is symmetric under
    a 180deg rotation about the vertical axis through its center (the long axis is a
    line, not a direction), so heading is inherently defined mod pi, not mod 2*pi --
    wrapping it to [0, pi) makes that symmetry explicit rather than silently treating
    two physically-identical poses as maximally different. Verified self-consistent
    against _sample_fallen_quat()'s own construction (fixed, predictable offset)."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, quat)
    long_axis_world = mat.reshape(3, 3)[:, 2]  # can's local Z (long axis) in world frame
    return float(np.arctan2(long_axis_world[1], long_axis_world[0]) % np.pi)


def wrap_heading_delta(delta: float) -> float:
    """Wraps a heading difference into [-pi/2, pi/2] -- the smallest rotation that
    reconciles two headings defined mod pi (see heading_from_quat()). Using the raw
    difference instead would treat a pi-flipped (but physically identical) heading as
    maximally far apart, which is wrong."""
    return float(((delta + np.pi / 2) % np.pi) - np.pi / 2)


# roll_impulse calibration (see PLAN.md and the plan doc this was designed against):
# measured empirically against CAN_ROLL_FRICTION below -- lin_speed 0.8-1.2 m/s spans
# ~4-15cm of travel once the can is already lying down, matching the target rolled
# range. Direction is drawn uniformly; spin is incidental (varies the settle heading)
# rather than the primary driver of travel distance.
CAN_ROLL_LINVEL_RANGE = (0.8, 1.2)  # m/s
CAN_ROLL_SPIN_RANGE = (4.0, 8.0)  # rad/s, random sign
CAN_ROLL_SETTLE_TIME_S = 6.0  # matches PLAN.md's own measured settle time for this scenario

BIN_CENTER_XY = np.array([0.30, 0.15])
# Bin: half-size (0.06, 0.06), wall thickness 0.005 -- see
# assets/scenes/pen_pickplace_scene.xml's bin body.
BIN_INTERIOR_HALF = np.array([0.055, 0.055])
BIN_FLOOR_Z = 0.01
# Bug fix: this used to be checked as `can_pos[2] < BIN_FLOOR_Z + CAN_SETTLED_Z_MARGIN`
# with no CAN_HALF_HEIGHT term at all -- inherited verbatim from desk_cleanup_env.py's
# own object/margin, which happened to be short enough that its margin alone covered
# resting height. This project's can (CAN_HALF_HEIGHT=0.038) is taller: an upright can
# resting dead-center on the bin floor settles at can_pos[2] ~= BIN_FLOOR_Z +
# CAN_HALF_HEIGHT ~= 0.048 (verified directly: manually placed upright at bin center,
# let physics settle, measured z=0.048) -- comfortably ABOVE the old 0.025 threshold,
# so a can placed perfectly upright always failed the "settled" check. Confirmed via a
# real eval run: two episodes visibly ended with the can placed correctly, both scored
# as failures. Fixed below by adding CAN_HALF_HEIGHT into the threshold explicitly, so
# CAN_SETTLED_Z_MARGIN is now purely the tolerance on top of the can's real resting
# height, not silently standing in for it.
CAN_SETTLED_Z_MARGIN = 0.015

OFF_TABLE_Z = -0.05  # can fell off the table edge -> not recoverable this episode


class SO101PenPickPlaceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        reward_type: str = "sparse",
        randomize_can_pose: bool = True,
        image_obs: bool = False,
        render_size: tuple[int, int] = (128, 128),
        seed: int = 0,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = physics_dt
        self._n_substeps = max(1, round(control_dt / physics_dt))
        self.control_dt = control_dt
        self.reward_type = reward_type
        self.randomize_can_pose = randomize_can_pose
        self.image_obs = image_obs
        self._render_size = render_size
        self._renderer: mujoco.Renderer | None = None
        self._np_random = np.random.RandomState(seed)

        self._home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self._home_key_id < 0:
            raise RuntimeError(f"{SCENE_XML} is missing its 'home' keyframe")

        self._joint_qpos_addr = {j: self.model.joint(j).qposadr[0] for j in ARM_JOINTS}
        self._joint_dof_addr = {j: self.model.joint(j).dofadr[0] for j in ARM_JOINTS}
        self._actuator_id = {j: self.model.actuator(f"{j}_act").id for j in ARM_JOINTS}
        self._ctrlrange = {j: tuple(self.model.actuator(f"{j}_act").ctrlrange) for j in ARM_JOINTS}
        self._max_delta = {
            j: (MAX_GRIPPER_SPEED_RAD_S if j == "gripper" else MAX_JOINT_SPEED_RAD_S) * control_dt
            for j in ARM_JOINTS
        }

        self._can_body_id = self.model.body("can").id
        # The can's <freejoint/> has no explicit name, so it isn't indexable by
        # name -- look it up via the body's own joint address instead.
        can_joint_id = self.model.body_jntadr[self._can_body_id]
        self._can_qpos_addr = self.model.jnt_qposadr[can_joint_id]  # freejoint: 7 values (pos+quat)
        self._can_dof_addr = self.model.jnt_dofadr[can_joint_id]  # freejoint: 6 values (linvel+angvel)
        self.gripper_frame_body_id = self.model.body("gripper_frame_link").id
        self.moving_jaw_body_id = self.model.body("moving_jaw_so101_v1_link").id

        self.action_space = spaces.Box(-1.0, 1.0, shape=(len(ARM_JOINTS),), dtype=np.float32)
        obs_spaces = {
            "agent_pos": spaces.Box(-np.inf, np.inf, shape=(2 * len(ARM_JOINTS),), dtype=np.float32)
        }
        if image_obs:
            h, w = render_size
            obs_spaces["pixels"] = spaces.Dict(
                {
                    "front": spaces.Box(0, 255, (h, w, 3), dtype=np.uint8),
                    "wrist": spaces.Box(0, 255, (h, w, 3), dtype=np.uint8),
                }
            )
        else:
            obs_spaces["environment_state"] = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Dict(obs_spaces)

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """`options={"can_xy": (x, y)}` places the can at an exact position instead of
        drawing a random one -- used by scripts/mimicgen_augment.py to replay/verify a
        candidate demonstration at a specific target position. Falls back to the normal
        random-draw behavior (or no repositioning at all, if randomize_can_pose=False)
        when omitted, so this is fully backward compatible with every existing caller.

        `options={"fallen": True}` (composable with `can_xy`, or with the random-draw
        path) spawns the can lying on its side instead of standing -- for triggering/
        testing the supervisor's failure-recovery path on demand (see PLAN.md). Changes
        only orientation/resting z; position selection is unaffected. Optionally combine
        with `options={"can_heading": <radians>}` to pin the fallen heading instead of
        drawing it randomly (see _sample_fallen_quat()) -- used by
        scripts/mimicgen_augment.py to spawn a candidate/replay at an exact target
        heading, same idea as `can_xy`'s exact-position override.

        `options={"fallen": True, "roll_impulse": True}`: after placing the can fallen,
        apply a randomized knock (CAN_ROLL_LINVEL_RANGE/CAN_ROLL_SPIN_RANGE) and step
        real physics for CAN_ROLL_SETTLE_TIME_S before returning, so the final resting
        pose is a physically-settled "knocked and rolled" one (PLAN.md's own
        recommendation for recording realistic recovery reference demos), not an
        idealized in-place tip. Requires `fallen=True`; ignored otherwise."""
        super().reset(seed=seed)
        if seed is not None:
            self._np_random = np.random.RandomState(seed)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_key_id)

        fallen = bool(options.get("fallen", False)) if options else False
        heading_override = options.get("can_heading") if options else None
        roll_impulse = fallen and bool(options.get("roll_impulse", False)) if options else False
        z = CAN_RADIUS if fallen else CAN_HALF_HEIGHT
        quat = _sample_fallen_quat(self._np_random, heading=heading_override) if fallen else CAN_UPRIGHT_QUAT

        if options and "can_xy" in options:
            xy = np.asarray(options["can_xy"], dtype=float)
            addr = self._can_qpos_addr
            self.data.qpos[addr : addr + 3] = (*xy, z)
            self.data.qpos[addr + 3 : addr + 7] = quat
        elif self.randomize_can_pose:
            xy = self._np_random.uniform(CAN_SAMPLING_XY_LOW, CAN_SAMPLING_XY_HIGH)
            addr = self._can_qpos_addr
            self.data.qpos[addr : addr + 3] = (*xy, z)
            self.data.qpos[addr + 3 : addr + 7] = quat

        mujoco.mj_forward(self.model, self.data)

        if roll_impulse:
            self._apply_roll_impulse_and_settle()

        return self._get_obs(), {}

    def _apply_roll_impulse_and_settle(self) -> None:
        """Gives the (already-fallen) can a randomized shove, then steps real physics
        forward CAN_ROLL_SETTLE_TIME_S of sim time so it settles into a physically
        plausible final pose before this reset() returns. Direction is uniform over
        the full circle -- calibrated empirically (see CAN_ROLL_LINVEL_RANGE's own
        comment) using this same uniform-random-direction setup, so the ~4-15cm target
        travel range only holds with this same distribution."""
        direction = self._np_random.uniform(-np.pi, np.pi)
        lin_speed = self._np_random.uniform(*CAN_ROLL_LINVEL_RANGE)
        spin = self._np_random.uniform(*CAN_ROLL_SPIN_RANGE) * self._np_random.choice([-1.0, 1.0])
        addr = self._can_dof_addr
        self.data.qvel[addr : addr + 3] = [lin_speed * np.cos(direction), lin_speed * np.sin(direction), 0.0]
        self.data.qvel[addr + 3 : addr + 6] = [0.0, 0.0, spin]
        n_settle_steps = int(round(CAN_ROLL_SETTLE_TIME_S / self.model.opt.timestep))
        for _ in range(n_settle_steps):
            mujoco.mj_step(self.model, self.data)

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        for i, joint in enumerate(ARM_JOINTS):
            act_id = self._actuator_id[joint]
            lo, hi = self._ctrlrange[joint]
            new_ctrl = self.data.ctrl[act_id] + action[i] * self._max_delta[joint]
            self.data.ctrl[act_id] = np.clip(new_ctrl, lo, hi)

        for _ in range(self._n_substeps):
            mujoco.mj_step(self.model, self.data)

        success = self._is_success()
        reward = self._compute_reward(success)
        truncated = bool(self.data.xpos[self._can_body_id][2] < OFF_TABLE_Z)
        return self._get_obs(), reward, False, truncated, {"succeed": success}

    def _can_placement_metrics(self):
        can_pos = self.data.xpos[self._can_body_id]
        inside_xy = bool(np.all(np.abs(can_pos[:2] - BIN_CENTER_XY) < BIN_INTERIOR_HALF))
        settled = bool(can_pos[2] < (BIN_FLOOR_Z + CAN_HALF_HEIGHT + CAN_SETTLED_Z_MARGIN))
        return inside_xy, settled, can_pos

    def _is_success(self) -> bool:
        inside_xy, settled, _ = self._can_placement_metrics()
        return inside_xy and settled

    def _compute_reward(self, success: bool) -> float:
        if self.reward_type == "sparse":
            return float(success)
        _inside_xy, _settled, can_pos = self._can_placement_metrics()
        dist_xy = float(np.linalg.norm(can_pos[:2] - BIN_CENTER_XY))
        r_close = float(np.exp(-10.0 * dist_xy))
        return 0.3 * r_close + 0.7 * float(success)

    def _get_obs(self) -> dict:
        qpos = np.array([self.data.qpos[self._joint_qpos_addr[j]] for j in ARM_JOINTS], dtype=np.float32)
        qvel = np.array([self.data.qvel[self._joint_dof_addr[j]] for j in ARM_JOINTS], dtype=np.float32)
        obs = {"agent_pos": np.concatenate([qpos, qvel])}
        if self.image_obs:
            obs["pixels"] = {"front": self._render_camera("front"), "wrist": self._render_camera("wrist")}
        else:
            obs["environment_state"] = self.data.xpos[self._can_body_id].astype(np.float32)
        return obs

    def _render_camera(self, name: str) -> np.ndarray:
        if self._renderer is None:
            h, w = self._render_size
            self._renderer = mujoco.Renderer(self.model, height=h, width=w)
        self._renderer.update_scene(self.data, camera=name)
        return self._renderer.render()

    def render(self):
        return self._render_camera("front")

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
