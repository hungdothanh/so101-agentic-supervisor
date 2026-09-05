"""Shared fingertip-space differential IK utility for SO101PenPickPlaceEnv.

Extracted from scripts/grasp_trigger_test.py's ScriptedGraspController, which already
validated this exact damped-least-squares Jacobian approach as mechanically correct --
that script's near-0% success rate comes from the hand-picked task waypoints fed into
it (see its own module docstring's "KNOWN LIMITATION" section), not from this control
method. Reused as-is by:
  - grasp_trigger_test.py (unchanged behavior, now imports from here instead of
    defining this inline)
  - scripts/mimicgen_augment.py, which drives the same IK toward waypoints extracted
    from a real human trajectory instead of hand-picked ones, and also uses
    fingertip_at() standalone (not through FingertipIK) to reconstruct a recorded
    episode's Cartesian trajectory from its logged joint states, for segmentation.
"""

from __future__ import annotations

import mujoco
import numpy as np

from so101_mujoco_env.pen_pickplace_env import SO101PenPickPlaceEnv

# moving_jaw_so101_v1_link body-local coords of the mesh vertex closest to
# gripper_frame_link at full closure. gripper_frame_link is embedded in the fixed
# palm's own mesh, not a usable grasp-point target: this is a hook-style single
# finger closing against that palm, and its fingertip only arrives near
# gripper_frame_link at the gripper's fully-closed limit -- at any other opening it
# can be many cm away. Targeting gripper_frame_link directly closes on empty air
# every time for a thin object (verified empirically on the pen); targeting this
# fingertip, evaluated at the closed limit, is the actual grasp point.
FINGERTIP_LOCAL = np.array([-0.01229868, -0.07568328, 0.02290582])

POSITION_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex")
POS_GAIN = 1.0


def fingertip_at(env: SO101PenPickPlaceEnv, gripper_qpos: float) -> np.ndarray:
    """World position of the grasp-point vertex if the gripper were at
    `gripper_qpos`, holding every other joint at its current value. Standalone (not
    part of FingertipIK) since mimicgen_augment.py also uses this on its own to
    reconstruct a recorded episode's Cartesian trajectory from logged joint states,
    with no IK step involved at all."""
    model, data = env.model, env.data
    gripper_qpos_addr = env._joint_qpos_addr["gripper"]
    saved = data.qpos[gripper_qpos_addr]
    data.qpos[gripper_qpos_addr] = gripper_qpos
    mujoco.mj_forward(model, data)
    mat = data.xmat[env.moving_jaw_body_id].reshape(3, 3)
    pos = data.xpos[env.moving_jaw_body_id] + mat @ FINGERTIP_LOCAL
    data.qpos[gripper_qpos_addr] = saved
    mujoco.mj_forward(model, data)
    return pos.copy()


class FingertipIK:
    """Per-tick damped-least-squares Jacobian IK: converts a Cartesian fingertip
    target into a normalized joint-space delta action for SO101PenPickPlaceEnv's 4
    position joints (shoulder_pan/lift, elbow_flex, wrist_flex) -- wrist_roll and
    gripper are driven separately by the caller, same convention
    ScriptedGraspController already used."""

    def __init__(self, env: SO101PenPickPlaceEnv):
        self.env = env
        self._position_dof_addr = [env._joint_dof_addr[j] for j in POSITION_JOINTS]
        self._gripper_closed_qpos = env._ctrlrange["gripper"][0]
        self._jacp = np.zeros((3, env.model.nv))

    def fingertip_at_closed(self) -> np.ndarray:
        """Fingertip position evaluated at the gripper's closed limit, holding other
        joints at their CURRENT value -- the actual grasp point, per
        FINGERTIP_LOCAL's own docstring."""
        return fingertip_at(self.env, self._gripper_closed_qpos)

    def position_action(self, target: np.ndarray) -> tuple[dict[str, float], float]:
        """Returns ({joint_name: normalized_delta}, |error|) driving the fingertip
        (evaluated at the gripper's closed limit) toward `target`."""
        model, data = self.env.model, self.env.data
        cur = self.fingertip_at_closed()
        err = target - cur
        mujoco.mj_jac(model, data, self._jacp, None, cur, self.env.moving_jaw_body_id)
        jac = self._jacp[:, self._position_dof_addr]
        lam = 1e-3
        dq = jac.T @ np.linalg.solve(jac @ jac.T + lam * np.eye(3), err)
        dq *= POS_GAIN
        actions = {}
        for i, joint in enumerate(POSITION_JOINTS):
            actions[joint] = float(np.clip(dq[i] / self.env._max_delta[joint], -1.0, 1.0))
        return actions, float(np.linalg.norm(err))
