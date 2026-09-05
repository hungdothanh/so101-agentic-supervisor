#!/usr/bin/env python3
"""Phase 2 verification: scripted grasp test for the can+bin fallback task.

Runs a scripted (non-learned) heuristic grasp controller through
SO101PenPickPlaceEnv's real action interface over N randomized resets, using
privileged ground-truth object pose -- same "cheat", for the same reason
(bootstrapping a difficulty estimate without hand-teleoperating), as
lerobot/examples/smolvla_robosuite_record_scripted.py.

This originally targeted a pen + pen-box and doubled as PLAN.md's pen-vs-can
decision trigger: that version scored 5% (1/20), well under the plan's 80%
threshold, because SO-101's gripper is a single hook-style finger closing
against a fixed palm right at gripper_frame_link -- for a pen lying flush on
the table, the palm's own bulk reaches the pen's position before the moving
jaw can close on it. Switched to the can+bin fallback per that decision --
see assets/scenes/pen_pickplace_scene.xml's header comment for the full
pivot rationale.

KNOWN LIMITATION, not yet resolved: the can version still scores close to
0%, for a related but distinct reason -- this script's position controller
is a naive per-tick damped-least-squares IK with no path planning or
collision avoidance, so the same fixed-palm bulk that plagued the pen also
clips the can (a big, 6cm-diameter target) while transiting to/past it, well
before the gripper starts closing. Mitigated some of this (an explicit
"rise to a safe height before moving horizontally" phase, frozen XY during
the vertical descent) but did not fully solve it -- some spawn positions
still produce a rough contact that destabilizes the sim. This is judged a
limitation of this quick scripted heuristic, not of the underlying env
(SO101PenPickPlaceEnv's reset/step/reward/success are smoke-tested
independently and are sound) or of the can+bin task itself -- a human
teleoperator (Phase 3) or a trained policy isn't bound by this controller's
blind, unplanned point-to-point IK. Left as-is rather than sunk-cost further
engineering a script whose only job was informing a decision that's already
made; revisit only if a real success-rate number is needed later.

Why this needs its own small IK, even though the env's real-time action
interface is pure joint-space (see PLAN.md -- no IK on the online control
path): turning "move the gripper toward this XYZ" into joint deltas needs
*some* kinematic reasoning, and hand-picking joint angles for a moving
target isn't practical. This mirrors the justification PLAN.md already gives
for Phase 10's (optional, offline-only) MimicGen differential IK -- this
script is that same kind of offline-only utility, used here to generate a
heuristic policy for a difficulty estimate, not to change the env's action
space.

Run inside the `lerobot_smolvla` conda env:
    python scripts/grasp_trigger_test.py --num-episodes=20
"""

from __future__ import annotations

import argparse

import numpy as np

from so101_mujoco_env.ik_utils import FingertipIK
from so101_mujoco_env.pen_pickplace_env import ARM_JOINTS, BIN_CENTER_XY, SO101PenPickPlaceEnv

GRIPPER_OPEN = 1.0
GRIPPER_CLOSE = -1.0

HOVER_HEIGHT = 0.14  # above the can's own (mid-height) z during approach -- clears the can's
# top (z=0.12) by 6cm, not the 2cm an earlier 0.08 gave (that clearance let the gripper's
# fixed-palm bulk clip the can mid-transit -- see scratch/debug_can_grasp.py from development)
LIFT_Z = 0.15
RELEASE_HEIGHT_OFFSET = 0.09  # above the bin's 0.06-tall walls, so the can drops in clear

DIST_THRESHOLD = 0.01
GRASP_HOLD_STEPS = 25  # gripper closes until it contacts the can and stalls; see MAX_GRIPPER_SPEED_RAD_S
RELEASE_HOLD_STEPS = 3
SETTLE_STEPS = 8
MAX_STEPS_PER_PHASE = 60

# POSITION_JOINTS, FINGERTIP_LOCAL, POS_GAIN moved to so101_mujoco_env/ik_utils.py
PHASES = [
    "rise",
    "approach",
    "descend",
    "grasp",
    "lift",
    "transport",
    "lower",
    "release",
    "settle",
    "retreat",
    "done",
]


class ScriptedGraspController:
    """Privileged-pose heuristic pick-and-place, driving env.step()'s real
    (joint-delta) action interface via the shared FingertipIK utility (see
    so101_mujoco_env/ik_utils.py -- extracted from here, behavior unchanged)."""

    def __init__(self, env: SO101PenPickPlaceEnv):
        self.env = env
        self._ik = FingertipIK(env)

    def action(self, phase: str, can_pos: np.ndarray, bin_center_xy: np.ndarray) -> tuple[np.ndarray, float]:
        gripper_frame_pos = self.env.data.xpos[self.env.gripper_frame_body_id]

        if phase == "rise":
            # Go straight up from wherever the arm currently is (self-canceling
            # XY, same trick as "lift" below) BEFORE moving horizontally toward
            # the can -- HOME_QPOS sits close to table height (tuned for the
            # original pen task), so moving toward the can's XY and hover
            # height at the same time let the gripper's own bulk clip it
            # mid-transit (verified empirically). Climbing first keeps the
            # horizontal leg of the trip well above table-level obstacles.
            target = np.array([gripper_frame_pos[0], gripper_frame_pos[1], LIFT_Z])
            gripper = GRIPPER_OPEN
        elif phase == "approach":
            target = np.array([can_pos[0], can_pos[1], can_pos[2] + HOVER_HEIGHT])
            gripper = GRIPPER_OPEN
        elif phase in ("descend", "grasp"):
            # Freeze XY at wherever "approach" already converged (self-canceling,
            # same trick as "rise"/"lift") rather than re-targeting can_pos's XY
            # fresh here too: even the ~1cm residual approach error, blended with
            # the new Z motion, was enough to swing the gripper's palm sideways
            # into the can while descending (verified empirically). Pure-Z motion
            # doesn't have that failure mode.
            xy = self._ik.fingertip_at_closed()[:2]
            target = np.array([xy[0], xy[1], can_pos[2]])
            gripper = GRIPPER_CLOSE if phase == "grasp" else GRIPPER_OPEN
        elif phase == "lift":
            target = np.array([gripper_frame_pos[0], gripper_frame_pos[1], LIFT_Z])
            gripper = GRIPPER_CLOSE
        elif phase == "transport":
            target = np.array([bin_center_xy[0], bin_center_xy[1], LIFT_Z])
            gripper = GRIPPER_CLOSE
        elif phase in ("lower", "release"):
            target = np.array([bin_center_xy[0], bin_center_xy[1], RELEASE_HEIGHT_OFFSET])
            gripper = GRIPPER_CLOSE if phase == "lower" else GRIPPER_OPEN
        else:  # settle, retreat
            z = LIFT_Z if phase == "retreat" else gripper_frame_pos[2]
            target = np.array([gripper_frame_pos[0], gripper_frame_pos[1], z])
            gripper = GRIPPER_OPEN

        pos_actions, pos_err = self._ik.position_action(target)

        action = np.zeros(len(ARM_JOINTS), dtype=np.float32)
        for i, joint in enumerate(ARM_JOINTS):
            if joint in pos_actions:
                action[i] = pos_actions[joint]
            elif joint == "gripper":
                action[i] = gripper
        return action, pos_err


def run_episode(env: SO101PenPickPlaceEnv, controller: ScriptedGraspController, max_steps: int, seed: int) -> bool:
    env.reset(seed=seed)
    phase_idx, phase_step = 0, 0
    for _ in range(max_steps):
        phase = PHASES[phase_idx]
        if phase == "done":
            break
        can_pos = env.data.xpos[env._can_body_id].copy()
        action, pos_err = controller.action(phase, can_pos, BIN_CENTER_XY)
        _obs, _reward, _term, _trunc, info = env.step(action)
        if info["succeed"]:
            return True

        phase_step += 1
        advance = False
        if phase in ("rise", "approach", "descend", "lift", "transport", "lower"):
            advance = pos_err < DIST_THRESHOLD
        elif phase == "grasp":
            advance = phase_step >= GRASP_HOLD_STEPS
        elif phase == "release":
            advance = phase_step >= RELEASE_HOLD_STEPS
        elif phase == "settle":
            advance = phase_step >= SETTLE_STEPS
        elif phase == "retreat":
            advance = pos_err < DIST_THRESHOLD
        if advance or phase_step >= MAX_STEPS_PER_PHASE:
            phase_idx += 1
            phase_step = 0

    return bool(env._is_success())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = SO101PenPickPlaceEnv(control_dt=0.05, reward_type="sparse", randomize_can_pose=True, seed=args.seed)
    controller = ScriptedGraspController(env)

    n_success = 0
    for ep in range(args.num_episodes):
        success = run_episode(env, controller, args.max_steps, seed=args.seed * 1000 + ep)
        n_success += int(success)
        print(f"episode {ep + 1}/{args.num_episodes}: {'SUCCESS' if success else 'fail'}")

    rate = n_success / args.num_episodes
    print(f"\n{n_success}/{args.num_episodes} = {rate:.0%} success")


if __name__ == "__main__":
    main()
