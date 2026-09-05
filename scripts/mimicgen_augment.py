#!/usr/bin/env python3
"""MimicGen-style dataset augmentation for the can+bin task -- see
/home/rik/.claude/plans/immutable-munching-llama.md for the full design.

Built incrementally; current capability: load the reference dataset
(hungdo2401/so101_mimicgen_refs) + its logged starting poses (mimicgen_ref_poses.json,
produced by scripts/record_dataset.py --can-xy-log-path), replay each reference
episode's recorded actions through SO101PenPickPlaceEnv's real physics, and confirm
each one is a genuine, verified success (not just visually-judged by the operator at
record time) using the same _is_success() check eval_policy.py uses. This both
verifies the 9 reference demos before anything is built on top of them, AND
reconstructs each one's Cartesian fingertip trajectory (needed for segmentation, the
next step) as a side effect of the same replay pass.

UNIT CONVERSION: reused directly from scripts/eval_policy.py's own
build_observation()/action_to_env_delta() pattern -- the recorded dataset's actions
are absolute degrees/pct (SOFollower-style contract), SO101PenPickPlaceEnv.step()
wants a normalized per-step delta. See that script's module docstring for the full
rationale; duplicated here rather than imported, matching this project's existing
precedent (eval_policy.py itself duplicates these from so101_mujoco.py) of keeping
each script self-contained rather than cross-importing small conversion utilities.

Run inside the `lerobot` conda env, from the repo root:
    python scripts/mimicgen_augment.py verify-refs
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from so101_mujoco_env.ik_utils import FingertipIK, fingertip_at  # noqa: E402
from so101_mujoco_env.pen_pickplace_env import (  # noqa: E402
    ARM_JOINTS,
    CAN_SAMPLING_XY_HIGH,
    CAN_SAMPLING_XY_LOW,
    SO101PenPickPlaceEnv,
)

AUG_REPO_ID = "hungdo2401/so101_mimicgen_aug"
COMBINED_REPO_ID = "hungdo2401/so101_baseline_plus_mimicgen"
BASELINE_REPO_ID = "hungdo2401/so101_baseline"

SINGLE_TASK = "pick up the can and place it in the bin"  # matches so101_baseline's own tasks.parquet exactly
RENDER_SIZE = (480, 480)  # matches so101_baseline's recorded resolution (SO101MujocoConfig.camera_shapes)

# Dataset schema -- must match hungdo2401/so101_baseline's meta/info.json exactly (verified directly)
# so lerobot.datasets.aggregate.aggregate_datasets can combine the two later (Step 7).
AUG_DATASET_FEATURES = {
    "action": {"dtype": "float32", "shape": (len(ARM_JOINTS),), "names": [f"{j}.pos" for j in ARM_JOINTS]},
    "observation.state": {"dtype": "float32", "shape": (len(ARM_JOINTS),), "names": [f"{j}.pos" for j in ARM_JOINTS]},
    "observation.images.front": {"dtype": "video", "shape": (*RENDER_SIZE, 3), "names": ["height", "width", "channels"]},
    "observation.images.wrist": {"dtype": "video", "shape": (*RENDER_SIZE, 3), "names": ["height", "width", "channels"]},
}

GRIPPER_OPEN_ACTION = 1.0  # raw normalized env action, matches grasp_trigger_test.py's GRIPPER_OPEN
GRIPPER_CLOSE_ACTION = -1.0
DIST_THRESHOLD = 0.01  # matches grasp_trigger_test.py
MAX_STEPS_PER_WAYPOINT = 60  # matches grasp_trigger_test.py's MAX_STEPS_PER_PHASE
GRASP_HOLD_TICKS = 25  # matches grasp_trigger_test.py's GRASP_HOLD_STEPS -- the gripper closes at a
# finite rate (MAX_GRIPPER_SPEED_RAD_S), so it needs sustained ticks at the grasp point to actually
# close on the can before transport starts moving away
LIFT_Z = 0.15  # matches grasp_trigger_test.py's LIFT_Z -- see generate_candidate()'s lift phase

REFS_REPO_ID = "hungdo2401/so101_mimicgen_refs"
REFS_POSE_LOG = _PROJECT_ROOT / "mimicgen_ref_poses.json"

# Duplicated from lerobot_bridge/.../so101_mujoco.py -- same convention (0% =
# joint's min angle = CLOSED, 100% = max angle = OPEN).
GRIPPER_RANGE_RAD = (-0.174533, 1.74533)


def gripper_pct_to_rad(pct: float) -> float:
    lo, hi = GRIPPER_RANGE_RAD
    return float(lo + np.clip(pct, 0.0, 100.0) / 100.0 * (hi - lo))


def gripper_rad_to_pct(rad: float) -> float:
    lo, hi = GRIPPER_RANGE_RAD
    return float(np.clip((rad - lo) / (hi - lo), 0.0, 1.0) * 100.0)


def joint_rad_to_deg_pct(qpos_rad: np.ndarray) -> np.ndarray:
    """Inverse of action_to_env_delta()'s unit convention: absolute joint values in
    SO101PenPickPlaceEnv's own radians -> the recorded dataset's degree/pct
    convention (5 arm joints in degrees, gripper in [0, 100] pct). Used when
    writing generated episodes into the augmented dataset (Step 6)."""
    out = np.rad2deg(np.asarray(qpos_rad, dtype=np.float64)).astype(np.float32)
    out[5] = gripper_rad_to_pct(qpos_rad[5])
    return out


def action_to_env_delta(action_deg_pct: np.ndarray, qpos_rad: np.ndarray, env: SO101PenPickPlaceEnv) -> np.ndarray:
    """Recorded absolute degree/pct action -> SO101PenPickPlaceEnv's normalized
    per-step delta action. Identical to eval_policy.py's function of the same name."""
    delta = np.zeros(len(ARM_JOINTS), dtype=np.float32)
    for i, joint in enumerate(ARM_JOINTS):
        target_rad = gripper_pct_to_rad(action_deg_pct[i]) if joint == "gripper" else np.deg2rad(action_deg_pct[i])
        raw_delta = target_rad - qpos_rad[i]
        delta[i] = np.clip(raw_delta / env._max_delta[joint], -1.0, 1.0)
    return delta


def load_reference_poses() -> dict[int, np.ndarray]:
    if not REFS_POSE_LOG.exists():
        raise FileNotFoundError(
            f"{REFS_POSE_LOG} not found -- expected the JSON produced by "
            f"record_dataset.py --can-xy-log-path when the reference episodes were recorded."
        )
    with open(REFS_POSE_LOG) as f:
        raw = json.load(f)
    return {int(k): np.asarray(v, dtype=float) for k, v in raw.items()}


def replay_episode(dataset, episode_idx: int, can_xy: np.ndarray, env: SO101PenPickPlaceEnv) -> dict:
    """Replay one recorded episode's actions through real physics, starting the can
    at its logged position. Returns success flag + two reconstructed Cartesian
    fingertip trajectories (one point per tick): 'fingertip_path', evaluated at the
    CURRENT gripper opening, for display/segmentation only (shows the fingertip's
    actual path, gripper motion included); and 'fingertip_closed_path', evaluated
    at the gripper's CLOSED limit at every tick (holding other joints as recorded)
    -- the convention FingertipIK.position_action() always measures its error in,
    since it targets the closed-limit fingertip point regardless of current
    opening (see ik_utils.py). extract_waypoints()/generate_candidate() need
    targets in THAT frame: feeding 'fingertip_path' (current-opening) as an IK
    target while the controller measures error in the closed-limit frame is a
    systematic offset large enough that the gripper closes on empty air next to
    the can instead of on it -- verified directly (dev trace: gripper closed to
    its exact limit, can never left the table)."""
    row = dataset.meta.episodes[episode_idx]
    start, end = row["dataset_from_index"], row["dataset_to_index"]

    obs, _ = env.reset(options={"can_xy": can_xy})
    qpos_rad = obs["agent_pos"][: len(ARM_JOINTS)]
    gripper_closed_qpos = env._ctrlrange["gripper"][0]

    success = False
    fingertip_path = []
    fingertip_closed_path = []
    wrist_roll_path = []
    gripper_pct_path = []
    for i in range(start, end):
        action_deg_pct = dataset[i]["action"].numpy()
        env_action = action_to_env_delta(action_deg_pct, qpos_rad, env)
        obs, _reward, _term, truncated, info = env.step(env_action)
        qpos_rad = obs["agent_pos"][: len(ARM_JOINTS)]

        gripper_rad = qpos_rad[5]
        fingertip_path.append(fingertip_at(env, gripper_rad))
        fingertip_closed_path.append(fingertip_at(env, gripper_closed_qpos))
        wrist_roll_path.append(float(qpos_rad[4]))  # ARM_JOINTS[4] == "wrist_roll"
        gripper_pct_path.append(float(action_deg_pct[5]))

        if info["succeed"]:
            success = True
        if truncated:
            break

    return {
        "episode_idx": episode_idx,
        "can_xy": can_xy,
        "success": success,
        "n_ticks": len(fingertip_path),
        "fingertip_path": np.array(fingertip_path),
        "fingertip_closed_path": np.array(fingertip_closed_path),
        "wrist_roll_path": np.array(wrist_roll_path),
        "gripper_pct_path": np.array(gripper_pct_path),
    }


# gripper_pct convention: 0% = closed, 100% = open (see GRIPPER_RANGE_RAD comment).
# A deadband between these two thresholds avoids chattering on borderline values --
# a tick only counts toward a transition once it's clearly on one side or the other.
GRIPPER_CLOSED_THRESHOLD = 30.0
# Checked the actual recorded gripper trajectories directly: home/resting sits at
# ~9%, but the approach-phase opening varies by episode/operator and sometimes only
# reaches the mid-40s% (not a full-width open) -- 50 was too strict and caused 3/9
# references to only detect a later, wider release-phase opening instead of the true
# (narrower) approach-phase one. 35 stays safely above the 9% resting baseline while
# catching every approach-phase opening observed across all 9 references.
GRIPPER_OPEN_THRESHOLD = 35.0
SUSTAINED_TICKS = 10  # a transition must hold for this many consecutive ticks to count


def _first_sustained_crossing(gripper_pct_path: np.ndarray, start: int, below: float | None, above: float | None) -> int | None:
    """First index >= start where the gripper stays continuously below `below` (or
    above `above`) for SUSTAINED_TICKS in a row. Returns None if it never happens."""
    n = len(gripper_pct_path)
    for i in range(start, n - SUSTAINED_TICKS + 1):
        window = gripper_pct_path[i : i + SUSTAINED_TICKS]
        if below is not None and np.all(window < below):
            return i
        if above is not None and np.all(window > above):
            return i
    return None


def segment_episode(result: dict) -> dict:
    """Splits a replayed episode into 3 phases on gripper-command threshold
    crossings: approach+grasp (can-relative, needs retargeting to a new can
    position), transport (bin-relative, fixed), place+retreat (bin-relative,
    fixed). See the MimicGen plan doc for why only 2 transitions/3 phases are
    needed for this task specifically (axially symmetric can, fixed bin).

    Recordings start near the resting/home gripper position (~9%), which is
    already BELOW the "closed" threshold -- searching for "closed" from tick 0
    would trivially (and wrongly) match immediately. Anchor on the approach's own
    open-gripper phase first (t_open), then search for the grasp-close AFTER that."""
    gripper_pct_path = result["gripper_pct_path"]

    t_open = _first_sustained_crossing(gripper_pct_path, start=0, below=None, above=GRIPPER_OPEN_THRESHOLD)
    if t_open is None:
        return {**result, "t_open": None, "t_grasp": None, "t_release": None, "segmented_ok": False}

    t_grasp = _first_sustained_crossing(gripper_pct_path, start=t_open, below=GRIPPER_CLOSED_THRESHOLD, above=None)
    if t_grasp is None:
        return {**result, "t_open": t_open, "t_grasp": None, "t_release": None, "segmented_ok": False}

    t_release = _first_sustained_crossing(gripper_pct_path, start=t_grasp, below=None, above=GRIPPER_OPEN_THRESHOLD)
    if t_release is None:
        return {**result, "t_open": t_open, "t_grasp": t_grasp, "t_release": None, "segmented_ok": False}

    return {**result, "t_open": t_open, "t_grasp": t_grasp, "t_release": t_release, "segmented_ok": True}


def _sample_indices(start: int, end: int, count: int) -> list[int]:
    """`count` evenly-spaced integer indices spanning [start, end] inclusive."""
    if end <= start:
        return [start] * count
    return [start + round(i * (end - start) / (count - 1)) for i in range(count)]


def extract_waypoints(seg: dict, n_transport: int = 4, n_place: int = 3) -> dict:
    """'approach_path' is the FULL per-tick fingertip path from t_open to t_grasp,
    CAN-RELATIVE (the caller translates every point by the same dx_dy) -- kept
    dense, not sparsely sampled, because point-to-point IK convergence between a
    handful of sparse waypoints reintroduces exactly the "no path planning"
    collision failure grasp_trigger_test.py's own docstring already documents: a
    dev trace on this file (zero net translation, so this should reduce to
    replaying the reference almost exactly) showed the can getting knocked
    10-20cm off course at the very waypoint that swept toward the grasp point,
    even after inserting an explicit hover-then-descend split. Tracking the
    human's own dense path tick-by-tick (see generate_candidate) reproduces the
    same collision-avoiding motion shape the human actually used instead.

    'transport_place' stays a SPARSE, BIN-RELATIVE (used as-is) waypoint list --
    that segment moves through open air toward a fixed bin, so the collision risk
    that motivates dense tracking during approach doesn't apply here, per the
    plan's own "transport/place replayed almost as-is" assumption (verified in
    segment_episode()'s own testing: fingertip@release clusters tightly across
    all 9 references regardless of can position).

    Sourced from 'fingertip_closed_path', not 'fingertip_path' -- these are IK
    control targets, and FingertipIK always measures error in the closed-limit
    frame (see replay_episode()'s docstring).

    Also returns 'approach_wrist_roll', the human's own recorded wrist_roll angle
    over the same span: FingertipIK's 4-DOF IK is redundant for a 3D position
    target and never controls wrist_roll at all, so it can converge to a
    different arm configuration (different roll) than the human used even while
    matching the fingertip point exactly -- for this single-jaw-against-fixed-palm
    gripper, the wrong roll means the can just isn't between the pincers when the
    jaw closes. Verified directly: even with 'fingertip_closed_path' wired in, the
    gripper closed fully (no stall on contact) and the can toppled rather than
    lifted. Replaying the recorded roll needs no retargeting since a pure XY
    translation of an axially-symmetric can doesn't change the needed approach
    orientation."""
    fp = seg["fingertip_closed_path"]
    wr = seg["wrist_roll_path"]
    t_open, t_grasp, t_release, n = seg["t_open"], seg["t_grasp"], seg["t_release"], seg["n_ticks"]

    approach_path = [fp[i].copy() for i in range(t_open, t_grasp + 1)]
    approach_wrist_roll = [float(wr[i]) for i in range(t_open, t_grasp + 1)]

    transport_idx = _sample_indices(t_grasp, t_release, n_transport)
    place_idx = _sample_indices(t_release, n - 1, n_place)
    transport_wps = [(fp[i].copy(), GRIPPER_CLOSE_ACTION) for i in transport_idx]
    place_wps = [(fp[i].copy(), GRIPPER_OPEN_ACTION) for i in place_idx]

    return {
        "approach_path": approach_path,
        "approach_wrist_roll": approach_wrist_roll,
        "transport_place": transport_wps + place_wps,
    }


def nearest_reference(target_xy: np.ndarray, references: dict[int, dict]) -> int:
    """Which reference episode's own can position is closest to target_xy --
    minimizes the translation distance, reducing IK-extrapolation/reachability
    risk per the plan."""
    dists = {ep: np.linalg.norm(np.asarray(seg["can_xy"]) - target_xy) for ep, seg in references.items()}
    return min(dists, key=dists.get)


def _compute_action(
    env: SO101PenPickPlaceEnv,
    ik: FingertipIK,
    target: np.ndarray,
    gripper_action: float,
    wrist_roll_target: float | None = None,
) -> tuple[np.ndarray, float]:
    pos_actions, err = ik.position_action(target)
    action = np.zeros(len(ARM_JOINTS), dtype=np.float32)
    for i, joint in enumerate(ARM_JOINTS):
        if joint in pos_actions:
            action[i] = pos_actions[joint]
        elif joint == "gripper":
            action[i] = gripper_action
        elif joint == "wrist_roll" and wrist_roll_target is not None:
            cur = env.data.qpos[env._joint_qpos_addr["wrist_roll"]]
            action[i] = float(np.clip((wrist_roll_target - cur) / env._max_delta["wrist_roll"], -1.0, 1.0))
    return action, err


def _build_frame(obs: dict, env: SO101PenPickPlaceEnv) -> dict:
    """One LeRobotDataset frame: 'observation.*' from `obs` (captured BEFORE this
    tick's action was applied) paired with 'action' read back from env.data.ctrl
    (the absolute target this tick's step() just committed) -- same
    obs_t/action_t pairing convention as lerobot's own
    examples/smolvla_robosuite_record_scripted.py's build_frame()."""
    action_ctrl_rad = np.array([env.data.ctrl[env._actuator_id[j]] for j in ARM_JOINTS])
    frame = {
        "observation.state": joint_rad_to_deg_pct(obs["agent_pos"][: len(ARM_JOINTS)]),
        "action": joint_rad_to_deg_pct(action_ctrl_rad),
        "task": SINGLE_TASK,
    }
    if env.image_obs:
        frame["observation.images.front"] = obs["pixels"]["front"]
        frame["observation.images.wrist"] = obs["pixels"]["wrist"]
    return frame


def run_waypoint_sequence(
    env: SO101PenPickPlaceEnv,
    ik: FingertipIK,
    waypoints: list[tuple[np.ndarray, float]],
    obs: dict | None = None,
    frames: list | None = None,
    stop_on_success: bool = True,
) -> tuple[bool, int, dict | None]:
    """Drive through a flat list of (fingertip_target, gripper_action) waypoints,
    advancing on convergence or after MAX_STEPS_PER_WAYPOINT -- same
    advance-on-convergence control loop as grasp_trigger_test.py's run_episode(),
    just over a flat waypoint list instead of named phases. `obs`/`frames` are
    optional recording hooks (see generate_candidate's `record` flag); when
    `frames` is None this costs nothing extra.

    `stop_on_success=True` (default): return the instant _is_success() first
    trips, which only checks position (low enough, inside the bin's xy
    footprint) -- never velocity -- so this can fire mid-drop, not once the can
    has visibly settled. Fine for the cheap dry-run accept/reject pass (Step 5),
    which only cares whether success happens at all. `stop_on_success=False`
    (used when record=True): keep running through the REST of the waypoint
    list regardless, so the written episode keeps its place+retreat tail --
    otherwise every recorded episode cuts off at the instant of release, with
    none of the settle/retreat frames the human reference episodes have.
    Verified directly: the pushed hungdo2401/so101_mimicgen_aug's episodes all
    ended right as the can left the gripper (visible in the HF dataset viewer)."""
    total_ticks = 0
    success_ever = False
    for target, gripper_action in waypoints:
        for _ in range(MAX_STEPS_PER_WAYPOINT):
            action, err = _compute_action(env, ik, target, gripper_action)
            new_obs, _reward, _term, truncated, info = env.step(action)
            total_ticks += 1
            if frames is not None and obs is not None:
                frames.append(_build_frame(obs, env))
            obs = new_obs
            success_ever = success_ever or info["succeed"]
            if info["succeed"] and stop_on_success:
                return True, total_ticks, obs
            if truncated:
                return success_ever, total_ticks, obs
            if err < DIST_THRESHOLD:
                break
    return (success_ever or bool(env._is_success())), total_ticks, obs


def _drive_one_tick(
    env: SO101PenPickPlaceEnv,
    ik: FingertipIK,
    target: np.ndarray,
    gripper_action: float,
    wrist_roll_target: float | None = None,
    obs: dict | None = None,
    frames: list | None = None,
) -> tuple[dict, float, bool, bool]:
    action, err = _compute_action(env, ik, target, gripper_action, wrist_roll_target)
    new_obs, _reward, _term, truncated, info = env.step(action)
    if frames is not None and obs is not None:
        frames.append(_build_frame(obs, env))
    return new_obs, err, bool(info["succeed"]), bool(truncated)


def generate_candidate(
    target_xy: np.ndarray,
    references: dict[int, dict],
    env: SO101PenPickPlaceEnv,
    record: bool = False,
) -> dict:
    """Reset the env with the can at target_xy, retarget the nearest reference's
    approach path by translation, and run four phases: (1) dense per-tick
    tracking along the translated approach path, gripper open -- reproduces the
    human's own collision-avoiding motion shape, see extract_waypoints(); (2) a
    fixed grasp-hold at the final approach point, gripper closed, giving the
    finite-rate gripper time to actually close on the can (matches
    grasp_trigger_test.py's GRASP_HOLD_STEPS); (3) an explicit straight-up lift
    (self-canceling XY, matching grasp_trigger_test.py's own "lift" phase) before
    (4) the sparse, bin-relative transport/place waypoints, used as-is (skipping
    their first entry, which duplicates the grasp point). The lift is needed for
    the same reason the approach got a hover-then-descend split: transport_place's
    first two waypoints jump from grasp height straight to a high point ~20-25cm
    away in one combined XY+Z move, which knocked/launched the can in dev
    testing (verified directly) -- separating "straight up" from "move to bin"
    avoids it.

    `record=False` (default): no rendering, cheap dry-run verification pass.
    `record=True`: also returns 'frames', a list of LeRobotDataset-ready frame
    dicts (requires `env` to have been constructed with image_obs=True) -- used
    to write accepted candidates into the augmented dataset (Step 6). Passing the
    identical target_xy/references/deterministic physics through this same
    function on an image_obs=True env reproduces the exact winning trajectory
    found during the dry run."""
    ref_ep = nearest_reference(target_xy, references)
    seg = references[ref_ep]
    wps = extract_waypoints(seg)

    dx_dy = target_xy - np.asarray(seg["can_xy"])
    offset = np.array([dx_dy[0], dx_dy[1], 0.0])
    translated_approach = [p + offset for p in wps["approach_path"]]
    approach_wrist_roll = wps["approach_wrist_roll"]
    grasp_wrist_roll = approach_wrist_roll[-1]

    obs, _ = env.reset(options={"can_xy": target_xy})
    ik = FingertipIK(env)

    frames: list | None = [] if record else None
    total_ticks = 0
    success = False
    truncated = False

    for target, wr_target in zip(translated_approach, approach_wrist_roll):
        obs, _err, success, truncated = _drive_one_tick(
            env, ik, target, GRIPPER_OPEN_ACTION, wrist_roll_target=wr_target, obs=obs, frames=frames
        )
        total_ticks += 1
        if success or truncated:
            break

    if not success and not truncated:
        grasp_target = translated_approach[-1]
        for _ in range(GRASP_HOLD_TICKS):
            obs, _err, success, truncated = _drive_one_tick(
                env, ik, grasp_target, GRIPPER_CLOSE_ACTION, wrist_roll_target=grasp_wrist_roll, obs=obs, frames=frames
            )
            total_ticks += 1
            if success or truncated:
                break

    if not success and not truncated:
        lift_xy = ik.fingertip_at_closed()[:2]
        lift_target = np.array([lift_xy[0], lift_xy[1], LIFT_Z])
        for _ in range(MAX_STEPS_PER_WAYPOINT):
            obs, err, success, truncated = _drive_one_tick(
                env, ik, lift_target, GRIPPER_CLOSE_ACTION, wrist_roll_target=grasp_wrist_roll, obs=obs, frames=frames
            )
            total_ticks += 1
            if success or truncated or err < DIST_THRESHOLD:
                break

    if not success and not truncated:
        success, tp_ticks, obs = run_waypoint_sequence(
            env, ik, wps["transport_place"][1:], obs=obs, frames=frames, stop_on_success=not record
        )
        total_ticks += tp_ticks

    result = {
        "target_xy": target_xy,
        "ref_episode": ref_ep,
        "success": success,
        "ticks": total_ticks,
    }
    if record:
        result["frames"] = frames
    return result


def load_segmented_references(env: SO101PenPickPlaceEnv) -> dict[int, dict]:
    """Replay + segment every reference episode, keyed by episode index. Raises if
    any reference fails verification or segmentation -- generation should never
    proceed on top of an unverified/unsegmentable reference."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    poses = load_reference_poses()
    dataset = LeRobotDataset(REFS_REPO_ID)
    references = {}
    for ep, can_xy in poses.items():
        result = replay_episode(dataset, ep, can_xy, env)
        if not result["success"]:
            raise RuntimeError(f"reference episode {ep} failed physics-replay verification -- fix before generating")
        seg = segment_episode(result)
        if not seg["segmented_ok"]:
            raise RuntimeError(f"reference episode {ep} could not be segmented (t_open={seg['t_open']}, t_grasp={seg['t_grasp']}, t_release={seg['t_release']})")
        references[ep] = seg
    return references


def cmd_generate(args: argparse.Namespace) -> None:
    env = SO101PenPickPlaceEnv(control_dt=0.05, randomize_can_pose=False)
    references = load_segmented_references(env)
    print(f"Loaded {len(references)} verified, segmented references.")

    rng = np.random.RandomState(args.seed)
    candidates = rng.uniform(CAN_SAMPLING_XY_LOW, CAN_SAMPLING_XY_HIGH, size=(args.n_candidates, 2))

    import time

    t0 = time.perf_counter()
    results = []
    for i, target_xy in enumerate(candidates):
        r = generate_candidate(target_xy, references, env)
        results.append(r)
        print(f"candidate {i + 1}/{len(candidates)}: target={np.round(target_xy, 3)} ref_ep={r['ref_episode']} -> {'SUCCESS' if r['success'] else 'fail'} ({r['ticks']} ticks)")
    elapsed = time.perf_counter() - t0

    n_ok = sum(r["success"] for r in results)
    print(f"\n{n_ok}/{len(results)} accepted ({n_ok / len(results):.1%})")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / len(results):.2f}s/candidate, dry run, no rendering)")

    env.close()


def cmd_build_dataset(args: argparse.Namespace) -> None:
    """Step 5+6 combined: dry-run-generate up to `args.n_candidates` targets (cheap,
    no rendering), then re-run every ACCEPTED one a second time on an
    image_obs=True env to reconstruct the exact same winning trajectory with
    frames, writing each into a LeRobotDataset at args.repo_id matching
    hungdo2401/so101_baseline's schema exactly (AUG_DATASET_FEATURES). Physics and
    this project's own IK controller are both deterministic given the same
    starting state and target, so replaying the identical (target_xy, ref_episode)
    pair on a fresh env reproduces the identical outcome -- verified per-episode
    below (skips with a warning on the rare mismatch rather than writing a
    partial/wrong episode)."""
    import time

    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    dry_env = SO101PenPickPlaceEnv(control_dt=0.05, randomize_can_pose=False)
    references = load_segmented_references(dry_env)
    print(f"Loaded {len(references)} verified, segmented references.")

    rng = np.random.RandomState(args.seed)
    candidates = rng.uniform(CAN_SAMPLING_XY_LOW, CAN_SAMPLING_XY_HIGH, size=(args.n_candidates, 2))

    t0 = time.perf_counter()
    accepted = []
    for i, target_xy in enumerate(candidates):
        r = generate_candidate(target_xy, references, dry_env)
        status = "accept" if r["success"] else "reject"
        print(f"dry-run {i + 1}/{len(candidates)}: target={np.round(target_xy, 3)} ref_ep={r['ref_episode']} -> {status} ({r['ticks']} ticks)")
        if r["success"]:
            accepted.append(target_xy)
        if args.max_episodes and len(accepted) >= args.max_episodes:
            break
    dry_env.close()
    dry_elapsed = time.perf_counter() - t0
    print(f"\n{len(accepted)}/{i + 1} accepted ({dry_elapsed:.1f}s dry run) -- rendering + writing dataset now.\n")

    render_env = SO101PenPickPlaceEnv(
        control_dt=0.05, randomize_can_pose=False, image_obs=True, render_size=RENDER_SIZE
    )

    root = HF_LEROBOT_HOME / args.repo_id
    if root.exists():
        dataset = LeRobotDataset(args.repo_id, root=root, image_writer_threads=4, image_writer_processes=0)
        print(f"Resuming existing dataset at {root} -- {dataset.meta.total_episodes} episode(s) already saved.")
    else:
        dataset = LeRobotDataset.create(
            args.repo_id,
            fps=20,
            features=AUG_DATASET_FEATURES,
            root=root,
            use_videos=True,
            image_writer_threads=4,
            image_writer_processes=0,
            robot_type="so101_mujoco",
        )

    n_written = 0
    t0 = time.perf_counter()
    try:
        for i, target_xy in enumerate(accepted):
            r = generate_candidate(target_xy, references, render_env, record=True)
            if not r["success"]:
                print(f"episode {i}: re-render mismatch (accepted on dry run, failed on record pass) -- skipping")
                continue
            for frame in r["frames"]:
                dataset.add_frame(frame)
            dataset.save_episode()
            n_written += 1
            print(f"episode {i + 1}/{len(accepted)}: written ({len(r['frames'])} frames, ref_ep={r['ref_episode']})")
    finally:
        if dataset.writer is not None and dataset.writer.image_writer is not None:
            dataset.writer.image_writer.stop()
        render_env.close()
        dataset.finalize()
    render_elapsed = time.perf_counter() - t0

    print(f"\n{n_written}/{len(accepted)} episodes written to {dataset.root} ({render_elapsed:.1f}s)")

    if args.push_to_hub:
        dataset.push_to_hub()
        print(f"Pushed to https://huggingface.co/datasets/{args.repo_id}")


def cmd_combine(args: argparse.Namespace) -> None:
    """Step 7: combine the original 50-episode baseline with the MimicGen-augmented
    set into a third dataset, WITHOUT touching either source -- per the plan,
    baseline + augmented only (the 9 reference episodes stay a separate,
    unaggregated dataset)."""
    from lerobot.datasets.aggregate import aggregate_datasets
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.utils.constants import HF_LEROBOT_HOME

    aggr_root = HF_LEROBOT_HOME / args.combined_repo_id
    if aggr_root.exists():
        raise FileExistsError(
            f"{aggr_root} already exists -- aggregate_datasets() only creates a fresh "
            f"dataset; remove it first if you want to rebuild the combined dataset."
        )

    print(f"Aggregating {args.baseline_repo_id} + {args.aug_repo_id} -> {args.combined_repo_id} ...")
    aggregate_datasets(repo_ids=[args.baseline_repo_id, args.aug_repo_id], aggr_repo_id=args.combined_repo_id)

    combined = LeRobotDataset(args.combined_repo_id)
    print(f"Combined dataset: {combined.meta.total_episodes} episodes, {combined.meta.total_frames} frames.")

    if args.push_to_hub:
        combined.push_to_hub()
        print(f"Pushed to https://huggingface.co/datasets/{args.combined_repo_id}")


def cmd_segment_refs(args: argparse.Namespace) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    poses = load_reference_poses()
    dataset = LeRobotDataset(REFS_REPO_ID)
    env = SO101PenPickPlaceEnv(control_dt=0.05, randomize_can_pose=False)

    for ep, can_xy in poses.items():
        result = replay_episode(dataset, ep, can_xy, env)
        seg = segment_episode(result)
        n = seg["n_ticks"]
        if not seg["segmented_ok"]:
            print(f"episode {ep}: can_xy={can_xy} -> SEGMENTATION FAILED (t_open={seg['t_open']}, t_grasp={seg['t_grasp']}, t_release={seg['t_release']}, n_ticks={n})")
            continue
        t_open, t_grasp, t_release = seg["t_open"], seg["t_grasp"], seg["t_release"]
        grasp_pos = seg["fingertip_path"][t_grasp]
        release_pos = seg["fingertip_path"][t_release]
        print(
            f"episode {ep}: can_xy={can_xy} -> home=[0,{t_open}) approach=[{t_open},{t_grasp}) "
            f"transport=[{t_grasp},{t_release}) place=[{t_release},{n}) | "
            f"fingertip@grasp={np.round(grasp_pos, 3)} fingertip@release={np.round(release_pos, 3)}"
        )

    env.close()


def cmd_verify_refs(args: argparse.Namespace) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    poses = load_reference_poses()
    dataset = LeRobotDataset(REFS_REPO_ID)
    print(f"Loaded {dataset.meta.total_episodes} reference episodes, {len(poses)} logged poses.")

    env = SO101PenPickPlaceEnv(control_dt=0.05, randomize_can_pose=False)

    results = []
    for ep in range(dataset.meta.total_episodes):
        if ep not in poses:
            print(f"episode {ep}: NO LOGGED POSE -- skipping (can't replay without a known start)")
            continue
        result = replay_episode(dataset, ep, poses[ep], env)
        results.append(result)
        status = "SUCCESS" if result["success"] else "FAIL"
        print(
            f"episode {ep}: can_xy={poses[ep]} -> {status} "
            f"({result['n_ticks']} ticks replayed)"
        )

    n_ok = sum(r["success"] for r in results)
    print(f"\n{n_ok}/{len(results)} reference episodes verified successful under real physics replay.")
    if n_ok < len(results):
        print("Failures above should be investigated/re-recorded before proceeding to segmentation --")
        print("a reference that doesn't actually succeed on replay would poison every position retargeted from it.")

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("verify-refs", help="replay all reference episodes and confirm real success")
    subparsers.add_parser("segment-refs", help="replay + split each reference into approach/transport/place phases")
    p_gen = subparsers.add_parser("generate", help="dry-run (no rendering): generate+verify candidates, report accept rate")
    p_gen.add_argument("--n-candidates", type=int, default=30)
    p_gen.add_argument("--seed", type=int, default=0)
    p_build = subparsers.add_parser("build-dataset", help="dry-run generate, then re-render + write accepted episodes to a LeRobotDataset")
    p_build.add_argument("--n-candidates", type=int, default=30)
    p_build.add_argument("--seed", type=int, default=0)
    p_build.add_argument("--max-episodes", type=int, default=0, help="stop accepting once this many candidates pass (0 = no cap, use all n-candidates)")
    p_build.add_argument("--repo-id", default=AUG_REPO_ID)
    p_build.add_argument("--push-to-hub", action="store_true")
    p_combine = subparsers.add_parser("combine", help="aggregate baseline + augmented datasets into a third, combined dataset")
    p_combine.add_argument("--baseline-repo-id", default=BASELINE_REPO_ID)
    p_combine.add_argument("--aug-repo-id", default=AUG_REPO_ID)
    p_combine.add_argument("--combined-repo-id", default=COMBINED_REPO_ID)
    p_combine.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    if args.command == "verify-refs":
        cmd_verify_refs(args)
    elif args.command == "segment-refs":
        cmd_segment_refs(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "build-dataset":
        cmd_build_dataset(args)
    elif args.command == "combine":
        cmd_combine(args)


if __name__ == "__main__":
    main()
