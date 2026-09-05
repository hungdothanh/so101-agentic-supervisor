#!/usr/bin/env python3
"""Sim-eval: measure a fine-tuned SmolVLA checkpoint's success rate directly against
SO101PenPickPlaceEnv (privileged ground-truth success/pose -- the same env used by
the scripted grasp-trigger test), rather than through lerobot-rollout/lerobot-eval.

WHY NOT lerobot-rollout OR lerobot-eval: lerobot-rollout drives the SO101Mujoco
Robot/Teleoperator bridge, which -- matching real so_follower hardware -- exposes
state+images only, with no success signal and no reset hook between attempts.
lerobot-eval needs a registered EnvConfig/gym environment (like the built-in
pusht/aloha/gym_manipulator ones), which would be new engineering roughly the size
of the bridge itself. This script instead reuses the checkpoint-loading pattern
from examples/smolvla_gym_lowcostrobot_finetuned.py in the lerobot repo, applied
directly to our own env.

UNIT/FEATURE CONVERSION (the part that's easy to get subtly wrong):
  - The checkpoint was fine-tuned on data recorded through the SO101Mujoco Robot
    bridge: degrees for the 5 arm joints, [0,100] for the gripper, camera keys
    renamed front->camera1 / wrist->camera2 (see --rename_map in the training
    command). SO101PenPickPlaceEnv's own `agent_pos` is NOT that -- it's raw
    radians, 12-wide (6 qpos + 6 qvel, for its own RL action interface).
    build_observation() below converts qpos only (drops qvel, which the
    checkpoint never saw) into the checkpoint's trained degrees/pct/camera1/
    camera2 feature space.
  - The checkpoint outputs ABSOLUTE degree/pct targets (the same SOFollower-style
    contract SO101Mujoco.send_action() expects), but SO101PenPickPlaceEnv.step()
    wants a normalized PER-STEP DELTA (its own RL action interface -- see that
    module's docstring). action_to_env_delta() converts absolute target ->
    one-step delta -> normalized by each joint's own max speed, mirroring
    SO101MujocoTeleop's integration in reverse. It reaches into
    env._max_delta/_ctrlrange directly for this -- the same "internal-use
    interface within this project" pattern scripts/grasp_trigger_test.py
    already relies on for the identical kind of joint-space math.
  - render_size is set to (480, 480) to match SO101MujocoConfig.camera_shapes'
    training-time resolution exactly (the env's own default is 128x128) --
    both get resized to the checkpoint's resize_imgs_with_padding=(512,512)
    internally either way, but starting from a much lower native resolution
    than training used would be a real, confounding train/eval visual mismatch,
    not a fair test of the policy itself.
  - The 3rd camera slot (--policy.empty_cameras=1 at training time) does NOT
    need to be supplied here -- verified directly in modeling_smolvla.py's
    prepare_images(): the model's own forward pass auto-pads any declared-but-
    missing camera key with a zeroed image, up to config.empty_cameras. Only
    camera1/camera2 need to be in the observation dict.

Run inside the `lerobot` conda env, from the repo root:
    python scripts/eval_policy.py \\
        --policy-path=hungdo2401/smolvla_so101_baseline \\
        --num-episodes=20 \\
        --single-task="pick up the can and place it in the bin"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from lerobot.common.control_utils import predict_action
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy

# Plain `python scripts/eval_policy.py` puts scripts/ on sys.path[0], not the repo
# root -- so101_mujoco_env wouldn't otherwise be importable. Same fix as
# lerobot_bridge/.../so101_mujoco_teleop.py's own sys.path insertion.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from so101_mujoco_env.pen_pickplace_env import ARM_JOINTS, SO101PenPickPlaceEnv  # noqa: E402

# Duplicated from lerobot_bridge/.../so101_mujoco.py -- same convention (0% =
# joint's min angle = CLOSED, 100% = max angle = OPEN). See that module's
# docstring for why this specific direction was chosen.
GRIPPER_RANGE_RAD = (-0.174533, 1.74533)


def gripper_rad_to_pct(angle_rad: float) -> float:
    lo, hi = GRIPPER_RANGE_RAD
    return float(np.clip((angle_rad - lo) / (hi - lo) * 100.0, 0.0, 100.0))


def gripper_pct_to_rad(pct: float) -> float:
    lo, hi = GRIPPER_RANGE_RAD
    return float(lo + np.clip(pct, 0.0, 100.0) / 100.0 * (hi - lo))


def build_observation(raw_obs: dict) -> dict:
    """SO101PenPickPlaceEnv's native obs -> the checkpoint's trained feature space."""
    qpos = raw_obs["agent_pos"][: len(ARM_JOINTS)]  # first half is qpos; second half (qvel) is dropped
    state = np.array(
        [
            np.rad2deg(qpos[0]),  # shoulder_pan
            np.rad2deg(qpos[1]),  # shoulder_lift
            np.rad2deg(qpos[2]),  # elbow_flex
            np.rad2deg(qpos[3]),  # wrist_flex
            np.rad2deg(qpos[4]),  # wrist_roll
            gripper_rad_to_pct(qpos[5]),  # gripper
        ],
        dtype=np.float32,
    )
    return {
        "observation.state": state,
        "observation.images.camera1": raw_obs["pixels"]["front"],
        "observation.images.camera2": raw_obs["pixels"]["wrist"],
    }


def action_to_env_delta(action_deg_pct: np.ndarray, qpos_rad: np.ndarray, env: SO101PenPickPlaceEnv) -> np.ndarray:
    """Checkpoint's absolute degree/pct target -> SO101PenPickPlaceEnv's normalized
    per-step delta action, via each joint's own _max_delta (see module docstring)."""
    delta = np.zeros(len(ARM_JOINTS), dtype=np.float32)
    for i, joint in enumerate(ARM_JOINTS):
        target_rad = gripper_pct_to_rad(action_deg_pct[i]) if joint == "gripper" else np.deg2rad(action_deg_pct[i])
        raw_delta = target_rad - qpos_rad[i]
        delta[i] = np.clip(raw_delta / env._max_delta[joint], -1.0, 1.0)
    return delta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--policy-path", required=True, help="e.g. hungdo2401/smolvla_so101_baseline, or a local checkpoint dir")
    parser.add_argument("--num-episodes", type=int, default=20)
    parser.add_argument(
        "--control-dt",
        type=float,
        default=0.05,
        help="must match the recording rate the checkpoint was trained at (SO101MujocoConfig.control_dt / "
        "dataset fps=20 -> 0.05s). SO101PenPickPlaceEnv's own default is 0.1s -- left as-is that would "
        "run the policy at half its trained control rate, a real train/eval fidelity mismatch.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="per-episode cap in control ticks. At the default 0.05s control_dt, 600 steps = 30 sim-"
        "seconds -- comfortably above the ~24.5s (489 ticks @ 20Hz) your own recorded episode 49 took.",
    )
    parser.add_argument("--single-task", required=True, help="must match the task string the checkpoint was fine-tuned on")
    parser.add_argument("--seed", type=int, default=0, help="episode 0 uses this seed, episode i uses seed+i")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--camera-preview",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="show a live front+wrist window while evaluating -- reuses the frames already rendered "
        "for the policy each step (no extra render pass). Off by default since a real benchmark run "
        "(e.g. 50 episodes) doesn't need a window; turn it on to visually inspect what's happening.",
    )
    parser.add_argument("--preview-scale", type=float, default=1.0, help="extra upscale for the preview window only")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading {args.policy_path} on {device} ...")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.to(torch.float32)
    policy.to(device)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        args.policy_path,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )

    env = SO101PenPickPlaceEnv(
        image_obs=True, randomize_can_pose=True, render_size=(480, 480), control_dt=args.control_dt
    )

    cam_window_name = "SO-101 eval: front | wrist"
    quit_requested = False

    def show_preview(raw_obs: dict, episode: int, step: int, status: str) -> bool:
        """Reuses the frames raw_obs already carries (no extra render pass -- same
        pattern as record_dataset.py's camera-preview). Returns False if the user
        closed the window or pressed Q/Esc."""
        import cv2

        frames = []
        for cam_name in ("front", "wrist"):
            bgr = cv2.cvtColor(raw_obs["pixels"][cam_name], cv2.COLOR_RGB2BGR)
            if args.preview_scale != 1.0:
                bgr = cv2.resize(bgr, None, fx=args.preview_scale, fy=args.preview_scale, interpolation=cv2.INTER_LINEAR)
            cv2.putText(bgr, cam_name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            frames.append(bgr)
        combined = cv2.hconcat(frames)
        cv2.putText(
            combined, f"episode {episode + 1}/{args.num_episodes}  step {step}  {status}",
            (10, combined.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2,
        )
        cv2.imshow(cam_window_name, combined)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):  # 27 == Esc
            return False
        return cv2.getWindowProperty(cam_window_name, cv2.WND_PROP_VISIBLE) >= 1

    successes = 0
    episode_results = []
    t_start = time.perf_counter()
    try:
        for episode in range(args.num_episodes):
            raw_obs, _ = env.reset(seed=args.seed + episode)
            policy.reset()  # clears the policy's internal action-chunk queue
            success = False

            if args.camera_preview and not show_preview(raw_obs, episode, 0, "running"):
                quit_requested = True
                break

            for step in range(args.max_steps):
                observation = build_observation(raw_obs)
                with torch.inference_mode():
                    action = predict_action(
                        observation,
                        policy,
                        device,
                        preprocessor,
                        postprocessor,
                        use_amp=False,
                        task=args.single_task,
                        robot_type="so101_mujoco",
                    )
                action_deg_pct = action.squeeze(0).cpu().numpy()

                qpos_rad = raw_obs["agent_pos"][: len(ARM_JOINTS)]
                env_action = action_to_env_delta(action_deg_pct, qpos_rad, env)

                raw_obs, _reward, _terminated, truncated, info = env.step(env_action)

                if args.camera_preview:
                    status = "SUCCESS" if info["succeed"] else ("truncated" if truncated else "running")
                    if not show_preview(raw_obs, episode, step + 1, status):
                        quit_requested = True
                        break

                if info["succeed"]:
                    success = True
                    break
                if truncated:  # can fell off the table -- not recoverable this episode
                    break

            successes += int(success)
            episode_results.append(success)
            print(f"episode {episode + 1}/{args.num_episodes}: {'SUCCESS' if success else 'fail'} (step {step + 1})")

            if quit_requested:
                print("Preview window closed / quit key pressed -- stopping early.")
                break
    finally:
        if args.camera_preview:
            import cv2

            cv2.destroyAllWindows()
        env.close()

    n_run = len(episode_results)
    elapsed = time.perf_counter() - t_start
    rate = successes / n_run if n_run else 0.0
    print(f"\nSuccess rate: {successes}/{n_run} = {rate:.1%}")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / n_run:.1f}s/episode)" if n_run else "\nElapsed: 0.0s")


if __name__ == "__main__":
    main()
