#!/usr/bin/env python3
"""LangGraph supervisor: wraps a SmolVLA checkpoint with a periodic VLM-based failure
check (does the front camera show the can standing/fallen/placed?) and routes to a
second "recovery" checkpoint when the can gets knocked over mid-grasp. See PLAN.md for
the full design (architecture, decision rationale, "what has to be built" ordering).

THIS FILE IMPLEMENTS PLAN.md's STEP 2 ONLY: the graph skeleton with `recover` stubbed
(no real recovery checkpoint trained yet -- pass --recovery-policy-path once one
exists, from step 4/5). Steps 3-5 (recording fallen-can reference demos, fine-tuning a
recovery checkpoint, wiring it in) are separate, hands-on work this script doesn't do.

Self-contained, matching this repo's existing convention (see eval_policy.py's own
docstring): duplicates build_observation()/action_to_env_delta()/gripper-pct
conversions from eval_policy.py rather than importing them, so this script doesn't
depend on that one's internals changing out from under it.

Run inside the `lerobot` conda env, from the repo root (needs an OPENAI_API_KEY in a
local .env file -- see .env.example):
    python scripts/supervisor.py \\
        --base-policy-path=hungdo2401/smolvla_so101_baseline_plus_mimicgen \\
        --single-task="pick up the can and place it in the bin" \\
        --num-episodes=5 --force-fallen-spawn
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import torch
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from lerobot.common.control_utils import predict_action
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla import SmolVLAPolicy

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from so101_mujoco_env.pen_pickplace_env import ARM_JOINTS, SO101PenPickPlaceEnv  # noqa: E402

load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")

# Duplicated from scripts/eval_policy.py -- see that module's docstring for the
# 0%=CLOSED/100%=OPEN convention this mirrors (lerobot_robot_so101_mujoco/so101_mujoco.py).
GRIPPER_RANGE_RAD = (-0.174533, 1.74533)


def gripper_rad_to_pct(angle_rad: float) -> float:
    lo, hi = GRIPPER_RANGE_RAD
    return float(np.clip((angle_rad - lo) / (hi - lo) * 100.0, 0.0, 100.0))


def gripper_pct_to_rad(pct: float) -> float:
    lo, hi = GRIPPER_RANGE_RAD
    return float(lo + np.clip(pct, 0.0, 100.0) / 100.0 * (hi - lo))


def build_observation(raw_obs: dict) -> dict:
    """SO101PenPickPlaceEnv's native obs -> the checkpoint's trained feature space.
    Duplicated from scripts/eval_policy.py -- see that module's docstring for the full
    unit-conversion rationale (radians/qvel-dropped -> trained degrees/pct/camera1/camera2)."""
    qpos = raw_obs["agent_pos"][: len(ARM_JOINTS)]
    state = np.array(
        [
            np.rad2deg(qpos[0]),
            np.rad2deg(qpos[1]),
            np.rad2deg(qpos[2]),
            np.rad2deg(qpos[3]),
            np.rad2deg(qpos[4]),
            gripper_rad_to_pct(qpos[5]),
        ],
        dtype=np.float32,
    )
    return {
        "observation.state": state,
        "observation.images.camera1": raw_obs["pixels"]["front"],
        "observation.images.camera2": raw_obs["pixels"]["wrist"],
    }


def action_to_env_delta(action_deg_pct: np.ndarray, qpos_rad: np.ndarray, env: SO101PenPickPlaceEnv) -> np.ndarray:
    """Duplicated from scripts/eval_policy.py -- checkpoint's absolute degree/pct target
    -> SO101PenPickPlaceEnv's normalized per-step delta action."""
    delta = np.zeros(len(ARM_JOINTS), dtype=np.float32)
    for i, joint in enumerate(ARM_JOINTS):
        target_rad = gripper_pct_to_rad(action_deg_pct[i]) if joint == "gripper" else np.deg2rad(action_deg_pct[i])
        raw_delta = target_rad - qpos_rad[i]
        delta[i] = np.clip(raw_delta / env._max_delta[joint], -1.0, 1.0)
    return delta


class SupervisorState(TypedDict):
    front_image: np.ndarray
    wrist_image: np.ndarray
    tick: int
    active_policy: Literal["base", "recovery"]
    status: Literal["running", "success", "can_fallen", "failed"]
    vlm_verdict: str
    recovery_attempts: int


# ---------------------------------------------------------------------------
# VLM classification (isolated so it can be exercised standalone, per PLAN.md's
# own step-2 verification bullet: "manually feed a handful of front-camera frames
# to the GPT-4o call directly and confirm the classification prompt returns the
# right label").
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """You are monitoring a robot arm pick-and-place task in simulation. \
The task is: "{task}".

Look at the front-camera image and classify the scene into EXACTLY ONE of these three \
labels:
- success: the can is placed inside the bin.
- can_fallen: the can is lying on its side on the table (not in the bin), regardless of \
whether the arm is touching it.
- running: the can is still standing upright on the table and the task is still in \
progress.

Respond with exactly one word: success, can_fallen, or running. No other text."""


def _image_to_data_url(image: np.ndarray) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def _extract_text(content) -> str:
    """response.content is a plain str for a simple OpenAI text reply, but
    langchain-core's newer "content blocks" model can return a list of blocks for
    other providers/response types -- handle both defensively."""
    if isinstance(content, str):
        return content
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and "text" in block:
            parts.append(block["text"])
    return "".join(parts)


def classify_scene(image: np.ndarray, task: str, model: str) -> tuple[str, str]:
    """Returns (label, raw_response_text). label is one of running/can_fallen/success,
    defaulting to "running" (the safest fallback -- keeps the base policy going rather
    than false-triggering a recovery) if the response doesn't parse cleanly."""
    llm = ChatOpenAI(model=model, max_completion_tokens=10, temperature=0)
    message = HumanMessage(
        content=[
            {"type": "text", "text": _CLASSIFY_PROMPT.format(task=task)},
            {"type": "image_url", "image_url": {"url": _image_to_data_url(image)}},
        ]
    )
    response = llm.invoke([message])
    raw = _extract_text(response.content).strip()
    lowered = raw.lower()
    for label in ("success", "can_fallen", "running"):
        if label in lowered:
            return label, raw
    return "running", raw


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


@dataclass
class PolicyBundle:
    policy: SmolVLAPolicy
    preprocessor: object
    postprocessor: object
    device: torch.device


def load_policy_bundle(path: str, device: torch.device) -> PolicyBundle:
    print(f"Loading {path} on {device} ...")
    policy = SmolVLAPolicy.from_pretrained(path)
    policy.to(torch.float32)
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config, path, preprocessor_overrides={"device_processor": {"device": str(device)}}
    )
    return PolicyBundle(policy=policy, preprocessor=preprocessor, postprocessor=postprocessor, device=device)


# ---------------------------------------------------------------------------
# Episode context: everything execute_policy needs to keep stepping the env that
# doesn't belong in SupervisorState (which mirrors PLAN.md's schema exactly, and
# only carries what routing/VLM/logging need -- not the full raw_obs qpos state
# each policy call requires).
# ---------------------------------------------------------------------------


class _EpisodeContext:
    def __init__(self) -> None:
        self.raw_obs: dict = {}
        self.last_active_policy: str | None = None
        self.stub_notice_shown = False
        self.truncated = False


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


def make_execute_policy_node(env: SO101PenPickPlaceEnv, policies: dict, ctx: _EpisodeContext, args):
    def execute_policy(state: SupervisorState) -> dict:
        bundle: PolicyBundle | None = policies[state["active_policy"]]

        if ctx.last_active_policy != state["active_policy"]:
            if bundle is not None:
                bundle.policy.reset()
            ctx.last_active_policy = state["active_policy"]

        n_action_steps = bundle.policy.config.n_action_steps if bundle is not None else args.n_action_steps_fallback

        tick = state["tick"]
        for _ in range(n_action_steps):
            if bundle is None:
                if not ctx.stub_notice_shown:
                    print("[stub] no recovery policy loaded -- holding position")
                    ctx.stub_notice_shown = True
                env_action = np.zeros(len(ARM_JOINTS), dtype=np.float32)
            else:
                observation = build_observation(ctx.raw_obs)
                with torch.inference_mode():
                    action = predict_action(
                        observation,
                        bundle.policy,
                        bundle.device,
                        bundle.preprocessor,
                        bundle.postprocessor,
                        use_amp=False,
                        task=args.single_task,
                        robot_type="so101_mujoco",
                    )
                action_deg_pct = action.squeeze(0).cpu().numpy()
                qpos_rad = ctx.raw_obs["agent_pos"][: len(ARM_JOINTS)]
                env_action = action_to_env_delta(action_deg_pct, qpos_rad, env)

            ctx.raw_obs, _reward, _terminated, truncated, info = env.step(env_action)
            ctx.truncated = bool(truncated)
            tick += 1
            if info["succeed"] or truncated or tick >= args.max_steps:
                break

        return {
            "front_image": ctx.raw_obs["pixels"]["front"],
            "wrist_image": ctx.raw_obs["pixels"]["wrist"],
            "tick": tick,
        }

    return execute_policy


def make_verify_state_node(env: SO101PenPickPlaceEnv, can_dof_addr: int, ctx: _EpisodeContext, args, classify_fn):
    def verify_state(state: SupervisorState) -> dict:
        if ctx.truncated:
            return {"status": "failed", "vlm_verdict": "(can fell off the table -- episode truncated, no VLM call)"}

        label, raw = classify_fn(state["front_image"], args.single_task, args.vlm_model)

        if env._is_success():
            raw_status = "success"
        elif label == "can_fallen":
            can_linvel = env.data.qvel[can_dof_addr : can_dof_addr + 3]
            speed = float(np.linalg.norm(can_linvel))
            if speed > args.can_settle_linvel_threshold:
                # Can is still visibly rolling/sliding -- defer recovery rather than
                # burn a grasp attempt on a target that will have moved by the time
                # the recovery policy's chunk finishes. See PLAN.md's "Can this catch
                # a can that's still rolling?" section.
                raw_status = "running"
            else:
                raw_status = "can_fallen"
        else:
            raw_status = "running"

        if raw_status == "can_fallen" and state["recovery_attempts"] >= args.max_recovery_attempts:
            status = "failed"
        elif state["tick"] >= args.max_steps:
            status = "failed"
        else:
            status = raw_status

        return {"status": status, "vlm_verdict": raw}

    return verify_state


def recover(state: SupervisorState) -> dict:
    stub_note = "" if state["recovery_attempts"] > 0 or state["active_policy"] == "recovery" else " (recovery policy: see --recovery-policy-path)"
    print(f"[recover] can_fallen detected -- recovery attempt {state['recovery_attempts'] + 1}{stub_note}")
    return {"recovery_attempts": state["recovery_attempts"] + 1, "active_policy": "recovery"}


def route_after_verify(state: SupervisorState) -> str:
    if state["status"] in ("success", "failed"):
        return "end"
    if state["status"] == "can_fallen":
        return "recover"
    return "execute_policy"


def build_graph(env: SO101PenPickPlaceEnv, policies: dict, ctx: _EpisodeContext, args, classify_fn=classify_scene):
    can_joint_id = env.model.body_jntadr[env._can_body_id]
    can_dof_addr = env.model.jnt_dofadr[can_joint_id]

    g = StateGraph(SupervisorState)
    g.add_node("execute_policy", make_execute_policy_node(env, policies, ctx, args))
    g.add_node("verify_state", make_verify_state_node(env, can_dof_addr, ctx, args, classify_fn))
    g.add_node("recover", recover)
    g.set_entry_point("execute_policy")
    g.add_edge("execute_policy", "verify_state")
    g.add_conditional_edges(
        "verify_state", route_after_verify, {"end": END, "recover": "recover", "execute_policy": "execute_policy"}
    )
    g.add_edge("recover", "execute_policy")
    return g.compile()


def initial_state() -> SupervisorState:
    return {
        "front_image": np.zeros((1, 1, 3), dtype=np.uint8),
        "wrist_image": np.zeros((1, 1, 3), dtype=np.uint8),
        "tick": 0,
        "active_policy": "base",
        "status": "running",
        "vlm_verdict": "",
        "recovery_attempts": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-policy-path", required=True)
    parser.add_argument("--recovery-policy-path", default=None, help="omit to run the stub (holds position) recover behavior")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--single-task", required=True)
    parser.add_argument("--control-dt", type=float, default=0.05, help="must match the recording rate the checkpoint(s) were trained at")
    parser.add_argument("--max-steps", type=int, default=600, help="per-episode control-tick cap, same concept as eval_policy.py's own")
    parser.add_argument("--max-recovery-attempts", type=int, default=3, help="guards against an infinite recover<->fail loop")
    parser.add_argument(
        "--can-settle-linvel-threshold",
        type=float,
        default=0.03,
        help="m/s -- below this the can is considered settled enough to commit a recovery attempt to it "
        "(placeholder default; tune during step 5's real eval)",
    )
    parser.add_argument("--n-action-steps-fallback", type=int, default=50, help="chunk length used only while active_policy=recovery has no checkpoint loaded (stub)")
    parser.add_argument("--vlm-model", default="gpt-4o-mini")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--force-fallen-spawn", action=argparse.BooleanOptionalAction, default=False, help="spawn the can already fallen every episode, to exercise the recovery path on demand")
    parser.add_argument("--graph-recursion-limit", type=int, default=500)
    args = parser.parse_args()

    device = torch.device(args.device)
    policies = {
        "base": load_policy_bundle(args.base_policy_path, device),
        "recovery": load_policy_bundle(args.recovery_policy_path, device) if args.recovery_policy_path else None,
    }

    env = SO101PenPickPlaceEnv(image_obs=True, randomize_can_pose=True, render_size=(480, 480), control_dt=args.control_dt)
    ctx = _EpisodeContext()
    graph = build_graph(env, policies, ctx, args)

    successes = 0
    results = []
    try:
        for episode in range(args.num_episodes):
            ctx.raw_obs, _ = env.reset(
                seed=args.seed + episode, options={"fallen": True} if args.force_fallen_spawn else None
            )
            ctx.last_active_policy = None
            ctx.stub_notice_shown = False
            ctx.truncated = False

            state = initial_state()
            state["front_image"] = ctx.raw_obs["pixels"]["front"]
            state["wrist_image"] = ctx.raw_obs["pixels"]["wrist"]

            result = graph.invoke(state, config={"recursion_limit": args.graph_recursion_limit})
            success = result["status"] == "success"
            successes += int(success)
            results.append(success)
            print(
                f"episode {episode + 1}/{args.num_episodes}: {'SUCCESS' if success else 'fail'} "
                f"(status={result['status']}, tick={result['tick']}, recovery_attempts={result['recovery_attempts']})"
            )
    finally:
        env.close()

    n_run = len(results)
    rate = successes / n_run if n_run else 0.0
    print(f"\nSuccess rate: {successes}/{n_run} = {rate:.1%}")


if __name__ == "__main__":
    main()
