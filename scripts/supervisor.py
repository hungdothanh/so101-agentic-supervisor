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
from lerobot.utils.rerun_visualization import init_rerun, log_rerun_data, shutdown_rerun

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
    # tick value at which the CURRENT attempt began (the initial base-policy run
    # counts as "attempt 0"; each can_fallen event that is genuinely new -- i.e. the
    # can wasn't already flagged can_fallen the previous poll -- starts a fresh
    # attempt and resets this). Per-attempt time budgets are measured relative to
    # this, not to the episode's absolute tick count.
    attempt_start_tick: int
    # True only on the single verify_state call where can_fallen is detected for the
    # first time since the last non-can_fallen reading (a real new failure event, as
    # opposed to a later poll simply confirming the same ongoing failure).
    is_new_fall_event: bool
    # The raw perceptual label from the PREVIOUS verify_state call (can_standing /
    # can_fallen / can_in_gripper / success) -- distinct from `status`, which collapses
    # can_standing and can_in_gripper into the same "running" value. is_new_fall_event
    # needs this finer-grained history to tell "still fumbling with an already-fallen
    # can" (last_label stays can_fallen, not a new event) apart from "was genuinely
    # holding it, then dropped it" (last_label was can_in_gripper -- a real new event).
    last_label: Literal["can_standing", "can_fallen", "can_in_gripper", "can_off_table", "success"]


# ---------------------------------------------------------------------------
# VLM classification (isolated so it can be exercised standalone, per PLAN.md's
# own step-2 verification bullet: "manually feed a handful of front-camera frames
# to the GPT-4o call directly and confirm the classification prompt returns the
# right label").
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """You are monitoring a robot arm pick-and-place task in simulation. \
The task is: "{task}".

Look at the front-camera image and classify the scene into EXACTLY ONE of these five \
labels:
- success: the can is placed inside the bin.
- can_in_gripper: the gripper's fingers are CLOSED and PINCHED around the can's body, \
visibly compressing against both sides of it, such that the can would move rigidly \
with the gripper if the arm moved. Look specifically at the gap between the finger \
tips and the can's surface: if there is daylight/space between them, or the fingers \
are open/spread apart, or the gripper is simply positioned above/beside/near the can \
without touching and clamping it, this is NOT can_in_gripper -- classify the can's own \
resting state instead (can_fallen or can_standing below). When in doubt, do NOT pick \
can_in_gripper; only choose it when the grasp is visually unambiguous.
- can_fallen: the can is lying on its side, resting on the table (not in the bin) and \
does NOT meet the can_in_gripper bar above -- this includes the arm reaching toward \
it, touching it, resting against it, or nudging it, as long as the gripper is not \
visibly clamped around it.
- can_standing: the can is upright, resting on the table, and does NOT meet the \
can_in_gripper bar above.
- can_off_table: the can is not visible anywhere in the image -- not on the table, not \
in the gripper, not in the bin. It has fallen or rolled off the table/out of the \
workspace and is unreachable. Only choose this if you're confident the can has \
genuinely left the scene; if the can might simply be hidden behind the arm or gripper \
from this camera angle, look for any partial glimpse of its body or color before \
concluding it's gone, and prefer can_fallen/can_standing/can_in_gripper if there's any \
doubt.

Respond with exactly one word: success, can_in_gripper, can_fallen, can_standing, or \
can_off_table. No other text."""


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
    """Returns (label, raw_response_text). label is one of
    can_standing/can_fallen/can_in_gripper/can_off_table/success, defaulting to
    "can_standing" (the safest fallback -- keeps the base policy going, doesn't claim a
    grasp that may not exist, and doesn't prematurely end the episode) if the response
    doesn't parse cleanly."""
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
    for label in ("success", "can_off_table", "can_in_gripper", "can_fallen", "can_standing"):
        if label in lowered:
            return label, raw
    return "can_standing", raw


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
        # Each checkpoint is language-conditioned on its OWN training task string --
        # the base and recovery policies were fine-tuned on different strings ("pick
        # up the can..." vs "pick up the fallen can..."), so feeding the wrong one to
        # whichever policy is active would be a real train/eval mismatch, not just a
        # cosmetic label.
        task = args.single_task if state["active_policy"] == "base" else args.recovery_single_task

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
                        task=task,
                        robot_type="so101_mujoco",
                    )
                action_deg_pct = action.squeeze(0).cpu().numpy()
                qpos_rad = ctx.raw_obs["agent_pos"][: len(ARM_JOINTS)]
                env_action = action_to_env_delta(action_deg_pct, qpos_rad, env)

            ctx.raw_obs, _reward, _terminated, truncated, info = env.step(env_action)
            ctx.truncated = bool(truncated)
            tick += 1

            if args.display_data:
                log_rerun_data(
                    observation={"front": ctx.raw_obs["pixels"]["front"], "wrist": ctx.raw_obs["pixels"]["wrist"]},
                    action={"env_delta": env_action},
                )

            if info["succeed"] or truncated or tick >= args.max_total_steps:
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
            physical_label = "success"
            still_sliding = False
        elif label == "can_fallen":
            physical_label = "can_fallen"
            can_linvel = env.data.qvel[can_dof_addr : can_dof_addr + 3]
            still_sliding = float(np.linalg.norm(can_linvel)) > args.can_settle_linvel_threshold
        elif label == "success":
            # The VLM itself guessed "success", but the ground-truth check
            # (env._is_success(), the same simulator-based signal eval_policy.py's
            # info["succeed"] uses) disagrees. Do NOT trust the VLM's guess here --
            # a frame where the can is being held directly above/at the bin's rim,
            # not yet released, is easy to visually mistake for "done", especially
            # since the classify prompt has no explicit label for that in-between
            # moment. Fall back to the physically-conservative reading: the can is
            # still held (if the episode really is finishing, env._is_success() will
            # correctly flip True on a very soon following tick once it's released).
            # print(
            #     f"[verify_state] tick={state['tick']} VLM guessed 'success' but "
            #     "env._is_success() (ground truth) says False -- overriding to "
            #     "can_in_gripper, not ending the episode on an unconfirmed guess"
            # )
            physical_label = "can_in_gripper"
            still_sliding = False
        else:
            physical_label = label  # "can_standing", "can_in_gripper", or "can_off_table"
            still_sliding = False

        if physical_label == "success":
            raw_status = "success"
        elif physical_label == "can_off_table":
            # Can has left the table/workspace entirely -- nothing the base or recovery
            # policy does can ever complete the task from here. End immediately rather
            # than burn the rest of the episode's tick budget on an unreachable goal.
            raw_status = "failed"
        elif physical_label == "can_fallen" and not still_sliding:
            raw_status = "can_fallen"
        else:
            # can_standing, can_in_gripper, or a still-sliding can_fallen we're
            # deliberately not acting on yet -- all just "task in progress" for routing.
            # See PLAN.md's "Can this catch a can that's still rolling?" section for why
            # a sliding can_fallen is deferred rather than committed to immediately.
            raw_status = "running"

        # A new fall event is only a can_fallen reading that follows either (a) the
        # can having genuinely been held (last_label == "can_in_gripper") and now
        # dropped -- a real, distinct failure -- or (b) this being the very first time
        # it's happened, while still under base-policy control. Plain "the can is
        # still lying there, arm hasn't managed to grasp it yet" -- i.e. can_fallen
        # following can_fallen, or can_fallen following can_standing while ALREADY in
        # recovery (a VLM flicker mid-grasp-attempt, not a real state change) -- is
        # explicitly NOT a new event, so it doesn't reset the per-attempt clock or
        # burn one of --max-recovery-attempts.
        is_new_fall_event = raw_status == "can_fallen" and (
            state["active_policy"] == "base" or state["last_label"] == "can_in_gripper"
        )
        ticks_this_attempt = state["tick"] - state["attempt_start_tick"]

        if raw_status in ("success", "failed"):
            # Terminal already (success, or can_off_table above) -- nothing about
            # attempt/tick bookkeeping can or should override that.
            status = raw_status
        elif is_new_fall_event and state["recovery_attempts"] >= args.max_recovery_attempts:
            # This would be yet another fresh attempt, but we've already used up the
            # ones we're willing to give it -- the arm keeps failing to hold onto the
            # can, so stop rather than reset the clock again.
            status = "failed"
        elif not is_new_fall_event and raw_status != "success" and ticks_this_attempt >= args.max_steps_per_attempt:
            # This SAME attempt (no fresh fall event since it started) has burned its
            # whole budget without succeeding -- give up rather than let it run forever
            # on a stall the state machine can't otherwise detect.
            status = "failed"
        elif state["tick"] >= args.max_total_steps:
            # Hard episode-wide safety net, independent of attempt bookkeeping.
            status = "failed"
        else:
            status = raw_status

        print(
            f"[verify_state] tick={state['tick']}"
            + (f" (attempt_tick={ticks_this_attempt})" if is_new_fall_event else "")
            + f" vlm_label={physical_label!r} -> status={status}"
            + (" [NEW FALL EVENT]" if is_new_fall_event else "")
            + (" [CAN OFF TABLE -- unreachable, ending episode]" if physical_label == "can_off_table" else "")
        )
        if args.display_data:
            import rerun as rr

            rr.log("vlm/verdict", rr.TextLog(f"tick={state['tick']} verdict={raw!r} -> status={status}"))
        return {"status": status, "vlm_verdict": raw, "is_new_fall_event": is_new_fall_event, "last_label": physical_label}

    return verify_state


def make_recover_node(args):
    def recover(state: SupervisorState) -> dict:
        attempt_num = state["recovery_attempts"] + 1
        reason = "initial knock-over" if state["active_policy"] == "base" else "dropped after being grasped"
        print(
            f"[recover] new can_fallen event ({reason}) -- recovery attempt {attempt_num}; "
            # f"resetting per-attempt clock (tick={state['tick']} -> fresh {args.max_steps_per_attempt}-tick budget)"
        )
        return {
            "recovery_attempts": attempt_num,
            "active_policy": "recovery",
            "attempt_start_tick": state["tick"],
        }

    return recover


def route_after_verify(state: SupervisorState) -> str:
    if state["status"] in ("success", "failed"):
        return "end"
    if state["status"] == "can_fallen":
        # Only detour through `recover` (which bumps recovery_attempts, resets the
        # per-attempt clock, and does the base->recovery policy switch) on a genuinely
        # NEW fall event. A later poll that still reads can_fallen because the SAME
        # attempt hasn't finished yet (SmolVLA's chunk is only n_action_steps=~50
        # ticks; picking a fallen can up and placing it typically needs several
        # chunks strung together, same as the base policy would) should just keep
        # running -- not re-count as a fresh attempt or reset the clock again.
        if state["is_new_fall_event"]:
            return "recover"
        return "execute_policy"
    return "execute_policy"


def build_graph(env: SO101PenPickPlaceEnv, policies: dict, ctx: _EpisodeContext, args, classify_fn=classify_scene):
    can_joint_id = env.model.body_jntadr[env._can_body_id]
    can_dof_addr = env.model.jnt_dofadr[can_joint_id]

    g = StateGraph(SupervisorState)
    g.add_node("execute_policy", make_execute_policy_node(env, policies, ctx, args))
    g.add_node("verify_state", make_verify_state_node(env, can_dof_addr, ctx, args, classify_fn))
    g.add_node("recover", make_recover_node(args))
    g.set_entry_point("execute_policy")
    g.add_edge("execute_policy", "verify_state")
    g.add_conditional_edges(
        "verify_state", route_after_verify, {"end": END, "recover": "recover", "execute_policy": "execute_policy"}
    )
    g.add_edge("recover", "execute_policy")
    return g.compile()


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% (default z) Wilson score interval -- same reporting discipline PLAN.md's
    verification section calls for and this project's README already used for the
    standing-can baseline/mimicgen comparison, now computed inline rather than by hand."""
    p_hat = successes / n
    denom = 1.0 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def initial_state() -> SupervisorState:
    return {
        "front_image": np.zeros((1, 1, 3), dtype=np.uint8),
        "wrist_image": np.zeros((1, 1, 3), dtype=np.uint8),
        "tick": 0,
        "active_policy": "base",
        "status": "running",
        "vlm_verdict": "",
        "recovery_attempts": 0,
        "attempt_start_tick": 0,
        "is_new_fall_event": False,
        "last_label": "can_standing",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-policy-path", required=True)
    parser.add_argument("--recovery-policy-path", default=None, help="omit to run the stub (holds position) recover behavior")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--single-task", required=True, help="task string the BASE checkpoint was fine-tuned on")
    parser.add_argument(
        "--recovery-single-task",
        default=None,
        help="task string the RECOVERY checkpoint was fine-tuned on (e.g. 'pick up the fallen can and place "
        "it in the bin') -- defaults to --single-task if omitted (fine for the stub recover behavior, wrong "
        "for a real recovery checkpoint trained on a different task string)",
    )
    parser.add_argument("--control-dt", type=float, default=0.05, help="must match the recording rate the checkpoint(s) were trained at")
    parser.add_argument(
        "--max-steps-per-attempt",
        type=int,
        default=600,
        help="tick budget given to EACH attempt (the initial base-policy run, and each "
        "fresh recovery attempt after a can_fallen event) -- same concept as "
        "eval_policy.py's own --max-steps, but reset every time a genuinely NEW "
        "can_fallen event fires rather than shared across the whole episode. 600 "
        "comfortably covers a single ~300-500-tick pick-and-place (see README).",
    )
    parser.add_argument(
        "--max-total-steps",
        type=int,
        default=None,
        help="hard episode-wide tick ceiling, independent of attempt bookkeeping -- purely "
        "a safety net against a runaway episode (e.g. the state machine oscillating "
        "can_fallen<->running without ever resolving). Defaults to "
        "--max-steps-per-attempt * (--max-recovery-attempts + 1) if not given.",
    )
    parser.add_argument(
        "--max-recovery-attempts",
        type=int,
        default=3,
        help="guards against an infinite recover<->fail loop. Counts DISTINCT recovery "
        "episodes (i.e. a genuinely NEW can_fallen event: the initial base->fallen "
        "transition, or a re-fall after the can was airborne/being carried), not "
        "individual VLM polls/chunks -- each such event resets the per-attempt clock "
        "(--max-steps-per-attempt) so the recovery policy gets a fresh full budget.",
    )
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
    parser.add_argument(
        "--display-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="stream camera feeds + action + VLM verdict to a live Rerun viewer window (requires "
        "rerun-sdk, same tool record_dataset.py's --display-data references) -- opens its own window, "
        "separate from this terminal's output",
    )
    args = parser.parse_args()
    if args.recovery_single_task is None:
        args.recovery_single_task = args.single_task
    if args.max_total_steps is None:
        args.max_total_steps = args.max_steps_per_attempt * (args.max_recovery_attempts + 1)

    device = torch.device(args.device)
    policies = {
        "base": load_policy_bundle(args.base_policy_path, device),
        "recovery": load_policy_bundle(args.recovery_policy_path, device) if args.recovery_policy_path else None,
    }

    env = SO101PenPickPlaceEnv(image_obs=True, randomize_can_pose=True, render_size=(480, 480), control_dt=args.control_dt)
    ctx = _EpisodeContext()
    graph = build_graph(env, policies, ctx, args)

    if args.display_data:
        init_rerun(session_name="so101_supervisor")

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
                f"EPISODE {episode + 1}/{args.num_episodes}: {'SUCCESS' if success else 'fail'} "
                f"(status={result['status']}, tick={result['tick']}, recovery_attempts={result['recovery_attempts']}) \n"
            )
    finally:
        env.close()
        if args.display_data:
            shutdown_rerun()

    n_run = len(results)
    rate = successes / n_run if n_run else 0.0
    print(f"\nSuccess rate: {successes}/{n_run} = {rate:.1%}")
    if n_run:
        lo, hi = wilson_ci(successes, n_run)
        print(f"95% CI (Wilson): [{lo:.1%}, {hi:.1%}]")


if __name__ == "__main__":
    main()