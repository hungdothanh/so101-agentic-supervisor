#!/usr/bin/env python3
"""Standalone routing test for scripts/supervisor.py's LangGraph state machine --
PLAN.md's own step-2 verification bullet: "force status='can_fallen' synthetically...
confirm the graph routes to recover and back, with recovery_attempts incrementing and
the hard-cap edge firing correctly when exhausted."

No pytest infra in this repo (matches its existing script-based convention, see
eval_policy.py/mimicgen_augment.py) -- this is a plain script with print-and-assert
checks, runnable standalone. Needs NO OpenAI API key: classify_scene() is swapped for
a scripted fake classifier, and both policy bundles are left as None (stub/hold-
position) so no SmolVLA checkpoint needs loading either -- this exercises the real
graph wiring and the real env (physics, _is_success(), qvel) with only the VLM call
faked out.

Run inside the `lerobot` conda env, from the repo root:
    python scripts/test_supervisor_graph.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.supervisor import _EpisodeContext, build_graph, initial_state  # noqa: E402
from so101_mujoco_env.pen_pickplace_env import BIN_CENTER_XY, SO101PenPickPlaceEnv  # noqa: E402

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" -- {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def make_scripted_classifier(labels: list[str]):
    """Pops one label per call; repeats the last one once exhausted."""
    calls = {"i": 0}

    def classify_fn(image, task, model):
        label = labels[min(calls["i"], len(labels) - 1)]
        calls["i"] += 1
        return label, f"(scripted call #{calls['i']}: {label})"

    return classify_fn


def make_args(**overrides) -> SimpleNamespace:
    defaults = dict(
        single_task="pick up the can and place it in the bin",
        recovery_single_task="pick up the fallen can and place it in the bin",
        display_data=False,
        max_steps=30,
        max_recovery_attempts=3,
        can_settle_linvel_threshold=0.03,
        n_action_steps_fallback=5,  # short chunks -> fast test cycles
        vlm_model="unused-in-this-test",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_recovery_attempts_exhausted() -> None:
    """Scripted classifier always says can_fallen; can spawns far from the bin so
    ground truth never overrides to success. Expect: exactly max_recovery_attempts
    recover cycles, then status="failed"."""
    env = SO101PenPickPlaceEnv(image_obs=True, randomize_can_pose=False)
    ctx = _EpisodeContext()
    args = make_args()
    classify_fn = make_scripted_classifier(["can_fallen"])
    graph = build_graph(env, {"base": None, "recovery": None}, ctx, args, classify_fn=classify_fn)

    raw_obs, _ = env.reset(options={"can_xy": (0.2, -0.1)})  # nowhere near BIN_CENTER_XY
    ctx.raw_obs = raw_obs
    state = initial_state()
    state["front_image"] = raw_obs["pixels"]["front"]
    state["wrist_image"] = raw_obs["pixels"]["wrist"]

    result = graph.invoke(state, config={"recursion_limit": 200})
    env.close()

    check("recovery-exhausted: ends failed", result["status"] == "failed", f"got {result['status']}")
    check(
        "recovery-exhausted: recovery_attempts == max_recovery_attempts",
        result["recovery_attempts"] == args.max_recovery_attempts,
        f"got {result['recovery_attempts']}",
    )
    check("recovery-exhausted: active_policy swapped to recovery", result["active_policy"] == "recovery")


def test_ground_truth_success_overrides_vlm() -> None:
    """Can spawns already settled in the bin -> env._is_success() is True from the
    very first verify_state call. Scripted classifier deliberately says can_fallen
    throughout, to confirm ground truth wins over the VLM verdict (per PLAN.md:
    verify_state "also checks the env's own _is_success() ground truth in sim")."""
    env = SO101PenPickPlaceEnv(image_obs=True, randomize_can_pose=False)
    ctx = _EpisodeContext()
    args = make_args()
    classify_fn = make_scripted_classifier(["can_fallen"])
    graph = build_graph(env, {"base": None, "recovery": None}, ctx, args, classify_fn=classify_fn)

    raw_obs, _ = env.reset(options={"can_xy": tuple(BIN_CENTER_XY)})
    ctx.raw_obs = raw_obs
    assert env._is_success(), "test setup bug: can_xy=BIN_CENTER_XY should already satisfy _is_success()"
    state = initial_state()
    state["front_image"] = raw_obs["pixels"]["front"]
    state["wrist_image"] = raw_obs["pixels"]["wrist"]

    result = graph.invoke(state, config={"recursion_limit": 200})
    env.close()

    check("ground-truth-success: ends success despite VLM saying can_fallen", result["status"] == "success", f"got {result['status']}")
    check("ground-truth-success: no recovery attempts spent", result["recovery_attempts"] == 0, f"got {result['recovery_attempts']}")


def test_tick_cap_timeout() -> None:
    """Scripted classifier always says running, can never reaches the bin (stub
    policies hold position) -> should time out via the tick cap, not the
    recovery-attempts cap, ending status="failed" with recovery_attempts still 0."""
    env = SO101PenPickPlaceEnv(image_obs=True, randomize_can_pose=False)
    ctx = _EpisodeContext()
    args = make_args(max_steps=12, n_action_steps_fallback=5)  # a couple short chunks -> hits cap fast
    classify_fn = make_scripted_classifier(["running"])
    graph = build_graph(env, {"base": None, "recovery": None}, ctx, args, classify_fn=classify_fn)

    raw_obs, _ = env.reset(options={"can_xy": (0.2, -0.1)})
    ctx.raw_obs = raw_obs
    state = initial_state()
    state["front_image"] = raw_obs["pixels"]["front"]
    state["wrist_image"] = raw_obs["pixels"]["wrist"]

    result = graph.invoke(state, config={"recursion_limit": 200})
    env.close()

    check("tick-cap: ends failed via timeout", result["status"] == "failed", f"got {result['status']}")
    check("tick-cap: tick reached the cap", result["tick"] >= args.max_steps, f"got {result['tick']}")
    check("tick-cap: no recovery attempts spent", result["recovery_attempts"] == 0, f"got {result['recovery_attempts']}")


if __name__ == "__main__":
    test_recovery_attempts_exhausted()
    test_ground_truth_success_overrides_vlm()
    test_tick_cap_timeout()

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) FAILED: {FAILURES}")
        sys.exit(1)
    print("All checks passed.")
