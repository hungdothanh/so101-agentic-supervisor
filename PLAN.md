# SO-101 Agentic VLA: Hierarchical VLM-Guided Failure Detection and Policy Recovery

This document was written during a planning conversation in the `so101_mujoco_sim2real` repo
and is meant to travel with this project directory as the source of truth going forward — a
fresh Claude Code session opened here should read this file first, before touching any code.

## Context

The sibling project `~/robotics_ws/so101_mujoco_sim2real` built a MuJoCo SO-101 pick-and-place
pipeline (env, gamepad teleop, LeRobot `Robot`/`Teleoperator` bridge, SmolVLA fine-tuning) and
used MimicGen-style dataset augmentation to raise sim-eval success from 39.0% to 61.0%. Its
failure-mode breakdown showed the remaining ~18-21% "catastrophic" failures (the arm clips the
can mid-grasp, knocking it from standing to lying on its side) didn't move — expected, since
MimicGen adds *position coverage*, not *recovery behavior*, and SmolVLA's `n_obs_steps=1` gives
it no temporal memory to notice and correct a bad grasp mid-attempt. That project is
sim2real-gap-closing scoped and stays as-is, on hold pending real hardware arrival.

This project has a different, narrower objective — not sim2real, just task performance — so it
gets its own repo: wrap the policy in a **LangGraph supervisor** that (1) periodically snapshots
the camera feed, (2) asks a VLM whether the can is still standing, fallen, or successfully
placed, and (3) if fallen, routes to a dedicated recovery path instead of letting the base policy
continue on an out-of-distribution state it was never trained to handle. Still sim-only for now,
same as the sibling project.

**Everything not listed below (the simulation, environment, robot bridge, and SmolVLA pipeline
itself) was copied over from `so101_mujoco_sim2real` as-is and works exactly as documented in
that repo's own README.md/PLAN.md** — read those for background on the underlying stack (gripper
design, control interface, camera setup, dataset schema, etc.) rather than re-deriving it here.

## Decision: Direction B (multi-policy routing), not Direction A (perception+geometry)

Two directions were on the table:
- **Direction A**: VLM flags `can_fallen` → a perception tool (GroundingDINO/YOLO) extracts the
  fallen can's bounding box/pose → an LLM computes explicit XYZ + wrist-pitch waypoints → a
  custom motion script drives the arm there via IK.
- **Direction B**: VLM flags `can_fallen` → LangGraph swaps in a second SmolVLA checkpoint
  fine-tuned specifically on picking up fallen cans → that policy runs the recovery end-to-end
  from raw camera frames.

**Decision: Direction B.** Three reasons, all grounded in what the sibling project already
learned the hard way:

1. **The sibling project already ran the Direction-A playbook once, on the easier problem, and
   it scored ~0%.** `scripts/grasp_trigger_test.py` is exactly "hand-designed geometric waypoints
   + IK" for the *standing*-can grasp (the easy case) — verified fresh at 0% success over 30
   episodes: the fixed-palm gripper clips the can before the jaw closes, in 14/15 instrumented
   episodes. Direction A proposes the same class of solution (hand-coded geometry + IK) for the
   *harder* case (a fallen can, different approach angle, same single-jaw-vs-fixed-palm gripper)
   with no new evidence it would fare better. Direction B is the same class of solution (learned
   policy from demonstrations) that already *did* work for the standing-can case — twice (the
   50-episode baseline, then MimicGen).
2. **The existing IK utility (`so101_mujoco_env/ik_utils.py`'s `FingertipIK`) only controls
   *position* (3D XYZ via 4 joints), not orientation.** Direction A's "set wrist pitch to -90°"
   needs full 6-DOF pose control that doesn't exist in this codebase yet — real new IK
   engineering, not a reuse of what's there. (Same class of gap bit the MimicGen work too:
   `FingertipIK` never controlled `wrist_roll` either, and fixing that was one of three bugs
   needed to get MimicGen working at all.)
3. **Direction B reuses a fully-built, proven pipeline almost as-is.** `scripts/mimicgen_augment.py`
   already does "record a few human reference demos → segment/retarget → verify via real physics
   → build a `LeRobotDataset` → fine-tune SmolVLA" for the standing-can task. The same recipe
   applies directly to "fallen can → bin": record a handful of reference demos of picking up a
   *fallen* can, reuse the existing segment/retarget/build/fine-tune machinery, get a second
   checkpoint. Cheap because it's not new engineering, just a new dataset.

One piece of Direction A's idea is still worth keeping: the VLM call itself is a lightweight
image→classification step, not the "extract precise coordinates for math" step — that's shared
infrastructure regardless of which direction was picked, and doesn't need a detector/bounding-box
model at all since SmolVLA (Direction B) consumes raw images directly, not extracted coordinates.

## Can this catch a can that's still rolling, not just one at rest?

Measured directly in the sibling project's sim before writing this plan: a fairly hard simulated
clip (0.9 m/s lateral + 8 rad/s spin, applied via `env.data.qvel` on the can's freejoint) sends
the can rolling/sliding for **~3.5s**, traveling **~7.5cm**, fully settled by **~6s** — bounded,
not "rolls off across the table."

**Not natively in real time, and this is architectural, not a bug to fix**: SmolVLA has no
closed-loop tracking — each inference call looks at one frame and outputs a whole
2.5-simulated-second action chunk (`n_action_steps=50` @ 20Hz) played back open-loop with zero
re-observation until it's exhausted (confirmed in `select_action()`'s action-queue logic in
LeRobot's `modeling_smolvla.py`). The supervisor's own per-chunk check cadence inherits that same
blind window. So a grasp attempt fired while the can is still moving targets a stale position.

**Why the plan mostly self-corrects anyway**: the can's ~3.5-6s settling time is roughly the same
order of magnitude as 1-2 chunk cycles. The `recover → execute_policy → verify_state` retry loop
below means a missed/stale attempt just retries 2.5s later with a fresh observation —
functionally "waiting it out," even though it isn't explicitly designed as one.

**Three cheap hardening additions, build in from the start (see "What has to be built," step 2):**
1. In sim, gate `recover` on the can's real velocity (`env.data.qvel`, free ground truth) before
   committing a full grasp-attempt chunk — avoids wasting attempts on an obviously-still-moving
   target. Sim-only trick, won't transfer to real hardware.
2. For something that generalizes to real hardware later: a cheap local frame-diff "has the
   scene stopped changing" check between consecutive camera frames as a fast pre-gate, before
   spending a GPT-4o call + a full grasp attempt.
3. Record recovery reference demos across a spread of rolled distances/orientations (informed by
   the ~5-15cm range measured above), not just "can tipped in place at its original spot" — so
   the recovery policy's training distribution matches what actually happens.

(Longer-term, if genuine real-time tracking ever becomes a hard requirement: `lerobot-rollout`
already exposes `--inference.type=rtc`, which pipelines chunk computation with execution rather
than blocking — not traced through in detail, flagged here as the lever to pull if the above
turns out not to be enough.)

## Architecture

### LangGraph state machine

```python
class SupervisorState(TypedDict):
    front_image: np.ndarray          # latest front-camera frame
    wrist_image: np.ndarray          # latest wrist-camera frame
    tick: int                        # control ticks elapsed this episode
    active_policy: Literal["base", "recovery"]
    status: Literal["running", "success", "can_fallen", "failed"]
    vlm_verdict: str                 # raw/parsed VLM response, for logging
    recovery_attempts: int           # guards against infinite recover→fail→recover loops
```

Nodes:
- **`execute_policy`**: runs `active_policy`'s SmolVLA checkpoint for one action-chunk's worth
  of ticks against the sim env (reuses `eval_policy.py`'s `build_observation()`/
  `action_to_env_delta()` conversion pattern), updates `front_image`/`wrist_image`/`tick`.
- **`verify_state`**: sends the latest front-camera frame to GPT-4o (via `langchain-openai`)
  with a prompt asking it to classify `success` / `can_fallen` / `running`; also checks the
  env's own `_is_success()` ground truth in sim (free, since this is sim-only for now — the VLM
  path is what will matter once real hardware arrives and ground truth isn't free anymore, but
  it's worth exercising it now so the graph doesn't need re-plumbing later).
- **`recover`**: swaps `active_policy` to `"recovery"`, increments `recovery_attempts`, loops
  back into `execute_policy`.

Edges (conditional on `status`):
- `execute_policy → verify_state` (always, after each chunk)
- `verify_state → END(success)` if `status == "success"`
- `verify_state → recover` if `status == "can_fallen"` and `recovery_attempts < MAX_RECOVERY_ATTEMPTS`
- `verify_state → END(failed)` if `status == "can_fallen"` and attempts exhausted, or `tick`
  exceeds a hard cap (mirrors `eval_policy.py`'s existing `--max-steps` timeout concept)
- `verify_state → execute_policy` if `status == "running"` (task still in progress, keep going)
- `recover → execute_policy`

**Check cadence**: call `verify_state` once per action-chunk (~2.5 sim-seconds at SmolVLA's
`n_action_steps=50`/20Hz), not every tick — matches "after the base policy attempts a grasp,"
and keeps the GPT-4o call count low (cost/latency) since it's not gating the 20Hz control loop
itself, only the higher-level chunk boundary.

### Integration point: a custom loop, not a `RolloutStrategy` subclass

Checked LeRobot's existing rollout strategies (`base`, `sentry`, `highlight`, `episodic`,
`dagger`) — none of them do VLM-triggered autonomous failure detection/recovery (`sentry` is
continuous recording+auto-upload, `dagger` is human-in-the-loop correction via keyboard/pedal).
No existing strategy to extend; a custom supervisor is genuinely the right shape.

Build the supervisor wrapping the **simple custom-loop style already proven in
`scripts/eval_policy.py`**, not a new `lerobot.rollout.strategies.RolloutStrategy` subclass:
`eval_policy.py` already shows exactly what a from-scratch policy↔env loop looks like for this
robot (unit conversions, `SmolVLAPolicy.from_pretrained()`, chunked action prediction), and it's
much less machinery to fight than LeRobot's strategy/config ABC for what's still a research
prototype. If this later needs to run live against real hardware through `lerobot-rollout`, it'd
be worth revisiting as a proper `RolloutStrategy` then — not blocking now.

**Both checkpoints loaded once at startup**, not dynamically loaded/unloaded per recovery
trigger — SmolVLA is only ~450M params, so keeping both in memory simultaneously avoids reload
latency.

## What has to be built (in order)

1. **New capability: spawn the can fallen (lying on its side), not just standing.** Currently
   `so101_mujoco_env/pen_pickplace_env.py`'s `reset()` hardcodes `CAN_UPRIGHT_QUAT` for every
   spawn path. Needed both to test the supervisor end-to-end in sim (trigger the failure state
   on demand) and to record recovery reference demos. Small, additive change (new
   `options={"can_xy": ..., "fallen": True}` path or similar), same pattern as the existing
   `can_xy` override.
2. **Supervisor skeleton with a stub recovery node first.** Build the LangGraph graph +
   `verify_state`'s GPT-4o call + the routing edges, with `recover` initially just a no-op/print
   (no real recovery policy yet). Include the sim-only velocity-gate check (`env.data.qvel` on
   the can, see "Can this catch a can that's still rolling?" above) at this stage too — cheap to
   add now, and lets step 5's end-to-end eval measure its effect from the start rather than
   retrofitting it later. Validates the state machine, the VLM prompt/parsing, and the
   integration loop in isolation from the harder "train a whole new policy" work.
3. **Record fallen-can reference demos** (gamepad, same as the sibling project's original 9
   references) — picking up a can lying on the table and placing it in the bin, at a handful of
   deliberately spread positions/orientations *and rolled distances* (use the impulse-based
   knock from this plan's verification step, not just a manually-placed fallen can, so the
   reference set covers the ~5-15cm rolled range actually observed rather than only an idealized
   in-place tip), using the fallen-spawn capability from step 1.
4. **Reuse `scripts/mimicgen_augment.py`'s pipeline** to segment/retarget/verify/build a recovery
   dataset, then fine-tune a second SmolVLA checkpoint on it — identical recipe to the sibling
   project's MimicGen work, new source demos only.
5. **Wire the real recovery checkpoint into the `recover` node**, replacing the stub. End-to-end
   eval: measure how often the supervisor catches a knocked-over can and how often the recovery
   checkpoint actually succeeds from that state.

## New dependencies

`langgraph`, `langchain`, `langchain-openai`, and an `OPENAI_API_KEY` (loaded from a local `.env`
file — already gitignored, see `.gitignore`; never commit the key itself). Nothing else in the
inherited stack changes.

## Verification

- **Already done, during planning**: knocked the can with a lateral-velocity + spin impulse
  directly via `env.data.qvel` and traced its position/speed for 8 simulated seconds — confirmed
  it travels a bounded ~7.5cm, stays actively moving for ~3.5s, and is fully settled by ~6s (see
  "Can this catch a can that's still rolling?" above). This is what the settling-time and
  rolled-distance numbers referenced throughout this plan are based on, not estimates.
- Step 1: reset with the new fallen-spawn option; confirm via `env.data.xpos`/`xquat` the can
  actually lands on its side, not standing, and that the existing standing-spawn path is
  unaffected when the option is omitted.
- Step 2: force `status="can_fallen"` synthetically (e.g. temporarily spawn the can fallen and
  skip straight to `verify_state`) and confirm the graph routes to `recover` and back, with
  `recovery_attempts` incrementing and the hard-cap edge firing correctly when exhausted.
- Step 2 (VLM): manually feed a handful of front-camera frames (standing can, fallen can, can
  correctly in bin) to the GPT-4o call directly and confirm the classification prompt returns
  the right label on each before trusting it inside the loop.
- Step 4: same reference-demo verification discipline as the sibling project's MimicGen work —
  replay every recorded reference through real physics and confirm genuine success before
  building anything on top of it (that project learned, the hard way, that a bad reference
  poisons everything downstream).
- Step 5: end-to-end sim run — deliberately induce a knock-over (or use the fallen-spawn option
  as a stand-in), confirm the full graph detects it, recovers, and reaches `_is_success()`, and
  report a recovery success rate over N triggered episodes the same way `eval_policy.py` already
  reports baseline success rate (Wilson CI, same discipline as the sibling project's eval).

## Setup note

This repo was seeded by copying `so101_mujoco_sim2real`'s `assets/`, `so101_mujoco_env/`,
`lerobot_bridge/`, `scripts/`, `notebooks/`, and `docs/` as-is (verified byte-identical under
`assets/`). Follow that repo's own README.md for environment setup (conda env, LeRobot install
at the pinned commit, `pip install -e lerobot_bridge`) — it's unchanged here. Additionally
install this project's new dependencies (above) once you're ready to start on the supervisor.
