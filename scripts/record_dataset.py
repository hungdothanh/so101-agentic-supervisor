#!/usr/bin/env python3
"""Phase 5 (and, later, Phase 6): record a SO-101 MuJoCo pick-and-place dataset via
gamepad teleop, with the can's spawn pose automatically re-randomized between episodes.

WHY THIS SCRIPT EXISTS INSTEAD OF PLAIN `lerobot-record`: stock lerobot-record's
`--dataset.reset_time_s` window is built for a human to reposition a REAL object by
hand between episodes -- it just runs the teleop loop unrecorded for that many
seconds, it has no generic "reset the environment" concept of its own (there's nothing
Robot-ABC-level to call). This MuJoCo twin CAN reset itself instantly and correctly
(see lerobot_bridge/.../so101_mujoco.py's `reset_scene()`), so this driver wraps
lerobot's own `record_loop()`/`LeRobotDataset` machinery -- the exact same pieces
lerobot-record itself uses -- and swaps that unrecorded human-reset window for an
instant `robot.reset_scene(randomize_can_pose=...)` + `teleop.reset_target()` call
(the latter is required too: the teleoperator tracks its own absolute joint target
internally, and has no other way to learn the robot's state just jumped -- see
so101_mujoco_teleop.py's `reset_target()` docstring for why skipping it would make the
next episode's first few ticks fight the reset).

Run inside the `lerobot` conda env, from the repo root. With --viewer on (the default)
you need PYGLFW_LIBRARY_VARIANT=x11 -- same env var, same reason, as the
gym_hil_desk_cleanup Franka scripts: MuJoCo's passive viewer uses GLFW, and this
machine's native Wayland GLFW backend segfaults MuJoCo's viewer on shutdown (confirmed
via an isolated launch_passive()/close() repro, independent of anything in this
script); forcing GLFW's X11 backend avoids the crash. Even with that env var, the
viewer's background render thread still doesn't join cleanly on normal Python exit --
see the os._exit(0) call at the end of main() and its comment for why:
    PYGLFW_LIBRARY_VARIANT=x11 python scripts/record_dataset.py \\
        --repo-id=<you>/so101_baseline --num-episodes=100 \\
        --single-task="pick up the can and place it in the bin"

Controls during recording (same keys as stock lerobot-record):
    N / Right-arrow   end this episode now (keep it) and move to the next
    R / Left-arrow    discard this episode and re-record it
    Q / Esc           stop the whole session (episodes already saved stay saved)

Episodes also end automatically after --episode-time-s if you don't press a key first
-- set it generously (default 30s) and just press N as soon as the can is in the bin,
rather than trying to tune it to an exact grasp duration.

Re-run with the same --repo-id and --resume to keep adding episodes across multiple
sessions (recording 80-100 episodes by hand in one sitting is unrealistic).

--mock swaps the real gamepad for the deterministic sine-sweep teleop
(SO101MujocoTeleopConfig.mock) -- only useful for smoke-testing this driver's plumbing
without hardware attached, never for an actual recording session.

LIVE VIEW -- two independent windows, both on by default:
  --camera-preview  front+wrist camera-feed window -- what the arm can actually see,
                     the view you need to actually drive the pick-and-place. Reuses the
                     exact frames robot.get_observation() already rendered this tick
                     (read out of the observation dict via a thin wrapper around the
                     identity observation processor), so it costs no extra render pass
                     and doesn't meaningfully affect cadence.
  --viewer           free-roam 3D MuJoCo window, same one joint_teleop.py always shows,
                     for overall orientation alongside the camera preview. Its
                     background render thread competes with the camera renders for the
                     same GPU -- measured ~20Hz -> ~8-12Hz on this machine's integrated
                     graphics when both are on. --no-viewer keeps just the camera
                     preview (the one you actually need to drive by) at full recording
                     speed if that tradeoff isn't worth it.
Both need a real display (X11/Wayland) -- pass --no-viewer --no-camera-preview for a
headless/automated run (e.g. --mock smoke tests in a sandbox with no display). Without
either one you have no way to see the arm/can/cameras at all -- not viable for an
actual teleop recording session, only for automated plumbing tests.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import make_default_processors
from lerobot.scripts.lerobot_record import record_loop
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.utils.cycle_timer import CycleTimer
from lerobot.utils.feature_utils import combine_feature_dicts
from lerobot.utils.keyboard_input import init_keyboard_listener
from lerobot.utils.utils import init_logging, log_say

from lerobot_robot_so101_mujoco import (
    SO101Mujoco,
    SO101MujocoConfig,
    SO101MujocoTeleop,
    SO101MujocoTeleopConfig,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo-id", required=True, help="e.g. your_hf_username/so101_baseline")
    parser.add_argument("--root", type=Path, default=None, help="defaults to $HF_LEROBOT_HOME/<repo-id>")
    parser.add_argument("--num-episodes", type=int, required=True)
    parser.add_argument("--episode-time-s", type=float, default=30.0)
    parser.add_argument(
        "--confirm-on-timeout",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="if an episode runs the full --episode-time-s without you pressing N or R "
        "(a natural timeout, not a deliberate end), ask 'Episode timed out. Keep it? "
        "[Y/n]' before saving it -- same idea as confirm_on_timeout in the Franka "
        "gym_hil record_config.json. --no-confirm-on-timeout for a fully hands-off run.",
    )
    parser.add_argument(
        "--prep-time-s",
        type=float,
        default=2.0,
        help="pause after the automatic reset, before recording resumes, so you can see "
        "the can's new spawn position and get your hands back on the gamepad",
    )
    parser.add_argument("--fps", type=int, default=20, help="should match SO101MujocoConfig.control_dt (default 0.05s = 20Hz)")
    parser.add_argument("--single-task", required=True, help="e.g. 'pick up the can and place it in the bin'")
    parser.add_argument(
        "--randomize-can-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Phase 5 wants this True (pose randomization only); leave True for Phase 6 too "
        "once visual/physics DR is layered on top -- this flag is orthogonal to that",
    )
    parser.add_argument(
        "--can-xy",
        type=float,
        nargs=2,
        default=None,
        metavar=("X", "Y"),
        help="record every episode this session at this exact fixed can position instead of a "
        "random draw (overrides --randomize-can-pose entirely). For recording MimicGen reference "
        "demos at deliberately chosen positions (corners/edges/center of the sampling box) -- run "
        "this script once per position you want a reference at, e.g. --can-xy 0.16 -0.18 for one "
        "corner, then rerun with a different --can-xy for the next.",
    )
    parser.add_argument(
        "--fallen",
        action="store_true",
        help="record fallen-can recovery demos instead of standing-can pick demos: every reset spawns "
        "the can lying on its side instead of standing. --can-xy-log-path logs a 3rd value (heading) "
        "per episode when this is set.",
    )
    parser.add_argument(
        "--can-heading-deg",
        type=float,
        default=None,
        metavar="DEG",
        help="only meaningful with --fallen: pin every episode's heading to this exact value instead "
        "of drawing it randomly -- same idea as --can-xy, run once per (position, heading) combo you "
        "want in the reference grid.",
    )
    parser.add_argument(
        "--roll-impulse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="only meaningful with --fallen: after placing the can fallen, additionally give it a "
        "randomized knock and settle it via real physics (so101_mujoco.py's reset_scene's own "
        "roll_impulse) -- PLAN.md's suggested extra realism (a knocked-and-rolled pose instead of an "
        "idealized in-place tip), NOT required to record usable references. Off by default: it "
        "requires the scene's can-friction calibration (see PLAN.md/plan notes) to produce a "
        "realistic ~5-15cm roll rather than an unrealistically long slide -- confirm that's applied "
        "before turning this on. With it off, --can-heading-deg pins the exact final heading, no "
        "physics settle step.",
    )
    parser.add_argument("--mock", action="store_true", help="deterministic sine-sweep teleop, no gamepad required (smoke-test only)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--display-data", action="store_true", help="stream camera/state to rerun (requires rerun-sdk)")
    parser.add_argument(
        "--camera-preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="front+wrist camera-feed window -- what the arm can actually see, the view you "
        "actually need to drive the pick-and-place. Nearly free (reuses the frames "
        "get_observation() already rendered this tick, no extra render pass). Needs a "
        "real display -- pass --no-camera-preview for headless runs.",
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=1.0,
        help="extra upscale factor for the camera-preview window ONLY, on top of the "
        "native render resolution (SO101MujocoConfig.camera_shapes, default 480x480 "
        "per camera, same as joint_teleop.py's --cameras -- already sharp at 1.0, no "
        "need to stretch a low-res source the way an earlier 128x128 default needed). "
        "Bump this only if you want the window even bigger than 480x480/camera.",
    )
    parser.add_argument(
        "--viewer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="free-roam 3D MuJoCo window for overall orientation, alongside --camera-preview. "
        "Costs real cadence on integrated graphics (measured ~20Hz -> ~8-12Hz on this "
        "machine, since its background redraw thread competes with the camera renders "
        "for the same GPU) -- pass --no-viewer to keep just the camera preview at full "
        "speed if that tradeoff isn't worth it for you.",
    )
    parser.add_argument("--play-sounds", action="store_true")
    parser.add_argument(
        "--can-xy-log-path",
        type=Path,
        default=None,
        help="optional: append each episode's starting can pose to this JSON file as "
        "{episode_index: [x, y]} (or {episode_index: [x, y, heading]} with --fallen) after "
        "it's saved. The dataset itself has no object-pose feature, so without this flag a "
        "recorded episode's starting can position is lost the moment the next reset "
        "overwrites it -- this is for scripts/mimicgen_augment.py, which needs known source "
        "poses to retarget from. Off by default; a normal recording session doesn't need it.",
    )
    args = parser.parse_args()

    init_logging()

    can_xy_log: dict[int, list[float]] = {}
    if args.can_xy_log_path:
        args.can_xy_log_path.parent.mkdir(parents=True, exist_ok=True)
        if args.can_xy_log_path.exists():
            with open(args.can_xy_log_path) as f:
                can_xy_log = {int(k): v for k, v in json.load(f).items()}

    robot = SO101Mujoco(SO101MujocoConfig(id="record", randomize_can_pose_on_connect=args.randomize_can_pose))
    teleop = SO101MujocoTeleop(SO101MujocoTeleopConfig(id="record", mock=args.mock))

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    root = args.root or (HF_LEROBOT_HOME / args.repo_id)
    num_cameras = len(robot.cameras)
    if args.resume:
        dataset = LeRobotDataset.resume(
            args.repo_id,
            root=root,
            image_writer_threads=4 * num_cameras,
        )
        sanity_check_dataset_robot_compatibility(dataset, robot, args.fps, dataset_features)
        print(f"Resuming {root} -- {dataset.meta.total_episodes} episode(s) already saved.")
    else:
        dataset = LeRobotDataset.create(
            args.repo_id,
            args.fps,
            root=root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=True,
            image_writer_threads=4 * num_cameras,
        )

    can_heading_rad = None if args.can_heading_deg is None else float(np.deg2rad(args.can_heading_deg))

    def _fallen_kwargs(pinned_heading: float | None = None, roll: bool | None = None) -> dict:
        """this session's --fallen settings, as reset_scene() kwargs -- empty dict
        (today's plain standing-can behavior) unless --fallen is set. `roll` defaults
        to this session's --roll-impulse setting; passing `roll=False` explicitly
        (used on retry) pins the exact prior settled pose instead of re-rolling, so a
        retry reproduces identical conditions rather than a fresh random knock."""
        if not args.fallen:
            return {}
        return {
            "fallen": True,
            "can_heading": can_heading_rad if pinned_heading is None else pinned_heading,
            "roll_impulse": args.roll_impulse if roll is None else roll,
        }

    teleop.connect()
    robot.connect()
    if args.can_xy is not None or args.fallen:
        # Override wherever connect()'s own reset just placed the can. connect() only
        # ever calls reset_scene(randomize_can_pose=...) internally -- it has no idea
        # about --fallen -- so without this, episode 0 of a --fallen session would
        # start with the can STANDING regardless (confirmed the hard way: a real
        # recording session logged episode 0's heading as exactly 0.0, the value
        # heading_from_quat() returns for an upright can, while every subsequent
        # between-episode reset -- which does go through _fallen_kwargs() -- logged a
        # genuine random heading). --can-xy alone (no --fallen) keeps its original
        # behavior unchanged.
        robot.reset_scene(randomize_can_pose=(args.can_xy is None), can_xy=args.can_xy, **_fallen_kwargs())

    listener, events = init_keyboard_listener()
    timer = CycleTimer(args.fps)

    # The pose the CURRENT episode attempt actually started with -- captured right
    # after every reset that begins a new episode (including connect()'s own initial
    # one), and re-applied verbatim on a retry so R / "n" redoes the identical setup
    # instead of drifting to a different pose. See reset_scene()'s docstring.
    current_can_xy = robot.can_xy
    current_can_heading = robot.can_heading if args.fallen else None

    mj_viewer_handle = None
    live_robot_observation_processor = robot_observation_processor
    cam_window_name = "SO-101 record: front | wrist"

    try:
        if args.viewer or args.camera_preview:
            # Set up inside the try/finally, not before it: if launch_passive() fails
            # (e.g. left on with no real display attached), robot/teleop are already
            # connected at this point and still need a clean disconnect.
            import cv2

            if args.viewer:
                import mujoco.viewer as mj_viewer

                mj_viewer_handle = mj_viewer.launch_passive(robot.model, robot.data)

            def _observe_and_show(obs):
                # robot_observation_processor is IdentityProcessorStep-only by default
                # (see make_default_robot_observation_processor()) -- calling it first,
                # then displaying, keeps this a pure passthrough for whatever actually
                # gets recorded, with the live view as a side effect only.
                processed = robot_observation_processor(obs)

                if mj_viewer_handle is not None:
                    if mj_viewer_handle.is_running():
                        mj_viewer_handle.sync()
                    else:
                        # User closed the 3D window -- stop after this episode, same as
                        # pressing Q (record_loop() only checks exit_early/rerecord
                        # mid-episode; stop_recording is checked between episodes).
                        events["stop_recording"] = True

                if args.camera_preview:
                    frames = []
                    for cam_name in ("front", "wrist"):
                        if cam_name not in obs:
                            continue
                        # Upscale the DISPLAY copy only -- obs itself (what actually
                        # gets recorded, via `processed` above) is untouched, still at
                        # the native render resolution.
                        bgr = cv2.cvtColor(obs[cam_name], cv2.COLOR_RGB2BGR)
                        if args.preview_scale != 1.0:
                            bgr = cv2.resize(
                                bgr, None, fx=args.preview_scale, fy=args.preview_scale,
                                interpolation=cv2.INTER_LINEAR,
                            )
                        cv2.putText(bgr, cam_name, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        frames.append(bgr)
                    if frames:
                        cv2.imshow(cam_window_name, cv2.hconcat(frames))
                        cv2.waitKey(1)

                return processed

            live_robot_observation_processor = _observe_and_show

        with VideoEncodingManager(dataset):
            target_episodes = dataset.meta.total_episodes + args.num_episodes if args.resume else args.num_episodes
            while dataset.num_episodes < target_episodes and not events["stop_recording"]:
                episode_index = dataset.num_episodes
                log_say(f"Recording episode {episode_index}", args.play_sounds)
                episode_start_t = time.perf_counter()
                record_loop(
                    robot=robot,
                    events=events,
                    fps=args.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=live_robot_observation_processor,
                    dataset=dataset,
                    teleop=teleop,
                    control_time_s=args.episode_time_s,
                    single_task=args.single_task,
                    display_data=args.display_data,
                    timer=timer,
                )
                episode_elapsed_s = time.perf_counter() - episode_start_t

                if (
                    args.confirm_on_timeout
                    and listener is not None
                    and not events["rerecord_episode"]
                    and episode_elapsed_s >= args.episode_time_s - (1.0 / args.fps)
                ):
                    # Ran the full --episode-time-s without you pressing N or R -- a
                    # natural timeout, not a deliberate end (if you HAD pressed one,
                    # rerecord_episode/exit_early already short-circuited record_loop()
                    # well before episode_time_s elapsed). The keyboard listener reads
                    # raw bytes off the same stdin fd input() needs (cbreak mode, echo
                    # off) -- stop it first or the two fight over the same keystrokes,
                    # then start a fresh one for the next episode.
                    listener.stop()
                    answer = input("Episode timed out. Keep it? [Y/n]: ").strip().lower()
                    discard = answer in ("n", "no")
                    listener, events = init_keyboard_listener()
                    if discard:
                        events["rerecord_episode"] = True

                if events["rerecord_episode"]:
                    log_say("Re-record episode", args.play_sounds)
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    timer.log_episode_summary("discarded episode")
                    timer.restart()
                    # Put the arm/can back the way THIS episode actually started, not a
                    # fresh random draw -- a discarded take should be retried under the
                    # same conditions, not a different one. Passing current_can_xy back
                    # in explicitly is required here: randomize_can_pose=False alone
                    # does NOT preserve the previous pose (mj_resetDataKeyframe always
                    # overwrites the can's qpos first) -- see reset_scene()'s docstring.
                    robot.reset_scene(
                        randomize_can_pose=False,
                        can_xy=current_can_xy,
                        **_fallen_kwargs(pinned_heading=current_can_heading, roll=False),
                    )
                    teleop.reset_target()
                    continue

                dataset.save_episode()
                timer.log_episode_summary(f"episode {episode_index}")
                timer.restart()

                if args.can_xy_log_path:
                    # current_can_xy/current_can_heading still reflect the episode just
                    # saved -- the next reset (a few lines below) doesn't happen until
                    # after this.
                    pose = [float(current_can_xy[0]), float(current_can_xy[1])]
                    if args.fallen:
                        pose.append(float(current_can_heading))
                    can_xy_log[episode_index] = pose
                    with open(args.can_xy_log_path, "w") as f:
                        json.dump(can_xy_log, f, indent=2)

                if dataset.num_episodes < target_episodes and not events["stop_recording"]:
                    # The auto-reset this whole script exists for -- see module
                    # docstring. Both calls are required together: resetting only the
                    # robot's physics state would leave the teleoperator's own tracked
                    # target stale (see reset_target()'s docstring).
                    if args.can_xy is not None:
                        current_can_xy = robot.reset_scene(randomize_can_pose=False, can_xy=args.can_xy, **_fallen_kwargs())
                    else:
                        current_can_xy = robot.reset_scene(randomize_can_pose=args.randomize_can_pose, **_fallen_kwargs())
                    current_can_heading = robot.can_heading if args.fallen else None
                    teleop.reset_target()
                    if args.prep_time_s > 0:
                        log_say("Reset the environment", args.play_sounds)
                        time.sleep(args.prep_time_s)
    finally:
        timer.log_run_summary()
        log_say("Stop recording", args.play_sounds, blocking=True)
        dataset.finalize()
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()
        if listener is not None:
            listener.stop()
        if mj_viewer_handle is not None:
            mj_viewer_handle.close()
        if args.camera_preview:
            import cv2

            cv2.destroyWindow(cam_window_name)

    if args.push_to_hub:
        dataset.push_to_hub()
    print(f"\nDone. {dataset.num_episodes} episodes saved to {dataset.root}")

    if args.viewer:
        # A normal Python return here hangs (confirmed via an isolated repro outside
        # this script, and specifically isolated to mujoco.viewer.launch_passive() --
        # a plain cv2 window closes and exits fine on its own): the passive viewer's
        # background render thread doesn't join cleanly on interpreter shutdown on
        # this machine, even with PYGLFW_LIBRARY_VARIANT=x11 (that env var fixes a
        # separate GLFW/Wayland segfault-on-close, but not this). Everything that
        # matters is already flushed and closed above (dataset finalized, robot/teleop
        # disconnected, viewer/window closed, hub push done) -- os._exit() skips only
        # the hanging thread-join, not any of our own cleanup.
        os._exit(0)


if __name__ == "__main__":
    main()
