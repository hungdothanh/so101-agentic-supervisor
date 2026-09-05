#!/usr/bin/env python3
"""Open a scene in the interactive MuJoCo viewer, starting from its "home"
keyframe -- NOT `python -m mujoco.viewer --mjcf=...`, which starts every
model from a plain `mujoco.MjData(model)` with `ctrl` at all zeros.

Why this matters: a scene's "home" keyframe sets both `qpos` (joint
positions) and `ctrl` (position-actuator targets) together. Confirmed by
reading mujoco.viewer's own source (_file_loader / _launch_internal in
site-packages/mujoco/viewer.py): the `--mjcf=` path always does a plain
`MjData(model)` and never touches keyframes. If the scene's home pose is
non-zero (any joint's `ref` != 0), qpos correctly starts there (`ref` sets
qpos0), but ctrl stays at 0 regardless. The position actuators then spend
their first moments driving every joint from "home" back toward ctrl=0 --
visible as the arm shaking and collapsing into a different pose the instant
the viewer opens. This script avoids that by constructing `data`, resetting
it to the named keyframe (which sets ctrl to match), and only then handing
that already-consistent state to the viewer.

Usage:
    python3 scripts/view_scene.py [path/to/scene.xml] [--key NAME]

Defaults to assets/scenes/pen_pickplace_scene.xml and keyframe "home".
Run in the `lerobot_smolvla` conda env (has mujoco).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENE = PROJECT_ROOT / "assets" / "scenes" / "pen_pickplace_scene.xml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scene", type=Path, nargs="?", default=DEFAULT_SCENE)
    parser.add_argument("--key", default="home", help='keyframe name to start from (default: "home")')
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.scene))
    data = mujoco.MjData(model)

    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, args.key)
    if key_id < 0:
        print(f'WARNING: no keyframe named "{args.key}" in {args.scene} -- starting from qpos0/ctrl=0 instead.')
    else:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        print(f'Starting from keyframe "{args.key}": qpos={data.qpos}, ctrl={data.ctrl}')

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
