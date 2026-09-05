#!/usr/bin/env python3
"""Fetch the canonical SO-101 URDF + meshes and convert to a true-to-scale MJCF.

Source is the same bucket LeRobot's own kinematics/IK examples use
(`examples/isaac_teleop_to_so101`), and the same one the earlier
`so101_sim2real` ROS2 project's `fetch_so101_urdf.py` pulled from:

    hf://buckets/lerobot/robot-urdfs/so101  ->  so101_new_calib.urdf + assets/*.stl

Unlike that earlier script (which produced a xacro fragment for a ROS2/Gazebo
build), this one converts straight to MJCF using MuJoCo's native URDF
compiler, with NO rescaling anywhere -- every link length/mass/inertia comes
straight from the fetched URDF, so the resulting arm is to-scale by
construction. Joint-space <position> actuators are then hand-added (URDF
<transmission> tags aren't picked up by MuJoCo's URDF importer).

Also fetches the real wrist-camera hex-nut mount bracket (the 3D-printed
adapter for a 32x32mm UVC USB camera module that bolts onto the wrist roll
follower) from TheRobotStudio/SO-ARM100's hardware repo, and attaches it to
gripper_link -- see add_wrist_camera() for how its pose was derived.

Produces (both gitignored raw-fetch artifacts aside):
    assets/so101/so101.xml     MJCF: to-scale arm body tree + position actuators + wrist camera
    assets/so101/meshes/*.stl  visual/collision meshes, flattened

Usage:
    python3 scripts/fetch_so101_urdf.py [--dest DIR]

Run in the `lerobot_smolvla` conda env (has huggingface_hub>=1.5 for
sync_bucket, and mujoco).
"""

from __future__ import annotations

import argparse
import shutil
import struct
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import coacd
import mujoco
import numpy as np
import trimesh

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSET_DIR = PROJECT_ROOT / "assets" / "so101"
BUCKET_URI = "hf://buckets/lerobot/robot-urdfs/so101"
UPSTREAM_URDF_NAME = "so101_new_calib.urdf"
EXPECTED_FK_FRAME = "gripper_frame_link"

# Canonical LeRobot SO-101 joint-name/unit contract (matches
# lerobot.robots.so_follower.SOFollower exactly: 5 arm joints + 1 gripper
# joint, all revolute, limits in radians). Order here is also actuator order.
CANONICAL_JOINTS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Default/reset pose (radians, CANONICAL_JOINTS order) -- picked interactively
# in the mujoco.viewer control panel. NOT applied anywhere in this file --
# kept here purely as the documented source of truth for these numbers.
#
# An earlier version applied this via each joint's `ref` attribute, on the
# theory that `ref` just sets qpos0 (the default starting qpos) with no
# other effect. That's wrong: `ref` shifts what a given qpos value MEANS
# geometrically -- the rotation MuJoCo actually applies during forward
# kinematics is (qpos - ref), not qpos. Confirmed directly (built a hinge
# joint with ref=1.0, set qpos=1.0, computed FK: resulting rotation was
# exactly 0deg, not 1 rad). Setting ref AND qpos to the same HOME_QPOS
# value therefore made every joint's *visual* rotation cancel to zero --
# the arm looked identical to the all-zero pose regardless of what
# HOME_QPOS said -- and silently ate into that joint's usable *range* on
# one side too, since `range` is enforced in raw qpos, not in the
# ref-shifted angle (this is what made shoulder_lift refuse to fold
# further: its true limit is -1.745, so with ref=-1.7 sitting almost on
# top of it, only ~0.045 rad of raw qpos remained on that side).
#
# The only correct way to start a model at a non-zero pose without
# distorting its kinematics is a <keyframe> (qpos/ctrl set directly, no
# reinterpretation) applied explicitly via mj_resetDataKeyframe -- see
# assets/scenes/pen_pickplace_scene.xml's "home" keyframe, and
# scripts/view_scene.py, which applies it before opening the viewer (since
# mujoco.viewer's own file loader never touches keyframes on its own). If
# you change these values, update that keyframe to match -- it does NOT
# read this dict, they're just meant to agree.
HOME_QPOS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -1.75,
    "elbow_flex": 1.62,
    "wrist_flex": 1.18,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}

# Where base_link sits in the scene's world frame. Lives here, not as a
# hand-edit on the generated so101.xml or a wrapper <body>/<frame> around the
# scene's <include>, because MuJoCo's <include> must stay a direct child of
# <mujoco> to correctly splice a multi-section file's <compiler>/<asset>/
# <actuator>/<contact> (verified empirically: nesting it one level down --
# inside a <frame> in <worldbody> -- broke on the included <compiler> tag).
# Only base_link's own pos can carry this offset. Currently (0, 0.15, 0):
# moved 15cm along the table's short edge, away from the front camera (which
# sits at y=-0.40), for more separation between the arm and the camera in
# its own frame.
ARM_MOUNT_POS = (0.0, 0.15, 0.0)

# moving_jaw_so101_v1_link is the ONE part that actually moves to grasp
# (a single hook-style finger closing against a fixed palm -- see PLAN.md's
# pen-vs-can pivot notes). Every other yellow part, including the
# similarly-shaped wrist_roll_follower_so101_v1 bracket right next to it, is
# a fixed structural piece that never moves. Both were the same stock
# yellow (1 0.82 0.12 1), making them visually indistinguishable during
# teleop -- confirmed as a real source of confusion: a user judged the
# gripper as "not touching" an object based on the fixed bracket's visual
# clearance, when the actual contact was on that bracket, not the jaw.
# Recoloring the real moving jaw distinctly so its position is trackable by
# eye during teleop, independent of the fixed bracket around it.
MOVING_JAW_COLOR = (0.15, 0.85, 0.25, 1.0)

# MuJoCo's default mesh collision is a single convex hull, which fills in
# any real concavity. Confirmed this wasn't harmless for two specific parts:
# wrist_roll_follower_so101_v1's hull is 2.58x its true volume (the fork
# notch the gripper's finger swings through gets filled in solid), and
# moving_jaw_so101_v1's own hull is 2.33x its true volume (its gripping-
# surface curve gets flattened outward). Together these erase real,
# task-relevant open space right where an object needs to sit to be
# grasped -- confirmed by a user who observed the arm pushing a small can
# away despite clear visual daylight between the can and both parts.
# CONVEX_DECOMPOSE_TARGETS lists (body, mesh) pairs to replace with a coacd
# decomposition instead of a single hull. threshold=0.1 chosen empirically:
# a finer threshold (0.05) gives ~50+ tiny parts (mostly capturing
# irrelevant bolt-hole concavities) for a similar volume-fidelity gain a
# threshold this fine barely improves on; coarser (0.2-0.3) drops back
# toward single-hull-like volume ratios. Needs `pip install coacd`.
CONVEX_DECOMPOSE_TARGETS = [
    ("gripper_link", "wrist_roll_follower_so101_v1"),
    ("moving_jaw_so101_v1_link", "moving_jaw_so101_v1"),
]
CONVEX_DECOMPOSE_THRESHOLD = 0.1

# Position-actuator gains: placeholders for Phase 0 (loadable + roughly
# stable), retuned in Phase 2/3 against teleop response and env stability.
ACTUATOR_KP = {
    "shoulder_pan": 50.0,
    "shoulder_lift": 50.0,
    "elbow_flex": 50.0,
    "wrist_flex": 25.0,
    "wrist_roll": 15.0,
    "gripper": 10.0,
}

# Per-joint force limit (N*m), NOT a single global value mirroring the
# URDF's declared <limit effort="10"> for every joint. That uniform 10 N*m
# is fine for shoulder_pan/shoulder_lift/elbow_flex (large downstream
# inertia -- they're moving most of the arm's own mass), but is wildly
# excessive for wrist_flex/wrist_roll/gripper, which drive progressively
# smaller downstream masses (wrist_roll spins just the gripper assembly
# about its own long axis; gripper moves only the small moving_jaw).
# Confirmed empirically: commanding a large step target with forcerange=10
# on these joints produces a runaway spin the position controller can't
# arrest -- gripper peaked at 563 rad/s and never reached its target at
# all; wrist_roll peaked at 331 rad/s and stalled ~1.6 rad short of target;
# even wrist_flex (which does eventually recover) spiked to 117 rad/s. This
# isn't a kp/kv tuning problem -- swept kv from 2.25 to 200 for wrist_roll
# with zero effect, because the RAW (pre-clamp) force was already so large
# in both the kp and kv terms that every value clamped to the identical
# forcerange boundary, making the dynamics identical regardless of kv.
# Swept forcerange downward per joint instead (holding kp/kv unchanged)
# until each one reached a large step target cleanly (final qvel exactly
# 0, qpos exactly at target) with a physically reasonable peak velocity;
# these values also happen to land much closer to a real Feetech
# STS3215's actual torque output (~1-3 N*m) than the placeholder 10.
ACTUATOR_FORCERANGE = {
    "shoulder_pan": (-10.0, 10.0),
    "shoulder_lift": (-10.0, 10.0),
    "elbow_flex": (-10.0, 10.0),
    "wrist_flex": (-2.0, 2.0),
    "wrist_roll": (-0.5, 0.5),
    "gripper": (-0.2, 0.2),
}

# Damping: use MuJoCo's own `dampratio` (available on <position> actuators),
# NOT a hand-picked kv. Without any velocity damping, every joint is an
# undamped spring (force = kp*(ctrl-qpos), zero friction anywhere) and rings
# forever once perturbed -- confirmed empirically: with kv=0/damping=0 under
# the default Euler integrator, a displaced arm never settles (residual
# |qvel| ~0.6-1.6 rad/s after 1s). Fixing that needs two things together:
# MuJoCo's recommended integrator for velocity-feedback actuators
# ("implicitfast" -- a plain kv is numerically UNSTABLE under the default
# Euler integrator, velocities blew up to 1000+ rad/s), and correct damping.
#
# An earlier version used a hand-picked kv = 0.15*kp (same ratio for every
# joint) instead of dampratio, on the theory that it "settles to ~0 residual
# velocity within ~1s" -- true, but "~1s" was the tell: for wrist_roll and
# gripper specifically, whose downstream inertia is far smaller than
# shoulder_pan/shoulder_lift/elbow_flex's (wrist_roll spins just the
# gripper assembly about its own long axis; gripper moves only the small
# moving_jaw), that fixed ratio is wildly wrong -- verified directly:
# `dampratio="1"` (critical damping, computed from each joint's ACTUAL
# inertia) gives wrist_roll a kv of ~0.049, not the ~2.25 the fixed 0.15
# ratio produced -- 45x too much damping for its real inertia. Instead of
# settling, that oversized kv produced a visibly slow, ~0.5s exponential
# creep after every command (confirmed: qvel trace was still -0.4 rad/s
# 300 steps/0.6s after a step command, vs dampratio=1's clean settle to
# exactly 0 within ~40-50 steps) -- read by a person watching it live as
# the wrist continuing to shake/drift for a long moment after every
# adjustment, exactly the reported symptom. dampratio=1 is the textbook
# critically-damped response (fast, no overshoot) and is correct BY
# CONSTRUCTION for every joint's own inertia, so it doesn't need
# per-joint retuning the way a hand-picked kv ratio does.
JOINT_DAMPING = 0.01

# Real hardware: TheRobotStudio/SO-ARM100's "Hex-Nut Recess Wrist Camera"
# adapter, designed for a 32x32mm UVC USB camera module bolted onto the
# wrist roll follower's hex-nut recesses (see that repo's README under
# Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module/). Confirmed
# against user-supplied reference photos of the real assembled hardware to
# be the right part -- an earlier pass wrongly concluded it wasn't (based on
# a misread of the photos) and substituted a fabricated wedge shape instead;
# that substitution was reverted.
WRIST_CAM_MOUNT_URL = (
    "https://raw.githubusercontent.com/TheRobotStudio/SO-ARM100/main/Optional/"
    "SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module/stl/SO-ARM101_camera_wrist_mount.stl"
)
WRIST_CAM_MOUNT_FILENAME = "wrist_cam_mount.stl"

# This mount STL was confirmed to share the exact same native coordinate
# frame as the canonical Wrist_Roll_Follower_SO101.stl part (bounding boxes
# match to <0.01mm once converted mm->m), so it's attached to gripper_link
# using the SAME pos/quat MuJoCo already assigned to the
# wrist_roll_follower_so101_v1 mesh geom -- no manual alignment needed, and
# not something to change: that transform is the bracket's real, physically
# correct screw-mount position.
#
# The camera-mounting plate (4 M2 holes + center lens/cable cutout) has two
# opposite faces; WRIST_CAM_PLATE_NORMAL_LOCAL is this face's outward
# (camera-facing) normal in the STL's own native (mm) frame. An earlier pass
# picked the WRONG one of the two -- verified by rendering the isolated
# mount from 8 directions and (incorrectly) reading off which one faced the
# render camera -- and it pointed the camera sensor ~122deg away from the
# gripper (dot product -0.527 with the direction to gripper_frame_link).
# The other candidate normal, tested here, gives +0.492 (a legitimate,
# if not perfectly boresighted, "look toward the workspace" direction) --
# confirming it's the correct one.
WRIST_CAM_PLATE_NORMAL_LOCAL = np.array([0.4226, 0.9063, 0.0])

# Sized and positioned against the bracket's ACTUAL hole, found by ray-casting
# the compiled mesh (not the raw STL) once the mount's real installed pose
# (base attachment + the user's calibrated WRIST_CAM_MOUNT_EXTRA_* above) was
# known: the hole is roughly circular, ~11.2mm in radius, and the plate is ~4.4mm thick
# at that point (measured as the gap between the plate's front and back face
# clusters along WRIST_CAM_PLATE_NORMAL_LOCAL). The hole's center matched the
# plate-centroid calculation below to <0.1mm, so that part was already
# correct -- the previous 7mm-radius cylinder, pushed 16mm out from the
# plate, was just undersized for the ~11mm hole and floated well clear of it
# instead of sitting seated in it.
CAMERA_CYLINDER_RADIUS = 0.0095  # slightly under the ~11.2mm hole radius: sits snugly with a small lip
CAMERA_CYLINDER_HALF_LENGTH = 0.009  # ~18mm total: pokes a little past both faces of the ~4.4mm-thick plate
CAMERA_FORWARD_BIAS = 0.002  # small forward (lens-side) bias so it's not perfectly symmetric in the hole

# ---------------------------------------------------------------------------
# MANUAL FINE-TUNING: edit these two if the mount's position/orientation
# still doesn't look right after regenerating, then re-run this script.
# Nothing else in this file needs to change -- both apply as one rigid
# transform on top of the bracket's base attachment (see add_wrist_camera),
# so the camera cylinder/sensor move and rotate together with the bracket
# and stay "through the hole" no matter what you set here.
#
#   WRIST_CAM_MOUNT_EXTRA_RPY_DEG: extra rotation in degrees, applied about
#     gripper_link's OWN fixed x/y/z axes (extrinsic X-Y-Z), AFTER the
#     bracket's base attachment orientation. E.g. [0, 0, 90] spins the
#     mount 90deg about gripper_link's z-axis without moving its attach
#     point; [0, 180, 0] flips it to face the opposite way.
#   WRIST_CAM_MOUNT_EXTRA_POS: extra translation in meters, in
#     gripper_link's frame, applied AFTER the base attachment position.
# ---------------------------------------------------------------------------
WRIST_CAM_MOUNT_EXTRA_RPY_DEG = np.array([90.0, 0.0, -90.0])
WRIST_CAM_MOUNT_EXTRA_POS = np.array([-0.0150, 0.0240, -0.0325])


def fetch(dest: Path) -> None:
    from huggingface_hub import sync_bucket

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Syncing {BUCKET_URI} -> {dest}")
    sync_bucket(BUCKET_URI, str(dest), quiet=False)


def prepare_urdf_for_mujoco(fetch_dir: Path) -> tuple[Path, dict]:
    """Flatten meshes into assets/so101/meshes/, rewrite mesh paths, inject
    a <mujoco> compiler block, and write an intermediate URDF MuJoCo can load
    directly (mesh paths resolved via compiler meshdir, no ROS package:// URIs).
    """
    src_urdf = fetch_dir / UPSTREAM_URDF_NAME
    tree = ET.parse(src_urdf)
    root = tree.getroot()

    mesh_dir = ASSET_DIR / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for stl in sorted((fetch_dir / "assets").glob("*.stl")):
        shutil.copy2(stl, mesh_dir / stl.name)
        copied.append(stl.name)

    for mesh_el in root.iter("mesh"):
        fn = mesh_el.get("filename")
        if fn:
            mesh_el.set("filename", Path(fn).name)

    # MuJoCo URDF extension: compiler options as a <mujoco> child of <robot>.
    # angle="radian" matches the URDF's own joint limit units (no conversion).
    # fusestatic="false": without it, MuJoCo's URDF importer fuses any body
    # attached via a "fixed" URDF joint into its parent (and fuses the root
    # link, which has no joint at all, straight into world). That silently
    # deletes base_link and gripper_frame_link from the compiled model --
    # the latter is this project's designated FK reference frame, so it must
    # survive as an addressable body.
    mujoco_el = ET.Element("mujoco")
    ET.SubElement(
        mujoco_el,
        "compiler",
        {"angle": "radian", "meshdir": "meshes", "balanceinertia": "true", "fusestatic": "false"},
    )
    root.insert(0, mujoco_el)

    joints_found = {}
    for joint_el in root.findall("joint"):
        name = joint_el.get("name")
        limit_el = joint_el.find("limit")
        joints_found[name] = {
            "type": joint_el.get("type"),
            "lower": float(limit_el.get("lower")) if limit_el is not None else None,
            "upper": float(limit_el.get("upper")) if limit_el is not None else None,
        }

    link_names = [link.get("name") for link in root.findall("link")]

    out_urdf = ASSET_DIR / "_so101_new_calib.mujoco_intermediate.urdf"
    xml_str = ET.tostring(root, encoding="unicode")
    out_urdf.write_text('<?xml version="1.0"?>\n' + xml_str + "\n")

    manifest = {
        "source": f"{BUCKET_URI}/{UPSTREAM_URDF_NAME}",
        "links": link_names,
        "joints": joints_found,
        "meshes_copied": copied,
        "canonical_joints_matched": all(j in joints_found for j in CANONICAL_JOINTS),
        "fk_frame_present": EXPECTED_FK_FRAME in link_names,
    }
    return out_urdf, manifest


def convert_to_mjcf(intermediate_urdf: Path) -> Path:
    """Compile the URDF with MuJoCo and save the canonical MJCF."""
    model = mujoco.MjModel.from_xml_path(str(intermediate_urdf))
    mjcf_path = ASSET_DIR / "so101.xml"
    mujoco.mj_saveLastXML(str(mjcf_path), model)
    return mjcf_path


def add_actuators(mjcf_path: Path) -> None:
    """Hand-add joint-space <position> actuators (URDF <transmission> tags
    aren't picked up by MuJoCo's URDF importer), plus the damping (joint
    damping + actuator dampratio) and integrator needed for them to
    actually settle -- see the comment above ACTUATOR_FORCERANGE for why
    all of these are needed together, and why dampratio specifically
    (not a hand-picked kv)."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    joint_names_in_model = {j.get("name") for j in root.iter("joint")}
    missing = [j for j in CANONICAL_JOINTS if j not in joint_names_in_model]
    if missing:
        raise RuntimeError(f"Expected joints missing from compiled MJCF: {missing}")

    for joint_el in root.iter("joint"):
        if joint_el.get("name") in CANONICAL_JOINTS:
            joint_el.set("damping", str(JOINT_DAMPING))

    option_el = root.find("option")
    if option_el is None:
        option_el = ET.Element("option")
        root.insert(list(root).index(root.find("compiler")) + 1, option_el)
    option_el.set("integrator", "implicitfast")

    actuator_el = root.find("actuator")
    if actuator_el is None:
        actuator_el = ET.SubElement(root, "actuator")

    joint_limits = {j.get("name"): j.get("range") for j in root.iter("joint") if j.get("range")}

    for jname in CANONICAL_JOINTS:
        kp = ACTUATOR_KP[jname]
        fr_lo, fr_hi = ACTUATOR_FORCERANGE[jname]
        ET.SubElement(
            actuator_el,
            "position",
            {
                "name": f"{jname}_act",
                "joint": jname,
                "kp": str(kp),
                "dampratio": "1",
                "ctrlrange": joint_limits[jname],
                "forcerange": f"{fr_lo} {fr_hi}",
            },
        )

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, xml_declaration=True, encoding="utf-8")


def add_self_collision_exclusions(mjcf_path: Path) -> None:
    """Exclude collision between every directly-connected (parent/child) body
    pair in the kinematic tree.

    Real-world reason, not a modeling shortcut: adjacent links are bolted
    together (motor housings, brackets), so their collision meshes overlap
    by design at the joint. Confirmed empirically -- without this, the
    compiled model is NOT stable: base_link and shoulder_link alone
    interpenetrate ~2.5cm, and letting MuJoCo's contact solver resolve that
    produces exploding velocities (max |qvel| > 1000) within one second of
    simulated time, even with the arm sitting still and unactuated.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    worldbody = root.find("worldbody")

    contact_el = root.find("contact")
    if contact_el is None:
        contact_el = ET.SubElement(root, "contact")

    def walk(parent_body_el):
        parent_name = parent_body_el.get("name")
        for child in parent_body_el.findall("body"):
            child_name = child.get("name")
            if parent_name and child_name:
                ET.SubElement(contact_el, "exclude", {"body1": parent_name, "body2": child_name})
            walk(child)

    for top_body in worldbody.findall("body"):
        walk(top_body)

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, xml_declaration=True, encoding="utf-8")


def set_arm_mount_pos(mjcf_path: Path) -> None:
    """Set base_link's own pos to ARM_MOUNT_POS -- see that constant's
    comment for why this has to be base_link's pos, not a wrapper element."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    base_link = root.find(".//body[@name='base_link']")
    if base_link is None:
        raise RuntimeError(f"{mjcf_path} has no base_link body")
    base_link.set("pos", " ".join(str(v) for v in ARM_MOUNT_POS))
    tree.write(mjcf_path, xml_declaration=True, encoding="utf-8")


def recolor_moving_jaw(mjcf_path: Path) -> None:
    """Recolor the moving_jaw_so101_v1_link body's geom to MOVING_JAW_COLOR
    -- see that constant's comment for why."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    moving_jaw_body = root.find(".//body[@name='moving_jaw_so101_v1_link']")
    if moving_jaw_body is None:
        raise RuntimeError(f"{mjcf_path} has no moving_jaw_so101_v1_link body")
    geoms = moving_jaw_body.findall("geom")
    if len(geoms) != 1:
        raise RuntimeError(f"expected exactly 1 geom on moving_jaw_so101_v1_link, found {len(geoms)}")
    geoms[0].set("rgba", " ".join(str(v) for v in MOVING_JAW_COLOR))
    tree.write(mjcf_path, xml_declaration=True, encoding="utf-8")


def add_convex_decompositions(mjcf_path: Path, mesh_dir: Path) -> None:
    """For each (body, mesh) in CONVEX_DECOMPOSE_TARGETS, decompose that
    mesh's STL into convex parts (coacd) and add them as the body's real
    collision geoms, turning the original single-mesh geom visual-only --
    see CONVEX_DECOMPOSE_TARGETS's comment for why a single convex hull
    isn't good enough for these two parts specifically."""
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    asset_el = root.find("asset")

    for body_name, base_mesh_name in CONVEX_DECOMPOSE_TARGETS:
        stl_path = mesh_dir / f"{base_mesh_name}.stl"
        mesh = trimesh.load(stl_path)
        cmesh = coacd.Mesh(mesh.vertices, mesh.faces)
        parts = coacd.run_coacd(cmesh, threshold=CONVEX_DECOMPOSE_THRESHOLD)

        part_names = []
        for i, (verts, faces) in enumerate(parts):
            part_name = f"{base_mesh_name}_hull{i}"
            trimesh.Trimesh(vertices=verts, faces=faces).export(mesh_dir / f"{part_name}.stl")
            part_names.append(part_name)
            ET.SubElement(asset_el, "mesh", {"name": part_name, "file": f"meshes/{part_name}.stl"})

        body_el = root.find(f".//body[@name='{body_name}']")
        geoms = [g for g in body_el.findall("geom") if g.get("mesh") == base_mesh_name]
        if len(geoms) != 1:
            raise RuntimeError(f"expected exactly 1 geom referencing {base_mesh_name} on {body_name}, found {len(geoms)}")
        orig_geom = geoms[0]
        orig_geom.set("contype", "0")
        orig_geom.set("conaffinity", "0")

        orig_pos = orig_geom.get("pos", "0 0 0")
        orig_quat = orig_geom.get("quat", "1 0 0 0")
        for part_name in part_names:
            ET.SubElement(
                body_el,
                "geom",
                {
                    "type": "mesh",
                    "mesh": part_name,
                    "pos": orig_pos,
                    "quat": orig_quat,
                    "rgba": "0 0 0 0",
                    "contype": "1",
                    "conaffinity": "1",
                },
            )
        print(f"{base_mesh_name}: {len(part_names)} convex parts (was 1 hull) on {body_name}")

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, xml_declaration=True, encoding="utf-8")


def fetch_wrist_cam_mount() -> Path:
    """Download the real wrist-camera mount bracket STL (mm units, native
    OnShape export -- unlike the arm's own bucket meshes, this one needs a
    scale="0.001 0.001 0.001" in MJCF)."""
    dest = ASSET_DIR / "meshes" / WRIST_CAM_MOUNT_FILENAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Fetching {WRIST_CAM_MOUNT_URL} -> {dest}")
    urllib.request.urlretrieve(WRIST_CAM_MOUNT_URL, dest)
    return dest


def _load_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (normals, triangles, areas) -- triangles is (n,3,3), all in the STL's native units."""
    with open(path, "rb") as f:
        f.read(80)
        ntri = struct.unpack("<I", f.read(4))[0]
        normals = np.zeros((ntri, 3))
        tris = np.zeros((ntri, 3, 3))
        for i in range(ntri):
            normals[i] = struct.unpack("<fff", f.read(12))
            for j in range(3):
                tris[i, j] = struct.unpack("<fff", f.read(12))
            f.read(2)
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    return normals, tris, areas


def compute_wrist_cam_plate_centroid_local(stl_path: Path) -> np.ndarray:
    """Area-weighted centroid (meters, STL-native frame) of the camera-mounting
    plate face, identified by its known outward normal (see
    WRIST_CAM_PLATE_NORMAL_LOCAL's derivation notes above)."""
    normals, tris, areas = _load_binary_stl(stl_path)
    centroids_mm = tris.mean(axis=1)
    mask = (normals @ WRIST_CAM_PLATE_NORMAL_LOCAL) > 0.999
    if mask.sum() == 0:
        raise RuntimeError(
            "No triangles matched the expected camera-plate normal -- has the "
            "upstream mount STL changed? Re-derive WRIST_CAM_PLATE_NORMAL_LOCAL."
        )
    centroid_mm = (centroids_mm[mask] * areas[mask, None]).sum(axis=0) / areas[mask].sum()
    return centroid_mm / 1000.0


def add_wrist_camera(mjcf_path: Path, mount_stl_path: Path) -> None:
    """Attach the real wrist-cam mount bracket, a cylinder representing the
    camera passing through its hole, and an MJCF <camera> sensor to
    gripper_link.

    Bracket placement: reuses the exact pos/quat MuJoCo already assigned to
    the wrist_roll_follower_so101_v1 mesh geom on gripper_link, since the
    mount STL shares that mesh's native coordinate frame -- this is the
    bracket's real, physically correct screw-mount position and isn't
    something to adjust.

    Camera cylinder + sensor: BOTH placed along the bracket's own hole
    normal (WRIST_CAM_PLATE_NORMAL_LOCAL, transformed into gripper_link's
    frame) -- one direction for position and aim, so the camera necessarily
    looks straight through the hole, matching the real hardware and the
    user's reference photos. That normal isn't a free choice: of the plate's
    two opposite faces, only one is physically the outward, camera-facing
    side, and picking the correct one is a one-time geometric fact about
    this STL (see the constant's own comment for how it was verified).
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()

    wrist_roll_follower_geom = None
    for geom in root.iter("geom"):
        if geom.get("mesh") == "wrist_roll_follower_so101_v1":
            wrist_roll_follower_geom = geom
            break
    if wrist_roll_follower_geom is None:
        raise RuntimeError("wrist_roll_follower_so101_v1 geom not found -- can't anchor the camera mount")
    # ElementTree has no getparent(); find gripper_link by scanning bodies instead.
    gripper_link = None
    for body in root.iter("body"):
        if body.get("name") == "gripper_link":
            gripper_link = body
            break
    if gripper_link is None:
        raise RuntimeError("gripper_link body not found")

    base_pos = np.array([float(x) for x in wrist_roll_follower_geom.get("pos").split()])
    base_quat = np.array([float(x) for x in wrist_roll_follower_geom.get("quat").split()])

    extra_quat = np.zeros(4)
    mujoco.mju_euler2Quat(extra_quat, np.radians(WRIST_CAM_MOUNT_EXTRA_RPY_DEG), "XYZ")
    mount_quat = np.zeros(4)
    mujoco.mju_mulQuat(mount_quat, extra_quat, base_quat)  # extra applied AFTER base, in gripper_link's frame
    mount_pos = base_pos + WRIST_CAM_MOUNT_EXTRA_POS

    def rotate(v: np.ndarray) -> np.ndarray:
        out = np.zeros(3)
        mujoco.mju_rotVecQuat(out, v, mount_quat)
        return out

    plate_centroid_local = compute_wrist_cam_plate_centroid_local(mount_stl_path)
    plate_pos_gripper = rotate(plate_centroid_local) + mount_pos
    outward_normal_gripper = rotate(WRIST_CAM_PLATE_NORMAL_LOCAL)
    outward_normal_gripper /= np.linalg.norm(outward_normal_gripper)

    cam_center = plate_pos_gripper + outward_normal_gripper * CAMERA_FORWARD_BIAS

    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(outward_normal_gripper, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, outward_normal_gripper)

    cam_body_quat = np.zeros(4)
    mujoco.mju_mat2Quat(cam_body_quat, np.column_stack([right, up, outward_normal_gripper]).flatten())
    cam_quat = np.zeros(4)
    mujoco.mju_mat2Quat(cam_quat, np.column_stack([right, up, -outward_normal_gripper]).flatten())
    lens_pos = cam_center + outward_normal_gripper * (CAMERA_CYLINDER_HALF_LENGTH + 0.003)

    asset_el = root.find("asset")
    ET.SubElement(
        asset_el,
        "mesh",
        {
            "name": "wrist_cam_mount",
            "file": WRIST_CAM_MOUNT_FILENAME,
            "scale": "0.001 0.001 0.001",
            "content_type": "model/stl",
        },
    )

    def fmt(v) -> str:
        return " ".join(f"{x:.6f}" for x in v)

    bracket_geom = ET.Element(
        "geom",
        {
            "pos": fmt(mount_pos),
            "quat": fmt(mount_quat),
            "type": "mesh",
            "rgba": "0.05 0.05 0.05 1",
            "mesh": "wrist_cam_mount",
        },
    )
    camera_body_geom = ET.Element(
        "geom",
        {
            "type": "cylinder",
            "pos": fmt(cam_center),
            "quat": fmt(cam_body_quat),
            "size": f"{CAMERA_CYLINDER_RADIUS} {CAMERA_CYLINDER_HALF_LENGTH}",
            "rgba": "0.08 0.08 0.08 1",  # distinct dark color from the (yellow) bracket
        },
    )
    camera_el = ET.Element(
        "camera",
        {"name": "wrist", "pos": fmt(lens_pos), "quat": fmt(cam_quat), "fovy": "60"},
    )
    gripper_link.append(bracket_geom)
    gripper_link.append(camera_body_geom)
    gripper_link.append(camera_el)

    ET.indent(tree, space="  ")
    tree.write(mjcf_path, xml_declaration=True, encoding="utf-8")


def make_meshdir_include_safe(mjcf_path: Path) -> None:
    """Bake the compiler's meshdir into every mesh's file path, then drop it.

    Empirically confirmed (see conversation history / PLAN.md Phase 1 notes):
    when this file is spliced into a scene via MJCF's <include>, the merged
    document's <compiler meshdir="..."> is NOT honored for resolving <mesh
    file="..."> paths -- only the literal file="..." string is used, resolved
    relative to the file that defines the <mesh> element. Standalone loading
    (mj_loadXML on this file directly) DOES honor meshdir. Baking meshdir into
    each file path and then removing it makes both loading paths resolve to
    the identical final path, so this file works loaded either way.
    """
    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    compiler_el = root.find("compiler")
    meshdir = compiler_el.get("meshdir", "") if compiler_el is not None else ""
    if not meshdir:
        return
    for mesh_el in root.iter("mesh"):
        fn = mesh_el.get("file")
        if fn:
            mesh_el.set("file", str(Path(meshdir) / fn))
    if "meshdir" in compiler_el.attrib:
        del compiler_el.attrib["meshdir"]
    ET.indent(tree, space="  ")
    tree.write(mjcf_path, xml_declaration=True, encoding="utf-8")


def name2id(model, objtype, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, objtype, name)
    if obj_id < 0:
        raise RuntimeError(f"'{name}' not found in compiled model (got fused away or renamed?)")
    return obj_id


def report(mjcf_path: Path, manifest: dict) -> None:
    model = mujoco.MjModel.from_xml_path(str(mjcf_path))
    data = mujoco.MjData(model)

    print("\n=== Smoke test: model loaded OK ===")
    print(f"nq={model.nq} nv={model.nv} nbody={model.nbody} nu={model.nu}")
    expected_bodies = set(manifest["links"])
    actual_bodies = {model.body(i).name for i in range(model.nbody)}
    missing_bodies = expected_bodies - actual_bodies
    if missing_bodies:
        raise RuntimeError(f"Bodies missing from compiled model (fused away?): {missing_bodies}")

    print("\n=== Joint limits (compiled MJCF, radians) ===")
    for jname in CANONICAL_JOINTS:
        jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        lo, hi = model.jnt_range[jid]
        upstream = manifest["joints"][jname]
        match = "OK" if abs(lo - upstream["lower"]) < 1e-4 and abs(hi - upstream["upper"]) < 1e-4 else "MISMATCH"
        print(f"  {jname:16s} mjcf=({lo:+.5f}, {hi:+.5f})  urdf=({upstream['lower']:+.5f}, {upstream['upper']:+.5f})  [{match}]")

    # Derived reach: grid-search shoulder_lift/elbow_flex/wrist_flex (the
    # joints whose axes are perpendicular to shoulder_pan, i.e. the ones that
    # actually change horizontal extension) at shoulder_pan=0, and measure
    # the max horizontal (xy) distance from the shoulder_pan joint's own
    # position to the gripper_frame_link site -- i.e. the max reach across
    # the arm's real range of motion, not just one hand-picked pose.
    pan_id = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_pan")
    lift_id = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "shoulder_lift")
    elbow_id = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "elbow_flex")
    wrist_id = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_flex")
    gripper_body_id = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_frame_link")
    shoulder_body_id = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "shoulder_link")

    import numpy as np

    n_samples = 15
    lift_range = np.linspace(*model.jnt_range[lift_id], n_samples)
    elbow_range = np.linspace(*model.jnt_range[elbow_id], n_samples)
    wrist_range = np.linspace(*model.jnt_range[wrist_id], n_samples)

    data.qpos[model.jnt_qposadr[pan_id]] = 0.0
    max_reach = 0.0
    best_pose = None
    for lift in lift_range:
        for elbow in elbow_range:
            for wrist in wrist_range:
                data.qpos[model.jnt_qposadr[lift_id]] = lift
                data.qpos[model.jnt_qposadr[elbow_id]] = elbow
                data.qpos[model.jnt_qposadr[wrist_id]] = wrist
                mujoco.mj_kinematics(model, data)
                shoulder_pos = data.xpos[shoulder_body_id]
                gripper_pos = data.xpos[gripper_body_id]
                horiz = np.linalg.norm((gripper_pos - shoulder_pos)[:2])
                if horiz > max_reach:
                    max_reach = horiz
                    best_pose = (lift, elbow, wrist)

    print("\n=== Derived horizontal reach ===")
    print(f"  max horizontal reach (shoulder_link -> gripper_frame_link): {max_reach * 100:.1f} cm")
    print(f"  at (shoulder_lift={best_pose[0]:+.3f}, elbow_flex={best_pose[1]:+.3f}, wrist_flex={best_pose[2]:+.3f}) rad")
    print("  (plan's rough estimate: ~30-35cm -- sanity-check against that, not a hard target)")

    print(f"\n=== canonical_joints_matched: {manifest['canonical_joints_matched']} ===")
    print(f"=== fk_frame_present ({EXPECTED_FK_FRAME}): {manifest['fk_frame_present']} ===")

    cam_id = name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist")
    mujoco.mj_forward(model, data)
    cam_pos = data.cam_xpos[cam_id]
    frame_id = name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_frame_link")
    grasp_pos = data.xpos[frame_id]
    cam_forward_world = -data.cam_xmat[cam_id].reshape(3, 3)[:, 2]
    to_grasp_world = grasp_pos - cam_pos
    to_grasp_world /= np.linalg.norm(to_grasp_world)
    aim_quality = float(np.dot(cam_forward_world, to_grasp_world))
    print("\n=== Wrist camera ===")
    print(f"  aim quality (dot of view dir with direction-to-gripper_frame_link, 1.0=perfect): {aim_quality:.3f}")
    # Won't be near 1.0: the camera looks straight through the bracket's
    # real hole, whichever direction that physically faces (see
    # WRIST_CAM_PLATE_NORMAL_LOCAL) -- not a direction chosen for best aim.
    # Expected around +0.49 (a real, if not perfectly boresighted, "toward
    # the workspace" direction); a negative value would mean the wrong one
    # of the plate's two opposite face normals got used.
    if aim_quality < 0.0:
        print("  WARNING: wrist camera is facing away from the gripper's TCP frame -- check WRIST_CAM_PLATE_NORMAL_LOCAL's sign.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("/tmp/so101_urdf_fetch"),
        help="scratch dir for the raw bucket sync (default: /tmp/so101_urdf_fetch)",
    )
    args = parser.parse_args()

    fetch(args.dest)
    intermediate_urdf, manifest = prepare_urdf_for_mujoco(args.dest)
    mjcf_path = convert_to_mjcf(intermediate_urdf)
    add_actuators(mjcf_path)
    set_arm_mount_pos(mjcf_path)
    recolor_moving_jaw(mjcf_path)
    add_convex_decompositions(mjcf_path, ASSET_DIR / "meshes")
    add_self_collision_exclusions(mjcf_path)
    mount_stl_path = fetch_wrist_cam_mount()
    add_wrist_camera(mjcf_path, mount_stl_path)
    make_meshdir_include_safe(mjcf_path)
    intermediate_urdf.unlink()  # scratch file, not a deliverable

    missing = [j for j in CANONICAL_JOINTS if j not in manifest["joints"]]
    if missing:
        print(f"\nWARNING: canonical joints missing from upstream URDF: {missing}")
    if not manifest["fk_frame_present"]:
        print(f"\nWARNING: expected FK frame '{EXPECTED_FK_FRAME}' not found in fetched URDF.")

    report(mjcf_path, manifest)

    print(f"\nDone. Wrote {mjcf_path} and {len(manifest['meshes_copied'])} meshes to {ASSET_DIR / 'meshes'}.")


if __name__ == "__main__":
    main()
