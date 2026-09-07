"""Version migration, immutable vendor inputs, and head/hand pose contracts."""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from greenhouse_sim import robot_hardware, robot_kinematics, robot_model, robot_scene


def test_one_v12_source_for_kinematics_asset_and_camera_resolution():
    root = robot_model.validate_urdf()
    assert root.attrib["name"] == "RBY1_A_v1.2"
    assert len(root.findall("link")) == 32
    assert robot_kinematics.DEFAULT_URDF == robot_model.DEFAULT_URDF
    assert robot_scene.DEFAULT_ROBOT_ASSET == robot_model.DEFAULT_ASSET
    assert robot_hardware.ROBOT_ROOT == "/RBY1_A_v1_2"
    assert robot_model.D405_RESOLUTION == (848, 408)
    lower, upper = robot_model.joint_limits_degrees(("head_0", "head_1"))
    np.testing.assert_allclose(np.radians(lower), [-1.57, -1.57])
    np.testing.assert_allclose(np.radians(upper), [1.57, 1.57])


def test_all_link_preview_matches_arm_fk_and_moves_head():
    model = robot_kinematics.Rby1Kinematics()
    pose = robot_scene.SDK_READY_POSE_DEGREES
    links = model.all_link_transforms(pose)
    assert len(links) == 32
    for side in ("left", "right"):
        arm = [pose[f"{side}_arm_{i}"] for i in range(7)]
        np.testing.assert_allclose(links[f"ee_{side}"], model.forward(side, arm, np.eye(4)), atol=1e-12)
    tilted = model.all_link_transforms({**pose, "head_0": 70, "head_1": -40})
    assert not np.allclose(tilted["link_head_2"], links["link_head_2"])
    np.testing.assert_allclose(tilted["ee_left"], links["ee_left"], atol=1e-12)
    with pytest.raises(ValueError, match="limits"):
        model.all_link_transforms({**pose, "head_0": 100})
    opened = model.all_link_transforms(pose, prismatic_m={"gripper_finger_l1": -.04, "gripper_finger_l2": .04})
    assert np.linalg.norm(opened["ee_finger_l1"][:3, 3] - opened["ee_finger_l2"][:3, 3]) > .08


def test_new_bracket_sources_match_manifest_hashes():
    manifest = robot_hardware.load_manifest()
    for name in ("camera_bracket_d405", "head_camera_bracket_d405"):
        part = robot_hardware._part(manifest, name)
        assert hashlib.sha256((robot_hardware.ASSET_DIR / part["source"]).read_bytes()).hexdigest() == part["source_sha256"]
    assert robot_hardware._part(manifest, "camera_bracket_d405")["source"] == "D405_Wrist_Bracket_v2-Body.stl"
    assert robot_hardware._part(manifest, "head_camera_bracket_d405")["source"] == "HeadCam_Bracket_D405.FCStd"


def test_import_aliases_preserve_mesh_bytes_and_all_joints(tmp_path):
    import build_robot
    before = robot_model.DEFAULT_URDF.read_bytes()
    prepared = build_robot._prepare_import_urdf(robot_model.DEFAULT_URDF, tmp_path / "aliases")
    source = ET.fromstring(before)
    imported = ET.parse(prepared).getroot()
    assert robot_model.DEFAULT_URDF.read_bytes() == before
    assert [ET.tostring(j) for j in source.findall("joint")] == [ET.tostring(j) for j in imported.findall("joint")]
    for old, new in zip(source.findall(".//mesh"), imported.findall(".//mesh"), strict=True):
        from pathlib import Path
        original = robot_model.DEFAULT_URDF.parent / old.attrib["filename"]
        alias = Path(new.attrib["filename"])
        assert alias.read_bytes() == original.read_bytes()
        assert "." not in alias.stem
