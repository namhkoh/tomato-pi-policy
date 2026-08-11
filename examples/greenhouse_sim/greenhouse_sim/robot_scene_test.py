"""Tests for fitted RB-Y1 greenhouse placement and startup state."""

from __future__ import annotations

import numpy as np

from greenhouse_sim import robot_scene


def test_ready_pose_matches_official_model_a_vector() -> None:
    assert [robot_scene.SDK_READY_POSE_DEGREES[f"torso_{index}"] for index in range(6)] == [
        0.0,
        45.0,
        -90.0,
        45.0,
        0.0,
        0.0,
    ]
    assert [robot_scene.SDK_READY_POSE_DEGREES[f"right_arm_{index}"] for index in range(7)] == [
        0.0,
        -5.0,
        0.0,
        -120.0,
        0.0,
        70.0,
        0.0,
    ]
    assert [robot_scene.SDK_READY_POSE_DEGREES[f"left_arm_{index}"] for index in range(7)] == [
        0.0,
        5.0,
        0.0,
        -120.0,
        0.0,
        70.0,
        0.0,
    ]


def test_greenhouse_pose_replaces_only_right_arm_for_safe_precontact() -> None:
    for name, expected in robot_scene.SDK_READY_POSE_DEGREES.items():
        if not name.startswith("right_arm_"):
            assert robot_scene.READY_POSE_DEGREES[name] == expected
    assert [robot_scene.READY_POSE_DEGREES[f"right_arm_{index}"] for index in range(7)] == list(
        robot_scene.GREENHOUSE_PRECONTACT_RIGHT_ARM_DEGREES
    )
    assert [robot_scene.READY_POSE_DEGREES[f"left_arm_{index}"] for index in range(7)] == [
        robot_scene.SDK_READY_POSE_DEGREES[f"left_arm_{index}"] for index in range(7)
    ]


def test_default_pose_faces_the_robot_toward_the_vine_row() -> None:
    yaw = np.deg2rad(robot_scene.DEFAULT_YAW_DEGREES)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(robot_scene.DEFAULT_POSITION_M[:2], [6.99114, 3.93])
    target_gutter_back = 3.256061
    assert robot_scene.DEFAULT_POSITION_M[1] - 0.295 - target_gutter_back > 0.35


def test_generated_robot_references_with_ready_state_and_hardware() -> None:
    import pytest

    try:
        from pxr import PhysxSchema
    except ImportError:
        pytest.skip("PhysxSchema requires a running SimulationApp")
    from pxr import Usd
    from pxr import UsdGeom
    from pxr import UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    placement = robot_scene.add_fitted_robot(stage)

    assert placement.pose_name == robot_scene.DEFAULT_POSE_NAME
    assert placement.right_gripper_removed
    assert len(placement.initialized_joints) == 22
    assert len(placement.cameras) == 3
    assert all(stage.GetPrimAtPath(path).IsA(UsdGeom.Camera) for path in placement.cameras)
    base_anchor = UsdPhysics.FixedJoint.Get(
        stage, f"{placement.root_path}/joints/benchmark_world_fixed"
    )
    assert base_anchor
    assert [str(path) for path in base_anchor.GetBody1Rel().GetTargets()] == [
        f"{placement.root_path}/base"
    ]
    for name, expected in robot_scene.READY_POSE_DEGREES.items():
        joint = stage.GetPrimAtPath(f"{placement.root_path}/joints/{name}")
        state = PhysxSchema.JointStateAPI.Get(joint, "angular")
        assert state.GetPositionAttr().Get() == expected
    assert not stage.GetPrimAtPath(placement.knife_blade).GetAttribute("tomato:cuttingSurface").Get()
    assert stage.GetPrimAtPath(placement.knife_cutting_edge).GetAttribute(
        "tomato:cuttingSurface"
    ).Get()
    assert not stage.GetPrimAtPath(placement.knife_support).GetAttribute("tomato:cuttingSurface").Get()

    # Imported collision instance proxies must remain inactive; the sibling
    # restored capsule is the single authoritative wrist contact shape.
    for side in ("right", "left"):
        link = f"{placement.root_path}/link_{side}_arm_5"
        imported = stage.GetPrimAtPath(f"{link}/collisions")
        assert imported.IsValid() and not imported.IsActive()
        capsule = UsdGeom.Capsule.Get(stage, f"{link}/restored_collisions/capsule_00")
        assert capsule.GetRadiusAttr().Get() == 0.075
        assert capsule.GetHeightAttr().Get() == 0.052
        assert stage.GetPrimAtPath(str(capsule.GetPath())).GetAttribute(
            "xformOp:transform"
        ).Get().ExtractTranslation()[2] == -0.024

    palm = stage.GetPrimAtPath(f"{placement.root_path}/ee_left/restored_collisions/contact_proxy")
    assert palm.GetAttribute("xformOp:scale").Get()[2] == 0.025
    assert palm.GetAttribute("xformOp:translate").Get()[2] == -0.0125
