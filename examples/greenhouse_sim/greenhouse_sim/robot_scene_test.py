"""Tests for fitted RB-Y1 greenhouse placement and startup state."""

from __future__ import annotations

import numpy as np

from greenhouse_sim import robot_scene


def test_ready_pose_matches_official_model_a_vector() -> None:
    assert [robot_scene.READY_POSE_DEGREES[f"torso_{index}"] for index in range(6)] == [
        0.0,
        45.0,
        -90.0,
        45.0,
        0.0,
        0.0,
    ]
    assert [robot_scene.READY_POSE_DEGREES[f"right_arm_{index}"] for index in range(7)] == [
        0.0,
        -5.0,
        0.0,
        -120.0,
        0.0,
        70.0,
        0.0,
    ]
    assert [robot_scene.READY_POSE_DEGREES[f"left_arm_{index}"] for index in range(7)] == [
        0.0,
        5.0,
        0.0,
        -120.0,
        0.0,
        70.0,
        0.0,
    ]


def test_default_pose_faces_the_robot_toward_the_vine_row() -> None:
    yaw = np.deg2rad(robot_scene.DEFAULT_YAW_DEGREES)
    rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(rotation @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    assert robot_scene.DEFAULT_POSITION_M[1] < 3.0922


def test_generated_robot_references_with_ready_state_and_hardware() -> None:
    import pytest

    try:
        from pxr import PhysxSchema
    except ImportError:
        pytest.skip("PhysxSchema requires a running SimulationApp")
    from pxr import Usd
    from pxr import UsdGeom

    stage = Usd.Stage.CreateInMemory()
    placement = robot_scene.add_fitted_robot(stage)

    assert len(placement.initialized_joints) == 22
    assert len(placement.cameras) == 3
    assert all(stage.GetPrimAtPath(path).IsA(UsdGeom.Camera) for path in placement.cameras)
    for name, expected in robot_scene.READY_POSE_DEGREES.items():
        joint = stage.GetPrimAtPath(f"{placement.root_path}/joints/{name}")
        state = PhysxSchema.JointStateAPI.Get(joint, "angular")
        assert state.GetPositionAttr().Get() == expected
    assert stage.GetPrimAtPath(placement.knife_blade).GetAttribute("tomato:cuttingSurface").Get()
    assert not stage.GetPrimAtPath(placement.knife_support).GetAttribute("tomato:cuttingSurface").Get()
