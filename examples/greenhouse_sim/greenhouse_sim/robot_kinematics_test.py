"""Regressions for exact RB-Y1 v1.0 benchmark kinematics."""

from __future__ import annotations

import numpy as np
import pytest

from greenhouse_sim import robot_kinematics


BASE = robot_kinematics.base_transform((6.99114, 3.78, -0.3050817), -90.0)
RIGHT_SAFE = (-101.724, -83.623, 34.196, -135.683, -57.431, 94.832, -74.920)


def test_forward_kinematics_matches_accepted_greenhouse_knife_root() -> None:
    model = robot_kinematics.Rby1Kinematics()
    ee = model.forward("right", RIGHT_SAFE, BASE)

    np.testing.assert_allclose(ee[:3, 3], [6.80511796, 3.49999921, 1.08376004], atol=1e-7)


def test_pose_ik_recovers_a_nearby_exact_arm_pose() -> None:
    model = robot_kinematics.Rby1Kinematics()
    expected = np.asarray((-89.85, -79.057, 44.849, -136.448, -22.506, 85.097, -81.711))
    desired = model.forward("right", expected, BASE)
    result = model.solve_pose("right", desired, RIGHT_SAFE, BASE)

    assert result.succeeded
    assert result.position_error_m < 1e-6
    assert result.orientation_error_rad < 1e-4
    np.testing.assert_allclose(
        model.forward("right", result.joint_degrees, BASE),
        desired,
        atol=1e-5,
    )


def test_position_ik_places_tool_point_without_orientation_constraint() -> None:
    model = robot_kinematics.Rby1Kinematics()
    expected = np.asarray((-95.0, -60.0, 25.0, -110.0, -20.0, 70.0, -30.0))
    local_point = np.asarray((0.0, -0.02, -0.08, 1.0))
    target = (model.forward("right", expected, BASE) @ local_point)[:3]

    result = model.solve_position(
        "right",
        local_point_m=local_point[:3],
        target_point_m=target,
        seed_degrees=RIGHT_SAFE,
        base_matrix=BASE,
    )

    assert result.succeeded
    actual = model.forward("right", result.joint_degrees, BASE) @ local_point
    np.testing.assert_allclose(actual[:3], target, atol=1e-3)


def test_point_force_capacity_scales_utilization_not_capacity() -> None:
    model = robot_kinematics.Rby1Kinematics()
    joints = np.asarray((-75.0, 5.0, -30.0, -90.0, 20.0, 60.0, 10.0))
    limits = (70.0, 70.0, 70.0, 40.0, 10.0, 10.0, 8.0)
    arguments = (
        "left",
        joints,
        BASE,
        (0.0, 0.0, -0.1025),
        (0.2, -0.9, 0.4),
    )

    one = model.point_force_capacity(*arguments, 40.0, limits)
    two = model.point_force_capacity(*arguments, 80.0, limits)

    assert one.force_capacity_n == two.force_capacity_n
    np.testing.assert_allclose(
        np.asarray(two.joint_utilization),
        2.0 * np.asarray(one.joint_utilization),
    )
    assert one.force_capacity_n > 0.0


def test_point_force_capacity_rejects_invalid_inputs() -> None:
    model = robot_kinematics.Rby1Kinematics()
    joints = np.zeros(7)

    with pytest.raises(ValueError, match="non-zero"):
        model.point_force_capacity(
            "left", joints, BASE, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, np.ones(7)
        )
    with pytest.raises(ValueError, match="seven positive"):
        model.point_force_capacity(
            "left", joints, BASE, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), 1.0, np.ones(6)
        )
