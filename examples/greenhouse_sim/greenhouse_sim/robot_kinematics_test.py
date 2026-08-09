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


def test_point_axes_ik_rejects_invalid_position_scale() -> None:
    model = robot_kinematics.Rby1Kinematics()
    with pytest.raises(ValueError, match="position_scale_m"):
        model.solve_position_axes(
            "left",
            local_point_m=(0.0, 0.0, -0.1),
            target_point_m=(0.0, 0.0, 0.0),
            seed_degrees=(0.0,) * 7,
            base_matrix=BASE,
            pointing_axis=2,
            pointing_direction=(0.0, 1.0, 0.0),
            transverse_axis=0,
            transverse_to=(1.0, 0.0, 0.0),
            position_scale_m=0.0,
        )


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


def test_sphere_capsule_clearance_is_signed_and_reports_nearest() -> None:
    obstacles = (
        robot_kinematics.CapsuleObstacle("far", (2.0, 0.0, 0.0), (2.0, 0.0, 1.0), 0.1),
        robot_kinematics.CapsuleObstacle("near", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0), 0.1),
    )

    clear = robot_kinematics.sphere_capsule_clearance((0.5, 0.0, 0.0), 0.2, obstacles)
    overlap = robot_kinematics.sphere_capsule_clearance((0.2, 0.0, 0.0), 0.2, obstacles)

    assert clear.nearest_obstacle == "near"
    assert clear.clearance_m == pytest.approx(0.2)
    assert overlap.clearance_m == pytest.approx(-0.1)


def test_oriented_box_clearance_accounts_for_projected_half_extent() -> None:
    obstacle = robot_kinematics.CapsuleObstacle(
        "stem", (1.0, 0.0, -1.0), (1.0, 0.0, 1.0), 0.1
    )
    result = robot_kinematics.oriented_box_capsule_clearance(
        (0.0, 0.0, 0.0),
        np.eye(3),
        (0.25, 0.5, 0.1),
        (obstacle,),
    )

    assert result.nearest_obstacle == "stem"
    assert result.clearance_m == pytest.approx(0.65)


def test_oriented_box_clearance_uses_entire_capsule_segment() -> None:
    obstacle = robot_kinematics.CapsuleObstacle(
        "diagonal",
        (-2.0, 0.6, 0.0),
        (2.0, 0.6, 0.0),
        0.05,
    )
    result = robot_kinematics.oriented_box_capsule_clearance(
        (0.0, 0.0, 0.0),
        np.eye(3),
        (0.25, 0.5, 0.1),
        (obstacle,),
    )

    assert result.nearest_obstacle == "diagonal"
    assert result.clearance_m == pytest.approx(0.05)


def test_tool_box_clearance_transforms_box_from_end_effector_frame() -> None:
    class TranslatedTool:
        @staticmethod
        def forward(side, arm_degrees, base_matrix):
            del side, arm_degrees, base_matrix
            transform = np.eye(4)
            transform[:3, 3] = (1.0, 2.0, 3.0)
            return transform

    obstacle = robot_kinematics.CapsuleObstacle(
        "stem",
        (1.5, 2.0, 2.0),
        (1.5, 2.0, 4.0),
        0.05,
    )
    result = robot_kinematics.tool_box_clearance(
        TranslatedTool(),
        "right",
        np.zeros(7),
        np.eye(4),
        (0.1, 0.0, 0.0),
        np.eye(3),
        (0.1, 0.2, 0.3),
        (obstacle,),
    )

    assert result.nearest_obstacle == "stem"
    assert result.clearance_m == pytest.approx(0.25)


def _route(name, sign, minimum, mean, lateral):
    return {
        "name": name,
        "x_sign": sign,
        "lateral_distance_m": lateral,
        "lift_distance_m": 0.08,
        "minimum_clearance": {"clearance_m": minimum},
        "mean_clearance_m": mean,
        "feasible": True,
        "solutions": [object()],
        "offsets": [object()],
    }


def test_route_selector_chooses_direction_before_widest_tied_route() -> None:
    candidates = [
        _route("negative", -1.0, -0.006, 0.030, 0.04),
        _route("positive_short", 1.0, -0.006, 0.016, 0.04),
        _route("positive_wide", 1.0, -0.006, 0.003, 0.12),
    ]

    selected = robot_kinematics.select_tool_clearance_route(candidates)

    assert selected["name"] == "negative"


def test_route_selector_maximizes_separation_inside_safe_direction() -> None:
    candidates = [
        _route("negative", -1.0, -0.008, 0.030, 0.04),
        _route("positive_short", 1.0, -0.004, 0.012, 0.04),
        _route("positive_wide", 1.0, -0.004, 0.007, 0.12),
    ]

    selected = robot_kinematics.select_tool_clearance_route(candidates)

    assert selected["name"] == "positive_wide"
