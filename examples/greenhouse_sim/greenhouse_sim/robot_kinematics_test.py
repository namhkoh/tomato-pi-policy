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


def test_arm_joint_limit_margin_reports_nearest_authored_limit() -> None:
    model = robot_kinematics.Rby1Kinematics()
    lower, upper = model.arm_limits_degrees("right")
    centre = 0.5 * (lower + upper)
    near_upper = centre.copy()
    near_upper[1] = upper[1] - 12.5

    assert model.arm_joint_limit_margin_degrees("right", centre) > 12.5
    assert model.arm_joint_limit_margin_degrees("right", near_upper) == pytest.approx(12.5)
    with pytest.raises(ValueError, match="seven"):
        model.arm_joint_limit_margin_degrees("right", near_upper[:6])


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


def test_signed_transverse_direction_returns_opposed_jaw_closing_axes() -> None:
    positive = robot_kinematics.signed_transverse_direction(
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        1.0,
    )
    negative = robot_kinematics.signed_transverse_direction(
        (0.0, -1.0, 0.0),
        (0.0, 0.0, 1.0),
        -1.0,
    )

    np.testing.assert_allclose(positive, (1.0, 0.0, 0.0), atol=1e-12)
    np.testing.assert_allclose(negative, -positive, atol=1e-12)
    with pytest.raises(ValueError, match="parallel"):
        robot_kinematics.signed_transverse_direction(
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0),
        )


def test_rotate_horizontal_direction_applies_signed_world_yaw() -> None:
    np.testing.assert_allclose(
        robot_kinematics.rotate_horizontal_direction((0.0, -2.0, 0.0), 90.0),
        (1.0, 0.0, 0.0),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        robot_kinematics.rotate_horizontal_direction((0.0, -2.0, 0.0), -90.0),
        (-1.0, 0.0, 0.0),
        atol=1e-12,
    )


def test_point_axes_ik_honours_explicit_jaw_closing_direction() -> None:
    model = robot_kinematics.Rby1Kinematics()
    expected = np.asarray(
        (-120.779, 0.740, -20.440, -74.103, -5.287, 109.523, -3.747)
    )
    expected_pose = model.forward("left", expected, BASE)
    local_point = np.asarray((0.0, 0.0, -0.1025, 1.0))
    target = (expected_pose @ local_point)[:3]

    result = model.solve_position_axes(
        "left",
        local_point_m=local_point[:3],
        target_point_m=target,
        seed_degrees=expected,
        base_matrix=BASE,
        pointing_axis=2,
        pointing_direction=expected_pose[:3, 2],
        transverse_axis=0,
        transverse_to=expected_pose[:3, 1],
        transverse_direction=expected_pose[:3, 0],
    )

    assert result.succeeded
    actual = model.forward("left", result.joint_degrees, BASE)
    assert np.dot(actual[:3, 0], expected_pose[:3, 0]) > 0.999
    np.testing.assert_allclose((actual @ local_point)[:3], target, atol=1e-3)


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
    with pytest.raises(ValueError, match="maximum_evaluations"):
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
            maximum_evaluations=0,
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


def test_oriented_box_clearance_rejects_greenhouse_wall_overlap() -> None:
    wall = robot_kinematics.BoxObstacle(
        "greenhouse_wall",
        (1.0, -2.0, -2.0),
        (1.1, 2.0, 2.0),
    )

    clear = robot_kinematics.oriented_box_box_clearance(
        (0.0, 0.0, 0.0), np.eye(3), (0.25, 0.5, 0.1), (wall,)
    )
    overlap = robot_kinematics.oriented_box_box_clearance(
        (0.9, 0.0, 0.0), np.eye(3), (0.25, 0.5, 0.1), (wall,)
    )

    assert clear.nearest_obstacle == "greenhouse_wall"
    assert clear.clearance_m == pytest.approx(0.75)
    assert overlap.clearance_m < 0.0


def test_oriented_foliage_box_avoids_inflated_aabb_corner_rejection() -> None:
    angle = np.radians(45.0)
    rotation = (
        (float(np.cos(angle)), float(-np.sin(angle)), 0.0),
        (float(np.sin(angle)), float(np.cos(angle)), 0.0),
        (0.0, 0.0, 1.0),
    )
    foliage = robot_kinematics.OrientedBoxObstacle(
        "leaf",
        (0.0, 0.0, 0.0),
        rotation,
        (1.0, 0.01, 0.10),
    )
    inflated = np.sqrt(0.5) * 1.01
    foliage_aabb = robot_kinematics.BoxObstacle(
        "leaf_aabb",
        (-inflated, -inflated, -0.10),
        (inflated, inflated, 0.10),
    )

    exact = robot_kinematics.oriented_box_oriented_box_clearance(
        (0.0, 0.50, 0.0),
        np.eye(3),
        (0.05, 0.05, 0.05),
        (foliage,),
    )
    conservative_aabb = robot_kinematics.oriented_box_box_clearance(
        (0.0, 0.50, 0.0),
        np.eye(3),
        (0.05, 0.05, 0.05),
        (foliage_aabb,),
    )

    assert exact.clearance_m > 0.20
    assert conservative_aabb.clearance_m < 0.0


def test_capsule_and_sphere_clear_exact_oriented_foliage_box() -> None:
    angle = np.radians(45.0)
    rotation = (
        (float(np.cos(angle)), float(-np.sin(angle)), 0.0),
        (float(np.sin(angle)), float(np.cos(angle)), 0.0),
        (0.0, 0.0, 1.0),
    )
    foliage = robot_kinematics.OrientedBoxObstacle(
        "leaf",
        (0.0, 0.0, 0.0),
        rotation,
        (1.0, 0.01, 0.10),
    )
    capsule = robot_kinematics.CapsuleObstacle(
        "arm",
        (-0.1, 0.50, 0.0),
        (0.1, 0.50, 0.0),
        0.05,
    )

    assert robot_kinematics.capsule_oriented_box_clearance(
        capsule, foliage
    ) > 0.15
    assert robot_kinematics.sphere_oriented_box_clearance(
        (0.0, 0.50, 0.0), 0.05, (foliage,)
    ).clearance_m > 0.20


def test_capsule_clearance_rejects_greenhouse_box_overlap() -> None:
    box = robot_kinematics.BoxObstacle(
        "gutter", (-0.5, -0.5, -0.5), (0.5, 0.5, 0.5)
    )
    clear = robot_kinematics.CapsuleObstacle(
        "arm", (-1.0, 1.0, 0.0), (1.0, 1.0, 0.0), 0.1
    )
    overlap = robot_kinematics.CapsuleObstacle(
        "arm", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.1
    )

    assert robot_kinematics.capsule_box_clearance(clear, box) == pytest.approx(0.4)
    assert robot_kinematics.capsule_box_clearance(overlap, box) < 0.0


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


def test_capsule_capsule_clearance_is_signed() -> None:
    first = robot_kinematics.CapsuleObstacle(
        "first", (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 0.1
    )
    separated = robot_kinematics.CapsuleObstacle(
        "separated", (0.0, 0.5, -1.0), (0.0, 0.5, 1.0), 0.1
    )
    crossing = robot_kinematics.CapsuleObstacle(
        "crossing", (0.0, 0.0, -1.0), (0.0, 0.0, 1.0), 0.1
    )

    assert robot_kinematics.capsule_capsule_clearance(first, separated) == pytest.approx(0.3)
    assert robot_kinematics.capsule_capsule_clearance(first, crossing) == pytest.approx(-0.2)


def test_vectorized_segment_distances_match_scalar_kernel() -> None:
    rng = np.random.default_rng(7)
    first_start = rng.normal(size=3)
    first_end = rng.normal(size=3)
    second_starts = rng.normal(size=(32, 3))
    second_ends = rng.normal(size=(32, 3))
    second_ends[0] = second_starts[0]

    actual = robot_kinematics._segment_to_segments_distances(
        first_start,
        first_end,
        second_starts,
        second_ends,
    )
    expected = np.asarray(
        [
            robot_kinematics._segment_segment_distance(
                first_start,
                first_end,
                start,
                end,
            )
            for start, end in zip(second_starts, second_ends, strict=True)
        ]
    )

    np.testing.assert_allclose(actual, expected, atol=1e-12)

def test_arm_obstacle_clearance_detects_a_vine_capsule_overlap() -> None:
    model = robot_kinematics.Rby1Kinematics()
    left_ready = (0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0)
    arm_capsule = model.arm_capsules("left", left_ready, BASE)[-1]

    result = model.arm_obstacle_clearance(
        "left", left_ready, BASE, (arm_capsule,)
    )

    assert result.clearance_m < 0.0
    assert result.nearest_obstacle is not None
    assert "link_left_arm_" in result.nearest_obstacle


def test_fixed_body_oriented_box_clearance_detects_foliage_overlap() -> None:
    model = robot_kinematics.Rby1Kinematics()
    capsule = model.fixed_body_capsules(BASE)[0]
    centre = 0.5 * (
        np.asarray(capsule.start_m) + np.asarray(capsule.end_m)
    )
    foliage = robot_kinematics.OrientedBoxObstacle(
        path="foliage",
        centre_m=tuple(centre),
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        half_extents_m=(0.001, 0.001, 0.001),
    )

    result = model.fixed_body_oriented_box_clearance(BASE, (foliage,))

    assert result.clearance_m < 0.0
    assert result.nearest_obstacle is not None
    assert "link_torso_" in result.nearest_obstacle
    assert "foliage" in result.nearest_obstacle


def test_inter_arm_clearance_accepts_ready_pose_and_rejects_overlap() -> None:
    model = robot_kinematics.Rby1Kinematics()
    left_ready = (0.0, 5.0, 0.0, -120.0, 0.0, 70.0, 0.0)
    colliding_left = (
        -85.643995858, 107.071334251, 83.310427291, -82.849954157,
        -36.521995776, 85.314361771, 51.134158417,
    )
    colliding_right = (
        -177.567950194, 98.179681533, -88.657286263, -97.901238505,
        -10.845725360, -1.684610892, -74.568687054,
    )

    ready = model.inter_arm_clearance(left_ready, RIGHT_SAFE, BASE)
    overlap = model.inter_arm_clearance(colliding_left, colliding_right, BASE)

    assert ready.clearance_m > 0.20
    assert overlap.clearance_m < -0.14
    assert overlap.nearest_obstacle is not None
    assert "link_left_arm_" in overlap.nearest_obstacle
    assert "link_right_arm_" in overlap.nearest_obstacle
