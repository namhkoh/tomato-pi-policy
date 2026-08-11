"""Regressions for deterministic target-conditioned base placement."""

from __future__ import annotations

import numpy as np

from greenhouse_sim import base_planner
from greenhouse_sim import robot_kinematics


class _FakeModel:
    def solve_position_axes(self, _side, *, target_point_m, base_matrix, **_kwargs):
        advance = -float(base_matrix[1, 3])
        segment = round(float(target_point_m[2]))
        succeeded = segment < 2 or advance >= 0.03
        return robot_kinematics.IKResult(
            joint_degrees=(float(segment),) + (0.0,) * 6,
            position_error_m=0.0 if succeeded else 0.02,
            orientation_error_rad=0.0,
            cost=0.0,
            succeeded=succeeded,
        )

    def forward(self, _side, arm_degrees, base_matrix):
        matrix = np.asarray(base_matrix, dtype=np.float64).copy()
        matrix[:3, 3] = (1.0, 1.0, float(arm_degrees[0]))
        return matrix

    def fixed_body_clearance(self, _base_matrix, _obstacles):
        return robot_kinematics.ClearanceResult(float("inf"), None)

    def arm_obstacle_clearance(
        self, _side, _arm_degrees, _base_matrix, _obstacles
    ):
        return robot_kinematics.ClearanceResult(float("inf"), None)


def test_planner_advances_until_a_distal_segment_is_reachable() -> None:
    candidates = tuple(
        base_planner.GraspCandidate(
            collider=f"link_{segment}",
            body=f"body_{segment}",
            segment=segment,
            role="petiole_grasp" if segment == 3 else "petiole_cut_zone",
            centre_m=(0.0, -1.0, float(segment)),
            axis=(0.0, 0.0, 1.0),
        )
        for segment in range(4)
    )

    plan = base_planner.plan_target_conditioned_base(
        _FakeModel(),
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=candidates,
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7,),
        minimum_camera_clearance_m=0.0,
    )

    assert plan is not None
    assert plan.advance_m == 0.03
    assert plan.selected_grasp_segment == 3
    np.testing.assert_allclose(plan.position_m, (0.0, -0.03, 0.0))


class _BodyClearanceFakeModel(_FakeModel):
    def fixed_body_clearance(self, base_matrix, _obstacles):
        clearance = abs(float(base_matrix[1, 3])) - 0.1
        return robot_kinematics.ClearanceResult(clearance, "torso/vine")


def test_planner_rejects_base_pose_that_overlaps_vine_body() -> None:
    candidate = base_planner.GraspCandidate(
        collider="link_1",
        body="body_1",
        segment=1,
        role="petiole_grasp",
        centre_m=(0.0, -1.0, 1.0),
        axis=(0.0, 0.0, 1.0),
    )

    plan = base_planner.plan_target_conditioned_base(
        _BodyClearanceFakeModel(),
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=(candidate,),
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7,),
        advances_m=(0.0,),
        lateral_offsets_m=(0.0, 0.3),
        minimum_camera_clearance_m=0.0,
        minimum_body_clearance_m=0.01,
    )

    assert plan is not None
    assert plan.lateral_m == 0.3
    assert np.isclose(plan.body_clearance_m, 0.2)
    assert plan.nearest_body_obstacle == "torso/vine"


class _ArmClearanceFakeModel(_FakeModel):
    def arm_obstacle_clearance(
        self, _side, _arm_degrees, base_matrix, _obstacles
    ):
        clearance = abs(float(base_matrix[1, 3])) - 0.1
        return robot_kinematics.ClearanceResult(clearance, "arm/vine")


def test_planner_rejects_base_pose_with_a_waiting_arm_in_the_vine() -> None:
    candidate = base_planner.GraspCandidate(
        collider="link_1",
        body="body_1",
        segment=1,
        role="petiole_grasp",
        centre_m=(0.0, -1.0, 1.0),
        axis=(0.0, 0.0, 1.0),
    )

    plan = base_planner.plan_target_conditioned_base(
        _ArmClearanceFakeModel(),
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=(candidate,),
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7,),
        advances_m=(0.0,),
        lateral_offsets_m=(0.0, 0.3),
        left_waiting_degrees=(0.0,) * 7,
        minimum_camera_clearance_m=0.0,
        minimum_arm_obstacle_clearance_m=0.01,
    )

    assert plan is not None
    assert plan.lateral_m == 0.3
    np.testing.assert_allclose(plan.position_m, (0.0, 0.3, 0.0))


class _TrajectoryClearanceFakeModel(_FakeModel):
    def solve_position_axes(self, _side, *, target_point_m, **_kwargs):
        return robot_kinematics.IKResult(
            joint_degrees=(float(target_point_m[2]),) + (0.0,) * 6,
            position_error_m=0.0,
            orientation_error_rad=0.0,
            cost=0.0,
            succeeded=True,
        )

    def arm_obstacle_clearance(
        self, _side, arm_degrees, base_matrix, _obstacles
    ):
        progress = float(arm_degrees[0])
        nominal_lane = abs(float(base_matrix[1, 3])) < 0.1
        clearance = -0.02 if nominal_lane and 0.25 < progress < 0.75 else 0.05
        return robot_kinematics.ClearanceResult(clearance, "arm/vine")


def test_planner_rejects_clear_endpoint_with_colliding_approach_chord() -> None:
    candidate = base_planner.GraspCandidate(
        collider="link_1",
        body="body_1",
        segment=1,
        role="petiole_grasp",
        centre_m=(0.0, -1.0, 1.0),
        axis=(0.0, 0.0, 1.0),
    )
    diagnostics = {}

    plan = base_planner.plan_target_conditioned_base(
        _TrajectoryClearanceFakeModel(),
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=(candidate,),
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7,),
        advances_m=(0.0,),
        lateral_offsets_m=(0.0, 0.3),
        left_waiting_degrees=(0.0,) * 7,
        left_approach_start_degrees=(0.0,) * 7,
        minimum_camera_clearance_m=0.0,
        minimum_trajectory_clearance_m=0.01,
        trajectory_samples=5,
        diagnostics=diagnostics,
    )

    assert plan is not None
    assert plan.lateral_m == 0.3
    assert diagnostics["attempts"][0]["grasp_arm_clearance_m"] == 0.05
    assert diagnostics["attempts"][0]["trajectory_arm_clearance_m"] == -0.02

class _PayloadClearanceFakeModel(_FakeModel):
    def forward(self, _side, arm_degrees, base_matrix):
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, 3] = (
            float(base_matrix[0, 3]),
            float(base_matrix[1, 3]),
            float(arm_degrees[0]),
        )
        return matrix


def test_planner_rejects_neighbour_collision_with_finger_payload() -> None:
    candidate = base_planner.GraspCandidate(
        collider="target",
        body="target_body",
        segment=1,
        role="petiole_grasp",
        centre_m=(0.0, -1.0, 1.0),
        axis=(0.0, 0.0, 1.0),
        excluded_finger_colliders=("target",),
    )
    neighbour = robot_kinematics.CapsuleObstacle(
        path="neighbour",
        start_m=(0.0, 0.0, 1.0),
        end_m=(0.0, 0.0, 1.0),
        radius_m=0.03,
    )

    plan = base_planner.plan_target_conditioned_base(
        _PayloadClearanceFakeModel(),
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=(candidate,),
        obstacles=(neighbour,),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 10.0),
        camera_radius_m=0.0,
        left_payload_boxes=((
            "ee_finger_l1",
            (0.0, 0.0, 0.0),
            np.eye(3),
            (0.05, 0.05, 0.05),
        ),),
        seeds=((0.0,) * 7,),
        advances_m=(0.0,),
        lateral_offsets_m=(0.0, 0.3),
        left_waiting_degrees=(0.0,) * 7,
        minimum_camera_clearance_m=0.0,
        minimum_payload_clearance_m=0.01,
    )

    assert plan is not None
    assert plan.lateral_m == 0.3
    assert plan.payload_clearance_m > 0.01
    assert "ee_finger_l1" in plan.nearest_payload_obstacle


def test_joint_space_route_search_goes_around_blocked_direct_chord() -> None:
    def valid(values):
        return float(np.linalg.norm(values)) > 0.30

    route = base_planner._plan_joint_space_route(
        (-0.9, 0.0),
        (0.9, 0.0),
        (-1.0, -1.0),
        (1.0, 1.0),
        valid,
        seed=11,
        max_iterations=2000,
    )

    assert route
    points = [
        np.asarray((-0.9, 0.0)),
        *map(np.asarray, route),
        np.asarray((0.9, 0.0)),
    ]
    for first, second in zip(points, points[1:]):
        for fraction in np.linspace(0.0, 1.0, 51):
            assert valid(first + fraction * (second - first))

class _BimanualFakeModel(_FakeModel):
    def solve_position_axes(self, _side, *, target_point_m, base_matrix, **_kwargs):
        return robot_kinematics.IKResult(
            joint_degrees=(float(target_point_m[2]),) + (0.0,) * 6,
            position_error_m=0.0,
            orientation_error_rad=0.0,
            cost=0.0,
            succeeded=True,
        )

    def inter_arm_clearance(self, _left, _right, base_matrix):
        clearance = -float(base_matrix[1, 3])
        return robot_kinematics.ClearanceResult(clearance, "left/right")


def test_planner_selects_lateral_pose_with_bimanual_clearance() -> None:
    candidate = base_planner.GraspCandidate(
        collider="link_3",
        body="body_3",
        segment=3,
        role="petiole_grasp",
        centre_m=(0.0, -1.0, 3.0),
        axis=(0.0, 0.0, 1.0),
    )

    plan = base_planner.plan_target_conditioned_base(
        _BimanualFakeModel(),
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=(candidate,),
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7,),
        advances_m=(0.0,),
        lateral_offsets_m=(-0.3, -0.2, -0.1),
        right_waiting_degrees=(0.0,) * 7,
        minimum_camera_clearance_m=0.0,
        minimum_inter_arm_clearance_m=0.05,
    )

    assert plan is not None
    assert plan.lateral_m == -0.3
    assert plan.minimum_inter_arm_clearance_m == 0.3
    np.testing.assert_allclose(plan.position_m, (0.0, -0.3, 0.0))
