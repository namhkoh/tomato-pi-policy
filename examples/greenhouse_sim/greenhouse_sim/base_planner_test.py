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


def test_preferred_cross_organ_candidates_do_not_share_segment_filter() -> None:
    candidates = (
        base_planner.GraspCandidate(
            collider="short_petiole",
            body="short_petiole_body",
            segment=1,
            role="petiole_grasp",
            centre_m=(0.0, -1.0, 1.0),
            axis=(0.0, 0.0, 1.0),
        ),
        base_planner.GraspCandidate(
            collider="long_petiole",
            body="long_petiole_body",
            segment=4,
            role="petiole_grasp",
            centre_m=(0.0, -1.0, 4.0),
            axis=(0.0, 0.0, 1.0),
        ),
    )
    diagnostics = {}

    plan = base_planner.plan_target_conditioned_base(
        _FakeModel(),
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=candidates,
        candidates_are_preferred=True,
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7,),
        advances_m=(0.0,),
        minimum_camera_clearance_m=0.0,
        stop_on_first_feasible=True,
        diagnostics=diagnostics,
    )

    assert plan is not None
    assert plan.selected_grasp_collider == "short_petiole"
    assert plan.selected_grasp_segment == 1
    assert diagnostics["minimum_segment"] == 1
    assert diagnostics["candidates_are_preferred"] is True


def test_additional_feasibility_check_rejects_left_only_base() -> None:
    candidate = base_planner.GraspCandidate(
        collider="target",
        body="target_body",
        segment=1,
        role="petiole_grasp",
        centre_m=(0.0, -1.0, 1.0),
        axis=(0.0, 0.0, 1.0),
    )
    checked_positions = []

    def require_second_depth(**payload):
        position = np.asarray(payload["position_m"], dtype=np.float64)
        checked_positions.append(position.copy())
        return {
            "feasible": bool(position[1] <= -0.029),
            "mode": "test_bimanual_endpoint",
        }

    plan = base_planner.plan_target_conditioned_base(
        _FakeModel(),
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=(candidate,),
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7,),
        advances_m=(0.0, 0.03),
        minimum_camera_clearance_m=0.0,
        stop_on_first_feasible=True,
        additional_feasibility_check=require_second_depth,
    )

    assert plan is not None
    assert len(checked_positions) == 2
    assert plan.advance_m == 0.03
    assert plan.attempts[0].additional_feasibility["feasible"] is False
    assert plan.attempts[-1].additional_feasibility["feasible"] is True


def test_position_feasibility_check_runs_before_left_ik() -> None:
    class CountingModel(_FakeModel):
        def __init__(self):
            self.left_ik_calls = 0

        def solve_position_axes(self, *args, **kwargs):
            self.left_ik_calls += 1
            return super().solve_position_axes(*args, **kwargs)

    candidate = base_planner.GraspCandidate(
        collider='target',
        body='target_body',
        segment=1,
        role='petiole_grasp',
        centre_m=(0.0, -1.0, 1.0),
        axis=(0.0, 0.0, 1.0),
    )
    checked_positions = []

    def require_second_depth(**payload):
        position = np.asarray(payload['position_m'], dtype=np.float64)
        checked_positions.append(position.copy())
        return {
            'feasible': bool(position[1] <= -0.029),
            'mode': 'test_right_endpoint',
        }

    model = CountingModel()
    diagnostics = {}
    plan = base_planner.plan_target_conditioned_base(
        model,
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=(candidate,),
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7,),
        advances_m=(0.0, 0.03),
        minimum_camera_clearance_m=0.0,
        stop_on_first_feasible=True,
        position_feasibility_check=require_second_depth,
        diagnostics=diagnostics,
    )

    assert plan is not None
    assert len(checked_positions) == 2
    assert model.left_ik_calls == 1
    assert plan.advance_m == 0.03
    assert diagnostics['position_rejections'][0]['reason'] == (
        'additional_position_feasibility'
    )


def test_explicit_advance_direction_preserves_an_aisle_aligned_lattice() -> None:
    candidate = base_planner.GraspCandidate(
        collider="target",
        body="target_body",
        segment=1,
        role="petiole_grasp",
        centre_m=(1.0, 1.0, 1.0),
        axis=(0.0, 0.0, 1.0),
    )
    checked_positions = []

    def reject_position(**payload):
        checked_positions.append(np.asarray(payload["position_m"]).copy())
        return {"feasible": False, "mode": "test_explicit_aisle_direction"}

    plan = base_planner.plan_target_conditioned_base(
        _FakeModel(),
        nominal_position_m=(2.0, 3.0, 0.0),
        yaw_degrees=90.0,
        candidates=(candidate,),
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7,),
        advances_m=(0.020,),
        advance_direction_xy=(0.0, 1.0),
        minimum_camera_clearance_m=0.0,
        position_feasibility_check=reject_position,
    )

    assert plan is None
    assert len(checked_positions) == 1
    np.testing.assert_allclose(checked_positions[0], (2.0, 3.020, 0.0))


def test_position_feasibility_check_prunes_unreachable_target_candidates() -> None:
    class CountingModel(_FakeModel):
        def __init__(self):
            self.left_ik_targets = []

        def solve_position_axes(self, *args, **kwargs):
            self.left_ik_targets.append(tuple(kwargs['target_point_m']))
            return super().solve_position_axes(*args, **kwargs)

    candidates = tuple(
        base_planner.GraspCandidate(
            collider=collider,
            body=f'{collider}_body',
            segment=segment,
            role='petiole_grasp',
            centre_m=(0.0, -1.0, float(segment)),
            axis=(0.0, 0.0, 1.0),
        )
        for collider, segment in (
            ('right_unreachable', 1),
            ('eligible', 2),
        )
    )
    model = CountingModel()

    plan = base_planner.plan_target_conditioned_base(
        model,
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=candidates,
        candidates_are_preferred=True,
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7,),
        advances_m=(0.03,),
        minimum_camera_clearance_m=0.0,
        stop_on_first_feasible=True,
        position_feasibility_check=lambda **_payload: {
            'feasible': True,
            'mode': 'test_right_endpoint_candidates',
            'eligible_candidate_colliders': ('eligible',),
        },
    )

    assert plan is not None
    assert plan.selected_grasp_collider == 'eligible'
    assert model.left_ik_targets == [(0.0, -1.02, 2.0)]


def test_endpoint_feasibility_check_runs_before_trajectory_sampling() -> None:
    class CountingModel(_FakeModel):
        def __init__(self):
            self.arm_clearance_calls = 0

        def arm_obstacle_clearance(self, *args, **kwargs):
            self.arm_clearance_calls += 1
            return super().arm_obstacle_clearance(*args, **kwargs)

    candidate = base_planner.GraspCandidate(
        collider='target',
        body='target_body',
        segment=1,
        role='petiole_grasp',
        centre_m=(0.0, -1.0, 1.0),
        axis=(0.0, 0.0, 1.0),
    )
    diagnostics = {}
    model = CountingModel()

    plan = base_planner.plan_target_conditioned_base(
        model,
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=(candidate,),
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7,),
        advances_m=(0.0,),
        minimum_camera_clearance_m=0.0,
        endpoint_feasibility_check=lambda **_payload: {
            'feasible': False,
            'mode': 'test_inter_arm_endpoint',
        },
        diagnostics=diagnostics,
    )

    assert plan is None
    assert model.arm_clearance_calls == 1
    assert diagnostics['attempts'][0]['additional_feasibility'] == {
        'feasible': False,
        'mode': 'test_inter_arm_endpoint',
    }
    assert diagnostics['attempts'][0][
        'nearest_trajectory_arm_obstacle'
    ] == 'endpoint_not_clear'


def test_endpoint_check_can_prune_remaining_candidate_orientations() -> None:
    candidates = tuple(
        base_planner.GraspCandidate(
            collider=collider,
            body=f"{collider}_body",
            segment=index,
            role="petiole_grasp",
            centre_m=(0.0, -1.0, 1.0),
            axis=(0.0, 0.0, 1.0),
        )
        for index, collider in enumerate(("blocked", "clear"), start=1)
    )
    checked = []

    def endpoint_check(**payload):
        collider = payload["candidate"].collider
        checked.append(collider)
        return {
            "feasible": collider == "clear",
            "prune_candidate": collider == "blocked",
        }

    plan = base_planner.plan_target_conditioned_base(
        _FakeModel(),
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=candidates,
        candidates_are_preferred=True,
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7,),
        advances_m=(0.0,),
        minimum_camera_clearance_m=0.0,
        grasp_approach_yaw_offsets_degrees=(0.0, 20.0),
        grasp_transverse_signs=(-1.0, 1.0),
        endpoint_feasibility_check=endpoint_check,
        stop_on_first_feasible=True,
    )

    assert plan is not None
    assert plan.selected_grasp_collider == "clear"
    assert checked == ["blocked", "clear"]
class _PointingDirectionFakeModel(_FakeModel):
    def __init__(self) -> None:
        self.pointing_directions = []

    def solve_position_axes(
        self, _side, *, target_point_m, pointing_direction, **_kwargs
    ):
        self.pointing_directions.append(tuple(float(value) for value in pointing_direction))
        return robot_kinematics.IKResult(
            joint_degrees=(float(target_point_m[2]),) + (0.0,) * 6,
            position_error_m=0.0,
            orientation_error_rad=0.0,
            cost=0.0,
            succeeded=True,
        )


def test_planner_points_left_ee_positive_z_away_from_target_bearing() -> None:
    for yaw_degrees, target, expected in (
        (90.0, (0.0, 1.0, 1.0), (0.0, -1.0, 0.0)),
        (-90.0, (0.0, 1.0, 1.0), (0.0, -1.0, 0.0)),
        (90.0, (1.0, 1.0, 1.0), (-2**-0.5, -2**-0.5, 0.0)),
    ):
        candidate = base_planner.GraspCandidate(
            collider="target",
            body="target_body",
            segment=1,
            role="petiole_grasp",
            centre_m=target,
            axis=(1.0, 0.0, 0.0),
        )
        model = _PointingDirectionFakeModel()
        plan = base_planner.plan_target_conditioned_base(
            model,
            nominal_position_m=(0.0, 0.0, 0.0),
            yaw_degrees=yaw_degrees,
            candidates=(candidate,),
            obstacles=(),
            jaw_local_point_m=(0.0, 0.0, 0.0),
            camera_local_centre_m=(0.0, 0.0, 0.0),
            camera_radius_m=0.0,
            seeds=((0.0,) * 7,),
            advances_m=(0.0,),
            minimum_camera_clearance_m=0.0,
        )
        assert plan is not None
        np.testing.assert_allclose(model.pointing_directions[-1], expected, atol=1e-12)


class _TransverseDirectionFakeModel(_FakeModel):
    def __init__(self) -> None:
        self.pointing_directions = []
        self.transverse_directions = []

    def solve_position_axes(
        self,
        _side,
        *,
        target_point_m,
        pointing_direction,
        transverse_direction,
        **_kwargs,
    ):
        self.pointing_directions.append(
            tuple(float(value) for value in pointing_direction)
        )
        self.transverse_directions.append(
            tuple(float(value) for value in transverse_direction)
        )
        return robot_kinematics.IKResult(
            joint_degrees=(float(target_point_m[2]),) + (0.0,) * 6,
            position_error_m=0.0,
            orientation_error_rad=0.0,
            cost=0.0,
            succeeded=True,
        )


def test_planner_evaluates_and_records_angle_and_jaw_roll_orientations() -> None:
    candidate = base_planner.GraspCandidate(
        collider="target",
        body="target_body",
        segment=1,
        role="petiole_grasp",
        centre_m=(0.0, -1.0, 1.0),
        axis=(0.0, 0.0, 1.0),
    )
    model = _TransverseDirectionFakeModel()
    diagnostics = {}

    plan = base_planner.plan_target_conditioned_base(
        model,
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=(candidate,),
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7,),
        advances_m=(0.0,),
        minimum_camera_clearance_m=0.0,
        grasp_approach_yaw_offsets_degrees=(0.0, 20.0),
        grasp_transverse_signs=(1.0, -1.0),
        diagnostics=diagnostics,
    )

    assert plan is not None
    assert set(model.transverse_directions) == {
        (1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0),
        (0.9396926207859084, 0.3420201433256687, 0.0),
        (-0.9396926207859084, -0.3420201433256687, 0.0),
    }
    assert len(model.pointing_directions) == 4
    assert {
        attempt["grasp_approach_yaw_offset_degrees"]
        for attempt in diagnostics["attempts"]
    } == {0.0, 20.0}
    assert {attempt["grasp_transverse_sign"] for attempt in diagnostics["attempts"]} == {
        1.0,
        -1.0,
    }
    assert plan.selected_grasp_approach_yaw_offset_degrees in (0.0, 20.0)
    assert plan.selected_grasp_transverse_sign in (-1.0, 1.0)


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


class _FoliageTrajectoryClearanceFakeModel(_TrajectoryClearanceFakeModel):
    def fixed_body_oriented_box_clearance(self, _base_matrix, _obstacles):
        return robot_kinematics.ClearanceResult(float("inf"), None)
    def arm_obstacle_clearance(self, _side, _arm_degrees, _base_matrix, _obstacles):
        return robot_kinematics.ClearanceResult(0.05, "arm/vine")



    def arm_oriented_box_clearance(
        self,
        _side,
        arm_degrees,
        base_matrix,
        _obstacles,
    ):
        progress = float(arm_degrees[0])
        nominal_lane = abs(float(base_matrix[1, 3])) < 0.1
        clearance = -0.02 if nominal_lane and 0.25 < progress < 0.75 else 0.05
        return robot_kinematics.ClearanceResult(clearance, "arm/foliage")


def test_planner_rejects_foliage_intersection_on_clear_capsule_route() -> None:
    candidate = base_planner.GraspCandidate(
        collider="link_1",
        body="body_1",
        segment=1,
        role="petiole_grasp",
        centre_m=(0.0, -1.0, 1.0),
        axis=(0.0, 0.0, 1.0),
    )
    foliage = robot_kinematics.OrientedBoxObstacle(
        path="foliage",
        centre_m=(100.0, 100.0, 100.0),
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        half_extents_m=(0.01, 0.01, 0.01),
    )
    diagnostics = {}

    plan = base_planner.plan_target_conditioned_base(
        _FoliageTrajectoryClearanceFakeModel(),
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=(candidate,),
        obstacles=(),
        foliage_obstacles=(foliage,),
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
        minimum_foliage_clearance_m=0.001,
        trajectory_samples=5,
        diagnostics=diagnostics,
    )

    assert plan is not None
    assert plan.lateral_m == 0.3
    assert diagnostics["attempts"][0][
        "trajectory_arm_foliage_clearance_m"
    ] == -0.02
    assert diagnostics["minimum_foliage_clearance_m"] == 0.001

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


def test_open_finger_must_clear_target_branch_foliage_during_approach() -> None:
    model = _PayloadClearanceFakeModel()
    foliage = robot_kinematics.OrientedBoxObstacle(
        path="target_branch_foliage",
        centre_m=(0.0, 0.0, 0.0),
        rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        half_extents_m=(0.02, 0.02, 0.02),
    )
    finger_box = ((
        "ee_finger_l1",
        (0.0, 0.0, 0.0),
        np.eye(3),
        (0.01, 0.01, 0.01),
    ),)
    clearance = base_planner._tool_payload_foliage_clearance(
        model,
        (0.0,) * 7,
        np.eye(4),
        (foliage,),
        finger_box,
        (foliage.path,),
    )

    assert clearance.clearance_m < 0.0
    assert clearance.nearest_obstacle.startswith("ee_finger_l1")


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


def test_joint_space_route_rejects_coarse_path_that_fails_fine_validation() -> None:
    def valid(values):
        return abs(float(values[0])) > 0.025

    route = base_planner._plan_joint_space_route(
        (-0.9,),
        (0.9,),
        (-1.0,),
        (1.0,),
        valid,
        seed=3,
        max_iterations=1,
        edge_resolution=1.0,
        validation_edge_resolution=0.01,
    )

    # Coarse endpoint-only sampling misses the narrow forbidden interval.
    # Fine validation must reject it; one dimension offers no alternate path.
    assert route is None


def test_planner_bounds_expensive_joint_space_route_searches(monkeypatch) -> None:
    model = _TrajectoryClearanceFakeModel()
    model.arm_limits_degrees = lambda _side: (
        np.full(7, -180.0),
        np.full(7, 180.0),
    )
    searches = []

    def no_route(*args, **kwargs):
        searches.append((args, kwargs))
        return None

    monkeypatch.setattr(base_planner, "_plan_joint_space_route", no_route)
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
        model,
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=(candidate,),
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7, (1.0,) * 7),
        advances_m=(0.0,),
        lateral_offsets_m=(0.0,),
        left_waiting_degrees=(0.0,) * 7,
        left_approach_start_degrees=(0.0,) * 7,
        minimum_camera_clearance_m=0.0,
        minimum_trajectory_clearance_m=0.01,
        trajectory_samples=5,
        joint_space_search_iterations=100,
        maximum_joint_space_route_searches=1,
        diagnostics=diagnostics,
    )

    assert plan is None
    assert len(searches) == 1
    assert diagnostics["joint_space_route_searches"] == 1
    assert diagnostics["maximum_joint_space_route_searches"] == 1


def test_online_planner_stops_after_first_safe_plan() -> None:
    class CountingModel(_FakeModel):
        def __init__(self):
            self.solve_calls = 0

        def solve_position_axes(self, *args, **kwargs):
            self.solve_calls += 1
            return super().solve_position_axes(*args, **kwargs)

    model = CountingModel()
    candidate = base_planner.GraspCandidate(
        collider="link_1",
        body="body_1",
        segment=1,
        role="petiole_grasp",
        centre_m=(0.0, -1.0, 1.0),
        axis=(0.0, 0.0, 1.0),
    )

    plan = base_planner.plan_target_conditioned_base(
        model,
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=(candidate,),
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7, (1.0,) * 7),
        advances_m=(0.03, 0.06),
        minimum_camera_clearance_m=0.0,
        stop_on_first_feasible=True,
    )

    assert plan is not None
    assert model.solve_calls == 1

def test_target_relative_ingress_offsets_are_opt_in_and_audited(
    monkeypatch,
) -> None:
    class RecordingModel(_FakeModel):
        def __init__(self):
            self.target_points = []

        def solve_position_axes(self, *args, **kwargs):
            self.target_points.append(tuple(kwargs["target_point_m"]))
            return super().solve_position_axes(*args, **kwargs)

    sampled_route_lengths = []
    original_sampler = base_planner._sample_grasp_trajectory

    def record_route(*args, **kwargs):
        sampled_route_lengths.append(len(kwargs["waypoint_degrees"]))
        return original_sampler(*args, **kwargs)

    monkeypatch.setattr(base_planner, "_sample_grasp_trajectory", record_route)

    model = RecordingModel()
    candidate = base_planner.GraspCandidate(
        collider="link_1",
        body="body_1",
        segment=1,
        role="petiole_grasp",
        centre_m=(0.0, -1.0, 1.0),
        axis=(0.0, 0.0, 1.0),
    )

    plan = base_planner.plan_target_conditioned_base(
        model,
        nominal_position_m=(0.0, 0.0, 0.0),
        yaw_degrees=0.0,
        candidates=(candidate,),
        obstacles=(),
        jaw_local_point_m=(0.0, 0.0, 0.0),
        camera_local_centre_m=(0.0, 0.0, 0.0),
        camera_radius_m=0.0,
        seeds=((0.0,) * 7,),
        advances_m=(0.0,),
        minimum_camera_clearance_m=0.0,
        target_relative_ingress_offsets_m=(0.02, 0.04, 0.06),
        target_relative_ingress_gateways_m=(
            (0.06, 0.03, 0.05),
            (0.06, 0.03, 0.05, 2.0),
        ),
        stop_on_first_feasible=False,
    )

    assert plan is not None
    np.testing.assert_allclose(
        model.target_points,
        (
            (0.0, -1.02, 1.0),
            (0.0, -1.00, 1.0),
            (0.0, -0.98, 1.0),
            (0.0, -0.96, 1.0),
            (0.01, -1.00, 1.0 + 0.05 / 3.0),
            (0.02, -0.98, 1.0 + 0.10 / 3.0),
            (0.03, -0.96, 1.05),
            (0.03, -0.96, 1.05),
            (0.03 / 9.0, -1.00, 1.0 + 0.05 / 9.0),
            (0.03 * 4.0 / 9.0, -0.98, 1.0 + 0.05 * 4.0 / 9.0),
            (0.03, -0.96, 1.05),
            (0.03, -0.96, 1.05),
        ),
    )
    assert sampled_route_lengths == [0, 0, 3, 3, 4, 4, 4, 4]
    assert [
        screen["index"]
        for screen in plan.attempts[0].approach_route_screens
    ] == [0, 1, 2, 3]
    assert all(
        screen["bottleneck"] == "arm"
        for screen in plan.attempts[0].approach_route_screens
    )


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
