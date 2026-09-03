"""Regressions for task-semantic robot/contact safety policy."""

from __future__ import annotations

import json
import pathlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from interactive_greenhouse import (
    _LEFT_CLEARANCE_INGRESS_WAYPOINTS_DEGREES,
    _LEFT_GRIPPER_OPEN_WIDTH_M,
    _LEFT_GRASP_APPROACH_YAW_OFFSETS_DEGREES,
    _LEFT_GRIPPER_PLANNING_GEOMETRY,
    _LEFT_JAW_CENTRE_M,
    _LEFT_MULTISTART_SEEDS_DEGREES,
    _LEFT_GREENHOUSE_WAITING_DEGREES,
    _LEFT_READY_DEGREES,
    _LOCAL_REQUESTED_BASE_LATERAL_OFFSETS_M,
    _RIGHT_AISLE_RETREAT_OFFSETS_M,
    _RIGHT_APPROACH_OFFSETS_M,
    _RIGHT_COMMITTED_CUT_ALIGNMENT_SIDE_CANDIDATES_M,
    _RIGHT_COMMITTED_CUT_EXIT_SIDE_M,
    _RIGHT_CUT_MINIMUM_TARGET_INTERSECTION_M,
    _RIGHT_COMMITTED_CUT_LIVE_REPLAN_GUARD_M,
    _RIGHT_COMMITTED_CUT_LIVE_REPLAN_TRIGGER_M,
    _RIGHT_COMMITTED_CUT_PLANNING_CLEARANCE_M,
    _RIGHT_COMMITTED_CUT_RECOVERY_DECREASE_TOLERANCE_M,
    _RIGHT_COMMITTED_CUT_SEGMENT_LENGTH_M,
    _RIGHT_COMMITTED_CUT_STUB_CANDIDATES_M,
    _RIGHT_COMMITTED_CUT_DAMPING_NM_S_RAD,
    _RIGHT_COMMITTED_CUT_STIFFNESS_NM_RAD,
    _RBY1_ARM_DAMPING_NM_S_RAD,
    _RBY1_ARM_EFFORT_LIMITS_NM,
    _RBY1_ARM_STIFFNESS_NM_RAD,
    _RBY1_TORSO_DAMPING_NM_S_RAD,
    _RBY1_TORSO_EFFORT_LIMITS_NM,
    _RBY1_TORSO_STIFFNESS_NM_RAD,
    _RIGHT_RIGID_RECOVERY_DAMPING_NM_S_RAD,
    _RIGHT_RIGID_RECOVERY_STIFFNESS_NM_RAD,
    _RIGHT_RIGID_RECOVERY_TORSO_DAMPING_NM_S_RAD,
    _RIGHT_RIGID_RECOVERY_TORSO_STIFFNESS_NM_RAD,
    _RIGHT_BASE_ENDPOINT_SIDE_CANDIDATES_M,
    _RIGHT_BASE_ENDPOINT_ROLLS_DEGREES,
    _RIGHT_ONLINE_BASE_ENDPOINT_ROLLS_DEGREES,
    _RIGHT_ROUTE_REPLAN_LIMIT,
    _RIGHT_PRESTAGE_ROUTE_REPLAN_LIMIT,
    _RIGHT_PRESTAGE_SEGMENT_LIMIT,
    _RIGHT_ROUTE_RRT_MAX_ITERATIONS,
    _RIGHT_PRESTAGE_PLANNING_SETTLE_LIMIT,
    _RIGHT_SAFE_DEGREES,
    _RIGHT_READY_DEGREES,
    _RIGHT_SERVO_MAX_ATTEMPTS,
    _RIGHT_LIVE_WAYPOINT_REPLAN_MAX_ATTEMPTS,
    _LEFT_MID_APPROACH_ADDITIONAL_PRETENSION_M,
    _LEFT_CUT_CORRIDOR_MAXIMUM_INITIAL_PRETENSION_M,
    _emit,
    _TeleopCameraRecorder,
    BladeContactMonitor,
    RobotContactDiagnostics,
    InteractionController,
    parse_args,
    _bimanual_startup_joint_state_settled,
    _bounded_arm_motion_steps,
    _default_inspection_camera_pose,
    _force_control_unload_fraction,
    _minimum_jerk_fraction,
    _greenhouse_startup_arm_targets,
    _guarded_stop_accepted_payload_clearance_m,
    _startup_task_stow_waypoints,
    _startup_task_stow_reached,
    _post_recovery_tracking_required,
    _right_committed_cut_recovery_torso_target,
    _right_committed_cut_stationary_brake_torso_target,
    _right_prestage_inward_fallback_sides,
    _right_base_endpoint_route_feasible,
    _right_local_cut_continuation_screen,
    _pose_nullspace_seed_candidates,
    _blade_traversal_contact_step,
    _bounded_robot_forward_nudge,
    _bounded_disengage_ik_acceptable,
    _closing_jaw_corridor_overlap,
    _contact_avoiding_jaw_reentry_targets,
    _endpoint_hold_clearance_guard_m,
    _endpoint_hold_replan_stop,
    _geometric_cut_reaction_force,
    _guarded_jaw_handoff_step,
    _jaw_corridor_handoff_acceptable,
    _guarded_reentry_pose_acceptable,
    _jaw_anchor_microseat_target,
    _joint_reserve_recovery_acceptable,
    _initial_pretension_release_retry_allowed,
    _left_mid_approach_pretension_distances,
    _left_mid_approach_pretension_direction_specs,
    _held_branch_palm_collision_filter_pairs,
    _left_payload_component_accepts_target_contact,
    _left_mid_pretension_release_candidates,
    _left_mid_pretension_restore_action,
    _left_pretension_release_target,
    _left_cut_corridor_pretension_release_required,
    _left_pretension_pull_specs,
    _left_committed_cut_axial_pretension_direction,
    _minimum_left_counterhold_capacity_n,
    _left_counterhold_posture_acceptable,
    _planning_left_pretension_candidates,
    _select_left_committed_cut_axial_pretension,
    _selected_right_goal_direction,
    _select_left_pretension_pull,
    _live_payload_replan_stop,
    _local_requested_base_depth_advances,
    _left_clearance_route_measured_handoff_allowed,
    _lateral_backstop_seat_target,
    _lateral_jaw_disengage_target,
    _measured_state_reconstruction_limits,
    _measured_state_reconstruction_acceptable,
    _measured_state_settled_position_limit_m,
    _movement_measurement_seed,
    _open_jaw_capture_corridor_overlap,
    _open_finger_reseat_allowed,
    _opposed_backstop_closure_schedule,
    _opposed_finger_contact,
    _planned_right_endpoint_hint,
    _planned_route_start_reconciliation_required,
    _target_grasp_axial_offsets,
    _planned_foliage_escape_floor,
    _right_goal_foliage_escape_floor,
    _right_goal_rigid_vine_planning_escape,
    _probe_capture_interval_steps,
    _right_committed_cut_replan_action,
    _right_committed_cut_replan_settle_mode,
    _right_committed_cut_live_planning_clearance_m,
    _right_committed_cut_maximum_recovery_m,
    _right_committed_cut_measured_stop_reserve_acceptable,
    _right_committed_cut_recovery_floor_m,
    _right_committed_cut_recovery_feedback_pose,
    _right_committed_cut_intersection_bounded_feedback_pose,
    _right_committed_cut_physical_target_intersection_m,
    _right_committed_cut_rigid_recovery_acceptable,
    _right_committed_cut_recovery_brake_complete,
    _right_committed_cut_stationary_recovery_stage,
    _right_committed_cut_requires_stationary_recovery_first,
    _right_committed_cut_stationary_recovery_acceptable,
    _right_committed_cut_rigid_recovery_correction,
    _right_committed_cut_rigid_recovery_path_acceptable,
    _right_committed_cut_solve_retryable,
    _right_committed_cut_rigid_endpoint_rejection,
    _probe_unsafe_contacts,
    _required_probe_payload_clearance,
    _scaled_motion_steps,
    _right_cartesian_entry_offset_routes,
    _right_cartesian_entry_route_rotation,
    _right_cartesian_entry_execution_route,
    _right_cartesian_entry_route_shortlist,
    _right_compound_prestage_sides,
    _right_leading_edge_point_candidates,
    _right_live_route_replan_searches,
    _right_live_waypoint_replan_overrides,
    _right_live_waypoint_replan_searches,
    _right_live_route_settle_retryable,
    _right_live_route_active_retreat_retryable,
    _right_failed_live_route_recovery_eligible,
    _right_provisional_live_foliage_route_acceptable,
    _right_provisional_retreat_floor_m,
    _right_route_waypoint_seed_candidates,
    _right_stow_connection_waypoint_candidates,
    _right_stow_connection_two_bend_candidates,
    _right_committed_cut_roll_refinement,
    _right_bidirectional_cut_roll_candidates,
    _right_cut_direction_aisle_family,
    _right_committed_live_roll_refinement_requests,
    _right_committed_live_transverse_refinement_requests,
    _right_committed_alignment_retreat_sides,
    _right_committed_alignment_retreat_route_candidates,
    _right_committed_cut_segment_count,
    _right_committed_cut_segment_sides,
    _right_committed_cut_fracture_cycle_sides,
    _right_committed_cut_preflight_sides,
    _right_target_screen_cut_sides,
    _right_committed_cut_segment_steps,
    _right_committed_cut_bounded_transverse_correction,
    _right_committed_cut_target_drift_correction,
    _right_committed_cut_tracking_feedback_correction,
    _right_committed_cut_contact_plane_correction,
    _right_committed_cut_live_tracking_update,
    _right_committed_cut_candidate_rank,
    _right_committed_cut_configuration_matches_hint,
    _right_nearby_rolls,
    _right_approach_joint_reserve_acceptable,
    _right_motion_joint_limit_margin_degrees,
    _right_joint_reserve_recovery_candidates,
    _right_joint_reserve_recovery_target,
    _right_route_endpoint_joint_reserve_candidate,
    _right_approach_offsets_from_side,
    _right_endpoint_clearance_screen,
    _right_cut_stub_search_order,
    _right_protected_transverse_cut_geometry,
    _right_continuation_failure_is_left_independent,
    _right_endpoint_failure_is_left_independent,
    _right_expanded_endpoint_search_enabled,
    _base_planning_joint_space_route_budget,
    _right_expanded_endpoint_roll_candidates,
    _right_entry_pretension_release_required,
    _stationary_right_pretension_release_clear,
    _right_outer_endpoint_ik_screen,
    _right_reachable_approach_endpoint_ik_screen,
    _knife_rotation_for_roll,
    _right_route_foliage_planning_clearance_m,
    _right_guarded_route_tail_action,
    _right_guarded_retreat_action,
    _right_waypoint_settle_retry_allowed,
    _previous_completed_right_waypoint,
    _previous_distinct_completed_right_waypoint,
    _right_recovery_cartesian_side_m,
    _right_recovery_cartesian_sides,
    _shortcut_collision_clear_joint_route,
    _right_waypoint_route_strategy,
    _right_waypoint_search_overrides,
    _retryable_live_base_plan_rejection,
    _select_physics_vines,
    _subreserve_clearance_recovery_acceptable,
    _subreserve_runtime_clearance_acceptable,
    _stationary_foliage_settle_eligible,
    _target_contact_point,
    _target_contact_supports_guarded_close,
    _tangential_jaw_capture_target,
    _transformed_cube_bounds,
    _retreat_search_complete,
)
from greenhouse_sim import robot_hardware, robot_kinematics


def test_report_emit_replaces_complete_json(tmp_path) -> None:
    report_path = tmp_path / "report.json"

    _emit({"stage": "first"}, report_path)
    _emit({"stage": "second", "complete": True}, report_path)

    assert json.loads(report_path.read_text(encoding="utf-8")) == {
        "stage": "second",
        "complete": True,
    }

def test_blade_target_path_uses_the_selected_articulated_segment_radius(
    monkeypatch,
) -> None:
    class _Direction:
        def __init__(self, value) -> None:
            self._value = np.asarray(value, dtype=np.float64)

        def GetNormalized(self):
            return self._value / np.linalg.norm(self._value)

    class _Matrix:
        def Transform(self, value):
            return np.asarray(value, dtype=np.float64)

        def TransformDir(self, value):
            return _Direction(value)

    matrix = _Matrix()
    fake_pxr = SimpleNamespace(
        Usd=SimpleNamespace(
            TimeCode=SimpleNamespace(Default=lambda: None),
        ),
        UsdGeom=SimpleNamespace(
            Xformable=lambda _prim: SimpleNamespace(
                ComputeLocalToWorldTransform=lambda _time: matrix,
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "pxr", fake_pxr)

    monitor = BladeContactMonitor.__new__(BladeContactMonitor)
    monitor._np = np
    monitor._stage = SimpleNamespace(GetPrimAtPath=lambda _path: object())
    monitor._cutting = SimpleNamespace(
        CutTarget=lambda **values: SimpleNamespace(**values),
    )
    monitor._active_target = "Vine_0002/SubStem_07"
    monitor._targets = {
        monitor._active_target: (
            object(),
            SimpleNamespace(
                key=monitor._active_target,
                organ_label="SubStem_07",
                radius_m=0.0028690388582134248,
                cut_force_n=66.3,
            ),
        ),
    }
    collider = "/World/Vine/Physics/Organ/Link_001/Collider"
    monitor._target_colliders = {collider: monitor._active_target}
    monitor._collider_target_frames = {
        collider: {
            "body": "/World/Vine/Physics/Organ/Link_001",
            "centre_local": np.zeros(3),
            "axis_local": np.asarray((1.0, 0.0, 0.0)),
            "radius_m": 0.007155266719223556,
            "segment": 1,
            "arc_start_m": 0.010,
        },
    }

    geometry = monitor.target_path_geometry(0.0195)

    assert geometry is not None
    assert geometry["segment"] == 1
    assert geometry["radius_m"] == pytest.approx(0.007155266719223556)
    # The stored centre is the segment's virtual junction, so a cumulative
    # 19.5 mm stub remains a full 19.5 mm projection in that live frame.
    np.testing.assert_allclose(geometry["point_m"], (0.0195, 0.0, 0.0))



def test_ui_ik_click_renders_planning_state_before_solver(tmp_path) -> None:
    controller = InteractionController.__new__(InteractionController)
    calls = []

    def run_grasp():
        calls.append("grasp")
        return {"succeeded": True}

    controller._grasp_manager = None
    controller._blade_cutting = None
    controller._airflow = SimpleNamespace(step=lambda: None)
    controller._visual_pull = SimpleNamespace(step=lambda: None)
    controller._pending = ["run_grasp_ik"]
    controller._run_grasp_ik = run_grasp
    controller._run_full_ik = lambda: {"succeeded": True}
    controller._ik_running = False
    controller._run_grasp_ik_button = SimpleNamespace(enabled=True)
    controller._run_full_ik_button = SimpleNamespace(enabled=True)
    controller._status_label = SimpleNamespace(text="")
    controller._target_label = SimpleNamespace(text="")
    controller._targets = [["SubStem_07"]]
    controller._vine = 0
    controller._target = 0
    controller._runtimes = [SimpleNamespace(name="Vine_0002")]
    controller._report = {}
    controller._report_path = tmp_path / "ui_ik.json"

    controller.process(None)

    assert calls == []
    assert controller._pending == ["execute_grasp_ik"]
    assert controller._ik_running
    assert not controller._run_grasp_ik_button.enabled
    assert not controller._run_full_ik_button.enabled
    assert "planning collision-safe approach" in controller._status_label.text
    assert controller._report["ui_ik_sequence"] == {
        "mode": "grasp",
        "stage": "planning",
        "succeeded": None,
    }

    controller.process(None)

    assert calls == ["grasp"]
    assert controller._pending == []
    assert not controller._ik_running
    assert controller._run_grasp_ik_button.enabled
    assert controller._run_full_ik_button.enabled
    assert controller._status_label.text == "Grasp IK sequence passed"
    assert controller._report["ui_ik_sequence"] == {
        "mode": "grasp",
        "stage": "complete",
        "succeeded": True,
    }





def test_right_aisle_retreat_screens_the_existing_safe_posture_first() -> None:
    assert _RIGHT_AISLE_RETREAT_OFFSETS_M[0] == (0.0, 0.0, 0.0)
    assert len(set(_RIGHT_AISLE_RETREAT_OFFSETS_M)) == len(
        _RIGHT_AISLE_RETREAT_OFFSETS_M
    )
    assert _retreat_search_complete(
        _RIGHT_AISLE_RETREAT_OFFSETS_M[0], True, 0.001
    )
    assert not _retreat_search_complete(
        _RIGHT_AISLE_RETREAT_OFFSETS_M[0], False, 0.020
    )
    assert _retreat_search_complete(
        _RIGHT_AISLE_RETREAT_OFFSETS_M[1], True, 0.012
    )
    assert not _retreat_search_complete(
        _RIGHT_AISLE_RETREAT_OFFSETS_M[1], True, 0.0119
    )


def test_selected_left_plan_recovers_its_proven_right_endpoint() -> None:
    preposition = {
        'selected_grasp_collider': '/grasp',
        'solution': {'joint_degrees': [1.0] * 7},
        'attempts': [
            {
                'collider': '/grasp',
                'solution': {'joint_degrees': [1.0] * 7},
                'additional_feasibility': {
                    'feasible': True,
                    'side_m': -0.16,
                    'cut_stub_m': 0.015,
                    'cut_transverse_offset_m': [0.001, -0.002, 0.0],
                    'selected': {
                        'roll_degrees': 15.0,
                        'edge_wing_local_m': [-0.014, -0.07, 0.0],
                        'solution': {'joint_degrees': [2.0] * 7},
                    },
                },
            }
        ],
    }

    assert _planned_right_endpoint_hint(preposition) == {
        'roll_degrees': 15.0,
        'edge_wing_local_m': [-0.014, -0.07, 0.0],
        'joint_degrees': [2.0] * 7,
        'side_m': -0.16,
        'cut_stub_m': 0.015,
        'cut_transverse_offset_m': [0.001, -0.002, 0.0],
    }
    preposition['attempts'][0]['additional_feasibility']['feasible'] = False
    assert _planned_right_endpoint_hint(preposition) is None


def _cut_obstacle(path, start, end, radius_m=0.001):
    return SimpleNamespace(
        path=path,
        start_m=np.asarray(start, dtype=np.float64),
        end_m=np.asarray(end, dtype=np.float64),
        radius_m=radius_m,
    )


def test_protected_cut_offset_moves_transversely_away_from_neighbour() -> None:
    geometry = {
        "point_m": np.zeros(3),
        "axis": np.asarray([0.0, 0.0, 1.0]),
        "radius_m": 0.003,
    }
    adjusted = _right_protected_transverse_cut_geometry(
        geometry,
        (
            _cut_obstacle(
                "/protected",
                (-0.010, 0.0, -0.010),
                (-0.010, 0.0, 0.010),
            ),
        ),
    )

    assert np.allclose(adjusted["transverse_offset_m"], (0.0015, 0.0, 0.0))
    assert np.dot(adjusted["transverse_offset_m"], adjusted["axis"]) == pytest.approx(0.0)
    assert adjusted["transverse_offset_reference"] == "/protected"
    assert adjusted["target_intersection_margin_m"] == pytest.approx(0.0015)


def test_protected_cut_offset_clamps_to_target_intersection_reserve() -> None:
    adjusted = _right_protected_transverse_cut_geometry(
        {
            "point_m": np.zeros(3),
            "axis": np.asarray([0.0, 0.0, 1.0]),
            "radius_m": 0.002,
        },
        (
            _cut_obstacle(
                "/protected",
                (-0.010, 0.0, -0.010),
                (-0.010, 0.0, 0.010),
            ),
        ),
        desired_offset_m=0.001,
        minimum_target_intersection_m=0.0015,
    )

    assert adjusted["transverse_offset_distance_m"] == pytest.approx(0.0005)
    assert adjusted["target_intersection_margin_m"] == pytest.approx(0.0015)

def test_protected_cut_offset_rotation_preserves_intersection_reserve() -> None:
    geometry = {
        "point_m": np.zeros(3),
        "axis": np.asarray([0.0, 0.0, 1.0]),
        "radius_m": 0.003,
    }
    obstacles = (
        _cut_obstacle(
            "/protected",
            (-0.010, 0.0, -0.010),
            (-0.010, 0.0, 0.010),
        ),
    )
    adjusted = _right_protected_transverse_cut_geometry(
        geometry,
        obstacles,
        desired_offset_m=0.0015,
        minimum_target_intersection_m=0.0015,
        offset_rotation_degrees=30.0,
    )

    assert np.allclose(
        adjusted["transverse_offset_m"],
        (0.0015 * np.cos(np.radians(30.0)), 0.00075, 0.0),
    )
    assert adjusted["transverse_offset_distance_m"] == pytest.approx(0.0015)
    assert adjusted["target_intersection_margin_m"] == pytest.approx(0.0015)
    assert adjusted["transverse_offset_rotation_degrees"] == 30.0
    assert np.dot(
        adjusted["transverse_offset_m"], adjusted["axis"]
    ) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        _right_protected_transverse_cut_geometry(
            geometry, obstacles, offset_rotation_degrees=45.001
        )


def test_protected_cut_offset_excludes_target_branch_paths() -> None:
    adjusted = _right_protected_transverse_cut_geometry(
        {
            "point_m": np.zeros(3),
            "axis": np.asarray([0.0, 0.0, 1.0]),
            "radius_m": 0.003,
        },
        (
            _cut_obstacle(
                "/target/Collider",
                (-0.004, 0.0, -0.010),
                (-0.004, 0.0, 0.010),
            ),
            _cut_obstacle(
                "/protected",
                (0.0, -0.010, -0.010),
                (0.0, -0.010, 0.010),
            ),
        ),
        excluded_obstacle_paths=("/target",),
    )

    assert adjusted["transverse_offset_reference"] == "/protected"
    assert np.allclose(adjusted["transverse_offset_m"], (0.0, 0.0015, 0.0))


def test_protected_cut_offset_stays_nominal_without_transverse_direction() -> None:
    adjusted = _right_protected_transverse_cut_geometry(
        {
            "point_m": np.zeros(3),
            "axis": np.asarray([0.0, 0.0, 1.0]),
            "radius_m": 0.003,
        },
        (
            _cut_obstacle(
                "/axial",
                (0.0, 0.0, -0.010),
                (0.0, 0.0, 0.010),
            ),
        ),
    )

    assert np.allclose(adjusted["point_m"], np.zeros(3))
    assert np.allclose(adjusted["transverse_offset_m"], np.zeros(3))
    assert adjusted["transverse_offset_reference"] is None


def test_right_approach_establishes_orientation_outside_old_first_pose() -> None:
    assert _RIGHT_APPROACH_OFFSETS_M[:6] == (
        -0.200,
        -0.180,
        -0.160,
        -0.140,
        -0.120,
        -0.100,
    )
    assert _RIGHT_APPROACH_OFFSETS_M[-1] == -0.015
    assert all(
        later > earlier
        for earlier, later in zip(
            _RIGHT_APPROACH_OFFSETS_M,
            _RIGHT_APPROACH_OFFSETS_M[1:],
        )
    )


def test_planned_right_endpoint_starts_at_its_audited_approach_suffix() -> None:
    assert _RIGHT_BASE_ENDPOINT_SIDE_CANDIDATES_M == (
        -0.200,
        -0.180,
        -0.160,
        -0.140,
        -0.120,
        -0.100,
        -0.060,
    )
    assert _RIGHT_ONLINE_BASE_ENDPOINT_ROLLS_DEGREES[:2] == (74.5, -74.5)
    assert 60.0 in _RIGHT_ONLINE_BASE_ENDPOINT_ROLLS_DEGREES
    assert -60.0 in _RIGHT_ONLINE_BASE_ENDPOINT_ROLLS_DEGREES
    assert 70.0 in _RIGHT_ONLINE_BASE_ENDPOINT_ROLLS_DEGREES
    assert -70.0 in _RIGHT_ONLINE_BASE_ENDPOINT_ROLLS_DEGREES
    assert 74.0 in _RIGHT_ONLINE_BASE_ENDPOINT_ROLLS_DEGREES
    assert -74.0 in _RIGHT_ONLINE_BASE_ENDPOINT_ROLLS_DEGREES
    assert 74.5 in _RIGHT_ONLINE_BASE_ENDPOINT_ROLLS_DEGREES
    assert -74.5 in _RIGHT_ONLINE_BASE_ENDPOINT_ROLLS_DEGREES
    assert 75.0 in _RIGHT_ONLINE_BASE_ENDPOINT_ROLLS_DEGREES
    assert -75.0 in _RIGHT_ONLINE_BASE_ENDPOINT_ROLLS_DEGREES
    assert _right_approach_offsets_from_side(-0.140) == (
        _RIGHT_APPROACH_OFFSETS_M[3:]
    )
    with np.testing.assert_raises(ValueError):
        _right_approach_offsets_from_side(-0.130)


def test_target_grasp_axial_offsets_stay_inside_physical_segment() -> None:
    assert _LEFT_GRASP_APPROACH_YAW_OFFSETS_DEGREES[0] == -45.0
    assert _target_grasp_axial_offsets(0.020, 0.0015) == (
        0.0,
        0.001,
        -0.001,
        0.002,
        -0.002,
        0.0035,
        -0.0035,
        0.003,
        -0.003,
        0.004,
        -0.004,
        0.006,
        -0.006,
        0.007,
        -0.007,
        0.008,
        -0.008,
    )
    assert _target_grasp_axial_offsets(0.004, 0.002) == (0.0,)
    assert _target_grasp_axial_offsets(
        0.020,
        0.0015,
        requested_offsets_m=(0.006, -0.006, 0.006),
    ) == (0.006, -0.006)
    with np.testing.assert_raises(ValueError):
        _target_grasp_axial_offsets(0.0, 0.0015)
    assert _target_grasp_axial_offsets(
        0.020,
        0.0015,
        requested_offsets_m=(0.009,),
    ) == ()


def test_far_right_prestage_uses_proven_bridge_then_steps_outward() -> None:
    assert _right_compound_prestage_sides(-0.200) == (
        -0.160,
        -0.180,
        -0.200,
    )
    assert _right_compound_prestage_sides(-0.180) == (-0.160, -0.180)
    assert _right_compound_prestage_sides(-0.160) == (-0.160,)
    assert _right_compound_prestage_sides(-0.140) == (-0.140,)
    assert _right_prestage_inward_fallback_sides(-0.200) == (
        -0.160,
        -0.140,
        -0.120,
        -0.100,
        -0.060,
    )
    assert _right_prestage_inward_fallback_sides(-0.140) == (
        -0.140,
        -0.120,
        -0.100,
        -0.060,
    )
    with np.testing.assert_raises(ValueError):
        _right_compound_prestage_sides(-0.190)


def test_guarded_right_entry_retries_rotate_route_families() -> None:
    routes = ({'name': 'a'}, {'name': 'b'}, {'name': 'c'})
    assert _right_cartesian_entry_route_rotation(routes, 0) == routes
    assert _right_cartesian_entry_route_rotation(routes, 1) == (
        routes[1], routes[2], routes[0]
    )
    assert _right_cartesian_entry_route_rotation(routes, 4) == (
        routes[1], routes[2], routes[0]
    )
    assert _right_cartesian_entry_route_rotation((), 3) == ()
    with np.testing.assert_raises(ValueError):
        _right_cartesian_entry_route_rotation(routes, -1)


def test_guarded_right_entry_retry_preserves_distinct_detour() -> None:
    full = ('detour_a', 'detour_b', 'outer')
    shortcut = ('outer',)

    assert _right_cartesian_entry_execution_route(full, shortcut, 0) == shortcut
    assert _right_cartesian_entry_execution_route(full, shortcut, 1) == full
    with np.testing.assert_raises(ValueError):
        _right_cartesian_entry_execution_route(full, shortcut, -1)


def test_blocked_inner_right_waypoint_uses_local_joint_detour() -> None:
    assert _right_waypoint_route_strategy(-0.200, False) == "outer_entry"
    assert _right_waypoint_route_strategy(-0.180, False) == (
        "local_joint_detour"
    )
    assert _right_waypoint_route_strategy(-0.160, False) == (
        "local_joint_detour"
    )
    assert _right_waypoint_route_strategy(-0.140, False) == (
        "local_joint_detour"
    )
    assert _right_waypoint_route_strategy(-0.120, False) == (
        "local_joint_detour"
    )
    assert _right_waypoint_route_strategy(-0.120, True) == "direct"
    assert _right_waypoint_route_strategy(
        -0.180, False, outer_side_m=-0.180
    ) == "outer_entry"


def test_guarded_right_route_reuses_only_a_clean_audited_tail() -> None:
    assert _right_guarded_route_tail_action(3, None) == (
        "continue_audited_tail"
    )
    assert _right_guarded_route_tail_action(1, None) == (
        "connect_audited_endpoint"
    )
    assert _right_guarded_route_tail_action(
        3, {"reason": "live_payload_replan_reserve"}
    ) == "replan"


def test_guarded_right_retreat_retries_only_within_predictive_budget() -> None:
    guarded = {"reason": "live_payload_replan_reserve"}

    assert _right_guarded_retreat_action(None, 1, 2) == "complete"
    assert _right_guarded_retreat_action(guarded, 1, 2) == (
        "retry_from_measured_stop"
    )
    assert _right_guarded_retreat_action(guarded, 2, 2) == "fail_closed"
    assert _right_guarded_retreat_action(
        {"reason": "contact"}, 1, 2
    ) == "fail_closed"
    assert _right_guarded_retreat_action(guarded, 2) == (
        "retry_from_measured_stop"
    )
    assert _right_guarded_retreat_action(guarded, 3) == "fail_closed"



def test_stopped_move_reconstructs_from_last_applied_command() -> None:
    target = np.full(7, 30.0)
    applied = np.arange(7, dtype=np.float64)

    assert np.allclose(
        _movement_measurement_seed(target, applied, {'reason': 'guard'}),
        applied,
    )
    assert np.allclose(
        _movement_measurement_seed(target, applied, None),
        target,
    )
    with np.testing.assert_raises(ValueError):
        _movement_measurement_seed(target, applied[:6], None)



def test_guard_stop_clearance_uses_only_the_pose_physics_accepted() -> None:
    predictive_stop = {
        "reason": "live_payload_replan_reserve",
        "clearance_m": 0.005443,
        "previous_clearance_m": 0.005745,
    }

    assert _guarded_stop_accepted_payload_clearance_m(
        predictive_stop
    ) == pytest.approx(0.005745)
    assert _guarded_stop_accepted_payload_clearance_m(
        {**predictive_stop, "applied_command": True}
    ) == pytest.approx(0.005443)
    with pytest.raises(ValueError):
        _guarded_stop_accepted_payload_clearance_m(
            {"reason": "live_payload_replan_reserve"}
        )

def test_first_inner_settle_requires_a_completed_clear_waypoint() -> None:
    measured = {-0.160: np.zeros(7), -0.140: np.ones(7)}

    assert _right_waypoint_settle_retry_allowed(
        -0.120, 0, False, measured
    )
    assert not _right_waypoint_settle_retry_allowed(
        -0.120, 1, False, measured
    )
    assert not _right_waypoint_settle_retry_allowed(
        -0.120, 0, False, {}
    )
    assert _right_waypoint_settle_retry_allowed(
        -0.120, 4, True, {}
    )
    assert _right_waypoint_settle_retry_allowed(
        -0.140, 0, False, {}, outer_side_m=-0.140
    )


def test_live_right_retreat_uses_nearest_completed_outer_waypoint() -> None:
    measured = {
        -0.160: np.zeros(7),
        -0.140: np.ones(7),
        -0.100: np.full(7, 2.0),
    }

    assert np.isclose(
        _previous_completed_right_waypoint(-0.120, measured),
        -0.140,
    )
    assert _previous_completed_right_waypoint(-0.160, measured) is None


def test_initial_inner_retreat_skips_the_current_completed_posture() -> None:
    measured = {
        -0.160: np.zeros(7),
        -0.140: np.ones(7),
    }

    assert np.isclose(
        _previous_distinct_completed_right_waypoint(
            -0.120, measured, np.ones(7)
        ),
        -0.160,
    )
    assert np.isclose(
        _previous_distinct_completed_right_waypoint(
            -0.120, measured, np.full(7, 1.5)
        ),
        -0.140,
    )
    assert np.isclose(
        _previous_distinct_completed_right_waypoint(
            -0.120,
            measured,
            np.full(7, 1.5),
            preference_rank=1,
        ),
        -0.160,
    )
    assert (
        _previous_distinct_completed_right_waypoint(
            -0.140, measured, np.zeros(7)
        )
        is None
    )


def test_obstructed_right_recovery_resolves_completed_side_then_steps_out() -> None:
    assert np.isclose(
        _right_recovery_cartesian_side_m(-0.160, 0),
        -0.160,
    )
    assert np.isclose(
        _right_recovery_cartesian_side_m(-0.160, 1),
        -0.180,
    )
    assert np.isclose(
        _right_recovery_cartesian_side_m(-0.160, 2),
        -0.200,
    )
    assert np.isclose(
        _right_recovery_cartesian_side_m(-0.160, 8),
        -0.200,
    )
    assert _right_recovery_cartesian_sides(-0.160) == (
        -0.160,
        -0.180,
        -0.200,
    )
    assert _right_recovery_cartesian_sides(-0.200) == (-0.200,)


def test_right_route_shortcut_keeps_only_newly_audited_edges() -> None:
    start = np.asarray([-1.0, 0.0])
    first = np.asarray([-0.5, 1.0])
    second = np.asarray([0.5, 1.0])
    goal = np.asarray([1.0, 0.0])

    def valid(edge_start, edge_end):
        return not (
            np.allclose(edge_start, start)
            and np.allclose(edge_end, goal)
        )

    shortened = _shortcut_collision_clear_joint_route(
        start,
        (first, second),
        goal,
        valid,
    )

    assert len(shortened) == 1
    assert np.allclose(shortened[0], second)
    assert _shortcut_collision_clear_joint_route(
        start,
        (first, second),
        goal,
        lambda _start, _end: True,
    ) == ()



def test_cut_direction_accepts_safe_inward_and_outward_families() -> None:
    robot_forward = np.asarray([0.0, 2.0, 0.0])

    assert (
        _right_cut_direction_aisle_family([0.0, 1.0, 0.0], robot_forward)
        == "into_crop_row"
    )
    assert (
        _right_cut_direction_aisle_family([0.0, -3.0, 0.0], robot_forward)
        == "toward_aisle"
    )
    assert (
        _right_cut_direction_aisle_family([1.0, 0.01, 0.0], robot_forward)
        is None
    )
    with pytest.raises(ValueError):
        _right_cut_direction_aisle_family([0.0, 0.0, 0.0], robot_forward)


def test_committed_cut_searches_local_then_opposed_stroke_family() -> None:
    assert _right_bidirectional_cut_roll_candidates(60.0) == (
        60.0,
        75.0,
        45.0,
        90.0,
        30.0,
        105.0,
        15.0,
        120.0,
        0.0,
        -120.0,
        -105.0,
        -135.0,
        -90.0,
        -150.0,
        -75.0,
        -165.0,
        -60.0,
        -180.0,
    )
    assert _right_bidirectional_cut_roll_candidates(0.0) == (
        0.0,
        15.0,
        -15.0,
        30.0,
        -30.0,
        45.0,
        -45.0,
        60.0,
        -60.0,
        -180.0,
        -165.0,
        165.0,
        -150.0,
        150.0,
        -135.0,
        135.0,
        -120.0,
        120.0,
    )
    with pytest.raises(ValueError):
        _right_bidirectional_cut_roll_candidates(float("nan"))


def test_outer_waypoint_roll_recovery_is_bounded_to_proven_wing() -> None:
    outer = _right_nearby_rolls(0.0, outer_waypoint=True)
    assert outer[:4] == (5.0, -5.0, 10.0, -10.0)
    assert len(outer) == 26
    roll_78_index = outer.index(78.0)
    assert outer[roll_78_index : roll_78_index + 2] == (78.0, -78.0)
    assert outer[-2:] == (120.0, -120.0)
    assert len(_right_nearby_rolls(0.0, outer_waypoint=False)) == 20
    assert all(
        -120.0 <= roll <= 120.0
        for roll in _right_nearby_rolls(118.0, outer_waypoint=True)
    )

def test_committed_cut_refines_routed_roll_into_nearby_direct_frames() -> None:
    rolls = _right_committed_cut_roll_refinement(-66.0)

    assert rolls[0] == -66.0
    assert rolls[1:5] == (-65.0, -67.0, -64.0, -68.0)
    assert -72.0 in rolls
    assert len(rolls) == len(set(rolls))
    assert all(-120.0 <= roll <= 120.0 for roll in rolls)

def test_committed_live_roll_refinement_uses_best_near_clear_routes() -> None:
    rejections = [
        {
            "stub_m": 0.018,
            "roll_degrees": 70.0,
            "edge_wing_local_m": [0.0, -0.018, 0.0],
            "error": "live full-stroke preflight was not clear",
            "segments": [
                {"eligible": False, "payload_clearance_m": 0.005121}
            ],
        },
        {
            "stub_m": 0.019,
            "roll_degrees": 70.0,
            "edge_wing_local_m": [0.0, -0.018, 0.0],
            "error": "live full-stroke preflight was not clear",
            "segments": [
                {"eligible": False, "payload_clearance_m": 0.005437}
            ],
        },
        {
            "stub_m": 0.017,
            "roll_degrees": 55.0,
            "edge_wing_local_m": [0.0, -0.018, 0.0],
            "error": "live alignment endpoint is unavailable",
        },
    ]

    requests = _right_committed_live_roll_refinement_requests(
        rejections,
        {(0.019, 70.0, 0.0, -0.018, 0.0)},
        minimum_source_clearance_m=0.0051,
        maximum_sources=1,
    )

    assert requests[:4] == (
        (0.019, 71.0, (0.0, -0.018, 0.0)),
        (0.019, 69.0, (0.0, -0.018, 0.0)),
        (0.019, 72.0, (0.0, -0.018, 0.0)),
        (0.019, 68.0, (0.0, -0.018, 0.0)),
    )
    assert len(requests) == 20
    assert all(request[0] == 0.019 for request in requests)

def test_committed_live_transverse_refinement_uses_near_clear_plane() -> None:
    near_clear = {
        "stub_m": 0.0198,
        "roll_degrees": 32.375,
        "edge_wing_local_m": [-0.01418102, -0.018369995, 0.0],
        "transverse_offset_rotation_degrees": 0.0,
        "error": "live full-stroke preflight was not clear",
        "segments": [
            *(
                {"eligible": True, "payload_clearance_m": 0.006}
                for _ in range(22)
            ),
            {"eligible": False, "payload_clearance_m": 0.005469},
        ],
    }
    shorter_prefix = {
        "stub_m": 0.0195,
        "roll_degrees": 45.0,
        "edge_wing_local_m": [-0.01418102, -0.018369995, 0.0],
        "error": "live full-stroke preflight was not clear",
        "segments": [
            *(
                {"eligible": True, "payload_clearance_m": 0.007}
                for _ in range(10)
            ),
            {"eligible": False, "payload_clearance_m": 0.00549},
        ],
    }
    wing = (-0.01418102, -0.018369995, 0.0)

    requests = _right_committed_live_transverse_refinement_requests(
        [shorter_prefix, near_clear],
        {(0.0198, 32.375, *wing, 0.0)},
        rotation_offsets_degrees=(-2.0, 2.0),
        maximum_sources=1,
    )

    assert requests == (
        (0.0198, 32.375, wing, -2.0),
        (0.0198, 32.375, wing, 2.0),
    )
    with pytest.raises(ValueError):
        _right_committed_live_transverse_refinement_requests(
            [near_clear],
            (),
            rotation_offsets_degrees=(0.0,),
        )
    with pytest.raises(ValueError):
        _right_committed_live_transverse_refinement_requests(
            [near_clear],
            (),
            maximum_sources=0,
        )


def test_committed_live_refinement_prioritizes_long_safe_endpoint_prefix() -> None:
    near_clear = {
        "stub_m": 0.018,
        "roll_degrees": 70.0,
        "edge_wing_local_m": [0.0, -0.018, 0.0],
        "error": "live full-stroke preflight was not clear",
        "segments": [
            *(
                {"eligible": True, "payload_clearance_m": 0.009}
                for _ in range(4)
            ),
            {"eligible": False, "payload_clearance_m": 0.0054},
        ],
    }
    long_prefix = {
        "stub_m": 0.019,
        "roll_degrees": 45.0,
        "edge_wing_local_m": [0.0, -0.018, 0.0],
        "error": "live full-stroke preflight was not clear",
        "segments": [
            *(
                {"eligible": True, "payload_clearance_m": 0.0078}
                for _ in range(5)
            ),
            {
                "eligible": False,
                "payload_clearance_m": 0.0049,
                "error": "endpoint_payload_clearance",
            },
        ],
    }

    requests = _right_committed_live_roll_refinement_requests(
        [near_clear, long_prefix],
        {
            (0.018, 70.0, 0.0, -0.018, 0.0),
            (0.019, 45.0, 0.0, -0.018, 0.0),
        },
        minimum_source_clearance_m=0.0051,
        maximum_sources=1,
    )

    assert requests[:4] == (
        (0.019, 46.0, (0.0, -0.018, 0.0)),
        (0.019, 44.0, (0.0, -0.018, 0.0)),
        (0.019, 47.0, (0.0, -0.018, 0.0)),
        (0.019, 43.0, (0.0, -0.018, 0.0)),
    )




def test_committed_live_refinement_ranks_equal_prefixes_at_failure() -> None:
    near_guard = {
        "stub_m": 0.019,
        "roll_degrees": 45.0,
        "edge_wing_local_m": [0.0, -0.018, 0.0],
        "error": "live full-stroke preflight was not clear",
        "segments": [
            *(
                {"eligible": True, "payload_clearance_m": 0.0060}
                for _ in range(20)
            ),
            {"eligible": False, "payload_clearance_m": 0.005495},
        ],
    }
    clearance_cliff = {
        "stub_m": 0.017,
        "roll_degrees": 45.0,
        "edge_wing_local_m": [0.0, -0.018, 0.0],
        "error": "live full-stroke preflight was not clear",
        "segments": [
            *(
                {"eligible": True, "payload_clearance_m": 0.0063}
                for _ in range(20)
            ),
            {
                "eligible": False,
                "payload_clearance_m": 0.00509,
                "error": "endpoint_payload_clearance",
            },
        ],
    }

    requests = _right_committed_live_roll_refinement_requests(
        [clearance_cliff, near_guard],
        {
            (0.017, 45.0, 0.0, -0.018, 0.0),
            (0.019, 45.0, 0.0, -0.018, 0.0),
        },
        minimum_source_clearance_m=0.0051,
        maximum_sources=1,
    )

    assert requests[:4] == (
        (0.019, 46.0, (0.0, -0.018, 0.0)),
        (0.019, 44.0, (0.0, -0.018, 0.0)),
        (0.019, 47.0, (0.0, -0.018, 0.0)),
        (0.019, 43.0, (0.0, -0.018, 0.0)),
    )


def test_committed_cut_segment_count_supports_staged_alignment() -> None:
    assert _RIGHT_COMMITTED_CUT_ALIGNMENT_SIDE_CANDIDATES_M == (
        -0.035,
        -0.042,
        -0.050,
        -0.060,
        -0.100,
    )
    assert _RIGHT_COMMITTED_CUT_EXIT_SIDE_M == 0.015
    safe_bimanual_sides = _right_committed_cut_preflight_sides(
        -0.015,
        _RIGHT_COMMITTED_CUT_EXIT_SIDE_M,
    )
    assert np.allclose(
        safe_bimanual_sides,
        tuple(np.arange(-0.01375, 0.0151, 0.00125)),
    )

    assert _right_committed_cut_segment_count(-0.015, 0.035) == 40
    assert _right_committed_cut_segment_count(-0.035, 0.035) == 56
    assert _right_committed_cut_segment_count(-0.060, 0.035) == 76
    with pytest.raises(ValueError):
        _right_committed_cut_segment_count(-0.034, 0.035)

    sides = _right_committed_cut_segment_sides(-0.035, 0.0)
    assert np.allclose(
        sides,
        tuple(np.arange(-0.03375, 0.0001, 0.00125)),
    )
    long_sides = _right_committed_cut_segment_sides(-0.100, 0.0)
    assert len(long_sides) == 80
    assert np.isclose(long_sides[0], -0.09875)
    assert long_sides[-1] == 0.0

    preflight_sides = _right_committed_cut_preflight_sides(-0.035, 0.035)
    assert len(preflight_sides) == 56
    assert np.isclose(preflight_sides[0], -0.03375)
    assert np.isclose(preflight_sides[27], 0.0)
    assert np.isclose(preflight_sides[-1], 0.035)


    target_screen_sides = _right_target_screen_cut_sides()
    assert len(target_screen_sides) == 41
    assert target_screen_sides[0] == -0.035
    assert np.isclose(target_screen_sides[28], 0.0)
    assert target_screen_sides[-1] == _RIGHT_COMMITTED_CUT_EXIT_SIDE_M



def test_committed_cut_stub_search_stays_bounded_and_near_nominal() -> None:
    assert _right_cut_stub_search_order(
        0.016,
        _RIGHT_COMMITTED_CUT_STUB_CANDIDATES_M,
    ) == (
        0.016,
        0.015,
        0.017,
        0.014,
        0.018,
        0.013,
        0.019,
        0.0195,
        0.0198,
    )
    assert max(_RIGHT_COMMITTED_CUT_STUB_CANDIDATES_M) <= 0.020
    assert max(_RIGHT_COMMITTED_CUT_STUB_CANDIDATES_M) + 0.00514 < 0.025
    with pytest.raises(ValueError):
        _right_cut_stub_search_order(0.016, ())



def test_committed_cut_configuration_matches_only_exact_preplanned_frame() -> None:
    hint = {
        "cut_stub_m": 0.019,
        "roll_degrees": 78.0,
        "edge_wing_local_m": [-0.01418102, -0.001, 0.0],
    }

    assert _right_committed_cut_configuration_matches_hint(
        0.019,
        78.0,
        [-0.01418102, -0.001, 0.0],
        hint,
    )
    assert not _right_committed_cut_configuration_matches_hint(
        0.018,
        78.0,
        [-0.01418102, -0.001, 0.0],
        hint,
    )
    assert not _right_committed_cut_configuration_matches_hint(
        0.019,
        75.0,
        [-0.01418102, -0.001, 0.0],
        hint,
    )
    assert not _right_committed_cut_configuration_matches_hint(
        0.019,
        78.0,
        [-0.01418102, -0.006, 0.0],
        hint,
    )
    assert not _right_committed_cut_configuration_matches_hint(
        0.019,
        78.0,
        [-0.01418102, -0.001, 0.0],
        None,
    )

def test_selected_knife_roll_rotates_the_physical_cut_direction() -> None:
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    preferred = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    base = robot_hardware.cut_aligned_knife_rotation(axis, preferred)
    zero_roll = _knife_rotation_for_roll(
        axis, preferred, 0.0, robot_hardware
    )
    selected = _knife_rotation_for_roll(
        axis, preferred, 78.0, robot_hardware
    )

    base_direction = base @ robot_hardware.KNIFE_CUT_DIRECTION_LOCAL
    selected_direction = (
        selected @ robot_hardware.KNIFE_CUT_DIRECTION_LOCAL
    )
    cosine = float(
        np.clip(np.dot(base_direction, selected_direction), -1.0, 1.0)
    )
    angle_degrees = np.degrees(np.arccos(cosine))

    assert np.allclose(zero_roll, base)
    assert np.allclose(selected.T @ selected, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(selected), 1.0)
    assert np.isclose(angle_degrees, 78.0)
def test_selected_right_goal_direction_matches_exact_roll_and_wing() -> None:
    exact = {
        "roll_degrees": 60.0,
        "edge_wing_local_m": [-0.01418102, -0.001, 0.0],
        "cut_direction": [-0.87, 0.47, 0.10],
    }
    goal = {
        "selected_roll_degrees": 60.0,
        "selected_edge_wing_local_m": [-0.01418102, -0.001, 0.0],
        "direction_candidates": [
            {
                "roll_degrees": 78.0,
                "edge_wing_local_m": [-0.01418102, -0.001, 0.0],
            },
            {
                "roll_degrees": 60.0,
                "edge_wing_local_m": [-0.01418102, -0.018, 0.0],
            },
            exact,
        ],
    }

    assert _selected_right_goal_direction(goal) is exact





def test_committed_cut_segment_timing_tracks_without_falling_below_cut_gate() -> None:
    physics_dt_s = 1.0 / 240.0
    minimum_cut_speed_m_s = 0.01

    steps = _right_committed_cut_segment_steps(
        180,
        physics_dt_s,
        minimum_cut_speed_m_s,
    )
    assert steps == 24
    assert (
        _RIGHT_COMMITTED_CUT_SEGMENT_LENGTH_M / (steps * physics_dt_s)
        == pytest.approx(0.0125)
    )
    assert (
        _RIGHT_COMMITTED_CUT_SEGMENT_LENGTH_M / (steps * physics_dt_s)
        > minimum_cut_speed_m_s
    )

    bounded_slow_steps = _right_committed_cut_segment_steps(
        600,
        physics_dt_s,
        minimum_cut_speed_m_s,
    )
    assert bounded_slow_steps == 27
    assert (
        _RIGHT_COMMITTED_CUT_SEGMENT_LENGTH_M
        / (bounded_slow_steps * physics_dt_s)
        >= 1.10 * minimum_cut_speed_m_s
    )

    with pytest.raises(ValueError):
        _right_committed_cut_segment_steps(0, physics_dt_s, minimum_cut_speed_m_s)
    with pytest.raises(ValueError):
        _right_committed_cut_segment_steps(180, 0.0, minimum_cut_speed_m_s)
    with pytest.raises(ValueError):
        _right_committed_cut_segment_steps(180, physics_dt_s, 0.0)


def test_committed_cut_transverse_registration_is_bounded() -> None:
    correction = _right_committed_cut_bounded_transverse_correction(
        np.zeros(3),
        np.asarray([0.006, 0.0, 0.0]),
    )
    assert np.allclose(correction, [0.002, 0.0, 0.0])

    repeated = _right_committed_cut_bounded_transverse_correction(
        correction,
        np.asarray([0.006, 0.0, 0.0]),
    )
    assert np.allclose(repeated, correction)
    redirected = _right_committed_cut_bounded_transverse_correction(
        repeated,
        np.asarray([-0.001, 0.002, 0.0]),
    )
    assert np.linalg.norm(redirected) == pytest.approx(0.002)

    with pytest.raises(ValueError):
        _right_committed_cut_bounded_transverse_correction(
            np.zeros(2), np.zeros(3)
        )


def test_committed_cut_registration_uses_absolute_target_drift() -> None:
    correction = _right_committed_cut_target_drift_correction(
        np.zeros(3),
        np.asarray([0.006, 0.004, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    )
    assert np.allclose(correction, [0.0, 0.002, 0.0])

    repeated = _right_committed_cut_target_drift_correction(
        np.zeros(3),
        np.asarray([0.006, 0.004, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    )
    assert np.allclose(repeated, correction)

    advanced_along_cut = _right_committed_cut_target_drift_correction(
        np.zeros(3),
        np.asarray([0.010, 0.004, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    )
    assert np.allclose(advanced_along_cut, correction)

    with pytest.raises(ValueError):
        _right_committed_cut_target_drift_correction(
            np.zeros(3), np.ones(3), np.zeros(3)
        )


def test_committed_cut_tracking_feedback_is_transverse_bounded_and_fresh() -> None:
    correction = _right_committed_cut_tracking_feedback_correction(
        np.asarray([0.004, 0.003, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    )
    assert np.allclose(correction, [0.0, 0.001, 0.0])

    opposite = _right_committed_cut_tracking_feedback_correction(
        np.asarray([0.0, -0.0004, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
    )
    assert np.allclose(opposite, [0.0, -0.0004, 0.0])
    with pytest.raises(ValueError):
        _right_committed_cut_tracking_feedback_correction(
            np.zeros(3),
            np.zeros(3),
        )
    with pytest.raises(ValueError):
        _right_committed_cut_tracking_feedback_correction(
            np.zeros(3),
            np.ones(3),
            maximum_correction_m=0.0,
        )


def test_committed_cut_latches_transverse_plane_after_physical_load() -> None:
    precontact_latch, precontact, latched_now = (
        _right_committed_cut_contact_plane_correction(
            None,
            np.asarray([0.0002, -0.0004, 0.0001]),
            0.9,
        )
    )
    assert precontact_latch is None
    assert not latched_now
    assert np.allclose(precontact, [0.0002, -0.0004, 0.0001])

    force_latch, force_tracked, latched_now = (
        _right_committed_cut_contact_plane_correction(
            None,
            np.asarray([0.0018, 0.0, 0.0]),
            66.29,
            latch_force_n=66.3,
        )
    )
    assert force_latch is None
    assert not latched_now
    assert np.allclose(force_tracked, [0.0018, 0.0, 0.0])

    force_latch, force_loaded, latched_now = (
        _right_committed_cut_contact_plane_correction(
            None,
            np.asarray([0.002, 0.0, 0.0]),
            66.3,
            latch_force_n=66.3,
        )
    )
    assert latched_now
    assert np.allclose(force_latch, [0.002, 0.0, 0.0])
    assert np.allclose(force_loaded, force_latch)

    latch, loaded, latched_now = _right_committed_cut_contact_plane_correction(
        None,
        np.asarray([0.0003, -0.0002, 0.0001]),
        1.0,
    )
    assert latched_now
    assert np.allclose(latch, [0.0003, -0.0002, 0.0001])
    assert np.allclose(loaded, latch)

    reused, commanded, latched_now = (
        _right_committed_cut_contact_plane_correction(
            latch,
            np.asarray([0.002, 0.0, 0.0]),
            44.0,
        )
    )
    assert not latched_now
    assert np.allclose(reused, latch)
    assert np.allclose(commanded, latch)

    with pytest.raises(ValueError):
        _right_committed_cut_contact_plane_correction(
            None,
            np.zeros(2),
            1.0,
        )

def test_committed_cut_live_tracking_rebases_until_force_contact() -> None:
    precontact = _right_committed_cut_live_tracking_update(
        np.zeros(3),
        np.asarray([0.0, 0.008, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0007]),
        None,
        0.0,
        latch_force_n=66.3,
    )
    assert precontact["rebase_to_live"]
    assert precontact["contact_plane_correction"] is None
    assert np.allclose(precontact["target_drift_correction"], np.zeros(3))
    assert np.allclose(
        precontact["tracking_correction"], [0.0, 0.0, 0.0007]
    )

    loaded = _right_committed_cut_live_tracking_update(
        np.zeros(3),
        np.asarray([0.0, 0.008, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, 0.0007]),
        None,
        66.3,
        latch_force_n=66.3,
    )
    assert loaded["rebase_to_live"]
    assert loaded["latched_now"]
    assert np.allclose(
        loaded["contact_plane_correction"], [0.0, 0.0, 0.0007]
    )

    retained = _right_committed_cut_live_tracking_update(
        np.asarray([0.0, 0.008, 0.0]),
        np.asarray([0.0, 0.012, 0.0]),
        np.asarray([1.0, 0.0, 0.0]),
        np.asarray([0.0, 0.0, -0.0004]),
        loaded["contact_plane_correction"],
        44.0,
        latch_force_n=66.3,
    )
    assert not retained["rebase_to_live"]
    assert not retained["latched_now"]
    assert np.allclose(
        retained["tracking_correction"],
        loaded["contact_plane_correction"],
    )


def test_committed_cut_fracture_cycle_retracts_and_reloads_with_fresh_motion() -> None:
    cycle = _right_committed_cut_fracture_cycle_sides(0.015)
    assert [phase for phase, _, _ in cycle] == [
        "unload_01",
        "unload_02",
        "unload_03",
        "unload_04",
        "reload_01",
        "reload_02",
        "reload_03",
        "reload_04",
    ]
    assert np.allclose(
        [side for _, side, _ in cycle],
        [
            0.01375, 0.0125, 0.01125, 0.0100,
            0.01125, 0.0125, 0.01375, 0.0150,
        ],
    )
    assert [qualifies for _, _, qualifies in cycle] == [
        False, False, False, False, True, True, True, True
    ]
    starts = (("start", 0.015, False), *cycle)
    assert all(
        not np.isclose(first[1], second[1], atol=1e-12, rtol=0.0)
        for first, second in zip(starts, cycle)
    )
    with pytest.raises(ValueError):
        _right_committed_cut_fracture_cycle_sides(
            0.015,
            reload_distance_m=0.004,
            segment_length_m=0.0025,
        )


def test_committed_cut_prefers_interior_edge_margin_before_clearance() -> None:
    wings = (
        (-0.01418102, -0.07047998, 0.0),
        (-0.01418102, -0.018369995, 0.0),
        (-0.01418102, -0.001, 0.0),
    )
    tip = {
        "goal": {"selected_edge_wing_local_m": wings[-1]},
        "payload_clearance_m": 0.0126,
    }
    interior = {
        "goal": {"selected_edge_wing_local_m": wings[1]},
        "payload_clearance_m": 0.00559,
    }

    interior_rank = _right_committed_cut_candidate_rank(interior, wings)
    tip_rank = _right_committed_cut_candidate_rank(tip, wings)
    assert interior_rank[0] == pytest.approx(0.017369995)
    assert tip_rank[0] == 0.0
    assert interior_rank > tip_rank


def test_committed_alignment_retreat_reverses_measured_approach() -> None:
    measured_sides = (
        -0.200,
        -0.180,
        -0.160,
        -0.140,
        -0.120,
        -0.100,
        -0.060,
        -0.050,
        -0.042,
        -0.035,
        -0.030,
        -0.025,
        -0.020,
        -0.015,
    )

    assert _right_committed_alignment_retreat_sides(
        -0.035, measured_sides
    ) == (-0.020, -0.025, -0.030, -0.035)
    assert _right_committed_alignment_retreat_route_candidates(
        (-0.020, -0.025, -0.030, -0.035)
    ) == (
        (-0.035,),
        (-0.030, -0.035),
        (-0.025, -0.030, -0.035),
        (-0.020, -0.025, -0.030, -0.035),
    )
    assert _right_committed_alignment_retreat_sides(
        -0.100, measured_sides
    ) == (
        -0.020,
        -0.025,
        -0.030,
        -0.035,
        -0.042,
        -0.050,
        -0.060,
        -0.100,
    )
    with pytest.raises(ValueError):
        _right_committed_alignment_retreat_sides(-0.055, measured_sides)
    with pytest.raises(ValueError):
        _right_committed_alignment_retreat_sides(float("nan"), measured_sides)


    with pytest.raises(ValueError):
        _right_committed_alignment_retreat_route_candidates(())
    with pytest.raises(ValueError):
        _right_committed_alignment_retreat_route_candidates((-0.035, -0.035))
def test_live_route_replan_expands_fixed_frame_before_full_search() -> None:
    searches = _right_live_route_replan_searches(45.0)

    assert searches[0] == (
        'live_receding_horizon_fixed',
        (45.0,),
    )
    assert searches[1][0] == 'live_receding_horizon_nearby'
    assert searches[1][1][:4] == (50.0, 40.0, 55.0, 35.0)
    assert 45.0 not in searches[1][1]
    assert searches[-1] == ('live_receding_horizon_full', None)


def test_live_waypoint_replan_changes_roll_but_keeps_blade_wing() -> None:
    rolls, wings = _right_live_waypoint_replan_overrides(
        30.0,
        [-0.014, -0.07047998, 0.0],
    )

    assert rolls[:4] == (31.0, 29.0, 32.0, 28.0)
    assert 30.0 not in rolls
    assert len(wings) == 1
    assert np.allclose(wings[0], [-0.014, -0.07047998, 0.0])


def test_live_waypoint_replan_searches_nearest_rolls_progressively() -> None:
    searches = _right_live_waypoint_replan_searches(
        30.0,
        [-0.014, -0.07047998, 0.0],
    )

    assert [search[0] for search in searches] == [
        "live_waypoint_nearby_batch_1",
        "live_waypoint_nearby_batch_2",
        "live_waypoint_nearby_batch_3",
        "live_waypoint_nearby_batch_4",
    ]
    assert searches[0][1] == (31.0, 29.0, 32.0, 28.0)
    assert searches[1][1] == (33.0, 27.0, 34.0, 26.0)
    assert tuple(
        roll for _, rolls, _ in searches for roll in rolls
    ) == _right_nearby_rolls(30.0, outer_waypoint=False)
    assert all(len(wings) == 1 for _, _, wings in searches)
    assert all(
        np.allclose(wings[0], [-0.014, -0.07047998, 0.0])
        for _, _, wings in searches
    )


def test_right_route_uses_measured_settled_foliage_reserve() -> None:
    report = {
        "preplanning_vine_settle": {
            "planning_foliage_clearance_mm": 1.789,
        }
    }

    assert np.isclose(
        _right_route_foliage_planning_clearance_m(report),
        _RIGHT_COMMITTED_CUT_LIVE_REPLAN_TRIGGER_M,
    )
    assert np.isclose(
        _right_route_foliage_planning_clearance_m({}),
        _RIGHT_COMMITTED_CUT_LIVE_REPLAN_TRIGGER_M,
    )
    assert _RIGHT_ROUTE_REPLAN_LIMIT == 8
    assert _RIGHT_PRESTAGE_ROUTE_REPLAN_LIMIT == 13
    assert _RIGHT_PRESTAGE_SEGMENT_LIMIT == 48
    assert _RIGHT_PRESTAGE_PLANNING_SETTLE_LIMIT == 3
    assert np.isclose(
        _right_route_foliage_planning_clearance_m(
            {
                "preplanning_vine_settle": {
                    "planning_foliage_clearance_mm": float("nan"),
                }
            }
        ),
        _RIGHT_COMMITTED_CUT_LIVE_REPLAN_TRIGGER_M,
    )


def test_right_endpoint_hold_preserves_live_foliage_replan_reserve() -> None:
    report = {
        'preplanning_vine_settle': {
            'planning_foliage_clearance_mm': 1.65,
        }
    }
    foliage = {
        'nearest_obstacle': '/World/InteractiveVines/Vine/FoliageContact_1',
    }

    assert np.isclose(
        _endpoint_hold_clearance_guard_m(
            'right', foliage, 0.0005, report
        ),
        _RIGHT_COMMITTED_CUT_LIVE_REPLAN_TRIGGER_M,
    )
    assert np.isclose(
        _endpoint_hold_clearance_guard_m(
            'left', foliage, 0.0005, report
        ),
        0.0005,
    )
    assert np.isclose(
        _endpoint_hold_clearance_guard_m(
            'right', {'nearest_obstacle': '/World/Gutter'}, 0.005, report
        ),
        0.005,
    )

def test_endpoint_hold_replans_only_from_positive_dynamic_vine_gap() -> None:
    foliage = '/World/InteractiveVines/Vine/FoliageContact_1'

    assert _endpoint_hold_replan_stop(
        0.00014, foliage, 0.0056, 0.0060, 70
    )
    assert not _endpoint_hold_replan_stop(
        0.00009, foliage, 0.0056, 0.0060, 70
    )
    assert not _endpoint_hold_replan_stop(
        -0.0001, foliage, 0.0056, 0.0060, 70
    )
    assert not _endpoint_hold_replan_stop(
        0.00014, foliage, 0.0056, None, 70
    )
    assert not _endpoint_hold_replan_stop(
        0.00014, '/World/Greenhouse/Gutter', 0.0056, 0.0060, 70
    )



def test_live_right_route_retries_only_positive_gap_dynamic_foliage() -> None:
    candidate = {
        'solution': {'succeeded': True},
        'endpoint_state_payload_clearance': {'clearance_m': 0.019},
        'endpoint_state_required_payload_clearance_m': 0.0005,
        'endpoint_state_inter_arm_clearance_m': 0.086,
        'cartesian_entry_route_search': {
            'candidates': [
                {
                    'failure': {
                        'kind': 'clearance',
                        'payload': {
                            'clearance_m': 0.002423,
                            'required_clearance_m': 0.002631,
                            'runtime_required_clearance_m': 0.0005,
                            'nearest_obstacle': (
                                '/World/InteractiveVines/Vine_0002/'
                                'FoliageContact_0178'
                            ),
                        },
                        'inter_arm': {'clearance_m': 0.086},
                    }
                }
            ]
        },
    }

    assert _right_live_route_settle_retryable([candidate])
    candidate['cartesian_entry_route_search']['candidates'][0][
        'failure'
    ]['payload']['clearance_m'] = -0.001
    assert not _right_live_route_settle_retryable([candidate])
    candidate['cartesian_entry_route_search']['candidates'][0][
        'failure'
    ]['payload']['clearance_m'] = 0.002423
    candidate['cartesian_entry_route_search']['candidates'][0][
        'failure'
    ]['payload']['nearest_obstacle'] = '/World/Greenhouse/Gutter'
    assert not _right_live_route_settle_retryable([candidate])


def test_live_inner_waypoint_retries_a_narrow_foliage_endpoint_deficit() -> None:
    candidate = {
        'endpoint_payload_clearance': {
            'clearance_m': 0.002631,
            'nearest_obstacle': (
                '/World/InteractiveVines/Vine_0002/'
                'FoliageContact_0178'
            ),
        },
        'required_payload_clearance_m': 0.002757,
        'rejection': 'endpoint_payload_clearance',
    }

    assert _right_live_route_settle_retryable([candidate])
    candidate['endpoint_payload_clearance']['clearance_m'] = 0.00055
    assert not _right_live_route_settle_retryable([candidate])
    candidate['endpoint_payload_clearance']['clearance_m'] = 0.002631
    candidate['endpoint_payload_clearance'][
        'nearest_obstacle'
    ] = '/World/Greenhouse/Gutter'
    assert not _right_live_route_settle_retryable([candidate])


def test_active_retreat_allows_a_wider_positive_gap_foliage_deficit() -> None:
    candidate = {
        'endpoint_payload_clearance': {
            'clearance_m': 0.001422,
            'nearest_obstacle': (
                '/World/InteractiveVines/Vine_0002/Physics/'
                'Organ_0112/Link_000/FoliageContact_0178_01'
            ),
        },
        'required_payload_clearance_m': 0.002670,
        'rejection': 'endpoint_payload_clearance',
    }

    assert not _right_live_route_settle_retryable([candidate])
    assert _right_live_route_active_retreat_retryable([candidate])
    candidate['endpoint_payload_clearance']['clearance_m'] = 0.00055
    assert not _right_live_route_active_retreat_retryable([candidate])
    candidate['endpoint_payload_clearance']['clearance_m'] = 0.001422
    candidate['endpoint_payload_clearance'][
        'nearest_obstacle'
    ] = '/World/Greenhouse/Gutter'
    assert not _right_live_route_active_retreat_retryable([candidate])


def test_provisional_retreat_floor_requires_exact_positive_foliage_gap() -> None:
    trajectory = {
        'clearance_m': 0.000789,
        'nearest_obstacle': (
            '/World/InteractiveVines/Vine_0002/'
            'FoliageContact_0181'
        ),
    }

    assert np.isclose(
        _right_provisional_retreat_floor_m(True, trajectory),
        0.0005,
    )
    assert _right_provisional_retreat_floor_m(False, trajectory) is None
    trajectory['clearance_m'] = 0.00049
    assert _right_provisional_retreat_floor_m(True, trajectory) is None
    trajectory['clearance_m'] = 0.000789
    trajectory['nearest_obstacle'] = '/World/Greenhouse/Gutter'
    assert _right_provisional_retreat_floor_m(True, trajectory) is None


def test_failed_live_route_defers_only_guarded_positive_gap_foliage() -> None:
    termination = {
        'reason': 'live_payload_replan_reserve',
        'previous_clearance_m': 0.00289,
    }
    candidate = {
        'endpoint_payload_clearance': {
            'clearance_m': 0.002216,
            'nearest_obstacle': (
                '/World/InteractiveVines/Vine_0002/Physics/'
                'Organ_0112/Link_000/FoliageContact_0178_01'
            ),
        },
        'required_payload_clearance_m': 0.002646,
        'rejection': 'endpoint_payload_clearance',
    }

    assert _right_failed_live_route_recovery_eligible(
        termination, [candidate]
    )
    assert not _right_failed_live_route_recovery_eligible(
        {'reason': 'contact', 'previous_clearance_m': 0.00289},
        [candidate],
    )
    candidate['endpoint_payload_clearance']['clearance_m'] = 0.00055
    assert not _right_failed_live_route_recovery_eligible(
        termination, [candidate]
    )


def test_provisional_right_route_is_narrow_runtime_clear_live_foliage_only() -> None:
    payload = {
        'clearance_m': 0.002423,
        'nearest_obstacle': (
            '/World/InteractiveVines/Vine_0002/FoliageContact_0178'
        ),
    }

    assert _right_provisional_live_foliage_route_acceptable(
        payload, 0.0005, 0.002709, 0.086
    )
    payload['clearance_m'] = 0.0021
    assert _right_provisional_live_foliage_route_acceptable(
        payload, 0.0005, 0.002709, 0.086
    )
    payload['clearance_m'] = 0.0016
    assert _right_provisional_live_foliage_route_acceptable(
        payload, 0.0005, 0.002709, 0.086
    )
    payload['clearance_m'] = 0.0009
    assert _right_provisional_live_foliage_route_acceptable(
        payload, 0.0005, 0.002709, 0.086
    )
    payload['clearance_m'] = 0.00055
    assert not _right_provisional_live_foliage_route_acceptable(
        payload, 0.0005, 0.002709, 0.086
    )
    payload['clearance_m'] = 0.004
    assert not _right_provisional_live_foliage_route_acceptable(
        payload, 0.005, 0.006, 0.086
    )
    payload['clearance_m'] = 0.002423
    payload['nearest_obstacle'] = '/World/Greenhouse/Gutter'
    assert not _right_provisional_live_foliage_route_acceptable(
        payload, 0.0005, 0.002709, 0.086
    )


def test_right_approach_rejects_only_numerically_saturated_joint_branches() -> None:
    assert _right_approach_joint_reserve_acceptable(0.874)
    assert _right_approach_joint_reserve_acceptable(0.5)
    assert not _right_approach_joint_reserve_acceptable(0.000573)
    assert not _right_approach_joint_reserve_acceptable(float('nan'))


def test_right_route_endpoint_admits_only_positive_escape_candidates() -> None:
    assert _right_route_endpoint_joint_reserve_candidate(0.000573)
    assert _right_route_endpoint_joint_reserve_candidate(0.5)
    assert not _right_route_endpoint_joint_reserve_candidate(0.0)
    assert not _right_route_endpoint_joint_reserve_candidate(-0.001)
    assert not _right_route_endpoint_joint_reserve_candidate(float("nan"))


def test_right_motion_reserve_excludes_only_orientation_wrist_joint() -> None:
    class LimitsModel:
        def arm_limits_degrees(self, _side):
            return -np.full(7, 10.0), np.full(7, 10.0)

    wrist_limited = np.zeros(7)
    wrist_limited[5] = 10.0
    elbow_limited = wrist_limited.copy()
    elbow_limited[2] = 9.8

    assert np.isclose(
        _right_motion_joint_limit_margin_degrees(
            LimitsModel(), wrist_limited
        ),
        10.0,
    )
    assert np.isclose(
        _right_motion_joint_limit_margin_degrees(
            LimitsModel(), elbow_limited
        ),
        0.2,
    )


def test_joint_reserve_recovery_requires_monotonic_positive_escape() -> None:
    assert _joint_reserve_recovery_acceptable(
        0.000573, (0.000573, 1.319, 2.638, 15.680)
    )
    assert _joint_reserve_recovery_acceptable(0.8, (0.8, 0.7, 0.6))
    assert not _joint_reserve_recovery_acceptable(
        0.000573, (0.000573, 0.4, 0.3, 0.8)
    )
    assert not _joint_reserve_recovery_acceptable(
        0.000573, (0.000573, 0.2, 0.49)
    )
    assert not _joint_reserve_recovery_acceptable(
        -0.001, (-0.001, 0.8)
    )
    assert not _joint_reserve_recovery_acceptable(
        0.8, (0.8, 0.49, 0.9)
    )


def test_right_joint_reserve_recovery_moves_only_saturated_non_wrist_axes() -> None:
    class LimitsModel:
        @staticmethod
        def arm_limits_degrees(_side):
            return np.full(7, -180.0), np.full(7, 180.0)

    joints = np.zeros(7)
    joints[2] = 179.9994
    joints[5] = 179.9994
    target = _right_joint_reserve_recovery_target(LimitsModel(), joints)

    assert target is not None
    assert np.isclose(target[2], 175.0)
    assert np.isclose(target[5], joints[5])
    assert np.allclose(np.delete(target, (2, 5)), 0.0)
    assert _right_joint_reserve_recovery_target(
        LimitsModel(), np.zeros(7)
    ) is None


def test_right_joint_reserve_recovery_candidates_add_local_camera_detours() -> None:
    class LimitsModel:
        @staticmethod
        def arm_limits_degrees(_side):
            return np.full(7, -180.0), np.full(7, 180.0)

    joints = np.zeros(7)
    joints[2] = 179.9994
    candidates = _right_joint_reserve_recovery_candidates(
        LimitsModel(), joints
    )

    assert candidates[0]['mode'] == 'direct'
    assert np.isclose(candidates[0]['joint_degrees'][2], 175.0)
    assert candidates[1]['compensation_joints'] == (6,)
    assert candidates[1]['compensation_offsets_degrees'] == (5.0,)
    assert np.isclose(candidates[1]['joint_degrees'][6], 5.0)
    assert candidates[2]['compensation_offsets_degrees'] == (-5.0,)
    assert any(
        candidate['mode'] == 'two_joint_compensation'
        for candidate in candidates
    )
    assert _right_joint_reserve_recovery_candidates(
        LimitsModel(), np.zeros(7)
    ) == ()


def test_live_payload_servo_replans_before_decreasing_foliage_command() -> None:
    obstacle = '/World/InteractiveVines/Vine/FoliageContact_1'

    assert _live_payload_replan_stop(0.0003, obstacle, 0.0026, 0.0008, 2)
    assert not _live_payload_replan_stop(0.0008, obstacle, 0.0026, 0.0003, 2)
    assert not _live_payload_replan_stop(0.0003, obstacle, 0.0026, None, 0)
    # A live retry carries the previous measured-safe clearance into its
    # first predicted command so repeated replans cannot ratchet toward a leaf.
    assert _live_payload_replan_stop(
        0.0007, obstacle, 0.0026, 0.0008, 0
    )
    # Clearance is evaluated for the next command before drive targets are
    # authored. A predicted penetration must stop at the prior safe state.
    assert _live_payload_replan_stop(
        -0.0012, obstacle, 0.0026, 0.0007, 84
    )
    assert not _live_payload_replan_stop(
        -0.0012, obstacle, 0.0026, None, 0
    )
    assert not _live_payload_replan_stop(
        0.0003, '/World/Greenhouse/Gutter', 0.0026, 0.0008, 2
    )
    rigid_vine = (
        '/World/InteractiveVines/Vine_0002/Physics/Organ_0029/'
        'Link_054/Collider'
    )
    assert not _live_payload_replan_stop(
        0.0053, rigid_vine, 0.0055, 0.0054, 20
    )
    assert _live_payload_replan_stop(
        0.0053,
        rigid_vine,
        0.0055,
        0.0054,
        20,
        include_rigid_vines=True,
    )
    # Broad foliage keeps its compliant route guard while rigid stem capsules
    # can use the larger committed-cut reserve in the same staged motion.
    assert _live_payload_replan_stop(
        0.0053,
        rigid_vine,
        0.0026,
        0.0054,
        20,
        include_rigid_vines=True,
        rigid_vine_guard_m=0.0055,
    )
    assert not _live_payload_replan_stop(
        0.0053,
        obstacle,
        0.0026,
        0.0054,
        20,
        include_rigid_vines=True,
        rigid_vine_guard_m=0.0055,
    )
    assert not _live_payload_replan_stop(
        0.0053,
        '/World/Greenhouse/Gutter',
        0.0055,
        0.0054,
        20,
        include_rigid_vines=True,
    )
    # A measured rigid-stem recovery can absorb one bounded physics fluctuation
    # but cannot ratchet below the floor anchored at its first safe stop.
    assert not _live_payload_replan_stop(
        0.005448,
        rigid_vine,
        0.0055,
        0.005468,
        1,
        include_rigid_vines=True,
        decrease_tolerance_m=0.000025,
        recovery_floor_m=0.005443,
    )
    assert _live_payload_replan_stop(
        0.005442,
        rigid_vine,
        0.0055,
        0.005448,
        2,
        include_rigid_vines=True,
        decrease_tolerance_m=0.000025,
        recovery_floor_m=0.005443,
    )
    with pytest.raises(ValueError):
        _live_payload_replan_stop(
            0.005448,
            rigid_vine,
            0.0055,
            0.005468,
            1,
            include_rigid_vines=True,
            decrease_tolerance_m=-0.000001,
        )


def test_committed_cut_trigger_preserves_braking_room_above_guard() -> None:
    rigid_vine = (
        "/World/InteractiveVines/Vine_0002/Physics/Organ_0029/"
        "Link_054/Collider"
    )
    assert _live_payload_replan_stop(
        0.00558,
        rigid_vine,
        _RIGHT_COMMITTED_CUT_LIVE_REPLAN_TRIGGER_M,
        0.00561,
        28,
        include_rigid_vines=True,
    )
    assert not _live_payload_replan_stop(
        0.00558,
        rigid_vine,
        _RIGHT_COMMITTED_CUT_LIVE_REPLAN_GUARD_M,
        0.00561,
        28,
        include_rigid_vines=True,
    )


def test_committed_cut_retries_only_predictive_right_vine_stops() -> None:
    guarded = {
        'reason': 'live_payload_replan_reserve',
        'payload_side': 'right',
        'nearest_obstacle': '/World/InteractiveVines/Vine/Collider',
    }
    assert _right_committed_cut_replan_action(None, 0, 4) == 'complete'
    assert _right_committed_cut_replan_action(guarded, 0, 4) == (
        'retry_from_measured_stop'
    )
    assert _right_committed_cut_replan_action(guarded, 4, 4) == 'fail_closed'
    assert _right_committed_cut_replan_action(
        {**guarded, 'payload_side': 'left'}, 0, 4
    ) == 'fail_closed'
    assert _right_committed_cut_replan_action(guarded, 7) == (
        'retry_from_measured_stop'
    )
    assert _right_committed_cut_replan_action(guarded, 8) == 'fail_closed'
    foliage_error = (
        "no right tool endpoint clears the live crop geometry; best is "
        "1.2 mm below reserve: /World/InteractiveVines/Vine/"
        "FoliageContact_01"
    )
    assert _right_committed_cut_solve_retryable(foliage_error)
    assert not _right_committed_cut_solve_retryable(
        foliage_error.replace("FoliageContact_01", "Collider")
    )


def test_committed_cut_recovers_safe_subreserve_rigid_endpoint() -> None:
    error = (
        "no right tool endpoint clears the live crop geometry; best is "
        "5.1 mm below a 5.5 mm reserve"
    )
    rigid = {
        "roll_degrees": 68.0,
        "endpoint_payload_clearance": {
            "component": "knife_blade",
            "clearance_m": 0.0051276664556703586,
            "nearest_obstacle": (
                "/World/InteractiveVines/Vine_0002/Physics/"
                "Organ_0029/Link_054/Collider"
            ),
        },
        "required_payload_clearance_m": 0.0055,
        "eligible": False,
    }
    screen = {
        "stage": "right_endpoint_screen",
        "rejections": [rigid],
        "succeeded": False,
    }
    selected = _right_committed_cut_rigid_endpoint_rejection(
        error,
        screen,
    )
    assert selected == rigid
    below_hard_floor = {
        **rigid,
        "endpoint_payload_clearance": {
            **rigid["endpoint_payload_clearance"],
            "clearance_m": 0.00499,
        },
    }
    assert _right_committed_cut_rigid_endpoint_rejection(
        error,
        {**screen, "rejections": [below_hard_floor]},
    ) == below_hard_floor
    contacting_endpoint = {
        **rigid,
        "endpoint_payload_clearance": {
            **rigid["endpoint_payload_clearance"],
            "clearance_m": 0.0,
        },
    }
    assert _right_committed_cut_rigid_endpoint_rejection(
        error,
        {**screen, "rejections": [contacting_endpoint]},
    ) is None
    foliage = {
        **rigid,
        "endpoint_payload_clearance": {
            **rigid["endpoint_payload_clearance"],
            "nearest_obstacle": (
                rigid["endpoint_payload_clearance"]["nearest_obstacle"]
                + "/FoliageContact_0001"
            ),
        },
    }
    assert _right_committed_cut_rigid_endpoint_rejection(
        error,
        {**screen, "rejections": [foliage]},
    ) is None


def test_committed_cut_replans_rigid_tissue_without_stationary_delay() -> None:
    rigid = {
        "nearest_obstacle": (
            "/World/InteractiveVines/Vine_0002/Physics/"
            "Organ_0029/Link_054/Collider"
        )
    }
    foliage = {
        "nearest_obstacle": (
            "/World/InteractiveVines/Vine_0002/Physics/"
            "Organ_0029/Link_054/FoliageContact_0001"
        )
    }

    assert _right_committed_cut_replan_settle_mode(rigid) == (
        "immediate_measured_replan"
    )
    assert _right_committed_cut_replan_settle_mode(foliage) == (
        "stationary_foliage_settle"
    )
    assert _right_committed_cut_replan_settle_mode(
        {"nearest_obstacle": "/World/Greenhouse/Gutter"}
    ) == "fail_closed"


def test_committed_cut_live_ik_preserves_runtime_replan_guard() -> None:
    assert np.isclose(
        _right_committed_cut_live_planning_clearance_m(),
        max(
            _RIGHT_COMMITTED_CUT_PLANNING_CLEARANCE_M,
            _RIGHT_COMMITTED_CUT_LIVE_REPLAN_GUARD_M,
        ),
    )
    assert _RIGHT_COMMITTED_CUT_LIVE_REPLAN_TRIGGER_M > (
        _RIGHT_COMMITTED_CUT_LIVE_REPLAN_GUARD_M
    )
    assert (
        _RIGHT_COMMITTED_CUT_LIVE_REPLAN_TRIGGER_M
        - _RIGHT_COMMITTED_CUT_LIVE_REPLAN_GUARD_M
        >= 0.0001
    )
    assert np.isclose(
        _right_committed_cut_recovery_floor_m(
            {"clearance_m": 0.005582}
        ),
        _RIGHT_COMMITTED_CUT_LIVE_REPLAN_GUARD_M,
    )
    first_stop = {"clearance_m": 0.005468}
    floor_m = _right_committed_cut_recovery_floor_m(first_stop)
    assert np.isclose(
        floor_m,
        0.005468 - _RIGHT_COMMITTED_CUT_RECOVERY_DECREASE_TOLERANCE_M,
    )
    assert np.isclose(
        _right_committed_cut_recovery_floor_m(
            {"clearance_m": 0.005448},
            floor_m,
        ),
        floor_m,
    )


    maximum_recovery_m = _right_committed_cut_maximum_recovery_m(
        0.0028690388582134248,
        0.001,
    )
    assert maximum_recovery_m == pytest.approx(
        0.0002690388582134247
    )
    assert (
        0.0028690388582134248
        - 0.001
        - maximum_recovery_m
    ) == pytest.approx(
        _RIGHT_CUT_MINIMUM_TARGET_INTERSECTION_M
        + _RIGHT_COMMITTED_CUT_LIVE_REPLAN_TRIGGER_M
        - _RIGHT_COMMITTED_CUT_LIVE_REPLAN_GUARD_M
    )
    with pytest.raises(ValueError):
        _right_committed_cut_maximum_recovery_m(
            0.0028690388582134248,
            -0.001,
        )


def test_committed_cut_accepts_synchronized_measured_stop_reserve() -> None:
    assert _right_committed_cut_measured_stop_reserve_acceptable(
        0.005596699889832732,
        0.005867521291302611,
        _RIGHT_COMMITTED_CUT_LIVE_REPLAN_GUARD_M,
    )
    assert not _right_committed_cut_measured_stop_reserve_acceptable(
        0.005596699889832732,
        0.00561,
        _RIGHT_COMMITTED_CUT_LIVE_REPLAN_GUARD_M,
    )
    assert _right_committed_cut_measured_stop_reserve_acceptable(
        0.00555,
        _RIGHT_COMMITTED_CUT_LIVE_REPLAN_TRIGGER_M,
        _RIGHT_COMMITTED_CUT_LIVE_REPLAN_GUARD_M,
    )
    with pytest.raises(ValueError):
        _right_committed_cut_measured_stop_reserve_acceptable(
            0.00555,
            0.0056,
            0.0049,
        )

def test_committed_cut_uses_stationary_feedback_inside_live_guard() -> None:
    assert _right_committed_cut_requires_stationary_recovery_first(0.005584)
    assert not _right_committed_cut_requires_stationary_recovery_first(0.0056)
    assert not _right_committed_cut_requires_stationary_recovery_first(0.006)
    with pytest.raises(ValueError):
        _right_committed_cut_requires_stationary_recovery_first(0.004999)



def test_committed_cut_recovery_brake_requires_safe_rebound() -> None:
    rebound = (0.005169, 0.005201, 0.005243, 0.005258)
    assert _right_committed_cut_recovery_brake_complete(
        [0.0, 0.0, 0.0, 0.0, 1.4, -1.4, 0.0],
        rebound,
    )
    assert not _right_committed_cut_recovery_brake_complete(
        [0.0, 0.0, 0.0, 0.0, 1.501, 0.0, 0.0],
        rebound,
    )
    assert not _right_committed_cut_recovery_brake_complete(
        [0.0] * 7,
        (0.00530, 0.00525, 0.00524),
    )
    assert not _right_committed_cut_recovery_brake_complete(
        [0.0] * 7,
        (0.004999, 0.00510, 0.00520),
    )
    assert not _right_committed_cut_recovery_brake_complete(
        [0.0] * 6,
        rebound,
    )
    with pytest.raises(ValueError):
        _right_committed_cut_recovery_brake_complete(
            [0.0] * 7,
            rebound,
            maximum_speed_degrees_s=0.0,
        )



def test_committed_cut_stationary_recovery_stage_has_no_child_backlink() -> None:
    stationary = {"stage": "stationary", "steps": 19, "accepted": True}
    planning = {
        "execution_mode": "near_guard_stationary_cartesian_feedback",
        "stationary_reserve_recovery": stationary,
    }

    stage = _right_committed_cut_stationary_recovery_stage("recovery", planning)

    assert stage["steps"] == 19
    assert stage["rigid_clearance_recovery"] is planning
    assert "rigid_clearance_recovery" not in stationary
    json.dumps({"stages": [stationary, stage]})

def test_committed_cut_rigid_recovery_increases_exact_capsule_clearance() -> None:
    obstacle = robot_kinematics.CapsuleObstacle(
        path="/World/InteractiveVines/Vine/Physics/Organ/Link/Collider",
        start_m=(0.0, 0.2, -1.0),
        end_m=(0.0, 0.2, 1.0),
        radius_m=0.01,
    )
    centre = np.zeros(3)
    rotation = np.eye(3)
    half_extents = np.asarray([0.1, 0.1, 0.1])
    cut_direction = np.asarray([1.0, 0.0, 0.0])
    before = robot_kinematics.oriented_box_capsule_clearance(
        centre, rotation, half_extents, (obstacle,)
    ).clearance_m

    correction = _right_committed_cut_rigid_recovery_correction(
        np.zeros(3),
        centre,
        rotation,
        half_extents,
        obstacle,
        cut_direction,
        maximum_correction_m=0.00035,
    )
    after = robot_kinematics.oriented_box_capsule_clearance(
        centre + correction,
        rotation,
        half_extents,
        (obstacle,),
    ).clearance_m

    assert np.allclose(correction, [0.0, -0.00025, 0.0])
    assert np.dot(correction, cut_direction) == pytest.approx(0.0)
    assert after > before
    capped = _right_committed_cut_rigid_recovery_correction(
        correction,
        centre + correction,
        rotation,
        half_extents,
        obstacle,
        cut_direction,
        maximum_correction_m=0.00035,
    )
    assert np.linalg.norm(capped) == pytest.approx(0.00035)
    braking = _right_committed_cut_rigid_recovery_correction(
        np.zeros(3),
        centre,
        rotation,
        half_extents,
        obstacle,
        cut_direction,
        maximum_correction_m=0.00035,
        step_m=0.00035,
    )
    assert np.linalg.norm(braking) == pytest.approx(0.00035)
    assert np.dot(braking, cut_direction) == pytest.approx(0.0)


def test_committed_cut_rigid_recovery_rejects_invalid_geometry() -> None:
    obstacle = robot_kinematics.CapsuleObstacle(
        path="/Collider",
        start_m=(0.0, 0.2, -1.0),
        end_m=(0.0, 0.2, 1.0),
        radius_m=0.01,
    )
    with pytest.raises(ValueError):
        _right_committed_cut_rigid_recovery_correction(
            np.zeros(3),
            np.zeros(3),
            np.eye(3),
            np.ones(3),
            obstacle,
            np.zeros(3),
            maximum_correction_m=0.00035,
        )
    with pytest.raises(ValueError):
        _right_committed_cut_rigid_recovery_correction(
            np.zeros(3),
            np.zeros(3),
            np.eye(3),
            np.ones(3),
            obstacle,
            np.asarray([1.0, 0.0, 0.0]),
            maximum_correction_m=0.0,
        )


def test_committed_cut_rigid_recovery_requires_measured_clearance_gain() -> None:
    assert _right_committed_cut_rigid_recovery_acceptable(
        0.00548, 0.00568, None
    )
    assert not _right_committed_cut_rigid_recovery_acceptable(
        0.00548, 0.00549, None
    )
    assert not _right_committed_cut_rigid_recovery_acceptable(
        0.00548,
        0.00568,
        {"reason": "live_payload_replan_reserve"},
    )


def test_stationary_rigid_recovery_requires_guard_gain_and_hard_floor() -> None:
    samples = (0.00549, 0.00555, 0.005621)
    assert _right_committed_cut_stationary_recovery_acceptable(
        0.005595, samples, None, True
    )
    assert not _right_committed_cut_stationary_recovery_acceptable(
        0.005595, samples[:-1], None, True
    )
    assert not _right_committed_cut_stationary_recovery_acceptable(
        0.005595, (0.004999, 0.0057), None, True
    )
    assert not _right_committed_cut_stationary_recovery_acceptable(
        0.005595, samples, None, False
    )
    assert not _right_committed_cut_stationary_recovery_acceptable(
        0.005595,
        samples,
        {"reason": "live_payload_replan_reserve"},
        True,
    )
    assert not _right_committed_cut_stationary_recovery_acceptable(
        0.005595, (), None, True
    )


def test_rigid_recovery_feedback_pose_is_bounded_and_preserves_rotation() -> None:
    desired = np.eye(4)
    desired[:3, 3] = (1.0, 2.0, 3.0)
    desired[:3, :3] = np.asarray(
        ((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    )
    measured = desired.copy()
    measured[:3, 3] -= (0.0001, 0.0, 0.0)
    feedback = _right_committed_cut_recovery_feedback_pose(
        desired,
        measured,
    )
    assert np.allclose(feedback[:3, :3], desired[:3, :3])
    assert np.allclose(feedback[:3, 3], desired[:3, 3] + (0.0004, 0.0, 0.0))

    measured[:3, 3] = desired[:3, 3] - (0.001, 0.0, 0.0)
    bounded = _right_committed_cut_recovery_feedback_pose(
        desired,
        measured,
        maximum_feedback_m=0.0006,
    )
    assert np.allclose(bounded[:3, 3], desired[:3, 3] + (0.0006, 0.0, 0.0))
    assert np.allclose(
        _right_committed_cut_recovery_feedback_pose(desired, desired),
        desired,
    )
    with pytest.raises(ValueError):
        _right_committed_cut_recovery_feedback_pose(
            np.eye(3),
            desired,
        )
    with pytest.raises(ValueError):
        _right_committed_cut_recovery_feedback_pose(
            desired,
            measured,
            maximum_feedback_m=0.0,
        )
    with pytest.raises(ValueError):
        _right_committed_cut_recovery_feedback_pose(
            desired,
            measured,
            feedback_gain=0.0,
        )


def test_rigid_recovery_feedback_preserves_target_intersection_reserve() -> None:
    pose = np.eye(4)
    pose[:3, 3] = (1.0014, 2.0, 3.0)
    bounded, clamped = _right_committed_cut_intersection_bounded_feedback_pose(
        pose,
        (1.0014, 2.0, 3.0),
        0.00175,
        (0.00025, 0.0, 0.0),
        (1.0, 2.0, 3.0),
    )
    assert clamped
    assert bounded[:3, 3] == pytest.approx((1.00015, 2.0, 3.0))
    assert _right_committed_cut_physical_target_intersection_m(
        0.00175,
        (0.00025, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        bounded[:3, 3],
    ) == pytest.approx(0.0016)

    unchanged, clamped = _right_committed_cut_intersection_bounded_feedback_pose(
        pose,
        (1.0001, 2.0, 3.0),
        0.00175,
        (0.00025, 0.0, 0.0),
        (1.0, 2.0, 3.0),
    )
    assert not clamped
    assert np.allclose(unchanged, pose)


def test_physical_target_intersection_uses_accumulated_correction_norm() -> None:
    assert _right_committed_cut_physical_target_intersection_m(
        0.00175,
        (0.00025, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        (1.0002, 2.0, 3.0),
    ) == pytest.approx(0.00155)
    assert _right_committed_cut_physical_target_intersection_m(
        0.00175,
        (0.00025, 0.0, 0.0),
        (1.0, 2.0, 3.0),
        (0.9999, 2.0, 3.0),
    ) == pytest.approx(0.00185)
    with pytest.raises(ValueError):
        _right_committed_cut_physical_target_intersection_m(
            0.0,
            (0.00025, 0.0, 0.0),
            (1.0, 2.0, 3.0),
            (1.0, 2.0, 3.0),
        )


def test_rigid_recovery_drive_gains_preserve_hardware_effort_limits() -> None:
    assert np.allclose(
        _RIGHT_COMMITTED_CUT_STIFFNESS_NM_RAD,
        32.0 * np.asarray(_RBY1_ARM_STIFFNESS_NM_RAD),
    )
    assert np.allclose(
        _RIGHT_COMMITTED_CUT_DAMPING_NM_S_RAD,
        np.sqrt(32.0) * np.asarray(_RBY1_ARM_DAMPING_NM_S_RAD),
    )
    assert np.allclose(
        _RIGHT_RIGID_RECOVERY_STIFFNESS_NM_RAD,
        2.0 * np.asarray(_RBY1_ARM_STIFFNESS_NM_RAD),
    )
    assert np.allclose(
        _RIGHT_RIGID_RECOVERY_DAMPING_NM_S_RAD,
        2.0 * np.asarray(_RBY1_ARM_DAMPING_NM_S_RAD),
    )
    assert _RBY1_ARM_EFFORT_LIMITS_NM == (
        70.0,
        70.0,
        70.0,
        40.0,
        10.0,
        10.0,
        8.0,
    )
    assert np.allclose(
        _RIGHT_RIGID_RECOVERY_TORSO_STIFFNESS_NM_RAD,
        2.0 * np.asarray(_RBY1_TORSO_STIFFNESS_NM_RAD),
    )
    assert np.allclose(
        _RIGHT_RIGID_RECOVERY_TORSO_DAMPING_NM_S_RAD,
        2.0 * np.asarray(_RBY1_TORSO_DAMPING_NM_S_RAD),
    )
    assert _RBY1_TORSO_EFFORT_LIMITS_NM == (
        270.0,
        270.0,
        270.0,
        120.0,
        120.0,
        120.0,
    )


def test_post_recovery_tracking_persists_for_latched_torso_frame() -> None:
    nominal = np.asarray((0.0, 45.0, -90.0, 45.0, 0.0, 0.0))
    assert not _post_recovery_tracking_required(False, nominal, nominal)
    assert _post_recovery_tracking_required(True, nominal, nominal)
    assert _post_recovery_tracking_required(
        False,
        nominal + np.asarray((0.0, 0.0, 0.02, 0.0, 0.0, 0.0)),
        nominal,
    )
    with pytest.raises(ValueError, match="finite six-vectors"):
        _post_recovery_tracking_required(False, nominal[:5], nominal)


def test_committed_cut_recovery_torso_target_is_a_one_shot_latch() -> None:
    nominal = np.asarray((0.0, 45.0, -90.0, 45.0, 0.0, 0.0))
    first_measured = nominal + np.asarray(
        (0.02, 0.03, -0.06, 0.09, 0.04, -0.01)
    )
    later_measured = nominal + np.asarray(
        (0.05, 0.03, -0.11, 0.15, 0.07, -0.02)
    )

    first_target = _right_committed_cut_recovery_torso_target(
        nominal, first_measured, False
    )
    reused_target = _right_committed_cut_recovery_torso_target(
        first_target, later_measured, True
    )

    np.testing.assert_allclose(first_target, first_measured)
    np.testing.assert_allclose(reused_target, first_measured)
    reused_target[0] += 1.0
    np.testing.assert_allclose(first_target, first_measured)
    with pytest.raises(ValueError, match="finite six-vectors"):
        _right_committed_cut_recovery_torso_target(
            nominal[:5], first_measured, False
        )
    with pytest.raises(ValueError, match="finite six-vectors"):
        _right_committed_cut_recovery_torso_target(
            nominal, np.full(6, np.nan), True
        )


def test_committed_cut_stationary_brake_holds_measured_torso_entry() -> None:
    measured = np.asarray((0.15, 45.02, -90.11, 45.14, 0.20, -0.11))

    target = _right_committed_cut_stationary_brake_torso_target(measured)

    np.testing.assert_allclose(target, measured)
    target[0] += 1.0
    np.testing.assert_allclose(
        measured,
        np.asarray((0.15, 45.02, -90.11, 45.14, 0.20, -0.11)),
    )
    with pytest.raises(ValueError, match="finite six-vector"):
        _right_committed_cut_stationary_brake_torso_target(measured[:5])
    with pytest.raises(ValueError, match="finite six-vector"):
        _right_committed_cut_stationary_brake_torso_target(np.full(6, np.nan))


def test_committed_cut_rigid_recovery_path_reaches_guard_monotonically() -> None:
    assert _right_committed_cut_rigid_recovery_path_acceptable(
        0.00548,
        (0.00547, 0.00550, 0.00562),
        0.005455,
    )
    assert not _right_committed_cut_rigid_recovery_path_acceptable(
        0.00548,
        (0.00544, 0.00552),
        0.005455,
    )
    assert not _right_committed_cut_rigid_recovery_path_acceptable(
        0.00548,
        (0.00547, 0.00549),
        0.005455,
    )


def test_stationary_foliage_settle_accepts_only_shallow_soft_overlap() -> None:
    payload = {
        'clearance_m': -0.000282,
        'nearest_obstacle': (
            '/World/InteractiveVines/Vine/Physics/Organ/'
            'Link/FoliageContact_01'
        ),
    }

    assert _stationary_foliage_settle_eligible(payload)
    assert not _stationary_foliage_settle_eligible(
        {**payload, 'clearance_m': -0.000501}
    )
    assert not _stationary_foliage_settle_eligible(
        {
            **payload,
            'nearest_obstacle': (
                '/World/InteractiveVines/Vine/Physics/Organ/'
                'Link/Collider'
            ),
        }
    )
    assert not _stationary_foliage_settle_eligible(
        {**payload, 'clearance_m': 0.0001}
    )


def test_live_base_retry_accepts_only_positive_gap_dynamic_vine_deficit() -> None:
    diagnostics = {
        'minimum_payload_clearance_m': 0.005,
        'minimum_foliage_clearance_m': 0.0005,
        'attempts': [
            {
                'solution': {'succeeded': True},
                'additional_feasibility': {'feasible': True},
                'trajectory_payload_clearance_m': 0.0038,
                'nearest_trajectory_payload_obstacle': (
                    '/World/InteractiveVines/Vine_0002/Physics/Organ_0029/'
                    'Link_034/Collider'
                ),
            }
        ],
    }

    assert _retryable_live_base_plan_rejection(diagnostics)

    diagnostics['attempts'][0]['trajectory_payload_clearance_m'] = -0.001
    assert not _retryable_live_base_plan_rejection(diagnostics)

    diagnostics['attempts'][0]['trajectory_payload_clearance_m'] = 0.0038
    diagnostics['attempts'][0]['nearest_trajectory_payload_obstacle'] = (
        '/World/Greenhouse/Gutter/Collider'
    )
    assert not _retryable_live_base_plan_rejection(diagnostics)


def test_left_measured_handoff_is_only_for_small_clearance_route_residual() -> None:
    assert _left_clearance_route_measured_handoff_allowed(
        "left_planned_route_0",
        0.0041,
        np.radians(0.3),
    )
    assert not _left_clearance_route_measured_handoff_allowed(
        "left_preplanned_final_approach",
        0.0041,
        np.radians(0.3),
    )
    assert not _left_clearance_route_measured_handoff_allowed(
        "left_planned_route_0",
        0.0051,
        np.radians(0.3),
    )
    assert not _left_clearance_route_measured_handoff_allowed(
        "left_planned_route_0",
        0.0041,
        np.radians(1.01),
    )


def test_guarded_handoff_accepts_a_persistent_target_inside_open_jaws() -> None:
    result = _jaw_corridor_handoff_acceptable(
        np.eye(4),
        _LEFT_JAW_CENTRE_M,
        0.003,
        _LEFT_GRIPPER_PLANNING_GEOMETRY,
    )

    assert result['acceptable']
    assert result['signed_clearance_m'] > 0.0
    consecutive = 0
    for _ in range(4):
        consecutive, accepted, _mode = _guarded_jaw_handoff_step(
            0.006,
            consecutive,
            corridor_acceptable=result['acceptable'],
        )
    assert consecutive == 4
    assert accepted


def test_jaw_corridor_rejects_target_outside_finger_depth() -> None:
    target = np.asarray(_LEFT_JAW_CENTRE_M, dtype=np.float64)
    target[1] = 0.014

    result = _jaw_corridor_handoff_acceptable(
        np.eye(4),
        target,
        0.003,
        _LEFT_GRIPPER_PLANNING_GEOMETRY,
    )

    assert not result['acceptable']
    assert result['signed_clearance_m'] < 0.0


def test_subreserve_recovery_only_accepts_positive_monotonic_escape() -> None:
    assert _subreserve_clearance_recovery_acceptable(
        0.000384,
        0.000383,
        0.000430,
        0.0005,
    )
    assert not _subreserve_clearance_recovery_acceptable(
        0.000384,
        0.000350,
        0.000430,
        0.0005,
    )
    assert not _subreserve_clearance_recovery_acceptable(
        0.000384,
        0.000383,
        0.000390,
        0.0005,
    )
    assert not _subreserve_clearance_recovery_acceptable(
        0.0005,
        0.0005,
        0.0006,
        0.0005,
    )
    assert not _subreserve_clearance_recovery_acceptable(
        0.00005,
        0.00005,
        0.0002,
        0.0005,
    )


def test_runtime_subreserve_floor_only_applies_to_approved_foliage_escape() -> None:
    assert _subreserve_runtime_clearance_acceptable(
        0.000379,
        '/World/Vine/FoliageContact_01',
        0.000379,
    )
    assert not _subreserve_runtime_clearance_acceptable(
        0.000378,
        '/World/Vine/FoliageContact_01',
        0.000379,
    )
    assert not _subreserve_runtime_clearance_acceptable(
        0.0004,
        '/World/Vine/Collider',
        0.000379,
    )
    assert not _subreserve_runtime_clearance_acceptable(
        0.0004,
        '/World/Vine/FoliageContact_01',
        None,
    )


def test_planned_foliage_escape_requires_monotonic_endpoint_improvement() -> None:
    start = {
        'clearance_m': 0.00051,
        'nearest_obstacle': '/World/Vine/FoliageContact_01',
    }
    trajectory = {'clearance_m': 0.000505}
    assert _planned_foliage_escape_floor(
        start,
        trajectory,
        {'clearance_m': 0.00055},
        0.0005,
    ) == 0.0001
    assert _planned_foliage_escape_floor(
        start,
        trajectory,
        {'clearance_m': 0.00052},
        0.0005,
    ) is None
    assert _planned_foliage_escape_floor(
        start,
        {'clearance_m': 0.00049},
        {'clearance_m': 0.00055},
        0.0005,
    ) is None



def test_selected_right_goal_allows_only_guarded_foliage_escape() -> None:
    wing = [-0.014, -0.001, 0.0]
    candidate = {
        "roll_degrees": -45.0,
        "edge_wing_local_m": wing,
        "solution": {"succeeded": True},
        "counterhold_clear": True,
        "minimum_inter_arm_clearance_m": 0.13,
        "payload_trajectory_clearance": {"clearance_m": 0.0020},
        "required_payload_clearance_m": 0.0025,
        "endpoint_state_payload_clearance": {"clearance_m": 0.013},
    }
    goal = {
        "selected_roll_degrees": -45.0,
        "selected_edge_wing_local_m": wing,
        "direction_candidates": [candidate],
    }
    start = {
        "clearance_m": 0.0020,
        "nearest_obstacle": "/World/Vine/FoliageContact_01",
    }

    assert np.isclose(
        _right_goal_foliage_escape_floor(goal, start), 0.0005
    )

    candidate["minimum_inter_arm_clearance_m"] = 0.004
    assert _right_goal_foliage_escape_floor(goal, start) is None

    candidate["minimum_inter_arm_clearance_m"] = 0.13
    start["nearest_obstacle"] = "/World/Greenhouse/Gutter"
    assert _right_goal_foliage_escape_floor(goal, start) is None


def test_selected_right_goal_allows_only_hard_floor_rigid_vine_escape() -> None:
    wing = [-0.014, -0.018, 0.0]
    candidate = {
        "roll_degrees": -75.0,
        "edge_wing_local_m": wing,
        "solution": {"succeeded": True},
        "counterhold_clear": True,
        "minimum_inter_arm_clearance_m": 0.095,
        "payload_trajectory_clearance": {"clearance_m": 0.005085},
        "required_payload_clearance_m": 0.0051,
        "endpoint_state_payload_clearance": {"clearance_m": 0.00846},
    }
    goal = {
        "selected_roll_degrees": -75.0,
        "selected_edge_wing_local_m": wing,
        "direction_candidates": [candidate],
    }
    start = {
        "clearance_m": 0.005085,
        "nearest_obstacle": (
            "/World/InteractiveVines/Vine_0002/Physics/"
            "Organ_0029/Link_054/Collider"
        ),
    }

    assert _right_goal_rigid_vine_planning_escape(goal, start)

    candidate["payload_trajectory_clearance"]["clearance_m"] = 0.00499
    assert not _right_goal_rigid_vine_planning_escape(goal, start)
    candidate["payload_trajectory_clearance"]["clearance_m"] = 0.005085
    start["clearance_m"] = 0.00499
    assert not _right_goal_rigid_vine_planning_escape(goal, start)


def test_right_blade_candidates_stay_on_the_flat_leading_edge() -> None:
    points = _right_leading_edge_point_candidates(
        [0.0, -0.035, 0.0],
        [0.015, 0.035, 0.0065],
    )

    assert len(points) == 5
    assert all(np.isclose(point[0], -0.014) for point in points)
    assert all(np.isclose(point[2], 0.0) for point in points)
    assert np.allclose(
        [point[1] for point in points],
        np.linspace(-0.001, -0.069, 5),
    )
    assert all(point[0] < 0.0 for point in points)


def test_right_outer_endpoint_screen_records_first_reachable_knife_pose() -> None:
    class FakeModel:
        def __init__(self):
            self.calls = []

        def solve_pose(self, side, desired, seed, base_matrix):
            self.calls.append((side, desired.copy(), tuple(seed), base_matrix.copy()))
            return robot_kinematics.IKResult(
                joint_degrees=(1.0,) * 7,
                position_error_m=0.0,
                orientation_error_rad=0.0,
                cost=0.0,
                succeeded=True,
            )

    model = FakeModel()
    result = _right_outer_endpoint_ik_screen(
        model,
        base_matrix=np.eye(4),
        cut_point_m=[1.0, 0.0, 1.0],
        cut_axis=[0.0, 0.0, 1.0],
        preferred_cut_direction=[1.0, 0.0, 0.0],
        seeds=((0.0,) * 7,),
        roll_candidates=(0.0,),
    )

    assert result["feasible"]
    assert result["attempt_count"] == 5
    assert len(result["attempts"]) == 5
    assert result["selected"]["roll_degrees"] == 0.0
    assert model.calls[0][0] == "right"


def test_right_endpoint_screen_selects_outermost_reachable_side() -> None:
    class SideLimitedModel:
        def solve_pose(self, _side, desired, _seed, _base_matrix):
            succeeded = bool(float(desired[0, 3]) > 0.84)
            return robot_kinematics.IKResult(
                joint_degrees=(2.0,) * 7,
                position_error_m=0.0 if succeeded else 0.02,
                orientation_error_rad=0.0,
                cost=0.0,
                succeeded=succeeded,
            )

    result = _right_reachable_approach_endpoint_ik_screen(
        SideLimitedModel(),
        base_matrix=np.eye(4),
        cut_point_m=[1.0, 0.0, 1.0],
        cut_axis=[0.0, 0.0, 1.0],
        preferred_cut_direction=[1.0, 0.0, 0.0],
        seeds=((0.0,) * 7,),
        side_candidates_m=(-0.160, -0.140, -0.120),
        roll_candidates=(0.0,),
    )

    assert result["feasible"]
    assert np.isclose(result["side_m"], -0.140)
    assert [screen["feasible"] for screen in result["side_screens"]] == [
        False,
        True,
    ]


def test_right_outer_endpoint_screen_can_hold_a_selected_blade_wing() -> None:
    class FakeModel:
        def solve_pose(self, _side, _desired, _seed, _base_matrix):
            return robot_kinematics.IKResult(
                joint_degrees=(1.0,) * 7,
                position_error_m=0.0,
                orientation_error_rad=0.0,
                cost=0.0,
                succeeded=True,
            )

    selected_wing = (-0.014, -0.035, 0.0)
    result = _right_outer_endpoint_ik_screen(
        FakeModel(),
        base_matrix=np.eye(4),
        cut_point_m=[1.0, 0.0, 1.0],
        cut_axis=[0.0, 0.0, 1.0],
        preferred_cut_direction=[1.0, 0.0, 0.0],
        seeds=((0.0,) * 7,),
        wing_candidates=(selected_wing,),
        roll_candidates=(0.0,),
    )

    assert result["feasible"]
    assert result["attempt_count"] == 1
    assert np.allclose(
        result["selected"]["edge_wing_local_m"], selected_wing
    )


def test_right_outer_endpoint_screen_can_finish_all_roll_wing_batches() -> None:
    class FakeModel:
        def solve_pose(self, _side, _desired, _seed, _base_matrix):
            return robot_kinematics.IKResult(
                joint_degrees=(1.0,) * 7,
                position_error_m=0.0,
                orientation_error_rad=0.0,
                cost=0.0,
                succeeded=True,
            )

    kwargs = {
        "base_matrix": np.eye(4),
        "cut_point_m": [1.0, 0.0, 1.0],
        "cut_axis": [0.0, 0.0, 1.0],
        "preferred_cut_direction": [1.0, 0.0, 0.0],
        "seeds": ((0.0,) * 7,),
        "roll_candidates": (0.0, 5.0, -5.0),
    }
    early = _right_outer_endpoint_ik_screen(FakeModel(), **kwargs)
    expanded = _right_outer_endpoint_ik_screen(
        FakeModel(),
        stop_at_first_success_batch=False,
        **kwargs,
    )

    assert early["attempt_count"] == 12
    assert expanded["attempt_count"] == 15
    assert expanded["feasible"]


def test_reachable_endpoint_screen_can_finish_all_bounded_roll_batches() -> None:
    class FakeModel:
        def solve_pose(self, _side, _desired, _seed, _base_matrix):
            return robot_kinematics.IKResult(
                joint_degrees=(1.0,) * 7,
                position_error_m=0.0,
                orientation_error_rad=0.0,
                cost=0.0,
                succeeded=True,
            )

    result = _right_reachable_approach_endpoint_ik_screen(
        FakeModel(),
        base_matrix=np.eye(4),
        cut_point_m=[1.0, 0.0, 1.0],
        cut_axis=[0.0, 0.0, 1.0],
        preferred_cut_direction=[1.0, 0.0, 0.0],
        seeds=((0.0,) * 7,),
        side_candidates_m=(-0.160,),
        roll_candidates=(0.0, 5.0, -5.0),
        stop_at_first_success_batch=False,
    )

    assert result["feasible"]
    assert result["attempt_count"] == 15


def test_right_local_cut_continuation_audits_each_adjacent_chord() -> None:
    class ContinuationModel:
        def __init__(self, block_after=None):
            self.solve_calls = 0
            self.block_after = block_after

        def solve_pose(self, _side, _desired, _seed, _base_matrix):
            self.solve_calls += 1
            return robot_kinematics.IKResult(
                joint_degrees=(
                    float(self.solve_calls),
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ),
                position_error_m=0.0,
                orientation_error_rad=0.0,
                cost=0.0,
                succeeded=True,
            )

        def arm_obstacle_clearance(self, _side, degrees, *_args):
            blocked = (
                self.block_after is not None
                and float(np.asarray(degrees)[0]) >= self.block_after
            )
            return robot_kinematics.ClearanceResult(
                -0.010 if blocked else 0.020,
                "blocked_vine" if blocked else "clear_vine",
            )

        def inter_arm_clearance(self, *_args):
            return robot_kinematics.ClearanceResult(0.080, "inter_arm")

        def arm_joint_limit_margin_degrees(self, *_args):
            return 10.0

    start_attempt = {
        "roll_degrees": 0.0,
        "edge_wing_local_m": [-0.014, -0.035, 0.0],
        "solution": {
            "joint_degrees": [0.0] * 7,
            "position_error_m": 0.0,
            "orientation_error_rad": 0.0,
            "cost": 0.0,
            "succeeded": True,
        },
    }
    kwargs = {
        "base_matrix": np.eye(4),
        "cut_point_m": [1.0, 0.0, 1.0],
        "cut_axis": [0.0, 0.0, 1.0],
        "preferred_cut_direction": [1.0, 0.0, 0.0],
        "start_side_m": -0.060,
        "start_attempt": start_attempt,
        "left_degrees": np.zeros(7),
    }

    clear = _right_local_cut_continuation_screen(
        ContinuationModel(), **kwargs
    )
    blocked = _right_local_cut_continuation_screen(
        ContinuationModel(block_after=1.5), **kwargs
    )

    assert clear["feasible"]
    assert np.isclose(clear["final_side_m"], -0.035)
    assert [record["side_m"] for record in clear["waypoints"]] == [
        -0.050,
        -0.042,
        -0.035,
    ]
    assert not blocked["feasible"]
    assert np.isclose(blocked["final_side_m"], -0.042)
    assert blocked["reason"] == (
        "local continuation chord lost collision reserve"
    )

    full = _right_local_cut_continuation_screen(
        ContinuationModel(),
        stop_side_m=0.015,
        minimum_payload_clearance_m=0.008,
        committed_minimum_payload_clearance_m=0.0051,
        **kwargs,
    )

    assert full["feasible"]
    assert np.isclose(full["final_side_m"], 0.015)
    assert len(full["waypoints"]) == 43
    assert np.isclose(full["waypoints"][-1]["side_m"], 0.015)
    assert [record["phase"] for record in full["waypoints"][:3]] == [
        "approach",
        "approach",
        "approach",
    ]
    assert all(
        record["phase"] == "committed_cut"
        for record in full["waypoints"][3:]
    )
    assert [
        record["minimum_payload_clearance_m"]
        for record in full["waypoints"][:3]
    ] == pytest.approx([0.008] * 3)
    assert [
        record["minimum_payload_clearance_m"]
        for record in full["waypoints"][3:]
    ] == pytest.approx([0.0051] * 40)


def test_right_endpoint_pruning_requires_left_independent_failure() -> None:
    clear_right = {
        "endpoint_feasible": False,
        "arm_clearance_m": 0.006,
        "payload_clearance_m": 0.009,
        "arm_foliage_clearance_m": 0.002,
        "payload_foliage_clearance_m": 0.002,
        "joint_limit_margin_degrees": 1.0,
        "minimum_arm_clearance_m": 0.0055,
        "minimum_payload_clearance_m": 0.008,
        "minimum_foliage_clearance_m": 0.001,
        "minimum_joint_limit_margin_degrees": 0.0,
    }

    def screen(clearance):
        return {
            "endpoint_clearance_candidates": [
                {
                    "clearance": clearance,
                    "continuation": {
                        "feasible": False,
                        "reason": "outer endpoint lost collision reserve",
                    },
                }
            ]
        }

    rigid_blocked = {**clear_right, "arm_clearance_m": 0.005}
    assert _right_endpoint_failure_is_left_independent(
        screen(rigid_blocked)
    )
    assert not _right_endpoint_failure_is_left_independent(
        screen(clear_right)
    )

    right_chord_blocked = {
        "feasible": False,
        "reason": "local continuation chord lost collision reserve",
        "waypoints": [
            {
                "clearance": {
                    "route_minimum_payload": {"clearance_m": 0.003824},
                    "minimum_payload_clearance_m": 0.0051,
                }
            }
        ],
    }
    assert _right_continuation_failure_is_left_independent(
        right_chord_blocked
    )
    left_only_blocked = {
        "feasible": False,
        "reason": "local continuation chord lost collision reserve",
        "waypoints": [
            {
                "clearance": {
                    "route_minimum_inter_arm": {"clearance_m": 0.040},
                    "minimum_inter_arm_clearance_m": 0.050,
                }
            }
        ],
    }
    assert not _right_continuation_failure_is_left_independent(
        left_only_blocked
    )


def test_expanded_endpoint_search_is_deferred_for_online_multi_target() -> None:
    assert not _right_expanded_endpoint_search_enabled(
        exhaustive_base_planning=False,
        multi_target_planning=True,
    )
    assert not _right_expanded_endpoint_search_enabled(
        exhaustive_base_planning=False,
        multi_target_planning=False,
    )
    for exhaustive, multi_target in (
        (True, True),
        (True, False),
    ):
        assert _right_expanded_endpoint_search_enabled(
            exhaustive_base_planning=exhaustive,
            multi_target_planning=multi_target,
        )


def test_online_base_planning_retains_bounded_joint_space_fallback() -> None:
    assert _base_planning_joint_space_route_budget(False) == (2500, 3)
    assert _base_planning_joint_space_route_budget(True) == (2500, None)


def test_expanded_endpoint_rolls_honor_explicit_calibration_subset() -> None:
    assert _right_expanded_endpoint_roll_candidates(None) == (
        _RIGHT_BASE_ENDPOINT_ROLLS_DEGREES
    )
    assert _right_expanded_endpoint_roll_candidates([74.5]) == (74.5,)
    with pytest.raises(ValueError):
        _right_expanded_endpoint_roll_candidates([])
    with pytest.raises(ValueError):
        _right_expanded_endpoint_roll_candidates([float("nan")])


def test_pose_nullspace_seeds_move_only_redundant_joint() -> None:
    class RedundantModel:
        def arm_limits_degrees(self, _side):
            return np.full(7, -180.0), np.full(7, 180.0)

        def forward(self, _side, degrees, base):
            from scipy.spatial.transform import Rotation

            joints = np.asarray(degrees, dtype=np.float64)
            pose = np.eye(4)
            pose[:3, 3] = np.radians(joints[:3])
            pose[:3, :3] = Rotation.from_rotvec(
                np.radians(joints[3:6])
            ).as_matrix()
            return np.asarray(base, dtype=np.float64) @ pose

    seeds = _pose_nullspace_seed_candidates(
        RedundantModel(),
        "right",
        np.zeros(7),
        np.eye(4),
        steps_degrees=(1.0, 2.0),
    )

    assert len(seeds) == 4
    assert np.allclose(
        np.asarray(seeds)[:, :6],
        np.zeros((4, 6)),
        atol=1e-7,
    )
    assert np.allclose(
        np.asarray(seeds)[:, 6],
        (1.0, -1.0, 2.0, -2.0),
        atol=1e-7,
    )
    with pytest.raises(ValueError):
        _pose_nullspace_seed_candidates(
            RedundantModel(),
            "right",
            np.zeros(7),
            np.eye(4),
            steps_degrees=(),
        )


def test_requested_base_local_lattice_preserves_exact_pose_and_bounds() -> None:
    advances = _local_requested_base_depth_advances(0.020, 0.040)
    assert advances == pytest.approx(
        (0.020, 0.025, 0.015, 0.030, 0.010, 0.040, 0.0)
    )
    assert _LOCAL_REQUESTED_BASE_LATERAL_OFFSETS_M == pytest.approx(
        (0.0, -0.005, 0.005, -0.010, 0.010, -0.020, 0.020)
    )
    clipped = _local_requested_base_depth_advances(0.005, 0.012)
    assert clipped == pytest.approx((0.005, 0.010, 0.0, 0.012))
    with pytest.raises(ValueError, match="inside the aisle"):
        _local_requested_base_depth_advances(0.021, 0.020)


def test_right_local_cut_continuation_selects_clear_multistart_branch() -> None:
    class BranchModel:
        def solve_pose(self, _side, _desired, seed, _base_matrix):
            branch = -1.0 if float(seed[0]) < -50.0 else 1.0
            return robot_kinematics.IKResult(
                joint_degrees=(branch, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                position_error_m=0.0,
                orientation_error_rad=0.0,
                cost=0.0,
                succeeded=True,
            )

        def arm_obstacle_clearance(self, _side, degrees, *_args):
            blocked = float(np.asarray(degrees)[0]) > 0.0
            return robot_kinematics.ClearanceResult(
                -0.010 if blocked else 0.020,
                "blocked_local_branch" if blocked else "clear_alt_branch",
            )

        def inter_arm_clearance(self, *_args):
            return robot_kinematics.ClearanceResult(0.080, "inter_arm")

        def arm_joint_limit_margin_degrees(self, *_args):
            return 10.0

    start_attempt = {
        "roll_degrees": 0.0,
        "edge_wing_local_m": [-0.014, -0.035, 0.0],
        "solution": {
            "joint_degrees": [0.0] * 7,
            "position_error_m": 0.0,
            "orientation_error_rad": 0.0,
            "cost": 0.0,
            "succeeded": True,
        },
    }
    result = _right_local_cut_continuation_screen(
        BranchModel(),
        base_matrix=np.eye(4),
        cut_point_m=[1.0, 0.0, 1.0],
        cut_axis=[0.0, 0.0, 1.0],
        preferred_cut_direction=[1.0, 0.0, 0.0],
        start_side_m=-0.060,
        start_attempt=start_attempt,
        left_degrees=np.zeros(7),
        seeds=((-100.0,) + (0.0,) * 6,),
    )

    assert result["feasible"]
    assert all(
        waypoint["clearance_attempt_count"] == 2
        and waypoint["feasible_clearance_attempt_count"] == 1
        and np.isclose(
            waypoint["endpoint"]["selected"]["solution"][
                "joint_degrees"
            ][0],
            -1.0,
        )
        for waypoint in result["waypoints"]
    )


def test_right_local_cut_continuation_skips_fallback_when_warm_start_is_clear() -> None:
    class WarmStartModel:
        def __init__(self):
            self.solve_seeds = []

        def solve_pose(self, _side, _desired, seed, _base_matrix):
            self.solve_seeds.append(tuple(float(value) for value in seed))
            return robot_kinematics.IKResult(
                joint_degrees=tuple(float(value) for value in seed),
                position_error_m=0.0,
                orientation_error_rad=0.0,
                cost=0.0,
                succeeded=True,
            )

        def arm_obstacle_clearance(self, *_args):
            return robot_kinematics.ClearanceResult(0.020, "clear_vine")

        def inter_arm_clearance(self, *_args):
            return robot_kinematics.ClearanceResult(0.080, "inter_arm")

        def arm_joint_limit_margin_degrees(self, *_args):
            return 10.0

    model = WarmStartModel()
    result = _right_local_cut_continuation_screen(
        model,
        base_matrix=np.eye(4),
        cut_point_m=[1.0, 0.0, 1.0],
        cut_axis=[0.0, 0.0, 1.0],
        preferred_cut_direction=[1.0, 0.0, 0.0],
        start_side_m=-0.060,
        start_attempt={
            "roll_degrees": 0.0,
            "edge_wing_local_m": [-0.014, -0.035, 0.0],
            "solution": {
                "joint_degrees": [0.0] * 7,
                "position_error_m": 0.0,
                "orientation_error_rad": 0.0,
                "cost": 0.0,
                "succeeded": True,
            },
        },
        left_degrees=np.zeros(7),
        seeds=((-100.0,) + (0.0,) * 6,),
    )

    assert result["feasible"]
    assert len(model.solve_seeds) == len(result["waypoints"])
    assert all(seed[0] == 0.0 for seed in model.solve_seeds)
    assert all(
        waypoint["clearance_attempt_count"] == 1
        for waypoint in result["waypoints"]
    )


def test_right_endpoint_clearance_supports_right_only_audit() -> None:
    class RightOnlyModel:
        def arm_obstacle_clearance(self, *_args):
            return robot_kinematics.ClearanceResult(0.020, "clear_vine")

        def inter_arm_clearance(self, *_args):
            raise AssertionError("right-only audit must omit inter-arm geometry")

    result = _right_endpoint_clearance_screen(
        RightOnlyModel(),
        right_degrees=np.zeros(7),
        left_degrees=None,
        base_matrix=np.eye(4),
    )

    assert result["endpoint_feasible"]
    assert np.isinf(result["inter_arm_clearance_m"])
    assert result["inter_arm_nearest"] is None
def test_right_endpoint_clearance_screen_rejects_arm_foliage_overlap() -> None:
    class ClearanceModel:
        def __init__(self, foliage_clearance_m):
            self.foliage_clearance_m = foliage_clearance_m

        def arm_obstacle_clearance(self, *_args):
            return robot_kinematics.ClearanceResult(0.020, "arm_vine")

        def arm_oriented_box_clearance(self, *_args):
            return robot_kinematics.ClearanceResult(
                self.foliage_clearance_m, "arm_foliage"
            )

        def inter_arm_clearance(self, *_args):
            return robot_kinematics.ClearanceResult(0.080, "inter_arm")

    blocked = _right_endpoint_clearance_screen(
        ClearanceModel(-0.010),
        right_degrees=np.zeros(7),
        left_degrees=np.zeros(7),
        base_matrix=np.eye(4),
        foliage_obstacles=(object(),),
    )
    clear = _right_endpoint_clearance_screen(
        ClearanceModel(0.010),
        right_degrees=np.zeros(7),
        left_degrees=np.zeros(7),
        base_matrix=np.eye(4),
        foliage_obstacles=(object(),),
    )

    assert not blocked["feasible"]
    assert blocked["arm_foliage_nearest"] == "arm_foliage"
    assert clear["feasible"]


def test_right_endpoint_clearance_screen_rejects_low_joint_reserve() -> None:
    class LowReserveModel:
        def arm_obstacle_clearance(self, *_args):
            return robot_kinematics.ClearanceResult(0.020, "arm_vine")

        def inter_arm_clearance(self, *_args):
            return robot_kinematics.ClearanceResult(0.080, "inter_arm")

        def arm_joint_limit_margin_degrees(self, *_args):
            return 0.25

    result = _right_endpoint_clearance_screen(
        LowReserveModel(),
        right_degrees=np.zeros(7),
        left_degrees=np.zeros(7),
        base_matrix=np.eye(4),
    )

    assert not result["endpoint_feasible"]
    assert np.isclose(result["joint_limit_margin_degrees"], 0.25)
    assert np.isclose(result["minimum_joint_limit_margin_degrees"], 0.5)


def test_right_endpoint_clearance_screen_rejects_blocked_direct_route() -> None:
    class MidpointBlockedModel:
        def arm_obstacle_clearance(self, _side, degrees, *_args):
            q0 = float(np.asarray(degrees)[0])
            clearance_m = -0.010 if np.isclose(q0, 0.5) else 0.020
            return robot_kinematics.ClearanceResult(
                clearance_m, "midpoint_obstacle"
            )

        def inter_arm_clearance(self, *_args):
            return robot_kinematics.ClearanceResult(0.080, "inter_arm")

    endpoint_only = _right_endpoint_clearance_screen(
        MidpointBlockedModel(),
        right_degrees=np.ones(7),
        left_degrees=np.zeros(7),
        base_matrix=np.eye(4),
    )
    routed = _right_endpoint_clearance_screen(
        MidpointBlockedModel(),
        right_degrees=np.ones(7),
        left_degrees=np.zeros(7),
        base_matrix=np.eye(4),
        route_start_right_degrees=np.zeros(7),
        route_samples=3,
    )

    assert endpoint_only["feasible"]
    assert routed["endpoint_feasible"]
    assert not routed["route_feasible"]
    assert not routed["feasible"]
    assert np.isclose(
        routed["route_minimum_arm"]["clearance_m"], -0.010
    )
    assert np.isclose(routed["route_first_rejection"]["fraction"], 0.5)
    assert np.allclose(
        routed["route_first_rejection"]["right_joint_degrees"], 0.5
    )


def test_right_first_waypoint_does_not_narrow_the_orientation_search() -> None:
    roll, wing = _right_waypoint_search_overrides(
        False, 15.0, [-0.014, -0.035, 0.0]
    )
    assert roll is None
    assert wing is None

    roll, wing = _right_waypoint_search_overrides(
        True, 15.0, [-0.014, -0.035, 0.0]
    )
    assert roll == (15.0,)
    assert len(wing) == 1
    assert np.allclose(wing[0], [-0.014, -0.035, 0.0])


def test_right_cartesian_entry_routes_are_continuous_and_aisle_bounded() -> None:
    routes = _right_cartesian_entry_offset_routes()

    assert routes[0]["name"] == "aisle_only"
    handoff = next(
        route for route in routes
        if route["name"] == "aisle_handoff_160mm"
    )
    assert np.allclose(handoff["task_offsets_m"][-1], [0.0, 0.16, 0.0])
    prestage_distances = {
        'aisle_prestage_100mm': 0.10,
        'aisle_prestage_90mm': 0.09,
        'aisle_prestage_80mm': 0.08,
        'aisle_prestage_80mm_positive_40mm': 0.08,
        'aisle_prestage_80mm_negative_40mm': 0.08,
        'aisle_prestage_80mm_lift_60mm': 0.08,
        'aisle_prestage_80mm_positive_40mm_lift_60mm': 0.08,
        'aisle_prestage_80mm_negative_40mm_lift_60mm': 0.08,
    }
    assert all(
        np.isclose(
            next(
                route for route in routes if route['name'] == name
            )['task_offsets_m'][-1][1],
            distance,
        )
        for name, distance in prestage_distances.items()
    )
    assert len({route["name"] for route in routes}) == len(routes)
    assert any(
        np.isclose(route["task_offsets_m"][-1][2], 0.2)
        for route in routes
    )
    assert any(
        np.min(np.asarray(route["task_offsets_m"])[:, 2]) < 0.0
        for route in routes
    )
    routes_by_name = {route["name"]: route for route in routes}
    assert np.isclose(
        np.min(
            np.asarray(
                routes_by_name[
                    "drop_20mm_extended_aisle_then_lift"
                ]["task_offsets_m"]
            )[:, 2]
        ),
        -0.020,
    )
    assert np.isclose(
        np.min(
            np.asarray(
                routes_by_name[
                    "drop_30mm_extended_aisle_then_lift"
                ]["task_offsets_m"]
            )[:, 2]
        ),
        -0.030,
    )
    lateral_extended = np.asarray(
        routes_by_name[
            "drop_entry_positive_lateral_extended_aisle_then_lift"
        ]["task_offsets_m"]
    )
    assert np.isclose(np.max(lateral_extended[:, 0]), 0.040)
    assert np.isclose(lateral_extended[-1, 1], 0.550)
    assert np.isclose(lateral_extended[-1, 2], 0.140)
    assert np.isclose(
        np.max(
            np.asarray(
                routes_by_name[
                    "drop_entry_positive_60mm_lateral_extended_aisle_then_lift"
                ]["task_offsets_m"]
            )[:, 0]
        ),
        0.060,
    )
    assert np.isclose(
        np.max(
            np.asarray(
                routes_by_name[
                    "drop_entry_positive_80mm_lateral_extended_aisle_then_lift"
                ]["task_offsets_m"]
            )[:, 0]
        ),
        0.080,
    )
    for route in routes:
        offsets = np.asarray(route["task_offsets_m"])
        assert offsets.ndim == 2 and offsets.shape[1] == 3
        assert np.all(np.isfinite(offsets))
        deltas = np.diff(
            np.vstack((np.zeros((1, 3), dtype=np.float64), offsets)),
            axis=0,
        )
        assert np.max(np.linalg.norm(deltas, axis=1)) <= 0.0100001
        if route["name"].startswith("aisle_handoff_160mm"):
            assert np.isclose(offsets[-1][1], 0.16)
        elif route['name'] in prestage_distances:
            assert np.isclose(
                offsets[-1][1], prestage_distances[route['name']]
            )
        else:
            assert 0.3 <= offsets[-1][1] <= 0.55
        assert np.max(np.abs(offsets[:, 0])) <= 0.08
        assert np.max(offsets[:, 2]) <= 0.2


def test_right_route_waypoint_recovery_seeds_are_unique_and_safe_first() -> None:
    warm = np.arange(7, dtype=np.float64)
    seeds = _right_route_waypoint_seed_candidates(warm)

    assert np.allclose(seeds[0], warm)
    assert np.allclose(seeds[1], _RIGHT_SAFE_DEGREES)
    assert len(seeds) == len({tuple(seed) for seed in seeds})
    assert all(seed.shape == (7,) and np.all(np.isfinite(seed)) for seed in seeds)


def test_right_stow_connection_waypoints_are_bounded_distal_tucks() -> None:
    candidates = _right_stow_connection_waypoint_candidates(
        np.zeros(7),
        np.full(7, 10.0),
        np.full(7, -180.0),
        np.full(7, 180.0),
    )

    assert len(candidates) == 24
    assert {candidate['joint_index'] for candidate in candidates} == {3, 4, 5, 6}
    assert len({tuple(candidate['joint_degrees']) for candidate in candidates}) == 24
    assert all(
        np.all(candidate['joint_degrees'] > -180.0)
        and np.all(candidate['joint_degrees'] < 180.0)
        for candidate in candidates
    )


def test_right_stow_two_bend_waypoints_hold_tuck_across_traverse() -> None:
    candidates = _right_stow_connection_two_bend_candidates(
        np.zeros(7),
        np.full(7, 10.0),
        np.full(7, -180.0),
        np.full(7, 180.0),
    )

    assert len(candidates) == 24
    for candidate in candidates:
        tucked_start, tucked_goal = candidate['joint_degrees']
        joint_index = candidate['joint_index']
        offset = candidate['offset_degrees']
        assert np.isclose(tucked_start[joint_index], offset)
        assert np.isclose(tucked_goal[joint_index], 10.0 + offset)
        assert np.allclose(
            np.delete(tucked_start, joint_index),
            np.zeros(6),
        )
        assert np.allclose(
            np.delete(tucked_goal, joint_index),
            np.full(6, 10.0),
        )


def test_right_cartesian_entry_shortlist_keeps_distinct_safe_route_families() -> None:
    routes = _right_cartesian_entry_offset_routes()
    selected = _right_cartesian_entry_route_shortlist(routes)

    assert [route['name'] for route in selected] == [
        'aisle_prestage_100mm',
        'aisle_prestage_90mm',
        'aisle_prestage_80mm',
        'aisle_prestage_80mm_positive_40mm',
        'aisle_prestage_80mm_negative_40mm',
        'aisle_prestage_80mm_lift_60mm',
        'aisle_prestage_80mm_positive_40mm_lift_60mm',
        'aisle_prestage_80mm_negative_40mm_lift_60mm',
        'aisle_handoff_160mm',
        'aisle_handoff_160mm_positive_40mm',
        'aisle_handoff_160mm_negative_40mm',
        'aisle_handoff_160mm_lift_100mm',
        'aisle_handoff_160mm_positive_40mm_lift_100mm',
        'aisle_handoff_160mm_negative_40mm_lift_100mm',
    ]
    assert len(selected) == 14


def test_right_cartesian_entry_route_rejects_invalid_distances() -> None:
    for kwargs in (
        {"step_m": 0.0},
        {"step_m": float("nan")},
        {"aisle_distance_m": -0.1},
    ):
        try:
            _right_cartesian_entry_offset_routes(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid route inputs were accepted: {kwargs}")


def test_guarded_jaw_handoff_requires_persistent_small_drive_residual() -> None:
    consecutive = 0
    for error_m in (0.0024, 0.0023, 0.00245):
        consecutive, accepted, mode = _guarded_jaw_handoff_step(
            error_m, consecutive
        )
        assert not accepted
        assert mode is None
    consecutive, accepted, mode = _guarded_jaw_handoff_step(
        0.00235, consecutive
    )
    assert consecutive == 4
    assert accepted
    assert mode == "persistent_guarded_handoff"

    consecutive, accepted, mode = _guarded_jaw_handoff_step(0.003, consecutive)
    assert (consecutive, accepted, mode) == (0, False, None)
    consecutive, accepted, mode = _guarded_jaw_handoff_step(0.0019, 0)
    assert consecutive == 4
    assert accepted
    assert mode == "strict_jaw_centre_tolerance"


def test_committed_cut_axial_pretension_tensions_cut_to_grasp_span() -> None:
    direction = _left_committed_cut_axial_pretension_direction(
        np.asarray([1.0, 2.0, 3.0]),
        np.asarray([1.0, 5.0, 7.0]),
    )
    assert np.allclose(direction, [0.0, 0.6, 0.8])
    assert np.linalg.norm(direction) == pytest.approx(1.0)
    reverse = _left_committed_cut_axial_pretension_direction(
        np.asarray([1.0, 5.0, 7.0]), np.asarray([1.0, 2.0, 3.0])
    )
    assert np.allclose(reverse, -direction)
    with pytest.raises(ValueError):
        _left_committed_cut_axial_pretension_direction(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        _left_committed_cut_axial_pretension_direction(np.zeros(3), np.zeros(3))


def test_elbow_up_counterhold_seed_converges_inside_online_budget() -> None:
    model = robot_kinematics.Rby1Kinematics()
    base = robot_kinematics.base_transform(
        (10.7, 4.679695680355915, -0.15254085567917297),
        90.0,
    )
    pointing = np.asarray(
        [-0.421620935, -0.906772180, -0.000001496],
        dtype=np.float64,
    )
    transverse = np.asarray(
        [0.906407390, -0.421451270, -0.028363270],
        dtype=np.float64,
    )
    result = model.solve_position_axes(
        "left",
        local_point_m=_LEFT_JAW_CENTRE_M,
        target_point_m=(10.767863273, 5.067706111, 1.365662569),
        seed_degrees=_LEFT_MULTISTART_SEEDS_DEGREES[-1],
        base_matrix=base,
        pointing_axis=2,
        pointing_direction=pointing,
        transverse_axis=0,
        transverse_to=np.cross(pointing, transverse),
        transverse_direction=transverse,
        position_scale_m=0.002,
        maximum_evaluations=250,
    )
    capacity = model.point_force_capacity(
        "left",
        result.joint_degrees,
        base,
        _LEFT_JAW_CENTRE_M,
        (-0.001403642, 0.901521456, 0.432732128),
        66.3,
        _RBY1_ARM_EFFORT_LIMITS_NM,
    )

    assert result.succeeded
    assert result.evaluations <= 50
    assert model.arm_joint_limit_margin_degrees(
        "left", result.joint_degrees
    ) >= 38.0
    assert capacity.force_capacity_n >= 76.0


def test_counterhold_capacity_requires_full_blade_reaction_load() -> None:
    assert _minimum_left_counterhold_capacity_n(66.3) == pytest.approx(72.93)
    for invalid in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            _minimum_left_counterhold_capacity_n(invalid)


def test_counterhold_posture_requires_capacity_and_joint_reserve() -> None:
    assert _left_counterhold_posture_acceptable(72.93, 72.93, 5.0)
    assert not _left_counterhold_posture_acceptable(72.929, 72.93, 5.0)
    assert not _left_counterhold_posture_acceptable(80.0, 72.93, 4.999)
    for metrics in (
        (float("nan"), 72.93, 5.0),
        (80.0, float("inf"), 5.0),
        (80.0, 72.93, -1.0),
        (0.0, 72.93, 5.0),
    ):
        with pytest.raises(ValueError):
            _left_counterhold_posture_acceptable(*metrics)


def test_committed_axial_pretension_prefers_capacity_after_distance() -> None:
    def attempt(distance_m, capacity_n, joint_value, accepted=True):
        return {
            "distance_m": distance_m,
            "accepted": accepted,
            "hold_capacity": SimpleNamespace(force_capacity_n=capacity_n),
            "solution": SimpleNamespace(
                joint_degrees=(joint_value,) * 7
            ),
        }

    attempts = (
        attempt(0.002, 100.0, 0.0),
        attempt(0.003, 47.0, 0.1),
        attempt(0.003, 77.0, 2.0),
        attempt(0.003, 77.0, 1.0),
        attempt(0.004, 200.0, 0.0, accepted=False),
    )
    selected = _select_left_committed_cut_axial_pretension(
        attempts, np.zeros(7)
    )
    assert selected is attempts[3]
    assert (
        _select_left_committed_cut_axial_pretension(
            [attempt(0.003, 80.0, 0.0, accepted=False)],
            np.zeros(7),
        )
        is None
    )
    with pytest.raises(ValueError):
        _select_left_committed_cut_axial_pretension(
            attempts, np.zeros(6)
        )
    with pytest.raises(ValueError):
        _select_left_committed_cut_axial_pretension(
            [attempt(float("nan"), 80.0, 0.0)],
            np.zeros(7),
        )
    with pytest.raises(ValueError):
        _select_left_committed_cut_axial_pretension(
            [attempt(0.003, float("inf"), 0.0)],
            np.zeros(7),
        )


def test_left_pretension_specs_search_largest_safe_aisle_pull_first() -> None:
    specs = _left_pretension_pull_specs(
        petiole_axis=[0.0, 1.0, 1.0],
        aisle_direction=[0.0, -1.0, 0.0],
    )

    expected_distances = (
        0.016,
        0.014,
        0.012,
        2.0 * 0.016 / 3.0,
        0.010,
        0.008,
        0.006,
        0.016 / 3.0,
    )
    assert len(specs) == 3 * len(expected_distances)
    assert tuple(dict.fromkeys(
        spec["distance_m"] for spec in specs
    )) == pytest.approx(expected_distances)
    for distance_m in expected_distances:
        assert sum(
            np.isclose(spec["distance_m"], distance_m)
            for spec in specs
        ) == 3
    assert {spec["mode"] for spec in specs} == {
        "direct_aisle_left_counterpull",
        "direct_aisle",
        "petiole_axis_to_aisle",
    }
    diagonal = next(
        spec for spec in specs
        if spec["mode"] == "direct_aisle_left_counterpull"
    )["direction"]
    aisle = np.array([0.0, -1.0, 0.0])
    robot_left = np.cross(aisle, [0.0, 0.0, 1.0])
    assert np.dot(diagonal, aisle) == pytest.approx(np.cos(np.radians(35.0)))
    assert np.dot(diagonal, robot_left) == pytest.approx(
        np.sin(np.radians(35.0))
    )
    for spec in specs:
        assert np.isclose(np.linalg.norm(spec["direction"]), 1.0)
        assert np.dot(spec["direction"], [0.0, -1.0, 0.0]) > 0.0


def test_planning_pretension_uses_full_physical_mid_approach_hold() -> None:
    solve_calls = []

    class PretensionModel:
        def forward(self, _side, _degrees, _base):
            return np.eye(4)

        def solve_pose(self, _side, desired, seed, _base):
            translation = np.asarray(desired[:3, 3], dtype=np.float64)
            solve_calls.append((
                translation.copy(), np.asarray(seed, dtype=np.float64).copy()
            ))
            return robot_kinematics.IKResult(
                joint_degrees=tuple(translation) + (0.0,) * 4,
                position_error_m=0.0,
                orientation_error_rad=0.0,
                cost=0.0,
                succeeded=True,
            )

    screened = []

    def trajectory_screen(degrees):
        screened.append(tuple(degrees))
        return {
            "feasible": np.linalg.norm(degrees[:3]) >= 0.016 - 1e-12,
            "minimum_clearance_m": 0.010,
        }
    posture_screened = []

    def posture_screen(degrees):
        posture_screened.append(tuple(degrees))
        return {"feasible": True, "force_capacity_n": 80.0}


    result = _planning_left_pretension_candidates(
        PretensionModel(),
        base_matrix=np.eye(4),
        left_degrees=np.zeros(7),
        petiole_axis=[0.0, 1.0, 0.0],
        trajectory_screen=trajectory_screen,
        posture_screen=posture_screen,
    )

    assert result["feasible"]
    expected_direction = np.array([
        -np.cos(np.radians(35.0)),
        np.sin(np.radians(35.0)),
        0.0,
    ])
    candidate = result["candidates"][0]
    solution = np.asarray(
        candidate["solution"]["joint_degrees"]
    )[:3]
    initial = np.asarray(
        candidate["initial_solution"]["joint_degrees"]
    )[:3]
    assert result["mode"] == "direct_aisle_left_counterpull"
    assert np.isclose(result["initial_distance_m"], 0.016)
    assert np.isclose(result["additional_distance_m"], 0.009)
    assert np.isclose(result["total_distance_m"], 0.025)
    assert np.allclose(candidate["direction"], expected_direction)
    assert np.allclose(solution, 0.025 * expected_direction)
    assert np.allclose(initial, 0.016 * expected_direction)
    assert any(
        np.allclose(target[:3], 0.025 * expected_direction)
        and np.allclose(seed[:3], 0.016 * expected_direction)
        for target, seed in solve_calls
    )
    assert screened
    assert posture_screened
    assert candidate["posture"]["force_capacity_n"] == 80.0

    rejected = _planning_left_pretension_candidates(
        PretensionModel(),
        base_matrix=np.eye(4),
        left_degrees=np.zeros(7),
        petiole_axis=[0.0, 1.0, 0.0],
        trajectory_screen=trajectory_screen,
        posture_screen=lambda _degrees: {"feasible": False},
    )
    assert not rejected["feasible"]
    assert rejected["attempts"]
    assert all(not attempt["posture"]["feasible"] for attempt in rejected["attempts"])



def test_initial_pretension_release_retry_is_one_shot_at_outer_entry() -> None:
    pull = {"distance_m": 0.015}
    start = np.zeros(7)

    assert _initial_pretension_release_retry_allowed(
        -0.16, -0.16, 0, pull, start, False
    )
    assert not _initial_pretension_release_retry_allowed(
        -0.16, -0.16, 1, pull, start, False
    )
    assert not _initial_pretension_release_retry_allowed(
        -0.14, -0.16, 0, pull, start, False
    )
    assert not _initial_pretension_release_retry_allowed(
        -0.16, -0.16, 0, pull, start, True
    )
    assert not _initial_pretension_release_retry_allowed(
        -0.16, -0.16, 0, {"distance_m": 0.0}, start, False
    )


def test_left_pretension_selects_largest_diagonal_pull_then_best_reserve() -> None:
    def assessment(
        distance_m: float,
        payload_m: float,
        inter_arm_m: float,
        mode: str = "direct_aisle",
    ):
        return {
            "mode": mode,
            "distance_m": distance_m,
            "payload_clearance": {"clearance_m": payload_m},
            "inter_arm_clearance": {"clearance_m": inter_arm_m},
        }

    candidates = [
        assessment(0.015, 0.030, 0.160, "petiole_axis_to_aisle"),
        assessment(0.015, 0.018, 0.145, "direct_aisle_left_counterpull"),
        assessment(0.015, 0.020, 0.150),
        assessment(0.015, 0.025, 0.155),
        assessment(0.005, 0.009, 0.150),
        assessment(0.005, 0.012, 0.140, "petiole_axis_to_aisle"),
        assessment(0.010, 0.030, 0.160),
    ]

    assert _select_left_pretension_pull(candidates) is candidates[1]
    assert _select_left_pretension_pull([]) is None


def test_right_entry_releases_pull_only_for_live_foliage_reserve() -> None:
    report = {
        'preplanning_vine_settle': {
            'planning_foliage_clearance_mm': 1.5,
        }
    }
    payload = {
        'clearance_m': 0.0012,
        'nearest_obstacle': (
            '/World/InteractiveVines/Vine/FoliageContact_1'
        ),
    }
    assert _right_entry_pretension_release_required(payload, report)
    payload['clearance_m'] = 0.006
    assert not _right_entry_pretension_release_required(payload, report)
    payload['clearance_m'] = 0.0012
    payload['nearest_obstacle'] = '/World/Greenhouse/Gutter'
    assert not _right_entry_pretension_release_required(payload, report)


def test_pretension_release_requires_stationary_right_foliage_reserve() -> None:
    report = {
        "preplanning_vine_settle": {
            "planning_foliage_clearance_mm": 1.5,
        }
    }

    assert not _stationary_right_pretension_release_clear(
        {"clearance_m": 0.00559}, report
    )
    assert _stationary_right_pretension_release_clear(
        {"clearance_m": 0.0056}, report
    )


def test_left_pretension_release_target_retains_bounded_residual() -> None:
    start = np.zeros(7)
    pulled = np.full(7, 10.0)

    target = _left_pretension_release_target(
        start, pulled, 0.015, 0.003
    )

    assert np.allclose(target, 2.0)
    assert np.allclose(
        _left_pretension_release_target(start, pulled, 0.015, 0.0),
        start,
    )
    assert np.allclose(
        _left_pretension_release_target(start, pulled, 0.015, 0.015),
        pulled,
    )


def test_mid_pretension_recovery_searches_largest_nonzero_release_first() -> None:
    start = np.zeros(7)
    pulled = np.full(7, 9.0)

    candidates = _left_mid_pretension_release_candidates(
        start, pulled, 0.009
    )

    assert np.allclose(
        [candidate["released_distance_m"] for candidate in candidates],
        [0.009, 0.0075, 0.006, 0.0045, 0.003, 0.0015],
    )
    assert np.allclose(candidates[0]["target_degrees"], start)
    assert np.allclose(
        candidates[-1]["target_degrees"], np.full(7, 7.5)
    )
    assert all(candidate["released_distance_m"] > 0.0 for candidate in candidates)


def test_mid_pretension_restore_retries_only_predictive_right_foliage_stops() -> None:
    termination = {
        "reason": "live_payload_replan_reserve",
        "payload_side": "right",
        "clearance_m": 0.005491,
        "guard_m": 0.0056,
        "nearest_obstacle": (
            "link_right_arm_5/capsule_00 <-> "
            "/World/InteractiveVines/Vine/Physics/Organ/Link/"
            "FoliageContact_1"
        ),
    }

    assert _left_mid_pretension_restore_action(termination, 1, 2) == (
        "retry_from_measured_stop"
    )
    assert _left_mid_pretension_restore_action(termination, 2, 2) == (
        "fail_closed"
    )
    assert _left_mid_pretension_restore_action(
        {**termination, "payload_side": "left"}, 1, 2
    ) == "fail_closed"
    assert _left_mid_pretension_restore_action(
        {**termination, "nearest_obstacle": "/World/Greenhouse/Gutter"},
        1,
        2,
    ) == "fail_closed"
    assert _left_mid_pretension_restore_action(None, 1, 2) == "complete"
    assert _left_mid_pretension_restore_action(termination, 2) == (
        "retry_from_measured_stop"
    )
    assert _left_mid_pretension_restore_action(termination, 3) == (
        "fail_closed"
    )


def test_mid_approach_pretension_searches_largest_bounded_pull_first() -> None:
    distances = _left_mid_approach_pretension_distances(0.009)

    assert np.allclose(
        distances,
        [0.009, 0.0075, 0.006, 0.0045, 0.003, 0.0015],
    )
    assert distances[0] == _LEFT_MID_APPROACH_ADDITIONAL_PRETENSION_M
    assert all(0.0 < distance <= 0.009 for distance in distances)

    for invalid in (0.0, -0.001, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            _left_mid_approach_pretension_distances(invalid)


def test_mid_approach_pretension_reuses_safe_direction_fallbacks() -> None:
    selected = np.array([-0.573576, -0.819152, 0.0])
    specs = _left_mid_approach_pretension_direction_specs(
        "direct_aisle_left_counterpull",
        selected,
        np.array([-0.35, -0.79, 0.51]),
        np.array([0.0, -1.0, 0.0]),
    )

    assert [spec["mode"] for spec in specs] == [
        "direct_aisle_left_counterpull",
        "direct_aisle",
        "petiole_axis_to_aisle",
    ]
    assert np.allclose(specs[0]["direction"], selected / np.linalg.norm(selected))
    assert all(np.isclose(np.linalg.norm(spec["direction"]), 1.0) for spec in specs)

    with pytest.raises(ValueError):
        _left_mid_approach_pretension_direction_specs(
            "unsupported", selected, [0.0, -1.0, 0.0], [0.0, -1.0, 0.0]
        )


def test_verified_seating_pull_is_retained_inside_cut_corridor_envelope() -> None:
    maximum = _LEFT_CUT_CORRIDOR_MAXIMUM_INITIAL_PRETENSION_M

    assert np.isclose(maximum, 0.016)
    assert _left_cut_corridor_pretension_release_required(
        {"distance_m": maximum + 0.001}
    )
    assert not _left_cut_corridor_pretension_release_required(
        {"distance_m": maximum}
    )
    assert not _left_cut_corridor_pretension_release_required(None)


def test_mid_approach_pull_and_live_replan_budget_are_bounded() -> None:
    assert np.isclose(_LEFT_MID_APPROACH_ADDITIONAL_PRETENSION_M, 0.009)
    assert _RIGHT_LIVE_WAYPOINT_REPLAN_MAX_ATTEMPTS == 12
    assert _RIGHT_SERVO_MAX_ATTEMPTS == 7
    assert _RIGHT_ROUTE_RRT_MAX_ITERATIONS == 100


def test_shorter_mid_approach_pull_preserves_validated_speed() -> None:
    assert _scaled_motion_steps(180, 0.003, 0.005) == 108
    assert _scaled_motion_steps(1, 0.001, 0.005) == 1


def test_pre_authoring_gripper_geometry_includes_palm_and_open_jaws() -> None:
    palm = _LEFT_GRIPPER_PLANNING_GEOMETRY["palm"]
    fingers = _LEFT_GRIPPER_PLANNING_GEOMETRY["fingers"]
    finger_1 = fingers["ee_finger_l1"]
    finger_2 = fingers["ee_finger_l2"]

    assert palm["collider_min_ee_m"] == (-0.063, -0.0325, -0.025)
    assert palm["collider_max_ee_m"] == (0.063, 0.0325, 0.0)
    assert (
        0.5 * (palm["collider_min_ee_m"][2] + palm["collider_max_ee_m"][2])
        == -0.0125
    )

    assert finger_1["collider_min_ee_m"][0] == _LEFT_GRIPPER_OPEN_WIDTH_M
    assert finger_2["collider_max_ee_m"][0] == -_LEFT_GRIPPER_OPEN_WIDTH_M
    assert (
        finger_1["collider_min_ee_m"][0]
        - finger_2["collider_max_ee_m"][0]
    ) == 0.10


def test_blade_traversal_requires_moving_consecutive_contact() -> None:
    steps, ready = _blade_traversal_contact_step(False, 0.10, 1)
    assert (steps, ready) == (0, False)

    steps, ready = _blade_traversal_contact_step(True, 0.0, 0)
    assert (steps, ready) == (0, False)

    steps, ready = _blade_traversal_contact_step(True, 0.02, 0)
    assert (steps, ready) == (1, False)
    steps, ready = _blade_traversal_contact_step(True, 0.02, steps)
    assert (steps, ready) == (2, True)

    steps, ready = _blade_traversal_contact_step(False, 0.02, steps)
    assert (steps, ready) == (0, False)


def test_structural_guarded_close_rejects_selected_branch_foliage() -> None:
    contact = {
        "key": "Vine_0002/SubStem_14",
        "role": "foliage_grasp",
        "fingers": ["left_finger_1"],
        "force_n": 2.0,
    }

    assert _target_contact_supports_guarded_close(
        contact, "Vine_0002/SubStem_14"
    )
    assert not _target_contact_supports_guarded_close(
        contact,
        "Vine_0002/SubStem_14",
        structural_only=True,
    )
    contact["role"] = "petiole_grasp"
    assert _target_contact_supports_guarded_close(
        contact,
        "Vine_0002/SubStem_14",
        structural_only=True,
    )


def test_bounded_robot_nudge_respects_session_and_aisle_limits() -> None:
    first = _bounded_robot_forward_nudge(
        [0.0, 4.25, 0.0],
        [0.0, 4.25, 0.0],
        90.0,
        0.01,
        4.0,
        4.40,
    )
    assert abs(first["position_m"][1] - 4.26) < 1e-9
    assert abs(first["applied_delta_m"] - 0.01) < 1e-9
    assert not first["limited"]

    session_limited = _bounded_robot_forward_nudge(
        [0.0, 4.33, 0.0],
        [0.0, 4.25, 0.0],
        90.0,
        0.05,
        4.0,
        4.40,
    )
    assert abs(session_limited["position_m"][1] - 4.35) < 1e-9
    assert abs(session_limited["forward_offset_m"] - 0.10) < 1e-9
    assert session_limited["limited"]

    aisle_limited = _bounded_robot_forward_nudge(
        [0.0, 4.25, 0.0],
        [0.0, 4.25, 0.0],
        90.0,
        0.10,
        4.0,
        4.27,
    )
    assert abs(aisle_limited["position_m"][1] - 4.27) < 1e-9
    assert aisle_limited["limited"]


def test_opposed_finger_contact_requires_both_fingers_and_nonzero_load() -> None:
    assert not _opposed_finger_contact(None)
    assert not _opposed_finger_contact(
        {"fingers": ["left_finger_1", "left_finger_2"], "force_n": 0.0}
    )
    assert not _opposed_finger_contact(
        {"fingers": ["left_finger_1"], "force_n": 5.0}
    )
    assert _opposed_finger_contact(
        {"fingers": ["left_finger_1", "left_finger_2"], "force_n": 5.0}
    )


def test_backstop_closure_holds_free_finger_closed_before_loaded_finger() -> None:
    schedule = _opposed_backstop_closure_schedule(
        {"fingers": ["left_finger_1"], "force_n": 4.0}
    )

    assert schedule[0]["stage"] == "opposite_finger_backstop"
    assert schedule[0]["left_finger_1"] == 1.0
    assert schedule[0]["left_finger_2"] == 0.0
    assert all(step["left_finger_2"] == 0.0 for step in schedule)
    assert [step["left_finger_1"] for step in schedule] == [
        1.0,
        0.65,
        0.40,
        0.20,
        0.0,
    ]


def test_backstop_closure_mirrors_for_second_finger_contact() -> None:
    schedule = _opposed_backstop_closure_schedule(
        {"fingers": ["left_finger_2"], "force_n": 4.0}
    )

    assert all(step["left_finger_1"] == 0.0 for step in schedule)
    assert schedule[0]["left_finger_2"] == 1.0
    assert schedule[-1]["left_finger_2"] == 0.0
    assert _opposed_backstop_closure_schedule(None) == ()
    assert _opposed_backstop_closure_schedule(
        {"fingers": ["left_finger_1", "left_finger_2"]}
    ) == ()


def test_guarded_close_requires_loaded_contact_on_selected_branch() -> None:
    contact = {
        "key": "Vine_0002/SubStem_00",
        "fingers": ["left_finger_1"],
        "force_n": 7.0,
    }

    assert _target_contact_supports_guarded_close(
        contact, "Vine_0002/SubStem_00"
    )
    assert not _target_contact_supports_guarded_close(
        contact, "Vine_0002/SubStem_01"
    )
    assert not _target_contact_supports_guarded_close(
        {**contact, "force_n": 0.0}, "Vine_0002/SubStem_00"
    )


def test_target_contact_point_requires_finite_selected_target_point() -> None:
    contact = {
        "key": "Vine_0002/SubStem_00",
        "fingers": ["left_finger_1"],
        "force_n": 7.0,
        "point_m": [10.5, 5.0, 0.7],
    }

    assert _target_contact_point(contact, "Vine_0002/SubStem_00") == (
        10.5,
        5.0,
        0.7,
    )
    assert _target_contact_point(contact, "Vine_0002/SubStem_01") is None
    assert _target_contact_point(
        {**contact, "point_m": [10.5, float("nan"), 0.7]},
        "Vine_0002/SubStem_00",
    ) is None


def test_transformed_cube_bounds_include_scale_rotation_and_translation() -> None:
    matrix = np.eye(4)
    matrix[:3, :3] = np.asarray(
        [
            [0.0, -3.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
        ]
    )
    matrix[:3, 3] = [10.0, 5.0, 1.0]

    minimum, maximum = _transformed_cube_bounds(matrix, 0.5)
    assert np.allclose(minimum, [8.5, 4.0, -1.0])
    assert np.allclose(maximum, [11.5, 6.0, 3.0])


def test_moving_state_reconstruction_limits_are_bounded() -> None:
    position, orientation = _measured_state_reconstruction_limits(0.002, 0.01)

    assert np.isclose(position, 0.0055)
    assert np.isclose(orientation, np.radians(1.0) + 0.0125)
    assert _measured_state_reconstruction_limits(1.0, 1.0) == (
        0.010,
        np.radians(3.0),
    )
    with pytest.raises(ValueError):
        _measured_state_reconstruction_limits(-0.001, 0.0)





def test_measured_state_load_allowance_keeps_unloaded_numeric_reserve() -> None:
    assert _measured_state_settled_position_limit_m("left", "grasped") == 0.005
    assert (
        _measured_state_settled_position_limit_m("left", "orphan_retained")
        == 0.005
    )
    assert _measured_state_settled_position_limit_m("left", "rigged") == 0.0035
    assert _measured_state_settled_position_limit_m("right", "grasped") == 0.004
    assert _measured_state_settled_position_limit_m("right", "rigged") == 0.0035
    with pytest.raises(ValueError):
        _measured_state_settled_position_limit_m("centre", "grasped")




def test_measured_state_reconstruction_has_separate_bounded_tolerance() -> None:
    assert _measured_state_reconstruction_acceptable(0.00026, 0.0091)
    assert _measured_state_reconstruction_acceptable(0.003, 0.017453292519943295)
    assert not _measured_state_reconstruction_acceptable(0.00301, 0.0)
    assert not _measured_state_reconstruction_acceptable(0.0, 0.0175)
    assert not _measured_state_reconstruction_acceptable(float("nan"), 0.0)


def test_tangential_capture_centres_depth_without_crossing_contact() -> None:
    ee = np.eye(4)
    jaw = np.asarray([0.0, 0.0, -0.1025])
    contact = np.asarray([0.050, 0.015, -0.1030])

    result = _tangential_jaw_capture_target(ee, jaw, contact)

    assert np.allclose(result["contact_point_ee_m"], contact)
    assert np.isclose(result["lateral_contact_offset_m"], 0.050)
    assert np.isclose(result["correction_ee_m"][0], 0.0)
    assert np.allclose(
        result["target_point_m"],
        [0.0, 0.015, -0.1030],
    )
    assert not result["limited"]


def test_lateral_disengage_moves_touched_finger_outward() -> None:
    ee = np.eye(4)
    result = _lateral_jaw_disengage_target(
        ee,
        [0.0, 0.0, -0.1025],
        [0.050, 0.015, -0.1030],
    )

    assert np.allclose(result["translation_ee_m"], [0.012, 0.0, 0.0])
    assert np.allclose(result["target_pose"][:3, 3], [0.012, 0.0, 0.0])
    assert np.isclose(result["predicted_contact_offset_after_m"], 0.038)


def test_lateral_disengage_clears_contact_near_outer_finger_face() -> None:
    result = _lateral_jaw_disengage_target(
        np.eye(4),
        [0.0, 0.0, -0.1025],
        [-0.06417, 0.0, -0.103],
    )

    assert np.isclose(result["requested_translation_m"], 0.01617)
    assert np.allclose(result["translation_ee_m"], [-0.01617, 0.0, 0.0])
    assert np.isclose(result["predicted_contact_offset_after_m"], -0.048)


def test_contact_avoiding_reentry_detours_outer_finger_corner() -> None:
    result = _contact_avoiding_jaw_reentry_targets(
        np.eye(4),
        [0.0, 0.0, -0.1025],
        [-0.06417, -0.01632, -0.103],
    )

    assert np.allclose(
        result['tangential_translation_ee_m'], [0.0, 0.030, 0.0]
    )
    assert np.allclose(
        result['lateral_translation_ee_m'], [-0.03417, 0.0, 0.0]
    )
    assert np.isclose(result['predicted_contact_offset_after_m'], -0.030)
    assert [phase['stage'] for phase in result['phases']] == [
        'tangential_depart',
        'separated_lateral_transfer',
        'corridor_return',
    ]
    assert np.allclose(
        result['phases'][0]['target_pose'][:3, 3],
        [0.0, 0.030, 0.0],
    )
    assert np.allclose(
        result['phases'][1]['target_pose'][:3, 3],
        [-0.03417, 0.030, 0.0],
    )
    assert np.allclose(
        result['phases'][2]['target_pose'][:3, 3],
        [-0.03417, -0.010, 0.0],
    )


def test_contact_avoiding_reentry_mirrors_both_detour_axes() -> None:
    result = _contact_avoiding_jaw_reentry_targets(
        np.eye(4),
        [0.0, 0.0, -0.1025],
        [0.06417, 0.01632, -0.103],
    )

    assert np.allclose(
        result['tangential_translation_ee_m'], [0.0, -0.030, 0.0]
    )
    assert np.allclose(
        result['lateral_translation_ee_m'], [0.03417, 0.0, 0.0]
    )
    assert np.isclose(result['predicted_contact_offset_after_m'], 0.030)


def test_guarded_reentry_accepts_only_bounded_open_jaw_residual() -> None:
    assert _guarded_reentry_pose_acceptable(
        succeeded=False,
        position_error_m=0.001,
        orientation_error_rad=np.radians(1.0),
    )
    assert _guarded_reentry_pose_acceptable(
        succeeded=True,
        position_error_m=0.01,
        orientation_error_rad=0.5,
    )
    assert not _guarded_reentry_pose_acceptable(
        succeeded=False,
        position_error_m=0.00101,
        orientation_error_rad=0.0,
    )
    assert not _guarded_reentry_pose_acceptable(
        succeeded=False,
        position_error_m=0.0,
        orientation_error_rad=np.radians(1.01),
    )


def test_near_exact_disengage_requires_ten_mm_guaranteed_motion() -> None:
    accepted, guaranteed_m = _bounded_disengage_ik_acceptable(
        succeeded=False,
        position_error_m=0.0006224,
        orientation_error_rad=0.008681,
        commanded_translation_m=0.012,
        contact_radius_m=0.1227,
    )
    assert accepted
    assert guaranteed_m > 0.010

    accepted, guaranteed_m = _bounded_disengage_ik_acceptable(
        succeeded=False,
        position_error_m=0.001,
        orientation_error_rad=np.radians(1.0),
        commanded_translation_m=0.012,
        contact_radius_m=0.1227,
    )
    assert not accepted
    assert guaranteed_m < 0.010


def test_backstop_seating_moves_positive_tissue_to_negative_finger() -> None:
    result = _lateral_backstop_seat_target(
        np.eye(4),
        [0.010, -0.005, -0.110],
        [0.050, 0.005, -0.090],
        "left_finger_1",
    )

    assert np.isclose(result["applied_correction_m"], 0.011)
    assert np.allclose(result["translation_ee_m"], [0.011, 0.0, 0.0])
    assert np.isclose(result["predicted_minimum_ee_m"][0], -0.001)
    assert not result["limited"]


def test_backstop_seating_mirrors_and_bounds_correction() -> None:
    result = _lateral_backstop_seat_target(
        np.eye(4),
        [-0.050, -0.005, -0.110],
        [-0.020, 0.005, -0.090],
        "left_finger_2",
    )

    assert np.isclose(result["requested_correction_m"], -0.021)
    assert np.isclose(result["applied_correction_m"], -0.012)
    assert np.allclose(result["translation_ee_m"], [-0.012, 0.0, 0.0])
    assert result["limited"]


def test_anchor_microseat_targets_opposed_finger_centre_strip() -> None:
    result = _jaw_anchor_microseat_target(
        np.eye(4),
        [0.00458, 0.01873, -0.1105],
        "left_finger_1",
    )

    assert np.allclose(
        result["translation_ee_m"],
        [0.006, 0.00473, 0.0],
    )
    assert np.allclose(
        result["predicted_anchor_ee_m"],
        [-0.00142, 0.014, -0.1105],
    )
    assert result["limited"]


def test_anchor_microseat_mirrors_for_second_loaded_finger() -> None:
    result = _jaw_anchor_microseat_target(
        np.eye(4),
        [-0.004, -0.019, -0.105],
        "left_finger_2",
    )

    assert np.allclose(
        result["translation_ee_m"],
        [-0.0055, -0.005, 0.0],
    )
    assert np.allclose(
        result["predicted_anchor_ee_m"],
        [0.0015, -0.014, -0.105],
    )
    assert not result["limited"]


def test_open_finger_reseat_requires_progress_and_a_remaining_gap() -> None:
    assert _open_finger_reseat_allowed(
        "opposite_finger_backstop", 0.019, 0.013
    )
    assert not _open_finger_reseat_allowed(
        "loaded_finger_40_percent", 0.019, 0.013
    )
    assert not _open_finger_reseat_allowed(
        "opposite_finger_backstop", 0.0135, 0.013
    )
    assert not _open_finger_reseat_allowed(
        "opposite_finger_backstop", 0.006, 0.005
    )


def test_open_jaw_capture_corridor_rejects_longitudinal_escape() -> None:
    assert _open_jaw_capture_corridor_overlap(
        [0.047, 0.010, -0.108],
        [0.053, 0.018, -0.098],
    )
    assert not _open_jaw_capture_corridor_overlap(
        [0.047, 0.019, -0.108],
        [0.053, 0.025, -0.098],
    )


def test_closed_jaw_corridor_requires_real_channel_intersection() -> None:
    assert _closing_jaw_corridor_overlap(
        [-0.003, -0.005, -0.110],
        [0.003, 0.005, -0.090],
        0.10,
    )
    assert not _closing_jaw_corridor_overlap(
        [-0.003, -0.005, -0.110],
        [0.003, 0.005, -0.090],
        0.50,
    )
    assert not _closing_jaw_corridor_overlap(
        [0.004, -0.005, -0.110],
        [0.010, 0.005, -0.090],
        0.0,
    )
    assert not _closing_jaw_corridor_overlap(
        [-0.003, -0.005, -0.060],
        [0.003, 0.005, -0.040],
        0.0,
    )


def test_geometric_cut_reaction_requires_edge_penetration_and_is_capped() -> None:
    radius = 0.004
    required = 60.0
    assert _geometric_cut_reaction_force(0.006, radius, required) == 0.0
    assert 0.0 < _geometric_cut_reaction_force(0.005, radius, required) < required
    assert _geometric_cut_reaction_force(0.0035, radius, required) >= required
    assert _geometric_cut_reaction_force(0.0, radius, required) <= 3.0 * required


class _Prim:
    def __init__(self, name: str) -> None:
        self._name = name

    def GetName(self) -> str:
        return self._name


def test_requested_target_vine_is_physics_enabled_first() -> None:
    vines = [_Prim(f"Vine_{index:04d}") for index in range(4)]

    selected = _select_physics_vines(vines, "Vine_0002", 2)

    assert [vine.GetName() for vine in selected] == ["Vine_0002", "Vine_0000"]


def test_only_right_support_gets_selected_orphan_envelope_clearance() -> None:
    descendant = "/World/InteractiveVines/Vine_0000/Physics/Organ_0090/Link_000/Collider"
    main_stem = "/World/InteractiveVines/Vine_0000/Physics/Organ_0034/Link_053/Collider"
    grasp = {"orphan_colliders": (descendant,)}
    protected = 0.008

    assert _required_probe_payload_clearance(
        "right",
        {"component": "knife_support_02", "nearest_obstacle": descendant},
        grasp,
        protected,
    ) == 0.0005
    assert _required_probe_payload_clearance(
        "right",
        {"component": "wrist_d405", "nearest_obstacle": descendant},
        grasp,
        protected,
    ) == protected
    assert _required_probe_payload_clearance(
        "right",
        {"component": "knife_support_02", "nearest_obstacle": main_stem},
        grasp,
        protected,
    ) == protected
    assert _required_probe_payload_clearance(
        "left",
        {"component": "ee_finger_l1", "nearest_obstacle": descendant},
        grasp,
        protected,
    ) == protected


def test_greenhouse_structure_keeps_rigid_clearance() -> None:
    assert _required_probe_payload_clearance(
        "right",
        {
            "component": "right_arm_capsules",
            "nearest_obstacle": "/World/Main_Cultivation_Zone/Env/Wall_02",
        },
        {"orphan_colliders": ()},
        0.005,
    ) == 0.01


def test_payload_clearance_uses_positive_compliant_foliage_gap() -> None:
    payload = {
        "component": "wrist_camera_bracket",
        "nearest_obstacle": (
            "/World/InteractiveVines/Vine_0002/Physics/Organ_0001/"
            "Link_000/FoliageContact_0001"
        ),
    }

    assert _required_probe_payload_clearance(
        "left",
        payload,
        {"colliders": ()},
        0.008,
    ) == 0.0005


def test_left_fingers_may_handle_only_the_selected_orphan_subtree() -> None:
    selected = "/World/InteractiveVines/Vine_0000/Physics/Organ_0060/Link_003/Collider"
    descendant = "/World/InteractiveVines/Vine_0000/Physics/Organ_0090/Link_000/Collider"
    main_stem = "/World/InteractiveVines/Vine_0000/Physics/Organ_0034/Link_053/Collider"
    blade_target = "/World/InteractiveVines/Vine_0000/Physics/Organ_0060/Link_000/Collider"
    finger = "/World/RBY1/ee_finger_l1/restored_collisions/contact_proxy"
    camera = "/World/RBY1/ee_left/WristD405/Collision"
    blade = "/World/RBY1/ee_right/DeleafKnife/BladeCollision"

    summary = {
        "pairs": [
            {"collider0": finger, "collider1": descendant, "maximum_impulse_ns": 0.1},
            {"collider0": camera, "collider1": descendant, "maximum_impulse_ns": 0.1},
            {"collider0": finger, "collider1": main_stem, "maximum_impulse_ns": 0.1},
            {"collider0": blade, "collider1": blade_target, "maximum_impulse_ns": 0.1},
        ]
    }
    unsafe = _probe_unsafe_contacts(
        summary,
        {"colliders": (blade_target,)},
        {
            "collider": selected,
            "colliders": (selected,),
            "orphan_colliders": (selected, descendant),
        },
    )

    assert [(record["collider0"], record["collider1"]) for record in unsafe] == [
        (camera, descendant),
        (finger, main_stem),
    ]


def test_only_the_gripper_may_overlap_a_physically_held_branch() -> None:
    assert _left_payload_component_accepts_target_contact(
        "ee_finger_l1", False
    )
    assert not _left_payload_component_accepts_target_contact(
        "ee_left", False
    )
    assert _left_payload_component_accepts_target_contact(
        "ee_left", True
    )
    assert not _left_payload_component_accepts_target_contact(
        "wrist_d405", True
    )
    assert not _left_payload_component_accepts_target_contact(
        "wrist_camera_bracket", True
    )
    assert not _left_payload_component_accepts_target_contact(
        "left_arm_capsules", True
    )


def test_held_branch_filter_targets_only_owned_foliage_and_palm() -> None:
    foliage = (
        "/World/InteractiveVines/Vine_0002/Physics/Organ_0119/"
        "Link_000/FoliageContact_0236"
    )
    structural = (
        "/World/InteractiveVines/Vine_0002/Physics/Organ_0098/"
        "Link_004/Collider"
    )

    assert _held_branch_palm_collision_filter_pairs(
        (structural, foliage, foliage),
        "/World/RBY1",
    ) == (
        (
            foliage,
            "/World/RBY1/ee_left/restored_collisions/contact_proxy",
        ),
    )
    assert _held_branch_palm_collision_filter_pairs(
        (structural,),
        "/World/RBY1",
    ) == ()

def test_active_contact_summary_excludes_pairs_after_contact_lost() -> None:
    diagnostics = object.__new__(RobotContactDiagnostics)
    diagnostics._reported_bodies = 2
    diagnostics._pairs = {
        ("/lost", "/robot"): {
            "collider0": "/lost",
            "collider1": "/robot",
            "maximum_impulse_ns": 2.0,
            "active": False,
        },
        ("/active", "/robot"): {
            "collider0": "/active",
            "collider1": "/robot",
            "maximum_impulse_ns": 1.0,
            "active": True,
        },
    }
    diagnostics._active_pairs = {("/active", "/robot")}

    assert diagnostics.active_summary == {
        "reported_bodies": 2,
        "pairs": [diagnostics._pairs[("/active", "/robot")]],
    }
    assert len(diagnostics.summary["pairs"]) == 2

def test_teleop_contact_policy_defaults_to_non_pausing_monitor(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["interactive_greenhouse.py"])
    assert parse_args().teleop_contact_policy == "monitor"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "interactive_greenhouse.py",
            "--teleop-contact-policy",
            "rollback",
        ],
    )
    assert parse_args().teleop_contact_policy == "rollback"


def test_parser_accepts_planned_fixed_robot_base(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "interactive_greenhouse.py",
            "--robot-position-mode",
            "planned-fixed",
            "--robot-position",
            "1",
            "2",
            "3",
        ],
    )

    args = parse_args()
    assert args.robot_position_mode == "planned-fixed"
    assert args.robot_position == [1.0, 2.0, 3.0]


def test_parser_accepts_grasp_only_probe(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "interactive_greenhouse.py",
            "--bimanual-probe",
            "grasp",
        ],
    )

    assert parse_args().bimanual_probe == "grasp"


def test_parser_defaults_to_online_base_planning_budget(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["interactive_greenhouse.py"])

    args = parse_args()

    assert args.base_planning_budget == "online"


def test_probe_video_defaults_capture_all_views_at_exact_cadence(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["interactive_greenhouse.py"])

    args = parse_args()

    assert args.probe_video_cameras == (
        "inspection",
        "head",
        "left_wrist",
        "right_wrist",
    )
    assert args.probe_video_width == 480
    assert args.probe_video_height == 270
    assert _probe_capture_interval_steps(args.probe_video_hz) == 20


def test_probe_video_cadence_rejects_non_integral_physics_divisor() -> None:
    assert _probe_capture_interval_steps(24.0) == 10
    with np.testing.assert_raises_regex(
        ValueError,
        "must divide the physics rate exactly",
    ):
        _probe_capture_interval_steps(23.0)

def test_planned_route_start_reconciles_only_inside_physical_bound() -> None:
    assert not _planned_route_start_reconciliation_required(
        0.0,
        0.5,
    )
    assert _planned_route_start_reconciliation_required(
        0.095,
        0.5,
    )
    with pytest.raises(ValueError):
        _planned_route_start_reconciliation_required(0.5001, 0.5)
    for invalid in (
        (float("nan"), 0.5),
        (0.1, 0.0),
        (-0.1, 0.5),
    ):
        with pytest.raises(ValueError):
            _planned_route_start_reconciliation_required(*invalid)


def test_probe_recorder_reuses_last_valid_frame_on_black_renderer_sample(
    tmp_path,
) -> None:
    from PIL import Image

    valid = np.full((4, 6, 4), 127, dtype=np.uint8)
    valid[:, :, 3] = 255
    black = np.zeros_like(valid)

    class Annotator:
        def __init__(self) -> None:
            self.frames = [valid, black]

        def get_data(self):
            return self.frames.pop(0)

    frames = tmp_path / "frames"
    frames.mkdir()
    recorder = object.__new__(_TeleopCameraRecorder)
    recorder._np = np
    recorder._Image = Image
    recorder._frames_directory = frames
    recorder._streams = {"inspection": (None, None, Annotator())}
    recorder._last_rgb = {}

    first_images, first_errors = recorder.capture(0)
    second_images, second_errors = recorder.capture(1)

    assert first_errors == {}
    assert pathlib.Path(first_images["inspection"]).name == (
        "000000_inspection.png"
    )
    assert pathlib.Path(second_images["inspection"]).name == (
        "000001_inspection.png"
    )
    assert "all-black RGB sample" in second_errors["inspection"]
    assert "reused previous valid RGB sample" in second_errors["inspection"]
    np.testing.assert_array_equal(
        np.asarray(Image.open(frames / "000000_inspection.png")),
        np.asarray(Image.open(frames / "000001_inspection.png")),
    )


def test_greenhouse_waiting_pose_is_folded_with_joint_and_inter_arm_reserve() -> None:
    model = robot_kinematics.Rby1Kinematics()
    base = robot_kinematics.base_transform((0.0, 0.0, 0.0), 90.0)

    assert model.arm_joint_limit_margin_degrees(
        "left", _LEFT_GREENHOUSE_WAITING_DEGREES
    ) >= 9.9
    assert model.inter_arm_clearance(
        _LEFT_GREENHOUSE_WAITING_DEGREES,
        _RIGHT_SAFE_DEGREES,
        base,
    ).clearance_m >= 0.20


def test_greenhouse_benchmark_startup_uses_task_safe_stow_arms() -> None:
    targets = _greenhouse_startup_arm_targets()

    assert targets["left"] == _LEFT_GREENHOUSE_WAITING_DEGREES
    assert targets["right"] == _RIGHT_SAFE_DEGREES


def test_greenhouse_teleop_startup_preserves_sdk_ready_arms() -> None:
    targets = _greenhouse_startup_arm_targets(
        task_safe=False
    )

    assert targets["left"] == _LEFT_READY_DEGREES
    assert targets["right"] == _RIGHT_READY_DEGREES


def test_greenhouse_startup_override_changes_only_left_curriculum_arm() -> None:
    left = (1.0, 2.0, 3.0, -120.0, 5.0, 70.0, 7.0)

    targets = _greenhouse_startup_arm_targets(left)

    assert targets["left"] == left
    assert targets["right"] == _RIGHT_SAFE_DEGREES


def test_greenhouse_startup_rejects_malformed_left_override() -> None:
    with pytest.raises(ValueError, match="seven finite"):
        _greenhouse_startup_arm_targets((0.0,) * 6)


def test_greenhouse_startup_rejects_non_boolean_pose_mode() -> None:
    with pytest.raises(ValueError, match="task_safe must be a boolean"):
        _greenhouse_startup_arm_targets(task_safe=1)


def test_measured_task_stow_gate_uses_half_degree_limit() -> None:
    target = np.asarray(_RIGHT_SAFE_DEGREES)
    assert _startup_task_stow_reached(target + 0.49, target)
    assert not _startup_task_stow_reached(target + 0.51, target)
    with pytest.raises(ValueError, match="finite arm states"):
        _startup_task_stow_reached(target[:6], target)


def test_startup_task_stow_waypoints_are_replayed_for_both_arms() -> None:
    for side in ("left", "right"):
        waypoints = _startup_task_stow_waypoints(side)
        assert len(waypoints) == 1
        assert len(waypoints[0]) == 7
        assert np.isfinite(waypoints[0]).all()

    with pytest.raises(ValueError, match="unsupported arm side"):
        _startup_task_stow_waypoints("torso")


def test_bimanual_startup_requires_low_speed_and_small_hold_error() -> None:
    target = np.zeros(7)

    assert _bimanual_startup_joint_state_settled(
        np.full(7, 0.1), target, np.full(7, 0.2)
    )
    assert not _bimanual_startup_joint_state_settled(
        np.full(7, 0.51), target, np.zeros(7)
    )
    assert not _bimanual_startup_joint_state_settled(
        target, target, np.full(7, 0.51)
    )
    assert not _bimanual_startup_joint_state_settled(
        np.zeros(6), target, np.zeros(7)
    )
    with pytest.raises(ValueError, match="limits must be positive"):
        _bimanual_startup_joint_state_settled(
            target,
            target,
            np.zeros(7),
            maximum_speed_degrees_s=0.0,
        )


def test_left_clearance_ingress_waypoint_is_finite() -> None:
    assert len(_LEFT_CLEARANCE_INGRESS_WAYPOINTS_DEGREES) == 1
    waypoint = _LEFT_CLEARANCE_INGRESS_WAYPOINTS_DEGREES[0]
    assert len(waypoint) == 7
    assert np.isfinite(waypoint).all()


def test_right_base_requires_a_local_cut_continuation() -> None:
    endpoint_only = {
        "endpoint_feasible": True,
        "continuation_feasible": False,
    }
    complete = {
        "endpoint_feasible": True,
        "continuation_feasible": True,
    }

    assert not _right_base_endpoint_route_feasible(endpoint_only)
    assert _right_base_endpoint_route_feasible(complete)


def test_minimum_jerk_profile_has_stationary_endpoints_and_is_monotonic() -> None:
    fractions = np.linspace(0.0, 1.0, 101)
    profile = np.asarray(
        [_minimum_jerk_fraction(value) for value in fractions]
    )

    assert profile[0] == 0.0
    assert profile[-1] == 1.0
    assert np.all(np.diff(profile) >= 0.0)
    assert _minimum_jerk_fraction(0.5) == 0.5


def test_short_receding_horizon_move_is_raised_to_physical_bounds() -> None:
    bounded = _bounded_arm_motion_steps(
        np.zeros(7),
        np.asarray((3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)),
        24,
    )

    assert bounded > 24
    assert _bounded_arm_motion_steps(np.zeros(7), np.zeros(7), 24) == 24


def test_default_inspection_camera_stays_on_robot_side_of_gutter() -> None:
    placement = (10.58, 5.10, -0.15)
    eye, target = _default_inspection_camera_pose(placement)

    assert eye[1] < target[1] < placement[1]
    assert eye[2] > target[2]
    assert target[2] > placement[2]


def test_guarded_force_unload_cannot_reverse_behind_audited_start() -> None:
    assert _force_control_unload_fraction(
        0.01, 0.02, guarded_segment=True
    ) == 0.0
    assert _force_control_unload_fraction(
        0.01, 0.02, guarded_segment=False
    ) == -0.01


def test_committed_live_refinement_stays_inside_flat_cutting_edge() -> None:
    rejection = {
        "stub_m": 0.019,
        "roll_degrees": 70.0,
        "edge_wing_local_m": [-0.01418102, -0.018369995, 0.0],
        "error": "live full-stroke preflight was not clear",
        "segments": [{"eligible": False, "payload_clearance_m": 0.005446}],
    }

    requests = _right_committed_live_roll_refinement_requests(
        [rejection],
        {(0.019, 70.0, -0.01418102, -0.018369995, 0.0)},
        minimum_source_clearance_m=0.0051,
        maximum_sources=1,
        wing_minimum_y=-0.07047998,
        wing_maximum_y=-0.001,
        minimum_wing_interior_margin_m=0.005,
        maximum_wing_shift_steps=4,
    )

    shifted = requests[20:]
    assert len(shifted) == 8
    assert np.allclose(
        [request[2][1] for request in shifted[:4]],
        [-0.017369995, -0.019369995, -0.016369995, -0.020369995],
    )
    assert all(request[1] == 70.0 for request in shifted)
    assert all(
        min(request[2][1] + 0.07047998, -0.001 - request[2][1])
        >= 0.005
        for request in shifted
    )


def test_committed_live_refinement_crosses_roll_with_outer_safe_edge() -> None:
    rejection = {
        "stub_m": 0.019,
        "roll_degrees": 70.0,
        "edge_wing_local_m": [-0.01418102, -0.018369995, 0.0],
        "error": "live full-stroke preflight was not clear",
        "segments": [{"eligible": False, "payload_clearance_m": 0.005446}],
    }

    requests = _right_committed_live_roll_refinement_requests(
        [rejection],
        {(0.019, 70.0, -0.01418102, -0.018369995, 0.0)},
        minimum_source_clearance_m=0.0051,
        maximum_sources=1,
        wing_minimum_y=-0.07047998,
        wing_maximum_y=-0.001,
        minimum_wing_interior_margin_m=0.005,
        maximum_wing_shift_steps=2,
        cross_refinement_roll_delta_degrees=1,
        cross_refinement_wing_tail_steps=1,
    )

    crossed = requests[24:]
    assert len(crossed) == 4
    assert {request[1] for request in crossed} == {69.0, 71.0}
    assert np.allclose(
        sorted({request[2][1] for request in crossed}),
        [-0.020369995, -0.016369995],
    )



def test_committed_live_fractional_roll_refinement_targets_ik_boundary() -> None:
    wing = (-0.01418102, -0.018369995, 0.0)
    rejection = {
        "stub_m": 0.0195,
        "roll_degrees": 33.0,
        "edge_wing_local_m": list(wing),
        "error": "live full-stroke preflight was not clear",
        "segments": [
            *(
                {"eligible": True, "payload_clearance_m": 0.006}
                for _ in range(22)
            ),
            {"eligible": False, "payload_clearance_m": 0.005244095},
        ],
    }
    tried = {(0.0195, 33.0, *wing)}

    requests = _right_committed_live_roll_refinement_requests(
        [rejection],
        tried,
        minimum_source_clearance_m=0.005,
        maximum_sources=1,
        fractional_roll_offsets_degrees=(
            -1.25,
            -1.5,
            -1.75,
            1.25,
            1.5,
            1.75,
        ),
        include_standard_roll_refinement=False,
    )

    assert requests == (
        (0.0195, 31.75, wing),
        (0.0195, 31.5, wing),
        (0.0195, 31.25, wing),
        (0.0195, 34.25, wing),
        (0.0195, 34.5, wing),
        (0.0195, 34.75, wing),
    )
    for invalid in ((0.0,), (float("nan"),), (10.0,)):
        with pytest.raises(ValueError, match="fractional live roll offsets"):
            _right_committed_live_roll_refinement_requests(
                [rejection],
                tried,
                minimum_source_clearance_m=0.005,
                fractional_roll_offsets_degrees=invalid,
            )
    with pytest.raises(TypeError, match="standard live roll refinement flag"):
        _right_committed_live_roll_refinement_requests(
            [rejection],
            tried,
            minimum_source_clearance_m=0.005,
            include_standard_roll_refinement=1,
        )
