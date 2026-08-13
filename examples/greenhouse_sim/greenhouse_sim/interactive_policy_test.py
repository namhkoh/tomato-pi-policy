"""Regressions for task-semantic robot/contact safety policy."""

from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from interactive_greenhouse import (
    _LEFT_GRIPPER_OPEN_WIDTH_M,
    _LEFT_GRIPPER_PLANNING_GEOMETRY,
    _LEFT_GREENHOUSE_WAITING_DEGREES,
    _RIGHT_SAFE_DEGREES,
    RobotContactDiagnostics,
    parse_args,
    _blade_traversal_contact_step,
    _bounded_robot_forward_nudge,
    _closing_jaw_corridor_overlap,
    _geometric_cut_reaction_force,
    _left_pretension_pull_specs,
    _lateral_backstop_seat_target,
    _lateral_jaw_disengage_target,
    _measured_state_reconstruction_acceptable,
    _open_jaw_capture_corridor_overlap,
    _opposed_backstop_closure_schedule,
    _opposed_finger_contact,
    _probe_unsafe_contacts,
    _required_probe_payload_clearance,
    _select_physics_vines,
    _target_contact_point,
    _target_contact_supports_guarded_close,
    _tangential_jaw_capture_target,
    _transformed_cube_bounds,
)
from greenhouse_sim import robot_kinematics


def test_left_pretension_specs_search_largest_safe_aisle_pull_first() -> None:
    specs = _left_pretension_pull_specs(
        petiole_axis=[0.0, 1.0, 1.0],
        aisle_direction=[0.0, -1.0, 0.0],
    )

    assert len(specs) == 6
    assert [spec["distance_m"] for spec in specs] == [
        0.015,
        0.015,
        0.01,
        0.01,
        0.005,
        0.005,
    ]
    assert {spec["mode"] for spec in specs} == {
        "petiole_axis_to_aisle",
        "direct_aisle",
    }
    for spec in specs:
        assert np.isclose(np.linalg.norm(spec["direction"]), 1.0)
        assert np.dot(spec["direction"], [0.0, -1.0, 0.0]) > 0.0


def test_pre_authoring_gripper_geometry_matches_full_open_50_mm_jaws() -> None:
    fingers = _LEFT_GRIPPER_PLANNING_GEOMETRY["fingers"]
    finger_1 = fingers["ee_finger_l1"]
    finger_2 = fingers["ee_finger_l2"]

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


def test_measured_state_reconstruction_has_separate_bounded_tolerance() -> None:
    assert _measured_state_reconstruction_acceptable(0.00026, 0.0091)
    assert _measured_state_reconstruction_acceptable(0.001, 0.017453292519943295)
    assert not _measured_state_reconstruction_acceptable(0.00101, 0.0)
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


def test_parser_defaults_to_online_base_planning_budget(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["interactive_greenhouse.py"])

    args = parse_args()

    assert args.base_planning_budget == "online"


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
    ).clearance_m >= 0.10
