"""Regressions for task-semantic robot/contact safety policy."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from interactive_greenhouse import (
    RobotContactDiagnostics,
    parse_args,
    _blade_traversal_contact_step,
    _bounded_robot_forward_nudge,
    _closing_jaw_corridor_overlap,
    _geometric_cut_reaction_force,
    _opposed_finger_contact,
    _probe_unsafe_contacts,
    _required_probe_payload_clearance,
    _select_physics_vines,
)


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
        [0.0, 4.27, 0.0],
        [0.0, 4.25, 0.0],
        90.0,
        0.02,
        4.0,
        4.40,
    )
    assert abs(session_limited["position_m"][1] - 4.28) < 1e-9
    assert abs(session_limited["forward_offset_m"] - 0.03) < 1e-9
    assert session_limited["limited"]

    aisle_limited = _bounded_robot_forward_nudge(
        [0.0, 4.25, 0.0],
        [0.0, 4.25, 0.0],
        90.0,
        0.03,
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
