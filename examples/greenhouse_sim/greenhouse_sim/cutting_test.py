"""Deterministic acceptance cases for directional blade-mediated cuts."""

from __future__ import annotations

import numpy as np

from greenhouse_sim import cutting
from greenhouse_sim import vine_physics
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdPhysics


def _target() -> cutting.CutTarget:
    return cutting.CutTarget(
        key="Vine_0000/SubStem_00",
        organ_label="SubStem_00",
        centre_m=np.zeros(3),
        axis=np.array([0.0, 0.0, 1.0]),
        radius_m=0.003,
        cut_force_n=66.3,
    )


def _sample(
    edge_x: float,
    *,
    impulse_ns: float = 1.0,
    velocity_x: float = 0.2,
    point_z: float = 0.005,
    edge_axis=(0.0, 1.0, 0.0),
    counterhold_active: bool = False,
    commanded_velocity=None,
) -> cutting.BladeContactSample:
    return cutting.BladeContactSample(
        point_m=np.array([0.0, 0.0, point_z]),
        impulse_ns=np.array([impulse_ns, 0.0, 0.0]),
        edge_centre_m=np.array([edge_x, 0.0, 0.0]),
        edge_axis=np.asarray(edge_axis, dtype=np.float64),
        cutting_direction=np.array([1.0, 0.0, 0.0]),
        edge_velocity_m_s=np.array([velocity_x, 0.0, 0.0]),
        dt_s=0.01,
        counterhold_active=counterhold_active,
        commanded_edge_velocity_m_s=(
            None
            if commanded_velocity is None
            else np.asarray(commanded_velocity, dtype=np.float64)
        ),
    )


def test_sustained_forceful_transverse_sweep_cuts() -> None:
    gate = cutting.DirectionalCutGate()
    target = _target()

    assert gate.observe(target, _sample(-0.004)) is None
    assert gate.observe(target, _sample(-0.002)) is None
    decision = gate.observe(target, _sample(0.0))

    assert decision is not None
    assert decision.organ_label == "SubStem_00"
    assert decision.requested_stub_m == 0.005
    assert decision.cut_work_j >= decision.required_work_j
    assert decision.peak_force_n == 100.0
    assert decision.forward_travel_m == 0.004
    assert decision.edge_axis_alignment == 1.0
    assert decision.motion_transverse_alignment == 1.0


def test_insufficient_force_never_accumulates_cut_work() -> None:
    gate = cutting.DirectionalCutGate()
    target = _target()

    decisions = [
        gate.observe(target, _sample(edge_x, impulse_ns=0.5))
        for edge_x in (-0.004, -0.002, 0.0, 0.002)
    ]

    assert decisions == [None, None, None, None]
    progress = gate.progress_for(target.key)
    assert progress.work_j == 0.0
    assert progress.rejections["insufficient_force"] == 4


def test_low_force_physical_entry_records_side_but_not_work() -> None:
    gate = cutting.DirectionalCutGate()
    target = _target()

    assert gate.observe(
        target,
        _sample(-0.004, impulse_ns=0.5, counterhold_active=True),
    ) is None
    progress = gate.progress_for(target.key)
    assert progress.minimum_signed_side_m == -0.004
    assert progress.counterhold_start_side_m == -0.004
    assert progress.work_j == 0.0

    decision = None
    for _ in range(3):
        decision = gate.observe(
            target,
            _sample(
                0.012,
                counterhold_active=True,
                commanded_velocity=(0.2, 0.0, 0.0),
            ),
        )
        if decision is not None:
            break

    assert decision is not None
    assert decision.cut_work_j >= decision.required_work_j


def test_reverse_or_axial_motion_and_parallel_edge_are_rejected() -> None:
    target = _target()
    cases = (
        (_sample(-0.004, velocity_x=-0.2), "wrong_or_slow_direction"),
        (
            cutting.BladeContactSample(
                point_m=np.zeros(3),
                impulse_ns=np.array([1.0, 0.0, 0.0]),
                edge_centre_m=np.array([-0.004, 0.0, 0.0]),
                edge_axis=np.array([0.0, 1.0, 0.0]),
                cutting_direction=np.array([1.0, 0.0, 0.0]),
                edge_velocity_m_s=np.array([0.02, 0.0, 0.2]),
                dt_s=0.01,
            ),
            "motion_not_transverse",
        ),
        (_sample(-0.004, edge_axis=(0.0, 0.0, 1.0)), "edge_not_transverse"),
        (_sample(-0.004, point_z=0.04), "outside_axial_cut_zone"),
    )

    for sample, reason in cases:
        gate = cutting.DirectionalCutGate()
        assert gate.observe(target, sample) is None
        assert gate.progress_for(target.key).rejections[reason] == 1


def test_disconnected_taps_do_not_combine_into_a_cut() -> None:
    gate = cutting.DirectionalCutGate()
    target = _target()
    assert gate.observe(target, _sample(-0.004)) is None
    assert gate.observe(target, _sample(-0.002)) is None
    for _ in range(gate.parameters.contact_memory_steps + 1):
        gate.finish_step(set())

    assert gate.observe(target, _sample(0.0)) is None
    progress = gate.progress_for(target.key)
    assert progress.work_j == 0.0
    assert progress.forward_travel_m == 0.0


def test_pushing_a_moving_petiole_does_not_count_as_cut_travel() -> None:
    gate = cutting.DirectionalCutGate()

    for shift in (0.0, 0.002, 0.004, 0.006):
        target = cutting.CutTarget(
            key="Vine_0000/SubStem_00",
            organ_label="SubStem_00",
            centre_m=np.asarray([shift, 0.0, 0.0]),
            axis=np.asarray([0.0, 0.0, 1.0]),
            radius_m=0.003,
            cut_force_n=66.3,
        )
        sample = _sample(-0.004 + shift)
        assert gate.observe(target, sample) is None

    progress = gate.progress_for(target.key)
    assert progress.valid_contact_steps == 4
    assert progress.forward_travel_m == 0.0
    assert progress.work_j == 0.0


def test_counterheld_rigid_petiole_accumulates_virtual_fracture_travel() -> None:
    gate = cutting.DirectionalCutGate()
    decision = None

    for shift in (0.0, 0.002, 0.004):
        target = cutting.CutTarget(
            key="Vine_0000/SubStem_00",
            organ_label="SubStem_00",
            centre_m=np.asarray([shift, 0.0, 0.0]),
            axis=np.asarray([0.0, 0.0, 1.0]),
            radius_m=0.003,
            cut_force_n=66.3,
        )
        decision = gate.observe(
            target,
            _sample(-0.004 + shift, counterhold_active=True),
        )

    assert decision is not None
    assert decision.forward_travel_m == 0.004
    assert decision.virtual_penetration_m == 0.004
    assert decision.counterheld_contact_steps == 3


def test_counterheld_commanded_penetration_drives_rigid_fracture_proxy() -> None:
    gate = cutting.DirectionalCutGate()
    target = _target()

    first = gate.observe(
        target,
        _sample(
            -0.004,
            velocity_x=0.0,
            counterhold_active=True,
            commanded_velocity=(0.2, 0.0, 0.0),
        ),
    )
    decision = gate.observe(
        target,
        _sample(
            -0.004,
            velocity_x=0.0,
            counterhold_active=True,
            commanded_velocity=(0.2, 0.0, 0.0),
        ),
    )

    assert first is None
    assert decision is not None
    assert decision.forward_travel_m == 0.004
    assert decision.virtual_penetration_m == 0.004


def test_counterheld_full_diameter_traction_handles_guide_displacement() -> None:
    gate = cutting.DirectionalCutGate()
    target = _target()
    decision = None

    for _ in range(4):
        decision = gate.observe(
            target,
            _sample(
                0.012,
                velocity_x=0.0,
                counterhold_active=True,
                commanded_velocity=(0.2, 0.0, 0.0),
            ),
        )
        if decision is not None:
            break

    assert decision is not None
    assert decision.virtual_penetration_m >= 2.0 * target.radius_m
    assert decision.cut_work_j >= decision.required_work_j


def test_latest_contact_feedback_clears_on_a_contact_gap() -> None:
    gate = cutting.DirectionalCutGate()
    target = _target()

    assert gate.observe(target, _sample(-0.004)) is None
    progress = gate.progress_for(target.key)
    assert progress.last_effective_force_n == 100.0
    assert progress.last_forward_speed_m_s == 0.2
    assert progress.last_contact_valid

    gate.finish_step(set())

    assert progress.last_effective_force_n == 0.0
    assert progress.last_forward_speed_m_s == 0.0
    assert not progress.last_contact_valid


def test_internal_geometric_cut_uses_runtime_detachable_organ_base() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Organ")
    base = UsdPhysics.Joint.Define(stage, "/World/Organ/BaseJoint")
    internal = UsdPhysics.Joint.Define(stage, "/World/Organ/Joint_001")
    base.CreateJointEnabledAttr(True)
    internal.CreateJointEnabledAttr(True)
    rig = vine_physics.PlantRig(
        root_path="/World/Organ",
        links=[],
        joints={},
        cut_joints={"SubStem_00": str(base.GetPath())},
    )

    class _Centreline:
        @staticmethod
        def arc_lengths():
            return np.asarray([0.0, 0.010])

    severer = cutting.Severer(
        stage,
        rig,
        {7: _Centreline()},
        {"SubStem_00": 7},
    )
    record = severer.cut("SubStem_00", stub_length_m=0.010)

    assert record.geometric_joint_path == str(internal.GetPath())
    assert record.geometric_stub_m == 0.010
    assert record.joint_path == str(base.GetPath())
    assert record.realised_stub_m == 0.0
    assert record.release_mode == "maximal_coordinate_organ_base"
    assert not base.GetJointEnabledAttr().Get()
    assert internal.GetJointEnabledAttr().Get()
