"""Deterministic acceptance cases for directional blade-mediated cuts."""

from __future__ import annotations

import numpy as np

from greenhouse_sim import cutting


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
) -> cutting.BladeContactSample:
    return cutting.BladeContactSample(
        point_m=np.array([0.0, 0.0, point_z]),
        impulse_ns=np.array([impulse_ns, 0.0, 0.0]),
        edge_centre_m=np.array([edge_x, 0.0, 0.0]),
        edge_axis=np.asarray(edge_axis, dtype=np.float64),
        cutting_direction=np.array([1.0, 0.0, 0.0]),
        edge_velocity_m_s=np.array([velocity_x, 0.0, 0.0]),
        dt_s=0.01,
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
