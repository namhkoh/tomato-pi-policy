from __future__ import annotations

from greenhouse_sim import cutting


def test_directional_cut_gate_reset_clears_progress_and_completion():
    gate = cutting.DirectionalCutGate()
    progress = gate.progress_for("Vine_0002/SubStem_02")
    progress.work_j = 0.25
    gate._completed.add("Vine_0002/SubStem_02")

    gate.reset()

    assert gate.summary() == {}
    assert gate._completed == set()
