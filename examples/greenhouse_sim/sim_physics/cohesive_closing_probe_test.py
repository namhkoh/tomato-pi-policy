"""Pure law/USD checks only. No SimulationApp or native simulation launch."""
from dataclasses import replace
import json

import numpy as np
import pytest

from sim_physics.cohesive import CohesiveParameters
from sim_physics.cohesive_native_probe import CouponConfig, measured_jumps
from sim_physics.cohesive_closing_probe import (
    ClosingConfig, ClosingHistory, DURATION_S, ROOT, author, coefficient_snapshot,
    protocol, reconstructed_interface_force, main, activate_for_native, ProtocolAcceptance,
    MOMENTUM_LIMIT_N, ENERGY_LIMIT_J, INTACT_ENERGY_LIMIT_J,
)


@pytest.fixture
def config():
    return ClosingConfig(CouponConfig(CohesiveParameters(1e8, 2e8, 1e4, 2.)), "damaged")


def advance(history, target, count=300):
    start = history.last_jumps.copy()
    for t in np.linspace(0, 1, count+1)[1:]:
        trial = history.trial(start+t*(np.asarray(target)-start))
        history.commit(trial)
    return trial


def uniform(normal):
    return np.tile([normal, 0., 0.], (4, 1))


@pytest.mark.parametrize("case", ["intact", "damaged", "failed"])
@pytest.mark.parametrize("hz", [960, 1920])
def test_continuous_bounded_protocol_all_cases_and_rates(config, case, hz):
    c = replace(config, case=case, coupon=replace(config.coupon, physics_hz=hz))
    times = np.arange(DURATION_S*hz+1)/hz
    commands = np.array([protocol(t, c)[1] for t in times])
    assert commands[0] == 0
    assert np.max(commands) <= .001 and np.min(commands) >= -.001
    assert np.max(np.abs(np.diff(commands)))*hz <= .001
    assert protocol(3.75, c) == ("open_hold", c.open_command_m)
    assert protocol(7.5, c) == ("compression_hold", -c.compression_command_m)
    assert protocol(11., c) == ("reopen_hold", c.reopen_command_m)
    for boundary in (.5, 3.5, 4., 7., 8., 10.):
        assert abs(protocol(boundary+1e-9, c)[1]-protocol(boundary-1e-9, c)[1]) < 1e-11


def test_configuration_and_time_reject_unsupported_cases(config):
    with pytest.raises(TypeError):
        ClosingConfig()
    for c in (replace(config.coupon, physics_hz=480), replace(config.coupon, shear_ratio=.1),
              replace(config.coupon, case="damage_disabled")):
        with pytest.raises(ValueError):
            ClosingConfig(c, "intact")
    with pytest.raises(ValueError):
        replace(config, case="blade")
    for t in (-1., 11.1, np.nan, True):
        with pytest.raises(ValueError):
            protocol(t, config)


def test_partial_damage_close_reopen_retention_and_trial_commit(config):
    history = ClosingHistory(config)
    opened = advance(history, uniform(2.5e-4))
    assert all(0 < d < 1 for d in opened["damage"])
    states = history.states
    trial = history.trial(history.last_jumps+uniform(-1e-6))
    assert history.states is states and trial["responses"][0].state is not states[0]
    closed = advance(history, uniform(-2e-5))
    np.testing.assert_allclose(closed["force_a"], 0., atol=1e-14)
    np.testing.assert_array_equal(closed["damage"], opened["damage"])
    assert np.all(closed["stiffness"] > 0)  # Closing NEVER destroys bond settings.
    reopened = advance(history, uniform(5e-5))
    np.testing.assert_array_equal(reopened["damage"], opened["damage"])
    np.testing.assert_allclose(reopened["force_a"], reopened["stiffness"]*uniform(5e-5))
    assert reopened["dissipated"] == opened["dissipated"]


def test_failure_remains_failed_after_contact_and_reopening(config):
    history = ClosingHistory(replace(config, case="failed"))
    failed = advance(history, uniform(4.5e-4))
    assert all(failed["fully_separated"])
    assert failed["dissipated"] == pytest.approx(8e-6)
    for jump in (-2e-5, 0., 1e-4):
        result = advance(history, uniform(jump))
        assert all(result["fully_separated"])
        np.testing.assert_array_equal(result["stiffness"], np.zeros((4, 3)))
        np.testing.assert_array_equal(result["force_a"], np.zeros((4, 3)))


def test_negative_intact_cycle_and_no_command_state_evidence(config):
    history = ClosingHistory(replace(config, case="intact"))
    for jump in (2.5e-5, -2e-5, 5e-5):
        result = advance(history, uniform(jump))
        np.testing.assert_array_equal(result["damage"], np.zeros(4))
        assert result["dissipated"] == 0
    before = history.states
    protocol(7.5, history.config)
    assert history.states is before
    with pytest.raises(RuntimeError, match="Intact"):
        advance(history, uniform(1.1e-4))


def test_guards_do_not_commit_and_compression_shear_is_not_erased(config):
    history = ClosingHistory(config)
    before = history.states
    for jump in (uniform(.0011), uniform(1e-4)):
        with pytest.raises(RuntimeError):
            history.trial(jump)
        assert history.states is before
    with pytest.raises(ValueError):
        history.trial(np.zeros((3, 3)))
    small = np.tile([-1e-6, 1e-6, 0.], (4, 1))
    trial = history.trial(small)
    assert np.all(trial["force_a"][:, 0] == 0)
    assert np.all(trial["force_a"][:, 1] > 0)
    np.testing.assert_allclose(reconstructed_interface_force(trial["stiffness"], small), trial["force_a"])


@pytest.fixture
def stage():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    s = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(s, 1.)
    return s


def test_actual_helper_integration_dynamic_bodies_remote_support_only(stage, config):
    from pxr import UsdPhysics
    f = author(stage, config)
    assert len(f["links"]) == 4 and "facet_drives" not in f and "bridge" not in f
    for path in f["paths"]:
        prim = stage.GetPrimAtPath(path)
        assert not UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
        assert UsdPhysics.MassAPI(prim).GetDensityAttr().Get() == 1000.
    fixed = [UsdPhysics.Joint(p) for p in stage.Traverse() if p.IsA(UsdPhysics.FixedJoint)]
    assert len(fixed) == 1
    assert not fixed[0].GetBody0Rel().GetTargets()
    assert list(map(str, fixed[0].GetBody1Rel().GetTargets())) == [f["paths"][0]]
    assert fixed[0].GetExcludeFromArticulationAttr().Get()
    np.testing.assert_allclose(measured_jumps(f["frames"], f["local"])[0], 0., atol=1e-18)
    for link in f["links"]:
        assert not link.joint.GetJointEnabledAttr().Get()
        assert link.joint.GetCollisionEnabledAttr().Get()
        assert not link.joint.GetPrim().HasAPI(UsdPhysics.DriveAPI, "transX")
        for axis in ("transY", "transZ"):
            assert link.joint.GetPrim().HasAPI(UsdPhysics.DriveAPI, axis)
    rows, stiffness = coefficient_snapshot(f)
    np.testing.assert_array_equal(stiffness, np.tile([100., 200., 200.], (4, 1)))
    json.dumps(rows, allow_nan=False)


def test_helper_limit_closing_reopening_and_free_on_failure(stage, config):
    f = author(stage, config)
    h = f["history"]
    for opening in (2.5e-4, -2e-5, 5e-5):
        trial = advance(h, uniform(opening))
        for link, response in zip(f["links"], trial["responses"]):
            link.update(response.state)
        rows, k = coefficient_snapshot(f)
        assert all(r["normal_high"] == 0 for r in rows)
        assert np.all(k > 0)
        if opening < 0:
            np.testing.assert_array_equal(reconstructed_interface_force(k, uniform(opening)), np.zeros((4, 3)))
    failed = advance(h, uniform(4.5e-4))
    for link, response in zip(f["links"], failed["responses"]):
        link.update(response.state)
    rows, k = coefficient_snapshot(f)
    assert all(r["normal_high"] is None and r["normal_axis_free"]
               and not r["normal_limit_present"] for r in rows)
    assert all(link.limit is None and not link.soft for link in f["links"])
    np.testing.assert_array_equal(k, np.zeros((4, 3)))
    json.dumps(rows, allow_nan=False)
    assert all(not j.GetJointEnabledAttr().Get() for j in f["joints"])


def test_hard_limit_trap_detected_and_no_native_activation_from_fallback(stage, config):
    f = author(stage, config)
    f["links"][0].soft["stiffness"].Set(0.)
    with pytest.raises(RuntimeError, match="hard"):
        coefficient_snapshot(f)
    assert all(not j.GetJointEnabledAttr().Get() for j in f["joints"])
    f["links"][0].soft["stiffness"].Set(100.)
    if not all(link.report()["native_schema_registered"] for link in f["links"]):
        with pytest.raises(RuntimeError, match="not registered"):
            activate_for_native(f)


def test_cli_requires_review_opt_in_before_output_or_app(tmp_path):
    output = tmp_path/"not_created"
    with pytest.raises(SystemExit):
        main(["--output", str(output), "--case", "intact", "--kn-pa-m", "1e8",
              "--kt-pa-m", "2e8", "--strength-pa", "1e4", "--gc-j-m2", "2"])
    assert not output.exists()


@pytest.mark.parametrize("contact_error", ["", False, RuntimeError(""), None])
def test_any_non_none_contact_error_stops_before_state_commit(stage, config, monkeypatch, contact_error):
    """Exercise the real run callback with fake readbacks; NEVER native stepping."""
    from types import SimpleNamespace
    from sim_physics import cohesive_closing_probe as probe, blade_loading_probe
    from greenhouse_sim import physics_clock

    f = author(stage, config)
    for joint in f["joints"]:
        joint.GetJointEnabledAttr().Set(True)  # In-memory authoring only.
    initial_states = f["history"].states
    dt = 1/config.coupon.physics_hz
    sim = SimpleNamespace(current_time=0., get_physics_dt=lambda: dt)
    body = SimpleNamespace(
        prim_paths=f["paths"], get_masses=lambda: np.full((2, 1), .004),
        get_inertias=lambda: np.tile(np.eye(3)*1e-7, (2, 1, 1)),
        get_transforms=lambda: np.zeros((2, 7)), get_velocities=lambda: np.zeros((2, 6)))
    sensor = SimpleNamespace(sensor_count=1, get_net_contact_forces=lambda step: np.zeros((1, 3)))
    sim.physics_sim_view = SimpleNamespace(
        set_subspace_roots=lambda path: None,
        create_rigid_body_view=lambda path: body,
        create_rigid_contact_view=lambda path, filter_patterns: sensor)
    def fake_frames(values):
        frames = f["frames"].copy()
        if contact_error is None and sim.current_time > 0:
            frames[1, 0, 3] += 1e-6  # Missing/native-mismatched elasticity: residual .7 mN.
        return frames
    monkeypatch.setattr(probe, "pose_matrices", fake_frames)

    class FakeContacts:
        error = contact_error
        subscription = None
        def subscribe(self):
            self.rows, self.friction = [], []

    class FakeClock:
        ticks = 0
        def __init__(self, context, **kwargs):
            pass
        def tick(self, before, after):
            self.ticks += 1
            assert self.ticks == 1, "A contact fault must prevent the next solve"
            before(SimpleNamespace(simulation_time_s=0.), dt)
            sim.current_time = dt
            after(SimpleNamespace(simulation_time_s=dt), dt)
        def report(self):
            return dict(fake_ticks=self.ticks, native_launched=False)

    monkeypatch.setattr(blade_loading_probe, "NativeContacts", FakeContacts)
    monkeypatch.setattr(physics_clock, "PhysicsClock", FakeClock)
    report = probe.run(sim, f)
    assert report["state"] == "closing_coupon_stopped_unqualified"
    if contact_error is None:
        assert "momentum" in report["error"]
        assert report["records"][0]["momentum_residual_with_native_contact_n"] == pytest.approx(.0007)
    else:
        assert report["error"] == str(contact_error)
    assert report["timing"]["fake_ticks"] == 1
    assert len(report["records"]) == 1 and not report["records"][0]["accepted_step"]
    assert f["history"].states == initial_states
    assert f["loading"].GetMaxForceAttr().Get() == 0.
    assert not report["bounded_protocol_accepted"] and report["native_error_observer"] is None


def acceptance_row(config, index):
    """Synthetic contract inputs, NOT a native/mechanical integration test."""
    hz = config.coupon.physics_hz
    phase = protocol((index-1)/hz, config)[0]
    damage = .8 if config.case == "damaged" and index/hz > 3.5 else 0.
    failed = config.case == "failed" and index/hz > 3.5
    damage = 1. if failed else damage
    gap = 50e-6 if phase in ("reopening", "reopen_hold") else 25e-6
    if phase in ("closing", "compression_hold"):
        gap = -1e-8
    k = np.tile(np.array([100., 200., 200.])*(1-damage), (4, 1))
    return dict(t=index/hz, dt=1/hz, phase=phase, damage=[damage]*4,
        fully_separated=[failed]*4, measured_jumps_a_m=uniform(gap).tolist(),
        used_stiffness_n_m=k.tolist(), reconstructed_normal_force_a_n=(k[:, 0]*max(gap, 0.)).tolist(),
        used_unilateral_settings=[dict(normal_low=None if failed else -.002,
                                       normal_high=None if failed else 0.,
                                       normal_limit_present=not failed,
                                       normal_axis_free=failed) for _ in range(4)],
        native_contact_force_b_world_n=[.0075 if gap < 0 else 0., 0., 0.],
        dissipated_energy_j=8e-6 if failed else 4e-6 if damage else 0.,
        momentum_residual_with_native_contact_n=0.,
        cumulative=dict(energy_balance_residual_j=0.))


def contract_at(config, time):
    """Isolate one gate without replaying unrelated synthetic samples."""
    index = round(time*config.coupon.physics_hz)
    check = ProtocolAcceptance(config)
    check.count = index-1
    row = acceptance_row(config, index)
    return check, row


@pytest.mark.parametrize("case", ["intact", "damaged", "failed"])
@pytest.mark.parametrize("hz", [960, 1920])
def test_complete_synthetic_contract_all_cases_and_rates(config, case, hz):
    c = replace(config, case=case, coupon=replace(config.coupon, physics_hz=hz))
    check = ProtocolAcceptance(c)
    for i in range(1, DURATION_S*hz+1):
        check.observe(acceptance_row(c, i))
    check.finish()
    result = check.report()
    assert result["complete"]
    assert all(n == round(.2*hz) for n in result["hold_tail_samples"].values())
    assert result["maximum_abs_accepted_cumulative_energy_residual_j"] == 0
    assert result["momentum_limit_n"] == .00041
    assert result["energy_limit_j"] == .4e-6
    assert result["intact_additional_energy_limit_j"] == .025e-6


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("case", ["intact", "damaged", "failed"])
def test_exact_numerical_limits_inclusive_and_exceedance_latched(config, case, sign):
    c = replace(config, case=case)
    limit = INTACT_ENERGY_LIMIT_J if case == "intact" else ENERGY_LIMIT_J
    check = ProtocolAcceptance(c)
    row = acceptance_row(c, 1)
    row["momentum_residual_with_native_contact_n"] = sign*MOMENTUM_LIMIT_N
    row["cumulative"]["energy_balance_residual_j"] = sign*limit
    check.observe(row)
    assert check.count == 1
    for metric, value in (("momentum", sign*np.nextafter(MOMENTUM_LIMIT_N, np.inf)),
                          ("energy", sign*np.nextafter(limit, np.inf))):
        check = ProtocolAcceptance(c)
        row = acceptance_row(c, 1)
        if metric == "momentum":
            row["momentum_residual_with_native_contact_n"] = value
        else:
            row["cumulative"]["energy_balance_residual_j"] = value
        with pytest.raises(RuntimeError, match=metric):
            check.observe(row)
        assert check.count == 0 and not check.report()["complete"]
        # Returning/cancelling to zero cannot recover a transient failure.
        with pytest.raises(RuntimeError, match=metric):
            check.observe(acceptance_row(c, 1))


@pytest.mark.parametrize("end,target", [(4., 25e-6), (11., 50e-6)])
def test_intact_tail_settling_interval_and_each_facet_tolerance(config, end, target):
    c = replace(config, case="intact")
    check, row = contract_at(c, end-.2)
    row["measured_jumps_a_m"] = uniform(target*1.2).tolist()
    check.observe(row)  # Exclude settling interval, including its endpoint.
    for ratio in (.95, 1.05):
        check, row = contract_at(c, end)
        row["measured_jumps_a_m"] = uniform(target*ratio).tolist()
        check.observe(row)
    for ratio in (.949, 1.051):
        check, row = contract_at(c, end)
        row["measured_jumps_a_m"][3][0] = target*ratio
        with pytest.raises(RuntimeError, match="25/50"):
            check.observe(row)


@pytest.mark.parametrize("change", ["no_damage", "full_damage", "increased", "healed", "dissipation"])
def test_partial_requires_interior_damage_then_preserves_history(config, change):
    check, row = contract_at(config, 4.)
    if change in ("no_damage", "full_damage"):
        row["damage"][3] = 0. if change == "no_damage" else 1.
        with pytest.raises(RuntimeError, match="strictly partial"):
            check.observe(row)
        return
    check.observe(row)
    row = acceptance_row(config, check.count+1)
    if change == "dissipation":
        row["dissipated_energy_j"] += 1e-9
    else:
        row["damage"][0] += .01 if change == "increased" else -.01
    with pytest.raises(RuntimeError, match="preserved|healed"):
        check.observe(row)


@pytest.mark.parametrize("change", ["stiffness", "normal_high", "normal_low", "not_failed"])
def test_failed_hold_requires_all_facets_free_and_next_solve_free(config, change):
    c = replace(config, case="failed")
    check, row = contract_at(c, 4.)
    if change == "stiffness":
        row["used_stiffness_n_m"][2][1] = 1.
    elif change == "not_failed":
        row["fully_separated"][2] = False
    else:
        row["used_unilateral_settings"][2][change] = 0. if change == "normal_high" else -.002
    with pytest.raises(RuntimeError, match="Failed-case"):
        check.observe(row)
    check, row = contract_at(c, 3.6)  # Before hold-tail qualification.
    check.observe(row)
    row = acceptance_row(c, check.count+1)
    row["used_stiffness_n_m"][0][0] = 1.
    with pytest.raises(RuntimeError, match="Failed facet"):
        check.observe(row)


def test_no_compression_cohesive_force_and_required_native_contact(config):
    c = replace(config, case="intact")
    for change in ("force", "no_contact", "tensile_contact"):
        check, row = contract_at(c, 8.)
        if change == "force":
            row["reconstructed_normal_force_a_n"][0] = -1e-12
        else:
            row["native_contact_force_b_world_n"][0] = 0. if change == "no_contact" else -.001
        with pytest.raises(RuntimeError, match="compression"):
            check.observe(row)


def test_missing_samples_tails_nonfinite_and_intact_damage_never_pass(config):
    c = replace(config, case="intact")
    check = ProtocolAcceptance(c)
    with pytest.raises(RuntimeError, match="consecutive"):
        check.observe(acceptance_row(c, 2))
    check = ProtocolAcceptance(c)
    check.count = DURATION_S*c.coupon.physics_hz
    with pytest.raises(RuntimeError, match="Incomplete"):
        check.finish()
    assert not check.report()["complete"]
    for field in ("damage", "dissipated_energy_j", "momentum_residual_with_native_contact_n"):
        check = ProtocolAcceptance(c)
        row = acceptance_row(c, 1)
        row[field] = [np.nan]*4 if field == "damage" else np.nan
        with pytest.raises(RuntimeError, match="Nonfinite"):
            check.observe(row)
    check = ProtocolAcceptance(c)
    row = acceptance_row(c, 1)
    row["damage"] = [.001]*4
    with pytest.raises(RuntimeError, match="zero damage"):
        check.observe(row)


def test_snapshot_uses_helper_declared_finite_bound_and_rejects_tampering(stage, config):
    from sim_physics import cohesive_unilateral as helper
    f = author(stage, config)
    rows, k = coefficient_snapshot(f)
    for link, row in zip(f["links"], rows):
        assert row["normal_low"] == float(np.float32(helper.unilateral_settings(
            link.state, link.area_m2).normal_low_m)) == float(np.float32(-.002))
    json.dumps(rows, allow_nan=False)
    f["links"][0].limit.GetLowAttr().Set(-.001)
    with pytest.raises(RuntimeError, match="Unexpected"):
        coefficient_snapshot(f)


def test_faulted_native_observer_stops_before_any_readback():
    from types import SimpleNamespace
    from sim_physics.cohesive_closing_probe import run
    def fault(phase):
        assert phase == "run_entry"
        raise RuntimeError("")
    with pytest.raises(RuntimeError) as caught:
        run(None, {}, native_errors=SimpleNamespace(check=fault))
    assert str(caught.value) == ""


@pytest.mark.parametrize("residue", ["schema", "property", "tangent"])
def test_failed_snapshot_checks_actual_schema_property_absence(stage, config, residue):
    from pxr import Sdf, UsdPhysics
    f = author(stage, config)
    trial = advance(f["history"], uniform(4.5e-4))
    for link, response in zip(f["links"], trial["responses"]):
        link.update(response.state)
    link = f["links"][0]
    if residue == "schema":
        UsdPhysics.LimitAPI.Apply(link.joint.GetPrim(), "transX")
    elif residue == "property":
        link.joint.GetPrim().CreateAttribute("limit:transX:physics:high", Sdf.ValueTypeNames.Float).Set(0.)
    else:
        link.tangents[0].GetStiffnessAttr().Set(1.)
    with pytest.raises(RuntimeError, match="removed|Unexpected"):
        coefficient_snapshot(f)


@pytest.mark.parametrize("removal_error", [False, True])
def test_cli_observer_cleanup_before_publication_and_injected_monitor_cannot_qualify(tmp_path, monkeypatch, removal_error):
    """Fake app/session only: exercise real CLI publication and observer exit."""
    import sys
    from types import SimpleNamespace
    from sim_physics import cohesive_closing_probe as probe, native_errors
    class Logger:
        def is_log_enabled(self): return True
        def get_level_threshold(self): return 0
        def add_logger(self, callback):
            self.callback = callback
            return object()
        def remove_logger(self, handle):
            if removal_error:
                self.callback("omni.physx.plugin", 1, "", 0, "")
    observer_type = native_errors.NativeErrors
    monkeypatch.setattr(native_errors, "NativeErrors", lambda: observer_type(logging_interface=Logger()))
    repo = tmp_path/"repo"
    monkeypatch.setattr(probe, "__file__", str(repo/"examples/greenhouse_sim/sim_physics/cohesive_closing_probe.py"))
    monkeypatch.setattr(probe.tensile, "_hash", lambda path: "mock_hash_not_native_evidence")
    exits = []
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(
        SimulationApp=lambda settings: SimpleNamespace(close=lambda exit_code: exits.append(exit_code))))
    def fake_session(config, output, observer):
        observer.check("mock_session_not_native")
        return dict(state="mock_only", error=None, bounded_protocol_accepted=True,
                    records=[dict(fake_row=True)])
    monkeypatch.setattr(probe, "_native_session", fake_session)
    output = repo/"data/sim_physics/mock_closing"
    probe.main(["--native-run", "--output", str(output), "--case", "intact",
                "--kn-pa-m", "1e8", "--kt-pa-m", "2e8", "--strength-pa", "1e4", "--gc-j-m2", "2"])
    assert exits == [2] and (output/"trace.jsonl").exists()
    if removal_error:
        assert not (output/"report.json").exists()
        result = json.loads((output/"failure.json").read_text())
        assert result["native_error_observer"]["error_count"] == 1
        assert not result["protocol_result"]["bounded_protocol_accepted"]
    else:
        result = json.loads((output/"report.json").read_text())
        assert result["native_error_observer"]["closed"]
        assert result["native_error_observer"]["logging_interface_injected"]
        assert not (output/"failure.json").exists()
    assert not result["bounded_protocol_accepted"]
