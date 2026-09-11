"""Pure geometry/reference/USD and mocked readbacks only; no native application."""
from dataclasses import replace
import json
from types import SimpleNamespace
import numpy as np
import pytest

from sim_physics import material_band_native_probe as probe


@pytest.fixture
def config():
    return probe.Config(probe.band_api.BandConfig(source_manifest_path="fixture/manifest.json",
        source_target="fixture/SubStem", radius_m=.003, length_m=.008, density_kg_m3=950.,
        bulk_material=probe.Material(), cohesive_material=probe.CohesiveParameters(1e8, 2e8, 1e4, 2.)), "elastic")


@pytest.fixture
def stage():
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    return stage


def frames(fixture):
    f = np.tile(np.eye(4), (33, 1, 1))
    f[:, :3, 3] = fixture["initial_centres"]
    return f


def test_explicit_fixed_configuration(config):
    with pytest.raises(TypeError):
        probe.Config()
    for config2 in (replace(config.band, radius_m=.0031), replace(config.band, length_m=.01),
                    replace(config.band, axial_layers=8)):
        with pytest.raises(ValueError):
            replace(config, band=config2)
    for hz in (480, True):
        with pytest.raises(ValueError):
            replace(config, physics_hz=hz)


def test_independent_full_network_mass_energy_and_stiffness(config):
    n = probe.network(config)
    assert len(n["records"]) == 216
    assert n["jacobians"].shape == (216, 3, 193)
    assert sum(n["masses"][:32]) == pytest.approx(.00021350684841445066)
    assert n["masses"][32]/n["masses"][0] == pytest.approx(8.)
    K = np.einsum("nai,na,naj->ij", n["jacobians"], n["stiffness"], n["jacobians"])
    load = K@n["reference_q"]
    np.testing.assert_allclose(load[:-1], 0., atol=1e-8)
    assert load[-1] == pytest.approx(.05)
    assert n["reference_q"][-1] == pytest.approx(1.788108138772423e-5)
    energy = .5*np.sum(n["stiffness"]*n["reference_jumps"]**2)
    assert energy == pytest.approx(n["reference_energy"])
    bulk = np.array([r["kind"] == "bulk" for r in n["records"]])
    assert .5*np.sum(n["stiffness"][bulk]*n["reference_jumps"][bulk]**2) == pytest.approx(1.1865349295110658e-9)
    assert all(np.linalg.eigvalsh(I).min() > 0 for I in n["inertias"])


@pytest.mark.parametrize("case", ["rest", "elastic", "preseparated"])
@pytest.mark.parametrize("hz", [960, 1920])
def test_bounded_continuous_protocol(config, case, hz):
    c = replace(config, case=case, physics_hz=hz)
    n = probe.network(c)
    times = np.arange(round(c.duration*hz)+1)/hz
    values = np.array([probe.protocol(t, c, n)[1] for t in times])
    assert np.max(abs(values)) < .001
    assert np.max(abs(np.diff(values)))*hz < .001
    for t in (-1., c.duration+1., np.nan, True):
        with pytest.raises(ValueError):
            probe.protocol(t, c, n)


@pytest.mark.parametrize("case", ["elastic", "preseparated"])
def test_authoring_remote_distributed_boundaries_no_fracture_bypass(stage, config, case):
    from pxr import UsdPhysics
    c = replace(config, case=case)
    f = probe.author(stage, c)
    assert len(f["paths"]) == 33 and len(f["joints"]) == 216 and len(f["links"]) == 24
    rigid = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(rigid) == 33 and all(not UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get() for p in rigid)
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]) == 32
    assert not any(p.IsA(UsdPhysics.FixedJoint) or p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse())
    assert all(not h.joint.GetJointEnabledAttr().Get() for h in f["links"])
    assert len([r for r in f["net"]["records"] if r["kind"] == "platen"]) == 24
    assert {r["a"] for r in f["net"]["records"] if r["kind"] == "platen"} == set(range(24, 32))
    assert all(s["maxLinearVelocity"] == 1. and s["maxDepenetrationVelocity"] == 1.
               for s in f["body_settings"].values())
    assert probe.SPEED_M_S == .01 < probe.NATIVE_LINEAR_CAP_M_S
    pairs = set()
    for i, path in enumerate(f["paths"][:32]):
        api = UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(path))
        if api:
            pairs |= {(i, f["paths"].index(str(p))) for p in api.GetFilteredPairsRel().GetTargets()}
    assert pairs == f["net"]["band"].bulk_pairs and len(pairs) == 48
    setting, k = probe.coefficient_snapshot(f)
    g, _, _ = probe.measured(f, frames(f))
    cohesive = np.array([r["kind"] == "cohesive" for r in f["net"]["records"]])
    np.testing.assert_allclose(g[~cohesive], 0., atol=3e-10)
    np.testing.assert_allclose(g[cohesive, 0], c.initial_gap, atol=3e-10)
    if case == "preseparated":
        assert all(not h.state.bonded and not h.state.fully_separated for h in f["links"])
        assert all(h.limit is None and not h.soft for h in f["links"])
        assert np.all(k == 0) and all(s["normal_axis_free"] for s in setting)


def test_reference_jacobian_matches_authored_anchor_displacement(stage, config):
    from scipy.spatial.transform import Rotation
    f = probe.author(stage, config)
    initial = frames(f)
    base = probe.measured(f, initial)[0]
    q = f["net"]["reference_q"]
    moved = initial.copy()
    for i in range(32):
        moved[i, :3, 3] += q[6*i:6*i+3]
        moved[i, :3, :3] = Rotation.from_rotvec(q[6*i+3:6*i+6]/.003).as_matrix()
    moved[-1, 2, 3] += q[-1]
    observed = probe.measured(f, moved)[0]-base
    np.testing.assert_allclose(observed, f["net"]["reference_jumps"], atol=2e-12, rtol=1e-5)
    response = probe.anchor_response(f, moved, f["net"]["stiffness"], [h.state for h in f["links"]])
    np.testing.assert_allclose(response["force"][:32], 0., atol=.0006)  # USD float32 anchor rest offsets recorded.
    assert response["force"][-1, 2] == pytest.approx(-.05, abs=.0006)
    assert all(s.damage == 0 for s in response["states"])
    moved = initial.copy(); moved[16:, 2, 3] += .0002
    with pytest.raises(RuntimeError, match="damage"):
        probe.anchor_response(f, moved, f["net"]["stiffness"], [h.state for h in f["links"]])


def sample_row(phase="elastic_hold", t=3.5):
    return dict(phase=phase, t=t, maximum_cell_momentum_residual_n=0.,
        global_axial_momentum_residual_n=0., maximum_cell_angular_residual_nm=0.,
        cumulative_energy_residual_j=0., maximum_cell_drift_m=0., full_contact_force_n=0.,
        carriage_force_n=.05, distal_half_contact_axial_n=.025, mean_seam_gap_m=0.)


def test_reference_and_numerical_gates_cannot_pass_without_bulk_strain(stage, config):
    f = probe.author(stage, config)
    trial = dict(jumps=f["net"]["reference_jumps"].copy())
    zero = np.zeros((216, 3))
    assert probe.check_sample(f, sample_row(), trial, zero) == "elastic_hold"
    bulk = np.array([r["kind"] == "bulk" for r in f["net"]["records"]])
    trial["jumps"][bulk] = 0.
    with pytest.raises(RuntimeError, match="bulk anchor strain"):
        probe.check_sample(f, sample_row(), trial, zero)
    for key, value in (("maximum_cell_momentum_residual_n", .000511),
                       ("global_axial_momentum_residual_n", -.000511),
                       ("maximum_cell_angular_residual_nm", .000511*.003),
                       ("cumulative_energy_residual_j", .051*f["net"]["reference_energy"])):
        row = sample_row(); row[key] = value
        with pytest.raises(RuntimeError, match="residual"):
            probe.check_sample(f, row, dict(jumps=zero), zero)


def test_rest_contact_preload_and_separated_hold_gates(stage, config):
    f = probe.author(stage, replace(config, case="rest"))
    row = sample_row("rest", 1.); row["full_contact_force_n"] = .000511
    with pytest.raises(RuntimeError, match="preload"):
        probe.check_sample(f, row, {}, np.zeros((216, 3)))
    f["config"] = replace(config, case="preseparated")
    row = sample_row("compression_hold", 3.5); row["distal_half_contact_axial_n"] = 0.
    with pytest.raises(RuntimeError, match="compression|Compression"):
        probe.check_sample(f, row, {}, np.zeros((216, 3)))
    row = sample_row("separated_hold", 6.5); row["mean_seam_gap_m"] = .0001
    assert probe.check_sample(f, row, {}, np.zeros((216, 3))) == "separated_hold"


def test_full_contacts_keep_nonface_axis_pairs_and_reject_filtered_pairs(stage, config):
    f = probe.author(stage, config)
    # Sectors 0 and 2 share the central axial edge, NOT an adjacent face.
    paths = [f["paths"][i]+"/Solid" for i in (0, 2)]
    record = dict(colliders=paths, position=[0., 0., -.003], impulse=[1e-7, 0., 0.], separation_m=-1e-8)
    cf = np.zeros((32, 3)); cf[0, 0] = 1e-4; cf[2, 0] = -1e-4
    result = probe.contact_wrenches(f, [record], [], cf, frames(f), .001)
    assert result["nonface_records"] == 1 and result["total"] == pytest.approx(.0001)
    assert result["minimum"] == -1e-8 and result["callback_tensor_max_error_n"] < 1e-18
    record["colliders"] = [f["paths"][i]+"/Solid" for i in (0, 1)]
    with pytest.raises(RuntimeError, match="filtered"):
        probe.contact_wrenches(f, [record], [], cf, frames(f), .001)
    record["colliders"][1] = "/World/Unexpected"
    with pytest.raises(RuntimeError, match="Unexpected"):
        probe.contact_wrenches(f, [record], [], cf, frames(f), .001)


def test_fixed_header_sign_never_fits_reversed_tensor_and_friction_is_separate(stage, config):
    f = probe.author(stage, config)
    pair = [f["paths"][i]+"/Solid" for i in (0, 2)]
    normal = dict(colliders=pair, position=[0., 0., -.003], normal=[1., 0., 0.],
                  impulse=[1e-7, 0., 0.], separation_m=-1e-8)
    friction = dict(colliders=pair, position=[0., 0., -.003], impulse=[0., 2e-8, 0.])
    cf = np.zeros((32, 3)); cf[0, 0] = 1e-4; cf[2, 0] = -1e-4
    result = probe.contact_wrenches(f, [normal], [friction], cf, frames(f), .001)
    np.testing.assert_allclose(result["normal"], cf)
    assert result["force"][0, 1] == pytest.approx(2e-5)
    assert result["force"][2, 1] == pytest.approx(-2e-5)
    np.testing.assert_allclose(result["force"], cf+result["friction"])
    # A tolerated tensor discrepancy must never enter the physical wrench/work.
    perturbed_tensor = cf.copy(); perturbed_tensor[0, 2] = 1e-6
    reconciled = probe.contact_wrenches(f, [normal], [friction], perturbed_tensor, frames(f), .001)
    np.testing.assert_allclose(reconciled["force"], result["force"], atol=0., rtol=0.)
    assert reconciled["callback_tensor_max_error_n"] == pytest.approx(1e-6)
    assert result["total"] == pytest.approx(.00012)
    assert "callback_impulse_sign" not in result
    with pytest.raises(RuntimeError, match="Fixed collider0-positive NORMAL"):
        probe.contact_wrenches(f, [normal], [], -cf, frames(f), .001)
    with pytest.raises(RuntimeError, match="NORMAL tensor"):
        probe.contact_wrenches(f, [normal], [friction], result["force"], frames(f), .001)
    # Reversed HEADER order requires reversed impulse, never sorted IDs or fitted signs.
    reversed_normal = dict(normal, colliders=pair[::-1], impulse=[-1e-7, 0., 0.], normal=[-1., 0., 0.])
    reversed_result = probe.contact_wrenches(f, [reversed_normal], [], cf, frames(f), .001)
    np.testing.assert_allclose(reversed_result["normal"], cf)
    with pytest.raises(RuntimeError, match="NORMAL tensor"):
        probe.contact_wrenches(f, [dict(normal, colliders=pair[::-1])], [], cf, frames(f), .001)


def test_fixed_friction_couple_needs_no_force_based_torque_sign_guess(stage, config):
    f = probe.author(stage, config)
    pair = [f["paths"][i]+"/Solid" for i in (0, 2)]
    friction = [dict(colliders=pair, position=[0., 0., -.002], impulse=[1e-7, 0., 0.]),
                dict(colliders=pair, position=[0., 0., -.004], impulse=[-1e-7, 0., 0.])]
    result = probe.contact_wrenches(f, [], friction, np.zeros((32, 3)), frames(f), .001)
    np.testing.assert_allclose(result["force"], 0., atol=1e-18)
    assert result["torque"][0, 1] == pytest.approx(2e-7)
    assert result["torque"][2, 1] == pytest.approx(-2e-7)


def test_native_callback_preserves_header_ids_and_normal_friction_streams(monkeypatch):
    import pxr
    from sim_physics.blade_loading_probe import NativeContacts
    # Exercise the actual parser without native bindings/application.
    monkeypatch.setattr(pxr, "PhysicsSchemaTools", SimpleNamespace(intToSdfPath=lambda i: {9: "/Z", 2: "/A"}[i]), raising=False)
    vector = lambda x, y, z: SimpleNamespace(x=x, y=y, z=z)
    header = SimpleNamespace(collider0=9, collider1=2, contact_data_offset=0,
        num_contact_data=1, friction_anchors_offset=0, num_friction_anchors_data=1)
    data = SimpleNamespace(position=vector(0, 0, 0), normal=vector(1, 0, 0), impulse=vector(3, 0, 0), separation=-1e-8)
    friction = SimpleNamespace(position=vector(0, 0, 0), impulse=vector(0, -2, 0))
    parser = NativeContacts(); parser.callback([header], [data], [friction])
    assert parser.error is None
    assert parser.rows[0]["colliders"] == parser.friction[0]["colliders"] == ["/Z", "/A"]
    assert parser.rows[0]["impulse"] == [3., 0., 0.]
    assert parser.friction[0]["impulse"] == [0., -2., 0.]


def test_speed_watchdog_below_native_cap_and_depenetration_risk_fails_closed(stage, config):
    f = probe.author(stage, config); v = np.zeros((33, 6))
    v[0, 0] = .0101  # Previously silently clamped to .01 by the authored setting.
    row = probe.cap_status(f, [], frames(f), v, .001)
    assert not row["native_linear_cap_may_bind"]
    with pytest.raises(RuntimeError, match="speed watchdog"):
        probe.check_caps_and_speed(row)
    v[0, 0] = 1.
    row = probe.cap_status(f, [], frames(f), v, .001)
    assert row["native_linear_cap_may_bind"]
    with pytest.raises(RuntimeError, match="cap may bind"):
        probe.check_caps_and_speed(row)
    v[:] = 0.
    normal = dict(colliders=[f["paths"][i]+"/Solid" for i in (0, 2)],
        position=[0., 0., -.003], normal=[1., 0., 0.], separation_m=-.001)
    row = probe.cap_status(f, [normal], frames(f), v, .001)
    assert row["native_depenetration_cap_may_bind"]
    assert row["native_cap_activation_observed"] is None  # Never pretend direct solver telemetry.
    with pytest.raises(RuntimeError, match="cap may bind"):
        probe.check_caps_and_speed(row)
    normal["separation_m"] = -1e-8
    probe.check_caps_and_speed(probe.cap_status(f, [normal], frames(f), v, .001))


def test_native_readback_path_no_app_and_empty_contact_fault(stage, config, monkeypatch):
    from sim_physics import blade_loading_probe
    from greenhouse_sim import physics_clock
    f = probe.author(stage, replace(config, case="rest"))
    for joint in f["joints"]:
        joint.GetJointEnabledAttr().Set(True)
    poses = np.zeros((33, 7)); poses[:, :3] = f["initial_centres"]; poses[:, 6] = 1.
    coms = np.zeros((33, 7)); coms[:, 6] = 1.
    def body(indices):
        return SimpleNamespace(prim_paths=[f["paths"][i] for i in indices],
            get_transforms=lambda: poses[indices], get_velocities=lambda: np.zeros((len(indices), 6)),
            get_masses=lambda: f["net"]["masses"][indices, None],
            get_inertias=lambda: f["net"]["inertias"][indices].reshape(len(indices), 9),
            get_coms=lambda: coms[indices])
    cells, carriage = body(list(range(32))), body([32])
    sensor = SimpleNamespace(sensor_paths=f["paths"][:32], get_net_contact_forces=lambda dt: np.zeros((32, 3)))
    view = SimpleNamespace(set_subspace_roots=lambda path: None,
        create_rigid_body_view=lambda path: cells if path.endswith("*") else carriage,
        create_rigid_contact_view=lambda path: sensor)
    sim = SimpleNamespace(physics_sim_view=view, get_physics_dt=lambda: 1/960, current_time=0.)
    class Contacts:
        error = ""
        def subscribe(self): pass
    class Clock:
        def __init__(self, sim, **kwargs): pass
        def tick(self, before, after):
            before(SimpleNamespace(simulation_time_s=0.), 1/960)
            sim.current_time = 1/960
            after(SimpleNamespace(simulation_time_s=1/960), 1/960)
        def report(self): return dict(mock=True)
    observer = SimpleNamespace(check=lambda phase: None, report=lambda: dict(faulted=False, logging_interface_injected=True))
    monkeypatch.setattr(blade_loading_probe, "NativeContacts", Contacts)
    monkeypatch.setattr(physics_clock, "PhysicsClock", Clock)
    result = probe.run(sim, f, native_errors=observer)
    assert result["error"] == "" and not result["bounded_protocol_accepted"]
    assert len(result["records"]) == 1 and not result["records"][0]["accepted"]
    assert result["arrays"]["poses"].shape == (1, 33, 7)
    assert np.isnan(result["arrays"]["anchor_jumps_m"]).all()
    assert f["loading"].GetMaxForceAttr().Get() == 0.


def test_cli_requires_opt_in_before_source_reads_or_app(tmp_path):
    argv = ["--output", str(tmp_path/"not_created"), "--case", "rest",
            "--source-manifest", "missing", "--source-target", "missing/SubStem"]
    for name, value in [("radius-m", ".003"), ("length-m", ".008"), ("density-kg-m3", "950"),
        ("youngs-pa", "150000000"), ("poisson", ".3"), ("kn-pa-m", "1e8"),
        ("kt-pa-m", "2e8"), ("strength-pa", "1e4"), ("gc-j-m2", "2")]:
        argv += ["--"+name, value]
    with pytest.raises(SystemExit):
        probe.main(argv)
    assert not (tmp_path/"not_created").exists()


@pytest.mark.parametrize("field", ["maximum_cell_momentum_residual_n",
                                  "cumulative_energy_residual_j", "full_contact_force_n"])
def test_nonfinite_acceptance_fails_closed(stage, config, field):
    f = probe.author(stage, config)
    row = sample_row(); row[field] = float("nan")
    with pytest.raises(RuntimeError, match="Nonfinite"):
        probe.check_sample(f, row, {}, np.zeros((216, 3)))


def test_source_receipt_exact_target_and_no_source_edits(tmp_path, config):
    folder = tmp_path/"seed101_full"; folder.mkdir()
    asset = folder/"SubStem_41.usdc"; asset.write_bytes(b"hash fixture, not a USD/native source")
    manifest = folder/"manifest.json"
    manifest.write_text(json.dumps(dict(units="meters", components=[
        dict(id="SubStem_41", type="sub_stem", file=asset.name)])), encoding="utf-8")
    c = replace(config, band=replace(config.band, source_manifest_path=str(manifest),
                                    source_target="seed101_full/SubStem_41"))
    before = (manifest.read_bytes(), asset.read_bytes())
    result = probe.source_receipt(c)
    assert set(result) == {str(manifest.resolve()), str(asset.resolve())}
    assert all(len(h) == 64 for h in result.values())
    assert before == (manifest.read_bytes(), asset.read_bytes())
    with pytest.raises(ValueError, match="identity"):
        probe.source_receipt(replace(c, band=replace(c.band, source_target="wrong/SubStem_41")))
