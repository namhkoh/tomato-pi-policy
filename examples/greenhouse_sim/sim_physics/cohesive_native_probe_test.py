"""Pure adapter/USD checks only; never create an app or step native physics."""
from dataclasses import replace
import numpy as np
import pytest
from sim_physics.cohesive import CohesiveParameters
from sim_physics.cohesive_native_probe import (
    AXES, ROOT, CouponConfig, FacetBridge, author, command_travel,
    measured_jumps, reference_geometry, set_coefficients, main,
)


@pytest.fixture
def config():
    # Explicit UNCALIBRATED toy example: r0=.1mm, rf=.4mm, peak=.04N.
    return CouponConfig(CohesiveParameters(1e8, 2e8, 1e4, 2.0))


def advance(bridge, target, steps=200):
    start = bridge.last_jumps.copy()
    for fraction in np.linspace(0, 1, steps+1)[1:]:
        trial = bridge.trial(start + fraction*(np.asarray(target)-start))
        bridge.commit(trial)
    return trial


def test_no_default_material_or_unbounded_parameters(config):
    with pytest.raises(TypeError):
        CouponConfig()
    for changes in (dict(physics_hz=240), dict(shear_ratio=2), dict(density_kg_m3=0),
                    dict(carriage_max_force_n=.11), dict(maximum_speed_m_s=.02),
                    dict(maximum_damage_increment=.1), dict(area_m2=True),
                    dict(compression_tolerance_m=1e-5), dict(case="cut")):
        with pytest.raises(ValueError):
            replace(config, **changes)


def test_four_anchors_b_minus_a_and_corotation(config):
    frames, local = reference_geometry(config.area_m2)
    jumps, anchors = measured_jumps(frames, local)
    np.testing.assert_allclose(jumps, 0, atol=1e-18)
    np.testing.assert_allclose(anchors[0], anchors[1])
    assert len(np.unique(local[0, :, 1:], axis=0)) == 4
    frames[1, :3, 3] += [3e-5, 2e-5, -1e-5]
    expected = np.tile([3e-5, 2e-5, -1e-5], (4, 1))
    np.testing.assert_allclose(measured_jumps(frames, local)[0], expected, atol=1e-18)
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    frames[:, :3, 3] = frames[:, :3, 3] @ rotation.T + [1., 2., 3.]
    frames[:, :3, :3] = rotation
    np.testing.assert_allclose(measured_jumps(frames, local)[0], expected, atol=1e-15)


def test_area_scaled_positive_secant_and_action_reaction(config):
    bridge = FacetBridge(config)
    trial = advance(bridge, np.tile([2e-4, 0., 0.], (4, 1)))
    expected = config.area_m2/4 * trial["responses"][0].state.retained_stiffness * np.array([1e8, 2e8, 2e8])
    np.testing.assert_allclose(trial["stiffness"], np.tile(expected, (4, 1)))
    assert np.all(trial["stiffness"] > 0)  # Not negative envelope tangent.
    np.testing.assert_array_equal(trial["force_a"]+trial["force_b"], np.zeros((4, 3)))
    np.testing.assert_allclose(trial["force_a"], [r.force_a_n for r in trial["responses"]])


def test_failed_trials_do_not_commit_and_commands_cannot_damage(config):
    bridge = FacetBridge(config)
    trial = bridge.trial(np.tile([1e-6, 0., 0.], (4, 1)))
    assert bridge.states[0].maximum_effective_separation_m == 0
    assert trial["responses"][0].state.maximum_effective_separation_m > 0
    for target, reason in (([1e-3, 0, 0], "increment"), ([-1e-6, 0, 0], "compression"),
                           ([.0011, 0, 0], "1 mm")):
        with pytest.raises(RuntimeError, match=reason):
            bridge.trial(np.tile(target, (4, 1)))
    assert command_travel(4.5, config) > 0
    unchanged = bridge.trial(np.zeros((4, 3)))
    assert unchanged["stored"] == unchanged["dissipated"] == bridge.states[0].damage == 0


def test_irreversible_failure_has_no_residual_stiffness(config):
    bridge = FacetBridge(config)
    broken = advance(bridge, np.tile([4.5e-4, 0., 0.], (4, 1)))
    np.testing.assert_array_equal(broken["applied_damage"], np.ones(4))
    np.testing.assert_array_equal(broken["stiffness"], np.zeros((4, 3)))
    assert broken["dissipated"] == pytest.approx(config.area_m2*config.material.fracture_energy_j_m2)
    unloaded = advance(bridge, np.tile([3e-5, 1e-5, 0.], (4, 1)))
    assert unloaded["stored"] == unloaded["dissipation_increment"] == 0
    np.testing.assert_array_equal(unloaded["force_a"], np.zeros((4, 3)))


def test_disabled_damage_is_explicit_control_not_changed_material(config):
    disabled = FacetBridge(replace(config, case="damage_disabled"))
    trial = advance(disabled, np.tile([3e-4, 0., 0.], (4, 1)))
    np.testing.assert_array_equal(trial["applied_damage"], np.zeros(4))
    assert np.all(trial["shadow_damage"] > 0) and trial["dissipated"] == 0
    assert sum(trial["force_a"][:, 0]) == pytest.approx(config.area_m2*1e8*3e-4)
    assert all(s.parameters is config.material for s in disabled.states)
    sub = FacetBridge(replace(config, case="subcritical"))
    result = advance(sub, np.tile([2e-5, 1e-5, 0.], (4, 1)))
    assert result["dissipated"] == 0
    np.testing.assert_array_equal(result["applied_damage"], np.zeros(4))


@pytest.mark.parametrize("case", ["softening", "subcritical", "damage_disabled"])
def test_command_velocity_bounded_and_return_tensile(config, case):
    c = replace(config, case=case)
    t = np.linspace(0., 10., 10001)
    x = np.array([command_travel(v, c) for v in t])
    assert x[0] == 0 and x[-1] == pytest.approx(.2*c.amplitude_m)
    assert np.max(x) == pytest.approx(c.amplitude_m)
    assert np.min(x) >= 0 and np.max(np.abs(np.diff(x)/np.diff(t))) <= .001


def test_lagged_native_coefficients_differ_from_poststep_law(config):
    bridge = FacetBridge(config)
    before = advance(bridge, np.tile([1.5e-4, 0., 0.], (4, 1)))
    jumps = bridge.last_jumps + [1e-6, 0., 0.]
    after = bridge.trial(jumps)
    assert np.all((before["stiffness"]*jumps)[:, 0] > after["force_a"][:, 0])
    assert bridge.states[0] is before["responses"][0].state


@pytest.fixture
def stage():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    s = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(s, 1.)
    UsdGeom.SetStageUpAxis(s, "Z")
    return s


def test_usd_interface_has_no_parallel_weld_or_locked_dofs(stage, config):
    from pxr import UsdPhysics
    fixture = author(stage, config)
    assert len(fixture["joints"]) == 4
    for joint in fixture["joints"]:
        assert list(map(str, joint.GetBody0Rel().GetTargets())) == [fixture["paths"][0]]
        assert list(map(str, joint.GetBody1Rel().GetTargets())) == [fixture["paths"][1]]
        assert joint.GetCollisionEnabledAttr().Get() is True
        for axis in (*AXES, "rotX", "rotY", "rotZ"):
            assert not joint.GetPrim().HasAPI(UsdPhysics.LimitAPI, axis)
        for axis in AXES:
            d = UsdPhysics.DriveAPI(joint.GetPrim(), axis)
            assert d.GetTypeAttr().Get() == "force"
            assert d.GetDampingAttr().Get() == d.GetTargetPositionAttr().Get() == 0
    assert not any(p.IsA(UsdPhysics.FixedJoint) or p.HasAPI(UsdPhysics.ArticulationRootAPI)
                   for p in stage.Traverse())
    lab = UsdPhysics.PrismaticJoint.Get(stage, ROOT+"/LaboratoryCarriage")
    assert not lab.GetBody0Rel().GetTargets()
    assert not lab.GetLowerLimitAttr().HasAuthoredValueOpinion()
    assert not lab.GetUpperLimitAttr().HasAuthoredValueOpinion()
    set_coefficients(fixture, np.zeros((4, 3)))
    for row in fixture["facet_drives"]:
        for d in row:
            assert d.GetStiffnessAttr().Get() == d.GetDampingAttr().Get() == d.GetMaxForceAttr().Get() == 0
    assert all(j.GetJointEnabledAttr().Get() for j in fixture["joints"])


def test_density_contact_and_exclusive_stage(stage, config):
    from pxr import UsdPhysics
    fixture = author(stage, config)
    for i, path in enumerate(fixture["paths"]):
        body = stage.GetPrimAtPath(path)
        mass = UsdPhysics.MassAPI(body)
        assert mass.GetDensityAttr().Get() == config.density_kg_m3
        assert not mass.GetMassAttr().HasAuthoredValueOpinion()
        assert UsdPhysics.RigidBodyAPI(body).GetKinematicEnabledAttr().Get() == (i == 0)
        assert UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path+"/CompressionCollider")).GetCollisionEnabledAttr().Get()
    with pytest.raises(ValueError, match="New isolated"):
        author(stage, config)
    with pytest.raises(ValueError, match="Isolated stage"):
        author(stage, config, root="/World/Other")


def test_native_launch_requires_main_opt_in(tmp_path):
    with pytest.raises(SystemExit):
        main(["--output", str(tmp_path/"new"), "--kn-pa-m", "1e8", "--kt-pa-m", "2e8",
              "--strength-pa", "1e4", "--gc-j-m2", "2"])
    assert not (tmp_path/"new").exists()
