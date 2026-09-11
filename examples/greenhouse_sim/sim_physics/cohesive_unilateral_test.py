"""Pure-law and in-memory USD checks; no Kit/SimulationApp or native solver."""
from dataclasses import replace
import math

import numpy as np
import pytest

from sim_physics.cohesive import CohesiveParameters, CohesiveState, cohesive_response
from sim_physics.cohesive_unilateral import (
    MAXIMUM_ANCHOR_JUMP_M, configure_band_interfaces, configure_unilateral, unilateral_settings)


@pytest.fixture
def material():
    return CohesiveParameters(1e8, 2e8, 1e4, 2.)


def response(state, jump, area=1e-6):
    return cohesive_response(state, jump, [1., 0., 0.], area_m2=area)


def test_area_weighted_force_units_and_positive_opening_only(material):
    state = CohesiveState(material); settings = unilateral_settings(state, 1e-6)
    assert settings.normal_stiffness_n_m == 100.
    assert settings.tangent_stiffness_n_m == 200.
    assert settings.normal_low_m == -.002 and settings.normal_high_m == 0.
    for jump in ([.00005, 0., 0.], [-.00005, 0., 0.], [0., 0., 0.]):
        r = response(state, jump)
        assert r.state.damage == 0.
        assert r.force_a_n[0] == pytest.approx(settings.normal_stiffness_n_m * max(jump[0], 0.))
        assert r.force_b_n[0] == -r.force_a_n[0]


def test_compression_keeps_retained_limit_and_does_not_heal_or_damage(material):
    damaged = response(CohesiveState(material), [.0002, 0., 0.]).state
    before = unilateral_settings(damaged, 1e-6)
    compressed = response(damaged, [-.0003, 0., 0.])
    assert compressed.state.damage == damaged.damage
    assert unilateral_settings(compressed.state, 1e-6) == before
    assert compressed.force_a_n == (0., 0., 0.) and compressed.dissipation_increment_j == 0.
    reopened = response(compressed.state, [.0001, 0., 0.])
    assert reopened.force_a_n[0] == pytest.approx(before.normal_stiffness_n_m * .0001)
    assert reopened.state.damage == damaged.damage


def test_actual_shear_can_damage_under_compression_without_normal_penalty(material):
    r = response(CohesiveState(material), [-.0001, .00015, 0.])
    assert r.state.damage > 0 and r.force_a_n[0] == 0.
    settings = unilateral_settings(r.state, 1e-6)
    assert r.force_a_n[1] == pytest.approx(settings.tangent_stiffness_n_m * .00015)


@pytest.mark.parametrize('bonded', [True, False])
def test_exact_failure_or_initially_absent_bond_unbounds_normal(material, bonded):
    state = CohesiveState(material, bonded=bonded)
    if bonded:
        state = response(state, [material.final_separation_m, 0., 0.]).state
    settings = unilateral_settings(state, 1e-6)
    assert settings.normal_high_m == math.inf
    assert settings.normal_stiffness_n_m == settings.tangent_stiffness_n_m == 0.
    for jump in ([-.0001, 0., 0.], [.0001, .0001, 0.]):
        r = response(state, jump)
        assert r.force_a_n == (0., 0., 0.)


def test_last_representable_pre_failure_state_not_rounded_to_hard_or_free(material):
    k = np.nextafter(material.final_separation_m, 0.)
    state = CohesiveState(material, maximum_effective_separation_m=k, effective_separation_m=k)
    settings = unilateral_settings(state, 1e-6)
    assert not state.fully_separated and settings.normal_stiffness_n_m > 0
    assert settings.normal_high_m == 0.


@pytest.mark.parametrize('area', [0., -1., True, np.bool_(True), np.nan, math.inf, '1', 1+2j, np.complex64(1+2j), 1e-320, 1e38])
def test_invalid_or_native_unrepresentable_stiffness_fails_without_clamping(material, area):
    with pytest.raises(ValueError): unilateral_settings(CohesiveState(material), area)


def test_subdivision_and_loading_history_independence(material):
    state = CohesiveState(material)
    for value in np.linspace(0., .00023, 100):
        state = response(state, [value, 0., 0.]).state
    direct = response(CohesiveState(material), [.00023, 0., 0.]).state
    assert state == direct
    total = unilateral_settings(state, 8e-6)
    small = unilateral_settings(state, 1e-6)
    assert total.normal_stiffness_n_m == pytest.approx(8 * small.normal_stiffness_n_m)
    assert total.tangent_stiffness_n_m == pytest.approx(8 * small.tangent_stiffness_n_m)


@pytest.mark.parametrize('weight', [.5, 1., 2.])
def test_state_envelope_respects_shear_metric_and_checks_history(material, weight):
    material = replace(material, shear_stiffness_pa_m=material.normal_stiffness_pa_m * weight**2)
    bound = MAXIMUM_ANCHOR_JUMP_M * max(1., weight)
    for effective in (bound, np.nextafter(bound, 0.)):
        unilateral_settings(CohesiveState(material, maximum_effective_separation_m=effective), 1e-6)
    with pytest.raises(ValueError, match='envelope'):
        unilateral_settings(CohesiveState(material, maximum_effective_separation_m=np.nextafter(bound, math.inf)), 1e-6)


def test_effective_state_cannot_certify_actual_compression_guard(material):
    virgin = CohesiveState(material)
    compressed = response(virgin, [-.003, 0., 0.]).state
    assert compressed == virgin  # Intentionally out of domain, invisible to the law state.
    assert unilateral_settings(compressed, 1e-6) == unilateral_settings(virgin, 1e-6)
    # Therefore runtime MUST reject this measured vector before committing it.
    assert np.linalg.norm([-.003, 0., 0.]) > MAXIMUM_ANCHOR_JUMP_M


def test_finite_lower_spring_inactive_throughout_declared_compression_envelope(material):
    s = unilateral_settings(CohesiveState(material), 1e-6)
    low, high = np.float32(s.normal_low_m), np.float32(s.normal_high_m)
    # PxJointLinearLimitPair::isValid finite-bounds/width condition (SDK 5.6.1).
    assert np.isfinite([low, high, high-low]).all() and low < high
    for x in np.linspace(-MAXIMUM_ANCHOR_JUMP_M, 0., 50):
        assert max(float(low)-x, 0.) == max(x-float(high), 0.) == 0.
    assert float(low) < -MAXIMUM_ANCHOR_JUMP_M


@pytest.fixture
def authored(material):
    pytest.importorskip('pxr')
    from pxr import Usd, UsdGeom
    from sim_physics.material_band import BandConfig, author, build
    from sim_physics.mechanics import Material
    config = BandConfig(source_manifest_path='fixture/manifest.json', source_target='fixture/SubStem',
        radius_m=.003, length_m=.008, density_kg_m3=950., bulk_material=Material(), cohesive_material=material)
    stage = Usd.Stage.CreateInMemory(); UsdGeom.SetStageMetersPerUnit(stage, 1.)
    return stage, author(stage, build(config), root='/World/Band', proximal_support=True)


def fracture(authored):
    stage, fixture = authored
    record = next(r for r in fixture['interfaces'] if r['face'].fracture)
    return stage, record, record['anchors'][0]['joint']


def link(authored):
    stage, record, joint = fracture(authored)
    return configure_unilateral(joint, state=record['states'][0], area_m2=record['face'].weights_m2[0])


def prim_snapshot(prim):
    return (str(prim.GetMetadata('apiSchemas')),
        tuple((a.GetName(), str(a.Get())) for a in prim.GetAttributes()),
        tuple((r.GetName(), tuple(map(str, r.GetTargets()))) for r in prim.GetRelationships()))


def test_full_band_only_normal_replaced_geometry_mass_bulk_and_support_unchanged(authored):
    from pxr import Usd, UsdPhysics
    stage, fixture = authored
    protected = {str(p.GetPath()): prim_snapshot(p) for path in fixture['paths'] for p in Usd.PrimRange(stage.GetPrimAtPath(path))}
    protected.update({str(a['joint'].GetPath()): prim_snapshot(a['joint'].GetPrim())
        for r in fixture['interfaces'] if not r['face'].fracture for a in r['anchors']})
    protected.update({str(a['joint'].GetPath()): prim_snapshot(a['joint'].GetPrim()) for a in fixture['proximal_support']})
    handles = configure_band_interfaces(fixture)
    assert len(handles) == 24
    for _, _, h in handles:
        j = h.joint; p = j.GetPrim(); s = unilateral_settings(h.state, h.area_m2)
        assert not j.GetJointEnabledAttr().Get()
        assert j.GetExcludeFromArticulationAttr().Get() and j.GetCollisionEnabledAttr().Get()
        assert not p.HasAPI(UsdPhysics.DriveAPI, 'transX')
        assert not any(n.startswith('drive:transX:') for n in p.GetPropertyNames())
        assert h.limit.GetLowAttr().Get() == float(np.float32(-.002)) and h.limit.GetHighAttr().Get() == 0.
        assert h.soft['stiffness'].Get() == pytest.approx(s.normal_stiffness_n_m)
        assert all(h.soft[n].Get() == 0. for n in ('damping', 'restitution', 'bounceThreshold'))
        assert 'PhysxLimitAPI:transX' in p.GetMetadata('apiSchemas').GetAppliedItems()
        for d in h.tangents:
            assert d.GetStiffnessAttr().Get() == pytest.approx(s.tangent_stiffness_n_m)
            assert d.GetDampingAttr().Get() == d.GetTargetPositionAttr().Get() == d.GetTargetVelocityAttr().Get() == 0.
            assert d.GetMaxForceAttr().Get() == math.inf and d.GetTypeAttr().Get() == 'force'
    assert all(prim_snapshot(stage.GetPrimAtPath(path)) == before for path, before in protected.items())
    assert not any(p.IsA(UsdPhysics.FixedJoint) or p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse())
    assert not fixture['simulation_ready']


def test_postfetch_retained_updates_and_full_failure_never_leave_hard_limit(authored):
    h = link(authored); h.joint.GetJointEnabledAttr().Set(True)  # Caller activation, no stepping.
    p = h.state.parameters
    for jump in ([.0002, 0., 0.], [-.0002, 0., 0.], [.0001, 0., 0.], [p.final_separation_m, 0., 0.], [-.0001, 0., 0.]):
        state = response(h.state, jump, h.area_m2).state
        report = h.update(state)
        assert h.joint.GetJointEnabledAttr().Get()
        if state.retained_stiffness > 0:
            assert h.soft['damping'].Get() == 0.
            assert h.limit.GetLowAttr().Get() == float(np.float32(-.002))
            assert h.limit.GetHighAttr().Get() == 0.
            assert h.soft['stiffness'].Get() == pytest.approx(h.area_m2 * state.retained_stiffness * p.normal_stiffness_pa_m)
        else:
            assert_removed(h)
            assert report['normal_limit_removed'] and report['normal_limit_unbounded']
            assert report['normal_low_m'] is None and report['normal_high_m'] is None
        assert not report['native_response_validated']
        assert not report['actual_jump_envelope_checked'] and report['actual_jump_guard_owned_by_caller']
        assert report['maximum_anchor_jump_m'] == .001
    assert all(d.GetStiffnessAttr().Get() == 0. for d in h.tangents)


def assert_removed(h):
    from pxr import UsdPhysics
    prim = h.joint.GetPrim()
    assert h.limit is None and not h.soft
    assert not prim.HasAPI(UsdPhysics.LimitAPI, 'transX')
    assert not any(n in prim.GetMetadata('apiSchemas').GetAppliedItems()
                   for n in ('PhysicsLimitAPI:transX', 'PhysxLimitAPI:transX'))
    assert not any(n.startswith(('limit:transX:', 'physxLimit:transX:', 'drive:transX:'))
                   for n in prim.GetPropertyNames())
    assert all(d.GetStiffnessAttr().Get() == d.GetDampingAttr().Get() == 0. for d in h.tangents)


def test_failure_is_one_atomic_usd_notice_without_transient_hard_limit(authored):
    from pxr import Tf, Usd, UsdPhysics
    h = link(authored); snapshots = []; prim = h.joint.GetPrim()
    def changed(notice, stage):
        snapshots.append((prim.HasAPI(UsdPhysics.LimitAPI, 'transX'),
            tuple(n for n in prim.GetPropertyNames() if n.startswith(('limit:transX:', 'physxLimit:transX:'))),
            tuple(d.GetStiffnessAttr().Get() for d in h.tangents)))
    token = Tf.Notice.Register(Usd.Notice.ObjectsChanged, changed, prim.GetStage())
    failed = response(h.state, [h.state.parameters.final_separation_m, 0., 0.]).state
    try:
        h.update(failed)
    finally:
        token.Revoke()
    assert snapshots == [(False, (), (0., 0.))]
    assert_removed(h)


def test_pure_usd_parser_finite_active_limit_then_no_failed_limit(authored):
    from pxr import Sdf, UsdPhysics
    parse = getattr(UsdPhysics, 'LoadUsdPhysicsFromRange', None)
    if parse is None: pytest.skip('This USD build does not expose its read-only physics parser')
    h = link(authored)
    def descriptor():
        paths, rows = parse(h.joint.GetPrim().GetStage(), [Sdf.Path.absoluteRootPath])[UsdPhysics.ObjectType.D6Joint]
        return rows[list(paths).index(h.joint.GetPath())]
    limits = list(descriptor().jointLimits)
    assert len(limits) == 1 and limits[0].first == UsdPhysics.JointDOF.TransX
    assert limits[0].second.enabled
    assert limits[0].second.lower == float(np.float32(-.002)) and limits[0].second.upper == 0.
    h.update(response(h.state, [h.state.parameters.final_separation_m, 0., 0.]).state)
    assert list(descriptor().jointLimits) == []
    assert_removed(h)


@pytest.mark.parametrize('bonded', [True, False])
def test_initial_zero_retention_removes_limit_without_enabling_joint(authored, bonded):
    stage, record, joint = fracture(authored)
    state = CohesiveState(record['states'][0].parameters, bonded=bonded)
    if bonded: state = response(state, [state.parameters.final_separation_m, 0., 0.]).state
    h = configure_unilateral(joint, state=state, area_m2=record['face'].weights_m2[0])
    assert_removed(h)
    assert not joint.GetJointEnabledAttr().Get()
    h.update(response(state, [-.0001, 0., 0.]).state)
    assert_removed(h)


@pytest.mark.parametrize('change', ['limit_api', 'soft_api', 'property', 'drive'])
def test_failed_interface_cannot_resurrect_constraints(authored, change):
    from pxr import Sdf, UsdPhysics
    h = link(authored); prim = h.joint.GetPrim()
    failed = response(h.state, [h.state.parameters.final_separation_m, 0., 0.]).state
    h.update(failed)
    if change == 'limit_api': UsdPhysics.LimitAPI.Apply(prim, 'transX')
    if change == 'soft_api': prim.AddAppliedSchema('PhysxLimitAPI:transX')
    if change == 'property': prim.CreateAttribute('limit:transX:physics:high', Sdf.ValueTypeNames.Float).Set(0.)
    if change == 'drive': UsdPhysics.DriveAPI.Apply(prim, 'transX')
    before = prim.GetStage().GetRootLayer().ExportToString()
    with pytest.raises(ValueError): h.update(failed)
    assert prim.GetStage().GetRootLayer().ExportToString() == before


@pytest.mark.parametrize('change', ['enabled', 'articulation', 'no_contact', 'target', 'damping', 'wrong_stiffness', 'normal_limit', 'angular_drive', 'wrong_anchor', 'wrong_normal', 'kinematic', 'centimetres', 'break_force'])
def test_bad_initial_contract_rejected_without_writes(authored, change):
    from pxr import Gf, UsdGeom, UsdPhysics
    stage, record, joint = fracture(authored); prim = joint.GetPrim()
    if change == 'enabled': joint.GetJointEnabledAttr().Set(True)
    if change == 'articulation': joint.GetExcludeFromArticulationAttr().Set(False)
    if change == 'no_contact': joint.GetCollisionEnabledAttr().Set(False)
    if change == 'target': UsdPhysics.DriveAPI(prim, 'transY').GetTargetPositionAttr().Set(.001)
    if change == 'damping': UsdPhysics.DriveAPI(prim, 'transX').GetDampingAttr().Set(1.)
    if change == 'wrong_stiffness': UsdPhysics.DriveAPI(prim, 'transX').GetStiffnessAttr().Set(1.)
    if change == 'normal_limit': UsdPhysics.LimitAPI.Apply(prim, 'transX')
    if change == 'angular_drive': UsdPhysics.DriveAPI.Apply(prim, 'rotX')
    if change == 'wrong_anchor': joint.GetLocalPos0Attr().Set(Gf.Vec3f(.1, 0., 0.))
    if change == 'wrong_normal': joint.GetLocalRot0Attr().Set(Gf.Quatf(0., Gf.Vec3f(0., 1., 0.)))
    if change == 'kinematic': UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(joint.GetBody0Rel().GetTargets()[0])).GetKinematicEnabledAttr().Set(True)
    if change == 'centimetres': UsdGeom.SetStageMetersPerUnit(stage, .01)
    if change == 'break_force': joint.GetBreakForceAttr().Set(.5)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError): configure_unilateral(joint, state=record['states'][0], area_m2=record['face'].weights_m2[0])
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize('change', ['healing', 'material', 'bond', 'normal_drive', 'stiffness', 'hard_limit', 'frame', 'damping', 'lower_bound', 'infinite_bound', 'envelope'])
def test_live_tampering_and_healing_rejected_without_adapter_writes(authored, change):
    from pxr import Gf, UsdPhysics
    h = link(authored); state = response(h.state, [.0002, 0., 0.], h.area_m2).state; h.update(state)
    new = state
    if change == 'healing': new = CohesiveState(state.parameters)
    if change == 'material': new = CohesiveState(replace(state.parameters, fracture_energy_j_m2=3.))
    if change == 'bond': new = CohesiveState(state.parameters, bonded=False)
    if change == 'normal_drive': UsdPhysics.DriveAPI.Apply(h.joint.GetPrim(), 'transX')
    if change == 'stiffness': h.tangents[0].GetStiffnessAttr().Set(1000.)
    if change == 'hard_limit': h.soft['stiffness'].Set(0.)
    if change == 'frame': h.joint.GetLocalPos0Attr().Set(Gf.Vec3f(.1))
    if change == 'damping': h.soft['damping'].Set(1.)
    if change == 'lower_bound': h.limit.GetLowAttr().Set(-.0001)
    if change == 'infinite_bound': h.limit.GetLowAttr().Set(-math.inf)
    if change == 'envelope': new = CohesiveState(state.parameters, maximum_effective_separation_m=.003)
    stage = h.joint.GetPrim().GetStage(); before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError): h.update(new)
    assert h.state == state and stage.GetRootLayer().ExportToString() == before


def test_band_preflights_last_bad_anchor_before_first_write(authored):
    stage, fixture = authored
    last = [r for r in fixture['interfaces'] if r['face'].fracture][-1]
    last['anchors'][-1]['joint'].GetExcludeFromArticulationAttr().Set(False)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError): configure_band_interfaces(fixture)
    assert stage.GetRootLayer().ExportToString() == before


def test_coincident_rotated_geometry_and_normal_sign_contract(authored):
    from pxr import Gf, UsdGeom
    stage, fixture = authored
    transform = np.array([[0., -1., 0., .2], [1., 0., 0., .1], [0., 0., 1., .3], [0., 0., 0., 1.]])
    for cell, path in zip(fixture['band'].cells, fixture['paths']):
        body = UsdGeom.Xformable(stage.GetPrimAtPath(path))
        ops = body.GetOrderedXformOps()
        ops[0].Set(Gf.Vec3d(*(transform[:3, :3] @ cell.centre + transform[:3, 3])))
        ops[1].Set(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0., 0., 1.), 90.).GetQuat()))
    assert len(configure_band_interfaces(fixture)) == 24


def test_coincident_but_misplaced_quadrature_anchor_rejected_before_writes(authored):
    from pxr import Gf
    stage, fixture = authored
    joint = fracture(authored)[2]
    for side in (0, 1):
        attr = getattr(joint, f'GetLocalPos{side}Attr')()
        attr.Set(attr.Get() + Gf.Vec3f(.0001, 0., 0.))
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='quadrature'):
        configure_band_interfaces(fixture)
    assert stage.GetRootLayer().ExportToString() == before


def test_band_inventory_omission_rejected_without_writes(authored):
    stage, fixture = authored
    fixture['interfaces'] = fixture['interfaces'][:-1]
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='inventory'):
        configure_band_interfaces(fixture)
    assert stage.GetRootLayer().ExportToString() == before
