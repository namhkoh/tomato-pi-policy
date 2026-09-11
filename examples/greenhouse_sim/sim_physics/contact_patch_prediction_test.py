"""Pure tests for an explicitly approximate patch law; no native runtime."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sim_physics.contact_patch_prediction import (
    MODEL, MAX_ITERATIONS, compile_patches, solve,
)
from sim_physics.contact_coupled_prediction import (
    MaterialLaw, solve as normal_solve, _features, _kinematic_jacobian,
)
from sim_physics.contact_spring_probe import Coupon, layout, _ry

LAW = MaterialLaw('unilateral_kv_v1')


def coupon(**kwargs):
    return Coupon('synthetic-not-native', 'a'*64, (.33289975, .30481708),
        (.00940549, .00807237), (.000633, .000607, .000580),
        tuple(np.diag([3e-8, 3e-8, 2.5e-9]) for _ in range(3)),
        1000., 1.0573968428135787, .5, **kwargs)


def geometry(c=None):
    c = coupon() if c is None else c
    frames = layout(c)['frames']
    rows = []
    for x in _features(c, frames):
        base = dict(collider0=x['collider'], collider1=x['pad'],
                    point_world_m=x['point'].tolist())
        rows.append(dict(base, kind='normal', normal_on_0=x['normal'].tolist(),
                         separation_m=x['gap']))
        # Deliberate sub-tolerance offset: preserve exact friction anchor,
        # do NOT replace it with the nearest normal point.
        offset = layout(c)['pads'][x['pad']][:3, 1]*.3e-6
        rows.append(dict(base, kind='friction', point_world_m=(x['point']+offset).tolist()))
    jac = _kinematic_jacobian(frames)
    return dict(coupon=c, frames=frames, velocities=np.zeros((3, 6)), nativeJac=jac,
        native_rows=rows, generalized_velocity=np.zeros(8), step_id=1,
        native_geometry_step_id=1)


def system(patches=1):
    p = patches; m = 2*p
    T = np.zeros((p, 2, 2, 8))
    T[:, :, 0, 1] = 1.
    T[:, :, 1, 2] = 1.
    J = np.zeros((m, 8)); J[:, 0] = 1.
    return dict(M=np.eye(8), q=np.zeros(8), v=np.zeros(8),
        K=np.r_[np.zeros(6), [3., 2.]], C=np.r_[np.zeros(6), [.1, .2]],
        J=J, g=np.full(m, -.01), s=np.zeros(m),
        kc=np.full(m, 100.), dc=np.ones(m), h=.01,
        f=np.r_[-float(m), np.zeros(7)], law=LAW,
        feature_observed=[True]*m, patches=dict(model=MODEL,
            normal_indices=np.arange(m).reshape(p, 2).tolist(),
            tangent_jacobians=T.tolist(), surface_speeds_m_s=np.zeros((p, 2, 2)).tolist(),
            mu=.5, observed=[True]*p))


def test_isolated_import_no_native():
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    r = subprocess.run([sys.executable, '-B', '-c',
        'import sys; import sim_physics.contact_patch_prediction; '
        'assert not any(n.startswith(("pxr","omni","isaacsim")) for n in sys.modules)'],
        env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_free_exactly_existing_solver_without_contact_or_root_actuation():
    a = system(0)
    a['q'][6:] = [.004, -.002]; a['v'] = np.arange(8)*.001
    a['M'][0, 6] = a['M'][6, 0] = .2
    expected = normal_solve(**{k: v for k, v in a.items() if k != 'patches'})
    result = solve(**a)
    np.testing.assert_array_equal(result['tau_spring'], expected['tau_spring'])
    np.testing.assert_array_equal(result['v_pred'], expected['v_pred'])
    assert result['tau_spring'][:6] == [0.]*6
    assert result['friction_forces_n'] == []
    assert not result['measured_friction_used']


def test_static_two_anchor_stick_shares_total_patch_support_not_per_normal_load():
    a = system()
    # Unequal normal-feature forces .2 and 1.8 N; each anchor still gets
    # mu * (N0+N1)/2 = .5 N, rather than .1 and .9 N.
    a['g'] = np.array([-.002, -.018])
    a['f'][1:3] = [.6, .2]
    result = solve(**a)
    assert result['status'] == 'resolved'
    np.testing.assert_allclose(result['v_pred'], 0, atol=2e-16)
    np.testing.assert_allclose(result['friction_forces_n'], [[[-.3, -.1], [-.3, -.1]]], atol=1e-13)
    np.testing.assert_allclose(result['normal_forces_n'], [.2, 1.8], atol=1e-14)
    np.testing.assert_allclose(result['anchor_limits_n'], [[.5, .5]])
    assert result['friction_modes'] == [['stick', 'stick']]
    assert np.all(np.asarray(result['anchor_relative_power_w'])
                  <= np.asarray(result['anchor_power_error_bound_w']))
    assert np.max(result['anchor_power_error_bound_w']) < 1e-12
    assert not result['native_contact_law_parity'] and not result['native_qualified']


def test_static_joint_elastic_load_not_suppressed_with_all_free_root_coordinates():
    a = system()
    a['q'][6] = .1
    T = np.array(a['patches']['tangent_jacobians'])
    T[:, :, 0, 6] = .5
    a['patches']['tangent_jacobians'] = T.tolist()
    # +.6 N support in root-Y and +.3 Nm at joint balances -Kq.
    a['f'][1] = -.6
    result = solve(**a)
    assert result['status'] == 'resolved'
    np.testing.assert_allclose(result['v_pred'], 0, atol=2e-15)
    assert result['tau_joint'][0] == pytest.approx(-.3, abs=1e-14)
    np.testing.assert_allclose(result['friction_forces_n'], [[[.3, 0], [.3, 0]]], atol=1e-13)
    assert result['tau_spring'][:6] == [0.]*6


@pytest.mark.parametrize('direction', [[1., 0.], [-1., 0.], [0., 1.], [.6, .8]])
def test_slip_exact_circular_cone_and_opposes_motion(direction):
    a = system()
    direction = np.array(direction)
    a['f'][1:3] = 2*direction
    result = solve(**a)
    assert result['status'] == 'resolved'
    np.testing.assert_allclose(np.array(result['v_pred'])[1:3], .01*direction, atol=1e-11)
    force = np.array(result['friction_forces_n'])[0]
    np.testing.assert_allclose(force, np.tile(-.5*direction, (2, 1)), atol=1e-10)
    assert result['friction_modes'] == [['slip', 'slip']]
    assert np.max(result['anchor_relative_power_w']) <= 0
    assert result['cone_max_violation_n'] < 1e-10
    assert result['friction_fixed_point_residual_n'] < 1e-9


def test_two_anchor_moment_couple_retained_not_midpoint_lumped():
    a = system()
    T = np.array(a['patches']['tangent_jacobians'])
    T[0, 0, 0, 4] = -.008
    T[0, 1, 0, 4] = .008
    a['patches']['tangent_jacobians'] = T.tolist()
    a['f'][4] = .004
    result = solve(**a)
    assert result['status'] == 'resolved'
    np.testing.assert_allclose(result['v_pred'], 0, atol=1e-10)
    np.testing.assert_allclose(np.array(result['friction_forces_n'])[0, :, 0], [.25, -.25], atol=1e-9)


def test_no_contact_no_friction_no_retained_tangent_connection():
    a = system()
    a['g'][:] = .1; a['v'][0] = 1.; a['f'][0] = 0.; a['v'][1] = .5
    result = solve(**a)
    assert result['status'] == 'resolved'
    assert result['normal_forces_n'] == [0., 0.]
    np.testing.assert_allclose(result['friction_forces_n'], 0, atol=1e-14)
    assert result['v_pred'][1] == pytest.approx(.5)
    assert result['friction_modes'] == [['unsupported', 'unsupported']]


def test_zero_mu_is_normal_only_not_a_hidden_tangent_pin():
    a = system(); a['patches']['mu'] = 0.; a['v'][1] = 1.
    result = solve(**a)
    expected = normal_solve(**{k: v for k, v in a.items() if k != 'patches'})
    np.testing.assert_allclose(result['v_pred'], expected['v_pred'], atol=1e-14)
    np.testing.assert_allclose(result['tau_spring'], expected['tau_spring'], atol=1e-14)
    np.testing.assert_allclose(result['friction_forces_n'], 0, atol=1e-14)


def test_one_iteration_exhaustion_has_no_partial_effort():
    a = system(); a['f'][1] = 2.  # all-stick candidate violates the cone
    r = solve(**a, max_iterations=1)
    assert r['status'] == 'unresolved'
    assert r['reason'] == 'iteration_limit'
    assert r['tau_spring'] is r['tau_joint'] is r['v_pred'] is None


@pytest.mark.parametrize('what', ['normal', 'patch'])
def test_unobserved_support_fail_closed(what):
    a = system()
    if what == 'normal':
        a['feature_observed'][0] = False
    else:
        a['patches']['observed'][0] = False
    r = solve(**a)
    assert r['status'] == 'unresolved'
    assert r['tau_spring'] is None and r['reason'] == 'unobserved_active_feature'


def test_inputs_immutable_deterministic_and_json_finite():
    a = system(); a['f'][1] = 2.
    before = deepcopy(a)
    first = solve(**a); second = solve(**a)
    assert json.dumps(first, allow_nan=False, sort_keys=True) == json.dumps(second, allow_nan=False, sort_keys=True)
    for k in ('M','q','v','K','C','J','g','s','kc','dc','f'):
        np.testing.assert_array_equal(a[k], before[k])
    assert a['patches'] == before['patches']


def test_generalized_rotation_covariance_and_preserve_full_mass_coupling():
    a = system(); a['q'][6:] = [.02, -.01]; a['f'][1] = 2.
    a['M'][0, 6] = a['M'][6, 0] = .2
    a['M'][1, 7] = a['M'][7, 1] = -.1
    base = solve(**a)
    Q = np.eye(8); Q[:3,:3] = _ry(.4); Q[3:6,3:6] = _ry(.4)
    b = deepcopy(a)
    b.update(M=Q@a['M']@Q.T, q=Q@a['q'], v=Q@a['v'], f=Q@a['f'],
             K=Q@np.diag(a['K'])@Q.T, C=Q@np.diag(a['C'])@Q.T, J=a['J']@Q.T)
    b['patches']['tangent_jacobians'] = (np.asarray(a['patches']['tangent_jacobians'])@Q.T).tolist()
    rotated = solve(**b)
    assert base['status'] == rotated['status'] == 'resolved'
    np.testing.assert_allclose(rotated['v_pred'], Q@base['v_pred'], atol=1e-12)
    np.testing.assert_allclose(rotated['tau_spring'], Q@base['tau_spring'], atol=1e-12)


def test_contact_spring_energy_and_friction_work_identity():
    a = system(); a['q'][6:] = [.02, -.01]; a['v'][1:3] = [.1, .05]
    a['f'][:] = 0
    r = solve(**a)
    assert r['status'] == 'resolved' and all(r['normal_active'])
    w = np.asarray(r['v_pred']); h = a['h']; v = a['v']
    g1 = np.asarray(r['gap_pred_m']); q1 = a['q']+h*w
    E0 = .5*(v@a['M']@v + a['q']@(a['K']*a['q']) + a['g']@(a['kc']*a['g']))
    E1 = .5*(w@a['M']@w + q1@(a['K']*q1) + g1@(a['kc']*g1))
    identity = (-h*(w@(a['C']*w)+(a['J']@w)@(a['dc']*(a['J']@w)))
        -.5*(w-v)@a['M']@(w-v)-.5*h*h*(w@(a['K']*w)+(a['J']@w)@(a['kc']*(a['J']@w)))
        +h*np.sum(r['anchor_relative_power_w']))
    assert E1 < E0
    assert E1-E0 == pytest.approx(identity, abs=2e-12)


@pytest.mark.parametrize('change', [
    dict(q=[0]*7), dict(M=np.zeros((8,8))), dict(K=[1]*8), dict(C=[np.nan]*8),
    dict(h=0), dict(h=np.inf), dict(h=True), dict(f=[np.inf]*8),
    dict(law=MaterialLaw('signed_overlap_kv_v1')), dict(feature_observed=None),
    dict(feature_observed=[1,1]), dict(max_iterations=0), dict(max_iterations=65),
])
def test_invalid_inputs_fail_closed(change):
    a = system(); a.update(change)
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        solve(**a)


@pytest.mark.parametrize('change', [
    dict(mu=-1), dict(mu=np.nan), dict(mu=True), dict(observed=[1]),
    dict(normal_indices=[[0,0]]), dict(normal_indices=[[0,2]]),
    dict(normal_indices=[[0.,1.]]), dict(model='native_auto'),
    dict(surface_speeds_m_s=[[[1,0],[0,0]]]),
    dict(tangent_jacobians=np.zeros((1,2,2,8)).tolist()),
    dict(tangent_jacobians=np.full((1,2,2,8), np.nan).tolist()),
])
def test_bad_patch_geometry_material_contract_rejected(change):
    a = system(); a['patches'].update(change)
    with pytest.raises(ValueError):
        solve(**a)


def test_compile_exact_two_anchors_and_ignores_every_impulse():
    a = geometry(); original = deepcopy(a['native_rows'])
    J,g,s,report = compile_patches(**a)
    assert J.shape == (12,8) and len(g) == 12
    assert len(report['patch_geometry']) == 6
    assert report['patches']['mu'] == a['coupon'].friction
    assert np.asarray(report['patches']['tangent_jacobians']).shape == (6,2,2,8)
    assert not report['native_impulses_read'] and not report['impulse_replayed']
    for row in a['native_rows']:
        row['impulse_on_0_ns'] = [np.nan, np.inf, -1e99]
    again = compile_patches(**a)
    assert again[3] == report  # fields are not even inspected/serialized
    assert all('signed_impulse_on_body_ns' not in match for match in report['matches'])
    for patch in report['patch_geometry']:
        for anchor in patch['anchors']:
            point = original[anchor['native_row']]['point_world_m']
            np.testing.assert_allclose(anchor['current_bound_point_world_m'], point, atol=1e-17)
    json.dumps(report, allow_nan=False)
    np.testing.assert_array_equal(s, 0)


def test_binding_retains_exact_local_points_uses_current_not_reference_jacobian():
    a = geometry()
    _,_,_,first = compile_patches(**a)
    a['anchor_binding'] = first['anchor_binding']
    previous = deepcopy(a['frames'])
    a['frames'] = previous.copy()
    a['frames'][:, 1, 3] += .4e-6
    a['nativeJac'] = _kinematic_jacobian(a['frames'])
    a['native_geometry_frames'] = previous
    a['step_id'] = 2
    _,_,_,second = compile_patches(**a)
    assert second['anchor_binding'] == first['anchor_binding']
    for p1,p2 in zip(first['patch_geometry'], second['patch_geometry']):
        for x,y in zip(p1['anchors'], p2['anchors']):
            np.testing.assert_allclose(np.array(y['current_bound_point_world_m'])
                -x['current_bound_point_world_m'], [0,.4e-6,0], atol=1e-17)


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'shift', 'unknown', 'extra',
                                  'source', 'corrupt', 'stale', 'owner', 'normal'])
def test_geometry_changes_and_missing_manifolds_rejected(fault):
    a = geometry()
    _,_,_,first = compile_patches(**a)
    a['anchor_binding'] = first['anchor_binding']
    friction = [i for i,r in enumerate(a['native_rows']) if r['kind']=='friction']
    if fault == 'missing':
        a['native_rows'].pop(friction[0])
    elif fault == 'duplicate':
        a['native_rows'][friction[1]] = deepcopy(a['native_rows'][friction[0]])
    elif fault == 'shift':
        a['native_rows'][friction[0]]['point_world_m'][2] += 3e-6
    elif fault == 'unknown':
        a['native_rows'][friction[0]]['collider0'] = '/unknown'
    elif fault == 'extra':
        a['native_rows'].append(deepcopy(a['native_rows'][friction[0]]))
    elif fault == 'source':
        a['coupon'] = replace(a['coupon'], source_sha256='b'*64)
    elif fault == 'corrupt':
        a['anchor_binding']['patches'][0]['anchors'][0]['local_point_m'][0] += 1e-9
    elif fault == 'stale':
        a['step_id'] = 4
    elif fault == 'owner':
        for i in friction[:2]:
            r=a['native_rows'][i]
            r['collider0'],r['collider1']=r['collider1'],r['collider0']
    else:
        a['native_rows'][0]['normal_on_0'] = [0,1,0]
    with pytest.raises(ValueError):
        compile_patches(**a)


def test_reversed_original_order_is_explicit_and_geometry_invariant_at_binding():
    a = geometry()
    first = compile_patches(**a)
    for row in a['native_rows']:
        row['collider0'],row['collider1'] = row['collider1'],row['collider0']
        if row['kind']=='normal':
            row['normal_on_0'] = (-np.asarray(row['normal_on_0'])).tolist()
    second = compile_patches(**a)
    np.testing.assert_allclose(first[0],second[0])
    np.testing.assert_allclose(first[3]['patches']['tangent_jacobians'],
                               second[3]['patches']['tangent_jacobians'], atol=1e-15)
    assert second[3]['anchor_binding']['patches'][0]['anchors'][0]['owner'].endswith('Pad_0_minus')


def test_free_compile_has_no_fake_anchors():
    a=geometry(coupon(held_contacts=False))
    J,g,s,r=compile_patches(**a)
    assert J.shape==(0,8) and len(g)==len(s)==0
    assert r['anchor_binding']['patches']==[]
    assert r['patches']['tangent_jacobians']==[]


def _source_mass_fixture():
    # Analytic synthetic coupon spatial inertia, NOT a native dynamics readback.
    fixture = geometry(coupon(dt=1/1920)); c = fixture['coupon']
    J,g,s,report = compile_patches(**fixture)
    jac = fixture['nativeJac']; frames = fixture['frames']; M = np.zeros((8,8))
    for i in range(3):
        spatial = np.zeros((6,6))
        spatial[:3,:3] = np.eye(3)*c.masses[i]
        R = frames[i,:3,:3]
        spatial[3:,3:] = R@np.array(c.inertias[i])@R.T
        M += jac[i].T@spatial@jac[i]
    return dict(M=M, q=np.r_[np.zeros(6),.005,.005], v=np.zeros(8),
        K=np.r_[np.zeros(6),c.stiffness], C=np.r_[np.zeros(6),c.damping],
        J=J,g=g,s=s,kc=np.full(12,c.contact_stiffness),dc=np.full(12,c.contact_damping),
        h=1/1920,f=np.zeros(8),law=LAW,patches=report['patches'],
        feature_observed=report['feature_observed'])


def test_redundant_source_mass_patch_trust_step_solves_original_undamped_equations():
    a = _source_mass_fixture()
    r = solve(**a)
    assert r['status'] == 'resolved'
    assert r['numerical_trust_steps'] > 0 and r['iterations'] <= MAX_ITERATIONS
    assert r['relative_fixed_point_residual'] < 1e-8
    w = np.asarray(r['v_pred']); tau = np.asarray(r['tau_spring'])
    T = np.asarray(a['patches']['tangent_jacobians']).reshape(-1,8)
    N = np.asarray(r['normal_forces_n']); F = np.asarray(r['friction_forces_n']).ravel()
    original = a['M']@(w-a['v'])-a['h']*(a['f']+tau+a['J'].T@N+T.T@F)
    np.testing.assert_allclose(original, 0, atol=1e-17)
    np.testing.assert_allclose(tau, -a['K']*(a['q']+a['h']*w)-a['C']*w, atol=1e-17)
    assert tau[:6].tolist() == [0.]*6
    assert np.max(r['cone_max_violation_n']) < 1e-9


def test_redundant_fixture_exhaustion_never_returns_unqualified_torque():
    r = solve(**_source_mass_fixture(), max_iterations=2)
    assert r['status'] == 'unresolved' and r['reason'] == 'iteration_limit'
    assert r['tau_spring'] is r['tau_joint'] is r['v_pred'] is None
    assert np.isfinite(r['normal_law_residual_n'])
    assert np.isfinite(r['friction_fixed_point_residual_n'])


def test_geometry_binding_includes_material_and_rejects_coefficient_change():
    a = geometry(); _,_,_,r = compile_patches(**a)
    a['anchor_binding'] = r['anchor_binding']
    a['coupon'] = replace(a['coupon'], friction=.6)
    with pytest.raises(ValueError, match='binding changed'):
        compile_patches(**a)


def test_solver_has_no_measured_friction_force_argument():
    with pytest.raises(TypeError):
        solve(**system(), measured_friction_force=np.ones(4))


# Numerical acceleration only: all force seeds below are returned by solve.
def _bound_system():
    a = system()
    a['patches']['prediction_context'] = dict(schema='coupon_patch_prediction_context_v1',
        source_sha256='a'*64, coupon_sha256='b'*64, anchor_binding_sha256='c'*64,
        root='/World/SyntheticAlgebra', dt_s=a['h'], step_id=1)
    return a


def _advance(a):
    a['patches']['prediction_context']['step_id'] += 1


def test_all_stick_candidate_uses_all_eight_coordinates_not_root_pin():
    a = system()
    a['f'][1] = .5
    a['f'][3:6] = [.1, -.2, .3]  # unconstrained root rotations MUST move
    a['M'][3, 6] = a['M'][6, 3] = .2
    a['q'][6:] = [.02, -.01]
    from sim_physics.contact_patch_prediction import _solve
    cold = _solve(**a, _try_all_stick=False)
    fast = solve(**a, max_iterations=1)
    assert fast['all_stick_candidate']['status'] == 'accepted'
    assert fast['iterations'] == 0 and fast['tau_spring'][:6] == [0.]*6
    assert np.linalg.norm(fast['v_pred'][3:6]) > .001
    np.testing.assert_allclose(fast['v_pred'], cold['v_pred'], atol=2e-15)
    np.testing.assert_allclose(fast['tau_spring'], cold['tau_spring'], atol=2e-15)
    assert fast['relative_fixed_point_residual'] <= 1e-8
    assert fast['relative_equilibrium_residual'] <= 1e-8


def test_nonunique_tangent_forces_do_not_require_cold_warm_multiplier_equality():
    a = _bound_system(); a['f'][1] = .6
    T = np.asarray(a['patches']['tangent_jacobians'])
    T[0, 0, 0, 1] = 2.
    a['patches']['tangent_jacobians'] = T.tolist()
    first = solve(**a)
    seed = first['next_predicted_seed']; saved = deepcopy(seed)
    _advance(a)
    # Same bound anchors/basis, changed current point Jacobian: no stale J reused.
    T[0, 0, 0, 1] = 1.
    a['patches']['tangent_jacobians'] = T.tolist()
    cold = solve(**a)
    warm = solve(**a, previous_predicted_seed=seed)
    assert seed == saved
    assert warm['all_stick_candidate']['status'] == 'accepted'
    assert warm['previous_predicted_seed_used']
    np.testing.assert_allclose(warm['v_pred'], cold['v_pred'], atol=2e-15)
    np.testing.assert_allclose(warm['tau_spring'], cold['tau_spring'], atol=2e-15)
    assert not np.allclose(warm['friction_forces_n'], cold['friction_forces_n'])
    json.dumps(warm, allow_nan=False)
    assert warm['next_predicted_seed']['context']['step_id'] == 2


@pytest.mark.parametrize('fault', [
    'same_step', 'skip_step', 'source', 'coupon', 'anchor', 'root', 'dt',
    'schema', 'seed_schema', 'checksum', 'force_nan', 'force_inf',
    'material', 'mu', 'indices', 'extra_measured_field', 'unbound'])
def test_seed_provenance_invalid_before_prediction(monkeypatch, fault):
    import sim_physics.contact_patch_prediction as module
    a = _bound_system()
    seed = solve(**a)['next_predicted_seed']
    _advance(a)
    ctx = a['patches']['prediction_context']
    if fault == 'same_step': ctx['step_id'] = 1
    elif fault == 'skip_step': ctx['step_id'] = 3
    elif fault in ('source', 'coupon', 'anchor'):
        key = dict(source='source_sha256', coupon='coupon_sha256',
                   anchor='anchor_binding_sha256')[fault]
        ctx[key] = 'd'*64
    elif fault == 'root': ctx['root'] = '/World/Another'
    elif fault == 'dt': ctx['dt_s'] = a['h'] = .02
    elif fault == 'schema': ctx['schema'] = 'future_context'
    elif fault == 'seed_schema': seed['schema'] = 'native_force'
    elif fault == 'checksum': seed['friction_forces_n'][0][0][0] = 1e-20
    elif fault == 'force_nan': seed['friction_forces_n'][0][0][0] = np.nan
    elif fault == 'force_inf': seed['friction_forces_n'][0][0][0] = np.inf
    elif fault == 'material': a['K'][6] += .01
    elif fault == 'mu': a['patches']['mu'] = .6
    elif fault == 'indices': a['patches']['normal_indices'] = [[1, 0]]
    elif fault == 'extra_measured_field': seed['measured_force'] = [0, 0, 0]
    else: a['patches'].pop('prediction_context')
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid seed reached prediction')
    monkeypatch.setattr(module, '_solve', forbidden)
    with pytest.raises(ValueError):
        solve(**a, previous_predicted_seed=seed)


@pytest.mark.parametrize('change', [
    {'step_id': True}, {'step_id': -1}, {'dt_s': True}, {'dt_s': np.nan},
    {'source_sha256': 'not-a-hash'}, {'anchor_binding_sha256': None},
    {'schema': 'v0'}, {'root': ''},
])
def test_invalid_compiled_context_never_used(change):
    a = _bound_system()
    a['patches']['prediction_context'].update(change)
    with pytest.raises(ValueError):
        solve(**a)


def test_compiler_context_bound_to_source_anchor_step_and_dt():
    a = geometry(coupon(dt=1/1920))
    _, _, _, report = compile_patches(**a)
    ctx = report['patches']['prediction_context']
    binding = report['anchor_binding']
    assert ctx['source_sha256'] == a['coupon'].source_sha256
    assert ctx['coupon_sha256'] == binding['coupon_sha256']
    assert ctx['anchor_binding_sha256'] == binding['binding_sha256']
    assert ctx['dt_s'] == a['coupon'].dt and ctx['step_id'] == 1
    assert ctx['root'] == a['coupon'].root


def test_warm_cone_failure_falls_back_to_identical_cold_iterations_and_effort():
    from sim_physics.contact_patch_prediction import _solve
    a = _bound_system(); a['f'][1] = .5
    seed = solve(**a)['next_predicted_seed']
    _advance(a); a['f'][1:3] = [1.2, 1.6]
    baseline = _solve(**a, _try_all_stick=False)
    warm = solve(**a, previous_predicted_seed=seed)
    assert warm['all_stick_candidate']['status'] == 'rejected'
    for key in ('v_pred', 'tau_spring', 'friction_forces_n', 'normal_forces_n',
                'iterations', 'backtracks', 'numerical_trust_steps',
                'relative_fixed_point_residual', 'relative_equilibrium_residual'):
        np.testing.assert_array_equal(warm[key], baseline[key])


def test_rejected_candidate_exhaustion_no_seed_or_effort():
    a = _bound_system()
    seed = solve(**a)['next_predicted_seed']; _advance(a)
    a['f'][1] = 2.
    r = solve(**a, previous_predicted_seed=seed, max_iterations=1)
    assert r['status'] == 'unresolved' and r['reason'] == 'iteration_limit'
    assert r['tau_spring'] is r['v_pred'] is r['next_predicted_seed'] is None


def test_all_stick_factorization_failure_uses_unchanged_cold_solver(monkeypatch):
    import sim_physics.contact_patch_prediction as module
    a = system(); a['f'][1] = .5
    cold = module._solve(**a, _try_all_stick=False)
    def fail(*args, **kwargs):
        raise np.linalg.LinAlgError('synthetic failed candidate only')
    monkeypatch.setattr(module, '_all_stick_candidate', fail)
    r = solve(**a)
    assert r['all_stick_candidate']['reason'] == 'numerical_candidate_failure'
    np.testing.assert_array_equal(r['v_pred'], cold['v_pred'])
    np.testing.assert_array_equal(r['tau_spring'], cold['tau_spring'])
    assert r['iterations'] == cold['iterations']


def test_candidate_normal_branch_change_must_fall_back(monkeypatch):
    import sim_physics.contact_patch_prediction as module
    a = system(); a['f'][1] = .5
    original = module._all_stick_candidate
    def changed(*args):
        # Deliberately request the wrong branch; candidate must reject it.
        args = list(args); args[-2] = ~args[-2]
        return original(*args)
    monkeypatch.setattr(module, '_all_stick_candidate', changed)
    r = solve(**a)
    assert r['all_stick_candidate']['reason'] == 'normal_branch_changed'
    assert r['status'] == 'resolved' and r['iterations'] > 0


def test_candidate_original_equations_checked_not_just_cone_or_rank(monkeypatch):
    import sim_physics.contact_patch_prediction as module
    a = system(); a['f'][1] = .5
    original = module._all_stick_candidate
    def corrupted(*args):
        z, info = original(*args)
        z[0] += 1e-4  # apparently finite/compressive but violates exact normal law
        return z, info
    monkeypatch.setattr(module, '_all_stick_candidate', corrupted)
    cold = module._solve(**a, _try_all_stick=False)
    r = solve(**a)
    assert r['all_stick_candidate']['reason'] == 'original_fixed_point_residual'
    np.testing.assert_array_equal(r['tau_spring'], cold['tau_spring'])
    np.testing.assert_array_equal(r['v_pred'], cold['v_pred'])


def test_free_bound_seed_never_creates_contacts_or_changes_existing_prediction():
    a = system(0)
    a['patches']['prediction_context'] = _bound_system()['patches']['prediction_context']
    a['q'][6:] = [.002, -.004]; a['v'][3:6] = [.01, -.02, .03]
    first = solve(**a)
    assert first['next_predicted_seed']['friction_forces_n'] == []
    _advance(a)
    warm = solve(**a, previous_predicted_seed=first['next_predicted_seed'])
    base = normal_solve(**{k:v for k,v in a.items() if k != 'patches'})
    np.testing.assert_array_equal(warm['v_pred'], base['v_pred'])
    np.testing.assert_array_equal(warm['tau_spring'], base['tau_spring'])
    assert warm['friction_forces_n'] == []


def test_unbound_algebra_emits_no_seed_and_inputs_results_do_not_alias():
    assert solve(**system())['next_predicted_seed'] is None
    a = _bound_system(); before = deepcopy(a['patches'])
    r = solve(**a)
    r['next_predicted_seed']['context']['step_id'] = 99
    r['next_predicted_seed']['friction_forces_n'][0][0][0] = 99.
    assert a['patches'] == before
    assert r['friction_forces_n'][0][0][0] != 99.


def test_archived_native32_step2_keeps_original_failure_without_effort():
    """Optional immutable capture replay; never launches native or reads impulses."""
    import hashlib
    from dataclasses import fields
    from sim_physics.contact_patch_prediction import _solve, _prediction_context
    directory = (Path(__file__).resolve().parents[3] / 'data' / 'sim_physics'
                 / 'contact_spring_native_20260911_32')
    if not (directory / 'trace.json').is_file():
        pytest.skip('Ignored native32 capture not present; pure regressions still run')
    trace_bytes = (directory / 'trace.json').read_bytes()
    report_bytes = (directory / 'report.json').read_bytes()
    assert hashlib.sha256(trace_bytes).hexdigest() == '8daa9e04625818260a48a3231ab000ceec09ad4f8ff0531889b82c46d8273d42'
    assert hashlib.sha256(report_bytes).hexdigest() == '20a644b244366de0ecab786e1d2a4616d1e76e0bb1001b03fdbce55242a74ca9'
    row, = json.loads(trace_bytes)
    report = json.loads(report_bytes)
    c = Coupon(**{x.name: report['configuration'][x.name] for x in fields(Coupon)})
    before = row['native_prediction_before_step']
    geom = row['coupled_prediction']['geometry']
    frames = np.asarray(before['frames_world_m'])
    jac = np.asarray(before['body_world_com_jacobians'])
    J = []; g = []
    for x in geom['features']:
        i = int(x['collider'].split('/Link_')[1].split('/')[0])
        arm = np.asarray(x['point_world_m'])-frames[i, :3, 3]
        point_jac = jac[i, :3]+np.cross(jac[i, 3:].T, arm).T
        J.append(np.asarray(x['normal_on_body'])@point_jac)
        g.append(x['gap_m'])
    patches = deepcopy(geom['patches'])
    patches['prediction_context'] = _prediction_context(c, geom['anchor_binding'], 1)
    a = dict(M=np.asarray(before['mass_matrix']),
        q=np.r_[np.zeros(6), report['initial_readback']['q']],
        v=np.asarray(before['generalized_velocity']),
        K=np.r_[np.zeros(6), c.stiffness], C=np.r_[np.zeros(6), c.damping],
        J=np.array(J), g=np.array(g), s=np.zeros(len(g)),
        kc=np.full(len(g), c.contact_stiffness), dc=np.full(len(g), c.contact_damping),
        h=c.dt, f=np.asarray(before['native_known_external_force']),
        law=LAW, patches=patches, feature_observed=geom['feature_observed'])
    first = solve(**a)
    assert first['status'] == 'resolved'
    # Step2 current state is saved post-step1, with row1 reference BEFORE step1.
    after = row['native_prediction_after_step']
    J, g, s, compiled = compile_patches(c, row['frames_world_m'],
        row['body_velocities_world'], after['body_world_com_jacobians'],
        native_rows=row['contact_rows'], generalized_velocity=after['generalized_velocity'],
        step_id=2, native_geometry_step_id=1,
        native_geometry_frames=before['frames_world_m'],
        anchor_binding=geom['anchor_binding'])
    a.update(M=np.asarray(after['mass_matrix']), q=np.r_[np.zeros(6), row['q_rad']],
        v=np.asarray(after['generalized_velocity']),
        f=np.asarray(after['native_known_external_force']),
        J=J, g=g, s=s, patches=compiled['patches'],
        feature_observed=compiled['feature_observed'])
    cold = _solve(**a, _try_all_stick=False)
    warm = solve(**a, previous_predicted_seed=first['next_predicted_seed'])
    assert warm['all_stick_candidate']['status'] == 'rejected'
    assert warm['status'] == cold['status'] == 'unresolved'
    assert warm['reason'] == cold['reason'] == 'semismooth_line_search'
    assert warm['iterations'] == cold['iterations'] == 11
    assert warm['next_predicted_seed'] is warm['tau_spring'] is warm['v_pred'] is None
    for key in ('relative_fixed_point_residual', 'normal_law_residual_n',
                'friction_fixed_point_residual_n'):
        assert warm[key] == cold[key]
    assert warm['relative_fixed_point_residual'] > 1e-8
