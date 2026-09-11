"""Pure numerical/contact-feature tests; never import or start a native app."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sim_physics.contact_coupled_prediction import (
    MaterialLaw, compile_geometry, solve, _kinematic_jacobian, _features,
    POINT_TOL_M, MAX_ROWS,
)
from sim_physics.contact_spring_probe import Coupon, layout, _ry
from sim_physics.implicit_springs import elastic_effort


UNILATERAL = MaterialLaw('unilateral_kv_v1')
SIGNED = MaterialLaw('signed_overlap_kv_v1')


def coupon(**kwargs):
    return Coupon('synthetic-not-native.json', 'a'*64, (.33289975, .30481708),
        (.00940549, .00807237), (.000633, .000607, .000580),
        tuple(np.diag([3e-8, 3e-8, 2.5e-9]) for _ in range(3)),
        1000., 1.0573968428135787, .5, **kwargs)


def system(n=8, contacts=0):
    return dict(M=np.eye(n)*.001, q=np.zeros(n), v=np.zeros(n),
        K=np.r_[np.zeros(n-2), [.3, .2]], C=np.r_[np.zeros(n-2), [.009, .008]],
        J=np.zeros((contacts, n)), g=np.zeros(contacts), s=np.zeros(contacts),
        kc=np.full(contacts, 1000.), dc=np.full(contacts, 1.0573968428135787),
        h=1/240, f=np.zeros(n), law=UNILATERAL)


def scalar(**kwargs):
    args = dict(M=[[1.]], q=[0.], v=[0.], K=[0.], C=[0.], J=[[1.]],
                g=[-.1], s=[0.], kc=[1.], dc=[0.], h=.1, f=[0.],
                law=UNILATERAL, root_dofs=0)
    args.update(kwargs)
    return solve(**args)


def test_isolated_import_has_no_native_runtime():
    env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    result = subprocess.run([sys.executable, '-B', '-c',
        'import sys; import sim_physics.contact_coupled_prediction; '
        'assert not any(x.startswith(("pxr", "omni", "isaacsim")) for x in sys.modules)'],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_empty_contacts_exactly_reproduce_old_free_root_predictor():
    a = system(); a['q'][-2:] = [.004, .003]; a['v'] = np.arange(8)*.0001
    expected = elastic_effort(a['M'], a['q'], a['v'], a['K'], a['C'], a['f'], a['h'])
    before = deepcopy(a)
    result = solve(**a)
    assert result['status'] == 'resolved'
    np.testing.assert_allclose(result['tau_spring'], expected, atol=1e-16)
    assert result['tau_spring'][:6] == [0.]*6
    assert result['scalar_response_n'] == []
    for key in ('M', 'q', 'v', 'K', 'C', 'J', 'g'):
        np.testing.assert_array_equal(a[key], before[key])
    json.dumps(result, allow_nan=False)


def test_full_coupled_mass_not_constrained_root_block():
    a = system(); a['M'][0, 6] = a['M'][6, 0] = .0004
    a['q'][6] = .01
    result = solve(**a)
    assert abs(result['v_pred'][0]) > 0
    assert result['tau_spring'][0] == 0


def test_exact_loaded_static_equilibrium_keeps_unattenuated_Kq():
    a = system(contacts=1)
    a['q'][6] = .01; a['J'][0, 6] = 1.
    a['g'][0] = -.3*.01/1000
    result = solve(**a)
    assert result['status'] == 'resolved'
    np.testing.assert_allclose(result['v_pred'], 0, atol=1e-15)
    assert result['tau_spring'][6] == pytest.approx(-.003, abs=1e-15)
    assert result['scalar_response_n'] == pytest.approx([.003], abs=1e-15)


def test_spring_only_submission_reproduces_ideal_native_contact_without_double_force():
    rng = np.random.default_rng(20)
    for _ in range(30):
        a = system(contacts=12)
        r = rng.normal(size=(8, 8)); a['M'] = (r.T@r+np.eye(8))*.001
        a['J'] = rng.normal(size=(12, 8))*.03
        a['g'][:] = -.001
        a['q'][-2:] = rng.normal(size=2)*1e-4
        a['v'] = rng.normal(size=8)*1e-5
        result = solve(**a)
        assert result['status'] == 'resolved' and all(result['active'])
        h, J, kc, dc = (a[k] for k in ('h', 'J', 'kc', 'dc'))
        B = a['M']+(J.T*(h*dc+h*h*kc))@J
        vp = np.array(result['v_pred']); tau = np.array(result['tau_spring'])
        replay = np.linalg.solve(B, a['M']@a['v']+h*(tau-J.T@(kc*a['g'])))
        np.testing.assert_allclose(replay, vp, atol=2e-15, rtol=2e-14)
        assert np.all(tau[:6] == 0)


def test_fixed_active_stationary_energy_identity():
    a = system(contacts=3); a['J'][:, 6:] = [[.01, .02], [-.03, .01], [.02, -.01]]
    a['g'][:] = -.001; a['q'][-2:] = [.003, .004]
    a['v'][-2:] = [.001, -.002]
    r = solve(**a, )
    assert r['status'] == 'resolved' and all(r['active'])
    w = np.array(r['v_pred']); p = a['q']+a['h']*w; gp = np.array(r['gap_pred_m'])
    M, K, C, J, kc, dc, h, v, q, g = (a[k] for k in ('M','K','C','J','kc','dc','h','v','q','g'))
    E0 = .5*(v@M@v+q@(K*q)+g@(kc*g))
    E1 = .5*(w@M@w+p@(K*p)+gp@(kc*gp))
    identity = (-h*(w@(C*w)+(J@w)@(dc*(J@w)))
                -.5*(w-v)@M@(w-v)-.5*h*h*(w@(K*w)+(J@w)@(kc*(J@w))))
    assert E1 < E0
    assert E1-E0 == pytest.approx(identity, abs=1e-18)


def test_moving_surface_includes_stiffness_translation_term():
    r = scalar(g=[-.1], s=[.2], kc=[3.], dc=[.4])
    expected = .1*(.3+(.4+.1*3)*.2)/(1+.1*.4+.1**2*3)
    assert r['v_pred'] == pytest.approx([expected])


def test_explicit_signed_overlap_response_not_positive_grasp_evidence():
    signed = scalar(v=[1.], dc=[1.], h=.001, law=SIGNED)
    unilateral = scalar(v=[1.], dc=[1.], h=.001)
    assert signed['scalar_response_n'][0] < 0
    assert signed['negative_overlap_response'] == [True]
    assert signed['opening'] == [True] and signed['separated_pred'] == [False]
    assert unilateral['scalar_response_n'] == [0.]
    assert not signed['native_contact_law_parity']


@pytest.mark.parametrize('law', [SIGNED, UNILATERAL])
def test_separated_surface_has_no_adhesive_or_damping_connection(law):
    r = scalar(g=[.1], v=[1.], dc=[100.], law=law)
    assert r['scalar_response_n'] == [0.] and r['v_pred'] == [1.]
    assert r['separated_pred'] == [True]


def test_closing_surface_can_activate_in_current_coupled_solve():
    r = scalar(g=[.01], v=[-1.], h=.1)
    assert r['status'] == 'resolved' and r['iterations'] == 2
    assert r['gap_pred_m'][0] < 0 and r['scalar_response_n'][0] > 0


def test_iteration_exhaustion_never_returns_a_partial_effort():
    r = scalar(g=[.01], v=[-1.], h=.1, max_iterations=1)
    assert r['status'] == 'unresolved' and r['reason'] == 'iteration_limit'
    assert r['tau_spring'] is None and r['v_pred'] is None


def test_missing_feature_cannot_become_load_bearing_by_prediction():
    r = scalar(g=[.01], v=[-1.], feature_observed=[False])
    assert r['reason'] == 'unobserved_active_feature' and r['tau_spring'] is None
    r = scalar(g=[.1], v=[1.], feature_observed=[False])
    assert r['status'] == 'resolved' and r['feature_coverage_bound']
    assert r['scalar_response_n'] == [0.]
    with pytest.raises(ValueError): scalar(feature_observed=[1])


def test_discontinuous_opening_active_set_cycle_fails_closed():
    r = scalar(g=[.01], v=[-1.], h=.1, dc=[100.])
    assert r['status'] == 'unresolved' and r['reason'] == 'active_set_cycle'
    assert r['tau_spring'] is None


@pytest.mark.parametrize('change', [dict(h=0), dict(h=True), dict(h=float('nan')),
    dict(M=np.zeros((8,8))), dict(K=[-1]*8), dict(C=[float('nan')]*8),
    dict(K=[1]*8), dict(q=[0]*7), dict(J=np.empty((1,8))),
    dict(root_dofs=True), dict(max_iterations=0), dict(max_iterations=33),
    dict(law='unilateral_kv_v1'), dict(f=[float('inf')]*8)])
def test_invalid_solver_input_fails_closed(change):
    a = system(); a.update(change)
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        solve(**a)


def test_invalid_contact_coefficients_and_nonfinite_values():
    for key, value in [('kc', [-1.]), ('dc', [-1.]), ('g', [np.nan]), ('s', [np.inf])]:
        a = system(contacts=1); a[key] = value
        with pytest.raises(ValueError): solve(**a)
    with pytest.raises(ValueError): MaterialLaw('auto_native_sign')


def test_rotated_generalized_basis_preserves_coupled_solution():
    a = system(contacts=2); a['J'][:, 6:] = [[.02, .01], [.01, -.02]]
    a['g'][:] = -.001; a['q'][-2:] = [.004, -.003]
    r = solve(**a)
    t = np.eye(8); t[:3, :3] = _ry(.7); t[3:6, 3:6] = _ry(.7)
    t[6:, 6:] = [[0, -1], [1, 0]]
    b = dict(a, M=t@a['M']@t.T, q=t@a['q'], v=t@a['v'], f=t@a['f'],
             K=t@np.diag(a['K'])@t.T, C=t@np.diag(a['C'])@t.T, J=a['J']@t.T)
    rr = solve(**b)
    np.testing.assert_allclose(rr['v_pred'], t@r['v_pred'], atol=1e-14)
    np.testing.assert_allclose(rr['tau_spring'], t@r['tau_spring'], atol=1e-14)


def native_rows(c):
    """Synthetic PCM segment construction, NOT native observations.

    On the initial finite face: x=cx+(z-cz)*ax/az+side*radius.
    The radius is along the face normal, not projected onto a circular slice.
"""
    data = layout(c); result = []; radius = float(np.float32(.0028))
    for i, frame in enumerate(data['frames']):
        for side, suffix in ((-1, '_minus'), (1, '_plus')) if c.held_contacts else ():
            path = c.root+'/Pad_'+str(i)+suffix; pad = data['pads'][path]
            R = pad[:3, :3]; center = R.T@(frame[:3, 3]-pad[:3, 3])
            axis = R.T@frame[:3, 2]
            for edge in (-1, 1):
                z = edge*.008
                x = center[0]+(z-center[2])*axis[0]/axis[2]+side*radius
                point = pad[:3, 3]+R@[x, 0., z]
                normal = R@[-side, 0., 0.]
                result.append(dict(kind='normal', collider0=data['collider_paths'][i], collider1=path,
                    point_world_m=point.tolist(), normal_on_0=normal.tolist(),
                    separation_m=float(-side*(x+side*.004)),
                    impulse_on_0_ns=(normal*-.000001).tolist()))
    return result


def geometry(c=None):
    c = coupon() if c is None else c; frames = layout(c)['frames']
    jac = _kinematic_jacobian(frames); v = np.arange(8)*.00001
    return dict(coupon=c, frames=frames, velocities=np.einsum('bij,j->bi', jac, v),
        nativeJac=jac, native_rows=native_rows(c), generalized_velocity=v,
        step_id=1, native_geometry_step_id=1)


def test_finite_face_twelve_individual_edge_points_and_signed_rows_preserved():
    a = geometry(); before = deepcopy(a)
    J, g, s, report = compile_geometry(**a)
    assert J.shape == (12,8) and len(g) == 12 and np.all(s == 0)
    assert report['normal_rows'] == report['candidate_features'] == 12
    assert report['native_feature_comparison'] and not report['native_contact_law_parity']
    assert max(x['point_error_m'] for x in report['matches']) < 1e-15
    for feature in report['features']:
        pad = layout(a['coupon'])['pads'][feature['pad']]
        local = pad[:3, :3].T@(np.array(feature['point_world_m'])-pad[:3, 3])
        assert abs(local[2]) == pytest.approx(.008, abs=1e-15)
    # Different edge moment arms are retained, never one pad midpoint.
    assert abs(J[4, 4]-J[5, 4]) > .015
    assert np.dot(report['matches'][0]['signed_impulse_on_body_ns'], report['features'][0]['normal_on_body']) < 0
    json.dumps(report, allow_nan=False)
    np.testing.assert_array_equal(a['frames'], before['frames'])
    assert a['native_rows'] == before['native_rows']


def test_original_collider_order_swap_preserves_geometry_and_signed_impulse():
    a = geometry(); expected = compile_geometry(**a)
    for row in a['native_rows']:
        row['collider0'], row['collider1'] = row['collider1'], row['collider0']
        for key in ('normal_on_0', 'impulse_on_0_ns'):
            row[key] = (-np.array(row[key])).tolist()
    got = compile_geometry(**a)
    for x,y in zip(got[:3], expected[:3]): np.testing.assert_array_equal(x,y)
    assert got[3]['matches'] == expected[3]['matches']


def test_rotated_entire_coupon_normal_gap_and_torque_covariance():
    a = geometry(); J, g, _, _ = compile_geometry(**a)
    R = _ry(.7); rotated = geometry(replace(a['coupon'], rotation=R))
    Jr, gr, _, _ = compile_geometry(**rotated)
    transform = np.eye(8); transform[:3, :3] = R; transform[3:6, 3:6] = R
    np.testing.assert_allclose(gr, g, atol=1e-17)
    np.testing.assert_allclose(Jr@transform, J, atol=1e-15)


def test_frozen_material_point_gap_rate_matches_twist_finite_difference():
    a = geometry(); J, _, _, report = compile_geometry(**a)
    eps = 1e-7
    for j, feature in enumerate(report['features']):
        i = j//4; vel = a['velocities'][i]
        p = np.array(feature['point_world_m']); normal = np.array(feature['normal_on_body'])
        arm = p-a['frames'][i,:3,3]
        displaced = p+eps*(vel[:3]+np.cross(vel[3:], arm))
        fd = normal@(displaced-p)/eps
        assert fd == pytest.approx(J[j]@a['generalized_velocity'], abs=1e-11)


def test_free_coupon_empty_geometry_requires_no_contacts():
    a = geometry(coupon(held_contacts=False)); J,g,s,r = compile_geometry(**a)
    assert J.shape == (0,8) and g.size == s.size == 0
    assert r['normal_rows'] == 0
    a['native_rows'] = native_rows(coupon())
    with pytest.raises(ValueError, match='Unknown'): compile_geometry(**a)


def test_geometry_and_solver_contract_has_bound_features_and_spring_only_effort():
    a = geometry(); J,g,s,report = compile_geometry(**a)
    args = system(contacts=12)
    args.update(J=J, g=g, s=s, feature_observed=report['feature_observed'],
                v=a['generalized_velocity'], law=SIGNED)
    result = solve(**args)
    assert result['status'] == 'resolved' and result['feature_coverage_bound']
    assert len(result['tau_joint']) == 2 and result['tau_spring'][:6] == [0.]*6
    assert not result['contact_force_applied'] and not result['native_contact_law_parity']


def test_free_geometry_empty_mask_is_valid_for_solver():
    a = geometry(coupon(held_contacts=False)); J,g,s,report = compile_geometry(**a)
    args = system(); args.update(J=J,g=g,s=s,feature_observed=report['feature_observed'])
    assert solve(**args)['status'] == 'resolved'


def test_missing_reference_feature_not_rescued_by_current_separating_motion():
    a = geometry(); frames = a['frames']
    frames[:, 0, 3] += 49.9e-6
    # Compare the initial reference correctly; the current finite geometry
    # cannot compensate for an omitted originally penetrating normal row.
    a['native_geometry_frames'] = layout(a['coupon'])['frames']
    a['native_geometry_step_id'] = 0
    a['nativeJac'] = _kinematic_jacobian(frames)
    a['velocities'] = np.einsum('bij,j->bi', a['nativeJac'], a['generalized_velocity'])
    a['native_rows'] = a['native_rows'][1:]
    with pytest.raises(ValueError, match='Missing penetrating'):
        compile_geometry(**a)


@pytest.mark.parametrize('bad', ['point', 'normal', 'separation', 'missing_separation',
                                'unknown', 'duplicate', 'missing', 'nan', 'overflow'])
def test_native_feature_comparison_is_required_and_fails_closed(bad):
    a = geometry(); rows = a['native_rows']
    if bad == 'point': rows[0]['point_world_m'][2] += POINT_TOL_M*2
    if bad == 'normal': rows[0]['normal_on_0'] = [0.,1.,0.]
    if bad == 'separation': rows[0]['separation_m'] += 1e-5
    if bad == 'missing_separation': del rows[0]['separation_m']
    if bad == 'unknown': rows[0]['collider1'] = '/World/Other'
    if bad == 'duplicate': rows.append(deepcopy(rows[0]))
    if bad == 'missing': del rows[0]
    if bad == 'nan': rows[0]['separation_m'] = np.nan
    if bad == 'overflow': a['native_rows'] = [deepcopy(rows[0]) for _ in range(MAX_ROWS+1)]
    with pytest.raises((ValueError, KeyError)): compile_geometry(**a)


@pytest.mark.parametrize('bad', ['scale', 'reflection', 'jac', 'velocity', 'stale', 'unstamped',
                                'cap', 'corner', 'anchor', 'material'])
def test_geometry_binding_rejects_unsupported_conventions_and_features(bad):
    a = geometry()
    if bad == 'scale': a['frames'][0,0,0] *= 2
    if bad == 'reflection': a['frames'][0,:3,0] *= -1
    if bad == 'jac': a['nativeJac'][0,0,0] = -1
    if bad == 'velocity': a['velocities'][0,0] += .1
    if bad == 'stale': a['step_id'] = 3; a['native_geometry_step_id'] = 0
    if bad == 'unstamped': a['native_geometry_step_id'] = 0
    if bad in ('cap', 'corner'):
        a['frames'][:, 2 if bad == 'cap' else 1, 3] += .004 if bad == 'cap' else .006
        a['nativeJac'] = _kinematic_jacobian(a['frames'])
        a['velocities'] = np.einsum('bij,j->bi', a['nativeJac'], a['generalized_velocity'])
    if bad == 'anchor': a['frames'][1,0,3] += .001
    if bad == 'material': a['coupon'] = replace(a['coupon'], contact_model='rigid_control')
    with pytest.raises(ValueError): compile_geometry(**a)


def test_explicit_one_step_reference_refreshes_geometry_not_old_gaps_or_forces():
    a = geometry(); reference = a['frames'].copy()
    original = compile_geometry(**a)
    a['frames'][:,0,3] += 1e-6
    a['native_geometry_frames'] = reference; a['native_geometry_step_id'] = 0
    a['nativeJac'] = _kinematic_jacobian(a['frames'])
    a['velocities'] = np.einsum('bij,j->bi', a['nativeJac'], a['generalized_velocity'])
    result = compile_geometry(**a)
    assert not np.array_equal(result[1], original[1])
    assert result[3]['matches'] == original[3]['matches']
    assert not result[3]['impulse_replayed']
    for row in a['native_rows']:
        row['impulse_on_0_ns'] = (np.array(row['impulse_on_0_ns'])*-1000).tolist()
    changed = compile_geometry(**a)
    for x,y in zip(changed[:3],result[:3]): np.testing.assert_array_equal(x,y)


def test_friction_is_counted_but_never_added_to_prediction_or_contact_stiffness():
    a = geometry(); base = compile_geometry(**a)
    row = deepcopy(a['native_rows'][0]); row['kind'] = 'friction'
    row.pop('separation_m'); row.pop('normal_on_0')
    a['native_rows'].append(row)
    got = compile_geometry(**a)
    for x,y in zip(got[:3],base[:3]): np.testing.assert_array_equal(x,y)
    assert got[3]['friction_rows'] == 1 and not got[3]['friction_in_prediction']


def test_native21_point_regression_requires_persistent_segment_not_reclipped_edge():
    # Captured native21 step2, normal row5. Before-step Link_0 pose and raw
    # point/separation copied verbatim. Trace SHA256:
    # 8b0d3e83b2f1bb35fed418a8993b6d941b88b66ecc709bb2feda13eef1234ab3.
    c = coupon(); frames = layout(c)['frames']
    frames[0] = [
        [.9999910693326823, -5.7133950909036075e-10, -.0042262577865752595, 2.6359819457866251e-5],
        [5.7401493388093168e-10, 1., 6.3183520481010518e-10, -3.9923959971321921e-11],
        [.0042262577865752595, -6.3425549718401627e-10, .9999910693326823, -.030006051063537598],
        [0., 0., 0., 1.]]
    point = np.array([-.0028073135763406754, -3.486921515416519e-11, -.022034021094441414])
    feature = next(x for x in _features(c, frames) if x['id'].endswith('Pad_0_minus:z1'))
    assert np.linalg.norm(feature['point']-point) < 2e-8
    assert abs(feature['gap']-2.2485852241516113e-5) < 2e-8
    # Source anchor remains at the finite face edge even when the representative
    # contact point is not there. No expanded geometry or tolerance required.
    pad = layout(c)['pads'][feature['pad']]
    anchor_local = pad[:3,:3].T@(feature['source_pad_anchor']-pad[:3,3])
    point_local = pad[:3,:3].T@(feature['point']-pad[:3,3])
    assert anchor_local[2] == pytest.approx(.008, abs=1e-16)
    assert abs(point_local[2]-.008) > 6e-6


def test_cached_anchor_advects_with_body_but_radius_offset_follows_world_normal():
    c = coupon(); frames = layout(c)['frames']
    before = _features(c, frames)[0]
    frames[0,:3,:3] = frames[0,:3,:3]@_ry(.01)
    frames[:,2,3] -= 5e-6
    after = _features(c, frames)[0]
    radius = float(np.float32(.0028))
    local_centerline = frames[0,:3,:3].T@(after['point']+radius*after['normal']-frames[0,:3,3])
    np.testing.assert_allclose(local_centerline, [0.,0.,before['cylinder_axis_coordinate_m']], atol=1e-16)
    np.testing.assert_array_equal(after['source_pad_anchor'], before['source_pad_anchor'])
