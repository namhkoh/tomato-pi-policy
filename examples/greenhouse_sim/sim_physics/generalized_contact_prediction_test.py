"""Pure bounded generalization; no native binding, jobs or force application."""
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from sim_physics.contact_coupled_prediction import MaterialLaw, solve as normal_solve
from sim_physics.contact_patch_prediction import (
    GENERALIZED_API, GENERALIZED_MODEL, MODEL, TOLERANCE, solve, solve_generalized,
)
from sim_physics.contact_spring_probe import Coupon

LAW = MaterialLaw('unilateral_kv_v1')


def system(n=8, root=6, groups=(2,)):
    m = sum(groups); p = len(groups)
    T = np.zeros((p, 2, 2, n))
    T[:, :, 0, min(1, n-1)] = 1.
    if n > 2:
        T[:, :, 1, 2] = 1.
    J = np.zeros((m, n)); J[:, 0] = 1.
    indices = []; offset = 0
    for count in groups:
        indices.append(list(range(offset, offset+count))); offset += count
    return dict(M=np.eye(n), q=np.zeros(n), v=np.zeros(n),
        K=np.r_[np.zeros(root), np.full(n-root, 3.)],
        C=np.r_[np.zeros(root), np.full(n-root, .2)],
        J=J, g=np.full(m, -.01), s=np.zeros(m),
        kc=np.full(m, 100.), dc=np.ones(m), h=.01,
        f=np.r_[-float(m), np.zeros(n-1)], law=LAW, root_dofs=root,
        feature_observed=[True]*m, patches=dict(model=MODEL,
            normal_indices=indices, tangent_jacobians=T.tolist(),
            surface_speeds_m_s=np.zeros((p, 2, 2)).tolist(),
            mu=.5, observed=[True]*p))


def pure_normal(a):
    return normal_solve(**{k: v for k, v in a.items() if k != 'patches'})


def coupon_call(a):
    return solve(**{k: v for k, v in a.items() if k != 'root_dofs'})


@pytest.mark.parametrize('n,root', [(1, 0), (5, 0), (6, 0), (6, 6),
    (8, 6), (17, 0), (20, 6), (64, 0), (64, 6)])
def test_free_exact_normal_solver_including_all_six_unactuated_root_coordinates(n, root):
    a = system(n, root, ())
    a['q'][root:] = .01
    a['v'] = np.arange(n)*.001
    a['f'] = np.arange(n)*.002
    if n > 1:
        a['M'][0, -1] = a['M'][-1, 0] = .2
    expected = pure_normal(a); result = solve_generalized(**a)
    assert result['status'] == 'resolved'
    np.testing.assert_array_equal(result['v_pred'], expected['v_pred'])
    np.testing.assert_array_equal(result['tau_spring'], expected['tau_spring'])
    assert result['tau_spring'][:root] == [0.]*root
    assert len(result['tau_joint']) == n-root
    assert result['friction_forces_n'] == []
    assert result['algebra_api'] == GENERALIZED_API
    assert result['root_dofs'] == root
    assert result['floating_root_present'] == (root == 6)
    assert not result['native_qualified']
    assert not result['native_geometry_authenticated']
    assert not result['contact_force_applied']
    assert not result['friction_force_applied']
    assert not result['measured_friction_used']
    assert not result['previous_predicted_seed_used']
    assert 'next_predicted_seed' not in result


@pytest.mark.parametrize('n,root', [(3, 0), (6, 6), (8, 6), (42, 6), (64, 6)])
@pytest.mark.parametrize('groups', [(1,), (3,), (4,), (1, 2, 3, 4), (2,)*7, (4,)*16])
def test_variable_patches_partition_and_two_anchors_share_all_normal_support(n, root, groups):
    a = system(n, root, groups)
    a['f'][1] = .1*sum(groups)
    r = solve_generalized(**a)
    assert r['status'] == 'resolved'
    np.testing.assert_allclose(r['v_pred'], 0, atol=3e-15)
    np.testing.assert_allclose(r['normal_forces_n'], 1, atol=2e-13)
    np.testing.assert_allclose(r['anchor_limits_n'],
        np.repeat((np.array(groups)*.25)[:, None], 2, axis=1), atol=2e-13)
    assert np.asarray(r['friction_forces_n']).shape == (len(groups), 2, 2)
    assert r['tau_spring'][:root] == [0.]*root
    assert r['relative_fixed_point_residual'] <= TOLERANCE
    assert r['relative_equilibrium_residual'] <= TOLERANCE
    assert np.all(np.array(r['anchor_relative_power_w'])
                  <= np.array(r['anchor_power_error_bound_w']))


def test_many_coordinates_coupled_mass_and_distinct_anchor_moment_no_root_pin():
    a = system(42, 6, (1, 3))
    # Kinematic tangents depend on a remote elastic coordinate: root gravity
    # support simultaneously balances its actual elastic torque.
    T = np.array(a['patches']['tangent_jacobians'])
    T[:, :, 0, 40] = .5
    a['patches']['tangent_jacobians'] = T.tolist()
    a['M'][0, 40] = a['M'][40, 0] = .2
    a['q'][40] = .1; a['f'][1] = -.6
    r = solve_generalized(**a)
    assert r['status'] == 'resolved'
    np.testing.assert_allclose(r['v_pred'], 0, atol=3e-14)
    assert r['tau_joint'][34] == pytest.approx(-.3, abs=1e-13)
    assert r['tau_spring'][:6] == [0.]*6
    # A root direction not represented in J/T remains free, not welded.
    a['v'][5] = .3
    r = solve_generalized(**a)
    assert r['v_pred'][5] == pytest.approx(.3, abs=1e-13)


def test_one_coordinate_rank_one_tangent_response_supported_without_fake_second_dof():
    a = system(1, 0, (1,))
    a['J'][:] = 0.; a['f'][:] = .2; a['q'][:] = 0.
    r = solve_generalized(**a)
    assert r['status'] == 'resolved'
    np.testing.assert_allclose(r['v_pred'], 0, atol=1e-15)
    np.testing.assert_allclose(r['friction_forces_n'], [[[-.1, 0], [-.1, 0]]], atol=1e-13)


@pytest.mark.parametrize('groups', [(1,), (3,), (4,), (1, 2, 3, 4)])
def test_variable_normal_counts_in_sliding_branch_keep_original_cone_and_equilibrium(groups):
    a = system(12, 6, groups)
    m = sum(groups)
    a['g'] = -np.arange(1, m+1)*.001
    # Cross coupling makes normal load depend on tangent motion: this exercises
    # every normal-feature column in the cone-radius derivative, not just KKT.
    a['J'][:, 1] = .1
    a['f'][0] = -float(np.sum(-100*a['g']))
    a['f'][1] = 2*m
    r = solve_generalized(**a)
    assert r['status'] == 'resolved'
    assert r['all_stick_candidate']['status'] == 'rejected'
    assert any(mode == 'slip' for pair in r['friction_modes'] for mode in pair)
    N = np.array(r['normal_forces_n'])
    expected = [a['patches']['mu']*np.sum(N[ids])/2 for ids in a['patches']['normal_indices']]
    np.testing.assert_allclose(r['anchor_limits_n'], np.repeat(np.array(expected)[:, None], 2, axis=1))
    assert r['relative_fixed_point_residual'] <= TOLERANCE
    assert r['relative_equilibrium_residual'] <= TOLERANCE
    assert np.all(np.array(r['anchor_relative_power_w']) <= np.array(r['anchor_power_error_bound_w']))


@pytest.mark.parametrize('surface_speed,mode,expected_velocity', [
    (.005, 'stick', .005), (.1, 'slip', .01), (-.1, 'slip', -.01)])
def test_moving_surface_relative_power_not_false_lab_frame_passivity(surface_speed, mode, expected_velocity):
    a = system(6, 6)
    speeds = np.zeros((1, 2, 2)); speeds[:, :, 0] = surface_speed
    a['patches']['surface_speeds_m_s'] = speeds.tolist()
    r = solve_generalized(**a)
    assert r['status'] == 'resolved'
    assert r['v_pred'][1] == pytest.approx(expected_velocity, abs=1e-12)
    assert r['friction_modes'] == [[mode, mode]]
    force = np.array(r['friction_forces_n']).reshape(2, 2)
    lab_speed = np.tile(np.array(r['v_pred'])[1:3], (2, 1))
    relative_speed = lab_speed-speeds.reshape(2, 2)
    np.testing.assert_allclose(r['tangent_speeds_m_s'], relative_speed[None], atol=1e-14)
    np.testing.assert_allclose(r['anchor_relative_power_w'],
        np.einsum('ij,ij->i', force, relative_speed)[None], atol=1e-15)
    assert np.sum(force*lab_speed) > 0  # Surface can do physical work.
    assert np.all(np.array(r['anchor_relative_power_w'])
                  <= np.array(r['anchor_power_error_bound_w']))


def test_galilean_relative_velocity_invariance_no_measured_force():
    a = system(6, 6, (1, 3)); a['f'][1] = 3.
    base = solve_generalized(**a)
    shifted = deepcopy(a)
    speed = np.array([.03, -.02, .04, 0, 0, 0])
    shifted['v'] += speed
    shifted['s'] += a['J']@speed
    shifted['patches']['surface_speeds_m_s'] = (
        np.array(a['patches']['tangent_jacobians'])@speed).tolist()
    result = solve_generalized(**shifted)
    assert result['status'] == base['status'] == 'resolved'
    np.testing.assert_allclose(result['v_pred'], np.array(base['v_pred'])+speed, atol=2e-13)
    np.testing.assert_allclose(result['normal_forces_n'], base['normal_forces_n'], atol=2e-12)
    np.testing.assert_allclose(result['friction_forces_n'], base['friction_forces_n'], atol=2e-12)
    np.testing.assert_allclose(result['tau_spring'], base['tau_spring'], atol=2e-13)


def test_iteration_exhaustion_and_unobserved_features_emit_no_effort():
    a = system(); a['f'][1] = 2.
    r = solve_generalized(**a, max_iterations=1)
    assert r['status'] == 'unresolved' and r['reason'] == 'iteration_limit'
    assert r['tau_spring'] is r['tau_joint'] is r['v_pred'] is None
    for mask in ('normal', 'patch'):
        a = system()
        (a['feature_observed'] if mask == 'normal' else a['patches']['observed'])[0] = False
        r = solve_generalized(**a)
        assert r['status'] == 'unresolved'
        assert r['tau_spring'] is r['v_pred'] is None


@pytest.mark.parametrize('what', ['mu0', 'open'])
def test_unsupported_tangents_do_not_create_connection(what):
    a = system(12, 6, (1, 3)); a['v'][1] = .5
    if what == 'mu0':
        a['patches']['mu'] = 0.
    else:
        a['g'][:] = .1; a['f'][0] = 0.; a['v'][0] = 1.
    r = solve_generalized(**a); expected = pure_normal(a)
    assert r['status'] == 'resolved'
    np.testing.assert_allclose(r['v_pred'], expected['v_pred'], atol=1e-13)
    np.testing.assert_allclose(r['tau_spring'], expected['tau_spring'], atol=1e-13)
    np.testing.assert_allclose(r['friction_forces_n'], 0, atol=1e-13)


@pytest.mark.parametrize('root', [True, 1, 3, 5, 7, -1, 6., None])
def test_invalid_explicit_roots(root):
    a = system(); a['root_dofs'] = root
    with pytest.raises(ValueError):
        solve_generalized(**a)


@pytest.mark.parametrize('n,root', [(0, 0), (65, 6), (5, 6)])
def test_invalid_dimension_bounds(n, root):
    a = system(); a['q'] = np.zeros(n); a['root_dofs'] = root
    with pytest.raises(ValueError):
        solve_generalized(**a)


@pytest.mark.parametrize('key', ['K', 'C'])
@pytest.mark.parametrize('form', ['vector', 'symmetric', 'antisymmetric', 'nan'])
def test_exact_raw_root_zeros_before_matrix_symmetrization(key, form):
    a = system()
    if form == 'vector':
        a[key][0] = 1e-30
    else:
        a[key] = np.diag(a[key])
        a[key][0, 7] = 1e-30
        a[key][7, 0] = (-1e-30 if form == 'antisymmetric' else 1e-30)
        if form == 'nan':
            a[key][0, 0] = np.nan
    with pytest.raises(ValueError):
        solve_generalized(**a)


@pytest.mark.parametrize('what', [
    'duplicate', 'missing', 'negative', 'overflow', 'zero_features', 'five_features',
    'float_index', 'bool_index', 'scalar_groups', 'seventeen_patches', 'one_anchor', 'three_anchors',
    'wrong_n', 'nan_T', 'zero_T', 'nan_surface', 'complex_surface',
    'mu_negative', 'mu_nan', 'mu_bool', 'mu_vector', 'mu_string',
    'normal_none', 'normal_integer', 'patch_integer', 'signed_law',
    'context', 'measured_force', 'seed', 'bad_model',
    'singular_M', 'negative_K', 'nan_q', 'complex_J', 'kc_zero', 'dc_negative', 'dt_zero',
])
def test_invalid_geometry_material_coverage_and_forbidden_force_context(what):
    a = system()
    p = a['patches']
    if what == 'duplicate': p['normal_indices'] = [[0, 0]]
    elif what == 'missing': p['normal_indices'] = [[0]]
    elif what == 'negative': p['normal_indices'] = [[-1, 0]]
    elif what == 'overflow': p['normal_indices'] = [[0, 99]]
    elif what == 'zero_features': p['normal_indices'] = [[]]
    elif what == 'five_features': p['normal_indices'] = [[0, 1, 2, 3, 4]]
    elif what == 'float_index': p['normal_indices'] = [[0., 1.]]
    elif what == 'bool_index': p['normal_indices'] = [[False, True]]
    elif what == 'scalar_groups': p['normal_indices'] = np.array(1)
    elif what == 'seventeen_patches': p['normal_indices'] = [[i] for i in range(17)]
    elif what in ('one_anchor', 'three_anchors', 'wrong_n', 'nan_T', 'zero_T'):
        T = np.array(p['tangent_jacobians'])
        if what == 'one_anchor': T = T[:, :1]
        elif what == 'three_anchors': T = np.repeat(T[:, :1], 3, axis=1)
        elif what == 'wrong_n': T = T[..., :7]
        elif what == 'nan_T': T[0, 0, 0, 0] = np.nan
        else: T[:] = 0
        p['tangent_jacobians'] = T
    elif what in ('nan_surface', 'complex_surface'):
        p['surface_speeds_m_s'] = np.full((1, 2, 2), np.nan if what == 'nan_surface' else 1j)
    elif what.startswith('mu_'):
        p['mu'] = {'negative': -.1, 'nan': np.nan, 'bool': True,
                   'vector': [.5], 'string': '.5'}[what[3:]]
    elif what == 'normal_none': a['feature_observed'] = None
    elif what == 'normal_integer': a['feature_observed'] = [1, 1]
    elif what == 'patch_integer': p['observed'] = [1]
    elif what == 'signed_law': a['law'] = MaterialLaw('signed_overlap_kv_v1')
    elif what == 'context': p['prediction_context'] = {}
    elif what == 'measured_force': p['friction_forces_n'] = [[[0, 0], [0, 0]]]
    elif what == 'seed': p['previous_predicted_seed'] = None
    elif what == 'bad_model': p['model'] = 'native_certified'
    elif what == 'singular_M': a['M'][7, 7] = 0.
    elif what == 'negative_K': a['K'][7] = -1.
    elif what == 'nan_q': a['q'][7] = np.nan
    elif what == 'complex_J': a['J'] = a['J'].astype(complex)+1j
    elif what == 'kc_zero': a['kc'][0] = 0.
    elif what == 'dc_negative': a['dc'][0] = -.1
    elif what == 'dt_zero': a['h'] = 0.
    with pytest.raises(ValueError):
        solve_generalized(**a)


def test_cold_only_signature_and_inputs_not_mutated():
    a = system(20, 6, (1, 3))
    before = deepcopy(a)
    first = solve_generalized(**a); second = solve_generalized(**a)
    assert json.dumps(first, sort_keys=True, allow_nan=False) == json.dumps(second, sort_keys=True, allow_nan=False)
    for key in a:
        if key not in ('patches', 'law'):
            np.testing.assert_array_equal(a[key], before[key])
    assert a['patches'] == before['patches']
    for field in ('previous_predicted_seed', 'prediction_context', 'observed_friction_force'):
        with pytest.raises(TypeError):
            solve_generalized(**a, **{field: None})


@pytest.mark.parametrize('what', ['moving', 'n64', 'root0', 'seven', 'one_feature'])
def test_legacy_coupon_validation_not_broadened(what):
    if what == 'moving':
        a = system(); a['patches']['surface_speeds_m_s'] = [[[.1, 0], [.1, 0]]]
    elif what == 'n64': a = system(64, 6)
    elif what == 'root0':
        a = system(8, 0); a['q'][:] = .1
    elif what == 'seven': a = system(groups=(2,)*7)
    else: a = system(groups=(1,))
    with pytest.raises(ValueError):
        coupon_call(a)


def test_eight_coordinate_complete_shared_kernel_outputs_equal():
    for p in (0, 1, 3, 6):
        for drive in (0., .4, 2.):
            a = system(groups=(2,)*p)
            a['f'][1] = drive
            generalized = solve_generalized(**a)
            legacy = coupon_call(a)
            for key, value in legacy.items():
                if key != 'next_predicted_seed':
                    assert generalized[key] == value, key



def actual_anchors(a, counts):
    """Synthetic source geometry with no zero padding or duplicated anchors."""
    assert len(counts) == len(a['patches']['normal_indices'])
    a['patches']['model'] = GENERALIZED_MODEL
    a['patches']['tangent_jacobians'] = [
        np.array(block)[:count].tolist()
        for block, count in zip(a['patches']['tangent_jacobians'], counts)]
    a['patches']['surface_speeds_m_s'] = [
        np.array(block)[:count].tolist()
        for block, count in zip(a['patches']['surface_speeds_m_s'], counts)]
    return a


@pytest.mark.parametrize('counts,groups', [
    ((1,), (1,)), ((1,), (4,)), ((2,), (1,)), ((1, 2), (1, 4)),
    ((2, 1, 2), (3, 1, 2)), ((2, 2, 2, 2, 2, 1), (2, 2, 2, 2, 2, 1)),
    ((1,)*16, (4,)*16),
])
@pytest.mark.parametrize('drive_per_normal', [.1, 2.])
def test_actual_one_or_two_anchor_support_share_and_ragged_outputs(counts, groups, drive_per_normal):
    # Includes the reported native58 six-pair count SHAPE, not measured geometry.
    a = actual_anchors(system(45, 6, groups), counts)
    a['f'][1] = drive_per_normal*sum(groups)
    before = deepcopy(a['patches'])
    r = solve_generalized(**a)
    assert r['status'] == 'resolved'
    assert r['model'] == GENERALIZED_MODEL and r['anchor_counts'] == list(counts)
    assert not r['native_contact_law_parity'] and not r['native_qualified']
    N = np.array(r['normal_forces_n'])
    for k, (count, ids) in enumerate(zip(counts, a['patches']['normal_indices'])):
        cap = .5*np.sum(N[ids])/count
        np.testing.assert_allclose(r['anchor_limits_n'][k], [cap]*count, atol=1e-12)
        for key in ('friction_forces_n', 'tangent_speeds_m_s'):
            assert np.array(r[key][k]).shape == (count, 2)
        for key in ('friction_modes', 'anchor_relative_power_w', 'anchor_power_error_bound_w'):
            assert len(r[key][k]) == count
        assert np.all(np.array(r['anchor_relative_power_w'][k])
                      <= np.array(r['anchor_power_error_bound_w'][k]))
    assert r['relative_equilibrium_residual'] <= TOLERANCE
    assert r['relative_fixed_point_residual'] <= TOLERANCE
    assert len(r['tau_spring']) == 45 and r['tau_spring'][:6] == [0.]*6
    assert a['patches'] == before  # No fabricated second anchor added in-place.
    if drive_per_normal > .5:
        assert r['all_stick_candidate']['status'] == 'rejected'


def test_single_anchor_receives_full_patch_bound_not_half_with_fake_twin():
    a = actual_anchors(system(6, 6, (2,)), (1,))
    a['f'][1] = .75
    r = solve_generalized(**a)
    assert r['status'] == 'resolved'
    np.testing.assert_allclose(r['v_pred'], 0, atol=1e-14)
    np.testing.assert_allclose(r['friction_forces_n'], [[[-.75, 0.]]], atol=1e-13)
    np.testing.assert_allclose(r['anchor_limits_n'], [[1.]], atol=1e-13)
    assert r['friction_modes'] == [['stick']]
    assert r['anchor_counts'] == [1]


def test_single_physical_anchor_does_not_invent_opposing_moment_anchor():
    a = actual_anchors(system(6, 6, (2,)), (1,))
    T = np.array(a['patches']['tangent_jacobians'])
    T[0, 0, 0, 4] = -.008
    a['patches']['tangent_jacobians'] = T.tolist()
    a['f'][4] = .004
    r = solve_generalized(**a)
    assert r['status'] == 'resolved'
    # A single point cannot independently cancel pure torque with a fictitious
    # opposing tangent force. Its no-slip constraint permits coupled rotation.
    assert r['v_pred'][4] > 3.9e-5
    assert r['v_pred'][1] > 0
    assert len(r['friction_forces_n'][0]) == 1
    assert r['relative_equilibrium_residual'] <= TOLERANCE


def test_one_anchor_rank_one_coordinate_model_is_not_padded():
    a = actual_anchors(system(1, 0, (1,)), (1,))
    a['J'][:] = 0.; a['f'][:] = .2
    r = solve_generalized(**a)
    assert r['status'] == 'resolved'
    np.testing.assert_allclose(r['friction_forces_n'], [[[-.2, 0]]], atol=1e-13)
    assert r['anchor_counts'] == [1]


@pytest.mark.parametrize('moving', [False, True])
def test_mixed_anchor_coupled_normal_cap_derivative_and_relative_motion(moving):
    a = actual_anchors(system(12, 6, (1, 3, 4)), (1, 2, 1))
    a['g'] = -np.arange(1, 9)*.002
    a['J'][:, 1] = .1
    a['f'][0] = float(np.sum(100*a['g'])); a['f'][1] = 10.
    if moving:
        a['patches']['surface_speeds_m_s'] = [
            [[.1, -.02]], [[-.02, .04], [.03, -.01]], [[.05, .01]]]
    r = solve_generalized(**a)
    assert r['status'] == 'resolved'
    assert r['all_stick_candidate']['status'] == 'rejected'
    N = np.array(r['normal_forces_n'])
    for k, ids in enumerate(a['patches']['normal_indices']):
        cap = .5*np.sum(N[ids])/r['anchor_counts'][k]
        np.testing.assert_allclose(r['anchor_limits_n'][k], cap, atol=1e-12)
        assert np.all(np.array(r['anchor_relative_power_w'][k])
                      <= np.array(r['anchor_power_error_bound_w'][k]))
    assert r['relative_equilibrium_residual'] <= TOLERANCE
    assert r['relative_fixed_point_residual'] <= TOLERANCE


def test_mixed_ragged_surface_speeds_preserve_source_anchor_order_and_can_supply_work():
    a = actual_anchors(system(6, 6, (1, 2)), (1, 2))
    # Two patches have opposing surface velocities. Same-point duplicate
    # placeholders are never created, and each actual anchor has its own speed.
    a['patches']['surface_speeds_m_s'] = [[[.02, 0]], [[-.01, 0], [-.03, 0]]]
    r = solve_generalized(**a)
    assert r['status'] == 'resolved'
    for k, (T, surface) in enumerate(zip(a['patches']['tangent_jacobians'],
                                       a['patches']['surface_speeds_m_s'])):
        expected = np.array(T)@np.array(r['v_pred'])-surface
        np.testing.assert_allclose(r['tangent_speeds_m_s'][k], expected, atol=1e-14)
    assert r['anchor_counts'] == [1, 2]


@pytest.mark.parametrize('bad', ['zero', 'three', 'missing_speed', 'extra_speed',
    'missing_patch', 'nan_tangent', 'nan_speed', 'complex_speed', 'legacy_name'])
def test_variable_anchor_invalid_counts_matching_finite_speeds_and_explicit_model(bad):
    a = actual_anchors(system(8, 6, (1, 2)), (1, 2))
    p = a['patches']
    if bad == 'zero':
        p['tangent_jacobians'][0] = []; p['surface_speeds_m_s'][0] = []
    elif bad == 'three':
        p['tangent_jacobians'][0] *= 3; p['surface_speeds_m_s'][0] *= 3
    elif bad == 'missing_speed': p['surface_speeds_m_s'][1] = [[0, 0]]
    elif bad == 'extra_speed': p['surface_speeds_m_s'][0] = [[0, 0], [0, 0]]
    elif bad == 'missing_patch': p['tangent_jacobians'].pop()
    elif bad == 'nan_tangent': p['tangent_jacobians'][0][0][0][0] = np.nan
    elif bad == 'nan_speed': p['surface_speeds_m_s'][0][0][0] = np.nan
    elif bad == 'complex_speed': p['surface_speeds_m_s'][0][0][0] = 1j
    elif bad == 'legacy_name': p['model'] = MODEL
    with pytest.raises(ValueError):
        solve_generalized(**a)


def test_explicit_generalized_model_and_anchor_count_do_not_broaden_legacy_coupon():
    a = actual_anchors(system(), (1,))
    with pytest.raises(ValueError):
        coupon_call(a)
    a = actual_anchors(system(), (2,))
    with pytest.raises(ValueError):
        coupon_call(a)


def test_variable_anchor_observation_coverage_and_iteration_exhaustion_remain_fail_closed():
    a = actual_anchors(system(45, 6, (2, 2, 2, 2, 2, 1)), (2, 2, 2, 2, 2, 1))
    a['patches']['observed'][-1] = False
    r = solve_generalized(**a)
    assert r['status'] == 'unresolved' and r['tau_spring'] is None
    assert r['anchor_counts'] == [2, 2, 2, 2, 2, 1]
    a = actual_anchors(system(8, 6, (1,)), (1,))
    a['f'][1] = 2.
    r = solve_generalized(**a, max_iterations=1)
    assert r['status'] == 'unresolved' and r['reason'] == 'iteration_limit'
    assert r['v_pred'] is r['tau_spring'] is r['tau_joint'] is None


def test_uniform_two_anchor_new_model_has_identical_numeric_kernel_results():
    for p in (0, 1, 6):
        for drive in (0., .4, 2.):
            a = system(groups=(2,)*p); a['f'][1] = drive
            legacy = coupon_call(a)
            actual_anchors(a, (2,)*p)
            generalized = solve_generalized(**a)
            for key, value in legacy.items():
                if key not in ('next_predicted_seed', 'model', 'normal_support_basis'):
                    assert generalized[key] == value, key



def test_optional_immutable_native31_all_1920_cold_kernel_results_bitwise():
    # Archive is deliberately ignored, not required in portable source checkouts.
    root = Path(__file__).resolve().parents[3]
    folder = root/'data/sim_physics/contact_spring_native_20260911_31'
    if not (folder/'trace.json').is_file():
        pytest.skip('Optional immutable native31 archive unavailable')
    raw = (folder/'trace.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == '55334886f3dbcf6e7b9c4f908d394f4b6d248f4d7dd71db020881a81c4e91c3c'
    rows = json.loads(raw)
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    c = Coupon(**{x.name: report['configuration'][x.name] for x in fields(Coupon)})
    assert len(rows) == 1920
    for step, row in enumerate(rows, 1):
        pre = row['native_prediction_before_step']; geom = row['coupled_prediction']['geometry']
        frames = np.array(pre['frames_world_m']); jac = np.array(pre['body_world_com_jacobians'])
        J = []; g = []
        for x in geom['features']:
            i = int(x['collider'].split('/Link_')[1].split('/')[0])
            arm = np.array(x['point_world_m'])-frames[i, :3, 3]
            point_jac = jac[i, :3]+np.cross(jac[i, 3:].T, arm).T
            J.append(np.array(x['normal_on_body'])@point_jac); g.append(x['gap_m'])
        patches = {key: value for key, value in geom['patches'].items()
                   if key != 'prediction_context'}
        a = dict(M=pre['mass_matrix'],
            q=np.r_[np.zeros(6), report['initial_readback']['q'] if step == 1 else rows[step-2]['q_rad']],
            v=pre['generalized_velocity'], K=np.r_[np.zeros(6), c.stiffness],
            C=np.r_[np.zeros(6), c.damping], J=np.array(J), g=g, s=np.zeros(len(g)),
            kc=np.full(len(g), c.contact_stiffness), dc=np.full(len(g), c.contact_damping),
            h=c.dt, f=pre['native_known_external_force'], law=LAW,
            feature_observed=geom['feature_observed'], patches=patches, root_dofs=6)
        legacy = coupon_call(a); generalized = solve_generalized(**a)
        assert legacy['status'] == generalized['status'] == 'resolved', step
        # Canonical float JSON preserves exact kernel values, including iteration
        # and residual diagnostics, not merely a tolerant force comparison.
        expected = {k: v for k, v in legacy.items() if k != 'next_predicted_seed'}
        actual = {k: generalized[k] for k in expected}
        assert json.dumps(actual, sort_keys=True, allow_nan=False) == json.dumps(
            expected, sort_keys=True, allow_nan=False), step
