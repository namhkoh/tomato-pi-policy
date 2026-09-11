"""Fresh compiler tests: real A witness helper, no native application."""
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from . import grasp_contact_fresh as fresh
from .grasp_contact_geometry_test import fixture
from .contact_patch_prediction import GENERALIZED_MODEL, solve_generalized
from .contact_coupled_prediction import MaterialLaw

compile_contacts = fresh.compile_contacts


def test_isolated_import_is_pure():
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    result = subprocess.run([sys.executable, '-B', '-c',
        'import sys; import sim_physics.grasp_contact_fresh; '
        'assert not any(n.startswith(("pxr", "omni", "isaacsim")) for n in sys.modules)'],
        env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_one_generated_witness_not_two_native_points_and_explicit_coverage():
    a = fixture(); J, g, s, r = compile_contacts(**a)
    assert J.shape == (1, 7)
    np.testing.assert_allclose(g, [.0002], atol=1e-16)
    np.testing.assert_array_equal(s, [0.])
    feature = r['features'][0]
    np.testing.assert_allclose(feature['point_world_m'], [-.003, 0, 0], atol=1e-16)
    np.testing.assert_allclose(feature['box_witness_world_m'], [-.0032, 0, 0], atol=1e-16)
    assert feature['parameter'] == .5
    assert feature['capsule_feature'] == 'cylinder' and feature['box_feature'] == 'face'
    assert r['patches']['normal_indices'] == [[0]]
    assert np.array(r['patches']['tangent_jacobians']).shape == (1, 1, 2, 7)
    assert r['patches']['model'] == GENERALIZED_MODEL
    assert r['feature_covered'] == [True] and r['feature_observed'] == [False]
    assert r['patches']['observed'] == [True] and r['patch_covered'] == [True]
    assert r['features_generated'] and not r['native_features_observed']
    assert r['coverage_semantics'] == r['patches_observed_semantics'] == fresh.COVERAGE
    assert r['observed_pairs'][0]['native_normal_row_count'] == 2
    assert r['observed_pairs'][0]['native_friction_row_count'] == 2
    assert r['observed_pairs'][0]['native_pair_observed']
    assert 'per observed pair, not per native row' in r['contact_coefficient_interpretation']
    assert not r['native_contact_law_parity'] and not r['actuation_authorized']
    json.dumps(r, allow_nan=False)


def test_geometry_and_impulses_are_never_accessed_even_if_poisoned():
    class IdentityOnly(dict):
        def __getitem__(self, key):
            assert key in ('collider0', 'collider1', 'kind'), key
            return super().__getitem__(key)
        def get(self, key, *args):
            assert key in ('collider0', 'collider1', 'kind'), key
            return super().get(key, *args)
    a = fixture(); expected = compile_contacts(**a)
    a['rows'] = [IdentityOnly(dict(row, point_world_m=[np.nan]*3,
        normal_on_0=['not geometry'], impulse_on_0_ns=object(), separation_m=np.inf))
        for row in a['rows']]
    actual = compile_contacts(**a)
    for x, y in zip(expected[:3], actual[:3], strict=True):
        np.testing.assert_array_equal(x, y)
    assert actual[3] == expected[3]
    assert not actual[3]['native_contact_geometry_read']
    assert not actual[3]['contact_impulses_read']


@pytest.mark.parametrize('kind', ['normal', 'friction'])
def test_pair_identity_alone_can_generate_model_geometry_not_observed_features(kind):
    a = fixture()
    a['rows'] = [dict(collider0=a['chain'][0].collider, collider1=a['pads'][0].collider, kind=kind)]
    _, _, _, r = compile_contacts(**a)
    assert len(r['features']) == 1 and r['feature_observed'] == [False]
    assert r['observed_pairs'][0]['native_'+kind+'_row_count'] == 1
    assert not r['native_completeness_verified']


def test_native_row_multiplicity_never_scales_modeled_stiffness_or_feature_count():
    a = fixture(); base = compile_contacts(**a)
    a['rows'] *= 20
    actual = compile_contacts(**a)
    for x, y in zip(base[:3], actual[:3], strict=True):
        np.testing.assert_array_equal(x, y)
    assert actual[3]['patches'] == base[3]['patches']
    assert actual[3]['features'] == base[3]['features']
    assert actual[3]['observed_pairs'][0]['native_normal_row_count'] == 40
    assert actual[3]['model_normal_points_per_pair'] == 1
    assert not actual[3]['native_point_count_stiffness_parity']


def test_original_header_order_does_not_change_fresh_geometry_or_force_sign():
    a = fixture(); expected = compile_contacts(**a)
    for row in a['rows']:
        row['collider0'], row['collider1'] = row['collider1'], row['collider0']
        if row['kind'] == 'normal':
            row['normal_on_0'] = [900., -300., 20.]  # Not consumed.
    actual = compile_contacts(**a)
    for x, y in zip(expected[:3], actual[:3], strict=True):
        np.testing.assert_array_equal(x, y)
    assert expected[3] == actual[3]


def test_reference_frames_do_not_transport_or_seed_fresh_features():
    a = fixture(); a['current']['step_id'] = 1
    expected = compile_contacts(**a)
    for key in ('body_frames_world', 'robot_body_frames_world'):
        for frame in a['reference'][key]:
            frame[0][3] += 1.; frame[1][3] -= .5
    a['reference']['robot_body_velocities_world'][0] = [10., 20., 30., 1., 2., 3.]
    actual = compile_contacts(**a)
    for x, y in zip(expected[:3], actual[:3], strict=True):
        np.testing.assert_array_equal(x, y)
    assert actual[3] == expected[3]


@pytest.mark.parametrize('kind,translation', [
    ('edge', [-.0115, -.0215, 0.]),
    ('corner', [-.0115, -.0215, -.0515]),
])
def test_fresh_source_edge_corner_and_endcap_are_explicitly_new_geometry(kind, translation):
    a = fixture()
    for i in range(3):
        a['current']['robot_body_frames_world'][0][i][3] = translation[i]
    _, g, _, r = compile_contacts(**a)
    x = r['features'][0]
    assert x['box_feature'] == kind and g[0] < 0
    direction = np.array([1., 1., 0.] if kind == 'edge' else [1., 1., 1.])
    direction /= np.linalg.norm(direction)
    np.testing.assert_allclose(x['normal_on_plant'], direction, atol=2e-15)
    if kind == 'corner':
        assert x['capsule_feature'] == 'start_cap' and x['parameter'] == 0
        assert x['point_world_m'][2] < -.02
    assert r['geometry_model'] == fresh.MODEL
    assert not x['feature_observed']


def test_entire_authored_box_not_forced_to_old_inner_face():
    a = fixture(); a['current']['robot_body_frames_world'][0][0][3] = .0132
    # face_sign remains +1; this new model uses actual finite box geometry.
    _, _, _, r = compile_contacts(**a)
    np.testing.assert_allclose(r['features'][0]['normal_on_plant'], [-1, 0, 0], atol=1e-15)


def test_axis_box_intersection_is_rejected_without_inventing_normal():
    a = fixture(); a['current']['robot_body_frames_world'][0][0][3] = 0.
    with pytest.raises(ValueError, match='Fresh pair.*distance zero'):
        compile_contacts(**a)


def test_nonzero_COM_and_rotating_surface_use_box_normal_and_common_tangent_witness():
    a = fixture()
    a['current']['robot_body_velocities_world'][0] = [.1, .2, .3, 0, 0, 2.]
    a['current']['robot_com_local_poses'][0][0] = .001
    a['current']['native_com_local_poses'][0][2] = .005
    J, _, s, r = compile_contacts(**a)
    np.testing.assert_allclose(s, [.1], atol=1e-16)
    np.testing.assert_allclose(J[0, 4], -.005, atol=1e-16)
    # Identity box: deterministic tangents are +Z, -Y. Angular surface
    # velocity differs by .0004 m/s between box and capsule witnesses.
    np.testing.assert_allclose(r['patches']['surface_speeds_m_s'], [[[.3, -.2184]]], atol=1e-15)
    x = r['features'][0]; basis = np.array(x['tangent_basis_world'])
    robot_com = np.array([-.0122, 0., 0.])
    v_at_box = np.array([.1, .2, .3])+np.cross([0, 0, 2.], np.array(x['box_witness_world_m'])-robot_com)
    assert abs((basis@v_at_box)[1]-r['patches']['surface_speeds_m_s'][0][0][1]) == pytest.approx(.0004)
    assert r['normal_surface_velocity_at'] == 'box_witness'
    assert r['friction_surface_velocity_at'] == 'common_capsule_witness'


def _advance(frame, com_local, velocity, dt):
    frame = np.array(frame)
    omega = np.array(velocity[3:]); angle = np.linalg.norm(omega)*dt
    if np.linalg.norm(omega):
        u = omega/np.linalg.norm(omega)
        cross = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
        increment = np.eye(3)+np.sin(angle)*cross+(1-np.cos(angle))*(cross@cross)
    else:
        increment = np.eye(3)
    com = frame[:3, 3]+frame[:3, :3]@com_local
    rotation = increment@frame[:3, :3]
    frame[:3, :3] = rotation
    frame[:3, 3] = com+np.array(velocity[:3])*dt-rotation@com_local
    return frame.tolist()


def test_normal_gap_dot_matches_moving_capsule_and_box_witness_finite_difference():
    a = fixture()
    for i, x in enumerate([-.0115, -.0215, -.0515]):
        a['current']['robot_body_frames_world'][0][i][3] = x
    plant_com = np.array([.001, -.0005, .002]); robot_com = np.array([-.001, .002, .001])
    a['current']['native_com_local_poses'][0][:3] = plant_com.tolist()
    a['current']['robot_com_local_poses'][0][:3] = robot_com.tolist()
    pv = np.array([.02, -.01, .03, .4, -.3, .2])
    rv = np.array([-.01, .025, .015, -.1, .5, .35])
    a['current']['generalized_velocity'] = np.r_[pv, 0.].tolist()
    a['current']['robot_body_velocities_world'][0] = rv.tolist()
    J, _, s, _ = compile_contacts(**a)
    expected = (J@a['current']['generalized_velocity']-s)[0]
    samples = []
    h = 1e-7
    for dt in (-h, h):
        shifted = deepcopy(a)
        shifted['current']['body_frames_world'][0] = _advance(
            a['current']['body_frames_world'][0], plant_com, pv, dt)
        shifted['current']['robot_body_frames_world'][0] = _advance(
            a['current']['robot_body_frames_world'][0], robot_com, rv, dt)
        samples.append(compile_contacts(**shifted)[1][0])
    assert (samples[1]-samples[0])/(2*h) == pytest.approx(expected, abs=1e-8)


def test_actual_local_frames_are_used_and_input_geometry_is_not_mutated(monkeypatch):
    a = fixture(); cap_local = np.eye(4); cap_local[:3, 3] = [.001, .002, .003]
    pad_local = np.eye(4); pad_local[2, 3] = .002
    a['chain'][0] = replace(a['chain'][0], local_frame=cap_local)
    a['pads'][0] = replace(a['pads'][0], local_frame=pad_local)
    seen = []
    real = fresh.closest_capsule_box
    def inspect(start, end, half, radius):
        seen.append((start.copy(), end.copy(), half.copy(), radius))
        return real(start, end, half, radius)
    monkeypatch.setattr(fresh, 'closest_capsule_box', inspect)
    before = deepcopy(a['current']); raw_rows = deepcopy(a['rows'])
    _, _, _, r = compile_contacts(**a)
    assert len(seen) == 1
    np.testing.assert_allclose(seen[0][0], [.0142, .002, -.019], atol=1e-16)
    np.testing.assert_allclose(seen[0][1], [.0142, .002, .021], atol=1e-16)
    np.testing.assert_array_equal(seen[0][2], [.01, .02, .03])
    assert seen[0][3] == .003
    assert before == a['current'] and raw_rows == a['rows']
    assert r['model_normal_points_per_pair'] == 1


def test_common_world_rigid_transform_preserves_gap_and_rotates_generated_points():
    a = fixture()
    # Use a unique cap/corner witness. Parallel side contact has an interval
    # of minimizers: floating rigid-transform roundoff may select an endpoint.
    for i, x in enumerate([-.0115, -.0215, -.0515]):
        a['current']['robot_body_frames_world'][0][i][3] = x
    expected = compile_contacts(**a)
    theta = .37; c, s = np.cos(theta), np.sin(theta)
    H = np.eye(4); H[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    H[:3, 3] = [.2, -.1, .3]
    for key in ('body_frames_world', 'robot_body_frames_world'):
        a['current'][key] = [(H@frame).tolist() for frame in np.array(a['current'][key])]
    actual = compile_contacts(**a)
    np.testing.assert_allclose(actual[1], expected[1], atol=1e-16)
    old, new = expected[3]['features'][0], actual[3]['features'][0]
    for field in ('point_world_m', 'box_witness_world_m', 'axis_witness_world_m'):
        np.testing.assert_allclose(new[field], H[:3, :3]@old[field]+H[:3, 3], atol=1e-15)
    np.testing.assert_allclose(new['normal_on_plant'], H[:3, :3]@old['normal_on_plant'], atol=1e-14)


def test_parallel_nonunique_minimum_reports_one_witness_not_a_covariant_force_distribution():
    a = fixture(); before = compile_contacts(**a)[3]['features'][0]
    assert before['parameter'] == .5 and before['minimizer_interval'] == [0., 1.]
    angle = .37; c, s = np.cos(angle), np.sin(angle)
    H = np.eye(4); H[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
    H[:3, 3] = [.2, -.1, .3]
    for key in ('body_frames_world', 'robot_body_frames_world'):
        a['current'][key] = [(H@frame).tolist() for frame in np.array(a['current'][key])]
    _, gap, _, report = compile_contacts(**a)
    after = report['features'][0]
    original_point = H[:3, :3].T@(np.array(after['point_world_m'])-H[:3, 3])
    np.testing.assert_allclose(original_point[:2], [-.003, 0.], atol=1e-15)
    assert -.02-1e-15 <= original_point[2] <= .02+1e-15
    assert gap[0] == pytest.approx(before['gap_m'], abs=1e-16)
    assert report['model_normal_points_per_pair'] == 1
    assert not report['native_contact_law_parity']
    # Do not require the same t/torque lever: no near-parallel snap, added
    # manifold point or fabricated native force-distribution evidence exists.


@pytest.mark.parametrize('what', ['knife', 'leaf', 'internal', 'unattributed',
    'self', 'bad_kind', 'overflow', 'stale', 'source', 'duplicate_inventory',
    'missing_inventory', 'bad_frame', 'nan_mu', 'negative_mu', 'bool_mu', 'vector_mu'])
def test_invalid_or_unknown_contacts_fail_closed(what):
    a = fixture()
    if what == 'knife': a['rows'][0]['collider1'] = '/Robot/Knife'
    elif what == 'leaf':
        a['rows'][0]['collider0'] = '/Plant/Stem/Leaf'
        a['all_plant_colliders'].append('/Plant/Stem/Leaf')
    elif what == 'internal': a['rows'][0]['collider1'] = '/Plant/Stem/Other'
    elif what == 'unattributed': a['rows'][0]['collider0'] = '/Elsewhere/Unknown'
    elif what == 'self': a['rows'][0]['collider1'] = a['rows'][0]['collider0']
    elif what == 'bad_kind': a['rows'][0]['kind'] = 'point'
    elif what == 'overflow': a['rows'] *= 65
    elif what == 'stale': a['current']['step_id'] = 2
    elif what == 'source': a['current']['source_target'] = 'wrong'
    elif what == 'duplicate_inventory': a['all_plant_colliders'] *= 2
    elif what == 'missing_inventory': a['all_plant_colliders'] = ['/Wrong/Collider']
    elif what == 'bad_frame': a['current']['body_frames_world'][0][0][0] = 1.1
    else: a['mu'] = {'nan_mu': np.nan, 'negative_mu': -.1, 'bool_mu': True, 'vector_mu': [.5]}[what]
    with pytest.raises(ValueError):
        compile_contacts(**a)


def test_support_knife_is_outside_only_but_dynamic_knife_is_never_filtered():
    a = fixture(); support = '/Plant/Support/StemCollider'
    a['all_plant_colliders'].append(support)
    a['rows'].append(dict(collider0=support, collider1='/Robot/Knife', kind='normal'))
    _, _, _, r = compile_contacts(**a)
    assert r['ignored_outside_dynamic_system'] == [
        dict(row=4, colliders=[support, '/Robot/Knife'], reason='outside_dynamic_plant_system')]
    a['rows'].append(dict(collider0=a['chain'][0].collider, collider1='/Robot/Knife', kind='normal'))
    with pytest.raises(ValueError, match='knife/scene'):
        compile_contacts(**a)


def test_empty_pair_stream_is_not_contact_completeness_or_a_feature_observation():
    a = fixture(); a['rows'] = []
    J, g, s, r = compile_contacts(**a)
    assert J.shape == (0, 7) and g.shape == s.shape == (0,)
    assert r['feature_covered'] == r['feature_observed'] == r['patches']['observed'] == []
    assert r['observed_pairs'] == [] and r['features_generated']
    assert not r['native_features_observed'] and not r['native_completeness_verified']
    assert not r['unobserved_pairs_enumerated']


@pytest.mark.parametrize('fault', ['distance', 'normal', 'point', 'parameter', 'gap', 'nonfinite'])
def test_inconsistent_helper_contract_cannot_emit_solver_geometry(monkeypatch, fault):
    real = fresh.closest_capsule_box
    def wrong(*args):
        w = real(*args)
        if fault == 'distance': w['distance_m'] = 0.
        elif fault == 'normal': w['normal'] = [-1., 0., 0.]
        elif fault == 'point': w['capsule_point'][0] += .001
        elif fault == 'parameter': w['parameter'] = 2.
        elif fault == 'gap': w['gap_m'] += .001
        else: w['axis_point'][0] = np.nan
        return w
    monkeypatch.setattr(fresh, 'closest_capsule_box', wrong)
    with pytest.raises(ValueError):
        compile_contacts(**fixture())


def test_model_coverage_must_be_explicitly_selected_by_solver_caller():
    a = fixture(); a['current']['robot_body_frames_world'][0][0][3] = -.0129
    J, g, s, r = compile_contacts(**a)
    args = dict(M=np.eye(7), q=np.zeros(7), v=np.zeros(7), K=np.zeros(7), C=np.zeros(7),
        J=J, g=g, s=s, kc=np.full(1, 1000.), dc=np.full(1, 1.0573968428135787),
        h=1/240, f=np.zeros(7), law=MaterialLaw('unilateral_kv_v1'),
        patches=r['patches'], root_dofs=6)
    modeled = solve_generalized(**args, feature_observed=r['feature_covered'])
    observed = solve_generalized(**args, feature_observed=r['feature_observed'])
    assert modeled['status'] == 'resolved' and not modeled['native_qualified']
    assert modeled['anchor_counts'] == [1] and len(modeled['normal_forces_n']) == 1
    assert observed['status'] == 'unresolved' and observed['tau_spring'] is None


def test_synthetic_45_coordinate_six_pairs_are_six_features_not_native_row_count():
    a = fixture(); original = a['chain'][0]
    a['chain'] = [replace(original, body='/Plant/Stem_'+str(i),
        collider='/Plant/Stem_'+str(i)+'/StemCollider') for i in range(3)]
    a['all_plant_colliders'] = [c.collider for c in a['chain']]
    a['rows'] = []
    for k, (cap, pad) in enumerate((cap, pad) for cap in a['chain'] for pad in a['pads']):
        for _ in range(1 if k == 5 else 2):
            for kind in ('normal', 'friction'):
                a['rows'].append(dict(collider0=cap.collider, collider1=pad.collider, kind=kind))
    for name in ('reference', 'current'):
        a[name]['body_paths'] = [c.body for c in a['chain']]
        a[name]['body_frames_world'] = np.tile(np.eye(4), (3, 1, 1)).tolist()
        a[name]['generalized_velocity'] = np.zeros(45).tolist()
        jac = np.zeros((3, 6, 45)); jac[:, :, :6] = np.eye(6)
        a[name]['body_world_com_jacobians'] = jac.tolist()
        a[name]['native_com_local_poses'] = [[0, 0, 0, 0, 0, 0, 1]]*3
    J, g, s, r = compile_contacts(**a)
    assert J.shape == (6, 45) and g.shape == s.shape == (6,)
    assert r['patches']['normal_indices'] == [[i] for i in range(6)]
    assert [len(x) for x in r['patches']['tangent_jacobians']] == [1]*6
    assert r['feature_covered'] == [True]*6 and r['feature_observed'] == [False]*6
    assert [p['native_normal_row_count'] for p in r['observed_pairs']] == [2, 2, 2, 2, 2, 1]
    assert r['root_columns_retained'] == 6
