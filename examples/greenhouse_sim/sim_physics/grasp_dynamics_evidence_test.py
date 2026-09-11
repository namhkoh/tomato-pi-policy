"""Pure telemetry regression tests; no simulator, stage or native imports."""
import copy
import json
import numpy as np
import pytest

from sim_physics.grasp_dynamics_evidence import grasp_dynamics_evidence, _chain_lookup


def inputs():
    bodies = [f'/World/Target/Segment_{i:03d}' for i in range(15)]
    fingers = ['/World/Robot/finger1', '/World/Robot/finger2']
    bf = np.repeat(np.eye(4)[None], 15, axis=0)
    bf[:, 0, 3] = np.arange(15) / 10
    ff = np.repeat(np.eye(4)[None], 2, axis=0)
    ff[0, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    ff[0, :3, 3] = [1, 2, 3]
    return dict(body_frames=bf, body_paths=bodies, finger_frames=ff, finger_paths=fingers,
        rows=[(0, bodies[3]+'/StemCollider', [1.1, 2.2, 3.3], [1., 0., 0.], [-1e-20, 0., 0.], -.0002),
              (1, bodies[4]+'/StemCollider', [.5, .1, .2], [-1., 0., 0.], [.001, 0., 0.], .0001)],
        qf=[-.0025, .004], qdotf=[.001, -.002], targetf=[-.002, .002],
        gravityf=[-.31, -.32], drivecapsf=[.19, .18], dt=1/240, step_id=np.int64(719))


def test_full_same_step_copies_signed_contacts_and_reconstructed_pd():
    data = inputs(); r = grasp_dynamics_evidence(**data)
    assert r['body_count'] == 15 and r['finger_count'] == 2
    assert r['step_id'] == 719 and r['dt_s'] == 1/240
    np.testing.assert_allclose(r['reconstructed_unclipped_pd_force_n'], [.095, -.39])
    np.testing.assert_allclose(r['configured_gravity_plus_drive_upper_n'], [.5, .5])
    assert r['actual_drive_effort_n'] is None
    assert r['effort_basis'].endswith('not_actual_native_effort')
    assert not r['changes_grasp_or_release_decision'] and not r['force_closure_verified']
    assert not r['native_provenance_verified_by_helper']
    assert not r['original_raw_collider_order'] and not r['friction_included']
    a, b = r['contacts']
    assert a['chain_body_index'] == 3 and b['chain_body_index'] == 4
    np.testing.assert_allclose(a['point_in_finger_body_m'], [.2, -.1, .3])
    np.testing.assert_allclose(a['point_in_chain_body_m'], [.8, 2.2, 3.3])
    assert a['impulse_on_finger_world_ns'] == [-1e-20, 0., 0.]
    assert b['impulse_on_finger_world_ns'] == [.001, 0., 0.]
    json.dumps(r, allow_nan=False)
    # Output and every input are independent copies, not aliases.
    data['body_frames'][3, 0, 3] = 99
    data['rows'][0][2][0] = 99
    data['qf'][0] = 99
    assert r['body_frames_world_m'][3][0][3] == pytest.approx(.3)
    assert a['point_world_m'][0] == 1.1 and r['finger_positions_m'][0] == -.0025
    r['finger_frames_world_m'][0][0][0] = 99
    assert data['finger_frames'][0, 0, 0] == 0


@pytest.mark.parametrize('other', ['/World/Other/StemCollider',
    '/World/Target/Segment_003/LeafCollider', '/World/Target/Segment_003',
    '/World/Target/Segment_003/StemCollider/child'])
def test_unknown_colliders_never_receive_prefix_or_selected_body_attribution(other):
    data = inputs(); data['rows'] = [(1, other, [1, 2, 3], [0, 1, 0], [0, -.001, 0], -.0001)]
    c = grasp_dynamics_evidence(**data)['contacts'][0]
    assert c['other_collider'] == other and c['point_world_m'] == [1, 2, 3]
    assert c['point_in_finger_body_m'] == [1, 2, 3]
    assert c['chain_body_index'] is None and c['chain_body_path'] is None
    assert c['point_in_chain_body_m'] is None


def test_ordered_path_lookup_cache_does_not_reuse_wrong_body_index():
    data = inputs(); before = _chain_lookup.cache_info()
    grasp_dynamics_evidence(**data); grasp_dynamics_evidence(**data)
    assert _chain_lookup.cache_info().hits > before.hits
    data['body_paths'] = data['body_paths'][::-1]
    data['body_frames'] = data['body_frames'][::-1]
    c = grasp_dynamics_evidence(**data)['contacts'][0]
    assert c['chain_body_index'] == 11
    np.testing.assert_allclose(c['point_in_chain_body_m'], [.8, 2.2, 3.3])


def test_empty_contacts_and_new_sample_do_not_latch_old_rows():
    data = inputs(); grasp_dynamics_evidence(**data)
    data.update(rows=[], step_id=720)
    r = grasp_dynamics_evidence(**data)
    assert r['contacts'] == [] and r['contact_row_count'] == 0 and r['step_id'] == 720


@pytest.mark.parametrize('step', [-1, True, np.bool_(False), 1., '1', None])
def test_bad_step_ids_rejected(step):
    data = inputs(); data['step_id'] = step
    with pytest.raises(ValueError): grasp_dynamics_evidence(**data)


@pytest.mark.parametrize('dt', [0, -1, np.nan, np.inf, True, '0.01', None])
def test_bad_timestep_rejected(dt):
    data = inputs(); data['dt'] = dt
    with pytest.raises(ValueError): grasp_dynamics_evidence(**data)


@pytest.mark.parametrize('key', ['qf', 'qdotf', 'targetf', 'gravityf', 'drivecapsf'])
@pytest.mark.parametrize('bad', [[0], [0, 0, 0], [0, np.nan], [np.inf, 0]])
def test_finger_vector_count_and_finiteness(key, bad):
    data = inputs(); data[key] = bad
    with pytest.raises(ValueError): grasp_dynamics_evidence(**data)


def test_negative_drive_budget_and_overflow_fail_without_clipping():
    data = inputs(); data['drivecapsf'] = [-.1, .2]
    with pytest.raises(ValueError): grasp_dynamics_evidence(**data)
    data = inputs(); data['targetf'] = [1.7e308, 0]
    with pytest.raises(ValueError, match='Unrepresentable'): grasp_dynamics_evidence(**data)
    data = inputs(); data['gravityf'] = [1.7e308, 0]; data['drivecapsf'] = [1.7e308, .2]
    with pytest.raises(ValueError, match='Unrepresentable'): grasp_dynamics_evidence(**data)


@pytest.mark.parametrize('key', ['body_frames', 'finger_frames'])
@pytest.mark.parametrize('fault', ['missing', 'nan', 'reflection', 'scale', 'row'])
def test_frame_count_and_rigid_transform_validation(key, fault):
    data = inputs(); a = data[key]
    if fault == 'missing': data[key] = a[:-1]
    elif fault == 'nan': a[0, 0, 3] = np.nan
    elif fault == 'reflection': a[0, :3, 0] *= -1
    elif fault == 'scale': a[0, :3, 0] *= 1.01
    else: a[0, 3, 0] = .01
    with pytest.raises(ValueError): grasp_dynamics_evidence(**data)


@pytest.mark.parametrize('key', ['body_paths', 'finger_paths'])
@pytest.mark.parametrize('fault', ['duplicate', 'relative', 'string', 'empty'])
def test_path_inventory_validation(key, fault):
    data = inputs()
    if fault == 'duplicate': data[key][1] = data[key][0]
    elif fault == 'relative': data[key][0] = 'relative'
    elif fault == 'string': data[key] = '/World/NotASequence'
    else: data[key] = []
    with pytest.raises(ValueError): grasp_dynamics_evidence(**data)


@pytest.mark.parametrize('field,value', [(0, -1), (0, 2), (0, True), (0, .5),
    (1, None), (1, 'relative'), (2, [np.nan, 0, 0]), (3, [0, np.inf, 0]),
    (4, [0, 0]), (5, np.nan), (5, True)])
def test_malformed_contact_rows_fail(field, value):
    data = inputs(); row = list(data['rows'][0]); row[field] = value; data['rows'] = [row]
    with pytest.raises(ValueError): grasp_dynamics_evidence(**data)


def test_oriented_rows_are_not_renormalized_or_subject_to_new_force_gates():
    data = inputs()
    # Telemetry copies upstream normal semantics, even below squared-norm FTZ.
    tiny = -8.7742363e-38
    data['rows'] = [(0, '/World/Unknown', [0, 0, 0], [1, 1e-7, 0], [tiny, 6.73e-29, 0], 0.)]
    r = grasp_dynamics_evidence(**data)
    assert r['contacts'][0]['impulse_on_finger_world_ns'] == [tiny, 6.73e-29, 0]
    assert r['contacts'][0]['normal_on_finger_world'] == [1, 1e-7, 0]


def test_contact_count_is_bounded_without_silent_truncation():
    data = inputs(); data['rows'] = [data['rows'][0]] * 256
    assert grasp_dynamics_evidence(**data)['contact_row_count'] == 256
    data['rows'].append(data['rows'][0])
    with pytest.raises(ValueError, match='overflow'): grasp_dynamics_evidence(**data)
    data['rows'] = [(0,)]
    with pytest.raises(ValueError, match='Six-field'): grasp_dynamics_evidence(**data)


def test_rotation_translation_local_coordinates_invariant_and_no_input_mutation():
    data = inputs(); before = copy.deepcopy(data)
    baseline = grasp_dynamics_evidence(**data)
    world = np.eye(4); world[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    world[:3, 3] = [2, 3, 4]
    data['body_frames'] = world @ data['body_frames']; data['finger_frames'] = world @ data['finger_frames']
    data['rows'] = [(i, other, world[:3, :3]@p+world[:3, 3], world[:3, :3]@n,
                     world[:3, :3]@j, s) for i, other, p, n, j, s in data['rows']]
    moved = grasp_dynamics_evidence(**data)
    for a, b in zip(baseline['contacts'], moved['contacts']):
        np.testing.assert_allclose(a['point_in_finger_body_m'], b['point_in_finger_body_m'], atol=1e-14)
        np.testing.assert_allclose(a['point_in_chain_body_m'], b['point_in_chain_body_m'], atol=1e-14)
    original = inputs()
    np.testing.assert_array_equal(original['body_frames'], before['body_frames'])
    json.dumps(moved, allow_nan=False)
