"""Fake native views and pure predictor only; no SimulationApp or source writes."""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace as S

import numpy as np
import pytest

from . import contact_spring_actuation as module
from .contact_spring_actuation import FreshHoldSprings
from .implicit_springs import ImplicitJointSprings
from .plant_contact_binding import capture, deserialize, _pack, _hash, _native_binding
from .plant_contact_binding_test import fixture as binding_fixture, rehash
from .shaft_grasp import ShaftCapsule


def bound_geometry():
    p = capture(binding_fixture())
    chain = list(binding_fixture().grasp_observer.core.chain)
    for i in range(5, 15):
        body = '/T/Branch/S'+str(i)
        chain.append(ShaftCapsule(body, body+'/StemCollider', np.eye(4), .003, .01, .0005))
    pads = binding_fixture().grasp_observer.core.pads
    p['chain'] = [_pack(s) for s in chain]
    p['joints'] = [dict(path='/T/J'+str(i), body0=a.body, body1=b.body)
                   for i, (a, b) in enumerate(zip(chain[1:], chain[2:]))]
    p['plant_colliders'] = [dict(collider=s.collider, body_index=i, kind='capsule')
                            for i, s in enumerate(chain)]
    p['plant_colliders'].append(dict(collider=chain[4].body+'/Leaf/Mesh', body_index=4, kind='hull'))
    p['all_plant_collider_paths'] = sorted(r['collider'] for r in p['plant_colliders'])
    p['binding_sha256'] = _hash(_native_binding(p, chain, pads))
    rehash(p)
    return deserialize(p, source_target=p['source_target'], binding_sha256=p['binding_sha256'],
                       sha256=p['sha256'])


class View:
    def __init__(self, binding):
        self.count = 1
        self.shared_metatype = S(fixed_base=False, dof_names=[
            'Joint'+str(i)+':rot'+axis for i in range(13) for axis in 'XYZ'])
        self.link_paths = [[s.body for s in binding.chain[binding.cut_index:]]]
        self.k = np.linspace(.2, .3, 39, dtype=np.float32)[None, :]
        self.c = np.full((1, 39), .01, dtype=np.float32)
        self.caps = np.full((1, 39), .5, dtype=np.float32)
        self.q = np.linspace(-.005, .005, 39, dtype=np.float32)[None, :]
        self.v = np.zeros((1, 39), dtype=np.float32)
        self.root = np.zeros((1, 6), dtype=np.float32)
        self.targets = np.zeros((1, 39), dtype=np.float32)
        self.velocity_targets = self.targets.copy()
        self.types = np.ones((1, 39), dtype=np.uint32)
        self.effort = np.full((1, 39), .012, dtype=np.float32)
        self.writes = []
        self.coefficient_writes = []
        self.bad_readback = False
        self.readback_error = False
        self.setter_error = False

    def get_dof_stiffnesses(self): return self.k
    def get_dof_dampings(self): return self.c
    def get_dof_positions(self): return self.q
    def get_dof_velocities(self): return self.v
    def get_root_velocities(self): return self.root
    def get_dof_max_forces(self): return self.caps
    def get_dof_position_targets(self): return self.targets
    def get_dof_velocity_targets(self): return self.velocity_targets
    def get_drive_types(self): return self.types

    def set_dof_stiffnesses(self, value, indices):
        self.coefficient_writes.append('K'); self.k = value.copy()

    def set_dof_dampings(self, value, indices):
        self.coefficient_writes.append('C'); self.c = value.copy()

    def set_dof_actuation_forces(self, value, indices):
        assert value.shape == (1, 39) and value.dtype == np.float32
        assert indices.dtype == np.uint32 and indices.tolist() == [0]
        self.writes.append(value.copy())
        self.effort = value.copy()
        if self.setter_error: raise RuntimeError('setter failed after possible write')

    def get_dof_actuation_forces(self):
        if self.readback_error: raise RuntimeError('readback unavailable')
        result = self.effort.copy()
        if self.bad_readback:
            result[0, 0] = np.nextafter(result[0, 0], np.float32(np.inf))
        return result


def case():
    binding = bound_geometry(); a = View(binding)
    drives = dict(names=list(a.shared_metatype.dof_names), stiffness=a.k.tolist(),
                  damping=a.c.tolist(), max_forces=a.caps.tolist(), targets=a.targets.tolist())
    springs = ImplicitJointSprings(a)
    a.coefficient_writes.clear()
    return a, springs, binding, drives


def records(a, b, step=1, contact=True):
    jac = np.zeros((14, 6, 45))
    jac[:, :, :6] = np.eye(6)
    jac[1, :3, 6:9] = np.eye(3)
    robot = np.tile(np.eye(4), (2, 1, 1)); robot[:, 0, 3] = [-.0052, .0052]
    before = dict(schema='full_plant_native_prediction_snapshot_v1', step_id=step,
        source_target=b.source_target, body_paths=a.link_paths[0][:],
        joint_names=list(a.shared_metatype.dof_names), q_rad=a.q[0].astype(float).tolist(),
        generalized_velocity=np.r_[a.root[0], a.v[0]].astype(float).tolist(),
        mass_matrix=np.eye(45).tolist(), native_known_noncontact_force=[0.]*45,
        native_com_jacobian_velocity_check_passed=True, contact_force_included=False,
        constraint_forces_included=False, external_root_support_enabled_caller_asserted=True,
        body_frames_world=np.tile(np.eye(4), (14, 1, 1)).tolist(),
        body_world_com_jacobians=jac.tolist(), native_com_local_poses=[[0, 0, 0, 0, 0, 0, 1]]*14,
        robot_body_paths=[p.body for p in b.pads], robot_body_frames_world=robot.tolist(),
        robot_body_velocities_world=np.zeros((2, 6)).tolist(),
        robot_com_local_poses=[[0, 0, 0, 0, 0, 0, 1]]*2)
    ref = deepcopy(before); ref['step_id'] = step-1
    rows = [dict(collider0=b.chain[2].collider, collider1=b.pads[0].collider, kind='normal',
                 point_world_m=[99, 99, 99], normal_on_0=[0, 0, 0], separation_m=99,
                 impulse_on_0_ns=[999, 999, 999])] if contact else []
    previous = dict(before=ref, step_id=step, dt_s=1/240,
        contacts=dict(error=None, rows=rows, row_count=len(rows),
                      plant_collider_paths=list(b.plant_collider_paths)),
        after_generalized_velocity=before['generalized_velocity'][:])
    return previous, dict(before=before, step_id=step+1, dt_s=1/240)


def adapter(args):
    return FreshHoldSprings(*args[1:], experimental_hold_only=True)


@pytest.mark.parametrize('contact', [False, True])
def test_real_fresh_solver_only_writes_exact_intrinsic_packet_with_bounded_evidence(contact):
    args = case(); a, _, b, d = args; p, c = records(a, b, contact=contact)
    saved = deepcopy((p, c, d)); control = adapter(args)
    expected = module.grasp_contact_shadow.predict(p, c, b, d, geometry_model='fresh_box')
    effort, r = control.step(p, c)
    np.testing.assert_array_equal(effort, np.asarray(expected['result']['tau_joint'], dtype=np.float32))
    np.testing.assert_array_equal(a.effort[0], effort)
    assert len(a.writes) == 1 and a.coefficient_writes == []
    assert effort.shape == (39,) and effort.dtype == np.float32
    assert (p, c, d) == saved
    assert r['actual_effort_write_verified'] is True and r['write_attempted'] is True
    assert r['observed_pair_count'] == int(contact) and r['algebra_residuals']
    assert r['before_step'] == 1 and r['expected_after_step'] == 2
    assert not r['native_qualified'] and not r['physical_calibration'] and not r['training_eligible']
    assert not r['contact_force_submitted'] and not r['friction_force_submitted'] and not r['root_actuated']
    assert 'mass_matrix' not in json.dumps(r) and len(json.dumps(r)) < 5000
    effort[:] = 9  # Returned array has no alias into native command buffer.
    assert not np.any(a.effort == 9)


@pytest.mark.parametrize('key', ['stiffness', 'damping', 'max_forces', 'targets'])
@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1., True, '0.1'])
def test_bad_original_parameter_values_reject_at_construction(key, bad):
    a, s, b, d = case()
    d[key][0][0] = bad
    with pytest.raises(ValueError): FreshHoldSprings(s, b, d, experimental_hold_only=True)
    assert not a.writes and not a.coefficient_writes


def test_active_compressed_contact_retains_native_authority_and_original_spring_law():
    args = case(); a, _, b, d = args; p, c = records(a, b)
    for record in (p, c):
        record['before']['robot_body_frames_world'][0][0][3] = -.0048
    expected = module.grasp_contact_shadow.predict(p, c, b, d, geometry_model='fresh_box')
    assert max(expected['result']['normal_forces_n']) > .1
    effort, report = adapter(args).step(p, c)
    vp = np.asarray(expected['result']['v_pred'])
    expected_intrinsic = -np.asarray(d['stiffness'][0])*(a.q[0]+c['dt_s']*vp)-np.asarray(d['damping'][0])*vp
    np.testing.assert_array_equal(effort, expected_intrinsic.astype(np.float32))
    assert len(a.writes) == 1 and report['actual_effort_write_verified']
    assert not report['contact_force_submitted'] and not report['friction_force_submitted']


def test_float32_rounding_cannot_cross_nonrepresentable_original_bound(monkeypatch):
    a, s, b, d = case()
    # A fake float64 readback exercises the independent POST-quantization guard.
    # Real native float32 caps are representable already.
    cap = .1
    a.caps = np.full((1, 39), cap, dtype=float)
    d['max_forces'] = a.caps.tolist()
    control = FreshHoldSprings(s, b, d, experimental_hold_only=True)
    original = module.grasp_contact_shadow.predict
    def predict(*pos, **kw):
        result = original(*pos, **kw)
        result['result']['tau_joint'][0] = cap
        result['result']['tau_spring'][0] = cap
        return result
    monkeypatch.setattr(module.grasp_contact_shadow, 'predict', predict)
    with pytest.raises(ValueError, match='Float32 effort'): control.step(*records(a, b))
    assert not a.writes


@pytest.mark.parametrize('fault', ['signed_zero', 'dtype', 'shape', 'nan', 'interrupt'])
def test_exact_readback_is_bitwise_float32_and_failures_latch(monkeypatch, fault):
    args = case(); a, _, b, _ = args; a.q[:] = 0
    control = adapter(args)
    original = a.get_dof_actuation_forces
    def readback():
        r = original()
        if fault == 'signed_zero':
            r[0, 0] = np.copysign(r[0, 0], -np.copysign(1., r[0, 0]))
        elif fault == 'dtype': r = r.astype(float)
        elif fault == 'shape': r = r[0]
        elif fault == 'nan': r[0, 0] = np.nan
        elif fault == 'interrupt': raise KeyboardInterrupt('interrupted after write')
        return r
    monkeypatch.setattr(a, 'get_dof_actuation_forces', readback)
    with pytest.raises((ValueError, RuntimeError, KeyboardInterrupt)):
        control.step(*records(a, b, contact=False))
    assert control.error and len(a.writes) == 1
    with pytest.raises(RuntimeError, match='latched'): control.step(*records(a, b, 2))


def test_source_caps_checked_even_when_predictor_reports_resolved_and_cap_pass():
    a, s, b, d = case(); a.q[:] = 1000
    control = FreshHoldSprings(s, b, d, experimental_hold_only=True)
    with pytest.raises(ValueError): control.step(*records(a, b, contact=False))
    assert control.error and not a.writes


@pytest.mark.parametrize('opt_in', [False, None, 1, np.bool_(True), 'True'])
def test_explicit_python_true_required(opt_in):
    a, s, b, d = case()
    with pytest.raises(ValueError): FreshHoldSprings(s, b, d, experimental_hold_only=opt_in)
    assert not a.writes and not a.coefficient_writes


def test_no_default_enable_or_implicit_bootstrap():
    a, s, b, d = case()
    with pytest.raises(TypeError): FreshHoldSprings(s, b, d)
    control = FreshHoldSprings(s, b, d, experimental_hold_only=True)
    with pytest.raises(ValueError): control.step(None, records(a, b)[1])
    assert control.error and not a.writes


@pytest.mark.parametrize('fault', ['native_k', 'native_c', 'saved_k', 'saved_c', 'cap',
    'target', 'velocity_target', 'type', 'body_order', 'dof_order', 'count', 'fixed',
    'source', 'material', 'numeric_bool', 'numeric_nan', 'numeric_complex', 'names_duplicate'])
def test_native_or_source_contract_change_rejects_without_write(fault):
    args = case(); a, s, b, _ = args; p, c = records(a, b); control = adapter(args)
    if fault == 'native_k': a.k[0, 0] = .1
    elif fault == 'native_c': a.c[0, 0] = .1
    elif fault == 'saved_k': s.k[0] += .1
    elif fault == 'saved_c': s.c[0] += .1
    elif fault == 'cap': a.caps[0, 0] = .6
    elif fault == 'target': a.targets[0, 0] = .1
    elif fault == 'velocity_target': a.velocity_targets[0, 0] = .1
    elif fault == 'type': a.types[0, 0] = 0
    elif fault == 'body_order': a.link_paths[0].reverse()
    elif fault == 'dof_order': a.shared_metatype.dof_names.reverse()
    elif fault == 'count': a.count = 2
    elif fault == 'fixed': a.shared_metatype.fixed_base = True
    elif fault == 'source': control.binding = replace(b, source_target='other/target')
    elif fault == 'material': control.binding = replace(b, finger_friction=.7)
    elif fault == 'numeric_bool': s.k = [True]*39
    elif fault == 'numeric_nan': a.q[0, 0] = np.nan
    elif fault == 'numeric_complex': a.q = a.q.astype(complex)
    elif fault == 'names_duplicate': a.shared_metatype.dof_names[0] = a.shared_metatype.dof_names[1]
    original = a.effort.copy()
    with pytest.raises(ValueError): control.step(p, c)
    np.testing.assert_array_equal(a.effort, original)
    assert control.error and not a.writes and not a.coefficient_writes


@pytest.mark.parametrize('fault', ['root_released', 'root_integer', 'previous_released', 'step_gap',
    'step_bool', 'negative_step', 'dt_changed', 'dt_nan', 'dt_bool', 'inventory', 'leaf_omission',
    'source', 'order', 'body', 'error', 'row_count', 'overflow', 'fetch_v', 'q', 'v', 'root_v',
    'double_contact_force', 'double_constraint_force', 'jacobian', 'schema', 'knife', 'leaf'])
def test_stale_incomplete_unsupported_or_released_records_never_write(fault):
    args = case(); a, _, b, _ = args; p, c = records(a, b); n = c['before']
    if fault == 'root_released': n['external_root_support_enabled_caller_asserted'] = False
    elif fault == 'root_integer': n['external_root_support_enabled_caller_asserted'] = 1
    elif fault == 'previous_released': p['before']['external_root_support_enabled_caller_asserted'] = False
    elif fault == 'step_gap': c['step_id'] += 1
    elif fault == 'step_bool': n['step_id'] = True
    elif fault == 'negative_step': p['before']['step_id'] = -1
    elif fault == 'dt_changed': c['dt_s'] /= 2
    elif fault == 'dt_nan': c['dt_s'] = float('nan')
    elif fault == 'dt_bool': c['dt_s'] = True
    elif fault == 'inventory': p['contacts']['plant_collider_paths'].reverse()
    elif fault == 'leaf_omission': p['contacts']['plant_collider_paths'].remove(b.unsupported_plant_colliders[0])
    elif fault == 'source': n['source_target'] = 'wrong/source'
    elif fault == 'order': n['joint_names'].reverse()
    elif fault == 'body': n['body_paths'].reverse()
    elif fault == 'error': p['contacts']['error'] = 'native callback overflow'
    elif fault == 'row_count': p['contacts']['row_count'] = 2
    elif fault == 'overflow': p['contacts'].update(rows=p['contacts']['rows']*257, row_count=257)
    elif fault == 'fetch_v': p['after_generalized_velocity'][6] = .01
    elif fault == 'q': n['q_rad'][0] = np.nextafter(n['q_rad'][0], np.inf)
    elif fault == 'v': a.v[0, 0] = .1
    elif fault == 'root_v': a.root[0, 0] = .1
    elif fault == 'double_contact_force': n['contact_force_included'] = True
    elif fault == 'double_constraint_force': n['constraint_forces_included'] = True
    elif fault == 'jacobian': n['native_com_jacobian_velocity_check_passed'] = False
    elif fault == 'schema': n['schema'] = 'invented'
    elif fault == 'knife': p['contacts']['rows'][0]['collider1'] = '/Robot/Knife'
    elif fault == 'leaf': p['contacts']['rows'][0]['collider0'] = b.unsupported_plant_colliders[0]
    control = adapter(args)
    with pytest.raises((ValueError, KeyError)): control.step(p, c)
    assert control.error and not a.writes


def test_contiguous_steps_only_and_any_fault_latches():
    args = case(); a, _, b, _ = args; control = adapter(args)
    control.step(*records(a, b, 1)); control.step(*records(a, b, 2))
    with pytest.raises(ValueError): control.step(*records(a, b, 2))
    with pytest.raises(RuntimeError, match='latched'): control.step(*records(a, b, 3))
    assert len(a.writes) == 2


def test_skip_after_success_is_not_a_new_bootstrap():
    args = case(); a, _, b, _ = args; control = adapter(args)
    control.step(*records(a, b, 10))
    with pytest.raises(ValueError): control.step(*records(a, b, 12))
    assert len(a.writes) == 1


@pytest.mark.parametrize('fault', ['unresolved', 'cap_flag', 'too_large', 'tau_nan', 'root_extra',
    'mixed_spring', 'bad_residual', 'root_model', 'geometry_step', 'uncovered', 'material',
    'contact_submitted'])
def test_unresolved_or_malformed_proposal_fails_before_submission(monkeypatch, fault):
    args = case(); a, _, b, _ = args; original = module.grasp_contact_shadow.predict
    def predict(*pos, **kw):
        result = original(*pos, **kw); r = result['result']
        if fault == 'unresolved': r['status'] = 'unresolved'
        elif fault == 'cap_flag': result['configured_force_caps_passed'] = False
        elif fault == 'too_large': r['tau_joint'][0] = r['tau_spring'][0] = .50000001
        elif fault == 'tau_nan': r['tau_joint'][0] = float('nan')
        elif fault == 'root_extra': r['tau_joint'] = [0.]*6+r['tau_joint']
        elif fault == 'mixed_spring': r['tau_spring'][0] += .01
        elif fault == 'bad_residual': r['relative_equilibrium_residual'] = float('nan')
        elif fault == 'root_model': r['root_dofs'] = 6
        elif fault == 'geometry_step': result['geometry']['current_step'] -= 1
        elif fault == 'uncovered': result['geometry']['feature_covered'] = [False]
        elif fault == 'material': result['original_finger_compliance']['stiffness_n_m'] *= 2
        elif fault == 'contact_submitted': r['contact_force_applied'] = True
        return result
    monkeypatch.setattr(module.grasp_contact_shadow, 'predict', predict)
    with pytest.raises(ValueError): adapter(args).step(*records(a, b))
    assert not a.writes


@pytest.mark.parametrize('change', ['q', 'v', 'root', 'K', 'caps'])
def test_live_state_and_parameters_are_rechecked_after_prediction(monkeypatch, change):
    args = case(); a, s, b, _ = args; original = module.grasp_contact_shadow.predict
    def predict(*pos, **kw):
        result = original(*pos, **kw)
        if change == 'q': a.q[0, 0] += .001
        elif change == 'v': a.v[0, 0] += .001
        elif change == 'root': a.root[0, 0] += .001
        elif change == 'K': s.k[0] += .001
        elif change == 'caps': a.caps[0, 0] += .001
        return result
    monkeypatch.setattr(module.grasp_contact_shadow, 'predict', predict)
    with pytest.raises(ValueError): adapter(args).step(*records(a, b))
    assert not a.writes


@pytest.mark.parametrize('failure', ['bad_readback', 'readback_error', 'setter_error'])
def test_possible_write_followed_by_failure_is_latched_never_retried(failure):
    args = case(); a, _, b, _ = args; control = adapter(args); setattr(a, failure, True)
    with pytest.raises(RuntimeError): control.step(*records(a, b))
    assert len(a.writes) == 1 and control.error and control.last_step is None
    setattr(a, failure, False)
    with pytest.raises(RuntimeError, match='latched'): control.step(*records(a, b))
    assert len(a.writes) == 1 and a.coefficient_writes == []


def test_original_parameters_copied_not_aliased_and_binding_must_be_bound():
    args = case(); a, s, b, d = args; control = adapter(args)
    d['stiffness'][0][0] = 999
    control.step(*records(a, b))  # Original retained copy, not later caller edits.
    assert control._drive['stiffness'][0][0] != 999 and s.k[0] != 999
    for invalid in (S(), replace(b, schema='plant_contact_authored_v1', binding_sha256=None)):
        with pytest.raises(ValueError): FreshHoldSprings(s, invalid, control._drive, experimental_hold_only=True)


def test_reentrant_prediction_latches_even_if_callback_swallows_exception(monkeypatch):
    args = case(); a, _, b, _ = args; control = adapter(args); p, c = records(a, b)
    original = module.grasp_contact_shadow.predict
    def predict(*pos, **kw):
        with pytest.raises(RuntimeError, match='Reentrant'): control.step(p, c)
        return original(*pos, **kw)
    monkeypatch.setattr(module.grasp_contact_shadow, 'predict', predict)
    with pytest.raises(RuntimeError): control.step(p, c)
    assert control.error and not a.writes


def test_solver_cannot_mutate_caller_trace_or_original_drives(monkeypatch):
    args = case(); a, _, b, d = args; p, c = records(a, b); saved = deepcopy((p, c, d))
    original = module.grasp_contact_shadow.predict
    def predict(previous, current, binding, drives, **kw):
        result = original(previous, current, binding, drives, **kw)
        previous['contacts']['rows'][0]['point_world_m'][0] = -123
        drives['stiffness'][0][0] = -1
        return result
    monkeypatch.setattr(module.grasp_contact_shadow, 'predict', predict)
    adapter(args).step(p, c)
    assert (p, c, d) == saved


def predicted_forces(monkeypatch, binding, normals, frictions, pad_indices):
    original = module.grasp_contact_shadow.predict
    def predict(*pos, **kw):
        result = original(*pos, **kw); r = result['result']; g = result['geometry']
        pair = g['observed_pairs'][0]; feature = g['features'][0]
        g['observed_pairs'] = []; g['features'] = []
        for i, pad_index in enumerate(pad_indices):
            cap = binding.chain[2+i].collider; pad = binding.pads[pad_index].collider
            g['observed_pairs'].append(dict(pair, cap=cap, pad=pad))
            g['features'].append(dict(feature, cap=cap, pad=pad))
        g['feature_covered'] = [True]*len(normals)
        g['patches']['normal_indices'] = [[i] for i in range(len(normals))]
        r['normal_forces_n'] = normals; r['friction_forces_n'] = frictions
        r['anchor_counts'] = [1]*len(normals)
        return result
    monkeypatch.setattr(module.grasp_contact_shadow, 'predict', predict)


@pytest.mark.parametrize('normals,frictions,pads,expected', [
    ([.25, -.25], [[[0., 0.]], [[0., 0.]]], [0, 0], None),
    ([0., 0.], [[[.26, 0.]], [[-.26, 0.]]], [0, 0], None),
    ([0.], [[[.3, .4]]], [0], None),
    ([.2], [[[.3, 0.]]], [0], None),
    ([.49999], [[[0., 0.]]], [0], [.49999, 0.]),
    ([.3, .3], [[[.1, 0.]], [[-.1, 0.]]], [0, 1], [.4, .4]),
    ([.1, -.1], [[[.03, .04]], [[-.03, -.04]]], [0, 0], [.3, 0.]),
])
def test_predicted_target_guard_sums_norms_by_exact_pad_without_cancellation(
        monkeypatch, normals, frictions, pads, expected):
    args = case(); a, _, b, _ = args
    predicted_forces(monkeypatch, b, normals, frictions, pads)
    control = adapter(args)
    if expected is None:
        with pytest.raises(ValueError, match='upper bound >=0.5'): control.step(*records(a, b))
        assert control.error and not a.writes
    else:
        _, r = control.step(*records(a, b))
        np.testing.assert_allclose(list(r['predicted_target_per_finger_contact_upper_bound_n'].values()), expected)
        assert r['predicted_target_finger_limit_n'] == .5
        assert r['actual_all_contact_guard_still_required'] is True
        assert r['predicted_load_scope'] == 'discovered_target_shaft_pad_pairs_only_not_all_contacts'
        assert r['source_force_caps_are_mechanical_certification'] is False


@pytest.mark.parametrize('fault', ['missing_normal', 'extra_normal', 'nan_normal',
    'missing_friction', 'nan_friction', 'wrong_anchor_shape', 'wrong_anchor_count',
    'normal_order', 'pad_lookalike', 'unknown_cap', 'duplicate_pair', 'feature_identity'])
def test_malformed_predicted_load_mapping_rejects_before_any_write(monkeypatch, fault):
    args = case(); a, _, b, _ = args; original = module.grasp_contact_shadow.predict
    def predict(*pos, **kw):
        result = original(*pos, **kw); r = result['result']; g = result['geometry']
        if fault == 'missing_normal': del r['normal_forces_n']
        elif fault == 'extra_normal': r['normal_forces_n'].append(.1)
        elif fault == 'nan_normal': r['normal_forces_n'][0] = float('nan')
        elif fault == 'missing_friction': r['friction_forces_n'] = []
        elif fault == 'nan_friction': r['friction_forces_n'][0][0][0] = float('nan')
        elif fault == 'wrong_anchor_shape': r['friction_forces_n'][0].append([.1, 0.])
        elif fault == 'wrong_anchor_count': r['anchor_counts'][0] = 2
        elif fault == 'normal_order': g['patches']['normal_indices'] = [[1]]
        elif fault == 'pad_lookalike': g['observed_pairs'][0]['pad'] += 'Extra'
        elif fault == 'unknown_cap': g['observed_pairs'][0]['cap'] += 'Extra'
        elif fault == 'feature_identity': g['features'][0]['pad'] = b.pads[1].collider
        elif fault == 'duplicate_pair':
            g['observed_pairs'] *= 2; g['features'] *= 2; g['feature_covered'] *= 2
            g['patches']['normal_indices'] = [[0], [1]]
            r['anchor_counts'] = [1, 1]; r['normal_forces_n'] *= 2; r['friction_forces_n'] *= 2
        return result
    monkeypatch.setattr(module.grasp_contact_shadow, 'predict', predict)
    with pytest.raises((ValueError, KeyError)): adapter(args).step(*records(a, b))
    assert not a.writes


def test_empty_pair_no_normal_arrays_reports_exact_zero_and_never_invents_contacts():
    args = case(); a, _, b, _ = args
    _, r = adapter(args).step(*records(a, b, contact=False))
    assert r['observed_pair_count'] == 0
    assert list(r['predicted_target_per_finger_contact_upper_bound_n'].values()) == [0., 0.]


def test_empty_pairs_cannot_hide_nonempty_predicted_normal_array(monkeypatch):
    args = case(); a, _, b, _ = args; original = module.grasp_contact_shadow.predict
    def predict(*pos, **kw):
        result = original(*pos, **kw)
        result['result']['normal_forces_n'] = [.1]
        return result
    monkeypatch.setattr(module.grasp_contact_shadow, 'predict', predict)
    with pytest.raises(ValueError): adapter(args).step(*records(a, b, contact=False))
    assert not a.writes
