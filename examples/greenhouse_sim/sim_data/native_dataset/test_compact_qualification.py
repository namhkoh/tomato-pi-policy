"""Synthetic CPU receipts only: these fixtures never qualify real native data."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from ..depth_preview import sha256
from . import compact_qualification as q
from .dual_storage import write_pair
from .test_capture_storage import synthetic_callback


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')
    return sha256(path)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.fixture
def proof(tmp_path, monkeypatch):
    """Exercise real storage/readers with explicitly substituted fixture producer pins."""
    root = tmp_path/'qualification'
    root.mkdir()
    worker = tmp_path/'native_same_callback_worker_20260915_v1.py'
    queue = tmp_path/'queue_same_callback_native_20260916_v1.py'
    source = tmp_path/'source.json'
    for p in (worker, queue, source):
        p.write_text('synthetic fixture only: '+p.name, encoding='utf-8')
    monkeypatch.setattr(q, 'WORKER_SHA256', sha256(worker))
    monkeypatch.setattr(q, 'QUEUE_SHA256', sha256(queue))
    plan_path = tmp_path/'plan.json'
    target_cases = [dict(target_id='synthetic/target'+str(i),
        views=[dict(candidate_id=f't{i}_v{j}') for j in range(9)]) for i in range(4)]
    binding = {str(source): sha256(source)}
    plan = dict(split='train', resolution=[1696, 816], maximum_native_frames=36,
        target_cases=target_cases, training_approved=False, source_cap_reset=False,
        physical_motion_commanded=False, hidden_cut_coordinates_executable=False,
        source_bindings=binding, implementation_bindings=binding, prerequisite_bindings=binding)
    monkeypatch.setattr(q, 'DIAGNOSTIC_PLAN_SHA256', put(plan_path, plan))
    storage = {str(p): q._LOADED[str(p)] for p in q._STORAGE_PATHS}
    queue_bindings = {str(p): sha256(p) for p in
        (worker, queue, plan_path, q._HERE/'dual_storage.py', q._HERE/'capture_storage.py', *q._REVIEW_PATHS)}
    put(root/'queue_000002.json', dict(state='native_diagnostic_complete_no_default_change',
        bindings=queue_bindings, exit_code=0, training_approved=False, training_diversity_increment=0))
    callback = synthetic_callback()
    pair = write_pair(root/'raw_same_callback/t0_v0', root/'capture/t0_v0', callback)
    rows = [dict(candidate_id=s['candidate_id'], target_id=case['target_id'], requested_spec=s,
        state='rejected_pose', training_approved=False) for case in target_cases for s in case['views']]
    rows[0].update(state='native_captured_pending_review', **pair['compact_result'],
        eligible_annotation=True, automatic_annotation_eligible=False, label_reason=callback['label']['reason'])
    result = dict(state='native_generated_multiview_pilot_complete_pending_review',
        source_assets_unchanged=True, training_approved=False, captured_frames=1,
        compact_native_storage=True, experiment_script_sha256=sha256(worker),
        storage_implementation_bindings=storage, records=rows, same_callback_storage=[pair])
    request = dict(plan_path=str(plan_path), plan_sha256=sha256(plan_path),
        compact_native_storage=True, same_callback_raw_compact_diagnostic=True,
        training_diversity_increment=0, training_started=False, render_profile_experiment=False,
        experiment_script_sha256=sha256(worker), storage_implementation_bindings=storage)
    raw_result = deepcopy(result)
    raw_result['compact_native_storage'] = False
    raw_result['records'][0].update(pair['raw_result'])
    raw_request = dict(request, compact_native_storage=False, same_callback_compact_capture=str(root/'capture'))
    result_hash = put(root/'capture/result.json', result)
    raw_result_hash = put(root/'raw_same_callback/result.json', raw_result)
    request_hash = put(root/'capture/request.json', request)
    raw_request_hash = put(root/'raw_same_callback/request.json', raw_request)
    proof = dict(state='native_same_callback_storage_and_annotation_replay_passed', captured_frames=1,
        exact_native_npy_files=3, full_prim_identity_tables_equal=True, caller_buffers_unchanged=True,
        worker_sha256=sha256(worker), implementation_bindings=storage, training_approved=False,
        training_diversity_increment=0, source_removed=False, result_sha256=result_hash,
        raw_result_sha256=raw_result_hash)
    for folder, res, rh, qh in (('capture', result, result_hash, request_hash),
                                ('raw_same_callback', raw_result, raw_result_hash, raw_request_hash)):
        record = dict(sample=str(root/folder/'t0_v0'), target_id='synthetic/target0',
            decision='hold_visual_clarity', reason=callback['label']['reason'],
            label_replayed_exact=True, native_callback_hashes_verified=True, source_and_file_hashes_verified=True,
            trace_replayed_exact=True, training_approved=False, source_cap_reset=False,
            physical_execution_approved=False, visual_review_performed=False,
            sample_sha256=res['records'][0]['sample_sha256'], label_sha256=res['records'][0]['label_sha256'],
            rgb_sha256=pair['exact_observations']['inputs/rgb.png']['sha256'])
        audit = dict(state='completed_automatic_annotation_replay', capture=str(root/folder), records=[record],
            counts={'hold_visual_clarity': 1}, training_approved=False, source_assets_unchanged=True,
            original_reviews_modified=False, result_sha256=rh, request_sha256=qh, plan_sha256=sha256(plan_path),
            review_code_bindings={str(p): q._LOADED[str(p)] for p in q._REVIEW_PATHS})
        prefix = 'compact' if folder == 'capture' else 'raw'
        proof[prefix+'_audit_sha256'] = put(root/(prefix+'_audit.json'), audit)
        put(root/(folder+'_postexit_audit.json'), audit)
    path = root/'same_callback_qualification.json'
    put(path, proof)
    return path


def check(path):
    return q.check_qualification(path, expected_sha256=sha256(path))


@pytest.fixture
def proof_v2(proof, monkeypatch):
    # The v1 fixture/QUEUE_SHA256 substitution stays untouched. V2 has a separate
    # explicitly substituted synthetic producer and an exact loaded guard path.
    root = proof.parent.parent
    producer = root/'queue_same_callback_native_20260916_v2.py'
    guard = root/'native_process_guard.py'
    for path in (producer, guard):
        path.write_text('synthetic fixture only: '+path.name, encoding='utf-8')
    monkeypatch.setattr(q, 'QUEUE_V2_SHA256', sha256(producer))
    monkeypatch.setattr(q, '_QUEUE_V2_CODE_PATHS', (guard,))
    monkeypatch.setitem(q._LOADED, str(guard), sha256(guard))
    receipt = proof.parent/'queue_000002.json'
    document = read(receipt)
    document['bindings'].pop(str(root/'queue_same_callback_native_20260916_v1.py'))
    document['bindings'].update({str(p): sha256(p) for p in (producer, guard)})
    put(receipt, document)
    return proof


def test_reviewed_queue_producer_pins_are_explicit_constants():
    assert q.QUEUE_SHA256 == '50b458e946c7f71c356f4dd9378ed6330845892dbc4fa656da90a11ca341fe91'
    assert q.QUEUE_V2_SHA256 == 'bd4c42d2cc9ebebaa03ea00d264a3493985aef07b1e7aa8779c27f23c62a7115'
    assert q._QUEUE_V2_CODE_PATHS == (q._HERE/'native_process_guard.py',)
    assert str(q._QUEUE_V2_CODE_PATHS[0]) in q._LOADED


def test_legacy_queue_sha_fixture_and_v1_without_guard_binding(proof):
    bindings = read(proof.parent/'queue_000002.json')['bindings']
    assert bindings[str(proof.parent.parent/'queue_same_callback_native_20260916_v1.py')] == q.QUEUE_SHA256
    assert not any(Path(p).name == 'native_process_guard.py' for p in bindings)
    assert q.verify_checked_qualification(check(proof))


def test_v2_complete_receipt_includes_pinned_guard(proof_v2):
    receipt = check(proof_v2)
    for dependency in q._QUEUE_V2_CODE_PATHS:
        assert receipt['bindings'][str(dependency)] == q._LOADED[str(dependency)]
    assert q.verify_checked_qualification(receipt)
    assert q.qualify_storage(proof_v2) == receipt['bindings']


@pytest.mark.parametrize('change', ['missing', 'wrong_hash', 'wrong_path', 'duplicate_path'])
def test_v2_exact_loaded_guard_binding_required(proof_v2, change):
    path = proof_v2.parent/'queue_000002.json'
    document = read(path)
    bindings = document['bindings']
    guard = q._QUEUE_V2_CODE_PATHS[0]
    if change == 'missing':
        bindings.pop(str(guard))
    elif change == 'wrong_hash':
        bindings[str(guard)] = 'a'*64
    else:
        copy = guard.parent/'other'/guard.name
        copy.parent.mkdir()
        copy.write_bytes(guard.read_bytes())
        bindings[str(copy)] = sha256(copy)
        if change == 'wrong_path':
            bindings.pop(str(guard))
    put(path, document)
    with pytest.raises(ValueError, match='v2 native process guard binding'):
        check(proof_v2)


@pytest.mark.parametrize('rebind', [False, True])
def test_v2_guard_changed_after_load_cannot_self_pin(proof_v2, rebind):
    guard = q._QUEUE_V2_CODE_PATHS[0]
    guard.write_bytes(b'changed guard after qualifier import')
    if rebind:
        path = proof_v2.parent/'queue_000002.json'
        document = read(path)
        document['bindings'][str(guard)] = sha256(guard)
        put(path, document)
    with pytest.raises(ValueError, match='Compact implementation changed'):
        check(proof_v2)


def test_v2_guard_rehashed_after_qualification(proof_v2):
    receipt = check(proof_v2)
    q._QUEUE_V2_CODE_PATHS[0].write_bytes(b'changed guard after qualification')
    with pytest.raises(ValueError, match='Changed qualification binding'):
        q.verify_checked_qualification(receipt)


@pytest.mark.parametrize('version', ['v1', 'v2'])
@pytest.mark.parametrize('change', ['unknown', 'renamed', 'wrong_hash', 'rebound_bytes', 'duplicate', 'both_versions'])
def test_known_unambiguous_queue_producer_required(request, version, change):
    proof = request.getfixturevalue('proof' if version == 'v1' else 'proof_v2')
    path = proof.parent/'queue_000002.json'
    document = read(path)
    bindings = document['bindings']
    producer = proof.parent.parent/('queue_same_callback_native_20260916_'+version+'.py')
    if change == 'wrong_hash':
        bindings[str(producer)] = 'a'*64
    elif change == 'rebound_bytes':
        producer.write_bytes(b'different producer even with matching receipt hash')
        bindings[str(producer)] = sha256(producer)
    else:
        if change == 'unknown':
            other = producer.with_name('queue_same_callback_native_20260916_v99.py')
        elif change == 'renamed':
            other = producer.with_name('unreviewed_queue.py')
        elif change == 'duplicate':
            other = producer.parent/'other'/producer.name
        else:
            other = producer.with_name('queue_same_callback_native_20260916_'+('v2' if version == 'v1' else 'v1')+'.py')
        other.parent.mkdir(exist_ok=True)
        other.write_bytes(producer.read_bytes())
        bindings[str(other)] = sha256(other)
        if change in ('unknown', 'renamed'):
            bindings.pop(str(producer))
    put(path, document)
    with pytest.raises(ValueError, match='queue producer|queue source binding'):
        check(proof)


def test_complete_full_sha_receipt_and_controller_interface(proof):
    before = {p: sha256(p) for p in proof.parent.rglob('*') if p.is_file()}
    receipt = check(proof)
    assert receipt['captured_frames'] == 1 and receipt['exact_native_npy_files'] == 3
    assert receipt['storage_backend'] == q.STORAGE_BACKEND
    assert receipt['bindings'][str(proof)] == sha256(proof)
    assert q.verify_checked_qualification(receipt)
    assert q.qualify_storage(proof) == receipt['bindings']
    assert q.qualify_storage(proof, expected_sha256=sha256(proof)) == receipt['bindings']
    for path, expected in before.items():
        assert receipt['bindings'][str(path)] == expected
        assert sha256(path) == expected
    assert set(q.implementation_bindings()) <= receipt['bindings'].keys()


@pytest.mark.parametrize('pin', [None, '', 'x'*64, 'a'*64, 1])
def test_caller_pin_required(proof, pin):
    with pytest.raises(ValueError):
        q.check_qualification(proof, expected_sha256=pin)


@pytest.mark.parametrize('field,value', [
    ('state', 'failed'), ('captured_frames', 0), ('captured_frames', True), ('captured_frames', 37),
    ('captured_frames', 2), ('exact_native_npy_files', 0), ('exact_native_npy_files', 2),
    ('exact_native_npy_files', 6), ('exact_native_npy_files', 3.0),
    ('full_prim_identity_tables_equal', False), ('caller_buffers_unchanged', False),
    ('training_approved', True), ('training_diversity_increment', 1), ('source_removed', True),
    ('worker_sha256', 'a'*64), ('implementation_bindings', {})])
def test_invalid_rebound_proof_fails(proof, field, value):
    doc = read(proof)
    doc[field] = value
    put(proof, doc)
    with pytest.raises(ValueError):
        check(proof)


@pytest.mark.parametrize('name', ['raw_audit.json', 'compact_audit.json', 'capture_postexit_audit.json',
    'raw_same_callback_postexit_audit.json', 'capture/result.json', 'raw_same_callback/result.json',
    'capture/request.json', 'raw_same_callback/request.json', 'queue_000002.json',
    'raw_same_callback/t0_v0/same_callback_storage.json'])
def test_missing_required_artifact(proof, name):
    (proof.parent/name).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        check(proof)


@pytest.mark.parametrize('name', ['raw_audit.json', 'compact_audit.json',
    'capture/result.json', 'raw_same_callback/result.json', 'capture/request.json'])
def test_referenced_hash_is_not_just_presence(proof, name):
    with (proof.parent/name).open('ab') as stream:
        stream.write(b' ')
    with pytest.raises(ValueError):
        check(proof)


@pytest.mark.parametrize('field,value', [('state', 'native_diagnostic_failed_preserved'),
    ('exit_code', 1), ('exit_code', -1), ('exit_code', False), ('exit_code', '0'), ('bindings', {})])
def test_queue_zero_exit_and_bound_producer_required(proof, field, value):
    path = proof.parent/'queue_000002.json'
    doc = read(path)
    doc[field] = value
    put(path, doc)
    with pytest.raises(ValueError):
        check(proof)


@pytest.mark.parametrize('folder', ['', 'capture', 'raw_same_callback'])
def test_failure_marker_always_rejects(proof, folder):
    put(proof.parent/folder/'failure.json', {})
    with pytest.raises(ValueError, match='Failed storage diagnostic'):
        check(proof)


@pytest.mark.parametrize('folder', ['capture', 'raw_same_callback'])
def test_postexit_must_match_original_not_counts_only(proof, folder):
    path = proof.parent/(folder+'_postexit_audit.json')
    doc = read(path)
    doc['records'][0]['sample_sha256'] = 'a'*64
    put(path, doc)
    with pytest.raises(ValueError, match='Post-exit audit differs'):
        check(proof)


@pytest.mark.parametrize('field,value', [('review_code_bindings', {}), ('records', []),
    ('state', 'incomplete'), ('training_approved', True), ('counts', {'invented': 1})])
def test_rebound_both_audits_still_fail(proof, field, value):
    doc = read(proof.parent/'compact_audit.json')
    doc[field] = value
    marker = read(proof)
    marker['compact_audit_sha256'] = put(proof.parent/'compact_audit.json', doc)
    put(proof.parent/'capture_postexit_audit.json', doc)
    put(proof, marker)
    with pytest.raises(ValueError):
        check(proof)


@pytest.mark.parametrize('field,value', [('same_callback_storage', []), ('records', []),
    ('source_assets_unchanged', False), ('storage_implementation_bindings', {})])
def test_rebound_result_cannot_omit_native_evidence(proof, field, value):
    doc = read(proof.parent/'capture/result.json')
    doc[field] = value
    marker = read(proof)
    marker['result_sha256'] = put(proof.parent/'capture/result.json', doc)
    put(proof, marker)
    with pytest.raises(ValueError):
        check(proof)


@pytest.mark.parametrize('name', sorted(q.ARRAYS))
@pytest.mark.parametrize('folder', ['capture', 'raw_same_callback'])
def test_every_array_checked_not_count_only(proof, name, folder):
    directory = proof.parent/folder/'t0_v0'
    if folder == 'capture':
        entry = read(directory/'bundle.json')['files'][name]
        path = directory/entry['stored_path']
    else:
        path = directory/name
    raw = path.read_bytes()
    path.write_bytes(raw[:-1]+bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError):
        check(proof)


def test_changed_source_and_final_rehash(proof):
    receipt = check(proof)
    source = proof.parent.parent/'source.json'
    source.write_bytes(b'changed source')
    with pytest.raises(ValueError, match='Changed qualification binding'):
        q.verify_checked_qualification(receipt)
    with pytest.raises(ValueError, match='hash mismatch'):
        check(proof)


def test_new_failure_or_queue_after_check(proof):
    receipt = check(proof)
    put(proof.parent/'queue_000003.json', dict(state='failed'))
    with pytest.raises(ValueError, match='Queue changed'):
        q.verify_checked_qualification(receipt)


def test_strict_json_and_original_name(tmp_path):
    path = tmp_path/'same_callback_qualification.json'
    path.write_text('{"state":1,"state":2}', encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate'):
        check(path)
    path.write_text('{"value":NaN}', encoding='utf-8')
    with pytest.raises(ValueError, match='Nonfinite'):
        check(path)
    with pytest.raises(ValueError, match='Original same-callback'):
        q.check_qualification(tmp_path/'other.json', expected_sha256='a'*64)
