'''Fixed two-plan query-V2/reference56 registration for serial_phases only.

Job adds annotation_epoch, annotation_policy_sha256, storage_qualification,
storage_qualification_sha256 and matched_short. The latter pins capture_path,
request_sha256, result_sha256, audit_path/audit_sha256 and
completion_path/completion_sha256. No backend/budget/command overrides.

Old header receipts are authenticated, NOT replayed or relabeled here. Native
V2 replay is separate from the pending matched RGB/Z/ID/camera comparison and
visual review. Neither audit completion nor these rerenders grants diversity,
admission, photometric equivalence, or short-budget qualification.
'''
from collections import Counter
from copy import deepcopy
from pathlib import Path

from ..native_original_capture import contracts as oc, serial_queue as legacy
from .. import native_multitarget_plan as batch
from ..native_budget import evidence as budget_evidence
from . import compact_qualification as storage, query_audit_v2 as audit
from .bundle import SampleReader

KIND = 'query_v2_qualification.v1'
JOB_FIELDS = ('annotation_epoch', 'annotation_policy_sha256', 'storage_qualification',
              'storage_qualification_sha256', 'matched_short')
ANNOTATION = dict(annotation_epoch='greenhouse.native_query_selection.v2',
    annotation_policy_sha256='079ef8cc6eeed9d645b3e11e3302f78b0d59b84868a6c4f4763dd50469df52a0')
FROZEN_CODE = {
    'compact_query_v2.py': 'd530e001b5020ba44022bf2a5a8452c5c967e83e5797425b0d5d92faa33d5b31',
    'query_audit_v2.py': 'd02fbe409b7b6311ed9ee6b0e869a40604bf456aedd2f59060512cdb02940886'}
_ROOT = Path(__file__).resolve().parents[4]
_PRIOR = _ROOT/'data/sim_data/collection_batches/native_reference_scale_20260916_v4'
# folder, proposals, targets, captured, automatic, plan/request/result/audit/completion SHA.
CASES = (
    ('job_0001_seed101_full_round1', 12, 2, 11, 4,
     'b1f448a6afd5d4af5ed6f1e203f7842c29a0ab962587fa4da2503a5e480ee7dd',
     '4c9b5a4de9974a12a845806d8975daf3f024b6f63739b6a2ce9c07474cee52b0',
     '4bfd0dc54442187faea5db1bd7de3d6fc4e74236f55bd6f6b7a6946426556a34',
     'cb06109fcdf1e29c4a668cc83fcc5c02862cd0336aff54350ba6c3f32854794e',
     '1e34829e4f7ae1414928df91c5f230db6e5891de3cc6a0f3b28d0b21cc43d324'),
    ('job_0008_seed43_full_round1', 24, 4, 14, 11,
     'f735ed23054e80cee472a8588dd969daf74f854e419f2a4d683e5256bc6b682b',
     '07eb38d4f8c07106e5ff6be32a5af70d1edfc938bf61777ee271808368fcbeab',
     '51e1099c792c978a9c203b219df0ac8ccaf9ee353ac02d91c0b3d04c834fd0a2',
     '7bed24d8b913f020d8cb6dd67725338bb87800917d0fb89f8e660837acbfc9ef',
     'b77a6af108869d554af8ebac6cec3be0762e7256cc16901487ff279391cb665b'))
_LOADED = oc.merge_bindings(audit.implementation_bindings(),
    {str(Path(__file__).resolve()): oc.sha256(__file__)})


def implementation_bindings():
    oc.bind_all(_LOADED)
    oc.require(audit.annotation_fields() == ANNOTATION, 'Unregistered annotation policy')
    for name, pin in FROZEN_CODE.items():
        oc.require(_LOADED[str(Path(__file__).resolve().parent/name)] == pin,
                   'Unregistered query worker/auditor')
    return dict(_LOADED)


def _no_failure(root):
    oc.require(not (Path(root)/'failure.json').exists(), 'Failed matched capture/completion')


def _matched(path, pin, supplied, plan):
    case = next((c for c in CASES if path == (_PRIOR/c[0]/'plan.json').resolve() and pin == c[5]), None)
    oc.require(case is not None, 'Only the two registered prior plans are allowed')
    folder = path.parent
    expected = dict(capture_path=str(folder/'capture'), request_sha256=case[6], result_sha256=case[7],
        audit_path=str(folder/'automatic_audit.json'), audit_sha256=case[8],
        completion_path=str(folder/'campaign_result.json'), completion_sha256=case[9])
    oc.require(supplied == expected, 'Exact registered old capture bindings required')
    pins = {str(path): pin, str(folder/'capture/request.json'): case[6],
        str(folder/'capture/result.json'): case[7], expected['audit_path']: case[8],
        expected['completion_path']: case[9]}
    oc.bind_all(pins)
    _no_failure(folder)
    _no_failure(folder/'capture')
    request, result, review, done = (oc.read_json(p) for p in (
        folder/'capture/request.json', folder/'capture/result.json',
        folder/'automatic_audit.json', folder/'campaign_result.json'))
    oc.require(request['plan_path'] == str(path) and request['plan_sha256'] == pin
        and request['training_started'] is False, 'Old request/plan mismatch')
    oc.require(result['state'] == 'native_generated_multiview_pilot_complete_pending_review'
        and result['source_assets_unchanged'] is True and result['training_approved'] is False
        and result['source_cap_reset'] is False, 'Incomplete old result')
    oc.require(all(d['native_instance_backend'] == 'fast' and d['render_budget_profile'] == 'warm56_then8_trial'
        and d['render_profile_experiment'] is False for d in (request, result)), 'Old backend/budget differs')
    oc.require(review['state'] == 'completed_automatic_annotation_replay'
        and review['capture'] == str(folder/'capture') and review['plan_sha256'] == pin
        and review['request_sha256'] == case[6] and review['result_sha256'] == case[7]
        and review['training_approved'] is False and review['original_reviews_modified'] is False,
        'Old audit receipt does not bind this capture')
    oc.require(done['state'] == 'native_complete_pending_review' and type(done['native_exit_code']) is int
        and done['native_exit_code'] == 0 and done['job'] == str(folder)
        and done['source_family'] == plan['source_family'] and done['training_approved'] is False,
        'Missing exact saved old completion declaration')
    oc.require(plan['maximum_native_frames'] == len(result['records']) == case[1]
        and len(plan['target_cases']) == result['target_count'] == case[2]
        and result['captured_frames'] == done['captured'] == case[3]
        and result['automatically_clear_annotation_candidates'] == done['automatic'] == case[4]
        and review['counts'] == done['audit_counts'] and sum(review['counts'].values()) == case[3],
        'Old planned/captured/audited membership differs')
    oc.bind_all(pins)
    return dict(case_id=case[0], plan_path=str(path), plan_sha256=pin, **expected,
        bindings=pins, old_buffer_replay_performed=False,
        completion_evidence='pinned_saved_exit0_declaration_not_independent_OS_attestation')


def validate_plan(path, pin, query):
    legacy._keys(query, JOB_FIELDS, 'query qualification fields')
    oc.require(all(query[k] == v for k, v in ANNOTATION.items()), 'Explicit fixed annotation epoch/policy required')
    code = implementation_bindings()
    path = oc.pin(legacy._path(str(path)), pin)
    plan = oc.read_json(path)
    count = plan['maximum_native_frames']
    oc.require(type(count) is int and 1 <= count <= 24 and 1 <= len(plan['target_cases']) <= 4,
               'Qualification is bounded to four targets / 24 proposals')
    oc.require(plan['split'] == 'train' and oc.FROZEN_SPLITS.get(plan['source_family']) == 'train',
               'Fixed TRAIN donor required')
    matched = _matched(path, pin, query['matched_short'], plan)
    anchor = batch.check(plan, replay_geometry=True)
    oc.require(len(batch.capture_jobs(plan, anchor)) == count, 'Plan proposal count mismatch')
    proof_path = legacy._path(query['storage_qualification'])
    proof = storage.check_qualification(proof_path, expected_sha256=query['storage_qualification_sha256'])
    pins = oc.merge_bindings(code, matched['bindings'], proof['bindings'],
        plan['source_bindings'], plan['prerequisite_bindings'], plan['implementation_bindings'])
    roots = [_PRIOR, path.parent, proof_path.parent]
    # Preserve every base package/capture/plan, not just the anchor's files.
    for base in [anchor, *(oc.read_json(c['base_pair_plan']) for c in plan['target_cases'])]:
        roots.extend(base[k] for k in ('source_capture', 'variant_directory', 'prerequisite_directory'))
        source_plan = legacy._path(base['source_collection_plan'])
        roots.extend((source_plan.parent, oc.read_json(source_plan)['package']))
    oc.bind_all(pins)
    storage.verify_checked_qualification(proof)
    return dict(plan=plan, count=count, bindings=pins,
        roots=[*roots, *(Path(p).parent for p in pins)], query=deepcopy(query),
        storage=deepcopy(proof), matched_short=matched)


def command(isaac, submitted, pin, capture, checked):
    oc.require(checked is not None and checked['matched_short']['plan_sha256'] == pin,
               'Checked fixed query job required')
    query = checked['query']
    return [str(legacy._path(isaac)), '-m', audit.WORKER_MODULE,
        '--batch-plan', str(submitted), '--plan-sha256', pin,
        '--annotation-policy-sha256', query['annotation_policy_sha256'],
        '--storage-qualification', query['storage_qualification'],
        '--storage-qualification-sha256', query['storage_qualification_sha256'],
        '--instance-backend', 'fast', '--render-budget', 'reference56', '--output', str(capture)]


def postexit_audit(submitted, pin, capture, folder, checked):
    oc.require(checked is not None, 'Checked query job required')
    implementation_bindings()
    oc.bind_all(checked['bindings'])
    oc.pin(submitted, pin)
    matched = checked['matched_short']
    _matched(Path(matched['plan_path']), pin, checked['query']['matched_short'], checked['plan'])
    storage.verify_checked_qualification(checked['storage'])
    _no_failure(capture)
    request, result = (oc.read_json(capture/name) for name in ('request.json', 'result.json'))
    request_pin, result_pin = (oc.sha256(capture/name) for name in ('request.json', 'result.json'))
    audit.verify_epoch_documents(request, result)
    oc.require(request['plan_path'] == str(submitted) and request['plan_sha256'] == pin
        and result['plan_sha256'] == pin and result['request_sha256'] == request_pin,
        'Query completion differs from exact submitted plan/request')
    for doc in (request, result):
        oc.require(doc['storage_qualification'] == checked['storage']
            and doc['compact_native_storage'] is True and doc['storage_backend'] == storage.STORAGE_BACKEND,
            'Query storage differs from pinned job proof')
        oc.require(doc['native_instance_backend'] == 'fast' and doc['render_budget_profile'] == 'reference56'
            and doc['render_profile_experiment'] is False, 'Only fast/reference56 is registered')
    oc.require(request['experimental_budget_override'] is False and result['experimental_short_profile'] is False
        and result['render_budget_subframes_per_view'] == result['total_requested_subframes_per_view'] == 56
        and result['target_count'] == len(checked['plan']['target_cases'])
        and result['source_cap_reset'] is False, 'Experimental query result or cap reset rejected')
    review = audit.audit_capture(capture, submitted, result_sha256=result_pin)
    oc.require(review['state'] == audit.AUDIT_STATE and review['capture'] == str(capture)
        and review['plan_sha256'] == pin and review['request_sha256'] == request_pin
        and review['result_sha256'] == result_pin
        and all(review[k] == v for k, v in ANNOTATION.items()), 'Wrong postexit query audit receipt')
    captured = [r for r in result['records'] if r['state'] == 'native_captured_pending_review']
    by_target = Counter()
    for row in captured:
        sample = oc.safe_file(capture, row['candidate_id'])
        reader = SampleReader(sample, expected_bindings={'sample.json': row['sample_sha256']})
        meta, expected = reader.metadata, budget_evidence('reference56', 0)
        oc.require(meta['native_instance_backend'] == 'fast'
            and meta['render_budget'] == row['render_budget']
            and all(meta['render_budget'].get(k) == v for k, v in expected.items())
            and meta['render_budget']['actual_orchestrator_requests'] == 7
            and meta['synchronization']['render_budget_subframes'] == 56,
            'Per-sample native backend/budget differs from fixed reference')
        by_target[row['target_id']] += 1
    oc.require(len(captured) == len(review['records']) == result['captured_frames']
        and sum(review['counts'].values()) == len(captured), 'Incomplete postexit replay')
    _matched(Path(matched['plan_path']), pin, checked['query']['matched_short'], checked['plan'])
    oc.bind_all(checked['bindings'])
    storage.verify_checked_qualification(checked['storage'])
    _no_failure(capture)
    audit_path = oc.new_destination(folder/'query_v2_postexit_audit.json', [capture, submitted.parent, *checked['roots']])
    oc.write_new(audit_path, review)
    audit_pin = oc.sha256(audit_path)
    comparison = dict(schema='greenhouse.native_query_reference_comparison_handoff.v1',
        old=deepcopy(matched), new=dict(capture_path=str(capture), plan_path=str(submitted), plan_sha256=pin,
            request_sha256=request_pin, result_sha256=result_pin, audit_path=str(audit_path), audit_sha256=audit_pin,
            **ANNOTATION), backend='fast', new_render_budget='reference56',
        old_budget_rule='first_captured_frame_56_then_8_verify_each_actual_sample',
        planned_proposals=checked['count'], captured_frames=len(captured),
        captured_by_target={c['target_id']: by_target[c['target_id']] for c in checked['plan']['target_cases']},
        comparisons_performed=False, old_annotation_epoch_unchanged=True,
        native_query_qualification_granted=False, short_budget_qualified=False,
        equivalence_cutoff=None, training_diversity_increment=0, source_cap_reset=False, training_approved=False)
    return dict(request_sha256=request_pin, result_sha256=result_pin, audit_path=str(audit_path),
        audit_sha256=audit_pin, counts=review['counts'], matched_comparison=comparison,
        training_diversity_increment=0, **ANNOTATION, bindings={str(audit_path): audit_pin})
