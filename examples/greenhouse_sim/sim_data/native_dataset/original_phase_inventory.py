"""Read-only inventory for serial_phases original_batch.v1 receipts.

Replay the exact published request and ordered file handoffs, then independently
replay each explicitly selected original batch. Earlier jobs and qualification
controls authenticate ordering only; their observations are never added. Saved
process declarations do not constitute independent OS attestation. This reader
does not launch, mutate captures, approve training, or reset source caps.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import inventory as inv, original_inventory as v1, original_inventory_v2 as v2
from . import serial_phases as q
from ..native_original_capture import audit, contracts as oc

POLICY = 'greenhouse.original_phase_observed_adapter.v1'
PRODUCER_SHA256 = 'c33425d0775abff783d8b1559903b92ad1cc5920d694fb921460660b9a57d49b'
_CODE = q.implementation_bindings()
oc.require(_CODE[str(Path(q.__file__).resolve())] == PRODUCER_SHA256, 'Unexpected phase producer')
_LOADED = oc.merge_bindings(_CODE, {str(Path(__file__).resolve()): oc.sha256(__file__)})


@dataclass(frozen=True)
class PhaseCapturePin:
    capture_path: str
    result_sha256: str
    launcher_receipt_path: str
    launcher_receipt_sha256: str


def implementation_bindings():
    oc.bind_all(_LOADED)
    oc.require(q.implementation_bindings() == _CODE, 'Loaded phase producer changed')
    return dict(_LOADED)


def _flags(doc):
    oc.require(all(doc[k] is v for k, v in q.FLAGS.items()), 'Phase nonpromotion flags differ')


def _producer(doc):
    _flags(doc)
    oc.require(doc['producer_module'] == q.PRODUCER_MODULE
        and doc['producer_implementation_bindings'] == _CODE, 'Exact phase producer required')


def _bound(b, path):
    path = inv._path(path)
    oc.require(str(path) in b.hashes, 'Unbound phase prerequisite: ' + str(path))
    return b.document(path)


def _absent(state, root):
    state['absent'].update(inv._path(root) / n for n in ('failure.json', 'superseded_idle_queue.json'))
    oc.require(not any(p.exists() for p in state['absent']), 'Failed or superseded evidence')


def _saved_predecessor(b, root, request, state):
    """Replay saved terminal/receipts without asking whether an old PID is live."""
    pred = request['predecessor']
    q.legacy._keys(pred, ('kind', 'request_path', 'request_sha256', 'supervisor'), 'predecessor')
    oc.require(pred['kind'] == 'original_serial39.v1', 'Wrong phase predecessor')
    q._identity(pred['supervisor'], inv._path(pred['request_path']), pred['request_sha256'])
    config = b.document(pred['request_path'], pred['request_sha256'])
    prior = _bound(b, root / 'predecessor_completion.json')
    _flags(prior)
    oc.require(prior['state'] == 'serial39_completed_and_exact_supervisor_absent'
        and prior['supervisor'] == pred['supervisor']
        and prior['process_exit_evidence'] == 'exact_PID_absent_from_successful_CIM_query_not_OS_exit_code',
        'Saved phase predecessor identity differs')
    b.mapping(prior['bindings'])
    old = inv._path(config['output'])
    _absent(state, old)
    published, terminal = (_bound(b, old / n) for n in ('request.json', 'result.json'))
    code = q.legacy.implementation_bindings()
    oc.require(published == dict(config, producer_module=q.legacy.PRODUCER_MODULE,
        producer_implementation_bindings=code, training_approved=False, source_cap_reset=False),
        'Published serial39 request differs')
    q.legacy._flags(terminal)
    oc.require(terminal['schema'] == config['schema'] == q.legacy.SCHEMA
        and terminal['state'] == q.legacy.RESULT_STATE and config['max_planned_cases'] == 39
        and terminal['planned_cases'] == 39 and len(config['batches']) == len(terminal['records']) == 2
        and terminal['producer_module'] == q.legacy.PRODUCER_MODULE
        and terminal['producer_implementation_bindings'] == code
        and terminal['queue_request_path'] == str(old / 'request.json')
        and terminal['queue_request_sha256'] == b.hashes[str(old / 'request.json')],
        'Incomplete serial39 terminal')
    last_path = old / 'batch_002' / 'original_launcher_receipt.json'
    last = _bound(b, last_path)
    _, _, replay = v2.original_observed_rows(v2.SerialCapturePin(str(last_path.parent / 'capture'),
        last['result_sha256'], str(last_path), b.hashes[str(last_path)]))
    expected = oc.merge_bindings(replay, {str(old / n): b.hashes[str(old / n)] for n in ('request.json', 'result.json')})
    oc.require(prior['bindings'] == expected, 'Saved serial39 replay bindings differ')
    for i, ((family, count), row) in enumerate(zip(q.legacy.BATCH_SHAPE, terminal['records'], strict=True), 1):
        folder = old / f'batch_{i:03d}'
        _absent(state, folder / 'capture')
        path = folder / 'original_launcher_receipt.json'
        receipt = _bound(b, path)
        review = _bound(b, receipt['audit_path'])
        oc.require(row == dict(batch_index=i, family=family, planned_cases=count, capture=str(folder / 'capture'),
            result_sha256=receipt['result_sha256'], audit_counts=review['counts'], launcher_receipt_path=str(path),
            launcher_receipt_sha256=b.hashes[str(path)]), 'Ordered serial39 handoff differs')
    b.protected_roots.add(str(old))
    return prior


def _request(b, receipt, state):
    path = inv._path(receipt['input_request_path'])
    request = b.document(path, receipt['input_request_sha256'])
    key = str(path)
    if key in state['requests']:
        return state['requests'][key]
    q.legacy._keys(request, ('schema', 'output', 'isaac_python', 'native_deps', 'max_jobs',
        'max_planned_frames', 'planned_frames', 'predecessor', 'phases'), 'phase request')
    oc.require(request['schema'] == q.SCHEMA and type(request['max_jobs']) is int and request['max_jobs'] == 64
        and type(request['max_planned_frames']) is int and request['max_planned_frames'] == 4096,
        'Frozen phase request bounds required')
    root = inv._path(request['output'])
    _absent(state, root)
    published = _bound(b, root / 'request.json')
    oc.require(published == dict(request, input_request_path=str(path),
        input_request_sha256=receipt['input_request_sha256'], producer_module=q.PRODUCER_MODULE,
        producer_implementation_bindings=_CODE, **q.FLAGS), 'Phase publication differs from input')
    phases, jobs, phase_ids, job_ids = request['phases'], [], set(), set()
    seen_controls = {k: set() for k in q.QUALIFICATIONS}
    oc.require(isinstance(phases, list) and 1 <= len(phases) <= 64, 'Bounded nonempty phases required')
    for phase in phases:
        q.legacy._keys(phase, ('id', 'jobs'), 'phase')
        q._unique_id(phase['id'], phase_ids)
        oc.require(isinstance(phase['jobs'], list) and phase['jobs'], 'Nonempty phase jobs required')
        for job in phase['jobs']:
            kind = job['worker_kind']
            helper = q.QUALIFICATIONS.get(kind)
            extra = helper.JOB_FIELDS if helper else ()
            q.legacy._keys(job, ('id', 'worker_kind', 'plan_path', 'plan_sha256', *extra), 'job')
            q._unique_id(job['id'], job_ids)
            oc.require(len(jobs) < 64, 'Too many phase jobs')
            if helper:
                seen = seen_controls[kind]
                oc.require(len(seen) < q.QUALIFICATION_LIMITS[kind] and job['plan_sha256'] not in seen,
                    'Repeated qualification control')
                seen.add(job['plan_sha256'])
            reused = [j for j in jobs if inv._path(j['plan_path']) == inv._path(job['plan_path'])
                or j['plan_sha256'] == job['plan_sha256']]
            oc.require(not reused or (len(reused) == 1 and q._seed101_control_reuse(reused[0], job, q.persistence_kind.KIND)),
                'Aliased source plan')
            checked = q.validate_plan(kind, job['plan_path'], job['plan_sha256'],
                query={k: deepcopy(job[k]) for k in extra} if helper else None)
            b.mapping(checked['bindings'])
            b.protected_roots.update(str(inv._path(p)) for p in checked['roots'])
            jobs.append(dict(phase_id=phase['id'], **deepcopy(job), checked=checked))
    oc.require(type(request['planned_frames']) is int
        and request['planned_frames'] == sum(j['checked']['count'] for j in jobs) <= 4096, 'Wrong planned frame total')
    prior = _saved_predecessor(b, root, request, state)
    initial = oc.merge_bindings(_CODE, {str(path): receipt['input_request_sha256'],
        request['predecessor']['request_path']: request['predecessor']['request_sha256']},
        *(j['checked']['bindings'] for j in jobs), prior['bindings'],
        {str(root / n): b.hashes[str(root / n)] for n in
            ('request.json', 'predecessor_completion.json', 'checkpoint_000001.json')})
    _checkpoint(b, root, 1, [], receipt['input_request_sha256'], 'clean_initial_boundary')
    value = dict(root=root, request=request, jobs=jobs, prior=prior, bindings=initial, completed=[], envelopes=[])
    state['requests'][key] = value
    return value


def _checkpoint(b, root, number, records, request_sha, state):
    doc = _bound(b, root / f'checkpoint_{number:06d}.json')
    _flags(doc)
    oc.require(doc['schema'] == q.CHECKPOINT_SCHEMA and doc['state'] == state
        and doc['request_sha256'] == request_sha and doc['completed'] == records
        and type(doc['next_job_index']) is int and doc['next_job_index'] == len(records), 'Phase checkpoint differs')


def _envelope(b, context, index, state):
    root, request, jobs = (context[k] for k in ('root', 'request', 'jobs'))
    while len(context['envelopes']) <= index:
        i = len(context['envelopes'])
        job = jobs[i]
        folder = root / job['phase_id'] / job['id']
        capture, submitted, path = folder / 'capture', folder / 'submitted' / 'plan.json', folder / 'receipt.json'
        _absent(state, capture)
        receipt = _bound(b, path)
        _producer(receipt)
        expected = oc.merge_bindings(context['bindings'], {str(submitted): job['plan_sha256']})
        oc.require(receipt['schema'] == q.RECEIPT_SCHEMA
            and receipt['state'] == 'owned_exit0_postexit_audited_pending_admission'
            and type(receipt['exit_code']) is int and receipt['exit_code'] == 0
            and type(receipt['job_index']) is int and receipt['job_index'] == i
            and receipt['phase_id'] == job['phase_id'] and receipt['job_id'] == job['id']
            and receipt['worker_kind'] == job['worker_kind'] and receipt['source_plan_path'] == job['plan_path']
            and receipt['plan_sha256'] == job['plan_sha256'] and receipt['submitted_plan_path'] == str(submitted)
            and receipt['capture'] == str(capture) and receipt['planned_source_frames'] == job['checked']['count']
            and receipt['input_request_path'] == str(inv._path(context['request_path']))
            and receipt['input_request_sha256'] == context['request_sha256']
            and receipt['source_bindings'] == expected, 'Phase job/ordered source handoff differs')
        b.mapping(expected)
        worker = receipt['owned_worker']
        command = q.worker_command(request['isaac_python'], job['worker_kind'], submitted, job['plan_sha256'],
            capture, checked=job['checked'])
        event_path = folder / 'launch.json'
        oc.require(type(worker['worker_pid']) is int and worker['worker_pid'] > 0 and worker['command'] == command
            and worker['event_path'] == str(event_path), 'Owned phase worker differs')
        event = b.document(event_path, worker['event_sha256'])
        _flags(event)
        oc.require(event['schema'] == q.RECEIPT_SCHEMA and event['state'] == 'owned_child_running'
            and event['job_index'] == i and event['phase_id'] == job['phase_id'] and event['job_id'] == job['id']
            and event['worker_pid'] == worker['worker_pid'] and event['command'] == command
            and event['bindings'] == expected and event['resource_evidence'] == worker['resource_evidence']
            and event['owner_visibility'] == worker['owner_visibility'], 'Phase launch event differs')
        _resources(b, worker)
        reviewed = receipt['audit']
        b.mapping(reviewed['bindings'])
        oc.require(reviewed['bindings'].get(str(capture / 'request.json')) == reviewed['request_sha256']
            and reviewed['bindings'].get(str(capture / 'result.json')) == reviewed['result_sha256']
            and reviewed['bindings'].get(reviewed['audit_path']) == reviewed['audit_sha256'], 'Unbound postexit audit')
        context['envelopes'].append((receipt, job, capture, submitted))
        row = dict(phase_id=job['phase_id'], job_id=job['id'], worker_kind=job['worker_kind'],
            receipt_path=str(path), receipt_sha256=b.hashes[str(path)])
        context['completed'].append(row)
        context['bindings'] = oc.merge_bindings(expected, reviewed['bindings'],
            {str(path): b.hashes[str(path)], str(event_path): worker['event_sha256']})
        # The selected last receipt need not claim a later checkpoint/completion.
        if i < index:
            _advance_checkpoint(b, context, i)
    return context['envelopes'][index]


def _advance_checkpoint(b, context, index):
    name = context['root'] / f'checkpoint_{index + 2:06d}.json'
    _checkpoint(b, context['root'], index + 2, context['completed'][:index + 1],
        context['request_sha256'], 'owned_exit0_audited_clean_boundary')
    context['bindings'][str(name)] = b.hashes[str(name)]


def _resources(b, worker):
    evidence = worker['resource_evidence']
    memory, process = evidence['memory'], evidence['process_inventory']
    oc.require(evidence['allowed'] is True and memory['checked'] is True and memory['allowed'] is True
        and type(memory['commit_headroom_bytes']) is int and memory['commit_headroom_bytes'] >= 20 * q.legacy.GIB
        and type(evidence['disk_free_bytes']) is int and evidence['disk_free_bytes'] >= 60 * q.legacy.GIB
        and process['blockers'] == [] and process['classifications'] and process['no_blockers_observed'] is True
        and evidence['exclusive_launch_guaranteed'] is False, 'Stored launch resources insufficient')
    owner = worker['owner_visibility']
    decision = q.legacy.guard.classify_process(owner['metadata'], native_roots=q.legacy.guard.DEFAULT_NATIVE_ROOTS)
    oc.require(owner['current_pid_exception_used'] is False and owner['running_code_attested'] is False
        and owner['classification'] == decision and decision['blocking'] is False
        and decision['classification'] == 'unrelated_python', 'Phase owner guard declaration differs')
    for proof in process.get('bridge_proofs', []):
        b.mapping(proof['bindings'])


def _original(b, pin, state):
    oc.require(isinstance(pin, PhaseCapturePin), 'Explicit PhaseCapturePin required')
    capture, path = inv._path(pin.capture_path), inv._path(pin.launcher_receipt_path)
    receipt = b.document(path, pin.launcher_receipt_sha256)
    _producer(receipt)
    oc.require(receipt['worker_kind'] == 'original_batch.v1', 'Qualification controls are not original observations')
    b.mapping(receipt['source_bindings'])
    context = _request(b, receipt, state)
    context.update(request_path=receipt['input_request_path'], request_sha256=receipt['input_request_sha256'])
    index = receipt['job_index']
    oc.require(type(index) is int and 0 <= index < len(context['jobs']), 'Invalid phase job index')
    # Complete a checkpoint omitted when an earlier explicit pin was last.
    if len(context['envelopes']) and len(context['envelopes']) <= index:
        _advance_checkpoint(b, context, len(context['envelopes']) - 1)
    checked, job, actual_capture, submitted = _envelope(b, context, index, state)
    oc.require(checked == receipt and actual_capture == capture and path == capture.parent / 'receipt.json',
        'Explicit phase capture identity differs')
    reviewed = receipt['audit']
    oc.require(reviewed['result_sha256'] == pin.result_sha256
        and reviewed['audit_path'] == str(capture.parent / 'original_postexit_audit.json'), 'Wrong original audit location')
    result = b.document(capture / 'result.json', pin.result_sha256)
    request = b.document(capture / 'request.json', reviewed['request_sha256'])
    oc.require(result['schema'] == request['schema'] == oc.RESULT_SCHEMA
        and result['state'] == 'original_native_capture_complete_automatically_audited'
        and result['admission'] == oc.ADMISSION and result['request_sha256'] == reviewed['request_sha256']
        and result['plan_sha256'] == request['plan_sha256'] == receipt['plan_sha256']
        and request['plan_path'] == str(submitted), 'Original submitted request/result mismatch')
    automatic = b.document(capture / 'automatic_audit.json', result['automatic_audit_sha256'])
    postexit = b.document(reviewed['audit_path'], reviewed['audit_sha256'])
    replay = audit.audit_capture(capture, result_sha256=pin.result_sha256)
    oc.require(replay == postexit == automatic and replay['counts'] == reviewed['counts'],
        'Original phase audit differs from fresh native replay')
    tree = q._tree_pins(capture)
    oc.require(reviewed['bindings'] == oc.merge_bindings(tree, {reviewed['audit_path']: reviewed['audit_sha256']}),
        'Original postexit capture membership differs')
    # Explicit field projection, not an invented legacy producer receipt.
    projection = dict(request_sha256=reviewed['request_sha256'], plan_path=str(submitted),
        plan_sha256=receipt['plan_sha256'], owned_worker=receipt['owned_worker'], audit_sha256=reviewed['audit_sha256'])
    rows, summary = v2._project_original((capture, projection, result, request, job['checked']['plan'], replay),
        b, pin, state['absent'])
    contract = dict(schema=receipt['schema'], state=receipt['state'], producer_module=q.PRODUCER_MODULE,
        producer_implementation_bindings=deepcopy(_CODE), input_request_path=receipt['input_request_path'],
        input_request_sha256=receipt['input_request_sha256'], job_index=index, phase_id=receipt['phase_id'],
        job_id=receipt['job_id'], source_plan_path=receipt['source_plan_path'],
        predecessor_completion_path=str(context['root'] / 'predecessor_completion.json'),
        predecessor_completion_sha256=b.hashes[str(context['root'] / 'predecessor_completion.json')],
        resource_evidence_basis='pinned_producer_declaration_not_live_OS_attestation')
    for row in rows:
        row['provenance'].update(adapter_policy=POLICY, phase_launcher=deepcopy(contract))
    summary.update(phase_launcher=contract, preceding_job_observations_implicitly_added=False)
    b.protected_roots.update((str(context['root']), str(capture)))
    return rows, summary


def build_inventory(captures):
    """Explicit original observations only; fresh cumulative selection is separate."""
    try:
        pins = list(captures)
        oc.require(pins and all(isinstance(p, PhaseCapturePin) for p in pins), 'Nonempty explicit phase pins required')
        oc.require(len({str(inv._path(p.capture_path)) for p in pins}) == len(pins), 'Duplicate phase capture')
        b, state = inv._Bindings(), dict(requests={}, absent=set())
        code = implementation_bindings()
        b.mapping(code)
        rows, coverage, contexts = [], [], {}
        for pin in sorted(pins, key=lambda p: str(p.capture_path)):
            added, summary = _original(b, pin, state)
            contexts.update(summary.pop('scene_contexts'))
            rows.extend(added); coverage.append(summary)
        rows.sort(key=lambda r: r['sample_id'])
        oc.require(len({r['sample_id'] for r in rows}) == len(rows), 'Duplicate phase observation')
        v1._finish(b, state['absent'])
        result = dict(schema=inv.SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
            state='observed_inventory_only_not_global_admission', adapter_policy=POLICY, records=rows, scene_contexts=contexts,
            counts=dict(captured_rows=len(rows), receipts=len(coverage), decisions=dict(Counter(r['decision'] for r in rows)),
                source_kinds=dict(Counter(r['source_kind'] for r in rows)),
                original_donor_families=dict(Counter(r['source_family'] for r in rows)),
                conservative_view_cap_groups=dict(Counter(r['source_target'] for r in rows)),
                variant_targets=dict(Counter(r['target_id'] for r in rows))), exact_duplicates=inv._duplicates(rows),
            scope=dict(explicit_receipts=coverage, extra_observed_roots=[], receipt_discovery_performed=False,
                capture_roots_scanned=False, global_complete=False, phase_completion_claimed=False,
                prerequisite_observations_implicitly_added=False, qualification_controls_implicitly_added=False,
                observed_split='train', heldout_inventory_included=False, historical_training_rows=0,
                near_image_comparison_performed=False, morphology_equivalence_finalized=False,
                scene_identity_policy=v1.SCENE_POLICY, renderer_settings_used_to_grant_view_identity=False,
                audit_execution_independently_verified=True, label_derivation_replayed=True, trace_derivation_replayed=True,
                callback_validation_scope='independent original native ID/RGB/Z/camera replay',
                audit_decision_counts_basis='independent original native replay'),
            provenance=dict(source_bindings=dict(sorted(b.hashes.items())), read_only_source_roots=sorted(b.protected_roots),
                implementation_bindings=code), training_approved=False, source_cap_reset=False, admission_performed=False,
            original_reviews_modified=False, captures_modified=False, images_generated=False,
            depth_recomputed=False, compact_qualification_granted=False)
        result['sha256'] = inv._hash(result)
        return result
    except (KeyError, TypeError, IndexError, OSError) as exc:
        raise ValueError('Malformed/missing original phase evidence: ' + str(exc)) from exc


def original_observed_rows(pin):
    result = build_inventory([pin])
    summary = deepcopy(result['scope']['explicit_receipts'][0])
    summary['scene_contexts'] = result['scene_contexts']
    return result['records'], summary, result['provenance']['source_bindings']
