"""Read-only serial-original receipt adapter; never launch or select data.

SerialCapturePin has the same four explicit path/hash fields as V1's pin, but
accepts ONLY serial_queue's versioned owned-batch receipt. V5 receipts remain
with frozen original_inventory. No fabricated V5 event/state, monkeypatch,
validate_request(completed_output), live CIM query, or native process call.

original_observed_rows(pin) -> rows, capture_summary, full bindings
build_inventory(serial_captures, *, v5_originals=(), generated_receipts=(),
                extra_observed_roots=()) -> common observed-inventory packet

The exact frozen serial producer and its loaded helper map are mandatory. Saved
pilot/predecessor declarations are joined to their actual pinned files. Pilot
and selected batch native audits are replayed. A batch-2 pin also authenticates
and replays batch 1, but its rows are NOT silently included: supply both pins
for both observations. No whole-queue/global completion is inferred from a
per-batch receipt, and 39 scheduled cases are never counted as 39 observations.
Launcher/process/resource declarations are not independent OS attestation.

Physical identities, original ancestry, nonpromotion and source caps retain
V1 semantics. Its small row projection is copied below with an explicit checked
input tuple; the original validators and audit remain unchanged and code-bound.
All captured holds/exclusions survive; pre-render holds remain coverage only.
New inventory seals require fresh cumulative duplicate coverage before counting.
"""
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import inventory as inv
from . import original_inventory as v1
from .bundle import SampleReader, safe_path
from ..native_original_capture import contracts as oc
from ..native_original_capture import audit as original_audit
from ..native_original_capture import serial_queue as producer


POLICY = 'greenhouse.original_serial_observed_adapter.v2'
PRODUCER_SHA256 = '38b98b3523a0a30eb75426fbe4090a259e454d5dc94ad1daca37e8f2180e188c'
_PRODUCER_PATH = str(Path(producer.__file__).resolve())
_PRODUCER_CODE = producer.implementation_bindings()
oc.require(_PRODUCER_CODE.get(_PRODUCER_PATH) == PRODUCER_SHA256, 'Unexpected frozen serial producer')
_LOADED_CODE = oc.merge_bindings(_PRODUCER_CODE, {str(Path(__file__).resolve()): oc.sha256(__file__)})
SCENE_POLICY = v1.SCENE_POLICY
_original_basis, _features, _identities = v1._original_basis, v1._features, v1._identities


@dataclass(frozen=True)
class SerialCapturePin:
    capture_path: str
    result_sha256: str
    launcher_receipt_path: str
    launcher_receipt_sha256: str


def implementation_bindings():
    oc.bind_all(_LOADED_CODE)
    oc.require(producer.implementation_bindings() == _PRODUCER_CODE, 'Loaded serial producer changed')
    return dict(_LOADED_CODE)


def _producer(document):
    producer._flags(document)
    oc.require(document['producer_module'] == producer.PRODUCER_MODULE
        and document['producer_implementation_bindings'] == _PRODUCER_CODE,
        'Exact loaded serial producer/helper bindings required')


def _bound_document(bindings, path):
    path = inv._path(path)
    oc.require(str(path) in bindings.hashes, 'Serial prerequisite is not source-bound: ' + str(path))
    return bindings.document(path)


def _saved_predecessor(bindings, root, request, config, absent):
    """Static replay of producer's saved handoff joins, NOT a new PID query."""
    prior = _bound_document(bindings, root / 'predecessor_completion.json')
    producer._flags(prior)
    pred = request['predecessor']
    producer._keys(pred, ('request_path', 'request_sha256', 'supervisor'), 'V5 predecessor')
    producer._keys(pred['supervisor'], ('pid', 'creation_date', 'executable', 'command_line'), 'V5 supervisor')
    producer._keys(request['pilot'], ('capture_path', 'result_sha256', 'launcher_receipt_path',
        'launcher_receipt_sha256'), 'pilot')
    oc.require(prior['state'] == 'exact_v5_terminal_and_supervisor_absent'
        and prior['supervisor'] == pred['supervisor']
        and prior['process_exit_evidence'] == 'exact_PID_absent_from_successful_CIM_query_not_OS_exit_code',
        'Saved predecessor identity/exit declaration differs')
    supervisor = pred['supervisor']
    oc.require(type(supervisor['pid']) is int and supervisor['pid'] > 0
        and producer.guard._born({'CreationDate': supervisor['creation_date']}) is not None,
        'Saved supervisor identity required')
    argv = producer.guard._argv(supervisor['command_line'])
    oc.require(argv and inv._path(argv[0]) == inv._path(supervisor['executable']), 'Saved supervisor executable differs')
    args = argv[1:]
    while args and args[0] in ('-B', '-u'):
        args = args[1:]
    oc.require(args == ['-m', 'sim_data.native_capture_v5.campaign_queue', '--request',
        str(inv._path(pred['request_path'])), '--request-sha256', pred['request_sha256']],
        'Saved supervisor command differs from predecessor request')
    v5root, scale = inv._path(config['output']), inv._path(config['scale_output']) / 'result.json'
    fixed = {str(v5root / 'request.json'), str(v5root / 'result.json'), str(scale)}
    pins = prior['bindings']
    extra = set(pins) - fixed
    oc.require(fixed <= pins.keys() and len(extra) == 1 and len(pins) == 4, 'Exact predecessor handoff bindings required')
    event_path = inv._path(next(iter(extra)))
    oc.require(event_path.parent == v5root and inv.re.fullmatch(r'queue_\d{6}.json', event_path.name),
        'Saved predecessor event escaped V5 output')
    for path, expected in pins.items():
        oc.require(bindings.hashes.get(path) == expected, 'Receipt/predecessor handoff pin mismatch')
    bindings.mapping(pins)
    terminal, published = bindings.document(v5root / 'result.json'), bindings.document(v5root / 'request.json')
    producer._flags(terminal)
    oc.require(terminal['state'] == 'serial_probe_and_scale_complete_pending_admission'
        and terminal['request_sha256'] == pred['request_sha256']
        and terminal['scale_result_sha256'] == pins[str(scale)], 'Wrong predecessor terminal/scale join')
    oc.require({k: published[k] for k in config} == config and published['training_approved'] is False,
        'Published V5 request differs')
    bindings.mapping(published['implementation_bindings'])
    for module in (producer.v5, producer.campaign, producer.guard):
        path = str(Path(module.__file__).resolve())
        oc.require(published['implementation_bindings'].get(path) == _PRODUCER_CODE[path], 'Wrong V5 producer binding')
    scale_result = bindings.document(scale)
    producer._flags(scale_result)
    oc.require(isinstance(scale_result['records'], list) and scale_result['records']
        and scale_result['new_biological_families'] == 0, 'Incomplete predecessor scale declaration')
    event = bindings.document(event_path)
    producer._flags(event)
    oc.require(event['state'] == 'scale_coordinator_running_in_same_cpu_process'
        and event['worker_pid'] == supervisor['pid']
        and all(event['bindings'].get(p) == h for p, h in published['implementation_bindings'].items()),
        'Wrong predecessor supervisor event')
    bindings.mapping(event['bindings'])
    absent.update((v5root / 'failure.json', v5root / 'superseded_idle_queue.json'))
    bindings.protected_roots.update((str(v5root), str(scale.parent)))
    return prior


def _saved_queue(bindings, receipt, state):
    path = inv._path(receipt['queue_request_path'])
    oc.require(receipt['source_bindings'].get(str(path)) == receipt['queue_request_sha256'],
        'Published queue request is not receipt-bound')
    request = bindings.document(path, receipt['queue_request_sha256'])
    key = str(path)
    if key in state['queues']:
        return state['queues'][key]
    root = path.parent
    oc.require(path == root / 'request.json' and inv._path(request['output']) == root, 'Serial published queue location differs')
    _producer(request)
    producer._keys(request, ('schema', 'output', 'isaac_python', 'native_deps', 'max_planned_cases',
        'pilot', 'predecessor', 'batches', 'producer_module', 'producer_implementation_bindings',
        'training_approved', 'source_cap_reset'), 'published serial request')
    oc.require(request['schema'] == producer.SCHEMA and type(request['max_planned_cases']) is int
        and request['max_planned_cases'] == 39 and isinstance(request['batches'], list)
        and len(request['batches']) == 2, 'Exact ordered 39-case serial request required')
    plans, seen = [], set()
    for entry, (family, count) in zip(request['batches'], producer.BATCH_SHAPE, strict=True):
        producer._keys(entry, ('plan_path', 'plan_sha256'), 'serial batch')
        plan_path = inv._path(entry['plan_path'])
        oc.require(bindings.hashes.get(str(plan_path)) == entry['plan_sha256'], 'Queue batch plan is not receipt-bound')
        plan = bindings.document(plan_path, entry['plan_sha256'])
        oc.require(plan['schema'] == oc.SCHEMA and plan['source_family'] == family
            and plan['split'] == 'train' and plan['family_assignments'] == oc.FROZEN_SPLITS
            and plan['geometry_mode'] == 'unmodified_original' and plan['admission'] == oc.ADMISSION
            and plan['scene_policy'] == oc.SCENE_POLICY and plan['implementation_bindings'] == oc.LOADED_IMPLEMENTATION
            and plan['stage_reuse'] == 'one_original_donor_scene_one_native_product'
            and type(plan['sample_count_limit']) is int and plan['sample_count_limit'] == count
            and [c['case_id'] for c in plan['cases']] == [f'sample_{i:04d}' for i in range(1, count + 1)],
            'Frozen original serial batch scope changed')
        for case in plan['cases']:
            target = case['target_id']
            oc.require(target.startswith(family + '/') and case['source_row']['target_id'] == target
                and case['conservative_view_cap_group'] == target and case['pose_prior']['prior_target_id'] == target
                and case['pose_request'] == dict(mode='exact_prior', native_pixel_xy=None), 'Serial original target/prior/cap differs')
            camera = inv._hash([family, case['expected_calibration']])
            oc.require(camera not in seen, 'Duplicate scheduled actual camera')
            seen.add(camera)
        plans.append(plan)
    oc.require(len({inv._path(e['plan_path']) for e in request['batches']}) == 2, 'Duplicate serial plan')
    pred = request['predecessor']
    config_path = inv._path(pred['request_path'])
    oc.require(bindings.hashes.get(str(config_path)) == pred['request_sha256'], 'Predecessor input is not source-bound')
    config = bindings.document(config_path, pred['request_sha256'])
    pilot = request['pilot']
    for p, h in ((inv._path(pilot['capture_path']) / 'result.json', pilot['result_sha256']),
                 (inv._path(pilot['launcher_receipt_path']), pilot['launcher_receipt_sha256'])):
        oc.require(receipt['source_bindings'].get(str(p)) == h, 'Pilot is not receipt-bound')
    oc.require(config['schema'] == producer.v5.SCHEMA
        and inv._path(request['pilot']['capture_path']) == inv._path(config['output']) / 'original_capture',
        'Serial pilot/predecessor mismatch')
    prior = _saved_predecessor(bindings, root, request, config, state['absent'])
    qualification = _bound_document(bindings, root / 'pilot_qualification.json')
    # The unchanged public V1 helper independently authenticates/replays ONLY
    # the prerequisite pilot. Its observation is not silently added to rows.
    actual = producer.qualify_pilot(request['pilot'], config)
    oc.require(qualification == actual, 'Saved serial pilot qualification differs from fresh replay')
    bindings.mapping(actual['verified_bindings'])
    context = dict(root=root, request=request, plans=plans, predecessor=prior,
        qualification=qualification, queue_request_path=key, queue_request_sha256=receipt['queue_request_sha256'])
    state['queues'][key] = context
    bindings.protected_roots.add(str(root))
    return context


def _serial(bindings, pin, state):
    oc.require(isinstance(pin, SerialCapturePin), 'Explicit SerialCapturePin required')
    capture, receipt_path = inv._path(pin.capture_path), inv._path(pin.launcher_receipt_path)
    receipt = bindings.document(receipt_path, pin.launcher_receipt_sha256)
    key = str(capture)
    identity = (str(receipt_path), pin.launcher_receipt_sha256, pin.result_sha256)
    if key in state['receipts']:
        cached = state['receipts'][key]
        oc.require(cached['identity'] == identity, 'Conflicting serial capture pins')
        return cached
    oc.require(capture.is_dir() and not (capture / 'failure.json').exists(), 'Missing/failed serial capture')
    _producer(receipt)
    oc.require(receipt['schema'] == producer.RECEIPT_SCHEMA and receipt['state'] == producer.RECEIPT_STATE
        and type(receipt['exit_code']) is int and receipt['exit_code'] == 0
        and type(receipt['batch_index']) is int and receipt['batch_index'] in (1, 2), 'Versioned owned serial exit-0 receipt required')
    bindings.mapping(receipt['source_bindings'])
    oc.require(all(receipt['source_bindings'].get(p) == h for p, h in _PRODUCER_CODE.items()), 'Missing serial producer chain')
    context = _saved_queue(bindings, receipt, state)
    root, index = context['root'], receipt['batch_index']
    folder = root / f'batch_{index:03d}'
    entry, plan = context['request']['batches'][index - 1], context['plans'][index - 1]
    oc.require(capture == folder / 'capture' and receipt_path == folder / 'original_launcher_receipt.json'
        and inv._path(receipt['capture']) == capture and receipt['result_sha256'] == pin.result_sha256
        and inv._path(receipt['plan_path']) == inv._path(entry['plan_path'])
        and receipt['plan_sha256'] == entry['plan_sha256'], 'Serial batch/receipt/plan location mismatch')
    if index == 1:
        queued = context['request']
        pred, pilot = queued['predecessor'], queued['pilot']
        known = oc.merge_bindings(_PRODUCER_CODE, context['predecessor']['bindings'],
            {str(inv._path(e['plan_path'])): e['plan_sha256'] for e in queued['batches']},
            {str(inv._path(pred['request_path'])): pred['request_sha256'],
             str(inv._path(pilot['capture_path']) / 'result.json'): pilot['result_sha256'],
             str(inv._path(pilot['launcher_receipt_path'])): pilot['launcher_receipt_sha256']},
            {str(root / name): bindings.hashes[str(root / name)] for name in
                ('request.json', 'pilot_qualification.json', 'predecessor_completion.json')})
        sources = receipt['source_bindings']
        extra = set(sources) - set(known)
        oc.require(len(extra) == 1 and all(sources.get(p) == h for p, h in known.items()),
            'Exact initial serial input/handoff bindings required')
        # The producer did not emit a separate input-path field; its sole
        # remaining source binding is the external queue request it loaded.
        input_path = next(iter(extra))
        raw_request = {k: v for k, v in queued.items() if k not in
            ('producer_module', 'producer_implementation_bindings', 'training_approved', 'source_cap_reset')}
        oc.require(_bound_document(bindings, input_path) == raw_request, 'Serial input/publication differs')
        context.update(input_request_path=input_path, input_request_sha256=sources[input_path])
    if index > 1:
        previous_folder = root / f'batch_{index - 1:03d}'
        previous_path = previous_folder / 'original_launcher_receipt.json'
        previous = _bound_document(bindings, previous_path)
        prior_pin = SerialCapturePin(str(previous_folder / 'capture'), previous['result_sha256'],
            str(previous_path), receipt['source_bindings'].get(str(previous_path)))
        checked = _serial(bindings, prior_pin, state)
        expected = oc.merge_bindings(checked['receipt']['source_bindings'], checked['handoff'])
        oc.require(receipt['source_bindings'] == expected, 'Prior batch handoff bindings differ')
    worker = receipt['owned_worker']
    event_path = inv._path(worker['event_path'])
    oc.require(type(worker['worker_pid']) is int and worker['worker_pid'] > 0
        and worker['command'] == producer.batch_command(context['request']['isaac_python'], entry, capture)
        and event_path == folder / 'queue_000001.json', 'Owned serial command/event mismatch')
    event = bindings.document(event_path, worker['event_sha256'])
    producer._flags(event)
    oc.require(event['state'] == producer.LAUNCH_STATE and event['producer_module'] == producer.PRODUCER_MODULE
        and event['batch_index'] == index and event['worker_pid'] == worker['worker_pid']
        and event['command'] == worker['command'] and event['bindings'] == receipt['source_bindings'],
        'Serial event/receipt join differs')
    evidence = event['resource_evidence']
    process = evidence['process_inventory']
    oc.require(evidence['allowed'] is True and evidence['memory']['checked'] is True
        and evidence['memory']['allowed'] is True and type(evidence['memory']['commit_headroom_bytes']) is int
        and evidence['memory']['commit_headroom_bytes'] >= 20 * producer.GIB
        and type(evidence['disk_free_bytes']) is int and evidence['disk_free_bytes'] >= 60 * producer.GIB
        and process['blockers'] == [] and process['classifications'] and process['no_blockers_observed'] is True
        and evidence['exclusive_launch_guaranteed'] is False, 'Missing successful stored launch resource declaration')
    for proof in process.get('bridge_proofs', []):
        bindings.mapping(proof['bindings'])
    result = bindings.document(capture / 'result.json', pin.result_sha256)
    request = bindings.document(capture / 'request.json', receipt['request_sha256'])
    oc.require(result['schema'] == request['schema'] == oc.RESULT_SCHEMA
        and result['state'] == 'original_native_capture_complete_automatically_audited'
        and result['admission'] == oc.ADMISSION and result['request_sha256'] == receipt['request_sha256']
        and result['plan_sha256'] == request['plan_sha256'] == receipt['plan_sha256']
        and inv._path(request['plan_path']) == inv._path(entry['plan_path']), 'Serial request/result/plan mismatch')
    audit_path = inv._path(receipt['audit_path'])
    oc.require(audit_path == folder / 'original_postexit_audit.json', 'Serial post-exit audit escaped batch')
    postexit = bindings.document(audit_path, receipt['audit_sha256'])
    automatic = bindings.document(capture / 'automatic_audit.json', result['automatic_audit_sha256'])
    bindings.mapping(plan['source_bindings'])
    bindings.mapping(plan['implementation_bindings'])
    for name, expected in result['smoke_bindings'].items():
        bindings.bind(safe_path(capture, name), expected)
    replay = original_audit.audit_capture(capture, result_sha256=pin.result_sha256)
    oc.require(replay == postexit == automatic, 'Serial post-exit audit differs from independent native replay')
    rows, summary = _project_original((capture, receipt, result, request, plan, replay), bindings, pin, state['absent'])
    contract = dict(schema=receipt['schema'], state=receipt['state'], producer_module=receipt['producer_module'],
        producer_implementation_bindings=deepcopy(receipt['producer_implementation_bindings']), batch_index=index,
        queue_request_path=context['queue_request_path'], queue_request_sha256=context['queue_request_sha256'],
        input_request_path=context['input_request_path'], input_request_sha256=context['input_request_sha256'],
        pilot_qualification_path=str(root / 'pilot_qualification.json'),
        pilot_qualification_sha256=bindings.hashes[str(root / 'pilot_qualification.json')],
        predecessor_completion_path=str(root / 'predecessor_completion.json'),
        predecessor_completion_sha256=bindings.hashes[str(root / 'predecessor_completion.json')],
        predecessor_exit_evidence=context['predecessor']['process_exit_evidence'],
        resource_evidence_basis='pinned_producer_declaration_not_live_OS_attestation')
    for row in rows:
        row['provenance']['serial_launcher'] = deepcopy(contract)
    summary['serial_launcher'] = contract
    summary['verified_preceding_batches'] = list(range(1, index))
    bindings.protected_roots.update((str(folder), str(inv._path(entry['plan_path']).parent), plan['package']))
    bindings.protected_roots.update(str(Path(p).parent) for p in plan['source_bindings'])
    handoff = {str(receipt_path): pin.launcher_receipt_sha256, str(audit_path): receipt['audit_sha256'],
        str(event_path): worker['event_sha256'], str(capture / 'request.json'): receipt['request_sha256'],
        str(capture / 'result.json'): pin.result_sha256}
    checked = dict(identity=identity, rows=rows, summary=summary, receipt=receipt, handoff=handoff)
    state['receipts'][key] = checked
    return checked


# Row projection copied from frozen original_inventory.py SHA256
# 038b943dc97660bc6a5bf06cec6d474cdd28fcd4ca8c5581a87be1ec9d964028.
# Only the input tuple is supplied by the distinct serial validator above.
def _project_original(authenticated, bindings, pin, absent):
    capture, receipt, result, request, plan, replay = authenticated
    absent.add(capture / 'failure.json')
    cases, captured, audited = plan['cases'], result['capture']['records'], replay['records']
    ids = [c['case_id'] for c in cases]
    inv._require(ids == list(inv._unique(cases, 'case_id')) and len(ids) == plan['sample_count_limit']
                 and ids == [r['case_id'] for r in captured] == [r['case_id'] for r in audited],
                 'Missing/reordered/duplicate original cases')
    inv._require(replay['counts'] == dict(Counter(r['decision'] for r in audited)), 'Original audit counts differ')
    basis, ancestry = _original_basis(bindings, plan)
    rows, held, contexts, sequences, rgb_seen = [], [], {}, [], set()
    for case, record, review in zip(cases, captured, audited):
        target = case['target_id']
        inv._require(case['source_row']['target_id'] == case['conservative_view_cap_group'] == target
                     and record['target_id'] == review['target_id'] == target
                     and target.startswith(plan['source_family'] + '/'), 'Requested original target/cap differs')
        folder = safe_path(capture, case['case_id'])
        if record['state'] == 'held_pre_render':
            inv._require(review['decision'] == 'held_pre_render' and review['reason'] == record['reason']
                         and isinstance(record['reason'], str) and record['reason'] and not folder.exists(),
                         'Unexplained pre-render hold')
            absent.add(folder)
            held.append(deepcopy(review))
            continue
        inv._require(record['state'] == 'native_captured', 'Unexpected original capture state')
        absent.add(folder / 'bundle.json')
        inv._require(not (folder / 'bundle.json').exists(), 'Original raw-only audit cannot authenticate compact storage')
        expected = {'sample.json': record['sample_sha256'], 'supervision/label.json': record['label_sha256']}
        if record['trace_sha256'] is not None:
            expected['supervision/query_trace.json'] = record['trace_sha256']
        else:
            absent.add(folder / 'supervision/query_trace.json')
        reader = SampleReader(folder, expected_bindings=expected)
        logical = {}
        for name in sorted(set(reader.metadata['files']) | expected.keys()):
            raw = reader.read(name)
            if name.endswith('.json'):
                inv._parse(raw)
            logical[name] = dict(sha256=inv.digest(raw), bytes=len(raw))
            bindings.bind(safe_path(folder, name), inv.digest(raw))
        meta, label = reader.metadata, reader.json('supervision/label.json')
        trace = reader.json('supervision/query_trace.json') if record['trace_sha256'] is not None else None
        inv._require(meta['schema_version'] == oc.SAMPLE_SCHEMA and meta['sample_id'] == case['case_id']
                     and meta['pose_prior'] == case['pose_prior'] and meta['historical_labels_inherited'] is False
                     and meta['training_sample_approved'] is False and label['training_approved'] is False,
                     'Fresh original sample/pose-prior provenance required')
        inv._require(type(label['eligible']) is bool and (trace is None or type(trace['passed']) is bool),
                     'Boolean native label/trace decisions required')
        passed = label['eligible'] and trace is not None and trace['passed']
        decision = inv.STRICT if passed else inv.HOLD if label['eligible'] else inv.EXCLUDE
        inv._require(decision == review['decision'] and review['label_replayed_exact'] is True
                     and review['trace_replayed_exact'] is (trace is not None)
                     and review['native_callback_hashes_verified'] is True and review['native_ID_masks_replayed'] is True,
                     'Original independently replayed annotation decision differs')
        rgb = reader.image('inputs/rgb.png')
        inv._require(rgb.dtype == np.uint8 and rgb.shape == (816, 1696, 3), 'Native RGB1696 required')
        fresh = meta['synchronization']['freshness']
        inv._require(type(fresh['callback_sequence']) is int and fresh['callback_sequence'] > 0
                     and fresh['rgb_sha256'] == inv.digest(rgb.tobytes())
                     and fresh['camera_sha256'] == inv.fingerprint(meta['calibration']), 'Stale original callback')
        sequences.append(fresh['callback_sequence'])
        decoded = inv.decoded_rgb_digest(rgb.tobytes(), width=1696, height=816)
        inv._require(decoded not in rgb_seen, 'Repeated native RGB within original batch')
        rgb_seen.add(decoded)
        actual_basis = dict(basis, lighting=deepcopy(meta['lighting']), scene_counts=deepcopy(meta['scene_counts']),
                            renderer=meta['renderer'])
        scene, camera = _features(meta, actual_basis)
        contexts[inv._hash(actual_basis)] = actual_basis
        lineage = dict(deepcopy(ancestry), original_source_row=deepcopy(case['source_row']),
            original_source_row_sha256=inv._hash(case['source_row']), requested_target_id=target,
            conservative_view_cap_group=target, pose_prior=deepcopy(case['pose_prior']))
        row = dict(sample_id=inv._hash(dict(capture=str(capture), candidate_id=case['case_id'])),
            candidate_id=case['case_id'], sample_path=str(folder), capture_path=str(capture),
            source_kind='original_native', target_id=target, variant_target=target,
            source_family=plan['source_family'], original_donor_family=plan['source_family'],
            source_target=target, conservative_view_cap_group=target, split='train', resolution=list(oc.RESOLUTION),
            decision=decision, context_id='original:' + target, geometry_sha256=basis['target_geometry_sha256'],
            image_bytes=logical['inputs/rgb.png']['bytes'], encoded_rgb_sha256=logical['inputs/rgb.png']['sha256'],
            decoded_rgb_sha256=decoded, decoded_rgb_bytes=1696 * 816 * 3, callback_rgb_sha256=fresh['rgb_sha256'],
            annotation_review=dict(passed=passed, method='automatic', evidence_id=pin.launcher_receipt_sha256,
                evidence_basis='pinned_owned_exit0_and_independent_original_native_audit_replay',
                audit_execution_independently_verified=True),
            provenance=dict(receipt_path=str(inv._path(pin.launcher_receipt_path)), receipt_sha256=pin.launcher_receipt_sha256,
                request_sha256=receipt['request_sha256'], result_sha256=pin.result_sha256,
                plan_path=receipt['plan_path'], plan_sha256=receipt['plan_sha256'],
                sample_sha256=record['sample_sha256'], label_sha256=record['label_sha256'], trace_sha256=record['trace_sha256'],
                logical_files=logical, callback=deepcopy(meta['synchronization']), lineage=lineage,
                native_instance_backend=meta['native_instance_backend'], native_instance_sha256=meta['native_instance_sha256'],
                native_mapping_sha256=meta['native_mapping_sha256'], rendered_camera_params=deepcopy(meta['rendered_camera_params']),
                geometry_screen=deepcopy(meta['geometry_screen']), owned_worker=deepcopy(receipt['owned_worker']),
                process_admission=deepcopy(request['process_admission']), audit_review_method=replay['review_method'],
                automatic_audit_sha256=result['automatic_audit_sha256'], postexit_audit_sha256=receipt['audit_sha256'],
                independent_replay_sha256=inv._hash(replay), audit_declarations=deepcopy(review),
                trace_evidence_basis='fresh_original_saved_native_query_trace_replay',
                renderer_settings=dict(status='unknown_not_recorded_by_original_collector', value=None),
                scene_identity_policy=SCENE_POLICY, adapter_policy=POLICY, admission=deepcopy(oc.ADMISSION)),
            training_approved=False, source_cap_reset=False, background_preserved=True,
            independent_geometry_qualification=False, depth_recomputed=False)
        _identities(row, scene, camera)
        rows.append(row)
    inv._require(sequences == sorted(set(sequences)), 'Stale/repeated original callback sequence')
    source = dict(receipt_path=str(inv._path(pin.launcher_receipt_path)), receipt_sha256=pin.launcher_receipt_sha256,
        capture_path=str(capture), storage_root=str(capture), plan_path=receipt['plan_path'], source_kind='original_native',
        planned_rows=len(cases), captured_rows=len(rows), noncaptured_rows=len(held), held_pre_render=held,
        requested_target_ids=[c['target_id'] for c in cases], plan_coverage=deepcopy(plan['coverage']),
        historical_training_rows=0, independent_replay_sha256=inv._hash(replay), scene_contexts=contexts)
    return rows, source


def _state():
    return dict(queues={}, receipts={}, absent=set())


def original_observed_rows(pin):
    """Read ONLY the explicitly pinned serial batch; prior replay is not data."""
    try:
        bindings, state = inv._Bindings(), _state()
        bindings.mapping(implementation_bindings())
        checked = _serial(bindings, pin, state)
        v1._finish(bindings, state['absent'])
        return checked['rows'], checked['summary'], dict(sorted(bindings.hashes.items()))
    except (KeyError, TypeError, IndexError, OSError) as exc:
        raise ValueError('Malformed/missing serial original evidence: ' + str(exc)) from exc


def build_inventory(serial_captures, *, v5_originals=(), generated_receipts=(), extra_observed_roots=()):
    try:
        return _build(list(serial_captures), list(v5_originals), list(generated_receipts), list(extra_observed_roots))
    except (KeyError, TypeError, IndexError, OSError) as exc:
        raise ValueError('Malformed/missing serial observed inventory: ' + str(exc)) from exc


def _build(serial, originals, generated, extras):
    oc.require(all(isinstance(p, SerialCapturePin) for p in serial), 'Explicit SerialCapturePin required')
    oc.require(all(isinstance(e, inv.ObservedRoot) for e in extras), 'Explicit ObservedRoot required')
    # V1-only callers retain the frozen route exactly, including its policy.
    if not serial:
        return v1.build_inventory(originals, generated_receipts=generated, extra_observed_roots=extras)
    bindings, state = inv._Bindings(), _state()
    implementation = implementation_bindings()
    bindings.mapping(implementation)
    rows, coverage, contexts = [], [], {}
    base = None
    if originals or generated or any(e.receipts for e in extras):
        base = v1.build_inventory(originals, generated_receipts=generated, extra_observed_roots=extras)
        bindings.mapping(base['provenance']['source_bindings'])
        bindings.protected_roots.update(base['provenance']['read_only_source_roots'])
        rows.extend(base['records']); coverage.extend(base['scope']['explicit_receipts']); contexts.update(base['scene_contexts'])
    else:
        for e in extras:
            oc.require(isinstance(e, inv.ObservedRoot) and inv._path(e.path).is_dir() and not e.receipts,
                'Explicit unindexed ObservedRoot required')
        oc.require(len({str(inv._path(e.path)) for e in extras}) == len(extras), 'Duplicate extra observed root')
    seen = {str(inv._path(c['capture_path'])) for c in coverage}
    for pin in sorted(serial, key=lambda p: str(p.capture_path)):
        oc.require(isinstance(pin, SerialCapturePin), 'Explicit SerialCapturePin required')
        key = str(inv._path(pin.capture_path))
        oc.require(key not in seen, 'Duplicate capture would double-count rows')
        seen.add(key)
        checked = _serial(bindings, pin, state)
        source = deepcopy(checked['summary'])
        contexts.update(source.pop('scene_contexts'))
        rows.extend(checked['rows']); coverage.append(source)
    rows.sort(key=lambda r: r['sample_id'])
    oc.require(len({r['sample_id'] for r in rows}) == len(rows), 'Duplicate observation identity')
    if base:
        state['absent'].update(inv._path(c['capture_path']) / 'failure.json' for c in base['scope']['explicit_receipts'])
    v1._finish(bindings, state['absent'])
    scope = deepcopy(base['scope']) if base else dict(extra_observed_roots=[dict(path=str(inv._path(e.path)),
        receipt_count=0, status='unindexed_no_receipts', recursively_scanned=False, complete=False) for e in extras],
        receipt_discovery_performed=False, capture_roots_scanned=False, global_complete=False,
        observed_split='train', heldout_inventory_included=False, historical_training_rows=0,
        near_image_comparison_performed=False, morphology_equivalence_finalized=False,
        scene_identity_policy=SCENE_POLICY, renderer_settings_used_to_grant_view_identity=False,
        callback_validation_scope='independent original native ID/RGB/Z/camera replay',
        audit_execution_independently_verified=True, label_derivation_replayed=True,
        trace_derivation_replayed=True, trace_probe_counters_independently_verified=True,
        audit_decision_counts_basis='independent original native replay')
    scope.update(explicit_receipts=coverage, serial_queue_completion_claimed=False,
        prerequisite_observations_implicitly_added=False,
        verified_not_indexed_serial_captures=sorted(set(state['receipts']) - {str(inv._path(p.capture_path)) for p in serial}),
        serial_queues=[dict(queue_request_path=c['queue_request_path'], queue_request_sha256=c['queue_request_sha256'],
            scheduled_cases=39, scheduled_batches=2) for _, c in sorted(state['queues'].items())])
    result = dict(schema=inv.SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        state='observed_inventory_only_not_global_admission', adapter_policy=POLICY, records=rows, scene_contexts=contexts,
        counts=dict(captured_rows=len(rows), receipts=len(coverage), decisions=dict(Counter(r['decision'] for r in rows)),
            source_kinds=dict(Counter(r['source_kind'] for r in rows)),
            original_donor_families=dict(Counter(r['source_family'] for r in rows)),
            conservative_view_cap_groups=dict(Counter(r['source_target'] for r in rows)),
            variant_targets=dict(Counter(r['target_id'] for r in rows))), exact_duplicates=inv._duplicates(rows), scope=scope,
        provenance=dict(source_bindings=dict(sorted(bindings.hashes.items())),
            read_only_source_roots=sorted(bindings.protected_roots), implementation_bindings=implementation),
        training_approved=False, source_cap_reset=False, admission_performed=False, original_reviews_modified=False,
        captures_modified=False, images_generated=False, depth_recomputed=False, compact_qualification_granted=False)
    result['sha256'] = inv._hash(result)
    return result
