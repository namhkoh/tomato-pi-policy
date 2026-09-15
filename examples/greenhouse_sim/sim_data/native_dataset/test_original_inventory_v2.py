"""Synthetic CPU serial receipts, NEVER native pilot/collection proof.

Reuse V1's saved-1696-buffer fixture and real label/trace/ID replay. Only its
source-USD reconstruction, FK, smoke and plan construction are stand-ins. All
serial receipt/pin/handoff checks below exercise real files and real hashes.
"""
import ast
from copy import deepcopy
from dataclasses import asdict, replace
import inspect
from pathlib import Path
import shutil
import subprocess

import pytest

from . import original_inventory_v2 as oi, original_inventory as v1, inventory as inv
from .test_original_inventory import original, reseal, coverage
from .test_inventory import case as generated_case, write, pin, json_at
from ..native_original_capture import audit, contracts as oc

q = oi.producer
FLAGS = dict(training_approved=False, source_cap_reset=False)


@pytest.fixture
def serial(original, monkeypatch):
    f = original
    # A truthful one-case V5 prerequisite, not a serial receipt relabeled V5.
    f['plan']['cases'] = f['plan']['cases'][:1]
    f['plan']['sample_count_limit'] = 1
    f['plan']['coverage']['requested_cases'] = 1
    f['result']['capture']['records'] = f['result']['capture']['records'][:1]
    f['expected_plan'].clear(); f['expected_plan'].update(deepcopy(f['plan']))
    f['automatic'] = audit.audit_records(f['capture'], f['plan'], f['result']['capture']['records'])
    reseal(f)
    root = f['root'] / 'serial_output'
    plans, entries, captures = [], [], []
    registry = [deepcopy(f['plan'])]
    def check_plan(value):
        assert value in registry, 'UNIT frozen source/pose plan changed'
    monkeypatch.setattr(audit, 'check_plan', check_plan)
    for index, (family, count) in enumerate(q.BATCH_SHAPE, 1):
        plan = deepcopy(f['plan'])
        target = family + '/Petiole'
        plan.update(source_family=family, target_id=target, conservative_view_cap_group=target,
            sample_count_limit=count, stage_reuse='one_original_donor_scene_one_native_product',
            scene_policy=deepcopy(oc.SCENE_POLICY))
        plan['coverage'].update(requested_cases=count)
        plan['source_row'].update(target_id=target, variant_id=family, source_plant_id=family, split_group=family)
        plan['pose_prior']['prior_target_id'] = target
        plan['pose_request'] = dict(mode='exact_prior', native_pixel_xy=None)
        if index == 2:
            donor = Path(plan['package']) / 'plants/components' / family
            shutil.copytree(Path(plan['package']) / 'plants/components/seed73_full', donor)
            variant = deepcopy(plan['original_variant'])
            variant.update(variant_id=family, source_plant_id=family, split_group=family)
            plan.update(original_variant=variant, scene_variants=[variant])
            prior_root = f['root'] / 'legacy23'
            shutil.copytree(f['root'] / 'legacy', prior_root)
            prior = json_at(prior_root / 'manifest.json')
            asset_pins = {str(Path(plan['package']) / 'scene.usd'): pin(Path(plan['package']) / 'scene.usd')}
            asset_pins.update({str(p): pin(p) for p in donor.glob('*.usdc')})
            prior.update(variants=[variant], source_usd_sha256=asset_pins)
            write(prior_root / 'manifest.json', prior)
            clear = deepcopy(json_at(f['root'] / 'clear.json'))
            clear['jobs'][0].update(plant_family=family, source_manifest_path=str(donor / 'manifest.json'))
            plan['source_row']['source_manifest_sha256'] = pin(donor / 'manifest.json')
            clear['jobs'][0]['targets'] = [deepcopy(plan['source_row'])]
            clear['source_bindings_sha256'] = dict(asset_pins, **{str(donor/'manifest.json'): pin(donor/'manifest.json')})
            plan['source_collection_plan'] = str(f['root'] / 'clear23.json')
            plan['source_collection_plan_sha256'] = write(Path(plan['source_collection_plan']), clear)
            plan['pose_prior'].update(source_capture=str(prior_root), source_manifest_sha256=pin(prior_root/'manifest.json'))
            plan['source_bindings'].update(clear['source_bindings_sha256'])
            plan['source_bindings'].update({str(p): pin(p) for p in [prior_root/'manifest.json',
                prior_root/'old_848/sample.json', Path(plan['source_collection_plan'])]})
        first = deepcopy(f['plan']['cases'][0])
        for k in first:
            if k != 'case_id' and k in plan:
                first[k] = deepcopy(plan[k])
        plan['cases'] = []
        for n in range(1, count + 1):
            case = deepcopy(first)
            case['case_id'] = f'sample_{n:04d}'
            if n > 1:
                case['expected_calibration']['camera_to_world_usd_row_vectors'][3][0] += n / 100
            plan['cases'].append(case)
        path = f['root'] / f'plan_batch_{index}.json'
        entries.append(dict(plan_path=str(path), plan_sha256=write(path, plan)))
        plans.append(plan); registry.append(deepcopy(plan))
        capture = root / f'batch_{index:03d}' / 'capture'
        capture.mkdir(parents=True)
        records = []
        for n, case in enumerate(plan['cases'], 1):
            if index == n == 1:
                folder = capture / 'sample_0001'
                shutil.copytree(f['folder'], folder)
                meta = json_at(folder / 'sample.json')
                meta['pose_prior'] = deepcopy(case['pose_prior'])
                record = deepcopy(f['result']['capture']['records'][0])
                record['sample_sha256'] = write(folder / 'sample.json', meta)
            else:
                record = dict(case_id=case['case_id'], target_id=target, state='held_pre_render',
                    reason='UNIT geometry hold, no image', admission=deepcopy(oc.ADMISSION))
            records.append(record)
        automatic = audit.audit_records(capture, plan, records)
        request = deepcopy(f['request'])
        request.update(plan_path=str(path), plan_sha256=entries[-1]['plan_sha256'])
        result = deepcopy(f['result'])
        result.update(plan_sha256=entries[-1]['plan_sha256'], request_sha256=write(capture/'request.json', request),
            automatic_audit_sha256=write(capture/'automatic_audit.json', automatic))
        result['capture']['records'] = records
        write(capture / 'result.json', result)
        captures.append(dict(capture=capture, request=request, result=result, automatic=automatic, plan=plan))
    config = dict(schema=q.v5.SCHEMA, output=str(f['capture'].parent), scale_output=str(f['root']/'scale'),
        original_plan=str(f['root']/'plan.json'), original_plan_sha256=pin(f['root']/'plan.json'))
    pred_path = f['root']/'v5_input.json'
    pred_sha = write(pred_path, config)
    cpu = f['root']/'cpu/python.exe'
    supervisor = dict(pid=7654321, creation_date='2026-09-16T01:00:00+00:00', executable=str(cpu),
        command_line=subprocess.list2cmdline([str(cpu), '-B', '-m', 'sim_data.native_capture_v5.campaign_queue',
            '--request', str(pred_path), '--request-sha256', pred_sha]))
    pred_root = f['capture'].parent
    old_code = v1.implementation_bindings()
    write(pred_root/'request.json', dict(config, implementation_bindings=old_code, training_approved=False))
    scale_path = Path(config['scale_output'])/'result.json'
    scale_sha = write(scale_path, dict(records=[{'UNIT_ONLY': True}], new_biological_families=0, **FLAGS))
    write(pred_root/'result.json', dict(state='serial_probe_and_scale_complete_pending_admission',
        request_sha256=pred_sha, scale_result_sha256=scale_sha, **FLAGS))
    pred_event = pred_root/'queue_000099.json'
    write(pred_event, dict(state='scale_coordinator_running_in_same_cpu_process', worker_pid=supervisor['pid'],
        bindings=old_code, **FLAGS))
    pred_pins = {str(p): pin(p) for p in [pred_root/'request.json', pred_root/'result.json', scale_path, pred_event]}
    predecessor = dict(state='exact_v5_terminal_and_supervisor_absent', supervisor=supervisor, bindings=pred_pins,
        process_exit_evidence='exact_PID_absent_from_successful_CIM_query_not_OS_exit_code', **FLAGS)
    request = dict(schema=q.SCHEMA, output=str(root), isaac_python=str(f['root']/'isaac/python.bat'),
        native_deps=str(f['root']/'deps'), max_planned_cases=39, batches=entries, pilot=asdict(f['pin']),
        predecessor=dict(request_path=str(pred_path), request_sha256=pred_sha, supervisor=supervisor))
    external = f['root']/'serial_input.json'
    write(external, request)
    code = q.implementation_bindings()
    write(root/'request.json', dict(request, producer_module=q.PRODUCER_MODULE, producer_implementation_bindings=code, **FLAGS))
    write(root/'pilot_qualification.json', q.qualify_pilot(request['pilot'], config))
    write(root/'predecessor_completion.json', predecessor)
    paths = [external, pred_path, Path(f['pin'].launcher_receipt_path), f['capture']/'result.json',
        root/'request.json', root/'pilot_qualification.json', root/'predecessor_completion.json']
    base = oc.merge_bindings(code, pred_pins, {str(p): pin(p) for p in paths},
        {e['plan_path']: e['plan_sha256'] for e in entries})
    for index, batch in enumerate(captures, 1):
        capture, entry = batch['capture'], entries[index-1]
        folder = capture.parent
        command = q.batch_command(request['isaac_python'], entry, capture)
        event = dict(state=q.LAUNCH_STATE, worker_pid=5500+index, command=command, batch_index=index,
            producer_module=q.PRODUCER_MODULE, bindings=deepcopy(base), **FLAGS,
            resource_evidence=dict(allowed=True, memory=dict(checked=True, allowed=True, commit_headroom_bytes=20*q.GIB),
                disk_free_bytes=60*q.GIB, process_inventory=dict(blockers=[], classifications=[{'UNIT_ONLY': True}],
                    no_blockers_observed=True, bridge_proofs=[]), exclusive_launch_guaranteed=False))
        worker = dict(worker_pid=event['worker_pid'], command=command, event_path=str(folder/'queue_000001.json'),
            event_sha256=write(folder/'queue_000001.json', event))
        receipt = dict(schema=q.RECEIPT_SCHEMA, state=q.RECEIPT_STATE, exit_code=0, batch_index=index,
            producer_module=q.PRODUCER_MODULE, producer_implementation_bindings=code,
            queue_request_path=str(root/'request.json'), queue_request_sha256=pin(root/'request.json'),
            plan_path=entry['plan_path'], plan_sha256=entry['plan_sha256'], capture=str(capture),
            request_sha256=pin(capture/'request.json'), result_sha256=pin(capture/'result.json'), owned_worker=worker,
            audit_path=str(folder/'original_postexit_audit.json'),
            audit_sha256=write(folder/'original_postexit_audit.json', batch['automatic']), source_bindings=deepcopy(base), **FLAGS)
        receipt_path = folder/'original_launcher_receipt.json'
        batch.update(receipt=receipt, pin=oi.SerialCapturePin(str(capture), receipt['result_sha256'],
            str(receipt_path), write(receipt_path, receipt)))
        base = oc.merge_bindings(base, {str(p): pin(p) for p in [receipt_path, folder/'original_postexit_audit.json',
            folder/'queue_000001.json', capture/'request.json', capture/'result.json']})
    # Explicitly completed destination; adapter must never call fresh-output
    # validation, process inventory, native launcher, preparation or queue.run.
    def forbidden(*a, **k):
        pytest.fail('Inventory invoked a launcher/fresh-destination/live-process API')
    for name in ('validate_request', 'run', 'predecessor_completion', 'supervisor_rows', 'resource_evidence'):
        monkeypatch.setattr(q, name, forbidden)
    monkeypatch.setattr(q.campaign, 'run_checked', forbidden)
    monkeypatch.setattr(q.guard, 'process_inventory', forbidden)
    return dict(original=f, root=root, request=request, captures=captures, plans=plans, registry=registry)


def repin(batch, *, event=False):
    receipt = batch['receipt']
    if event:
        path = Path(receipt['owned_worker']['event_path'])
        value = json_at(path)
        value['bindings'] = deepcopy(receipt['source_bindings'])
        receipt['owned_worker']['event_sha256'] = write(path, value)
    batch['pin'] = replace(batch['pin'], result_sha256=receipt['result_sha256'],
        launcher_receipt_sha256=write(Path(batch['pin'].launcher_receipt_path), receipt))
    return batch['pin']


def build(f, indices=(0, 1), **kwargs):
    return oi.build_inventory([f['captures'][i]['pin'] for i in indices], **kwargs)


def test_serial_completed_batches_real_native_replay_and_observed_only(serial):
    before = {str(p): pin(p) for p in serial['original']['root'].rglob('*') if p.is_file()}
    out = build(serial)
    assert out['counts']['captured_rows'] == 1 and out['counts']['decisions'] == {inv.STRICT: 1}
    assert [(s['planned_rows'], s['captured_rows'], s['noncaptured_rows'])
        for s in out['scope']['explicit_receipts']] == [(16, 1, 15), (23, 0, 23)]
    row, = out['records']
    assert row['source_kind'] == 'original_native' and row['context_id'] == 'original:seed73_full/Petiole'
    assert row['provenance']['lineage']['frozen_family_assignments'] == oc.FROZEN_SPLITS
    assert row['provenance']['lineage']['pose_prior']['role'] == 'historical_pose_only_never_training_observation'
    contract = row['provenance']['serial_launcher']
    assert contract['state'] == q.RECEIPT_STATE and contract['producer_module'] == q.PRODUCER_MODULE
    assert contract['producer_implementation_bindings'] == q.implementation_bindings()
    assert row['annotation_review']['audit_execution_independently_verified']
    assert row['provenance']['renderer_settings']['value'] is None
    assert row['source_target'] == row['conservative_view_cap_group'] == 'seed73_full/Petiole'
    for field in ('training_approved', 'source_cap_reset', 'admission_performed', 'images_generated', 'compact_qualification_granted'):
        assert out[field] is False
    assert not out['scope']['global_complete'] and not out['scope']['serial_queue_completion_claimed']
    assert not out['scope']['prerequisite_observations_implicitly_added']
    assert out['scope']['verified_not_indexed_serial_captures'] == []
    assert out['sha256'] == inv._hash({k: v for k, v in out.items() if k != 'sha256'})
    assert before == {str(p): pin(p) for p in serial['original']['root'].rglob('*') if p.is_file()}


def test_batch_two_checks_prior_without_adding_rows(serial):
    out = build(serial, (1,))
    assert out['counts']['captured_rows'] == 0 and out['counts']['receipts'] == 1
    assert out['scope']['explicit_receipts'][0]['planned_rows'] == 23
    assert out['scope']['verified_not_indexed_serial_captures'] == [str(serial['captures'][0]['capture'])]
    rows, summary, pins = oi.original_observed_rows(serial['captures'][1]['pin'])
    assert rows == [] and summary['verified_preceding_batches'] == [1]
    assert str(serial['captures'][0]['capture']/'sample_0001/inputs/rgb.png') in pins


def test_projection_ast_identical_except_checked_tuple_input():
    old = ast.parse(inspect.getsource(v1._original)).body[0]
    new = ast.parse(inspect.getsource(oi._project_original)).body[0]
    assert [ast.dump(n) for n in old.body[1:]] == [ast.dump(n) for n in new.body[1:]]
    assert pin(Path(q.__file__)) == oi.PRODUCER_SHA256
    assert oi.implementation_bindings()[str(Path(oi.__file__).resolve())] == pin(Path(oi.__file__))


def test_v1_only_route_unchanged(original):
    old = v1.build_inventory([original['pin']])
    new = oi.build_inventory([], v5_originals=[original['pin']])
    for packet in (old, new):
        packet.pop('created_utc'); packet.pop('sha256')
    assert new == old


def test_v5_receipt_cannot_be_relabelled_serial(original):
    with pytest.raises(ValueError):
        oi.build_inventory([oi.SerialCapturePin(**asdict(original['pin']))])


def test_mixed_v5_generated_and_serial_keep_claims_and_physical_dedup(serial, generated_case):
    old = v1.build_inventory([serial['original']['pin']], generated_receipts=[generated_case['pin']])
    out = build(serial, v5_originals=[serial['original']['pin']], generated_receipts=[generated_case['pin']])
    by_id = {r['sample_id']: r for r in out['records']}
    assert all(by_id[r['sample_id']] == r for r in old['records'])
    assert out['counts']['captured_rows'] == 5
    assert not out['scope']['audit_execution_independently_verified']  # generated declarations stay declarations
    assert out['exact_duplicates']['decoded_rgb_sha256']['duplicate_excess_rows'] == 3
    assert out['exact_duplicates']['scene_camera_sha256']['duplicate_excess_rows'] == 3
    assert not any(r['source_cap_reset'] for r in out['records'])
    from . import provisional
    splits = dict(oc.FROZEN_SPLITS, donor='train')
    with pytest.raises(ValueError, match='Different pair coverage'):
        provisional.select_observed_train(out, splits, coverage(old))
    assert not provisional.select_observed_train(out, splits, coverage(out))['training_approved']


@pytest.mark.parametrize('field,value', [
    ('state', v1.LAUNCH_STATE), ('schema', 'generic_exit_receipt'), ('exit_code', 1), ('exit_code', False),
    ('batch_index', 2), ('batch_index', True), ('training_approved', True), ('source_cap_reset', True),
    ('producer_module', 'sim_data.native_capture_v5.campaign_queue'),
    ('producer_implementation_bindings', {}), ('request_sha256', '0'*64), ('result_sha256', '0'*64),
    ('queue_request_sha256', '0'*64), ('plan_sha256', '0'*64), ('audit_sha256', '0'*64)])
def test_exact_owned_serial_receipt_contract(serial, field, value):
    batch = serial['captures'][0]
    batch['receipt'][field] = value
    repin(batch)
    with pytest.raises(ValueError):
        build(serial, (0,))


@pytest.mark.parametrize('kind', ['producer', 'queue', 'pilot', 'event_pid', 'event_command',
    'resource', 'postexit', 'failure', 'compact', 'native_ID', 'held_directory'])
def test_producer_event_resource_and_native_evidence_fail_closed(serial, kind):
    batch = serial['captures'][0]
    receipt = batch['receipt']
    capture = batch['capture']
    if kind in ('producer', 'queue', 'pilot'):
        key = {'producer': oi._PRODUCER_PATH, 'queue': str(serial['root']/'request.json'),
            'pilot': serial['original']['pin'].launcher_receipt_path}[kind]
        del receipt['source_bindings'][key]
        repin(batch, event=True)
    elif kind.startswith('event') or kind == 'resource':
        path = Path(receipt['owned_worker']['event_path'])
        event = json_at(path)
        if kind == 'event_pid': event['worker_pid'] += 1
        elif kind == 'event_command': event['command'][2] = 'sim_data.native_generated_views'
        else: event['resource_evidence']['process_inventory']['no_blockers_observed'] = False
        receipt['owned_worker']['event_sha256'] = write(path, event)
        repin(batch)
    elif kind == 'postexit':
        altered = dict(batch['automatic'], review_method='invented')
        receipt['audit_sha256'] = write(Path(receipt['audit_path']), altered)
        repin(batch)
    elif kind == 'failure': write(capture/'failure.json', {})
    elif kind == 'compact': write(capture/'sample_0001/bundle.json', {})
    elif kind == 'native_ID': (capture/'sample_0001/supervision/renderer_instance_id.npy').write_bytes(b'UNIT TAMPER')
    else: (capture/'sample_0002').mkdir()
    with pytest.raises(ValueError):
        build(serial, (0,))


def resave_queue(f):
    batch, request = f['captures'][0], f['request']
    for i, (entry, plan) in enumerate(zip(request['batches'], f['plans'])):
        entry['plan_sha256'] = write(Path(entry['plan_path']), plan)
        batch['receipt']['source_bindings'][entry['plan_path']] = entry['plan_sha256']
    path = f['root']/'request.json'
    sha = write(path, dict(request, producer_module=q.PRODUCER_MODULE,
        producer_implementation_bindings=q.implementation_bindings(), **FLAGS))
    batch['receipt']['queue_request_sha256'] = sha
    batch['receipt']['source_bindings'][str(path)] = sha
    repin(batch, event=True)


@pytest.mark.parametrize('kind', ['split', 'count', 'camera', 'cap', 'prior', 'reframe', 'geometry', 'policy'])
def test_resealed_queue_cannot_change_frozen_case_scope(serial, kind):
    plan = serial['plans'][0]
    if kind == 'split': plan['family_assignments']['seed73_full'] = 'test'
    elif kind == 'count': plan['sample_count_limit'] = 15
    elif kind == 'camera': plan['cases'][1]['expected_calibration'] = deepcopy(plan['cases'][0]['expected_calibration'])
    elif kind == 'cap': plan['cases'][0]['conservative_view_cap_group'] = 'new_cap'
    elif kind == 'prior': plan['cases'][0]['pose_prior']['prior_target_id'] = 'seed73_full/Main'
    elif kind == 'reframe': plan['cases'][0]['pose_request']['mode'] = 'reframe_head_only'
    elif kind == 'geometry': plan['geometry_mode'] = 'generated'
    else: plan['scene_policy']['render_subframes'] = 8
    resave_queue(serial)
    with pytest.raises(ValueError, match='Frozen original serial|Serial original target|Duplicate scheduled'):
        build(serial, (0,))


@pytest.mark.parametrize('kind', ['missing_handoff', 'changed_receipt', 'extra_handoff', 'bad_prior_audit'])
def test_batch_two_requires_exact_prior_batch_handoff(serial, kind):
    first, second = serial['captures']
    if kind == 'missing_handoff':
        del second['receipt']['source_bindings'][str(first['capture']/'request.json')]
    elif kind == 'changed_receipt':
        first['receipt']['exit_code'] = 1
        repin(first)
    elif kind == 'extra_handoff':
        extra = serial['root']/'unexpected.json'
        second['receipt']['source_bindings'][str(extra)] = write(extra, {'UNIT_ONLY': True})
    else:
        path = Path(first['receipt']['audit_path'])
        first['receipt']['audit_sha256'] = write(path, dict(first['automatic'], counts={}))
        repin(first)
        second['receipt']['source_bindings'].update({str(path): pin(path),
            first['pin'].launcher_receipt_path: first['pin'].launcher_receipt_sha256})
    repin(second, event=True)
    with pytest.raises(ValueError):
        build(serial, (1,))


@pytest.mark.parametrize('name', ['pilot_qualification.json', 'predecessor_completion.json'])
def test_saved_prerequisites_replayed_not_merely_hash_accepted(serial, name):
    path = serial['root']/name
    value = json_at(path)
    value['state'] = 'invented_complete'
    batch = serial['captures'][0]
    batch['receipt']['source_bindings'][str(path)] = write(path, value)
    repin(batch, event=True)
    with pytest.raises(ValueError, match='Saved'):
        build(serial, (0,))


@pytest.mark.parametrize('name', ['label', 'query_trace'])
def test_rehashed_native_annotation_still_requires_exact_independent_replay(serial, name):
    batch = serial['captures'][0]
    path = batch['capture']/'sample_0001/supervision'/(name+'.json')
    value = json_at(path); value['tampered'] = True
    field = 'label_sha256' if name == 'label' else 'trace_sha256'
    sha = write(path, value)
    batch['result']['capture']['records'][0][field] = sha
    batch['automatic']['records'][0][field] = sha
    batch['result']['automatic_audit_sha256'] = write(batch['capture']/'automatic_audit.json', batch['automatic'])
    batch['receipt']['audit_sha256'] = write(Path(batch['receipt']['audit_path']), batch['automatic'])
    batch['receipt']['result_sha256'] = write(batch['capture']/'result.json', batch['result'])
    repin(batch)
    with pytest.raises(ValueError, match='Stale stored'):
        build(serial, (0,))


def test_explicit_duplicate_and_conflicting_cached_capture_rejected(serial):
    p = serial['captures'][0]['pin']
    with pytest.raises(ValueError, match='Duplicate capture'):
        oi.build_inventory([p, p])
    bindings, state = inv._Bindings(), oi._state()
    bindings.mapping(oi.implementation_bindings())
    oi._serial(bindings, p, state)
    with pytest.raises(ValueError, match='Conflicting serial capture pins'):
        oi._serial(bindings, replace(p, result_sha256='0'*64), state)


@pytest.mark.parametrize('kind', ['source', 'failure', 'bundle'])
def test_final_full_sha_and_absent_markers_after_native_replay(serial, monkeypatch, kind):
    capture = serial['captures'][0]['capture']
    asset = serial['original']['root']/'package/plants/components/seed73_full/Petiole.usdc'
    real = audit.audit_capture
    def after(path, **kwargs):
        result = real(path, **kwargs)
        if Path(path) == capture:
            if kind == 'source': asset.write_bytes(b'UNIT changed after actual audit')
            elif kind == 'failure': write(capture/'failure.json', {})
            else: write(capture/'sample_0001/bundle.json', {})
        return result
    monkeypatch.setattr(audit, 'audit_capture', after)
    with pytest.raises(ValueError):
        build(serial, (0,))


def test_shared_queue_pilot_and_batch_replay_once_each(serial, monkeypatch):
    real, calls = audit.audit_capture, []
    def counted(path, **kwargs):
        calls.append(str(Path(path)))
        return real(path, **kwargs)
    monkeypatch.setattr(audit, 'audit_capture', counted)
    build(serial)
    assert calls == [str(serial['original']['capture']), *[str(b['capture']) for b in serial['captures']]]


def test_import_is_cpu_only_and_no_fake_v5_translation():
    import sys
    assert 'omni.kit' not in sys.modules and 'isaacsim' not in sys.modules
    assert 'owned_original_worker_exited_and_reaudited' not in inspect.getsource(oi)


@pytest.mark.parametrize('values', [[{}], [None]])
def test_explicit_pin_objects_required(values):
    with pytest.raises(ValueError, match='Explicit SerialCapturePin'):
        oi.build_inventory(values)


@pytest.mark.parametrize('kind', ['missing', 'different_input', 'unbound_extra'])
def test_exact_external_input_request_is_part_of_producer_contract(serial, kind):
    batch = serial['captures'][0]
    path = serial['original']['root']/'serial_input.json'
    if kind == 'missing':
        del batch['receipt']['source_bindings'][str(path)]
    elif kind == 'different_input':
        value = json_at(path); value['max_planned_cases'] = 38
        batch['receipt']['source_bindings'][str(path)] = write(path, value)
    else:
        extra = serial['root']/'unexpected.json'
        batch['receipt']['source_bindings'][str(extra)] = write(extra, {})
    repin(batch, event=True)
    with pytest.raises(ValueError, match='initial serial input|input/publication'):
        build(serial, (0,))


def test_serial_receipt_is_not_accepted_by_frozen_v1(serial):
    p = serial['captures'][0]['pin']
    with pytest.raises(ValueError):
        v1.original_observed_rows(v1.OriginalCapturePin(**asdict(p)))
