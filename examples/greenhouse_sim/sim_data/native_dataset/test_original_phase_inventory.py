"""Synthetic CPU receipts; real saved 1696-buffer replay, never native proof."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess

import pytest

from . import original_phase_inventory as oi, inventory as inv
from .test_original_inventory import original
from .test_original_inventory_v2 import serial
from .test_inventory import write, pin, json_at

q, oc = oi.q, oi.oc


@pytest.fixture
def phase(serial, monkeypatch):
    s = serial
    external = s['original']['root'] / 'serial_input.json'
    code = q.legacy.implementation_bindings()
    records = []
    for index, ((family, count), batch) in enumerate(zip(q.legacy.BATCH_SHAPE, s['captures'], strict=True), 1):
        receipt = batch['receipt']
        records.append(dict(batch_index=index, family=family, planned_cases=count,
            capture=str(batch['capture']), result_sha256=receipt['result_sha256'],
            audit_counts=batch['automatic']['counts'], launcher_receipt_path=batch['pin'].launcher_receipt_path,
            launcher_receipt_sha256=batch['pin'].launcher_receipt_sha256))
    write(s['root'] / 'result.json', dict(schema=q.legacy.SCHEMA, state=q.legacy.RESULT_STATE,
        producer_module=q.legacy.PRODUCER_MODULE, producer_implementation_bindings=code,
        queue_request_path=str(s['root'] / 'request.json'), queue_request_sha256=pin(s['root'] / 'request.json'),
        planned_cases=39, records=records, training_approved=False, source_cap_reset=False))
    _, _, previous = oi.v2.original_observed_rows(s['captures'][1]['pin'])
    previous = oc.merge_bindings(previous, {str(s['root'] / n): pin(s['root'] / n) for n in ('request.json', 'result.json')})
    cpu = s['original']['root'] / 'cpu/python.exe'
    cpu.parent.mkdir(exist_ok=True)
    cpu.write_bytes(b'CPU test fixture, never executable')
    supervisor = dict(pid=8765432, creation_date='2026-09-16T01:00:00+00:00', executable=str(cpu),
        command_line=subprocess.list2cmdline([str(cpu), '-B', '-m', q.legacy.PRODUCER_MODULE,
            '--request', str(external), '--request-sha256', pin(external)]))
    root = s['original']['root'] / 'phase_output'
    jobs = [dict(id=f'J{i}', worker_kind='original_batch.v1', **e) for i, e in enumerate(s['request']['batches'])]
    request = dict(schema=q.SCHEMA, output=str(root), isaac_python=s['request']['isaac_python'],
        native_deps=s['request']['native_deps'], max_jobs=64, max_planned_frames=4096, planned_frames=39,
        predecessor=dict(kind='original_serial39.v1', request_path=str(external), request_sha256=pin(external),
            supervisor=supervisor), phases=[dict(id='originals', jobs=jobs)])
    input_path = s['original']['root'] / 'phase_input.json'
    input_pin = write(input_path, request)
    code = q.implementation_bindings()
    write(root / 'request.json', dict(request, input_request_path=str(input_path), input_request_sha256=input_pin,
        producer_module=q.PRODUCER_MODULE, producer_implementation_bindings=code, **q.FLAGS))
    prior = dict(state='serial39_completed_and_exact_supervisor_absent', supervisor=supervisor,
        process_exit_evidence='exact_PID_absent_from_successful_CIM_query_not_OS_exit_code', bindings=previous, **q.FLAGS)
    write(root / 'predecessor_completion.json', prior)
    write(root / 'checkpoint_000001.json', dict(schema=q.CHECKPOINT_SCHEMA, state='clean_initial_boundary',
        request_sha256=input_pin, completed=[], next_job_index=0, created_utc='UNIT', **q.FLAGS))
    monkeypatch.setattr(q.original_plan, 'check_plan', lambda p: p in s['registry'] or pytest.fail('Changed source plan'))
    checked = [q.validate_plan(j['worker_kind'], j['plan_path'], j['plan_sha256']) for j in jobs]
    sources = oc.merge_bindings(code, {str(input_path): input_pin, str(external): pin(external)},
        *(c['bindings'] for c in checked), previous,
        {str(root / n): pin(root / n) for n in ('request.json', 'predecessor_completion.json', 'checkpoint_000001.json')})
    completed, batches = [], []
    for i, (job, old) in enumerate(zip(jobs, s['captures'], strict=True)):
        folder = root / 'originals' / job['id']
        capture, submitted = folder / 'capture', folder / 'submitted/plan.json'
        submitted.parent.mkdir(parents=True)
        submitted.write_bytes(Path(job['plan_path']).read_bytes())
        sources[str(submitted)] = job['plan_sha256']
        shutil.copytree(old['capture'], capture)
        native_request = deepcopy(old['request'])
        native_request['plan_path'] = str(submitted)
        result = deepcopy(old['result'])
        result['request_sha256'] = write(capture / 'request.json', native_request)
        automatic = oi.audit.audit_records(capture, old['plan'], result['capture']['records'])
        result['automatic_audit_sha256'] = write(capture / 'automatic_audit.json', automatic)
        result_pin = write(capture / 'result.json', result)
        audit_path = folder / 'original_postexit_audit.json'
        audit_pin = write(audit_path, automatic)
        reviewed = dict(request_sha256=result['request_sha256'], result_sha256=result_pin,
            audit_path=str(audit_path), audit_sha256=audit_pin, counts=automatic['counts'],
            bindings=oc.merge_bindings(q._tree_pins(capture), {str(audit_path): audit_pin}))
        owner_meta = dict(ProcessId=8765433, ParentProcessId=12345, CreationDate='2026-09-16T01:00:00+00:00',
            Name='python.exe', ExecutablePath=str(cpu), CommandLine=subprocess.list2cmdline([str(cpu), '-B', '-m',
                q.PRODUCER_MODULE, '--request', str(input_path), '--request-sha256', input_pin]))
        owner_class = q.legacy.guard.classify_process(owner_meta, native_roots=q.legacy.guard.DEFAULT_NATIVE_ROOTS)
        owner = dict(metadata=owner_meta, classification=owner_class, current_pid_exception_used=False,
            running_code_attested=False)
        resource = dict(allowed=True, memory=dict(checked=True, allowed=True, commit_headroom_bytes=20*q.legacy.GIB),
            disk_free_bytes=60*q.legacy.GIB, process_inventory=dict(blockers=[], classifications=[owner_class],
                no_blockers_observed=True, bridge_proofs=[]), exclusive_launch_guaranteed=False)
        command = q.worker_command(request['isaac_python'], 'original_batch.v1', submitted, job['plan_sha256'], capture)
        event = dict(schema=q.RECEIPT_SCHEMA, state='owned_child_running', worker_pid=6000+i, command=command,
            job_index=i, phase_id='originals', job_id=job['id'], bindings=deepcopy(sources),
            resource_evidence=resource, owner_visibility=owner, **q.FLAGS)
        event_path = folder / 'launch.json'
        worker = dict(worker_pid=6000+i, command=command, event_path=str(event_path),
            event_sha256=write(event_path, event), resource_evidence=resource, owner_visibility=owner)
        receipt = dict(schema=q.RECEIPT_SCHEMA, state='owned_exit0_postexit_audited_pending_admission',
            producer_module=q.PRODUCER_MODULE, producer_implementation_bindings=code,
            input_request_path=str(input_path), input_request_sha256=input_pin,
            job_index=i, phase_id='originals', job_id=job['id'], worker_kind='original_batch.v1',
            source_plan_path=job['plan_path'], plan_sha256=job['plan_sha256'], submitted_plan_path=str(submitted),
            capture=str(capture), planned_source_frames=checked[i]['count'], exit_code=0,
            owned_worker=worker, source_bindings=deepcopy(sources), audit=reviewed, **q.FLAGS)
        receipt_path = folder / 'receipt.json'
        capture_pin = oi.PhaseCapturePin(str(capture), result_pin, str(receipt_path), write(receipt_path, receipt))
        batches.append(dict(pin=capture_pin, receipt=receipt, event=event, result=result, capture=capture))
        completed.append(dict(phase_id='originals', job_id=job['id'], worker_kind='original_batch.v1',
            receipt_path=str(receipt_path), receipt_sha256=capture_pin.launcher_receipt_sha256))
        sources = oc.merge_bindings(sources, reviewed['bindings'], {str(receipt_path): capture_pin.launcher_receipt_sha256,
            str(event_path): worker['event_sha256']})
        cp = root / f'checkpoint_{i+2:06d}.json'
        sources[str(cp)] = write(cp, dict(schema=q.CHECKPOINT_SCHEMA, state='owned_exit0_audited_clean_boundary',
            request_sha256=input_pin, completed=deepcopy(completed), next_job_index=i+1, created_utc='UNIT', **q.FLAGS))
    def forbidden(*a, **kw):
        pytest.fail('Read-only adapter invoked native/live process/create-output API')
    for name in ('run', 'validate_request', 'predecessor_completion', 'owner_visibility'):
        monkeypatch.setattr(q, name, forbidden)
    return dict(root=root, batches=batches, serial=s)


def repin(batch, event=False):
    receipt = batch['receipt']
    if event:
        receipt['owned_worker']['event_sha256'] = write(Path(receipt['owned_worker']['event_path']), batch['event'])
    batch['pin'] = replace(batch['pin'], launcher_receipt_sha256=write(Path(batch['pin'].launcher_receipt_path), receipt))


def test_phase_observations_replay_and_keep_original_caps(phase):
    root = phase['serial']['original']['root']
    before = {str(p): pin(p) for p in root.rglob('*') if p.is_file()}
    out = oi.build_inventory([b['pin'] for b in phase['batches']])
    assert out['counts']['captured_rows'] == 1 and out['counts']['decisions'] == {inv.STRICT: 1}
    assert [(s['captured_rows'], s['noncaptured_rows']) for s in out['scope']['explicit_receipts']] == [(1, 15), (0, 23)]
    row, = out['records']
    assert row['resolution'] == [1696, 816] and row['source_target'] == row['conservative_view_cap_group']
    assert row['provenance']['adapter_policy'] == oi.POLICY
    assert row['provenance']['phase_launcher']['producer_module'] == q.PRODUCER_MODULE
    assert row['provenance']['lineage']['frozen_family_assignments'] == oc.FROZEN_SPLITS
    assert not out['training_approved'] and not row['source_cap_reset'] and not row['independent_geometry_qualification']
    assert not out['scope']['global_complete'] and not out['scope']['phase_completion_claimed']
    assert out['sha256'] == inv._hash({k: v for k, v in out.items() if k != 'sha256'})
    assert before == {str(p): pin(p) for p in root.rglob('*') if p.is_file()}


def test_selected_later_receipt_does_not_add_prerequisite_rows(phase):
    out = oi.build_inventory([phase['batches'][1]['pin']])
    assert out['counts']['captured_rows'] == 0 and out['counts']['receipts'] == 1
    assert not out['scope']['prerequisite_observations_implicitly_added']
    assert not out['scope']['qualification_controls_implicitly_added']


@pytest.mark.parametrize('field,value', [('producer_module', 'invented'), ('exit_code', False),
    ('worker_kind', 'matched_pair.v2'), ('training_approved', True), ('job_index', 1), ('plan_sha256', '0'*64)])
def test_coherently_pinned_wrong_receipt_rejected(phase, field, value):
    batch = phase['batches'][0]
    batch['receipt'][field] = value
    repin(batch)
    with pytest.raises(ValueError):
        oi.build_inventory([batch['pin']])


def test_resealed_wrong_native_command_rejected(phase):
    batch = phase['batches'][0]
    batch['receipt']['owned_worker']['command'] = ['invented']
    batch['event']['command'] = ['invented']
    repin(batch, event=True)
    with pytest.raises(ValueError, match='worker differs'):
        oi.build_inventory([batch['pin']])


def test_resealed_native_label_change_fails_replay(phase):
    batch = phase['batches'][0]
    label_path = batch['capture'] / 'sample_0001/supervision/label.json'
    label = json_at(label_path)
    label['eligible'] = False
    label_pin = write(label_path, label)
    batch['result']['capture']['records'][0]['label_sha256'] = label_pin
    result_pin = write(batch['capture'] / 'result.json', batch['result'])
    reviewed = batch['receipt']['audit']
    reviewed['result_sha256'] = result_pin
    reviewed['bindings'].update({str(label_path): label_pin, str(batch['capture'] / 'result.json'): result_pin})
    batch['pin'] = replace(batch['pin'], result_sha256=result_pin)
    repin(batch)
    with pytest.raises(ValueError):
        oi.build_inventory([batch['pin']])


def test_duplicate_receipt_cannot_double_count(phase):
    pin_value = phase['batches'][0]['pin']
    with pytest.raises(ValueError, match='Duplicate phase capture'):
        oi.build_inventory([pin_value, pin_value])


def test_missing_ordered_checkpoint_rejected(phase):
    (phase['root'] / 'checkpoint_000002.json').unlink()
    with pytest.raises(ValueError):
        oi.build_inventory([phase['batches'][1]['pin']])
