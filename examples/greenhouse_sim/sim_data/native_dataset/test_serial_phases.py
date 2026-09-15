'''CPU-only runner boundaries: native execution/source checks/audits mocked.'''
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import threading
from types import SimpleNamespace
import pytest
from . import serial_phases as q


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    q.oc.write_new(path, value)
    return q.oc.sha256(path)


def replace(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')
    return q.oc.sha256(path)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    runtime, cpu, deps = tmp_path/'runtime'/'python.bat', tmp_path/'cpu'/'python.exe', tmp_path/'deps'
    for path in (runtime, cpu):
        path.parent.mkdir()
        path.write_text('CPU fixture; never execute', encoding='utf-8')
    deps.mkdir()
    asset = tmp_path/'sources'/'asset.json'
    source = {str(asset): put(asset, dict(source=True))}
    base = dict(split='train', source_family='seed17_full', source_bindings=source,
        implementation_bindings=q.oc.LOADED_IMPLEMENTATION, package=str(asset.parent),
        source_collection_plan=str(tmp_path/'source_plans'/'plan.json'),
        pose_prior=dict(source_capture=str(tmp_path/'source_capture'), prior_plan=str(tmp_path/'priors'/'plan.json')))
    original = dict(base, sample_count_limit=3, cases=[{}, {}, {}])
    pair = dict(base, sample_count_limit=2, modes=['original_control', 'generated_variant'],
        source_capture=str(tmp_path/'pair_source'), variant_directory=str(tmp_path/'variant'),
        prerequisite_directory=str(tmp_path/'sensor'))
    put(Path(pair['source_capture'])/'manifest.json', dict(package=base['package']))
    put(Path(pair['prerequisite_directory'])/'request.json', dict(source_bindings=source))
    prereq_path = Path(pair['prerequisite_directory'])/'result.json'
    prereq = {str(prereq_path): put(prereq_path, dict(cpu_only=True))}
    paths, plans = [], [original, pair]
    for i, plan in enumerate(plans):
        path = tmp_path/f'plans_{i}'/'plan.json'
        paths.append(dict(id=f'J{i}', worker_kind=q.WORKERS[i], plan_path=str(path), plan_sha256=put(path, plan)))
    older_path = tmp_path/'older.json'
    older_pin = put(older_path, dict(output=str(tmp_path/'old_v5'), scale_output=str(tmp_path/'old_v4')))
    config = dict(schema=q.legacy.SCHEMA, output=str(tmp_path/'serial39'), max_planned_cases=39,
        batches=[dict(plan_path=j['plan_path'], plan_sha256=j['plan_sha256']) for j in paths],
        predecessor=dict(request_path=str(older_path), request_sha256=older_pin))
    pred_path = tmp_path/'predecessor.json'
    pred_pin = put(pred_path, config)
    supervisor = dict(pid=7654321, creation_date='2026-09-16T01:00:00+00:00', executable=str(cpu),
        command_line=subprocess.list2cmdline([str(cpu), '-B', '-m', q.legacy.PRODUCER_MODULE,
            '--request', str(pred_path), '--request-sha256', pred_pin]))
    request = dict(schema=q.SCHEMA, output=str(tmp_path/'output'), isaac_python=str(runtime), native_deps=str(deps),
        max_jobs=64, max_planned_frames=4096, planned_frames=5,
        predecessor=dict(kind='original_serial39.v1', request_path=str(pred_path), request_sha256=pred_pin, supervisor=supervisor),
        phases=[dict(id='originals', jobs=[paths[0]]), dict(id='controls', jobs=[paths[1]])])
    checks = []
    monkeypatch.setattr(q.original_plan, 'check_plan', lambda p: checks.append(('original', deepcopy(p))) or True)
    monkeypatch.setattr(q.pair_plan, 'check_plan', lambda p: checks.append(('pair', deepcopy(p))) or True)
    monkeypatch.setattr(q.pair_plan, 'verify_sensor_prerequisite', lambda p: dict(bindings=prereq))
    return SimpleNamespace(request=request, config=config, paths=paths, plans=plans, source=source,
        prereq=prereq, checks=checks, tmp=tmp_path, output=Path(request['output']))


def test_validators_and_fixed_commands_preserve_input(fixture):
    f, before = fixture, deepcopy(fixture.request)
    _, jobs = q.validate_request(f.request)
    assert f.request == before and [j['checked']['count'] for j in jobs] == [3, 2]
    assert [kind for kind, _ in f.checks] == ['original', 'pair']
    for job in jobs:
        folder = f.output/job['phase_id']/job['id']
        submitted, capture = folder/'submitted'/'plan.json', folder/'capture'
        command = q.worker_command(f.request['isaac_python'], job['worker_kind'], submitted, job['plan_sha256'], capture)
        assert command[0] == f.request['isaac_python'] and command[-2:] == ['--output', str(capture)]
        assert ('--plan-sha256' in command) is (job['worker_kind'] == q.WORKERS[0])
        q.oc.new_destination(capture, [submitted.parent])
        with pytest.raises(ValueError):
            q.oc.new_destination(capture, [folder])


@pytest.mark.parametrize('kind', ['unknown', 'compact', 'argv', 'duplicate_id', 'reserved_id', 'traversal',
    'duplicate_phase', 'duplicate_plan', 'total', 'total_bool', 'bounds', 'max_jobs_bool', 'jobs65', 'empty',
    'heldout', 'cases65', 'pair3', 'plan_tamper', 'source_overlap', 'pred_overlap', 'existing_output',
    'predecessor_kind', 'supervisor_command', 'supervisor_self'])
def test_request_rejections(fixture, kind):
    f, r = fixture, deepcopy(fixture.request)
    first = r['phases'][0]['jobs'][0]
    if kind in ('unknown', 'compact'): first['worker_kind'] = 'compact_query_v2' if kind == 'compact' else 'arbitrary.module'
    elif kind == 'argv': first['argv'] = ['--bypass']
    elif kind == 'duplicate_id': r['phases'][1]['jobs'][0]['id'] = 'j0'
    elif kind == 'reserved_id': first['id'] = 'CON'
    elif kind == 'traversal': first['id'] = '../escape'
    elif kind == 'duplicate_phase': r['phases'][1]['id'] = 'ORIGINALS'
    elif kind == 'duplicate_plan': r['phases'][1]['jobs'][0].update(plan_path=first['plan_path'], plan_sha256=first['plan_sha256'])
    elif kind == 'total': r['planned_frames'] = 4
    elif kind == 'total_bool': r['planned_frames'] = True
    elif kind == 'bounds': r['max_planned_frames'] = 4097
    elif kind == 'max_jobs_bool': r['max_jobs'] = True
    elif kind == 'jobs65': r['phases'] *= 65
    elif kind == 'empty': r['phases'][0]['jobs'] = []
    elif kind in ('heldout', 'cases65', 'pair3'):
        job = r['phases'][1 if kind == 'pair3' else 0]['jobs'][0]
        path, plan = Path(job['plan_path']), q.oc.read_json(job['plan_path'])
        if kind == 'heldout': plan['source_family'] = 'seed999_full'
        else: plan['sample_count_limit'] = 3 if kind == 'pair3' else 65
        job['plan_sha256'] = replace(path, plan)
        f.config['batches'][1 if kind == 'pair3' else 0]['plan_sha256'] = job['plan_sha256']
        pred = r['predecessor']
        pred['request_sha256'] = replace(Path(pred['request_path']), f.config)
        pred['supervisor']['command_line'] = pred['supervisor']['command_line'].replace(f.request['predecessor']['request_sha256'], pred['request_sha256'])
    elif kind == 'plan_tamper': replace(Path(first['plan_path']), {})
    elif kind == 'source_overlap': r['output'] = str(Path(next(iter(f.source))).parent/'new')
    elif kind == 'pred_overlap': r['output'] = str(Path(f.config['output'])/'new')
    elif kind == 'existing_output': f.output.mkdir()
    elif kind == 'predecessor_kind': r['predecessor']['kind'] = 'anything'
    elif kind == 'supervisor_command': r['predecessor']['supervisor']['command_line'] += ' --extra'
    elif kind == 'supervisor_self': r['predecessor']['supervisor']['pid'] = q.os.getpid()
    with pytest.raises((ValueError, KeyError, FileNotFoundError)):
        q.validate_request(r)


def supervisor_row(f):
    s = f.request['predecessor']['supervisor']
    return dict(ProcessId=s['pid'], CreationDate=s['creation_date'], ExecutablePath=s['executable'], CommandLine=s['command_line'])


def predecessor_files(f, monkeypatch):
    root, pins, records = Path(f.config['output']), {}, []
    producer = q.legacy.implementation_bindings()
    published_pin = put(root/'request.json', dict(f.config, producer_module=q.legacy.PRODUCER_MODULE,
        producer_implementation_bindings=producer, training_approved=False, source_cap_reset=False))
    for index, (family, count) in enumerate(q.legacy.BATCH_SHAPE, 1):
        folder = root/f'batch_{index:03d}'
        result_pin = put(folder/'capture'/'result.json', {})
        audit_path = folder/'original_postexit_audit.json'
        audit_pin = put(audit_path, dict(counts={'hold': count}))
        receipt = folder/'original_launcher_receipt.json'
        pin = put(receipt, dict(result_sha256=result_pin, audit_path=str(audit_path), audit_sha256=audit_pin))
        pins.update({str(receipt): pin, str(audit_path): audit_pin})
        records.append(dict(batch_index=index, family=family, planned_cases=count, capture=str(folder/'capture'),
            result_sha256=result_pin, audit_counts={'hold': count}, launcher_receipt_path=str(receipt), launcher_receipt_sha256=pin))
    put(root/'result.json', dict(schema=q.legacy.SCHEMA, state=q.legacy.RESULT_STATE,
        queue_request_path=str(root/'request.json'), queue_request_sha256=published_pin,
        producer_module=q.legacy.PRODUCER_MODULE, producer_implementation_bindings=producer,
        planned_cases=39, records=records, training_approved=False, source_cap_reset=False))
    calls = []
    monkeypatch.setattr(q.predecessor_adapter, 'original_observed_rows', lambda pin: calls.append(pin) or ([], {}, pins))
    return root, calls


def test_predecessor_wait_then_exact_replay(fixture, monkeypatch):
    f = fixture
    monkeypatch.setattr(q.legacy, 'supervisor_rows', lambda pid: [supervisor_row(f)])
    assert q.predecessor_completion(f.request['predecessor'], f.config) is None
    root, calls = predecessor_files(f, monkeypatch)
    assert q.predecessor_completion(f.request['predecessor'], f.config) is None and not calls
    monkeypatch.setattr(q.legacy, 'supervisor_rows', lambda pid: [])
    result = q.predecessor_completion(f.request['predecessor'], f.config)
    assert len(calls) == 1 and calls[0].capture_path == str(root/'batch_002'/'capture')
    assert result['process_exit_evidence'].endswith('not_OS_exit_code')
    q.oc.bind_all(result['bindings'])


@pytest.mark.parametrize('kind', ['absent_no_terminal', 'partial', 'wrong_request', 'wrong_producer', 'missing_receipt',
    'failed', 'reused_pid', 'metadata_missing', 'query_error', 'replay_error'])
def test_predecessor_failure_is_not_ignored(fixture, monkeypatch, kind):
    f = fixture
    root, _ = predecessor_files(f, monkeypatch)
    monkeypatch.setattr(q.legacy, 'supervisor_rows', lambda pid: [])
    if kind == 'absent_no_terminal': (root/'result.json').unlink()
    elif kind in ('partial', 'wrong_request', 'wrong_producer'):
        value = q.oc.read_json(root/'result.json')
        if kind == 'partial': value['records'].pop()
        elif kind == 'wrong_request': value['queue_request_sha256'] = 'f'*64
        else: value['producer_module'] = 'made_up'
        replace(root/'result.json', value)
    elif kind == 'missing_receipt': (root/'batch_001'/'original_launcher_receipt.json').unlink()
    elif kind == 'failed': put(root/'failure.json', {})
    elif kind in ('reused_pid', 'metadata_missing'):
        row = supervisor_row(f)
        row['CreationDate'] = None if kind == 'metadata_missing' else 'changed'
        monkeypatch.setattr(q.legacy, 'supervisor_rows', lambda pid: [row])
    elif kind == 'query_error': monkeypatch.setattr(q.legacy, 'supervisor_rows', lambda pid: (_ for _ in ()).throw(OSError('CIM failed')))
    elif kind == 'replay_error': monkeypatch.setattr(q.predecessor_adapter, 'original_observed_rows', lambda pin: (_ for _ in ()).throw(ValueError('bad receipt')))
    with pytest.raises((ValueError, OSError)):
        q.predecessor_completion(f.request['predecessor'], f.config)


@pytest.fixture
def harness(fixture, monkeypatch):
    f = fixture
    request_path = f.tmp/'request.json'
    request_pin = put(request_path, f.request)
    put(Path(f.config['output'])/'result.json', {})
    monkeypatch.setattr(q, 'owner_lock', nullcontext)
    monkeypatch.setattr(q.legacy, 'supervisor_rows', lambda pid: [])
    prior = dict(bindings={f.request['predecessor']['request_path']: f.request['predecessor']['request_sha256']})
    monkeypatch.setattr(q, 'predecessor_completion', lambda *a: deepcopy(prior))
    monkeypatch.setattr(q.legacy, 'resource_evidence', lambda p: dict(allowed=True, process_inventory=dict(blockers=[])))
    monkeypatch.setattr(q, 'owner_visibility', lambda report: dict(cpu_fixture=True))
    sleeps, launches, pipeline, audits = [], [], [], []
    monkeypatch.setattr(q.time, 'sleep', lambda seconds: sleeps.append(seconds))
    def runner(command, log_path, *, bindings, environment, native, reserve, launch_check, announce):
        reserve()
        launch_check()
        q.oc.bind_all(bindings)
        assert native and environment['PYTHONPATH'].split(q.os.pathsep)[0] == f.request['native_deps']
        launches.append(command)
        pipeline.append('launch')
        announce(2000 + len(launches))
        submitted, capture = Path(command[command.index('--plan')+1]), Path(command[-1])
        q.oc.new_destination(capture, [submitted.parent])
        capture.mkdir()
        put(capture/'request.json', dict(plan_path=str(submitted), plan_sha256=q.oc.sha256(submitted),
            worker_module=q.native_pair_v2.WORKER_MODULE, worker_implementation_bindings=q._PAIR_CODE))
        put(capture/'result.json', dict(plan_sha256=q.oc.sha256(submitted), request_sha256=q.oc.sha256(capture/'request.json'),
            state='generated_native_pair_captured_pending_visual_review', worker_module=q.native_pair_v2.WORKER_MODULE,
            worker_implementation_bindings=q._PAIR_CODE, samples=[dict(sample_id=n) for n in ('original_control', 'generated_variant')]))
        pipeline.append('exit0')
        return 0
    def original_audit(capture, *, result_sha256):
        pipeline.append('audit')
        audits.append(str(capture))
        return dict(counts={'hold_visual_clarity': 3})
    def pair_build(submitted, capture, output):
        pipeline.append('build')
        put(output/'result.json', dict(cpu_only=True))
    def pair_audit(directories, output, *, requalify):
        assert requalify is False
        pipeline.append('audit')
        audits.append(str(directories[0]))
        result = dict(integrity_held_pairs=0, record_decisions={'hold_visual_clarity': 1, 'reject_clear_task': 1})
        put(output, result)
        return result
    monkeypatch.setattr(q.legacy.campaign, 'run_checked', runner)
    monkeypatch.setattr(q.legacy.audit, 'audit_capture', original_audit)
    monkeypatch.setattr(q.native_clear_annotation, 'build', pair_build)
    monkeypatch.setattr(q.automated_native_review, 'run', pair_audit)
    return SimpleNamespace(f=f, request_path=request_path, request_pin=request_pin, launches=launches,
        pipeline=pipeline, audits=audits, sleeps=sleeps, runner=runner, original_audit=original_audit,
        pair_audit=pair_audit, run=lambda: q.run(request_path, request_pin))


def stop(h, pin=None):
    put(h.f.output/'stop.json', dict(schema=q.STOP_SCHEMA, request_sha256=pin or h.request_pin, action='stop'))


def test_full_serial_order_copies_and_holds_continue(harness):
    h = harness
    before = [Path(j['plan_path']).read_bytes() for j in h.f.paths]
    result = h.run()
    assert result['state'] == 'complete' and not result['cooperative_stop_acknowledged']
    assert h.pipeline == ['launch', 'exit0', 'audit', 'launch', 'exit0', 'build', 'audit']
    assert [p for p, _ in h.f.checks] == ['original', 'pair', 'original', 'pair']
    for index, record in enumerate(result['records']):
        receipt = q.oc.read_json(q.oc.pin(record['receipt_path'], record['receipt_sha256']))
        submitted, capture = Path(receipt['submitted_plan_path']), Path(receipt['capture'])
        assert submitted == capture.parent/'submitted'/'plan.json'
        assert submitted.read_bytes() == before[index] == Path(h.f.paths[index]['plan_path']).read_bytes()
        assert receipt['exit_code'] == 0 and receipt['audit']['counts']
        assert receipt['source_plan_path'] == h.f.paths[index]['plan_path']
        assert all(receipt[k] is False for k in q.FLAGS)
        q.oc.bind_all(receipt['source_bindings'])
        q.oc.bind_all(receipt['audit']['bindings'])
    q.oc.bind_all(result['bindings'])


@pytest.mark.parametrize('when', ['predecessor_wait', 'resource_wait', 'during_child', 'during_audit'])
def test_stop_ack_only_at_clean_boundary(harness, monkeypatch, when):
    h = harness
    if when == 'predecessor_wait':
        monkeypatch.setattr(q, 'predecessor_completion', lambda *a: None)
        monkeypatch.setattr(q.time, 'sleep', lambda seconds: stop(h))
    elif when == 'resource_wait':
        monkeypatch.setattr(q.legacy, 'resource_evidence', lambda p: dict(allowed=False))
        monkeypatch.setattr(q.time, 'sleep', lambda seconds: stop(h))
    elif when == 'during_child':
        def runner(*a, **k):
            announce = k['announce']
            k['announce'] = lambda pid: (announce(pid), stop(h))
            return h.runner(*a, **k)
        monkeypatch.setattr(q.legacy.campaign, 'run_checked', runner)
    else:
        def audit(*a, **k):
            stop(h)
            assert not (h.f.output/'result.json').exists()
            return h.original_audit(*a, **k)
        monkeypatch.setattr(q.legacy.audit, 'audit_capture', audit)
    result = h.run()
    assert result['state'] == 'stopped_clean' and result['cooperative_stop_acknowledged']
    assert result['bindings'][str(h.f.output/'stop.json')] == q.oc.sha256(h.f.output/'stop.json')
    expected = int(when in ('during_child', 'during_audit'))
    assert len(h.launches) == len(h.audits) == len(result['records']) == expected
    assert not (h.f.output/'failure.json').exists()


@pytest.mark.parametrize('kind', ['exit', 'bool_exit', 'exception', 'uncertain_cleanup', 'audit', 'stop_then_audit_error',
    'source_tamper', 'copy_tamper', 'event_tamper', 'failure_file', 'wrong_stop', 'resource_race', 'prior_receipt_tamper'])
def test_first_failure_never_continues_or_acknowledges_stop(harness, monkeypatch, kind):
    h = harness
    def runner(*a, **k):
        if kind == 'copy_tamper':
            reserve = k['reserve']
            def tamper():
                reserve()
                Path(a[0][a[0].index('--plan')+1]).write_text('{}', encoding='utf-8')
            k['reserve'] = tamper
        code = h.runner(*a, **k)
        capture = Path(h.launches[-1][-1])
        if kind == 'exit': return 2
        if kind == 'bool_exit': return False
        if kind == 'exception': raise OSError('owned wait error')
        if kind == 'uncertain_cleanup': raise q.legacy.campaign.OwnedChildCleanupError('uncertain')
        if kind == 'source_tamper': replace(Path(next(iter(h.f.source))), {})
        if kind == 'event_tamper': replace(capture.parent/'launch.json', {})
        if kind == 'failure_file': put(capture/'failure.json', {})
        if kind == 'wrong_stop': stop(h, 'f'*64)
        if kind == 'prior_receipt_tamper' and len(h.launches) == 2: replace(h.f.output/'originals'/'J0'/'receipt.json', {})
        return code
    monkeypatch.setattr(q.legacy.campaign, 'run_checked', runner)
    if kind in ('audit', 'stop_then_audit_error'):
        def fail(*a, **k):
            if kind == 'stop_then_audit_error': stop(h)
            raise ValueError('audit failed')
        monkeypatch.setattr(q.legacy.audit, 'audit_capture', fail)
    if kind == 'resource_race':
        calls = []
        monkeypatch.setattr(q.legacy, 'resource_evidence', lambda p: calls.append(1) or dict(allowed=len(calls) == 1))
    with pytest.raises((ValueError, OSError, q.legacy.campaign.OwnedChildCleanupError)):
        h.run()
    assert len(h.launches) <= (2 if kind == 'prior_receipt_tamper' else 1)
    failure = q.oc.read_json(h.f.output/'failure.json')
    assert not failure['cooperative_stop_acknowledged'] and not (h.f.output/'result.json').exists()


@pytest.mark.parametrize('kind', ['wrong_worker', 'wrong_plan', 'missing_mode', 'integrity_hold'])
def test_pair_receipt_and_audit_fail_closed(harness, monkeypatch, kind):
    h = harness
    def runner(*a, **k):
        code = h.runner(*a, **k)
        if len(h.launches) == 2 and kind != 'integrity_hold':
            path = Path(h.launches[-1][-1])/('request.json' if kind == 'wrong_plan' else 'result.json')
            value = q.oc.read_json(path)
            if kind == 'wrong_plan': value['plan_path'] = h.f.paths[1]['plan_path']
            elif kind == 'wrong_worker': value['worker_implementation_bindings'] = {}
            else: value['samples'].pop()
            replace(path, value)
        return code
    monkeypatch.setattr(q.legacy.campaign, 'run_checked', runner)
    if kind == 'integrity_hold':
        monkeypatch.setattr(q.automated_native_review, 'run', lambda *a, **k: dict(integrity_held_pairs=1, record_decisions={'hold': 2}))
    with pytest.raises(ValueError):
        h.run()
    assert len(h.launches) == 2 and not (h.f.output/'controls'/'J1'/'receipt.json').exists()


def test_existing_partial_output_is_never_resumed(harness):
    h = harness
    put(h.f.output/'originals'/'J0'/'partial.json', {})
    with pytest.raises(ValueError, match='New, non-root'): h.run()
    assert not h.launches


def test_post_resource_pin_check_before_popen(harness, monkeypatch):
    h, calls = harness, []
    def resource(path):
        calls.append(1)
        if len(calls) == 2:
            replace(Path(next(iter(h.f.source))), {})
        return dict(allowed=True, process_inventory=dict(blockers=[]))
    monkeypatch.setattr(q.legacy, 'resource_evidence', resource)
    with pytest.raises(ValueError, match='pinned file'):
        h.run()
    assert not h.launches and (h.f.output/'failure.json').exists()


def test_full_validated_job_and_frame_bounds(fixture):
    f, r = fixture, deepcopy(fixture.request)
    plan = dict(f.plans[0], sample_count_limit=64, cases=[{}]*64)
    jobs = []
    for i in range(65):
        path = f.tmp/f'bounded_{i}'/'plan.json'
        jobs.append(dict(id=f'B{i}', worker_kind=q.WORKERS[0], plan_path=str(path), plan_sha256=put(path, plan)))
    r['phases'] = [dict(id='bounded', jobs=jobs[:64])]
    r['planned_frames'] = 4096
    _, checked = q.validate_request(r)
    assert len(checked) == 64 and sum(j['checked']['count'] for j in checked) == 4096
    r['phases'][0]['jobs'] = jobs
    with pytest.raises(ValueError, match='64 manifest jobs'):
        q.validate_request(r)


def test_full_validator_and_prerequisite_errors_are_not_suppressed(fixture, monkeypatch):
    f = fixture
    def fail(*a):
        raise ValueError('full frozen validator rejects source')
    monkeypatch.setattr(q.original_plan, 'check_plan', fail)
    with pytest.raises(ValueError, match='full frozen'):
        q.validate_plan(q.WORKERS[0], f.paths[0]['plan_path'], f.paths[0]['plan_sha256'])
    monkeypatch.setattr(q.pair_plan, 'verify_sensor_prerequisite', fail)
    with pytest.raises(ValueError, match='full frozen'):
        q.validate_plan(q.WORKERS[1], f.paths[1]['plan_path'], f.paths[1]['plan_sha256'])


def test_capture_supervisor_requires_actual_live_identity(fixture, monkeypatch):
    f, pred = fixture, fixture.request['predecessor']
    monkeypatch.setattr(q.legacy, 'supervisor_rows', lambda pid: [supervisor_row(f)])
    assert q.capture_supervisor(pred['supervisor']['pid'], pred['request_path'], pred['request_sha256']) == pred['supervisor']
    monkeypatch.setattr(q.legacy, 'supervisor_rows', lambda pid: [])
    with pytest.raises(ValueError, match='before exit'):
        q.capture_supervisor(pred['supervisor']['pid'], pred['request_path'], pred['request_sha256'])


def owner_row(request_path, *, pid=None, exe=None):
    exe = str(Path(exe or q.sys.executable).resolve())
    return dict(ProcessId=pid or q.os.getpid(), ParentProcessId=120,
        CreationDate='2026-09-16T01:00:00+00:00', Name=Path(exe).name, ExecutablePath=exe,
        CommandLine=subprocess.list2cmdline([exe, '-B', '-m', q.PRODUCER_MODULE,
            '--request', str(request_path), '--request-sha256', 'a'*64]))


def test_frozen_guard_sees_owner_noncurrent_and_keeps_native_children_blocked(tmp_path):
    guard = q.legacy.guard
    owner = owner_row(tmp_path/'serial_phases_request.json', pid=q.os.getpid()+100000)
    child = dict(ProcessId=owner['ProcessId']+1, ParentProcessId=owner['ProcessId'],
        CreationDate='2026-09-16T01:01:00+00:00', Name='kit.exe',
        ExecutablePath=r'D:\isaac-sim-6.0.1\kit\kit.exe',
        CommandLine=r'D:\isaac-sim-6.0.1\kit\kit.exe -m sim_data.native_dataset.native_pair_v2')
    report = guard.classify_processes([owner, child], native_roots=guard.DEFAULT_NATIVE_ROOTS)
    assert report['classifications'][0]['classification'] == 'unrelated_python'
    assert report['classifications'][0]['blocking'] is False
    assert [r['ProcessId'] for r in report['blockers']] == [child['ProcessId']]
    assert report['classifications'][1]['classification'] == 'native_kit'
    assert not report['exclusive_launch_guaranteed']
    # The registered workers use the actual Kit-only CIM filter, unchanged.
    from ..native_generated_pair import check_process_admission
    assert check_process_admission([child], child['ProcessId'])['allowed_kit_pids'] == [child['ProcessId']]
    with pytest.raises(ValueError, match='Another Kit'):
        check_process_admission([child, dict(child, ProcessId=child['ProcessId']+1)], child['ProcessId'])


def test_new_owner_cannot_claim_existing_pinned_supervisor_exemption(tmp_path):
    row = owner_row(tmp_path/'serial_phases_request.json')
    with pytest.raises(ValueError, match='Known coordinator'):
        q.legacy.guard.supervisor_proof(row['ExecutablePath'], q.legacy.guard._argv(row['CommandLine']), q.implementation_bindings())


@pytest.mark.parametrize('marker', ['isaac', 'native_generated', 'native_same_callback'])
def test_owner_native_marked_request_path_still_blocks(tmp_path, marker):
    row = owner_row(tmp_path/(marker+'_request.json'))
    decision = q.legacy.guard.classify_process(row, native_roots=q.legacy.guard.DEFAULT_NATIVE_ROOTS)
    assert decision['blocking'] and decision['classification'] == 'unknown_relevant_python'


@pytest.mark.parametrize('kind', ['valid', 'native_marker', 'missing_metadata', 'changed_command', 'kit_name', 'wrong_exe'])
def test_owner_boundary_reclassification_uses_unmodified_guard(tmp_path, monkeypatch, kind):
    row = owner_row(tmp_path/('isaac_request.json' if kind == 'native_marker' else 'phase_request.json'))
    observed = q.legacy.guard.classify_process(row, native_roots=q.legacy.guard.DEFAULT_NATIVE_ROOTS)
    # Simulate the existing self exception; the new boundary must NOT trust it.
    observed.update(classification='current_cpu_supervisor', blocking=False)
    if kind == 'changed_command': row['CommandLine'] += ' --changed'
    elif kind == 'kit_name': observed['Name'] = 'kit.exe'
    elif kind == 'wrong_exe': row['ExecutablePath'] = r'C:\elsewhere\python.exe'
    monkeypatch.setattr(q.legacy, 'supervisor_rows', lambda pid: [] if kind == 'missing_metadata' else [row])
    if kind == 'valid':
        result = q.owner_visibility(dict(classifications=[observed]))
        assert result['classification']['classification'] == 'unrelated_python'
        assert result['current_pid_exception_used'] is False
    else:
        with pytest.raises(ValueError): q.owner_visibility(dict(classifications=[observed]))


def test_incompatible_owner_never_reaches_popen(harness, monkeypatch):
    def reject(report):
        raise ValueError('Owner blocks frozen guard')
    monkeypatch.setattr(q, 'owner_visibility', reject)
    with pytest.raises(ValueError, match='Owner blocks'):
        harness.run()
    assert not harness.launches


def test_implementation_pins_and_compact_unregistered():
    pins = q.implementation_bindings()
    assert pins[str(Path(q.__file__).resolve())] == q.oc.sha256(q.__file__)
    assert pins[str(Path(q.native_pair_v2.__file__).resolve())] == q.PAIR_SHA256
    assert set(q.oc.LOADED_IMPLEMENTATION) <= pins.keys()
    with pytest.raises(ValueError, match='Unregistered'):
        q.worker_command('anything', 'compact_query_v2', Path('p'), 'f'*64, Path('c'))


@pytest.mark.parametrize('name,pin', [
    ('C01_A.json', '330204abfe9d136d3f145b29167a8673b198801e3741301010bcfdcf8a0676e7'),
    ('C02_B.json', '70109da846836477c668814b7cd30b0113b81cd352e2b9e0b396fd5352329961'),
    ('C03_A.json', '3bc6fdc9417a50bbe4a531d92cf8b175fb1e419619c468bbb6fc978edf169756'),
    ('C04_A.json', '2a9f0e54ecefcddc4d0ec9beac0916fecbf9e9abcb9ce55f27f00305bfdc4ef5'),
    ('C05_A.json', '9b11f2f44302f179b095292c28464505e10c29bf156fff3cc687faf9ab52d751'),
])
def test_actual_control_plans_full_readonly_validation(name, pin):
    root = Path(q.__file__).resolve().parents[4]
    path = root/'data/sim_data/diagnostics/geometry_comparison_candidates_20260916_v1/native_plans'/name
    if not path.is_file():
        pytest.skip('Host-specific pinned control fixture absent')
    # Actual validators and sensor prerequisite, no mocks, no private distances.
    result = q.validate_plan(q.WORKERS[1], path, pin)
    assert result['count'] == 2 and result['plan']['split'] == 'train'
    q.oc.pin(path, pin)


def test_actual_original_plan_full_readonly_validation():
    root = Path(q.__file__).resolve().parents[4]
    path = root/'data/sim_data/diagnostics/original_seed73_pilot_plan_20260916_v1/plan.json'
    if not path.is_file():
        pytest.skip('Host-specific pinned original fixture absent')
    pin = '41e8e09182a9202230383d27f2cd460dae32c178e289a73391d75d739d442cba'
    result = q.validate_plan(q.WORKERS[0], path, pin)
    assert result['count'] == 1 and result['plan']['source_family'] == 'seed73_full'
    q.oc.pin(path, pin)


@pytest.mark.skipif(q.os.name != 'nt', reason='Windows OS mutex')
def test_real_os_mutex_contention_and_release(monkeypatch):
    # Isolated TEST mutex; never contend with an actual coordinator.
    monkeypatch.setattr(q, 'LOCK_NAME', q.LOCK_NAME + f'.test.{q.os.getpid()}')
    failures = []
    def contender():
        try:
            with q.owner_lock(): failures.append('unexpected acquisition')
        except ValueError: failures.append('busy')
    with q.owner_lock():
        thread = threading.Thread(target=contender)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive() and failures == ['busy']
    with q.owner_lock(): pass
