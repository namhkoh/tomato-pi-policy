"""Versioned post-reboot serial owner derived from frozen serial_phases v1.

The only changed execution prerequisite is an explicit pinned user stop/new-boot
handoff; no old campaign is claimed complete. All prepared worker validations,
resource/OS-mutex gates, owned exit and post-exit native audits remain unchanged.
Receipts have distinct v2 schemas and require the corresponding v2 adapter.
"""
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import ast
import ctypes
import importlib.util
import os
import re
import sys
import time
import traceback

from .. import generated_capture as pair_plan
from .. import native_clear_annotation, automated_native_review
from ..native_original_capture import contracts as oc, prepare as original_plan
from ..native_original_capture import serial_queue as legacy
from . import original_inventory_v2 as predecessor_adapter
from . import reboot_gate_v1 as reboot
from . import native_pair_v2
from . import serial_query_qualification as query_kind
from . import serial_persistence_qualification as persistence_kind

SCHEMA = 'greenhouse.native_reboot_phases.request.v1'
RECEIPT_SCHEMA = 'greenhouse.native_reboot_phases.job.v1'
CHECKPOINT_SCHEMA = 'greenhouse.native_reboot_phases.checkpoint.v1'
RESULT_SCHEMA = 'greenhouse.native_reboot_phases.result.v1'
STOP_SCHEMA = 'greenhouse.native_reboot_phases.stop.v1'
PRODUCER_MODULE = 'sim_data.native_dataset.serial_phases_reboot_v2'
WORKERS = ('original_batch.v1', 'matched_pair.v2', query_kind.KIND, persistence_kind.KIND)
QUALIFICATIONS = {query_kind.KIND: query_kind, persistence_kind.KIND: persistence_kind}
QUALIFICATION_LIMITS = {query_kind.KIND: 2, persistence_kind.KIND: 1}
PAIR_SHA256 = 'c863ff0f34dd8bab1117ff7b523a38290b0fc30e816bb51ca8d3ddd21d502667'
LOCK_NAME = r'Global\greenhouse.native_serial_phases.owner.v1'
FLAGS = dict(training_approved=False, source_cap_reset=False, visual_approval=False,
             independent_execution_attested=False)


def _code_closure():
    # Literal repository imports, including function-local imports, without Kit.
    root = Path(__file__).resolve().parents[1]
    search = (root.parent, root.parents[1])
    pending, found = [Path(__file__)], {}
    while pending:
        path = pending.pop().resolve()
        if str(path) in found:
            continue
        raw = path.read_bytes()
        found[str(path)] = oc.digest(raw)
        base = next(p for p in search if path.is_relative_to(p))
        package = '.'.join(path.relative_to(base).parts[:-1])
        for node in ast.walk(ast.parse(raw)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                name = '.' * node.level + (node.module or '')
                module = importlib.util.resolve_name(name, package) if node.level else name
                names = [module] + [module + '.' + a.name for a in node.names if a.name != '*']
            for name in names:
                for folder in search:
                    candidate = folder.joinpath(*name.split('.'))
                    pending.extend(p for p in (candidate.with_suffix('.py'), candidate / '__init__.py') if p.is_file())
    return found


_LOADED = oc.merge_bindings(predecessor_adapter.implementation_bindings(),
    query_kind.implementation_bindings(), persistence_kind.implementation_bindings(), _code_closure())
_PAIR_CODE = native_pair_v2.implementation_bindings()


def implementation_bindings():
    oc.bind_all(_LOADED)
    query_kind.implementation_bindings()
    persistence_kind.implementation_bindings()
    oc.require(_PAIR_CODE[str(Path(native_pair_v2.__file__).resolve())] == PAIR_SHA256,
               'Unregistered matched-pair executor')
    return dict(_LOADED)


@contextmanager
def owner_lock():
    """OS-held, non-configurable lock; no stale file lock and no process killing."""
    oc.require(os.name == 'nt', 'Windows owner lock required')
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateMutexW.argtypes = (ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR)
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.ReleaseMutex.argtypes = kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    handle = kernel.CreateMutexW(None, False, LOCK_NAME)
    oc.require(bool(handle), 'Cannot open serial owner mutex')
    acquired = False
    try:
        status = kernel.WaitForSingleObject(handle, 0)
        acquired = status in (0, 0x80)
        oc.require(status == 0, 'Serial owner busy, abandoned, or lock unavailable')
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def validate_plan(kind, path, pin, *, query=None):
    oc.require(kind in WORKERS, 'Unregistered worker kind')
    if kind in QUALIFICATIONS:
        return QUALIFICATIONS[kind].validate_plan(path, pin, query)
    path = oc.pin(legacy._path(str(path)), pin)
    plan = oc.read_json(path)
    oc.require(plan['split'] == 'train' and oc.FROZEN_SPLITS.get(plan['source_family']) == 'train',
               'Fixed original TRAIN donor required')
    if kind == WORKERS[0]:
        original_plan.check_plan(plan)
        count = plan['sample_count_limit']
        oc.require(type(count) is int and 1 <= count <= 64 and len(plan['cases']) == count,
                   'Original batch requires 1..64 cases')
        pins = oc.merge_bindings(plan['source_bindings'], plan['implementation_bindings'])
        roots = original_plan.protected_roots(plan)
    else:
        pair_plan.check_plan(plan)
        prerequisite = pair_plan.verify_sensor_prerequisite(plan)
        count = plan['sample_count_limit']
        oc.require(type(count) is int and count == 2 and plan['modes'] == ['original_control', 'generated_variant'],
                   'Exactly two matched control frames required')
        manifest = legacy._path(plan['source_capture']) / 'manifest.json'
        pins = oc.merge_bindings(plan['source_bindings'], plan.get('generator_code_bindings', {}),
            prerequisite['bindings'], oc.read_json(Path(plan['prerequisite_directory']) / 'request.json')['source_bindings'],
            {str(manifest): oc.sha256(manifest)})
        roots = [plan[k] for k in ('source_capture', 'variant_directory', 'prerequisite_directory')]
        roots += [oc.read_json(manifest)['package'], str(Path(plan['source_collection_plan']).parent)]
    pins = oc.merge_bindings(pins, {str(path): pin})
    oc.bind_all(pins)
    return dict(plan=plan, count=count, bindings=pins,
                roots=[*roots, path.parent, *(Path(p).parent for p in pins)])


def validate_request(request):
    legacy._keys(request, ('schema', 'output', 'isaac_python', 'native_deps', 'max_jobs',
        'max_planned_frames', 'planned_frames', 'predecessor', 'phases'), 'phase request')
    oc.require(request['schema'] == SCHEMA and type(request['max_jobs']) is int and request['max_jobs'] == 64
        and type(request['max_planned_frames']) is int and request['max_planned_frames'] == 4096,
        'Explicit initial bounds 64 jobs / 4096 planned frames required')
    output, isaac, deps = (legacy._path(request[k]) for k in ('output', 'isaac_python', 'native_deps'))
    oc.require(isaac.is_file() and deps.is_dir(), 'Missing native runtime')
    pred = request['predecessor']
    config, roots, _ = reboot.validate_predecessor(pred)
    roots += [isaac.parent, deps]
    roots += [Path(p).parent for p in implementation_bindings()]
    phases = request['phases']
    oc.require(isinstance(phases, list) and 1 <= len(phases) <= 64, 'Bounded nonempty phases required')
    jobs, phase_ids, job_ids = [], set(), set()
    qualification_pins = {kind: set() for kind in QUALIFICATIONS}
    for phase in phases:
        legacy._keys(phase, ('id', 'jobs'), 'phase')
        _unique_id(phase['id'], phase_ids)
        oc.require(isinstance(phase['jobs'], list) and phase['jobs'], 'Nonempty phase jobs required')
        for job in phase['jobs']:
            oc.require(len(jobs) < 64, 'At most 64 manifest jobs')
            kind = job['worker_kind']
            helper = QUALIFICATIONS.get(kind)
            extra = helper.JOB_FIELDS if helper is not None else ()
            legacy._keys(job, ('id', 'worker_kind', 'plan_path', 'plan_sha256', *extra), 'job')
            if helper is not None:
                seen = qualification_pins[kind]
                oc.require(len(seen) < QUALIFICATION_LIMITS[kind] and job['plan_sha256'] not in seen,
                           'Bounded distinct qualification jobs required')
                seen.add(job['plan_sha256'])
            _unique_id(job['id'], job_ids)
            path = legacy._path(job['plan_path'])
            reused = [j for j in jobs if legacy._path(j['plan_path']) == path or j['plan_sha256'] == job['plan_sha256']]
            oc.require(not reused or (len(reused) == 1 and _seed101_control_reuse(reused[0], job, persistence_kind.KIND)),
                       'Duplicate/aliased source plan; only exact seed101 V2 then V3 control reuse allowed')
            checked = validate_plan(job['worker_kind'], path, job['plan_sha256'],
                                    query={k: deepcopy(job[k]) for k in extra} if helper is not None else None)
            roots.extend(checked['roots'])
            jobs.append(dict(phase_id=phase['id'], **deepcopy(job), checked=checked))
    total = sum(j['checked']['count'] for j in jobs)
    oc.require(type(request['planned_frames']) is int and request['planned_frames'] == total <= 4096,
               'Exact bounded planned frame sum required')
    oc.new_destination(output, roots)
    return config, jobs


def _seed101_control_reuse(previous, current, v3_kind):
    '''Internal exception: one exact V2 seed101 control followed by its V3 control.

    v3_kind comes from the fixed code registry, never from request configuration.
    Call only for exactly one previous plan occurrence; a third use is forbidden.
    Neither a copied plan path nor another same-family plan qualifies.
    '''
    case = query_kind.CASES[0]
    path = (query_kind._PRIOR / case[0] / 'plan.json').resolve()
    return (previous['worker_kind'] == query_kind.KIND and current['worker_kind'] == v3_kind
        and all(legacy._path(j['plan_path']) == path and j['plan_sha256'] == case[5]
                for j in (previous, current)))


def _unique_id(value, seen):
    oc.require(isinstance(value, str) and re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,47}', value)
        and value.lower() not in seen and value.upper() not in
        {'CON', 'PRN', 'AUX', 'NUL', *(f'COM{i}' for i in range(10)), *(f'LPT{i}' for i in range(10))},
        'Unique safe phase/job ID required')
    seen.add(value.lower())


def predecessor_completion(pred, config):
    return reboot.completion(pred, config)


def _no_failure(root):
    oc.require(not any((Path(root) / n).exists() for n in ('failure.json', 'superseded_idle_queue.json')),
               'Failed or superseded capture/queue')


def _predecessor_absent(pred, config):
    reboot.assert_current_boot(pred, config)


def worker_command(isaac, kind, submitted, pin, capture, *, checked=None):
    oc.require(kind in WORKERS, 'Unregistered worker kind')
    if kind in QUALIFICATIONS:
        return QUALIFICATIONS[kind].command(isaac, submitted, pin, capture, checked)
    if kind == WORKERS[0]:
        return legacy.batch_command(isaac, dict(plan_path=str(submitted), plan_sha256=pin), capture)
    return [str(legacy._path(isaac)), '-m', native_pair_v2.WORKER_MODULE,
            '--plan', str(submitted), '--output', str(capture)]


def owner_visibility(process_inventory):
    '''Reclassify actual self metadata as another observer would, no exemption.'''
    pid = os.getpid()
    selected = [r for r in process_inventory['classifications'] if r['ProcessId'] == pid]
    rows = legacy.supervisor_rows(pid)
    oc.require(len(selected) == len(rows) == 1, 'Exact owner metadata missing')
    observed, row = selected[0], rows[0]
    oc.require(row['ProcessId'] == pid and row['ExecutablePath'] == observed['ExecutablePath']
        and legacy._path(row['ExecutablePath']) == Path(sys.executable).resolve()
        and oc.digest(row['CommandLine'].encode('utf-8')) == observed['command_line_sha256'],
        'Owner metadata changed between CIM snapshots')
    decision = legacy.guard.classify_process(dict(row, Name=observed['Name']),
        native_roots=legacy.guard.DEFAULT_NATIVE_ROOTS)
    oc.require(decision['blocking'] is False and decision['classification'] == 'unrelated_python',
        'Owner blocks frozen guard from child perspective; no supervisor exemption is available')
    return dict(metadata=dict(row, Name=observed['Name']), classification=decision,
        current_pid_exception_used=False, running_code_attested=False)


def _tree_pins(root):
    root = Path(root).resolve()
    paths = [p for p in root.rglob('*') if p.is_file()]
    oc.require(all(p.resolve().is_relative_to(root) for p in paths), 'Output file escapes owned tree')
    return {str(p.resolve()): oc.sha256(p) for p in paths}


def postexit_audit(kind, submitted, pin, capture, folder, *, checked=None):
    """Called only after actual owned exit zero; never promote visual decisions."""
    _no_failure(capture)
    raw = _tree_pins(capture)
    oc.bind_all(raw)
    if kind in QUALIFICATIONS:
        result = QUALIFICATIONS[kind].postexit_audit(submitted, pin, capture, folder, checked)
        oc.require(_tree_pins(capture) == raw, 'Query capture files changed during postexit audit')
        oc.bind_all(raw)
        _no_failure(capture)
        result['bindings'] = oc.merge_bindings(raw, result['bindings'])
        return result
    if kind == WORKERS[0]:
        completion, review = legacy.complete_batch(capture, dict(plan_path=str(submitted), plan_sha256=pin))
        audit_path = folder / 'original_postexit_audit.json'
        oc.write_new(audit_path, review)
        counts = review['counts']
    else:
        oc.require(kind == WORKERS[1], 'Unregistered audit')
        request, result = (oc.read_json(capture / name) for name in ('request.json', 'result.json'))
        oc.require(request['plan_path'] == str(submitted) and request['plan_sha256'] == pin
            and request['worker_module'] == result['worker_module'] == native_pair_v2.WORKER_MODULE
            and request['worker_implementation_bindings'] == result['worker_implementation_bindings'] == _PAIR_CODE
            and result['state'] == 'generated_native_pair_captured_pending_visual_review'
            and [s['sample_id'] for s in result['samples']] == ['original_control', 'generated_variant'],
            'Wrong submitted pair/worker/result membership')
        annotation = folder / 'annotations'
        audit_path = folder / 'pair_postexit_audit.json'
        oc.new_destination(annotation, [capture, submitted.parent])
        oc.new_destination(audit_path, [capture, submitted.parent])
        native_clear_annotation.build(submitted, capture, annotation)
        review = automated_native_review.run([annotation], audit_path, requalify=False)
        oc.require(type(review['integrity_held_pairs']) is int and review['integrity_held_pairs'] == 0
            and sum(review['record_decisions'].values()) == 2, 'Incomplete/integrity-held pair audit')
        counts = review['record_decisions']
        completion = dict(request_sha256=raw[str(capture / 'request.json')], result_sha256=raw[str(capture / 'result.json')])
    oc.bind_all(raw)
    _no_failure(capture)
    pins = oc.merge_bindings(raw, {str(audit_path): oc.sha256(audit_path)})
    if kind == WORKERS[1]:
        pins = oc.merge_bindings(pins, _tree_pins(annotation))
    return dict(**completion, audit_path=str(audit_path), audit_sha256=oc.sha256(audit_path),
                counts=counts, bindings=pins)


class _CleanStop(Exception):
    """Raised only where no owned child or unfinished audit exists."""


def _check_stop(output, request_sha256):
    path = output / 'stop.json'
    if path.exists():
        pin = oc.sha256(path)
        value = legacy._document(path, pin)
        oc.require(value == dict(schema=STOP_SCHEMA, request_sha256=request_sha256, action='stop'),
                   'Stop marker is not bound to this request')
        oc.pin(path, pin)
        raise _CleanStop({str(path): pin})


def run(request_path, request_sha256):
    oc.require(os.name == 'nt' and not any(n.split('.')[0] in ('isaacsim', 'omni', 'carb') for n in sys.modules),
               'Windows CPU-only coordinator required')
    with owner_lock():
        return _run(oc.pin(legacy._path(str(request_path)), request_sha256), request_sha256)


def _run(request_path, request_sha256):
    request = oc.read_json(request_path)
    config, jobs = validate_request(request)
    output = oc.new_destination(request['output'], [request_path])
    producer = implementation_bindings()
    bindings = oc.merge_bindings(producer, {str(request_path): request_sha256,
        request['predecessor']['request_path']: request['predecessor']['request_sha256']},
        *(j['checked']['bindings'] for j in jobs))
    output.mkdir(parents=True, exist_ok=False)
    records, captures = [], []
    counter = 0

    def save(name, value):
        path = output / name
        oc.write_new(path, dict(value, **FLAGS))
        return {str(path): oc.sha256(path)}

    def checkpoint(state):
        nonlocal counter
        counter += 1
        return save(f'checkpoint_{counter:06d}.json', dict(schema=CHECKPOINT_SCHEMA, state=state,
            request_sha256=request_sha256, completed=deepcopy(records), next_job_index=len(records),
            created_utc=datetime.now(timezone.utc).isoformat()))

    def stable():
        oc.bind_all(bindings)
        _predecessor_absent(request['predecessor'], config)
        for capture in captures:
            _no_failure(capture)

    def failure():
        save('failure.json', dict(schema=RESULT_SCHEMA, state='failed_stop_first', request_sha256=request_sha256,
            completed=records, traceback=traceback.format_exc(), cooperative_stop_acknowledged=False))

    try:
        bindings.update(save('request.json', dict(request, input_request_path=str(request_path),
            input_request_sha256=request_sha256, producer_module=PRODUCER_MODULE,
            producer_implementation_bindings=producer)))
        bindings.update(checkpoint('clean_initial_boundary'))
        while True:
            oc.bind_all(bindings)
            _check_stop(output, request_sha256)
            prior = predecessor_completion(request['predecessor'], config)
            if prior is not None:
                break
            time.sleep(30)
        bindings = oc.merge_bindings(bindings, prior['bindings'], save('predecessor_completion.json', prior))
        _, environment = legacy.v5.worker_environments(request['native_deps'])
        for index, job in enumerate(jobs):
            stable()
            _check_stop(output, request_sha256)
            folder = output / job['phase_id'] / job['id']
            oc.require(folder.resolve() == folder and folder.is_relative_to(output), 'Job escaped owned output')
            oc.new_destination(folder, job['checked']['roots'])
            submitted, capture = folder / 'submitted' / 'plan.json', folder / 'capture'
            submitted.parent.mkdir(parents=True, exist_ok=False)
            with submitted.open('xb') as stream:
                stream.write(oc.pin(job['plan_path'], job['plan_sha256']).read_bytes())
            oc.pin(submitted, job['plan_sha256'])
            oc.new_destination(capture, [submitted.parent, *job['checked']['roots']])
            bindings[str(submitted)] = job['plan_sha256']
            command = worker_command(request['isaac_python'], job['worker_kind'], submitted, job['plan_sha256'], capture,
                                     checked=job['checked'])
            launched = {}

            def reserve():
                while True:
                    stable()
                    _check_stop(output, request_sha256)
                    if legacy.resource_evidence(output)['allowed']:
                        return
                    time.sleep(30)

            def launch_check():
                stable()
                checked = validate_plan(job['worker_kind'], job['plan_path'], job['plan_sha256'],
                    query={k: deepcopy(job[k]) for k in QUALIFICATIONS[job['worker_kind']].JOB_FIELDS}
                        if job['worker_kind'] in QUALIFICATIONS else None)
                oc.require(checked == job['checked'], 'Plan/prerequisite changed before launch')
                oc.pin(submitted, job['plan_sha256'])
                oc.new_destination(capture, [submitted.parent, *checked['roots']])
                evidence = legacy.resource_evidence(output)
                oc.require(evidence['allowed'], 'Native resources changed before launch')
                launched['owner_visibility'] = owner_visibility(evidence['process_inventory'])
                oc.bind_all(bindings)
                _check_stop(output, request_sha256)
                launched['resource_evidence'] = evidence

            def announce(pid):
                oc.require(type(pid) is int and pid > 0, 'Owned child PID required')
                event = folder / 'launch.json'
                oc.write_new(event, dict(schema=RECEIPT_SCHEMA, state='owned_child_running', worker_pid=pid,
                    command=command, job_index=index, phase_id=job['phase_id'], job_id=job['id'],
                    bindings=bindings, resource_evidence=launched['resource_evidence'],
                    owner_visibility=launched['owner_visibility'], **FLAGS))
                launched.update(worker_pid=pid, command=command, event_path=str(event), event_sha256=oc.sha256(event))

            code = legacy.campaign.run_checked(command, folder / 'native.log', bindings=bindings,
                environment=environment, native=True, reserve=reserve, launch_check=launch_check, announce=announce)
            oc.require(type(code) is int and code == 0, 'Owned native worker failed; stop first')
            oc.pin(launched['event_path'], launched['event_sha256'])
            stable()
            audited = postexit_audit(job['worker_kind'], submitted, job['plan_sha256'], capture, folder,
                                     checked=job['checked'])
            stable()
            oc.pin(launched['event_path'], launched['event_sha256'])
            receipt_path = folder / 'receipt.json'
            receipt = dict(schema=RECEIPT_SCHEMA, state='owned_exit0_postexit_audited_pending_admission',
                producer_module=PRODUCER_MODULE, producer_implementation_bindings=producer,
                input_request_path=str(request_path), input_request_sha256=request_sha256,
                job_index=index, phase_id=job['phase_id'], job_id=job['id'], worker_kind=job['worker_kind'],
                source_plan_path=job['plan_path'], plan_sha256=job['plan_sha256'], submitted_plan_path=str(submitted),
                capture=str(capture), planned_source_frames=job['checked']['count'], exit_code=code,
                owned_worker=launched, source_bindings=deepcopy(bindings), audit=audited, **FLAGS)
            if job['worker_kind'] == query_kind.KIND:
                receipt.update(**query_kind.ANNOTATION, training_diversity_increment=0,
                    observation_role='matched_reference_qualification_control_not_new_diversity')
            elif job['worker_kind'] == persistence_kind.KIND:
                receipt.update(**persistence_kind.ANNOTATION, training_diversity_increment=0,
                    observation_role='persistence_qualification_control_not_new_diversity',
                    persistence_policy_sha256=persistence_kind.PERSISTENCE_POLICY_SHA256,
                    persistence_mode=persistence_kind.audit.STRICT_ONLY, qualification_witness_save_all=True,
                    native_persistence_qualified=False)
            oc.write_new(receipt_path, receipt)
            records.append(dict(phase_id=job['phase_id'], job_id=job['id'], worker_kind=job['worker_kind'],
                receipt_path=str(receipt_path), receipt_sha256=oc.sha256(receipt_path)))
            captures.append(capture)
            bindings = oc.merge_bindings(bindings, audited['bindings'], {str(receipt_path): oc.sha256(receipt_path),
                launched['event_path']: launched['event_sha256']})
            bindings.update(checkpoint('owned_exit0_audited_clean_boundary'))
            stable()
            _check_stop(output, request_sha256)
        oc.bind_all(bindings)
        return _terminal(save, 'complete', request_sha256, producer, records, bindings)
    except _CleanStop as stop:
        try:
            bindings = oc.merge_bindings(bindings, stop.args[0])
            oc.bind_all(bindings)
            bindings.update(checkpoint('stopped_clean'))
            return _terminal(save, 'stopped_clean', request_sha256, producer, records, bindings)
        except BaseException:
            failure()
            raise
    except BaseException:
        failure()
        raise


def _terminal(save, state, request_sha256, producer, records, bindings):
    result = dict(schema=RESULT_SCHEMA, state=state, input_request_sha256=request_sha256,
        producer_module=PRODUCER_MODULE, producer_implementation_bindings=producer, records=records,
        bindings=bindings, cooperative_stop_acknowledged=state == 'stopped_clean',
        coordinator_exit_code_claimed=False, **FLAGS)
    save('result.json', result)
    return deepcopy(result)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args(argv)
    run(args.request, args.request_sha256)


if __name__ == '__main__':
    main()
