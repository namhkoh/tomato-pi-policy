"""Serial bounded-view diagnostic followed by the unchanged V4 scale worker.

This coordinator owns only its children. It never interrupts another collector,
changes sensor defaults, resets source budgets, or approves a dataset.
"""
import argparse
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
import time

from ..dataset_review import read_json, write_json, require, verify_bindings
from ..depth_preview import sha256
from ..native_capture_v4 import campaign
from ..native_dataset.compact_qualification import qualify_storage
from ..native_dataset import native_process_guard
from . import bounded_views as probe

SCHEMA = 'greenhouse.serial_bounded12_then_scale_request.v1'


def validate_request(request):
    require(request.get('schema') == SCHEMA, 'Explicit serial probe/scale request required')
    fields = ('prior_campaign', 'controls', 'storage_qualification', 'view_plan',
              'output', 'scale_output', 'isaac_python', 'native_deps', 'schedule')
    paths = {}
    for name in fields:
        require(isinstance(request.get(name), str) and Path(request[name]).is_absolute(),
                'Absolute request path required: ' + name)
        paths[name] = Path(request[name]).resolve()
    if 'original_plan' in request:
        require(isinstance(request['original_plan'], str) and Path(request['original_plan']).is_absolute(),
                'Absolute original pilot plan required')
        paths['original_plan'] = Path(request['original_plan']).resolve()
        value = request.get('original_plan_sha256')
        require(isinstance(value, str) and len(value) == 64
                and all(c in '0123456789abcdef' for c in value), 'Original plan SHA256 required')
    else:
        require('original_plan_sha256' not in request, 'Original pin without a plan')
    for key in ('view_plan_sha256', 'schedule_sha256'):
        value = request.get(key)
        require(isinstance(value, str) and len(value) == 64
                and all(c in '0123456789abcdef' for c in value), 'Explicit SHA256 required')
    require(type(request.get('rounds')) is int and 1 <= request['rounds'] <= 100
            and type(request.get('seed_base')) is int and 0 <= request['seed_base'] < 2**32
            and type(request.get('max_targets')) is int and 1 <= request['max_targets'] <= 12,
            'Explicit bounded scale settings required')
    for name in ('output', 'scale_output'):
        target = paths[name]
        require(target != Path(target.anchor) and not target.exists(), 'Fresh bounded output required')
        campaign.check_queue_destination(target, paths['prior_campaign'], paths['controls'],
                                         paths['storage_qualification'])
        for other in ('prior_campaign', 'controls', 'output', 'scale_output'):
            if other != name:
                source = paths[other]
                require(not target.is_relative_to(source) and not source.is_relative_to(target),
                        'Output overlaps another campaign root')
        for other in ('view_plan', 'storage_qualification', 'schedule', 'isaac_python'):
            require(not paths[other].is_relative_to(target), 'Output contains an immutable input')
        if 'original_plan' in paths:
            require(not paths['original_plan'].is_relative_to(target), 'Output contains original plan')
    return paths


def scale_command(request, cpu_python):
    result = [str(cpu_python), '-m', 'sim_data.native_capture_v4.campaign']
    mapping = {'schedule': 'schedule', 'schedule_sha256': 'schedule-sha256',
        'prior_campaign': 'prior-campaign', 'controls': 'after-controls',
        'storage_qualification': 'storage-qualification', 'scale_output': 'output',
        'isaac_python': 'isaac-python', 'native_deps': 'native-deps', 'rounds': 'rounds',
        'seed_base': 'seed-base', 'max_targets': 'max-targets'}
    for key, flag in mapping.items():
        result += ['--' + flag, str(request[key])]
    return result


def worker_environments(native_deps, inherited=None):
    root = Path(__file__).resolve().parents[4]
    projects = [str(root/'examples'), str(root/'examples/greenhouse_sim')]
    cpu = dict(os.environ if inherited is None else inherited)
    cpu['OPENBLAS_NUM_THREADS'] = '1'
    cpu['PYTHONPATH'] = os.pathsep.join(projects)
    native = dict(cpu)
    native['PYTHONPATH'] = os.pathsep.join([str(Path(native_deps).resolve()), *projects])
    return cpu, native


def run_scale_in_process(request, log_path, bindings, announce):
    """Reuse this CPU supervisor; V4 still owns each native worker and guard.

    No second waiting Python coordinator should be mistaken for a renderer.
    The V4 arguments, output, native subprocess ownership and failures are kept.
    """
    verify_bindings(bindings)
    announce(os.getpid())
    with Path(log_path).open('x', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log):
            try:
                campaign.main(scale_command(request, sys.executable)[3:])
                code = 0
            except SystemExit as exc:
                code = 0 if exc.code is None else exc.code if type(exc.code) is int else 1
    verify_bindings(bindings)
    return code


def original_completion(capture, plan_path, plan_pin):
    capture = Path(capture)
    request = read_json(capture/'request.json')
    result = read_json(capture/'result.json')
    require(not (capture/'failure.json').exists(), 'Failed original pilot')
    require(Path(request['plan_path']).resolve() == Path(plan_path).resolve()
            and request['plan_sha256'] == result['plan_sha256'] == plan_pin
            and result['request_sha256'] == sha256(capture/'request.json'),
            'Original capture does not match submitted queue plan')
    return dict(request_sha256=sha256(capture/'request.json'), result_sha256=sha256(capture/'result.json'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--request-sha256', required=True)
    args = parser.parse_args(argv)
    require(os.name == 'nt', 'Explicit Windows coordinator')
    request_path = args.request.resolve()
    request = probe.read_pinned(request_path, args.request_sha256)
    paths = validate_request(request)
    require(paths['isaac_python'].is_file() and paths['native_deps'].is_dir(), 'Missing native runtime')
    require(not request_path.is_relative_to(paths['output'])
            and not request_path.is_relative_to(paths['scale_output']), 'Input request overlaps output')
    plan = probe.read_pinned(paths['view_plan'], request['view_plan_sha256'])
    probe.check(plan)
    probe._destination(paths['output']/'capture', plan)
    require(plan['original_v4']['schedule_path'] == str(paths['schedule'])
            and plan['original_v4']['schedule_sha256'] == request['schedule_sha256'],
            'Probe/scale schedule mismatch')
    bindings = {str(request_path): args.request_sha256,
        str(paths['view_plan']): request['view_plan_sha256'],
        str(Path(__file__).resolve()): sha256(__file__),
        str(Path(campaign.__file__).resolve()): sha256(campaign.__file__),
        **probe.implementation_bindings(), **native_process_guard.implementation_bindings()}
    if 'original_plan' in paths:
        from ..native_original_capture import prepare as original_prepare
        from ..native_original_capture.contracts import new_destination
        original = probe.read_pinned(paths['original_plan'], request['original_plan_sha256'])
        original_prepare.check_plan(original)
        require(original['sample_count_limit'] == len(original['cases']) == 1,
                'Initial original qualification is exactly one case')
        new_destination(paths['output']/'original_capture',
                        [*original_prepare.protected_roots(original), paths['original_plan'].parent])
        bindings[str(paths['original_plan'])] = request['original_plan_sha256']
        bindings.update(original['implementation_bindings'])
    output = paths['output']
    output.mkdir(parents=True)
    write_json(output/'request.json', dict(request, implementation_bindings=bindings, training_approved=False))
    counter = 0
    def status(state, **extra):
        nonlocal counter
        counter += 1
        event = output/f'queue_{counter:06d}.json'
        write_json(event, dict(state=state,
            updated_utc=datetime.now(timezone.utc).isoformat(), bindings=bindings,
            training_approved=False, source_cap_reset=False, **extra))
        return event
    for key in ('prior_campaign', 'controls'):
        while not (paths[key]/'result.json').exists():
            reports = sorted(paths[key].glob('progress_*.json' if key == 'prior_campaign' else 'queue_*.json'))
            if reports:
                previous = read_json(reports[-1]).get('state', '')
                require(not previous.startswith(('held_', 'failed_', 'stopped_', 'error')),
                        'Preceding worker requires diagnosis')
            verify_bindings(bindings)
            status('waiting_for_' + key)
            time.sleep(30)
    bindings.update(campaign.completed_controls(paths['controls']))
    bindings.update(qualify_storage(paths['storage_qualification']))
    bindings[str(paths['prior_campaign']/'result.json')] = sha256(paths['prior_campaign']/'result.json')
    from sim_physics.host_memory import preflight
    def reserve():
        while True:
            memory = preflight()
            free = shutil.disk_usage(output).free
            active = campaign.native_processes()
            if memory['allowed'] and memory['commit_headroom_bytes'] >= 20*2**30 and free >= 60*2**30 and not active:
                return
            status('waiting_for_safe_native_resources', memory=memory, disk_free_bytes=free, active_native=active)
            time.sleep(30)
    def launch_check():
        probe.check(probe.read_pinned(paths['view_plan'], request['view_plan_sha256']))
        require(not campaign.native_processes(), 'Native process appeared before probe launch')
    environment, native_environment = worker_environments(paths['native_deps'])
    code = campaign.run_checked([str(paths['isaac_python']), '-m', 'sim_data.native_capture_v5.bounded_views',
        'capture', '--plan', str(paths['view_plan']), '--plan-sha256', request['view_plan_sha256'],
        '--output', str(output/'capture')], output/'native.log', bindings=bindings,
        environment=native_environment, native=True, reserve=reserve, launch_check=launch_check,
        announce=lambda pid: status('probe_running', worker_pid=pid))
    require(code == 0 and (output/'capture/result.json').is_file()
            and not (output/'capture/failure.json').exists(), 'Native probe did not complete')
    from ..native_dataset.audit import audit_capture
    audit = audit_capture(output/'capture', paths['view_plan'])
    write_json(output/'automatic_audit.json', audit)
    result = read_json(output/'capture/result.json')
    summary = probe.summarize(plan, result)
    write_json(output/'probe_summary.json', summary)
    verify_bindings(bindings)
    status('probe_complete_pending_global_comparison', result_sha256=sha256(output/'capture/result.json'),
        audit_sha256=sha256(output/'automatic_audit.json'), summary_sha256=sha256(output/'probe_summary.json'))
    if 'original_plan' in paths:
        def original_check():
            original_prepare.check_plan(probe.read_pinned(paths['original_plan'], request['original_plan_sha256']))
            require(not campaign.native_processes(), 'Native process appeared before original pilot')
        original_command = [str(paths['isaac_python']), '-m', 'sim_data.native_original_capture.collector',
            '--plan', str(paths['original_plan']), '--plan-sha256', request['original_plan_sha256'],
            '--output', str(output/'original_capture')]
        original_launch = {}
        def original_started(pid):
            event = status('original_pilot_running', worker_pid=pid, command=original_command)
            original_launch.update(worker_pid=pid, command=original_command,
                                   event_path=str(event), event_sha256=sha256(event))
        code = campaign.run_checked(original_command, output/'original_native.log', bindings=bindings,
            environment=native_environment, native=True, reserve=reserve, launch_check=original_check,
            announce=original_started)
        require(code == 0 and (output/'original_capture/result.json').is_file()
                and not (output/'original_capture/failure.json').exists(), 'Original pilot did not complete')
        from ..native_original_capture.audit import audit_capture as original_audit
        completion = original_completion(output/'original_capture', paths['original_plan'],
                                         request['original_plan_sha256'])
        original_result_pin = completion['result_sha256']
        original_review = original_audit(output/'original_capture', result_sha256=original_result_pin)
        require(original_completion(output/'original_capture', paths['original_plan'],
                                    request['original_plan_sha256']) == completion,
                'Original handoff changed during post-exit audit')
        require(sha256(original_launch['event_path']) == original_launch['event_sha256'],
                'Owned original launch event changed')
        write_json(output/'original_postexit_audit.json', original_review)
        write_json(output/'original_launcher_receipt.json', dict(state='owned_original_worker_exited_and_reaudited',
            exit_code=code, plan_path=str(paths['original_plan']), plan_sha256=request['original_plan_sha256'],
            capture=str(output/'original_capture'), **completion, owned_worker=original_launch,
            audit_path=str(output/'original_postexit_audit.json'), audit_sha256=sha256(output/'original_postexit_audit.json'),
            source_bindings=bindings, training_approved=False, source_cap_reset=False))
        verify_bindings(bindings)
        status('original_pilot_complete_pending_visual_review', result_sha256=original_result_pin)
    # The production continuation keeps its existing six-view profile regardless
    # of the diagnostic's result. Any promotion needs separate measured review.
    code = run_scale_in_process(request, output/'scale.log', bindings,
        announce=lambda pid: status('scale_coordinator_running_in_same_cpu_process', worker_pid=pid))
    require(code == 0 and (paths['scale_output']/'result.json').is_file(), 'Scale continuation did not complete')
    verify_bindings(bindings)
    write_json(output/'result.json', dict(state='serial_probe_and_scale_complete_pending_admission',
        request_sha256=args.request_sha256, scale_result_sha256=sha256(paths['scale_output']/'result.json'),
        training_approved=False, source_cap_reset=False))


if __name__ == '__main__':
    main()
