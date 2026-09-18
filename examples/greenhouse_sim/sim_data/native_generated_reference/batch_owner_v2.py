"""CPU owner of one bounded native batch; no retries or competing renderer.

Uses the existing shared OS mutex, resource gate and exact owned-child lifetime.
Only an owned exit zero followed by independent native replay yields a receipt.
"""
from pathlib import Path
import os
import sys
import time
import traceback

from . import batch_execution_v2 as ex
from .owner_v1 import owner_lock, resources, _environment, _run_child
from ..native_original_capture import contracts as oc


def complete(output, request_path, request_sha256, exit_code):
    from .batch_audit_v2 import audit_capture
    oc.require(type(exit_code) is int and exit_code == 0, 'Native worker did not exit cleanly')
    request, _ = ex.preflight_request(request_path, request_sha256)
    output = Path(output).resolve()
    oc.require(str(output) == request['output'] and not (output/'owner_failure.json').exists(), 'Wrong/failed owned output')
    worker, exited = oc.read_json(output/'owned_worker.json'), oc.read_json(output/'exit.json')
    oc.require(worker['command'] == ex.worker_command(request_path, request_sha256, request)
        and type(exited['exit_code']) is int and exited['exit_code'] == 0
        and exited['owned_worker_sha256'] == oc.sha256(output/'owned_worker.json'), 'Owned command/exit changed')
    capture = output/'capture'; result_pin = oc.sha256(capture/'result.json')
    audit_started = time.perf_counter()
    audit = audit_capture(capture, result_sha256=result_pin)
    oc.write_new(output/'postexit_audit.json', audit)
    receipt = dict(schema=ex.RECEIPT, state='owned_exit0_and_independent_saved_native_replay', exit_code=0,
        execution_request_path=str(Path(request_path).resolve()), execution_request_sha256=request_sha256,
        capture=str(capture), result_sha256=result_pin, request_sha256=oc.sha256(capture/'request.json'),
        owned_worker_sha256=oc.sha256(output/'owned_worker.json'), exit_sha256=oc.sha256(output/'exit.json'),
        owner_started_sha256=oc.sha256(output/'owner_started.json'), postexit_replay_seconds=time.perf_counter()-audit_started,
        postexit_audit_sha256=oc.sha256(output/'postexit_audit.json'), source_bindings=request['source_bindings'],
        implementation_bindings=ex.implementation_bindings(), process_history_independently_attested=False, **ex.FLAGS)
    oc.write_new(output/'launcher_receipt.json', receipt)
    return receipt


def run(request_path, request_sha256):
    oc.require(os.name == 'nt' and not any(n.split('.')[0] in ('isaacsim', 'omni', 'carb') for n in sys.modules),
        'CPU-only Windows owner required')
    from ..native_dataset.reference_bridge_owner_v1 import current_owner_evidence
    current_owner_evidence()
    request_path = oc.pin(request_path, request_sha256)
    request, plan = ex.preflight_request(request_path, request_sha256)
    oc.require(not Path(sys.executable).resolve().is_relative_to(Path(request['isaac_python']).parent),
        'Owner requires a separate CPU interpreter')
    from .batch_prepare_v2 import check_plan
    replay_started = time.perf_counter()
    check_plan(plan); ex.checked_storage(request)
    replay_seconds = time.perf_counter()-replay_started
    output = oc.new_destination(request['output'], [request_path])
    oc.require(output.parent.is_dir(), 'Existing output parent required')
    with owner_lock():
        admission = resources(output.parent)
        ex.preflight_request(request_path, request_sha256)
        output.mkdir()
        oc.write_new(output/'owner_started.json', dict(execution_request_path=str(request_path),
            execution_request_sha256=request_sha256, owner_pid=os.getpid(), resources=admission,
            owner_full_plan_and_storage_replay_seconds=replay_seconds, **ex.FLAGS))
        try:
            command = ex.worker_command(request_path, request_sha256, request)
            native_started = time.perf_counter()
            code = _run_child(command, output, _environment(request))
            oc.write_new(output/'exit.json', dict(exit_code=code, native_process_wall_seconds=time.perf_counter()-native_started,
                owned_worker_sha256=oc.sha256(output/'owned_worker.json'), **ex.FLAGS))
            oc.require(code == 0, 'Native failure; no replay/continuation')
            from ..native_dataset.native_process_guard import native_processes
            current_owner_evidence()
            oc.require(not native_processes(), 'Native processes remain after owned exit; halt')
            return complete(output, request_path, request_sha256, code)
        except BaseException:
            oc.write_new(output/'owner_failure.json', dict(state='owner_failed_no_continuation', traceback=traceback.format_exc(), **ex.FLAGS))
            raise
