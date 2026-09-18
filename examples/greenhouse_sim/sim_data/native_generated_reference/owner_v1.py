"""Sole CPU owner for one explicitly requested two-frame native probe.

No waiting queue, retries, process killing, or automatic next campaign. Holds the
same future-owner OS mutex as serial_phases, without importing its active code.
On interruption, retain ownership until the one child exits; never orphan a Kit.
"""
from contextlib import contextmanager
from pathlib import Path
import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

from . import execution_v1 as ex
from ..native_original_capture import contracts as oc

LOCK_NAME = r'Global\greenhouse.native_serial_phases.owner.v1'


@contextmanager
def owner_lock():
    oc.require(os.name=='nt','Windows owner required')
    from ctypes import wintypes
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateMutexW.argtypes=(ctypes.c_void_p,wintypes.BOOL,wintypes.LPCWSTR)
    kernel.CreateMutexW.restype=wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes=(wintypes.HANDLE,wintypes.DWORD)
    kernel.WaitForSingleObject.restype=wintypes.DWORD
    kernel.ReleaseMutex.argtypes=kernel.CloseHandle.argtypes=(wintypes.HANDLE,)
    handle=kernel.CreateMutexW(None,False,LOCK_NAME)
    oc.require(bool(handle),'Cannot open owner mutex');acquired=False
    try:
        status=kernel.WaitForSingleObject(handle,0);acquired=status in (0,0x80)
        oc.require(status==0,'Owner busy, abandoned, or unavailable; no recovery/launch')
        yield
    finally:
        if acquired: kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def resources(output_parent):
    from sim_physics.host_memory import preflight
    from ..native_dataset.native_process_guard import process_inventory
    from ..native_dataset.reference_bridge_owner_v1 import current_owner_evidence
    owner=current_owner_evidence()
    memory=preflight();processes=process_inventory();free=shutil.disk_usage(output_parent).free
    oc.require(memory['checked'] is True and memory['allowed'] is True
        and memory['commit_headroom_bytes']>=20*2**30 and free>=60*2**30
        and processes['classifications'] and not processes['blockers'] and processes['no_blockers_observed'] is True,
        'Resource or process admission failed; no override/wait/retry')
    return dict(memory=memory,disk_free_bytes=free,process_inventory=processes,raw_owner_classification=owner,
                exclusive_launch_guaranteed=False)


def _environment(request):
    env=os.environ.copy()
    for name in ('PYTHONEXE','PYTHONHOME'):
        env.pop(name,None)
    examples=Path(__file__).resolve().parents[3]
    env['PYTHONPATH']=os.pathsep.join([request['native_deps'],str(examples),str(examples/'greenhouse_sim')])
    env['OPENBLAS_NUM_THREADS']='1';env['PYTHONDONTWRITEBYTECODE']='1'
    return env


def _run_child(command, output, environment):
    with (output/'native.log').open('x',encoding='utf-8') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment,
                               creationflags=subprocess.CREATE_NO_WINDOW)
        try:
            oc.write_new(output/'owned_worker.json',dict(command=command,launcher_pid=child.pid,
                owner_pid=os.getpid(),pid_kind='Popen_bootstrap_not_claimed_native_Kit_pid',**ex.FLAGS))
            return child.wait()
        except BaseException:
            # Do not release the mutex while our bootstrap/worker is still alive.
            # No worker interruption or automatic continuation is implemented.
            while child.poll() is None:
                try: child.wait()
                except KeyboardInterrupt: continue
            raise


def complete(output, request_path, request_sha256, exit_code):
    """Post-exit boundary: nonzero exit can never acquire an audited receipt."""
    from .audit_v1 import audit_capture
    oc.require(type(exit_code) is int and exit_code==0,'Native worker did not exit cleanly')
    request,_=ex.preflight_request(request_path,request_sha256)
    output=Path(output).resolve();oc.require(str(output)==request['output'],'Wrong owned output')
    oc.require(not (output/'owner_failure.json').exists(),'Failed owner cannot publish completion')
    worker=oc.read_json(output/'owned_worker.json')
    exited=oc.read_json(output/'exit.json')
    oc.require(worker['command']==ex.worker_command(request_path,request_sha256,request)
        and type(exited['exit_code']) is int and exited['exit_code']==0
        and exited['owned_worker_sha256']==oc.sha256(output/'owned_worker.json'),'Owned command/exit changed')
    capture=output/'capture';result_pin=oc.sha256(capture/'result.json')
    audit_started=time.perf_counter()
    audit=audit_capture(capture,result_sha256=result_pin)
    postexit_replay_seconds=time.perf_counter()-audit_started
    oc.write_new(output/'postexit_audit.json',audit)
    receipt=dict(schema=ex.RECEIPT,state='owned_exit0_and_independent_saved_native_replay',exit_code=0,
        execution_request_path=str(Path(request_path).resolve()),execution_request_sha256=request_sha256,
        capture=str(capture),result_sha256=result_pin,request_sha256=oc.sha256(capture/'request.json'),
        owned_worker_sha256=oc.sha256(output/'owned_worker.json'),exit_sha256=oc.sha256(output/'exit.json'),
        owner_started_sha256=oc.sha256(output/'owner_started.json'),postexit_replay_seconds=postexit_replay_seconds,
        postexit_audit_sha256=oc.sha256(output/'postexit_audit.json'),
        source_bindings=request['source_bindings'],implementation_bindings=ex.implementation_bindings(),
        process_history_independently_attested=False,**ex.FLAGS)
    oc.write_new(output/'launcher_receipt.json',receipt)
    return receipt


def run(request_path, request_sha256):
    oc.require(os.name=='nt' and not any(n.split('.')[0] in ('isaacsim','omni','carb') for n in sys.modules),
               'CPU-only Windows owner required')
    from ..native_dataset.reference_bridge_owner_v1 import current_owner_evidence
    current_owner_evidence()
    request_path=oc.pin(request_path,request_sha256)
    request,plan=ex.preflight_request(request_path,request_sha256)
    oc.require(not Path(sys.executable).resolve().is_relative_to(Path(request['isaac_python']).parent),
               'Owner must use a separate CPU interpreter, never the native bootstrap')
    from .prepare import check_plan
    replay_started=time.perf_counter();check_plan(plan)
    owner_full_plan_replay_seconds=time.perf_counter()-replay_started
    output=oc.new_destination(request['output'],[request_path])
    oc.require(output.parent.is_dir(),'Existing output parent required')
    with owner_lock():
        admission=resources(output.parent)
        ex.preflight_request(request_path,request_sha256)
        output.mkdir()
        oc.write_new(output/'owner_started.json',dict(execution_request_path=str(request_path),
            execution_request_sha256=request_sha256,owner_pid=os.getpid(),resources=admission,
            owner_full_plan_replay_seconds=owner_full_plan_replay_seconds,**ex.FLAGS))
        try:
            command=ex.worker_command(request_path,request_sha256,request)
            native_started=time.perf_counter();code=_run_child(command,output,_environment(request))
            oc.write_new(output/'exit.json',dict(exit_code=code,native_process_wall_seconds=time.perf_counter()-native_started,
                owned_worker_sha256=oc.sha256(output/'owned_worker.json'),**ex.FLAGS))
            oc.require(code==0,'Native failure; no replay/continuation')
            from ..native_dataset.native_process_guard import native_processes
            current_owner_evidence()
            oc.require(not native_processes(),'Native processes remain after owned exit; halt')
            return complete(output,request_path,request_sha256,code)
        except BaseException:
            oc.write_new(output/'owner_failure.json',dict(state='owner_failed_no_continuation',traceback=traceback.format_exc(),**ex.FLAGS))
            raise


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='action',required=True)
    build=sub.add_parser('prepare-request')
    for key in ('plan','plan-sha256','isaac-python','native-deps','output','request-output'):
        build.add_argument('--'+key,required=True)
    go=sub.add_parser('run');go.add_argument('--request',required=True);go.add_argument('--request-sha256',required=True)
    a=p.parse_args(argv)
    if a.action=='prepare-request':
        request=ex.make_request(a.plan,plan_sha256=a.plan_sha256,isaac_python=a.isaac_python,native_deps=a.native_deps,output=a.output)
        dest=oc.new_destination(a.request_output,[a.output,Path(a.plan).parent,*request['source_bindings']])
        oc.require(dest.parent.is_dir(),'Existing request parent required')
        oc.write_new(dest,request)
        value=dict(request_path=str(dest),request_sha256=oc.sha256(dest),native_launched=False)
    else: value=run(a.request,a.request_sha256)
    print(json.dumps(value,indent=2))


if __name__=='__main__':
    main()
