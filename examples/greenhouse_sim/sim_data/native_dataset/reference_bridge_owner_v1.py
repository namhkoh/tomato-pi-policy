"""Neutral, explicitly checked CPU entrypoint; same-process frozen owner reuse.

No process exemption. The actual CIM row must pass the UNCHANGED raw classifier
without process_inventory's current-PID exception. No native launch in check-only.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

from . import native_process_guard as guard
from ..native_original_capture import contracts as oc


def verify_owner_row(row, *, pid, executable):
    oc.require(row['ProcessId']==pid and Path(row['ExecutablePath']).resolve()==Path(executable).resolve(),
               'Actual CPU owner PID/executable mismatch')
    decision=guard.classify_process(row,native_roots=guard.DEFAULT_NATIVE_ROOTS)
    oc.require(decision['blocking'] is False and decision['classification']=='unrelated_python',
               'Actual owner command blocks unchanged raw guard; no exemption is permitted')
    return dict(metadata=row,classification=decision,current_pid_exception_used=False,
                supervisor_exception_used=False,exclusive_launch_guaranteed=False)


def current_owner_evidence():
    oc.require(os.name=='nt','Windows CPU owner evidence required')
    oc.require(not any(n.split('.')[0] in ('isaacsim','omni','carb') for n in sys.modules),
               'CPU owner must not contain a loaded native runtime')
    guard.implementation_bindings()
    command=('[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); '
        'ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_Process '
        f'-Filter "ProcessId = {os.getpid()}" -ErrorAction Stop | '
        'Select-Object ProcessId,ParentProcessId,CreationDate,Name,ExecutablePath,CommandLine)')
    raw=subprocess.check_output(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],
        text=True,encoding='utf-8',errors='strict',timeout=20,creationflags=subprocess.CREATE_NO_WINDOW)
    rows=json.loads(raw.lstrip('\ufeff'))
    oc.require(isinstance(rows,list) and len(rows)==1,'Missing/ambiguous actual owner process')
    return verify_owner_row(rows[0],pid=os.getpid(),executable=sys.executable)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',required=True);p.add_argument('--request-sha256',required=True)
    p.add_argument('--check-only',action='store_true',help='Read actual raw classification only; no request validation or native launch')
    a=p.parse_args(argv)
    evidence=current_owner_evidence()
    if a.check_only:
        print(json.dumps(dict(state='actual_owner_raw_classification_passed',evidence=evidence,
            request_validated=False,native_launched=False),indent=2));return
    from ..native_generated_reference.owner_v1 import run
    print(json.dumps(run(a.request,a.request_sha256),indent=2))


if __name__=='__main__':
    main()
