"""Exact owned-process exceptions for one bounded two-worker coordinator.

No generic Kit, Python, executable-path or ancestor exemption. Every admitted
process has a pinned PID/creation/executable/command/parent tuple from the owned
launch handshake. Existing single-worker guard implementations stay unchanged.
"""
from pathlib import Path
import ast
import ctypes
import json
import os
import shutil
import subprocess
import sys
from .dataset_review import require, verify_bindings
from .native_dataset import native_process_guard as guard
from sim_physics.host_memory import preflight

IDENTITY_KEYS=('ProcessId','ParentProcessId','CreationDate','Name','ExecutablePath','CommandLine')
KIT=Path('D:/isaac-sim-6.0.1/kit/python/kit.exe')


def snapshot():
    command=('[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); '
        'ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_Process -ErrorAction Stop '
        '| Select-Object ProcessId,ParentProcessId,CreationDate,Name,ExecutablePath,CommandLine)')
    raw=subprocess.check_output(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],
        text=True,encoding='utf-8',errors='strict',timeout=20,creationflags=subprocess.CREATE_NO_WINDOW)
    rows=json.loads(raw.lstrip('\ufeff'))
    require(isinstance(rows,list) and rows and len({r['ProcessId'] for r in rows})==len(rows),
        'Complete unique CIM snapshot required')
    return rows


def identity(row):
    value={k:row.get(k) for k in IDENTITY_KEYS}
    require(type(value['ProcessId']) is int and value['ProcessId']>0
        and type(value['ParentProcessId']) is int and value['ParentProcessId']>0
        and all(isinstance(value[k],str) and value[k] for k in IDENTITY_KEYS[2:]),
        'Exact PID, creation, executable, command and parent required')
    return value


def same_identity(row, expected):
    return all(row.get(k)==expected.get(k) for k in IDENTITY_KEYS)


def one(rows,pid):
    found=[r for r in rows if r['ProcessId']==pid]
    require(len(found)==1,'Exact owned process absent or ambiguous')
    return identity(found[0])


def command_argv(command):
    from ctypes import wintypes
    count=ctypes.c_int()
    shell=ctypes.WinDLL('shell32',use_last_error=True)
    shell.CommandLineToArgvW.argtypes=(wintypes.LPCWSTR,ctypes.POINTER(ctypes.c_int))
    shell.CommandLineToArgvW.restype=ctypes.POINTER(wintypes.LPWSTR)
    ptr=shell.CommandLineToArgvW(command,ctypes.byref(count))
    require(bool(ptr),'Cannot parse actual Windows command')
    try:return [ptr[i] for i in range(count.value)]
    finally:
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        kernel.LocalFree.argtypes=(ctypes.c_void_p,);kernel.LocalFree(ptr)


def command_child(row, coordinator, command):
    value=identity(row);argv=command_argv(value['CommandLine'])
    require(value['ParentProcessId']==coordinator['ProcessId'] and value['Name'].lower()=='cmd.exe'
        and Path(value['ExecutablePath']).resolve()==Path(os.environ['COMSPEC']).resolve(),
        'Command process must be exact owned cmd child')
    require(len(argv)>=3 and argv[1].lower()=='/c' and argv[2:]==command,
        'Owned cmd command differs from exact Popen request')
    return value


def native_child(row, command_identity, worker_args):
    value=identity(row);argv=command_argv(value['CommandLine'])
    require(value['ParentProcessId']==command_identity['ProcessId'] and value['Name'].lower()=='kit.exe'
        and Path(value['ExecutablePath']).resolve()==KIT.resolve()
        and argv[1:]==worker_args and Path(argv[0]).resolve()==KIT.resolve(),
        'Native child executable, parent or exact arguments differ')
    return value


def owned_allowlist(coordinator, slots):
    """Slots are immutable command+native receipts from no more than two launches."""
    require(len(slots)<=2 and len({s['slot'] for s in slots})==len(slots)
        and all(s['slot'] in (0,1) for s in slots),'Exactly bounded unique slots required')
    result=[identity(coordinator)]
    for slot in slots:
        command,native=identity(slot['command_identity']),identity(slot['native_identity'])
        require(command['ParentProcessId']==coordinator['ProcessId']
            and native['ParentProcessId']==command['ProcessId'],'Owned process parent chain differs')
        result.extend((command,native))
    require(len({r['ProcessId'] for r in result})==len(result),'Owned PID reuse or overlapping workers')
    return result


def classify_owned(rows, allowed, base_report, *, require_all=True):
    by={r['ProcessId']:r for r in rows};expected={r['ProcessId']:identity(r) for r in allowed}
    require(len(expected)==len(allowed),'Duplicate owned identity')
    for pid,value in expected.items():
        require(pid in by or not require_all,'Owned process disappeared before admission')
        if pid in by:require(same_identity(by[pid],value),'Owned process identity changed or PID reused')
    report=json.loads(json.dumps(base_report))
    require({r['ProcessId'] for r in report['classifications']}==set(by),'Classification snapshot differs')
    for decision in report['classifications']:
        if decision['ProcessId'] in expected:
            decision.update(classification='exact_owned_bounded_bulk_process',blocking=False,
                reason='exact_pid_creation_executable_command_parent',
                reasons=['exact_pid_creation_executable_command_parent'])
    report['blockers']=[r for r in report['classifications'] if r['blocking']]
    report.update(no_blockers_observed=not report['blockers'],owned_identities=list(expected.values()),
        generic_native_exemption=False,process_control_performed=False)
    return report


def inventory(allowed, *, rows=None):
    # Preserve the separately reviewed exact script-only MCP bridge recognition.
    import verified_socket_bridge_gate_v1 as bridge
    verify_bindings(bridge.BINDINGS)
    imports=[]
    for node in ast.walk(ast.parse(bridge.SCRIPT.read_bytes())):
        if isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
        elif isinstance(node,ast.ImportFrom):imports.append(node.module or '')
    require(not any(n.split('.')[0] in ('omni','isaacsim','carb','subprocess') for n in imports),
        'Reviewed socket bridge imports changed')
    rows=snapshot() if rows is None else rows
    proofs=[guard.bridge_proof(p,uv_executable=guard.DEFAULT_UV_EXECUTABLE) for p in guard.DEFAULT_BRIDGE_VENVS]
    report=guard.classify_processes(rows,bridge_proofs=proofs,native_roots=guard.DEFAULT_NATIVE_ROOTS)
    by={r['ProcessId']:r for r in rows}
    for pid,expected in bridge.IDENTITIES.items():
        row=by.get(pid)
        if row is None or row.get('Name')!='python.exe' or row.get('CommandLine')!=bridge.COMMAND:continue
        if not all(row.get(k)==v for k,v in expected.items()):continue
        if pid==61812:
            parent=by.get(72088,{})
            if parent.get('CommandLine')!=bridge.COMMAND or not all(parent.get(k)==v for k,v in bridge.IDENTITIES[72088].items()):continue
        decision=next(r for r in report['classifications'] if r['ProcessId']==pid)
        require(decision['classification']=='unknown_relevant_python','Bridge classification changed')
        decision.update(classification='exact_reviewed_script_socket_mcp_bridge',blocking=False,
            reason='exact_existing_reviewed_bridge',reasons=['exact_existing_reviewed_bridge'])
    result=classify_owned(rows,allowed,report)
    verify_bindings(bridge.BINDINGS)
    return result


def gpu_snapshot():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.total,memory.used',
        '--format=csv,noheader,nounits'],text=True,timeout=20,creationflags=subprocess.CREATE_NO_WINDOW)
    rows=[line.split(',') for line in raw.strip().splitlines()]
    require(len(rows)==1 and len(rows[0])==3,'Exactly one measured GPU required')
    index,total,used=(int(x.strip()) for x in rows[0])
    require(index==0 and 0<=used<=total,'Invalid GPU counters')
    return dict(index=index,total_mib=total,used_mib=used,free_mib=total-used)


def assess_resources(memory,disk_free,gpu,projected_additional_vram_mib):
    require(type(projected_additional_vram_mib) is int and projected_additional_vram_mib>=10240,
        'At least10GiB projected per-worker VRAM reserve required')
    return dict(allowed=bool(memory['allowed'] and memory['commit_headroom_bytes']>=20*2**30
        and disk_free>=60*2**30 and gpu['free_mib']-projected_additional_vram_mib>=4096),
        memory=memory,disk_free_bytes=disk_free,gpu=gpu,
        projected_additional_vram_mib=projected_additional_vram_mib,required_remaining_vram_mib=4096,
        projection_is_not_measured_peak=True)


def resource_evidence(output,allowed,projected_additional_vram_mib=10240):
    result=assess_resources(preflight(),shutil.disk_usage(output).free,gpu_snapshot(),projected_additional_vram_mib)
    result['process_inventory']=inventory(allowed)
    result['allowed']=result['allowed'] and not result['process_inventory']['blockers']
    return result
