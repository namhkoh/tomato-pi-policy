"""Read-only actual storage/pagefile environment for explicitly authorized D/C outputs.

This report makes no admission decision and changes no disk, RAM or GPU floor.
"""
from pathlib import Path
import hashlib,json,os,shutil,subprocess
from . import native848_data_roots_v1 as data_roots
SCHEMA='greenhouse.native848_actual_storage_environment.v1'
DATA_ROOTS_SHA='b3b5d6c77385b2776a6997dd7dbe14f1aae8e1d093a45140d0f2c033fc2c8d83'
ROOT_MARKER=data_roots.FUTURE_DATA/'dataset_root.json'
ROOT_MARKER_SHA='65c9d79a0993d698524ed2ee61985b276d7fd1838a4258786d9836d1b4881716'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def source_bindings():
    helper=Path(data_roots.__file__).resolve()
    if digest(helper)!=DATA_ROOTS_SHA or digest(ROOT_MARKER)!=ROOT_MARKER_SHA:
        raise ValueError('Explicit data roots or original C-root marker changed')
    return {str(Path(__file__).resolve()):digest(__file__),**data_roots.bindings()}
def storage_environment(output):
    output=data_roots.diagnostic(output);bindings=source_bindings();system_drive=os.environ.get('SystemDrive','').upper()
    if len(system_drive)!=2 or system_drive[1]!=':':raise ValueError('Actual Windows system drive unavailable')
    command=('[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); '
        'ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_PageFileUsage -ErrorAction Stop '
        '| Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage)')
    raw=subprocess.check_output(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],
        text=True,encoding='utf-8',errors='strict',timeout=20,creationflags=subprocess.CREATE_NO_WINDOW)
    pagefiles=json.loads(raw.lstrip('\ufeff'))
    if not isinstance(pagefiles,list):raise ValueError('Actual pagefile inventory must be an array')
    drives=[]
    for row in pagefiles:
        p=Path(row['Name'])
        if not p.is_absolute() or not p.drive:raise ValueError('Pagefile path is not absolute')
        if any(type(row[k]) is not int or row[k]<0 for k in ('AllocatedBaseSize','CurrentUsage','PeakUsage')):
            raise ValueError('Invalid actual pagefile counters')
        drives.append(p.drive.upper())
    measured=output
    while not measured.exists():
        parent=measured.parent
        if parent==measured:raise ValueError('No existing capture-volume ancestor')
        measured=parent
    usage=shutil.disk_usage(measured);capture_drive=output.drive.upper()
    return dict(schema=SCHEMA,output_path=str(output),system_drive=system_drive,
        capture_drive=capture_drive,pagefiles=pagefiles,pagefile_drives=sorted(set(drives)),
        capture_shares_system_volume=capture_drive==system_drive,
        capture_shares_any_pagefile_volume=capture_drive in drives,
        disk_usage_measured_path=str(measured),disk_total_bytes=usage.total,
        disk_used_bytes=usage.used,disk_free_bytes=usage.free,
        actual_environment_read_only=True,no_pagefile_or_volume_setting_changed=True,
        disk_memory_GPU_floors_changed=False,resource_admission_separately_required=True,
        source_bindings=bindings)
