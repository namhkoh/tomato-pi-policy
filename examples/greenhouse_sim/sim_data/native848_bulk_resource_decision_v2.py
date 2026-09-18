"""Explicit future-wave disk policy; frozen memory/GPU/identity gates retained.

No existing file, process, live policy, or native rendering setting is changed.
The old60GiB assessment is retained beside each newly computed decision.
"""
from pathlib import Path
import json
import os
import subprocess
from . import native848_bulk_sibling_gate_v5 as gate
from .dataset_review import require, verify_bindings
from .depth_preview import sha256

ROOT=Path(__file__).resolve().parents[3]
GIB=2**30
SCHEMA='greenhouse.native848_bounded_future_disk_policy.v1'
ASSESSMENT=ROOT/'data/sim_data/diagnostics/native848_future_disk_floor_readonly_assessment_20260917_v1/result.json'
ASSESSMENT_SHA='4975ac46a56fc3976388bcd5d8315cfbf7f428caf24894930c5b05cb2d7ef166'


def check_policy(policy):
    require(policy['schema']==SCHEMA and policy['runtime_disk_floor_bytes']==20*GIB
        and policy['initial_wave_disk_floor_bytes']==36*GIB
        and policy['additional_worker_disk_floor_bytes']==24*GIB
        and policy['maximum_workers']==4 and policy['frames_per_worker']==1024
        and type(policy['maximum_queue_frames']) is int and 0<policy['maximum_queue_frames']<=20480
        and policy['requires_exact_root_launch_review'] is True
        and policy['native_quality_checks_changed'] is False
        and policy['training_approved'] is False,'Exact bounded future disk policy required')
    require(policy['source_bindings'].get(str(ASSESSMENT.resolve()))==ASSESSMENT_SHA,
        'Measured storage/pagefile assessment must remain bound')
    verify_bindings(policy['source_bindings'])
    return policy


def storage_environment(output):
    output=Path(output).resolve()
    require(output.is_relative_to(ROOT/'data/sim_data/diagnostics') and output.drive.upper()=='D:',
        'Only measured D workspace volume qualified')
    system_drive=os.environ.get('SystemDrive','').upper()
    require(system_drive=='C:','System-volume placement changed')
    command=('[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); '
        'ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_PageFileUsage -ErrorAction Stop '
        '| Select-Object Name,AllocatedBaseSize,CurrentUsage,PeakUsage)')
    raw=subprocess.check_output(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],
        text=True,encoding='utf-8',errors='strict',timeout=20,creationflags=subprocess.CREATE_NO_WINDOW)
    pagefiles=json.loads(raw.lstrip('\ufeff'))
    require(isinstance(pagefiles,list) and pagefiles
        and all(Path(x['Name']).drive.upper()=='C:' for x in pagefiles),
        'Current pagefiles must remain on measured C volume')
    return dict(system_drive=system_drive,capture_drive='D:',pagefiles=pagefiles,
        actual_environment_read_only=True,no_pagefile_or_volume_setting_changed=True)


def actual_safe(evidence):
    memory=evidence['memory'];gpu=evidence['gpu'];inventory=evidence['process_inventory']
    return bool(memory['allowed'] and memory['commit_headroom_bytes']>=20*GIB
        and evidence['disk_free_bytes']>=20*GIB and gpu['free_mib']>=4096
        and not inventory['blockers'] and inventory['no_blockers_observed'] is True)


def projected_safe(evidence):
    require(type(evidence['projected_additional_vram_mib']) is int
        and evidence['projected_additional_vram_mib']>=10240
        and evidence['required_remaining_vram_mib']==4096,'Unchanged conservative GPU reserve required')
    require(evidence['disk_policy_phase'] in ('initial_wave','additional_worker'),'Unknown resource phase')
    floor=36*GIB if evidence['disk_policy_phase']=='initial_wave' else 24*GIB
    require(evidence['disk_admission_floor_bytes']==floor
        and evidence['runtime_disk_floor_bytes']==20*GIB,'Changed explicit disk thresholds')
    return bool(evidence['gpu']['free_mib']-evidence['projected_additional_vram_mib']>=4096
        and evidence['disk_free_bytes']>=floor)


def assess(legacy,*,phase,policy_evidence):
    require(phase in ('initial_wave','additional_worker'),'Explicit resource phase required')
    value=dict(legacy,legacy_60GiB_allowed=legacy['allowed'],disk_policy=policy_evidence,
        disk_policy_phase=phase,runtime_disk_floor_bytes=20*GIB,
        disk_admission_floor_bytes=(36 if phase=='initial_wave' else 24)*GIB,
        GPU_memory_identity_policy_changed=False)
    projected=projected_safe(value)
    value['allowed']=actual_safe(value) and projected
    return value


def resource_evidence(output,allowed,policy_item,*,phase='additional_worker'):
    path=Path(policy_item['path']).resolve()
    require(path.is_relative_to(ROOT) and sha256(path)==policy_item['sha256'],'Changed disk policy')
    check_policy(json.loads(path.read_text(encoding='utf-8-sig')))
    legacy=gate.resource_evidence(output,allowed)
    return assess(legacy,phase=phase,policy_evidence=policy_item)


def decide(evidence):
    projected=projected_safe(evidence);actual=actual_safe(evidence)
    require(evidence['allowed']==bool(actual and projected),'Resource decision inconsistent with measured counters')
    if not actual:return 'unsafe_current_state'
    if not projected:return 'defer_additional_worker'
    return 'admit_additional_worker'


def runtime_safe(measured):
    return bool(measured['memory']['allowed'] and measured['memory']['commit_headroom_bytes']>=20*GIB
        and measured['gpu']['free_mib']>=4096 and measured['disk_free_bytes']>=20*GIB)
