"""Explicit reboot handoff, never a fabricated completed collection campaign.

The caller pins the user-requested stop and resumed authorization. Old completed
jobs remain observations; unfinished jobs remain unfinished. Live admission also
checks the new boot. Saved audit replay checks the recorded handoff without
requiring the auditor to run during that same Windows boot.
"""
from copy import deepcopy
from pathlib import Path
import json
import os
import re
import subprocess

from ..native_original_capture import contracts as oc

SCHEMA = 'greenhouse.explicit_reboot_resume_authorization.v1'
KIND = 'user_requested_reboot.v1'
STATE = 'verified_user_stop_and_new_boot_not_prior_campaign_completion'


def boot_epoch_ms():
    oc.require(os.name == 'nt', 'Windows boot evidence required')
    cmd = ('[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); '
           'Get-CimInstance Win32_OperatingSystem -ErrorAction Stop | '
           'Select-Object LastBootUpTime | ConvertTo-Json -Compress')
    raw = subprocess.check_output(['powershell.exe', '-NoProfile', '-NonInteractive', '-Command', cmd],
        text=True, encoding='utf-8', errors='strict', timeout=20,
        creationflags=subprocess.CREATE_NO_WINDOW)
    match = re.fullmatch(r'/Date\((\d+)\)/', json.loads(raw.lstrip('\ufeff'))['LastBootUpTime'])
    oc.require(match is not None, 'Unsupported Windows boot timestamp')
    return int(match.group(1))


def validate_predecessor(pred):
    oc.require(isinstance(pred, dict) and set(pred) == {'kind', 'request_path', 'request_sha256'}
        and pred['kind'] == KIND, 'Explicit reboot predecessor required')
    path = oc.pin(pred['request_path'], pred['request_sha256'])
    config = oc.read_json(path)
    oc.require(set(config) == {'schema', 'user_instruction', 'previous_boot_epoch_ms',
        'resumed_boot_epoch_ms', 'pause_path', 'pause_sha256', 'source_bindings'},
        'Exact reboot authorization fields required')
    oc.require(config['schema'] == SCHEMA and config['user_instruction'] == 'resume collection'
        and type(config['previous_boot_epoch_ms']) is int
        and type(config['resumed_boot_epoch_ms']) is int
        and 0 < config['previous_boot_epoch_ms'] < config['resumed_boot_epoch_ms'],
        'Explicit user resumption after a different boot required')
    oc.bind_all(config['source_bindings'])
    pause_path = oc.pin(config['pause_path'], config['pause_sha256'])
    oc.require(config['source_bindings'].get(str(pause_path)) == config['pause_sha256'], 'Unbound pause')
    pause = oc.read_json(pause_path)
    oc.require(pause['schema'] == 'greenhouse.user_requested_reboot_pause.v1'
        and pause['state'] == 'stopped_and_saved_waiting_for_user_restart'
        and pause['all_eight_exact_owners_exited'] is True
        and pause['completed_native_captures_preserved'] is True
        and pause['active_native_processes'] == [] and pause['goal_20000_complete'] is False,
        'Verified user-stop checkpoint required')
    docs = {}
    for role in ('source_checkpoint', 'stop_receipt', 'verification'):
        p = oc.pin(pause[role]['path'], pause[role]['sha256'])
        oc.require(config['source_bindings'].get(str(p)) == pause[role]['sha256'], 'Unbound stop source')
        docs[role] = oc.read_json(p)
    stop, checkpoint, verification = (docs[k] for k in ('stop_receipt', 'source_checkpoint', 'verification'))
    oc.require(stop['state'] == 'eight_exact_dataset_owners_stopped_for_user_reboot'
        and stop['captures_deleted'] is False and stop['prior_receipts_rewritten'] is False
        and stop['system_reboot_issued'] is False and stop['goal_20000_complete'] is False,
        'Wrong user-stop receipt')
    rows = stop['records']; ids = [r['pid'] for r in rows]
    oc.require(len(rows) == len(set(ids)) == 8 and all(
        r['state'] == 'stopped_on_user_request' and r['termination_called'] is True
        and r['observed_exit_code'] == 125 and r['clean_collection_completion_claimed'] is False for r in rows),
        'Each previous owner must retain its actual interrupted exit')
    oc.require(verification['remaining_owner_count'] == 0 and verification['active_native_processes'] == []
        and set(verification['stopped_owner_ids']) == set(ids), 'Stop verification differs')
    completed = checkpoint['completed_jobs']; count = checkpoint['completed_job_count']
    oc.require(checkpoint['schema'] == 'greenhouse.collection_restart_handoff.v1'
        and checkpoint['state'] == 'pre_restart_checkpoint_only_not_automatic_resume'
        and checkpoint['resolution'] == oc.RESOLUTION and checkpoint['goal_accepted_train'] == 20000
        and checkpoint['training_approved'] is False and 0 < count < 60
        and [r['index'] for r in completed] == list(range(1, count+1))
        and checkpoint['pending_wave_job_indices'] == list(range(count+1, 61))
        and pause['completed_current_wave_jobs'] == count
        and pause['unfinished_wave_jobs'] == checkpoint['pending_wave_job_indices'],
        'Completed/pending wave boundary changed')
    oc.bind_all(checkpoint['bindings'])
    bindings = oc.merge_bindings(config['source_bindings'], checkpoint['bindings'], {str(path): pred['request_sha256']})
    roots = [path.parent, pause_path.parent, *(Path(r['job']).parent for r in completed)]
    return config, roots, bindings


def saved_completion(pred, config):
    current, _, bindings = validate_predecessor(pred)
    oc.require(current == config, 'Reboot authorization changed')
    return dict(state=STATE, predecessor=deepcopy(pred),
        previous_boot_epoch_ms=config['previous_boot_epoch_ms'],
        resumed_boot_epoch_ms=config['resumed_boot_epoch_ms'],
        prior_campaign_complete=False, old_pid_disappearance_treated_as_success=False,
        prior_captures_implicitly_added=False, user_authorization_basis='explicit_resume_instruction',
        boot_evidence_basis='successful_live_CIM_query_at_launch_not_independent_OS_attestation',
        bindings=bindings, training_approved=False, source_cap_reset=False,
        visual_approval=False, independent_execution_attested=False)


def assert_current_boot(pred, config):
    oc.pin(pred['request_path'], pred['request_sha256'])
    oc.require(boot_epoch_ms() == config['resumed_boot_epoch_ms'], 'Another reboot requires a new explicit continuation')


def completion(pred, config):
    result = saved_completion(pred, config)
    assert_current_boot(pred, config)
    return result
