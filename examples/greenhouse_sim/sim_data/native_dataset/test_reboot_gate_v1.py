"""Synthetic stop metadata tests; no native execution or review qualification."""
from copy import deepcopy
from pathlib import Path

import pytest

from . import reboot_gate_v1 as g


@pytest.fixture
def evidence(tmp_path):
    def save(name, value):
        p = tmp_path/name
        g.oc.write_new(p, value)
        return dict(path=str(p), sha256=g.oc.sha256(p))
    marker = save('completed_job.json', dict(synthetic=True))
    checkpoint = save('checkpoint.json', dict(schema='greenhouse.collection_restart_handoff.v1',
        state='pre_restart_checkpoint_only_not_automatic_resume', resolution=[1696,816],
        goal_accepted_train=20000, training_approved=False, completed_job_count=1,
        completed_jobs=[dict(index=1, job=str(tmp_path/'old/job1'))],
        pending_wave_job_indices=list(range(2,61)), bindings={marker['path']:marker['sha256']}))
    ids = list(range(100,108))
    stop = save('stop.json', dict(state='eight_exact_dataset_owners_stopped_for_user_reboot',
        captures_deleted=False, prior_receipts_rewritten=False, system_reboot_issued=False,
        goal_20000_complete=False, records=[dict(pid=pid,state='stopped_on_user_request',
            termination_called=True,observed_exit_code=125,clean_collection_completion_claimed=False) for pid in ids]))
    verification = save('verified.json', dict(remaining_owner_count=0, active_native_processes=[],stopped_owner_ids=ids))
    pause = save('pause.json', dict(schema='greenhouse.user_requested_reboot_pause.v1',
        state='stopped_and_saved_waiting_for_user_restart', all_eight_exact_owners_exited=True,
        completed_native_captures_preserved=True, active_native_processes=[], goal_20000_complete=False,
        source_checkpoint=checkpoint, stop_receipt=stop, verification=verification,
        completed_current_wave_jobs=1, unfinished_wave_jobs=list(range(2,61))))
    config = dict(schema=g.SCHEMA,user_instruction='resume collection',previous_boot_epoch_ms=10,
        resumed_boot_epoch_ms=20,pause_path=pause['path'],pause_sha256=pause['sha256'],
        source_bindings={p['path']:p['sha256'] for p in (pause,checkpoint,stop,verification)})
    auth = save('authorization.json', config)
    return dict(pred=dict(kind=g.KIND,request_path=auth['path'],request_sha256=auth['sha256']),
        config=config, marker=marker, save=save)


def test_saved_replay_retains_interruption_without_claiming_old_campaign_complete(evidence):
    value = g.saved_completion(evidence['pred'], evidence['config'])
    assert value['state'] == g.STATE
    for key in ('prior_campaign_complete','old_pid_disappearance_treated_as_success',
                'prior_captures_implicitly_added','training_approved','source_cap_reset'):
        assert value[key] is False


def test_live_replay_requires_exact_new_boot_but_saved_replay_does_not(evidence,monkeypatch):
    monkeypatch.setattr(g,'boot_epoch_ms',lambda:20)
    assert g.completion(evidence['pred'],evidence['config'])['resumed_boot_epoch_ms']==20
    monkeypatch.setattr(g,'boot_epoch_ms',lambda:21)
    with pytest.raises(ValueError,match='Another reboot'):
        g.completion(evidence['pred'],evidence['config'])
    assert g.saved_completion(evidence['pred'],evidence['config'])['resumed_boot_epoch_ms']==20


@pytest.mark.parametrize('change',[
    dict(user_instruction='continue without asking'),
    dict(resumed_boot_epoch_ms=10),
    dict(previous_boot_epoch_ms=True),
])
def test_changed_resume_or_same_boot_rejected(evidence,change):
    config=dict(evidence['config'],**change)
    auth=evidence['save']('changed_auth.json',config)
    pred=dict(evidence['pred'],request_path=auth['path'],request_sha256=auth['sha256'])
    with pytest.raises(ValueError,match='Explicit user resumption'):
        g.validate_predecessor(pred)


def test_changed_completed_source_rejected(evidence):
    Path(evidence['marker']['path']).write_text('{}')
    with pytest.raises(ValueError,match='Changed or missing pinned file'):
        g.validate_predecessor(evidence['pred'])


def test_old_predecessor_shape_not_silently_relabelled(evidence):
    pred=dict(evidence['pred'],kind='original_serial39.v1',supervisor={'pid':123})
    with pytest.raises(ValueError,match='Explicit reboot predecessor'):
        g.validate_predecessor(pred)


def test_new_producer_and_reader_keep_native_mutex_but_distinct_receipts():
    from . import serial_phases as old, serial_phases_reboot_v2 as new
    from . import original_reboot_phase_inventory_v2 as reader
    assert old.LOCK_NAME==new.LOCK_NAME
    assert old.SCHEMA!=new.SCHEMA and old.RECEIPT_SCHEMA!=new.RECEIPT_SCHEMA
    assert old.WORKERS==new.WORKERS and old.FLAGS==new.FLAGS
    assert reader.PRODUCER_SHA256==g.oc.sha256(new.__file__)
    assert reader.q is new
