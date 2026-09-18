"""Explicit whole-coordinator closure before annotation of either finite-two slot."""
from pathlib import Path
import re
from . import native848_bulk_sibling_gate_v5 as gate
from . import native848_fully_labeled_sibling_worker_v1 as worker
from . import native848_fully_labeled_two_admission_v1 as admission
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256

SCHEMA='greenhouse.native848_fully_labeled_two_closure.v1'


def authenticate_slot(trial,result_sha256,parent_closure,*,rows=None):
    """Require five explicit pins: metadata/result/terminal/outer_launch/outer_exit.

    Returns closure/source bindings for a typed annotation dispatcher. This
    never calls the serial-owner annotation checker or claims a per-slot lock.
    """
    import run_native848_fully_labeled_two_v1 as owner
    require(set(parent_closure)=={'metadata','result','terminal','outer_launch','outer_exit'},
        'Explicit actual whole-owner closure and outer wait pins required')
    trial=Path(trial).resolve();bindings={str(Path(__file__).resolve()):sha256(__file__)}
    def tracked(spec):
        path=owner.pin(spec);bindings[str(path)]=spec['sha256'];return read_json(path)
    parent=tracked(parent_closure['result']);meta=tracked(parent_closure['metadata'])
    complete=tracked(parent_closure['terminal']);outer=tracked(parent_closure['outer_exit']);launch=tracked(parent_closure['outer_launch'])
    root=Path(meta['trial_path']).resolve()
    require(root.parent in owner.OUTPUT_ROOTS and trial.parent==root and trial.name in ('slot_0','slot_1')
        and Path(parent_closure['result']['path']).resolve()==root/'result.json'
        and Path(parent_closure['terminal']['path']).resolve()==root/'owner_complete.json'
        and not (root/'failure.json').exists(),'Actual successful finite-two parent namespace required')
    require(parent['schema']==owner.SCHEMA and parent['state']==owner.RESULT_STATE
        and parent['requested_workers']==parent['started_workers']==2 and parent['unlaunched_slots']==[]
        and parent['incomplete_slots']==[] and parent['all_owned_command_children_waited'] is True
        and parent['all_recorded_identities_absent'] is True and parent['training_approved'] is False
        and parent['accepted_training_increment']==0,'Both complete owned slots required')
    require(complete['result_sha256']==parent_closure['result']['sha256'] and complete['both_slots_complete'] is True
        and complete['coordinator_native_lock_released'] is True and complete['all_owned_command_children_waited'] is True
        and complete['all_recorded_identities_absent'] is True and complete['capture_only'] is True
        and complete['training_approved'] is False and complete['accepted_training_increment']==0,
        'Truthful whole-coordinator lock release required')
    exact=gate.identity(meta['exact_identity'])
    require(meta['pid']==exact['ProcessId'] and exact['CreationDate']=='/Date('+str(meta['creation_unix_ms'])+')/'
        and Path(meta['executable']).resolve()==Path(exact['ExecutablePath']).resolve()
        and meta['command_line']==exact['CommandLine'] and parent['coordinator_identity']==exact,
        'Original actual full6 coordinator identity required')
    command=gate.command_argv(meta['command_line'])
    require(launch['pid']==outer['pid']==meta['pid'] and launch['command']==command
        and outer['returncode']==0 and outer['method']=='subprocess_wait_on_owned_process'
        and outer['launch_sha256']==parent_closure['outer_launch']['sha256'],
        'Actual outer Popen wait0 required after both children')
    require(Path(parent_closure['outer_exit']['path']).resolve().parent==Path(parent_closure['outer_launch']['path']).resolve().parent,
        'Outer wait/launch folders differ')
    owner_path=Path(owner.__file__).resolve();owner_sha=sha256(owner_path)
    require(parent['source_bindings'].get(str(owner_path))==owner_sha and command.count(str(owner_path))==1,
        'Actual typed finite-two owner source required')
    def argument(flag):
        require(command.count(flag)==1 and command.index(flag)+1<len(command),'Unique owner argument required')
        return command[command.index(flag)+1]
    require(Path(argument('--output')).resolve()==root,'Actual owner output differs')
    intent=tracked(parent['intent']);require(Path(parent['intent']['path']).resolve()==root/'intent.json'
        and intent['coordinator_identity']==exact and intent['source_bindings']==parent['source_bindings'],'Actual intent differs')
    request=tracked(intent['request']);review=tracked(intent['launch_review'])
    require(Path(argument('--request')).resolve()==Path(intent['request']['path']).resolve()
        and argument('--request-sha256')==intent['request']['sha256']
        and Path(argument('--review')).resolve()==Path(intent['launch_review']['path']).resolve()
        and argument('--review-sha256')==intent['launch_review']['sha256'],'Exact request/review command pins differ')
    plans=owner.validate_request(request,review,intent['request']['sha256'])
    require(len(parent['results'])==len(parent['actual_owned_waits'])==len(parent['retirement_receipts'])==2
        and {r['slot'] for r in parent['results']}=={0,1},'Exactly two actual slot closures required')
    recorded=[];selected=None
    for entry in sorted(parent['results'],key=lambda r:r['slot']):
        index=entry['slot'];folder=root/f'slot_{index}'
        value=tracked(entry['result']);terminal=tracked(entry['complete'])
        require(Path(entry['result']['path']).resolve()==folder/'result.json'
            and Path(entry['complete']['path']).resolve()==folder/'slot_complete.json'
            and not (folder/'capture/failure.json').exists() and not (folder/'failure.json').exists(),
            'Exact successful slot files required')
        require(value['schema']==owner.SLOT_SCHEMA and value['parent_completion_required'] is True
            and terminal['result_sha256']==entry['result']['sha256'] and terminal['parent_completion_required'] is True
            and 'serial_native_lock_released_after_this_receipt' not in terminal
            and value['parent_intent']==parent['intent'] and value['source_bindings']==parent['source_bindings'],
            'Slot must depend on this completed coordinator, never claim serial ownership')
        verified=owner.completed_slot(folder,plans[index][1],request['plans'][index]['sha256'],request['scene_preflights'][index]['sha256'],exact)
        require(all(value[k]==v for k,v in verified.items()),'Actual complete slot capture differs')
        require(dict(path=value['owned_exit_path'],sha256=value['owned_exit_sha256']) in parent['actual_owned_waits']
            and value['retirement_receipt'] in parent['retirement_receipts'],'Parent omitted actual slot wait/retirement')
        retirement=tracked(value['retirement_receipt'])
        require(retirement['slot']==index and retirement['actual_owned_wait_verified'] is True
            and retirement['owned_returncode']==0 and retirement['all_recorded_identities_absent'] is True
            and retirement['owned_exit']==dict(path=value['owned_exit_path'],sha256=value['owned_exit_sha256'])
            and 0<retirement['bounded_retirement_wait_seconds']<=60,'Exact bounded real-wait retirement required')
        identities=[retirement['command_identity'],retirement['native_identity'],*retirement['auxiliary_identities']]
        snapshot=retirement['retirement_snapshot'];observed={r['ProcessId'] for r in snapshot}
        require(not any(r['ProcessId'] in observed for r in identities),'Recorded child remained in final snapshot')
        _,inventory=admission.record_inventory(snapshot,exact,retirement['recorded_slots'],admitting=False,register=False)
        require(not inventory['blockers'],'Unknown process in exact final retirement snapshot')
        require(retirement['deferred_poll_evidence'],'Actual sampled retirement required')
        previous=-1
        for poll in retirement['deferred_poll_evidence']:
            require(poll['elapsed_seconds']>=previous and poll['admission_deferred'] is True,'Invalid retirement poll order')
            previous=poll['elapsed_seconds'];resources=poll['resource_check']
            require(resources is not None and admission.assess(resources['memory'],resources['disk_free_bytes'],resources['gpu'])['allowed'],
                'Every actual retirement poll requires unchanged reserve evidence')
            _,actual_inventory=admission.record_inventory(poll['process_snapshot'],exact,retirement['recorded_slots'],admitting=False,register=False)
            require(actual_inventory['classifications']==poll['process_inventory']['classifications']
                and actual_inventory['blockers']==poll['process_inventory']['blockers'],'Retirement classification snapshot differs')
        require(retirement['deferred_poll_evidence'][-1]['pending_owned_pids']==[], 'Last retirement poll still has owned identities')
        verify_bindings(retirement['source_bindings']);bindings.update(retirement['source_bindings'])
        recorded.extend(identities)
        if folder==trial:
            require(entry['result']['sha256']==result_sha256,'Selected slot result changed');selected=value
    verify_bindings(parent['source_bindings']);bindings.update(parent['source_bindings'])
    rows=gate.snapshot() if rows is None else rows;by={r['ProcessId']:r for r in rows}
    for original in [exact,*recorded]:
        now=by.get(original['ProcessId'])
        if now is None:continue
        now=gate.identity(now)
        old_birth=re.fullmatch(r'/Date\((\d+)\)/',original['CreationDate']);new_birth=re.fullmatch(r'/Date\((\d+)\)/',now['CreationDate'])
        require(old_birth and new_birth and int(new_birth[1])>int(old_birth[1]) and not gate.same_identity(now,original),
            'Actual coordinator/owned identity remains or full identity reuse unproven')
    require(selected is not None,'Selected completed slot absent')
    return dict(schema=SCHEMA,parent_closure=parent_closure,slot=selected['bounded_two_worker_slot'],
        both_actual_waits_verified=True,actual_outer_wait0_verified=True,coordinator_lock_released=True,
        source_bindings=bindings,training_approved=False,accepted_training_increment=0)
