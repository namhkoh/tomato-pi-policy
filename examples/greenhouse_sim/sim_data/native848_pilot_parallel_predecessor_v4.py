"""Verify actual filtered-pilot parallel whole closure and every recorded child exit."""
from pathlib import Path
import json,re
from . import native848_bulk_sibling_gate_v5 as gate
from . import native848_data_roots_v1 as roots
from .depth_preview import sha256
from .dataset_review import require,verify_bindings
ROOT=Path(__file__).resolve().parents[3]
IMPLEMENTATIONS={'greenhouse.native848_pilot_parallel.v1':'v1','greenhouse.native848_pilot_parallel.v2':'v2','greenhouse.native848_pilot_parallel.v3':'v3','greenhouse.native848_pilot_parallel.v4':'v4'}
SCHEMAS=set(IMPLEMENTATIONS)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def implementation_paths(schema):
    require(schema in IMPLEMENTATIONS,'Unknown typed parallel predecessor schema')
    version=IMPLEMENTATIONS[schema]
    return version,ROOT/('data/sim_data/diagnostics/collection_20k_848_20260916_v1/run_native848_pilot_parallel_'+version+'.py'),Path(__file__).with_name('native848_pilot_sibling_worker_'+version+'.py')


def validate_retirement_evidence(proof,slot,coordinator,owned,*,maximum_wait_seconds=10,require_deferred_polls=False):
    require(proof['slot']==slot['slot'] and proof['command_identity']==slot['command_identity']
        and proof.get('native_identity')==slot['native_identity']
        and proof['auxiliary_identities']==slot.get('auxiliary_identities',[])
        and proof['owned_exit']==slot['owned_exit'] and proof['launch']==slot['launch']
        and proof['actual_owned_wait_verified'] is True and proof['owned_returncode']==owned['returncode']
        and proof['all_recorded_identities_absent'] is True and slot['all_recorded_identities_absent'] is True
        and 0<proof['bounded_retirement_wait_seconds']<=maximum_wait_seconds and proof['process_control_performed'] is False,
        'Parallel3 exact actual owned retirement evidence differs')
    identities=gate.owned_allowlist(coordinator,[slot]);rows=proof['retirement_snapshot']
    require(len({r['ProcessId'] for r in rows})==len(rows),'Duplicate retirement snapshot PID')
    require(gate.same_identity(gate.one(rows,coordinator['ProcessId']),coordinator),'Retirement snapshot coordinator differs')
    require(not {i['ProcessId'] for i in identities[1:]}&{r['ProcessId'] for r in rows},
        'Retirement snapshot still contains a recorded owned identity')
    if require_deferred_polls:validate_deferred_polls(proof,slot,coordinator)
    return slot.get('auxiliary_identities',[])


def validate_deferred_polls(proof,slot,coordinator):
    recorded=proof['recorded_slots'];own=[s for s in recorded if s['slot']==slot['slot']]
    require(len(own)==1 and all(own[0].get(k)==slot.get(k) for k in
        ('command_identity','native_identity','auxiliary_identities')),'Deferred retirement omitted exact source slot')
    expected=gate.owned_allowlist(coordinator,recorded);polls=proof['deferred_poll_evidence']
    require(isinstance(polls,list) and polls and polls[-1]['process_snapshot']==proof['retirement_snapshot']
        and polls[-1]['pending_owned_pids']==[],'Final deferred poll must prove all current records absent')
    previous=-1.
    for poll in polls:
        process=poll['process_inventory'];rows=poll['process_snapshot'];elapsed=poll['elapsed_seconds'];resources=poll['resource_check']
        require(type(elapsed) in (int,float) and elapsed>=previous and poll['admission_deferred'] is True
            and process['admission_deferred'] is True and process['exact_recorded_identities_only'] is True
            and process['owned_identities']==expected and process['no_blockers_observed'] is True and process['blockers']==[],
            'Deferred retirement must preserve exact records and block all admission')
        checked=gate.classify_owned(rows,expected,process,require_all=False)
        require(checked['no_blockers_observed'] is True and checked['blockers']==[], 'Deferred snapshot classification differs')
        require(resources['memory']['allowed'] is True and resources['memory']['commit_headroom_bytes']>=20*2**30
            and resources['disk_free_bytes']>=60*2**30 and resources['gpu']['free_mib']>=4096,
            'Every deferred poll must retain actual resource-floor evidence')
        previous=elapsed
    require(proof['elapsed_retirement_seconds']>=previous,'Final retirement elapsed time differs')


def check_parallel(value,*,rows=None):
    bindings={}
    def pinned(item):
        path=roots.resolve_evidence(item['path'])
        require(sha256(path)==item['sha256'],'Changed parallel predecessor evidence')
        bindings[str(path)]=item['sha256'];return path
    def tracked(path,expected=None):
        path=Path(path).resolve();pin=sha256(path)
        require(expected is None or pin==expected,'Changed derived predecessor receipt')
        bindings[str(path)]=pin;return read(path)
    metadata=read(pinned(value['metadata']));trial=Path(metadata['trial_path']).resolve()
    require(roots.diagnostic(trial)==trial,'Parallel predecessor must be an actual allowed data trial')
    result_path=pinned(value['result']);terminal_path=pinned(value['terminal'])
    require(result_path==trial/'result.json' and terminal_path==trial/'owner_complete.json'
        and not (trial/'failure.json').exists(),'Whole parallel predecessor must have a successful terminal receipt')
    result,terminal=read(result_path),read(terminal_path)
    require(result['schema'] in SCHEMAS and terminal['result_sha256']==sha256(result_path)
        and result['training_approved'] is False,'Parallel terminal result binding differs')
    if result['schema'] in SCHEMAS:
        require(terminal['whole_parallel_owner_complete'] is True and result['requested_worker_count'] in (2,3,4)
            and len(result['frame_counts'])==result['requested_worker_count']
            and all(type(n) is int and 1<=n<=512 for n in result['frame_counts'])
            and all(type(r['slot']) is int and 0<=r['slot']<result['requested_worker_count'] for r in result['results']),'New parallel bound changed')
    if result['schema'] in SCHEMAS:
        intent_path=pinned(result['intent']);require(intent_path==trial/'intent.json','Wrong parallel predecessor intent')
        intent=read(intent_path)
        require(intent['schema']==result['schema'] and intent['native_workers']==result['requested_worker_count']
            and intent['frame_counts']==result['frame_counts'] and intent['planned_frames']==result['planned_frames']==sum(result['frame_counts']),
            'Parallel predecessor declared count differs')
    require(not result.get('harmless_bootstrap_failures') and not result.get('incomplete_slots'),
        'Failed or incomplete predecessors need explicit repair; no automatic retry')
    coordinator=gate.identity(result['coordinator_identity'])
    birth=re.fullmatch(r'/Date\((\d+)\)/',coordinator['CreationDate'])
    require(birth and int(birth[1])==metadata['creation_unix_ms'] and coordinator['ProcessId']==metadata['pid']
        and Path(coordinator['ExecutablePath']).resolve()==Path(metadata['executable']).resolve(),
        'Exact parallel owner PID/creation/executable differs')
    actual=gate.command_argv(coordinator['CommandLine']);recorded=gate.command_argv(metadata['command_line'])
    require(actual and recorded and Path(actual[0]).resolve()==Path(recorded[0]).resolve()
        and actual[1:]==recorded[1:],'Exact parallel owner command arguments differ')
    require(terminal['all_owned_command_children_waited'] is True and terminal['whole_parallel_owner_complete'] is True,
        'Actual whole pilot owner waits required')
    request_path=pinned(intent['request']);review_path=pinned(intent['launch_review'])
    request,review=read(request_path),read(review_path)
    version,owner_path,worker_path=implementation_paths(result['schema'])
    if version in ('v3','v4'):
        require(result.get('unretired_owned_slots')==[] and result.get('all_recorded_owned_identities_absent') is True
            and terminal.get('all_recorded_identities_absent') is True,'Every recorded owned identity must retire before parallel3 closure')
    require(review['owner_sha256']==sha256(owner_path) and review['worker_sha256']==sha256(worker_path)
        and review['gate_sha256']==sha256(gate.__file__) and review['request_sha256']==sha256(request_path)
        and review['native_launch_review_passed'] is True and review['blocking_findings']==[]
        and review['native_worker_count']==result['requested_worker_count']==request['worker_count']
        and result['source_bindings'].get(str(owner_path.resolve()))==sha256(owner_path)
        and result['source_bindings'].get(str(worker_path.resolve()))==sha256(worker_path)
        and request['schema']==result['schema'] and request['plans']==intent['plans']
        and request['frame_counts']==result['frame_counts'] and request['scene_preflights']==intent['scene_preflights'],
        'Exact reviewed pilot parent implementation/request required')
    def arg(argv,name):
        require(argv.count(name)==1 and argv.index(name)+1<len(argv),'Missing/repeated pilot owner argument')
        return argv[argv.index(name)+1]
    require(str(owner_path.resolve()) in [str(Path(a).resolve()) for a in actual]
        and Path(arg(actual,'--output')).resolve()==trial
        and Path(arg(actual,'--request')).resolve()==request_path and arg(actual,'--request-sha256')==sha256(request_path)
        and Path(arg(actual,'--review')).resolve()==review_path and arg(actual,'--review-sha256')==sha256(review_path),
        'Pilot parent OS command differs from actual request/review')
    verify_bindings(result['source_bindings']);bindings.update(result['source_bindings'])
    slots=result.get('completed_slots',result.get('slots'))
    require(isinstance(slots,list) and 1<=len(slots)<=4 and len({s['slot'] for s in slots})==len(slots),
        'Unique actual launched parallel slot inventory required')
    results={row['slot']:row for row in result['results']}
    require(len(results)==len(result['results']),'Repeated completed slot result')
    skipped={row['slot']:row for row in result.get('skipped_admissions',[])}
    require(not set(results)&set(skipped),'Captured and skipped slots overlap')
    seen_pids={coordinator['ProcessId']}
    for slot in slots:
        index=slot['slot'];folder=trial/f'slot_{index}'
        command=gate.identity(slot['command_identity']);native=gate.identity(slot['native_identity'])
        require(not seen_pids&{command['ProcessId'],native['ProcessId']} and command['ProcessId']!=native['ProcessId'],
            'Repeated predecessor PID identity')
        seen_pids.update((command['ProcessId'],native['ProcessId']))
        launch_path=folder/'launch.json';exit_path=folder/'owned_exit.json'
        # Old probe inventory used slots without explicit exit pins; its result
        # rows carry the exact same owned exit and launch binding instead.
        if 'owned_exit' in slot:
            require(pinned(slot['owned_exit'])==exit_path and pinned(slot['launch'])==launch_path,'Wrong retired sibling paths')
        launch,owned=tracked(launch_path),tracked(exit_path)
        require(owned['method']=='subprocess_wait_on_owned_process' and owned['pid']==launch['pid']==command['ProcessId']
            and owned['launch_sha256']==sha256(launch_path),'Actual owned child wait/launch binding differs')
        gate.command_child(command,coordinator,launch['command']);gate.native_child(native,command,launch['command'][1:])
        if version in ('v3','v4'):
            require(arg(launch['command'],'-m')=='sim_data.native848_pilot_sibling_worker_'+version,'Parallel3 native worker module differs')
            retirement_path=pinned(slot['retirement_receipt'])
            require(retirement_path==folder/'all_recorded_identities_exited.json','Wrong per-slot retirement evidence path')
            retirement=read(retirement_path)
            auxiliaries=validate_retirement_evidence(retirement,slot,coordinator,owned,maximum_wait_seconds=60 if version=='v4' else 10,require_deferred_polls=(version=='v4'))
            for identity in auxiliaries:
                require(identity['ProcessId'] not in seen_pids,'Duplicate recorded auxiliary PID')
                seen_pids.add(identity['ProcessId'])
            for path,h in retirement['source_bindings'].items():tracked(path,h)
        ready=tracked(folder/'native_ready.json')
        ticket=tracked(folder/'ticket.json',ready['ticket_sha256'])
        require(ready['native_identity']==native and ready['command_identity']==command and ready['slot']==index
            and ticket['coordinator_identity']==coordinator and ticket['slot']==index
            and Path(ticket['output']).resolve()==folder/'capture','Native registration and ticket identities differ')
        require(ticket['schema']=='greenhouse.native848_pilot_parallel_ticket.'+version
            and ticket['worker_count']==result['requested_worker_count'] and ticket['frames_per_worker']==result['frame_counts'][index]
            and ticket['plan']==intent['plans'][index] and ticket['scene_preflight']==intent['scene_preflights'][index],
            'Pilot ticket plan/preflight/count differs')
        if index in results:
            row=results[index]
            item=row.get('owner_result',dict(path=row.get('owner_result_path'),sha256=row.get('owner_result_sha256')))
            require(pinned(item)==folder/'result.json' and owned['returncode']==0,'Actual successful per-slot result required')
            receipt=read(folder/'result.json')
            if result['schema'] in SCHEMAS:
                slot_terminal=tracked(folder/'owner_complete.json')
                require(slot_terminal['result_sha256']==sha256(folder/'result.json') and slot_terminal['parallel_slot_only'] is True,
                        'Actual parallel slot terminal missing or relabeled')
                if version in ('v3','v4'):
                    require(slot_terminal.get('all_recorded_identities_absent') is True
                        and slot_terminal.get('retirement_receipt')==slot['retirement_receipt'],
                        'Parallel3 slot terminal must bind its exact all-identity retirement')
            require(receipt['state']=='owned_unique9mm_filtered_capture_exited_pending_census_annotation_and_visual_review'
                and Path(receipt['owned_exit_path']).resolve()==exit_path and receipt['owned_exit_sha256']==sha256(exit_path),
                'Per-slot result does not bind its real native exit')
            context_path=Path(receipt['context_path']).resolve()
            require(context_path==folder/'capture/batch_context.json','Wrong slot context')
            context=tracked(context_path,receipt['context_sha256'])
            release=tracked(folder/'release.json')
            require(context['schema']=='greenhouse.native848_unique9mm_pilot_context.v1'
                and context['cpu_scene_preflight']==ticket['scene_preflight']
                and release['ticket_sha256']==sha256(folder/'ticket.json')
                and release['resource_admission']['allowed'] is True
                and release['resource_admission']['process_inventory']['no_blockers_observed'] is True,
                'Actual native pilot context/admission differs')
            if version in ('v2','v3','v4'):
                admission=tracked(folder/'actual_process_admission.json')
                require(admission['same_snapshot_classification'] is True and admission['require_all_live_identities'] is True
                    and admission['process_inventory']['no_blockers_observed'] is True,'Actual v2 process admission required')
                require(context['source_bindings'].get(str((folder/'actual_process_admission.json').resolve()))==sha256(folder/'actual_process_admission.json'),
                    'Native context omitted actual retirement-aware admission')
            for name in ('ticket.json','command_identity.json','native_ready.json','release.json'):
                path=folder/name;tracked(path,context['source_bindings'].get(str(path.resolve())))
                require(context['source_bindings'].get(str(path.resolve()))==sha256(path),'Native context omitted actual process receipt')
        else:
            # A resource-denied child can exit before SimulationApp after an
            # explicit cancel. It remains an actual exit, never a capture.
            require(index in skipped and skipped[index]['stage']=='before_simulationapp',
                'Launched slot lacks a completed capture or explicit nonnative cancellation')
            hold=skipped[index]
            require(hold['simulation_app_started'] is False and pinned(hold['owned_exit'])==exit_path
                and pinned(hold['launch'])==launch_path and not (folder/'simulation_app_starting.json').exists()
                and not (folder/'capture/batch_context.json').exists(),'Canceled child may not have started native simulation')
            failure=read(pinned(hold['failure']))
            require(failure['simulation_app_construction_started'] is False and failure['ticket_sha256']==sha256(folder/'ticket.json'),
                'Canceled worker did not retain actual pre-SimulationApp evidence')
    require(set(results)<=set(s['slot'] for s in slots),'Completed result absent from launched identity inventory')
    rows=gate.snapshot() if rows is None else rows
    present={row['ProcessId'] for row in rows}
    require(not seen_pids&present,'Whole predecessor owner/native command processes still present or PID reused')
    return dict(metadata=metadata,coordinator_identity=coordinator,source_bindings=bindings,
        completed_capture_slots=sorted(results),skipped_slots=sorted(skipped),all_recorded_native_children_absent=True,all_recorded_auxiliary_identities_absent=(version in ('v3','v4')),
        completion_method='verified_parallel_terminal_and_actual_per_slot_owned_exits',synthetic_root_native_exit=False)


def check(value,*,rows=None):
    require(set(value)=={'metadata','result','terminal'},'Exact typed actual pilot parallel predecessor required')
    return check_parallel(value,rows=rows)
