"""Verify typed actual serial or parallel predecessor completion; never synthesize an exit."""
from pathlib import Path
import json
import re
from . import native848_bulk_sibling_gate_v5 as gate
from . import native848_data_roots_v1 as roots
from .depth_preview import sha256
from .dataset_review import require,verify_bindings

ROOT=Path(__file__).resolve().parents[3]
D=ROOT/'data/sim_data/diagnostics'
SCHEMAS={'greenhouse.original848_two_worker_probe.v1','greenhouse.original848_sustained_workers.v1',
         'greenhouse.original848_sustained_workers.v2','greenhouse.native848_multianchor_parallel.v1',
         'greenhouse.native848_multianchor_parallel.v2'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


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
    if result['schema'] in ('greenhouse.native848_multianchor_parallel.v1','greenhouse.native848_multianchor_parallel.v2'):
        require(terminal['whole_parallel_owner_complete'] is True and result['requested_worker_count'] in (2,3,4)
            and len(result['frame_counts'])==result['requested_worker_count']
            and all(type(n) is int and 2<=n<=512 for n in result['frame_counts'])
            and all(type(r['slot']) is int and 0<=r['slot']<result['requested_worker_count'] for r in result['results']),'New parallel bound changed')
    if result['schema'] in ('greenhouse.native848_multianchor_parallel.v1','greenhouse.native848_multianchor_parallel.v2'):
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
    verify_bindings(result['source_bindings'])
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
        ready=tracked(folder/'native_ready.json')
        ticket=tracked(folder/'ticket.json',ready['ticket_sha256'])
        require(ready['native_identity']==native and ready['command_identity']==command and ready['slot']==index
            and ticket['coordinator_identity']==coordinator and ticket['slot']==index
            and Path(ticket['output']).resolve()==folder/'capture','Native registration and ticket identities differ')
        if index in results:
            row=results[index]
            item=row.get('owner_result',dict(path=row.get('owner_result_path'),sha256=row.get('owner_result_sha256')))
            require(pinned(item)==folder/'result.json' and owned['returncode']==0,'Actual successful per-slot result required')
            receipt=read(folder/'result.json')
            if result['schema'] in ('greenhouse.native848_multianchor_parallel.v1','greenhouse.native848_multianchor_parallel.v2'):
                slot_terminal=tracked(folder/'owner_complete.json')
                require(slot_terminal['result_sha256']==sha256(folder/'result.json') and slot_terminal['parallel_slot_only'] is True,
                        'Actual parallel slot terminal missing or relabeled')
            require(receipt['state']=='owned_original848_bulk_capture_exited_pending_independent_CPU_admission'
                and Path(receipt['owned_exit_path']).resolve()==exit_path and receipt['owned_exit_sha256']==sha256(exit_path),
                'Per-slot result does not bind its real native exit')
            context_path=Path(receipt['context_path']).resolve()
            require(context_path==folder/'capture/batch_context.json','Wrong slot context')
            context=tracked(context_path,receipt['context_sha256'])
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
        completed_capture_slots=sorted(results),skipped_slots=sorted(skipped),all_recorded_native_children_absent=True,
        completion_method='verified_parallel_terminal_and_actual_per_slot_owned_exits',synthetic_root_native_exit=False)


SERIAL_OWNERS={
 'run_native848_bulk_v2.py':('8e9131cc92a64d2e6c83ba7a5e0d52c8a1dd4f8b81adb6b2f0ede3cf6fa5bf0c','sim_data.native848_bulk_worker_v1'),
 'run_native848_multianchor_v1.py':('c92ec4cd9409dfbaed657bd5344ea9cd78380deb1c7cc77c62c135ebea1237b8','sim_data.native848_multianchor_worker_v1')}
H=D/'collection_20k_848_20260916_v1'


def check_serial(value,*,rows=None):
    require(set(value)=={'metadata','owned_exit','terminal','launch'},'Typed serial predecessor pin set required')
    bindings={}
    def pinned(item):
        p=roots.resolve_evidence(item['path']);require(sha256(p)==item['sha256'],'Changed serial predecessor evidence')
        bindings[str(p)]=item['sha256'];return p
    def tracked(p,expected=None):
        p=Path(p).resolve();h=sha256(p);require(expected is None or h==expected,'Changed serial completion source')
        bindings[str(p)]=h;return read(p)
    paths={k:pinned(v) for k,v in value.items()};meta=read(paths['metadata']);trial=Path(meta['trial_path']).resolve()
    require(roots.diagnostic(trial)==trial and paths['owned_exit']==trial/'owned_exit.json'
        and paths['launch']==trial/'launch.json' and paths['terminal']==trial/'owner_complete.json'
        and not (trial/'failure.json').exists(),'Actual successful serial predecessor required')
    owned,launch,terminal=[read(paths[k]) for k in ('owned_exit','launch','terminal')]
    result=tracked(trial/'result.json',terminal['result_sha256']);intent=tracked(trial/'intent.json')
    require(terminal['serial_native_lock_released_after_this_receipt'] is True
        and result['state']=='owned_original848_bulk_capture_exited_pending_independent_CPU_admission'
        and result['training_approved'] is False and owned['returncode']==0
        and owned['method']=='subprocess_wait_on_owned_process' and owned['pid']==launch['pid']
        and owned['launch_sha256']==sha256(paths['launch'])
        and result['owned_exit_sha256']==sha256(paths['owned_exit'])
        and Path(result['owned_exit_path']).resolve()==paths['owned_exit'],'Serial owner/actual native wait differs')
    require(type(meta['pid']) is int and type(meta['creation_unix_ms']) is int
        and isinstance(meta['executable'],str) and isinstance(meta['command_line'],str),'Exact serial metadata identity required')
    observed=meta.get('exact_identity')
    require(isinstance(observed,dict) and observed['ProcessId']==meta['pid']
        and observed['CreationDate']=='/Date('+str(meta['creation_unix_ms'])+')/'
        and Path(observed['ExecutablePath']).resolve()==Path(meta['executable']).resolve()
        and observed['CommandLine']==meta['command_line'],'Serial observed PID/creation/executable/command metadata differs')
    argv=gate.command_argv(meta['command_line'])
    require(argv and Path(argv[0]).resolve()==Path(meta['executable']).resolve(),'Serial metadata executable differs')
    found=[(H/name,settings) for name,settings in SERIAL_OWNERS.items() if str(H/name).lower() in [str(Path(a)).lower() for a in argv]]
    require(len(found)==1,'Unapproved serial predecessor owner')
    owner,(expected,module)=found[0];tracked_owner=pinned(dict(path=str(owner),sha256=expected))
    require(intent['schema']=='greenhouse.original848_bulk_owner_intent.v1'
        and intent['source_bindings']==result['source_bindings']
        and result['source_bindings'].get(str(tracked_owner))==expected,'Serial source owner binding differs')
    def arg(args,name):
        require(args.count(name)==1 and args.index(name)+1<len(args),'Missing/repeated exact command argument')
        return args[args.index(name)+1]
    plan=Path(intent['plan_path']).resolve();plan_sha=intent['plan_sha256']
    require(Path(arg(argv,'--output')).resolve()==trial and Path(arg(argv,'--plan')).resolve()==plan
        and arg(argv,'--plan-sha256')==plan_sha==result['plan_sha256']==launch['plan_sha256']
        and arg(launch['command'],'-m')==module and Path(arg(launch['command'],'--output')).resolve()==trial/'capture'
        and Path(arg(launch['command'],'--plan')).resolve()==plan
        and arg(launch['command'],'--plan-sha256')==plan_sha,'Exact serial native command differs')
    pinned(dict(path=str(plan),sha256=plan_sha))
    review_path=Path(intent['review_path']).resolve();review_sha=intent['review_sha256']
    require(Path(arg(argv,'--review')).resolve()==review_path and arg(argv,'--review-sha256')==review_sha,
            'Serial predecessor launch review arguments differ')
    review=tracked(review_path,review_sha)
    require(review['owner_sha256']==expected and review['plan_sha256']==plan_sha
        and review['native_launch_review_passed'] is True and review['blocking_findings']==[],
        'Serial predecessor actual reviewed owner scope differs')
    for stem,filename in [('capture_result','result.json'),('manifest','batch_manifest.json'),('context','batch_context.json')]:
        path=Path(result[stem+'_path']).resolve();require(path==trial/'capture'/filename,'Serial native closure escaped capture')
        tracked(path,result[stem+'_sha256'])
    verify_bindings(result['source_bindings']);bindings.update(result['source_bindings'])
    present={r['ProcessId'] for r in (gate.snapshot() if rows is None else rows)}
    require(meta['pid'] not in present and launch['pid'] not in present,'Serial predecessor owner/owned command remains or PID reused')
    return dict(metadata=meta,source_bindings=bindings,completion_method='actual_serial_terminal_and_owned_command_wait',
        synthetic_root_native_exit=False,serial_native_kit_identity_not_in_legacy_receipt=True)


def check(value,*,rows=None):
    if set(value)=={'metadata','owned_exit','terminal','launch'}:return check_serial(value,rows=rows)
    require(set(value)=={'metadata','result','terminal'},'Unknown or mixed predecessor receipt type')
    return check_parallel(value,rows=rows)
