"""Verify actual whole parallel-owner completion; never synthesize an exit."""
from pathlib import Path
import json
import re
from . import native848_bulk_sibling_gate_v5 as gate
from .depth_preview import sha256
from .dataset_review import require,verify_bindings

ROOT=Path(__file__).resolve().parents[3]
D=ROOT/'data/sim_data/diagnostics'
SCHEMAS={'greenhouse.original848_two_worker_probe.v1','greenhouse.original848_sustained_workers.v1',
         'greenhouse.original848_sustained_workers.v2'}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def check(value,*,rows=None):
    bindings={}
    def pinned(item):
        path=Path(item['path']).resolve()
        require(path.is_relative_to(ROOT) and sha256(path)==item['sha256'],'Changed parallel predecessor evidence')
        bindings[str(path)]=item['sha256'];return path
    def tracked(path,expected=None):
        path=Path(path).resolve();pin=sha256(path)
        require(expected is None or pin==expected,'Changed derived predecessor receipt')
        bindings[str(path)]=pin;return read(path)
    metadata=read(pinned(value['metadata']));trial=Path(metadata['trial_path']).resolve()
    require(trial.is_relative_to(D),'Parallel predecessor must be an actual workspace trial')
    result_path=pinned(value['result']);terminal_path=pinned(value['terminal'])
    require(result_path==trial/'result.json' and terminal_path==trial/'owner_complete.json'
        and not (trial/'failure.json').exists(),'Whole parallel predecessor must have a successful terminal receipt')
    result,terminal=read(result_path),read(terminal_path)
    require(result['schema'] in SCHEMAS and terminal['result_sha256']==sha256(result_path)
        and result['training_approved'] is False,'Parallel terminal result binding differs')
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
