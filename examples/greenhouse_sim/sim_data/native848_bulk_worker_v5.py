"""Admission-only wrapper for two to four owned sustained compact bulk workers.

Frozen v1 capture, profile, callback, geometry and static checks are reused.
This wrapper never acquires or releases the coordinator's native mutex.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import sys
import time
import traceback
from . import native848_bulk_worker_v1 as producer
from . import native848_bulk_plan_v1 as api
from . import native848_bulk_sibling_gate_v5 as gate
from .native848_bulk_io_v1 import save_json
from .dataset_review import require,verify_bindings
from .depth_preview import sha256

FROZEN_PRODUCER_SHA='5306fe59b579aa8e7bfc021ac39639ef7e0e62b06a851b12a6ab79a96daa204a'
ROOT=Path(__file__).resolve().parents[3]
D=ROOT/'data/sim_data/diagnostics'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def await_json(path,timeout_seconds=120):
    end=time.monotonic()+timeout_seconds
    while not path.exists():
        require(time.monotonic()<end,'Timed out waiting for owned launch handshake '+str(path))
        require(not (path.parent/'CANCEL_BEFORE_SIMULATIONAPP').exists(),'Coordinator cancelled before SimulationApp')
        time.sleep(.1)
    return read(path)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ticket',type=Path,required=True)
    parser.add_argument('--ticket-sha256',required=True)
    args=parser.parse_args(argv);ticket_path=args.ticket.resolve()
    require(ticket_path.is_relative_to(D) and sha256(ticket_path)==args.ticket_sha256,'Changed owned ticket')
    ticket=read(ticket_path);control=ticket_path.parent;output=Path(ticket['output']).resolve()
    require(ticket['schema']=='greenhouse.original848_sustained_worker_ticket.v2' and type(ticket['worker_count']) is int and ticket['worker_count'] in (2,3,4)
        and type(ticket['slot']) is int and 0<=ticket['slot']<ticket['worker_count'] and ticket['frames_per_worker']==1024
        and output==control/'capture' and not output.exists(),'Invalid bounded ticket output')
    require(sha256(producer.__file__)==FROZEN_PRODUCER_SHA,'Frozen capture implementation changed')
    verify_bindings(ticket['source_bindings'])
    app=None;succeeded=False;simulation_app_construction_started=False
    try:
        command=await_json(control/'command_identity.json')
        require(command['ticket_sha256']==args.ticket_sha256,'Command receipt names another ticket')
        rows=gate.snapshot();coordinator=ticket['coordinator_identity']
        require(gate.same_identity(gate.one(rows,coordinator['ProcessId']),coordinator),'Coordinator identity changed')
        gate.command_child(gate.one(rows,command['identity']['ProcessId']),coordinator,command['command'])
        require(gate.same_identity(gate.one(rows,command['identity']['ProcessId']),command['identity']),
            'Owned command identity changed')
        native=gate.native_child(gate.one(rows,os.getpid()),command['identity'],command['command'][1:])
        require(Path(sys.executable).resolve()==gate.KIT.resolve(),'Native bootstrap required')
        save_json(control/'native_ready.json',dict(slot=ticket['slot'],native_identity=native,
            command_identity=command['identity'],ticket_sha256=args.ticket_sha256,
            simulation_app_started=False,created_utc=datetime.now(timezone.utc).isoformat()))
        release=await_json(control/'release.json')
        require(release['ticket_sha256']==args.ticket_sha256 and release['slot']==ticket['slot']
            and release['coordinator_identity']==coordinator and release['worker_count']==ticket['worker_count'],'Changed release identity')
        own=next(s for s in release['slots'] if s['slot']==ticket['slot'])
        require(own['native_identity']==native and own['command_identity']==command['identity'],
            'Release does not name exact current native child')
        process=gate.inventory(gate.owned_allowlist(coordinator,release['slots']))
        require(not process['blockers'],'Unknown process or sibling identity at worker admission')
        require(release['resource_admission']['allowed'] is True,'Coordinator resource admission failed')
        require(sha256(ticket_path)==args.ticket_sha256,'Ticket changed during handshake')
        plan_path=Path(ticket['plan']['path']).resolve()
        require(sha256(plan_path)==ticket['plan']['sha256'],'Plan changed')
        plan=read(plan_path);require(plan['max_frames']==ticket['frames_per_worker']==1024,'Sustained shard is exactly1024 scheduled poses')
        output.mkdir()
        save_json(output/'request.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
            plan_path=str(plan_path),plan_sha256=ticket['plan']['sha256'],ticket_path=str(ticket_path),
            ticket_sha256=args.ticket_sha256,release_sha256=sha256(control/'release.json'),
            process_admission=process,host_memory_preflight=release['resource_admission']['memory'],
            available_disk_bytes=release['resource_admission']['disk_free_bytes'],
            automatic_retries=False,training_approved=False))
        # As in frozen v1: no USD/pxr plan evaluation before SimulationApp owns ABI.
        from isaacsim import SimulationApp
        require(sha256(plan['profile_evidence']['path'])==plan['profile_evidence']['sha256'],'Profile changed')
        profile=read(plan['profile_evidence']['path'])
        if profile['schema']==api.ORIGINAL_PROFILE_SCHEMA:
            config=dict(headless=True,width=848,height=408,multi_gpu=False,renderer='RaytracedLighting',
                sync_loads=False,disable_viewport_updates=True,extra_args=['--/app/settings/persistent=false'])
        else:
            config=dict(headless=True,width=848,height=408,multi_gpu=False,renderer='RealTimePathTracing',
                anti_aliasing=2,sync_loads=False,disable_viewport_updates=True,
                extra_args=['--/app/settings/persistent=false','--/rtx/rtpt/enabled=true'])
        simulation_app_construction_started=True
        save_json(control/'simulation_app_starting.json',dict(ticket_sha256=args.ticket_sha256,
            native_identity=native,simulation_app_construction_started=True,training_approved=False))
        app=SimulationApp(config)
        checked=api.check(plan)
        wrapper_bindings={**ticket['source_bindings'],str(ticket_path):args.ticket_sha256,
            **{str((control/name).resolve()):sha256(control/name) for name in
                ('command_identity.json','native_ready.json','release.json')}}
        for path,pin in wrapper_bindings.items():
            require(path not in checked['source_bindings'] or checked['source_bindings'][path]==pin,
                'Wrapper binding conflicts with immutable capture source')
            checked['source_bindings'][path]=pin
        verify_bindings(checked['source_bindings'])
        result=producer.capture(app,output,plan_path,plan,checked)
        require(sha256(plan_path)==ticket['plan']['sha256'],'Bulk plan changed during capture')
        result.update(plan_sha256=ticket['plan']['sha256'],bounded_parallel_slot=ticket['slot'],
            wrapper_sha256=sha256(__file__),native_identity=native)
        save_json(output/'result.json',result);succeeded=True
    except BaseException as exc:
        destination=output if output.is_dir() else control
        save_json(destination/'failure.json',dict(state='owned_original848_parallel_worker_failed',
            error=traceback.format_exc(),exception_type=type(exc).__name__,
            simulation_app_construction_started=simulation_app_construction_started,
            ticket_sha256=args.ticket_sha256,native_pid=os.getpid(),
            automatic_retries=False,training_approved=False,
            accepted_training_increment=0));raise
    finally:
        if app is not None:app.close(exit_code=0 if succeeded else 1)


if __name__=='__main__':main()
