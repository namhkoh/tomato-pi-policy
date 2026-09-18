"""Admit against one actual snapshot, with evidence for naturally retired siblings.

Never weakens the frozen process classifier. A missing sibling requires its
real successful owned wait and absence of every recorded native identity.
"""
from pathlib import Path
import time
import copy
import os
import uuid
from datetime import datetime,timezone
from . import native848_bulk_sibling_gate_v5 as gate
from .native848_bulk_io_v1 import save_json
from .dataset_review import require,read_json
from .depth_preview import sha256

SCHEMA='greenhouse.native848_owned_retirement_snapshot.v2'
class ReceiptPending(Exception):pass


def partition(parent,coordinator,slots,rows,*,protected_slot=None):
    parent=Path(parent).resolve();by={r['ProcessId']:r for r in rows}
    require(len(by)==len(rows),'Duplicate OS process inventory')
    require(gate.same_identity(gate.one(rows,coordinator['ProcessId']),coordinator),'Coordinator missing or reused')
    gate.owned_allowlist(coordinator,slots)
    live=[];retired=[];pins={}
    def read(path):
        path=Path(path).resolve();pins[str(path)]=sha256(path);return read_json(path)
    for slot in slots:
        identities=[slot['command_identity'],slot['native_identity']]
        for identity in identities+slot.get('auxiliary_identities',[]):
            observed=by.get(identity['ProcessId'])
            require(observed is None or gate.same_identity(observed,identity),'Owned identity changed or PID reused')
        present=[r['ProcessId'] in by for r in identities]
        if all(present):live.append(slot);continue
        require(slot['slot']!=protected_slot,'Current admitting worker identity disappeared')
        folder=parent/f"slot_{slot['slot']}";exit_path=folder/'owned_exit.json'
        if not exit_path.exists():raise ReceiptPending('Actual owned wait is not yet published')
        owned=read(exit_path);launch=read(folder/'launch.json')
        require(owned['returncode']==0 and owned['method']=='subprocess_wait_on_owned_process'
            and owned['pid']==launch['pid']==slot['command_identity']['ProcessId']
            and owned['launch_sha256']==sha256(folder/'launch.json'), 'Successful exact owned child wait required')
        gate.command_child(slot['command_identity'],coordinator,launch['command'])
        gate.native_child(slot['native_identity'],slot['command_identity'],launch['command'][1:])
        ready=read(folder/'native_ready.json');ticket=read(folder/'ticket.json')
        require(ready['native_identity']==slot['native_identity'] and ready['command_identity']==slot['command_identity']
            and ready['ticket_sha256']==sha256(folder/'ticket.json') and ready['slot']==slot['slot']
            and ticket['slot']==slot['slot'] and ticket['coordinator_identity']==coordinator
            and Path(ticket['output']).resolve()==folder/'capture','Retirement launch/ticket identity differs')
        if any(present) or any(r['ProcessId'] in by for r in slot.get('auxiliary_identities',[])):
            raise ReceiptPending('Actual exited child descendants have not all disappeared')
        retired.append(dict(slot=slot['slot'],command_identity=slot['command_identity'],native_identity=slot['native_identity'],
            auxiliary_identities=copy.deepcopy(slot.get('auxiliary_identities',[])),
            owned_exit=dict(path=str(exit_path),sha256=sha256(exit_path)),launch=dict(path=str(folder/'launch.json'),sha256=sha256(folder/'launch.json')),
            successful_owned_wait_verified=True,all_recorded_identities_absent=True))
    return dict(live_slots=live,retired_slots=retired,source_bindings=pins)


def persist_failure(parent,coordinator,slots,rows,process,exc,phase,**details):
    """Create-only evidence from the exact failed snapshot; no new exemptions."""
    path=Path(parent).resolve()/('process_admission_failure_'+str(os.getpid())+'_'+uuid.uuid4().hex+'.json')
    save_json(path,dict(schema='greenhouse.native848_owned_process_failure_snapshot.v1',
        captured_utc=datetime.now(timezone.utc).isoformat(),phase=phase,
        exception_type=type(exc).__name__,error=str(exc),coordinator_identity=coordinator,
        recorded_slots=copy.deepcopy(slots),process_snapshot=rows,
        process_snapshot_available=rows is not None,process_inventory=process,
        same_snapshot_classification=process is not None,
        automatic_retry=False,process_control_performed=False,training_approved=False,**details))
    exc.add_note('Exact process failure snapshot: '+str(path)+' SHA256 '+sha256(path))
    return path


def register_preserving_telemetry(rows,coordinator,slots):
    """A naturally absent auxiliary remains recorded until the slot retires."""
    fresh=gate.register_telemetry(rows,coordinator,slots)
    previous={s['slot']:s for s in slots}
    for slot in fresh:
        known=previous[slot['slot']].get('auxiliary_identities',[])
        current=slot.get('auxiliary_identities',[])
        if known:
            require(not current or current==known,'Registered auxiliary identity may not be replaced or forgotten')
            slot['auxiliary_identities']=copy.deepcopy(known)
    gate.owned_allowlist(coordinator,fresh)
    return fresh


def inventory(parent,coordinator,slots,*,protected_slot=None,on_pending=None,timeout_seconds=10,register_telemetry=False,check_stop=None):
    require(0<timeout_seconds<=10,'Only bounded retirement-receipt waiting is permitted')
    end=time.monotonic()+timeout_seconds;rows=None;process=None
    try:
        while True:
            if on_pending is not None:on_pending()
            rows=gate.snapshot();process=None
            if check_stop is not None:check_stop()
            try:value=partition(parent,coordinator,slots,rows,protected_slot=protected_slot)
            except ReceiptPending:
                require(time.monotonic()<end,'Timed out awaiting actual successful retirement receipt')
                time.sleep(.1);continue
            if register_telemetry:value['live_slots']=register_preserving_telemetry(rows,coordinator,value['live_slots'])
            process=gate.inventory(gate.owned_allowlist(coordinator,value['live_slots']),rows=rows)
            require(process['no_blockers_observed'] is True and not process['blockers'],'Unknown process remains at admission')
            value.update(schema=SCHEMA,process_inventory=process,
                same_snapshot_classification=True,require_all_live_identities=True,
                bounded_receipt_wait_seconds=timeout_seconds,process_control_performed=False)
            return value
    except BaseException as exc:
        persist_failure(parent,coordinator,slots,rows,process,exc,'inventory',
            protected_slot=protected_slot,telemetry_registration_requested=register_telemetry)
        raise


def wait_completed_slot(parent,coordinator,slot,*,timeout_seconds=10,on_pending=None):
    """After a real owned wait, retain the entire record until every identity exits.

    The polling budget is at most ten seconds. Existing bounded OS inventory and
    resource query calls retain their own call timeouts; this is not a hard
    ten-second wall-clock guarantee if an external system query stalls.

    Nonzero exits can be reaped but never become successful admission receipts.
    The unchanged partition contract still requires returncode zero at admission.
    A bootstrap that never registered native_ready has only its exact command
    recorded; it must not have started SimulationApp or acquired auxiliaries.
    """
    require(0<timeout_seconds<=10,'Only bounded owned-retirement waiting is permitted')
    parent=Path(parent).resolve();folder=parent/f"slot_{slot['slot']}"
    rows=None;pins={};end=time.monotonic()+timeout_seconds
    def read(path):
        path=Path(path).resolve();pins[str(path)]=sha256(path);return read_json(path)
    try:
        command=gate.identity(slot['command_identity']);native=slot.get('native_identity')
        auxiliaries=slot.get('auxiliary_identities',[])
        launch=read(folder/'launch.json');owned=read(folder/'owned_exit.json')
        require(type(owned['returncode']) is int and owned['method']=='subprocess_wait_on_owned_process'
            and owned['pid']==launch['pid']==command['ProcessId']
            and owned['launch_sha256']==sha256(folder/'launch.json'),'Actual exact owned process wait required before retirement')
        gate.command_child(command,coordinator,launch['command'])
        identities=[command]
        if native is not None:
            gate.owned_allowlist(coordinator,[slot])
            gate.native_child(native,command,launch['command'][1:])
            ready=read(folder/'native_ready.json');ticket=read(folder/'ticket.json')
            require(ready['native_identity']==native and ready['command_identity']==command
                and ready['ticket_sha256']==sha256(folder/'ticket.json') and ready['slot']==slot['slot']
                and ticket['slot']==slot['slot'] and ticket['coordinator_identity']==coordinator
                and Path(ticket['output']).resolve()==folder/'capture','Retirement launch/ticket identity differs')
            identities.extend([native,*auxiliaries])
        else:
            require(not auxiliaries and not (folder/'native_ready.json').exists()
                and not (folder/'simulation_app_starting.json').exists(),
                'Unregistered bootstrap may not discard a native or auxiliary receipt')
        while True:
            rows=gate.snapshot();by={r['ProcessId']:r for r in rows}
            require(len(by)==len(rows),'Duplicate OS process inventory')
            require(gate.same_identity(gate.one(rows,coordinator['ProcessId']),coordinator),'Coordinator missing or reused')
            for identity in identities:
                observed=by.get(identity['ProcessId'])
                require(observed is None or gate.same_identity(observed,identity),'Owned identity changed or PID reused')
            if not any(identity['ProcessId'] in by for identity in identities):
                return dict(slot=slot['slot'],command_identity=command,
                    **({'native_identity':native} if native is not None else {}),
                    auxiliary_identities=copy.deepcopy(auxiliaries),owned_exit=dict(path=str(folder/'owned_exit.json'),sha256=sha256(folder/'owned_exit.json')),
                    launch=dict(path=str(folder/'launch.json'),sha256=sha256(folder/'launch.json')),
                    actual_owned_wait_verified=True,owned_returncode=owned['returncode'],
                    all_recorded_identities_absent=True,retirement_snapshot=rows,
                    bounded_retirement_wait_seconds=timeout_seconds,source_bindings=pins,
                    process_control_performed=False)
            if on_pending is not None:on_pending()
            require(time.monotonic()<end,'Timed out awaiting exit of all exact recorded owned identities')
            time.sleep(.1)
    except BaseException as exc:
        persist_failure(parent,coordinator,[slot],rows,None,exc,'completed_slot_retirement',
            bounded_retirement_wait_seconds=timeout_seconds)
        raise
