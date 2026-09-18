"""Admit against one actual snapshot, with evidence for naturally retired siblings.

Never weakens the frozen process classifier. A missing sibling requires its
real successful owned wait and absence of every recorded native identity.
"""
from pathlib import Path
import time
from . import native848_bulk_sibling_gate_v5 as gate
from .native848_bulk_io_v1 import save_json
from .dataset_review import require,read_json
from .depth_preview import sha256

SCHEMA='greenhouse.native848_owned_retirement_snapshot.v1'
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
            owned_exit=dict(path=str(exit_path),sha256=sha256(exit_path)),launch=dict(path=str(folder/'launch.json'),sha256=sha256(folder/'launch.json')),
            successful_owned_wait_verified=True,all_recorded_identities_absent=True))
    return dict(live_slots=live,retired_slots=retired,source_bindings=pins)


def inventory(parent,coordinator,slots,*,protected_slot=None,on_pending=None,timeout_seconds=10,register_telemetry=False):
    require(0<timeout_seconds<=10,'Only bounded retirement-receipt waiting is permitted')
    end=time.monotonic()+timeout_seconds
    while True:
        if on_pending is not None:on_pending()
        rows=gate.snapshot()
        try:value=partition(parent,coordinator,slots,rows,protected_slot=protected_slot)
        except ReceiptPending:
            require(time.monotonic()<end,'Timed out awaiting actual successful retirement receipt')
            time.sleep(.1);continue
        # The same snapshot drives both identity partitioning and the unchanged
        # foreign-process classifier; no broad exemptions or require_all=False.
        if register_telemetry:value['live_slots']=gate.register_telemetry(rows,coordinator,value['live_slots'])
        process=gate.inventory(gate.owned_allowlist(coordinator,value['live_slots']),rows=rows)
        require(process['no_blockers_observed'] is True and not process['blockers'],'Unknown process remains at admission')
        value.update(schema=SCHEMA,process_inventory=process,
            same_snapshot_classification=True,require_all_live_identities=True,
            bounded_receipt_wait_seconds=timeout_seconds,process_control_performed=False)
        return value
