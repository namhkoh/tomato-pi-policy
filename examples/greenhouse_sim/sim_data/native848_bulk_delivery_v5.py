"""Close immutable, contiguous committed prefixes while bulk capture continues.

This adds no renderer, frame admission, review decision or owner-exit claim.
Frozen CPU decisions are complete only after their newline-terminated log
record matches decision.json. The closure copies those completed decisions
atomically and verifies the actual source closure again before publication.
"""
from pathlib import Path
from fractions import Fraction
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
from .dataset_review import require, read_json, verify_bindings
from .depth_preview import sha256
from .capture_contract import fingerprint
from .native848_bulk_io_v1 import save_json
from .native848_bulk_plan_v1 import geometry_pose_key, check_profile
from .native848_bulk_visual_sampling_v1 import sample_bulk_chunk, PROTOCOL_SHA256
from . import native848_bulk_admission_v2 as admission
from . import native848_bulk_delivery_v1 as frozen

SCHEMA='greenhouse.original848_bulk_live_prefix_closure.v5'

LEGACY_EPOCH='greenhouse.native848_query_selection.v1'
FUTURE_EPOCH='greenhouse.native848_query_selection.v3'
LEGACY_CONSUMER_SHA='737eb13e93b78010b32be9b2854247d99959f0fb961327c70baa241e360bf0dd'
LEGACY_QUERY_SHA='896c81e5900ed31eb5a52989e15517a47d367b9aacb452bd8580e47260fe3964'
FUTURE_CONSUMER_SHA='0b17eddd6b3e7bd1883e3521b4b1311da0106d51bc09804e00e554d4f9ec9bbb'
FUTURE_QUERY_SHA='69bc1babbe462d7dee107c1a70832faaa688579cebe3493928119e98a7b5a462'
FUTURE_POLICY_REVIEW_SHA='7a6479039d976f87a27fc08a03c00d5dff77ba5baa5473f1314ee8e9b13f68f4'


def contract_for_epoch(epoch):
    root=Path(__file__).resolve().parent
    legacy_query=str(root/'native848_query_selection_v1.py')
    if epoch==LEGACY_EPOCH:
        consumer=str(root/'native848_bulk_admission_v2.py')
        return dict(annotation_epoch=epoch,consumer_sha256=LEGACY_CONSUMER_SHA,
            query_selector_sha256=LEGACY_QUERY_SHA,query_selector_dependencies={},
            source_bindings={consumer:LEGACY_CONSUMER_SHA,legacy_query:LEGACY_QUERY_SHA})
    require(epoch==FUTURE_EPOCH,'Unknown annotation epoch')
    review=root.parents[2]/'data/sim_data/diagnostics/native848_future_query_route_root_review_20260916_v1/result.json'
    return dict(annotation_epoch=epoch,consumer_sha256=FUTURE_CONSUMER_SHA,
        query_selector_sha256=FUTURE_QUERY_SHA,
        query_selector_dependencies={legacy_query:LEGACY_QUERY_SHA},
        source_bindings={str(root/'native848_bulk_admission_v4.py'):FUTURE_CONSUMER_SHA,
            str(root/'native848_query_selection_v3.py'):FUTURE_QUERY_SHA,legacy_query:LEGACY_QUERY_SHA,
            str(review):FUTURE_POLICY_REVIEW_SHA})


def annotation_contract(request,pins):
    if request['consumer_sha256']==LEGACY_CONSUMER_SHA:
        require(all(k not in request for k in ('annotation_epoch','query_selector_sha256','query_selector_dependencies')),
            'Frozen legacy request cannot masquerade as a new annotation contract')
        contract=contract_for_epoch(LEGACY_EPOCH)
    else:
        contract=contract_for_epoch(FUTURE_EPOCH)
        require(all(request.get(k)==contract[k] for k in
            ('consumer_sha256','annotation_epoch','query_selector_sha256','query_selector_dependencies')),
            'Explicit future consumer/epoch/selector/dependency binding differs')
    verify_bindings(contract['source_bindings'])
    for path,pin in contract['source_bindings'].items():add_pin(pins,path,pin)
    return contract


def check_annotation(saved,row,contract):
    epoch=contract['annotation_epoch'];label=saved['label'];trace=saved['query_trace'];selection=saved['query_selection']
    require(contract==contract_for_epoch(epoch),'Unknown annotation contract fields')
    require(label.get('annotation_epoch')==epoch and selection['schema']==epoch,
        'Label/selector epoch differs from the actual CPU request')
    if epoch==FUTURE_EPOCH:
        from . import native848_query_selection_v3 as query
        require(selection['annotation_epoch']==epoch and selection['separation_policy']==query.POLICY
            and selection['previous_selection_schema']==LEGACY_EPOCH,
            'Future separation policy or preserved selector provenance changed')
    if trace is None:
        require(label['eligible'] is False and row['automated_pass'] is False
            and selection['selected_query'] is None,'Missing trace may only remain an excluded local label')
        return
    expected='greenhouse.native848_query_trace.'+('v1' if epoch==LEGACY_EPOCH else 'v3')
    require(trace['schema']==expected and trace['annotation_epoch']==epoch
        and trace['query_pixel_uv']==label['query_pixel_uv'],'Composite trace epoch/query changed')
    selected=selection['selected_query']
    if trace['passed']:
        require(selected is not None and selected['both_passed'] is True and selected['local_passed'] is True
            and trace['legacy_trace']['passed'] is True and trace['fixed_grid']['passed'] is True
            and selected['legacy_trace']==trace['legacy_trace'] and selected['fixed_grid']==trace['fixed_grid']
            and selected['query_pixel_uv']==label['query_pixel_uv'],'Preserved local/legacy/grid query gates differ')
        if epoch==FUTURE_EPOCH:
            route=trace['route_separation']
            require(selected['all_checks_passed'] is True and route is not None and route['passed'] is True
                and route['schema']=='greenhouse.native848_query_route_separation.v1'
                and selected['route_separation']==route
                and selection['separation_pass_count']==trace['separation_pass_count']>=1,
                'Future route separation did not pass')
    else:
        require(row['automated_pass'] is False and selected is None
            and trace['original_query_retained_as_failed_fallback'] is True,
            'Failed fallback query cannot enter a passing population')
    if row['automated_pass']:require(trace['passed'] is True,'Eligible row lacks passed trace')


def read_hold_snapshot(checkpoint):
    path=Path(checkpoint).resolve()/'visual_holds.jsonl'
    existed=path.exists();raw=path.read_bytes() if existed else b''
    if raw and not raw.endswith(b'\n'):raise PrefixPending('Hold ledger has an incomplete append record')
    records=[json.loads(line) for line in raw.splitlines() if line]
    for row in records:frozen.aliases(row)
    return dict(source_ledger_path=str(path),source_existed=existed,
        source_ledger_sha256_at_read=hashlib.sha256(raw).hexdigest(),
        observed_utc=datetime.now(timezone.utc).isoformat(),records=records,raw=raw)


def publish_hold_snapshot(output,snapshot,pins):
    if snapshot is None:return None
    path=Path(output)/'checkpoint_visual_holds.snapshot.jsonl'
    with path.open('xb') as stream:stream.write(snapshot['raw'])
    pin=sha256(path);require(pin==snapshot['source_ledger_sha256_at_read'],'Hold snapshot bytes changed')
    add_pin(pins,path,pin)
    return dict(snapshot_path=str(path.resolve()),snapshot_sha256=pin,
        source_ledger_path=snapshot['source_ledger_path'],source_existed=snapshot['source_existed'],
        source_ledger_sha256_at_read=snapshot['source_ledger_sha256_at_read'],
        observed_utc=snapshot['observed_utc'],hold_records=len(snapshot['records']),
        immutable_snapshot=True,live_source_path_is_informational_not_a_binding=True,
        current_holds_rechecked_at_export=True)


def export_chunk(inventory_path,inventory_sha256,review_path,review_sha256,output):
    """Validate the explicit epoch, then reuse frozen actual-QA/global-hold export."""
    pins={}
    inventory=stable_json(inventory_path,pins,inventory_sha256)
    closure=stable_json(inventory['prefix_closure_path'],pins,inventory['prefix_closure_sha256'])
    require(closure['schema']==SCHEMA and closure['records']==inventory['all_capture_rows']
        and [r for r in closure['records'] if r['automated_pass']]==inventory['records']
        and closure['context_sha256']==inventory['context_sha256']
        and closure['run_plan_sha256']==inventory['run_plan_sha256']
        and closure['annotation_contract']==inventory['annotation_contract'],
        'Explicit new delivery closure/annotation contract required')
    verify_bindings(closure['source_bindings'])
    contract=inventory['annotation_contract']
    require(contract==contract_for_epoch(contract['annotation_epoch']),'Changed annotation contract')
    verify_bindings(contract['source_bindings'])
    snapshot=closure['checkpoint_hold_snapshot']
    require(snapshot==inventory['checkpoint_hold_snapshot'],'Hold snapshot lineage differs')
    if snapshot is not None:
        path=Path(snapshot['snapshot_path']).resolve()
        require(sha256(path)==snapshot['snapshot_sha256']
            and closure['source_bindings'].get(str(path))==snapshot['snapshot_sha256']
            and snapshot['source_ledger_path'] not in closure['source_bindings'],
            'Immutable hold snapshot missing or mutable ledger still bound')
    for row in inventory['all_capture_rows']:
        saved={k:stable_json(row[k+'_path'],pins,row[k+'_sha256']) for k in ('label','query_trace','query_selection')}
        check_annotation(saved,row,contract)
    # frozen.export_chunk enforces the unchanged presealed sample and actual QA,
    # then freshly reads checkpoint_rows AND checkpoint_holds before asset links.
    return frozen.export_chunk(inventory_path,inventory_sha256,review_path,review_sha256,output)



def parallel_scope(intent,ticket,scheduled_frames):
    """Explicit old probe or bounded sustained contract; no implicit expansion."""
    count=intent['native_workers'];slot=ticket['slot']
    require(type(count) is int and type(slot) is int
        and intent['automatic_expansion'] is False,'Explicit integer worker scope required')
    if intent['schema']=='greenhouse.original848_two_worker_probe.v1':
        require(count==2 and intent['planned_frames']==64 and scheduled_frames==32
            and ticket['schema']=='greenhouse.original848_two_worker_ticket.v1',
            'Frozen two-worker probe scope changed')
    elif intent['schema'] in ('greenhouse.original848_sustained_workers.v1',
            'greenhouse.original848_sustained_workers.v2'):
        version=intent['schema'].rsplit('.',1)[-1]
        allowed_counts=(2,3) if version=='v1' else (2,3,4)
        require(count in allowed_counts and intent['frames_per_worker']==1024
            and scheduled_frames==1024 and intent['planned_frames']==count*1024
            and ticket['schema']=='greenhouse.original848_sustained_worker_ticket.'+version
            and type(ticket['worker_count']) is int and ticket['worker_count']==count
            and ticket['frames_per_worker']==1024,'Explicit sustained worker scope changed')
    else:
        raise ValueError('Unknown parallel coordinator contract')
    require(0<=slot<count,'Ticket slot outside declared worker count')
    return count


def release_scope(release,intent,ticket,count):
    """Validate the declared slots; process identity equality is checked below."""
    if intent['schema'] in ('greenhouse.original848_sustained_workers.v1',
            'greenhouse.original848_sustained_workers.v2'):
        require(type(release['worker_count']) is int and release['worker_count']==count,
            'Release worker count differs from coordinator')
        completed=release['completed_slots']
        require(isinstance(completed,list),'Explicit completed sibling inventory required')
    else:
        completed=release.get('completed_slots',[])
    live=release['slots'];slots=[r['slot'] for r in live];done=[r['slot'] for r in completed]
    require(all(type(v) is int and 0<=v<count for v in slots+done)
        and len(set(slots))==len(slots) and len(set(done))==len(done)
        and not set(slots)&set(done) and ticket['slot'] in slots,
        'Invalid, repeated or live/completed overlapping declared slot')
    return completed


class PrefixPending(ValueError):
    """A scheduled producer decision or completed CPU frame is not available."""


def stable_json(path, pins, expected=None):
    path=Path(path).resolve()
    if not path.is_file():raise PrefixPending('Not yet committed: '+str(path))
    raw=path.read_bytes();pin=hashlib.sha256(raw).hexdigest()
    try:value=json.loads(raw)
    except (ValueError,UnicodeError) as exc:raise PrefixPending('Incomplete JSON: '+str(path)) from exc
    require(expected is None or pin==expected,'Changed bound JSON: '+str(path))
    require(sha256(path)==pin,'JSON changed during closure: '+str(path))
    add_pin(pins,path,pin)
    return value


def add_pin(pins,path,pin):
    name=str(Path(path).resolve())
    require(name not in pins or pins[name]==pin,'Conflicting source pin: '+name)
    pins[name]=pin


def pin_json(pin,pins):
    return stable_json(pin['path'],pins,pin['sha256'])


def committed_cpu_records(folder,pins):
    """Read only complete append records; an unfinished trailing line is pending.

The mutable append log is deliberately not hash-bound as a whole. Each selected
complete line must equal the immutable per-frame decision, which is hash-bound
and copied into the atomic prefix closure. Later log appends are irrelevant.
"""
    folder=Path(folder).resolve()
    require(not (folder/'failure.json').exists(),'CPU shard failed; cannot close prefix')
    request=stable_json(folder/'request.json',pins)
    require(request['training_approved'] is False,'Unexpected CPU approval')
    annotation_contract(request,pins)
    path=folder/'decisions.jsonl'
    if not path.exists():return request,{}
    rows={}
    for raw in path.read_bytes().splitlines(keepends=True):
        if not raw.endswith(b'\n'):continue
        row=json.loads(raw)
        name=row['observation_id']
        require(name not in rows,'Repeated complete CPU append record')
        rows[name]=row
    return request,rows


def continuity(before,after):
    if before is None:return
    a,b=before['synchronization'],after['synchronization']
    ta,tb=a['freshness'],b['freshness']
    require(b['request_index']>a['request_index']
        and b['callback_sequence']>a['callback_sequence']
        and tb['callback_sequence']>ta['callback_sequence']
        and Fraction(*b['reference_time'])>=Fraction(*a['reference_time']),
        'Native request/callback/reference continuity failed across prefix')
    require(all(tb[k]!=ta[k] for k in ('camera_sha256','rgb_sha256','depth_sha256')),
        'Adjacent native camera/buffer freshness failed across prefix')


def check_observation(obs,row,native,item,ctx,profile,pins,contract):
    require(obs['schema']=='greenhouse.original848_bulk_observation.v1'
        and obs['capture_role']=='production' and obs['split']=='train'
        and obs['training_approved'] is False and obs['source_family']==ctx['source_family']
        and obs['profile']==profile['profile'],'Wrong original production observation')
    for key in ('observation_id','source_pose_id','capture_index'):
        require(obs[key]==row[key]==native[key]==item[key],'Scheduled frame identity differs: '+key)
    require(row['target_id']==obs['target_id']==native['target_id']
        and row['source_target']==obs['target_id'] and row['source_family']==obs['source_family']
        and row['split']=='train' and row['training_approved'] is False,'Original target/split changed')
    sync,request=obs['synchronization'],obs['request_evidence']
    require(sync['static_before']==sync['static_after'] and bool(sync['static_before'])
        and request['native_requests']==request['callback_count']==1
        and request['request_index']==sync['request_index']
        and request['callback_sequence_after']==sync['callback_sequence']==sync['freshness']['callback_sequence']
        and request['reference_time']==sync['reference_time'],'Actual static/single callback evidence differs')
    require(request['requested_subframes']==profile['request_subframes']
        and request['render_settings']==profile['render_settings']
        and request['reset_api']==profile['reset_api'] and request['reset_returned_without_exception'] is True
        and request['wait_for_render'] is profile['wait_for_render'] is True
        and request['delta_time_seconds']==profile['delta_time_seconds']
        and request['native_reference_time_policy']==profile['native_reference_time_policy']
        and request['scheduler_clock_policy']==profile['scheduler_clock_policy']
        and abs(request['timeline_after_seconds']-request['timeline_before_seconds']-profile['delta_time_seconds'])<1e-7,
        'Actual renderer/reset/clock policy changed')
    proof=pin_json(obs['geometry_proof'],pins);pose=obs['robot_snapshot']
    key=geometry_pose_key(ctx['scene_revision'],pose,ctx['source_bindings_sha256'])
    require(obs['geometry_proof']['pose_key']==proof['pose_key']==key
        and proof['scene_revision']==ctx['scene_revision']
        and proof['source_bindings_sha256']==ctx['source_bindings_sha256']
        and proof['joint_degrees']==pose['joint_degrees']
        and proof['robot_root_to_world_usd_row_vectors']==pose['robot_root_to_world_usd_row_vectors']
        and proof['screen']==pose['visual_bound_screen'] and proof['screen']['passed'] is True,
        'Current full robot/scene geometry proof changed')
    pin_json(obs['mapping'],pins)
    for pin in obs['files'].values():
        require(sha256(pin['path'])==pin['sha256'],'Changed native frame buffer')
        add_pin(pins,pin['path'],pin['sha256'])
    require(row['rgb_sha256']==obs['files']['rgb']['sha256']
        and Path(row['rgb_path']).resolve()==Path(obs['files']['rgb']['path']).resolve(),'RGB binding differs')
    saved={}
    for kind in ('label','sample','workspace','query_trace','query_selection'):
        saved[kind]=pin_json(dict(path=row[kind+'_path'],sha256=row[kind+'_sha256']),pins)
    require(sha256(row['target_mask_path'])==row['target_mask_sha256'],'Target mask changed')
    add_pin(pins,row['target_mask_path'],row['target_mask_sha256'])
    sample=saved['sample'];label=saved['label'];workspace=saved['workspace']
    require(sample['observation_sha256']==row['observation_sha256']
        and Path(sample['observation_path']).resolve()==Path(row['observation_path']).resolve()
        and sample['synchronization']['freshness']==sync['freshness']
        and sample['robot_snapshot']==pose and sample['calibration']==obs['calibration'],
        'CPU sample no longer matches actual observation')
    require(row['label_metrics']==label.get('clarity',{})
        and row['query_metrics']==label.get('query_usability',{}),'Sampling metrics changed')
    check_annotation(saved,row,contract)
    if row['automated_pass']:
        require(row['decision']=='automated_candidate_pending_batch_QA'
            and label['eligible'] is True and workspace['result']['workspace_passed'] is True
            and saved['query_trace']['passed'] is True
            and label['target_id']==workspace['target_id']==row['target_id'],'Changed automated gate')


def geometry_hold_proof(capture,ctx,cached,pins):
    """Find the actual rejected full-pose proof, never a fabricated frame."""
    matches=[]
    for path in Path(ctx['geometry_proofs_directory']).glob('*.json'):
        value=read_json(path)
        if (value['joint_degrees']==cached['joint_degrees'] and
            np.allclose(value['robot_root_to_world_usd_row_vectors'],
                cached['robot_root_to_world_usd_row_vectors'],atol=1e-9,rtol=0)):
            pose={k:value[k] for k in ('joint_degrees','robot_root_to_world_usd_row_vectors')}
            require(value['pose_key']==geometry_pose_key(ctx['scene_revision'],pose,ctx['source_bindings_sha256'])
                and value['scene_revision']==ctx['scene_revision']
                and value['source_bindings_sha256']==ctx['source_bindings_sha256']
                and value['screen']['passed'] is False,'Invalid native geometry rejection proof')
            matches.append(dict(path=str(path.resolve()),sha256=sha256(path)))
    require(len(matches)==1,'Unique actual rejected geometry proof required')
    pin_json(matches[0],pins)
    return matches[0]


def prepare_review_prefix(context_path,context_sha256,admissions,prior,prior_sha256,output,
        protocol_path,protocol_sha256,*,capture_start,capture_stop,producer_intent,producer_launch,
        previous_closure=None,checkpoint=None):
    require(type(capture_start) is int and type(capture_stop) is int
        and 0<=capture_start<capture_stop and capture_stop-capture_start<=256,
        'Explicit contiguous range of at most256 scheduled indices required')
    output=Path(output).resolve();require(not output.exists(),'New immutable prefix output required')
    pins={};ctx=stable_json(context_path,pins,context_sha256);capture=Path(context_path).resolve().parent
    require(ctx['schema']=='greenhouse.original848_bulk_context.v1'
        and ctx['split']=='train' and ctx['training_approved'] is False,'Original TRAIN context required')
    plan=pin_json(dict(path=ctx['plan_path'],sha256=ctx['plan_sha256']),pins)
    require(plan['generated_geometry_used'] is False and plan['per_target_view_cap'] is None
        and plan['automatic_retries'] is False and plan['training_approved'] is False,
        'Qualified uncapped original plan required')
    require(ctx['profile_evidence']==plan['profile_evidence'],'Context profile differs')
    profile=pin_json(ctx['profile_evidence'],pins);check_profile(profile)
    require(ctx['source_bindings_sha256']==fingerprint(ctx['source_bindings']),'Flat source binding hash differs')
    for sources in (ctx['source_bindings'],profile['source_bindings'],plan['implementation_bindings']):
        for path,pin in sources.items():add_pin(pins,path,pin)
    intent=pin_json(producer_intent,pins);launch=pin_json(producer_launch,pins)
    require(launch['plan_sha256']==ctx['plan_sha256'] and intent['automatic_retries'] is False
        and intent['training_approved'] is False and launch['training_approved'] is False
        and type(launch['pid']) is int and launch['pid']>0,'Actual producer plan/launch differs')
    command=launch['command']
    if '--ticket' in command:
        ticket_path=Path(command[command.index('--ticket')+1]).resolve()
        require('--ticket-sha256' in command and ticket_path.parent==capture.parent,
            'Actual slot ticket does not own this capture')
        ticket_hash=command[command.index('--ticket-sha256')+1]
        ticket=stable_json(ticket_path,pins,ticket_hash)
        worker_count=parallel_scope(intent,ticket,len(plan['schedule']))
        require(ticket['plan']['sha256']==ctx['plan_sha256']
            and Path(ticket['plan']['path']).resolve()==Path(ctx['plan_path']).resolve()
            and Path(ticket['output']).resolve()==capture
            and ticket['coordinator_identity']==intent['coordinator_identity']
            and ticket['training_approved'] is False,'Actual parallel ticket/context differs')
        control=ticket_path.parent
        release=stable_json(control/'release.json',pins)
        completed_slots=release_scope(release,intent,ticket,worker_count)
        for completed in completed_slots:
            sibling=control.parent/f"slot_{completed['slot']}"
            require(Path(completed['owned_exit']['path']).resolve()==sibling/'owned_exit.json'
                and Path(completed['launch']['path']).resolve()==sibling/'launch.json',
                'Retired sibling receipt outside its actual slot')
            exited=pin_json(completed['owned_exit'],pins);retired_launch=pin_json(completed['launch'],pins)
            require(exited['method']=='subprocess_wait_on_owned_process'
                and exited['pid']==completed['command_identity']['ProcessId']==retired_launch['pid']
                and exited['launch_sha256']==completed['launch']['sha256']
                and completed['native_identity']['ProcessId']!=completed['command_identity']['ProcessId'],
                'Retired sibling owned-exit/identity binding differs')
        command_identity=stable_json(control/'command_identity.json',pins)
        native_ready=stable_json(control/'native_ready.json',pins)
        require(release['ticket_sha256']==native_ready['ticket_sha256']==command_identity['ticket_sha256']==ticket_hash
            and release['slot']==native_ready['slot']==ticket['slot']
            and release['coordinator_identity']==ticket['coordinator_identity']
            and release['resource_admission']['allowed'] is True
            and command_identity['command']==command
            and command_identity['identity']['ProcessId']==launch['pid']
            and native_ready['command_identity']==command_identity['identity'],
            'Parallel actual process/ticket/release binding differs')
        own=[r for r in release['slots'] if r['slot']==ticket['slot']]
        require(len(own)==1 and own[0]['native_identity']==native_ready['native_identity']
            and own[0]['command_identity']==command_identity['identity'],'Parallel native identity differs')
        for path in (ticket_path,control/'release.json',control/'command_identity.json',control/'native_ready.json'):
            require(ctx['source_bindings'].get(str(path.resolve()))==pins[str(path.resolve())],
                'Actual context did not bind slot proof')
        for path,pin in ticket['source_bindings'].items():add_pin(pins,path,pin)
        require(Path(producer_intent['path']).resolve()==control.parent/'intent.json',
            'Parallel producer intent is not actual coordinator parent')
    else:
        require(intent['plan_sha256']==ctx['plan_sha256']
            and Path(intent['plan_path']).resolve()==Path(ctx['plan_path']).resolve()
            and '--plan-sha256' in command and command[command.index('--plan-sha256')+1]==ctx['plan_sha256']
            and '--output' in command and Path(command[command.index('--output')+1]).resolve()==capture,
            'Actual serial producer command does not match this capture')
    for path,pin in intent['source_bindings'].items():add_pin(pins,path,pin)
    for module in (admission,frozen):add_pin(pins,module.__file__,sha256(module.__file__))
    add_pin(pins,__file__,sha256(__file__))
    old=stable_json(prior,pins,prior_sha256)
    require(old['frozen_family_splits'].get(ctx['source_family'])=='train','Held-out family cannot enter TRAIN')
    stable_json(protocol_path,pins,protocol_sha256)
    require(protocol_sha256==PROTOCOL_SHA256,'Presealed sampling protocol changed')
    cache=pin_json(dict(path=plan['cache_path'],sha256=plan['cache_sha256']),pins)
    poses={r['sample_id']:r for r in cache['records']}
    schedule=plan['schedule'];require(capture_stop<=len(schedule),'Range exceeds actual schedule')
    items=[]
    for index in range(capture_start,capture_stop):
        item=dict(schedule[index],capture_index=index)
        require(item['source_pose_id'] in poses,'Unknown scheduled pose');items.append(item)
    previous=None;previous_pin=None
    if capture_start:
        require(previous_closure is not None,'A nonzero prefix requires its immediately preceding sealed closure')
        previous_receipt=pin_json(previous_closure,pins)
        require(previous_receipt['schema'] in (SCHEMA,'greenhouse.original848_bulk_live_prefix_closure.v2',
            'greenhouse.original848_bulk_live_prefix_closure.v3','greenhouse.original848_bulk_live_prefix_closure.v4') and previous_receipt['context_sha256']==context_sha256
            and previous_receipt['capture_stop_exclusive']==capture_start
            and previous_receipt['run_plan_sha256']==ctx['plan_sha256'],'Prefix gap, overlap or changed context')
        previous_pin=previous_receipt['terminal_observation']
        if previous_pin:previous=pin_json(previous_pin,pins)
    else:require(previous_closure is None,'First prefix cannot carry a previous closure')
    seen=set();hold_snapshot=None
    for row in old['records']+old['preserved_hold_identity_rows']:seen.update(frozen.aliases(row))
    if checkpoint is not None:
        folder=Path(checkpoint).resolve()
        for path in [folder/'baseline.json',*sorted((folder/'chunks').glob('*.json'))]:
            stable_json(path,pins)
        hold_snapshot=read_hold_snapshot(folder)
        for row in frozen.checkpoint_rows(folder)+hold_snapshot['records']:seen.update(frozen.aliases(row))
    by={};active_contract=None
    for folder in admissions:
        request,logged=committed_cpu_records(folder,pins)
        contract=annotation_contract(request,pins)
        require(active_contract is None or active_contract==contract,'Mixed annotation epochs/consumers across shards')
        active_contract=contract
        require(request['context_sha256']==context_sha256 and request['prior_sha256']==prior_sha256
            and Path(request['context_path']).resolve()==Path(context_path).resolve(), 'CPU shard scope differs')
        for sources in (request['source_bindings'],request['workspace_bindings']):
            for path,pin in sources.items():add_pin(pins,path,pin)
        for name,row in logged.items():
            require(name not in by,'Repeated CPU decision across shards')
            by[name]=(row,Path(folder).resolve()/name/'decision.json',contract)
    require(active_contract is not None,'At least one explicit CPU annotation shard required')
    require(not any(Path(p).name=='visual_holds.jsonl' for p in pins),'Mutable hold ledger cannot be a closure source binding')
    verify_bindings(pins)  # Fresh initial flat source check; repeated at closure.
    rows=[];native_rows=[];geometry_holds=[];decision_copies=[]
    for item in items:
        name=item['observation_id'];native_path=capture/('decision_'+name+'.json')
        native=stable_json(native_path,pins)
        require(all(native[k]==item[k] for k in ('observation_id','source_pose_id','capture_index'))
            and native['target_id']==poses[item['source_pose_id']]['target_id']
            and native['training_approved'] is False,'Native scheduled decision differs')
        native_rows.append(dict(path=str(native_path),sha256=pins[str(native_path)],record=native))
        if native['state']=='rejected_native_geometry':
            require(native['native_requests']==0 and name not in by
                and not (capture/'frames'/name/'observation.json').exists(),'Rejected pose cannot also be a native frame')
            proof=geometry_hold_proof(capture,ctx,poses[item['source_pose_id']],pins)
            geometry_holds.append(dict(**native,geometry_proof=proof));continue
        require(native['state']=='queued_lossless_callback_pending_cpu_admission' and native['native_requests']==1,
            'Failed, stopped, unknown or retried native decision cannot close a prefix')
        if name not in by:raise PrefixPending('CPU frame not yet complete: '+name)
        logged,decision_path,contract=by[name];row=stable_json(decision_path,pins)
        require(row==logged,'Completed CPU append record differs from decision')
        obs_path=capture/'frames'/name/'observation.json'
        require(Path(row['observation_path']).resolve()==obs_path,'CPU observation is outside actual capture')
        obs=stable_json(obs_path,pins,row['observation_sha256'])
        require(obs['context_sha256']==context_sha256
            and Path(obs['context_path']).resolve()==Path(context_path).resolve(),'Frame context differs')
        check_observation(obs,row,native,item,ctx,profile,pins,contract);continuity(previous,obs)
        previous=obs;previous_pin=dict(path=str(obs_path),sha256=row['observation_sha256'])
        decision_copies.append((name,row))
        row=dict(row)
        if frozen.aliases(row)&seen:
            row.update(automated_pass=False,decision='exclude_global_duplicate_or_preserved_hold')
        seen.update(frozen.aliases(row));rows.append(row)
    # One flat fresh source verification at closure; no old control replay and
    # no forged successful process-exit/whole-batch result.
    verify_bindings(pins)
    require(not (capture/'failure.json').exists(),'Capture failed before prefix closure')
    for folder in admissions:require(not (Path(folder)/'failure.json').exists(),'CPU shard failed before prefix closure')
    chunk_id=f'prefix_{capture_start:08d}_{capture_stop:08d}'
    eligible=[r for r in rows if r['automated_pass']]
    chosen=sample_bulk_chunk(eligible,ctx['plan_sha256'],chunk_id,startup=capture_start==0,all_capture_rows=rows)
    output.mkdir(parents=True,exist_ok=False)
    hold_snapshot_receipt=publish_hold_snapshot(output,hold_snapshot,pins)
    snapshots=[]
    (output/'cpu_decisions').mkdir()
    for name,row in decision_copies:
        path=output/'cpu_decisions'/f'{name}.json';save_json(path,row)
        snapshots.append(dict(path=str(path),sha256=sha256(path),observation_id=name))
    closure=dict(schema=SCHEMA,state='immutable_committed_prefix_pending_actual_sampled_visual_QA',
        created_utc=datetime.now(timezone.utc).isoformat(),capture_start_inclusive=capture_start,
        capture_stop_exclusive=capture_stop,scheduled_indices=capture_stop-capture_start,
        context_path=str(Path(context_path).resolve()),context_sha256=context_sha256,
        run_plan_sha256=ctx['plan_sha256'],producer_intent=producer_intent,producer_launch=producer_launch,
        previous_closure=previous_closure,terminal_observation=previous_pin,
        native_decisions=native_rows,native_frames=len(rows),native_geometry_holds=geometry_holds,
        cpu_decision_snapshots=snapshots,records=rows,automated_candidates=len(eligible),
        annotation_contract=active_contract,checkpoint_hold_snapshot=hold_snapshot_receipt,
        source_bindings=pins,flat_source_bindings_verified_at_closure=True,
        CPU_completion_method='complete newline-terminated append record equals stable decision.json; atomic immutable closure copy',
        owner_completion_required=False,owner_completion_claimed=False,owned_exit_receipt=None,
        acceptance_scope='this contiguous committed prefix only',training_approved=False,accepted_training_increment=0)
    save_json(output/'closure.json',closure)
    inventory=dict(schema='greenhouse.original848_bulk_visual_chunk.v1',chunk_id=chunk_id,
        protocol_path=str(Path(protocol_path).resolve()),protocol_sha256=protocol_sha256,
        run_plan_sha256=ctx['plan_sha256'],context_sha256=context_sha256,trial=str(capture.parent),
        owned_result_sha256=None,owner_completion_claimed=False,acceptance_scope='committed_prefix_only',
        prefix_closure_path=str(output/'closure.json'),prefix_closure_sha256=sha256(output/'closure.json'),
        records=eligible,all_capture_rows=rows,native_geometry_holds=geometry_holds,selection=chosen,
        annotation_contract=active_contract,checkpoint_hold_snapshot=hold_snapshot_receipt,
        training_approved=False,accepted_training_increment=0)
    save_json(output/'inventory.json',inventory)
    result=dict(schema='greenhouse.original848_bulk_live_prefix_review.v5',
        closure_path=str(output/'closure.json'),closure_sha256=sha256(output/'closure.json'),
        inventory_path=str(output/'inventory.json'),inventory_sha256=sha256(output/'inventory.json'),
        capture_start_inclusive=capture_start,capture_stop_exclusive=capture_stop,
        native_frames=len(rows),native_geometry_holds=len(geometry_holds),automated_candidates=len(eligible),
        actual_QA_complete=False,owner_completion_claimed=False,training_approved=False,accepted_training_increment=0)
    save_json(output/'result.json',result)
    return result


# The frozen exporter keeps actual review/pixel bindings, global duplicate and
# preserved-hold exclusions, portable artifact copies and the20k total cap.
# export_chunk above adds explicit annotation and immutable-hold checks.


