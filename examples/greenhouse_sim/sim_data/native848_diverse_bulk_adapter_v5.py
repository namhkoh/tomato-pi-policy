"""Normalize actually completed parallel multi-anchor slots; unchanged per-image checks."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import numpy as np
from PIL import Image
from .dataset_review import require, read_json, write_json, verify_bindings
from .depth_preview import sha256
from .capture_contract import fingerprint
from .training_export import view_signature
from . import native848_bulk_delivery_v5 as delivery
from . import native848_data_roots_v1 as roots
from .native848_bulk_plan_v1 import check_profile, RESULT_STATE
from .native_dataset.clear848_workspace_v2 import WorkspaceChecker
from .native848_diverse_q3_post_admission_v1 import artifact, bound_json, argument, q3_pass

SCHEMA='greenhouse.native848_diverse_bulk_q3_adapter.v5'
STATE='original_bulk_q3_normalized_pending_group_or_individual_review'
ROOT=Path('D:/research/tomato-pi-policy')
H=ROOT/'data/sim_data/diagnostics/collection_20k_848_20260916_v1'
OWNER_SHA='bd0e3d54ef08f7189366f845962e7deda1ff45e29aa6de3f17dbf7dbe06972fa'
WORKER_SHA='5abb477b94f8d32ff1f52969e38fd1a102b1d9956bb0c245147f11d517abe793'
GATE_SHA='3766afb02212dfa0bb432bb61c6dc9b3e45d7fbd259d778be6e7bc2b9a7150d4'
PREDECESSOR_SHA='b2c9e49a2afb8a951064ad1bbbe66ed4039f14f34e5e27b5c2dcebf0aebe17ee'
FOLLOWER_SHA='54d077837ece88c6612bdef5b2b10076c167b7939daa73e6b251f8be4df8e955'
DELIVERY_SHA='6216c14c28463b3c139d5612f74f37c58fe2e61997d2d188c7186be5b84b4595'


def workspace_replay(meta, workspace, checker, sample_sha, workspace_sha):
    require(workspace['schema']=='greenhouse.native848_bulk_workspace.v1'
        and workspace['target_id']==meta['supervision']['target_id']
        and workspace['nominal_world_m']==meta['supervision']['nominal_world_m']
        and workspace['per_frame_camera_FK_verified'] is True,
        'Bulk workspace identity/camera proof differs')
    pose=checker.pose_and_probe(meta)
    inputs={k:np.asarray(pose[k],dtype=float).tolist() for k in ('base','torso','left','right','probe')}
    target=np.asarray(meta['supervision']['nominal_world_m'],dtype=float)
    require(target.shape==(3,) and np.isfinite(target).all(),'Finite actual target required')
    inputs['target']=target.tolist()
    key=hashlib.sha256(json.dumps(inputs,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    require(key==workspace['solve_input_sha256']
        and pose['probe'].tolist()==workspace['probe_ee_m'],'Changed exact workspace solve input')
    passed=workspace['result']['workspace_passed'] is True
    require(workspace['per_frame_cached_solution_FK_verified'] is passed,'Workspace success proof differs')
    if passed:
        joints=np.asarray(workspace['result']['candidate']['joint_degrees'],dtype=float)
        actual=(checker.model.forward('left',joints,pose['base'],pose['torso']) @ np.r_[pose['probe'],1])[:3]
        require(float(np.linalg.norm(actual-target))<.001
            and checker.model.arm_joint_limit_margin_degrees('left',joints)>=0
            and checker.model.inter_arm_clearance(joints,pose['right'],pose['base'],pose['torso']).clearance_m>=0,
            'Saved workspace solution fails actual FK/limits/opposite arm')
    return dict(schema='greenhouse.native848_diverse_bulk_workspace_replay.v1',
        sample_sha256=sample_sha,workspace_sha256=workspace_sha,solve_input_sha256=key,
        per_frame_camera_FK_verified=True,per_frame_cached_solution_FK_verified=passed,
        workspace_passed=passed,new_IK_solve=False,training_approved=False)


def validate_balanced_plan(plan,pins):
    """Bound total work and require multiple physical targets before image admission."""
    from collections import Counter
    cache=bound_json(plan['cache_path'],pins,plan['cache_sha256'])
    by={r['sample_id']:r for r in cache['records']}
    require(len(by)==len(cache['records']), 'Duplicate cached sample identity')
    ids=[r['source_pose_id'] for r in plan['schedule']]
    require(len(ids)==len(set(ids)) and set(ids)<=set(by), 'Unique scheduled source poses required')
    counts=Counter(by[x]['target_id'] for x in ids)
    require('seed41_full/SubStem_38' not in counts, 'Rejected dominant target remains excluded')
    require(len(counts)>=2 and max(counts.values())<=128
        and max(counts.values())*2<=len(ids), 'Balanced multi-target batch required: at least2 targets, maximum128 new pertarget and50percent perbatch')
    require(all(by[x]['source_target']==by[x]['target_id'] and by[x]['split']=='train' for x in ids),
        'Physical target or TRAIN identity differs')
    return dict(counts)


def validate_parallel_shape(intent, parent_result, slot):
    count=intent['native_workers']
    require(type(count) is int and count in (2,3,4) and type(slot) is int and 0<=slot<count,
        'Explicit declared worker count/slot differs')
    counts=intent['frame_counts']
    require(len(counts)==len(intent['plans'])==count and all(type(n) is int and 2<=n<=512 for n in counts)
        and intent['planned_frames']==sum(counts) and parent_result['frame_counts']==counts
        and parent_result['planned_frames']==sum(counts) and parent_result['requested_worker_count']==count,
        'Declared bounded parallel schedule differs')
    require(parent_result['schema']==intent['schema']=='greenhouse.native848_multianchor_parallel.v2'
        and parent_result['state']=='owned_sustained_bulk_slots_exited_pending_CPU_and_visual_QA'
        and all(v[k] is False for v in (intent,parent_result) for k in
            ('automatic_retries','automatic_expansion','training_approved')),
        'Unsupported parent completion or implicit expansion')
    for key in ('results','completed_slots'):
        entries=parent_result[key];ids=[e['slot'] for e in entries]
        require(len(ids)==len(set(ids)) and all(type(i) is int and 0<=i<count for i in ids),
            'Duplicate or undeclared parent slot')
    require(slot in {r['slot'] for r in parent_result['results']}
        and slot in {r['slot'] for r in parent_result['completed_slots']}, 'Current slot did not actually finish')
    require(parent_result['committed_frames']==sum(r['committed_frames'] for r in parent_result['results']),
        'Whole parent count differs')


def load_owned(trial,cpu,native_result_sha256,cpu_result_sha256):
    trial=Path(trial).resolve();cpu=Path(cpu).resolve();capture=trial/'capture';pins={};parent=trial.parent
    require(roots.diagnostic(trial)==trial and roots.diagnostic(cpu)==cpu and trial.name.startswith('slot_'),
        'Actual parallel slot directory required')
    for folder in (parent,trial,capture,cpu,cpu/'shard_0'):
        require(not (folder/'failure.json').exists(),'Failed parent/native/CPU run cannot normalize')
    parent_result=bound_json(parent/'result.json',pins)
    parent_complete=bound_json(parent/'owner_complete.json',pins)
    require(parent_complete['result_sha256']==sha256(parent/'result.json')
        and parent_complete['whole_parallel_owner_complete'] is True
        and parent_complete['all_owned_command_children_waited'] is True
        and parent_complete['training_approved'] is False,'Whole parallel parent not completed')
    intent=bound_json(parent/'intent.json',pins)
    artifact(roots.__file__,pins,'b3b5d6c77385b2776a6997dd7dbe14f1aae8e1d093a45140d0f2c033fc2c8d83')
    for path,h in roots.bindings().items():
        artifact(path,pins,h);require(intent['source_bindings'].get(path)==h,'Unbound explicit data-root contract')
    env=Path(__file__).with_name('native848_storage_environment_v1.py')
    artifact(env,pins,'826f3e3c85e886d9c6b4e21fbdf8164d9eb3e1b50c395fb91c3ef90ed456667b')
    require(intent['source_bindings'].get(str(env.resolve()))==pins[str(env.resolve())],'Unbound actual storage environment implementation')
    require(parent_result['intent']==dict(path=str(parent/'intent.json'),sha256=sha256(parent/'intent.json'))
        and parent_result['source_bindings']==intent['source_bindings']
        and parent_result['coordinator_identity']==intent['coordinator_identity'], 'Parent immutable intent differs')
    ticket=bound_json(trial/'ticket.json',pins);slot=ticket['slot']
    validate_parallel_shape(intent,parent_result,slot)
    require(trial.name=='slot_'+str(slot),'Slot path differs')
    request=bound_json(intent['request']['path'],pins,intent['request']['sha256'])
    review=bound_json(intent['launch_review']['path'],pins,intent['launch_review']['sha256'])
    require(request['schema']==intent['schema'] and request['worker_count']==intent['native_workers']
        and request['frame_counts']==intent['frame_counts'] and request['plans']==intent['plans']
        and request['automatic_retries'] is False and request['automatic_expansion'] is False
        and request['training_approved'] is False,'Exact parent request differs')
    require(review['owner_sha256']==OWNER_SHA and review['worker_sha256']==WORKER_SHA
        and review['gate_sha256']==GATE_SHA and review['request_sha256']==intent['request']['sha256']
        and review['native_worker_count']==intent['native_workers']
        and review['native_launch_review_passed'] is True and review['blocking_findings']==[]
        and review['training_approved'] is False,'Exact parallel launch review differs')
    for path,h in [(H/'run_native848_multianchor_parallel_v2.py',OWNER_SHA),
        (Path(__file__).with_name('native848_multianchor_sibling_worker_v2.py'),WORKER_SHA),
        (Path(__file__).with_name('native848_bulk_sibling_gate_v5.py'),GATE_SHA),
        (Path(__file__).with_name('native848_multianchor_parallel_predecessor_v2.py'),PREDECESSOR_SHA)]:
        artifact(path,pins,h);require(intent['source_bindings'].get(str(path.resolve()))==h,'Unbound parallel owner chain')
    for spec in (intent['request'],intent['launch_review']):
        require(intent['source_bindings'].get(str(Path(spec['path']).resolve()))==spec['sha256'],'Unbound launch authorization')
    result_entry=next(r for r in parent_result['results'] if r['slot']==slot)
    require(result_entry['owner_result']==dict(path=str(trial/'result.json'),sha256=native_result_sha256),
        'Parent does not bind current slot result')
    nr=bound_json(trial/'result.json',pins,native_result_sha256)
    complete=bound_json(trial/'owner_complete.json',pins)
    require(nr['state']=='owned_original848_bulk_capture_exited_pending_independent_CPU_admission'
        and nr['training_approved'] is False and nr['bounded_parallel_slot']==slot
        and nr['all_owned_native_children_waited'] is True and nr['source_bindings']==intent['source_bindings']
        and complete['result_sha256']==native_result_sha256 and complete['parallel_slot_only'] is True
        and complete['training_approved'] is False and result_entry['committed_frames']==nr['committed_frames'],
        'Actual slot completion differs; no serial-lock claim permitted')
    launch=bound_json(trial/'launch.json',pins)
    owned=bound_json(trial/'owned_exit.json',pins,nr['owned_exit_sha256'])
    require(Path(nr['owned_exit_path']).resolve()==trial/'owned_exit.json'
        and owned['returncode']==0 and owned['method']=='subprocess_wait_on_owned_process'
        and owned['pid']==launch['pid'] and owned['launch_sha256']==sha256(trial/'launch.json'),
        'Native actual owned exit differs')
    completed=next(r for r in parent_result['completed_slots'] if r['slot']==slot)
    require(completed['owned_exit']==dict(path=str(trial/'owned_exit.json'),sha256=sha256(trial/'owned_exit.json'))
        and completed['launch']==dict(path=str(trial/'launch.json'),sha256=sha256(trial/'launch.json')),
        'Actual retired identities/exit evidence differs')
    # Check every actually launched sibling has an owned exit, even a safely skipped bootstrap.
    for lp in sorted(parent.glob('slot_*/launch.json')):
        la=bound_json(lp,pins);ex=bound_json(lp.parent/'owned_exit.json',pins)
        require(ex['pid']==la['pid'] and ex['launch_sha256']==sha256(lp)
            and ex['method']=='subprocess_wait_on_owned_process', 'Unwaited actual sibling launch')
    command=launch['command'];plan_spec=intent['plans'][slot];plan_path=Path(plan_spec['path']).resolve()
    require(ticket['schema']=='greenhouse.native848_multianchor_parallel_ticket.v2'
        and ticket['worker_count']==intent['native_workers'] and ticket['frames_per_worker']==intent['frame_counts'][slot]
        and ticket['plan']==plan_spec and Path(ticket['output']).resolve()==capture
        and ticket['coordinator_identity']==intent['coordinator_identity']
        and ticket['source_bindings']==intent['source_bindings'] and ticket['training_approved'] is False
        and argument(command,'-m')=='sim_data.native848_multianchor_sibling_worker_v2'
        and Path(argument(command,'--ticket')).resolve()==trial/'ticket.json'
        and argument(command,'--ticket-sha256')==sha256(trial/'ticket.json')
        and plan_spec['sha256']==nr['plan_sha256']==launch['plan_sha256'],'Actual slot ticket/command/plan differs')
    identities=bound_json(trial/'command_identity.json',pins)
    ready=bound_json(trial/'native_ready.json',pins);release=bound_json(trial/'release.json',pins)
    own=[r for r in release['slots'] if r['slot']==slot]
    require(len(own)==1 and own[0]['command_identity']==identities['identity']==completed['command_identity']
        and own[0]['native_identity']==ready['native_identity']==completed['native_identity']
        and ready['command_identity']==identities['identity']
        and identities['identity']['ProcessId']==launch['pid'] and identities['command']==command
        and identities['identity']['ParentProcessId']==intent['coordinator_identity']['ProcessId']
        and ready['native_identity']['ParentProcessId']==identities['identity']['ProcessId']
        and ready['slot']==release['slot']==slot and ready['simulation_app_started'] is False
        and release['worker_count']==intent['native_workers']
        and release['coordinator_identity']==intent['coordinator_identity']
        and all(v['ticket_sha256']==sha256(trial/'ticket.json') for v in (identities,ready,release))
        and release['resource_admission']['allowed'] is True,'Parallel identity/release chain differs')
    native_request=bound_json(capture/'request.json',pins)
    require(native_request['ticket_sha256']==sha256(trial/'ticket.json')
        and Path(native_request['ticket_path']).resolve()==trial/'ticket.json'
        and native_request['release_sha256']==sha256(trial/'release.json')
        and native_request['plan_sha256']==nr['plan_sha256']
        and Path(native_request['plan_path']).resolve()==plan_path
        and not native_request['process_admission']['blockers']
        and native_request['process_admission']['no_blockers_observed'] is True,'Capture was not released by this parent')
    plan=bound_json(plan_path,pins,nr['plan_sha256'])
    require(plan['generated_geometry_used'] is False and plan['per_target_view_cap'] is None
        and plan['automatic_retries'] is False and plan['training_approved'] is False
        and 2<=len(plan['schedule'])==ticket['frames_per_worker']<=512,'Bounded original production plan required')
    validate_balanced_plan(plan,pins)
    native=bound_json(capture/'result.json',pins,nr['capture_result_sha256'])
    manifest=bound_json(capture/'batch_manifest.json',pins,nr['manifest_sha256'])
    ctx=bound_json(capture/'batch_context.json',pins,nr['context_sha256'])
    require(all(Path(nr[k+'_path']).resolve()==capture/v for k,v in
        [('capture_result','result.json'),('manifest','batch_manifest.json'),('context','batch_context.json')]),
        'Owner points outside actual capture')
    require(native['state']==RESULT_STATE and native['plan_sha256']==nr['plan_sha256']
        and native['manifest_sha256']==nr['manifest_sha256']
        and Path(native['manifest_path']).resolve()==capture/'batch_manifest.json'
        and manifest['schema']=='greenhouse.original848_bulk_manifest.v1'
        and manifest['context_sha256']==nr['context_sha256']
        and Path(manifest['context_path']).resolve()==capture/'batch_context.json'
        and manifest['source_assets_unchanged'] is True and manifest['automatic_retries'] is False
        and manifest['native_request_count']==manifest['native_callback_count']
        and native['committed_frames']==nr['committed_frames']==manifest['committed_frames']==len(manifest['observations']),
        'Native immutable completion/manifest differs')
    require(ctx['schema']=='greenhouse.original848_bulk_context.v1' and ctx['split']=='train'
        and ctx['training_approved'] is False and ctx['plan_sha256']==nr['plan_sha256']
        and Path(ctx['plan_path']).resolve()==plan_path
        and ctx['source_bindings_sha256']==fingerprint(ctx['source_bindings'])
        and ctx['profile_evidence']==plan['profile_evidence'],'Actual original context differs')
    require(native['bounded_parallel_slot']==slot and native['wrapper_sha256']==WORKER_SHA
        and native['native_identity']==ready['native_identity'],'Actual capture wrapper differs')
    for path in (trial/'ticket.json',trial/'command_identity.json',trial/'native_ready.json',trial/'release.json'):
        require(ctx['source_bindings'].get(str(path))==sha256(path),'Unbound actual admission handshake')
    profile=bound_json(ctx['profile_evidence']['path'],pins,ctx['profile_evidence']['sha256']);check_profile(profile)
    cpu_intent=bound_json(cpu/'intent.json',pins)
    require(cpu_intent['data_root_source_bindings']==roots.bindings()
        and cpu_intent['diagnostic_root_policy']==roots.SCHEMA,'CPU data-root contract differs')
    require(cpu_intent['shards']==1 and Path(cpu_intent['trial']).resolve()==trial
        and cpu_intent['CPU_only'] is True and cpu_intent['training_approved'] is False
        and cpu_intent['annotation_epoch']==delivery.FUTURE_EPOCH
        and cpu_intent['follower_sha256']==FOLLOWER_SHA
        and cpu_intent['consumer_sha256']==delivery.FUTURE_CONSUMER_SHA,'Exact one-shard Q3 follower required')
    artifact(H/'run_bulk_cpu_v6.py',pins,FOLLOWER_SHA)
    cr=bound_json(cpu/'result.json',pins,cpu_result_sha256)
    require(cr['state']=='CPU_bulk_shards_completed_pending_combined_review'
        and [Path(p).resolve() for p in cr['shard_paths']]==[cpu/'shard_0']
        and cr['context_sha256']==nr['context_sha256']
        and Path(cr['context_path']).resolve()==capture/'batch_context.json','CPU owner scope differs')
    cpu_launch=bound_json(cpu/'launch_0.json',pins);cpu_exit=bound_json(cpu/'owned_exit_0.json',pins)
    require(cr['owned_exits']==[cpu_exit] and cpu_exit['returncode']==0
        and cpu_exit['method']=='subprocess_wait_on_owned_process'
        and cpu_exit['pid']==cpu_launch['pid'] and cpu_exit['launch_sha256']==sha256(cpu/'launch_0.json'),
        'CPU actual owned exit differs')
    cmd=cpu_launch['command']
    require(argument(cmd,'-m')=='sim_data.native848_bulk_admission_v4'
        and argument(cmd,'--shard-count')=='1' and argument(cmd,'--shard-index')=='0'
        and Path(argument(cmd,'--output')).resolve()==cpu/'shard_0'
        and Path(argument(cmd,'--frames')).resolve()==capture/'frames'
        and Path(argument(cmd,'--context')).resolve()==capture/'batch_context.json'
        and argument(cmd,'--context-sha256')==nr['context_sha256'],'Actual CPU command differs')
    admission=bound_json(cpu/'shard_0/result.json',pins)
    require(admission['schema']=='greenhouse.original848_bulk_admission.v4'
        and admission['annotation_epoch']==delivery.FUTURE_EPOCH
        and admission['state']=='automated_frame_checks_complete_pending_batch_QA_and_owned_capture_completion'
        and admission['training_approved'] is False,'Completed genuine Q3 admission required')
    request,logged=delivery.committed_cpu_records(cpu/'shard_0',pins)
    contract=delivery.annotation_contract(request,pins)
    require(contract['annotation_epoch']==delivery.FUTURE_EPOCH
        and request['context_sha256']==nr['context_sha256']
        and Path(request['context_path']).resolve()==capture/'batch_context.json'
        and request['prior_sha256']==cpu_intent['prior_sha256']==argument(cmd,'--prior-sha256')
        and Path(request['prior_path']).resolve()==Path(cpu_intent['prior']).resolve()==Path(argument(cmd,'--prior')).resolve(),
        'CPU prior/context/Q3 request differs')
    prior=bound_json(request['prior_path'],pins,request['prior_sha256'])
    require(prior['frozen_family_splits'].get(ctx['source_family'])=='train','Non-TRAIN original family')
    require(len(admission['records'])==len(logged)==admission['native_frames_checked']==manifest['committed_frames']
        and {r['observation_id']:r for r in admission['records']}==logged
        and admission['automated_candidates']==sum(r['automated_pass'] for r in logged.values()),
        'Complete CPU population differs')
    for sources in (nr['source_bindings'],ctx['source_bindings'],profile['source_bindings'],
                    plan['implementation_bindings'],request['source_bindings'],request['workspace_bindings']):
        for path,pin in sources.items():delivery.add_pin(pins,path,pin)
    artifact(delivery.__file__,pins,DELIVERY_SHA)
    artifact(Path(__file__).with_name('native848_diverse_q3_post_admission_v1.py'),pins,
        '9e9dc67df6addc2f33ef02665e57763b35066c3d827cbb5353a8ccb03ea6c3a7')
    artifact(__file__,pins)
    verify_bindings(pins)
    return trial,cpu,plan,manifest,ctx,profile,contract,logged,pins


def run(trial,cpu,*,native_result_sha256,cpu_result_sha256,output):
    output=Path(output).resolve()
    require(roots.diagnostic(output)==output and not output.exists(),'New diagnostic output required')
    trial,cpu,plan,manifest,ctx,profile,contract,logged,pins=load_owned(
        trial,cpu,native_result_sha256,cpu_result_sha256)
    capture=trial/'capture';checker=WorkspaceChecker()
    for path,pin in checker.bindings.items():delivery.add_pin(pins,path,pin)
    cache=bound_json(plan['cache_path'],pins,plan['cache_sha256'])
    poses={r['sample_id']:r for r in cache['records']}
    observed={r['observation_id']:r for r in manifest['observations']}
    require(len(observed)==len(manifest['observations']) and set(observed)==set(logged),'Manifest/CPU IDs differ')
    require(len(manifest['decisions'])==len(plan['schedule']),'Native scheduled population incomplete')
    records=[];excluded=[];previous=None;actual_ids=set();replays={}
    for index,entry in enumerate(plan['schedule']):
        item=dict(entry,capture_index=index);name=item['observation_id']
        require(name not in actual_ids,'Repeated scheduled observation');actual_ids.add(name)
        native=bound_json(capture/('decision_'+name+'.json'),pins)
        require(native==manifest['decisions'][index] and native['capture_index']==index
            and native['observation_id']==name and native['source_pose_id']==item['source_pose_id']
            and native['target_id']==poses[item['source_pose_id']]['target_id'],'Actual scheduled decision differs')
        if native['state']=='rejected_native_geometry':
            require(native['native_requests']==0 and name not in logged
                and not (capture/'frames'/name/'observation.json').exists(),'Geometry rejection contains a frame')
            proof=delivery.geometry_hold_proof(capture,ctx,poses[item['source_pose_id']],pins)
            excluded.append(dict(sample_id=name,input_route='bulk_q3',original_decision=native['state'],
                decision='preserved_native_geometry_hold',geometry_proof=proof,
                candidate_for_individual_visual_review=False,training_approved=False));continue
        require(native['state']=='queued_lossless_callback_pending_cpu_admission'
            and native['native_requests']==1 and name in logged,'Unknown/stopped native decision')
        row=bound_json(cpu/'shard_0'/name/'decision.json',pins)
        require(row==logged[name],'CPU immutable decision differs from completed result')
        obs_path=capture/'frames'/name/'observation.json';commit=observed[name]
        require(Path(commit['path']).resolve()==Path(row['observation_path']).resolve()==obs_path
            and commit['sha256']==row['observation_sha256'],'Manifest observation pin differs')
        obs=bound_json(obs_path,pins,commit['sha256'])
        require(obs['context_sha256']==sha256(capture/'batch_context.json')
            and Path(obs['context_path']).resolve()==capture/'batch_context.json','Observation context differs')
        delivery.check_observation(obs,row,native,item,ctx,profile,pins,contract)
        delivery.continuity(previous,obs);previous=obs
        saved={k:bound_json(row[k+'_path'],pins,row[k+'_sha256']) for k in
               ('sample','label','query_trace','query_selection','workspace')}
        meta=saved['sample'];rgb=np.asarray(Image.open(row['rgb_path']).convert('RGB'))
        require(rgb.shape==(408,848,3) and hashlib.sha256(rgb.tobytes()).hexdigest()==row['decoded_rgb_sha256']
            and view_signature(meta,row['source_family'],None)==row['conservative_camera_signature'],
            'Native RGB/camera identity changed')
        replay=workspace_replay(meta,saved['workspace'],checker,row['sample_sha256'],row['workspace_sha256'])
        eligible=row['automated_pass'] is True
        require(not eligible or (q3_pass(saved['label'],saved['query_trace'],saved['query_selection'])
            and replay['workspace_passed']),'Prior candidate no longer passes actual Q3/workspace')
        artifacts={k:artifact(row[k+'_path'],pins,row[k+'_sha256']) for k in
                   ('rgb','sample','label','workspace','query_trace','query_selection','target_mask')}
        for k in ('buffers',):artifacts[k]=artifact(obs['files'][k]['path'],pins,obs['files'][k]['sha256'])
        artifacts.update(observation=artifact(obs_path,pins,commit['sha256']),
            context=artifact(capture/'batch_context.json',pins),mapping=artifact(obs['mapping']['path'],pins,obs['mapping']['sha256']))
        normalized=dict(row,id=name,sample_id=name,input_route='bulk_q3',
            source_plant_family=row['source_family'],source_sample_path=row['sample_path'],
            source_sample_sha256=row['sample_sha256'],artifacts=artifacts,
            bulk_workspace_replay=replay,original_decision=row['decision'],
            decision='candidate_pending_authorized_visual_review_and_global_grouping' if eligible else 'preserved_'+row['decision'],
            prior_candidate_for_individual_visual_review=eligible,candidate_for_individual_visual_review=eligible,
            workspace_passed=replay['workspace_passed'],render_profile_qualified=True,
            local_clarity_passed=saved['label']['eligible'] is True,q3_passed=q3_pass(saved['label'],saved['query_trace'],saved['query_selection']),
            full_trace_and_anchored_grid_passed=bool(saved['query_trace'] and saved['query_trace']['passed']),
            route_separation_passed=bool(saved['query_trace'] and (saved['query_trace'].get('route_separation') or {}).get('passed')),
            annotation_epoch=delivery.FUTURE_EPOCH,source_cap_reset=False,geometry_novelty_qualified=False,
            individual_visual_review=False,training_approved=False,accepted_training_increment=0)
        (records if eligible else excluded).append(normalized);replays[name]=replay
    require(set(observed)<=actual_ids and len(records)+len(excluded)==len(plan['schedule']),'Unaccounted pilot population')
    checker.finish();verify_bindings(pins)
    output.mkdir(parents=True,exist_ok=False)
    proof_dir=output/'workspace_replays';proof_dir.mkdir()
    for row in records+excluded:
        if row['sample_id'] not in replays:continue
        path=proof_dir/(row['sample_id']+'.json');write_json(path,replays[row['sample_id']])
        row['artifacts']['workspace_replay']=artifact(path,pins)
    primary=dict(original_admission=artifact(cpu/'shard_0/result.json',pins),
        native_result=artifact(trial/'result.json',pins,native_result_sha256),
        owner_complete=artifact(trial/'owner_complete.json',pins),cpu_owner_result=artifact(cpu/'result.json',pins,cpu_result_sha256),
        native_owned_exit=artifact(trial/'owned_exit.json',pins),context=artifact(capture/'batch_context.json',pins),
        manifest=artifact(capture/'batch_manifest.json',pins))
    parent=trial.parent
    primary.update(parallel_parent_result=artifact(parent/'result.json',pins),
        parallel_parent_complete=artifact(parent/'owner_complete.json',pins),
        parallel_parent_intent=artifact(parent/'intent.json',pins),
        parallel_slot_ticket=artifact(trial/'ticket.json',pins),parallel_slot_release=artifact(trial/'release.json',pins))
    user_policy=artifact(ROOT/'data/sim_data/diagnostics/native848_group_review_user_policy_20260917_v1/policy.json',pins)
    policy=read_json(user_policy['path'])
    require(policy['preserve_per_image_automated_gates'] is True and policy['group_representatives_and_flagged_visual_review_required'] is True
        and policy['individual_review_of_every_image_required'] is False and policy['excluded_target']=='seed41_full/SubStem_38',
        'Explicit user-approved group review policy required')
    result=dict(schema=SCHEMA,state=STATE,input_route='bulk_q3',created_utc=datetime.now(timezone.utc).isoformat(),
        implementation=artifact(__file__,pins),**primary,annotation_epoch=delivery.FUTURE_EPOCH,
        annotation_contract=contract,source_bindings=pins,records=records,excluded_records=excluded,
        proposed_population=len(plan['schedule']),native_frame_count=len(observed),candidate_count=len(records),
        excluded_count=len(excluded),original_admission_automated_candidates=sum(r['automated_pass'] for r in logged.values()),
        old_holds_revived=False,labels_recomputed=False,individual_visual_review=False,
        all_candidates_require_individual_visual_review=False,authorized_group_or_individual_review_required=True,
        review_policy=user_policy,source_cap_reset=False,native_source_route='completed_parallel_multianchor_slot',
        geometry_novelty_qualified=False,training_approved=False,accepted_training_increment=0)
    verify_bindings(pins);write_json(output/'result.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('trial','cpu','native-result-sha256','cpu-result-sha256','output'):parser.add_argument('--'+key,required=True)
    args=parser.parse_args()
    value=run(args.trial,args.cpu,native_result_sha256=args.native_result_sha256,
        cpu_result_sha256=args.cpu_result_sha256,output=args.output)
    print('DIVERSE_BULK_Q3_NORMALIZED',value['candidate_count'],value['excluded_count'])

