"""Normalize completed serial original bulk Q3 pilots; no capture, labels or export."""
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
from .native848_bulk_plan_v1 import check_profile, RESULT_STATE
from .native_dataset.clear848_workspace_v2 import WorkspaceChecker
from .native848_diverse_q3_post_admission_v1 import artifact, bound_json, argument, q3_pass

SCHEMA='greenhouse.native848_diverse_bulk_q3_adapter.v1'
STATE='original_bulk_q3_normalized_pending_individual_review'
ROOT=Path('D:/research/tomato-pi-policy')
H=ROOT/'data/sim_data/diagnostics/collection_20k_848_20260916_v1'
OWNER_SHA='8e9131cc92a64d2e6c83ba7a5e0d52c8a1dd4f8b81adb6b2f0ede3cf6fa5bf0c'
FOLLOWER_SHA='ed10329d595c5e77ed560bcf470c883bb453aabf3bd2eb885766eed79a0e8483'
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


def load_owned(trial,cpu,native_result_sha256,cpu_result_sha256):
    trial=Path(trial).resolve();cpu=Path(cpu).resolve();capture=trial/'capture';pins={}
    for folder in (trial,capture,cpu,cpu/'shard_0'):
        require(not (folder/'failure.json').exists(),'Failed original/CPU run cannot normalize')
    nr=bound_json(trial/'result.json',pins,native_result_sha256)
    complete=bound_json(trial/'owner_complete.json',pins)
    require(nr['state']=='owned_original848_bulk_capture_exited_pending_independent_CPU_admission'
        and nr['training_approved'] is False and complete['result_sha256']==native_result_sha256
        and complete['serial_native_lock_released_after_this_receipt'] is True,'Whole serial owner not completed')
    intent=bound_json(trial/'intent.json',pins);launch=bound_json(trial/'launch.json',pins)
    owned=bound_json(trial/'owned_exit.json',pins,nr['owned_exit_sha256'])
    require(Path(nr['owned_exit_path']).resolve()==trial/'owned_exit.json'
        and owned['returncode']==0 and owned['method']=='subprocess_wait_on_owned_process'
        and owned['pid']==launch['pid'] and owned['launch_sha256']==sha256(trial/'launch.json'),
        'Native actual owned exit differs')
    require(intent['schema']=='greenhouse.original848_bulk_owner_intent.v1'
        and intent['automatic_retries'] is False and intent['training_approved'] is False
        and intent['source_bindings']==nr['source_bindings'],'Wrong serial original owner')
    owner=artifact(H/'run_native848_bulk_v2.py',pins,OWNER_SHA)
    require(intent['source_bindings'].get(owner['path'])==OWNER_SHA,'Unbound serial owner implementation')
    command=launch['command'];plan_path=Path(intent['plan_path']).resolve()
    require(argument(command,'-m')=='sim_data.native848_bulk_worker_v1'
        and Path(argument(command,'--output')).resolve()==capture
        and Path(argument(command,'--plan')).resolve()==plan_path
        and argument(command,'--plan-sha256')==intent['plan_sha256']==nr['plan_sha256']==launch['plan_sha256'],
        'Actual native command/plan differs')
    plan=bound_json(plan_path,pins,nr['plan_sha256'])
    require(plan['generated_geometry_used'] is False and plan['per_target_view_cap'] is None
        and plan['automatic_retries'] is False and plan['training_approved'] is False
        and 1<=len(plan['schedule'])<=32,'Small bounded original production plan required')
    review=bound_json(intent['review_path'],pins,intent['review_sha256'])
    require(review['owner_sha256']==OWNER_SHA and review['plan_sha256']==nr['plan_sha256']
        and review['native_launch_review_passed'] is True and review['blocking_findings']==[]
        and review['training_approved'] is False,'Native launch review differs')
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
    profile=bound_json(ctx['profile_evidence']['path'],pins,ctx['profile_evidence']['sha256']);check_profile(profile)
    cpu_intent=bound_json(cpu/'intent.json',pins)
    require(cpu_intent['shards']==1 and Path(cpu_intent['trial']).resolve()==trial
        and cpu_intent['CPU_only'] is True and cpu_intent['training_approved'] is False
        and cpu_intent['annotation_epoch']==delivery.FUTURE_EPOCH
        and cpu_intent['follower_sha256']==FOLLOWER_SHA
        and cpu_intent['consumer_sha256']==delivery.FUTURE_CONSUMER_SHA,'Exact one-shard Q3 follower required')
    artifact(H/'run_bulk_cpu_v5.py',pins,FOLLOWER_SHA)
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
    require(output.is_relative_to(ROOT/'data/sim_data/diagnostics') and not output.exists(),'New diagnostic output required')
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
            decision='candidate_pending_individual_visual_review_and_global_grouping' if eligible else 'preserved_'+row['decision'],
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
    result=dict(schema=SCHEMA,state=STATE,input_route='bulk_q3',created_utc=datetime.now(timezone.utc).isoformat(),
        implementation=artifact(__file__,pins),**primary,annotation_epoch=delivery.FUTURE_EPOCH,
        annotation_contract=contract,source_bindings=pins,records=records,excluded_records=excluded,
        proposed_population=len(plan['schedule']),native_frame_count=len(observed),candidate_count=len(records),
        excluded_count=len(excluded),original_admission_automated_candidates=sum(r['automated_pass'] for r in logged.values()),
        old_holds_revived=False,labels_recomputed=False,individual_visual_review=False,
        all_candidates_require_individual_visual_review=True,source_cap_reset=False,
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

