"""Offline full144 original-plant sweep annotation; no target-query shortcut.

One actual owner and one saved static scene are authenticated once. Every frame
uses the unchanged all-petiole9mm, verified joint, fixed-body workspace, and
ambiguity routines. Proposal identity never chooses the final answer.
"""
from pathlib import Path
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import argparse,json
import numpy as np
from PIL import Image
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256
from .capture_contract import fingerprint
from .native848_bulk_io_v1 import save_json
from .native848_bulk_workspace_v1 import BulkWorkspace
from .native848_bulk_plan_v1 import check_profile
from .native848_pair_audit_v2 import verify_camera
from .native_greenhouse_pair import assert_same_camera
from . import native848_scene_sweep_capture_v1 as producer
from . import native848_pilot_annotation_v16 as original
from . import native848_fully_labeled_coverage_v3 as coverage
from . import native848_all_petiole_9mm_v2 as labels
from . import native848_unique9mm_ambiguity_v3 as ambiguity
from . import background_scoring_v1 as background

SCHEMA='greenhouse.native848_scene_sweep_9mm_evaluation.v1'
FROZEN={original.__file__:'c3b709eba54f0ca65e23251165ea1021d4c203019369ee9aad3f089e756b2fc3',
    coverage.__file__:'8ca7167e0973f297bc23c129484e0e379794f713dcc1bdb104ceceb6adfc9536',
    labels.__file__:'c5704371017b9f5943664e8a4d156d5e7f2e6d71f8b48ef8e5f4efee01ae13c1',
    labels.joint_ownership.__file__:'37f7b121202a3e04c45166ab47c8e976bc7d4c74cfa546d1ac1ae65bf4d2f6aa',
    ambiguity.__file__:'2f9936a03bf2b405dd97fac19988afcbb4abe13612055c65c13098291d44754d',
    background.__file__:'fce68b2343e90b744aac4ed3d9eea2b18ad191f2e98e016670fce795ddb2dec6'}
evaluate_frame=coverage.evaluate_frame
assess_frame=ambiguity.assess_frame
geometry_9mm=labels.geometry_9mm


def validate_geometry(actual,expected,path=()):
    """Exact structure/semantic fields; frozen1nm metric and1e-6px roundoff bounds."""
    if isinstance(expected,dict):
        require(isinstance(actual,dict) and set(actual)==set(expected),'Geometry field population differs')
        for key in expected:validate_geometry(actual[key],expected[key],path+(key,))
    elif isinstance(expected,list):
        require(isinstance(actual,list) and len(actual)==len(expected),'Geometry sample population differs')
        for a,b in zip(actual,expected):validate_geometry(a,b,path)
    elif type(expected) in (int,float):
        require(type(actual) in (int,float) and np.isfinite(actual) and np.isfinite(expected), 'Invalid geometry numeric value')
        tolerance=1e-6 if 'pixel_xy' in path else 1e-9
        require(abs(actual-expected)<=tolerance,'Current source geometry exceeds frozen numeric roundoff bound')
    else:
        require(type(actual) is type(expected) and actual==expected,'Geometry semantic field differs')


def unique(rows,key):
    indexed={r[key]:r for r in rows};require(len(indexed)==len(rows),'Duplicate '+key);return indexed


def validate_ledger(result,records,observations,profile):
    """Authenticate every scheduled decision and the single writer's global clock."""
    planned=unique(records,'sample_id');frames=unique(result['frames'],'observation_id')
    holds=unique(result['holds'],'sample_id');names=[o['sample_id'] for o in observations]
    require(not(set(frames)&set(holds)) and set(frames)|set(holds)==set(planned)
        and names==[r['sample_id'] for r in records if r['sample_id'] in frames]
        and names==list(frames),'Complete ordered capture/hold partition required')
    require(all(h['reason']=='whole_robot_scene_collision' and h['screen']['passed'] is False for h in holds.values()),
        'Only actual collision holds supported')
    steps=list(profile['warmup_steps'])+[profile['request_subframes']]*len(observations)
    require(result['request_count']==result['callback_count']==len(result['requests'])==len(steps)
        and result['warmup_requests']==result['requests'][:len(profile['warmup_steps'])],
        'One actual warmup sequence plus production ledger required')
    previous_ref=None;previous_timeline=None
    for i,(event,subframes) in enumerate(zip(result['requests'],steps),1):
        require(event['request_index']==event['callback_sequence_after']==i
            and event['callback_sequence_before']==i-1 and event['native_requests']==event['callback_count']==1
            and event['requested_subframes']==subframes and event['delta_time_seconds']==profile['delta_time_seconds']
            and event['wait_for_render'] is True
            and abs(event['timeline_after_seconds']-event['timeline_before_seconds']-profile['delta_time_seconds'])<1e-7,
            'Native request/profile/callback ledger differs')
        reference=Fraction(*event['reference_time'])
        require(previous_ref is None or (reference>=previous_ref and event['previous_reference_time'] is not None
            and Fraction(*event['previous_reference_time'])==previous_ref),'Native reference clock reset/regressed')
        require(previous_timeline is None or event['timeline_before_seconds']>=previous_timeline-1e-7,'Scheduler clock regressed')
        previous_ref=reference;previous_timeline=event['timeline_after_seconds']
    previous=None
    for j,obs in enumerate(observations,len(profile['warmup_steps'])+1):
        sync=obs['synchronization'];event=result['requests'][j-1];name=obs['sample_id']
        require(obs['observation_id']==name and sync['request_index']==sync['callback_sequence']==frames[name]['request_index']==j
            and obs['request_evidence']==event and event['render_settings']==profile['render_settings']
            and event['reset_returned_without_exception'] is True,
            'Actual frame/reset callback authority differs')
        token=sync['freshness']
        require(token['callback_sequence']==j and token['reference_time']==sync['reference_time']==event['reference_time'],
            'Frame native reference token differs')
        if previous is not None:
            require(all(token[k]!=previous[k] for k in ('camera_sha256','rgb_sha256','depth_sha256')),
                'Repeated camera or stale native RGB-D')
        previous=token


def authenticate_capture(capture,read,local):
    trial=capture.parent
    require(capture.name=='capture' and not (capture/'failure.json').exists(),'Successful actual capture required')
    complete=read(local(trial/'owner_complete.json'))
    require(complete['training_approved'] is False and complete['accepted_training_increment']==0
        and Path(complete['result']['path']).resolve()==capture/'result.json'
        and Path(complete['owned_exit']['path']).resolve()==trial/'owned_exit.json'
        and Path(complete['owner_started']['path']).resolve()==trial/'owner_started.json'
        and Path(complete['native_identity']['path']).resolve()==capture/'native_identity.json',
        'One real owned capture closure required')
    exited=read(complete['owned_exit'],within=trial);started=read(complete['owner_started'],within=trial)
    require(exited['returncode']==0 and Path(exited['owned_worker']['path']).resolve()==trial/'owned_worker.json',
        'Actual native exit0 required')
    worker=read(exited['owned_worker'],within=trial);identity=read(complete['native_identity'],within=capture)
    result=read(complete['result'],within=capture);request=read(complete['request']);checked=producer.check_request(request)
    require(result['schema']==producer.RESULT_SCHEMA and result['request']==complete['request']==started['request']
        and result['native_identity']==complete['native_identity'] and complete['frames']==len(result['frames'])
        and result['training_approved'] is False and result['accepted_training_increment']==0
        and result['stage_count']==result['render_product_count']==1 and result['scene_assets_unchanged'] is True,
        'Actual completed single-scene result differs')
    command=started['command'];gate=producer.production.identity_gate
    require(identity['expected_command']==worker['command']==command
        and identity['command']['ProcessId']==worker['launcher_pid'] and identity['owner']['ProcessId']==worker['owner_pid']
        and gate.same_identity(identity['owner'],started['resources']['raw_owner_classification']['metadata']),
        'Actual single owner/native identity differs')
    gate.command_child(identity['command'],identity['owner'],command);gate.native_child(identity['native'],identity['command'],command[1:])
    require(command[1:6]==['-B','-u','-m',producer.MODULE,'--capture'],'Wrong native sweep module')
    def argument(flag):
        require(command.count(flag)==1 and command.index(flag)+1<len(command),'Exact worker argument required')
        return command[command.index(flag)+1]
    require(Path(argument('--output')).resolve()==capture
        and Path(argument('--request')).resolve()==Path(complete['request']['path']).resolve()
        and argument('--request-sha256')==complete['request']['sha256'],'Native command request/output differs')
    require(result['context']==local(capture/'context.json'),'Context escaped actual capture')
    context=read(result['context'],within=capture)
    require(producer.validate_result(result,Path(complete['request']['path']),capture)==context,
        'Shared native result validator disagrees with the authenticated context')
    require(context['schema']==producer.CONTEXT_SCHEMA and context['request']==complete['request']
        and context['native_identity']==complete['native_identity'] and context['profile']==request['profile']
        and context['scene_anchor']==request['scene_anchor'],'Common context authority differs')
    observations=[]
    for frame in result['frames']:
        observation=read(frame,within=capture/'frames')
        require(observation['schema']==producer.OBSERVATION_SCHEMA and observation['context']==result['context']
            and observation['training_approved'] is False and observation['accepted_training_increment']==0,
            'Wrong actual observation schema or context')
        observations.append(observation)
    validate_ledger(result,checked['records'],observations,checked['profile'])
    return complete,result,request,checked,context,observations


def prepare_inventory(context,census,reports,catalogue,source_plan):
    """No proposed target is accepted by, or even passed to, this inventory."""
    require(context['dataset_split']=='train' and context['full_scene_census']==census
        and context['source_collection_plan']==census['source_collection_plan']
        and census['complete_active_plant_anatomy'] is True
        and census['active_counts']['component_plants']==144 and census['active_counts']['backdrop_instances']==0
        and census['removed_roots']==[],'Complete unchanged original144 population required')
    actual=dict(census);saved=actual.pop('deterministic_census_sha256')
    require(fingerprint(actual)==saved,'Census fingerprint differs')
    indexed=unique(reports,'plant_id');roots=unique(census['all_plant_roots'],'plant_root')
    require(all(source_plan['family_assignments'][r['source_family']]==r['source_split']=='train' for r in roots.values()),
        'Every scene source family must remain TRAIN')
    require(catalogue==original.expected_catalogue(context,indexed)
        and len(catalogue)==census['active_counts']['components'],'Exact complete native catalogue required')
    return coverage.target_inventory(context,indexed,catalogue),indexed


def evaluate_capture(capturepath,output):
    capture=Path(capturepath).resolve();output=Path(output).resolve()
    require(capture.is_dir() and not output.exists() and not output.is_relative_to(capture.parent),'Fresh offline output required')
    pins={}
    def bound(spec,*,within=None):
        path=Path(spec['path']).resolve()
        require(sha256(path)==spec['sha256'],'Changed source '+str(path))
        if within is not None:require(path.is_relative_to(within),'Artifact escaped actual owner capture')
        pins[str(path)]=spec['sha256'];return path
    def read(spec,*,within=None):return read_json(bound(spec,within=within))
    def local(path):return dict(path=str(Path(path).resolve()),sha256=sha256(path))
    for path,h in FROZEN.items():bound(dict(path=path,sha256=h))
    complete,result,request,checked,context,observations=authenticate_capture(capture,read,local)
    for bindings in (request['implementation_bindings'],context['source_bindings'],checked['profile']['source_bindings']):
        verify_bindings(bindings)
        for path,h in bindings.items():require(path not in pins or pins[path]==h,'Conflicting source pin');pins[path]=h
    census=read(context['census'],within=capture);reports=read(context['reports'],within=capture);catalogue=read(context['catalogue'],within=capture)
    require(census==checked['census'],'Actual scene differs from request expected census')
    source_plan=read(context['source_collection_plan']);inventory,report_by=prepare_inventory(context,census,reports,catalogue,source_plan)
    for family in {r['source_family'] for r in census['all_plant_roots']}:
        report=report_by[family];manifest=bound(dict(path=report['manifest_path'],sha256=report['manifest_sha256']))
        require(context['source_bindings'].get(str(manifest))==report['manifest_sha256'],'Manifest absent from native source closure')
        for c in report['components'].values():
            path=str((manifest.parent/c['file']).resolve())
            require(context['source_bindings'].get(path)==c['asset_sha256'],'Component absent from native source closure')
    profile=checked['profile'];check_profile(profile)
    require(context['renderer_settings']==profile['render_settings'],'Actual native optics differ')
    planned=unique(checked['records'],'sample_id');index=coverage.CatalogueIndex(catalogue)
    output.mkdir();checker=BulkWorkspace();summaries=[]
    for path,h in {**checker.bindings,str(Path(__file__).resolve()):sha256(__file__),str(Path(producer.__file__).resolve()):sha256(producer.__file__)}.items():
        require(path not in pins or pins[path]==h,'Conflicting implementation pin');pins[path]=h
    try:
        for receipt,obs in zip(result['frames'],observations):
            name=obs['sample_id'];record=planned[name];pose=obs['robot_snapshot']
            assert_same_camera(obs['calibration'],record['calibration'])
            require(pose['joint_degrees']==record['joint_degrees']
                and np.allclose(pose['robot_root_to_world_usd_row_vectors'],record['robot_root_to_world_usd_row_vectors'],atol=1e-9,rtol=0)
                and np.allclose(pose['camera_to_head_column_vectors'],record['camera_to_head_column_vectors'],atol=1e-9,rtol=0)
                and pose['visual_bound_screen']['passed'] is True and pose['whole_robot_collision_checked'] is True,
                'Actual embodied camera or full-robot collision screen differs')
            require(obs['target_variant']==record['target_variant'] and obs['plant_root']==record['plant_root']
                and obs['source_family']==record['source_family']
                and obs['target_id']==record['target_variant']['variant_id']+'/'+record['source_row']['component_id']
                and obs['target_id_scope']=='private_pose_proposal_not_model_input_or_unique_label',
                'Proposal lineage differs; target hint is never a unique answer')
            plant=next(r for r in census['all_plant_roots'] if r['plant_root']==obs['plant_root'])
            expected=geometry_9mm(report_by[record['source_family']],record['source_row']['component_id'],plant['plant_to_world_usd_row_vectors'],obs['calibration'])
            validate_geometry(obs['geometry'],expected)
            with Image.open(bound(obs['files']['rgb'],within=capture)) as im:
                require(im.mode=='RGB' and im.size==(848,408),'Native unscaled RGB required');rgb=np.asarray(im).copy()
            with np.load(bound(obs['files']['buffers'],within=capture),allow_pickle=False) as arrays:
                require(set(arrays.files)=={'depth_m','renderer_instance_id','depth_valid','rgba_alpha'},'Exact native callback buffers required')
                depth,ids,valid,alpha=[arrays[k].copy() for k in ('depth_m','renderer_instance_id','depth_valid','rgba_alpha')]
            mapping_value=read(obs['mapping'],within=capture);mapping={int(k):v for k,v in mapping_value['renderer_id_to_prim'].items()}
            payload=deepcopy(obs['native_payload_header']);payload.update(rgb=np.dstack((rgb,alpha)),distance_to_image_plane=depth,
                instance_id_segmentation=dict(data=ids,info=dict(idToLabels=mapping)))
            sync=obs['synchronization']
            _,_,nv,reference,token,nids,nmapping=producer.sensor.native_freshness(payload,obs['calibration'],sync['static_before'],sync['static_after'],sync['callback_sequence'])
            require(token==sync['freshness'] and reference==sync['reference_time'] and np.array_equal(nv,valid)
                and np.array_equal(nids,ids) and nmapping==mapping,'Saved native RGB-D-ID freshness/calibration differs')
            components,_,owners=coverage.component_masks(ids,mapping,index);unknown=[];categories=Counter()
            for rid in np.unique(ids):
                rid=int(rid);path=mapping.get(rid,'');count=int(np.sum(ids==rid));owner=owners[rid]
                if owner:category='authenticated_component_'+owner['organ_type']
                elif rid==0 and not path and np.all(~valid[ids==rid]) and np.all(np.isposinf(depth[ids==rid])):category='empty_background'
                elif path.startswith(('/World/Environment/GreenHouse/','/World/Gutters/','/World/RBY1/','/World/Floor','/World/Ground')):category='nonplant_greenhouse_or_robot'
                else:category='unknown_identity';unknown.append(dict(renderer_id=rid,prim_path=path,pixels=count))
                categories[category]+=count
            metadata=dict(schema_version='greenhouse.native848_no_query_9mm_sample.v1',sample_id=name,calibration=obs['calibration'],robot_snapshot=pose,
                rendered_camera_params=obs['rendered_camera_params'],synchronization=dict(**sync,scene_unchanged_during_capture=True,dynamic_recording_supported=False),
                geometry_screen=pose['visual_bound_screen'],source_family=obs['source_family'],split=context['dataset_split'],training_approved=False)
            verify_camera(metadata)
            annotation=evaluate_frame(metadata,rgb,depth,valid,components,catalogue,inventory,checker)
            for target in annotation['targets']:
                joint=target.get('attachment_joint_ownership')
                if joint is not None:require(all(pins.get(p)==h for p,h in joint['source_bindings'].items()),'Unbound actual joint geometry')
            folder=output/name;folder.mkdir();obspin={k:receipt[k] for k in ('path','sha256')}
            cov=dict(schema='greenhouse.native848_all_petiole_coverage_audit.v1',observation=obspin,context=result['context'],
                capture_split='train',scene_source_families={r['source_family']:r['source_split'] for r in census['all_plant_roots']},
                complete_all_eligible_ground_truth=not unknown and not annotation['target_census']['unknown_target_ids'],
                blocking_unknowns=unknown+annotation['target_census']['unknown_target_ids'],visible_cross_split_petiole_targets=[],
                active_catalogue_components=len(catalogue),catalogue_petiole_count=len(inventory),actual_fully_labeled_scene_census=context['census'],
                coverage_scope='complete144_manifest_backed_instances',plant_instance_count=144,clones_count_as_independent_families=False,
                pixel_categories=dict(categories),source_bindings=dict(pins),training_approved=False)
            save_json(folder/'coverage.json',cov)
            annotation['target_census'].update(full_scene_coverage=local(folder/'coverage.json'),complete=cov['complete_all_eligible_ground_truth'] and annotation['target_census']['catalogue_complete'])
            annotation.update(frame_blocked_by_unverified_full_scene_coverage=not cov['complete_all_eligible_ground_truth'],observation=obspin,source_bindings=dict(pins),
                implementation=local(labels.__file__),coverage_implementation=local(coverage.__file__),source_target_identity_policy=original.source_lineage_policy(annotation),
                single_target_pilot_eligible=False)
            banned=annotation['source_target_identity_policy']['banned_candidate_instance_ids'];strict=annotation['target_census']['complete'] and len(annotation['target_census']['candidate_target_ids'])==1
            annotation['single_target_automated_candidate']=bool(strict and not banned)
            annotation['state']='unique_candidate_pending_actual_full_frame_visual_review' if strict and not banned else 'held_not_proven_single_eligible_target'
            save_json(folder/'sample.json',metadata);save_json(folder/'annotation.json',annotation)
            assessment=assess_frame(metadata,annotation,checker,metadata_pin=local(folder/'sample.json'),annotation_pin=local(folder/'annotation.json'))
            save_json(folder/'ambiguity.json',assessment)
            unique_answer=bool(strict and not banned and assessment['single_answer_unambiguous'] and assessment['assessment_complete'])
            candidate=next((t for t in annotation['targets'] if t['automated_pass']),None) if strict else None
            foreground=candidate['plant_instance_id'] if candidate else record['target_variant']['variant_id']
            score=background.score(ids,mapping_value,catalogue,[foreground],depth=depth,valid=valid)
            bg=dict(sample_id=name,observation=obspin,context=result['context'],catalogue=context['catalogue'],mapping=obs['mapping'],buffers=obs['files']['buffers'],
                excluded_foreground_variant=foreground,foreground_basis='actual_sole_strict_candidate' if candidate else 'proposal_for_diagnostic_only',score=score,
                meets_root_additional_background_criterion=score['background_plant_pixel_fraction']>=.4 and score['unknown_unmapped_fraction']==0,
                criterion_is_not_visual_acceptance=True)
            save_json(folder/'background.json',bg)
            summaries.append(dict(frame_id=name,source_family=candidate['source_family'] if unique_answer else None,source_target_ids=[t['source_target_id'] for t in annotation['targets'] if t['automated_pass']],
                scene_initializer_family=obs['source_family'],split='train',metadata=local(folder/'sample.json'),annotation=local(folder/'annotation.json'),coverage=local(folder/'coverage.json'),ambiguity=local(folder/'ambiguity.json'),background=local(folder/'background.json'),
                rgb=obs['files']['rgb'],buffers=obs['files']['buffers'],complete_census=annotation['target_census']['complete'],candidate_target_ids=annotation['target_census']['candidate_target_ids'],
                strict_unique_automated_candidate=bool(strict),unique_automated_candidate=unique_answer,background_criterion_passed=bg['meets_root_additional_background_criterion'],
                banned_source_target_instance_ids=banned,reachable_alternative_ids=assessment['reachable_alternative_ids'],unknown_alternative_ids=assessment['unknown_alternative_ids'],
                actual_visual_review=False,training_approved=False,accepted_training_increment=0))
            print(json.dumps(dict(frame_id=name,unique_automated_candidate=unique_answer,background_plant_fraction=score['background_plant_pixel_fraction'])),flush=True)
        stats=checker.finish();verify_bindings(pins)
        value=dict(schema=SCHEMA,source_capture=str(capture),owner_complete=local(capture.parent/'owner_complete.json'),capture_result=complete['result'],context=result['context'],
            completed_whole_capture_authenticated=True,all_committed_frames_evaluated=True,frames_evaluated=len(summaries),records=summaries,
            unique_automated_candidates=sum(r['unique_automated_candidate'] for r in summaries),
            unique_candidates_with_background_criterion=sum(r['unique_automated_candidate'] and r['background_criterion_passed'] for r in summaries),
            all_scene_petiole_inventory_count=len(inventory),workspace_stats=stats,source_bindings=pins,
            target_hint_used_to_select_answer=False,native_control_performed=False,actual_visual_review=False,training_approved=False,accepted_training_increment=0)
        save_json(output/'result.json',value);return value
    finally:checker.finish()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();v=evaluate_capture(a.capture,a.output);print(json.dumps(dict(frames=v['frames_evaluated'],unique=v['unique_automated_candidates'],training_approved=False)))
