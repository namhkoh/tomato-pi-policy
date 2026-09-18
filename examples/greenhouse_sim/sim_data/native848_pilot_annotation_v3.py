"""Authenticate owned filtered pilot frames and evaluate every active petiole at9mm.

Candidate-only CPU evidence. Complete scene coverage and unique eligibility are
separate from actual visual review; this module never exports/trains/accepts.
"""
from pathlib import Path
from collections import Counter
from fractions import Fraction
import argparse,hashlib,json
import numpy as np
from PIL import Image
from . import native848_pilot_plan_v1 as plan_api
from . import native848_pilot_scene_v1 as scene_api
from . import native848_all_petiole_9mm_v1 as labels
from . import native848_unique9mm_ambiguity_v1 as ambiguity
from . import native848_pilot_failed_parent_recovery_v1 as recovery
from . import native848_pilot_parallel_predecessor_v2 as parallel_predecessor
from .native848_bulk_workspace_v1 import BulkWorkspace
from .native848_pair_audit_v2 import verify_camera
from .native_greenhouse_pair import assert_same_camera
from .native_sensor_payload import validate_native_static,decode_native_instances
from .capture_contract import fingerprint
from .capture_visibility import component_masks,owner_for_path
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256
from .native848_bulk_io_v1 import save_json
from .native848_data_roots_v1 import diagnostic

SCHEMA='greenhouse.native848_filtered_pilot_9mm_evaluation.v3'
OWNER_SCHEMAS={
    'greenhouse.native848_unique9mm_pilot_owner.v1':('run_native848_pilot_v1.py','0705b93d87a576243e1062753d46941b3546a9c774822a979234faf8ed1b92f0'),
    'greenhouse.native848_unique9mm_pilot_owner.v2':('run_native848_pilot_v2.py','0c8d0b6f7b687d31c2a4134719b48d9dd4f8b1a210c591f21faebe8efff57302')}
AMBIGUITY_SHA256='c42bca181ca07af4446c0ff5c24f14126573bc39dc8eeeb609a8e835fd9ac2fe'
ROOT=Path(__file__).resolve().parents[3]


def authenticate_owner(trial,owner,result_sha256,read,bound,parent_metadata,parent_recovery=None):
    require(owner['state']=='owned_unique9mm_filtered_capture_exited_pending_census_annotation_and_visual_review'
        and owner['complete_9mm_annotation'] is False and owner['training_approved'] is False,
        'Actual capture-only pilot owner result required')
    complete=read(trial/'owner_complete.json',sha256(trial/'owner_complete.json'))
    require(complete['result_sha256']==result_sha256 and complete['capture_only'] is True,'Actual pilot completion required')
    schema=owner['schema']
    if parent_recovery is not None:
        require(parent_metadata is None and schema=='greenhouse.native848_pilot_parallel_slot.v1'
            and complete['parallel_slot_only'] is True and owner['parallel_slot_only'] is True,
            'Failed-parent recovery is only for genuine completed parallel slots')
        proof=recovery.check(parent_recovery)
        slot=owner['bounded_parallel_slot'];parent=Path(proof['metadata']['trial_path']).resolve()
        require(proof['parent_failed'] is True and slot in proof['recoverable_slots']
            and trial==parent/f'slot_{slot}'
            and proof['successful_slot_results'][str(slot)]==dict(path=str(trial/'result.json'),sha256=result_sha256),
            'Recovery audit does not authorize this exact completed slot')
        for p,h in proof['source_bindings'].items():bound(p,h)
        bound(parent_recovery['path'],parent_recovery['sha256'])
        return dict(complete,parent_failed=True,failed_parent_recovery=parent_recovery,
            recovered_completed_slot=slot,whole_parent_success_claimed=False)
    if schema in OWNER_SCHEMAS:
        require(parent_metadata is None and complete['serial_native_lock_released_after_this_receipt'] is True,
            'Actual serial pilot terminal required')
        name,pin=OWNER_SCHEMAS[schema];path=ROOT/'data/sim_data/diagnostics/collection_20k_848_20260916_v1'/name
        require(owner['source_bindings'].get(str(path.resolve()))==pin and sha256(path)==pin,'Unknown serial pilot implementation')
        bound(path,pin)
    else:
        require(schema=='greenhouse.native848_pilot_parallel_slot.v1' and parent_metadata is not None
            and complete['parallel_slot_only'] is True and owner['parallel_slot_only'] is True,
            'Explicit genuine parallel slot requires whole-parent completion')
        parent=trial.parent;parent_result=parent/'result.json';parent_terminal=parent/'owner_complete.json'
        proof=parallel_predecessor.check(dict(metadata=parent_metadata,
            result=dict(path=str(parent_result),sha256=sha256(parent_result)),
            terminal=dict(path=str(parent_terminal),sha256=sha256(parent_terminal))))
        for p,h in proof['source_bindings'].items():bound(p,h)
        whole=read(parent_result,sha256(parent_result));slot=owner['bounded_parallel_slot']
        require(trial==parent/f'slot_{slot}' and whole['source_bindings']==owner['source_bindings'],
            'Parallel slot source closure differs from actual parent')
        entries=[r for r in whole['results'] if r['slot']==slot]
        require(len(entries)==1 and entries[0]['owner_result']==dict(path=str(trial/'result.json'),sha256=result_sha256),
            'Whole parent does not bind requested successful slot')
    return complete



def expected_catalogue(context,reports):
    census=context['full_scene_census'];roots={r['plant_root']:r for r in census['all_plant_roots'] if r['active_after_policy']}
    variants={v['plant_root']:v for v in context['anchor']['scene_variants']}
    require(set(roots)==set(census['retained_roots']) and set(roots)<=set(variants),'Active source roots differ')
    entries=[]
    for root,state in sorted(roots.items()):
        v=variants[root];report=reports[v['source_plant_id']]
        require(state['authenticated_component_plant'] is True and state['source_family']==v['source_plant_id']
                and state['source_split']==context['split'] and not v['added_components']
                and not v['added_component_paths'],'Unauthenticated/cross-split active plant')
        components=report['components']
        for key,c in components.items():
            chain=[key];parent=c['parent']
            while parent is not None:
                require(parent in components and parent not in chain,'Invalid component ancestry')
                chain.append(parent);parent=components[parent]['parent']
            entries.append(dict(component_id=key,prim_path=root+'/'+'/'.join(reversed(chain)),organ_type=c['type'],
                                variant_id=v['variant_id'],source_plant_id=v['source_plant_id'],split_group=v['split_group']))
    return [dict(r,component_index=i) for i,r in enumerate(sorted(entries,key=lambda r:r['prim_path']),1)]


def run(trial,result_sha256,output,*,parent_metadata=None,parent_recovery=None):
    trial=diagnostic(trial);output=diagnostic(output)
    require(not output.exists() and not (trial/'failure.json').exists(),'New output and successful owned pilot required')
    pins={}
    def bound(path,pin):
        path=Path(path).resolve();require(sha256(path)==pin,'Changed artifact '+str(path));pins[str(path)]=pin;return path
    def read(path,pin):return read_json(bound(path,pin))
    owner=read(trial/'result.json',result_sha256)
    closure=authenticate_owner(trial,owner,result_sha256,read,bound,parent_metadata,parent_recovery)
    exit_receipt=read(owner['owned_exit_path'],owner['owned_exit_sha256'])
    require(exit_receipt['returncode']==0 and exit_receipt['method']=='subprocess_wait_on_owned_process','Actual owned exit0 required')
    verify_bindings(owner['source_bindings']);pins.update(owner['source_bindings'])
    native=read(owner['capture_result_path'],owner['capture_result_sha256'])
    manifest=read(owner['manifest_path'],owner['manifest_sha256'])
    context=read(owner['context_path'],owner['context_sha256'])
    require(native['state']==plan_api.RESULT_STATE and context['schema']==plan_api.CONTEXT_SCHEMA
            and native['manifest_sha256']==owner['manifest_sha256'] and manifest['context_sha256']==owner['context_sha256']
            and native['committed_frames']==manifest['committed_frames']==owner['committed_frames'], 'New native pilot closure differs')
    require(not (trial/'capture/failure.json').exists() and manifest['source_assets_unchanged'] is True,'Native capture failure/source mutation')
    plan=read(context['plan_path'],context['plan_sha256'])
    require(plan['schema']==plan_api.SCHEMA and plan['capture_split']==context['split']
            and context['scene_policy']==scene_api.policy(context['split'])==plan['pilot_scene_policy'], 'Actual split/scene policy differs')
    census=read(context['full_scene_census_path'],context['full_scene_census_sha256'])
    require(census==context['full_scene_census'] and census['complete_active_plant_anatomy'] is True
            and census['active_counts']==context['scene_counts'] and census['active_counts']['backdrop_instances']==0,
            'Exact full filtered-scene census required')
    identity=dict(census);saved_identity=identity.pop('deterministic_census_sha256')
    require(fingerprint(identity)==saved_identity,'Scene census identity differs')
    cpu=read(context['cpu_scene_preflight']['path'],context['cpu_scene_preflight']['sha256'])
    require(cpu['deterministic_census_sha256']==saved_identity and cpu['plan_sha256']==context['plan_sha256'],
            'Native scene did not match exact CPU preflight')
    reports_list=read(context['source_reports_path'],context['source_reports_sha256'])
    reports={r['plant_id']:r for r in reports_list};require(len(reports)==len(reports_list),'Duplicate anatomical reports')
    catalogue=read(context['catalogue_path'],context['catalogue_sha256'])
    require(catalogue==expected_catalogue(context,reports) and len(catalogue)==context['scene_counts']['components'],
            'Incomplete actual active component catalogue')
    source_plan=read(context['anchor']['source_collection_plan'],census['source_collection_plan']['sha256'])
    roots={r['plant_root']:r for r in census['all_plant_roots'] if r['active_after_policy']}
    variants={v['variant_id']:v for v in context['anchor']['scene_variants'] if v['plant_root'] in roots}
    inventory=[]
    for c in catalogue:
        if c['organ_type']!='sub_stem':continue
        v=variants[c['variant_id']];family=c['source_plant_id'];report=reports[family]
        require(source_plan['family_assignments'][family]==context['split'],'Active plant belongs to another split')
        candidates=[t for t in report['targets'] if t['component_id']==c['component_id']]
        require(len(candidates)==1,'Unique anatomical target record required')
        t=candidates[0];comp=report['components'][c['component_id']]
        leaves=[x for x in t['expected_detached_component_ids'] if report['components'][x]['type']=='leaf']
        semantic=comp.get('deleafed') is False and report['components'].get(comp['parent'],{}).get('type')=='main_stem' and bool(leaves) and not t['protected_descendant_ids']
        inventory.append(dict(target_id=c['variant_id']+'/'+c['component_id'],source_family=family,report=report,
            plant_to_world_usd_row_vectors=roots[v['plant_root']]['plant_to_world_usd_row_vectors'],
            semantic_leaf_petiole_candidate=bool(semantic),anatomy_reason_codes=t['reason_codes']))
    pose_cache=read(plan['cache_path'],plan['cache_sha256'])
    pose_by_id={r['sample_id']:r for r in pose_cache['records']}
    require(len(pose_by_id)==len(pose_cache['records']), 'Duplicate planned pose')
    schedule={r['observation_id']:r for r in plan['schedule']}
    require(len(schedule)==plan['max_frames'] and len(manifest['decisions'])==plan['max_frames'],'Complete finite native decision population required')
    require({r['observation_id'] for r in manifest['decisions']}==set(schedule),'Native decision schedule differs')
    observations=manifest['observations'];require(len(observations)==manifest['committed_frames'],'Committed native population differs')
    profile=read(plan['profile_evidence']['path'],plan['profile_evidence']['sha256']);plan_api.check_profile(profile)
    require(sha256(ambiguity.__file__)==AMBIGUITY_SHA256,'Frozen no-query ambiguity policy changed')
    checker=BulkWorkspace();records=[]
    for path,h in {**checker.bindings,str(Path(__file__).resolve()):sha256(__file__),str(Path(labels.__file__).resolve()):sha256(labels.__file__),str(Path(ambiguity.__file__).resolve()):AMBIGUITY_SHA256,str(Path(parallel_predecessor.__file__).resolve()):sha256(parallel_predecessor.__file__),str(Path(recovery.__file__).resolve()):sha256(recovery.__file__)}.items():
        require(path not in pins or pins[path]==h,'Annotation source pin conflicts');pins[path]=h
    output.mkdir(parents=True)
    requests=set();callbacks=set();seen_names=set();previous=None
    for committed in observations:
        # FrameSink manifest rows contain immutable observation file/path pins.
        observation_path=committed.get('observation_path',committed.get('path'))
        observation_hash=committed.get('observation_sha256',committed.get('sha256'))
        obs=read(observation_path,observation_hash)
        name=obs['observation_id']
        require(name not in seen_names,'Repeated observation identity');seen_names.add(name)
        require(name in schedule and obs['schema']==plan_api.OBSERVATION_SCHEMA
                and obs['capture_role']=='qualification_candidate' and obs['source_pose_id']==schedule[name]['source_pose_id']
                and obs['context_sha256']==owner['context_sha256'] and obs['split']==context['split'], 'Unscheduled or wrong-epoch observation')
        cached=pose_by_id[obs['source_pose_id']]
        assert_same_camera(obs['calibration'],cached['calibration'])
        require(obs['target_id']==cached['target_id'] and obs['robot_snapshot']['joint_degrees']==cached['joint_degrees']
            and np.allclose(obs['robot_snapshot']['robot_root_to_world_usd_row_vectors'],cached['robot_root_to_world_usd_row_vectors'],atol=1e-9,rtol=0), 'Actual planned embodied pose differs')
        rgb_path=bound(obs['files']['rgb']['path'],obs['files']['rgb']['sha256'])
        with Image.open(rgb_path) as im:
            require(im.mode=='RGB' and im.size==(848,408),'Exact native RGB required');rgb=np.asarray(im).copy()
        with np.load(bound(obs['files']['buffers']['path'],obs['files']['buffers']['sha256']),allow_pickle=False) as a:
            require(set(a.files)=={'depth_m','renderer_instance_id','depth_valid','rgba_alpha'},'Exact callback buffers required')
            depth,ids,valid,alpha=[a[k].copy() for k in ('depth_m','renderer_instance_id','depth_valid','rgba_alpha')]
        mapping_value=read(obs['mapping']['path'],obs['mapping']['sha256'])
        mapping={int(k):v for k,v in mapping_value['renderer_id_to_prim'].items()}
        sync=obs['synchronization'];request=obs['request_evidence']
        require(sync['request_index'] not in requests and sync['callback_sequence'] not in callbacks,'Repeated native request/callback')
        requests.add(sync['request_index']);callbacks.add(sync['callback_sequence'])
        require(request['native_requests']==request['callback_count']==1 and request['requested_subframes']==profile['request_subframes']
                and request['render_settings']==profile['render_settings'] and request['reset_returned_without_exception'] is True
                and request['wait_for_render'] is True and request['delta_time_seconds']==profile['delta_time_seconds'], 'Native profile/reset/request differs')
        payload=dict(obs['native_payload_header']);payload.update(rgb=np.dstack((rgb,alpha)),distance_to_image_plane=depth,
            instance_id_segmentation=dict(data=ids,info=dict(idToLabels=mapping)))
        _,_,checked_valid,reference,token=validate_native_static(payload,obs['calibration'],sync['static_before'],sync['static_after'],sync['callback_sequence'])
        token.update(instance_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),mapping_sha256=fingerprint(mapping),reference_time=reference)
        require(token==sync['freshness'] and reference==sync['reference_time'] and np.array_equal(valid,checked_valid), 'Actual native callback replay differs')
        require(abs((request['timeline_after_seconds']-request['timeline_before_seconds'])-profile['delta_time_seconds'])<1e-7,'Native scheduler delta differs')
        current_reference=Fraction(*reference)
        if previous is not None:
            require(current_reference>=previous['reference'] and sync['callback_sequence']>previous['sequence']
                and all(token[k]!=previous['token'][k] for k in ('camera_sha256','rgb_sha256','depth_sha256')), 'Ordered freshness/continuity failed')
        previous=dict(reference=current_reference,sequence=sync['callback_sequence'],token=token)
        decode_native_instances(payload,[848,408])
        geometry=read(obs['geometry_proof']['path'],obs['geometry_proof']['sha256'])
        pose=obs['robot_snapshot'];key=plan_api.geometry_pose_key(context['scene_revision'],pose,context['source_bindings_sha256'])
        require(geometry['pose_key']==obs['geometry_proof']['pose_key']==key and geometry['screen']==pose['visual_bound_screen']
                and geometry['scene_revision']==context['scene_revision'] and geometry['source_bindings_sha256']==context['source_bindings_sha256']
                and geometry['screen']['passed'] is True and geometry['joint_degrees']==pose['joint_degrees']
                and geometry['robot_root_to_world_usd_row_vectors']==pose['robot_root_to_world_usd_row_vectors'], 'Per-frame robot geometry differs')
        components,_,_=component_masks(ids,mapping,catalogue)
        unknown=[];pixel_classes=Counter()
        for rid in np.unique(ids):
            rid=int(rid);path=mapping.get(rid,'');count=int(np.sum(ids==rid));owner_component=owner_for_path(path,catalogue)
            require(not any(path==r or path.startswith(r+'/') for r in census['removed_roots']), 'Removed plant appeared in actual native pixels')
            if owner_component:category='authenticated_component_'+owner_component['organ_type']
            elif rid==0 and not path and np.all(~valid[ids==rid]) and np.all(np.isposinf(depth[ids==rid])):category='empty_background'
            elif path.startswith(('/World/Environment/GreenHouse/','/World/Gutters/','/World/RBY1/','/World/Floor','/World/Ground')):category='nonplant_greenhouse_or_robot'
            else:category='unknown_identity';unknown.append(dict(renderer_id=rid,prim_path=path,pixels=count))
            pixel_classes[category]+=count
        metadata=dict(schema_version='greenhouse.native848_no_query_9mm_sample.v1',sample_id=name,
            calibration=obs['calibration'],robot_snapshot=pose,rendered_camera_params=obs['rendered_camera_params'],
            synchronization=dict(**sync,scene_unchanged_during_capture=True,dynamic_recording_supported=False),
            geometry_screen=geometry['screen'],source_family=obs['source_family'],split=obs['split'],training_approved=False)
        verify_camera(metadata)
        annotation=labels.evaluate_frame(metadata,rgb,depth,valid,components,catalogue,inventory,checker)
        folder=output/name;folder.mkdir()
        coverage=dict(schema='greenhouse.native848_all_petiole_coverage_audit.v1',
            observation_path=str(Path(observation_path).resolve()),observation_sha256=observation_hash,
            context_path=owner['context_path'],context_sha256=owner['context_sha256'],
            context=dict(path=owner['context_path'],sha256=owner['context_sha256']),
            capture_split=context['split'],scene_source_families={r['source_family']:r['source_split'] for r in roots.values()},
            complete_all_eligible_ground_truth=not unknown and not annotation['target_census']['unknown_target_ids'],
            blocking_unknowns=unknown+annotation['target_census']['unknown_target_ids'],visible_cross_split_petiole_targets=[],
            active_catalogue_components=len(catalogue),catalogue_petiole_count=len(inventory),
            actual_filtered_scene_census=dict(path=context['full_scene_census_path'],sha256=context['full_scene_census_sha256']),
            pixel_categories=dict(pixel_classes),source_bindings=dict(pins),training_approved=False)
        save_json(folder/'coverage.json',coverage)
        annotation['target_census']['full_scene_coverage']=dict(path=str(folder/'coverage.json'),sha256=sha256(folder/'coverage.json'))
        annotation['target_census']['complete']=coverage['complete_all_eligible_ground_truth'] and annotation['target_census']['catalogue_complete']
        annotation['frame_blocked_by_unverified_full_scene_coverage']=not coverage['complete_all_eligible_ground_truth']
        annotation['observation']=dict(path=str(Path(observation_path).resolve()),sha256=observation_hash)
        annotation['source_bindings']=dict(pins)
        annotation['implementation']=dict(path=str(Path(labels.__file__).resolve()),sha256=sha256(labels.__file__))
        count=len(annotation['target_census']['candidate_target_ids'])
        unique=annotation['target_census']['complete'] and count==1
        annotation['single_target_pilot_eligible']=False
        annotation['single_target_automated_candidate']=bool(unique)
        annotation['state']='unique_candidate_pending_actual_full_frame_visual_review' if unique else 'held_not_proven_single_eligible_target'
        save_json(folder/'sample.json',metadata);save_json(folder/'annotation.json',annotation)
        def spec(path):return dict(path=str(path),sha256=sha256(path))
        # Strict target geometry/status remains unchanged. The separate no-query
        # assessment can only hold a frame; it never promotes a failed target.
        assessment=ambiguity.assess_frame(read_json(folder/'sample.json'),read_json(folder/'annotation.json'),checker,
            metadata_pin=spec(folder/'sample.json'),annotation_pin=spec(folder/'annotation.json'))
        save_json(folder/'ambiguity.json',assessment)
        strict_unique=bool(unique)
        unique=bool(strict_unique and assessment['single_answer_unambiguous'] and assessment['assessment_complete'])
        candidate_families={t['source_family'] for t in annotation['targets'] if t['automated_pass']}
        candidate_family=next(iter(candidate_families)) if unique else None
        records.append(dict(frame_id=name,source_family=candidate_family,scene_initializer_family=obs['source_family'],split=obs['split'],
            metadata=spec(folder/'sample.json'),annotation=spec(folder/'annotation.json'),coverage=spec(folder/'coverage.json'),
            rgb=obs['files']['rgb'],buffers=obs['files']['buffers'],candidate_count=count,
            complete_census=annotation['target_census']['complete'],unique_automated_candidate=bool(unique),
            strict_unique_automated_candidate=strict_unique,ambiguity=spec(folder/'ambiguity.json'),
            reachable_alternative_ids=assessment['reachable_alternative_ids'],unknown_alternative_ids=assessment['unknown_alternative_ids'],
            candidate_target_ids=annotation['target_census']['candidate_target_ids'],actual_visual_review=False,
            training_approved=False,accepted_training_increment=0))
    stats=checker.finish();verify_bindings(pins)
    result=dict(schema=SCHEMA,owner_closure=closure,owner_result=dict(path=str(trial/'result.json'),sha256=result_sha256),
        records=records,frames_evaluated=len(records),unique_automated_candidates=sum(r['unique_automated_candidate'] for r in records),
        strict_unique_candidates=sum(r['strict_unique_automated_candidate'] for r in records),
        ambiguity_policy=ambiguity.POLICY,workspace_stats=stats,implementation=dict(path=str(Path(__file__).resolve()),sha256=sha256(__file__)),
        source_bindings=pins,actual_visual_review=False,training_approved=False,accepted_training_increment=0)
    save_json(output/'result.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--trial',required=True);parser.add_argument('--result-sha256',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--parent-metadata');parser.add_argument('--parent-metadata-sha256')
    parser.add_argument('--parent-recovery');parser.add_argument('--parent-recovery-sha256')
    args=parser.parse_args();require(bool(args.parent_metadata)==bool(args.parent_metadata_sha256),'Exact parent metadata pin pair required')
    require(bool(args.parent_recovery)==bool(args.parent_recovery_sha256),'Exact recovery audit pin pair required')
    require(not (args.parent_recovery and args.parent_metadata),'Explicit recovery and successful-parent modes are exclusive')
    parent_metadata=dict(path=str(Path(args.parent_metadata).resolve()),sha256=args.parent_metadata_sha256) if args.parent_metadata else None
    parent_recovery=dict(path=str(Path(args.parent_recovery).resolve()),sha256=args.parent_recovery_sha256) if args.parent_recovery else None
    value=run(args.trial,args.result_sha256,args.output,parent_metadata=parent_metadata,parent_recovery=parent_recovery)
    print(json.dumps(dict(frames=value['frames_evaluated'],unique_candidates=value['unique_automated_candidates'],training_approved=False)))
