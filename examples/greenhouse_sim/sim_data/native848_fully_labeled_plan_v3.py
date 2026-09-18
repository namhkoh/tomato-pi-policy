"""Create-only full144 labeled-population plans from authenticated bounded poses."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import numpy as np
from . import native848_pilot_plan_v1 as source_api
from . import native848_fully_labeled_scene_v1 as scene_api
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256
from .native848_bulk_io_v1 import save_json
SCHEMA='greenhouse.native848_fully_labeled_plan.v3'
CONTEXT_SCHEMA='greenhouse.native848_fully_labeled_context.v1'
OBSERVATION_SCHEMA='greenhouse.native848_fully_labeled_observation.v1'
RESULT_STATE='fully_labeled_population_captured_pending_native_visibility_uniqueness_and_independent_admission'
ORIGINAL_PROFILE_SCHEMA=source_api.ORIGINAL_PROFILE_SCHEMA
PAUSED_CLOCK_POLICY=source_api.PAUSED_CLOCK_POLICY
NATIVE_TIME_POLICY=source_api.NATIVE_TIME_POLICY
SCHEDULER_CLOCK_POLICY=source_api.SCHEDULER_CLOCK_POLICY
check_profile=source_api.check_profile
geometry_pose_key=source_api.geometry_pose_key
scene_policy=scene_api.policy
ROOT=Path(__file__).resolve().parents[3]

def implementation_bindings():
    return {str(Path(p).resolve()):sha256(p) for p in (__file__,scene_api.__file__,source_api.__file__)}

def normalized_record(record):
    r=deepcopy(record)
    legacy={k:r.pop(k) for k in list(r) if k in ('full_scene_cpu_preflight','cpu_workspace_preflight_9mm')}
    r['source_pose_cpu_receipts_lineage_only']=legacy
    r['new_population_scene_and_workspace_acceptance_inherited']=False
    return r

def original_target_key(record):
    row=record['source_row'];require(row['source_plant_id']==record['source_family'],'Source target family differs')
    return record['source_family']+'/'+row['component_id']

def camera_key(record):
    return (record['source_family'],tuple(np.asarray(record['camera_to_world_usd_row_vectors'],float).round(9).reshape(-1)))

def _one_source(spec):
    path=Path(spec['path']).resolve();require(sha256(path)==spec['sha256'],'Authenticated raw pose cache changed')
    value=read_json(path);data=source_api.validate_raw_cache(value)
    profile_spec=value['profile_evidence'];require(sha256(profile_spec['path'])==profile_spec['sha256'],'Profile changed')
    profile=read_json(profile_spec['path']);check_profile(profile)
    data.update(profile=profile,capture_split=data['anchor']['split'])
    data['source_bindings'].update({str(path):spec['sha256'],str(Path(profile_spec['path']).resolve()):profile_spec['sha256'],**profile['source_bindings']})
    return value,data

def _source(spec):
    path=Path(spec['path']).resolve();require(sha256(path)==spec['sha256'],'Pose source changed');value=read_json(path)
    if value.get('schema')==source_api.CACHE_SCHEMA:
        source,data=_one_source(spec);data['pose_anchor_sources']=[dict(cache=spec,anchor_reference_path=source['anchor_reference_path'],anchor_reference_sha256=source['anchor_reference_sha256'])];return source,data
    require(value.get('schema')=='greenhouse.native848_fully_labeled_pose_sources.v1' and value['caches'],'Explicit authenticated same-family cache bundle required')
    require(len({p['path'] for p in value['caches']})==len(value['caches']),'Duplicate raw cache source')
    batches=[_one_source(pin) for pin in value['caches']];first,data=deepcopy(batches[0]);records=[];pins={str(path):spec['sha256']};anchors=[]
    reference=data['anchor'];cal=reference['expected_calibration'];reference_manifest=read_json(Path(reference['source_capture'])/'manifest.json')
    for pin,(cache,part) in zip(value['caches'],batches):
        anchor=part['anchor']
        require(all(anchor[k]==reference[k] for k in ('source_family','split','original_variant','expected_scene_counts','scene_variants','scene_code_bindings')),'Multi-anchor population provenance differs')
        require(part['source_plan']['family_assignments']==data['source_plan']['family_assignments'] and part['source_plan']['package']==data['source_plan']['package'],'Multi-anchor split/package differs')
        require(cache['profile_evidence']==first['profile_evidence'],'Multi-anchor capture profile differs')
        manifest=read_json(Path(anchor['source_capture'])/'manifest.json')
        require(all(manifest[k]==reference_manifest[k] for k in ('lighting','renderer','unbundled_external_prop_roots_excluded')),'Multi-anchor original lighting/renderer/external scope differs')
        require(part['source_plan']['configuration'].get('clear_capture')==data['source_plan']['configuration'].get('clear_capture')=='robot_head_close_diffuse_v1','Multi-anchor lighting policy differs')
        require(np.allclose(anchor['expected_robot_snapshot']['camera_to_head_column_vectors'],reference['expected_robot_snapshot']['camera_to_head_column_vectors'],atol=1e-9,rtol=0),'Multi-anchor calibrated head mount differs')
        require(all(anchor['expected_calibration'][k]==cal[k] for k in cal if k!='camera_to_world_usd_row_vectors'),'Multi-anchor optics differ')
        for rec in part['records']:
            # The native scene uses the first anchor only for population; every body remains independently authenticated.
            from .native848_pair_audit_v2 import world_from_row
            actual=world_from_row(reference,rec['source_row']);own=world_from_row(anchor,rec['source_row'])
            require(actual==own,'Multi-anchor foreground world geometry differs')
        records.extend(part['records']);source_api.merge(pins,part['source_bindings'])
        anchors.append(dict(cache=pin,anchor_reference_path=cache['anchor_reference_path'],anchor_reference_sha256=cache['anchor_reference_sha256']))
    require(len({r['sample_id'] for r in records})==len(records),'Duplicate multi-anchor sample IDs')
    data.update(records=records,source_bindings=pins,pose_anchor_sources=anchors)
    return dict(value,profile_evidence=first['profile_evidence']),data

def validate_prior_schedule(spec):
    path=Path(spec['path']).resolve();require(sha256(path)==spec['sha256'],'Prior schedule ledger changed')
    value=read_json(path);require(value['schema']=='greenhouse.native848_fully_labeled_prior_schedule.v1','Typed prior full-population schedule required')
    verify_bindings(value['source_bindings']);require(value['clones_count_as_new_source_targets'] is False,'Source-target identity required')
    require(value['prior_plans'] and len({p['path'] for p in value['prior_plans']})==len(value['prior_plans']),'Exact prior plan list required')
    reconstructed=[]
    for pin in value['prior_plans']:
        pp=Path(pin['path']).resolve();require(sha256(pp)==pin['sha256']==value['source_bindings'].get(str(pp)),'Prior plan changed or unbound')
        plan=read_json(pp);require(plan['schema'] in ('greenhouse.native848_fully_labeled_plan.v1','greenhouse.native848_fully_labeled_plan.v2',SCHEMA),'Prior schedule from another population epoch')
        cp=Path(plan['cache_path']).resolve();require(sha256(cp)==plan['cache_sha256']==value['source_bindings'].get(str(cp)),'Prior selected pose cache changed or unbound')
        records=read_json(cp)['records'];require([r['sample_id'] for r in records]==plan['selected_sample_ids'] and len(records)==plan['max_frames'],'Prior selected schedule differs')
        require([r['source_pose_id'] for r in plan['schedule']]==plan['selected_sample_ids'],'Prior runtime schedule differs')
        reconstructed.extend(dict(source_family=r['source_family'],original_target_key=original_target_key(r),camera_to_world_usd_row_vectors=r['camera_to_world_usd_row_vectors'],source_pose_id=r['sample_id'],plan_path=str(pp)) for r in records)
    require(value['records']==reconstructed,'Prior ledger omitted or changed a pinned scheduled view')
    cameras=set();counts=Counter()
    for row in reconstructed:
        key=(row['source_family'],tuple(np.asarray(row['camera_to_world_usd_row_vectors'],float).round(9).reshape(-1)))
        require(key not in cameras,'Duplicate prior source-family camera');cameras.add(key);counts[row['original_target_key']]+=1
    require(not counts or max(counts.values())<=200,'Prior original-target view cap already exceeded')
    return value,cameras,counts,{**value['source_bindings'],str(path):spec['sha256']}

def prepare(source,selected_sample_ids,output,*,prior_schedule):
    output=Path(output).resolve();require(output.is_relative_to(ROOT/'data/sim_data/diagnostics') and not output.exists(),'Fresh diagnostic plan folder required')
    old,data=_source(source);by={r['sample_id']:r for r in data['records']};ids=list(selected_sample_ids)
    require(ids and len(ids)==len(set(ids)) and set(ids)<=by.keys(),'Unique authenticated source poses required')
    split=data['capture_split'];records=[normalized_record(by[i]) for i in ids]
    output.mkdir(parents=True);cache=output/'pose_cache.json';save_json(cache,dict(schema='greenhouse.native848_fully_labeled_poses.v2',records=records,training_approved=False))
    plan=dict(schema=SCHEMA,source_pose_cache=source,prior_schedule=prior_schedule,cache_path=str(cache),cache_sha256=sha256(cache),
        profile_evidence=old['profile_evidence'],capture_split=split,scene_policy=scene_policy(split),
        **scene_api.metadata(split),native_population_census_verified=False,
        selected_sample_ids=ids,schedule=[dict(observation_id='full144_'+i,source_pose_id=i,capture_role='fully_labeled_qualification_candidate') for i in ids],
        max_frames=len(ids),resolution=[848,408],instance_backend='fast',queue_frames=8,queue_bytes=256*2**20,writer_threads=2,
        geometry_policy='native_full_screen_each_new_complete_joint_pose',annotation_contract='no_query_exactly_one_eligible_petiole_nominal_9mm',
        per_target_max_planned_views=200,view_cap_identity='original_source_family_and_component_across_clones_and_prior_schedule',automatic_retries=False,source_acceptance_inherited=False,scene_preflight_required=True,
        implementation_bindings=implementation_bindings())
    check(plan);pp=output/'plan.json';save_json(pp,plan)
    return dict(path=str(pp),sha256=sha256(pp))

def check(plan):
    require(plan.get('schema')==SCHEMA and plan['implementation_bindings']==implementation_bindings(),'New population plan implementation differs')
    split=plan['capture_split'];require(plan['scene_policy']==scene_policy(split),'Full144 population policy differs')
    require(all(plan.get(k)==v for k,v in scene_api.metadata(split).items()) and plan['native_population_census_verified'] is False,'Pending population admission metadata required')
    require(plan['resolution']==[848,408] and plan['instance_backend']=='fast' and plan['queue_frames']==8 and plan['queue_bytes']==256*2**20 and plan['writer_threads']==2,'Native format differs')
    require(plan['scene_preflight_required'] is True and plan['annotation_contract']=='no_query_exactly_one_eligible_petiole_nominal_9mm' and plan['geometry_policy']=='native_full_screen_each_new_complete_joint_pose' and plan['per_target_max_planned_views']==200 and plan['view_cap_identity']=='original_source_family_and_component_across_clones_and_prior_schedule' and all(plan[k] is False for k in ('automatic_retries','source_acceptance_inherited')),'Scope differs')
    old,data=_source(plan['source_pose_cache']);require(data['capture_split']==split and old['profile_evidence']==plan['profile_evidence'],'Source split/profile differs')
    by={r['sample_id']:r for r in data['records']};ids=plan['selected_sample_ids'];require(len(ids)==len(set(ids))==plan['max_frames'] and 1<=len(ids)<=512 and set(ids)<=by.keys(),'Finite exact source subset required')
    records=[normalized_record(by[i]) for i in ids];cp=Path(plan['cache_path']);require(sha256(cp)==plan['cache_sha256'] and read_json(cp)['records']==records,'Normalized pose cache differs')
    require(plan['schedule']==[dict(observation_id='full144_'+i,source_pose_id=i,capture_role='fully_labeled_qualification_candidate') for i in ids],'New population schedule differs')
    prior,cameras,counts,prior_pins=validate_prior_schedule(plan['prior_schedule'])
    for record in records:
        key=camera_key(record);require(key not in cameras,'Camera duplicates prior full-population schedule');cameras.add(key);counts[original_target_key(record)]+=1
    require(max(counts.values())<=200,'Original source family/component200view cap exceeded across prior schedule and clones')
    pins={**data['source_bindings'],**prior_pins,**implementation_bindings(),str(cp.resolve()):sha256(cp),str(Path(plan['source_pose_cache']['path']).resolve()):plan['source_pose_cache']['sha256']};verify_bindings(pins)
    return dict(anchor=data['anchor'],profile=data['profile'],records=records,source_bindings=pins,capture_split=split,scene_policy=plan['scene_policy'],pose_anchor_sources=data['pose_anchor_sources'])
