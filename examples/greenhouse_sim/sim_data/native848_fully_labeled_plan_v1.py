"""Create-only full144 labeled-population plans from authenticated bounded poses."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
from . import native848_pilot_plan_v1 as source_api
from . import native848_fully_labeled_scene_v1 as scene_api
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256
from .native848_bulk_io_v1 import save_json
SCHEMA='greenhouse.native848_fully_labeled_plan.v1'
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

def _source(spec):
    path=Path(spec['path']).resolve();require(sha256(path)==spec['sha256'],'Authenticated source plan changed')
    value=read_json(path);data=source_api.check(value)
    return value,data

def prepare(source,selected_sample_ids,output):
    output=Path(output).resolve();require(output.is_relative_to(ROOT/'data/sim_data/diagnostics') and not output.exists(),'Fresh diagnostic plan folder required')
    old,data=_source(source);by={r['sample_id']:r for r in data['records']};ids=list(selected_sample_ids)
    require(ids and len(ids)==len(set(ids)) and set(ids)<=by.keys(),'Unique authenticated source poses required')
    split=data['capture_split'];records=[normalized_record(by[i]) for i in ids]
    output.mkdir(parents=True);cache=output/'pose_cache.json';save_json(cache,dict(schema='greenhouse.native848_fully_labeled_poses.v1',records=records,training_approved=False))
    plan=dict(schema=SCHEMA,source_pose_plan=source,cache_path=str(cache),cache_sha256=sha256(cache),
        profile_evidence=old['profile_evidence'],capture_split=split,scene_policy=scene_policy(split),
        **scene_api.metadata(split),native_population_census_verified=False,
        selected_sample_ids=ids,schedule=[dict(observation_id='full144_'+i,source_pose_id=i,capture_role='fully_labeled_qualification_candidate') for i in ids],
        max_frames=len(ids),resolution=[848,408],instance_backend='fast',queue_frames=8,queue_bytes=256*2**20,writer_threads=2,
        geometry_policy='native_full_screen_each_new_complete_joint_pose',annotation_contract='no_query_exactly_one_eligible_petiole_nominal_9mm',
        per_target_max_planned_views=20,automatic_retries=False,source_acceptance_inherited=False,scene_preflight_required=True,
        implementation_bindings=implementation_bindings())
    check(plan);pp=output/'plan.json';save_json(pp,plan)
    return dict(path=str(pp),sha256=sha256(pp))

def check(plan):
    require(plan.get('schema')==SCHEMA and plan['implementation_bindings']==implementation_bindings(),'New population plan implementation differs')
    split=plan['capture_split'];require(plan['scene_policy']==scene_policy(split),'Full144 population policy differs')
    require(all(plan.get(k)==v for k,v in scene_api.metadata(split).items()) and plan['native_population_census_verified'] is False,'Pending population admission metadata required')
    require(plan['resolution']==[848,408] and plan['instance_backend']=='fast' and plan['queue_frames']==8 and plan['queue_bytes']==256*2**20 and plan['writer_threads']==2,'Native format differs')
    require(plan['scene_preflight_required'] is True and plan['annotation_contract']=='no_query_exactly_one_eligible_petiole_nominal_9mm' and plan['geometry_policy']=='native_full_screen_each_new_complete_joint_pose' and plan['per_target_max_planned_views']==20 and all(plan[k] is False for k in ('automatic_retries','source_acceptance_inherited')),'Scope differs')
    old,data=_source(plan['source_pose_plan']);require(data['capture_split']==split and old['profile_evidence']==plan['profile_evidence'],'Source split/profile differs')
    by={r['sample_id']:r for r in data['records']};ids=plan['selected_sample_ids'];require(len(ids)==len(set(ids))==plan['max_frames'] and 1<=len(ids)<=512 and set(ids)<=by.keys(),'Finite exact source subset required')
    records=[normalized_record(by[i]) for i in ids];cp=Path(plan['cache_path']);require(sha256(cp)==plan['cache_sha256'] and read_json(cp)['records']==records,'Normalized pose cache differs')
    require(plan['schedule']==[dict(observation_id='full144_'+i,source_pose_id=i,capture_role='fully_labeled_qualification_candidate') for i in ids],'New population schedule differs')
    require(max(Counter(r['target_id'] for r in records).values())<=20,'Physical target plan cap exceeded')
    pins={**data['source_bindings'],**implementation_bindings(),str(cp.resolve()):sha256(cp),str(Path(plan['source_pose_plan']['path']).resolve()):plan['source_pose_plan']['sha256']};verify_bindings(pins)
    return dict(anchor=data['anchor'],profile=data['profile'],records=records,source_bindings=pins,capture_split=split,scene_policy=plan['scene_policy'])
