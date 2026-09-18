"""Source-authenticated no-query 9 mm pilot poses; original family splits remain fixed."""
from pathlib import Path
from copy import deepcopy
from collections import Counter
import numpy as np
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256
from . import native848_bulk_plan_v1 as old
from .native848_bulk_io_v1 import save_json
SCHEMA='greenhouse.native848_unique9mm_pilot_plan.v1'
CACHE_SCHEMA='greenhouse.native848_unique9mm_pilot_poses.v1'
ANCHOR_SCHEMA='greenhouse.native848_pilot_raw_anchor.v1'
CONTEXT_SCHEMA='greenhouse.native848_unique9mm_pilot_context.v1'
OBSERVATION_SCHEMA='greenhouse.native848_unique9mm_pilot_observation.v1'
RESULT_STATE='unique9mm_pilot_captured_pending_complete_census_and_independent_admission'
POLICY_SCHEMA='greenhouse.native848_pilot_scene_policy.v1'
SPLITS=('train','validation','test')
ORIGINAL_PROFILE_SCHEMA=old.ORIGINAL_PROFILE_SCHEMA
PAUSED_CLOCK_POLICY=old.PAUSED_CLOCK_POLICY
NATIVE_TIME_POLICY=old.NATIVE_TIME_POLICY
SCHEDULER_CLOCK_POLICY=old.SCHEDULER_CLOCK_POLICY
check_profile=old.check_profile
geometry_pose_key=old.geometry_pose_key
ROOT=Path(__file__).resolve().parents[3]

def merge(dst,values):
    for p,h in values.items():require(p not in dst or dst[p]==h,'Conflicting source pin');dst[p]=h

def load_source(directory,sample_id):
    """The old loader's immutable/native checks, with actual original split authentication."""
    directory=Path(directory).resolve();require(sample_id.startswith('sample_') and sample_id[7:].isdigit(),'Exact native sample ID required')
    mp=directory/'manifest.json';manifest=read_json(mp);sp=directory/sample_id/'sample.json';sample=read_json(sp)
    require(manifest.get('state')=='pilot_ready_for_review' and manifest.get('source_assets_unchanged') is True,'Completed unchanged native source required')
    require(manifest.get('target_family_split') in SPLITS and any(x['sample_id']==sample_id for x in manifest['samples']),'Original split/sample required')
    require(sample.get('schema_version')=='greenhouse.rgbd_pilot_sample.v2' and sample['calibration']['resolution']==[848,408],'Original native848 sample required')
    require(sample['synchronization']['scene_unchanged_during_capture'] is True and sample['synchronization']['dynamic_recording_supported'] is False,'Static original source required')
    pp=Path(manifest['source_collection_plan_path']).resolve();require(sha256(pp)==manifest['source_collection_plan_sha256'],'Source plan changed');source=read_json(pp)
    jobs=[j for j in source['jobs'] if j['job_id']==manifest['collection_job_id']];require(len(jobs)==1,'Unique original source job required');job=jobs[0]
    require(job['split']==manifest['target_family_split']==source['family_assignments'][job['plant_family']],'Original split assignment differs')
    rows=[r for r in job['targets'] if r['target_id']==sample['supervision']['target_id']]
    require(len(rows)==1 and rows[0]['draft_id']==sample['supervision']['review_id'] and rows[0]['cut_region_proposal']==sample['supervision']['cut_region_proposal'],'Original source geometry differs')
    pins={str(mp):sha256(mp),str(sp):sha256(sp),str(pp):sha256(pp),**manifest['source_usd_sha256']}
    for relative,receipt in sample['files'].items():
        path=(sp.parent/relative).resolve();require(path.is_relative_to(sp.parent),'Source sample escape');pins[str(path)]=receipt['sha256']
    verify_bindings(pins);return manifest,sample,pins

def raw_anchor(directory,sample_id):
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    manifest,sample,pins=load_source(directory,sample_id);source=read_json(manifest['source_collection_plan_path'])
    job=next(j for j in source['jobs'] if j['job_id']==manifest['collection_job_id']);row=next(r for r in job['targets'] if r['target_id']==sample['supervision']['target_id']);family=job['plant_family']
    variants=[v for v in manifest['variants'] if v['variant_id']==row['variant_id']];require(len(variants)==1,'Exact source variant required');variant=variants[0]
    require(variant['source_plant_id']==family and variant['split_group']==family and variant['source_geometry_modified'] is False and not variant['added_components'] and not variant['added_component_paths'],'Unmodified original plant required')
    pose=sample['robot_snapshot'];cal=sample['calibration'];camera=np.asarray(pose['robot_root_to_world_usd_row_vectors']).T@Rby1Kinematics().all_link_transforms(pose['joint_degrees'])['link_head_2']@np.asarray(pose['camera_to_head_column_vectors'])
    require(np.allclose(camera.T,cal['camera_to_world_usd_row_vectors'],atol=1e-8,rtol=0),'Original camera FK differs')
    require(pose['joint_limits_checked'] is True and pose['visual_bound_screen']['passed'] is True,'Source robot geometry must pass')
    package=Path(source['package']);require(package.resolve()==Path(manifest['package']).resolve(),'Source package mismatch')
    scene_code={str(p.resolve()):sha256(p) for p in sorted((package/'env_panel/tomato_env').glob('*.py'))};require(scene_code,'Pinned source daylight required')
    return dict(schema_version=ANCHOR_SCHEMA,state='authenticated_raw_geometry_no_acceptance_inherited',resolution=[848,408],source_capture=str(Path(directory).resolve()),source_sample=sample_id,source_collection_plan=str(Path(manifest['source_collection_plan_path']).resolve()),source_row=deepcopy(row),source_family=family,original_variant=deepcopy(variant),split=job['split'],split_group=family,expected_scene_counts=deepcopy(manifest['scene_counts']),expected_robot_snapshot=deepcopy(pose),expected_calibration=deepcopy(cal),expected_original_world={k:deepcopy(sample['supervision'][k]) for k in ('plant_to_world_usd_row_vectors','nominal_world_m','interval_world_m')},scene_variants=deepcopy(manifest['variants']),scene_code_bindings=scene_code,source_bindings={**pins,**scene_code},generated_geometry_used=False,training_approved=False,source_visual_acceptance_inherited=False,source_workspace_acceptance_inherited=False)

def scene_policy(split):
    from .native848_pilot_scene_v1 import policy
    return policy(split)

def implementation_bindings():
    paths=[Path(__file__),Path(old.__file__)]
    for name in ('native848_pilot_scene_v1.py','native848_pilot_worker_v1.py'):
        p=Path(__file__).with_name(name)
        require(p.is_file(),'Pilot scene and worker must exist before plan sealing');paths.append(p)
    return {str(p.resolve()):sha256(p) for p in paths}

def validate_raw_cache(cache):
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    from .collection_plan import load_plan
    from .native848_pair_audit_v2 import world_from_row
    from .native_view_pose import bounded_reference_root
    from .capture_contract import project
    require(cache['schema']==CACHE_SCHEMA and cache['resolution']==[848,408] and cache['training_approved'] is False,'Explicit pilot raw cache required')
    ap=Path(cache['anchor_reference_path']).resolve();require(sha256(ap)==cache['anchor_reference_sha256'],'Raw anchor changed');anchor=read_json(ap)
    require(anchor==raw_anchor(anchor['source_capture'],anchor['source_sample']),'Raw anchor fields differ')
    source,reports=load_plan(anchor['source_collection_plan']);split=anchor['split'];family=anchor['source_family'];rows={r['target_id']:r for j in source['jobs'] if j['plant_family']==family and j['split']==split for r in j['targets']}
    model=Rby1Kinematics();ref=anchor['expected_robot_snapshot'];pins={};merge(pins,cache['source_bindings']);merge(pins,anchor['source_bindings']);merge(pins,{str(ap):sha256(ap)})
    cameras=set()
    for rec in cache['records']:
        require(rec['target_id'] in rows and rec['source_row']==rows[rec['target_id']] and rec['source_family']==family and rec['split']==split,'Record target/family/split mismatch')
        require(rec['target_id']!='seed41_full/SubStem_38','Banned target cannot be scheduled')
        require(rec['anchor_reference_path']==str(ap) and rec['anchor_reference_sha256']==sha256(ap),'Record anchor differs')
        require(Path(rec['source_collection_plan']).resolve()==Path(anchor['source_collection_plan']).resolve() and rec['source_collection_plan_sha256']==sha256(anchor['source_collection_plan']),'Record source plan differs')
        world=world_from_row(anchor,rec['source_row']);require(np.allclose(rec['target_world_m'],world['nominal_world_m'],atol=1e-9,rtol=0) and np.allclose(rec['interval_world_m'],world['interval_world_m'],atol=1e-9,rtol=0),'Original geometry changed')
        root=bounded_reference_root(ref,rec['requested_spec'],rec['target_world_m']);mount=np.asarray(rec['camera_to_head_column_vectors'])
        require(np.allclose(root,np.asarray(rec['robot_root_to_world_usd_row_vectors']).T,atol=1e-9,rtol=0) and np.allclose(mount,ref['camera_to_head_column_vectors'],atol=1e-9,rtol=0),'Root/mount differs')
        require(set(rec['joint_degrees'])==set(ref['joint_degrees']) and all(rec['joint_degrees'][k]==v for k,v in ref['joint_degrees'].items() if k not in ('head_0','head_1')),'Original body/arms changed')
        camera=root@model.all_link_transforms(rec['joint_degrees'])['link_head_2']@mount;cal=rec['calibration'];require(set(cal)==set(anchor['expected_calibration']),'Calibration keys differ')
        require(np.allclose(camera.T,cal['camera_to_world_usd_row_vectors'],atol=1e-9,rtol=0) and np.allclose(camera.T,rec['camera_to_world_usd_row_vectors'],atol=1e-9,rtol=0),'Camera FK mismatch')
        require(all(cal[k]==anchor['expected_calibration'][k] for k in cal if k!='camera_to_world_usd_row_vectors'),'Optics changed')
        key=tuple(camera.T.round(10).reshape(-1));require(key not in cameras,'Duplicate camera');cameras.add(key)
        projected=project([world['nominal_world_m']],cal)[0];require(projected['projection_status']=='in_frame' and np.linalg.norm(np.asarray(projected['pixel_xy'])-rec['requested_spec']['desired_pixel_xy'])<2,'Framing mismatch')
        require(rec['minimum_size_met'] is True and rec['predicted_diameter_px']>=8 and rec['projected_interval_length_px']>=12,'Original detail minimum failed')
        require(rec['full_scene_cpu_preflight']['screen']['passed'] is True,'Full CPU geometry required')
    verify_bindings(pins);return dict(cache=cache,anchor=anchor,source_plan=source,reports=reports,records=cache['records'],source_bindings=pins)

def _source(spec,mode):
    p=Path(spec['path']).resolve();require(sha256(p)==spec['sha256'],'Pilot source changed');value=read_json(p)
    if mode=='authenticated_legacy_plan':
        if value['schema']=='greenhouse.original848_multianchor_bulk_plan.v1':
            from .native848_multianchor_plan_v1 import check as checked
        else:checked=old.check
        data=checked(value);profile=value['profile_evidence']
    elif mode=='authenticated_all_split_cache':
        data=validate_raw_cache(value);profile=value['profile_evidence']
    else:raise ValueError('Unknown explicit pilot source mode')
    data=deepcopy(data);merge(data['source_bindings'],{str(p):spec['sha256']});return data,profile

def prepare(source,selected_sample_ids,output,*,source_mode='authenticated_legacy_plan'):
    output=Path(output).resolve();require(output.is_relative_to(ROOT/'data/sim_data/diagnostics') and not output.exists(),'Create-only pilot diagnostics required')
    data,profile=_source(source,source_mode);by={r['sample_id']:r for r in data['records']};ids=list(selected_sample_ids);require(ids and len(ids)==len(set(ids)) and set(ids)<=by.keys(),'Unique selected source poses required');records=[by[x] for x in ids]
    split=data['anchor']['split'];require(all(r['split']==split for r in records),'Mixed source split')
    output.mkdir(parents=True);cp=output/'pose_cache.json';save_json(cp,dict(schema=CACHE_SCHEMA,records=records,anchor_reference_path=data['cache']['anchor_reference_path'],anchor_reference_sha256=data['cache']['anchor_reference_sha256'],source_bindings=data['source_bindings'],resolution=[848,408],training_approved=False))
    plan=dict(schema=SCHEMA,source=source,source_mode=source_mode,cache_path=str(cp),cache_sha256=sha256(cp),profile_evidence=profile,capture_split=split,pilot_scene_policy=scene_policy(split),selected_sample_ids=ids,schedule=[dict(observation_id='pilot9_'+x,source_pose_id=x,capture_role='qualification_candidate') for x in ids],max_frames=len(ids),resolution=[848,408],instance_backend='fast',queue_frames=8,queue_bytes=256*2**20,writer_threads=2,geometry_policy='native_full_screen_each_new_complete_joint_pose',annotation_contract='no_query_exactly_one_eligible_petiole_nominal_9mm',per_target_max_planned_views=20,training_approved=False,automatic_retries=False,source_acceptance_inherited=False,scene_preflight_required=True,implementation_bindings=implementation_bindings())
    check(plan);pp=output/'plan.json';save_json(pp,plan);result=dict(schema='greenhouse.native848_unique9mm_pilot_prepared.v1',plan={'path':str(pp),'sha256':sha256(pp)},frames=len(ids),capture_split=split,target_counts=dict(Counter(r['target_id'] for r in records)),native_launched=False,accepted_increment=0);save_json(output/'result.json',result);return result

def check(plan):
    require(plan['schema']==SCHEMA and plan['implementation_bindings']==implementation_bindings(),'Pilot implementation changed')
    split=plan['capture_split'];require(split in SPLITS and plan['pilot_scene_policy']==scene_policy(split),'Pilot scene/split policy differs')
    require(plan['resolution']==[848,408] and plan['instance_backend']=='fast' and plan['queue_frames']==8 and plan['queue_bytes']==256*2**20 and plan['writer_threads']==2,'Native format/queue differs')
    require(plan['scene_preflight_required'] is True and plan['annotation_contract']=='no_query_exactly_one_eligible_petiole_nominal_9mm' and plan['geometry_policy']=='native_full_screen_each_new_complete_joint_pose' and plan['per_target_max_planned_views']==20 and all(plan[x] is False for x in ('training_approved','automatic_retries','source_acceptance_inherited')),'Pilot scope changed')
    data,profile_spec=_source(plan['source'],plan['source_mode']);require(data['anchor']['split']==split,'Source split differs');require(profile_spec==plan['profile_evidence'],'Profile source differs');require(sha256(profile_spec['path'])==profile_spec['sha256'],'Profile changed');profile=read_json(profile_spec['path']);check_profile(profile)
    cp=Path(plan['cache_path']);require(sha256(cp)==plan['cache_sha256'],'Pilot selected cache changed');cache=read_json(cp);by={r['sample_id']:r for r in data['records']};ids=plan['selected_sample_ids'];require(len(ids)==len(set(ids))==plan['max_frames'] and 1<=len(ids)<=512 and set(ids)<=by.keys(),'Unique bounded poses required');records=[by[x] for x in ids]
    require(cache['records']==records and cache['source_bindings']==data['source_bindings'] and cache['anchor_reference_path']==data['cache']['anchor_reference_path'] and cache['anchor_reference_sha256']==data['cache']['anchor_reference_sha256'],'Pilot subset differs')
    require(plan['schedule']==[dict(observation_id='pilot9_'+x,source_pose_id=x,capture_role='qualification_candidate') for x in ids],'Pilot schedule changed')
    counts=Counter(r['target_id'] for r in records);require(max(counts.values())<=20 and 'seed41_full/SubStem_38' not in counts and all(r['split']==split for r in records),'Target cap/ban/split mismatch')
    pins=dict(data['source_bindings']);merge(pins,profile['source_bindings']);merge(pins,plan['implementation_bindings']);merge(pins,{str(cp.resolve()):sha256(cp),str(Path(profile_spec['path']).resolve()):profile_spec['sha256']});verify_bindings(pins)
    return dict(data,cache=cache,profile=profile,records=records,source_bindings=dict(sorted(pins.items())),pilot_scene_policy=plan['pilot_scene_policy'],capture_split=split)
