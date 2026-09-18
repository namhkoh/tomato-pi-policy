"""Flat original848 bulk contract; explicit user-authorized uncapped source views."""
from pathlib import Path
import ast
import importlib.util
import numpy as np
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256

SCHEMA = 'greenhouse.original848_bulk_plan.v1'
OBSERVATION_SCHEMA = 'greenhouse.original848_bulk_observation.v1'
CONTEXT_SCHEMA = 'greenhouse.original848_bulk_context.v1'
PROFILE_SCHEMA = 'greenhouse.native848_bulk_profile_qualification.v1'
ORIGINAL_PROFILE_SCHEMA = 'greenhouse.native848_bulk_original_reset8_profile.v1'
PAUSED_CLOCK_POLICY = 'paused_timeline_zero_delta'
VIEW_POLICY = 'user_override_no_per_target_cap_total_train_goal20000'
RESULT_STATE = 'original848_bulk_captured_pending_independent_frame_admission'
NATIVE_TIME_POLICY = 'preserved_callback_clock_nondecreasing_not_frame_identity'
SCHEDULER_CLOCK_POLICY = 'observed_timeline_delta_after_initialization'
EXPLICIT_RENDER_SETTINGS = {'/rtx/rendermode':'RealTimePathTracing','/rtx/post/aa/op':2,
    '/rtx-transient/post/aa/limitedOps':False,'/rtx-transient/dlssg/enabled':False,
    '/omni/replicator/captureMotionBlur':False}


def geometry_pose_key(scene_revision, robot_snapshot, source_bindings_sha256):
    """Exact complete robot pose identity, including both head joints."""
    from .capture_contract import fingerprint
    root=np.asarray(robot_snapshot['robot_root_to_world_usd_row_vectors'],float)
    joints=robot_snapshot['joint_degrees']
    require(root.shape==(4,4) and np.isfinite(root).all() and joints
        and all(np.isfinite(float(v)) for v in joints.values()),'Finite complete geometry pose required')
    require(isinstance(scene_revision,str) and scene_revision and isinstance(source_bindings_sha256,str)
        and len(source_bindings_sha256)==64,'Explicit scene and source revision required')
    return fingerprint(dict(scene_revision=scene_revision,source_bindings_sha256=source_bindings_sha256,
        robot_root_to_world_usd_row_vectors=root.tolist(),joint_degrees=joints))


def implementation_bindings():
    root=Path(__file__).resolve().parent
    roots=(root.parent,root.parents[1])
    pending=[root/name for name in ('native848_bulk_plan_v1.py','native848_bulk_worker_v1.py','native848_bulk_io_v1.py')]
    found={}
    while pending:
        path=pending.pop().resolve()
        if str(path) in found:continue
        raw=path.read_bytes();found[str(path)]=sha256(path)
        base=next(p for p in roots if path.is_relative_to(p));package='.'.join(path.relative_to(base).parts[:-1])
        for node in ast.walk(ast.parse(raw)):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                name='.'*node.level+(node.module or '')
                name=importlib.util.resolve_name(name,package) if node.level else name
                names=[name]+[name+'.'+a.name for a in node.names if a.name!='*']
            for name in names:
                for folder in roots:
                    candidate=folder.joinpath(*name.split('.'))
                    pending.extend(p for p in (candidate.with_suffix('.py'),candidate/'__init__.py') if p.is_file())
    return dict(sorted(found.items()))


def check_profile(profile):
    """Explicit measured fast profile OR unchanged qualified original reset8."""
    require(profile['training_approved'] is False and profile['wait_for_render'] is True,
        'Profile never approves training and must wait for actual render completion')
    if profile['schema']==ORIGINAL_PROFILE_SCHEMA:
        q2=profile['original_qualification'];q1=profile['diagnostic_qualification']
        require(q2['sha256']=='4dee7a139c4c261780bae01d6dc71a9451405d5f1d387ebaf448d879140b73b4'
            and q1['sha256']=='f2e14930ee1202b326f195aad7a465649dbe316f1aa55364a0a428dce7ee0d69'
            and sha256(q2['path'])==q2['sha256'] and sha256(q1['path'])==q1['sha256'],
            'Exact existing original and diagnostic qualification required')
        original,diagnostic=read_json(q2['path']),read_json(q1['path'])
        require(original['qualified_for_original_production'] is True
            and original['runtime_adaptation_passed'] is True and original['individual_visual_review_passed'] is True
            and diagnostic['diagnostic_profile_qualified'] is True,'Existing native qualification failed')
        require(profile['qualified_for_original_production'] is True
            and profile['new_writer_integration_qualified'] is False
            and profile['profile']==original['profile'] and profile['request_subframes']==8
            and profile['warmup_steps']==[8]*7 and profile['render_settings']==diagnostic['render_settings']
            and profile['delta_time_seconds']==0 and profile['scheduler_clock_policy']==PAUSED_CLOCK_POLICY,
            'Unchanged original reset8 renderer and paused clock required')
        for item in (q1,q2):
            require(profile['source_bindings'].get(str(Path(item['path']).resolve()))==item['sha256'],
                'Original qualification missing from flat bindings')
    else:
        require(profile['schema']==PROFILE_SCHEMA and profile['qualified_for_bulk_production'] is True,
            'Actual qualified renderer profile required; candidate evidence cannot enable production')
        require(profile['delta_time_seconds']==1/60 and profile['scheduler_clock_policy']==SCHEDULER_CLOCK_POLICY,
            'Qualified advancing scheduler policy required')
        require(all(profile['render_settings'].get(k)==v for k,v in EXPLICIT_RENDER_SETTINGS.items()),
            'Only the actually qualified explicit renderer override is supported')
    require(isinstance(profile['profile'],str) and profile['profile']
        and type(profile['request_subframes']) is int and profile['request_subframes']>=1
        and isinstance(profile['warmup_steps'],list) and profile['warmup_steps']
        and all(type(n) is int and n>=1 for n in profile['warmup_steps'])
        and profile['reset_api']=='omni.usd.get_context().reset_renderer_accumulation'
        and isinstance(profile['render_settings'],dict) and profile['render_settings']
        and type(profile['delta_time_seconds']) in (int,float) and np.isfinite(profile['delta_time_seconds'])
        and profile['native_reference_time_policy']==NATIVE_TIME_POLICY,
        'Explicit renderer/reset/native clock policy required')
    required={'plan','capture_result','audit','visual_reviews','owned_exit'}
    require(required<=set(profile['evidence']),'Qualified profile lacks actual evidence')
    for name in required:
        item=profile['evidence'][name];path=str(Path(item['path']).resolve())
        require(profile['source_bindings'].get(path)==item['sha256'],'Profile evidence must be in flat binding closure')


def prepare(cache_path, profile_evidence, selected_sample_ids=None, *, queue_frames=8, queue_bytes=256*2**20, writer_threads=2):
    path=Path(cache_path).resolve();cache=read_json(path)
    ids=list(selected_sample_ids if selected_sample_ids is not None else cache['suggested_sample_ids'])
    plan=dict(schema=SCHEMA,cache_path=str(path),cache_sha256=sha256(path),profile_evidence=dict(profile_evidence),
        schedule=[dict(observation_id='bulk_'+name,source_pose_id=name,capture_role='production') for name in ids],
        selected_sample_ids=ids,max_frames=len(ids),resolution=[848,408],instance_backend='fast',
        view_policy=VIEW_POLICY,per_target_view_cap=None,total_train_goal=20000,training_approved=False,
        source_cap_reset=False,generated_geometry_used=False,geometry_novelty_qualified=False,
        queue_frames=queue_frames,queue_bytes=queue_bytes,writer_threads=writer_threads,
        geometry_policy='native_full_screen_each_new_complete_joint_pose',automatic_retries=False,
        implementation_bindings=implementation_bindings())
    check(plan)
    return plan


def check(plan):
    """One flat batch validation; never calls old Q/Q2 capture/audit replay."""
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    from .collection_plan import load_plan
    from .native848_pair_audit_v2 import world_from_row
    from .native_view_pose import bounded_reference_root
    from .capture_contract import project,fingerprint
    require(plan['schema']==SCHEMA and plan['resolution']==[848,408] and plan['instance_backend']=='fast'
        and plan['view_policy']==VIEW_POLICY and plan['per_target_view_cap'] is None and plan['total_train_goal']==20000
        and plan['geometry_policy']=='native_full_screen_each_new_complete_joint_pose'
        and all(plan[k] is False for k in ('training_approved','source_cap_reset','generated_geometry_used','geometry_novelty_qualified','automatic_retries')),
        'Changed explicit original bulk scope')
    require(type(plan['max_frames']) is int and 1<=plan['max_frames']<=plan['total_train_goal'],'Bounded total goal required')
    require(type(plan['queue_frames']) is int and plan['queue_frames']>=1 and type(plan['queue_bytes']) is int
        and plan['queue_bytes']>=5*2**20 and type(plan['writer_threads']) is int
        and 1<=plan['writer_threads']<=plan['queue_frames'],'Bounded positive writer queue required')
    require(plan['implementation_bindings']==implementation_bindings(),'Bulk code changed')
    require(sha256(plan['cache_path'])==plan['cache_sha256'],'Bulk pose cache changed')
    cache=read_json(plan['cache_path']);binding=plan['profile_evidence']
    require(sha256(binding['path'])==binding['sha256'],'Qualified profile changed')
    profile=read_json(binding['path']);check_profile(profile)
    require(cache['schema']=='greenhouse.original848_cached_embodied_camera_poses.v2'
        and cache['resolution']==[848,408] and cache['frozen_original_assets'] is True
        and all(cache[k] is False for k in ('generated_geometry_used','source_cap_reset','training_approved','native_launched')),
        'Prevalidated unmodified original camera cache required')
    require(sha256(cache['anchor_reference_path'])==cache['anchor_reference_sha256'],'Original anchor changed')
    anchor=read_json(cache['anchor_reference_path'])
    require(anchor['schema_version']=='greenhouse.original_native848_bank_reference.v2' and anchor['split']=='train','Original TRAIN scene anchor required')
    pins={}
    def merge(values):
        for p,h in values.items():
            require(p not in pins or pins[p]==h,'Conflicting flat source bindings');pins[p]=h
    for values in (plan['implementation_bindings'],cache['source_bindings'],profile['source_bindings'],anchor['source_bindings'],anchor['scene_code_bindings']):merge(values)
    merge({str(Path(plan['cache_path']).resolve()):plan['cache_sha256'],str(Path(binding['path']).resolve()):binding['sha256'],
        str(Path(cache['anchor_reference_path']).resolve()):cache['anchor_reference_sha256']})
    verify_bindings(pins)
    source_plan,reports=load_plan(anchor['source_collection_plan'])
    source_plan_sha256=sha256(anchor['source_collection_plan'])
    source_rows={r['target_id']:r for job in source_plan['jobs'] for r in job['targets']
        if job['plant_family']==anchor['source_family'] and job['split']=='train'}
    by={r['sample_id']:r for r in cache['records']};ids=plan['selected_sample_ids']
    require(len(by)==len(cache['records']) and len(ids)==len(set(ids))==plan['max_frames'] and set(ids)<=set(by),'Unique source poses required')
    require([r['source_pose_id'] for r in plan['schedule']]==ids and all(r['capture_role']=='production' for r in plan['schedule']), 'Production schedule differs')
    names=[r['observation_id'] for r in plan['schedule']]
    require(len(names)==len(set(names)) and all(n and Path(n).name==n and n not in ('.','..') for n in names),'Unique safe observations required')
    model=Rby1Kinematics();ref=anchor['expected_robot_snapshot'];camera_keys=set()
    for name in ids:
        r=by[name];target=r['target_id']
        require(target in source_rows and r['source_row']==source_rows[target] and r['source_target']==target
            and r['source_family']==anchor['source_family'] and r['split']=='train','Original target/family/split differs')
        require(r['anchor_reference_path']==cache['anchor_reference_path'] and r['anchor_reference_sha256']==cache['anchor_reference_sha256'],'Changed per-frame anchor')
        require(Path(r['source_collection_plan']).resolve()==Path(anchor['source_collection_plan']).resolve()
            and r['source_collection_plan_sha256']==source_plan_sha256,'Changed per-frame original source plan')
        world=world_from_row(anchor,r['source_row'])
        require(np.allclose(r['target_world_m'],world['nominal_world_m'],atol=1e-9,rtol=0)
            and np.allclose(r['interval_world_m'],world['interval_world_m'],atol=1e-9,rtol=0),'Changed source target geometry')
        root=bounded_reference_root(ref,r['requested_spec'],r['target_world_m']);mount=np.asarray(r['camera_to_head_column_vectors'])
        require(np.allclose(root,np.asarray(r['robot_root_to_world_usd_row_vectors']).T,atol=1e-9,rtol=0)
            and np.allclose(mount,ref['camera_to_head_column_vectors'],atol=1e-9,rtol=0),'Changed source root/mount')
        require(set(r['joint_degrees'])==set(ref['joint_degrees']) and all(r['joint_degrees'][k]==v for k,v in ref['joint_degrees'].items() if k not in ('head_0','head_1')),'Body/arm/torso source changed')
        camera=root@model.all_link_transforms(r['joint_degrees'])['link_head_2']@mount
        cal=r['calibration'];require(set(cal)==set(anchor['expected_calibration']),'Incomplete optical calibration')
        require(np.allclose(camera.T,cal['camera_to_world_usd_row_vectors'],atol=1e-9,rtol=0)
            and np.allclose(camera.T,r['camera_to_world_usd_row_vectors'],atol=1e-9,rtol=0),'Cached camera FK differs')
        require(all(cal[k]==anchor['expected_calibration'][k] for k in cal if k!='camera_to_world_usd_row_vectors'),'Optics changed')
        key=fingerprint(np.round(camera.T,10).tolist());require(key not in camera_keys,'Exact camera repeated within batch');camera_keys.add(key)
        projected=project([r['target_world_m']],cal)[0]
        require(projected['projection_status']=='in_frame' and np.linalg.norm(np.asarray(projected['pixel_xy'])-r['requested_spec']['desired_pixel_xy'])<2,'Source framing differs')
        require(r['minimum_size_met'] is True and r['predicted_diameter_px']>=8 and r['projected_interval_length_px']>=12,'Existing cut detail minimum failed')
    return dict(cache=cache,anchor=anchor,source_plan=source_plan,reports=reports,profile=profile,
        records=[by[name] for name in ids],source_bindings=dict(sorted(pins.items())))
