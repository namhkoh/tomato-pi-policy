"""Exactly two reviewed production6 reference views; merged-ID diagnostic only."""
from pathlib import Path
from copy import deepcopy
from . import native848_reset6_production_plan_v1 as baseline
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256
from .native848_bulk_io_v1 import save_json

D=Path('D:/research/tomato-pi-policy/data/sim_data/diagnostics')
SCHEMA='greenhouse.native848_farmerge_pair_plan.v1'
CPU_SCHEMA='greenhouse.native848_farmerge_pair_CPU.v1'
CONTEXT_SCHEMA='greenhouse.native848_farmerge_pair_context.v1'
OBSERVATION_SCHEMA='greenhouse.native848_farmerge_pair_observation.v1'
RESULT_STATE='farmerge_pair_captured_pending_native_coarse_ID_RGBD_quality_comparison_no_training_admission'
ROLE='farmerge_equivalence_diagnostic_no_training_admission'
MANIFEST_SCHEMA='greenhouse.native848_farmerge_pair_manifest.v1'
CAPTURE_SCHEMA='greenhouse.native848_farmerge_pair_capture.v1'
PROFILE_SCHEMA=baseline.PROFILE_SCHEMA
PAUSED_CLOCK_POLICY=baseline.PAUSED_CLOCK_POLICY
NATIVE_TIME_POLICY=baseline.NATIVE_TIME_POLICY
SCHEDULER_CLOCK_POLICY=baseline.SCHEDULER_CLOCK_POLICY
geometry_pose_key=baseline.geometry_pose_key
check_profile=baseline.check_profile
ORIGINAL_PLAN=dict(path=str(D/'native848_reset6_production_train67_36_prepare_20260917_v1/plan.json'),sha256='5280df09194d841774acf8c984f996fc980e027ef43b7fafd28e7be3f520e4cc')
ORIGINAL_CPU=dict(path=str(D/'native848_reset6_production_train67_36_prepare_20260917_v1/cpu_preflight.json'),sha256='f9702b2a4982792f444f94a078f0813a434250543ca007d370b36d3971d149b8')
PROTOTYPE=dict(path=str(D/'native848_far_component_merge67_36_CPU_20260917_v1/result.json'),sha256='cf9917ea3127d1ab2e73b892cf6736a941395a51bab4a15da7caa6a18295075b')
CLOSURE=dict(path=str(D/'native848_far_component_merge67_36_CPU_closure_20260917_v1/result.json'),sha256='552f0467f319af3253f9b7966c1d7826dafc1c6e8c8840390e31c6f0f22a2a2e')


def read(spec):return baseline.read(spec)


def metadata(split,*,native=False):
    require(split=='train','Only exact TRAIN67 experiment is supported')
    return dict(dataset_split=split,annotation_coverage=('complete_logical_anatomy_near_component_IDs_far_coarse_groups_native_census_verified' if native else
        'complete_logical_anatomy_near_component_IDs_far_coarse_groups_pending_native_census'),native_population_census_verified=native,
        no_query9mm_acceptance_pending=True,filtered_scene_annotation_export_eligible=False,diagnostic_equivalence_replay=True,
        training_export_eligible=False,ordinary_component_ID_annotation_compatible=False)


def policy():
    return dict(schema='greenhouse.native848_farmerge_pair_scene_policy.v1',plant_slots=144,logical_components=61386,
        render_meshes=11177,coarse_far_groups=143,original_collision_representation_retained=True,
        original_foreground_and_all_protected_dependencies_unchanged=True,world_coordinate_roundoff_bound_m=1e-6,
        welding=False,decimation=False,source_assets_modified=False,all36_pose_both_arm_exclusions_required=True,
        native_coarse_ID_qualification_pending=True)


def implementation_bindings():
    from . import native848_farmerge_pair_scene_v1 as scene
    require(sha256(baseline.__file__)=='75df5eb3eeeff110040d7ae9052d7e0b6211d83e19adb17980709b7119e6da23','Frozen production6 plan changed')
    return {str(Path(m).resolve()):sha256(m) for m in (__file__,baseline.__file__,scene.__file__)}


def check(plan):
    require(plan['schema']==SCHEMA and plan['implementation_bindings']==implementation_bindings(),'Typed diagnostic implementation changed')
    original=read(ORIGINAL_PLAN);data=baseline.check(original);cpu=read(ORIGINAL_CPU)
    require(plan['baseline_plan']==ORIGINAL_PLAN and plan['baseline_CPU']==ORIGINAL_CPU and plan['prototype']==PROTOTYPE
        and plan['prototype_closure']==CLOSURE and plan['scene_policy']==policy(),'Exact finite prototype lineage required')
    closed=read(CLOSURE);prototype=read(PROTOTYPE)
    require(closed['passed'] is True and closed['prototype_result']==PROTOTYPE and closed['native_capture_authorized'] is False
        and prototype['passed'] is True and prototype['actual_composed_render_meshes']==11177
        and prototype['max_world_coordinate_euclidean_error_m']<=1e-6,'Successful CPU packing closure required, never native qualification')
    ids=plan['selected_sample_ids'];by={r['sample_id']:r for r in data['records']}
    require(len(ids)==len(set(ids))==plan['max_frames']==2 and set(ids)<=by.keys(),'Exactly two distinct existing36 source views required')
    records=[by[k] for k in ids]
    require(read(dict(path=plan['cache_path'],sha256=plan['cache_sha256']))['records']==records,'Two-view selected cache differs')
    for key in ('profile_evidence','source_pose_profile_evidence','source_geometry_plan','source_geometry_preflight','capture_split','resolution','instance_backend','queue_frames','queue_bytes','writer_threads','automatic_retries','source_acceptance_inherited'):
        require(plan[key]==original[key],'Paired renderer/camera/source contract changed: '+key)
    require(plan['schedule']==[dict(observation_id='farmerge_pair_'+r['sample_id'],source_pose_id=r['sample_id'],capture_role=ROLE) for r in records]
        and all(plan.get(k)==v for k,v in metadata('train').items()) and plan['training_approved'] is False
        and plan['accepted_training_increment']==0 and plan['annotation_contract']=='farmerge_equivalence_only_no_training_admission'
        and plan['geometry_policy']=='unchanged_original_collision_stage_at_exact_actual_native_robot_pose'
        and plan['scene_preflight_required'] is True,'Explicit diagnostic-only capture contract required')
    require(plan['paired_reference_selection']['sha256']=='e1e861a878aa275f796f4a552d70ea747269303cd4db30f1b3eab75edfc16433','Only the actually reviewed most-separated pair is authorized as reference')
    selection=read(plan['paired_reference_selection'])
    require(selection['schema']=='greenhouse.native848_merged_diagnostic_reference_selection.v1'
        and selection['reference_selection_only'] is True and selection['native_capture_authorized'] is False
        and selection['new_samples_generated']==selection['accepted_training_increment']==0
        and selection['export_performed'] is False and selection['broad_render_profile_approval'] is False,
        'Actual reviewed pair reference selection required')
    pins=baseline.merge(data['source_bindings'],closed['source_bindings'],prototype['source_bindings'],implementation_bindings())
    for spec in (ORIGINAL_PLAN,ORIGINAL_CPU,PROTOTYPE,CLOSURE,plan['paired_reference_selection'],dict(path=plan['cache_path'],sha256=plan['cache_sha256'])):
        read(spec);pins[str(Path(spec['path']).resolve())]=spec['sha256']
    reference_pins=[selection[k] for k in ('source_trial_result','source_manifest','source_inventory','actual_visual_review_ledger',
        'actual_visual_group_review','actual_visual_review_validation')]
    require(set(selection['source_closure'])=={'metadata','result','terminal','outer_launch','outer_exit'}
        and selection['source_closure']['result']==selection['source_trial_result'],'Exact reference closure pins required')
    reference_pins.extend(selection['source_closure'].values())
    for spec in reference_pins:
        read(spec);pins[str(Path(spec['path']).resolve())]=spec['sha256']
    outer=read(selection['source_closure']['outer_exit']);launch=read(selection['source_closure']['outer_launch'])
    terminal=read(selection['source_closure']['terminal'])
    require(outer['returncode']==0 and outer['method']=='subprocess_wait_on_owned_process'
        and outer['pid']==launch['pid'] and outer['launch_sha256']==selection['source_closure']['outer_launch']['sha256']
        and terminal['result_sha256']==selection['source_trial_result']['sha256'] and terminal['capture_only'] is True,
        'Actual completed reference outer wait and terminal required')
    frames=selection['records'];require(len(frames)==2 and [f['source_pose_id'] for f in frames]==ids
        and len(selection['accepted_reference_observation_ids'])==8
        and all(f['observation_id'] in selection['accepted_reference_observation_ids'] for f in frames),'Reviewed pair order differs')
    for frame,source_id in zip(frames,ids):
        meta=read(frame['native_observation']);require(meta['source_pose_id']==source_id and meta['observation_id']==frame['observation_id'],'Reference frame identity differs')
        require(meta['profile']==data['profile']['profile'] and meta['request_evidence']['requested_subframes']==6
            and meta['files']=={k:frame[k] for k in ('rgb','buffers')},'Reference actual render budget or callback assets differ')
        for key in ('metadata','native_observation','rgb','buffers','annotation','ambiguity','actual_visual_review'):
            spec=frame[key];require(sha256(spec['path'])==spec['sha256'],'Changed paired reference asset');pins[str(Path(spec['path']).resolve())]=spec['sha256']
    verify_bindings(pins)
    return dict(data,records=records,source_bindings=pins,scene_policy=policy(),original_scene_policy=original['scene_policy'],
        prototype=prototype,prototype_closure=closed,original_CPU=cpu,paired_reference_selection=selection)


def validate_cpu(cpu,plan,plan_path,plan_sha):
    original=read(ORIGINAL_CPU)
    require(cpu['schema']==CPU_SCHEMA and cpu['plan_sha256']==plan_sha and Path(cpu['plan_path']).resolve()==Path(plan_path).resolve()
        and cpu['prototype']==PROTOTYPE and cpu['prototype_closure']==CLOSURE and cpu['original_collision_CPU']==ORIGINAL_CPU
        and cpu['scene_policy']==policy() and cpu['selected_sample_ids']==plan['selected_sample_ids']
        and cpu['poses']==[next(p for p in original['poses'] if p['sample_id']==i) for i in plan['selected_sample_ids']]
        and cpu['all_selected_pose_screens_passed'] is True and cpu['actual_native_render_census_pending'] is True
        and cpu['source_collision_geometry_reused_not_recomputed'] is True,'Explicit CPU packing plus unchanged collision proof required')
    verify_bindings(cpu['source_bindings'])


def prepare(reference_selection,output):
    output=Path(output).resolve();require(output.is_relative_to(D) and not output.exists(),'Create-only exact pair plan required')
    original=read(ORIGINAL_PLAN);data=baseline.check(original);selection=read(reference_selection);ids=[f['source_pose_id'] for f in selection['records']]
    by={r['sample_id']:r for r in data['records']};require(len(ids)==len(set(ids))==2 and set(ids)<=by.keys(),'Exactly2 original selected poses')
    output.mkdir(parents=True);cache=output/'pose_cache.json';save_json(cache,dict(schema='greenhouse.native848_farmerge_pair_poses.v1',records=[by[i] for i in ids],training_approved=False))
    plan={k:deepcopy(original[k]) for k in ('profile_evidence','source_pose_profile_evidence','source_geometry_plan','source_geometry_preflight','capture_split','resolution','instance_backend','queue_frames','queue_bytes','writer_threads','automatic_retries','source_acceptance_inherited')}
    plan.update(schema=SCHEMA,**metadata('train'),training_approved=False,accepted_training_increment=0,baseline_plan=ORIGINAL_PLAN,baseline_CPU=ORIGINAL_CPU,
        prototype=PROTOTYPE,prototype_closure=CLOSURE,paired_reference_selection=reference_selection,scene_policy=policy(),max_frames=2,
        selected_sample_ids=ids,schedule=[dict(observation_id='farmerge_pair_'+i,source_pose_id=i,capture_role=ROLE) for i in ids],
        cache_path=str(cache),cache_sha256=sha256(cache),scene_preflight_required=True,annotation_contract='farmerge_equivalence_only_no_training_admission',
        geometry_policy='unchanged_original_collision_stage_at_exact_actual_native_robot_pose',implementation_bindings=implementation_bindings())
    checked=check(plan);pp=output/'plan.json';save_json(pp,plan);original_cpu=read(ORIGINAL_CPU)
    cpu=dict(schema=CPU_SCHEMA,**metadata('train'),training_approved=False,accepted_training_increment=0,plan_path=str(pp),plan_sha256=sha256(pp),
        prototype=PROTOTYPE,prototype_closure=CLOSURE,original_collision_CPU=ORIGINAL_CPU,scene_policy=policy(),selected_sample_ids=ids,
        poses=[next(p for p in original_cpu['poses'] if p['sample_id']==i) for i in ids],pose_anchor_sources=checked['pose_anchor_sources'],
        all_selected_pose_screens_passed=True,actual_native_render_census_pending=True,source_collision_geometry_reused_not_recomputed=True,
        logical_original_census=original_cpu['full_scene_census'],source_bindings=checked['source_bindings'])
    validate_cpu(cpu,plan,pp,sha256(pp));cp=output/'cpu_preflight.json';save_json(cp,cpu)
    return dict(plan=dict(path=str(pp),sha256=sha256(pp)),cpu=dict(path=str(cp),sha256=sha256(cp)))
