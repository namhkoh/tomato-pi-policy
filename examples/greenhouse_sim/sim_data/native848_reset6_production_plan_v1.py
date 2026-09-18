"""Truthful limited production reset6 rollout using unchanged full144 geometry.

Source pose plans retain their actual reset8 provenance. This adapter records a
new production6 profile, reuses exact CPU geometry proof explicitly, and grants
no dataset acceptance. The first rollout is bounded to the new TRAIN67 36 poses.
"""
from pathlib import Path
from copy import deepcopy
from . import native848_fully_labeled_plan_v3 as baseline
from .dataset_review import require, read_json, verify_bindings
from .depth_preview import sha256
from .native848_bulk_io_v1 import save_json

ROOT=Path(__file__).resolve().parents[3]
SCHEMA='greenhouse.native848_reset6_production_plan.v1'
PROFILE_SCHEMA='greenhouse.native848_reset6_limited_production_profile.v1'
CPU_SCHEMA='greenhouse.native848_reset6_reused_geometry_preflight.v1'
CONTEXT_SCHEMA=baseline.CONTEXT_SCHEMA
OBSERVATION_SCHEMA=baseline.OBSERVATION_SCHEMA
RESULT_STATE=baseline.RESULT_STATE
ORIGINAL_PROFILE_SCHEMA=baseline.ORIGINAL_PROFILE_SCHEMA
PAUSED_CLOCK_POLICY=baseline.PAUSED_CLOCK_POLICY
NATIVE_TIME_POLICY=baseline.NATIVE_TIME_POLICY
SCHEDULER_CLOCK_POLICY=baseline.SCHEDULER_CLOCK_POLICY
geometry_pose_key=baseline.geometry_pose_key
scene_policy=baseline.scene_policy
BASELINE_MODULE_SHA='74243fafa4712d97fde3795daa52f3711b8d9bd8135e06239dfb49316d802f66'
SOURCE_PLAN_SHA='5c57ee8216229c35450c28ffbca1dc6bd5b3c2db898f7585d70e5bf1fdbf341f'
SOURCE_CPU_SHA='23867cdc9d455e5e1af470c190612698ebdfdfdeb21f6900f352e071ea19aef9'
REFERENCE_PROFILE_SHA='253cf90516f65400896bc5c20502676434d5996f8c81c43dd341def28657134b'
NUMERIC_SHA='e05c7454e8691f92bf95ab92b0c56c536768072918ecf7790ec79a5dbdf857a9'
TRACE_SHA='3dea17a1101fe60d2510b8a03a7ef12bd5c80f701b2c143e36534ad11b464e2e'
VISUAL_SHA='6cdc38f9c82cf47452773dcfc6013c3bd7bcddcf4fb39310089cb3115d69c9fe'

def pin(path):
    path=Path(path).resolve()
    return dict(path=str(path),sha256=sha256(path))

def read(spec):
    require(set(spec)=={'path','sha256'},'Exact artifact pin required')
    require(sha256(spec['path'])==spec['sha256'],'Changed limited reset6 evidence')
    return read_json(spec['path'])

def merge(*maps):
    result={}
    for values in maps:
        for p,h in values.items():
            p=str(Path(p).resolve())
            require(p not in result or result[p]==h,'Conflicting source binding')
            result[p]=h
    return result

def implementation_bindings():
    require(sha256(baseline.__file__)==BASELINE_MODULE_SHA,'Frozen source plan changed')
    return {str(Path(__file__).resolve()):sha256(__file__),
            str(Path(baseline.__file__).resolve()):BASELINE_MODULE_SHA}

def qualification(evidence):
    require(set(evidence)=={'numeric','trace','visual'},'Exact finite qualification set required')
    require([evidence[k]['sha256'] for k in ('numeric','trace','visual')]==[NUMERIC_SHA,TRACE_SHA,VISUAL_SHA],
            'Only actual completed reset6/8 diagnostic evidence permitted')
    numeric,trace,visual=[read(evidence[k]) for k in ('numeric','trace','visual')]
    require(numeric['same_callback_assets_authenticated'] is True
        and numeric['all_observed_plant_IDs_have_exact_clone_component_ownership'] is True
        and numeric['accepted_training_increment']==0,'Actual native ownership proof required')
    pairs={r['comparison']:r for r in numeric['comparisons']}
    for name in ('six_vs_eight_A_before','six_vs_eight_B','six_vs_eight_A_return'):
        p=pairs[name]
        require(p['calibration_exact'] is True and p['decoded_component_pixels_exact'] is True
            and p['depth_validity_agreement']==1 and p['optical_depth_abs_delta_m']['maximum']==0,
            'Finite paired calibration/labels/depth must match')
    require(trace['comparator_result']==evidence['numeric'] and trace['frames_evaluated']==6
        and trace['source_assets_unchanged_at_close'] is True and trace['accepted_training_increment']==0,
        'Actual six-frame unchanged trace required')
    for p in trace['comparisons']:
        require(all(p[k] is True for k in ('all_catalogue_status_reasons_equal','strict_ids_equal',
            'reachable_visible_ids_equal','unknown_ids_equal')),'Reset6/8 eligibility changed')
    require(visual['numeric_comparator']==evidence['numeric'] and visual['trace_result']==evidence['trace']
        and visual['actual_visual_reviews']==6 and visual['actual_native_crops_reviewed']==12
        and visual['human_review_claimed'] is False and visual['production_profile_approved'] is False
        and visual['training_export_eligible'] is False and visual['accepted_training_increment']==0,
        'Actual finite assistant review with no inherited production approval required')
    require(len(visual['records'])==6 and all(r['actual_full_native_rgb_review'] is True
        and r['actual_native_crops_reviewed']==2 for r in visual['records']),'Every diagnostic image must have actual visual review')
    display=read(visual['display_evidence'])
    pins=merge(numeric['source_bindings'],trace['frozen_predicate_sources'],
        {str(Path(v['path']).resolve()):v['sha256'] for v in evidence.values()},
        {str(Path(visual['display_evidence']['path']).resolve()):visual['display_evidence']['sha256']})
    for record in trace['records']:
        for k in ('trace','rgb','saved_reference'):
            read_value=record[k]
            require(sha256(read_value['path'])==read_value['sha256'],'Changed actual diagnostic frame/trace')
            pins=merge(pins,{read_value['path']:read_value['sha256']})
    verify_bindings(pins)
    return pins

def profile_value(reference,evidence):
    require(reference['sha256']==REFERENCE_PROFILE_SHA,'Exact original rendering reference required')
    original=read(reference);baseline.check_profile(original)
    pins=merge(original['source_bindings'],qualification(evidence),implementation_bindings(),
               {reference['path']:reference['sha256']})
    return dict(schema=PROFILE_SCHEMA,profile='full144_reset6_limited36_same_original_renderer.v1',
        request_subframes=6,warmup_steps=[8]*7,reset_api=original['reset_api'],
        render_settings=deepcopy(original['render_settings']),delta_time_seconds=0,wait_for_render=True,
        native_reference_time_policy=original['native_reference_time_policy'],
        scheduler_clock_policy=PAUSED_CLOCK_POLICY,reference_reset8_profile=reference,
        finite_reset6_qualification=evidence,qualified_for_limited_capture_rollout=True,
        broad_cross_scene_temporal_qualification=False,maximum_rollout_frames=36,
        permitted_source_plan_sha256=SOURCE_PLAN_SHA,all_frames_require_unchanged_CPU_and_visual_admission=True,
        renderer_settings_changed=False,render_budget_changed=True,scheduler_clock_changed=False,
        native_resolution=[848,408],training_approved=False,accepted_training_increment=0,
        source_bindings=pins)

def check_profile(profile):
    require(profile.get('schema')==PROFILE_SCHEMA,'Truthful typed production6 profile required')
    expected=profile_value(profile['reference_reset8_profile'],profile['finite_reset6_qualification'])
    require(profile==expected,'Production6 profile differs from finite reviewed rollout')
    return True

def plan_value(original,source_plan,source_cpu,profile):
    value=deepcopy(original)
    value.update(schema=SCHEMA,source_geometry_plan=source_plan,source_geometry_preflight=source_cpu,
        source_pose_profile_evidence=original['profile_evidence'],profile_evidence=profile,
        production_render_budget_changed_only=True,geometry_proof_reused_not_recomputed=True,
        adapter_adds_new_planned_cameras=False,source_pose_geometry_acceptance_inherited_only=True,
        final_native_visibility_and_visual_acceptance_inherited=False,
        implementation_bindings=implementation_bindings())
    return value

def check(plan):
    require(plan.get('schema')==SCHEMA,'Typed production6 plan required')
    sp,sc=plan['source_geometry_plan'],plan['source_geometry_preflight']
    require(sp['sha256']==SOURCE_PLAN_SHA and sc['sha256']==SOURCE_CPU_SHA,
        'Limited first rollout requires exact new TRAIN67 36-pose geometry')
    original,cpu=read(sp),read(sc);data=baseline.check(original)
    require(original['max_frames']==36 and original['capture_split']=='train'
        and len(original['schedule'])==len(original['selected_sample_ids'])==36,'Exact first36 source poses required')
    require(cpu['schema']=='greenhouse.native848_fully_labeled_cpu_preflight.v3'
        and cpu['plan_sha256']==sp['sha256'] and Path(cpu['plan_path']).resolve()==Path(sp['path']).resolve()
        and cpu['scene_policy']==original['scene_policy']
        and cpu['selected_sample_ids']==original['selected_sample_ids']
        and cpu['pose_anchor_sources']==data['pose_anchor_sources']
        and cpu['all_selected_pose_screens_passed'] is True
        and cpu['actual_native_census_comparison_pending'] is True,
        'Exact completed full144 source geometry proof required')
    profile=read(plan['profile_evidence']);check_profile(profile)
    require(plan==plan_value(original,sp,sc,plan['profile_evidence']),'Source poses, schedule, geometry or limits changed')
    require(plan['source_pose_profile_evidence']==profile['reference_reset8_profile'],
        'Source reset8 provenance differs from renderer reference')
    pins=merge(data['source_bindings'],cpu['source_bindings'],profile['source_bindings'],
        implementation_bindings(),{sp['path']:sp['sha256'],sc['path']:sc['sha256'],
        plan['profile_evidence']['path']:plan['profile_evidence']['sha256']})
    for key in ('full_scene_census','active_catalogue'):
        read(cpu[key]);pins=merge(pins,{cpu[key]['path']:cpu[key]['sha256']})
    verify_bindings(pins)
    return dict(data,profile=profile,source_bindings=pins,
        source_geometry_plan=sp,source_geometry_preflight=sc)

def cpu_value(plan,plan_path,plan_sha256):
    original=read(plan['source_geometry_preflight'])
    value=deepcopy(original)
    value.update(schema=CPU_SCHEMA,plan_path=str(Path(plan_path).resolve()),plan_sha256=plan_sha256,
        source_geometry_plan=plan['source_geometry_plan'],source_geometry_preflight=plan['source_geometry_preflight'],
        geometry_proof_reused_not_recomputed=True,render_profile_qualification_separate=True,
        production_profile=plan['profile_evidence'],new_native_quality_acceptance_claimed=False,
        source_bindings=merge(original['source_bindings'],implementation_bindings(),
            {str(Path(plan_path).resolve()):plan_sha256,
             plan['source_geometry_plan']['path']:plan['source_geometry_plan']['sha256'],
             plan['source_geometry_preflight']['path']:plan['source_geometry_preflight']['sha256'],
             plan['profile_evidence']['path']:plan['profile_evidence']['sha256']}))
    return value

def validate_cpu(cpu,plan,plan_path,plan_sha256):
    require(sha256(plan_path)==plan_sha256,'Actual adapted plan changed')
    require(cpu==cpu_value(plan,plan_path,plan_sha256),'Reused exact geometry binding differs')
    verify_bindings(cpu['source_bindings'])
    return True

def prepare(source_plan,source_cpu,reference_profile,evidence,output):
    output=Path(output).resolve()
    require(output.is_relative_to(ROOT/'data/sim_data/diagnostics') and not output.exists(),
        'Create-only diagnostic preparation directory required')
    profile=profile_value(reference_profile,evidence)
    original=read(source_plan)
    output.mkdir(parents=True)
    profile_path=output/'profile.json';save_json(profile_path,profile)
    plan=plan_value(original,source_plan,source_cpu,pin(profile_path));check(plan)
    plan_path=output/'plan.json';save_json(plan_path,plan)
    cpu=cpu_value(plan,plan_path,sha256(plan_path));validate_cpu(cpu,plan,plan_path,sha256(plan_path))
    cpu_path=output/'cpu_preflight.json';save_json(cpu_path,cpu)
    result=dict(schema='greenhouse.native848_reset6_limited_prepare.v1',profile=pin(profile_path),
        plan=pin(plan_path),cpu_preflight=pin(cpu_path),source_plan=source_plan,source_cpu=source_cpu,
        frames=36,geometry_recomputed=False,native_launched=False,accepted_increment=0)
    save_json(output/'result.json',result)
    return result
