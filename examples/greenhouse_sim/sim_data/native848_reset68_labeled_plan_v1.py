"""Six diagnostic-only ordinary-full144 requests: reset6 A-B-A, reset8 A-B-A."""
from pathlib import Path
from copy import deepcopy
from . import native848_fully_labeled_plan_v1 as baseline_api
from . import native848_fully_labeled_scene_v1 as scene_api
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256
from .native848_bulk_io_v1 import save_json

SCHEMA='greenhouse.native848_reset68_labeled_plan.v1'
CONTEXT_SCHEMA='greenhouse.native848_reset68_labeled_context.v1'
OBSERVATION_SCHEMA='greenhouse.native848_reset68_labeled_observation.v1'
RESULT_STATE='reset6_8_fullpopulation_ABA_captured_diagnostic_only_no_training_admission'
PROFILE='original848_reset6_8_fullpopulation_ABA_diagnostic.v1'
ROLE='reset6_8_ABA_diagnostic'
ANNOTATION_CONTRACT='reset6_8_ABA_diagnostic_only_no_training_admission'
ORIGINAL_PROFILE_SCHEMA=baseline_api.ORIGINAL_PROFILE_SCHEMA
PAUSED_CLOCK_POLICY=baseline_api.PAUSED_CLOCK_POLICY
NATIVE_TIME_POLICY=baseline_api.NATIVE_TIME_POLICY
SCHEDULER_CLOCK_POLICY=baseline_api.SCHEDULER_CLOCK_POLICY
check_profile=baseline_api.check_profile
geometry_pose_key=baseline_api.geometry_pose_key
ROOT=Path(__file__).resolve().parents[3]
BASELINE_SHA='9dc751913a157b0362f0d84b64f9e0b9d17841771ed45f1dd7ac4507e4a0b077'
BASELINE_CPU_SHA='88e17ba157e6217d96237eaa937d5a9945127c623f56be431580c20d82dde772'

def metadata(split):
    return dict(**scene_api.metadata(split),native_population_census_verified=False,
        render_budget_diagnostic_replay=True,training_export_eligible=False)

def implementation_bindings():
    return {str(Path(p).resolve()):sha256(p) for p in (__file__,baseline_api.__file__,scene_api.__file__)}

def pinned(spec):
    path=Path(spec['path']).resolve();require(sha256(path)==spec['sha256'],'Exact reset68 source changed');return read_json(path)

def schedule(ids):
    require(len(ids)==2 and len(set(ids))==2,'Exactly two authenticated source cameras required')
    # Source2 is the actually strong ordinary smoke frame. A is that pose;
    # source1 B retains its real competing petiole and is never accepted here.
    return [dict(observation_id=f'reset{budget}_{phase}_{ids[index]}',source_pose_id=ids[index],
        capture_role=ROLE,requested_subframes=budget,phase=phase)
        for budget in (6,8) for phase,index in [('A_before',1),('B',0),('A_return',1)]]

def prepare(baseline_plan,baseline_cpu,output):
    output=Path(output).resolve();require(output.is_relative_to(ROOT/'data/sim_data/diagnostics') and not output.exists(),'Create-only diagnostic plan required')
    original=pinned(baseline_plan);data=baseline_api.check(original)
    plan={key:deepcopy(original[key]) for key in ['profile_evidence','cache_path','cache_sha256','capture_split','resolution',
        'instance_backend','queue_frames','queue_bytes','writer_threads','geometry_policy','automatic_retries','source_acceptance_inherited','scene_preflight_required']}
    plan.update(schema=SCHEMA,baseline_plan=baseline_plan,baseline_cpu=baseline_cpu,**metadata(data['capture_split']),
        scene_policy=scene_api.policy(data['capture_split']),max_frames=6,selected_sample_ids=original['selected_sample_ids'],
        schedule=schedule(original['selected_sample_ids']),diagnostic_profile=PROFILE,
        render_budget_policy='only_per_request_budget_6_or_8_differs_from_reference_profile_8',
        annotation_contract=ANNOTATION_CONTRACT,implementation_bindings=implementation_bindings())
    check(plan);output.mkdir(parents=True);path=output/'plan.json';save_json(path,plan);return dict(path=str(path),sha256=sha256(path))

def check(plan):
    require(plan['schema']==SCHEMA and plan['implementation_bindings']==implementation_bindings(),'Exact typed reset68 implementation required')
    require(plan['baseline_plan']['sha256']==BASELINE_SHA and plan['baseline_cpu']['sha256']==BASELINE_CPU_SHA,'Only exact completed smoke67 poses and full CPU geometry permitted')
    original=pinned(plan['baseline_plan']);cpu=pinned(plan['baseline_cpu']);data=baseline_api.check(original);split=data['capture_split']
    require(original['max_frames']==2 and plan['max_frames']==6 and plan['selected_sample_ids']==original['selected_sample_ids']
        and plan['schedule']==schedule(original['selected_sample_ids']),'Exact six-request A-B-A schedules required')
    require(cpu['schema']=='greenhouse.native848_fully_labeled_cpu_preflight.v1'
        and cpu['plan_sha256']==BASELINE_SHA and cpu['scene_policy']==original['scene_policy']
        and cpu['selected_sample_ids']==original['selected_sample_ids'] and cpu['all_selected_pose_screens_passed'] is True
        and cpu['all_selected_9mm_workspace_checks_passed'] is True,'Exact prior full144 two-pose CPU checks required')
    for key in ['profile_evidence','cache_path','cache_sha256','capture_split','resolution','instance_backend','queue_frames',
        'queue_bytes','writer_threads','geometry_policy','automatic_retries','source_acceptance_inherited','scene_preflight_required']:
        require(plan[key]==original[key],'Unchanged reset68 source configuration required: '+key)
    require(data['profile']['request_subframes']==8 and data['profile']['warmup_steps']==[8]*7
        and data['profile']['delta_time_seconds']==0 and plan['diagnostic_profile']==PROFILE
        and plan['render_budget_policy']=='only_per_request_budget_6_or_8_differs_from_reference_profile_8',
        'Unchanged reference8 rendering/clock/warmup and explicit diagnostic overrides required')
    require(all(plan.get(k)==v for k,v in metadata(split).items()) and plan['scene_policy']==scene_api.policy(split)
        and plan['annotation_contract']==ANNOTATION_CONTRACT,'Explicit diagnostic-only ordinary full144 scope required')
    pins={**data['source_bindings'],**implementation_bindings(),**cpu['source_bindings'],
        str(Path(plan['baseline_plan']['path']).resolve()):BASELINE_SHA,str(Path(plan['baseline_cpu']['path']).resolve()):BASELINE_CPU_SHA}
    for key in ('full_scene_census','active_catalogue'):
        pinned(cpu[key]);pins[str(Path(cpu[key]['path']).resolve())]=cpu[key]['sha256']
    verify_bindings(pins);by={r['sample_id']:r for r in data['records']}
    result=dict(data);result.update(records=[deepcopy(by[item['source_pose_id']]) for item in plan['schedule']],
        source_bindings=pins,pose_anchor_sources=[dict(baseline_plan=plan['baseline_plan'])])
    return result
