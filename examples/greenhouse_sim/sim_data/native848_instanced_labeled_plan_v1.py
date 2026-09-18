"""Finite two-view equivalence replay, excluded from training admission."""
from pathlib import Path
from copy import deepcopy
from . import native848_fully_labeled_plan_v1 as baseline_api
from . import native848_instanced_labeled_scene_v1 as scene_api
from . import native848_instanced_static_guard_v1 as guard
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256
from .native848_bulk_io_v1 import save_json

SCHEMA='greenhouse.native848_instanced_labeled_plan.v1'
CONTEXT_SCHEMA='greenhouse.native848_instanced_labeled_context.v1'
OBSERVATION_SCHEMA='greenhouse.native848_instanced_labeled_observation.v1'
RESULT_STATE='instancing_equivalence_captured_pending_ID_depth_quality_comparison_no_training_admission'
ORIGINAL_PROFILE_SCHEMA=baseline_api.ORIGINAL_PROFILE_SCHEMA
PAUSED_CLOCK_POLICY=baseline_api.PAUSED_CLOCK_POLICY
NATIVE_TIME_POLICY=baseline_api.NATIVE_TIME_POLICY
SCHEDULER_CLOCK_POLICY=baseline_api.SCHEDULER_CLOCK_POLICY
check_profile=baseline_api.check_profile
geometry_pose_key=baseline_api.geometry_pose_key
ROLE='instancing_equivalence_diagnostic'
ROOT=Path(__file__).resolve().parents[3]

def metadata(split):
    return dict(**scene_api.metadata(split),native_population_census_verified=False,
        diagnostic_equivalence_replay=True,training_export_eligible=False)

def implementation_bindings():
    return {str(Path(p).resolve()):sha256(p) for p in (__file__,baseline_api.__file__,scene_api.__file__,guard.__file__)}

def pinned(spec):
    path=Path(spec['path']).resolve();require(sha256(path)==spec['sha256'],'Paired source changed');return read_json(path)

def prepare(baseline_plan,reference_capture,output):
    output=Path(output).resolve();require(output.is_relative_to(ROOT/'data/sim_data/diagnostics') and not output.exists(),'Create-only paired plan required')
    original=pinned(baseline_plan);data=baseline_api.check(original);require(original['max_frames']==2,'Exact two baseline poses required')
    plan={key:deepcopy(original[key]) for key in ['profile_evidence','cache_path','cache_sha256','capture_split','resolution',
        'instance_backend','queue_frames','queue_bytes','writer_threads','geometry_policy','automatic_retries','source_acceptance_inherited','scene_preflight_required']}
    plan.update(schema=SCHEMA,baseline_plan=baseline_plan,paired_reference_capture=reference_capture,
        **metadata(data['capture_split']),scene_policy=scene_api.policy(data['capture_split']),max_frames=2,
        selected_sample_ids=original['selected_sample_ids'],schedule=[dict(observation_id='instanced_pair_'+row['sample_id'],source_pose_id=row['sample_id'],capture_role=ROLE) for row in data['records']],
        annotation_contract='instancing_equivalence_only_no_training_admission',implementation_bindings=implementation_bindings())
    check(plan);output.mkdir(parents=True);path=output/'plan.json';save_json(path,plan);return dict(path=str(path),sha256=sha256(path))

def check(plan):
    require(plan['schema']==SCHEMA and plan['implementation_bindings']==implementation_bindings(),'Exact typed instanced replay implementation required')
    original=pinned(plan['baseline_plan']);data=baseline_api.check(original);split=data['capture_split']
    require(original['max_frames']==plan['max_frames']==2 and plan['selected_sample_ids']==original['selected_sample_ids'],'Exact paired source views required')
    for key in ['profile_evidence','cache_path','cache_sha256','capture_split','resolution','instance_backend','queue_frames',
        'queue_bytes','writer_threads','geometry_policy','automatic_retries','source_acceptance_inherited','scene_preflight_required']:
        require(plan[key]==original[key],'Unchanged paired capture configuration required: '+key)
    require(all(plan.get(k)==v for k,v in metadata(split).items()) and plan['scene_policy']==scene_api.policy(split)
        and plan['annotation_contract']=='instancing_equivalence_only_no_training_admission','Explicit diagnostic-only instanced scene required')
    require(plan['schedule']==[dict(observation_id='instanced_pair_'+r['sample_id'],source_pose_id=r['sample_id'],capture_role=ROLE) for r in data['records']],'Exact replay schedule required')
    reference=pinned(plan['paired_reference_capture'])
    require(reference['schema']=='greenhouse.native848_fully_labeled_owner.v1' and reference['plan_sha256']==plan['baseline_plan']['sha256']
        and reference['committed_frames']==reference['planned_frames']==2,'Actual paired ordinary capture required')
    require(sha256(reference['manifest_path'])==reference['manifest_sha256'] and sha256(reference['context_path'])==reference['context_sha256'],'Paired capture assets changed')
    pins={**data['source_bindings'],**implementation_bindings(),**reference['source_bindings'],
        str(Path(plan['baseline_plan']['path']).resolve()):plan['baseline_plan']['sha256'],
        str(Path(plan['paired_reference_capture']['path']).resolve()):plan['paired_reference_capture']['sha256'],
        str(Path(reference['manifest_path']).resolve()):reference['manifest_sha256'],str(Path(reference['context_path']).resolve()):reference['context_sha256']}
    verify_bindings(pins)
    return dict(data,source_bindings=pins,scene_policy=plan['scene_policy'],pose_anchor_sources=[dict(baseline_plan=plan['baseline_plan'])])
