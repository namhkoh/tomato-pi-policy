"""Task-v4 prototype: observable evidence is not privileged simulator truth.

This is a new perception/planning-intent contract, not a motion controller or
training release. Task-v3 examples and their contract hash remain unchanged.
"""
import math

from .capture_contract import fingerprint
from .dataset_review import require

TASK_ID = 'greenhouse.active_perception.rgb.v4'
CONTRACT = dict(task_id=TASK_ID, state='prototype_requires_visual_review',
    input='original_robot_camera_RGB_instruction_optional_candidate_pixel_and_observation_history',
    coordinates='original_848x408_continuous_pixels_top_left_origin_u_right_v_down',
    depth='native_Isaac_distance_to_image_plane_optical_Z_metres_sidecar',
    nominal_cut='10mm_along_petiole_from_attachment', evaluation_interval='10_to_20mm_arc_not_sphere',
    hidden_geometry='private_evaluator_only_never_executable_cut_or_model_input',
    target_modes=['candidate_query', 'scene_region'],
    unsupported=['physical_motion_commands', 'validated_reveal_trajectories', 'dynamic_capture',
                 'whole_plant_absence_from_single_occluded_image', 'agronomic_approval'])

SYSTEM_PROMPT = (
    'Inspect the original 848x408 robot-camera RGB and the task instruction. '
    'A candidate query may indicate a petiole OR an invalid object; it is not a cut point. '
    'Distinguish an eligible petiole from main stems, fruit and fruit-bearing structures. '
    'Only localize a cut when its petiole and attachment are visually identifiable. '
    'The nominal cut convention is 10 mm along the petiole away from its attachment; '
    'the evaluation interval is 10-20 mm along that petiole, not a sphere. '
    'A hidden region is not proof that no target exists. Do not infer hidden XYZ from foreground depth. '
    'Use unknown when evidence does not establish target eligibility. '
    'No-target means only the explicitly assessed, fully observed region. '
    'Return JSON with target_state (eligible, ineligible, unknown, none_in_region), '
    'cut_visibility (clear, partial, occluded, unknown, not_applicable), '
    'decision (localize, inspect, reject, reposition, reveal, no_target), '
    'cut_point_uv (two original-image pixel coordinates or null), '
    'next_action (inspect_cut_region, change_viewpoint, skip_candidate, reposition_robot, '
    'request_occluder_reveal, stop_no_target), execution_feasibility (unverified or blocked), '
    'and evidence_note (brief observable justification). '
    'Reveal is a request for a checked low-level skill, never permission to move a knife. '
    'Do not return robot motion commands or claim that cutting is safe.'
)

ACTIONS = dict(localize='inspect_cut_region', inspect='change_viewpoint', reject='skip_candidate',
               reposition='reposition_robot', reveal='request_occluder_reveal', no_target='stop_no_target')


def contract_hash():
    return fingerprint(dict(contract=CONTRACT, system_prompt=SYSTEM_PROMPT, actions=ACTIONS))


def pixel(value):
    require(isinstance(value, list) and len(value) == 2 and
            all(type(x) in (int, float) and math.isfinite(x) for x in value)
            and 0 <= value[0] < 848 and 0 <= value[1] < 408, 'Invalid original-image pixel')
    return value


def validate_answer(answer):
    require(isinstance(answer, dict) and set(answer) == {'target_state', 'cut_visibility', 'decision',
        'cut_point_uv', 'next_action', 'execution_feasibility', 'evidence_note'}, 'Invalid task-v4 answer fields')
    state, visibility, decision = (answer[k] for k in ('target_state', 'cut_visibility', 'decision'))
    require(isinstance(decision, str) and decision in ACTIONS
            and answer['next_action'] == ACTIONS[decision], 'Invalid decision/action combination')
    require(state in ('eligible', 'ineligible', 'unknown', 'none_in_region')
            and visibility in ('clear', 'partial', 'occluded', 'unknown', 'not_applicable'), 'Invalid target or visibility state')
    require(answer['execution_feasibility'] in ('unverified', 'blocked'), 'Perception cannot approve execution')
    require(isinstance(answer['evidence_note'], str) and 10 <= len(answer['evidence_note'].strip()) <= 1000,
            'Brief observable justification required')
    if decision == 'localize':
        require(state == 'eligible' and visibility in ('clear', 'partial')
                and answer['execution_feasibility'] == 'unverified', 'Localization needs identifiable target, not execution approval')
        pixel(answer['cut_point_uv'])
    else:
        require(answer['cut_point_uv'] is None, 'Unverified or rejected cuts must be null')
    if decision == 'reject':
        require(state == 'ineligible' and visibility == 'not_applicable', 'Reject is for an identified invalid candidate')
    elif decision == 'no_target':
        require(state == 'none_in_region' and visibility == 'not_applicable', 'No-target is scoped absence, not occlusion')
    elif decision in ('inspect', 'reveal'):
        require(state in ('eligible', 'unknown') and visibility in ('partial', 'occluded', 'unknown'), 'Inspection needs missing evidence')
    elif decision == 'reposition':
        require(state in ('eligible', 'unknown') and visibility != 'not_applicable'
                and answer['execution_feasibility'] == 'blocked', 'Reposition needs explicit feasibility evidence')
    return answer


def validate_evidence(answer, evidence, *, mode='candidate_query'):
    """Validate proposed supervision; not a replacement for visual/physics audit."""
    validate_answer(answer)
    require(mode in CONTRACT['target_modes'], 'Unknown task mode')
    require(isinstance(evidence, dict), 'Observation evidence required')
    decision = answer['decision']
    if decision == 'localize':
        require(evidence.get('target_identifiable') is True and evidence.get('cut_region_identifiable') is True,
                'Hidden geometry cannot justify localization')
    if answer['target_state'] == 'eligible':
        require(evidence.get('eligibility_observable') is True, 'Privileged eligibility is not observable evidence')
    if decision == 'reject':
        require(evidence.get('identified_ineligible_organ') in ('main_stem', 'fruit', 'peduncle', 'flower', 'already_cut_stub'),
                'Invalid candidate needs an identified organ, not an unmapped pixel')
    if decision == 'no_target':
        require(mode == 'scene_region' and evidence.get('assessed_region_fully_observed') is True
                and evidence.get('eligible_target_count_in_region') == 0
                and type(evidence.get('eligible_target_count_in_region')) is int,
                'Occluded or unsearched scene cannot prove absence')
        box = evidence.get('assessed_region_xyxy')
        require(isinstance(box, list) and len(box) == 4 and all(type(v) in (float, int) and math.isfinite(v) for v in box)
                and 0 <= box[0] < box[2] <= 848 and 0 <= box[1] < box[3] <= 408, 'Explicit assessed region required')
    if decision == 'reveal':
        require(evidence.get('occluder_identifiable') is True, 'Do not request manipulation of an unidentified occluder')
    if answer['execution_feasibility'] == 'blocked':
        require(evidence.get('feasibility_check') in ('unreachable', 'collision_blocked', 'joint_limit'), 'Missing feasibility check')
    return answer


def user_prompt(query=None, region=None):
    require((query is None) != (region is None), 'Supply a candidate query OR a bounded scene region')
    if query is not None:
        pixel(query)
        return (f'Inspect candidate pixel ({query[0]:.1f}, {query[1]:.1f}). Is this an eligible deleafing petiole? '
                'Localize its cut only if supported by the image; otherwise inspect or reject as appropriate.')
    require(isinstance(region, list) and len(region) == 4 and all(type(v) in (int, float) and math.isfinite(v) for v in region)
            and 0 <= region[0] < region[2] <= 848 and 0 <= region[1] < region[3] <= 408, 'Invalid assessed region')
    return f'Inspect region {region} for eligible deleafing targets. Distinguish absence from insufficient evidence.'


def motion_capture_gaps(metadata):
    """Report readiness gaps. Static records cannot be promoted to action demos."""
    sync, robot = metadata.get('synchronization', {}), metadata.get('robot_snapshot', {})
    gaps = []
    for key in ('engine_frame_id_verified', 'dynamic_recording_supported'):
        if sync.get(key) is not True: gaps.append(key)
    for key in ('whole_robot_collision_checked', 'motion_between_snapshots_validated'):
        if robot.get(key) is not True: gaps.append(key)
    if metadata.get('supervision', {}).get('physics_validated') is not True:
        gaps.append('physical_interaction_validation')
    # This helper reports prerequisites only; no dynamic recorder is implemented here.
    return dict(missing_prerequisites=gaps, dynamic_recorder_implemented_here=False,
                physical_execution_approved=False, action_training_eligible=False)
