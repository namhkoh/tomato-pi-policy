from copy import deepcopy

import pytest

from sim_data.active_perception_contract import (validate_answer, validate_evidence, user_prompt,
    motion_capture_gaps, contract_hash, TASK_ID)
from sim_data.active_perception_pilot import proposed_answer
from sim_data.training_contract import contract_hash as v3_hash


def visible():
    return proposed_answer('visible', dict(visibility='clear', cut_point_uv=[100., 200.]))


def test_existing_v3_hash_and_new_identity():
    assert v3_hash() == 'd96e97063f3abf3cca7af4e0cc5f52def259e00c699016c5d8e5f27975dd1c15'
    assert contract_hash() != v3_hash() and TASK_ID.endswith('.v4')


def test_visible_and_occluded_are_different_observability_not_absence():
    validate_evidence(visible(), dict(target_identifiable=True, cut_region_identifiable=True, eligibility_observable=True))
    hidden = proposed_answer('occluded')
    assert hidden['target_state'] == 'unknown' and hidden['cut_point_uv'] is None
    validate_evidence(hidden, {})


@pytest.mark.parametrize('change', [dict(cut_visibility='occluded'), dict(target_state='unknown'),
    dict(execution_feasibility='approved'), dict(cut_point_uv=[True, 20]), dict(cut_point_uv=[848, 20]),
    dict(cut_point_uv=[float('nan'), 20]), dict(next_action='cut_now'), dict(robot_xyz=[0,0,0])])
def test_invalid_or_executable_answers_rejected(change):
    answer = visible(); answer.update(change)
    with pytest.raises(ValueError): validate_answer(answer)


def test_world_truth_alone_cannot_justify_localization():
    with pytest.raises(ValueError, match='Hidden geometry'):
        validate_evidence(visible(), dict(world_target_exists=True, world_xyz=[1,2,3]))


def test_invalid_candidate_needs_known_organ_not_unmapped_background():
    answer = proposed_answer('invalid_candidate')
    validate_evidence(answer, dict(identified_ineligible_organ='main_stem'))
    with pytest.raises(ValueError): validate_evidence(answer, dict(identified_ineligible_organ='unmapped'))


def test_no_target_requires_bounded_fully_observed_region():
    answer = dict(visible(), target_state='none_in_region', decision='no_target', cut_visibility='not_applicable',
                  cut_point_uv=None, next_action='stop_no_target')
    evidence = dict(assessed_region_fully_observed=True, eligible_target_count_in_region=0,
                    assessed_region_xyxy=[10,20,300,350])
    validate_evidence(answer, evidence, mode='scene_region')
    with pytest.raises(ValueError): validate_evidence(answer, evidence, mode='candidate_query')
    for change in (dict(assessed_region_fully_observed=False), dict(eligible_target_count_in_region=1),
                   dict(eligible_target_count_in_region=False), dict(assessed_region_xyxy=None)):
        with pytest.raises(ValueError): validate_evidence(answer, {**evidence, **change}, mode='scene_region')


def test_reveal_requires_identified_occluder_and_does_not_authorize_motion():
    answer = dict(proposed_answer('occluded'), decision='reveal', next_action='request_occluder_reveal')
    with pytest.raises(ValueError): validate_evidence(answer, {})
    validate_evidence(answer, dict(occluder_identifiable=True))
    assert answer['cut_point_uv'] is None and answer['execution_feasibility'] == 'unverified'


def test_reposition_needs_actual_constraint_not_anatomical_rejection():
    answer = dict(visible(), decision='reposition', cut_point_uv=None, next_action='reposition_robot', execution_feasibility='blocked')
    with pytest.raises(ValueError): validate_evidence(answer, dict(eligibility_observable=True))
    validate_evidence(answer, dict(eligibility_observable=True, feasibility_check='collision_blocked'))


def test_query_prompt_does_not_presuppose_valid_petiole():
    assert 'Is this an eligible' in user_prompt(query=[200,100])
    assert 'absence from insufficient evidence' in user_prompt(region=[0,0,848,408])
    for kwargs in ({}, dict(query=[0,0], region=[0,0,10,10]), dict(region=[0,0,float('inf'),1])):
        with pytest.raises(ValueError): user_prompt(**kwargs)


def test_static_frames_cannot_become_dynamic_demonstrations():
    meta = dict(synchronization=dict(engine_frame_id_verified=False, dynamic_recording_supported=False),
                robot_snapshot={}, supervision={})
    result = motion_capture_gaps(meta)
    assert len(result['missing_prerequisites']) == 5
    assert result['action_training_eligible'] is False
