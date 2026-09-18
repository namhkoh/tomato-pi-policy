"""CPU tamper fixtures; these are not native evidence or dataset reviews."""
import copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from . import native848_conservative_join_v3 as consumer


def route_fixture():
    trial = (Path.cwd()/'fixture_reference_trial').resolve()
    intent = dict(plan_path=str(trial.parent/'fixture_plan.json'), plan_sha256='p'*64)
    receipt = dict(state=consumer.REFERENCE_RECEIPT, training_approved=False, plan_sha256=intent['plan_sha256'])
    owned = dict(returncode=0, method='subprocess_wait_on_owned_process', pid=123)
    launch = dict(pid=123, plan_sha256=intent['plan_sha256'], command=['python', '-m', consumer.REFERENCE_WORKER,
        '--plan', intent['plan_path'], '--plan-sha256', intent['plan_sha256'], '--output', str(trial/'capture')])
    plan = dict(schema=consumer.REFERENCE_PLAN, generated_geometry_used=False, source_cap_reset=False, training_approved=False)
    return trial, receipt, owned, launch, intent, plan


def test_explicit_reference_producer_route():
    consumer.check_owned_route(*route_fixture())


@pytest.mark.parametrize('mutation', ['worker', 'plan_path', 'plan_hash', 'plan_schema', 'duplicate_plan_flag', 'failed_exit'])
def test_mismatched_producer_or_plan_is_rejected(mutation):
    values = route_fixture(); trial, receipt, owned, launch, intent, plan = values
    if mutation == 'worker': launch['command'][2] = 'sim_data.native848_direct_worker_v2'
    elif mutation == 'plan_path': launch['command'][4] = str(trial/'other_plan.json')
    elif mutation == 'plan_hash': launch['command'][6] = 'q'*64
    elif mutation == 'plan_schema': plan['schema'] = 'greenhouse.original848_direct_camera_batch_plan.v1'
    elif mutation == 'duplicate_plan_flag': launch['command'][3:3] = ['--plan', intent['plan_path']]
    else: owned['returncode'] = 1
    with pytest.raises(ValueError): consumer.check_owned_route(*values)


def identity_fixture():
    anchor = dict(schema_version=consumer.REFERENCE_ANCHOR, split='train', source_family='seed17_full',
        generated_geometry_used=False, source_cap_reset=False,
        source_row=dict(target_id='seed17_full/SubStem_42', variant_id='seed17_full'),
        conservative_view_cap_group='seed17_full/SubStem_42')
    source = dict(target_id='seed17_full/SubStem_46', variant_id='seed17_full')
    target = source['target_id']; sample = str(Path.cwd()/'fixture_sample.json')
    rec = dict(sample_id='original_046', split='train', source_family='seed17_full', target_id=target,
        source_target=target, source_sample_path=sample)
    meta = dict(schema_version=consumer.REFERENCE_SAMPLE, sample_id=rec['sample_id'],
        original_geometry_only=True, generated_plant_native_pixels=0,
        supervision=dict(target_id=target, source_target_id=target, conservative_view_cap_group=target, split_group='seed17_full'))
    cache = dict(anchor_reference_sha256='a'*64, anchor_reference_path=str(Path.cwd()/'fixture_anchor.json'))
    audit = dict(sample_id=rec['sample_id'], target_id=target, source_target=target, sample_path=sample, **cache)
    return anchor, source, rec, meta, audit, cache


def test_captured_original_stem_owns_pool_not_reference_stem():
    values = identity_fixture(); before = copy.deepcopy(values)
    context = consumer.original_reference_identity_context(*values)
    assert context['conservative_view_cap_group'] == 'seed17_full/SubStem_46'
    assert context['source_row'] == values[1] and values == before


@pytest.mark.parametrize('mutation', ['generated_variant', 'generated_sample_schema', 'generated_pixels', 'source_pool',
                                    'metadata_pool', 'anchor_hash', 'sample_path', 'heldout'])
def test_relabelled_generated_or_remapped_source_is_rejected(mutation):
    values = identity_fixture(); anchor, source, rec, meta, audit, cache = values
    if mutation == 'generated_variant': source['variant_id'] = 'seed17_full_cr_fixture'
    elif mutation == 'generated_sample_schema': meta['schema_version'] = 'greenhouse.native848_pair_sample.v2'
    elif mutation == 'generated_pixels': meta['generated_plant_native_pixels'] = 1
    elif mutation == 'source_pool': rec['source_target'] = anchor['conservative_view_cap_group']
    elif mutation == 'metadata_pool': meta['supervision']['conservative_view_cap_group'] = anchor['conservative_view_cap_group']
    elif mutation == 'anchor_hash': audit['anchor_reference_sha256'] = 'b'*64
    elif mutation == 'sample_path': audit['sample_path'] = str(Path.cwd()/'other_sample.json')
    else: rec['split'] = 'validation'
    with pytest.raises(ValueError): consumer.original_reference_identity_context(*values)


def workspace_fixture():
    pose = dict(probe=np.array([0., 0., 0.]), base=np.eye(4), torso=[0]*6, right=[0]*7)
    model = SimpleNamespace(forward=lambda *a: np.eye(4), arm_joint_limit_margin_degrees=lambda *a: 2.,
        inter_arm_clearance=lambda *a: SimpleNamespace(clearance_m=.1))
    checker = SimpleNamespace(bindings={'fixture_robot':'h'}, model=model, pose_and_probe=lambda meta: pose)
    meta = dict(supervision=dict(target_id='seed17_full/SubStem_46', nominal_world_m=[0., 0., 0.]))
    proof = dict(source_bindings=checker.bindings, target_id=meta['supervision']['target_id'], nominal_world_m=[0., 0., 0.],
        base_fixed=True, torso_fixed=True, orientation_constrained=False, full_scene_arm_collision_checked=False,
        approach_path_checked=False, physical_cut_approved=False, probe_ee_m=[0., 0., 0.],
        result=dict(workspace_passed=True, status='position_ik_with_joint_limits_and_inter_arm_screen',
            candidate=dict(joint_degrees=[0.]*7, probe_world_m=[0., 0., 0.], independent_fk_position_error_m=0.,
                           joint_limit_margin_degrees=2., inter_arm_clearance_m=.1)))
    return proof, meta, checker


def test_saved_workspace_certificate_replays_without_optimizer():
    result = consumer.replay_saved_workspace(*workspace_fixture())
    assert result['saved_passing_solution_replayed'] and result['optimizer_rerun'] is False


@pytest.mark.parametrize('mutation', ['fk_error', 'joint_limit', 'arm_overlap', 'certificate_value', 'probe', 'bindings'])
def test_workspace_acceptance_cannot_survive_invalid_saved_solution(mutation):
    proof, meta, checker = workspace_fixture()
    if mutation == 'fk_error':
        moved = np.eye(4); moved[0,3] = .002; checker.model.forward = lambda *a: moved
    elif mutation == 'joint_limit': checker.model.arm_joint_limit_margin_degrees = lambda *a: -1.
    elif mutation == 'arm_overlap': checker.model.inter_arm_clearance = lambda *a: SimpleNamespace(clearance_m=-.1)
    elif mutation == 'certificate_value': proof['result']['candidate']['probe_world_m'] = [.1, 0., 0.]
    elif mutation == 'probe': proof['probe_ee_m'] = [.1, 0., 0.]
    else: proof['source_bindings'] = {'fixture_robot':'changed'}
    with pytest.raises(ValueError): consumer.replay_saved_workspace(proof, meta, checker)


def test_failed_workspace_proof_stays_held_without_resolving():
    proof, meta, checker = workspace_fixture()
    proof['result'] = dict(workspace_passed=False, status='search_time_limit_not_proof_of_unreachability', candidate=None)
    checker.model.forward = lambda *a: pytest.fail('A held proof must not be solved or promoted')
    result = consumer.replay_saved_workspace(proof, meta, checker)
    assert result['workspace_passed'] is False and result['held_result_not_reclassified'] is True


def test_old_admissions_use_frozen_v2_replay(monkeypatch, tmp_path):
    result = tmp_path/'result.json'; consumer.save(result, dict(state=consumer.previous.ORIGINAL_STATE))
    spec = dict(result=dict(path=str(result), sha256=consumer.digest(result)))
    sentinel = (['old-candidate'], ['old-hold'], ['old-exclusion'])
    monkeypatch.setattr(consumer.previous, 'replay_admission', lambda passed, pins: sentinel if passed is spec else None)
    assert consumer.replay_admission(spec, consumer.Pins()) is sentinel
