"""Synthetic unit evidence is NEVER native proof; opt-in real pilot uses no mocks.

Reuse the existing native-buffer test fixture. Only source USD/FK/smoke/plan
reconstruction are its documented stand-ins. Bank joins, hashes, saved native
label/ID/trace replay, review gating and publication checks are real CPU code.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import os
import sys

import pytest

from . import original_reference_bank as rb
from .test_original_inventory import original, reseal
from .test_original_inventory import test_actual_original_nonpassing_annotation_rows_are_not_dropped as make_nonpassing
from .test_inventory import write, pin, json_at
from ..native_original_capture import audit


@pytest.fixture
def reference(original):
    f = original
    plan, root = f['plan'], f['root']
    plan['scene_policy'] = deepcopy(rb.oc.SCENE_POLICY)
    plan['expected_calibration']['camera_path'] = rb.oc.HEAD_CAMERA
    plan['prior_calibration']['camera_path'] = rb.oc.HEAD_CAMERA
    prior = plan['pose_prior']
    prior.update(prior_target_id=plan['target_id'], allow_other_target_pose=False)
    clear = json_at(root/'clear.json')
    clear['configuration'] = {'clear_capture': rb.oc.SCENE_POLICY['profile']}
    plan['source_collection_plan_sha256'] = write(root/'clear.json', clear)
    old = deepcopy(clear)
    old['configuration'] = {'historical_grounding_only': True}
    old_pin = write(root/'old_plan.json', old)
    plan['prior_calibration']['resolution'] = [848, 408]
    # Synthetic prior metadata, not a fabricated real legacy observation.
    old_sample = dict(schema_version='greenhouse.rgbd_pilot_sample.v2', sample_id=prior['sample_id'],
        calibration=deepcopy(plan['prior_calibration']), robot_snapshot=deepcopy(plan['expected_robot_snapshot']),
        supervision=dict(target_id=plan['target_id'], cut_region_proposal=plan['source_row']['cut_region_proposal'],
                         review_id=plan['source_row']['draft_id']))
    sample_path = root/'legacy'/prior['sample_id']/'sample.json'
    prior['source_sample_sha256'] = write(sample_path, old_sample)
    manifest = json_at(root/'legacy/manifest.json')
    manifest.update(state='pilot_ready_for_review', source_assets_unchanged=True, source_geometry_modified=False,
        target_family_split='train', collection_job_id=plan['collection_job_id'],
        source_collection_plan_path=str(root/'old_plan.json'), source_collection_plan_sha256=old_pin,
        samples=[dict(sample_id=prior['sample_id'], target_review_id=plan['source_row']['draft_id'])])
    prior['source_manifest_sha256'] = write(root/'legacy/manifest.json', manifest)
    plan['source_bindings'].update({str(p): pin(p) for p in
        (root/'clear.json', root/'old_plan.json', root/'legacy/manifest.json', sample_path)})
    for case in plan['cases']:
        case['pose_prior'] = deepcopy(prior)
        case['prior_calibration'] = deepcopy(plan['prior_calibration'])
        case['expected_calibration'] = deepcopy(plan['expected_calibration'])
    meta = json_at(f['folder']/'sample.json')
    meta['pose_prior'] = deepcopy(prior)
    meta['calibration']['camera_path'] = rb.oc.HEAD_CAMERA
    meta['synchronization']['freshness']['camera_sha256'] = rb.inv.fingerprint(meta['calibration'])
    f['result']['capture']['records'][0]['sample_sha256'] = write(f['folder']/'sample.json', meta)
    f['expected_plan'].clear(); f['expected_plan'].update(deepcopy(plan))
    f['automatic'] = audit.audit_records(f['capture'], plan, f['result']['capture']['records'])
    reseal(f)
    return f


def review_pin(f, decision=rb.SUPPORTED_REVIEW, name='review.json'):
    paths = [f['folder']/p for p in ('inputs/rgb.png', 'sample.json', 'supervision/label.json')]
    trace = f['folder']/'supervision/query_trace.json'
    if trace.exists():
        paths.append(trace)
    paths += [f['capture']/'result.json', Path(f['receipt']['audit_path']), Path(f['pin'].launcher_receipt_path)]
    document = dict(schema=rb.REVIEW_SCHEMA, reviewer='UNIT ONLY', visual_review_performed=True,
        human_review_performed=False, training_approved=False, training_count_increment=0,
        target_id=f['plan']['target_id'], decision=decision, observations='SYNTHETIC test support, not native evidence',
        source_bindings={str(p): pin(p) for p in paths})
    path = f['root']/name
    return rb.ReviewPin(str(path), write(path, document))


def build(f, reviews=None):
    return rb.build_bank([f['pin']], reviews=[review_pin(f)] if reviews is None else reviews)


def captured(bank):
    return next(e for e in bank['entries'] if e['native_observation'] is not None)


def test_two_chains_scene_authority_and_all_cases(reference):
    rp = review_pin(reference)
    before = {str(p): pin(p) for p in reference['root'].rglob('*') if p.is_file()}
    out = build(reference, [rp]); entry = captured(out)
    assert out['counts'] == dict(planned_cases=2, captured_cases=1, native_decisions={rb.inv.STRICT: 1, 'held_pre_render': 1},
        target_ids=1, reference_ready_cases=1, reference_ready_targets=1)
    assert entry['pose_prior']['collection_plan']['path'] != entry['scene_authority']['clear_plan']['path']
    assert entry['pose_prior']['declaration']['prior_lighting']['dome_intensity'] == 1200
    assert entry['scene_authority']['actual_lighting']['dome_intensity'] == 6000
    assert 'source_and_appearance_bindings' not in entry['scene_authority']
    assert entry['scene_authority']['source_and_appearance_bindings_sha256'] == rb.inv._hash(reference['plan']['source_bindings'])
    assert entry['scene_authority']['source_asset_manifest'] == entry['pose_prior']['manifest']
    assert entry['pose_prior']['calibration']['resolution'] == [848, 408]
    proof = entry['native_observation']
    assert proof['calibration']['resolution'] == [1696, 816] and proof['audit_replayed_by_this_build']
    assert not proof['paired_848_1696_proof'] and not proof['legacy_sensor_prerequisite_satisfied']
    assert not proof['visual_truth_independently_verified']
    assert all(out[k] == v for k, v in rb._FLAGS.items())
    assert before == {str(p): pin(p) for p in reference['root'].rglob('*') if p.is_file()}
    assert 'omni.kit' not in sys.modules and 'isaacsim' not in sys.modules


@pytest.mark.parametrize('kind', ['absent', 'negative', 'conflict'])
def test_missing_negative_or_conflicting_review_keeps_strict_but_withholds_anchor(reference, kind):
    reviews = [] if kind == 'absent' else [review_pin(reference, 'hold_for_review')]
    if kind == 'conflict': reviews.append(review_pin(reference, name='positive.json'))
    out = build(reference, reviews); entry = captured(out)
    assert entry['native_decision'] == rb.inv.STRICT
    assert not entry['reference_status']['usable_as_original_reference']
    assert len(entry['visual_reviews']) == len(reviews)


@pytest.mark.parametrize('decision', [rb.inv.HOLD, rb.inv.EXCLUDE])
def test_hold_exclusion_and_prerender_statuses_survive(reference, decision):
    make_nonpassing(reference, decision)
    out = build(reference)
    assert out['counts']['native_decisions'] == {decision: 1, 'held_pre_render': 1}
    assert not out['counts']['reference_ready_cases']


@pytest.mark.parametrize('which', ['clear.json', 'old_plan.json', 'legacy/manifest.json', 'legacy/old_848/sample.json'])
def test_either_stale_source_chain_fails(reference, which):
    path = reference['root']/which
    value = json_at(path); value['tampered'] = True; write(path, value)
    with pytest.raises(ValueError): build(reference)


@pytest.mark.parametrize('change', ['target', 'missing_trace_pin', 'unsupported_schema', 'no_visual', 'approval', 'wrong_sample'])
def test_visual_review_cannot_rebind_native_proof(reference, change):
    rp = review_pin(reference); value = json_at(Path(rp.path))
    if change == 'target': value['target_id'] = 'seed73_full/Other'
    elif change == 'missing_trace_pin': value['source_bindings'].pop(str(reference['folder']/'supervision/query_trace.json'))
    elif change == 'unsupported_schema': value['schema'] = 'legacy_accept'
    elif change == 'no_visual': value['visual_review_performed'] = False
    elif change == 'approval': value['training_approved'] = True
    else: value['source_bindings'].pop(str(reference['folder']/'sample.json'))
    rp = replace(rp, sha256=write(Path(rp.path), value))
    with pytest.raises(ValueError): build(reference, [rp])


@pytest.mark.parametrize('change', ['reasons', 'missing_probe'])
def test_resealed_strict_trace_not_accepted(reference, change):
    path = reference['folder']/'supervision/query_trace.json'; value = json_at(path)
    if change == 'reasons': value['reasons'] = ['query_to_cut_chain_local_usability']
    else: value.pop('probe_count')
    sha = write(path, value)
    reference['result']['capture']['records'][0]['trace_sha256'] = sha
    reference['automatic']['records'][0]['trace_sha256'] = sha
    reseal(reference)
    with pytest.raises(ValueError): build(reference)


def test_duplicate_and_generic_capture_or_review_pins_refused(reference):
    with pytest.raises(ValueError): rb.build_bank([reference['pin'], reference['pin']])
    with pytest.raises(ValueError): rb.build_bank([{'capture_path': reference['pin'].capture_path}])
    rp = review_pin(reference)
    with pytest.raises(ValueError): build(reference, [rp, rp])


def test_create_only_bank_and_exact_external_anchor_pin(reference, tmp_path):
    output = tmp_path/'published/bank.json'; rp = review_pin(reference)
    info = rb.write_bank(output, [reference['pin']], reviews=[rp]); bank = json_at(output)
    entry = captured(bank)
    proof = rb.verify_anchor(output, bank_sha256=info['sha256'], entry_id=entry['id'])
    assert proof['schema'] == rb.PROOF_SCHEMA and not proof['training_approved']
    assert proof['pose_prior']['sample'] == entry['pose_prior']['sample']
    with pytest.raises(ValueError): rb.verify_anchor(output, bank_sha256='0'*64, entry_id=entry['id'])
    held = next(e for e in bank['entries'] if e['native_observation'] is None)
    with pytest.raises(ValueError): rb.verify_anchor(output, bank_sha256=info['sha256'], entry_id=held['id'])
    with pytest.raises(ValueError): rb.write_bank(output, [reference['pin']], reviews=[rp])
    bank['entries'][0]['source_cap_reset'] = True
    bank['sha256'] = rb.inv._hash({k: v for k, v in bank.items() if k != 'sha256'})
    with pytest.raises(ValueError): rb.check_bank(bank)


def test_new_chain_cannot_silently_use_old_scene_authority(reference):
    plan = reference['plan']
    plan['source_collection_plan'] = str(reference['root']/'old_plan.json')
    plan['source_collection_plan_sha256'] = pin(Path(plan['source_collection_plan']))
    reference['expected_plan'].clear(); reference['expected_plan'].update(deepcopy(plan))
    reseal(reference)
    with pytest.raises(ValueError): build(reference)


@pytest.mark.skipif(os.environ.get('ORIGINAL_REFERENCE_REAL_PILOT') != '1', reason='explicit saved real-pilot CPU replay opt-in')
def test_actual_seed73_pilot_no_mocks():
    repo = Path(__file__).resolve().parents[4]
    root = repo/'data/sim_data/diagnostics/native_bounded12_and_original_queue_20260916_v3'
    capture = rb.OriginalCapturePin(str(root/'original_capture'),
        '1c71be314e14bb24cf6723437ea10ca15569b7de20b9cb09b92174ee4f6fddec',
        str(root/'original_launcher_receipt.json'), '224e6525b597b4ffe7e0374513311921be7e533265b992daa0bb51a4da341a85')
    review = rb.ReviewPin(str(repo/'data/sim_data/dataset_reviews/original_seed73_native_20260916_v1/assistant_visual.json'),
        '318cb5bcfb94474a852f064d6e8595432487f4437d5a2ee3369fdd88b77e6bb2')
    bank = rb.build_bank([capture], reviews=[review]); entry, = bank['entries']
    assert bank['counts']['reference_ready_targets'] == 1
    assert entry['target_id'] == 'seed73_full/SubStem_41'
    assert entry['pose_prior']['declaration']['prior_lighting']['dome_intensity'] == 1200
    assert entry['scene_authority']['actual_lighting']['dome_intensity'] == 6000
    assert entry['native_observation']['audit_replayed_by_this_build']
    assert not entry['native_observation']['paired_848_1696_proof']
