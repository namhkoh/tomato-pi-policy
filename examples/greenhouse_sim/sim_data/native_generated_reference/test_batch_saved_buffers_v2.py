"""Opt-in CPU regression using existing native bytes, NOT new native evidence.

An isolated pytest directory gets a synthetic v2 wrapper around one real v1
generated frame. The reference proof/header is test-only. No production bank,
plan, owner receipt, capture, inventory or qualification is created or changed.
The actual full native arrays, generated hierarchy, optics, FK, metric anatomy,
labels and trace are replayed without mocks by the new sample auditor.
"""
from copy import deepcopy
import os
from pathlib import Path

import numpy as np
import pytest

from . import batch_execution_v2 as ex, batch_audit_v2 as audit

pytestmark = pytest.mark.skipif(os.environ.get('ORIGINAL_BATCH_EXISTING_BUFFER_TEST') != '1',
    reason='explicit existing-buffer CPU regression only')


def test_existing_native_arrays_full_v2_sample_replay_and_tamper(tmp_path):
    from ..native_dataset.bundle import SampleReader
    from ..native_dataset import inventory as inv
    from ..native_dataset.capture_storage import write_compact_native_sample
    from ..plant_variant_catalogue import load_for_inspection
    from ..collection_plan import load_plan
    from ..capture_contract import fingerprint
    from ..capture_sensor import calibration_for_native_resolution
    from ..native_clear_labels import derive
    from ..automated_native_review import trace_review
    from .batch_scene_v2 import current_scene_evidence
    from .batch_pose_v2 import reference_evidence
    root = Path(__file__).resolve().parents[4]
    prior = root/'data/sim_data/collection_batches/native_reference_scale_20260916_v4/job_0050_seed19_full_round4'
    old_plan = ex.oc.read_json(ex.oc.pin(prior/'plan.json', '07292bd865f83414eef678d2475b3e323245206efc40fe3dd58664d278ba7c0c'))
    old_result = ex.oc.read_json(ex.oc.pin(prior/'capture/result.json', 'd6a5bb4b3bd1bd65b57f961929e09ec552cae3a0e0f1ef6cfc13355a950bab06'))
    name = 'SubStem_42_view_001'
    old_record = next(r for r in old_result['records'] if r['candidate_id'] == name)
    old_case = next(c for c in old_plan['target_cases'] if c['target_id'] == old_record['target_id'])
    base = ex.oc.read_json(ex.oc.pin(old_case['base_pair_plan'], old_case['base_pair_plan_sha256']))
    spec = next(s for s in old_case['views'] if s['candidate_id'] == name)
    reader = SampleReader(prior/'capture'/name, expected_bindings={'sample.json': old_record['sample_sha256'],
        'supervision/label.json': old_record['label_sha256']})
    reader.verify_all()
    meta = deepcopy(reader.metadata)
    assert meta['synchronization']['render_budget_subframes'] == 56
    manifest = ex.oc.read_json(Path(base['source_capture'])/'manifest.json')
    authority = dict(actual_lighting=meta['lighting'], actual_renderer=meta['renderer'],
        expected_scene_counts=meta['scene_counts'], policy=ex.oc.SCENE_POLICY,
        clear_plan=dict(path=base['source_collection_plan'], sha256=ex.oc.sha256(base['source_collection_plan'])),
        scene_variants=manifest['variants'], original_variant=base['original_variant'])
    plan = dict(source_family=base['source_family'], scene_authority=authority, input_policy=meta['input_policy'])
    reference = {k: deepcopy(base['expected_robot_snapshot'][k]) for k in
        ('joint_degrees', 'robot_root_to_world_usd_row_vectors', 'camera_to_head_column_vectors')}
    reference_cal = calibration_for_native_resolution(base['expected_calibration'], ex.oc.RESOLUTION)
    native = dict(robot_snapshot=reference, calibration=reference_cal,
        geometry_screen=base['expected_robot_snapshot']['visual_bound_screen'], audit_replayed_by_this_build=True,
        sample=dict(path='UNIT_ONLY_NOT_NATIVE_REFERENCE', sha256='0'*64),
        native_plan=dict(path='UNIT_ONLY_NOT_A_PRODUCTION_PLAN', sha256='0'*64))
    case = dict(source_row=base['source_row'], generated_row=base['generated_row'],
        target_id=base['generated_row']['target_id'], conservative_view_cap_group=base['conservative_view_cap_group'],
        expected_robot_snapshot=reference, expected_calibration=reference_cal, native_reference=native,
        expected_original_world={k: deepcopy(meta['supervision'][k]) for k in
            ('plant_to_world_usd_row_vectors', 'nominal_world_m', 'interval_world_m')})
    ids = reader.array('supervision/renderer_instance_id.npy')
    identity = reader.json('supervision/identities.json')
    mapping = {int(k): v for k, v in identity['renderer_id_to_prim'].items()}
    rgb, depth = reader.image('inputs/rgb.png'), reader.array('inputs/depth_m.npy')
    valid, mask = reader.image('inputs/depth_valid.png') != 0, reader.image('supervision/target_visible.png') != 0
    components, organs = reader.array('supervision/component_id.npy'), reader.image('supervision/organ_type.png')
    meta.pop('files')
    meta.update(schema_version=ex.SAMPLE, state='fresh_native_reference_batch_pending_replay',
        historical_labels_inherited=False, native_instance_backend='legacy',
        native_instance_sha256=ex.oc.digest(ids.tobytes()), native_mapping_sha256=fingerprint(mapping),
        old_plant_native_pixels=0, requested_spec=deepcopy(spec), pose_reference_evidence=reference_evidence(case),
        scene_evidence=current_scene_evidence(plan, lighting=meta['lighting'], counts=meta['scene_counts'], renderer=meta['renderer']))
    meta['synchronization']['actual_orchestrator_requests'] = 7
    generated = load_for_inspection(base['variant_directory'], base['source_collection_plan'])
    _, reports = load_plan(base['source_collection_plan']); reports = {r['plant_id']: r for r in reports}
    catalogue = identity['component_catalogue']
    label = derive(meta, generated['report'], rgb, depth, valid, components, catalogue)
    trace = trace_review(meta, generated['report'], label, rgb, depth, valid, components, catalogue) if label['eligible'] else None
    stored = write_compact_native_sample(tmp_path/name, rgb, depth, valid, meta, ids,
        mapping, catalogue, components, organs, mask, label, trace)
    record = dict(candidate_id=name, target_id=case['target_id'], requested_spec=spec,
        screen=meta['geometry_screen'], robot_snapshot=meta['robot_snapshot'], calibration=meta['calibration'],
        decision=inv.EXCLUDE if not label['eligible'] else inv.STRICT if trace['passed'] else inv.HOLD, **stored)
    reviewed, _ = audit.review_sample(tmp_path, plan, case, spec, record, reports, generated, inv._Bindings())
    assert reviewed['native_ID_masks_replayed'] and reviewed['native_callback_hashes_verified']
    assert reviewed['label_replayed_exact'] and not reviewed['training_approved']
    bad = deepcopy(case); bad['conservative_view_cap_group'] = 'different_donor/SubStem_42'
    with pytest.raises(ValueError, match='ancestry'):
        audit.review_sample(tmp_path, plan, bad, spec, record, reports, generated, inv._Bindings())
    bad = deepcopy(record); bad['robot_snapshot']['camera_to_head_column_vectors'][0][3] += .01
    with pytest.raises(ValueError, match='policy'):
        audit.review_sample(tmp_path, plan, case, spec, bad, reports, generated, inv._Bindings())
    bundle = ex.oc.read_json(tmp_path/name/'bundle.json')
    path = tmp_path/name/bundle['files']['inputs/depth_m.npy']['stored_path']
    data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
    with pytest.raises(ValueError):
        audit.review_sample(tmp_path, plan, case, spec, record, reports, generated, inv._Bindings())
