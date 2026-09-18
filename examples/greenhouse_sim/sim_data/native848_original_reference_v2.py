"""Authenticated original native848 reference; no generated catalogue dependency."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from .dataset_review import require, read_json, verify_bindings
from .depth_preview import sha256
from .capture_sensor import LEGACY_RESOLUTION, calibration_for_native_resolution
from .native_greenhouse_pair import load_source

SCHEMA = 'greenhouse.original_native848_bank_reference.v2'
BANK_SHA256 = '259c81dcc3e38c9f9f7bebaba574a01deeaa76a40dd3a5e62fdf3242de237098'
FLAGS = dict(generated_geometry_used=False, source_cap_reset=False,
             generated_target_inherits_review_or_workspace=False,
             geometry_novelty_qualified=False, training_approved=False)


def _build(bank_path, entry_id, *, full):
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    from .native848_pair_audit_v2 import world_from_row
    bank_path = Path(bank_path).resolve()
    require(sha256(bank_path) == BANK_SHA256, 'Pinned reviewed reference bank required')
    bank = read_json(bank_path)
    require(bank['resolution'] == [848, 408] and bank['training_approved'] is False
            and bank['maximum_views_per_original_target'] == 12, 'Changed bank scope')
    matches = [e for e in bank['entries'] if e['id'] == entry_id]
    require(len(matches) == 1, 'Unique exact bank reference required')
    entry = matches[0]
    require(entry['native848_original_reference'] is True and entry['split'] == 'train'
            and entry['source_cap_reset'] is False and entry['generated_novelty_qualified'] is False
            and entry['generated_target_inherits_review_or_workspace'] is False
            and entry['training_approved'] is False, 'Original TRAIN reference required')
    manifest, sample, source_bindings = load_source(entry['source_capture'], entry['source_sample'])
    sample_path = Path(entry['source_capture']).resolve()/entry['source_sample']/'sample.json'
    require(sample_path == Path(entry['source_sample_path']).resolve()
            and sha256(sample_path) == entry['source_sample_sha256'], 'Changed exact source sample')
    source_plan_path = Path(entry['source_plan_path']).resolve()
    require(source_plan_path == Path(manifest['source_collection_plan_path']).resolve()
            and sha256(source_plan_path) == entry['source_plan_sha256']
            == manifest['source_collection_plan_sha256'], 'Changed original collection plan')
    source = read_json(source_plan_path)
    family, row = entry['source_family'], entry['source_row']
    require(source['family_assignments'][family] == 'train'
            and entry['original_view_cap_group'] == row['target_id'] == entry['target_id']
            == sample['supervision']['target_id'], 'Changed original target or split')
    jobs = [j for j in source['jobs'] if j['job_id'] == manifest['collection_job_id']]
    require(len(jobs) == 1 and jobs[0]['split'] == 'train' and jobs[0]['plant_family'] == family
            and [r for r in jobs[0]['targets'] if r['target_id'] == row['target_id']] == [row]
            and row['draft_id'] == sample['supervision']['review_id']
            and row['cut_region_proposal'] == sample['supervision']['cut_region_proposal'],
            'Original scheduled geometry differs')
    require(entry['calibration'] == sample['calibration']
            == calibration_for_native_resolution(sample['calibration'], LEGACY_RESOLUTION)
            and entry['robot_snapshot'] == sample['robot_snapshot'], 'Reference embodiment differs')
    pose = sample['robot_snapshot']
    camera = np.asarray(pose['robot_root_to_world_usd_row_vectors']).T @ \
        Rby1Kinematics().all_link_transforms(pose['joint_degrees'])['link_head_2'] @ \
        np.asarray(pose['camera_to_head_column_vectors'])
    require(np.allclose(camera.T, sample['calibration']['camera_to_world_usd_row_vectors'], atol=1e-8, rtol=0),
            'Reference camera does not match mounted robot FK')
    variants = [v for v in manifest['variants'] if v['variant_id'] == row['variant_id']]
    require(len(variants) == 1 and variants[0]['source_plant_id'] == family
            and variants[0]['split_group'] == family
            and variants[0]['source_geometry_modified'] is False
            and not variants[0]['added_components'] and not variants[0]['added_component_paths'],
            'Unmodified original placement required')
    review = entry['source_review']
    rgb_path = sample_path.parent/'inputs/rgb.png'
    require(review['decision'] == 'accept' and review['rgb_sha256'] == sha256(rgb_path),
            'Pinned actual visual acceptance required')
    workspace_path, label_path = Path(entry['workspace_proof_path']), Path(entry['source_label_path'])
    require(sha256(workspace_path) == entry['workspace_proof_sha256']
            and sha256(label_path) == entry['source_label_sha256'], 'Changed source label/workspace')
    proof, label = read_json(workspace_path), read_json(label_path)
    verify_bindings(proof['source_bindings'])
    require(proof['sample_sha256'] == sha256(sample_path) and proof['result']['workspace_passed'] is True
            and proof['target_id'] == label['target_id'] == row['target_id'], 'Source workspace target differs')
    original_world = {k: deepcopy(sample['supervision'][k]) for k in
                      ('plant_to_world_usd_row_vectors', 'nominal_world_m', 'interval_world_m')}
    require(original_world == entry['original_world']
            and np.allclose(proof['nominal_world_m'], original_world['nominal_world_m'], atol=1e-8, rtol=0),
            'Source world geometry differs')
    package = Path(manifest['package']).resolve()
    require(package == Path(source['package']).resolve(), 'Original package differs')
    scene_code = {str(p.resolve()): sha256(p) for p in sorted((package/'env_panel/tomato_env').glob('*.py'))}
    require(str(package/'env_panel/tomato_env/daylight.py') in scene_code, 'Pinned daylight source required')
    anchor = dict(schema_version=SCHEMA, state='original_native848_reference_bound_not_executed',
        resolution=[848, 408], reference_bank_path=str(bank_path), reference_bank_sha256=BANK_SHA256,
        reference_entry_id=entry_id, source_capture=entry['source_capture'], source_sample=entry['source_sample'],
        source_collection_plan=entry['source_plan_path'], source_row=deepcopy(row), source_family=family,
        original_variant=deepcopy(variants[0]), split='train', split_group=family,
        conservative_view_cap_group=row['target_id'], expected_scene_counts=deepcopy(manifest['scene_counts']),
        expected_robot_snapshot=deepcopy(pose), expected_calibration=deepcopy(sample['calibration']),
        expected_original_world=original_world, scene_variants=deepcopy(manifest['variants']),
        source_review=deepcopy(review), source_workspace=dict(path=str(workspace_path),sha256=sha256(workspace_path)),
        scene_code_bindings=scene_code,
        source_bindings={**source_bindings, str(bank_path): BANK_SHA256,
            str(workspace_path.resolve()): sha256(workspace_path), str(label_path.resolve()): sha256(label_path),
            **proof['source_bindings'], **scene_code}, **FLAGS)
    actual_world = world_from_row(anchor, row)
    require(all(np.allclose(actual_world[k], v, atol=1e-9, rtol=0) for k,v in original_world.items()),
            'Original row-to-world reconstruction differs')
    report = None
    if full:
        from .collection_plan import load_plan
        loaded, reports = load_plan(source_plan_path)
        require(loaded == source, 'Loaded source plan differs')
        report = next(r for r in reports if r['plant_id'] == family)
    verify_bindings(anchor['source_bindings'])
    return anchor, report


def prepare(bank_path, entry_id):
    return _build(bank_path, entry_id, full=False)[0]


def check(anchor, *, full=False):
    require(anchor['schema_version'] == SCHEMA and all(anchor[k] is v for k,v in FLAGS.items()),
            'Changed original reference scope')
    expected, report = _build(anchor['reference_bank_path'], anchor['reference_entry_id'], full=full)
    require(anchor == expected, 'Original reference fields or source pins changed')
    return report
