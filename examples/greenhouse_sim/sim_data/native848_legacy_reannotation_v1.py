"""Typed, create-only 9 mm diagnostics for authenticated legacy native848 frames.

The source metadata and historical synchronization contract remain unchanged.
This adapter grants no full-scene coverage, visual acceptance, or export approval.
"""
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import time

import numpy as np
from PIL import Image

from .capture_contract import fingerprint
from .capture_visibility import component_masks
from .collection_plan import load_plan
from .dataset_review import require
from .native848_all_petiole_9mm_v1 import evaluate_frame, geometry_9mm
from .native848_bulk_workspace_v1 import BulkWorkspace

SCHEMA = 'greenhouse.native848_legacy_reannotation.v1'
INVENTORY_SHA = '87a5298c248fa97f91688045bb26c664d74d0f4d51f40379371073bb23121a92'
BALANCED_SHA = '74681f8091c8c4a41f3ae5623f54805b51fa48cd69aa6c754b8393ffc5ba003b'
SOURCE_TYPES = {
    'greenhouse.rgbd_pilot_sample.v2': 'historical_pilot_separate_native_buffers',
    'greenhouse.original848_reference_direct_camera_sample.v2': 'reference_direct_separate_native_buffers',
    'greenhouse.original848_direct_camera_sample.v1': 'direct_separate_native_buffers',
    'greenhouse.original848_short_camera_sample.v1': 'short_separate_native_buffers',
    'greenhouse.original848_bulk_sample.v1': 'compact_native_buffers',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


class BoundSources:
    """Verified immutable input cache. Call verify() again before publishing."""
    def __init__(self):
        self.bindings = {}
        self.json_cache = {}
        self.plan_cache = {}

    def path(self, spec):
        path = Path(spec['path']).resolve()
        key = str(path)
        if key not in self.bindings:
            self.bindings[key] = digest(path)
        require(self.bindings[key] == spec['sha256'], 'Changed legacy source: ' + key)
        return path

    def read(self, spec):
        path = self.path(spec)
        if str(path) not in self.json_cache:
            self.json_cache[str(path)] = json.loads(path.read_text(encoding='utf-8-sig'))
        return self.json_cache[str(path)]

    def verify(self):
        for path, expected in self.bindings.items():
            require(digest(path) == expected, 'Legacy source changed during diagnostic: ' + path)


@dataclass(frozen=True)
class LegacyFrame:
    source_type: str
    metadata: dict
    rgb: np.ndarray
    depth: np.ndarray
    valid: np.ndarray
    renderer_ids: np.ndarray
    components: np.ndarray
    catalogue: list
    mapping: dict
    authentication: dict


def load_legacy_frame(row, sources):
    """Normalize only array storage; retain the exact original metadata object."""
    meta = sources.read(row['sample'])
    schema = meta['schema_version']
    require(schema in SOURCE_TYPES and row['metadata_schema'] == schema,
            'Unsupported or substituted legacy source type')
    require(meta['supervision']['target_id'] == row['target_id'], 'Legacy target substitution')
    raw = row['raw_artifacts']
    with Image.open(sources.path(raw['rgb'])) as im:
        require(im.mode == 'RGB' and im.size == (848, 408), 'Original RGB848 required')
        rgb = np.asarray(im).copy()
    if schema == 'greenhouse.original848_bulk_sample.v1':
        require(row['raw_format'] == 'compact_native_npz', 'Wrong compact source route')
        obs = sources.read(raw['observation'])
        ctx = sources.read(raw['context'])
        require(meta['observation_sha256'] == raw['observation']['sha256']
                and Path(meta['observation_path']).resolve() == sources.path(raw['observation']),
                'Legacy observation binding differs')
        require(obs['observation_id'] == meta['sample_id'] and obs['target_id'] == row['target_id']
                and obs['calibration'] == meta['calibration'] and obs['robot_snapshot'] == meta['robot_snapshot'],
                'Legacy observation state differs')
        require(obs['context_sha256'] == raw['context']['sha256']
                and Path(obs['context_path']).resolve() == sources.path(raw['context']),
                'Legacy context binding differs')
        for name in ('rgb', 'buffers'):
            require(obs['files'][name] == raw[name], 'Legacy sensor asset substitution')
        require(ctx['catalogue_sha256'] == raw['catalogue']['sha256']
                and Path(ctx['catalogue_path']).resolve() == sources.path(raw['catalogue'])
                and obs['mapping']['sha256'] == raw['mapping']['sha256']
                and Path(obs['mapping']['path']).resolve() == sources.path(raw['mapping']),
                'Legacy identity source substitution')
        catalogue = sources.read(raw['catalogue'])
        identity = sources.read(raw['mapping'])
        with np.load(sources.path(raw['buffers']), allow_pickle=False) as arrays:
            require(set(arrays.files) == {'depth_m', 'depth_valid', 'renderer_instance_id', 'rgba_alpha'},
                    'Unexpected legacy compact buffers')
            depth, valid, ids, alpha = [arrays[k] for k in
                ('depth_m', 'depth_valid', 'renderer_instance_id', 'rgba_alpha')]
        require(alpha.shape == (408, 848) and alpha.dtype == np.uint8, 'Invalid native alpha buffer')
        saved_components = None
    else:
        require(row['raw_format'] == 'native_separate_npy_png', 'Wrong separate source route')
        require(row['native_source_sample']['sha256'] == row['sample']['sha256'], 'Source sample copy differs')
        sources.path(row['native_source_sample'])
        for role, rel in [('rgb', 'inputs/rgb.png'), ('depth', 'inputs/depth_m.npy'),
                          ('validity', 'inputs/depth_valid.png'), ('renderer_ids', 'supervision/renderer_instance_id.npy'),
                          ('components', 'supervision/component_id.npy'), ('identities', 'supervision/identities.json')]:
            require(raw[role]['sha256'] == meta['files'][rel]['sha256'], 'Historical artifact hash differs')
        depth = np.load(sources.path(raw['depth']), allow_pickle=False)
        ids = np.load(sources.path(raw['renderer_ids']), allow_pickle=False)
        saved_components = np.load(sources.path(raw['components']), allow_pickle=False)
        with Image.open(sources.path(raw['validity'])) as im:
            validity = np.asarray(im).copy()
        require(validity.shape == (408, 848) and set(np.unique(validity)) <= {0, 255}, 'Invalid validity PNG')
        valid = validity == 255
        identity = sources.read(raw['identities'])
        catalogue = identity['component_catalogue']
    require(depth.shape == valid.shape == ids.shape == (408, 848) and depth.dtype == np.float32
            and valid.dtype == bool and ids.dtype == np.uint32, 'Exact native array types required')
    cal = meta['calibration']
    near, far = cal['clipping_range_m']
    require(cal['resolution'] == [848, 408] and cal['crop_resize'] is None
            and cal['depth_convention'] == 'optical_axis_z_metres_not_ray_range'
            and np.array_equal(valid, np.isfinite(depth) & (depth > 0) & (depth >= near) & (depth <= far)),
            'Native depth/calibration differs')
    mapping = {int(key): value for key, value in identity['renderer_id_to_prim'].items()}
    require(not (set(map(int, np.unique(ids))) - set(mapping) - {0, 1}), 'Unmapped renderer IDs')
    components, _, _ = component_masks(ids, mapping, catalogue)
    if saved_components is not None:
        require(np.array_equal(components, saved_components), 'Component map does not reproduce from native renderer IDs')
    sync = meta['synchronization']
    fresh = sync['freshness']
    require(sync['scene_unchanged_during_capture'] is True and sync['dynamic_recording_supported'] is False,
            'Historical static synchronization unavailable')
    hashes = dict(camera_sha256=fingerprint(cal), rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
                  depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest())
    require(all(fresh[key] == value for key, value in hashes.items()), 'Original decoded camera/RGB/depth hash differs')
    instance_hash = hashlib.sha256(ids.tobytes()).hexdigest()
    instance_pins = [v for v in [fresh.get('instance_sha256'), sync.get('instance_buffer_sha256'),
                                meta.get('native_instance_sha256')] if v is not None]
    require(instance_pins and all(v == instance_hash for v in instance_pins), 'Original renderer buffer hash differs')
    mapping_pins = [v for v in [fresh.get('mapping_sha256'), meta.get('native_mapping_sha256')] if v is not None]
    require(all(v == fingerprint(mapping) for v in mapping_pins), 'Original renderer mapping hash differs')
    if 'static_after' in sync:
        require(sync['static_before'] == sync['static_after'], 'Static scene changed')
    if 'static_guard_after' in sync:
        require(sync['static_guard'] == sync['static_guard_after'], 'Static scene guard changed')
    auth = dict(source_type=SOURCE_TYPES[schema], original_schema=schema, metadata=row['sample'],
                original_synchronization=sync, raw_artifacts=raw, decoded_hashes=hashes,
                renderer_buffer_sha256=instance_hash, renderer_mapping_sha256=fingerprint(mapping),
                source_metadata_rewritten=False, new_pilot_profile_claimed=False,
                historical_synchronization_preserved=True, normalized_array_storage_only=True,
                source_schema_dispatch_explicit=True)
    return LegacyFrame(SOURCE_TYPES[schema], meta, rgb, depth, valid, ids, components, catalogue, mapping, auth)


def reports_for_row(row, sources):
    plans = list(row['source_plans'])
    if not plans and row['metadata_schema'] == 'greenhouse.original848_short_camera_sample.v1':
        plan = sources.read(row['native_plan'])
        cache = sources.read(dict(path=plan['cache_path'], sha256=plan['cache_sha256']))
        plans.append(dict(path=cache['source_collection_plan'], sha256=cache['source_collection_plan_sha256']))
    require(plans, 'Missing authenticated legacy anatomy plan')
    reports = {}
    for spec in plans:
        path = sources.path(spec)
        if str(path) not in sources.plan_cache:
            plan, found = load_plan(path)
            sources.bindings.update(plan['source_bindings_sha256'])
            sources.plan_cache[str(path)] = found
        for report in sources.plan_cache[str(path)]:
            family = report['plant_id']
            require(family not in reports or reports[family] == report, 'Conflicting original anatomy reports')
            reports[family] = report
    return reports


def primary_inventory(frame, row, reports):
    """All same-plant targets use its recorded transform; other plants stay unknown."""
    variant, component = row['target_id'].split('/')
    family = row['source_plant_family']
    require(family in reports, 'Primary source family absent from authenticated plan')
    inventory = []
    for item in frame.catalogue:
        if item['organ_type'] != 'sub_stem' or item['variant_id'] != variant:
            continue
        require(item['source_plant_id'] == family, 'Primary variant/family mismatch')
        inventory.append(dict(target_id=variant+'/'+item['component_id'], source_family=family,
                              report=reports[family], plant_to_world_usd_row_vectors=
                              frame.metadata['supervision']['plant_to_world_usd_row_vectors']))
    require(any(entry['target_id'] == row['target_id'] for entry in inventory), 'Primary target absent from catalogue')
    return inventory


def run(inventory_path, selection_path, output, mode='balanced79'):
    output = Path(output).resolve()
    require(not output.exists(), 'Create-only legacy diagnostic directory required')
    sources = BoundSources()
    inventory = sources.read(dict(path=str(inventory_path), sha256=INVENTORY_SHA))
    selected = sources.read(dict(path=str(selection_path), sha256=BALANCED_SHA))
    require(len(inventory['records']) == 849 and len(selected['records']) == 79, 'Frozen legacy population differs')
    ids = {r['id']: r for r in inventory['records']}
    require(all(ids[r['id']] == r for r in selected['records']), 'Balanced rows differ from inventory')
    rows = selected['records'] if mode == 'balanced79' else inventory['records']
    if mode == 'exact477train':
        rows = [r for r in rows if r['split'] == 'train' and r['registry_origin'] == 'user_selected_diverse_base']
        require(len(rows) == 477, 'Exact original TRAIN477 population differs')
    rows = sorted(rows, key=lambda r: (r['registry_origin'] != 'user_selected_diverse_base',
                                      r['split'] != 'train', r['source_plant_family'], r['target_id'], r['id']))
    output.mkdir(parents=True)
    checker = BulkWorkspace()
    summaries = []
    start = time.perf_counter()
    sources.bindings[str(Path(__file__).resolve())] = digest(__file__)
    from . import native848_all_petiole_9mm_v1 as label_api
    sources.bindings[str(Path(label_api.__file__).resolve())] = digest(label_api.__file__)
    for index, row in enumerate(rows):
        frame = load_legacy_frame(row, sources)
        reports = reports_for_row(row, sources)
        entries = primary_inventory(frame, row, reports)
        annotation = evaluate_frame(frame.metadata, frame.rgb, frame.depth, frame.valid, frame.components,
                                    frame.catalogue, entries, checker, scene_coverage=None)
        primary = next(t for t in annotation['targets'] if t['target_id'] == row['target_id'])
        entry = next(t for t in entries if t['target_id'] == row['target_id'])
        geometry = geometry_9mm(entry['report'], row['target_id'].split('/')[1],
                               entry['plant_to_world_usd_row_vectors'], frame.metadata['calibration'])
        old = frame.metadata['supervision']['nominal_world_m']
        require(np.allclose(old, geometry['legacy_10mm_comparison']['world_m'], atol=1e-7, rtol=0),
                'Original primary 10mm position does not reproduce from original anatomy/transform')
        annotation.update(legacy_source=frame.authentication, original_registry_id=row['id'],
                          original_split=row['split'], original_registry_origin=row['registry_origin'],
                          original_primary_target_id=row['target_id'], primary_geometry_9mm=geometry,
                          source_transform_scope='recorded_primary_plant_only_other_plant_transforms_unknown',
                          root_visual_QA_complete=False, accepted_training_increment=0)
        name = row['id'] + '.json'
        write(output/name, annotation)
        summary = dict(id=row['id'], split=row['split'], registry_origin=row['registry_origin'],
                       target_id=row['target_id'], source_plant_family=row['source_plant_family'],
                       source_type=frame.source_type, primary_status=primary['status'], primary_reason=primary['reason'],
                       old_10mm_pixel_xy=geometry['legacy_10mm_comparison']['projected']['pixel_xy'],
                       nominal_9mm_pixel_xy=geometry['nominal']['projected']['pixel_xy'],
                       primary_workspace=primary.get('workspace'),
                       candidate_target_ids=annotation['target_census']['candidate_target_ids'],
                       unknown_target_count=len(annotation['target_census']['unknown_target_ids']),
                       catalogue_target_count=len(annotation['targets']),
                       annotation=dict(path=str(output/name), sha256=digest(output/name)), accepted_increment=0)
        summaries.append(summary)
        if (index+1) % 10 == 0 or index+1 == len(rows):
            print(json.dumps(dict(processed=index+1, total=len(rows),
                                  primary_statuses=dict(Counter(r['primary_status'] for r in summaries)))), flush=True)
    stats = checker.finish()
    sources.bindings.update(checker.bindings)
    sources.verify()
    result = dict(schema=SCHEMA, mode=mode, record_count=len(summaries),
                  summary=dict(primary_statuses=dict(Counter(r['primary_status'] for r in summaries)),
                               primary_reasons=dict(Counter(r['primary_reason'] for r in summaries)),
                               source_types=dict(Counter(r['source_type'] for r in summaries)),
                               splits=dict(Counter(r['split'] for r in summaries))),
                  records=summaries, source_bindings=sources.bindings, workspace_stats=stats,
                  full_scene_coverage_complete=False, full_scene_unknowns_preserved=True,
                  root_visual_QA_complete=False, native_launched=False, originals_modified=False,
                  accepted_increment=0, training_approved=False, elapsed_seconds=time.perf_counter()-start)
    write(output/'result.json', result)
    print(json.dumps(dict(result=str(output/'result.json'), sha256=digest(output/'result.json'),
                          summary=result['summary'], accepted_increment=0)), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--mode', choices=('balanced79', 'exact477train', 'all849'), default='balanced79')
    args = parser.parse_args()
    run(args.inventory, args.selection, args.output, args.mode)


if __name__ == '__main__':
    main()
