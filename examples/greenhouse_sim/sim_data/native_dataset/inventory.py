"""Read pinned, completed native audit receipts into an OBSERVED inventory.

No receipt discovery, audit replay, renderer, depth reconstruction, admission,
image output, cap reset or global-completeness assertion. All captured rows,
including holds/exclusions, survive. Exact duplicate groups are diagnostic only.

``build_inventory([ReceiptPin(...)], extra_observed_roots=[ObservedRoot(...)])``
is read-only. Pins explicitly name receipt, capture, plan and optional alternate
storage root; no newest-file selection or implicit capture-directory scan occurs.
Extra roots require explicit receipt pins, or remain visibly unindexed. The
returned records provide admission.decoded_rgb_digest and physical-context
features, NOT finalized global morphology/near-image groups or release evidence.

``write_inventory(new_path, pins, ...)`` exclusively creates one JSON after all
verification succeeds. Existing files, captures and reviews are never changed.
CLI: --receipt RECEIPT SHA256 CAPTURE PLAN (repeatable), --output NEW_JSON;
--extra-observed-root ROOT explicitly declares an unindexed root. Alternate
storage and pinned extra-root receipts are supported by the Python API.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from .admission import decoded_rgb_digest
from .bundle import SampleReader, digest, safe_path
from ..capture_contract import fingerprint
from ..capture_visibility import component_masks


SCHEMA = 'greenhouse.pinned_native_observed_inventory.v1'
STRICT = 'accept_strict_automatic_annotation_candidate'
HOLD = 'hold_visual_clarity'
EXCLUDE = 'exclude_geometry_or_visibility'
DECISIONS = {STRICT, HOLD, EXCLUDE}
CAPTURED = 'native_captured_pending_review'
COMPLETED = 'native_generated_multiview_pilot_complete_pending_review'
AUDITED = 'completed_automatic_annotation_replay'
MAX_JSON_BYTES = 64 * 1024 * 1024


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(value):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value),
             'Missing/invalid SHA256 pin')
    return value


def _path(value):
    path = Path(value)
    _require(path.is_absolute(), 'Explicit absolute path required: ' + str(path))
    return path.resolve()


def _id(value):
    _require(isinstance(value, str) and bool(value.strip()), 'Nonempty identity required')
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def _hash(value):
    return digest(_canonical(value))


def _file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


_CODE_PATHS = [Path(__file__), Path(__file__).with_name('admission.py'), Path(__file__).with_name('bundle.py')]
_CODE_PATHS += [Path(__file__).parent.parent / p for p in
               ('native_lossless_codec.py', 'capture_contract.py', 'capture_visibility.py')]
_LOADED_CODE = {str(p.resolve()): _file_sha(p) for p in _CODE_PATHS}


def _parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            _require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Non-finite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


@dataclass(frozen=True)
class ReceiptPin:
    receipt_path: str
    receipt_sha256: str
    capture_path: str
    plan_path: str
    storage_root: str | None = None


@dataclass(frozen=True)
class ObservedRoot:
    """Explicit extra coverage; an empty receipts tuple means NOT indexed."""
    path: str
    receipts: tuple = ()


class _Bindings:
    def __init__(self):
        self.hashes = {}
        self.documents = {}
        self.protected_roots = set()

    def bind(self, path, expected):
        path, expected = _path(path), _sha(expected)
        key = str(path)
        _require(key not in self.hashes or self.hashes[key] == expected,
                 'Conflicting source hashes: ' + key)
        if key not in self.hashes:
            _require(path.is_file() and _file_sha(path) == expected, 'Stale bound file: ' + key)
            self.hashes[key] = expected
        return path

    def mapping(self, values):
        _require(isinstance(values, dict) and values, 'Nonempty original bindings required')
        for path, expected in values.items():
            self.bind(path, expected)

    def document(self, path, expected=None):
        path = _path(path)
        expected = expected or self.hashes.get(str(path))
        self.bind(path, expected)
        if str(path) not in self.documents:
            with path.open('rb') as stream:
                raw = stream.read(MAX_JSON_BYTES + 1)
            _require(len(raw) <= MAX_JSON_BYTES and digest(raw) == expected,
                     'Metadata changed or exceeds bound')
            self.documents[str(path)] = _parse(raw)
        return self.documents[str(path)]

    def finish(self):
        for path, expected in self.hashes.items():
            _require(_file_sha(path) == expected, 'Source changed during inventory: ' + path)


def _unique(rows, key):
    _require(isinstance(rows, list), 'Row list required')
    out = {}
    for row in rows:
        value = _id(row[key])
        _require(value not in out, 'Duplicate ' + key)
        out[value] = row
    return out


def _matrix(value):
    array = np.asarray(value, dtype=float)
    _require(array.shape == (4, 4) and np.isfinite(array).all(), 'Invalid actual pose matrix')
    return array.tolist()


def _case_lineage(bindings, plan, anchor, case):
    """Cross-bind copied ancestry to original TRAIN rows and generated assets.

    Structural/source validation only: no generator, USD, label or audit replay.
    A pin authenticates supplied bytes, not the historical execution of audit.
    """
    base_path = _path(case['base_pair_plan'])
    _require(bindings.hashes.get(str(base_path)) == case['base_pair_plan_sha256'],
             'Case base is not in original plan bindings')
    base = bindings.document(base_path, case['base_pair_plan_sha256'])
    bindings.mapping(base['source_bindings'])
    family = _id(plan['source_family'])
    for name in ('source_family', 'split', 'split_group', 'variant_directory',
                 'source_collection_plan', 'original_variant'):
        _require(base[name] == anchor[name], 'Case/anchor lineage mismatch: ' + name)
    _require(base['source_family'] == base['split_group'] == family and base['split'] == 'train',
             'Base original TRAIN ancestry mismatch')
    source_plan_path = _path(base['source_collection_plan'])
    original = bindings.document(source_plan_path)
    source_root = _path(base['source_capture'])
    manifest = bindings.document(source_root / 'manifest.json')
    _require(_path(manifest['source_collection_plan_path']) == source_plan_path
             and manifest['source_collection_plan_sha256'] == bindings.hashes[str(source_plan_path)]
             and manifest['target_family_split'] == original['family_assignments'].get(family) == 'train',
             'Original frozen TRAIN plan/capture mismatch')
    jobs = _unique(original['jobs'], 'job_id')
    job = jobs[manifest['collection_job_id']]
    _require(job['plant_family'] == family and job['split'] == 'train', 'Wrong original TRAIN job')
    source, generated = base['source_row'], base['generated_row']
    key = _id(source['component_id'])
    original_target = family + '/' + key
    _require(_unique(job['targets'], 'target_id').get(original_target) == source,
             'Base source row differs from original frozen job')
    _require(source['target_id'] == original_target and source['variant_id'] == family
             and source['source_plant_id'] == source['split_group'] == family,
             'Original source-row ancestry mismatch')
    source_manifest = _path(job['source_manifest_path'])
    donor = bindings.document(source_manifest, original['source_bindings_sha256'][str(source_manifest)])
    _require(source['source_manifest_sha256'] == bindings.hashes[str(source_manifest)]
             and key in _unique(donor['components'], 'id'), 'Source component/manifest mismatch')
    original_variants = [v for v in manifest['variants'] if v['variant_id'] == family]
    _require(original_variants == [base['original_variant']]
             and original_variants[0]['source_plant_id'] == original_variants[0]['split_group'] == family,
             'Original scene placement lineage mismatch')
    sample_name = _id(base['source_sample'])
    sample = bindings.document(safe_path(source_root, sample_name + '/sample.json'))
    _require(sample['sample_id'] == sample_name
             and sample['supervision']['target_id'] == original_target
             and sample['supervision']['split_group'] == family
             and sample['supervision']['cut_region_proposal'] == source['cut_region_proposal']
             and _unique(manifest['samples'], 'sample_id')[sample_name]['target_review_id'] == source['draft_id'],
             'Original captured reference/source row mismatch')
    directory = _path(base['variant_directory'])
    qualification = bindings.document(directory / 'qualification.json')
    geometry = bindings.document(directory / 'manifest.json')
    _require(not (directory / 'FAILED.json').exists()
             and qualification['variant_id'] == directory.name
             and qualification['source_family'] == qualification['split_group'] == family
             and qualification['split'] == 'train'
             and _path(qualification['source_manifest_path']) == source_manifest,
             'Generated qualification donor mismatch')
    version = qualification['version']
    if version in ('curved_relocated_petiole_static.v1', 'curved_relocated_rigid_leaf_static.v2'):
        _require(_path(qualification['source_plan_path']) == source_plan_path
                 and qualification['source_plan_sha256'] == bindings.hashes[str(source_plan_path)]
                 and qualification['frozen_family_assignments'] == original['family_assignments'],
                 'Generated frozen TRAIN source-plan mismatch')
        bindings.mapping(qualification['source_bindings'])
        target = _unique(qualification['targets'], 'component_id')[key]
        change = _unique(qualification['recipes'], 'component_id')[key]
        _require(change['source_target_id'] == original_target
                 and change['members'] == generated['expected_detached_component_ids'],
                 'Generated recipe source-component mismatch')
    else:
        _require(version == 'petiole_similarity_pilot.v1', 'Unsupported generated lineage schema')
        frozen = bindings.document(directory.parent / 'frozen_lineage.json')
        _require(frozen['source_plan_sha256'] == bindings.hashes[str(source_plan_path)]
                 and frozen['family_assignments'] == original['family_assignments'],
                 'Generated frozen TRAIN source-plan mismatch')
        target = _unique(qualification['changes'], 'component_id')[key]
        _require(target['members'] == generated['expected_detached_component_ids'],
                 'Generated subtree mismatch')
    for relative, expected in qualification['output_hashes'].items():
        bindings.bind(safe_path(directory, relative), expected)
    _require(qualification['output_hashes']['manifest.json'] == bindings.hashes[str(directory / 'manifest.json')],
             'Generated manifest output binding missing')
    component = _unique(geometry['components'], 'id')[key]
    asset = safe_path(directory, component['file'])
    _require(component['type'] == 'sub_stem'
             and qualification['output_hashes'][component['file']] == bindings.hashes[str(asset)]
             and generated['attachment_plant_m'] == component['attach_point']
             and generated['component_id'] == key and generated['variant_id'] == directory.name
             and generated['source_plant_id'] == generated['split_group'] == family
             and generated['target_id'] == case['target_id'] == directory.name + '/' + key
             and target['source_target_id'] == target['conservative_view_cap_group']
                 == generated['conservative_view_cap_group'] == base['conservative_view_cap_group']
                 == case['conservative_view_cap_group'] == original_target
             and target['cut_region_proposal'] == generated['cut_region_proposal'],
             'Generated target/component or source-cap relabel mismatch')
    bindings.protected_roots.update(map(str, (source_root, source_manifest.parent, directory)))
    return dict(base_pair_plan=str(base_path), base_pair_plan_sha256=case['base_pair_plan_sha256'],
        source_collection_plan=str(source_plan_path), source_plan_sha256=bindings.hashes[str(source_plan_path)],
        original_job_id=job['job_id'], original_source_row_sha256=_hash(source),
        frozen_family_assignments_sha256=_hash(original['family_assignments']),
        source_manifest_sha256=bindings.hashes[str(source_manifest)],
        generated_qualification_sha256=bindings.hashes[str(directory / 'qualification.json')],
        generated_row_sha256=_hash(generated), generated_component_sha256=_hash(component),
        generated_component_asset_sha256=bindings.hashes[str(asset)],
        structural_lineage_checked=True, geometry_derivation_replayed=False)


def _trace_consistency(label, trace):
    """Check the complete stored trace contract, NOT that probes were executed."""
    _require(type(label['eligible']) is bool, 'Boolean annotation eligibility required')
    if not label['eligible']:
        _require(trace is None, 'Excluded annotation unexpectedly has a trace')
        return
    _require(isinstance(trace, dict) and type(trace['passed']) is bool,
             'Complete native query trace required')
    reasons = trace['reasons']
    _require(isinstance(reasons, list) and all(isinstance(r, str) and r for r in reasons)
             and len(set(reasons)) == len(reasons) and trace['passed'] is (not reasons),
             'Trace passed/reasons contradiction')
    # The frozen producer has one documented early hold without probe metrics.
    if trace == dict(passed=False, reasons=['query_to_cut_chain_out_of_frame']):
        return
    def number(value, low, high):
        _require(type(value) in (int, float) and np.isfinite(value) and low <= value <= high,
                 'Invalid native trace metric')
        return value
    start, end = trace['arc_interval_m']
    _require(start == .008, 'Wrong native trace interval start')
    number(end, .045, 100.)
    _require(end == label['query_evidence']['arc_m'] and trace['native_depth_reconstructed'] is False
             and label['native_depth_reconstructed'] is False,
             'Missing native trace/query evidence')
    probes, pixels, gaps = (trace[k] for k in ('probe_count', 'unique_pixels', 'identity_or_depth_gap_count'))
    _require(all(type(n) is int for n in (probes, pixels, gaps))
             and 2 <= probes <= 12000 and 1 <= pixels <= probes and 0 <= gaps <= probes
             and probes >= int(np.ceil((end-start)/.0005)) + 1, 'Invalid native trace probe counts')
    number(trace['maximum_projected_step_px'], 0., .5 * (1+1e-9))
    interior = number(trace['minimum_interior_radius_px'], 0., 1696.)
    dark = number(trace['dark_fraction'], 0., 1.)
    median = number(trace['median_luminance'], 0., 255.)
    first, local = trace['first_gap_probes'], trace['local_usability_failures']
    _require(isinstance(first, list) and len(first) == min(gaps, 20)
             and isinstance(local, list)
             and len(local) <= int(np.ceil((end-start)/.005)) + 1, 'Incomplete native probe evidence')
    def position(row, field):
        number(row['arc_m'], start, end)
        x, y = row[field]
        number(x, 0., 1696. - 1e-9); number(y, 0., 816. - 1e-9)
        return x, y
    for row in first:
        xy = position(row, 'pixel_xy')
        _require(all(type(v) is int for v in xy) and type(row['exact_target']) is bool
                 and row['native_depth_status'] in ('unknown_invalid_depth', 'foreground_occlusion_evidence',
                     'depth_consistent_not_visibility_verified', 'unknown_no_target_surface_at_pixel')
                 and (not row['exact_target'] or row['native_depth_status'] != 'depth_consistent_not_visibility_verified'),
                 'Contradictory native gap probe')
    for row in local:
        position(row, 'pixel_uv')
        _require(isinstance(row['reasons'], list) and row['reasons']
                 and all(isinstance(r, str) and r for r in row['reasons']), 'Empty local failure evidence')
    expected = []
    if gaps: expected.append('query_to_cut_chain_identity_or_depth_gap')
    if interior < 2.: expected.append('query_to_cut_chain_too_thin_or_edge_on')
    if dark > .1 or median < 40.: expected.append('query_to_cut_chain_too_dark')
    if local: expected.append('query_to_cut_chain_local_usability')
    _require(reasons == expected, 'Trace metrics/reasons contradiction')


def _scene_basis(bindings, plan):
    """Content/placement evidence, excluding capture names, seeds and guard epochs.

    This is an exact recorded-scene signature, NOT a rigid/scale/name-invariant
    morphology metric. Asset-byte changes may conservatively split signatures.
    Missing scene/background evidence fails instead of falling back to a target
    name, scene count, renderer seed, static-guard generation or RGB hash.
    """
    anchor = bindings.document(plan['anchor_pair_plan'])
    bindings.mapping(anchor['source_bindings'])
    source = _path(anchor['source_capture']) / 'manifest.json'
    original = bindings.document(source)
    assets = original['source_usd_sha256']
    bindings.mapping(assets)
    scene_asset = _path(original['scene'])
    _require(str(scene_asset) in assets, 'Background scene asset is not pinned')
    variant = _path(anchor['variant_directory'])
    package = _path(original['package'])
    bindings.protected_roots.update(map(str, (variant, package, source.parent)))
    geometry = bindings.document(variant / 'manifest.json')
    components = _unique(geometry['components'], 'id')
    component_hashes = {}
    for name, component in components.items():
        asset = safe_path(variant, component['file'])
        _require(str(asset) in bindings.hashes, 'Generated component asset is not pinned')
        component_hashes[name] = bindings.hashes[str(asset)]
    physical_components = []
    for component in components.values():
        parent = component.get('parent')
        _require(parent is None or parent in components, 'Unknown geometry parent')
        physical_components.append(dict(
            asset_sha256=component_hashes[component['id']],
            parent_asset_sha256=component_hashes.get(parent),
            geometry={k: v for k, v in component.items() if k not in ('id', 'parent', 'file')}))
    physical_components.sort(key=_canonical)
    placements = []
    for row in original['variants']:
        # IDs/family/split are provenance, not new physical scene identities.
        _require(not row.get('added_components') and not row.get('added_component_paths'),
                 'Unsupported unqualified background component additions')
        family_root = package / 'plants/components' / _id(row['source_plant_id'])
        placed_assets = sorted(h for p, h in assets.items() if _path(p).is_relative_to(family_root))
        _require(placed_assets, 'Missing per-placement background plant geometry')
        placements.append(dict(plant_root=_id(row['plant_root']),
                               geometry_content_sha256=_hash(placed_assets), asset_count=len(placed_assets)))
    _require(placements and original['lighting'] and original['scene_counts'],
             'Full background/population/light evidence required')
    geometry_hash = _hash(physical_components)
    local_bindings = dict(plan['source_bindings'], **anchor['source_bindings'])
    textures = sorted(h for p, h in local_bindings.items()
        if Path(p).suffix.lower() in ('.png', '.jpg', '.jpeg', '.exr', '.hdr', '.dds', '.mdl', '.mtl')
        and (Path(p).is_relative_to(package) or Path(p).is_relative_to(variant)))
    return dict(background_scene_asset_sha256=assets[str(scene_asset)],
        background_asset_content_sha256=_hash(sorted(assets.values())), background_asset_count=len(assets),
        material_texture_content_sha256=_hash(textures), material_texture_count=len(textures),
        population_placements=sorted(placements, key=_canonical),
        replaced_plant_root=_id(anchor['original_variant']['plant_root']),
        generated_geometry_sha256=geometry_hash,
        excluded_external_prop_roots=sorted(original['unbundled_external_prop_roots_excluded']),
        renderer_settings=original['observed_renderer_settings'],
        original_lighting=original['lighting'], original_scene_counts=original['scene_counts'],
        original_renderer=original['renderer'])


def _features(meta, basis):
    cal, pose, sup = meta['calibration'], meta['robot_snapshot'], meta['supervision']
    _require(meta['lighting'] == basis['original_lighting']
             and meta['scene_counts'] == basis['original_scene_counts']
             and meta['renderer'] == basis['original_renderer'], 'Changed scene/background/light')
    joints = pose['joint_degrees']
    _require(isinstance(joints, dict) and joints and all(type(x) in (int, float)
             and np.isfinite(x) for x in joints.values()), 'Actual robot joints required')
    scene = dict(static_scene_sha256=_hash(basis), generated_plant_to_world=_matrix(sup['plant_to_world_usd_row_vectors']),
        actual_robot_root_to_world=_matrix(pose['robot_root_to_world_usd_row_vectors']),
        actual_robot_joint_degrees=joints)
    camera = {k: cal[k] for k in ('resolution', 'intrinsics', 'clipping_range_m',
        'focal_length_mm', 'apertures_mm', 'aperture_offsets_mm', 'depth_convention', 'crop_resize')}
    camera['camera_to_world'] = _matrix(cal['camera_to_world_usd_row_vectors'])
    camera['camera_to_head'] = _matrix(pose['camera_to_head_column_vectors'])
    return scene, camera


def _read_sample(bindings, folder, captured, audited):
    reader = SampleReader(folder, expected_bindings={
        'sample.json': captured['sample_sha256'],
        'supervision/label.json': captured['label_sha256']}, expected_json=(
            {'supervision/query_trace.json': captured['query_trace']}
            if captured.get('query_trace') is not None else {}))
    for key in ('sample_sha256', 'label_sha256'):
        _require(audited[key] == captured[key], 'Receipt/sample binding mismatch')
    reader.verify_all()
    # Record actual physical storage pins, including the compact completion
    # marker and raw metadata. Logical extras require capture-result bindings.
    names = set(reader.manifest['files'] if reader.manifest else reader.metadata['files'])
    names.update(('sample.json', 'supervision/label.json'))
    if captured.get('query_trace') is not None:
        names.add('supervision/query_trace.json')
    logical = {}
    for name in sorted(names):
        raw = reader.read(name)
        if name.endswith('.json'):
            _parse(raw)
        logical[name] = dict(sha256=digest(raw), bytes=len(raw))
        if reader.manifest:
            entry = reader.manifest['files'][name]
            bindings.bind(safe_path(folder, entry['stored_path']), entry['stored_sha256'])
        else:
            bindings.bind(safe_path(folder, name), digest(raw))
    if reader.manifest:
        bindings.bind(folder / 'bundle.json', _hash_file_snapshot(folder / 'bundle.json'))
    meta, label = reader.metadata, reader.json('supervision/label.json')
    rgb, depth = reader.image('inputs/rgb.png'), reader.array('inputs/depth_m.npy')
    valid = reader.image('inputs/depth_valid.png')
    components = reader.array('supervision/component_id.npy')
    target = reader.image('supervision/target_visible.png')
    instances = reader.array('supervision/renderer_instance_id.npy')
    identities = reader.json('supervision/identities.json')
    cal, sync = meta['calibration'], meta['synchronization']
    _require(cal['resolution'] == [1696, 816] and cal['crop_resize'] is None
             and cal['depth_convention'] == 'optical_axis_z_metres_not_ray_range',
             'Native train1696 calibration required')
    shape = (816, 1696)
    _require(rgb.shape == (*shape, 3) and rgb.dtype == np.uint8
             and depth.shape == shape and depth.dtype == np.float32
             and valid.shape == components.shape == target.shape == instances.shape == shape
             and components.dtype == instances.dtype == np.uint32
             and valid.dtype == target.dtype == np.uint8
             and set(np.unique(valid)) <= {0, 255} and set(np.unique(target)) <= {0, 255},
             'Invalid native buffers')
    _require(sync['method'] == 'frozen_scene_single_native_writer_payload'
             and sync['scene_unchanged_during_capture'] is True
             and sync['dynamic_recording_supported'] is False, 'Invalid callback synchronization')
    freshness = sync['freshness']
    _sha(sync['static_guard'])
    reference = sync['reference_time']
    _require(isinstance(reference, list) and len(reference) == 2
             and all(type(x) is int for x in reference) and reference[0] >= 0 and reference[1] > 0,
             'Invalid callback reference time')
    _require(type(freshness['callback_sequence']) is int and freshness['callback_sequence'] > 0,
             'Invalid callback sequence')
    _require(freshness['camera_sha256'] == fingerprint(cal)
             and freshness['rgb_sha256'] == digest(rgb.tobytes())
             and freshness['depth_sha256'] == digest(depth.tobytes()), 'Callback evidence mismatch')
    near, far = cal['clipping_range_m']
    _require(0 < near < far and np.isfinite([near, far]).all()
             and np.array_equal(valid != 0, np.isfinite(depth) & (depth > 0)
                                & (depth >= near) & (depth <= far)), 'Native depth validity mismatch')
    mapping = {int(k): v for k, v in identities['renderer_id_to_prim'].items()}
    _require(all(isinstance(v, str) and 0 <= k <= 0xffffffff for k, v in mapping.items())
             and not (set(map(int, np.unique(instances))) - mapping.keys() - {0, 1}),
             'Invalid native renderer mapping')
    rebuilt, organs, _ = component_masks(instances, mapping, identities['component_catalogue'])
    _require(np.array_equal(rebuilt, components)
             and np.array_equal(organs, reader.image('supervision/organ_type.png')),
             'Renderer/component evidence mismatch')
    variant, key = meta['supervision']['target_id'].split('/')
    targets = [c for c in identities['component_catalogue']
               if c['variant_id'] == variant and c['component_id'] == key]
    _require(len(targets) == 1 and np.array_equal(target != 0, components == targets[0]['component_index']),
             'Native target mask mismatch')
    _require(logical['inputs/rgb.png']['sha256'] == audited['rgb_sha256'], 'Receipt RGB mismatch')
    trace = reader.json('supervision/query_trace.json') if label['eligible'] else None
    _require(trace == captured.get('query_trace'), 'Capture query trace mismatch')
    _trace_consistency(label, trace)
    if label['eligible']:
        query = label['query_evidence']
        uv = query['projected']['pixel_xy']
        _require(query['visible'] is True and query['projected']['projection_status'] == 'in_frame'
                 and query['depth_status'] == 'depth_consistent_not_visibility_verified'
                 and len(uv) == 2 and all(type(v) in (int, float) and np.isfinite(v) for v in uv)
                 and 0 <= uv[0] < 1696 and 0 <= uv[1] < 816
                 and label['query_pixel_uv'] == [round(v, 1) for v in uv], 'Incomplete native query evidence')
        x, y = np.floor(uv).astype(int)
        _require(bool(target[y, x]) and bool(valid[y, x]), 'Query native identity/validity mismatch')
        for gap in trace.get('first_gap_probes', []):
            x, y = gap['pixel_xy']
            _require(gap['exact_target'] is bool(target[y, x])
                     and (gap['native_depth_status'] == 'unknown_invalid_depth') is (not bool(valid[y, x])),
                     'Stored gap contradicts native identity/validity')
    return meta, label, trace, logical, decoded_rgb_digest(rgb.tobytes(), width=1696, height=816)


def _hash_file_snapshot(path):
    # Completion marker has no external pin in audit_capture; its logical
    # contents are already checked against externally pinned sample/label files.
    return _file_sha(path)


def _receipt(bindings, pin):
    _require(isinstance(pin, ReceiptPin), 'Explicit ReceiptPin required')
    receipt_path, capture, plan_path = map(_path, (pin.receipt_path, pin.capture_path, pin.plan_path))
    storage = _path(pin.storage_root) if pin.storage_root else capture
    _require(capture.is_dir() and storage.is_dir(), 'Capture/storage directory missing')
    receipt = bindings.document(receipt_path, pin.receipt_sha256)
    _require(receipt['state'] == AUDITED and _path(receipt['capture']) == capture,
             'Wrong/incomplete receipt capture target')
    _require(receipt['training_approved'] is False and receipt['source_assets_unchanged'] is True
             and receipt['original_reviews_modified'] is False, 'Unsafe audit claim')
    for name in ('audit.py', 'bundle.py'):
        code = str(Path(__file__).with_name(name).resolve())
        _require(code in receipt['review_code_bindings'], 'Missing audit implementation binding')
    bindings.mapping(receipt['review_code_bindings'])
    _require(not (capture / 'failure.json').exists(), 'Failed capture cannot enter inventory')
    request = bindings.document(capture / 'request.json', receipt['request_sha256'])
    result = bindings.document(capture / 'result.json', receipt['result_sha256'])
    plan = bindings.document(plan_path, receipt['plan_sha256'])
    _require(_path(request['plan_path']) == plan_path and request['plan_sha256'] == receipt['plan_sha256'],
             'Capture request/plan mismatch')
    _require(request['training_started'] is False and result['state'] == COMPLETED
             and result['training_approved'] is False and result['source_assets_unchanged'] is True,
             'Incomplete/unsafe capture result')
    _require(plan['split'] == 'train' and plan['resolution'] == [1696, 816]
             and all(plan[k] is False for k in ('training_approved', 'source_cap_reset',
                 'physical_motion_commanded', 'hidden_cut_coordinates_executable')), 'Unsafe/non-TRAIN plan')
    for key in ('source_bindings', 'implementation_bindings', 'prerequisite_bindings'):
        bindings.mapping(plan[key])
    anchor_path = _path(plan['anchor_pair_plan'])
    anchor = bindings.document(anchor_path)
    _case_lineage(bindings, plan, anchor, dict(base_pair_plan=str(anchor_path),
        base_pair_plan_sha256=bindings.hashes[str(anchor_path)], target_id=anchor['generated_row']['target_id'],
        conservative_view_cap_group=anchor['conservative_view_cap_group']))
    planned = {}
    lineage = {}
    for case in plan['target_cases']:
        _require(case['target_id'] not in lineage, 'Repeated target case')
        lineage[case['target_id']] = _case_lineage(bindings, plan, anchor, case)
        for spec in case['views']:
            name = _id(spec['candidate_id'])
            _require(name not in planned and name not in ('.', '..') and not any(c in name for c in '/\\:'),
                     'Duplicate/unsafe planned candidate')
            planned[name] = (case, spec)
    captured = _unique(result['records'], 'candidate_id')
    _require(captured.keys() == planned.keys(), 'Incomplete planned capture decisions')
    for name, row in captured.items():
        case, spec = planned[name]
        _require(row['target_id'] == case['target_id'] and row['requested_spec'] == spec,
                 'Different capture target/view')
        _require(row['state'] in (CAPTURED, 'rejected_pose', 'rejected_possible_geometry_overlap'),
                 'Unknown capture state')
    captured = {k: v for k, v in captured.items() if v['state'] == CAPTURED}
    audited = {}
    for row in receipt['records']:
        folder = _path(row['sample'])
        _require(folder.parent == storage and folder.name in captured and folder.name not in audited,
                 'Receipt has different/duplicate storage target')
        audited[folder.name] = row
    _require(audited.keys() == captured.keys() and len(captured) == result['captured_frames'],
             'Receipt omitted or added captured rows')
    _require(dict(Counter(r['decision'] for r in audited.values())) == receipt['counts'],
             'Receipt decision counts mismatch')
    basis = _scene_basis(bindings, plan) if captured else None
    records, sequences = [], {}
    for name in sorted(captured):
        row, review = captured[name], audited[name]
        case = planned[name][0]
        folder = storage / name
        meta, label, trace, logical, decoded = _read_sample(bindings, folder, row, review)
        sequences[name] = meta['synchronization']['freshness']['callback_sequence']
        sup = meta['supervision']
        source_family, source_target = _id(plan['source_family']), _id(case['conservative_view_cap_group'])
        _require(source_target.startswith(source_family + '/'), 'Original donor/target mismatch')
        _require(meta['sample_id'] == name and sup['target_id'] == label['target_id']
                 == row['target_id'] == review['target_id'] == case['target_id'], 'Sample target mismatch')
        _require(sup['source_target_id'] == sup['conservative_view_cap_group']
                 == label['conservative_view_cap_group'] == review['source_target_group'] == source_target
                 and sup['split_group'] == label['source_plant_family'] == review['source_family'] == source_family,
                 'Original donor ancestry/cap group mismatch')
        _require(meta['robot_snapshot'] == row['robot_snapshot'] and meta['geometry_screen'] == row['screen']
                 and meta['geometry_screen']['passed'] is True, 'Actual pose/screen differs from capture')
        _require(meta['training_sample_approved'] is False and label['training_approved'] is False
                 and all(review[k] is False for k in ('training_approved', 'source_cap_reset',
                                                      'physical_execution_approved')), 'Approval/cap reset forbidden')
        _require(meta['input_policy'] == dict(clean_full_scene=True, isolation=False, diagnostic_overlays=False),
                 'Full clean background-preserving observation required')
        _require(type(label['eligible']) is bool and (trace is None or type(trace['passed']) is bool),
                 'Boolean label/trace decision required')
        automatic = bool(label['eligible'] and trace['passed'])
        decision = STRICT if automatic else HOLD if label['eligible'] else EXCLUDE
        _require(review['decision'] in DECISIONS and review['decision'] == decision
                 and row['automatic_annotation_eligible'] is automatic
                 and row['eligible_annotation'] is label['eligible'], 'Audit decision mismatch')
        _require(review['label_replayed_exact'] is True
                 and review['review_method'] == 'replayed_anatomy_exact_native_buffers_and_trace'
                 and review['trace_replayed_exact'] is (trace is not None)
                 and review['native_callback_hashes_verified'] is True
                 and review['source_and_file_hashes_verified'] is True
                 and review['reason'] == label['reason']
                 and review['trace_reasons'] == (trace['reasons'] if trace else None)
                 and review['render_profile'] == meta['render_budget']['profile'], 'Unvalidated/stale audit evidence')
        scene, camera = _features(meta, basis)
        scene_sha, camera_sha = _hash(scene), _hash(camera)
        sample_key = _hash(dict(capture=str(capture), candidate_id=name))
        records.append(dict(sample_id=sample_key, candidate_id=name, sample_path=str(folder),
            capture_path=str(capture), target_id=sup['target_id'], variant_target=sup['target_id'],
            source_family=source_family, original_donor_family=source_family,
            source_target=source_target, conservative_view_cap_group=source_target,
            split='train', resolution=[1696, 816], decision=decision,
            context_id=_hash(dict(source_family=source_family, source_target=source_target,
                                 geometry_sha256=basis['generated_geometry_sha256'])),
            geometry_sha256=basis['generated_geometry_sha256'],
            image_bytes=logical['inputs/rgb.png']['bytes'], encoded_rgb_sha256=logical['inputs/rgb.png']['sha256'],
            decoded_rgb_sha256=decoded, decoded_rgb_bytes=1696 * 816 * 3,
            callback_rgb_sha256=meta['synchronization']['freshness']['rgb_sha256'],
            scene_sha256=scene_sha, camera_sha256=camera_sha,
            scene_camera_sha256=_hash(dict(scene=scene_sha, camera=camera_sha)),
            scene_identity_basis=scene, camera_identity_basis=camera,
            annotation_review=dict(passed=automatic, method='automatic', evidence_id=pin.receipt_sha256,
                evidence_basis='pinned_audit_declaration_with_buffer_and_structural_consistency_checks',
                audit_execution_independently_verified=False),
            provenance=dict(receipt_path=str(receipt_path), receipt_sha256=pin.receipt_sha256,
                request_sha256=receipt['request_sha256'], result_sha256=receipt['result_sha256'],
                plan_path=str(plan_path), plan_sha256=receipt['plan_sha256'],
                sample_sha256=row['sample_sha256'], label_sha256=row['label_sha256'],
                logical_files=logical, callback=meta['synchronization'],
                lineage=lineage[case['target_id']], audit_review_method=review['review_method'],
                audit_declarations={k: review[k] for k in ('label_replayed_exact', 'trace_replayed_exact',
                    'native_callback_hashes_verified', 'source_and_file_hashes_verified')},
                trace_evidence_basis='stored_probe_counters_and_metrics_checked_for_internal_consistency_not_replayed'),
            training_approved=False, source_cap_reset=False, background_preserved=True,
            independent_geometry_qualification=False, depth_recomputed=False))
    ordered_sequences = [sequences[name] for name in captured]
    _require(ordered_sequences == sorted(set(ordered_sequences)), 'Stale/repeated callback sequence')
    _require(sum(r['decision'] == STRICT for r in records) == result['automatically_clear_annotation_candidates'],
             'Automatic capture count mismatch')
    return records, dict(receipt_path=str(receipt_path), receipt_sha256=pin.receipt_sha256,
        capture_path=str(capture), storage_root=str(storage), plan_path=str(plan_path),
        planned_rows=len(planned), captured_rows=len(captured), noncaptured_rows=len(planned)-len(captured),
        scene_context_basis=basis)


def _duplicates(records):
    output = {}
    for field in ('decoded_rgb_sha256', 'scene_camera_sha256'):
        groups = defaultdict(list)
        for row in records:
            groups[row[field]].append(row['sample_id'])
        repeated = {key: sorted(ids) for key, ids in sorted(groups.items()) if len(ids) > 1}
        output[field] = dict(groups=repeated, group_count=len(repeated),
            rows_in_duplicate_groups=sum(map(len, repeated.values())),
            duplicate_excess_rows=sum(len(ids)-1 for ids in repeated.values()),
            distinct_values=len(groups), diagnostic_only=True)
    return output


def build_inventory(receipts, *, extra_observed_roots=()):
    """Authenticate explicit receipts and return all rows; NEVER select/admit."""
    try:
        return _build(receipts, extra_observed_roots)
    except (KeyError, TypeError, IndexError, OSError) as exc:
        raise ValueError('Malformed/missing pinned native evidence: ' + str(exc)) from exc


def _build(receipts, extra_roots):
    bindings, records, coverage, scene_contexts = _Bindings(), [], [], {}
    pins = list(receipts)
    extras = []
    for declared in extra_roots:
        _require(isinstance(declared, ObservedRoot), 'Explicit ObservedRoot required')
        root = _path(declared.path)
        _require(root.is_dir() and str(root) not in {r['path'] for r in extras}, 'Missing/duplicate extra root')
        for pin in declared.receipts:
            _require(_path(pin.storage_root or pin.capture_path).is_relative_to(root),
                     'Extra receipt outside declared observed root')
        pins.extend(declared.receipts)
        extras.append(dict(path=str(root), receipt_count=len(declared.receipts),
            status='explicit_receipts_only' if declared.receipts else 'unindexed_no_receipts',
            recursively_scanned=False, complete=False))
    _require(pins, 'At least one explicit pinned audit receipt required')
    seen_receipts, seen_captures = set(), set()
    # Record actual adapter dependencies without importing audit or any renderer.
    implementation = dict(_LOADED_CODE)
    bindings.mapping(implementation)
    _require(all(isinstance(p, ReceiptPin) for p in pins), 'Explicit ReceiptPin required')
    for pin in sorted(pins, key=lambda p: str(p.receipt_path)):
        identity, capture = str(_path(pin.receipt_path)), str(_path(pin.capture_path))
        _require(identity not in seen_receipts and capture not in seen_captures,
                 'Duplicate receipt/capture would double-count rows')
        seen_receipts.add(identity); seen_captures.add(capture)
        rows, source = _receipt(bindings, pin)
        basis = source.pop('scene_context_basis')
        if basis is not None:
            scene_contexts[_hash(basis)] = basis
        records.extend(rows); coverage.append(source)
    records.sort(key=lambda row: row['sample_id'])
    _require(len({r['sample_id'] for r in records}) == len(records), 'Duplicate observation identity')
    bindings.finish()
    result = dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        state='observed_inventory_only_not_global_admission', records=records,
        scene_contexts=scene_contexts,
        counts=dict(captured_rows=len(records), receipts=len(coverage),
            decisions=dict(Counter(r['decision'] for r in records)),
            original_donor_families=dict(Counter(r['source_family'] for r in records)),
            conservative_view_cap_groups=dict(Counter(r['source_target'] for r in records)),
            variant_targets=dict(Counter(r['target_id'] for r in records))),
        exact_duplicates=_duplicates(records),
        scope=dict(explicit_receipts=coverage, extra_observed_roots=extras,
            receipt_discovery_performed=False, capture_roots_scanned=False,
            global_complete=False, observed_split='train', heldout_inventory_included=False,
            callback_validation_scope='RGB/depth/camera fingerprints plus stored renderer/component/target consistency; no independent instance callback fingerprint',
            audit_execution_independently_verified=False, label_derivation_replayed=False,
            trace_derivation_replayed=False, trace_probe_counters_independently_verified=False,
            audit_decision_counts_basis='pinned_declarations_with_row_membership_and_consistency_checks_not_execution_proof',
            near_image_comparison_performed=False, morphology_equivalence_finalized=False),
        provenance=dict(source_bindings=dict(sorted(bindings.hashes.items())),
                        read_only_source_roots=sorted(bindings.protected_roots),
                        implementation_bindings=implementation),
        training_approved=False, source_cap_reset=False, admission_performed=False,
        original_reviews_modified=False, captures_modified=False, images_generated=False,
        depth_recomputed=False, compact_qualification_granted=False)
    result['sha256'] = _hash(result)
    return result


def write_inventory(output, receipts, *, extra_observed_roots=()):
    """Build first, then exclusive-create a disjoint JSON. Parent must exist."""
    output = _path(output)
    _require(not output.exists() and output.parent.is_dir(), 'New output in existing directory required')
    result = build_inventory(receipts, extra_observed_roots=extra_observed_roots)
    protected = [Path(p) for p in result['provenance']['source_bindings']]
    roots = [Path(r[k]) for r in result['scope']['explicit_receipts'] for k in ('capture_path', 'storage_root')]
    roots += [Path(r['receipt_path']).parent for r in result['scope']['explicit_receipts']]
    roots += [Path(r['plan_path']).parent for r in result['scope']['explicit_receipts']]
    roots += [Path(p) for p in result['provenance']['read_only_source_roots']]
    roots += [Path(r['path']) for r in result['scope']['extra_observed_roots']]
    _require(output not in protected and all(not output.is_relative_to(p) for p in roots),
             'Inventory output must be outside captures/reviews/observed roots')
    raw = json.dumps(result, indent=2, allow_nan=False).encode() + b'\n'
    with output.open('xb') as stream:
        stream.write(raw)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', nargs=4, action='append', required=True,
                        metavar=('RECEIPT', 'SHA256', 'CAPTURE', 'PLAN'))
    parser.add_argument('--extra-observed-root', action='append', default=[])
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    result = write_inventory(args.output, [ReceiptPin(*r) for r in args.receipt],
        extra_observed_roots=[ObservedRoot(p) for p in args.extra_observed_root])
    print(json.dumps(dict(counts=result['counts'], sha256=result['sha256'], global_complete=False)))


if __name__ == '__main__':
    main()
