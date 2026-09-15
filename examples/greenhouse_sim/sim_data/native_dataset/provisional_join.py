"""Explicit joined-schema provisional accounting; NEVER masquerades as v1.

select_observed_train(joined_inventory, frozen_splits, pair_coverage, overlays=())
accepts inventory_join.SCHEMA and a fresh complete compact_image_index v2 graph.
Component inventory/attestation metadata and graph-bound RGB/metadata are freshly
checked, not all native arrays or audit execution. Caller must authenticate the
joined packet, graph execution and independent replay authorities externally.
Seals/pins are not self-authenticating execution evidence. No writer, capture,
model, receipt discovery, native audit replay, threshold tuning or approval.

The accounting kernel below is a provenance-labelled copy of frozen
provisional._select, SHA 2aaea0993cbf. Its two changed inputs are explicit derived
physical keys and previously validated near edges. AST tests enforce parity of
the remaining accounting block. Output has its OWN schema/code bindings and
retains the original join/aggregate graph hashes and per-component audit bases.
The inventory is never projected into a fake old-schema receipt.

Physical mapping uses the frozen original_inventory._neutral_basis field map
(SHA 038b943dc976). Recognized legacy generated contexts map to the existing
neutral context; neutral original/generated contexts are validated unchanged.
Renderer-settings provenance is retained but cannot create view diversity.
Background/material/population, target geometry, lighting/renderer/counts,
plant/robot matrices/joints and complete camera calibration/mount remain exact.
There is no tolerance, pose rounding, RGB-noise key, geometry-novelty claim or
name-based fallback. Unsupported/missing/conflicting schemas reject selection.
Derived values are separate; original row/context canonical bytes are untouched.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import math
from pathlib import Path

from . import inventory_join as join, provisional as policy, provisional_compact as graph_adapter
from .admission import _UnionFind, _round_robin, _require, _id, _sha
from ..native_capture_v4.io import Reader

SCHEMA = 'greenhouse.joined_observed_train_provisional_selection.v1'
PHYSICAL_SCHEMA = 'greenhouse.joined_physical_comparison_keys.v1'
SCENE_POLICY = 'greenhouse.neutral_observed_physical_context.v1'
POLICY_SHA256 = '2aaea0993cbfdbd32ae9d161d7a4fa5f2d76c72ff92753721078433b77b69e6a'
JOINER_SHA256 = '3c66d2ead0e79b98e5779e497c91ebd41eca5b715048c05a4a545a1c834c4257'
NEUTRAL_SOURCE_SHA256 = '038b943dc97660bc6a5bf06cec6d474cdd28fcd4ca8c5581a87be1ec9d964028'
STRICT, SOURCE_CAP, TRAIN_GOAL = policy.STRICT, policy.SOURCE_CAP, policy.TRAIN_GOAL
_LOADED_CODE = dict(join.implementation_bindings(), **graph_adapter.implementation_bindings())
_LOADED_CODE.update({str(Path(__file__).resolve()): join._file_sha(__file__),
    str(Path(policy.__file__).resolve()): POLICY_SHA256,
    str(Path(join.__file__).resolve()): JOINER_SHA256,
    str(Path(__file__).with_name('original_inventory.py').resolve()): NEUTRAL_SOURCE_SHA256})


def implementation_bindings():
    """Exact new adapter and frozen policy, mapping, join and graph dependencies."""
    return dict(_LOADED_CODE)


def _canonical(value):
    return join._canonical(value)


def _same(left, right, message):
    _require(_canonical(left) == _canonical(right), message)


def _bound_path(reader, packet, path, expected):
    _require(Path(path).is_absolute(), 'Absolute provenance path required')
    canonical = reader.resolve(path)
    _require(packet['provenance']['source_bindings'].get(str(canonical)) == expected,
             'Component pin missing/conflicting in joined union')
    return canonical


def _joined(reader, packet):
    """Reconstruct components exactly; re-read pinned METADATA, not native arrays."""
    join._sealed(packet)
    _require(packet['schema'] == join.SCHEMA and packet['state'] == join.STATE,
             'Explicit immutable joined inventory required')
    _require(all(packet[k] is False for k in join.FALSE_FLAGS)
             and packet['scope']['global_complete'] is False, 'Unapproved observed join required')
    _same(packet['provenance']['implementation_bindings'], join.implementation_bindings(),
          'Different joined inventory implementation')
    verification = packet['provenance']['verification']
    _require(verification['state'] == 'finished' and verification['stat_only_byte_acceptance'] is False
             and verification['initial_hashes'] == verification['final_hashes'] == verification['unique_files'],
             'Missing original full-union verification declaration')
    views = {r['sample_id']: r for r in packet['records']}
    _require(len(views) == len(packet['records']), 'Repeated joined row ID')
    components = packet['components']
    _require(isinstance(components, list) and components, 'Explicit joined components required')
    covered, context_ids, coverage, audit_bases = set(), set(), [], []
    for component in components:
        pin = component['pin']
        for field in ('external_authority', 'audit_basis', 'attestation_state'):
            _id(pin[field])
        for field, sha_field in (('inventory_path', 'inventory_file_sha256'),
                                 ('attestation_path', 'attestation_sha256')):
            _bound_path(reader, packet, pin[field], pin[sha_field])
        original = reader.json(pin['inventory_path'], pin['inventory_file_sha256'])
        attestation = reader.json(pin['attestation_path'], pin['attestation_sha256'])
        ids = component['row_ids']
        _require(isinstance(ids, list) and len(ids) == len(set(ids))
                 and not covered.intersection(ids) and set(ids) <= views.keys(), 'Component row coverage conflict')
        covered.update(ids)
        keys = component['scene_context_ids']
        _require(isinstance(keys, list) and len(keys) == len(set(keys))
                 and set(keys) <= packet['scene_contexts'].keys(), 'Component context coverage conflict')
        context_ids.update(keys)
        rows = [views[k] for k in ids]
        contexts = {k: packet['scene_contexts'][k] for k in keys}
        _require(join._digest(rows) == component['rows_sha256']
                 and join._digest(contexts) == component['scene_contexts_sha256'], 'Changed component rows/contexts')
        restored = dict(component['inventory_metadata'], records=rows, scene_contexts=contexts)
        _same(restored, original, 'Joined rows/context/claims differ from pinned original inventory')
        join._sealed(original)
        _require(original['schema'] == join.INPUT_SCHEMA and original['state'] == join.INPUT_STATE
                 and all(original[k] is False for k in join.FALSE_FLAGS), 'Unsupported component inventory')
        _require(attestation['state'] == pin['attestation_state'] and attestation['training_approved'] is False
                 and attestation['global_complete'] is False, 'Changed external replay attestation')
        _same(component['attestation_claims'], {k: v for k, v in attestation.items() if k not in join.BINDING_MAPS},
              'Changed external attestation claims')
        _same(component['attestation_binding_maps'], {k: dict(sha256=join._digest(v), entries=len(v))
              for k, v in attestation.items() if k in join.BINDING_MAPS}, 'Changed attestation binding-map metadata')
        found = False
        for prefix in ('inventory', 'observed_inventory'):
            if prefix+'_path' in attestation:
                found = True
                _require(reader.resolve(attestation[prefix+'_path']) == reader.resolve(pin['inventory_path'])
                         and attestation[prefix+'_file_sha256'] == pin['inventory_file_sha256']
                         and attestation[prefix+'_sha256'] == original['sha256'], 'Wrong attested inventory')
        _require(found, 'External attestation does not bind component inventory')
        # Membership only: all source bytes were checked by the external join;
        # resolving aliases again does not claim to have repeated those hashes.
        for source, expected in original['provenance']['source_bindings'].items():
            _bound_path(reader, packet, source, expected)
        for entry in original['scope']['explicit_receipts']:
            failure = reader.resolve(Path(entry['capture_path'])/'failure.json')
            _require(not failure.exists(), 'Capture failure appeared after original attestation')
        coverage.extend(original['scope']['explicit_receipts'])
        audit_bases.append(dict(pin=deepcopy(pin), inventory_sha256=original['sha256'],
            row_ids=deepcopy(ids), rows_sha256=component['rows_sha256'],
            inventory_scope=deepcopy(original['scope']),
            attestation_claims=deepcopy(component['attestation_claims']),
            trust_basis=component['trust_basis'], audit_execution_reverified_here=False))
    _require(covered == views.keys() and context_ids == packet['scene_contexts'].keys(),
             'Joined rows or contexts missing from original components')
    _same(coverage, packet['scope']['explicit_receipts'], 'Changed joined capture coverage')
    _same(join._counts(packet['records'], coverage), packet['counts'], 'Changed joined counts')
    return audit_bases


def _neutral_basis(basis):
    """Exact copied pure field map from frozen original_inventory._neutral_basis."""
    names = ('background_scene_asset_sha256', 'background_asset_content_sha256', 'background_asset_count',
             'material_texture_content_sha256', 'material_texture_count', 'population_placements',
             'excluded_external_prop_roots')
    return dict(schema=SCENE_POLICY, **{k: deepcopy(basis[k]) for k in names},
        target_plant_root=basis['replaced_plant_root'], target_geometry_sha256=basis['generated_geometry_sha256'],
        lighting=deepcopy(basis['original_lighting']), scene_counts=deepcopy(basis['original_scene_counts']),
        renderer=basis['original_renderer'])


_COMMON = {'background_scene_asset_sha256', 'background_asset_content_sha256', 'background_asset_count',
           'material_texture_content_sha256', 'material_texture_count', 'population_placements',
           'excluded_external_prop_roots'}
_LEGACY = _COMMON | {'replaced_plant_root', 'generated_geometry_sha256', 'renderer_settings',
                     'original_lighting', 'original_scene_counts', 'original_renderer'}
_NEUTRAL = _COMMON | {'schema', 'target_plant_root', 'target_geometry_sha256', 'lighting', 'scene_counts', 'renderer'}
_CAMERA = {'resolution', 'intrinsics', 'clipping_range_m', 'focal_length_mm', 'apertures_mm',
           'aperture_offsets_mm', 'depth_convention', 'crop_resize', 'camera_to_world', 'camera_to_head'}


def _numbers(values, length):
    _require(isinstance(values, list) and len(values) == length
             and all(type(x) in (int, float) and math.isfinite(x) for x in values), 'Finite exact physical values required')


def _matrix(value, size):
    _require(isinstance(value, list) and len(value) == size, 'Complete actual matrix required')
    for row in value:
        _numbers(row, size)


def physical_comparison_keys(packet):
    """Metadata-only derived mapping; unsupported evidence REJECTS, never falls back."""
    try:
        return _physical_keys(packet)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError('Unsupported/missing physical comparison evidence: '+str(exc)) from exc


def _physical_keys(packet):
    contexts, keys = {}, {}
    for row in packet['records']:
        scene, camera = row['scene_identity_basis'], row['camera_identity_basis']
        source_key = scene['static_scene_sha256']
        basis = packet['scene_contexts'][source_key]
        _require(join._digest(basis) == source_key and join._digest(scene) == row['scene_sha256']
                 and join._digest(camera) == row['camera_sha256'], 'Changed recorded physical identity')
        if set(basis) == _LEGACY:
            _require(row.get('source_kind') in (None, 'generated_native')
                     and isinstance(basis['renderer_settings'], dict) and basis['renderer_settings'],
                     'Unsupported legacy source/renderer provenance')
            neutral, transform = _neutral_basis(basis), 'generated_plant_to_world'
            source_schema = 'legacy_generated_scene_basis_from_inventory_v1'
        else:
            _require(set(basis) == _NEUTRAL and basis['schema'] == SCENE_POLICY
                     and row.get('source_kind') in ('original_native', 'generated_native'),
                     'Unsupported physical context schema/source kind')
            _require(row['provenance']['scene_identity_policy'] == SCENE_POLICY,
                     'Missing neutral physical provenance')
            neutral, transform, source_schema = deepcopy(basis), 'plant_to_world', SCENE_POLICY
        _require(set(scene) == {'static_scene_sha256', transform, 'actual_robot_root_to_world',
                                'actual_robot_joint_degrees'}, 'Unsupported physical pose fields')
        for name in ('background_scene_asset_sha256', 'background_asset_content_sha256',
                     'material_texture_content_sha256', 'target_geometry_sha256'):
            _sha(neutral[name])
        _require(neutral['target_geometry_sha256'] == row['geometry_sha256'], 'Different target geometry identity')
        for name in ('background_asset_count', 'material_texture_count'):
            _require(type(neutral[name]) is int and neutral[name] > 0, 'Missing physical asset coverage')
        _id(neutral['target_plant_root']); _id(neutral['renderer'])
        _require(isinstance(neutral['lighting'], dict) and neutral['lighting']
                 and isinstance(neutral['scene_counts'], dict) and neutral['scene_counts'], 'Missing full scene/light evidence')
        placements = neutral['population_placements']
        _require(isinstance(placements, list) and placements, 'Missing full population placements')
        seen = set()
        for placement in placements:
            _require(set(placement) == {'plant_root', 'geometry_content_sha256', 'asset_count'}, 'Unsupported placement')
            _id(placement['plant_root']); _sha(placement['geometry_content_sha256'])
            _require(placement['plant_root'] not in seen and type(placement['asset_count']) is int
                     and placement['asset_count'] > 0, 'Missing/duplicate population geometry')
            seen.add(placement['plant_root'])
        _require(neutral['target_plant_root'] in seen, 'Target root missing from complete population')
        excluded = neutral['excluded_external_prop_roots']
        _require(isinstance(excluded, list) and all(isinstance(p, str) and p for p in excluded)
                 and len(excluded) == len(set(excluded)), 'Missing/conflicting excluded-root evidence')
        _matrix(scene[transform], 4); _matrix(scene['actual_robot_root_to_world'], 4)
        joints = scene['actual_robot_joint_degrees']
        _require(isinstance(joints, dict) and joints and all(isinstance(k, str) and k for k in joints),
                 'Actual robot joints required')
        _numbers(list(joints.values()), len(joints))
        _require(set(camera) == _CAMERA and camera['resolution'] == [1696, 816]
                 and camera['crop_resize'] is None
                 and camera['depth_convention'] == 'optical_axis_z_metres_not_ray_range', 'Unsupported actual camera contract')
        _matrix(camera['camera_to_world'], 4); _matrix(camera['camera_to_head'], 4); _matrix(camera['intrinsics'], 3)
        for name in ('clipping_range_m', 'apertures_mm', 'aperture_offsets_mm'):
            _numbers(camera[name], 2)
        _numbers([camera['focal_length_mm']], 1)
        _require(camera['focal_length_mm'] > 0 and all(x > 0 for x in camera['apertures_mm'])
                 and 0 < camera['clipping_range_m'][0] < camera['clipping_range_m'][1], 'Invalid camera optics')
        neutral_sha = join._digest(neutral)
        derived_scene = dict(static_scene_sha256=neutral_sha, plant_to_world=deepcopy(scene[transform]),
            actual_robot_root_to_world=deepcopy(scene['actual_robot_root_to_world']),
            actual_robot_joint_degrees=deepcopy(joints))
        contexts[source_key] = dict(input_schema=source_schema, input_context_sha256=source_key,
            derived_context=neutral, derived_context_sha256=neutral_sha,
            renderer_settings_used_as_view_identity=False)
        _require(row['sample_id'] not in keys, 'Duplicate physical row identity')
        keys[row['sample_id']] = dict(input_scene_sha256=row['scene_sha256'], input_context_sha256=source_key,
            input_camera_sha256=row['camera_sha256'], scene_sha256=join._digest(derived_scene),
            camera_sha256=row['camera_sha256'], derived_scene_basis=derived_scene)
    return dict(schema=PHYSICAL_SCHEMA, contexts=contexts, rows=keys, input_rows_modified=False,
                input_contexts_modified=False, pose_tolerance_applied=False, morphology_equivalence_claimed=False)


def _views(packet, splits):
    _require(isinstance(splits, dict) and all(isinstance(k, str) and k.strip()
             and v in ('train', 'validation', 'test') for k, v in splits.items()), 'Frozen splits required')
    views = {}
    for row in packet['records']:
        key = _id(row['sample_id'])
        _require(key not in views, 'Repeated sample identity')
        family, target = _id(row['source_family']), _id(row['source_target'])
        _id(row['target_id']); _id(row['context_id'])
        _require(target.startswith(family + '/') and len(target) > len(family) + 1
                 and row['conservative_view_cap_group'] == target, 'Original source-cap identity mismatch')
        _require(family in splits and row['split'] == splits[family], 'Frozen donor split mismatch')
        _require(row['resolution'] == [1696, 816] and row['decision'] in policy.DECISIONS
                 and row['training_approved'] is False and row['source_cap_reset'] is False,
                 'Unsafe/unsupported observation')
        for field in ('encoded_rgb_sha256', 'decoded_rgb_sha256', 'scene_sha256', 'camera_sha256'):
            _sha(row[field])
        review = row['annotation_review']
        _require(type(review['passed']) is bool, 'Boolean annotation review required')
        _id(review['evidence_id'])
        views[key] = row
    _require(type(packet['counts']['captured_rows']) is int
             and packet['counts']['captured_rows'] == len(views), 'Inventory count mismatch')
    return views


def _account(views, overlays, near_edges, physical_keys):
    """Copied frozen accounting block; only pose/near-edge inputs differ (AST tested)."""
    aliases = _UnionFind(views)
    rgb_seen, pose_seen = {}, {}
    for key, row in views.items():
        aliases.union(key, rgb_seen.setdefault(row['decoded_rgb_sha256'], key))
        pose = (physical_keys[key]['scene_sha256'], physical_keys[key]['camera_sha256'])
        aliases.union(key, pose_seen.setdefault(pose, key))
    alias_groups = aliases.groups()
    negatives = defaultdict(set)
    for overlay in overlays:
        _require(set(overlay) <= {'sample_id', 'evidence_id', 'visual_hold', 'control_role'}
                 and ('visual_hold' in overlay or 'control_role' in overlay), 'Explicit overlay required')
        key = overlay['sample_id']
        _require(key in views, 'Overlay outside inventory')
        _id(overlay['evidence_id'])
        flags = negatives[alias_groups[key]]
        if 'visual_hold' in overlay:
            _require(type(overlay['visual_hold']) is bool, 'Boolean visual hold required')
            if overlay['visual_hold']:
                flags.add('explicit_visual_hold')
        if 'control_role' in overlay:
            _require(overlay['control_role'] in ('observation', 'control', 'diagnostic', 'replay'),
                     'Unknown control role')
            if overlay['control_role'] != 'observation':
                flags.add('control_role')
    duplicates = _UnionFind(views)
    duplicates.edges((key, aliases.find(key)) for key in views)
    duplicates.edges(near_edges)
    groups = duplicates.groups()
    group_splits = defaultdict(set)
    for key, row in views.items():
        group_splits[groups[key]].add(row['split'])
    tree, reasons = {}, {}
    for key in sorted(views):
        row = views[key]
        reason = sorted(negatives[alias_groups[key]])
        if row['split'] != 'train':
            reason.append('heldout_not_counted')
        if not (row['decision'] == STRICT and row['annotation_review']['passed'] is True
                and row['annotation_review']['method'] == 'automatic'):
            reason.append('strict_annotation_required')
        if len(group_splits[groups[key]]) > 1:
            reason.append('cross_split_view_duplicate')
        reasons[key] = reason
        if not reason:
            tree.setdefault(row['source_family'], {}).setdefault(row['source_target'], []).append(key)

    def schedule(node):
        return iter(node) if isinstance(node, list) else _round_robin(schedule(node[k]) for k in sorted(node))

    selected, used, counts = [], set(), Counter()
    for key in schedule(tree):
        row = views[key]
        source = row['source_target']
        if groups[key] in used:
            reasons[key].append('duplicate_view')
        elif counts[source] >= SOURCE_CAP:
            reasons[key].append('original_source_cap')
        else:
            selected.append(dict(sample_id=key, target_id=row['target_id'], source_family=row['source_family'],
                                 source_target=source, original_context_id=source, split='train', training_approved=False))
            used.add(groups[key]); counts[source] += 1
            if len(selected) == TRAIN_GOAL:
                break
    chosen = {r['sample_id'] for r in selected}
    return dict(selected=selected, excluded={k: reasons[k] or ['train_goal_reached'] for k in sorted(views) if k not in chosen},
                view_duplicate_groups=groups, alias_groups=alias_groups, source_counts=dict(counts))


def select_observed_train(inventory, frozen_splits, pair_coverage, *, overlays=()):
    """Read-only joined-schema accounting with fresh graph RGB/metadata checks."""
    try:
        reader = Reader()
        reader.bindings_from(implementation_bindings())
        packet, inventory_sha = policy._sealed(inventory)
        graph, graph_sha = policy._sealed(pair_coverage)
        audit_bases = _joined(reader, dict(packet, sha256=inventory_sha))
        splits, overlays = deepcopy(frozen_splits), deepcopy(list(overlays))
        views = _views(packet, splits)
        physical = physical_comparison_keys(packet)
        # Frozen v2 aggregate validation, including the genuine base graph seal;
        # the inventory itself is NEVER passed to a v1 selector or relabeled.
        base, graph_views = graph_adapter._projection(packet, inventory_sha, graph)
        _same(graph_views, views, 'Different graph/selection row inventory')
        bound, originals = graph_adapter._sources(packet, graph, views)
        for path in originals:
            reader.resolve(path)
        near_edges = policy._pair_edges(base, inventory_sha, views)
        accounting = _account(views, overlays, near_edges, physical['rows'])
        final_bound, final_originals = graph_adapter._sources(packet, graph, views)
        _require((bound, originals) == (final_bound, final_originals), 'Graph source changed during accounting')
        reader.finish()
        result = dict(schema=SCHEMA, state='joined_observed_train_provisional_candidates_only',
            inventory_sha256=inventory_sha, pair_coverage_sha256=graph_sha,
            frozen_splits_sha256=policy.digest(splits), overlays=overlays, overlays_sha256=policy.digest(overlays),
            **accounting, counts=dict(train=len(accounting['selected'])), train_goal=TRAIN_GOAL,
            train_shortfall=TRAIN_GOAL-len(accounting['selected']),
            policy=dict(max_views_per_original_source=SOURCE_CAP, augmented_contexts_qualified=False,
                metric=policy.METRIC, threshold=policy.CUTOFF, calibration_status='empirical_train_controls_not_globally_validated'),
            physical_key_derivation=physical, component_audit_basis=audit_bases,
            original_rows_sha256=join._digest(packet['records']), original_contexts_sha256=join._digest(packet['scene_contexts']),
            joined_coverage=deepcopy(packet['scope']),
            graph_adapter=dict(source_graph=dict(graph, sha256=graph_sha), validated_base_graph_sha256=base['sha256'],
                               graph_bound_rgb_metadata_reverified=True, pair_measurements_recomputed=False),
            implementation_bindings=implementation_bindings(),
            accounting_origin=dict(source_file=str(Path(policy.__file__).resolve()), source_sha256=POLICY_SHA256,
                source_function='_select', changes=['explicit_derived_physical_keys', 'prevalidated_near_edges']),
            scope=dict(pair_coverage_accounting_complete_for_supplied_rows=True, global_complete=False, heldout_counted=False,
                pair_measurements_recomputed=False, pruning_bounds_independently_verified=False,
                source_files_reverified=False, audit_execution_independently_verified=False,
                geometry_equivalence_finalized=False, prior_observations_discovered=False,
                component_metadata_reverified=True, graph_bound_rgb_metadata_reverified=True,
                all_native_arrays_reverified=False, native_audits_replayed=False),
            metadata_verification=reader.diagnostics(), calibration_validated=False,
            training_approved=False, source_cap_reset=False, admission_performed=False,
            original_reviews_modified=False, existing_selections_modified=False)
        result['sha256'] = policy.digest(result)
        return result
    except (KeyError, TypeError, IndexError, AttributeError, OSError) as exc:
        raise ValueError('Malformed joined provisional evidence: '+str(exc)) from exc
