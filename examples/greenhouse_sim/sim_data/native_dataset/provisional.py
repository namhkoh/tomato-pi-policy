"""Pure observed-TRAIN candidate accounting; no I/O, audit replay or approval.

select_observed_train(inventory, frozen_splits, pair_coverage, *, overlays=())
consumes a sealed inventory.build_inventory result. The caller authenticates
its files/receipts; an internal SHA256 is NOT evidence of independent execution.

The index adapter supplies near_image_index.build_graph's sealed mapping.
Its sorted sample_ids, image_pins, inventory hash, metric/cutoff and 129x129
nominal-cut patch definition must agree with this inventory. Both resolved and
exact+pruned counts must equal N*(N-1)//2, complete=True. Missing accounting,
invalid edges and conflicting RGB bindings fail. Declared pair execution and
pruning bounds remain the caller's responsibility; no images/pruning run here.
Seal mappings with value['sha256']=digest(value) before adding that field.

Overlays are {sample_id, evidence_id, visual_hold?: bool, control_role?: str}.
Roles are observation/control/diagnostic/replay. Negative overlays propagate
across exact RGB OR scene-camera aliases, NOT arbitrary near-image neighbours.
They cannot clear existing exclusions. All rows, including controls, failed
reviews and any known heldout rows, remain in the transitive duplicate graph.
All augmented contexts share their original source-target's 12-view budget.
There is no geometry novelty claim, heldout quota or globally validated cutoff.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import math

from .admission import _UnionFind, _digest as digest, _id, _require, _round_robin, _sha

SCHEMA = 'greenhouse.observed_train_provisional_selection.v1'
PAIR_SCHEMA = 'greenhouse.native_near_image_graph.v1'
METRIC = 'max_full_and_junction_patch_rgb_mae_0to1_v1'
CUTOFF = 0.0391332848665376
TRAIN_GOAL, SOURCE_CAP = 20000, 12
STRICT = 'accept_strict_automatic_annotation_candidate'
DECISIONS = {STRICT, 'hold_visual_clarity', 'exclude_geometry_or_visibility'}


def _sealed(value):
    snapshot = deepcopy(value)
    expected = _sha(snapshot.pop('sha256'))
    _require(digest(snapshot) == expected, 'Stale sealed evidence')
    return snapshot, expected


def _pair_edges(receipt, inventory_sha256, views):
    fields = dict(schema=PAIR_SCHEMA, inventory_sha256=inventory_sha256, metric=METRIC,
                  threshold=CUTOFF, dimensions=[1696, 816], patch_radius=64,
                  patch_anchor='nominal_10mm_cut_point_not_anatomical_attachment')
    _require(all(receipt[k] == v for k, v in fields.items()), 'Different pair coverage contract')
    _require(receipt['sample_ids'] == sorted(views), 'Different pair coverage sample inventory')
    _require(all(receipt[k] is False for k in ('global_collection_complete', 'threshold_calibrated',
                                             'training_approved')), 'Unapproved observed index required')
    counts = receipt['counts']
    names = ('total_pairs', 'exact_compared_pairs', 'bound_pruned_pairs', 'resolved_pairs')
    _require(all(type(counts[k]) is int and counts[k] >= 0 for k in names), 'Integer pair counts required')
    total = len(views) * (len(views) - 1) // 2
    _require(receipt['complete'] is True and counts['total_pairs'] == counts['resolved_pairs'] == total
             and counts['exact_compared_pairs'] + counts['bound_pruned_pairs'] == total,
             'Missing pair comparisons or inconsistent coverage accounting')
    for field in ('input_bindings', 'implementation_bindings'):
        _require(isinstance(receipt[field], dict) and receipt[field], 'Nonempty index bindings required')
        for name, sha in receipt[field].items():
            _id(name); _sha(sha)
    _require(isinstance(receipt['image_pins'], list)
             and [p['sample_id'] for p in receipt['image_pins']] == sorted(views), 'Incomplete image pin inventory')
    for pin in receipt['image_pins']:
        row = views[pin['sample_id']]
        _require(pin['decoded_rgb_sha256'] == row['decoded_rgb_sha256']
                 and pin['sha256'] == row['encoded_rgb_sha256']
                 and receipt['input_bindings'].get(pin['path']) == pin['sha256'], 'Index/inventory RGB mismatch')
        uv = pin['nominal_uv']
        _require(isinstance(uv, (list, tuple)) and len(uv) == 2
                 and all(type(v) in (int, float) and math.isfinite(v) for v in uv)
                 and 64 <= uv[0] < 1696-64 and 64 <= uv[1] < 816-64, 'Complete nominal-cut patch required')
    edges, scores = receipt['near_image_edges'], receipt['edge_scores']
    _require(isinstance(edges, list) and isinstance(scores, list) and len(edges) == len(scores),
             'Explicit scored near-image edges required')
    seen = set()
    for pair, score in zip(edges, scores):
        _require(isinstance(pair, (list, tuple)) and len(pair) == 2, 'Pair edge required')
        left, right = pair
        _require(left in views and right in views and left != right, 'Unknown/self pair endpoint')
        key = tuple(sorted((left, right)))
        _require(key not in seen, 'Repeated near-image edge')
        seen.add(key)
        _require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= CUTOFF,
                 'Invalid near-edge score')
    _require(len(seen) <= counts['exact_compared_pairs'], 'Near edges exceed exact comparisons')
    return sorted(seen)


def select_observed_train(inventory, frozen_splits, pair_coverage, *, overlays=()):
    """Return detached candidates/reasons, not a release or completeness proof.

Only supplied observed rows count. Supply the cumulative inventory, not a new
batch alone; this function never discovers prior selections or missing roots.
Pair scores/bounds are trusted index declarations, NOT recomputed here.
"""
    try:
        return _select(inventory, frozen_splits, pair_coverage, overlays)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError('Malformed provisional adapter evidence: ' + str(exc)) from exc


def _select(inventory, frozen_splits, pair_coverage, overlays):
    packet, inventory_sha = _sealed(inventory)
    coverage, coverage_sha = _sealed(pair_coverage)
    splits, overlays = deepcopy(frozen_splits), deepcopy(list(overlays))
    _require(packet['schema'] == 'greenhouse.pinned_native_observed_inventory.v1'
             and packet['state'] == 'observed_inventory_only_not_global_admission'
             and packet['training_approved'] is False and packet['source_cap_reset'] is False,
             'Unapproved observed inventory required')
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
        _require(row['resolution'] == [1696, 816] and row['decision'] in DECISIONS
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
    aliases = _UnionFind(views)
    rgb_seen, pose_seen = {}, {}
    for key, row in views.items():
        aliases.union(key, rgb_seen.setdefault(row['decoded_rgb_sha256'], key))
        pose = (row['scene_sha256'], row['camera_sha256'])
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
    duplicates.edges(_pair_edges(coverage, inventory_sha, views))
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
    result = dict(schema=SCHEMA, state='observed_train_provisional_candidates_only',
        inventory_sha256=inventory_sha, pair_coverage_sha256=coverage_sha,
        frozen_splits_sha256=digest(splits), overlays=overlays, overlays_sha256=digest(overlays),
        selected=selected, excluded={k: reasons[k] or ['train_goal_reached'] for k in sorted(views) if k not in chosen},
        view_duplicate_groups=groups, alias_groups=alias_groups, source_counts=dict(counts),
        counts=dict(train=len(selected)), train_goal=TRAIN_GOAL, train_shortfall=TRAIN_GOAL-len(selected),
        policy=dict(max_views_per_original_source=SOURCE_CAP, augmented_contexts_qualified=False,
                    metric=METRIC, threshold=CUTOFF, calibration_status='empirical_train_controls_not_globally_validated'),
        scope=dict(pair_coverage_accounting_complete_for_supplied_rows=True, global_complete=False, heldout_counted=False,
                   pair_measurements_recomputed=False, pruning_bounds_independently_verified=False,
                   source_files_reverified=False, audit_execution_independently_verified=False,
                   geometry_equivalence_finalized=False, prior_observations_discovered=False),
        calibration_validated=False, training_approved=False, source_cap_reset=False,
        admission_performed=False, original_reviews_modified=False)
    result['sha256'] = digest(result)
    return result
