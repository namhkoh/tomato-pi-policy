"""Pure, evidence-relative representative proposal; NEVER budget/admission.

pack_representatives(snapshot, pairs, controls, reviews, contribution_plan,
                     *, policy) accepts PinnedJSON for all six documents.
Expected byte pins and the loaded kernel/metric code must be authenticated by
the caller. Pins/qualification declarations do NOT prove native execution,
calibration or visual-review truth. No files, native APIs, clock or randomness.

Schemas below deliberately do not impersonate inventory/admission schemas.
Snapshot contains neutral contexts/samples, original pool IDs, frozen 16/4/4
splits and explicit overlays. Pairs includes every geometry pair and an external
complete near-image attestation. Resolved geometry scores are recomputed with
the unchanged set metric; unknowns never establish separation. Exact geometry
identity is distinct from zero feature distance and near-edge connectivity.

Controls must explicitly qualify the metric/cutoff and name native, visual,
nuisance, calibration and validation evidence. Reviews bind actual native
sample/audit/RGB and descriptor identities. Their truth remains external.

The contribution plan fixes baseline sample IDs and ALL proposed samples for
each candidate context. Each retained representative is checked against every
actual contributing context (including nonrepresentative original-pool bridges)
and every contributing source original. A new source original must also be far
from earlier representatives. Samples of a representative itself are the only
self-comparison exception; no alias can occupy a second proposal role.

Order is frozen family/target/context lexical order. Greedy is maximal ONLY
relative to these inputs/constraints, NOT maximum. All old pools are returned
unchanged, including for rejected/unknown contexts. No quota, cap checking,
biological-family novelty, entitlements, release, or policy adoption is implied.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import math

from .admission import _UnionFind, _digest, _id, _require, _sha
from . import morphology_set_v1 as metric

VERSION = 'representative_packing.v1'
SCHEMAS = {k: 'greenhouse.representative_packing.' + k + '.v1' for k in
           ('snapshot', 'pairs', 'controls', 'reviews', 'contribution_plan', 'policy', 'result')}
ORDER = 'source_family_source_target_context_id.lexical.v1'
METRIC_SHA256 = '454dda7127ac905b9dcacd0f3aca6833186314dda8bf18d3f973e99f5bf78814'
FRAME_SHA256 = '7efb31bd80d8f84d931e5bd7c5807cca33bf03b35a51eeac8e11987126f4e555'
STRICT = 'accept_strict_automatic_annotation_candidate'
DECISIONS = {STRICT, 'hold_visual_clarity', 'exclude_geometry_or_visibility'}
PROOF_ROLES = {'calibration', 'native_controls', 'visual_controls', 'nuisance_controls', 'validation_controls'}
REVIEW_ROLES = {'native_plan', 'native_result', 'native_audit', 'visual_review', 'generated_rgb'}


@dataclass(frozen=True)
class PinnedJSON:
    data: bytes
    sha256: str

    def read(self):
        _require(type(self.data) is bytes and sha256(self.data).hexdigest() == _sha(self.sha256),
                 'Changed pinned JSON bytes')
        def unique(items):
            result = {}
            for key, value in items:
                _require(key not in result, 'Duplicate JSON key')
                result[key] = value
            return result
        value = json.loads(self.data, object_pairs_hook=unique)
        _require(isinstance(value, dict), 'JSON object required')
        _digest(value)  # Nonfinite values anywhere are invalid.
        return value


def _fields(value, names):
    _require(isinstance(value, dict) and set(value) == set(names.split()), 'Unexpected/missing fields: ' + names)


def _ids(values, available):
    _require(isinstance(values, list) and all(isinstance(k, str) for k in values)
             and len(set(values)) == len(values) and set(values) <= set(available), 'Unknown/duplicate IDs')
    return sorted(values)


def _proofs(values, roles):
    _require(isinstance(values, dict) and set(values) == roles, 'Complete proof roles required')
    for value in values.values():
        _fields(value, 'evidence_id sha256')
        _id(value['evidence_id']); _sha(value['sha256'])


def _snapshot(value):
    _fields(value, 'schema inventory_sha256 contexts samples frozen_splits overlays training_approved source_cap_reset')
    _sha(value['inventory_sha256'])
    _require(value['training_approved'] is False and value['source_cap_reset'] is False, 'Unsafe snapshot approval')
    splits = value['frozen_splits']
    _require(isinstance(splits, dict) and Counter(splits.values()) == Counter(train=16, validation=4, test=4),
             'Frozen original 16/4/4 splits required')
    for family in splits: _id(family)
    contexts, samples = value['contexts'], value['samples']
    _require(isinstance(contexts, dict) and contexts and isinstance(samples, dict), 'Context/sample maps required')
    geometry_aliases = _UnionFind(contexts); geometries = {}; originals = {}
    for key, row in contexts.items():
        _id(key)
        _fields(row, 'kind source_family source_target source_context_id old_pool_id geometry_identity_sha256 frame_reference_sha256 descriptor descriptor_sha256')
        family, target, source = row['source_family'], row['source_target'], row['source_context_id']
        _require(family in splits and isinstance(target, str) and target.startswith(family + '/')
                 and len(target) > len(family)+1 and source in contexts, 'Context ancestry differs')
        _id(row['old_pool_id']); _sha(row['geometry_identity_sha256']); _sha(row['descriptor_sha256'])
        _sha(row['frame_reference_sha256'])
        _require(row['descriptor_sha256'] == _digest(row['descriptor']), 'Descriptor seal differs')
        if row['descriptor'] is not None: metric.validate(row['descriptor'])
        original = contexts[source]
        _require(original['kind'] == 'original' and original['source_context_id'] == source
                 and all(row[k] == original[k] for k in ('source_family', 'source_target', 'old_pool_id', 'frame_reference_sha256')),
                 'Original ancestry/pool changed')
        _require(row['kind'] in ('original', 'generated') and (row['kind'] == 'original') == (key == source),
                 'Original/generated identity differs')
        if key == source:
            _require(originals.setdefault(target, key) == key, 'Ambiguous original target')
        geometry_aliases.union(key, geometries.setdefault(row['geometry_identity_sha256'], key))
    aliases = _UnionFind(samples); rgb_seen, pose_seen = {}, {}
    for key, row in samples.items():
        _id(key)
        _fields(row, 'context_id source_family source_target split decoded_rgb_sha256 scene_sha256 camera_sha256 sample_sha256 encoded_rgb_sha256 audit_sha256 annotation_review decision')
        _require(row['context_id'] in contexts, 'Unknown sample context')
        context = contexts[row['context_id']]
        _require(all(row[k] == context[k] for k in ('source_family', 'source_target'))
                 and row['split'] == splits[row['source_family']], 'Sample ancestry/split differs')
        for field in ('decoded_rgb_sha256', 'scene_sha256', 'camera_sha256', 'sample_sha256', 'encoded_rgb_sha256', 'audit_sha256'):
            _sha(row[field])
        review = row['annotation_review']; _fields(review, 'passed method evidence_id')
        _require(type(review['passed']) is bool and review['method'] in ('automatic', 'manual')
                 and row['decision'] in DECISIONS, 'Invalid annotation declaration')
        _id(review['evidence_id'])
        aliases.union(key, rgb_seen.setdefault(row['decoded_rgb_sha256'], key))
        pose = (row['scene_sha256'], row['camera_sha256'])
        aliases.union(key, pose_seen.setdefault(pose, key))
    return contexts, samples, geometry_aliases.groups(), aliases.groups()


def _pairs(value, snapshot_pin, contexts, samples, geometry_aliases):
    _fields(value, 'schema snapshot_sha256 metric metric_implementation_sha256 frame_implementation_sha256 records view_graph')
    _require(value['snapshot_sha256'] == snapshot_pin and value['metric'] == metric.METRIC
             and value['metric_implementation_sha256'] == METRIC_SHA256
             and value['frame_implementation_sha256'] == FRAME_SHA256, 'Pair scope/metric binding differs')
    table = {}; unknown = 0
    _require(isinstance(value['records'], list), 'Explicit pair records required')
    for row in value['records']:
        _require(isinstance(row, dict) and row.get('state') in ('resolved', 'unknown'), 'Pair state required')
        _fields(row, 'left right state ' + ('distance' if row['state'] == 'resolved' else 'reason'))
        a, b = row['left'], row['right']
        _require(a in contexts and b in contexts and a != b, 'Pair outside context inventory')
        pair = tuple(sorted((a, b))); _require(pair not in table, 'Repeated/conflicting pair')
        if row['state'] == 'resolved':
            score = row['distance']; x, y = contexts[a]['descriptor'], contexts[b]['descriptor']
            _require(type(score) in (int, float) and math.isfinite(score) and score >= 0,
                     'Finite nonnegative pair distance required')
            _require(x is not None and y is not None and x['leaf_count'] > 0 and y['leaf_count'] > 0,
                     'Missing/empty descriptors cannot resolve a pair')
            _require(metric.distance(x, y) == score, 'Recomputed pair distance differs')
            _require(geometry_aliases[a] != geometry_aliases[b] or score == 0, 'Exact geometry alias contradicts metric')
        else:
            _id(row['reason']); unknown += 1
        table[pair] = row
    _require(len(table) == len(contexts)*(len(contexts)-1)//2, 'Incomplete global geometry pairs')
    graph = value['view_graph']
    _fields(graph, 'inventory_sha256 evidence_id sha256 external_authority complete sample_ids total_pairs resolved_pairs near_edges')
    _sha(graph['inventory_sha256']); _sha(graph['sha256']); _id(graph['evidence_id']); _id(graph['external_authority'])
    _require(graph['complete'] is True and set(_ids(graph['sample_ids'], samples)) == set(samples)
             and type(graph['total_pairs']) is int and type(graph['resolved_pairs']) is int
             and graph['total_pairs'] == graph['resolved_pairs'] == len(samples)*(len(samples)-1)//2,
             'Incomplete externally verified global view coverage')
    seen = set()
    _require(isinstance(graph['near_edges'], list), 'Explicit view edges required')
    for edge in graph['near_edges']:
        _require(isinstance(edge, list) and len(edge) == 2 and all(k in samples for k in edge)
                 and edge[0] != edge[1], 'Invalid view edge')
        pair = tuple(sorted(edge)); _require(pair not in seen, 'Repeated view edge'); seen.add(pair)
    return table, unknown


def _reviews(value, pins, contexts, samples):
    _fields(value, 'schema snapshot_sha256 pairs_sha256 metric contexts')
    _require(value['snapshot_sha256'] == pins['snapshot'] and value['pairs_sha256'] == pins['pairs']
             and value['metric'] == metric.METRIC and isinstance(value['contexts'], list), 'Review scope differs')
    rows = {}; holds = []
    for row in value['contexts']:
        _fields(row, 'context_id source_context_id sample_id descriptor_sha256 source_descriptor_sha256 sample_sha256 decision artifacts')
        key, sample = row['context_id'], row['sample_id']
        _require(key in contexts and key not in rows and sample in samples and samples[sample]['context_id'] == key,
                 'Unknown/duplicate review context or witness')
        context = contexts[key]; source = context['source_context_id']
        _require(key != source and row['source_context_id'] == source
                 and row['descriptor_sha256'] == context['descriptor_sha256']
                 and row['source_descriptor_sha256'] == contexts[source]['descriptor_sha256']
                 and row['sample_sha256'] == samples[sample]['sample_sha256'], 'Review native/descriptor joins differ')
        _require(row['decision'] in ('geometry_distinct_plausible', 'hold', 'novelty_withheld'), 'Unknown geometry review')
        _proofs(row['artifacts'], REVIEW_ROLES)
        _require(row['artifacts']['native_audit']['sha256'] == samples[sample]['audit_sha256']
                 and row['artifacts']['generated_rgb']['sha256'] == samples[sample]['encoded_rgb_sha256'],
                 'Native review audit/RGB differs')
        rows[key] = row
        if row['decision'] == 'hold':
            holds.append(dict(sample_id=sample, evidence_id=pins['reviews'], visual_hold=True))
    return rows, holds


def _sample_flags(samples, aliases, overlays, near_edges):
    flags = defaultdict(set); _require(isinstance(overlays, list), 'Explicit overlay list required')
    for overlay in overlays:
        _require(isinstance(overlay, dict) and set(overlay) <= {'sample_id', 'evidence_id', 'visual_hold', 'control_role'}
                 and ('visual_hold' in overlay or 'control_role' in overlay), 'Explicit overlay required')
        key = overlay['sample_id']; _require(key in samples, 'Overlay outside snapshot'); _id(overlay['evidence_id'])
        if 'visual_hold' in overlay:
            _require(type(overlay['visual_hold']) is bool, 'Boolean visual hold required')
            if overlay['visual_hold']: flags[aliases[key]].add('explicit_visual_hold')
        if 'control_role' in overlay:
            _require(overlay['control_role'] in ('observation', 'control', 'diagnostic', 'replay'), 'Unknown control role')
            if overlay['control_role'] != 'observation': flags[aliases[key]].add('control_role')
    union = _UnionFind(samples); union.edges(near_edges); leaders = {}
    for key, group in aliases.items():
        union.union(key, leaders.setdefault(group, key))
    groups = union.groups(); splits = defaultdict(set)
    for key, row in samples.items(): splits[groups[key]].add(row['split'])
    result = {}
    for key, row in samples.items():
        reasons = set(flags[aliases[key]])
        if row['split'] != 'train': reasons.add('heldout_not_counted')
        if len(splits[groups[key]]) > 1: reasons.add('cross_split_view_duplicate')
        if not (row['decision'] == STRICT and row['annotation_review']['passed'] is True
                and row['annotation_review']['method'] == 'automatic'): reasons.add('strict_annotation_required')
        result[key] = sorted(reasons)
    return result, groups


def pack_representatives(snapshot, pairs, controls, reviews, contribution_plan, *, policy):
    """Detached evidence-relative proposal. All six arguments are PinnedJSON."""
    try:
        return _pack(snapshot, pairs, controls, reviews, contribution_plan, policy)
    except (KeyError, TypeError, IndexError, AttributeError, OverflowError) as exc:
        raise ValueError('Malformed packing evidence: ' + str(exc)) from exc


def _pack(snapshot, pairs, controls, reviews, contribution_plan, policy):
    args = dict(snapshot=snapshot, pairs=pairs, controls=controls, reviews=reviews,
                contribution_plan=contribution_plan, policy=policy)
    _require(all(isinstance(v, PinnedJSON) for v in args.values()), 'Six explicitly pinned documents required')
    pins = {k: v.sha256 for k, v in args.items()}; docs = {k: v.read() for k, v in args.items()}
    for k, value in docs.items(): _require(value['schema'] == SCHEMAS[k], 'Wrong versioned ' + k + ' schema')
    snap, pair_doc, control, review_doc, plan, pol = (docs[k] for k in args)
    contexts, samples, geometry_aliases, aliases = _snapshot(snap)
    table, unknown = _pairs(pair_doc, pins['snapshot'], contexts, samples, geometry_aliases)
    _require(pair_doc['view_graph']['inventory_sha256'] == snap['inventory_sha256'], 'View inventory binding differs')
    review_rows, review_holds = _reviews(review_doc, pins, contexts, samples)
    _fields(control, 'schema snapshot_sha256 pairs_sha256 reviews_sha256 metric threshold status external_authority qualified_context_ids evidence')
    _require(all(control[k+'_sha256'] == pins[k] for k in ('snapshot', 'pairs', 'reviews'))
             and control['metric'] == metric.METRIC and control['status'] == 'externally_qualified_controls',
             'Explicit externally qualified controls required')
    _id(control['external_authority']); _proofs(control['evidence'], PROOF_ROLES)
    evidence_pins = {}
    refs = list(control['evidence'].values()) + [pair_doc['view_graph']]
    refs += [ref for row in review_rows.values() for ref in row['artifacts'].values()]
    for ref in refs:
        name = ref['evidence_id'].replace('\\', '/')
        _require(evidence_pins.setdefault(name, ref['sha256']) == ref['sha256'], 'Conflicting evidence pins')
    qualified = set(_ids(control['qualified_context_ids'], contexts))
    _require(qualified and all(k in review_rows and review_rows[k]['decision'] == 'geometry_distinct_plausible'
                              for k in qualified), 'Native visual qualification required for exact context scope')
    _fields(pol, 'schema enabled version ordering metric metric_implementation_sha256 frame_implementation_sha256 threshold controls_sha256 frozen_splits_sha256')
    _require(pol['enabled'] is True and pol['version'] == VERSION and pol['ordering'] == ORDER
             and pol['metric'] == metric.METRIC and pol['metric_implementation_sha256'] == METRIC_SHA256
             and pol['frame_implementation_sha256'] == FRAME_SHA256
             and pol['controls_sha256'] == pins['controls']
             and pol['frozen_splits_sha256'] == _digest(snap['frozen_splits']), 'Explicit frozen opt-in policy required')
    threshold = pol['threshold']
    _require(type(threshold) in (int, float) and math.isfinite(threshold) and threshold >= 0
             and type(control['threshold']) in (int, float) and control['threshold'] == threshold,
             'Explicit calibrated finite threshold required; no default')
    _fields(plan, 'schema snapshot_sha256 pairs_sha256 reviews_sha256 policy_sha256 baseline_sample_ids candidate_samples_by_context')
    _require(all(plan[k+'_sha256'] == pins[k] for k in ('snapshot', 'pairs', 'reviews', 'policy')), 'Contribution scope differs')
    baseline = _ids(plan['baseline_sample_ids'], samples); candidates = plan['candidate_samples_by_context']
    _require(isinstance(candidates, dict) and set(candidates) <= set(contexts), 'Explicit candidate map required')
    proposed = set(baseline); candidate_samples = {}
    for key, ids in candidates.items():
        ids = _ids(ids, samples)
        _require(contexts[key]['kind'] == 'generated' and ids and all(samples[s]['context_id'] == key for s in ids)
                 and not proposed.intersection(ids), 'Wrong/empty/repeated candidate contribution')
        proposed.update(ids); candidate_samples[key] = ids
    flags, view_groups = _sample_flags(samples, aliases, snap['overlays'] + review_holds, pair_doc['view_graph']['near_edges'])
    geometry_splits = defaultdict(set)
    for key, context in contexts.items():
        geometry_splits[geometry_aliases[key]].add(snap['frozen_splits'][context['source_family']])
    for key, row in samples.items():
        if len(geometry_splits[geometry_aliases[row['context_id']]]) > 1:
            flags[key] = sorted(set(flags[key]) | {'cross_split_exact_geometry_alias'})
    _require(all(not flags[s] for s in baseline), 'Invalid baseline contribution; never silently remove holds/splits')
    _require(len({view_groups[s] for s in baseline}) == len(baseline), 'Duplicate baseline view contributions')
    actual = {samples[s]['context_id'] for s in baseline}
    originals = {contexts[k]['source_context_id'] for k in actual}
    selected = []; selected_samples = list(baseline); used_views = {view_groups[s] for s in baseline}
    rejected, checks = {}, {}
    order = sorted(candidates, key=lambda k: (contexts[k]['source_family'], contexts[k]['source_target'], k))

    def check(a, b, role):
        pair = tuple(sorted((a, b)))
        if a == b or geometry_aliases[a] == geometry_aliases[b]:
            return dict(left=a, right=b, role=role, state='blocked', reason='exact_geometry_alias')
        evidence = table[pair]
        if evidence['state'] == 'unknown':
            return dict(left=a, right=b, role=role, state='blocked', reason='unknown_pair', detail=evidence['reason'])
        return dict(left=a, right=b, role=role, state='separated' if evidence['distance'] > threshold else 'blocked',
                    reason='above_threshold' if evidence['distance'] > threshold else 'near_pair', distance=evidence['distance'])

    for key in order:
        ids = candidate_samples[key]; reasons = sorted({reason for s in ids for reason in flags[s]})
        if key not in qualified: reasons.append('native_visual_qualification_required')
        else:
            witness_flags = set(flags[review_rows[key]['sample_id']]) - {'control_role'}
            if witness_flags: reasons.append('invalid_native_visual_witness')
        candidate_views = {view_groups[s] for s in ids}
        if len(candidate_views) != len(ids) or candidate_views & used_views: reasons.append('duplicate_view')
        if any(geometry_aliases[key] == geometry_aliases[k] for k in contexts if contexts[k]['kind'] == 'original'):
            reasons.append('exact_original_geometry_alias')
        source = contexts[key]['source_context_id']
        witnesses = [check(key, other, 'actual_contributor') for other in sorted(actual)]
        witnesses += [check(key, other, 'source_original') for other in sorted(originals | {source})]
        # The new candidate can bring an original not previously contributing.
        witnesses += [check(other, source, 'new_source_original') for other in selected if source not in originals]
        checks[key] = witnesses
        reasons += [w['reason'] for w in witnesses if w['state'] == 'blocked']
        if reasons:
            rejected[key] = sorted(set(reasons)); continue
        selected.append(key); selected_samples.extend(ids); actual.add(key); originals.add(source); used_views.update(candidate_views)

    # Independent final invariant, not merely a record of greedy-loop checks.
    final_checks = []
    for key in selected:
        final_checks += [check(key, other, 'final_actual_contributor') for other in sorted(actual - {key})]
        final_checks += [check(key, other, 'final_source_original') for other in sorted(originals)]
    _require(all(w['state'] == 'separated' for w in final_checks), 'Final contributing-context spacing failed')
    result = dict(schema=SCHEMAS['result'], version=VERSION, state='evidence_relative_packing_proposal_only',
        evidence_relative=True, input_bindings=pins, metric=metric.METRIC, threshold=threshold, ordering=ORDER,
        candidate_order=order, proposed_representative_ids=selected, rejected=rejected,
        candidate_pair_checks=checks, final_pair_checks=final_checks,
        baseline_sample_ids=baseline, contributing_sample_ids=sorted(selected_samples),
        contributing_context_ids=sorted(actual), contributing_source_original_ids=sorted(originals),
        old_pool_by_context={k: contexts[k]['old_pool_id'] for k in sorted(contexts)},
        geometry_alias_groups=geometry_aliases, sample_alias_groups=aliases, view_duplicate_groups=view_groups,
        frozen_splits=deepcopy(snap['frozen_splits']), overlays=deepcopy(snap['overlays']),
        review_hold_overlays=review_holds, sample_exclusions=flags,
        coverage=dict(contexts=len(contexts), total_pairs=len(table), resolved_pairs=len(table)-unknown,
                      unknown_pairs=unknown, pair_membership_complete=True, distances_complete=unknown == 0,
                      view_coverage_basis='externally_pinned_attestation_not_reexecution'),
        external_authority=control['external_authority'], maximal_relative_to_declared_constraints=True,
        maximum_claimed=False, biological_family_novelty_claimed=False,
        baseline_unchanged=True, baseline_geometry_spacing_requalified=False,
        context_entitlements_granted=False, source_cap_reset=False, caps_checked=False,
        training_approved=False, policy_adopted=False, global_collection_complete=False,
        native_execution_verified=False, visual_review_truth_verified=False, calibration_reexecuted=False,
        source_files_reverified=False, implementation_files_reverified=False)
    result['sha256'] = _digest(result)
    return result
