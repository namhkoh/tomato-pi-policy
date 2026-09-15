"""Pure observed-context adapter, not calibration, native proof or a selector.

adapt_contexts consumes PinnedJSON bytes for inventory, the ACTUAL observed
morphology report, its extracted packets (mapping exact report path -> pin),
and the existing image graph. No files, USD, capture, or source mutation.
Expected byte hashes must come from the caller's trusted evidence registry;
a self-issued hash cannot authenticate execution or a human review.

Without controls, all variants retain their original 12-view pool. Optional
controls are an EXTERNAL, pinned qualification for exact context IDs, not a
threshold fitted here. They bind this report, graph, reviews, nuisance report
and frozen splits; carry metric, threshold, context_ids, positive_context_ids,
and qualification_status='externally_qualified_exact_contexts'. No defaults.

Reviews use schema REVIEW_SCHEMA, bindings, overlays (provisional's format),
and contexts. Each context entry binds descriptor + original descriptor,
sample/encoded RGB/audit/qualification hashes and named native artifacts.
Only positive controls require the original/generated paired artifacts.
Other exact contexts can use their own native plan/result/audit/RGB and visual
review, decision='geometry_distinct_plausible', without rerendering an original.
The caller must authenticate those artifacts and their native review:
this adapter checks declared bindings, NOT their execution or visual truth.

controls.metric selects bottleneck v2 or morphology_set_v1.METRIC explicitly.
Bottleneck retains its old nuisance contract; mixed cardinalities are unknown
and their conservative merge edges CAN COLLAPSE THE ENTIRE INVENTORY to one
pool. Set mode compares all nonempty cardinalities, but requires a NEW sealed
set-nuisance replay, matching metric_proof_sha256, and metric-bound review
distances. No old cutoff or old nuisance maximum transfers. Nuisance records
are checked for complete schedule/score/summary consistency, not reexecuted
inside this adapter. Source files and review truth remain external trust.
Missing/unsupported pairs are merged, never declared distinct. The review
decision novelty_withheld retains the source pool WITHOUT a clear-label hold
or a measured equivalence edge. No threshold is supplied by either mode.

Output contexts retain admission.finalize_context_groups' provenance fields,
but these OBSERVED groups are NOT its globally calibrated GROUP_SCHEMA.
Downstream selection must enforce eligible_sample_ids, view_duplicate_groups
and one shared 12-view cap per group across the cumulative inventory. Do not
append counts to prior selections. No heldout quota or new donor is created.
"""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
import math

from . import morphology_frame_v2 as frame, morphology_set_v1 as leaf_set, provisional
from .admission import _UnionFind, _digest as digest, _id, _require, _sha

SCHEMA = 'greenhouse.observed_geometry_context_adapter.v2'
CONTROL_SCHEMA = 'greenhouse.explicit_geometry_controls.v2'
REVIEW_SCHEMA = 'greenhouse.joint_geometry_reviews.v2'
METRIC = 'v2_normalized_linf_unordered_leaf_bottleneck'
SET_NUISANCE_SCHEMA = 'greenhouse.actual_morphology_set_nuisance_evaluation.v1'
ARTIFACTS = {'native_pair_plan', 'native_pair_result', 'native_pair_audit',
             'visual_review', 'original_rgb', 'generated_rgb'}
VIEW_ARTIFACTS = {'native_plan', 'native_result', 'native_audit', 'visual_review', 'generated_rgb'}
POSITIVE_DECISIONS = {'joint_visual_geometry_distinct', 'geometry_distinct_plausible'}
DECISIONS = POSITIVE_DECISIONS | {'hold', 'novelty_withheld'}


@dataclass(frozen=True)
class PinnedJSON:
    data: bytes
    sha256: str

    def read(self):
        _require(type(self.data) is bytes and sha256(self.data).hexdigest() == _sha(self.sha256),
                 'Changed pinned JSON bytes')
        def unique(pairs):
            result = {}
            for key, value in pairs:
                _require(key not in result, 'Duplicate JSON key')
                result[key] = value
            return result
        value = json.loads(self.data, object_pairs_hook=unique)
        _require(isinstance(value, dict), 'JSON object required')
        digest(value)  # Reject NaN/infinity, even in unconsumed evidence.
        return value


def _bindings(value):
    _require(isinstance(value, dict) and value, 'Nonempty evidence bindings required')
    for name, pin in value.items():
        _id(name); _sha(pin)
    normalized = {}
    for name, pin in value.items():
        _require(normalized.setdefault(name.replace('\\', '/'), pin) == pin, 'Conflicting normalized binding')
    return normalized


def _contexts(report, packets, views, splits):
    contexts, described, joins, originals, loaded = {}, {}, {}, {}, {}
    for row in report['records']:
        path = row['extracted_packet']
        if path not in loaded:
            pin = packets[path]
            _require(pin.sha256 == row['extracted_packet_sha256'], 'Wrong extracted packet pin')
            packet, _ = provisional._sealed(pin.read())
            _require(packet['schema'] == 'greenhouse.actual_output_morphology.v1'
                     and packet['catalogue_replayed'] is True and packet['frozen_splits'] == splits
                     and packet['frozen_splits_sha256'] == digest(splits), 'Unverified extraction/splits')
            bound = _bindings(packet['bindings']); _bindings(packet['code_bindings'])
            loaded[path] = packet, bound
        packet, bound = loaded[path]
        _require(packets[path].sha256 == row['extracted_packet_sha256'], 'Conflicting packet pin')
        matches = [r for r in packet['records'] if r['context_id'] == row['context_id']]
        _require(len(matches) == 1, 'Missing/ambiguous actual output row')
        source = matches[0]; key = _id(source['context_id']); family = source['source_family']
        _require(row['kind'] == source['kind'] and row['source_target'] == source['source_target']
                 and source['split'] == splits[family] == 'train', 'Descriptor ancestry mismatch')
        identity = source['identity']
        _require(key == 'actual-output:' + digest(identity)
                 and all(identity[k] == source[k] for k in ('kind', 'source_family', 'source_target')),
                 'Actual mesh/context identity mismatch')
        _sha(identity['manifest_sha256']); _bindings(identity['mesh_sha256'])
        _require(row['input_geometry_sha256'] == source['input_geometry_sha256']
                 == digest(source['input_geometry']), 'Changed extracted geometry')
        fresh = frame.describe(source['input_geometry'])
        _require(digest(fresh) == digest(row['frame_v2']), 'V2 result differs from extracted geometry')
        provenance = {k: source[k] for k in ('source_family', 'source_target', 'source_context_id')}
        provenance.update(qualified_geometry=False)
        _require(contexts.setdefault(key, provenance) == provenance
                 and described.setdefault(key, fresh) == fresh, 'Conflicting repeated context')
        if source['kind'] == 'original':
            _require(key == source['source_context_id'] and not row['associated_sample_ids']
                     and originals.setdefault(source['source_target'], key) == key, 'Ambiguous original context')
        else:
            _require(source['kind'] == 'generated', 'Unknown geometry kind')
        for sample in row['associated_sample_ids']:
            _require(sample in views and sample not in joins and source['kind'] == 'generated',
                     'Unknown/repeated sample-context join')
            view = views[sample]; lineage = view['provenance']['lineage']
            original = [r for r in packet['records'] if r['context_id'] == source['source_context_id']]
            _require(len(original) == 1, 'Missing original extraction')
            qualification = packet['directory'].replace('\\', '/') + '/qualification.json'
            _require(all(view[k] == source[k] for k in ('source_family', 'source_target', 'target_id'))
                     and lineage['frozen_family_assignments_sha256'] == digest(splits)
                     and bound.get(lineage['source_collection_plan'].replace('\\', '/')) == lineage['source_plan_sha256']
                     and bound.get(qualification) == lineage['generated_qualification_sha256']
                     and original[0]['identity']['manifest_sha256'] == lineage['source_manifest_sha256']
                     and identity['mesh_sha256'][source['source_target'].split('/')[-1]]
                     == lineage['generated_component_asset_sha256'], 'Sample/output lineage mismatch')
            joins[sample] = key
    for key, source in contexts.items():
        original = contexts.get(source['source_context_id'])
        _require(original is not None and original['source_context_id'] == source['source_context_id']
                 and all(source[k] == original[k] for k in ('source_family', 'source_target')),
                 'Missing/conflicting original ancestry')
        if described[key]['descriptor'] is not None and described[source['source_context_id']]['descriptor'] is not None:
            _require(described[key]['frame_reference_sha256']
                     == described[source['source_context_id']]['frame_reference_sha256'], 'Changed donor frame')
    for sample, view in views.items():
        if sample not in joins:
            source = originals.get(view['source_target'], 'original:' + view['source_target'])
            contexts.setdefault(source, dict(source_family=view['source_family'], source_target=view['source_target'],
                                            source_context_id=source, qualified_geometry=False))
            key = 'unresolved:' + view['context_id']; joins[sample] = key
            record = dict(source_family=view['source_family'], source_target=view['source_target'],
                          source_context_id=source, qualified_geometry=False)
            _require(contexts.setdefault(key, record) == record, 'Conflicting unresolved context')
    return contexts, described, joins


def _distance(metric, left, right):
    if left is None or right is None:
        return None
    if metric == METRIC:
        return frame.distance(left, right) if left['leaf_count'] == right['leaf_count'] else None
    _require(metric == leaf_set.METRIC, 'Unknown geometry metric')
    return leaf_set.compare(left, right)['distance'] if left['leaf_count'] and right['leaf_count'] else None


def _set_nuisance(control, nuisance):
    value, _ = provisional._sealed(nuisance)
    proof = value['metric_proof']; definition = value['control_definition']
    _require(value['schema'] == SET_NUISANCE_SCHEMA and value['metric'] == leaf_set.METRIC
             and proof['metric'] == leaf_set.METRIC and proof['frame_version'] == frame.FRAME_VERSION
             and proof['descriptor_schema'] == frame.SCHEMA
             and _sha(control['metric_proof_sha256']) == digest(proof), 'Set metric proof mismatch')
    bound = _bindings(value['source_bindings'])
    for role in ('metric_implementation', 'frame_implementation'):
        item = proof[role]
        _require(bound.get(item['path'].replace('\\', '/')) == _sha(item['sha256']), 'Unbound metric implementation')
    _require(value['frozen_splits_sha256'] == control['frozen_splits_sha256']
             and value['threshold_selected'] is None
             and all(value[k] is False for k in ('calibration_validated', 'qualified_geometry', 'training_approved')),
             'Nuisance is not a calibration/qualification receipt')
    _require(value['control_definition_sha256'] == digest(definition)
             and value['measurement_records_sha256'] == digest(value['controls']), 'Changed nuisance records/definition')
    baselines = value['baseline_rows']; rotations = definition['rotations']; scales = definition['uniform_scales']
    _require(isinstance(baselines, dict) and baselines and type(value['summary']['actual_rows']) is int
             and value['summary']['actual_rows'] == len(baselines), 'Incomplete nuisance baselines')
    for key, base in baselines.items():
        _id(key); _sha(base['descriptor_sha256']); _sha(base['input_geometry_sha256'])
        _require(bound.get(base['source_packet'].replace('\\', '/')) == _sha(base['source_packet_sha256'])
                 and type(base['source_packet_row']) is int and base['source_packet_row'] >= 0, 'Unbound nuisance baseline')
    _require(all(type(s) in (int, float) and math.isfinite(s) and s > 0 for s in scales)
             and len(scales) == len(set(scales)), 'Unique positive nuisance scales required')
    rotation_keys = [(r['group'], r['rotation_index']) for r in rotations]
    _require(rotation_keys and len(rotation_keys) == len(set(rotation_keys)), 'Unique nuisance rotations required')
    expected = {(key, group, index, scale) for key in baselines for group, index in rotation_keys for scale in scales}
    seen, summaries = set(), {}
    for row in value['controls']:
        key = (row['row_id'], row['group'], row['rotation_index'], row['uniform_scale'])
        _require(key in expected and key not in seen, 'Repeated/unknown nuisance transform')
        seen.add(key); base = baselines[row['row_id']]; score = row['comparison']
        _require(row['baseline_descriptor_sha256'] == base['descriptor_sha256']
                 and row['state'] == 'descriptor_available_not_calibrated' and row['holds'] == []
                 and score['metric'] == leaf_set.METRIC, 'Wrong/held nuisance measurement')
        _sha(row['input_geometry_sha256']); _sha(row['descriptor_sha256'])
        numbers = [score[k] for k in ('distance', 'fixed_linf', 'leaf_forward_linf', 'leaf_reverse_linf', 'leaf_hausdorff_linf')]
        _require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in numbers)
                 and score['leaf_hausdorff_linf'] == max(numbers[2:4])
                 and score['distance'] == max(numbers[1:4]), 'Inconsistent set nuisance score')
        _require(type(row['passed']) is bool and row['passed'] == (score['distance'] <= frame.NUMERICAL_TOLERANCE)
                 and type(row['selection_unchanged']) is bool, 'Inconsistent nuisance pass/selection')
        summary = summaries.setdefault(row['group'], dict(comparisons=0, passes_at_1e_9=0, holds=0,
                                                          selection_changes=0, max_distance=0.))
        summary['comparisons'] += 1; summary['passes_at_1e_9'] += row['passed']
        summary['selection_changes'] += not row['selection_unchanged']
        summary['max_distance'] = max(summary['max_distance'], score['distance'])
    _require(seen == expected and summaries == value['summary']['controls'], 'Incomplete nuisance schedule/summary')


def _qualified(control, nuisance, review, contexts, described, views, joins, baseline):
    metric = control['metric']
    _require(control['schema'] == CONTROL_SCHEMA and control['frame_version'] == frame.FRAME_VERSION
             and metric in (METRIC, leaf_set.METRIC)
             and control['qualification_status'] == 'externally_qualified_exact_contexts', 'Explicit V2 qualification required')
    threshold = control['threshold']
    _require(type(threshold) in (int, float) and math.isfinite(threshold) and threshold >= 0,
             'Explicit finite geometry threshold required')
    schema = SET_NUISANCE_SCHEMA if metric == leaf_set.METRIC else 'greenhouse.actual_morphology_frame_v2_evaluation.v1'
    _require(nuisance['schema'] == schema and nuisance.get('metric', METRIC) == metric
             and nuisance['descriptor_schema'] == frame.SCHEMA
             and nuisance['frame_version'] == frame.FRAME_VERSION, 'Pinned V2 nuisance evidence required')
    _bindings(nuisance['source_bindings'])
    if metric == leaf_set.METRIC:
        _set_nuisance(control, nuisance)
    else:
        _bindings(nuisance['output_bindings'])
    definition = nuisance['control_definition']
    _require(definition['numerical_tolerance'] == frame.NUMERICAL_TOLERANCE
             and len(set(definition['uniform_scales'])) > 1 and any(definition['translation'])
             and definition['leaf_order'] == 'reversed' and definition['irrelevant_names'] == 'renamed'
             and definition['rotations'], 'Incomplete nuisance families')
    groups = nuisance['summary']['controls']
    _require(isinstance(groups, dict) and groups, 'Missing nuisance comparisons')
    for group in groups.values():
        _require(type(group['comparisons']) is int and group['comparisons'] > 0
                 and all(type(group[k]) is int for k in ('passes_at_1e_9', 'holds', 'selection_changes'))
                 and group['passes_at_1e_9'] == group['comparisons'] and group['holds'] == 0
                 and group['selection_changes'] == 0
                 and type(group['max_distance']) in (int, float)
                 and 0 <= group['max_distance'] <= min(threshold, frame.NUMERICAL_TOLERANCE),
                 'Unresolved/failed nuisance controls')
    scope = control['context_ids']; positive = control['positive_context_ids']
    _require(scope and scope == sorted(set(scope)) and positive and positive == sorted(set(positive))
             and set(positive) <= set(scope) <= set(contexts), 'Explicit positive/exact context scope required')
    bound = _bindings(review['bindings']); accepted = set(); seen = set()
    for row in review['contexts']:
        key = row['context_id']; _require(key in contexts and key not in seen, 'Unknown/duplicate context review')
        seen.add(key); source = contexts[key]['source_context_id']; sample = row['sample_id']
        _require(key != source and sample in views and joins[sample] == key, 'Review sample/context mismatch')
        view = views[sample]
        _require(row['descriptor_sha256'] == described[key].get('descriptor_sha256')
                 and row['source_descriptor_sha256'] == described[source].get('descriptor_sha256')
                 and row['source_context_id'] == source
                 and row['sample_sha256'] == view['provenance']['sample_sha256']
                 and row['annotation_evidence_id'] == view['annotation_review']['evidence_id']
                 and row['generated_qualification_sha256'] == view['provenance']['lineage']['generated_qualification_sha256'],
                 'Review descriptor/native bindings mismatch')
        roles = set(row['artifacts'])
        _require(roles == ARTIFACTS if key in positive else roles in (ARTIFACTS, VIEW_ARTIFACTS),
                 'Complete matched control or own native review artifacts required')
        for artifact in row['artifacts'].values():
            _require(bound.get(artifact['path'].replace('\\', '/')) == _sha(artifact['sha256']), 'Unbound review artifact')
        _require(row['artifacts']['generated_rgb']['sha256'] == view['encoded_rgb_sha256'], 'Review RGB mismatch')
        if roles == VIEW_ARTIFACTS:
            _require(row['artifacts']['native_plan']['sha256'] == view['provenance']['plan_sha256']
                     and row['artifacts']['native_result']['sha256'] == view['provenance']['result_sha256']
                     and row['artifacts']['native_audit']['sha256'] == view['annotation_review']['evidence_id'],
                     'Own native plan/result/audit mismatch')
        _require(row['decision'] in DECISIONS, 'Unqualified review decision')
        if key in positive:
            _require(row['decision'] == 'joint_visual_geometry_distinct', 'Positive requires joint paired review')
        static_reasons = set(baseline['excluded'].get(sample, [])) - {
            'original_source_cap', 'duplicate_view', 'train_goal_reached', 'control_role'}
        if row['decision'] in POSITIVE_DECISIONS and not static_reasons:
            _require(described[key]['descriptor'] is not None and described[source]['descriptor'] is not None,
                     'Held descriptor cannot qualify')
            distance = _distance(metric, described[key]['descriptor'], described[source]['descriptor'])
            if metric == leaf_set.METRIC:
                _require(row['geometry_metric'] == metric and row['metric_proof_sha256'] == control['metric_proof_sha256']
                         and type(row['distance_to_original']) in (int, float)
                         and row['distance_to_original'] == distance, 'Review metric/distance proof mismatch')
            else:
                _require(row.get('geometry_metric', METRIC) == METRIC, 'Review metric mismatch')
            _require(distance is not None and distance > threshold,
                     'Positive/context is not distinct under supplied cutoff')
            if roles == ARTIFACTS:
                _require(row['artifacts']['original_rgb']['sha256'] != view['encoded_rgb_sha256'], 'Identical positive RGB')
            accepted.add(key)
    _require(set(positive) <= accepted and set(scope) <= seen, 'Missing qualified positive/context reviews')
    return set(scope) & accepted, threshold


def adapt_contexts(inventory, descriptor_report, packets, image_graph, *, frozen_splits,
                   controls=None, nuisance=None, reviews=None):
    """Return detached observed budget inputs; missing qualification gives NO extras."""
    try:
        return _adapt(inventory, descriptor_report, packets, image_graph, frozen_splits, controls, nuisance, reviews)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError('Malformed geometry adapter evidence: ' + str(exc)) from exc


def _adapt(inventory, descriptor_report, packets, image_graph, splits, controls, nuisance, reviews):
    splits = dict(splits)
    _require(Counter(splits.values()) == Counter(train=16, validation=4, test=4), 'Original 16/4/4 donor reservations required')
    inv, report, graph = inventory.read(), descriptor_report.read(), image_graph.read()
    review = reviews.read() if reviews else dict(schema=REVIEW_SCHEMA, overlays=[], contexts=[])
    _require(review['schema'] == REVIEW_SCHEMA, 'Versioned joint reviews required')
    overlays = list(review['overlays'])
    for row in review['contexts']:
        _require(row['decision'] in DECISIONS, 'Unqualified review decision')
        if row['decision'] == 'hold':
            overlays.append(dict(sample_id=row['sample_id'], evidence_id=reviews.sha256, visual_hold=True))
    baseline = provisional.select_observed_train(inv, splits, graph, overlays=overlays)
    _require(report['schema'] == 'greenhouse.observed_actual_morphology_snapshot.v1'
             and report['inventory_sha256'] == inv['sha256'] and report['threshold_selected'] is None
             and report['qualified_geometry'] is False and report['training_approved'] is False
             and report['counts']['records'] == len(report['records']), 'Actual unqualified descriptor snapshot required')
    report_bindings = _bindings(report['bindings'])
    _require(inventory.sha256 in report_bindings.values(), 'Descriptor report lacks inventory byte pin')
    views = {r['sample_id']: r for r in inv['records']}
    contexts, described, joins = _contexts(report, packets, views, splits)
    qualified, threshold, edges, unresolved, compared = set(), None, [], 0, 0
    metric = METRIC
    unknown_edges = []
    missing = sorted(set(contexts) - {k for k, v in described.items() if v['descriptor'] is not None})
    blockers = ['explicit_qualified_controls_required'] if controls is None else []
    if controls is not None:
        _require(nuisance is not None and reviews is not None, 'Pinned nuisance and joint reviews required')
        control = controls.read()
        metric = control['metric']
        expected = dict(descriptor_report_sha256=descriptor_report.sha256, inventory_sha256=inv['sha256'],
                        image_graph_sha256=graph['sha256'], reviews_sha256=reviews.sha256,
                        nuisance_sha256=nuisance.sha256, frozen_splits_sha256=digest(splits))
        _require(all(control[k] == value for k, value in expected.items()), 'Stale qualification scope/pins')
        qualified, threshold = _qualified(control, nuisance.read(), review, contexts, described, views, joins, baseline)
        keys = sorted(contexts)
        for index, left in enumerate(keys):
            for right in keys[index+1:]:
                a, b = described.get(left, {}).get('descriptor'), described.get(right, {}).get('descriptor')
                distance = _distance(metric, a, b)
                if distance is None:
                    unresolved += 1; unknown_edges.append([left, right])
                else:
                    compared += 1
                    if distance <= threshold:
                        edges.append([left, right])
        if unknown_edges:
            blockers.append('uncertain_pairs_conservatively_merged')
    if missing:
        blockers.append('missing_or_held_actual_descriptors')
    union = _UnionFind(contexts); union.edges(edges); union.edges(unknown_edges)
    for key, source in contexts.items():
        source['qualified_geometry'] = key in qualified
        if key in qualified:
            source['geometry_evidence_id'] = controls.sha256
        else:
            union.union(key, source['source_context_id'])
    groups = union.groups()
    total = len(contexts) * (len(contexts)-1) // 2
    group_splits = {}
    for key, context in contexts.items():
        group_splits.setdefault(groups[key], set()).add(splits[context['source_family']])
    eligible = sorted(k for k in views if not (set(baseline['excluded'].get(k, [])) - {
        'original_source_cap', 'duplicate_view', 'train_goal_reached'})
        and len(group_splits[groups[joins[k]]]) == 1)
    result = dict(schema=SCHEMA, state='observed_context_budget_inputs_not_admission', contexts=contexts,
        groups=groups, sample_context_ids=joins, eligible_sample_ids=eligible,
        view_duplicate_groups=baseline['view_duplicate_groups'], baseline_selection=baseline,
        additional_budget_context_ids=sorted(k for k in qualified if groups[k] != groups[contexts[k]['source_context_id']]),
        frozen_splits=splits, max_views_per_group=12, train_goal=20000, heldout_counted=False,
        bindings=dict(inventory=inventory.sha256, descriptors=descriptor_report.sha256, image_graph=image_graph.sha256,
                      controls=controls.sha256 if controls else None, reviews=reviews.sha256 if reviews else None,
                      nuisance=nuisance.sha256 if nuisance else None),
        geometry=dict(metric=metric, threshold=threshold, equivalence_edges=edges, compared_pairs=compared,
                      conservative_merge_edges=unknown_edges,
                      pair_accounting_complete=controls is not None and compared+unresolved == total,
                      total_pairs=total, comparisons_complete=controls is not None and compared == total
                      and not missing and not unresolved, unresolved_pairs=unresolved,
                      unexamined_pairs=total-compared-unresolved,
                      missing_context_ids=missing, blockers=blockers),
        scope=dict(global_collection_complete=False, native_execution_verified=False, source_files_reverified=False,
                   review_truth_verified=False, nuisance_transforms_replayed=False, new_biological_families=0),
        training_approved=False, source_cap_reset=False, calibration_validated=False, admission_performed=False)
    result['sha256'] = digest(result)
    return result
