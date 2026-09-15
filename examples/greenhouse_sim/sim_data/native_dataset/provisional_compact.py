"""Read-only v2 graph admission to the UNCHANGED provisional candidate policy.

select_observed_train(inventory, frozen_splits, pair_coverage, *, overlays=())
passes raw-only v1 graphs through exactly; compact inventory requires v2 even
when a v1 graph names the existing identity PNGs. For compact/mixed v2 graphs, authenticate
the aggregate seal and reconstruct/check the original v1 graph hash, validate
complete inventory coverage and physical/logical source pins, then delegate.
Only the internal projection is v1: the returned v2 result retains the entire
original graph, its hash, and the delegated selection hash. Nothing is relabeled
as an original v1 receipt. No input mappings or original modules are changed.

This rechecks graph-bound RGB/metadata bytes, NOT native arrays, labels/trace
derivation, pair scores or pruning execution. The caller must authenticate the
inventory/graph and audit provenance externally; internal seals are not proof
of independent execution. Missing observed roots remain missing. All original
policy scope limitations and false approvals survive, without new native-data,
geometry-novelty, completeness, storage-qualification or training claims.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from . import provisional as policy

SCHEMA = 'greenhouse.observed_train_provisional_selection.v2'
GRAPH_SCHEMA = 'greenhouse.native_near_image_graph.v2'
POLICY_SHA256 = '2aaea0993cbfdbd32ae9d161d7a4fa5f2d76c72ff92753721078433b77b69e6a'
GRAPH_ADAPTER_SHA256 = 'cc52a8e9222f97b83872f84b3f0424b51c842c5c742ed0f7d4708599f551827e'
_SELF_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_BASE_FIELDS = set('schema inventory_sha256 sample_ids image_pins threshold metric patch_radius patch_anchor '
    'dimensions near_image_edges edge_scores counts complete pruning input_bindings implementation_bindings '
    'scope global_collection_complete threshold_calibrated training_approved depth_recomputed images_modified'.split())
_ADAPTER_FIELDS = set('source_adapter base_graph_schema base_graph_sha256 compact_image_pins '
    'compact_validation_scope compact_sources_rechecked source_nominal_coordinates '
    'full_native_audit_performed images_extracted images_reencoded'.split())


def implementation_bindings():
    """Own version and frozen policy/index dependencies, without renderer imports."""
    from . import compact_image_index as index
    code = dict(index._LOADED_CODE)
    for path, expected in ((policy.__file__, POLICY_SHA256), (index.__file__, GRAPH_ADAPTER_SHA256)):
        policy._require(index._sha(path) == expected, 'Frozen provisional/index implementation changed')
        code[str(Path(path).resolve())] = expected
    code[str(Path(__file__).resolve())] = _SELF_SHA256
    index._verify(code)
    return code


def _absolute(value):
    policy._id(value)
    path = Path(value)
    policy._require(path.is_absolute(), 'Explicit absolute source path required')
    return str(path.resolve())


def _projection(packet, inventory_sha, graph):
    from . import compact_image_index as index
    from . import near_image_index as base_index
    policy._require(set(graph) == _BASE_FIELDS | _ADAPTER_FIELDS, 'Unknown/incomplete v2 aggregate contract')
    required = dict(schema=GRAPH_SCHEMA, source_adapter=index.SOURCE_ADAPTER,
        base_graph_schema=policy.PAIR_SCHEMA, compact_sources_rechecked=True,
        compact_validation_scope='manifest_sample_label_eligible_trace_and_rgb_not_full_native_audit',
        source_nominal_coordinates='caller_supplied_bound_original_label',
        scope='all_pairs_of_explicit_observed_pins_only')
    for key, expected in required.items():
        policy._require(type(graph[key]) is type(expected) and graph[key] == expected,
                        'Changed compact aggregate declaration: ' + key)
    for key in ('full_native_audit_performed', 'images_extracted', 'images_reencoded',
                'depth_recomputed', 'images_modified'):
        policy._require(graph[key] is False, 'Unexpected compact aggregate claim: ' + key)
    pruning = graph['pruning']
    policy._require(set(pruning) == {'method', 'tile', 'guard'}
        and pruning['method'] == 'weighted_native_block_mean_MAE_lower_bound'
        and type(pruning['tile']) is int and 1 <= pruning['tile'] <= 256
        and type(pruning['guard']) is float and pruning['guard'] == base_index.BOUND_GUARD,
        'Different pruning contract')
    policy._require(graph['implementation_bindings'] == index._LOADED_CODE, 'Changed graph implementation pins')
    views = {row['sample_id']: row for row in packet['records']}
    policy._require(len(views) == len(packet['records']), 'Duplicate inventory sample')
    # Reconstruct the original base exactly, including its smaller binding maps.
    base = {k: deepcopy(graph[k]) for k in _BASE_FIELDS}
    base['schema'] = policy.PAIR_SCHEMA
    base['input_bindings'] = {}
    for pin in graph['image_pins']:
        policy._require(set(pin) == {'sample_id', 'path', 'sha256', 'decoded_rgb_sha256', 'nominal_uv'},
                        'Unexpected image pin fields')
        index._merge(base['input_bindings'], {_absolute(pin['path']): policy._sha(pin['sha256'])})
    base['implementation_bindings'] = {str(Path(m.__file__).resolve()): graph['implementation_bindings'][
        str(Path(m.__file__).resolve())] for m in (base_index, index.near_image)}
    policy._require(policy.digest(base) == policy._sha(graph['base_graph_sha256']), 'Stale original base graph hash')
    # Complete N*(N-1)/2 accounting, RGB pins, all endpoints/scores and original
    # metric/cutoff/nominal patch gates, without running candidate selection yet.
    edges = policy._pair_edges(base, inventory_sha, views)
    policy._require([list(pair) for pair in edges] == graph['near_image_edges'], 'Noncanonical aggregate edges')
    base['sha256'] = graph['base_graph_sha256']
    return base, views


def _sources(packet, graph, views):
    from . import compact_image_index as index
    inventory_bindings = packet['provenance']['source_bindings']
    policy._require(isinstance(inventory_bindings, dict) and inventory_bindings, 'Inventory source pins required')
    policy._require(isinstance(graph['compact_image_pins'], list), 'Explicit compact pin inventory required')
    compact, source_bindings, originals = {}, {}, {}
    for value in graph['compact_image_pins']:
        source = index._snapshot(index.CompactImagePin(**value))
        policy._require(source.sample_id in views and source.sample_id not in compact, 'Unknown/duplicate compact pin')
        compact[source.sample_id] = source
    policy._require(list(compact) == sorted(compact), 'Sorted compact pin inventory required')
    for pin in graph['image_pins']:
        key = pin['sample_id']
        row = views[key]
        root = Path(_absolute(row['sample_path']))
        originals[row['sample_path']] = str(root)
        original_image = _absolute(pin['path'])
        originals[pin['path']] = original_image
        provenance = row['provenance']
        logical = provenance['logical_files']
        for name, expected in (('sample.json', provenance['sample_sha256']),
                               ('supervision/label.json', provenance['label_sha256']),
                               ('inputs/rgb.png', row['encoded_rgb_sha256'])):
            policy._require(logical[name]['sha256'] == policy._sha(expected), 'Inventory logical pin mismatch')
        marker = str(root/'bundle.json')
        is_compact = marker in inventory_bindings or (root/'bundle.json').exists()
        policy._require(is_compact == (key in compact), 'Compact source omitted or raw source relabeled')
        if key in compact:
            source = compact[key]
            policy._require(_absolute(source.root) == str(root), 'Compact root differs from inventory')
            originals[source.root] = str(root)
            policy._require(source.sample_sha256 == provenance['sample_sha256']
                and source.label_sha256 == provenance['label_sha256']
                and source.encoded_rgb_sha256 == row['encoded_rgb_sha256']
                and source.decoded_rgb_sha256 == row['decoded_rgb_sha256'], 'Compact/inventory pin mismatch')
            image, bound = index._resolve(source)  # Fresh SampleReader; no native-array/audit replay.
            raw_manifest = index.bundle.read_bounded(root/'bundle.json')
            policy._require(index.bundle.digest(raw_manifest) == source.bundle_sha256, 'Compact manifest changed')
            for name, entry in json.loads(raw_manifest)['files'].items():
                path = str(index.bundle.safe_path(root, entry['stored_path']))
                if path in bound:
                    policy._require(logical[name] == dict(sha256=entry['logical_sha256'], bytes=entry['logical_bytes']),
                                    'Inventory compact logical-file binding mismatch')
            expected = asdict(image)
            expected['nominal_uv'] = list(expected['nominal_uv'])
            actual = dict(pin, nominal_uv=list(pin['nominal_uv']))
            policy._require(expected == actual, 'Compact aggregate image/nominal pin mismatch')
            index._merge(source_bindings, bound)
        else:
            policy._require(original_image == str(root/'inputs/rgb.png'), 'Raw image differs from inventory source')
            index._merge(source_bindings, {original_image: pin['sha256']})
    policy._require(source_bindings == graph['input_bindings'], 'Missing/extra/conflicting aggregate source bindings')
    for path, expected in source_bindings.items():
        policy._require(inventory_bindings.get(path) == expected, 'Graph source not pinned by inventory: ' + path)
    index._verify(source_bindings)
    return source_bindings, originals


def _require_raw_inventory(packet):
    bindings = packet.get('provenance', {}).get('source_bindings', {})
    policy._require(isinstance(bindings, dict), 'Inventory source bindings must be a mapping')
    markers = {Path(path).resolve() for path in bindings if Path(path).name == 'bundle.json'}
    for row in packet['records']:
        # Legacy pure accounting fixtures may omit storage paths. Actual sealed
        # inventory records use sample_path as the SampleReader root.
        if 'sample_path' in row:
            marker = Path(row['sample_path'])/'bundle.json'
            policy._require(marker.resolve() not in markers and not marker.exists(),
                            'Compact inventory requires a provenance-preserving v2 graph')


def select_observed_train(inventory, frozen_splits, pair_coverage, *, overlays=()):
    """Return candidates only; v1 parity or a provenance-preserving v2 result.

    v2 validates aggregate accounting and existing graph-bound bytes, not all
    inventory files/native arrays. External audit/coverage obligations remain.
    """
    try:
        if isinstance(pair_coverage, dict) and pair_coverage.get('schema') == policy.PAIR_SCHEMA:
            policy._require(not _ADAPTER_FIELDS.intersection(pair_coverage), 'V2 provenance cannot masquerade as raw v1')
            packet, _ = policy._sealed(inventory)
            _require_raw_inventory(packet)
            result = policy.select_observed_train(inventory, frozen_splits, pair_coverage, overlays=overlays)
            _require_raw_inventory(packet)
            return result
        return _select(inventory, frozen_splits, pair_coverage, overlays)
    except (KeyError, TypeError, IndexError, AttributeError, OSError) as exc:
        raise ValueError('Malformed compact provisional evidence: ' + str(exc)) from exc


def _select(inventory, frozen_splits, pair_coverage, overlays):
    from . import compact_image_index as index
    packet, inventory_sha = policy._sealed(inventory)
    graph, graph_sha = policy._sealed(pair_coverage)
    code = implementation_bindings()
    base, views = _projection(packet, inventory_sha, graph)
    bound, originals = _sources(packet, graph, views)
    result = policy.select_observed_train(inventory, frozen_splits, base, overlays=overlays)
    # Reopen every compact source after delegation and rehash all graph inputs.
    final_bound, final_originals = _sources(packet, graph, views)
    policy._require((bound, originals) == (final_bound, final_originals), 'Source identity changed during selection')
    for original, expected in originals.items():
        policy._require(_absolute(original) == expected, 'Original source path retargeted')
    index._verify(code)
    delegated_sha = result.pop('sha256')
    result.update(schema=SCHEMA, pair_coverage_sha256=graph_sha,
        graph_adapter=dict(source_graph=dict(graph, sha256=graph_sha),
            delegated_pair_coverage_sha256=base['sha256'], delegated_selection_schema=policy.SCHEMA,
            delegated_selection_sha256=delegated_sha, implementation_bindings=code,
            aggregate_accounting_revalidated=True, graph_bound_rgb_metadata_reverified=True,
            inventory_native_arrays_reverified=False, audit_execution_independently_verified=False,
            pair_measurements_recomputed=False, pruning_bounds_independently_verified=False,
            storage_qualification_granted=False))
    result['sha256'] = policy.digest(result)
    return result
