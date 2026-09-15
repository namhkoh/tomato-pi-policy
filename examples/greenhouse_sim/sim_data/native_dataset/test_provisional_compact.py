"""Synthetic graph/provenance tests only; no native qualification or collection."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest

from . import compact_image_index as compact_index, near_image_index as index
from . import provisional as policy, provisional_compact as adapter
from .bundle import SampleReader, digest
from .test_compact_image_index import source, physical_pin, payload
from .test_provisional import row, inventory, coverage, sealed


def fixture_graph(root, *, all_raw=False):
    sources = [source(root, name, np.full((816, 1696, 3), level, np.uint8), (800.5, 400.5))
               for name, level in [('a', 20), ('b', 22), ('c', 200)]]
    pins, rows, bindings = [], [], {}
    for i, (pin, raw) in enumerate(sources):
        compact = i > 0 and not all_raw
        folder = Path(pin.root) if compact else raw
        pins.append(pin if compact else index.ImagePin(pin.sample_id, str(raw/'inputs/rgb.png'),
            pin.encoded_rgb_sha256, pin.decoded_rgb_sha256, pin.nominal_uv))
        reader = SampleReader(folder, expected_bindings={'sample.json': pin.sample_sha256,
            'supervision/label.json': pin.label_sha256}, expected_json={'supervision/query_trace.json': {'passed': True}})
        names = set(reader.manifest['files'] if reader.manifest else reader.metadata['files'])
        names |= {'sample.json', 'supervision/label.json', 'supervision/query_trace.json'}
        logical = {name: dict(sha256=digest(reader.read(name)), bytes=len(reader.read(name))) for name in names}
        rows.append(row(pin.sample_id, target_id='synthetic/target', sample_path=str(folder),
            encoded_rgb_sha256=pin.encoded_rgb_sha256, decoded_rgb_sha256=pin.decoded_rgb_sha256,
            provenance=dict(sample_sha256=pin.sample_sha256, label_sha256=pin.label_sha256, logical_files=logical)))
        bindings.update({str(path): index._sha(path) for path in folder.rglob('*') if path.is_file()})
    packet = inventory(rows)
    packet['provenance'] = dict(source_bindings=bindings)
    packet = sealed(packet)
    graph = compact_index.build_graph(pins, inventory_sha256=packet['sha256'], threshold=policy.CUTOFF)
    images = [physical_pin(p) if isinstance(p, compact_index.CompactImagePin) else p for p in pins]
    base = index.build_graph(images, inventory_sha256=packet['sha256'], threshold=policy.CUTOFF)
    return packet, graph, base, sources


@pytest.fixture(scope='module')
def shared(tmp_path_factory):
    return fixture_graph(tmp_path_factory.mktemp('synthetic_compact_provisional'))


@pytest.fixture
def case(shared):
    # Files shared read-only across mapping-tamper tests. Mutable-file tests use
    # their own fresh fixtures. No mutation of production files or module state.
    return deepcopy(shared)


def select(packet, graph, **kwargs):
    return adapter.select_observed_train(packet, {'A': 'train'}, graph, **kwargs)


def rebase(graph, base):
    """Adversarially reseal even the embedded base, testing beyond stale hashes."""
    rebuilt = deepcopy(base)
    rebuilt.pop('sha256')
    for key in rebuilt.keys()-{'schema', 'input_bindings', 'implementation_bindings'}:
        rebuilt[key] = deepcopy(graph[key])
    rebuilt['input_bindings'] = {str(Path(p['path']).resolve()): p['sha256'] for p in graph['image_pins']}
    graph['base_graph_sha256'] = policy.digest(rebuilt)
    return sealed(graph)


@pytest.mark.parametrize('mode', ['empty', 'cap', 'heldout', 'holds', 'overlay', 'pruned'])
def test_raw_v1_exact_unchanged_policy_parity(mode):
    rows = [row('a'), row('b'), row('c')]
    splits, overlays, edges, pruned = {'A': 'train'}, (), [], 0
    if mode == 'empty':
        rows = []
    elif mode == 'cap':
        rows = [row(str(i)) for i in range(30)]
    elif mode == 'heldout':
        rows[1] = row('b', family='V', split='validation', decoded_rgb_sha256=rows[0]['decoded_rgb_sha256'])
        splits['V'] = 'validation'
    elif mode == 'holds':
        rows[1]['decision'] = 'hold_visual_clarity'
        edges = [['a', 'b'], ['b', 'c']]
    elif mode == 'overlay':
        overlays = [dict(sample_id='a', evidence_id='synthetic', visual_hold=True)]
    else:
        pruned = 3
    packet = inventory(rows)
    graph = coverage(packet, edges, pruned)
    before = deepcopy((packet, splits, graph, overlays))
    assert adapter.select_observed_train(packet, splits, graph, overlays=overlays) == policy.select_observed_train(
        packet, splits, graph, overlays=overlays)
    assert (packet, splits, graph, overlays) == before


def test_mixed_v2_exact_policy_output_and_explicit_reversible_provenance(case):
    packet, graph, base, _ = case
    overlays = [dict(sample_id='b', evidence_id='synthetic-hold', visual_hold=True)]
    before = deepcopy((packet, graph, overlays))
    expected = policy.select_observed_train(packet, {'A': 'train'}, base, overlays=overlays)
    result = select(packet, graph, overlays=overlays)
    assert result['schema'] == adapter.SCHEMA != expected['schema']
    assert result['pair_coverage_sha256'] == graph['sha256'] != base['sha256']
    provenance = result['graph_adapter']
    assert provenance['source_graph'] == graph
    assert provenance['delegated_pair_coverage_sha256'] == base['sha256']
    assert provenance['delegated_selection_sha256'] == expected['sha256']
    restored = deepcopy(result)
    restored.pop('graph_adapter')
    restored.update(schema=policy.SCHEMA, pair_coverage_sha256=base['sha256'], sha256=expected['sha256'])
    assert restored == expected
    assert result['scope'] == expected['scope']
    for field in ('training_approved', 'source_cap_reset', 'admission_performed', 'calibration_validated', 'original_reviews_modified'):
        assert result[field] is False
    for field in ('inventory_native_arrays_reverified', 'audit_execution_independently_verified',
                  'pair_measurements_recomputed', 'pruning_bounds_independently_verified', 'storage_qualification_granted'):
        assert provenance[field] is False
    assert (packet, graph, overlays) == before
    assert sealed(result) == result
    result['graph_adapter']['source_graph']['compact_image_pins'][0]['root'] = 'changed output'
    assert (packet, graph, overlays) == before


def test_all_raw_v2_supported_without_inventing_compact_sources(tmp_path):
    packet, graph, base, _ = fixture_graph(tmp_path, all_raw=True)
    assert select(packet, base) == policy.select_observed_train(packet, {'A': 'train'}, base)
    result = select(packet, graph)
    assert result['selected'] == policy.select_observed_train(packet, {'A': 'train'}, base)['selected']
    assert result['graph_adapter']['source_graph']['compact_image_pins'] == []


def test_plain_v1_identity_png_graph_cannot_hide_mixed_compact_provenance(case):
    packet, _, base, _ = case
    assert not adapter._ADAPTER_FIELDS.intersection(base)
    assert policy.select_observed_train(packet, {'A': 'train'}, base)['selected']
    with pytest.raises(ValueError, match='Compact inventory requires'):
        select(packet, base)


@pytest.mark.parametrize('evidence', ['binding_only', 'marker_only'])
def test_v1_rejects_either_inventory_bundle_pin_or_actual_marker(tmp_path, evidence):
    root = tmp_path/'sample'
    root.mkdir()
    marker = root/'bundle.json'
    packet = inventory([row('a', sample_path=str(root))])
    if evidence == 'binding_only':
        packet['provenance'] = dict(source_bindings={str(marker): 'a'*64})
        assert not marker.exists()
    else:
        marker.write_text('{}', encoding='utf-8')
        packet['provenance'] = dict(source_bindings={})
    packet = sealed(packet)
    graph = coverage(packet)
    with pytest.raises(ValueError, match='Compact inventory requires'):
        select(packet, graph)


@pytest.mark.parametrize('field,value', [('complete', False), ('sample_ids', ['a', 'b']),
    ('sample_ids', ['c', 'b', 'a']), ('inventory_sha256', '0'*64), ('metric', 'new_metric'),
    ('threshold', .04), ('patch_radius', 32), ('patch_anchor', 'attachment'), ('dimensions', [848, 408]),
    ('near_image_edges', [['a', 'missing']]), ('near_image_edges', [['a', 'a']]),
    ('near_image_edges', [['b', 'a']]), ('edge_scores', [1.]),
    ('global_collection_complete', True), ('threshold_calibrated', True), ('training_approved', True),
    ('depth_recomputed', True), ('images_modified', True), ('scope', 'global')])
def test_aggregate_revalidation_even_after_both_seals_rewritten(case, field, value):
    packet, graph, base, _ = case
    graph[field] = value
    with pytest.raises(ValueError):
        select(packet, rebase(graph, base))


@pytest.mark.parametrize('field,value', [('total_pairs', 2), ('resolved_pairs', 2),
    ('exact_compared_pairs', 0), ('bound_pruned_pairs', 3), ('total_pairs', True), ('resolved_pairs', -1)])
def test_full_pair_accounting_required(case, field, value):
    packet, graph, base, _ = case
    graph['counts'][field] = value
    with pytest.raises(ValueError):
        select(packet, rebase(graph, base))


@pytest.mark.parametrize('field,value', [('source_adapter', 'other'), ('base_graph_schema', 'other'),
    ('base_graph_sha256', '0'*64), ('compact_sources_rechecked', False), ('compact_sources_rechecked', 1),
    ('compact_validation_scope', 'full_native_audit'), ('source_nominal_coordinates', 'inferred'),
    ('full_native_audit_performed', True), ('images_extracted', True), ('images_reencoded', True),
    ('implementation_bindings', {})])
def test_adapter_claims_and_source_code_pins_are_not_dropped(case, field, value):
    packet, graph, _, _ = case
    graph[field] = value
    with pytest.raises(ValueError):
        select(packet, sealed(graph))


@pytest.mark.parametrize('field,value', [('method', 'other'), ('tile', 0), ('tile', True), ('tile', 257), ('guard', 0.)])
def test_original_pruning_contract(case, field, value):
    packet, graph, base, _ = case
    graph['pruning'][field] = value
    with pytest.raises(ValueError):
        select(packet, rebase(graph, base))


@pytest.mark.parametrize('field', ['bundle_sha256', 'sample_sha256', 'label_sha256',
                                  'encoded_rgb_sha256', 'decoded_rgb_sha256'])
def test_compact_pin_must_match_inventory_and_existing_bytes(case, field):
    packet, graph, _, _ = case
    graph['compact_image_pins'][0][field] = '0'*64
    with pytest.raises(ValueError):
        select(packet, sealed(graph))


@pytest.mark.parametrize('fault', ['omit', 'duplicate', 'unknown', 'raw_root', 'another_root', 'relative_root',
                                 'nominal', 'extra_field', 'extra_input', 'missing_input', 'unbound_input'])
def test_no_partial_mixed_coverage_or_noncompact_impersonation(case, fault):
    packet, graph, base, sources = case
    pin = graph['compact_image_pins'][0]
    if fault == 'omit':
        graph['compact_image_pins'].pop(0)
    elif fault == 'duplicate':
        graph['compact_image_pins'].append(deepcopy(pin))
    elif fault == 'unknown':
        pin['sample_id'] = 'outside-inventory'
    elif fault == 'raw_root':
        pin['root'] = str(sources[1][1])
    elif fault == 'another_root':
        pin['root'] = sources[2][0].root
    elif fault == 'relative_root':
        pin['root'] = 'relative'
    elif fault == 'nominal':
        pin['nominal_uv'] = [900., 400.]
    elif fault == 'extra_field':
        pin['invented_approval'] = True
    else:
        key = str(Path(pin['root'])/'bundle.json')
        if fault == 'extra_input':
            graph['input_bindings'][key+'extra'] = pin['bundle_sha256']
        elif fault == 'missing_input':
            graph['input_bindings'].pop(key)
        else:
            packet['provenance']['source_bindings'][key] = '0'*64
            packet = sealed(packet)
            graph['inventory_sha256'] = packet['sha256']
    with pytest.raises(ValueError):
        select(packet, rebase(graph, base))


@pytest.mark.parametrize('logical', ['sample.json', 'supervision/label.json', 'inputs/rgb.png',
                                    'supervision/query_trace.json'])
def test_inventory_logical_claims_must_match_compact_pins(case, logical):
    packet, graph, base, _ = case
    packet['records'][1]['provenance']['logical_files'][logical]['sha256'] = '0'*64
    packet = sealed(packet)
    graph['inventory_sha256'] = packet['sha256']
    with pytest.raises(ValueError):
        select(packet, rebase(graph, base))


def test_unknown_or_disguised_schema_and_stale_seals(case):
    packet, graph, _, _ = case
    for value in (dict(graph, schema='other'), dict(graph, schema=policy.PAIR_SCHEMA),
                  dict(graph, extra_approval=True)):
        with pytest.raises(ValueError):
            select(packet, sealed(value))
    graph['complete'] = False
    with pytest.raises(ValueError, match='Stale sealed'):
        select(packet, graph)
    with pytest.raises(ValueError):
        select(packet, None)


def test_actual_byte_tamper_detected_not_just_mapping_seals(tmp_path):
    packet, graph, _, sources = fixture_graph(tmp_path)
    path = payload(sources[1][0], 'supervision/label.json')
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError):
        select(packet, graph)


def test_readonly_no_native_array_or_audit_replay_and_no_pair_recompute(case):
    packet, graph, _, _ = case
    active = [True]
    reads = []
    def audit(event, args):
        if not active[0]:
            return
        if event in ('os.mkdir', 'os.remove', 'os.rename', 'os.rmdir', 'os.link', 'os.symlink', 'os.truncate'):
            raise AssertionError('Adapter attempted write: '+event)
        if event == 'open':
            path, _, flags = args
            assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            assert not str(path).endswith(('.npy', '.ghn'))
            reads.append(str(path))
    sys.addaudithook(audit)
    try:
        result = select(packet, graph)
    finally:
        active[0] = False
    assert reads and result['graph_adapter']['aggregate_accounting_revalidated'] is True
    assert result['scope']['source_files_reverified'] is False  # Original inventory-wide scope unchanged.
    source_text = Path(adapter.__file__).read_text(encoding='utf-8')
    assert 'build_graph(' not in source_text and 'audit_capture(' not in source_text
    assert 'monkeypatch' not in source_text and 'setattr(' not in source_text
    assert 'exec(' not in source_text
    assert index._sha(policy.__file__) == adapter.POLICY_SHA256
    assert index._sha(compact_index.__file__) == adapter.GRAPH_ADAPTER_SHA256
    assert result['graph_adapter']['implementation_bindings'][str(Path(adapter.__file__).resolve())] == index._sha(adapter.__file__)
