"""Tiny file-backed metadata fixtures; no renderer, native audit or dataset writes."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from . import inventory_join as join
from ..native_capture_v4.test_io import link_dir, unlink_dir


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, allow_nan=False).encode())
    return join._file_sha(path)


def seal(value):
    value.pop('sha256', None)
    value['sha256'] = join._digest(value)
    return value


def publish(component, *, update_counts=True):
    if update_counts:
        component.packet['counts'].update(join._counts(component.packet['records'],
                                                      component.packet['scope']['explicit_receipts']))
    seal(component.packet)
    digest = write(component.path, component.packet)
    prefix = component.prefix
    component.report.update({prefix+'_path': str(component.path), prefix+'_file_sha256': digest,
                             prefix+'_sha256': component.packet['sha256']})
    report_sha = write(component.report_path, component.report)
    component.pin = join.AttestedInventoryPin(str(component.path), digest, str(component.report_path),
        report_sha, 'independent_fixture_replay_complete_observed_only',
        'explicit test caller; NOT a signature or a real native audit',
        'fixture authority claims saved-buffer replay; join must not assert it reran that replay')
    return component


def make_component(tmp_path, name='a', *, empty=False, original=False, family='seed73_full', compact=False):
    directory = tmp_path/name
    directory.mkdir()
    shared = tmp_path/'shared-source.json'
    if not shared.exists():
        write(shared, {'geometry': 'opaque fixture, no native proof'})
    producer = tmp_path/'producer.txt'
    if not producer.exists():
        producer.write_text('external fixture producer', encoding='utf-8')
    capture = directory/'capture'
    sample = capture/'view01'
    sample.mkdir(parents=True)
    paths = {k: directory/k for k in ('receipt.json', 'plan.json')}
    paths.update({'request.json': capture/'request.json', 'result.json': capture/'result.json',
                  'sample.json': sample/'sample.json'})
    bound = {str(p): write(p, {'opaque_fixture': k, 'component': name}) for k, p in paths.items()}
    bound.update({str(shared): join._file_sha(shared), str(producer): join._file_sha(producer)})
    if compact:
        marker = sample/'bundle.json'
        bound[str(marker)] = write(marker, {'opaque_fixture_marker': True})
    context = {'schema': 'neutral_fixture' if original else 'legacy_generated_fixture', 'pose': [-0.0, 2, 3.0]}
    context_key = join._digest(context)
    scene = {'static_scene_sha256': context_key, 'plant_to_world' if original else 'generated_plant_to_world': [1, 2, 3]}
    camera = {'actual_camera_pose': [name, 0.0, 1.0]}
    scene_sha, camera_sha = join._digest(scene), join._digest(camera)
    cap = family+'/SubStem_41'
    row = dict(sample_id=name+'_sample', candidate_id='view01', sample_path=str(sample),
        capture_path=str(capture), source_family=family, original_donor_family=family,
        target_id='original_target' if original else 'generated_'+name,
        source_target=cap, conservative_view_cap_group=cap,
        split='train', resolution=[1696, 816], decision='accept_strict_automatic_annotation_candidate',
        encoded_rgb_sha256='a'*64, decoded_rgb_sha256='b'*64, geometry_sha256='c'*64,
        scene_identity_basis=scene, scene_sha256=scene_sha, camera_identity_basis=camera,
        camera_sha256=camera_sha, scene_camera_sha256=join._digest(dict(scene=scene_sha, camera=camera_sha)),
        annotation_review=dict(passed=True, evidence_id=bound[str(paths['receipt.json'])],
            evidence_basis='old original replay' if original else 'old structural checks',
            audit_execution_independently_verified=original),
        provenance=dict(receipt_path=str(paths['receipt.json']), receipt_sha256=bound[str(paths['receipt.json'])],
            plan_path=str(paths['plan.json']), plan_sha256=bound[str(paths['plan.json'])],
            request_sha256=bound[str(paths['request.json'])], result_sha256=bound[str(paths['result.json'])],
            future_opaque_provenance={'not_rewritten': [1, -0.0, False]}),
        training_approved=False, source_cap_reset=False, depth_recomputed=False)
    if original:
        row['source_kind'] = 'original_native'
    coverage = dict(receipt_path=str(paths['receipt.json']), receipt_sha256=bound[str(paths['receipt.json'])],
        capture_path=str(capture), storage_root=str(capture), plan_path=str(paths['plan.json']),
        planned_rows=2, captured_rows=0 if empty else 1, noncaptured_rows=2 if empty else 1,
        untouched_held_targets=['never-rendered'])
    packet = dict(schema=join.INPUT_SCHEMA, state=join.INPUT_STATE,
        records=[] if empty else [row], scene_contexts={} if empty else {context_key: context}, counts={},
        scope=dict(global_complete=False, explicit_receipts=[coverage],
            extra_observed_roots=[dict(path=str(tmp_path/'unindexed'), complete=False,
                                      status='unindexed_no_receipts')],
            audit_execution_independently_verified=original,
            audit_decision_counts_basis='unchanged prior adapter basis'),
        exact_duplicates={'opaque_existing_diagnostic': True},
        provenance=dict(source_bindings=bound, implementation_bindings={str(producer): join._file_sha(producer)},
                        read_only_source_roots=[]), **{key: False for key in join.FALSE_FLAGS})
    if original:
        packet['adapter_policy'] = 'original_fixture_v2'
        packet['counts']['source_kinds'] = {} if empty else {'original_native': 1}
    report = dict(state='independent_fixture_replay_complete_observed_only', training_approved=False,
                  global_complete=False, replay_records=[{'basis': 'external fixture claim only'}],
                  implementation_bindings={str(producer): join._file_sha(producer)})
    return publish(SimpleNamespace(path=tmp_path/(name+'_inventory.json'), packet=packet, report=report,
        report_path=tmp_path/(name+'_replay.json'), prefix='observed_inventory' if original else 'inventory',
        shared=shared, sample=sample))


def test_byte_exact_rows_contexts_metadata_audit_basis_and_new_coverage(tmp_path):
    a = make_component(tmp_path)
    b = make_component(tmp_path, 'b', original=True)
    z = make_component(tmp_path, 'z', empty=True, compact=True)
    originals = {p: p.read_bytes() for p in (a.path, b.path, z.path, a.report_path)}
    packet = join.join_inventories([b.pin, z.pin, a.pin])
    assert packet == join.join_inventories([a.pin, b.pin, z.pin])
    assert packet['schema'] == join.SCHEMA != join.INPUT_SCHEMA
    assert packet['counts']['captured_rows'] == 2 and packet['counts']['receipts'] == 3
    views = {r['sample_id']: r for r in packet['records']}
    for source in (a, b, z):
        detail = next(c for c in packet['components'] if c['pin']['inventory_path'] == str(source.path))
        assert join._canonical([views[k] for k in detail['row_ids']]) == join._canonical(source.packet['records'])
        assert detail['rows_sha256'] == join._digest(source.packet['records'])
        assert join._canonical({k: packet['scene_contexts'][k] for k in detail['scene_context_ids']}) == join._canonical(source.packet['scene_contexts'])
        assert detail['inventory_metadata'] == {k: v for k, v in source.packet.items() if k not in ('records', 'scene_contexts')}
        assert detail['external_execution_claims_independently_verified_by_joiner'] is False
        assert detail['attestation_claims']['replay_records'] == source.report['replay_records']
    assert 'source_kind' not in views['a_sample']
    assert views['b_sample']['source_kind'] == 'original_native'
    assert not views['a_sample']['annotation_review']['audit_execution_independently_verified']
    assert views['b_sample']['annotation_review']['audit_execution_independently_verified']
    assert all(packet[k] is False for k in join.FALSE_FLAGS)
    assert packet['scope']['audit_execution_independently_verified'] is False
    assert not packet['scope']['physical_contexts_normalized']
    assert len(packet['scope']['extra_observed_roots']) == 3
    assert sum(e['captured_rows'] == 0 for e in packet['scope']['explicit_receipts']) == 1
    needed = join.graph_requirement(packet)
    assert needed['sample_ids'] == ['a_sample', 'b_sample'] and needed['total_pairs'] == 1
    assert needed['inventory_sha256'] == packet['sha256'] not in (a.packet['sha256'], b.packet['sha256'])
    assert needed['graph_execution_verified'] is False
    for p, raw in originals.items():
        assert p.read_bytes() == raw


def test_one_initial_and_final_hash_per_unique_union_file(tmp_path, monkeypatch):
    a, b = make_component(tmp_path), make_component(tmp_path, 'b')
    calls = []
    original = join.io.Reader._hash
    def counted(reader, path, phase):
        calls.append((str(path), phase))
        return original(reader, path, phase)
    monkeypatch.setattr(join.io.Reader, '_hash', counted)
    packet = join.join_inventories([a.pin, b.pin])
    stats = packet['provenance']['verification']
    assert stats['initial_hashes'] == stats['final_hashes'] == stats['unique_files']
    assert stats['metadata_hashes'] == 4
    assert stats['state'] == 'finished' and not stats['stat_only_byte_acceptance']
    assert calls.count((str(a.shared), 'initial')) == calls.count((str(a.shared), 'final')) == 1
    assert len(calls) == stats['unique_files']*2


@pytest.mark.parametrize('field,value', [(k, True) for k in join.FALSE_FLAGS] +
    [('schema', join.SCHEMA), ('state', 'release_approved')])
def test_unsafe_packet_contract(tmp_path, field, value):
    a = make_component(tmp_path)
    a.packet[field] = value
    publish(a)
    with pytest.raises(ValueError):
        join.join_inventories([a.pin])


@pytest.mark.parametrize('field,value', [('training_approved', True), ('source_cap_reset', True),
    ('depth_recomputed', True), ('resolution', [848, 408]), ('decision', 'approved'),
    ('source_target', 'other/SubStem_41'), ('conservative_view_cap_group', 'new_budget'),
    ('original_donor_family', 'other'), ('split', 'unknown'), ('scene_sha256', '0'*64),
    ('scene_camera_sha256', '0'*64)])
def test_unsafe_row_contract(tmp_path, field, value):
    a = make_component(tmp_path)
    a.packet['records'][0][field] = value
    publish(a)
    with pytest.raises(ValueError):
        join.join_inventories([a.pin])


@pytest.mark.parametrize('field,value', [('external_authority', ''), ('audit_basis', '  '),
    ('attestation_state', 'different'), ('inventory_file_sha256', '0'*64), ('attestation_sha256', '0'*64)])
def test_explicit_attestation_pins_and_trust_required(tmp_path, field, value):
    a = make_component(tmp_path)
    with pytest.raises(ValueError):
        join.join_inventories([replace(a.pin, **{field: value})])


@pytest.mark.parametrize('mutation', ['wrong_inventory_pin', 'wrong_content_seal', 'wrong_path',
    'missing_join', 'partial_join', 'self', 'failed', 'approval', 'failure_field', 'ambiguous_join'])
def test_unrelated_incomplete_or_failed_attestation(tmp_path, mutation):
    a = make_component(tmp_path)
    if mutation == 'wrong_inventory_pin': a.report['inventory_file_sha256'] = '0'*64
    if mutation == 'wrong_content_seal': a.report['inventory_sha256'] = '0'*64
    if mutation == 'wrong_path': a.report['inventory_path'] = str(tmp_path/'not-inventory.json')
    if mutation == 'missing_join':
        for k in ('inventory_path', 'inventory_file_sha256', 'inventory_sha256'): a.report.pop(k)
    if mutation == 'partial_join': a.report.pop('inventory_file_sha256')
    if mutation == 'approval': a.report['training_approved'] = True
    if mutation == 'failure_field': a.report['failures'] = ['incomplete replay']
    if mutation == 'ambiguous_join': a.report['observed_inventory_path'] = str(a.path)
    if mutation == 'failed': a.report['state'] = 'failed_independent_replay'
    pin = replace(a.pin, attestation_sha256=write(a.report_path, a.report))
    if mutation == 'failed': pin = replace(pin, attestation_state=a.report['state'])
    if mutation == 'self': pin = replace(pin, attestation_path=str(a.path), attestation_sha256=pin.inventory_file_sha256)
    with pytest.raises(ValueError):
        join.join_inventories([pin])


@pytest.mark.parametrize('mutation', ['duplicate_component', 'capture', 'row_id', 'split', 'missing_coverage',
    'missing_rows', 'count', 'context', 'missing_context', 'row_receipt', 'missing_plan_pin'])
def test_conflicts_and_missing_coverage(tmp_path, mutation):
    a, b = make_component(tmp_path), make_component(tmp_path, 'b')
    if mutation == 'duplicate_component':
        with pytest.raises(ValueError, match='Duplicate inventory'): join.join_inventories([a.pin, a.pin])
        return
    if mutation == 'capture': b.packet['scope']['explicit_receipts'][0]['capture_path'] = a.packet['scope']['explicit_receipts'][0]['capture_path']
    if mutation == 'row_id': b.packet['records'][0]['sample_id'] = a.packet['records'][0]['sample_id']
    if mutation == 'split': b.packet['records'][0]['split'] = 'test'
    if mutation == 'missing_coverage': b.packet['scope']['explicit_receipts'] = []
    if mutation == 'missing_rows': b.packet['records'] = []
    if mutation == 'count': b.packet['counts']['captured_rows'] = 99
    if mutation == 'context': next(iter(b.packet['scene_contexts'].values()))['pose'] = [99]
    if mutation == 'missing_context': b.packet['scene_contexts'] = {}
    if mutation == 'row_receipt': b.packet['records'][0]['provenance']['receipt_sha256'] = '0'*64
    if mutation == 'missing_plan_pin': del b.packet['provenance']['source_bindings'][b.packet['scope']['explicit_receipts'][0]['plan_path']]
    publish(b, update_counts=mutation != 'count')
    with pytest.raises(ValueError):
        join.join_inventories([a.pin, b.pin])


@pytest.mark.parametrize('phase', ['initial', 'final'])
def test_full_content_tamper_fails_even_same_size(tmp_path, monkeypatch, phase):
    a = make_component(tmp_path)
    def change(): a.shared.write_bytes(b'X'*a.shared.stat().st_size)
    if phase == 'initial': change()
    else:
        original = join.io.Reader.finish
        def finish(reader):
            change()
            return original(reader)
        monkeypatch.setattr(join.io.Reader, 'finish', finish)
    with pytest.raises(ValueError, match='Changed bound|Source changed'):
        join.join_inventories([a.pin])


def test_conflicting_alias_hash_rejected_before_cached_return(tmp_path):
    a = make_component(tmp_path)
    alias = tmp_path/'alias'
    link_dir(tmp_path/'a', alias)
    try:
        original = next(iter(a.packet['provenance']['source_bindings']))
        a.report['source_bindings'] = {str(alias/Path(original).name): '0'*64}
        publish(a)
        with pytest.raises(ValueError): join.join_inventories([a.pin])
    finally:
        unlink_dir(alias)


def test_second_component_conflict_fails_before_cached_file_return(tmp_path, monkeypatch):
    a, b = make_component(tmp_path), make_component(tmp_path, 'b')
    b.packet['provenance']['source_bindings'][str(a.shared)] = 'f'*64
    publish(b)
    calls = []
    original = join.io.Reader._hash
    def counted(reader, path, phase):
        calls.append((str(path), phase))
        return original(reader, path, phase)
    monkeypatch.setattr(join.io.Reader, '_hash', counted)
    with pytest.raises(ValueError, match='Conflicting original hashes'):
        join.join_inventories([a.pin, b.pin])
    assert calls.count((str(a.shared), 'initial')) == 1


def test_same_hash_alias_is_hashed_once_but_both_routes_guarded(tmp_path):
    a = make_component(tmp_path)
    alias = tmp_path/'source_alias'
    link_dir(tmp_path/'a', alias)
    try:
        source = next(iter(a.packet['provenance']['source_bindings']))
        alias_path = str(alias/Path(source).name)
        a.packet['provenance']['source_bindings'][alias_path] = a.packet['provenance']['source_bindings'][source]
        publish(a)
        packet = join.join_inventories([a.pin])
        assert alias_path not in packet['provenance']['source_bindings']
        assert alias_path in packet['components'][0]['inventory_metadata']['provenance']['source_bindings']
        stats = packet['provenance']['verification']
        assert stats['initial_hashes'] == stats['final_hashes'] == stats['unique_files']
        assert stats['original_paths'] > stats['unique_files']
    finally:
        unlink_dir(alias)


@pytest.mark.parametrize('case', ['retarget', 'new_symlink', 'noncanonical'])
def test_original_routes_checked_again_after_full_hash_sweep(tmp_path, monkeypatch, case):
    a = make_component(tmp_path)
    first, second = tmp_path/'route-a', tmp_path/'route-b'
    first.mkdir(); second.mkdir()
    for directory in (first, second): (directory/'asset').write_bytes(b'identical')
    route = tmp_path/'route'
    if case == 'new_symlink':
        route.mkdir()
        (route/'asset').write_bytes(b'identical')
    else:
        link_dir(first, route)
    if case == 'noncanonical':
        (first/'child').mkdir(); (second/'child').mkdir()
        raw = str(route/'child'/'..'/'asset')
    else:
        raw = str(route/'asset')
    a.packet['provenance']['source_bindings'][raw] = join._file_sha(route/'asset')
    publish(a)
    original = join.io.Reader.finish
    def finish(reader):
        if case == 'new_symlink':
            (route/'asset').unlink(); route.rmdir()
        else:
            unlink_dir(route)
        link_dir(second, route)
        return original(reader)
    monkeypatch.setattr(join.io.Reader, 'finish', finish)
    try:
        with pytest.raises(ValueError, match='target changed|topology changed'):
            join.join_inventories([a.pin])
    finally:
        unlink_dir(route)


@pytest.mark.parametrize('marker,late', [('failure.json', False), ('bundle.json', False),
                                      ('failure.json', True), ('bundle.json', True)])
def test_unbound_markers_do_not_change_attested_raw_contract(tmp_path, monkeypatch, marker, late):
    a = make_component(tmp_path)
    path = (a.sample if marker == 'bundle.json' else a.sample.parent)/marker
    if not late: write(path, {'unexpected': True})
    else:
        original = join.io.Reader.finish
        def finish(reader):
            original(reader)
            write(path, {'unexpected': True})
        monkeypatch.setattr(join.io.Reader, 'finish', finish)
    with pytest.raises(ValueError): join.join_inventories([a.pin])


def test_pinned_compact_marker_not_relabelled_or_requalified(tmp_path):
    a = make_component(tmp_path, compact=True)
    packet = join.join_inventories([a.pin])
    assert packet['records'] == a.packet['records']
    assert packet['compact_qualification_granted'] is False
    assert str(a.sample/'bundle.json') in packet['provenance']['source_bindings']
    (a.sample/'bundle.json').unlink()
    with pytest.raises(ValueError): join.join_inventories([a.pin])


def test_omitted_probe_declared_unindexed_never_counted_or_scanned(tmp_path):
    a = make_component(tmp_path)
    omitted = tmp_path/'v5_probe'
    omitted.mkdir()
    write(omitted/'opaque.json', {'claimed_frames': 4})
    declaration = join.UnindexedRoot(str(omitted), 'V5 diagnostic only; no adapter admitted')
    packet = join.join_inventories([a.pin], unindexed_roots=[declaration])
    assert packet['counts']['captured_rows'] == 1
    extra = packet['scope']['extra_observed_roots'][-1]['declaration']
    assert extra['path'] == str(omitted) and extra['complete'] is False
    assert extra['recursively_scanned'] is False
    assert str(omitted/'opaque.json') not in packet['provenance']['source_bindings']
    with pytest.raises(ValueError, match='overlaps'):
        join.write_join(omitted/'join.json', [a.pin], unindexed_roots=[declaration])
    with pytest.raises(ValueError, match='Duplicate|duplicate'):
        join.join_inventories([a.pin], unindexed_roots=[declaration, declaration])
    with pytest.raises(ValueError, match='overlaps'):
        join.join_inventories([a.pin], unindexed_roots=[join.UnindexedRoot(str(a.sample.parent), 'wrong')])


@pytest.mark.parametrize('decision', ['hold_visual_clarity', 'exclude_geometry_or_visibility'])
def test_hold_and_excluded_rows_survive_unchanged_in_graph_requirement(tmp_path, decision):
    a = make_component(tmp_path)
    a.packet['records'][0]['decision'] = decision
    a.packet['records'][0]['annotation_review']['passed'] = False
    publish(a)
    packet = join.join_inventories([a.pin])
    assert packet['records'] == a.packet['records']
    assert join.graph_requirement(packet)['sample_ids'] == [a.packet['records'][0]['sample_id']]
    a.packet['records'][0]['annotation_review']['passed'] = True
    publish(a)
    with pytest.raises(ValueError, match='annotation review'):
        join.join_inventories([a.pin])


def test_old_selector_cannot_accept_new_join_with_old_graph(tmp_path):
    from . import provisional, test_provisional
    a = make_component(tmp_path)
    packet = join.join_inventories([a.pin])
    graph = test_provisional.coverage(a.packet)
    with pytest.raises(ValueError, match='Unapproved observed inventory'):
        provisional.select_observed_train(packet, {'seed73_full': 'train'}, graph)
    assert join.graph_requirement(packet)['inventory_sha256'] != graph['inventory_sha256']


def test_fresh_writer_and_no_source_overwrite(tmp_path):
    a = make_component(tmp_path)
    destination = tmp_path/'join.json'
    packet = join.write_join(destination, [a.pin])
    assert json.loads(destination.read_bytes()) == packet
    with pytest.raises(ValueError, match='Fresh output'): join.write_join(destination, [a.pin])
    with pytest.raises(ValueError, match='overlaps'): join.write_join(a.sample/'new.json', [a.pin])
    assert not (a.sample/'new.json').exists()


def test_changed_seal_duplicate_json_keys_and_relative_inputs(tmp_path):
    a = make_component(tmp_path)
    packet = deepcopy(a.packet); packet['counts']['captured_rows'] = 999
    pin = replace(a.pin, inventory_file_sha256=write(a.path, packet))
    a.report['inventory_file_sha256'] = pin.inventory_file_sha256
    pin = replace(pin, attestation_sha256=write(a.report_path, a.report))
    with pytest.raises(ValueError, match='seal'): join.join_inventories([pin])
    a.path.write_bytes(b'{"schema":1,"schema":2}')
    with pytest.raises(ValueError, match='Duplicate JSON key'):
        join.join_inventories([replace(a.pin, inventory_file_sha256=join._file_sha(a.path))])
    with pytest.raises(ValueError, match='absolute'):
        join.join_inventories([replace(a.pin, inventory_path='relative.json')])


def test_no_audit_replay_import_or_native_entrypoint():
    import ast
    tree = ast.parse(Path(join.__file__).read_text(encoding='utf-8'))
    imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    assert imports == ['collections', 'copy', 'dataclasses', 'pathlib', 'native_capture_v4', 'native_capture_v3']
    called = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not {'audit_capture', 'collect', 'Popen', 'SimulationApp', 'build_inventory'} & called
