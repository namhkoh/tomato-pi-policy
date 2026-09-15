"""Synthetic joined-schema/physical-key/parity tests; no real native jobs/audits."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path

import numpy as np
import pytest

from . import provisional_join as adapter, provisional as policy, inventory_join as join
from . import compact_image_index as compact, near_image_index as index
from .bundle import SampleReader, digest
from . import test_provisional as old, test_inventory_join as joined_fixture
from .test_compact_image_index import source, payload


def matrix(x=0.0):
    return [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0], [x, 0.0, 0.0, 1.0]]


def legacy_context():
    return dict(background_scene_asset_sha256='1'*64, background_asset_content_sha256='2'*64,
        background_asset_count=2, material_texture_content_sha256='3'*64, material_texture_count=1,
        population_placements=[dict(plant_root='/World/Plant', geometry_content_sha256='4'*64, asset_count=2)],
        excluded_external_prop_roots=[], replaced_plant_root='/World/Plant', generated_geometry_sha256='5'*64,
        renderer_settings={'/rtx/pathtracing/spp': 64}, original_lighting={'day': 1, 'minutes': 600},
        original_scene_counts={'components': 2, 'component_plants': 1}, original_renderer='RealTimePathTracing')


def camera(x=0.0):
    return dict(resolution=[1696, 816], intrinsics=[[941.0, 0.0, 848.0], [0.0, 941.0, 408.0], [0.0, 0.0, 1.0]],
        clipping_range_m=[0.04, 10.0], focal_length_mm=10.0, apertures_mm=[18.0, 8.66],
        aperture_offsets_mm=[0.0, 0.0], depth_convention='optical_axis_z_metres_not_ray_range',
        crop_resize=None, camera_to_world=matrix(x), camera_to_head=matrix())


def physical_row(name, *, neutral=False, x=0.0):
    context = adapter._neutral_basis(legacy_context()) if neutral else legacy_context()
    context_sha = join._digest(context)
    scene = dict(static_scene_sha256=context_sha, actual_robot_root_to_world=matrix(),
                 actual_robot_joint_degrees={'head_0': 0.0, 'head_1': 0.0})
    scene['plant_to_world' if neutral else 'generated_plant_to_world'] = matrix()
    optical = camera(x)
    row = old.row(name, geometry_sha256='5'*64, scene_identity_basis=scene,
        camera_identity_basis=optical, scene_sha256=join._digest(scene), camera_sha256=join._digest(optical),
        provenance={'scene_identity_policy': adapter.SCENE_POLICY} if neutral else {})
    row['scene_camera_sha256'] = join._digest(dict(scene=row['scene_sha256'], camera=row['camera_sha256']))
    if neutral:
        row['source_kind'] = 'original_native'
    return row, context


def physical_packet():
    records, contexts = [], {}
    for name, neutral, x in [('a', False, 0.0), ('b', True, 0.0), ('c', False, 1.0)]:
        row, context = physical_row(name, neutral=neutral, x=x)
        records.append(row); contexts[join._digest(context)] = context
    return dict(records=records, scene_contexts=contexts)


def reseal_physical(packet):
    # Fixture mutation helper updates all content keys, challenging more than a stale seal.
    contexts = {}
    for row in packet['records']:
        context = packet['scene_contexts'][row['scene_identity_basis']['static_scene_sha256']]
        key = join._digest(context)
        contexts[key] = context
        row['scene_identity_basis']['static_scene_sha256'] = key
        row['scene_sha256'] = join._digest(row['scene_identity_basis'])
        row['camera_sha256'] = join._digest(row['camera_identity_basis'])
        row['scene_camera_sha256'] = join._digest(dict(scene=row['scene_sha256'], camera=row['camera_sha256']))
    packet['scene_contexts'] = contexts


def test_cross_schema_mapping_no_input_mutation_or_zero_distance_rgb_diversity():
    packet = physical_packet()
    before = join._canonical(packet)
    derived = adapter.physical_comparison_keys(packet)
    a, b, c = (derived['rows'][k] for k in ('a', 'b', 'c'))
    assert packet['records'][0]['scene_sha256'] != packet['records'][1]['scene_sha256']
    assert packet['records'][0]['decoded_rgb_sha256'] != packet['records'][1]['decoded_rgb_sha256']
    assert (a['scene_sha256'], a['camera_sha256']) == (b['scene_sha256'], b['camera_sha256'])
    assert c['camera_sha256'] != a['camera_sha256']
    views = {r['sample_id']: r for r in packet['records']}
    result = adapter._account(views, [], [], derived['rows'])
    assert [r['sample_id'] for r in result['selected']] == ['a', 'c']
    assert result['excluded']['b'] == ['duplicate_view']
    assert join._canonical(packet) == before
    assert derived['input_rows_modified'] is derived['input_contexts_modified'] is False
    assert not derived['pose_tolerance_applied']


def test_render_settings_do_not_make_new_pose_identity_but_remain_in_input():
    packet = physical_packet()
    before = adapter.physical_comparison_keys(packet)
    context = packet['scene_contexts'][packet['records'][0]['scene_identity_basis']['static_scene_sha256']]
    context['renderer_settings']['/rtx/pathtracing/spp'] = 8
    reseal_physical(packet)
    after = adapter.physical_comparison_keys(packet)
    assert after['rows']['a']['scene_sha256'] == before['rows']['a']['scene_sha256']
    assert context['renderer_settings']['/rtx/pathtracing/spp'] == 8


@pytest.mark.parametrize('field', ['background_scene_asset_sha256', 'background_asset_content_sha256',
    'material_texture_content_sha256', 'population_placements', 'original_lighting', 'original_renderer'])
def test_material_environment_placement_and_light_are_not_discarded(field):
    packet = physical_packet()
    before = adapter.physical_comparison_keys(packet)['rows']['a']['scene_sha256']
    context = packet['scene_contexts'][packet['records'][0]['scene_identity_basis']['static_scene_sha256']]
    if field.endswith('sha256'): context[field] = '9'*64
    elif field == 'population_placements': context[field][0]['geometry_content_sha256'] = '9'*64
    elif field == 'original_lighting': context[field]['minutes'] += 1
    else: context[field] = 'different_renderer'
    reseal_physical(packet)
    assert adapter.physical_comparison_keys(packet)['rows']['a']['scene_sha256'] != before


@pytest.mark.parametrize('mutation', ['unknown_context', 'unknown_neutral', 'missing_mount', 'missing_joints',
    'unknown_camera', 'wrong_depth', 'wrong_resolution', 'missing_light', 'geometry_mismatch', 'bad_matrix',
    'missing_settings', 'wrong_legacy_kind', 'missing_neutral_policy', 'missing_population', 'unknown_pose'])
def test_unsupported_or_missing_physical_evidence_rejects(mutation):
    packet = physical_packet()
    row = packet['records'][0]
    context = packet['scene_contexts'][row['scene_identity_basis']['static_scene_sha256']]
    if mutation == 'unknown_context': context['unknown_foreground'] = 'unmapped'
    if mutation == 'unknown_neutral':
        neutral = packet['scene_contexts'][packet['records'][1]['scene_identity_basis']['static_scene_sha256']]
        neutral['schema'] = 'unknown.v99'
    if mutation == 'missing_mount': del row['camera_identity_basis']['camera_to_head']
    if mutation == 'missing_joints': row['scene_identity_basis']['actual_robot_joint_degrees'] = {}
    if mutation == 'unknown_camera': row['camera_identity_basis']['future_camera_control'] = 1
    if mutation == 'wrong_depth': row['camera_identity_basis']['depth_convention'] = 'ray_range'
    if mutation == 'wrong_resolution': row['camera_identity_basis']['resolution'] = [848, 408]
    if mutation == 'missing_light': context['original_lighting'] = None
    if mutation == 'geometry_mismatch': row['geometry_sha256'] = '0'*64
    if mutation == 'bad_matrix': row['scene_identity_basis']['actual_robot_root_to_world'] = [[0.0]]
    if mutation == 'missing_settings': del context['renderer_settings']
    if mutation == 'wrong_legacy_kind': row['source_kind'] = 'original_native'
    if mutation == 'missing_neutral_policy': packet['records'][1]['provenance'].clear()
    if mutation == 'missing_population': context['population_placements'] = []
    if mutation == 'unknown_pose': row['scene_identity_basis']['synthetic_noise_epoch'] = 1
    reseal_physical(packet)
    with pytest.raises((ValueError, KeyError)):
        adapter.physical_comparison_keys(packet)


@pytest.mark.parametrize('mode', ['empty', 'cap', 'balance', 'heldout', 'cross_split', 'holds',
                                  'overlay', 'control', 'rgb_alias', 'near_chain'])
def test_accounting_exact_frozen_policy_parity(mode):
    rows = [old.row('a'), old.row('b'), old.row('c')]
    overlays, edges, splits = [], [], {'A': 'train'}
    if mode == 'empty': rows = []
    if mode == 'cap': rows = [old.row(f'row{i:03d}') for i in range(30)]
    if mode == 'balance':
        rows = [old.row(str(i), family='A' if i % 2 else 'B', target=('A' if i % 2 else 'B')+'/stem'+str(i%3)) for i in range(80)]
        splits['B'] = 'train'
    if mode in ('heldout', 'cross_split'):
        rows[1] = old.row('b', family='V', split='validation'); splits['V'] = 'validation'
        if mode == 'cross_split': edges = [['a', 'b']]
    if mode == 'holds': rows[1]['decision'] = 'hold_visual_clarity'
    if mode == 'overlay': overlays = [dict(sample_id='a', evidence_id='hold', visual_hold=True)]
    if mode == 'control': overlays = [dict(sample_id='b', evidence_id='control', control_role='control')]
    if mode == 'rgb_alias': rows[1]['decoded_rgb_sha256'] = rows[0]['decoded_rgb_sha256']
    if mode == 'near_chain': edges = [['a', 'b'], ['b', 'c']]
    packet = old.inventory(rows)
    expected = policy.select_observed_train(packet, splits, old.coverage(packet, edges), overlays=overlays)
    views = adapter._views(packet, splits)
    actual = adapter._account(views, overlays, edges, {r['sample_id']: r for r in rows})
    assert actual == {k: expected[k] for k in actual}


def _function(path, name):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_accounting_ast_parity_only_two_explicit_changed_inputs():
    original = _function(policy.__file__, '_select')
    copied = _function(adapter.__file__, '_account')
    start = next(i for i, n in enumerate(original.body) if isinstance(n, ast.Assign) and
                 any(isinstance(t, ast.Name) and t.id == 'aliases' for t in n.targets))
    end = next(i for i, n in enumerate(original.body) if isinstance(n, ast.Assign) and
               any(isinstance(t, ast.Name) and t.id == 'chosen' for t in n.targets))
    expected = ast.Module(body=deepcopy(original.body[start:end+1]), type_ignores=[])
    actual = ast.Module(body=deepcopy(copied.body[1:-1]), type_ignores=[])
    class RestoreInputs(ast.NodeTransformer):
        poses = edges = 0
        def visit_Assign(self, node):
            if any(isinstance(t, ast.Name) and t.id == 'pose' for t in node.targets):
                assert ast.unparse(node.value) == "(physical_keys[key]['scene_sha256'], physical_keys[key]['camera_sha256'])"
                node.value = ast.parse("(row['scene_sha256'], row['camera_sha256'])", mode='eval').body
                self.poses += 1
            return self.generic_visit(node)
        def visit_Call(self, node):
            if isinstance(node.func, ast.Attribute) and ast.unparse(node.func) == 'duplicates.edges' and len(node.args) == 1 and isinstance(node.args[0], ast.Name) and node.args[0].id == 'near_edges':
                node.args[0] = ast.parse('_pair_edges(coverage, inventory_sha, views)', mode='eval').body
                self.edges += 1
            return self.generic_visit(node)
    transform = RestoreInputs(); transform.visit(actual)
    assert (transform.poses, transform.edges) == (1, 1)
    assert ast.dump(actual) == ast.dump(expected)


def test_neutral_mapping_ast_identical_to_frozen_original_adapter():
    path = Path(adapter.__file__).with_name('original_inventory.py')
    assert join._file_sha(path) == adapter.NEUTRAL_SOURCE_SHA256
    original = _function(path, '_neutral_basis')
    copied = _function(adapter.__file__, '_neutral_basis')
    assert ast.dump(ast.Module(body=original.body[1:], type_ignores=[])) == ast.dump(ast.Module(body=copied.body[1:], type_ignores=[]))


def fixture_graph(root):
    components, image_pins = [], []
    for i, name in enumerate(('a', 'b', 'c')):
        component = joined_fixture.make_component(root, name, original=(i == 1), family='A')
        image_pin, raw_root = source(root/'pixels', name+'_sample', np.full((816, 1696, 3), [20, 22, 200][i], np.uint8), (800.5, 400.5))
        folder = raw_root if i == 0 else Path(image_pin.root)
        row = component.packet['records'][0]
        physical, context = physical_row(name+'_sample', neutral=(i == 1), x=1.0 if i == 2 else 0.0)
        for key in ('scene_identity_basis', 'camera_identity_basis', 'scene_sha256', 'camera_sha256', 'scene_camera_sha256', 'geometry_sha256'):
            row[key] = physical[key]
        row.update(sample_path=str(folder), capture_path=str(folder.parent), candidate_id=folder.name,
                   context_id='original-target' if i == 1 else 'generated-target', target_id='synthetic/target',
                   encoded_rgb_sha256=image_pin.encoded_rgb_sha256, decoded_rgb_sha256=image_pin.decoded_rgb_sha256)
        row['annotation_review']['method'] = 'automatic'
        row['provenance'].update(physical['provenance'])
        bound = component.packet['provenance']['source_bindings']
        for key in ('request', 'result'):
            path = folder.parent/(key+'.json')
            digest_value = joined_fixture.write(path, {'fixture': name, 'kind': key})
            bound[str(path)] = digest_value; row['provenance'][key+'_sha256'] = digest_value
        reader = SampleReader(folder, expected_bindings={'sample.json': image_pin.sample_sha256,
                                                       'supervision/label.json': image_pin.label_sha256},
                              expected_json={'supervision/query_trace.json': {'passed': True}})
        names = set(reader.manifest['files'] if reader.manifest else reader.metadata['files'])
        names.update(('sample.json', 'supervision/label.json', 'supervision/query_trace.json'))
        row['provenance'].update(sample_sha256=image_pin.sample_sha256, label_sha256=image_pin.label_sha256,
            logical_files={name: dict(sha256=digest(reader.read(name)), bytes=len(reader.read(name))) for name in names})
        bound.update({str(p): index._sha(p) for p in folder.rglob('*') if p.is_file()})
        coverage = component.packet['scope']['explicit_receipts'][0]
        coverage.update(capture_path=str(folder.parent), storage_root=str(folder.parent))
        component.packet['scene_contexts'] = {join._digest(context): context}
        joined_fixture.publish(component)
        components.append(component)
        image_pins.append(image_pin if i else index.ImagePin(row['sample_id'], str(folder/'inputs/rgb.png'),
            row['encoded_rgb_sha256'], row['decoded_rgb_sha256'], (800.5, 400.5)))
    packet = join.join_inventories([c.pin for c in components])
    graph = compact.build_graph(image_pins, inventory_sha256=packet['sha256'], threshold=policy.CUTOFF)
    return packet, graph, components, image_pins


@pytest.fixture(scope='module')
def shared(tmp_path_factory):
    return fixture_graph(tmp_path_factory.mktemp('provisional_join'))


@pytest.fixture
def case(shared):
    return deepcopy(shared)


def test_real_mixed_graph_new_schema_no_spoof_and_preserved_claims(case):
    packet, graph, components, _ = case
    before = join._canonical((packet, graph))
    result = adapter.select_observed_train(packet, {'A': 'train'}, graph)
    assert result['schema'] == adapter.SCHEMA not in (policy.SCHEMA, join.INPUT_SCHEMA)
    assert result['inventory_sha256'] == packet['sha256']
    assert result['pair_coverage_sha256'] == graph['sha256']
    assert result['graph_adapter']['source_graph'] == graph
    assert result['counts'] == {'train': 2}
    assert result['excluded']['b_sample'] == ['duplicate_view']
    assert [b['inventory_scope'] for b in result['component_audit_basis']] == [c.packet['scope'] for c in components]
    assert result['component_audit_basis'][1]['inventory_scope']['audit_execution_independently_verified'] is True
    assert result['scope']['audit_execution_independently_verified'] is False
    assert not result['scope']['all_native_arrays_reverified']
    assert not result['training_approved'] and not result['source_cap_reset'] and not result['existing_selections_modified']
    assert join._canonical((packet, graph)) == before
    assert result['original_rows_sha256'] == join._digest(packet['records'])
    assert result['original_contexts_sha256'] == join._digest(packet['scene_contexts'])
    assert old.sealed(result) == result


def test_actual_zero_camera_motion_alias_propagates_control_hold(case):
    packet, graph, _, _ = case
    result = adapter.select_observed_train(packet, {'A': 'train'}, graph,
        overlays=[dict(sample_id='b_sample', evidence_id='fixture-control', control_role='control')])
    assert result['counts'] == {'train': 1}
    assert 'control_role' in result['excluded']['a_sample']
    assert result['selected'][0]['sample_id'] == 'c_sample'


@pytest.mark.parametrize('mutation', ['old_schema', 'row_mutation', 'context_mutation', 'missing_component',
    'missing_row_membership', 'wrong_component_file_pin', 'audit_claim_promoted', 'capture_coverage',
    'approval', 'partial_hash_claim'])
def test_join_resealing_cannot_change_original_rows_or_attestations(case, mutation):
    packet, graph, _, _ = case
    if mutation == 'old_schema': packet['schema'] = join.INPUT_SCHEMA
    if mutation == 'row_mutation': packet['records'][0]['decision'] = 'hold_visual_clarity'
    if mutation == 'context_mutation': next(iter(packet['scene_contexts'].values()))['unexpected'] = True
    if mutation == 'missing_component': packet['components'].pop()
    if mutation == 'missing_row_membership': packet['components'][0]['row_ids'] = []
    if mutation == 'wrong_component_file_pin': packet['components'][0]['pin']['inventory_file_sha256'] = '0'*64
    if mutation == 'audit_claim_promoted': packet['components'][0]['attestation_claims']['new_training_approved'] = True
    if mutation == 'capture_coverage': packet['scope']['explicit_receipts'][0]['captured_rows'] = 999
    if mutation == 'approval': packet['training_approved'] = True
    if mutation == 'partial_hash_claim': packet['provenance']['verification']['final_hashes'] -= 1
    packet = old.sealed(packet)
    with pytest.raises(ValueError): adapter.select_observed_train(packet, {'A': 'train'}, graph)


@pytest.mark.parametrize('mutation', ['old_inventory', 'missing_pair', 'missing_compact', 'raw_schema',
    'wrong_threshold', 'wrong_rgb', 'code_binding'])
def test_graph_resealing_cannot_hide_partial_or_wrong_aggregate(case, mutation):
    packet, graph, _, _ = case
    if mutation == 'old_inventory': graph['inventory_sha256'] = '0'*64
    if mutation == 'missing_pair': graph['counts']['resolved_pairs'] -= 1
    if mutation == 'missing_compact': graph['compact_image_pins'] = []
    if mutation == 'raw_schema': graph['schema'] = policy.PAIR_SCHEMA
    if mutation == 'wrong_threshold': graph['threshold'] += 0.001
    if mutation == 'wrong_rgb': graph['image_pins'][0]['sha256'] = '0'*64
    if mutation == 'code_binding': graph['implementation_bindings'].clear()
    graph = old.sealed(graph)
    with pytest.raises(ValueError): adapter.select_observed_train(packet, {'A': 'train'}, graph)


def test_compact_payload_tamper_rejected(tmp_path):
    packet, graph, _, pins = fixture_graph(tmp_path)
    path = payload(pins[1], 'inputs/rgb.png')
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError): adapter.select_observed_train(packet, {'A': 'train'}, graph)


def test_component_file_tamper_rejected(tmp_path):
    packet, graph, components, _ = fixture_graph(tmp_path)
    components[0].path.write_bytes(components[0].path.read_bytes()+b' ')
    with pytest.raises(ValueError, match='Changed bound file'):
        adapter.select_observed_train(packet, {'A': 'train'}, graph)


def test_later_capture_failure_cannot_reuse_old_success_attestation(tmp_path):
    packet, graph, _, _ = fixture_graph(tmp_path)
    capture = Path(packet['records'][0]['capture_path'])
    joined_fixture.write(capture/'failure.json', {'fixture_failure': True})
    with pytest.raises(ValueError, match='Capture failure appeared'):
        adapter.select_observed_train(packet, {'A': 'train'}, graph)


def test_runtime_never_calls_old_selector_or_launches_native():
    tree = ast.parse(Path(adapter.__file__).read_text(encoding='utf-8'))
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert 'policy.select_observed_train' not in calls and 'policy._select' not in calls
    assert not any(name.endswith(('.audit_capture', '.collect', '.Popen')) or 'SimulationApp' in name for name in calls)
    assert not any(isinstance(n, ast.Assign) and any(ast.unparse(t).endswith("['schema']") for t in n.targets) for n in ast.walk(tree))
