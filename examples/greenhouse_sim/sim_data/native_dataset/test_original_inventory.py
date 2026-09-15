"""Synthetic CPU fixtures, never pilot proof; real saved native label/ID/trace replay.

Only source USD reconstruction, robot FK and native smoke are fixture stand-ins.
The adapter's launcher/hash/order checks and saved-buffer auditor are exercised.
"""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from . import inventory as inv, original_inventory as oi, provisional
from .test_inventory import case as generated_case, write, pin, json_at
from ..native_original_capture import audit, contracts as oc
from ..native_original_capture.prepare import CASE_FIELDS
from ..native_original_capture.test_audit import arrays, tokens
from ..capture_contract import write_sample, project, fingerprint
from ..capture_visibility import write_visibility, component_masks


@pytest.fixture
def original(tmp_path, monkeypatch):
    root = tmp_path / 'original_unit'
    capture, folder = root / 'queue/original_capture', root / 'queue/original_capture/sample_0001'
    family, target = 'seed73_full', 'seed73_full/Petiole'
    args = arrays()
    base, meta, report, rgb, z, valid = args[:6]
    meta['calibration'].update(focal_length_mm=2., apertures_mm=[3., 2.], aperture_offsets_mm=[0., 0.])
    base['expected_calibration'] = deepcopy(meta['calibration'])
    prior = dict(role='historical_pose_only_never_training_observation', source_capture=str(root / 'legacy'),
        sample_id='old_848', source_sample_sha256='filled below', prior_target_id=family+'/Main',
        prior_plan=str(root / 'old_plan.json'), prior_lighting={'dome_intensity': 1200},
        allow_other_target_pose=True, historical_review_inherited=False)
    base.update(target_id=target, source_family=family, conservative_view_cap_group=target, pose_prior=prior)
    base['source_row'].update(target_id=target, component_id='Petiole', variant_id=family,
                              source_plant_id=family, split_group=family, draft_id='UNIT_TARGET')
    meta.update(sample_id='sample_0001', pose_prior=deepcopy(prior), rendered_camera_params={'UNIT_ONLY': True})
    meta['supervision'].update(target_id=target, source_target_id=target, conservative_view_cap_group=target,
                               split_group=family, variant_id=family)
    meta['supervision']['projected_interval'] = project(meta['supervision']['interval_world_m'], meta['calibration'])
    meta['supervision']['depth_evidence'] = {'status': 'UNIT TEST ONLY'}
    report['components']['Main']['parent'] = None
    plant_root = '/World/Unit'
    catalogue = [
        dict(component_id='Main', prim_path=plant_root+'/Main', organ_type='main_stem',
             variant_id=family, source_plant_id=family, split_group=family, component_index=1),
        dict(component_id='Petiole', prim_path=plant_root+'/Main/Petiole', organ_type='sub_stem',
             variant_id=family, source_plant_id=family, split_group=family, component_index=2)]
    ids = np.zeros((816, 1696), np.uint32)
    ids[args[6] == 1], ids[args[6] == 2] = 100, 200
    mapping = {0: '/World/Background', 100: plant_root+'/Main/Petiole/mesh', 200: plant_root+'/Main/mesh'}
    components, organs, _ = component_masks(ids, mapping, catalogue)
    target_mask = (components == 2).astype(np.uint8) * 255
    meta.update(native_instance_sha256=oc.digest(ids.tobytes()), native_mapping_sha256=fingerprint(mapping))
    scene_file = root / 'package/scene.usd'
    scene_file.parent.mkdir(parents=True)
    scene_file.write_bytes(b'UNIT ONLY NOT A NATIVE SCENE')
    manifest = root / 'package/plants/components' / family / 'manifest.json'
    manifest.parent.mkdir(parents=True)
    geometry = []
    for name, parent in [('Main', None), ('Petiole', 'Main')]:
        (manifest.parent / (name+'.usdc')).write_bytes(('UNIT '+name).encode())
        geometry.append(dict(id=name, parent=parent, file=name+'.usdc', type='sub_stem' if parent else 'main_stem'))
    write(manifest, {'components': geometry})
    base['source_row']['source_manifest_sha256'] = pin(manifest)
    variants = [dict(plant_root=plant_root, variant_id=family, source_plant_id=family, split_group=family,
                    added_components={}, added_component_paths={}, source_geometry_modified=False)]
    asset_pins = {str(p): pin(p) for p in [scene_file, *sorted(manifest.parent.glob('*.usdc'))]}
    clear = dict(family_assignments=deepcopy(oc.FROZEN_SPLITS), jobs=[dict(job_id='job_unit', plant_family=family,
        split='train', source_manifest_path=str(manifest), targets=[deepcopy(base['source_row'])])],
        source_bindings_sha256=dict(asset_pins, **{str(manifest): pin(manifest)}))
    write(root / 'clear.json', clear)
    write(root / 'old_plan.json', clear)
    write(root / 'legacy/old_848/sample.json', {'resolution': [848, 408], 'historical_pose_only': True})
    prior['source_sample_sha256'] = pin(root / 'legacy/old_848/sample.json')
    write(root / 'legacy/manifest.json', dict(scene=str(scene_file), package=str(scene_file.parent),
        variants=variants, source_usd_sha256=asset_pins, lighting=prior['prior_lighting'],
        observed_renderer_settings={'aa': 3}, renderer='RealTimePathTracing', scene_counts=meta['scene_counts'],
        unbundled_external_prop_roots_excluded=[]))
    prior['source_manifest_sha256'] = pin(root / 'legacy/manifest.json')
    meta['pose_prior'] = deepcopy(prior)
    sources = {str(p): pin(p) for p in [root/'clear.json', root/'old_plan.json', root/'legacy/manifest.json',
        root/'legacy/old_848/sample.json', manifest, *[Path(p) for p in asset_pins]]}
    plan = dict(base, schema=oc.SCHEMA, split='train', geometry_mode='unmodified_original',
        family_assignments=deepcopy(oc.FROZEN_SPLITS), admission=deepcopy(oc.ADMISSION),
        package=str(scene_file.parent), source_collection_plan=str(root/'clear.json'),
        source_collection_plan_sha256=pin(root/'clear.json'), collection_job_id='job_unit',
        original_variant=variants[0], scene_variants=variants, excluded_external_roots=[],
        source_bindings=sources, implementation_bindings=deepcopy(oc.LOADED_IMPLEMENTATION),
        prior_calibration=deepcopy(meta['calibration']), sample_count_limit=2, coverage=dict(requested_cases=2, requested_original_targets=1,
            native_reachability_or_clarity_measured=False, unselected_manifest_entries_individually_verified=False))
    plan['cases'] = [dict(case_id=f'sample_{i:04d}', **{k: deepcopy(plan[k]) for k in CASE_FIELDS}) for i in (1, 2)]
    fresh = [plan, meta, report, rgb, z, valid, components, target_mask, catalogue]
    tokens(fresh)
    label, trace, decision = audit.review_arrays(*fresh)
    assert decision == inv.STRICT
    write_sample(folder, rgb, z, valid, meta)
    write_visibility(folder, ids, mapping, catalogue, components, organs, target_mask > 0)
    write(folder/'supervision/label.json', label)
    write(folder/'supervision/query_trace.json', trace)
    record = dict(case_id='sample_0001', target_id=target, state='native_captured', source_assets_unchanged=True,
        admission=deepcopy(oc.ADMISSION), sample_sha256=pin(folder/'sample.json'),
        label_sha256=pin(folder/'supervision/label.json'), trace_sha256=pin(folder/'supervision/query_trace.json'),
        scene_counts=meta['scene_counts'], geometry_screen=meta['geometry_screen'], lighting=meta['lighting'], label_reason=label['reason'])
    held = dict(case_id='sample_0002', target_id=target, state='held_pre_render',
                reason='UNIT pre-render geometry hold', admission=deepcopy(oc.ADMISSION))
    expected_plan = deepcopy(plan)
    def fixture_plan_check(value):
        assert value == expected_plan, 'UNIT original source/pose plan changed'
    monkeypatch.setattr(audit, 'check_plan', fixture_plan_check)
    monkeypatch.setattr(audit, 'original_report', lambda clear, job: deepcopy(report))
    monkeypatch.setattr(audit, 'verify_camera', lambda metadata: None)
    monkeypatch.setattr(audit, 'verify_smoke', lambda *args: True)
    automatic = audit.audit_records(capture, plan, [record, held])
    request = dict(schema=oc.RESULT_SCHEMA, plan_path=str(root/'plan.json'),
        host_memory_preflight={'allowed': True}, process_admission=dict(worker_pid=4322,
            no_unrelated_kit_process_at_admission=True), training_started=False)
    result = dict(schema=oc.RESULT_SCHEMA, state='original_native_capture_complete_automatically_audited',
        capture=dict(records=[record, held], source_assets_unchanged=True, stage_count=1, greenhouse_render_product_count=1),
        smoke_bindings={}, admission=deepcopy(oc.ADMISSION))
    command = [str(root/'isaac/python.bat'), '-m', 'sim_data.native_original_capture.collector',
               '--plan', str(root/'plan.json'), '--plan-sha256', '', '--output', str(capture)]
    receipt = dict(state=oi.LAUNCH_STATE, exit_code=0, capture=str(capture), plan_path=str(root/'plan.json'),
        owned_worker=dict(worker_pid=4321, command=command, event_path=str(root/'queue/queue_000003.json')),
        audit_path=str(root/'queue/original_postexit_audit.json'), training_approved=False, source_cap_reset=False)
    fixture = dict(root=root, capture=capture, folder=folder, plan=plan, request=request, result=result,
                   automatic=automatic, receipt=receipt, expected_plan=expected_plan, report=report)
    reseal(fixture)
    return fixture


def reseal(f):
    """Repair synthetic outer pins only; NEVER redo a stored audit implicitly."""
    root, receipt, result, request = (f[k] for k in ('root', 'receipt', 'result', 'request'))
    plan_sha = write(root/'plan.json', f['plan'])
    request['plan_sha256'] = result['plan_sha256'] = receipt['plan_sha256'] = plan_sha
    result['request_sha256'] = receipt['request_sha256'] = write(f['capture']/'request.json', request)
    result['automatic_audit_sha256'] = write(f['capture']/'automatic_audit.json', f['automatic'])
    receipt['audit_sha256'] = write(Path(receipt['audit_path']), f.get('postexit', f['automatic']))
    receipt['result_sha256'] = write(f['capture']/'result.json', result)
    receipt['source_bindings'] = dict(f['plan']['implementation_bindings'], **oi._LOADED_PRODUCER,
                                    **{str(root/'plan.json'): plan_sha})
    worker = receipt['owned_worker']
    worker['command'][6] = plan_sha
    event = dict(state='original_pilot_running', worker_pid=worker['worker_pid'], command=worker['command'],
                 bindings=receipt['source_bindings'], training_approved=False, source_cap_reset=False)
    worker['event_sha256'] = write(Path(worker['event_path']), event)
    return receipt_pin(f)


def receipt_pin(f):
    path = f['root']/'queue/original_launcher_receipt.json'
    sha = write(path, f['receipt'])
    f['pin'] = oi.OriginalCapturePin(str(f['capture']), f['receipt']['result_sha256'], str(path), sha)
    return f['pin']


def build(f, **kwargs):
    return oi.build_inventory([f['pin']], **kwargs)


def test_original_common_rows_replayed_provenance_and_held_coverage(original):
    before = {str(p): pin(p) for p in original['root'].rglob('*') if p.is_file()}
    out = build(original)
    row, = out['records']
    assert out['schema'] == inv.SCHEMA and out['counts']['captured_rows'] == 1
    assert out['counts']['source_kinds'] == {'original_native': 1}
    assert row['source_target'] == row['target_id'] == row['variant_target'] == 'seed73_full/Petiole'
    assert row['context_id'] == 'original:seed73_full/Petiole'
    assert row['annotation_review']['passed'] and row['annotation_review']['audit_execution_independently_verified']
    lineage = row['provenance']['lineage']
    assert lineage['frozen_family_assignments'] == oc.FROZEN_SPLITS and len(oc.FROZEN_SPLITS) == 24
    assert lineage['pose_prior']['prior_target_id'] == 'seed73_full/Main'
    assert not any(k.startswith('generated_') for k in lineage)
    assert lineage['original_source_row'] == original['plan']['source_row']
    assert row['provenance']['process_admission']['worker_pid'] != row['provenance']['owned_worker']['worker_pid']
    assert row['provenance']['renderer_settings']['value'] is None
    context = out['scene_contexts'][row['scene_identity_basis']['static_scene_sha256']]
    assert context['lighting']['dome_intensity'] == 6000
    assert lineage['pose_prior']['prior_lighting']['dome_intensity'] == 1200
    summary, = out['scope']['explicit_receipts']
    assert (summary['planned_rows'], summary['captured_rows'], summary['noncaptured_rows']) == (2, 1, 1)
    assert summary['held_pre_render'][0]['reason'] == 'UNIT pre-render geometry hold'
    assert not out['training_approved'] and not out['admission_performed'] and not out['scope']['global_complete']
    assert not row['source_cap_reset'] and not row['independent_geometry_qualification']
    assert not row['provenance']['admission']['historical_training_rows']
    assert all(str(original['folder']/name) in out['provenance']['source_bindings'] for name in row['provenance']['logical_files'])
    assert out['sha256'] == inv._hash({k: v for k, v in out.items() if k != 'sha256'})
    assert before == {str(p): pin(p) for p in original['root'].rglob('*') if p.is_file()}
    rows, summary, bindings = oi.original_observed_rows(original['pin'])
    assert rows == out['records'] and summary['captured_rows'] == 1
    assert bindings == out['provenance']['source_bindings']


@pytest.mark.parametrize('field,value', [('state', 'generic_exit_claim'), ('exit_code', 1), ('exit_code', False),
    ('training_approved', True), ('source_cap_reset', True), ('request_sha256', '0'*64),
    ('result_sha256', '0'*64), ('plan_sha256', '0'*64)])
def test_launcher_receipt_must_bind_successful_exact_handoff(original, field, value):
    original['receipt'][field] = value
    receipt_pin(original)
    with pytest.raises(ValueError):
        build(original)


def test_required_producer_chain_includes_process_guard():
    guard = str(Path(oi.__file__).with_name('native_process_guard.py').resolve())
    assert guard in {str(p) for p in oi._PRODUCER}
    assert oi.implementation_bindings()[guard] == oi._LOADED_PRODUCER[guard] == pin(Path(guard))


@pytest.mark.parametrize('mutation', ['missing', 'wrong_hash'])
def test_guard_binding_required_even_in_coherently_resealed_launcher_event(original, monkeypatch, mutation):
    receipt = original['receipt']
    guard = str(Path(oi.__file__).with_name('native_process_guard.py').resolve())
    if mutation == 'missing':
        receipt['source_bindings'].pop(guard)
    else:
        receipt['source_bindings'][guard] = '0'*64
    worker = receipt['owned_worker']
    event = json_at(Path(worker['event_path']))
    event['bindings'] = deepcopy(receipt['source_bindings'])
    worker['event_sha256'] = write(Path(worker['event_path']), event)
    receipt_pin(original)
    def no_replay(*args, **kwargs):
        pytest.fail('Unbound guard must fail before independent audit replay')
    monkeypatch.setattr(audit, 'audit_capture', no_replay)
    with pytest.raises(ValueError, match='Known launcher producer|Conflicting source hashes'):
        build(original)


@pytest.mark.parametrize('failure', ['missing_receipt', 'failure_marker', 'producer', 'event_pin', 'event_pid', 'event_command',
                                   'command', 'postexit', 'compact_marker', 'held_directory'])
def test_no_generic_exit_or_unbound_producer_or_event(original, failure):
    f = original
    if failure == 'missing_receipt':
        Path(f['pin'].launcher_receipt_path).unlink()
    elif failure == 'failure_marker':
        write(f['capture']/'failure.json', {})
    elif failure == 'producer':
        del f['receipt']['source_bindings'][str(oi._PRODUCER[0])]
        receipt_pin(f)
    elif failure == 'event_pin':
        write(Path(f['receipt']['owned_worker']['event_path']), {})
    elif failure in ('event_pid', 'event_command'):
        worker = f['receipt']['owned_worker']
        event = json_at(Path(worker['event_path']))
        if failure == 'event_pid':
            event['worker_pid'] += 1
        else:
            event['command'][2] = 'sim_data.native_generated_views'
        worker['event_sha256'] = write(Path(worker['event_path']), event)
        receipt_pin(f)
    elif failure == 'command':
        f['receipt']['owned_worker']['command'] += ['--bad-extra']
        receipt_pin(f)
    elif failure == 'postexit':
        f['postexit'] = dict(f['automatic'], review_method='invented')
        reseal(f)
    elif failure == 'compact_marker':
        write(f['folder']/'bundle.json', {})
    else:
        (f['capture']/'sample_0002').mkdir()
    with pytest.raises(ValueError):
        build(f)


@pytest.mark.parametrize('mutation', ['missing', 'reordered', 'extra', 'count'])
def test_complete_ordered_case_coverage(original, mutation):
    rows = original['result']['capture']['records']
    if mutation == 'missing': rows.pop()
    elif mutation == 'reordered': rows.reverse()
    elif mutation == 'extra': rows.append(deepcopy(rows[0]))
    else: original['automatic']['counts'][inv.STRICT] += 1
    reseal(original)
    with pytest.raises(ValueError):
        build(original)


@pytest.mark.parametrize('name', ['label', 'query_trace'])
def test_rehashed_stale_label_or_trace_fails_real_independent_replay(original, name):
    path = original['folder']/'supervision'/(name+'.json')
    value = json_at(path)
    value['tampered'] = True
    field = 'label_sha256' if name == 'label' else 'trace_sha256'
    original['result']['capture']['records'][0][field] = write(path, value)
    original['automatic']['records'][0][field] = pin(path)
    reseal(original)
    with pytest.raises(ValueError, match='Stale stored'):
        build(original)


@pytest.mark.parametrize('failure', ['rgb', 'depth', 'IDs', 'catalogue', 'source_asset', 'clear_plan'])
def test_changed_native_or_original_source_bytes_rejected(original, failure):
    choices = {'rgb': original['folder']/'inputs/rgb.png', 'depth': original['folder']/'inputs/depth_m.npy',
        'IDs': original['folder']/'supervision/renderer_instance_id.npy',
        'catalogue': original['folder']/'supervision/identities.json', 'clear_plan': original['root']/'clear.json',
        'source_asset': original['root']/'package/plants/components/seed73_full/Petiole.usdc'}
    choices[failure].write_bytes(b'UNIT tampering')
    with pytest.raises(ValueError):
        build(original)


@pytest.mark.parametrize('failure', ['cap', 'prior_target', 'lighting', 'mask'])
def test_rehashed_saved_metadata_cannot_relabel_or_relight_native_proof(original, failure):
    path = original['folder']/'sample.json'
    meta = json_at(path)
    if failure == 'cap': meta['supervision']['conservative_view_cap_group'] = 'seed73_full/Main'
    elif failure == 'prior_target': meta['pose_prior']['prior_target_id'] = 'seed73_full/Petiole'
    elif failure == 'lighting': meta['lighting']['dome_intensity'] = 1200
    else:
        target_path = original['folder']/'supervision/target_visible.png'
        from PIL import Image
        Image.fromarray(np.zeros((816, 1696), np.uint8)).save(target_path)
        meta['files']['supervision/target_visible.png']['sha256'] = pin(target_path)
    original['result']['capture']['records'][0]['sample_sha256'] = write(path, meta)
    reseal(original)
    with pytest.raises(ValueError):
        build(original)


def test_frozen_original_heldout_plan_rejected_even_if_replay_stubbed(original, monkeypatch):
    original['plan']['family_assignments']['seed73_full'] = 'test'
    reseal(original)
    monkeypatch.setattr(audit, 'audit_capture', lambda *a, **kw: original['automatic'])
    with pytest.raises(ValueError, match='Frozen original TRAIN'):
        build(original)


def test_native_failure_appearing_after_replay_rejected(original, monkeypatch):
    real = audit.audit_capture
    def changed(*args, **kwargs):
        result = real(*args, **kwargs)
        write(original['capture']/'failure.json', {})
        return result
    monkeypatch.setattr(audit, 'audit_capture', changed)
    with pytest.raises(ValueError, match='appeared during inventory'):
        build(original)


def test_repeated_original_capture_and_unpinned_extra_roots(original):
    with pytest.raises(ValueError, match='Duplicate capture'):
        oi.build_inventory([original['pin'], original['pin']])
    out = build(original, extra_observed_roots=[inv.ObservedRoot(str(original['root']/'legacy'))])
    assert out['scope']['extra_observed_roots'][0]['status'] == 'unindexed_no_receipts'
    assert not out['scope']['global_complete'] and out['counts']['captured_rows'] == 1


def test_mixed_generated_authentication_lineage_and_native_claims_preserved(original, generated_case):
    old = inv.build_inventory([generated_case['pin']])
    out = build(original, generated_receipts=[generated_case['pin']])
    assert out['counts']['captured_rows'] == 4
    assert out['counts']['decisions'] == {inv.STRICT: 2, inv.HOLD: 1, inv.EXCLUDE: 1}
    assert not out['scope']['audit_execution_independently_verified']
    by_id = {r['sample_id']: r for r in out['records']}
    for row in old['records']:
        new = by_id[row['sample_id']]
        assert new['source_kind'] == 'generated_native'
        assert new['annotation_review'] == row['annotation_review']
        assert new['provenance']['lineage'] == row['provenance']['lineage']
        assert new['provenance']['audit_declarations'] == row['provenance']['audit_declarations']
        assert new['provenance']['legacy_scene_identity']['scene_sha256'] == row['scene_sha256']
        assert new['camera_sha256'] == row['camera_sha256']
        assert new['source_target'] == row['source_target'] and new['context_id'] == row['context_id']
    assert out['exact_duplicates']['decoded_rgb_sha256']['duplicate_excess_rows'] == 2
    assert out['exact_duplicates']['scene_camera_sha256']['duplicate_excess_rows'] == 2


def test_neutral_scene_key_unifies_actual_same_scene_without_source_kind_or_settings(original):
    out = build(original)
    row, = out['records']
    basis = out['scene_contexts'][row['scene_identity_basis']['static_scene_sha256']]
    legacy = dict(basis, replaced_plant_root=basis['target_plant_root'],
        generated_geometry_sha256=basis['target_geometry_sha256'], original_lighting=basis['lighting'],
        original_scene_counts=basis['scene_counts'], original_renderer=basis['renderer'], renderer_settings={'aa': 999})
    normalized = oi._neutral_basis(legacy)
    assert normalized == basis
    fake_generated = deepcopy(row)  # Unit physical-key comparison, not a receipt or invented collected row.
    fake_generated.update(sample_id='synthetic_other', source_kind='generated_native')
    duplicates = inv._duplicates([row, fake_generated])
    assert duplicates['decoded_rgb_sha256']['duplicate_excess_rows'] == 1
    assert duplicates['scene_camera_sha256']['duplicate_excess_rows'] == 1
    fake_generated['decoded_rgb_sha256'] = inv._hash('RGB noise is not a new camera view')
    assert inv._duplicates([row, fake_generated])['scene_camera_sha256']['duplicate_excess_rows'] == 1
    old_light = deepcopy(legacy)
    old_light['original_lighting']['dome_intensity'] = 1200
    assert inv._hash(oi._neutral_basis(old_light)) != inv._hash(basis)


def coverage(packet):
    """Synthetic complete pair declaration to exercise unchanged pure selector."""
    rows = sorted(packet['records'], key=lambda r: r['sample_id'])
    paths = {r['sample_id']: str(Path(r['sample_path'])/'inputs/rgb.png') for r in rows}
    n = len(rows)*(len(rows)-1)//2
    result = dict(schema=provisional.PAIR_SCHEMA, inventory_sha256=packet['sha256'],
        metric=provisional.METRIC, threshold=provisional.CUTOFF, dimensions=[1696, 816], patch_radius=64,
        patch_anchor='nominal_10mm_cut_point_not_anatomical_attachment', sample_ids=[r['sample_id'] for r in rows],
        global_collection_complete=False, threshold_calibrated=False, training_approved=False, complete=True,
        counts=dict(total_pairs=n, resolved_pairs=n, exact_compared_pairs=n, bound_pruned_pairs=0),
        image_pins=[dict(sample_id=r['sample_id'], path=paths[r['sample_id']], sha256=r['encoded_rgb_sha256'],
            decoded_rgb_sha256=r['decoded_rgb_sha256'], nominal_uv=[848., 408.]) for r in rows],
        input_bindings={paths[r['sample_id']]: r['encoded_rgb_sha256'] for r in rows},
        implementation_bindings={'UNIT_ONLY': 'a'*64}, near_image_edges=[], edge_scores=[])
    result['sha256'] = inv._hash(result)
    return result


def test_old_pair_coverage_is_not_reusable_and_mixed_rows_feed_unchanged_selector(original, generated_case):
    old = inv.build_inventory([generated_case['pin']])
    mixed = build(original, generated_receipts=[generated_case['pin']])
    splits = dict(oc.FROZEN_SPLITS, donor='train')
    with pytest.raises(ValueError, match='Different pair coverage'):
        provisional.select_observed_train(mixed, splits, coverage(old))
    selected = provisional.select_observed_train(mixed, splits, coverage(mixed))
    assert selected['training_approved'] is False


def test_original_and_generated_do_not_receive_separate_source_caps(original):
    packet = build(original)
    first = packet['records'][0]
    rows = []
    for i in range(14):
        row = deepcopy(first)
        row.update(sample_id='UNIT_'+str(i), sample_path=str(original['root']/f'only_synthetic_row_{i}'),
            source_kind='original_native' if i % 2 else 'generated_native', context_id=f'UNIT_CONTEXT_{i}',
            decoded_rgb_sha256=inv._hash(['rgb', i]), encoded_rgb_sha256=inv._hash(['png', i]),
            scene_sha256=inv._hash(['scene', i]), camera_sha256=inv._hash(['camera', i]))
        rows.append(row)
    packet['records'], packet['counts']['captured_rows'] = rows, len(rows)
    packet.pop('sha256'); packet['sha256'] = inv._hash(packet)
    selection = provisional.select_observed_train(packet, oc.FROZEN_SPLITS, coverage(packet))
    assert len(selection['selected']) == 12


def test_heldout_exact_alias_contaminates_original_training_candidate(original):
    packet = build(original)
    row = deepcopy(packet['records'][0])
    row.update(sample_id='UNIT_HELDOUT', split='test', source_family='seed31_full', source_target='seed31_full/Petiole',
               conservative_view_cap_group='seed31_full/Petiole', sample_path=str(original['root']/'UNIT_HELDOUT'))
    packet['records'].append(row)
    packet['counts']['captured_rows'] = 2
    packet.pop('sha256'); packet['sha256'] = inv._hash(packet)
    selection = provisional.select_observed_train(packet, oc.FROZEN_SPLITS, coverage(packet))
    assert not selection['selected']


def test_import_remains_cpu_only():
    import sys
    assert 'omni.kit' not in sys.modules and 'isaacsim' not in sys.modules
    assert str(Path(oi.__file__).resolve()) in oi.implementation_bindings()


@pytest.mark.parametrize('decision', [inv.HOLD, inv.EXCLUDE])
def test_actual_original_nonpassing_annotation_rows_are_not_dropped(original, decision):
    from PIL import Image
    f, folder = original, original['folder']
    meta = json_at(folder/'sample.json')
    with Image.open(folder/'inputs/rgb.png') as image:
        rgb = np.asarray(image).copy()
    if decision == inv.EXCLUDE:
        rgb[:] = 0
    else:
        rgb[402:414, 925:980] = 0
    Image.fromarray(rgb).save(folder/'inputs/rgb.png')
    meta['files']['inputs/rgb.png']['sha256'] = pin(folder/'inputs/rgb.png')
    meta['synchronization']['freshness']['rgb_sha256'] = oc.digest(rgb.tobytes())
    record = f['result']['capture']['records'][0]
    record['sample_sha256'] = write(folder/'sample.json', meta)
    depth = np.load(folder/'inputs/depth_m.npy', allow_pickle=False)
    components = np.load(folder/'supervision/component_id.npy', allow_pickle=False)
    with Image.open(folder/'inputs/depth_valid.png') as im:
        valid = np.asarray(im) == 255
    with Image.open(folder/'supervision/target_visible.png') as im:
        mask = np.asarray(im).copy()
    catalogue = json_at(folder/'supervision/identities.json')['component_catalogue']
    label, trace, actual = audit.review_arrays(f['plan'], meta, f['report'], rgb, depth, valid, components, mask, catalogue)
    assert actual == decision
    record['label_reason'] = label['reason']
    record['label_sha256'] = write(folder/'supervision/label.json', label)
    trace_path = folder/'supervision/query_trace.json'
    if trace is None:
        trace_path.unlink()
        record['trace_sha256'] = None
    else:
        record['trace_sha256'] = write(trace_path, trace)
    f['automatic'] = audit.audit_records(f['capture'], f['plan'], f['result']['capture']['records'])
    reseal(f)
    out = build(f)
    assert out['counts']['decisions'] == {decision: 1}
    assert out['records'][0]['annotation_review']['passed'] is False
    assert out['records'][0]['provenance']['audit_declarations']['trace_replayed_exact'] is (trace is not None)
    assert out['scope']['explicit_receipts'][0]['noncaptured_rows'] == 1


def test_full_final_source_hash_check_not_only_initial_cache(original, monkeypatch):
    real = audit.audit_capture
    asset = original['root']/'package/plants/components/seed73_full/Petiole.usdc'
    def changed(*args, **kwargs):
        result = real(*args, **kwargs)
        asset.write_bytes(b'UNIT changed after replay')
        return result
    monkeypatch.setattr(audit, 'audit_capture', changed)
    with pytest.raises(ValueError, match='Source changed during inventory'):
        build(original)
