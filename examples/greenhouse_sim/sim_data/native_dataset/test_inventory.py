"""Synthetic CPU fixtures only; no real capture, audit replay or image output."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from . import inventory as inv
from .admission import decoded_rgb_digest
from .bundle import digest, pack_sample
from ..capture_contract import fingerprint


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    return pin(path)


def pin(path):
    return digest(path.read_bytes())


def json_at(path):
    return json.loads(path.read_text())


def reseal(case):
    """Repair outer synthetic fixture pins, never a real audit."""
    root = case['root']
    plan, request, result, receipt = (case[k] for k in ('plan', 'request', 'result', 'receipt'))
    for row in result['records']:
        if row['state'] != inv.CAPTURED:
            continue
        folder = root / 'capture' / row['candidate_id']
        row['sample_sha256'] = pin(folder / 'sample.json')
        row['label_sha256'] = pin(folder / 'supervision/label.json')
        for reviewed in receipt['records']:
            if Path(reviewed['sample']).name == row['candidate_id']:
                reviewed['sample_sha256'], reviewed['label_sha256'] = row['sample_sha256'], row['label_sha256']
                reviewed['rgb_sha256'] = pin(folder / 'inputs/rgb.png')
    request['plan_sha256'] = write(root / 'plan.json', plan)
    receipt['plan_sha256'] = request['plan_sha256']
    receipt['request_sha256'] = write(root / 'capture/request.json', request)
    receipt['result_sha256'] = write(root / 'capture/result.json', result)
    receipt_hash = write(root / 'reviews/receipt.json', receipt)
    case['pin'] = inv.ReceiptPin(str(root / 'reviews/receipt.json'), receipt_hash,
                               str(root / 'capture'), str(root / 'plan.json'))
    return case['pin']


def sample(root, name, index, eligible, passed):
    folder = root / 'capture' / name
    cal = dict(resolution=[1696, 816], camera_path='/World/Robot/Camera',
        camera_to_world_usd_row_vectors=np.eye(4).tolist(),
        intrinsics=[[1000, 0, 848], [0, 1000, 408], [0, 0, 1]],
        focal_length_mm=2., apertures_mm=[3., 2.], aperture_offsets_mm=[0., 0.],
        clipping_range_m=[.01, 100.], crop_resize=None,
        depth_convention='optical_axis_z_metres_not_ray_range')
    pose = dict(joint_degrees={'head': 0.}, robot_root_to_world_usd_row_vectors=np.eye(4).tolist(),
                camera_to_head_column_vectors=np.eye(4).tolist())
    rgb = np.full((816, 1696, 3), 90, np.uint8)
    z = np.full((816, 1696), 2., np.float32)
    components = np.zeros((816, 1696), np.uint32)
    components[300:330, 800:840] = 5
    organs = np.where(components == 5, 2, 0).astype(np.uint8)
    buffers = {'inputs/rgb.png': rgb, 'inputs/depth_m.npy': z,
        'inputs/depth_valid.png': np.full(z.shape, 255, np.uint8),
        'supervision/renderer_instance_id.npy': components,
        'supervision/component_id.npy': components, 'supervision/organ_type.png': organs,
        'supervision/target_visible.png': (components == 5).astype(np.uint8) * 255}
    files = {}
    for logical, array in buffers.items():
        file = folder / logical
        file.parent.mkdir(parents=True, exist_ok=True)
        if logical.endswith('.npy'):
            np.save(file, array, allow_pickle=False)
        else:
            Image.fromarray(array).save(file)
        files[logical] = dict(sha256=pin(file), role='observation' if logical.startswith('inputs/')
                              else 'ground_truth_supervision')
    ids = dict(renderer_id_to_prim={'0': 'BACKGROUND', '5': '/World/Generated/Stem'},
        component_catalogue=[dict(variant_id='variant', component_id='Stem', component_index=5,
                                 prim_path='/World/Generated/Stem', organ_type='sub_stem')])
    files['supervision/identities.json'] = dict(
        sha256=write(folder / 'supervision/identities.json', ids), role='ground_truth_supervision')
    metadata = dict(sample_id=name, training_sample_approved=False, calibration=cal, robot_snapshot=pose,
        geometry_screen=dict(passed=True), scene_counts={'components': 1, 'backdrop_instances': 142},
        lighting={'intensity': 1500}, renderer='RealTimePathTracing', render_budget={'profile': 'reference56'},
        input_policy=dict(clean_full_scene=True, isolation=False, diagnostic_overlays=False), files=files,
        synchronization=dict(method='frozen_scene_single_native_writer_payload',
            scene_unchanged_during_capture=True, dynamic_recording_supported=False, static_guard='a'*64,
            reference_time=[0, 1000000000], freshness=dict(callback_sequence=index,
                camera_sha256=fingerprint(cal), rgb_sha256=digest(rgb.tobytes()), depth_sha256=digest(z.tobytes()))),
        supervision=dict(target_id='variant/Stem', source_target_id='donor/Stem', split_group='donor',
            conservative_view_cap_group='donor/Stem', plant_to_world_usd_row_vectors=np.eye(4).tolist()))
    label = dict(target_id='variant/Stem', source_plant_family='donor', conservative_view_cap_group='donor/Stem',
                 eligible=eligible, reason='synthetic_only', training_approved=False, native_depth_reconstructed=False)
    if eligible:
        label.update(query_pixel_uv=[820., 315.], query_evidence=dict(arc_m=.045, visible=True,
            depth_status='depth_consistent_not_visibility_verified',
            projected=dict(pixel_xy=[820., 315.], projection_status='in_frame')))
    trace = dict(passed=passed, reasons=[] if passed else ['query_to_cut_chain_too_dark'],
        arc_interval_m=[.008, .045], probe_count=80, unique_pixels=20, maximum_projected_step_px=.49,
        minimum_interior_radius_px=3., dark_fraction=0. if passed else .2, median_luminance=90.,
        identity_or_depth_gap_count=0, first_gap_probes=[], local_usability_failures=[],
        native_depth_reconstructed=False) if eligible else None
    write(folder / 'sample.json', metadata)
    write(folder / 'supervision/label.json', label)
    if trace is not None:
        write(folder / 'supervision/query_trace.json', trace)
    row = dict(candidate_id=name, state=inv.CAPTURED, requested_spec={'candidate_id': name},
        target_id='variant/Stem', robot_snapshot=pose, screen={'passed': True}, query_trace=trace,
        automatic_annotation_eligible=eligible and passed, eligible_annotation=eligible)
    review = dict(sample=str(folder), target_id='variant/Stem', source_target_group='donor/Stem', source_family='donor',
        decision=inv.STRICT if eligible and passed else inv.HOLD if eligible else inv.EXCLUDE,
        review_method='replayed_anatomy_exact_native_buffers_and_trace', label_replayed_exact=True,
        trace_replayed_exact=trace is not None, native_callback_hashes_verified=True, source_and_file_hashes_verified=True,
        reason='synthetic_only', trace_reasons=trace['reasons'] if trace else None, render_profile='reference56',
        training_approved=False, source_cap_reset=False, physical_execution_approved=False)
    return row, review


@pytest.fixture
def case(tmp_path):
    root = tmp_path / 'fixture'
    asset = root / 'package/scene.usd'
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b'synthetic opaque scene fixture, NOT USD qualification')
    donor_asset = root / 'package/plants/components/donor/stem.usdc'
    donor_asset.parent.mkdir(parents=True)
    donor_asset.write_bytes(b'synthetic background donor geometry')
    generated = root / 'variant/stem.usdc'
    generated.parent.mkdir()
    generated.write_bytes(b'synthetic opaque generated component')
    geometry = generated.parent / 'manifest.json'
    write(geometry, {'components': [dict(id='Stem', parent=None, file='stem.usdc',
                                       attach_point=[0, 0, 0], transform={'translate': [0, 0, 0]}, type='sub_stem')]})
    donor_manifest = donor_asset.parent / 'manifest.json'
    write(donor_manifest, {'components': [dict(id='Stem', file='stem.usdc', type='sub_stem')]})
    source_row = dict(draft_id='N_donor_Stem', target_id='donor/Stem', component_id='Stem', variant_id='donor',
        source_plant_id='donor', split_group='donor', source_manifest_sha256=pin(donor_manifest),
        cut_region_proposal={'origin': 'original'})
    generated_row = dict(target_id='variant/Stem', component_id='Stem', variant_id='variant',
        source_plant_id='donor', split_group='donor', conservative_view_cap_group='donor/Stem',
        cut_region_proposal={'origin': 'generated'}, attachment_plant_m=[0, 0, 0], expected_detached_component_ids=['Stem'])
    original_plan = root / 'original_plan.json'
    write(original_plan, dict(family_assignments={'donor': 'train'},
        source_bindings_sha256={str(donor_manifest): pin(donor_manifest), str(donor_asset): pin(donor_asset)},
        jobs=[dict(job_id='job_001', plant_family='donor', split='train',
            source_manifest_path=str(donor_manifest), targets=[source_row])]))
    source_sample = root / 'original/reference/sample.json'
    write(source_sample, dict(sample_id='reference', supervision=dict(target_id='donor/Stem', split_group='donor',
                                                                   cut_region_proposal=source_row['cut_region_proposal'])))
    original_variant = dict(plant_root='/World/Plants/1', variant_id='donor', source_plant_id='donor', split_group='donor')
    original = root / 'original/manifest.json'
    write(original, dict(source_usd_sha256={str(asset): pin(asset), str(donor_asset): pin(donor_asset)}, scene=str(asset), package=str(asset.parent),
        variants=[original_variant], samples=[dict(sample_id='reference', target_review_id=source_row['draft_id'])],
        source_collection_plan_path=str(original_plan), source_collection_plan_sha256=pin(original_plan),
        target_family_split='train', collection_job_id='job_001',
        unbundled_external_prop_roots_excluded=[], observed_renderer_settings={'aa': 3},
        lighting={'intensity': 1500}, scene_counts={'components': 1, 'backdrop_instances': 142},
        renderer='RealTimePathTracing'))
    qualification = root / 'variant/qualification.json'
    write(qualification, dict(version='curved_relocated_rigid_leaf_static.v2', variant_id='variant',
        source_family='donor', split_group='donor', split='train', source_manifest_path=str(donor_manifest),
        source_plan_path=str(original_plan), source_plan_sha256=pin(original_plan), frozen_family_assignments={'donor': 'train'},
        source_bindings={str(donor_manifest): pin(donor_manifest), str(donor_asset): pin(donor_asset)},
        output_hashes={'manifest.json': pin(geometry), 'stem.usdc': pin(generated)},
        targets=[dict(component_id='Stem', source_target_id='donor/Stem', conservative_view_cap_group='donor/Stem',
                      cut_region_proposal=generated_row['cut_region_proposal'])],
        recipes=[dict(component_id='Stem', source_target_id='donor/Stem', members=['Stem'])]))
    sources = {str(p): pin(p) for p in (asset, donor_asset, donor_manifest, generated, geometry, original,
                                       source_sample, original_plan, qualification)}
    anchor = root / 'anchor.json'
    write(anchor, dict(source_capture=str(original.parent), source_bindings=dict(sources),
        variant_directory=str(generated.parent), original_variant=original_variant, source_sample='reference',
        source_family='donor', split_group='donor', split='train', source_collection_plan=str(original_plan),
        source_row=source_row, generated_row=generated_row, conservative_view_cap_group='donor/Stem'))
    sources[str(anchor)] = pin(anchor)
    review_code = {str(Path(inv.__file__).with_name(name).resolve()): pin(Path(inv.__file__).with_name(name))
                   for name in ('audit.py', 'bundle.py')}
    pairs = [sample(root, 'strict', 1, True, True), sample(root, 'hold', 2, True, False),
             sample(root, 'excluded', 3, False, False)]
    rows, reviews = map(list, zip(*pairs))
    plan = dict(split='train', resolution=[1696, 816], source_family='donor', training_approved=False,
        source_cap_reset=False, physical_motion_commanded=False, hidden_cut_coordinates_executable=False,
        source_bindings=sources, implementation_bindings=review_code, prerequisite_bindings=review_code,
        anchor_pair_plan=str(anchor), target_cases=[dict(target_id='variant/Stem', conservative_view_cap_group='donor/Stem',
            base_pair_plan=str(anchor), base_pair_plan_sha256=pin(anchor),
            views=[r['requested_spec'] for r in rows])])
    result = dict(state=inv.COMPLETED, source_assets_unchanged=True, training_approved=False,
                  records=rows, captured_frames=3, automatically_clear_annotation_candidates=1)
    receipt = dict(state=inv.AUDITED, capture=str(root / 'capture'), records=reviews,
        counts={inv.STRICT: 1, inv.HOLD: 1, inv.EXCLUDE: 1}, training_approved=False,
        source_assets_unchanged=True, original_reviews_modified=False, review_code_bindings=review_code)
    fixture = dict(root=root, plan=plan, result=result, receipt=receipt,
                   request=dict(plan_path=str(root / 'plan.json'), training_started=False))
    reseal(fixture)
    return fixture


def build(case, **kwargs):
    return inv.build_inventory([case['pin']], **kwargs)


def test_all_rows_actual_image_bytes_and_decoded_digest(case):
    result = build(case)
    assert result['counts']['captured_rows'] == 3
    assert result['counts']['decisions'] == {inv.STRICT: 1, inv.HOLD: 1, inv.EXCLUDE: 1}
    for row in result['records']:
        assert row['source_family'] == row['original_donor_family'] == 'donor'
        assert row['source_target'] == row['conservative_view_cap_group'] == 'donor/Stem'
        assert row['variant_target'] == 'variant/Stem'
        assert row['image_bytes'] == (Path(row['sample_path']) / 'inputs/rgb.png').stat().st_size
        with Image.open(Path(row['sample_path']) / 'inputs/rgb.png') as image:
            assert row['decoded_rgb_sha256'] == decoded_rgb_digest(np.asarray(image).tobytes(), width=1696, height=816)
        assert row['scene_identity_basis']['static_scene_sha256'] in result['scene_contexts']
    assert result['exact_duplicates']['decoded_rgb_sha256']['duplicate_excess_rows'] == 2
    assert result['exact_duplicates']['scene_camera_sha256']['duplicate_excess_rows'] == 2
    assert not result['scope']['global_complete']

@pytest.mark.parametrize('change', ['recompress', 'noise'])
def test_encoded_bytes_are_not_decoded_identity_and_noise_not_new_pose(case, change):
    before = next(r for r in build(case)['records'] if r['candidate_id'] == 'strict')
    folder = case['root'] / 'capture/strict'
    with Image.open(folder / 'inputs/rgb.png') as image:
        pixels = np.asarray(image).copy()
    if change == 'noise':
        pixels[0, 0, 0] += 1
    Image.fromarray(pixels).save(folder / 'inputs/rgb.png', compress_level=0)
    metadata = json_at(folder / 'sample.json')
    metadata['files']['inputs/rgb.png']['sha256'] = pin(folder / 'inputs/rgb.png')
    metadata['synchronization']['freshness']['rgb_sha256'] = digest(pixels.tobytes())
    write(folder / 'sample.json', metadata); reseal(case)
    after = next(r for r in build(case)['records'] if r['candidate_id'] == 'strict')
    assert before['encoded_rgb_sha256'] != after['encoded_rgb_sha256']
    assert (before['decoded_rgb_sha256'] == after['decoded_rgb_sha256']) is (change == 'recompress')
    assert before['scene_camera_sha256'] == after['scene_camera_sha256']


def test_scene_scope_independent_of_other_bound_assets(case):
    reader = inv._Bindings()
    reader.mapping(case['plan']['source_bindings'])
    before = inv._scene_basis(reader, case['plan'])
    extra = case['root'] / 'package/unrelated.png'
    extra.write_bytes(b'not a scene dependency')
    reader.bind(extra, pin(extra))
    assert inv._scene_basis(reader, case['plan']) == before


def test_protected_generated_asset_output_and_wrong_plan(case):
    with pytest.raises(ValueError):
        inv.write_inventory(str(case['root'] / 'variant/forbidden.json'), [case['pin']])
    assert not (case['root'] / 'variant/forbidden.json').exists()
    other = case['root'] / 'plan-copy.json'
    other.write_bytes((case['root'] / 'plan.json').read_bytes())
    p = case['pin']
    with pytest.raises(ValueError, match='request/plan'):
        inv.build_inventory([inv.ReceiptPin(p.receipt_path, p.receipt_sha256, p.capture_path, str(other))])


@pytest.mark.parametrize('key', ['result_sha256', 'request_sha256', 'plan_sha256', 'review_code_bindings'])
def test_legacy_unbound_receipts_rejected(case, key):
    receipt = deepcopy(case['receipt']); receipt.pop(key)
    path = case['root'] / 'reviews/legacy.json'
    sha = write(path, receipt); p = case['pin']
    with pytest.raises(ValueError):
        inv.build_inventory([inv.ReceiptPin(str(path), sha, p.capture_path, p.plan_path)])


def test_pins_and_caller_inputs_are_unchanged(case):
    before = deepcopy(case)
    result = build(case)
    assert case == before
    with pytest.raises(ValueError):
        inv.build_inventory([str(case['root'])])
    assert not result['training_approved'] and not result['source_cap_reset'] and not result['admission_performed']
    assert result['scope']['audit_execution_independently_verified'] is False
    assert result['scope']['trace_probe_counters_independently_verified'] is False
    for row in result['records']:
        assert row['provenance']['lineage']['structural_lineage_checked'] is True
        assert row['provenance']['audit_declarations']['label_replayed_exact'] is True
        assert row['annotation_review']['audit_execution_independently_verified'] is False


def update_base(case, base):
    """Re-pin synthetic outer wrappers, leaving the original source plan fixed."""
    path = case['root'] / 'anchor.json'
    sha = write(path, base)
    case['plan']['source_bindings'][str(path)] = sha
    case['plan']['target_cases'][0]['base_pair_plan_sha256'] = sha
    reseal(case)


@pytest.mark.parametrize('fault', ['base_hash', 'coordinated_cap', 'source_row', 'source_job',
    'frozen_train', 'generated_component', 'generated_proposal', 'generated_attachment', 'source_reference'])
def test_lineage_anchored_beyond_resealed_capture_claims(case, fault):
    root = case['root']; base = json_at(root / 'anchor.json')
    if fault == 'base_hash':
        case['plan']['target_cases'][0]['base_pair_plan_sha256'] = '0' * 64
        reseal(case)
    elif fault == 'coordinated_cap':
        # All copied capture/review/base ancestry agrees, but the ORIGINAL row
        # and generated qualification still reserve donor/Stem. Must not pass.
        replacement = 'donor/Other'
        case['plan']['target_cases'][0]['conservative_view_cap_group'] = replacement
        base['conservative_view_cap_group'] = base['generated_row']['conservative_view_cap_group'] = replacement
        base['source_row']['target_id'] = replacement
        for row, review in zip(case['result']['records'], case['receipt']['records']):
            folder = root / 'capture' / row['candidate_id']
            meta, label = json_at(folder / 'sample.json'), json_at(folder / 'supervision/label.json')
            meta['supervision']['source_target_id'] = meta['supervision']['conservative_view_cap_group'] = replacement
            label['conservative_view_cap_group'] = review['source_target_group'] = replacement
            write(folder / 'sample.json', meta); write(folder / 'supervision/label.json', label)
        update_base(case, base)
    elif fault in ('source_row', 'generated_proposal', 'generated_attachment'):
        if fault == 'source_row': base['source_row']['cut_region_proposal'] = {'forged': True}
        if fault == 'generated_proposal': base['generated_row']['cut_region_proposal'] = {'forged': True}
        if fault == 'generated_attachment': base['generated_row']['attachment_plant_m'] = [1, 2, 3]
        update_base(case, base)
    else:
        relative = {'source_job': 'original/manifest.json', 'frozen_train': 'original_plan.json',
            'generated_component': 'variant/manifest.json', 'source_reference': 'original/reference/sample.json'}[fault]
        path = root / relative; value = json_at(path)
        if fault == 'source_job': value['collection_job_id'] = 'wrong_job'
        if fault == 'frozen_train': value['family_assignments']['donor'] = 'test'
        if fault == 'generated_component': value['components'][0]['id'] = 'Other'
        if fault == 'source_reference': value['supervision']['target_id'] = 'donor/Other'
        sha = write(path, value)
        base['source_bindings'][str(path)] = case['plan']['source_bindings'][str(path)] = sha
        if fault in ('frozen_train', 'generated_component'):
            # Keep all nested hashes coherent: fail the actual lineage rule,
            # not merely a stale wrapper hash.
            qualification_path = root / 'variant/qualification.json'
            qualification = json_at(qualification_path)
            if fault == 'frozen_train':
                original_path = root / 'original/manifest.json'
                original = json_at(original_path); original['source_collection_plan_sha256'] = sha
                original_sha = write(original_path, original)
                base['source_bindings'][str(original_path)] = case['plan']['source_bindings'][str(original_path)] = original_sha
                qualification['source_plan_sha256'] = sha
                qualification['frozen_family_assignments']['donor'] = 'test'
            else:
                qualification['output_hashes']['manifest.json'] = sha
            qualified_sha = write(qualification_path, qualification)
            base['source_bindings'][str(qualification_path)] = case['plan']['source_bindings'][str(qualification_path)] = qualified_sha
        update_base(case, base)
    with pytest.raises(ValueError): build(case)


@pytest.mark.parametrize('fault', ['passed_with_reasons', 'minimal', 'missing_native_flag', 'depth_reconstructed',
    'missing_probe_count', 'zero_probes', 'boolean_probes', 'too_many_pixels', 'missing_gap_evidence',
    'metrics_conflict', 'missing_metrics', 'query_arc', 'missing_query_evidence', 'gap_counter', 'false_gap'])
def test_complete_native_trace_required_even_when_wrappers_resealed(case, fault):
    folder = case['root'] / 'capture/strict'
    trace = deepcopy(case['result']['records'][0]['query_trace'])
    label = json_at(folder / 'supervision/label.json')
    if fault == 'passed_with_reasons': trace['reasons'] = ['query_to_cut_chain_too_dark']
    elif fault == 'minimal': trace = {'passed': True, 'reasons': []}
    elif fault == 'missing_native_flag': trace.pop('native_depth_reconstructed')
    elif fault == 'depth_reconstructed': trace['native_depth_reconstructed'] = True
    elif fault == 'missing_probe_count': trace.pop('probe_count')
    elif fault == 'zero_probes': trace['probe_count'] = 0
    elif fault == 'boolean_probes': trace['probe_count'] = True
    elif fault == 'too_many_pixels': trace['unique_pixels'] = trace['probe_count'] + 1
    elif fault == 'missing_gap_evidence': trace.pop('first_gap_probes')
    elif fault == 'metrics_conflict': trace['dark_fraction'] = .9
    elif fault == 'missing_metrics': trace.pop('maximum_projected_step_px')
    elif fault == 'query_arc': trace['arc_interval_m'][1] = .05
    elif fault == 'missing_query_evidence': label.pop('query_evidence')
    elif fault == 'gap_counter': trace['identity_or_depth_gap_count'] = 1
    elif fault == 'false_gap':
        trace.update(passed=False, reasons=['query_to_cut_chain_identity_or_depth_gap'],
            identity_or_depth_gap_count=1, first_gap_probes=[dict(arc_m=.01, pixel_xy=[820, 315],
                exact_target=False, native_depth_status='depth_consistent_not_visibility_verified')])
        case['result']['records'][0]['automatic_annotation_eligible'] = False
        case['result']['automatically_clear_annotation_candidates'] = 0
        case['receipt']['records'][0]['decision'] = inv.HOLD
        case['receipt']['counts'] = {inv.HOLD: 2, inv.EXCLUDE: 1}
    case['result']['records'][0]['query_trace'] = trace
    case['receipt']['records'][0]['trace_reasons'] = trace['reasons']
    write(folder / 'supervision/query_trace.json', trace)
    write(folder / 'supervision/label.json', label); reseal(case)
    with pytest.raises(ValueError): build(case)


def test_documented_early_out_of_frame_hold_is_not_discarded(case):
    trace = dict(passed=False, reasons=['query_to_cut_chain_out_of_frame'])
    case['result']['records'][1]['query_trace'] = trace
    case['receipt']['records'][1]['trace_reasons'] = trace['reasons']
    write(case['root'] / 'capture/hold/supervision/query_trace.json', trace); reseal(case)
    result = build(case)
    assert result['counts']['captured_rows'] == 3
    assert next(r for r in result['records'] if r['candidate_id'] == 'hold')['decision'] == inv.HOLD


def test_key_ignores_guard_epochs_timing_and_requested_framing(case):
    before = build(case)
    path = case['root'] / 'capture/strict/sample.json'
    meta = json_at(path)
    meta['synchronization']['static_guard'] = 'b' * 64
    meta['render_budget']['render_seconds'] = 999
    meta['robot_snapshot']['desired_cut_pixel_xy'] = [5, 6]
    meta['robot_snapshot']['framing_error_degrees'] = 999
    case['result']['records'][0]['robot_snapshot'] = meta['robot_snapshot']
    write(path, meta); reseal(case)
    after = build(case)
    assert [(r['scene_sha256'], r['camera_sha256']) for r in before['records']] == [
        (r['scene_sha256'], r['camera_sha256']) for r in after['records']]


@pytest.mark.parametrize('kind', ['camera', 'robot', 'plant'])
def test_actual_pose_changes_identity(case, kind):
    before = next(r for r in build(case)['records'] if r['candidate_id'] == 'strict')
    path = case['root'] / 'capture/strict/sample.json'; meta = json_at(path)
    if kind == 'camera':
        meta['calibration']['camera_to_world_usd_row_vectors'][3][0] = .1
        meta['synchronization']['freshness']['camera_sha256'] = fingerprint(meta['calibration'])
    elif kind == 'robot':
        meta['robot_snapshot']['joint_degrees']['head'] = 10.
        case['result']['records'][0]['robot_snapshot'] = meta['robot_snapshot']
    else:
        meta['supervision']['plant_to_world_usd_row_vectors'][3][0] = .1
    write(path, meta); reseal(case)
    after = next(r for r in build(case)['records'] if r['candidate_id'] == 'strict')
    assert before['scene_camera_sha256'] != after['scene_camera_sha256']


@pytest.mark.parametrize('relative', ['reviews/receipt.json', 'capture/result.json', 'capture/request.json',
    'plan.json', 'package/scene.usd', 'capture/strict/sample.json', 'capture/strict/inputs/rgb.png',
    'capture/strict/supervision/label.json', 'capture/strict/inputs/depth_m.npy'])
def test_stale_sources_rejected(case, relative):
    file = case['root'] / relative
    file.write_bytes(file.read_bytes() + b' ')
    with pytest.raises(ValueError):
        build(case)


@pytest.mark.parametrize('kind', ['omit', 'duplicate', 'target', 'donor', 'cap', 'callback', 'mask',
    'background', 'light', 'resolution', 'decision', 'claim', 'review_code', 'failed', 'sequence'])
def test_resealed_but_inconsistent_evidence_rejected(case, kind):
    path = case['root'] / 'capture/strict/sample.json'; meta = json_at(path)
    if kind == 'omit': case['receipt']['records'].pop()
    elif kind == 'duplicate': case['receipt']['records'].append(deepcopy(case['receipt']['records'][0]))
    elif kind == 'target': case['result']['records'][0]['target_id'] = 'other/Stem'
    elif kind == 'donor': meta['supervision']['split_group'] = 'other'
    elif kind == 'cap': meta['supervision']['conservative_view_cap_group'] = 'new_cap/Stem'
    elif kind == 'callback': meta['synchronization']['freshness']['depth_sha256'] = '0' * 64
    elif kind == 'sequence': meta['synchronization']['freshness']['callback_sequence'] = 2
    elif kind == 'mask':
        f = path.parent / 'supervision/target_visible.png'
        Image.fromarray(np.zeros((816, 1696), np.uint8)).save(f)
        meta['files']['supervision/target_visible.png']['sha256'] = pin(f)
    elif kind == 'background': meta['input_policy']['isolation'] = True
    elif kind == 'light': meta['lighting']['intensity'] = 2
    elif kind == 'resolution': meta['calibration']['resolution'] = [848, 408]
    elif kind == 'decision': case['receipt']['records'][0]['decision'] = inv.EXCLUDE
    elif kind == 'claim': case['receipt']['training_approved'] = True
    elif kind == 'review_code': case['receipt']['review_code_bindings'] = {}
    elif kind == 'failed': write(case['root'] / 'capture/failure.json', {'error': 'synthetic'})
    write(path, meta); reseal(case)
    with pytest.raises(ValueError): build(case)


def test_duplicate_receipt_and_wrong_capture_rejected(case):
    with pytest.raises(ValueError, match='Duplicate receipt/capture'):
        inv.build_inventory([case['pin'], case['pin']])
    p = case['pin']
    other = inv.ReceiptPin(p.receipt_path, p.receipt_sha256, str(case['root']), p.plan_path)
    with pytest.raises(ValueError): inv.build_inventory([other])


def test_extra_roots_explicit_never_silently_scanned(case):
    extra = case['root'] / 'unindexed'; extra.mkdir()
    result = build(case, extra_observed_roots=[inv.ObservedRoot(str(extra))])
    assert result['scope']['extra_observed_roots'][0]['status'] == 'unindexed_no_receipts'
    assert not result['scope']['global_complete']
    with pytest.raises(ValueError, match='outside declared'):
        inv.build_inventory([], extra_observed_roots=[inv.ObservedRoot(str(extra), (case['pin'],))])
    explicit = inv.build_inventory([], extra_observed_roots=[inv.ObservedRoot(str(case['root']), (case['pin'],))])
    assert len(explicit['records']) == 3


def test_write_create_only_and_disjoint(case, tmp_path):
    output = tmp_path / 'new.json'
    inv.write_inventory(str(output), [case['pin']])
    saved = output.read_bytes()
    with pytest.raises(ValueError): inv.write_inventory(str(output), [case['pin']])
    assert output.read_bytes() == saved
    with pytest.raises(ValueError):
        inv.write_inventory(str(case['root'] / 'capture/new.json'), [case['pin']])
    assert not (case['root'] / 'capture/new.json').exists()


def test_compact_reader_matches_raw_requires_explicit_storage(case):
    raw = build(case)
    compact = case['root'] / 'compact'; compact.mkdir()
    for row in case['result']['records']:
        pack_sample(case['root'] / 'capture' / row['candidate_id'], compact / row['candidate_id'],
                    sample_sha256=row['sample_sha256'], label_sha256=row['label_sha256'], query_trace=row['query_trace'])
    receipt = deepcopy(case['receipt'])
    for row in receipt['records']: row['sample'] = str(compact / Path(row['sample']).name)
    path = case['root'] / 'reviews/compact.json'; sha = write(path, receipt)
    p = case['pin']
    pinned = inv.ReceiptPin(str(path), sha, p.capture_path, p.plan_path, str(compact))
    rebuilt = inv.build_inventory([pinned])
    for a, b in zip(raw['records'], rebuilt['records']):
        for field in ('decoded_rgb_sha256', 'encoded_rgb_sha256', 'scene_camera_sha256', 'decision', 'sample_id'):
            assert a[field] == b[field]
    with pytest.raises(ValueError): inv.build_inventory([inv.ReceiptPin(str(path), sha, p.capture_path, p.plan_path)])


def test_no_source_mutation_and_final_recheck(case, monkeypatch, tmp_path):
    before = {str(p): pin(p) for p in case['root'].rglob('*') if p.is_file()}
    build(case)
    assert before == {str(p): pin(p) for p in case['root'].rglob('*') if p.is_file()}
    original = inv._Bindings.finish
    def mutate(reader):
        (case['root'] / 'package/scene.usd').write_bytes(b'changed concurrently')
        original(reader)
    monkeypatch.setattr(inv._Bindings, 'finish', mutate)
    with pytest.raises(ValueError, match='changed during'):
        inv.write_inventory(str(tmp_path / 'never.json'), [case['pin']])
    assert not (tmp_path / 'never.json').exists()


def test_duplicate_json_keys_rejected(case):
    path = case['root'] / 'reviews/receipt.json'
    raw = path.read_text(); path.write_text(raw.replace('{', '{"state":"fake",', 1))
    p = case['pin']
    with pytest.raises(ValueError, match='Duplicate JSON'):
        inv.build_inventory([inv.ReceiptPin(str(path), pin(path), p.capture_path, p.plan_path)])


def test_zero_capture_receipt_keeps_coverage_without_training_counts(case):
    for row in case['result']['records']:
        row['state'] = 'rejected_pose'
    case['result']['captured_frames'] = 0
    case['result']['automatically_clear_annotation_candidates'] = 0
    case['receipt']['records'] = []
    case['receipt']['counts'] = {}
    reseal(case)
    result = build(case)
    assert result['records'] == []
    assert result['scope']['explicit_receipts'][0]['noncaptured_rows'] == 3
    assert not result['scope']['global_complete']
