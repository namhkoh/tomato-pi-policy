"""Independent saved-callback, original geometry and strict-quality reset8 replay."""
from pathlib import Path
import hashlib
import numpy as np
from . import native848_original_short_plan_v1 as api
from . import native848_short_validation_v5 as raw
from .native848_pair_audit_v2 import image_array, verify_camera, expected_catalogue, world_from_row
from .capture_sensor import LEGACY_RESOLUTION, sensor_profile
from .capture_contract import fingerprint, project, depth_evidence, jsonable
from .capture_visibility import component_masks, interval_visibility, view_quality, ORGAN_IDS
from .native_greenhouse_pair import assert_same_camera
from .dataset_review import require, read_json, safe_file, verify_bindings
from .depth_preview import sha256

SCHEMA = 'greenhouse.original848_short_buffer_geometry_audit.v1'
STATE = 'original848_short_buffers_geometry_replayed_pending_individual_visual_review'


def audit_capture(capture, *, plan_path, plan_sha256, result_sha256):
    from .collection_plan import load_plan
    from .native_dataset.clear848_workspace_v2 import WorkspaceChecker
    capture, plan_path = Path(capture).resolve(), Path(plan_path).resolve()
    require(sha256(plan_path) == plan_sha256 and sha256(capture / 'result.json') == result_sha256
        and not (capture / 'failure.json').exists(), 'Changed or failed original short capture')
    plan = read_json(plan_path)
    cache, anchor, report = api.check(plan, full=True)
    result, request = read_json(capture / 'result.json'), read_json(capture / 'request.json')
    require(result['state'] == api.RESULT_STATE and result['resolution'] == [848, 408]
        and result['mode'] == plan['mode'] and result['profile'] == api.PROFILE
        and result['render_budget_subframes'] == api.BUDGET and result['source_assets_unchanged'] is True
        and result['full_greenhouse_stage_count'] == result['capture_render_product_count'] == 1
        and result['pose_search_attempts'] == result['runtime_head_optimizations'] == 0
        and result['training_approved'] is False and result['source_cap_reset'] is False
        and result['geometry_novelty_qualified'] is False and result['accepted_training_increment'] == 0
        and result['plan_sha256'] == request['plan_sha256'] == plan_sha256
        and Path(request['plan_path']).resolve() == plan_path and request['automatic_retries'] is False
        and result['render_probes'] == [] and result['short_profile_comparisons_passed'] is False
        and result['established56_every_frame'] is False, 'Changed explicit original short scope')
    require(request['process_admission']['no_unrelated_kit_process_at_admission'] is True
        and request['host_memory_preflight']['commit_headroom_bytes'] >= 20*2**30
        and request['available_disk_bytes'] >= 60*2**30, 'Changed native admission')
    pins = read_json(capture / 'evidence_bindings.json')
    actual_files = {str(p.resolve()) for p in capture.rglob('*') if p.is_file()
        and p not in (capture / 'result.json', capture / 'evidence_bindings.json')}
    require(set(pins) == actual_files, 'Incomplete raw/processed capture manifest')
    verify_bindings(pins)
    require(not any(capture.rglob('unrequested_callbacks')), 'Unrequested callbacks outside explicit captures')
    for path in (plan_path, capture / 'result.json', capture / 'evidence_bindings.json'):
        pins[str(path)] = sha256(path)
    q = api.qualification(plan)
    pins.update(q['source_bindings'])
    pins[str(Path(plan['qualification_evidence']['path']).resolve())] = plan['qualification_evidence']['sha256']
    adapted = api.adaptation_evidence(plan)
    if adapted is not None:
        pins.update(adapted['source_bindings'])
        pins[str(Path(plan['original_adaptation_evidence']['path']).resolve())] = plan['original_adaptation_evidence']['sha256']
    warmup = raw.verify_request(capture / 'warmup', [8]*7)
    require(warmup['settings'] == plan['render_settings'], 'Changed warmup settings')
    phase = read_json(capture / 'source_phase_bindings.json')
    verify_bindings(phase['original_source_bindings'])
    verify_bindings(phase['full_scene_source_bindings'])
    pins.update(phase['original_source_bindings'])
    pins.update(phase['full_scene_source_bindings'])
    require(phase['original_sources_verified_after_warmup'] is True
        and phase['full_scene_baseline_phase'] == 'after_explicit56_warmup_original_scene_no_substitution'
        and phase['source_hash_validation_removed'] is False
        and phase['warmup_request_sha256'] == sha256(capture / 'warmup/request_evidence.json'), 'Changed source lifecycle')
    guard = read_json(capture / 'static_monitor_initialization.json')
    require(guard['phase'] == 'after_explicit56_warmup_before_first_short_request'
        and guard['warmup_request_sha256'] == phase['warmup_request_sha256']
        and guard['static_guard_exclusions_modified'] is False, 'Changed static monitor lifecycle')
    geometry_reference = read_json(capture / 'geometry_cache_reference.json')
    require(geometry_reference['cached_result_equal'] is True
        and geometry_reference['generated_plant_triangle_refinement'] is True, 'Full screen reference missing')
    require([{k: d[k] for k in ('sample_id', 'source_pose_id', 'capture_role')} for d in result['decisions']]
        == plan['schedule'], 'Unaccounted planned observations')
    captured = []
    for decision in result['decisions']:
        name = decision['sample_id']
        require(read_json(capture / ('decision_'+name+'.json')) == decision, 'Decision changed')
        screen = read_json(capture / ('geometry_screen_'+name+'.json'))
        require(screen['sample_id'] == name and screen['source_pose_id'] == decision['source_pose_id']
            and screen['generated_plant_triangle_refinement'] is True, 'Geometry receipt differs')
        state = decision['state']
        require(state in ('rejected_native_geometry', 'captured_pending_audit_and_quality')
            and (screen['screen']['passed'] is True) == (state == 'captured_pending_audit_and_quality'), 'Invalid geometry decision')
        if state == 'captured_pending_audit_and_quality':
            captured.append(decision)
        else:
            require(plan['mode'] == 'production', 'Missing adaptation control')
    require([s['sample_id'] for s in result['samples']] == [r['sample_id'] for r in captured]
        and guard['first_sample_id'] == captured[0]['sample_id'], 'Missing or reordered samples')
    cache_by = {r['sample_id']: r for r in cache['records']}
    _, reports = load_plan(anchor['source_collection_plan'])
    expected_components = expected_catalogue(anchor, 'original_control', {r['plant_id']: r for r in reports}, None)
    manifest = read_json(Path(anchor['source_capture']) / 'manifest.json')
    previous = None
    previous_request = warmup['requests'][-1]
    records, controls = [], []
    workspace = WorkspaceChecker()
    try:
        for index, decision in enumerate(captured):
            name, source_id = decision['sample_id'], decision['source_pose_id']
            folder = capture / name
            mp = folder / 'sample.json'
            meta, rec = read_json(mp), cache_by[source_id]
            require(meta == result['samples'][index] and meta['schema_version'] == api.SAMPLE_SCHEMA
                and meta['sample_id'] == name and meta['source_pose_id'] == source_id
                and meta['capture_role'] == decision['capture_role']
                and meta['training_sample_approved'] is False and meta['native_instance_backend'] == 'fast'
                and meta['historical_labels_inherited'] is False and meta['sensor'] == sensor_profile(LEGACY_RESOLUTION)
                and meta['scene_counts'] == anchor['expected_scene_counts']
                and meta['lighting'] == manifest['lighting'] and meta['renderer'] == manifest['renderer']
                and meta['input_policy'] == dict(clean_full_scene=True, isolation=False, diagnostic_overlays=False)
                and meta['profile'] == api.PROFILE and meta['render_budget_subframes'] == api.BUDGET
                and meta['qualification_evidence_path'] == plan['qualification_evidence']['path']
                and meta['qualification_evidence_sha256'] == plan['qualification_evidence']['sha256'], 'Changed sample policy')
            pose = meta['robot_snapshot']
            require(pose['joint_degrees'] == rec['joint_degrees'] and pose['joint_limits_checked'] is True
                and np.allclose(pose['robot_root_to_world_usd_row_vectors'], rec['robot_root_to_world_usd_row_vectors'], atol=1e-9, rtol=0)
                and np.allclose(pose['camera_to_head_column_vectors'], rec['camera_to_head_column_vectors'], atol=1e-9, rtol=0)
                and abs(pose['floor_alignment']['vertical_adjustment_m']) < 1e-8, 'Changed embodied pose')
            require(meta['geometry_screen'] == pose['visual_bound_screen']
                == read_json(capture / ('geometry_screen_'+name+'.json'))['screen']
                and meta['geometry_screen']['passed'] is True, 'Missing actual full scene geometry screen')
            require(meta['direct_camera'] == dict(pose_cache_sha256=plan['cache_sha256'], cached_sample_id=source_id,
                anchor_reference_sha256=rec['anchor_reference_sha256'], runtime_pose_search=False,
                runtime_head_optimization=False, stage_reused=True, render_product_reused=True), 'Changed camera cache provenance')
            for relative, info in meta['files'].items():
                require(sha256(safe_file(folder, relative)) == info['sha256'], 'Changed processed native artifact')
            cal = meta['calibration']
            assert_same_camera(cal, rec['calibration'])
            verify_camera(meta)
            sync = meta['synchronization']
            evidence = raw.verify_request(capture / 'raw' / name, [api.BUDGET])
            current_request = evidence['requests'][0]
            require(current_request['request_index_before'] == previous_request['request_index_after']
                and current_request['callback_sequence_before'] == previous_request['callback_sequence_after'],
                'Unaccounted requests or callbacks between warmup and planned snapshots')
            previous_request = current_request
            require(sync['method'] == 'frozen_scene_single_native_writer_payload'
                and sync['scene_unchanged_during_capture'] is True and sync['dynamic_recording_supported'] is False
                and sync['profile'] == api.PROFILE and sync['render_budget_subframes'] == api.BUDGET
                and sync['request_evidence'] == evidence and meta['render_settings'] == evidence['settings'] == plan['render_settings']
                and sync['photometric_identity_to_long_render_claimed'] is False, 'Changed budget/settings/raw request')
            if index == 0:
                require(sync['static_guard'] == guard['baseline_token'], 'First request differs from initialized static guard')
            payload = raw.load_raw(capture / 'raw' / name / 'callback_00')
            changed = previous is not None and fingerprint(cal) != previous['camera_sha256']
            rgb, depth, valid, reference, token, ids, mapping = raw.freshness(payload, cal, sync['static_guard'],
                sync['static_guard_after'], evidence['requests'][0]['callback_sequence_after'], previous, changed)
            previous = token
            require(token == sync['freshness'] and reference == sync['reference_time']
                and jsonable(payload['pilot_render_frame']) == sync['native_render_frame'], 'Freshness/reference replay differs')
            require(np.array_equal(rgb, image_array(folder / 'inputs/rgb.png'))
                and np.array_equal(depth, np.load(folder / 'inputs/depth_m.npy', allow_pickle=False), equal_nan=True)
                and np.array_equal(valid, image_array(folder / 'inputs/depth_valid.png') == 255), 'Processed RGB/Z differ from raw callback')
            identity = read_json(folder / 'supervision/identities.json')
            catalogue = identity['component_catalogue']
            require(catalogue == expected_components and len(catalogue) == meta['scene_counts']['components']
                and identity['organ_ids'] == ORGAN_IDS and {int(k): v for k, v in identity['renderer_id_to_prim'].items()} == mapping,
                'Changed full original component hierarchy')
            require(np.array_equal(ids, np.load(folder / 'supervision/renderer_instance_id.npy', allow_pickle=False))
                and meta['native_instance_sha256'] == hashlib.sha256(ids.tobytes()).hexdigest()
                and meta['native_mapping_sha256'] == fingerprint(mapping), 'Changed native identity')
            components, organs, owners = component_masks(ids, mapping, catalogue)
            require(np.array_equal(components, np.load(folder / 'supervision/component_id.npy', allow_pickle=False))
                and np.array_equal(organs, image_array(folder / 'supervision/organ_type.png')), 'Derived mask replay differs')
            row, sup = rec['source_row'], meta['supervision']
            world = world_from_row(anchor, row)
            require(sup['target_id'] == sup['source_target_id'] == sup['conservative_view_cap_group'] == row['target_id']
                and sup['split_group'] == anchor['source_family'] and sup['cut_region_proposal'] == row['cut_region_proposal']
                and sup['cut_safety_validated'] is False, 'Changed target ancestry')
            require(all(np.allclose(sup[k], v, atol=1e-9, rtol=0) for k, v in world.items()), 'Changed metric cut geometry')
            target = next(c for c in catalogue if c['variant_id'] == row['variant_id'] and c['component_id'] == row['component_id'])
            radius = row['cut_region_proposal']['nominal']['petiole_radius_m']
            nominal, interval = project([world['nominal_world_m']], cal)[0], project(world['interval_world_m'], cal)
            visibility, mask = interval_visibility(nominal, interval, depth, valid, ids, mapping, owners, target, radius)
            require(sup['nominal_projected'] == nominal and sup['projected_interval'] == interval
                and sup['depth_evidence'] == depth_evidence(nominal, depth, valid, radius)
                and sup['visibility_evidence'] == visibility and meta['quality'] == view_quality(cal, nominal, interval, radius, visibility, rgb, mask)
                and meta['native_target_pixels'] == int(mask.sum())
                and np.array_equal(image_array(folder / 'supervision/target_visible.png'), mask.astype(np.uint8)*255), 'Visibility replay differs')
            generated = [i for i, p in mapping.items() if p.startswith('/World/GeneratedNativePilot/')]
            require(meta['original_geometry_only'] is True and meta['generated_plant_native_pixels'] == int(np.isin(ids, generated).sum()) == 0,
                'Generated geometry entered original frame')
            annotation = raw.prior.annotations(meta, report, rgb, depth, valid, components, catalogue, mask)
            require(annotation == read_json(folder / 'annotation.json'), 'Strict label/full selected trace replay differs')
            saved_workspace = read_json(folder / 'workspace.json')
            verify_bindings(saved_workspace['source_bindings'])
            proof = workspace.check_sample(mp, sha256(mp))
            require(proof['result']['workspace_passed'] == saved_workspace['result']['workspace_passed']
                and proof['target_id'] == annotation['label']['target_id']
                and np.allclose(proof['nominal_world_m'], annotation['label']['nominal_world_m'], atol=1e-8, rtol=0), 'Workspace replay differs')
            automated = bool(annotation['label']['eligible'] and annotation['query_trace'] is not None
                and annotation['query_trace']['passed'] and proof['result']['workspace_passed'])
            require(automated == decision['actual_automated_criteria_passed'], 'Changed automated quality conclusion')
            out = dict(sample_id=name, source_pose_id=source_id, capture_role=decision['capture_role'],
                target_id=row['target_id'], source_target=row['target_id'], sample_path=str(mp), sample_sha256=sha256(mp),
                rgb_sha256=sha256(folder / 'inputs/rgb.png'), anchor_reference_path=rec['anchor_reference_path'],
                anchor_reference_sha256=rec['anchor_reference_sha256'], same_callback_buffers_replayed=True,
                full_native_identity_masks_replayed=True, metric_cut_geometry_recomputed=True,
                actual_automated_criteria_passed=automated, render_profile_qualified=plan['mode'] == 'production',
                profile=api.PROFILE, render_budget_subframes=api.BUDGET,
                qualification_evidence_path=plan['qualification_evidence']['path'],
                qualification_evidence_sha256=plan['qualification_evidence']['sha256'], accepted_training_increment=0)
            (controls if decision['capture_role'] == 'adaptation_control' else records).append(out)
    finally:
        workspace.finish()
    if plan['mode'] == 'adaptation':
        passed = len(controls) == 3 and all(r['actual_automated_criteria_passed'] for r in controls)
        runtime = dict(passed=passed, sequence=['A', 'B', 'A'], control_sample_ids=[r['sample_id'] for r in controls],
            source_pose_ids=[r['source_pose_id'] for r in controls], evidence_path=None, evidence_sha256=None)
    else:
        runtime = dict(passed=True, sequence=['A', 'B', 'A'],
            control_sample_ids=read_json(adapted['audit_path'])['runtime_adaptation']['control_sample_ids'],
            source_pose_ids=read_json(adapted['audit_path'])['runtime_adaptation']['source_pose_ids'],
            evidence_path=plan['original_adaptation_evidence']['path'], evidence_sha256=plan['original_adaptation_evidence']['sha256'])
    verify_bindings(pins)
    return dict(schema=SCHEMA, state=STATE, mode=plan['mode'], profile=api.PROFILE, render_budget_subframes=api.BUDGET,
        records=records, adaptation_records=controls, runtime_adaptation=runtime, source_bindings=pins,
        native_frames=len(records)+len(controls), pose_search_attempts=0, established56_every_frame=False,
        short_profile_comparisons_passed=False, training_approved=False, strict_clarity_qualified=False,
        original_workspace_qualified=False, geometry_novelty_qualified=False, source_cap_reset=False, accepted_training_increment=0)
