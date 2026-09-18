"""Replay actual cached-pose native848 buffers and metric target geometry."""
from pathlib import Path
import hashlib
import numpy as np
from . import native848_direct_plan_v2 as api
from .native848_pair_audit_v2 import image_array, verify_camera, expected_catalogue, world_from_row
from .capture_sensor import LEGACY_RESOLUTION, sensor_profile
from .capture_contract import fingerprint, project, depth_evidence
from .capture_visibility import component_masks, interval_visibility, view_quality, ORGAN_IDS
from .native_sensor_payload import decode_native_instances
from .native_greenhouse_pair import assert_same_camera
from .dataset_review import require, read_json, safe_file, verify_bindings
from .depth_preview import sha256


def audit_capture(capture, *, plan_path, plan_sha256, result_sha256):
    from .collection_plan import load_plan
    capture, plan_path = Path(capture).resolve(), Path(plan_path).resolve()
    require(sha256(plan_path) == plan_sha256 and sha256(capture/'result.json') == result_sha256
        and not (capture/'failure.json').exists(), 'Changed or failed direct batch')
    plan = read_json(plan_path)
    cache, plans, generated = api.check(plan, full=True)
    result, request = read_json(capture/'result.json'), read_json(capture/'request.json')
    require(result['state'] == api.RESULT_STATE and result['resolution'] == [848, 408]
        and result['profile'] == api.PROFILE and result['source_assets_unchanged'] is True
        and result['full_greenhouse_stage_count'] == result['capture_render_product_count'] == 1
        and result['pose_search_attempts'] == result['runtime_head_optimizations'] == 0
        and result['training_approved'] is False and result['source_cap_reset'] is False
        and result['plan_sha256'] == request['plan_sha256'] == plan_sha256
        and Path(request['plan_path']) == plan_path, 'Changed direct batch scope')
    require(request['process_admission']['no_unrelated_kit_process_at_admission'] is True
        and request['host_memory_preflight']['commit_headroom_bytes'] >= 20*2**30
        and request['available_disk_bytes'] >= 60*2**30, 'Changed native admission')
    smoke = read_json(capture/'calibration_smoke.json')
    require(sha256(capture/'calibration_smoke.json') == result['calibration_smoke_sha256']
        and smoke['passed'] is True and smoke['native_identity_and_occluder_smoke_passed'] is True,
        'Native optical-Z/identity/occluder smoke missing')
    pins = {str(p): sha256(p) for p in (plan_path, capture/'result.json', capture/'request.json',
        capture/'calibration_smoke.json', capture/'geometry_cache_reference.json')}
    geometry_reference = read_json(capture/'geometry_cache_reference.json')
    require(geometry_reference['cached_result_equal'] is True
        and geometry_reference['generated_plant_triangle_refinement'] is True, 'Cached screen reference missing')
    cached = {r['sample_id']: r for r in cache['records']}
    first = plans[cached[plan['selected_sample_ids'][0]]['pair_plan_path']]
    _, loaded = load_plan(first['source_collection_plan'])
    reports = {r['plant_id']: r for r in loaded}
    manifest = read_json(Path(first['source_capture'])/'manifest.json')
    expected_ids = [d['sample_id'] for d in result['decisions'] if d['state'] == 'captured_pending_audit_and_quality']
    require([d['sample_id'] for d in result['decisions']] == plan['selected_sample_ids']
        and [s['sample_id'] for s in result['samples']] == expected_ids, 'Unaccounted planned frames')
    for decision in result['decisions']:
        dp = capture/('decision_'+decision['sample_id']+'.json')
        gp = capture/('geometry_screen_'+decision['sample_id']+'.json')
        pins[str(dp)], pins[str(gp)] = sha256(dp), sha256(gp)
        require(read_json(dp) == decision, 'Decision receipt changed')
        screen = read_json(gp)['screen']
        require((screen['passed'] is True) == (decision['state'] == 'captured_pending_audit_and_quality'),
            'Geometry rejection/capture mismatch')
    require(result['render_probes'] == [] and result['short_profile_comparisons_passed'] is False
        and result['established56_every_frame'] is True, 'Changed established56 profile scope')
    qualified = False
    previous = None
    records = []
    for index, name in enumerate(expected_ids):
        folder = capture/name
        mp = folder/'sample.json'
        meta, rec = read_json(mp), cached[name]
        pair = plans[rec['pair_plan_path']]
        pins[str(mp)] = sha256(mp)
        require(meta == result['samples'][index], 'Saved/result sample differs')
        require(meta['schema_version'] == api.SAMPLE_SCHEMA and meta['training_sample_approved'] is False
            and meta['native_instance_backend'] == 'fast' and meta['historical_labels_inherited'] is False
            and meta['sensor'] == sensor_profile(LEGACY_RESOLUTION)
            and meta['scene_counts'] == first['expected_scene_counts']
            and meta['lighting'] == manifest['lighting'] and meta['renderer'] == manifest['renderer']
            and meta['input_policy'] == dict(clean_full_scene=True, isolation=False, diagnostic_overlays=False),
            'Changed native848 camera/scene policy')
        pose = meta['robot_snapshot']
        require(pose['joint_degrees'] == rec['joint_degrees'] and pose['joint_limits_checked'] is True
            and np.allclose(pose['robot_root_to_world_usd_row_vectors'], rec['robot_root_to_world_usd_row_vectors'], atol=1e-9, rtol=0)
            and np.allclose(pose['camera_to_head_column_vectors'], rec['camera_to_head_column_vectors'], atol=1e-9, rtol=0)
            and abs(pose['floor_alignment']['vertical_adjustment_m']) < 1e-8, 'Actual robot differs from cached pose')
        require(meta['geometry_screen'] == pose['visual_bound_screen']
            == read_json(capture/('geometry_screen_'+name+'.json'))['screen']
            and meta['geometry_screen']['passed'] is True, 'Missing actual geometry screen')
        direct = meta['direct_camera']
        require(direct == dict(pose_cache_sha256=plan['cache_sha256'], cached_sample_id=name,
            pair_plan_sha256=rec['pair_plan_sha256'], runtime_pose_search=False,
            runtime_head_optimization=False, stage_reused=True, render_product_reused=True), 'Cache provenance differs')
        for relative, info in meta['files'].items():
            path = safe_file(folder, relative)
            require(sha256(path) == info['sha256'], 'Saved native artifact changed')
            pins[str(path)] = info['sha256']
        cal = meta['calibration']
        assert_same_camera(cal, rec['calibration'])
        verify_camera(meta)
        rgb = image_array(folder/'inputs/rgb.png')
        depth = np.load(folder/'inputs/depth_m.npy', allow_pickle=False)
        rawvalid = image_array(folder/'inputs/depth_valid.png')
        valid = rawvalid == 255
        require(rgb.shape == (408, 848, 3) and rgb.dtype == np.uint8 and depth.shape == (408, 848)
            and depth.dtype == np.float32 and rawvalid.shape == (408, 848) and set(np.unique(rawvalid)) <= {0, 255},
            'Invalid native848 arrays')
        near, far = cal['clipping_range_m']
        require(np.array_equal(valid, np.isfinite(depth) & (depth>0) & (depth>=near) & (depth<=far)), 'Depth validity differs')
        sync, fresh = meta['synchronization'], meta['synchronization']['freshness']
        budget = 56
        require(sync['method'] == 'frozen_scene_single_native_writer_payload'
            and sync['scene_unchanged_during_capture'] is True and sync['dynamic_recording_supported'] is False
            and sync['profile'] == api.PROFILE and sync['render_budget_subframes'] == budget,
            'Changed temporal contract')
        require(fresh['rgb_sha256'] == hashlib.sha256(rgb.tobytes()).hexdigest()
            and fresh['depth_sha256'] == hashlib.sha256(depth.tobytes()).hexdigest()
            and fresh['camera_sha256'] == fingerprint(cal), 'Native callback hashes differ')
        if previous is not None:
            require(fresh['callback_sequence'] > previous['callback_sequence']
                and all(fresh[k] != previous[k] for k in ('rgb_sha256', 'depth_sha256', 'camera_sha256')),
                'Stale buffers after camera move')
        previous = fresh
        ids = np.load(folder/'supervision/renderer_instance_id.npy', allow_pickle=False)
        identity = read_json(folder/'supervision/identities.json')
        catalogue = identity['component_catalogue']
        require(catalogue == expected_catalogue(pair, 'generated_variant', reports, generated)
            and len(catalogue) == meta['scene_counts']['components'] and identity['organ_ids'] == ORGAN_IDS,
            'Full native component hierarchy differs')
        mapping = {int(k): v for k, v in identity['renderer_id_to_prim'].items()}
        decode_native_instances({'instance_id_segmentation': {'data': ids, 'info': {'idToLabels': mapping}}}, LEGACY_RESOLUTION)
        require(meta['native_instance_sha256'] == hashlib.sha256(ids.tobytes()).hexdigest()
            and meta['native_mapping_sha256'] == fingerprint(mapping), 'Native identity hashes differ')
        components, organs, owners = component_masks(ids, mapping, catalogue)
        require(np.array_equal(components, np.load(folder/'supervision/component_id.npy', allow_pickle=False))
            and np.array_equal(organs, image_array(folder/'supervision/organ_type.png')), 'Derived masks differ')
        row, sup = pair['generated_row'], meta['supervision']
        world = world_from_row(pair, row)
        require(sup['target_id'] == row['target_id'] and sup['source_target_id'] == sup['conservative_view_cap_group'] == pair['conservative_view_cap_group']
            and sup['split_group'] == pair['source_family'] and sup['cut_region_proposal'] == row['cut_region_proposal']
            and sup['cut_safety_validated'] is False, 'Changed target ancestry')
        for key, value in world.items():
            require(np.allclose(sup[key], value, atol=1e-9, rtol=0), 'Metric geometry differs: '+key)
        target = next(c for c in catalogue if c['variant_id'] == row['variant_id'] and c['component_id'] == row['component_id'])
        nominal, interval = project([world['nominal_world_m']], cal)[0], project(world['interval_world_m'], cal)
        radius = row['cut_region_proposal']['nominal']['petiole_radius_m']
        visibility, mask = interval_visibility(nominal, interval, depth, valid, ids, mapping, owners, target, radius)
        require(sup['nominal_projected'] == nominal and sup['projected_interval'] == interval
            and sup['depth_evidence'] == depth_evidence(nominal, depth, valid, radius)
            and sup['visibility_evidence'] == visibility
            and meta['quality'] == view_quality(cal, nominal, interval, radius, visibility, rgb, mask)
            and meta['native_target_pixels'] == int(mask.sum())
            and np.array_equal(image_array(folder/'supervision/target_visible.png'), mask.astype(np.uint8)*255),
            'Native visibility replay differs')
        oldroot = pair['original_variant']['plant_root']
        oldids = [i for i, path in mapping.items() if path == oldroot or path.startswith(oldroot+'/')]
        require(meta['old_plant_native_pixels'] == int(np.isin(ids, oldids).sum()) == 0, 'Original foreground remains visible')
        records.append(dict(sample_id=name, target_id=row['target_id'], source_target=pair['conservative_view_cap_group'],
            sample_path=str(mp), sample_sha256=sha256(mp), rgb_sha256=sha256(folder/'inputs/rgb.png'),
            pair_plan_path=rec['pair_plan_path'], pair_plan_sha256=rec['pair_plan_sha256'],
            same_callback_buffers_replayed=True, full_native_identity_masks_replayed=True,
            metric_cut_geometry_recomputed=True, render_profile_qualified=budget == 56,
            accepted_training_increment=0))
    verify_bindings(pins)
    return dict(schema='greenhouse.native848_direct_buffer_geometry_audit.v2',
        state='direct_native848_buffers_geometry_replayed_pending_clarity_workspace_novelty',
        records=records, source_bindings=pins, short_profile_comparisons_passed=qualified,
        native_frames=len(records), pose_search_attempts=0, established56_every_frame=True, training_approved=False,
        strict_clarity_qualified=False, generated_workspace_qualified=False,
        geometry_novelty_qualified=False, source_cap_reset=False, accepted_training_increment=0)
