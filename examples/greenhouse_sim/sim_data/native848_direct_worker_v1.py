"""One loaded greenhouse and mounted camera; apply cached poses and capture.

No broad viewpoint search, runtime head optimization, physics trajectory, image
resize, source-cap reset, or automatic training acceptance is performed here.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import shutil
import time
import traceback
import numpy as np
from . import native848_direct_plan_v1 as api
from .dataset_review import require, read_json, write_json, verify_bindings
from .depth_preview import sha256
from .capture_contract import fingerprint, jsonable, project, depth_evidence, write_sample
from .capture_sensor import LEGACY_RESOLUTION, calibration_for_native_resolution, sensor_profile
from .native_greenhouse_pair import assert_same_camera
from .native_generated_pair import windows_worker_admission


def apply_cached_pose(stage, robot, record):
    from pxr import Usd, UsdGeom
    from greenhouse_sim import robot_hardware, robot_kinematics
    from .capture_scene import calibration, mounted_camera_to_head
    from .floor_alignment import align_robot_to_floor, PACKAGE_FLOOR
    root = np.asarray(record['robot_root_to_world_usd_row_vectors']).T
    mount = np.asarray(record['camera_to_head_column_vectors'])
    require(np.allclose(mounted_camera_to_head(stage), mount, atol=1e-9, rtol=0), 'Live camera mount differs')
    links = robot_kinematics.Rby1Kinematics().all_link_transforms(record['joint_degrees'])
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        robot_hardware._set_transform(stage.GetPrimAtPath(robot['root']), root[:3, :3], root[:3, 3])
        for name, matrix in links.items():
            prim = stage.GetPrimAtPath(robot['root'] + '/' + name)
            require(bool(prim), 'Missing robot link ' + name)
            robot_hardware._set_transform(prim, matrix[:3, :3], matrix[:3, 3])
    floor = align_robot_to_floor(stage, robot['root'], PACKAGE_FLOOR)
    actual = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(robot['root'])))
    require(abs(floor['vertical_adjustment_m']) < 1e-8 and np.allclose(actual, root.T, atol=1e-9, rtol=0),
        'Cached floor support differs; do not alter the precomputed camera')
    cal = calibration_for_native_resolution(calibration(stage), LEGACY_RESOLUTION)
    assert_same_camera(cal, record['calibration'])
    return dict(joint_degrees=record['joint_degrees'], robot_root_to_world_usd_row_vectors=actual.tolist(),
        camera_to_head_column_vectors=mount.tolist(), floor_alignment=floor,
        desired_cut_pixel_xy=record['requested_spec']['desired_pixel_xy'],
        framing_error_degrees=record['framing_error_degrees'],
        pose_sampling='precomputed_bounded_reference_and_head_joints.v1',
        source_body_orientation_preserved=True, arm_and_torso_joints_preserved=True,
        joint_limits_checked=True, whole_robot_collision_checked=False,
        motion_between_snapshots_validated=False, arm_reachability='pending_fresh_workspace_check'), cal


def capture(app, output, plan, cache, plans):
    import omni.replicator.core as rep
    from PIL import Image
    from .native_scene import prepare_native_scene
    from .generated_capture import substitute_plant
    from .capture_pilot import calibration_smoke, make_writer, source_hashes
    from .capture_scene import calibration, target_world_geometry
    from .capture_viewpoints import component_catalogue, static_obstacles, scene_triangle_refiner, visible_bounds
    from .capture_visibility import component_masks, interval_visibility, view_quality, write_visibility
    from .native_sensor_payload import validate_native_static, decode_native_instances
    from .static_guard import StaticSceneMonitor
    from .static_geometry_cache import StaticGeometryScreenCache
    from .training_screen import StaticBoundScreen
    from .static_render import reference_payload, consolidated_payload, convergence, LONG_RGB_LIMITS
    from .review_camera import HEAD_CAMERA
    from .native848_pair_audit_v2 import world_from_row

    started = time.perf_counter()
    smoke = calibration_smoke(app, rep, include_instances=True, instance_backend='fast')
    require(smoke['passed'] and smoke['native_identity_and_occluder_smoke_passed'], 'Native sensor smoke failed')
    write_json(output / 'calibration_smoke.json', smoke)
    smoke_seconds = time.perf_counter() - started
    by_id = {r['sample_id']: r for r in cache['records']}
    anchor = plans[by_id[plan['selected_sample_ids'][0]]['pair_plan_path']]
    scene = prepare_native_scene(app, anchor)
    stage, robot = scene['stage'], scene['robot']
    substitution = substitute_plant(stage, scene['original_variant'], scene['records'], scene['variants'], scene['generated'])
    reports = [*scene['reports'], substitution['report']]
    catalogue = component_catalogue(stage, substitution['records'], reports, substitution['variants'])
    require(len(catalogue) == scene['counts']['components'], 'Full component population changed')
    variant = next(v for v in substitution['variants'] if v['variant_id'] == scene['generated']['variant_id'])
    worlds = {}
    for path, pair in plans.items():
        world = target_world_geometry(stage, variant, pair['generated_row'])
        expected = world_from_row(pair, pair['generated_row'])
        require(all(np.allclose(world[k], expected[k], atol=1e-9, rtol=0) for k in world), 'Generated world geometry differs')
        worlds[path] = world
    original_hashes = source_hashes(stage)
    product = rep.create.render_product(HEAD_CAMERA, LEGACY_RESOLUTION)
    writer = make_writer(rep, include_instances=True, instance_backend='fast')
    writer.attach([product])
    geometry = StaticGeometryScreenCache(stage, robot['root'], include_generated_plants=True)
    monitor = None
    previous = None
    samples, decisions, probes, timings = [], [], [], []
    profiled_geometry = False
    try:
        for name in plan['selected_sample_ids']:
            tick = time.perf_counter()
            record = by_id[name]
            pair = plans[record['pair_plan_path']]
            row, world = pair['generated_row'], worlds[record['pair_plan_path']]
            pose, cal = apply_cached_pose(stage, robot, record)
            require(np.allclose(world['nominal_world_m'], record['target_world_m'], atol=1e-9, rtol=0), 'Cached target is stale')
            applied = time.perf_counter()
            screen = geometry()
            if not profiled_geometry:
                full = StaticBoundScreen(static_obstacles(stage, robot['root']),
                    scene_triangle_refiner(stage, include_generated_plants=True))(visible_bounds(stage, robot['root']))
                require(full == screen, 'Cached static geometry screen differs from full screen')
                write_json(output / 'geometry_cache_reference.json', dict(screen=full,
                    cached_result_equal=True, generated_plant_triangle_refinement=True))
                profiled_geometry = True
            write_json(output / ('geometry_screen_' + name + '.json'), dict(sample_id=name,
                screen=screen, generated_plant_triangle_refinement=True))
            if not screen['passed']:
                decision = dict(sample_id=name, state='rejected_native_geometry', elapsed_seconds=time.perf_counter() - tick)
                decisions.append(decision)
                write_json(output / ('decision_' + name + '.json'), decision)
                print('DIRECT848_GEOMETRY_HOLD', name, flush=True)
                continue
            pose['visual_bound_screen'] = screen
            screened = time.perf_counter()
            if monitor is None:
                monitor = StaticSceneMonitor(stage, robot['root'])
            before = monitor.begin()
            rendering = time.perf_counter()
            if not samples:
                payload, settling = reference_payload(rep, writer, cal)
                subframes = 56
            else:
                payload, settling = consolidated_payload(rep, writer, cal, subframes=8)
                subframes = 8
            snapshot_seconds = time.perf_counter() - rendering
            native_sequence = writer.sequence
            probe_extra_seconds = 0.0
            if name in plan['long_reference_probe_ids'] and subframes == 8:
                probe_started = time.perf_counter()
                short = payload
                reference, reference_settings = reference_payload(rep, writer, cal)
                comparison = convergence(short, reference, cal,
                    roi_pixel=record['requested_spec']['desired_pixel_xy'], limits=LONG_RGB_LIMITS)
                probe_dir = output / 'render_probes'
                probe_dir.mkdir(exist_ok=True)
                short_path = probe_dir / (name + '_8.png')
                long_path = probe_dir / (name + '_reference.png')
                Image.fromarray(np.asarray(short['rgb'])[:, :, :3]).save(short_path)
                Image.fromarray(np.asarray(reference['rgb'])[:, :, :3]).save(long_path)
                probe = dict(sample_id=name, comparison=comparison, short_callback_sequence=native_sequence,
                    reference_callback_sequence=writer.sequence, short_rgb_sha256=sha256(short_path),
                    reference_rgb_sha256=sha256(long_path), reference_settings=reference_settings,
                    saved_training_candidate='long_reference', additional_render_subframes=56)
                probes.append(probe)
                write_json(probe_dir / (name + '.json'), probe)
                require(comparison['geometry_and_identity_equal'], 'Short/reference geometry or native identity differs')
                payload = reference
                subframes += 56
                probe_extra_seconds = time.perf_counter() - probe_started
            rgb, depth, valid, reference_time, freshness = validate_native_static(
                payload, cal, before, monitor.token(), writer.sequence, previous)
            previous = freshness
            assert_same_camera(calibration_for_native_resolution(calibration(stage), LEGACY_RESOLUTION), cal)
            require(scene['settings'].get('/rtx/rendermode') == scene['old_manifest']['renderer'], 'Renderer changed')
            instances, mapping = decode_native_instances(payload, LEGACY_RESOLUTION)
            components, organs, owners = component_masks(instances, mapping, catalogue)
            target = next(c for c in catalogue if c['variant_id'] == row['variant_id'] and c['component_id'] == row['component_id'])
            radius = row['cut_region_proposal']['nominal']['petiole_radius_m']
            nominal = project([world['nominal_world_m']], cal)[0]
            interval = project(world['interval_world_m'], cal)
            visibility, mask = interval_visibility(nominal, interval, depth, valid, instances, mapping, owners, target, radius)
            quality = view_quality(cal, nominal, interval, radius, visibility, rgb, mask)
            old_root = scene['original_variant']['plant_root']
            old_ids = [i for i, path in mapping.items() if path == old_root or path.startswith(old_root + '/')]
            require(not np.isin(instances, old_ids).any(), 'Original foreground remains in generated frame')
            metadata = dict(schema_version=api.SAMPLE_SCHEMA, sample_id=name,
                state='native_static_diagnostic_pending_review', training_sample_approved=False,
                sensor=sensor_profile(LEGACY_RESOLUTION), calibration=cal, robot_snapshot=pose,
                scene_counts=scene['counts'], lighting=scene['old_manifest']['lighting'],
                renderer=scene['old_manifest']['renderer'], geometry_screen=screen,
                quality=quality, generated_plant_triangle_refinement=True,
                rendered_camera_params=jsonable(payload['camera_params']),
                input_policy=dict(clean_full_scene=True, isolation=False, diagnostic_overlays=False),
                native_target_pixels=int(mask.sum()), old_plant_native_pixels=0,
                native_instance_backend='fast', historical_labels_inherited=False,
                native_instance_sha256=hashlib.sha256(instances.tobytes()).hexdigest(),
                native_mapping_sha256=fingerprint(mapping),
                direct_camera=dict(pose_cache_sha256=plan['cache_sha256'], cached_sample_id=name,
                    pair_plan_sha256=record['pair_plan_sha256'], runtime_pose_search=False,
                    runtime_head_optimization=False, stage_reused=True, render_product_reused=True),
                synchronization=dict(method='frozen_scene_single_native_writer_payload',
                    scene_unchanged_during_capture=True, dynamic_recording_supported=False,
                    reference_time=reference_time, freshness=freshness, static_guard=before,
                    native_render_frame=jsonable(payload['pilot_render_frame']), engine_frame_id_verified=False,
                    render_budget_subframes=subframes, profile=api.PROFILE,
                    settling=settling, photometric_identity_to_long_render_claimed=False),
                supervision=dict(**world, target_id=row['target_id'], source_target_id=pair['source_row']['target_id'],
                    split_group=pair['split_group'], conservative_view_cap_group=pair['conservative_view_cap_group'],
                    cut_region_proposal=row['cut_region_proposal'], nominal_projected=nominal,
                    projected_interval=interval, depth_evidence=depth_evidence(nominal, depth, valid, radius),
                    visibility_evidence=visibility, cut_safety_validated=False))
            folder = output / name
            write_sample(folder, rgb, depth, valid, metadata)
            write_visibility(folder, instances, mapping, catalogue, components, organs, mask)
            samples.append(read_json(folder / 'sample.json'))
            timing = dict(sample_id=name, pose_apply_and_floor_s=applied-tick,
                geometry_screen_s=screened-applied, snapshot_render_s=snapshot_seconds,
                render_probe_extra_s=probe_extra_seconds,
                validation_and_artifact_write_s=time.perf_counter()-rendering-snapshot_seconds-probe_extra_seconds,
                total_frame_s=time.perf_counter()-tick, render_budget_subframes=subframes)
            timings.append(timing)
            decision = dict(sample_id=name, state='captured_pending_audit_and_quality', timing=timing)
            decisions.append(decision)
            write_json(output / ('decision_' + name + '.json'), decision)
            print('DIRECT848_CAPTURED', name, 'seconds', round(timing['total_frame_s'], 3), 'quality', quality, flush=True)
    finally:
        if monitor is not None:
            monitor.close()
        geometry.close()
        writer.detach()
        product.destroy()
    require(source_hashes(stage) == original_hashes, 'Loaded source USD changed during batch')
    verify_bindings(cache['source_bindings'])
    require(not any(not layer.anonymous and layer.dirty for layer in stage.GetUsedLayers()), 'Source layer dirtied')
    return dict(state=api.RESULT_STATE, resolution=[848, 408], profile=api.PROFILE,
        samples=samples, decisions=decisions, phase_timings=timings, render_probes=probes,
        short_profile_comparisons_passed=len(probes) == 3 and all(p['comparison']['passed'] for p in probes),
        calibration_smoke_sha256=sha256(output / 'calibration_smoke.json'),
        full_greenhouse_stage_count=1, capture_render_product_count=1,
        calibration_smoke_uses_separate_diagnostic_stage=True, pose_search_attempts=0,
        runtime_head_optimizations=0, geometry_cache=geometry.diagnostics(),
        calibration_smoke_seconds=smoke_seconds, scene_setup_seconds=scene['setup_seconds'],
        elapsed_seconds=time.perf_counter()-started, source_assets_unchanged=True,
        source_cap_reset=False, geometry_novelty_qualified=False, training_approved=False,
        accepted_training_increment=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(sha256(args.plan) == args.plan_sha256, 'Changed direct plan')
    plan = read_json(args.plan)
    cache, plans, _ = api.check(plan)
    output = args.output.resolve()
    protected = [Path(cache['variant_directory']).resolve(), Path(plan['cache_path']).parent.resolve()]
    protected += [Path(p['source_capture']).resolve() for p in plans.values()]
    require(not output.exists() and all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected),
        'New disjoint direct capture output required')
    from sim_physics.host_memory import preflight
    memory = preflight()
    disk = shutil.disk_usage(output.parent).free
    require(memory['allowed'] and memory['commit_headroom_bytes'] >= 20*2**30 and disk >= 60*2**30, 'Native reserves unavailable')
    process = windows_worker_admission(None)
    output.mkdir(parents=True)
    write_json(output / 'request.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        plan_path=str(args.plan.resolve()), plan_sha256=args.plan_sha256,
        host_memory_preflight=memory, available_disk_bytes=disk, process_admission=process,
        training_approved=False, automatic_retries=False))
    app = None
    succeeded = False
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, width=848, height=408, multi_gpu=False,
            renderer='RaytracedLighting', sync_loads=False, disable_viewport_updates=True,
            extra_args=['--/app/settings/persistent=false']))
        api.check(plan, full=True)
        result = capture(app, output, plan, cache, plans)
        require(sha256(args.plan) == args.plan_sha256, 'Direct plan changed during capture')
        verify_bindings(plan['implementation_bindings'])
        result['plan_sha256'] = args.plan_sha256
        write_json(output / 'result.json', result)
        succeeded = True
    except BaseException:
        write_json(output / 'failure.json', dict(state='native848_direct_batch_failed',
            error=traceback.format_exc(), training_approved=False))
        raise
    finally:
        if app is not None:
            app.close(exit_code=0 if succeeded else 1)


if __name__ == '__main__':
    main()
