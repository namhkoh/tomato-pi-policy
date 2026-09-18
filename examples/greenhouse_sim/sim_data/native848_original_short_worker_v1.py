"""Original-scene reset8 capture with one explicit warmup and saved raw callbacks."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import shutil
import time
import traceback
import numpy as np
from . import native848_original_short_plan_v1 as api
from . import native848_short_validation_v5 as raw
from .native848_original_direct_worker_v2 import apply_cached_pose
from .dataset_review import require, read_json, write_json, verify_bindings
from .depth_preview import sha256
from .capture_contract import fingerprint, jsonable, project, depth_evidence, write_sample
from .capture_sensor import LEGACY_RESOLUTION, calibration_for_native_resolution, sensor_profile
from .native_greenhouse_pair import assert_same_camera


def single_request(rep, writer, subframes, *, pause_timeline=None):
    """Exactly one orchestrator request; missing callbacks fail without retries."""
    if pause_timeline is None:
        import omni.timeline
        pause_timeline = omni.timeline.get_timeline_interface().pause
    pause_timeline()
    before = writer.sequence
    writer.request_index += 1
    rep.orchestrator.step(rt_subframes=subframes, pause_timeline=True, delta_time=0.0, wait_for_render=True)
    if writer.capture_error is not None:
        raise RuntimeError('Native writer rejected a callback') from writer.capture_error
    require(writer.latest is not None and writer.sequence > before, 'No fresh callback from the single requested snapshot')
    return writer.latest


def take(rep, writer, settings, folder, steps):
    """Same v5 raw receipt contract, with an explicit no-retry request boundary."""
    import omni.usd
    folder.mkdir(parents=True, exist_ok=False)
    writer.begin_capture(folder / 'all_callbacks')
    observed = {k: jsonable(settings.get(k)) for k in raw.prior.SETTINGS}
    evidence = dict(requested_steps=steps, settings=observed, requests=[], hidden_settling=False)
    started = time.perf_counter()
    try:
        evidence['effective_subframe_policy'] = [raw.prior.checked_subframe_floor(observed, n) for n in steps]
        value = omni.usd.get_context().reset_renderer_accumulation()
        evidence['reset'] = dict(api=api.RESET_API, returned_without_exception=True,
            returned_value=jsonable(value), temporal_cleanliness_proven=False)
        for i, n in enumerate(steps):
            seq, req = writer.sequence, writer.request_index
            attempt = dict(subframes=n, callback_sequence_before=seq, request_index_before=req)
            try:
                payload = single_request(rep, writer, n)
                raw.save_raw(folder / f'callback_{i:02d}', payload)
                attempt['normalized_from_callback_sequence'] = writer.latest_event['callback_sequence']
            except BaseException:
                attempt['error'] = traceback.format_exc()
                raise
            finally:
                attempt.update(native_requests=writer.request_index-req, request_index_after=writer.request_index,
                    callback_sequence_after=writer.sequence)
                evidence['requests'].append(attempt)
        require(all(r['native_requests'] == 1 for r in evidence['requests']), 'Unexpected extra native request')
        require(observed == {k: jsonable(settings.get(k)) for k in raw.prior.SETTINGS}, 'Settings changed during request')
        return payload, evidence
    except BaseException:
        evidence['error'] = traceback.format_exc()
        raise
    finally:
        evidence['callbacks'] = writer.events
        writer.folder = folder.parent / 'unrequested_callbacks'
        writer.events = []
        evidence['elapsed_seconds'] = time.perf_counter()-started
        write_json(folder / 'request_evidence.json', evidence)


def capture(app, output, plan, cache, anchor, report):
    import omni.replicator.core as rep
    from .native848_original_scene_v2 import prepare_native_scene
    from .capture_pilot import source_hashes
    from .capture_scene import calibration, target_world_geometry
    from .capture_viewpoints import component_catalogue, static_obstacles, scene_triangle_refiner, visible_bounds
    from .capture_visibility import component_masks, interval_visibility, view_quality, write_visibility
    from .static_guard import StaticSceneMonitor
    from .static_geometry_cache import StaticGeometryScreenCache
    from .training_screen import StaticBoundScreen
    from .review_camera import HEAD_CAMERA
    from .native848_pair_audit_v2 import world_from_row, verify_camera
    from .native_dataset.clear848_workspace_v2 import WorkspaceChecker
    started = time.perf_counter()
    by_id = {r['sample_id']: r for r in cache['records']}
    scene = prepare_native_scene(app, anchor)
    stage, robot = scene['stage'], scene['robot']
    catalogue = component_catalogue(stage, scene['records'], scene['reports'], scene['variants'])
    require(len(catalogue) == scene['counts']['components'], 'Original component population changed')
    original_pins = source_hashes(stage)
    product = rep.create.render_product(HEAD_CAMERA, LEGACY_RESOLUTION)
    writer = raw.recording_writer(rep, output / 'unrequested_callbacks')
    writer.attach([product])
    geometry = StaticGeometryScreenCache(stage, robot['root'], include_generated_plants=True)
    checker = WorkspaceChecker()
    monitor = None
    previous = None
    full_pins = None
    samples, decisions, timings = [], [], []
    geometry_checked = False
    try:
        for item in plan['schedule']:
            tick = time.perf_counter()
            name, source_id = item['sample_id'], item['source_pose_id']
            rec = by_id[source_id]
            row = rec['source_row']
            world = target_world_geometry(stage, scene['original_variant'], row)
            expected = world_from_row(anchor, row)
            require(all(np.allclose(world[k], expected[k], atol=1e-9, rtol=0) for k in world), 'Original world geometry changed')
            pose, cal = apply_cached_pose(stage, robot, rec)
            applied = time.perf_counter()
            screen = geometry()
            if not geometry_checked:
                full = StaticBoundScreen(static_obstacles(stage, robot['root']),
                    scene_triangle_refiner(stage, include_generated_plants=True))(visible_bounds(stage, robot['root']))
                require(full == screen, 'Cached screen differs from full geometry')
                write_json(output / 'geometry_cache_reference.json', dict(screen=full,
                    cached_result_equal=True, generated_plant_triangle_refinement=True))
                geometry_checked = True
            write_json(output / ('geometry_screen_'+name+'.json'), dict(sample_id=name, source_pose_id=source_id,
                screen=screen, generated_plant_triangle_refinement=True))
            if not screen['passed']:
                decision = dict(**item, state='rejected_native_geometry')
                decisions.append(decision)
                write_json(output / ('decision_'+name+'.json'), decision)
                require(plan['mode'] == 'production', 'Adaptation geometry control failed')
                continue
            pose['visual_bound_screen'] = screen
            screened = time.perf_counter()
            require({k: jsonable(scene['settings'].get(k)) for k in raw.prior.SETTINGS} == plan['render_settings'],
                'Actual renderer settings differ from qualification')
            if monitor is None:
                _, warmup = take(rep, writer, scene['settings'], output / 'warmup', [8]*7)
                require(warmup['settings'] == plan['render_settings'], 'Warmup renderer differs')
                verify_bindings(original_pins)
                full_pins = source_hashes(stage)
                write_json(output / 'source_phase_bindings.json', dict(original_source_bindings=original_pins,
                    full_scene_source_bindings=full_pins, original_sources_verified_after_warmup=True,
                    full_scene_baseline_phase='after_explicit56_warmup_original_scene_no_substitution',
                    warmup_request_sha256=sha256(output / 'warmup/request_evidence.json'), source_hash_validation_removed=False))
                monitor = StaticSceneMonitor(stage, robot['root'])
                write_json(output / 'static_monitor_initialization.json', dict(
                    phase='after_explicit56_warmup_before_first_short_request', first_sample_id=name,
                    baseline_token=monitor.token(), static_guard_exclusions_modified=False,
                    warmup_request_sha256=sha256(output / 'warmup/request_evidence.json')))
            before = monitor.begin()
            rendering = time.perf_counter()
            payload, request = take(rep, writer, scene['settings'], output / 'raw' / name, [api.BUDGET])
            changed = previous is not None and fingerprint(cal) != previous['camera_sha256']
            rgb, depth, valid, reference, token, ids, mapping = raw.freshness(payload, cal, before,
                monitor.token(), writer.sequence, previous, changed)
            previous = token
            assert_same_camera(calibration_for_native_resolution(calibration(stage), LEGACY_RESOLUTION), cal)
            require(request['settings'] == plan['render_settings'], 'Snapshot renderer changed')
            components, organs, owners = component_masks(ids, mapping, catalogue)
            target = next(c for c in catalogue if c['variant_id'] == row['variant_id'] and c['component_id'] == row['component_id'])
            radius = row['cut_region_proposal']['nominal']['petiole_radius_m']
            nominal, interval = project([world['nominal_world_m']], cal)[0], project(world['interval_world_m'], cal)
            visibility, mask = interval_visibility(nominal, interval, depth, valid, ids, mapping, owners, target, radius)
            require(not np.isin(ids, [i for i, p in mapping.items() if p.startswith('/World/GeneratedNativePilot/')]).any(),
                'Generated geometry entered original frame')
            meta = dict(schema_version=api.SAMPLE_SCHEMA, sample_id=name, source_pose_id=source_id,
                capture_role=item['capture_role'], profile=api.PROFILE, render_budget_subframes=api.BUDGET,
                qualification_evidence_path=plan['qualification_evidence']['path'],
                qualification_evidence_sha256=plan['qualification_evidence']['sha256'],
                state='native_static_diagnostic_pending_review', training_sample_approved=False,
                sensor=sensor_profile(LEGACY_RESOLUTION), calibration=cal, robot_snapshot=pose,
                scene_counts=scene['counts'], lighting=scene['old_manifest']['lighting'], renderer=scene['old_manifest']['renderer'],
                geometry_screen=screen, quality=view_quality(cal, nominal, interval, radius, visibility, rgb, mask),
                generated_plant_triangle_refinement=True, rendered_camera_params=jsonable(payload['camera_params']),
                render_settings=request['settings'], input_policy=dict(clean_full_scene=True, isolation=False, diagnostic_overlays=False),
                native_target_pixels=int(mask.sum()), generated_plant_native_pixels=0, original_geometry_only=True,
                native_instance_backend='fast', historical_labels_inherited=False,
                native_instance_sha256=hashlib.sha256(ids.tobytes()).hexdigest(), native_mapping_sha256=fingerprint(mapping),
                direct_camera=dict(pose_cache_sha256=plan['cache_sha256'], cached_sample_id=source_id,
                    anchor_reference_sha256=rec['anchor_reference_sha256'], runtime_pose_search=False,
                    runtime_head_optimization=False, stage_reused=True, render_product_reused=True),
                synchronization=dict(method='frozen_scene_single_native_writer_payload', scene_unchanged_during_capture=True,
                    dynamic_recording_supported=False, reference_time=reference, freshness=token, static_guard=before,
                    static_guard_after=monitor.token(), native_render_frame=jsonable(payload['pilot_render_frame']),
                    engine_frame_id_verified=False, render_budget_subframes=api.BUDGET, profile=api.PROFILE,
                    request_evidence=request, photometric_identity_to_long_render_claimed=False),
                supervision=dict(**world, target_id=row['target_id'], source_target_id=row['target_id'],
                    split_group=anchor['split_group'], conservative_view_cap_group=row['target_id'],
                    cut_region_proposal=row['cut_region_proposal'], nominal_projected=nominal, projected_interval=interval,
                    depth_evidence=depth_evidence(nominal, depth, valid, radius), visibility_evidence=visibility, cut_safety_validated=False))
            verify_camera(meta)
            folder = output / name
            write_sample(folder, rgb, depth, valid, meta)
            write_visibility(folder, ids, mapping, catalogue, components, organs, mask)
            meta = read_json(folder / 'sample.json')
            annotation = raw.prior.annotations(meta, report, rgb, depth, valid, components, catalogue, mask)
            proof = checker.check_sample(folder / 'sample.json', sha256(folder / 'sample.json'))
            write_json(folder / 'annotation.json', annotation)
            write_json(folder / 'workspace.json', proof)
            automated = bool(annotation['label']['eligible'] and annotation['query_trace'] is not None
                and annotation['query_trace']['passed'] and proof['result']['workspace_passed'])
            samples.append(meta)
            timing = dict(sample_id=name, pose_apply_and_floor_s=applied-tick, geometry_screen_s=screened-applied,
                snapshot_render_s=request['elapsed_seconds'], total_frame_s=time.perf_counter()-tick,
                render_budget_subframes=api.BUDGET)
            timings.append(timing)
            decision = dict(**item, state='captured_pending_audit_and_quality', timing=timing,
                actual_automated_criteria_passed=automated)
            decisions.append(decision)
            write_json(output / ('decision_'+name+'.json'), decision)
            print('ORIGINAL_SHORT848_CAPTURED', name, item['capture_role'], 'actual_quality', automated, flush=True)
    finally:
        if monitor is not None:
            monitor.close()
        geometry.close()
        checker.finish()
        writer.detach()
        product.destroy()
    require(full_pins is not None and source_hashes(stage) == full_pins, 'Source map changed after initialized baseline')
    verify_bindings(original_pins)
    verify_bindings(cache['source_bindings'])
    require(not any(not layer.anonymous and layer.dirty for layer in stage.GetUsedLayers()), 'Source layer dirtied')
    require(not any(output.rglob('unrequested_callbacks')), 'Unrequested callbacks occurred')
    write_json(output / 'evidence_bindings.json', {str(p.resolve()): sha256(p) for p in output.rglob('*') if p.is_file()})
    return dict(state=api.RESULT_STATE, resolution=[848, 408], mode=plan['mode'], profile=api.PROFILE,
        render_budget_subframes=api.BUDGET, samples=samples, decisions=decisions, phase_timings=timings,
        render_probes=[], short_profile_comparisons_passed=False, established56_every_frame=False,
        full_greenhouse_stage_count=1, capture_render_product_count=1, pose_search_attempts=0,
        runtime_head_optimizations=0, geometry_cache=geometry.diagnostics(), scene_setup_seconds=scene['setup_seconds'],
        elapsed_seconds=time.perf_counter()-started, source_assets_unchanged=True, source_cap_reset=False,
        geometry_novelty_qualified=False, training_approved=False, accepted_training_increment=0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    require(sha256(args.plan) == args.plan_sha256, 'Changed short plan')
    plan = read_json(args.plan)
    output = args.output.resolve()
    # No USD/pxr plan evaluation before SimulationApp owns its native ABI.
    require(plan['schema'] == api.SCHEMA and not output.exists(), 'New short capture output required')
    from sim_physics.host_memory import preflight
    from .native_generated_pair import windows_worker_admission
    memory = preflight()
    disk = shutil.disk_usage(output.parent).free
    require(memory['allowed'] and memory['commit_headroom_bytes'] >= 20*2**30 and disk >= 60*2**30, 'Native reserves unavailable')
    process = windows_worker_admission(None)
    output.mkdir(parents=True)
    write_json(output / 'request.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        plan_path=str(args.plan.resolve()), plan_sha256=args.plan_sha256, host_memory_preflight=memory,
        available_disk_bytes=disk, process_admission=process, training_approved=False, automatic_retries=False))
    app = None
    succeeded = False
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, width=848, height=408, multi_gpu=False, renderer='RaytracedLighting',
            sync_loads=False, disable_viewport_updates=True, extra_args=['--/app/settings/persistent=false']))
        cache, anchor, report = api.check(plan, full=True)
        protected = [Path(plan['cache_path']).parent.resolve(), Path(anchor['source_capture']).resolve()]
        require(all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected), 'Disjoint output required')
        result = capture(app, output, plan, cache, anchor, report)
        require(sha256(args.plan) == args.plan_sha256, 'Plan changed during capture')
        verify_bindings(plan['implementation_bindings'])
        result['plan_sha256'] = args.plan_sha256
        write_json(output / 'result.json', result)
        succeeded = True
    except BaseException:
        write_json(output / 'failure.json', dict(state='original848_short_batch_failed', error=traceback.format_exc(), training_approved=False))
        raise
    finally:
        if app is not None:
            app.close(exit_code=0 if succeeded else 1)


if __name__ == '__main__':
    main()
