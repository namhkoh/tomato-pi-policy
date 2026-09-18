"""One full native greenhouse/product, bounded views of one generated plant.

Only main creates Kit. Fixed mounted optics, native RGB/ID/optical-Z, 56 requested
subframes for every capture, lossless qualified compact persistence and original
annotation gates. No source layer edits, motion approval, retries or companion.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import time
import traceback

from . import batch_execution_v2 as ex
from ..native_original_capture import contracts as oc


def collect(app, output, plan, request):
    ex.require_running_app(app)
    import numpy as np
    import omni.replicator.core as rep
    from .batch_scene_v2 import prepare_scene, substitute_generated
    from .batch_pose_v2 import screened_reference, reference_evidence, verify_pose
    from .audit_v1 import world_from_row
    from ..capture_scene import calibration, target_world_geometry, mounted_camera_to_head
    from ..capture_sensor import calibration_for_native_resolution, sensor_profile
    from ..capture_contract import jsonable, project, depth_evidence, fingerprint
    from ..capture_pilot import make_writer, step_payload, source_hashes
    from ..capture_visibility import component_masks, interval_visibility, view_quality
    from ..static_geometry_cache import StaticGeometryScreenCache
    from ..static_guard import StaticSceneMonitor
    from ..native_sensor_payload import validate_native_static, decode_native_instances
    from ..native_greenhouse_pair import assert_same_camera
    from ..native_view_pose import set_reference_snapshot
    from ..native_clear_labels import derive
    from ..automated_native_review import trace_review
    from ..native_dataset.inventory import _trace_consistency, STRICT, HOLD, EXCLUDE
    from ..native_dataset.capture_storage import write_compact_native_sample
    from ..native_dataset import compact_qualification

    started = time.perf_counter()
    proof = ex.checked_storage(request)
    context = prepare_scene(app, plan)
    changed = substitute_generated(context, plan)
    stage, robot = context['stage'], context['robot']
    replacement, catalogue = changed['replacement'], changed['catalogue']
    stage_hashes = source_hashes(stage)
    oc.bind_all(stage_hashes)
    oc.require(all(plan['source_bindings'].get(p) == h for p, h in stage_hashes.items()), 'Unbound loaded USD')
    mount = mounted_camera_to_head(stage).copy()
    product = rep.create.render_product(oc.HEAD_CAMERA, oc.RESOLUTION)
    writer, attached, cache, monitor = None, False, None, None
    records, timings, previous = [], [], None
    try:
        writer = make_writer(rep, include_instances=True, instance_backend='legacy')
        writer.attach([product]); attached = True
        cache = StaticGeometryScreenCache(stage, robot['root'], include_generated_plants=True)
        setup_seconds = time.perf_counter()-started
        for case, spec in ex.jobs(plan):
            frame_started = time.perf_counter(); timing = {}
            row, name = case['generated_row'], spec['candidate_id']
            record = dict(candidate_id=name, target_id=row['target_id'], requested_spec=deepcopy(spec), **ex.FLAGS)
            variant = next(v for v in replacement['variants'] if v['variant_id'] == row['variant_id'])
            world = target_world_geometry(stage, variant, row)
            for key, value in world_from_row(case, row).items():
                oc.require(np.allclose(world[key], value, atol=1e-9, rtol=0), 'Actual target world geometry changed')
            t = time.perf_counter()
            try:
                pose = set_reference_snapshot(stage, robot, screened_reference(case), spec, world['nominal_world_m'])
            except ValueError as exc:
                record.update(state='rejected_pose', reason=str(exc))
                oc.write_new(output/(name+'_decision.json'), record); records.append(record)
                print('ORIGINAL_REFERENCE_BATCH_REJECTED', name, str(exc), flush=True)
                continue
            oc.require(np.allclose(mounted_camera_to_head(stage), mount, atol=1e-9, rtol=0), 'Camera mount moved')
            cal = calibration_for_native_resolution(calibration(stage), oc.RESOLUTION)
            screen = cache()
            record.update(screen=screen, robot_snapshot=deepcopy(pose), calibration=deepcopy(cal))
            timing['mounted_pose_geometry_s'] = time.perf_counter()-t
            if not screen['passed']:
                record.update(state='rejected_possible_geometry_overlap')
                oc.write_new(output/(name+'_decision.json'), record); records.append(record)
                print('ORIGINAL_REFERENCE_BATCH_REJECTED', name, 'geometry', flush=True)
                continue
            nominal, interval = project([world['nominal_world_m']], cal)[0], project(world['interval_world_m'], cal)
            oc.require(nominal['projection_status'] == 'in_frame'
                and np.linalg.norm(np.asarray(nominal['pixel_xy'])-2*np.asarray(spec['desired_pixel_xy'])) < 4,
                'Actual mounted camera differs from planned framing')
            t = time.perf_counter(); before_request = writer.request_index
            for _ in range(6):
                step_payload(rep, writer, subframes=8)
            if monitor is None:
                monitor = StaticSceneMonitor(stage, robot['root'])
            before = monitor.begin()
            payload = step_payload(rep, writer, subframes=8)
            requests = writer.request_index-before_request
            oc.require(requests == 7, 'Every view requires seven 8-subframe requests')
            timing['native_56_subframes_s'] = time.perf_counter()-t
            rgb, depth, valid, reference, token = validate_native_static(
                payload, cal, before, monitor.token(), writer.sequence, previous=previous)
            previous = token
            assert_same_camera(calibration_for_native_resolution(calibration(stage), oc.RESOLUTION), cal)
            oc.require(context['settings'].get('/rtx/rendermode') == context['scene_evidence']['renderer'], 'Renderer changed')
            t = time.perf_counter()
            ids, mapping = decode_native_instances(payload, oc.RESOLUTION)
            components, organs, owners = component_masks(ids, mapping, catalogue)
            target = next(c for c in catalogue if c['variant_id'] == row['variant_id'] and c['component_id'] == row['component_id'])
            radius = row['cut_region_proposal']['nominal']['petiole_radius_m']
            visibility, mask = interval_visibility(nominal, interval, depth, valid, ids, mapping, owners, target, radius)
            quality = view_quality(cal, nominal, interval, radius, visibility, rgb, mask)
            old_root = context['original_variant']['plant_root']
            old_ids = [i for i, path in mapping.items() if path == old_root or path.startswith(old_root+'/')]
            old_pixels = int(np.isin(ids, old_ids).sum())
            oc.require(old_pixels == 0, 'Replaced foreground remains in the native image')
            evidence = deepcopy(context['scene_evidence'])
            meta = dict(schema_version=ex.SAMPLE, state='fresh_native_reference_batch_pending_replay',
                sample_id=name, training_sample_approved=False, historical_labels_inherited=False,
                sensor=sensor_profile(oc.RESOLUTION), calibration=cal, robot_snapshot=deepcopy(pose),
                pose_reference_evidence=reference_evidence(case), requested_spec=deepcopy(spec),
                scene_counts=evidence['scene_counts'], lighting=evidence['lighting'], renderer=evidence['renderer'],
                scene_evidence=evidence, geometry_screen=screen, quality=quality, native_instance_backend='legacy',
                native_instance_sha256=oc.digest(ids.tobytes()), native_mapping_sha256=fingerprint(mapping),
                native_target_pixels=int(mask.sum()), old_plant_native_pixels=old_pixels,
                generated_plant_triangle_refinement=True, input_policy=deepcopy(plan['input_policy']),
                rendered_camera_params=jsonable(payload['camera_params']),
                synchronization=dict(method='frozen_scene_single_native_writer_payload',
                    scene_unchanged_during_capture=True, dynamic_recording_supported=False, reference_time=reference,
                    freshness=token, static_guard=before, native_render_frame=jsonable(payload['pilot_render_frame']),
                    engine_frame_id_verified=False, render_budget_subframes=56, actual_orchestrator_requests=requests),
                supervision=dict(**world, target_id=row['target_id'], source_target_id=case['source_row']['target_id'],
                    split_group=plan['source_family'], conservative_view_cap_group=case['conservative_view_cap_group'],
                    cut_region_proposal=deepcopy(row['cut_region_proposal']), nominal_projected=nominal,
                    projected_interval=interval, depth_evidence=depth_evidence(nominal, depth, valid, radius),
                    visibility_evidence=visibility, cut_safety_validated=False))
            verify_pose(meta, case, spec, world['nominal_world_m'])
            report = context['generated']['report']
            label = derive(meta, report, rgb, depth, valid, components, catalogue)
            trace = trace_review(meta, report, label, rgb, depth, valid, components, catalogue) if label['eligible'] else None
            _trace_consistency(label, trace)
            decision = EXCLUDE if not label['eligible'] else STRICT if trace['passed'] else HOLD
            timing['native_masks_pose_label_trace_s'] = time.perf_counter()-t
            t = time.perf_counter()
            stored = write_compact_native_sample(output/name, rgb, depth, valid, meta, ids,
                mapping, catalogue, components, organs, mask, label, trace)
            record.update(state='native_captured', decision=decision, **stored)
            oc.write_new(output/(name+'_decision.json'), record); records.append(record)
            timing['compact_storage_s'] = time.perf_counter()-t
            oc.require(source_hashes(stage) == stage_hashes, 'Loaded source USD changed during capture')
            timings.append(dict(candidate_id=name, seconds=timing, total_s=time.perf_counter()-frame_started))
            print('ORIGINAL_REFERENCE_BATCH_CAPTURED', name, decision, flush=True)
        oc.require(not any(not l.anonymous and l.dirty for l in stage.GetUsedLayers()), 'Source layer dirtied')
        oc.bind_all(plan['source_bindings']); oc.bind_all(plan['implementation_bindings']); ex.implementation_bindings()
        compact_qualification.verify_checked_qualification(proof)
        return dict(records=records, captured_frames=sum(r['state'] == 'native_captured' for r in records),
            stage_count=1, render_product_count=1, source_assets_unchanged=True, phase_timings=timings,
            setup_seconds=setup_seconds, elapsed_seconds=time.perf_counter()-started,
            geometry_cache=cache.diagnostics(), storage_backend=request['storage_backend'],
            storage_qualification=proof, **ex.FLAGS)
    finally:
        try:
            if monitor is not None: monitor.close()
            if cache is not None: cache.close()
        finally:
            try:
                if attached: writer.detach()
            finally:
                product.destroy()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request', required=True); p.add_argument('--request-sha256', required=True)
    args = p.parse_args(argv)
    request, plan = ex.preflight_request(args.request, args.request_sha256)
    from sim_physics.host_memory import preflight
    from ..native_generated_pair import windows_worker_admission
    memory = preflight()
    oc.require(memory['checked'] is True and memory['allowed'] is True
        and memory['commit_headroom_bytes'] >= 20*2**30, 'Native memory reserve unavailable')
    admission = windows_worker_admission(None)
    output = Path(request['output'])/'capture'
    oc.require(output.parent.is_dir() and not output.exists(), 'Owned create-only capture directory required')
    started, owned = oc.read_json(output.parent/'owner_started.json'), oc.read_json(output.parent/'owned_worker.json')
    oc.require(started['execution_request_path'] == str(Path(args.request).resolve())
        and started['execution_request_sha256'] == args.request_sha256 and owned['owner_pid'] == started['owner_pid']
        and owned['command'] == ex.worker_command(args.request, args.request_sha256, request),
        'Missing exact owned-launch declaration')
    output.mkdir()
    oc.write_new(output/'request.json', dict(schema=ex.RESULT, execution_request_path=str(Path(args.request).resolve()),
        execution_request_sha256=args.request_sha256, plan_sha256=request['plan_sha256'],
        owner_started_sha256=oc.sha256(output.parent/'owner_started.json'),
        owned_worker_sha256=oc.sha256(output.parent/'owned_worker.json'),
        host_memory_preflight=memory, process_admission=admission, training_started=False))
    app, succeeded = None, False
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, width=1696, height=816, multi_gpu=False,
            renderer='RaytracedLighting', sync_loads=False, disable_viewport_updates=True,
            extra_args=['--/app/settings/persistent=false']))
        result = collect(app, output, plan, request)
        ex.preflight_request(args.request, args.request_sha256)
        result.update(schema=ex.RESULT, state='bounded_batch_capture_complete_pending_owned_exit',
            request_sha256=oc.sha256(output/'request.json'), plan_sha256=request['plan_sha256'],
            implementation_bindings=ex.implementation_bindings(), native_exit_status_must_be_checked_by_owner=True)
        oc.write_new(output/'result.json', result); succeeded = True
    except BaseException:
        oc.write_new(output/'failure.json', dict(state='bounded_batch_worker_failed', traceback=traceback.format_exc(), **ex.FLAGS))
        raise
    finally:
        if app is not None:
            try:
                app.close(exit_code=0 if succeeded else 1)
            except BaseException:
                if not (output/'failure.json').exists():
                    oc.write_new(output/'failure.json', dict(state='bounded_batch_cleanup_failed', traceback=traceback.format_exc(), **ex.FLAGS))
                raise


if __name__ == '__main__':
    main()
