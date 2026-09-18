"""Explicit bounded native producer; importing this module never imports USD/Kit.

One full stage/product, original then session-only generated substitution.
Raw native writer only; no companion process, observer, approvals or retries.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import time
import traceback

from . import execution_v1 as ex
from ..native_original_capture import contracts as oc


def collect(app, output, plan):
    ex.require_running_app(app)
    import numpy as np
    import omni.replicator.core as rep
    from .scene import prepare_scene, substitute_generated
    from .audit_v1 import pair_evidence, world_from_row
    from ..capture_scene import calibration, target_world_geometry
    from ..capture_sensor import calibration_for_native_resolution, sensor_profile
    from ..capture_contract import jsonable, project, depth_evidence, fingerprint, write_sample
    from ..capture_pilot import make_writer, step_payload, source_hashes
    from ..capture_viewpoints import component_catalogue, static_obstacles, scene_triangle_refiner, visible_bounds
    from ..capture_visibility import component_masks, interval_visibility, view_quality, write_visibility
    from ..training_screen import StaticBoundScreen
    from ..static_guard import StaticSceneMonitor
    from ..native_sensor_payload import validate_native_static, decode_native_instances
    from ..native_greenhouse_pair import assert_same_camera
    from ..native_clear_labels import derive
    from ..automated_native_review import trace_review
    from ..native_dataset.inventory import _trace_consistency, STRICT, HOLD, EXCLUDE

    started = time.perf_counter()
    context = prepare_scene(app,plan)  # Full bank/geometry replay occurs AFTER app.
    prepare_scene_seconds = time.perf_counter()-started
    stage,robot = context['stage'],context['robot']
    records,variants,reports = context['records'],context['variants'],context['reports']
    product = rep.create.render_product(oc.HEAD_CAMERA,oc.RESOLUTION)
    writer,attached = None,False
    captured,samples,timings = [],[],[]
    try:
        writer = make_writer(rep,include_instances=True,instance_backend='legacy')
        writer.attach([product]); attached=True
        for mode in ex.MODES:
            frame_started = time.perf_counter(); timing={}
            row = plan['source_row' if mode == 'original_control' else 'generated_row']
            if mode == 'generated_variant':
                ex.require_running_app(app)
                changed = substitute_generated(context,plan)
                replacement = changed['replacement']
                records,variants,reports = replacement['records'],replacement['variants'],changed['reports']
            cal = calibration_for_native_resolution(calibration(stage),oc.RESOLUTION)
            assert_same_camera(cal,plan['expected_calibration'])
            catalogue = component_catalogue(stage,records,reports,variants)
            oc.require(len(catalogue) == context['counts']['components'], 'Full component population changed')
            variant = next(v for v in variants if v['variant_id'] == row['variant_id'])
            world = target_world_geometry(stage,variant,row)
            expected_world = world_from_row(plan,row)
            for key,value in expected_world.items():
                oc.require(np.allclose(world[key],value,atol=1e-9,rtol=0), 'Actual target world geometry changed')
            screen = StaticBoundScreen(static_obstacles(stage,robot['root']),
                scene_triangle_refiner(stage,include_generated_plants=True))(visible_bounds(stage,robot['root']))
            nominal,interval = project([world['nominal_world_m']],cal)[0],project(world['interval_world_m'],cal)
            reasons = ['static_geometry_screen_failed'] if not screen['passed'] else []
            if any(p['projection_status'] != 'in_frame' for p in [nominal,*interval]):
                reasons.append('interval_clipped_or_out_of_frame')
            timing['actual_geometry_camera_framing_s'] = time.perf_counter()-frame_started
            if reasons:
                captured.append(dict(candidate_id=mode,target_id=row['target_id'],state='held_pre_render',
                                     screen=screen,reasons=reasons))
                timings.append(dict(mode=mode,seconds=timing)); continue
            # Future write-only observer seam: after actual pose/geometry/framing,
            # before warmup. No observer is installed and no predictions gate views.
            t=time.perf_counter()
            phase_hashes = source_hashes(stage)
            oc.bind_all(phase_hashes)
            oc.require(all(plan['source_bindings'].get(p) == h for p,h in phase_hashes.items()), 'Unbound loaded USD')
            timing['loaded_source_hash_and_binding_s']=time.perf_counter()-t
            t=time.perf_counter();payload_seconds=[]
            for _ in range(6):
                payload_started=time.perf_counter()
                step_payload(rep,writer,subframes=8)
                payload_seconds.append(time.perf_counter()-payload_started)
            timing['six_warmup_payloads_s']=time.perf_counter()-t
            timing['warmup_payload_seconds']=payload_seconds
            t=time.perf_counter()
            monitor=StaticSceneMonitor(stage,robot['root'])
            timing['static_monitor_setup_s']=time.perf_counter()-t
            try:
                before=monitor.begin(); t=time.perf_counter()
                payload=step_payload(rep,writer,subframes=8)
                timing['final_payload_s']=time.perf_counter()-t
                t=time.perf_counter()
                # Same camera across substitution: distinct-camera previous= is
                # intentionally not used. Pair replay checks sequence/static change.
                rgb,depth,valid,reference,token=validate_native_static(payload,cal,before,monitor.token(),writer.sequence)
                assert_same_camera(calibration_for_native_resolution(calibration(stage),oc.RESOLUTION),cal)
                oc.require(context['settings'].get('/rtx/rendermode') == context['scene_evidence']['renderer'], 'Renderer changed')
                ids,mapping=decode_native_instances(payload,oc.RESOLUTION)
                components,organs,owners=component_masks(ids,mapping,catalogue)
                target=next(c for c in catalogue if c['variant_id']==row['variant_id'] and c['component_id']==row['component_id'])
                radius=row['cut_region_proposal']['nominal']['petiole_radius_m']
                visibility,mask=interval_visibility(nominal,interval,depth,valid,ids,mapping,owners,target,radius)
                quality=view_quality(cal,nominal,interval,radius,visibility,rgb,mask)
                old_root=context['original_variant']['plant_root']
                old_ids=[i for i,path in mapping.items() if path==old_root or path.startswith(old_root+'/')]
                evidence=deepcopy(context['scene_evidence'])
                meta=dict(schema_version=ex.SAMPLE,state='fresh_native_reference_pair_pending_replay',
                    sample_id=mode,training_sample_approved=False,historical_labels_inherited=False,
                    sensor=sensor_profile(oc.RESOLUTION),calibration=cal,robot_snapshot=deepcopy(context['pose']),
                    scene_counts=evidence['scene_counts'],lighting=evidence['lighting'],renderer=evidence['renderer'],
                    scene_evidence=evidence,geometry_screen=screen,quality=quality,native_instance_backend='legacy',
                    native_instance_sha256=oc.digest(ids.tobytes()),native_mapping_sha256=fingerprint(mapping),
                    native_target_pixels=int(mask.sum()),old_plant_native_pixels=int(np.isin(ids,old_ids).sum()),
                    input_policy=deepcopy(plan['input_policy']),rendered_camera_params=jsonable(payload['camera_params']),
                    synchronization=dict(method='frozen_scene_single_native_writer_payload',
                        scene_unchanged_during_capture=True,dynamic_recording_supported=False,reference_time=reference,
                        freshness=token,static_guard=before,native_render_frame=jsonable(payload['pilot_render_frame']),
                        engine_frame_id_verified=False,render_budget_subframes=56),
                    supervision=dict(**world,target_id=row['target_id'],source_target_id=plan['source_row']['target_id'],
                        split_group=plan['source_family'],conservative_view_cap_group=plan['conservative_view_cap_group'],
                        cut_region_proposal=deepcopy(row['cut_region_proposal']),nominal_projected=nominal,
                        projected_interval=interval,depth_evidence=depth_evidence(nominal,depth,valid,radius),
                        visibility_evidence=visibility,cut_safety_validated=False))
                report=next(r for r in reports if r['plant_id']==row['variant_id'])
                timing['native_validation_masks_metadata_s']=time.perf_counter()-t
                t=time.perf_counter()
                label=derive(meta,report,rgb,depth,valid,components,catalogue)
                trace=trace_review(meta,report,label,rgb,depth,valid,components,catalogue) if label['eligible'] else None
                _trace_consistency(label,trace)
                decision=EXCLUDE if not label['eligible'] else STRICT if trace['passed'] else HOLD
                timing['label_and_trace_s']=time.perf_counter()-t
                t=time.perf_counter()
                folder=output/mode
                write_sample(folder,rgb,depth,valid,meta)
                write_visibility(folder,ids,mapping,catalogue,components,organs,mask)
                oc.write_new(folder/'supervision/label.json',label)
                if trace is not None: oc.write_new(folder/'supervision/query_trace.json',trace)
                samples.append(oc.read_json(folder/'sample.json'))
                captured.append(dict(candidate_id=mode,target_id=row['target_id'],state='native_captured',
                    screen=screen,decision=decision,sample_sha256=oc.sha256(folder/'sample.json'),
                    label_sha256=oc.sha256(folder/'supervision/label.json'),query_trace=trace))
                timing['raw_storage_s']=time.perf_counter()-t
                t=time.perf_counter()
                oc.require(source_hashes(stage)==phase_hashes, 'Loaded USD changed during native capture')
                timing['postcapture_loaded_source_hash_s']=time.perf_counter()-t
                timings.append(dict(mode=mode,seconds=timing,total_s=time.perf_counter()-frame_started))
            finally:
                monitor.close()
        oc.require(not any(not layer.anonymous and layer.dirty for layer in stage.GetUsedLayers()), 'Source layer dirtied')
        t=time.perf_counter()
        oc.bind_all(plan['source_bindings']);oc.bind_all(plan['implementation_bindings']);ex.implementation_bindings()
        final_binding_seconds=time.perf_counter()-t
        return dict(records=captured,pair_evidence=pair_evidence(samples),stage_count=1,render_product_count=1,
            source_assets_unchanged=True,phase_timings=timings,setup_seconds=context['setup_seconds'],
            prepare_scene_total_seconds=prepare_scene_seconds,
            scene_setup_scope='frozen_full_proof_replay_plus_USD_loading_population_pose_not_individually_split',
            final_source_binding_seconds=final_binding_seconds,
            elapsed_seconds=time.perf_counter()-started,**ex.FLAGS)
    finally:
        try:
            if attached: writer.detach()
        finally:
            product.destroy()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--request',required=True);p.add_argument('--request-sha256',required=True)
    args=p.parse_args(argv)
    worker_started=time.perf_counter()
    request,plan=ex.preflight_request(args.request,args.request_sha256)
    initial_hash_seconds=time.perf_counter()-worker_started
    from sim_physics.host_memory import preflight
    from ..native_generated_pair import windows_worker_admission
    memory=preflight()
    oc.require(memory['checked'] is True and memory['allowed'] is True
        and memory['commit_headroom_bytes']>=20*2**30,'Native memory reserve unavailable')
    admission=windows_worker_admission(None)  # Exact singleton Kit, never CPU-owner exemption.
    output=Path(request['output'])/'capture'
    oc.require(output.parent.is_dir() and not output.exists(),'Owned create-only capture directory required')
    started=oc.read_json(output.parent/'owner_started.json')
    owned=oc.read_json(output.parent/'owned_worker.json')
    oc.require(started['execution_request_path']==str(Path(args.request).resolve())
        and started['execution_request_sha256']==args.request_sha256
        and owned['owner_pid']==started['owner_pid']
        and owned['command']==ex.worker_command(args.request,args.request_sha256,request), 'Missing exact owned-launch declaration')
    output.mkdir()
    oc.write_new(output/'request.json',dict(schema=ex.RESULT,execution_request_path=str(Path(args.request).resolve()),
        execution_request_sha256=args.request_sha256,plan_sha256=request['plan_sha256'],
        owner_started_sha256=oc.sha256(output.parent/'owner_started.json'),
        owned_worker_sha256=oc.sha256(output.parent/'owned_worker.json'),
        host_memory_preflight=memory,process_admission=admission,training_started=False))
    app=None;succeeded=False
    try:
        app_started=time.perf_counter()
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,width=1696,height=816,multi_gpu=False,
            renderer='RaytracedLighting',sync_loads=False,disable_viewport_updates=True,
            extra_args=['--/app/settings/persistent=false']))
        ex.require_running_app(app)
        app_startup_seconds=time.perf_counter()-app_started
        result=collect(app,output,plan)
        t=time.perf_counter()
        ex.preflight_request(args.request,args.request_sha256)
        final_preflight_seconds=time.perf_counter()-t
        result.update(schema=ex.RESULT,state='bounded_pair_capture_complete_pending_owned_exit',
            request_sha256=oc.sha256(output/'request.json'),plan_sha256=request['plan_sha256'],
            implementation_bindings=ex.implementation_bindings(),native_exit_status_must_be_checked_by_owner=True,
            worker_timings=dict(initial_hash_preflight_s=initial_hash_seconds,SimulationApp_startup_s=app_startup_seconds,
                final_hash_preflight_s=final_preflight_seconds,total_before_close_s=time.perf_counter()-worker_started))
        oc.write_new(output/'result.json',result);succeeded=True
    except BaseException:
        oc.write_new(output/'failure.json',dict(state='bounded_reference_worker_failed',traceback=traceback.format_exc(),**ex.FLAGS))
        raise
    finally:
        if app is not None:
            try:
                app.close(exit_code=0 if succeeded else 1)
            except BaseException:
                if not (output/'failure.json').exists():
                    oc.write_new(output/'failure.json',dict(state='bounded_reference_cleanup_failed',traceback=traceback.format_exc(),**ex.FLAGS))
                raise


if __name__ == '__main__':
    main()
