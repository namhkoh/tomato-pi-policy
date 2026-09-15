"""One native greenhouse stage and render product, several actual robot views.

Static geometry-guided capture only. No physical robot API, VLM action or training
approval. Full56-subframe reference budget, native optical-Z and all gates remain.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import time
import traceback
import numpy as np
from .dataset_review import read_json,write_json,require,verify_bindings
from .depth_preview import sha256
from .native_view_plan import check
from .capture_sensor import HIRES_RESOLUTION,calibration_for_native_resolution,sensor_profile
from .capture_contract import jsonable,project,depth_evidence,write_sample
from .native_sensor_payload import validate_native_static,decode_native_instances


def collect(app,output,plan,base):
    import omni.replicator.core as rep
    import omni.timeline
    from .native_scene import prepare_native_scene
    from .generated_capture import substitute_plant
    from .capture_scene import calibration,target_world_geometry,mounted_camera_to_head
    from .native_view_pose import set_reference_snapshot
    from .capture_pilot import make_writer,step_payload,source_hashes
    from .capture_viewpoints import component_catalogue
    from .capture_visibility import component_masks,interval_visibility,view_quality,write_visibility
    from .static_geometry_cache import StaticGeometryScreenCache
    from .static_guard import StaticSceneMonitor
    from .review_camera import HEAD_CAMERA
    from .native_clear_labels import derive
    from .automated_native_review import trace_review
    from .native_greenhouse_pair import assert_same_camera

    started=time.perf_counter()
    context=prepare_native_scene(app,base)
    stage=context['stage'];generated=context['generated'];row=base['generated_row']
    substitution=substitute_plant(stage,context['original_variant'],context['records'],context['variants'],generated)
    reports=[*context['reports'],substitution['report']]
    catalogue=component_catalogue(stage,substitution['records'],reports,substitution['variants'])
    require(len(catalogue)==context['counts']['components'],'Changed complete plant population')
    variant=next(v for v in substitution['variants'] if v['variant_id']==row['variant_id'])
    target=next(c for c in catalogue if c['variant_id']==row['variant_id'] and c['component_id']==row['component_id'])
    world=target_world_geometry(stage,variant,row)
    require(np.allclose(world['nominal_world_m'],plan['expected_nominal_world_m'],atol=1e-9,rtol=0),
            'Changed generated target geometry')
    robot=dict(context['robot'],pose_degrees=base['expected_robot_snapshot']['joint_degrees'])
    mount=mounted_camera_to_head(stage).copy()
    optical=calibration_for_native_resolution(calibration(stage),HIRES_RESOLUTION)
    radius=row['cut_region_proposal']['nominal']['petiole_radius_m']
    stage_hashes=source_hashes(stage)
    omni.timeline.get_timeline_interface().pause()
    product=rep.create.render_product(HEAD_CAMERA,HIRES_RESOLUTION)
    writer=make_writer(rep,include_instances=True,instance_backend='legacy');writer.attach([product])
    cache=StaticGeometryScreenCache(stage,robot['root'],include_generated_plants=True)
    monitor=None
    previous=None
    records=[]
    setup_seconds=time.perf_counter()-started
    try:
        for spec in plan['views']:
            view_started=time.perf_counter()
            name=spec['candidate_id']
            decision=dict(candidate_id=name,training_approved=False,requested_spec=spec)
            try:
                pose=set_reference_snapshot(stage,robot,base['expected_robot_snapshot'],spec,world['nominal_world_m'])
            except ValueError as exc:
                decision.update(state='rejected_pose',reason=str(exc))
                write_json(output/(name+'_decision.json'),decision);records.append(decision)
                print('NATIVE_VIEW_REJECTED',name,decision['reason'],flush=True)
                continue
            require(np.allclose(mounted_camera_to_head(stage),mount,atol=1e-9,rtol=0),'Camera mount moved')
            cal=calibration_for_native_resolution(calibration(stage),HIRES_RESOLUTION)
            for field in ('camera_path','resolution','intrinsics','focal_length_mm','apertures_mm',
                          'aperture_offsets_mm','depth_convention','crop_resize'):
                require(cal[field]==optical[field],'Camera optics changed: '+field)
            screen_started=time.perf_counter();screen=cache();screen_seconds=time.perf_counter()-screen_started
            decision.update(screen=screen,geometry_screen_seconds=screen_seconds,robot_snapshot=pose)
            if not screen['passed']:
                decision.update(state='rejected_possible_geometry_overlap')
                write_json(output/(name+'_decision.json'),decision);records.append(decision)
                print('NATIVE_VIEW_REJECTED',name,'geometry',flush=True)
                continue
            nominal=project([world['nominal_world_m']],cal)[0]
            require(nominal['projection_status']=='in_frame','Solved target is not in frame')
            require(np.linalg.norm(np.asarray(nominal['pixel_xy'])-2*np.asarray(spec['desired_pixel_xy']))<4,
                    'Actual mounted camera differs from solved framing')
            for _ in range(6):step_payload(rep,writer,subframes=8)
            if monitor is None:monitor=StaticSceneMonitor(stage,robot['root'])
            before=monitor.begin()
            payload=step_payload(rep,writer,subframes=8)
            rgb,depth,valid,reference,freshness=validate_native_static(
                payload,cal,before,monitor.token(),writer.sequence,previous=previous)
            previous=freshness
            assert_same_camera(calibration_for_native_resolution(calibration(stage),HIRES_RESOLUTION),cal)
            require(context['settings'].get('/rtx/rendermode')==context['old_manifest']['renderer'],'Renderer changed')
            instances,mapping=decode_native_instances(payload,HIRES_RESOLUTION)
            components,organs,owners=component_masks(instances,mapping,catalogue)
            interval=project(world['interval_world_m'],cal)
            visibility,mask=interval_visibility(nominal,interval,depth,valid,instances,mapping,owners,target,radius)
            quality=view_quality(cal,nominal,interval,radius,visibility,rgb,mask)
            metadata=dict(schema_version='greenhouse.generated_native_multiview_sample.v1',
                state='native_static_multiview_candidate_pending_review',sample_id=name,
                training_sample_approved=False,sensor=sensor_profile(HIRES_RESOLUTION),calibration=cal,
                robot_snapshot=pose,scene_counts=context['counts'],lighting=context['old_manifest']['lighting'],
                renderer=context['old_manifest']['renderer'],geometry_screen=screen,quality=quality,
                generated_plant_triangle_refinement=True,rendered_camera_params=jsonable(payload['camera_params']),
                input_policy=dict(clean_full_scene=True,isolation=False,diagnostic_overlays=False),
                native_target_pixels=int(mask.sum()),
                synchronization=dict(method='frozen_scene_single_native_writer_payload',
                    scene_unchanged_during_capture=True,dynamic_recording_supported=False,
                    reference_time=reference,freshness=freshness,static_guard=before,
                    native_render_frame=jsonable(payload['pilot_render_frame']),
                    engine_frame_id_verified=False,render_budget_subframes=56),
                supervision=dict(**world,target_id=row['target_id'],source_target_id=base['source_row']['target_id'],
                    split_group=base['split_group'],conservative_view_cap_group=base['conservative_view_cap_group'],
                    cut_region_proposal=row['cut_region_proposal'],nominal_projected=nominal,
                    projected_interval=interval,depth_evidence=depth_evidence(nominal,depth,valid,radius),
                    visibility_evidence=visibility,cut_safety_validated=False))
            label=derive(metadata,generated['report'],rgb,depth,valid,components,catalogue)
            trace=trace_review(metadata,generated['report'],label,rgb,depth,valid,components,catalogue) if label['eligible'] else None
            folder=output/name
            write_sample(folder,rgb,depth,valid,metadata)
            write_visibility(folder,instances,mapping,catalogue,components,organs,mask)
            write_json(folder/'supervision/label.json',label)
            if trace is not None:write_json(folder/'supervision/query_trace.json',trace)
            require(source_hashes(stage)==stage_hashes,'Loaded source geometry changed during capture')
            verify_bindings(plan['source_bindings'])
            decision.update(state='native_captured_pending_review',eligible_annotation=label['eligible'],
                automatic_annotation_eligible=bool(trace is not None and trace['passed']),
                label_reason=label['reason'],label_sha256=sha256(folder/'supervision/label.json'),
                sample_sha256=sha256(folder/'sample.json'),query_trace=trace,
                elapsed_seconds=time.perf_counter()-view_started)
            write_json(output/(name+'_decision.json'),decision);records.append(decision)
            print('NATIVE_GENERATED_VIEW',name,label['eligible'],label['reason'],flush=True)
        verify_bindings(plan['implementation_bindings']);verify_bindings(base['source_bindings'])
        require(not any(not l.anonymous and l.dirty for l in stage.GetUsedLayers()),'Source layer dirtied')
        return dict(state='native_generated_multiview_pilot_complete_pending_review',
            records=records,captured_frames=sum(r['state']=='native_captured_pending_review' for r in records),
            eligible_annotation_candidates=sum(r.get('eligible_annotation',False) for r in records),
            automatically_clear_annotation_candidates=sum(r.get('automatic_annotation_eligible',False) for r in records),
            setup_seconds=setup_seconds,elapsed_seconds=time.perf_counter()-started,
            geometry_cache=cache.diagnostics(),scene_counts=context['counts'],
            source_assets_unchanged=True,training_approved=False,model_processor_executed=False,
            source_cap_reset=False,render_budget_subframes_per_view=56,
            robot_motion_or_dynamics_validated=False)
    finally:
        if monitor is not None:monitor.close()
        cache.close();writer.detach();product.destroy()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();plan=read_json(a.plan);base=check(plan);output=a.output.resolve()
    require(not output.exists(),'New multi-view output only')
    source_roots=[Path(base[k]).resolve() for k in ('source_capture','variant_directory','prerequisite_directory')]
    source_roots.append(Path(read_json(base['source_collection_plan'])['package']).resolve())
    require(all(not output.is_relative_to(s) and not s.is_relative_to(output) for s in source_roots),
            'Output must be disjoint from source data')
    from sim_physics.host_memory import preflight
    from .native_generated_pair import windows_worker_admission
    memory=preflight();require(memory['allowed'],'Native memory reserve unavailable; no override')
    admission=windows_worker_admission(None)
    output.mkdir(parents=True)
    write_json(output/'request.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),
        plan_path=str(a.plan.resolve()),plan_sha256=sha256(a.plan),host_memory_preflight=memory,
        process_admission=admission,training_started=False,automatic_retries=False))
    app=None
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,width=1696,height=816,multi_gpu=False,
            renderer='RaytracedLighting',sync_loads=False,disable_viewport_updates=True,
            extra_args=['--/app/settings/persistent=false']))
        result=collect(app,output,plan,base)
        verify_bindings(plan['implementation_bindings'])
        write_json(output/'result.json',result)
    except BaseException:
        write_json(output/'failure.json',dict(state='native_multiview_failed',error=traceback.format_exc(),
                                             training_approved=False))
        raise
    finally:
        if app is not None:app.close()


if __name__=='__main__':main()
