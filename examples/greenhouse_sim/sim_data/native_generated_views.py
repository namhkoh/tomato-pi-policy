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
from .native_budget import REFERENCE,TRIAL,evidence as budget_evidence


def collect(app,output,plan,base,*,profile_render=False,instance_backend='legacy',render_budget=REFERENCE):
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
    from .native_multitarget_plan import capture_jobs

    started=time.perf_counter()
    require(not profile_render or render_budget==REFERENCE,'Do not combine render-budget experiments')
    budget_evidence(render_budget,0)
    context=prepare_native_scene(app,base)
    stage=context['stage'];generated=context['generated'];anchor=base
    substitution=substitute_plant(stage,context['original_variant'],context['records'],context['variants'],generated)
    reports=[*context['reports'],substitution['report']]
    catalogue=component_catalogue(stage,substitution['records'],reports,substitution['variants'])
    require(len(catalogue)==context['counts']['components'],'Changed complete plant population')
    robot=dict(context['robot'],pose_degrees=base['expected_robot_snapshot']['joint_degrees'])
    mount=mounted_camera_to_head(stage).copy()
    optical=calibration_for_native_resolution(calibration(stage),HIRES_RESOLUTION)
    stage_hashes=source_hashes(stage)
    omni.timeline.get_timeline_interface().pause()
    product=rep.create.render_product(HEAD_CAMERA,HIRES_RESOLUTION)
    if instance_backend=='legacy':writer=make_writer(rep,include_instances=True,instance_backend='legacy')
    else:
        from .native_instance_adapter import make_native_writer
        writer=make_native_writer(rep,HIRES_RESOLUTION,instance_backend)
    writer.attach([product])
    cache=StaticGeometryScreenCache(stage,robot['root'],include_generated_plants=True)
    monitor=None
    previous=None
    records=[]
    captured_count=0
    setup_seconds=time.perf_counter()-started
    try:
        for base,target_plan,spec in capture_jobs(plan,anchor):
            view_started=time.perf_counter()
            row=base['generated_row']
            variant=next(v for v in substitution['variants'] if v['variant_id']==row['variant_id'])
            target=next(c for c in catalogue if c['variant_id']==row['variant_id'] and c['component_id']==row['component_id'])
            world=target_world_geometry(stage,variant,row)
            require(np.allclose(world['nominal_world_m'],target_plan['expected_nominal_world_m'],atol=1e-9,rtol=0),
                    'Changed generated target geometry')
            radius=row['cut_region_proposal']['nominal']['petiole_radius_m']
            name=spec['candidate_id']
            decision=dict(candidate_id=name,target_id=row['target_id'],training_approved=False,requested_spec=spec)
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
            render_settings=budget_evidence(render_budget,captured_count)
            request_before=writer.request_index
            native_render_started=time.perf_counter()
            if not profile_render and render_settings['requested_subframes']==56:
                for _ in range(6):step_payload(rep,writer,subframes=8)
            if monitor is None:monitor=StaticSceneMonitor(stage,robot['root'])
            before=monitor.begin()
            render_evidence=None
            if profile_render:
                from .native_render_probe import noise_probe
                payload,render_evidence=noise_probe(rep,writer,cal,roi_pixel=nominal['pixel_xy'])
            else:
                payload=step_payload(rep,writer,subframes=8)
            render_settings['actual_orchestrator_requests']=writer.request_index-request_before
            render_settings['render_seconds']=time.perf_counter()-native_render_started
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
                native_instance_backend=instance_backend,
                native_instance_equivalence=payload.get('native_instance_equivalence'),
                native_instance_mapping_scope=payload.get('native_instance_mapping_scope','legacy_full_mapping'),
                render_budget=render_settings,
                generated_plant_triangle_refinement=True,rendered_camera_params=jsonable(payload['camera_params']),
                input_policy=dict(clean_full_scene=True,isolation=False,diagnostic_overlays=False),
                native_target_pixels=int(mask.sum()),
                synchronization=dict(method='frozen_scene_single_native_writer_payload',
                    scene_unchanged_during_capture=True,dynamic_recording_supported=False,
                    reference_time=reference,freshness=freshness,static_guard=before,
                    native_render_frame=jsonable(payload['pilot_render_frame']),
                    engine_frame_id_verified=False,render_budget_subframes=render_settings['requested_subframes']),
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
            if render_evidence is not None:
                from PIL import Image
                for key in ('_candidate_rgb','_first_reference_rgb'):
                    Image.fromarray(render_evidence.pop(key)).save(folder/'review'/(key[1:]+'.png'))
                write_json(folder/'review/render_probe.json',render_evidence)
                decision['render_probe']=render_evidence
            require(source_hashes(stage)==stage_hashes,'Loaded source geometry changed during capture')
            verify_bindings(plan['source_bindings'])
            decision.update(state='native_captured_pending_review',eligible_annotation=label['eligible'],
                automatic_annotation_eligible=bool(trace is not None and trace['passed']),
                label_reason=label['reason'],label_sha256=sha256(folder/'supervision/label.json'),
                sample_sha256=sha256(folder/'sample.json'),query_trace=trace,
                render_budget=render_settings,
                elapsed_seconds=time.perf_counter()-view_started)
            write_json(output/(name+'_decision.json'),decision);records.append(decision)
            captured_count+=1
            print('NATIVE_GENERATED_VIEW',name,label['eligible'],label['reason'],flush=True)
        verify_bindings(plan['implementation_bindings']);verify_bindings(plan['source_bindings'])
        require(not any(not l.anonymous and l.dirty for l in stage.GetUsedLayers()),'Source layer dirtied')
        return dict(state='native_generated_multiview_pilot_complete_pending_review',
            records=records,captured_frames=sum(r['state']=='native_captured_pending_review' for r in records),
            eligible_annotation_candidates=sum(r.get('eligible_annotation',False) for r in records),
            automatically_clear_annotation_candidates=sum(r.get('automatic_annotation_eligible',False) for r in records),
            setup_seconds=setup_seconds,elapsed_seconds=time.perf_counter()-started,
            geometry_cache=cache.diagnostics(),scene_counts=context['counts'],
            source_assets_unchanged=True,training_approved=False,model_processor_executed=False,
            source_cap_reset=False,render_budget_subframes_per_view=56 if render_budget==REFERENCE else None,
            render_budget_profile=render_budget,experimental_short_profile=render_budget==TRIAL,
            render_profile_experiment=profile_render,
            native_instance_backend=instance_backend,
            target_count=len(plan.get('target_cases',[anchor])),
            total_requested_subframes_per_view=168 if profile_render else (56 if render_budget==REFERENCE else None),
            robot_motion_or_dynamics_validated=False)
    finally:
        if monitor is not None:monitor.close()
        cache.close();writer.detach();product.destroy()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--plan',type=Path);group.add_argument('--batch-plan',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--profile-render',action='store_true',help='Compare1x56 vs two7x8 references; save last reference only')
    p.add_argument('--instance-backend',choices=['legacy','fast','compare'],default='legacy')
    p.add_argument('--render-budget',choices=[REFERENCE,TRIAL],default=REFERENCE,
                   help='Short high-resolution profile is unqualified experimental data only')
    a=p.parse_args()
    require(not a.profile_render or a.render_budget==REFERENCE,'Do not combine render-budget experiments')
    plan_path=a.plan or a.batch_plan;plan=read_json(plan_path)
    if a.batch_plan:
        from .native_multitarget_plan import check as check_batch
        base=check_batch(plan,replay_geometry=False)
    else:base=check(plan)
    output=a.output.resolve()
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
        plan_path=str(plan_path.resolve()),plan_sha256=sha256(plan_path),host_memory_preflight=memory,
        process_admission=admission,training_started=False,automatic_retries=False,
        native_instance_backend=a.instance_backend,
        render_budget_profile=a.render_budget,experimental_budget_override=a.render_budget==TRIAL,
        render_profile_experiment=a.profile_render))
    app=None
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,width=1696,height=816,multi_gpu=False,
            renderer='RaytracedLighting',sync_loads=False,disable_viewport_updates=True,
            extra_args=['--/app/settings/persistent=false']))
        if a.batch_plan:
            base=check_batch(plan,replay_geometry=True)
        result=collect(app,output,plan,base,profile_render=a.profile_render,
                       instance_backend=a.instance_backend,render_budget=a.render_budget)
        verify_bindings(plan['implementation_bindings'])
        write_json(output/'result.json',result)
    except BaseException:
        write_json(output/'failure.json',dict(state='native_multiview_failed',error=traceback.format_exc(),
                                             training_approved=False))
        raise
    finally:
        if app is not None:app.close()


if __name__=='__main__':main()
