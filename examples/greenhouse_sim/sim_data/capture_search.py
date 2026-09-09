"""Optional bounded full-scene viewpoint search, called inside capture_pilot."""
from __future__ import annotations

import hashlib
import json
import time

import numpy as np
from pxr import UsdGeom

from .capture_contract import (RESOLUTION, depth_evidence, fingerprint, jsonable,
                               project, validate_static_capture, write_sample)
from .capture_scene import calibration, scene_guard, set_snapshot_pose, target_world_geometry
from .capture_viewpoints import (candidate_specs, component_catalogue, select_diverse,
                                 scene_triangle_refiner, screen_bounds, static_obstacles, visible_bounds)
from .capture_visibility import (component_masks, decode_instances, interval_visibility,
                                 view_quality, write_visibility)
from .review_camera import HEAD_CAMERA


def run_refined_capture(stage, rep, args, manifest, robot, records, reports, variants, rows, *, prepared_plan=None):
    from .capture_pilot import make_writer, step_payload
    if not manifest["calibration_smoke"].get("native_identity_and_occluder_smoke_passed"):
        raise ValueError("Refined capture requires the real-GPU instance/depth occlusion preflight")
    catalogue = component_catalogue(stage, records, reports, variants)
    obstacles = static_obstacles(stage, robot["root"])
    refine = scene_triangle_refiner(stage)
    original_x = float(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(robot["root"]))[3][0])
    by_variant = {v["variant_id"]: v for v in variants}
    plan = prepared_plan
    grounding = plan is not None and plan.get('schema_version')=='greenhouse.grounding_job_view_plan.v1'
    fast_screen=None
    if grounding:
        from .training_screen import StaticBoundScreen
        fast_screen=StaticBoundScreen(obstacles,refine)
    plan_path = args.output / "planned_views.json" if prepared_plan is not None else None
    if prepared_plan is not None:
        from .viewpoint_plan import focus_specs
        if getattr(args, "view_plan", None) or set(plan["selected"]) != {r["draft_id"] for r in rows}:
            raise ValueError("Conflicting or mismatched prepared native view plan")
        if not np.isclose(plan["original_base_x_m"], original_x, atol=1e-9, rtol=0):
            raise ValueError("Prepared view plan original base mismatch")
        for target_id, planned in plan["selected"].items():
            if grounding:
                from .training_plan import view_specs
                bound=plan['requested_views_per_target']
                allowed={s['candidate_id']:s for s in view_specs(original_x, plan['target_world_m'][target_id][0], target_id, bound,
                                                                vary_torso=plan.get('vary_torso',False),view_offset=plan.get('view_offset',0))}
            else:
                bound=3
                allowed = {s["candidate_id"]:s for s in focus_specs(original_x, plan["target_world_m"][target_id][0])}
            if len(planned) > bound or len({p["candidate_id"] for p in planned}) != len(planned):
                raise ValueError("Prepared native capture exceeds candidate bound")
            for item in planned:
                spec = allowed.get(item["candidate_id"])
                if spec is None or any(item.get(k) != v for k,v in spec.items()) or not item["visual_bound_screen"]["passed"]:
                    raise ValueError("Prepared native viewpoint was altered or not screened")
    if getattr(args, "view_plan", None):
        from .viewpoint_plan import load_plan
        plan = load_plan(args.view_plan, args.drafts, args.package)
        plan_path = args.view_plan
        if not np.isclose(plan["original_base_x_m"], original_x, atol=1e-9, rtol=0):
            raise ValueError("View plan original base pose does not match capture")
        rows = [r for r in rows if r["draft_id"] in plan["selected"]]
    manifest["viewpoint_selection"] = {
        "method": "bounded_geometry_guided_static_base_xy_head_framing_and_rendered_visibility_ranking.v1",
        "original_base_x_m": original_x, "maximum_approach_m": .3,
        "robot_yaw_and_arm_torso_pose_unchanged": True, "camera_resolution_unchanged": True,
        "lighting_unchanged_across_candidates": True, "all_candidate_decisions": [],
        "clear_view_sample_count": 0, "difficulty_assigned": False,
        "blind_evaluation_compatible": False, "robot_motion_commanded": False}
    if plan is not None:
        manifest["viewpoint_selection"].update(method="bounded_cpu_screened_base_xy_yaw_and_real_head_capture.v2",
            maximum_approach_m=.4, robot_yaw_and_arm_torso_pose_unchanged=False,
            arm_and_torso_joint_pose_unchanged=True, base_yaw_range_degrees=[150, 210],
            plan_path=str(plan_path.resolve()), plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())
        if plan.get('vary_torso'):
            manifest['viewpoint_selection'].update(arm_and_torso_joint_pose_unchanged=False,
                arm_joint_pose_unchanged=True,torso_pose_protocol='real_base_head_and_torso_snapshots.v1')
    product = rep.create.render_product(HEAD_CAMERA, RESOLUTION)
    writer = make_writer(rep, include_instances=True, instance_backend=getattr(args,'instance_backend','legacy'))
    writer.attach([product])
    previous, previous_instances = None, None
    monitor = None
    if grounding:
        from .static_guard import StaticSceneMonitor
        monitor = StaticSceneMonitor(stage, robot['root'])
    def store_captured(captured, row):
        sample_id = f"sample_{len(manifest['samples'])+1:04d}"
        meta = captured['metadata']
        meta['sample_id'] = sample_id
        directory = args.output / sample_id
        write_sample(directory, captured['rgb'], captured['depth'], captured['valid'], meta)
        write_visibility(directory, captured['instances'], captured['mapping'], catalogue,
                         captured['components'], captured['organs'], captured['target_mask'])
        for field,filename in [('short_reference_rgb','review/short_render_reference.png'),
                               ('diagnostic_reference_rgb','review/first_long_reference.png')]:
            if captured.get(field) is None: continue
            from PIL import Image
            Image.fromarray(captured[field]).save(directory/filename)
            path=directory/'sample.json'
            saved=json.loads(path.read_text(encoding='utf-8'))
            saved['files'][filename]=dict(sha256=hashlib.sha256((directory/filename).read_bytes()).hexdigest(),role='review_only')
            path.write_text(json.dumps(saved,indent=2,allow_nan=False),encoding='utf-8')
        summary = dict(sample_id=sample_id, target_review_id=row['draft_id'],
            nominal_pixel_xy=meta['supervision']['nominal_projected']['pixel_xy'],
            depth_status=meta['supervision']['depth_evidence']['status'],
            nominal_visibility_status=captured['visibility']['nominal']['status'],
            quality=captured['quality'], candidate_id=captured['candidate_id'])
        manifest['samples'].append(summary)
        manifest['viewpoint_selection']['clear_view_sample_count'] += int(captured['quality']['clear_view_gate_passed'])
        print('REFINED_PILOT_SAMPLE ' + json.dumps(summary), flush=True)
    try:
        for row in rows:
            if len(manifest["samples"]) >= args.max_samples:
                break
            variant = by_variant[row["variant_id"]]
            world = target_world_geometry(stage, variant, row)
            target = next(c for c in catalogue if c["variant_id"] == row["variant_id"]
                          and c["component_id"] == row["target_id"].split("/")[-1])
            captures = []
            if plan is not None and not np.allclose(world["nominal_world_m"], plan["target_world_m"][row["draft_id"]], atol=1e-9, rtol=0):
                raise ValueError("View plan target geometry does not match capture")
            specs = (plan["selected"][row["draft_id"]] if plan is not None else candidate_specs(original_x, world["nominal_world_m"][0]))
            for planned in specs:
                started = time.perf_counter()
                if grounding and len(manifest['samples']) >= args.max_samples:
                    break
                spec = {k: planned[k] for k in ("candidate_id", "root_x_m", "approach_from_original_m", "y_offset_m", "desired_pixel_xy")}
                if "root_yaw_degrees" in planned:
                    spec["root_yaw_degrees"] = planned["root_yaw_degrees"]
                if 'torso_bend_degrees' in planned: spec['torso_bend_degrees']=planned['torso_bend_degrees']
                decision = {"target_review_id": row["draft_id"], **spec, "state": "screening"}
                manifest["viewpoint_selection"]["all_candidate_decisions"].append(decision)
                try:
                    from .training_views import robot_for_spec
                    pose = set_snapshot_pose(stage, robot_for_spec(robot,spec), world["nominal_world_m"], spec["y_offset_m"],
                        spec["desired_pixel_xy"], root_x_m=spec["root_x_m"], root_yaw_degrees=spec.get("root_yaw_degrees"))
                except ValueError as exc:
                    decision.update(state="rejected_pose", reason=str(exc))
                    print("VIEWPOINT_REJECTED " + json.dumps(decision), flush=True)
                    continue
                bounds=visible_bounds(stage,robot['root'])
                screen = fast_screen(bounds) if fast_screen else screen_bounds(bounds, obstacles, refinement=refine)
                decision["visual_bound_screen"] = screen
                if not screen["passed"]:
                    decision["state"] = "rejected_possible_geometry_overlap"
                    print("VIEWPOINT_REJECTED " + json.dumps({"target":row["draft_id"],
                        "candidate":spec["candidate_id"], "overlap_count":screen["possible_overlap_count"],
                        "first_overlap":screen["possible_overlaps"][:1]}), flush=True)
                    continue
                screened_at = time.perf_counter()
                settling=None
                short_rgb=None
                diagnostic_reference_rgb=None
                if monitor:
                    from .static_render import reference_payload, settled_payload, consolidated_payload, noise_probe_payload
                    before,cal=monitor.begin(),calibration(stage)
                    profiler=None
                    if getattr(args,'profile_first_render',False) and not manifest['samples']:
                        import cProfile
                        profiler=cProfile.Profile()
                        profiler.enable()
                    try:
                        if getattr(args,'render_budget','established')=='warm56_then8':
                            if not manifest['samples']:
                                payload,settling=reference_payload(rep,writer,cal)
                            else:
                                payload,settling=consolidated_payload(rep,writer,cal,subframes=8)
                            settling['profile']='initial_full_warmup_then_eight_subframe_snapshots.v1'
                            settling['photometric_identity_to_long_render_claimed']=False
                        elif getattr(args,'render_budget','established')=='noise_probe':
                            payload,settling=noise_probe_payload(rep,writer,cal,roi_pixel=spec['desired_pixel_xy'])
                        elif getattr(args,'render_budget','established') in ('single56','single56_compare'):
                            payload,settling=consolidated_payload(rep,writer,cal,roi_pixel=spec['desired_pixel_xy'],
                                compare=args.render_budget=='single56_compare')
                        elif getattr(args,'render_reference_check',False):
                            payload,settling=settled_payload(rep,writer,cal,roi_pixel=spec['desired_pixel_xy'],
                                verify_long_reference=True)
                        else:
                            payload,settling=reference_payload(rep,writer,cal)
                    finally:
                        if profiler is not None:
                            import pstats
                            profiler.disable()
                            profiler.dump_stats(str(args.output/'first_render.pstats'))
                            with (args.output/'first_render_profile.txt').open('x',encoding='utf-8') as profile_log:
                                pstats.Stats(profiler,stream=profile_log).sort_stats('cumulative').print_stats(50)
                    short_rgb=settling.pop('_short_rgb',None)
                    diagnostic_reference_rgb=settling.pop('_reference_rgb',None)
                else:
                    for _ in range(6): step_payload(rep,writer)
                warmed_at = time.perf_counter()
                if not monitor: before,cal=scene_guard(stage),calibration(stage)
                guarded_at = time.perf_counter()
                if not monitor: payload=step_payload(rep,writer)
                if fingerprint(calibration(stage)) != fingerprint(cal):
                    raise ValueError("Camera changed during refined capture")
                if grounding:
                    import carb
                    if carb.settings.get_settings().get('/rtx/rendermode')!=manifest['renderer']:
                        raise ValueError('Renderer changed during native capture')
                rgb, depth, valid, reference, previous = validate_static_capture(
                    payload, cal, before, monitor.token() if monitor else scene_guard(stage), writer.sequence, previous)
                captured_at = time.perf_counter()
                instances, mapping = decode_instances(payload)
                instance_hash = hashlib.sha256(instances.tobytes()).hexdigest()
                if instance_hash == previous_instances:
                    raise ValueError("Unchanged instance mask after camera movement; refuse stale segmentation")
                previous_instances = instance_hash
                components, organs, owners = component_masks(instances, mapping, catalogue)
                nominal, interval = project([world["nominal_world_m"]], cal)[0], project(world["interval_world_m"], cal)
                radius = row["cut_region_proposal"]["nominal"]["petiole_radius_m"]
                visibility, target_mask = interval_visibility(nominal, interval, depth, valid,
                    instances, mapping, owners, target, radius)
                quality = view_quality(cal, nominal, interval, radius, visibility, rgb, target_mask)
                quality.update(valid_depth_fraction=float(valid.mean()), rgb_std=float(rgb.std()))
                decision.update(state="rendered_candidate", quality=quality,
                    nominal_visibility_status=visibility["nominal"]["status"],
                    sampled_interval_visible_pixel_fraction=visibility["sampled_interval_visible_pixel_fraction"])
                pose["visual_bound_screen"] = screen
                pose["pose_sampling"] = "bounded_geometry_guided_viewpoint_search_not_navigation"
                pose["whole_robot_collision_checked"] = False
                meta = {"schema_version":"greenhouse.rgbd_pilot_sample.v2", "state":"complete_pending_review",
                    "training_sample_approved":False,
                    "input_policy":{"clean_full_scene":True,"diagnostic_overlays":False,"isolation":False,
                        "allowed_observation_files":["inputs/rgb.png","inputs/depth_m.npy","inputs/depth_valid.png"]},
                    "calibration":cal,"robot_snapshot":pose,"viewpoint_candidate":spec,"quality":quality,
                    "synchronization":{"method":"frozen_scene_single_writer_payload_camera_content_and_instance_checks",
                        "reference_time":reference,"writer_callback_sequence":writer.sequence,
                        "native_render_frame":jsonable(payload["pilot_render_frame"]),
                        "engine_frame_id_verified":False,"dynamic_recording_supported":False,
                        "freshness":previous,"instance_buffer_sha256":instance_hash,"scene_state_sha256":before,
                        "scene_unchanged_during_capture":True,"stale_frame_fallback_allowed":False,"rt_subframes":8},
                    "rendered_camera_params":jsonable(payload["camera_params"]),
                    "supervision":{"review_id":row["draft_id"],"target_id":row["target_id"],
                        "variant_id":row["variant_id"],"split_group":row["split_group"],
                        "coverage":"one_provisional_cut_target_plus_visible_component_masks_for_two_detailed_plants",
                        "cut_region_proposal":row["cut_region_proposal"], **world,
                        "nominal_projected":nominal,"projected_interval":interval,
                        "depth_evidence":depth_evidence(nominal,depth,valid,radius),
                        "visibility_evidence":visibility,"semantic_instance_masks":"supervision/",
                        "difficulty":"unassigned","human_cut_approval":False,"grasp_region":None,
                        "executable_trajectory":None,"physics_validated":False,
                        "visibility_confirmed":visibility["nominal"]["visible_target_evidence"],
                        "visibility_confirmed_scope":"nominal_projected_pixel_identity_and_depth_only_not_entire_branch"}}
                captures.append({"candidate_id":spec["candidate_id"],"base_xy_m":
                    np.asarray(pose["robot_root_to_world_usd_row_vectors"])[3,:2].tolist(),
                    "quality":quality,"visibility":visibility,"metadata":meta,"rgb":rgb,"depth":depth,
                    "valid":valid,"instances":instances,"mapping":mapping,"components":components,
                    "organs":organs,"target_mask":target_mask,"short_reference_rgb":short_rgb,
                    "diagnostic_reference_rgb":diagnostic_reference_rgb})
                if monitor:
                    meta['synchronization'].update(scene_guard_method='initial_full_static_scan_and_USD_change_notice.v1',
                        initial_full_scene_guard_sha256=monitor.baseline,usd_notice_generation=monitor.generation,
                        rt_subframes=56 if getattr(args,'render_budget','established')=='single56' else 8,render_settling=settling)
                    meta['synchronization'].update(instance_annotation_backend=getattr(args,'instance_backend','legacy'),
                        instance_mapping_scope=payload.get('native_instance_mapping_scope','native_full_mapping_table'),
                        fast_legacy_equivalence=payload.get('native_instance_equivalence'))
                meta['performance_seconds']=dict(pose_and_geometry_screen=screened_at-started,
                    warmup=warmed_at-screened_at,guard=guarded_at-warmed_at,
                    native_capture=captured_at-guarded_at,mask_and_label=time.perf_counter()-captured_at)
                print('CAPTURE_TIMING '+json.dumps(meta['performance_seconds']),flush=True)
                print("VIEWPOINT_RENDERED " + json.dumps({"target":row["draft_id"],"candidate":spec["candidate_id"],
                    "quality":quality,"nominal_visibility":visibility["nominal"]["status"]}), flush=True)
                if grounding:
                    # Stream every distinct rendered candidate, including occlusions. No
                    # best-view-only selection and no plant-sized RGB-D array accumulation.
                    store_captured(captures.pop(), row)
            for captured in select_diverse(captures, min(3, args.max_samples-len(manifest["samples"]))):
                store_captured(captured, row)
    finally:
        if monitor:
            monitor.close()
        writer.detach()
        product.destroy()
        # Preserve the search audit even if native capture fails mid-run.
        (args.output / "viewpoint_search.json").write_text(json.dumps(jsonable(
            manifest["viewpoint_selection"]), indent=2, allow_nan=False), encoding="utf-8")
