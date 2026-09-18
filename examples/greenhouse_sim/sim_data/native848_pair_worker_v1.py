"""Bounded native848 full-greenhouse original/generated pair, never a release.

The capture body preserves the frozen native_generated_pair geometry, writer,
56-subframe budget and same-callback checks. This explicit producer owns new
schema identifiers and native848 dimensions. No legacy1696 prerequisite claim.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,shutil,time,traceback
import numpy as np
from . import native848_pair_plan_v1 as ap
from .dataset_review import require,read_json,write_json,verify_bindings
from .depth_preview import sha256
from .capture_sensor import LEGACY_RESOLUTION,calibration_for_native_resolution,sensor_profile
from .capture_contract import fingerprint,jsonable,project,depth_evidence,write_sample
from .generated_capture import substitute_plant,assert_pair_fresh
from .native_greenhouse_pair import assert_same_camera
from .native_generated_pair import windows_worker_admission

def capture(app, output, plan, *, profile_geometry_cache=False):
    import carb
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep
    from pxr import Usd, UsdGeom
    from launch_sim_data import load_local_payloads, populate
    from greenhouse_sim import robot_hardware, robot_kinematics
    from .capture_scene import capture_root, calibration, freeze_rigid_bodies, target_world_geometry
    from .capture_pilot import make_writer, step_payload, source_hashes
    from .capture_viewpoints import component_catalogue, static_obstacles, scene_triangle_refiner, visible_bounds
    from .capture_visibility import component_masks, interval_visibility, view_quality, write_visibility
    from .native_sensor_payload import validate_native_static, decode_native_instances
    from .collection_plan import load_plan
    from .floor_alignment import PACKAGE_FLOOR
    from .robot_preview import add_robot_preview
    from .review_camera import HEAD_CAMERA
    from .static_guard import StaticSceneMonitor
    from .training_screen import StaticBoundScreen
    from .plant_variant_catalogue import load_for_inspection

    from .native_scene import prepare_native_scene
    scene = prepare_native_scene(app, plan)
    (started, stage, records, reports, variants, generated, counts, robot, old_manifest, old_sample, original_bindings, original_variant, pose, settings, setup_seconds) = (
        scene[k] for k in ["started","stage","records","reports","variants","generated","counts","robot","old_manifest","old_sample","original_bindings","original_variant","pose","settings","setup_seconds"])
    product = rep.create.render_product(HEAD_CAMERA, LEGACY_RESOLUTION)
    writer = make_writer(rep, include_instances=True, instance_backend="legacy")
    writer.attach([product])
    samples = []
    phase_timings = []
    substitution = None
    geometry_profiles = []
    geometry_cache = None
    if profile_geometry_cache:
        from .static_geometry_cache import StaticGeometryScreenCache, profile_against_reference
        geometry_cache = StaticGeometryScreenCache(stage, robot['root'], include_generated_plants=True)
    try:
        for mode, row in zip(plan["modes"], (plan["source_row"], plan["generated_row"]), strict=True):
            if mode == "generated_variant":
                substitution = substitute_plant(stage, original_variant, records, variants, generated)
                records, variants = substitution["records"], substitution["variants"]
                reports = [*reports, substitution["report"]]
            frame_started = time.perf_counter()
            timing = {}
            cal = calibration_for_native_resolution(calibration(stage), LEGACY_RESOLUTION)
            assert_same_camera(cal, plan["expected_calibration"])
            screen = StaticBoundScreen(static_obstacles(stage, robot["root"]),
                scene_triangle_refiner(stage, include_generated_plants=True))(
                visible_bounds(stage, robot["root"]))
            # Preserve the exact rejection evidence even if no frame is captured.
            write_json(output/("geometry_screen_"+mode+".json"), dict(
                mode=mode, generated_plant_triangle_refinement=True, screen=screen,
                scope="plant_surfaces_vs_robot_bounds_not_collision_free_certification"))
            require(screen["passed"], "Current snapshot intersects robot/environment geometry")
            timing["calibration_and_geometry_screen_s"] = time.perf_counter()-frame_started
            if geometry_cache is not None:
                geometry_profiles.append(dict(mode=mode,**profile_against_reference(geometry_cache,screen)))
            phase_started = time.perf_counter()
            catalogue = component_catalogue(stage, records, reports, variants)
            require(len(catalogue) == counts["components"], "Active organ catalogue population changed")
            variant = next(v for v in variants if v["variant_id"] == row["variant_id"])
            world = target_world_geometry(stage, variant, row)
            target = next(c for c in catalogue if c["variant_id"] == row["variant_id"]
                          and c["component_id"] == row["component_id"])
            radius = row["cut_region_proposal"]["nominal"]["petiole_radius_m"]
            phase_hashes = source_hashes(stage)
            timing["catalogue_world_geometry_source_hash_s"] = time.perf_counter()-phase_started
            phase_started = time.perf_counter()
            for _ in range(6):
                step_payload(rep, writer, subframes=8)
            timing["six_native_warmup_payloads_s"] = time.perf_counter()-phase_started
            phase_started = time.perf_counter()
            monitor = StaticSceneMonitor(stage, robot["root"])
            timing["static_guard_initial_scan_s"] = time.perf_counter()-phase_started
            try:
                before = monitor.begin()
                phase_started = time.perf_counter()
                payload = step_payload(rep, writer, subframes=8)
                timing["final_native_payload_s"] = time.perf_counter()-phase_started
                phase_started = time.perf_counter()
                rgb, depth, valid, reference, freshness = validate_native_static(
                    payload, cal, before, monitor.token(), writer.sequence)
                assert_same_camera(calibration_for_native_resolution(calibration(stage), LEGACY_RESOLUTION), cal)
                require(settings.get("/rtx/rendermode") == old_manifest["renderer"], "Renderer changed")
                instances, mapping = decode_native_instances(payload, LEGACY_RESOLUTION)
                components, organs, owners = component_masks(instances, mapping, catalogue)
                nominal, interval = project([world["nominal_world_m"]], cal)[0], project(world["interval_world_m"], cal)
                visibility, mask = interval_visibility(nominal, interval, depth, valid, instances, mapping,
                                                       owners, target, radius)
                quality = view_quality(cal, nominal, interval, radius, visibility, rgb, mask)
                old_root = original_variant["plant_root"]
                old_ids = [i for i, path in mapping.items() if path == old_root or path.startswith(old_root+"/")]
                old_pixels = int(np.isin(instances, old_ids).sum())
                metadata = dict(schema_version=ap.SAMPLE_SCHEMA,
                    state="native_static_diagnostic_pending_visual_review",
                    sample_id=mode, training_sample_approved=False, sensor=sensor_profile(LEGACY_RESOLUTION),
                    calibration=cal, robot_snapshot=pose, scene_counts=counts, lighting=lighting,
                    renderer=old_manifest["renderer"], geometry_screen=screen, quality=quality,
                    generated_plant_triangle_refinement=True,
                    rendered_camera_params=jsonable(payload["camera_params"]),
                    input_policy=dict(clean_full_scene=True, isolation=False, diagnostic_overlays=False),
                    native_target_pixels=int(mask.sum()), old_plant_native_pixels=old_pixels,
                    native_instance_backend="legacy", historical_labels_inherited=False,
                    native_instance_sha256=hashlib.sha256(instances.tobytes()).hexdigest(),
                    native_mapping_sha256=fingerprint(mapping),
                    synchronization=dict(method="frozen_scene_single_native_writer_payload",
                        scene_unchanged_during_capture=True, dynamic_recording_supported=False,
                        reference_time=reference, freshness=freshness, static_guard=before,
                        native_render_frame=jsonable(payload["pilot_render_frame"]),
                        engine_frame_id_verified=False, render_budget_subframes=56),
                    supervision=dict(**world, target_id=row["target_id"], source_target_id=plan["source_row"]["target_id"],
                        split_group=plan["split_group"], conservative_view_cap_group=plan["conservative_view_cap_group"],
                        cut_region_proposal=row["cut_region_proposal"], nominal_projected=nominal,
                        projected_interval=interval, depth_evidence=depth_evidence(nominal, depth, valid, radius),
                        visibility_evidence=visibility, cut_safety_validated=False),
                    elapsed_scene_screen_and_capture_s=time.perf_counter()-frame_started)
                folder = output/mode
                timing["native_validation_masks_quality_metadata_s"] = time.perf_counter()-phase_started
                phase_started = time.perf_counter()
                write_sample(folder, rgb, depth, valid, metadata)
                write_visibility(folder, instances, mapping, catalogue, components, organs, mask)
                samples.append(read_json(folder/"sample.json"))
                timing["artifact_write_s"] = time.perf_counter()-phase_started
                phase_started = time.perf_counter()
                require(source_hashes(stage) == phase_hashes, "Loaded source USD changed during capture")
                timing["postcapture_source_hash_s"] = time.perf_counter()-phase_started
                phase_timings.append(dict(mode=mode,seconds=timing,
                    total_frame_s=time.perf_counter()-frame_started,render_subframes_unchanged=56))
                print("GENERATED_NATIVE_FRAME", mode, quality, flush=True)
            finally:
                monitor.close()
    finally:
        if geometry_cache is not None:
            geometry_cache.close()
        writer.detach()
        product.destroy()
    checks = assert_pair_fresh(*samples)
    verify_bindings(plan["source_bindings"])
    verify_bindings(original_bindings)
    require(not any(not layer.anonymous and layer.dirty for layer in stage.GetUsedLayers()), "Source layer dirtied")
    return dict(state=ap.RESULT_STATE,
        samples=samples, checks=checks, substitution=substitution and {
            k: substitution[k] for k in ("old_root", "new_root", "plant_to_world", "component_count_preserved")},
        setup_seconds=setup_seconds, elapsed_seconds=time.perf_counter()-started,
        phase_timings=phase_timings,profiling_did_not_change_capture_budget_or_gates=True,
        geometry_cache_profiles=geometry_profiles,geometry_cache_profile_enabled=profile_geometry_cache,
        source_assets_unchanged=True, scene_counts=counts, training_approved=False,
        local_plant_collision_qualified=False, low_level_policy_or_physics_action_tested=False,
        export_schema_qualified=False, original_controls_are_new_target_diversity=False)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(argv)
    require(sha256(a.plan)==a.plan_sha256,'Changed848 plan')
    plan=read_json(a.plan);ap.check(plan)
    output=a.output.resolve()
    package=Path(read_json(plan['source_collection_plan'])['package']).resolve()
    protected=[Path(plan['source_capture']).resolve(),Path(plan['variant_directory']).resolve(),package]
    require(not output.exists() and all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected),'New output disjoint from sources required')
    from sim_physics.host_memory import preflight
    reserve=preflight();disk=shutil.disk_usage(output.parent).free
    require(reserve['allowed'] and reserve['commit_headroom_bytes']>=20*2**30 and disk>=60*2**30,'Native reserves unavailable')
    admission=windows_worker_admission(None)
    output.mkdir(parents=True)
    write_json(output/'request.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),plan_path=str(a.plan.resolve()),plan_sha256=a.plan_sha256,resolution=[848,408],host_memory_preflight=reserve,available_disk_bytes=disk,process_admission=admission,training_approved=False,automatic_retries=False))
    app=None;succeeded=False
    try:
        from isaacsim import SimulationApp
        app=SimulationApp(dict(headless=True,width=848,height=408,multi_gpu=False,renderer='RaytracedLighting',sync_loads=False,disable_viewport_updates=True,extra_args=['--/app/settings/persistent=false']))
        ap.check(plan,full=True)
        result=capture(app,output,plan)
        require(sha256(a.plan)==a.plan_sha256,'Plan changed during native execution')
        ap.check(plan)
        result.update(resolution=[848,408],plan_sha256=a.plan_sha256,**ap.FLAGS)
        write_json(output/'result.json',result);succeeded=True
    except BaseException:
        write_json(output/'failure.json',dict(state='generated_native848_pair_failed',error=traceback.format_exc(),training_approved=False))
        raise
    finally:
        if app is not None:app.close(exit_code=0 if succeeded else 1)

if __name__=='__main__':main()
