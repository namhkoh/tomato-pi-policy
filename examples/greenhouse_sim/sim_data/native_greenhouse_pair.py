"""Bounded paired-resolution greenhouse diagnostic, not a dataset exporter.

Reconstruct one completed TRAINING-family snapshot; render both native sensor
modes without changing the camera mount, optics, source geometry or split.
Run with Isaac Python only after other native collection workers have exited.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import time
import traceback
import numpy as np

from .capture_contract import jsonable, project, depth_evidence, write_sample
from .capture_sensor import SUPPORTED_RESOLUTIONS, HIRES_RESOLUTION, sensor_profile, calibration_for_native_resolution
from .dataset_review import read_json, write_json, require, verify_bindings
from .depth_preview import sha256
from .native_sensor_payload import validate_native_static, decode_native_instances


def load_source(directory, sample_id):
    directory = Path(directory).resolve()
    require(sample_id.startswith("sample_") and sample_id[7:].isdigit(), "Exact sample identifier required")
    manifest_path = directory / "manifest.json"
    manifest = read_json(manifest_path)
    require(manifest.get("state") == "pilot_ready_for_review"
            and manifest.get("source_assets_unchanged") is True, "Completed unchanged native capture required")
    require(manifest.get("target_family_split") == "train", "Qualification uses training families only")
    require(any(s["sample_id"] == sample_id for s in manifest["samples"]), "Unknown source sample")
    sample_path = directory / sample_id / "sample.json"
    sample = read_json(sample_path)
    require(sample["calibration"]["resolution"] == [848, 408], "Legacy native reference required")
    require(sample.get("schema_version") == "greenhouse.rgbd_pilot_sample.v2", "Native sample schema required")
    require(sample["synchronization"]["scene_unchanged_during_capture"] is True
            and sample["synchronization"]["dynamic_recording_supported"] is False, "Frozen source required")
    bindings = {str(manifest_path): sha256(manifest_path), str(sample_path): sha256(sample_path),
                **manifest["source_usd_sha256"],
                manifest["source_collection_plan_path"]: manifest["source_collection_plan_sha256"]}
    for relative, receipt in sample["files"].items():
        path = (sample_path.parent / relative).resolve()
        require(path.is_relative_to(sample_path.parent), "Source file escapes sample")
        bindings[str(path)] = receipt["sha256"]
    verify_bindings(bindings)
    return manifest, sample, bindings


def assert_same_camera(actual, expected):
    require(actual.keys() == expected.keys(), "Calibration fields changed")
    numeric = {"resolution", "intrinsics", "camera_to_world_usd_row_vectors", "clipping_range_m",
               "focal_length_mm", "apertures_mm", "aperture_offsets_mm"}
    for key in expected:
        if key in numeric:
            a, b = np.asarray(actual[key]), np.asarray(expected[key])
            require(a.shape == b.shape and np.allclose(a, b, atol=1e-7, rtol=0),
                    "Reconstructed camera differs: " + key)
        else:
            require(actual[key] == expected[key], "Camera convention differs: " + key)


def assert_pair(low, high):
    expected = calibration_for_native_resolution(low["calibration"], HIRES_RESOLUTION)
    assert_same_camera(high["calibration"], expected)
    for key in ("nominal_world_m", "interval_world_m", "plant_to_world_usd_row_vectors"):
        require(np.allclose(low["supervision"][key], high["supervision"][key], atol=1e-9, rtol=0),
                "Pair changed physical target geometry")
    require(low["supervision"]["target_id"] == high["supervision"]["target_id"], "Pair target differs")
    a = np.asarray([p["pixel_xy"] for p in low["supervision"]["projected_interval"]])
    b = np.asarray([p["pixel_xy"] for p in high["supervision"]["projected_interval"]])
    require(a.shape == b.shape and np.allclose(2*a, b, atol=.01, rtol=0), "Pair projection is not 2x")
    return dict(same_camera_mount_optics_pose=True, same_target_geometry=True,
                native_pixel_scale=2, native_depth_units="metres",
                depth_images_resampled=False, visual_clarity_improvement_measured=False)


def capture_pair(app, output, old_manifest, old_sample, bindings):
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
    from .collection_plan import load_plan
    from .floor_alignment import PACKAGE_FLOOR
    from .robot_preview import add_robot_preview
    from .review_camera import HEAD_CAMERA
    from .static_guard import StaticSceneMonitor
    from .training_screen import StaticBoundScreen

    plan, reports = load_plan(old_manifest["source_collection_plan_path"])
    job = next(j for j in plan["jobs"] if j["job_id"] == old_manifest["collection_job_id"])
    require(job["split"] == "train", "Training-family pilot only")
    row = next(r for r in job["targets"] if r["draft_id"] == old_sample["supervision"]["review_id"])
    require(row["cut_region_proposal"] == old_sample["supervision"]["cut_region_proposal"],
            "Source anatomy proposal changed")
    package = Path(plan["package"])
    wrapper = capture_root(package / "house/green_house_base.usd")
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    stage.SetLoadRules(Usd.StageLoadRules.LoadNone())
    stage.GetRootLayer().subLayerPaths = list(wrapper.subLayerPaths)
    require(stage.GetRootLayer().anonymous and stage.GetPrimAtPath("/World/Gutters"),
            "Original greenhouse wrapper required")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetEditTarget(stage.GetSessionLayer())
    excluded = load_local_payloads(stage)
    records = []
    gutter_x, counts = populate(stage, package, app, records, review_plant=job["plant_family"])
    require(counts == old_manifest["scene_counts"], "Greenhouse population changed")
    require(excluded == old_manifest["unbundled_external_prop_roots_excluded"], "External-prop scope changed")
    variants = old_manifest["variants"]
    robot = add_robot_preview(stage, gutter_x=gutter_x, floor_path=PACKAGE_FLOOR, right_tool="gripper")
    freeze_rigid_bodies(stage)

    # Source lighting is reconstructed, never brightened for one target or resolution.
    import sys
    sys.path.insert(0, str(package / "env_panel"))
    from tomato_env import daylight
    require(plan["configuration"].get("clear_capture") == "robot_head_close_diffuse_v1",
            "Qualified clear-lighting source required")
    lighting = daylight.apply(stage, day=172, minutes=13*60, intensity=1500, dome_intensity=6000)
    require(lighting == old_manifest["lighting"], "Lighting reconstruction changed")
    settings = carb.settings.get_settings()
    require(old_manifest["renderer"] == "RealTimePathTracing", "Pilot currently requires qualified RTPT source")
    settings.set("/rtx/rendermode", old_manifest["renderer"])
    app.update()
    require(settings.get("/rtx/rendermode") == old_manifest["renderer"], "Renderer mismatch")
    rep.set_global_seed(plan["configuration"]["seed"])

    pose = old_sample["robot_snapshot"]
    root = np.asarray(pose["robot_root_to_world_usd_row_vectors"]).T
    links = robot_kinematics.Rby1Kinematics().all_link_transforms(pose["joint_degrees"])
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        robot_hardware._set_transform(stage.GetPrimAtPath(robot["root"]), root[:3, :3], root[:3, 3])
        for link, matrix in links.items():
            prim = stage.GetPrimAtPath(robot["root"] + "/" + link)
            require(bool(prim), "Missing robot link")
            robot_hardware._set_transform(prim, matrix[:3, :3], matrix[:3, 3])
    cal = calibration(stage)
    assert_same_camera(cal, old_sample["calibration"])
    screen = StaticBoundScreen(static_obstacles(stage, robot["root"]), scene_triangle_refiner(stage))(
        visible_bounds(stage, robot["root"]))
    require(screen["passed"], "Reconstructed snapshot failed geometry screen")

    catalogue = component_catalogue(stage, records, reports, variants)
    variant = next(v for v in variants if v["variant_id"] == row["variant_id"])
    world = target_world_geometry(stage, variant, row)
    for key in world:
        require(np.allclose(world[key], old_sample["supervision"][key], atol=1e-9, rtol=0),
                "Reconstructed target moved: " + key)
    target = next(c for c in catalogue if c["variant_id"] == row["variant_id"]
                  and c["component_id"] == row["target_id"].split("/")[-1])
    radius = row["cut_region_proposal"]["nominal"]["petiole_radius_m"]
    hashes_before = source_hashes(stage)
    require(all(hashes_before.get(p) == h for p, h in old_manifest["source_usd_sha256"].items()),
            "Loaded source layer set differs from reference")
    context.get_selection().clear_selected_prim_paths()
    omni.timeline.get_timeline_interface().pause()
    samples = []
    for resolution in SUPPORTED_RESOLUTIONS:
        folder = output / f"native_{resolution[0]}x{resolution[1]}"
        product = rep.create.render_product(HEAD_CAMERA, resolution)
        writer = make_writer(rep, include_instances=True, instance_backend="legacy")
        writer.attach([product])
        monitor = None
        try:
            # Existing reference budget at both resolutions: same 56 subframes.
            for _ in range(6):
                step_payload(rep, writer, subframes=8)
            monitor = StaticSceneMonitor(stage, robot["root"])
            native_cal = calibration_for_native_resolution(calibration(stage), resolution)
            assert_same_camera(native_cal, calibration_for_native_resolution(cal, resolution))
            before = monitor.begin()
            started = time.perf_counter()
            payload = step_payload(rep, writer, subframes=8)
            rgb, depth, valid, reference, token = validate_native_static(
                payload, native_cal, before, monitor.token(), writer.sequence)
            assert_same_camera(calibration(stage), cal)
            require(settings.get("/rtx/rendermode") == old_manifest["renderer"], "Renderer changed")
            instances, mapping = decode_native_instances(payload, resolution)
            components, organs, owners = component_masks(instances, mapping, catalogue)
            nominal = project([world["nominal_world_m"]], native_cal)[0]
            interval = project(world["interval_world_m"], native_cal)
            visibility, mask = interval_visibility(nominal, interval, depth, valid, instances, mapping,
                                                   owners, target, radius)
            quality = view_quality(native_cal, nominal, interval, radius, visibility, rgb, mask)
            metadata = dict(schema_version="greenhouse.native_resolution_pilot_sample.v1",
                state="diagnostic_pending_visual_and_export_qualification",
                sample_id=folder.name, training_sample_approved=False,
                sensor=sensor_profile(resolution), calibration=native_cal, robot_snapshot=pose,
                input_policy=dict(clean_full_scene=True, isolation=False, diagnostic_overlays=False),
                quality=quality, rendered_camera_params=jsonable(payload["camera_params"]),
                synchronization=dict(method="frozen_scene_single_native_writer_payload",
                    scene_unchanged_during_capture=True, dynamic_recording_supported=False,
                    reference_time=reference, freshness=token, static_guard=before,
                    native_render_frame=jsonable(payload["pilot_render_frame"]),
                    engine_frame_id_verified=False, render_budget_subframes=56),
                supervision=dict(**world, target_id=row["target_id"], split_group=row["split_group"],
                    nominal_projected=nominal, projected_interval=interval,
                    depth_evidence=depth_evidence(nominal, depth, valid, radius),
                    visibility_evidence=visibility, cut_safety_validated=False))
            metadata["last_step_and_validation_s"] = time.perf_counter()-started
            write_sample(folder, rgb, depth, valid, metadata)
            write_visibility(folder, instances, mapping, catalogue, components, organs, mask)
            metadata = read_json(folder / "sample.json")
            samples.append(metadata)
            print("NATIVE_PAIR_CAPTURED", folder.name, quality, flush=True)
        finally:
            if monitor is not None:
                monitor.close()
            writer.detach()
            product.destroy()
    pair_checks = assert_pair(*samples)
    require(source_hashes(stage) == hashes_before, "Source USD changed during paired capture")
    require(not any(not layer.anonymous and layer.dirty for layer in stage.GetUsedLayers()),
            "Source USD layer dirtied")
    verify_bindings(bindings)
    return dict(state="native_greenhouse_pair_captured_pending_visual_review", samples=samples,
                pair_checks=pair_checks, source_assets_unchanged=True,
                scene_counts=counts, geometry_screen=screen, lighting=lighting,
                training_approved=False, exporter_qualified=False, source_split="train",
                synchronized_dynamic_recording_supported=False, physical_pose_or_action_validated=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    manifest, sample, bindings = load_source(args.source_capture, args.sample)
    require(not output.exists() and not output.is_relative_to(args.source_capture.resolve())
            and not output.is_relative_to(Path(manifest["package"]).resolve()), "New output outside sources required")
    output.mkdir(parents=True)
    from sim_physics.host_memory import preflight
    reserve = preflight()
    write_json(output / "request.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        source_bindings=bindings, host_memory_preflight=reserve,
        source_sample=args.sample, sensor_modes=[sensor_profile(r) for r in SUPPORTED_RESOLUTIONS],
        training_started=False))
    app = None
    succeeded = False
    try:
        require(reserve["allowed"], "Native memory reserve unavailable; no override")
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, width=1696, height=816, multi_gpu=False,
            renderer="RaytracedLighting", sync_loads=False, disable_viewport_updates=True,
            extra_args=["--/app/settings/persistent=false"]))
        from .native_resolution_smoke import smoke
        smoke_output = output / "sensor_smoke"
        smoke_output.mkdir()
        smoke_result = smoke(app, smoke_output)
        write_json(smoke_output / "result.json", smoke_result)
        result = capture_pair(app, output, manifest, sample, bindings)
        result["native_sensor_smoke_passed"] = True
        write_json(output / "result.json", result)
        succeeded = True
    except BaseException:
        write_json(output / "failure.json", dict(state="native_greenhouse_pair_failed",
            traceback=traceback.format_exc(), training_approved=False))
        raise
    finally:
        if app is not None:
            app.close(exit_code=0 if succeeded else 1)


if __name__ == "__main__":
    main()
