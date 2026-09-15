"""Bounded original-only native collector: one scene/product, 1..64 target poses.

Only main explicitly launches Kit. Import/preparation/audit never launch it.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import time
import traceback

import numpy as np

from ..capture_contract import jsonable, project, depth_evidence, write_sample, fingerprint
from ..capture_sensor import sensor_profile, calibration_for_native_resolution
from ..native_sensor_payload import validate_native_static, decode_native_instances
from ..native_greenhouse_pair import assert_same_camera
from .contracts import (SAMPLE_SCHEMA, RESULT_SCHEMA, RESOLUTION, HEAD_CAMERA, ADMISSION,
                        require, pin, sha256, digest, read_json, bind_all, new_destination, write_new)
from .prepare import check_plan, protected_roots, case_plan


def collect(app, output, plan):
    """Prepared, same-donor original-only execution hook; no source rehash per frame."""
    import omni.replicator.core as rep
    from ..capture_scene import calibration
    from ..capture_pilot import make_writer, step_payload, source_hashes
    from ..capture_visibility import component_masks, interval_visibility, view_quality, write_visibility
    from ..static_guard import StaticSceneMonitor
    from ..static_geometry_cache import StaticGeometryScreenCache
    from ..native_clear_labels import derive
    from ..automated_native_review import trace_review
    from .scene import prepare_scene, set_case_pose

    started = time.perf_counter()
    context = prepare_scene(app, plan)
    stage, robot, catalogue = context["stage"], context["robot"], context["catalogue"]
    product = rep.create.render_product(HEAD_CAMERA, RESOLUTION)
    writer = None
    monitor = cache = None
    attached = False
    previous, records, seen_rgb = None, [], set()
    try:
        writer = make_writer(rep, include_instances=True, instance_backend="legacy")
        writer.attach([product])
        attached = True
        cache = StaticGeometryScreenCache(stage, robot["root"], include_generated_plants=False)
        for case in plan["cases"]:
            require(cache.diagnostics()["invalidating_notices"] == 0,
                    "Unmodified greenhouse changed between planned poses")
            case_started = time.perf_counter()
            current = case_plan(plan, case)
            name, row = case["case_id"], current["source_row"]
            try:
                pose, cal, world = set_case_pose(context, current)
                screen = cache()
                require(screen["passed"], "Snapshot failed static geometry screen")
                nominal = project([world["nominal_world_m"]], cal)[0]
                interval = project(world["interval_world_m"], cal)
                require(nominal["projection_status"] == "in_frame"
                        and all(p["projection_status"] == "in_frame" for p in interval), "Original interval is clipped/out of frame")
                if current["pose_request"]["mode"] == "reframe_head_only":
                    require(np.linalg.norm(np.asarray(nominal["pixel_xy"]) - current["pose_request"]["native_pixel_xy"]) < 4,
                            "Actual mounted framing differs from requested native pixel")
                # This is a bounded pre-render rejection, never visibility approval.
                require(previous is None or fingerprint(cal) != previous["camera_sha256"], "Repeated actual camera pose")
            except ValueError as exc:
                records.append(dict(case_id=name, target_id=row["target_id"], state="held_pre_render",
                                    reason=str(exc), admission=deepcopy(ADMISSION)))
                continue
            radius = row["cut_region_proposal"]["nominal"]["petiole_radius_m"]
            target = next(c for c in catalogue if c["variant_id"] == row["variant_id"] and c["component_id"] == row["component_id"])
            # Retain the established reference budget. No unqualified temporal shortcut.
            for _ in range(6):
                step_payload(rep, writer, subframes=8)
            if monitor is None:
                monitor = StaticSceneMonitor(stage, robot["root"])
            before = monitor.begin()
            payload = step_payload(rep, writer, subframes=8)
            rgb, depth, valid, reference, token = validate_native_static(
                payload, cal, before, monitor.token(), writer.sequence, previous=previous)
            require(cache.diagnostics()["invalidating_notices"] == 0,
                    "Non-robot scene changed during native warmup/capture")
            previous = token
            require(token["rgb_sha256"] not in seen_rgb, "Repeated native RGB within batch")
            seen_rgb.add(token["rgb_sha256"])
            assert_same_camera(calibration_for_native_resolution(calibration(stage), RESOLUTION), cal)
            require(context["settings"].get("/rtx/rendermode") == "RealTimePathTracing", "Renderer changed")
            instances, mapping = decode_native_instances(payload, RESOLUTION)
            components, organs, owners = component_masks(instances, mapping, catalogue)
            visibility, mask = interval_visibility(nominal, interval, depth, valid, instances, mapping, owners, target, radius)
            quality = view_quality(cal, nominal, interval, radius, visibility, rgb, mask)
            metadata = dict(schema_version=SAMPLE_SCHEMA, state="fresh_original_native_pending_automatic_audit",
                sample_id=name, training_sample_approved=False, sensor=sensor_profile(RESOLUTION),
                calibration=cal, robot_snapshot=pose, pose_request=current["pose_request"],
                geometry_screen=screen, scene_counts=context["counts"], lighting=context["lighting"],
                renderer="RealTimePathTracing", quality=quality, native_instance_backend="legacy",
                native_instance_sha256=digest(instances.tobytes()), native_mapping_sha256=fingerprint(mapping),
                input_policy=dict(clean_full_scene=True, isolation=False, diagnostic_overlays=False),
                pose_prior=current["pose_prior"], historical_labels_inherited=False,
                rendered_camera_params=jsonable(payload["camera_params"]),
                synchronization=dict(method="frozen_scene_single_native_writer_payload",
                    scene_unchanged_during_capture=True, dynamic_recording_supported=False,
                    reference_time=reference, freshness=token, static_guard=before,
                    native_render_frame=jsonable(payload["pilot_render_frame"]),
                    engine_frame_id_verified=False, render_budget_subframes=56),
                supervision=dict(**world, target_id=row["target_id"], source_target_id=row["target_id"],
                    split_group=plan["source_family"], conservative_view_cap_group=row["target_id"],
                    cut_region_proposal=deepcopy(row["cut_region_proposal"]),
                    nominal_projected=nominal, projected_interval=interval,
                    depth_evidence=depth_evidence(nominal, depth, valid, radius),
                    visibility_evidence=visibility, cut_safety_validated=False))
            label = derive(metadata, context["report"], rgb, depth, valid, components, catalogue)
            trace = trace_review(metadata, context["report"], label, rgb, depth, valid, components, catalogue) if label["eligible"] else None
            folder = Path(output) / name
            write_sample(folder, rgb, depth, valid, metadata)
            write_visibility(folder, instances, mapping, catalogue, components, organs, mask)
            write_new(folder / "supervision/label.json", label)
            if trace is not None:
                write_new(folder / "supervision/query_trace.json", trace)
            records.append(dict(case_id=name, target_id=row["target_id"], state="native_captured",
                sample_sha256=sha256(folder / "sample.json"), label_sha256=sha256(folder / "supervision/label.json"),
                trace_sha256=sha256(folder / "supervision/query_trace.json") if trace is not None else None,
                source_assets_unchanged=True, scene_counts=context["counts"], geometry_screen=screen,
                lighting=context["lighting"], elapsed_seconds=time.perf_counter()-case_started,
                label_reason=label["reason"], admission=deepcopy(ADMISSION)))
        require(source_hashes(stage) == context["source_hashes"], "Source USD changed during original batch")
        require(not any(not layer.anonymous and layer.dirty for layer in stage.GetUsedLayers()), "Source USD dirtied")
        bind_all(plan["source_bindings"])
        bind_all(plan["implementation_bindings"])
        return dict(records=records, source_assets_unchanged=True, elapsed_seconds=time.perf_counter()-started,
                    stage_count=1, greenhouse_render_product_count=1, geometry_cache=cache.diagnostics())
    finally:
        try:
            if monitor is not None:
                monitor.close()
        finally:
            try:
                if cache is not None:
                    cache.close()
            finally:
                try:
                    if attached:
                        writer.detach()
                finally:
                    product.destroy()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", required=True)
    p.add_argument("--plan-sha256", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args(argv)
    plan_path = pin(a.plan, a.plan_sha256)
    plan = read_json(plan_path)
    check_plan(plan)
    output = new_destination(a.output, [*protected_roots(plan), plan_path.parent])
    from sim_physics.host_memory import preflight
    from ..native_generated_pair import windows_worker_admission
    memory = preflight()
    require(memory["allowed"], "Native memory reserve unavailable; no override")
    admission = windows_worker_admission(None)
    bind_all(plan["implementation_bindings"])
    pin(plan_path, a.plan_sha256)
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "request.json", dict(schema=RESULT_SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
        plan_path=str(plan_path), plan_sha256=a.plan_sha256, host_memory_preflight=memory,
        process_admission=admission, automatic_retries=False, training_started=False))
    app = None
    succeeded = False
    try:
        from isaacsim import SimulationApp
        from ..native_resolution_smoke import smoke
        from ..generated_capture import check_smoke_metadata
        from .audit import audit_records, verify_smoke
        app = SimulationApp(dict(headless=True, width=1696, height=816, multi_gpu=False,
            renderer="RaytracedLighting", sync_loads=False, disable_viewport_updates=True,
            extra_args=["--/app/settings/persistent=false"]))
        smoke_output = output / "sensor_smoke"
        smoke_output.mkdir()
        smoke_result = smoke(app, smoke_output)
        check_smoke_metadata(smoke_result)
        write_new(smoke_output / "result.json", smoke_result)
        smoke_bindings = {f.relative_to(output).as_posix(): sha256(f) for f in smoke_output.rglob("*") if f.is_file()}
        verify_smoke(output, smoke_bindings)
        captured = collect(app, output, plan)
        automatic = audit_records(output, plan, captured["records"])
        write_new(output / "automatic_audit.json", automatic)
        result = dict(schema=RESULT_SCHEMA, state="original_native_capture_complete_automatically_audited",
            plan_sha256=a.plan_sha256, request_sha256=sha256(output / "request.json"),
            capture=captured, smoke_bindings=smoke_bindings,
            automatic_audit_sha256=sha256(output / "automatic_audit.json"), admission=deepcopy(ADMISSION),
            native_exit_status_must_be_checked_by_launcher=True)
        write_new(output / "result.json", result)
        succeeded = True
        print("ORIGINAL_NATIVE_COMPLETE", automatic["counts"], sha256(output / "result.json"), flush=True)
    except BaseException:
        write_new(output / "failure.json", dict(state="original_native_capture_failed",
            traceback=traceback.format_exc(), training_approved=False))
        raise
    finally:
        if app is not None:
            try:
                app.close(exit_code=0 if succeeded else 1)
            except BaseException:
                if not (output / "failure.json").exists():
                    write_new(output / "failure.json", dict(state="original_native_cleanup_failed",
                        traceback=traceback.format_exc(), training_approved=False))
                raise


if __name__ == "__main__":
    main()
