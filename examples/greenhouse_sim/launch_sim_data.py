"""Preview the supplied tomato greenhouse package in Isaac Sim 6.0.1.

Run with examples/greenhouse_sim/run_sim_data.cmd. Source assets are read-only;
plants, cameras and lighting edits are authored into the session layer.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import pathlib
import sys
import time
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PACK = ROOT / "data/sim_data/package_20260905/tomato_greenhouse_pack"


def populate(stage, package, app, review_records=None, review_plant=None):
    from pxr import Gf, Usd, UsdGeom

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    gutters = list(stage.GetPrimAtPath("/World/Gutters").GetChildren())
    centres = sorted(
        (float(cache.ComputeWorldBound(prim).ComputeAlignedRange().GetMidpoint()[0]), str(prim.GetPath()))
        for prim in gutters
    )
    centre = min(range(len(centres)), key=lambda i: abs(centres[i][0]))
    selected = centres[max(0, centre - 1):centre + 2]
    backdrop = sorted((package / "plants/backdrop").glob("backdrop_*.usd"))
    manifests = sorted((package / "plants/components").glob("*/manifest.json"))
    if not backdrop or len(manifests) < 2:
        raise RuntimeError("Package is missing backdrop plants or component manifests")
    if review_plant:
        selected_manifest = next((p for p in manifests if p.parent.name == review_plant), None)
        if selected_manifest is None:
            raise ValueError(f"Unknown review plant: {review_plant}")
        manifests = [selected_manifest] + [p for p in manifests if p != selected_manifest]

    def assemble(parent, manifest_path):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        components = manifest["components"]
        positions = {c["id"]: Gf.Vec3d(*c["transform"]["translate"]) for c in components}
        paths = {}
        by_id = {c["id"]: c for c in components}
        pending = set()

        def add(component_id):
            if component_id in paths:
                return paths[component_id]
            if component_id in pending:
                raise RuntimeError(f"Manifest parent cycle at {component_id}")
            pending.add(component_id)
            c = by_id[component_id]
            parent_id = c["parent"]
            parent_path = add(parent_id) if parent_id is not None else parent
            path = f"{parent_path}/{c['id']}"
            xform = UsdGeom.Xform.Define(stage, path)
            xform.GetPrim().GetReferences().AddReference((manifest_path.parent / c["file"]).as_posix())
            # This archive stores plant-frame positions. Convert them to local
            # offsets when reconstructing the manifest's parent hierarchy.
            offset = positions[c["id"]] - (positions[parent_id] if parent_id else Gf.Vec3d(0))
            xform.AddTranslateOp().Set(offset)
            paths[c["id"]] = path
            pending.remove(component_id)
            return path

        for c in components:
            add(c["id"])
        return len(components), paths

    UsdGeom.Xform.Define(stage, "/World/PackPlants")
    instances = detailed = parts = 0
    main_x = centres[centre][0]
    for gi, (cx, gutter_path) in enumerate(selected):
        for side in (-1, 1):
            for i in range(24):
                y = (i - 12) * 0.5
                path = f"/World/PackPlants/Gutter{gi}_{'L' if side < 0 else 'R'}_{i:03d}"
                xform = UsdGeom.Xform.Define(stage, path)
                xform.AddTranslateOp().Set(Gf.Vec3d(cx + side * 0.195, y, 0.90))
                if cx == main_x and side == 1 and i in (12, 13):
                    manifest_path = manifests[i - 12]
                    part_count, component_paths = assemble(path, manifest_path)
                    parts += part_count
                    if review_records is not None:
                        review_records.append({"plant_root": path, "manifest_path": str(manifest_path),
                                               "component_paths": component_paths})
                    detailed += 1
                else:
                    asset = backdrop[(i + gi * 7 + (5 if side > 0 else 0)) % len(backdrop)]
                    xform.GetPrim().GetReferences().AddReference(asset.as_posix())
                    xform.GetPrim().SetInstanceable(True)
                    instances += 1
            app.update()
        print(f"Populated {gutter_path}", flush=True)
    return main_x, {"gutters_in_asset": len(gutters), "populated_gutters": len(selected),
                    "backdrop_instances": instances, "component_plants": detailed, "components": parts}


def load_local_payloads(stage):
    from pxr import Sdf, Usd

    excluded = set()
    for prim in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(Usd.PrimAllPrimsPredicate)):
        payload = prim.GetMetadata("payload")
        if payload and any(item.assetPath.lower().startswith(("http:", "https:", "omniverse:"))
                           for item in payload.GetAppliedItems()):
            # Instance proxies cannot be edited; exclude their owning instance.
            while prim.IsInstanceProxy():
                prim = prim.GetParent()
            excluded.add(str(prim.GetPath()))
    with Sdf.ChangeBlock():
        for path in sorted(excluded):
            Sdf.CreatePrimInLayer(stage.GetSessionLayer(), path).active = False
    print(f"Excluded {len(excluded)} external prop roots; loading local payloads", flush=True)
    stage.Load()
    return sorted(excluded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=pathlib.Path, default=DEFAULT_PACK)
    parser.add_argument("--output", type=pathlib.Path, help="Run directory; review defaults to a fresh timestamped directory")
    parser.add_argument("--review", action="store_true", help="Open session-only Phase 1 anatomy review controls")
    parser.add_argument("--review-plant", help="Component plant directory to review first; implies --review")
    parser.add_argument("--review-history", type=pathlib.Path, action="append",
                        help="Additional anatomy_reviews directory to resume (repeatable)")
    parser.add_argument("--no-robot", action="store_true", help="Anatomy-only diagnostic preview")
    parser.add_argument("--right-tool", choices=("gripper", "knife_only"), default="gripper",
                        help="Annotation preview defaults to the original right gripper; does not change the physics workflow")
    parser.add_argument("--extra-cut-candidates", type=int, choices=range(4), default=0,
                        help="Add 0-3 attached lower leafy branches to EACH foreground plant (visual preview only)")
    parser.add_argument("--headless", action="store_true", help="Bounded camera smoke when used with --frames")
    parser.add_argument("--frames", type=int, default=0, help="Exit after this many ready-state updates (0: interactive)")
    args = parser.parse_args()
    if args.extra_cut_candidates and (args.review or args.review_plant):
        parser.error("Added branch variants are preview-only; annotation adapter is not implemented. Omit --review.")
    package = args.package.resolve()
    scene = package / "house/green_house_base.usd"
    if not scene.is_file():
        parser.error(f"Extract the package first; scene missing: {scene}")
    scene_sha256 = hashlib.sha256(scene.read_bytes()).hexdigest()
    if args.review_plant and not any(p.parent.name == args.review_plant
                                     for p in (package / "plants/components").glob("*/manifest.json")):
        parser.error("--review-plant must name an existing component plant directory")
    if args.output is None:
        name = ("phase1_review_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6]
                if args.review or args.review_plant else
                "candidate_preview_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6]
                if args.extra_cut_candidates else "preview_20260905")
        args.output = ROOT / "data/sim_data" / name
    if args.output.resolve().is_relative_to(package):
        parser.error("--output must be outside the source package")
    args.output = args.output.resolve()
    if not args.no_robot:
        from greenhouse_sim import robot_model
        if not robot_model.DEFAULT_ASSET.is_file():
            parser.error("Build Model A v1.2 first: D:/isaac-sim/python.bat examples/greenhouse_sim/build_robot.py (or use --no-robot)")
    args.output.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless, "width": 848, "height": 408,
                         "window_width": 1600, "window_height": 1000,
                         "renderer": "RaytracedLighting", "multi_gpu": False, "sync_loads": False})
    import carb
    import omni.ui as ui
    import omni.usd
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from pxr import Gf, UsdGeom

    settings = carb.settings.get_settings()
    settings.set_bool("/app/runLoops/main/rateLimitEnabled", True)
    settings.set_int("/app/runLoops/main/rateLimitFrequency", 60)
    context = omni.usd.get_context()
    print(f"Opening package: {scene}", flush=True)
    if not context.open_stage(str(scene), load_set=omni.usd.UsdContextInitialLoadSet.LOAD_NONE):
        raise RuntimeError(f"Cannot open {scene}")
    stage = context.get_stage()
    print("Base stage opened with payloads deferred", flush=True)
    stage.SetEditTarget(stage.GetSessionLayer())
    # The archive's optional props point to external assets not included in the
    # download. Disable only those payload prims before loading local payloads.
    excluded = load_local_payloads(stage)
    for _ in range(10):
        app.update()
    print(f"Local greenhouse loaded; {len(excluded)} unbundled prop payloads excluded", flush=True)
    review_records = []
    cx, counts = populate(stage, package, app, review_records, args.review_plant)
    branch_variants = []
    if args.extra_cut_candidates:
        from sim_data.candidate_branches import add_candidate_branches
        for record in review_records:
            branch_variants.append(add_candidate_branches(stage, record, args.extra_cut_candidates))
            app.update()
        (args.output / "candidate_branches.json").write_text(
            json.dumps(branch_variants, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Added {sum(len(v['candidates']) for v in branch_variants)} attached lower petiole candidates", flush=True)
    from sim_data.robot_preview import add_robot_preview, select_camera
    from sim_data.floor_alignment import PACKAGE_FLOOR
    robot = None if args.no_robot else add_robot_preview(stage, gutter_x=cx, floor_path=PACKAGE_FLOOR,
                                                       right_tool=args.right_tool)

    poses = {
        "Aisle view": ((cx + 0.8, -4.5, 1.9), (cx + 0.195, 0.0, 2.0)),
        "Plant close-up": ((cx + 0.8, -0.8, 1.55), (cx + 0.195, 0.0, 1.55)),
        "Greenhouse overview": ((cx + 7.5, -10.0, 7.5), (cx, 0.0, 1.8)),
    }
    if branch_variants:
        attachments = [c["attachment_world_m"] for c in branch_variants[0]["candidates"]]
        centre = tuple(sum(p[i] for p in attachments) / len(attachments) for i in range(3))
        poses["Added lower branches"] = ((centre[0] + 0.75, centre[1] - 0.5, centre[2] + 0.3),
                                        (centre[0] + 0.08, centre[1], centre[2]))
    cameras = {}
    for i, (name, (eye, target)) in enumerate(poses.items()):
        path = f"/World/PackCamera_{i}"
        camera = UsdGeom.Camera.Define(stage, path)
        camera.CreateFocalLengthAttr(20.0)
        camera.CreateClippingRangeAttr(Gf.Vec2f(0.03, 500.0))
        matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye), Gf.Vec3d(*target), Gf.Vec3d(0, 0, 1))
        UsdGeom.Xformable(camera).AddTransformOp().Set(matrix.GetInverse())
        cameras[name] = path
    viewport = get_active_viewport()
    if viewport is None:
        raise RuntimeError("No active viewport available")
    if robot is not None:
        cameras.update(robot["camera_paths"])
    select_camera(viewport, cameras["Robot head D405"] if robot else cameras["Aisle view"])

    sys.path.insert(0, str(package / "env_panel"))
    from tomato_env import panel

    lighting = panel.Panel()
    lighting.sun_cd.set_value(1500)
    lighting.dome_cd.set_value(1200)
    controls = ui.Window("Tomato package preview", width=380, height=370)
    with controls.frame:
        with ui.VStack(spacing=6):
            ui.Label("Latest supplied greenhouse / session preview", word_wrap=True)
            ui.Label(f"{counts['backdrop_instances']} instanced + {counts['component_plants']} detailed plants")
            if branch_variants:
                ui.Label(f"{len(branch_variants) * args.extra_cut_candidates} added lower leafy candidates; attached, unreviewed, visual-only.",
                         word_wrap=True)
            for name, camera_path in cameras.items():
                ui.Button(name, clicked_fn=lambda p=camera_path: select_camera(viewport, p))
            ui.Label(f"v1.2 robot + D405 RGB preview (848x408). Right tool: {args.right_tool}. Wrist adapters are simulator designs. Cutting, grasping and dynamics are not enabled in this package preview.",
                     word_wrap=True)
    # Dock controls so they do not cover the plant view.
    property_window = ui.Workspace.get_window("Property")
    if property_window:
        controls.dock_in(property_window, ui.DockPosition.BOTTOM, 0.4)
        lighting.window.dock_in(property_window, ui.DockPosition.SAME)

    report = {"scene": str(scene), "package": str(package), "state": "warming_up",
              "scene_sha256_at_launch": scene_sha256,
              **counts, "unbundled_payloads_excluded": excluded, "source_assets_modified": False,
              "physics_enabled": False, "robot_loaded": robot is not None, "robot": robot, "camera_paths": cameras,
              "anatomy_review_requested": bool(args.review or args.review_plant)}
    report["extra_cut_candidates_per_plant"] = args.extra_cut_candidates
    report["added_branch_candidates"] = sum(len(v["candidates"]) for v in branch_variants)
    report["added_branch_components"] = sum(len(v["added_components"]) for v in branch_variants)
    report["candidate_variant_ids"] = [v["variant_id"] for v in branch_variants]
    if branch_variants:
        report["candidate_branch_provenance"] = str(args.output / "candidate_branches.json")
    report_path = args.output / "status.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Capture the actual viewport without adding a persistent render product.
    for _ in range(90):
        app.update()
    capture = capture_viewport_to_file(viewport, str(args.output / "preview.png"))
    pending_capture = asyncio.ensure_future(capture.wait_for_result())
    deadline = time.monotonic() + 20
    while not pending_capture.done():
        app.update()
        if time.monotonic() > deadline:
            raise RuntimeError("Preview capture timed out")
    pending_capture.result()
    from PIL import Image
    while True:
        try:
            with Image.open(args.output / "preview.png") as captured:
                report["preview_resolution"] = list(captured.size)
                captured.verify()
            break
        except (OSError, SyntaxError):
            if time.monotonic() > deadline:
                raise RuntimeError("Preview PNG encoding did not complete")
            app.update()
    if branch_variants:
        # Diagnostic evidence of the additions; preserve the mounted head view.
        initial_camera = str(viewport.camera_path)
        select_camera(viewport, cameras["Added lower branches"])
        for _ in range(45):
            app.update()
        branch_image = args.output / "added_lower_branches.png"
        capture = capture_viewport_to_file(viewport, str(branch_image))
        pending_capture = asyncio.ensure_future(capture.wait_for_result())
        deadline = time.monotonic() + 20
        while not pending_capture.done():
            app.update()
            if time.monotonic() > deadline:
                raise RuntimeError("Added branch capture timed out")
        pending_capture.result()
        while True:
            try:
                with Image.open(branch_image) as captured:
                    captured.verify()
                break
            except (OSError, SyntaxError):
                if time.monotonic() > deadline:
                    raise RuntimeError("Added branch PNG encoding did not complete")
                app.update()
        report["added_branch_diagnostic_image"] = str(branch_image)
        select_camera(viewport, initial_camera)
    if hashlib.sha256(scene.read_bytes()).hexdigest() != scene_sha256:
        raise RuntimeError("Source greenhouse changed during launch; inspect before continuing")
    review_panel = None
    if args.review or args.review_plant:
        import omni.timeline
        from sim_data.gallery import BatchReviewPanel

        omni.timeline.get_timeline_interface().pause()
        history = list((ROOT / "data/sim_data").glob("*/anatomy_reviews")) + (args.review_history or [])
        review_panel = BatchReviewPanel(stage, review_records, viewport, args.output / "anatomy_reviews",
                                       package=package, history_directories=history)
        report["review_images_are_not_training_inputs"] = True
        report["review_workflow"] = "batch_gallery_and_exceptions"
        report["review_catalog_plants"] = len(review_panel.reports)
    report["state"] = "running"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PACKAGE_READY " + json.dumps(report), flush=True)
    heartbeat = time.monotonic()
    ready_frames = 0
    while app.is_running() and (args.frames <= 0 or ready_frames < args.frames):
        app.update()
        ready_frames += 1
        if time.monotonic() - heartbeat > 10:
            print("PACKAGE_HEARTBEAT", flush=True)
            heartbeat = time.monotonic()
    if review_panel is not None:
        review_panel.close()
    report["state"] = "completed" if args.frames > 0 else "closed"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
