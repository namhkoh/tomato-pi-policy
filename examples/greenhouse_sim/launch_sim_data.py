"""Preview the supplied tomato greenhouse package in Isaac Sim 6.0.1.

Run with examples/greenhouse_sim/run_sim_data.cmd. Source assets are read-only;
plants, cameras and lighting edits are authored into the session layer.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PACK = ROOT / "data/sim_data/package_20260905/tomato_greenhouse_pack"


def populate(stage, package, app):
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
        return len(components)

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
                    parts += assemble(path, manifests[i - 12])
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
    parser.add_argument("--output", type=pathlib.Path, default=ROOT / "data/sim_data/preview_20260905")
    args = parser.parse_args()
    package = args.package.resolve()
    scene = package / "house/green_house_base.usd"
    if not scene.is_file():
        parser.error(f"Extract the package first; scene missing: {scene}")
    args.output.mkdir(parents=True, exist_ok=True)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": False, "width": 1280, "height": 720,
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
    cx, counts = populate(stage, package, app)

    poses = {
        "Aisle view": ((cx + 0.8, -4.5, 1.9), (cx + 0.195, 0.0, 2.0)),
        "Plant close-up": ((cx + 0.8, -0.8, 1.55), (cx + 0.195, 0.0, 1.55)),
        "Greenhouse overview": ((cx + 7.5, -10.0, 7.5), (cx, 0.0, 1.8)),
    }
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
    viewport.set_texture_resolution((1280, 720))
    viewport.set_active_camera(cameras["Aisle view"])

    sys.path.insert(0, str(package / "env_panel"))
    from tomato_env import panel

    lighting = panel.Panel()
    lighting.sun_cd.set_value(1500)
    lighting.dome_cd.set_value(1200)
    controls = ui.Window("Tomato package preview", width=360, height=230)
    with controls.frame:
        with ui.VStack(spacing=6):
            ui.Label("Latest supplied greenhouse / session preview", word_wrap=True)
            ui.Label(f"{counts['backdrop_instances']} instanced + {counts['component_plants']} detailed plants")
            for name, camera_path in cameras.items():
                ui.Button(name, clicked_fn=lambda p=camera_path: viewport.set_active_camera(p))
            ui.Label("Geometry and lighting preview. Robot, cutting and physics are not attached to this package yet.",
                     word_wrap=True)
    # Dock controls so they do not cover the plant view.
    property_window = ui.Workspace.get_window("Property")
    if property_window:
        controls.dock_in(property_window, ui.DockPosition.BOTTOM, 0.4)
        lighting.window.dock_in(property_window, ui.DockPosition.SAME)

    report = {"scene": str(scene), "package": str(package), "state": "warming_up",
              **counts, "unbundled_payloads_excluded": excluded, "source_assets_modified": False,
              "physics_enabled": False, "robot_loaded": False, "camera_paths": cameras}
    report_path = args.output / "status.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Capture the actual viewport without adding a persistent render product.
    for _ in range(90):
        app.update()
    capture = capture_viewport_to_file(viewport, str(args.output / "preview.png"))
    for _ in range(20):
        app.update()
    report["state"] = "running"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PACKAGE_READY " + json.dumps(report), flush=True)
    heartbeat = time.monotonic()
    while app.is_running():
        app.update()
        if time.monotonic() - heartbeat > 10:
            print("PACKAGE_HEARTBEAT", flush=True)
            heartbeat = time.monotonic()
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
