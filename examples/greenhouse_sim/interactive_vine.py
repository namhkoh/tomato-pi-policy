"""Open a physics-enabled vine you can grab and pull with the mouse.

Run from the repository root with Isaac Sim's bundled interpreter. $ISAACSIM
stands for your Isaac Sim install; on Windows use %ISAACSIM%\\python.bat:

    $ISAACSIM/python.sh examples/greenhouse_sim/interactive_vine.py

The simulation starts playing immediately, so in the viewport you can
**Shift + left-click-drag any part of the plant** to apply a force to it: pull a
petiole, bend the stem, tug a truss. That is the fastest way to judge whether
the compliance feels like a real plant, which no scalar metric really answers.

Useful flags:

    --plants 3          more vines, spaced along a row
    --no-clips          omit the trellis, to watch the vine collapse without it
    --cut SubStem_00    sever one petiole a second after start-up
    --screenshot p.png  render one frame headless instead of opening a window
"""

# SimulationApp must exist before anything from omni or pxr is imported.
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

_DEFAULT_VINE = pathlib.Path("greenhouse/tomato_glb_20/tomato_000.glb")
_ROW_SPACING_M = 0.45


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vine", type=pathlib.Path, default=_DEFAULT_VINE)
    parser.add_argument("--plants", type=int, default=1)
    parser.add_argument("--segment", type=float, default=0.02, help="capsule length, in metres")
    parser.add_argument("--clip-spacing", type=float, default=0.30)
    parser.add_argument("--gravity", type=float, default=9.81, help="0 isolates constraint problems from load")
    parser.add_argument("--no-collision", dest="collide", action="store_false", default=True,
                        help="drop colliders, to separate contact problems from constraint problems")
    parser.add_argument(
        "--tear-force",
        type=float,
        default=0.0,
        help="petiole break force in N; 0 disables tearing (transients at start-up trip a low threshold)",
    )
    parser.add_argument("--no-clips", dest="clips", action="store_false", default=True)
    parser.add_argument(
        "--show-colliders", action="store_true", help="draw the capsule bodies instead of hiding them under the art"
    )
    parser.add_argument("--cut", type=str, default=None, help="sever this organ shortly after start-up")
    parser.add_argument("--screenshot", type=pathlib.Path, default=None, help="render one frame and exit")
    parser.add_argument("--settle-steps", type=int, default=240, help="steps before the screenshot")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--report", type=pathlib.Path, default=pathlib.Path("data/greenhouse_sim/interactive.json"))
    return parser.parse_args()


def _emit(report: dict, path: pathlib.Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if not args.vine.exists():
        print(f"vine asset not found: {args.vine}")
        return 1

    report: dict = {"stage": "starting", "vine": str(args.vine)}
    _emit(report, args.report)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.screenshot is not None})

    from greenhouse_sim import cutting
    from greenhouse_sim import glb
    from greenhouse_sim import organs
    from greenhouse_sim import skeleton as skeleton_module
    from greenhouse_sim import vine_physics
    from greenhouse_sim import vine_usd
    from greenhouse_sim import vine_visuals
    import numpy as np
    import omni.usd
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom

    plant = organs.load_plant(args.vine)
    asset = glb.read_glb(args.vine)
    skeletons = skeleton_module.skeletonise_plant(plant, segment_length=args.segment)

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    vine_physics.apply_scene_physics(stage, gravity=args.gravity)
    UsdGeom.Xform.Define(stage, "/World/Row")

    rigs, all_clips = [], []
    for index in range(max(1, args.plants)):
        root = f"/World/Row/Vine_{index:02d}"
        holder = UsdGeom.Xform.Define(stage, Sdf.Path(root))
        holder.AddTranslateOp().Set(Gf.Vec3d(index * _ROW_SPACING_M, 0.0, 0.0))
        # Author in world coordinates offset by the row position, since the
        # bodies are maximal-coordinate and do not inherit the holder xform.
        offset = np.array([index * _ROW_SPACING_M, 0.0, 0.0])
        rig = vine_physics.author_plant_physics(
            stage,
            plant,
            root,
            skeletons,
            lambda p, o=offset: vine_usd.gltf_to_usd(p) + o,
            properties=vine_physics.TissueProperties(tear_force_n=args.tear_force),
            visible_colliders=args.show_colliders,
            collidable=args.collide,
        )
        vine_visuals.attach_organ_visuals(
            stage, rig, plant, asset, lambda p, o=offset: vine_usd.gltf_to_usd(p) + o
        )
        rigs.append(rig)
        if args.clips:
            all_clips += vine_physics.add_trellis_clips(stage, rig, plant.root, spacing=args.clip_spacing)

    floor = min(link.start[2] for rig in rigs for link in rig.links)
    vine_physics.add_ground_plane(stage, height=floor)

    _add_lighting(stage)
    report.update(
        stage="rigged",
        plants=len(rigs),
        links=sum(len(r.links) for r in rigs),
        clips=len(all_clips),
        severable=len(rigs[0].cut_joints),
    )
    _emit(report, args.report)

    for _ in range(10):
        app.update()

    from isaacsim.core.api import SimulationContext

    context = SimulationContext(physics_dt=1.0 / 240.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
    context.initialize_physics()

    stem_paths = [link.path for link in rigs[0].links if link.organ == plant.root]
    rest = _sample(stage, stem_paths)
    # Every organ's base body, to find which parts of the plant fly away.
    base_links = {}
    for link in rigs[0].links:
        current = base_links.get(link.organ)
        if current is None or link.index < current.index:
            base_links[link.organ] = link
    organ_paths = [base_links[i].path for i in sorted(base_links)]
    organ_rest = _sample(stage, organ_paths)

    context.play()

    if args.cut:
        organ_indices = {organ.label: organ.index for organ in plant.organs}
        severer = cutting.Severer(stage, rigs[0], skeletons, organ_indices)
        for _ in range(120):
            context.step(render=args.screenshot is None)
        context.pause()
        if args.cut in rigs[0].cut_joints:
            severer.cut(args.cut)
            report["cut"] = args.cut
        context.play()

    if args.screenshot is None:
        print("simulation running -- Shift + left-click-drag in the viewport to pull the plant")
        while app.is_running():
            context.step(render=True)
        context.stop()
        app.close()
        return 0

    for _ in range(args.settle_steps):
        context.step(render=False)

    settled = _sample(stage, stem_paths)
    drop = _sag(rest, settled, stem_paths, report)

    organ_now = _sample(stage, organ_paths)
    moved = np.linalg.norm(organ_now - organ_rest, axis=1)
    labels = {o.index: o.label for o in plant.organs}
    order = sorted(base_links)
    link_counts = {}
    for link in rigs[0].links:
        link_counts[link.organ] = link_counts.get(link.organ, 0) + 1
    runaways = [
        {
            "organ": labels.get(order[i], str(order[i])),
            "moved_mm": round(float(moved[i]) * 1000, 1),
            "nlinks": link_counts.get(order[i], 0),
        }
        for i in np.argsort(-moved)[:10]
        if moved[i] > 0.1
    ]
    single = [i for i, o in enumerate(order) if link_counts.get(o, 0) == 1]
    multi = [i for i, o in enumerate(order) if link_counts.get(o, 0) > 1]
    report["runaway_single_link"] = f"{int((moved[single] > 0.1).sum())}/{len(single)}"
    report["runaway_multi_link"] = f"{int((moved[multi] > 0.1).sum())}/{len(multi)}"
    report["organs_tracked"] = len(order)
    report["organs_runaway"] = int((moved > 0.1).sum())
    report["worst_organs"] = runaways
    _emit(report, args.report)
    print(f"stem sag {drop:.1f} mm")

    _capture(stage, args, rigs, floor)
    report["stage"] = "done"
    _emit(report, args.report)
    context.stop()
    app.close()
    return 0


def _sag(rest, settled, paths, report: dict) -> float:
    """Displacement per link, so a collapse can be located rather than guessed."""
    import numpy as np

    if not rest.size:
        return 0.0
    delta = np.linalg.norm(settled - rest, axis=1)
    order = np.argsort(rest[:, 2])
    profile = [
        {"height_m": round(float(rest[i, 2]), 3), "moved_mm": round(float(delta[i]) * 1000.0, 1)}
        for i in order[:: max(1, len(order) // 12)]
    ]
    report["sag_profile"] = profile
    report["stem_sag_mm"] = round(float(np.nanmax(delta)) * 1000.0, 1)
    report["worst_link"] = paths[int(np.nanargmax(delta))]
    return float(np.nanmax(delta)) * 1000.0


def _sample(stage, paths):
    import numpy as np
    from pxr import Usd
    from pxr import UsdGeom

    rows = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            rows.append([np.nan] * 3)
            continue
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rows.append(list(matrix.ExtractTranslation()))
    return np.array(rows) if rows else np.zeros((0, 3))


def _add_lighting(stage) -> None:
    from pxr import Sdf
    from pxr import UsdLux

    dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/Lights/Dome"))
    dome.CreateIntensityAttr(1200.0)
    key = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/Lights/Key"))
    key.CreateIntensityAttr(2500.0)
    key.CreateAngleAttr(1.0)


def _capture(stage, args, rigs, floor: float) -> None:
    import numpy as np
    import omni.replicator.core as rep
    from pxr import Gf
    from pxr import Sdf
    from pxr import UsdGeom

    points = np.array([link.start for rig in rigs for link in rig.links])
    centre = points.mean(axis=0)
    top = float(points[:, 2].max())

    camera_path = Sdf.Path("/World/InspectionCamera")
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(24.0)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    eye = Gf.Vec3d(float(centre[0]) + 0.9, float(centre[1]) - 2.2, floor + 0.55 * (top - floor) + 0.4)
    target = Gf.Vec3d(float(centre[0]), float(centre[1]), floor + 0.5 * (top - floor))
    view = Gf.Matrix4d().SetLookAt(eye, target, Gf.Vec3d(0, 0, 1))
    UsdGeom.Xformable(camera.GetPrim()).AddTransformOp().Set(view.GetInverse())

    args.screenshot.parent.mkdir(parents=True, exist_ok=True)
    product = rep.create.render_product(str(camera_path), (args.width, args.height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([product])
    for _ in range(40):
        rep.orchestrator.step(rt_subframes=4)
    _write_png(annotator.get_data(), args.screenshot)


def _write_png(rgba, path: pathlib.Path) -> None:
    import numpy as np
    from PIL import Image

    Image.fromarray(np.asarray(rgba)[:, :, :3].astype(np.uint8)).save(path)


if __name__ == "__main__":
    raise SystemExit(main())
