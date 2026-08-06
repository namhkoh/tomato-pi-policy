"""Demonstrate and measure vine cutting physics on a single plant.

Run from the repository root with Isaac Sim's bundled interpreter. $ISAACSIM
stands for your Isaac Sim install; on Windows use %ISAACSIM%\\python.bat:

    $ISAACSIM/python.sh examples/greenhouse_sim/cut_demo.py

Rigs one vine as compliant capsule chains, lets it settle under gravity, cuts a
petiole, and reports whether the severed leaf actually fell away while the rest
of the plant stayed put. This is the check that the severance mechanism works
before any robot is introduced.
"""

# SimulationApp must exist before anything from omni or pxr is imported.
# ruff: noqa: PLC0415

from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))

_DEFAULT_VINE = pathlib.Path("greenhouse/tomato_glb_20/tomato_000.glb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vine", type=pathlib.Path, default=_DEFAULT_VINE)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--gui", dest="headless", action="store_false")
    parser.add_argument("--settle-steps", type=int, default=180, help="steps before cutting")
    parser.add_argument("--fall-steps", type=int, default=180, help="steps after cutting")
    parser.add_argument("--stub", type=float, default=0.0, help="stub length to leave, in metres")
    parser.add_argument("--organ", type=str, default=None, help="sub-stem label to cut")
    parser.add_argument("--segment", type=float, default=0.02, help="capsule length, in metres")
    # Kit floods stdout, so the machine-readable result goes to a file.
    parser.add_argument("--report", type=pathlib.Path, default=pathlib.Path("data/greenhouse_sim/cut_report.json"))
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

    report: dict = {"vine": str(args.vine), "stage": "starting"}
    _emit(report, args.report)

    from isaacsim import SimulationApp

    app = SimulationApp({"headless": args.headless})

    from greenhouse_sim import cutting
    from greenhouse_sim import organs
    from greenhouse_sim import skeleton as skeleton_module
    from greenhouse_sim import vine_physics
    from greenhouse_sim import vine_usd
    import numpy as np
    from pxr import UsdGeom

    print(f"segmenting {args.vine.name} ...")
    plant = organs.load_plant(args.vine)
    skeletons = skeleton_module.skeletonise_plant(plant, segment_length=args.segment)
    print(f"  {len(plant.organs)} organs, {len(skeletons)} skeletonised")
    report.update(organs=len(plant.organs), skeletonised=len(skeletons), stage="segmented")
    _emit(report, args.report)

    # Author straight into Kit's own stage; attaching an in-memory one would
    # require registering it with the stage cache for no benefit here.
    import omni.usd

    omni.usd.get_context().new_stage()
    stage = omni.usd.get_context().get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(stage.GetPrimAtPath("/World"))
    vine_physics.apply_scene_physics(stage)

    rig = vine_physics.author_plant_physics(stage, plant, "/World/Vine", skeletons, vine_usd.gltf_to_usd)
    print(f"  {len(rig.links)} capsule links, {len(rig.joints)} joints, {len(rig.cut_joints)} severable organs")
    report.update(links=len(rig.links), joints=len(rig.joints), severable=len(rig.cut_joints), stage="rigged")
    _emit(report, args.report)

    for _ in range(10):
        app.update()

    from isaacsim.core.api import SimulationContext

    context = SimulationContext(physics_dt=1.0 / 240.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
    context.initialize_physics()
    context.play()

    organ_indices = {organ.label: organ.index for organ in plant.organs}
    severer = cutting.Severer(stage, rig, skeletons, organ_indices)

    target = args.organ or _pick_lowest_substem(plant)
    if target not in rig.cut_joints:
        report.update(stage="failed", error=f"{target} not severable", available=sorted(rig.cut_joints)[:8])
        _emit(report, args.report)
        app.close()
        return 1

    tracked = rig.link_paths_for(organ_indices[target])
    print(f"\nsettling for {args.settle_steps} steps ...")
    for _ in range(args.settle_steps):
        context.step(render=not args.headless)

    settled = _positions(stage, tracked)
    reference = _positions(stage, _reference_links(rig, organ_indices[target]))
    print(f"  target organ settled at z={np.nanmean(settled[:, 2]):.4f} m")
    report.update(target=target, settled_z=float(np.nanmean(settled[:, 2])), stage="settled")
    _emit(report, args.report)

    # Mutating joint state while PhysX is mid-step is not safe; pause first.
    report.update(stage="pausing")
    _emit(report, args.report)
    context.pause()
    report.update(stage="paused", joint=rig.cut_joints[target])
    _emit(report, args.report)
    try:
        record = severer.cut(target, stub_length_m=args.stub)
        report.update(stage="cut_applied")
        _emit(report, args.report)
    except Exception:
        import traceback

        report.update(stage="failed", error=traceback.format_exc())
        _emit(report, args.report)
        app.close()
        return 1
    print(
        f"\ncut {record.organ_label}: {record.grade} "
        f"(requested stub {record.requested_stub_m * 1000:.1f} mm, "
        f"realised {record.realised_stub_m * 1000:.1f} mm, "
        f"quantisation {record.quantisation_error_m * 1000:.1f} mm)"
    )

    report.update(stage="resuming")
    _emit(report, args.report)
    context.play()
    report.update(stage="resumed")
    _emit(report, args.report)
    for step in range(args.fall_steps):
        context.step(render=not args.headless)
        if step % 30 == 0:
            report.update(stage=f"falling_{step}")
            _emit(report, args.report)

    report.update(stage="fell")
    _emit(report, args.report)
    try:
        fallen = _positions(stage, tracked)
        rest = _positions(stage, _reference_links(rig, organ_indices[target]))
        drop = float(np.nanmean(settled[:, 2]) - np.nanmean(fallen[:, 2]))
        disturbance = float(np.nanmax(np.abs(rest - reference))) if reference.size else 0.0
    except Exception:
        import traceback

        report.update(stage="failed_measuring", error=traceback.format_exc())
        _emit(report, args.report)
        app.close()
        return 1

    print(f"\nsevered organ fell {drop * 1000:.1f} mm")
    print(f"rest of plant moved at most {disturbance * 1000:.1f} mm")

    # A cut only counts if the organ actually came away and the rest of the
    # plant stayed put; either alone would pass while the mechanism is wrong.
    succeeded = bool(drop > 0.005 and disturbance < drop)
    report.update(
        stage="done",
        grade=record.grade,
        requested_stub_mm=record.requested_stub_m * 1000.0,
        realised_stub_mm=record.realised_stub_m * 1000.0,
        quantisation_mm=record.quantisation_error_m * 1000.0,
        drop_mm=drop * 1000.0,
        disturbance_mm=disturbance * 1000.0,
        summary=cutting.summarise(severer.cuts),
        succeeded=succeeded,
    )
    _emit(report, args.report)
    print("\nRESULT:", "cut mechanism works" if succeeded else "FAILED: severed organ did not come away")

    context.stop()
    app.close()
    return 0 if succeeded else 1


def _pick_lowest_substem(plant) -> str:
    """Lowest sub-stem: the one a bottom-up deleafing pass removes first."""
    labelled = [o for o in plant.labelled("SubStem") if o.attachment is not None]
    if not labelled:
        return ""
    # Organ attachments are in the GLB's Y-up frame; height is component 1.
    return min(labelled, key=lambda o: float(o.attachment[1])).label


def _reference_links(rig, organ_index: int) -> list[str]:
    """Links belonging to organs other than the one being cut."""
    return [link.path for link in rig.links if link.organ != organ_index][:64]


def _positions(stage, paths: list[str]):
    import numpy as np
    from pxr import Usd
    from pxr import UsdGeom

    if not paths:
        return np.zeros((0, 3))
    # One row per requested path, NaN where a prim is missing, so that arrays
    # sampled before and after a cut always align for comparison.
    rows = []
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            rows.append([np.nan] * 3)
            continue
        matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rows.append(list(matrix.ExtractTranslation()))
    return np.array(rows)


if __name__ == "__main__":
    raise SystemExit(main())
