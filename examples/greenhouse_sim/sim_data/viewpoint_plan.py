"""CPU-only, bounded robot-base/head pose planning before expensive native capture.

Geometry proposes views; rendered visibility and explicit visual review still
decide their usefulness. No free camera, robot control or navigation claim.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from .audit import DEFAULT_PACK
from .capture_contract import RESOLUTION, project, jsonable
from .capture_pilot import load_drafts, source_hashes
from .depth_preview import sha256

SCHEMA = "greenhouse.bounded_robot_view_plan.v1"


def focus_specs(original_x, target_x):
    if not np.isfinite([original_x, target_x]).all() or original_x <= target_x:
        raise ValueError("Expected original +X robot aisle")
    result = []
    # Same side; 40 cm maximum approach. Base yaw is an actual whole-robot pose.
    for approach in (.15, .20, .25, .30, .35, .40):
        x = original_x - approach
        if x - target_x < .35:
            continue
        for offset in (-.4, -.2, 0., .2, .4):
            for yaw in (150., 180., 210.):
                i = len(result)
                fraction = ((.38, .46), (.57, .57), (.44, .62))[(i//3) % 3]
                result.append({"candidate_id": f"focus_{i+1:03d}", "root_x_m": float(x),
                    "approach_from_original_m": approach, "y_offset_m": offset,
                    "root_yaw_degrees": yaw,
                    "desired_pixel_xy": [fraction[0]*RESOLUTION[0], fraction[1]*RESOLUTION[1]]})
    return result


def select_planned(records, count=6):
    """Keep projected geometric quality and base/yaw diversity; visibility unknown."""
    remaining = sorted(records, key=lambda r: (r["predicted_diameter_px"] * r["predicted_interval_px"],
                                                r["predicted_diameter_px"]), reverse=True)
    selected = []
    while remaining and len(selected) < count:
        diverse = [r for r in remaining if all(
            np.linalg.norm(np.asarray(r["base_xy_m"]) - s["base_xy_m"]) >= .12-1e-8
            or abs(r["root_yaw_degrees"]-s["root_yaw_degrees"]) >= 30 for s in selected)]
        chosen = (diverse or remaining)[0]
        selected.append(chosen)
        remaining = [r for r in remaining if r["candidate_id"] != chosen["candidate_id"]]
    return selected


def load_plan(path, drafts, package):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if (plan.get("schema_version") != SCHEMA or plan.get("state") != "ready_for_rendered_visibility_check"
            or plan.get("draft_sha256") != sha256(drafts)
            or Path(plan.get("package", "")).resolve() != Path(package).resolve()):
        raise ValueError("Stale/incompatible bounded view plan")
    for name, expected in plan["source_usd_sha256"].items():
        if sha256(name) != expected:
            raise ValueError("View plan source asset changed")
    if not plan.get("selected") or set(plan["selected"]) - {"B03", "B06"}:
        raise ValueError("Expected focused B03/B06 view plan")
    for target, records in plan["selected"].items():
        expected_specs = {r["candidate_id"]: r for r in focus_specs(plan["original_base_x_m"], plan["target_world_m"][target][0])}
        if not records or len(records) > 6 or len({r["candidate_id"] for r in records}) != len(records):
            raise ValueError("Invalid focused candidate count")
        for row in records:
            expected = expected_specs.get(row["candidate_id"])
            if (expected is None or any(row.get(k) != v for k, v in expected.items())
                    or row.get("state") != "geometry_screen_passed_visibility_unknown"
                    or not row["visual_bound_screen"]["passed"]):
                raise ValueError("Plan contains altered or unscreened candidate")
    return plan


def make_plan(drafts, package, output, per_target=6):
    # Loading/validating a plan happens BEFORE SimulationApp startup. Keep all
    # USD imports lazy so standalone OpenUSD cannot poison Kit's bindings.
    from pxr import Usd, UsdGeom
    from .capture_scene import (calibration, capture_root, freeze_rigid_bodies,
                                set_snapshot_pose, target_world_geometry)
    from .capture_viewpoints import screen_bounds, static_obstacles, visible_bounds, scene_triangle_refiner
    from launch_sim_data import load_local_payloads, populate
    from .candidate_branches import add_candidate_branches
    from .floor_alignment import PACKAGE_FLOOR
    from .robot_preview import add_robot_preview
    output, package, drafts = Path(output).resolve(), Path(package).resolve(), Path(drafts).resolve()
    if output.exists() or output.is_relative_to(package):
        raise ValueError("Choose a NEW plan directory outside source assets")
    saved, reports, rows = load_drafts(drafts)
    root = capture_root(package / "house/green_house_base.usd")
    stage = Usd.Stage.Open(root, load=Usd.Stage.LoadNone)
    UsdGeom.SetStageMetersPerUnit(stage, 1); UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetEditTarget(stage.GetSessionLayer())
    load_local_payloads(stage)
    class NoRenderer:
        def update(self): pass
    records = []
    gutter_x, counts = populate(stage, package, NoRenderer(), records)
    variants = [add_candidate_branches(stage, r, 3) for r in records]
    robot = add_robot_preview(stage, gutter_x=gutter_x, floor_path=PACKAGE_FLOOR, right_tool="gripper")
    freeze_rigid_bodies(stage)
    initial_hashes = source_hashes(stage)
    original_x = float(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(robot["root"]))[3][0])
    obstacles, refine = static_obstacles(stage, robot["root"]), scene_triangle_refiner(stage)
    by_variant = {v["variant_id"]: v for v in variants}
    result = {"schema_version": SCHEMA, "state": "planning", "package": str(package),
        "draft_sha256": sha256(drafts), "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_base_x_m": original_x, "scene_counts": counts, "source_usd_sha256": initial_hashes,
        "target_world_m": {}, "decisions": [], "selected": {}, "rendered_visibility_checked": False,
        "camera_mount_and_resolution_unchanged": True, "arm_and_torso_joints_unchanged": True,
        "navigation_validated": False, "training_approved": False}
    for row in rows:
        target_id = row["draft_id"]
        if target_id not in {"B03", "B06"}:
            continue
        world = target_world_geometry(stage, by_variant[row["variant_id"]], row)
        result["target_world_m"][target_id] = world["nominal_world_m"]
        viable = []
        for spec in focus_specs(original_x, world["nominal_world_m"][0]):
            decision = {"target_review_id": target_id, **spec}
            result["decisions"].append(decision)
            try:
                pose = set_snapshot_pose(stage, robot, world["nominal_world_m"], spec["y_offset_m"],
                    spec["desired_pixel_xy"], root_x_m=spec["root_x_m"], root_yaw_degrees=spec["root_yaw_degrees"])
                cal = calibration(stage)
                nominal, interval = project([world["nominal_world_m"]], cal)[0], project(world["interval_world_m"], cal)
                diameter = 2 * row["cut_region_proposal"]["nominal"]["petiole_radius_m"] * cal["intrinsics"][0][0] / nominal["camera_optical_xyz_m"][2]
                length = float(np.linalg.norm(np.diff([p["pixel_xy"] for p in interval], axis=0), axis=1).sum())
                decision.update(predicted_diameter_px=diameter, predicted_interval_px=length,
                    base_xy_m=np.asarray(pose["robot_root_to_world_usd_row_vectors"])[3, :2].tolist())
                if diameter < 3 or length < 5 or not all(p["projection_status"] == "in_frame" for p in interval):
                    decision["state"] = "rejected_projected_sampling"
                    continue
                screen = screen_bounds(visible_bounds(stage, robot["root"]), obstacles, refinement=refine)
                decision["visual_bound_screen"] = screen
                if not screen["passed"]:
                    decision["state"] = "rejected_possible_geometry_overlap"
                    continue
                decision["state"] = "geometry_screen_passed_visibility_unknown"
                viable.append(decision)
            except ValueError as exc:
                decision.update(state="rejected_pose", reason=str(exc))
        result["selected"][target_id] = select_planned(viable, per_target)
        print("FOCUSED_PLAN " + json.dumps({"target": target_id, "screened": len(focus_specs(original_x, world["nominal_world_m"][0])),
            "viable": len(viable), "selected": [{k: v for k, v in s.items() if k != "visual_bound_screen"} for s in result["selected"][target_id]]}), flush=True)
    if source_hashes(stage) != initial_hashes or any(not l.anonymous and l.dirty for l in stage.GetUsedLayers()):
        raise ValueError("Source USD changed during planning")
    load_drafts(drafts)
    result["state"] = "ready_for_rendered_visibility_check" if all(result["selected"].values()) else "blocked_no_geometry_candidates"
    result["source_assets_unchanged"] = True
    result["implementation_sha256"] = sha256(Path(__file__))
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(jsonable(result), indent=2, allow_nan=False), encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = make_plan(args.drafts, args.package, args.output)
    print(result["state"], args.output / "plan.json", flush=True)


if __name__ == "__main__":
    main()
