"""Read current USD state on the main thread; hand workers only plain data."""
from __future__ import annotations

import hashlib
import itertools

import numpy as np
from pxr import Gf, Usd, UsdGeom

from greenhouse_sim import robot_kinematics as rk, robot_model
from .audit import safe_asset
from .geometry import bounds_by_component
from .reachability import reconstruct_joints


def capture_snapshot(stage, record, report, target, *, borrowed=False, robot_path="/World/RBY1"):
    if report["status"] == "blocked" or target["status"] != "needs_review":
        raise ValueError("Choose an intact, structurally eligible target (not an excluded stub)")
    if UsdGeom.GetStageUpAxis(stage) != "Z" or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1):
        raise ValueError("Reachability requires metre-scale Z-up coordinates")
    root = stage.GetPrimAtPath(robot_path)
    if not root or root.GetAttribute("tomato:robotModel").Get() != robot_model.ROBOT_NAME:
        raise ValueError("Load the fitted RB-Y1 Model A v1.2 preview first")
    if root.GetAttribute("tomato:previewOnly").Get() is not True:
        raise ValueError("This diagnostic supports the static Phase 1 preview only")
    model, cache = rk.Rby1Kinematics(), UsdGeom.XformCache()

    def matrix(path):
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive():
            raise ValueError(f"Missing active pose: {path}")
        return np.asarray(cache.GetLocalToWorldTransform(prim), dtype=float).T

    base = matrix(robot_path + "/base")
    if (not np.isfinite(base).all() or not np.allclose(base[:3, :3].T @ base[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(base[:3, :3]), 1, atol=1e-6)):
        raise ValueError("Scaled/non-rigid robot base is unsupported")
    inverse_base = np.linalg.inv(base)
    links = {k: (inverse_base @ matrix(f"{robot_path}/{k}")).tolist() for k in ("base", *model._by_child)}
    angles, slides = reconstruct_joints(model, links)
    component = report["components"][target["component_id"]]
    component_matrix = matrix(record["component_paths"][target["component_id"]])
    local_attachment = np.asarray(target["attachment_plant_m"]) - component["translation_plant_m"]
    target_world = (component_matrix @ np.r_[local_attachment, 1])[:3]
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    tools, points = {}, {}
    for side in ("left", "right"):
        tools[side], points[side] = [], None
        ee = matrix(f"{robot_path}/ee_{side}")
        inverse_ee = np.linalg.inv(ee)
        finger_centers = []
        for name in (f"ee_{side}/visuals", f"ee_{side}/attachments",
                     f"ee_finger_{side[0]}1/visuals", f"ee_finger_{side[0]}2/visuals"):
            prim = stage.GetPrimAtPath(f"{robot_path}/{name}")
            if not prim or not prim.IsActive():
                continue
            if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
                raise ValueError("Restore scene visibility before checking reachability")
            bound = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            if bound.IsEmpty():
                continue
            corners = np.asarray(list(itertools.product(*zip(bound.GetMin(), bound.GetMax()))))
            local = (inverse_ee @ np.c_[corners, np.ones(8)].T).T[:, :3]
            low, high = local.min(axis=0), local.max(axis=0)
            center = (low + high) / 2
            tools[side].append({"path": str(prim.GetPath()), "center": center.tolist(),
                                "half_extents": ((high - low) / 2).tolist()})
            if name.startswith("ee_finger_"):
                finger_centers.append(center)
        if len(finger_centers) == 2:
            points[side] = np.mean(finger_centers, axis=0).tolist()
    bounds, _ = bounds_by_component(stage, record["component_paths"])
    obstacles = [{"path": record["component_paths"][key], "min": list(value.GetMin()), "max": list(value.GetMax())}
                 for key, value in bounds.items() if not value.IsEmpty()]
    gutter_root = stage.GetPrimAtPath("/World/Gutters")
    if gutter_root:
        for prim in gutter_root.GetChildren():
            value = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            if not value.IsEmpty():
                obstacles.append({"path": str(prim.GetPath()), "min": list(value.GetMin()), "max": list(value.GetMax())})
    from pathlib import Path
    manifest = Path(report["manifest_path"])
    files = [{"path": str(manifest), "sha256": report["manifest_sha256"]},
             {"path": str(robot_model.DEFAULT_URDF), "sha256": hashlib.sha256(robot_model.DEFAULT_URDF.read_bytes()).hexdigest()}]
    files.extend({"path": str(safe_asset(manifest.parent, c["file"])), "sha256": c["asset_sha256"]}
                 for c in report["components"].values())
    return {"robot_model": robot_model.ROBOT_NAME, "robot_path": robot_path,
            "target_id": target["target_id"], "target_status": target["status"],
            "target_definition": "manifest attachment diagnostic, NOT approved cut/grasp point",
            "probe_definition": "midpoint of rendered finger-bound centres; orientation unconstrained",
            "plant_root": record["plant_root"], "placement": "relocated_review_sample" if borrowed else "authored_preview_placement",
            "plant_world_matrix": matrix(record["plant_root"]).tolist(),
            "component_world_matrix": component_matrix.tolist(), "target_world_m": target_world.tolist(),
            "base_world_matrix": base.tolist(), "robot_link_matrices_base": links,
            "torso_degrees": [angles[f"torso_{i}"] for i in range(6)],
            "arm_degrees": {side: [angles[f"{side}_arm_{i}"] for i in range(7)] for side in ("left", "right")},
            "prismatic_m": slides, "probe_points_ee_m": points, "tool_boxes_ee": tools,
            "obstacle_boxes": obstacles, "gutter_geometry_present": bool(gutter_root),
            "right_tool": stage.GetPrimAtPath(robot_path + "/ee_right").GetAttribute("tomato:toolConfiguration").Get(),
            "source_files": files}
