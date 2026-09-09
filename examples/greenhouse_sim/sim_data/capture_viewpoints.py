"""Bounded offline robot-view candidates and conservative visual-bound screening.

No navigation, dynamics, contact permission or real robot commands. Geometry
guides this pilot, so these views must not be advertised as blind evaluation.
"""
from __future__ import annotations

import numpy as np
from pxr import Usd, UsdGeom

from .capture_contract import RESOLUTION
from .floor_alignment import PACKAGE_FLOOR


def component_catalogue(stage, records, reports, variants):
    # Draft reports are already augmented variants, not the unmodified source.
    by_variant = {r["plant_id"]: r for r in reports}
    by_root = {v["plant_root"]: v for v in variants}
    items = []
    for record in records:
        variant = by_root[record["plant_root"]]
        report = by_variant[variant["variant_id"]]
        components = {**report["components"], **variant["added_components"]}
        paths = {**record["component_paths"], **variant["added_component_paths"]}
        for key, path in paths.items():
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsActive():
                continue
            items.append({"component_id": key, "prim_path": path, "organ_type": components[key]["type"],
                          "variant_id": variant["variant_id"], "source_plant_id": variant["source_plant_id"],
                          "split_group": variant["split_group"]})
    return [{**item, "component_index": i} for i, item in enumerate(sorted(items, key=lambda c: c["prim_path"]), 1)]


def candidate_specs(original_x, target_x):
    """Fixed metre offsets in the existing aisle; never scale the robot/scene."""
    if not np.isfinite([original_x, target_x]).all() or original_x <= target_x:
        raise ValueError("Expected the existing robot on the +X aisle side")
    result = []
    # Existing distance is retained as a control. Nearer positions can fail
    # floor, joint, visual-bound or measured visibility checks without fallback.
    for approach in (0., .1, .2, .3):
        x = original_x - approach
        if x - target_x < .35:
            continue
        for offset in (-.30, 0., .30):
            index = len(result)
            fraction = ((.36, .43), (.57, .60), (.43, .66))[index % 3]
            result.append({"candidate_id": f"view_{index+1:03d}", "root_x_m": float(x),
                           "approach_from_original_m": approach, "y_offset_m": offset,
                           "desired_pixel_xy": [fraction[0] * RESOLUTION[0], fraction[1] * RESOLUTION[1]]})
    return result


def visible_bounds(stage, root_path="/World", exclude=()):
    """Per-renderable boundable, including instance proxies; not parent subtree boxes."""
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    root = stage.GetPrimAtPath(root_path)
    if not root:
        raise ValueError(f"Missing geometry root: {root_path}")
    records = []
    transforms = UsdGeom.XformCache()
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if any(path == p or path.startswith(p + "/") for p in exclude):
            continue
        if not prim.IsA(UsdGeom.Boundable):
            continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == "invisible" or imageable.ComputePurpose() not in ("default", "render"):
            continue
        bounds = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
        if bounds.IsEmpty():
            continue
        low, high = np.array(bounds.GetMin()), np.array(bounds.GetMax())
        if not np.isfinite([low, high]).all():
            raise ValueError(f"Non-finite visible bound: {path}")
        local = bbox.ComputeUntransformedBound(prim).ComputeAlignedRange()
        records.append({"path": path, "min": low.tolist(), "max": high.tolist(),
                        "local_min": list(local.GetMin()), "local_max": list(local.GetMax()),
                        "local_to_world_row": [list(r) for r in transforms.GetLocalToWorldTransform(prim)]})
    if not records:
        raise ValueError("No visible boundable geometry to screen")
    return records


def screen_bounds(robot_bounds, obstacles, margin_m=.01, refinement=None):
    """Reject broad-phase overlaps. AABB overlap is not proof of mesh collision."""
    if not robot_bounds or not obstacles or not np.isfinite(margin_m) or margin_m < 0:
        raise ValueError("Nonempty robot/obstacle geometry and valid margin required")
    lower = np.array([o["min"] for o in obstacles], float)
    upper = np.array([o["max"] for o in obstacles], float)
    if not np.isfinite([lower, upper]).all() or np.any(upper < lower):
        raise ValueError("Invalid obstacle bounds")
    overlaps = []
    broad_pairs, cleared_pairs = 0, 0
    minimum = float("inf")
    for part in robot_bounds:
        low, high = np.asarray(part["min"]), np.asarray(part["max"])
        if not np.isfinite([low, high]).all() or np.any(high < low):
            raise ValueError("Invalid robot bounds")
        gap = np.maximum(lower - high, low - upper)
        separation = np.linalg.norm(np.maximum(gap, 0), axis=1)
        minimum = min(minimum, float(separation.min()))
        for i in np.flatnonzero(separation <= margin_m):
            broad_pairs += 1
            if refinement is not None and not refinement(part, obstacles[i], margin_m):
                cleared_pairs += 1
                continue
            overlaps.append({"robot_path": part["path"], "scene_path": obstacles[i]["path"],
                             "aabb_separation_m": float(separation[i]),
                             "possible_overlap_not_exact_collision": True})
    return {"method": ("world_aabbs_then_scene_triangles_vs_robot_local_bounds" if refinement else
                       "all_visible_boundable_world_aabbs_with_instance_proxies"),
            "passed": not overlaps, "minimum_aabb_separation_m": minimum, "margin_m": margin_m,
            "possible_overlap_count": len(overlaps), "possible_overlaps": overlaps[:30],
            "robot_bound_count": len(robot_bounds), "scene_bound_count": len(obstacles),
            "broadphase_pair_count": broad_pairs, "pairs_cleared_by_triangle_refinement": cleared_pairs,
            "floor_checked_separately": True, "collision_free_certified": False,
            "not_checked": ["robot self collision", "path between snapshots", "dynamics and contact",
                            "invisible collision-only shapes", "containment inside closed plant volumes"]}


def triangles_intersect_box(triangles, lower, upper):
    """Vectorized triangle/box separating-axis test, including grazing contact."""
    triangles = np.asarray(triangles, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.isfinite(triangles).all():
        raise ValueError("Finite triangles required")
    if lower.shape != (3,) or upper.shape != (3,) or not np.isfinite([lower, upper]).all() or np.any(upper < lower):
        raise ValueError("Invalid triangle-test bounds")
    tri = triangles - (lower + upper) / 2
    half = (upper - lower) / 2
    possible = np.all(tri.max(axis=1) >= -half-1e-10, axis=1) & np.all(tri.min(axis=1) <= half+1e-10, axis=1)
    tri = tri[possible]
    if not len(tri):
        return False
    edges = np.roll(tri, -1, axis=1) - tri
    normals = np.cross(edges[:,0], edges[:,1])[:,None,:]
    crosses = np.cross(edges[:,:,None,:], np.eye(3)[None,None,:,:]).reshape(-1,9,3)
    axes = np.concatenate((normals, crosses), axis=1)
    projection = np.einsum("nvc,nac->nva", tri, axes)
    radius = np.abs(axes) @ half
    separated = (projection.min(axis=1) > radius+1e-10) | (projection.max(axis=1) < -radius-1e-10)
    return bool(np.any(~np.any(separated, axis=1)))


def scene_triangle_refiner(stage):
    """Cache static scene triangles; keep conservative AABBs for unsupported shapes.

    Tests scene triangle surfaces against enclosing robot local boxes, NOT exact
    robot meshes. Structural volumes retain their AABB result. Plants use a
    surface screen; containment inside closed plant volumes is not certified.
    """
    from .capture_contract import transform_points
    cache = {}

    def refine(robot, obstacle, margin):
        path = obstacle["path"]
        # Merged plant/backdrop meshes are open foliage surfaces. Structural
        # closed volumes retain conservative box checks, including containment.
        if not path.startswith("/World/PackPlants/"):
            return True
        if path not in cache:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsA(UsdGeom.Mesh):
                cache[path] = None
            else:
                mesh = UsdGeom.Mesh(prim)
                points = np.asarray(mesh.GetPointsAttr().Get(), float)
                counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), int)
                indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int)
                if (points.ndim != 2 or points.shape[1:] != (3,) or not len(counts) or not np.isin(counts,[3,4]).all()
                        or len(indices) != int(counts.sum()) or np.any(indices < 0) or np.any(indices >= len(points))):
                    cache[path] = None
                else:
                    # Keep BOTH quad diagonal choices as a conservative surface
                    # superset; do not assume the renderer's triangulation choice.
                    starts = np.r_[0, np.cumsum(counts)[:-1]]
                    holes = set(mesh.GetHoleIndicesAttr().Get() or [])
                    kept = np.array([i not in holes for i in range(len(counts))])
                    tris = indices[starts[(counts==3)&kept,None] + np.arange(3)]
                    quads = indices[starts[(counts==4)&kept,None] + np.arange(4)]
                    faces = np.concatenate((tris, *(quads[:,choice] for choice in
                        ([0,1,2], [0,2,3], [0,1,3], [1,2,3]))))
                    triangles = transform_points(points, obstacle["local_to_world_row"])[faces]
                    cache[path] = (triangles, triangles.min(axis=1), triangles.max(axis=1))
        cached = cache[path]
        if cached is None:
            return True
        triangles, low, high = cached
        near = np.all(high >= np.asarray(robot["min"])-margin, axis=1) & np.all(low <= np.asarray(robot["max"])+margin, axis=1)
        if not near.any():
            return False
        inverse = np.linalg.inv(robot["local_to_world_row"])
        local = transform_points(triangles[near].reshape(-1,3), inverse).reshape(-1,3,3)
        pad = margin * np.linalg.norm(inverse[:3,:3], axis=0)
        return triangles_intersect_box(local, np.asarray(robot["local_min"])-pad, np.asarray(robot["local_max"])+pad)

    return refine


def static_obstacles(stage, robot_root):
    return visible_bounds(stage, exclude=(robot_root, PACKAGE_FLOOR))


def rank_key(record):
    """Quality gates first; no diagnostic frame is silently promoted to approved."""
    q, v = record["quality"], record["visibility"]
    return (q["clear_view_gate_passed"], v["nominal"]["visible_target_evidence"],
            v["sampled_interval_visible_pixel_fraction"] or 0,
            min(q["estimated_petiole_diameter_px"], 10),
            min(q["projected_interval_length_px"], 20),
            -(q["target_mask_dark_fraction"] if q["target_mask_dark_fraction"] is not None else 1))


def select_diverse(records, count=3):
    """Keep ranked distinct views, preferring >=15 cm XY baseline diversity."""
    remaining = sorted(records, key=rank_key, reverse=True)
    selected = []
    while remaining and len(selected) < count:
        candidates = [r for r in remaining if all(np.linalg.norm(
            np.asarray(r["base_xy_m"]) - s["base_xy_m"]) >= .15 - 1e-8 for s in selected)]
        # Never trade a passing quality gate for diversity of a failed view.
        best_gate = remaining[0]["quality"]["clear_view_gate_passed"]
        candidates = [r for r in candidates if r["quality"]["clear_view_gate_passed"] == best_gate]
        chosen = candidates[0] if candidates else remaining[0]
        selected.append(chosen)
        remaining = [r for r in remaining if r["candidate_id"] != chosen["candidate_id"]]
    return selected
