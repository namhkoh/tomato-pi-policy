"""Session-only USD assembly and broad-phase geometry checks (not cut approval)."""

from __future__ import annotations

import math
from pathlib import Path

from .audit import safe_asset


def assemble_plant(stage, root_path, report):
    from pxr import Gf, Usd, UsdGeom

    if report["status"] == "blocked":
        raise ValueError("Cannot assemble a plant with structural audit errors")
    components, paths = report["components"], {}
    directory = Path(report["manifest_path"]).parent
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Xform.Define(stage, root_path)
        # Iterative parent-first resolution avoids depending on manifest order.
        pending = set(components)
        while pending:
            ready = sorted(k for k in pending if components[k]["parent"] is None or components[k]["parent"] in paths)
            if not ready:
                raise ValueError("Unresolvable parent graph")
            for key in ready:
                component = components[key]
                parent = component["parent"]
                path = f"{paths[parent] if parent else root_path}/{key}"
                prim = UsdGeom.Xform.Define(stage, path)
                prim.GetPrim().GetReferences().AddReference(safe_asset(directory, component["file"]).as_posix())
                position = Gf.Vec3d(*component["translation_plant_m"])
                parent_position = Gf.Vec3d(*components[parent]["translation_plant_m"]) if parent else Gf.Vec3d(0)
                prim.AddTranslateOp().Set(position - parent_position)
                paths[key] = path
                pending.remove(key)
    return paths


def bounds_by_component(stage, paths):
    """Use each component's own meshes, never its descendant organs' bounds."""
    from pxr import Gf, Usd, UsdGeom

    owners = {path: key for key, path in paths.items()}
    bounds = {key: Gf.Range3d() for key in paths}
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "proxy"])
    counts = {key: 0 for key in paths}
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        owner = prim
        while owner and str(owner.GetPath()) not in owners:
            owner = owner.GetParent()
        if not owner:
            continue
        key = owners[str(owner.GetPath())]
        bounds[key].UnionWith(cache.ComputeWorldBound(prim).ComputeAlignedRange())
        counts[key] += 1
    return bounds, counts


def point_bounds_distance(point, bounds):
    if bounds.IsEmpty():
        return None
    low, high = bounds.GetMin(), bounds.GetMax()
    return math.sqrt(sum(max(low[i] - point[i], 0, point[i] - high[i]) ** 2 for i in range(3)))


def audit_geometry(report):
    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.SetStageUpAxis(stage, "Z")
    # Anonymous root is setup-only; imported component layers are never edited.
    paths = assemble_plant(stage, "/Plant", report)
    bounds, counts = bounds_by_component(stage, paths)
    cache = UsdGeom.XformCache()
    components = {}
    for key, component in report["components"].items():
        actual = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(paths[key])).ExtractTranslation()
        error = math.dist(actual, component["translation_plant_m"])
        value = bounds[key]
        components[key] = {
            "mesh_count": counts[key], "translation_error_m": error,
            "own_bounds_plant_m": None if value.IsEmpty() else [list(value.GetMin()), list(value.GetMax())],
            "attachment_to_own_bounds_m": point_bounds_distance(component["attachment_plant_m"], value),
            "attachment_to_parent_bounds_m": point_bounds_distance(
                component["attachment_plant_m"], bounds[component["parent"]]
            ) if component["parent"] else None,
        }
    warnings = []
    for key, item in components.items():
        if item["mesh_count"] == 0 or item["own_bounds_plant_m"] is None:
            warnings.append({"component_id": key, "code": "missing_geometry"})
        if item["translation_error_m"] > 1e-6:
            warnings.append({"component_id": key, "code": "assembled_translation_mismatch"})
        # A diagnostic tolerance, not a main-stem clearance or agronomic rule.
        for field in ("attachment_to_own_bounds_m", "attachment_to_parent_bounds_m"):
            if item[field] is not None and item[field] > 0.002:
                warnings.append({"component_id": key, "code": field + "_over_2mm", "distance_m": item[field]})
    for target in report["targets"]:
        members = set(target["expected_detached_component_ids"])
        target["geometry_review_flags"] = [w for w in warnings if w["component_id"] in members]
    return {
        "scope": "assembled_translation_and_mesh_aabb_only",
        "not_a_surface_or_collision_validation": True, "components": components, "warnings": warnings,
        "maximum_translation_error_m": max((v["translation_error_m"] for v in components.values()), default=0),
        "agronomic_eligibility": "unreviewed", "physical_executability": "not_tested",
    }
