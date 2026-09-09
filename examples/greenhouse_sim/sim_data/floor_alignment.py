"""Geometric floor placement for a static preview, not contact simulation."""
from __future__ import annotations

import numpy as np
from pxr import Gf, Usd, UsdGeom

PACKAGE_FLOOR = "/World/Environment/GreenHouse/floor"


def floor_triangles(stage, floor_path):
    """Read visible, triangulated floor geometry in metre-scale world space."""
    if UsdGeom.GetStageUpAxis(stage) != "Z" or not np.isclose(
            UsdGeom.GetStageMetersPerUnit(stage), 1.0):
        raise ValueError("Floor alignment requires a metre-scale Z-up stage")
    root = stage.GetPrimAtPath(floor_path)
    if not root or not root.IsActive():
        raise ValueError(f"Missing floor geometry: {floor_path}")
    triangles = []
    cache = UsdGeom.XformCache()
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        if mesh.ComputeVisibility() == "invisible" or mesh.ComputePurpose() not in ("default", "render"):
            continue
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=int)
        indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
        points = mesh.GetPointsAttr().Get()
        if (points is None or not len(points) or not len(counts) or np.any(counts != 3)
                or len(indices) != 3 * len(counts) or np.any(indices < 0) or np.any(indices >= len(points))):
            raise ValueError(f"Valid triangulated floor meshes required: {prim.GetPath()}")
        matrix = cache.GetLocalToWorldTransform(prim)
        world = np.asarray([matrix.Transform(Gf.Vec3d(*p)) for p in points])
        if not np.isfinite(world).all():
            raise ValueError("Non-finite floor geometry")
        faces = indices.reshape(-1, 3)
        holes = set(mesh.GetHoleIndicesAttr().Get() or [])
        triangles.extend(world[face] for i, face in enumerate(faces) if i not in holes)
    if not triangles:
        raise ValueError(f"No visible floor triangles: {floor_path}")
    return np.asarray(triangles)


def surface_height(triangles, x, y):
    """Highest actual triangle hit at XY; never the whole-floor AABB maximum."""
    if not np.isfinite([x, y]).all():
        raise ValueError("Non-finite support location")
    a, b, c = np.moveaxis(triangles, 1, 0)
    determinant = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
    valid = np.abs(determinant) > 1e-12
    if not valid.any():
        raise ValueError("No floor surface under support location")
    a, b, c, determinant = a[valid], b[valid], c[valid], determinant[valid]
    u = ((b[:, 1] - c[:, 1]) * (x - c[:, 0]) + (c[:, 0] - b[:, 0]) * (y - c[:, 1])) / determinant
    v = ((c[:, 1] - a[:, 1]) * (x - c[:, 0]) + (a[:, 0] - c[:, 0]) * (y - c[:, 1])) / determinant
    inside = (u >= -1e-10) & (v >= -1e-10) & (u + v <= 1 + 1e-10)
    if not inside.any():
        raise ValueError(f"No floor surface under support location {(x, y)}")
    return float(np.max((u * a[:, 2] + v * b[:, 2] + (1 - u - v) * c[:, 2])[inside]))


def align_robot_to_floor(stage, robot_path, floor_path=PACKAGE_FLOOR):
    """Translate the complete robot vertically; reject unsupported/unlevel poses.

    Uses rendered wheel bounds and floor triangle hits at wheel centres plus
    chassis footprint samples. This limited static check is NOT a collision,
    suspension, wheel traction, or dynamics validation.
    """
    triangles = floor_triangles(stage, floor_path)
    root = stage.GetPrimAtPath(robot_path)
    transform = UsdGeom.Xformable(root)
    operations = transform.GetOrderedXformOps()
    if len(operations) != 1 or operations[0].GetOpType() != UsdGeom.XformOp.TypeTransform or transform.GetResetXformStack():
        raise ValueError("Preview root must have one inherited matrix transform")
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    bounds = {}
    for name in ("wheel_l", "wheel_r", "base"):
        prim = stage.GetPrimAtPath(f"{robot_path}/{name}")
        if not prim or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            raise ValueError(f"Missing robot support geometry: {name}")
        bound = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if bound.IsEmpty() or not np.isfinite([*bound.GetMin(), *bound.GetMax()]).all():
            raise ValueError(f"Missing visible robot support geometry: {name}")
        bounds[name] = bound
    wheels = []
    for name in ("wheel_l", "wheel_r"):
        bound = bounds[name]
        x, y = list(bound.GetMidpoint())[:2]
        wheels.append({"link": name, "sample_world_xy_m": [x, y],
                       "floor_world_z_m": surface_height(triangles, x, y),
                       "bottom_before_world_z_m": float(bound.GetMin()[2])})
    base = bounds["base"]
    floor_heights = [w["floor_world_z_m"] for w in wheels]
    floor_heights.extend(surface_height(triangles, x, y)
                         for x in (base.GetMin()[0], base.GetMax()[0])
                         for y in (base.GetMin()[1], base.GetMax()[1]))
    shifts = [w["floor_world_z_m"] - w["bottom_before_world_z_m"] for w in wheels]
    tolerance = 1e-4  # 0.1 mm numerical placement tolerance, not a contact offset.
    if np.ptp(floor_heights) > tolerance or np.ptp(shifts) > tolerance:
        raise ValueError("Uneven support: static preview requires a level floor and both wheels aligned")
    dz = max(shifts)
    if base.GetMin()[2] + dz < max(floor_heights) - tolerance:
        raise ValueError("Chassis would intersect the floor with wheels seated")
    parent = UsdGeom.XformCache().GetLocalToWorldTransform(root.GetParent())
    if abs(parent.GetDeterminant()) < 1e-12:
        raise ValueError("Non-invertible robot parent transform")
    matrix = Gf.Matrix4d(operations[0].Get())
    matrix.SetTranslateOnly(matrix.ExtractTranslation() + parent.GetInverse().TransformDir(Gf.Vec3d(0, 0, dz)))
    # Source asset opinions are never edited, even if called with a root edit target.
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        operations[0].Set(matrix)
    for wheel in wheels:
        wheel["bottom_after_world_z_m"] = wheel["bottom_before_world_z_m"] + dz
        wheel["clearance_m"] = wheel["bottom_after_world_z_m"] - wheel["floor_world_z_m"]
    position = UsdGeom.XformCache().GetLocalToWorldTransform(root).ExtractTranslation()
    return {"method": "floor_triangle_hits_and_rendered_wheel_bounds.v1",
            "floor_path": floor_path, "vertical_adjustment_m": dz,
            "robot_position_world_m": list(position), "wheel_supports": wheels,
            "base_minimum_clearance_m": float(base.GetMin()[2] + dz - max(floor_heights)),
            "physics_validated": False}
