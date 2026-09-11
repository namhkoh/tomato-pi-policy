"""Capture native cooked convex data from ONE already-open, stopped Kit stage.

No SimulationApp/timeline ownership, stepping, stage authoring, cache clearing,
screen replacement, or collider changes. Importing this module needs no Kit.

After the caller creates its diagnostic stage with the existing builders:
    from sim_physics.cooked_geometry_probe import capture_current_stage
    report = await capture_current_stage(
        "D:/research/tomato-pi-policy/data/sim_physics/cooked_probe_NEW.json")

The output must not exist. Call before play, not on a running/paused Fabric
scene. Kit updates service cooking only while the timeline remains stopped.
Use a parent-process wall timeout as well: an event-loop deadline cannot
interrupt a stalled native extension call.

Installed Isaac Sim 6.0.1 evidence (omni.physx 110.1.13):
* bindings/_physx.pyi:1989: request_convex_collision_representation, keyword
  stage_id, collision_prim_id, run_asynchronously, on_result(result, convexes).
* physxtests/tests/PhysxCookingInterface.py:42,84: async completion and
  cancel_collision_representation_task(task=..., invoke_callback=False).
* physxdemos/scenes/ConvexMeshDataDemo.py:215: returned vertices are drawn
  through the source mesh's full local-to-world transform (including scale).
  Polygon index_base/num_vertices address convex.indices, not triangle triples.
* The example permits cooking before simulation; it is NOT a query of an
  existing actor's attached shapes. RESULT_VALID is native cooking evidence,
  not proof of live actor identity, physical contact, or path clearance.
"""

import asyncio
import hashlib
import json
import math
import operator
from pathlib import Path
import sys
import time

import numpy as np


def _json_value(value):
    """Stable JSON values for USD arrays, matrices, quaternions and assets."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError("Nonfinite source value")
        return float(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if hasattr(value, "resolvedPath") and hasattr(value, "path"):
        return {"asset_path": value.path, "resolved_path": value.resolvedPath}
    if hasattr(value, "GetReal") and hasattr(value, "GetImaginary"):
        return {"real": float(value.GetReal()), "imaginary": _json_value(value.GetImaginary())}
    if isinstance(value, np.integer):
        return int(value)
    try:
        return [_json_value(item) for item in value]
    except TypeError as exc:
        raise ValueError("Unsupported source value: " + type(value).__name__) from exc


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def world_vertices(vertices, local_to_world, meters_per_unit):
    """Column-matrix convention; apply full affine transform, then USD units."""
    points = np.asarray(vertices, dtype=float)
    matrix = np.asarray(local_to_world, dtype=float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or matrix.shape != (4, 4)
            or not np.isfinite(points).all() or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-12)
            or abs(np.linalg.det(matrix[:3, :3])) < 1e-15
            or not math.isfinite(meters_per_unit) or meters_per_unit <= 0):
        raise ValueError("Invalid vertex transform or USD units")
    return ((points @ matrix[:3, :3].T + matrix[:3, 3]) * meters_per_unit).tolist()


def pack_convex(convex):
    """Copy native callback-owned buffers immediately; validate indexed faces."""
    vertices = [[float(v.x), float(v.y), float(v.z)] for v in convex.vertices]
    points = np.asarray(vertices, float)
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 4
            or not np.isfinite(points).all()
            or np.linalg.matrix_rank(points - points.mean(0)) != 3):
        raise ValueError("Invalid or degenerate native convex vertices")
    indices = [operator.index(i) for i in convex.indices]
    if not indices or min(indices) < 0 or max(indices) >= len(vertices):
        raise ValueError("Invalid native convex indices")
    polygons = []
    edges = {}
    for face in convex.polygons:
        base, count = operator.index(face.index_base), operator.index(face.num_vertices)
        plane = [float(v) for v in face.plane]
        if (count < 3 or base < 0 or base + count > len(indices)
                or len(plane) != 4 or not np.isfinite(plane).all()
                or np.linalg.norm(plane[:3]) <= 1e-15):
            raise ValueError("Invalid native polygon span or plane")
        face_indices = indices[base:base + count]
        if len(set(face_indices)) != count:
            raise ValueError("Repeated polygon vertex")
        for a, b in zip(face_indices, face_indices[1:] + face_indices[:1]):
            edge = tuple(sorted((a, b)))
            edges[edge] = edges.get(edge, 0) + 1
        polygons.append({"index_base": base, "num_vertices": count, "plane_local_raw": plane})
    if len(polygons) < 4 or not edges or any(count != 2 for count in edges.values()):
        raise ValueError("Native convex topology is not a closed polyhedron")
    result = {"vertices_collider_local": vertices, "indices": indices, "polygons": polygons}
    result["raw_geometry_sha256"] = fingerprint(result)
    return result


def select_targets(paths, robot_root="/World/RBY1", plant_root="/World/Plant"):
    """Fail on missing/ambiguous targets; never silently substitute another plant."""
    suffixes = [
        (robot_root, "/ee_right/attachments/RightWristCamera/BracketCollision"),
        (plant_root, "/MainStem_26/MainStem_26"),
        (plant_root, "/MainStem_27/MainStem_27"),
    ]
    selected = []
    for root, suffix in suffixes:
        matches = [p for p in paths if p.startswith(root.rstrip("/") + "/") and p.endswith(suffix)]
        if len(matches) != 1:
            raise ValueError(f"Expected one {root}*{suffix}; found {len(matches)}")
        selected.append(matches[0])
    return selected


async def request_convexes(cooking, stage_id, prim_ids, next_update, guard, *,
                           valid_result, timeout_s=60.0):
    """Dependency-injected orchestration; no Kit import or stage mutations."""
    if not math.isfinite(timeout_s) or not 0 < timeout_s <= 90:
        raise ValueError("Cooking deadline must be positive and <=90 seconds")
    if not prim_ids:
        raise ValueError("At least one explicit collision prim is required")
    deadline = time.monotonic() + timeout_s
    rows, tasks = {}, {}
    accepting = True

    def callback(path, result, convexes):
        if not accepting:
            return
        if path in rows:
            rows[path] = {"status": "invalid", "error": "Duplicate native callback"}
            return
        label = str(getattr(result, "name", result))
        if result != valid_result:
            rows[path] = {"status": "native_error", "result": label}
            return
        try:
            parts = [pack_convex(c) for c in convexes]
            if not parts:
                raise ValueError("RESULT_VALID with no convexes")
            # The installed demo warns that decomposition order is unstable.
            parts.sort(key=lambda part: part["raw_geometry_sha256"])
            rows[path] = {"status": "valid", "result": label, "convexes": parts,
                          "convex_count": len(parts)}
        except Exception as exc:
            rows[path] = {"status": "invalid", "result": label, "error": str(exc)}

    try:
        for path, prim_id in prim_ids.items():
            guard()
            tasks[path] = cooking.request_convex_collision_representation(
                stage_id=stage_id, collision_prim_id=prim_id, run_asynchronously=True,
                on_result=lambda result, convexes, path=path: callback(path, result, convexes))
        while len(rows) < len(prim_ids):
            guard()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise asyncio.TimeoutError
            await asyncio.wait_for(next_update(), timeout=remaining)
        guard()
    except asyncio.TimeoutError:
        for path in prim_ids:
            rows.setdefault(path, {"status": "timeout", "error": "Cooking deadline exceeded"})
    finally:
        accepting = False
        # A timed-out row is not a completed native task.
        cancellation_errors = []
        for path, task in tasks.items():
            if path not in rows or rows[path]["status"] == "timeout":
                try:
                    cooking.cancel_collision_representation_task(task=task, invoke_callback=False)
                except Exception as exc:
                    cancellation_errors.append(f"{path}: {exc}")
        if cancellation_errors:
            raise RuntimeError("Native cancellation failed: " + "; ".join(cancellation_errors))
    return rows


def _source_snapshot(stage, paths):
    """Bind composed geometry, transforms, settings attrs and contributing layers."""
    from pxr import Usd, UsdGeom, UsdPhysics
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    layers = {stage.GetRootLayer().identifier: stage.GetRootLayer(),
              stage.GetSessionLayer().identifier: stage.GetSessionLayer()}
    meshes = {}
    for path in paths:
        prim = stage.GetPrimAtPath(path)
        if (not prim or not prim.IsActive() or not prim.IsA(UsdGeom.Mesh)
                or not prim.HasAPI(UsdPhysics.CollisionAPI)
                or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()):
            raise ValueError("Expected active collision mesh: " + path)
        approximation = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        if approximation not in ("convexHull", "convexDecomposition"):
            raise ValueError("Unsupported convex approximation: " + str(approximation))
        ancestors = []
        current = prim
        while current and not current.IsPseudoRoot():
            for spec in current.GetPrimStack():
                layers[spec.layer.identifier] = spec.layer
            ancestors.append({"path": str(current.GetPath()),
                              "schemas": list(current.GetAppliedSchemas()),
                              "physics_attributes": {
                                  a.GetName(): _json_value(a.Get()) for a in current.GetAttributes()
                                  if a.GetName().startswith(("physics:", "physx"))}})
            current = current.GetParent()
        matrix = np.asarray(transforms.GetLocalToWorldTransform(prim)).T.tolist()
        meshes[path] = {
            "approximation": approximation,
            "attributes": {a.GetName(): _json_value(a.Get()) for a in prim.GetAttributes()},
            "ancestors": ancestors,
            "local_to_world_column_matrix": matrix,
            "transform_source": "stopped_stage_USD_not_live_Fabric_body_pose",
        }
    bindings = []
    for identifier, layer in sorted(layers.items()):
        real_path = str(layer.realPath)
        binding = {"identifier": identifier, "real_path": real_path,
                   "composed_layer_text_sha256": hashlib.sha256(
                       layer.ExportToString().encode("utf-8")).hexdigest()}
        if real_path:
            if not Path(real_path).is_file():
                raise ValueError("Cannot bind nonlocal/missing source layer: " + real_path)
            binding["file_sha256"] = _file_hash(real_path)
        bindings.append(binding)
    snapshot = {"meters_per_unit": float(UsdGeom.GetStageMetersPerUnit(stage)),
                "up_axis": str(UsdGeom.GetStageUpAxis(stage)), "meshes": meshes,
                "layer_bindings": bindings}
    snapshot["source_sha256"] = fingerprint(snapshot)
    return snapshot


def write_new_report(path, report):
    """Create only a new report; never overwrite source or an earlier result."""
    path = Path(path)
    if not path.parent.is_dir():
        raise ValueError("Report parent directory must already exist")
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


async def capture_current_stage(output, *, robot_root="/World/RBY1",
                                plant_root="/World/Plant", timeout_s=60.0):
    """Capture three existing colliders; caller owns one stopped diagnostic stage.

    A successful capture is advisory, not automatically eligible to clear any
    existing screen. In particular this API does not expose actor/shape handles.
    Native cooking may populate its existing cache; no cache settings are changed.
    """
    output = Path(output)
    if output.exists() or not output.parent.is_dir():
        raise ValueError("A NEW report path in an existing directory is required")
    if not math.isfinite(timeout_s) or not 0 < timeout_s <= 90:
        raise ValueError("Cooking deadline must be positive and <=90 seconds")
    # All runtime imports are deliberately lazy. No SimulationApp is constructed.
    import carb.settings
    import omni.kit.app
    import omni.timeline
    import omni.usd
    from omni.physx import get_physx_cooking_interface
    from omni.physx.bindings import _physx
    from pxr import PhysicsSchemaTools, Sdf, Usd, UsdPhysics, UsdUtils

    context = omni.usd.get_context()
    stage = context.get_stage()
    if stage is None:
        raise ValueError("Caller must first open the diagnostic stage")
    timeline = omni.timeline.get_timeline_interface()

    def guard():
        if not timeline.is_stopped():
            raise RuntimeError("Capture requires a stopped timeline; it will not stop it")
        if context.get_stage() != stage:
            raise RuntimeError("Active diagnostic stage changed during cooking")

    guard()
    stage_id = UsdUtils.StageCache.Get().GetId(stage)
    if not stage_id.IsValid():
        raise ValueError("Active stage must already be registered in USD StageCache")
    paths = select_targets([str(p.GetPath()) for p in
                            Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies())
                            if p.HasAPI(UsdPhysics.CollisionAPI)],
                           robot_root, plant_root)
    source = _source_snapshot(stage, paths)
    settings = carb.settings.get_settings()

    def cooking_settings():
        keys = sorted({getattr(_physx, name) for name in dir(_physx)
                       if name.startswith("SETTING_")
                       and isinstance(getattr(_physx, name), str)
                       and getattr(_physx, name).startswith("/physics")})
        return {key: _json_value(settings.get(key)) for key in keys}

    settings_before = cooking_settings()
    report = {
        "schema": "native_cooked_geometry_probe_v1",
        "status": "pending", "source": source,
        "cooking_api": "omni.physx.get_physx_cooking_interface().request_convex_collision_representation",
        "stage_id": stage_id.ToLongInt(), "python_version": sys.version,
        "usd_version": list(Usd.GetVersion()),
        "physx_binding": {"path": str(_physx.__file__), "sha256": _file_hash(_physx.__file__)},
        "physics_settings": settings_before,
        "coordinate_contract": {
            "vertices": "collider-local coordinates in USD stage units",
            "planes": "raw callback collider-local plane coefficients; not world planes",
            "matrix": "full USD local-to-world; column-vector convention, includes scale",
            "world_meters": "(R_full @ vertex_local + translation) * meters_per_unit",
            "basis": "installed ConvexMeshDataDemo._draw_convex_hulls",
            "native_nonuniform_scale_crosscheck_performed": False,
        },
        "live_actor_shape_identity_verified": False, "native_contact_validated": False,
        "collision_free_path_certified": False, "eligible_to_replace_screen": False,
        "timeline_started": False, "physics_steps_requested": 0, "geometry_authored": False,
        "limitations": [
            "RESULT_VALID supplies this USD prim's native cooking representation, not attached live actor/shape identity.",
            "Returned decomposition order is nondeterministic; parts sorted by raw-geometry hash only.",
            "USD attributes and exposed /physics settings bind explicit values and defaults available in this build; internal cooker defaults are version-bound by the binding binary hash.",
            "No source triangles, visual meshes or new decompositions replace actual colliders.",
            "Contact/rest offsets are retained in source attributes and are not baked into the reported solid vertices.",
            "Source/world transforms are stopped-stage USD. Native transform/scale agreement and scene-query equivalence still require a reviewed native check.",
            "An outer process watchdog remains necessary for a stalled native call or Kit update.",
        ],
    }
    started = time.monotonic()
    try:
        ids = {path: PhysicsSchemaTools.sdfPathToInt(Sdf.Path(path)) for path in paths}
        rows = await request_convexes(
            get_physx_cooking_interface(), stage_id.ToLongInt(), ids,
            omni.kit.app.get_app().next_update_async, guard,
            valid_result=_physx.PhysxCollisionRepresentationResult.RESULT_VALID,
            timeout_s=timeout_s)
        guard()
        after = _source_snapshot(stage, paths)
        report["source_unchanged"] = after["source_sha256"] == source["source_sha256"]
        report["settings_unchanged"] = cooking_settings() == settings_before
        for path, row in rows.items():
            if row["status"] == "valid":
                matrix = source["meshes"][path]["local_to_world_column_matrix"]
                for part in row["convexes"]:
                    part["vertices_world_m"] = world_vertices(
                        part["vertices_collider_local"], matrix, source["meters_per_unit"])
                row["source_mesh_sha256"] = fingerprint(source["meshes"][path])
        report["colliders"] = rows
        report["source_sha256_after"] = after["source_sha256"]
        report["status"] = ("captured_advisory" if report["source_unchanged"]
                            and report["settings_unchanged"]
                            and all(row["status"] == "valid" for row in rows.values())
                            else "blocked")
    except Exception as exc:
        report.update(status="blocked", error_type=type(exc).__name__, error=str(exc))
    report["wall_seconds"] = time.monotonic() - started
    report["payload_sha256"] = fingerprint(report)
    write_new_report(output, report)
    return report


if __name__ == "__main__":
    raise SystemExit(
        "This probe does not launch Kit/SimulationApp. In the caller's already-open, "
        "stopped diagnostic stage, import and await capture_current_stage(NEW_JSON_PATH)."
    )
