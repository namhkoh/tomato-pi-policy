"""Partial-plant mesh/ray SHADOW diagnostics, never a capture or label gate.

Portable USD is imported lazily in this standalone CPU reader, never in a native
worker. Source layers are read; only anonymous composition is authored. No RGB,
native depth, masks or saved decisions enter the predictor. No visibility pass,
skip, ranking, camera override or synthetic sensor output is provided.

Only the generated plant is reconstructed. Known supported opaque blockers can
be predicted; absence of one is UNKNOWN for the full greenhouse. Five rays per
pixel are diagnostics, NOT full pixel-footprint or continuous visibility proof.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from ..capture_contract import project, transform_points
from ..cut_regions import _oriented_chain, _sample
from ..dataset_review import require, verify_bindings
from ..depth_preview import sha256
from ..native_clear_labels import interval_arc_samples

SCHEMA = "greenhouse.partial_plant_visibility_shadow.v1"
POLICY = dict(pixel_offsets=[[.5, .5], [.25, .25], [.75, .25], [.25, .75], [.75, .75]],
    numerical_ray_epsilon_m=1e-8, barycentric_edge_epsilon=1e-8,
    quad_rule="same_face_hit_under_both_diagonals_required",
    material_rule="explicit_static_opaque_UsdPreviewSurface_only",
    coverage="generated_plant_only_surroundings_and_robot_not_reconstructed",
    skip_allowed=False, ranking_allowed=False, depth_output_allowed=False,
    training_approved=False, continuous_visibility_verified=False,
    pixel_footprint_coverage_complete=False)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(value).hexdigest()


def frozen_array(value, dtype=float):
    a = np.asarray(value, dtype=dtype)
    return np.frombuffer(a.tobytes(), dtype=a.dtype).reshape(a.shape)


@dataclass(frozen=True)
class Mesh:
    component: str
    organ: str
    prim_path: str
    lower: np.ndarray
    upper: np.ndarray
    triangles_a: np.ndarray
    triangles_b: np.ndarray
    face_ids: np.ndarray
    face_count: int
    unknown_reasons: tuple[str, ...]


@dataclass(frozen=True)
class Snapshot:
    meshes: tuple[Mesh, ...]
    components_json: bytes
    plant_to_world: np.ndarray
    source_bindings: tuple[tuple[str, str], ...]
    provenance_json: bytes

    @property
    def fingerprint(self):
        return digest(self.provenance_json)

    def finish(self):
        verify_bindings(dict(self.source_bindings))


def triangulate(points, counts, indices, holes=()):
    """Two independent diagonal choices per quad; no invented ngon certainty."""
    p = np.asarray(points, float)
    counts, indices = np.asarray(counts, int), np.asarray(indices, int)
    require(p.ndim == 2 and p.shape[1] == 3 and len(p) and np.isfinite(p).all(), "Invalid mesh points")
    require(counts.ndim == indices.ndim == 1 and np.all(counts >= 3)
            and counts.sum() == len(indices) and np.all((indices >= 0) & (indices < len(p))), "Invalid mesh topology")
    holes = set(map(int, holes))
    require(all(0 <= h < len(counts) for h in holes), "Invalid mesh hole index")
    a, b, ids, offset, unsupported = [], [], [], 0, False
    for face, count in enumerate(counts):
        f = indices[offset:offset+count]; offset += count
        if face in holes: continue
        if count == 3:
            a.append(p[f]); b.append(p[f]); ids.append(face)
        elif count == 4:
            a.extend([p[f[[0, 1, 2]]], p[f[[0, 2, 3]]]])
            b.extend([p[f[[0, 1, 3]]], p[f[[1, 2, 3]]]])
            ids.extend([face, face])
        else:
            unsupported = True
    return (frozen_array(np.asarray(a).reshape(-1, 3, 3)),
            frozen_array(np.asarray(b).reshape(-1, 3, 3)), frozen_array(ids, int), len(counts), unsupported)


def triangle_hits(triangles, origin, direction, near, far):
    """Return ray distances and strict-interior hits; never optical-Z labels."""
    tri = np.asarray(triangles)
    if not len(tri): return np.empty(0), np.empty(0, bool)
    e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    h = np.cross(np.broadcast_to(direction, e2.shape), e2)
    det = np.einsum("ij,ij->i", e1, h)
    valid = np.abs(det) > 1e-14
    inv = np.zeros_like(det); inv[valid] = 1/det[valid]
    s = np.asarray(origin)-tri[:, 0]
    u = inv*np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = inv*(q @ direction)
    distance = inv*np.einsum("ij,ij->i", e2, q)
    eps = POLICY["barycentric_edge_epsilon"]
    valid &= (u >= -eps) & (v >= -eps) & (u+v <= 1+eps) & (distance >= near) & (distance <= far)
    strict = valid & (u > eps) & (v > eps) & (u+v < 1-eps)
    return np.where(valid, distance, np.inf), strict


def mesh_hits(mesh, origin, direction, near, far):
    da, sa = triangle_hits(mesh.triangles_a, origin, direction, near, far)
    db, sb = triangle_hits(mesh.triangles_b, origin, direction, near, far)
    possible = min(float(da.min(initial=np.inf)), float(db.min(initial=np.inf)))
    fa, fb = np.full(mesh.face_count, np.inf), np.full(mesh.face_count, np.inf)
    np.minimum.at(fa, mesh.face_ids, np.where(sa, da, np.inf))
    np.minimum.at(fb, mesh.face_ids, np.where(sb, db, np.inf))
    # BOTH choices must hit the SAME face. All-A vs all-B mesh hits alone
    # would falsely certify some mixed-diagonal meshes.
    certain = float(np.maximum(fa, fb).min(initial=np.inf))
    return possible, certain


def ray_boxes(lower, upper, origin, direction, near, far):
    lower, upper = np.asarray(lower), np.asarray(upper)
    enter, leave = np.full(len(lower), near), np.full(len(lower), far)
    for axis in range(3):
        if abs(direction[axis]) < 1e-15:
            leave[(origin[axis] < lower[:, axis]) | (origin[axis] > upper[:, axis])] = -np.inf
        else:
            a = (lower[:, axis]-origin[axis])/direction[axis]
            b = (upper[:, axis]-origin[axis])/direction[axis]
            enter = np.maximum(enter, np.minimum(a, b)); leave = np.minimum(leave, np.maximum(a, b))
    return enter <= leave, enter


def _material_reasons(prim):
    from pxr import UsdGeom, UsdShade
    reasons = []
    material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
    if not material: return ["unbound_material"]
    surface_outputs = [o for o in material.GetOutputs() if o.GetBaseName().endswith("surface")]
    if len(surface_outputs) != 1 or surface_outputs[0].GetBaseName() != "surface":
        return ["unsupported_surface_render_context"]
    connection = material.GetSurfaceOutput().GetConnectedSource()
    if not connection: return ["unknown_surface_connection"]
    shader = UsdShade.Shader(connection[0].GetPrim())
    if not shader or shader.GetIdAttr().Get() != "UsdPreviewSurface":
        return ["unsupported_surface_shader"]
    opacity = shader.GetInput("opacity")
    if (not opacity or opacity.HasConnectedSource() or opacity.GetAttr().GetNumTimeSamples()
            or opacity.Get() != 1.): reasons.append("opacity_not_explicit_constant_one")
    for name in ("opacityThreshold", "displacement", "transmission"):
        value = shader.GetInput(name)
        if value and (value.HasConnectedSource() or value.GetAttr().GetNumTimeSamples() or value.Get() not in (None, 0.)):
            reasons.append("unsupported_"+name)
    displacement = material.GetDisplacementOutput()
    if displacement and displacement.HasConnectedSource(): reasons.append("material_displacement")
    display = UsdGeom.PrimvarsAPI(prim).GetPrimvar("displayOpacity")
    if display and display.HasAuthoredValue():
        values = display.Get()
        if display.GetAttr().GetNumTimeSamples() or values is None or not np.all(np.asarray(values) == 1.):
            reasons.append("nonopaque_display_primvar")
    if any(child.IsA(UsdGeom.Subset) for child in prim.GetChildren()): reasons.append("material_or_geometry_subsets")
    return reasons


def load_partial_plant(directory, plant_to_world, *, expected_bindings):
    """Read pinned assets with portable USD; only anonymous composition changes."""
    from pxr import Usd, UsdGeom
    from ..audit import audit_manifest
    from ..geometry import assemble_plant
    started = time.perf_counter()
    directory = Path(directory).resolve()
    pins = {str(Path(p).resolve()): h for p, h in expected_bindings.items()}
    manifest = directory / "manifest.json"
    require(str(manifest) in pins, "Externally bound generated manifest required")
    matrix = frozen_array(plant_to_world)
    require(matrix.shape == (4, 4) and np.isfinite(matrix).all()
            and np.allclose(matrix[:, 3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            and np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-9, rtol=0),
            "Rigid bound plant transform required")
    verify_bindings(pins)
    report = audit_manifest(manifest)
    require(report["status"] != "blocked", "Invalid generated manifest")
    require(all(str((directory / c["file"]).resolve()) in pins and
                pins[str((directory / c["file"]).resolve())] == c["asset_sha256"]
                for c in report["components"].values()), "Unbound generated mesh")
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdGeom.SetStageUpAxis(stage, "Z")
    root = "/World/GeneratedNativePilot/" + directory.name
    paths = assemble_plant(stage, root, report)
    owners = {p: key for key, p in paths.items()}
    cache = UsdGeom.XformCache()
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    meshes = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Boundable): continue
        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility() == "invisible" or imageable.ComputePurpose() not in ("default", "render"):
            continue
        owner = prim
        while owner and str(owner.GetPath()) not in owners: owner = owner.GetParent()
        if not owner: continue
        key = owners[str(owner.GetPath())]; component = report["components"][key]
        reasons = []
        if not prim.IsA(UsdGeom.Mesh):
            bounds = bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            require(not bounds.IsEmpty(), "Unsupported shape without bounds")
            corners = np.array([[x, y, z] for x in (bounds.GetMin()[0], bounds.GetMax()[0])
                for y in (bounds.GetMin()[1], bounds.GetMax()[1]) for z in (bounds.GetMin()[2], bounds.GetMax()[2])])
            world = transform_points(corners, matrix)
            ta = tb = frozen_array(np.empty((0, 3, 3))); face_ids = frozen_array([], int); faces = 0
            reasons.append("unsupported_boundable")
        else:
            mesh = UsdGeom.Mesh(prim)
            points = np.asarray(mesh.GetPointsAttr().Get(), float)
            world = transform_points(transform_points(points, np.asarray(cache.GetLocalToWorldTransform(prim))), matrix)
            ta, tb, face_ids, faces, unsupported = triangulate(world, mesh.GetFaceVertexCountsAttr().Get(),
                mesh.GetFaceVertexIndicesAttr().Get(), mesh.GetHoleIndicesAttr().Get() or [])
            if not faces: reasons.append("empty_mesh_no_surface")
            if unsupported: reasons.append("unsupported_polygon")
            if mesh.GetSubdivisionSchemeAttr().Get() != "none": reasons.append("subdivision_not_native_tessellation")
            if mesh.GetDoubleSidedAttr().Get() is not True: reasons.append("single_sided_material_geometry")
            if any(a.GetNumTimeSamples() for a in (mesh.GetPointsAttr(), mesh.GetFaceVertexCountsAttr(),
                                                  mesh.GetFaceVertexIndicesAttr())): reasons.append("time_sampled_mesh")
            reasons.extend(_material_reasons(prim))
        cursor = prim
        while cursor:
            if cursor.IsA(UsdGeom.Xformable) and UsdGeom.Xformable(cursor).TransformMightBeTimeVarying():
                reasons.append("time_varying_transform")
            cursor = cursor.GetParent()
        meshes.append(Mesh(key, component["type"], str(prim.GetPath()), frozen_array(world.min(axis=0)),
            frozen_array(world.max(axis=0)), ta, tb, face_ids, faces, tuple(sorted(set(reasons)))))
    used = {str(Path(layer.realPath).resolve()) for layer in stage.GetUsedLayers() if layer.realPath}
    require(used <= pins.keys(), "USD resolved an unbound layer")
    require(stage.GetRootLayer().anonymous and all(not layer.dirty for layer in stage.GetUsedLayers() if layer.realPath),
            "Source USD layer unexpectedly modified")
    require(meshes, "No generated plant geometry")
    provenance = dict(schema=SCHEMA, directory=str(directory), plant_to_world=matrix.tolist(),
        bindings=pins, usd_version=list(Usd.GetVersion()), policy=POLICY,
        mesh_inventory=[dict(component=m.component, path=m.prim_path, organ=m.organ,
            triangles=len(m.triangles_a), unknown_reasons=list(m.unknown_reasons)) for m in meshes])
    snapshot = Snapshot(tuple(meshes), canonical(report["components"]), matrix, tuple(sorted(pins.items())), canonical(provenance))
    snapshot.finish()
    return snapshot, dict(build_wall_seconds=time.perf_counter()-started, meshes=len(meshes),
        triangles_per_diagonal=sum(len(m.triangles_a) for m in meshes),
        immutable_array_bytes=sum(a.nbytes for m in meshes for a in
            (m.lower, m.upper, m.triangles_a, m.triangles_b, m.face_ids)),
        unsupported_meshes=sum(bool(m.unknown_reasons) for m in meshes),
        unknown_mesh_reason_counts=dict(Counter(r for m in meshes for r in m.unknown_reasons)),
        usd_version=list(Usd.GetVersion()))


def _pixel_ray(calibration, pixel):
    k = np.asarray(calibration["intrinsics"], float)
    transform = np.asarray(calibration["camera_to_world_usd_row_vectors"], float)
    optical = np.linalg.solve(k, [*pixel, 1.])
    require(optical[2] > 0, "Camera ray behind sensor")
    direction = (optical * [1, -1, -1]) @ transform[:3, :3]
    length = float(np.linalg.norm(direction)); direction /= length
    near, far = np.asarray(calibration["clipping_range_m"], float) * length / optical[2]
    return transform[3, :3], direction, float(near), float(far)


def _ray(snapshot, origin, direction, near, far, allowed):
    boxes, entry = ray_boxes([m.lower for m in snapshot.meshes], [m.upper for m in snapshot.meshes],
                             origin, direction, near, far)
    tested, unknown, hits = 0, [], []
    for i in np.flatnonzero(boxes):
        mesh = snapshot.meshes[i]
        tested += len(mesh.triangles_a) + len(mesh.triangles_b)
        possible, certain = mesh_hits(mesh, origin, direction, near, far)
        if mesh.unknown_reasons:
            unknown.append(dict(component=mesh.component, prim_path=mesh.prim_path,
                reasons=list(mesh.unknown_reasons), bound_entry_m=float(entry[i])))
        elif np.isfinite(possible):
            hits.append((mesh, possible, certain))
    allowed_hits = [(mesh, p, c) for mesh, p, c in hits if mesh.component in allowed]
    blockers = [(mesh, p, c) for mesh, p, c in hits if mesh.component not in allowed and np.isfinite(c)]
    metrics = dict(aabb_candidates=int(boxes.sum()), triangle_tests=tested,
                   fruit_bound_candidates=int(sum(boxes[i] and m.organ == "fruit" for i, m in enumerate(snapshot.meshes))))
    if not allowed_hits or not any(np.isfinite(c) for _, _, c in allowed_hits):
        return dict(status="unknown", reasons=["allowed_surface_missing_or_tessellation_edge"],
                    unknown_meshes=unknown, **metrics)
    front = min(p for _, p, _ in allowed_hits)
    # Unknown material/shape in front remains unknown, not a guessed opaque blocker.
    nearby_unknown = [u for u in unknown if u["bound_entry_m"] <= front]
    if nearby_unknown:
        return dict(status="unknown", reasons=["unsupported_foreground_or_allowed_mesh"],
                    unknown_meshes=nearby_unknown, **metrics)
    blockers = [(mesh, p, c) for mesh, p, c in blockers if c < front-POLICY["numerical_ray_epsilon_m"]]
    if blockers:
        mesh, possible, certain = min(blockers, key=lambda row: row[2])
        return dict(status="predicted_blocked", blocker_component=mesh.component, blocker_prim_path=mesh.prim_path,
            blocker_organ=mesh.organ, blocker_ray_distance_upper_m=certain, allowed_ray_distance_lower_m=front,
            distances_are="geometry_ray_parameters_not_native_optical_z", unknown_meshes=[], **metrics)
    uncertain_hits = [mesh.prim_path for mesh, p, c in hits if mesh.component not in allowed and p < front
                      and not np.isfinite(c)]
    return dict(status="unknown" if uncertain_hits else "no_blocker_found_in_partial_model",
        reasons=["tessellation_or_edge_disagreement"] if uncertain_hits else ["full_scene_coverage_unavailable"],
        uncertain_hit_paths=uncertain_hits, unknown_meshes=[], **metrics)


def parent_proxy_gap(point, parent, plant_to_world):
    """Point-to-capsule surface proxy only; never used to exclude ray candidates."""
    values = []
    for chain in parent["capsules_local_m"]:
        for a, b in zip(chain, chain[1:]):
            ends = transform_points(np.asarray([a[:3], b[:3]]) + parent["translation_plant_m"], plant_to_world)
            delta = ends[1]-ends[0]; length2 = float(delta @ delta)
            if length2 <= 1e-16: continue
            t = float(np.clip((point-ends[0]) @ delta / length2, 0, 1))
            values.append(float(np.linalg.norm(point-(ends[0]+t*delta))-(a[3]+t*(b[3]-a[3]))))
    return min(values) if values else None


def inspect_interval(snapshot, calibration, component_id):
    """Only geometry+calibration enter. Never accept RGB/depth/labels/decisions."""
    started = time.perf_counter()
    require(calibration["resolution"] == [1696, 816] and calibration.get("crop_resize") is None,
            "Unchanged native camera dimensions required")
    transform = np.asarray(calibration["camera_to_world_usd_row_vectors"], float)
    require(transform.shape == (4, 4) and np.isfinite(transform).all()
            and np.allclose(transform[:3, :3] @ transform[:3, :3].T, np.eye(3), atol=1e-7, rtol=0),
            "Rigid camera pose required")
    components = json.loads(snapshot.components_json)
    component = components[component_id]; parent = components[component["parent"]]
    chain, lengths, _, _ = _oriented_chain(component, 1e-6)
    groups = [("interval", interval_arc_samples(lengths)), ("proximal", np.linspace(.004, .030, 27))]
    rows, cache = {}, {}
    for group, arcs in groups:
        rows[group] = []
        for arc in arcs:
            geometry = _sample(chain, lengths, float(arc), component["translation_plant_m"])
            point = transform_points([geometry["point_plant_m"]], snapshot.plant_to_world)[0]
            projected = project([point], calibration)[0]
            allowed = (component_id, parent["id"]) if group == "proximal" and arc < .008 else (component_id,)
            row = dict(arc_m=float(arc), allowed_components=list(allowed),
                       parent_capsule_gap_m=parent_proxy_gap(point, parent, snapshot.plant_to_world))
            if projected["projection_status"] != "in_frame":
                row.update(status="unknown", reasons=[projected["projection_status"]]); rows[group].append(row); continue
            pixel = tuple(np.floor(projected["pixel_xy"]).astype(int))
            key = (pixel, allowed)
            if key not in cache:
                rays = [_ray(snapshot, *_pixel_ray(calibration, np.asarray(pixel)+offset), allowed)
                        for offset in POLICY["pixel_offsets"]]
                states = {r["status"] for r in rays}
                state = "predicted_blocked" if states == {"predicted_blocked"} else (
                    "no_blocker_found_in_partial_model" if states == {"no_blocker_found_in_partial_model"} else "unknown")
                cache[key] = dict(status=state, rays=rays,
                    reasons=["pixel_ray_disagreement_or_unknown"] if state == "unknown" else [])
            row.update(pixel_xy=list(map(int, pixel)), **cache[key]); rows[group].append(row)
    blocked = any(row["status"] == "predicted_blocked" for row in rows["interval"])
    evidence = dict(schema=SCHEMA, snapshot_sha256=snapshot.fingerprint, calibration_sha256=digest(canonical(calibration)),
        component_id=component_id, status="predicted_blocked" if blocked else "unknown",
        full_scene_visibility="not_established", policy=POLICY, **rows,
        unique_pixel_allowed_sets=len(cache), rays_evaluated=5*len(cache),
        triangle_tests=sum(r["triangle_tests"] for entry in cache.values() for r in entry["rays"]),
        wall_seconds=time.perf_counter()-started, skip_allowed=False, ranking_allowed=False,
        native_depth_reconstructed=False, labels_modified=False, training_approved=False)
    # Force plain bounded JSON types and reject NaN/Infinity in published evidence.
    return json.loads(canonical(evidence))
