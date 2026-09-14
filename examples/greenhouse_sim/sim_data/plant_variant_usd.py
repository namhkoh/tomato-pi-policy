"""USD writer/CPU qualification for static plant variants; never launches Isaac.

Material textures, topology and UVs are retained. Mesh points/normals and extents
are baked consistently; inherited physics APIs are removed from the new copies.
Native full-greenhouse visibility and collision qualification remain mandatory.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import shutil

import numpy as np

from .audit import audit_manifest, safe_asset
from .cut_regions import load_rule, propose_cut_region
from .plant_variants import (VERSION, digest, file_hash, load_training_sources, plan_variant,
                             require, training_envelope, unit, validate_similarity)


def write_json_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def mesh_arrays(stage):
    """Read actual authored mesh points, not manifest capsules or stale extents."""
    from pxr import Gf, Usd, UsdGeom
    points, triangles = [], []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        p = np.array(mesh.GetPointsAttr().Get(), dtype=float)
        require(p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all(), "Invalid mesh points")
        counts = list(mesh.GetFaceVertexCountsAttr().Get())
        indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        require(sum(counts) == len(indices) and all(n >= 3 for n in counts)
                and (not indices or (min(indices) >= 0 and max(indices) < len(p))), "Invalid mesh topology")
        offset = 0
        for n in counts:
            face = indices[offset:offset+n]
            # Diagnostic fan triangulation; native renderer tessellation is separate.
            triangles.extend([p[[face[0], face[j], face[j+1]]] for j in range(1, n-1)])
            offset += n
        points.append(p)
    require(bool(points), "Component has no mesh")
    return np.concatenate(points), np.asarray(triangles).reshape(-1, 3, 3)


def first_ray_hit(triangles, origin, direction):
    """Two-sided Moller-Trumbore; diagnostic surface intersection, not contact."""
    tri = np.asarray(triangles)
    if not len(tri):
        return None
    e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    h = np.cross(np.broadcast_to(direction, e2.shape), e2)
    determinant = np.einsum("ij,ij->i", e1, h)
    valid = np.abs(determinant) > 1e-14
    inv = np.zeros_like(determinant)
    inv[valid] = 1/determinant[valid]
    s = np.asarray(origin)-tri[:, 0]
    u = inv*np.einsum("ij,ij->i", s, h)
    q = np.cross(s, e1)
    v = inv*(q @ direction)
    distance = inv*np.einsum("ij,ij->i", e2, q)
    valid &= (u >= -1e-8) & (v >= -1e-8) & (u+v <= 1+1e-8) & (distance > 1e-8)
    return float(np.min(distance[valid])) if valid.any() else None


def cut_surface_probe(stage, component, parent):
    """Radial rays at 10/15/20mm must meet the copied petiole surface.

    A sparse diagnostic only: not watertightness, botanical approval, protected
    organ clearance, cutter access, native visibility, or a collision certificate.
    """
    from .cut_regions import _oriented_chain, _sample
    rule = load_rule()
    proposal = propose_cut_region(component, parent, rule)
    require(proposal["status"] == "proposed_geometry_only", "Invalid generated cut proposal")
    _, tri = mesh_arrays(stage)
    chain, cumulative, _, _ = _oriented_chain(component, rule["anchor_match_tolerance_m"])
    rows = []
    for arc in (.01, .015, .02):
        s = _sample(chain, cumulative, arc, component["translation_plant_m"])
        center = np.array(s["point_plant_m"])-component["translation_plant_m"]
        tangent = unit(s["tangent_plant"])
        basis = np.eye(3)[np.argmin(np.abs(tangent))]
        a = unit(np.cross(tangent, basis))
        b = np.cross(tangent, a)
        hits = [first_ray_hit(tri, center, math.cos(theta)*a+math.sin(theta)*b)
                for theta in np.linspace(0, 2*math.pi, 8, endpoint=False)]
        radius = s["petiole_radius_m"]
        valid = all(hit is not None and .2*radius <= hit <= 3*radius for hit in hits)
        rows.append(dict(arc_m=arc, capsule_radius_m=radius, radial_hits_m=hits, passed=valid))
    return dict(scope="24_radial_rays_on_authored_fan_triangulated_petiole",
                passed=all(r["passed"] for r in rows), samples=rows,
                radial_ratio_bounds=[.2, 3.], thresholds="engineering_diagnostic_not_biology",
                native_render_mesh_verified=False, watertightness_validated=False,
                protected_organ_clearance_validated=False)


def copy_component(source_path, output_path, matrix, scale, source_root, copied_textures):
    from pxr import Gf, Sdf, Usd, UsdGeom, Vt
    r = validate_similarity(matrix, scale)
    source = Usd.Stage.Open(str(source_path))
    require(bool(source) and bool(source.GetDefaultPrim()), "USD default prim required")
    require(UsdGeom.GetStageMetersPerUnit(source) == 1 and UsdGeom.GetStageUpAxis(source) == "Z",
            "Only native meters/Z component assets")
    require(not source.GetRootLayer().subLayerPaths, "Layered component unsupported")
    layer = Sdf.Layer.CreateAnonymous("generated.usda")
    layer.TransferContent(source.GetRootLayer())
    target = Usd.Stage.Open(layer)
    removed, meshes, source_zero_normals, source_faceless_meshes = 0, 0, 0, 0
    dependencies = {}
    for prim in target.Traverse():
        require(not prim.HasAuthoredReferences() and not prim.HasAuthoredPayloads()
                and not prim.IsInstanceable(), "Composed/instanced component unsupported")
        require(prim.GetTypeName() in ("Xform", "Mesh", "Scope", "Material", "Shader", "GeomSubset"),
                "Unsupported component primitive: " + prim.GetTypeName())
        xf = UsdGeom.Xformable(prim)
        if xf:
            require(np.allclose(np.array(xf.GetLocalTransformation()), np.eye(4), atol=1e-12, rtol=0),
                    "Source mesh transforms must be identity")
        schemas = list(prim.GetAppliedSchemas())
        retained = [s for s in schemas if "physics" not in s.lower() and "physx" not in s.lower()]
        removed += len(schemas)-len(retained)
        prim.SetMetadata("apiSchemas", Sdf.TokenListOp.CreateExplicit(retained))
        for attr in list(prim.GetAttributes()):
            name = attr.GetName()
            require(not attr.GetNumTimeSamples(), "Animated component unsupported")
            if name.startswith(("physics:", "physx", "physxSchema")):
                prim.RemoveProperty(name)
                continue
            value = attr.Get()
            if value is None:
                continue
            if attr.GetTypeName() == Sdf.ValueTypeNames.Asset:
                # All dependencies must be inside this donor family.
                relative = value.path
                dependency = safe_asset(source_root, relative)
                require(dependency.is_file(), "Missing texture: " + relative)
                destination = safe_asset(output_path.parent, relative)
                require(destination != output_path, "Texture/component path collision")
                fingerprint = file_hash(dependency)
                if destination not in copied_textures:
                    require(not destination.exists(), "Refuse overwriting a generated dependency")
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(dependency, destination)
                    copied_textures[destination] = fingerprint
                require(copied_textures[destination] == fingerprint, "Conflicting texture dependency")
                dependencies[relative] = fingerprint
            require(attr.GetTypeName() != Sdf.ValueTypeNames.AssetArray, "Asset arrays unsupported")
            # Source exports use points and normals; role-aware handling includes primvars:normals.
            if attr.GetTypeName().role in ("Point", "Normal", "Vector"):
                require(attr.GetTypeName().isArray, "Unsupported scalar spatial attribute")
                array = np.asarray(value, dtype=float)
                require(array.ndim == 2 and array.shape[1] == 3 and np.isfinite(array).all(),
                        "Invalid spatial mesh attribute")
                if name == "extent":
                    continue
                if attr.GetTypeName().role == "Point":
                    transformed = scale*(array @ r.T)
                elif attr.GetTypeName().role == "Normal":
                    lengths = np.linalg.norm(array, axis=1)
                    # Source exports contain zero normals on some degenerate
                    # faces. Preserve those zeros, report them, and never invent
                    # a direction or silently alter source topology to hide it.
                    source_zero_normals += int(np.sum(lengths <= 1e-12))
                    transformed = array @ r.T
                    nonzero = lengths > 1e-12
                    transformed[nonzero] /= lengths[nonzero, None]
                else:
                    raise ValueError("Unqualified vector attribute: " + name)
                attr.Set(Vt.Vec3fArray([Gf.Vec3f(*row) for row in transformed.tolist()]))
            if "tangent" in name.lower():
                raise ValueError("Unqualified authored tangent channel")
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            source_faceless_meshes += int(len(mesh.GetFaceVertexCountsAttr().Get()) == 0)
            p = np.asarray(mesh.GetPointsAttr().Get())
            mesh.CreateExtentAttr().Set(Vt.Vec3fArray([
                Gf.Vec3f(*p.min(axis=0).tolist()), Gf.Vec3f(*p.max(axis=0).tolist())]))
            meshes += 1
    require(meshes > 0, "No supported meshes in component")
    require(not output_path.exists(), "Refuse overwriting generated component")
    require(layer.Export(str(output_path)), "USD export failed")
    # Read back the serialized points to catch wrong USD role/matrix conventions.
    reread = Usd.Stage.Open(str(output_path))
    expected, _ = mesh_arrays(source)
    actual, _ = mesh_arrays(reread)
    require(expected.shape == actual.shape and np.allclose(actual, scale*(expected @ r.T),
            atol=2e-7, rtol=1e-6), "Baked mesh mismatch")
    return dict(meshes=meshes, removed_physics_apis=removed,
                preserved_source_faceless_meshes=source_faceless_meshes,
                preserved_source_zero_normals=source_zero_normals, texture_hashes=dependencies,
                output_sha256=file_hash(output_path))


def write_variant(source, planned, output_root):
    from pxr import Usd
    provenance = deepcopy(planned["provenance"])
    source_path = Path(provenance["source_manifest_path"])
    root = Path(output_root).resolve()
    directory = root / provenance["variant_id"]
    require(not directory.exists(), "Variant output already exists")
    require(not root.is_relative_to(source_path.parent), "Cannot generate inside a source family")
    require(file_hash(source_path) == provenance["source_manifest_sha256"], "Source changed after planning")
    raw_components = source["report"]["components"]
    for key, c in raw_components.items():
        require(file_hash(safe_asset(source_path.parent, c["file"])) ==
                provenance["source_component_hashes"][key], "Source asset changed after planning")
    directory.mkdir(parents=True, exist_ok=False)
    try:
        members = {}
        for change in provenance["changes"]:
            for key in change["members"]:
                require(key not in members, "Overlapping target subtrees")
                members[key] = change
        output_hashes, textures, geometry_records = {}, {}, {}
        for c in planned["manifest"]["components"]:
            key = c["id"]
            change = members.get(key)
            asset = safe_asset(source_path.parent, c["file"])
            output = safe_asset(directory, c["file"])
            output.parent.mkdir(parents=True, exist_ok=True)
            r, s = (change["rotation"], change["scale"]) if change else (np.eye(3), 1.)
            receipt = copy_component(asset, output, r, s, source_path.parent, textures)
            output_hashes[c["file"]] = receipt["output_sha256"]
            geometry_records[key] = receipt
        write_json_new(directory / "manifest.json", planned["manifest"])
        audit = audit_manifest(directory / "manifest.json")
        require(audit["status"] != "blocked", "Generated manifest audit failed")
        diagnostics = []
        for change in provenance["changes"]:
            c = audit["components"][change["component_id"]]
            parent = audit["components"][c["parent"]]
            stage = Usd.Stage.Open(str(safe_asset(directory, c["file"])))
            probe = cut_surface_probe(stage, c, parent)
            recomputed = propose_cut_region(c, parent, load_rule())
            require(recomputed == change["cut_region_proposal"], "Serialized cut labels diverged")
            change["local_mesh_validation"] = ("sparse_surface_probe_passed" if probe["passed"]
                                                else "sparse_surface_probe_failed")
            diagnostics.append(dict(component_id=c["id"], surface_probe=probe))
        # Bind all material dependencies as well as manifest/mesh bytes.
        for path, value in textures.items():
            require(file_hash(path) == value, "Copied texture mismatch")
            output_hashes[path.relative_to(directory).as_posix()] = value
        output_hashes["manifest.json"] = file_hash(directory / "manifest.json")
        for key, c in raw_components.items():
            require(file_hash(safe_asset(source_path.parent, c["file"])) ==
                    provenance["source_component_hashes"][key], "Source changed during generation")
        require(file_hash(source_path) == provenance["source_manifest_sha256"], "Source manifest changed")
        provenance.update(state="cpu_generated_pending_native_and_visual_review",
            output_hashes=output_hashes, usd_version=list(Usd.GetVersion()),
            implementation_hashes={p.name: file_hash(p) for p in (
                Path(__file__), Path(__file__).with_name("plant_variants.py"),
                Path(__file__).with_name("audit.py"), Path(__file__).with_name("cut_regions.py"))},
            structural_audit_status=audit["status"], local_surface_diagnostics=diagnostics,
            surface_probe_pass_count=sum(d["surface_probe"]["passed"] for d in diagnostics),
            component_receipts=geometry_records, source_assets_unchanged=True,
            # A radial pass alone never enables collection or training.
            collection_approved=False, collision_validation="not_tested")
        write_json_new(directory / "qualification.json", provenance)
        return provenance
    except Exception as exc:
        write_json_new(directory / "FAILED.json", dict(state="failed_not_usable", error=str(exc)))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layouts", type=int, default=20)
    parser.add_argument("--targets-per-layout", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260915)
    args = parser.parse_args()
    require(1 <= args.layouts <= 20, "Bounded prototype only: 1..20 layouts")
    require(not args.output.exists(), "Use a NEW output directory; no resume/overwrite")
    plan, sources = load_training_sources(args.source_plan)
    envelope = training_envelope(plan, sources)
    variants = []
    families = sorted(sources)
    for i in range(args.layouts):
        family = families[i % len(families)]
        variants.append((family, plan_variant(sources[family], envelope, args.seed+i, args.targets_per_layout)))
    args.output.mkdir(parents=True, exist_ok=False)
    write_json_new(args.output / "training_envelope.json", envelope)
    write_json_new(args.output / "frozen_lineage.json", dict(
        source_plan_sha256=file_hash(args.source_plan), family_assignments=plan["family_assignments"],
        no_split_reassignment=True, heldout_used_for_fitting=False))
    results = []
    for family, variant in variants:
        print("GENERATING", variant["provenance"]["variant_id"], flush=True)
        r = write_variant(sources[family], variant, args.output)
        results.append({k: r[k] for k in ("variant_id", "source_family", "split_group",
                         "generated_target_count", "surface_probe_pass_count", "state")})
        print("CPU_GENERATED", json.dumps(results[-1]), flush=True)
    all_changes = [c for _, v in variants for c in v["provenance"]["changes"]]
    write_json_new(args.output / "pilot.json", dict(schema_version="greenhouse.plant_variant_pilot.v1",
        generator=VERSION, state="cpu_generated_pending_native_and_visual_review",
        layouts=results, original_donor_families=len({r["source_family"] for r in results}),
        exact_generated_morphology_hashes=len({c["exact_morphology_hash"] for c in all_changes}),
        original_source_targets=len({c["source_target_id"] for c in all_changes}),
        exact_hash_is_not_near_duplicate_or_biological_novelty_approval=True,
        max_views_per_conservative_source_target=12,
        new_independent_source_families=0, accepted_training_images=0,
        source_plan_sha256=file_hash(args.source_plan), training_eligible=False,
        native_capture_started=False, collision_validation="not_tested"))


if __name__ == "__main__":
    main()
