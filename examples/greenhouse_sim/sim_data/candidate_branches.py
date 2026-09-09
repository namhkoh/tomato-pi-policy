"""Reversible, visual-only lower petiole variants of the supplied component plants.

Reuse complete leaf-bearing subtrees at existing deleafed nodes. This does not
generate physical cut joints, approve labels, or alter the source manifests.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

from .audit import PROTECTED_TYPES, audit_manifest, descendants, safe_asset

RECIPE_VERSION = "lower_leaf_petiole_preview.v1"
# Explicit, reproducible attachment sites in the two foreground package plants.
# Donors extend towards +X (the robot aisle). Their complete geometry is kept.
RECIPES = {
    "seed101_full": (("SubStem_35", "SubStem_56"),
                     ("SubStem_37", "SubStem_61"),
                     ("SubStem_39", "SubStem_64")),
    "seed103_full": (("SubStem_36", "SubStem_60"),
                     ("SubStem_39", "SubStem_65"),
                     ("SubStem_41", "SubStem_68")),
}


def _capsule_surface_gap(point, component):
    """Approximate attachment check, NOT a mesh/collision or botanical test."""
    origin = component["translation_plant_m"]
    p = [point[i] - origin[i] for i in range(3)]
    gaps = []
    for chain in component["capsules_local_m"]:
        for a, b in zip(chain, chain[1:]):
            v = [b[i] - a[i] for i in range(3)]
            length2 = sum(x * x for x in v)
            if length2 <= 1e-16:
                continue
            t = max(0.0, min(1.0, sum((p[i] - a[i]) * v[i] for i in range(3)) / length2))
            centre = [a[i] + t * v[i] for i in range(3)]
            radius = a[3] + t * (b[3] - a[3])
            gaps.append(max(0.0, math.dist(p, centre) - radius))
    if not gaps:
        raise ValueError("Attachment requires nondegenerate parent capsule geometry")
    return min(gaps)


def plan_branches(report, count=3):
    """Build an auditable variant; never inherit original human approvals."""
    if type(count) is not int or not 1 <= count <= 3:
        raise ValueError("Request between one and three added branches per plant")
    if report["status"] == "blocked":
        raise ValueError("Cannot augment a structurally blocked plant")
    plant_id = report["plant_id"]
    if plant_id not in RECIPES:
        raise ValueError(f"No reviewed placement recipe for {plant_id}")
    recipe = RECIPES[plant_id][:count]
    components = report["components"]
    signature = {"recipe_version": RECIPE_VERSION, "source_manifest_sha256": report["manifest_sha256"],
                 "source_asset_hashes": {k: c["asset_sha256"] for k, c in sorted(components.items())},
                 "pairs": recipe}
    fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    variant_id = f"{plant_id}+lower_petiole_{fingerprint[:12]}"
    result = {
        "schema_version": "greenhouse.candidate_branch_variant.v1", "recipe_version": RECIPE_VERSION,
        "variant_id": variant_id, "variant_sha256": fingerprint,
        "source_plant_id": plant_id, "split_group": plant_id,
        "source_manifest_path": report["manifest_path"], "source_manifest_sha256": report["manifest_sha256"],
        "scope": "visual_candidate_preview_only", "source_assets_modified": False,
        "physics_enabled": False, "annotation_workflow_supported": False,
        "collision_validation": "not_tested", "added_components": {}, "candidates": [],
        "replaced_stub_ids": [],
    }
    for index, (site_id, donor_id) in enumerate(recipe, 1):
        if site_id not in components or donor_id not in components:
            raise ValueError("Placement recipe does not match source component IDs")
        site, donor = components[site_id], components[donor_id]
        parent = components.get(site["parent"], {})
        if (site["type"] != "sub_stem" or site["deleafed"] is not True
                or parent.get("type") != "main_stem" or descendants(components, site_id) != [site_id]):
            raise ValueError(f"Cannot replace nonempty or non-deleafed node {site_id}")
        members = descendants(components, donor_id)
        kinds = Counter(components[key]["type"] for key in members)
        if (donor["type"] != "sub_stem" or donor["deleafed"] is not False
                or components.get(donor["parent"], {}).get("type") != "main_stem"
                or not kinds["leaf"] or any(k in PROTECTED_TYPES for k in kinds)):
            raise ValueError(f"Donor {donor_id} is not an intact, leaf-only petiole subtree")
        if donor["axis_plant"][0] < 0.5:
            raise ValueError("Donor must extend towards the robot-side aisle (+X)")
        if math.dist(donor["translation_plant_m"], donor["attachment_plant_m"]) > 1e-6:
            raise ValueError("Donor origin does not match its attachment")
        gap = _capsule_surface_gap(site["attachment_plant_m"], parent)
        if gap > 0.002:
            raise ValueError(f"Attachment {site_id} is separated from its parent capsule by {gap:.6f} m")
        delta = [site["attachment_plant_m"][i] - donor["attachment_plant_m"][i] for i in range(3)]
        ids = {key: f"Added{index:02d}_{key}" for key in members}
        if any(key in components or key in result["added_components"] for key in ids.values()):
            raise ValueError("Added component identity collision")
        for key in members:
            added = deepcopy(components[key])
            added.update(id=ids[key], parent=site["parent"] if key == donor_id else ids[added["parent"]],
                         source_component_id=key, source_plant_id=plant_id)
            for field in ("translation_plant_m", "attachment_plant_m"):
                added[field] = [added[field][i] + delta[i] for i in range(3)]
            result["added_components"][ids[key]] = added
        result["replaced_stub_ids"].append(site_id)
        result["candidates"].append({
            "target_id": f"{variant_id}/{ids[donor_id]}", "component_id": ids[donor_id],
            "parent_component_id": site["parent"], "replaced_stub_id": site_id,
            "source_donor_id": donor_id, "state": "attached", "status": "needs_review",
            "translation_from_donor_m": delta, "attachment_plant_m": list(site["attachment_plant_m"]),
            "axis_plant": list(donor["axis_plant"]), "parent_capsule_gap_m": gap,
            "expected_detached_component_ids": sorted(ids.values()), "descendant_type_counts": dict(kinds),
            "protected_descendant_ids": [], "anatomy_review": "pending",
            "agronomic_eligibility": "unreviewed", "observability": "not_measured",
            "physical_executability": "not_tested", "canonical_cut_point_m": None,
            "admissible_cut_region": None, "grasp_region": None,
            "review_notes": "New synthetic branch. Attachment is not an approved cut point. No source review inherited.",
        })
    return result


def add_candidate_branches(stage, record, count=3):
    """Apply a validated recipe ONLY in the current USD session layer."""
    from pxr import Gf, Usd, UsdGeom

    report = audit_manifest(record["manifest_path"])
    variant = plan_branches(report, count)
    source_paths = record["component_paths"]
    all_components = {**report["components"], **variant["added_components"]}
    paths = dict(source_paths)
    pending = set(variant["added_components"])
    order = []
    # Resolve/validate everything before changing the stage.
    while pending:
        ready = sorted(key for key in pending if all_components[key]["parent"] in paths)
        if not ready:
            raise ValueError("Unresolvable added branch graph")
        for key in ready:
            paths[key] = f"{paths[all_components[key]['parent']]}/{key}"
            if stage.GetPrimAtPath(paths[key]):
                raise ValueError("Added branch already exists; refuse duplicate augmentation")
            order.append(key)
            pending.remove(key)
    for candidate in variant["candidates"]:
        for key in (candidate["replaced_stub_id"], candidate["parent_component_id"]):
            prim = stage.GetPrimAtPath(source_paths[key])
            if not prim or not prim.IsActive():
                raise ValueError(f"Source attachment prim is missing/inactive: {key}")
    directory = Path(report["manifest_path"]).parent
    assets = {key: safe_asset(directory, all_components[key]["file"]) for key in order}
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        for key in order:
            c = all_components[key]
            prim = UsdGeom.Xform.Define(stage, paths[key])
            prim.GetPrim().GetReferences().AddReference(assets[key].as_posix())
            offset = Gf.Vec3d(*c["translation_plant_m"]) - Gf.Vec3d(*all_components[c["parent"]]["translation_plant_m"])
            prim.AddTranslateOp().Set(offset)
            prim.GetPrim().SetCustomDataByKey("candidate_variant", variant["variant_id"])
            prim.GetPrim().SetCustomDataByKey("source_component", c["source_component_id"])
        for key in variant["replaced_stub_ids"]:
            stage.GetPrimAtPath(source_paths[key]).SetActive(False)
    variant["plant_root"] = record["plant_root"]
    variant["added_component_paths"] = {key: paths[key] for key in order}
    root_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(record["plant_root"]))
    variant["plant_world_matrix_usd_row_vectors"] = [list(row) for row in root_matrix]
    for candidate in variant["candidates"]:
        candidate["attachment_world_m"] = list(root_matrix.Transform(Gf.Vec3d(*candidate["attachment_plant_m"])))
    return variant
