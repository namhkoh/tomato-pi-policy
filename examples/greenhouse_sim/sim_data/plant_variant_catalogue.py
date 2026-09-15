"""Read-only provenance validation and anonymous-stage inspection adapter.

Does not add variants to existing collection plans or approve training/collision
safety. Failed radial probes remain visible in receipts but never become rows.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path

from .audit import audit_manifest, descendants, safe_asset
from .cut_regions import load_rule, propose_cut_region
from .plant_variants import (VERSION, PHYSICS_FIELDS, digest, file_hash, normalized,
                             require, transformed_component, validate_similarity)


def load_for_inspection(directory, source_plan_path):
    directory, source_plan_path = Path(directory).resolve(), Path(source_plan_path).resolve()
    require(not (directory/"FAILED.json").exists(), "Failed generator output cannot be loaded")
    receipt_path = directory/"qualification.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("version") == "curved_relocated_petiole_static.v1":
        from .procedural_petiole_catalogue import load_for_inspection as load_curved
        return load_curved(directory, source_plan_path)
    require(receipt.get("version") == VERSION
            and receipt.get("state") == "cpu_generated_pending_native_and_visual_review",
            "Unknown or incomplete generator qualification")
    require(receipt["variant_id"] == directory.name, "Variant directory identity mismatch")
    require(all(receipt.get(k) is False for k in ("training_eligible", "physics_validated",
                "native_capture_validated", "review_decisions_inherited", "collection_approved")),
            "Inspection artifacts cannot claim data or physical approval")
    frozen = json.loads((directory.parent/"frozen_lineage.json").read_text(encoding="utf-8"))
    plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
    require(frozen["source_plan_sha256"] == file_hash(source_plan_path)
            and frozen["family_assignments"] == plan["family_assignments"],
            "Frozen source-plan binding or family reservations changed")
    family = receipt["source_family"]
    require(plan["family_assignments"].get(family) == "train"
            and receipt["split"] == "train" and receipt["split_group"] == family,
            "Generated donor/split mismatch")
    job = next(j for j in plan["jobs"] if j["plant_family"] == family)
    source_path = Path(job["source_manifest_path"]).resolve()
    require(source_path == Path(receipt["source_manifest_path"]).resolve(),
            "Unexpected donor manifest")
    pins = plan["source_bindings_sha256"]
    # Verify everything the legacy plan actually binds (currently USD/JSON).
    # Texture baselines are separately bound at generator copy time below.
    family_pins = {p: h for p, h in pins.items() if Path(p).resolve().is_relative_to(source_path.parent)}
    require(family_pins and family_pins.get(str(source_path)) == receipt["source_manifest_sha256"],
            "Missing source-family bindings")
    for path, expected in family_pins.items():
        require(file_hash(path) == expected, "Frozen donor asset or texture changed: " + path)
    for relative, expected in receipt["output_hashes"].items():
        require(file_hash(safe_asset(directory, relative)) == expected, "Generated file changed: " + relative)
    for filename, expected in receipt["implementation_hashes"].items():
        require(file_hash(Path(__file__).with_name(filename)) == expected,
                "Generator qualification code changed; explicitly requalify, do not silently reuse")
    require(receipt["output_hashes"].get("manifest.json") == file_hash(directory/"manifest.json"),
            "Generated manifest must be hash-bound")
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    envelope = json.loads((directory.parent/"training_envelope.json").read_text(encoding="utf-8"))
    require(digest(envelope) == receipt["configuration"]["envelope_sha256"],
            "Training morphology envelope changed")
    generated = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
    report = audit_manifest(directory/"manifest.json")
    original = audit_manifest(source_path)
    require(report["status"] != "blocked" and original["status"] != "blocked", "Structural audit failed")
    require(set(receipt["source_component_hashes"]) == set(original["components"])
            and all(receipt["source_component_hashes"][k] == c["asset_sha256"]
                    for k, c in original["components"].items()), "Donor mesh bindings disagree")
    require(generated.get("physics_supported") is False, "Only static-perception variants supported")
    raw_by_id = {c["id"]: c for c in raw["components"]}
    new_by_id = {c["id"]: c for c in generated["components"]}
    require(set(raw_by_id) == set(new_by_id), "Variant unexpectedly changes component identities")
    require(set(receipt["component_receipts"]) == set(raw_by_id), "Missing component receipt")
    allowed_targets = {t["component_id"] for t in job["targets"]}
    owners = {}
    for change in receipt["changes"]:
        key = change["component_id"]
        require(key in allowed_targets and change["source_target_id"] == family+"/"+key,
                "Target not in frozen source job")
        require(change["target_id"] == directory.name+"/"+key and change["novelty_approved"] is False
                and change["conservative_view_cap_group"] == change["source_target_id"],
                "Target identity/view-cap bypass")
        validate_similarity(change["rotation"], change["scale"])
        require(change["members"] == descendants(original["components"], key),
                "Generated subtree membership changed")
        require(all(original["components"][m]["type"] in ("sub_stem", "leaf") for m in change["members"]),
                "Protected donor organ was transformed")
        for member in change["members"]:
            require(member not in owners, "Overlapping variant subtrees")
            owners[member] = change
    require(len(receipt["changes"]) == receipt["generated_target_count"], "Target count mismatch")
    texture_bindings = {}
    for key, old in raw_by_id.items():
        actual_asset = report["components"][key]
        require(receipt["output_hashes"].get(actual_asset["file"]) == actual_asset["asset_sha256"]
                == receipt["component_receipts"][key]["output_sha256"],
                "Unbound generated mesh: " + key)
        c = owners.get(key)
        expected = (transformed_component(old, c["anchor_plant_m"], c["rotation"], c["scale"]) if c
                    else {k: v for k, v in old.items() if k not in PHYSICS_FIELDS})
        require(new_by_id[key] == expected, "Generated attachment/centerline metadata differs: " + key)
        for relative, sha in receipt["component_receipts"][key]["texture_hashes"].items():
            donor_texture = safe_asset(source_path.parent, relative)
            require(file_hash(donor_texture) == sha
                    and (str(donor_texture) not in family_pins or family_pins[str(donor_texture)] == sha)
                    and receipt["output_hashes"].get(Path(relative).as_posix()) == sha,
                    "Texture differs from its copy-time donor binding")
            texture_bindings[str(donor_texture)] = sha
    probes = {r["component_id"]: r["surface_probe"] for r in receipt["local_surface_diagnostics"]}
    require(set(probes) == {c["component_id"] for c in receipt["changes"]}, "Missing surface-probe receipt")
    from .geometry import audit_geometry
    geometry = audit_geometry(deepcopy(report))
    require(geometry["maximum_translation_error_m"] <= 1e-6, "Component-frame assembly mismatch")
    rows, rejected = [], []
    for change in receipt["changes"]:
        key = change["component_id"]
        component = report["components"][key]
        proposal = propose_cut_region(component, report["components"][component["parent"]], load_rule())
        require(proposal == change["cut_region_proposal"], "Stored cut point is stale")
        reasons = []
        if not probes[key]["passed"]:
            reasons.append("sparse_mesh_surface_probe_failed")
        reasons.extend(proposal["geometry_warnings"])
        # Inherited source defects are not automatically safe to use. Hold any
        # selected subtree with an attachment/bounds warning for native review.
        reasons.extend(sorted({w["code"] for w in geometry["warnings"]
                               if w["component_id"] in change["members"]}))
        if reasons:
            rejected.append(dict(component_id=key, reasons=reasons))
            continue
        rows.append(dict(draft_id="G_"+directory.name+"_"+key, target_id=change["target_id"],
            component_id=key, variant_id=directory.name, source_plant_id=family, split_group=family,
            label_origin="generated_centerline_prototype_not_execution_authority",
            cut_region_proposal=proposal, expected_detached_component_ids=change["members"],
            attachment_plant_m=component["attachment_plant_m"], training_label_approved=False,
            human_review_performed=False, physical_executability="not_tested",
            conservative_view_cap_group=change["conservative_view_cap_group"]))
    return dict(schema_version="greenhouse.generated_inspection_catalogue.v1",
        directory=str(directory), qualification_sha256=file_hash(receipt_path),
        source_plan_sha256=file_hash(source_plan_path), report=report, rows=rows, rejected=rejected,
        geometry=geometry,
        texture_bindings=texture_bindings, texture_binding_origin="generator_copy_time",
        all_textures_previously_bound_by_legacy_plan=all(p in family_pins for p in texture_bindings),
        source_family=family, split_group=family, variant_id=directory.name,
        state="provenance_checked_native_visual_collision_review_pending",
        training_eligible=False, collision_validation="not_tested")


def assemble_for_inspection(stage, root_path, catalogue):
    """A NEW anonymous-stage root only; no edits to source layers or live jobs."""
    from pxr import Sdf, Usd, UsdGeom
    from .geometry import assemble_plant
    require(stage.GetRootLayer().anonymous, "Inspection requires an anonymous stage")
    path = Sdf.Path(root_path)
    require(path.IsAbsolutePath() and path.IsPrimPath() and str(path) != "/"
            and not stage.GetPrimAtPath(path), "New absolute inspection prim path required")
    directory = Path(catalogue["directory"])
    require(file_hash(directory/"qualification.json") == catalogue["qualification_sha256"],
            "Qualification changed since loading")
    # Check assets again at composition, avoiding a stale checked catalogue.
    receipt = json.loads((directory/"qualification.json").read_text(encoding="utf-8"))
    for relative, expected in receipt["output_hashes"].items():
        require(file_hash(safe_asset(directory, relative)) == expected, "Generated asset changed before composition")
    require(audit_manifest(directory/"manifest.json") == catalogue["report"],
            "Inspection catalogue geometry changed after validation")
    paths = assemble_plant(stage, str(path), catalogue["report"])
    root = stage.GetPrimAtPath(path)
    for prim in Usd.PrimRange(root):
        require(not any("physics" in s.lower() or "physx" in s.lower() for s in prim.GetAppliedSchemas()),
                "Generated static inspection unexpectedly contains physics")
    variant = dict(variant_id=catalogue["variant_id"], source_plant_id=catalogue["source_family"],
        split_group=catalogue["split_group"], plant_root=str(path),
        added_components={}, added_component_paths={})
    return dict(variant=variant, record=dict(plant_root=str(path), component_paths=paths),
                rows=deepcopy(catalogue["rows"]), report=catalogue["report"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assemble", action="store_true", help="CPU anonymous-stage check; no renderer")
    args = parser.parse_args()
    require(not args.output.exists(), "Inspection output must be new")
    pilot = json.loads((args.pilot/"pilot.json").read_text(encoding="utf-8"))
    rows = []
    for item in pilot["layouts"]:
        directory = safe_asset(args.pilot, item["variant_id"])
        c = load_for_inspection(directory, args.source_plan)
        record = dict(variant_id=c["variant_id"], source_family=c["source_family"],
            checked_targets=len(c["rows"]), rejected=c["rejected"],
            qualification_sha256=c["qualification_sha256"],
            texture_binding_origin=c["texture_binding_origin"],
            all_textures_previously_bound_by_legacy_plan=c["all_textures_previously_bound_by_legacy_plan"])
        if args.assemble:
            from pxr import Usd, UsdGeom
            from .capture_viewpoints import component_catalogue
            from .capture_scene import target_world_geometry
            stage = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageMetersPerUnit(stage, 1)
            UsdGeom.SetStageUpAxis(stage, "Z")
            assembled = assemble_for_inspection(stage, "/World/GeneratedPlant", c)
            components = component_catalogue(stage, [assembled["record"]], [assembled["report"]],
                                              [assembled["variant"]])
            require(len(components) == len(c["report"]["components"]), "Assembled catalogue lost organs")
            for target in c["rows"]:
                world = target_world_geometry(stage, assembled["variant"], target)
                require(world["nominal_world_m"] == target["cut_region_proposal"]["nominal"]["point_plant_m"],
                        "World projection differs from generated label")
            geometry = c["geometry"]
            require(geometry["maximum_translation_error_m"] <= 1e-6, "Component frame assembly mismatch")
            record.update(assembled_components=len(components),
                maximum_translation_error_m=geometry["maximum_translation_error_m"],
                attachment_aabb_diagnostics=geometry["warnings"],
                aabb_check_is_not_surface_or_collision_validation=True)
        rows.append(record)
        print("INSPECTION_CHECKED", record["variant_id"], record["checked_targets"], flush=True)
    from .plant_variant_usd import write_json_new
    write_json_new(args.output, dict(schema_version="greenhouse.generated_pilot_inspection.v1",
        source_pilot_sha256=file_hash(args.pilot/"pilot.json"),
        source_plan_sha256=file_hash(args.source_plan), layouts=rows,
        implementation_sha256=file_hash(__file__), physics_validated=False,
        native_capture_validated=False, training_eligible=False,
        state="cpu_inspection_checked_native_visual_collision_review_pending"))


if __name__ == "__main__":
    main()
