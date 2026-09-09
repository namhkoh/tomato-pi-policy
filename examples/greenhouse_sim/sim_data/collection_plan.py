"""Deterministic, source-bound multi-plant pilot schedule; no USD/Kit imports.

This selects existing native petioles, not new branches or hidden-leaf edits.
Splits are reserved by target source family. Shared greenhouse/backdrop context
is NOT held out; this is not an environment-disjoint benchmark or training data.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from .audit import DEFAULT_PACK, audit_manifest
from .cut_regions import load_rule, propose_cut_region, rule_fingerprint
from .dataset_review import read_json, require, verify_bindings, write_json
from .depth_preview import sha256

SCHEMA = "greenhouse.native_collection_plan.v1"
REVIEWED_FAMILIES = ("seed101_full", "seed103_full")


def configuration(plants=2, targets=2, views=2, seed=0):
    for value, maximum in ((plants, 4), (targets, 3), (views, 3)):
        require(type(value) is int and 1 <= value <= maximum, "Invalid bounded pilot size")
    require(type(seed) is int and 0 <= seed <= 10000, "Invalid sampling seed")
    return {"plants": plants, "targets_per_plant": targets, "render_views_per_target": views,
            "seed": seed, "target_world_height_m": [1.05, 1.70],
            "original_plant_root_z_m": .90, "reviewed_families_train_only": list(REVIEWED_FAMILIES),
            "collect_split": "train", "source_geometry": "unmodified_native_components"}


def family_splits(families, seed):
    names = sorted(set(families))
    require(len(names) == len(families) and len(names) >= 6, "Need unique independent source families")
    # Previously inspected development families can never enter held-out groups.
    fresh = [n for n in names if n not in REVIEWED_FAMILIES]
    ranked = sorted(fresh, key=lambda n: (hashlib.sha256(f"{seed}:{n}".encode()).hexdigest(), n))
    holdout = min(4, max(1, len(names)//6))
    require(len(ranked) > 2*holdout, "Insufficient unreviewed source families")
    return {n: "test" if n in ranked[:holdout] else "validation" if n in ranked[holdout:2*holdout]
            else "train" for n in names}


def native_rows(report, rule):
    """Reconstruct labels and explicit exclusions without any image-based approval."""
    rows, decisions = [], []
    for target in report["targets"]:
        key = target["component_id"]
        component = report["components"][key]
        reasons = [] if target["status"] == "needs_review" else list(target["reason_codes"])
        relevant = set(target["expected_detached_component_ids"]) | {component["parent"]}
        reasons += [i["code"] for i in report["issues"]
                    if i["severity"] == "error" or i.get("component_id") in relevant]
        proposal = None
        if not reasons:
            proposal = propose_cut_region(component, report["components"][component["parent"]], rule)
            reasons += proposal["reason_codes"] + proposal["geometry_warnings"]
            if proposal["status"] != "proposed_geometry_only":
                reasons.append("no_valid_cut_proposal")
        decisions.append({"target_id": target["target_id"], "state": "excluded" if reasons else "geometry_candidate",
                          "reasons": sorted(set(reasons))})
        if reasons:
            continue
        rows.append({"draft_id": f"N_{report['plant_id']}_{key}", "target_id": target["target_id"],
            "component_id": key, "variant_id": report["plant_id"], "source_plant_id": report["plant_id"],
            "split_group": report["plant_id"], "cut_region_proposal": proposal,
            "source_manifest_sha256": report["manifest_sha256"],
            "attachment_plant_m": target["attachment_plant_m"],
            "expected_detached_component_ids": target["expected_detached_component_ids"],
            "protected_descendant_ids": target["protected_descendant_ids"],
            "label_origin": "unmodified_source_manifest_prototype_rule",
            "human_review_performed": False, "training_label_approved": False,
            "agronomic_eligibility": "unreviewed", "physical_executability": "not_tested"})
    return rows, decisions


def schedule(reports, rule, config):
    expected = configuration(config["plants"], config["targets_per_plant"], config["render_views_per_target"], config["seed"])
    require(config == expected, "Altered collection configuration")
    assignments = family_splits([r["plant_id"] for r in reports], config["seed"])
    jobs, decisions = [], []
    for report in sorted(reports, key=lambda r: r["plant_id"]):
        rows, excluded = native_rows(report, rule)
        low, high = config["target_world_height_m"]
        selected = []
        for row in rows:
            z = row["cut_region_proposal"]["nominal"]["point_plant_m"][2] + config["original_plant_root_z_m"]
            if low <= z <= high:
                selected.append(row)
            else:
                excluded.append({"target_id": row["target_id"], "state": "outside_initial_height_band",
                                 "height_m": z, "not_an_unreachable_or_ineligible_label": True})
        selected.sort(key=lambda r: (abs(r["cut_region_proposal"]["nominal"]["point_plant_m"][2]+.9-1.4), r["target_id"]))
        decisions.append({"family": report["plant_id"], "split": assignments[report["plant_id"]],
                          "geometry_candidates": len(rows), "in_initial_height_band": len(selected),
                          "target_decisions": excluded})
        if (selected and assignments[report["plant_id"]] == "train" and report["plant_id"] not in REVIEWED_FAMILIES
                and len(jobs) < config["plants"]):
            jobs.append({"job_id": f"job_{len(jobs)+1:03d}", "plant_family": report["plant_id"],
                "split": "train", "source_manifest_path": report["manifest_path"],
                "targets": selected[:config["targets_per_plant"]],
                "max_rendered_views_per_target": config["render_views_per_target"]})
    require(len(jobs) == config["plants"], "Insufficient native-target families for the bounded pilot")
    return {"family_assignments": assignments, "jobs": jobs, "selection_audit": decisions}


def source_reports(package):
    paths = sorted((Path(package)/"plants/components").glob("*/manifest.json"))
    require(bool(paths), "No source plant manifests")
    return [audit_manifest(p) for p in paths]


def bindings_for(package, reports):
    bindings = {}
    for report in reports:
        bindings[report["manifest_path"]] = report["manifest_sha256"]
        for component in report["components"].values():
            bindings[str((Path(report["manifest_path"]).parent/component["file"]).resolve())] = component["asset_sha256"]
    # Record the shared scene too; no source USD may be edited by collection.
    for folder in (Path(package)/"house", Path(package)/"plants/backdrop"):
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".usd", ".usda", ".usdc"}:
                bindings[str(path.resolve())] = sha256(path)
    return bindings


def build_plan(package, output, config):
    package, output = Path(package).resolve(), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(package), "Choose a new plan outside source assets")
    reports, rule = source_reports(package), load_rule()
    core = schedule(reports, rule, config)
    bindings = bindings_for(package, reports)
    verify_bindings(bindings)
    result = {"schema_version": SCHEMA, "state": "ready_for_bounded_native_capture",
        "created_utc": datetime.now(timezone.utc).isoformat(), "package": str(package),
        "configuration": config, "cut_rule": rule, "cut_rule_sha256": rule_fingerprint(rule),
        "source_bindings_sha256": bindings, **core,
        "split_scope": "target_source_families_only_shared_greenhouse_backdrop_context",
        "held_out_families_rendered_by_this_schedule": False,
        "difficulty": "unassigned", "training_dataset_approved": False,
        "camera_requirement": "mounted_RBY1_A_v1.2_head_D405_uncropped_848x408",
        "implementation_sha256": sha256(Path(__file__))}
    output.mkdir(parents=True)
    write_json(output/"plan.json", result)
    return result


def load_plan(path):
    plan = read_json(path)
    if plan.get("schema_version") == "greenhouse.grounding_collection_plan.v1":
        from .training_plan import load_plan as load_grounding
        return load_grounding(path)
    require(plan.get("schema_version") == SCHEMA and plan.get("state") == "ready_for_bounded_native_capture",
            "Invalid collection plan")
    require(plan.get("training_dataset_approved") is False and plan.get("difficulty") == "unassigned",
            "Collection plan cannot grant approval")
    require(plan.get("split_scope") == "target_source_families_only_shared_greenhouse_backdrop_context"
            and plan.get("held_out_families_rendered_by_this_schedule") is False, "Unsupported split scope")
    require(rule_fingerprint(plan["cut_rule"]) == plan["cut_rule_sha256"], "Cut rule changed")
    reports = source_reports(plan["package"])
    require(bindings_for(plan["package"], reports) == plan["source_bindings_sha256"], "Collection source assets changed")
    current = schedule(reports, plan["cut_rule"], plan["configuration"])
    require(all(plan.get(k) == v for k, v in current.items()), "Altered collection targets or family splits")
    verify_bindings(plan["source_bindings_sha256"])
    return plan, reports


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plants", type=int, default=2)
    parser.add_argument("--targets", type=int, default=2)
    parser.add_argument("--views", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    result = build_plan(args.package, args.output, configuration(args.plants, args.targets, args.views, args.seed))
    print(json.dumps({"state": result["state"], "jobs": [{k:v for k,v in j.items() if k!='targets'} for j in result["jobs"]],
                      "targets": [r["target_id"] for j in result["jobs"] for r in j["targets"]]}), flush=True)


if __name__ == "__main__":
    main()
