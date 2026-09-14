"""Read-only capacity audit for the clear-cut-point scale-up requirement.

Counts recorded source-plan identities, not rendered visibility, accepted images,
independent real plants or executable actions. Never changes a plan or release.
"""
import argparse
from datetime import datetime, timezone
import math
from pathlib import Path

from .dataset_review import read_json, require, write_json
from .depth_preview import sha256

SPLITS = ("train", "validation", "test")
SCENARIOS = {
    "20000_total_provisional_80_10_10": {"train": 16000, "validation": 2000, "test": 2000},
    "20000_train_plus_heldout_provisional": {"train": 20000, "validation": 2000, "test": 2000},
}


def inventory(plan, view_cap=12):
    require(type(view_cap) is int and view_cap == 12, "Use the unchanged clear-cutpoint 12-view cap")
    require(plan.get("schema_version") == "greenhouse.grounding_collection_plan.v1"
            and plan.get("state") == "ready_for_synthetic_grounding_capture", "Recorded grounding plan required")
    require(plan.get("training_dataset_approved") is False, "A collection plan cannot grant approval")
    config = plan["configuration"]
    require(config.get("clear_capture") == "robot_head_close_diffuse_v1"
            and config.get("source_geometry") == "unmodified_native_components",
            "Original clear-capture source profile required")
    assignments = plan["family_assignments"]
    require(assignments and all(s in SPLITS for s in assignments.values()), "Known family splits required")
    jobs = plan["jobs"]
    require(len({j["job_id"] for j in jobs}) == len(jobs), "Duplicate job ID")
    require(len({j["plant_family"] for j in jobs}) == len(jobs)
            and {j["plant_family"] for j in jobs} == set(assignments), "One job per reserved source family")
    audits = plan["selection_audit"]
    require(len({a["family"] for a in audits}) == len(audits)
            and {a["family"] for a in audits} == set(assignments), "One selection audit per family")
    by_family = {a["family"]: a for a in audits}
    records = []
    for job in jobs:
        family = job["plant_family"]
        require(job["split"] == assignments[family], "Job/family split mismatch")
        targets = job["targets"]
        selected = [t["target_id"] for t in targets]
        require(len(set(selected)) == len(selected), "Duplicate selected target")
        require(all(t["source_plant_id"] == family and t["split_group"] == family
                    and t["target_id"].startswith(family + "/") for t in targets), "Target lineage mismatch")
        audit = by_family[family]
        decisions = audit["exclusions"]  # Historical name: includes all native_rows decisions.
        ids = [d["target_id"] for d in decisions]
        require(len(set(ids)) == len(ids), "Duplicate anatomy decision")
        require(all(d["state"] in ("geometry_candidate", "excluded") for d in decisions),
                "Unknown geometry decision")
        geometry = {d["target_id"] for d in decisions if d["state"] == "geometry_candidate"}
        require(all(t.startswith(family + "/") for t in geometry), "Geometry family mismatch")
        require(type(audit["geometry_candidates"]) is int and len(geometry) == audit["geometry_candidates"]
                and type(audit["selected"]) is int and len(selected) == audit["selected"],
                "Recorded anatomy counts disagree with identities")
        require(set(selected) <= geometry, "Selected target lacks geometry-candidate provenance")
        records.append(dict(family=family, split=job["split"], scheduled_targets=len(selected),
                            recorded_geometry_candidates=len(geometry)))
    counts = {}
    for split in SPLITS:
        part = [r for r in records if r["split"] == split]
        selected = sum(r["scheduled_targets"] for r in part)
        all_geometry = sum(r["recorded_geometry_candidates"] for r in part)
        counts[split] = dict(families=len(part), scheduled_targets=selected,
            recorded_geometry_candidates=all_geometry,
            scheduled_image_upper_bound=selected*view_cap,
            all_geometry_image_upper_bound=all_geometry*view_cap)
    return dict(per_split=counts, per_family=records, max_views_per_target=view_cap,
                family_count=len(records),
                scheduled_targets=sum(r["scheduled_targets"] for r in records),
                recorded_geometry_candidates=sum(r["recorded_geometry_candidates"] for r in records),
                scheduled_image_upper_bound=sum(c["scheduled_image_upper_bound"] for c in counts.values()),
                all_geometry_image_upper_bound=sum(c["all_geometry_image_upper_bound"] for c in counts.values()),
                all_geometry_height_reach_visibility_qualified=False,
                repeated_shards_increase_target_capacity=False,
                resolution_increases_source_target_capacity=False,
                cloned_components_are_independent_source_families=False)


def assess(capacity, goals):
    require(set(goals) == set(SPLITS) and all(type(n) is int and n > 0 for n in goals.values()),
            "Positive explicit image goals per split required")
    cap = capacity["max_views_per_target"]
    parts = {}
    for split, requested in goals.items():
        c = capacity["per_split"][split]
        required_targets = math.ceil(requested/cap)
        parts[split] = dict(requested_accepted_images=requested,
            minimum_distinct_targets_at_view_cap=required_targets,
            additional_targets_beyond_schedule=max(0, required_targets-c["scheduled_targets"]),
            additional_targets_beyond_all_recorded_geometry=max(0, required_targets-c["recorded_geometry_candidates"]),
            scheduled_capacity_sufficient=c["scheduled_image_upper_bound"] >= requested,
            all_recorded_geometry_capacity_sufficient=c["all_geometry_image_upper_bound"] >= requested)
    return dict(per_split=parts, requested_total_images=sum(goals.values()),
        minimum_distinct_targets=sum(p["minimum_distinct_targets_at_view_cap"] for p in parts.values()),
        scheduled_capacity_sufficient=all(p["scheduled_capacity_sufficient"] for p in parts.values()),
        all_recorded_geometry_capacity_sufficient=all(p["all_recorded_geometry_capacity_sufficient"] for p in parts.values()),
        acceptance_yield_assumed=None, collection_eta=None,
        full_capture_feasibility_established=False, authorizes_collection=False, authorizes_training=False)


def report(plan_path):
    plan_path = Path(plan_path).resolve()
    plan = read_json(plan_path)
    capacity = inventory(plan)
    return dict(schema_version="greenhouse.clear_scale_capacity.v1",
        created_utc=datetime.now(timezone.utc).isoformat(),
        state="capacity_assessment_not_executable_collection_plan",
        source_plan=str(plan_path), source_plan_sha256=sha256(plan_path),
        evidence_scope="recorded_source_plan_identity_counts_not_new_USD_or_native_visibility_audit",
        source_assets_reaudited=False, user_minimum_training_images=20000,
        whether_20k_means_train_or_all_splits="training_only_user_confirmed",
        active_scenario="20000_train_plus_heldout_provisional",
        heldout_counts_status="2000_validation_and_2000_test_are_proposed_not_user_specified",
        capacity=capacity, scenarios={name: assess(capacity, goals) for name, goals in SCENARIOS.items()},
        changed_files_or_splits=False, training_ready=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(not output.exists(), "New report path required; no overwrite")
    result = report(args.plan)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    print(result["state"], result["capacity"]["scheduled_image_upper_bound"],
          result["capacity"]["all_geometry_image_upper_bound"], flush=True)


if __name__ == "__main__":
    main()
