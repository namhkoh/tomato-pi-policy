"""Static receipt traversal adapted from v3 reference_bank.build only.

Upstream file SHA256:
9eb87c8c0a16599201f20cea1819c7c1eb003be602542cce255dc72b0dade94d
All evidence/ranking/pose/compatibility validators remain unchanged imports.
Only the reader is injected and path guards route through it. No dynamic code
rebinding, source rewriting, exec, or global _Reader monkeypatching is used.
The AST equivalence test enumerates the small intentional traversal changes.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import re

from ..native_capture_v3.reference_bank import (
    SCHEMA, MAX_ROWS, _require, _unique, _one, _parse, _pose_evidence,
    ranking_evidence, _compatibility, _digest, rank_key, _tie, _camera_distance,
)


def replay(checkpoint, *, manifest_sha256, reader):
    """Read-only, fail-closed checkpoint adapter with an explicit root trust pin.

    SHA256 proves equality to caller-pinned receipts, not signer identity. Existing
    reviews are copied exactly, never fabricated, upgraded or silently overridden.
    All original source bindings in selected audits/plans are byte-verified. This
    can involve substantial disk reads, but never native jobs or image processing.
    """
    checkpoint = reader.resolve(checkpoint)
    draft = checkpoint / "draft"
    release = reader.json(draft / "manifest.json", manifest_sha256)
    _require(release.get("schema_version") == "greenhouse.clear_cutpoint_release.v1", "Unsupported checkpoint schema")
    hashes = release["files_sha256"]
    source = reader.json(draft / "source_manifest.json", hashes["source_manifest.json"])
    _require(release["source_manifest_copy_sha256"] == hashes["source_manifest.json"], "Source manifest pin differs")
    source_pins = source["source_bindings_sha256"]
    reviews = reader.json(draft / "reviews.json", hashes["reviews.json"])
    reader.json(checkpoint / "accepts.json", hashes["reviews.json"])
    accepted = _unique(reviews, "id", "review identity")
    negatives = reader.json(draft / "visual_exclusions.json", hashes["visual_exclusions.json"])
    # The checkpoint's bound negative history must never be resurrected by RGB duplication.
    _require(isinstance(negatives, list), "Unsupported exclusion history")
    held = {r["rgb_sha256"] for r in negatives if "rgb_sha256" in r}
    held_ids = {r["id"] for r in negatives if "id" in r}
    index_path = reader.bind(draft / "index.jsonl", hashes["index.jsonl"])
    rows = []
    with index_path.open("rb") as stream:
        while line := stream.readline(1024*1024 + 1):
            _require(len(line) <= 1024*1024 and len(rows) < MAX_ROWS, "Index size limit")
            rows.append(_parse(line))
    _unique(rows, "id", "index identity")
    # Resolve audit and sample hashes through the original export's bindings,
    # not scans of unrelated directories or nearest optical depth.
    by_hash = defaultdict(list)
    for path, digest in source_pins.items():
        if Path(path).name in ("audit.json", "sample.json"):
            by_hash[digest].append(path)
    entries, groups, skipped = [], {}, Counter()
    family_splits = {}
    for row in sorted(rows, key=lambda r: r["id"]):
        family, split = row["source_plant_family"], row["split"]
        _require(split in ("train", "validation", "test"), "Unknown split")
        _require(family_splits.setdefault(family, split) == split, "Conflicting donor split")
        if split != "train":
            skipped["heldout_split"] += 1
            continue
        _require(row["id"] not in held_ids and row["rgb_sha256"] not in held, "Reference has negative review history")
        review = accepted.get(row["id"], {})
        _require(review.get("decision") == "accept" and review.get("rgb_sha256") == row["rgb_sha256"]
                 and review.get("reviewer") and review.get("reason"), "Missing matching reviewed accept")
        audit_paths = by_hash.get(row["source_audit_sha256"], [])
        _require(len(audit_paths) == 1, "Missing/ambiguous bound source audit")
        audit_path = Path(audit_paths[0])
        audit = reader.json(audit_path, row["source_audit_sha256"])
        _require(audit.get("state") == "complete_engineering_audit_not_approval", "Incomplete source audit")
        reader.bindings_from(audit["bindings_sha256"])
        capture = reader.resolve(audit["source_run"])
        manifest_path = capture / "manifest.json"
        capture_hash = audit["bindings_sha256"].get(str(manifest_path))
        _require(source_pins.get(str(manifest_path)) == capture_hash, "Unbound original capture")
        manifest = reader.json(manifest_path, capture_hash)
        _require(manifest.get("state") == "pilot_ready_for_review" and
                 manifest.get("source_assets_unchanged") is True and
                 manifest.get("source_geometry_modified") is False, "Not a completed original capture")
        _require(manifest.get("target_family_split") == "train", "Capture split is not TRAIN")
        plan_path = Path(manifest["source_collection_plan_path"])
        plan_hash = manifest["source_collection_plan_sha256"]
        _require(audit["bindings_sha256"].get(str(plan_path)) == plan_hash and
                 source["source_plans_sha256"].get(str(plan_path)) == plan_hash, "Original collection plan unbound")
        plan = reader.json(plan_path, plan_hash)
        reader.bindings_from(plan["source_bindings_sha256"])
        reader.bindings_from(manifest["source_usd_sha256"])
        job = _one(plan["jobs"], lambda j: j["job_id"] == manifest["collection_job_id"], "source job")
        _require(job["plant_family"] == family and job["split"] == plan["family_assignments"].get(family) == "train",
                 "Frozen plan donor split mismatch")
        sample_paths = [Path(p) for p in by_hash.get(row["source_sample_sha256"], [])
                        if reader.resolve(Path(p).parent.parent) == capture]
        _require(len(sample_paths) == 1, "Missing/ambiguous original sample")
        sample_path = sample_paths[0]
        _require(audit["bindings_sha256"].get(str(sample_path)) == row["source_sample_sha256"], "Audit sample hash mismatch")
        sample = reader.json(sample_path, row["source_sample_sha256"])
        sid = sample["sample_id"]
        _require(re.fullmatch(r"sample_[0-9]+", sid) and sample_path.parent.name == sid, "Invalid original sample ID")
        recorded = _one(manifest["samples"], lambda s: s["sample_id"] == sid, "capture sample")
        audited = _one(audit["samples"], lambda s: s["sample_id"] == sid, "audited sample")
        sup = sample["supervision"]
        target = _one(job["targets"], lambda t: t["target_id"] == row["target_id"], "plan target")
        _require(sup["target_id"] == target["target_id"] == family + "/" + target["component_id"] and
                 sup["split_group"] == target["split_group"] == target["source_plant_id"] == family and
                 sup["variant_id"] == target["variant_id"] == family, "Non-original or mismatched target")
        _require(sup["cut_region_proposal"] == target["cut_region_proposal"] and
                 sup["review_id"] == target["draft_id"] == recorded["target_review_id"] == audited["target_review_id"] and
                 audited["source_plant_family"] == family, "Source anatomy/review mismatch")
        original = _one(manifest["variants"], lambda v: v["source_plant_id"] == family, "original plant")
        _require(original["variant_id"] == family and original["added_components"] in ({}, []),
                 "Generated donor is not an original")
        sync = sample.get("synchronization", {})
        _require(sync.get("scene_unchanged_during_capture") is True and sync.get("dynamic_recording_supported") is False,
                 "Unverified static source")
        _pose_evidence(sample, audited)
        for name, receipt in sample["files"].items():
            path = reader.safe(sample_path.parent, name)
            _require(audit["bindings_sha256"].get(str(path)) == receipt["sha256"], "Unbound sample payload")
            reader.bind(path, receipt["sha256"])
        _require(sample["files"]["inputs/rgb.png"]["sha256"] == row["rgb_sha256"], "Reviewed RGB mismatch")
        label_name = row["files"]["label"]
        label = reader.json(reader.safe(draft, label_name), hashes[label_name])
        _require(label.get("eligible") is True and label.get("target_id") == row["target_id"] and
                 label.get("source_plant_family") == family and label.get("answer") == row["answer"] and
                 label.get("query_pixel_uv") == row["query_pixel_uv"] and
                 label.get("contract_sha256") == row["task_contract_sha256"], "Bound label/index mismatch")
        trace = None
        trace_path = sample_path.parent / "supervision/query_trace.json"
        trace_hash = audit["bindings_sha256"].get(str(trace_path))
        if trace_hash is not None:
            trace = reader.json(trace_path, trace_hash)
        entry = dict(id=row["id"], source_family=family, target_id=row["target_id"], split="train",
            conservative_view_cap_group=row["target_id"], source_collection_plan=str(plan_path),
            source_collection_plan_sha256=plan_hash, source_capture=str(capture),
            source_capture_manifest_sha256=capture_hash, source_sample=sid,
            source_sample_path=str(sample_path), source_sample_sha256=row["source_sample_sha256"],
            source_audit=str(audit_path), source_audit_sha256=row["source_audit_sha256"],
            rgb_sha256=row["rgb_sha256"], source_label=str(reader.safe(draft, label_name)),
            source_label_sha256=hashes[label_name], review=deepcopy(review),
            review_file=str(draft / "reviews.json"), review_file_sha256=hashes["reviews.json"],
            source_row=deepcopy(target), calibration=deepcopy(sample["calibration"]),
            robot_snapshot=deepcopy(sample["robot_snapshot"]),
            evidence=ranking_evidence(row, label, audited, sample["calibration"]["resolution"], trace),
            strict_trace_path=str(trace_path) if trace_hash else None, strict_trace_sha256=trace_hash,
            reference_role="historical_848_pose_prior" if sample["calibration"]["resolution"] == [848, 408]
                           else "existing_native_original_pose_prior",
            new_native_training_rows=0, training_approved=False, source_cap_reset=False,
            new_geometry_and_native_qualification_required=True)
        group = _compatibility(entry, manifest, original)
        entry["compatibility_group"] = _digest(group)
        groups[entry["compatibility_group"]] = group
        entries.append(entry)
    # Retain compatible alternatives across original plans, but don't treat
    # repeated sample/RGB/camera metadata inside one target group as new views.
    kept, seen_samples, seen_rgb = [], set(), set()
    for entry in sorted(entries, key=lambda e: (rank_key(e), _tie(e))):
        scope = (entry["compatibility_group"], entry["target_id"])
        if ((scope, entry["source_sample_sha256"]) in seen_samples or
            (scope, entry["rgb_sha256"]) in seen_rgb or
            any((e["compatibility_group"], e["target_id"]) == scope and
                _camera_distance(e, entry) <= 1e-6 for e in kept)):
            skipped["duplicate_reference_sample_rgb_or_camera_pose"] += 1
            continue
        kept.append(entry)
        seen_samples.add((scope, entry["source_sample_sha256"]))
        seen_rgb.add((scope, entry["rgb_sha256"]))
    unique_views = []
    for entry in sorted(kept, key=_tie):
        if not any(e["target_id"] == entry["target_id"] and _camera_distance(e, entry) <= 1e-6 for e in unique_views):
            unique_views.append(entry)
    donors = {family: dict(references=sum(e["source_family"] == family for e in kept),
                           targets=sorted({e["target_id"] for e in kept if e["source_family"] == family}))
              for family in sorted({e["source_family"] for e in kept})}
    by_plan = {}
    for entry in sorted(kept, key=lambda e: (rank_key(e), _tie(e))):
        index = by_plan.setdefault(entry["source_collection_plan"], dict(
            source_collection_plan_sha256=entry["source_collection_plan_sha256"], targets={}))
        index["targets"].setdefault(entry["target_id"], []).append(entry["id"])
    reader.finish()
    return dict(schema=SCHEMA, state="screened_reviewed_original_train_pose_priors_not_training_data",
        checkpoint=str(checkpoint), checkpoint_manifest_sha256=manifest_sha256,
        entries=sorted(kept, key=lambda e: (e["compatibility_group"], e["target_id"], rank_key(e), _tie(e))),
        groups=dict(sorted(groups.items())), by_plan=dict(sorted(by_plan.items())),
        source_bindings=dict(sorted(reader.bindings.items())),
        donors=donors, counts=dict(references=len(kept), donors=len(donors),
            targets=len({e["target_id"] for e in kept}), compatibility_groups=len(groups),
            unique_target_camera_poses=len(unique_views),
            historical_848_references=sum(e["reference_role"] == "historical_848_pose_prior" for e in kept),
            new_native_training_rows=0), exclusions=dict(sorted(skipped.items())),
        ranking_policy="strict_trace_then_parent_proxy_contrast_interior_interval_width.v1",
        selection_policy="evidence_anchor_then_farthest_camera_robot_pose.v1",
        training_approved=False, source_cap_reset=False, physical_motion_commanded=False,
        new_geometry_and_native_qualification_required=True)
