"""Append-only prototype reviews and versioned, reference-based RGB-D indices.

Local reviewer roles are self-declared, not authenticated identities. An assistant
recommendation is never a human confirmation. Neither enables training in v1:
the source cut rule is still horticulturally unvalidated. No execution output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import uuid

from .dataset_review import (OBSERVATIONS, SCHEMA, read_json, write_json, require,
                             safe_file, verify_bindings)
from .depth_preview import sha256

REVIEW_SCHEMA = "greenhouse.rgbd_prototype_review.v1"
DECISIONS = {"assistant": {"recommend", "hold", "reject"}, "human": {"confirm", "hold", "reject"}}


def checked_audit(path):
    audit = read_json(path)
    require(audit["schema_version"] == SCHEMA and audit["state"] == "complete_engineering_audit_not_approval",
            "Expected a completed engineering audit")
    require(audit["training_dataset_approved"] is False, "This workflow handles prototype data only")
    verify_bindings(audit["bindings_sha256"])
    for name, expected in audit["cards_sha256"].items():
        require(sha256(safe_file(Path(path).parent, name)) == expected, "Review card changed")
    return audit


def validate_record(row, audit, audit_hash):
    require(row.get("schema_version") == REVIEW_SCHEMA and row.get("audit_sha256") == audit_hash,
            "Stale or incompatible review")
    role, decision = row.get("reviewer_role"), row.get("decision")
    require(role in DECISIONS and decision in DECISIONS[role], "Role cannot grant this decision")
    require(isinstance(row.get("reviewer"), str) and row["reviewer"].strip()
            and isinstance(row.get("notes"), str) and row["notes"].strip(), "Reviewer and evidence notes required")
    matches = [s for s in audit["samples"] if s["sample_id"] == row.get("sample_id")]
    require(len(matches) == 1, "Unknown reviewed sample")
    sample = matches[0]
    require(row.get("target_review_id") == sample["target_review_id"], "Review target mismatch")
    require(row.get("inspected") == ["full_scene_rgb", "cut_overlay", "native_identity_mask", "native_depth"],
            "Explicit visual inspection checklist required")
    if decision in {"confirm", "recommend"}:
        require(sample["quality"]["clear_view_gate_passed"] is True, "Cannot promote a failed clear-view gate")
    require(row.get("human_prototype_label_confirmation") is (role == "human" and decision == "confirm"),
            "Assistant review cannot impersonate human confirmation")
    require(row.get("training_eligible") is False and row.get("physical_cut_approved") is False
            and row.get("horticultural_validation") == "pending", "Prototype review cannot approve training or execution")
    require(re.fullmatch(r"[0-9a-f]{32}", row.get("review_id", "")) is not None, "Invalid review ID")


def record(audit_path, sample_id, role, reviewer, decision, notes, output, *, inspected=False, supersedes=None):
    audit_path, output = Path(audit_path).resolve(), Path(output).resolve()
    audit = checked_audit(audit_path)
    require(not output.is_relative_to(Path(audit["source_run"])), "Do not write reviews inside captured data")
    require(inspected, "Inspect RGB, cut overlay, mask and native depth before recording a review")
    sample = next((s for s in audit["samples"] if s["sample_id"] == sample_id), None)
    require(sample is not None, "Unknown sample")
    row = {"schema_version": REVIEW_SCHEMA, "review_id": uuid.uuid4().hex,
           "created_utc": datetime.now(timezone.utc).isoformat(), "audit_sha256": sha256(audit_path),
           "sample_id": sample_id, "target_review_id": sample["target_review_id"],
           "reviewer_role": role, "reviewer": reviewer, "decision": decision, "notes": notes,
           "inspected": ["full_scene_rgb", "cut_overlay", "native_identity_mask", "native_depth"],
           "human_prototype_label_confirmation": role == "human" and decision == "confirm",
           "training_eligible": False, "physical_cut_approved": False, "horticultural_validation": "pending",
           "supersedes_review_id": None, "identity_authentication": "local_self_declared_role_not_authenticated"}
    if supersedes:
        previous = read_json(supersedes)
        validate_record(previous, audit, row["audit_sha256"])
        require(previous["sample_id"] == sample_id and previous["reviewer_role"] == role, "Cannot supersede another sample/role")
        row["supersedes_review_id"] = previous["review_id"]
    validate_record(row, audit, row["audit_sha256"])
    output.mkdir(parents=True, exist_ok=True)
    path = output / (row["review_id"] + ".json")
    write_json(path, row)
    return path


def active_reviews(rows, audit, audit_hash):
    """No implicit last-writer-wins: conflicting tips or incomplete history fail closed."""
    ids = {}
    for row in rows:
        validate_record(row, audit, audit_hash)
        require(row["review_id"] not in ids, "Duplicate review ID")
        ids[row["review_id"]] = row
    superseded = set()
    for row in rows:
        previous_id = row["supersedes_review_id"]
        if previous_id is not None:
            require(previous_id in ids, "Missing superseded review history")
            previous = ids[previous_id]
            require(previous["sample_id"] == row["sample_id"] and previous["reviewer_role"] == row["reviewer_role"],
                    "Cross-sample/role supersession")
            visited = {row["review_id"]}
            while previous_id is not None:
                require(previous_id not in visited and previous_id in ids, "Cyclic or missing review history")
                visited.add(previous_id)
                previous_id = ids[previous_id]["supersedes_review_id"]
            superseded.add(row["supersedes_review_id"])
    active = {}
    for identifier, row in ids.items():
        if identifier not in superseded:
            key = (row["sample_id"], row["reviewer_role"])
            require(key not in active, "Conflicting reviews: explicitly supersede the earlier review")
            active[key] = row
    return active


def queue_for(sample, assistant, human):
    decisions = {r["decision"] for r in (assistant, human) if r}
    if "reject" in decisions:
        return "rejected"
    if not sample["quality"]["clear_view_gate_passed"] or "hold" in decisions:
        return "diagnostic_hold"
    if human and human["decision"] == "confirm":
        return "human_confirmed_prototype_not_trainable"
    if assistant and assistant["decision"] == "recommend":
        return "recommended_for_human_review"
    return "visual_review_pending"


def validate_splits(rows):
    groups = {}
    for row in rows:
        require(row["split"] in {"unassigned", "train", "validation", "test"}, "Invalid split")
        key = row["source_plant_family"]
        require(key and groups.setdefault(key, row["split"]) == row["split"], "Source plant family leaks across splits")
    return groups


def build(audit_path, reviews, output, version):
    audit_path, reviews, output = Path(audit_path).resolve(), Path(reviews).resolve(), Path(output).resolve()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", version) is not None, "Invalid version")
    require(not output.exists(), "Choose a NEW dataset version directory; no overwrite")
    audit = checked_audit(audit_path)
    require(reviews.is_dir(), "Review directory missing")
    for root in (Path(audit["source_run"]), audit_path.parent, reviews):
        require(not output.is_relative_to(root) and not root.is_relative_to(output), "Keep package separate from sources/reviews")
    files = sorted(reviews.glob("*.json"))
    history = [read_json(path) for path in files]
    audit_hash = sha256(audit_path)
    active = active_reviews(history, audit, audit_hash)
    bindings = {**audit["bindings_sha256"], str(audit_path): audit_hash,
                **{str(p): sha256(p) for p in files},
                **{str(audit_path.parent / n): h for n, h in audit["cards_sha256"].items()}}
    def relative(path):
        return Path(os.path.relpath(path, output)).as_posix()
    rows = []
    run = Path(audit["source_run"])
    for sample in audit["samples"]:
        sid = sample["sample_id"]
        meta = read_json(run / sid / "sample.json")
        assistant, human = (active.get((sid, role)) for role in ("assistant", "human"))
        rows.append({"schema_version": "greenhouse.rgbd_review_index_row.v1", "sample_id": sid,
                     "source_plant_family": sample["source_plant_family"], "split": "unassigned",
                     "difficulty": "unassigned", "review_queue": queue_for(sample, assistant, human),
                     "training_eligible": False, "camera_provenance": sample["camera"],
                     "observation": {name: {"path": relative(run / sid / name), "sha256": meta["files"][name]["sha256"]}
                                     for name in OBSERVATIONS},
                     "calibration": meta["calibration"], "robot_snapshot": meta["robot_snapshot"],
                     "provisional_supervision": meta["supervision"],
                     "supervision_files": {name: {"path": relative(run / sid / name), **detail}
                                           for name, detail in meta["files"].items() if detail["role"] == "ground_truth_supervision"},
                     "review_card": relative(audit_path.parent / (sid + ".png")),
                     "source_metadata": {"path": relative(run / sid / "sample.json"), "sha256": sha256(run / sid / "sample.json")},
                     "engineering_checks": sample, "assistant_review": assistant, "human_review": human})
    groups = validate_splits(rows)
    verify_bindings(bindings)
    output.mkdir(parents=True)
    # This is an INDEX referencing original artifacts, not a self-contained image copy.
    with (output / "samples.jsonl").open("x", encoding="utf-8") as stream:
        import json
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    write_json(output / "review_history.json", history)
    lines = ["# Robot-head RGB-D prototype review", "",
             f"Version: `{version}`. **Review index only; no approved training samples.**", "",
             f"All {len(rows)} full-resolution inputs passed the robot-mounted camera/FK check. "
             "These are simulated RB-Y1 A v1.2 head-camera poses, not cinematic cameras or live lab robot measurements.", "",
             "Images below are REVIEW CARDS: full-scene RGB above, labelled 4x crops below. "
             "Only the unaltered 848x408 RGB/native depth/validity files are observation candidates. "
             "Cards, masks and projected geometry never enter the model observation.", "",
             "White: nominal cut 10 mm from manifest attachment. Magenta: 10-20 mm centreline interval. "
             "Green: exact visible petiole mask. Depth: direct native camera-Z, yellow nearer/purple farther (0.04-2 m).", "",
             "| Sample | Target | Review queue | Petiole width | Interval pixels |", "|---|---|---|---|---|"]
    for row in rows:
        s = row["engineering_checks"]
        lines.append(f"| {row['sample_id']} | {s['target_review_id']} | {row['review_queue']} | "
                     f"{s['quality']['estimated_petiole_diameter_px']:.2f} px | {s['quality']['projected_interval_length_px']:.2f} |")
    for row in sorted(rows, key=lambda r: (r["assistant_review"] is None, r["sample_id"])):
        sid = row["sample_id"]
        lines.extend(["", f"## {sid}: {row['engineering_checks']['target_review_id']}", "",
                      f"Queue: `{row['review_queue']}`. Human review: " + (row["human_review"]["decision"] if row["human_review"] else "pending") + ".", ""])
        if row["assistant_review"]:
            lines.extend(["Assistant inspection: " + row["assistant_review"]["notes"], ""])
        else:
            lines.extend(["Automated integrity/geometry checks completed; no explicit assistant visual-review decision recorded.", ""])
        lines.extend([f"[Original RGB]({relative(run/sid/'inputs/rgb.png')}) | "
                      f"[Native depth NPY]({relative(run/sid/'inputs/depth_m.npy')}) | "
                      f"[Source metadata]({relative(run/sid/'sample.json')})", "",
                      f"![{sid} robot view and review-only crops]({row['review_card']})", ""])
    lines.extend(["## Limits and next approval", "", *["- " + text for text in audit["limitations"]], "",
                  "Human `confirm` records agreement with the prototype point/interval in this image only; "
                  "it does not certify agronomy, physical safety or an executable trajectory. "
                  "No human confirmation is inferred from assistant review. Local roles are self-declared, not authenticated.", "",
                  f"Source plant families represented: {len(groups)}. Splits and difficulty remain unassigned; "
                  "camera views, branch copies and future appearance variants from one family must remain together.", "",
                  "This package references source files relatively; keep the source run/review directory with it. "
                  "It is not a portable self-contained training archive.", ""])
    with (output / "review.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    verify_bindings(bindings)
    result = {"schema_version": "greenhouse.rgbd_review_package.v1", "version": version,
              "state": "complete_review_index_not_training_dataset", "created_utc": datetime.now(timezone.utc).isoformat(),
              "packaging": "relative_reference_index_not_self_contained", "source_audit_sha256": audit_hash,
              "source_bindings_sha256": bindings, "source_family_splits": groups, "sample_count": len(rows),
              "training_eligible_count": 0, "training_dataset_approved": False,
              "queues": {q: sum(r["review_queue"] == q for r in rows) for q in sorted({r["review_queue"] for r in rows})},
              "files": {name: sha256(output/name) for name in ("samples.jsonl", "review_history.json", "review.md")},
              "implementation_sha256": sha256(Path(__file__))}
    write_json(output / "manifest.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    review = sub.add_parser("record")
    review.add_argument("--sample", required=True)
    review.add_argument("--role", choices=DECISIONS, required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--decision", choices=["recommend", "confirm", "hold", "reject"], required=True)
    review.add_argument("--notes", required=True)
    review.add_argument("--inspected-rgb-mask-depth", action="store_true")
    review.add_argument("--supersedes", type=Path)
    package = sub.add_parser("build")
    package.add_argument("--reviews", type=Path, required=True)
    package.add_argument("--version", required=True)
    for command in (review, package):
        command.add_argument("--audit", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "record":
        print(record(args.audit, args.sample, args.role, args.reviewer, args.decision, args.notes, args.output,
                     inspected=args.inspected_rgb_mask_depth, supersedes=args.supersedes))
    else:
        result = build(args.audit, args.reviews, args.output, args.version)
        print(result["state"], result["queues"])


if __name__ == "__main__":
    main()
