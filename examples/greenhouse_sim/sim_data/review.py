"""Explicit human review records; no automatic cut eligibility or control output."""

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

from .audit import safe_asset


# Scope separates visual anatomy from camera visibility and robot accessibility.
REASONS = {
    "anatomy_matches": ("Anatomy matches", "anatomy_confirmed", "anatomy", "Attachment and detached membership look correct; cut location not assessed."),
    "old_stub": ("Already-deleafed stub", "excluded", "anatomy", "Already-deleafed stub; not an intact petiole target."),
    "wrong_anatomy": ("Wrong anatomy / membership", "excluded", "anatomy", "Attachment or detached membership is incorrect."),
    "attachment_unclear": ("Attachment / geometry unclear", "unresolved", "anatomy", "Attachment or geometry needs investigation."),
    "occluded": ("Occluded in this view", "unresolved", "visibility", "Cannot judge anatomy from this occluded view."),
    "out_of_view": ("Outside camera view", "unresolved", "visibility", "Target is outside the inspected camera view; anatomy not assessed."),
    "out_of_workspace": ("Outside intended workspace", "excluded", "workspace", "Outside the reviewer's intended workspace; robot reachability not measured."),
}


def _review_row(report, target_id, decision, reviewer, notes, reason_code=None, evidence=None, supersedes=()):
    allowed = {"anatomy_confirmed", "excluded", "unresolved"}
    if decision not in allowed or not reviewer.strip() or not notes.strip():
        raise ValueError("Choose a review decision and provide reviewer name and notes")
    target = next((t for t in report["targets"] if t["target_id"] == target_id), None)
    if target is None:
        raise ValueError("Target is not in this audit")
    if decision == "anatomy_confirmed" and (target["status"] != "needs_review" or report["status"] == "blocked"):
        raise ValueError(f"{target_id}: excluded or structurally blocked target cannot be anatomy-confirmed")
    path = Path(report["manifest_path"])
    if hashlib.sha256(path.read_bytes()).hexdigest() != report["manifest_sha256"]:
        raise ValueError("Manifest changed since audit; re-audit before recording a decision")
    for component in report["components"].values():
        asset = safe_asset(path.parent, component["file"])
        if hashlib.sha256(asset.read_bytes()).hexdigest() != component["asset_sha256"]:
            raise ValueError("Component asset changed since audit; re-audit before recording a decision")
    reason = REASONS.get(reason_code) if reason_code else None
    if reason_code and (reason is None or decision != reason[1]):
        raise ValueError("Reason and decision do not match")
    row = {
        "schema_version": "greenhouse.anatomy_review.v2" if reason else "greenhouse.anatomy_review.v1",
        "review_id": uuid.uuid4().hex,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "target_id": target_id, "manifest_sha256": report["manifest_sha256"],
        "component_asset_hashes": {key: value["asset_sha256"] for key, value in report["components"].items()},
        "rules_version": report["rules_version"], "decision": decision,
        "reviewer": reviewer.strip(), "notes": notes.strip(),
        "cut_approval": False, "physical_executability": "not_tested",
    }
    if reason:
        row.update(reason_code=reason_code, review_scope=reason[2])
        row["supersedes_review_ids"] = sorted(set(supersedes))
    if evidence is not None:
        if (evidence.get("target_id") != target_id or evidence.get("manifest_sha256") != report["manifest_sha256"]
                or evidence.get("component_asset_hashes") != row["component_asset_hashes"]
                or evidence.get("training_input_allowed") is not False):
            raise ValueError("Thumbnail does not match target/source")
        image = Path(evidence["image_path"])
        if hashlib.sha256(image.read_bytes()).hexdigest() != evidence["image_sha256"]:
            raise ValueError("Thumbnail changed since capture")
        row["evidence"] = evidence
    return row


def _output_directory(report, output):
    path = Path(report["manifest_path"])
    output = Path(output).resolve()
    source_root = path.parents[3] if path.parent.parent.name == "components" else path.parent
    if output.is_relative_to(source_root):
        raise ValueError("Review output must not be inside the source package")
    return output


def record_review(report, target_id, decision, reviewer, notes, output, *, reason_code=None, evidence=None, supersedes=(), view_context=None):
    row = _review_row(report, target_id, decision, reviewer, notes, reason_code, evidence, supersedes)
    if view_context is not None:
        if view_context.get("training_input_allowed") is not False:
            raise ValueError("Review view context must remain diagnostic-only")
        row["view_context"] = view_context
    output = _output_directory(report, output)
    output.mkdir(parents=True, exist_ok=True)
    result = output / f"{row['review_id']}.json"
    with result.open("x", encoding="utf-8") as stream:
        json.dump(row, stream, indent=2)
    return result


def record_batch(entries, reviewer, reason_code, notes, output, evidence, *, supersedes=None):
    """One atomic file for explicitly selected, captured cards. No partial approvals."""
    if not entries or len({t for _, t in entries}) != len(entries):
        raise ValueError("Select at least one distinct inspected target")
    if reason_code not in REASONS:
        raise ValueError("Choose a review reason")
    reason = REASONS[reason_code]
    rows = []
    for report, target_id in entries:
        if target_id not in evidence:
            raise ValueError("Every selected card must have a completed thumbnail")
        rows.append(_review_row(report, target_id, reason[1], reviewer,
                                notes.strip() or reason[3], reason_code, evidence[target_id],
                                (supersedes or {}).get(target_id, ())))
    # Validate ALL selections and sources before creating an output file.
    for report, _ in entries:
        output = _output_directory(report, output)
    output.mkdir(parents=True, exist_ok=True)
    batch_id = uuid.uuid4().hex
    result = output / (batch_id + ".json")
    temporary = output / (batch_id + ".pending")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump({"schema_version": "greenhouse.anatomy_batch.v1", "batch_id": batch_id,
                   "explicit_selection": True, "training_input_allowed": False, "records": rows}, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, result)
    return result
