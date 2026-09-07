"""Deterministic review sampling and fingerprint-aware history; no Isaac imports."""

from collections import defaultdict, deque
import hashlib
import json
from datetime import datetime
from pathlib import Path


def fingerprints(report):
    return {k: v["asset_sha256"] for k, v in report["components"].items()}


def attach_cached_geometry(reports, audit_root):
    """Reuse only a matching completed USD audit. Missing checks remain unknown."""
    runs = sorted(Path(audit_root).glob("*/summary.json"), reverse=True)
    for report in reports:
        for summary in runs:
            path = summary.parent / (report["plant_id"] + ".json")
            try:
                cached = json.loads(path.read_text(encoding="utf-8"))
                geometry = cached["geometry_audit"]
                if (cached["manifest_sha256"] != report["manifest_sha256"]
                        or cached["rules_version"] != report["rules_version"]
                        or fingerprints(cached) != fingerprints(report)
                        or geometry["scope"] != "assembled_translation_and_mesh_aabb_only"):
                    continue
                report["geometry_audit"] = geometry
                report["geometry_cache_source"] = str(path)
                break
            except (OSError, ValueError, KeyError, TypeError):
                continue


class History:
    def __init__(self, reports, directories):
        self.reports = {r["plant_id"]: r for r in reports}
        self.directories = sorted(set(Path(p).resolve() for p in directories))
        self.refresh()

    def refresh(self):
        self.rows = defaultdict(list)
        self.ignored = []
        seen = set()
        for directory in self.directories:
            for path in sorted(directory.glob("*.json")):
                try:
                    content = json.loads(path.read_text(encoding="utf-8"))
                    rows = content["records"] if content.get("schema_version") == "greenhouse.anatomy_batch.v1" else [content]
                    for row in rows:
                        if row["schema_version"] not in {"greenhouse.anatomy_review.v1", "greenhouse.anatomy_review.v2"}:
                            raise ValueError("unknown review schema")
                        target_id = row["target_id"]
                        report = self.reports.get(target_id.split("/")[0])
                        if report is None:
                            continue
                        if (row["manifest_sha256"] != report["manifest_sha256"]
                                or row["component_asset_hashes"] != fingerprints(report)
                                or row["rules_version"] != report["rules_version"]):
                            self.ignored.append(f"Stale review: {path.name} ({target_id})")
                            continue
                        if target_id not in {t["target_id"] for t in report["targets"]}:
                            raise ValueError("unknown target")
                        if row["decision"] not in {"anatomy_confirmed", "excluded", "unresolved"}:
                            raise ValueError("unknown decision")
                        if row["schema_version"] == "greenhouse.anatomy_review.v2":
                            from .review import REASONS
                            reason = REASONS.get(row.get("reason_code"))
                            if reason is None or row["decision"] != reason[1] or row.get("review_scope") != reason[2]:
                                raise ValueError("reason/scope/decision mismatch")
                            if (not isinstance(row.get("supersedes_review_ids", []), list)
                                    or not all(isinstance(v, str) for v in row.get("supersedes_review_ids", []))):
                                raise ValueError("invalid supersession list")
                        target = next(t for t in report["targets"] if t["target_id"] == target_id)
                        if row["decision"] == "anatomy_confirmed" and (report["status"] == "blocked" or target["status"] != "needs_review"):
                            raise ValueError("ineligible anatomy confirmation")
                        if not isinstance(row["timestamp_utc"], str) or not row["reviewer"].strip():
                            raise ValueError("invalid reviewer/timestamp")
                        if datetime.fromisoformat(row["timestamp_utc"]).tzinfo is None:
                            raise ValueError("timestamp has no timezone")
                        if row.get("cut_approval") is not False:
                            raise ValueError("unexpected cutting approval")
                        if row["review_id"] not in seen:
                            self.rows[target_id].append(row)
                            seen.add(row["review_id"])
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
                    self.ignored.append(f"Unreadable review: {path.name}: {exc}")
        for rows in self.rows.values():
            rows.sort(key=lambda r: (datetime.fromisoformat(r["timestamp_utc"]), r["review_id"]))

    def latest(self, target_id):
        rows = self.rows.get(target_id, [])
        return rows[-1] if rows else None

    def issues(self, target_id):
        rows = self.rows.get(target_id, [])
        if not rows:
            return []
        latest = rows[-1]
        flags = []
        if latest["decision"] == "unresolved":
            flags.append("human_unresolved")
        if latest.get("review_scope") == "visibility":
            flags.append("visibility_review_does_not_confirm_anatomy")
        # Keep legacy notes intact: never silently reinterpret a prior exclusion.
        note = latest.get("notes", "").lower()
        if latest["decision"] == "excluded" and not latest.get("review_scope") and any(
                word in note for word in ("high", "bound", "boud", "reach", "view")):
            flags.append("legacy_exclusion_needs_scope")
        remaining = {r["review_id"]: r for r in rows}
        for row in rows:
            for prior_id in row.get("supersedes_review_ids", []):
                prior = remaining.get(prior_id)
                if prior and datetime.fromisoformat(prior["timestamp_utc"]) < datetime.fromisoformat(row["timestamp_utc"]):
                    del remaining[prior_id]
        if len({(r["decision"], r.get("review_scope", "legacy")) for r in remaining.values()}) > 1:
            flags.append("conflicting_decisions_require_explicit_rereview")
        return flags

    def complete(self, target_id):
        latest = self.latest(target_id)
        return latest is not None and latest["decision"] != "unresolved" and not self.issues(target_id)


def target_flags(report, target):
    members = set(target["expected_detached_component_ids"])
    members.add(report["components"][target["component_id"]]["parent"])
    flags = {i["code"] for i in report["issues"] if i.get("component_id") in members or i.get("component_id") is None}
    geometry = report.get("geometry_audit")
    if geometry is None:
        flags.add("geometry_not_checked")
    else:
        flags.update(w["code"] for w in geometry.get("warnings", []) if w["component_id"] in members)
    if target["status"] == "blocked":
        flags.add("structurally_blocked")
    return sorted(flags)


def interleave(entries):
    """Plant-balanced, stable order, not a long run of adjacent stem indices."""
    plants = defaultdict(list)
    for entry in entries:
        plants[entry[0]["plant_id"]].append(entry)
    queues = []
    for plant in sorted(plants):
        rows = sorted(plants[plant], key=lambda e: hashlib.sha256(e[1]["target_id"].encode()).hexdigest())
        queues.append(deque(rows))
    result = []
    while queues:
        for queue in queues:
            if queue:
                result.append(queue.popleft())
        queues = [q for q in queues if q]
    return result


def review_queue(reports, history, mode="Mixed sample"):
    entries = [(r, t) for r in reports for t in r["targets"]]
    if mode == "All / revisit":
        return interleave(entries)
    entries = [e for e in entries if not history.complete(e[1]["target_id"])]
    flagged, normal, negatives = [], [], []
    for entry in entries:
        report, target = entry
        if target_flags(report, target) or history.issues(target["target_id"]):
            flagged.append(entry)
        elif target["status"] == "excluded":
            negatives.append(entry)
        else:
            normal.append(entry)
    flagged = interleave(flagged)
    # Unresolved human feedback takes precedence over routine geometry warnings.
    flagged.sort(key=lambda e: not bool(history.issues(e[1]["target_id"])))
    if mode == "Exceptions":
        return flagged
    if mode == "Unreviewed":
        return interleave(entries)
    if mode != "Mixed sample":
        raise ValueError("Unknown queue mode")
    queues = [deque(flagged), deque(interleave(normal)), deque(interleave(negatives))]
    result = []
    # Each six-card page mixes two flags, three normal examples, and one negative
    # where available. Availability/coverage is shown; this is not a power analysis.
    while any(queues):
        for index in (0, 1, 1, 0, 1, 2):
            if queues[index]:
                result.append(queues[index].popleft())
    return result
