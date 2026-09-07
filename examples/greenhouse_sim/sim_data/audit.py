"""Audit supplied plant manifests without importing Isaac or changing source assets.

Run from examples/greenhouse_sim: python -m sim_data.audit --package ...
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import uuid

SCHEMA = "greenhouse.asset_audit.v1"
RULES = "intact_leaf_petiole_review.v1"
KNOWN_TYPES = {"main_stem", "sub_stem", "leaf", "truss", "fruit", "flower"}
PROTECTED_TYPES = KNOWN_TYPES - {"sub_stem", "leaf"}
ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PACK = ROOT / "data/sim_data/package_20260905/tomato_greenhouse_pack"


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def vector(value, size=3):
    return isinstance(value, list) and len(value) == size and all(finite_number(v) for v in value)


def safe_asset(root, name):
    """Reject traversal, absolute paths, alternate streams, and escaped symlinks."""
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError("asset file must be a relative POSIX path")
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("asset file escapes plant directory")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("resolved asset escapes plant directory")
    return result


def descendants(components, start):
    children = defaultdict(list)
    for key, component in components.items():
        children[component["parent"]].append(key)
    found, pending = set(), [start]
    while pending:
        key = pending.pop()
        if key in found:
            continue
        found.add(key)
        pending.extend(children[key])
    return sorted(found)


def audit_manifest(path):
    path = Path(path).resolve()
    report = {
        "schema_version": SCHEMA, "rules_version": RULES, "plant_id": path.parent.name,
        "manifest_path": str(path), "manifest_sha256": None, "issues": [],
        "components": {}, "targets": [], "source_assets_modified": False,
        "scope": "manifest_structure_only", "geometry_verified": False,
    }

    def issue(severity, code, component_id=None, **details):
        report["issues"].append({"severity": severity, "code": code,
                                 "component_id": component_id, **details})

    try:
        raw = path.read_bytes()
        report["manifest_sha256"] = hashlib.sha256(raw).hexdigest()
        manifest = json.loads(raw)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("components"), list):
            raise ValueError("manifest must contain a components list")
    except (OSError, ValueError) as exc:
        issue("error", "unreadable_manifest", detail=str(exc))
        report["status"] = "blocked"
        return report

    report["metadata"] = {k: manifest.get(k) for k in ("generator", "version", "seed", "units", "up_axis")}
    if manifest.get("units") != "meters" or manifest.get("up_axis") != "Z":
        issue("error", "unsupported_coordinate_convention")
    if not manifest["components"]:
        issue("error", "empty_plant")
    if manifest.get("component_count") != len(manifest["components"]):
        issue("error", "component_count_mismatch")

    components = report["components"]
    for entry in manifest["components"]:
        if not isinstance(entry, dict):
            issue("error", "invalid_component_record")
            continue
        key = entry.get("id")
        if not isinstance(key, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key):
            issue("error", "invalid_component_id")
            continue
        if key in components:
            issue("error", "duplicate_component_id", key)
            continue
        kind, parent = entry.get("type"), entry.get("parent")
        if not isinstance(kind, str) or kind not in KNOWN_TYPES:
            issue("error", "unknown_component_type", key)
        if parent is not None and not isinstance(parent, str):
            issue("error", "invalid_parent", key)
            parent = None
        transform = entry.get("transform")
        position = transform.get("translate") if isinstance(transform, dict) else None
        if not vector(position):
            issue("error", "invalid_translation", key)
        if isinstance(transform, dict) and set(transform) - {"translate"}:
            issue("error", "unsupported_transform_ops", key)
        attach, axis = entry.get("attach_point"), entry.get("axis")
        if not vector(attach):
            issue("error", "invalid_attachment", key)
        if not vector(axis) or math.dist(axis, [0, 0, 0]) < 1e-8:
            issue("error", "invalid_axis", key)
        elif abs(math.dist(axis, [0, 0, 0]) - 1) > 0.02:
            issue("warning", "nonunit_axis", key)
        for name in ("radius", "length"):
            if name in entry and (not finite_number(entry[name]) or entry[name] <= 0):
                issue("error", "invalid_dimension", key, field=name)
        if kind == "sub_stem" and not isinstance(entry.get("deleafed"), bool):
            issue("warning", "unknown_deleafed_state", key)

        capsules = entry.get("capsules", [])
        if not isinstance(capsules, list):
            issue("error", "invalid_capsules", key)
            capsules = []
        valid_capsules = []
        for chain in capsules:
            if (not isinstance(chain, list) or len(chain) < 2
                    or not all(vector(point, 4) and point[3] > 0 for point in chain)):
                issue("error", "invalid_capsule_chain", key)
                continue
            if all(math.dist(chain[0][:3], p[:3]) < 1e-9 for p in chain[1:]):
                issue("warning", "degenerate_capsule_chain", key)
            valid_capsules.append(chain)
        if kind == "sub_stem" and not valid_capsules:
            issue("warning", "missing_petiole_centerline", key)

        asset_hash = None
        try:
            asset = safe_asset(path.parent, entry.get("file"))
            if not asset.is_file():
                issue("error", "missing_component_file", key)
            else:
                asset_hash = hashlib.sha256(asset.read_bytes()).hexdigest()
        except (OSError, ValueError) as exc:
            issue("error", "invalid_component_file", key, detail=str(exc))
        components[key] = {
            "id": key, "type": kind if isinstance(kind, str) else "unknown", "parent": parent,
            "file": entry.get("file"), "asset_sha256": asset_hash,
            "translation_plant_m": position if vector(position) else None,
            "attachment_plant_m": attach if vector(attach) else None,
            "axis_plant": axis if vector(axis) else None,
            "radius_m": entry.get("radius") if finite_number(entry.get("radius")) else None,
            "length_m": entry.get("length") if finite_number(entry.get("length")) else None,
            "deleafed": entry.get("deleafed") if isinstance(entry.get("deleafed"), bool) else None,
            "capsules_local_m": valid_capsules,
        }

    for key, component in components.items():
        parent = component["parent"]
        if parent is not None and parent not in components:
            issue("error", "missing_parent", key, parent=parent)
        seen, current = set(), key
        while current in components:
            if current in seen:
                issue("error", "parent_cycle", key)
                break
            seen.add(current)
            current = components[current]["parent"]
        if component["parent"] is None and component["type"] != "main_stem":
            issue("error", "non_main_stem_root", key)
    if len([c for c in components.values() if c["parent"] is None]) != 1:
        issue("error", "root_count_mismatch")

    blocked = any(i["severity"] == "error" for i in report["issues"])
    for key, component in sorted(components.items()):
        if component["type"] != "sub_stem":
            continue
        subtree = descendants(components, key)
        types = Counter(components[k]["type"] for k in subtree)
        parent_type = components.get(component["parent"], {}).get("type")
        reasons = []
        status = "needs_review"
        if component["deleafed"] is True:
            reasons.append("already_deleafed")
        if parent_type != "main_stem":
            reasons.append("not_direct_main_stem_petiole")
        protected = [k for k in subtree if components[k]["type"] in PROTECTED_TYPES]
        if protected:
            reasons.append("protected_descendants")
        if not types["leaf"]:
            reasons.append("no_leaf_descendants")
        if reasons:
            status = "excluded"
        if blocked:
            status = "blocked"
            reasons.append("plant_structural_errors")
        if component["deleafed"] is None:
            status = "blocked"
            reasons.append("unknown_deleafed_state")
        report["targets"].append({
            "target_id": f"{path.parent.name}/{key}", "component_id": key,
            "status": status, "reason_codes": reasons or ["intact_leaf_petiole_requires_review"],
            "attachment_plant_m": component["attachment_plant_m"], "axis_plant": component["axis_plant"],
            "expected_detached_component_ids": subtree, "descendant_type_counts": dict(types),
            "protected_descendant_ids": protected,
            "anatomy_review": "pending", "agronomic_eligibility": "unreviewed",
            "observability": "not_measured", "physical_executability": "not_tested",
            "canonical_cut_point_m": None, "admissible_cut_region": None, "grasp_region": None,
            "review_notes": "Attachment is not an approved cut point. No cut/grasp offsets have been assumed.",
        })
    report["status"] = "blocked" if blocked else "needs_human_review"
    report["counts"] = {"components": len(components), "types": dict(Counter(c["type"] for c in components.values())),
                        "targets": dict(Counter(t["status"] for t in report["targets"]))}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-usd", action="store_true", help="Also check assembled bounds/transforms using pxr")
    args = parser.parse_args(argv)
    manifests = sorted((args.package / "plants/components").glob("*/manifest.json"))
    if not manifests:
        parser.error("No component manifests in package")
    output = args.output or ROOT / "data/sim_data/audits" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    if output.resolve().is_relative_to(args.package.resolve()):
        parser.error("Audit output must be outside the source package")
    if output.exists():
        parser.error("Output directory already exists; choose a new run to preserve previous evidence")
    output.mkdir(parents=True)
    reports = []
    for manifest in manifests:
        report = audit_manifest(manifest)
        if args.check_usd and report["status"] != "blocked":
            from .geometry import audit_geometry
            try:
                report["geometry_audit"] = audit_geometry(report)
            except Exception as exc:
                report["issues"].append({"severity": "error", "code": "usd_audit_failed", "detail": str(exc)})
                report["status"] = "blocked"
        (output / f"{manifest.parent.name}.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        reports.append(report)
        print(f"{report['plant_id']}: {report['status']} {report.get('counts', {}).get('targets', {})}", flush=True)
    summary = {
        "schema_version": SCHEMA, "rules_version": RULES, "plant_count": len(reports),
        "plants_blocked": sum(r["status"] == "blocked" for r in reports),
        "target_counts": dict(Counter(t["status"] for r in reports for t in r["targets"])),
        "issue_counts": dict(Counter(i["code"] for r in reports for i in r["issues"])),
        "geometry_warning_counts": dict(Counter(w["code"] for r in reports
                                                for w in r.get("geometry_audit", {}).get("warnings", []))),
        "usd_checks_requested": args.check_usd, "all_targets_require_human_review": True,
        "source_assets_modified": False, "reports": [f"{r['plant_id']}.json" for r in reports],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Audit saved to {output.resolve()}", flush=True)
    return 2 if summary["plants_blocked"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
