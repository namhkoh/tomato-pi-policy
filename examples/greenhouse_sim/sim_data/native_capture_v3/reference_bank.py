"""Hash-bound reviewed TRAIN original references, not a training dataset.

``build(checkpoint, manifest_sha256=...)`` consumes the existing clear-checkpoint
format. The caller pins draft/manifest.json; its recorded hashes, not freshly
invented replacements, authenticate the index, reviews, labels, audits, original
collection plans, captures, samples and source assets. No USD/Isaac, image decode,
annotation replay, source-code import, capture or source mutation is performed.
Historical audit implementation fingerprints are retained in their bound receipts;
they are NOT requalified against today's code. This is evidence reuse, not a new
geometry or native-sensor qualification. Unsupported provenance fails closed.

Entries preserve exact plan/capture/sample identities. ``select_views`` requires
one compatibility group and target; it ranks recorded evidence, then greedily
spreads actual camera/robot poses. Unknown metrics stay unknown. Identical camera
poses never buy another view, even when RGB, query pixels or robot joints differ.
848x408 inputs remain historical pose priors; all bank training counters are zero.
New geometry/poses still need native screen, mount, depth, trace and visual checks.

CLI (create-only output): python -m sim_data.native_capture_v3.reference_bank
  --checkpoint PATH --manifest-sha256 SHA256 --output NEW_PATH
Verify a serialized bank with ``check(bank)`` before consuming it. That rebuilds
from the pinned checkpoint and rejects altered metadata, rankings and counts too.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


SCHEMA = "greenhouse.reviewed_original_reference_bank.v1"
HEAD_CAMERA = "/World/RBY1/link_head_2/attachments/HeadCamera/D405/DepthCamera"
OPTICS = ("camera_path", "resolution", "intrinsics", "focal_length_mm", "apertures_mm",
          "aperture_offsets_mm", "clipping_range_m", "depth_convention", "crop_resize")
MAX_JSON_BYTES = 64 * 1024 * 1024
MAX_ROWS = 100000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def sha256(path):
    """Stream file bytes only; never load sensor arrays or source geometry."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "Duplicate JSON key: " + key)
        result[key] = value
    return result


def _parse(data):
    def invalid(value):
        raise ValueError("Non-finite JSON: " + value)
    return json.loads(data, object_pairs_hook=_pairs, parse_constant=invalid)


def _safe(root, name):
    root = Path(root).resolve()
    _require(isinstance(name, str) and name and not Path(name).is_absolute(), "Relative file required")
    path = (root / name).resolve()
    _require(path.is_relative_to(root) and path != root, "File escapes source directory")
    return path


class _Reader:
    """Per-build cache only. Conflicting pins fail before any cached read is used."""
    def __init__(self):
        self.bindings = {}
        self.documents = {}
        self.verified_maps = {}

    def bind(self, path, expected):
        path = Path(path).resolve()
        _require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected),
                 "Missing/invalid original SHA256: " + str(path))
        key = str(path)
        _require(key not in self.bindings or self.bindings[key] == expected,
                 "Conflicting original hashes: " + key)
        if key not in self.bindings:
            _require(path.is_file() and sha256(path) == expected, "Changed bound file: " + key)
            self.bindings[key] = expected
        return path

    def bindings_from(self, bindings):
        _require(isinstance(bindings, dict) and bindings, "Missing existing source bindings")
        # These objects belong to this reader's immutable decoded documents.
        # Rechecking a many-thousand-asset plan for each target is unnecessary;
        # finish() still rehashes every unique file after the complete build.
        if self.verified_maps.get(id(bindings)) is bindings:
            return
        for name, digest in bindings.items():
            _require(Path(name).is_absolute(), "Absolute original binding required")
            self.bind(name, digest)
        self.verified_maps[id(bindings)] = bindings

    def json(self, path, expected):
        path = self.bind(path, expected)
        if str(path) not in self.documents:
            with path.open("rb") as stream:
                data = stream.read(MAX_JSON_BYTES + 1)
            _require(len(data) <= MAX_JSON_BYTES, "Metadata size limit")
            _require(hashlib.sha256(data).hexdigest() == expected, "File changed during read")
            self.documents[str(path)] = _parse(data)
        return self.documents[str(path)]

    def finish(self):
        # Catch concurrent mutation; never update a stale pin to make it pass.
        for name, expected in self.bindings.items():
            _require(sha256(name) == expected, "Source changed during bank build: " + name)


def _unique(rows, key, description):
    result = {}
    for row in rows:
        value = row[key]
        _require(value not in result, "Duplicate " + description)
        result[value] = row
    return result


def _one(rows, predicate, description):
    found = [row for row in rows if predicate(row)]
    _require(len(found) == 1, "Missing/ambiguous " + description)
    return found[0]


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _matrix(value, *, column=False):
    _require(isinstance(value, list) and len(value) == 4 and
             all(isinstance(row, list) and len(row) == 4 and all(_number(x) for x in row)
                 for row in value), "Invalid pose matrix")
    row = [list(x) for x in zip(*value)] if column else value
    _require(all(abs(row[i][3] - (i == 3)) < 1e-7 for i in range(4)), "Non-affine pose")
    _require(all(abs(sum(row[i][k]*row[j][k] for k in range(3)) - (i == j)) < 1e-6
                 for i in range(3) for j in range(3)), "Non-rigid pose")
    a, b, c = row[:3]
    det = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    _require(abs(det-1) < 1e-6, "Reflected pose")
    return row


def _pose_evidence(sample, audited):
    cal, robot = sample["calibration"], sample["robot_snapshot"]
    camera = audited.get("camera", {})
    _require(cal.get("camera_path") == camera.get("camera_path") == HEAD_CAMERA,
             "Correct robot head camera required")
    _require(cal.get("resolution") in ([848, 408], [1696, 816]) and
             "crop_resize" in cal and cal["crop_resize"] is None, "Uncropped source sensor required")
    _require(all(camera.get(k) is True for k in ("mounted_robot_pov_verified",
             "mount_matches_reconstructed_robot", "urdf_joint_limits_checked")), "Missing audited head mount/FK")
    _require(audited.get("integrity_and_recomputed_annotations_passed") is True,
             "Missing original annotation audit")
    for key in ("fk_position_error_m", "fk_matrix_max_error"):
        _require(_number(camera.get(key)) and 0 <= camera[key] <= 1e-7, "Invalid audited FK error")
    observed = _matrix(cal["camera_to_world_usd_row_vectors"])
    root = _matrix(robot["robot_root_to_world_usd_row_vectors"])
    _matrix(robot["camera_to_head_column_vectors"], column=True)
    for key, value in (("camera_world_m", observed[3][:3]), ("base_world_m", root[3][:3])):
        recorded = camera.get(key)
        _require(isinstance(recorded, list) and len(recorded) == 3 and
                 all(_number(x) and abs(x-y) <= 1e-7 for x, y in zip(recorded, value)),
                 "Audited/source pose mismatch")
    joints = robot.get("joint_degrees", {})
    _require(joints and all(_number(v) for v in joints.values()) and robot.get("joint_limits_checked") is True,
             "Missing checked robot joints")
    _require(camera.get("head_joint_degrees") == {k: v for k, v in joints.items() if k.startswith("head_")}
             and {"head_0", "head_1"} <= joints.keys(), "Audited head joints differ")
    screen = robot.get("visual_bound_screen", {})
    _require(screen.get("passed") is True and screen.get("possible_overlap_count") == 0 and
             screen.get("possible_overlaps") == [] and screen.get("floor_checked_separately") is True and
             isinstance(screen.get("method"), str) and screen["method"], "Unscreened original source pose")
    floor = robot.get("floor_alignment", {})
    supports = floor.get("wheel_supports", [])
    _require(len(supports) == 2 and all(_number(s.get("clearance_m")) and
             abs(s["clearance_m"]) <= .005 for s in supports), "Missing source floor evidence")
    _require(_number(floor.get("base_minimum_clearance_m")) and floor["base_minimum_clearance_m"] >= 0,
             "Invalid source base clearance")


def _metric(value, source, units, scope, *, signed=False):
    if value is None:
        return dict(status="unknown", value=None, source=source, units=units, scope=scope)
    _require(_number(value) and (signed or value >= 0), "Invalid ranking evidence: " + source)
    return dict(status="observed", value=value, source=source, units=units, scope=scope)


def ranking_evidence(row, label, audited, resolution, trace=None):
    """Use only authenticated metadata supplied by build; proxies are not clear flags."""
    usability, legibility = label.get("query_usability", {}), row.get("legibility", {})
    parent = audited.get("parent_proxy_diagnostic", {})
    strict = dict(status="unknown", value=None, source="bound_native_query_trace",
                  scope="query_to_cut_native_depth_identity_and_local_usability")
    if trace is not None:
        _require(resolution == [1696, 816], "Native strict trace attached to historical reference")
        _require(type(trace.get("passed")) is bool and isinstance(trace.get("reasons"), list),
                 "Malformed strict trace")
        if trace["passed"]:
            _require(trace["reasons"] == [] and trace.get("native_depth_reconstructed") is False and
                     trace.get("identity_or_depth_gap_count") == 0 and
                     trace.get("local_usability_failures") == [] and
                     type(trace.get("probe_count")) is int and trace["probe_count"] > 0 and
                     type(trace.get("unique_pixels")) is int and trace["unique_pixels"] > 0 and
                     _number(trace.get("minimum_interior_radius_px")), "Incomplete strict trace pass")
        strict.update(status="observed", value=trace["passed"])
    return dict(strict_native_trace=strict,
        parent_clearance_proxy_m=_metric(parent.get("nominal_point_to_parent_surface_m"),
            "audit.parent_proxy_diagnostic.nominal_point_to_parent_surface_m", "m",
            "nominal_parent_capsule_proxy_not_interval_mesh_or_blade_clearance", signed=True),
        local_contrast_8bit=_metric(usability.get("local_contrast_8bit"),
            "label.query_usability.local_contrast_8bit", "8bit", "local_query_not_whole_trace"),
        interior_radius_px=_metric(usability.get("interior_radius_px"),
            "label.query_usability.interior_radius_px", "source_px", "query_mask_interior"),
        interval_proxy_px=_metric(legibility.get("interval_length_px"),
            "index.legibility.interval_length_px", "source_px", "projected_interval_proxy"),
        width_proxy_px=_metric(legibility.get("width_proxy_px"),
            "index.legibility.width_proxy_px", "source_px", "width_proxy"))


def rank_key(entry):
    """Ascending lexicographic evidence rank; no depth or RGB-noise tie breaker.

    Native strict pass first, unknown next, explicit failure last. For each
    subsequent metric observed precedes unknown, larger first. Pixel metrics
    are normalized to the historical 848-wide sensor, without resampling data.
    These priorities are a planning heuristic, not calibrated quality gates.
    """
    metrics = entry["evidence"]
    trace = metrics["strict_native_trace"]["value"]
    result = [0 if trace is True else 2 if trace is False else 1]
    scale = 848 / entry["calibration"]["resolution"][0]
    for key in ("parent_clearance_proxy_m", "local_contrast_8bit", "interior_radius_px",
                "interval_proxy_px", "width_proxy_px"):
        value = metrics[key]["value"]
        result.extend((value is None, 0 if value is None else -value*(scale if key.endswith("px") else 1)))
    return tuple(result)


def _camera_distance(a, b):
    a, b = a["calibration"]["camera_to_world_usd_row_vectors"], b["calibration"]["camera_to_world_usd_row_vectors"]
    position = math.dist(a[3][:3], b[3][:3]) / .02
    # Rotation chord distance is stable at zero (unlike acos near identity).
    orientation = math.sqrt(sum((a[i][j]-b[i][j])**2 for i in range(3) for j in range(3))) / .1233742584
    return max(position, orientation)


def _robot_distance(a, b):
    a, b = a["robot_snapshot"], b["robot_snapshot"]
    root = math.dist(a["robot_root_to_world_usd_row_vectors"][3][:3],
                     b["robot_root_to_world_usd_row_vectors"][3][:3]) / .02
    joints = a["joint_degrees"]
    other = b["joint_degrees"]
    return max(root, max((abs(v-other[k])/5 for k, v in joints.items() if k in other), default=0))


def _tie(entry):
    # Stable geometry-first order. Hashes identify provenance, not visual diversity.
    return (_digest(entry["calibration"]["camera_to_world_usd_row_vectors"]),
            entry["source_capture"], entry["source_sample"], entry["id"])


def select_views(bank, group_id, target_id, limit=6, *, minimum_camera_distance=1.0):
    """Select within ONE frozen plan-compatible group. Call check on loaded banks.

    Distance 1 means 2cm translation or approximately 5 degrees rotation. A tiny
    positive floor always collapses identical cameras, including different robot
    postures/RGB. Greedy farthest-camera, then robot, then evidence order follows
    the strongest evidence anchor. No claim of independent training diversity.
    """
    _require(type(limit) is int and limit > 0, "Positive integer view limit required")
    _require(_number(minimum_camera_distance) and minimum_camera_distance >= 0, "Invalid pose distance")
    _require(group_id in bank["groups"], "Unknown compatibility group")
    candidates = [e for e in bank["entries"] if e["compatibility_group"] == group_id and e["target_id"] == target_id]
    candidates.sort(key=lambda e: (rank_key(e), _tie(e)))
    selected = []
    while candidates and len(selected) < limit:
        if not selected:
            chosen = candidates[0]
        else:
            candidates = [e for e in candidates if min(_camera_distance(e, s) for s in selected)
                          > max(1e-6, minimum_camera_distance)]
            if not candidates:
                break
            chosen = min(candidates, key=lambda e: (
                -min(_camera_distance(e, s) for s in selected),
                -min(_robot_distance(e, s) for s in selected), rank_key(e), _tie(e)))
        selected.append(chosen)
        candidates.remove(chosen)
    return deepcopy(selected)


def ranked_references(bank, source_collection_plan, target_id, *, source_collection_plan_sha256):
    """Batch-driver lookup by the ACTUAL original plan path/hash and target.

    Detached ranked entries retain their compatibility_group and all source
    identities. Different groups must not be merged into one shared-scene plan.
    Use select_views for diversity within a group. An absent target returns [].
    """
    path = str(Path(source_collection_plan).resolve())
    _require(path in bank["by_plan"], "Original collection plan absent from bank")
    index = bank["by_plan"][path]
    _require(index["source_collection_plan_sha256"] == source_collection_plan_sha256,
             "Original collection plan hash mismatch")
    entries = {e["id"]: e for e in bank["entries"]}
    return deepcopy([entries[key] for key in index["targets"].get(target_id, [])])


def _compatibility(entry, manifest, original):
    # Existing shared-scene plans compare the reconstructed mount at 1e-9.
    # USD round-off (~1e-16) must not manufacture hundreds of scene groups.
    # Quantize ONLY the grouping key more strictly (1e-10); keep the complete
    # original matrix in every entry and the original source hashes unchanged.
    mount = [[round(float(v), 10) or 0.0 for v in row]
             for row in entry["robot_snapshot"]["camera_to_head_column_vectors"]]
    return dict(source_collection_plan=entry["source_collection_plan"],
        source_collection_plan_sha256=entry["source_collection_plan_sha256"],
        source_family=entry["source_family"], split="train", original_variant=original,
        expected_scene_counts=manifest["scene_counts"], lighting=manifest["lighting"],
        renderer=manifest["renderer"],
        excluded_external_roots=manifest["unbundled_external_prop_roots_excluded"],
        optics={key: entry["calibration"][key] for key in OPTICS},
        camera_to_head_column_vectors=mount, mount_group_quantization=1e-10)


def build(checkpoint, *, manifest_sha256):
    """Read-only, fail-closed checkpoint adapter with an explicit root trust pin.

    SHA256 proves equality to caller-pinned receipts, not signer identity. Existing
    reviews are copied exactly, never fabricated, upgraded or silently overridden.
    All original source bindings in selected audits/plans are byte-verified. This
    can involve substantial disk reads, but never native jobs or image processing.
    """
    checkpoint = Path(checkpoint).resolve()
    draft, reader = checkpoint / "draft", _Reader()
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
        capture = Path(audit["source_run"]).resolve()
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
                        if Path(p).parent.parent.resolve() == capture]
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
            path = _safe(sample_path.parent, name)
            _require(audit["bindings_sha256"].get(str(path)) == receipt["sha256"], "Unbound sample payload")
            reader.bind(path, receipt["sha256"])
        _require(sample["files"]["inputs/rgb.png"]["sha256"] == row["rgb_sha256"], "Reviewed RGB mismatch")
        label_name = row["files"]["label"]
        label = reader.json(_safe(draft, label_name), hashes[label_name])
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
            rgb_sha256=row["rgb_sha256"], source_label=str(_safe(draft, label_name)),
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


def check(bank):
    """Fresh read-only verification; rejects edited bank entries and stale sources."""
    _require(bank.get("schema") == SCHEMA, "Unknown reference bank")
    expected = build(bank["checkpoint"], manifest_sha256=bank["checkpoint_manifest_sha256"])
    _require(bank == expected, "Reference bank metadata or bindings changed")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _require(not args.output.exists(), "New output only")
    bank = build(args.checkpoint, manifest_sha256=args.manifest_sha256)
    _require(str(args.output.resolve()) not in bank["source_bindings"], "Output overlaps source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(bank, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(bank["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
