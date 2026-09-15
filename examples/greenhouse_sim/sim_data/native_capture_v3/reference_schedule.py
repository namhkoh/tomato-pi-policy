"""CPU-only, donor-balanced fresh camera-qualification queue from a checked bank.

plan(bank_path, reference_bank_sha256=..., output_root=..., fallback_limit=2)
chooses exactly one original plan-compatible group per TRAIN source family:
most distinct reviewed targets first, then the group's best rank_key anchor.
Fallbacks are ranked, bounded, same-group, distinct-camera original references.
The queue visits every donor's primary before any donor's fallback (round robin).

No qualification is reused, no native module/USD/pxr is imported, and no process
is launched. Each planned native_greenhouse_pair attempt produces TWO paired
sensor frames (848x408 and 1696x816), plus its separate known-surface controls.
None count as new training data or visual approval. The external serial launcher
owns execution, memory admission, failure receipts, review and generation.

A successful attempt must be verified using generated_capture's existing
verify_sensor_prerequisite against THAT attempt's exact sample and camera, before
collection using the SAME original source plan, family and compatibility group.
Sensor qualification is not target visibility, new-geometry safety or approval.
Changing fallback anchors requires a newly bound base plan for the actual anchor;
never reuse the primary's prerequisite binding after a fallback succeeds.

check(schedule) replays the checked bank and deterministic recipe and verifies
this module's own implementation bindings. It intentionally ignores later output
existence: a receipt is not proof reuse. require_fresh_outputs(schedule) is a
separate pre-launch guard; the native worker's own create-only checks still apply.
Only the CLI writes a create-only queue JSON, after plan() AND check() succeed.
"""
from collections import defaultdict
from copy import deepcopy
import argparse
import json
from pathlib import Path
import re

from . import reference_bank as rb


SCHEMA = "greenhouse.reference_camera_qualification_schedule.v1"
MODULE = "sim_data.native_greenhouse_pair"
MAX_FALLBACKS = 8


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _implementation_paths():
    return (Path(__file__).resolve(), Path(rb.__file__).resolve(),
            Path(__file__).with_name("__init__.py").resolve())


def _implementation_bindings():
    return {str(path): rb.sha256(path) for path in _implementation_paths()}


def _verify_implementation(bindings):
    _require(set(bindings) == {str(p) for p in _implementation_paths()}, "Missing schedule implementation binding")
    for name, expected in bindings.items():
        _require(rb.sha256(name) == expected, "Schedule implementation changed: " + name)


def _checked_bank(path, expected):
    reader = rb._Reader()
    result = reader.json(path, expected)
    _require(rb.check(result) is True, "Reference bank did not pass verification")
    reader.finish()
    return result


def _entry_order(entry):
    # Source identity breaks equal-evidence ties, never RGB statistics.
    return (rb.rank_key(entry), entry["source_sample_path"], entry["target_id"], entry["id"])


def _output_root(bank, bank_path, output_root):
    root = Path(output_root).resolve()
    _require(root != Path(root.anchor), "Filesystem root is not a qualification destination")
    protected = [Path(bank["checkpoint"]).resolve()]
    protected.extend(Path(e["source_capture"]).resolve() for e in bank["entries"])
    _require(not any(root == p or root.is_relative_to(p) or p.is_relative_to(root) for p in protected),
             "Qualification output overlaps original checkpoint/capture")
    for name in [str(bank_path), *bank["source_bindings"]]:
        path = Path(name).resolve()
        _require(not (path == root or path.is_relative_to(root) or root.is_relative_to(path)),
                 "Qualification output overlaps a bound source")
    return root


def _attempt(entry, family_index, anchor_index, root):
    result = deepcopy(entry)  # Exact original hashes, pose, calibration and review retained.
    identity = f"donor_{family_index:03d}_anchor_{anchor_index:02d}"
    destination = root / f"{family_index:03d}_{entry['source_family']}" / f"anchor_{anchor_index:02d}"
    result.update(attempt_id=identity, anchor_index=anchor_index,
        qualification_output=str(destination), qualification_module=MODULE,
        qualification_argv=["--source-capture", entry["source_capture"], "--sample", entry["source_sample"],
                            "--output", str(destination)],
        qualification_resolutions=[[848, 408], [1696, 816]],
        paired_sensor_frames=2, additional_known_surface_controls_required=True,
        reuse_existing_proof=False, sensor_prerequisite_verified=False,
        visual_approval=False, executed=False)
    return result


def _assemble(bank, bank_path, bank_hash, output_root, fallback_limit, implementation):
    _require(type(fallback_limit) is int and 0 <= fallback_limit <= MAX_FALLBACKS,
             f"fallback_limit must be an integer between 0 and {MAX_FALLBACKS}")
    _require(bank.get("training_approved") is False and bank.get("source_cap_reset") is False,
             "A reference bank cannot grant approval or reset caps")
    _require(bank.get("counts", {}).get("new_native_training_rows") == 0, "Bank is pose priors only")
    _require(bank.get("entries"), "Empty reference bank")
    root = _output_root(bank, bank_path, output_root)
    families = defaultdict(lambda: defaultdict(list))
    seen = set()
    for entry in bank["entries"]:
        family, group_id = entry["source_family"], entry["compatibility_group"]
        _require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", family), "Unsafe source-family identifier")
        _require(entry["id"] not in seen, "Duplicate bank reference identity")
        seen.add(entry["id"])
        _require(entry["split"] == "train" and entry["review"].get("decision") == "accept" and
                 entry.get("training_approved") is False and entry.get("source_cap_reset") is False,
                 "Reviewed TRAIN pose prior required")
        _require(entry["calibration"]["resolution"] == [848, 408],
                 "Existing native_greenhouse_pair requires an original 848x408 reference")
        _require(entry["calibration"]["camera_path"] == rb.HEAD_CAMERA and
                 entry["robot_snapshot"]["visual_bound_screen"].get("passed") is True,
                 "Screened robot-head source pose required")
        group = bank["groups"][group_id]
        _require(group["source_family"] == family and group["split"] == "train" and
                 group["source_collection_plan"] == entry["source_collection_plan"] and
                 group["source_collection_plan_sha256"] == entry["source_collection_plan_sha256"],
                 "Original plan/group/family mismatch")
        families[family][group_id].append(entry)
    jobs = []
    for family_index, family in enumerate(sorted(families), 1):
        candidates = []
        for group_id, entries in families[family].items():
            ranked = sorted(entries, key=_entry_order)
            targets = sorted({e["target_id"] for e in ranked})
            candidates.append((group_id, ranked, targets))
        # Coverage takes priority; never pool targets across incompatible plans.
        candidates.sort(key=lambda c: (-len(c[2]), _entry_order(c[1][0]), c[0]))
        group_id, ranked, targets = candidates[0]
        anchors = []
        for entry in ranked:
            if any(e["source_sample_sha256"] == entry["source_sample_sha256"] or
                   rb._camera_distance(e, entry) <= 1e-6 for e in anchors):
                continue
            anchors.append(entry)
            if len(anchors) == fallback_limit + 1:
                break
        attempts = [_attempt(e, family_index, i, root) for i, e in enumerate(anchors)]
        first = ranked[0]
        target_references = {target: [e["id"] for e in ranked if e["target_id"] == target] for target in targets}
        jobs.append(dict(job_id=f"donor_{family_index:03d}", source_family=family, split="train",
            source_collection_plan=first["source_collection_plan"],
            source_collection_plan_sha256=first["source_collection_plan_sha256"],
            compatibility_group=group_id, compatibility=deepcopy(bank["groups"][group_id]),
            reviewed_target_ids=targets, ranked_reference_ids_by_target=target_references,
            anchor=attempts[0], fallback_anchors=attempts[1:],
            candidate_group_summary=[dict(compatibility_group=g, reviewed_targets=len(t),
                best_reference_id=entries[0]["id"]) for g, entries, t in candidates],
            collection_after_qualification=dict(
                state="blocked_until_exact_successful_anchor_sensor_prerequisite_verified",
                source_collection_plan=first["source_collection_plan"],
                source_collection_plan_sha256=first["source_collection_plan_sha256"],
                source_family=family, compatibility_group=group_id,
                prerequisite_directory="successful_attempt.qualification_output",
                base_plan_source_sample="successful_attempt.source_sample",
                base_plan_source_capture="successful_attempt.source_capture",
                verifier="sim_data.generated_capture.verify_sensor_prerequisite",
                exact_sample_and_same_camera_required=True,
                new_targets_require_geometry_native_visibility_trace_and_review=True,
                generated_collection_plan_created=False, training_approved=False)))
    queue = []
    for round_index in range(fallback_limit + 1):
        for job in jobs:
            attempts = [job["anchor"], *job["fallback_anchors"]]
            if round_index >= len(attempts):
                continue
            attempt = attempts[round_index]
            queue.append(dict(sequence=len(queue) + 1, job_id=job["job_id"], source_family=job["source_family"],
                compatibility_group=job["compatibility_group"], attempt_id=attempt["attempt_id"],
                anchor_index=round_index, prior_attempt_ids=[a["attempt_id"] for a in attempts[:round_index]],
                run_condition="no_verified_donor_success_and_all_prior_donor_attempts_failed_or_explicitly_rejected"))
    return dict(schema=SCHEMA, state="cpu_planned_fresh_camera_qualification_not_executed",
        source_reference_bank=str(bank_path), source_reference_bank_sha256=bank_hash,
        source_bindings={str(bank_path): bank_hash}, implementation_bindings=implementation,
        qualification_output_root=str(root), fallback_limit=fallback_limit,
        policy="one_group_per_family_coverage_then_rank_round_robin_fallbacks.v1",
        proof_reuse_policy="never_reuse_all_donors_require_fresh_exact_anchor_pair",
        jobs=jobs, qualification_queue=queue,
        counts=dict(source_bank_references=len(bank["entries"]), donors=len(jobs), selected_groups=len(jobs),
            selected_reviewed_targets=sum(len(j["reviewed_target_ids"]) for j in jobs),
            primary_qualification_pairs=len(jobs), primary_paired_sensor_frames=2*len(jobs),
            fallback_qualification_pairs=len(queue)-len(jobs),
            maximum_qualification_attempts=len(queue), maximum_paired_sensor_frames=2*len(queue),
            additional_known_surface_control_frames_not_in_pair_counts=True,
            reused_proofs=0, executed_attempts=0, new_native_training_rows=0),
        source_cap_reset=False, training_approved=False, physical_motion_commanded=False,
        native_jobs_launched=False, model_run=False)


def require_fresh_outputs(schedule):
    """Read-only pre-launch guard for the whole unstarted schedule, not proof reuse."""
    for job in schedule["jobs"]:
        for attempt in [job["anchor"], *job["fallback_anchors"]]:
            _require(not Path(attempt["qualification_output"]).exists(), "Qualification output already exists; no reuse")
    return True


def plan(reference_bank, *, reference_bank_sha256, output_root, fallback_limit=2):
    """Check pinned bank, produce a deterministic metadata plan, create nothing."""
    path = Path(reference_bank).resolve()
    implementation = _implementation_bindings()
    bank = _checked_bank(path, reference_bank_sha256)
    result = _assemble(bank, path, reference_bank_sha256, output_root, fallback_limit, implementation)
    require_fresh_outputs(result)
    _verify_implementation(implementation)
    return result


def check(schedule):
    """Verify original bank, code and full recipe; never import a native verifier."""
    _require(schedule.get("schema") == SCHEMA, "Unknown reference schedule")
    _verify_implementation(schedule["implementation_bindings"])
    path = Path(schedule["source_reference_bank"]).resolve()
    digest = schedule["source_reference_bank_sha256"]
    _require(schedule["source_bindings"] == {str(path): digest}, "Missing/mismatched source bank binding")
    bank = _checked_bank(path, digest)
    expected = _assemble(bank, path, digest, schedule["qualification_output_root"],
                         schedule["fallback_limit"], schedule["implementation_bindings"])
    _require(schedule == expected, "Reference schedule metadata or ordering changed")
    _verify_implementation(schedule["implementation_bindings"])
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-bank", type=Path, required=True)
    parser.add_argument("--reference-bank-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--fallback-limit", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    _require(not output.exists(), "New queue JSON only")
    _require(not output.is_relative_to(args.output_root.resolve()), "Keep queue JSON outside qualification outputs")
    result = plan(args.reference_bank, reference_bank_sha256=args.reference_bank_sha256,
                  output_root=args.output_root, fallback_limit=args.fallback_limit)
    check(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result["counts"], sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
