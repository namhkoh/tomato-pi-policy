"""CPU-only, source-bound matched Y0/-20/+20 mm diagnostic preparation.

One selected target from a completed V4 job; any original 1..12 target count.
The original scene anchor, anatomy, optics, lighting and annotation policy stay
unchanged. New captures must use frozen native_generated_views.collect, raw
storage and reference56 for EVERY captured view. There is deliberately no
capture command, worker admission implementation or renderer registration here.

Native integration must create its OWN new request/result/failure receipts and
call validate_for_collect(app, plan) AFTER constructing SimulationApp. That
handoff replays the FULL original parent plan, never the three-view derivative.
CPU plan/verify do not import USD or claim native completion/visual approval.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from ..dataset_review import require, verify_bindings
from ..depth_preview import sha256
from ..native_budget import REFERENCE
from ..native_capture_v4 import prepare as v4
from ..native_dataset.bundle import SampleReader, safe_path
from ..native_multitarget_plan import OPTICS, assert_compatible, merge_bindings
from ..native_view_plan import propose_specs
from ..native_view_pose import bounded_reference_root

SCHEMA = "greenhouse.native_matched_parallax_plan.v1"
PROFILE = "view001_matched_y0_minus20_plus20_raw56.v1"
OFFSETS = (0., -.02, .02)
ROOT = Path(__file__).resolve().parents[4]
_CODE_ROOT = Path(__file__).resolve().parent.parent
FROZEN = {
    "native_generated_views.py": "573054884fa48da986b2926815f6fd225b591500ac247dbd95d0a2fc974fc8b7",
    "native_multitarget_plan.py": "32169cd27e87869c4bfdf39773fee3f101b01caacd15f95db609433fe197ee67",
    "native_view_plan.py": "0a23f67f5cdeea2bddb94b79a757a2d7c527a77800d70db90c69034b1c4c6fbe",
    "native_view_pose.py": "6da9cdaa780691e138893bdf69e9bec51609dc9fa8935c4f21db74dcd20d0628",
    "native_capture_v4/prepare.py": "04a2dc01ab44471641de0b711f5ea593595f54b3aa0b2a0d6dc62998bab56e42",
    "native_dataset/bundle.py": "6910647e2e226583fd7ab1e17767cf3cc0254672825d2f180c89336f0fdc1fca",
    "native_lossless_codec.py": "c1bca65e1e3c8c0150559855e79996dd68c7a5d2a5fdd3f7506fe63e1414109b",
    "native_budget.py": "21a1e608c3d129121653f696276ba9a881251e3af6e382226dde6982de71211a",
}
_LOADED = {str(p): sha256(p) for p in
           (Path(__file__).resolve(), Path(__file__).with_name("__init__.py").resolve())}
COLLECT_KWARGS = dict(profile_render=False, instance_backend="fast", render_budget=REFERENCE)
RECEIPT_CONTRACT = dict(
    request_schema="greenhouse.native_matched_parallax_request.v1",
    result_schema="greenhouse.native_matched_parallax_result.v1",
    failure_schema="greenhouse.native_matched_parallax_failure.v1",
    owner="separate_native_worker_not_cpu_preparation",
    required_files=["request.json", "result.json"], failure_file="failure.json",
    required_bindings=["plan_path", "plan_sha256", "implementation_bindings", "original_v4"],
    requires_native_admission=True, full_parent_replay_after_simulation_app=True,
    unchanged_source_end_hash_check=True, publish_completion_after_app_shutdown=True,
    all_three_decisions_required=True, failed_geometry_not_replaced=True,
    independent_native_annotation_replay_required=True, training_approved=False)


def implementation_bindings():
    pins = {str((_CODE_ROOT / name).resolve()): pin for name, pin in FROZEN.items()}
    merge_bindings(pins, v4.implementation_bindings())
    merge_bindings(pins, _LOADED)
    verify_bindings(pins)
    return pins


class _Inputs:
    """Invocation-local receipt ledger, not a decode/stat cache."""
    def __init__(self):
        self.pins = {}

    def json(self, path, expected=None):
        path = Path(path).resolve(strict=True)
        raw = path.read_bytes()
        pin = hashlib.sha256(raw).hexdigest()
        require(expected is None or pin == expected, "Pinned file changed: " + str(path))
        merge_bindings(self.pins, {str(path): pin})
        return json.loads(raw)

    def finish(self):
        verify_bindings(self.pins)  # unconditional final disk reread/hash


def _metadata(inputs, sample, record, base, case):
    reader = SampleReader(sample, expected_bindings={
        "sample.json": record["sample_sha256"],
        "supervision/label.json": record["label_sha256"]}, expected_json=(
            {"supervision/query_trace.json": record["query_trace"]}
            if record.get("query_trace") is not None else {}))
    meta = reader.metadata
    reader.json("supervision/label.json")
    if record.get("query_trace") is not None:
        trace_bytes = reader.read("supervision/query_trace.json")
        if reader.manifest is None:
            merge_bindings(inputs.pins, {str(sample / "supervision/query_trace.json"):
                                        hashlib.sha256(trace_bytes).hexdigest()})
    # Bind original physical bytes without decoding/reconstructing native arrays.
    if reader.manifest is not None:
        manifest = inputs.json(sample / "bundle.json")
        require(manifest == reader.manifest, "Bundle changed during metadata read")
        for entry in manifest["files"].values():
            merge_bindings(inputs.pins, {str(safe_path(sample, entry["stored_path"])): entry["stored_sha256"]})
    else:
        merge_bindings(inputs.pins, {str(sample / "sample.json"): record["sample_sha256"],
            str(sample / "supervision/label.json"): record["label_sha256"]})
        for name, entry in meta["files"].items():
            merge_bindings(inputs.pins, {str(safe_path(sample, name)): entry["sha256"]})
    require(meta["supervision"]["target_id"] == case["target_id"]
            and np.allclose(meta["supervision"]["nominal_world_m"], case["expected_nominal_world_m"],
                            atol=1e-9, rtol=0), "Saved baseline target/geometry differs")
    require(meta["robot_snapshot"] == record["robot_snapshot"]
            and meta["geometry_screen"]["passed"] is True, "Baseline pose/screen differs")
    spec = case["views"][0]
    pose = meta["robot_snapshot"]
    reference = base["expected_robot_snapshot"]
    require(pose["desired_cut_pixel_xy"] == spec["desired_pixel_xy"], "Baseline framing differs")
    require(np.allclose(np.asarray(pose["robot_root_to_world_usd_row_vectors"])[3, :2],
                        np.asarray(reference["robot_root_to_world_usd_row_vectors"])[3, :2],
                        atol=1e-9, rtol=0), "view001 is not the reference base position")
    require(all(pose["joint_degrees"][k] == v for k, v in reference["joint_degrees"].items()
                if k not in ("head_0", "head_1")), "Baseline body joints differ")
    require(np.allclose(pose["camera_to_head_column_vectors"], reference["camera_to_head_column_vectors"],
                        atol=1e-9, rtol=0),
            "Baseline camera mount differs")
    require(all(meta["calibration"][k] == base["expected_calibration"][k] for k in OPTICS),
            "Baseline native optics differ")
    return dict(sample_path=str(sample), sample_sha256=record["sample_sha256"],
        label_sha256=record["label_sha256"], rgb_sha256=meta["files"]["inputs/rgb.png"]["sha256"],
        requested_spec=deepcopy(spec), calibration=deepcopy(meta["calibration"]),
        lighting=deepcopy(meta["lighting"]), robot_snapshot=deepcopy(pose),
        original_label_reason=record["label_reason"],
        scope="saved_native_metadata_and_hashes_not_new_visual_or_annotation_replay")


def _load_parent(completed_path, completed_sha256, target_id):
    inputs = _Inputs()
    completed_path = Path(completed_path).resolve()
    require(completed_path.name == "campaign_result.json" and isinstance(completed_sha256, str)
            and len(completed_sha256) == 64, "Explicit completed V4 receipt and SHA256 required")
    done = inputs.json(completed_path, completed_sha256)
    job_root = completed_path.parent
    require(Path(done["job"]).resolve() == job_root and done["state"] == "native_complete_pending_review"
            and type(done["native_exit_code"]) is int and done["native_exit_code"] == 0
            and done["training_approved"] is False, "Successful completed job required")
    provenance = inputs.json(job_root / "reference_provenance.json")
    prepared_plan_path = Path(provenance["plan_path"]).resolve()
    require(prepared_plan_path.is_relative_to(job_root) and prepared_plan_path.parent != job_root,
            "Original preparation must belong to this completed job")
    prepared_path = prepared_plan_path.parent / "prepared.json"
    prepared = inputs.json(prepared_path)
    require(prepared == provenance, "Prepared/provenance receipts differ")
    require(all(prepared[k] == done[k] for k in ("job_id", "attempt_id", "seed", "source_family")),
            "Completed/prepared identity differs")
    schedule_path = Path(prepared["schedule"]).resolve()
    schedule = inputs.json(schedule_path, prepared["schedule_sha256"])
    scheduled, attempt = v4._scheduled(schedule, prepared["job_id"], prepared["attempt_id"])
    checked = v4.verify_prepared(prepared_path.parent, schedule_path, scheduled, attempt, prepared["seed"],
        max_targets=prepared["max_targets"], schedule_sha256=prepared["schedule_sha256"])
    require(checked == prepared, "Original V4 verification differs")
    inputs.json(prepared_path.parent / "prepare_request.json")
    parent = inputs.json(prepared_plan_path, prepared["plan_sha256"])
    parent_path = job_root / "plan.json"
    require(inputs.json(parent_path, prepared["plan_sha256"]) == parent, "Captured/prepared plans differ")
    cases = parent["target_cases"]
    require(1 <= len(cases) <= 12 and parent["views_per_target"] == 6
            and parent["maximum_native_frames"] == 6*len(cases), "Original 1..12-target six-view plan required")
    bases = [inputs.json(c["base_pair_plan"], c["base_pair_plan_sha256"]) for c in cases]
    require(parent["anchor_pair_plan"] == cases[0]["base_pair_plan"], "Original anchor differs")
    for case, base in zip(cases, bases, strict=True):
        assert_compatible(bases[0], base)
        expected = propose_specs(base["expected_robot_snapshot"]["robot_root_to_world_usd_row_vectors"],
                                 case["expected_nominal_world_m"], case["base_pair_plan_sha256"], 6)
        for spec in expected:
            spec["candidate_id"] = base["source_row"]["component_id"] + "_" + spec["candidate_id"]
        require(case["views"] == expected, "Original six-view deterministic policy differs")
    selected = [i for i, c in enumerate(cases) if c["target_id"] == target_id]
    require(len(selected) == 1, "Exactly one explicit original target required")
    index = selected[0]
    audit = inputs.json(job_root / "automatic_audit.json")
    result = inputs.json(job_root / "capture/result.json", audit["result_sha256"])
    request = inputs.json(job_root / "capture/request.json", audit["request_sha256"])
    require(audit["plan_sha256"] == request["plan_sha256"] == prepared["plan_sha256"]
            and Path(request["plan_path"]).resolve() == parent_path, "Completed native plan binding differs")
    require(audit["state"] == "completed_automatic_annotation_replay" and audit["training_approved"] is False
            and result["state"] == "native_generated_multiview_pilot_complete_pending_review"
            and result["training_approved"] is False, "Completed native/audit receipts required")
    planned = {s["candidate_id"]: (c["target_id"], s) for c in cases for s in c["views"]}
    rows = result["records"]
    require(len(rows) == len(planned) and len({r["candidate_id"] for r in rows}) == len(rows),
            "Incomplete/duplicate original capture decisions")
    for row in rows:
        require(planned.get(row["candidate_id"]) == (row["target_id"], row["requested_spec"])
                and row["state"] in ("native_captured_pending_review", "rejected_pose",
                                     "rejected_possible_geometry_overlap"), "Original capture decision differs")
    captured = {r["candidate_id"]: r for r in rows if r["state"] == "native_captured_pending_review"}
    reviewed = {Path(r["sample"]).name: r for r in audit["records"]}
    require(len(reviewed) == len(audit["records"]) == len(captured) == result["captured_frames"] == done["captured"]
            and reviewed.keys() == captured.keys(), "Captured/audited inventory differs")
    require(dict(Counter(r["decision"] for r in audit["records"])) == audit["counts"] == done["audit_counts"],
            "Completed audit counts differ")
    for name, row in captured.items():
        reviewed_row = reviewed[name]
        require(Path(reviewed_row["sample"]).resolve() == job_root / "capture" / name
                and reviewed_row["target_id"] == row["target_id"]
                and all(reviewed_row[k] == row[k] for k in ("sample_sha256", "label_sha256"))
                and reviewed_row["trace_replayed_exact"] is (row.get("query_trace") is not None)
                and all(reviewed_row[k] is True for k in ("label_replayed_exact",
                    "native_callback_hashes_verified", "source_and_file_hashes_verified")),
                "Original replay binding differs")
    case, base = cases[index], bases[index]
    baseline_id = case["views"][0]["candidate_id"]
    require(baseline_id in captured, "Selected view001 must have an actual completed native capture")
    baseline = _metadata(inputs, job_root / "capture" / baseline_id, captured[baseline_id], base, case)
    origin = dict(completed_path=str(completed_path), completed_sha256=completed_sha256,
        original_plan_path=str(parent_path), original_plan_sha256=prepared["plan_sha256"],
        prepared_path=str(prepared_path), prepared_sha256=inputs.pins[str(prepared_path)],
        schedule_path=str(schedule_path), schedule_sha256=prepared["schedule_sha256"],
        selected_target_id=target_id, original_target_count=len(cases),
        original_render_budget=request["render_budget_profile"], original_storage_backend=request["storage_backend"])
    inputs.finish()
    return parent, bases, index, origin, inputs.pins, baseline


def _derive(parent, bases, index, origin, input_pins, baseline):
    result = deepcopy(parent)
    case = deepcopy(parent["target_cases"][index])
    original = case["views"][0]
    reference = bases[index]["expected_robot_snapshot"]
    target = case["expected_nominal_world_m"]
    specs = []
    for suffix, lateral in zip(("y0", "ym20", "yp20"), OFFSETS, strict=True):
        spec = deepcopy(original)
        spec["candidate_id"] = bases[index]["source_row"]["component_id"] + "_parallax_" + suffix
        if lateral:
            spec["y_offset_m"] = original["y_offset_m"] + lateral
            spec["base_target_radius_m"] = math.hypot(spec["root_x_m"] - target[0], spec["y_offset_m"])
            spec["base_displacement_from_reference_m"] = abs(lateral)
        # Retain original view_policy; only deterministic physical-spec fields change.
        bounded_reference_root(reference, spec, target)
        specs.append(spec)
    case["views"] = specs
    result.update(schema=SCHEMA, experiment_profile=PROFILE, original_v4=deepcopy(origin),
        target_cases=[case], views_per_target=3, maximum_native_frames=3,
        baseline_evidence=deepcopy(baseline), native_receipt_contract=deepcopy(RECEIPT_CONTRACT),
        experiment=dict(maximum_selected_targets=1, lateral_offsets_m=list(OFFSETS), outward_offset_m=0.,
            capture_order=[s["candidate_id"] for s in specs],
            fixed_framing_source=original["candidate_id"], collect_kwargs=deepcopy(COLLECT_KWARGS),
            storage_backend="raw_frozen_native_collector", requested_subframes_every_capture=56,
            no_geometry_skip_replacement=True, initial_state="cpu_prepared_native_not_started",
            comparison_scope="fresh_matched_positions_not_pixel_equivalence_to_old_trial",
            original_y0_is_repeated_reference_not_diversity=True, new_context_credit=False,
            source_cap_reset=False, training_approved=False))
    merge_bindings(result["source_bindings"], input_pins)
    merge_bindings(result["implementation_bindings"], implementation_bindings())
    return result


def build(completed_path, completed_sha256, *, target_id):
    """CPU only; completed receipt SHA is externally supplied, never inferred."""
    implementation_bindings()
    result = _derive(*_load_parent(completed_path, completed_sha256, target_id))
    for field in ("source_bindings", "prerequisite_bindings", "implementation_bindings"):
        verify_bindings(result[field])
    return result


def check(plan):
    """CPU: reconstruct and compare every field; no subset/approval whitelist."""
    require(plan.get("schema") == SCHEMA and plan.get("experiment_profile") == PROFILE,
            "Explicit matched-parallax plan required")
    origin = plan["original_v4"]
    expected = build(origin["completed_path"], origin["completed_sha256"], target_id=origin["selected_target_id"])
    canonical = lambda value: json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))
    require(canonical(plan) == canonical(expected), "Matched-parallax derivation changed")
    # Original anchor is intentionally retained even when it is NOT selected.
    return _Inputs().json(plan["anchor_pair_plan"], plan["source_bindings"][plan["anchor_pair_plan"]])


def validate_for_collect(app, plan):
    """Native-owner handoff ONLY AFTER SimulationApp startup; does not render.

    Caller must independently perform current admission, own create-only native
    receipts, invoke frozen collect(app, ..., **COLLECT_KWARGS), and verify all
    pins again at completion. A returned base is not capture/release approval.
    """
    module = sys.modules.get("isaacsim")
    app_type = getattr(module, "SimulationApp", None)
    require(isinstance(app_type, type) and isinstance(app, app_type) and app.is_running(),
            "Running SimulationApp required before full original-plan USD replay")
    base = check(plan)
    from ..native_multitarget_plan import check as check_original
    origin = plan["original_v4"]
    inputs = _Inputs()
    parent = inputs.json(origin["original_plan_path"], origin["original_plan_sha256"])
    require(check_original(parent, replay_geometry=True) == base, "Full original parent replay differs")
    inputs.finish()
    for field in ("source_bindings", "prerequisite_bindings", "implementation_bindings"):
        verify_bindings(plan[field])
    return base


def _destination(path, plan):
    path = Path(path).resolve()
    root = (ROOT / "data/sim_data/diagnostics").resolve()
    require(path.is_relative_to(root) and path != root and path.suffix == ".json" and not path.exists(),
            "New diagnostic JSON destination required")
    inputs = _Inputs()
    sources = [Path(plan["original_v4"]["completed_path"]).parent]
    for case in plan["target_cases"]:
        base = inputs.json(case["base_pair_plan"], case["base_pair_plan_sha256"])
        sources.extend(Path(base[k]).resolve() for k in ("source_capture", "variant_directory", "prerequisite_directory"))
        collection = inputs.json(base["source_collection_plan"], plan["source_bindings"][base["source_collection_plan"]])
        sources.append(Path(collection["package"]).resolve())
    require(all(not path.is_relative_to(s) and not s.is_relative_to(path) for s in sources),
            "Diagnostic output overlaps source tree")
    require(all(not path.is_relative_to(Path(p)) and not Path(p).is_relative_to(path)
                for key in ("source_bindings", "prerequisite_bindings", "implementation_bindings") for p in plan[key]),
            "Diagnostic output overlaps pinned input")
    inputs.finish()
    return path


def write_plan(plan, output):
    """Create-only CPU artifact; never writes native request/result receipts."""
    check(plan)
    destination = _destination(output, plan)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        json.dump(plan, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return dict(plan_path=str(destination), plan_sha256=sha256(destination),
        schema=SCHEMA, targets=1, maximum_native_frames=3, native_launched=False,
        full_parent_geometry_replay="deferred_until_SimulationApp", native_receipts_created=False,
        training_approved=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("plan")
    create.add_argument("--completed", type=Path, required=True)
    create.add_argument("--completed-sha256", required=True)
    create.add_argument("--target-id", required=True)
    create.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--plan", type=Path, required=True)
    verify.add_argument("--plan-sha256", required=True)
    args = parser.parse_args(argv)
    if args.command == "plan":
        report = write_plan(build(args.completed, args.completed_sha256, target_id=args.target_id), args.output)
    else:
        inputs = _Inputs()
        plan = inputs.json(args.plan, args.plan_sha256)
        check(plan)
        inputs.finish()
        report = dict(state="cpu_plan_verified_pending_native_parent_replay", native_launched=False,
                      native_receipts_created=False, training_approved=False)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
