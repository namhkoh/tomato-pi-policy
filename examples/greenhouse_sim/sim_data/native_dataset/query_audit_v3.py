"""V3 persistence receipts and replay; frozen V2 annotation, no approval.

All-frame retention is default. Selective mode retains strict candidates plus
fixed hashed rejected controls. Unpersisted callbacks are producer declarations,
NOT independently pixel-replayable observations. Witness mode stores every
same callback under qualification_witness/ and allows actual replay of skips.
No boundary_bindings integration, queue registration or inventory admission.
"""
from copy import deepcopy
from pathlib import Path
import argparse
import hashlib
import re
import numpy as np

from ..dataset_review import read_json, write_json, require, verify_bindings
from ..depth_preview import sha256
from ..capture_contract import fingerprint
from ..automated_native_review import check_native_evidence
from ..plant_variant_catalogue import load_for_inspection
from ..native_budget import REFERENCE, TRIAL, evidence as budget_evidence
from . import query_audit_v2 as v2
from . import compact_qualification
from .bundle import SampleReader, digest

ALL_FRAMES = "all_frames"
STRICT_ONLY = "strict_candidates_only"
MODES = (ALL_FRAMES, STRICT_ONLY)
WORKER_MODULE = "sim_data.native_dataset.compact_query_v3"
DOCUMENT_SCHEMA = "greenhouse.native_query_v3_capture.v1"
CAPTURE_STATE = "native_query_v3_complete_pending_persistence_replay"
AUDIT_STATE = "completed_query_v3_persistence_replay_unqualified"
CALLBACK_STATE = "native_callback_complete_v3"
PRE_STATES = ("rejected_pose", "rejected_possible_geometry_overlap")
WITNESS_DIRECTORY = "qualification_witness"
SALT = "greenhouse.reject_controls.donor_disposition.sha256.v1"
_POLICY = dict(schema="greenhouse.native_selective_persistence.v1",
    modes=list(MODES), default=ALL_FRAMES, reject_control_denominator=64, reject_control_salt=SALT,
    reject_strata=["hold", "exclude"], stratum_includes_original_donor=True,
    control_hash_fields=["salt", "plan_sha256", "source_family", "disposition", "candidate_id"],
    control_rule="sha256_canonical_json_integer_mod_64_equals_zero",
    annotation_epoch=v2.ANNOTATION_EPOCH, annotation_policy_sha256=v2.POLICY_SHA256,
    native_capture_and_annotation_gates_unchanged=True,
    warmup_counter="validated_callbacks_not_persisted_samples",
    persistence_fields_in_annotation_metadata=False,
    qualification_witness_max_planned_views=12, qualification_witness_directory=WITNESS_DIRECTORY,
    skipped_receipts_are_not_pixel_replay=True, training_approved=False, source_cap_reset=False)
PERSISTENCE_POLICY_SHA256 = fingerprint(_POLICY)
_HERE = Path(__file__).resolve().parent
_FROZEN = {"compact_query_v2.py":"d530e001b5020ba44022bf2a5a8452c5c967e83e5797425b0d5d92faa33d5b31",
           "query_audit_v2.py":"d02fbe409b7b6311ed9ee6b0e869a40604bf456aedd2f59060512cdb02940886"}
_LOADED = dict(v2.implementation_bindings(), **{
    str(p.resolve()):sha256(p) for p in (Path(__file__), _HERE/"compact_query_v3.py")})


def implementation_bindings():
    for name, pin in _FROZEN.items():
        require(sha256(_HERE/name) == pin, "Frozen V2 implementation changed")
    require(all(_LOADED[p] == h for p,h in v2.implementation_bindings().items()), "V2 binding drift")
    verify_bindings(_LOADED)
    return dict(_LOADED)


def persistence_policy():
    return deepcopy(_POLICY)


def _hash(value):
    require(isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value), "Explicit SHA256 required")
    return value


def candidate_name(name):
    reserved = {"con", "prn", "aux", "nul", WITNESS_DIRECTORY,
                *(f"{prefix}{i}" for prefix in ("com", "lpt") for i in range(1,10))}
    require(isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]+", name) is not None
            and name.lower() not in reserved, "Unsafe candidate identifier")
    return name


def sample_directories(root, name):
    root = Path(root).resolve()
    candidate_name(name)
    primary, witness = (root/name).resolve(), (root/WITNESS_DIRECTORY/name).resolve()
    require(primary.parent == root and witness.parent == root/WITNESS_DIRECTORY,
            "Sample path escapes primary/witness root")
    return primary, witness


def configuration(mode=ALL_FRAMES, witness=False):
    require(mode in MODES and type(witness) is bool, "Explicit persistence mode/witness required")
    return dict(schema=DOCUMENT_SCHEMA, persistence_policy=persistence_policy(),
        persistence_policy_sha256=PERSISTENCE_POLICY_SHA256, persistence_mode=mode,
        qualification_witness_save_all=witness, native_persistence_qualified=False)


def validate_plan_mode(plan, mode, witness):
    configuration(mode, witness)
    require("target_cases" in plan, "V3 requires the existing batch plan")
    names = [candidate_name(s["candidate_id"]) for c in plan["target_cases"] for s in c["views"]]
    require(names and len({name.lower() for name in names}) == len(names), "Nonempty unique planned views required")
    require(not witness or len(names) <= 12, "Witness qualification limited to 12 planned views")


def retention(plan_sha256, source_family, disposition, name, mode=ALL_FRAMES):
    _hash(plan_sha256); candidate_name(name)
    require(isinstance(source_family, str) and source_family
            and disposition in ("strict", "hold", "exclude") and mode in MODES,
            "Invalid retention ancestry/disposition")
    control_hash = (fingerprint([SALT, plan_sha256, source_family, disposition, name])
                    if disposition != "strict" else None)
    control = control_hash is not None and int(control_hash, 16) % 64 == 0
    role = ("strict_candidate" if disposition == "strict" else "rejected_control" if control else
            "all_frames_rejection" if mode == ALL_FRAMES else "unpersisted_rejection")
    return dict(role=role, persist_primary=role != "unpersisted_rejection",
        rejected_control_selected=control, control_hash_sha256=control_hash,
        stratum=dict(source_family=source_family, disposition=disposition))


def annotation_summary(metadata, label, trace):
    require(all(metadata.get(k) == value for k,value in v2.annotation_fields().items()),
            "Wrong annotation metadata epoch")
    require(label["annotation_epoch"] == v2.ANNOTATION_EPOCH
            and label["annotation_policy_sha256"] == v2.POLICY_SHA256
            and type(label["eligible"]) is bool and label["training_approved"] is False,
            "Invalid unapproved V2 label")
    evidence = label["query_selection_evidence"]
    require(label["annotation_input_metadata_sha256"] ==
            fingerprint(v2.canonical_annotation_metadata(metadata)) == evidence["input_fingerprints"]["metadata"]
            and label["query_selection_evidence_sha256"] == fingerprint(evidence),
            "Annotation fingerprint mismatch")
    require((trace is not None) == label["eligible"], "Trace disposition mismatch")
    strict = False
    if trace is not None:
        require(trace["annotation_epoch"] == v2.ANNOTATION_EPOCH and type(trace["passed"]) is bool
                and trace["passed"] == (trace["legacy_trace"]["passed"] is True and trace["fixed_grid"]["passed"] is True)
                and trace["passed"] == (evidence["selected_query"] is not None), "Changed full trace conjunction")
        strict = trace["passed"]
    return dict(**v2.annotation_fields(), source_family=label["source_plant_family"],
        source_target=label["conservative_view_cap_group"], target_id=label["target_id"],
        eligible=label["eligible"], strict=strict,
        disposition="strict" if strict else "hold" if label["eligible"] else "exclude",
        label_reason=label["reason"], clarity_reasons=deepcopy(label.get("clarity", {}).get("reasons")),
        trace_reasons=deepcopy(trace["reasons"]) if trace is not None else None,
        annotation_input_metadata_sha256=label["annotation_input_metadata_sha256"],
        in_memory_label_fingerprint_sha256=fingerprint(label),
        in_memory_trace_fingerprint_sha256=fingerprint(trace) if trace is not None else None,
        selection_evidence_sha256=label["query_selection_evidence_sha256"],
        input_fingerprints=deepcopy(evidence["input_fingerprints"]),
        freshness=deepcopy(metadata["synchronization"]["freshness"]))


def validate_live_buffers(metadata, rgb, depth, valid, instances, components, organs, target_mask, catalogue):
    """Same shape/dtype/C-order and native evidence gates even without storage.

    No encoding or allocation of substitute buffers; storage-specific byte
    roundtrips apply only to buffers actually persisted.
    """
    require(isinstance(metadata, dict) and "files" not in metadata
            and metadata.get("training_sample_approved") is False
            and metadata.get("training_approved", False) is False, "Fresh unapproved metadata required")
    cal = metadata["calibration"]
    require(cal["resolution"] == [1696,816] and cal["crop_resize"] is None
            and cal["depth_convention"] == "optical_axis_z_metres_not_ray_range", "Native uncropped optical-Z required")
    shape = (816,1696)
    for name, array, dtype, expected_shape in (
        ("rgb", rgb, np.uint8, (*shape, 3)),
        ("depth", depth, np.float32, shape),
        ("valid", valid, np.bool_, shape),
        ("instances", instances, np.uint32, shape),
        ("components", components, np.uint32, shape),
        ("organs", organs, np.uint8, shape),
        ("target_mask", target_mask, np.bool_, shape),
    ):
        require(isinstance(array, np.ndarray) and array.shape == expected_shape
                and array.dtype == dtype and array.flags.c_contiguous,
                "Invalid native shape/dtype/C-order: " + name)
    check_native_evidence(metadata, rgb, depth, components, target_mask.astype(np.uint8)*255, catalogue)


def validate_annotation_inputs(summary, metadata, valid, components, target_mask, catalogue):
    """Reject post-annotation input drift even when there will be no pixels."""
    sup = metadata["supervision"]
    _summary_contract(summary, metadata["sample_id"], sup["target_id"],
                      sup["conservative_view_cap_group"], sup["split_group"])
    expected = dict(catalogue=fingerprint(catalogue), **{
        name:hashlib.sha256(array.tobytes()).hexdigest() for name,array in
        (("valid",valid),("components",components),("target_mask",target_mask.astype(np.uint8)*255))})
    require(all(summary["input_fingerprints"][name] == pin for name,pin in expected.items()),
            "Native annotation inputs changed before persistence")


def callback_counts(rows):
    callbacks = [r for r in rows if r["state"] == CALLBACK_STATE]
    return dict(planned_decisions=len(rows), pre_render_rejections=len(rows)-len(callbacks),
        validated_callbacks=len(callbacks),
        orchestrator_requests=sum(r["render_budget"]["actual_orchestrator_requests"] for r in callbacks),
        persisted_candidates=sum(r["persistence"]["role"] == "strict_candidate" for r in callbacks),
        persisted_rejected_controls=sum(r["persistence"]["role"] == "rejected_control" for r in callbacks),
        persisted_other_rejections=sum(r["persistence"]["role"] == "all_frames_rejection" for r in callbacks),
        persisted_samples=sum(r["primary"] is not None for r in callbacks),
        unpersisted_attempts=sum(r["primary"] is None for r in callbacks),
        witness_samples=sum(r["witness"] is not None for r in callbacks))


def verify_epoch_documents(request, result):
    expected = implementation_bindings()
    settings = configuration(request["persistence_mode"], request["qualification_witness_save_all"])
    for document in (request, result):
        require(all(document.get(k) == value for k,value in settings.items()), "Mixed persistence policy/mode")
        require(all(document.get(k) == value for k,value in v2.annotation_fields().items())
                and document.get("annotation_policy") == v2.annotation_policy(), "Mixed annotation policy")
        require(document.get("worker_module") == WORKER_MODULE
                and document.get("worker_implementation_bindings") == expected
                and document.get("compact_implementation_bindings") == expected, "Wrong V3 implementation map")
    require(request["storage_qualification"] == result["storage_qualification"], "Storage proof mismatch")
    compact_qualification.verify_checked_qualification(request["storage_qualification"])


def _summary_contract(summary, name, target, cap_group, family):
    require(set(summary) == set(("annotation_epoch annotation_policy_sha256 source_family source_target target_id "
        "eligible strict disposition label_reason clarity_reasons trace_reasons annotation_input_metadata_sha256 "
        "in_memory_label_fingerprint_sha256 in_memory_trace_fingerprint_sha256 selection_evidence_sha256 "
        "input_fingerprints freshness").split()), "Unknown/missing annotation summary fields")
    require(all(summary[k] == v for k,v in v2.annotation_fields().items())
            and (summary["target_id"], summary["source_target"], summary["source_family"]) == (target, cap_group, family),
            "Wrong callback summary ancestry/epoch")
    require(type(summary["eligible"]) is bool and type(summary["strict"]) is bool
            and (not summary["strict"] or summary["eligible"])
            and summary["disposition"] == ("strict" if summary["strict"] else "hold" if summary["eligible"] else "exclude"),
            "Inconsistent declared annotation disposition")
    require(isinstance(summary["label_reason"], str) and summary["label_reason"], "Missing label reason")
    for key in ("clarity_reasons", "trace_reasons"):
        value = summary[key]
        require(value is None or isinstance(value, list) and all(isinstance(r, str) for r in value), "Invalid reason list")
    require((summary["trace_reasons"] is not None) == summary["eligible"]
            and (summary["in_memory_trace_fingerprint_sha256"] is not None) == summary["eligible"],
            "Unexpected missing/present declared trace")
    for key in ("annotation_input_metadata_sha256", "in_memory_label_fingerprint_sha256", "selection_evidence_sha256"):
        _hash(summary[key])
    if summary["eligible"]: _hash(summary["in_memory_trace_fingerprint_sha256"])
    inputs = summary["input_fingerprints"]
    require(set(inputs) == {"metadata","report","catalogue","rgb","depth","valid","components","target_mask"},
            "Incomplete native input fingerprints")
    for value in inputs.values(): _hash(value)
    require(inputs["metadata"] == summary["annotation_input_metadata_sha256"], "Summary metadata hash mismatch")
    fresh = summary["freshness"]
    require(set(fresh) == {"callback_sequence","camera_sha256","rgb_sha256","depth_sha256"}
            and type(fresh["callback_sequence"]) is int and fresh["callback_sequence"] > 0, "Invalid callback token")
    for key in ("camera_sha256","rgb_sha256","depth_sha256"): _hash(fresh[key])
    require(fresh["rgb_sha256"] == inputs["rgb"] and fresh["depth_sha256"] == inputs["depth"],
            "Summary native fingerprint mismatch")


def _reader(folder, stored):
    require(isinstance(stored, dict) and set(stored) == {"sample_sha256","label_sha256","query_trace"},
            "Real compact storage bindings required")
    _hash(stored["sample_sha256"]); _hash(stored["label_sha256"])
    reader = SampleReader(folder, expected_bindings={"sample.json":stored["sample_sha256"],
        "supervision/label.json":stored["label_sha256"]}, expected_json=(
            {"supervision/query_trace.json":stored["query_trace"]} if stored["query_trace"] is not None else {}))
    require(reader.manifest is not None, "V3 stores compact samples only")
    reader.verify_all()
    return reader


def _replay(reader, row, cap_group, family, generated):
    """Unchanged V2 native evidence/annotation/trace gates on exact stored bytes."""
    meta = reader.metadata
    require(meta.get("schema_version") == v2.SAMPLE_SCHEMA
            and all(meta.get(k) == v for k,v in v2.annotation_fields().items()), "Mixed sample annotation epoch")
    require(meta["sample_id"] == row["candidate_id"] and meta["supervision"]["target_id"] == row["target_id"]
            and meta["robot_snapshot"] == row["robot_snapshot"] and meta["geometry_screen"] == row["screen"],
            "Sample target/robot/screen differs from planned capture record")
    require(meta["supervision"]["conservative_view_cap_group"] == cap_group, "Sample source-target ancestry differs from plan")
    rgb = reader.image("inputs/rgb.png")
    depth = reader.array("inputs/depth_m.npy")
    valid = reader.image("inputs/depth_valid.png") != 0
    components = reader.array("supervision/component_id.npy")
    target = reader.image("supervision/target_visible.png")
    catalogue = reader.json("supervision/identities.json")["component_catalogue"]
    check_native_evidence(meta, rgb, depth, components, target, catalogue)
    label, trace = v2.annotate_for_storage(meta, generated["report"], rgb, depth, valid, components, catalogue,
                                         target_mask=target)
    require(label["target_id"] == row["target_id"] and label["source_plant_family"] == family
            and label["conservative_view_cap_group"] == cap_group, "Label ancestry differs from plan")
    require(label == reader.json("supervision/label.json"), "Replayed label differs")
    if label["eligible"]:
        require(trace == reader.json("supervision/query_trace.json"), "Replayed trace differs")
    else:
        require("supervision/query_trace.json" not in reader.manifest["files"], "Excluded sample contains a trace")
    require(annotation_summary(meta, label, trace) == row["annotation"], "Replayed annotation summary differs")
    require(meta["render_budget"] == row["render_budget"], "Sample render budget differs")
    return dict(candidate_id=row["candidate_id"], target_id=row["target_id"],
        source_family=family, source_target_group=cap_group, disposition=row["annotation"]["disposition"],
        label_replayed_exact=True, trace_replayed_exact=trace is not None,
        native_callback_hashes_verified=True, source_and_file_hashes_verified=True,
        sample_sha256=digest(reader.read("sample.json")), label_sha256=digest(reader.read("supervision/label.json")),
        rgb_sha256=digest(reader.read("inputs/rgb.png")),
        training_approved=False, source_cap_reset=False, physical_execution_approved=False, visual_review_performed=False)


def audit_capture(capture, plan_path, *, result_sha256, storage_root=None):
    code = implementation_bindings()
    capture, plan_path = Path(capture).resolve(), Path(plan_path).resolve()
    root = Path(storage_root).resolve() if storage_root is not None else capture
    require(not (capture/"failure.json").exists(), "Failed capture cannot be audited")
    pins = {str(capture/"result.json"):_hash(result_sha256),
            str(capture/"request.json"):sha256(capture/"request.json"), str(plan_path):sha256(plan_path)}
    verify_bindings(pins)
    result, request, plan = read_json(capture/"result.json"), read_json(capture/"request.json"), read_json(plan_path)
    verify_epoch_documents(request, result)
    mode, witness = request["persistence_mode"], request["qualification_witness_save_all"]
    validate_plan_mode(plan, mode, witness)
    require(Path(request["plan_path"]).resolve() == plan_path
            and request["plan_sha256"] == result["plan_sha256"] == pins[str(plan_path)]
            and result["request_sha256"] == pins[str(capture/"request.json")], "Unbound V3 completion")
    require(plan.get("split") == "train" and plan.get("resolution") == [1696,816]
            and all(plan.get(k) is False for k in ("training_approved","source_cap_reset",
                "physical_motion_commanded","hidden_cut_coordinates_executable")), "Unexpected plan contract")
    require(request.get("training_started") is False and result.get("training_approved") is False
            and result.get("source_cap_reset") is False and result.get("source_assets_unchanged") is True
            and result.get("state") == CAPTURE_STATE, "Incomplete/unsafe V3 capture")
    require(request["native_instance_backend"] == result["native_instance_backend"]
            and request["render_budget_profile"] == result["render_budget_profile"]
            and request["render_profile_experiment"] is False, "Capture profile differs from request")
    for key in ("source_bindings","implementation_bindings","prerequisite_bindings"): verify_bindings(plan[key])
    anchor = read_json(plan["anchor_pair_plan"])
    generated = load_for_inspection(anchor["variant_directory"], anchor["source_collection_plan"])
    planned = [(c,s) for c in plan["target_cases"] for s in c["views"]]
    require([r["candidate_id"] for r in result["records"]] == [s["candidate_id"] for c,s in planned],
            "Missing/reordered/duplicate planned decisions")
    if not witness: require(not (root/WITNESS_DIRECTORY).exists(), "Unrequested witness store")
    records, attempts, witness_records = [], [], []
    previous = None
    ordinal = 0
    for row,(case,spec) in zip(result["records"], planned):
        name = candidate_name(row["candidate_id"])
        require(row["requested_spec"] == spec and row["target_id"] == case["target_id"]
                and row["training_approved"] is False, "Record target/view differs from plan")
        require(read_json(capture/(name+"_decision.json")) == row, "Attempt sidecar differs from result")
        folder, witness_folder = sample_directories(root, name)
        if row["state"] in PRE_STATES:
            require(not folder.exists() and not witness_folder.exists()
                    and not any(k in row for k in ("primary","witness","annotation","callback_ordinal",
                                                   "sample_sha256","label_sha256","query_trace")),
                    "Pre-render reject has unexplained pixels/callback")
            continue
        require(row["state"] == CALLBACK_STATE and row["source_checks_passed"] is True,
                "Unknown/incomplete callback state")
        require(not any(k in row for k in ("sample_sha256","label_sha256","query_trace","captured_frames")),
                "V3 forbids ambiguous flat sample bindings")
        ordinal += 1
        require(type(row["callback_ordinal"]) is int and row["callback_ordinal"] == ordinal, "Callback ordinal mismatch")
        cap_group = case["conservative_view_cap_group"]
        summary = row["annotation"]
        _summary_contract(summary, name, row["target_id"], cap_group, plan["source_family"])
        fresh = summary["freshness"]
        if previous is not None:
            require(fresh["callback_sequence"] > previous["callback_sequence"]
                    and all(fresh[k] != previous[k] for k in ("camera_sha256","rgb_sha256","depth_sha256")),
                    "Stale callback across retained/skipped views")
        previous = fresh
        settings = budget_evidence(request["render_budget_profile"], ordinal-1)
        require(all(row["render_budget"].get(k) == v for k,v in settings.items())
                and type(row["render_budget"]["actual_orchestrator_requests"]) is int
                and row["render_budget"]["actual_orchestrator_requests"] == settings["native_step_calls"],
                "Warmup must follow validated callbacks, not persistence")
        expected = retention(request["plan_sha256"], plan["source_family"], summary["disposition"], name, mode)
        require(row["persistence"] == expected, "Changed fixed persistence/control policy")
        primary, secondary = row["primary"], row["witness"]
        require((primary is not None) == expected["persist_primary"] and (secondary is not None) == witness,
                "Missing/unexpected primary or witness")
        require(row["pixels_persisted"] is (primary is not None or secondary is not None)
                and row["independent_pixel_replay_possible"] is row["pixels_persisted"],
                "Incorrect pixel availability declaration")
        left = _reader(folder, primary) if primary is not None else None
        right = _reader(witness_folder, secondary) if secondary is not None else None
        if left is None: require(not folder.exists(), "Unpersisted attempt has a sample directory")
        if right is None: require(not witness_folder.exists(), "Unrequested witness pixels")
        checked = right if right is not None else left
        reviewed = _replay(checked, row, cap_group, plan["source_family"], generated) if checked is not None else None
        if left is not None and right is not None:
            require(primary == secondary and left.manifest["files"].keys() == right.manifest["files"].keys(),
                    "Same-callback witness storage bindings differ")
            for logical in left.manifest["files"]:
                require(left.read(logical) == right.read(logical), "Different same-callback logical bytes")
        if left is not None:
            records.append(dict(reviewed, sample=str(folder), persistence_role=expected["role"],
                dataset_candidate=expected["role"] == "strict_candidate", native_persistence_qualified=False))
        if right is not None:
            witness_records.append(dict(reviewed, sample=str(witness_folder), dataset_candidate=False,
                                        control_role="qualification_witness", native_persistence_qualified=False))
        attempts.append(dict(candidate_id=name, declared_disposition=summary["disposition"],
            persistence_role=expected["role"], pixels_persisted=row["pixels_persisted"],
            independently_pixel_replayed=reviewed is not None,
            review_basis="exact_saved_native_annotation_replay" if reviewed else
                         "producer_receipt_structure_only_no_pixels_or_independent_annotation_replay",
            training_approved=False, source_cap_reset=False))
    counts = callback_counts(result["records"])
    require(result["counts"] == counts and counts["validated_callbacks"] == ordinal, "Counter mismatch")
    require(not any(k in result for k in ("captured_frames","automatically_clear_annotation_candidates")),
            "Ambiguous legacy image counts forbidden")
    verify_bindings(pins); verify_bindings(code)
    for key in ("source_bindings","implementation_bindings","prerequisite_bindings"): verify_bindings(plan[key])
    verify_epoch_documents(request, result)
    require(not (capture/"failure.json").exists(), "Capture failed during replay")
    return dict(state=AUDIT_STATE, capture=str(capture), records=records, attempts=attempts,
        witness_records=witness_records, declared_worker_counts=counts,
        replay_verified_primary_samples=len(records), replay_verified_unique_callbacks=sum(
            r["independently_pixel_replayed"] for r in attempts),
        unreplayable_attempts=sum(not r["independently_pixel_replayed"] for r in attempts),
        skipped_annotations_independently_verified=witness or counts["unpersisted_attempts"] == 0,
        qualification_retained_and_skipped_branches_covered=bool(witness and counts["persisted_samples"]
                                                                and counts["unpersisted_attempts"]),
        same_callback_byte_pairs=counts["persisted_samples"] if witness else 0,
        **configuration(mode,witness), **v2.annotation_fields(), annotation_policy=v2.annotation_policy(),
        worker_module=WORKER_MODULE, worker_implementation_bindings=code,
        result_sha256=result_sha256, request_sha256=pins[str(capture/"request.json")], plan_sha256=pins[str(plan_path)],
        source_assets_unchanged=True, original_reviews_modified=False,
        training_approved=False, source_cap_reset=False, physical_execution_approved=False, inventory_integration_performed=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--result-sha256", required=True)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    plan = read_json(args.plan)
    protected = [args.capture.resolve(), args.plan.resolve(), *(Path(p).resolve()
        for key in ("source_bindings","implementation_bindings","prerequisite_bindings") for p in plan[key])]
    if args.storage_root is not None: protected.append(args.storage_root.resolve())
    require(not output.exists() and output != Path(output.anchor)
            and all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in protected),
            "New audit receipt outside capture and immutable inputs required")
    review = audit_capture(args.capture, args.plan, result_sha256=args.result_sha256, storage_root=args.storage_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, review)


if __name__ == "__main__":
    main()
