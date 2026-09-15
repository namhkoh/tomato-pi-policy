"""Synthetic CPU qualification only; never captured/approved observations."""
from copy import deepcopy
import ast
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from ..capture_contract import fingerprint
from ..dataset_review import read_json
from ..depth_preview import sha256
from ..native_budget import REFERENCE, TRIAL, evidence as budget_evidence
from . import compact_query_v3 as worker
from . import query_audit_v3 as audit
from .bundle import SampleReader
from .test_query_selection_v2 import native_case, seal

v2 = audit.v2


def write_json(path, value):
    """Overwrite only synthetic pytest artifacts for adversarial receipt tests.

    Production create-only helpers are untouched and still used by the worker.
    """
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")


@pytest.fixture(scope="module")
def callbacks():
    """Native-sized synthetic fixtures, not real native qualification."""
    cases = {}
    for index, disposition in enumerate(("strict", "hold", "exclude"), 1):
        args = native_case()
        meta, report, rgb, depth, valid, components, catalogue = args
        args[5] = components = components.astype(np.uint32)
        if disposition == "hold":
            components[408,950] = 2
        elif disposition == "exclude":
            components[:,878:909] = 0
        # Distinct synthetic frame fingerprints; these are NOT visibility edits
        # to real observations. Fixture pixels are outside all target probes.
        rgb[0,0,0] += index
        depth[0,0] += index*.001
        meta["calibration"]["synthetic_fixture_index"] = index
        meta.update(schema_version=v2.SAMPLE_SCHEMA, **v2.annotation_fields(),
            training_sample_approved=False, robot_snapshot={"fixture":True},
            geometry_screen={"passed":True,"fixture":True},
            render_budget=dict(budget_evidence(REFERENCE,0),
                               actual_orchestrator_requests=7,render_seconds=.125))
        meta["supervision"]["conservative_view_cap_group"] = meta["supervision"]["target_id"]
        seal(args)
        meta["synchronization"]["freshness"]["callback_sequence"] = 7
        label, trace = v2.annotate_for_storage(*args, target_mask=(components==1).astype(np.uint8)*255)
        assert label["eligible"] is (disposition != "exclude")
        assert (trace is not None and trace["passed"]) is (disposition == "strict")
        callback = dict(metadata=meta, rgb=rgb, depth=depth, valid=valid, components=components,
            catalogue=catalogue, instances=components.copy(), mapping={"1":"/fixture/Petiole","2":"/fixture/Main"},
            organs=np.zeros_like(components,dtype=np.uint8), target_mask=components==1, label=label, trace=trace)
        cases[disposition] = (report, callback)
    return cases


def chosen_pin(disposition, selected=False, name="unit_test_only"):
    # Test-only search covers both immutable sampling branches, not a production
    # seed knob. Production uses the external, already-pinned plan digest.
    for index in range(10000):
        pin = f"{index:064x}"
        if audit.retention(pin,"fixture",disposition,name,audit.STRICT_ONLY)["rejected_control_selected"] == selected:
            return pin
    raise AssertionError("Could not construct deterministic test branch")


def persist(tmp_path, callback, disposition, *, mode=audit.STRICT_ONLY, selected=False, witness=False):
    return worker.persist_callback(tmp_path, callback["metadata"]["sample_id"], **callback,
        plan_sha256=chosen_pin(disposition,selected), persistence=mode, qualification_witness=witness)


def test_policy_is_fixed_private_stratified_and_has_no_tuning_knob():
    policy = audit.persistence_policy()
    assert policy["reject_control_denominator"] == 64
    assert policy["default"] == audit.ALL_FRAMES
    assert policy["reject_control_salt"] == "greenhouse.reject_controls.donor_disposition.sha256.v1"
    assert policy["reject_strata"] == ["hold","exclude"]
    assert policy["annotation_epoch"] == v2.ANNOTATION_EPOCH
    assert fingerprint(policy) == audit.PERSISTENCE_POLICY_SHA256
    policy["reject_control_denominator"] = 1
    assert audit.persistence_policy()["reject_control_denominator"] == 64
    hashes = []
    for donor, disposition in (("fixture","hold"),("fixture","exclude"),("other_donor","hold")):
        result = audit.retention("1"*64,donor,disposition,"sample_001",audit.STRICT_ONLY)
        # Compute the specified canonical JSON hash independently.
        expected = hashlib.sha256(json.dumps([audit.SALT,"1"*64,donor,disposition,"sample_001"],
            sort_keys=True,allow_nan=False).encode()).hexdigest()
        assert result["control_hash_sha256"] == expected
        assert result["rejected_control_selected"] is (int(expected,16)%64 == 0)
        assert result == audit.retention("1"*64,donor,disposition,"sample_001",audit.STRICT_ONLY)
        hashes.append(expected)
    assert len(set(hashes)) == 3


@pytest.mark.parametrize("disposition",["strict","hold","exclude"])
def test_default_all_frames_persists_every_disposition(tmp_path,callbacks,disposition):
    callback = callbacks[disposition][1]
    saved = worker.persist_callback(tmp_path,callback["metadata"]["sample_id"],**callback,
                                     plan_sha256=chosen_pin(disposition))
    assert saved["primary"] is not None and saved["witness"] is None
    assert saved["pixels_persisted"] and saved["independent_pixel_replay_possible"]
    assert saved["annotation"]["disposition"] == disposition
    assert SampleReader(tmp_path/callback["metadata"]["sample_id"]).verify_all()


@pytest.mark.parametrize("disposition",["hold","exclude"])
@pytest.mark.parametrize("selected",[False,True])
def test_strict_mode_fixed_reject_control_or_truthful_absence(tmp_path,callbacks,disposition,selected):
    saved = persist(tmp_path,callbacks[disposition][1],disposition,selected=selected)
    assert (saved["primary"] is not None) is selected
    assert saved["witness"] is None and saved["pixels_persisted"] is selected
    assert saved["persistence"]["role"] == ("rejected_control" if selected else "unpersisted_rejection")
    if not selected:
        assert list(tmp_path.iterdir()) == []
        assert not any(k in saved for k in ("sample_sha256","label_sha256","query_trace"))
        assert "in_memory_label_fingerprint_sha256" in saved["annotation"]


@pytest.mark.parametrize("disposition",["strict","hold","exclude"])
def test_witness_same_callback_storage_and_every_logical_byte(tmp_path,callbacks,disposition):
    callback = callbacks[disposition][1]
    before = deepcopy(callback["metadata"])
    saved = persist(tmp_path,callback,disposition,witness=True)
    name = before["sample_id"]
    witness = SampleReader(tmp_path/audit.WITNESS_DIRECTORY/name)
    assert witness.verify_all()
    assert v2.canonical_annotation_metadata(witness.metadata) == before
    assert callback["metadata"] == before and "files" not in before
    assert witness.json("supervision/label.json") == callback["label"]
    for logical, array in (("inputs/depth_m.npy",callback["depth"]),
                           ("supervision/component_id.npy",callback["components"]),
                           ("supervision/renderer_instance_id.npy",callback["instances"])):
        stream = io.BytesIO()
        np.save(stream,array,allow_pickle=False)
        assert witness.read(logical) == stream.getvalue()
        assert witness.array(logical).tobytes() == array.tobytes()
    assert np.array_equal(witness.image("inputs/rgb.png"),callback["rgb"])
    assert saved["witness"] is not None and saved["pixels_persisted"]
    if disposition == "strict":
        assert saved["primary"] == saved["witness"]
        primary = SampleReader(tmp_path/name)
        assert set(primary.manifest["files"]) == set(witness.manifest["files"])
        for logical in primary.manifest["files"]:
            assert primary.read(logical) == witness.read(logical)
    else:
        assert saved["primary"] is None and not (tmp_path/name).exists()


@pytest.mark.parametrize("fault",[
    "rgb_dtype","depth_dtype","valid_dtype","instances_dtype","components_dtype",
    "organs_dtype","target_dtype","rgb_shape","depth_order","rgb_fingerprint",
    "depth_fingerprint","target_identity","files","approval","crop","annotation_hash","trace_conjunction",
    "valid_after_annotation","components_after_annotation","catalogue_after_annotation",
])
def test_skipping_never_bypasses_live_integrity(tmp_path,callbacks,fault):
    callback = deepcopy(callbacks["hold"][1])
    if fault.endswith("_dtype"):
        key = fault[:-6]
        key = "target_mask" if key == "target" else key
        callback[key] = callback[key].astype(np.float64)
    elif fault == "rgb_shape": callback["rgb"] = callback["rgb"][:1]
    elif fault == "depth_order": callback["depth"] = np.asfortranarray(callback["depth"])
    elif fault == "rgb_fingerprint": callback["rgb"][0,0,0] ^= 1
    elif fault == "depth_fingerprint": callback["depth"][0,0] += 1
    elif fault == "target_identity": callback["target_mask"][0,0] = True
    elif fault == "files": callback["metadata"]["files"] = {}
    elif fault == "approval": callback["metadata"]["training_sample_approved"] = True
    elif fault == "crop": callback["metadata"]["calibration"]["crop_resize"] = [0,0,1,1]
    elif fault == "annotation_hash": callback["label"]["query_selection_evidence_sha256"] = "0"*64
    elif fault == "trace_conjunction": callback["trace"]["passed"] = True
    elif fault == "valid_after_annotation": callback["valid"][0,0] = False
    elif fault == "components_after_annotation": callback["components"][0,0] = 2
    elif fault == "catalogue_after_annotation": callback["catalogue"][1]["fixture_extra"] = True
    with pytest.raises((ValueError,KeyError)):
        persist(tmp_path,callback,"hold")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("name",["",".","..","../sample","a/b",r"a\b","C:sample","qualification_witness",
    "QUALIFICATION_WITNESS","CON","nul","LPT1","sample.","sample "," sample","a?b"])
def test_unsafe_or_reserved_candidate_rejected(name):
    with pytest.raises(ValueError): audit.candidate_name(name)


@pytest.mark.parametrize("mode,witness", [("strict",False),(None,False),(audit.ALL_FRAMES,1)])
def test_invalid_configuration_rejected(mode,witness):
    with pytest.raises(ValueError): audit.configuration(mode,witness)


def test_witness_bounded_without_capping_regular_capture():
    plan = {"target_cases":[{"views":[{"candidate_id":f"s{i}"} for i in range(13)]}]}
    audit.validate_plan_mode(plan,audit.ALL_FRAMES,False)
    with pytest.raises(ValueError,match="12"): audit.validate_plan_mode(plan,audit.ALL_FRAMES,True)
    plan["target_cases"][0]["views"].pop()
    audit.validate_plan_mode(plan,audit.STRICT_ONLY,True)
    plan["target_cases"][0]["views"].append({"candidate_id":"s0"})
    with pytest.raises(ValueError): audit.validate_plan_mode(plan,audit.ALL_FRAMES,False)
    with pytest.raises(ValueError):
        audit.validate_plan_mode({"target_cases":[{"views":[{"candidate_id":"A"},{"candidate_id":"a"}]}]},
                                 audit.ALL_FRAMES,False)


def documents(mode,witness):
    bindings = audit.implementation_bindings()
    return dict(**audit.configuration(mode,witness),**v2.annotation_fields(),
        annotation_policy=v2.annotation_policy(),worker_module=audit.WORKER_MODULE,
        worker_implementation_bindings=bindings,compact_implementation_bindings=bindings,
        storage_qualification={"synthetic_cpu_fixture":True},native_instance_backend="legacy",
        render_budget_profile=REFERENCE)


@pytest.fixture
def audit_case(tmp_path,monkeypatch,callbacks):
    """Real codec, selector and replay; stub ONLY external source/proof loading."""
    monkeypatch.setattr(audit,"load_for_inspection",lambda *_: {"report":callbacks["strict"][0]})
    monkeypatch.setattr(audit.compact_qualification,"verify_checked_qualification",lambda _: True)
    def make(disposition="hold", *, witness=False, mode=audit.STRICT_ONLY, selected=False):
        callback = deepcopy(callbacks[disposition][1])
        name = callback["metadata"]["sample_id"]
        capture = tmp_path/"capture"
        capture.mkdir()
        anchor = tmp_path/"anchor.json"
        write_json(anchor,dict(variant_directory="synthetic_variant",source_collection_plan="synthetic_collection"))
        pins = {str(anchor):sha256(anchor)}
        spec = dict(candidate_id=name,synthetic_cpu_fixture=True)
        target = callback["metadata"]["supervision"]["target_id"]
        plan = dict(split="train",resolution=[1696,816],source_family="fixture",
            training_approved=False,source_cap_reset=False,physical_motion_commanded=False,
            hidden_cut_coordinates_executable=False,source_bindings=pins,prerequisite_bindings=pins,
            implementation_bindings={str(Path(v2.__file__)):sha256(Path(v2.__file__))},
            anchor_pair_plan=str(anchor),target_cases=[
                dict(target_id=target,conservative_view_cap_group=target,views=[spec])])
        plan_path = tmp_path/"plan.json"
        for nonce in range(10000):
            plan["synthetic_test_nonce"] = nonce
            write_json(plan_path,plan)
            pin = sha256(plan_path)
            if disposition == "strict" or audit.retention(pin,"fixture",disposition,name,mode)["rejected_control_selected"] == selected:
                break
        else: raise AssertionError("No synthetic fixture branch")
        saved = worker.persist_callback(capture,name,**callback,plan_sha256=pin,
            persistence=mode,qualification_witness=witness)
        request = dict(documents(mode,witness),plan_path=str(plan_path),plan_sha256=pin,
                       training_started=False,render_profile_experiment=False)
        write_json(capture/"request.json",request)
        row = dict(candidate_id=name,target_id=target,requested_spec=spec,training_approved=False,
            robot_snapshot=callback["metadata"]["robot_snapshot"],screen=callback["metadata"]["geometry_screen"],
            state=audit.CALLBACK_STATE,source_checks_passed=True,callback_ordinal=1,
            render_budget=callback["metadata"]["render_budget"],**saved)
        result = dict(documents(mode,witness),state=audit.CAPTURE_STATE,training_approved=False,
            source_cap_reset=False,source_assets_unchanged=True,records=[row],counts=audit.callback_counts([row]),
            request_sha256=sha256(capture/"request.json"),plan_sha256=pin)
        case = dict(capture=capture,plan_path=plan_path,plan=plan,request=request,result=result,
                    callback=callback,row=row,folder=capture/name,anchor=anchor)
        save_case(case)
        return case
    return make


def save_case(case, *, sidecar=True):
    if sidecar:
        for row in case["result"]["records"]:
            write_json(case["capture"]/(row["candidate_id"]+"_decision.json"),row)
    write_json(case["capture"]/"result.json",case["result"])
    case["result_sha256"] = sha256(case["capture"]/"result.json")


def run_audit(case):
    return audit.audit_capture(case["capture"],case["plan_path"],result_sha256=case["result_sha256"])


@pytest.mark.parametrize("disposition",["strict","hold","exclude"])
@pytest.mark.parametrize("witness",[False,True])
@pytest.mark.parametrize("mode",audit.MODES)
def test_actual_end_to_end_persistence_replay(audit_case,disposition,witness,mode):
    case = audit_case(disposition,witness=witness,mode=mode)
    review = run_audit(case)
    primary = mode == audit.ALL_FRAMES or disposition == "strict"
    replayed = primary or witness
    assert review["state"] == audit.AUDIT_STATE
    assert review["native_persistence_qualified"] is False
    assert review["declared_worker_counts"]["validated_callbacks"] == 1
    assert review["declared_worker_counts"]["orchestrator_requests"] == 7
    assert review["replay_verified_primary_samples"] == int(primary)
    assert review["replay_verified_unique_callbacks"] == int(replayed)
    assert review["unreplayable_attempts"] == int(not replayed)
    assert review["same_callback_byte_pairs"] == int(witness and primary)
    assert review["skipped_annotations_independently_verified"] is replayed
    assert review["attempts"][0]["independently_pixel_replayed"] is replayed
    assert len(review["witness_records"]) == int(witness)
    for record in review["records"]:
        assert record["dataset_candidate"] is (disposition=="strict")
        assert record["label_replayed_exact"] and record["native_callback_hashes_verified"]
    for record in review["witness_records"]:
        assert not record["dataset_candidate"]
    for key in ("training_approved","source_cap_reset","inventory_integration_performed"):
        assert review[key] is False
    if not replayed:
        assert review["records"] == []
        assert "producer_receipt_structure_only_no_pixels" in review["attempts"][0]["review_basis"]


def test_fixed_rejected_control_is_replayed_but_not_a_dataset_candidate(audit_case):
    review = run_audit(audit_case("hold",selected=True))
    assert review["declared_worker_counts"]["persisted_rejected_controls"] == 1
    assert review["records"][0]["persistence_role"] == "rejected_control"
    assert not review["records"][0]["dataset_candidate"]


@pytest.mark.parametrize("fault",[
    "external_result_pin","request_pin","plan_pin","source_pin","worker_map","annotation_policy",
    "persistence_policy","policy_salt","state","approval","source_changed","failure",
    "missing_row","duplicate_row","wrong_target","wrong_spec","ordinal","counts","budget",
    "skip_fake_sample_hash","skip_fake_directory","skip_fake_witness","pixel_replay_claim",
    "changed_retention","summary_ancestry","summary_hash","freshness_sequence","input_hash",
    "sidecar","legacy_counts","missing_primary",
])
def test_audit_fail_closed_on_receipt_or_integrity_tamper(audit_case,fault):
    case = audit_case("strict" if fault=="missing_primary" else "hold")
    result,row = case["result"],case["row"]
    if fault=="external_result_pin": case["result_sha256"]="0"*64
    elif fault=="request_pin": result["request_sha256"]="0"*64
    elif fault=="plan_pin": result["plan_sha256"]="0"*64
    elif fault=="source_pin": write_json(case["anchor"],{"changed":True})
    elif fault=="worker_map": result["worker_implementation_bindings"]={}
    elif fault=="annotation_policy": result["annotation_policy_sha256"]="0"*64
    elif fault=="persistence_policy": result["persistence_policy_sha256"]="0"*64
    elif fault=="policy_salt": result["persistence_policy"]["reject_control_salt"]="other"
    elif fault=="state": result["state"]=v2.CAPTURE_STATE
    elif fault=="approval": result["training_approved"]=True
    elif fault=="source_changed": result["source_assets_unchanged"]=False
    elif fault=="failure": write_json(case["capture"]/"failure.json",{"error":"fixture"})
    elif fault=="missing_row": result["records"]=[]
    elif fault=="duplicate_row": result["records"].append(deepcopy(row))
    elif fault=="wrong_target": row["target_id"]="other/Petiole"
    elif fault=="wrong_spec": row["requested_spec"]["synthetic_cpu_fixture"]=False
    elif fault=="ordinal": row["callback_ordinal"]=0
    elif fault=="counts": result["counts"]["validated_callbacks"]=0
    elif fault=="budget": row["render_budget"]["requested_subframes"]=8
    elif fault=="skip_fake_sample_hash": row["sample_sha256"]="0"*64
    elif fault=="skip_fake_directory": case["folder"].mkdir()
    elif fault=="skip_fake_witness": row["witness"]={"sample_sha256":"0"*64}
    elif fault=="pixel_replay_claim": row["independent_pixel_replay_possible"]=True
    elif fault=="changed_retention": row["persistence"]["control_hash_sha256"]="0"*64
    elif fault=="summary_ancestry": row["annotation"]["source_family"]="other"
    elif fault=="summary_hash": row["annotation"]["in_memory_label_fingerprint_sha256"]="invalid"
    elif fault=="freshness_sequence": row["annotation"]["freshness"]["callback_sequence"]=False
    elif fault=="input_hash": row["annotation"]["input_fingerprints"]["rgb"]="0"*64
    elif fault=="sidecar": write_json(case["capture"]/(row["candidate_id"]+"_decision.json"),{})
    elif fault=="legacy_counts": result["captured_frames"]=1
    elif fault=="missing_primary": row["primary"]=None
    if fault != "external_result_pin": save_case(case,sidecar=fault!="sidecar")
    with pytest.raises((ValueError,KeyError,FileNotFoundError)): run_audit(case)


def test_unpersisted_annotation_digests_are_not_claimed_independently_verified(audit_case):
    case = audit_case("hold")
    # A structurally valid producer declaration cannot prove absent pixels.
    case["row"]["annotation"]["in_memory_label_fingerprint_sha256"] = "a"*64
    save_case(case)
    review = run_audit(case)
    assert review["unreplayable_attempts"] == 1
    assert not review["skipped_annotations_independently_verified"]
    assert not review["attempts"][0]["independently_pixel_replayed"]
    assert not review["native_persistence_qualified"]


@pytest.mark.parametrize("witness",[False,True])
def test_stored_annotation_summary_tamper_rejected_by_full_replay(audit_case,witness):
    case = audit_case("strict",witness=witness)
    case["row"]["annotation"]["in_memory_label_fingerprint_sha256"]="a"*64
    save_case(case)
    with pytest.raises(ValueError,match="annotation summary"): run_audit(case)


def test_witness_skip_replays_and_detects_altered_producer_summary(audit_case):
    case = audit_case("hold",witness=True)
    case["row"]["annotation"]["in_memory_label_fingerprint_sha256"]="a"*64
    save_case(case)
    with pytest.raises(ValueError,match="annotation summary"): run_audit(case)


def test_source_binding_rechecked_after_annotation(audit_case,monkeypatch):
    case = audit_case("strict")
    original = audit._replay
    def changed(*args,**kwargs):
        value = original(*args,**kwargs)
        write_json(case["anchor"],{"changed_during_replay":True})
        return value
    monkeypatch.setattr(audit,"_replay",changed)
    with pytest.raises(ValueError): run_audit(case)


def test_v2_auditor_does_not_silently_admit_v3_documents():
    with pytest.raises(ValueError):
        v2.verify_epoch_documents(documents(audit.STRICT_ONLY,False),documents(audit.STRICT_ONLY,False))


def test_no_worker_import_or_native_import_in_auditor():
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            text = ast.unparse(node)
            assert not any(s in text for s in ("compact_query_v3","isaacsim","omni","pxr","boundary_bindings"))


def test_replay_keeps_explicit_full_native_and_annotation_gates():
    tree = ast.parse(Path(audit.__file__).read_text(encoding="utf-8"))
    replay = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="_replay")
    calls = [(n.lineno,ast.unparse(n.func)) for n in ast.walk(replay) if isinstance(n,ast.Call)]
    native = next(line for line,fn in calls if fn=="check_native_evidence")
    annotation = next(line for line,fn in calls if fn=="v2.annotate_for_storage")
    assert native < annotation
    source = ast.unparse(replay)
    for guard in ("label == reader.json('supervision/label.json')",
                  "trace == reader.json('supervision/query_trace.json')",
                  "annotation_summary(meta, label, trace) == row['annotation']",
                  "meta['render_budget'] == row['render_budget']"):
        assert guard in source


def rebind_fixture(case):
    """Bind a changed synthetic multi-callback plan before writing its receipts."""
    for nonce in range(10000):
        case["plan"]["synthetic_test_nonce"] = nonce
        write_json(case["plan_path"],case["plan"])
        pin = sha256(case["plan_path"])
        decisions = [audit.retention(pin,"fixture",r["annotation"]["disposition"],
            r["candidate_id"],audit.STRICT_ONLY) for r in case["result"]["records"]]
        if all(d["persist_primary"] == (r["primary"] is not None)
               for r,d in zip(case["result"]["records"],decisions)):
            break
    else: raise AssertionError("No synthetic branch")
    for row,decision in zip(case["result"]["records"],decisions):
        row["persistence"] = decision
    case["request"]["plan_sha256"] = pin
    write_json(case["capture"]/"request.json",case["request"])
    case["result"]["request_sha256"] = sha256(case["capture"]/"request.json")
    case["result"]["plan_sha256"] = pin
    case["result"]["counts"] = audit.callback_counts(case["result"]["records"])
    save_case(case)
    return pin


@pytest.mark.parametrize("fault",[None,"warmup_reset","stale_sequence","stale_camera","stale_rgb","stale_depth"])
def test_skipped_consecutive_callbacks_keep_freshness_and_warmup(audit_case,fault):
    case = audit_case("hold")
    first = case["row"]
    second = deepcopy(first)
    second["candidate_id"] = "second_skipped"
    second["requested_spec"]["candidate_id"] = second["candidate_id"]
    second["callback_ordinal"] = 2
    fresh = second["annotation"]["freshness"]
    fresh["callback_sequence"] = 8
    for key in ("camera_sha256","rgb_sha256","depth_sha256"):
        fresh[key] = fingerprint(["synthetic_not_replayable",key])
    for key in ("rgb","depth"):
        second["annotation"]["input_fingerprints"][key] = fresh[key+"_sha256"]
    first["render_budget"] = dict(budget_evidence(TRIAL,0),actual_orchestrator_requests=7)
    second["render_budget"] = dict(budget_evidence(TRIAL,1),actual_orchestrator_requests=1)
    for document in (case["request"],case["result"]): document["render_budget_profile"] = TRIAL
    case["result"]["records"].append(second)
    case["plan"]["target_cases"][0]["views"].append(deepcopy(second["requested_spec"]))
    rebind_fixture(case)
    if fault == "warmup_reset":
        second["render_budget"] = dict(budget_evidence(TRIAL,0),actual_orchestrator_requests=7)
    elif fault == "stale_sequence":
        fresh["callback_sequence"] = first["annotation"]["freshness"]["callback_sequence"]
    elif fault is not None:
        key = fault[6:]+"_sha256"
        fresh[key] = first["annotation"]["freshness"][key]
        if key != "camera_sha256":
            second["annotation"]["input_fingerprints"][key[:-7]] = fresh[key]
    save_case(case)
    if fault is not None:
        with pytest.raises(ValueError): run_audit(case)
    else:
        review = run_audit(case)
        assert review["declared_worker_counts"]["validated_callbacks"] == 2
        assert review["declared_worker_counts"]["orchestrator_requests"] == 8
        assert review["unreplayable_attempts"] == 2 and review["records"] == []
        assert not review["native_persistence_qualified"]


def test_real_mixed_witness_replays_both_branches_once_per_callback(audit_case,callbacks):
    case = audit_case("hold",witness=True)
    report,callback = callbacks["strict"]
    callback = deepcopy(callback)
    meta = callback["metadata"]
    name = meta["sample_id"] = "second_strict"
    meta["synchronization"]["freshness"]["callback_sequence"] = 14
    meta["render_budget"] = dict(budget_evidence(REFERENCE,1),actual_orchestrator_requests=7,render_seconds=.125)
    callback["label"],callback["trace"] = v2.annotate_for_storage(meta,report,callback["rgb"],callback["depth"],
        callback["valid"],callback["components"],callback["catalogue"],
        target_mask=callback["target_mask"].astype(np.uint8)*255)
    saved = worker.persist_callback(case["capture"],name,**callback,
        plan_sha256=case["request"]["plan_sha256"],persistence=audit.STRICT_ONLY,qualification_witness=True)
    row = dict(candidate_id=name,target_id=meta["supervision"]["target_id"],
        requested_spec=dict(candidate_id=name,synthetic_cpu_fixture=True),training_approved=False,
        robot_snapshot=meta["robot_snapshot"],screen=meta["geometry_screen"],state=audit.CALLBACK_STATE,
        source_checks_passed=True,callback_ordinal=2,render_budget=meta["render_budget"],**saved)
    case["result"]["records"].append(row)
    case["plan"]["target_cases"][0]["views"].append(deepcopy(row["requested_spec"]))
    rebind_fixture(case)
    review = run_audit(case)
    assert review["qualification_retained_and_skipped_branches_covered"]
    assert review["replay_verified_unique_callbacks"] == 2
    assert review["replay_verified_primary_samples"] == review["same_callback_byte_pairs"] == 1
    assert len(review["witness_records"]) == 2
    assert review["declared_worker_counts"]["validated_callbacks"] == 2
    assert review["declared_worker_counts"]["orchestrator_requests"] == 14
    assert review["unreplayable_attempts"] == 0
    assert not review["native_persistence_qualified"]  # CPU fixture cannot qualify native persistence.


@pytest.mark.parametrize("store",["primary","witness"])
def test_corrupted_real_compact_payload_is_rejected(audit_case,store):
    case = audit_case("strict",witness=True)
    folder = case["folder"] if store=="primary" else case["capture"]/audit.WITNESS_DIRECTORY/case["row"]["candidate_id"]
    reader = SampleReader(folder)
    path = folder/reader.manifest["files"]["inputs/depth_m.npy"]["stored_path"]
    contents = bytearray(path.read_bytes())
    contents[-1] ^= 1
    path.write_bytes(contents)  # Deliberate corruption of synthetic pytest artifact only.
    with pytest.raises(ValueError): run_audit(case)
