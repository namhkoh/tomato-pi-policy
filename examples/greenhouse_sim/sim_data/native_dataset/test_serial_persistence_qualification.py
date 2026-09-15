"""Synthetic CPU tests only: no native runs, approvals or real data mutations."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import ast
import json

import numpy as np
import pytest

from . import serial_persistence_qualification as k
from .test_query_audit_v3 import callbacks
from .test_query_selection_v2 import seal


def put(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding="utf-8")
    return k.oc.sha256(path)


def _names():
    return [f"SubStem_{n}_view_{i:03d}" for n in (41,42) for i in range(1,7)]


@pytest.fixture
def registered(tmp_path,monkeypatch):
    root = tmp_path/"prior"
    source_plan = tmp_path/"original"/"plan.json"
    source = {str(source_plan):put(source_plan,dict(package=str(tmp_path/"package")))}
    base = dict(source_collection_plan=str(source_plan),source_capture=str(tmp_path/"source_capture"),
        variant_directory=str(tmp_path/"variant"),prerequisite_directory=str(tmp_path/"sensor"))
    base_path = tmp_path/"base"/"plan.json"
    source[str(base_path)] = put(base_path,base)
    proof_path = tmp_path/"proof"/"qualification.json"
    proof_pin = put(proof_path,dict(synthetic_cpu_only=True))
    proof = dict(schema=k.storage.SCHEMA,storage_backend=k.storage.STORAGE_BACKEND,path=str(proof_path),
        sha256=proof_pin,bindings={str(proof_path):proof_pin})
    def qualify(path,*,expected_sha256):
        k.oc.pin(path,expected_sha256)
        return deepcopy(proof)
    monkeypatch.setattr(k.storage,"check_qualification",qualify)
    monkeypatch.setattr(k.storage,"verify_checked_qualification",lambda p:k.oc.bind_all(p["bindings"]))
    monkeypatch.setattr(k.reference.batch,"check",lambda p,**kw:deepcopy(base))
    monkeypatch.setattr(k.reference.batch,"capture_jobs",lambda p,a:[None]*p["maximum_native_frames"])
    folder = root/"synthetic_seed101"
    plan_path = folder/"plan.json"
    plan = dict(split="train",source_family="seed101_full",maximum_native_frames=12,resolution=[1696,816],
        training_approved=False,source_cap_reset=False,physical_motion_commanded=False,
        hidden_cut_coordinates_executable=False,anchor_pair_plan=str(base_path),
        source_bindings=source,implementation_bindings=source,prerequisite_bindings=source,
        target_cases=[dict(base_pair_plan=str(base_path),target_id=f"seed101_full_cr_4000000/SubStem_{n}",
            conservative_view_cap_group=f"seed101_full/SubStem_{n}",
            views=[dict(candidate_id=f"SubStem_{n}_view_{i:03d}") for i in range(1,7)]) for n in (41,42)])
    # Fixture-only pin selection; production accepts one literal historical pin
    # and exposes NO plan nonce, salt or control-selection override.
    for index in range(10000):
        plan["synthetic_test_nonce"] = index
        pin = put(plan_path,plan)
        if all(not k.audit.retention(pin,"seed101_full",d,name,k.audit.STRICT_ONLY)["rejected_control_selected"]
               for name in _names() for d in ("hold","exclude")):
            break
    else: raise AssertionError("Could not make synthetic no-control fixture")
    common = dict(native_instance_backend="fast",render_budget_profile="warm56_then8_trial",
                  render_profile_experiment=False)
    rp = put(folder/"capture/request.json",dict(common,plan_path=str(plan_path),plan_sha256=pin,training_started=False))
    sp = put(folder/"capture/result.json",dict(common,state="native_generated_multiview_pilot_complete_pending_review",
        training_approved=False,source_assets_unchanged=True,source_cap_reset=False,records=[{}]*12,
        target_count=2,captured_frames=11,automatically_clear_annotation_candidates=4))
    counts = {"accept_strict_automatic_annotation_candidate":4,"exclude_geometry_or_visibility":7}
    ap = put(folder/"automatic_audit.json",dict(state="completed_automatic_annotation_replay",
        capture=str(folder/"capture"),plan_sha256=pin,request_sha256=rp,result_sha256=sp,
        counts=counts,training_approved=False,original_reviews_modified=False))
    cp = put(folder/"campaign_result.json",dict(state="native_complete_pending_review",native_exit_code=0,
        job=str(folder),source_family="seed101_full",training_approved=False,captured=11,automatic=4,audit_counts=counts))
    case = (folder.name,12,2,11,4,pin,rp,sp,ap,cp)
    monkeypatch.setattr(k.reference,"_PRIOR",root)
    monkeypatch.setattr(k.reference,"CASES",(case,))
    monkeypatch.setattr(k,"SOURCE_PLAN",plan_path)
    monkeypatch.setattr(k,"PLAN_SHA256",pin)
    fields = dict(**k.ANNOTATION,persistence_policy_sha256=k.PERSISTENCE_POLICY_SHA256,
        storage_qualification=str(proof_path),storage_qualification_sha256=proof_pin,
        matched_short=dict(capture_path=str(folder/"capture"),request_sha256=rp,result_sha256=sp,
            audit_path=str(folder/"automatic_audit.json"),audit_sha256=ap,
            completion_path=str(folder/"campaign_result.json"),completion_sha256=cp))
    return SimpleNamespace(path=plan_path,pin=pin,plan=plan,fields=fields,proof=proof,folder=folder,source=source)


def check(f):
    return k.validate_plan(f.path,f.pin,f.fields)


def test_exact_real_allowlist_control_coverage_without_loading_real_reports():
    assert k.SOURCE_PLAN.name == "plan.json" and k.SOURCE_PLAN.parent.name == "job_0001_seed101_full_round1"
    assert k.PLAN_SHA256 == "b1f448a6afd5d4af5ed6f1e203f7842c29a0ab962587fa4da2503a5e480ee7dd"
    assert len(_names()) == 12
    assert all(not k.audit.retention(k.PLAN_SHA256,"seed101_full",d,name,k.audit.STRICT_ONLY)[
        "rejected_control_selected"] for name in _names() for d in ("hold","exclude"))
    assert k.JOB_FIELDS == (*k.reference.JOB_FIELDS,"persistence_policy_sha256")


def test_code_bindings_pin_frozen_sources_and_exclude_runner():
    bindings = k.implementation_bindings()
    assert len(bindings) == 27
    for name,pin in k.FROZEN_CODE.items():
        assert bindings[str(Path(k.__file__).parent/name)] == pin
    assert not any(Path(p).name in ("serial_phases.py","boundary_bindings.py") for p in bindings)


def test_validation_private_handoff_and_fixed_command(registered,tmp_path):
    before = deepcopy(registered.fields)
    checked = check(registered)
    command = k.command(str(tmp_path/"runtime/python.bat"),tmp_path/"job/submitted/plan.json",
                        registered.pin,tmp_path/"job/capture",checked)
    assert command[1:3] == ["-m",k.audit.WORKER_MODULE]
    for flag,value in (("--persistence",k.audit.STRICT_ONLY),("--render-budget","reference56"),
                       ("--instance-backend","fast"),("--persistence-policy-sha256",k.PERSISTENCE_POLICY_SHA256),
                       ("--plan-sha256",registered.pin)):
        assert command[command.index(flag)+1] == value
    assert command.count("--qualification-witness-save-all") == 1 and "--profile-render" not in command
    assert checked["count"] == 12 and registered.fields == before
    checked["qualification"]["matched_short"]["capture_path"] = "changed_private_copy"
    assert registered.fields == before
    for source in (registered.folder,tmp_path/"package",tmp_path/"source_capture",tmp_path/"original"):
        with pytest.raises(ValueError): k.oc.new_destination(source/"new",checked["roots"])


@pytest.mark.parametrize("fault",[
    "source_path","source_pin","persistence_policy","annotation_policy","epoch","extra_field","proof_pin",
    "historical_result","historical_failure","historical_done","new_family","heldout","count_bool","count",
    "target","cap_group","candidate","order","geometry_source","worker_pin","audit_pin","reference_pin",
])
def test_registration_fails_closed(registered,monkeypatch,fault):
    f = registered
    if fault=="source_path": f.path = f.path.parent/"alias.json"
    elif fault=="source_pin": f.pin = "0"*64
    elif fault=="persistence_policy": f.fields["persistence_policy_sha256"]="0"*64
    elif fault=="annotation_policy": f.fields["annotation_policy_sha256"]="0"*64
    elif fault=="epoch": f.fields["annotation_epoch"]="old"
    elif fault=="extra_field": f.fields["persistence"]="all_frames"
    elif fault=="proof_pin": f.fields["storage_qualification_sha256"]="0"*64
    elif fault=="historical_result": put(f.folder/"capture/result.json",{})
    elif fault=="historical_failure": put(f.folder/"capture/failure.json",{})
    elif fault=="historical_done": put(f.folder/"campaign_result.json",{})
    elif fault=="geometry_source": put(Path(next(iter(f.source))),{})
    elif fault.endswith("_pin"):
        name = {"worker_pin":"compact_query_v3.py","audit_pin":"query_audit_v3.py",
                "reference_pin":"serial_query_qualification.py"}[fault]
        monkeypatch.setitem(k.FROZEN_CODE,name,"0"*64)
    else:
        plan = deepcopy(f.plan)
        if fault=="new_family": plan["source_family"]="seed73_full"
        elif fault=="heldout": plan["split"]="heldout"
        elif fault=="count_bool": plan["maximum_native_frames"]=True
        elif fault=="count": plan["maximum_native_frames"]=11
        elif fault=="target": plan["target_cases"][0]["target_id"]="foreign/Petiole"
        elif fault=="cap_group": plan["target_cases"][0]["conservative_view_cap_group"]="new_donor/SubStem_41"
        elif fault=="candidate": plan["target_cases"][0]["views"][0]["candidate_id"]="changed"
        elif fault=="order": plan["target_cases"].reverse()
        f.pin = put(f.path,plan)
        monkeypatch.setattr(k,"PLAN_SHA256",f.pin)  # Structural guards even with synthetic allowlist repin.
    with pytest.raises((ValueError,KeyError,FileNotFoundError)): check(f)


@pytest.mark.parametrize("fault",["qualification","plan","count","proof","source","matched"])
def test_checked_handoff_drift_rejected_before_command(registered,tmp_path,fault):
    checked = check(registered)
    if fault=="qualification": checked["qualification"]["persistence_policy_sha256"]="0"*64
    elif fault=="plan": checked["plan"]["source_family"]="new_donor"
    elif fault=="count": checked["count"]=1
    elif fault=="proof": checked["storage"]["sha256"]="0"*64
    elif fault=="source": put(Path(next(iter(registered.source))),{})
    elif fault=="matched": checked["matched_short"]["old_buffer_replay_performed"]=True
    with pytest.raises(ValueError):
        k.command(str(tmp_path/"python.bat"),tmp_path/"submitted/plan.json",registered.pin,tmp_path/"capture",checked)


@pytest.fixture(scope="module")
def native_sized_callbacks(callbacks):
    """Stamp only synthetic fixtures, never historical images or labels."""
    report = deepcopy(callbacks["strict"][0])
    component = report["components"].pop("Petiole")
    for number in (41,42):
        report["components"][f"SubStem_{number}"] = dict(deepcopy(component),id=f"SubStem_{number}")
    result = {}
    for ordinal,disposition in enumerate(("strict","exclude"),1):
        cb = deepcopy(callbacks[disposition][1])
        meta = cb["metadata"]
        meta["sample_id"] = f"SubStem_41_view_{ordinal:03d}"
        meta["native_instance_backend"] = "fast"
        meta["render_budget"] = dict(k.budget_evidence("reference56",ordinal-1),
                                    actual_orchestrator_requests=7,render_seconds=1.25)
        meta["supervision"].update(target_id="seed101_full_cr_4000000/SubStem_41",split_group="seed101_full",
            source_target_id="seed101_full/SubStem_41",conservative_view_cap_group="seed101_full/SubStem_41")
        for row in cb["catalogue"]:
            row["variant_id"] = "seed101_full_cr_4000000"
            if row["component_id"] == "Petiole": row["component_id"] = "SubStem_41"
        args = [meta,report,cb["rgb"],cb["depth"],cb["valid"],cb["components"],cb["catalogue"]]
        seal(args)
        meta["synchronization"].update(render_budget_subframes=56)
        meta["synchronization"]["freshness"]["callback_sequence"] = 7*ordinal
        cb["label"],cb["trace"] = k.audit.v2.annotate_for_storage(*args,
            target_mask=cb["target_mask"].astype(np.uint8)*255)
        assert cb["label"]["eligible"] is (disposition=="strict")
        result[disposition] = cb
    return report,result


@pytest.fixture
def captured(registered,native_sized_callbacks,tmp_path,monkeypatch):
    checked = check(registered)
    report,callbacks = native_sized_callbacks
    monkeypatch.setattr(k.audit,"load_for_inspection",lambda *_:{"report":report})
    folder = tmp_path/"new_job"
    submitted,capture = folder/"submitted/plan.json",folder/"capture"
    submitted.parent.mkdir(parents=True)
    submitted.write_bytes(registered.path.read_bytes())
    capture.mkdir()
    docs = dict(**k.audit.configuration(k.audit.STRICT_ONLY,True),**k.ANNOTATION,
        annotation_policy=k.audit.v2.annotation_policy(),worker_module=k.audit.WORKER_MODULE,
        worker_implementation_bindings=k.worker.implementation_bindings(),
        compact_implementation_bindings=k.worker.implementation_bindings(),storage_qualification=checked["storage"],
        compact_native_storage=True,storage_backend=k.storage.STORAGE_BACKEND,native_instance_backend="fast",
        render_budget_profile="reference56",render_profile_experiment=False)
    request = dict(docs,plan_path=str(submitted),plan_sha256=registered.pin,
                   training_started=False,experimental_budget_override=False)
    request_pin = put(capture/"request.json",request)
    rows = []
    for case in registered.plan["target_cases"]:
        for spec in case["views"]:
            name = spec["candidate_id"]
            row = dict(candidate_id=name,target_id=case["target_id"],requested_spec=deepcopy(spec),training_approved=False)
            if name in [callbacks[d]["metadata"]["sample_id"] for d in ("strict","exclude")]:
                disposition = "strict" if name.endswith("001") else "exclude"
                cb = deepcopy(callbacks[disposition])
                saved = k.worker.persist_callback(capture,name,**cb,plan_sha256=registered.pin,
                    persistence=k.audit.STRICT_ONLY,qualification_witness=True)
                meta = cb["metadata"]
                row.update(state=k.audit.CALLBACK_STATE,**saved,
                    robot_snapshot=meta["robot_snapshot"],screen=meta["geometry_screen"],
                    render_budget=meta["render_budget"],source_checks_passed=True,
                    callback_ordinal=1 if disposition=="strict" else 2)
            else: row.update(state="rejected_pose",reason="synthetic_CPU_fixture_pre_render_rejection")
            put(capture/(name+"_decision.json"),row)
            rows.append(row)
    result = dict(docs,state=k.audit.CAPTURE_STATE,records=rows,counts=k.audit.callback_counts(rows),
        request_sha256=request_pin,plan_sha256=registered.pin,training_approved=False,source_cap_reset=False,
        source_assets_unchanged=True,experimental_short_profile=False,render_budget_subframes_per_view=56,
        total_requested_subframes_per_view=56,target_count=2)
    put(capture/"result.json",result)
    return SimpleNamespace(folder=folder,submitted=submitted,capture=capture,checked=checked,pin=registered.pin,
        result=result,request=request,registered=registered)


def run(f):
    return k.postexit_audit(f.submitted,f.pin,f.capture,f.folder,f.checked)


def test_real_compact_witness_to_all_frames_complete_byte_parity(captured,monkeypatch):
    f = captured
    calls = []
    original = k.worker.persist_callback
    def persist(*args,**kwargs):
        calls.append((args[1],kwargs["persistence"],kwargs["qualification_witness"]))
        return original(*args,**kwargs)
    monkeypatch.setattr(k.worker,"persist_callback",persist)
    output = run(f)
    proof = k.oc.read_json(output["qualification_path"])
    assert proof == output["persistence_qualification"]
    assert calls == [("SubStem_41_view_001",k.audit.ALL_FRAMES,False),
                     ("SubStem_41_view_002",k.audit.ALL_FRAMES,False)]
    assert proof["counts"]["validated_callbacks"] == 2
    assert proof["counts"]["orchestrator_requests"] == 14
    assert proof["counts"]["persisted_candidates"] == proof["counts"]["unpersisted_attempts"] == 1
    assert proof["counts"]["pre_render_rejections"] == 10
    assert proof["observed_dispositions"] == {"strict":1,"exclude":1}
    assert proof["native_hold_callbacks_observed"] == 0
    assert proof["cpu_all_frames_logical_bytes_equal"]
    for name in ("training_approved","native_persistence_qualified","source_cap_reset",
                 "native_all_frames_cli_executed","native_hold_branch_qualified",
                 "native_rejected_control_branch_qualified","short_budget_qualified"):
        assert proof[name] is False
    for name in ("training_diversity_increment","witness_new_native_observations",
                 "cpu_all_frames_new_native_observations","cpu_all_frames_native_launches"):
        assert proof[name] == 0
    assert proof["owned_exit_verification_required_by_caller"]
    assert not (f.folder/k.PARITY_DIRECTORY/"request.json").exists()
    assert not (f.folder/k.PARITY_DIRECTORY/"result.json").exists()
    for row in proof["cpu_all_frames_parity"]:
        witness,mirror = (k.SampleReader(row[key]) for key in ("witness","cpu_all_frames_sample"))
        assert set(row["logical_files"]) == set(witness.manifest["files"]) == set(mirror.manifest["files"])
        assert row["native_npy_files_compared"] == 3
        for logical,evidence in row["logical_files"].items():
            raw = witness.read(logical)
            assert raw == mirror.read(logical)
            assert evidence == dict(bytes=len(raw),sha256=k.digest(raw),exact=True)
        assert row["source_callback"]["request_sha256"] == output["request_sha256"]
        assert row["training_diversity_increment"] == row["new_native_observations"] == 0
    k.oc.bind_all(output["bindings"])
    with pytest.raises(ValueError): run(f)


@pytest.mark.parametrize("fault",[
    "mode","witness","policy","epoch","worker_map","backend","budget","experiment",
    "target_count","subframes","proof","plan","request_pin","result_state","missing_row",
    "sidecar","sample_steps","sample_robot","failure","source","submitted_layout","capture_layout",
])
def test_postexit_rejects_native_contract_tamper(captured,fault):
    f = captured
    result = f.result
    if fault=="mode": result["persistence_mode"]=k.audit.ALL_FRAMES
    elif fault=="witness": result["qualification_witness_save_all"]=False
    elif fault=="policy": result["persistence_policy_sha256"]="0"*64
    elif fault=="epoch": result["annotation_epoch"]="legacy"
    elif fault=="worker_map": result["worker_implementation_bindings"]={}
    elif fault=="backend": result["native_instance_backend"]="legacy"
    elif fault=="budget": result["render_budget_profile"]="warm56_then8_trial"
    elif fault=="experiment": result["experimental_short_profile"]=True
    elif fault=="target_count": result["target_count"]=1
    elif fault=="subframes": result["total_requested_subframes_per_view"]=8
    elif fault=="proof": result["storage_qualification"]={"unqualified":True}
    elif fault=="plan": result["plan_sha256"]="0"*64
    elif fault=="request_pin": result["request_sha256"]="0"*64
    elif fault=="result_state": result["state"]="pending"
    elif fault=="missing_row": result["records"].pop()
    elif fault=="sidecar": put(f.capture/(_names()[0]+"_decision.json"),{})
    elif fault=="sample_steps":
        result["records"][0]["render_budget"]["actual_orchestrator_requests"]=1
        put(f.capture/(_names()[0]+"_decision.json"),result["records"][0])
    elif fault=="sample_robot":
        # Rehashed result claim does not bypass native metadata parity.
        result["records"][0]["robot_snapshot"]={"changed":True}
        put(f.capture/(_names()[0]+"_decision.json"),result["records"][0])
    elif fault=="failure": put(f.capture/"failure.json",{"failed":True})
    elif fault=="source": put(Path(next(iter(f.registered.source))),{})
    elif fault=="submitted_layout": f.submitted=f.folder/"plan.json"
    elif fault=="capture_layout": f.folder=f.folder/"foreign"
    put(f.capture/"result.json",result)
    with pytest.raises((ValueError,KeyError,FileNotFoundError)): run(f)
    assert not (f.folder/k.QUALIFICATION_FILE).exists()
    assert not (f.folder/k.PARITY_DIRECTORY).exists()


@pytest.mark.parametrize("which",["primary","witness"])
def test_real_corrupt_native_compact_payload_rejected(captured,which):
    f = captured
    root = f.capture/_names()[0] if which=="primary" else f.capture/k.audit.WITNESS_DIRECTORY/_names()[0]
    reader = k.SampleReader(root)
    payload = root/reader.manifest["files"]["inputs/depth_m.npy"]["stored_path"]
    changed = bytearray(payload.read_bytes()); changed[-1] ^= 1
    payload.write_bytes(changed)  # Corrupt synthetic pytest artifact only.
    with pytest.raises(ValueError): run(f)
    assert not (f.folder/k.QUALIFICATION_FILE).exists()


def test_all_frames_cpu_corruption_prevents_qualification(captured,monkeypatch):
    original = k.worker.persist_callback
    def persist(*args,**kwargs):
        saved = original(*args,**kwargs)
        folder = Path(args[0])/args[1]
        reader = k.SampleReader(folder)
        path = folder/reader.manifest["files"]["inputs/depth_m.npy"]["stored_path"]
        changed = bytearray(path.read_bytes()); changed[-1] ^= 1
        path.write_bytes(changed)
        return saved
    monkeypatch.setattr(k.worker,"persist_callback",persist)
    with pytest.raises(ValueError): run(captured)
    assert (captured.folder/k.AUDIT_FILE).is_file()
    assert not (captured.folder/k.QUALIFICATION_FILE).exists()


def test_source_drift_during_cpu_parity_stops_publication(captured,monkeypatch):
    original = k._all_frames_parity
    def changed(*args,**kwargs):
        result = original(*args,**kwargs)
        put(Path(next(iter(captured.registered.source))),{"source_changed_after_replay":True})
        return result
    monkeypatch.setattr(k,"_all_frames_parity",changed)
    with pytest.raises(ValueError): run(captured)
    assert not (captured.folder/k.QUALIFICATION_FILE).exists()


@pytest.mark.parametrize("fault",["zero","all_strict","all_skip","missing_witness","requests","unreplayable","control"])
def test_coverage_cannot_be_manufactured(fault):
    # Pure counter contract, not native observation evidence.
    rows=[]
    for i,d in enumerate(("strict","exclude")):
        rows.append(dict(state=k.audit.CALLBACK_STATE,persistence=dict(role="strict_candidate" if i==0 else
            "unpersisted_rejection"),primary={} if i==0 else None,witness={},render_budget={"actual_orchestrator_requests":7}))
    rows.extend({"state":"rejected_pose"} for _ in range(10))
    if fault=="zero": rows=[{"state":"rejected_pose"} for _ in range(12)]
    elif fault=="all_strict": rows[1].update(primary={},persistence={"role":"strict_candidate"})
    elif fault=="all_skip": rows[0].update(primary=None,persistence={"role":"unpersisted_rejection"})
    elif fault=="missing_witness": rows[1]["witness"]=None
    elif fault=="requests": rows[0]["render_budget"]["actual_orchestrator_requests"]=1
    elif fault=="control": rows[1].update(primary={},persistence={"role":"rejected_control"})
    counts=k.audit.callback_counts(rows)
    review=dict(declared_worker_counts=counts,replay_verified_unique_callbacks=2,
        replay_verified_primary_samples=1,same_callback_byte_pairs=1,witness_records=[{},{}],
        unreplayable_attempts=1 if fault=="unreplayable" else 0,
        skipped_annotations_independently_verified=True,qualification_retained_and_skipped_branches_covered=True)
    with pytest.raises(ValueError): k._coverage(review,dict(records=rows,counts=counts))


def test_no_native_launch_or_runner_import_and_no_schema_forgery():
    tree = ast.parse(Path(k.__file__).read_text(encoding="utf-8"))
    imports = [ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert not any(any(token in text for token in ("serial_phases","isaacsim","omni","pxr","boundary_bindings"))
                   for text in imports)
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert "worker.persist_callback" in calls and "audit.audit_capture" in calls
    assert not any(call in calls for call in ("worker.collect","worker._collect","worker.main",
                                             "reference.postexit_audit","reference.audit.audit_capture"))
    assert not any(any(token in call for token in ("Popen","run_checked","SimulationApp","subprocess")) for call in calls)
