"""One exact seed101 V3 strict+witness job, with post-exit CPU byte parity.

Registration adapter only: no CLI, process launch, renderer or runner imports.
The owning serial runner MUST call postexit_audit only after its own child exits
with integer code zero, pin the launch event, and verify the capture tree before
and after this call. This helper does not attest OS exit.

JOB_FIELDS extends the existing query fields by persistence_policy_sha256.
Backend=fast, budget=reference56, strict_candidates_only and witness=True are
fixed, not caller knobs. At most one such job may be registered. The exact
source plan may also appear once as a query_v2_qualification.v1 job; no other
duplicate/alias allowance is implied.

All primary/witness/CPU copies are qualification controls, zero new TRAIN
diversity. CPU copies exercise all_frames storage on the SAME saved callbacks;
they are not another native all_frames execution or additional observations.
No label, codec, camera, scene, policy or frozen worker modifications.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path

from ..native_original_capture import contracts as oc, serial_queue as legacy
from ..native_budget import evidence as budget_evidence
from . import serial_query_qualification as reference
from . import compact_query_v3 as worker, query_audit_v3 as audit
from . import compact_qualification as storage
from .bundle import ARRAYS, SampleReader, digest

KIND = "query_v3_persistence_qualification.v1"
JOB_FIELDS = (*reference.JOB_FIELDS, "persistence_policy_sha256")
ANNOTATION = deepcopy(reference.ANNOTATION)
PERSISTENCE_POLICY_SHA256 = "afc7075be2d56759a3892ff14966419008086b672d6d68bea58ab8d4a253feb2"
SOURCE_PLAN = (Path(__file__).resolve().parents[4] /
    "data/sim_data/collection_batches/native_reference_scale_20260916_v4/"
    "job_0001_seed101_full_round1/plan.json").resolve()
PLAN_SHA256 = "b1f448a6afd5d4af5ed6f1e203f7842c29a0ab962587fa4da2503a5e480ee7dd"
FROZEN_CODE = {
    "compact_query_v3.py":"1b62d22d40f26511225703e3df39b26e69a05cd0fa773e4e8d11e586e2dc1477",
    "query_audit_v3.py":"72a86add9ea0e0c6ea4b87391a8eba894c699e24e1dee4c6f574c548ba204b93",
    "serial_query_qualification.py":"3b8d1150f7b9c8c0c4fe4b5b8abf4a973727690e9a9fb02e88e7d21c5453c04a",
}
SCHEMA = "greenhouse.native_persistence_qualification.v1"
AUDIT_FILE = "query_v3_postexit_audit.json"
PARITY_DIRECTORY = "all_frames_cpu_parity"
QUALIFICATION_FILE = "persistence_qualification.json"
FLAGS = dict(training_approved=False, training_diversity_increment=0, source_cap_reset=False,
    visual_approval=False, native_persistence_qualified=False, independent_execution_attested=False)
_LOADED = oc.merge_bindings(worker.implementation_bindings(), reference.implementation_bindings(),
                           {str(Path(__file__).resolve()):oc.sha256(__file__)})


def implementation_bindings():
    oc.bind_all(_LOADED)
    oc.require(worker.implementation_bindings() == audit.implementation_bindings(),
               "Worker/auditor binding mismatch")
    oc.require(audit.PERSISTENCE_POLICY_SHA256 == PERSISTENCE_POLICY_SHA256
               and audit.v2.annotation_fields() == ANNOTATION, "Unregistered frozen policy")
    for name,pin in FROZEN_CODE.items():
        oc.require(_LOADED[str(Path(__file__).resolve().parent/name)] == pin,
                   "Unregistered frozen persistence implementation")
    reference.implementation_bindings()
    return dict(_LOADED)


def _fields(qualification):
    legacy._keys(qualification, JOB_FIELDS, "persistence qualification fields")
    oc.require(all(qualification[k] == value for k,value in ANNOTATION.items())
        and qualification["persistence_policy_sha256"] == PERSISTENCE_POLICY_SHA256,
        "Explicit frozen annotation/persistence policies required")


def _fixed_source(path, pin):
    path = legacy._path(str(path))
    oc.require(path == SOURCE_PLAN and pin == PLAN_SHA256, "Only exact seed101 job1 source plan is allowed")
    return oc.pin(path, pin)


def _shape(plan):
    oc.require(plan["source_family"] == "seed101_full" and plan["split"] == "train"
        and oc.FROZEN_SPLITS.get(plan["source_family"]) == "train"
        and plan["resolution"] == [1696,816]
        and type(plan["maximum_native_frames"]) is int and plan["maximum_native_frames"] == 12,
        "Exact original TRAIN donor / 12 native proposals required")
    expected = [(f"seed101_full_cr_4000000/SubStem_{n}",f"seed101_full/SubStem_{n}",
                 [f"SubStem_{n}_view_{i:03d}" for i in range(1,7)]) for n in (41,42)]
    actual = [(c["target_id"],c["conservative_view_cap_group"],[s["candidate_id"] for s in c["views"]])
              for c in plan["target_cases"]]
    oc.require(actual == expected, "Changed target ancestry/order or candidate identifiers")
    audit.validate_plan_mode(plan,audit.STRICT_ONLY,True)
    oc.require(all(not audit.retention(PLAN_SHA256,"seed101_full",d,name,audit.STRICT_ONLY)[
        "rejected_control_selected"] for _,_,names in expected for name in names for d in ("hold","exclude")),
        "Registered plan unexpectedly covers rejected-control retention")


def validate_plan(path, pin, qualification):
    _fields(qualification)
    code = implementation_bindings()
    path = _fixed_source(path,pin)
    _shape(oc.read_json(path))
    checked = reference.validate_plan(path,pin,{k:deepcopy(qualification[k]) for k in reference.JOB_FIELDS})
    oc.require(checked["count"] == 12, "Wrong bounded proposal count")
    checked.update(bindings=oc.merge_bindings(checked["bindings"],code),
        qualification=deepcopy(qualification),source_plan_path=str(path))
    return checked


def _recheck(checked, pin):
    oc.require(isinstance(checked,dict), "Validated persistence job required")
    _fields(checked["qualification"])
    code = implementation_bindings()
    path = _fixed_source(checked["source_plan_path"],pin)
    plan = oc.read_json(path)
    _shape(plan)
    oc.require(checked["plan"] == plan and checked["count"] == 12, "Checked plan changed")
    oc.bind_all(checked["bindings"])
    for key in ("source_bindings","implementation_bindings","prerequisite_bindings"):
        oc.bind_all(plan[key])
    matched = reference._matched(path,pin,checked["qualification"]["matched_short"],plan)
    oc.require(matched == checked["matched_short"], "Checked historical binding changed")
    proof = checked["storage"]
    oc.require(proof["path"] == checked["qualification"]["storage_qualification"]
        and proof["sha256"] == checked["qualification"]["storage_qualification_sha256"],
        "Checked storage proof changed")
    storage.verify_checked_qualification(proof)
    return code


def command(isaac, submitted, pin, capture, checked):
    _recheck(checked,pin)
    q = checked["qualification"]
    return [str(legacy._path(isaac)),"-m",audit.WORKER_MODULE,
        "--batch-plan",str(submitted),"--plan-sha256",pin,
        "--annotation-policy-sha256",q["annotation_policy_sha256"],
        "--persistence-policy-sha256",q["persistence_policy_sha256"],
        "--persistence",audit.STRICT_ONLY,"--qualification-witness-save-all",
        "--storage-qualification",q["storage_qualification"],
        "--storage-qualification-sha256",q["storage_qualification_sha256"],
        "--instance-backend","fast","--render-budget","reference56","--output",str(capture)]


def _bundle_bindings(reader):
    files = {str(reader.root/"bundle.json"):oc.sha256(reader.root/"bundle.json")}
    for entry in reader.manifest["files"].values():
        path = oc.safe_file(reader.root,entry["stored_path"])
        files[str(path)] = oc.sha256(path)
    return files


def _all_frames_parity(capture, destination, result, pin, request_pin):
    """Called ONLY after V3 replay; no native imports or alternative annotations."""
    destination.mkdir(parents=True,exist_ok=False)
    rows, bindings = [], {}
    for row in result["records"]:
        if row["state"] != audit.CALLBACK_STATE:
            continue
        name = row["candidate_id"]
        _,witness_path = audit.sample_directories(capture,name)
        witness = audit._reader(witness_path,row["witness"])
        metadata = audit.v2.canonical_annotation_metadata(witness.metadata)  # Removes ONLY files.
        ids = witness.json("supervision/identities.json")
        label = witness.json("supervision/label.json")
        trace = witness.json("supervision/query_trace.json") if label["eligible"] else None
        saved = worker.persist_callback(destination,name,
            rgb=witness.image("inputs/rgb.png"),depth=witness.array("inputs/depth_m.npy"),
            valid=witness.image("inputs/depth_valid.png") != 0, metadata=metadata,
            instances=witness.array("supervision/renderer_instance_id.npy"),
            mapping=ids["renderer_id_to_prim"],catalogue=ids["component_catalogue"],
            components=witness.array("supervision/component_id.npy"),organs=witness.image("supervision/organ_type.png"),
            target_mask=witness.image("supervision/target_visible.png") != 0,label=label,trace=trace,
            plan_sha256=pin,persistence=audit.ALL_FRAMES,qualification_witness=False)
        oc.require(saved["annotation"] == row["annotation"] and saved["primary"] == row["witness"]
            and saved["witness"] is None and saved["persistence"] == audit.retention(
                pin,metadata["supervision"]["split_group"],row["annotation"]["disposition"],name,audit.ALL_FRAMES),
            "CPU all_frames handoff changed annotation/storage/persistence")
        mirror = audit._reader(destination/name,saved["primary"])
        oc.require(set(mirror.manifest["files"]) == set(witness.manifest["files"]),
                   "CPU all_frames logical membership differs")
        evidence = {}
        for logical in witness.manifest["files"]:
            source_bytes,copy_bytes = witness.read(logical),mirror.read(logical)
            oc.require(source_bytes == copy_bytes, "CPU all_frames changed complete logical bytes: "+logical)
            evidence[logical] = dict(bytes=len(source_bytes),sha256=digest(source_bytes),exact=True)
        oc.require(ARRAYS <= evidence.keys(), "Missing complete native NPY files")
        bindings = oc.merge_bindings(bindings,_bundle_bindings(witness),_bundle_bindings(mirror))
        rows.append(dict(candidate_id=name,source_callback=dict(request_sha256=request_pin,
            callback_ordinal=row["callback_ordinal"],freshness=deepcopy(row["annotation"]["freshness"])),
            witness=str(witness_path),cpu_all_frames_sample=str(destination/name),
            storage_bindings=deepcopy(saved["primary"]),logical_files=evidence,
            native_npy_files_compared=len(ARRAYS),new_native_observations=0,training_diversity_increment=0))
    return rows,bindings


def _coverage(review,result):
    counts = audit.callback_counts(result["records"])
    n,k = counts["validated_callbacks"],counts["persisted_candidates"]
    oc.require(review["declared_worker_counts"] == result["counts"] == counts
        and counts["planned_decisions"] == 12 and 1 <= k < n <= 12
        and counts["persisted_samples"] == k and counts["unpersisted_attempts"] == n-k
        and counts["persisted_rejected_controls"] == counts["persisted_other_rejections"] == 0
        and counts["witness_samples"] == n and counts["orchestrator_requests"] == 7*n,
        "Insufficient strict+skip coverage or unexpected persistence/render accounting")
    oc.require(review["replay_verified_unique_callbacks"] == n
        and review["replay_verified_primary_samples"] == review["same_callback_byte_pairs"] == k
        and len(review["witness_records"]) == n and review["unreplayable_attempts"] == 0
        and review["skipped_annotations_independently_verified"] is True
        and review["qualification_retained_and_skipped_branches_covered"] is True,
        "Incomplete same-callback primary/witness replay")
    return counts


def postexit_audit(submitted, pin, capture, folder, checked):
    """Caller owns/attests child exit; this function never infers it from files."""
    code = _recheck(checked,pin)
    submitted, capture, folder = (Path(p).resolve() for p in (submitted,capture,folder))
    oc.require(submitted == folder/"submitted/plan.json" and capture == folder/"capture",
               "Exact submitted/capture sibling layout required")
    oc.pin(submitted,pin)
    protected = [capture,submitted.parent,*checked["roots"]]
    audit_path = oc.new_destination(folder/AUDIT_FILE,protected)
    parity_root = oc.new_destination(folder/PARITY_DIRECTORY,protected)
    proof_path = oc.new_destination(folder/QUALIFICATION_FILE,protected)
    reference._no_failure(capture)
    request_pin,result_pin = (oc.sha256(capture/name) for name in ("request.json","result.json"))
    input_pins = {str(capture/"request.json"):request_pin,str(capture/"result.json"):result_pin,str(submitted):pin}
    request,result = (oc.read_json(capture/name) for name in ("request.json","result.json"))
    audit.verify_epoch_documents(request,result)
    oc.require(request["plan_path"] == str(submitted) and request["plan_sha256"] == result["plan_sha256"] == pin
        and result["request_sha256"] == request_pin, "Completion differs from owned submitted request")
    for doc in (request,result):
        oc.require(doc["persistence_mode"] == audit.STRICT_ONLY and doc["qualification_witness_save_all"] is True
            and doc["persistence_policy_sha256"] == PERSISTENCE_POLICY_SHA256,
            "Only fixed strict+witness persistence is registered")
        oc.require(doc["storage_qualification"] == checked["storage"]
            and doc["compact_native_storage"] is True and doc["storage_backend"] == storage.STORAGE_BACKEND,
            "Wrong exact storage proof/backend")
        oc.require(doc["native_instance_backend"] == "fast" and doc["render_budget_profile"] == "reference56"
            and doc["render_profile_experiment"] is False, "Only fast/reference56 is registered")
    oc.require(request["experimental_budget_override"] is False and result["experimental_short_profile"] is False
        and result["render_budget_subframes_per_view"] == result["total_requested_subframes_per_view"] == 56
        and result["target_count"] == 2, "Changed target/render scope")
    review = audit.audit_capture(capture,submitted,result_sha256=result_pin)
    oc.require(review["state"] == audit.AUDIT_STATE and review["result_sha256"] == result_pin
        and review["request_sha256"] == request_pin and review["plan_sha256"] == pin
        and review["capture"] == str(capture), "Wrong postexit V3 replay")
    # Preserve a truthful unqualified audit even if branch coverage is inadequate.
    oc.write_new(audit_path,review)
    counts = _coverage(review,result)
    for row in result["records"]:
        if row["state"] != audit.CALLBACK_STATE:
            continue
        _,path = audit.sample_directories(capture,row["candidate_id"])
        reader = audit._reader(path,row["witness"])
        meta,expected = reader.metadata,budget_evidence("reference56",0)
        oc.require(meta["native_instance_backend"] == "fast"
            and meta["render_budget"] == row["render_budget"]
            and all(meta["render_budget"].get(k) == value for k,value in expected.items())
            and meta["render_budget"]["actual_orchestrator_requests"] == 7
            and meta["synchronization"]["render_budget_subframes"] == 56,
            "Stored callback differs from fixed native budget/backend")
    parity_rows,copy_pins = _all_frames_parity(capture,parity_root,result,pin,request_pin)
    oc.require(len(parity_rows) == counts["validated_callbacks"], "Incomplete CPU all_frames parity")
    oc.bind_all(input_pins); oc.bind_all(copy_pins)
    _recheck(checked,pin); reference._no_failure(capture)
    oc.bind_all(code)
    dispositions = Counter(r["annotation"]["disposition"] for r in result["records"] if r["state"] == audit.CALLBACK_STATE)
    receipt = dict(schema=SCHEMA,state="same_callback_persistence_checks_complete_pending_review",
        capture=str(capture),source_plan_path=checked["source_plan_path"],submitted_plan_path=str(submitted),
        plan_sha256=pin,request_sha256=request_pin,result_sha256=result_pin,
        audit_path=str(audit_path),audit_sha256=oc.sha256(audit_path),worker_kind=KIND,
        qualification_implementation_bindings=code,worker_implementation_bindings=worker.implementation_bindings(),
        **ANNOTATION,persistence_policy_sha256=PERSISTENCE_POLICY_SHA256,
        backend="fast",render_budget="reference56",persistence_mode=audit.STRICT_ONLY,
        native_witness_save_all=True,counts=counts,observed_dispositions=dict(dispositions),
        cpu_all_frames_parity=parity_rows,cpu_all_frames_logical_bytes_equal=True,
        cpu_all_frames_native_launches=0,cpu_all_frames_new_native_observations=0,
        witness_new_native_observations=0,native_all_frames_cli_executed=False,
        native_hold_branch_qualified=False,native_rejected_control_branch_qualified=False,short_budget_qualified=False,
        native_hold_callbacks_observed=dispositions["hold"],
        untested_qualification_scopes=["dedicated_native_hold_case","native_rejected_control_retention",
                                      "short_render_profile","native_all_frames_cli"],
        observation_role="persistence_qualification_control_not_new_diversity",
        owned_exit_verification_required_by_caller=True,source_assets_unchanged=True,**FLAGS)
    oc.write_new(proof_path,receipt)
    pins = oc.merge_bindings(input_pins,copy_pins,
        {str(audit_path):oc.sha256(audit_path),str(proof_path):oc.sha256(proof_path)})
    oc.bind_all(pins)
    return dict(request_sha256=request_pin,result_sha256=result_pin,audit_path=str(audit_path),
        audit_sha256=oc.sha256(audit_path),qualification_path=str(proof_path),qualification_sha256=oc.sha256(proof_path),
        counts=counts,persistence_qualification=receipt,bindings=pins,**ANNOTATION,
        persistence_policy_sha256=PERSISTENCE_POLICY_SHA256,**FLAGS)
