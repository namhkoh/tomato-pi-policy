"""Exact replay parity, real compact roundtrips and synthetic CPU audit tests."""
import ast
from pathlib import Path

import pytest

from ..depth_preview import sha256
from .test_compact_query_v2 import function_source, assert_scoped_ast_parity

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "audit.py"
FUTURE_AUDIT = HERE / "query_audit_v2.py"
BASELINE_SHA256 = "ba9f96539e4d35ccc0f215b5b7207561eb6df20fefe26d9998fe35135e3bd123"


def test_frozen_audit_baseline():
    assert sha256(BASELINE) == BASELINE_SHA256
    source = function_source(BASELINE, "audit_capture")
    assert_scoped_ast_parity(source, source)


@pytest.mark.parametrize("before,after", [
    ("verify_bindings(plan['source_bindings'])", "None"),
    ("verify_bindings(plan['implementation_bindings'])", "None"),
    ("verify_bindings(plan['prerequisite_bindings'])", "None"),
    ("require(seen==planned.keys()", "require(True"),
    ("reader.verify_all()", "None"),
    ("reader.array('inputs/depth_m.npy')", "invented_depth()"),
    ("check_native_evidence(", "skip_native_evidence("),
    ("label == reader.json('supervision/label.json')", "True"),
    ("trace == reader.json('supervision/query_trace.json')", "True"),
    ("trace == row.get('query_trace')", "True"),
    ("auto == row['automatic_annotation_eligible']", "True"),
    ("meta['supervision']['conservative_view_cap_group']==cap_group", "True"),
])
def test_parity_guard_detects_lost_audit_guards(before, after):
    source = function_source(BASELINE, "audit_capture")
    assert before in source
    with pytest.raises(AssertionError, match="Unlisted"):
        assert_scoped_ast_parity(source, source.replace(before, after, 1))


def test_native_hash_check_precedes_derivation_and_trace():
    tree = ast.parse(function_source(BASELINE, "audit_capture"))
    positions = {}
    for name in ("check_native_evidence", "derive", "trace_review"):
        matches = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == name]
        assert len(matches) == 1
        positions[name] = matches[0]
    assert list(positions.values()) == sorted(positions.values())


@pytest.mark.parametrize("eligible,trace,expected", [
    (False, None, "exclude_geometry_or_visibility"),
    (True, {"passed": False}, "hold_visual_clarity"),
    (True, {"passed": True}, "accept_strict_automatic_annotation_candidate"),
])
def test_frozen_decision_boundary_is_not_training_approval(eligible, trace, expected):
    """Execute only two boolean expressions, never derive/review native evidence."""
    tree = ast.parse(function_source(BASELINE, "audit_capture"))
    statements = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("auto", "decision"):
                statements[node.targets[0].id] = node
    assert set(statements) == {"auto", "decision"}
    code = compile(ast.Module(body=[statements["auto"], statements["decision"]], type_ignores=[]),
                   str(BASELINE), "exec")
    namespace = {"label": {"eligible": eligible}, "trace": trace}
    exec(code, namespace)
    assert namespace["decision"] == expected
    assert namespace["auto"] is (eligible and trace["passed"] if trace else False)


def test_v2_full_replay_parity_has_only_explicit_epoch_annotation_and_binding_edits():
    assert_scoped_ast_parity(function_source(BASELINE, "audit_capture"),
                            function_source(FUTURE_AUDIT, "audit_capture"),
                            reviewed_edits=[
    [
        "def audit_capture(capture, plan_path, *, storage_root=None):",
        "def audit_capture(capture, plan_path, *, result_sha256, storage_root=None):"
    ],
    [
        "    result = read_json(capture/'result.json'); plan = read_json(plan_path)",
        "    source_pins = {str(capture/'result.json'):result_sha256,\n                   str(capture/'request.json'):sha256(capture/'request.json'), str(plan_path):sha256(plan_path)}\n    require(isinstance(result_sha256, str) and len(result_sha256) == 64\n            and all(c in '0123456789abcdef' for c in result_sha256), 'Explicit result SHA256 required')\n    verify_bindings(source_pins)\n    result = read_json(capture/'result.json'); plan = read_json(plan_path)"
    ],
    [
        "    request = read_json(capture/'request.json')",
        "    request = read_json(capture/'request.json')\n    verify_epoch_documents(request, result)\n    require(result['request_sha256'] == source_pins[str(capture/'request.json')]\n            and result['plan_sha256'] == source_pins[str(plan_path)], 'Unbound query capture completion')\n    require('target_cases' in plan, 'Initial query-V2 audit requires a batch plan')"
    ],
    [
        "result.get('state')=='native_generated_multiview_pilot_complete_pending_review'",
        "result.get('state')==CAPTURE_STATE"
    ],
    [
        "        meta = reader.metadata",
        "        meta = reader.metadata\n        require(meta.get('schema_version') == SAMPLE_SCHEMA\n                and all(meta.get(k) == v for k,v in annotation_fields().items()), 'Mixed sample annotation epoch')"
    ],
    [
        "        label = derive(meta,generated['report'],rgb,depth,valid,components,catalogue)",
        "        label, trace = annotate_for_storage(meta,generated['report'],rgb,depth,valid,components,catalogue,\n                                             target_mask=target)"
    ],
    [
        "        trace = None\n        if label['eligible']:\n            trace = trace_review(meta,generated['report'],label,rgb,depth,valid,components,catalogue)\n            require(trace == reader.json('supervision/query_trace.json'), 'Replayed trace differs')",
        "        if label['eligible']:\n            require(trace == reader.json('supervision/query_trace.json'), 'Replayed trace differs')\n        else:\n            require(('supervision/query_trace.json' not in reader.manifest['files'] if reader.manifest\n                     else not (folder/'supervision/query_trace.json').exists()),\n                    'Excluded sample contains a trace')"
    ],
    [
        "review_method='replayed_anatomy_exact_native_buffers_and_trace',visual_review_performed=False,",
        "review_method='replayed_native_query_v2_full_selection_and_composite_trace',visual_review_performed=False,\n            **annotation_fields(),\n            annotation_input_metadata_sha256=label['annotation_input_metadata_sha256'],\n            query_selection_evidence_sha256=label['query_selection_evidence_sha256'],"
    ],
    [
        "    return dict(state='completed_automatic_annotation_replay',capture=str(capture),records=records,",
        "    verify_bindings(source_pins)\n    verify_bindings(plan['source_bindings'])\n    verify_epoch_documents(request, result)\n    return dict(state=AUDIT_STATE,capture=str(capture),records=records,\n        **annotation_fields(), annotation_policy=annotation_policy(),\n        worker_module=WORKER_MODULE, worker_implementation_bindings=implementation_bindings(),"
    ]
])


from copy import deepcopy
import json
import numpy as np
from ..capture_contract import fingerprint
from ..dataset_review import write_json, read_json
from . import query_audit_v2 as audit
from . import compact_query_v2 as worker
from .capture_storage import write_compact_native_sample
from .bundle import SampleReader, digest
from .test_query_selection_v2 import native_case, seal


def test_canonical_metadata_removes_only_files_and_is_private():
    original = dict(files={"stored": "binding"}, calibration={"resolution":[1696,816]},
                    synchronization={"freshness":{"rgb_sha256":"preserved"}},
                    elapsed_seconds=17.25, schema_version=audit.SAMPLE_SCHEMA,
                    **audit.annotation_fields())
    before = deepcopy(original)
    canonical = audit.canonical_annotation_metadata(original)
    assert canonical == {k:v for k,v in before.items() if k != "files"}
    canonical["calibration"]["resolution"][0] = 1
    assert original == before
    assert fingerprint(audit.canonical_annotation_metadata(before)) != fingerprint(before)
    for key in ("elapsed_seconds", "calibration", "synchronization", "annotation_epoch",
                "annotation_policy_sha256", "schema_version"):
        changed = deepcopy(before)
        changed.pop(key)
        assert fingerprint(audit.canonical_annotation_metadata(changed)) != fingerprint(
            audit.canonical_annotation_metadata(before))


@pytest.fixture(scope="module")
def annotated_cases():
    """Synthetic native-sized arrays only, NEVER captured/approved observations."""
    cases = {}
    for disposition in ("pass", "hold", "exclude"):
        args = native_case()
        meta, report, rgb, depth, valid, components, catalogue = args
        args[5] = components = components.astype(np.uint32)
        if disposition == "hold":
            components[408,950] = 2
        elif disposition == "exclude":
            components[:,878:909] = 0
        meta.update(schema_version=audit.SAMPLE_SCHEMA, **audit.annotation_fields(),
                    training_sample_approved=False, robot_snapshot={"fixture":True},
                    geometry_screen={"passed":True,"fixture":True},
                    render_budget={"profile":worker.REFERENCE})
        meta["supervision"]["conservative_view_cap_group"] = meta["supervision"]["target_id"]
        seal(args)
        label, trace = audit.annotate_for_storage(*args,
            target_mask=(components == 1).astype(np.uint8)*255)
        assert label["eligible"] is (disposition != "exclude")
        assert (trace is not None and trace["passed"]) is (disposition == "pass")
        callback = dict(metadata=meta, rgb=rgb, depth=depth, valid=valid,
            components=components, catalogue=catalogue, instances=components.copy(),
            mapping={"1":"/fixture/Petiole","2":"/fixture/Main"},
            organs=np.zeros_like(components,dtype=np.uint8),
            target_mask=components == 1, label=label, trace=trace)
        cases[disposition] = (args, callback)
    return cases


@pytest.mark.parametrize("disposition", ["pass", "hold", "exclude"])
def test_actual_compact_roundtrip_replays_entire_selection_evidence(tmp_path, annotated_cases, disposition):
    args, callback = annotated_cases[disposition]
    before = deepcopy(callback["metadata"])
    depth_bits = callback["depth"].view(np.uint32).copy()
    folder = tmp_path/"sample"
    stored = write_compact_native_sample(folder, **callback)
    reader = SampleReader(folder)
    assert reader.verify_all()
    assert "files" in reader.metadata and "files" not in before
    assert callback["metadata"] == before
    assert audit.canonical_annotation_metadata(reader.metadata) == before
    label, trace = audit.annotate_for_storage(reader.metadata, args[1],
        reader.image("inputs/rgb.png"), reader.array("inputs/depth_m.npy"),
        reader.image("inputs/depth_valid.png") != 0, reader.array("supervision/component_id.npy"),
        reader.json("supervision/identities.json")["component_catalogue"],
        target_mask=reader.image("supervision/target_visible.png"))
    assert label == callback["label"] == reader.json("supervision/label.json")
    assert trace == callback["trace"] == stored["query_trace"]
    evidence = label["query_selection_evidence"]
    assert evidence["input_fingerprints"]["metadata"] == fingerprint(before)
    assert label["annotation_input_metadata_sha256"] == fingerprint(before)
    assert label["query_selection_evidence_sha256"] == fingerprint(evidence)
    # This inequality reproduces the original bug if canonicalization is removed.
    assert fingerprint(reader.metadata) != label["annotation_input_metadata_sha256"]
    assert np.array_equal(reader.array("inputs/depth_m.npy").view(np.uint32), depth_bits)
    assert np.array_equal(callback["depth"].view(np.uint32), depth_bits)
    assert np.array_equal(reader.image("inputs/rgb.png"), callback["rgb"])
    assert set(reader.manifest["files"]) <= set(reader.metadata["files"]) | {
        "sample.json","supervision/label.json","supervision/query_trace.json"}
    assert ("supervision/query_trace.json" in reader.manifest["files"]) is (trace is not None)
    if trace:
        assert trace == reader.json("supervision/query_trace.json")
        assert trace["passed"] == (trace["legacy_trace"]["passed"] and trace["fixed_grid"]["passed"])
    assert not label["training_approved"] and not label["native_depth_reconstructed"]


def epoch_document():
    bindings = audit.implementation_bindings()
    return dict(**audit.annotation_fields(), annotation_policy=audit.annotation_policy(),
                worker_module=audit.WORKER_MODULE, worker_implementation_bindings=bindings,
                compact_implementation_bindings=bindings, storage_qualification={"fixture_only":True})


@pytest.mark.parametrize("field", ["annotation_epoch", "annotation_policy_sha256", "annotation_policy",
                                  "worker_module", "worker_implementation_bindings",
                                  "compact_implementation_bindings", "storage_qualification"])
@pytest.mark.parametrize("which", [0,1])
def test_epoch_or_binding_tamper_is_not_accepted(monkeypatch, field, which):
    documents = [epoch_document(), epoch_document()]
    documents[which][field] = {"tampered":True} if isinstance(documents[which][field],dict) else "wrong"
    monkeypatch.setattr(audit.compact_qualification, "verify_checked_qualification",
                        lambda _: pytest.fail("Mismatched receipt reached proof verifier"))
    with pytest.raises(ValueError):
        audit.verify_epoch_documents(*documents)


def test_epoch_policy_is_copied_and_false_scope_claims_preserved():
    policy = audit.annotation_policy()
    assert fingerprint(policy) == audit.POLICY_SHA256
    assert policy["metadata_fingerprint_omits_only"] == ["files"]
    assert policy["candidate_count"] == 41 and policy["anchor_mm"] == 8 and policy["step_mm"] == 5
    assert policy["legacy_trace_required"] is True
    assert policy["source_cap_reset"] is False and policy["training_approved"] is False
    policy["metadata_fingerprint_omits_only"].append("synchronization")
    assert audit.annotation_policy()["metadata_fingerprint_omits_only"] == ["files"]


def test_wrong_metadata_epoch_rejected_before_selector(monkeypatch, annotated_cases):
    args = list(annotated_cases["pass"][0])
    args[0] = deepcopy(args[0])
    args[0]["annotation_epoch"] = "legacy"
    monkeypatch.setattr(audit, "annotate_v2", lambda *a, **kw: pytest.fail("Selector should not run"))
    with pytest.raises(ValueError, match="epoch/policy"):
        audit.annotate_for_storage(*args, target_mask=(args[5] == 1).astype(np.uint8)*255)


def test_annotation_input_epoch_and_no_file_exceptions_hidden_in_worker():
    source = function_source(Path(worker.__file__), "_collect")
    assert "elapsed_seconds=" in source
    tree = ast.parse(source)
    metadata = next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id == "metadata" for t in n.targets))
    assert "elapsed_seconds" not in {k.arg for k in metadata.value.keywords}
    assert "files" not in {k.arg for k in metadata.value.keywords}
    annotation = source.index("label,trace=annotate_for_storage(")
    assert source.index("validate_native_static(") < annotation < source.index("write_compact_native_sample(")
    replay = function_source(FUTURE_AUDIT, "audit_capture")
    assert replay.index("reader.verify_all()") < replay.index("check_native_evidence(") < replay.index("annotate_for_storage(")


@pytest.fixture
def audit_case(tmp_path, monkeypatch, annotated_cases):
    """Exercise real audit+storage+selector, stub only source-report/proof loading."""
    report = annotated_cases["pass"][0][1]
    def load(directory, collection):
        assert directory == "synthetic_variant" and collection == "synthetic_collection"
        return {"report":report}
    monkeypatch.setattr(audit, "load_for_inspection", load)
    monkeypatch.setattr(audit.compact_qualification, "verify_checked_qualification", lambda _: True)

    def make(disposition="pass", *, callback_edit=None):
        args, template = annotated_cases[disposition]
        callback = deepcopy(template)
        if callback_edit is not None:
            callback_edit(callback)
        capture = tmp_path/("capture_"+disposition)
        capture.mkdir()
        name = callback["metadata"]["sample_id"]
        folder = capture/name
        stored = write_compact_native_sample(folder, **callback)
        anchor_path = tmp_path/("anchor_"+disposition+".json")
        write_json(anchor_path, dict(variant_directory="synthetic_variant",
                                    source_collection_plan="synthetic_collection"))
        binding = {str(anchor_path):sha256(anchor_path)}
        spec = {"candidate_id":name, "fixture_only":True}
        target = callback["metadata"]["supervision"]["target_id"]
        plan = dict(split="train", resolution=[1696,816], source_family="fixture",
            training_approved=False, source_cap_reset=False, physical_motion_commanded=False,
            hidden_cut_coordinates_executable=False, source_bindings=binding,
            implementation_bindings={str(BASELINE):sha256(BASELINE)}, prerequisite_bindings=binding,
            anchor_pair_plan=str(anchor_path), target_cases=[
                dict(target_id=target, conservative_view_cap_group=target, views=[spec])])
        plan_path = tmp_path/("plan_"+disposition+".json")
        write_json(plan_path, plan)
        request = dict(epoch_document(), plan_path=str(plan_path), plan_sha256=sha256(plan_path),
                       training_started=False)
        write_json(capture/"request.json", request)
        label, trace = callback["label"], callback["trace"]
        row = dict(candidate_id=name, target_id=target, requested_spec=spec,
            robot_snapshot=callback["metadata"]["robot_snapshot"], screen=callback["metadata"]["geometry_screen"],
            state="native_captured_pending_review", **stored,
            eligible_annotation=label["eligible"], label_reason=label["reason"],
            automatic_annotation_eligible=bool(trace is not None and trace["passed"]))
        result = dict(epoch_document(), state=audit.CAPTURE_STATE, training_approved=False,
            source_assets_unchanged=True, records=[row], captured_frames=1,
            automatically_clear_annotation_candidates=int(row["automatic_annotation_eligible"]),
            request_sha256=sha256(capture/"request.json"), plan_sha256=sha256(plan_path))
        write_json(capture/"result.json", result)
        return dict(capture=capture, plan_path=plan_path, result_sha256=sha256(capture/"result.json"),
                    folder=folder, result=result, callback=callback)
    return make


def run_audit(case):
    return audit.audit_capture(case["capture"], case["plan_path"], result_sha256=case["result_sha256"])


@pytest.mark.parametrize("disposition,expected", [
    ("pass", "accept_strict_automatic_annotation_candidate"),
    ("hold", "hold_visual_clarity"),
    ("exclude", "exclude_geometry_or_visibility"),
])
def test_complete_postexit_replay_uses_actual_writer_reader_and_selector(audit_case, disposition, expected):
    case = audit_case(disposition)
    review = run_audit(case)
    assert review["state"] == audit.AUDIT_STATE
    assert review["counts"] == {expected:1}
    assert review["result_sha256"] == case["result_sha256"]
    assert review["plan_sha256"] == sha256(case["plan_path"])
    record, = review["records"]
    assert record["decision"] == expected and record["label_replayed_exact"]
    assert record["trace_replayed_exact"] is (disposition != "exclude")
    assert record["native_callback_hashes_verified"] and record["source_and_file_hashes_verified"]
    assert record["annotation_input_metadata_sha256"] == case["callback"]["label"]["annotation_input_metadata_sha256"]
    for key in ("training_approved","source_cap_reset","physical_execution_approved","visual_review_performed"):
        assert record[key] is False
    assert review["original_reviews_modified"] is False


@pytest.mark.parametrize("fault", [
    "external_result_pin", "request_pin", "plan_pin", "source_pin",
    "worker_map", "policy_hash", "old_state", "missing_decision", "duplicate_decision",
    "unplanned_decision", "failure_marker",
])
def test_full_audit_rejects_binding_state_and_plan_tamper_before_annotation(audit_case, monkeypatch, fault):
    case = audit_case()
    result = case["result"]
    if fault == "external_result_pin":
        case["result_sha256"] = "0"*64
    elif fault == "request_pin":
        result["request_sha256"] = "0"*64
    elif fault == "plan_pin":
        result["plan_sha256"] = "0"*64
    elif fault == "source_pin":
        plan = read_json(case["plan_path"])
        Path(plan["anchor_pair_plan"]).write_text("{}", encoding="utf-8")
    elif fault == "worker_map":
        result["worker_implementation_bindings"] = {}
    elif fault == "policy_hash":
        result["annotation_policy_sha256"] = "0"*64
    elif fault == "old_state":
        result["state"] = "native_generated_multiview_pilot_complete_pending_review"
    elif fault == "missing_decision":
        result["records"] = []
    elif fault == "duplicate_decision":
        result["records"] *= 2
    elif fault == "unplanned_decision":
        result["records"][0]["candidate_id"] = "unplanned"
    elif fault == "failure_marker":
        write_json(case["capture"]/"failure.json", {"fixture_only":True})
    if fault != "external_result_pin":
        (case["capture"]/"result.json").write_text(json.dumps(result), encoding="utf-8")
        case["result_sha256"] = sha256(case["capture"]/"result.json")
    monkeypatch.setattr(audit, "annotate_for_storage", lambda *a, **kw: pytest.fail("Should reject before annotation"))
    with pytest.raises(ValueError):
        run_audit(case)


def test_corrupt_storage_is_rejected_before_annotation_even_though_files_is_canonicalized(audit_case, monkeypatch):
    case = audit_case()
    reader = SampleReader(case["folder"])
    entry = reader.manifest["files"]["inputs/rgb.png"]
    path = case["folder"]/entry["stored_path"]
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1]^1]))
    monkeypatch.setattr(audit, "annotate_for_storage", lambda *a, **kw: pytest.fail("Corrupt RGB reached annotation"))
    with pytest.raises(ValueError, match="Corrupt"):
        run_audit(case)


@pytest.mark.parametrize("fault", ["evidence", "input_fingerprint", "query"])
def test_rehashed_label_tamper_still_fails_exact_replay(audit_case, fault):
    def edit(callback):
        label = callback["label"]
        if fault == "evidence":
            label["query_selection_evidence"]["context"]["fixed_cache_hits"] += 1
            label["query_selection_evidence_sha256"] = fingerprint(label["query_selection_evidence"])
        elif fault == "input_fingerprint":
            label["annotation_input_metadata_sha256"] = "0"*64
        else:
            label["query_pixel_uv"][0] += 1
    case = audit_case(callback_edit=edit)
    # Real writer produces mutually consistent storage/row hashes of the tamper;
    # only recomputing the complete annotation/evidence can reject it.
    with pytest.raises(ValueError, match="Replayed label differs"):
        run_audit(case)


def test_rehashed_nonstorage_metadata_change_is_not_normalized_away(audit_case):
    def edit(callback):
        callback["metadata"]["elapsed_seconds"] = 123.456  # NOT a permitted post-annotation field
    case = audit_case(callback_edit=edit)
    with pytest.raises(ValueError, match="Replayed label differs"):
        run_audit(case)


def test_excluded_sample_cannot_smuggle_a_trace(audit_case):
    def edit(callback):
        callback["trace"] = {"passed":False,"fixture_only":True}
    case = audit_case("exclude", callback_edit=edit)
    with pytest.raises(ValueError, match="Excluded sample contains a trace"):
        run_audit(case)


def test_source_change_during_replay_fails_end_check(audit_case, monkeypatch):
    case = audit_case()
    original = audit.annotate_for_storage
    def annotate(*args, **kwargs):
        answer = original(*args, **kwargs)
        plan = read_json(case["plan_path"])
        Path(plan["anchor_pair_plan"]).write_text("{}", encoding="utf-8")
        return answer
    monkeypatch.setattr(audit, "annotate_for_storage", annotate)
    with pytest.raises(ValueError, match="Stale audit/review"):
        run_audit(case)
