"""Exact frozen-flow parity and CPU-only query-V2 admission tests. No native launch."""
import ast
from pathlib import Path
import textwrap

import numpy as np
import pytest

from ..depth_preview import sha256
from ..native_budget import REFERENCE, TRIAL, requested_subframes
from . import compact_views as baseline
from . import capture_storage
from .bundle import SampleReader
from .test_capture_storage import synthetic_callback, npy_bytes

HERE = Path(__file__).resolve().parent
BASELINE = HERE / "compact_views.py"
FUTURE_WORKER = HERE / "compact_query_v2.py"
BASELINE_SHA256 = "2485d8c5b29479f9ffe8b2c6bf21acf8bc7dbc05956cd8f4b38909c473205b53"
STORAGE_SHA256 = "f662fe79af908cf550db327909e41200ca82466cbd20b755e8ff890cddcb8e45"


def function_source(path, name):
    raw = Path(path).read_text(encoding="utf-8")
    function = next(n for n in ast.parse(raw).body
                    if isinstance(n, ast.FunctionDef) and n.name == name)
    return ast.get_source_segment(raw, function)


def assert_scoped_ast_parity(reference, candidate, *, reviewed_edits=()):
    """Apply only individually reviewed, unique textual edits, then compare ASTs.

    No generic removal of imports, assignments, guards or metadata is allowed.
    The production V2 edit list is deliberately not guessed at scaffold time.
    """
    expected = textwrap.dedent(reference)
    for before, after in reviewed_edits:
        assert expected.count(before) == 1, "Reviewed edit must match exactly once"
        expected = expected.replace(before, after, 1)
    dump = lambda source: ast.dump(ast.parse(textwrap.dedent(source)), include_attributes=False)
    assert dump(candidate) == dump(expected), "Unlisted capture/audit implementation change"


def test_frozen_capture_and_storage_baselines():
    assert sha256(BASELINE) == BASELINE_SHA256
    assert sha256(Path(capture_storage.__file__)) == STORAGE_SHA256
    source = function_source(BASELINE, "_collect")
    assert_scoped_ast_parity(source, source)


@pytest.mark.parametrize("before,after", [
    ("context=prepare_native_scene(app,base)", "context=unverified_scene(app,base)"),
    ("capture_jobs(plan,anchor)", "capture_jobs(plan,anchor)[:1]"),
    ("include_generated_plants=True", "include_generated_plants=False"),
    ("pose=set_reference_snapshot(", "pose=unmounted_camera("),
    ("subframes=8", "subframes=1"),
    ("range(6)", "range(0)"),
    ("previous=freshness", "previous=None"),
    ("validate_native_static(", "unvalidated_payload("),
    ("decode_native_instances(", "invented_instances("),
    ("source_hashes(stage)==stage_hashes", "True"),
    ("stored=write_compact_native_sample(", "stored=lossy_writer("),
    ("writer.detach()", "None"),
])
def test_parity_guard_detects_unlisted_capture_changes(before, after):
    source = function_source(BASELINE, "_collect")
    assert before in source
    altered = source.replace(before, after, 1)
    with pytest.raises(AssertionError, match="Unlisted"):
        assert_scoped_ast_parity(source, altered)


def test_reviewed_edits_do_not_hide_other_changes():
    source = "def fixture():\n    value=1\n    return value\n"
    permitted = source.replace("value=1", "value=2")
    assert_scoped_ast_parity(source, permitted, reviewed_edits=[("value=1", "value=2")])
    with pytest.raises(AssertionError, match="Unlisted"):
        assert_scoped_ast_parity(source, permitted.replace("return value", "return 0"),
                                reviewed_edits=[("value=1", "value=2")])
    with pytest.raises(AssertionError, match="exactly once"):
        assert_scoped_ast_parity(source, permitted, reviewed_edits=[("not_present", "value=2")])


def test_reference_and_existing_opt_in_budget_are_not_conflated():
    assert [requested_subframes(REFERENCE, n) for n in (0,1,39)] == [56,56,56]
    assert [requested_subframes(TRIAL, n) for n in (0,1,39)] == [56,8,8]
    node = ast.parse(function_source(BASELINE, "_collect")).body[0]
    defaults = dict(zip([a.arg for a in node.args.kwonlyargs], node.args.kw_defaults, strict=True))
    assert ast.literal_eval(defaults["profile_render"]) is False
    assert ast.literal_eval(defaults["instance_backend"]) == "legacy"
    assert ast.dump(defaults["render_budget"]) == ast.dump(ast.Name(id="REFERENCE", ctx=ast.Load()))


def test_annotation_boundary_remains_after_native_validation_before_storage():
    tree = ast.parse(function_source(BASELINE, "_collect"))
    calls = [(n.func.id, n.lineno) for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    positions = {}
    for name in ("validate_native_static", "decode_native_instances", "component_masks",
                 "interval_visibility", "view_quality", "derive", "trace_review",
                 "write_compact_native_sample"):
        matches = [line for called, line in calls if called == name]
        assert len(matches) == 1
        positions[name] = matches[0]
    assert list(positions.values()) == sorted(positions.values())
    assert "global " not in function_source(BASELINE, "_collect")


def test_arbitrary_annotation_json_roundtrips_without_changing_optical_z(tmp_path):
    """Synthetic storage check only; fixture is NOT a valid robot/native view."""
    callback = synthetic_callback()
    extension = {"fixture_only": [None, False, 8, 5.0, "opaque selection evidence"],
                 "nested": {"never_training_approval": True}}
    callback["label"]["scaffold_extension"] = extension
    callback["trace"]["scaffold_extension"] = extension
    depth_bits = callback["depth"].view(np.uint32).copy()
    rgb = callback["rgb"].copy()
    folder = tmp_path / "synthetic_only"
    stored = capture_storage.write_compact_native_sample(folder, **callback)
    reader = SampleReader(folder)
    assert reader.verify_all()
    assert reader.json("supervision/label.json") == callback["label"]
    assert reader.json("supervision/query_trace.json") == callback["trace"] == stored["query_trace"]
    assert reader.read("inputs/depth_m.npy") == npy_bytes(callback["depth"])
    assert np.array_equal(reader.array("inputs/depth_m.npy").view(np.uint32), depth_bits)
    assert np.array_equal(callback["depth"].view(np.uint32), depth_bits)
    assert np.array_equal(reader.image("inputs/rgb.png"), rgb)
    assert callback["metadata"].get("files") is None
    assert not reader.json("supervision/label.json")["training_approved"]


def test_v2_native_body_has_only_exact_annotation_and_epoch_edits():
    assert_scoped_ast_parity(function_source(BASELINE, "_collect"),
                            function_source(FUTURE_WORKER, "_collect"),
                            reviewed_edits=[
    [
        "    from ..native_clear_labels import derive\n    from ..automated_native_review import trace_review\n",
        ""
    ],
    [
        "            metadata=dict(schema_version='greenhouse.generated_native_multiview_sample.v1',",
        "            metadata=dict(schema_version=SAMPLE_SCHEMA, **annotation_fields(),"
    ],
    [
        "            label=derive(metadata,generated['report'],rgb,depth,valid,components,catalogue)\n            trace=trace_review(metadata,generated['report'],label,rgb,depth,valid,components,catalogue) if label['eligible'] else None",
        "            label,trace=annotate_for_storage(metadata,generated['report'],rgb,depth,valid,components,catalogue,\n                                             target_mask=mask.astype(np.uint8)*255)"
    ],
    [
        "        return dict(state='native_generated_multiview_pilot_complete_pending_review',",
        "        return dict(state=CAPTURE_STATE,"
    ]
])


def test_v2_main_only_adds_explicit_batch_epoch_and_completion_pins():
    assert_scoped_ast_parity(function_source(BASELINE, "main"),
                            function_source(FUTURE_WORKER, "main"),
                            reviewed_edits=[
    [
        "def main():",
        "def main(argv=None):"
    ],
    [
        "    group=p.add_mutually_exclusive_group(required=True)\n    group.add_argument('--plan',type=Path);group.add_argument('--batch-plan',type=Path)",
        "    p.add_argument('--batch-plan',type=Path,required=True)\n    p.add_argument('--plan-sha256',required=True)\n    p.add_argument('--annotation-policy-sha256',required=True)"
    ],
    [
        "    a=p.parse_args()\n    require(not a.profile_render",
        "    a=p.parse_args(argv)\n    require(a.annotation_policy_sha256==annotation_fields()['annotation_policy_sha256'],\n            'Caller annotation policy pin mismatch')\n    require(sha256(a.batch_plan)==a.plan_sha256, 'Caller plan pin mismatch')\n    require(not a.profile_render"
    ],
    [
        "    plan_path=a.plan or a.batch_plan;plan=read_json(plan_path)\n    if a.batch_plan:\n        from ..native_multitarget_plan import check as check_batch\n        base=check_batch(plan,replay_geometry=False)\n    else:base=check(plan)",
        "    plan_path=a.batch_plan;plan=read_json(plan_path)\n    require('target_cases' in plan, 'Initial query-V2 capture requires a batch plan')\n    from ..native_multitarget_plan import check as check_batch\n    base=check_batch(plan,replay_geometry=False)"
    ],
    [
        "    source_roots.append(Path(qualification['path']).parent)",
        "    source_roots.append(Path(qualification['path']).parent)\n    source_roots.append(plan_path.resolve())"
    ],
    [
        "    output.mkdir(parents=True)",
        "    require(sha256(plan_path)==a.plan_sha256, 'Caller plan changed before launch')\n    output.mkdir(parents=True)"
    ],
    [
        "        compact_implementation_bindings=compact_bindings,storage_qualification=qualification,",
        "        compact_implementation_bindings=compact_bindings,storage_qualification=qualification,\n        **annotation_fields(),annotation_policy=annotation_policy(),\n        worker_module=WORKER_MODULE,worker_implementation_bindings=compact_bindings,"
    ],
    [
        "        if a.batch_plan:\n            base=check_batch(plan,replay_geometry=True)",
        "        base=check_batch(plan,replay_geometry=True)"
    ],
    [
        "        write_json(output/'result.json',result)",
        "        result.update(request_sha256=sha256(output/'request.json'),plan_sha256=a.plan_sha256)\n        require(sha256(plan_path)==a.plan_sha256, 'Caller plan changed during capture')\n        verify_epoch_documents(read_json(output/'request.json'),result)\n        write_json(output/'result.json',result)"
    ]
])


def test_v2_wrapper_only_adds_batch_epoch_and_implementation_bindings():
    assert_scoped_ast_parity(function_source(BASELINE, "collect"),
                            function_source(FUTURE_WORKER, "collect"),
                            reviewed_edits=[
    [
        "    bindings = implementation_bindings()\n    result = _collect",
        "    bindings = implementation_bindings()\n    require('target_cases' in plan, 'Initial query-V2 capture requires a batch plan')\n    result = _collect"
    ],
    [
        "                  compact_implementation_bindings=bindings, storage_qualification=qualification)",
        "                  compact_implementation_bindings=bindings, storage_qualification=qualification,\n                  **annotation_fields(), annotation_policy=annotation_policy(),\n                  worker_module=WORKER_MODULE, worker_implementation_bindings=bindings)"
    ]
])


def test_exact_uint8_target_boundary_and_no_runtime_rebinding():
    from . import compact_query_v2 as worker
    from . import query_audit_v2 as audit
    tree = ast.parse(function_source(FUTURE_WORKER, "_collect"))
    annotation = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                      and isinstance(n.value, ast.Call)
                      and isinstance(n.value.func, ast.Name)
                      and n.value.func.id == "annotate_for_storage")
    captured = {}
    def annotate(*args, target_mask):
        captured["mask"] = target_mask
        return {}, None
    mask = np.array([[False, True]], dtype=bool)
    namespace = dict(np=np, mask=mask, metadata={}, generated={"report":{}},
        rgb=None, depth=None, valid=None, components=None, catalogue=None,
        annotate_for_storage=annotate)
    exec(compile(ast.Module(body=[annotation], type_ignores=[]), "<annotation-boundary>", "exec"), namespace)
    assert captured["mask"].dtype == np.uint8
    assert captured["mask"].tolist() == [[0,255]]
    assert mask.tolist() == [[False,True]]
    for module in (worker, audit):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "exec(" not in source and "setattr(" not in source and "sys.modules" not in source
    assert worker.implementation_bindings() == audit.implementation_bindings()
    for module in (worker, audit, audit.selector, capture_storage, baseline):
        path = Path(module.__file__).resolve()
        assert worker.implementation_bindings()[str(path)] == sha256(path)


# All simulator/plan operations below are explicit CPU stubs from the frozen
# compact-worker tests. No native launcher or guard is executed.
from .test_compact_views import main_fixture as legacy_main_fixture


@pytest.fixture
def worker_fixture(legacy_main_fixture, monkeypatch):
    import json
    import sys
    from . import compact_query_v2 as worker
    from . import query_audit_v2 as audit
    output, events, proof, modules = legacy_main_fixture
    plan_path = Path(sys.argv[sys.argv.index("--batch-plan")+1])
    plan = json.loads(plan_path.read_text())
    plan["target_cases"] = []
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    def collect(*args, **kwargs):
        events.append("collect")
        assert kwargs == dict(profile_render=False, instance_backend="fast", render_budget=TRIAL)
        return dict(state=audit.CAPTURE_STATE, training_approved=False)
    monkeypatch.setattr(worker, "_collect", collect)
    argv = sys.argv[1:] + ["--plan-sha256", sha256(plan_path),
                         "--annotation-policy-sha256", audit.POLICY_SHA256]
    return output, events, proof, modules, argv


def test_request_result_exact_epoch_source_and_storage_bindings(worker_fixture):
    import json
    from . import compact_query_v2 as worker
    from . import query_audit_v2 as audit
    output, events, proof, _, argv = worker_fixture
    worker.main(argv)
    request = json.loads((output/"request.json").read_text())
    result = json.loads((output/"result.json").read_text())
    audit.verify_epoch_documents(request, result)
    assert result["request_sha256"] == sha256(output/"request.json")
    assert result["plan_sha256"] == request["plan_sha256"]
    assert result["worker_implementation_bindings"] == worker.implementation_bindings()
    assert request["storage_qualification"] == result["storage_qualification"] == proof
    assert request["native_instance_backend"] == "fast"
    assert request["render_budget_profile"] == TRIAL
    assert events.index("hash-check") < events.index("app") < events.index("full-replay") < events.index("collect")
    assert events[-2:] == ["close", "recheck"]  # explicit post-exit document verification above
    assert request["training_started"] is False and result["training_approved"] is False


@pytest.mark.parametrize("flag", ["--plan-sha256", "--annotation-policy-sha256"])
def test_caller_pins_fail_before_output_or_app(worker_fixture, flag):
    from . import compact_query_v2 as worker
    output, events, _, _, argv = worker_fixture
    argv[argv.index(flag)+1] = "0"*64
    with pytest.raises(ValueError, match="pin mismatch"):
        worker.main(argv)
    assert not output.exists() and "app" not in events


@pytest.mark.parametrize("mode", ["missing_epoch_flag", "nonbatch", "existing", "memory", "proof_root"])
def test_new_cli_preserves_pre_native_admission(worker_fixture, mode):
    import json
    from . import compact_query_v2 as worker
    output, events, proof, modules, argv = worker_fixture
    if mode == "missing_epoch_flag":
        argv = argv[:-2]
    elif mode == "nonbatch":
        path = Path(argv[argv.index("--batch-plan")+1])
        path.write_text(json.dumps({"implementation_bindings":{}}), encoding="utf-8")
        argv[argv.index("--plan-sha256")+1] = sha256(path)
    elif mode == "existing":
        output.mkdir()
    elif mode == "memory":
        modules["sim_physics.host_memory"].preflight = lambda: dict(allowed=False)
    elif mode == "proof_root":
        argv[argv.index("--output")+1] = str(Path(proof["path"]).parent/"bad")
    with pytest.raises((ValueError, SystemExit)):
        worker.main(argv)
    assert "app" not in events and not (output/"request.json").exists()


def test_capture_failure_never_publishes_completed_result(worker_fixture, monkeypatch):
    from . import compact_query_v2 as worker
    output, events, _, _, argv = worker_fixture
    def fail(*args, **kwargs):
        raise ValueError("synthetic failure")
    monkeypatch.setattr(worker, "_collect", fail)
    with pytest.raises(ValueError, match="synthetic failure"):
        worker.main(argv)
    assert (output/"failure.json").exists() and not (output/"result.json").exists()
    assert events[-1] == "close"


def test_cli_help_remains_cpu_only():
    import subprocess
    import sys
    script = """import sys
from sim_data.native_dataset import compact_query_v2, query_audit_v2
for module in (compact_query_v2, query_audit_v2):
    try: module.main(['--help'])
    except SystemExit as exc: assert exc.code == 0
assert not any(n == 'isaacsim' or n == 'pxr' or n.startswith(('pxr.', 'omni.')) for n in sys.modules)
"""
    result = subprocess.run([sys.executable, "-B", "-c", script],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--annotation-policy-sha256" in result.stdout and "--result-sha256" in result.stdout
