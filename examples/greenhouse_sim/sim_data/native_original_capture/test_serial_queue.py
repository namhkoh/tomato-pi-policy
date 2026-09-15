"""CPU-only queue boundaries; native launches and source preparation are mocked."""
import ast
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from . import serial_queue as q
from . import prepare

def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    q.oc.write_new(path, value)
    return q.oc.sha256(path)


@pytest.fixture
def request_fixture(tmp_path):
    isaac = tmp_path / "isaac" / "python.bat"
    isaac.parent.mkdir()
    isaac.write_text("CPU fixture only", encoding="utf-8")
    cpu = tmp_path / "cpu" / "python.exe"
    cpu.parent.mkdir()
    cpu.write_text("CPU fixture only", encoding="utf-8")
    deps = tmp_path / "native_deps"
    deps.mkdir()
    pred_root, scale_root = tmp_path / "v5", tmp_path / "v4"
    pilot_plan = tmp_path / "pilot_plan" / "plan.json"
    pilot_plan_sha = put(pilot_plan, {"cpu_fixture": True})
    config = dict(schema=q.v5.SCHEMA, output=str(pred_root), scale_output=str(scale_root),
                  original_plan=str(pilot_plan), original_plan_sha256=pilot_plan_sha)
    pred_path = tmp_path / "v5_request.json"
    pred_sha = put(pred_path, config)
    supervisor = dict(pid=7654321, creation_date="2026-09-16T01:00:00+00:00", executable=str(cpu),
        command_line=subprocess.list2cmdline([str(cpu), "-B", "-m", "sim_data.native_capture_v5.campaign_queue",
            "--request", str(pred_path), "--request-sha256", pred_sha]))
    batches, plans = [], []
    for family, count in q.BATCH_SHAPE:
        cases = [dict(case_id=f"sample_{i:04d}", target_id=family + f"/target_{i}",
            pose_request=dict(mode="exact_prior", native_pixel_xy=None),
            pose_prior=dict(prior_target_id=family + f"/target_{i}"),
            expected_calibration={"cpu_camera_index": i}) for i in range(1, count + 1)]
        plan = dict(schema=q.oc.SCHEMA, source_family=family, split="train",
            family_assignments=q.oc.FROZEN_SPLITS, geometry_mode="unmodified_original",
            admission=q.oc.ADMISSION, scene_policy=q.oc.SCENE_POLICY,
            stage_reuse="one_original_donor_scene_one_native_product", sample_count_limit=count,
            cases=cases, implementation_bindings=q.oc.LOADED_IMPLEMENTATION,
            package=str(tmp_path / "assets"), source_collection_plan=str(tmp_path / "clear" / "plan.json"),
            pose_prior=dict(source_capture=str(tmp_path / "historical"),
                            prior_plan=str(tmp_path / "historical_plan" / "plan.json")),
            source_bindings={str(tmp_path / "assets" / "unread.usd"): "a"*64})
        path = tmp_path / (family + "_plan") / "plan.json"
        batches.append(dict(plan_path=str(path), plan_sha256=put(path, plan)))
        plans.append(plan)
    pilot = dict(capture_path=str(pred_root / "original_capture"), result_sha256="b"*64,
                 launcher_receipt_path=str(pred_root / "original_launcher_receipt.json"),
                 launcher_receipt_sha256="c"*64)
    request = dict(schema=q.SCHEMA, output=str(tmp_path / "new_queue"), isaac_python=str(isaac),
        native_deps=str(deps), max_planned_cases=39, pilot=pilot,
        predecessor=dict(request_path=str(pred_path), request_sha256=pred_sha, supervisor=supervisor),
        batches=batches)
    return SimpleNamespace(request=request, config=config, plans=plans, tmp=tmp_path)


def test_validate_no_preparation_or_source_read(request_fixture, monkeypatch):
    def forbidden(*a, **k):
        raise AssertionError("Queue must not reconstruct source plans")
    monkeypatch.setattr(prepare, "check_plan", forbidden)
    monkeypatch.setattr(prepare, "prepare_batch", forbidden)
    before = deepcopy(request_fixture.request)
    config, plans = q.validate_request(before)
    assert config == request_fixture.config
    assert [len(p["cases"]) for p in plans] == [16, 23]
    assert request_fixture.request == before
    assert not Path(before["output"]).exists()


@pytest.mark.parametrize("kind", [
    "bound", "boolean_bound", "unknown_field", "reorder", "duplicate", "stale_plan",
    "wrong_family", "changed_target", "reframe", "duplicate_camera", "policy", "implementation",
    "pilot_other_queue", "supervisor_other_request", "supervisor_unknown_birth", "output_source",
])
def test_request_rejections(request_fixture, kind):
    r = request_fixture.request
    if kind == "bound":
        r["max_planned_cases"] = 40
    elif kind == "boolean_bound":
        r["max_planned_cases"] = True
    elif kind == "unknown_field":
        r["skip_audit"] = True
    elif kind == "reorder":
        r["batches"].reverse()
    elif kind == "duplicate":
        r["batches"][1] = deepcopy(r["batches"][0])
    elif kind == "stale_plan":
        r["batches"][0]["plan_sha256"] = "0"*64
    elif kind == "pilot_other_queue":
        r["pilot"]["capture_path"] = str(request_fixture.tmp / "other")
    elif kind == "supervisor_other_request":
        r["predecessor"]["supervisor"]["command_line"] += " --skip"
    elif kind == "supervisor_unknown_birth":
        r["predecessor"]["supervisor"]["creation_date"] = None
    elif kind == "output_source":
        r["output"] = str(Path(r["batches"][0]["plan_path"]).parent / "bad")
    else:
        p = request_fixture.plans[0]
        if kind == "wrong_family":
            p["source_family"] = "seed13_full"
        elif kind == "changed_target":
            p["cases"][0]["pose_prior"]["prior_target_id"] = "other_target"
        elif kind == "reframe":
            p["cases"][0]["pose_request"]["mode"] = "reframe_head_only"
        elif kind == "duplicate_camera":
            p["cases"][1]["expected_calibration"] = p["cases"][0]["expected_calibration"]
        elif kind == "policy":
            p["scene_policy"] = dict(p["scene_policy"], render_subframes=8)
        elif kind == "implementation":
            p["implementation_bindings"] = {}
        path = Path(r["batches"][0]["plan_path"])
        path.write_text(json.dumps(p), encoding="utf-8")
        r["batches"][0]["plan_sha256"] = q.oc.sha256(path)
    with pytest.raises((ValueError, KeyError, TypeError)):
        q.validate_request(r)


def supervisor_row(f):
    s = f.request["predecessor"]["supervisor"]
    return dict(ProcessId=s["pid"], CreationDate=s["creation_date"],
                ExecutablePath=s["executable"], CommandLine=s["command_line"])


def terminal_fixture(f):
    root, scale = Path(f.config["output"]), Path(f.config["scale_output"])
    bindings = {str(Path(m.__file__).resolve()): q.oc.sha256(m.__file__) for m in (q.v5, q.campaign, q.guard)}
    put(root / "request.json", dict(f.config, implementation_bindings=bindings, training_approved=False))
    scale_sha = put(scale / "result.json", dict(records=[{"cpu_only": True}],
        new_biological_families=0, training_approved=False, source_cap_reset=False))
    put(root / "result.json", dict(state="serial_probe_and_scale_complete_pending_admission",
        request_sha256=f.request["predecessor"]["request_sha256"], scale_result_sha256=scale_sha,
        training_approved=False, source_cap_reset=False))
    put(root / "queue_000099.json", dict(state="scale_coordinator_running_in_same_cpu_process",
        worker_pid=f.request["predecessor"]["supervisor"]["pid"], bindings=bindings,
        training_approved=False, source_cap_reset=False))
    return root


def test_predecessor_live_no_terminal_is_wait_not_launch(request_fixture, monkeypatch):
    f = request_fixture
    monkeypatch.setattr(q, "supervisor_rows", lambda pid: [supervisor_row(f)])
    assert q.predecessor_completion(f.request["predecessor"], f.config) is None


def test_terminal_still_live_must_wait_then_exact_absence(request_fixture, monkeypatch):
    f = request_fixture
    terminal_fixture(f)
    monkeypatch.setattr(q, "supervisor_rows", lambda pid: [supervisor_row(f)])
    assert q.predecessor_completion(f.request["predecessor"], f.config) is None
    monkeypatch.setattr(q, "supervisor_rows", lambda pid: [])
    result = q.predecessor_completion(f.request["predecessor"], f.config)
    assert result["state"] == "exact_v5_terminal_and_supervisor_absent"
    assert len(result["bindings"]) == 4
    q.oc.bind_all(result["bindings"])


@pytest.mark.parametrize("kind", ["missing_terminal", "wrong_terminal_request", "wrong_supervisor_event",
    "changed_scale", "failed", "superseded", "pid_reuse", "metadata_missing", "query_failure"])
def test_predecessor_fail_closed(request_fixture, monkeypatch, kind):
    f = request_fixture
    root = terminal_fixture(f) if kind != "missing_terminal" else Path(f.config["output"])
    monkeypatch.setattr(q, "supervisor_rows", lambda pid: [])
    if kind == "wrong_terminal_request":
        path = root / "result.json"
        value = q.oc.read_json(path)
        value["request_sha256"] = "f"*64
        path.write_text(json.dumps(value), encoding="utf-8")
    elif kind == "wrong_supervisor_event":
        path = root / "queue_000099.json"
        value = q.oc.read_json(path)
        value["worker_pid"] += 1
        path.write_text(json.dumps(value), encoding="utf-8")
    elif kind == "changed_scale":
        (Path(f.config["scale_output"]) / "result.json").write_text("{}", encoding="utf-8")
    elif kind in ("failed", "superseded"):
        put(root / ("failure.json" if kind == "failed" else "superseded_idle_queue.json"), {})
    elif kind in ("pid_reuse", "metadata_missing"):
        row = supervisor_row(f)
        row["CreationDate"] = "2026-09-17T01:00:00+00:00" if kind == "pid_reuse" else None
        monkeypatch.setattr(q, "supervisor_rows", lambda pid: [row])
    elif kind == "query_failure":
        def fail(pid):
            raise OSError("CIM failed")
        monkeypatch.setattr(q, "supervisor_rows", fail)
    with pytest.raises((ValueError, OSError)):
        q.predecessor_completion(f.request["predecessor"], f.config)


@pytest.mark.parametrize("decision,count,noncaptured", [
    (q.STRICT, 1, 0), ("hold_visual_clarity", 1, 0),
    ("exclude_geometry_or_visibility", 1, 0), (q.STRICT, 2, 0), (q.STRICT, 1, 1),
])
def test_pilot_public_replay_exactly_once(request_fixture, monkeypatch, decision, count, noncaptured):
    f, calls = request_fixture, []
    def replay(pin):
        calls.append(pin)
        return ([dict(decision=decision, source_family="seed73_full", provenance=dict(
            plan_path=f.config["original_plan"], plan_sha256=f.config["original_plan_sha256"]))],
            dict(planned_rows=count, noncaptured_rows=noncaptured), {"cpu_bound": "hash"})
    monkeypatch.setattr(q.pilot_adapter, "original_observed_rows", replay)
    if decision == q.STRICT and count == 1 and noncaptured == 0:
        result = q.qualify_pilot(f.request["pilot"], f.config)
        assert result["decision"] == q.STRICT
        assert not result["training_approved"]
    else:
        with pytest.raises(ValueError):
            q.qualify_pilot(f.request["pilot"], f.config)
    assert len(calls) == 1


@pytest.mark.parametrize("commit,disk,allowed,blockers,expected", [
    (20,60,True,[],True), (19,60,True,[],False), (20,59,True,[],False),
    (20,60,False,[],False), (20,60,True,[{"ProcessId":8}],False),
])
def test_exact_reserve_boundaries(tmp_path, monkeypatch, commit, disk, allowed, blockers, expected):
    monkeypatch.setattr(q, "preflight", lambda: dict(checked=True, allowed=allowed, commit_headroom_bytes=commit*q.GIB))
    monkeypatch.setattr(q.shutil, "disk_usage", lambda p: SimpleNamespace(free=disk*q.GIB))
    monkeypatch.setattr(q.guard, "process_inventory", lambda: dict(blockers=blockers,
        classifications=[{"ProcessId":1}], no_blockers_observed=not blockers))
    assert q.resource_evidence(tmp_path)["allowed"] is expected


@pytest.mark.parametrize("kind", ["memory_unchecked", "memory_missing", "guard_empty", "guard_failure"])
def test_invalid_resource_measurements_do_not_clear(tmp_path, monkeypatch, kind):
    memory = dict(checked=True, allowed=True, commit_headroom_bytes=20*q.GIB)
    report = dict(blockers=[], classifications=[{"ProcessId":1}], no_blockers_observed=True)
    if kind == "memory_unchecked":
        memory["checked"] = False
    elif kind == "memory_missing":
        del memory["commit_headroom_bytes"]
    elif kind == "guard_empty":
        report["classifications"] = []
    def process():
        if kind == "guard_failure":
            raise OSError("CIM query failed")
        return report
    monkeypatch.setattr(q, "preflight", lambda: memory)
    monkeypatch.setattr(q.guard, "process_inventory", process)
    monkeypatch.setattr(q.shutil, "disk_usage", lambda p: SimpleNamespace(free=60*q.GIB))
    with pytest.raises((ValueError, OSError)):
        q.resource_evidence(tmp_path)


@pytest.fixture
def run_harness(request_fixture, monkeypatch):
    f = request_fixture
    # Fixtures exercise real queue request/binding/receipt handling; no renderer.
    pilot_result = Path(f.request["pilot"]["capture_path"]) / "result.json"
    f.request["pilot"]["result_sha256"] = put(pilot_result, {"cpu_fixture": True})
    receipt = Path(f.request["pilot"]["launcher_receipt_path"])
    if not receipt.exists():
        put(receipt, {"cpu_fixture": True})
    f.request["pilot"]["launcher_receipt_sha256"] = q.oc.sha256(receipt)
    input_path = f.tmp / "queue_input.json"
    input_pin = put(input_path, f.request)
    calls, launches, audits, waits, pipeline = [], [], [], [], []
    monkeypatch.setattr(q, "qualify_pilot", lambda *a: calls.append("pilot") or dict(
        decision=q.STRICT, verified_bindings={}, training_approved=False, source_cap_reset=False))
    predecessor = dict(state="exact_v5_terminal_and_supervisor_absent", bindings={
        f.request["predecessor"]["request_path"]: f.request["predecessor"]["request_sha256"]})
    monkeypatch.setattr(q, "predecessor_completion", lambda *a: predecessor)
    monkeypatch.setattr(q, "supervisor_rows", lambda pid: [])
    monkeypatch.setattr(q, "resource_evidence", lambda path: dict(allowed=True, cpu_fixture=True))
    monkeypatch.setattr(q.time, "sleep", lambda seconds: waits.append(seconds))
    def runner(command, log_path, *, bindings, environment, native, reserve, launch_check, announce):
        pipeline.append("reserve")
        reserve()
        launch_check()
        assert native is True
        assert environment["PYTHONPATH"].split(q.os.pathsep)[0] == f.request["native_deps"]
        assert command[1:3] == ["-m", "sim_data.native_original_capture.collector"]
        launches.append(command)
        pipeline.append("launch")
        announce(2000 + len(launches))
        capture = Path(command[-1])
        capture.mkdir()
        plan_sha = command[command.index("--plan-sha256") + 1]
        plan_path = command[command.index("--plan") + 1]
        put(capture / "request.json", dict(plan_path=plan_path, plan_sha256=plan_sha))
        put(capture / "result.json", dict(plan_sha256=plan_sha, request_sha256=q.oc.sha256(capture / "request.json")))
        pipeline.append("exit")
        return 0
    def audit(capture, *, result_sha256):
        audits.append(str(capture))
        pipeline.append("audit")
        assert result_sha256 == q.oc.sha256(Path(capture) / "result.json")
        return dict(counts={q.STRICT: 1}, cpu_fixture=True)
    monkeypatch.setattr(q.campaign, "run_checked", runner)
    monkeypatch.setattr(q.audit, "audit_capture", audit)
    run = lambda: q.run(input_path, input_pin)
    return SimpleNamespace(f=f, run=run, calls=calls, launches=launches, audits=audits, waits=waits,
        pipeline=pipeline, runner=runner, input_path=input_path, input_pin=input_pin,
        output=Path(f.request["output"]))


def test_serial_order_and_truthful_receipt_api(run_harness):
    h = run_harness
    h.run()
    assert h.calls == ["pilot"]
    assert len(h.launches) == len(h.audits) == 2
    assert h.pipeline == ["reserve", "launch", "exit", "audit"] * 2
    result = q.oc.read_json(h.output / "result.json")
    assert result["planned_cases"] == 39 and result["state"] == q.RESULT_STATE
    for index, record in enumerate(result["records"], 1):
        receipt = q.oc.read_json(q.oc.pin(record["launcher_receipt_path"], record["launcher_receipt_sha256"]))
        assert receipt["schema"] == q.RECEIPT_SCHEMA and receipt["state"] == q.RECEIPT_STATE
        assert receipt["batch_index"] == index and receipt["producer_module"] == q.PRODUCER_MODULE
        assert receipt["producer_implementation_bindings"] == q.implementation_bindings()
        assert str(Path(q.__file__).resolve()) in receipt["source_bindings"]
        assert receipt["exit_code"] == 0 and not receipt["training_approved"] and not receipt["source_cap_reset"]
        event = q.oc.read_json(q.oc.pin(receipt["owned_worker"]["event_path"], receipt["owned_worker"]["event_sha256"]))
        assert event["state"] == q.LAUNCH_STATE and event["state"] != "original_pilot_running"
        assert event["bindings"] == receipt["source_bindings"]
        assert event["command"] == receipt["owned_worker"]["command"] == h.launches[index-1]
        queue_request = q.oc.read_json(q.oc.pin(receipt["queue_request_path"], receipt["queue_request_sha256"]))
        assert queue_request["batches"][index-1]["plan_sha256"] == receipt["plan_sha256"]
        q.oc.bind_all(receipt["source_bindings"])


def test_waits_for_predecessor_before_reserving_or_launching(run_harness, monkeypatch):
    h, seen = run_harness, []
    def predecessor(*a):
        seen.append(len(h.launches))
        return None if len(seen) == 1 else dict(bindings={str(h.input_path):h.input_pin})
    monkeypatch.setattr(q, "predecessor_completion", predecessor)
    h.run()
    assert seen == [0,0] and h.waits == [30] and h.calls == ["pilot"]


@pytest.mark.parametrize("kind", ["exit", "audit", "changed_request", "failure_file", "event_tamper", "resource_race"])
def test_stop_first_failure_no_second_batch(run_harness, monkeypatch, kind):
    h = run_harness
    def runner(*args, **kwargs):
        result = h.runner(*args, **kwargs)
        capture = Path(h.launches[-1][-1])
        if kind == "exit":
            return 1
        if kind == "changed_request":
            (capture / "request.json").write_text("{}", encoding="utf-8")
        if kind == "failure_file":
            put(capture / "failure.json", {})
        if kind == "event_tamper":
            (capture.parent / "queue_000001.json").write_text("{}", encoding="utf-8")
        return result
    monkeypatch.setattr(q.campaign, "run_checked", runner)
    if kind == "audit":
        def fail(*a, **k):
            raise ValueError("CPU audit failure")
        monkeypatch.setattr(q.audit, "audit_capture", fail)
    if kind == "resource_race":
        calls = []
        def resource(path):
            calls.append(1)
            return dict(allowed=len(calls) == 1)
        monkeypatch.setattr(q, "resource_evidence", resource)
    with pytest.raises((ValueError, KeyError)):
        h.run()
    assert len(h.launches) <= 1
    assert (h.output / "failure.json").exists() and not (h.output / "result.json").exists()
    assert not (h.output / "batch_002").exists()
    assert not (h.output / "batch_001" / "original_launcher_receipt.json").exists()


def test_reserve_wait_rechecks_without_replaying_pilot(run_harness, monkeypatch):
    h, count = run_harness, []
    def resource(path):
        count.append(1)
        return dict(allowed=len(count)>1)
    monkeypatch.setattr(q, "resource_evidence", resource)
    h.run()
    assert h.waits == [30] and h.calls == ["pilot"] and len(h.launches) == 2


def test_source_code_map_and_no_native_or_prepare_import_in_queue():
    mapping = q.implementation_bindings()
    assert mapping[str(Path(q.__file__).resolve())] == q.oc.sha256(q.__file__)
    assert set(q.oc.LOADED_IMPLEMENTATION) <= mapping.keys()
    tree = ast.parse(Path(q.__file__).read_text(encoding="utf-8"))
    forbidden = {"prepare_batch", "check_plan", "SimulationApp", "collect"}
    assert not any(isinstance(n, ast.Call) and (
        isinstance(n.func, ast.Name) and n.func.id in forbidden or
        isinstance(n.func, ast.Attribute) and n.func.attr in forbidden) for n in ast.walk(tree))
    assert str(Path(q.__file__).resolve()) not in q.oc.LOADED_IMPLEMENTATION


def test_actual_frozen_plan_hashes_and_shallow_queue_compatibility(request_fixture, monkeypatch):
    root = Path(q.__file__).resolve().parents[4]
    names = [
        ("original_seed73_16view_plan_20260916_v1", "e3cc5b05efc1d92388c7a5eea291f6b8558b15926d9c9f6d14eec237e0e8a52d"),
        ("original_seed23_23view_plan_20260916_v1", "4d69e96efd141fd4b9c185289f2e232e333f7245f7b320538483bd33456de630"),
    ]
    paths = [(root / "data/sim_data/diagnostics" / name / "plan.json", pin) for name, pin in names]
    if not all(p.exists() for p, _ in paths):
        pytest.skip("Host-specific frozen diagnostic plans absent")
    def forbidden(*a, **k):
        raise AssertionError("No full source re-preparation in queue")
    monkeypatch.setattr(prepare, "prepare_batch", forbidden)
    monkeypatch.setattr(prepare, "check_plan", forbidden)
    request_fixture.request["batches"] = [dict(plan_path=str(p), plan_sha256=pin) for p, pin in paths]
    _, plans = q.validate_request(request_fixture.request)
    assert [len(p["cases"]) for p in plans] == [16,23]
    assert [len(p["implementation_bindings"]) for p in plans] == [73,73]
    assert [p["coverage"]["selected_same_target_priors"] for p in plans] == [16,23]


def test_terminal_mutation_during_verification_rejected(request_fixture, monkeypatch):
    f = request_fixture
    root = terminal_fixture(f)
    monkeypatch.setattr(q, "supervisor_rows", lambda pid: [])
    original = q.oc.bind_all
    calls = []
    def bind(mapping):
        original(mapping)
        calls.append(1)
        if len(calls) == 2:
            # Simulate external mutation after document validation, before handoff.
            (root / "result.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(q.oc, "bind_all", bind)
    with pytest.raises(ValueError, match="Changed or missing pinned file"):
        q.predecessor_completion(f.request["predecessor"], f.config)


def test_completed_first_receipt_cannot_change_during_second_batch(run_harness, monkeypatch):
    h = run_harness
    def runner(*args, **kwargs):
        code = h.runner(*args, **kwargs)
        if len(h.launches) == 2:
            (h.output / "batch_001" / "original_launcher_receipt.json").write_text("{}", encoding="utf-8")
        return code
    monkeypatch.setattr(q.campaign, "run_checked", runner)
    with pytest.raises(ValueError, match="Changed or missing pinned file"):
        h.run()
    assert not (h.output / "result.json").exists()


def test_pilot_failure_launches_nothing(run_harness, monkeypatch):
    h = run_harness
    def reject(*a):
        raise ValueError("Not a strict pilot")
    monkeypatch.setattr(q, "qualify_pilot", reject)
    with pytest.raises(ValueError, match="Not a strict pilot"):
        h.run()
    assert not h.launches and not (h.output / "batch_001").exists()


def test_guard_and_cim_queries_are_never_native_launches(request_fixture, monkeypatch):
    s = request_fixture.request["predecessor"]["supervisor"]
    calls = []
    def query(command, **kwargs):
        calls.append((command, kwargs))
        return "[]"
    monkeypatch.setattr(q.subprocess, "check_output", query)
    assert q.supervisor_rows(s["pid"]) == []
    command, kwargs = calls[0]
    assert command[:4] == ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command"]
    assert f'ProcessId = {s["pid"]}' in command[-1] and "-ErrorAction Stop" in command[-1]
    assert kwargs["creationflags"] == subprocess.CREATE_NO_WINDOW
