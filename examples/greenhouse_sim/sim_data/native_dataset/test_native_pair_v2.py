"""CPU-only capture parity, verified-scene lighting and executor receipt tests."""
import ast
import builtins
from copy import deepcopy
from pathlib import Path
import symtable
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from . import native_pair_v2 as worker
from .. import native_generated_pair as frozen
from ..dataset_review import read_json, write_json
from ..depth_preview import sha256

FROZEN_PATH = Path(frozen.__file__)
SCENE_PATH = FROZEN_PATH.with_name("native_scene.py")
WORKER_PATH = Path(worker.__file__)


def function(path, name):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)


def dump(node):
    return ast.dump(node, include_attributes=False)


def scene_return_keys():
    node = function(SCENE_PATH, "prepare_native_scene")
    returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
    assert len(returns) == 1 and returns[0] is node.body[-1]
    return {k.value for k in returns[0].value.keys}


def test_capture_ast_parity_only_import_depth_and_explicit_lighting_fix():
    old = function(FROZEN_PATH, "capture")
    new = function(WORKER_PATH, "capture")
    assignments = [n for n in new.body if isinstance(n, ast.Assign)
                   and any(isinstance(t, ast.Name) and t.id == "lighting" for t in n.targets)]
    assert len(assignments) == 1
    lighting = assignments[0]
    assert dump(lighting) == dump(ast.parse(
        'lighting = deepcopy(scene["old_manifest"]["lighting"])').body[0])
    position = new.body.index(lighting)
    assert dump(new.body[position - 1]) == dump(ast.parse(
        "scene = prepare_native_scene(app, plan)").body[0])
    new.body.remove(lighting)
    for n in ast.walk(new):
        if isinstance(n, ast.ImportFrom) and n.level:
            assert n.level == 2
            n.level = 1
    assert dump(new) == dump(old)


def test_shared_scene_applies_actual_daylight_and_requires_equality_before_return():
    body = function(SCENE_PATH, "prepare_native_scene").body
    apply_node = ast.parse(
        "lighting = daylight.apply(stage, day=172, minutes=13*60, "
        "intensity=1500, dome_intensity=6000)").body[0]
    equality_node = ast.parse(
        'require(lighting == old_manifest["lighting"], "Original lighting changed")').body[0]
    apply_index = next(i for i, n in enumerate(body) if dump(n) == dump(apply_node))
    assert dump(body[apply_index + 1]) == dump(equality_node)
    assert isinstance(body[-1], ast.Return)
    assert len([n for n in ast.walk(ast.Module(body=body, type_ignores=[]))
                if isinstance(n, ast.Return)]) == 1
    assert apply_index + 1 < len(body) - 1
    assert "old_manifest" in scene_return_keys()
    assert "lighting" not in scene_return_keys()
    # No subsequent reassignment that could sever the equality's provenance.
    for node in body[apply_index + 2:]:
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Name) and isinstance(subnode.ctx, ast.Store):
                assert subnode.id not in {"lighting", "old_manifest"}
    # Execute the actual two source statements, not a invented return contract.
    actual = {"profile": {"intensity": 1500}}
    namespace = dict(daylight=SimpleNamespace(apply=lambda *a, **k: actual),
                     stage=object(), old_manifest={"lighting": deepcopy(actual)},
                     require=worker.require)
    code = compile(ast.Module(body=body[apply_index:apply_index + 2], type_ignores=[]),
                   str(SCENE_PATH), "exec")
    exec(code, namespace)
    namespace["old_manifest"]["lighting"]["profile"]["intensity"] = 1
    with pytest.raises(ValueError, match="Original lighting changed"):
        exec(code, namespace)


def test_no_unbound_capture_global_and_frozen_regression_is_lighting():
    def missing(module):
        source = Path(module.__file__).read_text(encoding="utf-8")
        table = next(t for t in symtable.symtable(source, module.__file__, "exec").get_children()
                     if t.get_name() == "capture")
        return {s.get_name() for s in table.get_symbols()
                if s.is_referenced() and s.is_global()
                and s.get_name() not in vars(module) and not hasattr(builtins, s.get_name())}
    assert missing(frozen) == {"lighting"}
    assert missing(worker) == set()


def capture_harness(path, *, reject=None):
    """Execute the real capture AST with CPU stand-ins for native imports only.

    Tiny arrays test Python control flow/metadata, not native visibility approval.
    No Kit import, scene launch, artifact write or physical action occurs.
    """
    events, written, step_budgets, checks = [], {}, [], []
    original = {"variant_id": "original", "plant_root": "/Original"}
    generated = {"variant_id": "generated", "plant_root": "/Generated"}
    def row(variant):
        return dict(variant_id=variant, component_id="petiole", target_id=variant + "_target",
                    cut_region_proposal={"nominal": {"petiole_radius_m": .003}})
    cal = {"resolution": [1696, 816]}
    plan = dict(modes=["original_control", "generated_variant"],
                source_row=row("original"), generated_row=row("generated"),
                expected_calibration=cal, split_group="frozen_train_family",
                conservative_view_cap_group="original_target", source_bindings={"plan": "hash"})
    scene = dict(
        started=worker.time.perf_counter(),
        stage=SimpleNamespace(GetUsedLayers=lambda: []),
        records=[], reports=[], variants=[original], generated={"cpu_fixture": True},
        counts={"components": 1}, robot={"root": "/Robot"},
        old_manifest={"lighting": {"profile": {"intensity": 1500}},
                      "renderer": "RealTimePathTracing"},
        old_sample={}, original_bindings={"original": "hash"},
        original_variant=original, pose={"complete_snapshot": "CPU stand-in"},
        settings={"/rtx/rendermode": "RealTimePathTracing"}, setup_seconds=1.0)
    assert set(scene) == scene_return_keys()
    def gate(name, result):
        def call(*args, **kwargs):
            events.append(name)
            if reject == name:
                raise ValueError("CPU gate rejection: " + name)
            return result
        return call
    product = SimpleNamespace(destroy=lambda: events.append("product_destroy"))
    def render_product(camera, resolution):
        checks.append(("product", camera, resolution))
        return product
    writer = SimpleNamespace(
        sequence=0, attach=lambda products: checks.append(("attach", products)),
        detach=lambda: events.append("writer_detach"))
    def make_writer(rep, **kwargs):
        checks.append(("writer", kwargs))
        return writer
    def step_payload(rep, writer, *, subframes):
        writer.sequence += 1
        step_budgets.append(subframes)
        return {"camera_params": {"native_mock": True}, "pilot_render_frame": writer.sequence}
    mask = np.ones((2, 2), dtype=bool)
    def write_sample(folder, rgb, depth, valid, metadata):
        events.append("write_sample")
        written[folder / "sample.json"] = metadata
    substitution = dict(records=[], variants=[generated], report={}, old_root="/Original",
                        new_root="/Generated", plant_to_world=np.eye(4).tolist(),
                        component_count_preserved=True)
    namespace = dict(vars(worker))
    # Remove runtime import statements only; all capture gates/body remain intact.
    node = function(path, "capture")
    node.body = [n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    namespace.update(
        prepare_native_scene=gate("prepare_native_scene", scene),
        rep=SimpleNamespace(create=SimpleNamespace(render_product=render_product)),
        HEAD_CAMERA="/World/RBY1/link_head_2/attachments/HeadCamera/D405/DepthCamera",
        make_writer=make_writer, step_payload=step_payload,
        calibration=lambda stage: cal,
        calibration_for_native_resolution=lambda cal, resolution: cal,
        assert_same_camera=gate("assert_same_camera", None),
        StaticBoundScreen=lambda *a: gate("geometry_screen", {"passed": True}),
        static_obstacles=lambda *a: [], scene_triangle_refiner=lambda *a, **k: None,
        visible_bounds=lambda *a: [],
        write_json=lambda *a: None,
        component_catalogue=lambda stage, records, reports, variants: [
            dict(variant_id=variants[0]["variant_id"], component_id="petiole")],
        target_world_geometry=lambda *a: {
            "nominal_world_m": [0, 0, 1], "interval_world_m": [[0, 0, 1], [.01, 0, 1]]},
        source_hashes=gate("source_hashes", {"usd": "unchanged"}),
        StaticSceneMonitor=lambda *a: SimpleNamespace(
            begin=lambda: "static", token=lambda: "static",
            close=lambda: events.append("monitor_close")),
        validate_native_static=gate("validate_native_static", (
            np.zeros((2, 2, 3), dtype=np.uint8), np.ones((2, 2)), mask, {}, {})),
        decode_native_instances=gate("decode_native_instances", (
            np.ones((2, 2), dtype=np.int32), {1: "/Original/petiole"})),
        component_masks=lambda *a: ({}, {}, {}),
        project=lambda points, cal: [{"pixel_xy": [1, 1], "depth_m": 1} for _ in points],
        interval_visibility=gate("interval_visibility", ({"cpu_mock_only": True}, mask)),
        view_quality=gate("view_quality", {"cpu_mock_only": True}),
        sensor_profile=lambda resolution: {"resolution": list(resolution)},
        depth_evidence=lambda *a: {"cpu_mock_only": True},
        jsonable=lambda value: value, write_sample=write_sample,
        write_visibility=lambda *a: None, read_json=lambda path: written[path],
        substitute_plant=gate("substitute_plant", substitution),
        assert_pair_fresh=gate("assert_pair_fresh", {"cpu_mock_only": True}),
        verify_bindings=gate("verify_bindings", None))
    code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
    exec(code, namespace)
    run = lambda: namespace["capture"](None, Path("cpu-only-no-artifacts"), plan)
    return SimpleNamespace(run=run, events=events, written=written, scene=scene,
                           budgets=step_budgets, checks=checks)


def test_cpu_capture_reproduces_old_nameerror_and_v2_emits_verified_lighting():
    old = capture_harness(FROZEN_PATH)
    with pytest.raises(NameError, match="lighting"):
        old.run()
    assert not old.written
    assert old.events[-3:] == ["monitor_close", "writer_detach", "product_destroy"]
    current = capture_harness(WORKER_PATH)
    result = current.run()
    assert len(result["samples"]) == 2
    assert current.budgets == [8] * 14
    assert current.checks[0][0] == "product" and current.checks[0][2] == (1696, 816)
    assert current.checks[1] == ("writer", {"include_instances": True, "instance_backend": "legacy"})
    for sample in result["samples"]:
        lighting = sample["lighting"]
        verified = current.scene["old_manifest"]["lighting"]
        assert lighting == verified and lighting is not verified
        assert lighting["profile"] is not verified["profile"]
        assert sample["synchronization"]["render_budget_subframes"] == 56
        assert sample["calibration"]["resolution"] == [1696, 816]
        assert sample["robot_snapshot"] is current.scene["pose"]
        assert not sample["training_sample_approved"]
    assert not result["training_approved"]
    assert not result["original_controls_are_new_target_diversity"]
    assert current.events[-1] == "verify_bindings"


@pytest.mark.parametrize("gate", [
    "prepare_native_scene", "geometry_screen", "assert_same_camera",
    "validate_native_static", "decode_native_instances", "interval_visibility", "view_quality",
])
def test_capture_gate_rejection_is_not_promoted_or_written(gate):
    harness = capture_harness(WORKER_PATH, reject=gate)
    with pytest.raises(ValueError, match="CPU gate rejection: " + gate):
        harness.run()
    assert not harness.written
    if gate != "prepare_native_scene":
        assert harness.events[-2:] == ["writer_detach", "product_destroy"]


def test_implementation_bindings_exact_three_real_files():
    expected = {str(p.resolve()): sha256(p) for p in (FROZEN_PATH, SCENE_PATH, WORKER_PATH)}
    assert worker.implementation_bindings() == expected
    assert worker._LOADED_IMPLEMENTATION_BINDINGS == expected
    assert worker.windows_worker_admission is frozen.windows_worker_admission


@pytest.fixture
def main_harness(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    write_json(source / "manifest.json", {"package": str(tmp_path / "package")})
    plan_path = tmp_path / "plan.json"
    plan = dict(source_capture=str(source), variant_directory=str(tmp_path / "variant"),
                prerequisite_directory=str(tmp_path / "prerequisite"))
    write_json(plan_path, plan)
    output = tmp_path / "new-output"
    closes, launches, captures = [], [], []
    monkeypatch.setattr(worker, "check_plan", lambda plan: None)
    monkeypatch.setattr(worker, "verify_sensor_prerequisite",
                        lambda plan: {"bindings": {str(plan_path): sha256(plan_path)}})
    monkeypatch.setattr(worker, "windows_worker_admission",
                        lambda pid: {"controller_pid": pid, "cpu_fixture": True})
    memory = ModuleType("sim_physics.host_memory")
    memory.preflight = lambda: {"allowed": True}
    monkeypatch.setitem(sys.modules, "sim_physics.host_memory", memory)
    isaacsim = ModuleType("isaacsim")
    def create_app(config):
        launches.append(config)
        return SimpleNamespace(close=lambda *, exit_code: closes.append(exit_code))
    isaacsim.SimulationApp = create_app
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)
    def capture(app, output, plan, **kwargs):
        captures.append(kwargs)
        return {"state": "generated_native_pair_captured_pending_visual_review",
                "training_approved": False}
    monkeypatch.setattr(worker, "capture", capture)
    run = lambda: worker.main(["--plan", str(plan_path), "--output", str(output),
                              "--controller-pid", "123", "--profile-geometry-cache"])
    return SimpleNamespace(run=run, output=output, closes=closes, launches=launches,
                           captures=captures, plan_path=plan_path, memory=memory)


def test_main_request_and_result_match_public_binding_map(main_harness):
    h = main_harness
    h.run()
    request, result = read_json(h.output / "request.json"), read_json(h.output / "result.json")
    expected = worker.implementation_bindings()
    assert request["worker_implementation_bindings"] == result["worker_implementation_bindings"] == expected
    assert request["worker_module"] == result["worker_module"] == worker.WORKER_MODULE
    assert request["plan_sha256"] == sha256(h.plan_path)
    assert not request["training_started"] and not request["automatic_retries"]
    assert not result["training_approved"]
    assert h.closes == [0]
    assert h.captures == [{"profile_geometry_cache": True}]
    assert h.launches == [dict(headless=True, width=1696, height=816, multi_gpu=False,
        renderer="RaytracedLighting", sync_loads=False, disable_viewport_updates=True,
        extra_args=["--/app/settings/persistent=false"])]


def test_main_refuses_stale_loaded_binding_before_app(main_harness, monkeypatch):
    monkeypatch.setattr(worker, "_LOADED_IMPLEMENTATION_BINDINGS", {str(WORKER_PATH): "0" * 64})
    with pytest.raises(ValueError, match="Stale audit/review"):
        main_harness.run()
    assert not main_harness.launches and not main_harness.output.exists()


def test_main_detects_executor_change_during_capture(main_harness, monkeypatch, tmp_path):
    # Change a synthetic binding only; never touch the actual frozen sources.
    binding_file = tmp_path / "cpu-executor-fixture"
    binding_file.write_text("before", encoding="utf-8")
    bindings = {str(binding_file): sha256(binding_file)}
    monkeypatch.setattr(worker, "_LOADED_IMPLEMENTATION_BINDINGS", bindings)
    monkeypatch.setattr(worker, "implementation_bindings", lambda: dict(bindings))
    def capture(*a, **k):
        binding_file.write_text("after", encoding="utf-8")
        return {}
    monkeypatch.setattr(worker, "capture", capture)
    with pytest.raises(ValueError, match="Stale audit/review"):
        main_harness.run()
    assert not (main_harness.output / "result.json").exists()
    assert (main_harness.output / "failure.json").is_file()
    assert main_harness.closes == [1]


@pytest.mark.parametrize("failure", ["memory", "admission", "capture"])
def test_main_fail_closed_no_success_or_retry(main_harness, monkeypatch, failure):
    h = main_harness
    def reject(*a, **k):
        raise ValueError("CPU deliberate failure")
    if failure == "memory":
        h.memory.preflight = lambda: {"allowed": False}
    elif failure == "admission":
        monkeypatch.setattr(worker, "windows_worker_admission", reject)
    else:
        monkeypatch.setattr(worker, "capture", reject)
    with pytest.raises(ValueError):
        h.run()
    assert not (h.output / "result.json").exists()
    assert not read_json(h.output / "failure.json")["training_approved"]
    assert h.closes == ([1] if failure == "capture" else [])
    assert len(h.launches) == (1 if failure == "capture" else 0)


def test_main_refuses_existing_output(main_harness):
    h = main_harness
    h.output.mkdir()
    with pytest.raises(ValueError, match="New output outside sources required"):
        h.run()
    assert not h.launches
