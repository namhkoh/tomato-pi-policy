"""Pure tests: no Kit, PhysX binding import, SimulationApp, or native cooking."""
import asyncio
import copy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import sys
import subprocess

import numpy as np
import pytest

from sim_physics import cooked_geometry_probe as probe


def tetra(shift=0.0):
    points = [[shift, 0, 0], [1 + shift, 0, 0], [shift, 1, 0], [shift, 0, 1]]
    faces = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]
    planes = [(0, 0, -1, 0), (0, -1, 0, 0), (-1, 0, 0, shift), (1, 1, 1, -1-shift)]
    return SimpleNamespace(
        vertices=[SimpleNamespace(x=x, y=y, z=z) for x, y, z in points],
        indices=[i for face in faces for i in face],
        polygons=[SimpleNamespace(index_base=3*i, num_vertices=3, plane=plane)
                  for i, plane in enumerate(planes)])


class Cooking:
    def __init__(self):
        self.requests = []
        self.cancelled = []

    def request_convex_collision_representation(self, **kwargs):
        self.requests.append(kwargs)
        return len(self.requests)

    def cancel_collision_representation_task(self, **kwargs):
        self.cancelled.append(kwargs)


def run_request(cooking, update, guard=lambda: None, **kwargs):
    return asyncio.run(probe.request_convexes(
        cooking, 42, {"/a": 11, "/b": 12}, update, guard, valid_result="VALID", **kwargs))


@pytest.mark.parametrize("parent_has_native_modules", [False, True])
def test_import_does_not_load_native_runtime(monkeypatch, parent_has_native_modules):
    # Other tests may legitimately import native packages in the parent process.
    # Isolate this import, and reject any attempted runtime import in the child.
    if parent_has_native_modules:
        monkeypatch.setitem(sys.modules, "omni.physx", SimpleNamespace())
        monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace())
    script = """
import importlib.abc
import sys
class NoNativeRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'omni', 'isaacsim', 'carb'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
before = set(sys.modules)
sys.meta_path.insert(0, NoNativeRuntime())
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[2])  # Isaac's isolated Python omits its bundled numpy path.
import sim_physics.cooked_geometry_probe
added = set(sys.modules) - before
assert not any(name.split('.')[0] in {'omni', 'isaacsim', 'carb'} for name in added)
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(Path(probe.__file__).resolve().parents[1]),
         str(Path(np.__file__).resolve().parents[1])],
        capture_output=True, text=True, timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_copies_native_buffers_and_polygon_index_spans():
    native = tetra()
    packed = probe.pack_convex(native)
    assert packed["indices"] == [0, 2, 1, 0, 1, 3, 0, 3, 2, 1, 2, 3]
    assert packed["polygons"][3]["index_base"] == 9
    assert packed["polygons"][3]["plane_local_raw"] == [1, 1, 1, -1]
    before = copy.deepcopy(packed)
    native.vertices[0].x = 100
    native.indices[0] = 3
    assert packed == before


@pytest.mark.parametrize("defect", [
    "nan", "flat", "out_of_range", "negative_span", "oversize_span",
    "bad_plane", "repeated_face_vertex", "open", "fractional_index",
])
def test_invalid_native_geometry_fails_closed(defect):
    native = tetra()
    if defect == "nan":
        native.vertices[0].x = float("nan")
    elif defect == "flat":
        native.vertices[3].z = 0
    elif defect == "out_of_range":
        native.indices[0] = 99
    elif defect == "negative_span":
        native.polygons[0].index_base = -1
    elif defect == "oversize_span":
        native.polygons[0].num_vertices = 100
    elif defect == "bad_plane":
        native.polygons[0].plane = [0, 0, 0, 0]
    elif defect == "repeated_face_vertex":
        native.indices[0] = native.indices[1]
    elif defect == "open":
        native.polygons.pop()
    elif defect == "fractional_index":
        native.indices[0] = 0.5
    with pytest.raises((ValueError, TypeError)):
        probe.pack_convex(native)


def test_transform_keeps_local_rotation_scale_translation_and_stage_units():
    matrix = [[0, -3, 0, 10], [2, 0, 0, 20], [0, 0, 4, 30], [0, 0, 0, 1]]
    actual = probe.world_vertices([[1, 2, 3]], matrix, .01)
    assert actual[0] == pytest.approx([.04, .22, .42])


@pytest.mark.parametrize("units", [0, -1, float("nan")])
def test_invalid_units_rejected(units):
    with pytest.raises(ValueError):
        probe.world_vertices([[0, 0, 0]], np.eye(4), units)


def test_singular_transform_rejected():
    with pytest.raises(ValueError):
        probe.world_vertices([[0, 0, 0]], np.zeros((4, 4)), 1)


def test_exact_target_selection_excludes_other_plants_and_detects_ambiguity():
    bracket = "/World/RBY1/ee_right/attachments/RightWristCamera/BracketCollision"
    stem26 = "/World/Plant/MainStem_00/MainStem_26/MainStem_26/MainStem_26"
    stem27 = "/World/Plant/MainStem_00/MainStem_27/MainStem_27/MainStem_27"
    paths = [bracket, stem26, stem27, stem26.replace("/Plant/", "/OtherPlant/")]
    assert probe.select_targets(paths) == [bracket, stem26, stem27]
    with pytest.raises(ValueError, match="found 0"):
        probe.select_targets(paths[:2])
    with pytest.raises(ValueError, match="found 2"):
        probe.select_targets(paths + [stem26.replace("/MainStem_00/", "/DifferentBranch/")])


BRACKET = "/World/RBY1/ee_right/attachments/RightWristCamera/BracketCollision"
ADAPTER = "/World/RBY1/ee_right/attachments/RightWristCamera/AdapterCollision"
STEM26 = "/World/Plant/MainStem_00/MainStem_26/MainStem_26/MainStem_26"
STEM27 = "/World/Plant/MainStem_00/MainStem_27/MainStem_27/MainStem_27"


def test_explicit_selection_is_ordered_copied_and_can_include_adapter():
    requested = [ADAPTER, STEM27, BRACKET, STEM26]
    selected = probe._explicit_targets(requested, "/World/RBY1", "/World/Plant")
    assert selected == requested and selected is not requested
    requested.clear()
    assert selected == [ADAPTER, STEM27, BRACKET, STEM26]
    assert probe._explicit_targets((ADAPTER,), "/World/RBY1", "/World/Plant") == [ADAPTER]


@pytest.mark.parametrize("paths", [
    [], (), ADAPTER, {ADAPTER}, {"path": ADAPTER}, [None], [1], [Path(ADAPTER)],
    [""], ["World/RBY1/Mesh"], ["/"], ["/World"], ["/World/RBY1"], ["/World/Plant"],
    ["/World/RBY10/Mesh"], ["/World/PlantBackup/Mesh"], ["/Elsewhere/Mesh"],
    ["/World/RBY1/../Plant/Mesh"], ["/World//RBY1/Mesh"], [ADAPTER + "/"],
    [ADAPTER + ".points"], ["/World/RBY1{variant=one}/Mesh"],
    ["/World/RBY1/*"], ["/World/RBY1/**"], [ADAPTER + " "],
    [ADAPTER.replace("/", "\\")], [ADAPTER, ADAPTER],
])
def test_invalid_explicit_selection_fails_before_runtime_imports(tmp_path, monkeypatch, paths):
    # No Kit module can be imported even if another test has loaded it.
    monkeypatch.setitem(sys.modules, "carb", None)
    output = tmp_path / "not_created.json"
    with pytest.raises(ValueError):
        asyncio.run(probe.capture_current_stage(output, collider_paths=paths))
    assert not output.exists()


@pytest.mark.parametrize("robot_root,plant_root", [
    ("/", "/World/Plant"), ("/World", "/World/Plant"),
    ("/World/RBY1", "/World"), ("World/RBY1", "/World/Plant"),
    ("/World/RBY1/", "/World/Plant"), ("/World/RBY1", "/World/RBY1"),
    ("/World/RBY1", "/World/RBY1/Plant"), ("/World/Plant/Robot", "/World/Plant"),
    ("/World/RBY1.points", "/World/Plant"), (None, "/World/Plant"),
])
def test_explicit_selection_rejects_broad_invalid_or_overlapping_roots(robot_root, plant_root):
    with pytest.raises(ValueError):
        probe._explicit_targets([ADAPTER], robot_root, plant_root)


def test_explicit_selection_uses_the_specified_roots_not_default_prefixes():
    adapter = ADAPTER.replace("/World/RBY1", "/Diagnostic/Robot")
    stem = STEM26.replace("/World/Plant", "/Diagnostic/Plant")
    assert probe._explicit_targets([adapter, stem], "/Diagnostic/Robot", "/Diagnostic/Plant") == [adapter, stem]
    with pytest.raises(ValueError, match="strictly under"):
        probe._explicit_targets([ADAPTER], "/Diagnostic/Robot", "/Diagnostic/Plant")


def collision_mesh(stage, path, approximation="convexDecomposition"):
    """Small in-memory USD fixture, never a source asset or native actor."""
    from pxr import UsdGeom, UsdPhysics
    mesh = UsdGeom.Mesh.Define(stage, path)
    native = tetra()
    mesh.CreatePointsAttr([(v.x, v.y, v.z) for v in native.vertices])
    mesh.CreateFaceVertexCountsAttr([3] * 4)
    mesh.CreateFaceVertexIndicesAttr(native.indices)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr(approximation)
    return mesh.GetPrim()


@pytest.fixture
def capture_runtime(monkeypatch):
    """Mock Kit/PhysX orchestration around CPU-only, in-memory USD snapshots."""
    import pxr
    from pxr import Usd, UsdUtils
    stage = Usd.Stage.CreateInMemory()
    cache = UsdUtils.StageCache.Get()
    cache.Insert(stage)
    cooking = Cooking()

    async def update():
        for request in cooking.requests:
            request["on_result"]("VALID", [tetra()])

    def module(name, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, value)
        if "." in name:
            parent, child = name.rsplit(".", 1)
            monkeypatch.setattr(sys.modules[parent], child, value, raising=False)
        return value

    module("carb")
    module("carb.settings", get_settings=lambda: SimpleNamespace(get=lambda key: None))
    module("omni")
    module("omni.kit")
    module("omni.kit.app", get_app=lambda: SimpleNamespace(next_update_async=update))
    module("omni.timeline", get_timeline_interface=lambda: SimpleNamespace(is_stopped=lambda: True))
    module("omni.usd", get_context=lambda: SimpleNamespace(get_stage=lambda: stage))
    module("omni.physx", get_physx_cooking_interface=lambda: cooking)
    module("omni.physx.bindings")
    module("omni.physx.bindings._physx", __file__=probe.__file__,
           PhysxCollisionRepresentationResult=SimpleNamespace(RESULT_VALID="VALID"))
    ids = {}

    def prim_id(path):
        return ids.setdefault(str(path), len(ids) + 1)

    monkeypatch.setattr(pxr, "PhysicsSchemaTools", SimpleNamespace(sdfPathToInt=prim_id), raising=False)
    try:
        yield SimpleNamespace(stage=stage, cooking=cooking, ids=ids)
    finally:
        cache.Erase(stage)


def test_explicit_capture_never_traverses_or_substitutes_and_reports_exact_scope(
        capture_runtime, monkeypatch, tmp_path):
    from pxr import Usd
    runtime = capture_runtime
    # Default targets do not exist; only the explicitly requested adapter does.
    collision_mesh(runtime.stage, ADAPTER)
    before = runtime.stage.GetRootLayer().ExportToString()

    def forbidden(*args, **kwargs):
        raise AssertionError("Explicit selection must not walk or fall back")

    monkeypatch.setattr(Usd, "PrimRange", SimpleNamespace(Stage=forbidden))
    monkeypatch.setattr(probe, "select_targets", forbidden)
    output = tmp_path / "adapter.json"
    report = asyncio.run(probe.capture_current_stage(output, collider_paths=[ADAPTER]))
    assert report["status"] == "captured_advisory"
    assert report["requested_scope"] == {
        "selection_mode": "explicit_collider_paths",
        "robot_root": "/World/RBY1", "plant_root": "/World/Plant",
        "requested_collider_paths": [ADAPTER], "selected_collider_paths": [ADAPTER],
        "stage_traversal_for_selection": False, "scope_expanded": False,
    }
    assert list(report["colliders"]) == list(report["source"]["meshes"]) == [ADAPTER]
    assert len(runtime.cooking.requests) == 1
    assert runtime.stage.GetRootLayer().ExportToString() == before
    assert report["source_unchanged"] and report["settings_unchanged"]
    assert not report["eligible_to_replace_screen"]
    payload_hash = report.pop("payload_sha256")
    assert probe.fingerprint(report) == payload_hash
    assert output.exists()


def test_default_capture_still_selects_exactly_three_and_does_not_add_adapter(capture_runtime, tmp_path):
    runtime = capture_runtime
    for path in (BRACKET, STEM26, STEM27, ADAPTER):
        collision_mesh(runtime.stage, path)
    report = asyncio.run(probe.capture_current_stage(tmp_path / "default.json"))
    assert report["status"] == "captured_advisory"
    assert list(report["colliders"]) == [BRACKET, STEM26, STEM27]
    assert report["requested_scope"]["selection_mode"] == "legacy_three_targets"
    assert report["requested_scope"]["requested_collider_paths"] is None
    assert report["requested_scope"]["selected_collider_paths"] == [BRACKET, STEM26, STEM27]
    assert report["requested_scope"]["stage_traversal_for_selection"] is True
    assert len(runtime.cooking.requests) == 3


def test_explicit_capture_custom_roots_multiple_paths_and_input_mutation(capture_runtime, tmp_path):
    runtime = capture_runtime
    adapter = ADAPTER.replace("/World/RBY1", "/Diagnostic/Robot")
    stem = STEM26.replace("/World/Plant", "/Diagnostic/Plant")
    expected = [stem, adapter]
    requested = list(expected)
    collision_mesh(runtime.stage, adapter)
    collision_mesh(runtime.stage, stem, "convexHull")
    request = runtime.cooking.request_convex_collision_representation

    def mutate_caller_list(**kwargs):
        requested[:] = ["/World/Unexpected"]
        return request(**kwargs)

    runtime.cooking.request_convex_collision_representation = mutate_caller_list
    report = asyncio.run(probe.capture_current_stage(
        tmp_path / "custom.json", robot_root="/Diagnostic/Robot", plant_root="/Diagnostic/Plant",
        collider_paths=requested))
    assert report["status"] == "captured_advisory"
    assert list(report["colliders"]) == expected
    assert report["requested_scope"]["requested_collider_paths"] == expected
    assert report["requested_scope"]["selected_collider_paths"] == expected
    assert list(runtime.ids) == expected


@pytest.mark.parametrize("defect", [
    "missing", "inactive", "non_mesh", "no_collision", "disabled",
    "no_mesh_collision", "triangle_mesh", "broad_assembly",
])
def test_explicit_capture_keeps_existing_snapshot_validation_and_starts_no_cooking(
        capture_runtime, tmp_path, defect):
    from pxr import UsdGeom, UsdPhysics
    runtime = capture_runtime
    path = ADAPTER
    if defect == "broad_assembly":
        collision_mesh(runtime.stage, ADAPTER)
        path = ADAPTER.rsplit("/", 1)[0]
    elif defect == "non_mesh":
        prim = UsdGeom.Cube.Define(runtime.stage, path).GetPrim()
        UsdPhysics.CollisionAPI.Apply(prim)
    elif defect != "missing":
        prim = collision_mesh(runtime.stage, path)
        if defect == "inactive":
            prim.SetActive(False)
        elif defect == "no_collision":
            prim.RemoveAPI(UsdPhysics.CollisionAPI)
        elif defect == "disabled":
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        elif defect == "no_mesh_collision":
            prim.RemoveAPI(UsdPhysics.MeshCollisionAPI)
            prim.RemoveProperty("physics:approximation")
        elif defect == "triangle_mesh":
            UsdPhysics.MeshCollisionAPI(prim).CreateApproximationAttr("none")
    output = tmp_path / "invalid.json"
    with pytest.raises(ValueError, match="collision mesh|convex approximation"):
        asyncio.run(probe.capture_current_stage(output, collider_paths=[path]))
    assert runtime.cooking.requests == []
    assert not output.exists()


def test_one_missing_explicit_collider_blocks_entire_request(capture_runtime, tmp_path):
    collision_mesh(capture_runtime.stage, ADAPTER)
    with pytest.raises(ValueError, match="collision mesh"):
        asyncio.run(probe.capture_current_stage(
            tmp_path / "not_partial.json", collider_paths=[ADAPTER, STEM26]))
    assert capture_runtime.cooking.requests == []
    assert not (tmp_path / "not_partial.json").exists()


def test_source_hash_binds_geometry_transform_and_cooking_settings():
    original = {"points": [[0, 0, 0]], "scale": [1, 1, 1], "approximation": "convexDecomposition"}
    baseline = probe.fingerprint(original)
    assert probe.fingerprint(dict(reversed(list(original.items())))) == baseline
    for key, value in [("points", [[0, 0, .001]]), ("scale", [1, 1, 2]),
                       ("approximation", "convexHull")]:
        assert probe.fingerprint(dict(original, **{key: value})) != baseline


def test_async_api_keywords_and_stable_part_order():
    cooking = Cooking()

    async def update():
        for i, request in enumerate(cooking.requests):
            parts = [tetra(), tetra(2)]
            request["on_result"]("VALID", parts if i == 0 else list(reversed(parts)))

    rows = run_request(cooking, update)
    assert rows["/a"]["status"] == rows["/b"]["status"] == "valid"
    assert rows["/a"]["convexes"] == rows["/b"]["convexes"]
    assert cooking.cancelled == []
    assert [(r["stage_id"], r["collision_prim_id"], r["run_asynchronously"])
            for r in cooking.requests] == [(42, 11, True), (42, 12, True)]


@pytest.mark.parametrize("result,data,status", [
    ("COOKING_FAILED", [], "native_error"), ("VALID", [], "invalid"),
    ("VALID", [SimpleNamespace(vertices=[])], "invalid"),
])
def test_native_errors_and_empty_result_do_not_pass(result, data, status):
    cooking = Cooking()

    async def update():
        for request in cooking.requests:
            request["on_result"](result, data)

    rows = run_request(cooking, update)
    assert all(row["status"] == status for row in rows.values())


def test_duplicate_callback_is_invalid():
    cooking = Cooking()

    async def update():
        for request in cooking.requests:
            request["on_result"]("VALID", [tetra()])
            request["on_result"]("VALID", [tetra()])

    assert all(row["status"] == "invalid" for row in run_request(cooking, update).values())


def test_timeout_cancels_only_own_pending_requests_and_ignores_late_callbacks():
    cooking = Cooking()

    async def update():
        cooking.requests[0]["on_result"]("VALID", [tetra()])
        await asyncio.sleep(1)

    rows = run_request(cooking, update, timeout_s=.01)
    assert rows["/a"]["status"] == "valid"
    assert rows["/b"]["status"] == "timeout"
    assert cooking.cancelled == [{"task": 2, "invoke_callback": False}]
    cooking.requests[1]["on_result"]("VALID", [tetra()])
    assert rows["/b"]["status"] == "timeout"


def test_stage_or_timeline_guard_failure_cancels_pending_without_changing_stage():
    cooking = Cooking()
    checks = []

    def guard():
        checks.append(True)
        if len(checks) == 3:
            raise RuntimeError("stage changed")

    async def update():
        raise AssertionError("Must not service another update")

    with pytest.raises(RuntimeError, match="stage changed"):
        run_request(cooking, update, guard)
    assert cooking.cancelled == [{"task": 1, "invoke_callback": False},
                                 {"task": 2, "invoke_callback": False}]


def test_caller_cancellation_cancels_native_tasks():
    cooking = Cooking()

    async def scenario():
        entered = asyncio.Event()

        async def update():
            entered.set()
            await asyncio.sleep(10)

        task = asyncio.create_task(probe.request_convexes(
            cooking, 42, {"/a": 11}, update, lambda: None,
            valid_result="VALID", timeout_s=1))
        await entered.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())
    assert cooking.cancelled == [{"task": 1, "invoke_callback": False}]


def test_cancellation_failure_still_attempts_every_owned_pending_task():
    class FailingCancel(Cooking):
        def cancel_collision_representation_task(self, **kwargs):
            super().cancel_collision_representation_task(**kwargs)
            if kwargs["task"] == 1:
                raise RuntimeError("native cancellation fault")

    cooking = FailingCancel()

    async def update():
        await asyncio.sleep(1)

    with pytest.raises(RuntimeError, match="cancellation fault"):
        run_request(cooking, update, timeout_s=.01)
    assert cooking.cancelled == [{"task": 1, "invoke_callback": False},
                                 {"task": 2, "invoke_callback": False}]


def test_empty_request_cannot_be_mistaken_for_complete_capture():
    async def update():
        raise AssertionError

    with pytest.raises(ValueError, match="explicit collision prim"):
        asyncio.run(probe.request_convexes(
            Cooking(), 42, {}, update, lambda: None, valid_result="VALID"))


@pytest.mark.parametrize("timeout", [0, -1, 91, float("inf")])
def test_invalid_timeout_starts_no_requests(timeout):
    cooking = Cooking()

    async def update():
        raise AssertionError

    with pytest.raises(ValueError):
        run_request(cooking, update, timeout_s=timeout)
    assert cooking.requests == []


def test_report_creation_is_exclusive_and_hashes_file(tmp_path):
    report = tmp_path / "capture.json"
    probe.write_new_report(report, {"status": "advisory"})
    assert len(probe._file_hash(report)) == 64
    before = report.read_bytes()
    with pytest.raises(FileExistsError):
        probe.write_new_report(report, {"status": "overwrite"})
    assert report.read_bytes() == before
    with pytest.raises(ValueError):
        probe.write_new_report(tmp_path / "missing" / "capture.json", {})
    with pytest.raises(ValueError):
        probe.write_new_report(tmp_path / "nan.json", {"x": float("nan")})
    assert not Path(tmp_path / "nan.json").exists()
