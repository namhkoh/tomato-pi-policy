"""Pure tests: no Kit, PhysX binding import, SimulationApp, or native cooking."""
import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace
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
