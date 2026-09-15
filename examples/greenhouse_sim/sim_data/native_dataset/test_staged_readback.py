"""CPU contract tests only: never imports Isaac/Warp or launches native work."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from . import staged_readback as sr
from ..native_instance_adapter import LEGACY, FAST

RES = (848, 408)


class FakeArray:
    def __init__(self, value, device="cuda:0"):
        self.value, self.device, self.calls = value, device, 0

    def numpy(self):
        self.calls += 1
        return self.value


WP = SimpleNamespace(array=FakeArray)


class Annotator:
    def __init__(self, name, template_name=None, device="cpu", do_array_copy=True, public_name=None):
        self.source_name = name
        self.name = public_name or name
        self.device = device
        self.do_array_copy = do_array_copy
        self.template_name = (template_name + "buffPtr") if template_name else name


class Writer:
    def attach(self, products, **kwargs):
        self.products = products


REP = SimpleNamespace(Writer=Writer, annotators=SimpleNamespace(Annotator=Annotator))


def payload(resolution=RES):
    w, h = resolution
    rgb = np.arange(h*w*4, dtype=np.uint8).reshape(h, w, 4)
    depth = np.full((h, w), 2.0, np.float32)
    # Bit equality includes NaN payload, +/-zero, +/-Inf and subnormal values.
    depth.view(np.uint32).flat[:6] = [0x7fc00001, 0x80000000, 0, 0x7f800000, 0xff800000, 1]
    ids = np.full((h, w), 2**31 + 17, np.uint32)
    ids.flat[:2] = [0, 1]
    annotation = dict(data=ids, info={"idToLabels": {
        0: "BACKGROUND", 1: "UNLABELLED", 2**31 + 17: "/World/Stem", 8: "/World/Unobserved"}})
    return dict(rgb=rgb, distance_to_image_plane=depth,
        staged_rgb=FakeArray(rgb.copy()), staged_z=FakeArray(depth.copy()),
        camera_params={"renderProductResolution": resolution, "unchanged": np.eye(4)},
        ReferenceTime={"referenceTimeNumerator": 0, "referenceTimeDenominator": 1},
        reference_time=(0, 1), pilot_render_frame={"frame": np.uint64(71)},
        rp_native={"camera": "/World/Robot/Camera", "resolution": resolution},
        **{FAST: deepcopy(annotation), LEGACY: deepcopy(annotation)})


def writer(mode="compare", backend="compare", resolution=RES):
    names = ["rgb", "distance_to_image_plane", "camera_params", "ReferenceTime", "pilot_render_frame", FAST]
    if backend == "compare":
        names.append(LEGACY)
    base = SimpleNamespace(annotators=[Annotator(n) for n in names])
    return sr._new_writer(REP, WP, resolution, backend, mode, base, [])


@pytest.mark.parametrize("resolution", [RES, (1696, 816)])
def test_same_callback_bit_exact_owned_buffers_and_full_ids(resolution):
    source = payload(resolution)
    native = writer(resolution=resolution)
    native.request_index = 7
    native.write(source)
    assert native.sequence == 1 and native.capture_error is None
    record = native.readback_records[0]
    assert record["same_callback_exact"] and record["scope"] == sr.SCOPE
    assert not record["native_qualified"] and not record["training_approved"]
    assert record["request_index"] == 7
    assert native.latest["native_instance_equivalence"]["passed"]
    assert set(native.latest[LEGACY]["info"]["idToLabels"]) == {0, 1, 2**31 + 17}
    assert native.latest["pilot_render_frame"]["frame"] == 71
    for name, (_, alias, _) in sr.SENSORS.items():
        assert source[alias].calls == 1
        assert np.array_equal(sr._bits(native.latest[name]), sr._bits(source[name]))
        assert not np.shares_memory(native.latest[name], source[alias].value)
        assert native.latest[name].flags.owndata and alias not in native.latest
    saved = native.latest["rgb"].copy()
    source["staged_rgb"].value[:] = 0
    source["rgb"][:] = 0
    source[FAST]["data"][:] = 0
    source["camera_params"]["unchanged"][:] = 0
    assert np.array_equal(native.latest["rgb"], saved)
    assert native.latest[LEGACY]["data"][-1, -1] == 2**31 + 17
    assert native.latest["camera_params"]["unchanged"][0, 0] == 1


@pytest.mark.parametrize("name", ["rgb", "distance_to_image_plane"])
@pytest.mark.parametrize("pixel", [0, -1])
def test_single_byte_anywhere_fails_without_publication(name, pixel):
    source = payload()
    alias = sr.SENSORS[name][1]
    source[alias].value.view(np.uint8).reshape(-1)[pixel] ^= 1
    native = writer()
    with pytest.raises(ValueError, match="Same-callback"):
        native.write(source)
    assert native.sequence == 0 and native.latest is None and not native.readback_records
    assert native.capture_error is not None


@pytest.mark.parametrize("mutation", [
    "cpu_missing", "gpu_missing", "numpy_gpu", "cpu_warp", "gpu_dict",
    "rgb_dtype", "z_dtype", "rgb_crop", "z_channel", "empty", "second_product",
    "no_frame", "no_camera", "wrong_resolution", "reference_mismatch",
    "full_id_pixel", "full_id_prim",
])
def test_reject_unverified_or_inconsistent_payload(mutation):
    source = payload()
    if mutation == "cpu_missing": source.pop("rgb")
    elif mutation == "gpu_missing": source.pop("staged_z")
    elif mutation == "numpy_gpu": source["staged_rgb"] = source["rgb"]
    elif mutation == "cpu_warp": source["staged_z"].device = "cpu"
    elif mutation == "gpu_dict": source["staged_z"] = {"data": source["staged_z"]}
    elif mutation == "rgb_dtype": source["staged_rgb"].value = source["rgb"].astype(np.float32)
    elif mutation == "z_dtype": source["staged_z"].value = source["distance_to_image_plane"].astype(np.float16)
    elif mutation == "rgb_crop": source["staged_rgb"].value = source["rgb"][:, :-1]
    elif mutation == "z_channel": source["staged_z"].value = source["distance_to_image_plane"][..., None]
    elif mutation == "empty": source["staged_z"].value = np.empty(0, np.float32)
    elif mutation == "second_product": source["rp_second"] = source["rp_native"]
    elif mutation == "no_frame": source.pop("pilot_render_frame")
    elif mutation == "no_camera": source.pop("camera_params")
    elif mutation == "wrong_resolution": source["rp_native"]["resolution"] = (848, 407)
    elif mutation == "reference_mismatch": source["reference_time"] = (1, 1)
    elif mutation == "full_id_pixel": source[FAST]["data"][-1, -1] = 0
    elif mutation == "full_id_prim": source[FAST]["info"]["idToLabels"][2**31 + 17] = "/World/Wrong"
    with pytest.raises(ValueError):
        writer().write(source)


def test_poisoned_writer_cannot_return_stale_or_recover_silently():
    native = writer()
    native.write(payload())
    broken = payload()
    broken.pop("staged_z")
    with pytest.raises(ValueError):
        native.write(broken)
    assert native.sequence == 1 and native.latest is None
    with pytest.raises(RuntimeError, match="poisoned"):
        native.write(payload())
    assert len(native.readback_records) == 1


@pytest.mark.parametrize("mode", ["compare", "staged"])
def test_distinct_annotator_keys_and_unchanged_cpu_metadata(mode):
    native = writer(mode)
    by_name = {a.name: a for a in native.annotators}
    assert len(by_name) == len(native.annotators)
    for name, (template, alias, _) in sr.SENSORS.items():
        assert by_name[alias].template_name == template + "buffPtr"
        assert by_name[alias].device == "cuda" and not by_name[alias].do_array_copy
        assert (name in by_name) == (mode == "compare")
    for name in ("camera_params", "ReferenceTime", "pilot_render_frame", LEGACY, FAST):
        assert by_name[name].device == "cpu" and by_name[name].do_array_copy
    with pytest.raises(ValueError):
        native.attach(["one", "two"])
    native.attach(["one"])
    assert native.products == ["one"]


def test_staged_trial_no_comparison_claim_or_hidden_cpu_fallback():
    source = payload()
    native = writer("staged")
    with pytest.raises(ValueError, match="CPU mirror"):
        native.write(source)
    for name in sr.SENSORS:
        source.pop(name)
    native = writer("staged")
    native.write(source)
    assert not native.readback_records[0]["same_callback_exact"]
    assert native.readback_records[0]["comparisons"] == {}


def test_default_delegates_without_warp_or_sdk(monkeypatch):
    sentinel = object()
    calls = []
    def original(*args):
        calls.append(args)
        return sentinel
    monkeypatch.setattr(sr, "_cpu_writer", original)
    monkeypatch.setattr(sr, "_runtime_bindings", lambda *a: pytest.fail("CPU loaded CUDA APIs"))
    assert sr.make_staged_writer("rep", (1696, 816), "fast") is sentinel
    assert calls == [("rep", (1696, 816), "fast")]


@pytest.mark.parametrize("mode,resolution,backend", [
    ("unknown", (1696, 816), "fast"), ("compare", RES, "fast"), ("compare", (1696, 816), "legacy"),
])
def test_bad_opt_in_fails_before_warp_import(mode, resolution, backend):
    with pytest.raises(ValueError):
        sr.make_staged_writer(REP, resolution, backend, mode=mode)


def test_staged_public_factory_requires_proof_before_base_factory(monkeypatch):
    monkeypatch.setitem(sys.modules, "warp", WP)
    monkeypatch.setattr(sr, "_runtime_bindings", lambda *a: [])
    monkeypatch.setattr(sr, "implementation_bindings", lambda: [])
    monkeypatch.setattr(sr, "_cpu_writer", lambda *a: pytest.fail("must reject before creating writer"))
    with pytest.raises(ValueError, match="Pinned same-callback"):
        sr.make_staged_writer(REP, (1696, 816), "fast", mode="staged")


def test_sdk_pin_checker_is_read_only_and_rejects_mutation(tmp_path, monkeypatch):
    path = tmp_path / "fake_sdk.py"
    path.write_text("# synthetic API fixture\n", encoding="utf-8")
    expected = sr.file_binding(path)["sha256"]
    monkeypatch.setattr(sr, "SDK_PINS", {"fake_sdk.py": expected})
    assert sr.verify_local_api(tmp_path) == [sr.file_binding(path)]
    path.write_text("# mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unverified"):
        sr.verify_local_api(tmp_path)


def comparison_receipt(tmp_path):
    native = writer()
    for _ in range(3):
        native.write(payload())
    pins = []
    for name in ("request.json", "result.json", "a_sample.json", "b_sample.json", "c_sample.json"):
        p = tmp_path / name
        p.write_text("{}", encoding="utf-8")
        pins.append(sr.file_binding(p))
    proof = dict(schema=sr.SCHEMA, mode="compare", state="diagnostic_complete",
        native_qualified=False, bindings=[], resolution=list(RES), instance_backend="compare",
        render_budget="reference56", callbacks=native.readback_records, artifact_bindings=pins,
        captured_comparisons=[dict(callback_sequence=c["callback_sequence"],
            rgb_sha256=c["rgb_sha256"], depth_sha256=c["depth_sha256"], camera_sha256=str(i)*64)
            for i, c in enumerate(native.readback_records)])
    return proof


@pytest.mark.parametrize("mutation", [
    None, "hash", "mode", "state", "bindings", "qualified", "too_few", "same_camera",
    "scope", "mismatch", "nan_digest", "duplicate_sequence", "capture_digest",
    "mutated_artifact", "missing_pins", "short_budget", "partial_bytes", "partial_shape",
    "missing_camera_evidence", "duplicate_pins", "wrong_callback_backend",
])
def test_explicit_compare_receipt_gate(tmp_path, mutation):
    proof = comparison_receipt(tmp_path)
    if mutation == "mode": proof["mode"] = "staged"
    elif mutation == "state": proof["state"] = "diagnostic_failed"
    elif mutation == "bindings": proof["bindings"] = [{"sha256": "different"}]
    elif mutation == "qualified": proof["native_qualified"] = True
    elif mutation == "too_few": proof["captured_comparisons"].pop()
    elif mutation == "same_camera":
        for c in proof["captured_comparisons"]: c["camera_sha256"] = "0"*64
    elif mutation == "scope": proof["callbacks"][0]["scope"] = "different_renderer_run"
    elif mutation == "mismatch": proof["callbacks"][0]["comparisons"]["rgb"]["differing_bytes"] = 1
    elif mutation == "nan_digest": proof["callbacks"][0]["comparisons"]["rgb"]["cpu_sha256"] = "NaN"
    elif mutation == "duplicate_sequence": proof["callbacks"][0]["callback_sequence"] = 2
    elif mutation == "capture_digest": proof["captured_comparisons"][0]["depth_sha256"] = "f"*64
    elif mutation == "mutated_artifact":
        Path(proof["artifact_bindings"][0]["path"]).write_text("changed", encoding="utf-8")
    elif mutation == "missing_pins": proof["artifact_bindings"] = []
    elif mutation == "short_budget": proof["render_budget"] = "warm56_then8_trial"
    elif mutation == "partial_bytes": proof["callbacks"][0]["comparisons"]["rgb"]["compared_bytes"] = 1
    elif mutation == "partial_shape": proof["callbacks"][0]["comparisons"]["rgb"]["shape"] = [1, 1, 4]
    elif mutation == "missing_camera_evidence": proof["callbacks"][0].pop("camera_params_sha256")
    elif mutation == "duplicate_pins": proof["artifact_bindings"][1] = proof["artifact_bindings"][0]
    elif mutation == "wrong_callback_backend": proof["callbacks"][0]["instance_backend"] = "fast"
    p = tmp_path / "proof.json"
    p.write_text(json.dumps(proof), encoding="utf-8")
    sha = sr.file_binding(p)["sha256"] if mutation != "hash" else "0"*64
    if mutation is None:
        assert sr.require_compare_receipt(p, sha, [], RES, "compare") == sr.file_binding(p)
    else:
        with pytest.raises(ValueError):
            sr.require_compare_receipt(p, sha, [], RES, "compare")


@pytest.fixture
def wrapper():
    root = Path(__file__).resolve().parents[4]
    path = root / "data/sim_data/diagnostics/staged_readback_wrapper_20260916_v1.py"
    if not path.exists():
        pytest.skip("Ignored diagnostic wrapper is not distributed with the library")
    spec = importlib.util.spec_from_file_location("isolated_readback_wrapper", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrapper_factory_restored_after_error(wrapper):
    from .. import native_instance_adapter
    original = native_instance_adapter.make_native_writer
    with pytest.raises(RuntimeError, match="synthetic"):
        with wrapper.installed_factory("compare", None, []):
            assert native_instance_adapter.make_native_writer is not original
            raise RuntimeError("synthetic")
    assert native_instance_adapter.make_native_writer is original


def test_wrapper_cpu_delegates_and_restores_argv(wrapper, monkeypatch):
    from .. import native_generated_views
    original = sys.argv
    calls = []
    monkeypatch.setattr(native_generated_views, "main", lambda: calls.append(sys.argv.copy()))
    wrapper.main(["--plan", "fixture", "--output", "fixture_out"])
    assert calls[0][1:] == ["--plan", "fixture", "--output", "fixture_out"]
    assert sys.argv is original


def test_wrapper_rejects_non_diagnostic_before_launch(wrapper, monkeypatch):
    from .. import native_generated_views
    monkeypatch.setattr(native_generated_views, "main", lambda: pytest.fail("native launch"))
    with pytest.raises(ValueError, match="New diagnostic"):
        wrapper.main(["--readback-mode", "compare", "--instance-backend", "fast",
                      "--output", str(wrapper.ROOT / "never_run")])


def test_transfer_exception_stops_callback_and_no_reference_reuse():
    source = payload()
    def fail():
        raise RuntimeError("synthetic D2H error")
    source["staged_z"].numpy = fail
    native = writer()
    with pytest.raises(RuntimeError, match="D2H"):
        native.write(source)
    assert native.sequence == 0 and native.latest is None and native.capture_error


@pytest.mark.parametrize("mismatch", [None, "rgb", "frame", "camera", "sample_hash"])
def test_wrapper_capture_binding_without_images(wrapper, tmp_path, mismatch):
    from ..capture_contract import jsonable
    source = payload()
    native = writer()
    native.write(source)
    callback = native.readback_records[0]
    sample = dict(calibration={"camera_path": callback["camera"]},
        rendered_camera_params=jsonable(source["camera_params"]),
        synchronization=dict(native_render_frame=jsonable(source["pilot_render_frame"]),
            freshness=dict(callback_sequence=1, camera_sha256="a"*64,
                           rgb_sha256=callback["rgb_sha256"], depth_sha256=callback["depth_sha256"])))
    if mismatch == "rgb": sample["synchronization"]["freshness"]["rgb_sha256"] = "f"*64
    if mismatch == "frame": sample["synchronization"]["native_render_frame"]["frame"] = 72
    if mismatch == "camera": sample["rendered_camera_params"]["unchanged"][0][0] = 99
    folder = tmp_path / "capture" / "one"
    folder.mkdir(parents=True)
    sample_path = folder / "sample.json"
    sample_path.write_text(json.dumps(sample), encoding="utf-8")
    sample_hash = sr.file_binding(sample_path)["sha256"]
    if mismatch == "sample_hash": sample_hash = "f"*64
    output = folder.parent
    (output / "request.json").write_text("{}", encoding="utf-8")
    (output / "result.json").write_text(json.dumps(dict(
        state="native_generated_multiview_pilot_complete_pending_review", captured_frames=1,
        records=[dict(state="native_captured_pending_review", candidate_id="one", sample_sha256=sample_hash)])),
        encoding="utf-8")
    if mismatch is not None:
        with pytest.raises(ValueError):
            wrapper.captured_evidence(output, [callback])
    else:
        pins, captures = wrapper.captured_evidence(output, [callback])
        assert len(pins) == 3 and captures[0]["callback_sequence"] == 1
        assert captures[0]["depth_sha256"] == callback["depth_sha256"]
    assert not list(output.rglob("*.png")) and not list(output.rglob("*.npy"))


def test_wrapper_failure_receipt_create_only_no_native(wrapper, tmp_path, monkeypatch):
    from .. import native_generated_views, native_instance_adapter
    monkeypatch.setattr(wrapper, "ROOT", tmp_path)
    diagnostics = tmp_path / "data/sim_data/diagnostics"
    diagnostics.mkdir(parents=True)
    receipt = diagnostics / "receipt.json"
    monkeypatch.setattr(sr, "verify_local_api", lambda *a: [])
    monkeypatch.setattr(sr, "implementation_bindings", lambda: [])
    calls = []
    def fail_before_native():
        calls.append(True)
        raise RuntimeError("synthetic pre-native failure")
    monkeypatch.setattr(native_generated_views, "main", fail_before_native)
    original = native_instance_adapter.make_native_writer
    args = ["--readback-mode", "compare", "--instance-backend", "fast",
            "--readback-receipt", str(receipt), "--output", str(diagnostics / "never_created")]
    with pytest.raises(RuntimeError, match="pre-native"):
        wrapper.main(args)
    assert native_instance_adapter.make_native_writer is original
    saved = receipt.read_bytes()
    report = json.loads(saved)
    assert report["state"] == "diagnostic_failed" and not report["native_qualified"]
    assert not report["counter_verified"] and report["callbacks"] == []
    with pytest.raises(ValueError, match="create-only"):
        wrapper.main(args)
    assert receipt.read_bytes() == saved and len(calls) == 1
