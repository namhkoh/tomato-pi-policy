"""Synthetic callback storage tests, NOT native-capture qualification or approval.

All observations here are fabricated unit-test buffers in temporary directories.
Even an 'eligible' fixture label exercises serialization only, never eligibility
of real data. No simulator, collector, training job or existing capture is used.
"""
from copy import deepcopy
import io
import json
from pathlib import Path

import numpy as np
import pytest

from ..capture_contract import write_sample
from ..capture_sensor import HIRES_RESOLUTION, LEGACY_RESOLUTION
from ..capture_visibility import component_masks, write_visibility
from ..dataset_review import write_json
from ..native_sensor_payload import decode_native_instances, validate_native_static
from . import capture_storage as storage
from .bundle import ARRAYS, EXTRAS, REQUIRED, SampleReader, digest


def synthetic_callback(resolution=HIRES_RESOLUTION):
    """Exercise the real CPU callback checks with explicitly synthetic evidence."""
    width, height = resolution
    calibration = dict(resolution=list(resolution), camera_path="/World/Synthetic/Camera",
        intrinsics=[[width, 0, width / 2], [0, height, height / 2], [0, 0, 1]],
        camera_to_world_usd_row_vectors=np.eye(4).tolist(), clipping_range_m=[.01, 100],
        crop_resize=None, depth_convention="optical_axis_z_metres_not_ray_range")
    projection = np.zeros((4, 4))
    projection[0, 0] = projection[1, 1] = 2
    projection[2, 2], projection[2, 3], projection[3, 2] = -1.0002, -1, -.02
    rgba = np.full((height, width, 4), 255, np.uint8)
    rgba[:, :, 0] = np.arange(width, dtype=np.uint16) % 256
    rgba[:, :, 1] = (np.arange(height, dtype=np.uint16) % 256)[:, None]
    depth = np.full((height, width), 2, np.float32)
    # Distinct quiet/signalling NaN payloads, signed zeros, infinities and ULPs.
    depth.view(np.uint32)[0, :9] = [0x7FC12345, 0x7FA54321, 0xFFC12345,
        0x7F800000, 0xFF800000, 0x80000000, 0, 0x3F800001, 0x3F7FFFFF]
    ids = np.zeros((height, width), np.uint32)
    ids[20:60, 20:100] = 0xFFFFFFFF
    ids[30:40, 30:50] = 5
    payload = dict(rgb=rgba, distance_to_image_plane=depth,
        ReferenceTime=dict(referenceTimeNumerator=10, referenceTimeDenominator=60),
        reference_time=[10, 60], rp_head=dict(camera=calibration["camera_path"], resolution=list(resolution)),
        camera_params=dict(renderProductResolution=list(resolution), metersPerSceneUnit=1.,
            cameraViewTransform=np.eye(4).reshape(-1), cameraProjection=projection.reshape(-1)),
        instance_id_segmentation=dict(data=ids, info=dict(idToLabels={
            "0": "BACKGROUND", "4294967295": "/World/Synthetic/Stem",
            "5": "/World/Synthetic/Stem/Leaf"})))
    rgb, depth, valid, reference, freshness = validate_native_static(
        payload, calibration, "a" * 64, "a" * 64, 1)
    instances, mapping = decode_native_instances(payload, resolution)
    catalogue = [dict(variant_id="synthetic", component_id="Stem", component_index=0xFFFFFFFE,
                     organ_type="sub_stem", prim_path="/World/Synthetic/Stem"),
                 dict(variant_id="synthetic", component_id="Leaf", component_index=8,
                     organ_type="leaf", prim_path="/World/Synthetic/Stem/Leaf")]
    components, organs, _ = component_masks(instances, mapping, catalogue)
    metadata = dict(state="synthetic_callback_fixture_not_native_qualification",
        training_sample_approved=False, calibration=calibration,
        synchronization=dict(method="frozen_scene_single_native_writer_payload",
            scene_unchanged_during_capture=True, dynamic_recording_supported=False,
            freshness=freshness, reference_time=reference, static_guard="a" * 64),
        supervision=dict(target_id="synthetic/Stem", projected_interval=[],
            nominal_projected=dict(projection_status="out_of_frame"),
            depth_evidence=dict(status="synthetic_storage_fixture_only")))
    label = dict(eligible=True, training_approved=False, fixture_only=True,
        reason="synthetic_callback_storage_test_not_native_qualification")
    return dict(rgb=rgb, depth=depth, valid=valid, metadata=metadata, instances=instances,
        mapping=mapping, catalogue=catalogue, components=components, organs=organs,
        target_mask=components == 0xFFFFFFFE, label=label,
        trace=dict(passed=False, reasons=["synthetic_fixture_not_native_qualification"]))


@pytest.fixture
def callback():
    return synthetic_callback()


def npy_bytes(array):
    stream = io.BytesIO()
    np.save(stream, array, allow_pickle=False)
    return stream.getvalue()


def snapshot(directory):
    return {p.relative_to(directory).as_posix(): p.read_bytes()
            for p in directory.rglob("*") if p.is_file()}


@pytest.mark.parametrize("resolution", [LEGACY_RESOLUTION, HIRES_RESOLUTION])
def test_exact_native_bytes_match_original_writers_without_review_pngs(tmp_path, resolution):
    callback = synthetic_callback(resolution)
    source, compact = tmp_path / "synthetic_raw", tmp_path / "synthetic_compact"
    write_sample(source, callback["rgb"], callback["depth"], callback["valid"], deepcopy(callback["metadata"]))
    write_visibility(source, *(callback[k] for k in
        ("instances", "mapping", "catalogue", "components", "organs", "target_mask")))
    write_json(source / "supervision/label.json", callback["label"])
    write_json(source / "supervision/query_trace.json", callback["trace"])
    before = snapshot(source)
    raw = SampleReader(source, expected_bindings={
        "sample.json": digest(before["sample.json"]),
        "supervision/label.json": digest(before["supervision/label.json"])},
        expected_json={"supervision/query_trace.json": callback["trace"]})
    assert raw.verify_all()

    result = storage.write_compact_native_sample(compact, **callback)
    reader = SampleReader(compact)
    assert reader.verify_all()
    for name in REQUIRED - {"supervision/identities.json"}:
        assert reader.read(name) == raw.read(name)
    assert reader.json("supervision/identities.json") == raw.json("supervision/identities.json")
    for name in EXTRAS - {"sample.json"}:
        assert reader.json(name) == raw.json(name)
    expected_metadata = deepcopy(raw.metadata)
    expected_metadata["files"] = {k: v for k, v in expected_metadata["files"].items() if k in REQUIRED}
    # JSON semantics match; original Windows text writers may use CRLF, whereas
    # the new logical JSON bytes use LF and bind their own original-byte hash.
    expected_metadata["files"]["supervision/identities.json"]["sha256"] = digest(
        reader.read("supervision/identities.json"))
    assert reader.metadata == expected_metadata
    assert reader.array("inputs/depth_m.npy").tobytes() == callback["depth"].tobytes()
    assert np.array_equal(reader.image("inputs/rgb.png"), callback["rgb"])
    assert np.array_equal(reader.image("inputs/depth_valid.png") != 0, callback["valid"])
    assert np.array_equal(reader.image("supervision/target_visible.png") != 0, callback["target_mask"])
    assert before == snapshot(source)

    manifest = reader.manifest
    assert set(manifest["files"]) == REQUIRED | EXTRAS
    assert set(reader.metadata["files"]) == REQUIRED
    for name, entry in reader.metadata["files"].items():
        assert entry["sha256"] == digest(reader.read(name))
        assert entry["role"] == ("observation" if name.startswith("inputs/") else "ground_truth_supervision")
    assert manifest["omitted_review_files"] == {}
    assert manifest["review_policy"] == "not_generated_review_only_pngs_not_model_input"
    assert manifest["rgb_policy"] == "unchanged_original_png"
    assert manifest["depth_policy"] == "exact_native_optical_z_npy_bytes"
    assert all(manifest[k] is False for k in ("lossy", "depth_recomputed", "training_approved", "source_removed"))
    assert result == dict(sample_sha256=digest(reader.read("sample.json")),
        label_sha256=digest(reader.read("supervision/label.json")), query_trace=callback["trace"])
    assert result["sample_sha256"] == manifest["source_sample_sha256"]
    assert result["label_sha256"] == manifest["source_label_sha256"]
    assert result["query_trace"] == manifest["source_query_trace"]
    assert snapshot(compact).keys() == {"bundle.json"} | {e["stored_path"] for e in manifest["files"].values()}
    assert not list(compact.rglob("review")) and not list(compact.rglob("*.npy"))
    assert len(list(compact.rglob("*.ghn"))) == 3
    assert len(list(compact.rglob("*.png"))) == 4
    for name in ("review/overlay.png", "review/visible_target.png"):
        with pytest.raises(ValueError, match="omitted or unavailable"):
            reader.read(name)


def test_all_three_arrays_use_exact_codec_and_writer_verifies_reader(callback, tmp_path, monkeypatch):
    encoded, verified = [], []
    original_encode, original_verify = storage.encode, SampleReader.verify_all

    def encode(raw):
        encoded.append(raw)
        return original_encode(raw)

    def verify(reader):
        verified.append(reader.root)
        return original_verify(reader)

    monkeypatch.setattr(storage, "encode", encode)
    monkeypatch.setattr(SampleReader, "verify_all", verify)
    directory = tmp_path / "sample"
    storage.write_compact_native_sample(directory, **callback)
    assert set(encoded) == {npy_bytes(callback[k]) for k in ("depth", "instances", "components")}
    assert len(encoded) == 3 and verified == [directory.resolve()]
    reader = SampleReader(directory)
    assert {name for name, entry in reader.manifest["files"].items()
            if entry["encoding"] == "native_npy_byteplanes_zstd"} == ARRAYS


def test_does_not_edit_caller_data(callback, tmp_path):
    before = deepcopy(callback)
    storage.write_compact_native_sample(tmp_path / "sample", **callback)
    for name, value in callback.items():
        if isinstance(value, np.ndarray):
            assert value.tobytes() == before[name].tobytes()
        else:
            assert value == before[name]
    assert "files" not in callback["metadata"]


@pytest.mark.parametrize("field", ["rgb", "depth", "valid", "instances", "components", "organs", "target_mask"])
@pytest.mark.parametrize("fault", ["shape", "dtype", "fortran"])
def test_native_buffer_contract_before_any_writes(callback, tmp_path, field, fault):
    if fault == "shape":
        callback[field] = callback[field][:-1].copy()
    elif fault == "dtype":
        callback[field] = np.zeros(callback[field].shape, dtype=np.float64)
    else:
        callback[field] = np.asfortranarray(callback[field])
    directory = tmp_path / "bad"
    with pytest.raises(ValueError, match="native shape/dtype/C-order"):
        storage.write_compact_native_sample(directory, **callback)
    assert not directory.exists()


@pytest.mark.parametrize("fault", ["method", "scene", "dynamic", "rgb", "depth", "camera", "target"])
def test_callback_evidence_must_match_saved_buffers(callback, tmp_path, fault):
    sync = callback["metadata"]["synchronization"]
    if fault == "method":
        sync["method"] = "synthetic_depth_reconstruction"
    elif fault == "scene":
        sync["scene_unchanged_during_capture"] = False
    elif fault == "dynamic":
        sync["dynamic_recording_supported"] = True
    elif fault == "rgb":
        callback["rgb"][0, 0, 0] ^= 1
    elif fault == "depth":
        # A different NaN payload is numerically 'equal_nan', but not exact.
        callback["depth"].view(np.uint32)[0, 0] ^= 1
    elif fault == "camera":
        callback["metadata"]["calibration"]["intrinsics"][0][0] += 1
    else:
        callback["target_mask"][0, 0] = True
    directory = tmp_path / "bad"
    with pytest.raises(ValueError):
        storage.write_compact_native_sample(directory, **callback)
    assert not directory.exists()


@pytest.mark.parametrize("fault", ["resolution", "crop", "depth_convention", "sample_approval",
    "label_approval", "missing_trace", "invalid_trace", "old_files", "json_nan"])
def test_reject_invalid_metadata_before_creation(callback, tmp_path, fault):
    metadata = callback["metadata"]
    if fault == "resolution":
        metadata["calibration"]["resolution"] = [3, 2]
    elif fault == "crop":
        metadata["calibration"]["crop_resize"] = [0, 0, 100, 100]
    elif fault == "depth_convention":
        metadata["calibration"]["depth_convention"] = "ray_range"
    elif fault == "sample_approval":
        metadata["training_sample_approved"] = True
    elif fault == "label_approval":
        callback["label"]["training_approved"] = True
    elif fault == "missing_trace":
        callback["trace"] = None
    elif fault == "invalid_trace":
        callback["trace"] = []
    elif fault == "old_files":
        metadata["files"] = {"old.npy": {"sha256": "a" * 64}}
    else:
        callback["label"]["nonfinite"] = float("nan")
    directory = tmp_path / "bad"
    with pytest.raises(ValueError):
        storage.write_compact_native_sample(directory, **callback)
    assert not directory.exists()


def test_ineligible_label_can_omit_trace(callback, tmp_path):
    callback["label"]["eligible"] = False
    callback["trace"] = None
    result = storage.write_compact_native_sample(tmp_path / "sample", **callback)
    reader = SampleReader(tmp_path / "sample")
    assert reader.verify_all() and result["query_trace"] is None
    assert reader.manifest["source_query_trace"] is None
    assert "supervision/query_trace.json" not in reader.manifest["files"]


@pytest.mark.parametrize("existing", ["empty_directory", "old_capture", "file"])
def test_create_only_never_changes_old_content(callback, tmp_path, existing):
    directory = tmp_path / "old"
    if existing == "file":
        directory.write_bytes(b"immutable existing file")
        before = directory.read_bytes()
    else:
        directory.mkdir()
        if existing == "old_capture":
            (directory / "sample.json").write_bytes(b"immutable old capture")
        before = snapshot(directory)
    with pytest.raises(ValueError, match="Create-only"):
        storage.write_compact_native_sample(directory, **callback)
    assert before == (directory.read_bytes() if directory.is_file() else snapshot(directory))


def test_manifest_is_last_and_files_are_create_only(callback, tmp_path, monkeypatch):
    directory = tmp_path / "sample"
    writes, original_open = [], Path.open

    def open_file(path, mode="r", *args, **kwargs):
        if path.is_relative_to(directory) and any(flag in mode for flag in "xwa+"):
            assert mode == "xb"
            if path.name == "bundle.json":
                assert len(list(directory.rglob("*.ghn"))) == 3
                assert len(writes) == len(REQUIRED | EXTRAS)
            else:
                assert not (directory / "bundle.json").exists()
            writes.append(path)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    storage.write_compact_native_sample(directory, **callback)
    assert writes[-1] == directory / "bundle.json"
    assert len(writes) == len(REQUIRED | EXTRAS) + 1


def test_nonexact_codec_never_creates_destination(callback, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "decode", lambda encoded: b"wrong NPY")
    with pytest.raises(ValueError, match="Non-exact native roundtrip"):
        storage.write_compact_native_sample(tmp_path / "bad", **callback)
    assert not (tmp_path / "bad").exists()


def test_interrupted_payload_never_publishes_manifest(callback, tmp_path, monkeypatch):
    directory = tmp_path / "sample"
    original_open = Path.open

    def open_file(path, mode="r", *args, **kwargs):
        if mode == "xb" and path.name == "depth_valid.png":
            raise OSError("simulated disk failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(OSError, match="simulated disk failure"):
        storage.write_compact_native_sample(directory, **callback)
    assert not (directory / "bundle.json").exists()
    with pytest.raises(FileNotFoundError):
        SampleReader(directory)
    before = snapshot(directory)
    with pytest.raises(ValueError, match="Create-only"):
        storage.write_compact_native_sample(directory, **callback)
    assert before == snapshot(directory)


def test_disk_readback_failure_never_publishes_manifest(callback, tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "read_bounded", lambda path: b"changed on disk")
    with pytest.raises(ValueError, match="changed before publish"):
        storage.write_compact_native_sample(tmp_path / "bad", **callback)
    assert not (tmp_path / "bad/bundle.json").exists()


@pytest.mark.parametrize("name", sorted(REQUIRED | EXTRAS))
def test_reader_rejects_every_corrupted_payload(callback, tmp_path, name):
    directory = tmp_path / "sample"
    storage.write_compact_native_sample(directory, **callback)
    reader = SampleReader(directory)
    path = directory / reader.manifest["files"][name]["stored_path"]
    raw = path.read_bytes()
    path.write_bytes(raw[:-1] + bytes([raw[-1] ^ 1]))
    with pytest.raises(ValueError, match="Corrupt"):
        SampleReader(directory).verify_all()


@pytest.mark.parametrize("field", ["source_sample_sha256", "source_label_sha256", "source_query_trace"])
def test_reader_rejects_changed_capture_bindings(callback, tmp_path, field):
    directory = tmp_path / "sample"
    storage.write_compact_native_sample(directory, **callback)
    reader = SampleReader(directory)
    reader.manifest[field] = "changed binding"
    (directory / "bundle.json").write_text(json.dumps(reader.manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="binding|query trace"):
        SampleReader(directory).verify_all()


def test_reader_rejects_missing_eligible_trace_binding(callback, tmp_path):
    directory = tmp_path / "sample"
    storage.write_compact_native_sample(directory, **callback)
    manifest = SampleReader(directory).manifest
    del manifest["source_query_trace"]
    (directory / "bundle.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="query trace"):
        SampleReader(directory).verify_all()
