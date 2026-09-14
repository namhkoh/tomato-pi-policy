from copy import deepcopy
import json
import numpy as np
import pytest
from .capture_contract import project
from .capture_sensor import HIRES_RESOLUTION, calibration_for_native_resolution
from .capture_sensor_test import example_calibration
from .dataset_review import write_json
from .depth_preview import sha256
from .native_greenhouse_pair import assert_pair, assert_same_camera, load_source


def pair():
    low_cal = example_calibration()
    high_cal = calibration_for_native_resolution(low_cal, HIRES_RESOLUTION)
    world = dict(nominal_world_m=[0, 0, -1], interval_world_m=[[0, 0, -1], [.01, 0, -1]],
                 plant_to_world_usd_row_vectors=np.eye(4).tolist(), target_id="seed7/SubStem_42")
    return [dict(calibration=c, supervision=dict(**deepcopy(world),
                 projected_interval=project(world["interval_world_m"], c))) for c in (low_cal, high_cal)]


def test_pair_preserves_metric_geometry_with_double_pixels():
    low, high = pair()
    result = assert_pair(low, high)
    assert result["native_pixel_scale"] == 2 and not result["depth_images_resampled"]
    assert not result["visual_clarity_improvement_measured"]


@pytest.mark.parametrize("fault", ["pose", "optics", "pixels", "world", "identity", "keys", "resolution"])
def test_pair_rejects_pose_optics_geometry_or_coordinate_change(fault):
    low, high = pair()
    if fault == "pose": high["calibration"]["camera_to_world_usd_row_vectors"][3][0] += .001
    if fault == "optics": high["calibration"]["focal_length_mm"] += .001
    if fault == "pixels": high["supervision"]["projected_interval"][0]["pixel_xy"][0] += 1
    if fault == "world": high["supervision"]["nominal_world_m"][0] += .001
    if fault == "identity": high["supervision"]["target_id"] = "another_target"
    if fault == "keys": high["calibration"]["unexpected"] = True
    if fault == "resolution": high["calibration"]["resolution"] = [848, 408]
    with pytest.raises(ValueError): assert_pair(low, high)


def source(tmp_path):
    capture = tmp_path / "capture"
    sample_dir = capture / "sample_0001"
    sample_dir.mkdir(parents=True)
    asset = tmp_path / "source.usda"
    asset.write_text("#usda 1.0")
    plan = tmp_path / "plan.json"
    write_json(plan, {})
    rgb = sample_dir / "rgb_fixture.txt"
    rgb.write_text("unit-test bytes, not an image or dataset")
    sample = dict(schema_version="greenhouse.rgbd_pilot_sample.v2", calibration=example_calibration(),
        synchronization=dict(scene_unchanged_during_capture=True, dynamic_recording_supported=False),
        files={"rgb_fixture.txt": {"sha256": sha256(rgb)}})
    manifest = dict(state="pilot_ready_for_review", source_assets_unchanged=True,
        target_family_split="train", samples=[{"sample_id": "sample_0001"}],
        source_usd_sha256={str(asset): sha256(asset)},
        source_collection_plan_path=str(plan), source_collection_plan_sha256=sha256(plan))
    write_json(sample_dir / "sample.json", sample)
    write_json(capture / "manifest.json", manifest)
    return capture, sample, manifest


def test_source_verifies_reference_and_all_file_hashes(tmp_path):
    capture, sample, manifest = source(tmp_path)
    m, s, hashes = load_source(capture, "sample_0001")
    assert m == manifest and s == sample and len(hashes) == 5
    (capture / "sample_0001/rgb_fixture.txt").write_text("changed")
    with pytest.raises(ValueError): load_source(capture, "sample_0001")


@pytest.mark.parametrize("fault", ["heldout", "incomplete", "changed_source", "dynamic", "wrong_resolution", "escape", "unknown_sample"])
def test_source_rejects_unqualified_reference(tmp_path, fault):
    capture, sample, manifest = source(tmp_path)
    identifier = "sample_0001"
    if fault == "heldout": manifest["target_family_split"] = "test"
    if fault == "incomplete": manifest["state"] = "initializing"
    if fault == "changed_source": manifest["source_assets_unchanged"] = False
    if fault == "dynamic": sample["synchronization"]["dynamic_recording_supported"] = True
    if fault == "wrong_resolution": sample["calibration"]["resolution"] = [1696, 816]
    if fault == "escape": sample["files"]["../../plan.json"] = {"sha256": manifest["source_collection_plan_sha256"]}
    if fault == "unknown_sample": identifier = "sample_0002"
    (capture / "sample_0001/sample.json").write_text(json.dumps(sample), encoding="utf-8")
    (capture / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError): load_source(capture, identifier)


@pytest.mark.parametrize("identifier", ["../sample_0001", "sample_../../", "sample_", "/sample_0001"])
def test_no_source_path_traversal(tmp_path, identifier):
    with pytest.raises(ValueError): load_source(tmp_path, identifier)
