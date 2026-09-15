"""CPU tests for generated native pair contracts; no renderer is launched."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from .capture_sensor import HIRES_RESOLUTION, calibration_for_native_resolution
from .capture_sensor_test import example_calibration
from .dataset_review import write_json
from .depth_preview import sha256
from .generated_capture import check_plan, assert_pair_fresh, verify_sensor_prerequisite, check_smoke_metadata
from .native_greenhouse_pair_test import pair as resolution_pair


def matched_pair():
    cal = calibration_for_native_resolution(example_calibration(), HIRES_RESOLUTION)
    cal["camera_path"] = "/World/RBY1/HeadCamera"
    a = dict(calibration=cal, robot_snapshot={"joint_degrees": [0, 1]}, scene_counts={"components": 8},
        lighting={"intensity": 1500}, native_target_pixels=25, old_plant_native_pixels=25,
        supervision=dict(source_target_id="seed7/SubStem_42", target_id="seed7/SubStem_42",
            plant_to_world_usd_row_vectors=np.eye(4).tolist(), nominal_world_m=[0, 0, 1]),
        synchronization=dict(static_guard="a"*64, freshness=dict(callback_sequence=7,
            rgb_sha256="a"*64, depth_sha256="b"*64)))
    b = deepcopy(a)
    b["supervision"]["target_id"] = "seed7_variant/SubStem_42"
    b["supervision"]["nominal_world_m"] = [.001, 0, 1]
    b["old_plant_native_pixels"] = 0
    b["synchronization"] = dict(static_guard="b"*64, freshness=dict(
        callback_sequence=14, rgb_sha256="c"*64, depth_sha256="d"*64))
    return a, b


def test_fresh_same_camera_changed_plant_pair():
    result = assert_pair_fresh(*matched_pair())
    assert result["new_native_target_identity_observed"]
    assert not result["training_approved"] and not result["dynamic_synchronization_supported"]


@pytest.mark.parametrize("fault", ["camera", "pose", "counts", "lighting", "lineage", "same_id",
    "placement", "callback", "guard", "rgb", "depth", "invisible", "old_pixels", "old_label"])
def test_pair_rejects_stale_or_unmatched_capture(fault):
    a, b = matched_pair()
    if fault == "camera": b["calibration"]["camera_to_world_usd_row_vectors"][3][0] += .01
    if fault == "pose": b["robot_snapshot"]["joint_degrees"][0] = 2
    if fault == "counts": b["scene_counts"]["components"] = 7
    if fault == "lighting": b["lighting"]["intensity"] = 2000
    if fault == "lineage": b["supervision"]["source_target_id"] = "another"
    if fault == "same_id": b["supervision"]["target_id"] = a["supervision"]["target_id"]
    if fault == "placement": b["supervision"]["plant_to_world_usd_row_vectors"][3][0] = 1
    if fault == "callback": b["synchronization"]["freshness"]["callback_sequence"] = 7
    if fault == "guard": b["synchronization"]["static_guard"] = a["synchronization"]["static_guard"]
    if fault in ("rgb", "depth"):
        b["synchronization"]["freshness"][fault+"_sha256"] = a["synchronization"]["freshness"][fault+"_sha256"]
    if fault == "invisible": b["native_target_pixels"] = 0
    if fault == "old_pixels": b["old_plant_native_pixels"] = 1
    if fault == "old_label": b["supervision"]["nominal_world_m"] = a["supervision"]["nominal_world_m"]
    with pytest.raises(ValueError):
        assert_pair_fresh(a, b)


def simple_plan(tmp_path):
    asset = tmp_path/"source.txt"
    asset.write_text("unit-test source binding")
    a = dict(source_plant_id="seed7", split_group="seed7", component_id="SubStem_42", target_id="seed7/SubStem_42")
    b = dict(**a)
    b.update(target_id="seed7_variant/SubStem_42", variant_id="seed7_variant",
             conservative_view_cap_group=a["target_id"])
    cal = calibration_for_native_resolution(example_calibration(), HIRES_RESOLUTION)
    cal["camera_path"] = "/World/RBY1/HeadCamera"
    return dict(schema_version="greenhouse.generated_native_pair_plan.v1",
        state="cpu_planned_pending_native_sensor_prerequisite", modes=["original_control", "generated_variant"],
        sample_count_limit=2, resolution=list(HIRES_RESOLUTION), source_family="seed7",
        split="train", split_group="seed7", source_row=a, generated_row=b,
        conservative_view_cap_group=a["target_id"], camera_path=cal["camera_path"], expected_calibration=cal,
        training_eligible=False, collision_qualified=False, dynamics_supported=False,
        higher_resolution_execution_verified=False, source_bindings={str(asset): sha256(asset)})


def test_fixed_two_frame_plan_has_no_training_authority(tmp_path):
    check_plan(simple_plan(tmp_path))


@pytest.mark.parametrize("fault", ["count", "boolcount", "mode", "resolution", "heldout", "family",
                                  "same_target", "cap", "camera", "approval", "source"])
def test_plan_rejects_scope_changes(tmp_path, fault):
    p = simple_plan(tmp_path)
    if fault == "count": p["sample_count_limit"] = 3
    if fault == "boolcount": p["sample_count_limit"] = True
    if fault == "mode": p["modes"] = ["generated_variant"]
    if fault == "resolution": p["resolution"] = [848, 408]
    if fault == "heldout": p["split"] = "test"
    if fault == "family": p["generated_row"]["source_plant_id"] = "seed13"
    if fault == "same_target": p["generated_row"]["target_id"] = p["source_row"]["target_id"]
    if fault == "cap": p["generated_row"]["conservative_view_cap_group"] = "new_seed"
    if fault == "camera": p["camera_path"] = "/World/Cinematic"
    if fault == "approval": p["training_eligible"] = True
    if fault == "source": (tmp_path/"source.txt").write_text("changed")
    with pytest.raises(ValueError):
        check_plan(p)


def prerequisite(tmp_path):
    plan = simple_plan(tmp_path)
    root = tmp_path/"qualification"
    root.mkdir()
    reference = tmp_path/"capture/sample_0001/sample.json"
    reference.parent.mkdir(parents=True)
    reference.write_text("unit-test reference bytes")
    plan.update(prerequisite_directory=str(root), source_capture=str(reference.parent.parent), source_sample="sample_0001")
    samples = resolution_pair()
    for sample, name in zip(samples, ("native_848x408", "native_1696x816"), strict=True):
        sample["calibration"]["camera_path"] = plan["camera_path"]
        folder = root/name
        folder.mkdir()
        data = folder/"fixture.txt"
        data.write_text("not an image; unit-test binding")
        sample["files"] = {"fixture.txt": {"sha256": sha256(data)}}
        write_json(folder/"sample.json", sample)
    result = dict(state="native_greenhouse_pair_captured_pending_visual_review",
        native_sensor_smoke_passed=True, source_assets_unchanged=True, source_split="train",
        geometry_screen={"passed": True}, training_approved=False, samples=samples)
    write_json(root/"result.json", result)
    write_json(root/"request.json", {"source_bindings": {str(reference): sha256(reference)}})
    (root/"sensor_smoke").mkdir()
    smoke = smoke_fixture()
    write_json(root/"sensor_smoke/result.json", smoke)
    for record in smoke["checks"]:
        folder = root/"sensor_smoke"/record["name"]
        folder.mkdir()
        write_json(folder/"diagnostic.json", record)
        for filename in ("rgb_diagnostic.png", "native_optical_z_m.npy", "validity_diagnostic.png", "native_instance_ids.npy"):
            (folder/filename).write_bytes(b"unit-test binding fixture, not a sensor image")
    return plan, root, result


def test_completed_numerical_prerequisite_is_not_visual_approval(tmp_path):
    p, root, _ = prerequisite(tmp_path)
    proof = verify_sensor_prerequisite(p)
    assert proof["native_numerical_prerequisite_passed"]
    assert not proof["visual_approval"] and not proof["training_approval"]
    assert len(proof["bindings"]) == 27


def smoke_fixture():
    records = []
    for name, z, path in (("pose_0", 2., "/World/CalibrationCube"), ("pose_1", 2.4, "/World/CalibrationCube"),
                          ("occluder_visible", 1., "/World/KnownOccluder"), ("occluder_hidden", 2.4, "/World/CalibrationCube")):
        records.append(dict(name=name, native_resolution=list(HIRES_RESOLUTION), no_training_input=True,
            known_surface=[dict(pixel_xy=xy, native_prim_path=path, expected_z_m=z, native_z_m=z)
                           for xy in ([848, 408], [940, 408])]))
    return dict(state="native_resolution_smoke_passed_not_training_data", native_depth_reconstructed=False,
                rgb_upscaling_performed=False, checks=records)


@pytest.mark.parametrize("fault", ["empty", "depth", "identity", "resolution", "probe", "nan"])
def test_smoke_requires_measured_known_surface_records(fault):
    smoke = smoke_fixture()
    if fault == "empty": smoke["checks"] = [{}, {}, {}, {}]
    if fault == "depth": smoke["checks"][0]["known_surface"][0]["native_z_m"] = 2.1
    if fault == "identity": smoke["checks"][0]["known_surface"][0]["native_prim_path"] = "/Wrong"
    if fault == "resolution": smoke["checks"][0]["native_resolution"] = [848, 408]
    if fault == "probe": smoke["checks"][0]["known_surface"][1]["pixel_xy"] = [848, 408]
    if fault == "nan": smoke["checks"][0]["known_surface"][0]["native_z_m"] = float("nan")
    with pytest.raises(ValueError):
        check_smoke_metadata(smoke)


@pytest.mark.parametrize("fault", ["missing", "failure", "no_smoke", "heldout", "geometry", "reference", "files", "projection"])
def test_incomplete_or_changed_native_prerequisite_blocks(tmp_path, fault):
    p, root, result = prerequisite(tmp_path)
    if fault == "missing": p["prerequisite_directory"] = str(tmp_path/"not_ready")
    if fault == "failure": write_json(root/"failure.json", {"error": "failed"})
    if fault == "no_smoke": result["native_sensor_smoke_passed"] = False
    if fault == "heldout": result["source_split"] = "test"
    if fault == "geometry": result["geometry_screen"]["passed"] = False
    if fault == "reference": p["source_sample"] = "sample_9999"
    if fault == "files": (root/"native_1696x816/fixture.txt").write_text("changed")
    if fault == "projection": result["samples"][1]["calibration"]["intrinsics"][0][0] += 1
    (root/"result.json").write_text(__import__("json").dumps(result))
    with pytest.raises(ValueError):
        verify_sensor_prerequisite(p)
