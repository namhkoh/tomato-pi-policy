"""Synthetic CPU receipts only; no native renderer, assets or training rows."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from greenhouse_sim.robot_kinematics import Rby1Kinematics
from ..cut_regions import rule_fingerprint, load_rule
from . import prepare as module
from .contracts import FROZEN_SPLITS, HEAD_CAMERA, ADMISSION, sha256, read_json, safe_file, new_destination, write_new


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return sha256(path)


@pytest.fixture
def source(tmp_path, monkeypatch):
    package, capture = tmp_path / "package", tmp_path / "legacy"
    asset = package / "house/original.usd"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"UNIT TEST PLACEHOLDER NOT A RENDERABLE ASSET")
    bindings = {str(asset): sha256(asset)}
    def row(family, component):
        return dict(target_id=family + "/" + component, component_id=component,
            draft_id="N_" + family + "_" + component, variant_id=family,
            source_plant_id=family, split_group=family, cut_region_proposal={"unit_test_geometry": component})
    jobs = [dict(job_id=f"job_{i:03d}", plant_family=family, split=split,
                 source_manifest_path=str(package / "plants/components" / family / "manifest.json"),
                 targets=[row(family, "SubStem_41"), row(family, "SubStem_44")])
            for i, (family, split) in enumerate(sorted(FROZEN_SPLITS.items()), 1)]
    job = next(j for j in jobs if j["plant_family"] == "seed73_full")
    rule = load_rule()
    old = dict(schema_version="greenhouse.grounding_collection_plan.v1",
        state="ready_for_synthetic_grounding_capture", training_dataset_approved=False,
        family_assignments=deepcopy(FROZEN_SPLITS), jobs=jobs,
        split_scope="target_source_families_only_shared_greenhouse_backdrop_context",
        configuration=dict(seed=0, source_geometry="unmodified_native_components", original_plant_root_z_m=.9),
        source_bindings_sha256=bindings, package=str(package), cut_rule=rule, cut_rule_sha256=rule_fingerprint(rule))
    clear = deepcopy(old)
    clear["configuration"]["clear_capture"] = "robot_head_close_diffuse_v1"
    old_path, clear_path = tmp_path / "old_plan/plan.json", tmp_path / "clear_plan/plan.json"
    write(old_path, old)
    write(clear_path, clear)
    model = Rby1Kinematics()
    joints = {name: 0. for name, joint in model._by_name.items() if joint.kind == "revolute"}
    samples = []
    for index, target in enumerate(job["targets"], 1):
        sid = f"sample_{index:04d}"
        root = np.eye(4)
        root[3, 0] = index * .01
        mount = np.eye(4)
        camera = (root.T @ model.all_link_transforms(joints)["link_head_2"] @ mount).T
        cal = dict(camera_path=HEAD_CAMERA, resolution=[848, 408], crop_resize=None,
            depth_convention="optical_axis_z_metres_not_ray_range", camera_to_world_usd_row_vectors=camera.tolist(),
            intrinsics=[[500., 0., 424.], [0., 500., 204.], [0., 0., 1.]], clipping_range_m=[.04, 10.],
            focal_length_mm=10., apertures_mm=[16.96, 8.16], aperture_offsets_mm=[0., 0.])
        files = {}
        for name in ("inputs/rgb.png", "inputs/depth_m.npy", "supervision/renderer_instance_id.npy"):
            path = capture / sid / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((sid + name + " UNIT TEST ONLY").encode())
            files[name] = {"sha256": sha256(path)}
        sample = dict(schema_version="greenhouse.rgbd_pilot_sample.v2", sample_id=sid,
            calibration=cal, robot_snapshot=dict(joint_degrees=joints, robot_root_to_world_usd_row_vectors=root.tolist(),
                camera_to_head_column_vectors=mount.tolist(), old_visibility_approval="MUST NOT INHERIT"),
            synchronization=dict(scene_unchanged_during_capture=True, dynamic_recording_supported=False),
            supervision=dict(target_id=target["target_id"], split_group="seed73_full", variant_id="seed73_full",
                review_id=target["draft_id"], cut_region_proposal=target["cut_region_proposal"],
                plant_to_world_usd_row_vectors=np.eye(4).tolist()), files=files)
        write(capture / sid / "sample.json", sample)
        samples.append(dict(sample_id=sid, target_review_id=target["draft_id"]))
    variant = dict(variant_id="seed73_full", source_plant_id="seed73_full", split_group="seed73_full",
                   plant_root="/World/Original", added_components={}, added_component_paths={}, source_geometry_modified=False)
    manifest = dict(state="pilot_ready_for_review", source_assets_unchanged=True, source_geometry_modified=False,
        target_family_split="train", samples=samples, source_collection_plan_path=str(old_path),
        source_collection_plan_sha256=sha256(old_path), source_usd_sha256=bindings,
        collection_job_id=job["job_id"], variants=[variant], lighting={"dome_intensity": 1200},
        scene_counts={"components": 2}, unbundled_external_prop_roots_excluded=[])
    write(capture / "manifest.json", manifest)
    events = []
    monkeypatch.setattr(module, "original_report", lambda *args: events.append("original_report") or {})
    monkeypatch.setattr(module, "appearance_bindings", lambda *args: events.append("appearance") or {})
    monkeypatch.setattr(module, "code_bindings", lambda: {str(Path(module.__file__)): sha256(module.__file__)})
    def options():
        return dict(clear_plan=clear_path, clear_plan_sha256=sha256(clear_path),
            source_capture=capture, source_manifest_sha256=sha256(capture / "manifest.json"))
    def case(index=1):
        return dict(target_id=job["targets"][index-1]["target_id"], sample_id=f"sample_{index:04d}",
            source_sample_sha256=sha256(capture / f"sample_{index:04d}/sample.json"))
    return dict(root=tmp_path, package=package, asset=asset, capture=capture, clear_path=clear_path,
                options=options, case=case, events=events)


def test_single_is_same_contract_as_batch_and_no_source_writes(source):
    before = {str(p): p.read_bytes() for p in source["root"].rglob("*") if p.is_file()}
    single = module.prepare_plan(**source["options"](), **source["case"]())
    assert single["sample_count_limit"] == 1 and single["admission"] == ADMISSION
    assert single["geometry_mode"] == "unmodified_original"
    assert "old_visibility_approval" not in single["expected_robot_snapshot"]
    assert single["coverage"]["frozen_train_target_rows"] == 32
    assert single["coverage"]["selected_same_target_priors"] == 1
    assert single["coverage"]["native_reachability_or_clarity_measured"] is False
    assert module.check_plan(single)
    assert {str(p): p.read_bytes() for p in source["root"].rglob("*") if p.is_file()} == before


def test_batch_shares_source_report_and_appearance_verification(source):
    plan = module.prepare_batch(**source["options"](), cases=[source["case"](1), source["case"](2)])
    assert len(plan["cases"]) == 2 and len(plan["batch_request"]) == 2
    assert source["events"] == ["appearance", "original_report"]
    assert plan["coverage"]["requested_original_targets"] == 2
    assert plan["cases"][0]["pose_prior"]["prior_target_id"] != plan["cases"][1]["pose_prior"]["prior_target_id"]


@pytest.mark.parametrize("field,value", [
    ("geometry_mode", "generated_variant"), ("sample_count_limit", 65), ("split", "test"),
    ("conservative_view_cap_group", "new_budget"), ("source_family", "seed99_full"),
    ("admission", {"training_approved": True}), ("scene_policy", {"render_subframes": 8}),
])
def test_plan_mutations_fail(source, field, value):
    plan = module.prepare_plan(**source["options"](), **source["case"]())
    plan[field] = value
    with pytest.raises(ValueError):
        module.check_plan(plan)


def test_changed_target_requires_opt_in_and_never_changes_prior_identity(source):
    case = source["case"]()
    case["target_id"] = "seed73_full/SubStem_44"
    with pytest.raises(ValueError, match="explicit pose-only"):
        module.prepare_plan(**source["options"](), **case)
    plan = module.prepare_plan(**source["options"](), **case, allow_other_target_pose=True, reframe_pixel_xy=[848., 408.])
    assert plan["pose_prior"]["prior_target_id"] == "seed73_full/SubStem_41"
    assert plan["target_id"] == "seed73_full/SubStem_44"
    assert plan["pose_request"]["mode"] == "reframe_head_only"
    assert plan["source_row"]["cut_region_proposal"]["unit_test_geometry"] == "SubStem_44"


@pytest.mark.parametrize("pixel", [[0, 408], [1696, 408], [848, 816], [True, 4], [float("nan"), 2]])
def test_invalid_native_reframing_rejected(source, pixel):
    with pytest.raises(ValueError):
        module.prepare_plan(**source["options"](), **source["case"](), allow_other_target_pose=True, reframe_pixel_xy=pixel)


def test_duplicates_bounds_heldout_and_unknown_target_rejected(source):
    case = source["case"]()
    for cases in ([], [case] * 65, [case, case]):
        with pytest.raises(ValueError):
            module.prepare_batch(**source["options"](), cases=cases)
    for target in ("seed31_full/SubStem_41", "seed73_full/SubStem_999"):
        with pytest.raises(ValueError):
            module.prepare_plan(**source["options"](), **dict(case, target_id=target), allow_other_target_pose=True)


def test_frozen_family_change_even_with_new_plan_pin_fails(source):
    plan = read_json(source["clear_path"])
    plan["family_assignments"]["seed73_full"] = "test"
    write(source["clear_path"], plan)
    with pytest.raises(ValueError, match="24-family"):
        module.prepare_plan(**source["options"](), **source["case"]())


@pytest.mark.parametrize("mutation", ["arm_missing", "head_missing", "unknown", "limit", "mount_scale", "mount_reflect", "mount_projective", "fk"])
def test_complete_rigid_robot_prior_required(source, mutation):
    path = source["capture"] / "sample_0001/sample.json"
    sample = read_json(path)
    pose = sample["robot_snapshot"]
    if mutation == "arm_missing":
        del pose["joint_degrees"]["right_arm_3"]
    elif mutation == "head_missing":
        del pose["joint_degrees"]["head_0"]
    elif mutation == "unknown":
        pose["joint_degrees"]["not_a_joint"] = 0
    elif mutation == "limit":
        pose["joint_degrees"]["right_arm_3"] = 10000
    elif mutation == "mount_scale":
        pose["camera_to_head_column_vectors"][0][0] = 2
    elif mutation == "mount_reflect":
        pose["camera_to_head_column_vectors"][0][0] = -1
    elif mutation == "mount_projective":
        pose["camera_to_head_column_vectors"][3][0] = .01
    else:
        sample["calibration"]["camera_to_world_usd_row_vectors"][3][0] += .1
    write(path, sample)
    with pytest.raises(ValueError):
        module.prepare_plan(**source["options"](), **source["case"]())


def test_stale_source_files_and_stale_root_pins_rejected(source):
    options = source["options"]()
    source["asset"].write_bytes(b"CHANGED")
    with pytest.raises(ValueError, match="Changed or missing pinned"):
        module.prepare_plan(**options, **source["case"]())


def test_traversal_destinations_and_exclusive_receipts(tmp_path):
    for name in ("../escape", "C:/escape", "a\\b"):
        with pytest.raises(ValueError):
            safe_file(tmp_path, name)
    with pytest.raises(ValueError):
        new_destination(tmp_path / "protected/new", [tmp_path / "protected"])
    path = tmp_path / "receipt.json"
    write_new(path, {"test_only": True})
    with pytest.raises(FileExistsError):
        write_new(path, {})


def test_duplicate_json_keys_fail(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"a":1,"a":2}')
    with pytest.raises(ValueError, match="Duplicate JSON"):
        read_json(path)
