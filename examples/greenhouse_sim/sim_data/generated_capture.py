"""Frozen plan and scene substitution for a native generated-plant diagnostic.

No production-plan mutation, no training approval, no robot commands. USD is
imported only inside CPU planning/scene functions, never at module import.
"""
from copy import deepcopy
from pathlib import Path
import argparse

import numpy as np

from .capture_sensor import HIRES_RESOLUTION, calibration_for_native_resolution
from .dataset_review import read_json, write_json, require, verify_bindings, safe_file
from .depth_preview import sha256
from .native_greenhouse_pair import load_source, assert_same_camera, assert_pair

SCHEMA = "greenhouse.generated_native_pair_plan.v1"


def check_smoke_metadata(smoke):
    require(smoke.get("state") == "native_resolution_smoke_passed_not_training_data"
            and smoke.get("native_depth_reconstructed") is False
            and smoke.get("rgb_upscaling_performed") is False, "Missing native known-surface smoke evidence")
    expected = [("pose_0", 2., "/World/CalibrationCube"), ("pose_1", 2.4, "/World/CalibrationCube"),
                ("occluder_visible", 1., "/World/KnownOccluder"), ("occluder_hidden", 2.4, "/World/CalibrationCube")]
    records = smoke.get("checks", [])
    require([r.get("name") for r in records] == [r[0] for r in expected], "Incomplete smoke check sequence")
    for record, (name, z, prim_path) in zip(records, expected, strict=True):
        require(record.get("native_resolution") == list(HIRES_RESOLUTION)
                and record.get("no_training_input") is True, "Wrong native smoke resolution/scope")
        probes = record.get("known_surface", [])
        require(len(probes) == 2 and [r.get("pixel_xy") for r in probes] == [[848, 408], [940, 408]],
                "Missing native center/off-axis optical-Z probes")
        for probe in probes:
            require(probe.get("native_prim_path") == prim_path and probe.get("expected_z_m") == z
                    and type(probe.get("native_z_m")) in (int, float)
                    and np.isfinite(probe["native_z_m"]) and abs(probe["native_z_m"]-z) <= .002,
                    "Native smoke depth or identity evidence disagrees")


def prepare_plan(source_capture, sample_id, variant_directory, qualification_directory):
    from .plant_variant_catalogue import load_for_inspection
    source_capture, variant_directory = Path(source_capture).resolve(), Path(variant_directory).resolve()
    manifest, sample, bindings = load_source(source_capture, sample_id)
    source_plan = read_json(manifest["source_collection_plan_path"])
    c = load_for_inspection(variant_directory, manifest["source_collection_plan_path"])
    target_id = sample["supervision"]["target_id"]
    family, component = target_id.split("/")
    require(family == c["source_family"] and manifest["target_family_split"] == "train",
            "Matched TRAIN donor required")
    job = next(j for j in source_plan["jobs"] if j["job_id"] == manifest["collection_job_id"])
    source_row = next(r for r in job["targets"] if r["target_id"] == target_id)
    generated = [r for r in c["rows"] if r["component_id"] == component]
    require(len(generated) == 1, "Generated target absent or withheld by geometry checks")
    require(source_row["cut_region_proposal"] == sample["supervision"]["cut_region_proposal"],
            "Source anatomical labels changed")
    require(source_row["cut_region_proposal"] != generated[0]["cut_region_proposal"],
            "Diagnostic requires actually changed target geometry")
    require(sample["supervision"]["nominal_projected"]["projection_status"] == "in_frame",
            "Source target is not in the captured view")
    original_variants = [v for v in manifest["variants"] if v["source_plant_id"] == family]
    require(len(original_variants) == 1, "One exact original foreground plant required")
    require(original_variants[0]["variant_id"] == family
            and not original_variants[0]["added_components"], "Unmodified original control required")
    receipt = read_json(variant_directory/"qualification.json")
    for relative, expected in receipt["output_hashes"].items():
        path = safe_file(variant_directory, relative)
        require(sha256(path) == expected, "Generated asset changed")
        bindings[str(path)] = expected
    for path in (variant_directory/"qualification.json",
                 variant_directory.parent/"frozen_lineage.json",
                 variant_directory.parent/"training_envelope.json"):
        bindings[str(path)] = sha256(path)
    bindings.update(c["texture_bindings"])
    # Snapshot all currently referenced file identities; no existing plan is changed.
    verify_bindings(bindings)
    return dict(schema_version=SCHEMA, state="cpu_planned_pending_native_sensor_prerequisite",
        source_capture=str(source_capture), source_sample=sample_id,
        source_collection_plan=manifest["source_collection_plan_path"],
        source_row=source_row, generated_row=generated[0], variant_directory=str(variant_directory),
        original_variant=original_variants[0], source_bindings=bindings,
        prerequisite_directory=str(Path(qualification_directory).resolve()),
        resolution=list(HIRES_RESOLUTION), camera_path=sample["calibration"]["camera_path"],
        expected_calibration=calibration_for_native_resolution(sample["calibration"], HIRES_RESOLUTION),
        expected_scene_counts=manifest["scene_counts"], expected_robot_snapshot=sample["robot_snapshot"],
        modes=["original_control", "generated_variant"], sample_count_limit=2,
        source_family=family, split="train", split_group=family,
        conservative_view_cap_group=target_id, training_eligible=False,
        collision_qualified=False, dynamics_supported=False,
        visual_review="pending", higher_resolution_execution_verified=False)


def check_plan(plan):
    require(plan.get("schema_version") == SCHEMA
            and plan.get("state") == "cpu_planned_pending_native_sensor_prerequisite", "Unknown generated capture plan")
    require(plan.get("modes") == ["original_control", "generated_variant"]
            and type(plan.get("sample_count_limit")) is int and plan["sample_count_limit"] == 2,
            "Bounded two-frame diagnostic required")
    require(plan.get("resolution") == list(HIRES_RESOLUTION), "Explicit qualified native resolution required")
    family = plan["source_family"]
    require(plan["split"] == "train" and plan["split_group"] == family, "TRAIN donor lineage required")
    require(all(plan.get(k) is False for k in ("training_eligible", "collision_qualified", "dynamics_supported",
                                              "higher_resolution_execution_verified")),
            "Planning cannot grant native/data/physical approval")
    a, b = plan["source_row"], plan["generated_row"]
    require(a["source_plant_id"] == b["source_plant_id"] == family
            and a["split_group"] == b["split_group"] == family
            and a["component_id"] == b["component_id"], "Source/generated target mismatch")
    require(a["target_id"] == family+"/"+a["component_id"]
            and b["target_id"] == b["variant_id"]+"/"+b["component_id"]
            and a["target_id"] != b["target_id"], "Matched target identities required")
    require(plan["conservative_view_cap_group"] == b["conservative_view_cap_group"] == a["target_id"],
            "Variant cannot reset donor target view cap")
    require(plan["camera_path"] == plan["expected_calibration"]["camera_path"]
            and plan["expected_calibration"]["resolution"] == list(HIRES_RESOLUTION),
            "Camera calibration identity mismatch")
    verify_bindings(plan["source_bindings"])


def verify_sensor_prerequisite(plan):
    """Require the completed paired native sensor test; numerical evidence, not visual approval."""
    directory = Path(plan["prerequisite_directory"])
    require(not (directory/"failure.json").exists(), "Prior sensor diagnostic failed")
    result_path, request_path = directory/"result.json", directory/"request.json"
    require(result_path.is_file() and request_path.is_file(), "Native camera qualification has not completed")
    result, request = read_json(result_path), read_json(request_path)
    require(result.get("state") == "native_greenhouse_pair_captured_pending_visual_review"
            and result.get("native_sensor_smoke_passed") is True
            and result.get("source_assets_unchanged") is True
            and result.get("source_split") == "train"
            and result.get("geometry_screen", {}).get("passed") is True,
            "Incomplete native camera/scene qualification")
    require(result.get("training_approved") is False, "Sensor diagnostic is not a training release")
    verify_bindings(request["source_bindings"])
    require(str(Path(plan["source_capture"])/plan["source_sample"]/"sample.json") in request["source_bindings"],
            "Sensor qualification used a different reference sample")
    bindings = {str(result_path): sha256(result_path), str(request_path): sha256(request_path)}
    samples = []
    for name, resolution in (("native_848x408", [848, 408]), ("native_1696x816", list(HIRES_RESOLUTION))):
        sample_path = directory/name/"sample.json"
        sample = read_json(sample_path)
        require(sample["calibration"]["resolution"] == resolution
                and sample["supervision"]["target_id"] == plan["source_row"]["target_id"],
                "Qualification sample target or resolution differs")
        require(sample in result["samples"], "Qualification result and stored sample disagree")
        for relative, info in sample["files"].items():
            path = safe_file(sample_path.parent, relative)
            require(sha256(path) == info["sha256"], "Qualification file hash mismatch")
            bindings[str(path)] = info["sha256"]
        bindings[str(sample_path)] = sha256(sample_path)
        samples.append(sample)
    assert_pair(*samples)
    assert_same_camera(samples[-1]["calibration"], plan["expected_calibration"])
    smoke_path = directory/"sensor_smoke/result.json"
    smoke = read_json(smoke_path)
    check_smoke_metadata(smoke)
    for record in smoke["checks"]:
        folder = directory/"sensor_smoke"/record["name"]
        diagnostic = safe_file(folder, "diagnostic.json")
        require(read_json(diagnostic) == record, "Smoke summary differs from recorded diagnostic")
        for relative in ("diagnostic.json", "rgb_diagnostic.png", "native_optical_z_m.npy",
                         "validity_diagnostic.png", "native_instance_ids.npy"):
            path = safe_file(folder, relative)
            bindings[str(path)] = sha256(path)
    bindings[str(smoke_path)] = sha256(smoke_path)
    return dict(bindings=bindings, native_numerical_prerequisite_passed=True,
                visual_approval=False, training_approval=False)


def substitute_plant(stage, original_variant, records, variants, generated_catalogue):
    """Session-only substitution at the EXACT original world transform.

    Surrounding plants/gutters, robot and cameras are not changed. The original
    foreground root is deactivated in this disposable stage, not deleted on disk.
    """
    from pxr import Gf, Usd, UsdGeom
    from .plant_variant_catalogue import assemble_for_inspection
    require(stage.GetRootLayer().anonymous, "Disposable anonymous stage required")
    old_path = original_variant["plant_root"]
    old = stage.GetPrimAtPath(old_path)
    require(bool(old) and old.IsActive(), "Missing active original plant")
    require(original_variant["source_plant_id"] == generated_catalogue["source_family"]
            and original_variant["split_group"] == generated_catalogue["split_group"], "Donor lineage mismatch")
    matching = [r for r in records if r["plant_root"] == old_path]
    require(len(matching) == 1 and sum(v["plant_root"] == old_path for v in variants) == 1,
            "Exactly one foreground record may be replaced")
    world = UsdGeom.XformCache().GetLocalToWorldTransform(old)
    new_path = "/World/GeneratedNativePilot/" + generated_catalogue["variant_id"]
    require(not stage.GetPrimAtPath(new_path), "Generated root already exists")
    replacement = assemble_for_inspection(stage, new_path, generated_catalogue)
    require(set(matching[0]["component_paths"]) == set(replacement["record"]["component_paths"]),
            "Substitution changed component population")
    parent = stage.GetPrimAtPath(new_path).GetParent()
    parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(parent)
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        UsdGeom.Xformable(stage.GetPrimAtPath(new_path)).AddTransformOp().Set(world*parent_world.GetInverse())
        old.SetActive(False)
    actual = UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(new_path))
    require(np.allclose(actual, world, atol=1e-10, rtol=0), "Generated plant placement moved")
    require(not old.IsActive(), "Original plant remains active under replacement")
    new_records = [replacement["record"] if r["plant_root"] == old_path else deepcopy(r) for r in records]
    new_variants = [replacement["variant"] if v["plant_root"] == old_path else deepcopy(v) for v in variants]
    return dict(records=new_records, variants=new_variants, report=replacement["report"],
                old_root=old_path, new_root=new_path, plant_to_world=np.asarray(actual).tolist(),
                component_count_preserved=True, source_files_changed=False)


def assert_pair_fresh(control, generated):
    assert_same_camera(control["calibration"], generated["calibration"])
    require(control["robot_snapshot"] == generated["robot_snapshot"]
            and control["scene_counts"] == generated["scene_counts"]
            and control["lighting"] == generated["lighting"], "Control pair changed robot/surroundings/light")
    require(control["supervision"]["source_target_id"] == generated["supervision"]["source_target_id"]
            and control["supervision"]["target_id"] != generated["supervision"]["target_id"],
            "Control/variant target lineage mismatch")
    require(np.allclose(control["supervision"]["plant_to_world_usd_row_vectors"],
                        generated["supervision"]["plant_to_world_usd_row_vectors"], atol=1e-10, rtol=0),
            "Control pair changed plant placement")
    a, b = control["synchronization"], generated["synchronization"]
    require(b["freshness"]["callback_sequence"] > a["freshness"]["callback_sequence"], "Stale writer callback")
    require(a["static_guard"] != b["static_guard"], "Plant substitution did not change the static scene")
    require(all(a["freshness"][k] != b["freshness"][k] for k in ("rgb_sha256", "depth_sha256")),
            "Native buffers unchanged after plant substitution")
    require(control["native_target_pixels"] > 0 and generated["native_target_pixels"] > 0,
            "Cannot verify changed target identity without visible native pixels")
    require(generated["old_plant_native_pixels"] == 0, "Old plant still present in native buffers")
    require(generated["supervision"]["nominal_world_m"] != control["supervision"]["nominal_world_m"],
            "Variant reused original metric cut label")
    return dict(same_mounted_camera_robot_pose_lighting=True, plant_placement_preserved=True,
                changed_metric_label=True, new_native_target_identity_observed=True,
                callback_and_buffers_changed=True, dynamic_synchronization_supported=False,
                visual_clarity_approved=False, training_approved=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-capture", type=Path, required=True)
    parser.add_argument("--sample", required=True)
    parser.add_argument("--variant", type=Path, required=True)
    parser.add_argument("--prerequisite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "Plan output must be new")
    plan = prepare_plan(args.source_capture, args.sample, args.variant, args.prerequisite)
    check_plan(plan)
    write_json(args.output, plan)
    print("GENERATED_PAIR_PLANNED_NOT_EXECUTED", args.output, flush=True)


if __name__ == "__main__":
    main()
