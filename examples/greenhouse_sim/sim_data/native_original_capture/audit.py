"""Automatic replay of fresh original native buffers; never global release approval."""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import argparse
import json

import numpy as np
from PIL import Image

from ..capture_contract import fingerprint, project, transform_points
from ..capture_visibility import component_masks, ORGAN_IDS
from ..native_sensor_payload import decode_native_instances
from ..native_greenhouse_pair import assert_same_camera
from ..native_clear_labels import derive
from ..automated_native_review import check_native_evidence, trace_review, POLICY as TRACE_POLICY
from ..native_clear_contract import contract_hash
from .contracts import (SAMPLE_SCHEMA, RESULT_SCHEMA, AUDIT_SCHEMA, RESOLUTION, ADMISSION,
    require, pin, sha256, digest, read_json, safe_file, bind_all, new_destination, write_new)
from .prepare import check_plan, original_report, protected_roots, case_plan


def image_array(path):
    with Image.open(path) as im:
        return np.asarray(im).copy()


def verify_camera(metadata):
    """Native-size adaptation of dataset_review's mounted FK/projection checks."""
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    cal, pose = metadata["calibration"], metadata["robot_snapshot"]
    root = np.asarray(pose["robot_root_to_world_usd_row_vectors"]).T
    links = Rby1Kinematics().all_link_transforms(pose["joint_degrees"])
    expected = root @ links["link_head_2"] @ np.asarray(pose["camera_to_head_column_vectors"])
    observed = np.asarray(cal["camera_to_world_usd_row_vectors"])
    require(np.allclose(expected, observed.T, atol=1e-7, rtol=0), "Fresh camera disagrees with mounted FK")
    params = metadata["rendered_camera_params"]
    require(params["renderProductResolution"] == RESOLUTION and params["metersPerSceneUnit"] == 1,
            "Native renderer dimensions/units changed")
    require(np.allclose(np.asarray(params["cameraViewTransform"]).reshape(4, 4), np.linalg.inv(observed),
                        atol=5e-5, rtol=0), "Native renderer camera differs from FK")
    probes = np.asarray([[0, 0, -1], [.1, .15, -1.2], [-.2, -.1, -2]])
    clip = np.column_stack((probes, np.ones(3))) @ np.asarray(params["cameraProjection"]).reshape(4, 4)
    require(np.isfinite(clip).all() and np.all(np.abs(clip[:, 3]) > 1e-8), "Invalid native camera projection")
    pixels = (clip[:, :2] / clip[:, 3, None] * [1, -1] + 1) * (np.asarray(RESOLUTION) / 2)
    projected = [p["pixel_xy"] for p in project(transform_points(probes, observed), cal)]
    require(np.allclose(pixels, projected, atol=.01, rtol=0), "Native projection differs from original optics")


def validate_catalogue(plan, catalogue, reports):
    expected = {}
    for variant in plan["scene_variants"]:
        family = variant["variant_id"]
        components = reports[family]["components"]
        for key, component in components.items():
            chain, parent = [key], component["parent"]
            while parent is not None:
                require(parent in components and parent not in chain, "Invalid original component ancestry")
                chain.append(parent)
                parent = components[parent]["parent"]
            path = variant["plant_root"] + "/" + "/".join(reversed(chain))
            expected[path] = dict(component_id=key, prim_path=path, organ_type=component["type"],
                variant_id=family, source_plant_id=family, split_group=family)
    rows = [dict(expected[path], component_index=i) for i, path in enumerate(sorted(expected), 1)]
    require(catalogue == rows and len(rows) == plan["expected_scene_counts"]["components"],
            "Saved catalogue differs from complete original component hierarchy")


def verify_pose_request(plan, metadata):
    require(metadata["pose_prior"] == plan["pose_prior"] and metadata["pose_request"] == plan["pose_request"],
            "Pose provenance changed")
    prior, pose = plan["expected_robot_snapshot"], metadata["robot_snapshot"]
    cal = deepcopy(metadata["calibration"])
    if plan["pose_request"]["mode"] == "exact_prior":
        require(pose == prior, "Exact prior pose changed")
    else:
        require(set(pose) == set(prior), "Unexpected pose fields")
        for key in prior:
            if key != "joint_degrees":
                require(pose[key] == prior[key], "Head reframing changed root or camera mount")
        a, b = pose["joint_degrees"], prior["joint_degrees"]
        require(a.keys() == b.keys() and all(a[k] == b[k] for k in b if k not in ("head_0", "head_1")),
                "Head reframing changed arm/torso joints")
        nominal = metadata["supervision"]["nominal_projected"]
        require(nominal["projection_status"] == "in_frame"
                and np.linalg.norm(np.asarray(nominal["pixel_xy"]) - plan["pose_request"]["native_pixel_xy"]) < 4,
                "Reframed target differs from requested native pixel")
        cal["camera_to_world_usd_row_vectors"] = plan["expected_calibration"]["camera_to_world_usd_row_vectors"]
    assert_same_camera(cal, plan["expected_calibration"])


def review_arrays(plan, metadata, report, rgb, depth, valid, components, target_mask, catalogue):
    """Pure fresh-native gate; synthetic unit arrays are never capture evidence."""
    require(metadata.get("schema_version") == SAMPLE_SCHEMA
            and metadata.get("state") == "fresh_original_native_pending_automatic_audit"
            and metadata.get("training_sample_approved") is False
            and metadata.get("historical_labels_inherited") is False, "Not a fresh original native sample")
    verify_pose_request(plan, metadata)
    require(metadata["scene_counts"] == plan["expected_scene_counts"]
            and metadata["geometry_screen"]["passed"] is True
            and metadata["renderer"] == "RealTimePathTracing"
            and metadata["native_instance_backend"] == "legacy", "Scene/geometry/renderer evidence mismatch")
    lighting = metadata["lighting"]
    require(lighting["day"] == 172 and lighting["minutes"] == 780 and lighting["dome_intensity"] == 6000,
            "Wrong clear lighting profile")
    require(metadata["input_policy"] == dict(clean_full_scene=True, isolation=False, diagnostic_overlays=False),
            "Unsafe observation scope")
    require(metadata["synchronization"]["render_budget_subframes"] == 56, "Unqualified render budget")
    sup = metadata["supervision"]
    require(sup["target_id"] == sup["source_target_id"] == sup["conservative_view_cap_group"] == plan["target_id"]
            and sup["split_group"] == plan["source_family"]
            and sup["cut_region_proposal"] == plan["source_row"]["cut_region_proposal"]
            and np.allclose(sup["plant_to_world_usd_row_vectors"], plan["expected_plant_to_world"], atol=1e-9, rtol=0),
            "Transplanted original anatomy or cap group")
    require(rgb.shape == (816, 1696, 3) and rgb.dtype == np.uint8
            and depth.shape == valid.shape == components.shape == (816, 1696)
            and depth.dtype == np.float32 and valid.dtype == bool and components.dtype == np.uint32,
            "Exact native buffers required")
    check_native_evidence(metadata, rgb, depth, components, target_mask, catalogue)
    label = derive(metadata, report, rgb, depth, valid, components, catalogue)
    trace = trace_review(metadata, report, label, rgb, depth, valid, components, catalogue) if label["eligible"] else None
    decision = ("exclude_geometry_or_visibility" if not label["eligible"] else
                "accept_strict_automatic_annotation_candidate" if trace["passed"] else "hold_visual_clarity")
    return label, trace, decision


def _audit_sample(capture, plan, record, reports):
    require(record["target_id"] == plan["target_id"] and record["source_assets_unchanged"] is True
            and record["admission"] == ADMISSION, "Original result identity changed")
    folder = safe_file(capture, record["case_id"])
    metadata = read_json(pin(folder / "sample.json", record["sample_sha256"]))
    require(metadata["sample_id"] == record["case_id"]
            and metadata["scene_counts"] == record["scene_counts"]
            and metadata["geometry_screen"] == record["geometry_screen"]
            and metadata["lighting"] == record["lighting"], "Capture/sample evidence differs")
    required = {"inputs/rgb.png", "inputs/depth_m.npy", "inputs/depth_valid.png",
                "supervision/renderer_instance_id.npy", "supervision/component_id.npy", "supervision/identities.json",
                "supervision/target_visible.png", "supervision/organ_type.png"}
    require(required <= metadata["files"].keys(), "Missing native supervision payload")
    for name, value in metadata["files"].items():
        pin(safe_file(folder, name), value["sha256"])
    rgb = image_array(folder / "inputs/rgb.png")
    depth = np.load(folder / "inputs/depth_m.npy", allow_pickle=False)
    raw_valid = image_array(folder / "inputs/depth_valid.png")
    require(raw_valid.dtype == np.uint8 and set(np.unique(raw_valid)) <= {0, 255}, "Invalid native validity image")
    ids = np.load(folder / "supervision/renderer_instance_id.npy", allow_pickle=False)
    identities = read_json(folder / "supervision/identities.json")
    require(identities["organ_ids"] == ORGAN_IDS, "Organ mapping changed")
    ids, mapping = decode_native_instances(dict(instance_id_segmentation=dict(data=ids,
        info=dict(idToLabels=identities["renderer_id_to_prim"]))), RESOLUTION)
    require(digest(ids.tobytes()) == metadata["native_instance_sha256"]
            and fingerprint(mapping) == metadata["native_mapping_sha256"], "Native ID callback fingerprint differs")
    catalogue = identities["component_catalogue"]
    validate_catalogue(plan, catalogue, reports)
    components, organs, _ = component_masks(ids, mapping, catalogue)
    require(np.array_equal(components, np.load(folder / "supervision/component_id.npy", allow_pickle=False))
            and np.array_equal(organs, image_array(folder / "supervision/organ_type.png")), "Native ID-derived masks differ")
    verify_camera(metadata)
    label, trace, decision = review_arrays(plan, metadata, reports[plan["source_family"]], rgb, depth,
        raw_valid == 255, components, image_array(folder / "supervision/target_visible.png"), catalogue)
    require(label == read_json(pin(folder / "supervision/label.json", record["label_sha256"]))
            and record["label_reason"] == label["reason"], "Stale stored label")
    trace_path = folder / "supervision/query_trace.json"
    if trace is None:
        require(record["trace_sha256"] is None and not trace_path.exists(), "Unexpected inherited query trace")
    else:
        require(trace == read_json(pin(trace_path, record["trace_sha256"])), "Stale stored strict query trace")
    return dict(case_id=record["case_id"], decision=decision, target_id=plan["target_id"],
        source_family=plan["source_family"], split="train", conservative_view_cap_group=plan["target_id"],
        reason=label["reason"], clarity_reasons=label.get("clarity", {}).get("reasons"),
        trace_reasons=trace["reasons"] if trace else None,
        sample_sha256=record["sample_sha256"], rgb_sha256=metadata["files"]["inputs/rgb.png"]["sha256"],
        label_sha256=record["label_sha256"], trace_sha256=record["trace_sha256"],
        label_replayed_exact=True, trace_replayed_exact=trace is not None,
        native_callback_hashes_verified=True, native_ID_masks_replayed=True,
        admission=deepcopy(ADMISSION), historical_image_or_review_promoted=False)


def audit_records(capture, plan, records):
    """Shared source verification/report reconstruction once per batch, not per frame."""
    capture = Path(capture).resolve()
    check_plan(plan)
    require([r["case_id"] for r in records] == [c["case_id"] for c in plan["cases"]], "Missing/reordered/duplicate cases")
    clear = read_json(plan["source_collection_plan"])
    families = {v["source_plant_id"] for v in plan["scene_variants"]}
    reports = {j["plant_family"]: original_report(clear, j) for j in clear["jobs"] if j["plant_family"] in families}
    reviewed, seen_rgb = [], set()
    for case, record in zip(plan["cases"], records):
        require(record["admission"] == ADMISSION and record["target_id"] == case["target_id"], "Case ancestry/admission changed")
        if record["state"] == "held_pre_render":
            require(isinstance(record["reason"], str) and record["reason"]
                    and not (capture / case["case_id"]).exists(), "Held case contains an unexplained capture")
            reviewed.append(dict(case_id=case["case_id"], target_id=case["target_id"],
                                 decision="held_pre_render", reason=record["reason"], admission=deepcopy(ADMISSION)))
            continue
        require(record["state"] == "native_captured", "Unknown native case state")
        result = _audit_sample(capture, case_plan(plan, case), record, reports)
        require(result["rgb_sha256"] not in seen_rgb, "Duplicate native image within batch")
        seen_rgb.add(result["rgb_sha256"])
        reviewed.append(result)
    bind_all(plan["source_bindings"])
    bind_all(plan["implementation_bindings"])
    return dict(schema=AUDIT_SCHEMA, records=reviewed, counts=dict(Counter(r["decision"] for r in reviewed)),
        review_method="fresh_saved_native_anatomy_ID_Z_clarity_query_trace_replay",
        contract_sha256=contract_hash(), trace_policy=TRACE_POLICY, admission=deepcopy(ADMISSION),
        physical_execution_approved=False, worker_exit_must_be_checked_by_launcher=True)


def verify_smoke(capture, bindings):
    from ..generated_capture import check_smoke_metadata
    root = Path(capture).resolve()
    names = ("pose_0", "pose_1", "occluder_visible", "occluder_hidden")
    files = ("diagnostic.json", "rgb_diagnostic.png", "native_optical_z_m.npy", "validity_diagnostic.png", "native_instance_ids.npy")
    expected = {"sensor_smoke/result.json"} | {"sensor_smoke/" + n + "/" + f for n in names for f in files}
    require(set(bindings) == expected, "Incomplete or expanded native smoke bindings")
    for name, value in bindings.items():
        pin(safe_file(root, name), value)
    smoke = read_json(root / "sensor_smoke/result.json")
    check_smoke_metadata(smoke)
    for record in smoke["checks"]:
        folder = root / "sensor_smoke" / record["name"]
        require(read_json(folder / "diagnostic.json") == record, "Smoke summary differs from saved diagnostic")
        z = np.load(folder / "native_optical_z_m.npy", allow_pickle=False)
        ids = np.load(folder / "native_instance_ids.npy", allow_pickle=False)
        rgb, valid = image_array(folder / "rgb_diagnostic.png"), image_array(folder / "validity_diagnostic.png")
        require(z.shape == ids.shape == valid.shape == (816, 1696) and z.dtype == np.float32
                and ids.dtype == np.uint32 and rgb.shape == (816, 1696, 3)
                and rgb.dtype == valid.dtype == np.uint8 and set(np.unique(valid)) <= {0, 255}, "Malformed smoke native buffers")
        token = record["freshness"]
        require(token["camera_sha256"] == fingerprint(record["calibration"])
                and token["rgb_sha256"] == digest(rgb.tobytes()) and token["depth_sha256"] == digest(z.tobytes()),
                "Smoke native callback fingerprints differ")
        for probe in record["known_surface"]:
            x, y = probe["pixel_xy"]
            require(valid[y, x] == 255 and float(z[y, x]) == probe["native_z_m"],
                    "Smoke optical-Z probes differ from saved native buffers")
    return True


def audit_capture(capture, *, result_sha256):
    capture = Path(capture).resolve()
    require(not (capture / "failure.json").exists(), "Failed capture cannot be audited as successful")
    result = read_json(pin(capture / "result.json", result_sha256))
    require(result.get("schema") == RESULT_SCHEMA
            and result.get("state") == "original_native_capture_complete_automatically_audited"
            and result["admission"] == ADMISSION, "Incomplete original-only result")
    request = read_json(pin(capture / "request.json", result["request_sha256"]))
    require(request["plan_sha256"] == result["plan_sha256"] and request["schema"] == RESULT_SCHEMA
            and request["host_memory_preflight"]["allowed"] is True
            and request["process_admission"]["no_unrelated_kit_process_at_admission"] is True,
            "Unbound plan or missing native launch admission")
    plan = read_json(pin(request["plan_path"], request["plan_sha256"]))
    require(result["capture"]["source_assets_unchanged"] is True
            and result["capture"]["stage_count"] == result["capture"]["greenhouse_render_product_count"] == 1,
            "Unexpected native execution scope")
    verify_smoke(capture, result["smoke_bindings"])
    automatic = audit_records(capture, plan, result["capture"]["records"])
    require(automatic == read_json(pin(capture / "automatic_audit.json", result["automatic_audit_sha256"])),
            "Automatic audit changed on replay")
    pin(capture / "result.json", result_sha256)
    return automatic


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--capture", required=True)
    p.add_argument("--result-sha256", required=True)
    p.add_argument("--output", help="Optional NEW audit receipt; omitted means read-only replay")
    a = p.parse_args(argv)
    receipt = audit_capture(a.capture, result_sha256=a.result_sha256)
    if a.output:
        request = read_json(Path(a.capture) / "request.json")
        plan = read_json(request["plan_path"])
        output = new_destination(a.output, [a.capture, Path(request["plan_path"]).parent, *protected_roots(plan)])
        output.parent.mkdir(parents=True, exist_ok=True)
        write_new(output, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
