"""CPU-only, pinned preparation of ONE original target and ONE historical pose."""
from copy import deepcopy
from pathlib import Path
import argparse
import json
import re

import numpy as np

from ..capture_sensor import calibration_for_native_resolution
from ..cut_regions import rule_fingerprint
from .contracts import (SCHEMA, HEAD_CAMERA, RESOLUTION, FROZEN_SPLITS, SCENE_POLICY,
    ADMISSION, require, pin, sha256, read_json, bind_all, merge_bindings, safe_file,
    new_destination, write_new, code_bindings, verify_loaded_code)


def _plan(path, expected, *, clear):
    path = pin(path, expected)
    plan = read_json(path)
    require(plan.get("schema_version") == "greenhouse.grounding_collection_plan.v1"
            and plan.get("state") == "ready_for_synthetic_grounding_capture"
            and plan.get("training_dataset_approved") is False, "Unapproved original grounding plan required")
    require(plan.get("family_assignments") == FROZEN_SPLITS, "Frozen 24-family reservations changed")
    require(plan.get("split_scope") == "target_source_families_only_shared_greenhouse_backdrop_context",
            "Unsupported split scope")
    config = plan["configuration"]
    require(config.get("seed") == 0 and type(config.get("seed")) is int
            and config.get("source_geometry") == "unmodified_native_components"
            and config.get("original_plant_root_z_m") == .9, "Original source geometry/placement required")
    if clear:
        require(config.get("clear_capture") == SCENE_POLICY["profile"]
                and config.get("renderer_mode", SCENE_POLICY["renderer"]) == SCENE_POLICY["renderer"],
                "Existing RTPT clear-profile plan required")
    require(rule_fingerprint(plan["cut_rule"]) == plan["cut_rule_sha256"], "Changed anatomical rule")
    jobs = plan["jobs"]
    require(len({j["job_id"] for j in jobs}) == len(jobs)
            and len({j["plant_family"] for j in jobs}) == len(jobs), "Duplicate source jobs")
    for job in jobs:
        require(job["split"] == FROZEN_SPLITS.get(job["plant_family"]), "Job split differs from frozen family")
        targets = job["targets"]
        require(len({t["target_id"] for t in targets}) == len(targets), "Duplicate original target")
        for row in targets:
            family = job["plant_family"]
            require(row["target_id"] == family + "/" + row["component_id"]
                    and row["variant_id"] == row["source_plant_id"] == row["split_group"] == family,
                    "Non-original target identity")
    return path, plan


def _one(rows, predicate, message):
    found = [r for r in rows if predicate(r)]
    require(len(found) == 1, message)
    return found[0]


def original_report(plan, job):
    """Reconstruct original proposals from the selected donor, not image labels."""
    from ..audit import audit_manifest
    from ..collection_plan import native_rows
    path = Path(job["source_manifest_path"]).resolve()
    package = Path(plan["package"]).resolve()
    require(path == package / "plants/components" / job["plant_family"] / "manifest.json",
            "Original donor manifest path mismatch")
    require(plan["source_bindings_sha256"].get(str(path)) == sha256(path), "Unbound donor manifest")
    report = audit_manifest(path)
    require(report["status"] != "blocked", "Original structural audit failed")
    for component in report["components"].values():
        asset = safe_file(path.parent, component["file"])
        require(plan["source_bindings_sha256"].get(str(asset)) == component["asset_sha256"],
                "Unbound original component")
    rows, _ = native_rows(report, plan["cut_rule"])
    for row in job["targets"]:
        require(row in rows, "Frozen target differs from freshly reconstructed original anatomy")
    return report


def _pose(sample):
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    cal, pose = sample["calibration"], sample["robot_snapshot"]
    require(cal["camera_path"] == HEAD_CAMERA and cal["resolution"] == [848, 408]
            and cal.get("crop_resize") is None
            and cal["depth_convention"] == "optical_axis_z_metres_not_ray_range", "Mounted native legacy pose required")
    for matrix in (pose["robot_root_to_world_usd_row_vectors"], cal["camera_to_world_usd_row_vectors"]):
        a = np.asarray(matrix, dtype=float)
        require(a.shape == (4, 4) and np.isfinite(a).all()
                and np.allclose(a[:, 3], [0, 0, 0, 1], atol=1e-8, rtol=0)
                and np.allclose(a[:3, :3] @ a[:3, :3].T, np.eye(3), atol=1e-7, rtol=0)
                and np.isclose(np.linalg.det(a[:3, :3]), 1, atol=1e-7, rtol=0), "Non-rigid pose prior")
    require(isinstance(pose["joint_degrees"], dict) and pose["joint_degrees"]
            and all(type(v) in (float, int) and np.isfinite(v) for v in pose["joint_degrees"].values()),
            "Finite explicit robot joints required")
    mount = np.asarray(pose["camera_to_head_column_vectors"], float)
    require(mount.shape == (4, 4) and np.isfinite(mount).all()
            and np.allclose(mount[3], [0, 0, 0, 1], atol=1e-8, rtol=0)
            and np.allclose(mount[:3, :3] @ mount[:3, :3].T, np.eye(3), atol=1e-8, rtol=0)
            and np.isclose(np.linalg.det(mount[:3, :3]), 1, atol=1e-8, rtol=0),
            "Rigid homogeneous proper camera-to-head mount required")
    model = Rby1Kinematics()
    expected = {name for name, joint in model._by_name.items() if joint.kind == "revolute"}
    require(len(expected) == 22 and set(pose["joint_degrees"]) == expected,
            "Exact complete Model-A 22-joint arm/torso/head snapshot required")
    # The original static worker holds continuous wheels and prismatic grippers
    # at authored zero; no omitted arm joint may exploit FK's zero default.
    links = model.all_link_transforms(pose["joint_degrees"])
    camera = np.asarray(pose["robot_root_to_world_usd_row_vectors"]).T @ links["link_head_2"] @ mount
    require(np.allclose(camera, np.asarray(cal["camera_to_world_usd_row_vectors"]).T, atol=1e-7, rtol=0),
            "Historical camera disagrees with complete robot FK")
    # Carry physical pose only, never old framing or visibility conclusions.
    return cal, {key: deepcopy(pose[key]) for key in ("joint_degrees",
        "robot_root_to_world_usd_row_vectors", "camera_to_head_column_vectors")}


def _single_plan(clear_plan, *, clear_plan_sha256, target_id, source_capture,
                 source_manifest_sha256, sample_id, source_sample_sha256,
                 allow_other_target_pose=False, reframe_pixel_xy=None, _cache=None):
    """Return a deterministic plan; no directory creation or renderer imports.

    SHA pins are explicit caller trust roots, not signatures. A different original
    target on the SAME donor needs explicit opt-in and receives no old labels.
    """
    require(type(allow_other_target_pose) is bool, "Explicit pose-only target-switch flag required")
    if reframe_pixel_xy is not None:
        require(allow_other_target_pose and isinstance(reframe_pixel_xy, list) and len(reframe_pixel_xy) == 2
                and all(type(v) in (float, int) and np.isfinite(v) and 0 < v < bound
                        for v, bound in zip(reframe_pixel_xy, RESOLUTION)), "Explicit bounded native-pixel head reframing required")
    require(isinstance(sample_id, str) and re.fullmatch(r"sample_[0-9]+", sample_id), "Exact legacy sample ID required")
    cache = {} if _cache is None else _cache
    def once(key, function):
        if key not in cache:
            cache[key] = function()
        return cache[key]
    clear_path, clear = once("clear", lambda: _plan(clear_plan, clear_plan_sha256, clear=True))
    source_capture = Path(source_capture).resolve(strict=True)
    manifest_path = once("manifest_path", lambda: pin(source_capture / "manifest.json", source_manifest_sha256))
    sample_path = pin(safe_file(source_capture, sample_id + "/sample.json"), source_sample_sha256)
    manifest, sample = once("manifest", lambda: read_json(manifest_path)), read_json(sample_path)
    require(manifest.get("state") == "pilot_ready_for_review"
            and manifest.get("source_assets_unchanged") is True
            and manifest.get("source_geometry_modified") is False
            and manifest.get("target_family_split") == "train", "Completed original TRAIN pose source required")
    require(sample.get("schema_version") == "greenhouse.rgbd_pilot_sample.v2"
            and sample.get("sample_id") == sample_id, "Exact original sample required")
    recorded = _one(manifest["samples"], lambda s: s["sample_id"] == sample_id, "Missing/duplicate source sample")
    sync = sample["synchronization"]
    require(sync.get("scene_unchanged_during_capture") is True
            and sync.get("dynamic_recording_supported") is False, "Frozen historical snapshot required")
    old_path, old = once("old_plan", lambda: _plan(manifest["source_collection_plan_path"],
        manifest["source_collection_plan_sha256"], clear=False))
    require(old["source_bindings_sha256"] == clear["source_bindings_sha256"]
            and Path(old["package"]).resolve() == Path(clear["package"]).resolve()
            and old["cut_rule"] == clear["cut_rule"], "Pose source and clear plan differ in assets/anatomical rule")
    old_job = _one(old["jobs"], lambda j: j["job_id"] == manifest["collection_job_id"], "Missing original source job")
    family = old_job["plant_family"]
    require(FROZEN_SPLITS.get(family) == old_job["split"] == "train", "TRAIN original donor required")
    job = _one(clear["jobs"], lambda j: j["plant_family"] == family, "Missing exact clear-plan donor")
    row = _one(job["targets"], lambda r: r["target_id"] == target_id, "Requested target is not an exact original donor target")
    sup = sample["supervision"]
    prior_row = _one(old_job["targets"], lambda r: r["target_id"] == sup["target_id"], "Historical target absent from original plan")
    require(sup["split_group"] == sup["variant_id"] == family
            and sup["review_id"] == prior_row["draft_id"] == recorded["target_review_id"]
            and sup["cut_region_proposal"] == prior_row["cut_region_proposal"], "Historical source anatomy mismatch")
    require(target_id == prior_row["target_id"] or allow_other_target_pose, "Different target requires explicit pose-only opt-in")
    cal, pose = _pose(sample)
    variants = manifest["variants"]
    require(len({v["plant_root"] for v in variants}) == len(variants), "Duplicate foreground placement")
    for v in variants:
        require(v["variant_id"] == v["source_plant_id"] == v["split_group"]
                and v["source_plant_id"] in FROZEN_SPLITS and not v["added_components"]
                and not v["added_component_paths"] and v["source_geometry_modified"] is False,
                "Historical scene contains generated foreground components")
    variant = _one(variants, lambda v: v["source_plant_id"] == family, "Exact original donor placement required")
    payloads = {str(safe_file(sample_path.parent, name)): value["sha256"] for name, value in sample["files"].items()}
    require({"inputs/rgb.png", "inputs/depth_m.npy", "supervision/renderer_instance_id.npy"} <= sample["files"].keys(),
            "Historical native buffers must be bound even though never admitted")
    # Shared textures and daylight implementation are not included in old plan USD bindings.
    package = Path(clear["package"]).resolve()
    extra = once("appearance", lambda: appearance_bindings(package))
    bindings = merge_bindings(clear["source_bindings_sha256"], manifest["source_usd_sha256"], payloads, extra,
        {str(clear_path): clear_plan_sha256, str(old_path): manifest["source_collection_plan_sha256"],
         str(manifest_path): source_manifest_sha256, str(sample_path): source_sample_sha256})
    once("original_report", lambda: original_report(clear, job))
    result = dict(schema=SCHEMA, state="prepared_original_pending_fresh_native_capture",
        source_collection_plan=str(clear_path), source_collection_plan_sha256=clear_plan_sha256,
        collection_job_id=job["job_id"], package=str(package), target_id=target_id,
        source_family=family, split="train", split_group=family, conservative_view_cap_group=target_id,
        source_row=deepcopy(row), original_variant=deepcopy(variant), scene_variants=deepcopy(variants),
        pose_prior=dict(role="historical_pose_only_never_training_observation", source_capture=str(source_capture),
            source_manifest_sha256=source_manifest_sha256, sample_id=sample_id, source_sample_sha256=source_sample_sha256,
            prior_target_id=prior_row["target_id"], prior_plan=str(old_path), prior_lighting=manifest["lighting"],
            allow_other_target_pose=allow_other_target_pose, historical_review_inherited=False),
        expected_robot_snapshot=deepcopy(pose), prior_calibration=deepcopy(cal),
        pose_request=dict(mode="exact_prior" if reframe_pixel_xy is None else "reframe_head_only",
                          native_pixel_xy=deepcopy(reframe_pixel_xy)),
        expected_plant_to_world=deepcopy(sup["plant_to_world_usd_row_vectors"]),
        expected_calibration=calibration_for_native_resolution(cal, RESOLUTION),
        expected_scene_counts=manifest["scene_counts"], excluded_external_roots=manifest["unbundled_external_prop_roots_excluded"],
        scene_policy=deepcopy(SCENE_POLICY), family_assignments=deepcopy(FROZEN_SPLITS),
        geometry_mode="unmodified_original", sample_count_limit=1, resolution=list(RESOLUTION),
        historical_lighting_reproduction_claimed=False, source_bindings=bindings,
        implementation_bindings=once("code", code_bindings), admission=deepcopy(ADMISSION))
    return result


def appearance_bindings(package):
    from greenhouse_sim.robot_kinematics import DEFAULT_URDF
    extra = {str(p.resolve()): sha256(p) for p in (package / "env_panel").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts}
    for p in package.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".exr", ".hdr", ".mdl", ".mtlx"}:
            extra[str(p.resolve())] = sha256(p)
    extra[str(Path(DEFAULT_URDF).resolve())] = sha256(DEFAULT_URDF)
    return extra


CASE_FIELDS = ("target_id", "source_row", "conservative_view_cap_group", "pose_prior",
               "expected_robot_snapshot", "prior_calibration", "expected_calibration", "pose_request")


def prepare_batch(clear_plan, *, clear_plan_sha256, source_capture, source_manifest_sha256, cases):
    """One donor/capture/scene, 1..64 exact target/pose cases; hashes shared per call.

    Each case contains target_id, sample_id, source_sample_sha256 and optionally
    allow_other_target_pose. Identical physical poses cannot claim extra frames.
    No public verification bypass or trusted cache is accepted.
    """
    from .contracts import canonical, digest
    require(isinstance(cases, list) and 1 <= len(cases) <= 64, "Explicit batch bound is 1..64 cases")
    verify_loaded_code()
    cache, prepared, poses = {}, [], set()
    for item in cases:
        require(isinstance(item, dict) and {"target_id", "sample_id", "source_sample_sha256"} <= item.keys()
                and not item.keys() - {"target_id", "sample_id", "source_sample_sha256", "allow_other_target_pose", "reframe_pixel_xy"},
                "Unknown or missing case request fields")
        plan = _single_plan(clear_plan, clear_plan_sha256=clear_plan_sha256,
            source_capture=source_capture, source_manifest_sha256=source_manifest_sha256,
            **item, _cache=cache)
        pose_key = digest(canonical([plan["expected_calibration"]["camera_to_world_usd_row_vectors"],
            plan["pose_request"], plan["target_id"] if plan["pose_request"]["mode"] != "exact_prior" else None]))
        require(pose_key not in poses, "Duplicate camera pose cannot claim another native image")
        poses.add(pose_key)
        prepared.append(plan)
    plan = prepared[0]
    for other in prepared[1:]:
        for key in ("source_family", "original_variant", "scene_variants", "expected_scene_counts",
                    "expected_plant_to_world", "excluded_external_roots", "scene_policy"):
            require(other[key] == plan[key], "Batch escaped the unchanged original donor scene")
    bindings = merge_bindings(*(p["source_bindings"] for p in prepared))
    # Full byte verification once for the union, not once per target.
    bind_all(bindings)
    plan = dict(plan, source_bindings=bindings, sample_count_limit=len(cases),
        cases=[dict(case_id=f"sample_{i:04d}", **{key: p[key] for key in CASE_FIELDS})
               for i, p in enumerate(prepared, 1)],
        batch_request=deepcopy(cases), stage_reuse="one_original_donor_scene_one_native_product")
    clear = cache["clear"][1]
    family_job = next(j for j in clear["jobs"] if j["plant_family"] == plan["source_family"])
    recorded = cache["manifest"]["samples"]
    plan["coverage"] = dict(
        frozen_train_target_rows=sum(len(j["targets"]) for j in clear["jobs"] if j["split"] == "train"),
        donor_original_target_rows=len(family_job["targets"]),
        manifest_listed_same_target_priors={row["target_id"]: sum(s["target_review_id"] == row["draft_id"] for s in recorded)
                                            for row in family_job["targets"]},
        requested_cases=len(cases), requested_original_targets=len({p["target_id"] for p in prepared}),
        selected_same_target_priors=sum(p["target_id"] == p["pose_prior"]["prior_target_id"] for p in prepared),
        selected_changed_target_priors=sum(p["target_id"] != p["pose_prior"]["prior_target_id"] for p in prepared),
        native_reachability_or_clarity_measured=False,
        unselected_manifest_entries_individually_verified=False)
    bind_all(plan["implementation_bindings"])
    verify_loaded_code()
    return plan


def prepare_plan(clear_plan, *, clear_plan_sha256, target_id, source_capture,
                 source_manifest_sha256, sample_id, source_sample_sha256, allow_other_target_pose=False, reframe_pixel_xy=None):
    """Default one-case pilot; same contract and validator as bounded expansion."""
    return prepare_batch(clear_plan, clear_plan_sha256=clear_plan_sha256,
        source_capture=source_capture, source_manifest_sha256=source_manifest_sha256,
        cases=[dict(target_id=target_id, sample_id=sample_id, source_sample_sha256=source_sample_sha256,
                    allow_other_target_pose=allow_other_target_pose, reframe_pixel_xy=reframe_pixel_xy)])


def case_plan(plan, case):
    require(case in plan["cases"], "Unscheduled original target/pose")
    return dict(plan, **{key: case[key] for key in CASE_FIELDS})


def check_plan(plan):
    require(plan.get("schema") == SCHEMA, "Unknown original-only plan schema")
    prior = plan["pose_prior"]
    expected = prepare_batch(plan["source_collection_plan"], clear_plan_sha256=plan["source_collection_plan_sha256"],
        source_capture=prior["source_capture"], source_manifest_sha256=prior["source_manifest_sha256"],
        cases=plan["batch_request"])
    require(plan == expected, "Original plan fields, policy, ancestry or code bindings changed")
    return True


def protected_roots(plan):
    return [plan["package"], plan["pose_prior"]["source_capture"],
            str(Path(plan["source_collection_plan"]).parent), str(Path(plan["pose_prior"]["prior_plan"]).parent),
            *[str(Path(p).parent) for p in plan["implementation_bindings"]],
            *[str(Path(p).parent) for p in plan["source_bindings"]]]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("clear-plan", "clear-plan-sha256", "source-capture", "source-manifest-sha256", "output"):
        p.add_argument("--" + name, required=True)
    for name in ("target-id", "sample-id", "source-sample-sha256", "cases", "cases-sha256"):
        p.add_argument("--" + name)
    p.add_argument("--allow-other-target-pose", action="store_true")
    p.add_argument("--reframe-pixel-xy", type=float, nargs=2)
    args = vars(p.parse_args(argv))
    output = args.pop("output")
    cases, cases_hash = args.pop("cases"), args.pop("cases_sha256")
    if cases:
        require(not any(args[k] for k in ("target_id", "sample_id", "source_sample_sha256", "allow_other_target_pose", "reframe_pixel_xy")),
                "Choose cases file OR single-case arguments")
        for key in ("target_id", "sample_id", "source_sample_sha256", "allow_other_target_pose", "reframe_pixel_xy"):
            args.pop(key)
        plan = prepare_batch(**args, cases=read_json(pin(cases, cases_hash)))
    else:
        require(cases_hash is None, "Cases hash without cases file")
        plan = prepare_plan(**args)
    output = new_destination(output, protected_roots(plan))
    output.parent.mkdir(parents=True, exist_ok=True)
    write_new(output, plan)
    print(json.dumps(dict(plan=str(output), plan_sha256=sha256(output), target_id=plan["target_id"], native_launch=False)))


if __name__ == "__main__":
    main()
