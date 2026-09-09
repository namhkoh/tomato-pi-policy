"""Offline, hash-bound engineering review of native robot-head RGB-D pilots.

No rendering, robot connection, source edits, human approval or training export.
Run with Isaac Python for the CPU-only USD mount/asset check.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

from .capture_contract import project, transform_points
from .capture_visibility import component_masks, interval_visibility, view_quality, ORGAN_IDS
from .cut_regions import rule_fingerprint
from .depth_preview import colour_depth, sha256

OBSERVATIONS = ("inputs/rgb.png", "inputs/depth_m.npy", "inputs/depth_valid.png")
HEAD_CAMERA = "/World/RBY1/link_head_2/attachments/HeadCamera/D405/DepthCamera"
SCHEMA = "greenhouse.rgbd_engineering_audit.v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def safe_file(root, relative):
    require(isinstance(relative, str) and relative and "\\" not in relative,
            "Expected a relative POSIX artifact path")
    parts = relative.split("/")
    require(all(p not in ("", ".", "..") and ":" not in p for p in parts), "Unsafe artifact path")
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()) and path.is_file(), "Artifact missing or outside source")
    return path


def remember(bindings, path, expected=None):
    path = Path(path).resolve()
    actual = sha256(path)
    require(expected is None or actual == expected, f"Source hash mismatch: {path}")
    bindings[str(path)] = actual


def verify_bindings(bindings):
    require(bool(bindings), "Missing immutable source bindings")
    for path, expected in bindings.items():
        require(sha256(path) == expected, f"Stale audit/review: {path}")


def reference_robot():
    """Rebuild ONLY the robot in an in-memory USD, never touch the live Kit stage."""
    from pxr import Usd, UsdGeom
    from greenhouse_sim.robot_kinematics import Rby1Kinematics, DEFAULT_URDF
    from .capture_scene import calibration, mounted_camera_to_head
    from .robot_preview import add_robot_preview
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetEditTarget(stage.GetSessionLayer())
    add_robot_preview(stage, gutter_x=-.2, right_tool="gripper")
    return Rby1Kinematics(), mounted_camera_to_head(stage), calibration(stage), DEFAULT_URDF


def verify_robot_camera(metadata, model, expected_mount, expected_calibration):
    cal, robot = metadata["calibration"], metadata["robot_snapshot"]
    require(cal["camera_path"] == HEAD_CAMERA, "Reject non-mounted/cinematic camera")
    require(cal["resolution"] == [848, 408] and cal.get("crop_resize") is None,
            "Expected uncropped 848x408 robot sensor input")
    for key in ("intrinsics", "clipping_range_m"):
        require(np.allclose(cal[key], expected_calibration[key], rtol=0, atol=1e-7),
                f"Camera optics changed: {key}")
    mount = np.asarray(robot["camera_to_head_column_vectors"])
    require(np.allclose(mount, expected_mount, rtol=0, atol=1e-8), "Camera mount changed")
    links = model.all_link_transforms(robot["joint_degrees"])  # Validates every URDF joint limit.
    root = np.asarray(robot["robot_root_to_world_usd_row_vectors"]).T
    require(root.shape == (4, 4) and np.isfinite(root).all()
            and np.allclose(root[3], [0, 0, 0, 1], rtol=0, atol=1e-8)
            and np.allclose(root[:3, :3].T @ root[:3, :3], np.eye(3), rtol=0, atol=1e-8)
            and np.isclose(np.linalg.det(root[:3, :3]), 1, rtol=0, atol=1e-8), "Non-rigid robot root")
    expected = root @ links["link_head_2"] @ expected_mount
    observed = np.asarray(cal["camera_to_world_usd_row_vectors"]).T
    require(np.allclose(expected, observed, rtol=0, atol=1e-7), "Camera pose disagrees with robot FK")
    rendered = np.asarray(metadata["rendered_camera_params"]["cameraViewTransform"]).reshape(4, 4)
    require(np.allclose(rendered, np.linalg.inv(observed.T), rtol=0, atol=5e-5),
            "Renderer view disagrees with robot camera")
    params = metadata["rendered_camera_params"]
    require(params["renderProductResolution"] == [848, 408] and params["metersPerSceneUnit"] == 1,
            "Renderer resolution/units mismatch")
    probes = np.asarray([[0, 0, -1], [.1, .15, -1.2], [-.2, -.1, -2]])
    clip = np.column_stack((probes, np.ones(3))) @ np.asarray(params["cameraProjection"]).reshape(4, 4)
    require(np.isfinite(clip).all() and np.all(np.abs(clip[:, 3]) > 1e-8), "Invalid renderer projection")
    pixels = (clip[:, :2] / clip[:, 3, None] * [1, -1] + 1) * [424, 204]
    expected_pixels = [p["pixel_xy"] for p in project(transform_points(probes, observed.T), cal)]
    require(np.allclose(pixels, expected_pixels, rtol=0, atol=.01), "Renderer projection disagrees with optics")
    return {"camera_path": HEAD_CAMERA, "mounted_robot_pov_verified": True,
            "fk_position_error_m": float(np.linalg.norm(expected[:3, 3] - observed[:3, 3])),
            "fk_matrix_max_error": float(np.max(np.abs(expected - observed))),
            "mount_matches_reconstructed_robot": True, "urdf_joint_limits_checked": True,
            "view_source": "simulated_robot_pose_not_live_lab_robot",
            "base_world_m": root[:3, 3].tolist(), "camera_world_m": observed[:3, 3].tolist(),
            "head_joint_degrees": {k: v for k, v in robot["joint_degrees"].items() if k.startswith("head_")},
            "navigation_and_arm_reachability_validated": False}


def check_sample(directory, metadata, draft, model, mount, reference_cal):
    """Recompute geometry, projection, all masks and visibility from saved native buffers."""
    cal, sup = metadata["calibration"], metadata["supervision"]
    require(metadata["state"] == "complete_pending_review" and metadata["training_sample_approved"] is False,
            "Expected immutable, unapproved pilot sample")
    require(metadata["input_policy"] == {"clean_full_scene": True, "diagnostic_overlays": False,
            "isolation": False, "allowed_observation_files": list(OBSERVATIONS)}, "Unsafe observation policy")
    require(sup["human_cut_approval"] is False and sup["physics_validated"] is False
            and sup["executable_trajectory"] is None and sup["grasp_region"] is None,
            "Prototype sample must not imply human/physical/trajectory approval")
    camera = verify_robot_camera(metadata, model, mount, reference_cal)
    for key in ("target_id", "variant_id", "split_group", "cut_region_proposal"):
        require(sup[key] == draft[key], f"Source draft mismatch: {key}")
    proposal = sup["cut_region_proposal"]
    require(proposal["status"] == "proposed_geometry_only" and not proposal["geometry_warnings"],
            "Unresolved cut geometry")
    world = transform_points([proposal["nominal"]["point_plant_m"]], sup["plant_to_world_usd_row_vectors"])
    require(np.allclose(world[0], sup["nominal_world_m"], rtol=0, atol=1e-10), "World cut geometry mismatch")
    samples = proposal["accepted_centerline_interval"]["samples"]
    points = []
    for a, b in zip(samples, samples[1:]):
        steps = max(1, int(np.ceil((b["arc_distance_m"] - a["arc_distance_m"]) / .001)))
        points.extend(np.asarray(a["point_plant_m"]) * (1-t) + np.asarray(b["point_plant_m"]) * t
                      for t in np.linspace(0, 1, steps, endpoint=False))
    points.append(samples[-1]["point_plant_m"])
    interval_world = transform_points(points, sup["plant_to_world_usd_row_vectors"])
    require(np.allclose(interval_world, sup["interval_world_m"], rtol=0, atol=1e-10), "Interval geometry mismatch")
    projected = project(world, cal)[0]
    interval = project(interval_world, cal)
    require(projected == sup["nominal_projected"] and interval == sup["projected_interval"],
            "Saved cut projection disagrees with calibration")
    rgb = np.asarray(Image.open(directory / OBSERVATIONS[0]))
    depth = np.load(directory / OBSERVATIONS[1], allow_pickle=False)
    saved_valid = np.asarray(Image.open(directory / OBSERVATIONS[2]))
    require(rgb.shape == (408, 848, 3) and rgb.dtype == np.uint8, "RGB shape/type mismatch")
    require(depth.shape == (408, 848) and depth.dtype == np.float32, "Native depth shape/type mismatch")
    require(cal["depth_convention"] == "optical_axis_z_metres_not_ray_range", "Wrong depth convention")
    near, far = cal["clipping_range_m"]
    valid = np.isfinite(depth) & (depth > 0) & (depth >= near) & (depth <= far)
    require(np.array_equal(saved_valid, valid.astype(np.uint8)*255), "Depth validity mask mismatch")
    identities = read_json(directory / "supervision/identities.json")
    require(identities["organ_ids"] == ORGAN_IDS and identities["ids_stable_across_frames"] is False,
            "Incorrect native identity convention")
    instances = np.load(directory / "supervision/renderer_instance_id.npy", allow_pickle=False)
    require(instances.shape == depth.shape and instances.dtype == np.uint32, "Native ID shape/type mismatch")
    mapping = {int(k): v for k, v in identities["renderer_id_to_prim"].items()}
    require(not (set(map(int, np.unique(instances))) - mapping.keys() - {0, 1}), "Missing native IDs")
    catalogue = identities["component_catalogue"]
    require(len({c["component_index"] for c in catalogue}) == len(catalogue)
            and all(c["component_index"] > 0 for c in catalogue), "Duplicate/invalid component IDs")
    components, organs, owners = component_masks(instances, mapping, catalogue)
    require(np.array_equal(components, np.load(directory / "supervision/component_id.npy", allow_pickle=False)),
            "Component mask ownership mismatch")
    require(np.array_equal(organs, np.asarray(Image.open(directory / "supervision/organ_type.png"))),
            "Organ mask ownership mismatch")
    matches = [c for c in catalogue if c["component_id"] == draft["component_id"] and c["variant_id"] == sup["variant_id"]]
    require(len(matches) == 1 and matches[0]["split_group"] == draft["source_plant_id"], "Target catalogue mismatch")
    radius = proposal["nominal"]["petiole_radius_m"]
    visibility, target = interval_visibility(projected, interval, depth, valid, instances, mapping, owners, matches[0], radius)
    require(visibility == sup["visibility_evidence"], "Saved visibility disagrees with native buffers")
    require(np.array_equal(target.astype(np.uint8)*255, np.asarray(Image.open(directory / "supervision/target_visible.png"))),
            "Target mask mismatch")
    quality = view_quality(cal, projected, interval, radius, visibility, rgb, target)
    require(all(metadata["quality"].get(k) == v for k, v in quality.items()), "Saved quality gate mismatch")
    sync = metadata["synchronization"]
    require(sync["scene_unchanged_during_capture"] is True and sync["dynamic_recording_supported"] is False
            and sync["engine_frame_id_verified"] is False, "Unsupported static synchronization claims")
    # A back-projected DEPTH pixel is a front surface sample, not the anatomical centreline.
    x, y = np.floor(projected["pixel_xy"]).astype(int)
    surface = np.linalg.solve(np.asarray(cal["intrinsics"]), [x+.5, y+.5, 1]) * float(depth[y, x]) if valid[y, x] else None
    centre = np.asarray(projected["camera_optical_xyz_m"])
    result = {"sample_id": metadata["sample_id"], "target_review_id": sup["review_id"],
              "source_plant_family": sup["split_group"], "camera": camera,
              "integrity_and_recomputed_annotations_passed": True, "quality": quality,
              "nominal_pixel_xy": projected["pixel_xy"], "nominal_optical_xyz_m": centre.tolist(),
              "nominal_world_m": world[0].tolist(),
              "native_surface_depth_m": float(depth[y, x]) if valid[y, x] else None,
              "surface_to_centreline_distance_mm": float(np.linalg.norm(surface-centre)*1000) if surface is not None else None,
              "nominal_visibility_status": visibility["nominal"]["status"],
              "sampled_interval_visible_pixel_fraction": visibility["sampled_interval_visible_pixel_fraction"],
              "target_visible_mask_pixels": visibility["target_visible_mask_pixels"],
              "parent_proxy_diagnostic": proposal["parent_proxy_diagnostic"],
              "recommended_queue": "human_review" if quality["clear_view_gate_passed"] else "diagnostic_hold",
              "human_approval": False, "horticultural_validation": "pending", "training_eligible": False,
              "visual_review": "pending_explicit_reviewer_record"}
    return result, (rgb, depth, valid, target)


def review_canvas(metadata, buffers):
    """Review-only magnification: original RGB/depth/masks remain untouched."""
    rgb, depth, valid, target = buffers
    point = metadata["supervision"]["nominal_projected"]["pixel_xy"]
    x, y = map(int, point)
    left, top = max(0, min(x-48, 848-96)), max(0, min(y-48, 408-96))
    box = (left, top, left+96, top+96)
    canvas = Image.new("RGB", (1152, 838), (22, 26, 33))
    canvas.paste(Image.fromarray(rgb), (0, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), metadata["sample_id"] + " - ORIGINAL full-scene ROBOT HEAD RGB (848x408)", fill="white")
    draw.rectangle((left, top+28, left+96, top+124), outline="yellow", width=1)
    evidence = metadata['supervision'].get('visibility_evidence', {})
    visible = evidence.get('nominal', {}).get('visible_target_evidence') is True
    visible = visible and 0 <= x < 848 and 0 <= y < 408 and bool(target[y, x])
    legend = ('VISIBLE nominal cut\nWhite: nominal 10 mm\nMagenta: verified 10-20 mm' if visible else
              'CUT NOT VISIBLE\nNo white/magenta cut marks\nCrop is occluder evidence,\nNOT a cut on foreground')
    draw.text((858, 45), 'Yellow: crop location\n' + legend +
              '\nGreen: native petiole mask\n\nBottom: 4x NEAREST\nreview-only crops\n\nNo training approval\nNo cut safety approval', fill="white")
    marked = Image.fromarray(rgb).copy()
    if visible:
        interval = Image.new('L', (848, 408)); ink = ImageDraw.Draw(interval)
        probes = evidence.get('interval_probes', [])
        for a, b in zip(probes, probes[1:]):
            if a.get('visible_target_evidence') is True and b.get('visible_target_evidence') is True:
                ink.line([tuple(p['projected']['pixel_xy']) for p in (a, b)], fill=255)
        pixels = rgb.copy()
        pixels[(np.asarray(interval) != 0) & target] = (255, 0, 190)
        marked = Image.fromarray(pixels)
        painter = ImageDraw.Draw(marked)
        painter.line((point[0]-3, point[1], point[0]+3, point[1]), fill="white", width=1)
        painter.line((point[0], point[1]-3, point[0], point[1]+3), fill="white", width=1)
    masked = rgb.copy()
    masked[target] = (0, 255, 80)
    tiles = [marked, Image.fromarray(masked), Image.fromarray(colour_depth(depth, valid, .04, 2.))]
    title = 'RGB + visible cut' if visible else 'RGB at hidden nominal: NOT a visible cut'
    for i, (tile, label) in enumerate(zip(tiles, (title, "Exact native visible petiole", "Native camera-Z: yellow near / purple far"))):
        draw.text((i*384+6, 440), label, fill="white")
        canvas.paste(tile.crop(box).resize((384, 384), Image.Resampling.NEAREST), (i*384, 454))
    return canvas


def review_card(path, metadata, buffers):
    review_canvas(metadata, buffers).save(path)


def audit(run, output, *, stream_cards=False):
    run, output = Path(run).resolve(), Path(output).resolve()
    require(not output.exists(), "Choose a new audit directory; no overwrite")
    require(not output.is_relative_to(run) and not run.is_relative_to(output), "Keep audit separate from captured inputs")
    manifest = read_json(run / "manifest.json")
    require(not output.is_relative_to(Path(manifest["package"]).resolve()), "Do not write reviews into source assets")
    require(manifest["state"] == "pilot_ready_for_review" and manifest["source_assets_unchanged"] is True,
            "Expected a complete immutable capture pilot")
    require(manifest["robot"]["model"] == "RBY1_A_v1.2", "Expected RB-Y1 Model A v1.2")
    bindings = {}
    remember(bindings, run / "manifest.json")
    for path, expected in manifest["source_usd_sha256"].items():
        remember(bindings, path, expected)
    if manifest.get("source_catalogue_kind") == "native_collection_plan.v1":
        from .collection_plan import load_plan
        remember(bindings, manifest["source_collection_plan_path"], manifest["source_collection_plan_sha256"])
        drafts, reports = load_plan(manifest["source_collection_plan_path"])
        jobs = [j for j in drafts["jobs"] if j["job_id"] == manifest["collection_job_id"]]
        require(len(jobs) == 1 and manifest["target_family_split"] == jobs[0]["split"], "Collection job/split mismatch")
        rows = jobs[0]["targets"]
        require(manifest["source_geometry_modified"] is False, "Native pilot changed source geometry")
        for path, expected in drafts["source_bindings_sha256"].items():
            remember(bindings, path, expected)
    else:
        remember(bindings, manifest["source_draft_path"], manifest["source_draft_sha256"])
        from .capture_pilot import load_drafts
        drafts, reports, rows = load_drafts(manifest["source_draft_path"])
        remember(bindings, drafts["source_variants_path"], drafts["source_variants_sha256"])
    require(rule_fingerprint(manifest["cut_rule"]) == manifest["cut_rule_sha256"] == drafts["cut_rule_sha256"],
            "Cut rule mismatch")
    for report in reports:
        remember(bindings, report["manifest_path"], report["manifest_sha256"])
        for component in report["components"].values():
            remember(bindings, Path(report["manifest_path"]).parent / component["file"], component["asset_sha256"])
    model, mount, cal, urdf = reference_robot()
    remember(bindings, urdf)
    results, cards, seen = [], [], set()
    if stream_cards:
        output.mkdir(parents=True)
    for entry in manifest["samples"]:
        sid = entry["sample_id"]
        require(re.fullmatch(r"sample_[0-9]{4}", sid) is not None and sid not in seen, "Invalid/duplicate sample ID")
        seen.add(sid)
        directory = run / sid
        metadata_path = safe_file(run, sid + "/sample.json")
        remember(bindings, metadata_path)
        metadata = read_json(metadata_path)
        require(metadata["sample_id"] == sid and metadata["supervision"]["review_id"] == entry["target_review_id"],
                "Sample identity mismatch")
        for name, detail in metadata["files"].items():
            remember(bindings, safe_file(directory, name), detail["sha256"])
            expected_role = "observation" if name in OBSERVATIONS else "review_only" if name.startswith("review/") else "ground_truth_supervision"
            require(detail["role"] == expected_role, "Artifact role mismatch")
        require({name for name, d in metadata["files"].items() if d["role"] == "observation"} == set(OBSERVATIONS),
                "Unexpected model observation")
        draft = next(r for r in rows if r["draft_id"] == entry["target_review_id"])
        result, buffers = check_sample(directory, metadata, draft, model, mount, cal)
        results.append(result)
        if stream_cards:
            review_card(output / (sid + '.png'), metadata, buffers)
            cards.append((sid, None, None))
        else:
            cards.append((sid, metadata, buffers))
    require(bool(results), "Empty pilot")
    verify_bindings(bindings)
    if not stream_cards:
        output.mkdir(parents=True)
        for sid, metadata, buffers in cards:
            review_card(output / (sid + ".png"), metadata, buffers)
    result = {"schema_version": SCHEMA, "state": "complete_engineering_audit_not_approval",
              "created_utc": datetime.now(timezone.utc).isoformat(), "source_run": str(run),
              "bindings_sha256": bindings, "source_cut_rule": manifest["cut_rule"],
              "samples": results, "training_dataset_approved": False,
              "cards_sha256": {sid+".png": sha256(output / (sid+".png")) for sid, _, _ in cards},
              "limitations": ["Static synchronization only; native dynamic frame IDs unavailable",
                  "Mask coverage is two detailed plants, not all backdrop organs",
                  "10 mm is measured from manifest attachment, not the outer stem surface",
                  "Centreline labels differ from visible depth surface; depth is not replaced by geometry",
                  "Parent capsule diagnostic is not a mesh, whole-interval or blade clearance check",
                  "No physical grasp/cut, collision-free navigation, horticultural approval or training eligibility",
                  "Geometry-guided simulated robot snapshots, not blind view selection or live lab poses"],
              "implementation_sha256": {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name("capture_visibility.py"))}}
    verify_bindings(bindings)
    write_json(output / "audit.json", result)  # Completion marker written last.
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = audit(args.run, args.output)
    print(json.dumps({"state": result["state"], "samples": len(result["samples"]),
                      "clear_views": sum(s["quality"]["clear_view_gate_passed"] for s in result["samples"]),
                      "robot_pov_verified": all(s["camera"]["mounted_robot_pov_verified"] for s in result["samples"])}, indent=2))


if __name__ == "__main__":
    main()
