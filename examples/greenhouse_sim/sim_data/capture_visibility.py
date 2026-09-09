"""Renderer-identity visibility evidence, never cut/trajectory approval.

All masks live under supervision/, not in the model observation. IDs are local
to each captured renderer payload; the component catalogue supplies provenance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .capture_contract import RESOLUTION, depth_evidence, jsonable

ORGAN_IDS = {"unmapped": 0, "main_stem": 1, "sub_stem": 2, "leaf": 3,
             "fruit": 4, "flower": 5, "peduncle": 6, "other_component": 7}


def owner_for_path(path, catalogue):
    """Deepest component wins: a child leaf is NOT its parent petiole."""
    matches = [c for c in catalogue if path == c["prim_path"] or path.startswith(c["prim_path"] + "/")]
    return max(matches, key=lambda c: len(c["prim_path"]), default=None)


def decode_instances(payload):
    annotation = payload.get("instance_id_segmentation")
    if not isinstance(annotation, dict) or not {"data", "info"} <= annotation.keys():
        raise ValueError("Missing native instance segmentation in writer payload")
    data = np.asarray(annotation["data"])
    if data.shape != (RESOLUTION[1], RESOLUTION[0]) or data.dtype != np.uint32:
        raise ValueError("Instance segmentation must be uncolorized uint32 at sensor resolution")
    labels = annotation["info"].get("idToLabels")
    if not isinstance(labels, dict):
        raise ValueError("Missing renderer instance-to-prim mapping")
    mapping = {}
    for key, value in labels.items():
        if not isinstance(value, str):
            raise ValueError("Renderer identity must map to a prim path, not semantic guesses")
        identifier = int(key)
        if identifier < 0 or identifier > np.iinfo(np.uint32).max:
            raise ValueError("Invalid renderer instance ID")
        mapping[identifier] = value
    # 0/1 can be background/unlabelled sentinels; all other observed IDs require identity.
    missing = set(map(int, np.unique(data))) - mapping.keys() - {0, 1}
    if missing:
        raise ValueError(f"Unmapped renderer instance IDs: {sorted(missing)[:10]}")
    if not any(v.startswith("/") for v in mapping.values()):
        raise ValueError("No scene prim identities in segmentation")
    return data.copy(), mapping


def component_masks(instances, mapping, catalogue):
    components = np.zeros(instances.shape, np.uint32)
    organs = np.zeros(instances.shape, np.uint8)
    owners = {}
    for identifier in np.unique(instances):
        identifier = int(identifier)
        owner = owner_for_path(mapping.get(identifier, ""), catalogue)
        owners[identifier] = owner
        if owner is not None:
            mask = instances == identifier
            components[mask] = owner["component_index"]
            organs[mask] = ORGAN_IDS.get(owner["organ_type"], ORGAN_IDS["other_component"])
    return components, organs, owners


def interval_visibility(nominal, interval, depth, valid, instances, mapping, owners, target, radius_m):
    """Sample centreline projections using exact pixel identity AND depth.

    Counts unique pixels, not oversampled 1 mm points. This is not an amodal
    surface visibility fraction; branch-wide visibility and tool clearance are unknown.
    """
    def probe(point):
        evidence = depth_evidence(point, depth, valid, radius_m)
        result = {"projected": point, "depth_evidence": evidence, "visible_target_evidence": False}
        if point["projection_status"] != "in_frame":
            return {**result, "status": point["projection_status"]}
        x, y = np.floor(point["pixel_xy"]).astype(int)
        identifier = int(instances[y, x])
        owner = owners.get(identifier)
        matches = owner is not None and owner["component_index"] == target["component_index"]
        consistent = evidence["status"] == "depth_consistent_not_visibility_verified"
        if matches and consistent:
            status = "target_identity_and_depth_consistent"
        elif evidence["status"] == "foreground_occlusion_evidence" and not matches:
            status = "foreground_occluder_identified" if mapping.get(identifier, "").startswith("/") else "foreground_occluder_unmapped"
        elif matches:
            status = "unknown_target_identity_depth_conflict"
        else:
            status = "unknown_no_target_identity_at_pixel"
        return {**result, "status": status, "visible_target_evidence": matches and consistent,
                "pixel_xy": [int(x), int(y)], "renderer_instance_id": identifier,
                "observed_prim_path": mapping.get(identifier), "observed_component": owner}

    nominal_evidence = probe(nominal)
    probes = [probe(p) for p in interval]
    by_pixel = {}
    for p in probes:
        if "pixel_xy" in p:
            by_pixel.setdefault(tuple(p["pixel_xy"]), []).append(p)
    # If two centreline points fold onto one pixel, require both depth checks.
    visible = sum(all(p["visible_target_evidence"] for p in group) for group in by_pixel.values())
    target_ids = [i for i, owner in owners.items() if owner and owner["component_index"] == target["component_index"]]
    target_mask = np.isin(instances, target_ids)
    occluders = {}
    for p in probes:
        if p["status"] == "foreground_occluder_identified":
            path = p["observed_prim_path"]
            occluders[path] = {"prim_path": path, "component": p["observed_component"]}
    return {"schema_version": "greenhouse.interval_visibility.v1",
            "method": "exact_rendered_instance_identity_plus_optical_z_at_projected_pixels",
            "nominal": nominal_evidence, "interval_probes": probes,
            "unique_in_frame_interval_pixels": len(by_pixel), "visible_interval_pixels": visible,
            "sampled_interval_visible_pixel_fraction": visible / len(by_pixel) if by_pixel else None,
            "interval_fully_in_frame": all(p["projection_status"] == "in_frame" for p in interval),
            "amodal_surface_visibility_fraction": None, "target_visible_mask_pixels": int(target_mask.sum()),
            "identified_foreground_occluders": list(occluders.values()),
            "branch_wide_visibility_confirmed": False, "cut_safety_validated": False}, target_mask


def view_quality(calibration, nominal, interval, radius_m, visibility, rgb, target_mask):
    z = nominal["camera_optical_xyz_m"][2]
    diameter = 2 * radius_m * min(calibration["intrinsics"][0][0], calibration["intrinsics"][1][1]) / z if z > 0 else 0
    points = [p["pixel_xy"] for p in interval if p["projection_status"] == "in_frame"]
    length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()) if len(points) >= 2 else 0
    # Photometric diagnostic only; no exposure/material modification.
    luminance = rgb.astype(float) @ np.array([.2126, .7152, .0722])
    values = luminance[target_mask]
    dark = float(np.mean(values < 20)) if values.size else None
    reasons = []
    if not visibility["nominal"]["visible_target_evidence"]:
        reasons.append("nominal_target_visibility_not_verified")
    if not visibility["interval_fully_in_frame"]:
        reasons.append("interval_not_fully_in_frame")
    if (visibility["sampled_interval_visible_pixel_fraction"] or 0) < .8:
        reasons.append("insufficient_sampled_interval_visibility")
    if diameter < 3:
        reasons.append("estimated_petiole_width_below_3px")
    if length < 4:
        reasons.append("projected_interval_below_4px")
    if dark is None or dark > .6:
        reasons.append("target_mask_too_dark")
    return {"estimated_petiole_diameter_px": float(diameter), "projected_interval_length_px": length,
            "target_mask_dark_fraction": dark, "clear_view_gate_passed": not reasons,
            "clear_view_rejection_reasons": reasons,
            "thresholds": {"minimum_diameter_px": 3, "minimum_interval_px": 4,
                           "minimum_sampled_visibility": .8, "maximum_target_dark_fraction": .6},
            "thresholds_are": "engineering_pilot_gates_not_validated_training_or_difficulty_rules",
            "difficulty": "unassigned", "training_approved": False}


def write_visibility(directory, instances, mapping, catalogue, components, organs, target_mask):
    """Store ground-truth masks separately; update metadata file hashes last."""
    from PIL import Image
    directory = Path(directory)
    output = directory / "supervision"
    output.mkdir(exist_ok=False)
    np.save(output / "renderer_instance_id.npy", instances, allow_pickle=False)
    np.save(output / "component_id.npy", components, allow_pickle=False)
    Image.fromarray(organs).save(output / "organ_type.png")
    Image.fromarray(target_mask.astype(np.uint8) * 255).save(output / "target_visible.png")
    (output / "identities.json").write_text(json.dumps(jsonable({"renderer_id_to_prim": mapping,
        "component_catalogue": catalogue, "organ_ids": ORGAN_IDS,
        "unmapped_scope": "background greenhouse and backdrop plants lack per-organ manifests",
        "ids_stable_across_frames": False}), indent=2, allow_nan=False), encoding="utf-8")
    # An exact-mask preview, not a filled/amodal guess. Never modify inputs/rgb.png.
    source = np.asarray(Image.open(directory / "inputs/rgb.png").convert("RGB")).copy()
    source[target_mask] = np.rint(.35 * source[target_mask] + .65 * np.array([0, 255, 80])).astype(np.uint8)
    Image.fromarray(source).save(directory / "review/visible_target.png")
    path = directory / "sample.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    for file in [*sorted(output.iterdir()), directory / "review/visible_target.png"]:
        metadata["files"][file.relative_to(directory).as_posix()] = {
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            "role": "ground_truth_supervision" if file.parent == output else "review_only"}
    path.write_text(json.dumps(jsonable(metadata), indent=2, allow_nan=False), encoding="utf-8")
