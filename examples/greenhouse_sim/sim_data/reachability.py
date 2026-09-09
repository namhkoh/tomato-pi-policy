"""Bounded, non-commanding position-IK diagnostics for anatomy review.

Workers consume plain snapshots only: no USD, UI, physical robot or human review
writes. A found point-IK solution is never a grasp/path/cutting approval.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import time

import numpy as np

from greenhouse_sim import robot_kinematics as rk

SCHEMA = "greenhouse.reachability.v1"


def fingerprint(snapshot):
    return hashlib.sha256(json.dumps(snapshot, sort_keys=True, allow_nan=False).encode()).hexdigest()


def reconstruct_joints(model, link_matrices):
    """Recover actual static link poses, not stale imported joint-state attrs.

    Reject arbitrary link edits/scaling inconsistent with the model. Matrices
    use column vectors and are relative to the measured base link.
    """
    from scipy.spatial.transform import Rotation

    expected_names = {"base", *model._by_child}
    if set(link_matrices) != expected_names:
        raise ValueError("Robot links do not match the v1.2 model")
    matrices = {k: np.asarray(v, dtype=float) for k, v in link_matrices.items()}
    for matrix in matrices.values():
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-7)
                or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)):
            raise ValueError("Robot link transforms must be finite rigid metre-scale poses")
    angles, slides = {}, {}
    for joint in model._by_name.values():
        motion = np.linalg.inv(joint.origin) @ np.linalg.inv(matrices[joint.parent]) @ matrices[joint.child]
        if joint.kind in ("revolute", "continuous"):
            axis = joint.axis / np.linalg.norm(joint.axis)
            value = float(np.dot(Rotation.from_matrix(motion[:3, :3]).as_rotvec(), axis))
            # Equivalent angles at +/-pi can otherwise exceed an asymmetric limit.
            equivalents = [value + turns * 2 * np.pi for turns in (-1, 0, 1)
                           if joint.lower_rad - 1e-7 <= value + turns * 2 * np.pi <= joint.upper_rad + 1e-7]
            if not equivalents:
                raise ValueError(f"Measured joint outside URDF limits: {joint.name}")
            value = min(equivalents, key=abs)
            angles[joint.name] = float(np.degrees(np.clip(value, joint.lower_rad, joint.upper_rad)))
        elif joint.kind == "prismatic":
            value = float(np.dot(motion[:3, 3], joint.axis) / np.dot(joint.axis, joint.axis))
            if not joint.lower_rad - 1e-7 <= value <= joint.upper_rad + 1e-7:
                raise ValueError(f"Measured finger outside URDF limits: {joint.name}")
            slides[joint.name] = float(np.clip(value, joint.lower_rad, joint.upper_rad))
    reproduced = model.all_link_transforms(angles, prismatic_m=slides)
    if any(not np.allclose(matrices[k], reproduced[k], atol=2e-6, rtol=0) for k in matrices):
        raise ValueError("Live link poses are inconsistent with URDF FK; reset/review the robot pose")
    return angles, slides


def endpoint_screen(model, snapshot, side, joints, check_cancel=lambda: None):
    """Conservative, explicitly partial endpoint overlaps, never a safe path."""
    base = np.asarray(snapshot["base_world_matrix"])
    torso = snapshot["torso_degrees"]
    current = snapshot["arm_degrees"]
    inter = model.inter_arm_clearance(joints if side == "left" else current["left"],
                                      joints if side == "right" else current["right"], base, torso)
    minimum, nearest = inter.clearance_m, inter.nearest_obstacle
    candidates = model.arm_capsules(side, joints, base, torso)
    hand_matrix = model.forward(side, joints, base, torso)
    boxes = []
    for part in snapshot["tool_boxes_ee"].get(side, []):
        center = (hand_matrix @ np.r_[part["center"], 1])[:3]
        boxes.append(rk.OrientedBoxObstacle(part["path"], tuple(center),
                     tuple(map(tuple, hand_matrix[:3, :3])), tuple(part["half_extents"])))
    overlaps = []
    if minimum < 0:
        overlaps.append({"pair": nearest, "clearance_m": float(minimum)})
    for item in snapshot["obstacle_boxes"]:
        check_cancel()
        obstacle = rk.BoxObstacle(item["path"], tuple(item["min"]), tuple(item["max"]))
        for capsule in candidates:
            distance = rk.capsule_box_clearance(capsule, obstacle)
            pair = f"{capsule.path} <-> {obstacle.path}"
            if distance < minimum:
                minimum, nearest = distance, pair
            if distance < 0:
                overlaps.append({"pair": pair, "clearance_m": float(distance)})
        for box in boxes:
            distance = rk.oriented_box_box_clearance(box.centre_m, np.asarray(box.rotation), box.half_extents_m, (obstacle,)).clearance_m
            pair = f"{box.path} <-> {obstacle.path}"
            if distance < minimum:
                minimum, nearest = distance, pair
            if distance < 0:
                overlaps.append({"pair": pair, "clearance_m": float(distance)})
    return {"status": "possible_overlap" if overlaps else "no_overlap_in_screened_geometry",
            "minimum_clearance_m": float(minimum) if np.isfinite(minimum) else None,
            "nearest_pair": nearest, "overlap_count": len(overlaps), "overlaps": overlaps[:20],
            "scope": "arm capsules vs stationary other arm; arm/tool boxes vs gutters and selected plant own AABBs",
            "not_checked": ["full self collision including torso/base", "other plants and remaining greenhouse meshes",
                            "moving tool vs stationary arm/tool", "approach path", "allowed target contacts"],
            "collision_free_certified": False}


class SearchStopped(Exception):
    pass


def check_reachability(snapshot, *, model=None, cancel_event=None, seconds_per_arm=4.0,
                       maximum_seeds=5, maximum_evaluations=160):
    """Finite deterministic multistart search, with a sound outer-radius rejection."""
    if seconds_per_arm <= 0 or maximum_seeds < 1 or maximum_evaluations < 1:
        raise ValueError("Search budgets must be positive")
    model = rk.Rby1Kinematics() if model is None else model
    from pathlib import Path
    for source in snapshot.get("source_files", []):
        if hashlib.sha256(Path(source["path"]).read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError("Source changed since anatomy audit; re-audit before checking reachability")
    started = time.monotonic()
    output = {"schema_version": SCHEMA, "snapshot": snapshot, "snapshot_sha256": fingerprint(snapshot),
              "arms": {}, "constraints": {"base_fixed": True, "torso_fixed": True, "other_arm_fixed": True,
                  "orientation_constrained": False, "seconds_per_arm": seconds_per_arm,
                  "maximum_seeds": maximum_seeds, "maximum_evaluations_per_seed": maximum_evaluations},
              "grasp_pose": "not_assessed_no_approved_grasp_region",
              "approach_path": "not_tested", "bimanual": "not_tested",
              "cutting": "not_applicable_to_gripper_probe", "cut_approval": False,
              "physical_executability": "not_tested", "human_review_modified": False,
              "training_label_approved": False}
    target = np.asarray(snapshot["target_world_m"], dtype=float)
    if target.shape != (3,) or not np.isfinite(target).all():
        raise ValueError("Target must be a finite world-space point")
    base, torso = np.asarray(snapshot["base_world_matrix"]), snapshot["torso_degrees"]
    for index, side in enumerate(("left", "right")):
        deadline = time.monotonic() + seconds_per_arm

        def check_cancel():
            if cancel_event is not None and cancel_event.is_set():
                raise SearchStopped("cancelled")
            if time.monotonic() >= deadline:
                raise SearchStopped("time_budget_exhausted")

        result = {"status": "no_solution_found", "attempts": 0, "best_position_error_m": None,
                  "candidate": None, "endpoint_screen": {"status": "not_tested"}}
        output["arms"][side] = result
        point = snapshot["probe_points_ee_m"].get(side)
        if point is None:
            result["status"] = "unsupported_or_missing_gripper"
            continue
        reach = model.maximum_endpoint_reach_m(side) + float(np.linalg.norm(point))
        distance = float(np.linalg.norm(target - model.arm_shoulder_position_m(side, base, torso)))
        result.update(shoulder_distance_m=distance, conservative_outer_reach_m=reach)
        if distance > reach + .001:
            result["status"] = "outside_outer_reach_bound"
            continue
        lower, upper = model.arm_limits_degrees(side)
        seeds = [snapshot["arm_degrees"][side], (lower + upper) / 2]
        rng = np.random.default_rng(731 + index)
        seeds.extend(rng.uniform(lower + 2, upper - 2) for _ in range(maximum_seeds))
        try:
            for seed in seeds[:maximum_seeds]:
                check_cancel()
                result["attempts"] += 1
                solution = model.solve_position(side, local_point_m=point, target_point_m=target,
                    seed_degrees=seed, base_matrix=base, torso_degrees=torso,
                    maximum_evaluations=maximum_evaluations, check_cancel=check_cancel)
                if result["best_position_error_m"] is None or solution.position_error_m < result["best_position_error_m"]:
                    result["best_position_error_m"] = solution.position_error_m
                if not solution.succeeded:
                    continue
                result["status"] = "position_ik_found"
                matrix = model.forward(side, solution.joint_degrees, base, torso)
                result["candidate"] = {**asdict(solution), "ee_world_matrix": matrix.tolist(),
                    "probe_world_m": (matrix @ np.r_[point, 1])[:3].tolist(),
                    "joint_limit_margin_degrees": model.arm_joint_limit_margin_degrees(side, solution.joint_degrees),
                    "arm_capsules": [asdict(c) for c in model.arm_capsules(side, solution.joint_degrees, base, torso)]}
                result["endpoint_screen"] = endpoint_screen(model, snapshot, side, solution.joint_degrees, check_cancel)
                break
        except SearchStopped as exc:
            result["search_stop_reason"] = str(exc)
            if result["candidate"] is None:
                result["status"] = str(exc)
            if cancel_event is not None and cancel_event.is_set():
                break
    output["elapsed_seconds"] = time.monotonic() - started
    return output
