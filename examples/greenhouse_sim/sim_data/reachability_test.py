"""Reach diagnostics cannot silently become human/physical approvals."""
import copy
import json
import threading

import numpy as np
import pytest

from greenhouse_sim import robot_kinematics as rk, robot_scene
from sim_data.reachability import check_reachability, fingerprint, reconstruct_joints


def simple_snapshot(model=None):
    model = model or rk.Rby1Kinematics()
    pose = robot_scene.SDK_READY_POSE_DEGREES
    arms = {side: [pose[f"{side}_arm_{i}"] for i in range(7)] for side in ("left", "right")}
    point = [0, 0, -.105]
    base = np.eye(4)
    target = (model.forward("left", arms["left"], base) @ np.r_[point, 1])[:3]
    return {"base_world_matrix": base.tolist(), "torso_degrees": [pose[f"torso_{i}"] for i in range(6)],
            "arm_degrees": arms, "target_world_m": target.tolist(),
            "probe_points_ee_m": {"left": point, "right": point}, "tool_boxes_ee": {}, "obstacle_boxes": []}


def test_live_pose_roundtrip_including_head_fingers_and_limits():
    model = rk.Rby1Kinematics()
    pose = {**robot_scene.SDK_READY_POSE_DEGREES, "head_0": 33, "head_1": -21}
    slides = {"gripper_finger_l1": -.03, "gripper_finger_l2": .02}
    links = model.all_link_transforms(pose, prismatic_m=slides)
    angles, measured = reconstruct_joints(model, links)
    for key, value in pose.items():
        assert angles[key] == pytest.approx(value, abs=1e-7)
    for key, value in slides.items():
        assert measured[key] == pytest.approx(value, abs=1e-8)
    links["link_left_arm_3"][0, 3] += .01
    with pytest.raises(ValueError, match="inconsistent"):
        reconstruct_joints(model, links)


def test_scaled_link_is_not_a_valid_robot_pose():
    model = rk.Rby1Kinematics()
    links = model.all_link_transforms(robot_scene.SDK_READY_POSE_DEGREES)
    links["ee_left"][0, 0] *= 2
    with pytest.raises(ValueError):
        reconstruct_joints(model, links)


def test_known_fk_point_reached_but_never_a_grasp_or_path_approval():
    model = rk.Rby1Kinematics()
    snapshot = simple_snapshot(model)
    result = check_reachability(snapshot, model=model, maximum_seeds=2)
    left = result["arms"]["left"]
    assert left["status"] == "position_ik_found"
    assert left["best_position_error_m"] < .001
    assert left["candidate"]["joint_limit_margin_degrees"] >= 0
    assert not left["endpoint_screen"]["collision_free_certified"]
    assert not result["cut_approval"] and not result["human_review_modified"]
    assert not result["training_label_approved"]
    assert result["bimanual"] == "not_tested"
    assert not result["constraints"]["orientation_constrained"]
    json.dumps(result, allow_nan=False)


def test_outer_bound_is_configuration_specific_and_bypasses_solver(monkeypatch):
    model = rk.Rby1Kinematics()
    snapshot = simple_snapshot(model)
    snapshot["target_world_m"] = [10, 0, 4]
    monkeypatch.setattr(model, "solve_position", lambda *a, **k: pytest.fail("outside bound should not use IK"))
    result = check_reachability(snapshot, model=model)
    assert all(a["status"] == "outside_outer_reach_bound" and a["attempts"] == 0 for a in result["arms"].values())


def test_multistart_failure_is_not_unreachable(monkeypatch):
    model = rk.Rby1Kinematics()
    snapshot = simple_snapshot(model)
    calls = []
    def failed(*args, **kwargs):
        calls.append(kwargs["seed_degrees"])
        return rk.IKResult(tuple(kwargs["seed_degrees"]), .08, 0, 1, False)
    monkeypatch.setattr(model, "solve_position", failed)
    result = check_reachability(snapshot, model=model, maximum_seeds=3)
    assert result["arms"]["left"]["status"] == "no_solution_found"
    assert result["arms"]["left"]["attempts"] == 3
    assert not np.allclose(calls[0], calls[1])


def test_endpoint_overlap_does_not_erase_point_ik_success():
    snapshot = simple_snapshot()
    snapshot["obstacle_boxes"] = [{"path": "fixture_block", "min": [-2, -2, -2], "max": [2, 2, 2]}]
    result = check_reachability(snapshot, maximum_seeds=1)
    left = result["arms"]["left"]
    assert left["status"] == "position_ik_found"
    assert left["endpoint_screen"]["status"] == "possible_overlap"


def test_cancel_and_timeout_are_not_unreachable():
    cancel = threading.Event()
    cancel.set()
    result = check_reachability(simple_snapshot(), cancel_event=cancel)
    assert result["arms"]["left"]["status"] == "cancelled"
    result = check_reachability(simple_snapshot(), seconds_per_arm=1e-12)
    assert result["arms"]["left"]["status"] == "time_budget_exhausted"


def test_bounded_ik_honours_callback_and_evaluation_budget():
    model = rk.Rby1Kinematics()
    def stop():
        raise RuntimeError("test cancellation")
    with pytest.raises(RuntimeError, match="test cancellation"):
        model.solve_position("left", local_point_m=[0, 0, 0], target_point_m=[0, 0, 1],
                             seed_degrees=[0] * 7, base_matrix=np.eye(4), check_cancel=stop)
    with pytest.raises(ValueError, match="maximum_evaluations"):
        model.solve_position("left", local_point_m=[0, 0, 0], target_point_m=[0, 0, 1],
                             seed_degrees=[0] * 7, base_matrix=np.eye(4), maximum_evaluations=0)


def test_state_fingerprint_changes_with_robot_plant_tool_and_source():
    source = simple_snapshot()
    original = fingerprint(source)
    for field, value in (("target_world_m", [1, 2, 3]), ("right_tool", "knife_only"),
                         ("placement", "relocated_review_sample"), ("source_files", [{"sha256": "changed"}])):
        changed = copy.deepcopy(source)
        changed[field] = value
        assert fingerprint(changed) != original


def test_changed_source_rejected(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text("changed")
    snapshot = simple_snapshot()
    snapshot["source_files"] = [{"path": str(path), "sha256": "stale"}]
    with pytest.raises(ValueError, match="Source changed"):
        check_reachability(snapshot)


def test_real_snapshot_tracks_live_base_plant_and_tool(tmp_path):
    from pxr import Gf, Usd, UsdGeom
    from greenhouse_sim import robot_model
    from sim_data.audit import DEFAULT_PACK, audit_manifest
    from sim_data.geometry import assemble_plant
    from sim_data.robot_preview import add_robot_preview, configure_right_tool
    from sim_data.reachability_snapshot import capture_snapshot
    manifest = DEFAULT_PACK / "plants/components/seed101_full/manifest.json"
    if not manifest.is_file() or not robot_model.DEFAULT_ASSET.is_file():
        pytest.skip("Local package/generated robot required")
    report = audit_manifest(manifest)
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetEditTarget(stage.GetSessionLayer())
    add_robot_preview(stage, gutter_x=-.2, right_tool="gripper")
    paths = assemble_plant(stage, "/World/TestPlant", report)
    plant = UsdGeom.Xformable(stage.GetPrimAtPath("/World/TestPlant"))
    plant.AddTranslateOp().Set(Gf.Vec3d(-.005, 0, .9))
    record = {"plant_root": "/World/TestPlant", "component_paths": paths}
    target = next(t for t in report["targets"] if t["status"] == "needs_review")
    before = stage.GetSessionLayer().ExportToString()
    snapshot = capture_snapshot(stage, record, report, target, borrowed=True)
    assert stage.GetSessionLayer().ExportToString() == before
    assert snapshot["placement"] == "relocated_review_sample"
    np.testing.assert_allclose(snapshot["target_world_m"], np.asarray(target["attachment_plant_m"]) + [-.005, 0, .9])
    assert all(snapshot["probe_points_ee_m"][side] is not None for side in ("left", "right"))
    assert not snapshot["gutter_geometry_present"]
    configure_right_tool(stage, "/World/RBY1", "knife_only")
    changed = capture_snapshot(stage, record, report, target)
    assert changed["probe_points_ee_m"]["right"] is None
    assert fingerprint(snapshot) != fingerprint(changed)
