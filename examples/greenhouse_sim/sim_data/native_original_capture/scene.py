"""Explicit original-only scene hook, adapted from native_greenhouse_pair.

No generated catalogue/substitution. Imports that need Kit stay inside functions.
"""
import numpy as np

from ..capture_sensor import calibration_for_native_resolution
from ..native_greenhouse_pair import assert_same_camera
from .contracts import RESOLUTION, require, read_json, bind_all


def restore_pose(stage, robot, pose):
    """Exact existing paired-worker FK application; guarded by AST parity tests."""
    from pxr import Usd
    from greenhouse_sim import robot_hardware, robot_kinematics

    root = np.asarray(pose["robot_root_to_world_usd_row_vectors"]).T
    links = robot_kinematics.Rby1Kinematics().all_link_transforms(pose["joint_degrees"])
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        robot_hardware._set_transform(stage.GetPrimAtPath(robot["root"]), root[:3, :3], root[:3, 3])
        for link, matrix in links.items():
            prim = stage.GetPrimAtPath(robot["root"] + "/" + link)
            require(bool(prim), "Missing robot link")
            robot_hardware._set_transform(prim, matrix[:3, :3], matrix[:3, 3])


def prepare_scene(app, plan):
    from pathlib import Path
    import sys
    import carb
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep
    from pxr import Usd, UsdGeom
    from launch_sim_data import load_local_payloads, populate
    from ..capture_scene import (capture_root, calibration, freeze_rigid_bodies,
                                 target_world_geometry, mounted_camera_to_head)
    from ..capture_pilot import source_hashes
    from ..capture_viewpoints import (component_catalogue, static_obstacles,
                                     scene_triangle_refiner, visible_bounds)
    from ..collection_plan import load_plan
    from ..floor_alignment import PACKAGE_FLOOR
    from ..robot_preview import add_robot_preview
    from ..training_screen import StaticBoundScreen

    bind_all(plan["source_bindings"])
    original_plan, reports = load_plan(plan["source_collection_plan"])
    job = next(j for j in original_plan["jobs"] if j["job_id"] == plan["collection_job_id"])
    require(job["split"] == "train" and plan["source_row"] in job["targets"], "Original source job changed")
    package = Path(plan["package"])
    wrapper = capture_root(package / "house/green_house_base.usd")
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    stage.SetLoadRules(Usd.StageLoadRules.LoadNone())
    stage.GetRootLayer().subLayerPaths = list(wrapper.subLayerPaths)
    require(stage.GetRootLayer().anonymous and stage.GetPrimAtPath("/World/Gutters"),
            "Original greenhouse wrapper required")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetEditTarget(stage.GetSessionLayer())
    excluded = load_local_payloads(stage)
    records = []
    gutter_x, counts = populate(stage, package, app, records, review_plant=job["plant_family"])
    require(counts == plan["expected_scene_counts"], "Greenhouse population changed")
    require(excluded == plan["excluded_external_roots"], "External-prop scope changed")
    variants = plan["scene_variants"]
    robot = add_robot_preview(stage, gutter_x=gutter_x, floor_path=PACKAGE_FLOOR, right_tool="gripper")
    freeze_rigid_bodies(stage)
    sys.path.insert(0, str(package / "env_panel"))
    from tomato_env import daylight
    require(Path(daylight.__file__).resolve() == package / "env_panel/tomato_env/daylight.py",
            "Dynamic daylight import escaped pinned original package")
    lighting = daylight.apply(stage, day=172, minutes=13*60, intensity=1500, dome_intensity=6000)
    # This is the pinned CLEAR scene, explicitly not a photometric replay of the prior.
    settings = carb.settings.get_settings()
    settings.set("/rtx/rendermode", "RealTimePathTracing")
    app.update()
    require(settings.get("/rtx/rendermode") == "RealTimePathTracing", "Renderer mismatch")
    rep.set_global_seed(original_plan["configuration"]["seed"])
    pose = plan["expected_robot_snapshot"]
    restore_pose(stage, robot, pose)
    cal = calibration(stage)
    assert_same_camera(cal, plan["prior_calibration"])
    mount = mounted_camera_to_head(stage)
    require(np.allclose(mount, pose["camera_to_head_column_vectors"], atol=1e-8, rtol=0), "Mounted head transform changed")
    # Each case is screened by the shared notice-invalidated geometry cache.
    # A bad first pose must not prevent later valid cases from being attempted.
    screen = None
    catalogue = component_catalogue(stage, records, reports, variants)
    require(len(catalogue) == counts["components"], "Original component population changed")
    row = plan["source_row"]
    world = target_world_geometry(stage, plan["original_variant"], row)
    # Match source PLACEMENT only, never requested-target labels from the old image.
    prior_path = Path(plan["pose_prior"]["source_capture"]) / plan["pose_prior"]["sample_id"] / "sample.json"
    prior_matrix = read_json(prior_path)["supervision"]["plant_to_world_usd_row_vectors"]
    require(np.allclose(world["plant_to_world_usd_row_vectors"], prior_matrix, atol=1e-9, rtol=0), "Original donor placement changed")
    target = next(c for c in catalogue if c["variant_id"] == row["variant_id"] and c["component_id"] == row["component_id"])
    report = next(r for r in reports if r["plant_id"] == plan["source_family"])
    stage_hashes = source_hashes(stage)
    manifest = read_json(Path(plan["pose_prior"]["source_capture"]) / "manifest.json")
    require(stage_hashes == manifest["source_usd_sha256"], "Loaded source USD set differs from original prior")
    context.get_selection().clear_selected_prim_paths()
    omni.timeline.get_timeline_interface().pause()
    native_cal = calibration_for_native_resolution(cal, RESOLUTION)
    assert_same_camera(native_cal, plan["expected_calibration"])
    return dict(stage=stage, robot=robot, calibration=native_cal, world=world, target=target,
                report=report, catalogue=catalogue, screen=screen, lighting=lighting,
                counts=counts, settings=settings, source_hashes=stage_hashes)


def set_case_pose(context, plan):
    """Move actual robot joints only; never a free camera or source-geometry edit."""
    from copy import deepcopy
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    from ..capture_scene import calibration, mounted_camera_to_head, target_world_geometry, plan_head_pose
    stage, robot = context["stage"], context["robot"]
    world = target_world_geometry(stage, plan["original_variant"], plan["source_row"])
    pose = deepcopy(plan["expected_robot_snapshot"])
    if plan["pose_request"]["mode"] == "reframe_head_only":
        pose["joint_degrees"], _ = plan_head_pose(Rby1Kinematics(), pose["joint_degrees"],
            np.asarray(pose["robot_root_to_world_usd_row_vectors"]).T,
            np.asarray(pose["camera_to_head_column_vectors"]), world["nominal_world_m"],
            plan["prior_calibration"]["intrinsics"], np.asarray(plan["pose_request"]["native_pixel_xy"]) / 2)
    restore_pose(stage, robot, pose)
    require(np.allclose(mounted_camera_to_head(stage), pose["camera_to_head_column_vectors"],
                        atol=1e-8, rtol=0), "Mounted camera moved")
    cal = calibration_for_native_resolution(calibration(stage), RESOLUTION)
    if plan["pose_request"]["mode"] == "exact_prior":
        assert_same_camera(cal, plan["expected_calibration"])
    else:
        for key in cal:
            if key != "camera_to_world_usd_row_vectors":
                require(cal[key] == plan["expected_calibration"][key], "Head reframing changed camera optics")
    return pose, cal, world
