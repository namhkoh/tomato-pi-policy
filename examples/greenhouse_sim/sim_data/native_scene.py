"""Shared validated full-greenhouse static scene; no native capture or robot commands.

Extracted without changing geometry, lighting, optics, floor placement or source
bindings. USD/Isaac imports stay inside the function after SimulationApp starts.
"""
from pathlib import Path
import time
import numpy as np
from .dataset_review import require,read_json
from .native_greenhouse_pair import load_source,assert_same_camera


def prepare_native_scene(app, plan):
    import carb
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep
    from pxr import Usd,UsdGeom
    from launch_sim_data import load_local_payloads,populate
    from greenhouse_sim import robot_hardware,robot_kinematics
    from .capture_scene import capture_root,calibration,freeze_rigid_bodies,target_world_geometry
    from .collection_plan import load_plan
    from .floor_alignment import PACKAGE_FLOOR
    from .robot_preview import add_robot_preview
    from .plant_variant_catalogue import load_for_inspection

    started = time.perf_counter()
    old_manifest, old_sample, original_bindings = load_source(plan["source_capture"], plan["source_sample"])
    original_plan, reports = load_plan(plan["source_collection_plan"])
    generated = load_for_inspection(plan["variant_directory"], plan["source_collection_plan"])
    require(next((r for r in generated["rows"] if r["target_id"] == plan["generated_row"]["target_id"]), None)
            == plan["generated_row"], "Generated target became stale or withheld")
    original_rows = [r for j in original_plan["jobs"] for r in j["targets"]
                     if r["target_id"] == plan["source_row"]["target_id"]]
    require(original_rows == [plan["source_row"]], "Source target changed")
    package = Path(original_plan["package"])
    wrapper = capture_root(package/"house/green_house_base.usd")
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    stage.SetLoadRules(Usd.StageLoadRules.LoadNone())
    stage.GetRootLayer().subLayerPaths = list(wrapper.subLayerPaths)
    require(stage.GetRootLayer().anonymous and stage.GetPrimAtPath("/World/Gutters"), "Original greenhouse required")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetEditTarget(stage.GetSessionLayer())
    excluded = load_local_payloads(stage)
    records = []
    gutter_x, counts = populate(stage, package, app, records, review_plant=plan["source_family"])
    require(counts == plan["expected_scene_counts"] == old_manifest["scene_counts"], "Greenhouse population changed")
    require(excluded == old_manifest["unbundled_external_prop_roots_excluded"], "External prop scope changed")
    variants = old_manifest["variants"]
    robot = add_robot_preview(stage, gutter_x=gutter_x, floor_path=PACKAGE_FLOOR, right_tool="gripper")
    freeze_rigid_bodies(stage)

    import sys
    sys.path.insert(0, str(package/"env_panel"))
    from tomato_env import daylight
    require(original_plan["configuration"].get("clear_capture") == "robot_head_close_diffuse_v1",
            "Qualified diffuse-lighting source required")
    lighting = daylight.apply(stage, day=172, minutes=13*60, intensity=1500, dome_intensity=6000)
    require(lighting == old_manifest["lighting"], "Original lighting changed")
    settings = carb.settings.get_settings()
    require(old_manifest["renderer"] == "RealTimePathTracing", "Original renderer profile required")
    settings.set("/rtx/rendermode", old_manifest["renderer"])
    app.update()
    rep.set_global_seed(original_plan["configuration"]["seed"])

    pose = plan["expected_robot_snapshot"]
    require(pose == old_sample["robot_snapshot"], "Robot reference changed")
    root = np.asarray(pose["robot_root_to_world_usd_row_vectors"]).T
    links = robot_kinematics.Rby1Kinematics().all_link_transforms(pose["joint_degrees"])
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        robot_hardware._set_transform(stage.GetPrimAtPath(robot["root"]), root[:3, :3], root[:3, 3])
        for link, matrix in links.items():
            prim = stage.GetPrimAtPath(robot["root"]+"/"+link)
            require(bool(prim), "Missing robot link")
            robot_hardware._set_transform(prim, matrix[:3, :3], matrix[:3, 3])
    assert_same_camera(calibration(stage), old_sample["calibration"])
    original_variant = next(v for v in variants if v["variant_id"] == plan["source_row"]["variant_id"])
    require(original_variant == plan["original_variant"], "Foreground source placement changed")
    old_world = target_world_geometry(stage, original_variant, plan["source_row"])
    for key, value in old_world.items():
        require(np.allclose(value, old_sample["supervision"][key], atol=1e-9, rtol=0),
                "Source world geometry moved: "+key)
    context.get_selection().clear_selected_prim_paths()
    omni.timeline.get_timeline_interface().pause()
    setup_seconds = time.perf_counter()-started

    return {
        'started': started,
        'stage': stage,
        'records': records,
        'reports': reports,
        'variants': variants,
        'generated': generated,
        'counts': counts,
        'robot': robot,
        'old_manifest': old_manifest,
        'old_sample': old_sample,
        'original_bindings': original_bindings,
        'original_variant': original_variant,
        'pose': pose,
        'settings': settings,
        'setup_seconds': setup_seconds,
    }
