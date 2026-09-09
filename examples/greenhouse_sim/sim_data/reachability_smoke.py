"""Real Isaac UI test; relocated fixture, NOT a demonstrated grasp or cut."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import uuid

from sim_data.audit import DEFAULT_PACK, ROOT, audit_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / "data/sim_data/reachability_smokes" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    if output.exists() or output.resolve().is_relative_to(DEFAULT_PACK.resolve()):
        parser.error("Choose a new output directory outside source assets")
    output.mkdir(parents=True)
    report = audit_manifest(DEFAULT_PACK / "plants/components/seed101_full/manifest.json")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "hide_ui": False, "width": 1280, "height": 720,
                         "window_width": 1600, "window_height": 1100,
                         "renderer": "RaytracedLighting", "sync_loads": False, "multi_gpu": False})
    exit_code = 1
    try:
        import numpy as np
        import omni.usd
        import omni.kit.renderer_capture
        from omni.kit.viewport.utility import get_active_viewport
        from pxr import Gf, UsdGeom, UsdLux
        from greenhouse_sim.robot_kinematics import Rby1Kinematics
        from sim_data.geometry import assemble_plant
        from sim_data.robot_preview import add_robot_preview
        from sim_data.gallery import BatchReviewPanel
        from sim_data.reachability_snapshot import capture_snapshot
        from sim_data.reachability_ui import GHOST

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, "Z")
        stage.SetEditTarget(stage.GetSessionLayer())
        add_robot_preview(stage, gutter_x=-.2, right_tool="gripper")
        paths = assemble_plant(stage, "/World/FixturePlant", report)
        translation = UsdGeom.Xformable(stage.GetPrimAtPath("/World/FixturePlant")).AddTranslateOp()
        translation.Set(Gf.Vec3d(-.005, 0, .9))
        record = {"plant_root": "/World/FixturePlant", "manifest_path": report["manifest_path"], "component_paths": paths}
        target = next(t for t in report["targets"] if t["status"] == "needs_review")
        snapshot = capture_snapshot(stage, record, report, target)
        # Deliberate fixture relocation guarantees an attainable point for UI testing.
        # It is NOT evidence that the authored plant has a safe grasp/cut solution.
        model = Rby1Kinematics()
        matrix = model.forward("left", snapshot["arm_degrees"]["left"],
                               np.asarray(snapshot["base_world_matrix"]), snapshot["torso_degrees"])
        point = (matrix @ np.r_[snapshot["probe_points_ee_m"]["left"], 1])[:3]
        translation.Set(Gf.Vec3d(*(point - np.asarray(target["attachment_plant_m"]))))
        UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1500)
        UsdGeom.Camera.Define(stage, "/World/StartCamera")
        for _ in range(10):
            app.update()
        viewport = get_active_viewport()
        viewport.set_active_camera("/World/StartCamera")
        panel = BatchReviewPanel(stage, [record], viewport, output / "human_reviews", history_directories=[])
        panel.entries = [(report, target)]
        panel.index = 0
        panel.show()
        from sim_data.review_camera import HEAD_CAMERA, view_context
        assert str(viewport.camera_path) == HEAD_CAMERA
        assert tuple(viewport.get_texture_resolution()) == (848, 408)
        panel.move(1)
        assert str(viewport.camera_path) == HEAD_CAMERA, "Target selection reset robot POV"
        panel.set_review_view("diagnostic_closeup")
        assert str(viewport.camera_path).endswith("/ReviewCamera")
        assert tuple(viewport.get_texture_resolution()) == (1280, 720)
        panel.set_review_view("robot_head")
        assert str(viewport.camera_path) == HEAD_CAMERA
        assert view_context(stage, viewport, panel.scene, target)["view_kind"] == "robot_head"
        panel.window.position_x = 0
        panel.window.position_y = 0
        panel.window.height = 1060
        for _ in range(30):
            app.update()  # Let Kit settle camera/view changes before taking the static snapshot.
        reach = panel.reachability

        def finish_task():
            deadline = time.monotonic() + 40
            updates = 0
            while reach.task and not reach.task.done():
                app.update()
                updates += 1
                if time.monotonic() > deadline:
                    raise RuntimeError("Reach UI task timed out")
            if reach.task:
                reach.task.result()
            return updates

        before = capture_snapshot(stage, record, report, target)
        reach.start()
        updates = finish_task()
        assert reach.result, reach.label.text
        assert reach.result["arms"]["left"]["candidate"], reach.result["arms"]
        saved = list((output / "reachability").glob("*.json"))
        assert len(saved) == 1
        reach.show_candidate("left")
        assert reach.result is not None, "Outline invalidated its own result"
        assert stage.GetPrimAtPath(GHOST)
        after = capture_snapshot(stage, record, report, target)
        assert before == after, "Diagnostic changed robot/target geometry"
        for _ in range(30):
            app.update()
        screenshot = output / "diagnostic_fixture_ui.png"
        omni.kit.renderer_capture.acquire_renderer_capture_interface().capture_next_frame_swapchain(str(screenshot))
        from PIL import Image
        deadline = time.monotonic() + 20
        while True:
            app.update()
            try:
                with Image.open(screenshot) as captured:
                    captured.verify()
                break
            except (OSError, SyntaxError):
                if time.monotonic() > deadline:
                    raise RuntimeError("UI capture did not finish")
        translation.Set(translation.Get() + Gf.Vec3d(.01, 0, 0))
        assert reach.result is None and not stage.GetPrimAtPath(GHOST), "Stale result/outline survived target motion"
        reach.start()
        reach.cancel()
        finish_task()
        assert reach.result is None and len(list((output / "reachability").glob("*.json"))) == 1
        panel.start_gallery()
        deadline = time.monotonic() + 180
        while not panel.capture_task.done():
            app.update()
            if time.monotonic() > deadline:
                raise RuntimeError("Head-POV gallery timed out")
        panel.capture_task.result()
        assert len(panel.evidence) == 6, panel.gallery_message.text
        assert all(e["view_context"]["view_kind"] == "robot_head" and
                   e["view_context"]["resolution"] == [848, 408] for e in panel.evidence.values())
        assert not any(model.as_bool for model in panel.checked.values())
        panel.close()
        assert str(viewport.camera_path) == "/World/StartCamera"
        assert not stage.GetPrimAtPath(GHOST) and not stage.GetPrimAtPath("/World/Phase1Review")
        assert not (output / "human_reviews").exists()
        assert not [layer.identifier for layer in stage.GetUsedLayers() if not layer.anonymous and layer.dirty]
        fresh = audit_manifest(report["manifest_path"])
        assert report["manifest_sha256"] == fresh["manifest_sha256"] and report["components"] == fresh["components"]
        result = {"state": "passed", "scope": "relocated fixture position-IK UI, NOT physical grasp/cut",
                  "default_head_pov_survives_target_change": True, "explicit_closeup_and_head_switch": True,
                  "six_gallery_images_from_mounted_head_848x408": True,
                  "app_updates_during_search": updates, "robot_and_target_unchanged_by_check": True,
                  "outline_does_not_move_robot": True, "geometry_change_invalidates": True,
                  "cancel_discards_result": True, "human_review_performed": False,
                  "source_assets_unchanged": True, "training_input_allowed": False,
                  "screenshot": screenshot.name, "diagnostic": str(saved[0])}
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("REACHABILITY_SMOKE_PASSED " + str(output), flush=True)
        exit_code = 0
    except Exception:
        detail = traceback.format_exc()
        (output / "error.json").write_text(json.dumps({"state": "failed", "traceback": detail}, indent=2), encoding="utf-8")
        print("REACHABILITY_SMOKE_FAILED " + str(output) + "\n" + detail, flush=True)
    finally:
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
