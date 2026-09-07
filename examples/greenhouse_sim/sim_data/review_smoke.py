"""Headless Isaac smoke for the real review UI and source-layer preservation.

Produces a review-only viewport PNG, not a training observation or human review.
"""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback
import uuid

from sim_data.audit import DEFAULT_PACK, ROOT, audit_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capture-ui", action="store_true", help="Also capture the rendered review gallery window")
    args = parser.parse_args()
    output = args.output or ROOT / "data/sim_data/review_smokes" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    if output.exists() or output.resolve().is_relative_to(args.package.resolve()):
        parser.error("Choose a new output directory outside source assets")
    output.mkdir(parents=True)
    report = audit_manifest(args.package / "plants/components/seed101_full/manifest.json")

    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True, "hide_ui": not args.capture_ui, "width": 1280, "height": 720,
                         "window_width": 1600, "window_height": 1000,
                         "renderer": "RaytracedLighting", "sync_loads": False, "multi_gpu": False})
    exit_code = 0
    try:
        import omni.usd
        from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
        from pxr import Usd, UsdGeom, UsdLux
        from sim_data.geometry import assemble_plant
        from sim_data.gallery import BatchReviewPanel

        omni.usd.get_context().new_stage()
        stage = omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, "Z")
        # Match the package launcher: Kit camera tools may author on updates.
        stage.SetEditTarget(stage.GetSessionLayer())
        paths = assemble_plant(stage, "/World/Plant", report)
        with Usd.EditContext(stage, stage.GetSessionLayer()):
            UsdGeom.Camera.Define(stage, "/World/StartCamera")
            UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1500)
        for _ in range(10):
            app.update()
        viewport = get_active_viewport()
        viewport.set_texture_resolution((1280, 720))
        viewport.set_active_camera("/World/StartCamera")
        panel = BatchReviewPanel(stage, [{"plant_root": "/World/Plant", "manifest_path": report["manifest_path"],
                                         "component_paths": paths}], viewport, output / "human_reviews",
                                 package=args.package, history_directories=[])
        panel.move(1)
        panel.move(-1)
        panel.isolate()
        assert panel.scene.visibility_paths
        panel.isolate()
        assert not panel.scene.visibility_paths
        panel.save("anatomy_confirmed")  # Empty reviewer/notes must reject, never create evidence.
        assert not (output / "human_reviews").exists()
        panel.start_gallery()
        deadline = time.monotonic() + 180
        while not panel.capture_task.done():
            app.update()
            if time.monotonic() > deadline:
                raise RuntimeError("Batch gallery timed out")
        panel.capture_task.result()
        assert len(panel.evidence) == 6, panel.gallery_message.text
        assert not any(model.as_bool for model in panel.checked.values())
        assert len({key.split('/')[0] for key in panel.evidence}) >= 2
        panel.gallery_reason.get_item_value_model().set_value(1)
        panel.select_captured()
        panel.save_selected()  # Empty reviewer rejects even after explicit selection.
        assert not (output / "human_reviews").exists()
        panel.clear_selection()
        assert not any(model.as_bool for model in panel.checked.values())
        if args.capture_ui:
            import omni.kit.renderer_capture
            panel.window.set_visibility_changed_fn(None)
            panel.window.visible = False
            panel.gallery.position_x = 0
            panel.gallery.position_y = 0
            for _ in range(20):
                app.update()
            renderer_capture = omni.kit.renderer_capture.acquire_renderer_capture_interface()
            renderer_capture.capture_next_frame_swapchain(str(output / "gallery_ui.png"))
            for _ in range(20):
                app.update()
            from PIL import Image
            deadline = time.monotonic() + 15
            while True:
                try:
                    with Image.open(output / "gallery_ui.png") as image:
                        image.verify()
                    break
                except (OSError, SyntaxError):
                    if time.monotonic() > deadline:
                        raise RuntimeError("Gallery UI screenshot did not complete")
                    app.update()
        for _ in range(100):
            app.update()
        capture = capture_viewport_to_file(viewport, str(output / "review_only.png"))
        pending = asyncio.ensure_future(capture.wait_for_result())
        for _ in range(300):
            app.update()
            if pending.done():
                pending.result()
                break
        else:
            raise RuntimeError("Review capture timed out")
        # Viewport completion can precede the encoder's asynchronous file write.
        from PIL import Image
        deadline = time.monotonic() + 15
        while True:
            try:
                with Image.open(output / "review_only.png") as captured:
                    assert captured.size == (1280, 720)
                    captured.verify()
                break
            except (OSError, SyntaxError):
                if time.monotonic() > deadline:
                    raise RuntimeError("Review PNG was not completely written")
                app.update()
                time.sleep(0.01)
        assert (output / "review_only.png").stat().st_size > 10000
        panel.close()
        remaining = stage.GetPrimAtPath("/World/Phase1Review")
        assert not remaining, [(s.layer.identifier, str(s.path)) for s in remaining.GetPrimStack()]
        assert not stage.GetPrimAtPath("/World/Phase1ReviewSample")
        assert UsdGeom.Imageable(stage.GetPrimAtPath('/World/Plant')).ComputeVisibility() != 'invisible'
        dirty_sources = [layer.identifier for layer in stage.GetUsedLayers() if not layer.anonymous and layer.dirty]
        assert not dirty_sources, f"Dirty source layers: {dirty_sources}"
        after = audit_manifest(report["manifest_path"])
        assert report["manifest_sha256"] == after["manifest_sha256"]
        assert report["components"] == after["components"]
        for catalog_report in panel.reports:
            fresh = audit_manifest(catalog_report["manifest_path"])
            assert catalog_report["manifest_sha256"] == fresh["manifest_sha256"]
            assert catalog_report["components"] == fresh["components"]
        result = {"state": "passed", "scope": "multi_plant_batch_review_ui",
                  "next_previous_isolate_restore": True, "source_assets_unchanged": True,
                  "gallery_cards": len(panel.evidence), "review_catalog_plants": len(panel.reports),
                  "unchecked_by_default": True, "empty_reviewer_rejected": True,
                  "human_review_performed": False, "training_input_allowed": False,
                  "gallery_ui_capture_requested": args.capture_ui,
                  "capture": "review_only.png"}
        (output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print("REVIEW_SMOKE_PASSED " + str(output), flush=True)
    except Exception:
        exit_code = 1
        detail = traceback.format_exc()
        (output / "error.json").write_text(json.dumps({"state": "failed", "traceback": detail}, indent=2), encoding="utf-8")
        print("REVIEW_SMOKE_FAILED " + str(output) + "\n" + detail, flush=True)
    finally:
        app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
