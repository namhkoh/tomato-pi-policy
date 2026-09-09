"""Bounded, clean full-greenhouse RGB-D pilot. No live/lab robot connection.

Run with Isaac Python from examples/greenhouse_sim:
python -m sim_data.capture_pilot --drafts ../../data/sim_data/cut_region_drafts/<run>/draft_labels.json
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback
import uuid

import numpy as np

from .audit import ROOT, DEFAULT_PACK
from .capture_contract import (PILOT_TARGETS, RESOLUTION, depth_evidence, fingerprint, jsonable,
                               project, validate_static_capture, write_sample)


def load_drafts(path):
    from .label_drafts import prepare_drafts
    saved = json.loads(Path(path).read_text(encoding="utf-8"))
    if saved.get("state") != "draft_ready_for_human_review" or "cut_rule" not in saved:
        raise ValueError("Expected a completed prototype cut-region review packet")
    current, reports = prepare_drafts(saved["source_variants_path"], saved["cut_rule"])
    if saved["source_variants_sha256"] != current["source_variants_sha256"] or saved["cut_rule_sha256"] != current["cut_rule_sha256"]:
        raise ValueError("Stale draft provenance")
    if len(saved["labels"]) != len(current["labels"]):
        raise ValueError("Draft target count mismatch")
    for row, expected in zip(saved["labels"], current["labels"]):
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError(f"Draft/source mismatch at {expected['draft_id']}")
    rows = [r for r in saved["labels"] if r["draft_id"] in PILOT_TARGETS]
    if tuple(r["draft_id"] for r in rows) != PILOT_TARGETS:
        raise ValueError("Expected exactly the provisional B03/B05/B06 targets")
    if any(r["cut_region_proposal"]["geometry_warnings"] for r in rows):
        raise ValueError("A pilot target has unresolved geometry warnings")
    return saved, reports, rows


_frame_annotator_registered = False


def make_writer(rep, *, include_instances=False, instance_backend='legacy'):
    global _frame_annotator_registered
    if not _frame_annotator_registered:
        from omni.syntheticdata import SyntheticData
        rep.AnnotatorRegistry.register_annotator_from_node(
            name="pilot_render_frame", node_type_id="omni.syntheticdata.SdFrameIdentifier",
            input_rendervars=[SyntheticData.NodeConnectionTemplate("PostProcessDispatch",
                attributes_mapping={"outputs:renderResults":"inputs:renderResults"})])
        _frame_annotator_registered = True
    class PilotFrameWriter(rep.Writer):
        def __init__(self):
            self.version = "0.1.0"
            self.annotators = [rep.AnnotatorRegistry.get_annotator(name, device="cpu") for name in
                               ("rgb", "distance_to_image_plane", "camera_params", "ReferenceTime", "pilot_render_frame")]
            if include_instances:
                from .native_instances import LEGACY, FAST
                names={'legacy':[LEGACY],'fast':[FAST],'compare':[LEGACY,FAST]}[instance_backend]
                self.annotators.extend(rep.AnnotatorRegistry.get_annotator(
                    name, init_params={"colorize": False}, device="cpu") for name in names)
            self.sequence, self.latest, self.request_index = 0, None, 0
            self.capture_error=None

        def write(self, data):
            # Copy the whole synchronized callback payload before another render can reuse buffers.
            try:
                if include_instances and instance_backend!='legacy':
                    from .native_instances import normalize_payload
                    data=normalize_payload(data,instance_backend)
                self.latest = deepcopy(data)
                self.sequence += 1
            except Exception as exc:
                self.capture_error=exc
                raise

        def write_metadata(self):
            self._is_metadata_written = True  # The pilot writes its own complete manifest.

    return PilotFrameWriter()


def step_payload(rep, writer, subframes=8):
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    timeline.pause()
    for _ in range(4):
        writer.request_index += 1
        previous_sequence = writer.sequence
        rep.orchestrator.step(rt_subframes=subframes, pause_timeline=True, delta_time=0.0, wait_for_render=True)
        if getattr(writer,'capture_error',None) is not None:
            raise RuntimeError('Native writer rejected a callback') from writer.capture_error
        if writer.latest is not None and writer.sequence > previous_sequence:
            return writer.latest
    raise RuntimeError("No fresh synchronized writer callback after four attempts; refuse cached observations")


def calibration_smoke(app, rep, *, include_instances=False, instance_backend='legacy'):
    """Real GPU two-pose check: known front surface at 2 m; no greenhouse altered."""
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux
    from .capture_scene import calibration, scene_guard
    from .review_camera import HEAD_CAMERA
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    cube = UsdGeom.Cube.Define(stage, "/World/CalibrationCube")
    cube.CreateSizeAttr(1)
    cube_op = cube.AddTranslateOp()
    cube_op.Set(Gf.Vec3d(0, 0, -2.5))
    UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(1000)
    camera = UsdGeom.Camera.Define(stage, HEAD_CAMERA)
    camera.CreateFocalLengthAttr(24)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(20.955*408/848)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 100))
    camera_op = camera.AddTranslateOp()
    camera_op.Set(Gf.Vec3d(0))
    product = rep.create.render_product(HEAD_CAMERA, RESOLUTION)
    writer = make_writer(rep, include_instances=include_instances, instance_backend=instance_backend)
    writer.attach([product])
    checks, previous, first = [], None, None
    try:
        for offset, expected_depth in ((0.0, 2.0), (.15, 2.4)):
            camera_op.Set(Gf.Vec3d(offset, 0, 0))
            cube_op.Set(Gf.Vec3d(0, 0, -expected_depth-.5))
            # Warm-up and final data are both from the same unchanged static state.
            for _ in range(3):
                step_payload(rep, writer)
            guard = scene_guard(stage)
            payload = step_payload(rep, writer)
            rgb, depth, valid, reference, previous = validate_static_capture(
                payload, calibration(stage), guard, scene_guard(stage), writer.sequence, previous)
            if not valid[204,424] or abs(float(depth[204,424]) - expected_depth) > .002:
                raise ValueError("Known-plane depth does not match optical-axis Z")
            if not valid[204,470] or abs(float(depth[204,470]) - expected_depth) > .002:
                raise ValueError("Off-axis depth is not optical-axis Z")
            if first is not None and np.array_equal(rgb, first):
                raise ValueError("Moving the camera did not change RGB; possible stale frame")
            first = rgb
            if include_instances:
                from .capture_visibility import decode_instances
                ids, labels = decode_instances(payload)
                if labels.get(int(ids[204,424])) != "/World/CalibrationCube":
                    raise ValueError("Native instance identity does not match the known calibration cube")
            checks.append({"camera_x_m": offset, "expected_depth_m":expected_depth,
                           "measured_center_depth_m": float(depth[204,424]),
                           "reference_time": reference, "native_render_frame":jsonable(payload["pilot_render_frame"]),
                           "freshness":previous})
        if include_instances:
            # Identity and depth must change together when a known occluder is
            # introduced/removed. These diagnostic renders never become inputs.
            from .capture_visibility import decode_instances
            blocker = UsdGeom.Cube.Define(stage, "/World/KnownOccluder")
            blocker.CreateSizeAttr(.2)
            blocker.AddTranslateOp().Set(Gf.Vec3d(.15, 0, -1.1))
            last_ids = None
            for visible in (True, False):
                blocker.CreateVisibilityAttr("inherited" if visible else "invisible")
                for _ in range(4):
                    step_payload(rep, writer)
                guard = scene_guard(stage)
                payload = step_payload(rep, writer)
                _, depth, valid, reference, token = validate_static_capture(
                    payload, calibration(stage), guard, scene_guard(stage), writer.sequence)
                ids, labels = decode_instances(payload)
                expected = "/World/KnownOccluder" if visible else "/World/CalibrationCube"
                if labels.get(int(ids[204,424])) != expected or not valid[204,424]:
                    raise ValueError("Known foreground occlusion identity check failed")
                if abs(float(depth[204,424]) - (1.0 if visible else 2.4)) > .002:
                    raise ValueError("Occlusion segmentation and depth disagree")
                digest = hashlib.sha256(ids.tobytes()).hexdigest()
                if digest == last_ids:
                    raise ValueError("Stale segmentation after changing known occluder")
                last_ids = digest
                checks.append({"known_occluder_visible": visible, "expected_pixel_prim": expected,
                               "measured_depth_m": float(depth[204,424]), "instance_sha256": digest,
                               "reference_time": reference, "freshness": token})
    finally:
        writer.detach()
        product.destroy()
    print("CAPTURE_CALIBRATION_SMOKE_PASSED " + json.dumps(checks), flush=True)
    return {"passed": True, "scope": "two_camera_poses_pinhole_projection_and_known_plane_depth",
            "native_identity_and_occluder_smoke_passed": include_instances, "checks": checks}


def source_hashes(stage):
    return {layer.realPath: hashlib.sha256(Path(layer.realPath).read_bytes()).hexdigest()
            for layer in stage.GetUsedLayers() if not layer.anonymous and layer.realPath and Path(layer.realPath).is_file()}


def review_markdown(manifest):
    lines = ["# Full-greenhouse head-camera RGB-D pilot", "",
             "Clean input images are rendered from the mounted RB-Y1 head camera at 848x408. "
             "The loaded greenhouse, neighbouring vines and original foliage remain present; no diagnostic isolation or recolouring is used.", "",
             "This is a static capture/plumbing pilot, NOT an approved training dataset, physical trajectory or cut demonstration. "
             "Targets B03/B05/B06 are provisional. B01/B02/B04 remain held for junction review.", "",
             "Poses use real URDF-limited head joints and whole-robot aisle translations. They are geometry-guided offline snapshots, "
             "not collision-validated poses or an autonomous view-selection policy. Cut labels cover only the selected target, not every eligible branch.", "",
             "White cross = projected proposed cut centre; magenta = projected 10-20 mm interval. "
             "Markers in review copies may mark HIDDEN geometry. A depth-consistent pixel is not confirmed target visibility. "
             "No easy/medium/hard label or automatic cut instruction is assigned.", "",
             "Synchronization scope: frozen scene + one writer payload, camera/projection validation and fresh callback/content checks. "
             "The installed renderer reports unavailable native frame IDs/zero simulation time in this static mode. "
             "Application callback counts are NOT engine frame IDs. This pilot is not valid for dynamic demonstration recording.", "",
             "| Sample | Target | Nominal pixel | Depth evidence |", "|---|---|---|---|"]
    for sample in manifest["samples"]:
        lines.append(f"| {sample['sample_id']} | {sample['target_review_id']} | {sample['nominal_pixel_xy']} | {sample['depth_status']} |")
    if "viewpoint_selection" in manifest:
        search = manifest["viewpoint_selection"]
        lines += ["", "## Viewpoint and visibility audit", "",
                  f"{search['clear_view_sample_count']} selected samples pass the provisional clear-view engineering gates. "
                  "Passing is not training approval. Failed-gate images are retained explicitly for diagnostic review, not called easy examples.", "",
                  "Views are screened with conservative visual geometry bounds, floor support and URDF head limits. "
                  "This is not full robot self-collision or path/dynamics validation. All camera optics, lighting and plant geometry stay unchanged.", "",
                  "Visibility requires the exact rendered petiole identity AND consistent depth at the projected pixel. "
                  "Child leaves are separate organs. Interval coverage counts unique projected pixels, not an amodal surface visibility fraction. "
                  "Only the two detailed plants have organ-level provenance; backdrop geometry is explicitly unmapped at organ level.", "",
                  "[All candidate poses and rejection reasons](viewpoint_search.json)", "",
                  "| Sample | Nominal identity/depth | Estimated width (px) | Interval (px) | Clear-view gates |",
                  "|---|---|---|---|---|"]
        for sample in manifest["samples"]:
            q = sample["quality"]
            lines.append(f"| {sample['sample_id']} | {sample['nominal_visibility_status']} | "
                         f"{q['estimated_petiole_diameter_px']:.2f} | {q['projected_interval_length_px']:.2f} | "
                         f"{'pass' if q['clear_view_gate_passed'] else ', '.join(q['clear_view_rejection_reasons'])} |")
    for sample in manifest["samples"]:
        name = sample["sample_id"]
        lines += ["", f"## {name} / {sample['target_review_id']}", "", "Clean full-scene input:", "",
                  f"![Clean greenhouse RGB]({name}/inputs/rgb.png)", "", "Separate review overlay (never a model input):", "",
                  f"![Projected proposal review]({name}/review/overlay.png)", "",
                  f"[Calibration, provisional labels and capture QA]({name}/sample.json)", "",
                  f"[Raw metric depth]({name}/inputs/depth_m.npy) / [Depth validity mask]({name}/inputs/depth_valid.png)"]
        if "quality" in sample:
            lines += ["", "Exact visible-petiole mask highlighted in green (review only; no inferred hidden pixels):", "",
                      f"![Rendered target identity]({name}/review/visible_target.png)", "",
                      f"[Binary visible-target mask]({name}/supervision/target_visible.png) / "
                      f"[Instance and organ identity catalogue]({name}/supervision/identities.json)"]
    return "\n".join(lines) + "\n"


def run(app, args, manifest):
    import carb
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    from pxr import Usd, UsdGeom
    from isaacsim.core.version import get_version
    from launch_sim_data import load_local_payloads, populate
    from .candidate_branches import add_candidate_branches
    from .capture_scene import calibration, capture_root, freeze_rigid_bodies, scene_guard, set_snapshot_pose, target_world_geometry
    from .floor_alignment import PACKAGE_FLOOR
    from .review_camera import HEAD_CAMERA
    from .robot_preview import add_robot_preview
    manifest["isaac_sim_version"] = str(get_version())
    manifest["replicator_extension"] = str(Path(rep.__file__).resolve())
    manifest["synchronous_rendering"] = {"supportMultiTickRate":carb.settings.get_settings().get("/rtx/hydra/supportMultiTickRate")}
    saved, reports, rows = load_drafts(args.drafts)
    manifest["calibration_smoke"] = calibration_smoke(app, rep, include_instances=args.refine_views)
    if args.smoke_only:
        return
    rep.set_global_seed(0)
    package = args.package.resolve()
    scene = package / "house/green_house_base.usd"
    context = omni.usd.get_context()
    wrapper = capture_root(scene)
    # Kit reinitializes anonymous identifiers passed to open_stage. Compose the
    # source into the context's new anonymous stage instead, with payloads deferred.
    context.new_stage()
    stage = context.get_stage()
    stage.SetLoadRules(Usd.StageLoadRules.LoadNone())
    stage.GetRootLayer().subLayerPaths = list(wrapper.subLayerPaths)
    if not stage.GetRootLayer().anonymous or not stage.GetPrimAtPath("/World/Gutters"):
        raise RuntimeError("Anonymous wrapper did not compose the supplied greenhouse")
    # Stage-level settings are authored only in the anonymous wrapper. Asset
    # relative paths still resolve against the original source sublayer.
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetEditTarget(stage.GetSessionLayer())
    excluded = load_local_payloads(stage)
    records = []
    gutter_x, counts = populate(stage, package, app, records)
    variants = [add_candidate_branches(stage, record, 3) for record in records]
    by_variant = {v["variant_id"]:v for v in variants}
    if {r["variant_id"] for r in rows} - by_variant.keys():
        raise ValueError("Loaded greenhouse variants do not match reviewed target identities")
    robot = add_robot_preview(stage, gutter_x=gutter_x, floor_path=PACKAGE_FLOOR, right_tool="gripper")
    manifest["rigid_bodies_frozen_in_capture_session_only"] = freeze_rigid_bodies(stage)
    sys.path.insert(0, str(package / "env_panel"))
    from tomato_env import daylight
    lighting = daylight.apply(stage, day=172, minutes=13*60, intensity=1500, dome_intensity=1200)
    context.get_selection().clear_selected_prim_paths()
    omni.timeline.get_timeline_interface().pause()
    settings = carb.settings.get_settings()
    settings.set_bool("/app/runLoops/main/rateLimitEnabled", False)
    # No target-dependent exposure tuning or lights: use the existing preview lighting.
    manifest.update(scene=str(scene), package=str(package), scene_counts=counts,
                    source_loaded_via_anonymous_wrapper=True,
                    unbundled_external_prop_roots_excluded=excluded, lighting=lighting,
                    robot=robot, variants=variants, source_draft_path=str(args.drafts.resolve()),
                    source_draft_sha256=hashlib.sha256(args.drafts.read_bytes()).hexdigest(),
                    cut_rule=saved["cut_rule"], cut_rule_sha256=saved["cut_rule_sha256"],
                    sources=[{"plant_id":r["plant_id"],"manifest_path":r["manifest_path"],
                              "manifest_sha256":r["manifest_sha256"]} for r in reports],
                    held_targets=["B01", "B02", "B04"], seed=0,
                    renderer="RaytracedLighting", source_usd_sha256=source_hashes(stage),
                    source_hash_scope="all_loaded_nonanonymous_USD_layers_plus_draft_manifest_component_fingerprints")
    if args.refine_views:
        from .capture_search import run_refined_capture
        run_refined_capture(stage, rep, args, manifest, robot, records, reports, variants, rows)
        finish_capture(stage, args, manifest)
        return
    product = rep.create.render_product(HEAD_CAMERA, RESOLUTION)
    writer = make_writer(rep)
    writer.attach([product])
    previous = None
    # Spread image-plane targets instead of teaching an always-centre shortcut.
    views = [(-.18, (.36, .43)), (0., (.57, .60)), (.18, (.43, .66))]
    try:
        for row_index, row in enumerate(rows):
            variant = by_variant[row["variant_id"]]
            for view_index in range(len(views)):
                if len(manifest["samples"]) >= args.max_samples:
                    break
                offset, fraction = views[(view_index+row_index) % len(views)]
                world = target_world_geometry(stage, variant, row)
                pose = set_snapshot_pose(stage, robot, world["nominal_world_m"], offset,
                                         [fraction[0]*848, fraction[1]*408])
                # A fixed warm-up after static pose changes; no mutation while recording.
                for _ in range(6):
                    step_payload(rep, writer)
                before = scene_guard(stage)
                cal = calibration(stage)
                payload = step_payload(rep, writer)
                if scene_guard(stage) != before or fingerprint(calibration(stage)) != fingerprint(cal):
                    raise ValueError("Scene/camera changed during synchronized capture")
                rgb, depth, valid, reference, previous = validate_static_capture(
                    payload, cal, before, scene_guard(stage), writer.sequence, previous)
                nominal = project([world["nominal_world_m"]], cal)[0]
                interval = project(world["interval_world_m"], cal)
                evidence = depth_evidence(nominal, depth, valid, row["cut_region_proposal"]["nominal"]["petiole_radius_m"])
                if nominal["projection_status"] != "in_frame":
                    raise ValueError("Head pose solver did not keep the intended target in frame")
                sample_id = f"sample_{len(manifest['samples'])+1:04d}"
                meta = {"schema_version":"greenhouse.rgbd_pilot_sample.v1", "sample_id":sample_id,
                        "state":"complete_pending_review", "training_sample_approved":False,
                        "input_policy":{"clean_full_scene":True, "diagnostic_overlays":False,
                                        "isolation":False, "allowed_observation_files":["inputs/rgb.png", "inputs/depth_m.npy", "inputs/depth_valid.png"]},
                        "calibration":cal, "robot_snapshot":pose,
                        "synchronization":{"method":"frozen_scene_single_writer_payload_camera_and_content_checks",
                                           "reference_time":reference, "writer_callback_sequence":writer.sequence,
                                           "native_render_frame":jsonable(payload["pilot_render_frame"]),
                                           "engine_frame_id_verified":False,"dynamic_recording_supported":False,
                                           "freshness":previous,
                                           "scene_state_sha256":before, "scene_unchanged_during_capture":True,
                                           "stale_frame_fallback_allowed":False, "rt_subframes":8,
                                           "timeline_seconds":omni.timeline.get_timeline_interface().get_current_time()},
                        "rendered_camera_params":jsonable(payload["camera_params"]),
                        "quality":{"valid_depth_fraction":float(valid.mean()), "rgb_std":float(rgb.std())},
                        "supervision":{"review_id":row["draft_id"],"target_id":row["target_id"],
                            "variant_id":row["variant_id"],"split_group":row["split_group"],
                            "coverage":"selected_target_probe_not_exhaustive_scene_annotations",
                            "cut_region_proposal":row["cut_region_proposal"], **world,
                            "nominal_projected":nominal,"projected_interval":interval,"depth_evidence":evidence,
                            "semantic_instance_masks":"not_exported_in_this_pilot", "difficulty":"unassigned",
                            "human_cut_approval":False,"grasp_region":None,"executable_trajectory":None,
                            "physics_validated":False,"visibility_confirmed":False}}
                write_sample(args.output/sample_id, rgb, depth, valid, meta)
                manifest["samples"].append({"sample_id":sample_id,"target_review_id":row["draft_id"],
                                             "nominal_pixel_xy":nominal["pixel_xy"], "depth_status":evidence["status"]})
                print("PILOT_SAMPLE " + json.dumps(manifest["samples"][-1]), flush=True)
    finally:
        writer.detach()
        product.destroy()
    finish_capture(stage, args, manifest)


def finish_capture(stage, args, manifest):
    current_hashes = source_hashes(stage)
    if any(current_hashes.get(p) != sha for p, sha in manifest["source_usd_sha256"].items()):
        raise ValueError("Source USD asset changed during capture")
    load_drafts(args.drafts)  # Re-audit manifests/meshes, not only loaded USD layers.
    if hashlib.sha256(args.drafts.read_bytes()).hexdigest() != manifest["source_draft_sha256"]:
        raise ValueError("Draft packet changed during capture")
    dirty_layers = [layer.identifier for layer in stage.GetUsedLayers() if not layer.anonymous and layer.dirty]
    if dirty_layers:
        raise ValueError(f"Capture dirtied a source USD layer: {dirty_layers}")
    manifest["source_assets_unchanged"] = True
    (args.output/"review.md").write_text(review_markdown(manifest), encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--package", type=Path, default=DEFAULT_PACK)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-samples", type=int, default=9, choices=range(1, 10))
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--view-plan", type=Path, help="Optional CPU-screened B03/B06 base/head view plan; requires --refine-views")
    parser.add_argument("--refine-views", action="store_true",
                        help="Screen bounded nearer robot poses and export native organ-identity visibility evidence")
    args = parser.parse_args(argv)
    if args.view_plan:
        if not args.refine_views:
            parser.error("--view-plan requires --refine-views")
        from .viewpoint_plan import load_plan
        load_plan(args.view_plan, args.drafts, args.package)
    load_drafts(args.drafts)
    args.output = (args.output or ROOT/"data/sim_data/rgbd_pilots"/
                   (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_")+uuid.uuid4().hex[:6])).resolve()
    if args.output.exists() or args.output.is_relative_to(args.package.resolve()):
        parser.error("Choose a NEW output directory outside the source package")
    args.output.mkdir(parents=True)
    manifest = {"schema_version":"greenhouse.rgbd_pilot.v1", "state":"initializing", "samples":[],
                "training_dataset_approved":False,"physics_enabled":False,"human_review_performed":False,
                "implementation_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                           [Path(__file__), Path(__file__).with_name("capture_scene.py"),
                                            Path(__file__).with_name("capture_contract.py")]}}
    if args.refine_views:
        manifest["implementation_sha256"].update({name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ("capture_visibility.py", "capture_viewpoints.py", "capture_search.py", "viewpoint_plan.py")})
    app = None
    code = 1
    try:
        from isaacsim import SimulationApp
        app = SimulationApp({"headless":True,"width":848,"height":408,"multi_gpu":False,
                             "renderer":"RaytracedLighting","sync_loads":False})
        run(app,args,manifest)
        manifest["state"] = ("calibration_smoke_passed" if args.smoke_only else
                             "pilot_ready_for_review" if manifest["samples"] else "blocked_no_screened_viewpoints")
        code = 0
    except Exception:
        manifest["state"] = "failed_do_not_train"
        manifest["error"] = traceback.format_exc()
        print(manifest["error"], flush=True)
    finally:
        (args.output/"manifest.json").write_text(json.dumps(jsonable(manifest),indent=2,allow_nan=False),encoding="utf-8")
        if app is not None:
            app.close()
    print("PILOT_RESULT " + manifest["state"] + " " + str(args.output), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
