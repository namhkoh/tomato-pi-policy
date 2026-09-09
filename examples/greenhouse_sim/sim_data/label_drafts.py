"""Build draft anatomy labels and diagnostic renders, never human approvals.

Example (from examples/greenhouse_sim, using Isaac Python):
python -m sim_data.label_drafts --variants path/to/candidate_branches.json
This reconstructs the saved plant variants in a separate headless scene. It does
not capture the live greenhouse or change its robot, cameras or annotations.
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import uuid

from .audit import ROOT, audit_manifest
from .candidate_branches import plan_branches
from .cut_regions import add_cut_region_overlay, load_rule, propose_cut_region, rule_fingerprint


def prepare_drafts(path, cut_rule=None):
    """Re-audit sources and regenerate variants rather than trusting stale labels."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    variants = json.loads(raw)
    if cut_rule is not None:
        rule_fingerprint(cut_rule)
    if not isinstance(variants, list) or not variants:
        raise ValueError("Expected a nonempty candidate-variant list")
    reports, labels, seen = [], [], set()
    for saved in variants:
        source = audit_manifest(saved["source_manifest_path"])
        variant = plan_branches(source, len(saved["candidates"]))
        for key, expected in variant.items():
            # Saved candidates additionally contain launch-time world positions.
            actual = saved.get(key)
            if key == "candidates":
                actual = [{k: t.get(k) for k in expected[i]} for i, t in enumerate(actual or [])]
            if actual != expected:
                raise ValueError(f"Variant/source mismatch at {key}; rebuild preview before labelling")
        if variant["variant_id"] in seen:
            raise ValueError("Duplicate variant in review packet")
        seen.add(variant["variant_id"])
        report = deepcopy(source)
        report["source_plant_id"] = source["plant_id"]
        report["plant_id"] = variant["variant_id"]
        report["components"].update(deepcopy(variant["added_components"]))
        for key in variant["replaced_stub_ids"]:
            del report["components"][key]
        report["targets"] = deepcopy(variant["candidates"])
        reports.append(report)
        for target in report["targets"]:
            key = target["component_id"]
            label = {
                "schema_version": "greenhouse.anatomy_draft.v1", "draft_id": f"B{len(labels) + 1:02d}",
                "target_id": target["target_id"], "variant_id": variant["variant_id"],
                "source_plant_id": source["plant_id"], "split_group": source["plant_id"],
                "component_id": key, "main_stem_parent_id": target["parent_component_id"],
                "leaf_component_ids": [k for k in target["expected_detached_component_ids"]
                                       if report["components"][k]["type"] == "leaf"],
                "expected_detached_component_ids": target["expected_detached_component_ids"],
                "protected_descendant_ids": target["protected_descendant_ids"],
                "attachment_plant_m": target["attachment_plant_m"],
                "petiole_centerline_capsules_local_m": report["components"][key]["capsules_local_m"],
                "label_origin": "source_manifest_and_validated_augmentation_recipe",
                "proposed_anatomy": "attached_leaf_bearing_petiole_directly_parented_to_main_stem",
                "human_review_status": "pending", "human_review_performed": False,
                "cut_approval": False, "cut_rule_status": "not_defined",
                "canonical_cut_point_m": None, "admissible_cut_region": None, "grasp_region": None,
                "agronomic_eligibility": "unreviewed", "physical_executability": "not_tested",
                "greenhouse_occlusion": "not_measured", "robot_reachability": "not_tested",
                "training_input_allowed": False, "training_label_approved": False,
                "notes": "Attachment is NOT a cut label. Diagnostic isolation removes occluders; do not infer greenhouse visibility.",
                "source_manifest_sha256": source["manifest_sha256"],
                "source_component_hashes": {k: c["asset_sha256"] for k, c in source["components"].items()},
                "variant_sha256": variant["variant_sha256"],
            }
            if cut_rule is not None:
                label["schema_version"] = "greenhouse.anatomy_and_cut_draft.v2"
                label["cut_rule_status"] = "prototype_user_agreed_horticultural_validation_pending"
                label["cut_region_proposal"] = propose_cut_region(
                    report["components"][key], report["components"][target["parent_component_id"]], cut_rule)
                label["notes"] += " Proposed cut geometry is separate from approved labels; the full blade stroke has not been checked."
            labels.append(label)
    packet = {"schema_version": "greenhouse.anatomy_draft_packet.v2" if cut_rule else "greenhouse.anatomy_draft_packet.v1", "state": "draft",
            "source_variants_path": str(path), "source_variants_sha256": hashlib.sha256(raw).hexdigest(),
            "human_review_performed": False, "training_input_allowed": False,
            "render_scope": "offline_plant_frame_reconstruction_not_live_robot_view", "labels": labels}
    if cut_rule is not None:
        packet["cut_rule"] = deepcopy(cut_rule)
        packet["cut_rule_sha256"] = rule_fingerprint(cut_rule)
    return packet, reports


def label_color_counts(image_path):
    """Reject blank/material-loss renders; this is not semantic label approval."""
    import numpy as np
    from PIL import Image
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    counts = {"petiole_blue_pixels": int(((b > r + 25) & (b > g + 15)).sum()),
              "leaf_green_pixels": int(((g > r + 25) & (g > b + 15)).sum()),
              "parent_red_pixels": int(((r > g + 25) & (r > b + 25)).sum())}
    if min(counts.values()) < 25:
        raise ValueError(f"Missing diagnostic colours / blank target render: {counts}")
    return counts


def markdown_packet(packet):
    has_cut_rule = "cut_rule" in packet
    nominal_mm = packet["cut_rule"]["nominal_distance_m"] * 1000 if has_cut_rule else None
    interval_mm = [s * 1000 for s in packet["cut_rule"]["accepted_interval_m"]] if has_cut_rule else None
    rows = ["# Draft labels: six added branches" if len(packet["labels"]) == 6 else "# Draft branch labels", "",
            "These are simulator-derived **anatomy and optional cut-geometry proposals**, not approved training or execution labels.", "",
            "Images are offline diagnostic reconstructions in plant coordinates, not live greenhouse or robot-head views. "
            "Neighbouring plants and the robot are absent. No visibility, reachability or cutting success is inferred.", "",
            "In the labelled views: **blue = target petiole; green = its leaf subtree; red = parent main stem; "
            "yellow = attachment (NOT the cutting point)**. Other organs are hidden for inspection.", "",
            "For each ID, reply `anatomy correct`, `wrong attachment`, `wrong leaf membership`, or `unclear`. " +
            (f"The prototype rule is {nominal_mm:g} mm nominal and {interval_mm[0]:g}-{interval_mm[1]:g} mm along the petiole; horticultural validation remains pending. "
             "White = nominal point; magenta = proposed interval. Marker sizes and band thickness are display-only, not radial tolerance. "
             "Please assess anatomy and cut-region placement separately. No grasp region or blade-clearance approval is set."
             if has_cut_rule else "This confirms anatomy only. The horticultural cut interval and grasp region remain unset."), "",
            "| ID | Source plant | Main-stem parent | Leaf meshes | Status |", "|---|---|---|---:|---|"]
    for row in packet["labels"]:
        rows.append(f"| {row['draft_id']} | {row['source_plant_id']} | {row['main_stem_parent_id']} | "
                    f"{len(row['leaf_component_ids'])} | Pending review |")
    for row in packet["labels"]:
        rows += ["", f"## {row['draft_id']} — {row['component_id']}", "",
                 f"Target: `{row['target_id']}`", "",
                 f"Attachment in the source plant frame (metres; not a cut point): `{row['attachment_plant_m']}`.", ""]
        if has_cut_rule:
            proposal = row["cut_region_proposal"]
            rows += [f"Cut geometry: **{proposal['status']}**. Rule: `{proposal['rule_id']}`.", ""]
            if proposal["nominal"]:
                rows += [f"Proposed nominal point (plant-frame metres): `{proposal['nominal']['point_plant_m']}`.", "",
                         f"Accepted prototype interval: **{interval_mm[0]:g}-{interval_mm[1]:g} mm arc length**, starting at the manifest attachment, not the yellow marker edge. "
                         "This is not a guaranteed external stub length or a full knife pose.", ""]
                diagnostic = proposal.get("parent_proxy_diagnostic", {})
                gap = diagnostic.get("nominal_point_to_parent_surface_m")
                if gap is not None:
                    rows += [f"Nominal point to parent capsule surface: **{gap * 1000:.2f} mm** "
                             "(approximation; not blade clearance).", ""]
                if proposal.get("geometry_warnings"):
                    rows += [f"**Geometry warning:** `{', '.join(proposal['geometry_warnings'])}`. "
                             "Review the junction; do not treat the proposal as executable.", ""]
            else:
                rows += [f"No cut coordinates generated: `{', '.join(proposal['reason_codes'])}`.", ""]
        for caption, key in (("Unmodified appearance; whole-plant close-up", "context"),
                             ("Labelled isolated anatomy", "labelled"),
                             ("Opposite-side attachment detail (diagnostic)", "junction"),
                             ("Proposed cut interval close-up (NOT execution approval)", "cut_region")):
            if key in row.get("images", {}):
                rows += [f"{caption}:", "", f"![{row['draft_id']} {caption}]({row['images'][key]['file']})", ""]
        rows += ["Per-target human decision: **pending**. Approved cut/grasp labels: **unset**."]
    return "\n".join(rows) + "\n"


def render_packet(app, packet, reports, output):
    import omni.usd
    from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
    from PIL import Image
    from .geometry import assemble_plant, bounds_by_component, point_bounds_distance

    context = omni.usd.get_context()

    def capture(viewport, filename):
        for _ in range(45):
            app.update()
        task = asyncio.ensure_future(capture_viewport_to_file(viewport, str(output / filename)).wait_for_result())
        deadline = time.monotonic() + 30
        while not task.done():
            app.update()
            if time.monotonic() > deadline:
                raise RuntimeError("Draft render timed out")
        task.result()
        while True:
            try:
                with Image.open(output / filename) as captured:
                    if captured.size != (1280, 720):
                        raise ValueError("Unexpected draft image resolution")
                    captured.verify()
                break
            except (OSError, SyntaxError):
                if time.monotonic() > deadline:
                    raise RuntimeError("Draft image encoding timed out")
                app.update()
        return {"file": filename, "sha256": hashlib.sha256((output / filename).read_bytes()).hexdigest(),
                "resolution": [1280, 720], "training_input_allowed": False}

    for report in reports:
        context.new_stage()
        stage = context.get_stage()
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        UsdGeom.SetStageUpAxis(stage, "Z")
        stage.SetEditTarget(stage.GetSessionLayer())
        paths = assemble_plant(stage, "/World/Plant", report)
        UsdLux.DomeLight.Define(stage, "/World/Light").CreateIntensityAttr(600)
        light = UsdLux.DistantLight.Define(stage, "/World/Key")
        light.CreateIntensityAttr(600)
        light.AddRotateXYZOp().Set(Gf.Vec3f(-35, -30, 0))
        camera = UsdGeom.Camera.Define(stage, "/World/DraftCamera")
        camera.CreateFocalLengthAttr(24)
        camera.CreateHorizontalApertureAttr(20.955)
        camera.CreateVerticalApertureAttr(20.955 * 720 / 1280)
        camera.CreateClippingRangeAttr(Gf.Vec2f(.01, 100))
        camera_op = camera.AddTransformOp()
        for _ in range(15):
            app.update()
        viewport = get_active_viewport()
        if viewport is None:
            raise RuntimeError("No viewport for draft rendering")
        viewport.set_texture_resolution((1280, 720))
        viewport.set_active_camera(str(camera.GetPath()))
        bounds, meshes = bounds_by_component(stage, paths)
        owners = {p: key for key, p in paths.items()}
        mesh_owners = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue
            owner = prim
            while owner and str(owner.GetPath()) not in owners:
                owner = owner.GetParent()
            if owner:
                mesh_owners.append((prim, owners[str(owner.GetPath())]))
        for target in report["targets"]:
            row = next(r for r in packet["labels"] if r["target_id"] == target["target_id"])
            members = set(row["expected_detached_component_ids"])
            if any(meshes[k] == 0 or bounds[k].IsEmpty() for k in members):
                raise ValueError("Candidate has missing rendered geometry")
            box = Gf.Range3d()
            for key in members:
                box.UnionWith(bounds[key])
            box.UnionWith(Gf.Vec3d(*row["attachment_plant_m"]))
            centre = box.GetMidpoint()
            distance = max(.50, box.GetSize().GetLength() * 1.7)
            direction = Gf.Vec3d(.7, -.7, .25).GetNormalized()
            matrix = Gf.Matrix4d().SetLookAt(centre + direction * distance, centre, Gf.Vec3d(0, 0, 1)).GetInverse()
            camera_op.Set(matrix)
            diagnostic = None
            previous_edit = stage.GetEditTarget()
            try:
                row["images"] = {"context": capture(viewport, row["draft_id"] + "_context.png")}
                # An isolated override layer avoids rebuilding referenced material
                # prims via ImportFromString (Hydra can retain invalid bindings).
                diagnostic = Sdf.Layer.CreateAnonymous("draft_anatomy_overrides")
                stage.GetSessionLayer().subLayerPaths.insert(0, diagnostic.identifier)
                stage.SetEditTarget(diagnostic)
                materials = {}
                for name, color in {"petiole": (.05, .35, 1), "leaves": (.05, .9, .25),
                                    "main_stem": (1, .12, .08), "attachment": (1, .8, .02)}.items():
                    material = UsdShade.Material.Define(stage, "/World/DraftLabels/" + name)
                    shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/Shader")
                    shader.CreateIdAttr("UsdPreviewSurface")
                    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
                    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.8)
                    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                    materials[name] = material
                for prim, owner in mesh_owners:
                    role = ("main_stem" if owner == row["main_stem_parent_id"] else
                            "petiole" if owner == row["component_id"] else "leaves" if owner in members else None)
                    if role is None:
                        UsdGeom.Imageable(prim).GetVisibilityAttr().Set("invisible")
                    else:
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[role], UsdShade.Tokens.strongerThanDescendants)
                marker = UsdGeom.Sphere.Define(stage, "/World/DraftLabels/Attachment_NOT_CutPoint")
                marker.CreateRadiusAttr(.009)
                marker.AddTranslateOp().Set(Gf.Vec3d(*row["attachment_plant_m"]))
                UsdShade.MaterialBindingAPI.Apply(marker.GetPrim()).Bind(materials["attachment"])
                row["images"]["labelled"] = capture(viewport, row["draft_id"] + "_labelled.png")
                row["images"]["labelled"]["colour_qa"] = label_color_counts(output / row["images"]["labelled"]["file"])
                row["diagnostic_camera"] = {"camera_to_plant_usd_row_vectors": [list(r) for r in matrix],
                    "resolution": [1280, 720], "focal_length_mm": 24,
                    "horizontal_aperture_mm": 20.955, "vertical_aperture_mm": 20.955 * 720 / 1280,
                    "frame": "plant", "camera_axes": "+X right, +Y up, -Z forward"}
                # A second angle helps inspect nodes hidden by their own leaves.
                axis = Gf.Vec3d(*report["components"][row["component_id"]]["axis_plant"])
                junction_centre = Gf.Vec3d(*row["attachment_plant_m"]) + axis * .04
                side = Gf.Vec3d(-axis[1], axis[0], .35).GetNormalized()
                junction_matrix = Gf.Matrix4d().SetLookAt(junction_centre + side * .30, junction_centre,
                                                         Gf.Vec3d(0, 0, 1)).GetInverse()
                stage.SetEditTarget(previous_edit)
                camera_op.Set(junction_matrix)
                row["images"]["junction"] = capture(viewport, row["draft_id"] + "_junction.png")
                row["images"]["junction"]["camera_to_plant_usd_row_vectors"] = [list(r) for r in junction_matrix]
                row["images"]["junction"]["note"] = "Same diagnostic intrinsics; alternate camera pose. Marker radius 9 mm is exaggerated for readability."
                if row.get("cut_region_proposal", {}).get("status") == "proposed_geometry_only":
                    stage.SetEditTarget(diagnostic)
                    marker.CreateRadiusAttr(.006)
                    add_cut_region_overlay(stage, row)
                    proposal = row["cut_region_proposal"]
                    target_point = Gf.Vec3d(*proposal["nominal"]["point_plant_m"])
                    # Look partly from the distal petiole side, so the parent
                    # does not hide the nominal marker as it can in a side view.
                    cut_direction = (axis.GetNormalized() * .8 + side * .6).GetNormalized()
                    cut_matrix = Gf.Matrix4d().SetLookAt(target_point + cut_direction * .18, target_point,
                                                        Gf.Vec3d(0, 0, 1)).GetInverse()
                    stage.SetEditTarget(previous_edit)
                    camera_op.Set(cut_matrix)
                    row["images"]["cut_region"] = capture(viewport, row["draft_id"] + "_cut_region.png")
                    row["images"]["cut_region"]["camera_to_plant_usd_row_vectors"] = [list(r) for r in cut_matrix]
                    row["images"]["cut_region"]["display_only"] = {"attachment_marker_radius_m": .006,
                        "nominal_marker_radius_m": .003, "band_width_to_petiole_diameter_ratio": 1.2,
                        "radial_tolerance_defined": False}
                row["geometry_diagnostics"] = {
                    "own_mesh_count": sum(meshes[k] for k in members),
                    "attachment_to_parent_aabb_m": point_bounds_distance(row["attachment_plant_m"], bounds[row["main_stem_parent_id"]]),
                    "not_surface_or_collision_validation": True}
                print(f"DRAFT_RENDERED {row['draft_id']} {row['target_id']}", flush=True)
            finally:
                stage.SetEditTarget(previous_edit)
                if diagnostic is not None:
                    stage.GetSessionLayer().subLayerPaths.remove(diagnostic.identifier)
                for prim, _ in mesh_owners:
                    if UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
                        raise RuntimeError("Diagnostic isolation was not fully restored")
        if any(layer.dirty for layer in stage.GetUsedLayers() if not layer.anonymous):
            raise RuntimeError("Source USD layer unexpectedly dirtied by diagnostic rendering")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-render", action="store_true", help="Metadata proposals only; no visual review evidence")
    parser.add_argument("--cut-rule", type=Path, help="Versioned prototype JSON; adds proposed geometry, never approval")
    args = parser.parse_args(argv)
    cut_rule = load_rule(args.cut_rule) if args.cut_rule else None
    packet, reports = prepare_drafts(args.variants, cut_rule)
    output = args.output or ROOT / "data/sim_data/label_drafts" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
    output = output.resolve()
    if output.exists() or any(output.is_relative_to(Path(r["manifest_path"]).parents[3]) for r in reports):
        parser.error("Choose a new output directory outside the source package")
    output.mkdir(parents=True)
    app = None
    exit_code = 0
    try:
        if not args.no_render:
            from isaacsim import SimulationApp
            app = SimulationApp({"headless": True, "width": 1280, "height": 720,
                                 "renderer": "RaytracedLighting", "multi_gpu": False, "sync_loads": False})
            render_packet(app, packet, reports, output)
        fresh, _ = prepare_drafts(args.variants, load_rule(args.cut_rule) if args.cut_rule else None)
        if (fresh["source_variants_sha256"] != packet["source_variants_sha256"]
                or [r["variant_sha256"] for r in fresh["labels"]] != [r["variant_sha256"] for r in packet["labels"]]
                or fresh.get("cut_rule_sha256") != packet.get("cut_rule_sha256")):
            raise RuntimeError("Sources changed during packet generation")
        packet["state"] = "draft_ready_for_human_review" if not args.no_render else "metadata_only_draft"
        packet["source_assets_unchanged"] = True
        review_markdown = markdown_packet(packet)
        (output / "draft_labels.json").write_text(json.dumps(packet, indent=2, allow_nan=False), encoding="utf-8")
        (output / "review.md").write_text(review_markdown, encoding="utf-8")
        print("DRAFT_PACKET_READY " + str(output), flush=True)
    except Exception:
        exit_code = 1
        detail = traceback.format_exc()
        (output / "error.json").write_text(json.dumps({"state": "failed", "traceback": detail}), encoding="utf-8")
        print(detail, flush=True)
    finally:
        if app is not None:
            app.close(exit_code=exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
