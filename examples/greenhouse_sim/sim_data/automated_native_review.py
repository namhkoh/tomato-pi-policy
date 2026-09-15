"""Ground-truth-assisted static annotation review; never grants release approval.

Read completed native pairs only. Write new review receipts outside captures.
No renderer, model inference, legacy decision edits, or hidden-coordinate actions.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .capture_contract import project, transform_points, depth_evidence, fingerprint
from .cut_regions import _oriented_chain, _sample
from .dataset_review import read_json, require, verify_bindings, safe_file
from .depth_preview import sha256
from .native_clear_contract import TASK, crop_box, model_answer, user_prompt, SYSTEM
from .native_clear_labels import derive
from .native_query_visibility import NativeQueryVisibility

POLICY = dict(version="greenhouse.automatic_native_annotation_review.v1",
    minimum_trace_arc_m=.008, maximum_step_m=.0005, maximum_projected_step_px=.5,
    maximum_probes=12000, minimum_trace_interior_px=2.,
    maximum_trace_dark_fraction=.1, minimum_trace_median_luma=40.,
    local_photometric_probe_step_m=.005, dark_luminance_8bit=25.,
    scope="static_native_annotation_only_not_training_release_or_physics")


def check_native_evidence(metadata, rgb, depth, components, target_mask, catalogue):
    sync = metadata["synchronization"]
    require(sync["method"] == "frozen_scene_single_native_writer_payload"
            and sync["scene_unchanged_during_capture"] is True
            and sync["dynamic_recording_supported"] is False, "Static native writer evidence required")
    token = sync["freshness"]
    require(token["camera_sha256"] == fingerprint(metadata["calibration"])
            and token["rgb_sha256"] == hashlib.sha256(rgb.tobytes()).hexdigest()
            and token["depth_sha256"] == hashlib.sha256(depth.tobytes()).hexdigest(),
            "Native callback fingerprints disagree with saved raw buffers")
    variant, key = metadata["supervision"]["target_id"].split("/")
    targets = [c for c in catalogue if c["variant_id"] == variant and c["component_id"] == key]
    require(len(targets) == 1 and target_mask.shape == components.shape
            and target_mask.dtype == np.uint8 and set(np.unique(target_mask)) <= {0, 255}
            and np.array_equal(target_mask == 255, components == targets[0]["component_index"]),
            "Saved target mask disagrees with native component identity")


def trace_review(metadata, report, annotation, rgb, depth, valid, components, catalogue):
    """Trace every projected chain segment, not just mask connectivity/endpoints.

    Thin foreground crossings cannot be bridged by a connected silhouette around
    an occluder. The radius+3mm Z tolerance is the existing centerline-versus-
    surface approximation, not a new synthetic depth calculation.
    """
    sup, cal = metadata["supervision"], metadata["calibration"]
    variant, key = sup["target_id"].split("/")
    target = next(c for c in catalogue if c["variant_id"] == variant and c["component_id"] == key)
    component = report["components"][key]
    chain, lengths, _, attachment_error = _oriented_chain(component, 1e-6)
    require(attachment_error <= 1e-6 and len(lengths) > 1, "Invalid oriented anatomy")
    end = float(annotation["query_evidence"]["arc_m"])
    start = POLICY["minimum_trace_arc_m"]
    require(np.isfinite(end) and .045 <= end <= lengths[-1], "Invalid distal query arc")
    matrix, origin = sup["plant_to_world_usd_row_vectors"], component["translation_plant_m"]

    def at(arc):
        geometry = _sample(chain, lengths, float(arc), origin)
        point = transform_points([geometry["point_plant_m"]], matrix)[0]
        return geometry, project([point], cal)[0]

    # Include every capsule-chain vertex: do not interpolate across bends.
    breaks = sorted(set([start, end] + [float(d) for d in lengths if start < d < end]))
    arcs = []
    for left, right in zip(breaks[:-1], breaks[1:]):
        a, b = at(left)[1], at(right)[1]
        if any(p["projection_status"] != "in_frame" for p in (a, b)):
            return dict(passed=False, reasons=["query_to_cut_chain_out_of_frame"])
        screen_length = np.linalg.norm(np.asarray(a["pixel_xy"])-b["pixel_xy"])
        count = int(np.ceil(max((right-left)/POLICY["maximum_step_m"],
                                screen_length/POLICY["maximum_projected_step_px"])))
        # Perspective can make arc-uniform pixels nonuniform. Refine below.
        require(len(arcs)+count+1 <= POLICY["maximum_probes"], "Trace exceeds bounded probe budget")
        arcs.extend(np.linspace(left, right, max(1, count)+1)[:-1].tolist())
    arcs.append(end)
    for _ in range(12):
        projected = [at(d)[1] for d in arcs]
        if any(p["projection_status"] != "in_frame" for p in projected):
            return dict(passed=False, reasons=["query_to_cut_chain_out_of_frame"])
        uv = np.asarray([p["pixel_xy"] for p in projected])
        gaps = np.linalg.norm(np.diff(uv, axis=0), axis=1)
        bad = np.flatnonzero(gaps > POLICY["maximum_projected_step_px"]*(1+1e-9))
        if not len(bad):
            break
        require(len(arcs)+len(bad) <= POLICY["maximum_probes"], "Perspective trace probe budget exceeded")
        arcs = sorted(arcs + [(arcs[i]+arcs[i+1])/2 for i in bad])
    else:
        raise ValueError("Perspective trace did not converge")
    mask = components == target["component_index"]
    usability = NativeQueryVisibility(rgb, mask, cal["resolution"])
    misses, pixels = [], set()
    for d, p in zip(arcs, projected, strict=True):
        x, y = np.floor(p["pixel_xy"]).astype(int)
        pixels.add((int(x), int(y)))
        evidence = depth_evidence(p, depth, valid, at(d)[0]["petiole_radius_m"])
        identity = bool(mask[y, x])
        if not identity or evidence["status"] != "depth_consistent_not_visibility_verified":
            misses.append(dict(arc_m=d, pixel_xy=[int(x), int(y)], exact_target=identity,
                               native_depth_status=evidence["status"]))
    xx, yy = np.array(sorted(pixels)).T
    luma = usability.luminance[yy, xx]
    interior = float(usability.interior[yy, xx].min())
    dark = float(np.mean(luma < POLICY["dark_luminance_8bit"]))
    median = float(np.median(luma))
    photo = []
    count = int(np.ceil((end-start)/POLICY["local_photometric_probe_step_m"]))+1
    for d in np.linspace(start, end, count):
        p = at(d)[1]
        q = [float(v) for v in p["pixel_xy"]]
        inspected = usability.inspect(q)
        if not inspected["passed"]:
            photo.append(dict(arc_m=float(d), pixel_uv=q, reasons=inspected["reasons"]))
    reasons = []
    if misses: reasons.append("query_to_cut_chain_identity_or_depth_gap")
    if interior < POLICY["minimum_trace_interior_px"]: reasons.append("query_to_cut_chain_too_thin_or_edge_on")
    if dark > POLICY["maximum_trace_dark_fraction"] or median < POLICY["minimum_trace_median_luma"]:
        reasons.append("query_to_cut_chain_too_dark")
    if photo: reasons.append("query_to_cut_chain_local_usability")
    return dict(passed=not reasons, reasons=reasons, arc_interval_m=[start, end],
        probe_count=len(arcs), unique_pixels=len(pixels), maximum_projected_step_px=float(gaps.max()),
        minimum_interior_radius_px=interior, dark_fraction=dark, median_luminance=median,
        identity_or_depth_gap_count=len(misses), first_gap_probes=misses[:20],
        local_usability_failures=photo, native_depth_reconstructed=False)


def check_model_input(folder, annotation, rgb):
    """Check clean crop and exact prompt/answer parity; never run/download a model."""
    inputs = read_json(folder/"model_inputs.json")
    require(inputs["task"] == TASK and inputs["inspection_only"] is True
            and inputs["hidden_geometry_or_depth_in_model_inputs"] is False, "Unsafe model input scope")
    expected_variants = {"original_rgb", "original_rgb_plus_native_crop"}
    require(set(inputs["variants"]) == expected_variants, "Unexpected model input variants")
    q = annotation["query_pixel_uv"]
    with Image.fromarray(rgb) as full, Image.open(folder/"inputs/query_crop.png") as saved:
        with full.crop(crop_box(q)) as expected:
            require(saved.mode == "RGB" and np.array_equal(np.asarray(saved), np.asarray(expected)),
                    "Query crop differs from unchanged native RGB")
    for crop, key in ((False, "original_rgb"), (True, "original_rgb_plus_native_crop")):
        item = inputs["variants"][key]
        require(item["images"] == ["inputs/rgb.png"]+(["inputs/query_crop.png"] if crop else []),
                "Unexpected observation input")
        expected = [dict(role="system", content=SYSTEM),
            dict(role="user", content="<image>\n"*(2 if crop else 1)+user_prompt(q, crop=crop)),
            dict(role="assistant", content=json.dumps(model_answer(annotation["answer"]), separators=(",", ":")))]
        require(item["messages"] == expected, "Prompt or label differs from native task contract")


def annotation_code_proof(prior, requalify=False, code_root=None):
    """Explicitly recheck old outputs with current code, never rewrite old pins.

    Ordinary review remains strict. Requalification only permits an old code
    fingerprint: every saved native buffer, label and model input is still
    checked and independently recomputed by review_pair below.
    """
    root=Path(code_root or Path(__file__).parent).resolve()
    require(isinstance(prior,dict) and prior,"Missing prior annotation code evidence")
    for name,sha in prior.items():
        p=Path(name).resolve()
        require(p.parent==root and p.suffix=='.py'
                and (p.name.startswith('native_clear_') or p.name=='native_query_visibility.py')
                and isinstance(sha,str) and len(sha)==64,"Unexpected annotation code binding")
    require(str(root/'native_clear_labels.py') in prior,"Missing prior label-deriver fingerprint")
    if not requalify:verify_bindings(prior)
    files={*root.glob('native_clear_*.py'),root/'native_query_visibility.py'}
    current={str(p.resolve()):sha256(p) for p in files}
    return dict(explicit_requalification=bool(requalify),prior_implementation_sha256=dict(prior),
        current_implementation_sha256=current,
        changed_prior_files=[p for p,h in prior.items() if current.get(p)!=h],
        old_implementation_executed=False,old_bindings_or_decisions_rewritten=False)


def review_pair(annotation_directory, *, requalify=False):
    """Return append-only automatic evidence; source artifacts remain immutable."""
    from .audit import audit_manifest
    from .generated_capture import check_plan, verify_sensor_prerequisite, assert_pair_fresh
    from .plant_variant_catalogue import load_for_inspection
    directory = Path(annotation_directory).resolve()
    request = read_json(directory/"request.json")
    result = read_json(directory/"result.json")
    require(result["state"] == "native_clear_annotation_pilot_pending_visual_review"
            and result["training_approved"] is False, "Expected completed annotation pilot")
    verify_bindings(result["source_bindings"])
    code_proof=annotation_code_proof(result["implementation_sha256"],requalify)
    plan_path = Path(request["plan"]).resolve()
    require(result["source_bindings"].get(str(plan_path)) == sha256(plan_path), "Unbound source plan")
    plan = read_json(plan_path)
    check_plan(plan)
    verify_sensor_prerequisite(plan)
    capture = Path(request["capture"]).resolve()
    pair = read_json(capture/"result.json")
    pair_request = read_json(capture/"request.json")
    require(result["source_bindings"].get(str(capture/"result.json")) == sha256(capture/"result.json")
            and result["source_bindings"].get(str(capture/"request.json")) == sha256(capture/"request.json")
            and pair_request["plan_sha256"] == sha256(plan_path)
            and Path(pair_request["plan_path"]).resolve() == plan_path, "Unbound native pair identity")
    require(not (capture/"failure.json").exists() and pair["source_assets_unchanged"] is True
            and pair["state"] == "generated_native_pair_captured_pending_visual_review",
            "Incomplete native capture")
    assert_pair_fresh(*pair["samples"])
    generated = load_for_inspection(plan["variant_directory"], plan["source_collection_plan"])
    source_plan = read_json(plan["source_collection_plan"])
    job = next(j for j in source_plan["jobs"] if j["plant_family"] == plan["source_family"])
    reports = {"original_control": audit_manifest(job["source_manifest_path"]),
               "generated_variant": generated["report"]}
    require([r["name"] for r in result["records"]] == list(reports), "Incomplete annotation pair")
    bindings = {str(directory/p): sha256(directory/p) for p in ("request.json", "result.json")}
    records = []
    for row in result["records"]:
        name = row["name"]
        folder, source = directory/name, capture/name
        metadata, annotation = read_json(source/"sample.json"), read_json(folder/"label.json")
        require(sha256(folder/"label.json") == row["label_sha256"]
                and sha256(source/"sample.json") == row["source_sample_sha256"]
                and metadata in pair["samples"], "Annotation or sample identity changed")
        bindings[str(folder/"label.json")] = row["label_sha256"]
        for relative, info in metadata["files"].items():
            require(sha256(safe_file(source, relative)) == info["sha256"], "Changed native buffer")
        with Image.open(source/"inputs/rgb.png") as im: rgb = np.asarray(im).copy()
        with Image.open(source/"inputs/depth_valid.png") as im: raw_valid = np.asarray(im).copy()
        require(set(np.unique(raw_valid)) <= {0, 255}, "Invalid native validity mask")
        valid = raw_valid == 255
        depth = np.load(source/"inputs/depth_m.npy", allow_pickle=False)
        components = np.load(source/"supervision/component_id.npy", allow_pickle=False)
        catalogue = read_json(source/"supervision/identities.json")["component_catalogue"]
        with Image.open(source/"supervision/target_visible.png") as im: target_mask = np.asarray(im).copy()
        check_native_evidence(metadata, rgb, depth, components, target_mask, catalogue)
        derived = derive(metadata, reports[name], rgb, depth, valid, components, catalogue)
        require(annotation == derived and row["eligible"] is derived["eligible"], "Stale recomputed annotation")
        trace = None
        if not annotation["eligible"]:
            decision, reasons = "reject_clear_task", [annotation["reason"]]
        else:
            for relative in ("inputs/rgb.png", "inputs/depth_m.npy", "inputs/depth_valid.png", "supervision/target_visible.png"):
                require(sha256(safe_file(folder, relative)) == metadata["files"][relative]["sha256"],
                        "Annotation copy differs from native source")
                bindings[str(folder/relative)] = sha256(folder/relative)
            check_model_input(folder, annotation, rgb)
            for relative in ("model_inputs.json", "inputs/query_crop.png"):
                bindings[str(folder/relative)] = sha256(folder/relative)
            trace = trace_review(metadata, reports[name], annotation, rgb, depth, valid, components, catalogue)
            decision = "accept_automatic_annotation_only" if trace["passed"] else "hold_for_visual_review"
            reasons = trace["reasons"]
        records.append(dict(name=name, target_id=annotation["target_id"],
            source_plant_family=annotation["source_plant_family"], split=plan["split"],
            conservative_view_cap_group=annotation["conservative_view_cap_group"],
            rgb_path=str(source/"inputs/rgb.png"), rgb_sha256=metadata["files"]["inputs/rgb.png"]["sha256"],
            label_sha256=row["label_sha256"], decision=decision, reasons=reasons, trace=trace,
            visual_review_performed=False, training_approved=False, human_decision_requested=False))
    verify_bindings(bindings)
    verify_bindings(result["source_bindings"])
    verify_bindings(code_proof["current_implementation_sha256"])
    return dict(policy=POLICY, annotation_directory=str(directory), records=records,
        annotation_code_proof=code_proof,all_stored_labels_equal_current_derivation=True,
        bindings=bindings, basis="simulator_ground_truth_assisted_not_independent_botanical_validation",
        dynamic_or_physical_success_verified=False, training_approved=False,
        native_depth_reconstructed=False, hidden_cut_coordinates_executable=False,
        existing_reviews_or_splits_modified=False)


def run(directories, output, *, requalify=False):
    output = Path(output).resolve()
    dirs = [Path(p).resolve() for p in directories]
    require(dirs and len(set(dirs)) == len(dirs), "Distinct explicit annotation directories required")
    require(not output.exists() and all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in dirs),
            "New review receipt outside source annotations required")
    for directory in dirs:
        request_path = directory/"request.json"
        if request_path.is_file():
            request = read_json(request_path)
            roots = [Path(request[k]).resolve() for k in ("capture",) if k in request]
            if "plan" in request:
                plan_path = Path(request["plan"]).resolve()
                roots.append(plan_path.parent)
                if plan_path.is_file():
                    plan = read_json(plan_path)
                    roots.extend(Path(plan[k]).resolve() for k in
                        ("variant_directory", "source_capture", "prerequisite_directory") if k in plan)
            require(all(not output.is_relative_to(p) for p in roots),
                    "Review output must not be written into immutable captures, plans or assets")
    reviews = []
    for directory in dirs:
        try:
            reviews.append(review_pair(directory,requalify=requalify))
        except (ValueError, OSError, KeyError, TypeError, StopIteration, IndexError) as exc:
            reviews.append(dict(annotation_directory=str(directory), decision="hold_integrity_or_missing_evidence",
                error=f"{type(exc).__name__}: {exc}", records=[], training_approved=False,
                visual_review_performed=False, human_decision_requested=False))
    counts = {}
    for review in reviews:
        for row in review["records"]:
            counts[row["decision"]] = counts.get(row["decision"], 0)+1
    receipt = dict(schema=POLICY["version"], created_utc=datetime.now(timezone.utc).isoformat(),
        implementation_sha256={str(Path(__file__).resolve()): sha256(__file__)}, reviews=reviews,
        record_decisions=counts, integrity_held_pairs=sum(not r["records"] for r in reviews),
        unique_rgb_count=len({r["rgb_sha256"] for v in reviews for r in v["records"]}),
        accepted_unique_rgb_count=len({r["rgb_sha256"] for v in reviews for r in v["records"]
                                      if r["decision"] == "accept_automatic_annotation_only"}),
        production_training_approved_count=0, native_depth_reconstructed=False,
        explicit_annotation_requalification=bool(requalify),
        self_supervised_model_training_performed=False, requires_human_decision=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive create: concurrent reviewers cannot overwrite an existing decision.
    with output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requalify-existing",action="store_true",
        help="Append current-code evidence for unchanged old labels; never alter old pins or decisions")
    args = parser.parse_args()
    result = run(args.annotations, args.output,requalify=args.requalify_existing)
    print(json.dumps({k: result[k] for k in ("record_decisions", "integrity_held_pairs",
        "unique_rgb_count", "production_training_approved_count")}))


if __name__ == "__main__":
    main()
