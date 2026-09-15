"""Adversarial CPU tests; fixtures are tests, never native dataset evidence."""
import hashlib
import json

import numpy as np
from PIL import Image
import pytest

from sim_data import automated_native_review as review
from sim_data.native_clear_contract import crop_box, model_answer, user_prompt, SYSTEM, TASK


def fixture():
    rgb = np.full((408, 848, 3), 200, np.uint8)
    components = np.zeros((408, 848), np.int32)
    components[192:209, 399:602] = 7
    rgb[components == 7] = [70, 110, 50]
    depth = np.full((408, 848), .996, np.float32)
    valid = np.ones((408, 848), bool)
    component = dict(type="sub_stem", deleafed=False, translation_plant_m=[0, 0, -1],
        attachment_plant_m=[0, 0, -1], axis_plant=[1, 0, 0],
        capsules_local_m=[[[0, 0, 0, .007], [.2, 0, 0, .007]]])
    metadata = dict(supervision=dict(target_id="plant/petiole", plant_to_world_usd_row_vectors=np.eye(4).tolist()),
        calibration=dict(resolution=[848, 408], camera_to_world_usd_row_vectors=np.eye(4).tolist(),
            intrinsics=[[1000, 0, 400], [0, 1000, 200], [0, 0, 1]], clipping_range_m=[.01, 10]))
    annotation = dict(query_evidence=dict(arc_m=.16))
    return [metadata, dict(components=dict(petiole=component)), annotation, rgb, depth, valid,
            components, [dict(variant_id="plant", component_id="petiole", component_index=7)]]


def test_visible_trace_passes_without_mutating_buffers():
    args = fixture()
    before = [a.copy() for a in args[3:7]]
    result = review.trace_review(*args)
    assert result["passed"] and result["probe_count"] > 300
    assert result["maximum_projected_step_px"] <= .500000001
    assert result["native_depth_reconstructed"] is False
    for a, b in zip(args[3:7], before, strict=True):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("failure", ["other_stem", "fruit", "invalid_depth", "depth_gap", "thin", "dark", "contrast"])
def test_trace_rejects_intervening_failure_even_with_valid_endpoints(failure):
    args = fixture()
    if failure in ("other_stem", "fruit"):
        # One-pixel axis occlusion; silhouette still connected around this hole.
        args[6][200, 500] = 8 if failure == "other_stem" else 9
    elif failure == "invalid_depth": args[5][200, 500] = False
    elif failure == "depth_gap": args[4][200, 500] = .5
    elif failure == "thin": args[6][198:200, 500] = 0
    elif failure == "dark": args[3][192:209, 470:530] = 0
    elif failure == "contrast": args[3][180:220, 460:540] = [70, 110, 50]
    assert not review.trace_review(*args)["passed"]


def test_chain_vertex_not_shortcut_by_endpoint_line():
    args = fixture()
    args[1]["components"]["petiole"]["capsules_local_m"] = [
        [[0, 0, 0, .007], [.08, .04, 0, .007], [.2, 0, 0, .007]]]
    result = review.trace_review(*args)
    assert not result["passed"] and result["identity_or_depth_gap_count"] > 0


@pytest.mark.parametrize("query", [float("nan"), .01, .3])
def test_invalid_query_arc_is_not_guessed(query):
    args = fixture()
    args[2]["query_evidence"]["arc_m"] = query
    with pytest.raises(ValueError): review.trace_review(*args)


def test_out_of_frame_chain_is_held():
    args = fixture()
    args[0]["calibration"]["intrinsics"][0][2] = 840
    assert review.trace_review(*args)["reasons"] == ["query_to_cut_chain_out_of_frame"]


def test_perspective_refinement_bounds_screen_gaps():
    args = fixture()
    args[1]["components"]["petiole"]["capsules_local_m"] = [
        [[0, 0, 0, .007], [.2, 0, .8, .007]]]
    args[2]["query_evidence"]["arc_m"] = .55
    result = review.trace_review(*args)
    assert result["maximum_projected_step_px"] <= .500000001


def model_fixture(tmp_path):
    rgb = np.full((816, 1696, 3), 110, np.uint8)
    rgb[:, 850:] = [30, 140, 40]
    label = dict(query_pixel_uv=[1000., 500.], answer=dict(status="localized", cut_point_uv=[900., 450.],
                                                       visibility="clear", next_action="inspect_cut_region"))
    (tmp_path/"inputs").mkdir()
    with Image.fromarray(rgb) as full:
        with full.crop(crop_box(label["query_pixel_uv"])) as crop:
            crop.save(tmp_path/"inputs/query_crop.png")
    variants = {}
    for crop, key in ((False, "original_rgb"), (True, "original_rgb_plus_native_crop")):
        variants[key] = dict(images=["inputs/rgb.png"]+(["inputs/query_crop.png"] if crop else []),
            messages=[dict(role="system", content=SYSTEM), dict(role="user", content="<image>\n"*(2 if crop else 1)
                +user_prompt(label["query_pixel_uv"], crop=crop)), dict(role="assistant", content=json.dumps(
                    model_answer(label["answer"]), separators=(",", ":")))])
    payload = dict(task=TASK, inspection_only=True, hidden_geometry_or_depth_in_model_inputs=False, variants=variants)
    (tmp_path/"model_inputs.json").write_text(json.dumps(payload))
    return rgb, label, payload


def test_clean_model_input_parity(tmp_path):
    rgb, label, _ = model_fixture(tmp_path)
    review.check_model_input(tmp_path, label, rgb)


@pytest.mark.parametrize("failure", ["crop", "prompt", "answer", "depth", "scope"])
def test_model_input_corruption_rejected(tmp_path, failure):
    rgb, label, payload = model_fixture(tmp_path)
    if failure == "crop":
        with Image.new("RGB", (768, 768)) as im: im.save(tmp_path/"inputs/query_crop.png")
    elif failure == "prompt": payload["variants"]["original_rgb"]["messages"][1]["content"] += " hidden point"
    elif failure == "answer": payload["variants"]["original_rgb"]["messages"][2]["content"] = "{}"
    elif failure == "depth": payload["variants"]["original_rgb"]["images"].append("inputs/depth_m.npy")
    elif failure == "scope": payload["hidden_geometry_or_depth_in_model_inputs"] = True
    (tmp_path/"model_inputs.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError): review.check_model_input(tmp_path, label, rgb)


def test_missing_evidence_held_not_approved_or_escalated(tmp_path):
    result = review.run([tmp_path/"missing_capture"], tmp_path/"new_review.json")
    assert result["integrity_held_pairs"] == 1
    assert result["production_training_approved_count"] == 0
    assert result["requires_human_decision"] is False
    assert result["reviews"][0]["decision"] == "hold_integrity_or_missing_evidence"


def test_cannot_overwrite_review_or_write_inside_source(tmp_path):
    output = tmp_path/"old_review.json"
    output.write_text("prior human decision")
    with pytest.raises(ValueError): review.run([tmp_path/"source"], output)
    assert output.read_text() == "prior human decision"
    with pytest.raises(ValueError): review.run([tmp_path/"source"], tmp_path/"source/new.json")


def test_duplicate_pair_not_counted_twice(tmp_path):
    with pytest.raises(ValueError): review.run([tmp_path/"source"]*2, tmp_path/"review.json")


def test_cannot_write_receipt_into_native_capture(tmp_path):
    annotation = tmp_path/"annotations"
    annotation.mkdir()
    (annotation/"request.json").write_text(json.dumps(dict(capture=str(tmp_path/"capture"))))
    with pytest.raises(ValueError): review.run([annotation], tmp_path/"capture/new_review.json")
    assert not (tmp_path/"capture").exists()


@pytest.mark.parametrize("failure", [None, "rgb", "depth", "camera", "mask", "dynamic"])
def test_raw_native_fingerprints_and_exact_mask(failure):
    metadata, _, _, rgb, depth, _, components, catalogue = fixture()
    metadata["synchronization"] = dict(method="frozen_scene_single_native_writer_payload",
        scene_unchanged_during_capture=True, dynamic_recording_supported=False,
        freshness=dict(camera_sha256=review.fingerprint(metadata["calibration"]),
            rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),
            depth_sha256=hashlib.sha256(depth.tobytes()).hexdigest()))
    mask = (components == 7).astype(np.uint8)*255
    if failure == "rgb": rgb[0, 0] = 0
    elif failure == "depth": depth[0, 0] = 0
    elif failure == "camera": metadata["calibration"]["intrinsics"][0][0] += 1
    elif failure == "mask": mask[0, 0] = 255
    elif failure == "dynamic": metadata["synchronization"]["dynamic_recording_supported"] = True
    if failure is None:
        review.check_native_evidence(metadata, rgb, depth, components, mask, catalogue)
    else:
        with pytest.raises(ValueError):
            review.check_native_evidence(metadata, rgb, depth, components, mask, catalogue)
