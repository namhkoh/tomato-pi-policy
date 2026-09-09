from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from sim_data.dataset_review import (HEAD_CAMERA, SCHEMA, safe_file, verify_bindings,
                                     verify_robot_camera, audit, read_json)
from sim_data.dataset_package import (REVIEW_SCHEMA, active_reviews, build, queue_for,
                                      record, validate_record, validate_splits)
from sim_data.depth_preview import sha256


@pytest.fixture
def camera():
    class Model:
        def all_link_transforms(self, joints):
            if abs(joints["head_0"]) > 90:
                raise ValueError("joint limit")
            return {"link_head_2": np.eye(4)}
    cal = {"camera_path": HEAD_CAMERA, "resolution": [848, 408], "crop_resize": None,
           "intrinsics": [[470, 0, 424], [0, 470, 204], [0, 0, 1]], "clipping_range_m": [.04, 10],
           "camera_to_world_usd_row_vectors": np.eye(4).tolist()}
    projection = np.zeros((4, 4))
    projection[0, 0], projection[1, 1], projection[2, 3] = 470/424, 470/204, -1
    metadata = {"calibration": deepcopy(cal), "robot_snapshot": {
        "joint_degrees": {"head_0": 0, "head_1": 0}, "camera_to_head_column_vectors": np.eye(4).tolist(),
        "robot_root_to_world_usd_row_vectors": np.eye(4).tolist()},
        "rendered_camera_params": {"cameraViewTransform": np.eye(4).tolist(),
                                   "renderProductResolution": [848, 408], "metersPerSceneUnit": 1,
                                   "cameraProjection": projection.tolist()}}
    return metadata, Model(), np.eye(4), cal


def test_robot_pov_checks_pass_for_exact_fk(camera):
    result = verify_robot_camera(*camera)
    assert result["mounted_robot_pov_verified"] and result["fk_position_error_m"] == 0
    assert result["view_source"] == "simulated_robot_pose_not_live_lab_robot"
    assert not result["navigation_and_arm_reachability_validated"]


@pytest.mark.parametrize("kind", ["cinematic", "crop", "resolution", "intrinsics", "mount", "root", "pose", "renderer", "joint", "scale", "projection", "render_resolution"])
def test_camera_cannot_be_detached_reaimed_or_relabelled_as_robot(camera, kind):
    meta, model, mount, cal = camera
    if kind == "cinematic": meta["calibration"]["camera_path"] = "/OmniverseKit_Persp"
    elif kind == "crop": meta["calibration"]["crop_resize"] = [400, 200]
    elif kind == "resolution": meta["calibration"]["resolution"] = [1280, 720]
    elif kind == "intrinsics": meta["calibration"]["intrinsics"][0][0] += 1
    elif kind == "mount": meta["robot_snapshot"]["camera_to_head_column_vectors"][0][3] += .01
    elif kind == "root": meta["robot_snapshot"]["robot_root_to_world_usd_row_vectors"][3][0] += .01
    elif kind == "pose": meta["calibration"]["camera_to_world_usd_row_vectors"][3][0] += .01
    elif kind == "renderer": meta["rendered_camera_params"]["cameraViewTransform"][3][0] += .01
    elif kind == "scale": meta["robot_snapshot"]["robot_root_to_world_usd_row_vectors"][0][0] = 2
    elif kind == "projection": meta["rendered_camera_params"]["cameraProjection"][0][0] *= 2
    elif kind == "render_resolution": meta["rendered_camera_params"]["renderProductResolution"] = [1280, 720]
    else: meta["robot_snapshot"]["joint_degrees"]["head_0"] = 91
    with pytest.raises(ValueError): verify_robot_camera(meta, model, mount, cal)


@pytest.mark.parametrize("name", ["../leak", "/absolute", "C:/escape", "a\\b", "a//b", "a/./b", "a/../b"])
def test_artifact_paths_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError): safe_file(tmp_path, name)


def test_hash_binding_rejects_modified_or_missing_source(tmp_path):
    source = tmp_path / "rgb.png"
    source.write_bytes(b"original")
    bindings = {str(source): sha256(source)}
    verify_bindings(bindings)
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Stale"): verify_bindings(bindings)


@pytest.fixture
def package_source(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    review_root = tmp_path / "audit"
    review_root.mkdir()
    source = run / "manifest.json"
    source.write_text("{}")
    samples = []
    bindings = {str(source): sha256(source)}
    cards = {}
    for number, family in [(4, "seed103_full"), (5, "seed103_full"), (1, "seed101_full")]:
        sid = f"sample_{number:04}"
        directory = run / sid
        (directory / "inputs").mkdir(parents=True)
        files = {}
        for name in ("rgb.png", "depth_m.npy", "depth_valid.png"):
            path = directory / "inputs" / name
            path.write_bytes(b"fixture-only")
            files["inputs/"+name] = {"sha256": sha256(path), "role": "observation"}
            bindings[str(path)] = sha256(path)
        metadata = {"files": files, "calibration": {}, "robot_snapshot": {}, "supervision": {"human_cut_approval": False}}
        metadata_path = directory / "sample.json"
        metadata_path.write_text(json.dumps(metadata))
        bindings[str(metadata_path)] = sha256(metadata_path)
        card = review_root / (sid + ".png")
        card.write_bytes(b"review-only")
        cards[card.name] = sha256(card)
        samples.append({"sample_id": sid, "target_review_id": "B05" if number in [4, 5] else "B03",
                        "source_plant_family": family, "camera": {"mounted_robot_pov_verified": True},
                        "quality": {"clear_view_gate_passed": number != 1,
                                    "estimated_petiole_diameter_px": 3.6, "projected_interval_length_px": 8.9}})
    audit_path = review_root / "audit.json"
    audit_path.write_text(json.dumps({"schema_version": SCHEMA, "state": "complete_engineering_audit_not_approval",
        "training_dataset_approved": False, "source_run": str(run), "samples": samples,
        "bindings_sha256": bindings, "cards_sha256": cards, "limitations": ["prototype only"]}))
    records = tmp_path / "records"
    records.mkdir()
    return audit_path, records, tmp_path / "package"


def make_record(source, sample="sample_0004", role="assistant", decision="recommend", **kwargs):
    audit_path, records, _ = source
    return record(audit_path, sample, role, "Test reviewer", decision, "Inspected fixture", records, inspected=True, **kwargs)


def test_assistant_and_human_prototype_review_remain_distinct(package_source):
    assistant = read_json(make_record(package_source))
    human = read_json(make_record(package_source, role="human", decision="confirm"))
    assert not assistant["human_prototype_label_confirmation"]
    assert human["human_prototype_label_confirmation"]
    assert all(not r["training_eligible"] and not r["physical_cut_approved"] for r in [assistant, human])
    assert all(r["horticultural_validation"] == "pending" for r in [assistant, human])


@pytest.mark.parametrize("role,decision", [("assistant", "confirm"), ("human", "recommend"), ("robot", "confirm")])
def test_role_does_not_grant_unauthorized_approval(package_source, role, decision):
    with pytest.raises(ValueError): make_record(package_source, role=role, decision=decision)
    assert not list(package_source[1].iterdir())


def test_cannot_promote_failed_quality_gate_or_skip_visual_check(package_source):
    with pytest.raises(ValueError, match="failed clear-view"): make_record(package_source, sample="sample_0001")
    audit_path, output, _ = package_source
    with pytest.raises(ValueError, match="Inspect"):
        record(audit_path, "sample_0004", "assistant", "me", "recommend", "notes", output)


def test_edited_card_or_source_invalidates_reviews(package_source):
    audit_path, _, _ = package_source
    (audit_path.parent / "sample_0004.png").write_bytes(b"new picture")
    with pytest.raises(ValueError, match="card changed"): make_record(package_source)


def test_review_history_requires_explicit_non_cyclic_supersession(package_source):
    first = make_record(package_source)
    second = make_record(package_source, decision="hold", supersedes=first)
    audit_path, _, _ = package_source
    audit_data = read_json(audit_path)
    rows = [read_json(first), read_json(second)]
    active = active_reviews(rows, audit_data, sha256(audit_path))
    assert active[("sample_0004", "assistant")]["decision"] == "hold"
    with pytest.raises(ValueError, match="Missing superseded"):
        active_reviews(rows[1:], audit_data, sha256(audit_path))
    rows[0]["supersedes_review_id"] = rows[1]["review_id"]
    with pytest.raises(ValueError, match="Cyclic"):
        active_reviews(rows, audit_data, sha256(audit_path))


def test_conflicting_reviews_are_not_silently_overwritten(package_source):
    make_record(package_source)
    make_record(package_source, decision="hold")
    audit_path, records, output = package_source
    with pytest.raises(ValueError, match="Conflicting"): build(audit_path, records, output, "v1")
    assert not output.exists()


def test_stale_or_forged_review_cannot_enable_training(package_source):
    row = read_json(make_record(package_source))
    audit_path, _, _ = package_source
    data = read_json(audit_path)
    original = deepcopy(row)
    for key, value in [("training_eligible", True), ("human_prototype_label_confirmation", True),
                       ("audit_sha256", "stale"), ("physical_cut_approved", True), ("reviewer", " ")]:
        row = {**original, key: value}
        with pytest.raises(ValueError): validate_record(row, data, sha256(audit_path))


def test_versioned_index_keeps_inputs_separate_and_leaves_splits_unassigned(package_source):
    make_record(package_source)
    audit_path, records, output = package_source
    source_hash = sha256(audit_path)
    result = build(audit_path, records, output, "prototype.v1")
    rows = [json.loads(line) for line in (output / "samples.jsonl").read_text().splitlines()]
    assert result["training_eligible_count"] == 0 and result["sample_count"] == 3
    assert result["packaging"] == "relative_reference_index_not_self_contained"
    assert set(result["source_family_splits"].values()) == {"unassigned"}
    assert result["queues"] == {"recommended_for_human_review": 1, "diagnostic_hold": 1, "visual_review_pending": 1}
    assert all(set(r["observation"]) == {"inputs/rgb.png", "inputs/depth_m.npy", "inputs/depth_valid.png"} for r in rows)
    assert all(not r["training_eligible"] and r["difficulty"] == "unassigned" for r in rows)
    assert sha256(audit_path) == source_hash
    assert "REVIEW CARDS" in (output / "review.md").read_text()
    with pytest.raises(ValueError, match="NEW dataset"): build(audit_path, records, output, "prototype.v1")


def test_shared_source_family_cannot_leak_across_splits():
    with pytest.raises(ValueError, match="leaks"):
        validate_splits([{"source_plant_family": "seed1", "split": "train"},
                         {"source_plant_family": "seed1", "split": "test"}])


@pytest.mark.parametrize("sample_count,family_count", [(1, 1), (3, 2)])
def test_review_summary_counts_match_this_export(package_source, sample_count, family_count):
    audit_path, records, output = package_source
    audit = read_json(audit_path)
    audit["samples"] = audit["samples"][:sample_count]
    audit_path.write_text(json.dumps(audit))
    result = build(audit_path, records, output, "count-regression.v1")
    summary = (output / "review.md").read_text()
    assert f"All {sample_count} full-resolution inputs" in summary
    assert f"Source plant families represented: {family_count}." in summary
    assert result["sample_count"] == sample_count
    assert len(result["source_family_splits"]) == family_count


def test_manual_hold_overrides_numerical_pass():
    sample = {"quality": {"clear_view_gate_passed": True}}
    assert queue_for(sample, {"decision": "hold"}, {"decision": "confirm"}) == "diagnostic_hold"


REAL_RUN = Path(__file__).resolve().parents[3] / "data/sim_data/rgbd_pilots/refined_20260908_135740"


@pytest.mark.skipif(not (REAL_RUN / "manifest.json").is_file(), reason="Optional real native-depth pilot")
def test_real_pilot_robot_pov_geometry_masks_hashes_and_approval(tmp_path):
    result = audit(REAL_RUN, tmp_path / "real_audit")
    assert len(result["samples"]) == 9
    assert [s["sample_id"] for s in result["samples"] if s["quality"]["clear_view_gate_passed"]] == ["sample_0004", "sample_0005", "sample_0007"]
    assert all(s["camera"]["fk_position_error_m"] < 1e-12 for s in result["samples"])
    assert all(not s["human_approval"] and not s["training_eligible"] for s in result["samples"])
    verify_bindings(result["bindings_sha256"])
