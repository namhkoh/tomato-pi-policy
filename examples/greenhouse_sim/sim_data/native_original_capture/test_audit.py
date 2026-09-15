"""Actual native-size CPU label/trace math on explicitly SYNTHETIC unit buffers."""
from copy import deepcopy
import numpy as np
import pytest

from ..native_clear_contract_test import fixture as native_fixture
from ..capture_contract import fingerprint
from . import audit
from .contracts import SAMPLE_SCHEMA, ADMISSION, digest


def arrays():
    meta, report, rgb, depth, valid, components, catalogue = native_fixture()
    components = components.astype(np.uint32)
    prior = dict(joint_degrees={"head_0": 0., "head_1": 0., "arm": 0.},
        robot_root_to_world_usd_row_vectors=np.eye(4).tolist(), camera_to_head_column_vectors=np.eye(4).tolist())
    plan = dict(target_id="fixture/Petiole", source_family="fixture",
        source_row={"cut_region_proposal": {"unit_test_only": True}},
        expected_robot_snapshot=prior, expected_calibration=deepcopy(meta["calibration"]),
        expected_plant_to_world=np.eye(4).tolist(), expected_scene_counts={"components": 2},
        pose_prior={"prior_target_id": "fixture/Petiole", "historical_review_inherited": False},
        pose_request={"mode": "exact_prior", "native_pixel_xy": None})
    meta.update(schema_version=SAMPLE_SCHEMA, state="fresh_original_native_pending_automatic_audit",
        training_sample_approved=False, historical_labels_inherited=False,
        robot_snapshot=deepcopy(prior), pose_prior=deepcopy(plan["pose_prior"]), pose_request=deepcopy(plan["pose_request"]),
        scene_counts={"components": 2}, geometry_screen={"passed": True}, renderer="RealTimePathTracing",
        native_instance_backend="legacy", lighting={"day": 172, "minutes": 780, "dome_intensity": 6000},
        input_policy=dict(clean_full_scene=True, isolation=False, diagnostic_overlays=False))
    meta["supervision"].update(source_target_id=plan["target_id"], conservative_view_cap_group=plan["target_id"],
                               cut_region_proposal=plan["source_row"]["cut_region_proposal"])
    meta["synchronization"].update(method="frozen_scene_single_native_writer_payload", render_budget_subframes=56)
    target = (components == 1).astype(np.uint8) * 255
    args = [plan, meta, report, rgb, depth, valid, components, target, catalogue]
    tokens(args)
    return args


def tokens(args):
    args[1]["synchronization"]["freshness"] = dict(callback_sequence=1,
        camera_sha256=fingerprint(args[1]["calibration"]),
        rgb_sha256=digest(args[3].tobytes()), depth_sha256=digest(args[4].tobytes()))


def test_fresh_native_auto_accept_is_not_release_or_historical_approval():
    args = arrays()
    before = [a.copy() for a in args[3:8]]
    label, trace, decision = audit.review_arrays(*args)
    assert label["eligible"] and trace["passed"]
    assert decision == "accept_strict_automatic_annotation_candidate"
    assert label["training_approved"] is False and trace["native_depth_reconstructed"] is False
    assert ADMISSION["training_approved"] is False and ADMISSION["global_caps_and_duplicate_checks_required"]
    for a, b in zip(before, args[3:8]):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("failure", ["cut_id", "cut_z", "dark", "trace_id", "trace_z", "trace_dark"])
def test_native_label_or_strict_trace_rejects_ambiguity(failure):
    args = arrays()
    if failure == "cut_id":
        args[6][408, 878] = 2
        args[7][408, 878] = 0
    elif failure == "cut_z":
        args[4][408, 878] = .5
    elif failure == "dark":
        args[3][402:414, 860:925] = 0
    elif failure == "trace_id":
        args[6][408, 950] = 2
        args[7][408, 950] = 0
    elif failure == "trace_z":
        args[4][408, 950] = .5
    else:
        args[3][402:414, 925:980] = 0
    tokens(args)  # Test real geometry/clarity gates, not merely old fingerprints.
    label, trace, decision = audit.review_arrays(*args)
    assert decision != "accept_strict_automatic_annotation_candidate"
    assert not label["eligible"] or not trace["passed"]


@pytest.mark.parametrize("failure", ["callback", "legacy_resolution", "cap", "old_labels", "budget", "lighting", "camera", "pose"])
def test_native_contract_integrity_fail_closed(failure):
    args = arrays()
    meta = args[1]
    if failure == "callback":
        meta["synchronization"]["freshness"]["depth_sha256"] = "0" * 64
    elif failure == "legacy_resolution":
        meta["calibration"]["resolution"] = [848, 408]
    elif failure == "cap":
        meta["supervision"]["conservative_view_cap_group"] = "new_group"
    elif failure == "old_labels":
        meta["historical_labels_inherited"] = True
    elif failure == "budget":
        meta["synchronization"]["render_budget_subframes"] = 8
    elif failure == "lighting":
        meta["lighting"]["dome_intensity"] = 1200
    elif failure == "camera":
        meta["calibration"]["camera_to_world_usd_row_vectors"][3][0] = .1
    else:
        meta["robot_snapshot"]["joint_degrees"]["arm"] = 3
    with pytest.raises(ValueError):
        audit.review_arrays(*args)


def test_reframe_may_change_only_head_not_root_mount_or_arm():
    args = arrays()
    plan, meta = args[:2]
    request = dict(mode="reframe_head_only", native_pixel_xy=meta["supervision"]["nominal_projected"]["pixel_xy"])
    plan["pose_request"] = meta["pose_request"] = request
    meta["robot_snapshot"]["joint_degrees"]["head_0"] = 1.
    audit.verify_pose_request(plan, meta)
    meta["robot_snapshot"]["joint_degrees"]["arm"] = 1.
    with pytest.raises(ValueError, match="arm/torso"):
        audit.verify_pose_request(plan, meta)


def test_smoke_missing_or_extra_bindings_fail_without_loading_images(tmp_path):
    with pytest.raises(ValueError, match="smoke bindings"):
        audit.verify_smoke(tmp_path, {})


def test_failed_capture_cannot_be_replayed(tmp_path):
    (tmp_path / "failure.json").write_text("{}")
    with pytest.raises(ValueError, match="Failed capture"):
        audit.audit_capture(tmp_path, result_sha256="0" * 64)


@pytest.fixture
def saved_unit_capture(tmp_path, monkeypatch):
    """CPU serialization fixture. Only robot FK is replaced; native math is real."""
    from ..capture_contract import write_sample, project
    from ..capture_visibility import write_visibility, component_masks
    from .contracts import sha256, write_new
    args = arrays()
    plan, meta, report, rgb, depth, valid = args[:6]
    report["components"]["Main"]["parent"] = None
    root = "/World/Unit"
    catalogue = [
        dict(component_id="Main", prim_path=root+"/Main", organ_type="main_stem",
             variant_id="fixture", source_plant_id="fixture", split_group="fixture", component_index=1),
        dict(component_id="Petiole", prim_path=root+"/Main/Petiole", organ_type="sub_stem",
             variant_id="fixture", source_plant_id="fixture", split_group="fixture", component_index=2)]
    ids = np.zeros((816, 1696), np.uint32)
    ids[args[6] == 1] = 100
    ids[args[6] == 2] = 200
    mapping = {0: "/World/Background", 100: root+"/Main/Petiole/mesh", 200: root+"/Main/mesh"}
    components, organs, _ = component_masks(ids, mapping, catalogue)
    mask = components == 2
    plan["scene_variants"] = [dict(variant_id="fixture", source_plant_id="fixture", plant_root=root)]
    meta.update(native_instance_sha256=digest(ids.tobytes()), native_mapping_sha256=fingerprint(mapping))
    meta["sample_id"] = "sample_0001"
    meta["supervision"]["projected_interval"] = project(meta["supervision"]["interval_world_m"], meta["calibration"])
    meta["supervision"]["depth_evidence"] = {"status": "UNIT TEST ONLY"}
    fresh = [plan, meta, report, rgb, depth, valid, components, mask.astype(np.uint8)*255, catalogue]
    tokens(fresh)
    label, trace, _ = audit.review_arrays(*fresh)
    assert label["eligible"] and trace["passed"]
    folder = tmp_path / "sample_0001"
    write_sample(folder, rgb, depth, valid, meta)
    write_visibility(folder, ids, mapping, catalogue, components, organs, mask)
    write_new(folder / "supervision/label.json", label)
    write_new(folder / "supervision/query_trace.json", trace)
    record = dict(case_id="sample_0001", target_id=plan["target_id"], source_assets_unchanged=True,
        admission=deepcopy(ADMISSION), sample_sha256=sha256(folder / "sample.json"),
        label_sha256=sha256(folder / "supervision/label.json"),
        trace_sha256=sha256(folder / "supervision/query_trace.json"), scene_counts=meta["scene_counts"],
        geometry_screen=meta["geometry_screen"], lighting=meta["lighting"], label_reason=label["reason"])
    monkeypatch.setattr(audit, "verify_camera", lambda metadata: None)
    return tmp_path, plan, record, {"fixture": report}


def test_saved_native_masks_labels_and_trace_replay(saved_unit_capture):
    result = audit._audit_sample(*saved_unit_capture)
    assert result["decision"] == "accept_strict_automatic_annotation_candidate"
    assert result["native_ID_masks_replayed"] and result["trace_replayed_exact"]
    assert result["admission"]["training_approved"] is False


@pytest.mark.parametrize("name", ["label", "query_trace"])
def test_rehashed_label_or_trace_tampering_still_fails_replay(saved_unit_capture, name):
    from .contracts import read_json, sha256
    import json
    capture, plan, record, reports = saved_unit_capture
    path = capture / "sample_0001/supervision" / (name + ".json")
    value = read_json(path)
    value["tampered"] = True
    path.write_text(json.dumps(value))
    record["label_sha256" if name == "label" else "trace_sha256"] = sha256(path)
    with pytest.raises(ValueError, match="Stale"):
        audit._audit_sample(capture, plan, record, reports)


def test_unhashed_native_payload_change_rejected(saved_unit_capture):
    capture, plan, record, reports = saved_unit_capture
    path = capture / "sample_0001/inputs/depth_m.npy"
    depth = np.load(path, allow_pickle=False)
    depth[408, 878] = .5
    np.save(path, depth, allow_pickle=False)
    with pytest.raises(ValueError, match="Changed or missing pinned"):
        audit._audit_sample(capture, plan, record, reports)
