"""CPU-only provenance/selection tests; synthetic bytes are not capture evidence."""
from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from sim_data.native_capture_v3 import reference_bank as bank


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")
    return bank.sha256(path)


def matrix(x=0., angle=0.):
    c, s = math.cos(angle), math.sin(angle)
    return [[c, s, 0., 0.], [-s, c, 0., 0.], [0., 0., 1., 0.], [x, 0., 1., 1.]]


class Checkpoint:
    """Small real file/hash graph, with no mocked verification calls."""
    def __init__(self, root):
        self.root = root
        self.checkpoint = root / "checkpoint"
        self.draft = self.checkpoint / "draft"
        self.capture = root / "capture"
        self.sample_path = self.capture / "sample_0001/sample.json"
        self.asset = root / "original.usdc"
        self.asset.write_bytes(b"immutable synthetic asset fixture")
        self.asset_hash = bank.sha256(self.asset)
        self.target = dict(target_id="seed1/SubStem_1", component_id="SubStem_1",
            source_plant_id="seed1", split_group="seed1", variant_id="seed1", draft_id="N_seed1_1",
            cut_region_proposal={"nominal": {"point_plant_m": [0, 0, 1]}})
        self.plan = dict(jobs=[dict(job_id="job1", plant_family="seed1", split="train", targets=[self.target])],
            family_assignments={"seed1": "train"}, source_bindings_sha256={str(self.asset): self.asset_hash})
        self.sample = dict(sample_id="sample_0001", calibration=dict(camera_path=bank.HEAD_CAMERA,
            resolution=[848, 408], crop_resize=None, camera_to_world_usd_row_vectors=matrix(),
            intrinsics=[[400, 0, 424], [0, 400, 204], [0, 0, 1]], focal_length_mm=2.,
            apertures_mm=[4., 2.], aperture_offsets_mm=[0, 0], clipping_range_m=[.01, 10],
            depth_convention="optical_axis_z_metres_not_ray_range"),
            robot_snapshot=dict(robot_root_to_world_usd_row_vectors=matrix(),
                camera_to_head_column_vectors=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                joint_degrees={"head_0": 0., "head_1": 0., "torso_0": 0.}, joint_limits_checked=True,
                visual_bound_screen=dict(passed=True, method="fixture_recorded_screen",
                    possible_overlap_count=0, possible_overlaps=[], floor_checked_separately=True),
                floor_alignment=dict(wheel_supports=[dict(clearance_m=0), dict(clearance_m=0)],
                                     base_minimum_clearance_m=.001)),
            supervision={**deepcopy(self.target), "review_id": "N_seed1_1"},
            synchronization=dict(scene_unchanged_during_capture=True, dynamic_recording_supported=False))
        self.audited = dict(sample_id="sample_0001", target_review_id="N_seed1_1", source_plant_family="seed1",
            integrity_and_recomputed_annotations_passed=True,
            camera=dict(camera_path=bank.HEAD_CAMERA, mounted_robot_pov_verified=True,
                mount_matches_reconstructed_robot=True, urdf_joint_limits_checked=True,
                fk_position_error_m=0., fk_matrix_max_error=0., base_world_m=[0, 0, 1], camera_world_m=[0, 0, 1],
                head_joint_degrees={"head_0": 0., "head_1": 0.}),
            parent_proxy_diagnostic=dict(nominal_point_to_parent_surface_m=.004,
                full_interval_or_blade_clearance_validated=False))
        self.manifest = dict(state="pilot_ready_for_review", source_assets_unchanged=True,
            source_geometry_modified=False, target_family_split="train", collection_job_id="job1",
            variants=[dict(source_plant_id="seed1", variant_id="seed1", added_components=[])],
            samples=[dict(sample_id="sample_0001", target_review_id="N_seed1_1")],
            source_usd_sha256={str(self.asset): self.asset_hash}, scene_counts={"plants": 10},
            lighting={"unchanged": True}, renderer="RealTimePathTracing",
            unbundled_external_prop_roots_excluded=[])
        self.label = dict(eligible=True, target_id="seed1/SubStem_1", source_plant_family="seed1",
            answer={"status": "localized"}, query_pixel_uv=[400, 200], contract_sha256="a"*64,
            query_usability=dict(local_contrast_8bit=40., interior_radius_px=5.))
        self.row = dict(id="reference1", split="train", source_plant_family="seed1", target_id="seed1/SubStem_1",
            answer=deepcopy(self.label["answer"]), query_pixel_uv=[400, 200], task_contract_sha256="a"*64,
            files=dict(label="labels/ref.json"), legibility=dict(interval_length_px=20., width_proxy_px=8.))
        self.review = dict(id="reference1", decision="accept", reviewer="fixture", reviewer_type="assistant",
                           reason="Synthetic reviewed-reference test fixture only")
        self.negatives = []
        self.trace = None
        self.extra_rows = []

    def freeze(self):
        plan_path = self.root / "plan.json"
        plan_hash = write(plan_path, self.plan)
        self.manifest.update(source_collection_plan_path=str(plan_path), source_collection_plan_sha256=plan_hash)
        capture_hash = write(self.capture / "manifest.json", self.manifest)
        rgb = self.sample_path.parent / "inputs/rgb.png"
        rgb.parent.mkdir(parents=True, exist_ok=True)
        rgb.write_bytes(b"synthetic RGB fixture, never decoded or used for training")
        rgb_hash = bank.sha256(rgb)
        self.sample["files"] = {"inputs/rgb.png": {"sha256": rgb_hash}}
        sample_hash = write(self.sample_path, self.sample)
        pins = {str(plan_path): plan_hash, str(self.capture / "manifest.json"): capture_hash,
                str(self.sample_path): sample_hash, str(rgb): rgb_hash, str(self.asset): self.asset_hash}
        if self.trace is not None:
            path = self.sample_path.parent / "supervision/query_trace.json"
            pins[str(path)] = write(path, self.trace)
        self.audit_path = self.root / "audit/audit.json"
        audit_hash = write(self.audit_path, dict(state="complete_engineering_audit_not_approval",
            source_run=str(self.capture), bindings_sha256=pins, samples=[self.audited],
            implementation_sha256={"historical_not_current.py": "b"*64}))
        self.review["rgb_sha256"] = rgb_hash
        self.row.update(rgb_sha256=rgb_hash, source_sample_sha256=sample_hash, source_audit_sha256=audit_hash)
        source_hash = write(self.draft / "source_manifest.json",
            dict(source_bindings_sha256={**pins, str(self.audit_path): audit_hash},
                 source_plans_sha256={str(plan_path): plan_hash}))
        reviews_hash = write(self.draft / "reviews.json", [self.review])
        write(self.checkpoint / "accepts.json", [self.review])
        exclusions_hash = write(self.draft / "visual_exclusions.json", self.negatives)
        label_hash = write(self.draft / "labels/ref.json", self.label)
        rows = [self.row] + self.extra_rows
        index = self.draft / "index.jsonl"
        index.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        self.release = dict(schema_version="greenhouse.clear_cutpoint_release.v1",
            source_manifest_copy_sha256=source_hash, files_sha256={"source_manifest.json": source_hash,
                "reviews.json": reviews_hash, "visual_exclusions.json": exclusions_hash,
                "labels/ref.json": label_hash, "index.jsonl": bank.sha256(index)})
        self.pin = write(self.draft / "manifest.json", self.release)
        return self

    def build(self):
        return bank.build(self.checkpoint, manifest_sha256=self.pin)


@pytest.fixture
def source(tmp_path):
    return Checkpoint(tmp_path).freeze()


def test_original_hashes_reviews_and_bytes_preserved(source):
    before = {str(p): bank.sha256(p) for p in source.root.rglob("*") if p.is_file()}
    result = source.build()
    entry = result["entries"][0]
    assert entry["source_collection_plan_sha256"] == source.manifest["source_collection_plan_sha256"]
    assert entry["source_capture_manifest_sha256"] == bank.sha256(source.capture / "manifest.json")
    assert entry["source_sample_sha256"] == source.row["source_sample_sha256"]
    assert entry["review"] == source.review
    assert entry["robot_snapshot"] == source.sample["robot_snapshot"]
    assert result["counts"]["donors"] == result["counts"]["targets"] == 1
    assert result["counts"]["historical_848_references"] == 1
    assert result["counts"]["new_native_training_rows"] == 0
    assert not entry["source_cap_reset"] and not result["training_approved"]
    assert bank.check(result)
    assert before == {str(p): bank.sha256(p) for p in source.root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("name", ["asset", "plan", "capture", "sample", "rgb", "audit", "index", "review", "accepts", "label"])
def test_stale_files_fail_closed(source, name):
    paths = dict(asset=source.asset, plan=source.root / "plan.json", capture=source.capture / "manifest.json",
        sample=source.sample_path, rgb=source.sample_path.parent / "inputs/rgb.png", audit=source.audit_path,
        index=source.draft / "index.jsonl", review=source.draft / "reviews.json",
        accepts=source.checkpoint / "accepts.json", label=source.draft / "labels/ref.json")
    result = source.build()
    with paths[name].open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="Changed bound file"):
        bank.check(result)


@pytest.mark.parametrize("field", ["source_collection_plan_sha256", "source_sample_sha256", "review", "evidence",
                                    "calibration", "training_approved", "source_cap_reset"])
def test_serialized_bank_edits_rejected(source, field):
    result = source.build()
    result["entries"][0][field] = "altered"
    with pytest.raises(ValueError, match="bank metadata"):
        bank.check(result)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_heldout_excluded_before_dereferencing(source, split):
    source.extra_rows = [{**source.row, "id": "heldout", "split": split, "source_plant_family": "heldout_family",
                          "target_id": "heldout_family/SubStem_2", "source_sample_sha256": "c"*64}]
    result = source.freeze().build()
    assert result["counts"]["references"] == 1
    assert result["exclusions"]["heldout_split"] == 1


@pytest.mark.parametrize("fault", ["plan_split", "job_split", "capture_split", "family_conflict", "original_variant",
    "target_lineage", "review_hold", "review_missing_reason", "negative_history", "camera_path", "mount_unknown",
    "geometry_unknown", "geometry_failed", "overlaps", "floor", "joint_limits", "pose_audit_mismatch", "reflection",
    "fk_error", "anatomy", "target_review", "label", "duplicate_id"])
def test_rebound_but_semantically_invalid_evidence_rejected(source, fault):
    if fault == "plan_split": source.plan["family_assignments"]["seed1"] = "test"
    elif fault == "job_split": source.plan["jobs"][0]["split"] = "validation"
    elif fault == "capture_split": source.manifest["target_family_split"] = "test"
    elif fault == "family_conflict": source.extra_rows = [{**source.row, "id": "other", "split": "test"}]
    elif fault == "original_variant": source.manifest["variants"][0]["added_components"] = ["leaf"]
    elif fault == "target_lineage": source.sample["supervision"]["variant_id"] = "generated"
    elif fault == "review_hold": source.review["decision"] = "hold"
    elif fault == "review_missing_reason": source.review.pop("reason")
    elif fault == "negative_history": source.negatives = [{"id": "reference1", "decision": "hold"}]
    elif fault == "camera_path": source.sample["calibration"]["camera_path"] = "/World/Cinematic"
    elif fault == "mount_unknown": source.audited["camera"].pop("mount_matches_reconstructed_robot")
    elif fault == "geometry_unknown": source.sample["robot_snapshot"].pop("visual_bound_screen")
    elif fault == "geometry_failed": source.sample["robot_snapshot"]["visual_bound_screen"]["passed"] = False
    elif fault == "overlaps": source.sample["robot_snapshot"]["visual_bound_screen"]["possible_overlap_count"] = 1
    elif fault == "floor": source.sample["robot_snapshot"]["floor_alignment"]["wheel_supports"] = []
    elif fault == "joint_limits": source.sample["robot_snapshot"]["joint_limits_checked"] = False
    elif fault == "pose_audit_mismatch": source.sample["calibration"]["camera_to_world_usd_row_vectors"][3][0] = 1
    elif fault == "reflection": source.sample["calibration"]["camera_to_world_usd_row_vectors"][0][0] = -1
    elif fault == "fk_error": source.audited["camera"]["fk_matrix_max_error"] = .01
    elif fault == "anatomy": source.sample["supervision"]["cut_region_proposal"] = {}
    elif fault == "target_review": source.audited["target_review_id"] = "another"
    elif fault == "label": source.label["query_pixel_uv"] = [0, 0]
    elif fault == "duplicate_id": source.extra_rows = [deepcopy(source.row)]
    source.freeze()
    with pytest.raises(ValueError):
        source.build()


def test_missing_metrics_are_unknown_not_clear(source):
    source.row.pop("legibility")
    source.label.pop("query_usability")
    source.audited.pop("parent_proxy_diagnostic")
    entry = source.freeze().build()["entries"][0]
    assert all(m["status"] == "unknown" and m["value"] is None for m in entry["evidence"].values())
    assert "clear" not in entry
    assert not entry["training_approved"]


@pytest.mark.parametrize("value", [True, -1, "40"])
def test_invalid_metrics_not_promoted_to_evidence(source, value):
    source.label["query_usability"]["local_contrast_8bit"] = value
    with pytest.raises(ValueError, match="ranking evidence"):
        source.freeze().build()


def test_metadata_parser_rejects_duplicate_keys_and_nonfinite():
    for data in ('{"a":1,"a":2}', '{"a":NaN}', '{"a":Infinity}'):
        with pytest.raises(ValueError):
            bank._parse(data)


def test_binding_conflict_and_traversal(source):
    reader = bank._Reader()
    reader.bind(source.asset, source.asset_hash)
    with pytest.raises(ValueError, match="Conflicting"):
        reader.bind(source.asset, "d"*64)
    with pytest.raises(ValueError, match="escapes"):
        bank._safe(source.draft, "../outside.json")
    with pytest.raises(ValueError, match="original SHA256"):
        reader.bind(source.asset, None)


def test_rank_prefers_evidence_not_nearest_depth_or_rgb(source):
    first = source.build()["entries"][0]
    stronger = deepcopy(first)
    stronger["evidence"]["local_contrast_8bit"]["value"] = 100
    stronger["nominal_optical_z_m"] = 100
    stronger["rgb_std"] = 0
    assert bank.rank_key(stronger) < bank.rank_key(first)
    unknown = deepcopy(first)
    unknown["evidence"]["local_contrast_8bit"].update(status="unknown", value=None)
    assert bank.rank_key(first) < bank.rank_key(unknown)


def test_native_trace_requires_bound_complete_evidence(source):
    source.sample["calibration"]["resolution"] = [1696, 816]
    source.trace = dict(passed=True, reasons=[], native_depth_reconstructed=False, identity_or_depth_gap_count=0,
        local_usability_failures=[], probe_count=12, unique_pixels=12, minimum_interior_radius_px=3.)
    result = source.freeze().build()
    entry = result["entries"][0]
    assert entry["evidence"]["strict_native_trace"]["value"] is True
    assert entry["strict_trace_sha256"] == bank.sha256(entry["strict_trace_path"])
    assert result["counts"]["new_native_training_rows"] == 0
    assert result["counts"]["historical_848_references"] == 0
    source.trace.pop("probe_count")
    with pytest.raises(ValueError, match="Incomplete strict trace"):
        source.freeze().build()


def test_missing_native_trace_remains_unknown(source):
    source.sample["calibration"]["resolution"] = [1696, 816]
    entry = source.freeze().build()["entries"][0]
    assert entry["evidence"]["strict_native_trace"]["value"] is None
    assert entry["strict_trace_sha256"] is None


def selection_fixture(source):
    result = source.build()
    original = result["entries"][0]
    entries = []
    for i, x in enumerate((0., .04, .12, .2, .2)):
        entry = deepcopy(original)
        entry.update(id=str(i), source_sample="sample_" + str(i), rgb_sha256=str(i)*64)
        entry["calibration"]["camera_to_world_usd_row_vectors"] = matrix(x)
        entry["robot_snapshot"]["robot_root_to_world_usd_row_vectors"] = matrix(x)
        entries.append(entry)
    # Exact camera duplicate with changed joints/root and stronger RGB noise.
    entries[-1]["robot_snapshot"]["joint_degrees"]["torso_0"] = 80
    entries[-1]["robot_snapshot"]["robot_root_to_world_usd_row_vectors"] = matrix(2)
    entries[-1]["rgb_std"] = 10000
    entries[0]["evidence"]["local_contrast_8bit"]["value"] = 100
    result["entries"] = entries
    return result, original["compatibility_group"], original["target_id"]


def test_deterministic_diverse_views_no_zero_camera_count(source):
    result, group, target = selection_fixture(source)
    selected = bank.select_views(result, group, target, 10, minimum_camera_distance=0)
    assert len(selected) == 4
    assert selected[0]["calibration"]["camera_to_world_usd_row_vectors"][3][0] == 0
    assert selected[1]["calibration"]["camera_to_world_usd_row_vectors"][3][0] == .2
    result["entries"].reverse()
    assert selected == bank.select_views(result, group, target, 10, minimum_camera_distance=0)
    assert all(bank._camera_distance(a, b) > 0 for i, a in enumerate(selected) for b in selected[i+1:])
    selected[0]["review"]["decision"] = "modified_copy"
    assert all(e["review"]["decision"] == "accept" for e in result["entries"])


def test_camera_rotation_not_translation_alone_is_diversity(source):
    result, group, target = selection_fixture(source)
    result["entries"] = result["entries"][:2]
    result["entries"][1]["calibration"]["camera_to_world_usd_row_vectors"] = matrix(angle=.2)
    assert len(bank.select_views(result, group, target)) == 2


def test_plan_and_scene_compatibility_not_donor_only(source):
    entry = source.build()["entries"][0]
    original = source.manifest["variants"][0]
    a = bank._compatibility(entry, source.manifest, original)
    for field in ("source_collection_plan", "source_collection_plan_sha256"):
        modified = deepcopy(entry)
        modified[field] = "other"
        assert bank._digest(a) != bank._digest(bank._compatibility(modified, source.manifest, original))
    other = deepcopy(source.manifest)
    other["lighting"] = {"different": True}
    assert bank._digest(a) != bank._digest(bank._compatibility(entry, other, original))
    result, group, target = selection_fixture(source)
    result["entries"][-1]["compatibility_group"] = "another_plan"
    assert all(e["compatibility_group"] == group for e in bank.select_views(result, group, target, 10))
    with pytest.raises(ValueError, match="compatibility"):
        bank.select_views(result, "unknown", target)


def test_batch_driver_plan_target_ranked_lookup(source):
    result = source.build()
    path = source.manifest["source_collection_plan_path"]
    digest = source.manifest["source_collection_plan_sha256"]
    selected = bank.ranked_references(result, path, source.row["target_id"], source_collection_plan_sha256=digest)
    assert selected == result["entries"]
    assert bank.ranked_references(result, path, "absent", source_collection_plan_sha256=digest) == []
    selected[0]["review"]["decision"] = "changed"
    assert result["entries"][0]["review"]["decision"] == "accept"
    with pytest.raises(ValueError, match="hash mismatch"):
        bank.ranked_references(result, path, source.row["target_id"], source_collection_plan_sha256="0"*64)
    result["by_plan"][path]["targets"][source.row["target_id"]] = []
    with pytest.raises(ValueError, match="bank metadata"):
        bank.check(result)


def test_historical_empty_component_mapping_is_original(source):
    source.manifest["variants"][0]["added_components"] = {}
    assert source.freeze().build()["counts"]["references"] == 1


@pytest.mark.parametrize("different_sample_and_rgb", [False, True])
def test_bank_dedupes_same_sample_or_zero_camera_distance(source, different_sample_and_rgb):
    audit = json.loads(source.audit_path.read_text())
    pins = audit["bindings_sha256"]
    sample_hash, rgb_hash = source.row["source_sample_sha256"], source.row["rgb_sha256"]
    if different_sample_and_rgb:
        sample = deepcopy(source.sample)
        sample["sample_id"] = "sample_0002"
        sample["robot_snapshot"]["joint_degrees"]["torso_0"] = 10.
        folder = source.capture / "sample_0002"
        rgb = folder / "inputs/rgb.png"
        rgb.parent.mkdir(parents=True)
        rgb.write_bytes(b'different RGB noise, identical camera')
        rgb_hash = bank.sha256(rgb)
        sample["files"]["inputs/rgb.png"]["sha256"] = rgb_hash
        sample_hash = write(folder / "sample.json", sample)
        pins[str(rgb)] = rgb_hash
        pins[str(folder / "sample.json")] = sample_hash
        source.manifest["samples"].append(dict(sample_id="sample_0002", target_review_id="N_seed1_1"))
        pins[str(source.capture / "manifest.json")] = write(source.capture / "manifest.json", source.manifest)
        audited = deepcopy(source.audited)
        audited["sample_id"] = "sample_0002"
        audit["samples"].append(audited)
    audit_hash = write(source.audit_path, audit)
    source_manifest = json.loads((source.draft / "source_manifest.json").read_text())
    source_manifest["source_bindings_sha256"] = {**pins, str(source.audit_path): audit_hash}
    hashes = source.release["files_sha256"]
    hashes["source_manifest.json"] = write(source.draft / "source_manifest.json", source_manifest)
    source.release["source_manifest_copy_sha256"] = hashes["source_manifest.json"]
    a = {**source.row, "source_audit_sha256": audit_hash}
    b = {**a, "id": "second", "source_sample_sha256": sample_hash, "rgb_sha256": rgb_hash}
    index = source.draft / "index.jsonl"
    index.write_text(json.dumps(a) + "\n" + json.dumps(b) + "\n")
    hashes["index.jsonl"] = bank.sha256(index)
    reviews = [source.review, {**source.review, "id": "second", "rgb_sha256": rgb_hash}]
    hashes["reviews.json"] = write(source.draft / "reviews.json", reviews)
    write(source.checkpoint / "accepts.json", reviews)
    source.pin = write(source.draft / "manifest.json", source.release)
    result = source.build()
    assert result["counts"]["references"] == result["counts"]["unique_target_camera_poses"] == 1
    assert result["exclusions"]["duplicate_reference_sample_rgb_or_camera_pose"] == 1


def test_native_explicit_failed_trace_ranks_below_unknown(source):
    a = source.build()["entries"][0]
    b = deepcopy(a)
    b["evidence"]["strict_native_trace"].update(status="observed", value=False)
    assert bank.rank_key(a) < bank.rank_key(b)


def test_mount_roundoff_groups_but_real_mount_change_does_not(source):
    entry = source.build()["entries"][0]
    original = source.manifest["variants"][0]
    expected = bank._compatibility(entry, source.manifest, original)
    copied = deepcopy(entry)
    copied["robot_snapshot"]["camera_to_head_column_vectors"][0][1] = -2e-16
    copied["robot_snapshot"]["camera_to_head_column_vectors"][0][0] += 2e-16
    assert bank._digest(bank._compatibility(copied, source.manifest, original)) == bank._digest(expected)
    assert copied["robot_snapshot"]["camera_to_head_column_vectors"][0][1] == -2e-16
    copied["robot_snapshot"]["camera_to_head_column_vectors"][0][3] += .001
    assert bank._digest(bank._compatibility(copied, source.manifest, original)) != bank._digest(expected)


@pytest.mark.parametrize("limit,distance", [(0, 1), (True, 1), (1, -1), (1, float("nan"))])
def test_invalid_selection_options(source, limit, distance):
    result, group, target = selection_fixture(source)
    with pytest.raises(ValueError):
        bank.select_views(result, group, target, limit, minimum_camera_distance=distance)
