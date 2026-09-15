"""CPU-only scheduling fixtures; never launch or qualify a native sensor."""
from copy import deepcopy
import builtins
import json
from pathlib import Path

import pytest

from sim_data.native_capture_v3 import reference_bank as rb
from sim_data.native_capture_v3 import reference_schedule as rs
from sim_data.native_capture_v3.test_reference_bank import Checkpoint, matrix, write


@pytest.fixture
def source(tmp_path):
    checkpoint = Checkpoint(tmp_path).freeze()
    bank = checkpoint.build()
    path = tmp_path / "bank.json"
    digest = write(path, bank)
    return checkpoint, bank, path, digest, tmp_path / "future_qualification"


def make_plan(source, **kwargs):
    _, _, path, digest, root = source
    return rs.plan(path, reference_bank_sha256=digest, output_root=root, **kwargs)


def test_file_backed_checked_bank_exact_provenance_no_writes(source):
    checkpoint, bank, path, digest, root = source
    before = {str(p): rb.sha256(p) for p in checkpoint.root.rglob("*") if p.is_file()}
    schedule = make_plan(source)
    assert rs.check(schedule)
    assert schedule["source_bindings"] == {str(path): digest}
    assert set(schedule["implementation_bindings"]) == {str(p) for p in rs._implementation_paths()}
    assert str(Path(rs.__file__).resolve()) in schedule["implementation_bindings"]
    job = schedule["jobs"][0]
    anchor = job["anchor"]
    original = bank["entries"][0]
    for key, value in original.items():
        assert anchor[key] == value
    assert job["source_collection_plan_sha256"] == original["source_collection_plan_sha256"]
    assert job["compatibility"] == bank["groups"][original["compatibility_group"]]
    assert anchor["qualification_module"] == "sim_data.native_greenhouse_pair"
    assert anchor["qualification_argv"] == ["--source-capture", original["source_capture"],
        "--sample", original["source_sample"], "--output", anchor["qualification_output"]]
    assert anchor["qualification_resolutions"] == [[848, 408], [1696, 816]]
    assert not root.exists()
    assert before == {str(p): rb.sha256(p) for p in checkpoint.root.rglob("*") if p.is_file()}
    assert schedule["counts"]["primary_qualification_pairs"] == 1
    assert schedule["counts"]["primary_paired_sensor_frames"] == 2
    assert schedule["counts"]["new_native_training_rows"] == 0
    assert schedule["counts"]["reused_proofs"] == 0
    assert schedule["native_jobs_launched"] is False
    assert job["collection_after_qualification"]["exact_sample_and_same_camera_required"] is True


@pytest.fixture
def selection_bank(source, monkeypatch):
    """Stub only bank re-audit for selection-unit tests; real binding tests above/below."""
    checkpoint, base, path, _, root = source
    reference = base["entries"][0]
    template = base["groups"][reference["compatibility_group"]]
    groups, entries = {}, []

    def add(family, group_id, target, score, x, suffix):
        entry = deepcopy(reference)
        plan_path = str(checkpoint.root / (group_id + "_original_plan.json"))
        entry.update(id=suffix, source_family=family, target_id=family + "/" + target,
            compatibility_group=group_id, source_collection_plan=plan_path,
            source_sample="sample_" + suffix, source_sample_path=str(checkpoint.root / suffix / "sample.json"),
            source_sample_sha256=rb._digest(suffix))
        entry["calibration"]["camera_to_world_usd_row_vectors"] = matrix(x)
        entry["evidence"]["local_contrast_8bit"]["value"] = score
        group = deepcopy(template)
        group.update(source_family=family, source_collection_plan=plan_path)
        groups[group_id] = group
        entries.append(entry)

    add("seed1", "wide", "target1", 20, 0., "0001")
    add("seed1", "wide", "target2", 30, .1, "0002")
    add("seed1", "wide", "target1", 10, .2, "0003")
    add("seed1", "sparse", "target3", 999, .3, "0004")
    add("seed2", "other", "target4", 40, .4, "0005")
    bank = {**deepcopy(base), "entries": entries, "groups": groups}
    calls = []
    monkeypatch.setattr(rb, "check", lambda b: calls.append(b) or True)
    digest = write(path, bank)
    return checkpoint, bank, path, digest, root, calls


def selection_plan(fixture, **kwargs):
    return rs.plan(fixture[2], reference_bank_sha256=fixture[3], output_root=fixture[4], **kwargs)


def test_coverage_before_quality_best_ranked_anchor_balanced_queue(selection_bank):
    schedule = selection_plan(selection_bank)
    assert len(selection_bank[5]) == 1  # No unchecked plan path.
    first, second = schedule["jobs"]
    assert first["compatibility_group"] == "wide"  # Two targets beat the 999-contrast sparse group.
    assert first["reviewed_target_ids"] == ["seed1/target1", "seed1/target2"]
    assert first["anchor"]["id"] == "0002"
    assert [a["id"] for a in first["fallback_anchors"]] == ["0001", "0003"]
    assert second["fallback_anchors"] == []
    assert [q["source_family"] for q in schedule["qualification_queue"]] == ["seed1", "seed2", "seed1", "seed1"]
    assert schedule["counts"]["maximum_qualification_attempts"] == 4
    assert schedule["counts"]["maximum_paired_sensor_frames"] == 8
    assert schedule["counts"]["selected_groups"] == 2
    assert schedule["counts"]["selected_reviewed_targets"] == 3
    assert all(a["compatibility_group"] == "wide" for a in [first["anchor"], *first["fallback_anchors"]])
    assert schedule["qualification_queue"][2]["prior_attempt_ids"] == [first["anchor"]["attempt_id"]]


def test_equal_coverage_breaks_tie_with_best_rank_key(selection_bank):
    checkpoint, bank, path, _, root, _ = selection_bank
    bank["entries"] = [e for e in bank["entries"] if e["id"] != "0002"]
    digest = write(path, bank)
    schedule = rs.plan(path, reference_bank_sha256=digest, output_root=root)
    assert schedule["jobs"][0]["compatibility_group"] == "sparse"
    assert schedule["jobs"][0]["anchor"]["id"] == "0004"


def test_deterministic_when_bank_entries_groups_reordered(selection_bank):
    a = selection_plan(selection_bank)
    bank = selection_bank[1]
    bank["entries"].reverse()
    bank["groups"] = dict(reversed(list(bank["groups"].items())))
    digest = write(selection_bank[2], bank)
    b = rs.plan(selection_bank[2], reference_bank_sha256=digest, output_root=selection_bank[4])
    assert a["jobs"] == b["jobs"]
    assert a["qualification_queue"] == b["qualification_queue"]


@pytest.mark.parametrize("limit,expected", [(0, 0), (1, 1), (2, 2), (8, 2)])
def test_bounded_fallbacks(selection_bank, limit, expected):
    schedule = selection_plan(selection_bank, fallback_limit=limit)
    assert len(schedule["jobs"][0]["fallback_anchors"]) == expected


@pytest.mark.parametrize("value", [-1, 9, True, 1.5])
def test_invalid_fallback_options(source, value):
    with pytest.raises(ValueError, match="fallback_limit"):
        make_plan(source, fallback_limit=value)


def test_identical_camera_different_rgb_robot_is_not_fallback(selection_bank):
    bank, path = selection_bank[1:3]
    entries = bank["entries"]
    entries[0]["calibration"] = deepcopy(entries[1]["calibration"])
    entries[0]["rgb_sha256"] = "e"*64
    entries[0]["robot_snapshot"]["joint_degrees"]["torso_0"] = 90
    digest = write(path, bank)
    schedule = rs.plan(path, reference_bank_sha256=digest, output_root=selection_bank[4])
    assert [a["id"] for a in schedule["jobs"][0]["fallback_anchors"]] == ["0003"]


@pytest.mark.parametrize("fault", ["heldout", "non_reviewed", "native_resolution", "wrong_group", "wrong_plan", "camera", "unscreened", "unsafe_family"])
def test_invalid_selection_metadata_rejected_even_if_audit_stubbed(selection_bank, fault):
    bank, path = selection_bank[1:3]
    entry = bank["entries"][0]
    if fault == "heldout": entry["split"] = "test"
    elif fault == "non_reviewed": entry["review"]["decision"] = "hold"
    elif fault == "native_resolution": entry["calibration"]["resolution"] = [1696, 816]
    elif fault == "wrong_group": entry["compatibility_group"] = "other"
    elif fault == "wrong_plan": entry["source_collection_plan"] = "invented_plan.json"
    elif fault == "camera": entry["calibration"]["camera_path"] = "/World/Cinematic"
    elif fault == "unscreened": entry["robot_snapshot"]["visual_bound_screen"]["passed"] = False
    elif fault == "unsafe_family": entry["source_family"] = "../elsewhere"
    digest = write(path, bank)
    with pytest.raises(ValueError):
        rs.plan(path, reference_bank_sha256=digest, output_root=selection_bank[4])


@pytest.mark.parametrize("fault", ["code", "bank_binding", "plan", "sample", "group", "queue", "approval", "reuse", "counts"])
def test_schedule_tamper_rejected(source, fault):
    schedule = make_plan(source)
    if fault == "code": schedule["implementation_bindings"][str(Path(rs.__file__).resolve())] = "0"*64
    elif fault == "bank_binding": schedule["source_bindings"] = {}
    elif fault == "plan": schedule["jobs"][0]["source_collection_plan"] = "another_plan"
    elif fault == "sample": schedule["jobs"][0]["anchor"]["source_sample"] = "sample_9999"
    elif fault == "group": schedule["jobs"][0]["compatibility_group"] = "invented_group"
    elif fault == "queue": schedule["qualification_queue"][0]["sequence"] = 9
    elif fault == "approval": schedule["training_approved"] = True
    elif fault == "reuse": schedule["jobs"][0]["anchor"]["reuse_existing_proof"] = True
    elif fault == "counts": schedule["counts"]["new_native_training_rows"] = 2
    with pytest.raises(ValueError):
        rs.check(schedule)


def test_source_bank_and_original_assets_are_rechecked(source):
    checkpoint, _, path, _, _ = source
    schedule = make_plan(source)
    checkpoint.asset.write_bytes(b"changed original geometry")
    with pytest.raises(ValueError, match="Changed bound file"):
        rs.check(schedule)
    with path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="Changed bound file"):
        make_plan(source)


def test_own_code_file_changed_rejected_without_touching_real_code(source, monkeypatch):
    dummy = source[0].root / "fixture_implementation.py"
    dummy.write_text("# original test-only implementation pin")
    monkeypatch.setattr(rs, "_implementation_paths", lambda: (dummy,))
    schedule = make_plan(source)
    dummy.write_text("# changed test-only implementation pin")
    with pytest.raises(ValueError, match="implementation changed"):
        rs.check(schedule)


def test_no_native_or_usd_imports_or_process_launches(source, monkeypatch):
    previous = builtins.__import__

    def guarded(name, *args, **kwargs):
        assert not name.startswith(("pxr", "omni", "isaacsim", "sim_data.generated_capture", "sim_data.native_greenhouse_pair"))
        return previous(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("Unexpected process launch"))
    assert rs.check(make_plan(source))


def test_fresh_only_and_completed_directory_does_not_become_reused_proof(source):
    schedule = make_plan(source)
    destination = Path(schedule["jobs"][0]["anchor"]["qualification_output"])
    destination.mkdir(parents=True)
    write(destination / "result.json", {"looks_successful_but_not_verified": True})
    assert rs.check(schedule)  # Recipe remains immutable; check never adopts the result.
    assert schedule["counts"]["reused_proofs"] == 0
    with pytest.raises(ValueError, match="already exists"):
        rs.require_fresh_outputs(schedule)
    with pytest.raises(ValueError, match="already exists"):
        make_plan(source)


@pytest.mark.parametrize("where", ["capture", "checkpoint", "bank", "ancestor"])
def test_outputs_cannot_overlap_originals(source, where):
    checkpoint, _, path, digest, _ = source
    roots = dict(capture=checkpoint.capture / "new", checkpoint=checkpoint.checkpoint / "new",
                 bank=path, ancestor=checkpoint.root)
    with pytest.raises(ValueError, match="overlaps"):
        rs.plan(path, reference_bank_sha256=digest, output_root=roots[where])


def test_cli_checks_then_writes_only_new_queue_json(source):
    checkpoint, _, path, digest, root = source
    output = checkpoint.root / "queue.json"
    args = ["--reference-bank", str(path), "--reference-bank-sha256", digest,
            "--output-root", str(root), "--output", str(output)]
    rs.main(args)
    schedule = json.loads(output.read_text())
    assert rs.check(schedule)
    assert not root.exists()
    before = rb.sha256(output)
    with pytest.raises(ValueError, match="New queue JSON"):
        rs.main(args)
    assert rb.sha256(output) == before
