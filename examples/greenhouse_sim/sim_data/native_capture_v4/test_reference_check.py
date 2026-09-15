"""Parity and tamper fixtures for the explicitly opt-in v4 verifier."""
import ast
from copy import deepcopy
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ..native_capture_v3 import reference_bank as rb
from ..native_capture_v3 import reference_schedule as rs
from ..native_capture_v3 import test_reference_bank as legacy_tests
from . import _bank_replay
from . import reference_check as fast


class Checkpoint(legacy_tests.Checkpoint):
    def build(self):
        return fast.build(self.checkpoint, manifest_sha256=self.pin)


@pytest.fixture
def source(tmp_path):
    return Checkpoint(tmp_path).freeze()


def test_copied_traversal_has_only_documented_reader_changes():
    expected = inspect.getsource(rb.build)
    expected = expected.replace("def build(checkpoint, *, manifest_sha256):",
                                "def replay(checkpoint, *, manifest_sha256, reader):")
    expected = expected.replace('checkpoint = Path(checkpoint).resolve()',
                                'checkpoint = reader.resolve(checkpoint)')
    expected = expected.replace('draft, reader = checkpoint / "draft", _Reader()',
                                'draft = checkpoint / "draft"')
    expected = expected.replace('capture = Path(audit["source_run"]).resolve()',
                                'capture = reader.resolve(audit["source_run"])')
    expected = expected.replace('Path(p).parent.parent.resolve() == capture',
                                'reader.resolve(Path(p).parent.parent) == capture')
    expected = expected.replace("_safe(", "reader.safe(")
    assert ast.dump(ast.parse(expected)) == ast.dump(ast.parse(inspect.getsource(_bank_replay.replay)))


def test_exact_bank_parity_hash_counts_and_no_source_writes(source):
    before = {str(p): rb.sha256(p) for p in source.root.rglob("*") if p.is_file()}
    original_reader = rb._Reader
    original = rb.build(source.checkpoint, manifest_sha256=source.pin)
    diagnostics = {}
    actual = fast.build(source.checkpoint, manifest_sha256=source.pin, diagnostics=diagnostics)
    assert actual == original
    assert diagnostics["initial_hashes"] == diagnostics["final_hashes"] == len(actual["source_bindings"])
    assert diagnostics["initial_resolutions"] == diagnostics["final_resolutions"]
    assert fast.check_bank(actual)
    assert rb._Reader is original_reader
    assert before == {str(p): rb.sha256(p) for p in source.root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("name", ["asset", "plan", "capture", "sample", "rgb",
                                  "audit", "index", "review", "accepts", "label"])
def test_stale_sources_rejected(source, name):
    bank = source.build()
    paths = dict(asset=source.asset, plan=source.root / "plan.json",
        capture=source.capture / "manifest.json", sample=source.sample_path,
        rgb=source.sample_path.parent / "inputs/rgb.png", audit=source.audit_path,
        index=source.draft / "index.jsonl", review=source.draft / "reviews.json",
        accepts=source.checkpoint / "accepts.json", label=source.draft / "labels/ref.json")
    with paths[name].open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="Changed bound file"):
        fast.check_bank(bank)


@pytest.mark.parametrize("field", ["source_collection_plan_sha256", "source_sample_sha256",
                                    "review", "evidence", "calibration", "training_approved"])
def test_changed_bank_metadata_rejected(source, field):
    bank = source.build()
    bank["entries"][0][field] = "altered"
    with pytest.raises(ValueError, match="bank metadata"):
        fast.check_bank(bank)


@pytest.mark.parametrize("fault", [
    "plan_split", "job_split", "capture_split", "family_conflict", "original_variant",
    "target_lineage", "review_hold", "review_missing_reason", "negative_history",
    "camera_path", "mount_unknown", "geometry_unknown", "geometry_failed", "overlaps",
    "floor", "joint_limits", "pose_audit_mismatch", "reflection", "fk_error",
    "anatomy", "target_review", "label", "duplicate_id"])
def test_all_legacy_rebound_semantic_faults_fail(source, fault):
    # Reuse fixture mutation logic; source.build explicitly invokes v4, not a patch.
    legacy_tests.test_rebound_but_semantically_invalid_evidence_rejected(source, fault)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_heldout_exclusion_unchanged(source, split):
    legacy_tests.test_heldout_excluded_before_dereferencing(source, split)
    assert source.build() == rb.build(source.checkpoint, manifest_sha256=source.pin)


def test_unknown_metrics_and_zero_training_counts(source):
    source.row.pop("legibility")
    source.label.pop("query_usability")
    source.audited.pop("parent_proxy_diagnostic")
    result = source.freeze().build()
    assert all(e["status"] == "unknown" and e["value"] is None
               for e in result["entries"][0]["evidence"].values())
    assert result["counts"]["new_native_training_rows"] == 0
    assert result == rb.build(source.checkpoint, manifest_sha256=source.pin)


@pytest.mark.parametrize("new_rgb", [False, True])
def test_duplicate_camera_sample_semantics_unchanged(source, new_rgb):
    legacy_tests.test_bank_dedupes_same_sample_or_zero_camera_distance(source, new_rgb)
    assert source.build() == rb.build(source.checkpoint, manifest_sha256=source.pin)


@pytest.fixture
def schedule_source(source):
    bank = source.build()
    path = source.root / "bank.json"
    pin = legacy_tests.write(path, bank)
    schedule = rs.plan(path, reference_bank_sha256=pin, output_root=source.root / "future")
    return source, bank, path, pin, schedule


def test_schedule_parity_and_separate_bound_receipt(schedule_source):
    source, bank, path, pin, schedule = schedule_source
    original = deepcopy(schedule)
    stats = {}
    assert fast.check_schedule(schedule, diagnostics=stats)
    assert rs.check(schedule)
    assert schedule == original
    schedule_path = source.root / "schedule.json"
    schedule_pin = legacy_tests.write(schedule_path, schedule)
    receipt = fast.verify_files(path, bank_sha256=pin, schedule_path=schedule_path,
                                schedule_sha256=schedule_pin)
    assert receipt["bank_counts"] == bank["counts"]
    assert receipt["artifact_bindings"] == {str(path): pin, str(schedule_path): schedule_pin}
    assert str(Path(fast.__file__).resolve()) in receipt["implementation_bindings"]
    assert str(Path(_bank_replay.__file__).resolve()) in receipt["implementation_bindings"]
    assert receipt["schema"] == fast.VERSION
    assert receipt["passed"] and not receipt["native_jobs_launched"]
    assert not (source.root / "future").exists()
    assert rb.sha256(path) == pin and rb.sha256(schedule_path) == schedule_pin
    assert stats["bank"]["initial_hashes"] == stats["bank"]["final_hashes"] == len(bank["source_bindings"])


@pytest.mark.parametrize("fault", ["count", "anchor", "group", "queue", "code", "bank_pin"])
def test_schedule_tampering_fails(schedule_source, fault):
    _, _, _, _, schedule = schedule_source
    if fault == "count":
        schedule["counts"]["selected_reviewed_targets"] += 1
    elif fault == "anchor":
        schedule["jobs"][0]["anchor"]["source_sample"] = "different"
    elif fault == "group":
        schedule["jobs"][0]["compatibility_group"] = "different"
    elif fault == "queue":
        schedule["qualification_queue"][0]["sequence"] += 1
    elif fault == "code":
        schedule["implementation_bindings"] = {}
    else:
        schedule["source_reference_bank_sha256"] = "a" * 64
        schedule["source_bindings"] = {schedule["source_reference_bank"]: "a" * 64}
    with pytest.raises(ValueError):
        fast.check_schedule(schedule)


def test_explicit_artifact_pin_and_code_binding_required(schedule_source, monkeypatch):
    source, _, path, pin, _ = schedule_source
    with pytest.raises(ValueError, match="Changed bound file"):
        fast.verify_files(path, bank_sha256="a" * 64)
    synthetic_code = source.root / "synthetic_code"
    synthetic_code.write_bytes(b"bound")
    bindings = {**fast._LOADED_BINDINGS, str(synthetic_code): rb.sha256(synthetic_code)}
    monkeypatch.setattr(fast, "_LOADED_BINDINGS", bindings)
    synthetic_code.write_bytes(b"changed")
    with pytest.raises(ValueError, match="implementation changed"):
        fast.check_bank(source.build())


def test_cli_is_readonly(schedule_source, capsys):
    _, bank, path, pin, _ = schedule_source
    fast.main(["--bank", str(path), "--bank-sha256", pin])
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["bank_counts"] == bank["counts"]
    assert receipt["passed"]


def test_no_native_dependencies():
    # Other suites may legitimately import USD during collection. Test this
    # module's own imports in an isolated, bytecode-disabled CPU interpreter.
    package_root = str(Path(__file__).resolve().parents[2])
    command = ("import sys; sys.path.insert(0, " + repr(package_root) + "); "
               "import sim_data.native_capture_v4.reference_check; "
               "assert not any(n == 'pxr' or n == 'isaacsim' or n.startswith('omni.') "
               "for n in sys.modules)")
    subprocess.run([sys.executable, "-B", "-c", command], check=True,
                   capture_output=True, text=True, timeout=20)
