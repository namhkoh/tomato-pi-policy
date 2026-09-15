"""V4 wrapper tests: real file/hash/plan handoff, synthetic generation operations.

No plants or native frames are created. Only the generator/full-geometry replay
and completed-sensor-proof gate are replaced on the NEW v4 module in fixtures;
all bank/schedule checks, source selection, destination checks, and the actual
existing hash-only native-plan handoff remain live.
"""
from copy import deepcopy
import ast
import inspect
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ..native_capture_v3 import prepare as legacy
from ..native_capture_v3 import reference_schedule as schedule_api
from ..native_capture_v3.test_reference_bank import Checkpoint, write
from .. import generated_capture, native_multitarget_plan
from ..dataset_review import read_json, verify_bindings
from ..depth_preview import sha256
from . import prepare as module


class FixtureGeneration:
    def __init__(self, bank, job, proof):
        self.bank, self.job, self.proof = bank, job, proof
        self.events = []
        self.fail = None
        self.on_generate = None
        self.built_plan = None

    def load_training_sources(self, path):
        self.events.append(("load_sources", path))
        return {}, {self.job["source_family"]: {}}

    def training_envelope(self, frozen, sources):
        self.events.append(("envelope",))
        return {}

    def plan_change(self, source, component, envelope, seed):
        self.events.append(("plan_change", component, seed))
        if self.fail == "anchor":
            raise ValueError("fixture anchor rejected")

    def generate(self, plan, family, targets, seed, output):
        self.events.append(("generate", plan, family, targets, seed, str(output)))
        if self.on_generate:
            self.on_generate()
        if self.fail == "generate":
            raise ValueError("fixture generation failed")
        # Deliberately create no assets or variant directory.

    def prepare_plan(self, capture, sample, variant, proof):
        self.events.append(("prepare_plan", capture, sample, str(proof)))
        reference = next(e for e in self.bank["entries"] if e["source_sample"] == sample)
        row = deepcopy(reference["source_row"])
        generated = dict(row, variant_id=variant.name,
            target_id=variant.name + "/" + row["component_id"],
            conservative_view_cap_group=row["target_id"])
        return dict(schema_version=generated_capture.SCHEMA,
            state="cpu_planned_pending_native_sensor_prerequisite",
            source_capture=capture, source_sample=sample,
            source_collection_plan=reference["source_collection_plan"],
            source_family=reference["source_family"], source_row=row, generated_row=generated,
            prerequisite_directory=str(proof), split="train", split_group=reference["source_family"],
            conservative_view_cap_group=row["target_id"], resolution=[1696, 816],
            camera_path=reference["calibration"]["camera_path"],
            expected_calibration=dict(camera_path=reference["calibration"]["camera_path"], resolution=[1696, 816]),
            modes=["original_control", "generated_variant"], sample_count_limit=2,
            training_eligible=False, collision_qualified=False, dynamics_supported=False,
            higher_resolution_execution_verified=False,
            source_bindings={reference["source_collection_plan"]: reference["source_collection_plan_sha256"]})

    def check_plan(self, plan):
        self.events.append(("check_base",))
        return generated_capture.check_plan(plan)

    def verify_sensor_prerequisite(self, plan):
        self.events.append(("sensor_gate", plan["source_sample"], plan["prerequisite_directory"]))
        verify_bindings(self.proof)
        return {"bindings": self.proof}

    def build(self, anchor, bases, views):
        self.events.append(("build", views, len(bases)))
        cases, bindings = [], {}
        for path in bases:
            base = read_json(path)
            bindings.update(base["source_bindings"])
            bindings[str(path)] = sha256(path)
            cases.append(dict(base_pair_plan=str(path), base_pair_plan_sha256=sha256(path),
                target_id=base["generated_row"]["target_id"],
                conservative_view_cap_group=base["source_row"]["target_id"],
                views=[{"candidate_id": "fixture_" + str(i)} for i in range(views)]))
        result = dict(schema=native_multitarget_plan.SCHEMA, anchor_pair_plan=str(anchor),
            source_bindings=bindings, prerequisite_bindings=self.proof,
            implementation_bindings={str(Path(native_multitarget_plan.__file__).resolve()):
                                     sha256(native_multitarget_plan.__file__)},
            target_cases=cases, views_per_target=views, maximum_native_frames=views*len(cases),
            source_family=self.job["source_family"], split="train", resolution=[1696, 816],
            requested_render_subframes_per_view=56, source_cap_reset=False, training_approved=False,
            physical_motion_commanded=False, hidden_cut_coordinates_executable=False)
        self.built_plan = deepcopy(result)
        return result

    def check(self, plan, *, replay_geometry):
        self.events.append(("check_batch", replay_geometry))
        assert replay_geometry is True  # Production must request full replay.
        # Geometry itself is a fixture stub, but exercise the real hash/profile gates.
        return native_multitarget_plan.check(plan, replay_geometry=False)


@pytest.fixture
def fixture(tmp_path, monkeypatch):
    root = tmp_path / "sources"
    root.mkdir()
    checkpoint = Checkpoint(root)
    package = root / "package"
    package.mkdir()
    checkpoint.plan["package"] = str(package)
    checkpoint.freeze()
    bank = checkpoint.build()
    bank_path = root / "bank.json"
    bank_pin = write(bank_path, bank)
    schedule = schedule_api.plan(bank_path, reference_bank_sha256=bank_pin,
                                 output_root=tmp_path / "qualifications")
    schedule_path = tmp_path / "schedule.json"
    schedule_pin = write(schedule_path, schedule)
    job = schedule["jobs"][0]
    attempt = job["anchor"]
    proof_path = Path(attempt["qualification_output"]) / "fixture_proof.json"
    proof = {str(proof_path): write(proof_path, {"synthetic_fixture_not_native_evidence": True})}
    api = FixtureGeneration(bank, job, proof)

    def sensor_gate(candidate):
        api.events.append(("fresh_anchor", candidate["id"]))
        assert candidate == attempt
        verify_bindings(proof)
        return {"bindings": proof}

    monkeypatch.setattr(module, "verify_fresh_anchor", sensor_gate)
    monkeypatch.setattr(module, "_generation_api", lambda: api)
    return dict(checkpoint=checkpoint, bank=bank, schedule=schedule, schedule_path=schedule_path,
                schedule_pin=schedule_pin, job=job, attempt=attempt, proof=proof, api=api,
                output=tmp_path / "new_preparation")


def prepare(f, **kwargs):
    return module.prepare(f["schedule_path"], f["job"]["job_id"], f["attempt"]["attempt_id"],
                          810000, f["output"], schedule_sha256=f["schedule_pin"], **kwargs)


def handoff(f, **kwargs):
    return module.verify_prepared(f["output"], f["schedule_path"], f["job"], f["attempt"],
                                  810000, schedule_sha256=f["schedule_pin"], **kwargs)


def rewrite_receipts(f, change):
    for name in ("prepare_request.json", "prepared.json"):
        path = f["output"] / name
        value = read_json(path)
        change(value)
        write(path, value)


def rewrite_plan(f, change):
    path = f["output"] / "plan.json"
    value = read_json(path)
    change(value)
    pin = write(path, value)
    result_path = f["output"] / "prepared.json"
    result = read_json(result_path)
    result["plan_sha256"] = pin
    write(result_path, result)


def test_v4_flow_and_real_handoff_preserve_plan_and_source_bytes(fixture):
    f = fixture
    before = {str(p): sha256(p) for p in f["checkpoint"].root.rglob("*") if p.is_file()}
    result = prepare(f)
    assert handoff(f) == result
    assert result["schema"] == module.SCHEMA and result["preparation_module"] == module.MODULE
    assert result["reference_verifier"] == module.verifier.VERSION
    assert str(Path(module.__file__).resolve()) in result["implementation_bindings"]
    assert result["implementation_bindings"] != {str(Path(legacy.__file__).resolve()): sha256(legacy.__file__)}
    assert result["verification_io"]["bank"]["initial_hashes"] == len(f["bank"]["source_bindings"])
    assert result["verification_io"]["bank"]["final_hashes"] == len(f["bank"]["source_bindings"])
    assert result["reference_ids"][0] == f["attempt"]["id"]
    assert result["maximum_native_frames"] == 6
    assert read_json(f["output"] / "plan.json") == f["api"].built_plan
    assert all(sha256(path) == pin for path, pin in before.items())
    events = f["api"].events
    assert events[0][0] == "fresh_anchor"
    assert events.index(("check_base",)) < events.index(("build", 6, 1))
    assert ("check_batch", True) in events
    assert ("sensor_gate", f["attempt"]["source_sample"], f["attempt"]["qualification_output"]) in events
    assert not result["native_capture_started"] and not result["training_approved"]
    # Handoff only checks already prepared bindings; no extra generation operations.
    count = len(events)
    assert handoff(f) == result
    assert len(events) == count + 1  # exact fresh-anchor proof verification


def test_v3_handoff_rejects_v4_identity(fixture):
    f = fixture
    prepare(f)
    with pytest.raises(ValueError, match="implementation changed"):
        legacy.verify_prepared(f["output"], f["schedule_path"], f["job"], f["attempt"], 810000)


@pytest.mark.parametrize("field,value", [
    ("schema", "v3"), ("preparation_module", "sim_data.native_capture_v3.prepare"),
    ("reference_verifier", "unverified"), ("implementation_bindings", {}),
])
def test_v4_handoff_rejects_wrong_version_or_identity(fixture, field, value):
    prepare(fixture)
    rewrite_receipts(fixture, lambda d: d.update({field: value}))
    with pytest.raises(ValueError):
        handoff(fixture)


@pytest.mark.parametrize("field,value", [
    ("training_approved", True), ("source_cap_reset", True), ("native_capture_started", True),
    ("physical_execution_approved", True), ("seed", 810001), ("max_targets", 4),
    ("reference_ids", ["other"]), ("targets", ["other"]), ("maximum_native_frames", 12),
])
def test_handoff_rejects_rebound_receipt_claims(fixture, field, value):
    prepare(fixture)
    rewrite_receipts(fixture, lambda d: d.update({field: value}))
    with pytest.raises(ValueError):
        handoff(fixture)


@pytest.mark.parametrize("field,value", [
    ("split", "test"), ("resolution", [848, 408]), ("source_cap_reset", True),
    ("views_per_target", 5), ("maximum_native_frames", 12),
    ("requested_render_subframes_per_view", 8), ("prerequisite_bindings", {}),
])
def test_handoff_keeps_actual_native_plan_gates(fixture, field, value):
    prepare(fixture)
    rewrite_plan(fixture, lambda d: d.update({field: value}))
    with pytest.raises(ValueError):
        handoff(fixture)


@pytest.mark.parametrize("field,value", [
    ("source_sample", "different_sample"), ("source_capture", "different_capture"),
    ("source_family", "different_family"), ("prerequisite_directory", "different_proof"),
])
def test_rebound_base_identity_rejected(fixture, field, value):
    f = fixture
    prepare(f)
    path = Path(f["api"].built_plan["anchor_pair_plan"])
    base = read_json(path)
    base[field] = value
    pin = write(path, base)

    def rebind(plan):
        plan["target_cases"][0]["base_pair_plan_sha256"] = pin
        plan["source_bindings"][str(path)] = pin

    rewrite_plan(f, rebind)
    with pytest.raises(ValueError):
        handoff(f)


def test_changed_plan_bytes_and_changed_caller_job_fail(fixture):
    f = fixture
    prepare(f)
    changed_job = dict(f["job"], compatibility_group="other")
    with pytest.raises(ValueError, match="differs from schedule"):
        module.verify_prepared(f["output"], f["schedule_path"], changed_job, f["attempt"], 810000)
    with (f["output"] / "plan.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="plan missing or changed"):
        handoff(f)


@pytest.mark.parametrize("which", ["existing", "checkpoint", "capture", "qualification", "package"])
def test_destination_must_be_fresh_and_disjoint(fixture, which):
    f = fixture
    if which == "existing":
        f["output"].mkdir()
    else:
        f["output"] = dict(checkpoint=f["checkpoint"].checkpoint / "new",
            capture=f["checkpoint"].capture / "new",
            qualification=Path(f["schedule"]["qualification_output_root"]) / "new",
            package=Path(f["checkpoint"].plan["package"]) / "new")[which]
    with pytest.raises(ValueError, match="disjoint"):
        prepare(f)
    assert not f["api"].events


def test_exact_anchor_proof_required_before_generation(fixture, monkeypatch):
    f = fixture
    monkeypatch.setattr(module, "verify_fresh_anchor", legacy.verify_fresh_anchor)
    with pytest.raises(ValueError, match="qualification required"):
        prepare(f)
    assert not f["api"].events and not f["output"].exists()


@pytest.mark.parametrize("failure", ["anchor", "generate"])
def test_failure_never_creates_completed_preparation(fixture, failure):
    f = fixture
    f["api"].fail = failure
    with pytest.raises(ValueError):
        prepare(f)
    assert not (f["output"] / "prepared.json").exists()
    if failure == "anchor":
        assert not f["output"].exists()
    else:
        assert (f["output"] / "prepare_request.json").exists()
        with pytest.raises(ValueError, match="disjoint"):
            prepare(f)


@pytest.mark.parametrize("what", ["schedule", "proof"])
def test_changed_input_during_generation_cannot_complete(fixture, what):
    f = fixture
    path = f["schedule_path"] if what == "schedule" else Path(next(iter(f["proof"])))

    def tamper():
        with path.open("ab") as stream:
            stream.write(b" ")

    f["api"].on_generate = tamper
    with pytest.raises(ValueError):
        prepare(f)
    assert not (f["output"] / "prepared.json").exists()


def test_wrong_explicit_schedule_pin_fails_before_work(fixture):
    f = fixture
    with pytest.raises(ValueError, match="Changed bound file"):
        module.prepare(f["schedule_path"], f["job"]["job_id"], f["attempt"]["attempt_id"],
                       810000, f["output"], schedule_sha256="a"*64)
    assert not f["api"].events and not f["output"].exists()


def test_cli_prepare_and_verify_only_use_new_handoff(fixture, capsys):
    f = fixture
    args = ["--schedule", str(f["schedule_path"]), "--schedule-sha256", f["schedule_pin"],
            "--job-id", f["job"]["job_id"], "--attempt-id", f["attempt"]["attempt_id"],
            "--seed", "810000", "--output", str(f["output"])]
    module.main(args)
    assert capsys.readouterr().out.startswith("REFERENCE_JOB_PREPARED_V4 ")
    before = len(f["api"].events)
    module.main([*args, "--verify-only"])
    assert capsys.readouterr().out.startswith("REFERENCE_JOB_VERIFIED_V4 ")
    assert len(f["api"].events) == before + 1


def test_explicit_fast_call_no_legacy_prepare_or_global_patch():
    tree = ast.parse(inspect.getsource(module))
    calls = {ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    assert "verifier.check_schedule" in calls
    assert "legacy.prepare" not in calls and "legacy.verify_prepared" not in calls
    assert "setattr" not in calls and "exec" not in calls
    assert module.choose_references is legacy.choose_references
    assert module.validate_destination is legacy.validate_destination
    handoff_source = inspect.getsource(module._check_handoff_plan)
    assert "replay_geometry=False" in handoff_source


def test_cli_help_imports_no_native_modules():
    root = str(Path(__file__).resolve().parents[2])
    command = ("import sys; sys.path.insert(0, " + repr(root) + "); "
               "from sim_data.native_capture_v4 import prepare; "
               "assert not any(n=='pxr' or n=='isaacsim' or n.startswith('omni.') for n in sys.modules); "
               "prepare.main(['--help'])")
    result = subprocess.run([sys.executable, "-B", "-c", command], check=True,
                            capture_output=True, text=True, timeout=20)
    assert "--verify-only" in result.stdout and "--schedule-sha256" in result.stdout
