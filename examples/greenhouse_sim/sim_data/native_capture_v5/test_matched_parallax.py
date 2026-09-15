"""CPU fixtures only; no native app, render, capture or production mutation."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from . import matched_parallax as m
from .. import native_multitarget_plan as multi
from ..depth_preview import sha256
from ..native_dataset.bundle import REQUIRED
from ..native_view_plan import propose_specs


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def factory(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    receipts, calls = {}, []

    def verify(output, schedule, job, attempt, seed, **kwargs):
        receipt = receipts[Path(output)]
        assert receipt["schedule"] == str(schedule)
        assert receipt["job_id"] == job["job_id"] and receipt["attempt_id"] == attempt["attempt_id"]
        assert receipt["seed"] == seed
        assert kwargs == dict(max_targets=12, schedule_sha256=receipt["schedule_sha256"])
        calls.append(Path(output))
        return deepcopy(receipt)

    monkeypatch.setattr(m.v4, "verify_prepared", verify)

    def create(count=5, selected=None):
        selected = count-1 if selected is None else selected
        root = tmp_path / ("source" + str(len(receipts)))
        job_root = root / "completed_job"
        prep = job_root / "prepare_seed100"
        package = tmp_path / "data/sim_data/diagnostics/source_package"
        source_plan = write(root / "collection.json", dict(package=str(package)))
        proof = write(root / "proof.json", {"fixture": True})
        optics = {k: "unchanged" for k in multi.OPTICS}
        optics.update(resolution=[1696, 816], crop_resize=None)
        matrix = np.eye(4); matrix[3, :3] = [.8, 1., .1]
        reference = dict(robot_root_to_world_usd_row_vectors=matrix.tolist(),
            camera_to_head_column_vectors=np.eye(4).tolist(), visual_bound_screen={"passed": True},
            joint_degrees={"torso_0": 0., "right_arm_1": -5., "head_0": 0., "head_1": 0.})
        bases, cases = [], []
        for index in range(count):
            component = "SubStem_" + str(40+index)
            base = dict(source_family="original", split="train", split_group="original",
                variant_directory=str(root / "variant"), source_capture=str(root / "old_capture"),
                prerequisite_directory=str(root / "proof"), source_collection_plan=str(source_plan),
                original_variant={"variant_id": "original"}, expected_scene_counts={"plants": 2},
                expected_calibration=deepcopy(optics), expected_robot_snapshot=deepcopy(reference),
                source_row=dict(component_id=component, target_id="original/" + component),
                generated_row=dict(component_id=component, target_id="generated/" + component),
                conservative_view_cap_group="original/" + component)
            path = write(prep / (component + "_base.json"), base)
            target = [0., 1., .8]
            specs = propose_specs(matrix, target, sha256(path), 6)
            for spec in specs: spec["candidate_id"] = component + "_" + spec["candidate_id"]
            cases.append(dict(base_pair_plan=str(path), base_pair_plan_sha256=sha256(path),
                target_id=base["generated_row"]["target_id"], expected_nominal_world_m=target,
                views=specs, conservative_view_cap_group=base["conservative_view_cap_group"]))
            bases.append(base)
        parent = dict(schema=multi.SCHEMA, target_cases=cases, anchor_pair_plan=cases[0]["base_pair_plan"],
            views_per_target=6, maximum_native_frames=6*count, source_family="original", split="train",
            resolution=[1696, 816], requested_render_subframes_per_view=56, source_cap_reset=False,
            training_approved=False, physical_motion_commanded=False, hidden_cut_coordinates_executable=False,
            source_bindings={str(source_plan): sha256(source_plan),
                **{c["base_pair_plan"]: c["base_pair_plan_sha256"] for c in cases}},
            prerequisite_bindings={str(proof): sha256(proof)}, implementation_bindings={str(proof): sha256(proof)})
        prepared_plan = write(prep / "plan.json", parent)
        plan_path = write(job_root / "plan.json", parent)
        scheduled = dict(job_id="job1", anchor={"attempt_id": "anchor1"}, fallback_anchors=[])
        schedule = write(root / "schedule.json", {"jobs": [scheduled]})
        receipt = dict(schema=m.v4.SCHEMA, state=m.v4.STATE, job_id="job1", attempt_id="anchor1", seed=100,
            source_family="original", max_targets=12, schedule=str(schedule), schedule_sha256=sha256(schedule),
            plan_path=str(prepared_plan), plan_sha256=sha256(prepared_plan), implementation_bindings={})
        prepared = write(prep / "prepared.json", receipt)
        write(prep / "prepare_request.json", {})
        write(job_root / "reference_provenance.json", receipt)
        receipts[prep] = receipt
        case = cases[selected]
        sid = case["views"][0]["candidate_id"]
        sample = job_root / "capture" / sid
        files = {}
        for name in REQUIRED:
            file = sample / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(b"synthetic unread native buffer")
            files[name] = dict(sha256=sha256(file), role="observation" if name.startswith("inputs/")
                               else "ground_truth_supervision")
        pose = deepcopy(reference)
        pose["camera_to_head_column_vectors"][0][3] += 5e-17  # Real USD roundoff; frozen tolerance is 1e-9.
        pose["desired_cut_pixel_xy"] = deepcopy(case["views"][0]["desired_pixel_xy"])
        meta = dict(files=files, supervision=dict(target_id=case["target_id"], nominal_world_m=[0., 1., .8]),
            robot_snapshot=pose, geometry_screen={"passed": True}, calibration=optics,
            lighting=dict(day=172, minutes=780., dome_intensity=6000.))
        metadata = write(sample / "sample.json", meta)
        label = write(sample / "supervision/label.json", dict(eligible=False))
        rows = []
        for c in cases:
            for spec in c["views"]:
                row = dict(candidate_id=spec["candidate_id"], target_id=c["target_id"],
                    requested_spec=deepcopy(spec), state="rejected_possible_geometry_overlap")
                if spec["candidate_id"] == sid:
                    row.update(state="native_captured_pending_review", robot_snapshot=deepcopy(pose),
                        sample_sha256=sha256(metadata), label_sha256=sha256(label), query_trace=None,
                        label_reason="cut_interval_not_fully_native_visible")
                rows.append(row)
        result = write(job_root / "capture/result.json", dict(
            state="native_generated_multiview_pilot_complete_pending_review", records=rows,
            captured_frames=1, training_approved=False))
        request = write(job_root / "capture/request.json", dict(plan_path=str(plan_path),
            plan_sha256=sha256(plan_path), render_budget_profile="warm56_then8_trial", storage_backend="fixture_raw"))
        record = dict(sample=str(sample), target_id=case["target_id"], decision="exclude_geometry_or_visibility",
            sample_sha256=sha256(metadata), label_sha256=sha256(label), trace_replayed_exact=False,
            **{k: True for k in ("label_replayed_exact",
                               "native_callback_hashes_verified", "source_and_file_hashes_verified")})
        counts = {"exclude_geometry_or_visibility": 1}
        audit = write(job_root / "automatic_audit.json", dict(state="completed_automatic_annotation_replay",
            training_approved=False, records=[record], counts=counts, result_sha256=sha256(result),
            request_sha256=sha256(request), plan_sha256=sha256(plan_path)))
        done = write(job_root / "campaign_result.json", dict(job=str(job_root), state="native_complete_pending_review",
            native_exit_code=0, training_approved=False, job_id="job1", attempt_id="anchor1", seed=100,
            source_family="original", captured=1, audit_counts=counts))
        return SimpleNamespace(root=root, job_root=job_root, args=(done, sha256(done)), target_id=case["target_id"],
            parent=parent, bases=bases, selected=selected, case=case, meta=meta, sample=sample,
            prepared=prepared, request=request, result=result, audit=audit, calls=calls)
    return create


def build(fixture):
    return m.build(*fixture.args, target_id=fixture.target_id)


@pytest.mark.parametrize("count", [1, 2, 5, 12])
def test_one_selected_target_from_any_supported_parent(factory, count):
    f = factory(count)
    before = deepcopy(f.parent)
    plan = build(f)
    assert f.parent == before
    assert plan["maximum_native_frames"] == plan["views_per_target"] == 3
    assert len(plan["target_cases"]) == 1
    assert plan["anchor_pair_plan"] == f.parent["anchor_pair_plan"]
    assert plan["target_cases"][0]["base_pair_plan"] == f.case["base_pair_plan"]
    assert m.check(plan) == f.bases[0]
    jobs = multi.capture_jobs(plan, f.bases[0])
    assert len(jobs) == 3 and all(j[0] == f.bases[f.selected] for j in jobs)
    assert plan["source_cap_reset"] is False and plan["training_approved"] is False
    original = f.case["views"][0]
    allowed = {"candidate_id", "y_offset_m", "base_target_radius_m", "base_displacement_from_reference_m"}
    positions = []
    for lateral, (base, case, spec) in zip(m.OFFSETS, jobs, strict=True):
        assert {k: v for k, v in spec.items() if k not in allowed} == {
            k: v for k, v in original.items() if k not in allowed}
        assert spec["root_x_m"] == original["root_x_m"]
        assert spec["y_offset_m"] == pytest.approx(original["y_offset_m"] + lateral)
        assert spec["desired_pixel_xy"] == original["desired_pixel_xy"]
        positions.append(m.bounded_reference_root(base["expected_robot_snapshot"], spec,
                                                   case["expected_nominal_world_m"])[:2, 3].tolist())
    assert len({tuple(p) for p in positions}) == 3
    assert plan["experiment"]["collect_kwargs"] == dict(profile_render=False, instance_backend="fast",
                                                        render_budget="reference56")
    assert plan["experiment"]["requested_subframes_every_capture"] == 56
    assert plan["native_receipt_contract"]["owner"] == "separate_native_worker_not_cpu_preparation"


@pytest.mark.parametrize("field", ["x", "y", "framing", "yaw", "policy", "target", "extra_target", "anchor",
    "count", "approval", "cap", "budget", "storage", "light", "metadata_pin", "sources", "implementations",
    "receipt_owner", "original_count", "schema", "numeric_type", "unknown"])
def test_every_field_rebuilt_and_tamper_rejected(factory, field):
    p = build(factory())
    s = p["target_cases"][0]["views"][1]
    if field == "x": s["root_x_m"] += .001
    elif field == "y": s["y_offset_m"] += .001
    elif field == "framing": s["desired_pixel_xy"][0] += 1
    elif field == "yaw": s["root_yaw_degrees"] += 1
    elif field == "policy": s["view_policy"] = "relaxed"
    elif field == "target": p["target_cases"][0]["target_id"] = "foreign"
    elif field == "extra_target": p["target_cases"].append(deepcopy(p["target_cases"][0]))
    elif field == "anchor": p["anchor_pair_plan"] = p["target_cases"][0]["base_pair_plan"]
    elif field == "count": p["maximum_native_frames"] = 6
    elif field == "approval": p["training_approved"] = True
    elif field == "cap": p["target_cases"][0]["conservative_view_cap_group"] = "new"
    elif field == "budget": p["experiment"]["collect_kwargs"]["render_budget"] = "warm56_then8_trial"
    elif field == "storage": p["experiment"]["storage_backend"] = "compact"
    elif field == "light": p["baseline_evidence"]["lighting"]["dome_intensity"] = 8000
    elif field == "metadata_pin": p["baseline_evidence"]["sample_sha256"] = "0"*64
    elif field == "sources": p["source_bindings"] = {}
    elif field == "implementations": p["implementation_bindings"] = {}
    elif field == "receipt_owner": p["native_receipt_contract"]["owner"] = "cpu"
    elif field == "original_count": p["original_v4"]["original_target_count"] = 2
    elif field == "schema": p["schema"] = multi.SCHEMA
    elif field == "numeric_type": p["maximum_native_frames"] = 3.0
    else: p["unreviewed_override"] = True
    with pytest.raises(ValueError): m.check(p)


@pytest.mark.parametrize("field", ["native_exit_code", "state", "training_approved", "captured"])
def test_completed_parent_required_even_with_new_receipt_hash(factory, field):
    f = factory()
    d = json.loads(f.args[0].read_text())
    d[field] = {"native_exit_code": 1, "state": "running", "training_approved": True, "captured": 2}[field]
    write(f.args[0], d)
    with pytest.raises(ValueError): m.build(f.args[0], sha256(f.args[0]), target_id=f.target_id)


def test_explicit_pin_and_target_required(factory):
    f = factory()
    with pytest.raises(ValueError, match="Pinned file"):
        m.build(f.args[0], "0"*64, target_id=f.target_id)
    with pytest.raises(ValueError, match="Exactly one"):
        m.build(*f.args, target_id="absent")
    with pytest.raises(ValueError, match="view001"):
        m.build(*f.args, target_id=f.parent["target_cases"][0]["target_id"])


def test_v4_verification_not_bypassed(factory, monkeypatch):
    f = factory()
    def reject(*args, **kwargs): raise ValueError("Original V4 gate failed")
    monkeypatch.setattr(m.v4, "verify_prepared", reject)
    with pytest.raises(ValueError, match="Original V4"): build(f)


def test_no_trace_exclusion_uses_exact_frozen_audit_semantics(factory):
    f = factory()
    build(f)  # Excluded labels have no trace, and trace_replayed_exact is False.
    audit = json.loads(f.audit.read_text())
    audit["records"][0]["trace_replayed_exact"] = True
    write(f.audit, audit)
    with pytest.raises(ValueError, match="Original replay binding differs"): build(f)


@pytest.mark.parametrize("name", ["sample.json", "supervision/label.json", "inputs/depth_m.npy", "inputs/rgb.png"])
def test_changed_baseline_bytes_rejected(factory, name):
    f = factory()
    p = build(f)
    path = f.sample / name
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError): m.check(p)


def test_unconditional_final_reread(factory):
    f = factory()
    inputs = m._Inputs()
    inputs.json(f.args[0], f.args[1])
    f.args[0].write_bytes(f.args[0].read_bytes() + b" ")
    with pytest.raises(ValueError, match="Stale"): inputs.finish()


def test_native_handoff_checks_full_original_only_after_app(factory, monkeypatch):
    f = factory(5)
    p = build(f)
    calls = []
    def replay(parent, *, replay_geometry):
        assert parent == f.parent and len(parent["target_cases"]) == 5 and replay_geometry is True
        calls.append("full_original")
        return deepcopy(f.bases[0])
    monkeypatch.setattr(multi, "check", replay)
    m.check(p)
    assert calls == []
    with pytest.raises(ValueError, match="Running SimulationApp"): m.validate_for_collect(None, p)
    class FakeApp:
        def __init__(self, running): self.running = running
        def is_running(self): return self.running
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(SimulationApp=FakeApp))
    with pytest.raises(ValueError, match="Running SimulationApp"): m.validate_for_collect(FakeApp(False), p)
    assert calls == []
    assert m.validate_for_collect(FakeApp(True), p) == f.bases[0]
    assert calls == ["full_original"]


def test_create_only_cpu_cli_no_native_receipts(factory, capsys):
    f = factory()
    out = m.ROOT / "data/sim_data/diagnostics/matched/plan.json"
    args = ["plan", "--completed", str(f.args[0]), "--completed-sha256", f.args[1],
            "--target-id", f.target_id, "--output", str(out)]
    m.main(args)
    original = out.read_bytes()
    m.main(["verify", "--plan", str(out), "--plan-sha256", sha256(out)])
    assert '"native_launched": false' in capsys.readouterr().out
    with pytest.raises(ValueError): m.main(args)
    assert out.read_bytes() == original
    assert {p.name for p in out.parent.iterdir()} == {"plan.json"}
    with pytest.raises(SystemExit): m.main(["capture"])


def test_disjoint_diagnostic_destination_and_explicit_runtime_pins(factory):
    f = factory()
    p = build(f)
    for path in [f.job_root / "new.json", m.ROOT / "data/sim_data/diagnostics/source_package/new.json",
                 m.ROOT / "outside.json", m.ROOT / "data/sim_data/diagnostics/new.py"]:
        with pytest.raises(ValueError): m._destination(path, p)
    namespace = Path(m.__file__).resolve().parent
    assert {Path(s).name for s in p["implementation_bindings"] if Path(s).parent == namespace} == {
        "matched_parallax.py", "__init__.py"}


def test_import_and_cli_help_cannot_load_usd_or_native():
    script = """
import importlib.abc, sys
class BlockNative(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'pxr', 'isaacsim', 'omni'}:
            raise AssertionError('Native import during CPU preparation: ' + fullname)
sys.meta_path.insert(0, BlockNative())
from sim_data.native_capture_v5 import matched_parallax
try:
    matched_parallax.main(['--help'])
except SystemExit as exc:
    assert exc.code == 0
"""
    result = subprocess.run([sys.executable, "-B", "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
