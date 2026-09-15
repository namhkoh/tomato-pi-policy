"""Focused CPU tests; all temporary files stay inside this NEW namespace."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

from . import bounded_views as v5
from .. import native_multitarget_plan as multi
from ..native_view_plan import propose_specs
from ..depth_preview import sha256


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.fixture
def case(monkeypatch):
    with tempfile.TemporaryDirectory(prefix=".cpu-test-", dir=Path(__file__).resolve().parent) as folder:
        root = Path(folder).resolve()
        monkeypatch.setattr(v5, "ROOT", root)
        prepared_root = root / "prepared"
        source_plan = write(root / "source_plan.json", dict(
            package=str(root / "data/sim_data/diagnostics/source_package")))
        bases, cases = [], []
        for component in ("SubStem_41", "SubStem_42"):
            matrix = np.eye(4)
            matrix[3, :3] = [.8, 1., 0.]
            reference = dict(robot_root_to_world_usd_row_vectors=matrix.tolist(),
                camera_to_head_column_vectors=np.eye(4).tolist(), visual_bound_screen={"passed": True})
            base = dict(source_family="original", split="train", split_group="original",
                variant_directory=str(root / "variant"), source_capture=str(root / "source"),
                prerequisite_directory=str(root / "proof"), source_collection_plan=str(source_plan),
                original_variant={"variant_id": "original"}, expected_scene_counts={"plants": 2},
                expected_calibration={k: "unchanged" for k in multi.OPTICS},
                expected_robot_snapshot=reference,
                source_row=dict(component_id=component, target_id="original/" + component),
                generated_row=dict(component_id=component, target_id="generated/" + component),
                conservative_view_cap_group="original/" + component)
            path = write(prepared_root / (component + "_base.json"), base)
            target = [0., 1., .8]
            specs = propose_specs(matrix, target, sha256(path), 6)
            for spec in specs:
                spec["candidate_id"] = component + "_" + spec["candidate_id"]
            cases.append(dict(base_pair_plan=str(path), base_pair_plan_sha256=sha256(path),
                target_id=base["generated_row"]["target_id"], expected_nominal_world_m=target,
                views=specs, conservative_view_cap_group=base["conservative_view_cap_group"]))
            bases.append(base)
        parent = dict(schema=multi.SCHEMA, target_cases=cases, anchor_pair_plan=cases[0]["base_pair_plan"],
            views_per_target=6, maximum_native_frames=12, source_family="original", split="train",
            resolution=[1696, 816], requested_render_subframes_per_view=56, source_cap_reset=False,
            training_approved=False, physical_motion_commanded=False, hidden_cut_coordinates_executable=False,
            source_bindings={str(source_plan): sha256(source_plan),
                **{c["base_pair_plan"]: c["base_pair_plan_sha256"] for c in cases}},
            prerequisite_bindings={}, implementation_bindings={})
        plan_path = write(prepared_root / "plan.json", parent)
        job = dict(job_id="job1", anchor={"attempt_id": "anchor1"}, fallback_anchors=[])
        schedule_path = write(root / "schedule.json", {"jobs": [job]})
        receipt = dict(schema=v5.v4.SCHEMA, state=v5.v4.STATE,
            job_id="job1", attempt_id="anchor1", seed=3010000, max_targets=12,
            plan_path=str(plan_path), plan_sha256=sha256(plan_path), implementation_bindings={})
        prepared = write(prepared_root / "prepared.json", receipt)
        write(prepared_root / "prepare_request.json", {})
        verified = []
        def verify(output, schedule, actual_job, attempt, seed, **kwargs):
            assert output == prepared_root and schedule == schedule_path
            assert actual_job == job and attempt == job["anchor"] and seed == 3010000
            assert kwargs == dict(max_targets=12, schedule_sha256=sha256(schedule_path))
            verified.append(True)
            return deepcopy(receipt)
        monkeypatch.setattr(v5.v4, "verify_prepared", verify)
        args = (prepared, sha256(prepared), schedule_path, sha256(schedule_path))
        yield SimpleNamespace(root=root, parent=parent, bases=bases, args=args, verified=verified)


def build(case):
    return v5.build(*case.args, render_budget=v5.TRIAL)


def test_public_derivation_prefix_unique_interior_and_frozen_jobs(case):
    before = deepcopy(case.parent)
    plan = build(case)
    assert case.verified == [True]
    assert plan["schema"] != multi.SCHEMA and plan["experiment_profile"] == v5.PROFILE
    assert plan["maximum_native_frames"] == 24 and plan["views_per_target"] == 12
    assert plan["experiment"]["render_budget"] == v5.TRIAL
    assert not plan["source_cap_reset"] and not plan["experiment"]["new_context_credit"]
    assert case.parent == before
    for original, derived, base in zip(case.parent["target_cases"], plan["target_cases"], case.bases):
        assert derived["views"][:6] == original["views"]
        assert derived["conservative_view_cap_group"] == original["conservative_view_cap_group"]
        xy = {(s["root_x_m"], s["y_offset_m"]) for s in derived["views"]}
        assert len(xy) == 12
        ref_x, ref_y = np.asarray(base["expected_robot_snapshot"]["robot_root_to_world_usd_row_vectors"])[3, :2]
        for spec, (out, lat) in zip(derived["views"][6:], v5.INTERIOR_OFFSETS):
            assert spec["root_x_m"] - ref_x == pytest.approx(out)
            assert spec["y_offset_m"] + derived["expected_nominal_world_m"][1] - ref_y == pytest.approx(lat)
            assert 0 < out < .04 and abs(lat) < .04
            assert spec["root_yaw_degrees"] == original["views"][0]["root_yaw_degrees"]
    assert v5.check(plan) == case.bases[0]
    jobs = multi.capture_jobs(plan, case.bases[0])
    assert len(jobs) == 24
    assert [j[2]["candidate_id"] for j in jobs] == list(plan["candidate_blocks"])


def test_opposite_aisle_and_determinism(case):
    base = deepcopy(case.bases[0])
    base["expected_robot_snapshot"]["robot_root_to_world_usd_row_vectors"][3][0] = -.8
    c = deepcopy(case.parent["target_cases"][0])
    specs = propose_specs(base["expected_robot_snapshot"]["robot_root_to_world_usd_row_vectors"],
                          c["expected_nominal_world_m"], c["base_pair_plan_sha256"], 6)
    for spec in specs:
        spec["candidate_id"] = "SubStem_41_" + spec["candidate_id"]
    c["views"] = specs
    a = v5._extend(c, base)
    assert a == v5._extend(c, base) and a[:6] == specs
    assert all(s["root_x_m"] < -.8 and s["opposite_aisle"] for s in a[6:])


@pytest.mark.parametrize("change", ["prefix", "extra", "duplicate", "count", "target",
    "cap", "split", "approval", "budget", "profile", "code_coverage", "source_pin", "origin"])
def test_whole_derivation_tamper_rejected(case, change):
    p = build(case)
    if change == "prefix": p["target_cases"][0]["views"][0]["desired_pixel_xy"][0] += 1
    elif change == "extra": p["target_cases"][0]["views"][6]["root_x_m"] += .1
    elif change == "duplicate": p["target_cases"][0]["views"][6] = deepcopy(p["target_cases"][0]["views"][0])
    elif change == "count": p["maximum_native_frames"] = 25
    elif change == "target": p["target_cases"][0]["target_id"] = "different"
    elif change == "cap": p["target_cases"][0]["conservative_view_cap_group"] = "reset"
    elif change == "split": p["split"] = "test"
    elif change == "approval": p["training_approved"] = True
    elif change == "budget": p["experiment"]["render_budget"] = "one_subframe"
    elif change == "profile": p["schema"] = multi.SCHEMA
    elif change == "code_coverage": p["implementation_bindings"] = {}
    elif change == "source_pin": p["source_bindings"] = {}
    elif change == "origin": p["original_v4"]["prepared_sha256"] = "0"*64
    with pytest.raises(ValueError):
        v5.check(p)


def test_parent_receipt_and_schedule_hashes_required(case):
    for index in (1, 3):
        args = list(case.args)
        args[index] = "0"*64
        with pytest.raises(ValueError, match="Pinned file"):
            v5.build(*args)
    assert not case.verified


def test_v4_rejection_cannot_be_bypassed(case, monkeypatch):
    def reject(*args, **kwargs):
        raise ValueError("Original V4 source lineage rejected")
    monkeypatch.setattr(v5.v4, "verify_prepared", reject)
    with pytest.raises(ValueError, match="Original V4"):
        build(case)


def test_original_geometry_replay_is_deferred_and_checks_parent(case, monkeypatch):
    p = build(case)
    calls = []
    def replay(parent, *, replay_geometry):
        assert parent == case.parent and replay_geometry is True
        calls.append(True)
        return deepcopy(case.bases[0])
    monkeypatch.setattr(multi, "check", replay)
    v5.check(p)
    assert calls == []
    v5.check(p, replay_geometry=True)
    assert calls == [True]


def synthetic_result(plan):
    rows = []
    for case in plan["target_cases"]:
        for i, spec in enumerate(case["views"]):
            captured = i % 3 != 2
            rows.append(dict(candidate_id=spec["candidate_id"], target_id=case["target_id"],
                requested_spec=deepcopy(spec),
                state="native_captured_pending_review" if captured else "rejected_possible_geometry_overlap",
                automatic_annotation_eligible=captured and i % 2 == 0,
                geometry_screen_seconds=.6, elapsed_seconds=6.0 if captured else 0.,
                render_budget=dict(render_seconds=4.0, requested_subframes=8)))
    return dict(state="native_generated_multiview_pilot_complete_pending_review",
        records=rows, captured_frames=sum(r["state"] == "native_captured_pending_review" for r in rows),
        elapsed_seconds=160., setup_seconds=30., training_approved=False)


def test_summary_keeps_retention_unknown_until_external_selection(case):
    plan = build(case)
    result = synthetic_result(plan)
    report = v5.summarize(plan, result)
    assert report["retention_state"] == "not_supplied"
    assert report["strict_state"] == "collector_declared_requires_native_audit_replay"
    assert all(b["proposed"] == 12 and b["externally_retained"] is None for b in report["blocks"].values())
    ids = report["blocks"]["extra6"]["strict_ids"][:1]
    report = v5.summarize(plan, result, retained_ids=ids)
    assert report["blocks"]["extra6"]["externally_retained"] == 1
    assert not report["global_dedup_and_caps_verified"] and not report["source_cap_reset"]
    with pytest.raises(ValueError):
        v5.summarize(plan, result, retained_ids=["foreign"])
    with pytest.raises(ValueError):
        v5.summarize(plan, result, retained_ids=ids*2)
    result["records"].pop()
    with pytest.raises(ValueError):
        v5.summarize(plan, result)


def test_cli_plan_verify_create_only_no_native(case, capsys):
    out = case.root / "data/sim_data/diagnostics/v5/plan.json"
    args = ["plan", "--prepared", str(case.args[0]), "--prepared-sha256", case.args[1],
        "--schedule", str(case.args[2]), "--schedule-sha256", case.args[3],
        "--render-budget", v5.TRIAL, "--output", str(out)]
    v5.main(args)
    original = out.read_bytes()
    v5.main(["verify", "--plan", str(out), "--plan-sha256", sha256(out)])
    assert '"native_launched": false' in capsys.readouterr().out
    with pytest.raises(ValueError):
        v5.main(args)
    assert original == out.read_bytes()
    assert "isaacsim" not in sys.modules and "omni.replicator.core" not in sys.modules


@pytest.mark.parametrize("failure", [None, "geometry", "collector"])
def test_capture_entry_only_calls_frozen_collector_once_after_checks(case, monkeypatch, failure):
    from .. import native_generated_views, native_generated_pair
    from sim_physics import host_memory
    plan = build(case)
    plan_path = write(case.root / "submitted_v5.json", plan)
    events = []
    monkeypatch.setattr(host_memory, "preflight", lambda: {"allowed": True})
    monkeypatch.setattr(native_generated_pair, "windows_worker_admission", lambda _:
                        events.append("admission") or {"fixture": True})
    class App:
        def __init__(self, settings):
            assert settings["width"] == 1696 and settings["height"] == 816
            assert settings["disable_viewport_updates"] and settings["headless"]
            events.append("app")
        def close(self):
            events.append("close")
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(SimulationApp=App))
    def replay(parent, *, replay_geometry):
        events.append("replay")
        assert parent == case.parent
        if failure == "geometry": raise ValueError("synthetic geometry failure")
        return case.bases[0]
    monkeypatch.setattr(multi, "check", replay)
    def collect(app, output, submitted, base, **kwargs):
        events.append("collect")
        assert kwargs == dict(profile_render=False, instance_backend="fast", render_budget=v5.TRIAL)
        assert submitted == plan and base == case.bases[0]
        if failure == "collector": raise ValueError("synthetic collector failure")
        return synthetic_result(submitted)
    monkeypatch.setattr(native_generated_views, "collect", collect)
    output = case.root / "data/sim_data/diagnostics/capture"
    if failure:
        with pytest.raises(ValueError, match="synthetic"):
            v5.capture(plan_path, sha256(plan_path), output)
        assert (output / "failure.json").exists() and not (output / "result.json").exists()
    else:
        result = v5.capture(plan_path, sha256(plan_path), output)
        assert result["experiment_profile"] == v5.PROFILE
        assert (output / "experiment_summary.json").exists()
    assert events[0:2] == ["admission", "app"] and events[-1] == "close"
    assert events.count("collect") == (failure != "geometry")
    assert events.index("replay") > events.index("app")
    assert not list(output.rglob("*.png")) and not list(output.rglob("*.npy"))


def test_capture_bad_plan_pin_rejected_before_app(case, monkeypatch):
    plan = build(case)
    path = write(case.root / "submitted.json", plan)
    monkeypatch.setitem(sys.modules, "isaacsim", SimpleNamespace(
        SimulationApp=lambda *a: pytest.fail("native startup")))
    with pytest.raises(ValueError, match="Pinned file"):
        v5.capture(path, "0"*64, case.root / "data/sim_data/diagnostics/never")
    assert not (case.root / "data").exists()


def test_source_package_destination_rejected(case):
    plan = build(case)
    with pytest.raises(ValueError, match="overlaps pinned input"):
        v5._destination(case.root / "data/sim_data/diagnostics/source_package/new", plan)


def test_only_explicit_v5_runtime_files_are_pinned(case):
    plan = build(case)
    namespace = Path(v5.__file__).resolve().parent
    pinned = {Path(p).name for p in plan["implementation_bindings"] if Path(p).parent == namespace}
    assert pinned == {"bounded_views.py", "__init__.py"}
