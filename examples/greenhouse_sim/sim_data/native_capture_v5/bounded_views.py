"""Two-target, twelve-candidate diagnostic; frozen V4 preparation is input only.

plan/verify are CPU read-only except an explicitly requested new plan output.
capture starts native work ONLY when explicitly invoked, raw storage only.
No collector copy/patch: the pinned collect() and capture_jobs() consume the
checked lists generically. All original scene/physics/native gates remain.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np

from ..dataset_review import require, verify_bindings
from ..depth_preview import sha256
from ..native_budget import REFERENCE, TRIAL
from ..native_capture_v4 import prepare as v4
from ..native_multitarget_plan import assert_compatible
from ..native_view_plan import propose_specs
from ..native_view_pose import bounded_reference_root

SCHEMA = "greenhouse.native_bounded12_plan.v5.1"
PROFILE = "original6_plus_interior6_same_stage.v5.1"
# Strictly interior, six distinct physical base positions, not new render seeds.
INTERIOR_OFFSETS = ((.01, -.025), (.01, .025), (.02, -.0125),
                    (.02, .0125), (.03, -.025), (.03, .025))
ROOT = Path(__file__).resolve().parents[4]
_CODE_ROOT = Path(__file__).resolve().parent.parent
FROZEN = {
    "native_generated_views.py": "573054884fa48da986b2926815f6fd225b591500ac247dbd95d0a2fc974fc8b7",
    "native_multitarget_plan.py": "32169cd27e87869c4bfdf39773fee3f101b01caacd15f95db609433fe197ee67",
    "native_view_plan.py": "0a23f67f5cdeea2bddb94b79a757a2d7c527a77800d70db90c69034b1c4c6fbe",
    "native_view_pose.py": "6da9cdaa780691e138893bdf69e9bec51609dc9fa8935c4f21db74dcd20d0628",
    "native_capture_v4/prepare.py": "04a2dc01ab44471641de0b711f5ea593595f54b3aa0b2a0d6dc62998bab56e42",
}
_LOADED = {str(p): sha256(p) for p in (Path(__file__).resolve(), Path(__file__).with_name("__init__.py").resolve())}


def implementation_bindings():
    pins = {str((_CODE_ROOT / name).resolve()): value for name, value in FROZEN.items()}
    pins.update(_LOADED)
    verify_bindings(pins)
    return pins


def read_pinned(path, expected):
    """Hash and parse the same bytes; no inferred receipt approval."""
    path = Path(path).resolve(strict=True)
    raw = path.read_bytes()
    require(isinstance(expected, str) and hashlib.sha256(raw).hexdigest() == expected,
            "Pinned file changed: " + str(path))
    return json.loads(raw)


def _load_parent(prepared_path, prepared_sha256, schedule_path, schedule_sha256, *, replay_geometry=False):
    prepared_path = Path(prepared_path).resolve()
    schedule_path = Path(schedule_path).resolve()
    require(prepared_path.name == "prepared.json", "Explicit original V4 prepared.json required")
    receipt = read_pinned(prepared_path, prepared_sha256)
    schedule = read_pinned(schedule_path, schedule_sha256)
    job, attempt = v4._scheduled(schedule, receipt["job_id"], receipt["attempt_id"])
    checked = v4.verify_prepared(prepared_path.parent, schedule_path, job, attempt, receipt["seed"],
        max_targets=receipt["max_targets"], schedule_sha256=schedule_sha256)
    require(checked == receipt, "V4 receipt changed during handoff")
    parent_path = Path(receipt["plan_path"]).resolve()
    parent = read_pinned(parent_path, receipt["plan_sha256"])
    require(len(parent["target_cases"]) == 2 and parent["views_per_target"] == 6
            and parent["maximum_native_frames"] == 12, "Exactly two V4 targets with six proposals each required")
    bases = [read_pinned(c["base_pair_plan"], c["base_pair_plan_sha256"]) for c in parent["target_cases"]]
    require(parent["anchor_pair_plan"] == parent["target_cases"][0]["base_pair_plan"], "V4 anchor changed")
    for base in bases:
        assert_compatible(bases[0], base)
    if replay_geometry:
        # Same full parent-plan/catalogue replay that the frozen worker performs
        # after SimulationApp starts. Never execute this during CPU preflight.
        from ..native_multitarget_plan import check
        require(check(parent, replay_geometry=True) == bases[0], "Original V4 geometry replay differs")
    pins = {
        str(prepared_path): prepared_sha256, str(parent_path): receipt["plan_sha256"],
        str(prepared_path.parent / "prepare_request.json"): sha256(prepared_path.parent / "prepare_request.json"),
        str(schedule_path): schedule_sha256,
    }
    verify_bindings(pins)
    origin = dict(prepared_path=str(prepared_path), prepared_sha256=prepared_sha256,
        schedule_path=str(schedule_path), schedule_sha256=schedule_sha256,
        original_plan_path=str(parent_path), original_plan_sha256=receipt["plan_sha256"])
    return parent, bases, origin, pins, receipt["implementation_bindings"]


def _extend(case, base):
    prefix = deepcopy(case["views"])
    component = base["source_row"]["component_id"]
    reference = base["expected_robot_snapshot"]
    target = np.asarray(case["expected_nominal_world_m"], float)
    root = np.asarray(reference["robot_root_to_world_usd_row_vectors"], float)
    expected = propose_specs(root, target, case["base_pair_plan_sha256"], 6)
    for spec in expected:
        spec["candidate_id"] = component + "_" + spec["candidate_id"]
    require(prefix == expected, "Original six specs differ from the frozen deterministic policy")
    delta = root[3, :2] - target[:2]
    side = 1 if delta[0] > 0 else -1
    seed = hashlib.sha256((PROFILE + ":" + case["base_pair_plan_sha256"]).encode()).digest()
    rng = np.random.default_rng(int.from_bytes(seed[:8], "little"))
    extra = []
    for index, (outward, lateral) in enumerate(INTERIOR_OFFSETS, 7):
        dx, dy = delta[0] + side*outward, delta[1] + lateral
        spec = dict(deepcopy(prefix[0]), candidate_id=f"{component}_view_{index:03d}",
            root_x_m=float(target[0] + dx), y_offset_m=float(dy),
            desired_pixel_xy=[float(rng.uniform(.30, .70)*848), float(rng.uniform(.30, .65)*408)],
            base_target_radius_m=float(math.hypot(dx, dy)),
            base_displacement_from_reference_m=float(math.hypot(outward, lateral)),
            view_policy=PROFILE)
        extra.append(spec)
    all_specs = prefix + extra
    positions = []
    for spec in all_specs:
        matrix = bounded_reference_root(reference, spec, target)  # no USD import
        positions.append(matrix[:2, 3])
        u, v = spec["desired_pixel_xy"]
        require(.30*848 <= u <= .70*848 and .30*408 <= v <= .65*408
                and spec["framing_reference_resolution"] == [848, 408], "Framing bounds changed")
    require(all(np.linalg.norm(a-b) > 1e-9 for i, a in enumerate(positions) for b in positions[i+1:]),
            "Repeated physical base position")
    return all_specs


def _derive(parent, bases, origin, source_pins, v4_bindings, render_budget):
    require(render_budget in (REFERENCE, TRIAL), "Existing explicit render budget required")
    result = deepcopy(parent)
    result.update(schema=SCHEMA, experiment_profile=PROFILE, original_v4=deepcopy(origin),
        views_per_target=12, maximum_native_frames=24,
        experiment=dict(render_budget=render_budget, storage_backend="raw_frozen_native_collector",
            original_prefix_per_target=6, extra_per_target=6, maximum_targets=2,
            requested_native_frames_max=24, new_context_credit=False,
            source_cap_reset=False, retained_selection_required=True,
            comparison_scope="same_stage_prefix_vs_extra_not_cross_run_pixel_equivalence",
            capture_order="target_major_original6_then_extra6",
            initial_warmup_reset_between_blocks=False),
        candidate_blocks={})
    result["source_bindings"].update(source_pins)
    result["implementation_bindings"].update(v4_bindings)
    result["implementation_bindings"].update(implementation_bindings())
    for case, base in zip(result["target_cases"], bases, strict=True):
        case["views"] = _extend(case, base)
        for i, spec in enumerate(case["views"]):
            name = spec["candidate_id"]
            require(name not in result["candidate_blocks"], "Duplicate candidate ID")
            result["candidate_blocks"][name] = "prefix6" if i < 6 else "extra6"
    return result


def build(prepared_path, prepared_sha256, schedule_path, schedule_sha256, *, render_budget=REFERENCE):
    """Public CPU derivation: validate the original completed V4 receipt first."""
    implementation_bindings()
    data = _load_parent(prepared_path, prepared_sha256, schedule_path, schedule_sha256)
    result = _derive(*data, render_budget)
    verify_bindings(result["source_bindings"])
    return result


def check(plan, *, replay_geometry=False):
    """Rebuild and compare EVERY field; native entry also replays original geometry."""
    require(plan.get("schema") == SCHEMA and plan.get("experiment_profile") == PROFILE,
            "Explicit V5 bounded12 plan required, not a relabelled V4 plan")
    verify_bindings(plan["implementation_bindings"])
    verify_bindings(plan["source_bindings"])
    origin = plan["original_v4"]
    data = _load_parent(origin["prepared_path"], origin["prepared_sha256"],
        origin["schedule_path"], origin["schedule_sha256"], replay_geometry=replay_geometry)
    expected = _derive(*data, plan["experiment"]["render_budget"])
    require(plan == expected, "Changed V5 derivation, prefix, source identity or code coverage")
    return data[1][0]


def summarize(plan, result, *, retained_ids=None):
    """No image comparator or approval. Optional IDs come from an EXTERNAL gate.

    Timings are additive recorded work, not complete per-block wall clocks:
    rejected-pose/geometry decisions lack elapsed_seconds in the frozen collector.
    First-view warmup/cold-cache costs remain in the block that incurred them.
    """
    require(result.get("state") == "native_generated_multiview_pilot_complete_pending_review",
            "Complete collector result required")
    planned = {s["candidate_id"]: (c["target_id"], s) for c in plan["target_cases"] for s in c["views"]}
    groups = {k: dict(proposed=0, captured=0, strict=0, rejected=0, strict_ids=[],
        render_seconds=0., geometry_screen_seconds=0., captured_elapsed_seconds=0.,
        first56_views=0, externally_retained=None) for k in ("prefix6", "extra6")}
    seen, strict_ids = set(), set()
    for row in result["records"]:
        name = row["candidate_id"]
        require(name in planned and name not in seen, "Unexpected/duplicate capture decision")
        require((row["target_id"], row["requested_spec"]) == planned[name], "Decision differs from V5 plan")
        require(row["state"] in ("native_captured_pending_review", "rejected_pose",
                                "rejected_possible_geometry_overlap"), "Unknown capture state")
        seen.add(name)
        group = groups[plan["candidate_blocks"][name]]
        group["proposed"] += 1
        captured = row["state"] == "native_captured_pending_review"
        group["captured"] += captured
        group["rejected"] += not captured
        group["geometry_screen_seconds"] += row.get("geometry_screen_seconds", 0.)
        if captured:
            group["render_seconds"] += row["render_budget"]["render_seconds"]
            group["first56_views"] += row["render_budget"]["requested_subframes"] == 56
            group["captured_elapsed_seconds"] += row["elapsed_seconds"]
            if row.get("automatic_annotation_eligible") is True:
                group["strict"] += 1
                group["strict_ids"].append(name)
                strict_ids.add(name)
    require(seen == planned.keys() and sum(g["captured"] for g in groups.values()) == result["captured_frames"],
            "Incomplete V5 capture decisions")
    if retained_ids is not None:
        require(isinstance(retained_ids, (list, tuple)) and len(set(retained_ids)) == len(retained_ids)
                and set(retained_ids) <= strict_ids, "External retention must be a unique subset of strict IDs")
        for key, group in groups.items():
            group["externally_retained"] = sum(plan["candidate_blocks"][s] == key for s in retained_ids)
    return dict(schema="greenhouse.native_bounded12_summary.v5.1", experiment_profile=PROFILE,
        blocks=groups, collector_elapsed_seconds=result["elapsed_seconds"], setup_seconds=result["setup_seconds"],
        retention_state="not_supplied" if retained_ids is None else "external_selection_declared_not_replayed",
        strict_state="collector_declared_requires_native_audit_replay",
        global_dedup_and_caps_verified=False, training_approved=False, source_cap_reset=False,
        block_wall_times_complete=False, cold_start_costs_not_normalized=True)


def _destination(path, plan):
    """Only NEW diagnostic artifacts, disjoint from the pinned source trees."""
    path = Path(path).resolve()
    diagnostic_root = (ROOT / "data/sim_data/diagnostics").resolve()
    require(path != diagnostic_root and path.is_relative_to(diagnostic_root) and not path.exists(),
            "New diagnostic destination required")
    roots = [Path(plan["original_v4"]["prepared_path"]).parent]
    for case in plan["target_cases"]:
        base = read_pinned(case["base_pair_plan"], case["base_pair_plan_sha256"])
        roots.extend(Path(base[k]).resolve() for k in
                     ("source_capture", "variant_directory", "prerequisite_directory"))
        source = base["source_collection_plan"]
        package = read_pinned(source, plan["source_bindings"][source])["package"]
        roots.append(Path(package).resolve())
    for source in roots:
        require(not path.is_relative_to(source) and not source.is_relative_to(path),
                "Diagnostic destination overlaps pinned input")
    return path


def _write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def capture(plan_path, plan_sha256, output):
    """Public native entry; validation/admission precede SimulationApp import.

    Raw diagnostic only. No compact writer is selected or qualified here.
    One collect() invocation means no warmup/capture-budget reset between blocks.
    """
    started = time.perf_counter()
    plan_path = Path(plan_path).resolve()
    plan = read_pinned(plan_path, plan_sha256)
    check(plan, replay_geometry=False)
    output = _destination(output, plan)
    require(not plan_path.is_relative_to(output), "Capture output overlaps submitted plan")
    from sim_physics.host_memory import preflight
    from ..native_generated_pair import windows_worker_admission
    memory = preflight()
    require(memory["allowed"], "Native memory reserve unavailable; no override")
    admission = windows_worker_admission(None)
    # Import only after admission; identical frozen collector, no monkeypatch.
    from ..native_generated_views import collect
    budget = plan["experiment"]["render_budget"]
    output.mkdir(parents=True, exist_ok=False)
    _write_new(output / "request.json", dict(schema="greenhouse.native_bounded12_capture_request.v5.1",
        created_utc=datetime.now(timezone.utc).isoformat(), plan_path=str(plan_path),
        plan_sha256=plan_sha256, experiment_profile=PROFILE, original_v4=plan["original_v4"],
        v5_implementation_bindings=implementation_bindings(),
        storage_backend="raw_frozen_native_collector", host_memory_preflight=memory,
        process_admission=admission, training_started=False, automatic_retries=False,
        native_instance_backend="fast", render_budget_profile=budget,
        experimental_budget_override=budget == TRIAL, render_profile_experiment=False))
    app = None
    try:
        from isaacsim import SimulationApp
        app = SimulationApp(dict(headless=True, width=1696, height=816, multi_gpu=False,
            renderer="RaytracedLighting", sync_loads=False, disable_viewport_updates=True,
            extra_args=["--/app/settings/persistent=false"]))
        base = check(plan, replay_geometry=True)
        result = collect(app, output, plan, base, profile_render=False,
                         instance_backend="fast", render_budget=budget)
        verify_bindings(plan["implementation_bindings"])
        verify_bindings(plan["source_bindings"])
        require(sha256(plan_path) == plan_sha256, "Submitted V5 plan changed during capture")
        result.update(schema="greenhouse.native_bounded12_capture_result.v5.1",
            experiment_profile=PROFILE, original_v4=plan["original_v4"],
            v5_plan_sha256=plan_sha256, storage_backend="raw_frozen_native_collector",
            frozen_collector_sha256=FROZEN["native_generated_views.py"],
            adapter_elapsed_before_shutdown_seconds=time.perf_counter() - started)
        report = summarize(plan, result)
        _write_new(output / "experiment_summary.json", report)
        _write_new(output / "result.json", result)  # publish completion last
        return result
    except BaseException:
        _write_new(output / "failure.json", dict(state="native_bounded12_failed",
            experiment_profile=PROFILE, error=traceback.format_exc(), training_approved=False))
        raise
    finally:
        if app is not None:
            app.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("plan", help="CPU-only checked V4-to-V5 derivation")
    create.add_argument("--prepared", type=Path, required=True)
    create.add_argument("--prepared-sha256", required=True)
    create.add_argument("--schedule", type=Path, required=True)
    create.add_argument("--schedule-sha256", required=True)
    create.add_argument("--render-budget", choices=(REFERENCE, TRIAL), required=True)
    create.add_argument("--output", type=Path, required=True)
    for name in ("verify", "capture", "report"):
        command = commands.add_parser(name)
        command.add_argument("--plan", type=Path, required=True)
        command.add_argument("--plan-sha256", required=True)
        if name == "capture":
            command.add_argument("--output", type=Path, required=True)
        if name == "report":
            command.add_argument("--result", type=Path, required=True)
            command.add_argument("--result-sha256", required=True)
            command.add_argument("--retained-ids", type=Path)
            command.add_argument("--retained-ids-sha256")
    args = parser.parse_args(argv)
    if args.command == "plan":
        plan = build(args.prepared, args.prepared_sha256, args.schedule, args.schedule_sha256,
                     render_budget=args.render_budget)
        destination = _destination(args.output, plan)
        destination.parent.mkdir(parents=True, exist_ok=True)
        _write_new(destination, plan)
        print(json.dumps(dict(plan=str(destination), sha256=sha256(destination),
                              profile=PROFILE, targets=2, maximum_native_frames=24)))
    elif args.command == "capture":
        result = capture(args.plan, args.plan_sha256, args.output)
        print(json.dumps(dict(profile=PROFILE, captured=result["captured_frames"], training_approved=False)))
    else:
        plan = read_pinned(args.plan, args.plan_sha256)
        check(plan)
        if args.command == "verify":
            print(json.dumps(dict(profile=PROFILE, targets=2, maximum_native_frames=24,
                original_prefix_preserved=True, native_launched=False, full_geometry_replay="required_in_native_worker")))
        else:
            require(bool(args.retained_ids) == bool(args.retained_ids_sha256),
                    "External retained IDs require an explicit hash pin")
            retained = read_pinned(args.retained_ids, args.retained_ids_sha256) if args.retained_ids else None
            result = read_pinned(args.result, args.result_sha256)
            require(result.get("v5_plan_sha256") == args.plan_sha256
                    and result.get("experiment_profile") == PROFILE, "Result belongs to a different V5 plan")
            summary = summarize(plan, result, retained_ids=retained)
            if args.retained_ids:
                summary["external_retention_pin"] = dict(path=str(args.retained_ids.resolve()),
                                                         sha256=args.retained_ids_sha256)
            print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
