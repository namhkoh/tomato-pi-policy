"""Opt-in v4 CPU preparation and separately versioned launcher handoff.

Same CLI arguments as v3, plus optional --schedule-sha256 and --verify-only.
Only a NEW preparation output may be created; native capture is never launched.
The expensive schedule check explicitly uses reference_check.check_schedule.
Selection/destination/proof helpers and generation/native-plan gates are unchanged.

The short prepare flow and handoff checks are adapted from v3 prepare.py
SHA256 ca9dce6bf7f766190831d09097d89fe17438c6a77a4034604ce89b7aa4c05c80.
V4 receipts bind this module and its verifier/helpers, never claim v3 identity.
Native plan dictionaries retain their existing schema and implementation bindings.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from ..native_capture_v3 import prepare as legacy
from ..dataset_review import read_json, write_json, require, verify_bindings
from ..depth_preview import sha256
from . import reference_check as verifier
from .io import Reader

SCHEMA = "greenhouse.reference_bank_preparation.v4.1"
STATE = "generated_reference_bank_job_prepared_pending_native_capture"
MODULE = "sim_data.native_capture_v4.prepare"
_UPSTREAM = {str(Path(legacy.__file__).resolve()):
             "ca9dce6bf7f766190831d09097d89fe17438c6a77a4034604ce89b7aa4c05c80"}
_LOADED = {str(Path(__file__).resolve()): sha256(__file__)}

# Explicit imports, not patched globals or a call through legacy.prepare().
choose_references = legacy.choose_references
validate_destination = legacy.validate_destination
verify_fresh_anchor = legacy.verify_fresh_anchor


def implementation_bindings():
    bindings = {**verifier.implementation_bindings(), **_UPSTREAM, **_LOADED}
    verify_bindings(bindings)
    return bindings


def _read_pinned(path, expected=None):
    pin = sha256(path) if expected is None else expected
    reader = Reader()
    value = reader.json(path, pin)
    reader.finish()
    return value, pin


def _scheduled(schedule, job_id, attempt_id):
    jobs = [j for j in schedule["jobs"] if j["job_id"] == job_id]
    require(len(jobs) == 1, "Unknown or ambiguous scheduled job")
    job = jobs[0]
    attempts = [a for a in [job["anchor"], *job["fallback_anchors"]]
                if a["attempt_id"] == attempt_id]
    require(len(attempts) == 1, "Unknown or ambiguous scheduled anchor")
    return job, attempts[0]


def _generation_api():
    # Lazy: CLI help and the handoff must not import pip USD before SimulationApp.
    from ..generated_capture import prepare_plan, check_plan, verify_sensor_prerequisite
    from ..plant_variants import load_training_sources, training_envelope
    from ..procedural_petiole_v2 import generate, plan_change
    from ..native_multitarget_plan import build, check
    return SimpleNamespace(prepare_plan=prepare_plan, check_plan=check_plan,
        verify_sensor_prerequisite=verify_sensor_prerequisite,
        load_training_sources=load_training_sources, training_envelope=training_envelope,
        generate=generate, plan_change=plan_change, build=build, check=check)


def _check_handoff_plan(plan):
    from ..native_multitarget_plan import check
    return check(plan, replay_geometry=False)


def prepare(schedule_path, job_id, attempt_id, seed, output, *, max_targets=3,
            schedule_sha256=None):
    implementation = implementation_bindings()
    schedule_path = Path(schedule_path).resolve()
    schedule, schedule_pin = _read_pinned(schedule_path, schedule_sha256)
    verification_io = {}
    verifier.check_schedule(schedule, diagnostics=verification_io)
    job, attempt = _scheduled(schedule, job_id, attempt_id)
    bank_path = Path(schedule["source_reference_bank"])
    bank, bank_pin = _read_pinned(bank_path, schedule["source_reference_bank_sha256"])
    input_bindings = {str(schedule_path): schedule_pin, str(bank_path): bank_pin}
    references = choose_references(bank, job, attempt, seed, max_targets)
    output = validate_destination(output, bank, schedule)
    sensor_proof = verify_fresh_anchor(attempt)
    proof = Path(attempt["qualification_output"])

    api = _generation_api()
    frozen, sources = api.load_training_sources(job["source_collection_plan"])
    envelope = api.training_envelope(frozen, sources)
    source = sources[job["source_family"]]
    selected, exclusions = [], []
    for reference in references:
        component = reference["source_row"]["component_id"]
        try:
            api.plan_change(source, component, envelope, seed + len(selected)*104729)
        except ValueError as exc:
            if reference["id"] == attempt["id"]:
                raise ValueError("Exact qualified anchor cannot generate this seed: " + str(exc)) from exc
            exclusions.append(dict(target_id=reference["target_id"], reason=str(exc)))
            continue
        selected.append(reference)
    require(selected and selected[0]["id"] == attempt["id"], "Exact anchor must remain first")
    verify_bindings(implementation)
    verify_bindings(input_bindings)
    require(output.resolve() == output and not output.exists(),
            "Preparation destination changed or already exists")
    output.mkdir(parents=True)
    request = dict(schema=SCHEMA, preparation_module=MODULE,
        reference_verifier=verifier.VERSION, verification_io=verification_io,
        schedule=str(schedule_path), schedule_sha256=schedule_pin,
        job_id=job_id, attempt_id=attempt_id, seed=seed, max_targets=max_targets,
        reference_ids=[e["id"] for e in selected], source_collection_plan=job["source_collection_plan"],
        source_collection_plan_sha256=job["source_collection_plan_sha256"],
        compatibility_group=job["compatibility_group"], source_family=job["source_family"],
        training_approved=False, implementation_bindings=implementation,
        sensor_prerequisite_bindings=sensor_proof["bindings"])
    write_json(output / "prepare_request.json", request)
    variant = output / (job["source_family"] + "_cr_" + str(seed))
    api.generate(job["source_collection_plan"], job["source_family"],
                 [e["source_row"]["component_id"] for e in selected], seed, variant)
    bases = []
    for reference in selected:
        base = api.prepare_plan(reference["source_capture"], reference["source_sample"], variant, proof)
        api.check_plan(base)
        if reference["id"] == attempt["id"]:
            api.verify_sensor_prerequisite(base)
        path = output / (reference["source_row"]["component_id"] + "_base.json")
        write_json(path, base)
        bases.append(path)
    plan = api.build(bases[0], bases, 6)
    api.check(plan, replay_geometry=True)
    write_json(output / "plan.json", plan)
    verify_bindings(implementation)
    verify_bindings(input_bindings)
    verify_bindings(schedule["implementation_bindings"])
    verify_bindings(sensor_proof["bindings"])
    result = dict(state=STATE, **request, plan_path=str(output / "plan.json"),
        plan_sha256=sha256(output / "plan.json"), targets=[e["target_id"] for e in selected],
        excluded_proposals=exclusions, maximum_native_frames=plan["maximum_native_frames"],
        native_capture_started=False, source_cap_reset=False,
        physical_execution_approved=False, new_biological_families=0)
    write_json(output / "prepared.json", result)
    return result


def verify_prepared(output, schedule_path, job, attempt, seed, *, max_targets=3,
                    schedule_sha256=None):
    """Read-only v4 handoff; original plan/proof gates, not a new bank rebuild.

    Like the v3 handoff, checks full plan/source/proof bindings without USD replay.
    The unchanged native worker still performs full geometry replay after app init.
    The launcher must independently pin its schedule and call this before launch.
    """
    implementation = implementation_bindings()
    output, schedule_path = Path(output).resolve(), Path(schedule_path).resolve()
    result = read_json(output / "prepared.json")
    request = read_json(output / "prepare_request.json")
    for document in (request, result):
        require(document.get("schema") == SCHEMA and document.get("preparation_module") == MODULE
                and document.get("reference_verifier") == verifier.VERSION, "V4 preparation identity required")
        require(document.get("implementation_bindings") == implementation, "Preparation implementation changed")
    require(result.get("state") == STATE, "Preparation did not complete")
    require(all(result.get(k) == v for k, v in request.items()), "Preparation request/result changed")
    expected_pin = request["schedule_sha256"] if schedule_sha256 is None else schedule_sha256
    schedule, schedule_pin = _read_pinned(schedule_path, expected_pin)
    scheduled_job, scheduled_attempt = _scheduled(schedule, job["job_id"], attempt["attempt_id"])
    require(job == scheduled_job and attempt == scheduled_attempt, "Caller job/anchor differs from schedule")
    expected = dict(schedule=str(schedule_path), schedule_sha256=schedule_pin,
        job_id=job["job_id"], attempt_id=attempt["attempt_id"], seed=seed, max_targets=max_targets,
        source_collection_plan=job["source_collection_plan"],
        source_collection_plan_sha256=job["source_collection_plan_sha256"],
        compatibility_group=job["compatibility_group"], source_family=job["source_family"])
    require(all(result.get(k) == v for k, v in expected.items()), "Preparation differs from scheduled job")
    require(type(seed) is int and 0 <= seed < 2**32 and type(max_targets) is int
            and 1 <= max_targets <= 12, "Invalid seed or target bound")
    require(result.get("training_approved") is False and result.get("source_cap_reset") is False
            and result.get("native_capture_started") is False and result.get("physical_execution_approved") is False,
            "Preparation cannot grant approval")
    require(result.get("reference_ids") and result["reference_ids"][0] == attempt["id"],
            "Exact qualified anchor missing")
    require(result.get("plan_path") == str(output / "plan.json")
            and result.get("plan_sha256") == sha256(output / "plan.json"), "Prepared plan missing or changed")
    proof = verify_fresh_anchor(attempt)["bindings"]
    require(result.get("sensor_prerequisite_bindings") == proof, "Prepared proof belongs to another anchor")
    verify_bindings(proof)
    bank_path = Path(schedule["source_reference_bank"])
    bank, bank_pin = _read_pinned(bank_path, schedule["source_reference_bank_sha256"])
    entries = {e["id"]: e for e in bank["entries"]}
    plan = read_json(output / "plan.json")
    _check_handoff_plan(plan)
    cases, ids = plan["target_cases"], result["reference_ids"]
    require(1 <= len(cases) == len(ids) <= max_targets and len(set(ids)) == len(ids), "Prepared target bound changed")
    require(plan["prerequisite_bindings"] == proof and plan["source_family"] == job["source_family"]
            and plan["split"] == "train" and plan["resolution"] == [1696, 816], "Plan/proof/family mismatch")
    require(plan["anchor_pair_plan"] == cases[0]["base_pair_plan"] and plan["views_per_target"] == 6
            and plan["maximum_native_frames"] == result["maximum_native_frames"] == 6*len(cases),
            "Prepared anchor or view bound changed")
    source_targets = []
    for identity, case in zip(ids, cases, strict=True):
        require(identity in entries, "Prepared reference absent from pinned bank")
        reference = entries[identity]
        require(reference["compatibility_group"] == job["compatibility_group"], "Mixed reference groups")
        base_path = Path(case["base_pair_plan"]).resolve()
        require(base_path.parent == output and sha256(base_path) == case["base_pair_plan_sha256"],
                "Prepared base missing, changed or outside job")
        base = read_json(base_path)
        for key in ("source_capture", "source_sample", "source_collection_plan", "source_family", "source_row"):
            require(base[key] == reference[key], "Prepared base does not match reviewed reference: " + key)
        require(base["source_collection_plan"] == job["source_collection_plan"]
                and sha256(base["source_collection_plan"]) == job["source_collection_plan_sha256"]
                and Path(base["prerequisite_directory"]).resolve() == Path(attempt["qualification_output"]).resolve(),
                "Prepared base plan/proof changed")
        require(base["generated_row"]["target_id"] == case["target_id"]
                and base["generated_row"]["component_id"] == reference["source_row"]["component_id"]
                and case["conservative_view_cap_group"] == reference["target_id"] and len(case["views"]) == 6,
                "Prepared target identity or source cap changed")
        source_targets.append(reference["target_id"])
    require(result["targets"] == source_targets, "Prepared target inventory changed")
    verify_bindings({str(schedule_path): schedule_pin, str(bank_path): bank_pin})
    verify_bindings(implementation)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--schedule-sha256")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-targets", type=int, default=3)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    kwargs = dict(max_targets=args.max_targets, schedule_sha256=args.schedule_sha256)
    if args.verify_only:
        schedule, _ = _read_pinned(args.schedule, args.schedule_sha256)
        job, attempt = _scheduled(schedule, args.job_id, args.attempt_id)
        result = verify_prepared(args.output, args.schedule, job, attempt, args.seed, **kwargs)
        prefix = "REFERENCE_JOB_VERIFIED_V4"
    else:
        result = prepare(args.schedule, args.job_id, args.attempt_id, args.seed, args.output, **kwargs)
        prefix = "REFERENCE_JOB_PREPARED_V4"
    print(prefix, json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
