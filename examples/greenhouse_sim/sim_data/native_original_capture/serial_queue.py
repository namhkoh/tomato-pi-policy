"""Opt-in CPU supervisor for the frozen 16+23 original-native plans; no import launch.

CLI: python -B -m sim_data.native_original_capture.serial_queue
     --request ABS_JSON --request-sha256 SHA256

Request keys (all paths absolute):
  schema=SCHEMA, output, isaac_python, native_deps, max_planned_cases=39,
  pilot={capture_path,result_sha256,launcher_receipt_path,launcher_receipt_sha256},
  predecessor={request_path,request_sha256,
      supervisor={pid,creation_date,executable,command_line}},
  batches=[{plan_path,plan_sha256}, {plan_path,plan_sha256}].
The V5 input request identifies its output/scale roots and one-case pilot.
Create the request after the pilot has its pinned owned-exit receipt. V5 may
still be scaling: exact terminal identity AND supervisor disappearance are
required. CIM/read failures or reused PIDs never mean successful exit.

Gauss original_inventory_v2 contract (NOT accepted by the frozen V1 adapter):
  output/batch_NNN/capture/{request,result,automatic_audit}.json: unchanged worker
  output/batch_NNN/queue_000001.json: LAUNCH_STATE, PID, exact command, bindings
  output/batch_NNN/original_postexit_audit.json: unmodified full audit replay
  output/batch_NNN/original_launcher_receipt.json: RECEIPT_SCHEMA/RECEIPT_STATE,
    queue_request_path/sha256, batch_index, producer_module,
    producer_implementation_bindings, source_bindings, owned_worker
    {worker_pid,command,event_path,event_sha256}, plan_path/sha256, capture,
    request_sha256, result_sha256, exit_code=0, audit_path/sha256,
    training_approved=False, source_cap_reset=False.
Consumers must pin this actual producer, verify the queue request/batch and
event/receipt bindings, then independently replay the frozen original audit.
Producer declarations are not OS attestation or global dataset admission.

No queue prepare/check_plan, generated geometry, scene fork, warmup change,
source-verification bypass, retries, compaction, or automatic training approval.
Frozen worker source checks and post-exit replay remain mandatory.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

from . import contracts as oc
from . import audit
from .prepare import protected_roots
from ..native_capture_v4 import campaign
from ..native_capture_v5 import campaign_queue as v5
from ..native_dataset import native_process_guard as guard
from ..native_dataset import original_inventory as pilot_adapter
from sim_physics.host_memory import preflight

SCHEMA = "greenhouse.original_serial_queue_request.v1"
RECEIPT_SCHEMA = "greenhouse.original_serial_launcher_receipt.v1"
RECEIPT_STATE = "owned_original_batch_exited_and_reaudited"
LAUNCH_STATE = "original_batch_running"
RESULT_STATE = "original_serial_queue_complete_pending_global_admission"
PRODUCER_MODULE = "sim_data.native_original_capture.serial_queue"
STRICT = "accept_strict_automatic_annotation_candidate"
BATCH_SHAPE = (("seed73_full", 16), ("seed23_full", 23))
GIB = 2**30
_LOADED = oc.merge_bindings(pilot_adapter.implementation_bindings(),
    guard.implementation_bindings(), {str(Path(__file__).resolve()): oc.sha256(__file__)})


def implementation_bindings():
    """Exact loaded producer/helper code map, including this NEW queue."""
    oc.bind_all(_LOADED)
    return dict(_LOADED)


def _keys(value, expected, label):
    oc.require(isinstance(value, dict) and set(value) == set(expected), "Exact fields required: " + label)


def _path(value):
    oc.require(isinstance(value, str) and Path(value).is_absolute(), "Absolute path required")
    return Path(value).resolve()


def _sha(value):
    oc.require(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value), "Lowercase SHA256 required")
    return value


def _document(path, value):
    return oc.read_json(oc.pin(path, value))


def _flags(document):
    oc.require(document.get("training_approved") is False and document.get("source_cap_reset") is False,
               "No training approval or cap reset permitted")


def validate_request(request):
    """Read-only structural/plan-pin validation; never reconstruct source plans."""
    _keys(request, ("schema", "output", "isaac_python", "native_deps", "max_planned_cases",
                    "pilot", "predecessor", "batches"), "queue request")
    oc.require(request["schema"] == SCHEMA and type(request["max_planned_cases"]) is int
               and request["max_planned_cases"] == 39, "Explicit initial 39-case bound required")
    for name in ("output", "isaac_python", "native_deps"):
        _path(request[name])
    oc.require(_path(request["isaac_python"]).is_file() and _path(request["native_deps"]).is_dir(),
               "Missing native runtime")
    pilot = request["pilot"]
    _keys(pilot, ("capture_path", "result_sha256", "launcher_receipt_path", "launcher_receipt_sha256"), "pilot")
    for name in ("capture_path", "launcher_receipt_path"):
        _path(pilot[name])
    for name in ("result_sha256", "launcher_receipt_sha256"):
        _sha(pilot[name])
    pred = request["predecessor"]
    _keys(pred, ("request_path", "request_sha256", "supervisor"), "V5 predecessor")
    config = _document(_path(pred["request_path"]), _sha(pred["request_sha256"]))
    oc.require(config["schema"] == v5.SCHEMA, "Pinned V5 request required")
    for name in ("output", "scale_output", "original_plan"):
        _path(config[name])
    _sha(config["original_plan_sha256"])
    oc.require(_path(pilot["capture_path"]) == _path(config["output"]) / "original_capture",
               "Pilot must belong to this exact V5 predecessor")
    supervisor = pred["supervisor"]
    _keys(supervisor, ("pid", "creation_date", "executable", "command_line"), "V5 supervisor")
    oc.require(type(supervisor["pid"]) is int and supervisor["pid"] > 0
               and supervisor["pid"] != os.getpid(), "Distinct positive V5 supervisor PID required")
    executable = _path(supervisor["executable"])
    oc.require(executable.is_file() and executable.name.lower() in ("python.exe", "pythonw.exe"),
               "Exact CPU supervisor executable required")
    oc.require(guard._born({"CreationDate": supervisor["creation_date"]}) is not None,
               "Exact CIM supervisor creation date required")
    argv = guard._argv(supervisor["command_line"])
    oc.require(argv is not None and _path(argv[0]) == executable, "Exact V5 supervisor command required")
    args = argv[1:]
    while args and args[0] in ("-B", "-u"):
        args = args[1:]
    oc.require(args == ["-m", "sim_data.native_capture_v5.campaign_queue",
        "--request", str(_path(pred["request_path"])), "--request-sha256", pred["request_sha256"]],
        "Supervisor must execute this exact V5 request")
    batches = request["batches"]
    oc.require(isinstance(batches, list) and len(batches) == 2, "Exactly two ordered frozen batches required")
    plans, roots = [], [_path(config["output"]), _path(config["scale_output"]),
        _path(pred["request_path"]), _path(pilot["launcher_receipt_path"]).parent,
        _path(request["isaac_python"]).parent, _path(request["native_deps"])]
    prior_poses = set()
    for entry, (family, count) in zip(batches, BATCH_SHAPE, strict=True):
        _keys(entry, ("plan_path", "plan_sha256"), "batch")
        path = _path(entry["plan_path"])
        plan = _document(path, _sha(entry["plan_sha256"]))
        oc.require(plan["schema"] == oc.SCHEMA and plan["source_family"] == family
            and plan["split"] == "train" and plan["family_assignments"] == oc.FROZEN_SPLITS
            and plan["geometry_mode"] == "unmodified_original" and plan["admission"] == oc.ADMISSION
            and plan["scene_policy"] == oc.SCENE_POLICY
            and plan["stage_reuse"] == "one_original_donor_scene_one_native_product"
            and type(plan["sample_count_limit"]) is int and plan["sample_count_limit"] == count
            and len(plan["cases"]) == count, "Frozen original batch scope changed")
        oc.require(plan["implementation_bindings"] == oc.LOADED_IMPLEMENTATION,
                   "Frozen original implementation binding mismatch")
        for case in plan["cases"]:
            oc.require(case["pose_request"] == {"mode": "exact_prior", "native_pixel_xy": None}
                and case["pose_prior"]["prior_target_id"] == case["target_id"],
                "Initial queue accepts exact same-target priors only")
            identity = oc.digest(oc.canonical([family, case["expected_calibration"]]))
            oc.require(identity not in prior_poses, "Duplicate planned actual camera")
            prior_poses.add(identity)
        roots.extend([path.parent, *protected_roots(plan)])
        plans.append(plan)
    oc.require(len({_path(b["plan_path"]) for b in batches}) == 2, "Duplicate plan")
    oc.new_destination(request["output"], roots)
    return config, plans


def qualify_pilot(pilot, config):
    """Exactly one public V1 authentication/replay; no historical approval."""
    rows, summary, bindings = pilot_adapter.original_observed_rows(pilot_adapter.OriginalCapturePin(**pilot))
    oc.require(summary["planned_rows"] == 1 and len(rows) == 1 and summary["noncaptured_rows"] == 0
        and rows[0]["decision"] == STRICT and rows[0]["source_family"] == "seed73_full",
        "One fresh strict seed73 pilot required before expansion")
    provenance = rows[0]["provenance"]
    oc.require(_path(provenance["plan_path"]) == _path(config["original_plan"])
        and provenance["plan_sha256"] == config["original_plan_sha256"], "Pilot differs from V5 submitted plan")
    return dict(state="strict_original_pilot_replayed_once", pilot=deepcopy(pilot),
        plan_path=str(_path(config["original_plan"])), plan_sha256=config["original_plan_sha256"],
        decision=STRICT, verified_bindings=bindings, training_approved=False, source_cap_reset=False)


def supervisor_rows(pid):
    """Read only the requested PID; PowerShell/CIM failure never becomes absence."""
    oc.require(os.name == "nt" and type(pid) is int and pid > 0, "Windows supervisor PID required")
    command = ('[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false); '
        'ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_Process '
        f'-Filter "ProcessId = {pid}" -ErrorAction Stop | '
        'Select-Object ProcessId,CreationDate,ExecutablePath,CommandLine)')
    raw = subprocess.check_output(["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
        text=True, encoding="utf-8", errors="strict", timeout=20, creationflags=subprocess.CREATE_NO_WINDOW)
    rows = json.loads(raw.lstrip("\ufeff"))
    oc.require(isinstance(rows, list) and len(rows) <= 1, "Ambiguous supervisor CIM response")
    return rows


def supervisor_alive(supervisor, rows):
    """PID reuse or missing metadata fail closed instead of proving exit."""
    oc.require(isinstance(rows, list) and len(rows) <= 1, "Ambiguous supervisor rows")
    if not rows:
        return False
    row = rows[0]
    oc.require(type(row.get("ProcessId")) is int and row["ProcessId"] == supervisor["pid"]
        and row.get("CreationDate") == supervisor["creation_date"]
        and row.get("ExecutablePath") == supervisor["executable"]
        and row.get("CommandLine") == supervisor["command_line"], "V5 supervisor identity changed or PID reused")
    return True


def predecessor_completion(predecessor, config):
    """None means still active; never infer handoff from a temporary no-Kit gap."""
    oc.pin(predecessor["request_path"], predecessor["request_sha256"])
    root = _path(config["output"])
    oc.require(not any((root / n).exists() for n in ("failure.json", "superseded_idle_queue.json")),
               "V5 predecessor failed or was superseded")
    live = supervisor_alive(predecessor["supervisor"], supervisor_rows(predecessor["supervisor"]["pid"]))
    if not (root / "result.json").is_file():
        oc.require(live, "V5 supervisor exited without its exact terminal result")
        return None
    terminal_pin = oc.sha256(root / "result.json")
    terminal = _document(root / "result.json", terminal_pin)
    _flags(terminal)
    oc.require(terminal["state"] == "serial_probe_and_scale_complete_pending_admission"
        and terminal["request_sha256"] == predecessor["request_sha256"], "Wrong V5 terminal result")
    published_pin = oc.sha256(root / "request.json")
    published = _document(root / "request.json", published_pin)
    oc.require({k: published[k] for k in config} == config and published["training_approved"] is False,
               "V5 published request differs")
    oc.bind_all(published["implementation_bindings"])
    required = (Path(v5.__file__).resolve(), Path(campaign.__file__).resolve(), Path(guard.__file__).resolve())
    oc.require(all(published["implementation_bindings"].get(str(p)) == _LOADED[str(p)] for p in required),
               "Unexpected V5 producer implementation")
    scale_path = _path(config["scale_output"]) / "result.json"
    scale = _document(scale_path, terminal["scale_result_sha256"])
    _flags(scale)
    oc.require(isinstance(scale["records"], list) and bool(scale["records"])
        and scale["new_biological_families"] == 0, "Incomplete V4 continuation")
    events = sorted(root.glob("queue_[0-9][0-9][0-9][0-9][0-9][0-9].json"))
    oc.require(bool(events), "Missing V5 supervisor event")
    event_pin = oc.sha256(events[-1])
    event = _document(events[-1], event_pin)
    _flags(event)
    oc.require(event["state"] == "scale_coordinator_running_in_same_cpu_process"
        and event["worker_pid"] == predecessor["supervisor"]["pid"], "Wrong V5 terminal supervisor event")
    oc.require(all(event["bindings"].get(p) == h for p, h in published["implementation_bindings"].items()),
               "V5 supervisor event lost original producer/input pins")
    oc.bind_all(event["bindings"])
    pins = {str(root / "request.json"): published_pin, str(root / "result.json"): terminal_pin,
            str(scale_path): terminal["scale_result_sha256"], str(events[-1]): event_pin}
    oc.bind_all(pins)
    if live:
        return None
    return dict(state="exact_v5_terminal_and_supervisor_absent", supervisor=deepcopy(predecessor["supervisor"]),
        process_exit_evidence="exact_PID_absent_from_successful_CIM_query_not_OS_exit_code",
        bindings=pins,
        training_approved=False, source_cap_reset=False)


def resource_evidence(output):
    """Native guard + checked physical/commit reserve + destination-volume disk."""
    memory = preflight()
    report = guard.process_inventory()
    free = shutil.disk_usage(output).free
    oc.require(isinstance(report, dict) and isinstance(report.get("blockers"), list)
        and isinstance(report.get("classifications"), list) and bool(report["classifications"])
        and report.get("no_blockers_observed") is (not report["blockers"]), "Invalid process inventory")
    oc.require(memory.get("checked") is True and type(memory.get("allowed")) is bool
        and type(memory.get("commit_headroom_bytes")) is int, "Unchecked/malformed memory reserve")
    allowed = memory["allowed"] and memory["commit_headroom_bytes"] >= 20 * GIB and free >= 60 * GIB and not report["blockers"]
    return dict(allowed=bool(allowed), memory=memory, disk_free_bytes=free, process_inventory=report,
                exclusive_launch_guaranteed=False)


def batch_command(isaac_python, entry, capture):
    return [str(_path(isaac_python)), "-m", "sim_data.native_original_capture.collector",
        "--plan", str(_path(entry["plan_path"])), "--plan-sha256", entry["plan_sha256"],
        "--output", str(_path(str(capture)))]


def complete_batch(capture, entry):
    """Exact submitted plan/request/result followed by one full post-exit replay."""
    capture = _path(str(capture))
    before = v5.original_completion(capture, entry["plan_path"], entry["plan_sha256"])
    review = audit.audit_capture(capture, result_sha256=before["result_sha256"])
    oc.require(v5.original_completion(capture, entry["plan_path"], entry["plan_sha256"]) == before,
               "Original completion changed during post-exit replay")
    return before, review


def run(request_path, request_sha256):
    """Explicit Windows execution only; tests substitute native/process boundaries."""
    oc.require(os.name == "nt", "Explicit Windows CPU coordinator required")
    oc.require(not any(n.split(".")[0] in ("isaacsim", "omni", "carb") for n in sys.modules),
               "CPU coordinator must not have loaded native runtime")
    request_path = oc.pin(_path(str(request_path)), request_sha256)
    request = oc.read_json(request_path)
    config, plans = validate_request(request)
    output = _path(request["output"])
    oc.new_destination(output, [request_path])
    producer = implementation_bindings()
    bindings = oc.merge_bindings(producer, {str(request_path): request_sha256},
        {str(_path(b["plan_path"])): b["plan_sha256"] for b in request["batches"]},
        {str(_path(request["predecessor"]["request_path"])): request["predecessor"]["request_sha256"]},
        {str(_path(request["pilot"]["launcher_receipt_path"])): request["pilot"]["launcher_receipt_sha256"],
         str(_path(request["pilot"]["capture_path"]) / "result.json"): request["pilot"]["result_sha256"]})
    output.mkdir(parents=True, exist_ok=False)
    queue_request_path = output / "request.json"
    oc.write_new(queue_request_path, dict(request, producer_module=PRODUCER_MODULE,
        producer_implementation_bindings=producer, training_approved=False, source_cap_reset=False))
    queue_pin = oc.sha256(queue_request_path)
    bindings[str(queue_request_path)] = queue_pin
    counter = 0
    def status(state, **extra):
        nonlocal counter
        counter += 1
        oc.write_new(output / f"queue_{counter:06d}.json", dict(state=state,
            created_utc=datetime.now(timezone.utc).isoformat(), training_approved=False,
            source_cap_reset=False, **extra))
    try:
        qualified = qualify_pilot(request["pilot"], config)
        qualification_path = output / "pilot_qualification.json"
        oc.write_new(qualification_path, qualified)
        # The expensive pilot replay is once. Poll only immutable evidence/code pins.
        bindings[str(qualification_path)] = oc.sha256(qualification_path)
        while True:
            oc.bind_all(bindings)
            predecessor = predecessor_completion(request["predecessor"], config)
            if predecessor is not None:
                break
            status("waiting_for_exact_v5_terminal_and_supervisor_exit")
            time.sleep(30)
        predecessor_path = output / "predecessor_completion.json"
        oc.write_new(predecessor_path, predecessor)
        bindings = oc.merge_bindings(bindings, predecessor["bindings"],
                                    {str(predecessor_path): oc.sha256(predecessor_path)})
        _, environment = v5.worker_environments(request["native_deps"])
        records = []
        for index, (entry, plan) in enumerate(zip(request["batches"], plans, strict=True), 1):
            folder = output / f"batch_{index:03d}"
            oc.new_destination(folder, [*protected_roots(plan), _path(entry["plan_path"]).parent])
            folder.mkdir()
            capture = folder / "capture"
            command = batch_command(request["isaac_python"], entry, capture)
            launched = {}
            def reserve():
                while True:
                    oc.bind_all(bindings)
                    oc.require(not supervisor_alive(request["predecessor"]["supervisor"],
                        supervisor_rows(request["predecessor"]["supervisor"]["pid"])),
                        "V5 supervisor reappeared before batch")
                    evidence = resource_evidence(output)
                    if evidence["allowed"]:
                        return
                    status("waiting_for_safe_native_resources", batch_index=index, **evidence)
                    time.sleep(30)
            def launch_check():
                oc.pin(entry["plan_path"], entry["plan_sha256"])
                oc.require(not supervisor_alive(request["predecessor"]["supervisor"],
                    supervisor_rows(request["predecessor"]["supervisor"]["pid"])), "V5 supervisor active")
                evidence = resource_evidence(output)
                oc.require(evidence["allowed"], "Native resources changed before launch")
                launched["resource_evidence"] = evidence
            def announce(pid):
                oc.require(type(pid) is int and pid > 0, "Owned child PID required")
                event_path = folder / "queue_000001.json"
                oc.write_new(event_path, dict(state=LAUNCH_STATE, worker_pid=pid, command=command,
                    batch_index=index, producer_module=PRODUCER_MODULE, bindings=bindings,
                    resource_evidence=launched["resource_evidence"],
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    training_approved=False, source_cap_reset=False))
                launched.update(worker_pid=pid, command=command, event_path=str(event_path),
                                event_sha256=oc.sha256(event_path))
            code = campaign.run_checked(command, folder / "native.log", bindings=bindings,
                environment=environment, native=True, reserve=reserve, launch_check=launch_check, announce=announce)
            oc.require(type(code) is int and code == 0, "Original worker failed; no remaining batches launched")
            completion, review = complete_batch(capture, entry)
            oc.bind_all(bindings)
            oc.pin(launched["event_path"], launched["event_sha256"])
            audit_path = folder / "original_postexit_audit.json"
            oc.write_new(audit_path, review)
            receipt_path = folder / "original_launcher_receipt.json"
            receipt = dict(schema=RECEIPT_SCHEMA, state=RECEIPT_STATE, exit_code=code, batch_index=index,
                producer_module=PRODUCER_MODULE, producer_implementation_bindings=producer,
                queue_request_path=str(queue_request_path), queue_request_sha256=queue_pin,
                plan_path=str(_path(entry["plan_path"])), plan_sha256=entry["plan_sha256"],
                capture=str(capture), **completion,
                owned_worker={k: launched[k] for k in ("worker_pid", "command", "event_path", "event_sha256")},
                audit_path=str(audit_path), audit_sha256=oc.sha256(audit_path),
                source_bindings=bindings, training_approved=False, source_cap_reset=False)
            oc.write_new(receipt_path, receipt)
            records.append(dict(batch_index=index, family=plan["source_family"], planned_cases=len(plan["cases"]),
                capture=str(capture), result_sha256=completion["result_sha256"], audit_counts=review["counts"],
                launcher_receipt_path=str(receipt_path), launcher_receipt_sha256=oc.sha256(receipt_path)))
            bindings = oc.merge_bindings(bindings, {str(receipt_path): oc.sha256(receipt_path),
                str(audit_path): oc.sha256(audit_path), launched["event_path"]: launched["event_sha256"],
                str(capture / "request.json"): completion["request_sha256"],
                str(capture / "result.json"): completion["result_sha256"]})
            status("original_batch_complete_pending_global_admission", **records[-1])
        oc.bind_all(bindings)
        oc.write_new(output / "result.json", dict(schema=SCHEMA, state=RESULT_STATE,
            queue_request_path=str(queue_request_path), queue_request_sha256=queue_pin,
            producer_module=PRODUCER_MODULE, producer_implementation_bindings=producer,
            planned_cases=39, records=records, training_approved=False, source_cap_reset=False))
    except BaseException:
        oc.write_new(output / "failure.json", dict(state="original_serial_queue_failed_stop_first",
            traceback=traceback.format_exc(), training_approved=False, source_cap_reset=False))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True)
    parser.add_argument("--request-sha256", required=True)
    args = parser.parse_args(argv)
    run(args.request, args.request_sha256)


if __name__ == "__main__":
    main()
