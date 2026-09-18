"""Own exactly one two-morphology native diagnostic; no retry or dataset export."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import os
import subprocess
import sys
import time

from .dataset_review import read_json, require, verify_bindings
from .depth_preview import sha256
from .native848_bulk_io_v1 import save_json
from .native_generated_reference.owner_v1 import owner_lock, resources, _run_child
from .native848_task_telemetry_optout_v1 import child_environment
from .native_original_capture import serial_queue as launch_api

SCHEMA = "greenhouse.morphology_preview_owned_run.v1"
ROOT = Path(__file__).resolve().parents[3]


def run(request_path, request_sha256, output):
    from . import native848_morphology_diagnostic_v1 as worker
    request_path = Path(request_path).resolve()
    output = Path(output).resolve()
    require(sha256(request_path) == request_sha256, "Diagnostic request changed")
    request = read_json(request_path)
    worker.require_request_shape(request)
    require(not output.exists() and output.parent.is_dir(), "New output under an existing directory required")
    require(not request_path.is_relative_to(output), "Output cannot contain its input request")
    require(not any(name.split(".")[0] in ("isaacsim", "omni", "carb", "pxr") for name in sys.modules),
            "CPU owner must not load a native ABI")
    verify_bindings(request["implementation_bindings"])
    args = argparse.Namespace(request=request_path, request_sha256=request_sha256, output=output/"capture")
    command = worker.native_command(args)
    require(command[:5] == [str(Path("D:/isaac-sim-6.0.1/python.bat")), "-B", "-u", "-m",
                           "sim_data.native848_morphology_diagnostic_v1"], "Unexpected worker command")
    import inspect
    from .native_generated_reference import owner_v1 as owner_helpers
    helpers = [Path(owner_helpers.__file__), Path(inspect.getfile(launch_api.v5.worker_environments))]
    pins = {**{str(p.resolve()):sha256(p) for p in helpers}, str(request_path):request_sha256, str(Path(__file__).resolve()):sha256(__file__),
            **request["implementation_bindings"]}
    started = time.perf_counter()
    with owner_lock():
        evidence = resources(output.parent)
        output.mkdir(exist_ok=False)
        save_json(output/"owner_started.json", dict(schema=SCHEMA, created_utc=datetime.now(timezone.utc).isoformat(),
            command=command, owner_pid=os.getpid(), resource_evidence=evidence, source_bindings=pins,
            native_worker_count=1, max_frames=2, automatic_retries=False, training_approved=False,
            accepted_training_increment=0, prior_collection_restart=False))
        _, environment = launch_api.v5.worker_environments(ROOT/"data/sim_data/native_capture_deps_py312_v1")
        environment = child_environment(environment)
        try:
            exit_code = _run_child(command, output, environment)
            save_json(output/"owned_exit.json", dict(schema=SCHEMA, returncode=exit_code,
                method="subprocess_wait_on_owned_process", elapsed_seconds=time.perf_counter()-started,
                command=command, owned_worker_sha256=sha256(output/"owned_worker.json")))
            require(exit_code == 0, "Native diagnostic exited nonzero; preserve failure, no retry")
            from .native_dataset.reference_bridge_owner_v1 import current_owner_evidence
            from .native_dataset.native_process_guard import native_processes
            current_owner_evidence()
            require(not native_processes(), "Native processes remain after owned exit; no completion")
            capture = output/"capture"
            require(not (capture/"failure.json").exists(), "Native diagnostic failure marker present")
            result = read_json(capture/"result.json")
            require(result.get("training_approved") is False and result.get("accepted_training_increment") == 0,
                    "Native diagnostic cannot approve dataset samples")
            require(result.get("schema") == worker.RESULT_SCHEMA and result.get("request_sha256") == request_sha256,
                    "Saved result belongs to another request/schema")
            require(result.get("max_frames") == 2 and result.get("frames_per_variant") == 1,
                    "Wrong diagnostic frame limits")
            expected_ids = [read_json(v["qualification"]["path"])["variant_id"] for v in request["variants"]]
            frames = result.get("frames", [])
            require(len(frames) == 2 and [f["morphology_id"] for f in frames] == expected_ids,
                    "Saved morphology schedule differs")
            observation_paths = set()
            for frame in frames:
                spec = frame["observation"]
                observation = Path(spec["path"]).resolve()
                require(observation.is_relative_to(capture.resolve()) and observation not in observation_paths
                        and sha256(observation) == spec["sha256"], "Repeated, external or changed observation")
                observation_paths.add(observation)
                saved = read_json(observation)
                require(saved.get("morphology_id") == frame["morphology_id"]
                        and saved.get("training_approved") is False and saved.get("accepted_training_increment") == 0,
                        "Saved observation identity/acceptance differs")
            require(result.get("native_identity") == dict(path=str((capture/"native_identity.json").resolve()),
                        sha256=sha256(capture/"native_identity.json")), "Result native identity pin differs")
            identity = read_json(capture/"native_identity.json")
            worker.validate_native_identity(identity, command,
                owner_identity=evidence["raw_owner_classification"]["metadata"])
            owned = read_json(output/"owned_worker.json")
            require(owned["launcher_pid"] == identity["command_identity"]["ProcessId"], "Owned bootstrap differs")
            require(identity["owner_identity"]["ProcessId"] == os.getpid(), "Native worker had another owner")
            verify_bindings(pins)
            completed = dict(schema=SCHEMA, state="owned_native_diagnostic_exited_zero_not_training",
                native_result=dict(path=str(capture/"result.json"), sha256=sha256(capture/"result.json")),
                owned_exit_sha256=sha256(output/"owned_exit.json"), request_sha256=request_sha256,
                owner_started_sha256=sha256(output/"owner_started.json"), native_identity_sha256=sha256(capture/"native_identity.json"),
                elapsed_seconds=time.perf_counter()-started, training_approved=False, accepted_training_increment=0)
            save_json(output/"owner_complete.json", completed)
            print(json.dumps(completed, indent=2), flush=True)
            return completed
        except BaseException as exc:
            save_json(output/"owner_failure.json", dict(schema=SCHEMA, error=repr(exc),
                training_approved=False, accepted_training_increment=0, automatic_retry=False))
            raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--request", type=Path, required=True)
    p.add_argument("--request-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    a=p.parse_args()
    run(a.request, a.request_sha256, a.output)


if __name__ == "__main__":
    main()
