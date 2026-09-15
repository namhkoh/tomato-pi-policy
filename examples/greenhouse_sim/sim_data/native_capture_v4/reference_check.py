"""Future opt-in checks for unchanged v3 banks and camera schedules.

build(checkpoint, manifest_sha256=...) -> the EXACT v3 bank dictionary.
check_bank(bank, diagnostics=None) / check_schedule(schedule, diagnostics=None)
return True or fail closed. diagnostics is a caller-owned report dictionary;
nothing is added to, removed from, or repinned in the original artifacts.

CLI: --bank PATH --bank-sha256 ORIGINAL_PIN [--schedule PATH
--schedule-sha256 ORIGINAL_PIN]. Emits a separate verification receipt to stdout;
never writes source/job/artifact files or invokes preparation/native processes.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

from ..native_capture_v3 import reference_bank as rb
from ..native_capture_v3 import reference_schedule as rs
from . import io
from . import _bank_replay

VERSION = "greenhouse.reference_verification.v4.1"
UPSTREAM_PINS = {
    str(Path(rb.__file__).resolve()): "9eb87c8c0a16599201f20cea1819c7c1eb003be602542cce255dc72b0dade94d",
    str(Path(rs.__file__).resolve()): "919197a85767154aa07222da56675d142a37fe094e8ba480d63aa6a694ac2b62",
    str(Path(rb.__file__).with_name("__init__.py").resolve()):
        "c2c4caef12e7efa77ccb553ba5567fdecf229cfa062707ecc3c1890d9e445412",
}
_LOADED_BINDINGS = {
    str(p.resolve()): rb.sha256(p)
    for p in (Path(__file__), Path(io.__file__), Path(_bank_replay.__file__),
              Path(__file__).with_name("__init__.py"))
}


def _guard():
    # These checks bind on-disk implementation bytes, not an in-memory attestation.
    for path, expected in {**UPSTREAM_PINS, **_LOADED_BINDINGS}.items():
        rb._require(rb.sha256(path) == expected, "Verification implementation changed: " + path)


def implementation_bindings():
    _guard()
    return {**UPSTREAM_PINS, **_LOADED_BINDINGS}


def build(checkpoint, *, manifest_sha256, diagnostics=None):
    _guard()
    reader = io.Reader()
    result = _bank_replay.replay(checkpoint, manifest_sha256=manifest_sha256, reader=reader)
    _guard()
    if diagnostics is not None:
        diagnostics.update(reader.diagnostics())
    return result


def check_bank(bank, *, diagnostics=None):
    rb._require(bank.get("schema") == rb.SCHEMA, "Unknown reference bank")
    expected = build(bank["checkpoint"], manifest_sha256=bank["checkpoint_manifest_sha256"],
                     diagnostics=diagnostics)
    rb._require(bank == expected, "Reference bank metadata or bindings changed")
    return True


def check_schedule(schedule, *, diagnostics=None):
    _guard()
    rs._require(schedule.get("schema") == rs.SCHEMA, "Unknown reference schedule")
    rs._verify_implementation(schedule["implementation_bindings"])
    reader = io.Reader()
    path = reader.resolve(schedule["source_reference_bank"])
    digest = schedule["source_reference_bank_sha256"]
    rs._require(schedule["source_bindings"] == {str(path): digest},
                "Missing/mismatched source bank binding")
    bank = reader.json(path, digest)
    bank_stats = {}
    check_bank(bank, diagnostics=bank_stats)
    # Reuse the unchanged schedule recipe (including its destination guards).
    # Track its mutable destination route as well, without requiring empty outputs.
    reader.resolve(schedule["qualification_output_root"])
    expected = rs._assemble(bank, path, digest, schedule["qualification_output_root"],
                            schedule["fallback_limit"], schedule["implementation_bindings"])
    rs._require(schedule == expected, "Reference schedule metadata or ordering changed")
    reader.finish()
    rs._verify_implementation(schedule["implementation_bindings"])
    _guard()
    if diagnostics is not None:
        diagnostics.update(bank=bank_stats, schedule=reader.diagnostics())
    return True


def verify_files(bank_path, *, bank_sha256, schedule_path=None, schedule_sha256=None):
    """Pin caller-selected artifact bytes; return a new receipt, never a repinned bank."""
    _guard()
    reader = io.Reader()
    bank_path = reader.resolve(bank_path)
    bank = reader.json(bank_path, bank_sha256)
    details = {}
    if schedule_path is None:
        rb._require(schedule_sha256 is None, "Schedule hash without schedule")
        check_bank(bank, diagnostics=details)
    else:
        rb._require(schedule_sha256 is not None, "Explicit original schedule SHA256 required")
        schedule_path = reader.resolve(schedule_path)
        schedule = reader.json(schedule_path, schedule_sha256)
        rb._require(reader.resolve(schedule["source_reference_bank"]) == bank_path and
                    schedule["source_reference_bank_sha256"] == bank_sha256,
                    "Schedule binds a different source bank")
        check_schedule(schedule, diagnostics=details)
    reader.finish()
    _guard()
    return dict(schema=VERSION, passed=True,
                implementation_bindings=implementation_bindings(),
                artifact_bindings=dict(reader.bindings), artifact_io=reader.diagnostics(),
                verification_io=details, bank_counts=deepcopy(bank["counts"]),
                original_bank_and_schedule_unchanged=True,
                native_jobs_launched=False, training_approved=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--bank-sha256", required=True)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--schedule-sha256")
    args = parser.parse_args(argv)
    result = verify_files(args.bank, bank_sha256=args.bank_sha256,
                          schedule_path=args.schedule, schedule_sha256=args.schedule_sha256)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
