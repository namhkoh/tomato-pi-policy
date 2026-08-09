"""Acceptance aggregation for repeated deleafing simulator episodes."""

from __future__ import annotations


def acceptance_result(report: dict) -> dict:
    probe = report.get("bimanual_probe") or {}
    task = probe.get("task") or report.get("bimanual_task") or {}
    task_state = task.get("task") or {}
    cuts = [cut for cut in probe.get("physical_cuts", ()) if cut.get("benchmark_valid")]
    unsafe = probe.get("unsafe_contacts") or []
    checks = {
        "top_level_succeeded": report.get("succeeded") is True,
        "probe_succeeded": probe.get("succeeded") is True,
        "exactly_one_benchmark_cut": len(cuts) == 1,
        "intended_target_cut": len(cuts) == 1 and cuts[0].get("intended_target") is True,
        "zero_unsafe_contacts": len(unsafe) == 0,
        "blade_safety_clear": probe.get("blade_safety_clear") is True,
        "deposited": task_state.get("phase") == "deposited",
    }
    return {"accepted": all(checks.values()), "checks": checks}


def summarize(entries: list[dict]) -> dict:
    accepted = sum(bool(entry["acceptance"]["accepted"]) for entry in entries)
    targets = sorted({entry["target"] for entry in entries})
    return {
        "episodes": len(entries),
        "accepted": accepted,
        "failed": len(entries) - accepted,
        "acceptance_rate": accepted / len(entries) if entries else 0.0,
        "targets": targets,
        "all_accepted": bool(entries) and accepted == len(entries),
        "entries": entries,
    }
