from greenhouse_sim import repeatability


def _report(*, succeeded=True, phase="deposited", cuts=1, unsafe=0):
    return {
        "succeeded": succeeded,
        "bimanual_probe": {
            "succeeded": succeeded,
            "blade_safety_clear": True,
            "unsafe_contacts": [{}] * unsafe,
            "physical_cuts": [
                {"benchmark_valid": True, "intended_target": True} for _ in range(cuts)
            ],
            "task": {"task": {"phase": phase}},
        },
    }


def test_acceptance_requires_the_complete_physical_contract():
    assert repeatability.acceptance_result(_report())["accepted"]
    assert not repeatability.acceptance_result(_report(cuts=2))["accepted"]
    assert not repeatability.acceptance_result(_report(unsafe=1))["accepted"]
    assert not repeatability.acceptance_result(_report(phase="transported"))["accepted"]


def test_summary_does_not_hide_failed_trials():
    entries = [
        {"target": "Vine_0000/SubStem_00", "acceptance": repeatability.acceptance_result(_report())},
        {
            "target": "Vine_0000/SubStem_01",
            "acceptance": repeatability.acceptance_result(_report(succeeded=False)),
        },
    ]
    summary = repeatability.summarize(entries)
    assert summary["accepted"] == 1
    assert summary["failed"] == 1
    assert summary["acceptance_rate"] == 0.5
    assert not summary["all_accepted"]
