import dataclasses

import pytest

from greenhouse_sim import episode


@dataclasses.dataclass
class _Rig:
    cut_joints: dict
    junctions: dict


@dataclasses.dataclass
class _Runtime:
    name: str
    rig: _Rig


def _runtimes():
    return [
        _Runtime("Vine_0001", _Rig({"SubStem_02": "x"}, {"SubStem_02": object()})),
        _Runtime(
            "Vine_0000",
            _Rig(
                {"SubStem_01": "x", "SubStem_00": "y", "Leaf_00": "z"},
                {"SubStem_00": object(), "SubStem_01": object(), "Leaf_00": object()},
            ),
        ),
    ]


def test_candidates_are_stable_and_only_include_physical_petiole_targets():
    assert episode.candidate_targets(_runtimes()) == (
        ("Vine_0000", "SubStem_00"),
        ("Vine_0000", "SubStem_01"),
        ("Vine_0001", "SubStem_02"),
    )


def test_exact_target_preserves_the_accepted_baseline():
    target = episode.resolve_target(
        _runtimes(), vine_name="Vine_0000", organ_label="SubStem_00", seed=99
    )
    assert target.key == "Vine_0000/SubStem_00"
    assert target.candidate_count == 1


def test_seeded_auto_target_is_repeatable():
    first = episode.resolve_target(_runtimes(), vine_name="Vine_0000", organ_label="auto", seed=3)
    second = episode.resolve_target(_runtimes(), vine_name="Vine_0000", organ_label="auto", seed=3)
    assert first == second
    assert first.candidate_count == 2


def test_unknown_target_fails_with_available_targets():
    with pytest.raises(ValueError, match="available targets"):
        episode.resolve_target(
            _runtimes(), vine_name="Vine_0000", organ_label="SubStem_99", seed=0
        )
