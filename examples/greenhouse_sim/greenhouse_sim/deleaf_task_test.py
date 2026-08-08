"""Acceptance cases for the bi-manual deleafing task order."""

from __future__ import annotations

import numpy as np

from greenhouse_sim import deleaf_task


def _task() -> deleaf_task.BimanualDeleafTask:
    return deleaf_task.BimanualDeleafTask(
        "Vine_0000",
        "SubStem_00",
        deleaf_task.TaskParameters(
            required_grasp_steps=2,
            minimum_transport_clearance_m=0.10,
            drop_zone_min_m=(6.0, 3.5, -0.31),
            drop_zone_max_m=(7.5, 4.5, 0.0),
        ),
    )


def _grasp(task: deleaf_task.BimanualDeleafTask) -> None:
    for _ in range(2):
        task.observe_grasp(
            vine="Vine_0000",
            organ="SubStem_00",
            body_path="/Vine/SubStem/Link_002",
            finger_contacts={"left_finger_1", "left_finger_2"},
            force_n=2.0,
        )
        task.advance()


def test_complete_bimanual_sequence_succeeds() -> None:
    task = _task()
    _grasp(task)
    assert task.phase is deleaf_task.Phase.GRASPED
    assert task.observe_cut(
        vine="Vine_0000",
        organ="SubStem_00",
        physical_blade=True,
        intended_target=True,
    )
    task.observe_hold(grasp_active=True)
    assert task.observe_transport(0.12)
    assert task.observe_release()
    assert task.observe_deposit(
        centroid_m=np.array([6.8, 3.8, -0.28]),
        speed_m_s=0.05,
        floor_contact=True,
    )
    assert task.succeeded
    assert task.summary["phase"] == "deposited"


def test_cut_before_left_grasp_fails() -> None:
    task = _task()
    assert not task.observe_cut(
        vine="Vine_0000",
        organ="SubStem_00",
        physical_blade=True,
        intended_target=True,
    )
    assert task.phase is deleaf_task.Phase.FAILED
    assert task.failures[-1]["reason"] == "cut_before_left_grasp"


def test_losing_orphan_before_release_fails() -> None:
    task = _task()
    _grasp(task)
    task.observe_cut(
        vine="Vine_0000",
        organ="SubStem_00",
        physical_blade=True,
        intended_target=True,
    )
    task.observe_hold(grasp_active=False)
    assert task.phase is deleaf_task.Phase.FAILED
    assert task.failures[-1]["reason"] == "orphan_dropped_before_commanded_release"


def test_unintended_cut_and_early_release_fail() -> None:
    wrong = _task()
    _grasp(wrong)
    wrong.observe_cut(
        vine="Vine_0000",
        organ="SubStem_01",
        physical_blade=True,
        intended_target=False,
    )
    assert wrong.failures[-1]["reason"] == "unintended_organ_cut"

    early = _task()
    _grasp(early)
    early.observe_cut(
        vine="Vine_0000",
        organ="SubStem_00",
        physical_blade=True,
        intended_target=True,
    )
    assert not early.observe_release()
    assert early.failures[-1]["reason"] == "release_before_safe_transport"


def test_protected_contact_before_cut_fails_sequence() -> None:
    task = _task()
    _grasp(task)
    assert not task.observe_cut(
        vine="Vine_0000",
        organ="SubStem_00",
        physical_blade=True,
        intended_target=True,
        safe_path=False,
    )
    assert task.phase is deleaf_task.Phase.FAILED
    assert task.failures[-1]["reason"] == "protected_contact_before_cut"
