"""Tests for preserving the supplied greenhouse's tomato stem layout."""

from __future__ import annotations

import pathlib

import pytest

from greenhouse_sim import greenhouse_scene
from pxr import Usd


REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _first_gutter_bed() -> tuple[Usd.Stage, Usd.Prim]:
    source = Usd.Stage.Open(
        str(REPOSITORY_ROOT / "greenhouse" / "gh_tomato_test.usd"),
        load=Usd.Stage.LoadNone,
    )
    gutter = greenhouse_scene.select_gutter(source, "Gutter_01")
    gutter.Load()
    return source, greenhouse_scene.find_beds(source, gutter)[0]


def test_source_placement_matches_full_tomato_greenhouse() -> None:
    _source, bed = _first_gutter_bed()
    placement = greenhouse_scene.plan_source_placements(
        [bed],
        [pathlib.Path("tomato_000.usd")],
        plants_per_bed=1,
    )[0]

    assert placement.position == pytest.approx(
        (15.830962740706077, 5.580149838686983, 0.39298668133274306),
        abs=1e-9,
    )
    assert placement.yaw_degrees == pytest.approx(90.0, abs=1e-9)
    assert placement.source_prim == (
        "/World/Main_Cultivation_Zone/Beds/Gutter_01/BedSet_01/"
        "Stems/Side_2/tomato_stem_000"
    )


def test_procedural_placement_remains_available() -> None:
    _source, bed = _first_gutter_bed()
    placement = greenhouse_scene.plan_placements(
        [bed],
        [pathlib.Path("tomato_000.usd")],
        plants_per_bed=1,
        seed=0,
    )[0]

    assert placement.source_prim is None
    assert placement.position[2] == pytest.approx(0.4068094289789232)
