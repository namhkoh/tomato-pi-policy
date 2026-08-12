from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

from interactive_greenhouse import (  # noqa: E402
    _INCIDENTAL_FOLIAGE_IMPULSE_NS,
    _probe_unsafe_contacts,
)


def _pair(other, *, impulse, robot="/World/RBY1/ee_left/contact_proxy"):
    return {
        "collider0": other,
        "collider1": robot,
        "maximum_impulse_ns": impulse,
    }


def _unsafe(*pairs):
    return _probe_unsafe_contacts(
        {"pairs": list(pairs)},
        {"colliders": ("/World/InteractiveVines/target/cut",)},
        {
            "collider": "/World/InteractiveVines/target/grasp",
            "orphan_colliders": ("/World/InteractiveVines/target/grasp",),
        },
    )


def test_low_impulse_flexible_foliage_brush_is_not_unsafe():
    pair = _pair(
        "/World/InteractiveVines/Vine_0002/Physics/Organ_0116/"
        "Link_000/FoliageContact_0237",
        impulse=_INCIDENTAL_FOLIAGE_IMPULSE_NS,
    )
    assert _unsafe(pair) == []


def test_stronger_foliage_and_any_rigid_vine_contact_remain_unsafe():
    foliage = _pair(
        "/World/InteractiveVines/Vine_0002/Physics/Organ_0116/"
        "Link_000/FoliageContact_0237",
        impulse=_INCIDENTAL_FOLIAGE_IMPULSE_NS + 1e-5,
    )
    stem = _pair(
        "/World/InteractiveVines/Vine_0002/Physics/Organ_0116/Link_001/Collider",
        impulse=1e-5,
    )
    assert _unsafe(foliage, stem) == [foliage, stem]


def test_intended_grasp_and_floor_support_are_allowed():
    grasp = _pair(
        "/World/InteractiveVines/target/grasp",
        impulse=0.2,
        robot="/World/RBY1/ee_finger_l1/contact_proxy",
    )
    floor = _pair(
        "/World/Main_Cultivation_Zone/Env/GroundPlane/CollisionPlane",
        impulse=0.2,
        robot="/World/RBY1/wheel_l/contact_proxy",
    )
    assert _unsafe(grasp, floor) == []
