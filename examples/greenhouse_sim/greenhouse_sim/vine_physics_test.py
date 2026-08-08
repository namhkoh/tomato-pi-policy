"""Regressions for scene-level vine physics fixtures."""

from __future__ import annotations

import numpy as np

from greenhouse_sim import vine_physics
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdPhysics


def test_cut_zone_collider_radius_does_not_change_structural_mass() -> None:
    properties = vine_physics.TissueProperties()
    link = vine_physics.Link(
        path="/World/Petiole",
        organ=1,
        index=0,
        start=np.asarray([0.0, 0.0, 0.0]),
        end=np.asarray([0.0, 0.0, 0.04]),
        radius=0.011,
    )

    assert vine_physics._physical_collider_radius(link, 0.004) == 0.004
    structural_mass = vine_physics.capsule_mass(
        link.radius, link.length, properties.density_kg_m3
    )
    collision_radius_mass = vine_physics.capsule_mass(
        0.004, link.length, properties.density_kg_m3
    )
    assert structural_mass > collision_radius_mass


def test_physical_cut_zone_covers_every_segment_through_25_mm_stub() -> None:
    arcs = np.asarray([0.0, 0.008, 0.016, 0.024, 0.032, 0.040])

    assert vine_physics._petiole_cut_zone_segments(arcs) == (0, 1, 2, 3)
    assert vine_physics._petiole_grasp_segment(arcs) == 4


def test_cut_segment_frame_follows_bent_collider_and_preserves_stub_arc() -> None:
    links = [
        vine_physics.Link(
            path="/World/Petiole/Link_000",
            organ=7,
            index=0,
            start=np.asarray([0.0, 0.0, 0.0]),
            end=np.asarray([0.010, 0.0, 0.0]),
            radius=0.004,
        ),
        vine_physics.Link(
            path="/World/Petiole/Link_001",
            organ=7,
            index=1,
            start=np.asarray([0.010, 0.0, 0.0]),
            end=np.asarray([0.010, 0.012, 0.0]),
            radius=0.004,
        ),
    ]
    collider = vine_physics.ColliderInfo(
        path="/World/Petiole/Link_001/Collider",
        body_path=links[1].path,
        organ=7,
        organ_label="SubStem_00",
        segment=1,
        role="petiole_cut_zone",
    )
    rig = vine_physics.PlantRig(
        root_path="/World/Petiole",
        links=links,
        joints={},
        cut_joints={},
    )

    centre, axis, arc_start = rig.cut_segment_frame(collider)
    contact = np.asarray([0.010, 0.005, 0.0])
    offset = contact - centre
    signed_stub = float(np.dot(offset, axis))
    radial_distance = float(np.linalg.norm(offset - signed_stub * axis))

    np.testing.assert_allclose(axis, [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(centre, [0.010, -0.010, 0.0], atol=1e-12)
    assert arc_start == 0.010
    assert abs(signed_stub - 0.015) < 1e-12
    assert radial_distance < 1e-12


def test_synthetic_catch_plane_can_exclude_the_robot_tree() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/RBY1/base")

    vine_physics.add_ground_plane(
        stage,
        path="/World/Vine/CatchPlane",
        height=0.8,
        size=1.2,
        filtered_paths=("/World/RBY1/base",),
    )

    catcher = stage.GetPrimAtPath("/World/Vine/CatchPlane")
    assert catcher.HasAPI(UsdPhysics.CollisionAPI)
    filtered = UsdPhysics.FilteredPairsAPI.Get(stage, catcher.GetPath())
    assert filtered.GetFilteredPairsRel().GetTargets() == [Sdf.Path("/World/RBY1/base")]


def test_flush_cut_zone_is_filtered_from_every_plant_contact_body() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    paths = (
        ("/World/CutBody", "/World/CutBody/Collider"),
        ("/World/StemBody", "/World/StemBody/Collider"),
        ("/World/GraspBody", "/World/GraspBody/Collider"),
    )
    for body_path, collider_path in paths:
        UsdGeom.Xform.Define(stage, body_path)
        UsdGeom.Capsule.Define(stage, collider_path)
    infos = [
        vine_physics.ColliderInfo(
            path=collider_path,
            body_path=body_path,
            organ=index,
            organ_label=("SubStem_00" if index != 1 else "MainStem"),
            segment=0,
            role=("petiole_cut_zone" if index == 0 else "petiole_grasp"),
        )
        for index, (body_path, collider_path) in enumerate(paths)
    ]

    vine_physics._filter_cut_zones_from_plant(stage, infos)

    cut_zone = infos[0]
    filtered = UsdPhysics.FilteredPairsAPI.Get(stage, Sdf.Path(cut_zone.path))
    assert set(filtered.GetFilteredPairsRel().GetTargets()) == {
        Sdf.Path(info.body_path)
        for info in infos
        if info.body_path != cut_zone.body_path
    }
