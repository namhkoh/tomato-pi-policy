"""Regressions for scene-level vine physics fixtures."""

from __future__ import annotations

import numpy as np

from greenhouse_sim import organs
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


def test_oriented_foliage_proxy_has_right_handed_frame_and_minimum_thickness() -> None:
    angle = np.radians(31.0)
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rectangle = np.asarray(
        [
            [-0.04, -0.015, 0.0],
            [0.04, -0.015, 0.0],
            [0.04, 0.015, 0.0],
            [-0.04, 0.015, 0.0],
        ]
    ) @ rotation.T

    centre, frame, half_extents = vine_physics._oriented_proxy_box(rectangle)

    np.testing.assert_allclose(centre, np.zeros(3), atol=1e-12)
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) > 0.999999
    assert (
        2.0 * np.min(half_extents)
        >= vine_physics.FOLIAGE_CONTACT_MINIMUM_THICKNESS_M
    )


def test_foliage_proxy_is_robot_contactable_but_filtered_from_plant_bodies() -> None:
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage, "/World")
    UsdGeom.Xform.Define(stage, "/World/Vine")
    root_body = "/World/Vine/Root"
    branch_body = "/World/Vine/Branch"
    for path in (root_body, branch_body):
        UsdGeom.Xform.Define(stage, path)
    component = organs.Component(
        index=2,
        material="Leaf",
        tissue=organs.Tissue.FOLIAGE,
        vertices=np.asarray(
            [
                [-0.03, -0.01, 0.0],
                [0.03, -0.01, 0.0],
                [0.03, 0.01, 0.0],
                [-0.03, 0.01, 0.0],
            ]
        ),
        triangles=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )
    stem_component = organs.Component(
        index=0,
        material="TomatoStem",
        tissue=organs.Tissue.STEM,
        vertices=np.asarray([[0.0, 0.0, 0.0]]),
        triangles=np.empty((0, 3), dtype=np.int64),
    )
    plant = organs.Plant(
        name="test",
        root=0,
        metadata={},
        organs=[
            organs.Organ(0, stem_component, None, None, 0.0, "MainStem"),
            organs.Organ(1, stem_component, 0, np.zeros(3), 0.0, "SubStem_00"),
            organs.Organ(2, component, 1, np.zeros(3), 0.0, "Foliage_002"),
        ],
    )
    rig = vine_physics.PlantRig(
        root_path="/World/Vine",
        links=[
            vine_physics.Link(
                root_body,
                0,
                0,
                np.zeros(3),
                np.asarray([0.0, 0.0, 0.1]),
                0.005,
            ),
            vine_physics.Link(
                branch_body,
                1,
                0,
                np.zeros(3),
                np.asarray([0.1, 0.0, 0.0]),
                0.003,
            ),
        ],
        joints={},
        cut_joints={},
    )

    infos = vine_physics.author_foliage_contact_proxies(
        stage,
        rig,
        plant,
        lambda points: np.asarray(points, dtype=np.float64),
    )

    assert len(infos) == 1
    assert infos[0].organ_label == "SubStem_00"
    assert infos[0].role == "foliage_grasp"
    proxy = stage.GetPrimAtPath(infos[0].path)
    assert proxy.HasAPI(UsdPhysics.CollisionAPI)
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.guide],
    ).ComputeWorldBound(proxy).ComputeAlignedRange()
    minimum = np.asarray(bounds.GetMin(), dtype=np.float64)
    maximum = np.asarray(bounds.GetMax(), dtype=np.float64)
    assert np.all(component.vertices >= minimum - 1e-12)
    assert np.all(component.vertices <= maximum + 1e-12)
    filtered = UsdPhysics.FilteredPairsAPI.Get(stage, proxy.GetPath())
    assert filtered.GetFilteredPairsRel().GetTargets() == [Sdf.Path(root_body)]
