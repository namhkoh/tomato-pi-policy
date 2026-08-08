"""Regressions for scene-level vine physics fixtures."""

from __future__ import annotations

from greenhouse_sim import vine_physics
from pxr import Sdf
from pxr import Usd
from pxr import UsdGeom
from pxr import UsdPhysics


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
