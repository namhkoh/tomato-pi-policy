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
