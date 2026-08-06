"""Compose tomato vines into the greenhouse stage.

The greenhouse asset is left untouched: the benchmark scene is a new layer that
sublayers it and authors only the vine placements. Regenerating a scene, or
building several with different seeds, therefore never risks the source art.

Vines are rooted on the substrate surface of a bed and spaced along its length,
which is how high-wire tomato is actually grown. Placement is derived from each
bed's own world bounds rather than hard-coded coordinates, so it survives the
per-bed transforms in the greenhouse and works for any bed in either zone.
"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import random

import numpy as np

from greenhouse_sim import usd_env

usd_env.ensure_pxr()

from pxr import Gf  # noqa: E402
from pxr import Sdf  # noqa: E402
from pxr import Usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402

# In-row spacing for high-wire tomato; a 5 m bed then carries 20 plants.
DEFAULT_SPACING_M = 0.25

# Vines are modelled leaning to one side. Jittering the yaw stops a row from
# looking like one plant stamped N times, which would let a policy overfit to a
# single canopy pose.
DEFAULT_YAW_JITTER_DEG = 25.0

_SCENE_ROOT = "/World"
_VINE_SCOPE = "/World/Vines"


@dataclasses.dataclass(frozen=True)
class Placement:
    """One vine instance in the scene."""

    prim_path: str
    asset: pathlib.Path
    position: tuple[float, float, float]
    yaw_degrees: float
    bed: str


@dataclasses.dataclass(frozen=True)
class Scene:
    """Result of composing a benchmark scene."""

    path: pathlib.Path
    placements: list[Placement]
    beds: list[str]
    repairs: list[str] = dataclasses.field(default_factory=list)


def repair_environment(source: Usd.Stage, stage: Usd.Stage) -> list[str]:
    """Neutralise light textures the greenhouse asset cannot resolve.

    The shipped DomeLight points at an absolute path on the machine the asset
    was authored on. Left alone it logs a load failure every run and lights the
    scene with whatever the renderer falls back to, which is exactly the kind of
    silent, machine-dependent variation a vision benchmark must not have.
    Clearing the texture in the scene layer leaves a uniform sky at the light's
    authored colour and intensity, identical on every machine.
    """
    repairs = []
    for prim in source.Traverse():
        attribute = prim.GetAttribute("inputs:texture:file")
        if not attribute or not attribute.IsValid():
            continue
        asset = attribute.Get()
        if asset is None or not str(asset.path):
            continue
        if asset.resolvedPath:
            continue
        override = stage.OverridePrim(prim.GetPath())
        override.CreateAttribute("inputs:texture:file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(""))
        repairs.append(f"{prim.GetPath()}: cleared unresolvable texture {asset.path}")
    return repairs


def find_beds(stage: Usd.Stage) -> list[Usd.Prim]:
    """Every BedSet group in the greenhouse, in a stable order."""
    beds = [p for p in stage.Traverse() if p.GetName().startswith("BedSet")]
    return sorted(beds, key=lambda p: str(p.GetPath()))


def _bed_surface(bed: Usd.Prim, cache: UsdGeom.BBoxCache) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Centre, long-axis direction and half-length of a bed's planting surface.

    Measured on the Bed child (the substrate trough) rather than the BedSet,
    whose bounds include trellis strings rising several metres.
    """
    trough = bed.GetChild("Bed")
    if not trough or not trough.IsValid():
        return None
    box = cache.ComputeWorldBound(trough).ComputeAlignedRange()
    if box.IsEmpty():
        return None
    lower = np.array(box.GetMin(), dtype=np.float64)
    upper = np.array(box.GetMax(), dtype=np.float64)

    extents = upper - lower
    # Z is up in the greenhouse; the long horizontal extent is the row axis.
    long_axis = int(np.argmax(extents[:2]))
    direction = np.zeros(3)
    direction[long_axis] = 1.0

    centre = 0.5 * (lower + upper)
    centre[2] = upper[2]  # plant on the substrate surface
    return centre, direction, float(extents[long_axis] * 0.5)


def plan_placements(
    beds: list[Usd.Prim],
    vines: list[pathlib.Path],
    *,
    spacing: float = DEFAULT_SPACING_M,
    yaw_jitter_degrees: float = DEFAULT_YAW_JITTER_DEG,
    plants_per_bed: int | None = None,
    seed: int = 0,
) -> list[Placement]:
    """Lay vines out along each bed without authoring anything."""
    if not vines:
        raise ValueError("no vine assets supplied")

    rng = random.Random(seed)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    placements: list[Placement] = []

    for bed in beds:
        surface = _bed_surface(bed, cache)
        if surface is None:
            continue
        centre, direction, half_length = surface

        capacity = max(1, int((2.0 * half_length) // spacing))
        count = capacity if plants_per_bed is None else min(plants_per_bed, capacity)
        # Centre the run so end plants are not flush with the trough lip.
        start = -0.5 * (count - 1) * spacing

        for slot in range(count):
            position = centre + direction * (start + slot * spacing)
            asset = vines[len(placements) % len(vines)]
            placements.append(
                Placement(
                    prim_path=f"{_VINE_SCOPE}/Vine_{len(placements):04d}",
                    asset=asset,
                    position=(float(position[0]), float(position[1]), float(position[2])),
                    yaw_degrees=rng.uniform(-yaw_jitter_degrees, yaw_jitter_degrees),
                    bed=str(bed.GetPath()),
                )
            )
    return placements


def build_scene(
    greenhouse: str | pathlib.Path,
    vines: list[pathlib.Path],
    output: str | pathlib.Path,
    *,
    max_beds: int | None = 1,
    spacing: float = DEFAULT_SPACING_M,
    yaw_jitter_degrees: float = DEFAULT_YAW_JITTER_DEG,
    plants_per_bed: int | None = None,
    seed: int = 0,
) -> Scene:
    """Write a benchmark scene layering vine placements over the greenhouse."""
    greenhouse = pathlib.Path(greenhouse).resolve()
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    source = Usd.Stage.Open(str(greenhouse))
    if source is None:
        raise FileNotFoundError(f"could not open greenhouse stage {greenhouse}")
    beds = find_beds(source)[:max_beds]
    if not beds:
        raise ValueError(f"no BedSet prims found in {greenhouse}")

    placements = plan_placements(
        beds,
        vines,
        spacing=spacing,
        yaw_jitter_degrees=yaw_jitter_degrees,
        plants_per_bed=plants_per_bed,
        seed=seed,
    )

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    # Sublayer rather than reference so the greenhouse composes in unmodified
    # and the scene layer holds nothing but the vines.
    stage.GetRootLayer().subLayerPaths.append(_relative_asset_path(greenhouse, output))

    world = UsdGeom.Xform.Define(stage, Sdf.Path(_SCENE_ROOT))
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, Sdf.Path(_VINE_SCOPE))
    repairs = repair_environment(source, stage)

    for placement in placements:
        prim = UsdGeom.Xform.Define(stage, Sdf.Path(placement.prim_path))
        prim.AddTranslateOp().Set(Gf.Vec3d(*placement.position))
        prim.AddRotateZOp().Set(placement.yaw_degrees)
        prim.GetPrim().GetReferences().AddReference(_relative_asset_path(placement.asset.resolve(), output))

    stage.GetRootLayer().Save()
    return Scene(path=output, placements=placements, beds=[str(b.GetPath()) for b in beds], repairs=repairs)


def _relative_asset_path(target: pathlib.Path, referrer: pathlib.Path) -> str:
    """Asset path for `target` relative to the layer at `referrer`.

    Uses os.path.relpath rather than PurePath.relative_to(walk_up=...), which
    needs Python 3.12; Isaac Sim 5.1 ships 3.11.
    """
    try:
        relative = os.path.relpath(pathlib.Path(target).resolve(), referrer.resolve().parent)
    except ValueError:  # different drives on Windows
        return pathlib.Path(target).resolve().as_posix()
    return pathlib.PurePath(relative).as_posix()
