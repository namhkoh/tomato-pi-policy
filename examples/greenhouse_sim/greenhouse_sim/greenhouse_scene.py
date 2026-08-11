"""Compose tomato vines into the greenhouse stage.

The greenhouse asset is left untouched: the benchmark scene is a new layer that
sublayers it, suppresses its legacy all-gutter vines and stale robot payload,
then authors the benchmark vine placements. Regenerating a scene, or building
several with different seeds, therefore never risks the source art.

By default, each converted physics-ready vine inherits the exact root position
and two-sided facing authored for its matching stem in gh_tomato_test.usd.
The gutter and bed transforms are never altered. A procedural mode remains
available for generated layouts; it derives positions from bed bounds rather
than hard-coded coordinates.
"""

from __future__ import annotations

import dataclasses
import math
import os
import pathlib
import random
import re

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

# The supplied full-tomato greenhouse already contains the agronomically
# authored two-sided stem layout. Benchmark vines should inherit those root
# frames by default rather than being re-centred on the trough surface.
DEFAULT_PLACEMENT_MODE = "source"

_SCENE_ROOT = "/World"
_VINE_SCOPE = "/World/Vines"
_STALE_ROBOT_PATH = "/World/RB_Y1_Cam"
_STALE_PHYSICS_SCENE_PATH = "/PhysicsScene"
_EMBEDDED_STEM_SCOPE_NAME = "Stems"


@dataclasses.dataclass(frozen=True)
class Placement:
    """One vine instance in the scene."""

    prim_path: str
    asset: pathlib.Path
    position: tuple[float, float, float]
    yaw_degrees: float
    bed: str
    source_prim: str | None = None


@dataclasses.dataclass(frozen=True)
class Scene:
    """Result of composing a benchmark scene."""

    path: pathlib.Path
    placements: list[Placement]
    beds: list[str]
    gutter: str | None
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


def suppress_embedded_benchmark_content(source: Usd.Stage, stage: Usd.Stage) -> list[str]:
    """Disable legacy benchmark content without modifying the source asset."""
    repairs: list[str] = []
    stale_robot = source.GetPrimAtPath(_STALE_ROBOT_PATH)
    if stale_robot and stale_robot.IsValid():
        override = stage.OverridePrim(_STALE_ROBOT_PATH)
        override.GetPayloads().SetPayloads([])
        override.SetActive(False)
        repairs.append(f"{_STALE_ROBOT_PATH}: deactivated stale external robot/camera payload")

    stale_physics = source.GetPrimAtPath(_STALE_PHYSICS_SCENE_PATH)
    if stale_physics and stale_physics.IsValid():
        stage.OverridePrim(_STALE_PHYSICS_SCENE_PATH).SetActive(False)
        repairs.append(
            f"{_STALE_PHYSICS_SCENE_PATH}: deactivated invalid legacy physics scene"
        )

    embedded_stems = [
        prim
        for prim in source.Traverse()
        if prim.GetName() == _EMBEDDED_STEM_SCOPE_NAME
        and prim.GetPath().pathString.endswith(f"/{_EMBEDDED_STEM_SCOPE_NAME}")
    ]
    for stems in embedded_stems:
        stage.OverridePrim(stems.GetPath()).SetActive(False)
    if embedded_stems:
        repairs.append(
            f"deactivated {len(embedded_stems)} embedded all-gutter tomato stem scopes"
        )
    return repairs


def find_gutters(stage: Usd.Stage) -> list[Usd.Prim]:
    """Every gutter group in the greenhouse, in a stable order."""
    gutters = [p for p in stage.Traverse() if p.GetName().startswith("Gutter_")]
    return sorted(gutters, key=lambda p: str(p.GetPath()))


def select_gutter(stage: Usd.Stage, selector: str) -> Usd.Prim:
    """Resolve a gutter by full prim path or unique prim name."""
    if selector.startswith("/"):
        gutter = stage.GetPrimAtPath(selector)
        if gutter and gutter.IsValid() and gutter.GetName().startswith("Gutter_"):
            return gutter
        raise ValueError(f"gutter path does not exist: {selector}")

    matches = [gutter for gutter in find_gutters(stage) if gutter.GetName() == selector]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        available = ", ".join(gutter.GetName() for gutter in find_gutters(stage))
        raise ValueError(f"gutter {selector!r} not found; available: {available}")
    paths = ", ".join(str(gutter.GetPath()) for gutter in matches)
    raise ValueError(f"gutter name {selector!r} is ambiguous: {paths}")


def find_beds(stage: Usd.Stage, gutter: Usd.Prim | None = None) -> list[Usd.Prim]:
    """Every BedSet group under ``gutter`` (or the stage), stably ordered."""
    traversal = Usd.PrimRange(gutter) if gutter is not None else stage.Traverse()
    beds = [p for p in traversal if p.GetName().startswith("BedSet")]
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


def _source_stem_name(asset: pathlib.Path) -> str:
    match = re.fullmatch(r"tomato_(\d+)", pathlib.Path(asset).stem)
    if match is None:
        raise ValueError(
            f"source placement requires a tomato_NNN asset, got {asset.name!r}"
        )
    return f"tomato_stem_{int(match.group(1)):03d}"


def _source_stem_frame(prim: Usd.Prim) -> tuple[tuple[float, float, float], float]:
    """Convert an original Y-up stem frame into a Z-up asset root frame."""
    if not prim.IsA(UsdGeom.Xformable):
        raise ValueError(f"source stem is not transformable: {prim.GetPath()}")
    matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )

    def unit_axis(axis: Gf.Vec3d) -> np.ndarray:
        direction = np.asarray(matrix.TransformDir(axis), dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length <= 1e-12:
            raise ValueError(f"source stem has a degenerate frame: {prim.GetPath()}")
        return direction / length

    # Converted vines use (glTF X, -glTF Z, glTF Y) as their USD XYZ basis.
    # Therefore the placement's X axis is the source glTF X axis, its Y axis
    # is minus source glTF Z, and its Z axis is source glTF Y.
    placement_x = unit_axis(Gf.Vec3d(1.0, 0.0, 0.0))
    placement_y = -unit_axis(Gf.Vec3d(0.0, 0.0, 1.0))
    placement_z = unit_axis(Gf.Vec3d(0.0, 1.0, 0.0))
    if (
        abs(float(placement_x[2])) > 1e-6
        or abs(float(placement_y[2])) > 1e-6
        or not np.allclose(placement_z, (0.0, 0.0, 1.0), atol=1e-6)
        or float(np.dot(np.cross(placement_x, placement_y), placement_z))
        < 1.0 - 1e-6
    ):
        raise ValueError(
            f"source stem frame is not a rigid Z-up yaw: {prim.GetPath()}"
        )
    translation = matrix.ExtractTranslation()
    yaw_degrees = math.degrees(
        math.atan2(float(placement_x[1]), float(placement_x[0]))
    )
    return tuple(float(value) for value in translation), yaw_degrees


def plan_source_placements(
    beds: list[Usd.Prim],
    vines: list[pathlib.Path],
    *,
    plants_per_bed: int | None = None,
) -> list[Placement]:
    """Place converted vines at their exact authored full-greenhouse frames."""
    if not vines:
        raise ValueError("no vine assets supplied")
    count = (
        len(vines)
        if plants_per_bed is None
        else min(plants_per_bed, len(vines))
    )
    placements: list[Placement] = []
    for bed in beds:
        descendants = {prim.GetName(): prim for prim in Usd.PrimRange(bed)}
        for _slot in range(count):
            asset = vines[len(placements) % len(vines)]
            stem_name = _source_stem_name(asset)
            source_prim = descendants.get(stem_name)
            if source_prim is None:
                raise ValueError(
                    f"{bed.GetPath()} has no source placement for {stem_name}"
                )
            position, yaw_degrees = _source_stem_frame(source_prim)
            placements.append(
                Placement(
                    prim_path=f"{_VINE_SCOPE}/Vine_{len(placements):04d}",
                    asset=asset,
                    position=position,
                    yaw_degrees=yaw_degrees,
                    bed=str(bed.GetPath()),
                    source_prim=str(source_prim.GetPath()),
                )
            )
    return placements


def build_scene(
    greenhouse: str | pathlib.Path,
    vines: list[pathlib.Path],
    output: str | pathlib.Path,
    *,
    gutter: str | None = "Gutter_01",
    max_beds: int | None = None,
    spacing: float = DEFAULT_SPACING_M,
    yaw_jitter_degrees: float = DEFAULT_YAW_JITTER_DEG,
    plants_per_bed: int | None = None,
    seed: int = 0,
    placement_mode: str = DEFAULT_PLACEMENT_MODE,
) -> Scene:
    """Write a benchmark scene layering vine placements over the greenhouse."""
    greenhouse = pathlib.Path(greenhouse).resolve()
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)

    source = Usd.Stage.Open(str(greenhouse), load=Usd.Stage.LoadNone)
    if source is None:
        raise FileNotFoundError(f"could not open greenhouse stage {greenhouse}")
    selected_gutter = None if gutter is None else select_gutter(source, gutter)
    if selected_gutter is not None:
        selected_gutter.Load()
    else:
        source.Load()
    beds = find_beds(source, selected_gutter)
    if max_beds is not None:
        beds = beds[:max_beds]
    if not beds:
        raise ValueError(f"no BedSet prims found in {greenhouse}")

    if placement_mode == "source":
        placements = plan_source_placements(
            beds,
            vines,
            plants_per_bed=plants_per_bed,
        )
    elif placement_mode == "procedural":
        placements = plan_placements(
            beds,
            vines,
            spacing=spacing,
            yaw_jitter_degrees=yaw_jitter_degrees,
            plants_per_bed=plants_per_bed,
            seed=seed,
        )
    else:
        raise ValueError(f"unknown placement mode: {placement_mode}")

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, Sdf.Path(_SCENE_ROOT))
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, Sdf.Path(_VINE_SCOPE))
    repairs = repair_environment(source, stage)
    repairs.extend(suppress_embedded_benchmark_content(source, stage))
    # Author deactivation opinions before composing the source so its stale
    # external robot payload is never requested by this benchmark stage.
    stage.GetRootLayer().subLayerPaths.append(_relative_asset_path(greenhouse, output))

    for placement in placements:
        prim = UsdGeom.Xform.Define(stage, Sdf.Path(placement.prim_path))
        prim.AddTranslateOp().Set(Gf.Vec3d(*placement.position))
        prim.AddRotateZOp().Set(placement.yaw_degrees)
        prim.GetPrim().GetReferences().AddReference(_relative_asset_path(placement.asset.resolve(), output))

    stage.GetRootLayer().Save()
    return Scene(
        path=output,
        placements=placements,
        beds=[str(b.GetPath()) for b in beds],
        gutter=None if selected_gutter is None else str(selected_gutter.GetPath()),
        repairs=repairs,
    )


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
