"""Drive the vine's render meshes from its physics bodies.

The capsule chains in `vine_physics` are the simulation substrate, not the
plant: what must be seen is the original GLB geometry. This attaches each
organ's mesh as a child of the rigid body that carries it, so the art moves with
the physics and the capsules stay hidden.

Attachment is per organ rather than per capsule. That is the right granularity
for this task: an organ is the unit that gets severed, so when a petiole is cut
its whole mesh -- and its leaflets -- travel with the released body, which is
exactly the behaviour the benchmark scores. The cost is that bending *within* a
long organ does not deform its mesh, which matters only for the main stem, and
the main stem is clipped to the trellis and moves under 25 mm.
"""

from __future__ import annotations

import pathlib

import numpy as np

from greenhouse_sim import glb
from greenhouse_sim import organs
from greenhouse_sim import usd_env
from greenhouse_sim import vine_physics
from greenhouse_sim import vine_usd

usd_env.ensure_pxr()

from pxr import Sdf  # noqa: E402
from pxr import Usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402
from pxr import UsdShade  # noqa: E402
from pxr import Vt  # noqa: E402


def _link_frame(link: vine_physics.Link) -> tuple[np.ndarray, np.ndarray]:
    """Centre and rotation of a link's local frame, matching its capsule."""
    centre = 0.5 * (link.start + link.end)
    axis = link.end - link.start
    norm = float(np.linalg.norm(axis))
    if norm <= 0:
        return centre, np.eye(3)
    direction = axis / norm

    # Same +Z-onto-axis rotation the capsule is authored with.
    reference = np.array([0.0, 0.0, 1.0])
    dot = float(np.clip(np.dot(reference, direction), -1.0, 1.0))
    if dot > 1.0 - 1e-9:
        return centre, np.eye(3)
    if dot < -1.0 + 1e-9:
        return centre, np.diag([1.0, -1.0, -1.0])
    cross = np.cross(reference, direction)
    skew = np.array(
        [[0.0, -cross[2], cross[1]], [cross[2], 0.0, -cross[0]], [-cross[1], cross[0], 0.0]]
    )
    rotation = np.eye(3) + skew + skew @ skew / (1.0 + dot)
    return centre, rotation


def attach_organ_visuals(
    stage: Usd.Stage,
    rig: vine_physics.PlantRig,
    plant: organs.Plant,
    asset: glb.Glb,
    to_stage_frame,
) -> int:
    """Parent each organ's render mesh to the body that carries it.

    Returns the number of organs given visible geometry.
    """
    looks = Sdf.Path(rig.root_path).AppendChild("Looks")
    UsdGeom.Scope.Define(stage, looks)
    materials: dict[int, UsdShade.Material] = {}
    textures = _write_textures(asset, rig.root_path)

    carriers = _carrier_links(rig, plant)
    attached = 0
    for organ in plant.organs:
        link = carriers.get(organ.index)
        if link is None:
            continue

        centre, rotation = _link_frame(link)
        # Express the organ's vertices in the carrier body's local frame, so the
        # mesh rides along as that body is simulated.
        world = to_stage_frame(organ.component.vertices)
        local = (world - centre) @ rotation

        material_index = organ.component.material_index
        if material_index is not None and material_index not in materials and material_index < len(asset.materials):
            materials[material_index] = vine_usd._author_material(  # noqa: SLF001
                stage,
                looks.AppendChild(vine_usd._identifier(asset.materials[material_index].name)),  # noqa: SLF001
                asset.materials[material_index],
                textures,
            )

        path = Sdf.Path(link.path).AppendChild(f"Visual_{organ.index:04d}")
        mesh = UsdGeom.Mesh.Define(stage, path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(local.astype(np.float32)))
        mesh.CreateFaceVertexIndicesAttr(
            Vt.IntArray.FromNumpy(organ.component.triangles.ravel().astype(np.int32))
        )
        mesh.CreateFaceVertexCountsAttr(
            Vt.IntArray.FromNumpy(np.full(organ.component.num_triangles, 3, dtype=np.int32))
        )
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateExtentAttr(
            Vt.Vec3fArray.FromNumpy(np.vstack([local.min(axis=0), local.max(axis=0)]).astype(np.float32))
        )
        if organ.component.normals is not None:
            normals = to_stage_frame(organ.component.normals) @ rotation
            mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
            mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        if organ.component.uvs is not None:
            st = np.column_stack([organ.component.uvs[:, 0], 1.0 - organ.component.uvs[:, 1]]).astype(np.float32)
            primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
                "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
            )
            primvar.Set(Vt.Vec2fArray.FromNumpy(st))
        if material_index in materials:
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(materials[material_index])
            mesh.CreateDoubleSidedAttr(asset.materials[material_index].double_sided)
        attached += 1
    return attached


def _carrier_links(
    rig: vine_physics.PlantRig, plant: organs.Plant
) -> dict[int, vine_physics.Link]:
    """The body each organ's mesh rides on.

    Stem organs ride their own base link. Foliage and fruit have no bodies of
    their own -- they are mass hanging off a petiole -- so they ride the link of
    the nearest ancestor that does, which is what makes a severed leaf travel
    with its petiole.
    """
    bases: dict[int, vine_physics.Link] = {}
    for link in rig.links:
        current = bases.get(link.organ)
        if current is None or link.index < current.index:
            bases[link.organ] = link

    carriers: dict[int, vine_physics.Link] = {}
    for organ in plant.organs:
        index: int | None = organ.index
        while index is not None and index not in bases:
            index = plant.organs[index].parent
        if index is not None:
            carriers[organ.index] = bases[index]
    return carriers


def _write_textures(asset: glb.Glb, root_path: str) -> dict[int, str]:
    """Textures are shared with the converted vine asset, written beside it."""
    del root_path
    directory = pathlib.Path("data/greenhouse_sim/vines/textures")
    used = {
        material.base_color_image for material in asset.materials if material.base_color_image is not None
    }
    if not used:
        return {}
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for image in asset.images:
        if image.index not in used:
            continue
        name = f"{image.index:02d}_{vine_usd._identifier(image.name)}{image.suffix}"  # noqa: SLF001
        target = directory / name
        if not target.exists():
            target.write_bytes(image.data)
        paths[image.index] = target.resolve().as_posix()
    return paths
