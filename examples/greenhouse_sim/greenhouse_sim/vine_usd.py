"""Author a segmented vine as a structured, per-organ USD asset.

The written stage mirrors the plant's organ tree: every organ is an Xform whose
origin sits at its junction with its parent, carrying its geometry as a child
mesh and its child organs as child Xforms. That layout is what makes the rest of
the benchmark tractable -- severing a petiole is an operation on one subtree,
the junction is already the cut site and the natural joint anchor, and organ
prim paths give the scoring code stable identities to measure against.

Geometry is emitted directly in the greenhouse's Z-up metre frame rather than
relying on USD's unitsResolve fixups, which the hand-authored greenhouse objects
depend on and which make every downstream transform harder to reason about.
"""

from __future__ import annotations

import dataclasses
import pathlib

import numpy as np

from greenhouse_sim import glb
from greenhouse_sim import organs
from greenhouse_sim import usd_env

# USD is not on the import path under Isaac Sim's bare python, so it has to be
# located before pxr can be imported. See usd_env for why this is not a plain
# dependency.
usd_env.ensure_pxr()

from pxr import Gf  # noqa: E402
from pxr import Sdf  # noqa: E402
from pxr import Tf  # noqa: E402
from pxr import Usd  # noqa: E402
from pxr import UsdGeom  # noqa: E402
from pxr import UsdShade  # noqa: E402
from pxr import Vt  # noqa: E402

# glTF is Y-up, the greenhouse stage is Z-up: (x, y, z) -> (x, -z, y).
# Composed with the metadata mapping this is the identity, so the generator's
# Z-up attach points are already valid USD coordinates.
_GLTF_TO_USD = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])

_TEXTURE_DIR = "textures"


def gltf_to_usd(points: np.ndarray) -> np.ndarray:
    """Rotate glTF Y-up coordinates into the stage's Z-up frame."""
    return np.atleast_2d(np.asarray(points, dtype=np.float64)) @ _GLTF_TO_USD.T


def _identifier(name: str) -> str:
    return Tf.MakeValidIdentifier(name)


def _write_textures(asset: glb.Glb, used: set[int], directory: pathlib.Path) -> dict[int, str]:
    """Write the referenced textures beside the stage, returning relative paths.

    Image names repeat within a vine (four leaf textures share one name), so
    filenames are disambiguated by image index.
    """
    if not used:
        return {}
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for image in asset.images:
        if image.index not in used:
            continue
        filename = f"{image.index:02d}_{_identifier(image.name)}{image.suffix}"
        (directory / filename).write_bytes(image.data)
        paths[image.index] = f"./{_TEXTURE_DIR}/{filename}"
    return paths


def _author_material(
    stage: Usd.Stage, path: Sdf.Path, material: glb.Material, textures: dict[int, str]
) -> UsdShade.Material:
    shade_material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path.AppendChild("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(material.roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(material.metallic)

    texture_path = textures.get(material.base_color_image) if material.base_color_image is not None else None
    if texture_path is None:
        colour = Gf.Vec3f(*[float(c) for c in material.base_color[:3]])
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(colour)
    else:
        reader = UsdShade.Shader.Define(stage, path.AppendChild("stReader"))
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

        texture = UsdShade.Shader.Define(stage, path.AppendChild("diffuseTexture"))
        texture.CreateIdAttr("UsdUVTexture")
        texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(texture_path)
        texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
        texture.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")

    shade_material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return shade_material


def _author_mesh(
    stage: Usd.Stage,
    path: Sdf.Path,
    component: organs.Component,
    origin: np.ndarray,
    material: UsdShade.Material | None,
    *,
    double_sided: bool,
) -> UsdGeom.Mesh:
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = gltf_to_usd(component.vertices) - origin
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(component.triangles.ravel().astype(np.int32)))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(component.num_triangles, 3, dtype=np.int32)))
    # The source meshes are already triangulated art, not subdivision cages.
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(double_sided)
    mesh.CreateExtentAttr(
        Vt.Vec3fArray.FromNumpy(np.vstack([points.min(axis=0), points.max(axis=0)]).astype(np.float32))
    )

    if component.normals is not None:
        normals = gltf_to_usd(component.normals)
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(normals.astype(np.float32)))
        mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if component.uvs is not None:
        # glTF texture space runs V downward; USD expects it upward.
        st = np.column_stack([component.uvs[:, 0], 1.0 - component.uvs[:, 1]]).astype(np.float32)
        primvar = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
            "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex
        )
        primvar.Set(Vt.Vec2fArray.FromNumpy(st))

    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    return mesh


@dataclasses.dataclass(frozen=True)
class VineUsd:
    """Result of writing one vine."""

    path: pathlib.Path
    root_path: str
    organ_paths: dict[int, str]
    num_triangles: int


def write_vine(plant: organs.Plant, asset: glb.Glb, output: str | pathlib.Path) -> VineUsd:
    """Write `plant` as a per-organ USD asset at `output`."""
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.unlink(missing_ok=True)  # CreateNew refuses to overwrite.

    stage = Usd.Stage.CreateNew(str(output))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    root_name = _identifier(plant.name)
    root = UsdGeom.Xform.Define(stage, Sdf.Path(f"/{root_name}"))
    stage.SetDefaultPrim(root.GetPrim())

    used_materials = {o.component.material_index for o in plant.organs if o.component.material_index is not None}
    used_images = {
        asset.materials[i].base_color_image
        for i in used_materials
        if i < len(asset.materials) and asset.materials[i].base_color_image is not None
    }
    textures = _write_textures(asset, {i for i in used_images if i is not None}, output.parent / _TEXTURE_DIR)

    looks = Sdf.Path(f"/{root_name}/Looks")
    UsdGeom.Scope.Define(stage, looks)
    materials = {
        index: _author_material(
            stage, looks.AppendChild(_identifier(asset.materials[index].name)), asset.materials[index], textures
        )
        for index in sorted(used_materials)
        if index < len(asset.materials)
    }

    # Organ origins sit at the junction with the parent, so an organ's local
    # frame is anchored exactly where it will later be jointed and cut.
    origins: dict[int, np.ndarray] = {}
    paths: dict[int, str] = {}

    def organ_origin(organ: organs.Organ) -> np.ndarray:
        if organ.attachment is None:
            return np.zeros(3)
        return gltf_to_usd(organ.attachment)[0]

    def author(organ: organs.Organ, parent_path: Sdf.Path, parent_origin: np.ndarray) -> None:
        origin = organ_origin(organ)
        path = parent_path.AppendChild(_identifier(organ.label))
        xform = UsdGeom.Xform.Define(stage, path)
        xform.AddTranslateOp().Set(Gf.Vec3d(*(origin - parent_origin)))

        material_index = organ.component.material_index
        material = materials.get(material_index) if material_index is not None else None
        double_sided = bool(
            material_index is not None
            and material_index < len(asset.materials)
            and asset.materials[material_index].double_sided
        )
        _author_mesh(stage, path.AppendChild("Geom"), organ.component, origin, material, double_sided=double_sided)

        origins[organ.index] = origin
        paths[organ.index] = str(path)
        for child in plant.children_of(organ.index):
            author(plant.organs[child], path, origin)

    author(plant.organs[plant.root], root.GetPath(), np.zeros(3))

    stage.GetRootLayer().Save()
    return VineUsd(
        path=output,
        root_path=str(root.GetPath()),
        organ_paths=paths,
        num_triangles=sum(o.component.num_triangles for o in plant.organs),
    )
