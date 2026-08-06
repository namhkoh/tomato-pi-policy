"""Minimal glTF 2.0 binary (GLB) reader for the tomato vine assets.

Only the subset the vine assets actually use is supported: a single scene, a
single mesh whose primitives are indexed triangle lists, and POSITION/NORMAL/
TEXCOORD_0 attributes backed by one embedded buffer. Anything outside that
subset raises rather than being silently ignored, so an asset regenerated with
different export settings fails loudly instead of producing a wrong scene.

Reading GLB directly (rather than through trimesh or Isaac's asset converter)
keeps the pipeline's organ segmentation working on raw vertex data, which is
what the segmentation in ``organs.py`` needs.
"""

from __future__ import annotations

import dataclasses
import json
import pathlib
import struct
from typing import Any

import numpy as np

_MAGIC = b"glTF"
_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942

# glTF componentType -> numpy dtype.
_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}

# glTF accessor type -> number of components.
_TYPE_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}

_MODE_TRIANGLES = 4


class GlbError(Exception):
    """Raised when a GLB file is malformed or uses unsupported features."""


@dataclasses.dataclass(frozen=True)
class Image:
    """A texture image embedded in the binary chunk."""

    index: int
    name: str
    mime_type: str
    data: bytes

    @property
    def suffix(self) -> str:
        return ".jpg" if self.mime_type == "image/jpeg" else ".png"


@dataclasses.dataclass(frozen=True)
class Material:
    """The subset of a glTF PBR material that survives into UsdPreviewSurface."""

    index: int
    name: str
    base_color: tuple[float, float, float, float]
    metallic: float
    roughness: float
    base_color_image: int | None
    double_sided: bool


@dataclasses.dataclass(frozen=True)
class Primitive:
    """One material-homogeneous triangle batch of a mesh.

    The vine exporter fuses every organ sharing a material into a single
    primitive, so a primitive is a material group and not a plant part.
    """

    index: int
    material: str
    positions: np.ndarray  # (n_vertices, 3) float64, glTF Y-up metres
    triangles: np.ndarray  # (n_triangles, 3) int64 into positions
    material_index: int | None = None
    normals: np.ndarray | None = None  # (n_vertices, 3) float64
    uvs: np.ndarray | None = None  # (n_vertices, 2) float64

    @property
    def num_triangles(self) -> int:
        return int(self.triangles.shape[0])


@dataclasses.dataclass(frozen=True)
class Glb:
    """A parsed GLB file."""

    path: pathlib.Path
    primitives: list[Primitive]
    generator: str
    materials: list[Material] = dataclasses.field(default_factory=list)
    images: list[Image] = dataclasses.field(default_factory=list)

    @property
    def num_triangles(self) -> int:
        return sum(p.num_triangles for p in self.primitives)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Axis-aligned bounds over every primitive, as (lower, upper)."""
        lower = np.min([p.positions.min(axis=0) for p in self.primitives], axis=0)
        upper = np.max([p.positions.max(axis=0) for p in self.primitives], axis=0)
        return lower, upper


def _read_chunks(data: bytes, path: pathlib.Path) -> tuple[dict[str, Any], bytes]:
    if data[:4] != _MAGIC:
        raise GlbError(f"{path}: not a GLB file (bad magic {data[:4]!r})")
    version, total_length = struct.unpack_from("<II", data, 4)
    if version != 2:
        raise GlbError(f"{path}: unsupported GLB version {version}, expected 2")
    if total_length != len(data):
        raise GlbError(f"{path}: header length {total_length} != file size {len(data)}")

    gltf: dict[str, Any] | None = None
    buffer: bytes | None = None
    offset = 12
    while offset < len(data):
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset : offset + chunk_length]
        if len(chunk) != chunk_length:
            raise GlbError(f"{path}: truncated chunk at offset {offset}")
        offset += chunk_length
        if chunk_type == _CHUNK_JSON and gltf is None:
            gltf = json.loads(chunk.decode("utf-8"))
        elif chunk_type == _CHUNK_BIN and buffer is None:
            buffer = chunk

    if gltf is None:
        raise GlbError(f"{path}: no JSON chunk")
    if buffer is None:
        raise GlbError(f"{path}: no binary chunk; external buffers are not supported")
    return gltf, buffer


def _read_accessor(gltf: dict[str, Any], buffer: bytes, accessor_index: int, path: pathlib.Path) -> np.ndarray:
    """Read one accessor into an (count, n_components) array."""
    accessor = gltf["accessors"][accessor_index]
    if "bufferView" not in accessor:
        raise GlbError(f"{path}: sparse accessors are not supported")
    if accessor.get("sparse"):
        raise GlbError(f"{path}: sparse accessors are not supported")

    dtype = _COMPONENT_DTYPES.get(accessor["componentType"])
    if dtype is None:
        raise GlbError(f"{path}: unknown componentType {accessor['componentType']}")
    num_components = _TYPE_COMPONENTS[accessor["type"]]
    count = accessor["count"]
    element_size = np.dtype(dtype).itemsize * num_components

    view = gltf["bufferViews"][accessor["bufferView"]]
    if view.get("buffer", 0) != 0:
        raise GlbError(f"{path}: only the embedded buffer 0 is supported")
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    stride = view.get("byteStride", element_size)

    if stride == element_size:
        flat = np.frombuffer(buffer, dtype=dtype, count=count * num_components, offset=start)
        return flat.reshape(count, num_components)
    # Interleaved: copy out the leading element_size bytes of each stride.
    raw = np.frombuffer(buffer, dtype=np.uint8, count=stride * count, offset=start).reshape(count, stride)
    return raw[:, :element_size].copy().view(dtype).reshape(count, num_components)


def _read_images(gltf: dict[str, Any], buffer: bytes, path: pathlib.Path) -> list[Image]:
    images = []
    for index, image in enumerate(gltf.get("images", [])):
        if "uri" in image:
            raise GlbError(f"{path}: image {index} is external; only embedded textures are supported")
        view = gltf["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        data = buffer[start : start + view["byteLength"]]
        images.append(
            Image(
                index=index,
                name=image.get("name", f"image_{index}"),
                mime_type=image.get("mimeType", "image/png"),
                data=data,
            )
        )
    return images


def _read_materials(gltf: dict[str, Any]) -> list[Material]:
    textures = gltf.get("textures", [])
    materials = []
    for index, material in enumerate(gltf.get("materials", [])):
        pbr = material.get("pbrMetallicRoughness", {})
        texture_index = pbr.get("baseColorTexture", {}).get("index")
        image_index = textures[texture_index].get("source") if texture_index is not None else None
        materials.append(
            Material(
                index=index,
                name=material.get("name", f"material_{index}"),
                base_color=tuple(pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0])),
                metallic=float(pbr.get("metallicFactor", 1.0)),
                roughness=float(pbr.get("roughnessFactor", 1.0)),
                base_color_image=image_index,
                double_sided=bool(material.get("doubleSided", False)),
            )
        )
    return materials


def read_glb(path: str | pathlib.Path) -> Glb:
    """Parse a GLB file into its material-split triangle primitives.

    Raises GlbError if the file is malformed or uses features the vine assets
    are not expected to contain (multiple meshes, node transforms, skins,
    animations, morph targets, or non-triangle topology).
    """
    path = pathlib.Path(path)
    gltf, buffer = _read_chunks(path.read_bytes(), path)

    for unsupported in ("skins", "animations"):
        if gltf.get(unsupported):
            raise GlbError(f"{path}: {unsupported} are not supported")

    meshes = gltf.get("meshes", [])
    if len(meshes) != 1:
        raise GlbError(f"{path}: expected exactly 1 mesh, found {len(meshes)}")

    # The vine exporter bakes geometry into world space under a single
    # transform-free node. A transform here would silently offset every organ
    # relative to the metadata attach points, so reject it.
    for node in gltf.get("nodes", []):
        for key in ("matrix", "translation", "rotation", "scale"):
            if key in node:
                raise GlbError(f"{path}: node {node.get('name', '?')} has an unsupported {key} transform")

    materials = _read_materials(gltf)
    images = _read_images(gltf, buffer, path)

    primitives = []
    for index, primitive in enumerate(meshes[0].get("primitives", [])):
        if primitive.get("mode", _MODE_TRIANGLES) != _MODE_TRIANGLES:
            raise GlbError(f"{path}: primitive {index} is not a triangle list")
        if primitive.get("targets"):
            raise GlbError(f"{path}: primitive {index} has morph targets")
        if "indices" not in primitive:
            raise GlbError(f"{path}: primitive {index} is not indexed")

        attributes = primitive["attributes"]
        positions = _read_accessor(gltf, buffer, attributes["POSITION"], path).astype(np.float64)
        indices = _read_accessor(gltf, buffer, primitive["indices"], path).ravel().astype(np.int64)
        if indices.size % 3:
            raise GlbError(f"{path}: primitive {index} index count {indices.size} is not a multiple of 3")

        normals = None
        if "NORMAL" in attributes:
            normals = _read_accessor(gltf, buffer, attributes["NORMAL"], path).astype(np.float64)
        uvs = None
        if "TEXCOORD_0" in attributes:
            uvs = _read_accessor(gltf, buffer, attributes["TEXCOORD_0"], path).astype(np.float64)

        material_index = primitive.get("material")
        name = (
            materials[material_index].name
            if material_index is not None and material_index < len(materials)
            else "<none>"
        )
        primitives.append(
            Primitive(
                index=index,
                material=name,
                positions=positions,
                triangles=indices.reshape(-1, 3),
                material_index=material_index,
                normals=normals,
                uvs=uvs,
            )
        )

    if not primitives:
        raise GlbError(f"{path}: mesh has no primitives")

    return Glb(
        path=path,
        primitives=primitives,
        generator=gltf.get("asset", {}).get("generator", ""),
        materials=materials,
        images=images,
    )
