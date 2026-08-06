"""Tests for the GLB reader, built on synthetic files so they stay hermetic."""

from __future__ import annotations

import json
import pathlib
import struct
from typing import Any

import numpy as np
import pytest

from greenhouse_sim import glb


def _pad4(data: bytes, fill: bytes) -> bytes:
    return data + fill * (-len(data) % 4)


def _build_glb(gltf: dict[str, Any], buffer: bytes) -> bytes:
    json_chunk = _pad4(json.dumps(gltf).encode("utf-8"), b" ")
    bin_chunk = _pad4(buffer, b"\0")
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)
    return b"".join(
        [
            struct.pack("<4sII", b"glTF", 2, total),
            struct.pack("<II", len(json_chunk), 0x4E4F534A),
            json_chunk,
            struct.pack("<II", len(bin_chunk), 0x004E4942),
            bin_chunk,
        ]
    )


def _two_triangle_gltf(positions: np.ndarray, indices: np.ndarray) -> tuple[dict[str, Any], bytes]:
    """One primitive holding `positions` and `indices`, tightly packed."""
    position_bytes = positions.astype(np.float32).tobytes()
    index_bytes = indices.astype(np.uint32).tobytes()
    gltf = {
        "asset": {"version": "2.0", "generator": "test"},
        "materials": [{"name": "TomatoStem"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "nodes": [{"mesh": 0, "name": "plant"}],
        "scenes": [{"nodes": [0]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": len(positions), "type": "VEC3"},
            {"bufferView": 1, "componentType": 5125, "count": indices.size, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes)},
            {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(index_bytes)},
        ],
        "buffers": [{"byteLength": len(position_bytes) + len(index_bytes)}],
    }
    return gltf, position_bytes + index_bytes


def _write(tmp_path: pathlib.Path, gltf: dict[str, Any], buffer: bytes) -> pathlib.Path:
    path = tmp_path / "asset.glb"
    path.write_bytes(_build_glb(gltf, buffer))
    return path


def test_reads_positions_and_triangles(tmp_path: pathlib.Path) -> None:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], dtype=np.float64)
    indices = np.array([0, 1, 2], dtype=np.uint32)
    asset = glb.read_glb(_write(tmp_path, *_two_triangle_gltf(positions, indices)))

    assert len(asset.primitives) == 1
    primitive = asset.primitives[0]
    assert primitive.material == "TomatoStem"
    assert primitive.num_triangles == 1
    np.testing.assert_allclose(primitive.positions, positions)
    np.testing.assert_array_equal(primitive.triangles, [[0, 1, 2]])
    assert asset.generator == "test"


def test_bounds_spans_all_primitives(tmp_path: pathlib.Path) -> None:
    positions = np.array([[-1, 0, 2], [1, 0, 0], [0, 3, 0]], dtype=np.float64)
    asset = glb.read_glb(_write(tmp_path, *_two_triangle_gltf(positions, np.array([0, 1, 2], dtype=np.uint32))))
    lower, upper = asset.bounds()
    np.testing.assert_allclose(lower, [-1, 0, 0])
    np.testing.assert_allclose(upper, [1, 3, 2])


def test_honours_byte_stride(tmp_path: pathlib.Path) -> None:
    """Interleaved buffers must be de-interleaved, not read as tightly packed."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
    # Interleave POSITION (12 bytes) with a padding VEC3 of junk, stride 24.
    interleaved = np.zeros((3, 6), dtype=np.float32)
    interleaved[:, :3] = positions
    interleaved[:, 3:] = 99.0
    position_bytes = interleaved.tobytes()
    index_bytes = np.array([0, 1, 2], dtype=np.uint32).tobytes()

    gltf = {
        "asset": {"version": "2.0"},
        "materials": [{"name": "M"}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "material": 0}]}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3"},
            {"bufferView": 1, "componentType": 5125, "count": 3, "type": "SCALAR"},
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes), "byteStride": 24},
            {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(index_bytes)},
        ],
        "buffers": [{"byteLength": len(position_bytes) + len(index_bytes)}],
    }
    asset = glb.read_glb(_write(tmp_path, gltf, position_bytes + index_bytes))
    np.testing.assert_allclose(asset.primitives[0].positions, positions)


def test_reads_normals_and_uvs(tmp_path: pathlib.Path) -> None:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    normals = np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)
    uvs = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    gltf, buffer = _two_triangle_gltf(positions, np.array([0, 1, 2], dtype=np.uint32))

    offset = len(buffer)
    buffer = buffer + normals.tobytes() + uvs.tobytes()
    gltf["bufferViews"] += [
        {"buffer": 0, "byteOffset": offset, "byteLength": normals.nbytes},
        {"buffer": 0, "byteOffset": offset + normals.nbytes, "byteLength": uvs.nbytes},
    ]
    gltf["accessors"] += [
        {"bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC3"},
        {"bufferView": 3, "componentType": 5126, "count": 3, "type": "VEC2"},
    ]
    gltf["meshes"][0]["primitives"][0]["attributes"].update({"NORMAL": 2, "TEXCOORD_0": 3})

    primitive = glb.read_glb(_write(tmp_path, gltf, buffer)).primitives[0]
    np.testing.assert_allclose(primitive.normals, normals)
    np.testing.assert_allclose(primitive.uvs, uvs)


def test_reads_materials_and_embedded_textures(tmp_path: pathlib.Path) -> None:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    gltf, buffer = _two_triangle_gltf(positions, np.array([0, 1, 2], dtype=np.uint32))

    texture = b"\x89PNG\r\n\x1a\n" + b"pretend-png-bytes"
    offset = len(buffer)
    buffer = buffer + texture
    gltf["bufferViews"].append({"buffer": 0, "byteOffset": offset, "byteLength": len(texture)})
    gltf["images"] = [{"name": "leaf", "mimeType": "image/png", "bufferView": 2}]
    gltf["textures"] = [{"source": 0}]
    gltf["materials"] = [
        {
            "name": "Leaf",
            "doubleSided": True,
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8,
            },
        }
    ]

    asset = glb.read_glb(_write(tmp_path, gltf, buffer))
    material = asset.materials[0]
    assert material.name == "Leaf"
    assert material.base_color_image == 0
    assert material.roughness == pytest.approx(0.8)
    assert material.double_sided
    assert asset.images[0].data == texture
    assert asset.images[0].suffix == ".png"
    assert asset.primitives[0].material_index == 0


def test_rejects_external_texture(tmp_path: pathlib.Path) -> None:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    gltf, buffer = _two_triangle_gltf(positions, np.array([0, 1, 2], dtype=np.uint32))
    gltf["images"] = [{"uri": "leaf.png"}]
    with pytest.raises(glb.GlbError, match="external"):
        glb.read_glb(_write(tmp_path, gltf, buffer))


def test_rejects_bad_magic(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "bad.glb"
    path.write_bytes(b"NOPE" + b"\0" * 20)
    with pytest.raises(glb.GlbError, match="bad magic"):
        glb.read_glb(path)


def test_rejects_node_transform(tmp_path: pathlib.Path) -> None:
    """A node transform would offset geometry away from the metadata attach points."""
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    gltf, buffer = _two_triangle_gltf(positions, np.array([0, 1, 2], dtype=np.uint32))
    gltf["nodes"][0]["translation"] = [0.0, 1.0, 0.0]
    with pytest.raises(glb.GlbError, match="translation"):
        glb.read_glb(_write(tmp_path, gltf, buffer))


def test_rejects_non_triangle_topology(tmp_path: pathlib.Path) -> None:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    gltf, buffer = _two_triangle_gltf(positions, np.array([0, 1, 2], dtype=np.uint32))
    gltf["meshes"][0]["primitives"][0]["mode"] = 1  # LINES
    with pytest.raises(glb.GlbError, match="not a triangle list"):
        glb.read_glb(_write(tmp_path, gltf, buffer))


def test_rejects_multiple_meshes(tmp_path: pathlib.Path) -> None:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    gltf, buffer = _two_triangle_gltf(positions, np.array([0, 1, 2], dtype=np.uint32))
    gltf["meshes"].append(dict(gltf["meshes"][0]))
    with pytest.raises(glb.GlbError, match="expected exactly 1 mesh"):
        glb.read_glb(_write(tmp_path, gltf, buffer))


def test_rejects_skins(tmp_path: pathlib.Path) -> None:
    positions = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    gltf, buffer = _two_triangle_gltf(positions, np.array([0, 1, 2], dtype=np.uint32))
    gltf["skins"] = [{"joints": [0]}]
    with pytest.raises(glb.GlbError, match="skins"):
        glb.read_glb(_write(tmp_path, gltf, buffer))
