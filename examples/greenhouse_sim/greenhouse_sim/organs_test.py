"""Tests for organ segmentation.

Built on a synthetic plant with known topology, so a regression points at the
reconstruction rather than at a change in the art assets.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from greenhouse_sim import glb
from greenhouse_sim import organs

# Adjacency is measured between vertices, so the fixtures must be tessellated
# about as finely as the real assets (sub-millimetre spacing along the stem).
_RING_SPACING_M = 0.002


def _tube(start: np.ndarray, end: np.ndarray, radius: float, sides: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """A triangulated tube from `start` to `end`, ringed along its length."""
    axis = end - start
    length = float(np.linalg.norm(axis))
    axis = axis / length
    # Any vector not parallel to the axis yields a usable radial basis.
    seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(axis, seed)
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)

    angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
    ring = np.outer(np.cos(angles), u) * radius + np.outer(np.sin(angles), v) * radius
    num_rings = max(2, round(length / _RING_SPACING_M) + 1)
    centres = start + np.outer(np.linspace(0.0, length, num_rings), axis)
    vertices = np.vstack([centre + ring for centre in centres])

    triangles = []
    for r in range(num_rings - 1):
        base, nxt = r * sides, (r + 1) * sides
        for i in range(sides):
            j = (i + 1) % sides
            triangles.append([base + i, base + j, nxt + i])
            triangles.append([base + j, nxt + j, nxt + i])
    return vertices, np.array(triangles, dtype=np.int64)


def _merge(parts: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    vertices, triangles, offset = [], [], 0
    for part_vertices, part_triangles in parts:
        vertices.append(part_vertices)
        triangles.append(part_triangles + offset)
        offset += part_vertices.shape[0]
    return np.vstack(vertices), np.vstack(triangles)


def _primitive(index: int, material: str, parts: list[tuple[np.ndarray, np.ndarray]]) -> glb.Primitive:
    vertices, triangles = _merge(parts)
    return glb.Primitive(index=index, material=material, positions=vertices, triangles=triangles)


def _synthetic_plant() -> tuple[glb.Glb, dict[str, Any]]:
    """A vertical main stem with two petioles, each carrying one leaflet.

    Petiole 1 droops back so it nearly touches petiole 0, reproducing the
    spurious contact that a distance-only parent search would trip over.
    """
    main = _tube(np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 0.02)
    petiole_low = _tube(np.array([0.0, 0.30, 0.0]), np.array([0.0, 0.32, 0.20]), 0.004)
    petiole_high = _tube(np.array([0.0, 0.40, 0.0]), np.array([0.0, 0.33, 0.20]), 0.004)
    stem = _primitive(0, "TomatoStem", [main, petiole_low, petiole_high])

    leaf_low = _tube(np.array([0.0, 0.32, 0.20]), np.array([0.0, 0.32, 0.30]), 0.03)
    leaf_high = _tube(np.array([0.0, 0.33, 0.20]), np.array([0.0, 0.34, 0.30]), 0.03)
    foliage = _primitive(1, "Material.001", [leaf_low, leaf_high])

    asset = glb.Glb(path=type("P", (), {"stem": "synthetic"})(), primitives=[stem, foliage], generator="test")

    # Metadata is Blender Z-up: (x, y, z)_meta -> (x, z, -y)_glTF.
    metadata = {
        "counts": {"sub_stems": 2, "leaves": 2},
        "attachments": [
            {"id": "SubStem_00", "parent": "MainStem", "attach": [0.0, 0.0, 0.30]},
            {"id": "SubStem_01", "parent": "MainStem", "attach": [0.0, 0.0, 0.40]},
        ],
    }
    return asset, metadata


def test_metadata_frame_conversion() -> None:
    converted = organs.metadata_to_gltf(np.array([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(converted, [[1.0, 3.0, -2.0]])


def test_classify_tissue() -> None:
    assert organs.classify_tissue("TomatoStem") is organs.Tissue.STEM
    assert organs.classify_tissue("FruitRipe_1_2") is organs.Tissue.FRUIT
    assert organs.classify_tissue("Material.091") is organs.Tissue.FOLIAGE


def test_weld_collapses_coincident_vertices() -> None:
    positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    welded = organs.weld_vertices(positions)
    assert welded[0] == welded[2]
    assert welded[0] != welded[1]


def test_split_components_separates_disjoint_surfaces() -> None:
    stem, _ = _synthetic_plant()
    components = organs.split_components(stem.primitives[0])
    assert len(components) == 3  # main stem plus two petioles
    assert all(c.tissue is organs.Tissue.STEM for c in components)
    # Splitting must preserve the triangle budget exactly.
    assert sum(c.num_triangles for c in components) == stem.primitives[0].num_triangles


def test_segmentation_recovers_plant_topology() -> None:
    asset, metadata = _synthetic_plant()
    plant = organs.segment_plant(asset, metadata)

    assert len(plant.organs) == 5  # main stem, 2 petioles, 2 leaflets
    root = plant.organs[plant.root]
    assert root.label == "MainStem"
    assert root.parent is None

    petioles = [o for o in plant.organs if o.label.startswith("SubStem")]
    assert len(petioles) == 2
    # Both petioles hang off the main stem: the near-contact between the two
    # must not make one the parent of the other.
    assert all(o.parent == plant.root for o in petioles)

    foliage = [o for o in plant.organs if o.tissue is organs.Tissue.FOLIAGE]
    assert len(foliage) == 2
    assert all(o.parent in {p.index for p in petioles} for o in foliage)


def test_labels_follow_metadata_heights() -> None:
    """SubStem_00 sits lower on the stem than SubStem_01."""
    asset, metadata = _synthetic_plant()
    plant = organs.segment_plant(asset, metadata)
    by_label = {o.label: o for o in plant.organs if o.label.startswith("SubStem")}
    assert by_label["SubStem_00"].attachment[1] < by_label["SubStem_01"].attachment[1]


def test_descendants_span_the_cut_subtree() -> None:
    asset, metadata = _synthetic_plant()
    plant = organs.segment_plant(asset, metadata)
    petiole = next(o for o in plant.organs if o.label == "SubStem_00")

    detached = plant.descendants_of(petiole.index)
    assert petiole.index in detached
    # Cutting one petiole removes it and its leaflet, and nothing else.
    assert len(detached) == 2
    assert plant.root not in detached


def test_stem_junctions_are_contacts() -> None:
    """Stem organs are joined through surface adjacency, so their gaps are tiny.

    Foliage is deliberately excluded: it attaches by unbounded nearest-stem
    search, so a blade flaring away from its petiole tip may legitimately sit
    further out than the adjacency radius.
    """
    asset, metadata = _synthetic_plant()
    plant = organs.segment_plant(asset, metadata)
    stem_organs = [o for o in plant.organs if o.tissue is organs.Tissue.STEM and o.parent is not None]
    assert stem_organs
    for organ in stem_organs:
        assert organ.attachment_gap < organs.ADJACENCY_RADIUS_M


def test_rejects_mesh_without_stem_tissue() -> None:
    leaf = _primitive(0, "Material.001", [_tube(np.zeros(3), np.array([0.0, 0.1, 0.0]), 0.01)])
    asset = glb.Glb(path=type("P", (), {"stem": "leafy"})(), primitives=[leaf], generator="test")
    with pytest.raises(organs.SegmentationError, match="no stem tissue"):
        organs.segment_plant(asset, {"attachments": []})
