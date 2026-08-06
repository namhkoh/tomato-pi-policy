"""Tests for centreline fitting, on tubes whose true axis and radius are known."""

from __future__ import annotations

import numpy as np
import pytest

from greenhouse_sim import organs
from greenhouse_sim import skeleton


def _tube(centres: np.ndarray, radius: float, sides: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """Triangulated tube through an ordered list of centreline points."""
    vertices, triangles = [], []
    for index, centre in enumerate(centres):
        if index == 0:
            axis = centres[1] - centres[0]
        elif index == len(centres) - 1:
            axis = centres[-1] - centres[-2]
        else:
            axis = centres[index + 1] - centres[index - 1]
        axis = axis / np.linalg.norm(axis)
        seed = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(axis, seed)
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        angles = np.linspace(0, 2 * np.pi, sides, endpoint=False)
        vertices.append(centre + radius * (np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v)))

    for ring in range(len(centres) - 1):
        base, nxt = ring * sides, (ring + 1) * sides
        for i in range(sides):
            j = (i + 1) % sides
            triangles.append([base + i, base + j, nxt + i])
            triangles.append([base + j, nxt + j, nxt + i])
    return np.vstack(vertices), np.array(triangles, dtype=np.int64)


def _component(vertices: np.ndarray, triangles: np.ndarray) -> organs.Component:
    return organs.Component(
        index=0,
        material="TomatoStem",
        tissue=organs.Tissue.STEM,
        vertices=vertices,
        triangles=triangles,
    )


def test_straight_tube_recovers_axis_and_radius() -> None:
    centres = np.column_stack([np.zeros(60), np.zeros(60), np.linspace(0, 0.30, 60)])
    fitted = skeleton.extract_skeleton(_component(*_tube(centres, radius=0.004)), np.zeros(3))

    assert fitted.num_segments >= 8
    assert fitted.length == pytest.approx(0.30, abs=0.02)
    # The tube is on the Z axis, so X and Y stay near zero along the centreline.
    assert np.abs(fitted.points[:, :2]).max() < 1e-3
    np.testing.assert_allclose(fitted.radii, 0.004, atol=5e-4)


def test_centreline_is_ordered_from_the_base() -> None:
    """Nodes must run base-to-tip: stub length depends on that ordering."""
    centres = np.column_stack([np.zeros(60), np.zeros(60), np.linspace(0, 0.30, 60)])
    fitted = skeleton.extract_skeleton(_component(*_tube(centres, radius=0.004)), np.zeros(3))
    heights = fitted.points[:, 2]
    assert np.all(np.diff(heights) > 0)
    assert heights[0] < 0.02


def test_base_point_selects_which_end_is_the_base() -> None:
    centres = np.column_stack([np.zeros(60), np.zeros(60), np.linspace(0, 0.30, 60)])
    component = _component(*_tube(centres, radius=0.004))
    from_far_end = skeleton.extract_skeleton(component, np.array([0.0, 0.0, 0.30]))
    assert from_far_end.points[0, 2] > from_far_end.points[-1, 2]


def test_curved_tube_follows_its_own_length() -> None:
    """A tube curving back on itself must not be folded onto itself."""
    angle = np.linspace(0, np.pi, 80)
    centres = np.column_stack([0.1 * np.sin(angle), np.zeros(80), 0.1 * (1 - np.cos(angle))])
    fitted = skeleton.extract_skeleton(_component(*_tube(centres, radius=0.003)), centres[0])

    expected = float(np.linalg.norm(np.diff(centres, axis=0), axis=1).sum())
    assert fitted.length == pytest.approx(expected, rel=0.15)
    # Arc length must exceed the straight-line chord: a semicircle measured
    # through space rather than along the surface would collapse to the chord.
    chord = float(np.linalg.norm(fitted.points[-1] - fitted.points[0]))
    assert chord == pytest.approx(0.2, abs=0.02)
    assert fitted.length > 1.4 * chord


def test_arc_length_sampling_is_continuous() -> None:
    centres = np.column_stack([np.zeros(60), np.zeros(60), np.linspace(0, 0.30, 60)])
    fitted = skeleton.extract_skeleton(_component(*_tube(centres, radius=0.004)), np.zeros(3))

    # Interpolation, not snapping: a cut measured between nodes stays exact.
    midpoint = fitted.point_at(0.5 * fitted.length)
    assert midpoint[2] == pytest.approx(0.5 * fitted.length, abs=0.01)
    np.testing.assert_allclose(fitted.point_at(0.0), fitted.points[0], atol=1e-12)
    np.testing.assert_allclose(fitted.point_at(1e6), fitted.points[-1], atol=1e-12)
    assert fitted.radius_at(0.5 * fitted.length) == pytest.approx(0.004, abs=5e-4)


def test_rejects_degenerate_organ() -> None:
    vertices = np.zeros((4, 3))
    triangles = np.array([[0, 1, 2]], dtype=np.int64)
    with pytest.raises(skeleton.SkeletonError):
        skeleton.extract_skeleton(_component(vertices, triangles), np.zeros(3))


def test_skeletonise_plant_skips_foliage() -> None:
    centres = np.column_stack([np.zeros(60), np.zeros(60), np.linspace(0, 0.30, 60)])
    stem_vertices, stem_triangles = _tube(centres, radius=0.004)
    stem = _component(stem_vertices, stem_triangles)
    leaf = organs.Component(
        index=1,
        material="Material.001",
        tissue=organs.Tissue.FOLIAGE,
        vertices=stem_vertices,
        triangles=stem_triangles,
    )
    plant = organs.Plant(
        name="synthetic",
        organs=[
            organs.Organ(index=0, component=stem, parent=None, attachment=None, attachment_gap=0.0, label="MainStem"),
            organs.Organ(
                index=1, component=leaf, parent=0, attachment=np.zeros(3), attachment_gap=0.0, label="Foliage_001"
            ),
        ],
        root=0,
        metadata={},
    )
    skeletons = skeleton.skeletonise_plant(plant)
    assert set(skeletons) == {0}
