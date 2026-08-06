"""Fit centrelines to stem organs so they can be simulated as capsule chains.

The vine assets are render meshes: a petiole is a few thousand triangles of
tube, with no axis, radius or direction recorded anywhere. Physics needs the
opposite -- a thin ordered sequence of positions and radii it can turn into
capsule links and compliant joints, and that the scoring code can measure a cut
plane against.

Organs are traversed outward from their junction with the parent, so the
centreline is ordered base-to-tip by construction. That ordering is what makes
"how much petiole is left attached to the stem" a prefix length rather than a
search, which is the benchmark's headline measurement.

Distance is measured across the mesh surface rather than through space, so a
petiole that curves back toward the stem is still parameterised along its own
length instead of being folded onto itself.

Known limitation: a node's samples span one segment of surface distance, so on
an organ thick enough that its circumference approaches the segment length the
first node is still a partial ring and its radius reads low. Measured on the
main stem (~7 mm true radius) this recovers ~5.7 mm at the very base and is
accurate above it. Petioles, whose circumference is well under one segment, are
unaffected -- which is what matters, since they carry the cut sites.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

from greenhouse_sim import organs

# Capsule length along a petiole. Short enough that a chain bends smoothly and
# stub length is not coarsely quantised, long enough to keep link counts and
# solver cost sane on a plant with ~20 petioles.
DEFAULT_SEGMENT_LENGTH_M = 0.02

# Below this many vertices an organ is not a resolvable tube.
_MIN_VERTICES = 12

# Organ geometry is still in the GLB's Y-up frame at this stage.
_GLTF_UP = 1

# Vertices seeded as the organ's base ring: enough to wrap the tube once
# without reaching so far along it that the first node is displaced.
_SEED_FRACTION = 0.02
_MIN_SEEDS = 8


class SkeletonError(Exception):
    """Raised when an organ cannot be reduced to a centreline."""


@dataclasses.dataclass(frozen=True)
class Skeleton:
    """A centreline through one organ, ordered from its base outward."""

    points: np.ndarray  # (n_nodes, 3) float64, same frame as the source organ
    radii: np.ndarray  # (n_nodes,) float64

    @property
    def num_segments(self) -> int:
        return int(self.points.shape[0]) - 1

    @property
    def length(self) -> float:
        """Arc length along the centreline."""
        if self.points.shape[0] < 2:
            return 0.0
        return float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())

    def segment_lengths(self) -> np.ndarray:
        return np.linalg.norm(np.diff(self.points, axis=0), axis=1)

    def arc_lengths(self) -> np.ndarray:
        """Cumulative distance from the base at each node."""
        return np.concatenate([[0.0], np.cumsum(self.segment_lengths())])

    def point_at(self, arc_length: float) -> np.ndarray:
        """Position at `arc_length` from the base, clamped to the ends.

        Interpolating rather than snapping to a node keeps a measured cut
        position continuous, so stub length reflects where the blade actually
        crossed instead of the capsule resolution.
        """
        arcs = self.arc_lengths()
        clamped = float(np.clip(arc_length, 0.0, arcs[-1]))
        index = int(np.searchsorted(arcs, clamped, side="right")) - 1
        index = max(0, min(index, self.points.shape[0] - 2))
        span = arcs[index + 1] - arcs[index]
        blend = 0.0 if span <= 0 else (clamped - arcs[index]) / span
        return self.points[index] + blend * (self.points[index + 1] - self.points[index])

    def radius_at(self, arc_length: float) -> float:
        arcs = self.arc_lengths()
        return float(np.interp(np.clip(arc_length, 0.0, arcs[-1]), arcs, self.radii))


def _surface_distances(vertices: np.ndarray, triangles: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """Geodesic distance from the nearest of `sources` to every vertex.

    Seeded from a set rather than a point because a single seed makes the first
    bins one-sided patches of the tube wall instead of rings, which biases the
    fitted centre sideways and understates the radius. That error would land
    exactly at an organ's base -- where petioles are cut and stub length is
    measured -- so it is worth a virtual zero-cost super-source to avoid.
    """
    count = vertices.shape[0]
    edges = np.vstack([triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]])
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)

    rows = np.concatenate([edges[:, 0], np.full(sources.size, count)])
    cols = np.concatenate([edges[:, 1], sources])
    weights = np.concatenate([lengths, np.zeros(sources.size)])
    graph = scipy.sparse.coo_matrix((weights, (rows, cols)), shape=(count + 1, count + 1))
    return scipy.sparse.csgraph.dijkstra(graph.tocsr(), directed=False, indices=count)[:count]


def extract_skeleton(
    component: organs.Component,
    base_point: np.ndarray | None = None,
    *,
    segment_length: float = DEFAULT_SEGMENT_LENGTH_M,
) -> Skeleton:
    """Fit a base-to-tip centreline through one organ.

    `base_point` is the organ's junction with its parent; without it the
    traversal starts from the vertex furthest from the centroid, which is only
    meaningful for a free-standing organ.
    """
    welded = organs.weld_vertices(component.vertices)
    num_welded = int(welded.max()) + 1
    if num_welded < _MIN_VERTICES:
        raise SkeletonError(f"organ has only {num_welded} distinct vertices")

    # Collapse to welded positions so the surface graph is actually connected;
    # the render mesh is split along UV seams.
    positions = np.zeros((num_welded, 3))
    positions[welded] = component.vertices
    triangles = welded[component.triangles]

    if base_point is None:
        centroid = positions.mean(axis=0)
        anchor = positions[int(np.argmax(np.linalg.norm(positions - centroid, axis=1)))]
    else:
        anchor = np.asarray(base_point, dtype=np.float64)

    # Seed the whole base ring, approximated by the vertices nearest the anchor.
    seed_count = max(_MIN_SEEDS, int(_SEED_FRACTION * num_welded))
    sources = np.argsort(np.linalg.norm(positions - anchor, axis=1))[:seed_count]

    distances = _surface_distances(positions, triangles, sources)
    reachable = np.isfinite(distances)
    if reachable.sum() < _MIN_VERTICES:
        raise SkeletonError("organ surface is not connected enough to traverse")

    positions = positions[reachable]
    distances = distances[reachable]

    span = float(distances.max())
    if span <= 0:
        raise SkeletonError("organ has zero surface extent")

    # One node per segment, plus the endpoints.
    num_nodes = max(2, round(span / segment_length) + 1)
    edges = np.linspace(0.0, span, num_nodes + 1)
    bins = np.clip(np.digitize(distances, edges) - 1, 0, num_nodes - 1)

    centres, groups = [], []
    for node in range(num_nodes):
        members = positions[bins == node]
        if members.shape[0] < 3:
            continue  # too few samples to locate an axis reliably
        centres.append(members.mean(axis=0))
        groups.append(members)

    if len(centres) < 2:
        raise SkeletonError("organ did not yield enough centreline nodes")

    points = np.array(centres)
    axes = np.empty_like(points)
    axes[1:-1] = points[2:] - points[:-2]
    axes[0] = points[1] - points[0]
    axes[-1] = points[-1] - points[-2]
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = np.divide(axes, norms, out=np.zeros_like(axes), where=norms > 0)

    # Radius must be measured perpendicular to the local axis. Distance from
    # the bin centroid would fold in the axial spread within the bin, which is
    # half a segment long and so dwarfs a millimetre-scale petiole.
    radii = np.empty(points.shape[0])
    for index, (centre, members, axis) in enumerate(zip(points, groups, axes, strict=True)):
        offsets = members - centre
        radial = offsets - np.outer(offsets @ axis, axis)
        radii[index] = float(np.linalg.norm(radial, axis=1).mean())

    return Skeleton(points=points, radii=radii)


def skeletonise_plant(plant: organs.Plant, *, segment_length: float = DEFAULT_SEGMENT_LENGTH_M) -> dict[int, Skeleton]:
    """Centrelines for every stem organ, keyed by organ index.

    Foliage and fruit are skipped: they are blades and spheres, not tubes, and
    are simulated as single bodies hanging off the stem they attach to.
    """
    skeletons = {}
    for organ in plant.organs:
        if organ.tissue is not organs.Tissue.STEM:
            continue
        base = organ.attachment
        if base is None:
            # The main stem has no junction to start from. Rooting it at its
            # lowest point makes the centreline run ground-to-tip, so radii
            # taper the right way and organ height is a prefix along the stem
            # -- both of which the bottom-up deleafing rule depends on.
            base = organ.component.vertices[np.argmin(organ.component.vertices[:, _GLTF_UP])]
        try:
            skeletons[organ.index] = extract_skeleton(organ.component, base, segment_length=segment_length)
        except SkeletonError:
            continue
    return skeletons
