"""Recover per-organ structure from a fused tomato vine mesh.

The vine exporter welds every organ into one mesh split only by material, but
deleafing needs to address individual organs: which geometry is one compound
leaf, where its petiole meets the main stem, and what detaches when that
petiole is cut. Nothing in the GLB encodes that, so it is reconstructed here.

The reconstruction has three stages:

1. Split each material batch into connected components. Organs are separate
   surfaces in the source mesh, so a component is exactly one organ (one
   petiole segment, one leaflet blade, one fruit).
2. Root a tree over the stem-tissue components. Two organs that are attached
   interpenetrate, so surface adjacency gives the candidate edges and a
   shortest-path tree from the main stem orients them parent-to-child. Hop
   count dominates the edge weight so that a drooping petiole brushing a lower
   one cannot be mistaken for its parent; the surface gap only breaks ties.
3. Name the main stem's children from the generator's metadata attach points,
   by optimal one-to-one assignment rather than greedy nearest-match, so a
   single mismatched organ cannot cascade into a chain of wrong labels.

Foliage and fruit attach to their nearest stem organ rather than joining the
tree search, because leaflets of neighbouring leaves touch each other
constantly and would otherwise chain together.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import pathlib
from typing import Any

import numpy as np
import scipy.optimize
import scipy.sparse
import scipy.sparse.csgraph
import scipy.spatial

from greenhouse_sim import glb

# Source vertices coincide bit-for-bit where the exporter split UV/normal
# seams, so welding only needs to collapse exact duplicates. The tolerance is
# far below the ~0.2 mm feature size of a petiole.
WELD_TOLERANCE_M = 1e-6

# Attached organs interpenetrate; measured junction gaps on these assets are
# under 1 mm. 10 mm keeps a margin without bridging distinct organs, whose
# spacing along the stem is centimetres.
ADJACENCY_RADIUS_M = 0.010

_MAIN_STEM = "MainStem"


class Tissue(enum.Enum):
    """Material class of an organ, which determines its physical role."""

    STEM = "stem"
    FOLIAGE = "foliage"
    FRUIT = "fruit"


class SegmentationError(Exception):
    """Raised when a vine mesh cannot be resolved into a plausible plant."""


@dataclasses.dataclass(frozen=True)
class Component:
    """One connected surface within a material batch: a single plant organ."""

    index: int
    material: str
    tissue: Tissue
    vertices: np.ndarray  # (n_vertices, 3) float64, glTF Y-up metres
    triangles: np.ndarray  # (n_triangles, 3) int64 into vertices
    material_index: int | None = None
    normals: np.ndarray | None = None  # (n_vertices, 3) float64
    uvs: np.ndarray | None = None  # (n_vertices, 2) float64

    @property
    def num_triangles(self) -> int:
        return int(self.triangles.shape[0])

    @property
    def centroid(self) -> np.ndarray:
        return self.vertices.mean(axis=0)


@dataclasses.dataclass(frozen=True)
class Organ:
    """A component placed in the plant hierarchy."""

    index: int
    component: Component
    parent: int | None
    # Midpoint of the closest approach between this organ and its parent: the
    # junction, and for a petiole the agronomically correct flush cut site.
    attachment: np.ndarray | None
    # Gap between the two surfaces at the junction, a segmentation health check.
    attachment_gap: float
    label: str

    @property
    def tissue(self) -> Tissue:
        return self.component.tissue


@dataclasses.dataclass(frozen=True)
class Plant:
    """A vine resolved into labelled organs."""

    name: str
    organs: list[Organ]
    root: int
    metadata: dict[str, Any]

    def children_of(self, index: int) -> list[int]:
        return [o.index for o in self.organs if o.parent == index]

    def descendants_of(self, index: int) -> list[int]:
        """Every organ that detaches when `index` is severed, including itself."""
        collected: list[int] = []
        frontier = [index]
        while frontier:
            current = frontier.pop()
            collected.append(current)
            frontier.extend(self.children_of(current))
        return collected

    def labelled(self, prefix: str) -> list[Organ]:
        return [o for o in self.organs if o.label.startswith(prefix)]


def metadata_to_gltf(points: np.ndarray) -> np.ndarray:
    """Convert Blender Z-up metadata coordinates into the GLB's Y-up frame.

    The sidecar records attach points in the generator's Z-up frame while the
    GLB is exported Y-up, so (x, y, z) maps to (x, z, -y).
    """
    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    return np.column_stack([points[:, 0], points[:, 2], -points[:, 1]])


def classify_tissue(material: str) -> Tissue:
    if material == "TomatoStem":
        return Tissue.STEM
    if material.startswith("FruitRipe"):
        return Tissue.FRUIT
    return Tissue.FOLIAGE


def weld_vertices(positions: np.ndarray, tolerance: float = WELD_TOLERANCE_M) -> np.ndarray:
    """Map each vertex onto an identifier shared by coincident vertices."""
    quantized = np.round(positions / tolerance).astype(np.int64)
    _, inverse = np.unique(quantized, axis=0, return_inverse=True)
    return inverse.ravel()


def split_components(primitive: glb.Primitive, tolerance: float = WELD_TOLERANCE_M) -> list[Component]:
    """Split one material batch into its connected surfaces."""
    welded = weld_vertices(primitive.positions, tolerance)
    num_welded = int(welded.max()) + 1
    triangles = primitive.triangles

    corner_a, corner_b, corner_c = (welded[triangles[:, i]] for i in range(3))
    rows = np.concatenate([corner_a, corner_b, corner_c])
    cols = np.concatenate([corner_b, corner_c, corner_a])
    adjacency = scipy.sparse.coo_matrix(
        (np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(num_welded, num_welded)
    )
    _, labels = scipy.sparse.csgraph.connected_components(adjacency, directed=False)

    tissue = classify_tissue(primitive.material)
    triangle_labels = labels[corner_a]
    components = []
    for label in np.unique(triangle_labels):
        local_triangles = triangles[triangle_labels == label]
        used = np.unique(local_triangles)
        remap = np.full(primitive.positions.shape[0], -1, dtype=np.int64)
        remap[used] = np.arange(used.size)
        components.append(
            Component(
                index=len(components),
                material=primitive.material,
                tissue=tissue,
                vertices=primitive.positions[used],
                triangles=remap[local_triangles],
                material_index=primitive.material_index,
                normals=None if primitive.normals is None else primitive.normals[used],
                uvs=None if primitive.uvs is None else primitive.uvs[used],
            )
        )
    return components


def _closest_approach(
    tree_a: scipy.spatial.cKDTree, points_b: np.ndarray, upper_bound: float
) -> tuple[float, np.ndarray] | None:
    """Nearest point pair between a prebuilt tree and a point set."""
    distances, indices = tree_a.query(points_b, k=1, distance_upper_bound=upper_bound)
    nearest = int(np.argmin(distances))
    if not np.isfinite(distances[nearest]):
        return None
    midpoint = 0.5 * (points_b[nearest] + tree_a.data[indices[nearest]])
    return float(distances[nearest]), midpoint


def _stem_adjacency(
    components: list[Component], stem_indices: list[int], radius: float
) -> dict[tuple[int, int], tuple[float, np.ndarray]]:
    """Surface contacts between stem organs, keyed by ordered component pair."""
    trees = {i: scipy.spatial.cKDTree(components[i].vertices) for i in stem_indices}
    lowers = {i: components[i].vertices.min(axis=0) - radius for i in stem_indices}
    uppers = {i: components[i].vertices.max(axis=0) + radius for i in stem_indices}

    contacts: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    for position, first in enumerate(stem_indices):
        for second in stem_indices[position + 1 :]:
            # Bounding boxes reject the vast majority of pairs before any query.
            if np.any(lowers[first] > uppers[second]) or np.any(lowers[second] > uppers[first]):
                continue
            found = _closest_approach(trees[first], components[second].vertices, radius)
            if found is not None:
                contacts[first, second] = found
    return contacts


def _root_stem_tree(
    components: list[Component], stem_indices: list[int], root: int, radius: float
) -> dict[int, tuple[int, float, np.ndarray]]:
    """Orient stem contacts into a tree rooted at the main stem.

    Returns parent, gap and junction point per stem organ, excluding the root.
    """
    contacts = _stem_adjacency(components, stem_indices, radius)
    order = {component: position for position, component in enumerate(stem_indices)}

    size = len(stem_indices)
    matrix = scipy.sparse.dok_matrix((size, size), dtype=np.float64)
    for (first, second), (gap, _) in contacts.items():
        # Hop count dominates so a spurious organ-to-organ brush cannot outrank
        # the true parent; the gap only orders otherwise equal-length paths.
        weight = 1.0 + gap
        matrix[order[first], order[second]] = weight
        matrix[order[second], order[first]] = weight

    distances, predecessors = scipy.sparse.csgraph.dijkstra(
        matrix.tocsr(), directed=False, indices=order[root], return_predecessors=True
    )

    parents: dict[int, tuple[int, float, np.ndarray]] = {}
    for component in stem_indices:
        if component == root:
            continue
        predecessor = predecessors[order[component]]
        if predecessor < 0 or not np.isfinite(distances[order[component]]):
            continue  # Detached from the plant; handled by the caller.
        parent = stem_indices[predecessor]
        key = (min(parent, component), max(parent, component))
        gap, junction = contacts[key]
        parents[component] = (parent, gap, junction)
    return parents


class _StemIndex:
    """Nearest-stem-organ lookup over a single tree of all stem vertices."""

    def __init__(self, components: list[Component], stem_indices: list[int]) -> None:
        self._points = np.concatenate([components[i].vertices for i in stem_indices])
        self._owner = np.concatenate([np.full(components[i].vertices.shape[0], i) for i in stem_indices])
        self._tree = scipy.spatial.cKDTree(self._points)

    def nearest(self, target: np.ndarray) -> tuple[int, float, np.ndarray]:
        """Stem organ closest to `target`, with the gap and junction midpoint."""
        # Unbounded so an unusually isolated organ still lands on a real parent
        # rather than being dropped.
        distances, indices = self._tree.query(target, k=1)
        nearest = int(np.argmin(distances))
        point = self._points[indices[nearest]]
        return (
            int(self._owner[indices[nearest]]),
            float(distances[nearest]),
            0.5 * (target[nearest] + point),
        )


def _assign_labels(
    plan: dict[int, tuple[int, float, np.ndarray]], root: int, metadata: dict[str, Any]
) -> dict[int, str]:
    """Name the main stem's children from the generator's attach points.

    Solved as an assignment problem so that the labelling is globally
    consistent: greedy nearest-match lets one bad pairing displace the rest.
    """
    labels = {root: _MAIN_STEM}
    primary = sorted(index for index, (parent, _, _) in plan.items() if parent == root)
    named = [a for a in metadata.get("attachments", []) if a.get("parent") == _MAIN_STEM]
    if not primary or not named:
        return labels

    junctions = np.array([plan[index][2] for index in primary])
    targets = metadata_to_gltf(np.array([a["attach"] for a in named]))
    cost = np.linalg.norm(junctions[:, None, :] - targets[None, :, :], axis=2)

    rows, cols = scipy.optimize.linear_sum_assignment(cost)
    for row, col in zip(rows, cols, strict=True):
        labels[primary[row]] = named[col]["id"]
    for index in primary:
        labels.setdefault(index, f"Lateral_{index:03d}")
    return labels


def segment_plant(asset: glb.Glb, metadata: dict[str, Any], radius: float = ADJACENCY_RADIUS_M) -> Plant:
    """Resolve a vine mesh and its metadata sidecar into labelled organs."""
    components: list[Component] = []
    for primitive in asset.primitives:
        for component in split_components(primitive):
            components.append(dataclasses.replace(component, index=len(components)))

    stem_indices = [c.index for c in components if c.tissue is Tissue.STEM]
    if not stem_indices:
        raise SegmentationError(f"{asset.path}: no stem tissue found")
    # The main stem is the largest stem surface by a wide margin: it spans the
    # full plant height while laterals are centimetres long.
    root = max(stem_indices, key=lambda i: components[i].num_triangles)

    plan = _root_stem_tree(components, stem_indices, root, radius)

    # A stem organ the adjacency search could not reach is grafted onto the main
    # stem so the plant stays a single connected tree.
    root_index = _StemIndex(components, [root])
    for index in stem_indices:
        if index != root and index not in plan:
            plan[index] = root_index.nearest(components[index].vertices)

    stem_index = _StemIndex(components, stem_indices)
    for component in components:
        if component.tissue is not Tissue.STEM:
            plan[component.index] = stem_index.nearest(component.vertices)

    labels = _assign_labels(plan, root, metadata)

    organs = []
    for component in components:
        if component.index == root:
            organs.append(
                Organ(
                    index=root,
                    component=component,
                    parent=None,
                    attachment=None,
                    attachment_gap=0.0,
                    label=_MAIN_STEM,
                )
            )
            continue
        parent, gap, junction = plan[component.index]
        organs.append(
            Organ(
                index=component.index,
                component=component,
                parent=parent,
                attachment=junction,
                attachment_gap=gap,
                label=labels.get(component.index, f"{component.tissue.value.title()}_{component.index:03d}"),
            )
        )

    return Plant(name=asset.path.stem, organs=organs, root=root, metadata=metadata)


def load_plant(glb_path: str | pathlib.Path, radius: float = ADJACENCY_RADIUS_M) -> Plant:
    """Read a vine GLB and its sidecar metadata, then segment it."""
    glb_path = pathlib.Path(glb_path)
    metadata_path = glb_path.with_suffix(".json")
    if not metadata_path.exists():
        raise SegmentationError(f"{glb_path}: missing metadata sidecar {metadata_path.name}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return segment_plant(glb.read_glb(glb_path), metadata, radius)
