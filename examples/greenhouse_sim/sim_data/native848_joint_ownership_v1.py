"""Prove a source-authored attachment stub owned by the adjacent stem segment.

This adds no visibility/depth tolerance. Only the declared parent's immediate
main-stem parent can qualify, with an explicit secondary capsule ending at the
petiole attachment and an authenticated matching mesh seam.
"""
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from pxr import Usd, UsdGeom
from .cut_regions import _oriented_chain
from .dataset_review import require

SCHEMA = 'greenhouse.native848_verified_attachment_joint_owner.v1'
SOURCE_COORDINATE_TOLERANCE_M = 1e-6
MESH_SEAM_TOLERANCE_M = 2e-6


def candidate_stub(report, component_id):
    """Pure source topology prefilter; absence leaves the old evaluator intact."""
    components = report['components']
    target = components[component_id]
    parent = components.get(target.get('parent'), {})
    ancestor = components.get(parent.get('parent'), {})
    if target.get('type') != 'sub_stem' or parent.get('type') != 'main_stem' or ancestor.get('type') != 'main_stem':
        return None
    try:
        chain, _, _, _ = _oriented_chain(target, SOURCE_COORDINATE_TOLERANCE_M)
        chain = np.asarray(chain, float)
        tangent = chain[1, :3] - chain[0, :3]
        tangent /= np.linalg.norm(tangent)
        attachment = np.asarray(target['attachment_plant_m'], float)
        parent_root = np.asarray(parent['attachment_plant_m'], float)
        ancestor_origin = np.asarray(ancestor['translation_plant_m'], float)
        matches = []
        for index, raw in enumerate(ancestor['capsules_local_m'][1:], 1):
            stub = np.asarray(raw, float)
            if stub.shape != (2, 4) or not np.isfinite(stub).all():
                continue
            for reverse in (False, True):
                oriented = stub[::-1] if reverse else stub
                start, end = oriented[:, :3] + ancestor_origin
                delta = end - start
                length = float(np.linalg.norm(delta))
                endpoint_error = float(np.linalg.norm(end - attachment))
                radius_error = abs(float(oriented[-1, 3]) - float(chain[0, 3]))
                if length <= SOURCE_COORDINATE_TOLERANCE_M:
                    continue
                cosine = float(np.dot(delta / length, tangent))
                # The branch root must be inside this exact parent junction;
                # a remote ancestor or a generic trunk endpoint cannot qualify.
                root_gap = float(np.linalg.norm(start - parent_root))
                if (endpoint_error <= SOURCE_COORDINATE_TOLERANCE_M
                        and radius_error <= SOURCE_COORDINATE_TOLERANCE_M
                        and cosine >= 1. - 1e-6
                        and root_gap <= float(parent['radius_m']) + SOURCE_COORDINATE_TOLERANCE_M):
                    matches.append(dict(ancestor_component_id=parent['parent'],
                        declared_parent_component_id=target['parent'], stub_chain_index=index,
                        stub_reversed=reverse, attachment_plant_m=attachment.tolist(),
                        petiole_tangent_plant=tangent.tolist(), attachment_radius_m=float(chain[0, 3]),
                        stub_endpoint_error_m=endpoint_error, radius_error_m=radius_error,
                        stub_to_petiole_tangent_cosine=cosine, branch_root_to_declared_parent_root_m=root_gap,
                        stub_start_plant_m=start.tolist(), stub_end_plant_m=end.tolist()))
        return matches[0] if len(matches) == 1 else None
    except (KeyError, ValueError, AssertionError, IndexError, TypeError):
        return None


def mesh_seam(stub_vertices, petiole_vertices, candidate):
    """Require a distributed shared end ring, not a nearby/intersecting point."""
    center = np.asarray(candidate['attachment_plant_m'], float)
    axis = np.asarray(candidate['petiole_tangent_plant'], float)
    radius = float(candidate['attachment_radius_m'])
    a, b = np.asarray(stub_vertices, float), np.asarray(petiole_vertices, float)
    require(a.ndim == b.ndim == 2 and a.shape[1] == b.shape[1] == 3
            and np.isfinite(a).all() and np.isfinite(b).all(), 'Finite source mesh vertices required')
    a = a[np.linalg.norm(a-center, axis=1) <= radius + .001]
    b = b[np.linalg.norm(b-center, axis=1) <= radius + .001]
    if len(a) < 6 or len(b) < 6:
        return None
    distances = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    nearest = distances.min(axis=1)
    matched = a[nearest <= MESH_SEAM_TOLERANCE_M]
    if len(matched) < 6:
        return None
    # At least six distinct vertices must surround the attachment axis.
    matched = np.unique(np.round(matched, 9), axis=0)
    offsets = matched-center
    axial = offsets @ axis
    radial = offsets - axial[:, None]*axis
    radii = np.linalg.norm(radial, axis=1)
    if (len(matched) < 6 or np.max(np.abs(axial)) > MESH_SEAM_TOLERANCE_M
            or np.min(radii) < radius*.9 or np.max(radii) > radius*1.1):
        return None
    e0 = radial[0]/np.linalg.norm(radial[0])
    e1 = np.cross(axis, e0)
    angles = np.sort(np.arctan2(radial@e1, radial@e0))
    gap = float(np.max(np.diff(np.r_[angles, angles[0]+2*math.pi])))
    if gap >= math.pi:
        return None
    return dict(matched_unique_vertices=len(matched),
        maximum_matching_vertex_distance_m=float(nearest[nearest <= MESH_SEAM_TOLERANCE_M].max()),
        maximum_seam_axial_error_m=float(np.abs(axial).max()), maximum_ring_gap_radians=gap,
        matched_vertices_plant_m=matched.tolist(), source_meshes_modified=False)


def _bound(path, expected, bindings):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    require(digest == expected, 'Changed authenticated joint source')
    bindings[str(path)] = digest
    return path


def _mesh_points(path, translation, bindings):
    stage = Usd.Stage.Open(str(path))
    require(stage is not None, 'Readable source component mesh required')
    cache = UsdGeom.XformCache()
    arrays = []
    for layer in stage.GetUsedLayers():
        if layer.realPath:
            p = Path(layer.realPath).resolve()
            # A freshly computed hash is not source provenance. These source
            # component assets are self-contained; an external layer needs a
            # separately reviewed provenance contract and cannot qualify here.
            require(p == Path(path).resolve(), 'Self-contained source joint mesh required')
            _bound(p, bindings[str(p)], bindings)
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            require(not mesh.GetPointsAttr().GetTimeSamples(), 'Static authored mesh seam required')
            points = np.asarray(mesh.GetPointsAttr().Get(), float)
            matrix = np.asarray(cache.GetLocalToWorldTransform(prim), float)
            arrays.append((np.c_[points, np.ones(len(points))] @ matrix)[:, :3])
    require(bool(arrays), 'Actual component mesh required')
    return np.concatenate(arrays) + np.asarray(translation, float)


def verified_additional_parent(entry, catalogue):
    """Return an exact same-instance component index and its reconstructible proof."""
    variant, component_id = entry['target_id'].split('/')
    report = entry['report']
    require(report['plant_id'] == entry['source_family'], 'Exact anatomical source family required')
    candidate = candidate_stub(report, component_id)
    if candidate is None:
        return set(), None
    ancestor_id = candidate['ancestor_component_id']
    matches = [c for c in catalogue if c['variant_id'] == variant and c['component_id'] == ancestor_id]
    require(len(matches) == 1 and matches[0]['organ_type'] == 'main_stem'
            and matches[0]['source_plant_id'] == entry['source_family'], 'Exact same-instance joint owner required')
    bindings = {}
    manifest_path = _bound(report['manifest_path'], report['manifest_sha256'], bindings)
    manifest = json.loads(manifest_path.read_text())
    raw = {c['id']: c for c in manifest['components']}
    selected_ids = [component_id, candidate['declared_parent_component_id'], ancestor_id]
    paths = {}
    for key in selected_ids:
        component, source = report['components'][key], raw[key]
        fields = [('parent', 'parent'), ('type', 'type'), ('file', 'file'),
                  ('attachment_plant_m', 'attach_point'), ('capsules_local_m', 'capsules'), ('radius_m', 'radius')]
        require(all(component[a] == source[b] for a, b in fields)
                and component['translation_plant_m'] == source['transform']['translate'], 'Report joint topology differs from original manifest')
        p = (manifest_path.parent/source['file']).resolve()
        require(p.parent == manifest_path.parent, 'Task source component file required')
        paths[key] = _bound(p, component['asset_sha256'], bindings)
    stem = _mesh_points(paths[ancestor_id], report['components'][ancestor_id]['translation_plant_m'], bindings)
    petiole = _mesh_points(paths[component_id], report['components'][component_id]['translation_plant_m'], bindings)
    seam = mesh_seam(stem, petiole, candidate)
    for p, digest in list(bindings.items()):
        _bound(p, digest, bindings)
    if seam is None:
        return set(), None
    proof = dict(schema=SCHEMA, target_id=entry['target_id'], source_family=entry['source_family'],
        source_target_id=entry['source_family']+'/'+component_id, plant_instance_id=variant,
        additional_main_stem_component_id=ancestor_id, additional_component_index=matches[0]['component_index'],
        eligibility='exact_immediate_main_stem_ancestor_with_authored_matching_attachment_stub',
        candidate=candidate, mesh_seam=seam, source_bindings=bindings,
        allowed_only_below_arc_m=.008, depth_predicate_unchanged=True, nominal_and_support_ownership_unchanged=True)
    return {matches[0]['component_index']}, proof
