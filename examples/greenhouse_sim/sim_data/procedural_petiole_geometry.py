"""New curved petiole geometry for static VLM data, not calibrated plant dynamics.

The mesh and annotation capsule chain share one sampled metric centerline.
No Isaac/USD/model imports. Existing assets and similarity-generator contracts
remain unchanged. These geometric primitives do not approve a training target.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math

import numpy as np

VERSION = "procedural_petiole_geometry.v1"


def require(value, message):
    if not value:
        raise ValueError(message)


def unit(value):
    a = np.asarray(value, dtype=float)
    require(a.shape == (3,) and np.isfinite(a).all() and np.linalg.norm(a) > 1e-12,
            "Finite nonzero three-vector required")
    return a/np.linalg.norm(a)


def rotation_between(a, b):
    """Proper minimal rotation, including the antiparallel case."""
    a, b = unit(a), unit(b)
    cross, cosine = np.cross(a, b), float(np.clip(np.dot(a, b), -1, 1))
    sine = float(np.linalg.norm(cross))
    if sine < 1e-12:
        if cosine > 0:
            return np.eye(3)
        axis = unit(np.cross(a, np.eye(3)[np.argmin(abs(a))]))
        return 2*np.outer(axis, axis)-np.eye(3)
    x, y, z = cross/sine
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3)+sine*k+(1-cosine)*(k@k)


@dataclass(frozen=True)
class CurveSpec:
    length_m: float
    radius_m: float
    tip_radius_ratio: float = .45
    bend_normal_rad: float = .45
    bend_binormal_rad: float = .2
    bend_phase_rad: float = 0.
    spacing_m: float = .0015
    sides: int = 32
    rib_amplitude: float = .025
    rib_count: int = 5

    def validate(self):
        values = asdict(self)
        require(all(type(v) in (int, float) and math.isfinite(v) for v in values.values()),
                "Finite numeric curve parameters required")
        require(.025 <= self.length_m <= .45 and .0003 <= self.radius_m <= .005,
                "Curve dimensions outside explicit engineering bounds")
        require(.25 <= self.tip_radius_ratio <= 1 and .0005 <= self.spacing_m <= .002,
                "Invalid taper/sampling")
        require(abs(self.bend_normal_rad) <= .9 and abs(self.bend_binormal_rad) <= .65,
                "Excessive bend")
        require(type(self.sides) is int and 24 <= self.sides <= 64
                and type(self.rib_count) is int and 0 <= self.rib_count <= 8
                and 0 <= self.rib_amplitude <= .05, "Invalid radial sampling")
        return self


def curved_centerline(spec, direction=(1., 0., 0.)):
    """Integrate a smoothly varying tangent at fixed metric arc increments."""
    spec.validate()
    n = int(math.ceil(spec.length_m/spec.spacing_m))
    ds = spec.length_m/n
    t = (np.arange(n)+.5)/n
    # Smoothly varying heading, unlike rigidly rotating/scaling a donor curve.
    a = spec.bend_normal_rad*(1-np.cos(np.pi*t))/2
    b = spec.bend_binormal_rad*np.sin(np.pi*t)*np.sin(np.pi*t+spec.bend_phase_rad)
    tangents = np.column_stack((np.cos(a)*np.cos(b), np.sin(a)*np.cos(b), np.sin(b)))
    tangents = tangents@rotation_between((1, 0, 0), direction).T
    points = np.vstack((np.zeros(3), np.cumsum(tangents*ds, axis=0)))
    arc = np.arange(n+1)*ds
    radius = spec.radius_m*(1-(1-spec.tip_radius_ratio)*arc/spec.length_m)
    return dict(points=points, radius=radius, arc=arc, spec=asdict(spec),
                version=VERSION, actual_length_m=float(arc[-1]))


def transport_frames(points):
    points = np.asarray(points, float)
    require(points.ndim == 2 and points.shape[1] == 3 and len(points) >= 2
            and np.isfinite(points).all(), "Finite centerline required")
    segments = np.diff(points, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    require((lengths > 1e-10).all(), "Repeated centerline points")
    segments /= lengths[:, None]
    tangent = np.vstack((segments[0], segments[:-1]+segments[1:], segments[-1]))
    norms = np.linalg.norm(tangent, axis=1)
    require((norms > 1e-5).all(), "Reversing/degenerate centerline")
    tangent /= norms[:, None]
    normals = [unit(np.cross(tangent[0], np.eye(3)[np.argmin(abs(tangent[0]))]))]
    for a, b in zip(tangent[:-1], tangent[1:]):
        normal = rotation_between(a, b)@normals[-1]
        normals.append(unit(normal-b*np.dot(normal, b)))
    normals = np.asarray(normals)
    binormal = np.cross(tangent, normals)
    return np.stack((tangent, normals, binormal), axis=2)


def sample_curve(curve, distance_m):
    arc, points, radii = curve['arc'], curve['points'], curve['radius']
    require(math.isfinite(distance_m) and 0 <= distance_m <= arc[-1], "Arc outside curve")
    index = min(int(np.searchsorted(arc, distance_m, side='right'))-1, len(arc)-2)
    weight = (distance_m-arc[index])/(arc[index+1]-arc[index])
    return dict(point=(1-weight)*points[index]+weight*points[index+1],
                radius=float((1-weight)*radii[index]+weight*radii[index+1]),
                tangent=unit(points[index+1]-points[index]), arc_m=float(distance_m))


def sweep_mesh(curve):
    """Smooth capped tube with a UV seam and explicit outward face winding."""
    spec = CurveSpec(**curve['spec']).validate()
    p, r, arc = np.asarray(curve['points']), np.asarray(curve['radius']), np.asarray(curve['arc'])
    require(p.shape == (len(r), 3) and len(r) == len(arc) and (r > 0).all(), "Curve array mismatch")
    frame = transport_frames(p)
    angle = np.linspace(0, 2*np.pi, spec.sides+1)
    radial = (np.cos(angle)[None, :, None]*frame[:, None, :, 1]
              +np.sin(angle)[None, :, None]*frame[:, None, :, 2])
    rib = 1+spec.rib_amplitude*np.cos(spec.rib_count*angle)
    rings = p[:, None, :]+r[:, None, None]*rib[None, :, None]*radial
    # Cross the angular and longitudinal derivatives to include taper/curvature.
    longitudinal = np.gradient(rings, arc, axis=0, edge_order=1)
    angular = np.gradient(rings, angle, axis=1, edge_order=2)
    normals = np.cross(angular, longitudinal)
    require(np.all(np.linalg.norm(normals, axis=2) > 1e-14), "Degenerate tube normals")
    normals /= np.linalg.norm(normals, axis=2)[:, :, None]
    seam = normals[:, 0]+normals[:, -1]
    seam /= np.linalg.norm(seam, axis=1)[:, None]
    normals[:, 0], normals[:, -1] = seam, seam
    vertices = list(rings.reshape(-1, 3))
    vertex_normals = list(normals.reshape(-1, 3))
    uv = [[j/spec.sides, float(s/arc[-1])] for s in arc for j in range(spec.sides+1)]
    faces = []
    stride = spec.sides+1
    for i in range(len(p)-1):
        for j in range(spec.sides):
            a, b = i*stride+j, (i+1)*stride+j
            # Angular x longitudinal points outward; reverse the usual grid order.
            faces.extend(((a, a+1, b+1), (a, b+1, b)))
    for end in (0, len(p)-1):
        base = len(vertices)
        sign = -1 if end == 0 else 1
        vertices.append(p[end])
        vertex_normals.append(sign*frame[end, :, 0])
        uv.append([.5, .5])
        for j in range(spec.sides):
            vertices.append(rings[end, j])
            vertex_normals.append(sign*frame[end, :, 0])
            uv.append([.5+.5*np.cos(angle[j]), .5+.5*np.sin(angle[j])])
        for j in range(spec.sides):
            face = (base, base+1+j, base+1+(j+1)%spec.sides)
            faces.append(face if sign > 0 else (face[0], face[2], face[1]))
    vertices, faces = np.asarray(vertices), np.asarray(faces, dtype=np.int32)
    # Face normals are authoritative for winding; cap normal agrees with tangent.
    return dict(points=vertices, triangles=faces, normals=np.asarray(vertex_normals), uv=np.asarray(uv),
                extent=np.stack((vertices.min(axis=0), vertices.max(axis=0))), frames=frame)


def mesh_qa(mesh, weld_tolerance_m=1e-9):
    """Check finite geometry, winding consistency and geometric edge closure."""
    p, f, n = mesh['points'], mesh['triangles'], mesh['normals']
    require(p.ndim == 2 and p.shape[1] == 3 and n.shape == p.shape
            and f.ndim == 2 and f.shape[1] == 3 and np.isfinite(p).all()
            and np.isfinite(n).all() and np.issubdtype(f.dtype, np.integer)
            and f.min() >= 0 and f.max() < len(p), "Invalid mesh arrays")
    normals = np.cross(p[f[:, 1]]-p[f[:, 0]], p[f[:, 2]]-p[f[:, 0]])
    area2 = np.linalg.norm(normals, axis=1)
    require((area2 > 1e-13).all(), "Degenerate faces")
    agreement = np.einsum('ij,ij->i', normals, n[f].mean(axis=1))
    require((agreement > 0).all(), "Face winding and vertex normals disagree")
    _, welded = np.unique(np.round(p/weld_tolerance_m).astype(np.int64), axis=0, return_inverse=True)
    fw = welded[f]
    edges = np.concatenate((fw[:, [0, 1]], fw[:, [1, 2]], fw[:, [2, 0]]))
    canonical = np.sort(edges, axis=1)
    _, inverse, counts = np.unique(canonical, axis=0, return_inverse=True, return_counts=True)
    require((counts == 2).all(), "Non-manifold/open geometric edges")
    signs = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    require((np.bincount(inverse, weights=signs) == 0).all(), "Adjacent face winding inconsistent")
    volume = float(np.einsum('ij,ij->i', p[f[:, 0]], np.cross(p[f[:, 1]], p[f[:, 2]])).sum()/6)
    require(volume > 0, "Inward/zero signed volume")
    return dict(passed=True, vertices=len(p), triangles=len(f), volume_m3=volume,
                manifold_after_uv_normal_seam_weld=True, minimum_triangle_area_m2=float(area2.min()/2),
                self_intersections_tested=False, physics_validated=False)


def intrinsic_descriptor(curve, samples=21):
    """Similarity-invariant curve/radius shape; changing seed/name is not novelty."""
    length = float(curve['arc'][-1])
    sampled = [sample_curve(curve, d) for d in np.linspace(0, length, samples)]
    p = np.array([v['point'] for v in sampled])/length
    distances = np.linalg.norm(p[:, None, :]-p[None, :, :], axis=2)
    return np.r_[distances[np.triu_indices(samples, 1)], [v['radius']/length for v in sampled]]


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
