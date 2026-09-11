"""Pure geometry and USD authoring for an UNCALIBRATED discrete material band.

No application, stepping, source-asset loading/editing, plant wiring or release.
Four/eight axial layers times eight solid sectors fill a 32-sided prism. The
radius/length/source identifiers are explicit inputs, NOT inferred tissue data.
The polygon is inscribed: its volume is smaller than the circular cylinder by
1 - sin(pi/16)/(pi/16). This approximation is reported, never mass-corrected by
inventing density. Constant-radius geometry replaces rounded seam ends only in
a future separately reviewed integration; it is not identical capsule geometry.

Each shared face has three equal-area quadrature anchors matching its area,
centroid and second moments. Bulk springs use E*A/h and G*A/h (h is centroid
distance projected on the face normal). This is an explicit lattice law, NOT
proof of continuum E/Poisson/bending behaviour. Material.damping_ratio and
leaf_mass_kg are NOT used: no silent mass, damping or angular spring additions.

Only permanent bulk-neighbour collision pairs are filtered. Bulk compression is
carried by those springs; geometric cell overlap/gaps under strain remain an
unqualified discretization error, needing runtime bounds and channel checks.
Fracture-face contacts remain enabled. Fracture D6s are authored DISABLED until
a reviewed unilateral cohesive/contact adapter exists. Native drives are
bilateral; enabling these without that adapter is NOT authorized by this file.
All cells are dynamic. Optional remote proximal support uses elastic face
anchors to world, never kinematic seam cells or a weld across the fracture.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from .cohesive import CohesiveParameters, CohesiveState
from .mechanics import Material


STATUS = "uncalibrated_discrete_material_authoring_only"


@dataclass(frozen=True)
class BandConfig:
    source_manifest_path: str
    source_target: str
    radius_m: float
    length_m: float
    density_kg_m3: float
    bulk_material: Material
    cohesive_material: CohesiveParameters
    axial_layers: int = 4
    friction: float = 0.5
    contact_offset_m: float = 1e-5

    def __post_init__(self):
        if any(not isinstance(x, str) or not x.strip()
               for x in (self.source_manifest_path, self.source_target)):
            raise ValueError("Explicit source manifest and target required; no asset is loaded")
        if not isinstance(self.bulk_material, Material) or not isinstance(self.cohesive_material, CohesiveParameters):
            raise ValueError("Explicit bulk and cohesive materials required")
        for name in ("radius_m", "length_m", "density_kg_m3", "friction", "contact_offset_m"):
            v = getattr(self, name)
            if isinstance(v, (bool, str)) or not np.isscalar(v) or not np.isfinite(v):
                raise ValueError("Finite numeric band configuration required")
        if not (0.0005 <= self.radius_m <= 0.01 and 0.008 <= self.length_m <= 0.01
                and 100 <= self.density_kg_m3 <= 5000 and 0 <= self.friction <= 1
                and 0 < self.contact_offset_m <= 1e-5):
            raise ValueError("Band geometry/density/contact outside explicit coupon bounds")
        if type(self.axial_layers) is not int or self.axial_layers not in (4, 8):
            raise ValueError("Only 32-cell or axial-refined 64-cell topology")
        if self.density_kg_m3 != self.bulk_material.density_kg_m3:
            raise ValueError("Explicit density must agree with the declared Material prior")


def polygon_moments(points):
    """Exact area, centroid, central covariance of a CCW planar polygon."""
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("Finite planar polygon vertices required")
    q = np.roll(p, -1, axis=0)
    c = p[:, 0]*q[:, 1] - q[:, 0]*p[:, 1]
    area = float(c.sum()/2)
    if area <= 0:
        raise ValueError("Positive CCW polygon required")
    centre = ((p+q)*c[:, None]).sum(0)/(6*area)
    xx = np.sum((p[:, 0]**2+p[:, 0]*q[:, 0]+q[:, 0]**2)*c)/(12*area)
    yy = np.sum((p[:, 1]**2+p[:, 1]*q[:, 1]+q[:, 1]**2)*c)/(12*area)
    xy = np.sum((2*p[:, 0]*p[:, 1]+p[:, 0]*q[:, 1]+q[:, 0]*p[:, 1]+2*q[:, 0]*q[:, 1])*c)/(24*area)
    return area, centre, np.array([[xx, xy], [xy, yy]])-np.outer(centre, centre)


def face_geometry(points):
    """Outward frame columns = normal, tangent, bitangent; exact face moments."""
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 3 or not np.isfinite(p).all():
        raise ValueError("Finite 3D face vertices required")
    n = np.cross(p[1]-p[0], p[2]-p[0])
    norm = np.linalg.norm(n)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("Nondegenerate face required")
    n /= norm
    if np.max(np.abs((p-p[0])@n)) > 1e-12*np.max(np.linalg.norm(p-p[0], axis=1)):
        raise ValueError("Face vertices must be coplanar")
    t = p[1]-p[0]
    t /= np.linalg.norm(t)
    frame = np.column_stack((n, t, np.cross(n, t)))
    uv = (p-p[0]) @ frame[:, 1:]
    area, centre, covariance = polygon_moments(uv)
    return area, p[0]+frame[:, 1:]@centre, frame, uv, covariance


def face_anchors(points):
    """Three interior, noncollinear points; exact degree-two area moments.

    A covariance-scaled equilateral triangle has the required moments. Search
    its orientation for containment; fail rather than silently change weights.
    """
    area, centre, frame, uv, cov = face_geometry(points)
    eigenvalues, vectors = np.linalg.eigh(cov)
    if np.min(eigenvalues) <= 0:
        raise ValueError("Degenerate face covariance")
    root = vectors @ np.diag(np.sqrt(2*eigenvalues))
    origin = (centre-np.asarray(points)[0]) @ frame[:, 1:]
    edges = np.roll(uv, -1, axis=0)-uv
    for theta in np.linspace(0, 2*np.pi, 120, endpoint=False):
        a = theta+np.arange(3)*2*np.pi/3
        offsets = np.column_stack((np.cos(a), np.sin(a))) @ root.T
        rel = origin+offsets[:, None, :]-uv[None, :, :]
        cross = edges[None, :, 0]*rel[:, :, 1]-edges[None, :, 1]*rel[:, :, 0]
        if np.min(cross) >= -area*1e-12:
            return centre+offsets@frame[:, 1:].T, np.full(3, area/3)
    raise ValueError("No interior three-point quadrature for this face")


@dataclass
class Cell:
    index: int
    layer: int
    sector: int
    vertex_ids: tuple
    faces: tuple
    centre: np.ndarray
    volume_m3: float
    mass_kg: float
    inertia_kg_m2: np.ndarray


@dataclass
class Interface:
    index: int
    a: int
    b: int
    vertex_ids: tuple
    fracture: bool
    area_m2: float
    centre: np.ndarray
    frame: np.ndarray
    h_m: float
    anchors: np.ndarray
    weights_m2: np.ndarray
    stiffness_n_m: np.ndarray


@dataclass
class Band:
    config: BandConfig
    vertices: np.ndarray
    cells: list
    interfaces: list
    boundary_faces: list

    @property
    def bulk_pairs(self):
        return {(i.a, i.b) for i in self.interfaces if not i.fracture}

    def components(self, *, include_fracture):
        graph = defaultdict(set)
        for f in self.interfaces:
            if include_fracture or not f.fracture:
                graph[f.a].add(f.b)
                graph[f.b].add(f.a)
        unseen = set(range(len(self.cells)))
        result = []
        while unseen:
            pending = [min(unseen)]
            component = set()
            while pending:
                x = pending.pop()
                if x not in component:
                    component.add(x)
                    pending.extend(graph[x]-component)
            unseen -= component
            result.append(component)
        return result

    def summary(self):
        volume = sum(c.volume_m3 for c in self.cells)
        return dict(status=STATUS, source_manifest_path=self.config.source_manifest_path,
            source_target=self.config.source_target, cells=len(self.cells),
            interfaces=len(self.interfaces), fracture_interfaces=sum(f.fracture for f in self.interfaces),
            volume_m3=volume, mass_kg=sum(c.mass_kg for c in self.cells),
            circular_volume_fraction=volume/(np.pi*self.config.radius_m**2*self.config.length_m),
            source_assets_loaded_or_modified=False, constant_radius_polygon_approximation=True,
            bulk_law="EA_over_h_GA_over_h_discrete_face_springs_no_angular_parallel_spring",
            bulk_damping_ratio_used=False, continuum_qualified=False,
            simulation_ready=False, unilateral_fracture_runtime_implemented=False,
            physical_cut_verified=False, training_eligible=False)


def build(config):
    """Build an exactly conforming partition in band coordinates (stem = +Z)."""
    if not isinstance(config, BandConfig):
        raise ValueError("BandConfig required")
    c = config
    angles = np.arange(32)*2*np.pi/32
    ring = c.radius_m*np.column_stack((np.cos(angles), np.sin(angles)))
    xy = np.vstack((np.zeros((1, 2)), ring))
    z = np.linspace(-c.length_m/2, c.length_m/2, c.axial_layers+1)
    vertices = np.vstack([np.column_stack((xy, np.full(33, v))) for v in z])
    cells, owners = [], defaultdict(list)
    for layer in range(c.axial_layers):
        height = z[layer+1]-z[layer]
        for sector in range(8):
            ids = (0, *(1+(sector*4+k)%32 for k in range(5)))
            area, centre2, cov = polygon_moments(xy[list(ids)])
            centre = np.r_[centre2, (z[layer]+z[layer+1])/2]
            volume = area*height
            mass = c.density_kg_m3*volume
            inertia = mass*np.array([[cov[1, 1]+height**2/12, -cov[0, 1], 0],
                [-cov[0, 1], cov[0, 0]+height**2/12, 0], [0, 0, np.trace(cov)]])
            bottom = tuple(layer*33+i for i in ids)
            top = tuple((layer+1)*33+i for i in ids)
            faces = (tuple(reversed(bottom)), top, *(
                (bottom[j], bottom[(j+1)%6], top[(j+1)%6], top[j]) for j in range(6)))
            cell = Cell(len(cells), layer, sector, bottom+top, faces, centre, volume, mass, inertia)
            cells.append(cell)
            for face in faces:
                owners[tuple(sorted(face))].append((cell.index, face))
    interfaces, boundaries = [], []
    shear = c.bulk_material.youngs_modulus_pa/(2*(1+c.bulk_material.poisson_ratio))
    for key, pair in owners.items():
        if len(pair) == 1:
            boundaries.append(pair[0])
            continue
        if len(pair) != 2:
            raise ValueError("Nonmanifold cell partition")
        (a, face), (b, _) = pair
        points = vertices[list(face)]
        area, centre, frame, _, _ = face_geometry(points)
        h = float(np.dot(cells[b].centre-cells[a].centre, frame[:, 0]))
        if h <= 0:
            raise ValueError("Shared face normal must point from A toward B")
        fracture = cells[a].layer == c.axial_layers//2-1 and cells[b].layer == c.axial_layers//2
        anchors, weights = face_anchors(points)
        moduli = (np.array([c.cohesive_material.normal_stiffness_pa_m,
                           c.cohesive_material.shear_stiffness_pa_m,
                           c.cohesive_material.shear_stiffness_pa_m]) if fracture
                   else np.array([c.bulk_material.youngs_modulus_pa, shear, shear])/h)
        interfaces.append(Interface(len(interfaces), a, b, face, fracture, area, centre,
            frame, h, anchors, weights, weights[:, None]*moduli))
    band = Band(c, vertices, cells, interfaces, boundaries)
    validate(band)
    return band


def validate(band):
    """Validate geometry AND cached metadata before any USD authoring.

    Band arrays are inspection-friendly and mutable; never trust cached mass,
    adjacency, frames or stiffness when deciding which contacts to filter.
    """
    if not isinstance(band, Band) or not isinstance(band.config, BandConfig):
        raise ValueError("Configured Band required")
    c = band.config
    def same(actual, expected, name):
        a, e = np.asarray(actual), np.asarray(expected, float)
        if (a.shape != e.shape or a.dtype.kind not in "fiu" or not np.isfinite(a).all()
                or np.max(np.abs(a-e)) > 1e-10*max(float(np.max(np.abs(e))), 1e-30)):
            raise ValueError("Inconsistent band "+name)

    if len(band.cells) != 8*c.axial_layers or len(band.interfaces) != 16*c.axial_layers-8:
        raise ValueError("Unexpected topology")
    angles = np.arange(32)*2*np.pi/32
    xy = np.vstack((np.zeros((1, 2)), c.radius_m*np.column_stack((np.cos(angles), np.sin(angles)))))
    levels = np.linspace(-c.length_m/2, c.length_m/2, c.axial_layers+1)
    same(band.vertices, np.vstack([np.column_stack((xy, np.full(33, z))) for z in levels]), "reference vertices")
    tol = 1e-11*max(c.radius_m, c.length_m)
    planes = []
    owners = defaultdict(list)
    for index, cell in enumerate(band.cells):
        layer, sector = divmod(index, 8)
        if (cell.index, cell.layer, cell.sector) != (index, layer, sector):
            raise ValueError("Inconsistent cell identity")
        ids = (0, *(1+(sector*4+k)%32 for k in range(5)))
        expected_ids = tuple(layer*33+i for i in ids)+tuple((layer+1)*33+i for i in ids)
        if cell.vertex_ids != expected_ids:
            raise ValueError("Inconsistent cell vertex ownership")
        area, centre2, covariance = polygon_moments(xy[list(ids)])
        h = levels[layer+1]-levels[layer]
        mass = c.density_kg_m3*area*h
        same(cell.volume_m3, area*h, "cell volume")
        same(cell.mass_kg, mass, "cell mass")
        same(cell.centre, np.r_[centre2, (levels[layer]+levels[layer+1])/2], "cell centre")
        same(cell.inertia_kg_m2, mass*np.array([
            [covariance[1, 1]+h*h/12, -covariance[0, 1], 0],
            [-covariance[0, 1], covariance[0, 0]+h*h/12, 0],
            [0, 0, np.trace(covariance)]]), "cell inertia")
        edges = Counter()
        cell_planes = []
        points = band.vertices[list(cell.vertex_ids)]
        signed_volume = 0.
        for face in cell.faces:
            if len(face) < 3 or any(v not in cell.vertex_ids for v in face):
                raise ValueError("Invalid cell face ownership")
            p = band.vertices[list(face)]
            area, centre, frame, _, _ = face_geometry(p)
            normal = frame[:, 0]
            if np.max((points-centre)@normal) > tol:
                raise ValueError("Nonconvex or inward cell face")
            signed_volume += area*np.dot(centre-cell.centre, normal)/3
            cell_planes.append((centre, normal))
            owners[tuple(sorted(face))].append((cell.index, face, normal))
            for a, b in zip(face, face[1:]+face[:1]):
                edges[(a, b)] += 1
        if any(v != 1 or edges[(b, a)] != 1 for (a, b), v in edges.items()):
            raise ValueError("Cell is not oriented watertight")
        if not np.isclose(signed_volume, cell.volume_m3, rtol=1e-10, atol=0):
            raise ValueError("Polyhedron and exact prism volumes differ")
        planes.append(cell_planes)
    for entries in owners.values():
        if len(entries) > 2 or (len(entries) == 2 and not np.allclose(entries[0][2], -entries[1][2], atol=1e-12, rtol=0)):
            raise ValueError("Shared face orientation/manifold error")
    boundaries = [(entries[0][0], entries[0][1]) for entries in owners.values() if len(entries) == 1]
    if Counter(band.boundary_faces) != Counter(boundaries):
        raise ValueError("Inconsistent remote boundary faces")
    shared = {key: entries for key, entries in owners.items() if len(entries) == 2}
    seen = set()
    E = c.bulk_material.youngs_modulus_pa
    G = E/(2*(1+c.bulk_material.poisson_ratio))
    for index, f in enumerate(band.interfaces):
        key = tuple(sorted(f.vertex_ids))
        if f.index != index or key not in shared or key in seen:
            raise ValueError("Inconsistent shared interface identity")
        seen.add(key)
        a, b = shared[key]
        if (f.a, f.b, f.vertex_ids) != (a[0], b[0], a[1]):
            raise ValueError("Interface must connect its actual face neighbours")
        fracture = band.cells[f.a].layer == c.axial_layers//2-1 and band.cells[f.b].layer == c.axial_layers//2
        if type(f.fracture) is not bool or f.fracture != fracture:
            raise ValueError("Interface fracture classification disagrees with geometry")
        points = band.vertices[list(f.vertex_ids)]
        area, centre, frame, _, _ = face_geometry(points)
        h = float((band.cells[f.b].centre-band.cells[f.a].centre)@frame[:, 0])
        if h <= 0:
            raise ValueError("Invalid projected centroid distance")
        anchors, weights = face_anchors(points)
        for actual, expected, name in ((f.area_m2, area, "face area"), (f.centre, centre, "face centre"),
                (f.frame, frame, "face frame"), (f.h_m, h, "face distance"),
                (f.anchors, anchors, "face anchors"), (f.weights_m2, weights, "face weights")):
            same(actual, expected, name)
        moduli = (np.array([c.cohesive_material.normal_stiffness_pa_m,
            c.cohesive_material.shear_stiffness_pa_m, c.cohesive_material.shear_stiffness_pa_m])
            if fracture else np.array([E, G, G])/h)
        same(f.stiffness_n_m, weights[:, None]*moduli, "face stiffness")
    if seen != set(shared):
        raise ValueError("Missing shared interfaces")
    # Separating face planes suffice for this aligned prism partition. Touching
    # is allowed; positive-volume overlap is not. No shrink/offset is introduced.
    for a, ca in enumerate(band.cells):
        for b in range(a+1, len(band.cells)):
            cb = band.cells[b]
            separated = any(np.min((band.vertices[list(cb.vertex_ids)]-p)@n) >= -tol for p, n in planes[a])
            separated |= any(np.min((band.vertices[list(ca.vertex_ids)]-p)@n) >= -tol for p, n in planes[b])
            if not separated:
                raise ValueError("Positive-volume cell overlap")
    expected = 16*c.radius_m**2*np.sin(np.pi/16)*c.length_m
    if not np.isclose(sum(x.volume_m3 for x in band.cells), expected, rtol=1e-12, atol=0):
        raise ValueError("Partition volume mismatch")
    if len(band.cells) != 8*c.axial_layers or len(band.interfaces) != 16*c.axial_layers-8:
        raise ValueError("Unexpected topology")
    if sum(f.fracture for f in band.interfaces) != 8:
        raise ValueError("Exactly eight fracture interfaces required")
    halves = band.components(include_fracture=False)
    if len(band.components(include_fracture=True)) != 1 or len(halves) != 2:
        raise ValueError("Disconnected intact graph or fracture bypass")
    expected_halves = {frozenset(x.index for x in band.cells if x.layer < c.axial_layers//2),
                       frozenset(x.index for x in band.cells if x.layer >= c.axial_layers//2)}
    if set(map(frozenset, halves)) != expected_halves:
        raise ValueError("Permanent bond bypasses seam")
    return band.summary()


def author(stage, band, *, root, proximal_support, transform=None):
    """Author only on an empty metre stage; no export and no runtime activation.

    ``proximal_support`` is explicit: False leaves remote boundary faces for a
    later integration; True adds only elastic world-to-proximal-end anchors.
    Fracture D6s remain disabled in BOTH cases. Return persistent virgin states
    and axis-ordered drive handles for a separately reviewed runtime adapter.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt

    validate(band)
    if type(proximal_support) is not bool:
        raise ValueError("Explicit boolean proximal support selection required")
    path = Sdf.Path(root)
    if not path.IsAbsolutePath() or not path.IsPrimPath() or path == Sdf.Path.absoluteRootPath:
        raise ValueError("Absolute new prim root required")
    if stage.GetPrimAtPath(path) or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage), 1., atol=1e-12, rtol=0):
        raise ValueError("New root on metre stage required")
    if any(p.HasAPI(UsdPhysics.RigidBodyAPI) or p.HasAPI(UsdPhysics.CollisionAPI) for p in stage.Traverse()):
        raise ValueError("Isolated authoring only; no implicit production wiring")
    pose = np.eye(4) if transform is None else np.asarray(transform, float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all() or
            not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-12, rtol=0) or
            not np.allclose(pose[:3, :3].T@pose[:3, :3], np.eye(3), atol=1e-12, rtol=0) or
            not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-12, rtol=0)):
        raise ValueError("Rigid proper transform required; no geometry scaling")
    rotation, translation = pose[:3, :3], pose[:3, 3]

    def quat(matrix):
        m = np.eye(4)
        m[:3, :3] = matrix
        return Gf.Quatf(Gf.Matrix4d(m.T.tolist()).ExtractRotationQuat())

    def vec(x):
        return Gf.Vec3f(*map(float, x))

    root_xform = UsdGeom.Xform.Define(stage, path)
    # Supplied transform is world-space; do not inherit hidden ancestor scale.
    root_xform.SetResetXformStack(True)
    root_prim = root_xform.GetPrim()
    root_prim.CreateAttribute("band:status", Sdf.ValueTypeNames.String).Set(STATUS)
    root_prim.CreateAttribute("band:simulationReady", Sdf.ValueTypeNames.Bool).Set(False)
    for name, value in (("radiusM", band.config.radius_m), ("lengthM", band.config.length_m),
            ("densityKgM3", band.config.density_kg_m3),
            ("bulkYoungsModulusPa", band.config.bulk_material.youngs_modulus_pa),
            ("bulkPoissonRatio", band.config.bulk_material.poisson_ratio),
            ("cohesiveKnPaPerM", band.config.cohesive_material.normal_stiffness_pa_m),
            ("cohesiveKtPaPerM", band.config.cohesive_material.shear_stiffness_pa_m),
            ("cohesiveStrengthPa", band.config.cohesive_material.normal_strength_pa),
            ("cohesiveGcJPerM2", band.config.cohesive_material.fracture_energy_j_m2)):
        root_prim.CreateAttribute("band:"+name, Sdf.ValueTypeNames.Double).Set(float(value))
    for name, value in (("sourceManifest", band.config.source_manifest_path), ("sourceTarget", band.config.source_target)):
        root_prim.CreateAttribute("band:"+name, Sdf.ValueTypeNames.String).Set(value)
    material = UsdShade.Material.Define(stage, root+"/ContactMaterial")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(band.config.friction))
    api.CreateDynamicFrictionAttr(float(band.config.friction))
    api.CreateRestitutionAttr(0.)
    paths = [root+f"/Cells/Cell_{c.index:03d}" for c in band.cells]
    for cell, cell_path in zip(band.cells, paths):
        body = UsdGeom.Xform.Define(stage, cell_path)
        body.AddTranslateOp().Set(Gf.Vec3d(*(rotation@cell.centre+translation)))
        body.AddOrientOp().Set(quat(rotation))
        rigid = UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        rigid.CreateKinematicEnabledAttr(False)
        rigid.CreateVelocityAttr(vec([0, 0, 0]))
        rigid.CreateAngularVelocityAttr(vec([0, 0, 0]))
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateDensityAttr(float(band.config.density_kg_m3))
        mass.CreateMassAttr(float(cell.mass_kg))
        mass.CreateCenterOfMassAttr(vec([0, 0, 0]))
        values, axes = np.linalg.eigh(cell.inertia_kg_m2)
        if np.linalg.det(axes) < 0:
            axes[:, 0] *= -1
        mass.CreateDiagonalInertiaAttr(vec(values))
        mass.CreatePrincipalAxesAttr(quat(axes))
        body.GetPrim().AddAppliedSchema("PhysxRigidBodyAPI")
        for name, kind, value in (("sleepThreshold", Sdf.ValueTypeNames.Float, 0.),
                ("linearDamping", Sdf.ValueTypeNames.Float, 0.), ("angularDamping", Sdf.ValueTypeNames.Float, 0.),
                ("solverPositionIterationCount", Sdf.ValueTypeNames.Int, 32),
                ("solverVelocityIterationCount", Sdf.ValueTypeNames.Int, 8)):
            body.GetPrim().CreateAttribute("physxRigidBody:"+name, kind, custom=False).Set(value)
        body.GetPrim().AddAppliedSchema("PhysxContactReportAPI")
        body.GetPrim().CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float, custom=False).Set(0.)
        mesh = UsdGeom.Mesh.Define(stage, cell_path+"/Solid")
        points = band.vertices[list(cell.vertex_ids)]-cell.centre
        lookup = {v: k for k, v in enumerate(cell.vertex_ids)}
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points.astype(np.float32)))
        mesh.CreateFaceVertexCountsAttr([len(f) for f in cell.faces])
        mesh.CreateFaceVertexIndicesAttr([lookup[v] for f in cell.faces for v in f])
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateExtentAttr([vec(points.min(0)), vec(points.max(0))])
        mesh.CreateDisplayColorAttr([vec([.2, .5, .2] if cell.layer < band.config.axial_layers//2 else [.4, .65, .2])])
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim()).CreateCollisionEnabledAttr(True)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        mesh.GetPrim().AddAppliedSchema("PhysxCollisionAPI")
        for name, value in (("contactOffset", band.config.contact_offset_m), ("restOffset", 0.)):
            mesh.GetPrim().CreateAttribute("physxCollision:"+name, Sdf.ValueTypeNames.Float, custom=False).Set(float(value))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material, materialPurpose="physics")
    for a, b in sorted(band.bulk_pairs):
        UsdPhysics.FilteredPairsAPI.Apply(stage.GetPrimAtPath(paths[a])).CreateFilteredPairsRel().AddTarget(paths[b])

    def add_face(name, a, b, frame, anchors, stiffness, enabled):
        result = []
        for k, (anchor, ks) in enumerate(zip(anchors, stiffness)):
            joint = UsdPhysics.Joint.Define(stage, root+f"/{name}/Anchor_{k}")
            joint.CreateJointEnabledAttr(enabled)
            joint.CreateCollisionEnabledAttr(True)
            joint.CreateExcludeFromArticulationAttr(True)
            for side, index in enumerate((a, b)):
                if index is None:
                    local, orientation = rotation@anchor+translation, rotation@frame
                else:
                    getattr(joint, f"CreateBody{side}Rel")().SetTargets([paths[index]])
                    local, orientation = anchor-band.cells[index].centre, frame
                getattr(joint, f"CreateLocalPos{side}Attr")(vec(local))
                getattr(joint, f"CreateLocalRot{side}Attr")(quat(orientation))
            drives = []
            for axis, stiffness_value in zip(("transX", "transY", "transZ"), ks):
                drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), axis)
                drive.CreateTypeAttr("force")
                drive.CreateStiffnessAttr(float(stiffness_value))
                drive.CreateDampingAttr(0.)
                drive.CreateMaxForceAttr(float("inf"))
                drive.CreateTargetPositionAttr(0.)
                drive.CreateTargetVelocityAttr(0.)
                drives.append(drive)
            result.append(dict(joint=joint, drives=drives))
        return result

    interfaces = []
    for f in band.interfaces:
        records = add_face(f"Interfaces/Face_{f.index:03d}", f.a, f.b, f.frame, f.anchors,
                           f.stiffness_n_m, not f.fracture)
        interfaces.append(dict(face=f, anchors=records,
            states=tuple(CohesiveState(band.config.cohesive_material) for _ in range(3)) if f.fracture else ()))
    support = []
    if proximal_support:
        for index, face in band.boundary_faces:
            points = band.vertices[list(face)]
            if not np.allclose(points[:, 2], -band.config.length_m/2, atol=1e-13, rtol=0):
                continue
            # Reverse outward end normal: world A -> material B is +Z.
            points = points[::-1]
            _, _, frame, _, _ = face_geometry(points)
            anchors, weights = face_anchors(points)
            E = band.config.bulk_material.youngs_modulus_pa
            G = E/(2*(1+band.config.bulk_material.poisson_ratio))
            h = band.config.length_m/(2*band.config.axial_layers)
            support.extend(add_face(f"RemoteProximal/Cell_{index:03d}", None, index,
                frame, anchors, weights[:, None]*np.array([E, G, G])/h, True))
    return dict(band=band, paths=paths, interfaces=interfaces, proximal_support=support,
        transform=pose.copy(), summary=band.summary(), simulation_ready=False,
        runtime_required="unilateral cohesive active set; contact/gap/overlap/force/speed and energy guards")
