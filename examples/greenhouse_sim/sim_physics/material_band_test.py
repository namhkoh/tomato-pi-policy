"""Pure polygon/graph/USD checks. Never initialize Kit or native simulation."""
from dataclasses import replace
import numpy as np
import pytest

from sim_physics.cohesive import CohesiveParameters, cohesive_response
from sim_physics.mechanics import Material
from sim_physics.material_band import BandConfig, author, build, face_geometry, polygon_moments, validate


@pytest.fixture
def config():
    return BandConfig(
        source_manifest_path="data/sim_data/package_20260905/tomato_greenhouse_pack/plants/components/seed101_full/manifest.json",
        source_target="seed101_full/SubStem_41", radius_m=.003, length_m=.008,
        density_kg_m3=950., bulk_material=Material(),
        cohesive_material=CohesiveParameters(1e8, 2e8, 1e4, 2.))


@pytest.fixture
def band(config):
    return build(config)


@pytest.mark.parametrize("change", [dict(radius_m=True), dict(radius_m=0), dict(radius_m=np.nan),
    dict(length_m=.0079), dict(length_m=.0101), dict(axial_layers=3), dict(axial_layers=True),
    dict(axial_layers=16), dict(density_kg_m3=1000), dict(friction=-1), dict(contact_offset_m=.0005),
    dict(source_target=""), dict(source_manifest_path=None), dict(bulk_material=None)])
def test_explicit_bounded_configuration(config, change):
    with pytest.raises(ValueError):
        replace(config, **change)


def test_no_default_source_geometry_or_material():
    with pytest.raises(TypeError):
        BandConfig()


@pytest.mark.parametrize("layers,interfaces,bulk", [(4, 56, 48), (8, 120, 112)])
def test_topology_volume_mass_and_disconnect(config, layers, interfaces, bulk):
    b = build(replace(config, axial_layers=layers))
    expected_volume = 16*config.radius_m**2*np.sin(np.pi/16)*config.length_m
    assert len(b.cells) == 8*layers
    assert len(b.interfaces) == interfaces
    assert len(b.bulk_pairs) == bulk
    assert sum(i.fracture for i in b.interfaces) == 8
    assert sum(x.volume_m3 for x in b.cells) == pytest.approx(expected_volume, rel=1e-12)
    assert sum(x.mass_kg for x in b.cells) == pytest.approx(950*expected_volume, rel=1e-12)
    np.testing.assert_allclose([x.mass_kg for x in b.cells], 950*expected_volume/(8*layers), rtol=1e-12)
    assert len(b.components(include_fracture=True)) == 1
    assert sorted(map(len, b.components(include_fracture=False))) == [4*layers, 4*layers]
    for a, d in b.bulk_pairs:
        assert (b.cells[a].layer < layers//2) == (b.cells[d].layer < layers//2)
    assert b.summary()["circular_volume_fraction"] == pytest.approx(.9935868511442058)


def test_polyhedral_mass_integral_independent_of_prism_formula(band):
    # Signed origin-tetrahedra integration, independent of 2D prism formulas.
    for cell in band.cells:
        volume, first, second = 0., np.zeros(3), np.zeros((3, 3))
        for face in cell.faces:
            p = band.vertices[list(face)]
            for k in range(1, len(p)-1):
                triangle = p[[0, k, k+1]]
                v = np.linalg.det(triangle)/6
                s = triangle.sum(0)
                volume += v
                first += v*s/4
                second += v*(triangle.T@triangle+np.outer(s, s))/20
        centre = first/volume
        central_second = second-volume*np.outer(centre, centre)
        inertia = band.config.density_kg_m3*(np.trace(central_second)*np.eye(3)-central_second)
        assert volume == pytest.approx(cell.volume_m3, rel=1e-12)
        np.testing.assert_allclose(centre, cell.centre, atol=1e-16)
        np.testing.assert_allclose(inertia, cell.inertia_kg_m2, rtol=1e-11, atol=1e-25)
        assert np.linalg.eigvalsh(inertia).min() > 0


def test_total_inertia_and_axial_refinement_invariance(config, band):
    def total(b):
        return sum((x.inertia_kg_m2+x.mass_kg*(np.dot(x.centre, x.centre)*np.eye(3)-np.outer(x.centre, x.centre))) for x in b.cells)
    refined = build(replace(config, axial_layers=8))
    np.testing.assert_allclose(total(refined), total(band), rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose(sum(c.mass_kg*c.centre for c in band.cells), 0., atol=1e-21)
    np.testing.assert_allclose(total(band), np.diag(np.diag(total(band))), atol=1e-24)
    assert total(band)[0, 0] == pytest.approx(total(band)[1, 1])


def test_radius_length_density_scaling_and_sector_rotation(config, band):
    scale = 1.25
    scaled = build(replace(config, radius_m=config.radius_m*scale, length_m=.01))
    denser = build(replace(config, density_kg_m3=1900., bulk_material=replace(config.bulk_material, density_kg_m3=1900.)))
    for a, b, d in zip(band.cells, scaled.cells, denser.cells):
        np.testing.assert_allclose(b.centre, scale*a.centre, atol=1e-17)
        assert b.mass_kg == pytest.approx(a.mass_kg*scale**3)
        np.testing.assert_allclose(b.inertia_kg_m2, a.inertia_kg_m2*scale**5, atol=1e-25)
        np.testing.assert_allclose(d.inertia_kg_m2, 2*a.inertia_kg_m2, atol=1e-25)
    angle = np.pi/4
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    np.testing.assert_allclose(band.cells[1].centre, rotation@band.cells[0].centre, atol=1e-17)
    np.testing.assert_allclose(band.cells[1].inertia_kg_m2, rotation@band.cells[0].inertia_kg_m2@rotation.T, atol=1e-25)


def test_shared_face_orientation_anchors_moments_and_stiffness(band):
    E = band.config.bulk_material.youngs_modulus_pa
    G = E/(2*(1+band.config.bulk_material.poisson_ratio))
    for f in band.interfaces:
        points = band.vertices[list(f.vertex_ids)]
        area, centre, frame, _, covariance = face_geometry(points)
        assert f.area_m2 == pytest.approx(area)
        assert sum(f.weights_m2) == pytest.approx(area)
        assert np.all(f.weights_m2 > 0)
        np.testing.assert_allclose(np.average(f.anchors, axis=0, weights=f.weights_m2), centre, atol=1e-16)
        uv = (f.anchors-centre)@frame[:, 1:]
        # Roundoff tolerance scales with face moments, not a near-zero cross term.
        np.testing.assert_allclose(uv.T@np.diag(f.weights_m2/area)@uv, covariance,
                                   rtol=1e-12, atol=32*np.finfo(float).eps*np.trace(covariance))
        assert np.linalg.matrix_rank(f.anchors[1:]-f.anchors[0]) == 2
        assert np.dot(band.cells[f.b].centre-band.cells[f.a].centre, f.frame[:, 0]) == pytest.approx(f.h_m)
        for idx in (f.a, f.b):
            local = f.anchors-band.cells[idx].centre
            np.testing.assert_allclose(local+band.cells[idx].centre, f.anchors, atol=1e-18)
            for face in band.cells[idx].faces:
                _, fc, ff, _, _ = face_geometry(band.vertices[list(face)])
                assert np.max((f.anchors-fc)@ff[:, 0]) < 1e-14
        k = np.array([1e8, 2e8, 2e8]) if f.fracture else np.array([E, G, G])/f.h_m
        np.testing.assert_allclose(f.stiffness_n_m, f.weights_m2[:, None]*k, rtol=1e-13)


def test_partition_covers_interior_once_and_is_watertight(band):
    assert validate(band)["cells"] == 32
    rng = np.random.default_rng(31)
    # Stay inside polygon's inradius and away from shared boundaries.
    radii = band.config.radius_m*.97*np.sqrt(rng.uniform(.01, 1, 100))
    angles = rng.uniform(0, 2*np.pi, 100)
    points = np.column_stack((radii*np.cos(angles), radii*np.sin(angles), rng.uniform(-.00399, .00399, 100)))
    count = np.zeros(len(points), int)
    for cell in band.cells:
        inside = np.ones(len(points), bool)
        for face in cell.faces:
            _, centre, frame, _, _ = face_geometry(band.vertices[list(face)])
            inside &= (points-centre)@frame[:, 0] <= 1e-14
        count += inside
    np.testing.assert_array_equal(count, np.ones(len(points), int))


@pytest.fixture
def stage():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    s = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(s, 1.)
    return s


@pytest.mark.parametrize("supported", [False, True])
def test_usd_contacts_dynamic_cells_and_disabled_fracture(stage, band, supported):
    from pxr import UsdPhysics
    fixture = author(stage, band, root="/World/Band", proximal_support=supported)
    assert not fixture["simulation_ready"]
    actual_filters = set()
    for cell, path in zip(band.cells, fixture["paths"]):
        prim = stage.GetPrimAtPath(path)
        assert not UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
        assert UsdPhysics.MassAPI(prim).GetMassAttr().Get() == pytest.approx(cell.mass_kg, rel=1e-6)
        assert UsdPhysics.MassAPI(prim).GetDensityAttr().Get() == 950
        assert UsdPhysics.CollisionAPI(stage.GetPrimAtPath(path+"/Solid")).GetCollisionEnabledAttr().Get()
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            for other in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                actual_filters.add((cell.index, fixture["paths"].index(str(other))))
    assert actual_filters == band.bulk_pairs
    assert len(fixture["proximal_support"]) == (24 if supported else 0)
    for record in fixture["interfaces"]:
        face = record["face"]
        assert len(record["states"]) == (3 if face.fracture else 0)
        for s in record["states"]:
            assert s.damage == 0 and s.maximum_effective_separation_m == 0
        for anchor in record["anchors"]:
            j = anchor["joint"]
            assert j.GetJointEnabledAttr().Get() == (not face.fracture)
            assert j.GetCollisionEnabledAttr().Get()
            for axis in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
                assert not j.GetPrim().HasAPI(UsdPhysics.LimitAPI, axis)
            for d in anchor["drives"]:
                assert d.GetDampingAttr().Get() == d.GetTargetPositionAttr().Get() == 0
                assert d.GetTypeAttr().Get() == "force"
                assert np.isinf(d.GetMaxForceAttr().Get())
    assert not any(p.IsA(UsdPhysics.FixedJoint) or p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse())


def test_usd_rigid_transform_coincident_anchors_and_remote_support(stage, band):
    from pxr import UsdGeom
    pose = np.array([[0., -1, 0, .2], [1, 0, 0, -.1], [0, 0, 1, 1.], [0, 0, 0, 1]])
    f = author(stage, band, root="/World/Band", proximal_support=True, transform=pose)
    cache = UsdGeom.XformCache()
    for record in f["interfaces"]:
        for point, anchor in zip(record["face"].anchors, record["anchors"]):
            expected = pose[:3, :3]@point+pose[:3, 3]
            for side in (0, 1):
                joint = anchor["joint"]
                body = stage.GetPrimAtPath(getattr(joint, f"GetBody{side}Rel")().GetTargets()[0])
                matrix = np.asarray(cache.GetLocalToWorldTransform(body)).T
                local = getattr(joint, f"GetLocalPos{side}Attr")().Get()
                np.testing.assert_allclose(matrix@np.r_[local, 1], np.r_[expected, 1], atol=2e-9)
    for anchor in f["proximal_support"]:
        joint = anchor["joint"]
        assert not joint.GetBody0Rel().GetTargets()
        assert joint.GetLocalPos0Attr().Get()[2] == pytest.approx(1-band.config.length_m/2)
        index = f["paths"].index(str(joint.GetBody1Rel().GetTargets()[0]))
        assert band.cells[index].layer == 0


@pytest.mark.parametrize("transform", [np.diag([2., 1, 1, 1]), np.diag([-1., 1, 1, 1]),
    np.diag([1.000001, 1., 1., 1.]), np.full((4, 4), np.nan)])
def test_author_rejects_geometry_changes_before_writing(stage, band, transform):
    with pytest.raises(ValueError, match="transform"):
        author(stage, band, root="/World/Band", proximal_support=False, transform=transform)
    assert not stage.GetPrimAtPath("/World/Band")


def test_no_source_or_existing_stage_mutation(stage, band):
    from pxr import UsdGeom, UsdPhysics
    unrelated = UsdGeom.Cube.Define(stage, "/World/Existing")
    UsdPhysics.CollisionAPI.Apply(unrelated.GetPrim())
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match="Isolated"):
        author(stage, band, root="/World/Band", proximal_support=False)
    assert stage.GetRootLayer().ExportToString() == before


def test_usd_full_inertia_and_shared_vertices_no_hidden_proxy(stage, band):
    from pxr import Gf, UsdGeom, UsdPhysics
    # Parent scale must not silently resize the explicitly world-space band.
    UsdGeom.Xform.Define(stage, "/World").AddScaleOp().Set(Gf.Vec3f(2.))
    fixture = author(stage, band, root="/World/Band", proximal_support=False)
    cache = UsdGeom.XformCache()
    occurrences = {}
    for cell, path in zip(band.cells, fixture["paths"]):
        body = stage.GetPrimAtPath(path)
        mass = UsdPhysics.MassAPI(body)
        principal = mass.GetPrincipalAxesAttr().Get()
        rotation = np.asarray(Gf.Matrix3d(Gf.Quatd(principal))).T
        inertia = rotation@np.diag(mass.GetDiagonalInertiaAttr().Get())@rotation.T
        # USD principal axes and moments are float32, unlike exact geometry.
        np.testing.assert_allclose(inertia, cell.inertia_kg_m2, rtol=2e-6,
            atol=8*np.finfo(np.float32).eps*np.linalg.norm(cell.inertia_kg_m2))
        mesh = UsdGeom.Mesh.Get(stage, path+"/Solid")
        matrix = np.asarray(cache.GetLocalToWorldTransform(body)).T
        local = np.asarray(mesh.GetPointsAttr().Get())
        world = local@matrix[:3, :3].T+matrix[:3, 3]
        for vertex_id, point in zip(cell.vertex_ids, world):
            np.testing.assert_allclose(point, band.vertices[vertex_id], atol=3e-10, rtol=0)
            if vertex_id in occurrences:
                np.testing.assert_allclose(point, occurrences[vertex_id], atol=3e-10, rtol=0)
            occurrences[vertex_id] = point
    colliders = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    assert len(colliders) == 32
    assert all(p.GetTypeName() == "Mesh" for p in colliders)


@pytest.mark.parametrize("layers", [4, 8])
@pytest.mark.parametrize("radius,length", [(.0005, .008), (.0005, .01), (.01, .008), (.01, .01)])
def test_geometry_bounds_keep_anchor_quadrature_valid(config, layers, radius, length):
    b = build(replace(config, axial_layers=layers, radius_m=radius, length_m=length))
    assert validate(b)["cells"] == 8*layers
    for f in b.interfaces:
        assert np.isfinite(f.anchors).all() and np.all(f.weights_m2 > 0)
        np.testing.assert_allclose(np.mean(f.anchors, axis=0), f.centre, atol=1e-16)


@pytest.mark.parametrize("tamper", ["mass", "inertia", "centre", "neighbour", "stiffness",
    "weights", "anchors", "frame", "area", "fracture", "boundary", "vertices", "winding"])
def test_mutated_band_rejected_before_any_stage_write(stage, band, tamper):
    f = next(f for f in band.interfaces if not f.fracture and band.cells[f.a].layer == band.cells[f.b].layer)
    if tamper == "mass": band.cells[0].mass_kg = -1.
    elif tamper == "inertia": band.cells[0].inertia_kg_m2[0, 0] = -1.
    elif tamper == "centre": band.cells[0].centre[0] += 1e-5
    elif tamper == "neighbour": f.b = (f.a+3)%8
    elif tamper == "stiffness": f.stiffness_n_m[0, 0] = -1.
    elif tamper == "weights": f.weights_m2 *= 2
    elif tamper == "anchors": f.anchors[0] += 1e-5*f.frame[:, 0]
    elif tamper == "frame": f.frame[:, 0] *= -1
    elif tamper == "area": f.area_m2 *= 2
    elif tamper == "fracture": f.fracture = True
    elif tamper == "boundary": band.boundary_faces[0] = band.boundary_faces[1]
    elif tamper == "vertices": band.vertices[0, 0] = np.nan
    elif tamper == "winding":
        cell = band.cells[0]
        cell.faces = (tuple(reversed(cell.faces[0])), *cell.faces[1:])
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError):
        author(stage, band, root="/World/Band", proximal_support=False)
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize("points", [[], [[0., 0., 0.]], np.zeros((3, 3)),
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, .01]],
    [[0, 0, 0], [1, 0, 0], [0, np.nan, 0]]])
def test_malformed_faces_fail_closed(points):
    with pytest.raises(ValueError):
        face_geometry(points)


def test_nonfinite_polygon_and_near_metre_stage_rejected(stage, band):
    from pxr import UsdGeom
    with pytest.raises(ValueError):
        polygon_moments([[0., 0.], [1., 0.], [0., np.nan]])
    UsdGeom.SetStageMetersPerUnit(stage, 1.000001)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match="metre"):
        author(stage, band, root="/World/Band", proximal_support=False)
    assert stage.GetRootLayer().ExportToString() == before


def test_refined_usd_preserves_fracture_and_external_contacts(stage, config):
    from pxr import UsdGeom, UsdPhysics
    b = build(replace(config, axial_layers=8))
    f = author(stage, b, root="/World/Band", proximal_support=True)
    knife = UsdGeom.Cube.Define(stage, "/World/ExternalKnife")
    UsdPhysics.CollisionAPI.Apply(knife.GetPrim())
    assert len(f["paths"]) == 64 and len(f["proximal_support"]) == 24
    filtered = set()
    for i, path in enumerate(f["paths"]):
        api = UsdPhysics.FilteredPairsAPI(stage.GetPrimAtPath(path))
        if api:
            for target in api.GetFilteredPairsRel().GetTargets():
                assert str(target) != str(knife.GetPath())
                filtered.add((i, f["paths"].index(str(target))))
    assert filtered == b.bulk_pairs
    assert all((x.a, x.b) not in filtered for x in b.interfaces if x.fracture)
    assert sum(len(x["states"]) for x in f["interfaces"]) == 24
    assert not f["simulation_ready"]


def test_authored_facet_state_closing_reopening_contract_not_native(stage, band):
    """Pure law integration contract only: no joint enabled or body stepped."""
    fixture = author(stage, band, root="/World/Band", proximal_support=False)
    before = stage.GetRootLayer().ExportToString()
    for record in fixture["interfaces"]:
        face = record["face"]
        if not face.fracture:
            continue
        for virgin, area in zip(record["states"], face.weights_m2):
            normal = face.frame[:, 0]
            opened = cohesive_response(virgin, 2e-4*normal, normal, area_m2=area)
            assert 0 < opened.state.damage < 1
            compressed = cohesive_response(opened.state, -2e-5*normal, normal, area_m2=area)
            np.testing.assert_allclose(compressed.force_a_n, 0., atol=1e-14)
            assert compressed.state.damage == opened.state.damage
            assert compressed.dissipation_increment_j == 0
            reopened = cohesive_response(compressed.state, 1.5e-4*normal, normal, area_m2=area)
            assert reopened.state.damage == opened.state.damage
            expected = area*opened.state.retained_stiffness*1e8*1.5e-4*normal
            np.testing.assert_allclose(reopened.force_a_n, expected)
            failed = cohesive_response(reopened.state, 4.1e-4*normal, normal, area_m2=area)
            assert failed.state.fully_separated
            for opening in (-2e-5, 0., 1e-4):
                result = cohesive_response(failed.state, opening*normal, normal, area_m2=area)
                assert result.state.fully_separated
                np.testing.assert_array_equal(result.force_a_n, [0., 0., 0.])
            assert virgin.damage == 0  # Trial evaluation did not commit history.
    assert stage.GetRootLayer().ExportToString() == before
    assert all(not a["joint"].GetJointEnabledAttr().Get()
               for r in fixture["interfaces"] if r["face"].fracture for a in r["anchors"])
