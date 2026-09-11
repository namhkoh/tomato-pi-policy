"""Pure geometry/in-memory USD only. Never imports or launches SimulationApp."""
from dataclasses import replace

import numpy as np
import pytest

from sim_physics import seam_interface as seam
from sim_physics.cohesive import cohesive_response


@pytest.fixture
def config():
    return seam.SeamConfig("unloaded/manifest.json", "explicit/SubStem", .003,
                           seam.CohesiveParameters(1e8, 2e8, 1e4, 2.))


def rotation(axis, angle):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    x, y, z = a
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3)+np.sin(angle)*skew+(1-np.cos(angle))*(skew@skew)


def setup(plane=None, body_rotations=None):
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateInMemory(); UsdGeom.SetStageMetersPerUnit(stage, 1.)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    UsdGeom.Xform.Define(stage, "/World")
    plane = np.eye(4) if plane is None else plane
    body_rotations = [np.eye(3), np.eye(3)] if body_rotations is None else body_rotations
    for path, distance, orient in zip(("/World/A", "/World/B"), (-.005, .005), body_rotations):
        body = UsdGeom.Xform.Define(stage, path)
        body.AddTranslateOp().Set(Gf.Vec3d(*(plane[:3, 3]+distance*plane[:3, 0])))
        matrix = np.eye(4); matrix[:3, :3] = orient
        body.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Matrix4d(matrix.T.tolist()).ExtractRotationQuat())
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim()).CreateKinematicEnabledAttr(False)
        m = UsdPhysics.MassAPI.Apply(body.GetPrim()); m.CreateMassAttr(.000267); m.CreateDensityAttr(950.)
        # Existing caller-owned flat-ended collider: the seam author must not touch it.
        shape = UsdGeom.Cylinder.Define(stage, path+"/FlatContact")
        shape.CreateRadiusAttr(.003); shape.CreateHeightAttr(.01); shape.CreateAxisAttr("X")
        UsdPhysics.CollisionAPI.Apply(shape.GetPrim())
    return stage


def attach(stage, config, **kwargs):
    return seam.author(stage, root="/World/Seam", body_a="/World/A", body_b="/World/B",
                       plane_world=kwargs.pop("plane_world", np.eye(4)), config=config, **kwargs)


def snapshot(stage):
    return {str(p.GetPath()): (p.GetTypeName(), tuple(p.GetAppliedSchemas()),
            [(a.GetName(), repr(a.Get())) for a in p.GetAttributes()],
            [(r.GetName(), tuple(map(str, r.GetTargets()))) for r in p.GetRelationships()])
            for p in stage.Traverse()}


@pytest.mark.parametrize("radius", [.0005, .003, .01])
def test_sector_quadrature_matches_independent_triangle_moments(config, radius):
    geometry = seam.partition(replace(config, radius_m=radius))
    assert len(geometry["facets"]) == 8 and geometry["anchors_m"].shape == (24, 3)
    areas = []
    for face in geometry["facets"]:
        vertices, q, w = face["vertices_m"], face["anchors_m"], face["weights_m2"]
        area = 0.; first = np.zeros(3); second = np.zeros((3, 3))
        for i in range(1, 5):
            tri = vertices[[0, i, i+1]]
            a = .5*np.linalg.norm(np.cross(tri[1]-tri[0], tri[2]-tri[0]))
            total = tri.sum(axis=0)
            area += a; first += a*total/3
            second += a*(tri.T@tri+np.outer(total, total))/12
        assert w.sum() == pytest.approx(area, rel=1e-12)
        np.testing.assert_allclose(w@q, first, rtol=1e-11, atol=area*radius*1e-12)
        np.testing.assert_allclose(np.einsum("n,ni,nj->ij", w, q, q), second,
                                   rtol=1e-10, atol=area*radius**2*1e-12)
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            assert np.min(np.cross(end-start, q-start)[:, 0]) >= -radius**2*1e-12
        areas.append(area)
    assert sum(areas) == pytest.approx(16*radius**2*np.sin(np.pi/16), rel=1e-12)
    assert geometry["circular_area_fraction"] == pytest.approx(.9935868511442058)


def test_area_stiffness_and_affine_rotation_energy_are_partition_exact(config):
    g = seam.partition(config); q, w = g["anchors_m"], g["weights_m2"]
    k = np.array([config.material.normal_stiffness_pa_m]+[config.material.shear_stiffness_pa_m]*2)
    translation = np.array([5e-6, 2e-6, -1e-6]); omega = np.array([.001, .0002, -.0003])
    jump = translation+np.cross(omega, q)
    discrete = .5*np.sum(w[:, None]*k*jump**2)
    W = np.array([[0., -omega[2], omega[1]], [omega[2], 0., -omega[0]], [-omega[1], omega[0], 0.]])
    reference = .5*(g["area_m2"]*np.sum(k*translation**2)+np.trace(np.diag(k)@W@g["second_area_moment_m4"]@W.T))
    assert discrete == pytest.approx(reference, rel=1e-12)
    assert g["area_m2"]*k[0] == pytest.approx(2809.3006370322454)


@pytest.mark.parametrize("transform", [False, True])
def test_author_only_existing_bodies_frames_states_and_disabled_constraints(config, transform):
    from pxr import Gf, UsdGeom, UsdPhysics
    plane = np.eye(4)
    if transform:
        plane[:3, :3] = rotation([1, 2, 3], .43); plane[:3, 3] = [1.4, -.3, .7]
    stage = setup(plane, [rotation([1, -2, 1], .21), rotation([2, 1, -1], -.34)])
    old = UsdPhysics.FixedJoint.Define(stage, "/World/OldWeld")
    old.CreateBody0Rel().SetTargets(["/World/A"]); old.CreateBody1Rel().SetTargets(["/World/B"])
    before = snapshot(stage)
    fixture = attach(stage, config, plane_world=plane)
    after = snapshot(stage)
    assert all(after[path] == values for path, values in before.items())
    assert len(fixture["links"]) == 24 and len({id(h.state) for h in fixture["links"]}) == 24
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]) == 2
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]) == 2
    assert not any(p.HasAPI(UsdPhysics.FilteredPairsAPI) for p in stage.Traverse())
    assert fixture["summary"]["existing_direct_joints"] == [dict(path="/World/OldWeld", type="PhysicsFixedJoint", enabled=True)]
    assert not fixture["summary"]["native_response_validated"]
    assert not fixture["summary"]["flat_contact_geometry_verified"]
    cache = UsdGeom.XformCache()
    for i, link in enumerate(fixture["links"]):
        j = link.joint; prim = j.GetPrim()
        assert not j.GetJointEnabledAttr().Get() and j.GetCollisionEnabledAttr().Get()
        assert j.GetExcludeFromArticulationAttr().Get()
        assert not prim.HasAPI(UsdPhysics.DriveAPI, "transX")
        assert link.limit.GetLowAttr().Get() == float(np.float32(-.002))
        assert link.limit.GetHighAttr().Get() == 0 and link.soft["damping"].Get() == 0
        positions = []
        for side, path in enumerate(("/World/A", "/World/B")):
            assert list(map(str, getattr(j, f"GetBody{side}Rel")().GetTargets())) == [path]
            body = np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path))).T
            local = np.array(getattr(j, f"GetLocalPos{side}Attr")().Get())
            positions.append(body[:3, :3]@local+body[:3, 3])
            quat = getattr(j, f"GetLocalRot{side}Attr")().Get()
            R = np.asarray(Gf.Matrix3d(Gf.Quatd(quat.GetReal(), Gf.Vec3d(*quat.GetImaginary())))).T
            np.testing.assert_allclose(body[:3, :3]@R, plane[:3, :3], atol=2e-6, rtol=0)
        target = plane[:3, :3]@fixture["geometry"]["anchors_m"][i]+plane[:3, 3]
        np.testing.assert_allclose(positions, np.broadcast_to(target, (2, 3)), atol=2e-8, rtol=0)
        assert link.area_m2 == fixture["geometry"]["weights_m2"][i]
        assert link.tangents[0].GetStiffnessAttr().Get() == pytest.approx(link.area_m2*config.material.shear_stiffness_pa_m, rel=2e-6)
    assert stage.GetPrimAtPath("/World/Seam").GetAttribute("seam:authoringComplete").Get()


def test_preexisting_states_partial_failed_and_unbonded_keep_disabled(config):
    from pxr import UsdPhysics
    virgin = seam.CohesiveState(config.material)
    partial = cohesive_response(virgin, [.0002, 0, 0], [1, 0, 0], area_m2=1e-6).state
    failed = cohesive_response(virgin, [.0005, 0, 0], [1, 0, 0], area_m2=1e-6).state
    partial = cohesive_response(partial, [0, 0, 0], [1, 0, 0], area_m2=1e-6).state
    failed = cohesive_response(failed, [0, 0, 0], [1, 0, 0], area_m2=1e-6).state
    unbonded = seam.CohesiveState(config.material, bonded=False)
    states = tuple([virgin, partial, failed, unbonded]*6)
    stage = setup(); f = attach(stage, config, states=states)
    for h, state in zip(f["links"], states):
        assert h.state is state and not h.joint.GetJointEnabledAttr().Get()
        assert h.tangents[0].GetStiffnessAttr().Get() == pytest.approx(h.area_m2*state.retained_stiffness*config.material.shear_stiffness_pa_m, rel=2e-6)
        if state.retained_stiffness == 0:
            assert h.limit is None and not h.joint.GetPrim().HasAPI(UsdPhysics.LimitAPI, "transX")


@pytest.mark.parametrize("kind", ["kinematic", "disabled", "inactive", "scale", "reflection", "wrong_normal", "units", "states", "material", "animated", "loaded_state"])
def test_bad_input_fails_before_new_authoring(config, kind):
    from pxr import Gf, UsdGeom, UsdPhysics
    stage = setup(); plane = np.eye(4); kwargs = {}
    if kind == "kinematic": UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath("/World/A")).GetKinematicEnabledAttr().Set(True)
    elif kind == "disabled": UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath("/World/A")).CreateRigidBodyEnabledAttr(False)
    elif kind == "inactive": stage.GetPrimAtPath("/World/A").SetActive(False)
    elif kind == "scale": UsdGeom.Xformable(stage.GetPrimAtPath("/World/A")).AddScaleOp().Set(Gf.Vec3f(2.))
    elif kind == "reflection": plane[1, 1] = -1.
    elif kind == "wrong_normal": plane[0, 0] = -1.; plane[1, 1] = -1.
    elif kind == "units": UsdGeom.SetStageMetersPerUnit(stage, .01)
    elif kind == "states": kwargs["states"] = [seam.CohesiveState(config.material)]*23
    elif kind == "material": kwargs["states"] = [seam.CohesiveState(seam.CohesiveParameters(2e8, 2e8, 1e4, 2.))]*24
    elif kind == "animated": UsdGeom.Xformable(stage.GetPrimAtPath("/World/A")).GetOrderedXformOps()[0].Set(Gf.Vec3d(-.005, 0, 0), 1.)
    elif kind == "loaded_state": kwargs["states"] = [cohesive_response(seam.CohesiveState(config.material), [.0002, 0, 0], [1, 0, 0], area_m2=1e-6).state]*24
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError): attach(stage, config, plane_world=plane, **kwargs)
    assert stage.GetRootLayer().ExportToString() == before


def test_no_overwrite_or_cell_builder_and_explicit_inputs(config, monkeypatch):
    from sim_physics import material_band
    monkeypatch.setattr(material_band, "build", lambda *a, **k: pytest.fail("No material cell construction allowed"))
    stage = setup(); attach(stage, config)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError): attach(stage, config)
    assert stage.GetRootLayer().ExportToString() == before
    with pytest.raises(TypeError): seam.SeamConfig()
    for bad in (True, float("nan"), 0., -1., "0.003"):
        with pytest.raises(ValueError): replace(config, radius_m=bad)
    with pytest.raises(ValueError): replace(config, source_target="")


@pytest.mark.parametrize("scalar", [np.float32, np.float64, np.longdouble])
def test_numpy_radius_is_normalized_and_summary_json_safe(config, scalar):
    import json
    c = replace(config, radius_m=scalar(.003))
    assert type(c.radius_m) is float
    f = attach(setup(), c)
    json.dumps(f["summary"], allow_nan=False)


def test_collapsed_rounded_anchors_are_rejected_before_authoring(config):
    stage = setup(body_rotations=[rotation([1, 2, 3], .43)]*2)
    before = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match="anchor moments"):
        attach(stage, replace(config, radius_m=1e-12))
    assert stage.GetRootLayer().ExportToString() == before
