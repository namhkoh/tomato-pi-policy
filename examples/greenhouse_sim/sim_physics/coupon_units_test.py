from types import SimpleNamespace
import numpy as np
import pytest

from .coupon_units import length_scale, poses_to_si, velocities_to_si, contacts_to_si, NativeSI, author_scaled, check_drive_contract
from .contact_spring_probe_test import coupon as make_coupon


@pytest.fixture
def coupon():
    return make_coupon()


@pytest.mark.parametrize('bad', [True, False, 1., 100., 0, -1, 10, None, '100'])
def test_scale_is_explicit(bad):
    with pytest.raises(ValueError): length_scale(bad)


def test_pose_and_velocity_dimensions_and_copy():
    pose = np.array([[100., 200., -300., 0., 0., 0., 1.]])
    velocity = np.array([[100., 200., -300., 1., 2., 3.]])
    np.testing.assert_array_equal(poses_to_si(pose, 100), [[1, 2, -3, 0, 0, 0, 1]])
    np.testing.assert_array_equal(velocities_to_si(velocity, 100), [[1, 2, -3, 1, 2, 3]])
    assert pose[0, 0] == velocity[0, 0] == 100


def test_contacts_preserve_sign_normal_and_friction():
    source = [dict(kind='normal', point_world_m=[100, 0, 0], impulse_on_0_ns=[-2, 0, 0],
                   normal_on_0=[-1, 0, 0], separation_m=-.1),
              dict(kind='friction', point_world_m=[0, 100, 0], impulse_on_0_ns=[0, -1, 0])]
    out = contacts_to_si(source, 100)
    assert out[0]['normal_on_0'] == [-1, 0, 0]
    assert out[0]['impulse_on_0_ns'] == [-.02, 0, 0]
    assert out[0]['separation_m'] == -.001
    assert out[1]['impulse_on_0_ns'] == [0, -.01, 0]
    assert source[0]['point_world_m'] == [100, 0, 0]


def test_native_read_only_and_torque_units():
    view = NativeSI(SimpleNamespace(get_dof_stiffnesses=lambda: np.array([[1000.]])), 100)
    np.testing.assert_array_equal(view.get_dof_stiffnesses(), [[.1]])
    with pytest.raises(AttributeError): view.set_dof_positions
    with pytest.raises(AttributeError): view.get_generalized_mass_matrices


@pytest.mark.parametrize('scale', [1, 100])
def test_authored_physical_equivalence(coupon, scale):
    from pxr import Usd, UsdGeom, UsdPhysics
    from .contact_spring_probe import author
    reference = Usd.Stage.CreateInMemory(); stage = Usd.Stage.CreateInMemory()
    data = author(reference, coupon)
    _, receipt = author_scaled(stage, coupon, scale)
    assert UsdGeom.GetStageMetersPerUnit(stage) == 1/scale
    assert UsdPhysics.GetStageKilogramsPerUnit(stage) == 1
    assert receipt['physical_parameters_tuned'] is False
    cache = UsdGeom.XformCache()
    for path in data['body_paths'] + list(data['pads']):
        before = np.array(cache.GetLocalToWorldTransform(reference.GetPrimAtPath(path)))
        after = np.array(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)))
        np.testing.assert_allclose(after[3, :3]/scale, before[3, :3])
        if path in data['pads']:
            np.testing.assert_allclose(after[:3, :3]/scale, before[:3, :3])
        else:
            np.testing.assert_allclose(after[:3, :3], before[:3, :3])
            a = stage.GetPrimAtPath(path); b = reference.GetPrimAtPath(path)
            np.testing.assert_allclose(np.array(a.GetAttribute('physics:diagonalInertia').Get())/scale**2,
                                       b.GetAttribute('physics:diagonalInertia').Get(), rtol=2e-7)
            assert a.GetAttribute('physics:mass').Get() == b.GetAttribute('physics:mass').Get()
    for path in data['joint_paths']:
        for attr in ('physics:localPos0', 'physics:localPos1'):
            np.testing.assert_allclose(np.array(stage.GetPrimAtPath(path).GetAttribute(attr).Get())/scale,
                                       reference.GetPrimAtPath(path).GetAttribute(attr).Get(), rtol=2e-7)
        for attr in ('stiffness', 'damping'):
            key = 'drive:angular:physics:'+attr
            assert stage.GetPrimAtPath(path).GetAttribute(key).Get()/scale**2 == pytest.approx(
                reference.GetPrimAtPath(path).GetAttribute(key).Get(), rel=2e-7)
    material = stage.GetPrimAtPath(coupon.root+'/ContactMaterial')
    assert material.GetAttribute('physxMaterial:compliantContactStiffness').Get() == pytest.approx(coupon.contact_stiffness)


def test_refuses_populated_stage(coupon):
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory(); UsdGeom.Xform.Define(stage, '/UserAsset')
    with pytest.raises(ValueError): author_scaled(stage, coupon, 100)


@pytest.mark.parametrize('bad', [None, 'get_drive_types', 'get_dof_stiffnesses', 'get_dof_dampings',
                                'get_dof_position_targets', 'get_dof_velocity_targets', 'get_dof_actuation_forces'])
def test_drive_checks_read_only(coupon, bad):
    values = dict(get_drive_types=np.ones((1, 2)), get_dof_stiffnesses=np.array([coupon.stiffness]),
                  get_dof_dampings=np.array([coupon.damping]), get_dof_position_targets=np.zeros((1, 2)),
                  get_dof_velocity_targets=np.zeros((1, 2)), get_dof_actuation_forces=np.zeros((1, 2)))
    if bad: values[bad][0, 0] += .1
    view = SimpleNamespace(**{k: (lambda v=v: v.copy()) for k, v in values.items()})
    if bad:
        with pytest.raises(RuntimeError, match=bad): check_drive_contract(view, coupon)
    else:
        assert len(check_drive_contract(view, coupon)) == 6
