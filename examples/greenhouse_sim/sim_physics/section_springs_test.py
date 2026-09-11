"""Pure mechanics and in-memory USD tests; no SimulationApp/native stepping."""
from copy import deepcopy
from dataclasses import replace
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from .section_springs import (MAX_FORCE, author_pair, coefficients, displacements,
    displacement_rates, inspect_pair, local_anchors, planar_response)
from .contact_spring_probe import (RADIUS_M, SPACING_M, _ry, author, bind, layout,
    sample, assess_tail, bind_rigid, rigid_coordinates, settings_readback,
    modeled_energy, assess_free_energy)
from .contact_spring_probe_test import coupon, FakeArticulation, equilibrium, tail


def pair(theta, omega=0.):
    a = np.eye(4); b = np.eye(4); b[:3, :3] = _ry(theta)
    local0 = np.eye(4); local0[2, 3] = SPACING_M/2
    local1 = np.eye(4); local1[2, 3] = -SPACING_M/2
    b[:3, 3] = a[:3, 3]+local0[:3, 3]-b[:3, :3]@local1[:3, 3]
    velocities = np.zeros((2, 6)); velocities[1, 4] = omega
    velocities[1, :3] = -np.cross(velocities[1, 3:], b[:3, :3]@local1[:3, 3])
    return a, b, local0, local1, velocities


def test_import_without_usd_native_or_app():
    env = os.environ.copy(); env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    code = ('import sys; import sim_physics.section_springs; '
            'assert not any(n.startswith(("pxr", "omni", "isaacsim")) for n in sys.modules)')
    done = subprocess.run([sys.executable, '-B', '-c', code], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_coefficients_copy_K_C_without_angular_unit_conversion():
    k, c, r = .3328997492790222, .00940549187362194, .0028
    p = coefficients(k, c, r)
    assert p['linear_stiffness_n_m'] == k/(2*r*r)
    assert p['linear_damping_n_s_m'] == c/(2*r*r)
    assert p['connector_count'] == 2 and not p['native_qualified']


@pytest.mark.parametrize('args', [(0, 1, .0028), (1, 0, .0028), (1, 1, 0),
    (1, 1, True), (True, 1, .0028), (float('nan'), 1, .0028),
    (1, float('inf'), .0028), (1, 1, 1e-300), (1, 1, 1e300),
    (1e300, 1, .0028), (1e-300, 1, .0028)])
def test_invalid_or_unrepresentable_coefficients(args):
    with pytest.raises(ValueError): coefficients(*args)


@pytest.mark.parametrize('theta', [-.04, -.005, 0., .005, .04, .3])
@pytest.mark.parametrize('omega', [-.7, 0., .7])
def test_energy_torque_damping_and_continuous_passivity(theta, omega):
    k, c, r = .33, .009, RADIUS_M
    x = planar_response(theta, omega, k, c, r)
    a, b, l0, l1, vel = pair(theta, omega)
    z = displacements(a, b, l0, l1, r)
    dzdt = displacement_rates(a, b, vel[0], vel[1], l0, l1, r)
    np.testing.assert_allclose(z, x['displacement_z_m'], atol=1e-17)
    np.testing.assert_allclose(dzdt, x['velocity_z_m_s'], atol=1e-17)
    assert .5*k/(2*r*r)*(z@z) == pytest.approx(x['energy_j'], rel=1e-13, abs=1e-18)
    assert x['elastic_torque_nm'] == pytest.approx(-k*math.sin(theta)*math.cos(theta))
    assert x['damping_torque_nm'] == pytest.approx(-c*math.cos(theta)**2*omega)
    assert x['damping_power_w'] <= 0
    assert x['total_power_w']+x['energy_rate_w'] == pytest.approx(x['damping_power_w'], abs=1e-16)
    dzdq = np.array([1., -1.])*r*math.cos(theta)
    assert np.dot(x['force_on_child_z_n'], dzdq) == pytest.approx(x['total_torque_nm'], abs=1e-16)
    h = 1e-7
    derivative = (planar_response(theta+h, 0, k, c, r)['energy_j']
                 -planar_response(theta-h, 0, k, c, r)['energy_j'])/(2*h)
    assert -derivative == pytest.approx(x['elastic_torque_nm'], rel=1e-8, abs=1e-11)


def test_small_angle_matches_but_finite_angle_is_not_linear_beam():
    q = 1e-6; x = planar_response(q, .2, .3, .008, RADIUS_M)
    assert x['elastic_torque_nm']/(-q) == pytest.approx(.3, rel=1e-11)
    assert x['damping_torque_nm']/(-.2) == pytest.approx(.008, rel=1e-11)
    at_right_angle = planar_response(math.pi/2, .2, .3, .008, RADIUS_M)
    assert abs(at_right_angle['elastic_torque_nm']) < 1e-15
    assert abs(at_right_angle['damping_torque_nm']) < 1e-15


def test_local_anchor_transforms_at_rest_and_no_strained_reanchoring():
    a, b, l0, l1, vel = pair(0)
    anchors = local_anchors(l0, l1, RADIUS_M)
    for first, second in anchors:
        np.testing.assert_allclose(a@first, b@second, atol=1e-15)
    a, b, _, _, _ = pair(.005)
    assert np.min(abs(displacements(a, b, l0, l1, RADIUS_M))) > 1e-5
    frame = np.eye(4); frame[:3, :3] = _ry(.6); frame[:3, 3] = [1, 2, 3]
    rotated = local_anchors(frame@l0, frame@l1, RADIUS_M)
    np.testing.assert_allclose(rotated, frame@anchors, atol=1e-14)


def test_rigid_world_pose_and_velocity_invariance_with_strained_anchors():
    a, b, l0, l1, vel = pair(.04, .3)
    transform = np.eye(4); transform[:3, :3] = _ry(.7)
    cz, sz = math.cos(.8), math.sin(.8)
    transform[:3, :3] = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@transform[:3, :3]
    transform[:3, 3] = [1., -2., .9]
    rotation = transform[:3, :3]; moved = [transform@a, transform@b]
    common_v = np.array([.3, -.1, .2]); common_w = np.array([.9, -.7, .4])
    moved_vel = np.array([np.r_[rotation@v[:3]+common_v+np.cross(common_w, f[:3, 3]),
                               rotation@v[3:]+common_w] for v, f in zip(vel, moved)])
    np.testing.assert_allclose(displacements(*moved, l0, l1, RADIUS_M),
                               displacements(a, b, l0, l1, RADIUS_M), atol=1e-15)
    np.testing.assert_allclose(displacement_rates(*moved, *moved_vel, l0, l1, RADIUS_M),
                               displacement_rates(a, b, *vel, l0, l1, RADIUS_M), atol=1e-15)
    common_only = np.array([np.r_[common_v+np.cross(common_w, f[:3, 3]), common_w] for f in moved])
    np.testing.assert_allclose(displacement_rates(*moved, *common_only, l0, l1, RADIUS_M), 0, atol=1e-15)


def test_axial_anchor_error_adds_real_energy_and_K_over_r_squared_stiffness():
    a, b, l0, l1, _ = pair(.02); p = coefficients(.3, .008, RADIUS_M)
    original = displacements(a, b, l0, l1, RADIUS_M)
    delta = 2e-6; b[2, 3] += delta
    moved = displacements(a, b, l0, l1, RADIUS_M)
    assert np.mean(moved) == pytest.approx(delta, abs=1e-17)
    increase = .5*p['linear_stiffness_n_m']*(moved@moved-original@original)
    assert increase == pytest.approx(.5*.3/RADIUS_M**2*delta**2, rel=1e-10)


@pytest.mark.parametrize('bad', ['scale', 'reflection', 'nan', 'shape'])
def test_invalid_frames_rejected(bad):
    frame = np.eye(4)
    if bad == 'scale': frame[0, 0] = 2
    elif bad == 'reflection': frame[0, 0] = -1
    elif bad == 'nan': frame[0, 3] = float('nan')
    else: frame = frame[:3]
    with pytest.raises(ValueError): local_anchors(frame, np.eye(4), RADIUS_M)


def stage_and_coupon():
    from pxr import Usd
    c = coupon(); stage = Usd.Stage.CreateInMemory()
    data = author(stage, c, model='section_springs')
    return stage, c, data


def test_section_author_has_four_massless_free_D6_and_no_parallel_angular_path():
    from pxr import Usd, UsdPhysics
    stage, c, data = stage_and_coupon()
    baseline = Usd.Stage.CreateInMemory(); author(baseline, c)
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(bodies) == 3
    for prim in stage.Traverse():
        if prim.GetTypeName() in ('PhysicsJoint', 'PhysicsRevoluteJoint'): continue
        original = baseline.GetPrimAtPath(prim.GetPath())
        assert original and prim.GetAppliedSchemas() == original.GetAppliedSchemas()
        for attr in prim.GetAttributes():
            assert attr.Get() == original.GetAttribute(attr.GetName()).Get()
    assert sum(p.GetTypeName() == 'PhysicsJoint' for p in stage.Traverse()) == 4
    for i, path in enumerate(data['joint_paths']):
        d = UsdPhysics.DriveAPI(stage.GetPrimAtPath(path), 'angular')
        assert d.GetStiffnessAttr().Get() == d.GetDampingAttr().Get() == 0
        desc = inspect_pair(stage, path, stiffness=c.stiffness[i], damping=c.damping[i], radius=RADIUS_M)
        assert desc['all_axes_unlimited'] and desc['authored_usd_verified']
        assert not desc['native_external_drive_coefficients_verified']
        for spring in desc['joint_paths']:
            prim = stage.GetPrimAtPath(spring)
            assert prim.GetAppliedSchemas() == ['PhysicsDriveAPI:transZ']
            joint = UsdPhysics.Joint(prim)
            assert joint.GetExcludeFromArticulationAttr().Get()
            assert joint.GetJointEnabledAttr().Get() and joint.GetCollisionEnabledAttr().Get()
            assert UsdPhysics.DriveAPI(prim, 'transZ').GetMaxForceAttr().Get() == MAX_FORCE
    assert not any(p.IsA(UsdPhysics.FixedJoint) for p in stage.Traverse())
    assert len(data['section_springs']) == 2


def test_section_bind_requires_stage_and_native_exact_zero_without_any_writes():
    stage, c, data = stage_and_coupon(); a = FakeArticulation(c)
    with pytest.raises(ValueError, match='requires its authored stage'): bind(a, c, model='section_springs')
    with pytest.raises(ValueError, match='exactly zero'): bind(a, c, model='section_springs', stage=stage)
    a.k *= 0; a.c *= 0
    report, predictor = bind(a, c, model='section_springs', stage=stage)
    assert predictor is None and a.writes == []
    assert report['native_angular_drives_zero_verified'] and report['native_angular_drives_disabled']
    assert not report['native_drives_disabled'] and report['external_section_drives_authored']
    assert not report['native_si_coefficients_verified'] and not report['external_d6_native_coefficients_verified']
    assert len(report['section_springs']) == 2
    a.c[0, 1] = 1e-30
    with pytest.raises(ValueError, match='exactly zero'): bind(a, c, model='section_springs', stage=stage)
    assert a.writes == []


@pytest.mark.parametrize('bad', ['limit', 'drive', 'body', 'external', 'collision', 'position', 'gain', 'angular'])
def test_author_binding_detects_extra_constraint_or_changed_section(bad):
    from pxr import Gf, UsdPhysics
    stage, c, data = stage_and_coupon()
    central = data['joint_paths'][0]; prim = stage.GetPrimAtPath(central+'_Section_minus')
    joint = UsdPhysics.Joint(prim)
    if bad == 'limit': UsdPhysics.LimitAPI.Apply(prim, 'transX')
    elif bad == 'drive': UsdPhysics.DriveAPI.Apply(prim, 'rotY')
    elif bad == 'body': joint.CreateBody1Rel().SetTargets([data['body_paths'][2]])
    elif bad == 'external': joint.CreateExcludeFromArticulationAttr(False)
    elif bad == 'collision': joint.CreateCollisionEnabledAttr(False)
    elif bad == 'position': joint.CreateLocalPos0Attr(Gf.Vec3f(0, 0, 0))
    elif bad == 'gain': UsdPhysics.DriveAPI(prim, 'transZ').CreateStiffnessAttr(1)
    else: UsdPhysics.DriveAPI(stage.GetPrimAtPath(central), 'angular').CreateDampingAttr(.01)
    with pytest.raises(ValueError): inspect_pair(stage, central, stiffness=c.stiffness[0], damping=c.damping[0], radius=RADIUS_M)


def test_no_overwrite_and_bad_model_does_not_author():
    from pxr import Usd
    stage, c, data = stage_and_coupon(); text = stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='no overwrite'):
        author_pair(stage, data['joint_paths'][0], stiffness=c.stiffness[0], damping=c.damping[0], radius=RADIUS_M)
    assert stage.GetRootLayer().ExportToString() == text
    empty = Usd.Stage.CreateInMemory()
    with pytest.raises(ValueError): author(empty, c, model='wrong')
    assert not list(empty.Traverse())


def test_section_sample_energy_gap_velocity_and_exact_law_tail():
    c = coupon(); inputs = equilibrium(c, q=(.035, .04))
    row = sample(c, **inputs, model='section_springs')
    q = np.array(row['q_rad']); k = np.array(c.stiffness)
    # Synthetic test moments set to the exact section oracle, NOT native data.
    row['contact_joint_moments_nm'] = (k*np.sin(q)*np.cos(q)).tolist()
    assessment = assess_tail(tail(c, row), c)
    assert assessment['passed'] and assessment['elastic_law'] == 'K*sin(theta)*cos(theta)'
    assert assessment['max_static_moment_residual_nm'] < 1e-15
    # Runtime geometry uses float32-authored anchors/gains, not ideal doubles.
    assert row['section_actual_fiber_energy_j'] == pytest.approx(.5*np.dot(k, np.sin(q)**2), rel=2e-6)
    for i, section in enumerate(row['section_springs']):
        assert section['axial_stiffness_n_m'] == pytest.approx(k[i]/RADIUS_M**2)
        assert section['centre_anchor_error_m'] < 1e-8
        np.testing.assert_allclose(section['native_velocity_derived_gap_rate_m_s'], 0, atol=1e-15)
    inconsistent = tail(c, row); inconsistent[2]['comparison_model'] = 'native'
    with pytest.raises(ValueError, match='differently bound'): assess_tail(inconsistent, c)
    with pytest.raises(ValueError): assess_tail(tail(c, row), c, model='native')
    damaged = deepcopy(inputs); damaged['frames'][1, 2, 3] += 2e-5
    x = sample(c, **damaged, model='section_springs')
    assert abs(x['section_springs'][0]['common_mode_z_m']) > 1e-5
    assert not assess_tail(tail(c, x), c)['gates']['joint_anchors']


class FakeRigid:
    """Only rigid-body sensors; no articulation API and no state setters."""
    count = 3
    def __init__(self, c):
        from pxr import Gf
        self.prim_paths = layout(c)['body_paths']
        poses = []
        for f in layout(c)['frames']:
            q = Gf.Matrix4d(f.T.tolist()).ExtractRotationQuat()
            poses.append([*f[:3, 3], *q.GetImaginary(), q.GetReal()])
        self.poses = np.array(poses, dtype=np.float32)
        self.mass = np.array(c.masses, dtype=np.float32)[:, None]
        self.com = np.tile([0., 0., 0., 0., 0., 0., 1.], (3, 1))
        self.inertia = np.array(c.inertias).transpose(0, 2, 1).reshape(3, 9).astype(np.float32)
        self.velocity = np.zeros((3, 6))
    def get_transforms(self): return self.poses.copy()
    def get_masses(self): return self.mass.copy()
    def get_coms(self): return self.com.copy()
    def get_inertias(self): return self.inertia.copy()
    def get_velocities(self): return self.velocity.copy()


def maximal_stage(c=None):
    from pxr import Usd
    c = coupon() if c is None else c
    stage = Usd.Stage.CreateInMemory()
    data = author(stage, c, model='section_springs_maximal')
    return stage, c, data


@pytest.mark.parametrize('held', [True, False])
def test_maximal_only_changes_topology_preserving_bodies_pads_material_rest(held):
    from pxr import Usd, UsdPhysics
    stage, c, data = maximal_stage(coupon(held_contacts=held))
    baseline = Usd.Stage.CreateInMemory(); author(baseline, c, model='section_springs')
    assert data['native_view_kind'] == 'rigid_body'
    assert not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in stage.Traverse())
    assert len([p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]) == 3
    for prim in stage.Traverse():
        metadata = prim.GetMetadata('apiSchemas')
        schemas = metadata.GetAppliedItems() if metadata else []
        assert not any('Articulation' in s or 'JointState' in s for s in schemas)
        original = baseline.GetPrimAtPath(prim.GetPath())
        for attr in prim.GetAttributes():
            if attr.GetName() == 'physics:excludeFromArticulation': continue
            assert attr.Get() == original.GetAttribute(attr.GetName()).Get()
    for i, path in enumerate(data['joint_paths']):
        central = UsdPhysics.RevoluteJoint(stage.GetPrimAtPath(path))
        assert central.GetExcludeFromArticulationAttr().Get()
        assert central.GetLowerLimitAttr().Get() == -float('inf')
        assert central.GetUpperLimitAttr().Get() == float('inf')
        assert UsdPhysics.DriveAPI(central.GetPrim(), 'angular').GetStiffnessAttr().Get() == 0
        args = dict(stiffness=c.stiffness[i], damping=c.damping[i], radius=RADIUS_M)
        assert inspect_pair(stage, path, topology='maximal', **args)['central_topology'] == 'maximal'
        with pytest.raises(ValueError, match='topology'): inspect_pair(stage, path, **args)
    assert len(settings_readback(stage, c, actual_dt=c.dt,
        model='section_springs_maximal')['authored_iteration_readback']) == 3
    with pytest.raises(ValueError): settings_readback(stage, c, actual_dt=c.dt)
    with pytest.raises(ValueError, match='bind_rigid'): bind(FakeArticulation(c), c, model='section_springs_maximal')


def test_topology_option_is_explicit_not_a_silent_weakening():
    stage, c, data = stage_and_coupon()
    args = dict(stiffness=c.stiffness[0], damping=c.damping[0], radius=RADIUS_M)
    for topology in ('maximal', 'auto', None):
        with pytest.raises(ValueError): inspect_pair(stage, data['joint_paths'][0], topology=topology, **args)


def test_maximal_bind_exact_native_rigid_data_without_articulation_or_setters():
    # Non-diagonal source inertia and rotated world reveal frame assumptions.
    r = _ry(.7); tensor = r@np.diag([2e-8, 3e-8, 1.5e-8])@r.T
    c = replace(coupon(rotation=_ry(.4)), inertias=(tensor, tensor, tensor))
    stage, c, _ = maximal_stage(c); view = FakeRigid(c)
    before = stage.GetRootLayer().ExportToString()
    report, unused = bind_rigid(view, c, stage=stage)
    assert unused is None and report['initial_native_frames_verified']
    assert report['native_body_paths'] == layout(c)['body_paths']
    assert report['native_masses_inertias_com_verified']
    assert not report['native_articulation_readbacks_used']
    assert not report['native_actor_topology_independently_verified']
    assert report['central_angular_drives_authored_zero']
    assert not report['native_angular_drives_zero_verified']
    assert not report['external_d6_native_coefficients_verified']
    assert not report['native_drives_disabled']
    assert stage.GetRootLayer().ExportToString() == before


@pytest.mark.parametrize('bad', ['order', 'duplicate', 'count', 'mass', 'inertia', 'com',
    'pose_nan', 'com_nan', 'bad_quat', 'translation', 'rotation', 'velocity_nan'])
def test_maximal_native_binding_fails_closed(bad):
    stage, c, _ = maximal_stage(); view = FakeRigid(c)
    if bad == 'order': view.prim_paths.reverse()
    elif bad == 'duplicate': view.prim_paths[1] = view.prim_paths[0]
    elif bad == 'count': view.count = 2
    elif bad == 'mass': view.mass[1] *= 1.1
    elif bad == 'inertia': view.inertia[0] *= 2
    elif bad == 'com': view.com[1, 0] = 1e-4
    elif bad == 'pose_nan': view.poses[0, 0] = np.nan
    elif bad == 'com_nan': view.com[0, 3] = np.nan
    elif bad == 'bad_quat': view.poses[0, 3:] *= 1.01
    elif bad == 'translation': view.poses[1, 0] += 2e-6
    elif bad == 'rotation': view.poses[:, 3:] = [0, 0, 0, 1]
    else: view.velocity[2, 4] = np.nan
    with pytest.raises(ValueError): bind_rigid(view, c, stage=stage)


@pytest.mark.parametrize('bad', ['root', 'weld', 'limit', 'kinematic', 'centre', 'angular'])
def test_maximal_topology_binding_rejects_hidden_support_or_changed_geometry(bad):
    from pxr import Gf, UsdPhysics
    stage, c, data = maximal_stage()
    prim = stage.GetPrimAtPath(data['joint_paths'][0])
    if bad == 'root': UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(data['body_paths'][0]))
    elif bad == 'weld': UsdPhysics.FixedJoint.Define(stage, c.root+'/ExtraWeld')
    elif bad == 'limit': UsdPhysics.RevoluteJoint(prim).CreateLowerLimitAttr(-1)
    elif bad == 'kinematic': UsdPhysics.RigidBodyAPI(stage.GetPrimAtPath(data['body_paths'][1])).CreateKinematicEnabledAttr(True)
    elif bad == 'centre': UsdPhysics.RevoluteJoint(prim).CreateLocalPos0Attr(Gf.Vec3f(0, 0, .016))
    else: UsdPhysics.DriveAPI(prim, 'angular').CreateDampingAttr(.001)
    with pytest.raises(ValueError): bind_rigid(FakeRigid(c), c, stage=stage)


def test_public_coordinates_are_projected_native_velocity_not_pose_FD():
    c = coupon(rotation=_ry(.5)); args = equilibrium(c, q=(.03, -.02))
    w = np.array(c.rotation)@np.array([0., 1., 0.])
    args['velocities'][1, 3:] = .2*w; args['velocities'][2, 3:] = -.1*w
    q, v, error = rigid_coordinates(args['frames'], args['velocities'])
    np.testing.assert_allclose(q, args['q'], atol=1e-15)
    np.testing.assert_allclose(v, [.2, -.3], atol=1e-15)
    np.testing.assert_allclose(error, 0, atol=1e-15)
    args['qdot'] = v
    row = sample(c, **args, model='section_springs_maximal')
    assert not row['independent_articulation_readback']
    assert row['joint_velocity_basis'] == 'projected_relative_native_body_angular_velocity'
    args['qdot'] = np.zeros(2)
    with pytest.raises(ValueError, match='projected native'): sample(c, **args, model='section_springs_maximal')
    args['frames'][0, 0, 0] = 2
    with pytest.raises(ValueError): rigid_coordinates(args['frames'], args['velocities'])


def test_modeled_kinetic_energy_uses_source_world_inertia_COM_velocity():
    c = coupon(); frames = layout(c)['frames']; vel = np.zeros((3, 6))
    vel[0, :3] = [.1, -.2, .3]; vel[0, 3:] = [2., 3., 5.]
    row = modeled_energy(c, frames, vel, model='section_springs_maximal')
    r = frames[0, :3, :3]; w = vel[0, 3:]
    expected = .5*c.masses[0]*.14 + .5*w@(r@c.inertias[0]@r.T)@w
    assert row['modeled_kinetic_energy_j'] == pytest.approx(expected)
    assert row['modeled_total_mechanical_energy_j'] == pytest.approx(expected+row['modeled_potential_energy_j'])
    t = np.eye(4); t[:3, :3] = _ry(.9); t[:3, 3] = [1, 2, 3]
    moved_vel = vel.copy(); moved_vel[:, :3] = vel[:, :3]@t[:3, :3].T; moved_vel[:, 3:] = vel[:, 3:]@t[:3, :3].T
    moved = modeled_energy(c, t@frames, moved_vel, model='section_springs_maximal')
    assert moved['modeled_kinetic_energy_j'] == pytest.approx(expected)
    assert moved['modeled_potential_energy_j'] == pytest.approx(row['modeled_potential_energy_j'], rel=1e-10)
    args = equilibrium(c); x = sample(c, **args, model='section_springs_maximal')
    assert x['modeled_potential_energy_j'] == x['section_actual_fiber_energy_j']
    assert x['modeled_initial_energy_reference_j'] > 7e-6
    assert not x['native_energy_readback']


def test_free_energy_requires_whole_run_and_catches_transient_even_if_tail_settles():
    c = coupon(held_contacts=False); model = 'section_springs_maximal'
    row = sample(c, **equilibrium(c, q=(0, 0)), model=model)
    settled = tail(c, row)
    assessment = assess_tail(settled, c, whole_run_samples=settled)
    assert assessment['passed']
    assert 'native_pose_agrees' not in assessment['gates']
    assert assessment['gates']['hinge_axis_alignment']
    assert not assess_tail(settled, c)['gates']['whole_run_free_energy']
    growth = sample(c, **equilibrium(c, q=(.04, .04)), model=model)
    whole = tail(c, growth, count=1)+tail(c, row)
    for i, x in enumerate(whole): x['step_id'] = i+1
    # Fields cannot hide inconsistent energy: recomputation uses frames/velocities.
    whole[0]['modeled_total_mechanical_energy_j'] = 0
    assert not assess_free_energy(whole, c, model=model)['gates']['bounded_from_initial']
    assert not assess_tail(whole[-121:], c, whole_run_samples=whole)['passed']
    with pytest.raises(ValueError): assess_free_energy(whole[1:], c, model=model)


def test_free_energy_does_not_accumulate_roundoff_or_ignore_native_kinetic_growth():
    c = coupon(held_contacts=False); model = 'section_springs_maximal'
    rows = tail(c, sample(c, **equilibrium(c, q=(0, 0)), model=model), count=10)
    for i, row in enumerate(rows):
        # Each increase only .5 nJ, but the cumulative rise exceeds allowance.
        row['body_velocities_world'][0][0] = math.sqrt(2*i*.5e-9/c.masses[0])
    result = assess_free_energy(rows, c, model=model)
    assert result['gates']['bounded_from_initial']
    assert not result['gates']['nonincreasing_with_roundoff']
    assert not result['passed']
    rows[1]['body_velocities_world'][0][0] = float('nan')
    with pytest.raises(ValueError): assess_free_energy(rows, c, model=model)


def test_angular_maximal_generic_D6_preserves_physics_and_has_no_fibers():
    from pxr import Usd, UsdPhysics
    c = coupon(iterations=(128,32)); stage = Usd.Stage.CreateInMemory()
    data = author(stage, c, model='angular_d6_maximal')
    baseline, _, _ = maximal_stage(c)
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint): continue
        original = baseline.GetPrimAtPath(prim.GetPath())
        assert prim.GetMetadata('apiSchemas') == original.GetMetadata('apiSchemas')
        for attr in prim.GetAttributes(): assert attr.Get() == original.GetAttribute(attr.GetName()).Get()
    assert 'section_springs' not in data and data['native_view_kind'] == 'rigid_body'
    assert [str(p.GetPath()) for p in stage.Traverse() if p.IsA(UsdPhysics.Joint)] == data['joint_paths']
    assert all(stage.GetPrimAtPath(p).GetTypeName() == 'PhysicsJoint' for p in data['joint_paths'])
    report, unused = bind_rigid(FakeRigid(c), c, stage=stage, model='angular_d6_maximal')
    assert unused is None and report['angular_d6_authored_verified']
    assert not report['central_angular_drives_authored_zero'] and report['section_springs'] == []
    assert not report['native_articulation_readbacks_used'] and not report['external_d6_native_coefficients_verified']
    for i, drive in enumerate(report['angular_d6_usd_drives']):
        assert drive['stiffness_usd'] == pytest.approx(c.stiffness[i]*math.pi/180, rel=1e-7)
        assert drive['damping_usd'] == pytest.approx(c.damping[i]*math.pi/180, rel=1e-7)
    read = settings_readback(stage, c, actual_dt=c.dt, model='angular_d6_maximal')
    assert len(read['authored_iteration_readback']) == 3
    with pytest.raises(ValueError): bind_rigid(FakeRigid(c), c, stage=stage)
    free = replace(c, held_contacts=False)
    row = sample(free, **equilibrium(free, q=(0,0)), model='angular_d6_maximal')
    assert not row['independent_articulation_readback'] and 'section_springs' not in row
    assert not assess_tail(tail(free,row),free)['passed']
    assert assess_tail(tail(free,row),free,whole_run_samples=tail(free,row))['passed']


@pytest.mark.parametrize('bad', ['gain57', 'damping', 'lock', 'extra_drive', 'nan_frame',
    'body', 'external', 'target', 'extra_joint', 'root'])
def test_angular_D6_binding_rejects_changed_contract(bad):
    from pxr import Gf, Usd, UsdPhysics
    c = coupon(); stage = Usd.Stage.CreateInMemory(); data = author(stage,c,model='angular_d6_maximal')
    prim = stage.GetPrimAtPath(data['joint_paths'][0]); drive = UsdPhysics.DriveAPI(prim,'rotY')
    if bad == 'gain57': drive.CreateStiffnessAttr(c.stiffness[0])
    elif bad == 'damping': drive.CreateDampingAttr(0.)
    elif bad == 'lock': UsdPhysics.LimitAPI(prim,'rotX').CreateLowAttr(-1.)
    elif bad == 'extra_drive': UsdPhysics.DriveAPI.Apply(prim,'rotZ')
    elif bad == 'nan_frame': UsdPhysics.Joint(prim).CreateLocalRot0Attr(Gf.Quatf(float('nan')))
    elif bad == 'body': UsdPhysics.Joint(prim).CreateBody1Rel().SetTargets([data['body_paths'][2]])
    elif bad == 'external': UsdPhysics.Joint(prim).CreateExcludeFromArticulationAttr(False)
    elif bad == 'target': drive.CreateTargetPositionAttr(.005)
    elif bad == 'extra_joint': UsdPhysics.Joint.Define(stage,c.root+'/Extra')
    else: UsdPhysics.ArticulationRootAPI.Apply(stage.GetPrimAtPath(data['body_paths'][0]))
    with pytest.raises(ValueError): bind_rigid(FakeRigid(c),c,stage=stage,model='angular_d6_maximal')
