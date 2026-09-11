"""CPU algebra, fake native views and in-memory USD only; never launch Kit."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from sim_physics.contact_spring_probe import (
    Coupon, INITIAL_JOINT_ANGLE_RAD, INITIAL_Q_TOLERANCE_RAD, MAX_CONTACT_ROWS, SPACING_M,
    _ry, assess_tail, author, bind, from_report, layout, sample, settings_readback,
)
from sim_physics.implicit_springs import elastic_effort


def coupon(**kwargs):
    return Coupon('synthetic-test-report.json', 'a'*64, (.3328997492790222, .3048170804977417),
        (.00940549187362194, .00807236973196268), (.000633, .000607, .000580),
        tuple(np.diag([3e-8, 3e-8, 2.5e-9]) for _ in range(3)),
        1000., 1.0573968428135787, .5, **kwargs)


def equilibrium(c, q=(.004, .003), step=1):
    """Synthetic force-neutral point couples, not fabricated native evidence."""
    q = np.array(q); data = layout(c); r = np.array(c.rotation)
    frames = np.tile(np.eye(4), (3, 1, 1)); frames[0, :3, :3] = r
    for i in range(1, 3):
        frames[i, :3, :3] = frames[i-1, :3, :3] @ _ry(q[i-1])
        frames[i, :3, 3] = (frames[i-1, :3, 3]
            + frames[i-1, :3, :3] @ [0, 0, SPACING_M/2]
            + frames[i, :3, :3] @ [0, 0, SPACING_M/2])
    spring = q*np.array(c.stiffness)
    body_torques = [-spring[0], spring[0]-spring[1], spring[1]]
    rows = []; lever = .008
    if c.held_contacts:
        for i, torque in enumerate(body_torques):
            for side in (-1, 1):
                f = r @ [side*abs(torque)/(2*lever), 0, 0]
                z = side*np.sign(torque)*lever
                rows.append(dict(collider0=data['collider_paths'][i],
                    collider1=c.root+'/Pad_'+str(i)+('_minus' if side == 1 else '_plus'),
                    kind='normal', point_world_m=(frames[i, :3, 3]+r @ [0, 0, z]).tolist(),
                    impulse_on_0_ns=(f*c.dt).tolist(), normal_on_0=(r @ [side, 0, 0]).tolist()))
    return dict(step_id=step, frames=frames, velocities=np.zeros((3, 6)),
        q=q, qdot=np.zeros(2), contact_rows=rows, full_normal_friction_stream=True)


def tail(c, row, count=121):
    result = []
    for i in range(count):
        x = deepcopy(row); x['step_id'] = i+1; result.append(x)
    return result


def test_import_isolated_without_native_runtime_or_app():
    env = os.environ.copy()
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    code = ('import sys; import sim_physics.contact_spring_probe; '
            'assert not any(n.startswith(("omni", "isaacsim", "pxr")) for n in sys.modules)')
    result = subprocess.run([sys.executable, '-B', '-c', code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_report_coefficients_and_hash_are_exact_not_fitted(tmp_path):
    report = dict(native_drive_parameters=dict(names=['Joint_002:0','Joint_002:1','Joint_003:1'],
        stiffness=[[.9,.3,.2]], damping=[[.8,.009,.008]], native_masses=[[.1],[.001],[.002],[.003]],
        native_inertias=[np.eye(3).reshape(-1).tolist()]*4),
        robot_probe=dict(finger_contact_compliance=dict(force_based=True,
            stiffness_n_m=1000., damping_n_s_m=1.)), configuration=dict(finger_friction=.5))
    path = tmp_path/'source.json'; path.write_text(json.dumps(report), encoding='utf-8')
    c = from_report(path)
    assert c.stiffness == (.3,.2) and c.damping == (.009,.008)
    assert c.masses == (.001,.002,.003)
    assert c.source_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    control = from_report(path, contact_model='rigid_control')
    assert control == replace(c, contact_model='rigid_control')
    with pytest.raises(ValueError, match='Only explicitly'): from_report(path, stiffness=(1,1))
    report['robot_probe']['finger_contact_compliance']['force_based'] = False
    path.write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='Force-based'): from_report(path)


@pytest.mark.parametrize('change', [dict(stiffness=(0,1)), dict(damping=(-1,1)),
    dict(masses=(1,2)), dict(masses=(1,float('nan'),1)), dict(dt=True), dict(dt=0),
    dict(dt=1/2000), dict(iterations=(32,4)), dict(iterations=(True,4)), dict(iterations=(128,4)),
    dict(held_contacts=1), dict(solver='bad'), dict(rotation=np.diag([-1,1,1])),
    dict(rotation=np.diag([2,1,1])), dict(source_sha256='bad'), dict(root='/World/../bad'),
    dict(inertias=tuple(np.diag([1,1,3]) for _ in range(3)))])
def test_invalid_preparation_fails_before_usd(change):
    with pytest.raises((ValueError, TypeError)): replace(coupon(), **change)


@pytest.mark.parametrize('held',[False,True])
def test_geometry_and_initial_strain_match_both_boundary_conditions(held):
    c = coupon(held_contacts=held); data = layout(c)
    assert len(data['body_paths']) == 3 and len(data['joint_paths']) == 2
    assert len(data['pads']) == (6 if held else 0)
    frames = data['frames']
    for i in range(2):
        np.testing.assert_allclose(frames[i,:3,3]+frames[i,:3,:3]@[0,0,SPACING_M/2],
            frames[i+1,:3,3]+frames[i+1,:3,:3]@[0,0,-SPACING_M/2],atol=1e-14)
        np.testing.assert_allclose(frames[i,:3,:3].T@frames[i+1,:3,:3], _ry(INITIAL_JOINT_ANGLE_RAD), atol=1e-14)
    np.testing.assert_array_equal(frames, layout(replace(c,held_contacts=not held))['frames'])


@pytest.mark.parametrize('iterations',[(16,4),(32,0),(128,32),(128,0)])
@pytest.mark.parametrize('held',[False,True])
def test_usd_has_no_weld_and_uniform_authored_iterations(iterations,held):
    from pxr import Usd, UsdGeom, UsdPhysics
    c = coupon(iterations=iterations,held_contacts=held); stage = Usd.Stage.CreateInMemory()
    data = author(stage,c)
    assert not any(p.IsA(UsdPhysics.FixedJoint) for p in stage.Traverse())
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    assert len(bodies) == 3
    roots = [str(p.GetPath()) for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    assert roots == [data['body_paths'][0]]
    for i,p in enumerate(bodies):
        assert not UsdPhysics.RigidBodyAPI(p).GetKinematicEnabledAttr().Get()
        assert UsdPhysics.MassAPI(p).GetMassAttr().Get() == pytest.approx(c.masses[i])
        assert p.GetAttribute('physxContactReport:threshold').Get() == 0
    for i,path in enumerate(data['joint_paths']):
        p=stage.GetPrimAtPath(path); j=UsdPhysics.RevoluteJoint(p)
        assert j.GetAxisAttr().Get() == 'Y' and j.GetCollisionEnabledAttr().Get()
        assert not j.GetExcludeFromArticulationAttr().Get()
        # Portable USD lacks the PhysxSchema plugin. Inspect the authored API
        # list, not GetAppliedSchemas(), which filters out unregistered types.
        authored=p.GetMetadata('apiSchemas').GetAppliedItems()
        assert 'PhysicsJointStateAPI:angular' in authored
        assert 'PhysxJointStateAPI:angular' not in authored
        drive=UsdPhysics.DriveAPI(p,'angular')
        assert drive.GetTypeAttr().Get() == 'force'
        assert drive.GetStiffnessAttr().Get()*180/math.pi == pytest.approx(c.stiffness[i])
        assert drive.GetDampingAttr().Get()*180/math.pi == pytest.approx(c.damping[i])
        assert p.GetAttribute('state:angular:physics:position').Get()*math.pi/180 == pytest.approx(INITIAL_JOINT_ANGLE_RAD)
    read=settings_readback(stage,c,actual_dt=c.dt)
    assert len(read['authored_iteration_readback']) == 4
    assert read['native_iteration_readback'] is None
    assert read['effective_native_iterations_verified'] is False
    text=stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='Empty standalone'): author(stage,c)
    assert stage.GetRootLayer().ExportToString() == text
    with pytest.raises(ValueError, match='Context changed'): settings_readback(stage,c,actual_dt=c.dt*2)
    stage.GetPrimAtPath(data['body_paths'][1]).GetAttribute('physxRigidBody:solverVelocityIterationCount').Set(3)
    with pytest.raises(ValueError, match='iteration'): settings_readback(stage,c,actual_dt=c.dt)


class FakeArticulation:
    count=1
    def __init__(self,c):
        self.shared_metatype=SimpleNamespace(fixed_base=False,dof_names=['Joint_001','Joint_002'])
        self.link_paths=[layout(c)['body_paths']]
        self.k=np.array([c.stiffness]);self.c=np.array([c.damping]);self.mass=np.array([c.masses])
        self.q=np.full((1,2),INITIAL_JOINT_ANGLE_RAD)
        self.writes=[]
    def get_dof_stiffnesses(self):return self.k.copy()
    def get_dof_dampings(self):return self.c.copy()
    def get_masses(self):return self.mass.copy()
    def get_dof_positions(self):return self.q.copy()
    def set_dof_stiffnesses(self,x,i):self.writes.append('stiffness');self.k=x.copy()
    def set_dof_dampings(self,x,i):self.writes.append('damping');self.c=x.copy()


def test_native_does_not_disable_drives_and_old_predictor_is_explicit_control():
    c=coupon();a=FakeArticulation(c)
    report,predictor=bind(a,c)
    assert predictor is None and a.writes == []
    assert report['native_si_coefficients_verified'] and not report['native_drives_disabled']
    assert report['initial_native_q_verified']
    assert report['initial_native_q_rad'] == [INITIAL_JOINT_ANGLE_RAD]*2
    report,predictor=bind(a,c,model='implicit_effort')
    assert predictor is not None and a.writes == ['stiffness','damping']
    np.testing.assert_array_equal(predictor.k,c.stiffness)
    np.testing.assert_array_equal(predictor.c,c.damping)
    assert not np.any(a.k) and not np.any(a.c)


@pytest.mark.parametrize('bad',['fixed','paths','names','stiffness','mass','damping'])
def test_binding_fails_before_disabling_any_drive(bad):
    c=coupon();a=FakeArticulation(c)
    if bad=='fixed':a.shared_metatype.fixed_base=True
    elif bad=='paths':a.link_paths[0].reverse()
    elif bad=='names':a.shared_metatype.dof_names.reverse()
    elif bad=='stiffness':a.k*=2
    elif bad=='damping':a.c*=2
    else:a.mass*=2
    with pytest.raises(ValueError):bind(a,c,model='implicit_effort')
    assert a.writes == []


@pytest.mark.parametrize('model',['native','implicit_effort','native_damping_explicit_stiffness'])
@pytest.mark.parametrize('q',[[[0.,0.]], [[.005,0.]], [[-.005,.005]],
    [[.005,.005+1.01*INITIAL_Q_TOLERANCE_RAD]], [[float('nan'),.005]],
    [[float('inf'),.005]], [.005,.005]])
def test_initial_native_strain_must_be_present_before_any_drive_write(model,q):
    c=coupon();a=FakeArticulation(c);a.q=np.array(q)
    with pytest.raises(ValueError,match='[Ii]nitial native'):bind(a,c,model=model)
    assert a.writes == []


def test_initial_native_float32_roundoff_is_not_missing_excitation():
    c=coupon();a=FakeArticulation(c);a.q=np.full((1,2),INITIAL_JOINT_ANGLE_RAD,dtype=np.float32)
    report,predictor=bind(a,c)
    assert report['initial_native_q_verified'] and predictor is None
    assert report['initial_q_tolerance_rad'] == INITIAL_Q_TOLERANCE_RAD


def test_signed_contact_moment_oracle_and_swapped_header_invariance():
    c=coupon();inputs=equilibrium(c);row=sample(c,**inputs)
    np.testing.assert_allclose(row['contact_joint_moments_nm'],row['spring_kq_nm'],rtol=1e-12)
    np.testing.assert_allclose(row['net_contact_force_n'],0,atol=1e-14)
    np.testing.assert_allclose(row['net_contact_torque_nm'],0,atol=1e-14)
    swapped=deepcopy(inputs)
    for x in swapped['contact_rows']:
        x['collider0'],x['collider1']=x['collider1'],x['collider0']
        x['impulse_on_0_ns']=(-np.array(x['impulse_on_0_ns'])).tolist()
        x['normal_on_0']=(-np.array(x['normal_on_0'])).tolist()
    other=sample(c,**swapped)
    for key in ('contact_joint_moments_nm','contact_wrenches_at_body_com','per_body_contact_upper_bound_n'):
        np.testing.assert_allclose(other[key],row[key])
    assessed=assess_tail(tail(c,row),c)
    assert assessed['passed'] and not assessed['native_qualified'] and not assessed['whole_plant_qualified']


def test_whole_coupon_rotation_does_not_change_scalar_stiffness_oracle():
    a=.7;r=np.array([[1,0,0],[0,math.cos(a),-math.sin(a)],[0,math.sin(a),math.cos(a)]])
    c=coupon(rotation=r);x=sample(c,**equilibrium(c))
    np.testing.assert_allclose(x['contact_joint_moments_nm'],x['spring_kq_nm'],atol=1e-14)
    assert assess_tail(tail(c,x),c)['passed']


def test_friction_anchor_is_counted_once_and_signed_normal_not_absolute_support():
    c=coupon();inputs=equilibrium(c);original=sample(c,**inputs)
    extra=deepcopy(inputs['contact_rows'][0]);extra['kind']='friction'
    extra.pop('normal_on_0');extra['impulse_on_0_ns']=[0,.0001*c.dt,0]
    inputs['contact_rows'].append(extra);friction=sample(c,**inputs)
    assert friction['net_contact_force_n'][1] == pytest.approx(.0001)
    assert friction['per_body_contact_upper_bound_n'][0]-original['per_body_contact_upper_bound_n'][0] == pytest.approx(.0001)
    inputs=equilibrium(c)
    for row in inputs['contact_rows']:row['impulse_on_0_ns']=(-np.array(row['impulse_on_0_ns'])).tolist()
    tension=sample(c,**inputs)
    assert tension['active_normal_pads'] == []
    np.testing.assert_allclose(tension['contact_joint_moments_nm'],-np.array(original['contact_joint_moments_nm']))


@pytest.mark.parametrize('bad',['stream','step','nan','unknown','overflow','normal','counts'])
def test_invalid_or_incomplete_evidence_fails_closed(bad):
    c=coupon();inputs=equilibrium(c)
    if bad=='stream':inputs['full_normal_friction_stream']=False
    elif bad=='step':inputs['step_id']=True
    elif bad=='nan':inputs['contact_rows'][0]['impulse_on_0_ns'][0]=float('nan')
    elif bad=='unknown':inputs['contact_rows'][0]['collider1']='/Other/Uncaptured'
    elif bad=='overflow':inputs['contact_rows']=[inputs['contact_rows'][0]]*(MAX_CONTACT_ROWS+1)
    elif bad=='normal':inputs['contact_rows'][0]['normal_on_0']=[0,0,0]
    else:inputs['velocities']=np.zeros((2,6))
    with pytest.raises(ValueError):sample(c,**inputs)


@pytest.mark.parametrize('bad',['velocity','body_velocity','pose','wrench','trivial','moment','pad','duration'])
def test_correct_looking_angle_alone_cannot_pass(bad):
    c=coupon();x=sample(c,**equilibrium(c))
    if bad=='velocity':x['qdot_rad_s']=[.263,0]
    elif bad=='body_velocity':x['body_velocities_world'][0][4]=.263
    elif bad=='pose':x['measured_relative_angle_rad'][0]+=.01
    elif bad=='wrench':x['net_contact_torque_nm'][0]=1e-4
    elif bad=='trivial':x['q_rad']=[0,0];x['contact_joint_moments_nm']=[0,0]
    elif bad=='moment':x['contact_joint_moments_nm'][0]*=.1
    elif bad=='pad':x['active_normal_pads'].pop()
    rows=tail(c,x,count=120 if bad=='duration' else 121)
    assert not assess_tail(rows,c)['passed']


def test_stale_tail_or_coefficient_source_cannot_pass():
    c=coupon();rows=tail(c,sample(c,**equilibrium(c)))
    rows[1]['step_id']=rows[0]['step_id']
    with pytest.raises(ValueError):assess_tail(rows,c)
    rows=tail(c,sample(c,**equilibrium(c)));rows[-1]['source_sha256']='b'*64
    with pytest.raises(ValueError):assess_tail(rows,c)


def test_contact_free_control_tests_recovery_not_loaded_stiffness():
    c=coupon(held_contacts=False);x=sample(c,**equilibrium(c,q=(0,0)))
    assert not x['contact_rows'] and not x['active_normal_pads']
    assert not assess_tail(tail(c,x),c)['passed']  # Settled tail alone is insufficient.
    assert assess_tail(tail(c,x),c,whole_run_samples=tail(c,x))['passed']
    strained=sample(c,**equilibrium(c,q=(.005,.005)))
    assert not assess_tail(tail(c,strained),c)['passed']


def test_unknown_contact_predictor_loses_static_stiffness_but_full_coupling_does_not():
    # Algebraic oracle, not a native measurement or a claimed PhysX fix.
    m,k,d,h,q=1e-8,.25,.008,1/240,.1
    missing=elastic_effort([[m]],[q],[0],[k],[d],[0],h)[0]
    coupled=elastic_effort([[m]],[q],[0],[k],[d],[k*q],h)[0]
    alpha=m/(m+h*d+h*h*k)
    assert missing == pytest.approx(-alpha*k*q)
    assert coupled == pytest.approx(-k*q)
    assert alpha < .0003
    assert h/(-math.log1p(-alpha)) > 15


@pytest.mark.parametrize('invalid', [None, True, 'rigid', '', 0])
def test_contact_model_requires_explicit_known_option(invalid):
    with pytest.raises(ValueError, match='contact model'):
        coupon(contact_model=invalid)


@pytest.mark.parametrize('model', ['native', 'section_springs', 'section_springs_maximal'])
def test_rigid_contact_control_only_omits_compliant_material_authoring(model):
    from pxr import Usd
    source = coupon(); control = replace(source, contact_model='rigid_control')
    default_stage = Usd.Stage.CreateInMemory(); rigid_stage = Usd.Stage.CreateInMemory()
    author(default_stage, source, model=model)
    data = author(rigid_stage, control, model=model)
    assert data['contact_model'] == 'rigid_control'
    assert [str(p.GetPath()) for p in rigid_stage.Traverse()] == [str(p.GetPath()) for p in default_stage.Traverse()]
    omitted = {'physxMaterial:compliantContactStiffness', 'physxMaterial:compliantContactDamping',
               'physxMaterial:compliantContactAccelerationSpring'}
    for prim in default_stage.Traverse():
        other = rigid_stage.GetPrimAtPath(prim.GetPath())
        assert prim.GetMetadata('apiSchemas') == other.GetMetadata('apiSchemas')
        for attr in prim.GetAuthoredAttributes():
            target = other.GetAttribute(attr.GetName())
            if attr.GetName() in omitted:
                assert not target or not target.HasAuthoredValueOpinion()
            else:
                assert target and target.Get() == attr.Get()
        assert {str(r.GetName()): list(r.GetTargets()) for r in prim.GetRelationships()} == {
            str(r.GetName()): list(r.GetTargets()) for r in other.GetRelationships()}
    before = rigid_stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError, match='Empty standalone'): author(rigid_stage, source, model=model)
    assert rigid_stage.GetRootLayer().ExportToString() == before
    report = control.report()
    assert report['contact_model_override'] and report['source_contact_model'] == 'compliant'
    assert not report['compliant_contact_coefficients_authored']
    assert report['source_compliant_contact_coefficients'] == dict(stiffness_n_m=source.contact_stiffness,
        damping_n_s_m=source.contact_damping, force_based=True)
    for key in ('stiffness', 'damping', 'masses', 'inertias', 'friction', 'source_sha256'):
        assert report[key] == source.report()[key]
    assert not report['native_qualified'] and not report['training_eligible']
    assert report['nominal_pad_inner_gap_m'] == pytest.approx(.0055)
    assert report['nominal_shaft_diameter_m'] == pytest.approx(.0056)
    assert report['nominal_rigid_pinch_interference_m'] == pytest.approx(.0001)
    assert report['rigid_held_control_ineligible']
    assert source.report()['compliant_contact_coefficients_authored']
    assert not source.report()['contact_model_override']


def test_contact_control_preserves_gates_and_rejects_mixed_contact_provenance():
    source = coupon(); control = replace(source, contact_model='rigid_control')
    row = sample(control, **equilibrium(control))
    assert row['contact_model'] == 'rigid_control'
    result = assess_tail(tail(control, row), control)
    assert not result['passed'] and result['gates'].pop('rigid_control_pinch_feasible') is False
    assert result['gates'] == assess_tail(tail(source, sample(source, **equilibrium(source))), source)['gates']
    with pytest.raises(ValueError, match='differently bound'):
        assess_tail(tail(control, row), source)
    row['contact_model'] = 'compliant'
    with pytest.raises(ValueError, match='differently bound'):
        assess_tail(tail(control, row), control)


@pytest.mark.parametrize('model', ['native', 'section_springs', 'section_springs_maximal'])
@pytest.mark.parametrize('iterations', [(128,32), (128,0)])
def test_high_iteration_comparison_changes_only_uniform_authored_iterations(model, iterations):
    from pxr import Usd
    source = coupon(); high = replace(source, iterations=iterations)
    original = Usd.Stage.CreateInMemory(); stage = Usd.Stage.CreateInMemory()
    author(original, source, model=model); author(stage, high, model=model)
    for prim in original.Traverse():
        other = stage.GetPrimAtPath(prim.GetPath())
        for attr in prim.GetAuthoredAttributes():
            expected = (128 if attr.GetName().endswith(':solverPositionIterationCount') else
                        iterations[1] if attr.GetName().endswith(':solverVelocityIterationCount') else attr.Get())
            assert other.GetAttribute(attr.GetName()).Get() == expected
    read = settings_readback(stage, high, actual_dt=high.dt, model=model)
    assert len(read['authored_iteration_readback']) == (3 if model == 'section_springs_maximal' else 4)
    assert all(r['iterations'] == list(iterations) for r in read['authored_iteration_readback'])
    assert read['native_iteration_readback'] is None and not read['effective_native_iterations_verified']
    assert high.contact_model == 'compliant'
    assert high.stiffness == source.stiffness and high.damping == source.damping
    assert high.masses == source.masses and high.inertias == source.inertias


def test_tgs_128_0_report_matches_128_32_except_explicit_iteration_choice():
    source = coupon(solver='TGS', dt=1/1920, iterations=(128,32))
    comparison = replace(source, iterations=(128,0))
    expected = source.report(); expected['iterations'] = [128,0]
    assert comparison.report() == expected
    assert not comparison.report()['native_qualified']
    assert coupon().iterations == (16,4)
    # Both choices retain the same failed static-load/velocity gates.
    for c in (source, comparison):
        inputs = equilibrium(c); inputs['contact_rows'] = []; inputs['qdot'][:] = .1
        result = assess_tail(tail(c, sample(c, **inputs), count=961), c)
        assert not result['gates']['stiffness_equilibrium']
        assert not result['gates']['joint_velocity']
        assert not result['passed']


class SplitFake(FakeArticulation):
    def __init__(self,c):
        super().__init__(c)
        self.caps = np.full((1,2), np.finfo(np.float32).max)
        self.drive_types = np.ones((1,2), dtype=np.uint8)
        self.velocity_targets = np.zeros((1,2)); self.commands = []
    def get_dof_max_forces(self): return self.caps.copy()
    def get_drive_types(self): return self.drive_types.copy()
    def get_dof_velocity_targets(self): return self.velocity_targets.copy()
    def set_dof_actuation_forces(self, value, indices):
        assert value.shape == (1,2) and value.dtype == np.float32
        np.testing.assert_array_equal(indices, [0])
        self.commands.append(value.copy())


def test_split_authors_original_bootstrap_then_zeros_only_K_and_sends_exact_effort():
    from pxr import Usd
    c = coupon(); stage = Usd.Stage.CreateInMemory(); original = Usd.Stage.CreateInMemory()
    author(stage,c,model='native_damping_explicit_stiffness'); author(original,c,model='native')
    assert stage.GetRootLayer().ExportToString() == original.GetRootLayer().ExportToString()
    a = SplitFake(c); report, split = bind(a,c,model='native_damping_explicit_stiffness')
    assert a.writes == ['stiffness'] and np.all(a.k == 0)
    np.testing.assert_array_equal(a.c, [c.damping])
    assert not report['native_drives_disabled'] and not report['native_angular_drives_disabled']
    assert report['initial_native_si_coefficients_verified'] and not report['native_si_coefficients_verified']
    assert report['split_scheme']['native_stiffness_zero_verified']
    assert report['split_scheme']['native_drive_types'] == [1,1]
    assert report['split_scheme']['explicit_effort_caps_enforced_by_helper']
    assert not report['split_scheme']['native_drive_caps_assumed_to_limit_explicit_effort']
    assert not report['split_scheme']['native_coupling_verified']
    a.q = np.array([[.004,-.003]])
    submitted = split.step(c.dt,root_constrained=False)
    expected = (-np.array(c.stiffness)*a.q[0]).astype(np.float32)
    np.testing.assert_array_equal(submitted, expected)
    np.testing.assert_array_equal(a.commands[0][0], expected)
    assert a.writes == ['stiffness']  # No damping/cap/state setters, mass solve or contact injection.


@pytest.mark.parametrize('bad', ['K', 'C', 'cap', 'drive_type', 'target', 'paths', 'fixed', 'q_nan', 'q_bound', 'dt', 'root'])
def test_split_faults_before_submitting_changed_or_invalid_contract(bad):
    c = coupon(); a = SplitFake(c); _, split = bind(a,c,model='native_damping_explicit_stiffness')
    dt = c.dt; root = False
    if bad == 'K': a.k[0,0] = 1e-6
    elif bad == 'C': a.c[0,0] *= 2
    elif bad == 'cap': a.caps[0,0] = .1
    elif bad == 'drive_type': a.drive_types[0,0] = 2
    elif bad == 'target': a.velocity_targets[0,0] = .1
    elif bad == 'paths': a.link_paths[0].reverse()
    elif bad == 'fixed': a.shared_metatype.fixed_base = True
    elif bad == 'q_nan': a.q[0,0] = np.nan
    elif bad == 'q_bound': a.q[0,0] = .05
    elif bad == 'dt': dt *= 2
    else: root = True
    with pytest.raises(ValueError): split.step(dt,root_constrained=root)
    assert a.commands == []


def test_split_rejects_insufficient_damping_before_any_write():
    c = replace(coupon(), damping=(1e-6,1e-6)); a = SplitFake(c)
    with pytest.raises(ValueError,match='C >= h'):
        bind(a,c,model='native_damping_explicit_stiffness')
    assert a.writes == [] and a.commands == []


@pytest.mark.parametrize('kind', [0,2,255,float('nan')])
def test_split_requires_native_force_drive_before_any_write(kind):
    c = coupon(); a = SplitFake(c); a.drive_types = np.array([[kind,1]])
    with pytest.raises(ValueError): bind(a,c,model='native_damping_explicit_stiffness')
    assert a.writes == [] and a.commands == []


def test_split_coupled_constant_mass_energy_identity_and_static_force():
    # Pure mathematical oracle. Includes an unactuated free-root coordinate;
    # it does NOT assert that native contact/position iterations implement it.
    c = coupon(); h = c.dt; K = np.diag([0.,*c.stiffness]); C = np.diag([0.,*c.damping])
    assert np.linalg.eigvalsh(C-h*K/2).min() >= 0
    rng = np.random.default_rng(17)
    for _ in range(30):
        a = rng.normal(size=(3,3)); M = (a.T@a+np.eye(3))*1e-8
        q = rng.normal(size=3)*.005; v = rng.normal(size=3)
        w = np.linalg.solve(M+h*C,M@v-h*K@q); p = q+h*w
        change = .5*(w@M@w+p@K@p-v@M@v-q@K@q)
        identity = -h*w@(C-h*K/2)@w-.5*(w-v)@M@(w-v)
        assert change == pytest.approx(identity,abs=1e-18) and change <= 0
        # Static externally balanced state has full K*q, independent of M.
        balanced = np.linalg.solve(M+h*C,-h*K@q+h*(K@q))
        np.testing.assert_allclose(balanced, 0., atol=1e-15, rtol=0)


def test_scalar_stability_condition_alone_is_not_energy_passivity():
    m,k,c,h = 1.,1.,.1,1.
    assert h*h*k < 4*m+2*h*c
    q,v = 0.,1.; w = (m*v-h*k*q)/(m+h*c); p = q+h*w
    assert .5*(m*w*w+k*p*p) > .5*(m*v*v+k*q*q)
