"""CPU-only signed blade pipeline and explicitly versioned seam model tests."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace as S
import json
import math
import numpy as np
import pytest

from sim_physics.bimanual import BimanualRobot
from sim_physics.contact_events import ContactEvents,NativeNormalContact,original_order_tool_contact
from sim_physics.contact_events_test import native_stub,native_header,native_point
from sim_physics.knife import (ShearGate,ShearParameters,KnifeGeometry,LEGACY_CUT_MODEL,
    BRITTLE_CUT_MODEL,KNIFE_IMPULSE_CONTRACT,knife_normal_impulse)
from sim_physics.knife_test import sample


def fixture(monkeypatch,model=BRITTLE_CUT_MODEL):
    from sim_physics import runtime
    monkeypatch.setattr(runtime,'pose_matrices',lambda values:np.asarray(values))
    r=object.__new__(BimanualRobot);r.cut_model=model;r.cut_authorized=True
    r.knife=object.__new__(KnifeGeometry);r.knife.local=np.eye(4);r.knife.size=np.array([.002,.05,.006])
    r.knife.collider='/World/R/knife/plate'
    r.rig=S(source_target='petiole',body_paths=['/World/T/Segment_000','/World/T/Segment_001'],cut_index=1)
    r.releases=[]
    r.rig.release_from_blade=lambda e:r.releases.append(e) or e
    r.seam=lambda frames:(np.zeros(3),np.array([0.,0.,1.]))
    r.pose=np.eye(4);r.right_palm=S(get_transforms=lambda:r.pose[None])
    r.expected_right=np.eye(4);r.cut_gate=ShearGate('petiole',ShearParameters(model=model))
    r.edge_contact_error=None;r.cut_contacts=0;r.cut_event=None
    r.event_monitor=ContactEvents(robot_root='/World/R',target_root='/World/T',fingers=[],floor_root=None)
    r.event_monitor.tool_contact=r._tool_contact
    r.event_monitor.native_full_contact_reporting=True
    return r


def step(r,i,forces=(.25,),*,reverse=False,stationary=False,held=True,other=None,
         offset=(0,0,0),normal=(1,0,0),friction=0.,dt=.005):
    r.pose[0,3]=0 if stationary else -i*.0001
    r.expected_right=r.pose.copy();r.event_monitor.begin_step();r.prepare_step(None)
    other=other or r.rig.body_paths[1]+'/StemCollider'
    pair=(r.knife.collider,other);sgn=-1 if reverse else 1
    if reverse: pair=pair[::-1]
    p=r.pose[:3,3]+offset
    rows=[native_point(tuple(sgn*force*dt*np.asarray(normal)),tuple(p),
        tuple(sgn*np.asarray(normal)),separation=-.0001) for force in forces]
    header=native_header(collider0=pair[0],collider1=pair[1],num_contact_data=len(rows),
        num_friction_anchors_data=int(friction!=0))
    r.event_monitor._callback([header],rows,[native_point((0,friction*dt,0))] if friction else [])
    return r.inspect_cut(dt,None,held,.001)


@pytest.mark.parametrize('model',[LEGACY_CUT_MODEL,BRITTLE_CUT_MODEL])
@pytest.mark.parametrize('reverse',[False,True])
@pytest.mark.parametrize('forces,passes',[
    ((.25,),True),((-.25,),False),((.25,-.25),False),
    ((.3,-.15),False),((.3,-.05),True)])
def test_real_callback_pipeline_preserves_sign_and_tensile_subtraction(native_stub,monkeypatch,model,reverse,forces,passes):
    r=fixture(monkeypatch,model);last=None
    for i in range(5):last=step(r,i,forces,reverse=reverse)
    assert bool(r.releases)==passes
    assert len(r.releases)<=1
    assert last['edge_signed_resistance_n']==pytest.approx(sum(forces))
    assert last['edge_unsigned_projection_n']==pytest.approx(sum(abs(f) for f in forces))
    assert last['edge_force_n']==pytest.approx(sum(abs(f) for f in forces))
    for row,force in zip(last['raw_normal_rows'],forces,strict=True):
        assert row['collider0']==(r.rig.body_paths[1]+'/StemCollider' if reverse else r.knife.collider)
        assert row['impulse'][0]==pytest.approx((-1 if reverse else 1)*force*.005)
        assert row['impulse_on_knife'][0]==pytest.approx(force*.005)
        assert row['signed_normal_impulse_ns']==pytest.approx(force*.005)
        assert row['eligible'] is True
    json.dumps(last,allow_nan=False)
    if passes:
        event=r.releases[0]
        assert event['minimum_signed_resistance_n']==pytest.approx(sum(forces))
        assert event['peak_force_n']==pytest.approx(sum(abs(f) for f in forces))
        assert event['force_contract']==KNIFE_IMPULSE_CONTRACT


@pytest.mark.parametrize('model',[LEGACY_CUT_MODEL,BRITTLE_CUT_MODEL])
def test_stationary_edge_only_explicit_brittle_model(native_stub,monkeypatch,model):
    r=fixture(monkeypatch,model)
    for i in range(5):
        result=step(r,i,stationary=True)
        if i<4:assert not r.releases
    assert bool(r.releases)==(model==BRITTLE_CUT_MODEL)
    assert result['gate_travel_m']==0
    if r.releases:
        e=r.releases[0]
        assert e['loading_travel_required'] is False
        assert e['minimum_loading_travel_m'] is None and e['loading_travel_requirement_met'] is None
        assert not e['measured_travel_is_tissue_work'] and not e['fracture_energy_used_as_evidence']
        assert not e['tissue_fracture_calibrated'] and e['engineering_approximation']


@pytest.mark.parametrize('change',[
    dict(forces=()),dict(forces=(0.,),friction=.3),dict(forces=(.1,),friction=.2),
    dict(held=False),dict(offset=(.01,0,0)),dict(offset=(0,0,.0031)),
    dict(other='/World/T/Other/StemCollider'),dict(other='/World/T/Segment_001/LeafCollider'),
    dict(normal=(-1,0,0)),dict(normal=(0,1,0))])
def test_brittle_negative_controls_never_timer_release(native_stub,monkeypatch,change):
    r=fixture(monkeypatch)
    for i in range(8):step(r,i,stationary=True,**change)
    assert not r.releases and r.cut_gate.dwell==0


@pytest.mark.parametrize('forces,friction',[((.4,-.15),0),((.25,),.251)])
def test_low_signed_net_never_hides_unsigned_or_friction_overcap(native_stub,monkeypatch,forces,friction):
    r=fixture(monkeypatch)
    with pytest.raises(RuntimeError,match='Unsigned'):
        step(r,0,forces,friction=friction)
    assert not r.releases and r.cut_gate.dwell==0


def test_interruption_restarts_entire_dwell_and_travel_window(native_stub,monkeypatch):
    r=fixture(monkeypatch)
    for i in range(4):step(r,i,stationary=True)
    step(r,4,forces=(-.01,),stationary=True)
    assert r.cut_gate.dwell==0 and r.cut_gate.loading_origin is None
    for i in range(4):step(r,i,stationary=True)
    assert not r.releases
    step(r,4,stationary=True)
    assert len(r.releases)==1


def test_raw_contract_keeps_left_observer_original_order_and_copies(monkeypatch):
    m=ContactEvents(robot_root='/World/R',target_root='/World/T',fingers=[],floor_root=None)
    observed=[];raw=[]
    m.normal_contact_observer=S(add_contact=lambda *args:observed.append(args),begin_step=lambda:None)
    @original_order_tool_contact
    def accept(row):raw.append(row);return True
    m.tool_contact=accept
    p=[0.,0,0];j=[-.001,0,0];n=[-1.,0,0]
    m.consume('/World/T/Stem','/World/R/knife', [j],[p],[n],[-.0001])
    assert observed[0]==('/World/T/Stem','/World/R/knife',p,n,tuple(j),-.0001)
    p[0]=99;j[0]=99;n[0]=99
    assert raw[0].point==(0.,0.,0.) and raw[0].impulse==(-.001,0.,0.)
    with pytest.raises(FrozenInstanceError):raw[0].normal=(1,0,0)
    m.close();assert m.normal_contact_observer is None


def test_raw_overflow_fault_latches_and_prepare_cannot_continue(native_stub,monkeypatch):
    r=fixture(monkeypatch)
    with pytest.raises(RuntimeError,match='overflow'):step(r,0,(0.,)*257)
    assert len(r.edge_contact_rows)==256 and not r.releases
    with pytest.raises(RuntimeError,match='overflow'):r.prepare_step(None)


@pytest.mark.parametrize('j',[[-8.7742363e-38,0,0],[-9.444774048741853e-21,6.73e-29,0],
                              [-4.2010698364712225e-8,0,0]])
def test_negative_native_scale_rows_retained_without_signed_zero_clipping(j):
    _,scalar=knife_normal_impulse([1,0,0],j)
    assert scalar==j[0] and scalar<0


def test_unrelated_microscopic_row_not_new_collinearity_fault(native_stub,monkeypatch):
    r=fixture(monkeypatch)
    result=step(r,0,(-1e-38,),other='/World/T/Leaf',normal=(.5,0,0))
    assert result['edge_contact_count']==0
    assert result['raw_normal_rows'][0]['rejection']=='not_seam_adjacent_shaft'
    assert r.event_monitor.error is None


@pytest.mark.parametrize('change',[dict(impulse_contract=None),dict(impulse_contract='guessed'),
    dict(normals=None),dict(tool_contact_upper_bound_n=None)])
def test_unproven_gate_contract_fails_closed(change):
    g=ShearGate('petiole')
    with pytest.raises(ValueError,match='contract'):sample(g,0,**change)
    assert not g.completed and g.dwell==0


@pytest.mark.parametrize('change',[dict(normals=[[2,0,0]]),dict(impulses=[[.00125,.001,0]]),
    dict(impulses=[[float('nan'),0,0]])])
def test_invalid_eligible_normal_row_faults(change):
    with pytest.raises(ValueError):sample(ShearGate('petiole'),0,**change)


def seam_fixture():
    from pxr import Usd,UsdGeom,UsdPhysics,Gf
    from sim_physics.plant import PlantRig
    stage=Usd.Stage.CreateInMemory()
    for name in ('A','B'):
        prim=UsdGeom.Xform.Define(stage,'/World/'+name).GetPrim()
        UsdGeom.Xformable(prim).AddTranslateOp().Set(Gf.Vec3d(1,2,3))
        body=UsdPhysics.RigidBodyAPI.Apply(prim)
        body.CreateVelocityAttr(Gf.Vec3f(.1,.2,.3));body.CreateAngularVelocityAttr(Gf.Vec3f(.4,.5,.6))
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr(.001)
    joint=UsdPhysics.FixedJoint.Define(stage,'/World/Seam')
    joint.CreateBody0Rel().SetTargets(['/World/A']);joint.CreateBody1Rel().SetTargets(['/World/B'])
    joint.CreateJointEnabledAttr(True)
    rig=PlantRig(stage,'/World','petiole',['/World/A','/World/B'],np.repeat(np.eye(4)[None],2,axis=0),
        np.zeros((3,3)),np.array([0.,.01,.02]),1,'/World/Seam',[])
    return rig,joint


def evidence(model):
    g=ShearGate('petiole',ShearParameters(model=model))
    return next(e for i in range(8) if (e:=sample(g,i*.0001 if model==LEGACY_CUT_MODEL else 0)) is not None)


@pytest.mark.parametrize('model',[LEGACY_CUT_MODEL,BRITTLE_CUT_MODEL])
def test_independent_release_only_joint_enabled_session_opinion(model):
    rig,joint=seam_fixture();original=rig.stage.GetRootLayer().ExportToString()
    e=evidence(model);result=rig.release_from_blade(e)
    assert joint.GetJointEnabledAttr().Get() is False
    assert rig.stage.GetRootLayer().ExportToString()==original
    session=rig.stage.GetSessionLayer().ExportToString()
    assert 'physics:jointEnabled = 0' in session
    assert all(name not in session for name in ('velocity','angularVelocity','mass','xformOp'))
    assert not result['physical_cut_verified'] and not result['training_eligible']
    with pytest.raises(ValueError,match='current state'):rig.release_from_blade(e)


@pytest.mark.parametrize('change',[
    dict(model='unknown'),dict(force_contract=None),dict(minimum_signed_resistance_n=-.25),
    dict(minimum_signed_resistance_n=.199),dict(peak_tool_contact_upper_bound_n=.501),
    dict(contact_dwell_s=.024),dict(flat_edge_contact_verified=False),dict(stable_left_grasp=False),
    dict(maximum_axial_contact_distance_m=.0031),dict(maximum_edge_axis_dot_stem=.3),
    dict(minimum_relative_step_m=-1.01e-6),dict(grasp_slip_m=.003),
    dict(loading_travel_required=True),dict(loading_travel_requirement_met=True),
    dict(minimum_loading_travel_m=.0003),dict(fracture_energy_used_as_evidence=True)])
def test_plant_rejects_brittle_evidence_before_any_topology_change(change):
    rig,joint=seam_fixture();before=rig.stage.GetSessionLayer().ExportToString()
    e=evidence(BRITTLE_CUT_MODEL);e.update(change)
    with pytest.raises(ValueError):rig.release_from_blade(e)
    assert joint.GetJointEnabledAttr().Get() and not rig.cut
    assert rig.stage.GetSessionLayer().ExportToString()==before


def test_old_magnitude_only_release_evidence_and_short_legacy_travel_rejected():
    rig,_=seam_fixture();e=evidence(LEGACY_CUT_MODEL)
    del e['force_contract']
    with pytest.raises(ValueError):rig.release_from_blade(e)
    e=evidence(LEGACY_CUT_MODEL);e['measured_relative_loading_travel_m']=0.
    with pytest.raises(ValueError,match='travel'):rig.release_from_blade(e)


def test_model_constructor_validates_before_authoring_and_reset_preserves_optin(monkeypatch):
    from sim_physics.full_robot import FullRobotGripper
    with pytest.raises(ValueError,match='cut model'):BimanualRobot(None,None,cut_model='guess')
    r=fixture(monkeypatch);r.held_plant_screen=S(workspace=(0,1),static=['stale'])
    r.kin=S(forward=lambda *args:np.eye(4));r.right=np.zeros(7);r.base=np.eye(4)
    monkeypatch.setattr(FullRobotGripper,'restore_authored_state',lambda self:None)
    r.edge_contact_rows=[{'stale':True}];r.edge_contact_error='old fault'
    r.restore_authored_state()
    assert r.cut_gate.parameters.model==BRITTLE_CUT_MODEL
    assert r.edge_contact_rows==[] and r.edge_contact_error is None and not r.cut_authorized
    assert ShearParameters().model==LEGACY_CUT_MODEL
