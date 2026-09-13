import numpy as np
import pytest
from pxr import Usd,UsdGeom,UsdPhysics,Sdf
from .finger_coupling import _author,CouplingMonitor


def fixture():
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,'Z')
    for name in ('ee_left','ee_finger_l1','ee_finger_l2'):
        prim=UsdGeom.Xform.Define(stage,'/Robot/'+name).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
    for i in (1,2):
        joint=UsdPhysics.PrismaticJoint.Define(stage,'/Robot/joints/gripper_finger_l'+str(i))
        joint.CreateBody0Rel().SetTargets(['/Robot/ee_left'])
        joint.CreateBody1Rel().SetTargets(['/Robot/ee_finger_l'+str(i)])
        joint.CreateLowerLimitAttr(-.05 if i==1 else 0.)
        joint.CreateUpperLimitAttr(0. if i==1 else .05)
    return stage


class SchemaStub:
    """Only USD authoring contract; never pretends to simulate a mimic."""
    def __init__(self,prim):self.prim=prim
    def CreateReferenceJointRel(self):return self.prim.CreateRelationship('testMimic:referenceJoint')
    def CreateGearingAttr(self):return self.prim.CreateAttribute('testMimic:gearing',Sdf.ValueTypeNames.Float)
    def CreateOffsetAttr(self):return self.prim.CreateAttribute('testMimic:offset',Sdf.ValueTypeNames.Float)
    def CreateNaturalFrequencyAttr(self):return self.prim.CreateAttribute('testMimic:naturalFrequency',Sdf.ValueTypeNames.Float)
    def CreateDampingRatioAttr(self):return self.prim.CreateAttribute('testMimic:dampingRatio',Sdf.ValueTypeNames.Float)


def test_only_session_jaw_relation_authored_no_plant_or_force_or_pose_changes():
    stage=fixture();before=stage.GetRootLayer().ExportToString()
    r=_author(stage,'/Robot',SchemaStub)
    assert stage.GetRootLayer().ExportToString()==before
    layer=stage.GetSessionLayer().ExportToString()
    assert 'gripper_finger_l1' in layer and 'gripper_finger_l2' in layer
    assert all(word not in layer for word in ('xformOp','drive:','velocity','mass','Collision','Plant'))
    assert not r['native_parameters_readback'] and not r['tissue_or_grasp_qualified']


@pytest.mark.parametrize('change',['wrong_parent','wrong_child','wrong_limit','units','up_axis'])
def test_geometry_mismatch_refused_before_any_edit(change):
    s=fixture();j=UsdPhysics.PrismaticJoint(s.GetPrimAtPath('/Robot/joints/gripper_finger_l2'))
    if change=='wrong_parent':j.GetBody0Rel().SetTargets(['/Robot/Other'])
    elif change=='wrong_child':j.GetBody1Rel().SetTargets(['/Robot/ee_finger_l1'])
    elif change=='wrong_limit':j.GetLowerLimitAttr().Set(-.05)
    elif change=='units':UsdGeom.SetStageMetersPerUnit(s,.01)
    elif change=='up_axis':UsdGeom.SetStageUpAxis(s,'Y')
    before=s.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):_author(s,'/Robot',SchemaStub)
    assert s.GetSessionLayer().ExportToString()==before


def test_measured_relation_never_sets_state_or_proves_grasp():
    m=CouplingMonitor();q=np.array([-.008,.0081]);v=np.array([.01,-.01]);before=q.copy()
    r=m.observe(q,v,step=1)
    np.testing.assert_array_equal(q,before)
    assert r['position_sum_m']==pytest.approx(.0001)
    assert not r['grasp_verified'] and not r['native_constraint_authenticity_verified']
    with pytest.raises(RuntimeError):m.observe(q,v,step=1)
    with pytest.raises(RuntimeError):m.observe([-.008,.009],v,step=2)


def test_uncoupled_controller_remains_default():
    from .force_closure import ForceClosure
    c=ForceClosure(.003,.001)
    assert c.physical_gear_coupling_modeled is False


def test_coupling_requires_complete_profile_and_never_watched_launch(tmp_path):
    from .benchmark import main
    with pytest.raises(ValueError,match='Coupled fingers'):
        main(['--output',str(tmp_path/'unused'),'--coupled-fingers-trial'])
    assert not (tmp_path/'unused').exists()
