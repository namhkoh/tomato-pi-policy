from types import SimpleNamespace as S
import pytest
from pxr import Sdf,Usd,UsdGeom,UsdPhysics,UsdShade
from sim_physics.seam_contact_compliance import apply
from sim_physics.plant import _collision


def fixture():
    stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    paths=['/T/S0','/T/S1','/T/S2']
    for path in paths:
        body=UsdGeom.Xform.Define(stage,path).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(body);UsdPhysics.MassAPI.Apply(body).CreateMassAttr(.001)
        shape=UsdGeom.Cylinder.Define(stage,path+'/StemCollider')
        shape.CreateRadiusAttr(.003);shape.CreateHeightAttr(.02);_collision(shape.GetPrim())
    joint=UsdPhysics.FixedJoint.Define(stage,'/T/Seam');joint.CreateJointEnabledAttr(True)
    joint.CreateBody0Rel().SetTargets([paths[0]]);joint.CreateBody1Rel().SetTargets([paths[1]])
    return S(stage=stage,root='/T',body_paths=paths,cut_index=1,cut=False,
        cut_joint_path='/T/Seam',stem_contact_model='flat_cylinders_v1')


def test_only_contact_material_changes_in_session_layer():
    f=fixture();before=f.stage.GetRootLayer().ExportToString()
    r=apply(f,diagnostic_only=True)
    assert f.stage.GetRootLayer().ExportToString()==before
    assert not r['knife_softened'] and not r['release_thresholds_changed']
    assert not r['calibrated'] and not r['training_eligible']
    assert not r['geometry_changed'] and not r['seam_joint_changed'] and not r['masses_changed']
    assert UsdPhysics.Joint.Get(f.stage,f.cut_joint_path).GetJointEnabledAttr().Get()
    for i,path in enumerate(f.body_paths):
        p=f.stage.GetPrimAtPath(path+'/StemCollider')
        bound,_=UsdShade.MaterialBindingAPI(p).ComputeBoundMaterial(materialPurpose='physics')
        if i==2:assert not bound
        else:
            assert str(bound.GetPath())==r['material_path']
            m=bound.GetPrim()
            assert m.GetAttribute('physxMaterial:compliantContactStiffness').Get()==1000
            assert m.GetAttribute('physxMaterial:compliantContactDamping').Get()==2
            assert m.GetAttribute('physxMaterial:compliantContactAccelerationSpring').Get() is False
            assert UsdPhysics.MaterialAPI(m).GetDynamicFrictionAttr().Get()==.5
        assert UsdGeom.Cylinder(p).GetRadiusAttr().Get()==pytest.approx(.003)
        assert UsdPhysics.MassAPI(f.stage.GetPrimAtPath(path)).GetMassAttr().Get()==pytest.approx(.001)


@pytest.mark.parametrize('change',[dict(cut=True),dict(cut_index=0),dict(cut_index=True),
    dict(stem_contact_model='flush_capsules_v1')])
def test_invalid_rig_rejected_without_mutation(change):
    f=fixture();f.__dict__.update(change);before=f.stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):apply(f,diagnostic_only=True)
    assert f.stage.GetSessionLayer().ExportToString()==before


@pytest.mark.parametrize('mode',['joint','shape','collision','offset','material','repeat','units'])
def test_all_prerequisites_validated_before_any_new_binding(mode):
    f=fixture();p=f.stage.GetPrimAtPath(f.body_paths[1]+'/StemCollider')
    if mode=='joint':UsdPhysics.Joint.Get(f.stage,f.cut_joint_path).CreateJointEnabledAttr(False)
    elif mode=='shape':f.stage.RemovePrim(str(p.GetPath()))
    elif mode=='collision':UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr(False)
    elif mode=='offset':p.GetAttribute('physxCollision:restOffset').Set(-.001)
    elif mode=='material':
        m=UsdShade.Material.Define(f.stage,'/Other')
        UsdShade.MaterialBindingAPI.Apply(p).Bind(m,materialPurpose='physics')
    elif mode=='repeat':apply(f,diagnostic_only=True)
    elif mode=='units':UsdGeom.SetStageMetersPerUnit(f.stage,.01)
    before=f.stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):apply(f,diagnostic_only=True)
    assert f.stage.GetSessionLayer().ExportToString()==before


def test_explicit_opt_in_required():
    f=fixture();before=f.stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):apply(f)
    assert f.stage.GetSessionLayer().ExportToString()==before


def test_production_default_and_cli_guard(tmp_path):
    from sim_physics.benchmark import parser,main
    assert not parser().parse_args(['--output','unused']).seam_contact_compliance
    with pytest.raises(ValueError,match='isolated blade feedback'):
        main(['--output',str(tmp_path/'unused'),'--seam-contact-compliance'])
    assert not (tmp_path/'unused').exists()
