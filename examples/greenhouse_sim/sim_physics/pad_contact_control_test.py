from types import SimpleNamespace
import pytest
from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Sdf
from .pad_contact_control import apply
from .benchmark import main


def fixture():
    stage=Usd.Stage.CreateInMemory();mat=UsdShade.Material.Define(stage,'/R/ProbeFingerMaterial')
    api=UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr(.5);api.CreateDynamicFrictionAttr(.5);api.CreateRestitutionAttr(0.)
    for name,value in (('Stiffness',1000.),('Damping',2.)):
        mat.GetPrim().CreateAttribute('physxMaterial:compliantContact'+name,Sdf.ValueTypeNames.Float).Set(value)
    paths=['/R/palm','/R/finger1','/R/finger2'];colliders=[]
    for finger in paths[1:]:
        p=UsdGeom.Cube.Define(stage,finger+'/pad').GetPrim();colliders.append(str(p.GetPath()))
        UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True)
        UsdShade.MaterialBindingAPI.Apply(p).Bind(mat,materialPurpose='physics')
    return SimpleNamespace(stage=stage,root='/R',robot=None,paths=paths,collider_paths=colliders,
        finger_contact_compliance=dict(stiffness_n_m=1000.,damping_n_s_m=2.,force_based=True))


def test_explicit_control_changes_only_session_contact_law_and_labels_it():
    f=fixture();original=f.stage.GetRootLayer().ExportToString()
    r=apply(f,diagnostic_only=True)
    assert r['contact_model_changed'] and not r['force_limits_changed']
    assert r['original_compliant_prior']['stiffness_n_m']==1000.
    assert not r['original_compliant_contact_qualified']
    assert f.stage.GetRootLayer().ExportToString()==original


def test_no_implicit_or_runtime_application():
    f=fixture()
    with pytest.raises(ValueError):apply(f)
    f.robot=object()
    with pytest.raises(ValueError):apply(f,diagnostic_only=True)


def test_unknown_material_cannot_be_overwritten():
    f=fixture();f.finger_contact_compliance['stiffness_n_m']=2000.
    with pytest.raises(ValueError,match='differs'):apply(f,diagnostic_only=True)


def test_cli_scope_is_not_production(tmp_path):
    out=tmp_path/'never_create'
    with pytest.raises(ValueError,match='Rigid pad control'):
        main(['--output',str(out),'--rigid-pad-control'])
    assert not out.exists()
