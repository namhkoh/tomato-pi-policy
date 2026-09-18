"""CPU control/attribute tamper checks; never image or geometry-review labels."""
from pathlib import Path
import numpy as np
import pytest
from .procedural_petiole_controlled_v1 import (control_for, controlled_curve,
    copy_static_component, verify_visible_component)


@pytest.mark.parametrize('amplitude', [.002, .025])
def test_predeclared_displacement_preserves_anchor_and_sampled_radii(amplitude):
    points = np.array([[1.,2.,3.],[1.03,2.,3.],[1.08,2.01,3.],[1.15,2.04,3.03]])
    radii = np.array([.003,.0028,.0025,.0015]); before = points.copy()
    curve,direction = controlled_curve(points,radii,[0,0,1],amplitude)
    again,other_direction = controlled_curve(points,radii,[0,0,1],amplitude)
    displacements = curve['points']-(points-points[0])
    assert np.array_equal(points,before) and np.array_equal(curve['points'][0],[0,0,0])
    assert np.array_equal(curve['radius'],radii)
    assert np.allclose(displacements[-1],amplitude*direction,atol=1e-15)
    assert np.isclose(np.max(np.linalg.norm(displacements,axis=1)),amplitude,atol=1e-15)
    assert np.array_equal(curve['points'],again['points']) and np.array_equal(direction,other_direction)
    assert (np.diff(curve['arc'])>0).all()


@pytest.mark.parametrize('amplitude', [0., .001, .003, -.002, float('nan'), True])
def test_unplanned_strength_is_rejected(amplitude):
    with pytest.raises(ValueError): control_for(amplitude)


def test_degenerate_anatomical_direction_is_rejected_without_fallback_search():
    with pytest.raises(ValueError): controlled_curve([[0,0,0],[0,0,.1]],[.003,.002],[0,0,1],.002)


def usd_fixture(tmp_path):
    from pxr import Usd,UsdGeom,UsdPhysics,Gf,Vt,Sdf
    source=tmp_path/'source.usda'; output=tmp_path/'copy.usda'
    stage=Usd.Stage.CreateNew(str(source)); root=UsdGeom.Xform.Define(stage,'/Root').GetPrim();stage.SetDefaultPrim(root)
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    mesh=UsdGeom.Mesh.Define(stage,'/Root/Mesh')
    mesh.CreatePointsAttr([(0,0,0),(1,0,0),(0,1,0)])
    mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
    mesh.CreateNormalsAttr([(0,0,.75)]*3);mesh.SetNormalsInterpolation('vertex')
    mesh.CreateExtentAttr([(0,0,0),(1,1,0)])
    uv=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'vertex');uv.Set([(0,0),(1,0),(0,1)])
    UsdPhysics.RigidBodyAPI.Apply(root).CreateRigidBodyEnabledAttr(True)
    stage.GetRootLayer().Save();stage=None
    receipt=copy_static_component(source,output,tmp_path,{})
    return source,output,receipt


def test_static_copy_preserves_nonunit_normals_and_uv_verbatim(tmp_path):
    from pxr import Usd,UsdGeom
    source,output,receipt=usd_fixture(tmp_path)
    proof=verify_visible_component(source,output,spatial_changes_allowed=False)
    assert not proof['changed_visible_attributes'] and receipt['removed_physics_apis']
    stage=Usd.Stage.Open(str(output));normals=UsdGeom.Mesh(stage.GetPrimAtPath('/Root/Mesh')).GetNormalsAttr().Get()
    assert np.array_equal(np.asarray(normals),[[0,0,.75]]*3)


@pytest.mark.parametrize('channel',['points','normals','primvars:st'])
def test_protected_visible_channel_tamper_is_rejected(tmp_path,channel):
    from pxr import Usd
    source,output,_=usd_fixture(tmp_path)
    stage=Usd.Stage.Open(str(output));attr=stage.GetPrimAtPath('/Root/Mesh').GetAttribute(channel)
    values=attr.Get();values[0]=tuple(.125+v for v in values[0]);attr.Set(values);stage.GetRootLayer().Save();stage=None
    with pytest.raises(ValueError): verify_visible_component(source,output,spatial_changes_allowed=False)


def test_selected_subtree_still_cannot_change_uv(tmp_path):
    from pxr import Usd
    source,output,_=usd_fixture(tmp_path)
    stage=Usd.Stage.Open(str(output));attr=stage.GetPrimAtPath('/Root/Mesh').GetAttribute('primvars:st')
    values=attr.Get();values[0]=(.4,.4);attr.Set(values);stage.GetRootLayer().Save();stage=None
    with pytest.raises(ValueError): verify_visible_component(source,output,spatial_changes_allowed=True)
