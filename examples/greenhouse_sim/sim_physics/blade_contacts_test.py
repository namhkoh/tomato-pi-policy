import numpy as np
import pytest
from pxr import Usd,UsdGeom,UsdPhysics
from greenhouse_sim.robot_model import DEFAULT_ASSET
from sim_physics.blade_contacts import refine_blade_contacts,plate_profile


def test_native_source_plate_and_mount_are_partitioned_without_visual_edit():
    if not DEFAULT_ASSET.is_file(): pytest.skip('Robot asset not installed')
    stage=Usd.Stage.CreateInMemory()
    root=UsdGeom.Xform.Define(stage,'/Robot').GetPrim()
    root.GetReferences().AddReference(str(DEFAULT_ASSET));stage.Load('/Robot')
    knife='/Robot/ee_right/attachments/DeleafKnife'
    original=np.array(UsdGeom.Mesh.Get(stage,knife+'/Blade').GetPointsAttr().Get())
    before=stage.GetRootLayer().ExportToString()
    result=refine_blade_contacts(stage,'/Robot')
    assert result['plate_z_range_m']==pytest.approx([-.0065,-.0005],abs=1e-7)
    assert result['original_box_z_range_m']==pytest.approx([-.0065,.0065],abs=1e-7)
    assert result['edge_size_m'][2]==pytest.approx(.006,abs=1e-7)
    assert np.asarray(result['edge_frame'])[2,3]==pytest.approx(-.0035,abs=1e-7)
    assert not UsdPhysics.CollisionAPI(stage.GetPrimAtPath(knife+'/BladeCollision')).GetCollisionEnabledAttr().Get()
    for path in result['collider_paths']:
        prim=stage.GetPrimAtPath(path)
        assert UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get()
        assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()=='convexHull'
    assert len(result['collider_paths'])==2
    def area(mesh):
        p=np.array(mesh.GetPointsAttr().Get(),float)
        tri=p[np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)]
        return np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2
    # Triangle clipping preserves the supplied surface; nothing is removed
    # merely to gain clearance. Convex hulls conservatively close each part.
    original_area=area(UsdGeom.Mesh.Get(stage,knife+'/Blade'))
    assert sum(area(UsdGeom.Mesh.Get(stage,p)) for p in result['collider_paths'])==pytest.approx(original_area,rel=1e-5)
    np.testing.assert_array_equal(UsdGeom.Mesh.Get(stage,knife+'/Blade').GetPointsAttr().Get(),original)
    assert stage.GetRootLayer().ExportToString()==before
    assert not result['source_asset_edited']


def test_unknown_profile_fails_closed():
    with pytest.raises(ValueError): plate_profile(np.full((3,3),np.nan),np.zeros((1,3,3)))
    with pytest.raises(ValueError): plate_profile(np.zeros((3,3)),np.zeros((1,3,3)))
