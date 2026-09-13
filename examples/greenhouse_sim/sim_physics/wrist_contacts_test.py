import numpy as np
import pytest
from pxr import Usd,UsdGeom,UsdPhysics
from greenhouse_sim.robot_model import DEFAULT_ASSET
from .wrist_contacts import refine,partitions


def area(mesh):
    p=np.array(mesh.GetPointsAttr().Get(),float)
    tri=p[np.array(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)]
    return np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1).sum()/2


def test_original_wrist_surfaces_are_all_retained_without_visual_or_source_edits():
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage,'/Robot').GetPrim().GetReferences().AddReference(str(DEFAULT_ASSET))
    before=stage.GetRootLayer().ExportToString()
    result=refine(stage,'/Robot')
    assert len(result['replaced_colliders'])==4
    assert 4<=len(result['collider_paths'])<=32
    for path in result['replaced_colliders']:
        source=UsdGeom.Mesh.Get(stage,path)
        assert not UsdPhysics.CollisionAPI(source).GetCollisionEnabledAttr().Get()
        parts=[p for p in result['collider_paths'] if p.startswith(path+'_Parts/')]
        assert sum(area(UsdGeom.Mesh.Get(stage,p)) for p in parts)==pytest.approx(area(source),rel=2e-5)
        for p in parts:
            mesh=UsdGeom.Mesh.Get(stage,p)
            assert UsdPhysics.CollisionAPI(mesh).GetCollisionEnabledAttr().Get()
            assert UsdPhysics.MeshCollisionAPI(mesh).GetApproximationAttr().Get()=='convexHull'
            np.testing.assert_allclose(UsdGeom.Xformable(mesh).GetLocalTransformation(),UsdGeom.Xformable(source).GetLocalTransformation())
    assert stage.GetRootLayer().ExportToString()==before
    assert not result['source_visual_edited'] and not result['native_qualified']


def test_unknown_or_degenerate_mesh_is_not_dropped():
    with pytest.raises(ValueError):partitions([[0,0,float('nan')]]*3,[3],[0,1,2])
    with pytest.raises(ValueError):partitions([[0,0,0],[1,0,0],[0,1,0]],[3],[0,1,2])
