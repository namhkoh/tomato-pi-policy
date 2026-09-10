from types import SimpleNamespace
import hashlib
import numpy as np
from pxr import Usd,UsdGeom,UsdPhysics
from sim_physics.greenhouse_context import populate,collision_prototype


def test_context_matches_preview_layout_and_preserves_source(tmp_path):
    folder=tmp_path/'plants/backdrop';folder.mkdir(parents=True)
    asset=folder/'backdrop_000.usd'
    source=Usd.Stage.CreateNew(str(asset))
    root=UsdGeom.Xform.Define(source,'/Plant').GetPrim();source.SetDefaultPrim(root)
    mesh=UsdGeom.Mesh.Define(source,'/Plant/Mesh')
    mesh.CreatePointsAttr([(-.1,0,0),(.1,0,0),(0,0,1)])
    mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2])
    source.GetRootLayer().Save()
    digest=hashlib.sha256(asset.read_bytes()).hexdigest()
    stage=Usd.Stage.CreateInMemory();UsdGeom.Xform.Define(stage,'/World/Gutters')
    for i,x in enumerate((-2.,0.,2.)):
        g=UsdGeom.Cube.Define(stage,f'/World/Gutters/Gutter{i}')
        g.CreateSizeAttr(.2);g.AddTranslateOp().Set((x,0,0))
    before=stage.GetRootLayer().ExportToString()
    robot=SimpleNamespace(base=np.eye(4),window=dict(centre=np.zeros(2),half_extent=2.))
    result=populate(stage,tmp_path,robot)
    assert result['context_plants']==143
    assert result['static_contact_plants']>0 and result['render_only_distant_plants']>0
    assert stage.GetRootLayer().ExportToString()==before
    assert hashlib.sha256(asset.read_bytes()).hexdigest()==digest
    for entry in result['instances']:
        prim=stage.GetPrimAtPath(entry['path'])
        assert prim.IsInstance()
        shape=stage.GetPrimAtPath(entry['path']+'/Mesh')
        assert shape.HasAPI(UsdPhysics.CollisionAPI)==entry['static_contact']
    assert not any(np.allclose(e['position_m'],[.195,0,.9]) for e in result['instances'])
