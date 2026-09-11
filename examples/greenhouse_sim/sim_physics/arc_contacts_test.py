import numpy as np
import pytest


def test_arc_partitions_preserve_every_source_surface_and_leave_window_open():
    from pxr import Usd,UsdGeom,UsdPhysics
    from scipy.spatial import ConvexHull
    from greenhouse_sim.robot_model import DEFAULT_ASSET,ROBOT_ROOT
    from sim_physics.arc_contacts import partitions,refine_arc_contacts
    stage=Usd.Stage.Open(str(DEFAULT_ASSET));root=ROBOT_ROOT+'/ee_right/attachments/DeleafKnife'
    m=UsdGeom.Mesh.Get(stage,root+'/Arc');v=np.array(m.GetPointsAttr().Get(),float)
    c=np.array(m.GetFaceVertexCountsAttr().Get(),int);f=np.array(m.GetFaceVertexIndicesAttr().Get(),int)
    pieces=partitions(v,c,f)
    area=lambda t:np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=1).sum()/2
    assert sum(area(x.reshape(-1,3,3)) for x in pieces)==pytest.approx(area(v[f].reshape(-1,3,3)),rel=1e-7)
    assert 2<=len(pieces)<=14
    # Visible upper window remains physically open; no entire-box shortcut.
    window=np.array([0.,-.09,.026])
    for piece in pieces:
        hull=ConvexHull(piece)
        assert not np.all(hull.equations[:,:3]@window+hull.equations[:,3]<=0)
    before=stage.GetRootLayer().ExportToString()
    result=refine_arc_contacts(stage,ROBOT_ROOT)
    assert len(result['collider_paths'])==len(pieces)
    assert not UsdPhysics.CollisionAPI(stage.GetPrimAtPath(root+'/ArcCollision')).GetCollisionEnabledAttr().Get()
    assert stage.GetRootLayer().ExportToString()==before
    for p in result['collider_paths']:
        assert UsdPhysics.CollisionAPI(stage.GetPrimAtPath(p)).GetCollisionEnabledAttr().Get()
        assert UsdPhysics.MeshCollisionAPI(stage.GetPrimAtPath(p)).GetApproximationAttr().Get()=='convexHull'


def test_unknown_arc_rejected():
    from sim_physics.arc_contacts import partitions
    with pytest.raises(ValueError): partitions(np.zeros((3,3)),[3],[0,1,2])
