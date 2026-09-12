import numpy as np
import pytest
from types import SimpleNamespace as S
from pxr import Gf,Usd,UsdGeom,UsdPhysics
from sim_physics.grasp_span import connected_span,finger_interval
from sim_physics.held_plant_screen import HeldPlantScreen


def chain(step=.01,count=12):
    return [(np.array([0,0,i*step]),np.array([0,0,(i+1)*step]),.002) for i in range(count)]


def test_same_physical_pad_can_span_more_than_two_neighbors():
    indices=connected_span(chain(),5,2,[0,0,1],[0,0,.055],[-.017,.017])
    assert indices=={3,4,5,6,7}
    assert connected_span(chain(.02,6),2,1,[0,0,1],[0,0,.055],[-.017,.017])=={1,2,3}


def test_support_and_disconnected_foldback_are_never_allowed():
    caps=chain();caps[-1]=caps[5]
    assert connected_span(caps,5,4,[0,0,1],[0,0,.055],[-.017,.017])=={4,5,6,7}
    assert 11 not in connected_span(caps,5,4,[0,0,1],[0,0,.055],[-.017,.017])


@pytest.mark.parametrize('kwargs',[
    dict(selected=0),dict(first_detachable=6),dict(axis=[0,0,2]),
    dict(interval=[0,0]),dict(interval=[float('nan'),1]),dict(origin=[0,0,1]),
])
def test_invalid_or_mismatched_span_fails_closed(kwargs):
    values=dict(selected=5,first_detachable=2,axis=[0,0,1],origin=[0,0,.055],interval=[-.017,.017])
    values.update(kwargs)
    with pytest.raises(ValueError):connected_span(chain(),**values)


def test_actual_complete_finger_bounds_not_index_count_define_interval():
    shapes=[(name,name,name,'box',(np.zeros(3),np.eye(3),np.array([.003,.005,.016])))
        for name in ('ee_finger_l1','ee_finger_l2')]
    world={s[2]:np.eye(4) for s in shapes}
    for f in world.values():f[2,3]=.055
    np.testing.assert_allclose(finger_interval(shapes,world,[0,0,1],[0,0,.055]),[-.017,.017])
    with pytest.raises(ValueError):finger_interval(shapes[:1],world,[0,0,1],[0,0,.055])
    with pytest.raises(ValueError):finger_interval(shapes,world,[0,0,1],[0,0,.055],margin=0)


def test_refined_screen_expects_only_contiguous_shaft_under_fingers():
    stage=Usd.Stage.CreateInMemory();paths=[];frames=np.repeat(np.eye(4)[None],12,axis=0)
    for i in range(12):
        path=f'/World/P/S{i}';paths.append(path);frames[i,2,3]=.005+.01*i
        UsdGeom.Xform.Define(stage,path).AddTranslateOp().Set(Gf.Vec3d(*frames[i,:3,3]))
        cap=UsdGeom.Capsule.Define(stage,path+'/StemCollider')
        cap.CreateRadiusAttr(.002);cap.CreateHeightAttr(.01);cap.CreateAxisAttr('Z')
        UsdPhysics.CollisionAPI.Apply(cap.GetPrim())
    rig=S(stage=stage,body_paths=paths,cut_index=2,rest_frames=frames)
    shapes=[(name,name,name,'box',(np.zeros(3),np.eye(3),np.array([.003,.005,.016])))
        for name in ('ee_finger_l1','ee_finger_l2')]
    world={s[2]:np.eye(4) for s in shapes}
    for i,f in enumerate(world.values()):f[:3,3]=[(-1 if i else 1)*.004,0,.055]
    screen=HeldPlantScreen(rig,shapes,'knife',arm='left',grasp_path=paths[5]);screen.snapshot(frames)
    assert not screen.check(world,grasp=True)  # Old +/-one link is too narrow.
    before=stage.GetRootLayer().ExportToString()
    receipt=screen.set_physical_grasp_span(frames,world,5,np.array([0,0,.055]))
    assert receipt['allowed_indices']==[3,4,5,6,7]
    assert screen.check(world,grasp=True)
    assert not screen.check(world,grasp=False)
    # A leaf is never covered by the shaft expected-contact list.
    screen.obstacles.append((paths[5]+'/Leaf','capsule',(np.array([0,0,.05]),np.array([0,0,.06]),.003),
        np.array([-.003,-.003,.047]),np.array([.003,.003,.063])))
    screen.lower=np.array([v[3] for v in screen.obstacles]);screen.upper=np.array([v[4] for v in screen.obstacles])
    assert not screen.check(world,grasp=True)
    assert screen.last_failure['plant_collider'].endswith('/Leaf')
    assert stage.GetRootLayer().ExportToString()==before
