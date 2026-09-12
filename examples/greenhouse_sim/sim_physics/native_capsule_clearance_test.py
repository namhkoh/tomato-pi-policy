"""Conservative capsule query containment; pure geometry/CPU USD, not native proof."""
from types import SimpleNamespace as S
import numpy as np
import pytest
from sim_physics.native_static_clearance import capsule_enclosing_box,NativeStaticClearance


@pytest.mark.parametrize('delta',[[0,0,0],[.2,0,0],[0,-.3,0],[.1,.2,-.4],[1e-160,0,0]])
def test_enclosing_box_contains_sweep_and_expanded_sphere_ends(delta):
    a=np.array([.1,-.2,.8]);b=a+delta;r=.02;m=.001
    centre,axes,half=capsule_enclosing_box(a,b,r)
    assert np.linalg.det(axes)==pytest.approx(1.)
    np.testing.assert_allclose(axes.T@axes,np.eye(3),atol=1e-14)
    rng=np.random.default_rng(41);directions=rng.normal(size=(300,3))
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    directions=np.vstack((directions,axes.T,-axes.T))
    # Endpoints included; the box must enclose full hemispheres, not only
    # centreline/cylinder samples. Whole-sphere ends are a stronger condition.
    for t in np.linspace(0,1,9):
        points=a+t*(b-a)+(r+m)*directions
        assert np.all(np.abs((points-centre)@axes)<=half+m+1e-14)


@pytest.mark.parametrize('a,b,r',[
    ([0,0,0],[0,0,1],0),([0,0,0],[0,0,1],True),
    ([0,0,float('nan')],[0,0,1],.1),([0,0],[0,0,1],.1),
    ([0,0,0],[0,0,1],float('inf'))])
def test_invalid_capsule_cannot_query(a,b,r):
    with pytest.raises(ValueError):capsule_enclosing_box(a,b,r)


def test_checked_native_query_retains_whole_box_and_original_margin():
    calls=[]
    def query(path,c,r,h):
        calls.append((path,c.copy(),r.copy(),h.copy()));return len(calls)==1
    native=NativeStaticClearance(query,[('/World/Stem','box',None,np.full(3,-1),np.full(3,1))])
    a=np.array([.1,0,0]);b=np.array([.1,.2,0]);radius=.02
    centre,axes,half=capsule_enclosing_box(a,b,radius)
    assert native.clear_capsule_checked('/World/Stem',a,b,radius,.001)
    np.testing.assert_allclose(calls[-1][1],centre)
    np.testing.assert_allclose(calls[-1][2],axes)
    np.testing.assert_allclose(calls[-1][3],half+.001)
    assert native.report()['capsule_enclosing_box_attempts']==1
    assert native.used_paths=={'/World/Stem'}
    with pytest.raises(ValueError):native.clear_capsule_checked('/World/Stem',a,b,radius,.0009)


@pytest.mark.parametrize('side',['right','left'])
def test_only_named_restored_arm_capsule_refines_static_bounds(side):
    from sim_physics.held_plant_screen_test import fixture
    screen,rig,world=fixture()
    path,body,link,kind,data=screen.shapes[0]
    if side=='left':
        path,body,link=[v.replace('right','left') for v in (path,body,link)]
        screen.arm='left';world={link:np.eye(4)}
    screen.shapes[0]=(body+'/restored_collisions/capsule_00',body,link,kind,data)
    frames=rig.rest_frames.copy();frames[1,1,3]=.5
    screen.static=[('/World/Static','box',(np.array([.1,0,0]),np.eye(3),np.full(3,.03)),
        np.array([.07,-.03,-.03]),np.array([.13,.03,.03]))]
    screen.snapshot(frames);assert not screen.check(world)
    calls=[]
    screen.native_static_query=S(clear_capsule_checked=lambda *a:calls.append(a) or True)
    assert screen.check(world) and calls[-1][0]=='/World/Static' and calls[-1][-1]==.001
    # A valid native hit keeps the rejection.
    screen.native_static_query=S(clear_capsule_checked=lambda *a:False)
    assert not screen.check(world)
    # Dynamic held leaves cannot be refined away using a static scene query.
    screen.native_static_query=S(clear_capsule_checked=lambda *a:True)
    screen.snapshot(rig.rest_frames);assert not screen.check(world)
    assert screen.last_failure['plant_collider'].endswith('/Leaf')
    screen.snapshot(frames)
    screen.shapes[0]=(path,body,link,kind,data)
    screen.native_static_query=S(clear_capsule_checked=lambda *a:pytest.fail('Unverified capsule queried'))
    assert not screen.check(world)
