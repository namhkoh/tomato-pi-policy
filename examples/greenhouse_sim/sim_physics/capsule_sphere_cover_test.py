import numpy as np
import pytest
from sim_physics.capsule_sphere_cover import cover
from sim_physics.native_static_clearance import NativeStaticClearance


def test_union_covers_entire_capsule_and_endcaps_including_native_rounding():
    rng=np.random.default_rng(5)
    for length,r in ((0.,.01),(.3,.035),(.08,.002),(.5,.05)):
        a=np.array([12.34567,-3.1443,1.4221]);axis=rng.normal(size=3);axis/=np.linalg.norm(axis)
        b=a+axis*length;centres,sr=cover(a,b,r,.001)
        directions=rng.normal(size=(4000,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
        t=np.r_[0.,1.,rng.random(3998)]
        points=a+t[:,None]*(b-a)+(r+.001)*directions
        distance=np.linalg.norm(points[:,None,:]-centres[None,:,:],axis=2).min(axis=1)
        assert np.all(distance<=sr)
        np.testing.assert_array_equal(centres,centres.astype(np.float32).astype(float))
        assert sr==float(np.float32(sr)) and len(centres)<=256


def test_worst_between_spheres_and_both_endpoints_are_covered():
    centres,r=cover([0,0,0],[0,0,.3],.03,.001)
    mids=(centres[:-1]+centres[1:])/2;mids[:,0]=.031
    points=np.r_[mids,[[0,0,-.031],[0,0,.331]]]
    assert np.all(np.linalg.norm(points[:,None,:]-centres[None,:,:],axis=2).min(1)<=r)


@pytest.mark.parametrize('r,m,opts',[(True,.001,{}),(.02,0.,{}),(.02,.001,{'max_spheres':1}),(.02,.001,{'extra_m':0.})])
def test_invalid_or_unbounded_cover_fails(r,m,opts):
    with pytest.raises(ValueError):cover([0,0,0],[0,0,.3],r,m,**opts)


def records():return [('/World/Stem','box',None,np.full(3,-.01),np.full(3,.01))]


def test_coarse_box_hit_can_clear_only_after_complete_sphere_cover_and_final_controls():
    calls=[]
    def sphere(path,point,r):
        calls.append(point.copy());return bool(point[0]==0)
    n=NativeStaticClearance(lambda *a:True,records(),sphere_query=sphere)
    assert n.clear_capsule_checked('/World/Stem',[.1,0,0],[.4,0,0],.02,.001)
    assert len(calls)>2 and n.used_paths=={'/World/Stem'}
    assert not n.validation_passed
    n.validate();n.close();r=n.report()
    assert r['final_validation_passed'] and r['sphere_cover_cleared_paths']==['/World/Stem']
    assert r['retained_rejections']==0


def test_any_sphere_hit_keeps_collision_rejection():
    n=NativeStaticClearance(lambda *a:True,records(),sphere_query=lambda *a:True)
    assert not n.clear_capsule_checked('/World/Stem',[.1,0,0],[.4,0,0],.02,.001)
    assert not n.used_paths and n.sphere_calls==2


@pytest.mark.parametrize('failure',['missing','nonbool','exception','epoch','budget','final_missing'])
def test_failed_query_or_coverage_never_creates_clearance(failure):
    calls=[]
    def sphere(path,p,r):
        calls.append(p.copy())
        if failure=='nonbool':return None
        if failure=='exception':raise RuntimeError('native callback fault')
        if failure=='missing':return False
        if failure=='epoch' and len(calls)==1:n.guard=lambda:(_ for _ in ()).throw(RuntimeError('physics advanced'))
        if p[0]==0:return not (failure=='final_missing' and len(calls)>1)
        return False
    n=NativeStaticClearance(lambda *a:True,records(),sphere_query=sphere,max_queries=3 if failure=='budget' else 20000)
    if failure=='final_missing':
        assert n.clear_capsule_checked('/World/Stem',[.1,0,0],[.4,0,0],.02,.001)
        with pytest.raises(RuntimeError):n.validate()
    else:
        with pytest.raises(RuntimeError):n.clear_capsule_checked('/World/Stem',[.1,0,0],[.4,0,0],.02,.001)
        assert not n.used_paths
    assert not n.validation_passed and not n.active


def test_sphere_mode_requires_native_static_flag_before_output(tmp_path):
    from sim_physics.benchmark import main
    output=tmp_path/'unused'
    with pytest.raises(ValueError,match='requires bimanual native'):
        main(['--output',str(output),'--native-capsule-sphere-cover'])
    assert not output.exists()
