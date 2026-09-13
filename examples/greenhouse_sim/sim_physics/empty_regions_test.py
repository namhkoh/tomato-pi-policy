import itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from .empty_regions import EmptyRegions
from .native_static_clearance import NativeStaticClearance
from .native_static_clearance_test import records


def test_all_corners_of_rotated_box_must_fit_in_native_empty_region():
    cache=EmptyRegions();centre=np.zeros(3);axes=np.eye(3);half=np.array([.05,.02,.01])
    cache.add('/stem',centre,axes,half)
    assert cache.contains('/stem',centre,axes,half-.001)
    assert not cache.contains('/other',centre,axes,half-.001)
    assert not cache.contains('/stem',centre,Rotation.from_euler('z',90,degrees=True).as_matrix(),half-.001)
    assert not cache.contains('/stem',[.002,0,0],axes,half-.001)


def test_random_containment_matches_eight_corner_test():
    rng=np.random.default_rng(16);cache=EmptyRegions();signs=np.array(list(itertools.product((-1,1),repeat=3)))
    c=np.array([.2,-.3,.4]);r=Rotation.random(random_state=rng).as_matrix();h=np.array([.05,.02,.01])
    cache.add('/stem',c,r,h)
    for _ in range(300):
        q=c+rng.uniform(-.02,.02,3);a=Rotation.random(random_state=rng).as_matrix();b=rng.uniform(.001,.02,3)
        inside=cache.contains('/stem',q,a,b)
        corners=(q+signs*b@a.T-c)@r
        assert inside==bool(np.all(np.abs(corners)+1e-7<=h))


def test_cached_regions_are_copied_and_bounded():
    c=np.zeros(3);r=np.eye(3);h=np.ones(3);cache=EmptyRegions()
    cache.add('/stem',c,r,h);c[:]=4;h[:]=.1
    assert cache.contains('/stem',np.zeros(3),r,np.ones(3)*.5)
    for i in range(140):
        for j in range(20):cache.add('/'+str(i),c,r,h)
    assert len(cache.regions)==128 and max(map(len,cache.regions.values()))<=16
    cache.clear();assert cache.report()['retained']==0


def make(query,**options):
    return NativeStaticClearance(query,records(),reuse_clear_regions=True,**options)


def test_large_native_miss_proves_small_queries_without_caching_positive_controls():
    calls=[]
    def query(*a):
        calls.append(a)
        return len(calls) in (1,3)
    n=make(query)
    for x in np.linspace(0,.004,9):
        assert n.clear_box_checked('/World/Stem',np.array([x,0,0]),np.eye(3),np.ones(3)*.01,.001)
    assert n.calls==2 and n.empty_regions.hits==8
    np.testing.assert_allclose(calls[1][3],np.ones(3)*.016)
    n.validate();n.close()
    assert n.calls==3 and n.validation_passed and not n.empty_regions.regions


def test_large_hit_never_overrides_original_small_miss_or_hit():
    calls=[]
    def query(path,c,a,h):
        calls.append(h.copy())
        return bool(len(calls)==1 or h[0]>.014 or c[0]>.002)
    n=make(query)
    assert n.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.ones(3)*.01,.001)
    assert not n.clear_box_checked('/World/Stem',np.array([.003,0,0]),np.eye(3),np.ones(3)*.01,.001)
    assert n.calls==5 and n.empty_regions.report()['retained']==0


@pytest.mark.parametrize('fault',['step','timeout','final_control','unknown_result'])
def test_unavailable_native_or_epoch_revokes_all_containment(fault,monkeypatch):
    import sim_physics.native_static_clearance as module
    calls=[]
    def query(*a):
        calls.append(a)
        if fault=='unknown_result' and len(calls)==2:return None
        return len(calls)==1
    n=make(query)
    if fault=='unknown_result':
        with pytest.raises(RuntimeError):n.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.ones(3)*.01,.001)
    else:
        assert n.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.ones(3)*.01,.001)
        if fault=='step':n.guard=lambda:(_ for _ in ()).throw(RuntimeError('physics advanced'))
        if fault=='timeout':monkeypatch.setattr(module.time,'perf_counter',lambda:n.started+9)
        with pytest.raises(RuntimeError):
            if fault=='final_control':n.validate()
            else:n.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.ones(3)*.01,.001)
    assert not n.active and not n.empty_regions.regions
