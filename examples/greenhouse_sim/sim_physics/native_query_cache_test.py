import numpy as np
import pytest
from .native_static_clearance import NativeStaticClearance
from .native_static_clearance_test import records


def test_repeated_exact_query_uses_two_native_calls_but_final_control_is_fresh():
    calls=[]
    def query(*a):calls.append(a);return len(calls) in (1,3)
    n=NativeStaticClearance(query,records(),memoize_queries=True,max_queries=3)
    for _ in range(100):assert n.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    assert n.calls==2 and n.cache_hits==99
    n.validate();n.close()
    assert len(calls)==3 and n.validation_passed and not n.query_cache


def test_final_missing_actor_revokes_cached_misses():
    calls=[]
    n=NativeStaticClearance(lambda *a:calls.append(a) or len(calls)==1,records(),memoize_queries=True)
    for _ in range(2):assert n.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    with pytest.raises(RuntimeError,match='Final native actor'):n.validate()
    assert not n.active and not n.query_cache


@pytest.mark.parametrize('change',['step','timeout','close'])
def test_cached_result_never_survives_epoch_or_lifecycle(change,monkeypatch):
    import sim_physics.native_static_clearance as module
    calls=[]
    n=NativeStaticClearance(lambda *a:calls.append(a) or len(calls)==1,records(),memoize_queries=True)
    assert n.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    if change=='step':n.guard=lambda:(_ for _ in ()).throw(RuntimeError('scene advanced'))
    elif change=='timeout':monkeypatch.setattr(module.time,'perf_counter',lambda:n.started+9)
    else:n.close()
    with pytest.raises(RuntimeError):n.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    assert len(calls)==2 and not n.query_cache


def test_float64_geometry_is_exact_not_rounded_and_hits_stay_blocked():
    calls=[]
    def query(*a):calls.append(a);return len(calls)!=2
    n=NativeStaticClearance(query,records(),memoize_queries=True)
    p=np.array([1.,0.,0.]);a=np.eye(3);h=np.ones(3)
    assert n.clear_box_checked('/World/Stem',p,a,h,.001)
    p[0]=np.nextafter(p[0],np.inf)
    assert not n.clear_box_checked('/World/Stem',p,a,h,.001)
    assert not n.clear_box_checked('/World/Stem',p,a,h,.001)
    assert len(calls)==3 and n.cache_hits==1


def test_sphere_cache_does_not_cache_positive_controls():
    calls=[]
    def sphere(path,point,radius):calls.append(radius);return radius>.01
    n=NativeStaticClearance(lambda *a:True,records(),memoize_queries=True,sphere_query=sphere)
    assert n._sphere_overlap('/World/Stem',np.zeros(3),.002,memoize=True) is False
    assert n._sphere_overlap('/World/Stem',np.zeros(3),.002,memoize=True) is False
    n._sphere_control('/World/Stem');n._sphere_control('/World/Stem')
    assert len(calls)==3 and n.cache_hits==1


def test_cache_is_bounded():
    n=NativeStaticClearance(lambda *a:True,records(),memoize_queries=True)
    for i in range(4100):
        assert n._overlap('/World/Stem',np.array([float(i),0,0]),np.eye(3),np.ones(3),memoize=True)
    assert len(n.query_cache)==4096
    n.close();assert not n.query_cache
