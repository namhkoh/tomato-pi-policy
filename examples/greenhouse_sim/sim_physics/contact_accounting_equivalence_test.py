"""Old full recomputation versus changed-pair recomputation, after EVERY row."""
import math
import numpy as np
from .contact_events import ContactEvents,_sum


class Reference(ContactEvents):
    def _refresh_totals(self,changed_key=None):
        normal={k:[] for k in self._buckets};friction={k:[] for k in self._buckets}
        for key,state in self._pair_states.items():
            for kind,value in state['normal'].items():normal[kind].append(value)
            kind=state['kind']
            if kind=='tool_candidate':
                kind='allowed_tool' if state['normal_count'] and state['all_tool_normals'] else 'unwanted'
            friction[kind].append(state['friction'])
            self.normal_pairs[key]=_sum(state['normal'].values())
            self.friction_pairs[key]=state['friction']
            self.pairs[key]=_sum((self.normal_pairs[key],self.friction_pairs[key]))
        for kind in self._buckets:
            f=_sum(friction[kind]);n=_sum(normal[kind])
            setattr(self,kind+'_friction_impulse',f);setattr(self,kind+'_impulse',_sum((n,f)))
        self.normal_impulse=_sum(self.normal_pairs.values())
        self.friction_impulse=_sum(self.friction_pairs.values())
        _sum((self.normal_impulse,self.friction_impulse))


def monitor(cls=ContactEvents):
    m=cls(robot_root='/R',target_root='/P',fingers=['/R/finger'],floor_root='/Floor')
    m.tool_contact=lambda robot,other,point,*rest:point[0]==0
    return m


def same(a,b):
    assert a.measurements(.002)==b.measurements(.002)
    assert a.pairs==b.pairs and a.normal_pairs==b.normal_pairs and a.friction_pairs==b.friction_pairs
    assert a._pair_states==b._pair_states


def test_random_mixed_original_order_headers_are_exact_after_each_consume():
    rng=np.random.default_rng(241);a=monitor();b=monitor(Reference)
    paths=['/R/finger/shape','/R/knife/shape','/R/arm/shape','/R/wheel_l/shape',
           '/P/Stem','/Neighbor/Leaf','/Floor/mesh']
    for step in range(8):
        a.begin_step();b.begin_step();same(a,b)
        for _ in range(180):
            first,second=rng.choice(paths,size=2,replace=False)
            count=int(rng.integers(0,5));friction_count=int(rng.integers(0,4))
            # Large exponent range tests summation/cancellation hazards while
            # retaining finite native magnitudes; no force limits are changed.
            impulses=(rng.normal(size=(count,3))*10.**rng.integers(-20,15)).tolist()
            frictions=(rng.normal(size=(friction_count,3))*10.**rng.integers(-20,15)).tolist()
            points=[[int(rng.integers(0,2)),0,0] for _ in impulses]
            args=(str(first),str(second),impulses,points,[[1,0,0]]*count,[0.]*count)
            kwargs=dict(friction_impulses=frictions,friction_points=[[0,0,0]]*friction_count)
            a.consume(*args,**kwargs);b.consume(*args,**kwargs);same(a,b)


def test_late_wrong_normal_revokes_only_that_pairs_friction_immediately():
    a=monitor();b=monitor(Reference)
    for m in (a,b):
        m.consume('/R/knife/shape','/P/Stem',[(.01,0,0)],[(0,0,0)],[(1,0,0)],[0.],
            friction_impulses=[(.02,0,0)])
        m.consume('/R/finger/shape','/P/Stem',[(.03,0,0)])
        m.consume('/R/knife/shape','/P/Stem',[(0,0,0)],[(1,0,0)],[(1,0,0)],[0.])
    same(a,b)
    assert a.measurements(1)['unwanted_friction_n']==.02
    assert a.measurements(1)['allowed_tool_friction_n']==0
    assert a.measurements(1)['allowed_target_contact_n']==.03


def test_explicit_rebuild_preserves_cached_results():
    m=monitor();m.consume('/R/arm/shape','/P/Stem',[(.001,0,0)])
    before=m.measurements(1);m._refresh_totals();assert m.measurements(1)==before
