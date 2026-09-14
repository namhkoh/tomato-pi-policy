"""Exact repeated right-arm IK in one frozen zero-motion station search.

Only kinematic solutions are reused. Native geometry, self/inter-arm checks,
epoch checks and motion authorization NEVER come from this cache. Left elbow
variants cannot affect the stock right-chain IK objective, but torso, parsed
URDF and solver changes invalidate its fixed model binding.
"""
import numpy as np
from greenhouse_sim.robot_kinematics import Rby1Kinematics,IKResult


_METHODS={name:getattr(Rby1Kinematics,name) for name in (
    'solve_pose','forward','_link_transform','_chain','_resolved_torso_degrees',
    'arm_limits_degrees','default_torso_degrees')}


def _stock(kin):
    return type(kin) is Rby1Kinematics and all(
        getattr(getattr(kin,name,None),'__func__',None) is method
        for name,method in _METHODS.items())


def _finite(value,shape):
    a=np.array(value,dtype=np.float64,copy=True)
    if a.shape!=shape or not np.isfinite(a).all():
        raise ValueError('Finite exact-shape frozen IK inputs required')
    return a


def _signature(kin):
    if not _stock(kin):raise RuntimeError('Frozen IK implementation changed')
    values=[_finite(kin.default_torso_degrees(),(6,)).tobytes()]
    # Traverse explicitly with a cycle bound; the stock FK uses this same
    # ordered ancestry, including fixed transforms before/after arm joints.
    link='ee_right';seen=set()
    while link!='base':
        if link in seen or len(seen)>=64:raise RuntimeError('Invalid frozen right-chain ancestry')
        seen.add(link);j=kin._by_child.get(link)
        if j is None or j.child!=link:raise RuntimeError('Frozen right-chain identity changed')
        values.append((j.name,j.parent,j.child,j.kind,
            _finite(j.origin,(4,4)).tobytes(),_finite(j.axis,(3,)).tobytes()))
        link=j.parent
    for limit in kin.arm_limits_degrees('right'):
        values.append(_finite(limit,(7,)).tobytes())
    return tuple(values)


class FrozenRightPoseSolver:
    def __init__(self,kin,guard,*,enabled=False):
        if type(enabled) is not bool:raise ValueError('Explicit frozen IK reuse mode required')
        self.kin=kin;self.guard=guard;self.enabled=enabled and _stock(kin)
        self.model=_signature(kin) if self.enabled else None
        self.entries={};self.hits=0;self.solves=0;self.requests=0

    def _check(self):
        self.guard()
        if self.enabled and _signature(self.kin)!=self.model:
            raise RuntimeError('Frozen right IK model or torso changed')

    def solve(self,pose,seed,base):
        self._check();self.requests+=1
        pose=_finite(pose,(4,4));seed=_finite(seed,(7,));base=_finite(base,(4,4))
        key=(base.tobytes(),pose.tobytes(),seed.tobytes())
        if self.enabled and key in self.entries:
            result=self.entries[key];self._check();self.hits+=1
            return result
        self.solves+=1
        result=self.kin.solve_pose('right',pose,seed,base,
            maximum_evaluations=200,joint_limit_margin_degrees=3.)
        self._check()
        if self.enabled and len(self.entries)<128:
            # Stock IKResult is frozen and contains only scalar/tuple values.
            # Never persist an exception, mutable replacement or invalid result.
            if (type(result) is not IKResult or type(result.joint_degrees) is not tuple
                    or type(result.succeeded) is not bool):
                raise RuntimeError('Unexpected frozen right IK result contract')
            _finite(result.joint_degrees,(7,))
            _finite([result.position_error_m,result.orientation_error_rad,result.cost],(3,))
            self.entries[key]=result
        return result

    def receipt(self):
        return dict(model='exact_frozen_right_chain_ik_reuse_v1',enabled=self.enabled,
            requests=self.requests,actual_solves=self.solves,cache_hits=self.hits,
            entries=len(self.entries),capacity=128,native_clearance_cached=False,
            motion_authorized=False)
