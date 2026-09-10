"""Cached actual-capsule self screen; no collision filters are added or changed.

Honors the scene's existing disabled adjacent-joint pairs and explicit mounting
filters. Non-capsule tooling remains subject to separate/native checks.
"""
import itertools
import numpy as np

from .capsule_surface import world_capsule


class SelfCapsuleScreen:
    def __init__(self,stage,robot_root):
        from pxr import Usd,UsdGeom,UsdPhysics
        self.root=robot_root;self.shapes=[];self.unsupported=[];self.excluded=set()
        cache=UsdGeom.XformCache()
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_root)):
            if prim.IsA(UsdPhysics.Joint):
                joint=UsdPhysics.Joint(prim)
                if joint.GetJointEnabledAttr().Get() and not joint.GetCollisionEnabledAttr().Get():
                    a=joint.GetBody0Rel().GetTargets();b=joint.GetBody1Rel().GetTargets()
                    if len(a)==len(b)==1: self.excluded.add(frozenset((str(a[0]),str(b[0]))))
            if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
                for path in UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets():
                    self.excluded.add(frozenset((str(prim.GetPath()),str(path))))
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(): continue
            path=str(prim.GetPath())
            body=prim
            while body and not body.IsPseudoRoot() and not body.HasAPI(UsdPhysics.RigidBodyAPI): body=body.GetParent()
            if not body or body.IsPseudoRoot(): raise ValueError('Robot collision shape has no rigid body: '+path)
            local=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(body)).T)@np.asarray(cache.GetLocalToWorldTransform(prim)).T
            cap=world_capsule(prim,local)
            if cap is None: self.unsupported.append(path);continue
            body_path=str(body.GetPath());link=body_path[len(robot_root)+1:]
            self.shapes.append((path,body_path,link,*cap))
        self.pairs=[]
        for i,j in itertools.combinations(range(len(self.shapes)),2):
            a,b=self.shapes[i],self.shapes[j]
            if a[1]==b[1] or any(frozenset((x,y)) in self.excluded for x in a[:2] for y in b[:2]): continue
            self.pairs.append((i,j))
        if not self.shapes or not self.pairs: raise ValueError('No self-collision capsule coverage')

    def check(self,body_world,*,margin=.003,include_clearances=False):
        from greenhouse_sim.robot_kinematics import _segment_segment_distance
        if not np.isfinite(margin) or margin<0: raise ValueError('Invalid self-clearance margin')
        capsules=[]
        for path,body,link,start,end,radius in self.shapes:
            matrix=np.asarray(body_world[link])
            if matrix.shape!=(4,4) or not np.isfinite(matrix).all(): raise ValueError('Invalid body transform')
            if not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-6):
                raise ValueError('Nonrigid body transform')
            capsules.append((matrix[:3,:3]@start+matrix[:3,3],matrix[:3,:3]@end+matrix[:3,3],radius))
        minimum=float('inf');nearest=None;clearances=[]
        for i,j in self.pairs:
            a,b=capsules[i],capsules[j]
            clearance=_segment_segment_distance(a[0],a[1],b[0],b[1])-a[2]-b[2]
            clearances.append(float(clearance))
            if clearance<minimum:
                minimum=float(clearance);nearest=[self.shapes[i][0],self.shapes[j][0]]
        result=dict(passed=minimum>=margin,minimum_clearance_m=minimum,required_margin_m=margin,
            nearest_pair=nearest,checked_pairs=len(self.pairs),capsules=len(self.shapes),
            unsupported_shapes=self.unsupported,whole_robot_self_collision_certified=not self.unsupported)
        if include_clearances: result['clearances_m']=clearances
        return result
