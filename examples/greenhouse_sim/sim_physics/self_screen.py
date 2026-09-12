"""Cached actual-capsule self screen; no collision filters are added or changed.

Honors the scene's existing disabled adjacent-joint pairs and explicit mounting
filters. Optional conservative tool boxes cover non-capsule collision shapes.
"""
import itertools
import numpy as np

from .capsule_surface import world_capsule


class SelfCapsuleScreen:
    def __init__(self,stage,robot_root,*,include_tool_boxes=False,fit_plate=False):
        from pxr import Usd,UsdGeom,UsdPhysics
        self.root=robot_root;self.shapes=[];self.unsupported=[];self.excluded=set();self.box_paths=[]
        cache=UsdGeom.XformCache()
        bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','guide','proxy'],False,True)
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
            body_path=str(body.GetPath());link=body_path[len(robot_root)+1:]
            if cap is not None:
                self.shapes.append((path,body_path,link,'capsule',cap));continue
            if not include_tool_boxes:
                self.unsupported.append(path);continue
            if fit_plate and path==robot_root+'/ee_right/attachments/DeleafKnife/BladePlateContact':
                from .tool_bounds import plate_box
                if (not prim.IsA(UsdGeom.Mesh) or
                        UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()!='convexHull'):
                    raise ValueError('Source convex blade plate required')
                value=plate_box(UsdGeom.Mesh(prim).GetPointsAttr().Get(),local)
                self.shapes.append((path,body_path,link,'box',value));self.box_paths.append(path)
                continue
            if not prim.IsA(UsdGeom.Boundable): raise ValueError('Unbounded tool collider: '+path)
            bound=bounds.ComputeUntransformedBound(prim).ComputeAlignedRange()
            if bound.IsEmpty(): raise ValueError('Empty tool collision bounds: '+path)
            low,high=np.asarray(bound.GetMin()),np.asarray(bound.GetMax())
            scale=np.linalg.norm(local[:3,:3],axis=0)
            if (not np.isfinite(np.r_[low,high,local.flatten()]).all()
                    or np.any(high<low) or np.any(scale<=0)):
                raise ValueError('Invalid tool bounds or transform')
            axes=local[:3,:3]/scale
            if not np.allclose(axes.T@axes,np.eye(3),atol=1e-6): raise ValueError('Sheared tool transform')
            centre=local[:3,:3]@((low+high)/2)+local[:3,3]
            self.shapes.append((path,body_path,link,'box',(centre,axes,scale*(high-low)/2)))
            self.box_paths.append(path)
        self.pairs=[]
        for i,j in itertools.combinations(range(len(self.shapes)),2):
            a,b=self.shapes[i],self.shapes[j]
            if a[1]==b[1] or any(frozenset((x,y)) in self.excluded for x in a[:2] for y in b[:2]): continue
            self.pairs.append((i,j))
        if not self.shapes or not self.pairs: raise ValueError('No self-collision shape coverage')

    def check(self,body_world,*,margin=.003,include_clearances=False):
        from greenhouse_sim.robot_kinematics import _segment_segment_distance,_segment_aabb_distance,_oriented_box_obb_separation
        if not np.isfinite(margin) or margin<0: raise ValueError('Invalid self-clearance margin')
        shapes=[];centres=[];radii=[]
        for path,body,link,kind,shape in self.shapes:
            matrix=np.asarray(body_world[link])
            if matrix.shape!=(4,4) or not np.isfinite(matrix).all(): raise ValueError('Invalid body transform')
            if (not np.allclose(matrix[:3,:3].T@matrix[:3,:3],np.eye(3),atol=1e-6)
                    or not np.allclose(matrix[3],[0,0,0,1]) or np.linalg.det(matrix[:3,:3])<0):
                raise ValueError('Nonrigid body transform')
            if kind=='capsule':
                start,end,radius=shape
                start=matrix[:3,:3]@start+matrix[:3,3];end=matrix[:3,:3]@end+matrix[:3,3]
                shapes.append((start,end,radius));centres.append((start+end)/2);radii.append(np.linalg.norm(end-start)/2+radius)
            else:
                centre,axes,half=shape;centre=matrix[:3,:3]@centre+matrix[:3,3];axes=matrix[:3,:3]@axes
                shapes.append((centre,axes,half));centres.append(centre);radii.append(np.linalg.norm(half))
        minimum=float('inf');nearest=None;clearances=[]
        for i,j in self.pairs:
            a,b=shapes[i],shapes[j];ka,kb=self.shapes[i][3],self.shapes[j][3]
            lower_bound=float(np.linalg.norm(centres[i]-centres[j])-radii[i]-radii[j])
            if lower_bound>=margin and not include_clearances:
                clearance=lower_bound  # conservative bound, not exact nearest distance
            elif ka==kb=='capsule':
                clearance=_segment_segment_distance(a[0],a[1],b[0],b[1])-a[2]-b[2]
            elif ka==kb=='box':
                clearance=_oriented_box_obb_separation(*a,*b)
            else:
                cap,box=(a,b) if ka=='capsule' else (b,a)
                centre,axes,half=box
                clearance=_segment_aabb_distance(axes.T@(cap[0]-centre),axes.T@(cap[1]-centre),half)-cap[2]
            clearances.append(float(clearance))
            if clearance<minimum:
                minimum=float(clearance);nearest=[self.shapes[i][0],self.shapes[j][0]]
        result=dict(passed=minimum>=margin,minimum_clearance_m=minimum,required_margin_m=margin,
            nearest_pair=nearest,checked_pairs=len(self.pairs),capsules=len(self.shapes)-len(self.box_paths),
            conservative_tool_boxes=self.box_paths,minimum_is_conservative_bound=True,
            unsupported_shapes=self.unsupported,whole_robot_self_collision_certified=False,
            all_shape_bounds_screened=not self.unsupported)
        if include_clearances: result['clearances_m']=clearances
        return result
