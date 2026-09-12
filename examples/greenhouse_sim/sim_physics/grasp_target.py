"""Ground-truth diagnostic grasp identity and actual finger/seam clearance.

These are privileged fixture annotations, never perception or training approval.
"""
import itertools
import numpy as np


def finger_seam_clearance(stage,robot_root,palm_goal,seam,axis,*,minimum=.01):
    """Project the original complete finger bounds onto the actual cut axis.

    The whole pad must be distal, with 10 mm from the cut plane. This is a
    placement screen, not proof of a collision-free knife approach or grasp.
    """
    from pxr import Usd,UsdGeom,UsdPhysics
    cache=UsdGeom.XformCache()
    bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','guide','proxy'],False,True)
    wrist=stage.GetPrimAtPath(robot_root+'/ee_left')
    inverse=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(wrist)).T)
    axis=np.asarray(axis,dtype=float)
    if (axis.shape!=(3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-9
            or not np.isfinite(minimum) or minimum<0): raise ValueError('Invalid cut axis or clearance')
    axis=axis/np.linalg.norm(axis)
    distances=[];paths=[]
    for name in ('ee_finger_l1','ee_finger_l2'):
        for prim in Usd.PrimRange(stage.GetPrimAtPath(robot_root+'/'+name)):
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(): continue
            bound=bounds.ComputeUntransformedBound(prim).ComputeAlignedRange()
            if bound.IsEmpty(): raise ValueError('Empty finger collision bounds')
            low,high=np.asarray(bound.GetMin()),np.asarray(bound.GetMax())
            corners=np.array(list(itertools.product(*zip(low,high))))
            matrix=palm_goal@inverse@np.asarray(cache.GetLocalToWorldTransform(prim)).T
            points=corners@matrix[:3,:3].T+matrix[:3,3]
            distances.extend(((points-seam)@axis).tolist());paths.append(str(prim.GetPath()))
    if not distances or not np.isfinite(distances).all(): raise ValueError('Invalid finger/seam geometry')
    clearance=float(min(distances))
    if clearance<minimum: raise ValueError(f'Grasp overlaps protected cut corridor: finger clearance {clearance:.6f} m < {minimum:.6f} m')
    return dict(minimum_finger_to_cut_plane_m=clearance,required_clearance_m=minimum,
        finger_colliders=paths,whole_tool_corridor_certified=False)


class TargetMarkers:
    """Optional diagnostic spheres; update only on render, no collisions.

    Default render purpose makes the opt-in guides visible without globally
    displaying hidden physics proxies. This is a non-training GUI only.
    """
    def __init__(self,stage,rig,grasp_index,*,grasp_offset_m=0.):
        from pxr import Sdf,Usd,UsdGeom
        self.stage=stage;self.rig=rig;self.grasp_index=grasp_index;self.visible=False;self.ops={}
        if not np.isfinite(grasp_offset_m):raise ValueError('Finite material-point offset required')
        self.grasp_offset_m=float(grasp_offset_m)
        with Usd.EditContext(stage,stage.GetSessionLayer()):
            self.root=UsdGeom.Xform.Define(stage,'/World/DiagnosticTargetMarkers')
            self.root.GetPrim().CreateAttribute('tomato:diagnosticOnly',Sdf.ValueTypeNames.Bool).Set(True)
            self.root.CreatePurposeAttr('default')
            for name,color in (('Attachment',(1.,.75,0.)),('Cut',(1.,1.,1.)),('Grasp',(0.,1.,1.))):
                sphere=UsdGeom.Sphere.Define(stage,str(self.root.GetPath())+'/'+name)
                sphere.CreateRadiusAttr(.003);sphere.CreateDisplayColorAttr([color])
                self.ops[name]=sphere.AddTranslateOp()
            self.root.MakeInvisible()
            self.update(rig.rest_frames)

    def toggle(self):
        from pxr import Usd
        self.visible=not self.visible
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            self.root.MakeVisible() if self.visible else self.root.MakeInvisible()

    def update(self,frames):
        from pxr import Gf,Usd
        i=self.rig.cut_index
        half=np.linalg.norm(self.rig.chain_world[i+1]-self.rig.chain_world[i])/2
        points=dict(Attachment=self.rig.chain_world[0],
            Cut=frames[i,:3,3]-half*frames[i,:3,2],
            Grasp=frames[self.grasp_index,:3,3]+self.grasp_offset_m*frames[self.grasp_index,:3,2])
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            for name,point in points.items(): self.ops[name].Set(Gf.Vec3d(*point))
