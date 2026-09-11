"""Cached held-petiole and local static collision geometry for right-arm planning.

Uses actual native body frames at planning time, not stale Fabric/USD poses.
Conservative tool boxes and convex leaf hulls are planning bounds, not new
contacts. Native guards remain mandatory for motion between path samples.
"""
import numpy as np


class HeldPlantScreen:
    def __init__(self,rig,robot_shapes,blade_path):
        from pxr import Usd,UsdGeom,UsdPhysics
        from scipy.spatial import ConvexHull
        from .capsule_surface import world_capsule
        self.shapes=[s for s in robot_shapes if s[2].startswith(('link_right_arm_','ee_right','ee_finger_r'))]
        self.local=[];self.blade_path=blade_path;self.last_failure=None
        self.seam_paths={rig.body_paths[i]+'/StemCollider' for i in (rig.cut_index-1,rig.cut_index)}
        cache=UsdGeom.XformCache()
        for i,path in enumerate(rig.body_paths):
            inverse=np.linalg.inv(rig.rest_frames[i])
            for prim in Usd.PrimRange(rig.stage.GetPrimAtPath(path)):
                if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(): continue
                matrix=inverse@np.asarray(cache.GetLocalToWorldTransform(prim)).T
                capsule=world_capsule(prim,matrix)
                if capsule is not None:
                    self.local.append((str(prim.GetPath()),i,'capsule',capsule));continue
                if (not prim.IsA(UsdGeom.Mesh)
                        or UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()!='convexHull'):
                    raise ValueError('Unsupported held-plant collision shape: '+str(prim.GetPath()))
                points=np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get(),float)
                points=points@matrix[:3,:3].T+matrix[:3,3]
                hull=ConvexHull(points)  # fail closed on a degenerate/unknown hull
                self.local.append((str(prim.GetPath()),i,'hull',(points[hull.vertices],points[hull.simplices],hull.equations)))
        if not self.local or not self.shapes: raise ValueError('Missing held-plant/right-arm geometry')
        self.static=[];self.workspace=None

    def include_static_scene(self,stage,robot_root,target_root,centre):
        """Cache active contacts in a bounded 2 m cube around the fixed shoulder.

        Every subsequent right-arm shape must remain inside this region; a
        detour cannot escape it into unchecked geometry. Traverses hidden and
        instanced colliders. No stage traversal takes place in a physics tick.
        """
        from pxr import Usd,UsdGeom,UsdPhysics
        from scipy.spatial import ConvexHull
        from .capsule_surface import world_capsule
        centre=np.asarray(centre,float)
        if centre.shape!=(3,) or not np.isfinite(centre).all(): raise ValueError('Invalid local scene centre')
        self.workspace=(centre-1.,centre+1.)
        cache=UsdGeom.XformCache()
        bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','guide','proxy'],False,True)
        records=[]
        for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
            path=str(prim.GetPath())
            if path.startswith((robot_root+'/',target_root+'/')): continue
            if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(): continue
            if not prim.IsA(UsdGeom.Boundable): raise ValueError('Unbounded scene collider: '+path)
            bound=bounds.ComputeWorldBound(prim).ComputeAlignedRange()
            low,high=np.asarray(bound.GetMin()),np.asarray(bound.GetMax())
            if bound.IsEmpty() or not np.isfinite([low,high]).all(): raise ValueError('Invalid scene bounds: '+path)
            if np.any(high<self.workspace[0]) or np.any(low>self.workspace[1]): continue
            body=prim
            while body and not body.IsPseudoRoot():
                if body.HasAPI(UsdPhysics.RigidBodyAPI):
                    api=UsdPhysics.RigidBodyAPI(body)
                    if api.GetRigidBodyEnabledAttr().Get() and not api.GetKinematicEnabledAttr().Get():
                        raise ValueError('Untracked dynamic obstacle: '+str(body.GetPath()))
                    break
                body=body.GetParent()
            matrix=np.asarray(cache.GetLocalToWorldTransform(prim)).T
            cap=world_capsule(prim,matrix)
            if cap is not None:
                records.append((path,'capsule',cap,low,high));continue
            if prim.IsA(UsdGeom.Mesh):
                mesh=UsdGeom.Mesh(prim);points=np.asarray(mesh.GetPointsAttr().Get(),float)
                points=points@matrix[:3,:3].T+matrix[:3,3]
                approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
                if approximation=='convexHull':
                    hull=ConvexHull(points)
                    records.append((path,'hull',(points[hull.simplices],hull.equations),low,high));continue
                if approximation in (None,'none'):
                    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int)
                    indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int);cursor=0;faces=[]
                    for count in counts:
                        face=indices[cursor:cursor+count];cursor+=count
                        if count not in (3,4): raise ValueError('Unsupported collision polygon: '+path)
                        faces.extend([face] if count==3 else [face[list(x)] for x in ((0,1,2),(0,2,3),(0,1,3),(1,2,3))])
                    triangles=points[np.asarray(faces,int)]
                    records.append((path,'triangles',(triangles,None),low,high));continue
            # Unknown cooked approximations and primitive shapes retain their
            # enclosing WORLD box. Never silently ignore an unsupported shape.
            records.append((path,'box',((low+high)/2,np.eye(3),(high-low)/2),low,high))
        self.static=records

    def snapshot(self,frames):
        frames=np.asarray(frames,float)
        if frames.ndim!=3 or frames.shape[1:]!=(4,4) or not np.isfinite(frames).all():
            raise ValueError('Finite native body frames required')
        result=list(self.static)
        for path,i,kind,data in self.local:
            r,t=frames[i,:3,:3],frames[i,:3,3]
            if not np.allclose(r.T@r,np.eye(3),atol=1e-5) or np.linalg.det(r)<0:
                raise ValueError('Nonrigid native plant frame')
            if kind=='capsule':
                a,b,radius=data;a=r@a+t;b=r@b+t
                value=(a,b,radius);low=np.minimum(a,b)-radius;high=np.maximum(a,b)+radius
            else:
                vertices,triangles,equations=data
                vertices=vertices@r.T+t;triangles=triangles@r.T+t
                normals=equations[:,:3]@r.T
                equations=np.column_stack([normals,equations[:,3]-normals@t])
                value=(triangles,equations);low=vertices.min(0);high=vertices.max(0)
            result.append((path,kind,value,low,high))
        self.obstacles=result
        self.lower=np.array([v[3] for v in result]);self.upper=np.array([v[4] for v in result])

    def check(self,body_world,*,stroke=False,margin=.001):
        from greenhouse_sim.robot_kinematics import _segment_segment_distance,_segment_aabb_distance
        from sim_data.capture_viewpoints import triangles_intersect_box
        from .capsule_surface import segment_triangles_distance
        if not np.isfinite(margin) or margin<.001: raise ValueError('Held-plant margin must be at least 1 mm')
        self.last_failure=None
        for path,body,link,kind,data in self.shapes:
            matrix=np.asarray(body_world[link],float);r,t=matrix[:3,:3],matrix[:3,3]
            if (matrix.shape!=(4,4) or not np.isfinite(matrix).all()
                    or not np.allclose(r.T@r,np.eye(3),atol=1e-5) or np.linalg.det(r)<0):
                raise ValueError('Invalid right-arm body transform')
            if kind=='capsule':
                a,b,radius=data;a=r@a+t;b=r@b+t
                low=np.minimum(a,b)-radius;high=np.maximum(a,b)+radius
            else:
                centre,axes,half=data;centre=r@centre+t;axes=r@axes
                extent=np.abs(axes)@half;low=centre-extent;high=centre+extent
            if self.workspace is not None and (np.any(low-margin<self.workspace[0]) or np.any(high+margin>self.workspace[1])):
                self.last_failure=dict(robot_collider=path,reason='right_shape_left_cached_scene_region')
                return False
            near=np.flatnonzero(np.all(self.upper>=low-margin,axis=1)&np.all(self.lower<=high+margin,axis=1))
            for j in near:
                other,okind,odata,_,_=self.obstacles[j]
                # Expected seam loading is still edge/force/direction qualified
                # by native callbacks. No collision filtering is changed here.
                if stroke and path==self.blade_path and other in self.seam_paths: continue
                if okind=='capsule':
                    oa,ob,orr=odata
                    if kind=='capsule': hit=_segment_segment_distance(a,b,oa,ob)<=radius+orr+margin
                    else: hit=_segment_aabb_distance(axes.T@(oa-centre),axes.T@(ob-centre),half)<=orr+margin
                elif okind=='box':
                    from greenhouse_sim.robot_kinematics import _oriented_box_obb_separation
                    oc,oa,oh=odata
                    if kind=='capsule': hit=_segment_aabb_distance(oa.T@(a-oc),oa.T@(b-oc),oh)<=radius+margin
                    else: hit=_oriented_box_obb_separation(centre,axes,half,oc,oa,oh)<=margin
                else:
                    triangles,equations=odata
                    if kind=='capsule':
                        inside=lambda p:equations is not None and np.all(equations[:,:3]@p+equations[:,3]<=1e-9)
                        hit=inside(a) or inside(b) or segment_triangles_distance(a,b,triangles)<=radius+margin
                    else:
                        local=(triangles-centre)@axes
                        hit=((equations is not None and np.all(equations[:,:3]@centre+equations[:,3]<=1e-9))
                             or triangles_intersect_box(local,-half-margin,half+margin))
                if hit:
                    self.last_failure=dict(robot_collider=path,plant_collider=other,margin_m=margin,
                        phase='stroke' if stroke else 'transit',conservative_overlap=True)
                    return False
        return True
