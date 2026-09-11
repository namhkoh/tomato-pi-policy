"""Cached held-petiole and local static collision geometry for right-arm planning.

Uses actual native body frames at planning time, not stale Fabric/USD poses.
Conservative tool boxes and convex leaf hulls are planning bounds, not new
contacts. Native guards remain mandatory for motion between path samples.
"""
import numpy as np


def collision_triangles(points,counts,indices):
    """Vectorized triangle/quad expansion; both quad diagonals stay covered."""
    points=np.asarray(points,float);counts=np.asarray(counts,int);indices=np.asarray(indices,int)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or not np.isfinite(points).all()
            or counts.ndim!=1 or indices.ndim!=1 or not np.isin(counts,[3,4]).all()
            or int(counts.sum())!=len(indices) or np.any(indices<0) or np.any(indices>=len(points))):
        raise ValueError('Invalid/unsupported collision mesh topology')
    starts=np.r_[0,np.cumsum(counts)[:-1]] if len(counts) else np.empty(0,int)
    triangles=indices[starts[counts==3,None]+np.arange(3)]
    quads=indices[starts[counts==4,None]+np.arange(4)]
    expanded=quads[:,np.array([[0,1,2],[0,2,3],[0,1,3],[1,2,3]])].reshape(-1,3)
    return points[np.concatenate((triangles,expanded))]


class TriangleIndex:
    """Conservative triangle-AABB lookup; narrow-phase tests remain unchanged."""
    def __init__(self,triangles):
        from scipy.spatial import cKDTree
        self.triangles=np.asarray(triangles,float)
        if self.triangles.ndim!=3 or self.triangles.shape[1:]!=(3,3) or not np.isfinite(self.triangles).all():
            raise ValueError('Finite collision triangles required')
        self.low=self.triangles.min(1);self.high=self.triangles.max(1)
        half=(self.high-self.low)/2
        self.radius=float(np.linalg.norm(half,axis=1).max()) if len(half) else 0.
        self.tree=cKDTree((self.low+self.high)/2) if len(half)>128 else None

    def query(self,low,high):
        low,high=np.asarray(low,float),np.asarray(high,float)
        if low.shape!=(3,) or high.shape!=(3,) or not np.isfinite([low,high]).all() or np.any(low>high):
            raise ValueError('Finite ordered triangle-query bounds required')
        indices=np.arange(len(self.low)) if self.tree is None else np.asarray(self.tree.query_ball_point(
            (low+high)/2,float(np.linalg.norm((high-low)/2))+self.radius+1e-9),int)
        keep=np.all(self.high[indices]>=low-1e-9,axis=1)&np.all(self.low[indices]<=high+1e-9,axis=1)
        return self.triangles[indices[keep]]


class HeldPlantScreen:
    def __init__(self,rig,robot_shapes,blade_path,*,arm='right',grasp_path=None):
        from pxr import Usd,UsdGeom,UsdPhysics
        from scipy.spatial import ConvexHull
        from .capsule_surface import world_capsule
        if arm not in ('left','right'): raise ValueError('Expected left or right arm')
        if grasp_path is not None and (arm!='left' or grasp_path not in rig.body_paths[rig.cut_index:]):
            raise ValueError('Expected one detachable left-grasp shaft')
        self.arm=arm
        prefixes=(f'link_{arm}_arm_',f'ee_{arm}',f'ee_finger_{arm[0]}')
        self.shapes=[s for s in robot_shapes if s[2].startswith(prefixes)]
        self.grasp_collider=None if grasp_path is None else grasp_path+'/StemCollider'
        index=rig.body_paths.index(grasp_path) if grasp_path is not None else None
        # The shaft is discretized into touching capsules. A real finger pad
        # may span an adjacent segment of the SAME detachable shaft. Never
        # include support-side segments, leaves or a separate branch.
        self.grasp_colliders=set() if index is None else {
            p+'/StemCollider' for p in rig.body_paths[max(rig.cut_index,index-1):index+2]}
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
        if not self.local or not self.shapes: raise ValueError('Missing held-plant/arm geometry')
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
                    indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int)
                    triangles=collision_triangles(points,counts,indices)
                    # Robot bounds (including margin) may never leave this
                    # workspace. Faces wholly outside cannot be touched.
                    near=np.all(triangles.max(1)>=self.workspace[0]-1e-9,axis=1)&np.all(triangles.min(1)<=self.workspace[1]+1e-9,axis=1)
                    triangles=triangles[near]
                    records.append((path,'triangles',(triangles,None),low,high));continue
            # Unknown cooked approximations and primitive shapes retain their
            # enclosing WORLD box. Never silently ignore an unsupported shape.
            records.append((path,'box',((low+high)/2,np.eye(3),(high-low)/2),low,high))
        self.static=records
        self.static_indices={p:TriangleIndex(data[0]) for p,kind,data,_,_ in records if kind in ('hull','triangles')}

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
        self.triangle_indices={}
        for path,kind,data,_,_ in result:
            if kind in ('hull','triangles'):
                cached=getattr(self,'static_indices',{}).get(path)
                self.triangle_indices[path]=cached if cached is not None and cached.triangles is data[0] else TriangleIndex(data[0])
        self.lower=np.array([v[3] for v in result]);self.upper=np.array([v[4] for v in result])

    def check(self,body_world,*,stroke=False,grasp=False,margin=.001):
        from greenhouse_sim.robot_kinematics import _segment_segment_distance,_segment_aabb_distance
        from sim_data.capture_viewpoints import triangles_intersect_box
        from .capsule_surface import segment_triangles_distance
        if not np.isfinite(margin) or margin<.001: raise ValueError('Held-plant margin must be at least 1 mm')
        if grasp and (self.arm!='left' or self.grasp_collider is None):
            raise ValueError('Grasp screening requires a selected left-grasp shaft')
        self.last_failure=None
        for path,body,link,kind,data in self.shapes:
            matrix=np.asarray(body_world[link],float);r,t=matrix[:3,:3],matrix[:3,3]
            if (matrix.shape!=(4,4) or not np.isfinite(matrix).all()
                    or not np.allclose(r.T@r,np.eye(3),atol=1e-5) or np.linalg.det(r)<0):
                raise ValueError('Invalid arm body transform')
            if kind=='capsule':
                a,b,radius=data;a=r@a+t;b=r@b+t
                low=np.minimum(a,b)-radius;high=np.maximum(a,b)+radius
            else:
                centre,axes,half=data;centre=r@centre+t;axes=r@axes
                extent=np.abs(axes)@half;low=centre-extent;high=centre+extent
            if self.workspace is not None and (np.any(low-margin<self.workspace[0]) or np.any(high+margin>self.workspace[1])):
                self.last_failure=dict(robot_collider=path,reason=self.arm+'_shape_left_cached_scene_region')
                return False
            near=np.flatnonzero(np.all(self.upper>=low-margin,axis=1)&np.all(self.lower<=high+margin,axis=1))
            for j in near:
                other,okind,odata,_,_=self.obstacles[j]
                # Expected seam loading is still edge/force/direction qualified
                # by native callbacks. No collision filtering is changed here.
                if stroke and path==self.blade_path and other in self.seam_paths: continue
                # Only the two fingers may intentionally touch the selected
                # shaft and its contiguous discretization neighbors. Leaves,
                # support, other branches, palm and camera stay checked.
                # This is a planning allowance, NOT a native collision filter
                # or evidence that contact/grasp has actually occurred.
                if grasp and link in ('ee_finger_l1','ee_finger_l2') and other in self.grasp_colliders: continue
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
                    triangles=self.triangle_indices[other].query(low-margin,high+margin)
                    if kind=='capsule':
                        inside=lambda p:equations is not None and np.all(equations[:,:3]@p+equations[:,3]<=1e-9)
                        hit=inside(a) or inside(b) or segment_triangles_distance(a,b,triangles)<=radius+margin
                    else:
                        local=(triangles-centre)@axes
                        hit=((equations is not None and np.all(equations[:,:3]@centre+equations[:,3]<=1e-9))
                             or triangles_intersect_box(local,-half-margin,half+margin))
                if hit:
                    # Optional, same-tick native static-actor refinement. Keep
                    # the entire proposed tool box and margin; dynamic target,
                    # capsules, source triangles and all unverified paths keep
                    # their original checks. A query error can never clear one.
                    native=getattr(self,'native_static_query',None)
                    fitted_tool=(self.arm=='right' and link=='ee_right' and body.endswith('/ee_right')
                        and path.startswith(tuple(body+'/attachments/'+name+'/'
                            for name in ('RightWristCamera','DeleafKnife'))))
                    if (fitted_tool and kind=='box' and okind=='box' and native is not None
                            and native.clear_box_checked(other,centre,axes,half,margin)):
                        continue
                    self.last_failure=dict(robot_collider=path,plant_collider=other,margin_m=margin,
                        phase='grasp' if grasp else 'stroke' if stroke else 'transit',conservative_overlap=True)
                    return False
        return True
