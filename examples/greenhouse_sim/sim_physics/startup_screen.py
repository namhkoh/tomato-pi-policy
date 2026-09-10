"""Reject static spawn overlaps before starting native robot dynamics.

Uses collision shapes, including hidden proxies and instance proxies. Robot
local boxes enclose actual shapes; a pass is only a conservative static screen,
not a certification of the subsequent trajectory or robot self collision.
"""
import numpy as np


def screen(stage,robot):
    from pxr import Usd,UsdGeom,UsdPhysics
    from sim_data.capture_viewpoints import triangles_intersect_box
    from .capsule_surface import world_capsule,segment_triangles_distance
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','guide','proxy'],False,True)
    transforms=UsdGeom.XformCache()
    robots=[];obstacles=[];hits=[];tested=0
    for prim in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get(): continue
        if not prim.IsA(UsdGeom.Boundable): raise ValueError('Unbounded collision prim '+str(prim.GetPath()))
        path=str(prim.GetPath());box=cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if box.IsEmpty(): raise ValueError('Empty collision bounds '+path)
        record=(prim,path,np.asarray(box.GetMin()),np.asarray(box.GetMax()))
        (robots if path.startswith(robot.root+'/') else obstacles).append(record)
    if not robots or not obstacles: raise ValueError('Missing robot or scene collision geometry')
    lower=np.array([r[2] for r in robots]);upper=np.array([r[3] for r in robots])
    for prim,path,low,high in obstacles:
        nearby=np.flatnonzero(np.all(upper>low+1e-6,axis=1)&np.all(lower<high-1e-6,axis=1))
        if not len(nearby): continue
        triangles=None;hull_equations=None
        approximation=UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() if prim.HasAPI(UsdPhysics.MeshCollisionAPI) else 'none'
        if prim.IsA(UsdGeom.Mesh) and approximation in ('none','convexHull'):
            mesh=UsdGeom.Mesh(prim);counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int)
            indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int)
            if np.isin(counts,[3,4]).all() and len(indices)==int(counts.sum()):
                points=np.asarray(mesh.GetPointsAttr().Get(),float)
                matrix=np.asarray(transforms.GetLocalToWorldTransform(prim)).T
                points=points@matrix[:3,:3].T+matrix[:3,3]
                if approximation=='convexHull':
                    from scipy.spatial import ConvexHull
                    hull=ConvexHull(points)
                    faces=hull.simplices;hull_equations=hull.equations
                else:
                    starts=np.r_[0,np.cumsum(counts)[:-1]]
                    tris=indices[starts[counts==3,None]+np.arange(3)]
                    quads=indices[starts[counts==4,None]+np.arange(4)]
                    # Both diagonal choices preserve a conservative superset.
                    faces=np.concatenate((tris,*(quads[:,choice] for choice in ([0,1,2],[0,2,3],[0,1,3],[1,2,3]))))
                triangles=points[faces]
        for i in nearby:
            rp,rpath,rlo,rhi=robots[i]
            body=robot.root+'/'+rpath[len(robot.root)+1:].split('/')[0]
            if robot.floor_root and (path==robot.floor_root or path.startswith(robot.floor_root+'/')) and body in {
                    robot.root+'/base',robot.root+'/wheel_l',robot.root+'/wheel_r'}: continue
            tested+=1;overlap=True
            if triangles is not None:
                near=np.all(triangles.max(axis=1)>=rlo,axis=1)&np.all(triangles.min(axis=1)<=rhi,axis=1)
                overlap=False
                if near.any():
                    inv=np.linalg.inv(np.asarray(transforms.GetLocalToWorldTransform(rp)).T)
                    local=triangles[near]@inv[:3,:3].T+inv[:3,3]
                    bounds=cache.ComputeUntransformedBound(rp).ComputeAlignedRange()
                    overlap=triangles_intersect_box(local,np.asarray(bounds.GetMin()),np.asarray(bounds.GetMax()))
                    capsule=world_capsule(rp,np.asarray(transforms.GetLocalToWorldTransform(rp)).T)
                    if overlap and capsule is not None:
                        start,end,radius=capsule
                        # Retain 1 mm conservatism beyond the actual capsule;
                        # this clears empty box corners, not physical contacts.
                        overlap=segment_triangles_distance(start,end,triangles[near])<=radius+.001
                if hull_equations is not None:
                    # Surface-only tests must not clear a robot box fully
                    # enclosed in a solid convex leaf collision shape.
                    centre=(rlo+rhi)/2
                    overlap=overlap or bool(np.all(hull_equations[:,:3]@centre+hull_equations[:,3]<=1e-8))
            if overlap: hits.append(dict(robot_collider=rpath,scene_collider=path))
    return dict(passed=not hits,possible_overlap_count=len(hits),possible_overlaps=hits[:30],
        robot_collision_shapes=len(robots),scene_collision_shapes=len(obstacles),tested_broad_pairs=tested,
        method='collision_boxes_with_triangle_and_uniform_capsule_surface_refinement',
        whole_path_certified=False,self_collision_certified=False)
