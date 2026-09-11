"""Source-derived knife contact partitions; never alter the supplied visible STL.

The mounting end is thicker than the long plate. A single enclosing box fills
empty space above the plate and puts the old semantic edge in that empty space.
Split the source triangles into two convex contact carriers at the thickness
transition, and derive the long leading strip from actual plate cross-sections.
"""
import numpy as np


def section_points(triangles,y):
    a=np.asarray(triangles,float).reshape(-1,3)
    b=np.roll(np.asarray(triangles,float),-1,axis=1).reshape(-1,3)
    mask=(a[:,1]-y)*(b[:,1]-y)<0
    a,b=a[mask],b[mask]
    if not len(a): raise ValueError('Empty knife cross-section')
    return a+(b-a)*((y-a[:,1])/(b[:,1]-a[:,1]))[:,None]


def plate_profile(points,triangles):
    points=np.asarray(points,float);triangles=np.asarray(triangles,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or triangles.ndim!=3
            or triangles.shape[1:]!=(3,3) or not np.isfinite(points).all()
            or not np.isfinite(triangles).all()):
        raise ValueError('Invalid source knife triangles')
    lo,hi=points.min(0),points.max(0)
    ys=np.linspace(lo[1]+.2*(hi[1]-lo[1]),lo[1]+.65*(hi[1]-lo[1]),5)
    sections=[section_points(triangles,y) for y in ys]
    zlow=np.array([s[:,2].min() for s in sections]);zhigh=np.array([s[:,2].max() for s in sections])
    if np.ptp(zlow)>1e-5 or np.ptp(zhigh)>1e-5: raise ValueError('Knife has no verified constant-thickness long plate')
    bottom,top=float(zlow.mean()),float(zhigh.mean())
    thicker=points[(points[:,2]<bottom-1e-6)|(points[:,2]>top+1e-6)]
    if not len(thicker) or top<=bottom: raise ValueError('Unknown plate/mount layout')
    split=float(thicker[:,1].min()-.0005)
    if split<=ys[-1] or split>=hi[1]-.002: raise ValueError('Nonseparable blade mounting end')
    # Long -X side is slightly slanted in the supplied CAD, not axis-aligned.
    xfront=np.array([s[:,0].min() for s in sections])
    slope,intercept=np.polyfit(ys,xfront,1)
    if np.max(np.abs(xfront-(slope*ys+intercept)))>.0001:
        raise ValueError('Leading plate side is not a straight edge')
    x=np.array([1.,-slope,0.]);x/=np.linalg.norm(x)
    y=np.array([slope,1.,0.]);y/=np.linalg.norm(y)
    # Exclude the rounded distal end and the thicker mounting end.
    low_y=lo[1]+.005;high_y=split-.005
    if high_y-low_y<.02: raise ValueError('Insufficient usable long cutting side')
    mid_y=(low_y+high_y)/2
    edge=np.eye(4);edge[:3,:3]=np.column_stack([x,y,[0,0,1]])
    edge[:3,3]=[slope*mid_y+intercept,mid_y,(top+bottom)/2]
    edge[:3,3]+=.001*x
    size=np.array([.002,(high_y-low_y)*np.sqrt(1+slope*slope),top-bottom])
    return dict(split_y_m=split,plate_z_range_m=[bottom,top],
        original_box_z_range_m=[float(lo[2]),float(hi[2])],edge_frame=edge,edge_size_m=size)


def refine_blade_contacts(stage,robot_root):
    from pxr import Usd,UsdGeom,UsdPhysics,Vt
    from .plant import matrix_attr,_collision
    from .mechanics import clip_mesh
    root=robot_root+'/ee_right/attachments/DeleafKnife'
    blade=UsdGeom.Mesh.Get(stage,root+'/Blade')
    if not blade: raise ValueError('Missing original blade mesh')
    cache=UsdGeom.XformCache()
    matrix=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root))).T)@np.asarray(cache.GetLocalToWorldTransform(blade.GetPrim())).T
    points=np.asarray(blade.GetPointsAttr().Get(),float)@matrix[:3,:3].T+matrix[:3,3]
    counts=np.asarray(blade.GetFaceVertexCountsAttr().Get(),int)
    indices=np.asarray(blade.GetFaceVertexIndicesAttr().Get(),int)
    if not np.all(counts==3): raise ValueError('Triangular original knife required')
    profile=plate_profile(points,points[indices.reshape(-1,3)])
    paths=[]
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        # Disable only the superseded enclosing box; replace ALL source surface
        # triangles, including the thicker mounting end, with contact hulls.
        old=stage.GetPrimAtPath(root+'/BladeCollision')
        UsdPhysics.CollisionAPI(old).CreateCollisionEnabledAttr(False)
        for positive,name in ((False,'BladePlateContact'),(True,'BladeMountContact')):
            vertices,_=clip_mesh(points,counts,indices,[0,profile['split_y_m'],0],[0,1,0],positive=positive)
            if not len(vertices): raise ValueError('Empty source blade partition')
            path=root+'/'+name;mesh=UsdGeom.Mesh.Define(stage,path)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(vertices)//3,3,dtype=np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(len(vertices),dtype=np.int32)))
            mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([vertices.min(0),vertices.max(0)],dtype=np.float32)))
            mesh.CreatePurposeAttr('guide');mesh.CreateVisibilityAttr('invisible')
            _collision(mesh.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
            paths.append(path)
        edge=UsdGeom.Cube.Get(stage,root+'/CuttingEdge')
        transform=profile['edge_frame'].copy()
        transform[:3,:3]*=profile['edge_size_m']/float(edge.GetSizeAttr().Get())
        matrix_attr(edge.GetPrim(),transform)
    return dict(model='source_triangle_plate_and_mount_convex_hulls',collider_paths=paths,
        source_visual_edited=False,source_asset_edited=False,
        **{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in profile.items()})
