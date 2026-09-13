"""Explicit source-surface convex arc partitions shared by native and planner.

The previous native convex decomposition was screened as one enclosing box,
filling the large open window. Fixed spatial partitions retain every source
triangle, preserve the visible tool and expose bounded carriers to both paths.
"""
import numpy as np


def partitions(points,counts,indices):
    from .mechanics import clip_mesh
    points=np.asarray(points,float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all(): raise ValueError('Invalid source arc')
    # Source arc: thin in X, 62 mm along Y, 51 mm in Z. Do not generalize to
    # unknown CAD silently. Up to 14 partitions, with no part omitted.
    low,high=points.min(0),points.max(0)
    if not (.004<high[0]-low[0]<.008 and .05<high[1]-low[1]<.08 and .04<high[2]-low[2]<.06):
        raise ValueError('Unknown source arc dimensions')
    edges=np.linspace(low[1]-.000001,high[1]+.000001,8)
    pieces=[]
    for a,b in zip(edges[:-1],edges[1:]):
        for lower_z,upper_z in ((low[2]-.000001,.020),(.020,high[2]+.000001)):
            vertices=points;cs=counts;ids=indices
            for origin,normal,positive in (([0,a,0],[0,1,0],True),([0,b,0],[0,1,0],False),
                                           ([0,0,lower_z],[0,0,1],True),([0,0,upper_z],[0,0,1],False)):
                vertices,_=clip_mesh(vertices,cs,ids,origin,normal,positive=positive)
                if not len(vertices): break
                cs=np.full(len(vertices)//3,3,dtype=int);ids=np.arange(len(vertices))
            if len(vertices): pieces.append(vertices)
    if not pieces: raise ValueError('Empty arc partitions')
    return pieces


def refine_arc_contacts(stage,robot_root,*,crossbar_edge=False):
    from pxr import Sdf,Usd,UsdGeom,UsdPhysics,Vt
    from .plant import _collision
    root=robot_root+'/ee_right/attachments/DeleafKnife'
    if type(crossbar_edge) is not bool:raise ValueError('Explicit crossbar edge mode required')
    if stage.GetPrimAtPath(root+'/CrossbarContact'):
        raise ValueError('Crossbar configuration already authored; rebuild the diagnostic stage')
    if crossbar_edge and stage.GetPrimAtPath(root+'/ArcContacts'):
        raise ValueError('Select crossbar geometry before authoring arc contacts')
    source=UsdGeom.Mesh.Get(stage,root+'/Arc')
    if not source: raise ValueError('Missing supplied arc')
    cache=UsdGeom.XformCache()
    matrix=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root))).T)@np.asarray(cache.GetLocalToWorldTransform(source.GetPrim())).T
    p=np.asarray(source.GetPointsAttr().Get(),float)@matrix[:3,:3].T+matrix[:3,3]
    counts=np.asarray(source.GetFaceVertexCountsAttr().Get(),int);indices=np.asarray(source.GetFaceVertexIndicesAttr().Get(),int)
    profile=None
    if crossbar_edge:
        from .crossbar_contacts import geometry
        profile=geometry(p,counts,indices);pieces=profile['support']
    else:pieces=partitions(p,counts,indices)
    paths=[]
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        group=UsdGeom.Scope.Define(stage,root+'/ArcContacts').GetPrim()
        group.CreateAttribute('tomato:contactPartCount',Sdf.ValueTypeNames.Int,custom=True).Set(len(pieces))
        UsdPhysics.CollisionAPI(stage.GetPrimAtPath(root+'/ArcCollision')).CreateCollisionEnabledAttr(False)
        named=[(root+f'/ArcContacts/Part_{i:02d}',vertices) for i,vertices in enumerate(pieces)]
        if profile is not None:named.append((root+'/CrossbarContact',profile['bar']))
        for path,vertices in named:
            mesh=UsdGeom.Mesh.Define(stage,path)
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
            mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(vertices)//3,3,dtype=np.int32)))
            mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(len(vertices),dtype=np.int32)))
            mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([vertices.min(0),vertices.max(0)],dtype=np.float32)))
            mesh.CreatePurposeAttr('guide');mesh.CreateVisibilityAttr('invisible')
            _collision(mesh.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
            paths.append(path)
        if profile is not None:
            from .blade_contacts import CROSSBAR_EDGE
            from .plant import matrix_attr
            edge=UsdGeom.Cube.Get(stage,root+'/CuttingEdge')
            transform=profile['edge_frame'].copy()
            transform[:3,:3]*=profile['edge_size_m']/float(edge.GetSizeAttr().Get())
            matrix_attr(edge.GetPrim(),transform)
            edge.GetPrim().CreateAttribute('tomato:edgeMode',Sdf.ValueTypeNames.Token).Set(CROSSBAR_EDGE)
    result=dict(model='source_triangle_arc_spatial_convex_partitions',collider_paths=paths,
        source_visual_edited=False,source_asset_edited=False)
    if profile is not None:
        result.update(model='source_crossbar_and_curved_support_convex_partitions',
            **{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in profile.items() if k not in ('support','bar')})
    return result
