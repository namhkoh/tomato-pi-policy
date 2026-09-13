"""Opt-in source-complete wrist bracket partitions shared by native/planner.

Replace opaque convexDecomposition carriers, not visible hardware. Clipping
retains every original triangle. Each small convex hull stays inside its cell;
these are still engineering contact approximations, not CAD fit certification.
"""
import itertools
import numpy as np


def partitions(points,counts,indices):
    from .mechanics import clip_mesh
    p=np.asarray(points,float);counts=np.asarray(counts,int);indices=np.asarray(indices,int)
    if (p.ndim!=2 or p.shape[1:]!=(3,) or not np.isfinite(p).all() or not np.all(counts==3)
            or counts.sum()!=len(indices) or np.any(indices<0) or np.any(indices>=len(p))):
        raise ValueError('Finite original triangular wrist mesh required')
    lo,hi=p.min(0),p.max(0);extent=hi-lo
    if np.any(extent<=0) or np.max(extent)>.2:raise ValueError('Bounded nondegenerate wrist hardware required')
    axes=np.argsort(extent)[-2:]
    # Eight cells along the two long source axes, including all mounting ends.
    edges=[np.linspace(lo[axis]-1e-7,hi[axis]+1e-7,n+1) for axis,n in zip(axes,(2,4),strict=True)]
    pieces=[]
    for i,j in itertools.product(range(2),range(4)):
        vertices=p;cs=counts;ids=indices
        for axis,grid,index in zip(axes,edges,(i,j),strict=True):
            for bound,positive in ((grid[index],True),(grid[index+1],False)):
                origin=np.zeros(3);origin[axis]=bound;normal=np.eye(3)[axis]
                vertices,_=clip_mesh(vertices,cs,ids,origin,normal,positive=positive)
                if not len(vertices):break
                cs=np.full(len(vertices)//3,3,dtype=int);ids=np.arange(len(vertices))
            if not len(vertices):break
        if len(vertices):
            if np.linalg.matrix_rank(vertices-vertices.mean(0),tol=1e-9)<3:
                raise ValueError('Degenerate wrist partition cannot silently lose a surface')
            pieces.append(vertices)
    if not pieces:raise ValueError('Empty wrist partitions')
    return pieces


def refine(stage,robot_root):
    from pxr import Usd,UsdGeom,UsdPhysics,Vt
    from .plant import _collision,matrix_attr
    old_paths=[];new_paths=[]
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        for side in ('left','right'):
            assembly=robot_root+f'/ee_{side}/attachments/{side.title()}WristCamera'
            for name in ('BracketCollision','AdapterCollision'):
                path=assembly+'/'+name;source=UsdGeom.Mesh.Get(stage,path)
                if (not source or not UsdPhysics.CollisionAPI(source).GetCollisionEnabledAttr().Get()
                        or UsdPhysics.MeshCollisionAPI(source).GetApproximationAttr().Get()!='convexDecomposition'):
                    raise ValueError('Original active decomposed wrist bracket required')
                p=np.asarray(source.GetPointsAttr().Get(),float)
                pieces=partitions(p,source.GetFaceVertexCountsAttr().Get(),source.GetFaceVertexIndicesAttr().Get())
                matrix=np.asarray(UsdGeom.Xformable(source).GetLocalTransformation()).T
                for i,vertices in enumerate(pieces):
                    part=path+f'_Parts/Part_{i:02d}';mesh=UsdGeom.Mesh.Define(stage,part)
                    matrix_attr(mesh.GetPrim(),matrix)
                    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
                    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(vertices)//3,3,dtype=np.int32)))
                    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(len(vertices),dtype=np.int32)))
                    mesh.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.array([vertices.min(0),vertices.max(0)],np.float32)))
                    mesh.CreatePurposeAttr('guide');mesh.CreateVisibilityAttr('invisible')
                    _collision(mesh.GetPrim());UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr('convexHull')
                    new_paths.append(part)
                UsdPhysics.CollisionAPI(source).CreateCollisionEnabledAttr(False);old_paths.append(path)
    return dict(model='source_complete_wrist_cell_hulls_v1',replaced_colliders=old_paths,collider_paths=new_paths,
        source_asset_edited=False,source_visual_edited=False,native_qualified=False)
