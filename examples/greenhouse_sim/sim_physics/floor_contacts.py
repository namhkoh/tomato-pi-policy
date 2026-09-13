"""Exact rectilinear solid-floor contact decomposition; visible mesh unchanged.

No AABB fill, flattening, infinite plane or convex hull over holes. Only a
closed, consistently wound axis-aligned triangle solid is supported. Grid-cell
parity, boundary area and enclosed volume must agree before any stage edit.
Unsupported source geometry is refused, never silently approximated.
"""
import numpy as np


def boxes(points,counts,indices):
    p=np.asarray(points,float);c=np.asarray(counts);ids=np.asarray(indices)
    if (p.ndim!=2 or p.shape[1]!=3 or not len(p) or not np.isfinite(p).all()
            or c.ndim!=1 or not len(c) or not np.all(c==3) or ids.ndim!=1
            or ids.dtype.kind not in 'iu' or len(ids)!=3*len(c)
            or ids.min()<0 or ids.max()>=len(p)):
        raise ValueError('Finite triangular floor source required')
    unique,inverse=np.unique(p,axis=0,return_inverse=True);faces=inverse[ids].reshape(-1,3)
    t=unique[faces];normal=np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]);norm=np.linalg.norm(normal,axis=1)
    if np.any(norm==0) or not np.all(np.count_nonzero(np.abs(normal)>norm[:,None]*1e-10,axis=1)==1):
        raise ValueError('Only nondegenerate axis-aligned rectilinear floor faces supported')
    edges={}
    for face in faces:
        for a,b in zip(face,np.roll(face,-1)):
            key=tuple(sorted((int(a),int(b))));edges.setdefault(key,[]).append((int(a),int(b)))
    if any(len(v)!=2 or v[0]!=v[1][::-1] for v in edges.values()):
        raise ValueError('Floor must be a closed consistently wound manifold')
    axes=[np.unique(unique[:,i]) for i in range(3)]
    if any(not 2<=len(a)<=16 for a in axes):raise ValueError('Rectilinear floor grid outside bounded exact decomposition')
    occupancy=np.zeros(tuple(len(a)-1 for a in axes),bool)
    # Ray +X, projected onto YZ. Duplicate diagonal hits on a tessellated
    # rectangular face are ONE crossing, not two contradictory parity votes.
    xt=t[np.abs(normal[:,0])>0]
    for cell in np.ndindex(occupancy.shape):
        centre=np.array([(a[i]+a[i+1])/2 for a,i in zip(axes,cell)])
        yz=xt[:,:,1:];a=yz[:,0];u=yz[:,1]-a;v=yz[:,2]-a;w=centre[1:]-a
        det=u[:,0]*v[:,1]-u[:,1]*v[:,0]
        s=(w[:,0]*v[:,1]-w[:,1]*v[:,0])/det
        q=(u[:,0]*w[:,1]-u[:,1]*w[:,0])/det
        hit=(s>=-1e-10)&(q>=-1e-10)&(s+q<=1+1e-10)
        crossings=np.unique(xt[hit,0,0])
        occupancy[cell]=np.count_nonzero(crossings>centre[0])%2==1
    widths=[np.diff(a) for a in axes];volume=0.;area=0.
    for cell in zip(*np.nonzero(occupancy)):
        size=np.array([w[i] for w,i in zip(widths,cell)]);volume+=float(np.prod(size))
        for axis in range(3):
            for sign in (-1,1):
                neighbor=list(cell);neighbor[axis]+=sign
                if neighbor[axis]<0 or neighbor[axis]>=occupancy.shape[axis] or not occupancy[tuple(neighbor)]:
                    area+=float(np.prod(np.delete(size,axis)))
    reference=unique.mean(0);v=t-reference
    source_volume=abs(float(np.einsum('ij,ij->i',v[:,0],np.cross(v[:,1],v[:,2])).sum()/6))
    source_area=float(norm.sum()/2)
    if (volume<=0 or not np.isclose(volume,source_volume,rtol=1e-9,atol=0)
            or not np.isclose(area,source_area,rtol=1e-9,atol=0)):
        raise ValueError('Exact floor cell union does not preserve source volume and boundary area')
    # Greedy coalescing preserves the exact occupied-cell union.
    remaining=occupancy.copy();result=[]
    while remaining.any():
        lo=np.array(np.argwhere(remaining)[0]);hi=lo+1
        for axis in range(3):
            while hi[axis]<remaining.shape[axis]:
                end=hi.copy();end[axis]+=1
                region=tuple(slice(a,b) for a,b in zip(lo,end))
                if not remaining[region].all():break
                hi=end
        remaining[tuple(slice(a,b) for a,b in zip(lo,hi))]=False
        result.append((np.array([a[i] for a,i in zip(axes,lo)]),np.array([a[i] for a,i in zip(axes,hi)])))
    if len(result)>64:raise ValueError('Exact floor decomposition exceeded 64 native boxes')
    return result,dict(source_triangle_count=len(t),box_count=len(result),grid_shape=list(occupancy.shape),
        source_volume_local3=source_volume,box_union_volume_local3=volume,
        source_area_local2=source_area,box_union_boundary_area_local2=area,
        geometry_equivalent=True,holes_or_steps_filled=False)


def apply(stage,floor_root):
    from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade
    root=stage.GetPrimAtPath(floor_root)
    if not root or not root.IsActive():raise ValueError('Existing original floor required')
    source=[]
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get():
            if not prim.IsA(UsdGeom.Mesh) or prim.HasAPI(UsdPhysics.RigidBodyAPI):
                raise ValueError('Only original static floor triangle colliders supported')
            mesh=UsdGeom.Mesh(prim)
            if (any(a.GetNumTimeSamples() for a in (mesh.GetPointsAttr(),mesh.GetFaceVertexCountsAttr(),mesh.GetFaceVertexIndicesAttr()))
                    or any(r.GetTargets() for r in prim.GetRelationships() if r.GetName().startswith('physics:'))):
                raise ValueError('Animated geometry or floor-specific collision relationships require another adapter')
            if mesh.GetHoleIndicesAttr().Get():raise ValueError('Explicit mesh holes require another adapter')
            if UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()!='none':
                raise ValueError('Do not replace a previously approximated floor')
            path=str(prim.GetPath())+'/ExactBoxContacts'
            if stage.GetPrimAtPath(path):raise ValueError('Do not overwrite floor contacts')
            parts,receipt=boxes(mesh.GetPointsAttr().Get(),mesh.GetFaceVertexCountsAttr().Get(),mesh.GetFaceVertexIndicesAttr().Get())
            source.append((prim,path,parts,receipt))
    if not source:raise ValueError('Original active floor triangle collision required')
    paths=[];receipts=[]
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        for prim,path,parts,receipt in source:
            material,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
            for i,(lo,hi) in enumerate(parts):
                shape=UsdGeom.Cube.Define(stage,path+f'/Box_{i:02d}');shape.CreateSizeAttr(2.)
                shape.AddTranslateOp().Set(Gf.Vec3d(*((lo+hi)/2)))
                shape.AddScaleOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*((hi-lo)/2)))
                shape.CreatePurposeAttr('guide');shape.CreateVisibilityAttr('invisible')
                UsdPhysics.CollisionAPI.Apply(shape.GetPrim()).CreateCollisionEnabledAttr(True)
                for schema in prim.GetAppliedSchemas():
                    if schema=='PhysxCollisionAPI':shape.GetPrim().AddAppliedSchema(schema)
                for attr in prim.GetAttributes():
                    if attr.GetName().startswith('physxCollision:') and attr.HasAuthoredValueOpinion():
                        shape.GetPrim().CreateAttribute(attr.GetName(),attr.GetTypeName(),custom=False).Set(attr.Get())
                if material:UsdShade.MaterialBindingAPI.Apply(shape.GetPrim()).Bind(material,materialPurpose='physics')
                paths.append(str(shape.GetPath()))
            UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Set(False)
            receipts.append(dict(source_collider=str(prim.GetPath()),**receipt))
    return dict(model='exact_rectilinear_floor_native_boxes_v1',sources=receipts,collider_paths=paths,
        source_visuals_changed=False,source_layers_changed=False,force_limits_changed=False,
        triangle_contact_material_warning_resolved=False,native_requalification_required=True)
