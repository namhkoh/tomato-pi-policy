"""Invertible transverse shear with an exactly preserved proximal mesh region.

The coordinate is projection on the donor's first straight centerline segment.
All faces intersecting x<=30mm are held entirely fixed; the smoothstep finishes
at x=60mm. Since displacement is perpendicular to this coordinate, no distal
face can enter the protected halfspace. No physical or native approval follows.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
from .audit import safe_asset
from .plant_variants import PHYSICS_FIELDS, digest
from .procedural_petiole_geometry import require, unit

PROTECTED_M = .030
DISTAL_M = .060
FACE_GUARD_EPS_M = 1e-9


def protected_face_indices(points, counts, indices, anchor, tangent):
    points=np.asarray(points,float);x=(points-np.asarray(anchor))@np.asarray(tangent)
    selected=[];faces=[];offset=0
    for face_index,count in enumerate(counts):
        face=np.asarray(indices[offset:offset+count],int);offset+=count
        require(len(face)>=3 and np.all((face>=0)&(face<len(points))), 'Invalid authored face')
        if float(x[face].min())<=PROTECTED_M+FACE_GUARD_EPS_M:
            selected.extend(face.tolist());faces.append(face_index)
    require(offset==len(indices),'Face topology count mismatch')
    return np.unique(selected).astype(int),faces,x


def mesh_guard(path,origin,anchor,tangent):
    from pxr import Usd,UsdGeom
    stage=Usd.Stage.Open(str(path));require(bool(stage),'Unreadable donor mesh')
    rows=[];guard=PROTECTED_M
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):continue
        mesh=UsdGeom.Mesh(prim);points=np.asarray(mesh.GetPointsAttr().Get(),float)+origin
        ids,faces,x=protected_face_indices(points,mesh.GetFaceVertexCountsAttr().Get(),
            mesh.GetFaceVertexIndicesAttr().Get(),anchor,tangent)
        if len(ids):guard=max(guard,float(x[ids].max()))
        rows.append(dict(mesh=str(prim.GetPath()),protected_vertex_indices=ids.tolist(),
            protected_face_indices=faces,source_point_count=len(points),
            source_points_sha256=digest(points.tolist())))
    require(rows and any(r['protected_face_indices'] for r in rows),'No proximal authored faces')
    guard+=FACE_GUARD_EPS_M
    require(PROTECTED_M<guard<DISTAL_M,'Authored proximal faces extend beyond supported blend')
    return guard,rows


class ProximalShear:
    radial_scale=1.
    def __init__(self,anchor,tangent,direction,amplitude,guard):
        self.anchor=np.asarray(anchor,float);self.tangent=unit(tangent);self.direction_vector=unit(direction)
        self.amplitude=float(amplitude);self.guard=float(guard)
        require(np.isfinite(self.anchor).all() and self.anchor.shape==(3,),'Finite attachment required')
        require(abs(float(self.tangent@self.direction_vector))<1e-12,'Shear must be transverse')
        require(self.amplitude==.025 and PROTECTED_M<self.guard<DISTAL_M,'Fixed25mm supported blend only')
        maximum_gradient=self.amplitude*1.5/(DISTAL_M-self.guard)
        worst=np.eye(3)+maximum_gradient*np.outer(self.direction_vector,self.tangent)
        require(np.linalg.det(worst)>.05 and np.linalg.cond(worst)<30,'Blend violates unchanged Jacobian guard')

    def coordinate(self,points):return (np.asarray(points,float)-self.anchor)@self.tangent

    def weights(self,points):
        x=self.coordinate(points);t=np.clip((x-self.guard)/(DISTAL_M-self.guard),0.,1.)
        return t*t*(3.-2.*t)

    def derivatives(self,points):
        x=self.coordinate(points);t=np.clip((x-self.guard)/(DISTAL_M-self.guard),0.,1.)
        return 6.*t*(1.-t)/(DISTAL_M-self.guard)

    def map(self,points):
        p=np.asarray(points,float);return p+self.amplitude*self.weights(p)[...,None]*self.direction_vector

    def jacobian(self,points,h=None):
        p=np.asarray(points,float);a=self.amplitude*self.derivatives(p)
        return np.eye(3)+a[...,None,None]*np.outer(self.direction_vector,self.tangent)

    def normals(self,points,normals):
        p=np.asarray(points,float);n=np.asarray(normals,float);j=self.jacobian(p)
        determinant=np.linalg.det(j);condition=np.linalg.cond(j)
        require(np.isfinite(j).all() and np.all(determinant>.05) and np.all(condition<30),
            'Unsafe analytic deformation Jacobian')
        changed=self.derivatives(p)!=0
        out=n.copy()
        if np.any(changed):
            out[changed]=np.linalg.solve(np.swapaxes(j[changed],-1,-2),n[changed][...,None])[...,0]
            length=np.linalg.norm(out[changed],axis=1)
            require(np.all(length>1e-12),'Degenerate source normal')
            out[changed]/=length[:,None]
        return out,dict(minimum_determinant=float(determinant.min()),maximum_condition_number=float(condition.max()),
            analytic_jacobian=True,source_normals_exact_where_jacobian_identity=True,
            global_injective_reason='longitudinal_coordinate_invariant_transverse_translation')

    def direction(self,point,value):return unit(self.jacobian(np.asarray(point))@np.asarray(value))


class RigidTranslation:
    radial_scale=1.
    def __init__(self,shift):self.shift=np.asarray(shift,float)
    def map(self,points):return np.asarray(points,float)+self.shift
    def jacobian(self,points,h=None):return np.broadcast_to(np.eye(3),np.asarray(points).shape[:-1]+(3,3)).copy()
    def direction(self,point,value):return np.asarray(value,float).copy()
    def normals(self,points,normals):
        return np.asarray(normals,float).copy(),dict(minimum_determinant=1.,maximum_condition_number=1.,
            analytic_jacobian=True,source_normals_exact_where_jacobian_identity=True,
            global_injective_reason='rigid_translation')


def component_transport(raw,change):
    if raw['id']==change['component_id']:return change['warp']
    require(raw['type']=='leaf' and raw['parent']==change['component_id'],'Only direct leaf transport supported')
    warp=change['warp'];attachment=np.asarray(raw['attach_point'],float)
    require(warp.coordinate(attachment)>=DISTAL_M,'Leaf attachment inside blend is unsupported')
    require(np.array_equal(warp.jacobian(attachment),np.eye(3)),'Leaf rigid orientation not certified')
    return RigidTranslation(warp.amplitude*warp.direction_vector)


def controlled_curve(old_absolute,radii,warp):
    old=np.asarray(old_absolute,float);radii=np.asarray(radii,float)
    require(old.ndim==2 and old.shape[1]==3 and len(old)>=2 and radii.shape==(len(old),)
        and np.isfinite(old).all() and np.isfinite(radii).all() and np.all(radii>0),
        'Finite source centerline/radii required')
    lengths=np.linalg.norm(np.diff(old,axis=0),axis=1);arc=np.r_[0.,np.cumsum(lengths)]
    require((lengths>1e-9).all() and arc[-1]>PROTECTED_M,'Supported nondegenerate source prefix required')
    require(np.allclose(old[0],warp.anchor,atol=1e-9,rtol=0),'Source attachment mismatch')
    # Preserve every original knot. Blend knots are intersections of EACH
    # source segment with longitudinal x-levels, not source-arc approximations.
    x=warp.coordinate(old);extras=[PROTECTED_M]
    levels=np.linspace(warp.guard,DISTAL_M,33)
    for i in range(len(lengths)):
        delta=x[i+1]-x[i]
        if abs(delta)<=1e-15:continue
        for level in levels:
            fraction=(level-x[i])/delta
            if 0.<fraction<1.:
                value=arc[i]+fraction*lengths[i]
                # Avoid a duplicate node differing only by arithmetic roundoff.
                if min(value-arc[i],arc[i+1]-value)>1e-10:extras.append(float(value))
    samples=np.unique(np.r_[arc,extras])
    points=np.column_stack([np.interp(samples,arc,old[:,i]) for i in range(3)])
    radius=np.interp(samples,arc,radii)
    for index,distance in enumerate(arc):
        at=np.flatnonzero(samples==distance);require(len(at)==1,'Original source knot missing')
        points[at[0]]=old[index];radius[at[0]]=radii[index]
    # On a piecewise-linear source chain, the longitudinal coordinate reaches
    # its maximum at a knot. These endpoints therefore prove identity for the
    # entire protected0-30mm centerline, including bends.
    protected=samples<=PROTECTED_M
    require(np.all(warp.coordinate(points[protected])<=warp.guard)
        and np.array_equal(warp.map(points[protected]),points[protected]),
        'Warp changes protected source centerline prefix')
    mapped=warp.map(points)-warp.anchor
    new_arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(mapped,axis=0),axis=1))]
    require(np.all(np.diff(new_arc)>1e-10),'Generated centerline degeneracy')
    return dict(points=mapped,radius=radius,arc=new_arc),samples


def transform_metadata(raw,change):
    result=deepcopy(raw);field=component_transport(raw,change)
    if raw['id']==change['component_id']:
        # Preserve authored attachment/origin/axis/radius verbatim, avoiding any
        # normalization or frame rotation in the protected physical interval.
        curve=change['curve'];origin=np.asarray(raw['transform']['translate'],float)
        points=curve['points']+np.asarray(change['new_anchor_m'])-origin
        result['capsules']=[np.c_[points,curve['radius']].tolist()]
        result['length']=float(curve['arc'][-1])
    else:
        shift=field.shift
        result['transform']={'translate':(np.asarray(raw['transform']['translate'])+shift).tolist()}
        result['attach_point']=(np.asarray(raw['attach_point'])+shift).tolist()
        # Rigid translation preserves axis and component-local capsules exactly.
    for key in PHYSICS_FIELDS:result.pop(key,None)
    return result


def proximal_mesh_identity(source_path,generated_path,source_component,generated_component,warp):
    from pxr import Usd,UsdGeom
    source=Usd.Stage.Open(str(source_path));new=Usd.Stage.Open(str(generated_path))
    require(source and new,'Readable proximal geometry required')
    require(source_component['translation_plant_m']==generated_component['translation_plant_m']
        and source_component['attachment_plant_m']==generated_component['attachment_plant_m']
        and source_component['axis_plant']==generated_component['axis_plant'],'Proximal authored metadata moved')
    origin=np.asarray(source_component['translation_plant_m']);rows=[]
    for prim in source.Traverse():
        if not prim.IsA(UsdGeom.Mesh):continue
        a=UsdGeom.Mesh(prim);b=UsdGeom.Mesh(new.GetPrimAtPath(prim.GetPath()));require(b,'Missing generated mesh')
        points=np.asarray(a.GetPointsAttr().Get(),float);actual=np.asarray(b.GetPointsAttr().Get(),float)
        counts=np.asarray(a.GetFaceVertexCountsAttr().Get());indices=np.asarray(a.GetFaceVertexIndicesAttr().Get())
        require(np.array_equal(counts,b.GetFaceVertexCountsAttr().Get())
            and np.array_equal(indices,b.GetFaceVertexIndicesAttr().Get()),'Proximal topology changed')
        ids,faces,x=protected_face_indices(points+origin,counts,indices,warp.anchor,warp.tangent)
        require(points.shape==actual.shape and np.array_equal(points[ids],actual[ids]),'Protected face vertices changed')
        new_x=warp.coordinate(actual+origin)
        # Float32 serialization can move coordinates by sub-micron roundoff.
        # Strictly require all unprotected faces to stay outside the halfspace.
        outside=[];offset=0
        for face_index,count in enumerate(counts):
            face=indices[offset:offset+count];offset+=count
            if face_index not in faces:
                require(np.min(new_x[face])>PROTECTED_M,'Distal face entered proximal region')
                outside.append(float(np.min(new_x[face])))
        normals=np.asarray(a.GetNormalsAttr().Get(),float);normal_new=np.asarray(b.GetNormalsAttr().Get(),float)
        interpolation=a.GetNormalsInterpolation();require(interpolation==b.GetNormalsInterpolation(),'Normal interpolation changed')
        if interpolation=='faceVarying':normal_ids=np.flatnonzero(np.isin(indices,ids))
        elif interpolation in ('vertex','varying'):normal_ids=ids
        else:raise ValueError('Unsupported proximal normal interpolation')
        require(np.array_equal(normals[normal_ids],normal_new[normal_ids]),'Protected source normals changed')
        rows.append(dict(mesh=str(prim.GetPath()),protected_face_count=len(faces),protected_vertex_count=len(ids),
            protected_faces_sha256=digest(faces),protected_points_sha256=digest(points[ids].tolist()),
            exact_points=True,exact_normals=True,exact_topology=True,
            minimum_distal_face_longitudinal_m=min(outside) if outside else None))
    require(rows and any(r['protected_face_count'] for r in rows),'No exact proximal mesh proof')
    return dict(protected_source_arc_m=[0.,PROTECTED_M],protected_halfspace_longitudinal_max_m=PROTECTED_M,
        zero_warp_until_m=warp.guard,distal_constant_shift_from_m=DISTAL_M,protected_source_centerline_identity_verified_separately=True,
        complete_intersecting_faces_exact=True,distal_faces_remain_outside_protected_halfspace=True,
        source_mesh_surface_equality_continuous_in_protected_halfspace=True,mesh_rows=rows,
        unchanged_numeric_radius_definition=True,physical_radius_reestimated=False)
