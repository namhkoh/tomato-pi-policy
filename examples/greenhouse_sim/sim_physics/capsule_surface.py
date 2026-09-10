"""Exact segment/triangle distance for conservative static capsule screening.

This is planning geometry, not a replacement for native PhysX contact. Unknown
shape scaling falls back to the existing enclosing-box test.
"""
import numpy as np


def _point_segments(point, a, b):
    v=b-a
    t=np.sum((point-a)*v,axis=-1)/np.maximum(np.sum(v*v,axis=-1),1e-30)
    delta=point-(a+np.clip(t,0,1)[...,None]*v)
    return np.sum(delta*delta,axis=-1)


def segment_triangles_distance(start,end,triangles):
    """Minimum Euclidean distance, including degenerate and parallel faces."""
    a,b=np.asarray(start,float),np.asarray(end,float)
    tri=np.asarray(triangles,float)
    if a.shape!=(3,) or b.shape!=(3,) or tri.ndim!=3 or tri.shape[1:]!=(3,3):
        raise ValueError('Expected segment endpoints and Nx3x3 triangles')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(tri).all():
        raise ValueError('Nonfinite collision geometry')
    if not len(tri): return float('inf')
    p=tri; q=np.roll(tri,-1,axis=1); u=b-a; v=q-p; w=a-p
    # Boundary minima: endpoint -> opposite segment for all four endpoints.
    best=np.minimum.reduce([_point_segments(a,p,q),_point_segments(b,p,q),
        _point_segments(p,a,b),_point_segments(q,a,b)]).min()
    uu=np.dot(u,u); vv=np.sum(v*v,axis=-1); uv=np.sum(u*v,axis=-1)
    uw=np.sum(u*w,axis=-1); vw=np.sum(v*w,axis=-1)
    determinant=uu*vv-uv*uv
    valid=determinant>1e-14*np.maximum(uu*vv,1e-30)
    denom=np.where(valid,determinant,1.)
    s=(uv*vw-vv*uw)/denom; t=(uu*vw-uv*uw)/denom
    inside=valid&(s>=0)&(s<=1)&(t>=0)&(t<=1)
    distances=np.sum((w+s[...,None]*u-t[...,None]*v)**2,axis=-1)
    best=min(best,np.min(np.where(inside,distances,np.inf)))
    origin=tri[:,0]; e=tri[:,1]-origin; f=tri[:,2]-origin
    normal=np.cross(e,f); nn=np.sum(normal*normal,axis=1)
    area=nn>1e-28
    def inside_face(points):
        r=points-origin
        ee=np.sum(e*e,axis=1); ff=np.sum(f*f,axis=1); ef=np.sum(e*f,axis=1)
        re=np.sum(r*e,axis=1); rf=np.sum(r*f,axis=1)
        det=ee*ff-ef*ef; safe=np.where(area,det,1.)
        x=(ff*re-ef*rf)/safe; y=(ee*rf-ef*re)/safe
        return area&(x>=-1e-12)&(y>=-1e-12)&(x+y<=1+1e-12)
    safe_nn=np.where(area,nn,1.)
    for endpoint in (a,b):
        signed=np.sum((endpoint-origin)*normal,axis=1)
        projected=endpoint-signed[:,None]*normal/safe_nn[:,None]
        best=min(best,np.min(np.where(inside_face(projected),signed*signed/safe_nn,np.inf)))
    velocity=np.sum(normal*u,axis=1)
    crosses=area&(np.abs(velocity)>1e-14*np.sqrt(safe_nn)*max(np.linalg.norm(u),1e-30))
    fraction=np.sum((origin-a)*normal,axis=1)/np.where(crosses,velocity,1.)
    points=a+fraction[:,None]*u
    if np.any(crosses&(fraction>=0)&(fraction<=1)&inside_face(points)): return 0.
    return float(np.sqrt(max(0.,best)))


def world_capsule(prim,matrix):
    """Return native USD capsule endpoints/radius only for uniform scaling."""
    from pxr import UsdGeom
    if not prim.IsA(UsdGeom.Capsule): return None
    rotation=np.asarray(matrix,float)[:3,:3]
    gram=rotation.T@rotation; scale=float(np.sqrt(np.trace(gram)/3))
    if scale<=0 or not np.allclose(gram,np.eye(3)*scale**2,rtol=1e-6,atol=1e-12):
        return None
    capsule=UsdGeom.Capsule(prim); axis='XYZ'.index(str(capsule.GetAxisAttr().Get()))
    delta=rotation[:,axis]*float(capsule.GetHeightAttr().Get())/2
    centre=np.asarray(matrix)[:3,3]
    return centre-delta,centre+delta,float(capsule.GetRadiusAttr().Get())*scale
