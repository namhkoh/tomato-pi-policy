"""Separating-plane lower bounds for an actual flat cylinder and a tool box.

Not a collision solver: a positive bound proves separation along one axis;
failure to find an axis remains a conservative rejection. In particular, the
hemispheres of a capsule bound must not be mistaken for native flat end caps.
"""
import itertools
import numpy as np


def separation(start,end,radius,centre,axes,half):
    a,b,c=np.asarray(start,float),np.asarray(end,float),np.asarray(centre,float)
    r=np.asarray(axes,float);h=np.asarray(half,float)
    if (a.shape!=(3,) or b.shape!=(3,) or c.shape!=(3,) or r.shape!=(3,3) or h.shape!=(3,)
            or not np.isfinite(np.r_[a,b,c,r.flat,h,radius]).all()
            or radius<=0 or np.any(h<=0)
            or not np.allclose(r.T@r,np.eye(3),atol=1e-8,rtol=0)):
        raise ValueError('Finite flat-cylinder and rigid positive box required')
    axis=b-a;length=float(np.linalg.norm(axis))
    if length<1e-9:raise ValueError('Nonzero flat-cylinder length required')
    u=axis/length;middle=(a+b)/2;delta=c-middle
    corners=c+np.array(list(itertools.product((-1.,1.),repeat=3)))*h@r.T
    radial=corners-middle;radial-=np.outer(radial@u,u)
    directions=np.vstack([u,r.T,np.cross(u,r.T),delta,radial])
    norms=np.linalg.norm(directions,axis=1);directions=directions[norms>1e-12]/norms[norms>1e-12,None]
    projection=directions@u
    axial=np.abs(projection)
    radial_extent=np.linalg.norm(directions-np.outer(projection,u),axis=1)
    cylinder_extent=(length/2)*axial+float(radius)*radial_extent
    box_extent=np.abs(directions@r)@h
    # Numerical reserve goes AGAINST clearance. This is not a tolerance
    # relaxation: callers still require the full original 1 mm scene margin.
    return float(np.max(np.abs(directions@delta)-cylinder_extent-box_extent)-1e-8)
