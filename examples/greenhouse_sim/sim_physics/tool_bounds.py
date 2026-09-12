"""Source-enclosing plate bounds, not new native collision geometry."""
import numpy as np


def plate_box(points,local):
    """Smallest source-XY edge-aligned rectangle, retaining full Z thickness.

    The source plate has a slanted leading edge. An axis-aligned enclosing box
    fills a long empty wedge. Every source point remains inside this oriented
    box, with an extra micrometre of padding; scene/self margins are separate.
    This is an authored convex-hull bound, not a native cooking certificate.
    """
    from scipy.spatial import ConvexHull
    points=np.asarray(points,float);local=np.asarray(local,float)
    if (points.ndim!=2 or points.shape[1:]!=(3,) or len(points)<4
            or local.shape!=(4,4) or not np.isfinite(points).all()
            or not np.isfinite(local).all()
            or not np.allclose(local[3],[0,0,0,1],rtol=0,atol=1e-10)
            or not np.allclose(local[:3,:3].T@local[:3,:3],np.eye(3),rtol=0,atol=1e-6)
            or np.linalg.det(local[:3,:3])<0):
        raise ValueError('Finite source plate and rigid local frame required')
    polygon=points[ConvexHull(points[:,:2]).vertices,:2]
    best=None
    for edge in np.roll(polygon,-1,axis=0)-polygon:
        length=np.linalg.norm(edge)
        if length<1e-12:continue
        c,s=edge/length
        axes=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
        projected=points@axes;lo,hi=projected.min(0),projected.max(0)
        volume=float(np.prod(hi-lo))
        if best is None or volume<best[0]:best=(volume,axes,(lo+hi)/2,(hi-lo)/2)
    if best is None or best[0]<=0:raise ValueError('Nondegenerate plate required')
    _,axes,centre,half=best;half=half+1e-6
    if np.any(np.abs(points@axes-centre)>half):raise RuntimeError('Source enclosure failed')
    return local[:3,:3]@axes@centre+local[:3,3],local[:3,:3]@axes,half
