"""Small reobserved shaft-axis correction; not an observation authenticity gate."""
import numpy as np
from scipy.spatial.transform import Rotation


def align_to_axis(rotation, rest_axis, observed_axis):
    r=np.asarray(rotation,float);a=np.asarray(rest_axis,float);b=np.asarray(observed_axis,float)
    if (r.shape!=(3,3) or a.shape!=(3,) or b.shape!=(3,)
            or not np.isfinite(np.r_[r.flat,a,b]).all()
            or not np.allclose(r.T@r,np.eye(3),rtol=0,atol=1e-6)
            or np.linalg.det(r)<0 or abs(np.linalg.norm(a)-1)>1e-5
            or abs(np.linalg.norm(b)-1)>1e-5):
        raise ValueError('Rigid palm and unit anatomical axes required')
    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b)
    cross=np.cross(a,b);sine=np.linalg.norm(cross);cosine=np.clip(a@b,-1.,1.)
    angle=float(np.arctan2(sine,cosine))
    if angle>np.radians(10):
        raise ValueError('Observed shaft moved >10 degrees; new grasp strategy required')
    if sine<1e-12:return r.copy()
    return Rotation.from_rotvec(cross*(angle/sine)).as_matrix()@r


def approach_rotation(start, goal, fraction):
    """Rotate smoothly during approach; hold final orientation during pull."""
    if not np.isfinite(fraction) or not 0<=fraction<=1.15+1e-12:
        raise ValueError('Bounded approach fraction required')
    if np.array_equal(start,goal):return np.array(start,copy=True)
    vector=Rotation.from_matrix(np.asarray(start).T@np.asarray(goal)).as_rotvec()
    return np.asarray(start)@Rotation.from_rotvec(min(fraction,1.)*vector).as_matrix()
