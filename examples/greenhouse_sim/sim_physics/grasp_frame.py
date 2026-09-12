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
    """Rotate during approach; hold orientation only when extending past goal."""
    if not np.isfinite(fraction) or not 0<=fraction<=1.15+1e-12:
        raise ValueError('Bounded approach fraction required')
    if np.array_equal(start,goal):return np.array(start,copy=True)
    vector=Rotation.from_matrix(np.asarray(start).T@np.asarray(goal)).as_rotvec()
    return np.asarray(start)@Rotation.from_rotvec(min(fraction,1.)*vector).as_matrix()


def checked_approach_retraction(start,goal,distance):
    """Retrace the refreshed translation corridor, not the rotated palm Z.

    Reobservation can change the goal translation AND rotation independently.
    This vector is a bounded path proposal, not a new scene-clearance or grasp
    certificate. The existing path orientation, native contact/slip guards,
    and mandatory post-motion reobservation still govern execution.
    """
    a,b=np.asarray(start,float),np.asarray(goal,float)
    if (a.shape!=(4,4) or b.shape!=(4,4) or not np.isfinite([a,b]).all()
            or isinstance(distance,(bool,np.bool_)) or not np.isscalar(distance)
            or not np.isfinite(distance) or not 0<=distance<=.01):
        raise ValueError('Rigid approach endpoints and finite 0..10 mm retraction required')
    for frame in (a,b):
        if (not np.allclose(frame[3],[0,0,0,1],rtol=0,atol=1e-9)
                or not np.allclose(frame[:3,:3].T@frame[:3,:3],np.eye(3),rtol=0,atol=1e-6)
                or np.linalg.det(frame[:3,:3])<0):
            raise ValueError('Rigid checked approach endpoints required')
    delta=a[:3,3]-b[:3,3];length=float(np.linalg.norm(delta))
    if not np.isfinite(length) or length<distance+.008-1e-12:
        raise ValueError('Retraction must leave 8 mm inside the actual checked approach')
    return distance*delta/length
