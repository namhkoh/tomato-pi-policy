"""Necessary wrist-corridor geometry, not a replacement for full arm checks.

A capsule coaxial with the final revolute joint occupies the same volume for
every wrist roll. Prove this from the parsed chain AND actual cached collider;
do not infer it merely from a link name or several sampled IK solutions.
"""
import numpy as np


def capsule_in_wrist(shape,kin):
    """Return a contained, roll-invariant capsule or None when unproven.

    Preserve the native collider/body identity, changing only the planning
    coordinate link. Tiny off-axis roundoff is projected away and SUBTRACTED
    from the radius. Thus rejection uses a contained subset, never an enlarged
    surrogate that could eliminate a valid elbow configuration. Full original
    geometry is still checked for every accepted path.
    """
    path,body,link,kind,value=shape
    if link!='link_right_arm_5' or kind!='capsule':return None
    joints=getattr(kin,'_by_child',{})
    tool=joints.get('ee_right')
    if tool is None or tool.kind!='fixed' or tool.child!='ee_right':return None
    joint=joints.get(tool.parent)
    if (joint is None or joint.parent!=link or joint.child!=tool.parent
            or joint.kind!='revolute'):return None
    matrices=[np.asarray(j.origin,float) for j in (joint,tool)]
    for m in matrices:
        if (m.shape!=(4,4) or not np.isfinite(m).all()
                or not np.allclose(m[3],[0,0,0,1],rtol=0,atol=1e-12)
                or not np.allclose(m[:3,:3].T@m[:3,:3],np.eye(3),rtol=0,atol=1e-12)
                or np.linalg.det(m[:3,:3])<0):return None
    axis=np.asarray(joint.axis,float)
    if (axis.shape!=(3,) or not np.isfinite(axis).all()
            or abs(np.linalg.norm(axis)-1)>1e-12):return None
    axis=axis/np.linalg.norm(axis)
    start,end,radius=value
    points=np.asarray([start,end],float)
    if (points.shape!=(2,3) or not np.isfinite(points).all()
            or not np.isfinite(radius) or radius<=0):raise ValueError('Invalid native wrist capsule')
    origin,fixed=matrices
    local=(points-origin[:3,3])@origin[:3,:3]
    axial=(local@axis)[:,None]*axis
    error=float(np.linalg.norm(local-axial,axis=1).max())
    if error>1e-8:return None  # Not a genuinely coaxial collider.
    contained_radius=float(radius)-error-1e-10
    if contained_radius<=0:return None
    wrist=(axial-fixed[:3,3])@fixed[:3,:3]
    return (path,body,'ee_right','capsule',(wrist[0],wrist[1],contained_radius))
