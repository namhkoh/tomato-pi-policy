"""Physical grasp-span proposals on one connected detachable shaft.

Planning expected-contact identity only. No collision filters, native contact
evidence, grasp state or cut permission is changed by this calculation.
"""
import numpy as np


def connected_span(capsules,selected,first_detachable,axis,origin,interval):
    """Walk from the selected capsule only while its axial extent meets pads.

    No fixed neighbor count, no skipping a gap to a folded-back distal limb.
    The caller supplies a single ordered source shaft, not an arbitrary plant
    graph. Only the detachable side can participate; protected support stays
    excluded even when a finger projection would intersect it.
    """
    axis=np.asarray(axis,float);origin=np.asarray(origin,float);interval=np.asarray(interval,float)
    if (axis.shape!=(3,) or origin.shape!=(3,) or interval.shape!=(2,)
            or not np.isfinite(np.r_[axis,origin,interval]).all()
            or abs(np.linalg.norm(axis)-1)>1e-5 or interval[1]<=interval[0]
            or type(selected) is not int or type(first_detachable) is not int
            or not 0<=first_detachable<=selected<len(capsules)):
        raise ValueError('Ordered shaft, selected detachable body and physical pad interval required')
    extent=[]
    for a,b,radius in capsules:
        a=np.asarray(a,float);b=np.asarray(b,float)
        if (a.shape!=(3,) or b.shape!=(3,) or not np.isfinite(np.r_[a,b,radius]).all()
                or radius<=0):raise ValueError('Finite physical capsules required')
        ends=(np.array([a,b])-origin)@axis
        extent.append((float(min(ends)-radius),float(max(ends)+radius)))
    def intersects(i):return extent[i][1]>=interval[0] and extent[i][0]<=interval[1]
    if not intersects(selected):raise ValueError('Selected shaft is outside the actual finger span')
    accepted={selected}
    for step in (-1,1):
        i=selected+step
        while first_detachable<=i<len(capsules) and intersects(i):
            accepted.add(i);i+=step
    return accepted


def finger_interval(shapes,body_world,axis,origin,*,margin=.001):
    """Enclose both complete actual finger colliders at one commanded pose."""
    axis=np.asarray(axis,float);origin=np.asarray(origin,float)
    if (axis.shape!=(3,) or origin.shape!=(3,) or not np.isfinite(np.r_[axis,origin]).all()
            or abs(np.linalg.norm(axis)-1)>1e-5 or not np.isfinite(margin) or margin!=.001):
        raise ValueError('Measured shaft axis and unchanged 1 mm planning margin required')
    bounds=[];fingers=set()
    for path,body,link,kind,data in shapes:
        if link not in ('ee_finger_l1','ee_finger_l2'):continue
        frame=np.asarray(body_world[link],float)
        if (frame.shape!=(4,4) or not np.isfinite(frame).all()
                or not np.allclose(frame[:3,:3].T@frame[:3,:3],np.eye(3),atol=1e-6)
                or np.linalg.det(frame[:3,:3])<=0):raise ValueError('Rigid finger pose required')
        r,t=frame[:3,:3],frame[:3,3]
        if kind=='capsule':
            a,b,radius=data
            ends=(np.array([r@a+t,r@b+t])-origin)@axis
            low,high=min(ends)-radius,max(ends)+radius
        elif kind=='box':
            centre,axes,half=data
            midpoint=float((r@centre+t-origin)@axis)
            width=float(np.abs(axis@(r@axes))@half)
            low,high=midpoint-width,midpoint+width
        else:raise ValueError('Unsupported finger shape for physical grasp span')
        if not np.isfinite([low,high]).all() or low>=high:raise ValueError('Finite positive finger extent required')
        bounds.append((low,high));fingers.add(link)
    if fingers!={'ee_finger_l1','ee_finger_l2'}:raise ValueError('Both complete fingers required')
    return np.array([min(v[0] for v in bounds)-margin,max(v[1] for v in bounds)+margin])
