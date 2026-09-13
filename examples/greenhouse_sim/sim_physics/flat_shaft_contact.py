"""Conservative current material-point witness for experimental cylinder sides.

No enclosing capsule, old contact force, or commanded pose verifies proximity.
The caller separately validates generated geometry and signed native forces.
"""
import numpy as np


def current_side_witness(point,radius,half_height,shaft_world,pad_world,pad_half,axis,sign,offset):
    from .shaft_grasp import _array,_pose,_number
    q=_array(point,(3,));half=_array(pad_half,(3,))
    sw,pw=_pose(shaft_world),_pose(pad_world)
    _number(radius,np.finfo(float).tiny,1.);_number(half_height,np.finfo(float).tiny,1.)
    _number(offset,0.,.002)
    if type(axis) is not int or axis not in (0,1,2) or type(sign) is not int or sign not in (-1,1) or np.any(half<=0):
        raise ValueError('Positive pad dimensions and signed face axis required')
    return _current_side_witness_validated(q,radius,half_height,sw,pw,half,axis,sign,offset)


def _current_side_witness_validated(q,radius,half_height,sw,pw,half,axis,sign,offset):
    """Internal arithmetic only, with the identical public validation upstream.

    ShaftGraspEvidence owns immutable per-step _poses batches and validated
    frozen shape descriptors. Rechecking those two matrices for EVERY contact
    row duplicates the same checks; no contact/proximity check is skipped.
    Public callers must use current_side_witness, not this internal function.
    """
    rho=float(np.linalg.norm(q[:2]))
    if not rho:raise ValueError('Cylinder side point needs a radial direction')
    material=np.r_[q[:2]*(radius/rho),np.clip(q[2],-half_height,half_height)]
    point_now=sw[:3,:3]@material+sw[:3,3]
    local=(point_now-pw[:3,3])@pw[:3,:3]
    face=np.clip(local,-half,half);face[axis]=sign*half[axis]
    signed_gap=float(sign*(local[axis]-face[axis]))
    tangent=local-face;tangent[axis]=0.
    tangent_gap=float(np.linalg.norm(tangent))
    # Ensure the shaft centreline has not tunnelled to the back of the pad.
    centre=sw[:3,:3]@np.array([0.,0.,material[2]])+sw[:3,3]
    centre_local=(centre-pw[:3,3])@pw[:3,:3]
    inner_side=float(sign*(centre_local[axis]-face[axis]))
    return dict(model='current_flat_cylinder_material_side_witness_v1',
        surface_gap_m=signed_gap,tangential_face_gap_m=tangent_gap,
        minimum_spine_distance_on_inner_side_m=inner_side,
        passed=bool(-.001<=signed_gap<=offset+1e-6 and tangent_gap<=offset+1e-6
                    and inner_side>=radius-.001))
