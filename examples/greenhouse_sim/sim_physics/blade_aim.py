"""Bounded contact-placement proposal; NEVER a change to the release seam.

The native loaded edge must still satisfy the original 3 mm axial window,
leading-strip, force, dwell, grasp and collision guards. Engineering diagnostic
only: a command offset is not evidence that a cut occurred.
"""
import numpy as np


def edge_centre(seam, axis, offset_m=0.):
    if (isinstance(offset_m,(bool,np.bool_)) or not isinstance(offset_m,(float,int,np.floating,np.integer))
            or not np.isfinite(offset_m) or not -.0015<=offset_m<=.0025):
        raise ValueError('Blade contact aim offset must be finite within -1.5..+2.5 mm; distal extension requires section clearance')
    seam,axis=np.asarray(seam,float),np.asarray(axis,float)
    if (seam.shape!=(3,) or axis.shape!=(3,) or not np.isfinite([seam,axis]).all()
            or abs(np.linalg.norm(axis)-1)>1e-5):
        raise ValueError('Measured finite seam and unit shaft axis required')
    return seam+float(offset_m)*axis


def section_placement(axis,normal,*,offset_m,radius,half_thickness):
    """Conservative proximal-stump half-space AND unchanged 3 mm cut window.

    A capped cylinder's support in the positive blade-normal direction is
    r*sin(angle). The closest blade-body plane is offset*cos(angle)-h.
    Separating those planes avoids the attached stump without a collision
    exclusion or material removal. All source crossbar vertices bound h.
    This static proposal is not evidence of physical traversal or a safe path.
    """
    axis,normal=np.asarray(axis,float),np.asarray(normal,float)
    if (axis.shape!=(3,) or normal.shape!=(3,) or not np.isfinite([axis,normal]).all()
            or abs(np.linalg.norm(axis)-1)>1e-5 or abs(np.linalg.norm(normal)-1)>1e-5
            or not np.isfinite([offset_m,radius,half_thickness]).all()
            or not .0015<offset_m<=.0025 or not .0005<=radius<=.01
            or not .0001<=half_thickness<=.005):
        raise ValueError('Measured source crossbar thickness, shaft radius, unit axes and distal proposal required')
    cosine=min(1.,abs(float(axis@normal)))
    if cosine<.9:
        return dict(passed=False,reason='nontransverse_blade_plane',motion_authorized=False)
    sine=float(np.sqrt(max(0.,1-cosine*cosine)))
    gap=float(offset_m*cosine-half_thickness-radius*sine)
    extent=float(offset_m+radius*sine/cosine)
    return dict(model='source_blade_capped_shaft_halfspace_v1',
        passed=bool(gap>=.0001 and extent<=.003-.00005),
        proximal_stump_clearance_m=gap,required_face_clearance_m=.0001,
        maximum_section_axial_distance_m=extent,original_cut_window_m=.003,
        planning_axial_reserve_m=.00005,source_half_thickness_m=float(half_thickness),
        shaft_radius_m=float(radius),blade_normal_axis_cosine=cosine,
        native_contact_checks_still_required=True,collision_geometry_changed=False,
        motion_authorized=False,cut_authorized=False)
