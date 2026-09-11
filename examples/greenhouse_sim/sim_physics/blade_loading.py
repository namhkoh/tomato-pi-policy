"""Mechanical loading diagnostics only: no grasp assertion or seam-release API."""
import numpy as np
from .knife import ShearParameters,leading_face_normal


def slide_parameters(limit_n,desired_speed_m_s=.001):
    if (not np.isfinite([limit_n,desired_speed_m_s]).all()
            or not .05<=limit_n<=.45 or not 0<desired_speed_m_s<=.001):
        raise ValueError('Finite 0.05..0.45 N loading limit and <=1 mm/s approach required')
    # Use the implicit native joint velocity drive. Explicit high-gain force
    # updates at 240 Hz can overshoot on a 50 g carriage before contact.
    return dict(stiffness_n_m=0.,damping_ns_m=float(limit_n/desired_speed_m_s),
        maximum_force_n=float(limit_n),target_velocity_m_s=float(desired_speed_m_s))


def joint_anchors(frames,local_points):
    frames=np.asarray(frames,float);points=np.asarray(local_points,float)
    if frames.shape!=(2,4,4) or points.shape!=(2,3) or not np.isfinite(np.r_[frames.ravel(),points.ravel()]).all():
        raise ValueError('Two finite native frames and authored joint anchors required')
    return np.einsum('nij,nj->ni',frames[:,:3,:3],points)+frames[:,:3,3]


def joint_angular_error(frames,local_rotations):
    frames=np.asarray(frames,float);rotations=np.asarray(local_rotations,float)
    if frames.shape!=(2,4,4) or rotations.shape!=(2,3,3) or not np.isfinite(rotations).all():
        raise ValueError('Two native bodies and authored joint rotations required')
    world=frames[:,:3,:3]@rotations
    return float(np.arccos(np.clip((np.trace(world[0].T@world[1])-1)/2,-1,1)))


def loading_sample_safe(total_force_n,min_separation_m,speed_m_s,anchor_gap_m,seam_enabled,angular_error_rad=0.):
    values=[total_force_n,min_separation_m,speed_m_s,anchor_gap_m,angular_error_rad]
    return bool(np.isfinite(values).all() and 0<=total_force_n<=.5
        and min_separation_m>=-.001 and 0<=speed_m_s<=.02
        and 0<=anchor_gap_m<=.001 and 0<=angular_error_rad<=np.radians(1.) and seam_enabled is True)


class LoadingWindow:
    """Measure whether force/dwell/net advance can coexist at the intact seam.

    This deliberately cannot release anything and never supplies held=True to
    ShearGate. A bench window is not a qualified robot cut or tissue calibration.
    """
    def __init__(self):
        self.parameters=ShearParameters();self.origin=None;self.previous=None
        self.dwell=0.;self.travel=0.;self.max_dwell=0.;self.max_travel=0.;self.observed=False

    def sample(self,dt,force_n,relative_advance_m,eligible):
        if not np.isfinite([dt,force_n,relative_advance_m]).all() or dt<=0 or force_n<0:
            raise ValueError('Invalid native loading sample')
        p=self.parameters
        forward=self.previous is None or relative_advance_m-self.previous>=-1e-6
        if not eligible or not forward or not p.force_n<=force_n<=p.maximum_force_n:
            self.origin=None;self.previous=None;self.dwell=0.;self.travel=0.
        else:
            if self.origin is None:self.origin=relative_advance_m
            self.previous=relative_advance_m;self.dwell+=dt
            self.travel=max(0.,relative_advance_m-self.origin)
            self.max_dwell=max(self.max_dwell,self.dwell);self.max_travel=max(self.max_travel,self.travel)
            self.observed |= self.dwell>=p.dwell_s and self.travel>=p.minimum_loading_travel_m
        return dict(force_window_dwell_s=self.dwell,net_loaded_advance_m=self.travel)

    def report(self):
        return dict(force_displacement_window_observed=bool(self.observed),
            maximum_contiguous_force_window_s=self.max_dwell,maximum_net_loaded_advance_m=self.max_travel,
            robot_grasp_verified=False,seam_release_authorized=False,tissue_fracture_calibrated=False)
