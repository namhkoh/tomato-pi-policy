"""Project finger target state to the current bounded PD effort interval.

Only controller targets change, never physical state, actuator limits, contact
classification, compression limits or cut permission. Native feedback must be
read synchronously by the caller before the same control step.
"""
import numpy as np


def project(half_gaps,positions,velocities,caps,*,minimum):
    values=[np.asarray(x,dtype=float) for x in (half_gaps,positions,velocities,caps)]
    gaps,q,v,cap=values
    if (any(x.shape!=(2,) or not np.isfinite(x).all() for x in values)
            or isinstance(minimum,(bool,np.bool_)) or not np.isfinite(minimum) or not 0<=minimum<.01
            or np.any(cap<=0) or np.any(cap>.15+1e-8)
            or np.any(gaps<minimum) or np.any(gaps>.025)
            or q[0]>1e-6 or q[1]<-1e-6 or np.any(abs(q)>.05+1e-6)
            or np.any(abs(v)>.051)):
        raise ValueError('Finite native finger state, PD budgets and bounded target gaps required')
    signs=np.array([-1.,1.]);raw=signs*gaps
    # -cap <= 200*(target-q)-5*v <= cap, including damping.
    low=q+(5*v-cap)/200;high=q+(5*v+cap)/200
    geometry_low=np.array([-.025,minimum]);geometry_high=np.array([-minimum,.025])
    intersection_low=np.maximum(low,geometry_low);intersection_high=np.minimum(high,geometry_high)
    overlap=intersection_low<=intersection_high
    projected=np.clip(raw,low,high)
    projected=np.clip(projected,geometry_low,geometry_high)
    out=projected*signs
    return out,dict(model='measured_finger_target_antiwindup_v1',
        pre_positions_m=q.tolist(),pre_velocities_m_s=v.tolist(),raw_targets_m=raw.tolist(),
        projected_targets_m=projected.tolist(),target_correction_m=(projected-raw).tolist(),
        unsaturated_interval_intersects_geometry=overlap.tolist(),
        resulting_unclipped_pd_n=(200*(projected-q)-5*v).tolist(),
        measured_effort=False,contact_used_as_applied_force=False,changes_physical_state=False)
