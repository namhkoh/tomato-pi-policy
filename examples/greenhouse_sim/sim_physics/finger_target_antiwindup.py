"""Project finger target state to the current bounded PD effort interval.

Only controller targets change, never physical state, actuator limits, contact
classification, compression limits or cut permission. Native feedback must be
read synchronously by the caller before the same control step.
"""
import numpy as np


def project(half_gaps,positions,velocities,caps,*,minimum,retention_preload=False,symmetric=False,maximum=.025,
            closing_effort_cap_n=None):
    if type(symmetric) is not bool:raise ValueError('Explicit symmetric aperture mode required')
    if type(retention_preload) is not bool:raise ValueError('Explicit retention-preload profile required')
    maximum_pd_n=.30 if retention_preload else .15
    native_cap_bound=max(maximum_pd_n+1e-8,float(np.float32(maximum_pd_n)))
    values=[np.asarray(x,dtype=float) for x in (half_gaps,positions,velocities,caps)]
    gaps,q,v,cap=values
    if (any(x.shape!=(2,) or not np.isfinite(x).all() for x in values)
            or isinstance(minimum,(bool,np.bool_)) or not np.isfinite(minimum) or not 0<=minimum<.01
            or isinstance(maximum,(bool,np.bool_)) or not np.isfinite(maximum) or not minimum<maximum<=.025
            or np.any(cap<=0) or np.any(cap>native_cap_bound)
            or np.any(gaps<minimum) or np.any(gaps>maximum)
            or q[0]>1e-6 or q[1]<-1e-6 or np.any(abs(q)>.05+1e-6)
            or np.any(abs(v)>.051)):
        raise ValueError('Finite native finger state, PD budgets and bounded target gaps required')
    signs=np.array([-1.,1.]);raw=signs*gaps
    # -cap <= 200*(target-q)-5*v <= cap, including damping.
    low=q+(5*v-cap)/200;high=q+(5*v+cap)/200
    if closing_effort_cap_n is not None:
        if (not (retention_preload and symmetric)
                or isinstance(closing_effort_cap_n,(bool,np.bool_))
                or not np.isscalar(closing_effort_cap_n) or not np.isfinite(closing_effort_cap_n)
                or not 0<=closing_effort_cap_n<=maximum_pd_n):
            raise ValueError('Bounded symmetric retention closing-effort backoff required')
        # Closing has opposite signs at the two fingers. Restrict only that
        # side of each ORIGINAL motor interval. Including measured q and v
        # prevents an opening jaw from outrunning a slow reference backoff
        # and paradoxically increasing the commanded squeeze.
        high[0]=min(high[0],q[0]+(5*v[0]+closing_effort_cap_n)/200)
        low[1]=max(low[1],q[1]+(5*v[1]-closing_effort_cap_n)/200)
    geometry_low=np.array([-maximum,minimum]);geometry_high=np.array([-minimum,maximum])
    intersection_low=np.maximum(low,geometry_low);intersection_high=np.minimum(high,geometry_high)
    overlap=intersection_low<=intersection_high
    projected=np.clip(raw,low,high)
    projected=np.clip(projected,geometry_low,geometry_high)
    common_interval=None
    if symmetric:
        if abs(gaps[0]-gaps[1])>1e-9:raise ValueError('Symmetric raw targets required')
        # q_target=(-gap,+gap). Intersect BOTH native PD intervals with the
        # physical compression range; never translate the commanded jaw center.
        common_low=max(minimum,-high[0],low[1])
        common_high=min(maximum,-low[0],high[1])
        if common_low>common_high:
            raise RuntimeError('No symmetric aperture within both native PD budgets and geometry')
        gap=float(np.clip(gaps[0],common_low,common_high))
        projected=signs*gap;common_interval=[float(common_low),float(common_high)]
    out=projected*signs
    return out,dict(model='measured_finger_target_antiwindup_v1',
        pre_positions_m=q.tolist(),pre_velocities_m_s=v.tolist(),raw_targets_m=raw.tolist(),
        projected_targets_m=projected.tolist(),target_correction_m=(projected-raw).tolist(),
        unsaturated_interval_intersects_geometry=overlap.tolist(),
        resulting_unclipped_pd_n=(200*(projected-q)-5*v).tolist(),
        measured_effort=False,contact_used_as_applied_force=False,changes_physical_state=False,
        maximum_non_gravity_pd_n=maximum_pd_n,
        closing_effort_backoff_cap_n=closing_effort_cap_n,
        closing_effort_backoff_uses_measured_state=closing_effort_cap_n is not None,
        commanded_maximum_half_gap_m=maximum,changes_physical_joint_limits=False,
        symmetric_aperture_command=symmetric,common_gap_interval_m=common_interval,
        load_profile='retention_preload_v1' if retention_preload else 'legacy_preload_v1')
