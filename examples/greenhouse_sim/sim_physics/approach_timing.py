"""Retiming only of an already checked joint path; never motion authority.

The old fixed four-second index ramp can command a long lift/orientation path
much faster than a short approach. Preserve every knot and interpolation edge.
Conservative engineering speed bounds are not hardware-calibrated dynamics.
"""
import numpy as np
from scipy.spatial.transform import Rotation


class ApproachTiming:
    def __init__(self,robot):
        q=np.asarray(robot.plan['approach'],float)
        if q.ndim!=2 or q.shape[1]!=7 or not 2<=len(q)<=10000 or not np.isfinite(q).all():
            raise ValueError('Finite already-checked seven-joint approach required')
        frames=np.asarray([robot.kin.forward('right',row,robot.base) for row in q])
        if frames.shape!=(len(q),4,4) or not np.isfinite(frames).all():
            raise ValueError('Finite FK for every original approach knot required')
        self.joint_speed=30.;self.wrist_speed=.1;self.rotation_speed=30.
        dq=np.max(np.abs(np.diff(q,axis=0)),axis=1)
        dx=np.linalg.norm(np.diff(frames[:,:3,3],axis=0),axis=1)
        dr=np.degrees(Rotation.from_matrix(frames[1:,:3,:3]@np.swapaxes(frames[:-1,:3,:3],1,2)).magnitude())
        seconds=np.maximum.reduce((dq/self.joint_speed,dx/self.wrist_speed,dr/self.rotation_speed,
            np.full(len(dq),1e-9)))
        self.knots=np.r_[0.,np.cumsum(seconds)]
        # Smoothstep has peak derivative1.5. This stretches the clock; it does
        # not move/smooth shortcut waypoints or change cut-feed timing.
        self.duration=max(4.,1.5*float(self.knots[-1]))
        if self.duration>45.:raise RuntimeError('Rate-bounded approach exceeds45s; new path required')
        self.original_maximum_joint_step_degrees=float(dq.max())
        self.original_maximum_wrist_step_m=float(dx.max())

    def fraction(self,elapsed):
        if isinstance(elapsed,bool) or not np.isfinite(elapsed) or elapsed<0:
            raise ValueError('Finite nonnegative approach elapsed time required')
        u=min(float(elapsed)/self.duration,1.);u=u*u*(3-2*u)
        where=u*self.knots[-1]
        low=min(int(np.searchsorted(self.knots,where,side='right')-1),len(self.knots)-2)
        alpha=(where-self.knots[low])/(self.knots[low+1]-self.knots[low])
        return float(np.clip((low+alpha)/(len(self.knots)-1),0.,1.))

    def report(self):
        return dict(model='same_path_rate_bounded_approach_v1',duration_s=self.duration,
            original_duration_s=4.,joint_speed_bound_degrees_s=self.joint_speed,
            sampled_wrist_speed_bound_m_s=self.wrist_speed,sampled_orientation_speed_bound_degrees_s=self.rotation_speed,
            maximum_original_joint_step_degrees=self.original_maximum_joint_step_degrees,
            maximum_original_wrist_step_m=self.original_maximum_wrist_step_m,
            path_knots_changed=False,tracking_limit_changed=False,contact_limits_changed=False,
            cut_feed_or_postrelease_deadlines_changed=False,acceleration_certified=False,
            native_tracking_required=True,motion_authorized=False)
