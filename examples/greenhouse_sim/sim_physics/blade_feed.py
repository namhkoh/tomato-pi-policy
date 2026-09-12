"""Diagnostic path feed from fresh, guard-accepted native blade contact.

Only retimes/reverses the already screened one-dimensional stroke. This is
not a contact detector, effort limit, release authorization or tissue model.
"""
import math
import numpy as np


class BladeFeed:
    def __init__(self, offsets, *, radius):
        values=np.asarray(offsets,dtype=float)
        if (values.ndim!=1 or len(values)<2 or not np.isfinite(values).all()
                or not np.all(np.diff(values)>0) or not -.025<=values[0]<0<values[-1]<=.02
                or isinstance(radius,(bool,np.bool_)) or not np.isfinite(radius) or not 0<radius<.01):
            raise ValueError('Bounded complete screened stroke and shaft radius required')
        self.start=float(values[0]);self.end=float(values[-1]);self.offset=self.start
        # Stop free-space feed before the 1 mm leading strip plus shaft radius.
        # Additional 1 mm accounts for contact offsets and tracking lag, NOT
        # additional cut eligibility or an expansion of the actual seam.
        self.near_offset=-(float(radius)+.002)
        self.observed_step=0;self.commanded_step=None;self.started_step=None
        self.normal=0.;self.upper=0.;self.released=False;self.receipt=None
        self.near=False

    def observe(self, knife, *, step, guards_passed, released):
        if (type(step) is not int or step!=self.observed_step+1 or guards_passed is not True
                or type(released) is not bool or self.released and not released
                or knife.get('raw_normal_rows_complete') is not True):
            raise RuntimeError('Fresh complete guard-accepted native blade observation required')
        from .knife import KNIFE_IMPULSE_CONTRACT
        if knife.get('force_contract')!=KNIFE_IMPULSE_CONTRACT:
            raise RuntimeError('Original-order signed native blade contract required')
        normal=knife.get('edge_signed_resistance_n');upper=knife.get('tool_contact_upper_bound_n')
        if (any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float,np.integer,np.floating))
                for v in (normal,upper)) or not np.isfinite([normal,upper]).all()
                or not 0<=upper<=.5 or abs(normal)>upper+1e-7):
            raise RuntimeError('Invalid or unsafe measured blade load')
        self.normal=float(normal);self.upper=float(upper);self.released=released
        self.observed_step=step

    def command(self, *, step, dt):
        if (type(step) is not int or step<1 or step!=self.observed_step
                or self.commanded_step is not None and step!=self.commanded_step+1
                or self.released):
            raise RuntimeError('Fresh preceding intact-seam native sample required for blade feed')
        if isinstance(dt,(bool,np.bool_)) or not math.isfinite(dt) or abs(dt-1/240)>1e-12:
            raise ValueError('Qualified 240 Hz blade feed required')
        if self.started_step is None:self.started_step=step
        if (step-self.started_step)*dt>=30:
            raise RuntimeError('Blade load acquisition timed out; no timed release')
        self.near=self.near or self.offset>=self.near_offset or self.upper>.01
        if self.upper>.32 or self.normal>.28:
            speed=-.0005;mode='backoff_load'
        elif .22<=self.normal<=.28:
            speed=0.;mode='hold_valid_load'
        elif self.upper>.10 and self.normal<.01:
            speed=0.;mode='hold_nonqualifying_load'
        elif self.upper>.01:
            speed=min(.00005,max(0.,(.26-self.normal)*.0002));mode='load_feedback'
        elif self.near:
            speed=.0001;mode='near_contact'
        else:
            speed=.002;mode='free_space'
        previous=self.offset
        proposed=previous+speed*dt
        # Never jump over the transition into the slow near-contact zone.
        if not self.near:proposed=min(proposed,self.near_offset)
        self.offset=float(np.clip(proposed,self.start,self.end))
        if self.offset>=self.end:raise RuntimeError('Screened stroke exhausted without a qualified cut')
        self.commanded_step=step
        self.receipt=dict(mode='native_blade_feed_v1',state=mode,command_step=step,
            observation_step=self.observed_step,offset_m=self.offset,delta_m=self.offset-previous,
            signed_resistance_n=self.normal,all_contact_upper_bound_n=self.upper,
            desired_signed_load_n=.26,holding_load_band_n=[.22,.28],backoff_load_n=.32,
            near_contact_speed_m_s=.0001,loading_speed_limit_m_s=.00005,
            backoff_speed_m_s=.0005,cut_authorized=False,force_limit_guaranteed=False)
        return (self.offset-self.start)/(self.end-self.start)
