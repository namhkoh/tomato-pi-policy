"""Experimental acceleration from fresh knife AND grasp state, not cut physics.

Retains independent force/slip/geometry guards. Never makes a release decision.
Filtered history affects speed only; native safety evidence stays unsmoothed.
"""
from collections import deque
import math
import numpy as np
from .postrelease_feed import validate


class SupportAwareFeed:
    def __init__(self,maximum):
        self.maximum=validate(maximum);self.step=None;self.consumed=None
        self.speed=.0003;self.stable=0;self.slips=deque(maxlen=49);self.receipt=None

    def observe(self,record,*,step):
        if (type(step) is not int or step<1 or self.step is not None and
                (step!=self.step+1 or self.consumed!=self.step)
                or record.get('native_guards_passed') is not True or record.get('cut') is not True
                or abs(record['t']-step/480)>1e-9):
            raise RuntimeError('Fresh guard-accepted post-release feed sample required')
        c=record['contact'];loads=record['robot']['per_finger_contact_upper_bound_n']
        if (c.get('adapter_valid') is not True or c.get('step_id')!=step
                or len(loads)!=2 or {p.rsplit('/',1)[-1] for p in loads}!={'ee_finger_l1','ee_finger_l2'}
                or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v)
                    or not 0<=v<.5 for v in loads.values())):
            raise RuntimeError('Exact fresh left-finger all-contact loads required')
        upper=record['knife']['tool_contact_upper_bound_n']
        if isinstance(upper,bool) or not isinstance(upper,(int,float)) or not math.isfinite(upper) or not 0<=upper<=.5:
            raise RuntimeError('Finite current full-tool load required')
        cut_only='cut_only_park' in record
        if cut_only:
            park=record['cut_only_park']
            if (park.get('model')!='native_unheld_left_park_v1' or park.get('step_id')!=step
                    or c.get('bilateral') is not False or record['slip_m'] is not None
                    or max(loads.values())>.005):
                raise RuntimeError('Current verified unloaded left park required')
            slip=None;drift=0.;headroom=1.;safe=True
        else:
            slip=record['slip_m']
            if (c.get('bilateral') is not True or isinstance(slip,bool)
                    or not isinstance(slip,(int,float)) or not math.isfinite(slip) or not 0<=slip<.003):
                raise RuntimeError('Current bilateral bounded grasp required for acceleration')
            self.slips.append(float(slip))
            drift=(max(self.slips)-min(self.slips))/.1 if len(self.slips)==49 else None
            safe=bool(drift is not None and drift<.001 and slip<.0015 and max(loads.values())<.40)
            headroom=float(np.clip((.40-max(loads.values()))/.10,0,1))
        safe=safe and upper<.20
        self.stable=self.stable+1 if safe else 0
        self.cap=(.0003+(self.maximum-.0003)*headroom if self.stable>=96 else .0003)
        if not cut_only and (max(loads.values())>=.40 or slip>=.002):self.cap=0.
        self.step=step
        self.receipt=dict(model='support_aware_postrelease_feed_trial_v1',step=step,
            knife_load_n=upper,left_finger_all_contact_n=dict(loads),slip_m=slip,
            slip_window_excursion_rate_m_s=drift,stable_speed_control_dwell_s=self.stable/480,
            maximum_speed_m_s=self.maximum,measured_support_speed_cap_m_s=self.cap,
            cut_only=cut_only,force_limits_changed=False,cut_authorized=False,
            measured_completion=False,physical_speed_qualified=False)

    def command(self,requested,*,step,dt):
        if (step!=self.step or self.consumed==step or isinstance(dt,bool)
                or not math.isfinite(dt) or abs(dt-1/480)>1e-12
                or isinstance(requested,bool) or not math.isfinite(requested) or not -.0005<=requested<=.002):
            raise RuntimeError('Fresh unused bounded support-aware speed command required')
        # Slow/stop/reverse immediately. Only acceleration is slew-limited.
        target=min(requested,self.cap) if requested>0 else requested
        self.speed=min(target,max(0.,self.speed)+.001*dt) if target>0 else target
        self.consumed=step
        self.receipt.update(requested_speed_m_s=requested,command_speed_m_s=self.speed,
            maximum_acceleration_m_s2=.001)
        return self.speed
