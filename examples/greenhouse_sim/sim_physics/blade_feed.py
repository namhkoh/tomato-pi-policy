"""Diagnostic path feed from fresh, guard-accepted native blade contact.

Only retimes/reverses the already screened one-dimensional stroke. This is
not a contact detector, effort limit, release authorization or tissue model.
"""
import math
from collections import deque
import numpy as np


class BladeFeed:
    def __init__(self, offsets, *, radius, dwell_feedback=False, compliant_rate=False, friction_budget=False, physics_hz=240, loaded_advance=False, faster_cut=False):
        from .diagnostic_rate import frequency
        self.physics_hz=frequency(physics_hz)
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
        if type(dwell_feedback) is not bool:
            raise ValueError('Explicit boolean dwell feedback required')
        self.dwell_feedback=dwell_feedback
        self.loads=deque(maxlen=1+int(.025*self.physics_hz))
        if type(compliant_rate) is not bool:
            raise ValueError('Explicit isolated compliant-rate comparison required')
        self.compliant_rate=compliant_rate
        self.near_speed=.0003 if compliant_rate else .0001
        self.loading_speed=.0003 if compliant_rate else .00005
        self.load_gain=.00125 if compliant_rate else .0002
        if type(friction_budget) is not bool or friction_budget and not (compliant_rate and dwell_feedback):
            raise ValueError('Friction budget requires explicit compliant rate and minimum-window feedback')
        self.friction_budget=friction_budget
        if (type(loaded_advance) is not bool or loaded_advance and not
                (friction_budget and self.physics_hz==480)):
            raise ValueError('Loaded advance requires explicit 480 Hz compliant minimum-window friction-budget control')
        self.loaded_advance=loaded_advance
        self.loading_geometry_verified=False
        # Commanded progress is not measured penetration or fracture work.
        # Half the existing near-contact speed; no load threshold/cap increase.
        self.loaded_speed=.00015 if loaded_advance else 0.
        if type(faster_cut) is not bool or faster_cut and not loaded_advance:
            raise ValueError('Faster cut requires complete480Hz geometry-qualified loaded advance')
        self.faster_cut=faster_cut;self.free_speed=.002
        if faster_cut:
            # Explicit command-rate trial only. Keep the same slow-zone
            # boundary, raw force/geometry gates, dwell, backoff and timeout.
            self.free_speed=.010;self.near_speed=.001;self.loading_speed=.001
            self.loaded_speed=.00075;self.load_gain=.005
        # A controller setpoint equal to the .22 N entry boundary approaches
        # that boundary asymptotically and can get stuck below it after native
        # float32 quantization (native287: .219945 N). Aim INSIDE the existing
        # band. This changes only feed, never measured cut or hard-force gates.
        self.acquisition_target=.25 if loaded_advance else .22 if dwell_feedback else .26
        # Contact upper bound includes friction; it is not the signed normal
        # load tested by the cut gate. At mu=.5, .26 N normal may require
        # .39 N total. .40 N is a CONTROL backoff, not a raised .50 N guard.
        # The retained .10 N margin is not an analytical overshoot guarantee.
        self.backoff_load=.40 if friction_budget else .32

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
        geometry=False
        if self.loaded_advance:
            from .knife import DOWNWARD_CUT_MODEL
            geometry=knife.get('loading_geometry_verified')
            if knife.get('cut_model')!=DOWNWARD_CUT_MODEL or type(geometry) is not bool:
                raise RuntimeError('Current downward-rim geometry/support observation required')
        self.loading_geometry_verified=geometry
        self.normal=float(normal);self.upper=float(upper);self.released=released
        # Consecutive contact samples only. This window affects feed commands,
        # NEVER the cut detector's unsmoothed per-step force/direction evidence.
        if self.upper>.01:
            self.loads.append(self.normal)
        else:
            self.loads.clear()
        self.observed_step=step

    def command(self, *, step, dt):
        if (type(step) is not int or step<1 or step!=self.observed_step
                or self.commanded_step is not None and step!=self.commanded_step+1
                or self.released):
            raise RuntimeError('Fresh preceding intact-seam native sample required for blade feed')
        if isinstance(dt,(bool,np.bool_)) or not math.isfinite(dt) or abs(dt-1/self.physics_hz)>1e-12:
            raise ValueError('Matching diagnostic blade feed step required')
        if self.started_step is None:self.started_step=step
        if (step-self.started_step)*dt>=30:
            raise RuntimeError('Blade load acquisition timed out; no timed release')
        self.near=self.near or self.offset>=self.near_offset or self.upper>.01
        control_load=min(self.loads) if self.dwell_feedback and self.loads else self.normal
        if self.upper>self.backoff_load or self.normal>.28:
            speed=-.0005;mode='backoff_load'
        elif self.loaded_advance and self.upper>.01 and not self.loading_geometry_verified:
            speed=0.;mode='hold_unqualified_contact'
        elif .22<=control_load<=.28:
            speed=self.loaded_speed
            mode='loaded_downward_advance' if self.loaded_advance else 'hold_valid_load'
        elif self.upper>.10 and self.normal<.01:
            speed=0.;mode='hold_nonqualifying_load'
        elif self.upper>.01:
            speed=min(self.loading_speed,max(0.,(self.acquisition_target-control_load)*self.load_gain));mode='load_feedback'
        elif self.near:
            speed=self.near_speed;mode='near_contact'
        else:
            speed=self.free_speed;mode='free_space'
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
            desired_signed_load_n=self.acquisition_target,holding_load_band_n=[.22,.28],backoff_load_n=self.backoff_load,
            near_contact_speed_m_s=self.near_speed,loading_speed_limit_m_s=self.loading_speed,
            backoff_speed_m_s=.0005,cut_authorized=False,force_limit_guaranteed=False)
        self.receipt.update(compliant_rate_comparison=self.compliant_rate,
            faster_cut_trial=self.faster_cut,free_space_speed_limit_m_s=self.free_speed,
            force_or_geometry_limits_changed=False,
            loaded_advance_comparison=self.loaded_advance,
            current_loading_geometry_verified=self.loading_geometry_verified,
            loaded_speed_limit_m_s=self.loaded_speed,
            friction_budget_comparison=self.friction_budget,hard_full_contact_guard_n=.5,
            loading_gain_m_per_n_s=self.load_gain,
            maximum_loading_increment_m=self.loading_speed*dt,
            loading_profile_qualified=False)
        self.receipt.update(dwell_feedback=self.dwell_feedback,control_load_n=control_load,
            control_load_statistic=('minimum_last_seven_consecutive_contact_samples' if self.physics_hz==240
                else 'minimum_consecutive_samples_spanning_25ms') if self.dwell_feedback else 'latest_sample',
            control_window_capacity_samples=self.loads.maxlen,physics_hz=self.physics_hz,
            control_load_sample_count=len(self.loads) if self.dwell_feedback else 1,
            minimum_contact_load_target_n=self.acquisition_target if self.dwell_feedback else None,
            release_evidence_filtered=False)
        return (self.offset-self.start)/(self.end-self.start)
