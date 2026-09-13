"""Bounded finger drive targets from preceding, guard-accepted native contacts.

This is not a grasp detector. The existing opposing-contact, dwell, slip and
all-contact guards still decide whether a grasp or subsequent motion is valid.
"""
import numpy as np


class ForceClosure:
    # Drive effort excludes independently measured gravity feed-forward. The
    # original total-motor and 0.5 N all-contact budgets remain upper bounds.
    drive_limit_n = .15
    def __init__(self, radius, compression, *, retention_preload=False, symmetric=False, pregrasp_half_aperture=.025, effort_bounded_target=False):
        if type(effort_bounded_target) is not bool or effort_bounded_target and not (retention_preload and symmetric):
            raise ValueError('Effort-bounded target requires original symmetric retention profile')
        self.effort_bounded_target=effort_bounded_target
        if type(symmetric) is not bool:raise ValueError('Explicit symmetric aperture mode required')
        self.symmetric=symmetric
        if type(retention_preload) is not bool:
            raise ValueError('Explicit retention-preload profile required')
        self.retention_preload=retention_preload
        # Controller settings, NOT damage thresholds or contact authorization.
        # Higher-profile upper bound .24*(1+.5)=.36 N leaves margin below the
        # unchanged .5 N native all-contact guard. Actual friction/contact may
        # differ; that guard still runs before every release decision.
        self.desired_support_n=.24 if retention_preload else .12
        self.drive_limit_n=.30 if retention_preload else .15
        self.backoff_contact_n=.40 if retention_preload else .30
        if (not np.isfinite([radius, compression]).all() or not 0 < radius < .02
                or not .00025 <= compression <= .001):
            raise ValueError('Finite shaft radius and bounded compression required')
        # A drive reference inside a contact is NOT measured penetration.
        # With Kp=200, a 1 mm nominal reference bias can request only ~0.2 N
        # at the nominal shaft surface, below the existing 0.24 N target.
        # This opt-in permits only the existing 0.30 N / 200 N/m bias.
        # Actual native 1 mm penetration and all-contact guards are unchanged.
        self.nominal_target_bias=self.drive_limit_n/200 if effort_bounded_target else compression
        if effort_bounded_target and radius<=self.nominal_target_bias:
            raise ValueError('Effort-bounded reference cannot cross the zero-aperture joint limit')
        self.minimum = max(0., radius-self.nominal_target_bias)
        from .pregrasp_aperture import validate
        self.opening = validate(radius, pregrasp_half_aperture)
        self.slow_gap = min(self.opening, radius+.002)
        self.gaps = np.full(2, self.opening)
        self.support = np.zeros(2)
        self.loads = np.zeros(2)
        self.geometry_valid = True
        self.observed_step = 0
        self.commanded_step = None
        self.fraction = 0.
        self.receipt = None

    def command(self, fraction, *, step, dt):
        if (type(step) is not int or step < 0 or step != self.observed_step
                or self.commanded_step is not None and step != self.commanded_step+1):
            raise RuntimeError('Fresh preceding post-fetch contact required for closure')
        if (isinstance(fraction, (bool, np.bool_)) or not np.isfinite([fraction, dt]).all()
                or not self.fraction <= fraction <= 1 or abs(dt-1/240) > 1e-12):
            raise ValueError('Monotone closure and qualified 240 Hz step required')
        scheduled = self.opening-fraction*(self.opening-self.minimum)
        current=float(np.mean(self.gaps))
        if self.symmetric and abs(self.gaps[0]-self.gaps[1])>1e-9:
            raise RuntimeError('Symmetric aperture controller state lost its fixed center')
        # Fast free-space closure ends 2 mm outside the shaft. Every near-
        # contact increment is <=2.09 micrometres; measured contact on either
        # pad also forces slow motion even if the other pad is still clear.
        desired = np.full(2, scheduled)
        if self.symmetric:
            # Regulate one aperture, not two independent target translations.
            # Opposite pad loads may differ when supporting an external wrench.
            # This is a software command constraint, NOT a native gear/weld or
            # a physical pose constraint on either finger or the held object.
            if current<=self.slow_gap+1e-12 or np.max(self.loads)>.01:
                error=self.desired_support_n-float(np.mean(self.support))
                speed=0. if abs(error)<=.03 else float(np.clip(error/self.desired_support_n,-1,1))*.0005
                if not self.geometry_valid:speed=-.005 if np.max(self.loads)>.01 else 0.
                if np.max(self.loads)>self.backoff_contact_n:speed=-.005
                desired_gap=max(scheduled,current-speed*dt)
            else:desired_gap=max(scheduled,self.slow_gap) if self.geometry_valid else current
            desired[:]=desired_gap
        else:
            for i in range(2):
                if self.gaps[i] <= self.slow_gap+1e-12 or self.loads[i] > .01:
                    error = self.desired_support_n-self.support[i]
                    speed = 0. if abs(error) <= .03 else np.clip(error/self.desired_support_n, -1., 1.)*.0005
                    # Non-qualifying contact cannot authorize further squeeze.
                    if not self.geometry_valid: speed = -.005 if self.loads[i]>.01 else 0.
                    if self.loads[i] > self.backoff_contact_n: speed = -.005
                    desired[i] = max(scheduled, self.gaps[i]-speed*dt)
                else:
                    desired[i] = max(scheduled, self.slow_gap) if self.geometry_valid else self.gaps[i]
        self.gaps = np.clip(desired, self.minimum, self.opening)
        self.fraction = float(fraction)
        self.commanded_step = step
        self.receipt = dict(mode='native_force_closure_v1', command_step=step,
            effort_bounded_position_reference=self.effort_bounded_target,
            maximum_nominal_target_bias_m=self.nominal_target_bias,
            actual_native_penetration_guard_m=.001,
            commanded_pregrasp_half_aperture_m=self.opening,changes_physical_joint_limits=False,
            observation_step=self.observed_step, half_gaps_m=self.gaps.tolist(),
            preceding_compressive_support_n=self.support.tolist(),
            preceding_all_contact_upper_bound_n=self.loads.tolist(),
            desired_support_n=self.desired_support_n, near_contact_speed_limit_m_s=.0005,
            load_profile='retention_preload_v1' if self.retention_preload else 'legacy_preload_v1',
            controller_backoff_contact_n=self.backoff_contact_n,native_hard_contact_limit_n=.5,
            symmetric_aperture_command=self.symmetric,physical_gear_coupling_modeled=False,
            maximum_non_gravity_drive_effort_n=self.drive_limit_n,
            geometry_valid=self.geometry_valid,maximum_backoff_speed_m_s=.005,
            grasp_verified=False)
        return self.gaps.copy()

    def observe(self, contact, loads, *, step, guards_passed):
        if (guards_passed is not True or type(step) is not int
                or self.commanded_step is None or step != self.commanded_step+1
                or step != self.observed_step+1 or contact.get('step_id') != step
                or contact.get('adapter_valid') is not True
                or contact.get('normal_only') is not True):
            raise RuntimeError('Same-step validated native closure observation required')
        support = np.asarray(contact['compressive_support_n'], float)
        loads = np.asarray(loads, float)
        if (support.shape != (2,) or loads.shape != (2,)
                or not np.isfinite(np.r_[support, loads]).all()
                or np.any(loads < 0) or np.any(loads >= .5)
                or type(contact.get('stem_only')) is not bool):
            raise RuntimeError('Invalid or unsafe finger contact; closure refused')
        self.geometry_valid = contact['stem_only']
        self.support = support.copy() if self.geometry_valid else np.zeros(2)
        self.loads = loads.copy()
        self.observed_step = step
