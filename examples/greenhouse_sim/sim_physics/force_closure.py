"""Bounded finger drive targets from preceding, guard-accepted native contacts.

This is not a grasp detector. The existing opposing-contact, dwell, slip and
all-contact guards still decide whether a grasp or subsequent motion is valid.
"""
import numpy as np


class ForceClosure:
    # Drive effort excludes independently measured gravity feed-forward. The
    # original total-motor and 0.5 N all-contact budgets remain upper bounds.
    drive_limit_n = .15
    def __init__(self, radius, compression):
        if (not np.isfinite([radius, compression]).all() or not 0 < radius < .02
                or not .00025 <= compression <= .001):
            raise ValueError('Finite shaft radius and bounded compression required')
        self.minimum = max(0., radius-compression)
        self.slow_gap = min(.025, radius+.002)
        self.gaps = np.full(2, .025)
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
        scheduled = .025-fraction*(.025-self.minimum)
        # Fast free-space closure ends 2 mm outside the shaft. Every near-
        # contact increment is <=2.09 micrometres; measured contact on either
        # pad also forces slow motion even if the other pad is still clear.
        desired = np.full(2, scheduled)
        for i in range(2):
            if self.gaps[i] <= self.slow_gap+1e-12 or self.loads[i] > .01:
                error = .12-self.support[i]
                speed = 0. if abs(error) <= .03 else np.clip(error/.12, -1., 1.)*.0005
                # A non-qualifying contact cannot authorize further squeeze.
                # Back away under the same effort cap and reobserve; stable
                # opposing contact is still required before right-arm motion.
                if not self.geometry_valid: speed = -.005 if self.loads[i]>.01 else 0.
                if self.loads[i] > .30: speed = -.005
                desired[i] = max(scheduled, self.gaps[i]-speed*dt)
            else:
                desired[i] = max(scheduled, self.slow_gap) if self.geometry_valid else self.gaps[i]
        self.gaps = np.clip(desired, self.minimum, .025)
        self.fraction = float(fraction)
        self.commanded_step = step
        self.receipt = dict(mode='native_force_closure_v1', command_step=step,
            observation_step=self.observed_step, half_gaps_m=self.gaps.tolist(),
            preceding_compressive_support_n=self.support.tolist(),
            preceding_all_contact_upper_bound_n=self.loads.tolist(),
            desired_support_n=.12, near_contact_speed_limit_m_s=.0005,
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
