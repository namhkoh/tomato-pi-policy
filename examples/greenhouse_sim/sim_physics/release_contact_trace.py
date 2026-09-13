"""Passive full-plant contact trace for isolated release diagnosis.

Observed impulses only: no reporting-completeness certificate, force controller,
collision filter or cut authority. Physics-step binding is caller asserted.
"""
import math

from .plant_contact_stream import PlantContactStream


class ReleaseContactTrace(PlantContactStream):
    def __init__(self, monitor, plant_colliders):
        if monitor.full_contact_observer is not None:
            raise ValueError('Never replace an existing native contact observer')
        if not monitor.native_full_contact_reporting:
            raise ValueError('Native normal and friction reporting required')
        super().__init__(plant_colliders)
        self._last_step = None
        monitor.full_contact_observer = self

    def measured_snapshot(self, *, step, dt, cut):
        if (type(step) is not int or step < 1 or type(cut) is not bool
                or type(dt) not in (int, float) or not math.isfinite(dt) or dt <= 0
                or self._last_step is not None and step != self._last_step+1):
            raise ValueError('Adjacent caller-bound physics steps and finite dt required')
        self._healthy()
        result = self.snapshot()
        pairs = {}
        for row in result['rows']:
            key = tuple(sorted((row['collider0'], row['collider1'])))
            p = pairs.setdefault(key, dict(normal_impulse_upper_ns=0., friction_impulse_upper_ns=0.,
                                          minimum_separation_m=None, normal_rows=0, friction_rows=0))
            kind = row['kind']
            p[kind+'_rows'] += 1
            p[kind+'_impulse_upper_ns'] += math.hypot(*row['impulse_on_0_ns'])
            if kind == 'normal':
                s = row['separation_m']
                p['minimum_separation_m'] = s if p['minimum_separation_m'] is None else min(s, p['minimum_separation_m'])
        summary = []
        for key, p in sorted(pairs.items()):
            upper = (p['normal_impulse_upper_ns']+p['friction_impulse_upper_ns'])/dt
            if not math.isfinite(upper):
                raise ValueError('Unrepresentable observed contact upper bound')
            summary.append(dict(collider_paths=list(key), observed_force_upper_n=upper, **p))
        result.update(model='passive_release_contact_trace_v1', caller_step=step, dt_s=dt,
                      released_at_fetch=cut, pair_summary=summary,
                      contact_generation_basis='caller_bound_post_fetch_not_native_generation_timestamp',
                      contact_work_measured=False, contact_law_or_control_changed=False)
        self._last_step = step
        return result
