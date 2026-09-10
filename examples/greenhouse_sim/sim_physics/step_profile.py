"""Diagnostic timings of the installed Isaac 6 physics-only step sequence."""
import time
import numpy as np


class MeasuredStep:
    def __init__(self,context):
        self.context=context
        self.physics=context.get_physics_context()
        self.samples={k:[] for k in ('stage','playing','time','dt','simulate')}

    def get_physics_dt(self): return self.context.get_physics_dt()
    def render(self): return self.context.render()

    def step(self,*,render=False):
        if render: raise ValueError('MeasuredStep is physics-only')
        start=time.perf_counter()
        if self.context.stage is None: raise RuntimeError('No physics stage')
        end=time.perf_counter();self.samples['stage'].append((end-start)*1000);start=end
        playing=self.context.is_playing()
        end=time.perf_counter();self.samples['playing'].append((end-start)*1000);start=end
        if not playing: raise RuntimeError('Diagnostic physics timeline is not playing')
        current=self.context.current_time
        end=time.perf_counter();self.samples['time'].append((end-start)*1000);start=end
        dt=self.physics.get_physics_dt()
        end=time.perf_counter();self.samples['dt'].append((end-start)*1000);start=end
        # Same interface and arguments as PhysicsContext._step; do not bypass
        # the physics manager, native contacts, events or tensor updates.
        self.physics._physics_sim_interface.simulate(dt,current)
        self.samples['simulate'].append((time.perf_counter()-start)*1000)

    def report(self):
        return {k:dict(samples=len(v),p50_ms=float(np.median(v)),
                       p95_ms=float(np.percentile(v,95)),maximum_ms=float(max(v)))
                for k,v in self.samples.items() if v}
