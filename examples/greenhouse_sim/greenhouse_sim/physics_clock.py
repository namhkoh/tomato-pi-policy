"""One explicit physics step per tick. Counters are NOT native camera IDs."""
from collections import deque
from dataclasses import dataclass
import math
import time

import numpy as np


def advance_one_physics_step(context, *, render=False):
    context.step(render=False)
    if render:
        context.render()


@dataclass(frozen=True)
class PhysicsStamp:
    episode: int
    step: int
    simulation_time_s: float


class PhysicsClock:
    """Fixed-dt control with separate bounded physics/control and render timing."""

    def __init__(self, context, *, physics_hz=240, render_hz=30, window=4096):
        if (type(physics_hz) is not int or type(render_hz) is not int
                or physics_hz <= 0 or render_hz < 0
                or (render_hz and physics_hz % render_hz)
                or type(window) is not int or window < 1):
            raise ValueError('Positive physics rate and integer-divisor render rate required')
        self.context = context
        self.dt = 1.0 / physics_hz
        self.render_stride = physics_hz // render_hz if render_hz else 0
        if hasattr(context, 'get_physics_dt') and not math.isclose(
                float(context.get_physics_dt()), self.dt, rel_tol=1e-6, abs_tol=1e-9):
            raise ValueError('Context physics dt differs from controller clock')
        self.episode = 0
        self.step_index = 0
        self.samples = {k: deque(maxlen=window) for k in (
            'physics_control_ms', 'before_step_ms', 'native_step_ms', 'after_step_ms', 'render_ms')}
        self.total_wall_s = 0.0
        self.render_count = 0

    @property
    def stamp(self):
        return PhysicsStamp(self.episode, self.step_index, self.step_index * self.dt)

    def tick(self, *, before=None, after=None, before_render=None):
        start = time.perf_counter()
        if before is not None:
            before(self.stamp, self.dt)
        before_end = time.perf_counter()
        advance_one_physics_step(self.context)
        physics_end = time.perf_counter()
        self.step_index += 1
        if after is not None:
            after(self.stamp, self.dt)
        end = time.perf_counter()
        self.samples['before_step_ms'].append((before_end-start)*1000)
        self.samples['native_step_ms'].append((physics_end-before_end)*1000)
        self.samples['after_step_ms'].append((end-physics_end)*1000)
        self.samples['physics_control_ms'].append((end-start)*1000)
        if self.render_stride and self.step_index % self.render_stride == 0:
            if before_render is not None:
                before_render(self.stamp)
            self.context.render()
            self.samples['render_ms'].append((time.perf_counter()-end)*1000)
            self.render_count += 1
        self.total_wall_s += time.perf_counter()-start
        return self.stamp

    def reset_epoch(self):
        """Call AFTER physical reset. Does not move bodies or certify frames."""
        self.episode += 1
        self.step_index = 0
        self.total_wall_s = 0.0
        self.render_count = 0
        for samples in self.samples.values():
            samples.clear()

    def report(self):
        timing={}
        for key,samples in self.samples.items():
            timing[key]=dict(samples=len(samples), **(
                dict(zip(('p50','p95','p99','max'),
                         map(float,np.percentile(samples,[50,95,99,100])),strict=True))
                if samples else {}))
        return dict(physics_steps=self.step_index,simulated_seconds=self.stamp.simulation_time_s,
                    tick_wall_seconds=self.total_wall_s,renders=self.render_count,
                    real_time_factor=self.stamp.simulation_time_s/self.total_wall_s if self.total_wall_s else None,
                    timing=timing,native_sensor_synchronization_verified=False)
