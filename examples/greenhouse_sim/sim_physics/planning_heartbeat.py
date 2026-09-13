"""Render-only UI service during synchronous frozen-scene planning.

The caller must bracket this with its epoch guard. A render callback is NOT
allowed to confer clearance or advance physics; any scene/timeline/native
change must invalidate the enclosing query and its cached results.
"""
import math
import time


class FrozenTimeline:
    """Temporarily stop automatic animation time, NEVER set/rewind its value.

    commit() is the documented main-thread immediate-state operation. The
    enclosing native epoch must already exist and check after acquire/close;
    pending user edits committed here must not become accepted new snapshots.
    """
    def __init__(self,timeline):
        self.timeline=timeline;self.old=timeline.is_auto_updating();self.closed=False
        if type(self.old) is not bool:raise RuntimeError('Explicit timeline auto-update state required')
        try:
            timeline.set_auto_update(False);timeline.commit();self.check()
        except BaseException:
            self.close();raise

    def check(self):
        if self.closed or self.timeline.is_auto_updating() is not False:
            raise RuntimeError('Planning timeline auto-update changed')

    def close(self):
        if self.closed:return
        try:
            self.timeline.set_auto_update(self.old);self.timeline.commit()
            if self.timeline.is_auto_updating() is not self.old:
                raise RuntimeError('Planning timeline auto-update restoration failed')
        finally:self.closed=True


class PlanningHeartbeat:
    def __init__(self,render,healthy,*,clock=time.monotonic):
        if not all(callable(v) for v in (render,healthy,clock)):
            raise ValueError('Render-only callback and liveness guard required')
        self.render,self.healthy,self.clock=render,healthy,clock
        self.last=None;self.rendering=False;self.renders=0;self.render_seconds=0.

    def __call__(self):
        if self.rendering:raise RuntimeError('Reentrant planning render')
        if self.healthy() is not True:raise RuntimeError('Planning cancelled or app closed')
        now=self.clock()
        if not math.isfinite(now):raise RuntimeError('Invalid planning clock')
        if self.last is not None and now-self.last<1/15:return
        self.rendering=True
        try:
            self.render();self.renders+=1
            if self.healthy() is not True:raise RuntimeError('Planning cancelled or app closed')
        finally:
            self.rendering=False
            end=self.clock();self.last=end;self.render_seconds+=max(0.,end-now)

    def report(self):
        return dict(model='frozen_plan_render_only_heartbeat_v1',maximum_render_hz=15,
            renders=self.renders,render_wall_s=self.render_seconds,physics_steps_requested=0,
            requires_enclosing_epoch_guard=True,planning_budget_extended=False)
