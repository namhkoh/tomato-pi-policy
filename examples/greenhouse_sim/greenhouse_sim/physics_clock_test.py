import pytest

from greenhouse_sim.physics_clock import PhysicsClock


class Context:
    def __init__(self): self.events=[]
    def step(self, *, render): self.events.append(('step',render))
    def render(self): self.events.append(('render',))
    def get_physics_dt(self): return 1/240


def test_contact_each_tick_render_only_at_scheduled_boundaries():
    context=Context(); clock=PhysicsClock(context,render_hz=30)
    contacts=[]; visuals=[]
    for _ in range(24):
        clock.tick(after=lambda s,dt: contacts.append((s.step,dt)),
                   before_render=lambda s: visuals.append(s.step))
    assert contacts==[(n,1/240) for n in range(1,25)]
    assert visuals==[8,16,24]
    assert context.events.count(('step',False))==24
    assert context.events.count(('render',))==3
    assert clock.stamp.simulation_time_s==.1


def test_headless_reset_bounded_timing_and_no_native_frame_claim():
    clock=PhysicsClock(Context(),render_hz=0,window=2)
    for _ in range(5): clock.tick()
    assert clock.report()['timing']['physics_control_ms']['samples']==2
    assert clock.report()['native_sensor_synchronization_verified'] is False
    clock.reset_epoch()
    assert clock.stamp.episode==1 and clock.stamp.step==0
    assert not clock.report()['timing']['physics_control_ms']['samples']


@pytest.mark.parametrize('kwargs',[dict(physics_hz=0),dict(render_hz=29),dict(window=0),dict(physics_hz=True)])
def test_bad_rates_rejected(kwargs):
    with pytest.raises(ValueError): PhysicsClock(Context(),**kwargs)


def test_wrong_context_timestep_rejected():
    c=Context(); c.get_physics_dt=lambda:1/60
    with pytest.raises(ValueError): PhysicsClock(c)
