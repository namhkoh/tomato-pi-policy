from types import SimpleNamespace
import pytest
from sim_physics.step_profile import MeasuredStep


def test_measured_step_calls_the_same_interface_once():
    calls=[]
    native=SimpleNamespace(simulate=lambda dt,t:calls.append((dt,t)))
    physics=SimpleNamespace(get_physics_dt=lambda:1/240,_physics_sim_interface=native)
    context=SimpleNamespace(stage=object(),current_time=.25,is_playing=lambda:True,
        get_physics_context=lambda:physics,get_physics_dt=lambda:1/240,render=lambda:calls.append('render'))
    step=MeasuredStep(context);step.step()
    assert calls==[(1/240,.25)]
    assert all(v['samples']==1 for v in step.report().values())
    with pytest.raises(ValueError): step.step(render=True)
    context.is_playing=lambda:False
    with pytest.raises(RuntimeError): step.step()
    assert calls==[(1/240,.25)]
