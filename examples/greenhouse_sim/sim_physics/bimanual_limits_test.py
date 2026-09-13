from copy import deepcopy
import pytest
from sim_physics.bimanual_limits import violations


def record():
    return dict(max_speed_m_s=0.,max_gripper_net_contact_n=0.,support_error_m=0.,
        contact=dict(min_separation=0.),robot=dict(allowed_tool_contact_n=0.,minimum_tool_separation_m=0.),slip_m=None)


FIELDS=[('max_speed_m_s',20.),('max_gripper_net_contact_n',3.),('support_error_m',1e-5),
        ('slip_m',.003),('contact.min_separation',-.001),
        ('robot.allowed_tool_contact_n',.5),('robot.minimum_tool_separation_m',-.001)]


def put(r,path,value):
    keys=path.split('.')
    for key in keys[:-1]:r=r[key]
    r[keys[-1]]=value


@pytest.mark.parametrize('path,limit',FIELDS)
def test_original_strict_bounds_and_no_mutation(path,limit):
    r=record();put(r,path,limit);before=deepcopy(r)
    assert violations(r)==[] and r==before
    put(r,path,limit*1.001)
    failures=violations(r)
    assert len(failures)==1 and failures[0]['reason']=='limit_exceeded'
    assert failures[0]['limit']==limit


@pytest.mark.parametrize('path,limit',FIELDS)
@pytest.mark.parametrize('bad',[float('nan'),float('inf'),float('-inf'),True,'0'])
def test_invalid_values_never_look_safe_or_emit_nonfinite_json(path,limit,bad):
    import json
    r=record();put(r,path,bad)
    failures=violations(r)
    assert len(failures)==1 and failures[0]['reason']=='invalid_measurement'
    json.dumps(failures,allow_nan=False)


def test_reports_all_failures_without_hiding_first_or_missing_fields():
    r=record()
    for path,limit in FIELDS:put(r,path,limit*2)
    assert len(violations(r))==7
    del r['contact']
    with pytest.raises(KeyError):violations(r)


@pytest.mark.parametrize('path',['max_speed_m_s','max_gripper_net_contact_n','support_error_m','slip_m','robot.allowed_tool_contact_n'])
def test_negative_magnitudes_fail(path):
    r=record();put(r,path,-1e-10)
    assert violations(r)[0]['reason']=='invalid_measurement'
