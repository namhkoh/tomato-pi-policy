import pytest
from sim_physics.contact_events import ContactEvents


def monitor():
    return ContactEvents(robot_root='/World/R',target_root='/World/Target',
        fingers=['/World/R/finger'],floor_root='/World/Floor')


def test_contact_accounting_does_not_cancel_opposing_forces():
    m=monitor();m.consume('/World/R/arm/shape','/World/Gutter',[(.001,0,0),(-.001,0,0)])
    assert m.measurements(.01)['unwanted_contact_n']==pytest.approx(.2)
    m.begin_step();assert m.measurements(.01)['unwanted_contact_n']==0


def test_only_target_finger_and_floor_support_are_allowed():
    m=monitor()
    m.consume('/World/Target/Stem','/World/R/finger/shape',[(.001,0,0)])
    m.consume('/World/Floor/mesh','/World/R/wheel_l/shape',[(0,0,1)])
    assert m.measurements(.01)['unwanted_contact_n']==0
    assert m.measurements(.01)['allowed_target_contact_n']==pytest.approx(.1)
    m.consume('/World/Neighbor/Leaf','/World/R/finger/shape',[(.002,0,0)])
    assert m.measurements(.01)['unwanted_contact_n']==pytest.approx(.2)


def test_self_contact_and_malformed_data_fail_closed():
    m=monitor();m.consume('/World/R/left/shape','/World/R/right/shape',[(.01,0,0)])
    assert m.measurements(.01)['self_contact_n']==1.
    with pytest.raises(ValueError): m.consume('/World/R/a','/World/Gutter',[(float('nan'),0,0)])
    with pytest.raises(ValueError): m.measurements(0)
    m.error='bad native buffer'
    m.begin_step()  # A callback fault stays latched until explicit rebind.
    with pytest.raises(RuntimeError): m.measurements(.01)
