"""Left native-refinement lifecycle and conservative scope; not native proof."""
from types import SimpleNamespace as S
import numpy as np
import pytest
from sim_physics.bimanual import BimanualRobot


@pytest.mark.parametrize('link,suffix',[
    ('ee_finger_l1','/restored_collisions/contact_proxy'),
    ('ee_finger_l2','/restored_collisions/contact_proxy'),
    ('ee_left','/restored_collisions/contact_proxy'),
    ('ee_left','/attachments/LeftWristCamera/BracketCollision')])
def test_left_native_miss_refines_only_static_bounds_and_keeps_whole_box(link,suffix):
    from sim_physics.held_plant_screen_test import fixture
    screen,rig,_=fixture('box');screen.arm='left'
    _,_,_,kind,data=screen.shapes[0];body='/World/R/'+link
    screen.shapes=[(body+suffix,body,link,kind,data)]
    frames=rig.rest_frames.copy();frames[1,1,3]=.5
    screen.static=[('/World/Static','box',(np.array([.1,0,0]),np.eye(3),np.full(3,.03)),
                    np.array([.07,-.03,-.03]),np.array([.13,.03,.03]))]
    screen.snapshot(frames);world={link:np.eye(4)};calls=[]
    assert not screen.check(world)
    screen.native_static_query=S(clear_box_checked=lambda *a:calls.append(a) or True)
    assert screen.check(world)
    assert calls[-1][0]=='/World/Static' and calls[-1][-1]==.001
    np.testing.assert_array_equal(calls[-1][3],data[2])
    screen.native_static_query=S(clear_box_checked=lambda *a:False)
    assert not screen.check(world)  # Actual native hit never cleared.
    screen.native_static_query=S(clear_box_checked=lambda *a:True)
    screen.snapshot(rig.rest_frames)
    assert not screen.check(world)  # Dynamic leaf is not a static box.
    screen.snapshot(frames)
    screen.shapes=[(body+suffix+'_lookalike',body,link,kind,data)]
    if '/attachments/' not in suffix:
        screen.native_static_query=S(clear_box_checked=lambda *a:pytest.fail('unknown pad queried'))
        assert not screen.check(world)


def harness(monkeypatch,fault=None):
    import sim_physics.held_plant_screen as h
    import sim_physics.native_static_clearance as n
    calls=[]
    def action(name):
        calls.append(name)
        if fault==name:raise RuntimeError('injected '+name)
    native=S(validate=lambda:action('validate'),close=lambda:action('close'),
             report=lambda:dict(closed='close' in calls,final_validation_passed='validate' in calls))
    screen=S(snapshot=lambda frames:None,last_failure={'reason':'synthetic_obstacle'},
             check=lambda *a,**kw:False if fault=='geometry' else action('check') is None)
    def factory(*a,**kw):
        assert kw['lazy_coverage'] is True
        action('initialize');return native
    monkeypatch.setattr(h,'HeldPlantScreen',lambda *a,**kw:screen)
    monkeypatch.setattr(n,'current_scene_query',factory)
    r=BimanualRobot.__new__(BimanualRobot)
    r.held_plant_screen=S(workspace=(np.zeros(3),np.ones(3)),static=[],static_indices={})
    r.rig=object();r.self_screen=S(shapes=[]);r.knife=S(collider='/Knife');r.grasp_path='/Shaft'
    r.slides={'gripper_finger_l1':-.025,'gripper_finger_l2':.025};r.radius=.003;r.grasp_compression=.0005
    r.fractions=np.array([0.,1.,1.025]);r.path_q=np.zeros((3,7));r.right=np.zeros(7)
    r.body_world=lambda *a:{};r.stage=object();r.robot=object()
    r.native_static_clearance=True;r.native_static_planning_seconds=8.
    return r,calls,screen


def test_grasp_native_acceptance_validates_closes_and_restores_proposal_aperture(monkeypatch):
    r,calls,screen=harness(monkeypatch);old={'prior':True};r.planning_slides=old
    result=r.screen_grasp_scene(np.empty(0))
    assert result['passed'] and not result['native_grasp_verified']
    assert calls[-2:]==['validate','close']
    assert r.planning_slides is old and screen.native_static_query is None


@pytest.mark.parametrize('fault',['initialize','check','validate','close'])
def test_any_native_grasp_query_failure_prevents_acceptance_and_cleans_up(monkeypatch,fault):
    r,calls,screen=harness(monkeypatch,fault)
    with pytest.raises(RuntimeError):r.screen_grasp_scene(np.empty(0))
    assert not r.grasp_scene_screen['passed'] and not hasattr(r,'planning_slides')
    if fault!='initialize':
        assert calls.count('close')==1 and screen.native_static_query is None


def test_geometric_grasp_rejection_stays_rejected_and_closes_query(monkeypatch):
    r,calls,screen=harness(monkeypatch,'geometry')
    result=r.screen_grasp_scene(np.empty(0))
    assert not result['passed'] and result['failure']['reason']=='synthetic_obstacle'
    assert 'validate' not in calls and calls.count('close')==1
    assert screen.native_static_query is None
