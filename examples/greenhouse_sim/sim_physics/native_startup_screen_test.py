"""Injected native lifecycle contracts; these tests do not simulate a cut."""
from types import SimpleNamespace as S
import numpy as np
import pytest
from pxr import Usd, UsdGeom, UsdPhysics, Gf
from sim_physics.native_startup_screen import _screen
from sim_physics.benchmark import parser, main


def fixture():
    stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    UsdPhysics.Scene.Define(stage,'/World/QualificationPhysics')
    body=UsdGeom.Xform.Define(stage,'/World/RBY1/arm');body.AddTranslateOp().Set(Gf.Vec3d(0))
    plant=UsdGeom.Xform.Define(stage,'/World/Plant');plant.AddTranslateOp().Set(Gf.Vec3d(2,0,0))
    robot_path='/World/RBY1/arm/collider';scene_path='/World/Plant/collider'
    for path in (robot_path,scene_path):
        cube=UsdGeom.Cube.Define(stage,path);cube.CreateSizeAttr(.1)
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    state=S(time=0.,playing=False,steps=[],closed=0,epoch=None,missing=None,hit=False,
        bad_count=False,unknown=False,load=None,loads=0,query_hook=None,queries=0)
    def subscribe(**kw):
        state.steps.append(kw['fn'])
        return S(unsubscribe=lambda:setattr(state,'closed',state.closed+1))
    def load():
        state.loads+=1
        if state.load:state.load()
    def overlap_box(half,centre,quat,callback,any_hit):
        state.queries+=1
        candidates=[]
        for path,c in ((robot_path,np.zeros(3)),(scene_path,np.array([2.,0,0]))):
            if path==state.missing:continue
            if np.all(np.abs(np.asarray(centre)-c)<=np.asarray(half)+.05):candidates.append(path)
        if state.hit and scene_path not in candidates:candidates.append(scene_path)
        if state.unknown:candidates.append('/Unknown/shape')
        for path in candidates:callback(S(collision=path))
        if state.query_hook:state.query_hook()
        return len(candidates)+int(state.bad_count)
    class Epoch:
        def __init__(self,*args):self.changed=False;self.closed=False;state.epoch=self
        def check(self):
            if self.changed:raise RuntimeError('epoch changed')
        def close(self):self.closed=True
        def report(self):return dict(closed=self.closed)
    screen=S(shapes=[(robot_path,'/World/RBY1/arm','arm','box',(np.zeros(3),np.eye(3),np.full(3,.05)))],
        check=lambda *a:dict(passed=True))
    robot=S(root='/World/RBY1',body_paths=['/World/RBY1/arm'],rig=S(body_paths=['/World/Plant']),
        self_screen=screen,floor_root=None)
    context=S(get_stage=lambda:stage)
    timeline=S(is_stopped=lambda:not state.playing,is_playing=lambda:state.playing,get_current_time=lambda:state.time)
    physx=S(subscribe_physics_on_step_events=subscribe,force_load_physics_from_usd=load)
    query=S(overlap_box=overlap_box,overlap_sphere=lambda *a:pytest.fail('box fixture must not use spheres'))
    return (stage,robot,context,timeline,physx,query,Epoch),state


def test_complete_actor_controls_zero_steps_no_source_change_and_cleanup():
    args,s=fixture();stage=args[0];before=stage.GetRootLayer().ExportToString()
    result=_screen(*args)
    assert result['passed'] and result['physics_steps']==0 and not result['motion_authorized']
    assert result['initial_robot_actor_controls']==result['final_robot_actor_controls']==1
    assert result['native']['initial_positive_controls']==result['native']['final_positive_controls']==1
    assert result['native']['final_validation_passed'] and s.epoch.closed and s.closed==1
    assert stage.GetRootLayer().ExportToString()==before and s.loads==1


@pytest.mark.parametrize('bad',['playing','time','units','scene','steps','pose','scene_actor','robot_actor',
    'callback_count','unknown','epoch','query_pose','self','overlap','inventory'])
def test_invalid_startup_never_gets_a_clearance(bad):
    args,s=fixture();stage,robot=args[:2]
    if bad=='playing':s.playing=True
    if bad=='time':s.time=1.
    if bad=='units':UsdGeom.SetStageMetersPerUnit(stage,.01)
    if bad=='scene':UsdPhysics.Scene.Define(stage,'/World/SecondPhysics')
    if bad=='steps':s.load=lambda:[fn(.01) for fn in s.steps]
    if bad=='pose':s.load=lambda:stage.GetPrimAtPath('/World/RBY1/arm').GetAttribute('xformOp:translate').Set(Gf.Vec3d(.001,0,0))
    if bad=='scene_actor':s.missing='/World/Plant/collider'
    if bad=='robot_actor':s.missing='/World/RBY1/arm/collider'
    if bad=='callback_count':s.bad_count=True
    if bad=='unknown':s.unknown=True
    if bad=='epoch':s.query_hook=lambda:setattr(s.epoch,'changed',True)
    if bad=='query_pose':s.query_hook=lambda:stage.GetPrimAtPath('/World/Plant').GetAttribute('xformOp:translate').Set(Gf.Vec3d(2.0001,0,0))
    if bad=='self':robot.self_screen.check=lambda *a:dict(passed=False)
    if bad=='overlap':s.hit=True
    if bad=='inventory':robot.self_screen.shapes=[]
    result=_screen(*args)
    assert not result['passed'] and not result['motion_authorized']
    assert s.closed==len(s.steps)
    if s.epoch:assert s.epoch.closed
    if bad in ('playing','time','units','scene'):assert s.loads==0


def test_native_control_lost_after_check_rejects():
    args,s=fixture()
    def hook():
        if s.queries==3:s.missing='/World/Plant/collider'
    s.query_hook=hook
    result=_screen(*args)
    assert not result['passed'] and 'Missing native actor' in result['error']


def test_loading_quantization_is_recorded_but_not_confused_with_query_writes():
    args,s=fixture()
    s.load=lambda:args[0].GetPrimAtPath('/World/RBY1/arm').GetAttribute('xformOp:translate').Set(Gf.Vec3d(1e-7,0,0))
    result=_screen(*args)
    assert result['passed'] and result['maximum_loading_matrix_element_change']==1e-7
    assert result['maximum_authored_pose_change_during_queries']==0


def test_native_startup_remains_explicit_and_diagnostic_only(tmp_path):
    assert not parser().parse_args(['--output','unused']).native_startup_clearance
    with pytest.raises(ValueError,match='isolated cut contact'):
        main(['--output',str(tmp_path/'unused'),'--native-startup-clearance'])
    assert not (tmp_path/'unused').exists()


def test_failed_final_controls_revoke_every_grasp_proposal(monkeypatch):
    from . import startup_pose_search
    args,state=fixture();args[1].startup_right_pose_search=True
    def search(*unused):
        state.missing='/World/Plant/collider'
        return dict(proposed_grasp={'left':[1]*7},proposed_grasps=[{'left':[1]*7}])
    monkeypatch.setattr(startup_pose_search,'search',search)
    result=_screen(*args);out=result['right_pose_search']
    assert out['proposed_grasp'] is None and out['proposed_grasps'] is None
    assert not out['final_native_controls_passed'] and not result['passed']
