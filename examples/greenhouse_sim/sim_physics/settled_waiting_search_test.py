from types import SimpleNamespace as S
from copy import deepcopy
import numpy as np
import pytest
from .settled_waiting_search import search,offsets


def case(monkeypatch):
    from . import startup_pose_search,held_plant_screen,settled_waiting_search
    frames=np.tile(np.eye(4),(2,1,1));calls=[]
    def screen(label):
        def check(world,**kw):
            assert kw=={'margin':.005};calls.append(label);return True
        return S(check=check,snapshot=lambda value:np.testing.assert_array_equal(value,frames),last_failure=None)
    rest=screen('rest');current=screen('current')
    monkeypatch.setattr(startup_pose_search,'waiting_target_screen',lambda r:rest)
    monkeypatch.setattr(held_plant_screen,'HeldPlantScreen',lambda *a:current)
    monkeypatch.setattr(settled_waiting_search,'swept_target_screen',lambda *a:screen('swept'))
    def forward(arm,q,base):
        value=np.eye(4);value[:3,3]=q[:3];return value
    def solve(arm,goal,*a,**k):
        assert k['joint_limit_margin_degrees']==3.
        q=np.zeros(7);q[:3]=goal[:3,3];return S(succeeded=True,joint_degrees=q)
    robot=S(rig=S(body_paths=['/P/0','/P/1'],rest_frames=frames.copy(),source_target='source/petiole'),
        held_plant_screen=S(workspace=(np.full(3,-1),np.ones(3)),static=[],static_indices={}),
        knife=S(collider='/Knife'),self_screen=S(shapes=['right_shapes']),
        initial_q=np.zeros(7),right=np.zeros(7),base=np.eye(4),path_q=[np.zeros(7),np.ones(7)],
        kin=S(forward=forward,solve_pose=solve,
            arm_limits_degrees=lambda *a:(np.full(7,-180),np.full(7,180)),
            inter_arm_clearance=lambda *a:S(clearance_m=.02)),
        check_self=lambda *a:dict(passed=True),body_world=lambda *a:{})
    record=dict(t=.9,native_guards_passed=True,cut=False,grasp_dynamics=dict(step_id=432,
        dt_s=1/480,body_paths=robot.rig.body_paths,body_frames_world_m=frames.tolist(),
        model='grasp_dynamics_post_fetch_telemetry_v1'))
    kwargs=dict(step=432,time_s=.9,physics_hz=480,guard=lambda:None,
        history=[(i,frames.copy()) for i in range(1,433)])
    return robot,frames,record,kwargs,rest,current,calls


def test_bounded_offsets_are_unique_and_metric():
    grid=offsets()
    assert len(grid)==35 and len({tuple(o) for o in grid})==35
    assert max(np.linalg.norm(o) for o in grid)<.07


def test_current_and_rest_checks_produce_only_restart_proposals(monkeypatch):
    r,f,record,kw,_,_,calls=case(monkeypatch);before=deepcopy(record);right=r.right.copy()
    out=search(r,f,record,**kw)
    assert len(out['proposed_waiting_poses'])==8 and calls==['rest','current','swept']*8
    assert not out['motion_authorized'] and not out['native_startup_certified']
    assert not out['settled_equilibrium_verified'] and not out['full_left_scene_path_checked']
    assert out['relaunch_and_all_native_checks_required'] and out['physics_steps_during_search']==0
    assert record==before;np.testing.assert_array_equal(r.right,right)


@pytest.mark.parametrize('bad',['step','time','guards','cut','paths','frames','dt','model'])
def test_unbound_observation_rejected(monkeypatch,bad):
    r,f,record,kw,*_=case(monkeypatch)
    if bad=='step':kw['step']-=1
    elif bad=='time':record['t']+=.01
    elif bad=='guards':record['native_guards_passed']=False
    elif bad=='cut':record['cut']=True
    elif bad=='paths':record['grasp_dynamics']['body_paths']=['/Wrong']
    elif bad=='frames':record['grasp_dynamics']['body_frames_world_m'][0][0][3]=.001
    elif bad=='dt':record['grasp_dynamics']['dt_s']=1/240
    else:record['grasp_dynamics']['model']='unknown'
    with pytest.raises(ValueError,match='Fresh'):search(r,f,record,**kw)


@pytest.mark.parametrize('bad',['rest','current','self','interarm','fk','limits'])
def test_any_geometric_failure_leaves_no_proposal(monkeypatch,bad):
    r,f,record,kw,rest,current,_=case(monkeypatch)
    if bad in ('rest','current'):(rest if bad=='rest' else current).check=lambda *a,**k:False
    elif bad=='self':r.check_self=lambda *a:dict(passed=False)
    elif bad=='interarm':r.kin.inter_arm_clearance=lambda *a:S(clearance_m=.005)
    else:r.kin.solve_pose=lambda *a,**k:S(succeeded=True,joint_degrees=np.full(7,1. if bad=='fk' else 179.))
    assert search(r,f,record,**kw)['proposed_waiting_poses']==[]


def test_final_guard_failure_returns_nothing(monkeypatch):
    r,f,record,kw,*_=case(monkeypatch);checks=[]
    def guard():
        checks.append(1)
        if len(checks)==51:raise RuntimeError('native sample advanced')
    kw['guard']=guard
    with pytest.raises(RuntimeError,match='advanced'):search(r,f,record,**kw)


def test_cli_rejects_unqualified_mode_before_output(tmp_path):
    from .benchmark import main
    from .ground_truth_trial import main as public
    out=tmp_path/'none'
    with pytest.raises(ValueError,match='Settled waiting'):
        main(['--output',str(out),'--screen-settled-waiting'])
    for extra in (['--mode','right_only'],['--watch'],['--screen-approach-start']):
        with pytest.raises(SystemExit):
            public(['--output',str(out),'--process-zone-trial','--milestone','cut_action',
                '--screen-settled-waiting',*extra])
    assert not out.exists()


@pytest.mark.parametrize('bad',['missing','duplicate','last','nonrigid'])
def test_incomplete_or_invalid_settling_history_rejected(monkeypatch,bad):
    r,f,record,kw,*_=case(monkeypatch)
    if bad=='missing':kw['history'].pop(0)
    elif bad=='duplicate':kw['history'][0]=(2,f.copy())
    elif bad=='last':kw['history'][-1][1][0,0,3]=.001
    else:kw['history'][12][1][0,0,0]=2
    with pytest.raises(ValueError):search(r,f,record,**kw)


def test_observed_swept_leaf_blocks_pose_that_both_endpoints_clear():
    from .held_plant_screen_test import fixture
    from .startup_pose_search import waiting_target_screen
    from .settled_waiting_search import swept_target_screen
    screen,rig,world=fixture('box')
    r=S(rig=rig,self_screen=S(shapes=screen.shapes),knife=S(collider=screen.blade_path))
    history=np.repeat(rig.rest_frames[None],3,axis=0)
    history[1,1,0,3]=.4
    world['link_right_arm_5'][0,3]=.4
    endpoint=waiting_target_screen(r)
    assert endpoint.check(world,margin=.005)
    before=rig.stage.GetRootLayer().ExportToString()
    swept=swept_target_screen(r,history)
    assert not swept.check(world,margin=.005)
    assert swept.last_failure['plant_collider'].endswith('/Leaf')
    assert rig.stage.GetRootLayer().ExportToString()==before
