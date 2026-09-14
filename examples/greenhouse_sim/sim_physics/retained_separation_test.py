from types import SimpleNamespace
import numpy as np
import pytest
from .retained_separation import Separation,direction,prerequisites,branch_screen


def sample(step=20,amount=0.,**kwargs):
    palm=np.eye(4);palm[0,3]=amount
    r=dict(t=step/480,cut=True,native_guards_passed=True,contact=dict(bilateral=True),
        slip_m=.001,knife=dict(tool_contact_upper_bound_n=.25),
        robot=dict(max_finger_contact_upper_bound_n=.3),palm=palm.tolist())
    r.update(kwargs);return r


def controller():return Separation(np.zeros((17,7)),np.eye(4),[.002,0,0],np.eye(4),step=20)


def test_normal_points_to_distal_material_not_into_parent():
    e=np.array([[0,0,-1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1.]])
    assert np.array_equal(direction(e,[1,0,0]),[1,0,0])
    assert np.array_equal(direction(e,[-1,0,0]),[-1,0,0])
    with pytest.raises(ValueError):direction(e,[0,1,0])


@pytest.mark.parametrize('change',[dict(cut=False),dict(native_guards_passed=False),
    dict(contact=dict(bilateral=False)),dict(slip_m=.003),dict(slip_m=float('nan')),
    dict(knife=dict(tool_contact_upper_bound_n=.501)),dict(robot=dict(max_finger_contact_upper_bound_n=.5)),
    dict(t=21/480),dict(slip_m=True)])
def test_bad_native_evidence_never_authorizes_target_motion(change):
    with pytest.raises(RuntimeError):prerequisites(sample(**change),20)


@pytest.mark.parametrize('change',[dict(slip_m=.0022),dict(knife=dict(tool_contact_upper_bound_n=.41)),
    dict(robot=dict(max_finger_contact_upper_bound_n=.41))])
def test_separation_holds_below_unchanged_hard_limits(change):
    s=controller();s.observe(sample(**change),step=20,target=np.eye(4));s.command(step=20,dt=1/480)
    assert s.offset==0 and s.receipt['command_speed_m_s']==0


def test_fresh_command_bounded_and_single_use():
    s=controller();s.observe(sample(),step=20,target=np.eye(4));s.command(step=20,dt=1/480)
    assert s.offset==pytest.approx(.0005/480)
    with pytest.raises(RuntimeError):s.command(step=20,dt=1/480)
    with pytest.raises(RuntimeError):s.observe(sample(),step=20,target=np.eye(4))


@pytest.mark.parametrize('dt',[True,float('nan'),float('inf'),1/240])
def test_bad_control_period_rejected(dt):
    s=controller();s.observe(sample(),step=20,target=np.eye(4))
    with pytest.raises(RuntimeError):s.command(step=20,dt=dt)


def test_command_endpoint_is_not_material_movement():
    s=controller();s.offset=.002
    for step in range(20,34):
        s.observe(sample(step,.002),step=step,target=np.eye(4));s.command(step=step,dt=1/480)
    assert not s.complete


def test_both_palm_and_target_motion_and_dwell_required():
    s=controller();s.offset=.002;target=np.eye(4);target[0,3]=.0018
    for step in range(20,32):
        s.observe(sample(step,.002),step=step,target=target);s.command(step=step,dt=1/480)
        assert s.complete==(step==31)
    assert not s.receipt['commanded_motion_used_as_completion']


def test_no_teleported_target_or_off_axis_palm():
    s=controller();target=np.eye(4);target[0,3]=.004
    with pytest.raises(RuntimeError):s.observe(sample(),step=20,target=target)
    s=controller();r=sample();r['palm'][1][3]=.0006
    with pytest.raises(RuntimeError):s.observe(r,step=20,target=np.eye(4))


def test_only_released_adjacent_shaft_pair_gets_planning_allowance():
    local=[('/rig/0/StemCollider',0,'capsule',([0,0,0],[0,0,1],.003)),
        ('/rig/1/StemCollider',1,'capsule',([0,0,0],[0,0,1],.003)),
        ('/rig/1/Leaf',1,'hull',(np.array([[0,0,0],[1,1,1]]),None,None))]
    original=SimpleNamespace(local=local,shapes=['original'],workspace='unchanged')
    b=branch_screen(original,SimpleNamespace(cut_index=1,body_paths=['/rig/0','/rig/1']))
    assert b.seam_paths=={'/rig/0/StemCollider'} and b.blade_path=='/rig/1/StemCollider'
    assert len(b.shapes)==2 and b.shapes[1][3]=='box' and len(b.local)==1
    assert original.shapes==['original'] and original.workspace=='unchanged'


@pytest.mark.parametrize('protected_leaf',[False,True])
def test_actual_branch_screen_rejects_protected_static_or_parent_leaf(protected_leaf):
    from .held_plant_screen_test import fixture
    from pxr import UsdGeom,UsdPhysics
    original,rig,_=fixture('box')
    path=('/World/P/Support/ProtectedLeaf' if protected_leaf else '/World/Obstacle')
    if protected_leaf:
        import itertools
        obstacle=UsdGeom.Mesh.Define(rig.stage,path)
        obstacle.CreatePointsAttr(list(itertools.product((.09,.11),(-.01,.01),(-.01,.01))))
        obstacle.CreateFaceVertexCountsAttr([3]);obstacle.CreateFaceVertexIndicesAttr([0,1,2])
        UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(obstacle.GetPrim()).CreateApproximationAttr('convexHull')
        from .held_plant_screen import HeldPlantScreen
        original=HeldPlantScreen(rig,original.shapes,original.blade_path)
    else:
        obstacle=UsdGeom.Cube.Define(rig.stage,path)
        obstacle.CreateSizeAttr(.01);obstacle.AddTranslateOp().Set((.1,0,0))
        UsdPhysics.CollisionAPI.Apply(obstacle.GetPrim())
    original.include_static_scene(rig.stage,'/World/R','/World/P',np.zeros(3))
    b=branch_screen(original,rig);b.snapshot(rig.rest_frames)
    before=rig.stage.GetRootLayer().ExportToString()
    assert not b.check({'1':rig.rest_frames[1]},stroke=True)
    assert b.last_failure['plant_collider']==path
    assert rig.stage.GetRootLayer().ExportToString()==before


def test_positive_blade_feed_pauses_but_overload_backoff_remains():
    from .proportional_backoff_test import controller as through,sample as observe
    for load,expected in ((.25,0.),(.42,-.0005)):
        s=through(False);observe(s,load);start=s.offset
        s.command(step=20,dt=1/480,hold_forward=True)
        assert s.receipt['command_speed_m_s']==pytest.approx(expected)
        assert s.offset==pytest.approx(start+expected/480) and not s.complete


def test_explicit_cli_default_off_and_rejects_incomplete_recipe(tmp_path):
    from .benchmark import parser,main
    assert not parser().parse_args(['--output',str(tmp_path)]).retained_separation_trial
    with pytest.raises(ValueError,match='Retained separation'):
        main(['--output',str(tmp_path),'--retained-separation-trial'])


@pytest.mark.parametrize('mode',['bimanual','right_only'])
def test_launcher_accepts_only_guarded_bimanual_separation(mode,monkeypatch,tmp_path):
    from pathlib import Path
    from .ground_truth_trial import main
    class BeforeWrite(Exception):pass
    def stop(*a,**k):raise BeforeWrite()
    monkeypatch.setattr(Path,'mkdir',stop)
    args=['--output',str(tmp_path/'new'),'--mode',mode,'--milestone','cut_action',
        '--process-zone-trial','--through-stroke-trial','--material-clearance-trial',
        '--postrelease-feed-m-s','.001','--support-aware-feed-trial','--retained-separation-trial']
    if mode=='bimanual':
        with pytest.raises(BeforeWrite):main(args)
    else:
        with pytest.raises(ValueError,match='Retained separation'):main(args)
