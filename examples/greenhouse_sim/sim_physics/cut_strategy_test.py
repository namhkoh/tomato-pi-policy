from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace as S
import numpy as np
import pytest

from .cut_strategy import choose, support_ready
from .knife import ShearGate, ShearParameters, BRITTLE_CUT_MODEL
from .knife_test import sample
from .signed_blade_test import seam_fixture


def unheld_evidence(**change):
    g=ShearGate('petiole',ShearParameters(model=BRITTLE_CUT_MODEL),strategy='right_only')
    args=dict(held=False,slip=None,cut_only_ready=True);args.update(change)
    return next((e for _ in range(8) if (e:=sample(g,0.,**args)) is not None),None)


def test_cut_only_is_explicit_and_never_fabricates_a_grasp():
    e=unheld_evidence()
    assert e['cut_strategy']=='right_only' and not e['stable_left_grasp']
    assert e['grasp_slip_m'] is None and e['cut_only_ready']
    rig,joint=seam_fixture();before=rig.stage.GetRootLayer().ExportToString()
    with pytest.raises(ValueError,match='cross execution'):rig.release_from_blade(e)
    assert joint.GetJointEnabledAttr().Get() and not rig.cut
    result=rig.release_from_blade(e,strategy='right_only')
    assert not joint.GetJointEnabledAttr().Get() and rig.cut
    assert before==rig.stage.GetRootLayer().ExportToString()
    assert not result['physical_cut_verified'] and not result['training_eligible']


@pytest.mark.parametrize('change',[
    dict(cut_only_ready=False),dict(held=True),dict(slip=0.),
    dict(impulses=[[.0009,0,0]]),dict(impulses=[[-.00125,0,0]],normals=[[-1.,0,0]]),
    dict(tool_contact_upper_bound_n=.501),dict(points=[[0.,0.,.004]]),
    dict(edge_contact_verified=False),dict(normals=[[0.,1.,0.]],impulses=[[0.,.00125,0.]])])
def test_cut_only_keeps_contact_direction_force_identity_and_mode_gates(change):
    assert unheld_evidence(**change) is None


@pytest.mark.parametrize('change',[
    dict(cut_strategy='bimanual'),dict(stable_left_grasp=True),dict(grasp_slip_m=0.),
    dict(cut_only_ready=False),dict(contact_dwell_s=.024),dict(peak_tool_contact_upper_bound_n=.501)])
def test_release_independently_rejects_bad_cut_only_evidence(change):
    rig,joint=seam_fixture();e=unheld_evidence();e.update(change)
    before=rig.stage.GetSessionLayer().ExportToString()
    with pytest.raises(ValueError):rig.release_from_blade(e,strategy='right_only')
    assert not rig.cut and joint.GetJointEnabledAttr().Get()
    assert before==rig.stage.GetSessionLayer().ExportToString()


def test_bimanual_cannot_use_cut_only_readiness_to_skip_grasp():
    g=ShearGate('petiole',ShearParameters(model=BRITTLE_CUT_MODEL))
    assert all(sample(g,0.,held=False,slip=None,cut_only_ready=True) is None for _ in range(10))
    assert not support_ready('bimanual',True,.003)


@pytest.mark.parametrize('b,r,drop,expected',[
    ('clear','clear',True,'bimanual'),('clear','blocked',False,'bimanual'),
    ('blocked','clear',True,'right_only_from_checked_park'),
    ('blocked','clear',False,'skip'),('blocked','blocked',True,'skip'),
    ('unknown','clear',True,'inspect_or_retry'),('blocked','unknown',True,'inspect_or_retry')])
def test_strategy_selection_does_not_confuse_unknown_with_collision(b,r,drop,expected):
    assert choose(bimanual=b,right_only=r,drop_allowed=drop)==expected


def only_cli(out):
    from .diagnostic_rate_test import cli
    remove={'--cut-convergence-trial','--require-retention-screen','--physical-grasp-span',
            '--settle-retention-preload','--effort-bounded-grasp-target','--preload-force-servo'}
    return [a for a in cli(out) if a not in remove]+['--right-only-cut-trial','--branch-contact-fixture']


def test_complete_cut_only_profile_validates_before_native_launch(tmp_path,monkeypatch):
    from .benchmark import main,parser
    assert not parser().parse_args(['--output','unused']).right_only_cut_trial
    out=tmp_path/'unused';args=only_cli(out)
    class Validated(Exception):pass
    def stop(self,*a,**kw):
        assert self==out
        raise Validated()
    monkeypatch.setattr(Path,'mkdir',stop)
    with pytest.raises(Validated):main(args)
    for flag in ('--right-only-cut-trial','--branch-contact-fixture','--native-startup-clearance'):
        with pytest.raises(ValueError):main([a for a in args if a!=flag])
    for flag in ('--gui','--require-retention-screen','--measured-withdrawal','--physical-grasp-span'):
        with pytest.raises(ValueError):main(args+[flag])


def park_fixture():
    q=np.array([[.1,.2,-.008,.008]],dtype=np.float32)
    f=S(names=['a','b','f1','f2'],left_indices=[0,1],finger_indices=[2,3],
        initial_q=np.degrees(q[0,:2].astype(float)),paths=['palm','f1','f2'],
        pregrasp_half_aperture=.008,targets=q.copy())
    f.robot=S(get_dof_positions=lambda:q,get_dof_position_targets=lambda:f.targets)
    r=dict(t=1/480,contact=dict(step_id=1,adapter_valid=True,bilateral=False),slip_m=None,
           robot=dict(per_finger_contact_upper_bound_n={'f1':0.,'f2':0.}))
    return f,q,r


def test_native_park_readback_has_no_commands_or_retention_credit():
    from .cut_only import parked_left
    f,q,r=park_fixture();before=q.copy();targets=f.targets.copy()
    e=parked_left(f,r,step=1,physics_hz=480)
    assert not e['left_grasp_verified'] and not e['retention_expected'] and not e['deposit_expected']
    np.testing.assert_array_equal(q,before);np.testing.assert_array_equal(f.targets,targets)


@pytest.mark.parametrize('fault',['stale','bilateral','slip','command','joint','closed','contact','nan','sensor'])
def test_park_refuses_contact_tracking_or_stale_evidence(fault):
    from .cut_only import parked_left
    f,q,r=park_fixture()
    if fault=='stale':r['t']=0
    elif fault=='bilateral':r['contact']['bilateral']=True
    elif fault=='slip':r['slip_m']=0.
    elif fault=='command':f.targets[0,0]+=.01
    elif fault=='joint':q[0,0]+=.006
    elif fault=='closed':q[0,2]=-.006
    elif fault=='contact':r['robot']['per_finger_contact_upper_bound_n']['f1']=.006
    elif fault=='nan':q[0,0]=np.nan
    else:r['contact']['adapter_valid']=False
    with pytest.raises(RuntimeError):parked_left(f,r,step=1,physics_hz=480)


def test_final_cut_only_left_screen_has_no_grasp_collision_allowance(monkeypatch):
    from .withdrawal_native_check_test import setup
    from .withdrawal_native_check import check
    f,frames,s,_,_=setup(monkeypatch);f.cut_strategy='right_only'
    e=check(f,frames,4800)
    assert e['left_scene']['expected_grasp_colliders']==[]
    assert s.checks[-1][2]['grasp'] is False


def test_left_park_sends_original_joint_reference_not_resolved_grasp_path():
    from .full_robot import FullRobotGripper
    f=object.__new__(FullRobotGripper);f.initial_q=np.arange(7,dtype=float)
    f.base=np.eye(4);f.path_q=np.full((2,7),123.)
    resets=[];commands=[]
    f.event_monitor=S(begin_step=lambda:resets.append(True))
    f.kin=S(arm_limits_degrees=lambda side:(np.full(7,-180.),np.full(7,180.)),
            forward=lambda side,q,base:np.eye(4))
    f._command_left_drives=lambda q,pose:commands.append((q.copy(),pose.copy()))
    f.hold_left_park()
    np.testing.assert_array_equal(commands[0][0],f.initial_q)
    assert resets==[True]
    f.initial_q[0]=180.
    with pytest.raises(RuntimeError):f.hold_left_park()
    assert len(commands)==1
