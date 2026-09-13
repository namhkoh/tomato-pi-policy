from types import SimpleNamespace as S
from unittest.mock import Mock
import numpy as np
import pytest

from .pregrasp_aperture import validate, closure_samples
from .force_closure import ForceClosure
from .benchmark import main, parser
from .bimanual import BimanualRobot
from .full_robot import FullRobotGripper
from .plant_test import native


@pytest.mark.parametrize('opening', [.025, .008, .005])
def test_open_command_and_feedback_stay_in_selected_interval(opening):
    c=ForceClosure(.003,.0005,pregrasp_half_aperture=opening,symmetric=True,retention_preload=True)
    np.testing.assert_array_equal(c.gaps,[opening,opening])
    for step in range(1200):
        fraction=max(0.,min(1.,(step-120)/240))
        before=c.gaps.copy();gaps=c.command(fraction,step=step,dt=1/240)
        if not fraction:np.testing.assert_array_equal(gaps,[opening,opening])
        if max(before)<=c.slow_gap+1e-12:assert np.max(abs(gaps-before))<=.0005/240+1e-12
        assert np.all(gaps>=c.minimum) and np.all(gaps<=opening)
        c.observe(dict(step_id=step+1,adapter_valid=True,normal_only=True,
            stem_only=True,compressive_support_n=[0.,0.]),[0.,0.],step=step+1,guards_passed=True)
    assert c.drive_limit_n==.30 and c.desired_support_n==.24 and c.backoff_contact_n==.40
    assert c.receipt['native_hard_contact_limit_n']==.5 and not c.receipt['grasp_verified']


@pytest.mark.parametrize('opening',[True,False,np.nan,np.inf,0.,-.001,.004999,.025001,'0.008',None,complex(.008,0)])
def test_bad_or_contacting_initial_opening_rejected(opening):
    with pytest.raises(ValueError):validate(.003,opening)
    with pytest.raises(ValueError):ForceClosure(.003,.0005,pregrasp_half_aperture=opening)


@pytest.mark.parametrize('value',['.008','0','nan','inf','.026'])
def test_cli_cannot_change_opening_outside_checked_trial(tmp_path,value):
    out=tmp_path/'not_created'
    with pytest.raises(ValueError,match='Pre-grasp opening'):
        main(['--output',str(out),'--pregrasp-half-aperture-m',value])
    assert not out.exists()


def test_default_and_geometric_closure_use_the_same_opening():
    assert parser().parse_args(['--output','unused']).pregrasp_half_aperture_m==.025
    robot=object.__new__(BimanualRobot)
    robot.pregrasp_half_aperture=.008;robot.radius=.003;robot.grasp_compression=.0005
    robot.targets=np.zeros((1,2));robot.finger_indices=[0,1];robot.index=np.array([0]);robot.robot=Mock()
    for fraction,expected in [(0.,.008),(.5,.00525),(1.,.0025)]:
        robot.close(fraction)
        np.testing.assert_allclose(robot.targets,[[-expected,expected]])


def test_antiwindup_preserves_opening_and_refuses_infeasible_symmetric_effort():
    from .finger_target_antiwindup import project
    kw=dict(minimum=.0025,maximum=.008,retention_preload=True,symmetric=True)
    gaps,receipt=project([.008,.008],[-.008,.008],[0,0],[.3,.3],**kw)
    np.testing.assert_array_equal(gaps,[.008,.008])
    assert receipt['commanded_maximum_half_gap_m']==.008
    assert not receipt['changes_physical_joint_limits']
    with pytest.raises(RuntimeError,match='No symmetric aperture'):
        project([.008,.008],[-.02,.02],[0,0],[.3,.3],**kw)


@pytest.mark.parametrize('effort_reference',[False,True])
def test_collision_screen_covers_approach_and_entire_actual_closure(monkeypatch,effort_reference):
    from . import held_plant_screen
    seen=[]
    class Screen:
        last_failure=None
        def __init__(self,*a,**k):pass
        def snapshot(self,frames):pass
        def check(self,bodies,*,grasp):
            assert grasp;seen.append(bodies);return True
    monkeypatch.setattr(held_plant_screen,'HeldPlantScreen',Screen)
    r=object.__new__(BimanualRobot);r.pregrasp_half_aperture=.008
    r.effort_bounded_grasp_target=effort_reference;r.force_closer=S(minimum=.0015)
    r.slides={'gripper_finger_l1':-.008,'gripper_finger_l2':.008}
    r.held_plant_screen=S(workspace={},static=[],static_indices=[])
    r.rig=S(stem_contact_model='flush_capsules_v1');r.self_screen=S(shapes=[])
    r.knife=S(collider='/Knife');r.grasp_path='/Shaft';r.right=np.zeros(7)
    r.fractions=np.array([0,.5,1.]);r.path_q=np.zeros((3,7));r.radius=.003;r.grasp_compression=.0005
    r.body_world=lambda *args:dict(r.planning_slides)
    result=r.screen_grasp_scene(np.eye(4)[None])
    assert result['passed'] and not result['native_grasp_verified']
    assert seen[:3]==[r.slides]*3
    np.testing.assert_array_equal([x['gripper_finger_l2'] for x in seen[3:]],closure_samples(.008,.0015 if effort_reference else .0025))
    assert max(abs(np.diff([x['gripper_finger_l2'] for x in seen[3:]])))<=.001
    assert not hasattr(r,'planning_slides')


def test_authored_initial_native_joints_use_opening_without_source_or_limit_changes(native):
    from pxr import Gf,Usd,UsdGeom,UsdPhysics
    from .plant import build
    stage,record=native
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant')).AddTranslateOp().Set(Gf.Vec3d(0,0,.35))
    rig=build(stage,record,'SubStem_41');source=stage.GetRootLayer().ExportToString()
    first=FullRobotGripper(stage,rig,compliant_fingers=True,ground_height=lambda x,y:0.)
    def limits(robot):
        return [[stage.GetPrimAtPath(robot.root+'/joints/'+name).GetAttribute(a).Get()
            for a in ('physics:lowerLimit','physics:upperLimit','drive:linear:physics:stiffness',
                'drive:linear:physics:damping','drive:linear:physics:maxForce')] for name in robot.slides]
    original_limits=limits(first)
    assert first.slides=={'gripper_finger_l1':-.025,'gripper_finger_l2':.025}
    stage.RemovePrim(first.root)
    robot=FullRobotGripper(stage,rig,compliant_fingers=True,pregrasp_half_aperture=.008,ground_height=lambda x,y:0.)
    assert limits(robot)==original_limits and stage.GetRootLayer().ExportToString()==source
    assert not robot.report()['grasp_weld']
    for name,value in robot.slides.items():
        joint=stage.GetPrimAtPath(robot.root+'/joints/'+name)
        assert joint.GetAttribute('state:linear:physics:position').Get()==pytest.approx(value)
        assert UsdPhysics.DriveAPI(joint,'linear').GetTargetPositionAttr().Get()==pytest.approx(value)
    robot.restore_authored_state()
    assert robot.slides=={'gripper_finger_l1':-.008,'gripper_finger_l2':.008}
