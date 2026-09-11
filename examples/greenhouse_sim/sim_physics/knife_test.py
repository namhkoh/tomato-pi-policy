import numpy as np
import pytest

from sim_physics.knife import ShearGate,KnifeGeometry
from sim_physics.plant_test import native


def sample(gate,x,**extra):
    edge=np.eye(4);edge[0,3]=-x
    data=dict(dt=.005,edge=edge,centre=np.zeros(3),axis=np.array([0.,0.,1.]),
        points=[[-x,0,0]],impulses=[[.00125,0,0]],held=True,slip=.001)
    data.update(extra)
    return gate.observe(**data)


def test_force_grasp_transverse_measured_motion_required():
    g=ShearGate('petiole');events=[sample(g,i*.0001) for i in range(12)]
    assert sum(e is not None for e in events)==1
    e=next(e for e in events if e)
    assert e['flat_edge_contact_verified'] and not e['commanded_motion_used_as_evidence']
    assert not e['tissue_fracture_calibrated']


@pytest.mark.parametrize('extra',[
    {'held':False},{'slip':.0031},{'slip':float('nan')},{'impulses':[[.0001,0,0]]},
    {'impulses':[[.01,0,0]]},{'points':[[0,0,.004]]},
    {'axis':np.array([1.,0,0])},{'points':[],'impulses':[]}])
def test_negative_controls_cannot_cut(extra):
    g=ShearGate('petiole')
    assert all(sample(g,i*.0001,**extra) is None for i in range(20))


def test_stationary_loading_reversed_motion_and_separate_taps_do_not_cut():
    for motion in ([0.]*20,[-i*.0001 for i in range(20)]):
        g=ShearGate('petiole')
        assert all(sample(g,x) is None for x in motion)
    g=ShearGate('petiole')
    assert all(sample(g,i*.0001,held=i%3!=0) is None for i in range(20))


def test_free_approach_displacement_is_not_qualified_cutting_travel():
    g=ShearGate('petiole')
    assert sample(g,0.,points=[],impulses=[]) is None
    assert all(sample(g,.001) is None for _ in range(20))
    assert g.dwell>=.025 and g.travel==0.
    # Losing force restarts the loading reference, not merely its counter.
    assert sample(g,.002,impulses=[[0,0,0]]) is None
    assert all(sample(g,.003) is None for _ in range(20))
    assert g.travel==0.


def test_submicrometre_contact_jitter_cannot_accumulate_a_cut():
    g=ShearGate('petiole')
    # Each reverse step is below the numerical tolerance. The old positive
    # increment sum accumulated >0.3 mm despite <1 um of net displacement.
    assert all(sample(g,(i%2)*.0000008) is None for i in range(1000))
    assert g.travel<.000001 and not g.completed


def test_geometry_sized_stroke_covers_shaft_without_fixed_overtravel():
    from sim_physics.knife import transverse_stroke_offsets
    path=transverse_stroke_offsets(.003,.002)
    assert path[0]==-.025 and path[-1]==pytest.approx(.005)
    assert np.diff(path).min()>0 and np.diff(path).max()<=.0005+1e-12
    for radius,width in ((0,.002),(.003,float('nan')),(.02,.002)):
        with pytest.raises(ValueError): transverse_stroke_offsets(radius,width)
    close=transverse_stroke_offsets(.003,.002,.008)
    assert close[0]==-.008 and close[-1]==path[-1]
    assert np.diff(close).max()<=.0005+1e-12
    for standoff in (.0069,.026,float('nan')):
        with pytest.raises(ValueError): transverse_stroke_offsets(.003,.002,standoff)


def test_original_knife_and_blade_release_preserve_source(native):
    from pxr import Gf,UsdGeom,UsdPhysics
    from sim_physics.plant import build
    from sim_physics.bimanual import BimanualRobot
    stage,record=native
    stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant')).AddTranslateOp().Set(Gf.Vec3d(0,0,.9))
    rig=build(stage,record,'SubStem_41')
    original=stage.GetRootLayer().ExportToString()
    robot=BimanualRobot(stage,rig,sparse_contacts=True,approach_tilt=10,torso_degrees=[0]*6,
        ground_height=lambda x,y:.101)
    k=robot.knife
    assert robot.knife_mount['changed']
    assert robot.knife_mount['rotation_wrist_axis']=='Z'
    assert robot.knife_mount['flat_edge_faces']=='wrist_minus_y'
    assert k.local[2,3]<0
    assert (-k.local[:3,0])[1]<-.99
    assert k.local[0,2]>.99
    from sim_physics.knife import mount_forward
    assert not mount_forward(stage,robot.root)['changed']
    assert k.size[0]==pytest.approx(.002) and .045<k.size[1]<.055
    assert k.size[2]==pytest.approx(.006,abs=1e-6)
    assert k.collider.endswith('/BladePlateContact')
    frame=k.frame(np.eye(4))
    assert k.on_edge(frame[:3,3],frame)
    assert not k.on_edge(frame[:3,3]+.01*frame[:3,0],frame)
    desired=k.wrist_for_edge(np.zeros(3),np.array([-1.,0,0]),np.array([0.,0,1.]))
    np.testing.assert_allclose(k.frame(desired),np.eye(4),atol=1e-8)
    with pytest.raises(ValueError): rig.release_from_blade({})
    gate=ShearGate(rig.source_target)
    event=next(e for i in range(12) if (e:=sample(gate,i*.0001)) is not None)
    result=rig.release_from_blade(event)
    assert result['event']=='blade_contact_joint_release'
    assert not result['physical_cut_verified']
    assert not UsdPhysics.Joint.Get(stage,rig.cut_joint_path).GetJointEnabledAttr().Get()
    rig.restore_authored_state()
    assert stage.GetRootLayer().ExportToString()==original

    # Disabling a replacement support piece cannot silently erase protection.
    part=stage.GetPrimAtPath(robot.arc_contacts['collider_paths'][0])
    UsdPhysics.CollisionAPI(part).CreateCollisionEnabledAttr(False)
    with pytest.raises(ValueError,match='arc support'): KnifeGeometry(stage,robot.root)


def test_knife_roll_corrects_old_mount_without_moving_flange_or_camera():
    from pxr import Usd,UsdGeom
    from greenhouse_sim.robot_model import DEFAULT_ASSET,ROBOT_ROOT
    from sim_physics.plant import matrix_attr
    from sim_physics.knife import mount_forward
    stage=Usd.Stage.Open(str(DEFAULT_ASSET));stage.SetEditTarget(stage.GetSessionLayer())
    root=stage.GetPrimAtPath(ROBOT_ROOT+'/ee_right/attachments/DeleafKnife')
    camera=stage.GetPrimAtPath(ROBOT_ROOT+'/ee_right/attachments/RightWristCamera')
    cache=UsdGeom.XformCache();camera_before=np.asarray(cache.GetLocalToWorldTransform(camera)).copy()
    original=stage.GetRootLayer().ExportToString()
    old=np.eye(4);old[:3,:3]=[[0,0,-1],[-1,0,0],[0,1,0]]
    old[:3,3]=[.001,.002,.003]
    matrix_attr(root,old)
    result=mount_forward(stage,ROBOT_ROOT)
    after=np.asarray(UsdGeom.Xformable(root).GetLocalTransformation()).T
    np.testing.assert_allclose(after[:3,3],old[:3,3],atol=1e-10)
    np.testing.assert_allclose(after[:3,:3],np.diag([-1,-1,1])@old[:3,:3],atol=1e-10)
    assert result['changed'] and not mount_forward(stage,ROBOT_ROOT)['changed']
    cache.Clear();np.testing.assert_array_equal(np.asarray(cache.GetLocalToWorldTransform(camera)),camera_before)
    assert stage.GetRootLayer().ExportToString()==original
    matrix_attr(root,np.eye(4))
    with pytest.raises(ValueError,match='Unknown knife mounting'): mount_forward(stage,ROBOT_ROOT)


def test_bimanual_replan_cannot_skip_self_screen(monkeypatch):
    from sim_physics.bimanual import BimanualRobot
    from sim_physics.full_robot import FullRobotGripper
    robot=object.__new__(BimanualRobot)
    robot.self_screen=object();robot.right=np.zeros(7)
    monkeypatch.setattr(FullRobotGripper,'plan_approach',lambda self:setattr(self,'path_q',np.zeros((3,7))))
    monkeypatch.setattr(robot,'check_self',lambda *args:dict(passed=False,minimum_clearance_m=-.01))
    with pytest.raises(RuntimeError,match='path index 0'): robot.plan_approach()


def test_failed_planning_time_is_reported(monkeypatch):
    from sim_physics.bimanual import BimanualRobot
    robot=object.__new__(BimanualRobot)
    def fail(*args): raise RuntimeError('blocked geometry')
    monkeypatch.setattr(robot,'_plan_cut',fail)
    with pytest.raises(RuntimeError,match='blocked geometry'): robot.plan_cut(None,None)
    assert robot.planning_wall_seconds>=0


def test_bimanual_grasp_stops_at_shaft_width_without_relaxing_force_limit(monkeypatch):
    from sim_physics.bimanual import BimanualRobot
    from sim_physics.full_robot import FullRobotGripper
    robot=object.__new__(BimanualRobot);robot.radius=.003
    fractions=[]
    monkeypatch.setattr(FullRobotGripper,'close',lambda self,f:fractions.append(f))
    robot.close(0.);robot.close(.5);robot.close(1.)
    np.testing.assert_allclose(fractions,[0.,.45,.9])
    robot.grasp_compression=.001;robot.close(1.)
    assert fractions[-1]==pytest.approx(.92)
    for f in (float('nan'),-.1,1.1):
        with pytest.raises(ValueError): robot.close(f)


def test_reposition_schedule_preserves_default_and_requires_reobservation_before_motion():
    from sim_physics.bimanual_probe import sequence_times
    original=sequence_times(0)
    assert original==dict(delay=0.,plan=3.5,approach=4.,stroke=8.,end=14.)
    moved=sequence_times(.008)
    assert moved==dict(delay=1.5,plan=5.,approach=5.5,stroke=9.5,end=15.5)
    for value in (-.001,.011,float('nan')):
        with pytest.raises(ValueError): sequence_times(value)


def test_reset_invalidates_scene_snapshot_and_planning_time(monkeypatch):
    from types import SimpleNamespace
    from sim_physics.bimanual import BimanualRobot
    from sim_physics.full_robot import FullRobotGripper
    robot=object.__new__(BimanualRobot)
    robot.rig=SimpleNamespace(source_target='petiole')
    robot.held_plant_screen=SimpleNamespace(workspace=(0,1),static=['old'])
    robot.kin=SimpleNamespace(forward=lambda *args:np.eye(4))
    robot.right=np.zeros(7);robot.base=np.eye(4)
    robot.planning_slides={'old':1};robot.planning_wall_seconds=1.
    monkeypatch.setattr(FullRobotGripper,'restore_authored_state',lambda self:None)
    robot.restore_authored_state()
    assert robot.held_plant_screen.workspace is None and robot.held_plant_screen.static==[]
    assert robot.planning_wall_seconds is None and not hasattr(robot,'planning_slides')
    assert robot.plan is None and robot.cut_event is None and not robot.cut_authorized


def test_known_bad_torso_station_rejected_before_native_physics(native):
    from pxr import Gf,UsdGeom
    from sim_physics.plant import build
    from sim_physics.bimanual import BimanualRobot
    stage,record=native;stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant')).AddTranslateOp().Set(Gf.Vec3d(-.005,0,.9))
    rig=build(stage,record,'SubStem_41')
    with pytest.raises(RuntimeError,match='Pregrasp self-collision screen'):
        BimanualRobot(stage,rig,sparse_contacts=True,approach_tilt=10,grasp_roll=180,
            station_offset=(.16,.22),torso_degrees=[0,0,0,0,0,20],ground_height=lambda x,y:.101)


def test_transit_detour_checks_entire_path_and_never_changes_contact_margin():
    from types import SimpleNamespace
    from sim_physics.bimanual import BimanualRobot
    robot=object.__new__(BimanualRobot);robot.right=np.array([0.,-5,0,-120,0,70,0]);robot.base=np.eye(4)
    # A strip blocks the direct transition in q0; the shoulder must open first.
    def clearance(left,right,base):
        blocked=5<right[0]<15 and right[1]>-20
        return SimpleNamespace(clearance_m=.009 if blocked else .011)
    robot.kin=SimpleNamespace(arm_limits_degrees=lambda side:(np.full(7,-179.),np.full(7,179.)),
        inter_arm_clearance=clearance)
    robot.check_self=lambda *args:dict(passed=True)
    goal=robot.right.copy();goal[0]=20;goal[1]=-25
    result=robot.right_transit(np.zeros(7),goal)
    assert result is not None
    path,minimum,evidence=result
    assert evidence['method']=='outward_shoulder_waypoint' and minimum==pytest.approx(.011)
    np.testing.assert_array_equal(path[0],robot.right);np.testing.assert_array_equal(path[-1],goal)
    assert np.max(np.abs(np.diff(path,axis=0)))<=1+1e-10
    assert all(clearance(None,q,None).clearance_m>=.01 for q in path)
    robot.check_self=lambda *args:dict(passed=False)
    assert robot.right_transit(np.zeros(7),goal) is None
    with pytest.raises(ValueError): robot.right_transit(np.zeros(7),np.full(7,float('nan')))


@pytest.mark.parametrize('tilt',[-10.,0.,10.])
def test_oblique_plane_preserves_anatomical_axis_and_existing_angular_gates(tilt):
    from sim_physics.knife import cut_plane_normal
    d=np.array([-2.,0,0]);axis=np.array([0.,0.,3.])
    before_d=d.copy();before_axis=axis.copy()
    normal=cut_plane_normal(d,axis,tilt)
    knife=object.__new__(KnifeGeometry);knife.local=np.eye(4);knife.size=np.array([.002,.07148,.013])
    frame=knife.wrist_for_edge(np.zeros(3),d,normal)
    np.testing.assert_allclose(frame[:3,:3].T@frame[:3,:3],np.eye(3),atol=1e-10)
    np.testing.assert_array_equal(d,before_d);np.testing.assert_array_equal(axis,before_axis)
    unit_axis=axis/np.linalg.norm(axis)
    assert abs(np.dot(frame[:3,1],unit_axis))<.3
    assert abs(np.dot(-frame[:3,0],unit_axis))<.3
    assert frame[2,2]>0 and np.linalg.det(frame[:3,:3])==pytest.approx(1)
    assert np.dot(frame[:3,2],unit_axis)==pytest.approx(np.cos(np.radians(tilt)))


@pytest.mark.parametrize('direction,axis,tilt',[
    ([0,0,0],[0,0,1],0),([-1,0,0],[0,0,0],0),
    ([-1,0,0],[1,0,0],0),([-1,0,0],[0,0,1],11),
    ([-1,0,0],[0,0,1],float('nan')),([-1,0,float('nan')],[0,0,1],0)])
def test_invalid_oblique_proposals_fail_closed(direction,axis,tilt):
    from sim_physics.knife import cut_plane_normal
    with pytest.raises(ValueError): cut_plane_normal(direction,axis,tilt)


def test_both_transverse_plane_signs_satisfy_same_physical_gate():
    # World-up is not material identity. Neither sign removes contact/travel,
    # grasp, load, transverse direction or target-local axial requirements.
    from sim_physics.knife import cut_plane_normal
    for sign in (1,-1):
        knife=object.__new__(KnifeGeometry);knife.local=np.eye(4);knife.size=np.array([.002,.05,.006])
        gate=ShearGate('petiole');events=[]
        for i in range(12):
            centre=np.array([-i*.0001,0.,0.])
            frame=knife.wrist_for_edge(centre,[-1.,0,0],cut_plane_normal([-1,0,0],[0,0,sign],0))
            events.append(sample(gate,i*.0001,edge=frame))
        assert sum(e is not None for e in events)==1
