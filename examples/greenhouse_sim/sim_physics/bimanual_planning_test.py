"""Planner must reject a full tool-corridor conflict before running arm IK."""
from types import SimpleNamespace as S
import json
import numpy as np
import pytest
from sim_physics.bimanual import BimanualRobot


def test_complete_rigid_corridor_precedes_endpoint_ik(monkeypatch):
    import sim_physics.rigid_tool_screen as module
    robot=BimanualRobot.__new__(BimanualRobot)
    robot.stage=None;robot.root='/World/R';robot.rig=S(root='/World/P')
    robot.held_plant_screen=S(workspace=[np.zeros(3),np.ones(3)],static=[],snapshot=lambda frames:None)
    robot.goal=np.eye(4);robot.goal[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
    robot.radius=.003;robot.right=np.zeros(7);robot.base=np.eye(4)
    robot.stroke_offsets=np.linspace(-.025,.005,61)
    robot.seam=lambda frames:(np.zeros(3),np.array([0.,0.,1.]))
    def wrist(point,*unused):
        result=np.eye(4);result[:3,3]=point;return result
    robot.knife=S(size=np.array([.002,.05,.006]),wrist_for_edge=wrist)
    def forbidden_ik(*args,**kwargs):raise AssertionError('Blocked tool corridor reached arm IK')
    robot.kin=S(solve_pose=forbidden_ik);checked=[]
    class Screen:
        def __init__(self,*args):pass
        def check(self,frames):
            assert frames.shape==(61,4,4)
            assert np.linalg.norm(frames[-1,:3,3]-frames[0,:3,3])==pytest.approx(.03)
            checked.append(frames)
            return dict(passed=False,reason='end_of_stroke_tool_collision',sample=60,native_validated=False)
    monkeypatch.setattr(module,'RigidToolScreen',Screen)
    with pytest.raises(RuntimeError,match='IK_attempted=0'):
        robot._plan_cut([],np.zeros(7))
    assert len(checked)==756
    attempts=robot.plan_diagnostics['endpoint_attempts']
    assert all(not a['ik_attempted'] and a['rejection']=='rigid_tool_corridor' for a in attempts)
    assert robot.plan is None and not robot.plan_diagnostics['whole_scene_path_certified']


def test_final_validation_precedes_plan_acceptance_and_always_closes():
    robot=object.__new__(BimanualRobot);events=[]
    candidate={'candidate':'provisional'}
    def plan(*args):
        robot.plan=candidate
        return 'legacy_result'
    def validate():
        assert robot.plan is None
        events.append('validate')
    robot._plan_cut=plan
    robot.plan_diagnostics={}
    native=S(validate=validate,close=lambda:events.append('close'),
             report=lambda:{'final_validation_passed':True})
    robot.held_plant_screen=S(native_static_query=native)
    assert robot.plan_cut(None,None)=='legacy_result'
    assert robot.plan is candidate and events==['validate','close']
    assert robot.held_plant_screen.native_static_query is None
    assert robot.plan_diagnostics['native_static_clearance']['final_validation_passed']


@pytest.mark.parametrize('failure',['step','missing_actor','budget'])
def test_finalization_invalidates_earlier_clearances_and_discards_plan(failure,monkeypatch):
    from .native_static_clearance import NativeStaticClearance
    import sim_physics.native_static_clearance as module
    state={'epoch':0,'present':True,'time':0.}
    monkeypatch.setattr(module.time,'perf_counter',lambda:state['time'])
    def guard():
        if state['epoch']:raise RuntimeError('physics pre-step epoch changed')
    def query(path,centre,axes,half):
        return bool(state['present'] and half[0]>.01)
    records=[('/World/Stem','box',None,np.full(3,-.01),np.full(3,.01))]
    native=NativeStaticClearance(query,records,guard=guard)
    robot=object.__new__(BimanualRobot)
    robot.held_plant_screen=S(native_static_query=native);robot.plan_diagnostics={}
    def plan(*args):
        assert native.clear_box('/World/Stem',np.zeros(3),np.eye(3),np.full(3,.002),.001)
        robot.plan={'uses_prior_clearance':True}
        if failure=='step':state['epoch']+=1
        elif failure=='missing_actor':state['present']=False
        else:state['time']=9.
    robot._plan_cut=plan
    with pytest.raises(RuntimeError,match='final validation'):robot.plan_cut(None,None)
    assert robot.plan is None and native.closed
    assert robot.held_plant_screen.native_static_query is None
    assert not robot.plan_diagnostics['native_static_clearance']['final_validation_passed']


@pytest.mark.parametrize('cleanup_failure',['close','report'])
def test_cleanup_never_masks_original_planning_error(cleanup_failure):
    robot=object.__new__(BimanualRobot);robot.plan_diagnostics={};events=[]
    original=RuntimeError('original blocked geometry')
    def plan(*args):
        robot.plan={'unusable':True}
        raise original
    def validate():raise AssertionError('Failed planner must not validate')
    def close():
        events.append('close')
        if cleanup_failure=='close':raise RuntimeError('close fault')
    def report():
        events.append('report')
        if cleanup_failure=='report':raise RuntimeError('report fault')
        return {}
    robot._plan_cut=plan
    robot.held_plant_screen=S(native_static_query=S(validate=validate,close=close,report=report))
    with pytest.raises(RuntimeError,match='original blocked geometry') as caught:robot.plan_cut(None,None)
    assert caught.value is original and original.__notes__
    assert events==['close','report']
    assert robot.plan is None and robot.held_plant_screen.native_static_query is None


def test_cleanup_failure_prevents_otherwise_valid_plan_acceptance():
    robot=object.__new__(BimanualRobot);robot.plan_diagnostics={}
    def plan(*args):robot.plan={'candidate':True}
    def close():raise RuntimeError('unsubscribe fault')
    robot._plan_cut=plan
    robot.held_plant_screen=S(native_static_query=S(
        validate=lambda:None,close=close,report=lambda:{}))
    with pytest.raises(RuntimeError,match='cleanup failed'):robot.plan_cut(None,None)
    assert robot.plan is None and robot.held_plant_screen.native_static_query is None


def test_uninitialized_failure_preserves_original_exception_and_timing():
    robot=object.__new__(BimanualRobot)
    def plan(*args):raise RuntimeError('blocked geometry')
    robot._plan_cut=plan
    with pytest.raises(RuntimeError,match='blocked geometry'):robot.plan_cut(None,None)
    assert robot.planning_wall_seconds>=0 and robot.plan is None


@pytest.mark.parametrize('prior',[None,{'stale_prior_plan':True}])
@pytest.mark.parametrize('failure',['missing_robot','factory'])
def test_query_initialization_failure_has_fresh_diagnostics(monkeypatch,prior,failure):
    import sim_physics.native_static_clearance as module
    robot=object.__new__(BimanualRobot)
    robot.plan={'stale_candidate':True};robot.plan_diagnostics=prior
    robot.stage=object();robot.native_static_clearance=True
    robot.seam=lambda frames:(np.zeros(3),np.array([0.,0.,1.]))
    robot.held_plant_screen=S(workspace=[np.zeros(3),np.ones(3)],static=[],
        snapshot=lambda frames:None,native_static_query=None)
    original=RuntimeError('mock subscription initialization failure');calls=[]
    original.add_note('original cleanup evidence')
    def factory(*args):
        calls.append('factory')
        assert robot.plan is None
        assert robot.plan_diagnostics is not prior
        assert robot.plan_diagnostics['endpoint_attempts']==[]
        assert not robot.plan_diagnostics['native_static_clearance']['final_validation_passed']
        raise original
    monkeypatch.setattr(module,'current_scene_query',factory)
    if failure=='factory':robot.robot=object()
    message='mock subscription' if failure=='factory' else 'initialized robot scene'
    with pytest.raises(RuntimeError,match=message) as caught:robot.plan_cut([],np.zeros(7))
    if failure=='factory':
        assert caught.value is original and caught.value.__notes__==['original cleanup evidence']
    assert calls==(['factory'] if failure=='factory' else [])
    assert robot.plan is None and robot.planning_wall_seconds>=0
    assert robot.held_plant_screen.native_static_query is None
    diagnostics=robot.plan_diagnostics
    assert diagnostics is not prior and 'stale_prior_plan' not in diagnostics
    assert diagnostics['endpoint_attempts']==diagnostics['path_failures']==[]
    assert not diagnostics['complete_tool_stroke_screened_before_IK']
    assert not diagnostics['whole_scene_path_certified']
    evidence=diagnostics['native_static_clearance']
    assert evidence['initialization_status']=='failed' and evidence['errors']
    assert not evidence['final_validation_passed']
    # A factory may query internally before it raises; never invent a zero.
    assert evidence['query_count']==(None if failure=='factory' else 0)
    assert evidence['query_count_known'] is (failure=='missing_robot')
    json.dumps(diagnostics,allow_nan=False)


def test_snapshot_failure_records_native_not_started_zero_queries(monkeypatch):
    import sim_physics.native_static_clearance as module
    robot=object.__new__(BimanualRobot);robot.stage=object();robot.native_static_clearance=True
    robot.plan_diagnostics={'stale_prior_plan':True}
    robot.seam=lambda frames:(np.zeros(3),np.array([0.,0.,1.]))
    original=RuntimeError('snapshot failed before native setup')
    def snapshot(frames):raise original
    def forbidden(*args):raise AssertionError('Failed snapshot must not start native queries')
    robot.held_plant_screen=S(workspace=[np.zeros(3),np.ones(3)],static=[],
        snapshot=snapshot,native_static_query=None)
    monkeypatch.setattr(module,'current_scene_query',forbidden)
    with pytest.raises(RuntimeError,match='snapshot failed') as caught:robot.plan_cut([],np.zeros(7))
    assert caught.value is original and robot.plan is None and robot.planning_wall_seconds>=0
    diagnostics=robot.plan_diagnostics
    assert diagnostics['endpoint_attempts']==[] and 'stale_prior_plan' not in diagnostics
    assert not diagnostics['held_plant_native_snapshot_screened']
    evidence=diagnostics['native_static_clearance']
    assert evidence['initialization_status']=='not_started'
    assert evidence['query_count']==0 and evidence['query_count_known']
    assert not evidence['final_validation_passed']


def test_zero_query_helper_report_survives_later_setup_failure(monkeypatch):
    import sim_physics.native_static_clearance as native_module
    import sim_physics.rigid_tool_screen as rigid_module
    def forbidden(*args):raise AssertionError('No records should produce no queries')
    native=native_module.NativeStaticClearance(forbidden,[])
    robot=object.__new__(BimanualRobot);robot.stage=object();robot.native_static_clearance=True
    robot.plan_diagnostics=None
    robot.seam=lambda frames:(np.zeros(3),np.array([0.,0.,1.]))
    robot.held_plant_screen=S(workspace=[np.zeros(3),np.ones(3)],static=[],
        snapshot=lambda frames:None,native_static_query=None)
    robot.robot=S(get_dof_positions=lambda:np.zeros((1,0)));robot.names=[];robot.slides=[]
    monkeypatch.setattr(native_module,'current_scene_query',lambda *args:native)
    original=RuntimeError('rigid screen initialization failed')
    def fail(*args):raise original
    monkeypatch.setattr(rigid_module,'RigidToolScreen',fail)
    with pytest.raises(RuntimeError,match='rigid screen initialization') as caught:robot.plan_cut([],np.zeros(7))
    assert caught.value is original and robot.plan is None and native.closed
    assert robot.held_plant_screen.native_static_query is None
    evidence=robot.plan_diagnostics['native_static_clearance']
    assert evidence['initialization_status']=='ready'
    assert evidence['query_count']==0 and evidence['query_count_known']
    assert evidence['closed'] and not evidence['final_validation_passed']
    assert robot.plan_diagnostics['endpoint_attempts']==[]
    json.dumps(robot.plan_diagnostics,allow_nan=False)


def single_proposal_robot(monkeypatch, failure=None):
    """Exercise the real planner/transit methods, with no USD or native runtime.

    Only geometry/IK responses and the bounded-search backend are fakes. The
    single proposal, its fresh-axis resolution, candidate loops and every
    existing endpoint/stroke/transit acceptance branch are the real code.
    """
    import sim_physics.rigid_tool_screen as rigid_module
    import sim_physics.joint_path as path_module
    from sim_physics.cut_proposal import SCHEMA, parse_cut_proposal

    robot = BimanualRobot.__new__(BimanualRobot)
    robot.stage = None
    robot.root = '/World/R'
    robot.rig = S(root='/World/P', source_target='seed101_full/SubStem_41')
    robot.held_plant_screen = S(workspace=[np.full(3, -10.), np.full(3, 10.)],
        static=[], snapshot=lambda frames: None, last_failure={'synthetic_obstruction': True})
    robot.radius = .003
    robot.right = np.zeros(7)
    robot.base = np.eye(4)
    robot.stroke_offsets = np.linspace(-.025, .005, 61)
    centre = np.array([.12, -.23, 1.4])  # Nonzero CURRENT centre; not in proposal.
    axis = np.array([0., 0., 1.])
    robot.seam = lambda frames: (centre.copy(), axis.copy())
    # The old relative150 world direction must remain fixed after a 20-degree
    # left-basis change: the planner should report relative130, not reuse150.
    direction = np.array([np.cos(np.radians(150.)), np.sin(np.radians(150.)), 0.])
    basis = np.array([np.cos(np.radians(20.)), np.sin(np.radians(20.)), 0.])
    robot.goal = np.eye(4)
    robot.goal[:3, :3] = np.column_stack([np.cross(axis, -basis), axis, -basis])
    robot.cut_proposal = parse_cut_proposal(dict(schema=SCHEMA,
        source_target=robot.rig.source_target, world_direction=direction.tolist(),
        normal_sign=-1, wing_m=-.01778771311987213, plane_tilt_degrees=0.,
        maximum_projection_degrees=1., provenance=dict(
            sources={name: {'path': name+'.json', 'sha256': 'a'*64}
                     for name in ('report', 'snapshot')},
            derivation={'method': 'synthetic integration fixture, not native evidence'})),
        source_target=robot.rig.source_target)
    calls = {key: [] for key in ('rigid', 'ik', 'arm', 'self', 'scene', 'transit', 'search', 'order')}
    phase = ['endpoint']

    def blocked(check):
        # Stroke failure is interior, not the already-checked endpoint.
        return (failure == phase[0]+'_'+check
                and (phase[0] != 'stroke' or len(calls['ik']) >= 6))

    def wrist(point, d, normal, wing):
        np.testing.assert_allclose(d, direction, atol=1e-14)
        np.testing.assert_allclose(normal, -axis, atol=1e-14)
        assert wing == robot.cut_proposal.wing_m
        result = np.eye(4)
        result[:3, :3] = np.column_stack([-d, np.cross(normal, -d), normal])
        result[:3, 3] = point
        return result

    robot.knife = S(size=np.array([.002, .04952825137749362, .006]), wrist_for_edge=wrist)

    class Screen:
        def __init__(self, *args): pass

        def check(self, frames):
            calls['order'].append('rigid')
            calls['rigid'].append(frames.copy())
            assert frames.shape == (61, 4, 4)
            np.testing.assert_allclose(frames[:, :3, 3],
                centre + robot.stroke_offsets[:, None]*direction, atol=1e-14)
            np.testing.assert_allclose(frames[:, :3, :3],
                np.repeat(frames[:1, :3, :3], len(frames), axis=0), atol=1e-14)
            return dict(passed=failure != 'rigid', checked_frames=61,
                        reason='synthetic_complete_stroke_check', sample=60,
                        native_validated=False)

    monkeypatch.setattr(rigid_module, 'RigidToolScreen', Screen)

    def solve(side, desired, seed, base, maximum_evaluations):
        assert side == 'right' and maximum_evaluations == 250
        phase[0] = 'endpoint' if not calls['ik'] else 'stroke'
        calls['order'].append(phase[0]+'_ik')
        calls['ik'].append((phase[0], desired.copy()))
        return S(succeeded=not blocked('ik'), position_error_m=0., orientation_error_rad=0.,
                 evaluations=1, joint_degrees=np.array([2., 0., 0., 0., 0., 0., 0.]))

    def arm(left, right, base):
        calls['arm'].append((phase[0], np.array(right).copy()))
        return S(clearance_m=.009 if blocked('arm') else .02)

    def self_check(left, right):
        calls['self'].append((phase[0], np.array(right).copy()))
        return {'passed': not blocked('self')}

    def scene(left, right, *, stroke=False):
        calls['scene'].append((phase[0], np.array(right).copy(), stroke))
        assert stroke is (phase[0] == 'stroke')
        return not blocked('scene')

    robot.kin = S(solve_pose=solve, inter_arm_clearance=arm,
        arm_limits_degrees=lambda side: (np.full(7, -180.), np.full(7, 180.)))
    robot.check_self = self_check
    robot.check_held_plant = scene
    real_transit = robot.right_transit

    def transit(left, goal):
        phase[0] = 'transit'
        calls['order'].append('transit')
        calls['transit'].append(goal.copy())
        return real_transit(left, goal)

    robot.right_transit = transit

    def bounded_search(start, goal, lower, upper, valid):
        calls['search'].append('fallback')
        # No expensive search. Verify the REAL transit validator also rejects
        # the injected obstruction; it cannot bypass a failed direct/waypoint.
        assert failure in ('transit_arm', 'transit_self', 'transit_scene')
        assert not valid((start+goal)/2)
        return None

    monkeypatch.setattr(path_module, 'connect_path', bounded_search)
    return robot, calls, centre, direction


def test_single_world_proposal_screens_one_entire_stroke_before_zero_ik(monkeypatch):
    robot, calls, _, _ = single_proposal_robot(monkeypatch, 'rigid')
    with pytest.raises(RuntimeError, match='endpoints=1, IK_attempted=0'):
        robot.plan_cut([], np.zeros(7))
    assert len(calls['rigid']) == 1
    assert not calls['ik'] and not calls['transit'] and not calls['search']
    attempts = robot.plan_diagnostics['endpoint_attempts']
    assert len(attempts) == 1 and attempts[0]['angle'] == pytest.approx(130.)
    assert attempts[0]['rigid_tool_corridor']['sample'] == 60
    assert attempts[0]['rejection'] == 'rigid_tool_corridor'
    assert robot.plan is None  # No fallback grid or partial corridor acceptance.


@pytest.mark.parametrize('failure,expected', [
    ('endpoint_ik', 'endpoint_IK'), ('endpoint_arm', 'endpoint_arm_clearance'),
    ('endpoint_self', 'endpoint_self_collision'), ('endpoint_scene', 'endpoint_held_plant'),
    ('stroke_ik', 'stroke_IK'), ('stroke_arm', 'stroke_arm_clearance'),
    ('stroke_self', 'stroke_self_collision'), ('stroke_scene', 'stroke_held_plant'),
    ('transit_arm', 'bounded_transit_arm_self_or_plant_clearance'),
    ('transit_self', 'bounded_transit_arm_self_or_plant_clearance'),
    ('transit_scene', 'bounded_transit_arm_self_or_plant_clearance'),
])
def test_single_proposal_never_bypasses_endpoint_stroke_or_transit_guards(monkeypatch, failure, expected):
    robot, calls, _, _ = single_proposal_robot(monkeypatch, failure)
    with pytest.raises(RuntimeError, match='endpoints=1'):
        robot.plan_cut([], np.zeros(7))
    assert len(calls['rigid']) == 1 and calls['order'][0] == 'rigid'
    assert len(robot.plan_diagnostics['endpoint_attempts']) == 1
    assert robot.plan is None
    if failure.startswith('endpoint'):
        assert robot.plan_diagnostics['endpoint_attempts'][0]['rejection'] == expected
        assert len(calls['ik']) == 1 and not calls['transit']
    elif failure.startswith('stroke'):
        assert robot.plan_diagnostics['path_failures'][0]['rejection'] == expected
        assert len(calls['ik']) == 6 and not calls['transit']
    else:
        assert robot.plan_diagnostics['path_failures'][0]['rejection'] == expected
        assert len(calls['ik']) == 62 and len(calls['transit']) == 1
        assert calls['search'] == ['fallback']
        # Four direct/shoulder alternatives and bounded-search validation all
        # retain the injected arm/self/scene obstruction.
        check = failure.removeprefix('transit_')
        assert len([row for row in calls[check] if row[0] == 'transit']) >= 5


def test_single_proposal_success_requires_full_stroke_and_real_transit_checks(monkeypatch):
    robot, calls, centre, direction = single_proposal_robot(monkeypatch)
    robot.plan_cut([], np.zeros(7))
    assert len(calls['rigid']) == 1 and len(calls['ik']) == 62
    assert calls['order'] == ['rigid', 'endpoint_ik'] + ['stroke_ik']*61 + ['transit']
    assert len(calls['transit']) == 1 and not calls['search']
    for check in ('arm', 'self', 'scene'):
        assert sum(row[0] == 'endpoint' for row in calls[check]) == 1
        assert sum(row[0] == 'stroke' for row in calls[check]) == 61
        transit = [row[1] for row in calls[check] if row[0] == 'transit']
        # The real direct-transit method checks start, interior and goal at the
        # unchanged <=1-degree sampling; not merely the endpoint.
        np.testing.assert_allclose(np.array(transit)[:, 0], [0., 1., 2.])
    assert robot.plan['stroke'].shape == (61, 7)
    assert robot.plan['approach'].shape == (3, 7)
    assert robot.plan['angle'] == pytest.approx(130.)
    np.testing.assert_allclose(robot.plan['direction'], direction, atol=1e-14)
    np.testing.assert_allclose(robot.plan['centre'], centre, atol=1e-14)
    assert robot.plan['minimum_interarm_m'] == .02
    assert not robot.plan_diagnostics['whole_scene_path_certified']
    assert not robot.plan_diagnostics['single_world_proposal']['planning_permission']


@pytest.mark.parametrize('failure',[None,'rigid'])
@pytest.mark.parametrize('basis_angle',[0.,np.pi,1e-14])
def test_single_proposal_parallel_legacy_basis_keeps_world_direction_without_nan(monkeypatch,failure,basis_angle):
    robot,calls,_,direction=single_proposal_robot(monkeypatch,failure)
    c,s=np.cos(basis_angle),np.sin(basis_angle)
    robot.goal=np.eye(4);robot.goal[:3,:3]=[[c,0.,s],[0.,1.,0.],[-s,0.,c]]
    with np.errstate(divide='raise',invalid='raise'):
        if failure is None:robot.plan_cut([],np.zeros(7))
        else:
            with pytest.raises(RuntimeError,match='endpoints=1, IK_attempted=0'):
                robot.plan_cut([],np.zeros(7))
    diagnostics=robot.plan_diagnostics
    single=diagnostics['single_world_proposal']
    assert not single['relative_angle_defined']
    assert single['relative_angle_unavailable_reason']=='legacy_approach_parallel_to_stem_axis'
    np.testing.assert_allclose(single['direction_world'],direction,atol=1e-14)
    assert len(calls['rigid'])==1 and diagnostics['endpoint_attempts'][0]['angle'] is None
    json.dumps(diagnostics,allow_nan=False)
    if failure is None:
        assert robot.plan['angle'] is None and len(calls['ik'])==62 and len(calls['transit'])==1
        np.testing.assert_allclose(robot.plan['direction'],direction,atol=1e-14)
        json.dumps({k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in robot.plan.items()},allow_nan=False)
    else:assert robot.plan is None and not calls['ik'] and not calls['transit']


def test_default_grid_rejects_parallel_legacy_basis_before_any_candidate(monkeypatch):
    robot,calls,_,_=single_proposal_robot(monkeypatch)
    robot.cut_proposal=None;robot.goal=np.eye(4)
    with np.errstate(divide='raise',invalid='raise'):
        with pytest.raises(ValueError,match='no orientation-grid basis'):
            robot.plan_cut([],np.zeros(7))
    assert robot.plan is None and not calls['rigid'] and not calls['ik']
    assert robot.plan_diagnostics['endpoint_attempts']==[]
    json.dumps(robot.plan_diagnostics,allow_nan=False)
