"""Incremental planner contracts with synthetic geometry/IK, never native proof.

Exercise the REAL candidate, dense-stroke, transit and acceptance-wrapper code.
Only geometry/IK responses and the bounded transit-search backend are mocked.
No USD stage, SimulationApp, scene query, asset edit or robot motion is used.
These regressions intentionally fail until incremental planning is integrated.
"""
from types import SimpleNamespace as S

import numpy as np
import pytest

from sim_physics.bimanual import BimanualRobot


def grid_fixture(monkeypatch, *, eligible=(1,), faults=None, max_proposals=None,
                 native_validation=False, final_fault=None):
    import sim_physics.joint_path as path_module
    import sim_physics.rigid_tool_screen as rigid_module

    robot = BimanualRobot.__new__(BimanualRobot)
    robot.stage = None
    robot.root = '/World/SyntheticRobot'
    robot.rig = S(root='/World/SyntheticPlant')
    robot.plan = None
    robot.plan_diagnostics = None
    robot.cut_proposal = None  # The original 756-proposal grid, not a new family.
    robot.radius = .003
    robot.right = np.zeros(7)
    robot.base = np.eye(4)
    robot.goal = np.eye(4)
    robot.goal[:3, :3] = [[1., 0., 0.], [0., 0., 1.], [0., -1., 0.]]
    robot.stroke_offsets = np.linspace(-.025, .005, 61)
    centre = np.array([.12, -.23, 1.4])
    axis = np.array([0., 0., 1.])
    left = np.linspace(-3., 3., 7)
    robot.seam = lambda frames: (centre.copy(), axis.copy())
    robot.held_plant_screen = S(workspace=[np.full(3, -10.), np.full(3, 10.)],
        static=[], snapshot=lambda frames: None,
        last_failure={'synthetic_obstruction': True}, native_static_query=None)
    calls = {key: [] for key in ('rigid', 'ik', 'arm', 'self', 'scene',
                                 'transit', 'search', 'order', 'native')}
    state = S(candidate=0, pending_endpoint=False, phase='initial', stroke_index=-1)
    faults = {} if faults is None else faults
    eligible = set(eligible)

    def injected(check):
        fault = faults.get(state.candidate)
        target = fault[0] if isinstance(fault, tuple) else fault
        # Reject an INTERIOR stroke sample, after four accepted samples.
        hit = (target == state.phase+'_'+check
               and (state.phase != 'stroke' or state.stroke_index == 4))
        if hit and isinstance(fault, tuple):
            raise fault[1]
        return hit

    def wrist(point, direction, normal, wing):
        frame = np.eye(4)
        frame[:3, :3] = np.column_stack((-direction, np.cross(normal, -direction), normal))
        frame[:3, 3] = point
        return frame

    robot.knife = S(size=np.array([.002, .05, .006]), wrist_for_edge=wrist)

    class Screen:
        def __init__(self, supplied_robot, supplied_left):
            assert supplied_robot is robot
            np.testing.assert_array_equal(supplied_left, left)

        def check(self, frames):
            if state.candidate in eligible:
                assert any(row[0] == state.candidate and row[1] == 'stroke'
                           for row in calls['ik']), 'Next endpoint enumerated before trying prior eligible path'
            state.candidate += 1
            if max_proposals is not None:
                assert state.candidate <= max_proposals, 'Enumeration continued after sufficient candidates'
            assert robot.plan is None, 'A provisional/failed candidate leaked into enumeration'
            assert frames.shape == (61, 4, 4) and np.isfinite(frames).all()
            np.testing.assert_allclose(frames[:, :3, :3],
                np.repeat(frames[:1, :3, :3], 61, axis=0), rtol=0., atol=1e-14)
            np.testing.assert_allclose(np.linalg.norm(np.diff(frames[:, :3, 3], axis=0), axis=1),
                np.diff(robot.stroke_offsets), rtol=0., atol=1e-14)
            assert np.linalg.norm(frames[-1, :3, 3]-frames[0, :3, 3]) == pytest.approx(.03)
            calls['rigid'].append(state.candidate)
            calls['order'].append((state.candidate, 'rigid'))
            state.pending_endpoint = True
            state.phase = 'endpoint'
            state.stroke_index = -1
            return dict(passed=state.candidate in eligible, checked_frames=61, sample=60,
                        reason='synthetic_complete_corridor', native_validated=False)

    monkeypatch.setattr(rigid_module, 'RigidToolScreen', Screen)

    def solve(side, desired, seed, base, maximum_evaluations):
        assert side == 'right' and maximum_evaluations == 250
        if state.pending_endpoint:
            state.phase = 'endpoint'
            state.pending_endpoint = False
            np.testing.assert_array_equal(seed, robot.right)
        else:
            state.phase = 'stroke'
            state.stroke_index += 1
        calls['order'].append((state.candidate, state.phase+'_ik'))
        calls['ik'].append((state.candidate, state.phase, state.stroke_index, np.array(seed).copy()))
        # First endpoint is intentionally FARTHER from park than the second:
        # accepting it must not await distance sorting of remaining proposals.
        q = np.zeros(7)
        q[0] = 20. if state.candidate == 1 else 2.
        if state.phase == 'stroke':
            q[1] = state.stroke_index*.001
        return S(succeeded=not injected('ik'), position_error_m=0., orientation_error_rad=0.,
                 evaluations=1, joint_degrees=q)

    def arm(supplied_left, right, base):
        np.testing.assert_array_equal(supplied_left, left)
        calls['arm'].append((state.candidate, state.phase, np.array(right).copy()))
        return S(clearance_m=.009 if injected('arm') else .02)

    def self_check(supplied_left, right):
        calls['self'].append((state.candidate, state.phase, np.array(right).copy()))
        return {'passed': not injected('self')}

    def scene(supplied_left, right, *, stroke=False):
        calls['scene'].append((state.candidate, state.phase, np.array(right).copy()))
        assert stroke is (state.phase == 'stroke')
        return not injected('scene')

    robot.kin = S(solve_pose=solve, inter_arm_clearance=arm,
        arm_limits_degrees=lambda side: (np.full(7, -180.), np.full(7, 180.)))
    robot.check_self = self_check
    robot.check_held_plant = scene
    real_transit = robot.right_transit

    def transit(supplied_left, goal):
        state.phase = 'transit'
        calls['order'].append((state.candidate, 'transit'))
        calls['transit'].append(state.candidate)
        return real_transit(supplied_left, goal)

    robot.right_transit = transit

    def bounded_search(start, goal, lower, upper, valid):
        calls['search'].append(state.candidate)
        assert faults.get(state.candidate) in ('transit_arm', 'transit_self', 'transit_scene')
        assert not valid((start+goal)/2), 'Transit fallback bypassed the same obstruction'
        return None

    monkeypatch.setattr(path_module, 'connect_path', bounded_search)
    if native_validation:
        def validate():
            calls['native'].append('validate')
            assert robot.plan is None, 'Wrapper exposed a plan before final validation'
            assert calls['transit'], 'Final validation preceded complete path construction'
            if final_fault is not None:
                raise final_fault

        def close():
            calls['native'].append('close')

        def report():
            calls['native'].append('report')
            return dict(synthetic_test_only=True, query_count=0,
                        final_validation_passed='validate' in calls['native'] and final_fault is None)

        robot.held_plant_screen.native_static_query = S(validate=validate, close=close, report=report)
    return robot, calls, state, left, centre, axis


def test_first_full_path_stops_grid_without_waiting_for_nearer_endpoints(monkeypatch):
    robot, calls, _, left, centre, axis = grid_fixture(monkeypatch,
        eligible=(1, 2), max_proposals=1, native_validation=True)
    robot.plan_cut([], left)
    assert calls['rigid'] == [1]
    assert calls['order'] == [(1, 'rigid'), (1, 'endpoint_ik')] + [(1, 'stroke_ik')]*61 + [(1, 'transit')]
    assert len(calls['ik']) == 62 and calls['transit'] == [1] and not calls['search']
    assert calls['native'] == ['validate', 'close', 'report']
    assert robot.held_plant_screen.native_static_query is None
    assert len(robot.plan_diagnostics['endpoint_attempts']) == 1
    assert robot.plan_diagnostics['path_failures'] == []
    assert not robot.plan_diagnostics['whole_scene_path_certified']
    assert robot.plan_diagnostics['minimum_required_interarm_m'] == .01
    assert robot.plan_diagnostics['held_plant_margin_m'] == .001
    assert robot.plan['stroke'].shape == (61, 7)
    np.testing.assert_array_equal(robot.plan['stroke'][:, 0], np.full(61, 20.))
    np.testing.assert_array_equal(robot.plan['centre'], centre)
    np.testing.assert_array_equal(robot.plan['axis'], axis)
    for check in ('arm', 'self', 'scene'):
        assert sum(row[1] == 'endpoint' for row in calls[check]) == 1
        assert sum(row[1] == 'stroke' for row in calls[check]) == 61
        transit = [row[2] for row in calls[check] if row[1] == 'transit']
        np.testing.assert_array_equal(np.array(transit)[:, 0], np.arange(21.))
    assert robot.plan['minimum_interarm_m'] == .02


@pytest.mark.parametrize('failure,expected', [
    ('stroke_ik', 'stroke_IK'), ('stroke_arm', 'stroke_arm_clearance'),
    ('stroke_self', 'stroke_self_collision'), ('stroke_scene', 'stroke_held_plant'),
    ('transit_arm', 'bounded_transit_arm_self_or_plant_clearance'),
    ('transit_self', 'bounded_transit_arm_self_or_plant_clearance'),
    ('transit_scene', 'bounded_transit_arm_self_or_plant_clearance'),
])
def test_failed_first_path_tries_second_and_discards_every_first_path_sample(monkeypatch, failure, expected):
    robot, calls, _, left, _, _ = grid_fixture(monkeypatch,
        eligible=(1, 2), faults={1: failure}, max_proposals=2)
    robot.plan_cut([], left)
    assert calls['rigid'] == [1, 2]
    failures = robot.plan_diagnostics['path_failures']
    assert len(failures) == 1 and failures[0]['rejection'] == expected
    assert failures[0]['angle'] == 0 and robot.plan['angle'] == 15
    expected_first_stroke = 5 if failure.startswith('stroke') else 61
    assert sum(r[0] == 1 and r[1] == 'stroke' for r in calls['ik']) == expected_first_stroke
    assert sum(r[0] == 2 and r[1] == 'stroke' for r in calls['ik']) == 61
    assert calls['transit'] == ([2] if failure.startswith('stroke') else [1, 2])
    assert calls['search'] == ([] if failure.startswith('stroke') else [1])
    # Real helper/transit outputs must contain ONLY the second candidate, not
    # the first candidate's partial stroke or its failed shoulder detours.
    assert robot.plan['stroke'].shape == (61, 7)
    np.testing.assert_array_equal(robot.plan['stroke'][:, 0], np.full(61, 2.))
    np.testing.assert_allclose(robot.plan['stroke'][:, 1], np.arange(61)*.001, atol=0., rtol=0.)
    np.testing.assert_array_equal(robot.plan['approach'][:, 0], [0., 1., 2.])
    np.testing.assert_array_equal(robot.plan['approach'][:, 1:], np.zeros((3, 6)))
    first_second_seed = next(r[3] for r in calls['ik'] if r[0] == 2 and r[1] == 'stroke')
    np.testing.assert_array_equal(first_second_seed, [2., 0., 0., 0., 0., 0., 0.])
    np.testing.assert_array_equal(robot.right, np.zeros(7))


def test_all_756_rigid_rejections_still_skip_every_ik_and_path_attempt(monkeypatch):
    robot, calls, _, left, _, _ = grid_fixture(monkeypatch, eligible=())
    with pytest.raises(RuntimeError, match='endpoints=756, IK_attempted=0'):
        robot.plan_cut([], left)
    assert len(calls['rigid']) == 756
    assert not calls['ik'] and not calls['transit'] and not calls['search']
    attempts = robot.plan_diagnostics['endpoint_attempts']
    assert len(attempts) == 756
    assert all(not a['ik_attempted'] and a['rejection'] == 'rigid_tool_corridor' for a in attempts)
    assert robot.plan is None and robot.plan_diagnostics['path_failures'] == []


def test_failed_partial_stroke_then_grid_exhaustion_never_publishes_a_plan(monkeypatch):
    robot, calls, _, left, _, _ = grid_fixture(monkeypatch, faults={1: 'stroke_scene'})
    robot.plan = {'stale_plan': True}
    with pytest.raises(RuntimeError, match='endpoints=756, IK_attempted=1'):
        robot.plan_cut([], left)
    assert len(calls['rigid']) == 756 and len(calls['ik']) == 6
    assert not calls['transit'] and not calls['search']
    assert robot.plan is None
    assert [f['rejection'] for f in robot.plan_diagnostics['path_failures']] == ['stroke_held_plant']


@pytest.mark.parametrize('error_type', [RuntimeError, KeyboardInterrupt])
def test_candidate_exception_halts_grid_and_wrapper_discards_plan_and_closes(monkeypatch, error_type):
    original = error_type('synthetic path guard fault')
    robot, calls, _, left, _, _ = grid_fixture(monkeypatch,
        faults={1: ('stroke_scene', original)}, max_proposals=1, native_validation=True)
    robot.plan = {'stale_plan': True}
    with pytest.raises(error_type) as caught:
        robot.plan_cut([], left)
    assert caught.value is original and robot.plan is None
    assert calls['rigid'] == [1] and len(calls['ik']) == 6
    assert not calls['transit']
    assert calls['native'] == ['close', 'report']
    assert robot.held_plant_screen.native_static_query is None
    assert robot.planning_wall_seconds >= 0


def test_incremental_success_cannot_bypass_wrapper_final_validation_failure(monkeypatch):
    original = RuntimeError('synthetic final snapshot validation fault')
    robot, calls, _, left, _, _ = grid_fixture(monkeypatch,
        max_proposals=1, native_validation=True, final_fault=original)
    with pytest.raises(RuntimeError) as caught:
        robot.plan_cut([], left)
    assert caught.value is original and robot.plan is None
    assert calls['rigid'] == [1] and len(calls['ik']) == 62 and calls['transit'] == [1]
    assert calls['native'] == ['validate', 'close', 'report']
    assert robot.held_plant_screen.native_static_query is None
    assert not robot.plan_diagnostics['native_static_clearance']['final_validation_passed']


@pytest.mark.parametrize('failure', [None, 'stroke_scene', 'transit_scene'])
def test_extracted_candidate_helper_boolean_contract_and_input_preservation(monkeypatch, failure):
    robot, calls, state, left, centre, axis = grid_fixture(monkeypatch,
        faults={} if failure is None else {1: failure})
    state.candidate = 1
    direction = np.array([0., -1., 0.])
    endpoint = np.array([20., 0., 0., 0., 0., 0., 0.])
    candidate = (20., 0., direction.copy(), endpoint.copy(), 1, 0., axis.copy())
    original_arrays = [value.copy() for value in (left, centre, axis, candidate[2], candidate[3], candidate[6])]
    failures = []
    result = robot._try_cut_candidate(left, centre, axis, candidate, 0., failures)
    assert result is (failure is None)
    assert not calls['rigid']  # Endpoint screening belongs to the calling grid.
    assert all(row[1] == 'stroke' for row in calls['ik'])
    if failure is None:
        assert robot.plan is not None and failures == [] and calls['transit'] == [1]
        assert robot.plan['stroke'].shape == (61, 7)
    else:
        assert robot.plan is None and len(failures) == 1
        assert failures[0]['rejection'] == ('stroke_held_plant' if failure == 'stroke_scene'
                                          else 'bounded_transit_arm_self_or_plant_clearance')
    for value, original in zip((left, centre, axis, candidate[2], candidate[3], candidate[6]), original_arrays):
        np.testing.assert_array_equal(value, original)
