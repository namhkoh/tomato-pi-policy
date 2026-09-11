"""Mock native views/epoch only: never launch or import a SimulationApp."""
from copy import deepcopy
from types import SimpleNamespace as S
import json

import numpy as np
import pytest

from sim_physics.held_plant_screen import HeldPlantScreen
from sim_physics.native_static_clearance import NativeStaticClearance
from sim_physics.withdrawal_native_check import check


def setup(monkeypatch, *, failure=None):
    import sim_physics.native_static_clearance as native_module
    state = S(factory_calls=0, wrist_reads=0, body_reads=0, checks=[], queries=[],
              closes=0, revision=0, native=None, fk_calls=[], failure=failure)
    root = '/World/Robot'
    links = ['ee_right', 'link_right_arm_2', 'ee_left', 'ee_finger_l1', 'ee_finger_l2']
    paths = [root+'/'+link for link in links]
    # Deliberately scrambled native path order, with per-body non-FK positions.
    native_paths = paths[::-1]
    poses = np.array([[float(links.index(p.rsplit('/', 1)[1]))*.02, 0., 0., 0., 0., 0., 1.]
                      for p in native_paths])
    wrist = np.array([[0., 0., 0., 0., 0., 0., 1.]])
    def wrist_read():
        state.wrist_reads += 1
        return wrist
    def body_read():
        state.body_reads += 1
        return poses
    shapes = [(root+'/'+link+'/shape', root+'/'+link, link, 'box', None) for link in links]
    class Self:
        def __init__(self):
            self.shapes = shapes
        def check(self, world):
            state.checks.append(('self', deepcopy(world), {}))
            if state.failure == 'interrupt': raise KeyboardInterrupt('test interrupt')
            if state.failure == 'self_exception': raise RuntimeError('self unavailable')
            if state.failure == 'epoch': state.revision += 1
            return dict(passed=state.failure != 'self', all_shape_bounds_screened=state.failure != 'unsupported',
                unsupported_shapes=['missing'] if state.failure == 'unsupported' else [],
                checked_pairs=5, minimum_clearance_m=.004, required_margin_m=.003,
                whole_robot_self_collision_certified=False)
    class Held(HeldPlantScreen):
        def __init__(self):
            self.arm = 'right'; self.shapes = shapes[:2]
            self.local = [(f'/World/Plant/Segment_{i:03d}/StemCollider', i, 'capsule', None) for i in range(15)]
            self.static = [('/World/Obstacle', 'box', None, np.zeros(3), np.ones(3))]
            self.static_indices = {}
            self.workspace = (np.full(3, -2.), np.full(3, 2.))
            self.obstacles = ['old-controller-snapshot']; self.lower = np.array([[-99.]])
            self.upper = np.array([[99.]]); self.triangle_indices = {'old': 'unchanged'}
            self.last_failure = {'old': True}; self.native_static_query = S(old_query=True)
        def snapshot(self, frames):
            self.obstacles = ['fresh', np.array(frames, copy=True)]
            self.lower = np.zeros((1, 3)); self.upper = np.ones((1, 3))
            self.triangle_indices = {'new': True}
        def check(self, world, *, stroke=False, grasp=False, margin=.001):
            state.checks.append((self.arm, deepcopy(world), dict(stroke=stroke, grasp=grasp, margin=margin,
                grasp_colliders=sorted(getattr(self, 'grasp_colliders', ())),
                frames=self.obstacles[1].copy(), fresh_query=self.native_static_query is state.native)))
            assert self.obstacles[0] == 'fresh'
            if state.failure == self.arm:
                self.last_failure = {'robot_collider': self.arm, 'plant_collider': '/World/Leaf', 'conservative_overlap': True}
                return False
            if self.arm == 'right':
                assert self.native_static_query.clear_box_checked('/World/Obstacle', [2, 2, 2], np.eye(3), [.1]*3, margin)
            self.last_failure = None
            return True
    def forbidden(*args, **kwargs):
        raise AssertionError('Command FK, robot motion or existing query must not be used')
    def forward(arm, q, base):
        state.fk_calls.append((arm, q.copy(), base.copy()))
        return np.eye(4)
    body_paths = [f'/World/Plant/Segment_{i:03d}' for i in range(15)]
    fixture = S(root=root, stage=object(), right=np.arange(7, dtype=float), base=np.eye(4),
        kin=S(forward=forward), body_world=forbidden, command_right=forbidden,
        right_palm=S(prim_paths=[paths[0]], count=1, get_transforms=wrist_read),
        robot_bodies=S(prim_paths=native_paths, get_transforms=body_read), body_paths=paths,
        self_screen=Self(), held_plant_screen=Held(),
        rig=S(body_paths=body_paths, cut_index=1), grasp_path=body_paths[3],
        planning_slides={'keep': .123}, native_static_planning_seconds=60.)
    frames = np.repeat(np.eye(4)[None], 15, axis=0)
    frames[:, 2, 3] = 1 + np.arange(15)/100
    def factory(stage, records, **kwargs):
        state.factory_calls += 1
        assert stage is fixture.stage and kwargs == {}  # retain existing 8-second default
        assert records is not fixture.held_plant_screen.static
        if state.failure == 'factory': raise RuntimeError('factory coverage unavailable')
        def guard():
            if state.revision: raise RuntimeError('native epoch advanced')
        def query(path, centre, axes, half):
            state.queries.append((path, np.array(centre), np.array(half)))
            if state.failure == 'budget' and len(state.queries) == 2:
                raise RuntimeError('query_budget_exhausted')
            coverage = np.allclose(centre, [.5]*3)
            if state.failure == 'final_coverage' and len(state.queries) >= 3:
                return False
            return bool(coverage)
        def close():
            state.closes += 1
            if state.failure == 'close': raise RuntimeError('unsubscribe failed')
            if state.failure == 'close_epoch': state.revision += 1
        def epoch():
            return dict(revision=state.revision, subscriptions_closed=state.closes > 0,
                cleanup_errors=[], invalidation_reasons=['advanced'] if state.revision else [])
        state.native = NativeStaticClearance(query, records, guard=guard, close_guard=close, epoch_report=epoch)
        if state.failure == 'report':
            state.native.report = lambda: (_ for _ in ()).throw(RuntimeError('report unavailable'))
        if state.failure == 'unvalidated_report':
            original = state.native.report
            state.native.report = lambda: dict(original(), final_validation_passed=False)
        return state.native
    monkeypatch.setattr(native_module, 'current_scene_query', factory)
    return fixture, frames, state, wrist, poses


def test_actual_native_bodies_fresh_plant_both_screens_and_epoch_close(monkeypatch):
    f, frames, s, _, _ = setup(monkeypatch)
    h = f.held_plant_screen; old_snapshot = h.obstacles; old_query = h.native_static_query
    old_lower = h.lower; old_upper = h.upper; old_indices = h.triangle_indices
    old_slides = f.planning_slides; before_frames = frames.copy()
    r = check(f, frames, (2, 4800))
    assert r['right_withdrawal_completed'] and r['clearance_verified']
    assert r['measured_sample_id'] == [2, 4800] and r['same_native_sample']
    assert s.factory_calls == 1 and s.closes == 1 and len(s.queries) == 3
    assert r['native_static']['final_coverage_checked'] == ['/World/Obstacle']
    assert r['native_static']['closed'] and r['native_static']['final_validation_passed']
    assert s.wrist_reads == 1 and s.body_reads == 1 and len(s.fk_calls) == 1
    assert s.fk_calls[0][0] == 'right'
    np.testing.assert_array_equal(s.fk_calls[0][1], f.right)
    assert [x[0] for x in s.checks] == ['self', 'right', 'left']
    for _, world, _ in s.checks:
        assert world['ee_finger_l2'][0, 3] == .08  # actual native, not FK/open aperture
        assert world['link_right_arm_2'][0, 3] == .02
    right = s.checks[1][2]; left = s.checks[2][2]
    assert right['stroke'] is False and right['grasp'] is False
    assert left['grasp'] is True and left['stroke'] is False
    assert right['margin'] == left['margin'] == .001
    assert left['grasp_colliders'] == [f'/World/Plant/Segment_{i:03d}/StemCollider' for i in (2, 3, 4)]
    assert right['fresh_query'] and left['fresh_query']
    np.testing.assert_array_equal(right['frames'], frames)
    assert h.obstacles is old_snapshot and h.native_static_query is old_query
    assert h.lower is old_lower and h.upper is old_upper and h.triangle_indices is old_indices
    assert h.last_failure == {'old': True} and f.planning_slides is old_slides
    assert h.arm == 'right' and not hasattr(h, 'grasp_colliders')
    np.testing.assert_array_equal(frames, before_frames)
    assert not r['whole_path_certified'] and not r['whole_scene_native_collision_certified']
    assert not r['full_forward_cutstroke_verified'] and not r['physical_cut_verified']
    json.dumps(r, allow_nan=False)


@pytest.mark.parametrize('fault', ['position', 'orientation'])
def test_nonendpoint_performs_no_body_screen_or_native_query(monkeypatch, fault):
    f, frames, s, wrist, _ = setup(monkeypatch)
    if fault == 'position': wrist[0, 0] = .001
    else: wrist[0, 3:] = [0, 0, 1, 0]
    # Neither plant cache nor body-view access is needed off endpoint.
    del f.held_plant_screen
    r = check(f, None, 4800)
    assert not r['endpoint_attained'] and not r['right_withdrawal_completed']
    assert not r['clearance_attempted'] and s.factory_calls == s.body_reads == 0
    assert s.checks == [] and r['native_static'] is None


@pytest.mark.parametrize('failure', ['self', 'unsupported', 'right', 'left', 'self_exception',
    'epoch', 'budget', 'final_coverage', 'close', 'close_epoch', 'report', 'unvalidated_report'])
def test_every_failed_check_epoch_validation_or_cleanup_fails_closed(monkeypatch, failure):
    f, frames, s, _, _ = setup(monkeypatch, failure=failure)
    r = check(f, frames, 4800)
    assert not r['right_withdrawal_completed'] and not r['clearance_verified']
    assert not r['whole_path_certified'] and not r['whole_scene_native_collision_certified']
    assert s.closes == 1 and s.native.closed
    if failure in ('right', 'left'):
        assert r[failure+'_scene']['failure']['plant_collider'] == '/World/Leaf'
    json.dumps(r, allow_nan=False)


def test_factory_failure_reports_unknown_query_count_no_fake_coverage(monkeypatch):
    f, frames, s, _, _ = setup(monkeypatch, failure='factory')
    r = check(f, frames, 4800)
    assert not r['right_withdrawal_completed']
    assert r['native_static']['initialization_status'] == 'failed'
    assert r['native_static']['query_count'] is None and not r['native_static']['query_count_known']
    assert s.closes == 0 and r['errors']


def test_interrupt_still_closes_owned_epoch_without_touching_controller(monkeypatch):
    f, frames, s, _, _ = setup(monkeypatch, failure='interrupt')
    old = f.held_plant_screen.native_static_query
    with pytest.raises(KeyboardInterrupt): check(f, frames, 4800)
    assert s.closes == 1 and s.native.closed and f.held_plant_screen.native_static_query is old


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'wrong_root', 'short_data', 'bad_pose', 'wrist_disagreement'])
def test_native_robot_mapping_coverage_and_same_sample_wrist_are_mandatory(monkeypatch, fault):
    f, frames, s, _, poses = setup(monkeypatch)
    if fault == 'missing': f.body_paths = f.body_paths[:-1]
    elif fault == 'duplicate': f.robot_bodies.prim_paths[1] = f.robot_bodies.prim_paths[0]
    elif fault == 'wrong_root':
        f.body_paths[0] = f.robot_bodies.prim_paths[-1] = '/World/Other/ee_right'
    elif fault == 'short_data': f.robot_bodies.get_transforms = lambda: poses[:-1]
    elif fault == 'bad_pose': poses[0, 0] = np.nan
    else: poses[-1, 0] = .00001
    r = check(f, frames, 4800)
    assert not r['right_withdrawal_completed'] and r['errors']
    assert s.closes == 1


@pytest.mark.parametrize('fault', ['short_frames', 'nan_frames', 'scale_frames', 'empty_static', 'missing_right_shape', 'bad_grasp'])
def test_incomplete_or_stale_geometry_inputs_never_query_or_complete(monkeypatch, fault):
    f, frames, s, _, _ = setup(monkeypatch)
    if fault == 'short_frames': frames = frames[:-1]
    elif fault == 'nan_frames': frames[0, 0, 3] = np.nan
    elif fault == 'scale_frames': frames[0, 0, 0] = 2
    elif fault == 'empty_static': f.held_plant_screen.static = []
    elif fault == 'missing_right_shape': f.held_plant_screen.shapes = f.held_plant_screen.shapes[:1]
    else: f.grasp_path = f.rig.body_paths[0]
    r = check(f, frames, 4800)
    assert not r['right_withdrawal_completed'] and r['errors'] and s.factory_calls == 0


@pytest.mark.parametrize('step', [-1, True, 1., None, (0, -1), (True, 1)])
def test_malformed_step_identifiers_rejected(monkeypatch, step):
    f, frames, s, _, _ = setup(monkeypatch)
    with pytest.raises(ValueError): check(f, frames, step)
    assert s.factory_calls == 0


def test_no_prior_success_latch_or_reuse_of_closed_query(monkeypatch):
    f, frames, s, wrist, _ = setup(monkeypatch)
    first = check(f, frames, 4800)
    assert first['right_withdrawal_completed']
    wrist[0, 0] = .01
    second = check(f, frames, 4801)
    assert not second['right_withdrawal_completed'] and s.factory_calls == 1
