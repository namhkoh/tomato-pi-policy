from types import SimpleNamespace as S
import numpy as np
import pytest
from .grasp_orientation_search import orientations,search


def fixture():
    point=np.array([.4,.2,1.]);goal=np.eye(4);goal[:3,3]=point+[0,0,.125]
    r=S(park_left_ready=False,grasp_skew=0.,approach_tilt=0.,grasp_roll=0,
        grasp_pitch=0.,grasp_depth=.125,approach_distance=.02,goal=goal,
        grasp_point=lambda f:point.copy(),rig=S(rest_frames=np.eye(4)[None]),
        initial_q=np.zeros(7),right=np.ones(7),base=np.eye(4),arc=.08,
        root='/World/Robot',stage=object(),self_screen=S(shapes=['all_shapes']))
    r.seam=lambda f:(np.zeros(3),np.array([1.,0.,0.]))
    r.kin=S(solve_pose=lambda side,pose,seed,base,**kw:S(succeeded=True,joint_degrees=tuple(seed)),
        inter_arm_clearance=lambda *a:S(clearance_m=.02))
    r.body_world=lambda l,r:dict(left=np.array(l),right=np.array(r))
    r.check_self=lambda *a:dict(passed=True)
    return r


def test_orientation_grid_preserves_exact_point_and_reconstructible_configuration():
    r=fixture();before=r.goal.copy();values=list(orientations(r))
    assert len(values)==111
    for start,goal,meta in values:
        np.testing.assert_allclose(goal[:3,3]-.125*goal[:3,2],[.4,.2,1.],atol=1e-14)
        np.testing.assert_allclose(start[:3,3]-goal[:3,3],.02*goal[:3,2],atol=1e-14)
        np.testing.assert_allclose(goal[:3,:3].T@goal[:3,:3],np.eye(3),atol=1e-14)
        assert np.linalg.det(goal[:3,:3])==pytest.approx(1.)
        assert abs(meta['grasp_pitch'])<=60 and meta['grasp_roll'] in (0,180)
        from .grasp_frame import pitch_for_pad_span
        y=np.array([0.,1.,0.]);z=np.array(meta['approach_vector']);rotation=np.column_stack([np.cross(y,z),y,z])
        if meta['grasp_roll']==180:rotation[:,:2]*=-1
        np.testing.assert_allclose(pitch_for_pad_span(rotation,meta['grasp_pitch']),goal[:3,:3],atol=1e-14)
    np.testing.assert_array_equal(r.goal,before)


@pytest.mark.parametrize('roll',[0,180])
@pytest.mark.parametrize('pitch',[-40.,20.])
def test_undo_original_roll_and_pitch_without_mutation(roll,pitch):
    from .grasp_frame import pitch_for_pad_span
    r=fixture();r.grasp_roll=roll;r.grasp_pitch=pitch
    if roll==180:r.goal[:3,:2]*=-1
    r.goal[:3,:3]=pitch_for_pad_span(r.goal[:3,:3],pitch)
    r.goal[:3,3]=r.grasp_point(None)+r.grasp_depth*r.goal[:3,2]
    before=r.goal.copy();values=list(orientations(r))
    assert len(values)==111
    assert values[0][2]['approach_vector']==pytest.approx([0,0,1])
    np.testing.assert_array_equal(r.goal,before)


@pytest.mark.parametrize('field,value',[('grasp_skew',1),('approach_tilt',1),('park_left_ready',True),('grasp_roll',90)])
def test_unsupported_frames_are_not_silently_repaired(field,value):
    r=fixture();setattr(r,field,value)
    with pytest.raises(ValueError):list(orientations(r))


def clear_fingers(monkeypatch):
    from . import grasp_target
    monkeypatch.setattr(grasp_target,'finger_seam_clearance',lambda *a:dict(minimum_finger_to_cut_plane_m=.02))


@pytest.fixture(autouse=True)
def rigid_hand_stub(monkeypatch):
    from . import rigid_grasp_screen
    monkeypatch.setattr(rigid_grasp_screen,'RigidGraspScreen',lambda *a:S(check=lambda *a:dict(passed=True)))


def test_proposal_needs_native_start_and_full_self_path_but_grants_no_motion(monkeypatch):
    clear_fingers(monkeypatch);r=fixture();calls=[];original=r.goal.copy()
    def native(world,shapes):
        calls.append(world);assert shapes==['all_shapes'];return dict(passed=True)
    out=search(r,S(check=native),lambda:None)
    assert out['proposed_grasp'] and len(calls)==8 and len(out['proposed_grasps'])==8
    assert len(out['proposed_grasp']['left_path_degrees'])==47
    assert out['candidates'][0]['native_startup_clear'] and out['candidates'][0]['left_self_path_clear']
    for field in ('motion_authorized','whole_path_certified','grasp_or_cut_verified','retention_verified','cut_entry_checked_with_left'):
        assert out[field] is False
    assert out['physics_steps']==0 and out['relaunch_required']
    np.testing.assert_array_equal(r.goal,original)
    np.testing.assert_array_equal(r.right,np.ones(7))


def test_native_obstruction_cannot_be_accepted(monkeypatch):
    clear_fingers(monkeypatch)
    out=search(fixture(),S(check=lambda *a:dict(passed=False)),lambda:None)
    assert out['proposed_grasp'] is None and len(out['candidates'])==111
    assert not any(row['native_startup_clear'] for row in out['candidates'])


def test_blocked_rigid_hand_never_spends_arm_ik_or_native_queries(monkeypatch):
    from . import rigid_grasp_screen
    clear_fingers(monkeypatch);r=fixture()
    monkeypatch.setattr(rigid_grasp_screen,'RigidGraspScreen',lambda *a:S(check=lambda *a:dict(passed=False)))
    r.kin.solve_pose=lambda *a,**k:pytest.fail('Blocked hand cannot be fixed by elbow IK')
    out=search(r,S(check=lambda *a:pytest.fail('Blocked hand cannot be a proposal')),lambda:None)
    assert out['proposed_grasp'] is None
    assert all(row['rejection']=='hand_target_corridor' for row in out['candidates'])
    assert out['rigid_hand_target_screened'] and not out['settled_target_checked']


def test_finger_cut_clearance_failure_never_reaches_native(monkeypatch):
    from . import grasp_target
    def reject(*a):raise ValueError('overlaps cut plane')
    monkeypatch.setattr(grasp_target,'finger_seam_clearance',reject)
    out=search(fixture(),S(check=lambda *a:pytest.fail('Unchecked finger overlap')),lambda:None)
    assert out['proposed_grasp'] is None
    assert all(row['rejection']=='finger_cut_clearance' for row in out['candidates'])


def test_path_interarm_failure_cannot_be_accepted(monkeypatch):
    clear_fingers(monkeypatch);r=fixture();calls=[0]
    def clearance(*a):
        calls[0]+=1;return S(clearance_m=.02 if calls[0]%2 else .005)
    r.kin.inter_arm_clearance=clearance
    out=search(r,S(check=lambda *a:dict(passed=True)),lambda:None)
    assert out['proposed_grasp'] is None


def test_final_control_reserve_and_epoch_are_mandatory(monkeypatch):
    clear_fingers(monkeypatch)
    out=search(fixture(),S(check=lambda *a:pytest.fail('No query reserve'),
        can_check_with_final_controls=lambda *a:False),lambda:None)
    assert out['proposed_grasp'] is None and out['budget_exhausted']
    assert out['query_budget_reserved_for_final_controls'] and len(out['candidates'])==1
    def stale():raise RuntimeError('stale scene')
    with pytest.raises(RuntimeError,match='stale scene'):search(fixture(),S(),stale)


def test_public_mode_validation_and_forwarding(monkeypatch,tmp_path):
    from . import ground_truth_trial,benchmark
    output=tmp_path/'unused'
    for extra in (['--mode','right_only'],['--mode','bimanual','--watch']):
        with pytest.raises(ValueError,match='zero-motion'):
            ground_truth_trial.main(['--output',str(output),'--screen-grasp-approach','--process-zone-trial',*extra])
    captured=[];monkeypatch.setattr(benchmark,'main',lambda argv:captured.append(benchmark.parser().parse_args(argv)))
    ground_truth_trial.main(['--output',str(output),'--mode','bimanual','--process-zone-trial','--screen-grasp-approach'])
    assert captured[0].native_startup_grasp_search and captured[0].require_retention_screen
    assert captured[0].native_startup_clearance and captured[0].native_static_clearance
    assert not output.exists()


def test_raw_cli_rejects_partial_profile_before_writing(tmp_path):
    from .benchmark import main
    with pytest.raises(ValueError,match='zero-motion'):
        main(['--output',str(tmp_path/'unused'),'--native-startup-grasp-search'])
    assert not (tmp_path/'unused').exists()
