from types import SimpleNamespace as S
import numpy as np
import pytest
from .cut_station_orbit import left_seed_proposals,search
from .cut_station_orbit_test import fixture


def test_bounded_unique_seeds_do_not_mutate_original_or_alias_each_other():
    original=np.arange(7,dtype=float);before=original.copy()
    seeds=left_seed_proposals(original,True)
    assert len(seeds)==6
    np.testing.assert_array_equal(seeds[0],before)
    assert all(s.shape==(7,) and np.isfinite(s).all() for s in seeds)
    assert len({tuple(s) for s in seeds})==6
    seeds[0][0]=100
    np.testing.assert_array_equal(original,before)
    assert all(not np.shares_memory(a,b) for i,a in enumerate(seeds) for b in seeds[i+1:])
    assert len(left_seed_proposals(before))==1


def test_identical_sdk_and_original_seed_are_deduplicated():
    from greenhouse_sim.robot_ready_pose import SDK_READY_POSE_DEGREES
    q=[SDK_READY_POSE_DEGREES[f'left_arm_{i}'] for i in range(7)]
    assert len(left_seed_proposals(q,True))==3


@pytest.mark.parametrize('q,expanded',[(np.zeros(6),True),([np.nan]*7,True),([np.inf]*7,False),(np.zeros(7),1)])
def test_invalid_seed_or_nonboolean_mode_rejected(q,expanded):
    with pytest.raises(ValueError):left_seed_proposals(q,expanded)


def test_alternate_elbow_needs_fresh_native_endpoints_and_same_wrist_goal(monkeypatch):
    import sim_physics.cut_station_orbit as m
    monkeypatch.setattr(m,'stations',lambda *a:iter(()))
    r,poses=fixture();r.station_left_seed_search=True;initial=r.initial_q.copy();calls=[]
    def native(world,shapes):
        assert shapes==['all_original_shapes'];calls.append(world['left'].copy())
        return dict(passed=bool(world['left'][6]<-110))
    result=search(r,S(check=native),lambda:None)
    assert result['expanded_left_seed_search'] and result['left_seed_candidates']==6
    assert result['maximum_candidates']==295*6
    assert result['proposed_station']['left_ik_seed_degrees'][6]<-110
    assert sum(q[6]<-110 for q in calls)==2
    assert result['candidates'][-1]['left_self_path_clear']
    assert all(np.array_equal(p,np.eye(4)) for side,p in poses if side=='left')
    np.testing.assert_array_equal(r.initial_q,initial)
    assert not result['motion_authorized'] and not result['whole_path_certified']
    assert result['physics_steps']==0 and result['relaunch_required']


def test_alternate_elbows_do_not_relax_interarm_clearance(monkeypatch):
    import sim_physics.cut_station_orbit as m
    monkeypatch.setattr(m,'stations',lambda *a:iter(()))
    r,_=fixture();r.station_left_seed_search=True
    r.kin.inter_arm_clearance=lambda *a:S(clearance_m=.005)
    result=search(r,S(check=lambda *a:dict(passed=True)),lambda:None)
    assert result['proposed_station'] is None
    assert len(result['candidates'])==6
    assert not any(row['left_self_path_clear'] for row in result['candidates'])


def test_no_queries_or_other_seeds_after_final_control_reserve_stop():
    r,_=fixture();r.station_left_seed_search=True
    def fail(*args):pytest.fail('Native query consumed final-control reserve')
    result=search(r,S(check=fail,can_check_with_final_controls=lambda *a:False),lambda:None)
    assert result['budget_exhausted'] and result['query_budget_reserved_for_final_controls']
    assert len(result['candidates'])==1 and result['proposed_station'] is None


def test_left_seed_search_cannot_change_parked_arm():
    r,_=fixture();r.station_left_seed_search=True;r.park_left_ready=True
    with pytest.raises(ValueError,match='parked arm'):search(r,S(),lambda:None)


@pytest.mark.parametrize('module',['ground_truth_trial','benchmark'])
def test_cli_rejects_motion_before_output_creation(tmp_path,module):
    from importlib import import_module
    output=tmp_path/'not_created'
    args=['--output',str(output),'--station-left-seed-search']
    if module=='ground_truth_trial':args+=['--mode','bimanual']
    with pytest.raises(ValueError,match='zero-motion bimanual'):
        import_module('sim_physics.'+module).main(args)
    assert not output.exists()


def test_public_right_only_search_rejected(tmp_path):
    from .ground_truth_trial import main
    with pytest.raises(ValueError,match='zero-motion bimanual'):
        main(['--output',str(tmp_path/'not_created'),'--mode','right_only',
              '--screen-station','--cut-station-orbit','--station-left-seed-search'])
    assert not (tmp_path/'not_created').exists()


def test_public_flag_forwards_only_a_guarded_zero_motion_search(tmp_path,monkeypatch):
    from . import benchmark
    from .ground_truth_trial import main
    captured=[]
    monkeypatch.setattr(benchmark,'main',lambda argv:captured.append(benchmark.parser().parse_args(argv)))
    output=tmp_path/'not_created'
    main(['--output',str(output),'--mode','bimanual','--process-zone-trial',
          '--screen-station','--cut-station-orbit','--station-left-seed-search'])
    a=captured[0]
    assert a.station_left_seed_search and a.native_startup_station_search and a.cut_station_orbit
    assert a.native_startup_clearance and a.native_static_clearance and a.require_retention_screen
    assert not a.right_only_cut_trial and not a.park_left_ready
    assert not output.exists()
