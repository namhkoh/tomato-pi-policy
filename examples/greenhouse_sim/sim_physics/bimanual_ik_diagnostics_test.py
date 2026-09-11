"""Exact failed stroke IK diagnostics; synthetic responses, no native runtime."""
import json
from types import SimpleNamespace as S

import numpy as np
import pytest

from sim_physics.bimanual import BimanualRobot


def candidate_fixture(*, failure=None, failure_index=3):
    robot=BimanualRobot.__new__(BimanualRobot)
    robot.plan=None
    robot.base=np.eye(4)
    robot.stroke_offsets=np.linspace(-.025,.005,61)
    centre=np.array([.1,.2,1.3]);axis=np.array([0.,0.,1.])
    direction=np.array([1.,0.,0.]);initial=np.arange(7,dtype=float)
    candidate=(0.,45.,direction,initial,1,0.,axis)
    calls={name:[] for name in ('ik','arm','self','scene','transit')}
    results=[]

    def wrist(point,*unused):
        frame=np.eye(4);frame[:3,3]=point;return frame

    def solve(desired,seed):
        index=len(calls['ik'])
        calls['ik'].append((desired.copy(),np.array(seed,copy=True)))
        result=S(joint_degrees=initial+(index+1)*.001,
            position_error_m=.000123456789,orientation_error_rad=.000987654321,
            evaluations=250,succeeded=not (failure=='ik' and index==failure_index))
        results.append(result)
        return result

    def hit(kind):return failure==kind and len(calls['ik'])-1==failure_index

    def arm(left,right,base):
        calls['arm'].append(np.array(right,copy=True))
        return S(clearance_m=.009 if hit('arm') else .02)

    def self_check(left,right):
        calls['self'].append(np.array(right,copy=True))
        return dict(passed=not hit('self'),sample_index=len(calls['ik'])-1)

    def scene(left,right,*,stroke):
        assert stroke is True
        calls['scene'].append(np.array(right,copy=True))
        return not hit('scene')

    def transit(left,q):
        calls['transit'].append(np.array(q,copy=True))
        if failure=='transit':return None
        return np.array([q,q]),.02,dict(method='synthetic_checked_transit')

    robot.knife=S(wrist_for_edge=wrist)
    robot.solve_right_pose=solve
    robot.kin=S(inter_arm_clearance=arm)
    robot.check_self=self_check;robot.check_held_plant=scene
    robot.held_plant_screen=S(last_failure={'reason':'synthetic_held_plant'})
    robot.right_transit=transit
    failures=[]
    args=(np.zeros(7),centre,axis,candidate,-12.,failures)
    return robot,args,calls,results


@pytest.mark.parametrize('index',[0,57,60])
def test_failed_ik_preserves_exact_result_and_zero_based_sample(index):
    robot,args,calls,results=candidate_fixture(failure='ik',failure_index=index)
    assert robot._try_cut_candidate(*args) is False
    assert robot.plan is None and len(args[-1])==1
    failure=args[-1][0];result=results[-1]
    assert failure['rejection']=='stroke_IK'
    assert failure['sample_index']==index
    assert failure['offset_m']==float(robot.stroke_offsets[index])
    assert failure['ik_result']==dict(joint_degrees=result.joint_degrees.tolist(),
        position_error_m=result.position_error_m,orientation_error_rad=result.orientation_error_rad,
        evaluations=result.evaluations,succeeded=False)
    # Small residuals must not override the solver's unsuccessful status.
    assert len(calls['ik'])==index+1 and not calls['transit']
    assert all(len(calls[name])==index for name in ('arm','self','scene'))
    for i,(_,seed) in enumerate(calls['ik']):
        expected=args[3][3] if i==0 else results[i-1].joint_degrees
        np.testing.assert_array_equal(seed,expected)
        np.testing.assert_array_equal(calls['ik'][i][0][:3,3],
            args[1]+robot.stroke_offsets[i]*args[3][2])
    if index:
        assert failure['self_screen']['sample_index']==index-1
    else:
        assert 'self_screen' not in failure
    saved=json.loads(json.dumps(failure,allow_nan=False))
    result.joint_degrees[0]=999.
    assert failure==saved  # No alias to the solver's mutable vector.


def test_failed_ik_preserves_unspecified_evaluation_count():
    from greenhouse_sim.robot_kinematics import IKResult
    robot,args,calls,_=candidate_fixture(failure='ik',failure_index=0)
    original=robot.solve_right_pose
    def solve(desired,seed):
        result=original(desired,seed)
        return IKResult(joint_degrees=tuple(result.joint_degrees),
            position_error_m=result.position_error_m,orientation_error_rad=result.orientation_error_rad,
            cost=0.,succeeded=False)  # evaluations defaults to None in the public result.
    robot.solve_right_pose=solve
    assert robot._try_cut_candidate(*args) is False
    failure=args[-1][0]
    assert failure['rejection']=='stroke_IK' and failure['sample_index']==0
    assert failure['ik_result']['evaluations'] is None
    assert json.loads(json.dumps(failure,allow_nan=False))['ik_result']['evaluations'] is None
    assert len(calls['ik'])==1 and not calls['transit'] and robot.plan is None


def test_success_still_solves_all_samples_before_transit():
    robot,args,calls,results=candidate_fixture()
    assert robot._try_cut_candidate(*args) is True
    assert args[-1]==[]
    assert all(len(calls[name])==61 for name in ('ik','arm','self','scene'))
    assert len(calls['transit'])==1
    np.testing.assert_array_equal(calls['transit'][0],args[3][3])
    np.testing.assert_array_equal(robot.plan['stroke'],[r.joint_degrees for r in results])
    assert robot.plan['stroke_samples']==61


@pytest.mark.parametrize('kind,rejection',[
    ('arm','stroke_arm_clearance'),('self','stroke_self_collision'),
    ('scene','stroke_held_plant'),('transit','bounded_transit_arm_self_or_plant_clearance')])
def test_other_rejections_do_not_acquire_failed_ik_evidence(kind,rejection):
    robot,args,calls,_=candidate_fixture(failure=kind)
    assert robot._try_cut_candidate(*args) is False
    failure=args[-1][0]
    assert failure['rejection']==rejection and robot.plan is None
    assert 'ik_result' not in failure and 'sample_index' not in failure
    assert len(calls['ik'])==(61 if kind=='transit' else 4)
    assert len(calls['transit'])==(1 if kind=='transit' else 0)


def test_solver_exception_propagates_without_retry_or_invented_result():
    robot,args,calls,_=candidate_fixture()
    error=RuntimeError('synthetic solver error')
    def fail(*unused):
        calls['ik'].append('attempt')
        raise error
    robot.solve_right_pose=fail
    with pytest.raises(RuntimeError) as caught:robot._try_cut_candidate(*args)
    assert caught.value is error
    assert calls['ik']==['attempt'] and not calls['transit']
    assert args[-1]==[] and robot.plan is None
