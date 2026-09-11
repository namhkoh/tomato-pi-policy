"""Pure fake-FK/receipt tests; no native execution or physical qualification."""
from copy import deepcopy
import json

import numpy as np
import pytest

from sim_physics import measured_withdrawal as mw


def q(x):
    a = np.zeros(7); a[0] = x
    return a


def fk(joints):
    pose = np.eye(4); pose[0,3] = joints[0]*.001; pose[1,3] = joints[2]*.001
    a = np.radians(joints[1]); c,s = np.cos(a),np.sin(a)
    pose[:3,:3] = [[c,-s,0],[s,c,0],[0,0,1]]
    return pose


def measurement(x=2.5, step=10):
    pose = fk(q(x)).tolist()
    return dict(sample_id=[3,step], right_q_deg=q(x).tolist(), right_wrist_world=pose,
        right_wrist_path='/Robot/ee_right',
        hold_targets=dict(left_goal_world=np.eye(4).tolist(), finger_targets_m=[.002,.002]),
        scene=dict(sample_id=[3,step], robot_body_world={'/Robot/ee_right':pose,'/Robot/ee_left':np.eye(4).tolist()},
                   plant_body_world={'/Plant/Stem':np.eye(4).tolist()},
                   left_q_deg=q(10).tolist(), finger_slides_m={'left1':.002,'left2':.002}))


def receipt(request):
    """Adapter contract EXAMPLE only; these booleans are deliberately fake."""
    return dict(passed=True, request_sha256=request['request_sha256'],
        snapshot_sha256=request['snapshot_sha256'], sample_id=request['sample_id'],
        margins_m=deepcopy(request['margins_m']), errors=[],
        checked_samples=len(request['path_q_deg']), checked_segments=len(request['phases']),
        checks={k:True for k in mw.CHECKS},
        native_static=dict(final_validation_passed=True, closed=True, errors=[],query_count=1,
            epoch=dict(revision=0,subscriptions_closed=True,cleanup_errors=[],invalidation_reasons=[]),
            used_static_colliders=['/Scene/Obstacle'],final_coverage_checked=['/Scene/Obstacle']))


def setup(*, validate=None, m=None, **changes):
    requests=[]
    def validator(request):
        requests.append(deepcopy(request))
        return receipt(request) if validate is None else validate(request)
    args=dict(stroke=np.array([q(i) for i in range(5)]),
        approach=np.array([q(i) for i in (-4,-2,0)]), measurement=measurement() if m is None else m,
        lower_deg=np.full(7,-180.),upper_deg=np.full(7,180.), max_speed_deg_s=np.full(7,10.),
        dt=.01,wrist_fk=fk,edge_in_wrist=np.eye(4),stroke_direction=[1.,0,0],validate=validator)
    args.update(changes)
    return mw.MeasuredWithdrawal(**args),requests,args


def next_sample(m, target=None):
    m=deepcopy(m);m['sample_id'][1]+=1;m['scene']['sample_id']=m['sample_id'].copy()
    if target is not None:
        m['right_q_deg']=list(target);m['right_wrist_world']=fk(target).tolist()
        m['scene']['robot_body_world']['/Robot/ee_right']=m['right_wrist_world']
    return m


def test_measured_start_only_backward_existing_prefix_and_reverse_approach():
    controller,requests,args=setup()
    initial=controller.initial_command();path=np.array(requests[0]['path_q_deg'])
    assert initial['target_right_q_deg']==q(2.5).tolist()
    np.testing.assert_array_equal(path[0],q(2.5));np.testing.assert_array_equal(path[-1],q(-4))
    assert path[:,0].max()==2.5 and np.all(np.diff(path[:,0])<=0)
    assert np.max(abs(np.diff(path,axis=0)))<=1
    assert requests[0]['margins_m']==dict(self_m=.003,interarm_m=.01,scene_m=.001)
    assert set(requests[0]['phases'])=={'extract','return'}
    assert requests[0]['seam_allowance']==[p=='extract' for p in requests[0]['phases']]
    assert args['measurement']==measurement()
    assert not initial['whole_path_certified'] and not initial['full_forward_cutstroke_verified']


def test_repeated_static_measurements_never_advance_by_elapsed_steps():
    controller,requests,args=setup();m=args['measurement']
    for _ in range(100):
        m=next_sample(m);result=controller.advance(m)
        assert result['status']=='moving' and result['waypoint_index']==1
        assert result['target_right_q_deg'][0]==pytest.approx(2.4)
        assert not result['right_withdrawal_completed']
    assert len({r['snapshot_sha256'] for r in requests})==len(requests)


def test_every_command_rate_bounded_freshly_screened_and_final_measured():
    controller,requests,args=setup();m=args['measurement'];hold=deepcopy(m['hold_targets'])
    target=controller.initial_command()['target_right_q_deg']
    for _ in range(200):
        m=next_sample(m,target);result=controller.advance(m)
        assert result['status']!='blocked',result
        assert result['hold_targets']==hold and result['receipt']['sample_id']==m['sample_id']
        if result['status']=='complete':break
        target=result['target_right_q_deg']
        assert np.max(abs(np.array(target)-m['right_q_deg']))<=.1+1e-14
    else:pytest.fail('No measured completion')
    assert result['target_right_q_deg'] is None and result['right_withdrawal_completed']
    assert requests[-1]['kind']=='endpoint' and requests[-1]['phases']==[]
    assert requests[-1]['seam_allowance']==[] and result['native_query_count']==len(requests)
    assert not result['native_provenance_verified']


@pytest.mark.parametrize('fault',['sample','episode','scene_step','hold','inventory','wrist_path',
                                'joint_limit','nonfinite','pose_mismatch','reflection','slide'])
def test_bad_next_measurement_blocks_without_command_or_callback(fault):
    controller,requests,args=setup();m=next_sample(args['measurement'])
    if fault=='sample':m['sample_id'][1]+=1
    elif fault=='episode':m['sample_id'][0]+=1
    elif fault=='scene_step':m['scene']['sample_id'][1]-=1
    elif fault=='hold':m['hold_targets']['finger_targets_m'][0]+=.0001
    elif fault=='inventory':m['scene']['plant_body_world']['/Plant/Leaf']=np.eye(4).tolist()
    elif fault=='wrist_path':m['right_wrist_path']='/Robot/other'
    elif fault=='joint_limit':m['right_q_deg'][0]=180
    elif fault=='nonfinite':m['right_q_deg'][0]=float('nan')
    elif fault=='pose_mismatch':m['right_q_deg'][0]+=2
    elif fault=='reflection':m['right_wrist_world'][0][0]=-1
    else:m['scene']['finger_slides_m']['left1']=True
    result=controller.advance(m)
    assert result['status']=='blocked' and result['target_right_q_deg'] is None
    assert len(requests)==1
    assert controller.advance(next_sample(args['measurement']))['status']=='blocked'
    with pytest.raises(ValueError):controller.initial_command()


@pytest.mark.parametrize('fault',['bool','missing','truthy','hash','snapshot','sample','sample_bool',
    'counts','counts_bool','margins','check','errors','native_pass','close','revision','revision_bool',
    'cleanup','epoch_changed','coverage','coverage_duplicate','query_zero','query_bool','query_budget','nan'])
def test_malformed_receipts_cannot_authorize_motion(fault):
    calls=0
    def bad(request):
        nonlocal calls
        calls+=1;r=receipt(request)
        if calls==1:return r
        if fault=='bool':return True
        if fault=='missing':return {'passed':True}
        if fault=='truthy':r['passed']=1
        elif fault=='hash':r['request_sha256']='0'*64
        elif fault=='snapshot':r['snapshot_sha256']='0'*64
        elif fault=='sample':r['sample_id']=[3,10]
        elif fault=='sample_bool':r['sample_id']=[True,True]
        elif fault=='counts':r['checked_segments']=0
        elif fault=='counts_bool':r['checked_samples']=True
        elif fault=='margins':r['margins_m']['self_m']=.002
        elif fault=='check':r['checks']['left_scene']=False
        elif fault=='errors':r['errors']=['failed']
        elif fault=='native_pass':r['native_static']['final_validation_passed']=False
        elif fault=='close':r['native_static']['closed']=False
        elif fault=='revision':r['native_static']['epoch']['revision']=1
        elif fault=='revision_bool':r['native_static']['epoch']['revision']=False
        elif fault=='cleanup':r['native_static']['epoch']['cleanup_errors']=['error']
        elif fault=='epoch_changed':r['native_static']['epoch']['invalidation_reasons']=['step']
        elif fault=='coverage':r['native_static']['final_coverage_checked']=[]
        elif fault=='coverage_duplicate':r['native_static']['final_coverage_checked']*=2
        elif fault=='query_zero':r['native_static']['query_count']=0
        elif fault=='query_bool':r['native_static']['query_count']=True
        elif fault=='query_budget':r['native_static']['query_count']=mw.MAX_QUERIES+1
        elif fault=='nan':r['extra']=float('nan')
        return r
    controller,requests,args=setup(validate=bad)
    result=controller.advance(next_sample(args['measurement']))
    assert result['status']=='blocked',result
    assert result['target_right_q_deg'] is None and calls==2


def test_changed_dynamic_snapshot_is_bound_and_rejected_by_validator():
    def validate(request):
        r=receipt(request)
        if request['measurement']['scene']['plant_body_world']['/Plant/Stem'][0][3]>.01:
            r['checks']['right_scene']=False
        return r
    controller,requests,args=setup(validate=validate);m=next_sample(args['measurement'])
    m['scene']['plant_body_world']['/Plant/Stem'][0][3]=.02
    result=controller.advance(m)
    assert result['status']=='blocked' and len(requests)==2
    assert requests[0]['snapshot_sha256']!=requests[1]['snapshot_sha256']


def test_validator_mutation_exception_timeout_and_interrupt_fail_closed(monkeypatch):
    for fault in ('mutation','exception','timeout','interrupt'):
        controller,requests,args=setup(); now=[0.]
        monkeypatch.setattr(mw.time,'monotonic',lambda:now[0])
        def bad(request):
            r=receipt(request)
            if fault=='mutation':request['path_q_deg'][0][0]+=1
            elif fault=='exception':raise RuntimeError('unavailable')
            elif fault=='timeout':now[0]=mw.MAX_SECONDS
            else:raise KeyboardInterrupt('interrupted')
            return r
        controller._validate=bad
        if fault=='interrupt':
            with pytest.raises(KeyboardInterrupt):controller.advance(next_sample(args['measurement']))
        else:assert controller.advance(next_sample(args['measurement']))['status']=='blocked'
        assert controller.advance(next_sample(args['measurement']))['status']=='blocked'


@pytest.mark.parametrize('fault',['dt','dt_bool','rate','limits','direction','path','off_stroke',
                                 'outside','junction','nonmonotone'])
def test_bad_construction_has_no_validation_permission(fault):
    changes={};m=measurement();calls=[]
    if fault=='dt':changes['dt']=0
    elif fault=='dt_bool':changes['dt']=True
    elif fault=='rate':changes['max_speed_deg_s']=np.zeros(7)
    elif fault=='limits':changes['upper_deg']=np.full(7,-180.)
    elif fault=='direction':changes['stroke_direction']=[2,0,0]
    elif fault=='path':changes['stroke']=np.zeros((1,7))
    elif fault=='off_stroke':
        m['right_q_deg'][2]=1;m['right_wrist_world']=fk(m['right_q_deg']).tolist()
        m['scene']['robot_body_world']['/Robot/ee_right']=m['right_wrist_world']
    elif fault=='outside':m=measurement(5)
    elif fault=='junction':changes['approach']=np.array([q(-4),q(-2),q(-1)])
    else:changes['stroke']=np.array([q(0),q(3),q(2),q(4)])
    with pytest.raises((ValueError,TypeError)):
        setup(m=m,validate=lambda request:calls.append(request),**changes)
    assert calls==[]


def test_receipt_and_initial_target_are_detached_not_mutable_authority():
    controller,requests,args=setup()
    result=controller.initial_command();result['target_right_q_deg'][0]=999
    result['hold_targets']['finger_targets_m'][0]=999
    assert controller.initial_command()['target_right_q_deg'][0]==2.5
    json.dumps(controller.initial_command(),allow_nan=False)


def test_joint_posture_not_wrist_pose_alone_advances_waypoints():
    controller,requests,args=setup();m=next_sample(args['measurement'],q(2.))
    m['right_q_deg'][6]=1.
    result=controller.advance(m)
    assert result['status']=='moving' and result['waypoint_index']==1


def test_cumulative_query_budget_and_update_budget():
    controller,requests,args=setup();controller._queries=mw.MAX_QUERIES
    assert controller.advance(next_sample(args['measurement']))['status']=='blocked'
    controller,requests,args=setup();controller._updates=mw.MAX_UPDATES
    assert controller.advance(next_sample(args['measurement']))['status']=='blocked'


def test_release_callback_failure_never_constructs_a_ready_controller():
    with pytest.raises(ValueError):setup(validate=lambda _:True)


def test_nonmonotonic_connector_rejected_without_clearance_callback():
    def curved(joints):
        p=fk(joints);p[0,3]+=.0004*np.sin(np.pi*joints[6]/2)
        return p
    m=measurement();m['right_q_deg'][6]=2.;m['right_wrist_world']=curved(m['right_q_deg']).tolist()
    m['scene']['robot_body_world']['/Robot/ee_right']=m['right_wrist_world']
    calls=[]
    with pytest.raises(ValueError,match='forward'):
        setup(m=m,wrist_fk=curved,validate=lambda r:calls.append(r))
    assert calls==[]


def test_wall_time_between_fetches_does_not_consume_next_validation_budget(monkeypatch):
    now=[0.];monkeypatch.setattr(mw.time,'monotonic',lambda:now[0])
    controller,requests,args=setup();m=args['measurement']
    for _ in range(3):
        now[0]+=30.  # Native RTF / user pause is NOT a validation deadline.
        m=next_sample(m);result=controller.advance(m)
        assert result['status']=='moving',result
    assert result['maximum_episode_simulation_seconds']==4800*.01
    assert result['validation_calls']==4 and result['max_validation_seconds']==8.


def test_late_fk_and_constructor_validation_fail_closed(monkeypatch):
    now=[0.];monkeypatch.setattr(mw.time,'monotonic',lambda:now[0])
    def slow(q):
        now[0]+=8.
        return fk(q)
    with pytest.raises(ValueError,match='deadline'):setup(wrist_fk=slow)
    now[0]=0.
    def late(request):
        now[0]+=8.
        return receipt(request)
    with pytest.raises(ValueError,match='deadline'):setup(validate=late)


@pytest.mark.parametrize('check',mw.CHECKS)
def test_each_required_guard_must_be_strict_true(check):
    def bad(request):
        r=receipt(request);r['checks'][check]=1
        return r
    with pytest.raises(ValueError,match='receipt'):setup(validate=bad)


def test_exhausted_budget_does_not_call_native_validator():
    controller,requests,args=setup();controller._queries=mw.MAX_QUERIES
    assert controller.advance(next_sample(args['measurement']))['status']=='blocked'
    assert len(requests)==1


def test_reentrant_validator_cannot_authorize_outer_command():
    controller,requests,args=setup();m=next_sample(args['measurement'])
    def reentrant(request):
        with pytest.raises(RuntimeError):controller.advance(m)
        return receipt(request)
    controller._validate=reentrant
    result=controller.advance(m)
    assert result['status']=='blocked' and result['target_right_q_deg'] is None


def test_actual_native_packet_is_geometry_checked_and_hash_bound():
    from sim_physics.withdrawal_controller import native_command_packet
    x=float(np.degrees(float(np.float32(np.radians(2.5)))))
    controller,requests,args=setup(m=measurement(x),quantize_command=native_command_packet)
    m=args['measurement'];target=controller.initial_command()['target_right_q_deg']
    for _ in range(4):
        m=next_sample(m,target);result=controller.advance(m)
        assert result['status']=='moving',result
        target=result['target_right_q_deg']
        np.testing.assert_array_equal(np.degrees(np.radians(target).astype(np.float32).astype(float)),target)
        assert requests[-1]['path_q_deg'][-1]==target
        assert result['receipt']['request_sha256']==requests[-1]['request_sha256']


@pytest.mark.parametrize('direction',[-1,2])
def test_quantizer_cannot_reverse_or_overshoot_requested_interval(direction):
    controller,requests,args=setup(quantize_command=lambda q,d:q+direction*(d-q))
    result=controller.advance(next_sample(args['measurement']))
    assert result['status']=='blocked' and result['target_right_q_deg'] is None
    assert len(requests)==1
