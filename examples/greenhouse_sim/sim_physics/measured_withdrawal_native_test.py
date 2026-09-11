"""Mock native views/query interface; real source FK, no SimulationApp."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as S

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from greenhouse_sim.robot_kinematics import Rby1Kinematics
from sim_physics.held_plant_screen import HeldPlantScreen
from sim_physics.native_static_clearance import NativeStaticClearance
from sim_physics.measured_withdrawal import _hash,MARGINS
from sim_physics.measured_withdrawal_native import NativeWithdrawalAdapter,RIGHT,LEFT,FINGERS,_raw_joint_world


def packed(world,paths):
    return np.array([np.r_[world[p][:3,3],Rotation.from_matrix(world[p][:3,:3]).as_quat()] for p in paths])


def setup(monkeypatch, fault=None):
    import sim_physics.native_static_clearance as native_module
    state=S(step=(7,10),fault=fault,queries=[],world_checks=[],factories=[],closed=0,revision=0,native=None)
    kin=Rby1Kinematics();root='/World/Robot'
    joints={n:j for n,j in kin._by_name.items() if j.kind!='fixed'}
    names=list(joints)[::-1]  # Native order deliberately differs from URDF order.
    q={n:float((j.lower_rad+j.upper_rad)/2) if np.isfinite(j.lower_rad+j.upper_rad) else 0. for n,j in joints.items()}
    values=np.array([[q[n] for n in names]])
    degrees={n:float(np.degrees(q[n])) for n,j in joints.items() if j.kind!='prismatic'}
    slides={n:q[n] for n,j in joints.items() if j.kind=='prismatic'}
    base=np.eye(4);base[:3,3]=[.1,.2,.3]
    world={root+'/'+n:base@p for n,p in kin.all_link_transforms(degrees,prismatic_m=slides).items()}
    paths=list(world);native_paths=paths[::-1]
    state.world=world;state.values=values;state.targets=values.copy();state.names=names
    def dof_positions():return state.values.copy()
    robot=S(count=1,shared_metatype=S(fixed_base=True,dof_names=names),link_paths=[paths],
        get_dof_positions=dof_positions,get_dof_position_targets=lambda:state.targets.copy())
    bodies=S(prim_paths=native_paths,get_transforms=lambda:packed(state.world,native_paths))
    wrist=S(count=1,prim_paths=[root+'/ee_right'],get_transforms=lambda:packed(state.world,[root+'/ee_right']))
    shapes=[]
    for side in ('left','right'):
        link='link_'+side+'_arm_0';body=root+'/'+link
        shapes.append((body+'/restored_collisions/capsule_0',body,link,'capsule',
                       (np.zeros(3),np.array([0,0,.001]),.001)))
    body=root+'/ee_right'
    shapes.append((body+'/attachments/RightWristCamera/BodyCollision',body,'ee_right','box',
                   (np.zeros(3),np.eye(3),np.full(3,.01))))
    class Self:
        def __init__(self):self.shapes=shapes;self.unsupported=[]
        def check(self,world,*,margin):
            state.world_checks.append(('self',deepcopy(world),margin))
            if state.fault=='step':state.step=(7,11)
            if state.fault=='epoch':state.revision+=1
            if state.fault=='interrupt':raise KeyboardInterrupt('interrupt')
            return dict(passed=state.fault!='self',all_shape_bounds_screened=state.fault!='unsupported',
                unsupported_shapes=['missing'] if state.fault=='unsupported' else [],
                checked_pairs=3,required_margin_m=margin,
                minimum_clearance_m=.002 if state.fault=='minimum' else .04)
    plants={f'/World/Plant/Segment_{i:03d}':np.eye(4) for i in range(3)}
    for i,p in enumerate(plants):plants[p][:3,3]=[2.,2.,i*.03]
    plant_paths=list(plants);state.plants=plants
    plant=S(prim_paths=plant_paths[::-1],get_transforms=lambda:packed(state.plants,plant.prim_paths))
    class Held(HeldPlantScreen):
        def __init__(self):
            self.arm='right';self.workspace=(np.full(3,-100.),np.full(3,100.))
            self.local=[(p+'/StemCollider',i,'capsule',(np.zeros(3),np.array([0,0,.01]),.002))
                        for i,p in enumerate(plant_paths)]
            self.static=[('/Scene/Obstacle','box',None,np.full(3,10.),np.full(3,11.))]
            self.static_indices={};self.shapes=[s for s in shapes if s[2] in ('link_right_arm_0','ee_right')]
            self.native_static_query=None;self.last_failure=None
        def snapshot(self,frames):self.frames=np.array(frames,copy=True)
        def check(self,world,*,stroke=False,grasp=False,margin):
            state.world_checks.append((self.arm,deepcopy(world),(stroke,grasp,margin,self.frames.copy())))
            if state.fault==self.arm:
                self.last_failure={'reason':self.arm+'_blocked'};return False
            return True
    fixture=S(root=root,stage=object(),kin=kin,robot=robot,robot_bodies=bodies,right_palm=wrist,
        names=names,body_paths=paths,collider_paths=[s[0] for s in shapes],self_screen=Self(),
        held_plant_screen=Held(),rig=S(body_paths=plant_paths,cut_index=1,cut=True),
        grasp_path=plant_paths[1],event_monitor=S(error=None,native_full_contact_reporting=True),
        targets=state.targets,pose={'DO_NOT_USE':999},planning_slides={'DO_NOT_USE':999},
        base=np.full((4,4),np.nan))  # Must use measured native base, not this cached field.
    hold=dict(left_goal_world=world[root+'/ee_left'].tolist(),finger_targets_m=[q[n] for n in FINGERS])
    guard=dict(sample_id=[7,10],passed=True,errors=[])
    grasp=dict(sample_id=[7,10],result=dict(step_id=10,bilateral=True,adapter_valid=True))
    def factory(stage,records,*,wall_limit_s,max_queries,lazy_coverage):
        state.factories.append(dict(seconds=wall_limit_s,records=deepcopy(records),
                                    max_queries=max_queries,lazy_coverage=lazy_coverage))
        assert stage is fixture.stage and 0<wall_limit_s<=8. and lazy_coverage is True
        if state.fault=='factory':raise RuntimeError('factory unavailable')
        def guard_epoch():
            if state.revision:raise RuntimeError('epoch changed')
        def query(path,c,a,h):
            state.queries.append((path,np.array(c),np.array(a),np.array(h)))
            coverage=np.allclose(c,[10.5]*3)
            if state.fault=='missing_coverage':return False
            if state.fault=='final_coverage' and coverage and len(state.queries)>1:return False
            if state.fault=='witness_hit' and not coverage:return True
            return bool(coverage)
        def close():
            state.closed+=1
            if state.fault=='close':raise RuntimeError('close failed')
            if state.fault=='close_step':state.step=(7,11)
        def epoch():return dict(revision=state.revision,subscriptions_closed=state.closed>0,
            cleanup_errors=[],invalidation_reasons=['changed'] if state.revision else [])
        native=NativeStaticClearance(query,records,guard=guard_epoch,close_guard=close,epoch_report=epoch,
            wall_limit_s=wall_limit_s,max_queries=max_queries,lazy_coverage=lazy_coverage)
        if state.fault=='report':native.report=lambda:(_ for _ in ()).throw(RuntimeError('report failed'))
        state.native=native;return native
    monkeypatch.setattr(native_module,'current_scene_query',factory)
    adapter=NativeWithdrawalAdapter(fixture,plant,lambda:state.step)
    args=dict(sample_id=(7,10),hold_targets=hold,guard_evidence=guard,grasp_evidence=grasp)
    return adapter,state,fixture,plant,args


def request(adapter,m,*,kind='next_segment',phase='extract',delta=.2):
    q=np.array(m['right_q_deg']);path=[q] if kind=='endpoint' else [q,q+np.array([delta,0,0,0,0,0,0])]
    phases=[] if kind=='endpoint' else [phase]
    r=dict(kind=kind,sample_id=m['sample_id'],snapshot_sha256=_hash(m),measurement=deepcopy(m),
        path_q_deg=np.array(path).tolist(),path_wrist_world=[adapter.wrist_fk(x).tolist() for x in path],
        phases=phases,seam_allowance=[p=='extract' for p in phases],margins_m=dict(MARGINS),
        maximum_joint_sample_step_deg=1.,max_queries=20000,max_seconds=8.,
        left_hold_fixed=True,full_forward_cutstroke_verified=False)
    r['request_sha256']=_hash(r)
    return r


def resign(r):
    r.pop('request_sha256',None);r['request_sha256']=_hash(r);return r


def test_live_named_native_snapshot_uses_all_joint_finger_torso_data(monkeypatch):
    adapter,state,f,plant,args=setup(monkeypatch)
    before=f.targets.copy();m=adapter.snapshot(**args)
    assert set(m['scene']['native_dof_positions'])==set(state.names)
    assert len(m['scene']['finger_slides_m'])==4
    assert len([n for n in m['scene']['revolute_degrees'] if n.startswith('torso_')])==6
    assert m['scene']['native_base_world']==state.world[f.root+'/base'].tolist()
    assert set(m['scene']['plant_body_world'])==set(f.rig.body_paths)
    np.testing.assert_array_equal(before,f.targets)
    assert not state.factories and not state.queries


def test_actual_start_future_fk_and_real_coverage_witness(monkeypatch):
    adapter,state,f,plant,args=setup(monkeypatch)
    state.world[f.root+'/ee_right'][0,3]+=2e-5 # Genuine small native/FK discrepancy.
    m=adapter.snapshot(**args);r=request(adapter,m);before=deepcopy(r)
    result=adapter.validate(r)
    assert result['passed'],result
    assert r==before and state.closed==1
    assert result['checked_samples']==2 and result['checked_segments']==1
    assert result['native_static']['query_count']==3
    assert result['cost']['initialization_queries']==0
    assert result['cost']['path_refinement_queries']==0
    assert result['cost']['witness_queries']==2
    assert result['cost']['final_actor_queries']==1
    assert result['cost']['total_queries']==3
    assert result['native_static']['used_static_colliders']==['/Scene/Obstacle']
    assert result['native_static']['final_coverage_checked']==['/Scene/Obstacle']
    assert result['coverage_witness']['actual_native_start']
    assert result['coverage_witness']['robot_collider'] in f.collider_paths
    checks=[x for x in state.world_checks if x[0]=='self']
    np.testing.assert_allclose(checks[0][1]['ee_right'],state.world[f.root+'/ee_right'])
    np.testing.assert_allclose(checks[-1][1]['ee_right'],adapter.wrist_fk(r['path_q_deg'][-1]))
    np.testing.assert_allclose(checks[-1][1]['ee_left'],state.world[f.root+'/ee_left'])
    assert not np.allclose(checks[0][1]['ee_right'],checks[-1][1]['ee_right'])
    assert f.held_plant_screen.native_static_query is None and not hasattr(f.held_plant_screen,'frames')
    assert all(result['checks'].values()) and not result['whole_path_certified']


@pytest.mark.parametrize('fault',['guard_false','guard_truthy','guard_stale','grasp_false','grasp_step','grasp_adapter',
                                'scene_step','missing_joint','duplicate_joint','plant_inventory','robot_inventory',
                                'collider_inventory','pose_mismatch','targets','full_stream','not_cut'])
def test_invalid_snapshot_never_creates_query(monkeypatch,fault):
    adapter,state,f,plant,args=setup(monkeypatch)
    if fault=='guard_false':args['guard_evidence']['passed']=False
    elif fault=='guard_truthy':args['guard_evidence']['passed']=1
    elif fault=='guard_stale':args['guard_evidence']['sample_id'][1]-=1
    elif fault=='grasp_false':args['grasp_evidence']['result']['bilateral']=False
    elif fault=='grasp_step':args['grasp_evidence']['result']['step_id']=9
    elif fault=='grasp_adapter':args['grasp_evidence']['result']['adapter_valid']=False
    elif fault=='scene_step':state.step=(7,11)
    elif fault=='missing_joint':f.robot.shared_metatype.dof_names=state.names[1:]
    elif fault=='duplicate_joint':f.robot.shared_metatype.dof_names=state.names[:-1]+[state.names[0]]
    elif fault=='plant_inventory':plant.prim_paths=plant.prim_paths[:-1]
    elif fault=='robot_inventory':f.body_paths=f.body_paths[:-1]
    elif fault=='collider_inventory':f.collider_paths=f.collider_paths[:-1]
    elif fault=='pose_mismatch':state.world[f.root+'/link_torso_2'][0,3]+=.001
    elif fault=='targets':args['hold_targets']['finger_targets_m'][0]+=.001
    elif fault=='full_stream':f.event_monitor.native_full_contact_reporting=False
    else:f.rig.cut=False
    with pytest.raises((ValueError,RuntimeError)):adapter.snapshot(**args)
    assert not state.factories


@pytest.mark.parametrize('fault',['self','unsupported','minimum','left','right','epoch','step','close',
    'close_step','report','missing_coverage','final_coverage','witness_hit','factory'])
def test_native_screen_failures_never_pass_and_always_cleanup(monkeypatch,fault):
    adapter,state,f,plant,args=setup(monkeypatch)
    m=adapter.snapshot(**args);r=request(adapter,m);state.fault=fault
    result=adapter.validate(r)
    assert not result['passed'] and result['errors'],result
    assert not any(result['checks'].values())
    assert state.closed==(0 if fault=='factory' else 1)
    assert not adapter.validate(r)['passed']


@pytest.mark.parametrize('fault',['hash','snapshot','sample','margins','density','start','fk','budget_bool',
                                'budget_large','time','seam_truthy','phase','hold'])
def test_request_tampering_and_malformed_paths_fail_before_query(monkeypatch,fault):
    adapter,state,f,plant,args=setup(monkeypatch);m=adapter.snapshot(**args);r=request(adapter,m)
    if fault=='hash':r['request_sha256']='bad'
    elif fault=='snapshot':r['snapshot_sha256']='bad'
    elif fault=='sample':r['sample_id']=[7,9]
    elif fault=='margins':r['margins_m']['scene_m']=.0009
    elif fault=='density':r['path_q_deg'][1][0]+=2
    elif fault=='start':r['path_q_deg'][0][0]+=.001
    elif fault=='fk':r['path_wrist_world'][1][0][3]+=.001
    elif fault=='budget_bool':r['max_queries']=True
    elif fault=='budget_large':r['max_queries']=20001
    elif fault=='time':r['max_seconds']=8.1
    elif fault=='seam_truthy':r['seam_allowance']=[1]
    elif fault=='phase':r['phases']=['teleport']
    else:r['left_hold_fixed']=False
    if fault!='hash':resign(r)
    result=adapter.validate(r)
    assert not result['passed'] and not state.factories


@pytest.mark.parametrize('change',['plant','robot','joint','target','clock'])
def test_same_label_stale_native_snapshot_not_accepted(monkeypatch,change):
    adapter,state,f,plant,args=setup(monkeypatch);m=adapter.snapshot(**args);r=request(adapter,m)
    if change=='plant':state.plants[f.rig.body_paths[-1]][0,3]+=.001
    elif change=='robot':state.world[f.root+'/ee_right'][0,3]+=.00002
    elif change=='joint':state.values[0,state.names.index('right_arm_0')]+=.001
    elif change=='target':state.targets[0,state.names.index('torso_0')]+=.001
    else:state.step=(7,11)
    result=adapter.validate(r)
    assert not result['passed'] and not state.factories


def test_query_budget_preflight_and_real_query_count(monkeypatch):
    adapter,state,f,plant,args=setup(monkeypatch);m=adapter.snapshot(**args);r=request(adapter,m)
    r['max_queries']=2;resign(r)
    assert not adapter.validate(r)['passed'] and not state.factories
    adapter,state,f,plant,args=setup(monkeypatch);m=adapter.snapshot(**args);r=request(adapter,m)
    r['max_queries']=3;resign(r);result=adapter.validate(r)
    assert result['passed'] and result['native_static']['max_queries']==3
    assert result['native_static']['query_count']==len(state.queries)==3


def test_return_and_endpoint_have_no_seam_allowance(monkeypatch):
    for kind in ('next_segment','endpoint'):
        adapter,state,f,plant,args=setup(monkeypatch);m=adapter.snapshot(**args)
        result=adapter.validate(request(adapter,m,kind=kind,phase='return'))
        assert result['passed'],result
        right=[c for c in state.world_checks if c[0]=='right']
        assert right and all(c[2][0] is False for c in right)
        left=[c for c in state.world_checks if c[0]=='left']
        assert all(c[2][1] is True and c[2][2]==.001 for c in left)


def test_interrupt_cleanup_and_no_commands(monkeypatch):
    adapter,state,f,plant,args=setup(monkeypatch);m=adapter.snapshot(**args);r=request(adapter,m)
    state.fault='interrupt'
    with pytest.raises(KeyboardInterrupt):adapter.validate(r)
    assert state.closed==1 and not adapter.last_receipt['passed']
    assert not hasattr(f.robot,'set_dof_position_targets')


def test_changed_station_target_is_not_a_grip_preserving_snapshot(monkeypatch):
    adapter,state,f,plant,args=setup(monkeypatch);adapter.snapshot(**args)
    state.step=(7,11);args['sample_id']=state.step
    args['guard_evidence']['sample_id']=[7,11];args['grasp_evidence']['sample_id']=[7,11]
    args['grasp_evidence']['result']['step_id']=11
    state.targets[0,state.names.index('head_0')]+=.01
    with pytest.raises(ValueError,match='Hold/station'):adapter.snapshot(**args)


@pytest.mark.parametrize('bad',[None,True,[],{'epoch':[]},
    {'epoch':{},'final_coverage_checked':['/Fake']},
    {'epoch':{},'final_coverage_checked':['/Fake'],'used_static_colliders':42}])
def test_malformed_native_reports_fail_closed_after_close(monkeypatch,bad):
    adapter,state,f,plant,args=setup(monkeypatch);m=adapter.snapshot(**args);r=request(adapter,m)
    import sim_physics.native_static_clearance as native_module
    original=native_module.current_scene_query
    def factory(*args,**kwargs):
        result=original(*args,**kwargs);result.report=lambda:deepcopy(bad);return result
    monkeypatch.setattr(native_module,'current_scene_query',factory)
    result=adapter.validate(r)
    assert not result['passed'] and result['errors'] and state.closed==1


def test_adapter_deadline_closes_query_and_no_pass(monkeypatch):
    import sim_physics.measured_withdrawal_native as module
    adapter,state,f,plant,args=setup(monkeypatch);m=adapter.snapshot(**args);r=request(adapter,m)
    now=[0.];monkeypatch.setattr(module.time,'monotonic',lambda:now[0])
    original=f.self_screen.check
    def slow(*args,**kwargs):
        now[0]+=9.;return original(*args,**kwargs)
    f.self_screen.check=slow
    result=adapter.validate(r)
    assert not result['passed'] and state.closed==1
    assert any('deadline' in e.lower() for e in result['errors'])


def test_reentrant_validation_cannot_leave_outer_receipt_passed(monkeypatch):
    adapter,state,f,plant,args=setup(monkeypatch);m=adapter.snapshot(**args);r=request(adapter,m)
    original=f.self_screen.check
    def reentrant(*args,**kwargs):
        assert not adapter.validate(r)['passed']
        return original(*args,**kwargs)
    f.self_screen.check=reentrant
    result=adapter.validate(r)
    assert not result['passed'] and state.closed==1


def test_real_interarm_ten_mm_gate_blocks_even_if_self_three_mm_mock_passes(monkeypatch):
    adapter,state,f,plant,args=setup(monkeypatch);m=adapter.snapshot(**args);r=request(adapter,m)
    # Use native arm transforms to place the cached right capsule exactly on
    # the left capsule; do not replace the actual 10 mm distance implementation.
    left,right=f.self_screen.shapes[:2]
    transform=np.linalg.inv(state.world[right[1]])@state.world[left[1]]
    a,b,radius=left[4]
    data=(transform[:3,:3]@a+transform[:3,3],transform[:3,:3]@b+transform[:3,3],radius)
    f.self_screen.shapes[1]=(*right[:4],data)
    f.held_plant_screen.shapes[0]=f.self_screen.shapes[1]
    result=adapter.validate(r)
    assert not result['passed'] and any('10 mm' in e for e in result['errors'])


def test_geometry_request_uses_measured_nondefault_torso_and_four_finger_slides(monkeypatch):
    adapter,state,f,plant,args=setup(monkeypatch);m=adapter.snapshot(**args)
    expected=f.kin.forward('right',m['right_q_deg'],np.array(m['scene']['native_base_world']),
        torso_degrees=[m['scene']['revolute_degrees']['torso_'+str(i)] for i in range(6)])
    np.testing.assert_allclose(adapter.wrist_fk(m['right_q_deg']),expected,atol=1e-12)
    assert all(name in m['scene']['finger_slides_m'] for name in
               ('gripper_finger_l1','gripper_finger_l2','gripper_finger_r1','gripper_finger_r2'))


@pytest.mark.parametrize('bad',[float('nan'),object()])
def test_unserializable_native_report_never_escapes_as_passed(monkeypatch,bad):
    adapter,state,f,plant,args=setup(monkeypatch);m=adapter.snapshot(**args);r=request(adapter,m)
    import sim_physics.native_static_clearance as native_module
    original=native_module.current_scene_query
    def factory(*args,**kwargs):
        n=original(*args,**kwargs);report=n.report
        n.report=lambda:{**report(),'extra':bad}
        return n
    monkeypatch.setattr(native_module,'current_scene_query',factory)
    result=adapter.validate(r)
    assert result['passed'] is False and state.closed==1
    with pytest.raises(RuntimeError):adapter.wrist_fk(m['right_q_deg'])


def test_frozen_pure_helper_consumes_actual_adapter_receipts_without_integration_edits(monkeypatch):
    from sim_physics.measured_withdrawal import MeasuredWithdrawal
    adapter,state,f,plant,args=setup(monkeypatch);m=adapter.snapshot(**args)
    q=np.array(m['right_q_deg']);axis=np.array([1,0,0,0,0,0,0])
    stroke=np.array([q-.2*axis,q+.2*axis]);approach=np.array([q-.4*axis,q-.2*axis])
    d=adapter.wrist_fk(stroke[1])[:3,3]-adapter.wrist_fk(stroke[0])[:3,3];d/=np.linalg.norm(d)
    low,high=f.kin.arm_limits_degrees('right')
    helper=MeasuredWithdrawal(stroke,approach,measurement=m,lower_deg=low,upper_deg=high,
        max_speed_deg_s=np.ones(7),dt=1/240,wrist_fk=adapter.wrist_fk,edge_in_wrist=np.eye(4),
        stroke_direction=d,validate=adapter.validate)
    initial=helper.initial_command()
    assert initial['status']=='ready' and initial['receipt']['passed']
    state.step=(7,11);args['sample_id']=state.step
    args['guard_evidence']['sample_id']=[7,11];args['grasp_evidence']['sample_id']=[7,11]
    args['grasp_evidence']['result']['step_id']=11
    result=helper.advance(adapter.snapshot(**args))
    assert result['status']=='moving',result
    assert result['native_query_count']==6
    assert result['receipt']['native_static']['max_queries']==20000-3
    assert not result['cut_authorized'] and result['hold_targets']==m['hold_targets']


def set_measured_joint(state,fixture,name,value):
    """Independent analytic mock native subtree motion; never a runtime setter."""
    j=fixture.kin._by_name[name];root=fixture.root
    local=np.eye(4)
    if j.kind=='prismatic':local[:3,3]=j.axis*value
    else:local[:3,:3]=Rotation.from_rotvec(j.axis*value).as_matrix()
    actual=state.world[root+'/'+j.parent]@j.origin@local
    delta=actual@np.linalg.inv(state.world[root+'/'+j.child])
    for path in list(state.world):
        link=path[len(root)+1:];node=link
        while node!='base':
            if node==j.child:
                state.world[path]=delta@state.world[path];break
            node=fixture.kin._by_child[node].parent
    state.values[0,state.names.index(name)]=value


def test_raw_full_fk_matches_source_fk_without_default_joint_substitution(monkeypatch):
    adapter,state,f,plant,args=setup(monkeypatch)
    values=dict(zip(state.names,state.values[0],strict=True))
    base=state.world[f.root+'/base']
    frames=_raw_joint_world(f.kin,values,base)
    assert set(frames)=={p[len(f.root)+1:] for p in state.world}
    for link,pose in frames.items():
        np.testing.assert_allclose(pose,state.world[f.root+'/'+link],rtol=0,atol=2e-14)


@pytest.mark.parametrize('name,value,units',[
    ('gripper_finger_r1',2.0772280606706772e-7,'m'), # Native53 release, exact observed value.
    ('gripper_finger_r2',.05000150203704834,'m'),
    ('left_arm_1',-.01745329611003399,'rad'),
    ('torso_0',np.pi+.01,'rad'),
])
def test_measured_excursion_preserved_and_screened_without_command_change(monkeypatch,name,value,units):
    adapter,state,f,plant,args=setup(monkeypatch)
    targets=state.targets.copy();set_measured_joint(state,f,name,value)
    # Preserve the left command pose while observing the changed torso frame.
    if name=='torso_0':args['hold_targets']['left_goal_world']=state.world[f.root+'/ee_left'].tolist()
    m=adapter.snapshot(**args);scene=m['scene'];j=f.kin._by_name[name]
    assert scene['native_dof_positions'][name]==value
    assert scene['measured_joint_positions_clamped'] is False
    assert scene['command_targets_within_source_limits'] is True
    assert scene['measured_joint_limit_excursions']==[dict(joint=name,observed=value,
        lower=j.lower_rad,upper=j.upper_rad,units=units,
        signed_excess=value-(j.lower_rad if value<j.lower_rad else j.upper_rad))]
    before=deepcopy(scene['robot_body_world'])
    result=adapter.validate(request(adapter,m))
    assert result['passed'],result
    actual=[c for c in state.world_checks if c[0]=='self'][0][1]
    for link,pose in actual.items():np.testing.assert_allclose(pose,before[f.root+'/'+link],rtol=0,atol=1e-14)
    future=adapter._future(m['right_q_deg'],actual)
    if name.startswith('gripper_finger_r'):
        np.testing.assert_allclose(future[j.child],state.world[f.root+'/'+j.child],rtol=0,atol=1e-13)
        assert scene['finger_slides_m'][name]==value
    np.testing.assert_array_equal(state.values[0,state.names.index(name)],value)
    np.testing.assert_array_equal(state.targets,targets)
    assert 'command targets' in result['joint_limits_scope']


@pytest.mark.parametrize('name',['gripper_finger_r1','gripper_finger_l1','head_0','torso_0','right_arm_0'])
def test_out_of_source_command_target_rejects_even_with_matching_controller_buffer(monkeypatch,name):
    adapter,state,f,plant,args=setup(monkeypatch)
    state.targets[0,state.names.index(name)]=f.kin._by_name[name].upper_rad+1e-8
    with pytest.raises(ValueError,match='command target outside source limits'):adapter.snapshot(**args)
    assert not state.factories and not state.queries


def test_legal_finger_stop_target_zero_allowed_but_positive_measurement_unclamped(monkeypatch):
    adapter,state,f,plant,args=setup(monkeypatch)
    state.targets[0,state.names.index('gripper_finger_r1')]=0.
    set_measured_joint(state,f,'gripper_finger_r1',2.0772280606706772e-7)
    m=adapter.snapshot(**args)
    assert m['scene']['native_dof_targets']['gripper_finger_r1']==0.
    assert adapter.validate(request(adapter,m))['passed']


@pytest.mark.parametrize('boundary',['low','high','above'])
def test_raw_measured_fk_never_authorizes_out_of_limit_right_commands(monkeypatch,boundary):
    adapter,state,f,plant,args=setup(monkeypatch);m=adapter.snapshot(**args)
    q=np.array(m['right_q_deg']);low,high=f.kin.arm_limits_degrees('right')
    q[0]=low[0] if boundary=='low' else high[0]+(1e-8 if boundary=='above' else 0.)
    with pytest.raises(ValueError,match='strict source limits'):adapter.wrist_fk(q)
    r=request(adapter,m);r['path_q_deg'][-1]=q.tolist();resign(r)
    assert not adapter.validate(r)['passed'] and not state.factories


def test_right_arm_measurement_outside_stop_remains_observable_not_commandable(monkeypatch):
    adapter,state,f,plant,args=setup(monkeypatch)
    set_measured_joint(state,f,'right_arm_0',f.kin._by_name['right_arm_0'].upper_rad+1e-7)
    m=adapter.snapshot(**args)
    assert m['scene']['measured_joint_limit_excursions'][0]['joint']=='right_arm_0'
    with pytest.raises(ValueError,match='strict source limits'):adapter.wrist_fk(m['right_q_deg'])
    assert not state.factories


@pytest.mark.parametrize('fault',['missing','extra','nan','bool','axis_zero','axis_nonunit','origin','kind','tree'])
def test_raw_measured_fk_invalid_data_fail_closed(monkeypatch,fault):
    adapter,state,f,plant,args=setup(monkeypatch)
    values=dict(zip(state.names,state.values[0],strict=True));name='right_arm_0'
    if fault=='missing':del values[name]
    elif fault=='extra':values['unknown']=0.
    elif fault=='nan':values[name]=float('nan')
    elif fault=='bool':values[name]=True
    else:
        j=f.kin._by_name[name]
        if fault=='axis_zero':j=replace(j,axis=np.zeros(3))
        elif fault=='axis_nonunit':j=replace(j,axis=j.axis*2)
        elif fault=='origin':j=replace(j,origin=np.diag([2.,1.,1.,1.]))
        elif fault=='kind':j=replace(j,kind='floating')
        else:j=replace(j,parent=j.child)
        f.kin._by_name[name]=j;f.kin._by_child[j.child]=j
    with pytest.raises(ValueError):_raw_joint_world(f.kin,values,state.world[f.root+'/base'])


def test_lazy_eighty_actor_inventory_retained_but_only_actual_witness_queried(monkeypatch):
    adapter,state,f,plant,args=setup(monkeypatch)
    f.held_plant_screen.static.extend((f'/Scene/Unused_{i:02d}','box',None,np.full(3,20.+i),np.full(3,21.+i))
                                    for i in range(79))
    records=deepcopy(f.held_plant_screen.static);m=adapter.snapshot(**args);r=request(adapter,m)
    r['max_queries']=3;resign(r);result=adapter.validate(r)
    assert result['passed'],result
    assert len(state.factories[0]['records'])==len(f.held_plant_screen.static)==80
    assert [x[0] for x in state.factories[0]['records']]==[x[0] for x in records]
    assert state.factories[0]['lazy_coverage'] is True and state.factories[0]['max_queries']==3
    assert result['cost']['initialization_queries']==0
    assert len(state.queries)==result['native_static']['query_count']==3
    assert {q[0] for q in state.queries}=={'/Scene/Obstacle'}
    assert len(result['native_static']['unchecked_static_colliders'])==79
    assert result['native_static']['final_coverage_checked']==['/Scene/Obstacle']
    assert result['native_static']['coverage_mode']=='lazy_before_exact_actor_refinement'


@pytest.mark.parametrize('limit,passed',[(4,False),(5,True)])
def test_lazy_path_controls_and_final_control_all_count_in_remaining_budget(monkeypatch,limit,passed):
    adapter,state,f,plant,args=setup(monkeypatch)
    original=type(f.held_plant_screen).check
    def needs_refinement(screen,world,**kwargs):
        if not original(screen,world,**kwargs):return False
        if screen.arm=='left':return True
        path,body,link,kind,(c,a,h)=next(s for s in screen.shapes if s[3]=='box')
        pose=world[link]
        return screen.native_static_query.clear_box_checked('/Scene/Obstacle',
            pose[:3,:3]@c+pose[:3,3],pose[:3,:3]@a,h,.001)
    monkeypatch.setattr(type(f.held_plant_screen),'check',needs_refinement)
    m=adapter.snapshot(**args);r=request(adapter,m);r['max_queries']=limit;resign(r)
    result=adapter.validate(r)
    assert result['passed'] is passed and state.closed==1
    assert len(state.queries)==result['native_static']['query_count']==limit
    if passed:
        assert result['cost']['path_refinement_queries']==4 # initial positive + three real robot bounds
        assert result['cost']['witness_queries']==0 and result['cost']['final_actor_queries']==1


@pytest.mark.parametrize('fault',['eager','budget'])
def test_factory_cannot_silently_ignore_lazy_mode_or_remaining_budget(monkeypatch,fault):
    adapter,state,f,plant,args=setup(monkeypatch);m=adapter.snapshot(**args)
    import sim_physics.native_static_clearance as native_module
    original=native_module.current_scene_query
    def wrong(*args,**kwargs):
        native=original(*args,**kwargs)
        if fault=='eager':native.lazy_coverage=False
        else:native.max_queries+=1
        return native
    monkeypatch.setattr(native_module,'current_scene_query',wrong)
    result=adapter.validate(request(adapter,m))
    assert not result['passed'] and state.closed==1 and not state.queries
