from copy import deepcopy
from types import SimpleNamespace as S
import numpy as np
import pytest
from . import postcut_egress as e


def inputs():
    return (dict(cut=True,native_guards_passed=True,knife=dict(tool_contact_upper_bound_n=.002)),
        dict(step=50,complete=True,cut_face_and_support_verified=True,cut_authorized=False,
            commanded_motion_used_as_completion=False,full_knife_load_n=.002))


@pytest.mark.parametrize('key,value',[('step',49),('complete',False),('cut_face_and_support_verified',False),
    ('cut_authorized',True),('commanded_motion_used_as_completion',True),('full_knife_load_n',.01),
    ('full_knife_load_n',float('nan')),('full_knife_load_n',True)])
def test_no_stale_or_loaded_egress(key,value):
    record,receipt=inputs();receipt[key]=value
    with pytest.raises(ValueError):e.prerequisites(record,receipt,50)


@pytest.mark.parametrize('key',['cut','native_guards_passed'])
def test_original_native_guards_required(key):
    record,receipt=inputs();record[key]=False
    with pytest.raises(ValueError):e.prerequisites(record,receipt,50)


def test_offsets_never_downward_or_larger_than_thirty_mm():
    from scipy.spatial.transform import Rotation
    for angles in ([0,0,0],[.7,-1.2,2.1]):
        f=np.eye(4);f[:3,:3]=Rotation.from_rotvec(angles).as_matrix()
        ds=e.offsets(f)
        assert 1<=len(ds)<=45
        assert all(0.0019<=np.linalg.norm(d)<=.0300000001 and d[2]>=0 for d in ds)


def test_dense_joint_samples_include_both_endpoints():
    a=np.arange(7,dtype=float);b=a+np.arange(7)/2
    q=e.joint_samples(a,b)
    assert np.array_equal(q[0],a) and np.array_equal(q[-1],b)
    assert np.max(abs(np.diff(q,axis=0)))<=.25


def setup(monkeypatch,*,blocked=False,epoch_fault=False,drift=False):
    import sim_physics.native_static_clearance as n
    names=e.RIGHT+e.LEFT;q=np.zeros((1,len(names)));world={'base':np.eye(4),'ee_right':np.eye(4)}
    state=S(checks=[],closed=False,validated=False,solves=0)
    def worlds(kin,values,base):
        result=deepcopy(world);result['ee_right'][0,3]=np.degrees(values[e.RIGHT[0]])*.001
        return result
    def solve(arm,desired,seed,base,**kw):
        state.solves+=1;end=np.zeros(7);end[0]=desired[0,3]/.001
        return S(succeeded=True,joint_degrees=end)
    f=S(names=names,base=np.eye(4),plan={},rig=S(cut=True),stage=object(),native_capsule_sphere_cover=True,
        kin=S(arm_limits_degrees=lambda arm:(np.full(7,-180),np.full(7,180)),
            forward=lambda *a:np.eye(4),solve_pose=solve,
            inter_arm_clearance=lambda *a:S(clearance_m=.02)),
        robot=S(get_dof_positions=lambda:q.copy(),get_dof_position_targets=lambda:q.copy()),
        self_screen=S(check=lambda w:dict(passed=True)))
    class Screen:
        static=[];last_failure={'obstacle':'held_branch'}
        def check(self,w,**kw):
            state.checks.append(kw)
            return not blocked
    right,left=Screen(),Screen()
    class Native:
        def validate(self):state.validated=True
        def close(self):state.closed=True
        def report(self):return dict(final_validation_passed=state.validated,closed=state.closed,errors=[],
            epoch=dict(revision=int(epoch_fault),subscriptions_closed=True,cleanup_errors=[],invalidation_reasons=[]))
    monkeypatch.setattr(n,'current_scene_query',lambda *a,**kw:Native())
    monkeypatch.setattr(e,'_robot_world',lambda f:deepcopy(world))
    monkeypatch.setattr(e,'_raw_joint_world',worlds)
    monkeypatch.setattr(e,'_screens',lambda *a:(right,left))
    monkeypatch.setattr(e,'offsets',lambda w:[np.array([.004,0,0])])
    if drift:
        def changed():
            result=q.copy()
            if state.closed:result[0,1]=.001
            return result
        f.robot.get_dof_positions=changed
    return f,state


def test_every_path_sample_has_normal_clearance_and_no_command(monkeypatch):
    f,s=setup(monkeypatch);record,r=inputs()
    path,report=e.plan(f,np.eye(4)[None],step=50,record=record,retraction=r)
    assert report['passed'] and not report['motion_authorized'] and not report['whole_path_certified']
    assert len(path)==17 and s.closed and s.validated
    assert len(s.checks)==2*(1+len(path))
    assert all(c['stroke'] is False for c in s.checks)
    assert np.array_equal(f.robot.get_dof_positions(),np.zeros((1,14))) and not f.plan


def test_initial_overlap_is_not_treated_as_escape_permission(monkeypatch):
    f,s=setup(monkeypatch,blocked=True);record,r=inputs()
    with pytest.raises(RuntimeError,match='start is not clear'):e.plan(f,np.eye(4)[None],step=50,record=record,retraction=r)
    assert s.closed and not s.validated and s.solves==0 and not f.postcut_egress_evidence['passed']


@pytest.mark.parametrize('failure',['epoch_fault','drift'])
def test_current_snapshot_and_closed_epoch_required(monkeypatch,failure):
    f,s=setup(monkeypatch,**{failure:True});record,r=inputs()
    with pytest.raises(RuntimeError):e.plan(f,np.eye(4)[None],step=50,record=record,retraction=r)
    assert s.closed and not f.postcut_egress_evidence['passed']
