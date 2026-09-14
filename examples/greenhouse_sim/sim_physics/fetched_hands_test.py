from types import SimpleNamespace as S
import numpy as np
import pytest
from .fetched_hands import FetchedHands
from .runtime import pose_matrices


def fixture():
    rng=np.random.default_rng(16)
    poses=rng.normal(size=(4,7));poses[:,3:]*=3.
    calls=[]
    def view(name,rows):
        def read():calls.append(name);return poses[rows]
        return S(get_transforms=read)
    robot=S(fingers=view('fingers',slice(0,2)),palm=view('left',slice(2,3)),
        right_palm=view('right',slice(3,4)),order=np.array([1,0]))
    return robot,poses,calls


def test_batched_snapshot_is_exactly_equal_to_independent_native_conversions():
    robot,poses,calls=fixture();out=FetchedHands.read(robot,step_id=3)
    assert calls==['fingers','left','right']
    np.testing.assert_array_equal(out.fingers,pose_matrices(poses[:2])[[1,0]])
    np.testing.assert_array_equal(out.left,pose_matrices(poses[2:3])[0])
    np.testing.assert_array_equal(out.right,pose_matrices(poses[3:4])[0])
    assert out.require(robot,3) is out
    assert calls==['fingers','left','right']


@pytest.mark.parametrize('step',[None,True,0,-1,3.,4])
def test_snapshot_cannot_be_reused_with_wrong_or_missing_post_fetch_step(step):
    robot,_,_=fixture();out=FetchedHands.read(robot,step_id=3)
    with pytest.raises(RuntimeError):out.require(robot,step)
    with pytest.raises(RuntimeError):out.require(S(),3)


def test_owned_snapshot_never_changes_with_native_buffer_and_is_read_only():
    robot,poses,calls=fixture();old=FetchedHands.read(robot,step_id=3)
    before=old.fingers.copy();poses[:,0]+=1.
    np.testing.assert_array_equal(old.fingers,before)
    for array in (old.fingers,old.left,old.right):
        with pytest.raises(ValueError):array.flat[0]=0.
    fresh=FetchedHands.read(robot,step_id=4)
    np.testing.assert_array_equal(fresh.fingers[:,0,3],old.fingers[:,0,3]+1.)
    assert len(calls)==6


@pytest.mark.parametrize('fault',['nan','zero_quaternion','finger_count','wrist_count','duplicate_order','float_order'])
def test_incomplete_or_invalid_native_views_fail_closed(fault):
    robot,poses,_=fixture()
    if fault=='nan':poses[0,1]=float('nan')
    elif fault=='zero_quaternion':poses[3,3:]=0.
    elif fault=='finger_count':robot.fingers=S(get_transforms=lambda:poses[:1])
    elif fault=='wrist_count':robot.palm=S(get_transforms=lambda:poses[:2])
    elif fault=='duplicate_order':robot.order=[0,0]
    else:robot.order=[1.,0.]
    with pytest.raises((ValueError,RuntimeError)):FetchedHands.read(robot,step_id=1)


def test_grasp_consumer_accepts_only_its_same_step_snapshot():
    from .bimanual_grasp_test import contact_robot
    robot,calls,poses,result=contact_robot()
    robot.palm=robot.right_palm=S(get_transforms=lambda:np.array([[0.,0.,0.,0.,0.,0.,1.]]))
    hands=FetchedHands.read(robot,step_id=37)
    returned=robot.contact_with_frames(.004,np.eye(4)[None],step_id=37,post_fetch=hands)
    assert returned is result and sum(c[0]=='finger_frames' for c in calls)==1
    with pytest.raises(RuntimeError,match='post-fetch step'):
        robot.contact_with_frames(.004,np.eye(4)[None],step_id=38,post_fetch=hands)


def test_cut_consumer_rejects_wrong_step_before_shear_gate(monkeypatch):
    from .signed_blade_test import fixture as blade_fixture
    robot=blade_fixture(monkeypatch)
    hands=FetchedHands(robot,37,np.eye(4)[None].repeat(2,axis=0),np.eye(4),np.eye(4))
    robot.edge_contact_rows=[]
    with pytest.raises(RuntimeError,match='post-fetch step'):
        robot.inspect_cut(.004,None,True,0.,post_fetch=hands,step_id=38)
    assert robot.releases==[]
