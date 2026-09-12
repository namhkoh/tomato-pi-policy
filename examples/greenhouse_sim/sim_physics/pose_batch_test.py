import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from sim_physics.shaft_grasp import _pose,_poses


def test_owned_read_only_batch_matches_scalar_without_state_cache():
    a=np.repeat(np.eye(4)[None],40,axis=0)
    a[:,:3,:3]=Rotation.random(40,random_state=4).as_matrix()
    a[:,:3,3]=np.random.default_rng(4).normal(size=(40,3))
    result=_poses(a)
    np.testing.assert_array_equal(result,np.array([_pose(m) for m in a]))
    assert not result.flags.writeable
    a[:]=0
    assert not np.any(result[:,3,3]==0)
    with pytest.raises(ValueError):_poses(a)


@pytest.mark.parametrize('entry,value',[( (3,0),1.01e-7),((0,0),1.00002),((0,0),-1),
    ((0,3),float('nan')),((0,3),float('inf')),((3,3),0)])
def test_one_bad_frame_rejects_entire_batch(entry,value):
    a=np.repeat(np.eye(4)[None],20,axis=0);a[13,entry[0],entry[1]]=value
    with pytest.raises(ValueError):_pose(a[13])
    with pytest.raises(ValueError):_poses(a)


@pytest.mark.parametrize('delta',[0,1e-7,np.nextafter(1e-7,np.inf),-1e-7])
def test_batch_preserves_scalar_boundary(delta):
    a=np.eye(4);a[3,1]=delta
    try:expected=_pose(a)
    except ValueError:
        with pytest.raises(ValueError):_poses(a[None])
    else:np.testing.assert_array_equal(_poses(a[None])[0],expected)
