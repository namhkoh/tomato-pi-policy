from copy import deepcopy
import numpy as np
import pytest

from sim_data.training_plan import view_specs
from sim_data.training_views import with_postures,robot_for_spec


def test_posture_sampling_is_deterministic_independent_and_nonmutating():
    specs=view_specs(.8,0,'a',4); before=deepcopy(specs)
    a=with_postures(specs,'a')
    assert a==with_postures(specs,'a') and a!=with_postures(specs,'b')
    assert specs==before
    assert all(0<=s['torso_bend_degrees']<=45 for s in a)


def test_torso_height_diversity_uses_real_joints_preserves_arms_and_changes_head_height():
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    from greenhouse_sim.robot_scene import SDK_READY_POSE_DEGREES
    m=Rby1Kinematics(); robot={'pose_degrees':dict(SDK_READY_POSE_DEGREES)}
    before=deepcopy(robot); heights=[]
    for bend in (0.,15.,30.,45.):
        changed=robot_for_spec(robot,{'torso_bend_degrees':bend})
        links=m.all_link_transforms(changed['pose_degrees']) # Validates exact URDF limits.
        heights.append(float(links['link_head_2'][2,3]))
        assert all(changed['pose_degrees'][n]==v for n,v in robot['pose_degrees'].items() if not n.startswith('torso_'))
    assert robot==before and max(heights)-min(heights)>.2


@pytest.mark.parametrize('value',[float('nan'),-1,46,True])
def test_invalid_torso_poses_fail(value):
    with pytest.raises(ValueError): robot_for_spec({'pose_degrees':{}},{'torso_bend_degrees':value})
