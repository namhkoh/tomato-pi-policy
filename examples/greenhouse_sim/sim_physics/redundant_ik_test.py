import numpy as np
import pytest
from sim_physics.redundant_ik import solve_fixed_joint


def test_pose_family_changes_posture_without_changing_tool_pose():
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    from sim_physics.redundant_ik import pose_family
    kin=Rby1Kinematics();q=np.array([-35.,-50,20,-75,15,60,10]);base=np.eye(4)
    target=kin.forward('right',q,base);original=q.copy()
    results=list(pose_family(kin,'right',target,q,base,steps_per_direction=4))
    assert len(results)==8
    lower,upper=kin.arm_limits_degrees('right')
    for result in results:
        candidate=np.asarray(result.joint_degrees)
        assert result.succeeded and result.position_error_m<5e-4 and result.orientation_error_rad<5e-3
        assert np.all(candidate>lower) and np.all(candidate<upper)
        assert np.linalg.norm(candidate-q)>.1
    np.testing.assert_array_equal(q,original)
    repeated=list(pose_family(kin,'right',target,q,base,steps_per_direction=4))
    np.testing.assert_allclose([r.joint_degrees for r in results],[r.joint_degrees for r in repeated])
    target[0,3]+=.01
    with pytest.raises(ValueError,match='verified initial'):
        list(pose_family(kin,'right',target,q,base))


@pytest.mark.parametrize('options',[dict(steps_per_direction=0),dict(steps_per_direction=33),
    dict(step_degrees=9),dict(step_degrees=float('nan'))])
def test_invalid_pose_family_budget_fails_closed(options):
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    from sim_physics.redundant_ik import pose_family
    kin=Rby1Kinematics();q=np.array([-35.,-50,20,-75,15,60,10]);base=np.eye(4)
    with pytest.raises(ValueError): list(pose_family(kin,'right',kin.forward('right',q,base),q,base,**options))


def test_fixed_shoulder_uses_exact_fk_and_preserves_joint_limits():
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    kin=Rby1Kinematics();q=np.array([-35.,-50,20,-75,15,60,10]);base=np.eye(4)
    target=kin.forward('right',q,base)
    result=solve_fixed_joint(kin,'right',target,q+np.array([2,5,-2,1,3,-1,2]),base,joint_degrees=-50)
    assert result.succeeded and result.joint_degrees[1]==pytest.approx(-50)
    np.testing.assert_allclose(kin.forward('right',result.joint_degrees,base),target,atol=1e-6)
    lower,upper=kin.arm_limits_degrees('right')
    assert np.all(np.asarray(result.joint_degrees)>lower) and np.all(np.asarray(result.joint_degrees)<upper)
    target[:3,3]+=10
    assert not solve_fixed_joint(kin,'right',target,q,base,joint_degrees=-50).succeeded
    with pytest.raises(ValueError): solve_fixed_joint(kin,'right',target,q,base,joint_degrees=180)
