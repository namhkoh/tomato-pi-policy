import numpy as np
import pytest
from sim_physics.redundant_ik import solve_fixed_joint


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
