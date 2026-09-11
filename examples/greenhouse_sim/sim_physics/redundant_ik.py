"""Bounded shoulder-branch proposals using the repository's exact URDF FK.

Fixing one of seven joints samples redundancy that a seed-regularized full-pose
solve can miss. Every result still needs full collision/path/native validation.
"""
import numpy as np


def solve_fixed_joint(kin,side,desired,seed,base,*,joint_index=1,joint_degrees,maximum_evaluations=250):
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation
    from greenhouse_sim.robot_kinematics import IKResult
    desired=np.asarray(desired,float);seed=np.asarray(seed,float)
    lower,upper=kin.arm_limits_degrees(side)
    if (desired.shape!=(4,4) or seed.shape!=(7,) or not np.isfinite(desired).all()
            or not np.isfinite(seed).all() or not isinstance(joint_index,int) or not 0<=joint_index<7
            or not np.isfinite(joint_degrees) or not lower[joint_index]<joint_degrees<upper[joint_index]
            or not isinstance(maximum_evaluations,int) or not 1<=maximum_evaluations<=1000):
        raise ValueError('Invalid bounded redundancy proposal')
    if (not np.allclose(desired[:3,:3].T@desired[:3,:3],np.eye(3),atol=1e-6)
            or np.linalg.det(desired[:3,:3])<0 or not np.allclose(desired[3],[0,0,0,1])):
        raise ValueError('Target pose must be rigid')
    indices=np.delete(np.arange(7),joint_index)
    q=np.radians(seed);q[joint_index]=np.radians(joint_degrees)
    low=np.radians(lower[indices])+1e-5;high=np.radians(upper[indices])-1e-5
    def unpack(x):
        result=q.copy();result[indices]=x;return np.degrees(result)
    def residual(x):
        actual=kin.forward(side,unpack(x),base)
        return np.r_[(actual[:3,3]-desired[:3,3])/.01,
            Rotation.from_matrix(desired[:3,:3]@actual[:3,:3].T).as_rotvec()/.15]
    result=least_squares(residual,np.clip(q[indices],low,high),bounds=(low,high),
        max_nfev=maximum_evaluations,xtol=1e-10,ftol=1e-10,gtol=1e-10)
    degrees=unpack(result.x);actual=kin.forward(side,degrees,base)
    position=float(np.linalg.norm(actual[:3,3]-desired[:3,3]))
    rotation=float(np.linalg.norm(Rotation.from_matrix(desired[:3,:3]@actual[:3,:3].T).as_rotvec()))
    return IKResult(joint_degrees=tuple(degrees),position_error_m=position,orientation_error_rad=rotation,
        cost=float(result.cost),succeeded=bool(result.success and position<5e-4 and rotation<5e-3),evaluations=int(result.nfev))
