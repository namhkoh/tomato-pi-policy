"""Bounded shoulder-branch proposals using the repository's exact URDF FK.

Fixing one of seven joints samples redundancy that a seed-regularized full-pose
solve can miss. Every result still needs full collision/path/native validation.
"""
import numpy as np


def pose_family(kin,side,desired,seed,base,*,steps_per_direction=16,step_degrees=4.):
    """Trace the local 7-DOF self-motion family, keeping the same tool pose.

    Predictor/corrector continuation avoids fixing a shoulder joint that may
    not span the redundant branch. These are proposals, NOT collision-free
    motions. The caller must screen every result and plan a separate transit.
    """
    from scipy.spatial.transform import Rotation
    desired=np.asarray(desired,float);seed=np.asarray(seed,float)
    lower,upper=kin.arm_limits_degrees(side)
    if (desired.shape!=(4,4) or seed.shape!=(7,) or not np.isfinite(desired).all()
            or not np.isfinite(seed).all() or type(steps_per_direction) is not int
            or not 1<=steps_per_direction<=32 or not np.isfinite(step_degrees)
            or not .5<=step_degrees<=8 or np.any(seed<=lower) or np.any(seed>=upper)):
        raise ValueError('Invalid bounded pose-family proposal')
    actual=kin.forward(side,seed,base)
    if (not np.allclose(desired[3],[0,0,0,1])
            or not np.allclose(desired[:3,:3].T@desired[:3,:3],np.eye(3),atol=1e-6)
            or np.linalg.det(desired[:3,:3])<0
            or np.linalg.norm(actual[:3,3]-desired[:3,3])>=5e-4
            or np.linalg.norm(Rotation.from_matrix(desired[:3,:3]@actual[:3,:3].T).as_rotvec())>=5e-3):
        raise ValueError('Pose-family continuation requires a verified initial IK solution')
    seen=[seed.copy()]
    for sign in (1.,-1.):
        q=seed.copy();previous=None
        for _ in range(steps_per_direction):
            actual=kin.forward(side,q,base);columns=[]
            epsilon=1e-5
            for i in range(7):
                perturb=q.copy();perturb[i]+=np.degrees(epsilon)
                moved=kin.forward(side,perturb,base)
                columns.append(np.r_[(moved[:3,3]-actual[:3,3])/.01,
                    Rotation.from_matrix(moved[:3,:3]@actual[:3,:3].T).as_rotvec()/.15]/epsilon)
            _,singular,vh=np.linalg.svd(np.asarray(columns).T,full_matrices=True)
            if singular[-1]<1e-5: break
            tangent=vh[-1]
            if previous is None:
                tangent*=sign*np.sign(tangent[np.argmax(abs(tangent))])
            elif np.dot(tangent,previous)<0: tangent=-tangent
            prediction=q+step_degrees*tangent
            if np.any(prediction<=lower+.001) or np.any(prediction>=upper-.001): break
            result=kin.solve_pose(side,desired,prediction,base,maximum_evaluations=100)
            candidate=np.asarray(result.joint_degrees)
            if (not result.succeeded or not np.isfinite(candidate).all()
                    or np.any(candidate<=lower) or np.any(candidate>=upper)
                    or np.linalg.norm(candidate-q)>2*step_degrees
                    or min(np.linalg.norm(candidate-old) for old in seen)<.1): break
            seen.append(candidate.copy());q=candidate;previous=tangent
            yield result


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
