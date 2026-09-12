"""Downward Cartesian proposals; never permissions to move or cut."""
import numpy as np


def downward_direction(axis):
    """Closest transverse direction to gravity, no upward/sideways fallback.

    A tilted petiole needs a small lateral component for a transverse cut.
    Reject slopes requiring >30 degrees from down; do not rotate the anatomy.
    """
    axis=np.asarray(axis,float)
    if axis.shape!=(3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1)>1e-5:
        raise ValueError('Measured unit stem axis required')
    axis=axis/np.linalg.norm(axis)
    down=np.array([0.,0.,-1.]);direction=down-axis*np.dot(down,axis)
    length=np.linalg.norm(direction)
    if length<np.cos(np.radians(30)):
        raise ValueError('Target cannot support a transverse cut within 30 degrees of down')
    return direction/length


def arm_extension(frames):
    """Shoulder-to-wrist chord / upper+forearm length, using exact URDF FK."""
    points=[np.asarray(frames['link_right_arm_'+str(i)],float)[:3,3] for i in (2,3,4)]
    lengths=[np.linalg.norm(b-a) for a,b in zip(points[:-1],points[1:])]
    if not np.isfinite(points).all() or min(lengths)<.01:
        raise ValueError('Valid right shoulder/elbow/wrist frames required')
    return float(np.linalg.norm(points[2]-points[0])/sum(lengths))


def cartesian_transit(robot,left,goal):
    """Straight wrist-position transit, smooth rotation, no joint-space detour.

    Every <=2 mm / <=1 degree pose plus <=1 degree joint interpolation is
    screened. This is sampled geometry only; native tracking/contact guards
    and reobservation still apply. A failed path is rejected, never rerouted.
    """
    from scipy.spatial.transform import Rotation,Slerp
    start=robot.kin.forward('right',robot.right,robot.base)
    end=robot.kin.forward('right',goal,robot.base)
    rotation=Rotation.from_matrix(np.array([start[:3,:3],end[:3,:3]]))
    angle=np.linalg.norm((rotation[1]*rotation[0].inv()).as_rotvec())
    count=max(2,int(np.ceil(np.linalg.norm(end[:3,3]-start[:3,3])/.002))+1,
        int(np.ceil(np.degrees(angle)))+1)
    fractions=np.linspace(0,1,count);rotations=Slerp([0,1],rotation)(fractions).as_matrix()
    path=[robot.right.copy()];minimum=float('inf');checks=0
    for alpha,rot in zip(fractions[1:],rotations[1:]):
        desired=np.eye(4);desired[:3,:3]=rot
        desired[:3,3]=(1-alpha)*start[:3,3]+alpha*end[:3,3]
        solved=robot.solve_right_pose(desired,path[-1])
        if not solved.succeeded:return None
        q=np.asarray(solved.joint_degrees)
        # Densify joints too, without changing the scheduled Cartesian knots.
        n=max(1,int(np.ceil(np.max(abs(q-path[-1])))))
        for row in np.linspace(path[-1],q,n+1)[1:]:
            checks+=1
            clearance=robot.kin.inter_arm_clearance(left,row,robot.base).clearance_m
            if (clearance<.01 or not robot.check_self(left,row)['passed']
                    or not robot.check_held_plant(left,row)):return None
            # Linear joint interpolation must stay close to the Cartesian line.
            point=robot.kin.forward('right',row,robot.base)[:3,3]
            delta=end[:3,3]-start[:3,3];length2=float(delta@delta)
            t=float((point-start[:3,3])@delta/length2) if length2>1e-16 else 0.
            closest=start[:3,3]+np.clip(t,0,1)*delta
            if np.linalg.norm(point-closest)>.0005:return None
            minimum=min(minimum,clearance)
        path.append(q)
    # Match the very same endpoint configuration used by the stroke solver.
    if np.max(abs(path[-1]-goal))>.25:return None
    return np.asarray(path),minimum,dict(method='straight_cartesian_no_detour',
        collision_checks=checks,maximum_cartesian_sample_step_m=.002,
        maximum_joint_sample_step_degrees=1.,maximum_line_error_m=.0005,
        whole_scene_certified=False)
