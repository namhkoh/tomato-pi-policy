"""Pre-spawn waiting-pose proposal, never a runtime withdrawal permission.

Raise the complete right wrist without changing its knife/camera orientation.
Startup, approach, ongoing native guards and final clearance must still pass.
"""
import numpy as np


def propose(kin, right, base, lift, retreat=0.):
    if (isinstance(lift, (bool, np.bool_)) or not isinstance(lift, (int, float))
            or not np.isfinite(lift) or not 0 <= lift <= .05):
        raise ValueError('Finite initial wrist lift within 0..50 mm required')
    if (isinstance(retreat,(bool,np.bool_)) or not isinstance(retreat,(int,float))
            or not np.isfinite(retreat) or not 0<=retreat<=.05):
        raise ValueError('Finite initial wrist retreat within 0..50 mm required')
    initial = np.array(right, dtype=float, copy=True)
    if lift == 0 and retreat == 0:
        return initial, None
    low, high = kin.arm_limits_degrees('right')
    if (initial.shape != (7,) or not np.isfinite(initial).all()
            or np.any(initial <= low) or np.any(initial >= high)):
        raise ValueError('Original in-limit right waiting configuration required')
    origin = kin.forward('right', initial, base)
    target = origin.copy(); target[2, 3] += lift
    # The fitted knife extends along wrist -Z. Withdraw the whole tool
    # toward its wrist along +Z, not toward the left fingers or into the
    # parent merely to increase world height. The original native checks
    # still determine whether this proposed waiting pose/path is usable.
    target[:3,3]+=retreat*origin[:3,2]
    solved = kin.solve_pose('right', target, initial, base, maximum_evaluations=400,
                           joint_limit_margin_degrees=3.)
    if not solved.succeeded:
        raise RuntimeError('Higher waiting-pose IK failed; no spawn or motion')
    q = np.array(solved.joint_degrees, dtype=float, copy=True)
    actual = kin.forward('right', q, base)
    from scipy.spatial.transform import Rotation
    position = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    rotation = float(Rotation.from_matrix(actual[:3, :3] @ target[:3, :3].T).magnitude())
    if (not np.isfinite(q).all() or np.any(q <= low) or np.any(q >= high)
            or not position < .0005 or not rotation < .005):
        raise RuntimeError('Higher waiting pose failed independent FK verification')
    return q, dict(model='prephysics_world_up_waiting_pose_proposal_v1',
        lift_world_z_m=float(lift),retreat_wrist_plus_z_m=float(retreat),
        original_right_degrees=initial.tolist(),
        proposed_right_degrees=q.tolist(), position_error_m=position,
        orientation_error_rad=rotation, desired_wrist_world=target.tolist(),
        source_assets_changed=False, native_pose_changed=False,
        motion_authorized=False, startup_and_whole_path_checks_required=True)
