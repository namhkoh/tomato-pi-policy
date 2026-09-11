"""Pure measured endpoint evidence; no runtime import, controller or release API.

Caller supplies the POST-FETCH native right-wrist world pose, the intended park
pose in that same metre world frame, and explicit clearance evidence for that
same (episode, physics-step) sample. Command targets, FK of desired joints and
elapsed time are not substitutes for the measured pose. This numerical helper
cannot authenticate sensor provenance or establish that clearance was checked.

Fixed strict tolerances match current full-pose IK acceptance in
greenhouse_sim.robot_kinematics.Rby1Kinematics.solve_pose and
sim_physics.redundant_ik.solve_fixed_joint: position < 5e-4 m, rotation < 5e-3 rad.
Their reuse here is an engineering endpoint criterion, not native calibration.
Caller must still verify release/sequence, continuous grasp, guarded motion and
the current full arm/tool/scene clearance. Endpoint evidence certifies neither
the path taken, full forward cutting stroke, tissue fracture nor physical safety.
"""
import math
import operator

import numpy as np


POSITION_TOLERANCE_M = 5e-4
ORIENTATION_TOLERANCE_RAD = 5e-3


def _pose(value):
    pose = np.array(value, dtype=float, copy=True)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('Finite 4x4 world pose required')
    rotation = pose[:3, :3]
    if (not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-8, rtol=0.)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0.)
            or not math.isclose(float(np.linalg.det(rotation)), 1., abs_tol=1e-6, rel_tol=0.)):
        raise ValueError('Proper rigid world pose required; no reflection, scale or shear')
    return pose


def _sample_id(value):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError('Sample ID must be (episode, physics_step)')
    result = []
    for item in value:
        if isinstance(item, (bool, np.bool_)):
            raise ValueError('Nonnegative integer episode and physics step required')
        try:
            number = operator.index(item)
        except TypeError as exc:
            raise ValueError('Nonnegative integer episode and physics step required') from exc
        if number < 0:
            raise ValueError('Nonnegative integer episode and physics step required')
        result.append(number)
    return tuple(result)


def withdrawal_evidence(measured_wrist_world, park_wrist_world, *, clearance_verified,
                        measured_sample_id, clearance_sample_id):
    """Return current endpoint evidence, without latching a prior success.

    clearance_verified must be a Python bool from the caller's CURRENT full
    clearance checks, not merely 'no contact' or an old path-planning success.
    Mismatched valid sample IDs return unverified; malformed inputs raise.
    Matrices within floating-point rigid-frame tolerance are accepted without
    silently repairing an invalid frame. The angle uses atan2(skew norm, trace),
    remaining well-defined near both zero and pi (including a 180-degree error).
    """
    if type(clearance_verified) is not bool:
        raise ValueError('Explicit boolean clearance evidence required')
    measured = _pose(measured_wrist_world)
    park = _pose(park_wrist_world)
    measured_id = _sample_id(measured_sample_id)
    clearance_id = _sample_id(clearance_sample_id)
    with np.errstate(over='ignore', invalid='ignore'):
        delta = measured[:3, 3] - park[:3, 3]
    position_error = math.hypot(*delta)
    if not math.isfinite(position_error):
        raise ValueError('Unrepresentable pose difference')
    relative = park[:3, :3] @ measured[:3, :3].T
    skew = np.array([relative[2, 1] - relative[1, 2],
                     relative[0, 2] - relative[2, 0],
                     relative[1, 0] - relative[0, 1]])
    rotation_error = math.atan2(math.hypot(*skew) / 2.,
                                float(np.clip((np.trace(relative) - 1.) / 2., -1., 1.)))
    attained = bool(position_error < POSITION_TOLERANCE_M
                    and rotation_error < ORIENTATION_TOLERANCE_RAD)
    same_sample = measured_id == clearance_id
    completed = attained and clearance_verified and same_sample
    return dict(model='measured_park_pose_with_explicit_clearance_v1',
                right_withdrawal_completed=completed, endpoint_attained=attained,
                clearance_verified=clearance_verified, same_native_sample=same_sample,
                measured_sample_id=list(measured_id), clearance_sample_id=list(clearance_id),
                position_error_m=position_error, orientation_error_rad=rotation_error,
                position_tolerance_m=POSITION_TOLERANCE_M,
                orientation_tolerance_rad=ORIENTATION_TOLERANCE_RAD,
                comparison='strict_less_than', native_provenance_verified_by_helper=False,
                whole_path_certified=False, full_forward_cutstroke_verified=False,
                physical_cut_verified=False, tissue_fracture_calibrated=False)
