"""Native angular-joint gravity allocation inside the original effort envelope.

Opt-in diagnostic controller improvement, not identified hardware dynamics.
PD stiffness/damping and the original 70% PD ceiling do not increase. Gravity
may use more than the historical 30% only by reserving less effort for PD.
At least 10% of source effort remains available for corrective control.
"""
import numpy as np


def source_limits(names,effort):
    """Budget only torso/arms/head; fixed-base wheel drives stay historical.

    The imported continuous wheel DOFs have no URDF effort entry. They are
    not arm actuators, and inventing a source limit for them would be wrong.
    Missing limits for a selected manipulation joint remain a hard error.
    """
    indices=[i for i,name in enumerate(names)
             if name.startswith(('torso_','left_arm_','right_arm_','head_'))]
    return indices,np.array([effort[names[i]] for i in indices],dtype=float)


def _down32(values):
    """Nonnegative float32 values rounded toward zero, never above a budget."""
    result=values.astype(np.float32)
    return np.where(result.astype(float)>values,
                    np.nextafter(result,np.float32(0)),result)


def allocate(gravity,total_effort,pd_ceiling):
    gravity=np.asarray(gravity,dtype=float)
    total=np.asarray(total_effort,dtype=float)
    ceiling=np.asarray(pd_ceiling,dtype=float)
    if (gravity.ndim!=1 or not gravity.size or total.shape!=gravity.shape
            or ceiling.shape!=gravity.shape or not np.isfinite(gravity).all()
            or not np.isfinite(total).all() or not np.isfinite(ceiling).all()
            or np.any(total<=0) or np.any(total>np.finfo(np.float32).max)
            or np.any(ceiling<.1*total) or np.any(ceiling>total)):
        raise ValueError('Finite angular gravity, source effort and corrective PD reserve required')
    if np.any(np.abs(gravity)>.9*total):
        raise ValueError('Native gravity exceeds 90% of source effort; insufficient dynamic reserve')
    feedforward=np.copysign(_down32(np.abs(gravity)),gravity).astype(np.float32)
    drive=_down32(np.minimum(ceiling,total-np.abs(feedforward.astype(float))))
    if np.any(np.abs(feedforward.astype(float))+drive.astype(float)>total):
        raise RuntimeError('Angular effort allocation exceeded the source envelope')
    return feedforward,drive
