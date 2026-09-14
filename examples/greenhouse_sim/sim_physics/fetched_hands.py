"""One post-fetch hand-pose snapshot, local to ONE after-step callback.

No cross-step cache, FK substitution, interpolation or skipped native step.
Read all three native views once and batch the identical quaternion conversion.
Consumers must bind the same fixture and post-fetch step explicitly.
"""
from dataclasses import dataclass
import numpy as np
from .runtime import pose_matrices


@dataclass(frozen=True)
class FetchedHands:
    owner: object
    step_id: int
    fingers: np.ndarray
    left: np.ndarray
    right: np.ndarray

    @classmethod
    def read(cls,robot,*,step_id):
        if type(step_id) is not int or step_id<1:
            raise ValueError('Positive post-fetch physics step required')
        fingers=np.asarray(robot.fingers.get_transforms())
        left=np.asarray(robot.palm.get_transforms())
        right=np.asarray(robot.right_palm.get_transforms())
        order=np.asarray(robot.order)
        if (fingers.shape!=(2,7) or left.shape!=(1,7) or right.shape!=(1,7)
                or order.shape!=(2,) or order.dtype.kind not in 'iu' or sorted(order.tolist())!=[0,1]):
            raise RuntimeError('Exact native two-finger and wrist views required')
        frames=pose_matrices(np.concatenate((fingers[order],left,right)))
        frames.setflags(write=False)
        return cls(robot,step_id,frames[:2],frames[2],frames[3])

    def require(self,robot,step_id):
        if robot is not self.owner or type(step_id) is not int or step_id!=self.step_id:
            raise RuntimeError('Hand snapshot belongs to a different robot or post-fetch step')
        return self
