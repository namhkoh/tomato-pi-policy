"""Joint-space park goal evaluated in a measured native station frame.

This is NOT an absolute world-space park goal. Only the seven unchanged right
reference joint angles are substituted; the actual wrist is never used as its
own goal. Reads only, with full native/URDF consistency and snapshot checks.
"""
import numpy as np
from .measured_withdrawal_native import _raw_joint_world, RIGHT
from .measured_withdrawal import _attained


class StationPark:
    def __init__(self, fixture, world, *, right_reference=None):
        a=fixture.robot;names=tuple(a.shared_metatype.dof_names)
        self.q=np.array(a.get_dof_positions(),dtype=float,copy=True)
        self.fixture=fixture;self.names=names
        self.world={k:np.array(v,dtype=float,copy=True) for k,v in world.items()}
        if (a.count!=1 or not a.shared_metatype.fixed_base or names!=tuple(fixture.names)
                or not names or len(set(names))!=len(names) or self.q.shape!=(1,len(names))
                or not np.isfinite(self.q).all() or 'base' not in world or 'ee_right' not in world):
            raise ValueError('Complete fixed-base native robot state required for joint-space park reference')
        values=dict(zip(names,self.q[0],strict=True))
        predicted=_raw_joint_world(fixture.kin,values,self.world['base'])
        if any(k not in predicted or not _attained(v,predicted[k]) for k,v in self.world.items()):
            raise ValueError('Measured station/body FK mismatch; no park reference repair')
        self.original=np.array(fixture.right,dtype=float,copy=True)
        self.explicit_reference=right_reference
        goal=np.array(fixture.right if right_reference is None else right_reference,dtype=float,copy=True)
        low,high=fixture.kin.arm_limits_degrees('right')
        if (goal.shape!=(7,) or not np.isfinite(goal).all() or np.any(goal<=low) or np.any(goal>=high)):
            raise ValueError('Unchanged strict-source-limit right park joints required')
        self.goal=goal
        values.update(zip(RIGHT,np.radians(goal),strict=True))
        self.pose=_raw_joint_world(fixture.kin,values,self.world['base'])['ee_right']
        self.verify(world)

    def verify(self, world):
        a=self.fixture.robot
        if (tuple(a.shared_metatype.dof_names)!=self.names
                or not np.array_equal(a.get_dof_positions(),self.q)
                or not np.array_equal(np.asarray(self.fixture.right),self.original)
                or self.explicit_reference is not None and not np.array_equal(self.explicit_reference,self.goal)
                or set(world)!=set(self.world)
                or any(not np.array_equal(world[k],self.world[k]) for k in world)):
            raise RuntimeError('Native station or unchanged park reference changed during verification')

    def report(self):
        return dict(model='native_station_unchanged_right_joint_park_v1',
                    reference_kind='initial_waiting_joints' if self.explicit_reference is None else 'explicit_screened_egress_joints',
                    goal_kind='right_joint_configuration_not_absolute_world_pose',
                    reference_right_degrees=self.goal.tolist(),
                    station_basis='current_native_base_and_nonright_joint_positions',
                    full_body_fk_consistency_checked=True,actual_wrist_used_as_goal=False,
                    native_state_changed=False,position_tolerance_m=.0005,orientation_tolerance_rad=.005)
