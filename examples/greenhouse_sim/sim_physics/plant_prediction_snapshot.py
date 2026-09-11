"""Read-only full-plant dynamics snapshots for a contact-prediction diagnostic.

No solver, inferred contacts, force application or state override. All six
floating root coordinates remain in the measured mass matrix. An external
support constraint is labelled separately, not silently Schur-eliminated.
Native COM Jacobian convention is checked against native body velocities.
"""
import numpy as np
from .runtime import pose_matrices
from .contact_coupled_prediction import _matrix


def _array(value,shape,name):
    x=np.array(value,dtype=float,copy=True)
    if x.shape!=shape or not np.isfinite(x).all():
        raise ValueError('Finite exact '+name+' required')
    return x


class PlantPredictionSnapshot:
    def __init__(self,articulation,*,source_target,expected_body_paths):
        self.a=articulation;self.names=list(articulation.shared_metatype.dof_names)
        self.paths=list(articulation.link_paths[0]);self.target=source_target
        if (articulation.count!=1 or articulation.shared_metatype.fixed_base
                or not isinstance(source_target,str) or not source_target
                or not 1<=len(self.names)<=58 or len(set(self.names))!=len(self.names)
                or self.paths!=list(expected_body_paths) or not self.paths
                or len(set(self.paths))!=len(self.paths)):
            raise ValueError('Exact single floating plant articulation inventory required')
        self.d=len(self.names);self.n=self.d+6;self.b=len(self.paths)

    def read(self,*,step,root_constrained):
        a=self.a
        if (type(step) is not int or step<0 or type(root_constrained) is not bool
                or a.count!=1 or a.shared_metatype.fixed_base
                or list(a.shared_metatype.dof_names)!=self.names
                or list(a.link_paths[0])!=self.paths):
            raise ValueError('Native plant snapshot inventory or step changed')
        q=_array(a.get_dof_positions(),(1,self.d),'plant q')[0]
        qv=_array(a.get_dof_velocities(),(1,self.d),'plant qdot')[0]
        root=_array(a.get_root_velocities(),(1,6),'plant root velocity')[0]
        v=np.r_[root,qv]
        rawM=_array(a.get_generalized_mass_matrices(),(1,self.n,self.n),'floating mass')[0]
        _matrix(rawM,self.n,'floating mass',positive=True)  # Check; retain raw readback below.
        jac=_array(a.get_jacobians(),(1,self.b,6,self.n),'COM Jacobians')[0]
        frames=pose_matrices(_array(a.get_link_transforms(),(1,self.b,7),'plant poses')[0])
        velocity=_array(a.get_link_velocities(),(1,self.b,6),'plant body velocities')[0]
        com=_array(a.get_coms(),(1,self.b,7),'native COM poses')[0]
        mapped=np.einsum('bij,j->bi',jac,v);error=float(np.max(abs(mapped-velocity)))
        allowance=1e-6+1e-5*float(np.max(abs(velocity)))
        if error>allowance:
            raise ValueError('Native COM Jacobian/velocity convention disagrees')
        gravity=_array(a.get_gravity_compensation_forces(),(1,self.n),'gravity compensation')[0]
        coriolis=_array(a.get_coriolis_and_centrifugal_compensation_forces(),(1,self.n),'Coriolis compensation')[0]
        return dict(schema='full_plant_native_prediction_snapshot_v1',step_id=step,
            source_target=self.target,body_paths=list(self.paths),joint_names=list(self.names),
            q_rad=q.tolist(),generalized_velocity=v.tolist(),mass_matrix=rawM.tolist(),
            body_world_com_jacobians=jac.tolist(),body_frames_world=frames.tolist(),
            body_velocities_world=velocity.tolist(),native_com_local_poses=com.tolist(),
            native_known_noncontact_force=(-gravity-coriolis).tolist(),
            point_velocity_max_error=error,point_velocity_error_allowance=allowance,
            native_com_jacobian_velocity_check_passed=True,
            external_root_support_enabled_caller_asserted=root_constrained,
            root_coordinates_retained=6,contact_force_included=False,
            constraint_forces_included=False,contact_law_parity=False,
            scope='diagnostic_readback_not_execution_or_training_approval',training_eligible=False)
