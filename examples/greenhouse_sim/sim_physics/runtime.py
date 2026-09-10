"""Batched PhysX body state; no per-tick stage walks, inference, or disk writes."""
import numpy as np


def pose_matrices(poses):
    """PhysX tensor XYZW quaternions to column-vector rigid transforms."""
    poses=np.asarray(poses,dtype=float)
    if poses.ndim!=2 or poses.shape[1]!=7 or not np.isfinite(poses).all():
        raise ValueError('Invalid native body transforms')
    q=poses[:,3:]; norm=np.linalg.norm(q,axis=1)
    if np.any(norm<1e-8): raise ValueError('Invalid native quaternion')
    x,y,z,w=(q/norm[:,None]).T
    m=np.tile(np.eye(4),(len(poses),1,1));m[:,:3,3]=poses[:,:3]
    m[:,0,0]=1-2*(y*y+z*z);m[:,0,1]=2*(x*y-z*w);m[:,0,2]=2*(x*z+y*w)
    m[:,1,0]=2*(x*y+z*w);m[:,1,1]=1-2*(x*x+z*z);m[:,1,2]=2*(y*z-x*w)
    m[:,2,0]=2*(x*z-y*w);m[:,2,1]=2*(y*z+x*w);m[:,2,2]=1-2*(x*x+y*y)
    return m


class PlantRuntime:
    def __init__(self,rig,simulation_view):
        self.rig=rig
        # Isaac 6 attaches an explicit stage ID. Reuse its initialized view;
        # the legacy implicit stage_id=-1 can refer to no attached stage.
        if simulation_view is None: raise RuntimeError('Initialize the stage physics view first')
        self.simulation=simulation_view
        self.simulation.set_subspace_roots('/')
        self.bodies=self.simulation.create_rigid_body_view(rig.root+'/*/Segment_*')
        paths=list(self.bodies.prim_paths)
        if set(paths)!=set(rig.body_paths): raise RuntimeError('Native body view does not match authored rig')
        self.order=np.array([paths.index(p) for p in rig.body_paths],dtype=int)
        articulation=(self.simulation.create_articulation_view(
            rig.cut_joint_path if rig.constraint_mode=='fixed_articulation' else rig.body_paths[rig.cut_index])
                      if rig.constraint_mode!='maximal' else None)
        self.articulation=articulation
        self.contacts=self.simulation.create_rigid_contact_view(rig.root+'/*/Segment_*')
        if self.contacts.sensor_count!=len(paths):
            raise RuntimeError('Native contact reporting is missing for plant bodies')
        self.drive_diagnostics=(dict(articulations=articulation.count,
            names=articulation.shared_metatype.dof_names,
            max_forces=np.asarray(articulation.get_dof_max_forces()).tolist(),
            native_masses=np.asarray(self.bodies.get_masses())[self.order].tolist(),
            native_inertias=np.asarray(self.bodies.get_inertias())[self.order].tolist(),
            motions=np.asarray(articulation.get_dof_motions()).tolist(),
            targets=np.asarray(articulation.get_dof_position_targets()).tolist(),
            stiffness=np.asarray(articulation.get_dof_stiffnesses()).tolist(),
            damping=np.asarray(articulation.get_dof_dampings()).tolist()) if articulation is not None
            else dict(articulations=0,native_drive_readback='unavailable_maximal_coordinate_joints'))
        self.force=np.zeros((len(paths),3),dtype=np.float32)
        self.indices=np.arange(len(paths),dtype=np.uint32)
        self.frames=None
        self.sample()

    def sample(self):
        self.frames=pose_matrices(self.bodies.get_transforms()[self.order])
        velocities=np.asarray(self.bodies.get_velocities())[self.order]
        if not np.isfinite(velocities).all(): raise RuntimeError('Nonfinite physics velocity')
        return self.frames,velocities

    def apply_tip_force(self,force):
        value=np.asarray(force,dtype=np.float32)
        if value.shape!=(3,) or not np.isfinite(value).all() or np.linalg.norm(value)>.2:
            raise ValueError('Diagnostic force must be finite and bounded to 0.2 N')
        self.force.fill(0);self.force[self.order[-1]]=value
        self.bodies.apply_forces(self.force,self.indices,is_global=True)

    def sync_visuals(self):
        self.rig.sync_visuals(self.frames)

    def tip(self):
        length=np.linalg.norm(self.rig.chain_world[-1]-self.rig.chain_world[-2])
        return self.frames[-1,:3,3]+self.frames[-1,:3,2]*length/2
