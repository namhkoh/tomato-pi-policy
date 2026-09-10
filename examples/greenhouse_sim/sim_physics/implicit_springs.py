"""Experimental coupled implicit elastic effort, not a robot motion controller.

Frozen-M backward Euler spring/damper step; related to implicit PD:
https://faculty.cc.gatech.edu/~turk/my_papers/stable_pd.pdf
Unlike that paper's first-order position predictor, this includes h**2 K.
No gain/mass inflation. Nonlinear inertia and contacts still need qualification.
"""
import numpy as np


def elastic_effort(mass, position, velocity, stiffness, damping, external, dt):
    q,v,k,c,f=[np.asarray(x,dtype=float) for x in
                (position,velocity,stiffness,damping,external)]
    m=np.asarray(mass,dtype=float)
    if (q.ndim!=1 or not len(q) or any(x.shape!=q.shape for x in (v,k,c,f))
            or m.shape!=(len(q),len(q)) or not np.isfinite(dt) or dt<=0
            or any(not np.isfinite(x).all() for x in (q,v,k,c,f,m))
            or np.any(k<0) or np.any(c<0)):
        raise ValueError('Invalid implicit spring system')
    if np.max(np.abs(m-m.T))>1e-5*np.max(np.abs(m))+1e-15:
        raise ValueError('Native mass matrix is not symmetric within float32 precision')
    # Symmetrize only roundoff from native float32 mass-matrix readback.
    m=(m+m.T)/2
    np.linalg.cholesky(m)  # Reject invalid physical inertia, do not regularize it.
    system=m+np.diag(dt*c+dt*dt*k)
    next_velocity=np.linalg.solve(system,m@v+dt*(f-k*q))
    return -k*(q+dt*next_velocity)-c*next_velocity


class ImplicitJointSprings:
    """Diagnostic springs. Unknown contact loads are NOT yet incorporated.

An external fixed support may constrain the six root DOFs before release.
After release all six root DOFs participate in the coupled mass solve.
"""
    def __init__(self, articulation):
        self.articulation=articulation
        if articulation.count!=1:
            raise ValueError('Implicit spring diagnostic requires one articulation')
        self.fixed_base=articulation.shared_metatype.fixed_base
        self.k=np.asarray(articulation.get_dof_stiffnesses(),dtype=float)[0].copy()
        self.c=np.asarray(articulation.get_dof_dampings(),dtype=float)[0].copy()
        self.indices=np.array([0],dtype=np.uint32)
        zeros=np.zeros((1,len(self.k)),dtype=np.float32)
        articulation.set_dof_stiffnesses(zeros,self.indices)
        articulation.set_dof_dampings(zeros,self.indices)

    def step(self, dt, external_joint_force=None, *, root_constrained=False):
        a=self.articulation
        mass=np.asarray(a.get_generalized_mass_matrices(),dtype=float)[0]
        q=np.asarray(a.get_dof_positions(),dtype=float)[0]
        v=np.asarray(a.get_dof_velocities(),dtype=float)[0]
        k,c=self.k,self.c
        external=(-np.asarray(a.get_gravity_compensation_forces(),dtype=float)[0]
                  -np.asarray(a.get_coriolis_and_centrifugal_compensation_forces(),dtype=float)[0])
        if external_joint_force is not None: external+=external_joint_force
        if not self.fixed_base:
            if root_constrained:
                # This is a conditional block, NOT a free-root Schur complement.
                mass=mass[6:,6:];external=external[6:]
            else:
                q=np.r_[np.zeros(6),q]
                v=np.r_[np.asarray(a.get_root_velocities(),dtype=float)[0],v]
                k=np.r_[np.zeros(6),k];c=np.r_[np.zeros(6),c]
        effort=elastic_effort(mass,q,v,k,c,external,dt)
        if not self.fixed_base and not root_constrained:
            if np.any(effort[:6]!=0): raise RuntimeError('Passive springs must not actuate the root')
            effort=effort[6:]
        a.set_dof_actuation_forces(effort[None,:].astype(np.float32),self.indices)
        return effort


class NativeBodyLoads:
    """Map known forces at a body's COM into native generalized coordinates.

Verify the Jacobian reference point against independently reported gravity
efforts. Never guess between a prim-origin and COM Jacobian convention.
"""
    def __init__(self, articulation, gravity):
        from .runtime import pose_matrices
        self.articulation=articulation
        self.skip=int(articulation.shared_metatype.fixed_base)
        self.paths=list(articulation.link_paths[0])
        self.local_com=np.asarray(articulation.get_coms())[0,:, :3].copy()
        self.mass=np.asarray(articulation.get_masses())[0].reshape(-1).copy()
        jac=self._jacobian()
        frames=pose_matrices(articulation.get_link_transforms()[0])
        com=np.einsum('lij,lj->li',frames[:,:3,:3],self.local_com)[self.skip:]
        self.reference_errors={}
        expected=-np.asarray(articulation.get_gravity_compensation_forces())[0]
        if np.linalg.norm(expected)<1e-10:
            raise ValueError('Gravity reference is required to verify Jacobian load mapping')
        candidates={'center_of_mass':jac[:,:3,:],
            'prim_origin':jac[:,:3,:]+np.cross(jac[:,3:,:].transpose(0,2,1),com[:,None,:]).transpose(0,2,1)}
        for name,linear in candidates.items():
            predicted=np.einsum('lci,lc->i',linear,self.mass[self.skip:,None]*np.asarray(gravity))
            self.reference_errors[name]=float(np.linalg.norm(predicted-expected)/np.linalg.norm(expected))
        self.reference=min(self.reference_errors,key=self.reference_errors.get)
        if self.reference_errors[self.reference]>2e-4:
            raise RuntimeError('Native force Jacobian disagrees with native gravity efforts')

    def _jacobian(self):
        a=self.articulation
        return np.asarray(a.get_jacobians(),dtype=float)[0].reshape(len(self.paths)-self.skip,6,-1)

    def at_com(self, path, force):
        from .runtime import pose_matrices
        force=np.asarray(force,dtype=float)
        if force.shape!=(3,) or not np.isfinite(force).all(): raise ValueError('Invalid body force')
        index=self.paths.index(path)
        if index<self.skip: raise ValueError('Cannot load a fixed articulation root')
        jac=self._jacobian()[index-self.skip]
        effort=jac[:3].T@force
        if self.reference=='prim_origin':
            frame=pose_matrices(self.articulation.get_link_transforms()[0])[index]
            lever=frame[:3,:3]@self.local_com[index]
            effort+=jac[3:].T@np.cross(lever,force)
        return effort
