"""Independent, read-only mass/Jacobian/body kinetic-energy crosscheck.

Installed tensor API: get_inertias is COM inertia expressed in body-prim
axes, column major; get_link_velocities is world COM linear/angular velocity.
Never rotates inertia by the COM principal-axes quaternion a second time.
No effort, filtering, state repair or execution approval.
"""
import numpy as np


def check(snapshot,masses,inertias):
    j=np.asarray(snapshot['body_world_com_jacobians'],float)
    frames=np.asarray(snapshot['body_frames_world'],float)
    velocity=np.asarray(snapshot['body_velocities_world'],float)
    v=np.asarray(snapshot['generalized_velocity'],float)
    m=np.asarray(snapshot['mass_matrix'],float)
    mass=np.asarray(masses,float).reshape(-1)
    inertia=np.asarray(inertias,float)
    b=len(mass);n=len(v)
    if (not b or not n or j.shape!=(b,6,n) or frames.shape!=(b,4,4)
            or velocity.shape!=(b,6) or m.shape!=(n,n) or inertia.size!=b*9
            or any(not np.isfinite(x).all() for x in (j,frames,velocity,v,m,mass,inertia))
            or np.any(mass<=0)):
        raise ValueError('Finite exact native kinetic-energy inventory required')
    local=inertia.reshape(b,3,3).transpose(0,2,1)
    rotation=frames[:,:3,:3]
    world=rotation@local@rotation.transpose(0,2,1)
    derived=(np.einsum('bci,bcj,b->ij',j[:,:3],j[:,:3],mass)
        +np.einsum('bci,bcd,bdj->ij',j[:,3:],world,j[:,3:]))
    if np.any(np.diag(m)<=0):raise ValueError('Positive native kinetic mass diagonal required')
    scale=np.sqrt(np.outer(np.diag(m),np.diag(m)))
    generalized=.5*float(v@m@v)
    body=.5*float(np.sum(mass*np.sum(velocity[:,:3]**2,axis=1))
        +np.einsum('bi,bij,bj->',velocity[:,3:],world,velocity[:,3:]))
    return dict(native_generalized_kinetic_j=generalized,native_body_kinetic_j=body,
        kinetic_difference_j=generalized-body,
        mass_max_diagonal_scaled_difference=float(np.max(abs(m-derived)/scale)),
        mass_relative_frobenius_difference=float(np.linalg.norm(m-derived)/np.linalg.norm(m)),
        jacobian_velocity_max_error=float(np.max(abs(np.einsum('bcj,j->bc',j,v)-velocity))),
        body_count=b,generalized_coordinate_count=n,read_only=True,
        native_effort_measured=False,physical_model_qualified=False)
