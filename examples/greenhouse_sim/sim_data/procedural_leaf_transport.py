"""Rigid leaf-blade transport attached to a curved petiole (static data only).

V1 continuous-warp receipts are immutable. This policy preserves each detailed
leaf shape, transporting its attachment and orientation without bending distant
leaf vertices through a petiole's projection field. No physics/novelty approval.
"""
import numpy as np
from .procedural_petiole_geometry import require, unit


class RigidLeafTransport:
    radial_scale = 1.

    def __init__(self, warp, attachment):
        self.old_anchor = np.asarray(attachment, float)
        require(self.old_anchor.shape == (3,) and np.isfinite(self.old_anchor).all(),
                'Invalid leaf attachment')
        # The anchor must still have a locally regular mapping; this is NOT a
        # relaxation of the petiole Jacobian guard.
        derivative = warp.jacobian(self.old_anchor[None, :])[0]
        require(np.linalg.det(derivative) > .05 and np.linalg.cond(derivative) < 30,
                'Leaf attachment warp folded or ill-conditioned')
        offset = self.old_anchor-warp.old[:-1]
        fraction = np.clip(np.einsum('si,si->s', offset, warp.delta)/warp.segment_length**2, 0, 1)
        centers = warp.old[:-1]+fraction[:, None]*warp.delta
        index = int(np.argmin(np.linalg.norm(self.old_anchor-centers, axis=1)))
        weight = float(fraction[index])
        old_frame = warp.interpolated_frames(warp.old_frames, np.array([index]), np.array([weight]))[0]
        s = (warp.arc[index]+weight*warp.segment_length[index])*warp.length_scale
        new_index = int(np.clip(np.searchsorted(warp.new_arc,s,side='right')-1,0,len(warp.new_arc)-2))
        w = (s-warp.new_arc[new_index])/(warp.new_arc[new_index+1]-warp.new_arc[new_index])
        new_frame = warp.interpolated_frames(warp.new_frames,np.array([new_index]),np.array([w]))[0]
        self.rotation = new_frame@old_frame.T
        require(np.allclose(self.rotation.T@self.rotation,np.eye(3),atol=1e-10)
                and abs(np.linalg.det(self.rotation)-1) < 1e-10,'Improper leaf rotation')
        self.new_anchor = warp.map(self.old_anchor[None,:])[0]
        self.source_arc_m = float(s/warp.length_scale)

    def map(self, points, chunk_size=1024):
        p = np.asarray(points,float)
        require(p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all(),'Invalid leaf vertices')
        return self.new_anchor+(p-self.old_anchor)@self.rotation.T

    def jacobian(self, points, h=1e-5):
        p = np.asarray(points,float)
        require(p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all(),'Invalid leaf sample')
        return np.broadcast_to(self.rotation,(len(p),3,3)).copy()

    def normals(self, points, normals):
        p,n = np.asarray(points,float),np.asarray(normals,float)
        require(p.ndim == 2 and p.shape[1] == 3 and n.shape == p.shape
                and np.isfinite(p).all() and np.isfinite(n).all(),'Invalid leaf normal samples')
        mapped = n@self.rotation.T
        lengths = np.linalg.norm(mapped,axis=1)
        nonzero = lengths > 1e-12
        mapped[nonzero] /= lengths[nonzero,None]
        mapped[~nonzero] = 0
        return mapped,dict(minimum_jacobian_determinant=float(np.linalg.det(self.rotation)),
            maximum_jacobian_condition=float(np.linalg.cond(self.rotation)),
            preserved_zero_normals=int((~nonzero).sum()),global_injectivity_proven=True,
            injectivity_scope='this_leaf_rigid_map_only_not_plant_self_intersection',
            transport='rigid_leaf_blade_at_curved_attachment',
            source_arc_m=self.source_arc_m,leaf_shape_preserved=True)

    def direction(self, at, direction):
        return unit(self.rotation@unit(direction))


def component_transport(raw, change):
    if raw['id'] == change['component_id']:
        return change['warp']
    require(raw['type'] == 'leaf' and raw['parent'] == change['component_id'],
            'Rigid leaf transport requires a direct target leaf; nested branches unsupported')
    return RigidLeafTransport(change['warp'],raw['attach_point'])
