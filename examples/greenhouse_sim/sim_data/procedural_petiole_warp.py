"""Continuous source-to-new-curve deformation with explicit geometry diagnostics.

Preserves source mesh topology/UV detail while changing centerline shape and
attachment location. Derivatives transform normals; no donor dynamics survive.
"""
import numpy as np
from .procedural_petiole_geometry import require, unit, transport_frames


class CurveWarp:
    def __init__(self, old_points, new_curve, new_anchor, radius_scale=1.):
        self.old = np.asarray(old_points, float)
        require(self.old.ndim == 2 and self.old.shape[1] == 3 and len(self.old) >= 2
                and np.isfinite(self.old).all(), 'Invalid donor curve')
        self.delta = np.diff(self.old, axis=0)
        self.segment_length = np.linalg.norm(self.delta, axis=1)
        require((self.segment_length > 1e-9).all(), 'Degenerate donor chain')
        self.arc = np.r_[0., np.cumsum(self.segment_length)]
        self.old_frames = transport_frames(self.old)
        self.new = np.asarray(new_curve['points'], float)
        self.new_arc = np.asarray(new_curve['arc'], float)
        self.new_frames = transport_frames(self.new)
        self.anchor = np.asarray(new_anchor, float)
        require(self.anchor.shape == (3,) and np.isfinite(self.anchor).all()
                and np.isfinite(radius_scale) and .65 <= radius_scale <= 1.5, 'Invalid anchor/radius scale')
        require(np.allclose(self.new[0],0,atol=1e-12) and self.new_arc[-1] > 0, 'New curve must start at local zero')
        self.radial_scale = radius_scale
        self.length_scale = self.new_arc[-1]/self.arc[-1]
        require(.6 <= self.length_scale <= 1.5, 'Excessive longitudinal scaling')

    @staticmethod
    def interpolated_frames(frames, indices, weight):
        mixed = (1-weight[:,None,None])*frames[indices]+weight[:,None,None]*frames[indices+1]
        t = mixed[:,:,0]
        t /= np.linalg.norm(t,axis=1)[:,None]
        n = mixed[:,:,1]-t*np.einsum('ij,ij->i',t,mixed[:,:,1])[:,None]
        n /= np.linalg.norm(n,axis=1)[:,None]
        return np.stack((t,n,np.cross(t,n)),axis=2)

    def map(self, points, chunk_size=1024):
        p = np.asarray(points,float)
        require(p.ndim == 2 and p.shape[1] == 3 and np.isfinite(p).all(), 'Invalid input points')
        outputs=[]
        for first in range(0,len(p),chunk_size):
            block=p[first:first+chunk_size]
            offset=block[:,None,:]-self.old[None,:-1,:]
            fractions=np.clip(np.einsum('nsi,si->ns',offset,self.delta)/self.segment_length[None,:]**2,0,1)
            candidates=self.old[None,:-1,:]+fractions[:,:,None]*self.delta[None,:,:]
            distances=np.linalg.norm(block[:,None,:]-candidates,axis=2)
            index=np.argmin(distances,axis=1)
            rows=np.arange(len(block));fraction=fractions[rows,index]
            center=candidates[rows,index]
            s=self.arc[index]+fraction*self.segment_length[index]
            frame=self.interpolated_frames(self.old_frames,index,fraction)
            new_s=s*self.length_scale
            new_index=np.clip(np.searchsorted(self.new_arc,new_s,side='right')-1,0,len(self.new_arc)-2)
            w=(new_s-self.new_arc[new_index])/(self.new_arc[new_index+1]-self.new_arc[new_index])
            new_center=(1-w[:,None])*self.new[new_index]+w[:,None]*self.new[new_index+1]
            new_frame=self.interpolated_frames(self.new_frames,new_index,w)
            local=np.einsum('nij,ni->nj',frame,block-center)
            local*=np.array([self.length_scale,self.radial_scale,self.radial_scale])
            outputs.append(self.anchor+new_center+np.einsum('nij,nj->ni',new_frame,local))
        return np.concatenate(outputs) if outputs else np.empty((0,3))

    def jacobian(self, points, h=1e-5):
        p=np.asarray(points,float)
        require(1e-7 <= h <= 1e-4,'Derivative step outside qualified range')
        columns=[]
        for axis in np.eye(3):
            columns.append((self.map(p+h*axis)-self.map(p-h*axis))/(2*h))
        result=np.stack(columns,axis=2)
        require(np.isfinite(result).all(),'Nonfinite warp derivative')
        return result

    def normals(self, points, normals):
        p,n=np.asarray(points,float),np.asarray(normals,float)
        require(n.shape==p.shape and np.isfinite(n).all(),'Invalid normal samples')
        j=self.jacobian(p)
        determinant=np.linalg.det(j)
        condition=np.linalg.cond(j)
        require((determinant>.05).all() and (condition<30).all(),
                'Warp folded or ill-conditioned; reject this generated geometry')
        mapped=np.linalg.solve(np.swapaxes(j,1,2),n[:,:,None])[:,:,0]
        lengths=np.linalg.norm(mapped,axis=1)
        nonzero=lengths>1e-12
        mapped[nonzero]/=lengths[nonzero,None]
        mapped[~nonzero]=0
        return mapped,dict(minimum_jacobian_determinant=float(determinant.min()),
            maximum_jacobian_condition=float(condition.max()),preserved_zero_normals=int((~nonzero).sum()),
            global_injectivity_proven=False)

    def direction(self, at, direction):
        return unit(self.jacobian(np.asarray(at,float)[None,:])[0]@unit(direction))
