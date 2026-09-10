"""SI-unit beam approximation and physical-state visual binding.

Material values are engineering priors, not measured tomato calibration.
"""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class Material:
    youngs_modulus_pa: float = 1.5e8
    density_kg_m3: float = 950.0
    damping_ratio: float = .7
    poisson_ratio: float = .3
    leaf_mass_kg: float = .0005

    def __post_init__(self):
        if not all(math.isfinite(v) and v>0 for v in (
                self.youngs_modulus_pa,self.density_kg_m3,self.damping_ratio,self.leaf_mass_kg)):
            raise ValueError('Finite positive material parameters required')
        if not math.isfinite(self.poisson_ratio) or not -1<self.poisson_ratio<.5:
            raise ValueError('Invalid Poisson ratio')


def beam_properties(radius,length,material,*,supported_inertia=0.0,extra_mass=0.0):
    if not all(math.isfinite(v) and v>0 for v in (radius,length)):
        raise ValueError('Finite positive radius and length required')
    if not all(math.isfinite(v) and v>=0 for v in (supported_inertia,extra_mass)):
        raise ValueError('Invalid supported inertia or mass')
    mass=material.density_kg_m3*math.pi*radius**2*length+extra_mass
    inertia=np.array([mass*(3*radius**2+length**2)/12]*2+[mass*radius**2/2])
    bend=material.youngs_modulus_pa*math.pi*radius**4/(4*length)
    torsion=material.youngs_modulus_pa/(2*(1+material.poisson_ratio))*math.pi*radius**4/(2*length)
    stiffness=np.array([bend,bend,torsion])
    effective=inertia+mass*(length/2)**2+supported_inertia
    damping=2*material.damping_ratio*np.sqrt(stiffness*effective)
    return dict(mass=mass,inertia=inertia,stiffness=stiffness,damping=damping,
                usd_stiffness=stiffness*math.pi/180,usd_damping=damping*math.pi/180)


def resample_chain(chain,*,cut_m=.01,max_segment_m=.025):
    points=np.asarray(chain,dtype=float)
    if points.ndim!=2 or points.shape[1]!=4 or len(points)<2 or not np.isfinite(points).all() or np.any(points[:,3]<=0):
        raise ValueError('Expected finite XYZ-radius centerline')
    spans=np.linalg.norm(np.diff(points[:,:3],axis=0),axis=1)
    if np.any(spans<=1e-8): raise ValueError('Degenerate centerline')
    arcs=np.r_[0,np.cumsum(spans)]
    if not (math.isfinite(cut_m) and 0<cut_m<arcs[-1] and math.isfinite(max_segment_m) and max_segment_m>0):
        raise ValueError('Cut must be inside petiole and segment size positive')
    knots=np.unique(np.r_[arcs,cut_m])
    samples=[0.0]
    for low,high in zip(knots[:-1],knots[1:]):
        samples.extend(np.linspace(low,high,int(np.ceil((high-low)/max_segment_m))+1)[1:])
    samples=np.asarray(samples)
    return np.column_stack([np.interp(samples,arcs,points[:,i]) for i in range(4)]),samples


def lamina_mass_properties(points,triangles,mass):
    """Uniform surface-density triangle integral, in the carrier body frame."""
    vertices=np.asarray(points,dtype=float)[np.asarray(triangles,dtype=int)]
    area=np.linalg.norm(np.cross(vertices[:,1]-vertices[:,0],vertices[:,2]-vertices[:,0]),axis=1)/2
    if not math.isfinite(mass) or mass<=0 or not np.isfinite(vertices).all() or area.sum()<=1e-12:
        raise ValueError('Finite, nondegenerate lamina and positive mass required')
    weights=area/area.sum()
    sums=vertices.sum(axis=1)
    center=np.sum(weights[:,None]*sums/3,axis=0)
    second=(np.einsum('nvi,nvj->nij',vertices,vertices)+np.einsum('ni,nj->nij',sums,sums))/12
    covariance=np.sum(weights[:,None,None]*second,axis=0)-np.outer(center,center)
    return mass,center,mass*(np.trace(covariance)*np.eye(3)-covariance)


def combine_mass_properties(parts):
    """Combine SI mass, COM and COM inertia tensors; no numerical mass floor."""
    mass=sum(p[0] for p in parts)
    center=sum(p[0]*np.asarray(p[1]) for p in parts)/mass
    inertia=np.zeros((3,3))
    for amount,position,tensor in parts:
        d=np.asarray(position)-center
        inertia+=tensor+amount*(np.dot(d,d)*np.eye(3)-np.outer(d,d))
    values,axes=np.linalg.eigh(inertia)
    if np.any(values<=0): raise ValueError('Nonpositive physical inertia')
    if np.linalg.det(axes)<0: axes[:,0]*=-1
    return dict(mass=mass,center=center,inertia=values,principal_axes=axes)


def segment_frames(points):
    points=np.asarray(points,dtype=float); frames=[]
    for a,b in zip(points[:-1],points[1:]):
        axis=b-a; axis/=np.linalg.norm(axis)
        helper=np.array([1.,0.,0.]) if abs(axis[0])<.9 else np.array([0.,1.,0.])
        x=np.cross(helper,axis);x/=np.linalg.norm(x);y=np.cross(axis,x)
        matrix=np.eye(4);matrix[:3,:3]=np.column_stack([x,y,axis]);matrix[:3,3]=(a+b)/2
        frames.append(matrix)
    return np.asarray(frames)


def nearest_segments(points,chain):
    points=np.asarray(points,dtype=float);chain=np.asarray(chain,dtype=float)
    a,b=chain[:-1],chain[1:];delta=b-a
    t=np.clip(np.einsum('nsi,si->ns',points[:,None,:]-a,delta)/np.sum(delta*delta,axis=1),0,1)
    distance=np.linalg.norm(points[:,None,:]-(a[None,:,:]+t[:,:,None]*delta),axis=2)
    return np.argmin(distance,axis=1),t


class SkinBinding:
    """Precomputed nearest-frame blend, with no weights across a cut interface."""
    def __init__(self,points,rest_frames,chain,*,first_segment=0,last_segment=None):
        self.points=np.asarray(points,dtype=float);frames=np.asarray(rest_frames,dtype=float)
        end=len(frames) if last_segment is None else last_segment+1
        if not 0<=first_segment<end<=len(frames): raise ValueError('Invalid skin segment interval')
        selected,t=nearest_segments(self.points,np.asarray(chain)[first_segment:end+1])
        selected+=first_segment
        fraction=t[np.arange(len(t)),selected-first_segment]
        self.other=np.clip(np.where(fraction<.5,selected-1,selected+1),first_segment,end-1)
        self.indices=selected;self.weight=np.abs(fraction-.5);self.inverse=np.linalg.inv(frames)

    def deform(self,frames):
        transform=np.asarray(frames)@self.inverse
        homogeneous=np.c_[self.points,np.ones(len(self.points))]
        first=np.einsum('nij,nj->ni',transform[self.indices],homogeneous)[:,:3]
        second=np.einsum('nij,nj->ni',transform[self.other],homogeneous)[:,:3]
        return first*(1-self.weight[:,None])+second*self.weight[:,None]


def clip_mesh(points,counts,indices,origin,normal,*,positive,uv=None):
    """Clip polygons at the seam, retaining face-varying UVs; no wound model."""
    points=np.asarray(points,dtype=float);indices=np.asarray(indices,dtype=int)
    out=[];out_uv=[];cursor=0
    for count in counts:
        face=points[indices[cursor:cursor+count]]
        attrs=None if uv is None else np.asarray(uv[cursor:cursor+count])
        polygon=[(p,None if attrs is None else attrs[i]) for i,p in enumerate(face)]
        clipped=[]
        for (a,ta),(b,tb) in zip(polygon,polygon[1:]+polygon[:1]):
            da=np.dot(a-origin,normal)*(1 if positive else -1)
            db=np.dot(b-origin,normal)*(1 if positive else -1)
            inside_a=da>=-1e-10;inside_b=db>=-1e-10
            if inside_a: clipped.append((a,ta))
            if inside_a != inside_b:
                blend=da/(da-db)
                clipped.append((a+blend*(b-a),None if ta is None else ta+blend*(tb-ta)))
        for i in range(1,len(clipped)-1):
            triangle=[clipped[0],clipped[i],clipped[i+1]]
            xyz=np.asarray([p for p,_ in triangle])
            if np.linalg.norm(np.cross(xyz[1]-xyz[0],xyz[2]-xyz[0]))<1e-14: continue
            out.extend(xyz)
            if uv is not None: out_uv.extend(t for _,t in triangle)
        cursor+=count
    return np.asarray(out).reshape(-1,3),None if uv is None else np.asarray(out_uv).reshape(-1,2)
