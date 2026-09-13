"""Bounded nominal wrist schedules; no IK, contacts or execution permission."""
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ABOVE_OFFSETS={'orient_via_above_30mm':.03,'orient_via_above_60mm':.06,
               'orient_via_above_90mm':.09}
SIDE_OFFSETS={'orient_via_side_plus_60mm':.06,'orient_via_side_minus_60mm':-.06,
              'orient_via_side_plus_120mm':.12,'orient_via_side_minus_120mm':-.12}
MODES=('simultaneous','orient_then_translate',*SIDE_OFFSETS,*ABOVE_OFFSETS)


def frames(start,end,*,mode='simultaneous'):
    a,b=np.array(start,float,copy=True),np.array(end,float,copy=True)
    if mode not in MODES:raise ValueError('Unknown bounded wrist transit mode')
    for f in (a,b):
        if (f.shape!=(4,4) or not np.isfinite(f).all()
                or not np.allclose(f[3],[0,0,0,1],atol=1e-10,rtol=0)
                or not np.allclose(f[:3,:3].T@f[:3,:3],np.eye(3),atol=1e-6,rtol=0)
                or np.linalg.det(f[:3,:3])<0):
            raise ValueError('Proper rigid wrist endpoints required')
    distance=float(np.linalg.norm(b[:3,3]-a[:3,3]))
    if distance>2:raise ValueError('Wrist transit exceeds bounded robot workspace')
    if mode in ABOVE_OFFSETS or mode in SIDE_OFFSETS:
        waypoint=b.copy()
        if mode in ABOVE_OFFSETS:waypoint[2,3]+=ABOVE_OFFSETS[mode]
        else:
            side=np.cross([0.,0.,1.],b[:3,3]-a[:3,3]);length=float(np.linalg.norm(side))
            if length<1e-12:
                # Explicit fixed axis for a vertical/zero translation. No
                # anatomical or collision inference is made by this proposal.
                side=np.array([1.,0.,0.])
            else:side/=length
            waypoint[:3,3]=(a[:3,3]+b[:3,3])/2+SIDE_OFFSETS[mode]*side
        first=frames(a,waypoint,mode='orient_then_translate')
        last=frames(waypoint,b,mode='simultaneous')
        return np.concatenate((first,last[1:]))
    rotations=Rotation.from_matrix(np.array([a[:3,:3],b[:3,:3]]))
    angle=float(np.linalg.norm((rotations[1]*rotations[0].inv()).as_rotvec()))
    nt=max(2,int(np.ceil(distance/.002))+1)
    nr=max(2,int(np.ceil(np.degrees(angle)))+1)
    def make(translation,rotation):
        out=np.repeat(np.eye(4)[None],len(translation),axis=0)
        out[:,:3,3]=a[:3,3]+translation[:,None]*(b[:3,3]-a[:3,3])
        out[:,:3,:3]=Slerp([0,1],rotations)(rotation).as_matrix()
        return out
    if mode=='simultaneous':
        alpha=np.linspace(0,1,max(nt,nr));out=make(alpha,alpha)
    elif angle<1e-12:
        alpha=np.linspace(0,1,nt);out=make(alpha,np.ones(nt))
    elif distance<1e-12:
        alpha=np.linspace(0,1,nr);out=make(np.zeros(nr),alpha)
    else:
        rotate=make(np.zeros(nr),np.linspace(0,1,nr))
        translate=make(np.linspace(0,1,nt),np.ones(nt))
        out=np.concatenate((rotate,translate[1:]))
    out[0]=a;out[-1]=b
    return out


def screen_modes(screen,start,end):
    """Elbow redundancy cannot repair the SAME nominal rigid-tool sweep.

    Test both explicit schedules once per requested wrist pose before arm IK.
    This only rejects/ranks proposals; full actual IK-path/native checks remain.
    A native query error propagates; there is no timeout-to-clear fallback.
    """
    allowed=[];evidence={}
    for mode in MODES:
        result=screen.check(frames(start,end,mode=mode),stroke=False)
        evidence[mode]=result
        if result['passed']:allowed.append(mode)
    return tuple(allowed),evidence
