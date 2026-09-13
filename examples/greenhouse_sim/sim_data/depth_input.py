"""Optional RGB-D input variant: native Isaac optical-Z rendered as a second image.

This is an explicit experiment (2026-09-13) outside the RGB-only task-v3 contract:
the model additionally sees the SAME view's depth, rendered deterministically from
the release's byte-preserved float32 optical-axis Z and validity mask. Nothing is
estimated, no hidden cut XYZ or masks enter the model, and the answer contract is
unchanged. A deployed model would need live depth from the head camera.
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image

DEPTH_INPUT='native_optical_z_inverse_turbo_0.25_1.5m.v1'
NEAR_M,FAR_M=.25,1.5
DEPTH_NOTE=('The second image is the depth map of the same view rendered from the head camera: '
    'turbo colormap of inverse depth between 0.25 m (red) and 1.5 m (blue), black = invalid depth. '
    'A cut region hidden behind a nearer leaf, fruit or stem shows as a depth discontinuity.')


def turbo(u):
    """Google turbo colormap polynomial approximation; u in [0,1] -> uint8 RGB."""
    u=np.clip(np.asarray(u,dtype=np.float64),0.,1.)
    r=0.13572138+4.61539260*u-42.66032258*u**2+132.13108234*u**3-152.94239396*u**4+59.28637943*u**5
    g=0.09140261+2.19418839*u+4.84296658*u**2-14.18503333*u**3+4.27729857*u**4+2.82956604*u**5
    b=0.10667330+12.64194608*u-60.58204836*u**2+110.36276771*u**3-89.90310912*u**4+27.34824973*u**5
    return (np.clip(np.stack([r,g,b],axis=-1),0.,1.)*255.+.5).astype(np.uint8)


def depth_files(root,row_id):
    root=Path(root).resolve();depth=root/'depth'/f'{row_id}.npy';valid=root/'depth'/f'{row_id}_valid.png'
    if not depth.is_file() or not valid.is_file(): raise ValueError('Native depth sidecar missing for '+row_id)
    return depth,valid


def render_depth(z,valid):
    z=np.asarray(z,dtype=np.float32);valid=np.asarray(valid)==255
    if z.shape!=(408,848) or valid.shape!=z.shape: raise ValueError('Expected 408x848 native depth and validity')
    with np.errstate(divide='ignore',invalid='ignore'):
        inverse=(1./z-1./FAR_M)/(1./NEAR_M-1./FAR_M)
    rgb=turbo(np.where(valid,inverse,0.))
    rgb[~valid]=0
    return Image.fromarray(rgb,'RGB')


def depth_image(root,row_id):
    depth,valid=depth_files(root,row_id)
    return render_depth(np.load(depth,allow_pickle=False),np.asarray(Image.open(valid)))


def query_depth_m(root,row_id,query_uv):
    depth,valid=depth_files(root,row_id)
    z=np.load(depth,allow_pickle=False,mmap_mode='r');v=np.asarray(Image.open(valid))==255
    x,y=np.floor(query_uv).astype(int)
    if not (0<=x<848 and 0<=y<408): raise ValueError('Query outside image')
    return float(z[y,x]) if v[y,x] else None


def depth_sentence(query_depth):
    where='is invalid' if query_depth is None else f'is {query_depth:.2f} m'
    return DEPTH_NOTE+f' The depth at the query pixel {where}.'
