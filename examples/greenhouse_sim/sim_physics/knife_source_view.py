"""Orthographic audit of original knife components and existing semantic rim.

Colors are current CODE labels, not a certification of the physical blade.
Only a diagnostic PNG is written; source assets and simulation are untouched.
"""
import argparse
from pathlib import Path
import numpy as np


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--crossbar',action='store_true')
    a=p.parse_args(argv)
    if a.output.exists():raise FileExistsError(a.output)
    from pxr import Usd,UsdGeom
    from greenhouse_sim.robot_model import DEFAULT_ASSET
    from .blade_contacts import refine_blade_contacts,LOWER_EDGE
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    stage=Usd.Stage.CreateInMemory()
    UsdGeom.Xform.Define(stage,'/Robot').GetPrim().GetReferences().AddReference(str(DEFAULT_ASSET))
    stage.SetEditTarget(stage.GetSessionLayer())
    profile=refine_blade_contacts(stage,'/Robot',edge_mode=LOWER_EDGE)
    if a.crossbar:
        from .arc_contacts import refine_arc_contacts
        profile=refine_arc_contacts(stage,'/Robot',crossbar_edge=True)
    root='/Robot/ee_right/attachments/DeleafKnife';cache=UsdGeom.XformCache()
    inverse=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(stage.GetPrimAtPath(root))).T)
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    for ax,(i,j) in zip(axes,((1,2),(1,0),(0,2)),strict=True):
        for name,color in [('Blade','#438bd3'),('Arc','#cf892e')]:
            mesh=UsdGeom.Mesh.Get(stage,root+'/'+name)
            transform=inverse@np.asarray(cache.GetLocalToWorldTransform(mesh.GetPrim())).T
            points=np.asarray(mesh.GetPointsAttr().Get())@transform[:3,:3].T+transform[:3,3]
            counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get())
            if not np.all(counts==3):raise ValueError('Triangular source mesh required')
            faces=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
            projected=points[:,[i,j]]*1000
            ax.add_collection(PolyCollection(projected[faces],facecolor=color,edgecolor='none',alpha=.6,label=name+' (code label)'))
            ax.update_datalim(projected)
        frame=np.array(profile['edge_frame']);size=np.array(profile['edge_size_m'])
        endpoints=frame[:3,3]+np.array([-.5,.5])[:,None]*size[1]*frame[:3,1]
        ax.plot(endpoints[:,i]*1000,endpoints[:,j]*1000,'m-',lw=3,
            label='Corrected crossbar cutting edge' if a.crossbar else 'Old mounting-plate semantic strip')
        ax.autoscale_view();ax.set_aspect('equal');ax.grid(alpha=.2)
        ax.set_xlabel('source '+'XYZ'[i]+' (mm)');ax.set_ylabel('source '+'XYZ'[j]+' (mm)')
    axes[0].legend(loc='upper left',fontsize=8)
    fig.suptitle('Unmodified knife geometry: component labels vs modeled cutting edge')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=120);plt.close(fig)


if __name__=='__main__':main()
