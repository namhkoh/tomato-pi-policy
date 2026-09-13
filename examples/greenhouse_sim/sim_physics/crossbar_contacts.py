"""Actual sharpened straight bar inside the legacy 'Arc' source component.

The widest-X component called 'Blade' by the old importer is the mounting
plate. The cutting bar is the narrow, beveled lower crossbar of the curved
component. Extract source triangles only: no invented bevel or changed mesh.
"""
import numpy as np


def geometry(points,counts,indices):
    from .arc_contacts import partitions
    from .blade_contacts import section_points
    from .mechanics import clip_mesh
    points=np.asarray(points,float);counts=np.asarray(counts,int);indices=np.asarray(indices,int)
    partitions(points,counts,indices)  # Existing source size/shape validation.
    if not np.all(counts==3):raise ValueError('Original triangular tool required')
    low,high=points.min(0),points.max(0)
    # Stay clear of both curved feet; all excluded source surfaces remain
    # collidable support pieces. The upper open window starts above 20 mm.
    lo,hi=low[1]+.015,high[1]-.015
    sections=[section_points(points[indices.reshape(-1,3)],y) for y in np.linspace(lo,hi,7)]
    lower=[s[s[:,2]<.020] for s in sections]
    if any(not len(s) for s in lower):raise ValueError('Missing lower crossbar')
    tips=np.array([s[np.argmin(s[:,2])] for s in lower])
    slope,intercept=np.polyfit(tips[:,1],tips[:,2],1)
    if (abs(slope)>.05 or np.max(abs(tips[:,2]-(slope*tips[:,1]+intercept)))>.00001
            or np.max(abs(tips[:,0]))>.0001
            or any(not .0029<=np.ptp(s[:,0])<=.0031 for s in lower)
            or any(not .014<=np.ptp(s[:,2])<=.015 for s in lower)):
        raise ValueError('Unverified straight beveled cutting bar; do not guess an edge')
    # Actual source tip: near X=0, not the +/-3 mm curved support or plate.
    up=np.array([0.,-slope,1.]);up/=np.linalg.norm(up)
    along=np.array([0.,1.,slope]);along/=np.linalg.norm(along)
    edge=np.eye(4);edge[:3,:3]=np.column_stack([up,along,np.cross(up,along)])
    mid=(lo+hi)/2;edge[:3,3]=[0.,mid,slope*mid+intercept]
    edge[:3,3]+=.0005*up
    size=np.array([.001,(hi-lo-.004)*np.sqrt(1+slope*slope),.001])
    # Partition every source triangle. Combine only the straight central
    # crossbar into one convex carrier; curved/support portions remain split.
    edges=np.unique(np.r_[np.linspace(low[1]-.000001,high[1]+.000001,8),lo,hi])
    support=[];bar=[]
    for a,b in zip(edges[:-1],edges[1:]):
        for lower_z,upper_z in ((low[2]-.000001,.020),(.020,high[2]+.000001)):
            vertices=points;cs=counts;ids=indices
            for origin,normal,positive in (([0,a,0],[0,1,0],True),([0,b,0],[0,1,0],False),
                    ([0,0,lower_z],[0,0,1],True),([0,0,upper_z],[0,0,1],False)):
                vertices,_=clip_mesh(vertices,cs,ids,origin,normal,positive=positive)
                if not len(vertices):break
                cs=np.full(len(vertices)//3,3,dtype=int);ids=np.arange(len(vertices))
            if len(vertices):
                (bar if a>=lo and b<=hi and upper_z==.020 else support).append(vertices)
    if not bar or not support or len(support)>18:raise ValueError('Invalid source crossbar partition')
    return dict(support=support,bar=np.concatenate(bar),edge_frame=edge,edge_size_m=size,
        source_bar_y_range_m=[float(lo),float(hi)],source_edge_slope=float(slope),
        legacy_component='Arc',physical_part='straight_beveled_lower_crossbar',
        edge_from_source_bevel=True,sharpness_calibrated=False)
