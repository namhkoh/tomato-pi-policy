"""Measured sharp-edge traversal of a cylindrical material-section envelope.

Pure geometry, NOT fracture or release authorization. Intersect the current
blade plane with the native shaft axis, then bound the resulting ellipse in
stroke/edge coordinates. Clearing the sharp edge is distinct from reaching an
arbitrary overshoot waypoint or demanding unloaded blade side faces before
withdrawal. Native force, contact provenance and retention remain mandatory.
"""
import math
import numpy as np


def clearance(edge, centre, axis, *, radius, tip_offset, half_span, margin=.0005):
    edge=np.asarray(edge,float);centre=np.asarray(centre,float);axis=np.asarray(axis,float)
    if (edge.shape!=(4,4) or centre.shape!=(3,) or axis.shape!=(3,)
            or not np.isfinite(np.r_[edge.flat,centre,axis,radius,tip_offset,half_span,margin]).all()
            or not np.allclose(edge[3],[0,0,0,1],rtol=0,atol=1e-8)
            or not np.allclose(edge[:3,:3].T@edge[:3,:3],np.eye(3),rtol=0,atol=1e-5)
            or np.linalg.det(edge[:3,:3])<=0 or abs(np.linalg.norm(axis)-1)>1e-5
            or not .0005<=radius<=.01 or abs(tip_offset-.0005)>1e-8
            or not .012<=half_span<=.018 or margin!=.0005):
        raise ValueError('Actual crossbar strip, rigid native section and bounded metric dimensions required')
    direction=-edge[:3,0];along=edge[:3,1];normal=edge[:3,2];p=edge[:3,3]
    cosine=float(normal@axis)
    if abs(cosine)<.9:raise RuntimeError('Blade plane is not sufficiently transverse to the material section')
    axial=float(normal@(p-centre)/cosine)
    midpoint=centre+axial*axis
    stroke_radius=radius*math.sqrt(1+(float(direction@axis)/cosine)**2)
    along_radius=radius*math.sqrt(1+(float(along@axis)/cosine)**2)
    axial_radius=radius*math.sqrt(max(0.,1/cosine**2-1))
    tip_clearance=float(tip_offset-np.dot(midpoint-p,direction)-stroke_radius)
    end_clearance=float(half_span-abs(np.dot(midpoint-p,along))-along_radius)
    within_seam=abs(axial)+axial_radius<=.003
    return dict(model='native_shaft_blade_plane_ellipse_v1',
        sharp_edge_cleared=bool(tip_clearance>=margin and end_clearance>=margin and within_seam),
        sharp_edge_clearance_m=tip_clearance,blade_end_clearance_m=end_clearance,
        section_plane_centre_world_m=midpoint.tolist(),plane_centre_axial_offset_m=axial,
        maximum_section_axial_distance_m=abs(axial)+axial_radius,
        section_within_original_3mm_seam=bool(within_seam),clearance_margin_m=margin,
        stroke_projected_radius_m=stroke_radius,edge_projected_radius_m=along_radius,
        fracture_verified=False,commanded_motion_used=False)


def cut_face_contact(record, *, knife, faces):
    """Permit bounded POST-release sliding only on the two actual cut faces.

Wrong-face contact never becomes cutting evidence. Any other loaded knife
part/scene pair revokes this sliding permission, including friction-only rows.
"""
    if (record.get('cut') is not True or record.get('native_guards_passed') is not True
            or not isinstance(knife,str) or not knife.endswith('/DeleafKnife/CrossbarContact')
            or len(faces)!=2 or len(set(faces))!=2
            or any(not isinstance(p,str) or not p.endswith('/StemCollider') for p in faces)):
        raise ValueError('Guarded released target and exact crossbar/two-face identity required')
    k=record['knife'];physical=k.get('physical_knife_contact',{})
    rows=k.get('raw_normal_rows');pairs=record.get('native_contact_pairs_n')
    separation=physical.get('minimum_separation_m',-math.inf)
    if (k.get('raw_normal_rows_complete') is not True or not isinstance(rows,list) or not isinstance(pairs,list)
            or not math.isfinite(separation) or separation<-.001):
        raise RuntimeError('Complete guarded physical knife contacts required')
    root=knife.rsplit('/',1)[0];total=0.;expected={frozenset((knife,p)) for p in faces};loaded=set();observed=set()
    for a,b,force in pairs:
        if a.startswith(root+'/') or b.startswith(root+'/'):
            if isinstance(force,bool) or not math.isfinite(force) or force<0:
                raise RuntimeError('Invalid full knife contact load')
            if force>0 and frozenset((a,b)) not in expected:return False
            if force>.01:loaded.add(frozenset((a,b)))
            total+=force
    upper=k['tool_contact_upper_bound_n']
    if not math.isfinite(upper) or not 0<=total<=upper+1e-6 or upper-total>1e-6:
        raise RuntimeError('Missing full knife pair provenance for post-release sliding')
    for r in rows:
        if r['collider0'].startswith(root+'/') or r['collider1'].startswith(root+'/'):
            if (not math.isfinite(r['separation']) or r['separation']<-.001
                    or frozenset((r['collider0'],r['collider1'])) not in expected):return False
            observed.add(frozenset((r['collider0'],r['collider1'])))
    return loaded<=observed
