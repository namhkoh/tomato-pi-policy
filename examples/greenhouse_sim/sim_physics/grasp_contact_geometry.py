"""Geometry-only grasp-contact predictor input; not contact or motion authority.

Native row geometry is transported from its explicit generation frame to the
next pre-solve frame. Only cylindrical shaft/inner planar finger faces are
supported. Caps, leaves, knives, new/unknown dynamic contacts and missing patch
anchors fail closed. No normal/friction impulse is read or replayed.
This is a frozen finite-feature approximation, not native PCM/friction parity.
"""
from collections import defaultdict
import numpy as np
from .shaft_grasp import ShaftCapsule,FingerPad,_pose
from .contact_patch_prediction import GENERALIZED_MODEL

POINT_TOL_M=2e-6
NORMAL_TOL=2e-4
MAX_ROWS=256


def _array(x,shape,label):
    raw=np.asarray(x)
    if raw.dtype.kind not in 'fiu':raise ValueError('Numeric '+label+' required')
    a=np.array(raw,dtype=float,copy=True)
    if a.shape!=shape or not np.isfinite(a).all():raise ValueError('Finite '+label+' required')
    return a


def _snapshot(value,target):
    if (value.get('source_target')!=target or type(value.get('step_id')) is not int
            or value['step_id']<0 or value.get('native_com_jacobian_velocity_check_passed') is not True):
        raise ValueError('Source-bound native pre-solve snapshot required')
    paths=value['body_paths'];robot=value['robot_body_paths'];n=len(value['generalized_velocity'])
    if (not 7<=n<=64 or not paths or len(set(paths))!=len(paths)
            or not robot or len(set(robot))!=len(robot) or set(paths)&set(robot)):
        raise ValueError('Exact disjoint body inventories required')
    frames={p:_pose(f) for p,f in zip(paths,value['body_frames_world'],strict=True)}
    robot_frames={p:_pose(f) for p,f in zip(robot,value['robot_body_frames_world'],strict=True)}
    jac=_array(value['body_world_com_jacobians'],(len(paths),6,n),'native Jacobians')
    com=_array(value['native_com_local_poses'],(len(paths),7),'plant COMs')
    rv=_array(value['robot_body_velocities_world'],(len(robot),6),'robot velocities')
    rc=_array(value['robot_com_local_poses'],(len(robot),7),'robot COMs')
    return dict(paths=paths,robot=robot,n=n,frames=frames,robot_frames=robot_frames,
        jac=jac,com=com,rv=rv,rc=rc)


def _cylinder_plane_feature(point,cap,cap_frame,pad,pad_frame,*,label):
    """Validate the supplied point, never project/repair it onto either shape.

    Only the cylindrical side away from its uncertain end rings is supported.
    The pad check is its finite tangential footprint: a signed normal gap is
    allowed, not silently removed. A reported friction anchor not on this same
    supported cylinder/plane feature is unsupported even if native PCM permits
    a pad-side, averaged, stale or off-surface anchor representation.
    """
    local=cap_frame[:3,:3].T@(point-cap_frame[:3,3])
    axial_clearance=float(cap.half_height_m-abs(local[2]))
    radius=float(np.linalg.norm(local[:2]));radial_error=abs(radius-cap.radius_m)
    context=label+' '+cap.collider+' / '+pad.collider
    if axial_clearance<=POINT_TOL_M:
        raise ValueError(context+': unsupported endcap/end-ring feature; axial_clearance_m='+repr(axial_clearance))
    if radius==0 or radial_error>POINT_TOL_M:
        raise ValueError(context+': point off cylindrical side; radial_error_m='+repr(radial_error))
    normal=pad.face_sign*pad_frame[:3,pad.face_axis]
    outward=cap_frame[:3,:2]@(local[:2]/radius)
    normal_error=float(np.linalg.norm(outward+normal))
    if normal_error>NORMAL_TOL:
        raise ValueError(context+': cylindrical side not aligned with inner plane; normal_error='+repr(normal_error))
    axes=[i for i in range(3) if i!=pad.face_axis]
    pad_local=pad_frame[:3,:3].T@(point-pad_frame[:3,3])
    face_excess=float(np.max(abs(pad_local[axes])-pad.half_extents_m[axes]))
    if face_excess>POINT_TOL_M:
        raise ValueError(context+': point outside finite pad face; face_excess_m='+repr(face_excess))
    return dict(radial_error_m=radial_error,axial_clearance_m=axial_clearance,
        side_normal_error=normal_error,finite_face_excess_m=face_excess)


def compile_contacts(chain,pads,*,source_target,all_plant_colliders,reference,current,rows,mu):
    """Return full floating J/g/s and patch tangents; never eliminate root DOFs.

Both snapshots carry actual body poses/velocities, not targets. Caller must
bind raw contact geometry to reference.step_id; no native generation timestamp
is available here. current is exactly that generation step or one step later.
Only a later dynamics caller may condition on an independently verified root
support. Forces present in input rows are deliberately not accessed.
"""
    chain=tuple(chain);pads=tuple(pads);inventory=set(all_plant_colliders)
    if (not chain or len(pads)!=2 or not all(isinstance(s,ShaftCapsule) for s in chain)
            or not all(isinstance(p,FingerPad) for p in pads)
            or len({s.collider for s in chain})!=len(chain)
            or len({p.collider for p in pads})!=2
            or not {s.collider for s in chain}<=inventory):
        raise ValueError('Complete source shaft/pad and plant collider inventory required')
    if isinstance(mu,(bool,np.bool_)) or not np.isscalar(mu) or not np.isfinite(mu) or mu<0:
        raise ValueError('Explicit finite nonnegative friction model required')
    ref=_snapshot(reference,source_target);cur=_snapshot(current,source_target)
    if (current['step_id']-reference['step_id'] not in (0,1)
            or ref['paths']!=cur['paths'] or ref['robot']!=cur['robot'] or ref['n']!=cur['n']):
        raise ValueError('Same inventories and current/one-step generation frame required')
    if not set(cur['paths'])<={s.body for s in chain}:
        raise ValueError('Dynamic body missing from complete source chain')
    shafts={s.collider:s for s in chain};padmap={p.collider:p for p in pads}
    groups=defaultdict(lambda:dict(normal=[],friction=[]));ignored=[]
    for index,row in enumerate(rows):
        if index>=MAX_ROWS:raise ValueError('Native contact row overflow')
        a,b=row['collider0'],row['collider1']
        if a==b or not ({a,b}&inventory):raise ValueError('Unattributed plant contact row')
        # The kinematic support/knife pair does not load a dynamic plant link.
        dynamic=[p for p in (a,b) if any(p.startswith(body+'/') for body in cur['paths'])]
        if not dynamic:
            ignored.append(dict(row=index,colliders=[a,b],reason='outside_dynamic_plant_system'))
            continue
        if len(dynamic)!=1 or dynamic[0] not in shafts:
            raise ValueError('Unsupported dynamic leaf/internal/nonshaft contact')
        cap=shafts[dynamic[0]];other=b if a==cap.collider else a
        if other not in padmap:raise ValueError('Unsupported dynamic shaft contact (knife/scene not modelled)')
        if row['kind'] not in ('normal','friction'):raise ValueError('Explicit contact kind required')
        # Copy geometry keys only. An impulse getter may raise without affecting this compiler.
        clean=dict(collider0=a,collider1=b,point=_array(row['point_world_m'],(3,),'contact point'))
        if row['kind']=='normal':
            clean.update(normal=_array(row['normal_on_0'],(3,),'normal on collider0'),
                separation=float(_array(row['separation_m'],(),'native separation')))
        groups[(cap.collider,other)][row['kind']].append(clean)
    if len(groups)>16:raise ValueError('Contact patch bound exceeded')
    J=[];g=[];speeds=[];indices=[];tangents=[];tangent_speeds=[];geometry=[];friction_geometry=[]
    for (cap_path,pad_path),group in sorted(groups.items()):
        if not 1<=len(group['normal'])<=4 or not 1<=len(group['friction'])<=2:
            raise ValueError('Each observed patch needs 1..4 normals and 1..2 genuine friction anchors')
        for kind in ('normal','friction'):
            points=[tuple(row['point']) for row in group[kind]]
            if len(set(points))!=len(points):raise ValueError('Duplicate native feature/anchor geometry')
        cap=shafts[cap_path];pad=padmap[pad_path]
        if cap.body not in cur['frames'] or pad.body not in cur['robot_frames']:
            raise ValueError('Source contact body missing from native inventory')
        ri=cur['robot'].index(pad.body);bi=cur['paths'].index(cap.body)
        refcap=ref['frames'][cap.body]@cap.local_frame;nowcap=cur['frames'][cap.body]@cap.local_frame
        refpad=ref['robot_frames'][pad.body]@pad.local_frame;nowpad=cur['robot_frames'][pad.body]@pad.local_frame
        n0=pad.face_sign*refpad[:3,pad.face_axis];normal=pad.face_sign*nowpad[:3,pad.face_axis]
        face0=refpad[:3,3]+pad.half_extents_m[pad.face_axis]*n0
        face=nowpad[:3,3]+pad.half_extents_m[pad.face_axis]*normal
        body=cur['frames'][cap.body];plant_com=body[:3,3]+body[:3,:3]@cur['com'][bi,:3]
        rb=cur['robot_frames'][pad.body];pad_com=rb[:3,3]+rb[:3,:3]@cur['rc'][ri,:3]
        tangent_axes=[i for i in range(3) if i!=pad.face_axis];basis=nowpad[:3,tangent_axes].T
        def at(point,directions):
            # velocity at a point = v_COM + omega cross lever.
            jac=directions@cur['jac'][bi,:3]+np.cross(point-plant_com,directions)@cur['jac'][bi,3:]
            surface=cur['rv'][ri,:3]+np.cross(cur['rv'][ri,3:],point-pad_com)
            return jac,directions@surface
        patch_indices=[]
        for row in group['normal']:
            oriented=row['normal']*(1 if row['collider0']==cap_path else -1)
            if np.linalg.norm(oriented-n0)>NORMAL_TOL:raise ValueError('Contact is not the authored inner pad plane')
            reference_check=_cylinder_plane_feature(row['point'],cap,refcap,pad,refpad,label='reference normal')
            axis_local=refcap[:3,:3].T@(row['point']+cap.radius_m*n0-refcap[:3,3])
            if np.linalg.norm(axis_local[:2])>POINT_TOL_M or abs(axis_local[2])>cap.half_height_m+POINT_TOL_M:
                raise ValueError('Normal feature is not on the finite cylindrical shaft')
            refgap=float(np.dot(row['point']-face0,n0))
            if abs(refgap-row['separation'])>POINT_TOL_M:
                raise ValueError('Native gap disagrees with its explicit generation geometry')
            anchor_local=refpad[:3,:3].T@(row['point']-refgap*n0-refpad[:3,3])
            if np.any(abs(anchor_local[tangent_axes])>pad.half_extents_m[tangent_axes]+POINT_TOL_M):
                raise ValueError('Normal seed lies outside finite pad face')
            point=nowcap[:3,:3]@axis_local+nowcap[:3,3]-cap.radius_m*normal
            current_check=_cylinder_plane_feature(point,cap,nowcap,pad,nowpad,label='current normal')
            j,s=at(point,normal[None]);patch_indices.append(len(J));J.append(j[0]);speeds.append(float(s[0]))
            g.append(float(np.dot(point-face,normal)))
            geometry.append(dict(cap=cap_path,pad=pad_path,point_world_m=point.tolist(),normal_on_plant=normal.tolist(),
                native_generation_gap_error_m=abs(refgap-row['separation']),
                reference_geometry=reference_check,current_geometry=current_check))
        T=[];st=[]
        for row in group['friction']:
            reference_check=_cylinder_plane_feature(row['point'],cap,refcap,pad,refpad,label='reference friction')
            owner=cap.body if row['collider0']==cap_path else pad.body
            f0=ref['frames'].get(owner,ref['robot_frames'].get(owner));f1=cur['frames'].get(owner,cur['robot_frames'].get(owner))
            local=f0[:3,:3].T@(row['point']-f0[:3,3]);point=f1[:3,:3]@local+f1[:3,3]
            current_check=_cylinder_plane_feature(point,cap,nowcap,pad,nowpad,label='current friction')
            jt,vt=at(point,basis);T.append(jt.tolist());st.append(vt.tolist())
            friction_geometry.append(dict(cap=cap_path,pad=pad_path,owner_body=owner,
                reference_point_world_m=row['point'].tolist(),point_world_m=point.tolist(),
                reference_geometry=reference_check,current_geometry=current_check))
        indices.append(patch_indices);tangents.append(T);tangent_speeds.append(st)
    n=cur['n'];count=len(J)
    if count>64:raise ValueError('Normal feature bound exceeded')
    patches=dict(model=GENERALIZED_MODEL,normal_indices=indices,tangent_jacobians=tangents,
        surface_speeds_m_s=tangent_speeds,mu=float(mu),observed=[True]*len(indices))
    return np.array(J).reshape(count,n),np.array(g),np.array(speeds),dict(patches=patches,
        feature_observed=[True]*count,features=geometry,friction_features=friction_geometry,
        ignored_outside_dynamic_system=ignored,
        reference_step=reference['step_id'],current_step=current['step_id'],source_target=source_target,
        contact_impulses_read=False,native_contact_law_parity=False,friction_position_bias=False,
        root_columns_retained=6,geometry_generation_binding='caller_asserted',
        feature_scope='cylinder_side_and_finite_inner_plane_only',points_projected=False,
        point_tolerance_m=POINT_TOL_M,normal_tolerance=NORMAL_TOL,training_eligible=False)
