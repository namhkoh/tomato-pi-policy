"""Original fitted knife geometry and a measured-contact seam-failure model.

This is NOT calibrated tissue fracture. A force-qualified transverse edge load
breaks one preauthored admissible seam (10 mm default); mesh penetration is not called cutting work.
No timer, commanded velocity, broad blade face or U-support can trigger release.
"""
from dataclasses import dataclass
import math

import numpy as np

# Proposal envelope only. sin(15 degrees)=0.259 < the UNCHANGED measured
# abs(edge-axis dot stem-axis)<0.3 gate. Default search remains 0,+/-10.
MAXIMUM_PLANE_TILT_DEGREES = 15.

LEGACY_CUT_MODEL='force_qualified_pre_authored_seam_release'
BRITTLE_CUT_MODEL='signed_edge_load_brittle_seam_v1'
CUT_MODELS=(LEGACY_CUT_MODEL,BRITTLE_CUT_MODEL)
KNIFE_IMPULSE_CONTRACT='world_normal_impulses_on_knife_v1'


def knife_normal_impulse(normal,impulse):
    """Validate a normal-only row without clipping its signed scalar.

    Same unit-normal/collinearity tolerances as the shaft normal-contact
    contract. A negative compliant impulse is retained, not positive evidence.
    """
    n=np.asarray(normal,float);j=np.asarray(impulse,float)
    if n.shape!=(3,) or j.shape!=(3,) or not np.isfinite(np.r_[n,j]).all():
        raise ValueError('Finite blade normal and impulse vectors required')
    length=math.hypot(*n)
    if abs(length-1.)>1e-4: raise ValueError('Unit blade contact normal required')
    n=n/length
    scalar=float(np.dot(j,n))
    if not math.isfinite(scalar) or math.hypot(*(j-scalar*n))>1e-12+1e-5*math.hypot(*j):
        raise ValueError('Blade impulse must be normal-only and collinear; no friction')
    return n,scalar


def resisting_face_normal(normal_on_knife,direction):
    """Oriented leading face; unlike leading_face_normal, sign is known."""
    n=np.asarray(normal_on_knife,float);d=np.asarray(direction,float)
    if n.shape!=(3,) or d.shape!=(3,) or not np.isfinite(np.r_[n,d]).all(): return False
    lengths=math.hypot(*n)*math.hypot(*d)
    return bool(lengths>1e-9 and np.dot(n,-d)/lengths>=np.cos(np.pi/6))


def leading_face_normal(normal,direction):
    """Native normal must face the source leading plane, not a broad face.

    Either collider-order sign is allowed; the 30-degree tolerance permits
    rounded shaft contact. This geometric test is not tissue calibration.
    """
    n=np.asarray(normal,float);d=np.asarray(direction,float)
    if n.shape!=(3,) or d.shape!=(3,) or not np.isfinite(np.r_[n,d]).all():return False
    lengths=np.linalg.norm(n)*np.linalg.norm(d)
    return bool(lengths>1e-9 and abs(np.dot(n,d))/lengths>=np.cos(np.pi/6))


def mount_forward(stage,robot_root,*,alignment='legacy'):
    """Distal knife with the corrected 180-degree wrist roll, session-only.

    EE -Z remains distal. Relative to the previous forward mount, rotate
    about EE Z, NOT Y again (which would put the tool back inside the wrist).
    Apply to the common parent so visuals, support, contacts and semantic edge
    stay together. Recognize explicit source/old/new frames: repeat is a no-op,
    and an unknown mounting fails closed rather than accumulating rotations.
    The opt-in camera alignment adds a wrist-Z roll to put source +Z (arc
    radial side) on the camera bracket's actual radial side. Global blade-down
    orientation must still be planned at the wrist; it is not a mounting claim.
    """
    if alignment not in ('legacy','camera'):
        raise ValueError('Knife alignment must be legacy or camera')
    from pxr import Usd,UsdGeom
    from .plant import matrix_attr
    wrist=stage.GetPrimAtPath(robot_root+'/ee_right')
    root=stage.GetPrimAtPath(str(wrist.GetPath())+'/attachments/DeleafKnife')
    if not wrist or not root: raise ValueError('Original fitted knife is missing')
    cache=UsdGeom.XformCache()
    inverse=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(wrist)).T)
    relative=inverse@np.asarray(cache.GetLocalToWorldTransform(root)).T
    parent=inverse@np.asarray(cache.GetLocalToWorldTransform(root.GetParent())).T
    source=np.array([[0.,0.,1.],[-1.,0.,0.],[0.,-1.,0.]])
    previous=np.diag([-1.,1.,-1.])@source
    legacy=np.diag([-1.,-1.,1.])@previous
    camera=stage.GetPrimAtPath(str(wrist.GetPath())+'/attachments/RightWristCamera')
    camera_side=None;camera_aligned=None;angle=0.
    if camera:
        camera_relative=inverse@np.asarray(cache.GetLocalToWorldTransform(camera)).T
        radial=camera_relative[:2,3]
        if np.isfinite(camera_relative).all() and np.linalg.norm(radial)>=.01:
            camera_side=radial/np.linalg.norm(radial)
            angle=float(np.arctan2(radial[1],radial[0]));c,s=np.cos(angle),np.sin(angle)
            camera_aligned=np.array([[c,-s,0],[s,c,0],[0,0,1.]])@legacy
    if alignment=='camera' and camera_aligned is None:
        raise ValueError('Right wrist camera must define an unambiguous radial mounting side')
    desired=camera_aligned if alignment=='camera' else legacy
    result=dict(changed=False,rotation_wrist_axis='Z',rotation_degrees=180,
        rotation_reference='previous_distal_mount',knife_extends_along='wrist_minus_z',
        flat_edge_faces='wrist_minus_y',curved_support_side='wrist_plus_x',
        source_asset_edited=False,
        hardware_fit='geometric_flange_alignment_not_CAD_fastener_certification')
    result.update(alignment=alignment,
        camera_side_wrist=None if camera_side is None else camera_side.tolist(),
        additional_roll_from_legacy_degrees=float(np.degrees(angle)) if alignment=='camera' else 0.,
        world_downward_orientation='requires_validated_wrist_pose_not_mount_rotation_alone')
    if alignment=='camera':
        result.update(flat_edge_faces='opposite_rotated_source_edge_x',
            curved_support_side='same_radial_side_as_right_wrist_camera',
            rotation_reference='legacy_distal_mount',rotation_degrees=float(np.degrees(angle)))
    if np.allclose(relative[:3,:3],desired,atol=1e-6): return result
    known=(source,previous,legacy)+(() if camera_aligned is None else (camera_aligned,))
    if not any(np.allclose(relative[:3,:3],r,atol=1e-6) for r in known):
        raise ValueError('Unknown knife mounting; do not guess a correction')
    corrected=relative.copy();corrected[:3,:3]=desired
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        matrix_attr(root,np.linalg.inv(parent)@corrected)
    result['changed']=True
    return result


@dataclass(frozen=True)
class ShearParameters:
    force_n: float = .2
    maximum_force_n: float = .5
    dwell_s: float = .025
    minimum_loading_travel_m: float = .0003
    maximum_grasp_slip_m: float = .003
    axial_tolerance_m: float = .003
    model: str = LEGACY_CUT_MODEL

    def __post_init__(self):
        if self.model not in CUT_MODELS: raise ValueError('Unknown explicit cut model')
        values=[v for k,v in vars(self).items() if k!='model']
        if not np.isfinite(values).all() or not all(v>0 for v in values):
            raise ValueError('Shear parameters must be finite and positive')
        if self.force_n>self.maximum_force_n:
            raise ValueError('Shear threshold exceeds the contact guard')


def cut_plane_normal(direction,stem_axis,tilt_degrees):
    """Small blade roll about a transverse stroke, not a changed stem axis.

    +/-15 degrees stays inside the existing measured shear angular gates.
    This only proposes geometry; it does not authorize contact or release.
    """
    d=np.array(direction,dtype=float,copy=True);axis=np.array(stem_axis,dtype=float,copy=True)
    if (d.shape!=(3,) or axis.shape!=(3,) or not np.isfinite(np.r_[d,axis]).all()
            or min(np.linalg.norm(d),np.linalg.norm(axis))<1e-9
            or not np.isfinite(tilt_degrees) or abs(tilt_degrees)>MAXIMUM_PLANE_TILT_DEGREES):
        raise ValueError('Finite transverse axes and blade tilt within +/-15 degrees required')
    d/=np.linalg.norm(d);axis/=np.linalg.norm(axis)
    if abs(np.dot(d,axis))>1e-6: raise ValueError('Blade stroke must remain transverse to the actual stem')
    angle=np.radians(tilt_degrees)
    return axis*np.cos(angle)+np.cross(d,axis)*np.sin(angle)


class KnifeGeometry:
    """Read actual source transforms once; no stage walks in the physics tick."""
    def __init__(self,stage,robot_root):
        from pxr import UsdGeom,UsdPhysics
        self.wrist_path=robot_root+'/ee_right'
        self.root=self.wrist_path+'/attachments/DeleafKnife'
        self.collider=self.root+('/BladePlateContact' if stage.GetPrimAtPath(self.root+'/BladePlateContact') else '/BladeCollision')
        cache=UsdGeom.XformCache()
        wrist=stage.GetPrimAtPath(self.wrist_path)
        inverse=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(wrist)).T)
        def local(name):
            prim=stage.GetPrimAtPath(self.root+'/'+name)
            if not prim or not prim.IsActive(): raise ValueError('Missing active knife '+name)
            return prim,inverse@np.asarray(cache.GetLocalToWorldTransform(prim)).T
        edge,matrix=local('CuttingEdge')
        plate,_=local(self.collider.rsplit('/',1)[1])
        blade,_=local('Blade');arc,_=local('Arc')
        arc_contact,_=local('ArcCollision')
        if (edge.GetAttribute('tomato:cuttingSurface').Get() is not True
                or blade.GetAttribute('tomato:cuttingSurface').Get() is not False
                or arc.GetAttribute('tomato:cuttingSurface').Get() is not False
                or not plate.HasAPI(UsdPhysics.CollisionAPI) or not UsdPhysics.CollisionAPI(plate).GetCollisionEnabledAttr().Get()
                or not arc_contact.HasAPI(UsdPhysics.CollisionAPI)):
            raise ValueError('Knife edge/plate/support semantics or contacts invalid')
        if UsdGeom.Imageable(blade).ComputeVisibility()=='invisible':
            raise ValueError('Original knife blade is not visible')
        if not UsdPhysics.CollisionAPI(arc_contact).GetCollisionEnabledAttr().Get():
            group=stage.GetPrimAtPath(self.root+'/ArcContacts')
            expected=group.GetAttribute('tomato:contactPartCount').Get() if group else None
            parts=list(group.GetChildren()) if group else []
            if (not isinstance(expected,int) or not 1<=expected<=14 or len(parts)!=expected
                    or any(not part.HasAPI(UsdPhysics.CollisionAPI)
                        or not UsdPhysics.CollisionAPI(part).GetCollisionEnabledAttr().Get() for part in parts)):
                raise ValueError('Knife arc support has missing or disabled contact partitions')
        self.size=np.linalg.norm(matrix[:3,:3],axis=0)*float(UsdGeom.Cube(edge).GetSizeAttr().Get())
        self.local=matrix.copy();self.local[:3,:3]/=np.linalg.norm(matrix[:3,:3],axis=0)
        if not np.allclose(self.local[:3,:3].T@self.local[:3,:3],np.eye(3),atol=1e-6):
            raise ValueError('Knife frame is not orthogonal')
        if not np.allclose(edge.GetAttribute('tomato:cuttingDirection').Get(),[-1,0,0]):
            raise ValueError('Unsupported source edge direction')
        # Verify original right tongs stay absent; never hide left fingers.
        for name in ('ee_right','ee_finger_r1','ee_finger_r2'):
            for scope in ('visuals','collisions','restored_collisions'):
                prim=stage.GetPrimAtPath(robot_root+'/'+name+'/'+scope)
                if prim and prim.IsActive(): raise ValueError('Right tongs still active')

    def frame(self,wrist):
        return np.asarray(wrist)@self.local

    def on_edge(self,point,frame):
        local=(np.asarray(point)-frame[:3,3])@frame[:3,:3]
        return bool(np.isfinite(local).all() and np.all(np.abs(local)<=self.size/2+.0006))

    def wrist_for_edge(self,centre,direction,plane_normal,wing=0.):
        # Blade-plane normal can differ slightly from the anatomical stem axis.
        # ShearGate always receives the actual measured stem axis separately.
        centre=np.array(centre,dtype=float,copy=True)
        direction=np.array(direction,dtype=float,copy=True);normal=np.array(plane_normal,dtype=float,copy=True)
        if (any(v.shape!=(3,) for v in (centre,direction,normal))
                or not np.isfinite(np.r_[centre,direction,normal]).all()
                or min(np.linalg.norm(direction),np.linalg.norm(normal))<1e-9):
            raise ValueError('Invalid blade pose vectors')
        direction/=np.linalg.norm(direction);normal/=np.linalg.norm(normal)
        if abs(np.dot(direction,normal))>1e-6: raise ValueError('Stroke must lie in the blade plane')
        edge=np.eye(4);edge[:3,:3]=np.column_stack([-direction,np.cross(normal,-direction),normal])
        if not np.isfinite(wing) or abs(wing)>self.size[1]/2-.005:
            raise ValueError('Contact must leave at least 5 mm from the knife end')
        edge[:3,3]=centre-wing*edge[:3,1]
        return edge@np.linalg.inv(self.local)


def transverse_stroke_offsets(stem_radius,edge_width,standoff=.025):
    """Cover the shaft, not an arbitrary 12 mm of post-seam overtravel.

    End only after the full leading strip clears the shaft radius plus 1 mm
    planning margin. Actual contact/force/travel still decide whether to cut;
    reaching this endpoint never releases a seam.
    """
    if (not np.isfinite([stem_radius,edge_width]).all() or min(stem_radius,edge_width)<=0):
        raise ValueError('Positive finite shaft radius and leading-strip width required')
    end=float(stem_radius+edge_width/2+.001)
    if end>.012: raise ValueError('Shaft exceeds the bounded diagnostic cutting corridor')
    if not np.isfinite(standoff) or not end+.002<=standoff<=.025:
        raise ValueError('Precontact standoff must clear the shaft/edge/margin plus 2 mm and remain <=25 mm')
    return np.linspace(-standoff,end,int(np.ceil((end+standoff)/.0005))+1)


class ShearGate:
    def __init__(self,target,parameters=None):
        self.target=target;self.parameters=parameters or ShearParameters()
        self.completed=False;self.reset_window()

    def reset_window(self):
        self.dwell=0.;self.travel=0.;self.peak=0.;self.steps=0
        self.previous=None;self.loading_origin=None;self.loading_direction=None
        self.minimum_resistance=None;self.peak_resistance=0.;self.peak_tool_upper=0.
        self.maximum_axial=0.;self.maximum_edge_dot=0.;self.maximum_direction_dot=0.
        self.minimum_step=0.;self.minimum_normal_cosine=1.
        self.signed_resistance_n=0.;self.unsigned_projection_n=0.

    def observe(self,*,dt,edge,centre,axis,points,impulses,held,slip,
                normals=None,impulse_contract=None,edge_contact_verified=False,
                tool_contact_upper_bound_n=None):
        """Signed resistance qualifies; noncancelling magnitudes only cap load.

        The caller must establish exact edge/collider provenance. An unsigned
        legacy vector list is NOT upgraded by guessing its collider order.
        The opt-in brittle model is a strength-only engineering approximation,
        not a displacement law, fracture energy or calibrated tissue cutting.
        """
        p=self.parameters
        if impulse_contract!=KNIFE_IMPULSE_CONTRACT or normals is None or tool_contact_upper_bound_n is None:
            self.reset_window()
            raise ValueError('Explicit on-knife normal impulse contract and full tool load bound required')
        edge=np.asarray(edge,float);centre=np.asarray(centre,float);axis=np.asarray(axis,float)
        points=np.asarray(points,float);impulses=np.asarray(impulses,float);normals=np.asarray(normals,float)
        if (edge.shape!=(4,4) or centre.shape!=(3,) or axis.shape!=(3,)
                or any(a.ndim not in (1,2) or a.shape not in ((0,),(len(a),3)) for a in (points,impulses,normals))
                or not len(points)==len(impulses)==len(normals)):
            self.reset_window();raise ValueError('Invalid native shear sample shapes')
        values=np.r_[dt,tool_contact_upper_bound_n,edge.flatten(),centre,axis,points.flatten(),impulses.flatten(),normals.flatten()]
        if (not np.isfinite(values).all() or dt<=0 or tool_contact_upper_bound_n<0
                or not np.allclose(edge[3],[0,0,0,1],rtol=0,atol=1e-8)
                or not np.allclose(edge[:3,:3].T@edge[:3,:3],np.eye(3),rtol=0,atol=1e-5)
                or np.linalg.det(edge[:3,:3])<=0 or abs(math.hypot(*axis)-1)>1e-4):
            self.reset_window();raise ValueError('Invalid native shear sample')
        relative=edge[:3,3]-centre;direction=-edge[:3,0]
        step=0. if self.previous is None else float(np.dot(relative-self.previous,direction))
        try:
            unit_normals=[knife_normal_impulse(n,j)[0] for n,j in zip(normals,impulses,strict=True)]
            projections=[float(np.dot(v,direction)) for v in impulses]
            resistance=-math.fsum(projections)/dt
            force=math.fsum(abs(v) for v in projections)/dt
            upper=max(float(tool_contact_upper_bound_n),force,math.fsum(math.hypot(*v) for v in impulses)/dt)
            if not np.isfinite([resistance,force,upper]).all(): raise ValueError('Overflowing native shear force')
        except (ValueError,OverflowError):
            self.reset_window();raise
        axial=max((abs(float(np.dot(point-centre,axis))) for point in points),default=0.)
        edge_dot=abs(float(np.dot(edge[:3,1],axis)));direction_dot=abs(float(np.dot(direction,axis)))
        normal_cosine=min((float(np.dot(n,-direction)/math.hypot(*direction)) for n in unit_normals),default=-1.)
        valid=bool(not self.completed and held and slip is not None and np.isfinite(slip)
            and 0<=slip<p.maximum_grasp_slip_m and len(points)>0 and edge_contact_verified is True
            and resistance>=p.force_n and force<=p.maximum_force_n and upper<=p.maximum_force_n
            and normal_cosine>=np.cos(np.pi/6) and edge_dot<.3 and direction_dot<.3
            and step>=-1e-6 and axial<=p.axial_tolerance_m)
        if not valid:
            self.reset_window()
            self.signed_resistance_n=resistance;self.unsigned_projection_n=force
            return None
        self.signed_resistance_n=resistance;self.unsigned_projection_n=force
        if self.loading_origin is None:
            # No approach displacement is credited to the first qualifying
            # contact sample. Both ends of measured advance need a valid load.
            self.loading_origin=relative.copy();self.loading_direction=direction.copy()
        self.previous=relative.copy()
        # Net advance, not a sum of positive jitter. Tiny permitted reverse
        # steps must subtract from travel instead of ratcheting a false cut.
        self.travel=max(0.,float(np.dot(relative-self.loading_origin,self.loading_direction)))
        self.dwell+=dt;self.peak=max(self.peak,force);self.steps+=1
        self.minimum_resistance=resistance if self.minimum_resistance is None else min(self.minimum_resistance,resistance)
        self.peak_resistance=max(self.peak_resistance,resistance);self.peak_tool_upper=max(self.peak_tool_upper,upper)
        self.maximum_axial=max(self.maximum_axial,axial);self.maximum_edge_dot=max(self.maximum_edge_dot,edge_dot)
        self.maximum_direction_dot=max(self.maximum_direction_dot,direction_dot);self.minimum_step=min(self.minimum_step,step)
        self.minimum_normal_cosine=min(self.minimum_normal_cosine,normal_cosine)
        travel_required=p.model==LEGACY_CUT_MODEL
        if self.dwell<p.dwell_s or (travel_required and self.travel<p.minimum_loading_travel_m): return None
        self.completed=True
        return dict(target=self.target,model=p.model,
            force_threshold_n=p.force_n,peak_force_n=self.peak,contact_dwell_s=self.dwell,
            force_contract=KNIFE_IMPULSE_CONTRACT,
            signed_resistance_definition='minus_sum_impulse_on_knife_dot_stroke_direction_over_dt',
            minimum_signed_resistance_n=self.minimum_resistance,peak_signed_resistance_n=self.peak_resistance,
            peak_tool_contact_upper_bound_n=self.peak_tool_upper,
            maximum_axial_contact_distance_m=self.maximum_axial,maximum_edge_axis_dot_stem=self.maximum_edge_dot,
            maximum_stroke_axis_dot_stem=self.maximum_direction_dot,minimum_relative_step_m=self.minimum_step,
            minimum_leading_normal_cosine=self.minimum_normal_cosine,
            measured_relative_loading_travel_m=self.travel,contact_steps=self.steps,
            travel_definition='net_advance_since_first_consecutive_qualified_contact',
            loading_travel_required=travel_required,
            minimum_loading_travel_m=p.minimum_loading_travel_m if travel_required else None,
            loading_travel_requirement_met=True if travel_required else None,
            measured_travel_is_tissue_work=False,fracture_energy_used_as_evidence=False,
            engineering_approximation=True,
            stable_left_grasp=True,grasp_slip_m=float(slip),flat_edge_contact_verified=True,
            commanded_motion_used_as_evidence=False,tissue_fracture_calibrated=False)
