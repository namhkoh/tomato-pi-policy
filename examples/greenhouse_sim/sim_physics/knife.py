"""Original fitted knife geometry and a measured-contact seam-failure model.

This is NOT calibrated tissue fracture. A force-qualified transverse edge load
breaks one preauthored 10 mm joint; mesh penetration is not called cutting work.
No timer, commanded velocity, broad blade face or U-support can trigger release.
"""
from dataclasses import dataclass

import numpy as np


def mount_forward(stage,robot_root):
    """Correct the inherited backward knife around the retained flange origin.

    EE -Z is distal (the original left fingers extend in -Z). The inherited
    knife occupies EE Z=0..124 mm, back into link_right_arm_6 (Z=0..46.5 mm).
    A 180-degree rotation about EE Y preserves the +Y flat cutting direction
    and puts the unchanged knife on the distal side. Session opinions only.
    """
    from pxr import Usd,UsdGeom
    from .plant import matrix_attr
    wrist=stage.GetPrimAtPath(robot_root+'/ee_right')
    root=stage.GetPrimAtPath(str(wrist.GetPath())+'/attachments/DeleafKnife')
    if not wrist or not root: raise ValueError('Original fitted knife is missing')
    cache=UsdGeom.XformCache()
    inverse=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(wrist)).T)
    relative=inverse@np.asarray(cache.GetLocalToWorldTransform(root)).T
    parent=inverse@np.asarray(cache.GetLocalToWorldTransform(root.GetParent())).T
    if relative[2,1]>.99: return dict(changed=False,knife_extends_along='wrist_minus_z')
    if relative[2,1]>-.99: raise ValueError('Unknown knife mounting; do not guess a correction')
    rotation=np.diag([-1.,1.,-1.,1.])
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        matrix_attr(root,np.linalg.inv(parent)@rotation@relative)
    return dict(changed=True,rotation_wrist_axis='Y',rotation_degrees=180,
        knife_extends_along='wrist_minus_z',source_asset_edited=False,
        hardware_fit='geometric_flange_alignment_not_CAD_fastener_certification')


@dataclass(frozen=True)
class ShearParameters:
    force_n: float = .2
    maximum_force_n: float = .5
    dwell_s: float = .025
    minimum_loading_travel_m: float = .0003
    maximum_grasp_slip_m: float = .003
    axial_tolerance_m: float = .003

    def __post_init__(self):
        values=list(vars(self).values())
        if not np.isfinite(values).all() or not all(v>0 for v in values):
            raise ValueError('Shear parameters must be finite and positive')
        if self.force_n>self.maximum_force_n:
            raise ValueError('Shear threshold exceeds the contact guard')


def cut_plane_normal(direction,stem_axis,tilt_degrees):
    """Small blade roll about a transverse stroke, not a changed stem axis.

    +/-10 degrees stays inside the existing measured shear angular gates.
    This only proposes geometry; it does not authorize contact or release.
    """
    d=np.array(direction,dtype=float,copy=True);axis=np.array(stem_axis,dtype=float,copy=True)
    if (d.shape!=(3,) or axis.shape!=(3,) or not np.isfinite(np.r_[d,axis]).all()
            or min(np.linalg.norm(d),np.linalg.norm(axis))<1e-9
            or not np.isfinite(tilt_degrees) or abs(tilt_degrees)>10):
        raise ValueError('Finite transverse axes and blade tilt within +/-10 degrees required')
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
        self.collider=self.root+'/BladeCollision'
        cache=UsdGeom.XformCache()
        wrist=stage.GetPrimAtPath(self.wrist_path)
        inverse=np.linalg.inv(np.asarray(cache.GetLocalToWorldTransform(wrist)).T)
        def local(name):
            prim=stage.GetPrimAtPath(self.root+'/'+name)
            if not prim or not prim.IsActive(): raise ValueError('Missing active knife '+name)
            return prim,inverse@np.asarray(cache.GetLocalToWorldTransform(prim)).T
        edge,matrix=local('CuttingEdge')
        plate,_=local('BladeCollision')
        blade,_=local('Blade');arc,_=local('Arc')
        arc_contact,_=local('ArcCollision')
        if (edge.GetAttribute('tomato:cuttingSurface').Get() is not True
                or blade.GetAttribute('tomato:cuttingSurface').Get() is not False
                or arc.GetAttribute('tomato:cuttingSurface').Get() is not False
                or not plate.HasAPI(UsdPhysics.CollisionAPI)
                or not arc_contact.HasAPI(UsdPhysics.CollisionAPI)):
            raise ValueError('Knife edge/plate/support semantics or contacts invalid')
        if UsdGeom.Imageable(blade).ComputeVisibility()=='invisible':
            raise ValueError('Original knife blade is not visible')
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


class ShearGate:
    def __init__(self,target,parameters=None):
        self.target=target;self.parameters=parameters or ShearParameters()
        self.completed=False;self.previous=None;self.reset_window()

    def reset_window(self):
        self.dwell=0.;self.travel=0.;self.peak=0.;self.steps=0

    def observe(self,*,dt,edge,centre,axis,points,impulses,held,slip):
        p=self.parameters
        values=np.r_[dt,edge.flatten(),centre,axis,np.asarray(points).flatten(),np.asarray(impulses).flatten()]
        if not np.isfinite(values).all() or dt<=0: raise ValueError('Invalid native shear sample')
        relative=edge[:3,3]-centre;direction=-edge[:3,0]
        step=0. if self.previous is None else float(np.dot(relative-self.previous,direction))
        self.previous=relative.copy()
        force=sum(abs(float(np.dot(v,direction))) for v in impulses)/dt
        valid=bool(not self.completed and held and slip is not None and np.isfinite(slip)
            and slip<p.maximum_grasp_slip_m and len(points)>0
            and len(points)==len(impulses) and p.force_n<=force<=p.maximum_force_n
            and abs(float(np.dot(edge[:3,1],axis)))<.3
            and abs(float(np.dot(direction,axis)))<.3 and step>=-1e-6
            and all(abs(float(np.dot(np.asarray(point)-centre,axis)))<=p.axial_tolerance_m for point in points))
        if not valid:
            self.reset_window();return None
        self.dwell+=dt;self.travel+=max(step,0);self.peak=max(self.peak,force);self.steps+=1
        if self.dwell<p.dwell_s or self.travel<p.minimum_loading_travel_m: return None
        self.completed=True
        return dict(target=self.target,model='force_qualified_pre_authored_seam_release',
            force_threshold_n=p.force_n,peak_force_n=self.peak,contact_dwell_s=self.dwell,
            measured_relative_loading_travel_m=self.travel,contact_steps=self.steps,
            stable_left_grasp=True,grasp_slip_m=float(slip),flat_edge_contact_verified=True,
            commanded_motion_used_as_evidence=False,tissue_fracture_calibrated=False)
