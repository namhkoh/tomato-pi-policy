"""Session-only, one-petiole PhysX qualification rig using supplied native assets.

The parent plant is a fixed support in this FIRST increment. Leaf laminas are
rigid carriers with convex contacts; stem art is skinned to its compliant chain.
Only an explicitly diagnostic seam-release API exists: no false cutting success.
"""
from dataclasses import dataclass,field
from pathlib import Path

import numpy as np
from pxr import Gf,Sdf,Usd,UsdGeom,UsdPhysics,UsdShade,Vt

from sim_data.audit import audit_manifest,safe_asset
from sim_data.cut_regions import _oriented_chain
from .mechanics import Material,beam_properties,resample_chain,segment_frames,SkinBinding,nearest_segments,clip_mesh,lamina_mass_properties,combine_mass_properties


def matrix_attr(prim,matrix):
    xform=UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp().Set(Gf.Matrix4d(np.asarray(matrix).T.tolist()))


def physics_schema(prim,name,attributes):
    """Author native PhysX schema opinions, also inspectable without starting Kit."""
    prim.AddAppliedSchema(name)
    for key,type_name,value in attributes:
        prim.CreateAttribute(key,type_name,custom=False).Set(value)


def world_matrix(prim,cache):
    return np.asarray(cache.GetLocalToWorldTransform(prim),dtype=float).T


def points_world(mesh,cache):
    points=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
    matrix=world_matrix(mesh.GetPrim(),cache)
    return points@matrix[:3,:3].T+matrix[:3,3]


def mesh_uv(mesh):
    pv=UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st')
    if not pv or not pv.HasValue(): return None
    values=np.asarray(pv.ComputeFlattened(),dtype=float)
    mode=pv.GetInterpolation()
    indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=int)
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),dtype=int)
    if mode in ('vertex','varying'): return values[indices]
    if mode=='faceVarying': return values
    if mode=='constant': return np.repeat(values[:1],len(indices),axis=0)
    if mode=='uniform': return np.repeat(values,counts,axis=0)
    raise ValueError('Unsupported source UV interpolation: '+str(mode))


@dataclass
class PlantRig:
    stage: object
    root: str
    source_target: str
    body_paths: list
    rest_frames: np.ndarray
    chain_world: np.ndarray
    arcs: np.ndarray
    cut_index: int
    cut_joint_path: str
    properties: list
    constraint_mode: str='articulation'
    visuals: list=field(default_factory=list)
    cut: bool=False

    def sync_visuals(self,frames):
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            for mesh,binding in self.visuals:
                points=binding.deform(frames).astype(np.float32)
                mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
                mesh.GetExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([points.min(0),points.max(0)])))

    def diagnostic_release(self):
        """Mechanism qualification only; not blade/contact-verified tissue cutting."""
        if self.constraint_mode=='fixed_articulation':
            raise ValueError('Fixed articulation needs a state-preserving topology transition before release')
        if self.cut: raise ValueError('Diagnostic seam is already released')
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            UsdPhysics.Joint.Get(self.stage,self.cut_joint_path).GetJointEnabledAttr().Set(False)
        self.cut=True
        return dict(event='diagnostic_joint_release',target=self.source_target,
                    material_arc_m=float(self.arcs[self.cut_index]),
                    physical_cut_verified=False,training_eligible=False)

    def restore_authored_state(self):
        """Only while simulation is stopped; caller must rebuild tensor views."""
        with Usd.EditContext(self.stage,self.stage.GetSessionLayer()):
            for path,frame in zip(self.body_paths,self.rest_frames,strict=True):
                matrix_attr(self.stage.GetPrimAtPath(path),frame)
                api=UsdPhysics.RigidBodyAPI(self.stage.GetPrimAtPath(path))
                api.CreateVelocityAttr(Gf.Vec3f(0));api.CreateAngularVelocityAttr(Gf.Vec3f(0))
            UsdPhysics.Joint.Get(self.stage,self.cut_joint_path).GetJointEnabledAttr().Set(True)
        self.cut=False
        self.sync_visuals(self.rest_frames)

    def report(self):
        return dict(source_target=self.source_target,body_count=len(self.body_paths),
                    constraint_mode=self.constraint_mode,
                    stem_visual_meshes=len(self.visuals),cut_material_arc_m=float(self.arcs[self.cut_index]),
                    material_calibration='engineering_prior_not_lab_calibrated',
                    parent_support='fixed_current_increment',leaves='rigid_lamina_convex_contact',
                    release_model='preauthored_10mm_seam_diagnostic_only',
                    physical_cut_verified=False,robot_grasp_verified=False,training_eligible=False)


def _body(stage,path,frame,props,*,kinematic):
    body=UsdGeom.Xform.Define(stage,path).GetPrim();matrix_attr(body,frame)
    api=UsdPhysics.RigidBodyAPI.Apply(body);api.CreateKinematicEnabledAttr(kinematic)
    mass=UsdPhysics.MassAPI.Apply(body);mass.CreateMassAttr(float(props['mass']))
    mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float,props['inertia'])))
    mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float,props.get('center',np.zeros(3)))))
    axes=np.eye(4);axes[:3,:3]=props.get('principal_axes',np.eye(3))
    mass.CreatePrincipalAxesAttr(Gf.Quatf(Gf.Matrix4d(axes.T.tolist()).ExtractRotationQuat()))
    physics_schema(body,'PhysxContactReportAPI',[
        ('physxContactReport:threshold',Sdf.ValueTypeNames.Float,0.)])
    physics_schema(body,'PhysxRigidBodyAPI',[
        ('physxRigidBody:solverPositionIterationCount',Sdf.ValueTypeNames.Int,16),
        ('physxRigidBody:solverVelocityIterationCount',Sdf.ValueTypeNames.Int,4),
        ('physxRigidBody:maxDepenetrationVelocity',Sdf.ValueTypeNames.Float,.2)])
    return body


def _collision(prim):
    UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
    physics_schema(prim,'PhysxCollisionAPI',[
        ('physxCollision:contactOffset',Sdf.ValueTypeNames.Float,.0005),
        ('physxCollision:restOffset',Sdf.ValueTypeNames.Float,0.)])


def _joint(stage,path,parent,child,anchor,frames,props,*,external):
    # The separable cross-section is a fixed interface, not a free hinge.
    # Bending belongs to the internal reduced-coordinate beam joints.
    joint=(UsdPhysics.FixedJoint if external else UsdPhysics.Joint).Define(stage,path)
    joint.CreateBody0Rel().SetTargets([parent]);joint.CreateBody1Rel().SetTargets([child])
    for side,frame in enumerate(frames):
        local=np.linalg.inv(frame)@np.r_[anchor,1]
        rotation=frame[:3,:3].T@frames[1][:3,:3]
        getattr(joint,f'CreateLocalPos{side}Attr')().Set(Gf.Vec3f(*map(float,local[:3])))
        quat=Gf.Matrix4d(np.block([[rotation,np.zeros((3,1))],[np.zeros((1,3)),np.ones((1,1))]]).T.tolist()).ExtractRotationQuat()
        getattr(joint,f'CreateLocalRot{side}Attr')().Set(Gf.Quatf(quat))
    joint.CreateExcludeFromArticulationAttr(external)
    joint.CreateJointEnabledAttr(True)
    joint.CreateCollisionEnabledAttr(False)
    if external:
        return joint
    for axis in ('transX','transY','transZ'):
        api=UsdPhysics.LimitAPI.Apply(joint.GetPrim(),axis);api.CreateLowAttr(1.);api.CreateHighAttr(-1.)
    for i,axis in enumerate(('rotX','rotY','rotZ')):
        drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),axis)
        drive.CreateTypeAttr('force');drive.CreateTargetPositionAttr(0.)
        drive.CreateStiffnessAttr(float(props['usd_stiffness'][i]))
        drive.CreateDampingAttr(float(props['usd_damping'][i]))
    return joint


def build(stage,record,component_id,*,root='/World/InteractionPhysics/Target',material=None,max_segment_m=.025,constraint_mode='articulation'):
    """Convert only a leaf-bearing native petiole; preserve package/source layers."""
    material=material or Material()
    if constraint_mode not in ('articulation','maximal','fixed_articulation'): raise ValueError('Invalid constraint mode')
    if stage.GetPrimAtPath(root): raise ValueError('Physics root already exists')
    report=audit_manifest(record['manifest_path']);component=report['components'][component_id]
    target=next(t for t in report['targets'] if t['component_id']==component_id)
    if target['status']=='excluded' or target['protected_descendant_ids']:
        raise ValueError('Target must be intact, leaf-bearing and free of protected descendants')
    members=target['expected_detached_component_ids']
    if any(report['components'][key]['type']!='leaf' for key in members if key!=component_id):
        raise ValueError('First increment supports a petiole with direct leaf descendants only')
    if any(report['components'][key]['parent']!=component_id for key in members if key!=component_id):
        raise ValueError('Nested leaf network requires a separate articulated adapter')
    source_path=record['component_paths'][component_id]
    cache=UsdGeom.XformCache();source=stage.GetPrimAtPath(source_path)
    source_frame=world_matrix(source,cache)
    if not np.allclose(source_frame[:3,:3].T@source_frame[:3,:3],np.eye(3),atol=1e-6):
        raise ValueError('Physics needs rigid metre transforms; source scaling is unsupported')
    local,_,_,_=_oriented_chain(component,1e-6)
    chain,arcs=resample_chain(local,max_segment_m=max_segment_m)
    xyz=chain[:,:3]@source_frame[:3,:3].T+source_frame[:3,3]
    frames=segment_frames(xyz);cut_index=int(np.flatnonzero(arcs==.01)[0])
    leaves=[key for key in members if key!=component_id]
    leaf_frames={key:world_matrix(stage.GetPrimAtPath(record['component_paths'][key]),cache) for key in leaves}
    leaf_carriers={key:max(cut_index,int(nearest_segments(np.array([frame[:3,3]]),xyz)[0][0])) for key,frame in leaf_frames.items()}
    leaf_mass_parts={}
    for key in leaves:
        vertices=[];triangles=[]
        inverse=np.linalg.inv(frames[leaf_carriers[key]])
        for prim in Usd.PrimRange(stage.GetPrimAtPath(record['component_paths'][key])):
            if not prim.IsA(UsdGeom.Mesh): continue
            mesh=UsdGeom.Mesh(prim);points=points_world(mesh,cache)
            offset=len(vertices);vertices.extend(points@inverse[:3,:3].T+inverse[:3,3])
            indices=list(mesh.GetFaceVertexIndicesAttr().Get());cursor=0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                face=indices[cursor:cursor+count];cursor+=count
                triangles.extend([[offset+face[0],offset+face[j],offset+face[j+1]] for j in range(1,count-1)])
        leaf_mass_parts[key]=lamina_mass_properties(vertices,triangles,material.leaf_mass_kg)
    lengths=np.linalg.norm(np.diff(xyz,axis=0),axis=1)
    props=[]
    for i,length in enumerate(lengths):
        support=sum(material.leaf_mass_kg*np.linalg.norm(frame[:3,3]-xyz[i])**2 for key,frame in leaf_frames.items() if leaf_carriers[key]>=i)
        p=beam_properties(float(np.mean(chain[i:i+2,3])),float(length),material,supported_inertia=support)
        parts=[(p['mass'],np.zeros(3),np.diag(p['inertia']))]
        parts.extend(leaf_mass_parts[key] for key in leaves if leaf_carriers[key]==i)
        p.update(combine_mass_properties(parts));props.append(p)
    body_paths=[root+('/Support' if i<cut_index else '/Branch')+f'/Segment_{i:03d}' for i in range(len(frames))]
    # Gather exact source mesh buffers BEFORE any visibility/session edits.
    own_meshes=[UsdGeom.Mesh(p) for p in Usd.PrimRange(source) if p.IsA(UsdGeom.Mesh)
                and not any(str(p.GetPath()).startswith(record['component_paths'][key]+'/') for key in leaves)]
    buffers=[(mesh,points_world(mesh,cache),mesh_uv(mesh)) for mesh in own_meshes]
    rig=PlantRig(stage,root,target['target_id'],body_paths,frames,xyz,arcs,cut_index,root+'/CutInterface',props,constraint_mode=constraint_mode)
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        # The supplied art already has static colliders. Visibility does NOT
        # disable physics: remove only the replaced target's duplicate contacts.
        for prim in Usd.PrimRange(source):
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        UsdGeom.Xform.Define(stage,root)
        UsdGeom.Xform.Define(stage,root+'/Branch')
        for i,(path,frame,p) in enumerate(zip(body_paths,frames,props,strict=True)):
            body=_body(stage,path,frame,p,kinematic=i<cut_index)
            if i==cut_index and constraint_mode=='articulation':
                # Explicit proximal root avoids automatic mid-chain rooting.
                UsdPhysics.ArticulationRootAPI.Apply(body)
                physics_schema(body,'PhysxArticulationAPI',[
                    ('physxArticulation:enabledSelfCollisions',Sdf.ValueTypeNames.Bool,False),
                    ('physxArticulation:solverPositionIterationCount',Sdf.ValueTypeNames.Int,16),
                    ('physxArticulation:solverVelocityIterationCount',Sdf.ValueTypeNames.Int,4)])
            shape=UsdGeom.Capsule.Define(stage,path+'/StemCollider')
            radius=min(float(np.mean(chain[i:i+2,3])),float(lengths[i])*.49)
            shape.CreateRadiusAttr(radius);shape.CreateHeightAttr(float(lengths[i])-2*radius)
            shape.CreateAxisAttr('Z');shape.CreatePurposeAttr('guide');_collision(shape.GetPrim())
            body.CreateAttribute('tomato:sourceTarget',Sdf.ValueTypeNames.String,custom=True).Set(target['target_id'])
            if i>=cut_index:
                joint=_joint(stage,rig.cut_joint_path if i==cut_index else root+f'/Branch/Joint_{i:03d}',
                       body_paths[i-1],path,xyz[i],frames[i-1:i+1],p,external=i==cut_index)
                if i==cut_index and constraint_mode=='fixed_articulation':
                    joint.CreateBody0Rel().ClearTargets(True)
                    joint.CreateLocalPos0Attr(Gf.Vec3f(*map(float,xyz[i])))
                    joint.CreateLocalRot0Attr(Gf.Quatf(Gf.Matrix4d(frame.T.tolist()).ExtractRotationQuat()))
                    joint.CreateExcludeFromArticulationAttr(False)
                    UsdPhysics.ArticulationRootAPI.Apply(joint.GetPrim())
                    physics_schema(joint.GetPrim(),'PhysxArticulationAPI',[
                        ('physxArticulation:enabledSelfCollisions',Sdf.ValueTypeNames.Bool,False),
                        ('physxArticulation:sleepThreshold',Sdf.ValueTypeNames.Float,0.),
                        ('physxArticulation:solverPositionIterationCount',Sdf.ValueTypeNames.Int,16),
                        ('physxArticulation:solverVelocityIterationCount',Sdf.ValueTypeNames.Int,4)])
        for key in leaves:
            i=leaf_carriers[key];path=body_paths[i]+'/Leaf_'+key
            prim=UsdGeom.Xform.Define(stage,path).GetPrim()
            prim.GetReferences().AddReference(safe_asset(Path(report['manifest_path']).parent,report['components'][key]['file']).as_posix())
            matrix_attr(prim,np.linalg.inv(frames[i])@leaf_frames[key]);stage.Load(path)
            for child in Usd.PrimRange(prim):
                if child.IsA(UsdGeom.Mesh):
                    _collision(child);UsdPhysics.MeshCollisionAPI.Apply(child).CreateApproximationAttr('convexHull')
            UsdGeom.Imageable(stage.GetPrimAtPath(record['component_paths'][key])).MakeInvisible()
        for mesh_index,(source_mesh,points,uv) in enumerate(buffers):
            counts=source_mesh.GetFaceVertexCountsAttr().Get();indices=source_mesh.GetFaceVertexIndicesAttr().Get()
            for positive,label in ((False,'Proximal'),(True,'Distal')):
                vertices,tex=clip_mesh(points,counts,indices,xyz[cut_index],frames[cut_index,:3,2],positive=positive,uv=uv)
                if not len(vertices): continue
                mesh=UsdGeom.Mesh.Define(stage,root+f'/Visuals/{label}_{mesh_index}')
                mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
                mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.arange(len(vertices),dtype=np.int32)))
                mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(vertices)//3,3,dtype=np.int32)))
                mesh.CreateSubdivisionSchemeAttr('none');mesh.CreateDoubleSidedAttr(True)
                mesh.CreateExtentAttr()
                if tex is not None:
                    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,'faceVarying').Set(Vt.Vec2fArray.FromNumpy(tex.astype(np.float32)))
                bound,_=UsdShade.MaterialBindingAPI(source_mesh.GetPrim()).ComputeBoundMaterial()
                if bound: UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(bound)
                mesh.GetPrim().CreateAttribute('tomato:sourceTarget',Sdf.ValueTypeNames.String,custom=True).Set(target['target_id'])
                skin=SkinBinding(vertices,frames,xyz,first_segment=cut_index if positive else 0,
                                 last_segment=None if positive else cut_index-1)
                rig.visuals.append((mesh,skin))
            UsdGeom.Imageable(source_mesh).MakeInvisible()
        rig.sync_visuals(frames)
    return rig
