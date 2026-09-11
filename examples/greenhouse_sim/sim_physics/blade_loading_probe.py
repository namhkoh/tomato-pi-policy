"""Native force-limited knife/coupon diagnostic, with seam release forbidden.

Uses the original approved knife and source-derived seam-adjacent capsules.
The two-body coupon isolates the current rigid local contact interface: it is
NOT the full plant, greenhouse, robot, grasp or bimanual qualification. Nothing
here changes benchmark thresholds, material parameters or source assets.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback
import numpy as np

from .blade_loading import LoadingWindow,joint_anchors,joint_angular_error,slide_parameters,leading_face_normal,loading_sample_safe


def author(stage,*,force_limit_n=.35,normal_sign=1,contact_stiffness_n_m=0.):
    from pxr import Usd,UsdGeom,UsdPhysics,UsdShade,Gf,Sdf
    from scipy.spatial.transform import Rotation
    from sim_data.audit import DEFAULT_PACK,audit_manifest
    from sim_data.geometry import assemble_plant
    from sim_data.depth_preview import sha256
    from greenhouse_sim.robot_hardware import _author_knife,load_manifest,MANIFEST_PATH,DERIVED_DIR
    from .plant import build,_body,_collision,_joint,matrix_attr,physics_schema
    from .knife import KnifeGeometry,mount_forward
    from .blade_contacts import refine_blade_contacts
    from .arc_contacts import refine_arc_contacts
    drive_parameters=slide_parameters(force_limit_n)
    if normal_sign not in (-1,1):raise ValueError('Blade normal sign must be +/-1')
    if contact_stiffness_n_m not in (0.,250.,500.,1000.,2000.):raise ValueError('Bounded diagnostic contact stiffness required')
    # Offline source extraction. The full source plant is never deactivated in
    # a running benchmark; this is a separate, explicitly labeled coupon stage.
    source=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(source,1);UsdGeom.SetStageUpAxis(source,'Z')
    manifest=DEFAULT_PACK/'plants/components/seed101_full/manifest.json'
    audited=audit_manifest(manifest);paths=assemble_plant(source,'/World/Plant',audited)
    rig=build(source,dict(manifest_path=str(manifest),component_paths=paths,plant_root='/World/Plant'),'SubStem_41')
    chosen=[rig.cut_index-1,rig.cut_index];frames=rig.rest_frames[chosen].copy()
    centre=np.array([0.,0.,1.]);frames[:,:3,3]+=centre-rig.chain_world[rig.cut_index]
    axis=frames[1,:3,2];direction=np.cross(axis,[0.,0.,1.])
    if np.linalg.norm(direction)<1e-9:direction=np.cross(axis,[0.,1.,0.])
    direction/=np.linalg.norm(direction)
    coupon_paths=['/World/Coupon/Bodies/Proximal','/World/Coupon/Bodies/Distal'];shapes=[];dimensions=[]
    for side,(index,path) in enumerate(zip(chosen,coupon_paths)):
        body=_body(stage,path,frames[side],rig.properties[index],kinematic=side==0)
        if side==1:UsdPhysics.ArticulationRootAPI.Apply(body)
        shape=UsdGeom.Capsule.Define(stage,path+'/StemCollider')
        original=UsdGeom.Capsule.Get(source,rig.body_paths[index]+'/StemCollider')
        radius=float(original.GetRadiusAttr().Get());height=float(original.GetHeightAttr().Get())
        shape.CreateRadiusAttr(radius);shape.CreateHeightAttr(height);shape.CreateAxisAttr('Z')
        shape.CreateDisplayColorAttr([Gf.Vec3f(.17,.43,.08)]);_collision(shape.GetPrim())
        shapes.append(str(shape.GetPath()));dimensions.append(dict(radius_m=radius,cylinder_height_m=height))
    joint=_joint(stage,'/World/Coupon/IntactSeam',*coupon_paths,centre,frames,rig.properties[chosen[1]],external=True)
    local=np.array([joint.GetLocalPos0Attr().Get(),joint.GetLocalPos1Attr().Get()],float)
    rotations=[]
    for q in (joint.GetLocalRot0Attr().Get(),joint.GetLocalRot1Attr().Get()):
        rotations.append(Rotation.from_quat([*q.GetImaginary(),q.GetReal()]).as_matrix())
    compliance=None
    if contact_stiffness_n_m:
        # Explicit experimental tissue-interface compliance. The knife remains
        # rigid; both source-capsule collision shapes bind this material. The
        # fixed coupon support makes carriage mass the effective moving mass.
        compliance=dict(stiffness_n_m=contact_stiffness_n_m,
            damping_ns_m=float(1.4*np.sqrt(contact_stiffness_n_m*.05)),
            effective_mass_kg=.05,calibrated=False,force_based=True,
            scope='two_body_coupon_native_contact_not_benchmark_default')
        material=UsdShade.Material.Define(stage,'/World/Coupon/ExperimentalInterface')
        api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        api.CreateStaticFrictionAttr(.5);api.CreateDynamicFrictionAttr(.5);api.CreateRestitutionAttr(0.)
        physics_schema(material.GetPrim(),'PhysxMaterialAPI',[
            ('physxMaterial:compliantContactStiffness',Sdf.ValueTypeNames.Float,contact_stiffness_n_m),
            ('physxMaterial:compliantContactDamping',Sdf.ValueTypeNames.Float,compliance['damping_ns_m']),
            ('physxMaterial:compliantContactAccelerationSpring',Sdf.ValueTypeNames.Bool,False)])
        for path in shapes:UsdShade.MaterialBindingAPI.Apply(stage.GetPrimAtPath(path)).Bind(material,materialPurpose='physics')
    root='/World/LoadCarriage';wrist_path=root+'/ee_right'
    carriage=_body(stage,wrist_path,np.eye(4),dict(mass=.05,inertia=np.full(3,.0001)),kinematic=False)
    physics_schema(carriage,'PhysxRigidBodyAPI',[
        ('physxRigidBody:sleepThreshold',Sdf.ValueTypeNames.Float,0.)])
    _author_knife(stage,wrist_path,load_manifest());mount=mount_forward(stage,root)
    plate=refine_blade_contacts(stage,root);arc=refine_arc_contacts(stage,root)
    knife=KnifeGeometry(stage,root)
    clearance=max(d['radius_m'] for d in dimensions)+knife.size[0]/2+.002
    initial=knife.wrist_for_edge(centre-clearance*direction,direction,normal_sign*axis)
    matrix_attr(stage.GetPrimAtPath(wrist_path),initial)
    # One native prismatic support holds orientation/transverse translation.
    # The loading axis is horizontal, so gravity is reacted by the support.
    axes=np.column_stack([direction,axis,np.cross(direction,axis)])
    slide=UsdPhysics.PrismaticJoint.Define(stage,root+'/Slide')
    slide.CreateBody1Rel().SetTargets([wrist_path]);slide.CreateAxisAttr('X')
    slide.CreateLocalPos0Attr(Gf.Vec3f(*map(float,initial[:3,3])));slide.CreateLocalPos1Attr(Gf.Vec3f(0))
    for side,rotation in enumerate((axes,initial[:3,:3].T@axes)):
        m=np.eye(4);m[:3,:3]=rotation
        getattr(slide,f'CreateLocalRot{side}Attr')().Set(Gf.Quatf(Gf.Matrix4d(m.T.tolist()).ExtractRotationQuat()))
    slide.CreateLowerLimitAttr(-.001);slide.CreateUpperLimitAttr(.012)
    slide.CreateExcludeFromArticulationAttr(True)
    drive=UsdPhysics.DriveAPI.Apply(slide.GetPrim(),'linear');drive.CreateTypeAttr('force')
    drive.CreateStiffnessAttr(drive_parameters['stiffness_n_m'])
    drive.CreateDampingAttr(drive_parameters['damping_ns_m'])
    drive.CreateMaxForceAttr(drive_parameters['maximum_force_n']);drive.CreateTargetVelocityAttr(0.)
    hashes={str(manifest):sha256(manifest),str(MANIFEST_PATH):sha256(MANIFEST_PATH)}
    for component in audited['components'].values():hashes[str(manifest.parent/component['file'])]=component['asset_sha256']
    for name in ('deleaf_knife_blade.stl','deleaf_knife_arc.stl'):hashes[str(DERIVED_DIR/name)]=sha256(DERIVED_DIR/name)
    return dict(knife=knife,joint=joint,drive=drive,local_anchors=local,local_rotations=np.asarray(rotations),coupon_paths=coupon_paths,shapes=shapes,
        direction=direction,initial_wrist=initial,frames=frames,source_hashes=hashes,
        metadata=dict(source_target=rig.source_target,cut_arc_m=float(rig.arcs[rig.cut_index]),
            source_capsules=dimensions,source_body_masses_kg=[rig.properties[i]['mass'] for i in chosen],
            source_body_inertias_kg_m2=[np.asarray(rig.properties[i]['inertia']).tolist() for i in chosen],
            knife_mount=mount,blade_contacts=plate,arc_contacts=arc,force_limit_n=force_limit_n,
            loading_drive=dict(model='implicit_native_prismatic_force_limited_velocity_drive',**drive_parameters),
            diagnostic_carriage_mass_kg=.05,carriage_mass_is_not_measured_robot_tool_mass=True,
            diagnostic_carriage_sleep_disabled=True,leading_face_normal_tolerance_degrees=30.,
            full_plant=False,full_robot=False,beam_bending_and_leaf_loads_present=False,
            copied_local_model='source_capsules_one_link_articulation_external_fixed_seam',
            applied_material_compliance=bool(compliance),contact_compliance=compliance,
            maximum_joint_angular_residual_degrees=1.,release_enabled=False,normal_sign=normal_sign))


class NativeContacts:
    def __init__(self):self.rows=[];self.friction=[];self.error=None;self.paths={};self.subscription=None
    def subscribe(self):
        from omni.physx import get_physx_simulation_interface
        self.subscription=get_physx_simulation_interface().subscribe_full_contact_report_events(self.callback)
    def callback(self,headers,data,friction):
        from pxr import PhysicsSchemaTools
        def path(value):
            if value not in self.paths:self.paths[value]=str(PhysicsSchemaTools.intToSdfPath(value))
            return self.paths[value]
        def vector(value):return [float(value.x),float(value.y),float(value.z)]
        try:
            for header in headers:
                begin=header.contact_data_offset;end=begin+header.num_contact_data
                if header.num_contact_data<0:raise ValueError('Invalid native contact count')
                if header.num_contact_data==0:begin=end=0
                if begin<0 or end<begin or end>len(data):raise ValueError('Invalid native contact span')
                pair=[path(header.collider0),path(header.collider1)]
                for c in data[begin:end]:
                    row=dict(colliders=pair,position=vector(c.position),normal=vector(c.normal),
                        impulse=vector(c.impulse),separation_m=float(c.separation))
                    if not np.isfinite(row['position']+row['normal']+row['impulse']+[row['separation_m']]).all():
                        raise ValueError('Nonfinite native contact')
                    self.rows.append(row)
                begin=header.friction_anchors_offset;end=begin+header.num_friction_anchors_data
                if header.num_friction_anchors_data<0:raise ValueError('Invalid native friction count')
                if header.num_friction_anchors_data==0:begin=end=0
                if begin<0 or end<begin or end>len(friction):raise ValueError('Invalid native friction span')
                for c in friction[begin:end]:
                    row=dict(colliders=pair,position=vector(c.position),impulse=vector(c.impulse))
                    if not np.isfinite(row['position']+row['impulse']).all():raise ValueError('Nonfinite native friction anchor')
                    self.friction.append(row)
        except Exception as error:self.error=str(error)


def run(sim,fixture,*,seconds=8.,force_limit_n=.35,physics_hz=240):
    from .runtime import pose_matrices
    from greenhouse_sim.physics_clock import PhysicsClock
    view=sim.physics_sim_view;view.set_subspace_roots('/')
    coupon=view.create_rigid_body_view('/World/Coupon/Bodies/*')
    order=[list(coupon.prim_paths).index(p) for p in fixture['coupon_paths']]
    wrist=view.create_rigid_body_view(fixture['knife'].wrist_path)
    if coupon.count!=2 or wrist.count!=1:raise RuntimeError('Unexpected native fixture body count')
    contacts=NativeContacts();contacts.subscribe();window=LoadingWindow();records=[];fault=None
    direction=fixture['direction'];knife=fixture['knife']
    clock=PhysicsClock(sim,physics_hz=physics_hz,render_hz=0);command=0.
    previous_wrist=pose_matrices(wrist.get_transforms())[0][:3,3].copy()
    def before(stamp,dt):
        nonlocal command
        contacts.rows=[];contacts.friction=[]
        if stamp.simulation_time_s>=.5 and command==0.:
            command=fixture['metadata']['loading_drive']['target_velocity_m_s']
            fixture['drive'].GetTargetVelocityAttr().Set(command)
    def after(stamp,dt):
        nonlocal previous_wrist
        if contacts.error:raise RuntimeError(contacts.error)
        frames=pose_matrices(coupon.get_transforms()[order]);anchors=joint_anchors(frames,fixture['local_anchors'])
        actual=pose_matrices(wrist.get_transforms())[0];edge=knife.frame(actual);axis=frames[1,:3,2]
        eligible=[];unwanted=0.
        for c in contacts.rows:
            other=next((p for p in c['colliders'] if p in fixture['shapes']),None)
            valid=(knife.collider in c['colliders'] and other is not None and knife.on_edge(c['position'],edge)
                and leading_face_normal(c['normal'],direction)
                and abs(np.dot(np.asarray(c['position'])-anchors[1],axis))<=.003)
            c['eligible_flat_edge_seam_contact']=bool(valid)
            if valid:eligible.append(c)
            else:unwanted+=float(np.linalg.norm(c['impulse']))/dt
        force=sum(abs(float(np.dot(c['impulse'],direction))) for c in eligible)/dt
        normal_magnitude=sum(float(np.linalg.norm(c['impulse'])) for c in contacts.rows)/dt
        friction_magnitude=sum(float(np.linalg.norm(c['impulse'])) for c in contacts.friction)/dt
        # An upper bound: no cancellation of opposite contact/anchor forces.
        # Friction is NEVER supplied to the normal edge-force qualification.
        total=normal_magnitude+friction_magnitude
        good_pairs={tuple(c['colliders']) for c in eligible}
        bad_pairs={tuple(c['colliders']) for c in contacts.rows if not c['eligible_flat_edge_seam_contact']}
        for c in contacts.friction:
            allowed=(tuple(c['colliders']) in good_pairs-bad_pairs and knife.on_edge(c['position'],edge)
                and abs(np.dot(np.asarray(c['position'])-anchors[1],axis))<=.003)
            c['same_verified_edge_contact_patch']=bool(allowed)
            if not allowed:unwanted+=float(np.linalg.norm(c['impulse']))/dt
        separation=min((c['separation_m'] for c in contacts.rows),default=0.)
        velocity=np.asarray(wrist.get_velocities()[0,:3],float)
        speed=float(np.linalg.norm(velocity))
        pose_speed=float(np.dot(actual[:3,3]-previous_wrist,direction)/dt)
        previous_wrist=actual[:3,3].copy()
        gap=float(np.linalg.norm(anchors[1]-anchors[0]))
        angular_error=joint_angular_error(frames,fixture['local_rotations'])
        intact=bool(fixture['joint'].GetJointEnabledAttr().Get())
        safe=loading_sample_safe(total,separation,speed,gap,intact,angular_error)
        relative=float(np.dot(edge[:3,3]-anchors[1],direction))
        transverse=abs(float(np.dot(direction,axis)))<.3 and abs(float(np.dot(edge[:3,1],axis)))<.3
        record=dict(t=stamp.simulation_time_s,dt=dt,commanded_slide_velocity_m_s=command,
            configured_drive_force_limit_n=force_limit_n,actual_drive_force_measured=False,
            native_wrist=actual.tolist(),native_edge=edge.tolist(),native_coupon_frames=frames.tolist(),
            joint_anchors_world_m=anchors.tolist(),joint_anchor_gap_m=gap,
            joint_angular_error_rad=angular_error,
            edge_force_n=force,unwanted_force_n=unwanted,total_contact_magnitude_n=total,
            normal_contact_magnitude_n=normal_magnitude,friction_anchor_magnitude_n=friction_magnitude,
            native_linear_velocity_world_m_s=velocity.tolist(),native_slide_velocity_m_s=float(np.dot(velocity,direction)),
            pose_derived_slide_velocity_m_s=pose_speed,friction_anchors=contacts.friction,
            minimum_contact_separation_m=separation,relative_advance_coordinate_m=relative,
            speed_m_s=speed,contacts=contacts.rows,seam_enabled=intact,safety_guards_pass=safe,
            **window.sample(dt,force,relative,bool(eligible) and transverse and safe))
        records.append(record)
        # Keep the same contact-force/penetration scale; this diagnostic never
        # relaxes a benchmark guard to manufacture sufficient displacement.
        if not safe:
            raise RuntimeError('Native loading force/penetration/speed/intact-seam guard')
    try:
        for _ in range(round(seconds*physics_hz)):clock.tick(before=before,after=after)
    except Exception as error:fault=str(error)
    finally:
        fixture['drive'].GetTargetVelocityAttr().Set(0.)
        fixture['drive'].GetMaxForceAttr().Set(0.)
        contacts.subscription=None
    return dict(state='completed_loading_diagnostic_not_cut_validation' if fault is None else 'failed_loading_diagnostic',
        error=fault,measurements=window.report(),parameters=asdict(window.parameters),records=records,timing=clock.report(),
        contact_reporting='full_native_points_and_friction_anchors',
        total_load_definition='sum_normal_point_and_friction_anchor_magnitudes_upper_bound',
        physical_cut_verified=False,robot_grasp_verified=False,seam_release_authorized=False,
        tissue_fracture_calibrated=False,training_eligible=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--force-limit-n',type=float,default=.35);p.add_argument('--seconds',type=float,default=8.)
    p.add_argument('--normal-sign',type=int,choices=(-1,1),default=1)
    p.add_argument('--contact-stiffness-n-m',type=float,choices=(0.,250.,500.,1000.,2000.),default=0.)
    p.add_argument('--physics-hz',type=int,choices=(240,480,960),default=240)
    a=p.parse_args(argv)
    slide_parameters(a.force_limit_n)
    if not np.isfinite(a.seconds) or not 4<=a.seconds<=12:raise ValueError('Bounded 4..12 second diagnostic required')
    output=a.output.resolve()
    if output.exists():raise ValueError('New diagnostic output required')
    from sim_data.audit import DEFAULT_PACK
    if output.is_relative_to(DEFAULT_PACK.resolve()):raise ValueError('Do not write inside source package')
    output.mkdir(parents=True)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'multi_gpu':False,'sync_loads':False})
    result=None;exit_code=2;previous_threads=None;phase='startup'
    try:
        import omni.usd
        import carb.settings
        from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf
        from isaacsim.core.api import SimulationContext
        from sim_data.depth_preview import sha256
        settings=carb.settings.get_settings();previous_threads=settings.get('/persistent/physics/numThreads')
        settings.set_int('/persistent/physics/numThreads',1)
        stage=omni.usd.get_context().get_stage();UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
        phase='author_source_coupon';print('BLADE_LOADING_PHASE '+phase,flush=True)
        fixture=author(stage,force_limit_n=a.force_limit_n,normal_sign=a.normal_sign,
            contact_stiffness_n_m=a.contact_stiffness_n_m)
        physics=UsdPhysics.Scene.Define(stage,'/World/LoadingPhysics')
        physics.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));physics.CreateGravityMagnitudeAttr(9.81)
        api=PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
        api.CreateSolverTypeAttr('PGS');api.CreateEnableGPUDynamicsAttr(False);api.CreateBroadphaseTypeAttr('MBP')
        api.CreateFrictionTypeAttr('patch')
        sim=SimulationContext(physics_dt=1/a.physics_hz,rendering_dt=1/60,physics_prim_path='/World/LoadingPhysics',
            stage_units_in_meters=1,backend='numpy',set_defaults=False)
        phase='native_reset';print('BLADE_LOADING_PHASE '+phase,flush=True)
        sim.get_physics_context().enable_fabric(True);sim.reset()
        scene_config=dict(solver=api.GetSolverTypeAttr().Get(),gpu_dynamics=api.GetEnableGPUDynamicsAttr().Get(),
            dt_s=sim.get_physics_dt(),gravity_m_s2=physics.GetGravityMagnitudeAttr().Get(),
            gravity_direction=list(physics.GetGravityDirectionAttr().Get()),friction_type=api.GetFrictionTypeAttr().Get())
        if (scene_config['solver']!='PGS' or scene_config['gpu_dynamics']
                or not np.isclose(scene_config['dt_s'],1/a.physics_hz) or not np.isclose(scene_config['gravity_m_s2'],9.81)
                or not np.allclose(scene_config['gravity_direction'],[0,0,-1]) or scene_config['friction_type']!='patch'):
            raise RuntimeError('Native scene configuration changed: '+json.dumps(scene_config))
        phase='native_loading';print('BLADE_LOADING_PHASE '+phase,flush=True)
        result=run(sim,fixture,seconds=a.seconds,force_limit_n=a.force_limit_n,physics_hz=a.physics_hz)
        result['native_scene']=scene_config
        result['fixture']=fixture['metadata'];result['source_assets_sha256']=fixture['source_hashes']
        result['source_assets_unchanged']=all(sha256(path)==digest for path,digest in fixture['source_hashes'].items())
        if not result['source_assets_unchanged']:raise RuntimeError('Source asset changed')
        with (output/'trace.jsonl').open('x',encoding='utf-8') as stream:
            for row in result.pop('records'):stream.write(json.dumps(row,allow_nan=False)+'\n')
        result['trace_sha256']=sha256(output/'trace.jsonl')
        with (output/'report.json').open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2,allow_nan=False)
        print('BLADE_LOADING_RESULT '+json.dumps({k:result[k] for k in
            ('state','error','measurements','trace_sha256','source_assets_unchanged')},allow_nan=False),flush=True)
        sim.stop();exit_code=0 if result['error'] is None else 2
    except Exception as error:
        failure=dict(state='failed_loading_setup_or_publication',phase=phase,error=str(error),
            traceback=traceback.format_exc(),physical_cut_verified=False,training_eligible=False)
        print('BLADE_LOADING_FAILURE '+json.dumps(failure),flush=True)
        with (output/'failure.json').open('x',encoding='utf-8') as stream:json.dump(failure,stream,indent=2)
    finally:
        if previous_threads is not None:settings.set_int('/persistent/physics/numThreads',previous_threads)
        app.close(exit_code=exit_code)


if __name__=='__main__':main()
