"""Small engine regression to separate joint-drive behavior from plant assets."""
import json
import math
from pathlib import Path
import argparse


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-dynamics',action='store_true')
    parser.add_argument('--physics-hz',type=int,default=240)
    parser.add_argument('--solver',choices=('PGS','TGS'),default='TGS')
    args=parser.parse_args()
    args.output=args.output.resolve()
    if args.output.exists(): raise ValueError('New output required')
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'multi_gpu':False,'sync_loads':False})
    passed=False
    try:
        import numpy as np
        import omni.usd
        from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf,Sdf
        from isaacsim.core.api import SimulationContext
        from .plant import _body,_joint,physics_schema
        stage=omni.usd.get_context().get_stage()
        UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
        scene=UsdPhysics.Scene.Define(stage,'/World/Physics')
        scene.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));scene.CreateGravityMagnitudeAttr(9.81)
        scene_api=PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
        scene_api.CreateEnableGPUDynamicsAttr(args.gpu_dynamics)
        scene_api.CreateBroadphaseTypeAttr('GPU' if args.gpu_dynamics else 'MBP')
        scene_api.CreateSolverTypeAttr(args.solver)
        roots=[];cases=[]
        for i,(kind,scale) in enumerate([('d6',1),('d6_rotated',1),('d6_rotated_vel0',1),('d6_rotated_fixedbase',1),('d6_rotated_fixedbase_vel0',1)]):
            root=f'/World/Probe_{i}';roots.append(root+'/Base')
            UsdGeom.Xform.Define(stage,root)
            f0=np.eye(4);f0[:3,3]=[i*.4,0,1]
            f1=f0.copy();f1[0,3]+=.05
            if kind.startswith('d6_rotated') or kind=='revolute_rotated':
                a=math.pi/4
                f1[:3,:3]=[[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]]
            if kind=='d6_x':
                f1[:3,3]=f0[:3,3]+[0,.05,0]
            properties=dict(mass=.001*scale,inertia=np.array([1e-8,1e-6,1e-6])*scale,
                usd_stiffness=np.full(3,.1*scale*math.pi/180),
                usd_damping=np.full(3,.001*scale*math.pi/180))
            if kind=='d6_rotated_iso': properties['inertia']=np.full(3,1e-6)
            base=_body(stage,roots[-1],f0,properties,kinematic=False)
            if kind=='d6_joint_rotated':
                a=math.pi/4
                properties['principal_axes']=np.array([[math.cos(a),-math.sin(a),0],[math.sin(a),math.cos(a),0],[0,0,1]])
            _body(stage,root+'/Tip',f1,properties,kinematic=False)
            UsdPhysics.ArticulationRootAPI.Apply(base)
            if kind.endswith('vel0'):
                physics_schema(base,'PhysxArticulationAPI',[
                    ('physxArticulation:sleepThreshold',Sdf.ValueTypeNames.Float,0.),
                    ('physxArticulation:solverPositionIterationCount',Sdf.ValueTypeNames.Int,32),
                    ('physxArticulation:solverVelocityIterationCount',Sdf.ValueTypeNames.Int,0)])
            if kind=='d6_rotated_awake':
                physics_schema(base,'PhysxArticulationAPI',[
                    ('physxArticulation:sleepThreshold',Sdf.ValueTypeNames.Float,0.),
                    ('physxArticulation:stabilizationThreshold',Sdf.ValueTypeNames.Float,0.)])
            if kind=='d6_rotated_highiter':
                physics_schema(base,'PhysxArticulationAPI',[
                    ('physxArticulation:solverPositionIterationCount',Sdf.ValueTypeNames.Int,128),
                    ('physxArticulation:solverVelocityIterationCount',Sdf.ValueTypeNames.Int,32)])
            fixed=UsdPhysics.FixedJoint.Define(stage,root+'/WorldAnchor')
            fixed.CreateBody1Rel().SetTargets([roots[-1]])
            fixed.CreateLocalPos0Attr(Gf.Vec3f(*f0[:3,3]));fixed.CreateLocalPos1Attr(Gf.Vec3f(0))
            if kind=='d6_external':
                support=f'/World/Support_{i}'
                _body(stage,support,f0,properties,kinematic=True)
                fixed.CreateBody0Rel().SetTargets([support])
                fixed.CreateLocalPos0Attr(Gf.Vec3f(0))
                fixed.CreateExcludeFromArticulationAttr(True)
            if kind.startswith('d6'):
                joint=_joint(stage,root+'/Joint',roots[-1],root+'/Tip',f0[:3,3],np.array([f0,f1]),properties,external=False)
                if kind=='d6_joint_rotated':
                    quaternion=Gf.Quatf(Gf.Rotation(Gf.Vec3d(0,0,1),45).GetQuat())
                    joint.CreateLocalRot0Attr(quaternion);joint.CreateLocalRot1Attr(quaternion)
            else:
                joint=UsdPhysics.RevoluteJoint.Define(stage,root+'/Joint')
                joint.CreateBody0Rel().SetTargets([roots[-1]]);joint.CreateBody1Rel().SetTargets([root+'/Tip'])
                joint.CreateLocalPos0Attr(Gf.Vec3f(0));joint.CreateLocalPos1Attr(Gf.Vec3f(-.05,0,0))
                joint.CreateAxisAttr('Y')
                drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),'angular')
                drive.CreateTypeAttr('force');drive.CreateStiffnessAttr(float(properties['usd_stiffness'][0]))
                drive.CreateDampingAttr(float(properties['usd_damping'][0]));drive.CreateTargetPositionAttr(0)
                if kind=='revolute_rotated':
                    local=np.linalg.inv(f1)@np.r_[f0[:3,3],1]
                    joint.CreateLocalPos1Attr(Gf.Vec3f(*map(float,local[:3])))
                    joint.CreateLocalRot1Attr(Gf.Quatf(Gf.Rotation(Gf.Vec3d(0,0,1),-45).GetQuat()))
            if kind.startswith('d6_rotated_fixedbase'):
                base.RemoveAPI(UsdPhysics.ArticulationRootAPI)
                UsdPhysics.ArticulationRootAPI.Apply(fixed.GetPrim())
                roots[-1]=str(fixed.GetPath())
                physics_schema(fixed.GetPrim(),'PhysxArticulationAPI',[
                    ('physxArticulation:sleepThreshold',Sdf.ValueTypeNames.Float,0.)])
                if kind.endswith('vel0'):
                    physics_schema(fixed.GetPrim(),'PhysxArticulationAPI',[
                        ('physxArticulation:solverPositionIterationCount',Sdf.ValueTypeNames.Int,32),
                        ('physxArticulation:solverVelocityIterationCount',Sdf.ValueTypeNames.Int,0)])
            cases.append(dict(kind=kind,mass_scale=scale,expected_angle_rad=.001*9.81*.05/.1))
        sim=SimulationContext(physics_dt=1/args.physics_hz,rendering_dt=1/60,physics_prim_path='/World/Physics',backend='numpy',set_defaults=False)
        sim.reset()
        effective_scene=dict(physics_dt=sim.get_physics_dt(),
            gpu_dynamics=scene_api.GetEnableGPUDynamicsAttr().Get(),
            solver=scene_api.GetSolverTypeAttr().Get())
        if effective_scene['gpu_dynamics']!=args.gpu_dynamics:
            raise RuntimeError('Context changed requested GPU dynamics; comparison is invalid')
        if effective_scene['solver']!=args.solver:
            raise RuntimeError('Context changed requested solver; comparison is invalid')
        views=[sim.physics_sim_view.create_articulation_view(root) for root in roots]
        history=[];tail_velocities=[]
        for step in range(3*args.physics_hz):
            sim.step(render=False)
            if step>=int(2.5*args.physics_hz):
                tail_velocities.append([np.asarray(view.get_dof_velocities()).copy() for view in views])
            if step%24==0:
                history.append([np.asarray(view.get_dof_positions()).tolist() for view in views])
        for index,(case,view) in enumerate(zip(cases,views,strict=True)):
            case['positions']=np.asarray(view.get_dof_positions()).tolist()
            case['native_stiffness']=np.asarray(view.get_dof_stiffnesses()).tolist()
            case['native_damping']=np.asarray(view.get_dof_dampings()).tolist()
            case['velocity']=np.asarray(view.get_dof_velocities()).tolist()
            case['body_transforms']=np.asarray(view.get_link_transforms()).tolist()
            case['body_velocities']=np.asarray(view.get_link_velocities()).tolist()
            case['angle_error_rad']=float(abs(np.linalg.norm(case['positions'])-case['expected_angle_rad']))
            case['tail_velocity_rms_rad_s']=float(np.sqrt(np.mean([
                np.sum(sample[index]**2) for sample in tail_velocities])))
            case['passed']=bool(case['angle_error_rad']<.001 and case['tail_velocity_rms_rad_s']<.01)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(dict(passed=all(case['passed'] for case in cases),
            effective_scene=effective_scene,cases=cases,history=history),indent=2),encoding='utf-8')
        passed=all(case['passed'] for case in cases)
        print(json.dumps(cases),flush=True)
        sim.stop()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close(exit_code=0 if passed else 2)


if __name__=='__main__':
    main()
