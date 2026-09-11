"""Standalone native runner for the small spring/contact coupon, not a robot task."""
import argparse
import json
from pathlib import Path
import traceback
import time

import numpy as np

from .contact_events import ContactEvents
from .contact_spring_probe import (MAX_CONTACT_ROWS,MODELS,SECTION_MODELS,from_report,author,bind,bind_rigid,
                                  settings_readback,sample,assess_tail)
from .runtime import pose_matrices


def finite_native(value,shape,label):
    """Validate before adding native arrays to strict JSON evidence."""
    result=np.array(value,dtype=float,copy=True)
    if result.shape!=shape or not np.isfinite(result).all():
        raise RuntimeError('Invalid native '+label+'; expected finite '+str(shape))
    return result


def read_state(view,*,maximal):
    """Keep body-derived maximal coordinates distinct from native DOF sensors."""
    if maximal:
        from .contact_spring_probe import rigid_coordinates
        poses=finite_native(view.get_transforms(),(3,7),'rigid body poses')
        velocity=finite_native(view.get_velocities(),(3,6),'rigid body velocities')
        frames=pose_matrices(poses)
        q,qdot,_=rigid_coordinates(frames,velocity)
    else:
        poses=finite_native(view.get_link_transforms(),(1,3,7),'articulation poses')[0]
        frames=pose_matrices(poses)
        velocity=finite_native(view.get_link_velocities(),(1,3,6),'articulation velocities')[0]
        q=finite_native(view.get_dof_positions(),(1,2),'joint positions')[0]
        qdot=finite_native(view.get_dof_velocities(),(1,2),'joint velocities')[0]
    return frames,velocity,q,qdot


class ContactRows(ContactEvents):
    """Copy full signed original-order rows, including each friction anchor once."""
    def begin_step(self):
        super().begin_step()
        self.rows=[]

    def consume(self,first,second,impulses,points=None,normals=None,separations=None,
                *,friction_impulses=(),friction_points=None):
        impulses=tuple(impulses);friction_impulses=tuple(friction_impulses)
        super().consume(first,second,impulses,points,normals,separations,
                       friction_impulses=friction_impulses,friction_points=friction_points)
        for kind,values,positions in (('normal',impulses,points),
                                      ('friction',friction_impulses,friction_points)):
            if values and (positions is None or len(positions)!=len(values)):
                raise ValueError('Full signed contact points required')
            if kind=='normal' and values and (normals is None or len(normals)!=len(values)):
                raise ValueError('Full original-order normals required')
            for i,(impulse,point) in enumerate(zip(values,() if positions is None else positions,strict=True)):
                if len(self.rows)>=MAX_CONTACT_ROWS:raise RuntimeError('Contact row overflow')
                row=dict(collider0=first,collider1=second,kind=kind,
                    point_world_m=list(point),impulse_on_0_ns=list(impulse))
                if kind=='normal':row['normal_on_0']=list(normals[i])
                self.rows.append(row)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source-report',type=Path,required=True)
    p.add_argument('--model',choices=MODELS,default='native')
    p.add_argument('--solver',choices=('PGS','TGS'),default='PGS')
    p.add_argument('--physics-hz',type=int,choices=(240,480),default=240)
    p.add_argument('--iterations',choices=('16/4','32/0'),default='16/4')
    p.add_argument('--rotation-degrees',type=int,choices=(0,45),default=0)
    p.add_argument('--seconds',type=float,default=3.)
    p.add_argument('--free-control',action='store_true',help='No pads: separately labelled unloaded control')
    p.add_argument('--zero-joint-friction',action='store_true',
        help='Explicit bearing-friction diagnostic via legacy native coefficient setter; contact friction unchanged')
    args=p.parse_args(argv)
    maximal=args.model=='section_springs_maximal'
    if maximal and args.zero_joint_friction:
        raise ValueError('Articulation joint-friction setter unavailable for maximal model')
    if not np.isfinite(args.seconds) or not 1<=args.seconds<=10:
        raise ValueError('Bounded 1..10 second coupon required')
    out=args.output.resolve()
    if out.exists():raise ValueError('New diagnostic output directory required')
    angle=np.radians(args.rotation_degrees);c,s=np.cos(angle),np.sin(angle)
    coupon=from_report(args.source_report,rotation=((c,-s,0),(s,c,0),(0,0,1)),
        iterations=tuple(map(int,args.iterations.split('/'))),dt=1/args.physics_hz,solver=args.solver,
        held_contacts=not args.free_control)
    from isaacsim import SimulationApp
    app=SimulationApp({'headless':True,'multi_gpu':False,'sync_loads':False})
    result=dict(state='failed_small_contact_spring_coupon',configuration=coupon.report(),
        comparison_model=args.model,training_eligible=False,error=None)
    rows=[];monitor=None;errors=None;sim=None
    out.mkdir(parents=True)
    try:
        import omni.usd
        import carb
        from pxr import PhysxSchema,UsdPhysics,Usd,UsdUtils
        from omni.physx import get_physx_simulation_interface
        import omni.physics.tensors as tensors
        from .native_errors import NativeErrors
        errors=NativeErrors()
        # Kit creates camera/render prims even on new_stage(). Author on a
        # genuinely empty owned layer, then attach it to THIS new process.
        context=omni.usd.get_context();stage=Usd.Stage.CreateInMemory()
        data=author(stage,coupon,model=args.model)
        cache=UsdUtils.StageCache.Get();stage_id=cache.Insert(stage).ToLongInt()
        attached=[]
        if not context.attach_stage_with_callback(stage_id=stage_id,
                on_finish_fn=lambda ok,message:attached.append((ok,message))):
            raise RuntimeError('Could not attach standalone coupon stage')
        deadline=time.monotonic()+30
        while not attached:
            if time.monotonic()>deadline or not app.is_running():
                raise RuntimeError('Standalone stage attachment timed out')
            app.update()
        if attached[0][0] is not True or context.get_stage()!=stage:
            raise RuntimeError('Standalone coupon stage attachment mismatch: '+str(attached))
        carb.settings.get_settings().set('/plugins/carb.tasking.plugin/threadCount',1)
        scene=stage.GetPrimAtPath(data['scene_path']);api=PhysxSchema.PhysxSceneAPI(scene)
        UsdPhysics.Scene(scene).CreateGravityMagnitudeAttr(0.)
        api.CreateSolverTypeAttr(coupon.solver);api.CreateEnableGPUDynamicsAttr(False)
        api.CreateBroadphaseTypeAttr('MBP');api.CreateFrictionTypeAttr('patch')
        # SimulationManager.initialize_physics() advances two warm-up steps.
        # This standalone probe owns the raw interface instead: parse only,
        # bind/verify the initial state, then explicitly simulate/fetch each dt.
        sim=get_physx_simulation_interface()
        if not sim.attach_stage(stage_id):raise RuntimeError('Native stage parse failed')
        if sim.get_attached_stage()!=stage_id:raise RuntimeError('Unexpected native stage ownership')
        errors.check('after_native_attach_before_any_step')
        monitor=ContactRows(robot_root=coupon.root,target_root=coupon.root,fingers=[],floor_root=None)
        monitor.subscribe()
        # PhysX defers insertion of articulation actors until its first solve.
        # Use one EXPLICIT, reported 1 us bootstrap for articulated models; retain
        # its signed contacts and require the same strict initial-q tolerance.
        # This is neither zero elapsed time nor a hidden Context warm-up/reset.
        # Independent rigid actors may be queried directly after parsing.
        # Unlike articulation insertion, they need not share its bootstrap.
        # Bind the maximal initial state BEFORE any solve, never relax its
        # initial-strain tolerance to compensate for bootstrap motion.
        bootstrap_dt=0. if maximal else 1e-6
        monitor.begin_step()
        if bootstrap_dt:
            sim.simulate(bootstrap_dt,0.);sim.fetch_results()
            monitor.measurements(bootstrap_dt)
        errors.check('after_explicit_bootstrap_or_maximal_parse')
        result['bootstrap_contact_rows']=[dict(row) for row in monitor.rows]
        result['settings']=settings_readback(stage,coupon,actual_dt=coupon.dt,model=args.model)
        result['stepping']=dict(owner='raw_physx_simulate_fetch',startup_physics_steps=int(bool(bootstrap_dt)),
            startup_physics_dt_s=bootstrap_dt,
            startup_solver_model=(None if maximal else 'section_springs_angular_drives_zero'
                                  if args.model in SECTION_MODELS else 'native_angular_drives'),
            simulation_context_reset_used=False,dt_source='explicit_simulate_argument')
        simulation_view=tensors.create_simulation_view('numpy',stage_id=stage_id)
        simulation_view.set_subspace_roots('/')
        view=(simulation_view.create_rigid_body_view(coupon.root+'/Link_*') if maximal else
              simulation_view.create_articulation_view(data['body_paths'][0]))
        frames,velocity,q,qdot=read_state(view,maximal=maximal)
        result['initial_readback']=dict(q=q.tolist(),qdot=qdot.tolist(),frames=frames.tolist(),
            native_body_velocities=velocity.tolist(),
            coordinate_source='derived_from_native_bodies' if maximal else 'native_articulation_dofs')
        if maximal:
            result['binding'],predictor=bind_rigid(view,coupon,stage=stage)
            result['joint_friction']=dict(native_readback=None,setter=None,
                reason='No articulation DOFs in maximal model',contact_friction_unchanged=True)
        else:
            result['binding'],predictor=bind(view,coupon,model=args.model,stage=stage)
            before_friction=finite_native(view.get_dof_friction_coefficients(),(1,2),'joint friction before')
            if args.zero_joint_friction:
                view.set_dof_friction_coefficients(np.zeros_like(before_friction),np.array([0],dtype=np.uint32))
            after_friction=finite_native(view.get_dof_friction_coefficients(),(1,2),'joint friction after')
            friction_properties=finite_native(view.get_dof_friction_properties(),(1,2,3),'friction properties')
            result['joint_friction']=dict(before=before_friction.tolist(),after=after_friction.tolist(),
                zero_requested=args.zero_joint_friction,contact_friction_unchanged=True,
                setter='legacy_dof_friction_coefficients' if args.zero_joint_friction else None,
                native_new_friction_properties=friction_properties.tolist(),
                setter_timing='after_shared_explicit_bootstrap_before_recorded_steps')
            if args.zero_joint_friction and np.any(after_friction!=0):
                raise RuntimeError('Native joint friction readback mismatch')
        for step in range(1,int(args.seconds*args.physics_hz)+1):
            if not app.is_running():raise RuntimeError('App closed before diagnostic completion')
            monitor.begin_step();errors.check('before_step')
            commanded_effort=None
            if predictor is not None:commanded_effort=predictor.step(coupon.dt,root_constrained=False)
            sim.simulate(coupon.dt,bootstrap_dt+(step-1)*coupon.dt);sim.fetch_results()
            errors.check('after_step');monitor.measurements(coupon.dt)
            frames,velocity,q,qdot=read_state(view,maximal=maximal)
            row=sample(coupon,step_id=step,frames=frames,velocities=velocity,q=q,qdot=qdot,
                contact_rows=monitor.rows,full_normal_friction_stream=monitor.native_full_contact_reporting,
                model=args.model)
            rows.append(row)
            # Projected incoming reaction includes more than drive torque;
            # do not report it as independently measured elastic effort.
            row['native_projected_incoming_joint_force_nm']=(None if maximal else finite_native(
                view.get_dof_projected_joint_forces(),(1,2),'projected incoming force')[0].tolist())
            row['explicit_predictor_effort_nm']=(None if commanded_effort is None else
                finite_native(commanded_effort,(2,),'predictor effort').tolist())
            if (np.max(np.abs(row['q_rad']))>=.05
                    or max(row['per_body_contact_upper_bound_n'])>1.
                    or np.max(np.linalg.norm(np.array(row['body_velocities_world'])[:,:3],axis=1))>1.):
                raise RuntimeError('Coupon small-angle/speed/contact qualification guard')
            if step%args.physics_hz==0:
                print('CONTACT_SPRING_SECOND '+json.dumps(dict(step=step,q=row['q_rad'],
                    contact_moments=row['contact_joint_moments_nm'],elastic=row['spring_kq_nm'])),flush=True)
        result['assessment']=assess_tail(rows[-args.physics_hz//2-1:],coupon,model=args.model,
            whole_run_samples=rows)
        tail=rows[-args.physics_hz//2-1:]
        integrated=np.diff([row['q_rad'] for row in tail],axis=0)/coupon.dt
        carried=np.array([row['qdot_rad_s'] for row in tail])
        result['velocity_diagnostics']=dict(
            integrated_joint_rate_rms_rad_s=np.sqrt(np.mean(integrated**2,axis=0)).tolist(),
            carried_joint_velocity_rms_rad_s=np.sqrt(np.mean(carried**2,axis=0)).tolist(),
            integrated_rate_replaces_native_velocity=False)
        errors.check('before_acceptance')
        if result['assessment']['passed']:result['state']='passed_small_contact_spring_coupon_not_plant'
    except Exception:
        result['error']=traceback.format_exc()
    finally:
        if monitor is not None:monitor.close()
        if sim is not None:sim.detach_stage()
        if errors is not None:
            try:errors.__exit__(None,None,None)
            except Exception:
                result['error']=(result['error'] or '')+'\n'+traceback.format_exc()
                result['state']='failed_small_contact_spring_coupon'
            finally:result['native_errors']=errors.report()
        result['sample_count']=len(rows)
        result['whole_run_max_abs_q_rad']=float(max((max(map(abs,r['q_rad'])) for r in rows),default=0.))
        (out/'trace.json').write_text(json.dumps(rows,allow_nan=False),encoding='utf-8')
        (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        print('CONTACT_SPRING_RESULT '+json.dumps(result),flush=True)
        app.close()
    return 0 if result['state'].startswith('passed_') else 2


if __name__=='__main__':raise SystemExit(main())
