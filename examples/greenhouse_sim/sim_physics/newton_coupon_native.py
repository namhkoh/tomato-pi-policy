"""Run an owned bundled-Newton CPU diagnostic; no Kit/UI or production edits."""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np

from .contact_spring_probe import from_report, layout, rigid_coordinates, sample, assess_tail
from .host_memory import preflight
from .newton_coupon import build, check_model, contact_rows
from .qualification_exit import exit_code
from .runtime import pose_matrices


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-report',type=Path,required=True)
    parser.add_argument('--free-control',action='store_true')
    parser.add_argument('--iterations',type=int,choices=(32,128),default=32)
    parser.add_argument('--friction-epsilon',type=float,choices=(.01,.0001),default=.01,
        help='Explicit IPC low-speed friction regularization; mu/K/C are unchanged')
    args=parser.parse_args(argv)
    out=args.output.resolve()
    if out.exists():raise ValueError('New diagnostic output directory required')
    coupon=from_report(args.source_report,held_contacts=not args.free_control)
    memory=preflight()
    if not memory['allowed']:
        print(json.dumps(dict(state='blocked_host_memory',memory=memory)),flush=True)
        return 2
    out.mkdir(parents=True)
    result=dict(state='failed_newton_reference_coupon',configuration=coupon.report(),
        memory=memory,training_eligible=False,physical_cut_verified=False,
        production_qualified=False,error=None,cleanup_failures=[],
        oracle='shared maximal-coordinate small-angle Kq equilibrium/energy checks',
        contact_law_parity=False)
    rows=[]
    try:
        import warp as wp
        wp.config.kernel_cache_dir=str(out.parent/'newton_reference_warp_cache')
        wp.init()
        model,solver,shapes,receipt=build(coupon,iterations=args.iterations,friction_epsilon=args.friction_epsilon)
        result['native_readback']=receipt
        current=model.state(); following=model.state(); control=model.control(); contacts=model.contacts()
        previous=wp.clone(solver.body_q_prev)
        frames=pose_matrices(current.body_q.numpy())
        if not np.allclose(frames,layout(coupon)['frames'],rtol=0,atol=1e-7) or np.any(current.body_qd.numpy()):
            raise RuntimeError('Initial native frame/velocity mismatch')
        result['bootstrap_steps']=0
        for step in range(1,721):
            started=time.perf_counter()
            check_model(model,coupon)
            current.clear_forces()
            wp.copy(previous,solver.body_q_prev)
            model.collide(current,contacts)
            count=int(contacts.rigid_contact_count.numpy()[0])
            if count<0 or count>min(128,contacts.rigid_contact_max):
                raise RuntimeError('Contact capacity/row bound exceeded')
            solver.step(current,following,control,contacts,coupon.dt)
            b0,b1,p0,p1,force,count_array=solver.collect_rigid_contact_forces(following.body_q,previous,contacts,coupon.dt)
            if int(count_array.numpy()[0]) != count:
                raise RuntimeError('Incomplete/stale solver contact force output')
            frames=pose_matrices(following.body_q.numpy())
            velocity=following.body_qd.numpy().astype(float)
            q,qdot,_=rigid_coordinates(frames,velocity)
            fetched=contact_rows(shape0=contacts.rigid_contact_shape0.numpy()[:count],
                shape1=contacts.rigid_contact_shape1.numpy()[:count],
                points0=p0.numpy()[:count],points1=p1.numpy()[:count],
                normals=contacts.rigid_contact_normal.numpy()[:count],forces1=force.numpy()[:count],
                shapes=shapes,dt=coupon.dt)
            row=sample(coupon,step_id=step,frames=frames,velocities=velocity,q=q,qdot=qdot,
                contact_rows=fetched,full_normal_friction_stream=True,model='angular_d6_maximal')
            row.update(engine='newton_vbd_reference',native_comparison_model='Newton revolute maximal',
                shared_oracle_model='angular_d6_maximal only selects Kq/derived-state arithmetic; NOT PhysX execution',
                native_contact_count=count,wall_s=time.perf_counter()-started)
            rows.append(row)
            if max(abs(q))>=.05 or max(row['per_body_contact_upper_bound_n'])>1. or max(np.linalg.norm(velocity[:,:3],axis=1))>1.:
                raise RuntimeError('Unchanged coupon small-angle/speed/contact guard')
            current,following=following,current
            if step%240==0:
                print('NEWTON_SECOND '+json.dumps(dict(step=step,q=q.tolist(),contacts=count)),flush=True)
        result['assessment']=assess_tail(rows[-121:],coupon,model='angular_d6_maximal',whole_run_samples=rows)
        if result['assessment']['passed']:result['state']='passed_newton_reference_coupon_not_plant'
    except Exception:
        result['error']=traceback.format_exc()
    finally:
        result['sample_count']=len(rows)
        if rows:
            times=[r['wall_s'] for r in rows]
            result['timing']=dict(total_s=sum(times),p50_s=float(np.median(times)),p95_s=float(np.percentile(times,95)),
                scope='CPU coupon with per-step diagnostics; first tick includes kernel compilation')
        (out/'trace.json').write_text(json.dumps(rows,allow_nan=False),encoding='utf-8')
        (out/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        print('NEWTON_RESULT '+json.dumps(dict(state=result['state'],steps=len(rows),
            error=result['error'],assessment=result.get('assessment'))),flush=True)
    return exit_code(result,passed_state='passed_newton_reference_coupon_not_plant')


if __name__=='__main__':raise SystemExit(main())
