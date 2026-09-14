"""Bounded, reobserved post-cut egress; no release or native pose assignment.

Every proposal retains tool orientation. A nearly unloaded existing cut-face
proximity may escape only along increasing conservative separation bounds.
All other pairs, and the final endpoint, use normal transit clearance.
The caller must continue native force/contact/retention guards during motion.
This is a frozen-scene sampled plan, not continuous collision certification.
"""
from copy import copy
import numpy as np

from .withdrawal_native_check import _robot_world, _screens
from .measured_withdrawal_native import _raw_joint_world, RIGHT, LEFT
from .measured_withdrawal import _attained


def offsets(wrist):
    """At most 45 small egress proposals; never a new downward cut stroke."""
    from .withdrawal_evidence import _pose
    f=_pose(wrist)
    directions=[np.array([0.,0.,1.])]
    for axis in (*np.eye(3)[:2], f[:3,0], f[:3,2]):
        horizontal=np.array(axis,copy=True);horizontal[2]=0.
        length=float(np.linalg.norm(horizontal))
        if length<1e-9:continue
        for sign in (1.,-1.):directions.append(sign*horizontal/length)
    return [distance*d for distance in (.002,.004,.008,.016,.03) for d in directions]


def joint_samples(start,end):
    a,b=np.asarray(start,float),np.asarray(end,float)
    if a.shape!=(7,) or b.shape!=(7,) or not np.isfinite([a,b]).all():
        raise ValueError('Two finite seven-joint configurations required')
    count=max(2,int(np.ceil(float(np.max(abs(b-a)))/.25))+1)
    return a+np.linspace(0,1,count)[:,None]*(b-a)


def prerequisites(record, receipt, step):
    if (type(step) is not int or step<1 or type(receipt) is not dict
            or receipt.get('step')!=step or receipt.get('complete') is not True
            or receipt.get('cut_face_and_support_verified') is not True
            or receipt.get('cut_authorized') is not False
            or receipt.get('commanded_motion_used_as_completion') is not False
            or record.get('cut') is not True or record.get('native_guards_passed') is not True):
        raise ValueError('Fresh measured unloaded reverse and native guards required')
    for load in (receipt.get('full_knife_load_n'),record.get('knife',{}).get('tool_contact_upper_bound_n')):
        if isinstance(load,(bool,np.bool_)) or not isinstance(load,(int,float)) or not np.isfinite(load) or not 0<=load<.01:
            raise ValueError('Measured unloaded full tool required before egress')


def drive_handoff(previous,planned_start):
    """Report the drive-reference discontinuity; no motion or safety authority.

    A measured pose can differ from the still-active position target. Preserve
    both to diagnose recontact instead of silently compensating that error or
    raising the unloaded force limit. This does not establish causality.
    """
    a,b=np.asarray(previous,float),np.asarray(planned_start,float)
    if a.shape!=(7,) or b.shape!=(7,) or not np.isfinite([a,b]).all():
        raise ValueError('Finite previous and planned seven-joint targets required')
    return dict(model='measured_egress_drive_handoff_diagnostic_v1',
        previous_drive_degrees=a.tolist(),planned_start_degrees=b.tolist(),
        start_minus_previous_degrees=(b-a).tolist(),
        maximum_absolute_reference_change_degrees=float(np.max(abs(b-a))),
        drive_compensation_applied=False,motion_authorized=False,
        contact_cause_established=False)


def plan(f, frames, *, step, record, retraction):
    prerequisites(record,retraction,step)
    if f.rig.cut is not True or f.plan is None:
        raise ValueError('Existing measured release and cut plan required')
    q0=np.array(f.robot.get_dof_positions(),dtype=float,copy=True)
    targets=np.array(f.robot.get_dof_position_targets(),copy=True)
    names=tuple(f.names);world=_robot_world(f)
    values=dict(zip(names,q0[0],strict=True))
    predicted=_raw_joint_world(f.kin,values,world['base'])
    if any(k not in predicted or not _attained(v,predicted[k]) for k,v in world.items()):
        raise ValueError('Full current native robot/FK mismatch; no egress')
    actual=world['ee_right'];start=np.degrees([values[n] for n in RIGHT])
    handoff=drive_handoff(np.degrees(targets[0,[names.index(n) for n in RIGHT]]),start)
    left_q=np.degrees([values[n] for n in LEFT])
    low,high=f.kin.arm_limits_degrees('right')
    if np.any(start<=low) or np.any(start>=high):raise ValueError('In-limit measured start required')
    # Correct only the rigid upstream station for the nominal arm IK. Every
    # solved pose is independently checked using ALL measured SI joint values.
    nominal=f.kin.forward('right',start,f.base)
    correction=actual@np.linalg.inv(nominal)
    inverse=np.linalg.inv(correction)
    right,left=_screens(f,frames)
    checks=copy(f.self_screen)
    evidence=dict(model='fresh_unloaded_postcut_egress_v1',step=step,attempts=[],
        passed=False,clearance_margin_m=.001,self_margin_m=.003,interarm_margin_m=.01,
        cut_face_allowance=False,source_assets_changed=False,pose_overwrite=False,
        motion_authorized=False,whole_path_certified=False,training_eligible=False,
        drive_handoff=handoff)
    f.postcut_egress_evidence=evidence
    from .native_static_clearance import current_scene_query
    native=None;path=None;escape=None
    def fresh():
        current=_robot_world(f)
        if (not np.array_equal(f.robot.get_dof_positions(),q0)
                or not np.array_equal(f.robot.get_dof_position_targets(),targets)
                or tuple(f.names)!=names
                or set(current)!=set(world)
                or any(not np.array_equal(v,current[k]) for k,v in world.items())):
            raise RuntimeError('Native state changed during post-cut egress planning')
    def body(q):
        v=dict(values);v.update(zip(RIGHT,np.radians(q),strict=True))
        return _raw_joint_world(f.kin,v,world['base'])
    def clear(w,q,*,escaping=False):
        if not checks.check(w)['passed']:return 'self_clearance'
        if f.kin.inter_arm_clearance(left_q,q,correction@f.base).clearance_m<.01:return 'interarm_clearance'
        if not right.check(w,stroke=escaping):return dict(right=right.last_failure)
        if not left.check(w,grasp=getattr(f,'cut_strategy','bimanual')=='bimanual',stroke=False):
            return dict(left=left.last_failure)
        return None
    try:
        options={}
        if getattr(f,'planning_heartbeat',None) is not None:options['heartbeat']=f.planning_heartbeat
        native=current_scene_query(f.stage,right.static,
            capsule_sphere_cover=bool(getattr(f,'native_capsule_sphere_cover',False)),**options)
        right.native_static_query=native;left.native_static_query=native
        initial=clear(world,start)
        if initial is not None:
            failure=initial.get('right',{}) if isinstance(initial,dict) else {}
            if (failure.get('robot_collider')!=getattr(right,'blade_path',None)
                    or failure.get('plant_collider') not in getattr(right,'seam_paths',())):
                evidence['rejection']=dict(current_start_not_clear=initial)
                raise RuntimeError('Post-cut egress start is not clear at the original margins')
            from .seam_escape import bind,gaps,monotone
            escape=bind(right);initial_gaps=gaps(escape,world)
            evidence.update(cut_face_allowance=True,escape_start_gaps_m=initial_gaps,
                escape_execution_load_limit_n=.01,escape_endpoint_full_margin_required=True)
            # IK/FK proposals use one consistent exact-source frame. The
            # separately checked native start can differ by tensor precision;
            # do not mistake that change of representation for robot motion.
            initial_gaps=gaps(escape,body(start))
            evidence['escape_planning_start_gaps_m']=initial_gaps
            remainder=clear(world,start,escaping=True)
            if remainder is not None:
                evidence['rejection']=dict(nonseam_start_not_clear=remainder)
                raise RuntimeError('Post-cut escape cannot ignore another collision pair')
        for delta in offsets(actual):
            attempt=dict(displacement_world_m=delta.tolist());evidence['attempts'].append(attempt)
            desired=actual.copy();desired[:3,3]+=delta
            solved=f.kin.solve_pose('right',inverse@desired,start,f.base,
                maximum_evaluations=200,joint_limit_margin_degrees=3.)
            attempt['ik_succeeded']=solved.succeeded
            if not solved.succeeded:continue
            end=np.asarray(solved.joint_degrees,float);endpoint=body(end)
            if not _attained(desired,endpoint['ee_right']):
                attempt['rejection']='actual_station_endpoint_error';continue
            trial=joint_samples(start,end);failure=None;previous_gaps=initial_gaps if escape is not None else None
            for i,q in enumerate(trial):
                if np.any(q<=low) or np.any(q>=high):failure='joint_limits';break
                w=body(q)
                expected=actual.copy();expected[:3,3]+=(i/(len(trial)-1))*delta
                if not _attained(expected,w['ee_right']):failure='nonrectilinear_wrist_path';break
                if escape is not None:
                    current=gaps(escape,w)
                    if not monotone(previous_gaps,current):failure='cut_face_separation_decreased';break
                    previous_gaps=current
                failure=clear(w,q,escaping=escape is not None and i<len(trial)-1)
                if failure is not None:break
            attempt['checked_samples']=i+1
            if failure is not None:attempt['rejection']=failure;continue
            path=trial;native.validate();fresh();break
        if path is None:raise RuntimeError('No bounded post-cut egress with original clearances')
    finally:
        if native is not None:
            native.close();evidence['native_static']=native.report()
        fresh()
    n=evidence['native_static'];epoch=n.get('epoch',{})
    if (n.get('final_validation_passed') is not True or n.get('closed') is not True or n.get('errors')
            or epoch.get('revision')!=0 or epoch.get('subscriptions_closed') is not True
            or epoch.get('cleanup_errors') or epoch.get('invalidation_reasons')):
        raise RuntimeError('Post-cut egress epoch/cleanup verification failed')
    duration=max(4.,float(np.max(abs(np.diff(path,axis=0))))*(len(path)-1)*1.5/8.)
    evidence.update(passed=True,path_degrees=path.tolist(),duration_s=duration,
        goal_right_degrees=path[-1].tolist(),goal_wrist_world=body(path[-1])['ee_right'].tolist(),
        measured_start_degrees=start.tolist(),measured_start_wrist_world=actual.tolist(),
        native_guards_required_during_execution=True)
    return path,evidence
