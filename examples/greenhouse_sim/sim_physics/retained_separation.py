"""Experimental post-release target accommodation, never a cut certificate.

Move the retained fragment at most 2 mm away from the blade's flat face. All
motion is by normal left-arm drives; the fingers stay feedback controlled.
Frozen geometry proposals are not deformation predictions. Native contact,
slip, tracking and reobserved material-section guards remain authoritative.
"""
from copy import copy
import numpy as np


def direction(edge, axis):
    from .withdrawal_evidence import _pose
    edge=_pose(edge);axis=np.asarray(axis,float)
    if axis.shape!=(3,) or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1)>1e-5:
        raise ValueError('Current unit distal shaft axis required')
    normal=np.array(edge[:3,2],copy=True)
    normal*=1. if normal@axis>=0 else -1.
    if normal@axis<.9:raise ValueError('Blade-normal separation does not point along the detachable shaft')
    return normal


def prerequisites(record,step):
    if (type(step) is not int or step<1 or record.get('native_guards_passed') is not True
            or record.get('cut') is not True or record.get('contact',{}).get('bilateral') is not True):
        raise RuntimeError('Fresh native released and bilaterally retained target required')
    values=[record.get('t'),record.get('slip_m'),
        record.get('knife',{}).get('tool_contact_upper_bound_n'),
        record.get('robot',{}).get('max_finger_contact_upper_bound_n')]
    if (any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,float)) for v in values)
            or not np.isfinite(values).all() or abs(values[0]-step/480)>1e-8
            or not 0<=values[1]<.003 or not 0<=values[2]<=.5 or not 0<=values[3]<.5):
        raise RuntimeError('Fresh finite bounded native separation observation required')
    return values


def branch_screen(right,rig):
    """Conservative released geometry versus static/protected source geometry.

    One already released adjacent shaft pair is handled separately by the
    distal-only monotone translation rule, not by changing native contacts.
    Leaves, other parent geometry, and every static obstacle remain checked.
    """
    screen=copy(right);screen.workspace=None;screen.native_static_query=None
    screen.shapes=[];screen.local=[s for s in right.local if s[1]<rig.cut_index]
    for path,i,kind,data in right.local:
        if i<rig.cut_index:continue
        if kind=='capsule':value=data
        else:
            vertices=data[0];lo=vertices.min(0);hi=vertices.max(0)
            kind='box';value=((lo+hi)/2,np.eye(3),(hi-lo)/2)
        screen.shapes.append((path,rig.body_paths[i],str(i),kind,value))
    # Existing release-plane contact only. The blade/robot allowances are
    # not available to arbitrary branch shapes in this screen.
    screen.blade_path=rig.body_paths[rig.cut_index]+'/StemCollider'
    screen.seam_paths={rig.body_paths[rig.cut_index-1]+'/StemCollider'}
    return screen


def plan(f,frames,*,record,step,read_frames):
    from .withdrawal_native_check import _robot_world,_screens
    from .measured_withdrawal_native import _raw_joint_world,LEFT,RIGHT
    from .measured_withdrawal import _attained
    from .whole_robot_target import WholeRobotTargetScreen
    from .native_static_clearance import current_scene_query
    from .seam_escape import bind,gaps,monotone
    prerequisites(record,step)
    if f.rig.cut is not True or f.cut_event is None or f.root_transition.receipt is None:
        raise RuntimeError('Verified native release transition required before target motion')
    frames=np.array(frames,float,copy=True);world=_robot_world(f)
    q0=np.array(f.robot.get_dof_positions(),copy=True)
    targets=np.array(f.robot.get_dof_position_targets(),copy=True)
    values=dict(zip(f.names,q0[0],strict=True))
    predicted=_raw_joint_world(f.kin,values,world['base'])
    if any(k not in predicted or not _attained(v,predicted[k]) for k,v in world.items()):
        raise RuntimeError('Complete native robot/FK mismatch before separation')
    left0=np.degrees([values[n] for n in LEFT]);right0=np.degrees([values[n] for n in RIGHT])
    # Keep the existing drive reference at the start, not a measured-pose reset.
    start=np.degrees(targets[0,[f.names.index(n) for n in LEFT]])
    if np.max(abs(start-left0))>.1:raise RuntimeError('Left drive handoff exceeds 0.1 degrees')
    correction=world['ee_left']@np.linalg.inv(f.kin.forward('left',left0,f.base))
    inverse=np.linalg.inv(correction)
    def body(left,right=right0):
        v=dict(values);v.update(zip(LEFT,np.radians(left),strict=True));v.update(zip(RIGHT,np.radians(right),strict=True))
        return _raw_joint_world(f.kin,v,world['base'])
    origin=body(start)['ee_left'];_,axis=f.seam(frames)
    delta=.002*direction(record['knife']['edge_frame'],axis)
    right,left=_screens(f,frames)
    left.set_physical_grasp_span(frames,world,f.body_index,f.grasp_point(frames))
    branch=branch_screen(right,f.rig);branch.snapshot(frames)
    evidence=dict(model='retained_fragment_separation_trial_v1',step=step,passed=False,
        displacement_world_m=delta.tolist(),maximum_displacement_m=.002,
        planning_sample_step_m=.000125,maximum_speed_m_s=.0005,
        source_assets_changed=False,native_collision_filters_changed=False,pose_overwrite=False,
        grasp_weld=False,whole_path_certified=False,deformation_prediction=False,
        existing_released_adjacent_shaft_contact_only=True,training_eligible=False)
    f.retained_separation_evidence=evidence
    def fresh():
        current=_robot_world(f)
        if (not np.array_equal(read_frames(),frames) or not np.array_equal(f.robot.get_dof_positions(),q0)
                or not np.array_equal(f.robot.get_dof_position_targets(),targets)
                or set(current)!=set(world)
                or any(not np.array_equal(v,current[k]) for k,v in world.items())):
            raise RuntimeError('Native state changed during separation planning')
    native=None;path=[];failure=None;previous=None
    try:
        native=current_scene_query(f.stage,right.static,lazy_coverage=True)
        right.native_static_query=native;left.native_static_query=native
        for fraction in np.linspace(0.,1.,17):
            desired=origin.copy();desired[:3,3]+=fraction*delta
            if fraction==0:q=start.copy()
            else:
                solved=f.kin.solve_pose('left',inverse@desired,path[-1],f.base,
                    maximum_evaluations=200,joint_limit_margin_degrees=3.)
                if not solved.succeeded:failure='left_ik';break
                q=np.array(solved.joint_degrees,float)
            if not _attained(desired,body(q)['ee_left']):failure='left_pose_error';break
            if path and np.max(abs(q-path[-1]))>.25:failure='left_joint_sample_step';break
            candidate=frames.copy();candidate[f.rig.cut_index:,:3,3]+=fraction*delta
            right.snapshot(candidate);left.snapshot(candidate)
            target=WholeRobotTargetScreen(left,f.self_screen.shapes).screen
            current=gaps(bind(right),body(q))
            if previous is not None and not monotone(previous,current):failure='cut_face_gap_decreased';break
            previous=current
            branch_world={str(i):candidate[i] for i in range(f.rig.cut_index,len(candidate))}
            if not branch.check(branch_world,stroke=True):failure=dict(branch=branch.last_failure);break
            # Both the held right pose and every existing downward-stroke
            # sample are checked against each proposed left/branch position.
            for rq in [right0,*f.plan['stroke']]:
                w=body(q,rq)
                if not f.self_screen.check(w)['passed']:failure='self_collision';break
                if f.kin.inter_arm_clearance(q,rq,correction@f.base).clearance_m<.01:
                    failure='interarm_clearance';break
                if not target.check(w,grasp=True,stroke=True):failure=dict(target=target.last_failure);break
                if not left.check(w,grasp=True):failure=dict(left=left.last_failure);break
                if not right.check(w,stroke=True):failure=dict(right=right.last_failure);break
            if failure is not None:break
            path.append(q)
        evidence['checked_left_samples']=len(path)
        if failure is not None:
            evidence['rejection']=failure
            raise RuntimeError('Retained-target separation corridor rejected: '+str(failure))
        native.validate();fresh()
    finally:
        if native is not None:
            native.close();evidence['native_static']=native.report()
        fresh()
    n=evidence['native_static'];epoch=n.get('epoch',{})
    if (n.get('final_validation_passed') is not True or n.get('closed') is not True or n.get('errors')
            or epoch.get('revision')!=0 or epoch.get('subscriptions_closed') is not True
            or epoch.get('cleanup_errors') or epoch.get('invalidation_reasons')):
        raise RuntimeError('Separation native epoch/cleanup verification failed')
    evidence.update(passed=True,path_degrees=np.array(path).tolist(),
        native_guards_required_during_execution=True,measured_completion_required=True)
    return Separation(path,origin,delta,frames[f.body_index],step=step)


def recheck(f,frames,*,record,step,read_frames):
    """Reobserve the moved plant before resuming the original downward path."""
    from .withdrawal_native_check import _robot_world,_screens
    from .measured_withdrawal_native import _raw_joint_world,LEFT,RIGHT
    from .whole_robot_target import WholeRobotTargetScreen
    from .native_static_clearance import current_scene_query
    prerequisites(record,step)
    frames=np.array(frames,copy=True);world=_robot_world(f)
    joints=np.array(f.robot.get_dof_positions(),copy=True)
    targets=np.array(f.robot.get_dof_position_targets(),copy=True)
    values=dict(zip(f.names,joints[0],strict=True))
    left_q=np.degrees([values[n] for n in LEFT]);right_q=np.degrees([values[n] for n in RIGHT])
    right,left=_screens(f,frames)
    left.set_physical_grasp_span(frames,world,f.body_index,f.grasp_point(frames))
    target=WholeRobotTargetScreen(left,f.self_screen.shapes).screen
    branch=branch_screen(right,f.rig);branch.snapshot(frames)
    if not branch.check({str(i):frames[i] for i in range(f.rig.cut_index,len(frames))},stroke=True):
        raise RuntimeError('Reobserved branch/protected-context corridor blocked: '+str(branch.last_failure))
    receipt=dict(model='reobserved_retained_target_cut_corridor_v1',step=step,passed=False,
        native_contact_guards_still_required=True,whole_path_certified=False,training_eligible=False)
    def fresh():
        now=_robot_world(f)
        if (not np.array_equal(read_frames(),frames) or not np.array_equal(f.robot.get_dof_positions(),joints)
                or not np.array_equal(f.robot.get_dof_position_targets(),targets)
                or set(now)!=set(world) or any(not np.array_equal(v,now[k]) for k,v in world.items())):
            raise RuntimeError('Native state changed during retained-target reobservation')
    native=None
    try:
        native=current_scene_query(f.stage,right.static,lazy_coverage=True)
        right.native_static_query=native;left.native_static_query=native
        for q in [right_q,*f.plan['stroke']]:
            v=dict(values);v.update(zip(RIGHT,np.radians(q),strict=True))
            w=_raw_joint_world(f.kin,v,world['base'])
            if (not f.self_screen.check(w)['passed']
                    or f.kin.inter_arm_clearance(left_q,q,f.base).clearance_m<.01
                    or not target.check(w,grasp=True,stroke=True)
                    or not left.check(w,grasp=True) or not right.check(w,stroke=True)):
                raise RuntimeError('Reobserved retained-target cut corridor no longer clear')
        native.validate();fresh()
    finally:
        if native is not None:native.close();receipt['native_static']=native.report()
        fresh()
    n=receipt['native_static'];epoch=n.get('epoch',{})
    if (n.get('final_validation_passed') is not True or n.get('closed') is not True or n.get('errors')
            or epoch.get('revision')!=0 or epoch.get('subscriptions_closed') is not True
            or epoch.get('cleanup_errors') or epoch.get('invalidation_reasons')):
        raise RuntimeError('Retained-target reobservation epoch/cleanup failed')
    receipt['passed']=True
    return receipt


class Separation:
    def __init__(self,path,origin,delta,target,*,step):
        self.path=np.array(path,float);self.origin=np.array(origin,float);self.delta=np.array(delta,float)
        self.target=np.array(target,float)
        if (self.path.shape!=(17,7) or not np.isfinite(self.path).all()
                or self.delta.shape!=(3,) or not np.isfinite(self.delta).all()
                or abs(np.linalg.norm(self.delta)-.002)>1e-9 or type(step) is not int or step<1):
            raise ValueError('Exact bounded screened separation path required')
        from .withdrawal_evidence import _pose
        self.origin=_pose(self.origin);self.target=_pose(self.target)
        self.first=step;self.step=step-1;self.consumed=None;self.offset=0.;self.complete=False;self.dwell=0
        self.receipt=None

    def observe(self,record,*,step,target):
        from .withdrawal_evidence import _pose
        values=prerequisites(record,step);palm=_pose(record['palm']);target=_pose(target)
        if step!=self.step+1 or step>self.first and self.consumed!=self.step:
            raise RuntimeError('Missing/reused separation observation or command')
        unit=self.delta/.002;travel=palm[:3,3]-self.origin[:3,3]
        amount=float(travel@unit);lateral=float(np.linalg.norm(travel-amount*unit))
        target_travel=target[:3,3]-self.target[:3,3]
        measured=float(target_travel@unit)
        rotation=self.origin[:3,:3].T@palm[:3,:3]
        if (lateral>.0005 or not -.0005<=amount<=.0025
                or (np.trace(rotation)-1)/2<np.cos(np.radians(1.))
                or np.linalg.norm(target_travel-travel)>.0025):
            raise RuntimeError('Retained-target separation tracking/deformation envelope exceeded')
        # Explicitly measure both palm AND retained target; command completion
        # alone is never evidence that the material actually moved.
        attained=amount>=.0018 and measured>=.0015 and abs(self.offset-.002)<1e-10
        self.dwell=self.dwell+1 if attained else 0
        self.complete=self.complete or self.dwell>=12
        self.safe=values[1]<.002 and values[2]<.4 and values[3]<.4
        self.step=step
        self.receipt=dict(step=step,complete=self.complete,palm_travel_m=amount,
            target_travel_m=measured,lateral_error_m=lateral,measured_dwell_s=self.dwell/480,
            commanded_motion_used_as_completion=False,native_grasp_required=True)
        return dict(self.receipt)

    def command(self,*,step,dt):
        if (type(step) is not int or step!=self.step or self.consumed==step
                or isinstance(dt,bool) or not np.isfinite(dt) or abs(dt-1/480)>1e-12):
            raise RuntimeError('Fresh single-use 480 Hz separation command required')
        if not self.complete and step-self.first>=12*480:
            raise RuntimeError('Retained separation timed out without measured target movement')
        speed=.0005 if self.safe and not self.complete else 0.
        self.offset=min(.002,self.offset+speed*dt);self.consumed=step
        q=np.array([np.interp(self.offset/.002,np.linspace(0.,1.,17),self.path[:,i]) for i in range(7)])
        self.receipt.update(command_offset_m=self.offset,command_speed_m_s=speed)
        return q
