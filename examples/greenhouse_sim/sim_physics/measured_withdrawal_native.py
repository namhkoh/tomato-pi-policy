"""Read-only native adapter for MeasuredWithdrawal; no commands or stepping.

Construct NativeWithdrawalAdapter(fixture, runtime.bodies, current_sample_id).
After fetched physics AND the original native guard/grasp checks, call snapshot
with explicit same-(episode,step) evidence. Pass adapter.wrist_fk and .validate
to the pure helper. Re-snapshot before each helper.advance(). Failed validation
latches this adapter; neither an old receipt nor a later sample can repair it.

guard_evidence = {sample_id: [...], passed: True, errors: []}
grasp_evidence = {sample_id: [...], result: actual contact_with_frames result}
The latter requires bilateral=True, adapter_valid=True, and matching step_id.
These caller-supplied attestations cannot be independently authenticated here.
No contact thresholds are recomputed/replaced; the caller retains ALL guards.

Full native robot/plant inventories and joint positions/targets are re-read
before/after synchronous validation. FK uses measured base, torso, head/wheel
and all finger positions, never fixture.pose/planning_slides defaults. Actual
non-right bodies stay at measured poses in future samples. Dynamic plant poses
are frozen only for that call, not predicted. Checks are sampled, not swept.
Raw measured FK reports nominal-stop excursions without clamping or inventing
compliance. Native drive targets and proposed right commands stay source-limited;
an out-of-limit measured right-arm start is not an authorized recovery command.

Each receipt creates a new static query epoch (up to 8 seconds and the supplied
remaining <=20000 queries, INCLUDING initialization). A real current robot-bound
clearance query supplies a used path when refinement used none; native.validate
then rechecks positive actor coverage. Native report fields are not fabricated.
Lazy positive controls cover each relied-upon actor within THIS call only; no
coverage or clearance survives a physics step. All coarse obstacles stay checked.
Unknown coverage, changing state, timeout and cleanup errors never pass.
"""
from copy import copy, deepcopy
import json
import time

import numpy as np

from .measured_withdrawal import (CHECKS, MARGINS, MAX_QUERIES, MAX_SAMPLES,
    MAX_SECONDS, _array, _attained, _hash, _json, _measurement)
from .withdrawal_evidence import _pose, _sample_id
from .withdrawal_native_check import _paths, _robot_world, _screens, _wrist
from .runtime import pose_matrices


RIGHT = tuple('right_arm_'+str(i) for i in range(7))
LEFT = tuple('left_arm_'+str(i) for i in range(7))
FINGERS = ('gripper_finger_l1','gripper_finger_l2')


def _raw_joint_world(kin, values_si, base):
    """Observation geometry only: exact URDF motions, no limit/clamp policy."""
    from greenhouse_sim.robot_kinematics import _axis_rotation
    joints=kin._by_name
    if set(values_si)!={n for n,j in joints.items() if j.kind!='fixed'}:
        raise ValueError('Complete measured SI joint inventory required')
    values={n:_array(v,(),'measured SI joint').item() for n,v in values_si.items()}
    pending=dict(kin._by_child);world={'base':_pose(base)}
    if (len(pending)!=len(joints) or {j.name for j in pending.values()}!=set(joints)
            or any(j.child!=child or joints[j.name] is not j for child,j in pending.items())):
        raise ValueError('Inconsistent source joint tree')
    while pending:
        progressed=False
        for child,j in list(pending.items()):
            if j.parent not in world:continue
            if child in world:raise ValueError('Duplicate/root child in source joint tree')
            motion=np.eye(4)
            if j.kind!='fixed':
                axis=_array(j.axis,(3,),'source joint axis')
                if abs(np.linalg.norm(axis)-1)>1e-8:raise ValueError('Nonunit source joint axis')
                if j.kind in ('revolute','continuous'):
                    motion[:3,:3]=_axis_rotation(axis,values[j.name])
                elif j.kind=='prismatic':motion[:3,3]=axis*values[j.name]
                else:raise ValueError('Unsupported source joint kind')
            world[child]=_pose(world[j.parent]@_pose(j.origin)@motion)
            del pending[child];progressed=True
        if not progressed:raise ValueError('Source joint tree not connected to base')
    return world


def _evidence(guard, grasp, sample):
    guard,grasp=deepcopy(guard),deepcopy(grasp)
    if (type(guard) is not dict or _sample_id(guard['sample_id']) != sample
            or guard.get('passed') is not True or guard.get('errors') != []):
        raise ValueError('Fresh explicitly passed native guard evidence required')
    if type(grasp) is not dict or _sample_id(grasp['sample_id']) != sample:
        raise ValueError('Fresh bilateral evidence sample required')
    r=grasp.get('result',{})
    if (type(r) is not dict or r.get('bilateral') is not True or r.get('adapter_valid') is not True
            or type(r.get('step_id')) is not int or r['step_id'] != sample[1]):
        raise ValueError('Same-step verified native bilateral result required')
    return json.loads(_json(guard)),json.loads(_json(grasp))


def _native_valid(n, limit):
    if type(n) is not dict or type(n.get('epoch')) is not dict:return False
    e=n['epoch'];used=n.get('used_static_colliders');covered=n.get('final_coverage_checked')
    if (type(used) is not list or type(covered) is not list or not covered
            or any(type(p) is not str or not p.startswith('/') for p in used+covered)
            or len(set(covered))!=len(covered) or not set(used)<=set(covered)):
        return False
    return (n.get('final_validation_passed') is True and n.get('closed') is True
        and n.get('errors')==[] and type(n.get('query_count')) is int and 1<=n['query_count']<=limit
        and type(e.get('revision')) is int and e['revision']==0
        and e.get('subscriptions_closed') is True and e.get('cleanup_errors')==[]
        and e.get('invalidation_reasons')==[])


class NativeWithdrawalAdapter:
    def __init__(self, fixture, plant_bodies, current_sample_id):
        if not callable(current_sample_id):
            raise ValueError('Live physics-clock sample reader required')
        self.f=fixture;self.plant=plant_bodies;self.current_sample_id=current_sample_id
        self._sample=None;self._m=None;self._blocked=None;self._busy=False
        self._hold=None;self._station_targets=None;self._inventories=None
        self.last_receipt=None

    def _clock(self, sample):
        if _sample_id(self.current_sample_id()) != sample:
            raise RuntimeError('Physics sample changed or stale evidence')

    def _capture(self, sample, hold, guard, grasp):
        self._clock(sample);f=self.f
        if (f.rig.cut is not True or f.event_monitor.error is not None
                or f.event_monitor.native_full_contact_reporting is not True):
            raise ValueError('Released seam and healthy full native contact stream required')
        a=f.robot;names=tuple(a.shared_metatype.dof_names)
        joints=f.kin._by_name
        expected={n for n,j in joints.items() if j.kind != 'fixed'}
        if (a.count != 1 or a.shared_metatype.fixed_base is not True
                or len(names) != len(set(names)) or set(names) != expected
                or list(names) != list(f.names)):
            raise ValueError('Exact full native DOF inventory required; no omitted torso/fingers')
        values=_array(a.get_dof_positions(),(1,len(names)),'native DOFs')[0]
        targets=_array(a.get_dof_position_targets(),(1,len(names)),'native targets')[0]
        q=dict(zip(names,values,strict=True));target=dict(zip(names,targets,strict=True))
        if not np.array_equal(_array(f.targets,(1,len(names)),'controller targets')[0],targets):
            raise ValueError('Native/controller target mismatch')
        degrees={};slides={};excursions=[]
        for name,v in q.items():
            j=joints[name]
            if not j.lower_rad <= target[name] <= j.upper_rad:
                raise ValueError('Native command target outside source limits: '+name)
            if j.kind=='prismatic':slides[name]=float(v)
            elif j.kind in ('revolute','continuous'):degrees[name]=float(np.degrees(v))
            else:raise ValueError('Unknown active native joint kind')
            if not j.lower_rad <= v <= j.upper_rad:
                excursions.append(dict(joint=name,observed=float(v),
                    lower=float(j.lower_rad) if np.isfinite(j.lower_rad) else None,
                    upper=float(j.upper_rad) if np.isfinite(j.upper_rad) else None,
                    signed_excess=float(v-j.lower_rad if v<j.lower_rad else v-j.upper_rad),
                    units='m' if j.kind=='prismatic' else 'rad'))
        if set(slides) != {'gripper_finger_'+s+str(i) for s in ('l','r') for i in (1,2)}:
            raise ValueError('Exact four native finger slides required')
        world=_robot_world(f);actual=_wrist(f)
        if not np.allclose(world['ee_right'],actual,atol=1e-7,rtol=0):
            raise ValueError('Native wrist views disagree')
        if set(a.link_paths[0]) != set(f.body_paths) or len(a.link_paths[0]) != len(f.body_paths):
            raise ValueError('Articulation/native rigid body inventory mismatch')
        shape_paths=[s[0] for s in f.self_screen.shapes]
        if (len(set(shape_paths)) != len(shape_paths) or set(shape_paths) != set(f.collider_paths)
                or f.self_screen.unsupported):
            raise ValueError('Complete supported cached robot collider inventory required')
        base=_pose(world['base'])
        predicted=_raw_joint_world(f.kin,q,base)
        if any(link not in predicted or not _attained(pose,predicted[link]) for link,pose in world.items()):
            raise ValueError('Full native body/FK inconsistency; no pose repair')
        plant_paths=_paths(self.plant.prim_paths,'plant view')
        expected_plant=_paths(f.rig.body_paths,'rig plant')
        if set(plant_paths) != set(expected_plant):
            raise ValueError('Incomplete native plant view')
        plant_poses=pose_matrices(_array(self.plant.get_transforms(),(len(plant_paths),7),'native plant poses'))
        plants={p:_pose(v).tolist() for p,v in zip(plant_paths,plant_poses,strict=True)}
        command_pose=_raw_joint_world(f.kin,{**q,**{n:target[n] for n in LEFT}},base)['ee_left']
        if (not _attained(hold['left_goal_world'],command_pose)
                or not np.array_equal(_array(hold['finger_targets_m'],(2,),'held finger targets'),[target[n] for n in FINGERS])):
            raise ValueError('Supplied hold targets do not match native drive targets')
        scene=dict(sample_id=list(sample),robot_body_world={f.root+'/'+n:p.tolist() for n,p in world.items()},
            plant_body_world=plants,left_q_deg=[degrees[n] for n in LEFT],finger_slides_m=slides,
            native_dof_positions={n:float(v) for n,v in q.items()},native_dof_targets={n:float(v) for n,v in target.items()},
            revolute_degrees=degrees,native_base_world=base.tolist(),
            measured_joint_limit_excursions=excursions,measured_joint_positions_clamped=False,
            command_targets_within_source_limits=True,
            robot_collider_paths=sorted(shape_paths),guard_evidence=guard,grasp_evidence=grasp)
        m=_measurement(dict(sample_id=list(sample),right_q_deg=[degrees[n] for n in RIGHT],
            right_wrist_world=actual.tolist(),right_wrist_path=f.root+'/ee_right',hold_targets=hold,scene=scene))
        self._clock(sample)
        return m

    def _right_values(self, right_q_deg):
        q=_array(right_q_deg,(7,),'proposed right joints')
        low,high=self.f.kin.arm_limits_degrees('right')
        if np.any(q<=low) or np.any(q>=high):
            raise ValueError('Proposed right command outside strict source limits')
        return {**self._m['scene']['native_dof_positions'],**dict(zip(RIGHT,np.radians(q),strict=True))}

    def snapshot(self, sample_id, *, hold_targets, guard_evidence, grasp_evidence):
        """Read native views only; stale/failed evidence raises before any query."""
        if self._busy:self._blocked='Reentrant snapshot is forbidden'
        if self._blocked:raise RuntimeError(self._blocked)
        try:
            sample=_sample_id(sample_id)
            if self._sample is not None and sample != (self._sample[0],self._sample[1]+1):
                raise ValueError('Exactly next adapter snapshot required')
            guard,grasp=_evidence(guard_evidence,grasp_evidence,sample)
            m=self._capture(sample,deepcopy(hold_targets),guard,grasp)
            again=self._capture(sample,deepcopy(hold_targets),guard,grasp)
            if _hash(m) != _hash(again):raise ValueError('Native views changed while snapshotting')
            targets={n:v for n,v in m['scene']['native_dof_targets'].items() if n not in RIGHT}
            inventory=(tuple(sorted(m['scene']['robot_body_world'])),tuple(sorted(m['scene']['plant_body_world'])),
                       tuple(m['scene']['robot_collider_paths']))
            if self._m is not None and (_hash(m['hold_targets']) != _hash(self._hold)
                    or targets != self._station_targets or inventory != self._inventories):
                raise ValueError('Hold/station target or native inventory changed')
            self._hold=deepcopy(m['hold_targets']);self._station_targets=targets;self._inventories=inventory
            self._sample=sample;self._m=m
            return deepcopy(m)
        except BaseException as exc:
            self._blocked=type(exc).__name__+': '+str(exc)
            raise

    def wrist_fk(self, right_q_deg):
        """Right FK using latest measured torso/base/fingers; no native writes."""
        if self._blocked or self._m is None:raise RuntimeError(self._blocked or 'Snapshot required')
        self._clock(self._sample)
        s=self._m['scene']
        return _raw_joint_world(self.f.kin,self._right_values(right_q_deg),s['native_base_world'])['ee_right']

    def _fresh(self):
        if self._m is None:raise ValueError('Native snapshot required')
        m=self._m;s=m['scene']
        actual=self._capture(self._sample,deepcopy(self._hold),s['guard_evidence'],s['grasp_evidence'])
        if _hash(actual) != _hash(m):raise ValueError('Native state changed since supplied snapshot')

    def _request(self, request):
        r=json.loads(_json(request));digest=r.pop('request_sha256')
        if digest != _hash(r):raise ValueError('Request hash mismatch')
        r['request_sha256']=digest
        if (_sample_id(r['sample_id']) != self._sample or r['snapshot_sha256'] != _hash(self._m)
                or _hash(r['measurement']) != _hash(self._m) or r['margins_m'] != MARGINS
                or r.get('left_hold_fixed') is not True or r.get('full_forward_cutstroke_verified') is not False
                or type(r.get('max_queries')) is not int or not 1 <= r['max_queries'] <= MAX_QUERIES
                or type(r.get('max_seconds')) not in (int,float) or not 0 < r['max_seconds'] <= MAX_SECONDS
                or r.get('maximum_joint_sample_step_deg') != 1.):
            raise ValueError('Stale, changed or unbounded withdrawal request')
        path=np.asarray(r['path_q_deg'])
        if path.ndim != 2 or path.shape[1] != 7 or not 1 <= len(path) <= MAX_SAMPLES:
            raise ValueError('Bounded full right path required')
        path=_array(path,path.shape,'path')
        low,high=self.f.kin.arm_limits_degrees('right')
        if (np.any(path<=low) or np.any(path>=high)
                or (len(path)>1 and np.max(abs(np.diff(path,axis=0)))>1.+1e-12)
                or not np.array_equal(path[0],self._m['right_q_deg'])):
            raise ValueError('Path limits, density or actual start mismatch')
        phases=r['phases'];allow=r['seam_allowance']
        if (type(phases) is not list or len(phases) != len(path)-1
                or any(p not in ('extract','return') for p in phases)
                or type(allow) is not list or any(type(b) is not bool for b in allow)
                or allow != [p=='extract' for p in phases]
                or any(a=='return' and b=='extract' for a,b in zip(phases,phases[1:]))
                or r['kind'] not in ('release_path','next_segment','endpoint')
                or (r['kind']=='endpoint') != (len(path)==1)):
            raise ValueError('Invalid phase or seam allowance scope')
        wrist=r['path_wrist_world']
        if len(wrist) != len(path):raise ValueError('Missing path wrist frames')
        for q,p in zip(path,wrist,strict=True):
            if not np.allclose(_pose(p),self.wrist_fk(q),atol=1e-9,rtol=0):
                raise ValueError('Path FK did not use current full native posture')
        return r,path,phases

    def _future(self,q,world):
        s=self._m['scene']
        fk=_raw_joint_world(self.f.kin,self._right_values(q),s['native_base_world'])
        # Follow exact URDF parentage, including fingers; no prefix lookalikes.
        future={n:p.copy() for n,p in world.items()}
        for link in world:
            node=link
            while node in self.f.kin._by_child:
                joint=self.f.kin._by_child[node]
                if joint.name==RIGHT[0]:future[link]=fk[link];break
                node=joint.parent
        return future

    def _interarm(self,world):
        from greenhouse_sim.robot_kinematics import CapsuleObstacle,capsule_sets_clearance
        groups=[]
        for side in ('left','right'):
            caps=[];links={'link_'+side+'_arm_'+str(i) for i in range(7)}
            for path,_,link,kind,data in self.f.self_screen.shapes:
                if link not in links or kind!='capsule':continue
                a,b,r=data;m=world[link];a=m[:3,:3]@a+m[:3,3];b=m[:3,:3]@b+m[:3,3]
                caps.append(CapsuleObstacle(path,tuple(a),tuple(b),float(r)))
            if not caps:raise ValueError('Missing actual inter-arm capsule coverage')
            groups.append(tuple(caps))
        value=float(capsule_sets_clearance(*groups).clearance_m)
        if not np.isfinite(value) or value<MARGINS['interarm_m']:
            raise ValueError('Inter-arm clearance below 10 mm')
        return value

    def _screen_world(self,world,right,left,stroke):
        result=copy(self.f.self_screen).check(world,margin=MARGINS['self_m'])
        if (result.get('passed') is not True or result.get('all_shape_bounds_screened') is not True
                or result.get('unsupported_shapes') != [] or type(result.get('checked_pairs')) is not int
                or result['checked_pairs']<=0 or result.get('required_margin_m') != MARGINS['self_m']
                or not np.isfinite(result.get('minimum_clearance_m',np.nan))
                or result['minimum_clearance_m']<MARGINS['self_m']):
            raise ValueError('Complete 3 mm self screen rejected: '+str(result))
        interarm=self._interarm(world)
        if right.check(world,stroke=stroke,margin=MARGINS['scene_m']) is not True:
            raise ValueError('Right scene screen rejected: '+str(right.last_failure))
        if left.check(world,grasp=True,stroke=False,margin=MARGINS['scene_m']) is not True:
            raise ValueError('Left scene screen rejected: '+str(left.last_failure))
        return float(result['minimum_clearance_m']),interarm

    def _coverage_witness(self,native,right,world):
        """One REAL robot-bound vs static-actor query, not a coverage placeholder."""
        if native.used_paths:return None
        if not native.coverage_boxes:raise ValueError('No eligible native static actor for positive coverage')
        # Merely selecting an eligible path is NOT evidence. The checked call
        # below must perform its positive control before any claimed clearance.
        other=sorted(native.coverage_boxes)[0]
        for path,body,link,kind,data in right.shapes:
            tool=(link=='ee_right' and path.startswith(tuple(body+'/attachments/'+n+'/' for n in ('RightWristCamera','DeleafKnife'))))
            arm=(link in {'link_right_arm_'+str(i) for i in range(7)} and kind=='capsule'
                 and path.startswith(body+'/restored_collisions/capsule_'))
            if not ((tool and kind=='box') or arm):continue
            m=world[link]
            if kind=='box':
                c,a,h=data;clear=native.clear_box_checked(other,m[:3,:3]@c+m[:3,3],m[:3,:3]@a,h,.001)
            else:
                a,b,r=data;clear=native.clear_capsule_checked(other,m[:3,:3]@a+m[:3,3],m[:3,:3]@b+m[:3,3],r,.001)
            if clear is not True:raise ValueError('Actual robot-bound coverage witness not clear')
            return dict(robot_collider=path,static_collider=other,actual_native_start=True,margin_m=.001)
        raise ValueError('No existing native-refinement-eligible robot bound for positive coverage')

    def validate(self, request):
        """Return truthful receipt, including failures/cleanup; NEVER move robot."""
        started=time.monotonic();native=None;r=None;error=None;checked=0;witness=None
        receipt=dict(passed=False,checks={k:False for k in CHECKS},errors=[],native_static=None,
                     checked_samples=0,checked_segments=0,native_provenance_verified=False)
        cost=dict(initialization_queries=None,path_refinement_queries=None,
                  witness_queries=None,final_actor_queries=None,total_queries=None,
                  query_epoch_recreated_each_validation=True,
                  lazy_actor_coverage_requested=True,
                  helper_total_query_cap_unchanged=MAX_QUERIES)
        if self._busy:self._blocked='Reentrant validation is forbidden'
        if self._blocked:
            receipt['errors']=[self._blocked or 'Reentrant validation'];return receipt
        self._busy=True
        try:
            self._fresh();r,path,phases=self._request(request)
            deadline=started+r['max_seconds']
            def check_time():
                self._clock(self._sample)
                if time.monotonic()>=deadline:raise RuntimeError('Withdrawal validation deadline exhausted')
            check_time()
            receipt.update(request_sha256=r['request_sha256'],snapshot_sha256=r['snapshot_sha256'],
                sample_id=r['sample_id'],margins_m=dict(MARGINS))
            frames=np.array([self._m['scene']['plant_body_world'][p] for p in self.f.rig.body_paths])
            right,left=_screens(self.f,frames)
            # Fresh epoch: at least one positive control + one real clearance
            # + one final positive control. No 80-actor eager initialization.
            if r['max_queries']<3:
                raise ValueError('Remaining queries cannot cover positive/clearance/final controls')
            from .native_static_clearance import current_scene_query,NativeStaticClearance
            native=current_scene_query(self.f.stage,right.static,wall_limit_s=deadline-time.monotonic(),
                max_queries=r['max_queries'],lazy_coverage=True)
            if not isinstance(native,NativeStaticClearance):raise ValueError('Unexpected native query implementation')
            if native.lazy_coverage is not True or native.calls!=0 or native.max_queries!=r['max_queries']:
                raise ValueError('Exact zero-query lazy epoch and supplied query budget required')
            cost['initialization_queries']=native.calls
            right.native_static_query=native;left.native_static_query=native
            world=_robot_world(self.f)
            # Screen actual native geometry, not a command/FK substitution.
            first=self._screen_world(world,right,left,bool(phases and phases[0]=='extract'))
            minima=list(first)
            for i,q in enumerate(path):
                check_time()
                # Junctions receive the stricter outgoing return allowance.
                phase=phases[min(i,len(phases)-1)] if phases else 'return'
                values=self._screen_world(self._future(q,world),right,left,phase=='extract')
                minima=[min(a,b) for a,b in zip(minima,values)];checked=i+1
            cost['path_refinement_queries']=native.calls-cost['initialization_queries']
            before_witness=native.calls
            witness=self._coverage_witness(native,right,world)
            cost['witness_queries']=native.calls-before_witness
            before_final=native.calls
            self._fresh();check_time();native.validate();check_time()
            cost['final_actor_queries']=native.calls-before_final
            receipt.update(checked_samples=checked,checked_segments=len(phases),
                minimum_self_clearance_m=minima[0],minimum_interarm_clearance_m=minima[1],
                actual_start_screened=True,coverage_witness=witness,
                future_pose_basis='current full native DOFs; right URDF subtree only',
                joint_limits_scope='native command targets and proposed right commands; measured excursions reported',
                whole_path_certified=False,full_forward_cutstroke_verified=False)
        except BaseException as exc:
            error=exc;receipt['errors'].append(type(exc).__name__+': '+str(exc))
        finally:
            if native is not None:
                try:native.close()
                except BaseException as exc:
                    receipt['errors'].append('close: '+type(exc).__name__+': '+str(exc));error=error or exc
                try:receipt['native_static']=deepcopy(native.report())
                except BaseException as exc:
                    receipt['errors'].append('report: '+type(exc).__name__+': '+str(exc));error=error or exc
            try:
                self._fresh()
                if r is not None and time.monotonic()-started>=r['max_seconds']:
                    raise RuntimeError('Withdrawal deadline exceeded during cleanup')
            except BaseException as exc:
                receipt['errors'].append('final_state: '+type(exc).__name__+': '+str(exc));error=error or exc
            n=receipt['native_static']
            valid=_native_valid(n,r['max_queries'] if r is not None else 0)
            if type(n) is dict and type(n.get('query_count')) is int:cost['total_queries']=n['query_count']
            if not valid and not receipt['errors']:receipt['errors'].append('Native final coverage/epoch incomplete')
            if self._blocked and self._blocked not in receipt['errors']:receipt['errors'].append(self._blocked)
            receipt['passed']=bool(valid and not receipt['errors'])
            if receipt['passed']:receipt['checks']={k:True for k in CHECKS}
            else:self._blocked='; '.join(receipt['errors'])
            receipt['wall_seconds']=time.monotonic()-started
            receipt['cost']=cost
            self._busy=False
        try:self.last_receipt=json.loads(_json(receipt))
        except Exception as exc:
            self._blocked='Unrepresentable native receipt: '+type(exc).__name__+': '+str(exc)
            self.last_receipt=dict(passed=False,checks={k:False for k in CHECKS},
                errors=[self._blocked],native_static=None,native_provenance_verified=False)
        if error is not None and not isinstance(error,Exception):raise error
        return deepcopy(self.last_receipt)
