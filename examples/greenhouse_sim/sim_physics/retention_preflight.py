"""Conservative static retention prerequisite, not a dynamic grasp certificate.

Uses the current guard-accepted native contact patch and known detached-body
mass/COM inventory. Gravity only; cutter loading and subsequent patch changes
are NOT certified. No pose, force, material, contact or cut setters.
"""
import numpy as np
from .grasp_wrench import gravity_wrench, static_capacity


def assess(record, *, step, time_s, body_paths, cut_index, masses, local_coms,
           current_frames, friction=.5):
    """Caller binds this record to its current native fetch, before motion.

    Hidden source geometry is used only for this privileged fixture's load
    estimate, not as a newly observed grasp or cut target. Failure of this
    fixed-patch approximation is not proof that all possible grasps fail.
    """
    paths=tuple(body_paths);n=len(paths)
    if (not paths or len(set(paths))!=n or type(cut_index) is not int
            or not 0<=cut_index<n or type(step) is not int or step<1
            or isinstance(time_s,bool) or not np.isfinite(time_s)):
        raise ValueError('Exact ordered branch identity and native clock required')
    telemetry=record.get('grasp_dynamics',{})
    if (record.get('native_guards_passed') is not True or record.get('cut') is not False
            or record.get('contact',{}).get('adapter_valid') is not True
            or record.get('contact',{}).get('bilateral') is not True
            or telemetry.get('step_id')!=step or record.get('t')!=time_s
            or telemetry.get('dt_s')!=1/240 or abs(time_s-step/240)>1e-10
            or telemetry.get('model')!='grasp_dynamics_post_fetch_telemetry_v1'
            or telemetry.get('contact_row_contract')!='shaft_grasp_core_oriented_signed_normal_rows_v1'
            or tuple(telemetry.get('body_paths',()))!=paths):
        raise ValueError('Current attached guard-accepted bilateral native telemetry required')
    frames=np.array(telemetry.get('body_frames_world_m'),float)
    mass=np.array(masses,float).reshape(-1);com=np.array(local_coms,float)
    current=np.asarray(current_frames,float)
    if (frames.shape!=(n,4,4) or current.shape!=frames.shape
            or not np.array_equal(frames,current) or mass.shape!=(n,) or com.shape!=(n,3)
            or any(not np.isfinite(v).all() for v in (frames,mass,com)) or np.any(mass<=0)
            or not np.allclose(frames[:,3],[0,0,0,1],rtol=0,atol=1e-7)
            or not np.allclose(frames[:,:3,:3].transpose(0,2,1)@frames[:,:3,:3],np.eye(3),rtol=0,atol=1e-5)
            or np.any(np.linalg.det(frames[:,:3,:3])<0)):
        raise ValueError('Unchanged current frames and complete finite native mass/COM binding required')
    slip=record.get('slip_m')
    if isinstance(slip,bool) or not isinstance(slip,(float,int)) or not np.isfinite(slip) or not 0<=slip<.003:
        raise ValueError('Current bounded grasp slip required')
    world=np.einsum('bij,bj->bi',frames[:,:3,:3],com)+frames[:,:3,3]
    origin=np.array(record.get('grasp_point'),float)
    load=gravity_wrench(mass[cut_index:],world[cut_index:],origin)
    allowed={p+'/StemCollider' for p in paths[cut_index:]}
    fingers=telemetry.get('finger_paths',())
    if len(fingers)!=2 or len(set(fingers))!=2:
        raise ValueError('Exact two-finger identity required')
    raw=telemetry.get('contacts')
    if not isinstance(raw,list) or not 1<=len(raw)<=256:
        raise ValueError('Bounded current contact patch required')
    contacts=[];excluded=0
    for row in raw:
        i=row.get('finger_index')
        if type(i) is not int or i not in (0,1) or row.get('finger_path')!=fingers[i]:
            raise ValueError('Contact finger identity mismatch')
        normal=np.array(row.get('normal_on_finger_world'),float)
        impulse=np.array(row.get('impulse_on_finger_world_ns'),float)
        point=np.array(row.get('point_world_m'),float)
        if any(v.shape!=(3,) or not np.isfinite(v).all() for v in (normal,impulse,point)):
            raise ValueError('Finite original oriented contact rows required')
        if not np.isclose(np.linalg.norm(normal),1.,rtol=0,atol=1e-6):
            raise ValueError('Unit oriented contact normal required')
        if row.get('other_collider') not in allowed or normal@impulse<=0:
            excluded+=1;continue
        contacts.append((point,-normal,i))
    result=dict(model='native_patch_retention_preflight_v1',step_id=step,time_s=float(time_s),
        target_body_count=n-cut_index,detached_mass_kg=float(mass[cut_index:].sum()),
        gravity_wrench_n_nm=load.tolist(),compressive_rows=len(contacts),excluded_rows=excluded,
        prerequisite_passed=False,capacity=None,native_sample_binding_checked=True,
        zero_load_rows_used=False,cut_authorized=False,dynamic_retention_verified=False,
        cutter_load_certified=False,physical_tissue_calibrated=False,training_eligible=False,
        assumptions=['known source/native mass inventory','unchanging current contact patch',
            'gravity only','8-sided friction cone','arbitrary force redistribution',
            'no actuator or dynamic guarantee'])
    if {v[2] for v in contacts}!={0,1}:
        result['rejection']='no_compressive_patch_on_both_fingers';return result
    capacity=static_capacity([v[0] for v in contacts],[v[1] for v in contacts],[v[2] for v in contacts],
        load,origin=origin,friction=friction,per_finger_budget=(.5,.5))
    result['capacity']=capacity
    result['prerequisite_passed']=bool(capacity['balance_found'] and capacity['within_contact_budgets'])
    if not result['prerequisite_passed']:result['rejection']='static_retention_not_established_within_contact_budgets'
    return result
