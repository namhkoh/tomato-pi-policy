"""Same-anatomy IK warm start for ZERO-MOTION native station searches only.

A previous run supplies seeds, not collision or motion authority. Translate
the initial station by the authored attachment displacement at another original
planting slot. Recompute floor height, both IK goals and every native screen.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def validate_mode(args):
    if not (args.full_robot_probe and args.bimanual_cut and args.cut_station_orbit
            and args.native_startup_station_search and args.native_startup_clearance
            and args.native_static_clearance and not args.watch_cut_trial
            and not args.right_only_cut_trial and args.station_proposal_report is None):
        raise ValueError('Station reference is only a zero-motion native bimanual search seed')


def initial_options(args,rig):
    validate_mode(args)
    path=Path(args.station_reference_report).resolve()
    with path.open('rb') as stream:raw=stream.read(2_000_001)
    if len(raw)>2_000_000:raise ValueError('Bounded station reference report required')
    report=json.loads(raw);config=report.get('configuration',{});robot=report.get('robot',{})
    for key in ('plant','target','grasp_arc_m','grasp_roll','grasp_pitch','grasp_depth_m',
                'grasp_skew','approach_vector','approach_distance','torso_degrees',
                'cut_arc_m','knife_edge_mode','knife_alignment','stem_contact_model'):
        if config.get(key)!=getattr(args,key):raise ValueError('Station reference task mismatch: '+key)
    from greenhouse_sim.robot_model import DEFAULT_ASSET
    if Path(robot.get('asset','')).resolve()!=DEFAULT_ASSET.resolve():
        raise ValueError('Station reference must use the current source robot asset')
    gates=report.get('gates',{})
    if not all(gates.get(k) is True for k in ('blade_contact_release',
            'full_forward_cut_stroke_verified','right_withdrawal_completed')):
        raise ValueError('Reference requires prior native cut/traversal/withdrawal evidence')
    def vector(value,n):
        if (not isinstance(value,list) or len(value)!=n
                or any(type(v) not in (int,float) for v in value) or not np.isfinite(value).all()):
            raise ValueError('Finite authored station reference vectors required')
        return np.array(value,dtype=float)
    old_attachment=vector(robot.get('ground_truth_grasp',{}).get('attachment_world_m'),3)
    current=np.asarray(rig.chain_world[0],float)
    if current.shape!=(3,) or not np.isfinite(current).all():raise ValueError('Finite current attachment required')
    pose=vector(robot.get('explicit_initial_station_xy_yaw'),3)
    pose[:2]+=(current-old_attachment)[:2]
    left=vector(config.get('left_ik_seed_degrees'),7)
    right=vector(config.get('right_ready_degrees'),7)
    # Source constructor independently checks all source joint/station limits.
    options=dict(station_pose=pose.tolist(),left_ik_seed_degrees=left.tolist(),
                 right_ready_degrees=right.tolist())
    return options,dict(model='same_anatomy_station_search_reference_v1',
        source_report=str(path),source_sha256=hashlib.sha256(raw).hexdigest(),
        initial_seed_options=options,attachment_translation_m=(current-old_attachment).tolist(),
        prior_checks_inherited=False,prior_path_replayed=False,motion_authorized=False,
        floor_height_recomputed=True,zero_motion_search_only=True)


def local_stations(original):
    from scipy.spatial.transform import Rotation
    original=np.asarray(original,float)
    for back,side,yaw in [(0.,0.,0.)]+[(back,side,yaw)
            for back in (.025,.05,.10,.15)
            for side,yaw in ((0.,0.),(.05,0.),(-.05,0.),(0.,-15.),(0.,15.))]:
        base=original.copy();base[:3,3]+=-back*original[:3,0]+side*original[:3,1]
        base[:3,:3]=Rotation.from_euler('z',yaw,degrees=True).as_matrix()@original[:3,:3]
        yield base,dict(reference_local=True,base_back_m=back,base_side_m=side,
                        base_yaw_delta_degrees=yaw)
