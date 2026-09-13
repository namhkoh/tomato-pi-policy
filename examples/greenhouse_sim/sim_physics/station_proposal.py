"""Read initial-pose proposals, never inherit a prior trial's motion authority."""
import hashlib
import json
from pathlib import Path
import numpy as np


def apply_to_arguments(args):
    intact=getattr(args,'greenhouse_cut_trial',False)
    isolated=(getattr(args,'isolate_station',False) and getattr(args,'branch_contact_fixture',False))
    if not ((intact or isolated) and args.native_startup_clearance and args.native_static_clearance
            and not any(getattr(args,name,False) for name in ('native_startup_pose_search',
                'native_startup_station_search','native_startup_heading_search','native_startup_approach_search'))):
        raise ValueError('Station proposal requires a NEW explicit greenhouse/isolated fixture with complete native controls')
    path=Path(args.station_proposal_report).resolve();raw=path.read_bytes()
    if len(raw)>2_000_000:raise ValueError('Bounded native startup report required')
    report=json.loads(raw);screen=report.get('native_startup_collision_screen',{})
    search=screen.get('right_pose_search',{});proposal=search.get('proposed_station')
    if (search.get('model')!='frozen_native_two_arm_station_search_v1'
            or search.get('final_native_controls_passed') is not True
            or search.get('physics_steps')!=0 or screen.get('physics_steps')!=0
            or search.get('original_spawn_unchanged') is not True
            or search.get('motion_authorized') is not False or not isinstance(proposal,dict)):
        raise ValueError('Unchanged zero-step final-controlled station proposal required')
    config=report.get('configuration',{})
    if config.get('park_left_ready',False)!=getattr(args,'park_left_ready',False):
        raise ValueError('Station proposal task mismatch: parked-left strategy')
    if config.get('target_row_slot',12)!=getattr(args,'target_row_slot',12):
        raise ValueError('Station proposal task mismatch: original planting slot')
    for key in ('approach_vector','approach_distance'):
        if config.get(key)!=getattr(args,key,None):
            raise ValueError('Station proposal task mismatch: '+key)
    # Same anatomical task and robot geometry; no mismatched grasp/torso recipe.
    keys=('plant','target','grasp_arc_m','grasp_roll','grasp_pitch','grasp_depth_m',
          'torso_degrees','cut_arc_m','knife_edge_mode','knife_alignment','stem_contact_model')
    for key in keys:
        if config.get(key)!=getattr(args,key):raise ValueError('Station proposal task mismatch: '+key)
    values={}
    for key,n in (('station_pose',3),('left_ik_seed_degrees',7),('right_ready_degrees',7)):
        value=proposal.get(key)
        if (not isinstance(value,list) or len(value)!=n
                or any(type(v) not in (int,float) for v in value)
                or not np.isfinite(value).all()):raise ValueError('Finite initial proposal required: '+key)
        values[key]=list(map(float,value))
    for key,value in values.items():setattr(args,key,value)
    return dict(model='fresh_launch_from_unprivileged_station_proposal_v1',source_report=str(path),
        source_sha256=hashlib.sha256(raw).hexdigest(),initial_pose_proposal=values,
        destination_scope='intact_greenhouse' if intact else 'isolated_source_branch_fixture',
        prior_path_replayed=False,prior_native_checks_inherited=False,
        fresh_startup_and_path_required=True,motion_authorized=False)
