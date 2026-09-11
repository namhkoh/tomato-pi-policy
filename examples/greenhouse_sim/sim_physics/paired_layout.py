"""Offline paired-grasp/cutter layout proposals; never authorize execution.

Uses supplied plant/robot geometry and exact existing knife mounting. Does not
step physics, use a grasp weld, change source assets, or create training data.
Native full-greenhouse validation must follow any returned proposal.
"""
import argparse
import itertools
import json
import time
from pathlib import Path

import numpy as np


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-layouts',type=int,default=24)
    parser.add_argument('--station-left',type=float,nargs='+',default=[.2285,.12,0.])
    parser.add_argument('--station-forward',type=float,default=.04)
    parser.add_argument('--multi-seed',action='store_true')
    parser.add_argument('--fixed-shoulders',type=float,nargs='+')
    parser.add_argument('--pose-family',type=int,default=0,help='Opt-in bounded continuation steps per direction, 0..32')
    parser.add_argument('--station-pose',type=float,nargs=3)
    parser.add_argument('--target',default='SubStem_41')
    parser.add_argument('--plant',default='seed101_full')
    parser.add_argument('--cut-standoff',type=float,default=.025)
    parser.add_argument('--grasp-depth',type=float,default=.125)
    parser.add_argument('--grasp-skew',type=float,default=0.)
    parser.add_argument('--approach-vector',type=float,nargs=3)
    parser.add_argument('--approach-tilts',type=float,nargs='+',default=[10.,-10.,30.])
    parser.add_argument('--radial-stations',action='store_true')
    parser.add_argument('--station-radii',type=float,nargs='+',default=[.45,.5,.55])
    parser.add_argument('--torso',type=float,nargs=6,default=[0.]*6)
    parser.add_argument('--grasp-arcs',type=float,nargs='+',default=[.05,.08])
    parser.add_argument('--station-headings',type=float,nargs='+',default=[-120.,-90.,180.,-150.,120.,0.])
    args=parser.parse_args(argv)
    if args.output.exists() or not 1<=args.max_layouts<=100 or not 0<=args.pose_family<=32:
        raise ValueError('New output and bounded layout/continuation counts required')
    args.output.mkdir(parents=True)
    from pxr import Gf,Usd,UsdGeom
    from sim_data.audit import DEFAULT_PACK,audit_manifest
    from sim_data.geometry import assemble_plant
    from .plant import build
    from .bimanual import BimanualRobot
    from .startup_screen import screen
    manifest=DEFAULT_PACK/'plants/components'/args.plant/'manifest.json'
    stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    audit=audit_manifest(manifest);paths=assemble_plant(stage,'/World/Plant',audit)
    stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant')).AddTranslateOp().Set(Gf.Vec3d(-.005,0,.9))
    record=dict(manifest_path=str(manifest),component_paths=paths,plant_root='/World/Plant')
    rig=build(stage,record,args.target);results=[];start=time.perf_counter()
    # Rank deterministic alternatives; retain original scene/guard geometry.
    configs=itertools.product(args.grasp_arcs,args.approach_tilts,(60.,0.),args.station_left,(180,0))
    options_list=[]
    if args.station_pose:
        for arc,tilt,roll in itertools.product(args.grasp_arcs,args.approach_tilts,(180,0)):
            options_list.append(dict(arc=arc,grasp_roll=roll,approach_tilt=tilt,
                station_pose=args.station_pose,grasp_depth=.125,approach_distance=.02))
    elif args.radial_stations:
        for arc,distance,yaw,vector,roll in itertools.product(args.grasp_arcs,args.station_radii,args.station_headings,
                ((1.,-1.,.2),(0.,0.,1.),(-1.,1.,.2)),(180,0)):
            forward=np.array([np.cos(np.radians(yaw)),np.sin(np.radians(yaw))])
            xy=rig.chain_world[rig.cut_index,:2]-distance*forward
            options_list.append(dict(arc=arc,grasp_roll=roll,approach_tilt=0.,approach_vector=vector,
                station_pose=[*xy,yaw],grasp_depth=.125,approach_distance=.02))
    else:
        for arc,tilt,yaw,left_offset,roll in configs:
            options_list.append(dict(arc=arc,grasp_roll=roll,approach_tilt=tilt,station_yaw=yaw,
                station_offset=(args.station_forward,left_offset),grasp_depth=.125,approach_distance=.02))
    for number,options in enumerate(options_list):
        if number>=args.max_layouts: break
        options['grasp_depth']=args.grasp_depth
        options['grasp_skew']=args.grasp_skew
        if args.approach_vector is not None: options['approach_vector']=args.approach_vector
        result=dict(options=options,torso_degrees=args.torso,endpoint_counts={},proposals=[],rejections=[]);results.append(result)
        print('PAIRED_LAYOUT '+json.dumps(options),flush=True)
        try:
            robot=BimanualRobot(stage,rig,sparse_contacts=True,torso_degrees=args.torso,
                cut_standoff=args.cut_standoff,ground_height=lambda x,y:.101,**options)
            spawn=screen(stage,robot)
            if not spawn['passed']: raise ValueError('Spawn overlaps: '+str(spawn['possible_overlaps']))
            left=robot.path_q[int(np.argmin(abs(robot.fractions-1.)))]
            robot.planning_slides={'gripper_finger_l1':-robot.radius,'gripper_finger_l2':robot.radius}
            centre,axis=robot.seam(rig.rest_frames)
            direction=-robot.goal[:3,2];direction-=axis*np.dot(direction,axis);direction/=np.linalg.norm(direction)
            robot.held_plant_screen.include_static_scene(stage,robot.root,rig.root,
                robot.body_world(robot.initial_q,robot.right)['link_right_arm_0'][:3,3])
            robot.held_plant_screen.snapshot(rig.rest_frames)
            result['grasp_screen']=robot.screen_grasp_scene(rig.rest_frames)
            if not result['grasp_screen']['passed']:
                raise ValueError('Left grasp corridor intersects unintended plant geometry')
            seeds=[robot.right]
            if args.multi_seed:
                lower,upper=robot.kin.arm_limits_degrees('right')
                for swivel,wrist in ((-60.,-120.),(60.,120.),(0.,-120.),(0.,120.)):
                    seed=robot.right.copy();seed[2]=swivel;seed[6]=wrist
                    seeds.append(np.clip(seed,lower+.1,upper-.1))
            if args.fixed_shoulders: seeds=[robot.right]*(1+len(args.fixed_shoulders))
            for angle,wing,normal_sign,seed_index in itertools.product(range(-180,180,30),(0.,-.018,.018),(1,-1),range(len(seeds))):
                d=direction*np.cos(np.radians(angle))+np.cross(axis,direction)*np.sin(np.radians(angle))
                normal=normal_sign*axis
                desired=robot.knife.wrist_for_edge(centre+robot.stroke_offsets[0]*d,d,normal,wing)
                if args.fixed_shoulders and seed_index:
                    from .redundant_ik import solve_fixed_joint
                    solution=solve_fixed_joint(robot.kin,'right',desired,seeds[seed_index],robot.base,
                        joint_degrees=args.fixed_shoulders[seed_index-1],maximum_evaluations=180)
                else: solution=robot.kin.solve_pose('right',desired,seeds[seed_index],robot.base,maximum_evaluations=180)
                from .redundant_ik import pose_family
                family=pose_family(robot.kin,'right',desired,solution.joint_degrees,robot.base,
                    steps_per_direction=args.pose_family) if args.pose_family and solution.succeeded else ()
                for family_index,solution in enumerate(itertools.chain([solution],family)):
                    reason='ik';q=np.asarray(solution.joint_degrees)
                    if solution.succeeded:
                        reason='interarm'
                        if robot.kin.inter_arm_clearance(left,q,robot.base).clearance_m>=.01:
                            reason='self'
                            self_result=robot.check_self(left,q)
                            if self_result['passed']:
                                reason='scene'
                                if robot.check_held_plant(left,q):
                                    reason='endpoint_clear'
                                    result['proposals'].append(dict(angle=angle,wing_m=wing,normal_sign=normal_sign,
                                        seed_index=seed_index,family_index=family_index,right_degrees=q.tolist(),left_degrees=left.tolist()))
                    if reason in ('self','scene'):
                        result['rejections'].append(dict(angle=angle,wing_m=wing,normal_sign=normal_sign,seed_index=seed_index,
                            family_index=family_index,reason=reason,detail=self_result if reason=='self' else robot.held_plant_screen.last_failure,
                            right_degrees=q.tolist()))
                    result['endpoint_counts'][reason]=result['endpoint_counts'].get(reason,0)+1
                    if len(result['proposals'])>=3: break
                if len(result['proposals'])>=3: break
            result['robot_base']=robot.base.tolist()
        except Exception as exc: result['error']=str(exc)
        finally:
            stage.RemovePrim('/World/RBY1')
            (args.output/'proposals.json').write_text(json.dumps(dict(layouts=results,
                configuration={**vars(args),'output':str(args.output)},
                wall_s=time.perf_counter()-start,native_validated=False,training_eligible=False),indent=2),encoding='utf-8')
            print('PAIRED_RESULT '+json.dumps(dict(endpoint_counts=result['endpoint_counts'],
                clear_proposals=len(result['proposals']),error=result.get('error'))),flush=True)
        if result['proposals']: break
    return 0


if __name__=='__main__': raise SystemExit(main())
