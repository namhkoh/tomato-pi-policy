"""Offline lower-rim ready-pose proposals; no execution or native qualification.

Uses the isolated ground-truth recipe and source assets. The source placement
is the existing isolated station, not a new greenhouse placement. Every result
still needs native startup, approach, grasp, stroke and force validation.
"""
import argparse
import json
import time
import itertools
import numpy as np
from pxr import Usd,UsdGeom,Gf
from .benchmark import parser
from .ground_truth_trial import arguments
from .knife import DOWNWARD_CUT_MODEL
from .blade_contacts import LOWER_EDGE


def fixture(*,source_plant=None,source_target=None,**changes):
    from sim_data.audit import DEFAULT_PACK,audit_manifest
    from sim_data.geometry import assemble_plant
    from .branch_contact_fixture import select_components
    from .plant import build
    from .bimanual import BimanualRobot
    a=parser().parse_args(arguments('unused_offline','bimanual','cut_action'))
    if source_plant is not None:
        if source_plant not in {p.parent.name for p in (DEFAULT_PACK/'plants/components').glob('*/manifest.json')}:
            raise ValueError('An existing original source plant is required')
        a.plant=source_plant
    if source_target is not None:a.target=source_target
    stage=Usd.Stage.CreateInMemory();stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    manifest=DEFAULT_PACK/'plants/components'/a.plant/'manifest.json'
    paths=assemble_plant(stage,'/World/Plant',audit_manifest(manifest))
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant')).AddTranslateOp().Set(Gf.Vec3d(-.005,0,.9))
    record=dict(manifest_path=str(manifest),component_paths=paths,plant_root='/World/Plant')
    select_components(stage,record,a.target)
    rig=build(stage,record,a.target,max_segment_m=a.max_segment_m,cut_m=a.cut_arc_m,
              constraint_mode=a.constraint_mode,stem_contact_model=a.stem_contact_model)
    options={name:getattr(a,name) for name in ('sparse_contacts','compliant_fingers','finger_gravity',
        'finger_actuator_limit_n','exact_grasp_arc','grasp_roll','grasp_pitch','approach_distance',
        'station_pose','torso_degrees','approach_vector','right_ready_degrees','left_ik_seed_degrees',
        'cut_style','knife_alignment','force_closure','effort_bounded_grasp_target','explicit_finger_effort',
        'finger_target_antiwindup','retention_preload','symmetric_finger_closure','physical_grasp_span',
        'preload_force_servo','staged_downward_transit')}
    options.update(arc=a.grasp_arc_m,grasp_depth=a.grasp_depth_m,cut_standoff=a.cut_standoff_m,
        grasp_compression=a.grasp_compression_m,pregrasp_half_aperture=a.pregrasp_half_aperture_m,
        ground_height=lambda x,y:.101,cut_model=DOWNWARD_CUT_MODEL,knife_edge_mode=LOWER_EDGE,
        blade_axial_aim_offset_m=a.blade_axial_aim_offset_m,diagnostic_physics_hz=a.physics_hz)
    options.update(changes)
    return BimanualRobot(stage,rig,**options)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seconds',type=float,default=180.)
    p.add_argument('--approach-tilt',type=float,default=0.)
    p.add_argument('--grasp-pitch',type=float,default=20.)
    p.add_argument('--grasp-roll',type=int,default=0)
    p.add_argument('--plant');p.add_argument('--target')
    a=p.parse_args(argv)
    if not np.isfinite(a.seconds) or not 0<a.seconds<=600:raise ValueError('Bounded offline search required')
    from .downward_cut import vertical_cut_frame,arm_extension
    from .rigid_tool_screen import RigidToolScreen
    r=fixture(source_plant=a.plant,source_target=a.target,
        approach_tilt=a.approach_tilt,grasp_pitch=a.grasp_pitch,grasp_roll=a.grasp_roll)
    centre,axis=r.seam(r.rig.rest_frames);aim=centre+r.blade_axial_aim_offset_m*axis
    start=time.monotonic();seed=r.right.copy();original=r.right.copy()
    r.held_plant_screen.include_static_scene(r.stage,r.root,r.rig.root,
        r.body_world(r.initial_q,r.right)['link_right_arm_0'][:3,3])
    r.held_plant_screen.snapshot(r.rig.rest_frames)
    r.planning_slides={'gripper_finger_l1':-r.radius,'gripper_finger_l2':r.radius}
    screen=RigidToolScreen(r,r.path_q[-1])
    usable=r.knife.size[1]/2-.005
    for extra in (.06,.03):
        for tilt in (15.,10.,0.,-10.,-15.):
            for sign,wing in itertools.product((-1,1),(0.,-usable,usable)):
                if time.monotonic()-start>a.seconds:return
                frame=vertical_cut_frame(axis,sign,tilt)
                if frame is None:continue
                d,normal=frame
                desired=r.knife.wrist_for_edge(aim+(r.stroke_offsets[0]-extra)*d,d,normal,wing)
                stroke=np.repeat(desired[None],len(r.stroke_offsets),axis=0)
                stroke[:,:3,3]+=(extra+r.stroke_offsets-r.stroke_offsets[0])[:,None]*d
                corridor=screen.check(stroke)
                if not corridor['passed']:
                    print(json.dumps(dict(tilt=tilt,normal_sign=sign,wing_m=wing,
                        extra_approach_m=extra,corridor=corridor,native_tested=False)),flush=True)
                    continue
                solve=r.solve_right_pose(desired,seed)
                q=np.asarray(solve.joint_degrees)
                result=dict(tilt=tilt,normal_sign=sign,wing_m=wing,extra_approach_m=extra,ik=bool(solve.succeeded),
                    right_ready_degrees=q.tolist(),native_tested=False,motion_authorized=False)
                if solve.succeeded:
                    result['extension']=arm_extension(r.body_world(r.path_q[-1],q))
                    result['arc_up']=r.knife.arc_up(r.kin.forward('right',q,r.base)).tolist()
                    r.right=q
                    try:
                        r.check_grasp_path();result['left_self_path_passed']=True
                        result['scene']=r.screen_grasp_scene(r.rig.rest_frames)
                    except Exception as exc:result['error']=str(exc)
                    finally:r.right=original.copy()
                print(json.dumps(result),flush=True)


if __name__=='__main__':main()
