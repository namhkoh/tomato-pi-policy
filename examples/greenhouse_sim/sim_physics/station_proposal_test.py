import json
from types import SimpleNamespace as S
import pytest
from .station_proposal import apply_to_arguments


def fixture(tmp_path):
    task=dict(plant='seed101_full',target='SubStem_41',grasp_arc_m=.08,grasp_roll=0,
        grasp_pitch=20.,grasp_depth_m=.125,torso_degrees=[0.]*6,cut_arc_m=.02,
        knife_edge_mode='source_crossbar_edge_v1',knife_alignment='camera',stem_contact_model='flat_cylinders_v1')
    args=S(**task,station_proposal_report=tmp_path/'report.json',greenhouse_cut_trial=True,
        native_startup_clearance=True,native_static_clearance=True,station_pose=[9,9,9],
        right_ready_degrees=[1]*7,left_ik_seed_degrees=[2]*7)
    search=dict(model='frozen_native_two_arm_station_search_v1',final_native_controls_passed=True,
        physics_steps=0,original_spawn_unchanged=True,motion_authorized=False,
        proposed_station=dict(station_pose=[.5,0.,0.],right_ready_degrees=[0.]*7,
            left_ik_seed_degrees=[.1]*7,left_path_degrees=['not_replayed']))
    report=dict(configuration=task,native_startup_collision_screen=dict(physics_steps=0,right_pose_search=search))
    return args,report,search


def test_new_launch_uses_only_initial_values_never_replays_path(tmp_path):
    args,report,_=fixture(tmp_path);args.station_proposal_report.write_text(json.dumps(report))
    receipt=apply_to_arguments(args)
    assert args.station_pose==[.5,0.,0.] and args.left_ik_seed_degrees==[.1]*7
    assert not hasattr(args,'left_path_degrees') and not receipt['prior_path_replayed']
    assert not receipt['prior_native_checks_inherited'] and not receipt['motion_authorized']
    assert len(receipt['source_sha256'])==64
    assert receipt['cut_frame_priority'] is None  # Old reports remain usable.


def test_new_station_carries_frame_order_not_a_cut_or_motion_certificate(tmp_path):
    args,report,search=fixture(tmp_path)
    search['proposed_station']['cut_frame_family']=dict(tilt=15.,normal_sign=-1,wing_m=0.)
    args.station_proposal_report.write_text(json.dumps(report))
    r=apply_to_arguments(args);p=r['cut_frame_priority']
    assert p['tilt']==15. and p['normal_sign']==-1 and p['wing_m']==0.
    assert p['order_only'] and p['zero_motion_station_family_only']
    assert not p['motion_authorized'] and not p['prior_cut_success_claimed']
    assert not p['prior_pose_or_path_replayed'] and r['fresh_startup_and_path_required']


@pytest.mark.parametrize('family',[None,{},dict(tilt=90,normal_sign=1,wing_m=0),
    dict(tilt=15,normal_sign=True,wing_m=0),dict(tilt=15,normal_sign=1,wing_m=float('nan')),
    dict(tilt=15,normal_sign=1,wing_m=.021),dict(tilt=15,normal_sign=1,wing_m=0,path='replay')])
def test_invalid_family_rejected_before_mutating_any_initial_arguments(tmp_path,family):
    args,report,search=fixture(tmp_path);search['proposed_station']['cut_frame_family']=family
    args.station_proposal_report.write_text(json.dumps(report))
    with pytest.raises(ValueError,match='family'):apply_to_arguments(args)
    assert args.station_pose==[9,9,9] and args.right_ready_degrees==[1]*7


def test_same_initial_proposal_in_isolation_still_requires_fresh_all_shape_checks(tmp_path):
    args,report,_=fixture(tmp_path);args.greenhouse_cut_trial=False
    args.isolate_station=True;args.branch_contact_fixture=True
    args.station_proposal_report.write_text(json.dumps(report))
    receipt=apply_to_arguments(args)
    assert receipt['destination_scope']=='isolated_source_branch_fixture'
    assert receipt['fresh_startup_and_path_required']
    assert not receipt['prior_native_checks_inherited'] and not receipt['motion_authorized']


@pytest.mark.parametrize('key,value',[('approach_vector',[1.,0.,0.]),('approach_distance',.02)])
def test_grasp_side_or_approach_cannot_inherit_a_different_initial_proposal(tmp_path,key,value):
    args,report,_=fixture(tmp_path);setattr(args,key,value)
    args.station_proposal_report.write_text(json.dumps(report))
    with pytest.raises(ValueError,match=key):apply_to_arguments(args)
    assert args.station_pose==[9,9,9]


@pytest.mark.parametrize('change',['final_control','steps','target','pose_nan','missing_pose','wrong_scene','another_search'])
def test_bad_proposal_cannot_change_arguments(tmp_path,change):
    args,report,search=fixture(tmp_path)
    if change=='final_control':search['final_native_controls_passed']=False
    elif change=='steps':search['physics_steps']=1
    elif change=='target':report['configuration']['target']='other'
    elif change=='pose_nan':search['proposed_station']['right_ready_degrees'][0]=float('nan')
    elif change=='missing_pose':del search['proposed_station']['left_ik_seed_degrees']
    elif change=='wrong_scene':args.greenhouse_cut_trial=False
    elif change=='another_search':args.native_startup_station_search=True
    args.station_proposal_report.write_text(json.dumps(report))
    with pytest.raises(ValueError):apply_to_arguments(args)
    assert args.station_pose==[9,9,9] and args.right_ready_degrees==[1]*7
