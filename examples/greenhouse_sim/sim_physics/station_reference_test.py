import json
from types import SimpleNamespace as S
import numpy as np
import pytest
from .station_reference import initial_options,local_stations,validate_mode
from .benchmark import parser
from .ground_truth_trial import arguments


def fixture(tmp_path):
    from greenhouse_sim.robot_model import DEFAULT_ASSET
    a=parser().parse_args(arguments(tmp_path/'unused','bimanual','cut_action')+
        ['--native-startup-station-search','--cut-station-orbit',
         '--station-reference-report',str(tmp_path/'reference.json')])
    config={k:v for k,v in vars(a).items() if not hasattr(v,'read_bytes')}
    report=dict(configuration=config,gates=dict(blade_contact_release=True,
        full_forward_cut_stroke_verified=True,right_withdrawal_completed=True),
        robot=dict(asset=str(DEFAULT_ASSET),ground_truth_grasp=dict(attachment_world_m=[1.,2.,3.]),
            explicit_initial_station_xy_yaw=[1.5,2.5,30.]))
    a.station_reference_report.write_text(json.dumps(report))
    return a,S(chain_world=np.array([[1.,7.5,3.]])),report


def test_translates_only_seed_station_preserves_source_and_requires_new_floor(tmp_path):
    a,rig,r=fixture(tmp_path);old=rig.chain_world.copy();config=list(a.station_pose)
    options,receipt=initial_options(a,rig)
    assert options['station_pose']==[1.5,8.,30.]
    assert options['right_ready_degrees']==a.right_ready_degrees
    assert options['left_ik_seed_degrees']==a.left_ik_seed_degrees
    assert a.station_pose==config
    np.testing.assert_array_equal(rig.chain_world,old)
    assert not receipt['motion_authorized'] and not receipt['prior_checks_inherited']
    assert receipt['floor_height_recomputed'] and receipt['zero_motion_search_only']


@pytest.mark.parametrize('key,value',[('watch_cut_trial',True),('native_startup_station_search',False),
    ('native_static_clearance',False),('native_startup_clearance',False),
    ('right_only_cut_trial',True),('station_proposal_report','another.json')])
def test_cannot_use_reference_as_live_motion_or_prior_clearance(tmp_path,key,value):
    a,rig,_=fixture(tmp_path);setattr(a,key,value)
    with pytest.raises(ValueError,match='zero-motion'):initial_options(a,rig)


@pytest.mark.parametrize('key',['plant','target','grasp_pitch','approach_vector','torso_degrees'])
def test_mismatched_task_cannot_supply_seed(tmp_path,key):
    a,rig,r=fixture(tmp_path);r['configuration'][key]='wrong'
    a.station_reference_report.write_text(json.dumps(r))
    with pytest.raises(ValueError,match='task mismatch'):initial_options(a,rig)


@pytest.mark.parametrize('mutation',['nan','bool','robot','unqualified','too_big'])
def test_invalid_reference_fails_before_pose_construction(tmp_path,mutation):
    a,rig,r=fixture(tmp_path)
    if mutation=='nan':r['robot']['ground_truth_grasp']['attachment_world_m'][0]=float('nan')
    if mutation=='bool':r['configuration']['right_ready_degrees'][0]=True
    if mutation=='robot':r['robot']['asset']='wrong_robot.usd'
    if mutation=='unqualified':r['gates']['full_forward_cut_stroke_verified']=False
    a.station_reference_report.write_text(' '*2_000_001 if mutation=='too_big' else json.dumps(r))
    with pytest.raises(ValueError):initial_options(a,rig)


def test_local_search_keeps_station_height_and_original_matrix():
    base=np.eye(4);base[:3,3]=[.4,.2,.102];old=base.copy();values=list(local_stations(base))
    assert len(values)==21
    np.testing.assert_array_equal(values[0][0],base)
    for m,_ in values:
        assert m[2,3]==base[2,3] and np.linalg.norm(m[:2,3]-base[:2,3])<=.159
        np.testing.assert_allclose(m[:3,:3].T@m[:3,:3],np.eye(3),atol=1e-12)
    np.testing.assert_array_equal(base,old)


def test_local_search_still_requires_native_endpoints(monkeypatch):
    from .cut_station_orbit_test import fixture as robot_fixture
    from .cut_station_orbit import search
    r,_=robot_fixture();r.station_reference_search=True
    result=search(r,S(check=lambda *a:dict(passed=False)),lambda:None)
    assert result['proposed_station'] is None and len(result['candidates'])==315
    assert result['reference_local_seed_search'] and result['physics_steps']==0


def test_public_reference_is_forwarded_only_to_zero_motion_search(tmp_path,monkeypatch):
    from . import ground_truth_trial,benchmark
    seen=[]
    monkeypatch.setattr(benchmark,'main',lambda v:seen.append(parser().parse_args(v)))
    ground_truth_trial.main(['--output',str(tmp_path/'unused'),'--mode','bimanual',
        '--process-zone-trial','--screen-station','--cut-station-orbit',
        '--station-reference-report',str(tmp_path/'reference.json')])
    validate_mode(seen[0])
    assert seen[0].station_reference_report==tmp_path/'reference.json'
