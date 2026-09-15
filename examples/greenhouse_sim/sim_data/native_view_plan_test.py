"""Capture viewpoints are bounded, physical-camera snapshots, not answer templates."""
import math
import numpy as np
import pytest
from sim_data.native_view_plan import propose_specs,implementation_paths
from sim_data import native_view_plan as planner


def test_bound_implementation_paths_exist_in_actual_repo_layout():
    paths=implementation_paths()
    assert len(set(p.resolve() for p in paths))==len(paths)
    assert all(p.is_file() for p in paths)
    assert {'native_scene.py','native_generated_views.py','automated_native_review.py',
            'launch_sim_data.py','robot_hardware.py','robot_kinematics.py'}<={p.name for p in paths}


def test_cli_creates_new_parent_and_refuses_overwrite(tmp_path,monkeypatch):
    import json
    import sys
    output=tmp_path/'new_experiment'/'plan.json'
    monkeypatch.setattr(sys,'argv',['planner','--base-plan','unused.json','--output',str(output)])
    monkeypatch.setattr(planner,'build',lambda *args: {'views':[{'id':'fixture'}]})
    monkeypatch.setattr(planner,'check',lambda _: None)
    planner.main()
    assert json.loads(output.read_text())=={'views':[{'id':'fixture'}]}
    before=output.read_bytes()
    with pytest.raises(ValueError,match='New view plan only'):planner.main()
    assert output.read_bytes()==before


def root(x,y):
    m=np.eye(4);m[3,:3]=[x,y,.4]
    if x>0:m[:2,:2]*=-1
    return m.tolist()


@pytest.mark.parametrize('side',[-1,1])
def test_bounded_deterministic_reference_heading_views(side):
    r=root(side*.32,.05);target=[0,0,1.4]
    specs=propose_specs(r,target,'recipe',6)
    assert specs==propose_specs(r,target,'recipe',6)
    assert len({s['candidate_id'] for s in specs})==6
    positions=[]
    for s in specs:
        dx=s['root_x_m'];dy=s['y_offset_m']
        assert np.sign(dx)==side and abs(dx)>=.32
        assert s['opposite_aisle']==(side<0) and not s['orbit_clear'] and not s['near_clear']
        assert s['root_yaw_degrees']==(180 if side>0 else 0)
        assert s['base_heading_preserved'] and s['base_displacement_from_reference_m']<=.057
        assert 848*.30<s['desired_pixel_xy'][0]<848*.70
        assert 408*.30<s['desired_pixel_xy'][1]<408*.65
        positions.append([dx,dy])
    distances=np.linalg.norm(np.array(positions)[:,None]-np.array(positions)[None,:],axis=2)
    assert distances[np.triu_indices(6,1)].min()>=.03
    assert len({tuple(s['desired_pixel_xy']) for s in specs})==6


def test_reference_yaw_is_not_replaced_by_target_bearing():
    matrix=np.array(root(.49,.51));yaw=math.radians(197)
    matrix[:2,:2]=[[math.cos(yaw),math.sin(yaw)],[-math.sin(yaw),math.cos(yaw)]]
    specs=propose_specs(matrix,[.13,.63,1.58],'formerly_hidden')
    assert all(abs(s['root_yaw_degrees']-197)<1e-9 for s in specs)
    assert specs[0]['root_x_m']==pytest.approx(.49)
    assert specs[0]['y_offset_m']+.63==pytest.approx(.51)


def test_framing_changes_with_recipe_identity_not_a_fixed_answer_location():
    a=propose_specs(root(.3,0),[0,0,1.4],'one')
    b=propose_specs(root(.3,0),[0,0,1.4],'two')
    assert [s['desired_pixel_xy'] for s in a]!=[s['desired_pixel_xy'] for s in b]


@pytest.mark.parametrize('count',[0,7,True,2.0])
def test_bad_count(count):
    with pytest.raises(ValueError):propose_specs(root(.3,0),[0,0,1.4],'one',count)


@pytest.mark.parametrize('r,t',[(root(0,.3),[0,0,1]),(root(np.nan,0),[0,0,1]),(np.eye(3),[0,0,1]),(root(.3,0),[0,0])])
def test_invalid_or_ambiguous_geometry(r,t):
    with pytest.raises(ValueError):propose_specs(r,t,'one')
