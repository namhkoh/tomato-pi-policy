from copy import deepcopy
import pytest
from sim_data.training_plan import configuration,view_specs
from sim_data.training_views import robot_for_spec,with_lean


def test_lean_is_explicit_deterministic_and_preserves_other_proposals():
    original=view_specs(.8,0,'target',3,vary_torso=True,clear_capture=True)
    before=deepcopy(original)
    lean=view_specs(.8,0,'target',3,vary_torso=True,clear_capture=True,lean_clear=True)
    assert original==before and lean==with_lean(original,'target')
    assert lean==view_specs(.8,0,'target',3,vary_torso=True,clear_capture=True,lean_clear=True)
    for old,new in zip(original,lean):
        assert 5<=new['torso_lean_degrees']<=30
        assert {k:v for k,v in new.items() if k not in ('candidate_id','torso_lean_degrees')}=={k:v for k,v in old.items() if k!='candidate_id'}
    with pytest.raises(ValueError):configuration(lean_clear=True)
    with pytest.raises(ValueError):view_specs(.8,0,'target',3,lean_clear=True)
    assert configuration(views=3,vary_torso=True,clear_capture=True,lean_clear=True)['lean_clear']


def test_lean_moves_real_head_forward_within_exact_urdf_limits():
    from greenhouse_sim.robot_kinematics import Rby1Kinematics
    from greenhouse_sim.robot_scene import SDK_READY_POSE_DEGREES
    model=Rby1Kinematics();robot={'pose_degrees':dict(SDK_READY_POSE_DEGREES)}
    before=deepcopy(robot)
    for bend in (0.,15.,30.,45.):
        old=robot_for_spec(robot,dict(torso_bend_degrees=bend))
        a=model.all_link_transforms(old['pose_degrees'])
        for lean in (5.,20.,30.):
            new=robot_for_spec(robot,dict(torso_bend_degrees=bend,torso_lean_degrees=lean))
            b=model.all_link_transforms(new['pose_degrees'])
            assert b['link_head_2'][0,3]>a['link_head_2'][0,3]
            assert all(new['pose_degrees'][n]==v for n,v in old['pose_degrees'].items() if n!='torso_3')
    assert robot==before


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-1,31,True])
def test_invalid_lean_rejected(value):
    with pytest.raises(ValueError):robot_for_spec({'pose_degrees':{}},dict(torso_bend_degrees=10,torso_lean_degrees=value))


def test_lean_without_bend_rejected():
    with pytest.raises(ValueError):robot_for_spec({'pose_degrees':{}},dict(torso_lean_degrees=10))
