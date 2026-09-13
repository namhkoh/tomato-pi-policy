from types import SimpleNamespace as S
import json
from pathlib import Path
import numpy as np
import pytest
from .ready_lift import propose


def recipe():
    from greenhouse_sim.robot_kinematics import Rby1Kinematics,base_transform
    root=Path(__file__).parent
    argv=json.loads((root/'ground_truth_trial.json').read_text())['argv']
    def vector(name,n):
        i=argv.index(name)+1
        return [float(v) for v in argv[i:i+n]]
    kin=Rby1Kinematics();kin.set_default_torso_degrees(vector('--torso-degrees',6))
    station=vector('--station-pose',3)
    base=base_transform([*station[:2],.10200000149011614],station[2])
    q=np.array(json.loads((root/'crossbar_process_zone_trial.json').read_text())['right_ready_degrees'])
    return kin,q,base


def test_actual_urdf_lift_preserves_arc_orientation_and_has_no_motion_authority():
    kin,q,base=recipe();original=q.copy()
    lifted,receipt=propose(kin,q,base,.01)
    a=kin.forward('right',q,base);b=kin.forward('right',lifted,base)
    assert np.linalg.norm(b[:3,3]-a[:3,3]-[0,0,.01])<.0005
    assert receipt['orientation_error_rad']<.005
    assert not receipt['motion_authorized'] and not receipt['native_pose_changed']
    assert receipt['startup_and_whole_path_checks_required']
    np.testing.assert_array_equal(q,original)


def test_zero_is_unchanged_copy_without_solving():
    q=np.arange(7.)
    result,receipt=propose(None,q,None,0.)
    np.testing.assert_array_equal(result,q)
    assert receipt is None and not np.shares_memory(result,q)


def test_original_tool_can_propose_a_wrist_axis_retreat_not_a_world_up_guess():
    kin,q,base=recipe();a=kin.forward('right',q,base)
    proposed,receipt=propose(kin,q,base,0.,.01);b=kin.forward('right',proposed,base)
    assert np.linalg.norm(b[:3,3]-a[:3,3]-.01*a[:3,2])<.0005
    assert receipt['retreat_wrist_plus_z_m']==.01 and not receipt['motion_authorized']


@pytest.mark.parametrize('value',[True,float('nan'),-.01,.051])
def test_invalid_retreat_refused(value):
    with pytest.raises(ValueError):propose(None,np.zeros(7),None,0.,value)


@pytest.mark.parametrize('value',[True,float('nan'),float('inf'),-.001,.050001,'0.01'])
def test_invalid_proposal_rejected(value):
    with pytest.raises(ValueError):propose(None,np.zeros(7),None,value)


def test_failed_ik_never_produces_a_pose(monkeypatch):
    kin,q,base=recipe()
    monkeypatch.setattr(kin,'solve_pose',lambda *a,**k:S(succeeded=False))
    with pytest.raises(RuntimeError,match='IK failed'):propose(kin,q,base,.01)


def test_false_solver_success_is_independently_rejected(monkeypatch):
    kin,q,base=recipe()
    monkeypatch.setattr(kin,'solve_pose',lambda *a,**k:S(succeeded=True,joint_degrees=q))
    with pytest.raises(RuntimeError,match='independent FK'):propose(kin,q,base,.01)
