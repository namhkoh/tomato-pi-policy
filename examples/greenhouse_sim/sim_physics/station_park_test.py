from copy import deepcopy
import numpy as np
import pytest
from .station_park import StationPark
from .measured_withdrawal_native import _raw_joint_world,RIGHT
from .measured_withdrawal_native_test import setup


def fixture(monkeypatch):
    _,state,f,_,_=setup(monkeypatch)
    values=dict(zip(state.names,state.values[0],strict=True))
    f.right=np.degrees([values[n] for n in RIGHT])
    world={k[len(f.root)+1:]:v.copy() for k,v in state.world.items()}
    return state,f,world


def test_measured_station_not_cached_base_and_no_writes(monkeypatch):
    s,f,w=fixture(monkeypatch);before=s.values.copy();targets=s.targets.copy()
    assert np.isnan(f.base).all()
    p=StationPark(f,w)
    np.testing.assert_allclose(p.pose,w['ee_right'],atol=1e-12)
    np.testing.assert_array_equal(s.values,before);np.testing.assert_array_equal(s.targets,targets)
    assert not p.report()['actual_wrist_used_as_goal'] and not p.report()['native_state_changed']


def test_goal_keeps_original_right_angles_not_current_wrist(monkeypatch):
    s,f,w=fixture(monkeypatch);old=StationPark(f,w).pose.copy()
    s.values[0,s.names.index(RIGHT[1])]+=.01
    values=dict(zip(s.names,s.values[0],strict=True));world=_raw_joint_world(f.kin,values,w['base'])
    p=StationPark(f,world)
    np.testing.assert_allclose(p.pose,old,atol=1e-12)
    assert np.linalg.norm(p.pose[:3,3]-world['ee_right'][:3,3])>.0005


@pytest.mark.parametrize('fault',['q','world','goal','names'])
def test_snapshot_changes_never_pass(monkeypatch,fault):
    s,f,w=fixture(monkeypatch);p=StationPark(f,w)
    if fault=='q':s.values[0,0]+=.000001
    elif fault=='world':w['ee_right'][0,3]+=.000001
    elif fault=='goal':f.right[0]+=.000001
    else:f.robot.shared_metatype.dof_names=s.names[::-1]
    with pytest.raises(RuntimeError):p.verify(w)


def test_inconsistent_native_pose_and_source_limit_goal_rejected(monkeypatch):
    _,f,w=fixture(monkeypatch);bad=deepcopy(w);bad['ee_right'][0,3]+=.001
    with pytest.raises(ValueError,match='FK mismatch'):StationPark(f,bad)
    f.right[0]=f.kin.arm_limits_degrees('right')[0][0]
    with pytest.raises(ValueError,match='strict-source-limit'):StationPark(f,w)


def test_new_mode_does_not_skip_existing_endpoint_or_collision_gates(monkeypatch):
    from .withdrawal_native_check_test import setup as legacy
    from .withdrawal_native_check import check
    import sim_physics.station_park as module
    f,frames,s,wrist,_=legacy(monkeypatch,failure='self')
    class Bound:
        def __init__(self,f,world):self.pose=np.eye(4);self.before=deepcopy(world)
        def verify(self,world):
            assert all(np.array_equal(world[k],self.before[k]) for k in world)
        def report(self):return dict(test_fixture=True)
    monkeypatch.setattr(module,'StationPark',Bound)
    r=check(f,frames,10,native_station_park=True)
    assert r['endpoint_attained'] and not r['right_withdrawal_completed']
    assert not r['self_screen']['passed'] and s.closes==1


def test_cli_requires_complete_diagnostic_before_creating_output(tmp_path):
    from .benchmark import main
    with pytest.raises(ValueError,match='Native-station park'):
        main(['--output',str(tmp_path/'unused'),'--native-station-park-reference'])
    assert not (tmp_path/'unused').exists()
