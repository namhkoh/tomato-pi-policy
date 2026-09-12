from types import SimpleNamespace
import numpy as np
import pytest
from sim_physics.finger_effort import command,FingerEffort


def test_opening_command_is_outward_and_bounded_with_gravity():
    total,r=command([-.0024,.00349],[0,0],[-.002384,.00569],[-.158,-.158],[.15,.15],dt=1/240)
    assert r['submitted_pd_n'][1]==.15
    np.testing.assert_allclose(total,[.0032-.158,.15-.158])
    assert not r['grasp_verified'] and not r['actual_drive_effort_measured']


@pytest.mark.parametrize('q,v,g,ff,caps,dt',[
    ([0],[0,0],[-.003,.003],[0,0],[.15,.15],1/240),
    ([0,0],[float('nan'),0],[-.003,.003],[0,0],[.15,.15],1/240),
    ([0,0],[0,0],[.003,.003],[0,0],[.15,.15],1/240),
    ([0,0],[0,0],[-.003,.003],[.41,0],[.15,.15],1/240),
    ([0,0],[0,0],[-.003,.003],[0,0],[.151,.15],1/240),
    ([0,0],[0,0],[-.003,.003],[0,0],[.15,.15],1/120)])
def test_invalid_commands_fail(q,v,g,ff,caps,dt):
    with pytest.raises(ValueError):command(q,v,g,ff,caps,dt=dt)


class Articulation:
    count=1
    shared_metatype=SimpleNamespace(dof_names=['arm','gripper_finger_l1','gripper_finger_l2'])
    def __init__(self):
        self.k=np.array([[6000.,200.,200.]],np.float32);self.c=np.array([[160.,5.,5.]],np.float32)
        self.eff=np.array([[10.,-.158,-.158]],np.float32)
        self.target=np.array([[0.,-.003,.003]],np.float32)
    def get_dof_stiffnesses(self):return self.k.copy()
    def get_dof_dampings(self):return self.c.copy()
    def set_dof_stiffnesses(self,x,i):self.k=x.copy()
    def set_dof_dampings(self,x,i):self.c=x.copy()
    def get_dof_positions(self):return np.array([[0.,-.004,.004]],np.float32)
    def get_dof_velocities(self):return np.zeros((1,3),np.float32)
    def get_dof_position_targets(self):return self.target.copy()
    def get_dof_actuation_forces(self):return self.eff.copy()
    def set_dof_actuation_forces(self,x,i):self.eff=x.copy()


def fixture():
    a=Articulation()
    return SimpleNamespace(robot=a,finger_indices=[1,2],index=np.array([0],np.uint32),
        targets=a.target.copy(),force_limits=np.array([[100.,.15,.15]],np.float32),
        finger_compensation=a.eff[0,1:].copy())


def test_native_submission_leaves_other_drives_and_efforts_unchanged():
    f=fixture();c=FingerEffort(f)
    np.testing.assert_array_equal(f.robot.k,[[6000.,0,0]])
    np.testing.assert_array_equal(f.robot.c,[[160.,0,0]])
    c.apply(f,step=0,dt=1/240)
    assert f.robot.eff[0,0]==10
    np.testing.assert_allclose(f.robot.eff[0,1:],[-.008,-.308],atol=1e-7)
    assert c.receipt['submission_readback_verified']
    with pytest.raises(RuntimeError,match='Contiguous'):c.apply(f,step=0,dt=1/240)
    with pytest.raises(RuntimeError,match='gravity-only'):c.apply(f,step=1,dt=1/240)


def test_changed_native_gains_and_stale_targets_reject_before_effort():
    f=fixture();c=FingerEffort(f);before=f.robot.eff.copy();f.robot.k[0,1]=1
    with pytest.raises(RuntimeError,match='gains'):c.apply(f,step=0,dt=1/240)
    np.testing.assert_array_equal(f.robot.eff,before)
    f.robot.k[0,1]=0;f.robot.target[0,1]=-.002
    with pytest.raises(RuntimeError,match='latest targets'):c.apply(f,step=0,dt=1/240)


def test_production_and_cut_cannot_enable_experiment(tmp_path):
    from sim_physics.benchmark import main
    output=tmp_path/'unused'
    with pytest.raises(ValueError,match='feedback HOLD ONLY'):
        main(['--output',str(output),'--explicit-finger-effort'])
    assert not output.exists()
