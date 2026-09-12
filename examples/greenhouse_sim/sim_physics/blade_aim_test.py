from types import SimpleNamespace as S
import numpy as np
import pytest
from sim_physics.blade_aim import edge_centre
from sim_physics.downward_transit_continuity_test import redundant_robot
from sim_physics.benchmark import parser, main


@pytest.mark.parametrize('offset',[0.,-.001,.0015,-.0015])
def test_aim_uses_actual_axis_and_does_not_mutate_seam(offset):
    seam=np.array([1.,2.,3.]);axis=np.array([.6,.8,0.]);copy=seam.copy()
    aim=edge_centre(seam,axis,offset)
    np.testing.assert_allclose(aim,seam+offset*axis)
    np.testing.assert_array_equal(seam,copy)
    assert not np.shares_memory(aim,seam)


@pytest.mark.parametrize('bad',[True,None,float('nan'),float('inf'),.0015001,-.0015001,'0'])
def test_aim_cannot_expand_to_an_unbounded_new_cut_location(bad):
    with pytest.raises(ValueError):edge_centre(np.zeros(3),np.array([0,0,1]),bad)


@pytest.mark.parametrize('axis',[[0,0,0],[0,0,2],[0,1], [float('nan'),0,1]])
def test_actual_measured_axis_required(axis):
    with pytest.raises(ValueError):edge_centre(np.zeros(3),axis,-.001)


def test_stroke_aim_and_ground_truth_release_centre_remain_separate(monkeypatch):
    import sim_physics.downward_cut as module
    monkeypatch.setattr(module,'arm_extension',lambda *a:.9)
    r=redundant_robot();r.blade_axial_aim_offset_m=-.001
    def fk(side,q,base):
        m=np.eye(4);m[:3,3]=q[:3];return m
    def solve(m,q):
        v=q.copy();v[:3]=m[:3,3];return S(succeeded=True,joint_degrees=v)
    r.kin.forward=fk;r.solve_right_pose=solve
    seam=np.array([.01,0,0]);axis=np.array([0.,0,1.]);q=np.array([.01,0,-.001,0,0,0,0])
    candidate=(0.,None,np.array([1.,0,0]),q,1,0.,np.array([0.,0,1.]))
    assert r._try_cut_candidate(np.zeros(7),seam,axis,candidate,0.,[])
    np.testing.assert_array_equal(r.plan['centre'],seam)
    np.testing.assert_array_equal(r.plan['axis'],axis)
    np.testing.assert_allclose(r.plan['blade_aim_centre'],[.01,0,-.001])
    np.testing.assert_array_equal(r.plan['approach'][-1],r.plan['stroke'][0])
    assert not r.plan['release_seam_or_tolerance_changed']


def test_nondefault_aim_cannot_enable_a_production_run(tmp_path):
    assert parser().parse_args(['--output','unused']).blade_axial_aim_offset_m==0
    with pytest.raises(ValueError,match='isolated downward'):
        main(['--output',str(tmp_path/'unused'),'--blade-axial-aim-offset-m','-.001'])
    assert not (tmp_path/'unused').exists()
