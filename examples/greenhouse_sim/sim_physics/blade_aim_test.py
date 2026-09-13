from types import SimpleNamespace as S
import numpy as np
import pytest
from sim_physics.blade_aim import edge_centre
from sim_physics.downward_transit_continuity_test import redundant_robot
from sim_physics.benchmark import parser, main


@pytest.mark.parametrize('offset',[0.,-.001,.0015,-.0015,.0023,.0025])
def test_aim_uses_actual_axis_and_does_not_mutate_seam(offset):
    seam=np.array([1.,2.,3.]);axis=np.array([.6,.8,0.]);copy=seam.copy()
    aim=edge_centre(seam,axis,offset)
    np.testing.assert_allclose(aim,seam+offset*axis)
    np.testing.assert_array_equal(seam,copy)
    assert not np.shares_memory(aim,seam)


@pytest.mark.parametrize('bad',[True,None,float('nan'),float('inf'),.0025001,-.0015001,'0'])
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


def test_distal_proposal_clears_original_stump_and_stays_in_original_window():
    from .blade_aim import section_placement
    axis=np.array([np.sqrt(1-.18453**2),0.,-.18453]);normal=np.array([1.,0.,0.])
    r=section_placement(axis,normal,offset_m=.0023,radius=.003184,half_thickness=.0015)
    assert r['passed'] and r['proximal_stump_clearance_m']>.0001
    assert r['maximum_section_axial_distance_m']<.00295
    assert not r['motion_authorized'] and not r['cut_authorized'] and not r['collision_geometry_changed']
    assert section_placement(axis,-normal,offset_m=.0023,radius=.003184,half_thickness=.0015)==r
    # Merely increasing a command is not permission: reject body overlap OR
    # an ellipse that extends past the unchanged original cut window.
    assert not section_placement(axis,normal,offset_m=.0016,radius=.003184,half_thickness=.0015)['passed']
    assert not section_placement(axis,normal,offset_m=.0025,radius=.003184,half_thickness=.0015)['passed']


def test_distal_candidate_cannot_reach_transit_without_geometry_certificate(monkeypatch):
    r=redundant_robot();r.blade_axial_aim_offset_m=.0025;r.cut_shaft_radius=.004
    r.knife.source_crossbar_half_thickness_m=.0015
    def forbidden(*a):raise AssertionError('Rejected material section must not plan a motion')
    r.right_transit=forbidden
    axis=np.array([np.sqrt(.96),0.,.2]);failures=[]
    candidate=(0.,None,np.array([0.,0.,-1.]),np.zeros(7),1,0.,np.array([1.,0.,0.]))
    assert not r._try_cut_candidate(np.zeros(7),np.zeros(3),axis,candidate,0.,failures)
    assert failures[-1]['rejection']=='blade_body_stump_or_original_cut_window'


def test_extended_aim_requires_runtime_material_section_validation(tmp_path):
    from .ground_truth_trial import main as trial
    with pytest.raises(ValueError,match='complete measured material-section'):
        trial(['--output',str(tmp_path/'unused'),'--mode','right_only','--milestone','cut_action',
            '--process-zone-trial','--blade-aim-offset-m','.0023'])
    assert not (tmp_path/'unused').exists()


def test_proximal_halfspace_bounds_the_whole_capped_cylinder():
    from .blade_aim import section_placement
    rng=np.random.default_rng(19);radius=.003184;h=.0015;offset=.0023
    for angle in rng.uniform(-.3,.3,40):
        axis=np.array([np.cos(angle),0.,np.sin(angle)])
        radial=np.array([-np.sin(angle),0.,np.cos(angle)])
        phi=rng.uniform(0,2*np.pi,500)
        # Cap and interior points, never extend the stump past its cut face.
        pts=(rng.uniform(-.02,0,500)[:,None]*axis
             +radius*np.cos(phi)[:,None]*radial
             +radius*np.sin(phi)[:,None]*np.array([0.,1.,0.]))
        actual_gap=offset*axis[0]-h-pts[:,0].max()
        bound=section_placement(axis,np.array([1.,0.,0.]),offset_m=offset,radius=radius,half_thickness=h)
        assert actual_gap>=bound['proximal_stump_clearance_m']-1e-12
