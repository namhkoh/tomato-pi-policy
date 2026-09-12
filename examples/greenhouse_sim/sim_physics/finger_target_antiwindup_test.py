from types import SimpleNamespace as S
from unittest.mock import Mock
import numpy as np
import pytest
from sim_physics.finger_target_antiwindup import project


def test_recorded_saturated_opening_target_has_no_centimetre_windup():
    q=np.array([-.0039461986,.0022733635]);v=np.array([.000234629,-.000086392])
    raw=np.array([.003419946,.014105273]);before=raw.copy()
    gaps,r=project(raw,q,v,[.15,.15],minimum=.0018388)
    assert gaps[1]<.0031
    np.testing.assert_array_equal(raw,before)
    old_pd=np.clip(200*(np.array([-1,1])*raw-q)-5*v,-.15,.15)
    new_pd=np.clip(200*(np.array([-1,1])*gaps-q)-5*v,-.15,.15)
    np.testing.assert_allclose(old_pd,new_pd,atol=1e-14)
    assert not r['changes_physical_state'] and not r['measured_effort']


@pytest.mark.parametrize('v',[[0,0],[.025,-.025],[-.05,.05]])
def test_projection_includes_damping_and_retains_geometry(v):
    gaps,r=project([.014,.014],[-.003,.003],v,[.1,.15],minimum=.0018)
    assert np.all(gaps>=.0018) and np.all(gaps<=.025)
    if all(r['unsaturated_interval_intersects_geometry']):
        assert np.all(abs(np.array(r['resulting_unclipped_pd_n']))<=np.array([.1,.15])+1e-14)


def test_geometry_bounds_take_precedence_when_interval_is_disjoint():
    gaps,r=project([.025,.025],[-.049,.049],[0,0],[.15,.15],minimum=.002)
    np.testing.assert_array_equal(gaps,[.025,.025])
    assert r['unsaturated_interval_intersects_geometry']==[False,False]


@pytest.mark.parametrize('which,value',[
    (0,[float('nan'),.003]),(0,[.026,.003]),(0,[.001,.003]),
    (1,[.003,.003]),(1,[-.06,.003]),(2,[float('inf'),0]),(2,[0,.052]),
    (3,[.16,.15]),(3,[0,.15])])
def test_invalid_feedback_cannot_update_controller(which,value):
    args=[[.003,.003],[-.003,.003],[0,0],[.15,.15]];args[which]=value
    with pytest.raises(ValueError):project(*args,minimum=.002)


def test_native_integration_updates_only_controller_targets():
    from sim_physics.bimanual import BimanualRobot
    from sim_physics.force_closure import ForceClosure
    a=Mock();a.get_dof_positions.return_value=np.array([[-.003,.003]])
    a.get_dof_velocities.return_value=np.zeros((1,2))
    f=S(force_closure_enabled=True,finger_target_antiwindup=True,force_closer=ForceClosure(.003,.0005),
        force_limits=np.array([[.15,.15]]),finger_indices=[0,1],targets=np.zeros((1,2)),index=[0],robot=a)
    BimanualRobot.close(f,1.,step=0,dt=1/240)
    np.testing.assert_array_equal(f.force_closer.gaps,[-f.targets[0,0],f.targets[0,1]])
    assert f.force_closer.receipt['antiwindup']['model']=='measured_finger_target_antiwindup_v1'
    assert not f.force_closer.receipt['grasp_verified']
    a.set_dof_positions.assert_not_called();a.set_dof_velocities.assert_not_called()


def test_opt_in_never_enables_production_or_native_finger_mode(tmp_path):
    from sim_physics.benchmark import main,parser
    assert not parser().parse_args(['--output','unused']).finger_target_antiwindup
    with pytest.raises(ValueError,match='explicit isolated'):
        main(['--output',str(tmp_path/'unused'),'--finger-target-antiwindup'])
    assert not (tmp_path/'unused').exists()
