import numpy as np
import pytest
from sim_physics.force_closure import ForceClosure
from sim_physics.finger_target_antiwindup import project


def test_asymmetric_support_does_not_translate_commanded_grasp_center():
    c=ForceClosure(.003,.001,retention_preload=True,symmetric=True)
    c.gaps[:]=.003;c.support[:]=[.32,.16];c.loads[:]=[.35,.19]
    result=c.command(1.,step=0,dt=1/240)
    np.testing.assert_allclose(result,[.003,.003])
    assert c.receipt['symmetric_aperture_command']
    assert not c.receipt['physical_gear_coupling_modeled']


def test_common_aperture_closes_under_low_support_and_opens_under_one_sided_overload():
    for load,expected in ((.1,.003-.0005/240),(.41,.003+.005/240)):
        c=ForceClosure(.003,.001,retention_preload=True,symmetric=True)
        c.gaps[:]=.003;c.loads[:]=[load,0]
        result=c.command(1.,step=0,dt=1/240)
        np.testing.assert_allclose(result,[expected,expected])


def test_independent_legacy_load_response_stays_available():
    c=ForceClosure(.003,.001,retention_preload=True)
    c.gaps[:]=.003;c.support[:]=[.32,.16];c.loads[:]=[.35,.19]
    result=c.command(1.,step=0,dt=1/240)
    assert result[0]>.003 and result[1]<.003


def test_shared_projection_intersects_both_pd_budgets_without_moving_center():
    gaps,r=project([.005,.005],[-.003,.0034],[0,0],[.3,.3],minimum=.002,
        retention_preload=True,symmetric=True)
    np.testing.assert_allclose(gaps,[.0045,.0045])
    assert sum(r['projected_targets_m'])==0
    assert max(abs(np.array(r['resulting_unclipped_pd_n'])))<=.3+1e-12
    assert not r['changes_physical_state']


def test_no_common_unsaturated_aperture_fails_before_submission():
    with pytest.raises(RuntimeError,match='No symmetric aperture'):
        project([.003,.003],[-.003,.01],[0,0],[.3,.3],minimum=.002,
            retention_preload=True,symmetric=True)


def test_no_hidden_recentering_of_invalid_command_state():
    c=ForceClosure(.003,.001,symmetric=True);c.gaps[:]=[.003,.004]
    with pytest.raises(RuntimeError,match='fixed center'):c.command(1.,step=0,dt=1/240)
    with pytest.raises(ValueError,match='Symmetric raw targets'):
        project([.003,.004],[-.003,.003],[0,0],[.15,.15],minimum=.002,symmetric=True)


@pytest.mark.parametrize('value',[None,1,'yes'])
def test_symmetric_mode_requires_explicit_bool(value):
    with pytest.raises(ValueError):ForceClosure(.003,.001,symmetric=value)
    with pytest.raises(ValueError):project([.003,.003],[-.003,.003],[0,0],[.15,.15],minimum=.002,symmetric=value)


def test_cli_symmetric_mode_cannot_escape_isolated_protocol(tmp_path):
    from sim_physics.benchmark import main
    with pytest.raises(ValueError,match='Symmetric finger closure'):
        main(['--output',str(tmp_path/'unused'),'--symmetric-finger-closure'])
    assert not (tmp_path/'unused').exists()
