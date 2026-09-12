import pytest
from sim_physics.benchmark import main,parser


@pytest.mark.parametrize('extra',[
    [],['--scene','package'],['--bimanual-cut'],['--bimanual-hold-control'],
    ['--explicit-finger-effort','--force-closure'],['--uniform-solver-iterations','128','0'],
    ['--stem-contact-model','flat_cylinders_v1']])
def test_incomplete_or_production_fixture_cannot_authorize_cut_diagnostic(tmp_path,extra):
    output=tmp_path/'unused'
    with pytest.raises(ValueError,match='complete explicit flat'):
        main(['--output',str(output),'--isolated-cut-contact-trial',*extra])
    assert not output.exists()


def test_cut_diagnostic_is_not_default():
    assert not parser().parse_args(['--output','unused']).isolated_cut_contact_trial
