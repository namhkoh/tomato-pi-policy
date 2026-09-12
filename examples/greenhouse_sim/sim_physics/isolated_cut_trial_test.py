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


@pytest.mark.parametrize('native',[False,True])
@pytest.mark.parametrize('velocity',[0,8])
def test_complete_explicit_comparison_reaches_precreation_boundary(tmp_path,monkeypatch,native,velocity):
    from pathlib import Path
    class ValidatedTrial(Exception):pass
    output=tmp_path/'unused'
    def stop(self,*args,**kwargs):
        assert self==output
        raise ValidatedTrial()
    monkeypatch.setattr(Path,'mkdir',stop)
    args=['--output',str(output),'--scene','package','--isolate-station','--full-robot-probe',
        '--bimanual-cut','--explicit-finger-effort','--isolated-cut-contact-trial','--force-closure',
        '--stem-contact-model','flat_cylinders_v1','--grasp-contact-frames','pre_solve_pgs_v1',
        '--spring-mode','native' if native else 'implicit_effort','--uniform-solver-iterations','128',str(velocity),
        '--sparse-contacts','--finger-gravity','--compliant-fingers','--seconds','30',
        '--force-newton','0','--solver','PGS']
    if native:args+=['--native-spring-cut-trial']
    with pytest.raises(ValidatedTrial):main(args)
    assert not output.exists()
