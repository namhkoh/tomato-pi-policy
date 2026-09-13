from pathlib import Path
import pytest
from . import benchmark,ground_truth_trial


def argv(monkeypatch,tmp_path):
    captured=[]
    with monkeypatch.context() as m:
        m.setattr(benchmark,'main',lambda a:captured.extend(a))
        ground_truth_trial.main(['--output',str(tmp_path/'new'),'--mode','bimanual',
            '--milestone','cut_action','--process-zone-trial','--through-stroke-trial',
            '--material-clearance-trial','--postcut-egress-trial','--coupled-fingers-trial',
            '--solver-convergence-trial','64'])
    return captured


def test_explicit_comparison_preserves_all_other_native_profile_guards(monkeypatch,tmp_path):
    options=argv(monkeypatch,tmp_path);a=benchmark.parser().parse_args(options)
    assert a.uniform_solver_iterations==[64,0] and a.physics_hz==480
    assert a.native_static_clearance and a.native_startup_clearance and a.coupled_fingers_trial
    assert a.material_clearance_trial and a.require_retention_screen
    assert a.postrelease_feed_m_s==.0003 and a.solver=='PGS'
    class Validated(Exception):pass
    monkeypatch.setattr(Path,'mkdir',lambda *a,**kw:(_ for _ in ()).throw(Validated()))
    with pytest.raises(Validated):benchmark.main(options)


@pytest.mark.parametrize('bad',['implicit_lower_count','wrong_count','missing_section'])
def test_no_silent_reduction_or_partial_fixture(monkeypatch,tmp_path,bad):
    options=argv(monkeypatch,tmp_path)
    if bad=='implicit_lower_count':
        i=options.index('--solver-convergence-trial');options[i:i+2]=[]
    if bad=='wrong_count':options+=['--uniform-solver-iterations','128','0']
    if bad=='missing_section':options.remove('--material-clearance-trial')
    with pytest.raises(ValueError):benchmark.main(options)
    assert not (tmp_path/'new').exists()
