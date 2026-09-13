import numpy as np
import pytest
from .cut_watch import OneShot,check_idle,idle_robot_view
from .ground_truth_trial import arguments
from .benchmark import main,parser
from pathlib import Path
from types import SimpleNamespace as S


def test_watch_only_allows_one_explicit_request_no_reset_or_automatic_run():
    s=OneShot();assert s.state=='ready' and not s.take()
    s.request();s.request();assert s.take() and not s.take()
    s.request();assert s.state=='running'
    s.finish();s.request();assert s.state=='finished' and not s.take()
    with pytest.raises(RuntimeError):s.finish()


def test_idle_reader_does_not_require_or_bind_robot_controls():
    view=S(count=1,shared_metatype=S(fixed_base=True))
    calls=[]
    def create(anchor):calls.append(anchor);return view
    fixture=S(anchor='/Robot/root')  # No .robot until the trial binds it.
    sim=S(physics_sim_view=S(create_articulation_view=create))
    assert idle_robot_view(sim,fixture) is view and calls==['/Robot/root']
    assert not hasattr(fixture,'robot')
    view.count=0
    with pytest.raises(RuntimeError):idle_robot_view(sim,fixture)


@pytest.mark.parametrize('changed',['timeline','plant','robot','nan'])
def test_idle_cannot_resume_changed_native_state(changed):
    a=np.zeros((2,3));b=a.copy();q=np.zeros((1,7));v=q.copy()
    if changed=='plant':b[0,0]=.001
    if changed=='robot':v[0,0]=.001
    if changed=='nan':b[0,0]=np.nan
    with pytest.raises(RuntimeError):check_idle(changed!='timeline',a,b,q,v)
    check_idle(True,a,a,q,q)


@pytest.mark.parametrize('mode',['bimanual','right_only'])
def test_watch_profile_preserves_physical_configuration_and_reaches_validation(tmp_path,monkeypatch,mode):
    out=tmp_path/'watch';baseline=vars(parser().parse_args(arguments(out,mode,'cut_action')))
    argv=arguments(out,mode,'cut_action',watch=True)
    watched=vars(parser().parse_args(argv))
    assert {k for k in baseline if baseline[k]!=watched[k]}=={'watch_cut_trial','render_hz'}
    class Validated(Exception):pass
    def stop(self,*a,**kw):assert self==out;raise Validated()
    monkeypatch.setattr(Path,'mkdir',stop)
    with pytest.raises(Validated):main(argv)


def test_watch_needs_explicit_cut_action_contract_before_native_launch():
    with pytest.raises(ValueError):arguments('unused','bimanual',watch=True)


@pytest.mark.parametrize('change',[['--gui'],['--robot-interactive'],['--render-hz','0'],
                                   ['--profile'],['--scene-profile']])
def test_watch_cannot_enable_other_unqualified_modes(tmp_path,change):
    out=tmp_path/'never_create'
    argv=arguments(out,'bimanual','cut_action',watch=True)+change
    with pytest.raises(ValueError,match='Watch requires'):main(argv)
    assert not out.exists()
