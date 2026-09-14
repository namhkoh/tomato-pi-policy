from pathlib import Path
import pytest
from .dispatcher_setting import DispatcherSetting,SETTING


class Settings:
    def __init__(self,value):self.value=value;self.writes=[]
    def get(self,key):assert key==SETTING;return self.value
    def set_bool(self,key,value):
        assert type(value) is bool
        self.set(key,value)
    def set(self,key,value):assert key==SETTING;self.writes.append(value);self.value=value
    def destroy_item(self,key):assert key==SETTING;self.writes.append('destroy');self.value=None


@pytest.mark.parametrize('previous',[None,False,True])
@pytest.mark.parametrize('mode',[None,'carb','physx'])
def test_explicit_apply_verify_and_exact_restore(previous,mode):
    settings=Settings(previous);scope=DispatcherSetting(settings,mode)
    result=scope.apply()
    expected=previous if mode is None else mode=='physx'
    assert settings.value is expected
    assert result['configured_physx_dispatcher'] is expected
    assert not result['native_effective_scheduler_verified']
    assert not result['solver_iterations_or_contacts_changed']
    scope.close();writes=list(settings.writes);scope.close()
    assert settings.value is previous and settings.writes==writes
    if mode is None:assert not writes
    with pytest.raises(RuntimeError,match='closed'):scope.apply()


def test_lost_setting_refuses_comparison_and_still_restores():
    settings=Settings(False);scope=DispatcherSetting(settings,'physx');scope.apply()
    settings.value=False
    with pytest.raises(RuntimeError,match='changed'):scope.verify()
    scope.close();assert settings.value is False


def test_setter_failure_remains_restorable():
    settings=Settings(None);scope=DispatcherSetting(settings,'physx')
    def fail(*args):settings.value=True;raise RuntimeError('write failed')
    settings.set_bool=fail
    with pytest.raises(RuntimeError,match='write failed'):scope.apply()
    scope.close();assert settings.value is None and settings.writes==['destroy']


@pytest.mark.parametrize('mode',['gpu','',True,1])
def test_bad_modes_never_write(mode):
    settings=Settings(False)
    with pytest.raises(ValueError):DispatcherSetting(settings,mode)
    assert not settings.writes


def test_cli_keeps_default_and_physics_contract(monkeypatch,tmp_path):
    from . import ground_truth_trial,benchmark
    captured=[]
    with monkeypatch.context() as m:
        m.setattr(benchmark,'main',lambda args:captured.extend(args))
        ground_truth_trial.main(['--output',str(tmp_path/'new'),'--mode','right_only',
            '--milestone','cut_action','--process-zone-trial','--through-stroke-trial',
            '--material-clearance-trial','--postcut-egress-trial','--physics-dispatcher','physx'])
    args=benchmark.parser().parse_args(captured)
    assert args.physics_dispatcher=='physx'
    assert args.physics_hz==480 and args.uniform_solver_iterations==[128,0]
    assert args.native_startup_clearance and args.native_static_clearance
    assert args.material_clearance_trial and args.solver=='PGS' and args.gravity==9.81
    assert benchmark.parser().parse_args(['--output','unused']).physics_dispatcher is None
    class Validated(Exception):pass
    monkeypatch.setattr(Path,'mkdir',lambda *a,**k:(_ for _ in ()).throw(Validated()))
    with pytest.raises(Validated):benchmark.main(captured)
