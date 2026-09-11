import pytest
from .bimanual_probe import experimental_spring_phase


@pytest.mark.parametrize('enabled,verified,started,result',[
    (False,False,False,'legacy'),(False,True,False,'legacy'),
    (True,False,False,'initialize'),(True,True,False,'contact'),(True,True,True,'contact')])
def test_explicit_hold_initialization_then_no_hidden_fallback(enabled,verified,started,result):
    assert experimental_spring_phase(enabled,verified,started)==result


@pytest.mark.parametrize('args',[(True,False,True),(False,True,True),(False,False,True)])
def test_started_experiment_cannot_silently_revert_to_legacy(args):
    with pytest.raises(RuntimeError):experimental_spring_phase(*args)


def test_booleans_required():
    with pytest.raises(ValueError):experimental_spring_phase(1,True,False)
