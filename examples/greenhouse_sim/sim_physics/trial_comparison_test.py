"""Public experimental overrides preserve the original physical guards."""
import sys
from types import SimpleNamespace
import pytest
from sim_physics import ground_truth_trial


def invoke(monkeypatch, extra):
    result=[]
    monkeypatch.setitem(sys.modules,'sim_physics.benchmark',SimpleNamespace(main=lambda args:result.extend(args)))
    ground_truth_trial.main(['--output','unused','--mode','bimanual','--process-zone-trial',*extra])
    return result


def test_grasp_and_thread_proposals_do_not_change_physics_guards(monkeypatch):
    args=invoke(monkeypatch,['--grasp-arc-m','.09','--physics-threads','1'])
    assert args[-4:]==['--grasp-arc-m','0.09','--physics-threads','1']
    for flag in ('--require-retention-screen','--physical-grasp-span','--force-closure',
                 '--native-startup-clearance','--native-static-clearance'):
        assert flag in args
    assert args[args.index('--physics-hz')+1]=='480'
    assert args[args.index('--uniform-solver-iterations')+1:args.index('--uniform-solver-iterations')+3]==['128','0']


@pytest.mark.parametrize('extra',[
    ['--grasp-arc-m','nan'],['--grasp-arc-m','.059'],['--grasp-arc-m','.181'],
    ['--grasp-arc-m','.09','--mode','right_only'],['--physics-threads','0']])
def test_invalid_comparisons_refused(monkeypatch,extra):
    with pytest.raises(SystemExit):invoke(monkeypatch,extra)


@pytest.mark.parametrize('arc',['.12','.14','.16','.18'])
def test_distal_hold_proposals_preserve_exact_anatomy_and_retention_gates(monkeypatch,arc):
    args=invoke(monkeypatch,['--grasp-arc-m',arc])
    assert args[-2:]==['--grasp-arc-m',str(float(arc))]
    for flag in ('--exact-grasp-arc','--require-retention-screen','--physical-grasp-span',
                 '--native-startup-clearance','--native-static-clearance','--force-closure'):
        assert flag in args
    assert '--native-retention-trial' not in args
    original=ground_truth_trial.arguments('unused','bimanual')
    assert original[original.index('--grasp-arc-m')+1]=='.08'
