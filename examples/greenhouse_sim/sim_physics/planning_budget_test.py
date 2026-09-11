"""Longer diagnostic planning cannot bypass epoch, query or safety checks."""
import numpy as np
import pytest
from .benchmark import main,parser
from .native_static_clearance import NativeStaticClearance


def test_default_budget_and_model_remain_legacy():
    args=parser().parse_args(['--output','unused'])
    assert args.native_static_planning_seconds==8.
    assert args.cut_model=='force_qualified_pre_authored_seam_release'


@pytest.mark.parametrize('budget',[0.,-1.,60.001,float('nan'),float('inf'),True])
def test_native_rejects_unbounded_budget(budget):
    with pytest.raises(ValueError):NativeStaticClearance(lambda *a:True,[],wall_limit_s=budget)


def test_explicit_budget_still_expires_and_invalidates(monkeypatch):
    from . import native_static_clearance as module
    clock=[0.];monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    records=[('/World/Stem','box',None,np.full(3,-1.),np.full(3,1.))]
    native=NativeStaticClearance(lambda *a:True,records,wall_limit_s=30.)
    clock[0]=9.
    assert not native.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    assert native.active and native.report()['wall_limit_s']==30.
    clock[0]=30.
    with pytest.raises(RuntimeError,match='query_budget_exhausted'):
        native.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.ones(3),.001)
    assert not native.active and not native.validation_passed


def test_longer_budget_does_not_allow_epoch_change():
    revision=[False]
    def guard():
        if revision[0]:raise RuntimeError('physics advanced')
    native=NativeStaticClearance(lambda *a:True,[],wall_limit_s=60.,guard=guard)
    revision[0]=True
    with pytest.raises(RuntimeError,match='physics advanced'):native.validate()
    assert not native.validation_passed


@pytest.mark.parametrize('args,match',[
    (['--native-static-planning-seconds','30'],'planning budget'),
    (['--bimanual-cut','--native-static-clearance','--native-static-planning-seconds','61'],'planning budget'),
    (['--cut-model','signed_edge_load_brittle_seam_v1'],'brittle seam model')])
def test_cli_rejects_before_simulator_or_output_creation(tmp_path,args,match):
    output=tmp_path/'uncreated'
    with pytest.raises(ValueError,match=match):main(['--output',str(output),*args])
    assert not output.exists()
