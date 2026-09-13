from types import SimpleNamespace as S
import numpy as np
import pytest
from .planning_heartbeat import PlanningHeartbeat
from .native_static_clearance import NativeStaticClearance
from .native_static_clearance_test import scene_runtime,records


def test_render_only_fixed_wall_rate_no_physics_request():
    clock=[0.];calls=[]
    h=PlanningHeartbeat(lambda:calls.append(clock[0]),lambda:True,clock=lambda:clock[0])
    h();h();clock[0]=.06;h();clock[0]=.07;h()
    assert calls==[0.,.07] and h.report()['physics_steps_requested']==0
    assert not h.report()['planning_budget_extended']


@pytest.mark.parametrize('during',[False,True])
def test_stop_is_checked_before_and_after_ui_event_delivery(during):
    healthy=[during]
    h=PlanningHeartbeat(lambda:healthy.__setitem__(0,False),lambda:healthy[0])
    with pytest.raises(RuntimeError,match='cancelled'):h()
    assert h.renders==int(during) and not h.rendering


def test_reentrant_ui_callback_cannot_start_nested_planning_render():
    h=PlanningHeartbeat(lambda:h(),lambda:True)
    with pytest.raises(RuntimeError,match='Reentrant'):h()
    assert not h.rendering


def test_epoch_change_during_render_invalidates_query_before_native_call():
    changed=[False];calls=[]
    def guard():
        if changed[0]:raise RuntimeError('physics or USD changed')
    n=NativeStaticClearance(lambda *a:calls.append(a) or False,[],guard=guard,
        heartbeat=lambda:changed.__setitem__(0,True))
    assert n._overlap('/S',np.zeros(3),np.eye(3),np.ones(3)) is None
    assert not calls and not n.active and n.errors


def test_cached_clearance_is_not_returned_after_ui_edit():
    n=NativeStaticClearance(lambda *a:False,[],memoize_queries=True)
    args=('/S',np.zeros(3),np.eye(3),np.ones(3))
    assert n._overlap(*args,memoize=True) is False and n.query_cache
    changed=[False]
    n.heartbeat=lambda:changed.__setitem__(0,True)
    def guard():
        if changed[0]:raise RuntimeError('scene changed')
    n.guard=guard
    assert n._overlap(*args,memoize=True) is None
    assert not n.query_cache and not n.validation_passed


def test_render_time_does_not_extend_planning_budget(monkeypatch):
    from . import native_static_clearance as module
    clock=[0.];monkeypatch.setattr(module.time,'perf_counter',lambda:clock[0])
    n=NativeStaticClearance(lambda *a:False,[],wall_limit_s=1,
        heartbeat=lambda:clock.__setitem__(0,2.))
    assert n._overlap('/S',np.zeros(3),np.eye(3),np.ones(3)) is None
    assert n.calls==0 and any('budget' in e for e in n.errors)


def test_closed_query_never_pumps_ui():
    calls=[];n=NativeStaticClearance(lambda *a:False,[],heartbeat=lambda:calls.append(1))
    n.close()
    assert n._overlap('/S',np.zeros(3),np.eye(3),np.ones(3)) is None and not calls


def test_watch_auto_run_requires_explicit_watch_before_any_launch(tmp_path):
    from .ground_truth_trial import main
    with pytest.raises(ValueError,match='explicit watch'):
        main(['--output',str(tmp_path/'unused'),'--mode','bimanual',
            '--process-zone-trial','--milestone','cut_action','--watch-auto-run'])
    assert not (tmp_path/'unused').exists()


def test_coupled_finger_watch_keeps_full_physical_validation(tmp_path,monkeypatch):
    from pathlib import Path
    from .ground_truth_trial import main
    class ReachedOutput(Exception):pass
    monkeypatch.setattr(Path,'mkdir',lambda *a,**k:(_ for _ in ()).throw(ReachedOutput()))
    with pytest.raises(ReachedOutput):
        main(['--output',str(tmp_path/'unused'),'--mode','bimanual',
            '--process-zone-trial','--milestone','cut_action',
            '--coupled-fingers-trial','--watch','--watch-auto-run'])


def timeline_fixture(monkeypatch,state):
    import omni.timeline
    timeline=omni.timeline.get_timeline_interface();state.auto=True;state.auto_calls=[]
    timeline.is_auto_updating=lambda:state.auto
    def setting(value):state.auto_calls.append(value);state.auto=value
    timeline.set_auto_update=setting;timeline.commit=lambda:None
    monkeypatch.setattr(omni.timeline,'get_timeline_interface',lambda:timeline)
    return timeline


def test_owned_freeze_prevents_render_time_advance_then_restores(scene_runtime,monkeypatch):
    from .native_static_clearance import current_scene_query
    state=scene_runtime;timeline_fixture(monkeypatch,state);renders=[]
    def render():
        renders.append(1)
        if state.auto:state.time+=1/60
    n=current_scene_query(state.stage,records(),heartbeat=render);state.owned.append(n)
    assert not state.auto
    assert n.clear_box_checked('/World/Stem',np.zeros(3),np.eye(3),np.full(3,.002),.001)
    n.validate();n.close()
    assert renders and state.time==3.5 and state.auto_calls==[False,True]
    assert n.validation_passed and state.notice_active==0


def test_render_that_reenables_clock_cannot_return_clearance(scene_runtime,monkeypatch):
    from .native_static_clearance import current_scene_query
    state=scene_runtime;timeline_fixture(monkeypatch,state)
    with pytest.raises(RuntimeError):
        current_scene_query(state.stage,records(),heartbeat=lambda:setattr(state,'auto',True))
    assert state.auto and state.notice_active==0
    assert not state.object_callbacks and state.step_callback is None


def test_commit_cannot_hide_pending_user_scrub_in_new_epoch(scene_runtime,monkeypatch):
    from .native_static_clearance import current_scene_query
    state=scene_runtime;timeline=timeline_fixture(monkeypatch,state)
    timeline.commit=lambda:setattr(state,'time',state.time+.01)
    with pytest.raises(RuntimeError,match='timeline_time_changed'):
        current_scene_query(state.stage,records(),heartbeat=lambda:None)
    assert state.auto_calls==[False,True] and state.notice_active==0
    assert not state.object_callbacks and state.step_callback is None
