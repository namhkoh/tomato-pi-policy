from copy import deepcopy
import inspect
import numpy as np
import pytest
from .through_stroke import ThroughStroke
from .knife import KNIFE_IMPULSE_CONTRACT


def fixture():
    return ThroughStroke([-.008,0,.005],fraction=.3,endpoint=[0,0,-.005],
        direction=[0,0,-1],physics_hz=480,step=20)


def sample(z=.0041,upper=0.,normal=0.):
    edge=np.eye(4);edge[:3,:3]=[[0,0,-1],[0,1,0],[1,0,0]];edge[2,3]=z
    return dict(native_guards_passed=True,cut=True,knife=dict(edge_frame=edge.tolist(),
        force_contract=KNIFE_IMPULSE_CONTRACT,raw_normal_rows_complete=True,
        tool_contact_upper_bound_n=upper,edge_signed_resistance_n=normal))


def observe(f,r=None,**kw):
    return f.observe(sample() if r is None else r,step=f.step+1,arc_up=[0,0,1],support_ready=True,**kw)


def command(f):return f.command(step=f.step,dt=1/480)


def test_release_and_commanded_fraction_cannot_complete_stroke():
    f=fixture();observe(f);f.offset=f.end
    for _ in range(30):command(f);observe(f)
    assert not f.complete


def test_measured_endpoint_requires_unloaded_dwell_and_actual_travel():
    f=fixture();observe(f)
    for i in range(12):
        command(f);r=observe(f,sample(z=-.005))
        assert r['complete'] is (i==11)
    assert r['net_world_down_m']==pytest.approx(.0091)


@pytest.mark.parametrize('upper,normal,state',[ (.42,.25,'backoff_load'),(.32,.29,'backoff_load'),
    (.12,0.,'hold_nonleading_contact'),(.2,.2,'forward')])
def test_force_feedback_does_not_raise_load_limit(upper,normal,state):
    f=fixture();observe(f,sample(upper=upper,normal=normal));old=f.offset;command(f)
    assert f.receipt['command_state']==state
    assert abs(f.offset-old)<=.0005/480+1e-12


@pytest.mark.parametrize('change',[{'cut':False},{'native_guards_passed':False}])
def test_bad_release_or_guards_refuse(change):
    f=fixture();r=sample();r.update(change)
    with pytest.raises(RuntimeError):observe(f,r)


@pytest.mark.parametrize('upper,normal',[(.501,.2),(.2,.3),(float('nan'),.1),(.1,True)])
def test_unsafe_or_invalid_loads_refuse(upper,normal):
    with pytest.raises(RuntimeError):observe(fixture(),sample(upper=upper,normal=normal))


def test_actual_arc_orientation_not_nominal_mount():
    f=fixture()
    with pytest.raises(RuntimeError):f.observe(sample(),step=20,arc_up=[0,0,-1],support_ready=True)


def test_stale_reused_or_skipped_samples_refuse():
    f=fixture();observe(f)
    with pytest.raises(RuntimeError):observe(f)
    command(f)
    with pytest.raises(RuntimeError):command(f)
    with pytest.raises(RuntimeError):f.observe(sample(),step=23,arc_up=[0,0,1],support_ready=True)


def test_missing_support_holds_and_cannot_complete():
    f=fixture();f.observe(sample(),step=20,arc_up=[0,0,1],support_ready=False)
    old=f.offset;command(f)
    assert f.offset==old and f.receipt['command_state']=='hold_support'


def test_lateral_escape_and_overshoot_refuse():
    for axis,value in ((0,.0031),(2,-.0056)):
        f=fixture();r=sample();r['knife']['edge_frame'][axis][3]=value
        with pytest.raises(RuntimeError):observe(f,r)


def test_no_native_writes_or_release_in_core():
    source=inspect.getsource(ThroughStroke)
    assert 'set_dof' not in source and 'CollisionAPI' not in source and 'release_from_blade' not in source


def test_runtime_preserves_failure_sample_and_reverses_last_sent_command():
    from .bimanual_probe import run
    src=inspect.getsource(run)
    assert src.index("finally:\n        if getattr(args,'stream_trajectory',False):records.close()")<src.index('retained=[r for r in records')
    assert 'through_fraction=released_stroke_fraction(last_right_command)' in src
    assert 'through.receipt is None' in src
    assert src.index('fixture.inspect_cut(')<src.index('through.observe(')


def test_incomplete_profile_refused_before_launch(tmp_path):
    from .benchmark import main
    with pytest.raises(ValueError,match='Through-stroke'):
        main(['--output',str(tmp_path/'unused'),'--through-stroke-trial'])
    assert not (tmp_path/'unused').exists()


@pytest.mark.parametrize('mode',['bimanual','right_only'])
def test_public_full_recipe_reaches_validation_without_launch(tmp_path,monkeypatch,mode):
    from pathlib import Path
    from .ground_truth_trial import main
    class Validated(Exception):pass
    def stop(self,*a,**k):raise Validated()
    monkeypatch.setattr(Path,'mkdir',stop)
    with pytest.raises(Validated):main(['--output',str(tmp_path/'trial'),'--mode',mode,
        '--process-zone-trial','--milestone','cut_action','--through-stroke-trial'])


def test_exit_status_cannot_inherit_limited_cut_pass():
    from types import SimpleNamespace
    from .benchmark import report_exit_code
    assert report_exit_code(dict(state='passed_cut_action_not_complete_robot_task'),
        SimpleNamespace(through_stroke_trial=True))==2
