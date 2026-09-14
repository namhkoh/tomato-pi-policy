import numpy as np
import pytest
from .postrelease_feed import proportional_backoff
from .through_stroke import ThroughStroke
from .cut_section_test import KNIFE,FACES,contact
from .support_aware_feed_test import record


@pytest.mark.parametrize('load,normal,expected',[(.39,0.,0.),(.4,0.,0.),(.401,0.,-.000005),
    (.42,0.,-.0001),(.5,0.,-.0005),(.32,.29,-.00005)])
def test_original_backoff_onset_and_maximum_reverse_retained(load,normal,expected):
    assert proportional_backoff(load,normal)==pytest.approx(expected)


@pytest.mark.parametrize('load,normal',[(.501,0.),(-.01,0.),(.2,.3),(True,0.),(.2,float('nan'))])
def test_invalid_native_loads_never_make_commands(load,normal):
    with pytest.raises(ValueError):proportional_backoff(load,normal)


def test_monotone_bounded_reverse_with_no_filtered_force_evidence():
    values=[proportional_backoff(u,0.) for u in np.linspace(.4,.5,100)]
    assert all(-.0005<=v<=0 for v in values)
    assert all(a>=b for a,b in zip(values,values[1:]))


def controller(proportional=True,**kwargs):
    options=dict(material_section=dict(radius=.003,tip_offset=.0005,half_span=.014,knife=KNIFE,faces=FACES),
        support_aware_feed=True,proportional_face_backoff=proportional)
    options.update(kwargs)
    return ThroughStroke([-.008,0.,.005],fraction=.3,endpoint=[0,0,-.005],
        direction=[0,0,-1],physics_hz=480,step=20,**options)


def sample(f,upper):
    r=record(20);r.update(contact(upper=upper))
    f.observe(r,step=20,arc_up=[0,0,1],support_ready=True,
        section_pose=dict(centre=[0.,0.,0.],axis=[1.,0.,0.]))


@pytest.mark.parametrize('enabled,expected',[(False,-.0005),(True,-.0001)])
def test_explicit_trial_changes_magnitude_only_with_same_native_observation(enabled,expected):
    f=controller(enabled);sample(f,.42);initial=f.offset;f.command(step=20,dt=1/480)
    assert f.receipt['command_state']=='backoff_load'
    assert f.receipt['command_speed_m_s']==pytest.approx(expected)
    assert f.offset-initial==pytest.approx(expected/480)
    assert not f.complete and not f.receipt['commanded_motion_used_as_completion']
    if enabled:assert f.receipt['hard_load_guard_n']==.5 and not f.receipt['force_limits_changed']


@pytest.mark.parametrize('kwargs',[dict(support_aware_feed=False),dict(material_section=None),dict(proportional_face_backoff=1)])
def test_incomplete_experiment_rejected(kwargs):
    with pytest.raises(ValueError):controller(**kwargs)


def test_hard_force_guard_unchanged_in_trial():
    with pytest.raises(RuntimeError):sample(controller(),.501)


def test_explicit_launcher_validation_and_default_off(monkeypatch,tmp_path):
    from pathlib import Path
    from .benchmark import main,parser
    from .ground_truth_trial import main as launch
    assert not parser().parse_args(['--output',str(tmp_path/'none')]).proportional_face_backoff_trial
    with pytest.raises(ValueError,match='Proportional face backoff'):
        main(['--output',str(tmp_path/'none'),'--proportional-face-backoff-trial'])
    class BeforeWrite(Exception):pass
    def stop(*a,**k):raise BeforeWrite()
    monkeypatch.setattr(Path,'mkdir',stop)
    with pytest.raises(BeforeWrite):launch(['--output',str(tmp_path/'none'),'--mode','bimanual',
        '--milestone','cut_action','--process-zone-trial','--through-stroke-trial','--material-clearance-trial',
        '--support-aware-feed-trial','--postrelease-feed-m-s','.001','--proportional-face-backoff-trial'])
