import pytest
from .knife_load import physical_load
from .contact_events_test import native_stub
from .signed_blade_test import fixture,step


def test_noncutting_load_and_friction_never_disappear():
    p={('/World/R/knife/arc','/World/Plant/stem'):.0015,
       ('/World/Plant/stem','/World/R/knife/edge'):.0008,
       ('/World/R/knife_neighbor/edge','/World/Plant/stem'):100.}
    rows=[dict(collider0='/World/Plant/stem',collider1='/World/R/knife/arc',separation=-.0007,eligible=False)]
    r=physical_load(p,rows,root='/World/R/knife',dt=.005)
    assert r['upper_bound_n']==pytest.approx(.46) and r['minimum_separation_m']==-.0007
    assert not r['cut_evidence']


@pytest.mark.parametrize('value',[float('nan'),float('inf'),-.1,True])
def test_bad_load_refused(value):
    with pytest.raises(RuntimeError):physical_load({('/World/R/knife/edge','/World/Plant/stem'):value},[],root='/World/R/knife',dt=.005)


def test_rejected_real_contact_causes_feed_backoff_not_free_advance():
    from .through_stroke_test import fixture,sample,observe,command
    f=fixture();r=sample(upper=.43,normal=0.);observe(f,r);old=f.offset;command(f)
    assert f.offset<old and f.receipt['command_state']=='backoff_load'


def test_actual_cut_pipeline_counts_wrong_face_normal_and_friction(native_stub,monkeypatch):
    r=fixture(monkeypatch)
    result=step(r,0,forces=(.25,),offset=(0,.10,0),friction=.15)
    assert not r.releases and result['edge_contact_count']==0
    assert result['tool_contact_upper_bound_n']==pytest.approx(.40)
    assert result['physical_knife_contact']['upper_bound_n']==pytest.approx(.40)


def test_actual_pipeline_cannot_split_load_budget_between_good_and_bad_faces(native_stub,monkeypatch):
    r=fixture(monkeypatch)
    with pytest.raises(RuntimeError,match='0.5 N'):
        step(r,0,forces=(.4,),offset=(0,.10,0),friction=.11)
    assert not r.releases
