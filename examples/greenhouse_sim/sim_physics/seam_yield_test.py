from types import SimpleNamespace as S
import numpy as np
import pytest
from .seam_yield import EdgeYield,NativeEdgeYield
from .knife import DOWNWARD_CUT_MODEL
from .blade_contacts import CROSSBAR_EDGE


def sample(advance=0.,*,centre_down=0.,qualified=True,cut=False):
    edge=np.eye(4)
    # +edge-X must be up, with a proper orientation.
    edge[:3,:3]=np.column_stack(([0,0,1],[0,1,0],[-1,0,0]))
    edge[2,3]=1-advance
    return dict(native_guards_passed=True,cut=cut,seam_world=[0,0,1-centre_down],
        native_contact_pairs_n=[],knife=dict(cut_model=DOWNWARD_CUT_MODEL,
            raw_normal_rows_complete=True,edge_frame=edge.tolist(),
            edge_signed_resistance_n=.24,tool_contact_upper_bound_n=.36,
            raw_normal_rows=[dict(eligible=True,impulse_on_knife=[0,0,.0005],separation=-.00025)],
            gate_diagnostic=dict(state='qualifying_contact' if qualified else 'rejected_contact',
                failed_conditions=[] if qualified else ['arc_up_and_world_down'],leading_normal_cosine=1.)))


def test_yield_is_measured_net_advance_not_contact_time_or_cut_authority():
    law=EdgeYield();assert law.observe(sample(),step=1)['state']=='yield_origin_measured'
    for step in range(2,20):assert law.observe(sample(),step=step)['stiffness_n_m']==1000.
    out=law.observe(sample(.0003),step=20)
    assert out['stiffness_n_m']==pytest.approx(1000*.00025/.00055)
    assert not out['cut_authorized'] and not out['native_material_readback']
    assert not out['calibrated'] and not out['geometry_changed']
    # Backoff is not healing; repeated positive jitter cannot ratchet damage.
    for step,advance in enumerate([.000299,.0003,.000299,.0003],21):
        assert law.observe(sample(advance),step=step)['stiffness_n_m']==out['stiffness_n_m']


def test_stationary_blade_or_blade_following_target_is_not_loaded_advance():
    law=EdgeYield();law.observe(sample(),step=1)
    assert law.observe(sample(centre_down=-.0003),step=2)['stiffness_n_m']==1000.
    assert law.observe(sample(.0003,centre_down=.0003),step=3)['stiffness_n_m']==1000.


def test_wrong_alignment_never_softens_and_after_release_stops_updates():
    law=EdgeYield()
    for i in range(1,10):assert law.observe(sample(.0001*i,qualified=False),step=i)['stiffness_n_m']==1000.
    assert law.origin is None
    law.observe(sample(),step=10)
    assert law.observe(sample(.0003,qualified=False),step=11)['stiffness_n_m']==1000.
    assert law.observe(sample(.001,cut=True),step=12)['stiffness_n_m']==1000.


@pytest.mark.parametrize('fault',['step','guard','model','rows','load','depth','angle'])
def test_invalid_evidence_fails_closed(fault):
    law=EdgeYield();row=sample();step=1
    if fault=='step':step=2
    elif fault=='guard':row['native_guards_passed']=False
    elif fault=='model':row['knife']['cut_model']='historical'
    elif fault=='rows':row['knife']['raw_normal_rows_complete']=False
    elif fault=='load':row['knife']['edge_signed_resistance_n']=.6
    elif fault=='angle':row['knife']['edge_frame']=np.eye(4).tolist()
    else:row['knife']['raw_normal_rows'][0]['separation']=.001
    with pytest.raises(RuntimeError):law.observe(row,step=step)
    assert law.stiffness==1000.


def adapter():
    from .seam_contact_compliance_test import fixture
    from .seam_contact_compliance import apply
    rig=fixture();apply(rig,diagnostic_only=True)
    f=S(knife=S(edge_mode=CROSSBAR_EDGE,collider='/Robot/CrossbarContact'))
    return rig,NativeEdgeYield(rig,f)


def test_only_session_material_is_updated_from_previous_sample():
    rig,a=adapter();before=rig.stage.GetRootLayer().ExportToString()
    a.apply(step=0);a.observe(sample(),step=1);a.apply(step=1)
    row=sample(.0003);a.observe(row,step=2)
    assert a.attr.Get()==1000.  # no material write in the post-fetch observer
    a.apply(step=2)
    assert a.attr.Get()==pytest.approx(1000*.00025/.00055,rel=1e-6)
    assert rig.stage.GetRootLayer().ExportToString()==before
    assert not rig.cut and not row['seam_yield']['cut_authorized']
    with pytest.raises(RuntimeError,match='Stale/reused'):a.apply(step=2)


def test_camera_cannot_use_softening_to_clear_stem():
    rig,a=adapter();row=sample()
    row['native_contact_pairs_n']=[['/Robot/Camera',rig.body_paths[0]+'/StemCollider',.05]]
    with pytest.raises(RuntimeError,match='Foreign contact'):a.observe(row,step=1)
    assert a.attr.Get()==1000.


def test_external_material_mutation_is_not_overwritten():
    from pxr import Usd
    rig,a=adapter()
    with Usd.EditContext(rig.stage,rig.stage.GetSessionLayer()):a.attr.Set(900.)
    with pytest.raises(RuntimeError,match='outside owned'):a.observe(sample(),step=1)
    assert a.attr.Get()==900.


def test_released_debris_does_not_continue_the_material_law():
    rig,a=adapter();row=sample(cut=True)
    row['native_contact_pairs_n']=[['/Robot/Torso',rig.body_paths[1]+'/StemCollider',.1]]
    a.observe(row,step=1);a.apply(step=1)
    assert a.updates==0 and a.attr.Get()==1000.
    assert row['seam_yield']['state']=='released_no_further_softening'


def test_experimental_yield_is_never_a_production_default(tmp_path):
    from .benchmark import parser,main
    assert not parser().parse_args(['--output','unused']).seam_contact_yield
    with pytest.raises(ValueError,match='Seam yielding'):
        main(['--output',str(tmp_path/'none'),'--seam-contact-yield'])
    assert not (tmp_path/'none').exists()
