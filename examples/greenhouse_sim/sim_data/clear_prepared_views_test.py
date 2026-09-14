"""Native-boundary reconstruction must preserve the complete source proposal."""
import pytest
from sim_data.training_plan import view_specs,prepared_view_specs


@pytest.mark.parametrize('lean',[False,True])
@pytest.mark.parametrize('orbit',[False,True])
@pytest.mark.parametrize('opposite',[False,True])
@pytest.mark.parametrize('offset',[0,12])
def test_prepared_native_flags_exactly_reconstruct_authorized_views(lean,orbit,opposite,offset):
    options=dict(vary_torso=True,clear_capture=True,oblique_clear=True,
                 lean_clear=lean,orbit_clear=orbit,opposite_aisle=opposite,view_offset=offset)
    plan=dict(target_world_m={'target':[0,0,1.3]},requested_views_per_target=3,**options)
    expected=view_specs(.8,0,'target',3,**options)
    assert list(prepared_view_specs(plan,'target',.8).values())==expected
    for flag in ('lean_clear','orbit_clear'):
        if options[flag]:
            changed=prepared_view_specs({**plan,flag:False},'target',.8)
            assert not any(changed.get(s['candidate_id'])==s for s in expected)


def test_original_prepared_view_defaults_are_unchanged():
    p=dict(target_world_m={'target':[0,0,1.3]},requested_views_per_target=3)
    assert list(prepared_view_specs(p,'target',.8).values())==view_specs(.8,0,'target',3)
