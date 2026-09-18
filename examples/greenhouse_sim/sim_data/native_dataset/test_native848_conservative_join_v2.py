"""Original-only identity contract fixtures, never native or review evidence."""
import copy
import pytest
from .native848_conservative_join_v2 import original_identity_context


def fixture():
    anchor=dict(split='train',source_family='seed17_full',conservative_view_cap_group='seed17_full/SubStem_42',
        source_row=dict(target_id='seed17_full/SubStem_42',variant_id='seed17_full'),
        generated_row=dict(target_id='seed17_full_cr_fixture/SubStem_42'))
    source=dict(target_id='seed17_full/SubStem_46',variant_id='seed17_full')
    rec=dict(split='train',source_family='seed17_full',target_id=source['target_id'],source_target=source['target_id'])
    return anchor,source,rec


def test_new_original_target_uses_its_own_source_pool_without_editing_anchor():
    anchor,source,rec=fixture();before=copy.deepcopy(anchor)
    context=original_identity_context(anchor,source,rec)
    assert context['source_row']==source and context['conservative_view_cap_group']=='seed17_full/SubStem_46'
    assert anchor==before and context is not anchor


def test_generated_variant_cannot_enter_original_only_route():
    anchor,source,rec=fixture();source['variant_id']='seed17_full_cr_fixture'
    with pytest.raises(ValueError):original_identity_context(anchor,source,rec)


def test_original_target_cannot_claim_another_pool():
    anchor,source,rec=fixture();rec['source_target']='seed17_full/SubStem_44'
    with pytest.raises(ValueError):original_identity_context(anchor,source,rec)


@pytest.mark.parametrize('change',[dict(split='test'),dict(source_family='seed19_full'),dict(target_id='seed17_full_cr_fixture/SubStem_46')])
def test_heldout_family_or_generated_target_alias_rejected(change):
    anchor,source,rec=fixture();rec.update(change)
    with pytest.raises(ValueError):original_identity_context(anchor,source,rec)
