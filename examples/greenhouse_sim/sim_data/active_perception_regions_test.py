import numpy as np
import pytest

from sim_data.active_perception_regions import leaf_region


def arrays():
    ids = np.zeros((408,848), np.uint32); ids[100:300,200:500] = 7
    catalogue = [dict(component_index=7, organ_type='leaf')]
    return ids, catalogue, np.ones(ids.shape,np.float32), np.ones(ids.shape,bool)


def test_leaf_covered_region_never_claims_hidden_target_or_absence():
    data = arrays(); result = leaf_region(*data)
    x,y,r,b = result['region_xyxy']
    assert r-x == b-y == 48 and (data[0][y:b,x:r] == 7).all()
    assert result['target_eligibility'] == 'unknown'
    assert result['hidden_target_existence'] == 'not_inferred'
    assert result['no_target_evidence'] is False and result['visual_review_required']


@pytest.mark.parametrize('organ', ['main_stem','fruit','unmapped'])
def test_non_leaf_regions_are_not_unknown_foliage_proposals(organ):
    data = arrays(); data[1][0]['organ_type'] = organ
    assert leaf_region(*data) is None


@pytest.mark.parametrize('issue', ['invalid','nan','zero','too_small','holes'])
def test_native_validity_and_whole_region_coverage_required(issue):
    ids,cat,z,valid=arrays()
    if issue=='invalid': valid[:]=False
    if issue=='nan': z[:]=np.nan
    if issue=='zero': z[:]=0
    if issue=='too_small': ids[:]=0; ids[200:220,200:220]=7
    if issue=='holes': ids[:,::20]=0
    assert leaf_region(ids,cat,z,valid) is None


@pytest.mark.parametrize('size', [0,31,33,97,100,True])
def test_invalid_region_size_rejected(size):
    with pytest.raises(ValueError,match='region size'): leaf_region(*arrays(),size=size)
