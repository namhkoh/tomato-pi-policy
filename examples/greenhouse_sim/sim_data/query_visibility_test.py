import numpy as np
import pytest

from sim_data.query_visibility import QueryVisibility


def fixture():
    mask=np.zeros((408,848),bool); mask[100:108,100:170]=True
    rgb=np.full((408,848,3),150,np.uint8); rgb[mask]=(50,100,30)
    return rgb,mask


def test_clear_native_segment_passes_without_mutation():
    rgb,mask=fixture(); original_rgb=rgb.copy(); original_mask=mask.copy()
    result=QueryVisibility(rgb,mask).inspect([135.4,104.2])
    assert result['passed'] and result['island_pixels']==560
    assert result['local_pixels']==264 and result['interior_radius_px']==4
    assert np.array_equal(rgb,original_rgb) and np.array_equal(mask,original_mask)


@pytest.mark.parametrize('kind,reason',[
    ('two_pixels','minimum_island_pixels'),('49_pixels','minimum_island_pixels'),
    ('thin','minimum_interior_radius_px'),('edge','minimum_frame_margin_px'),
    ('dark','local_target_too_dark'),('contrast','insufficient_local_contrast'),
    ('background','query_not_native_target')])
def test_unusable_query_fails_closed(kind,reason):
    rgb,mask=fixture(); query=[135.,104.]
    if kind in ('two_pixels','49_pixels'):
        mask[:]=False
        if kind=='two_pixels': mask[104,135:137]=True
        else: mask[101:108,132:139]=True
    if kind=='thin': mask[:]=False; mask[104,100:240]=True
    if kind=='edge': mask[:]=False; mask[0:25,130:140]=True; query=[135.,4.]
    if kind=='dark': rgb[mask]=5
    if kind=='contrast': rgb[:]=100
    if kind=='background': query=[80.,80.]
    result=QueryVisibility(rgb,mask).inspect(query)
    assert not result['passed'] and reason in result['reasons']


def test_nearby_large_island_cannot_rescue_two_pixel_query():
    rgb,mask=fixture(); mask[104,172:174]=True
    result=QueryVisibility(rgb,mask).inspect([172.,104.])
    assert not result['passed'] and result['island_pixels']==2


@pytest.mark.parametrize('query',[[848,0],[0,408],[-.1,20],[float('nan'),100]])
def test_invalid_query_rejected(query):
    with pytest.raises(ValueError): QueryVisibility(*fixture()).inspect(query)
