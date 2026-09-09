from copy import deepcopy
import numpy as np
import pytest

from sim_data.native_instances import FAST,LEGACY,normalize_payload


def payload():
    ids=np.full((408,848),4,np.uint32)
    return {FAST:dict(data=ids,info=dict(ids=np.array([4],np.uint32),labels=['/World/Stem'])),
            LEGACY:dict(data=ids.copy(),info=dict(idToLabels={'4':'/World/Stem'})), 'rgb':'untouched'}


def test_fast_adapter_preserves_native_identity_and_observation():
    data=payload(); result=normalize_payload(data,'fast')
    assert result['rgb']=='untouched' and FAST not in result
    assert result[LEGACY]['data'] is data[FAST]['data']
    assert result[LEGACY]['info']['idToLabels']=={4:'/World/Stem'}


def test_equivalence_compares_actual_pixels_and_paths():
    result=normalize_payload(payload(),'compare')
    assert result['native_instance_equivalence']['passed']
    assert result['native_instance_equivalence']['compared_pixels']==848*408


def test_unused_mapping_entries_are_not_repeated_in_each_frame():
    data=payload()
    data[FAST]['info']=dict(ids=np.array([4,100],np.uint32),labels=['/World/Stem','/World/NotVisible'])
    result=normalize_payload(data,'compare')
    assert result[LEGACY]['info']['idToLabels']=={4:'/World/Stem'}
    assert result['native_instance_equivalence']['passed']
    assert result['native_instance_mapping_scope']=='all_observed_renderer_IDs_only'


@pytest.mark.parametrize('kind',['pixels','paths','missing','duplicate','colorized','semantic'])
def test_fast_mismatch_fails_closed(kind):
    data=deepcopy(payload())
    if kind=='pixels': data[LEGACY]['data'][0,0]=0
    if kind=='paths': data[LEGACY]['info']['idToLabels']['4']='/World/Other'
    if kind=='missing': data[FAST]['info'].pop('ids')
    if kind=='duplicate': data[FAST]['info'].update(ids=np.array([4,4]),labels=['/World/A','/World/B'])
    if kind=='colorized': data[FAST]['data']=data[FAST]['data'].astype(np.uint8)
    if kind=='semantic': data[FAST]['info']['labels']=[{'class':'plant'}]
    with pytest.raises(ValueError): normalize_payload(data,'compare')
