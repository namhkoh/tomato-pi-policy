from copy import deepcopy
import numpy as np
import pytest
from .test_capture_storage import synthetic_callback
from .dual_storage import write_pair
from .capture_storage import write_compact_native_sample


@pytest.mark.parametrize('resolution',[(848,408),(1696,816)])
def test_independent_same_synthetic_callback_exact(tmp_path,resolution):
    callback=synthetic_callback(resolution)
    depth=callback['depth'].tobytes();mapping=deepcopy(callback['mapping'])
    value=write_pair(tmp_path/'raw',tmp_path/'compact',callback)
    assert value['native_npy_bytes_exact']==3 and len(value['exact_observations'])==7
    assert value['complete_prim_identity_table_equal'] and not value['training_approved']
    assert callback['depth'].tobytes()==depth and callback['mapping']==mapping
    assert (tmp_path/'raw/same_callback_storage.json').exists()


def test_no_trace_for_ineligible_callback(tmp_path):
    callback=synthetic_callback((848,408));callback['label']['eligible']=False;callback['trace']=None
    result=write_pair(tmp_path/'raw',tmp_path/'compact',callback)
    assert result['raw_result']['query_trace'] is None
    assert not (tmp_path/'raw/supervision/query_trace.json').exists()


@pytest.mark.parametrize('mode',['equal','nested','existing_raw','existing_compact'])
def test_destination_guard(tmp_path,mode):
    raw,compact=tmp_path/'raw',tmp_path/'compact'
    if mode=='equal':compact=raw
    if mode=='nested':compact=raw/'child'
    if mode=='existing_raw':raw.mkdir()
    if mode=='existing_compact':compact.mkdir()
    with pytest.raises(ValueError):write_pair(raw,compact,synthetic_callback((848,408)))
    assert not (raw/'sample.json').exists()


def test_private_buffer_mutation_detected(tmp_path):
    callback=synthetic_callback((848,408));original=callback['instances'].copy()
    def broken(directory,**private):
        private['instances'][100,100]=5
        return write_compact_native_sample(directory,**private)
    with pytest.raises(ValueError,match='Callback changed'):
        write_pair(tmp_path/'raw',tmp_path/'compact',callback,compact_writer=broken)
    assert np.array_equal(callback['instances'],original)
    assert not (tmp_path/'raw/same_callback_storage.json').exists()


def test_prim_table_mutation_detected(tmp_path):
    def broken(directory,**private):
        private['mapping']['5']='/World/Incorrect'
        return write_compact_native_sample(directory,**private)
    with pytest.raises(ValueError,match='Callback changed'):
        write_pair(tmp_path/'raw',tmp_path/'compact',synthetic_callback((848,408)),compact_writer=broken)
    assert not (tmp_path/'raw/same_callback_storage.json').exists()


def test_raw_format_cannot_impersonate_compact(tmp_path):
    import shutil
    from .bundle import digest
    def fake(directory,**private):
        shutil.copytree(tmp_path/'raw',directory)
        return dict(sample_sha256=digest((directory/'sample.json').read_bytes()),
            label_sha256=digest((directory/'supervision/label.json').read_bytes()),query_trace=private['trace'])
    with pytest.raises(ValueError,match='Compact bundle format required'):
        write_pair(tmp_path/'raw',tmp_path/'compact',synthetic_callback((848,408)),compact_writer=fake)
    assert not (tmp_path/'raw/same_callback_storage.json').exists()


def test_changed_raw_metadata_after_reader_open_rejected(tmp_path):
    def broken(directory,**private):
        value=write_compact_native_sample(directory,**private)
        with (tmp_path/'raw/sample.json').open('ab') as stream:stream.write(b' ')
        return value
    with pytest.raises(ValueError,match='Unexpected source binding'):
        write_pair(tmp_path/'raw',tmp_path/'compact',synthetic_callback((848,408)),compact_writer=broken)
    assert not (tmp_path/'raw/same_callback_storage.json').exists()
