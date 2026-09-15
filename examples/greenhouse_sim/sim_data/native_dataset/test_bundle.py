"""Storage contract tests use synthetic arrays, not native qualification claims."""
import io,json,hashlib
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
from .bundle import SampleReader,pack_sample,safe_path,digest

def write(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)

@pytest.fixture
def sample(tmp_path):
    root=tmp_path/'raw';root.mkdir()
    arrays={
      'inputs/depth_m.npy':np.array([[0.,-0.,np.inf],[np.nan,1.25,2.5]],dtype=np.float32),
      'supervision/renderer_instance_id.npy':np.arange(6,dtype=np.uint32).reshape(2,3),
      'supervision/component_id.npy':np.arange(6,dtype=np.uint32).reshape(2,3)}
    for name,a in arrays.items():
        stream=io.BytesIO();np.save(stream,a,allow_pickle=False);write(root/name,stream.getvalue())
    for name,a in {
      'inputs/rgb.png':np.arange(18,dtype=np.uint8).reshape(2,3,3),
      'inputs/depth_valid.png':np.array([[0,0,0],[0,255,255]],dtype=np.uint8),
      'supervision/target_visible.png':np.zeros((2,3),dtype=np.uint8),
      'supervision/organ_type.png':np.zeros((2,3),dtype=np.uint8),
      'review/overlay.png':np.ones((2,3,3),dtype=np.uint8),
      'review/visible_target.png':np.full((2,3,3),255,dtype=np.uint8)}.items():
        stream=io.BytesIO();Image.fromarray(a).save(stream,format='PNG');write(root/name,stream.getvalue())
    write(root/'supervision/identities.json',b'{"component_catalogue":[]}')
    files={p.relative_to(root).as_posix():dict(sha256=digest(p.read_bytes()),
        role='review_only' if p.parent.name=='review' else
             'observation' if p.parent.name=='inputs' else 'ground_truth_supervision')
        for p in root.rglob('*') if p.is_file()}
    write(root/'sample.json',json.dumps(dict(files=files,training_sample_approved=False)).encode())
    write(root/'supervision/label.json',b'{"eligible":true,"training_approved":false}')
    write(root/'supervision/query_trace.json',b'{"passed":true}')
    return root

def pack(sample,tmp_path):
    out=tmp_path/'compact'
    manifest=pack_sample(sample,out,sample_sha256=digest((sample/'sample.json').read_bytes()),
        label_sha256=digest((sample/'supervision/label.json').read_bytes()),query_trace={'passed':True})
    return out,manifest

def test_exact_all_bytes_and_native_array_bits(sample,tmp_path):
    before={p.relative_to(sample).as_posix():p.read_bytes() for p in sample.rglob('*') if p.is_file()}
    out,m=pack(sample,tmp_path);reader=SampleReader(out)
    assert reader.verify_all() and reader.metadata==json.loads(before['sample.json'])
    for name,raw in before.items():
        if name.startswith('review/'):
            with pytest.raises(ValueError):reader.read(name)
        else:assert reader.read(name)==raw
    assert reader.array('inputs/depth_m.npy').tobytes()==np.load(io.BytesIO(before['inputs/depth_m.npy']),allow_pickle=False).tobytes()
    assert np.array_equal(reader.image('inputs/rgb.png'),SampleReader(sample).image('inputs/rgb.png'))
    assert before=={p.relative_to(sample).as_posix():p.read_bytes() for p in sample.rglob('*') if p.is_file()}
    assert m['depth_recomputed'] is False and m['training_approved'] is False and m['source_removed'] is False
    assert len(m['omitted_review_files'])==2

@pytest.mark.parametrize('name',['../secret','a/../../b','/absolute','C:/secret','a\\b','.','a//b','a/./b','a/'])
def test_paths_rejected(tmp_path,name):
    with pytest.raises(ValueError):safe_path(tmp_path,name)

def test_create_only(sample,tmp_path):
    out,_=pack(sample,tmp_path)
    with pytest.raises(ValueError,match='Create-only'):
        pack_sample(sample,out,sample_sha256='a'*64,label_sha256='b'*64)

@pytest.mark.parametrize('field',['sample_sha256','label_sha256'])
def test_source_receipt_required(sample,tmp_path,field):
    args=dict(sample_sha256=digest((sample/'sample.json').read_bytes()),label_sha256=digest((sample/'supervision/label.json').read_bytes()))
    args[field]='0'*64
    with pytest.raises(ValueError,match='Unexpected source'):pack_sample(sample,tmp_path/'bad',**args)
    assert not (tmp_path/'bad').exists()

def test_changed_raw_file_rejected(sample,tmp_path):
    (sample/'inputs/rgb.png').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='Changed raw'):
        pack(sample,tmp_path)

@pytest.mark.parametrize('name',['inputs/depth_m.npy','inputs/rgb.png','supervision/component_id.npy','sample.json'])
def test_stored_corruption_rejected(sample,tmp_path,name):
    out,m=pack(sample,tmp_path);p=out/m['files'][name]['stored_path']
    raw=p.read_bytes();p.write_bytes(raw[:-1]+bytes([raw[-1]^1]))
    with pytest.raises(ValueError,match='Corrupt'):
        SampleReader(out).verify_all()

@pytest.mark.parametrize('field',['training_approved','depth_recomputed','lossy'])
def test_forbidden_claims(sample,tmp_path,field):
    out,m=pack(sample,tmp_path);m[field]=True;(out/'bundle.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='Unsupported compact'):SampleReader(out)

def test_repacking_not_new_data(sample,tmp_path):
    out,m=pack(sample,tmp_path)
    with pytest.raises(ValueError,match='Repacking'):
        pack_sample(out,tmp_path/'again',sample_sha256=m['source_sample_sha256'],label_sha256=m['source_label_sha256'])

@pytest.mark.parametrize('mutation',['omitted','aliased','encoding','extra'])
def test_manifest_forgery_rejected(sample,tmp_path,mutation):
    out,m=pack(sample,tmp_path)
    if mutation=='omitted':m['omitted_review_files']['review/overlay.png']='0'*64
    if mutation=='aliased':m['files']['inputs/rgb.png']['stored_path']=m['files']['inputs/depth_valid.png']['stored_path']
    if mutation=='encoding':m['files']['inputs/depth_m.npy']['encoding']='identity'
    if mutation=='extra':m['files']['unlisted.png']=dict(m['files']['inputs/rgb.png'],stored_path='payload/unlisted.png')
    (out/'bundle.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):SampleReader(out)

def test_missing_required_array(sample,tmp_path):
    meta=json.loads((sample/'sample.json').read_text());del meta['files']['inputs/depth_m.npy']
    (sample/'sample.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='Complete native'):pack(sample,tmp_path)

def test_non_nested_destinations(sample,tmp_path):
    for dest in [sample,sample/'child',sample.parent]:
        with pytest.raises(ValueError,match='Separate nonnested'):
            pack_sample(sample,dest,sample_sha256='x',label_sha256='x')

@pytest.mark.parametrize('missing',['inputs/rgb.png','inputs/depth_valid.png','supervision/identities.json'])
def test_required_observations_not_optional(sample,tmp_path,missing):
    meta=json.loads((sample/'sample.json').read_text());del meta['files'][missing]
    (sample/'sample.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='Complete native'):pack(sample,tmp_path)

def test_raw_extras_need_binding(sample):
    reader=SampleReader(sample)
    for name in ['supervision/label.json','supervision/query_trace.json']:
        with pytest.raises(ValueError,match='Unverified raw extra'):reader.json(name)
    with pytest.raises(ValueError,match='Unverified raw extra'):reader.verify_all()

def test_query_trace_tamper_before_pack(sample,tmp_path):
    (sample/'supervision/query_trace.json').write_text('{"passed":false}')
    with pytest.raises(ValueError,match='Unexpected source JSON'):pack(sample,tmp_path)
    assert not (tmp_path/'compact').exists()

def test_eligible_trace_cannot_be_omitted(sample,tmp_path):
    out,m=pack(sample,tmp_path);del m['files']['supervision/query_trace.json']
    (out/'bundle.json').write_text(json.dumps(m))
    with pytest.raises(ValueError,match='omitted or unavailable'):SampleReader(out)

def test_eligible_trace_is_required_from_caller(sample,tmp_path):
    with pytest.raises(ValueError,match='Unverified raw extra'):
        pack_sample(sample,tmp_path/'bad',sample_sha256=digest((sample/'sample.json').read_bytes()),
            label_sha256=digest((sample/'supervision/label.json').read_bytes()))

def test_bounded_read_uses_one_handle(monkeypatch):
    from . import bundle
    class Stream:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self,n):
            assert n==bundle.MAX_FILE_BYTES+1
            return b'valid'
    class PathStub:
        def open(self,mode):
            assert mode=='rb';return Stream()
    assert bundle.read_bounded(PathStub())==b'valid'

@pytest.mark.parametrize('name',['sample.json','supervision/label.json'])
def test_compact_authoritative_hash_cannot_be_ignored(sample,tmp_path,name):
    out,_=pack(sample,tmp_path)
    with pytest.raises(ValueError,match='Unexpected source binding'):
        SampleReader(out,expected_bindings={name:'0'*64}).verify_all()

def test_compact_authoritative_trace_cannot_be_ignored(sample,tmp_path):
    out,_=pack(sample,tmp_path)
    with pytest.raises(ValueError,match='Unexpected source JSON'):
        SampleReader(out,expected_json={'supervision/query_trace.json':{'passed':False}}).verify_all()
