"""Synthetic storage tests only; fixtures are not native qualification."""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from . import bundle as v1
from . import bundle_v2 as v2
from . import json_bytes_codec as codec
from .test_bundle import sample as raw_sample


@pytest.fixture
def source(raw_sample,tmp_path):
    out=tmp_path/'v1'
    meta=json.loads((raw_sample/'sample.json').read_bytes())
    # Explicit NaN payload, signed zero, infinities and unusual JSON spelling.
    bits=np.array([[0,0x80000000,0x7fc01234],[0xff800000,0x7f800000,0x3f800000]],np.uint32)
    stream=io.BytesIO();np.save(stream,bits.view(np.float32),allow_pickle=False)
    (raw_sample/'inputs/depth_m.npy').write_bytes(stream.getvalue())
    meta['files']['inputs/depth_m.npy']['sha256']=v1.digest(stream.getvalue())
    (raw_sample/'sample.json').write_bytes(json.dumps(meta,indent=3).encode()+b'\r\n')
    (raw_sample/'supervision/identities.json').write_bytes(b' {"component_catalogue": [],"precision":1.00000000000000000001e-9}\r\n')
    meta['files']['supervision/identities.json']['sha256']=v1.digest((raw_sample/'supervision/identities.json').read_bytes())
    (raw_sample/'sample.json').write_bytes(json.dumps(meta,indent=3).encode()+b'\r\n')
    m=v1.pack_sample(raw_sample,out,sample_sha256=v1.digest((raw_sample/'sample.json').read_bytes()),
        label_sha256=v1.digest((raw_sample/'supervision/label.json').read_bytes()),query_trace={'passed':True})
    return out,m


def args(source):
    root,m=source
    return dict(source_bundle_sha256=v1.digest((root/'bundle.json').read_bytes()),
        sample_sha256=m['source_sample_sha256'],label_sha256=m['source_label_sha256'],query_trace=m['source_query_trace'])


def convert(source,tmp_path):
    out=tmp_path/'v2'
    report=v2.pack_sample(source[0],out,**args(source))
    return out,report


def snapshot(root):
    return {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}


def edit_manifest(out,edit):
    p=out/'bundle.json';m=json.loads(p.read_bytes());edit(m);p.write_text(json.dumps(m),encoding='utf-8')


def test_all_bytes_npy_bits_pngs_and_source_manifest_preserved(source,tmp_path):
    before=snapshot(source[0]);out,report=convert(source,tmp_path)
    r=v2.SampleReader(out,expected_bindings={'bundle.json':report['bundle_sha256'],
        v2.SOURCE:args(source)['source_bundle_sha256']})
    old=v1.SampleReader(source[0])
    assert r.verify_all() and r.metadata==old.metadata
    assert r.read(v2.SOURCE)==before['bundle.json']
    assert r.read('bundle.json')==(out/'bundle.json').read_bytes()
    for name,entry in old.manifest['files'].items():
        assert old.read(name)==r.read(name)
        if not name.endswith('.json'):
            assert (source[0]/entry['stored_path']).read_bytes()==(out/r.manifest['files'][name]['stored_path']).read_bytes()
    assert r.array('inputs/depth_m.npy').view(np.uint32).tolist()==[[0,0x80000000,0x7fc01234],[0xff800000,0x7f800000,0x3f800000]]
    assert np.array_equal(r.image('inputs/rgb.png'),old.image('inputs/rgb.png'))
    assert report['added_copy_bytes']==sum(len(b) for b in snapshot(out).values())
    assert not report['training_approved'] and not report['source_removed'] and report['raw_bytes_deleted']==0
    assert snapshot(source[0])==before and len(r.manifest['omitted_review_files'])==2
    assert 'source_query_trace' not in r.manifest
    assert r.manifest['source_query_trace_sha256']==v1.digest(old.read('supervision/query_trace.json'))
    with pytest.raises(ValueError): v1.SampleReader(out)
    with pytest.raises(ValueError): v2.SampleReader(source[0])


def test_reader_public_copies_cannot_relax_validation(source,tmp_path):
    out,_=convert(source,tmp_path);r=v2.SampleReader(out)
    r.manifest['files']['inputs/rgb.png']['stored_sha256']='0'*64
    r.metadata['files'].clear()
    assert r.verify_all()
    edit_manifest(out,lambda m:m.update(training_approved=True))
    with pytest.raises(ValueError,match='Manifest changed'):r.read('inputs/rgb.png')


@pytest.mark.parametrize('name',['sample.json','supervision/label.json','supervision/identities.json',
    'supervision/query_trace.json','inputs/rgb.png','inputs/depth_m.npy'])
def test_payload_tampering(source,tmp_path,name):
    out,_=convert(source,tmp_path);r=v2.SampleReader(out)
    p=out/r.manifest['files'][name]['stored_path'];raw=p.read_bytes();p.write_bytes(raw+b'X')
    with pytest.raises(ValueError):r.verify_all()


@pytest.mark.parametrize('field',['source_bundle_sha256','sample_sha256','label_sha256','query_trace'])
def test_input_pins_before_any_output(source,tmp_path,field):
    supplied=args(source);supplied[field]={'passed':False} if field=='query_trace' else '0'*64
    out=tmp_path/'bad'
    with pytest.raises(ValueError):v2.pack_sample(source[0],out,**supplied)
    assert not out.exists()


@pytest.mark.parametrize('name',['bundle.json',v2.SOURCE,'sample.json','supervision/label.json','inputs/rgb.png'])
def test_explicit_read_pins_cannot_be_ignored(source,tmp_path,name):
    out,_=convert(source,tmp_path)
    with pytest.raises(ValueError):v2.SampleReader(out,expected_bindings={name:'0'*64}).verify_all()


def test_expected_json_and_unknown_bindings_fail_closed(source,tmp_path):
    out,_=convert(source,tmp_path)
    with pytest.raises(ValueError):v2.SampleReader(out,expected_json={'supervision/query_trace.json':{'passed':False}})
    with pytest.raises(ValueError):v2.SampleReader(out,expected_bindings={'unread.json':'0'*64})


@pytest.mark.parametrize('name',['../secret','/absolute','C:/x','a\\b','a//b','a/./b','a/../b','a/',
    'NUL','CON.json','COM1.txt','a.','a /b','é.json','a:b','.','..'])
def test_path_alias_spellings(tmp_path,name):
    with pytest.raises(ValueError):v2.safe_path(tmp_path,name)


@pytest.mark.parametrize('fault',['schema','approved','lossy','depth','removed','extra','missing','aliased','case_alias',
    'encoding','size','logical','trace','omission','totals','source_schema'])
def test_manifest_forgery(source,tmp_path,fault):
    out,_=convert(source,tmp_path)
    def change(m):
        if fault=='schema':m['schema']=v1.SCHEMA
        if fault in ('approved','lossy','depth','removed'):m[dict(approved='training_approved',lossy='lossy',depth='depth_recomputed',removed='source_removed')[fault]]=True
        if fault=='extra':m['extra']='field'
        if fault=='missing':del m['files']['inputs/rgb.png']
        if fault=='aliased':m['files']['inputs/rgb.png']['stored_path']=m['files']['inputs/depth_valid.png']['stored_path']
        if fault=='case_alias':m['files']['inputs/rgb.png']['stored_path']=m['files']['inputs/depth_valid.png']['stored_path'].upper()
        if fault=='encoding':m['files']['sample.json']['encoding']='identity'
        if fault=='size':m['files']['sample.json']['logical_bytes']=codec.MAX_BYTES+1
        if fault=='logical':m['files']['sample.json']['logical_sha256']='0'*64
        if fault=='trace':m['source_query_trace_sha256']='0'*64
        if fault=='omission':m['omitted_review_files']['review/missing.png']='0'*64
        if fault=='totals':m['stored_payload_bytes']+=1
        if fault=='source_schema':m['source_schema']=v2.SCHEMA
    edit_manifest(out,change)
    with pytest.raises((ValueError,FileNotFoundError)):v2.SampleReader(out).verify_all()


def test_create_only_raw_and_v2_repack_held(source,raw_sample,tmp_path):
    out,_=convert(source,tmp_path)
    for dest in (out,source[0],source[0]/'child',source[0].parent):
        with pytest.raises(ValueError):v2.pack_sample(source[0],dest,**args(source))
    with pytest.raises(ValueError,match='Raw input'):v2.pack_sample(raw_sample,tmp_path/'raw_bad',**args(source))
    altered=args(source);altered['source_bundle_sha256']=v1.digest((out/'bundle.json').read_bytes())
    with pytest.raises(ValueError,match='V1 compact'):v2.pack_sample(out,tmp_path/'again',**altered)


def test_extra_physical_file_and_hardlink_alias_rejected(source,tmp_path):
    out,_=convert(source,tmp_path);r=v2.SampleReader(out)
    p=out/'unexpected.txt';p.write_bytes(b'x')
    with pytest.raises(ValueError,match='inventory'):r.verify_all()
    p.unlink()  # Synthetic test-owned extra only.
    rgb=out/r.manifest['files']['inputs/rgb.png']['stored_path']
    linked=tmp_path/'linked.png';os.link(rgb,linked)
    with pytest.raises(ValueError,match='Hard-linked'):r.read('inputs/rgb.png')


def test_symlinked_root_and_payload_rejected(source,tmp_path):
    out,_=convert(source,tmp_path);link=tmp_path/'alias'
    try:link.symlink_to(out,target_is_directory=True)
    except OSError:pytest.skip('OS does not permit symlink creation')
    with pytest.raises(ValueError,match='Linked root'):v2.SampleReader(link)


def test_source_change_during_copy_leaves_no_successful_marker(source,tmp_path,monkeypatch):
    original=v2.codec.encode;calls=[]
    def altered(raw,level=3):
        result=original(raw,level);calls.append(1)
        if len(calls)==1:
            p=source[0]/'bundle.json';p.write_bytes(p.read_bytes()+b' ')
        return result
    monkeypatch.setattr(v2.codec,'encode',altered)
    out=tmp_path/'partial'
    with pytest.raises(ValueError,match='Source changed'):v2.pack_sample(source[0],out,**args(source))
    assert out.exists() and not (out/'bundle.json').exists()
    assert (source[0]/'bundle.json').exists()


def test_npy_payload_different_even_if_logical_equal_is_not_allowed(source,tmp_path):
    out,_=convert(source,tmp_path);r=v2.SampleReader(out)
    name='inputs/depth_m.npy'
    path=out/r.manifest['files'][name]['stored_path'];original=path.read_bytes()
    size=struct.unpack('<I',original[8:12])[0]
    # Valid equivalent GHN with changed header whitespace, not a probabilistic
    # assumption that another compression level changes a tiny fixture.
    header=original[12:12+size]+b' '
    packed=original[:8]+struct.pack('<I',len(header))+header+original[12+size:]
    assert v2.decode_npy(packed)==r.read(name) and packed!=original
    path.write_bytes(packed)
    def update(m):
        e=m['files'][name];old=e['stored_bytes'];e.update(stored_bytes=len(packed),stored_sha256=v1.digest(packed))
        m['stored_payload_bytes']+=len(packed)-old
    edit_manifest(out,update)
    with pytest.raises(ValueError,match='Original PNG/ghn'):v2.SampleReader(out)


def test_excluded_without_trace(source,tmp_path):
    root,m=source
    # Build a separate synthetic V1 fixture, preserving the test's original source.
    old=v1.SampleReader(root)
    raw=tmp_path/'excluded_raw';raw.mkdir()
    for name in old.manifest['files']:
        if name=='supervision/query_trace.json':continue
        p=raw/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(old.read(name))
    meta=json.loads((raw/'sample.json').read_bytes())
    meta['files']={k:v for k,v in meta['files'].items() if v['role']!='review_only'}
    (raw/'sample.json').write_text(json.dumps(meta))
    (raw/'supervision/label.json').write_text('{"eligible":false,"training_approved":false}')
    excluded=tmp_path/'excluded_v1'
    manifest=v1.pack_sample(raw,excluded,sample_sha256=v1.digest((raw/'sample.json').read_bytes()),
        label_sha256=v1.digest((raw/'supervision/label.json').read_bytes()))
    report=v2.pack_sample(excluded,tmp_path/'excluded_v2',**args((excluded,manifest)))
    r=v2.SampleReader(report['destination']);assert r.verify_all()
    assert r.manifest['source_query_trace_sha256'] is None


def _stat_copy(info,**changes):
    fields=('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink')
    values={k:getattr(info,k) for k in fields};values.update(changes)
    return SimpleNamespace(**values)


def test_windows_handle_ctime_semantics_do_not_masquerade_as_changes(tmp_path,monkeypatch):
    p=tmp_path/'bytes.json';p.write_bytes(b'{}');real=os.fstat
    monkeypatch.setattr(v2.os,'fstat',lambda fd:_stat_copy(real(fd),st_ctime_ns=1))
    assert v2._read(p)==b'{}'


def test_opened_object_identity_must_match(tmp_path,monkeypatch):
    p=tmp_path/'bytes.json';p.write_bytes(b'{}');real=os.fstat
    monkeypatch.setattr(v2.os,'fstat',lambda fd:_stat_copy(real(fd),st_ino=real(fd).st_ino+1))
    with pytest.raises(ValueError,match='before read'):v2._read(p)


def test_path_ctime_still_checked_before_and_after_read(tmp_path,monkeypatch):
    p=tmp_path/'bytes.json';p.write_bytes(b'{}');real=v2._regular;calls=[]
    def changed(path):
        info=real(path);calls.append(1)
        return _stat_copy(info,st_ctime_ns=info.st_ctime_ns+1) if len(calls)>1 else info
    monkeypatch.setattr(v2,'_regular',changed)
    with pytest.raises(ValueError,match='during read'):v2._read(p)


def test_source_physical_alias_is_rejected_before_destination(source,tmp_path):
    os.link(source[0]/'bundle.json',tmp_path/'source_alias.json')
    out=tmp_path/'never_created'
    with pytest.raises(ValueError,match='Hard-linked'):v2.pack_sample(source[0],out,**args(source))
    assert not out.exists()


@pytest.mark.parametrize('suffix',[b'tail',codec.zstd.ZstdCompressor(write_checksum=True).compress(b'{}'),
                                  struct.pack('<II',0x184D2A50,0)],ids=['tail','second-frame','skippable'])
def test_v2_guards_unchanged_native_codec_against_trailing_frames(source,suffix):
    name='inputs/depth_m.npy';entry=source[1]['files'][name]
    raw=(source[0]/entry['stored_path']).read_bytes();n=struct.unpack('<I',raw[8:12])[0]
    header=json.loads(raw[12:12+n]);payload=raw[12+n:]+suffix
    header['compressed_sha256']=codec.digest(payload)
    h=json.dumps(header).encode();forged=raw[:8]+struct.pack('<I',len(h))+h+payload
    with pytest.raises(ValueError,match='trailing'):v2._decode_native(forged)


def test_native_bound_checked_before_frozen_decode(source,monkeypatch):
    entry=source[1]['files']['inputs/depth_m.npy']
    raw=(source[0]/entry['stored_path']).read_bytes();n=struct.unpack('<I',raw[8:12])[0]
    header=json.loads(raw[12:12+n]);header['original_size']=codec.MAX_BYTES+1
    h=json.dumps(header).encode();forged=raw[:8]+struct.pack('<I',len(h))+h+raw[12+n:]
    def forbidden(_):raise AssertionError('frozen decode must not run')
    monkeypatch.setattr(v2,'decode_npy',forbidden)
    with pytest.raises(ValueError,match='Bounded frame'):v2._decode_native(forged)
