"""CPU-only codec attacks and exact byte round trips; no file conversions."""
import json
import struct
from types import SimpleNamespace

import pytest
import zstandard as zstd

from . import json_bytes_codec as codec


def modify(encoded, edit=None, payload=None):
    n=struct.unpack('<I',encoded[8:12])[0]
    h=json.loads(encoded[12:12+n])
    p=encoded[12+n:] if payload is None else payload
    if payload is not None: h['compressed_sha256']=codec.digest(p)
    if edit: edit(h)
    raw=json.dumps(h).encode()
    return codec.MAGIC+struct.pack('<I',len(raw))+raw+p


@pytest.mark.parametrize('raw',[
    b'{ "z": -0.0, "a": 1.00000000000000000000000001e-009 }\r\n',
    b'\xef\xbb\xbf{"name":"unicode \\u03b1"}\n',
    '{"name":"토마토", "b":1,"a":2}\n'.encode(),
    b'{"a":1,"a":2}', b'[true, false, null, 1000000000000000000000000000001]',
    b' '+b'\n'*100000+b'{}', bytes(range(256))*100,
], ids=['numeric', 'bom', 'unicode', 'duplicate', 'large-int', 'whitespace', 'opaque'])
@pytest.mark.parametrize('level',[1,3,9,12])
def test_exact_opaque_bytes_not_normalization(raw,level):
    # Duplicate/opaque bytes are transportable; consumer JSON parsing is separate.
    packed=codec.encode(raw,level)
    assert codec.decode(packed)==raw
    assert codec.encode(raw,level)==packed


@pytest.mark.parametrize('level',[0,13,-1,True,3.0,None])
def test_bad_levels(level):
    with pytest.raises(ValueError): codec.encode(b'{}',level)


@pytest.mark.parametrize('raw',[b'',None,'{}',bytearray(b'{}')])
def test_bounded_bytes_required(raw):
    with pytest.raises(ValueError): codec.encode(raw)


def test_input_and_output_allocation_bounds(monkeypatch):
    encoded=codec.encode(b' ' * 1024)
    monkeypatch.setattr(codec,'MAX_BYTES',512)
    with pytest.raises(ValueError): codec.encode(b' '*513)
    with pytest.raises(ValueError): codec.decode(encoded)


@pytest.mark.parametrize('fault',['magic','empty','short','header_length','header_truncated','payload_truncated','bitflip','tail'])
def test_corruption(fault):
    p=codec.encode(b'{"hello":"world"}')
    if fault=='magic': p=b'BADMAGIC'+p[8:]
    if fault=='empty': p=b''
    if fault=='short': p=p[:11]
    if fault=='header_length': p=p[:8]+struct.pack('<I',codec.MAX_HEADER+1)+p[12:]
    if fault=='header_truncated': p=p[:20]
    if fault=='payload_truncated': p=p[:-1]
    if fault=='bitflip': p=p[:-1]+bytes([p[-1]^1])
    if fault=='tail': p+=b'junk'
    with pytest.raises(ValueError): codec.decode(p)


@pytest.mark.parametrize('field,value',[
    ('schema','other'),('logical_bytes',0),('logical_bytes',True),('logical_bytes',codec.MAX_BYTES+1),
    ('logical_bytes',1),('logical_sha256','0'*64),('compressed_sha256','0'*64),('logical_sha256','bad'),
    ('level',False),('level',13),('extra','unexpected')])
def test_forged_headers(field,value):
    p=modify(codec.encode(b'{}'),lambda h:h.update({field:value}))
    with pytest.raises(ValueError): codec.decode(p)


@pytest.mark.parametrize('suffix',[b'x',zstd.ZstdCompressor(write_checksum=True).compress(b'{}'),
                                  struct.pack('<II',0x184D2A50,0)])
def test_trailing_data_frames_and_skippable_frames_even_with_rehashed_payload(suffix):
    p=codec.encode(b'{}');n=struct.unpack('<I',p[8:12])[0]
    forged=modify(p,payload=p[12+n:]+suffix)
    with pytest.raises(ValueError,match='trailing'): codec.decode(forged)


@pytest.mark.parametrize('options',[{'write_checksum':False},{'write_checksum':True,'write_content_size':False}])
def test_checksum_and_declared_content_size_required(options):
    p=modify(codec.encode(b'{}'),payload=zstd.ZstdCompressor(**options).compress(b'{}'))
    with pytest.raises(ValueError,match='parameters'): codec.decode(p)


@pytest.mark.parametrize('change',[{'window_size':codec.MAX_BYTES+1},{'dict_id':42}])
def test_window_and_dictionary_rejected_before_decompression(monkeypatch,change):
    frame=dict(content_size=2,window_size=2,dict_id=0,has_checksum=True);frame.update(change)
    monkeypatch.setattr(codec.zstd,'get_frame_parameters',lambda _:SimpleNamespace(**frame))
    with pytest.raises(ValueError,match='parameters'): codec.decode(codec.encode(b'{}'))


@pytest.mark.parametrize('raw',[b'{"a":1,"a":2}',b'{"x":NaN}',b'{"x":Infinity}',b'{"x":1e999}',b'\xff'])
def test_consumer_json_rejects_ambiguity_without_changing_codec(raw):
    assert codec.decode(codec.encode(raw))==raw
    with pytest.raises(ValueError): codec.parse_json(raw)


def test_bom_and_numeric_spelling_remain_in_original_bytes():
    raw=b'\xef\xbb\xbf{ "z": -0.0, "a": 1e-9 }\r\n'
    assert codec.parse_json(raw)=={'z':-0.0,'a':1e-9}
    assert codec.decode(codec.encode(raw))==raw
