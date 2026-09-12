import struct
import pytest
from sim_physics.pool_tags import decode,ENTRY,MAX_BUFFER_BYTES


def buffer(*tags):
    return struct.pack('<I4x',len(tags))+b''.join(ENTRY.pack(t,10,3,237137918208,5,2,64) for t in tags)


def test_x64_padding_and_64_bit_bytes_are_not_truncated():
    b=buffer(b'CBnb',b'Abc ');r=decode(b,len(b))
    assert ENTRY.size==40 and len(r)==2
    assert r[0]['paged_bytes']==237137918208 and r[0]['paged_allocs']==10
    assert r[1]['tag']=='Abc ' and r[1]['nonpaged_bytes']==64


@pytest.mark.parametrize('length',[0,7,47,49,True,MAX_BUFFER_BYTES+1])
def test_truncated_or_invalid_return_length_rejects(length):
    with pytest.raises(ValueError):decode(buffer(b'CBnb'),length)


def test_duplicate_tag_is_not_summed_twice():
    b=buffer(b'CBnb',b'CBnb')
    with pytest.raises(ValueError,match='Duplicate'):decode(b,len(b))


def test_zero_tags_and_non_ascii_tags_are_serializable():
    assert decode(buffer(),8)==[]
    b=buffer(b'\xff\0Aa');r=decode(b,len(b))[0]
    assert r['tag_hex']=='ff004161' and r['tag'].startswith('\\xff')
