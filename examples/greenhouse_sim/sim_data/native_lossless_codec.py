"""Exact native NPY sidecar storage; never computes or quantizes depth.

Reorders opaque bytes for Zstandard compression, preserving the entire NPY
file (header, dtype, shape, signed zeros and NaN bits). Sources are never
deleted or modified. Legacy capture readers/writers remain unchanged.
"""
from pathlib import Path
import hashlib
import io
import json
import math
import struct
import numpy as np

MAGIC=b'GHNPYB01'
MAX_BYTES=64*1024*1024
MAX_HEADER=4096


def _sha(raw):return hashlib.sha256(raw).hexdigest()


def _native_npy(raw):
    if not isinstance(raw,bytes) or not 0<len(raw)<=MAX_BYTES:raise ValueError('Bounded NPY bytes required')
    try:
        stream=io.BytesIO(raw);version=np.lib.format.read_magic(stream)
        readers={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}
        if version not in readers:raise ValueError('Unsupported NPY version')
        shape,fortran,dtype=readers[version](stream,max_header_size=MAX_HEADER)
        if (len(shape)!=2 or any(type(n) is not int or n<=0 for n in shape)
                or fortran or dtype.kind not in 'fui' or dtype.itemsize not in (1,2,4,8)):
            raise ValueError('Nonempty C-order numeric native image required')
        size=math.prod(shape)*dtype.itemsize
        if size>MAX_BYTES or stream.tell()+size!=len(raw):raise ValueError('NPY shape/file size mismatch')
        # Validate declared shape BEFORE allocation; never load pickle or trust
        # a tiny file claiming a huge array, even inside a valid codec envelope.
        array=np.frombuffer(raw,dtype=dtype,offset=stream.tell()).reshape(shape)
    except Exception as exc:raise ValueError('Numeric native NPY required') from exc
    if not isinstance(array,np.ndarray) or array.dtype.kind not in 'fui' or array.dtype.itemsize not in (1,2,4,8):
        raise ValueError('Numeric native NPY required')
    if array.ndim!=2 or array.size==0 or not array.flags.c_contiguous:raise ValueError('Nonempty C-order native image array required')
    return array


def encode(raw,level=9):
    import zstandard as zstd
    a=_native_npy(raw)
    if type(level) is not int or not 1<=level<=12:raise ValueError('Bounded compression level required')
    width=a.dtype.itemsize
    shuffled=b''.join(raw[i::width] for i in range(width))
    payload=zstd.ZstdCompressor(level=level,write_checksum=True).compress(shuffled)
    header=dict(schema='greenhouse.exact_native_npy_byteplanes.v1',original_size=len(raw),
        original_sha256=_sha(raw),compressed_sha256=_sha(payload),width=width,
        dtype=a.dtype.str,shape=list(a.shape),depth_recomputed=False,lossy=False)
    metadata=json.dumps(header,sort_keys=True,separators=(',',':')).encode()
    if len(metadata)>MAX_HEADER:raise ValueError('Unexpected codec header size')
    return MAGIC+struct.pack('<I',len(metadata))+metadata+payload


def decode(encoded):
    import zstandard as zstd
    if not isinstance(encoded,bytes) or not len(MAGIC)+4<len(encoded)<=MAX_BYTES+65536 or not encoded.startswith(MAGIC):
        raise ValueError('Bounded native codec container required')
    count=struct.unpack('<I',encoded[len(MAGIC):len(MAGIC)+4])[0]
    if not 0<count<=MAX_HEADER:raise ValueError('Invalid codec header length')
    offset=len(MAGIC)+4
    try:header=json.loads(encoded[offset:offset+count])
    except Exception as exc:raise ValueError('Invalid codec metadata') from exc
    if not isinstance(header,dict) or header.get('schema')!='greenhouse.exact_native_npy_byteplanes.v1':
        raise ValueError('Unknown native codec')
    size,width=header.get('original_size'),header.get('width')
    if type(size) is not int or not 0<size<=MAX_BYTES or type(width) is not int or width not in (1,2,4,8):
        raise ValueError('Invalid uncompressed size or byte width')
    if header.get('depth_recomputed') is not False or header.get('lossy') is not False:raise ValueError('Non-lossless metadata')
    payload=encoded[offset+count:]
    if _sha(payload)!=header.get('compressed_sha256'):raise ValueError('Compressed checksum mismatch')
    try:
        if zstd.frame_content_size(payload)!=size:raise ValueError('Wrong Zstandard content size')
        shuffled=zstd.ZstdDecompressor().decompress(payload,max_output_size=size)
    except zstd.ZstdError as exc:raise ValueError('Invalid compressed native payload') from exc
    if len(shuffled)!=size:raise ValueError('Decoded size mismatch')
    raw=bytearray(size);offset=0
    for i in range(width):
        length=(size+width-1-i)//width
        raw[i::width]=shuffled[offset:offset+length];offset+=length
    raw=bytes(raw)
    if _sha(raw)!=header.get('original_sha256'):raise ValueError('Native file checksum mismatch')
    a=_native_npy(raw)
    if a.dtype.str!=header.get('dtype') or list(a.shape)!=header.get('shape') or a.dtype.itemsize!=width:
        raise ValueError('Native array metadata mismatch')
    return raw


def pack_file(source,destination,level=9):
    """Create-only verified pack; source remains intact. No directory deletion."""
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if source==destination:raise ValueError('Source and destination must differ')
    raw=source.read_bytes();packed=encode(raw,level)
    if decode(packed)!=raw:raise ValueError('Non-exact codec roundtrip')
    with destination.open('xb') as stream:stream.write(packed)
    if source.read_bytes()!=raw:raise ValueError('Native source changed during packing')
    return dict(source=str(source),destination=str(destination),original_bytes=len(raw),
        packed_bytes=len(packed),original_sha256=_sha(raw),packed_sha256=_sha(packed),
        native_file_bytes_exact=True,depth_recomputed=False,source_removed=False)


def load_array(path):
    """Explicit adapter for raw NPY or this codec; never pickle-loads arrays."""
    raw=Path(path).read_bytes()
    return _native_npy(decode(raw) if raw.startswith(MAGIC) else raw)
