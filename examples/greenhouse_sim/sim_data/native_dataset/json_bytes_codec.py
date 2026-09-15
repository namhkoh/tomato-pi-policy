"""Bounded exact-byte JSON transport, not JSON reserialization.

Opaque source bytes (including whitespace, key order and numeric spelling) are
preserved. JSON semantics are checked by the consumer, not normalized here.
No files, native jobs, model inputs, approval or calibration are created.
"""
import hashlib
import json
import struct

import zstandard as zstd

MAGIC = b'GHJSON01'
SCHEMA = 'greenhouse.exact_json_bytes_zstd.v1'
ENCODING = 'json_bytes_zstd.v1'
MAX_BYTES = 64*1024*1024
MAX_HEADER = 4096
MAX_ENCODED = MAX_BYTES+65536
_FIELDS = {'schema','logical_bytes','logical_sha256','compressed_sha256','level'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def checked_sha(value):
    require(isinstance(value,str) and len(value)==64
            and all(c in '0123456789abcdef' for c in value), 'SHA256 required')
    return value


def parse_json(raw):
    """Strict consumer parsing; never used to rewrite source bytes."""
    def pairs(items):
        result = {}
        for k,v in items:
            require(k not in result, 'Duplicate JSON key')
            result[k] = v
        return result
    try:
        result = json.loads(raw, object_pairs_hook=pairs)
        json.dumps(result, allow_nan=False)  # Reject NaN/Infinity/exponent overflow.
    except (ValueError, TypeError, UnicodeError, RecursionError) as exc:
        raise ValueError('Invalid or ambiguous JSON') from exc
    return result


def encode(raw, level=3):
    """Compress original bytes; no parse/dump, float conversion or trimming."""
    require(isinstance(raw,bytes) and 0<len(raw)<=MAX_BYTES, 'Bounded source bytes required')
    require(type(level) is int and 1<=level<=12, 'Compression level must be 1..12')
    payload = zstd.ZstdCompressor(level=level, write_checksum=True, write_content_size=True).compress(raw)
    header = dict(schema=SCHEMA, logical_bytes=len(raw), logical_sha256=digest(raw),
                  compressed_sha256=digest(payload), level=level)
    metadata = json.dumps(header, sort_keys=True, separators=(',',':')).encode('ascii')
    require(len(metadata)<=MAX_HEADER, 'Codec header too large')
    result = MAGIC+struct.pack('<I',len(metadata))+metadata+payload
    require(len(result)<=MAX_ENCODED, 'Encoded JSON exceeds bound')
    return result


def decode_frame(payload, size):
    """Bounded single Zstandard frame, shared by the V2 transport guard."""
    require(isinstance(payload,bytes) and 0<len(payload)<=MAX_ENCODED
            and type(size) is int and 0<size<=MAX_BYTES, 'Bounded frame required')
    try:
        frame = zstd.get_frame_parameters(payload)
        require(frame.content_size==size and frame.window_size<=MAX_BYTES
                and frame.dict_id==0 and frame.has_checksum, 'Unsupported Zstandard frame parameters')
        raw = zstd.ZstdDecompressor(max_window_size=MAX_BYTES//1024).decompress(
            payload, max_output_size=size, allow_extra_data=False)
    except zstd.ZstdError as exc:
        raise ValueError('Invalid, truncated or trailing JSON/native frame') from exc
    require(len(raw)==size, 'Decoded frame size differs')
    return raw


def decode(encoded):
    """One checksummed, dictionary-free frame; bounded output AND window."""
    require(isinstance(encoded,bytes) and 12<len(encoded)<=MAX_ENCODED
            and encoded.startswith(MAGIC), 'Bounded JSON codec envelope required')
    length = struct.unpack('<I',encoded[8:12])[0]
    require(0<length<=MAX_HEADER and 12+length<len(encoded), 'Invalid codec header length')
    header = parse_json(encoded[12:12+length])
    require(isinstance(header,dict) and set(header)==_FIELDS and header['schema']==SCHEMA,
            'Unknown JSON codec schema/fields')
    size = header['logical_bytes']
    require(type(size) is int and 0<size<=MAX_BYTES, 'Invalid decoded size')
    require(type(header['level']) is int and 1<=header['level']<=12, 'Invalid codec level')
    checked_sha(header['logical_sha256']); checked_sha(header['compressed_sha256'])
    payload = encoded[12+length:]
    require(digest(payload)==header['compressed_sha256'], 'Compressed JSON checksum mismatch')
    raw = decode_frame(payload,size)
    require(len(raw)==size and digest(raw)==header['logical_sha256'], 'Decoded JSON checksum mismatch')
    return raw
