"""Explicit V2 lossless JSON storage; frozen V1 is neither changed nor patched.

SampleReader(root, *, expected_bindings=None, expected_json=None) exposes the
same native logical read/array/image/json/verify_all API. It accepts ONLY V2.
expected_bindings may additionally pin bundle.json and source_bundle.json.
read('source_bundle.json') returns the exact original V1 manifest bytes.

pack_sample(source, destination, *, source_bundle_sha256, sample_sha256,
            label_sha256, query_trace=None, level=3) converts ONE pinned V1
compact into a NEW V2 directory. Raw inputs, V2 repacks, and additional logical
payload kinds are explicitly unsupported. No labels/annotations are recomputed.
Every existing PNG and .ghn is copied exactly; JSON bytes are compressed without
reserialization, including the original V1 manifest as provenance. The new
manifest refers to the trace hash, never repeats the whole trace inline.

No native writer integration, release/admission/training qualification, source
deletion or storage reclamation. Copies ADD disk usage. Partial new output is
retained on error and is never a successful conversion. Metadata/JSON is bounded
to 64MiB per file; no temporary decoded images/files. This is a representation
converter, not a 20k release or a claim of biological independence.
"""
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import re
import stat
import struct

import numpy as np
from PIL import Image

from ..native_lossless_codec import decode as decode_npy, _native_npy, MAGIC as NPY_MAGIC
from . import bundle as v1
from . import json_bytes_codec as codec
from .json_bytes_codec import checked_sha, digest, parse_json, require

SCHEMA = 'greenhouse.compact_native_sample.v2'
ARRAYS = v1.ARRAYS
REQUIRED = v1.REQUIRED
EXTRAS = v1.EXTRAS
SOURCE = 'source_bundle.json'
MAX_FILE_BYTES = codec.MAX_BYTES
_ENTRY = {'stored_path','encoding','logical_bytes','stored_bytes','logical_sha256','stored_sha256'}
_FLAGS = dict(lossy=False, depth_recomputed=False, source_removed=False, training_approved=False)
_POLICY = dict(json_policy='exact_original_bytes_zstd', rgb_policy='unchanged_original_png',
               native_array_policy='unchanged_original_ghn_bytes')
_MANIFEST = {'schema','files','omitted_review_files','source_sample_sha256','source_label_sha256',
    'source_query_trace_sha256','source_bundle','source_schema','source_logical_bytes','stored_payload_bytes',
    *_FLAGS, *_POLICY}
_RESERVED = {'CON','PRN','AUX','NUL',*(f'{p}{i}' for p in ('COM','LPT') for i in range(1,10))}


def _name(name):
    require(isinstance(name,str) and name and name.isascii(), 'Portable relative path required')
    for part in name.split('/'):
        require(re.fullmatch(r'[A-Za-z0-9_.-]+',part) is not None and part not in ('.','..')
                and not part.endswith('.') and part.split('.')[0].upper() not in _RESERVED,
                'Unsafe or aliased relative path')
    return name


def _regular(path, directory=False):
    info = path.lstat()
    require(not stat.S_ISLNK(info.st_mode) and not getattr(info,'st_file_attributes',0)&0x400,
            'Symlink/junction/reparse path unsupported')
    require(stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode), 'Regular path required')
    if not directory:
        require(info.st_nlink==1, 'Hard-linked file alias unsupported')
    return info


def _absolute(path, *, exists=True):
    path = Path(os.path.abspath(path))
    # Do not resolve away links before checking them, including root ancestors.
    for ancestor in reversed((path,*path.parents)):
        if ancestor.exists() or ancestor.is_symlink():
            info = ancestor.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not getattr(info,'st_file_attributes',0)&0x400,
                    'Linked root/ancestor unsupported')
    if exists:
        _regular(path,directory=True)
    return path


def safe_path(root, name):
    _name(name)
    path = root.joinpath(*name.split('/'))
    require(path.is_relative_to(root), 'Path escapes root')
    for ancestor in path.parents:
        if ancestor==root: break
        if ancestor.exists() or ancestor.is_symlink(): _regular(ancestor,directory=True)
    if path.exists() or path.is_symlink(): _regular(path)
    return path


def _read(path, limit=MAX_FILE_BYTES):
    before = _regular(path)
    def identity(info):
        # Python 3.13 on Windows can expose creation time via lstat().ctime
        # but change time via fstat().ctime. Compare ctime only path-to-path;
        # opened-object identity/size/mtime plus byte hashes remain checked.
        return (info.st_dev,info.st_ino,info.st_mode,info.st_size,info.st_mtime_ns,info.st_nlink)
    require(0<before.st_size<=limit, 'Stored file exceeds bound')
    with path.open('rb') as stream:
        require(identity(os.fstat(stream.fileno()))==identity(before), 'File changed before read')
        raw = stream.read(limit+1)
    after = _regular(path)
    require(0<len(raw)<=limit and identity(after)==identity(before)
            and after.st_ctime_ns==before.st_ctime_ns, 'File changed during read')
    return raw


def _tree(root):
    _absolute(root)
    result, folded, inodes = set(), set(), set()
    for parent,dirs,files in os.walk(root,followlinks=False):
        for name in dirs+files:
            path = Path(parent)/name
            relative = path.relative_to(root).as_posix()
            _name(relative)
            require(relative.casefold() not in folded, 'Case-colliding path alias')
            folded.add(relative.casefold())
            info = _regular(path,directory=name in dirs)
            if name in files:
                key = (info.st_dev,info.st_ino)
                require(key not in inodes, 'Physical file alias')
                inodes.add(key); result.add(relative)
    return result


def _encoding(name):
    return ('native_npy_byteplanes_zstd' if name in ARRAYS else
            codec.ENCODING if name.endswith('.json') else 'identity')


def _decode_native(stored):
    # Keep frozen GHN decoding/bytes unchanged, but add V2 frame bounds and
    # trailing-frame rejection BEFORE calling it. This intentionally performs
    # a second bounded decompression; it is not a new native array codec.
    require(12<len(stored)<=codec.MAX_ENCODED and stored.startswith(NPY_MAGIC), 'Invalid native envelope')
    size=struct.unpack('<I',stored[8:12])[0]
    require(0<size<=codec.MAX_HEADER and 12+size<len(stored), 'Invalid native header bound')
    header=parse_json(stored[12:12+size])
    require(isinstance(header,dict), 'Invalid native header')
    codec.decode_frame(stored[12+size:],header.get('original_size'))
    return decode_npy(stored)


def _entry(root, name, item):
    require(isinstance(item,dict) and set(item)==_ENTRY, 'Invalid storage entry')
    safe_path(root,item['stored_path'])
    require(item['encoding']==_encoding(name), 'Wrong file encoding')
    require(type(item['logical_bytes']) is int and 0<item['logical_bytes']<=MAX_FILE_BYTES
            and type(item['stored_bytes']) is int and 0<item['stored_bytes']<=codec.MAX_ENCODED,
            'Unbounded entry size')
    checked_sha(item['logical_sha256']); checked_sha(item['stored_sha256'])


class SampleReader:
    def __init__(self, root, *, expected_bindings=None, expected_json=None):
        self.root = _absolute(root)
        self._expected_bindings = deepcopy(dict(expected_bindings or {}))
        self._expected_json = deepcopy(dict(expected_json or {}))
        for name,pin in self._expected_bindings.items(): _name(name); checked_sha(pin)
        for name in self._expected_json: _name(name)
        raw = _read(self.root/'bundle.json')
        self._manifest_sha256 = digest(raw)
        self._manifest = m = parse_json(raw)
        require(isinstance(m,dict) and set(m)==_MANIFEST and m['schema']==SCHEMA
                and all(m[k] is v for k,v in _FLAGS.items())
                and all(m[k]==v for k,v in _POLICY.items()), 'Unsupported V2 manifest')
        require(m['source_schema']==v1.SCHEMA and isinstance(m['files'],dict)
                and isinstance(m['omitted_review_files'],dict), 'Missing V1 source provenance')
        trace_pin = m['source_query_trace_sha256']
        if trace_pin is not None: checked_sha(trace_pin)
        names = REQUIRED|{'sample.json','supervision/label.json'}
        if trace_pin is not None: names = names|{'supervision/query_trace.json'}
        require(set(m['files'])==names, 'Unsupported/missing logical payloads')
        self._entries = dict(m['files'], **{SOURCE:m['source_bundle']})
        require(set(self._expected_bindings)|set(self._expected_json) <= names|{SOURCE,'bundle.json'},
                'Unknown external binding name')
        paths = {'bundle.json'}
        for name,item in self._entries.items():
            _entry(self.root,name,item)
            path = item['stored_path']
            require(path.casefold() not in {p.casefold() for p in paths}, 'Aliased stored path')
            paths.add(path)
        self._physical = paths
        require(_tree(self.root)==paths, 'Unexpected/missing physical files')
        for name,pin in m['omitted_review_files'].items():
            _name(name); checked_sha(pin)
            require(name.startswith('review/'), 'Only review files may be omitted')
        for k in ('source_logical_bytes','stored_payload_bytes'):
            require(type(m[k]) is int and m[k]==sum(e['logical_bytes' if k=='source_logical_bytes' else 'stored_bytes']
                for e in m['files'].values()), 'Wrong payload byte total')
        self._check_expected('bundle.json',raw)
        self._metadata = parse_json(self.read('sample.json'))
        meta = self._metadata
        require(isinstance(meta,dict) and meta.get('training_sample_approved') is False
                and meta.get('training_approved',False) is False and isinstance(meta.get('files'),dict),
                'Unapproved original metadata required')
        expected_omitted = {}
        for name,item in meta['files'].items():
            _name(name); checked_sha(item['sha256'])
            if item['role']=='review_only':
                require(name.startswith('review/'), 'Invalid review omission')
                expected_omitted[name] = item['sha256']
            else:
                require(name in REQUIRED and item['role']==('observation' if name.startswith('inputs/') else 'ground_truth_supervision')
                        and m['files'][name]['logical_sha256']==item['sha256'], 'Changed original file binding/role')
        require(set(meta['files'])==REQUIRED|set(expected_omitted)
                and expected_omitted==m['omitted_review_files'], 'Incomplete original files/omissions')
        require(checked_sha(m['source_sample_sha256'])==m['files']['sample.json']['logical_sha256']
                and checked_sha(m['source_label_sha256'])==m['files']['supervision/label.json']['logical_sha256'],
                'Changed sample/label binding')
        source = parse_json(self.read(SOURCE))
        require(source.get('schema')==v1.SCHEMA and source.get('lossy') is False
                and source.get('depth_recomputed') is False and source.get('training_approved') is False
                and set(source['files'])==names and source['omitted_review_files']==expected_omitted,
                'Source V1 manifest differs')
        require(source['source_sample_sha256']==m['source_sample_sha256']
                and source['source_label_sha256']==m['source_label_sha256'], 'Source provenance pins differ')
        for name,item in m['files'].items():
            old = source['files'][name]
            require(old['logical_sha256']==item['logical_sha256'] and old['logical_bytes']==item['logical_bytes'],
                    'Original logical bytes differ')
            if not name.endswith('.json'):
                require(old['stored_sha256']==item['stored_sha256'] and old['stored_bytes']==item['stored_bytes']
                        and old['encoding']==item['encoding'], 'Original PNG/ghn bytes differ')
        self._source_trace = source.get('source_query_trace')
        self._verify_extras()

    @property
    def manifest(self): return deepcopy(self._manifest)
    @property
    def metadata(self): return deepcopy(self._metadata)
    @property
    def expected_bindings(self): return deepcopy(self._expected_bindings)
    @property
    def expected_json(self): return deepcopy(self._expected_json)

    def _check_expected(self,name,raw):
        if name in self._expected_bindings:
            require(digest(raw)==self._expected_bindings[name], 'Unexpected source binding: '+name)
        if name in self._expected_json:
            require(parse_json(raw)==self._expected_json[name], 'Unexpected source JSON: '+name)

    def read(self,name):
        _name(name)
        _absolute(self.root)
        marker = _read(self.root/'bundle.json')
        require(digest(marker)==self._manifest_sha256, 'Manifest changed after construction')
        if name=='bundle.json': raw = marker
        else:
            require(name in self._entries, 'Logical file omitted or unavailable')
            entry = self._entries[name]
            stored = _read(safe_path(self.root,entry['stored_path']),codec.MAX_ENCODED)
            require(len(stored)==entry['stored_bytes'] and digest(stored)==entry['stored_sha256'], 'Corrupt stored bytes')
            raw = (codec.decode(stored) if entry['encoding']==codec.ENCODING else
                   _decode_native(stored) if entry['encoding']=='native_npy_byteplanes_zstd' else stored)
            require(len(raw)==entry['logical_bytes'] and digest(raw)==entry['logical_sha256'], 'Corrupt logical bytes')
        self._check_expected(name,raw)
        return raw

    def array(self,name):
        require(name in ARRAYS, 'Unknown native array')
        return _native_npy(self.read(name))

    def image(self,name):
        with Image.open(io.BytesIO(self.read(name))) as image:
            return np.asarray(image).copy()

    def json(self,name): return parse_json(self.read(name))

    def _verify_extras(self):
        label = self.json('supervision/label.json')
        require(isinstance(label,dict) and type(label.get('eligible')) is bool
                and label.get('training_approved') is False, 'Unapproved original label required')
        pin = self._manifest['source_query_trace_sha256']
        require(label['eligible']==(pin is not None), 'Eligible/trace presence mismatch')
        if pin is not None:
            raw = self.read('supervision/query_trace.json')
            require(digest(raw)==pin and parse_json(raw)==self._source_trace, 'Changed original trace')
        else: require(self._source_trace is None, 'Unexpected excluded trace')

    def verify_all(self):
        require(_tree(self.root)==self._physical, 'Physical inventory changed')
        for name in self._entries: self.read(name)
        self._verify_extras()
        return True


def pack_sample(source,destination,*,source_bundle_sha256,sample_sha256,label_sha256,query_trace=None,level=3):
    """New V1->V2 sidecar copy; source/PNG/.ghn bytes are never rewritten."""
    source,destination = _absolute(source),_absolute(destination,exists=False)
    require(source!=destination and not destination.is_relative_to(source)
            and not source.is_relative_to(destination), 'Separate nonnested destination required')
    require(not destination.exists() and destination.parent.is_dir(), 'Create-only destination in existing parent required')
    for pin in (source_bundle_sha256,sample_sha256,label_sha256): checked_sha(pin)
    require(type(level) is int and 1<=level<=12, 'Compression level must be 1..12')
    require((source/'bundle.json').is_file(), 'Raw input explicitly unsupported')
    original = _read(source/'bundle.json')
    require(digest(original)==source_bundle_sha256, 'Source bundle pin differs')
    old = parse_json(original)
    require(old.get('schema')==v1.SCHEMA, 'Only original V1 compact input supported')
    require(old.get('source_query_trace')==query_trace, 'Explicit original query trace required')
    names = REQUIRED|{'sample.json','supervision/label.json'}|({'supervision/query_trace.json'} if query_trace is not None else set())
    require(set(old['files'])==names, 'Additional logical payloads explicitly unsupported')
    physical = {'bundle.json'}
    for item in old['files'].values():
        safe_path(source,item['stored_path']); physical.add(item['stored_path'])
    require(len(physical)==len(names)+1 and _tree(source)==physical, 'V1 physical inventory/aliases differ')
    snapshot = {name:digest(_read(safe_path(source,name),codec.MAX_ENCODED)) for name in physical}
    for name in ARRAYS:
        _decode_native(_read(safe_path(source,old['files'][name]['stored_path']),codec.MAX_ENCODED))
    reader = v1.SampleReader(source,expected_bindings={'sample.json':sample_sha256,'supervision/label.json':label_sha256},
        expected_json={'supervision/query_trace.json':query_trace} if query_trace is not None else {})
    reader.verify_all()
    payloads,entries = {},{}
    for name in sorted(names|{SOURCE}):
        logical = original if name==SOURCE else reader.read(name)
        if name.endswith('.json'):
            parse_json(logical)
            packed = codec.encode(logical,level=level)
            require(codec.decode(packed)==logical, 'Non-exact JSON roundtrip')
        else:
            packed = _read(safe_path(source,old['files'][name]['stored_path']),codec.MAX_ENCODED)
            require((_decode_native(packed) if name in ARRAYS else packed)==logical, 'Non-exact original PNG/ghn')
        stored_path = ('provenance/' if name==SOURCE else 'payload/')+name+(
            '.ghj' if name.endswith('.json') else '.ghn' if name in ARRAYS else '')
        payloads[stored_path] = packed
        entries[name] = dict(stored_path=stored_path,encoding=_encoding(name),logical_bytes=len(logical),
            stored_bytes=len(packed),logical_sha256=digest(logical),stored_sha256=digest(packed))
    provenance = entries.pop(SOURCE)
    manifest = dict(schema=SCHEMA,files=entries,source_bundle=provenance,source_schema=v1.SCHEMA,
        omitted_review_files=deepcopy(old['omitted_review_files']),source_sample_sha256=sample_sha256,
        source_label_sha256=label_sha256,source_query_trace_sha256=(entries['supervision/query_trace.json']['logical_sha256']
            if query_trace is not None else None),source_logical_bytes=sum(e['logical_bytes'] for e in entries.values()),
        stored_payload_bytes=sum(e['stored_bytes'] for e in entries.values()),**_FLAGS,**_POLICY)
    marker = json.dumps(manifest,indent=2,allow_nan=False).encode('utf-8')
    require(len(marker)<=MAX_FILE_BYTES, 'Manifest exceeds bound')
    destination.mkdir(exist_ok=False)
    for name,raw in payloads.items():
        path = safe_path(destination,name); path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream: stream.write(raw)
        require(_read(path,codec.MAX_ENCODED)==raw, 'Changed written payload')
    require(_tree(source)==physical and all(digest(_read(safe_path(source,n),codec.MAX_ENCODED))==h
            for n,h in snapshot.items()), 'Source changed during conversion')
    # Marker is last; no existing destination, source, or partial is deleted.
    with (destination/'bundle.json').open('xb') as stream: stream.write(marker)
    checked = SampleReader(destination,expected_bindings={'bundle.json':digest(marker),SOURCE:source_bundle_sha256,
        'sample.json':sample_sha256,'supervision/label.json':label_sha256},expected_json=reader.expected_json)
    checked.verify_all()
    require(_tree(source)==physical and all(digest(_read(safe_path(source,n),codec.MAX_ENCODED))==h
            for n,h in snapshot.items()), 'Source changed at completion')
    return dict(schema='greenhouse.compact_json_conversion.v1',source=str(source),destination=str(destination),
        source_bundle_sha256=source_bundle_sha256,bundle_sha256=digest(marker),
        source_bytes=sum((source/n).stat().st_size for n in physical),
        added_copy_bytes=sum((destination/n).stat().st_size for n in _tree(destination)),
        exact_logical_files=len(entries),exact_png_and_ghn=True,original_manifest_bytes_preserved=True,
        raw_bytes_deleted=0,partial_output_cleanup_performed=False,**_FLAGS)
