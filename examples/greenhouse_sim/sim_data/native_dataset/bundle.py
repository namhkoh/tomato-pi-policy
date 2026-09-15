"""Create-only lossless native sidecars and a hash-validating logical reader.

Original captures/decisions are immutable. Review PNGs are explicitly omitted
from compact copies, never substituted for RGB. No interpolation, quantization,
depth reconstruction, split changes or training approval.
"""
from pathlib import Path, PurePosixPath
import hashlib
import io
import json
import numpy as np
from PIL import Image
from ..native_lossless_codec import encode, decode, _native_npy
from ..dataset_review import require

SCHEMA = 'greenhouse.compact_native_sample.v1'
ARRAYS = {'inputs/depth_m.npy', 'supervision/renderer_instance_id.npy', 'supervision/component_id.npy'}
EXTRAS = {'sample.json', 'supervision/label.json', 'supervision/query_trace.json'}
MAX_FILE_BYTES = 64*1024*1024
REQUIRED = {'inputs/rgb.png','inputs/depth_valid.png','supervision/target_visible.png',
            'supervision/organ_type.png','supervision/identities.json'} | ARRAYS

def digest(raw):
    return hashlib.sha256(raw).hexdigest()

def safe_path(root, name):
    require(isinstance(name,str) and name and '\\' not in name and ':' not in name,
            'Invalid logical path')
    parts = PurePosixPath(name).parts
    require(not name.startswith('/') and all(p not in ('.','..') for p in name.split('/'))
            and '' not in name.split('/'), 'Noncanonical logical path')
    root = Path(root).resolve(); path = root.joinpath(*parts).resolve()
    require(path.is_relative_to(root) and path != root, 'Path escapes sample')
    return path

def read_bounded(path):
    with path.open('rb') as stream:
        raw = stream.read(MAX_FILE_BYTES+1)
    require(0 < len(raw) <= MAX_FILE_BYTES, 'File changed while reading')
    return raw

class SampleReader:
    """Read unchanged raw samples or exact compact copies through logical names."""
    def __init__(self, root, *, expected_bindings=None, expected_json=None):
        self.root = Path(root).resolve()
        self.expected_bindings = dict(expected_bindings or {})
        self.expected_json = dict(expected_json or {})
        compact = self.root/'bundle.json'
        self.manifest = json.loads(read_bounded(compact)) if compact.exists() else None
        if self.manifest is not None:
            m = self.manifest
            require(m.get('schema') == SCHEMA and m.get('lossy') is False
                    and m.get('depth_recomputed') is False and m.get('training_approved') is False,
                    'Unsupported compact contract')
            require(isinstance(m.get('files'),dict) and 'sample.json' in m['files'], 'Missing logical files')
            require(isinstance(m.get('omitted_review_files'),dict), 'Missing omission inventory')
            paths=set()
            for name, entry in m['files'].items():
                safe_path(self.root,name)
                p=safe_path(self.root,entry['stored_path'])
                require(p not in paths, 'Aliased stored files'); paths.add(p)
                require(not name.startswith('review/'), 'Review image cannot enter compact model storage')
                expected='native_npy_byteplanes_zstd' if name in ARRAYS else 'identity'
                require(entry['encoding']==expected, 'Unexpected logical file encoding')
        self.metadata = json.loads(self.read('sample.json'))
        require(isinstance(self.metadata.get('files'),dict), 'Missing original file bindings')
        require(REQUIRED <= self.metadata['files'].keys(), 'Complete native observations required')
        for name in REQUIRED:
            role='observation' if name.startswith('inputs/') else 'ground_truth_supervision'
            require(self.metadata['files'][name]['role']==role, 'Incorrect required observation role')
        if self.manifest is not None:
            files=self.manifest['files']; omitted=self.manifest['omitted_review_files']
            for name, entry in self.metadata['files'].items():
                if entry['role']=='review_only':
                    require(name.startswith('review/') and omitted.get(name)==entry['sha256'],
                            'Wrong review omission')
                else:
                    require(name in files and files[name]['logical_sha256']==entry['sha256'],
                            'Missing original native file binding')
            require(set(omitted)=={k for k,v in self.metadata['files'].items() if v['role']=='review_only'},
                    'Invented review omission')
            require(set(files)<=set(self.metadata['files'])|EXTRAS, 'Unexpected logical content')
            require(self.manifest['source_sample_sha256']==files['sample.json']['logical_sha256'],
                    'Changed source metadata binding')
            require(self.manifest['source_label_sha256']==files['supervision/label.json']['logical_sha256'],
                    'Changed source label binding')
            self._verify_extras()

    def _verify_extras(self):
        label=self.json('supervision/label.json')
        if label.get('eligible') is True:
            trace=self.json('supervision/query_trace.json')
            if self.manifest is not None:
                require(trace==self.manifest.get('source_query_trace'), 'Changed or missing required query trace')

    def read(self, name):
        safe_path(self.root,name)
        if self.manifest is None:
            raw=read_bounded(safe_path(self.root,name))
            if hasattr(self,'metadata') and name in self.metadata['files']:
                require(digest(raw)==self.metadata['files'][name]['sha256'], 'Changed raw source file')
            if name in self.expected_bindings:
                require(digest(raw)==self.expected_bindings[name], 'Unexpected source binding: '+name)
            if name in self.expected_json:
                require(json.loads(raw)==self.expected_json[name], 'Unexpected source JSON: '+name)
            if name in EXTRAS-{'sample.json'}:
                require(name in self.expected_bindings or name in self.expected_json,
                        'Unverified raw extra requires capture binding: '+name)
            return raw
        entry=self.manifest['files'].get(name)
        require(entry is not None, 'Logical file omitted or unavailable: '+name)
        stored=read_bounded(safe_path(self.root,entry['stored_path']))
        require(len(stored)==entry['stored_bytes'] and digest(stored)==entry['stored_sha256'],
                'Corrupt stored native file')
        raw=decode(stored) if entry['encoding']=='native_npy_byteplanes_zstd' else stored
        require(len(raw)==entry['logical_bytes'] and digest(raw)==entry['logical_sha256'],
                'Corrupt logical native file')
        if name in self.expected_bindings:
            require(digest(raw)==self.expected_bindings[name], 'Unexpected source binding: '+name)
        if name in self.expected_json:
            require(json.loads(raw)==self.expected_json[name], 'Unexpected source JSON: '+name)
        return raw

    def array(self,name):
        require(name in ARRAYS, 'Unknown native array')
        return _native_npy(self.read(name))

    def image(self,name):
        with Image.open(io.BytesIO(self.read(name))) as image:
            return np.asarray(image).copy()

    def json(self,name):
        return json.loads(self.read(name))

    def verify_all(self):
        names=self.manifest['files'] if self.manifest else self.metadata['files']
        for name in names:self.read(name)
        self._verify_extras()
        return True

def pack_sample(source, destination, *, sample_sha256, label_sha256, query_trace=None):
    """Create an auditable compact copy, leaving original captures untouched."""
    source,destination=Path(source).resolve(),Path(destination).resolve()
    require(source!=destination and not destination.is_relative_to(source)
            and not source.is_relative_to(destination), 'Separate nonnested destination required')
    require(not destination.exists(), 'Create-only destination required')
    reader=SampleReader(source,expected_bindings={'sample.json':sample_sha256,
        'supervision/label.json':label_sha256},expected_json=(
            {'supervision/query_trace.json':query_trace} if query_trace is not None else {}))
    require(reader.manifest is None, 'Repacking is not new data')
    require(digest(reader.read('sample.json'))==sample_sha256, 'Unexpected source sample')
    require(digest(reader.read('supervision/label.json'))==label_sha256, 'Unexpected source label')
    reader.verify_all()
    # Write manifest last: any interrupted copy remains visibly incomplete.
    selected=[name for name,v in reader.metadata['files'].items() if v['role']!='review_only']
    selected += ['sample.json','supervision/label.json']
    if (source/'supervision/query_trace.json').is_file():selected.append('supervision/query_trace.json')
    require(ARRAYS<=set(selected), 'Complete native depth and identity buffers required')
    destination.mkdir(parents=True,exist_ok=False)
    entries={}
    for name in sorted(set(selected)):
        raw=reader.read(name)
        packed=encode(raw) if name in ARRAYS else raw
        if name in ARRAYS:require(decode(packed)==raw,'Non-exact native roundtrip')
        stored_name='payload/'+name+('.ghn' if name in ARRAYS else '')
        path=safe_path(destination,stored_name);path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(packed)
        require(reader.read(name)==raw,'Source changed during packing')
        entries[name]=dict(stored_path=stored_name,encoding='native_npy_byteplanes_zstd' if name in ARRAYS else 'identity',
            logical_bytes=len(raw),stored_bytes=len(packed),logical_sha256=digest(raw),stored_sha256=digest(packed))
    omitted={name:v['sha256'] for name,v in reader.metadata['files'].items() if v['role']=='review_only'}
    manifest=dict(schema=SCHEMA,files=entries,omitted_review_files=omitted,
        source_sample_sha256=sample_sha256,source_label_sha256=label_sha256,
        source_query_trace=query_trace,
        lossy=False,depth_recomputed=False,source_removed=False,training_approved=False,
        rgb_policy='unchanged_original_png',depth_policy='exact_native_optical_z_npy_bytes',
        review_policy='omitted_review_only_pngs_regenerable_not_model_input',
        source_logical_bytes=sum(e['logical_bytes'] for e in entries.values()),
        stored_payload_bytes=sum(e['stored_bytes'] for e in entries.values()))
    # Check original bindings again after all I/O.
    reader.verify_all()
    require(digest(reader.read('sample.json'))==sample_sha256 and
            digest(reader.read('supervision/label.json'))==label_sha256,'Source changed at publish')
    with (destination/'bundle.json').open('x',encoding='utf-8') as stream:
        json.dump(manifest,stream,indent=2,allow_nan=False)
    SampleReader(destination).verify_all()
    return manifest
