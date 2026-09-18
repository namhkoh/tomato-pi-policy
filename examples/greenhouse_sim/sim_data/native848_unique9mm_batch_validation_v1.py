"""Finite CPU validation cache around unchanged unique9mm predicates.

Use only in a dedicated single-threaded process. Each operation starts empty,
authenticates actual file bytes, supplies immutable read snapshots, and rehashes
the complete file union before publishing its terminal artifact or returning.
"""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import copy
import hashlib
import io
import json
import threading
import time
import numpy as np
from PIL import Image
from . import native848_unique_petiole_dataset_v1 as dataset1
from . import native848_unique_petiole_dataset_v2 as dataset2
from . import native848_unique9mm_group_review_v1 as groups
from . import native848_unique9mm_ambiguity_v1 as ambiguity
from . import native848_data_roots_v1 as roots

SCHEMA = 'greenhouse.native848_unique9mm_finite_validation_session.v1'
PINS = {
    dataset1.__file__: '35575a0edeea122ee93b1650da43cf2e258d6c71fa01c3960c04f6d3db6755f2',
    dataset2.__file__: 'b83c76cacaf7eb1a6a705b6a1c34d7975482e405e3243be691e05165f648505f',
    groups.__file__: '59e5bc30cb5125b51a0cd06e1498415d2f7b8144b19aac3da8320640b6c3af81',
    ambiguity.__file__: 'c42bca181ca07af4446c0ff5c24f14126573bc39dc8eeeb609a8e835fd9ac2fe',
}
require = dataset1.require
_ACTIVE = False
_REAL_HASH = dataset1.digest
_REAL_WRITE = dataset1.write
_REAL_IMAGE_OPEN = Image.open
_REAL_NP_LOAD = np.load


class SnapshotPath(type(Path())):
    def __new__(cls, path, session):
        obj = super().__new__(cls,path)
        obj._session = session
        return obj

    def __init__(self,path,session):
        super().__init__(path)
        self._session = session

    def read_bytes(self):
        return self._session.bytes(self)

    def read_text(self,encoding=None,errors=None):
        return self.read_bytes().decode(encoding or 'utf-8',errors or 'strict')

    def open(self,mode='r',buffering=-1,encoding=None,errors=None,newline=None):
        require(mode in ('r','rb','rt'), 'Verified source snapshots are read-only')
        raw = io.BytesIO(self.read_bytes())
        return raw if 'b' in mode else io.TextIOWrapper(raw,encoding=encoding or 'utf-8',errors=errors or 'strict',newline=newline)


class ArrayArchive:
    def __init__(self,arrays):
        self._arrays = arrays
        self.files = list(arrays)

    def __getitem__(self,key):
        return self._arrays[key].copy()

    def __enter__(self): return self
    def __exit__(self,*args): return False
    def close(self): pass


class NumpyReads:
    def __init__(self,session): self.session = session
    def __getattr__(self,name): return getattr(np,name)
    def load(self,path,*args,**kwargs):
        require(not args and kwargs == {'allow_pickle':False}, 'Exact safe NumPy read contract required')
        return self.session.arrays(path)


class ImageReads:
    def __init__(self,session): self.session = session
    def __getattr__(self,name): return getattr(Image,name)
    def open(self,path,*args,**kwargs):
        require(not args and not kwargs, 'Exact native image read contract required')
        return self.session.image(path)


class Session:
    def __init__(self,*,snapshot_budget_bytes=2*2**30):
        self.hashes = {}
        self.expected = {}
        self.byte_snapshots = {}
        self.decoded = {}
        self.snapshot_budget_bytes = snapshot_budget_bytes
        self.snapshot_bytes = 0
        self.snapshot_read_bytes = 0
        self.snapshot_hash_seconds = 0.
        self.digest_calls = 0
        self.hash_reads = 0
        self.hash_bytes = 0
        self.hash_seconds = 0.
        self.closing_reads = 0
        self.closing_bytes = 0
        self.closing_seconds = 0.
        self.closed = False
        self.started = time.perf_counter()

    def _key(self,path): return str(Path(path).resolve())

    def _actual(self,path,closing=False):
        start = time.perf_counter()
        h = hashlib.sha256(); size = 0
        with Path(path).open('rb') as stream:
            for chunk in iter(lambda:stream.read(2**20),b''):
                h.update(chunk); size += len(chunk)
        elapsed = time.perf_counter()-start
        if closing:
            self.closing_reads += 1; self.closing_bytes += size; self.closing_seconds += elapsed
        else:
            self.hash_reads += 1; self.hash_bytes += size; self.hash_seconds += elapsed
        return h.hexdigest(),size

    def digest(self,path):
        key = self._key(path); self.digest_calls += 1
        if key not in self.hashes:
            require(not self.closed, 'Cannot expand validation union after closing rehash')
            self.hashes[key] = self._actual(key)
        return self.hashes[key][0]

    def authenticate(self,path,expected):
        key = self._key(path)
        require(isinstance(expected,str) and len(expected)==64, 'Expected SHA256 required')
        require(key not in self.expected or self.expected[key] == expected, 'Conflicting pins for one source path')
        self.expected[key] = expected
        require(self.digest(key) == expected, 'Changed pinned file bytes')
        return key

    def read_pin(self,spec):
        require(set(spec) == {'path','sha256'}, 'Exact file pin required')
        path = roots.resolve_evidence(spec['path'])
        require(path.is_file(), 'Pinned source missing')
        key = self.authenticate(path,spec['sha256'])
        return SnapshotPath(key,self)

    def verify_bindings(self,bindings):
        require(bool(bindings), 'Missing immutable source bindings')
        for path,expected in bindings.items(): self.authenticate(path,expected)

    def _reserve(self,size):
        require(self.snapshot_bytes+size <= self.snapshot_budget_bytes, 'Finite snapshot memory budget exceeded; use a smaller batch')
        self.snapshot_bytes += size

    def bytes(self,path):
        key = self._key(path)
        if key not in self.byte_snapshots:
            expected = self.digest(key)
            payload = Path(key).read_bytes()
            started = time.perf_counter(); actual = hashlib.sha256(payload).hexdigest()
            self.snapshot_hash_seconds += time.perf_counter()-started; self.snapshot_read_bytes += len(payload)
            require(actual == expected, 'File changed between authentication and snapshot read')
            self._reserve(len(payload)); self.byte_snapshots[key] = payload
        return self.byte_snapshots[key]

    def image(self,path):
        key = ('image',self._key(path))
        if key not in self.decoded:
            with _REAL_IMAGE_OPEN(io.BytesIO(self.bytes(path))) as image:
                image.load(); saved = image.copy()
            self._reserve(len(saved.tobytes())); self.decoded[key] = saved
        return self.decoded[key].copy()

    def arrays(self,path):
        key = ('arrays',self._key(path))
        if key not in self.decoded:
            value = _REAL_NP_LOAD(io.BytesIO(self.bytes(path)),allow_pickle=False)
            if isinstance(value,np.ndarray):
                saved = value.copy(); saved.setflags(write=False); self._reserve(saved.nbytes)
            else:
                with value:
                    saved = {name:value[name].copy() for name in value.files}
                for array in saved.values(): array.setflags(write=False); self._reserve(array.nbytes)
            self.decoded[key] = saved
        saved = self.decoded[key]
        return saved.copy() if isinstance(saved,np.ndarray) else ArrayArchive(saved)

    def close(self):
        require(not self.closed, 'Validation session already closed')
        for path,(expected,_) in self.hashes.items():
            actual,_ = self._actual(path,closing=True)
            require(actual == expected, 'Distinct source union changed before final closure')
        self.closed = True

    def report(self):
        require(self.closed, 'Only closed validation sessions can emit completion evidence')
        return dict(schema=SCHEMA,implementation=dict(path=str(Path(__file__).resolve()),sha256=_REAL_HASH(__file__)),
            source_bindings={p:h for p,(h,_) in self.hashes.items()},expected_pins=self.expected,
            unique_files=len(self.hashes),unique_bytes=sum(size for _,size in self.hashes.values()),
            requested_digest_calls=self.digest_calls,initial_stream_hash_reads=self.hash_reads,
            initial_stream_hash_bytes=self.hash_bytes,initial_stream_hash_seconds=self.hash_seconds,
            closing_stream_hash_reads=self.closing_reads,closing_stream_hash_bytes=self.closing_bytes,
            closing_stream_hash_seconds=self.closing_seconds,snapshot_bytes=self.snapshot_bytes,
            snapshot_read_bytes=self.snapshot_read_bytes,snapshot_hash_seconds=self.snapshot_hash_seconds,
            elapsed_seconds=time.perf_counter()-self.started,full_distinct_union_rehashed=True,
            unchanged_per_frame_predicates=True,mtime_only_checks=False,cross_operation_cache=False,
            native_control_performed=False,training_started=False)


class Adapted:
    """Scoped module-local I/O adapters; no frozen source files are changed."""
    def __init__(self,session): self.session = session; self.stack = ExitStack()
    def __enter__(self):
        global _ACTIVE
        require(not _ACTIVE and threading.active_count() == 1, 'Dedicated single-threaded finite operation required')
        _ACTIVE = True
        try:
            for path,expected in PINS.items(): self.session.authenticate(path,expected)
            self.session.digest(__file__)
            for module in (dataset1,dataset2):
                self.stack.enter_context(patch.object(module,'digest',self.session.digest))
                self.stack.enter_context(patch.object(module,'read_pin',self.session.read_pin))
                self.stack.enter_context(patch.object(module,'np',NumpyReads(self.session)))
                self.stack.enter_context(patch.object(module,'Image',ImageReads(self.session)))
            self.stack.enter_context(patch.object(groups,'Image',ImageReads(self.session)))
            self.stack.enter_context(patch.object(ambiguity,'verify_bindings',self.session.verify_bindings))
            return self.session
        except BaseException:
            self.stack.close(); _ACTIVE = False; raise
    def __exit__(self,*args):
        global _ACTIVE
        self.stack.close(); _ACTIVE = False
        return False


def prepare(frames,output):
    """Produce the identical frozen sample population plus a cache receipt."""
    require(0 < len(frames) <= 512, 'Finite bounded frame batch required')
    output = roots.diagnostic(output)
    session = Session(); terminal_written = False
    def write(path,value):
        nonlocal terminal_written
        if Path(path).resolve() == output/'visual_packet.json':
            session.close()
            _REAL_WRITE(output/'validation_session.json',session.report())
            _REAL_WRITE(path,value); terminal_written = True
        else: _REAL_WRITE(path,value)
    with Adapted(session), patch.object(dataset1,'write',write):
        # The frozen prepare pins its newly published packet after the last write.
        # Hash that terminal directly; it is output, never an unverified input.
        original_digest = session.digest
        def digest(path):
            if terminal_written and Path(path).resolve() == output/'visual_packet.json': return _REAL_HASH(path)
            return original_digest(path)
        with patch.object(dataset1,'digest',digest): value = groups.prepare(copy.deepcopy(frames),output)
    require(terminal_written and session.closed, 'Packet terminal did not close its source union')
    return dict(**value,validation_session=dict(path=str(output/'validation_session.json'),sha256=_REAL_HASH(output/'validation_session.json')))


def validate_group_review(inventory_path,inventory_sha256,review_path,review_sha256):
    session = Session()
    with Adapted(session):
        result = groups.validate_group_review(inventory_path,inventory_sha256,review_path,review_sha256)
        session.close()
    return dict(**result,finite_validation=session.report())


def materialize_frames(frames,output,*,root_authorization):
    """Keep the frozen finite authorization and add this exact wrapper pin."""
    require(0 < len(frames) <= 512, 'Finite bounded frame batch required')
    output = roots.checkpoint_output(output)
    session = Session(); terminal_written = False
    def write(path,value):
        nonlocal terminal_written
        if Path(path).resolve() == output/'result.json':
            session.close()
            _REAL_WRITE(output/'validation_session.json',session.report())
            value = dict(value,validation_session=dict(path='validation_session.json',sha256=_REAL_HASH(output/'validation_session.json')))
            _REAL_WRITE(path,value); terminal_written = True
        else: _REAL_WRITE(path,value)
    with Adapted(session), patch.object(dataset2,'write',write):
        authorization = json.loads(session.read_pin(root_authorization).read_text())
        require(authorization.get('validation_cache_sha256') == session.digest(__file__), 'Exact additional cache implementation authorization required')
        result = dataset2.materialize_frames(copy.deepcopy(frames),output,root_authorization=root_authorization)
    require(terminal_written and session.closed, 'Dataset terminal did not close its source union')
    return result
