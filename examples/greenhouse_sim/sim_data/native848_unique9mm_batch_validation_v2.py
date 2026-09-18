"""Faster path canonicalization around unchanged finite packet/export validation.

Dedicated single-threaded operation; initial and closing full source-union hashes,
immutable read snapshots, unchanged per-frame predicates, explicit root export auth.
"""
from pathlib import Path
from unittest.mock import patch
import copy,json
from . import native848_unique9mm_batch_validation_v1 as base
from . import native848_annotation_cache_v1 as fast

BASE_SHA='6e54fa3c8d3c5a38b591e8dfcd7538052e32d259c00196c8a164342f5d44bbdd'
FAST_SHA='45d597a50775c61abd51507b5bbc6ba45eaa7064a7dfaa73ffecb39d29349174'

def session():
    value=fast.Session()
    value.authenticate(base.__file__,BASE_SHA)
    value.authenticate(fast.__file__,FAST_SHA)
    value.digest(__file__)
    return value

def receipt(value):
    return dict(value.report(),wrapper=dict(path=str(Path(__file__).resolve()),sha256=base._REAL_HASH(__file__)),
        path_cache_implementation=dict(path=str(Path(fast.__file__).resolve()),sha256=FAST_SHA))

def prepare(frames,output):
    base.require(0<len(frames)<=512,'Finite bounded frame batch required')
    output=base.roots.diagnostic(output)
    value=session();terminal=False
    terminal_path=output/'visual_packet.json'
    def write(path,data):
        nonlocal terminal
        if Path(path).resolve()==terminal_path:
            value.close()
            base._REAL_WRITE(output/'validation_session.json',receipt(value))
            base._REAL_WRITE(path,data);terminal=True
        else:base._REAL_WRITE(path,data)
    with base.Adapted(value),patch.object(base.dataset1,'write',write):
        original_digest=value.digest
        def digest(path):
            if terminal and Path(path).resolve()==terminal_path:return base._REAL_HASH(path)
            return original_digest(path)
        with patch.object(base.dataset1,'digest',digest):result=base.groups.prepare(copy.deepcopy(frames),output)
    base.require(terminal and value.closed,'Packet terminal did not close its source union')
    return dict(**result,validation_session=dict(path=str(output/'validation_session.json'),sha256=base._REAL_HASH(output/'validation_session.json')))

def validate_group_review(inventory_path,inventory_sha256,review_path,review_sha256):
    value=session()
    with base.Adapted(value):
        result=base.groups.validate_group_review(inventory_path,inventory_sha256,review_path,review_sha256)
        value.close()
    return dict(**result,finite_validation=receipt(value))

def materialize_frames(frames,output,*,root_authorization):
    base.require(0<len(frames)<=512,'Finite bounded frame batch required')
    output=base.roots.checkpoint_output(output)
    value=session();terminal=False
    def write(path,data):
        nonlocal terminal
        if Path(path).resolve()==output/'result.json':
            value.close()
            base._REAL_WRITE(output/'validation_session.json',receipt(value))
            data=dict(data,validation_session=dict(path='validation_session.json',sha256=base._REAL_HASH(output/'validation_session.json')))
            base._REAL_WRITE(path,data);terminal=True
        else:base._REAL_WRITE(path,data)
    with base.Adapted(value),patch.object(base.dataset2,'write',write):
        authorization=json.loads(value.read_pin(root_authorization).read_text())
        base.require(authorization.get('validation_cache_sha256')==value.digest(__file__),'Exact additional cache implementation authorization required')
        result=base.dataset2.materialize_frames(copy.deepcopy(frames),output,root_authorization=root_authorization)
    base.require(terminal and value.closed,'Dataset terminal did not close its source union')
    return result
