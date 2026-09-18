"""Finite immutable I/O cache for fully labeled population dataset and group review.

The same initial and closing full hashes and original per-frame checks apply.
Dedicated process; up to16 frames per operation, no native control or training.
"""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import copy,json,threading
from . import native848_unique9mm_batch_validation_v1 as io
from . import native848_annotation_cache_v1 as fast
from . import native848_fully_labeled_frame_validation_v5 as dataset1
from . import native848_unique_petiole_dataset_v7 as dataset2
from . import native848_unique9mm_group_review_v6 as groups
from . import native848_unique9mm_ambiguity_v2 as ambiguity
from . import native848_data_roots_v1 as roots
require=dataset1.require
_REAL_HASH=dataset1.digest
_REAL_WRITE=dataset1.write
_ACTIVE=False
PINS={'native848_fully_labeled_frame_validation_v5.py': 'a51ec6745ede143c7b14359f45a38ef66d54e3ee61e691faaea6c889b9954fc6', 'native848_unique_petiole_dataset_v7.py': '916b80336aafe3a26e0b875b3fbae06065ae24efda73b310d8115f73cdda392d', 'native848_unique9mm_group_review_v6.py': 'beef8479996544431cefd765bbe8ab0b85ab74134b0f0c0c03de5588a3163cc2', 'native848_unique9mm_ambiguity_v2.py': '8152cd4194b2544badf3886f35be7e9bff2a68b4df65aa9fe620108408a99ec5'}
AMBIGUITY_DATASET_SHA='35575a0edeea122ee93b1650da43cf2e258d6c71fa01c3960c04f6d3db6755f2'
IO_SHA='6e54fa3c8d3c5a38b591e8dfcd7538052e32d259c00196c8a164342f5d44bbdd'
FAST_SHA='45d597a50775c61abd51507b5bbc6ba45eaa7064a7dfaa73ffecb39d29349174'

def session():
    value=fast.Session()
    value.authenticate(io.__file__,IO_SHA)
    value.authenticate(fast.__file__,FAST_SHA)
    value.digest(__file__)
    return value

def receipt(value):
    return dict(value.report(),fully_labeled_validation=True,
        wrapper=dict(path=str(Path(__file__).resolve()),sha256=_REAL_HASH(__file__)),
        path_cache_implementation=dict(path=str(Path(fast.__file__).resolve()),sha256=FAST_SHA))

class Adapted:
    def __init__(self,value):self.session=value;self.stack=ExitStack()
    def __enter__(self):
        global _ACTIVE
        require(not _ACTIVE and threading.active_count()==1,'Dedicated single-threaded finite operation required')
        _ACTIVE=True
        try:
            for module in (dataset1,dataset2,groups,ambiguity):
                self.session.authenticate(module.__file__,PINS[Path(module.__file__).name])
            self.session.authenticate(ambiguity.dataset.__file__,AMBIGUITY_DATASET_SHA)
            for module in (dataset1,dataset2,ambiguity.dataset):
                for key,value in (('digest',self.session.digest),('read_pin',self.session.read_pin),
                    ('np',io.NumpyReads(self.session)),('Image',io.ImageReads(self.session))):
                    self.stack.enter_context(patch.object(module,key,value))
            self.stack.enter_context(patch.object(groups,'Image',io.ImageReads(self.session)))
            self.stack.enter_context(patch.object(ambiguity,'verify_bindings',self.session.verify_bindings))
            return self.session
        except BaseException:
            self.stack.close();_ACTIVE=False;raise
    def __exit__(self,*args):
        global _ACTIVE
        self.stack.close();_ACTIVE=False
        return False

def prepare(frames,output):
    require(0<len(frames)<=16,'Finite bounded full-population frame batch required')
    output=roots.diagnostic(output);value=session();terminal=False
    terminal_path=output/'visual_packet.json'
    def write(path,data):
        nonlocal terminal
        if Path(path).resolve()==terminal_path:
            value.close()
            _REAL_WRITE(output/'validation_session.json',receipt(value))
            _REAL_WRITE(path,data);terminal=True
        else:_REAL_WRITE(path,data)
    with Adapted(value),patch.object(dataset1,'write',write):
        original_digest=value.digest
        def digest(path):
            if terminal and Path(path).resolve()==terminal_path:return _REAL_HASH(path)
            return original_digest(path)
        with patch.object(dataset1,'digest',digest):result=groups.prepare(copy.deepcopy(frames),output)
    require(terminal and value.closed,'Packet terminal did not close its source union')
    return dict(**result,validation_session=dict(path=str(output/'validation_session.json'),sha256=_REAL_HASH(output/'validation_session.json')))

def validate_group_review(inventory_path,inventory_sha256,review_path,review_sha256):
    value=session()
    with Adapted(value):
        result=groups.validate_group_review(inventory_path,inventory_sha256,review_path,review_sha256)
        value.close()
    return dict(**result,finite_validation=receipt(value))

def materialize_frames(frames,output,*,root_authorization):
    require(0<len(frames)<=16,'Finite bounded full-population frame batch required')
    output=roots.checkpoint_output(output);value=session();terminal=False
    def write(path,data):
        nonlocal terminal
        if Path(path).resolve()==output/'result.json':
            value.close()
            _REAL_WRITE(output/'validation_session.json',receipt(value))
            data=dict(data,validation_session=dict(path='validation_session.json',sha256=_REAL_HASH(output/'validation_session.json')))
            _REAL_WRITE(path,data);terminal=True
        else:_REAL_WRITE(path,data)
    with Adapted(value),patch.object(dataset2,'write',write):
        authorization=json.loads(value.read_pin(root_authorization).read_text())
        require(authorization.get('validation_cache_sha256')==value.digest(__file__),'Exact additional cache authorization required')
        result=dataset2.materialize_frames(copy.deepcopy(frames),output,root_authorization=root_authorization)
    require(terminal and value.closed,'Dataset terminal did not close its source union')
    return result
