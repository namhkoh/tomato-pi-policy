"""Finite read cache around the unchanged annotation9 implementation."""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import os,json,time
from . import native848_pilot_annotation_v9 as annotation
from . import native848_unique9mm_batch_validation_v1 as validation

ANNOTATION_SHA='2ce5ee7aafb5d2577455980113fda0aeabb9ab71a2d100eec83819b866267152'
VALIDATION_SHA='6e54fa3c8d3c5a38b591e8dfcd7538052e32d259c00196c8a164342f5d44bbdd'
SCHEMA='greenhouse.native848_cached_annotation_operation.v2'

class Session(validation.Session):
    """Resolve each path spelling once, then recheck every alias at closure."""
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.path_aliases={}
        self.path_key_calls=0
    def _key(self,path):
        self.path_key_calls+=1
        raw=os.fspath(path)
        validation.require(isinstance(raw,str),'Text source path required')
        if not os.path.isabs(raw):raw=os.path.join(os.getcwd(),raw)
        if raw not in self.path_aliases:self.path_aliases[raw]=str(Path(raw).resolve())
        return self.path_aliases[raw]
    def close(self):
        for original,expected in self.path_aliases.items():
            validation.require(str(Path(original).resolve())==expected,'Source alias changed during finite operation')
        super().close()
    def report(self):
        result=super().report()
        result.update(path_key_calls=self.path_key_calls,resolved_aliases=len(self.path_aliases),
            source_aliases_rechecked_at_close=True)
        return result

def run(trial,result_sha256,output,*,parent_metadata=None,parent_recovery=None):
    output=Path(output).resolve()
    validation.require(not output.exists(),'Create-only cached annotation output required')
    session=Session();terminal=False
    real_save=annotation.save_json
    session.authenticate(annotation.__file__,ANNOTATION_SHA)
    session.authenticate(validation.__file__,VALIDATION_SHA)
    wrapper_sha=session.digest(__file__)
    terminal_path=output/'result.json'
    def save(path,value):
        nonlocal terminal
        if Path(path).resolve()==terminal_path:
            session.close()
            receipt=session.report()
            receipt.update(schema=SCHEMA,annotation_implementation=dict(path=str(Path(annotation.__file__).resolve()),sha256=ANNOTATION_SHA),
                wrapper=dict(path=str(Path(__file__).resolve()),sha256=wrapper_sha),
                unchanged_annotation_predicates=True,cache_scope='one_complete_annotation_operation',native_control_performed=False)
            real_save(output/'validation_session.json',receipt)
            value=dict(value,finite_validation=dict(path=str(output/'validation_session.json'),sha256=validation._REAL_HASH(output/'validation_session.json')))
            real_save(path,value);terminal=True
        else:real_save(path,value)
    def read_json(path):return json.loads(session.bytes(path).decode('utf-8-sig'))
    with validation.Adapted(session),ExitStack() as stack:
        for name,value in [('sha256',session.digest),('verify_bindings',session.verify_bindings),
                           ('read_json',read_json),('Image',validation.ImageReads(session)),
                           ('np',validation.NumpyReads(session)),('save_json',save)]:
            stack.enter_context(patch.object(annotation,name,value))
        result=annotation.run(trial,result_sha256,output,parent_metadata=parent_metadata,parent_recovery=parent_recovery)
    validation.require(terminal and session.closed,'Annotation result did not close authenticated source union')
    return dict(result,finite_validation=dict(path=str(output/'validation_session.json'),sha256=validation._REAL_HASH(output/'validation_session.json')))
