"""Finite read cache around the unchanged annotation13 predicates, at most16 frames."""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import os,json,time
from . import native848_pilot_annotation_v13 as annotation
from . import native848_unique9mm_batch_validation_v1 as validation

ANNOTATION_SHA='4a48ff331a72e0d34c9989c67ddebfa2e44ea97eaba250ce5e41a06c9a796625'
CACHE2_SHA='b9fabd98df0671bb882d99a737e7a4b920fa50a59b4412022d3dbe5d211aa48d'
VALIDATION_SHA='6e54fa3c8d3c5a38b591e8dfcd7538052e32d259c00196c8a164342f5d44bbdd'
SCHEMA='greenhouse.native848_cached_annotation_operation.v6'

from . import native848_annotation_cache_v2 as predecessor
Session=predecessor.Session

def run(trial,result_sha256,output,*,parent_metadata=None,parent_recovery=None,start_index=0,frame_count=16):
    validation.require(type(start_index) is int and start_index>=0 and type(frame_count) is int and 1<=frame_count<=16,'Finite annotation chunk of at most16 required')
    output=Path(output).resolve()
    validation.require(not output.exists(),'Create-only cached annotation output required')
    session=Session();terminal=False
    real_save=annotation.save_json
    session.authenticate(annotation.__file__,ANNOTATION_SHA)
    session.authenticate(validation.__file__,VALIDATION_SHA)
    session.authenticate(predecessor.__file__,CACHE2_SHA)
    session.authenticate(annotation.full_coverage.__file__,annotation.COVERAGE_SHA256)
    session.authenticate(annotation.ambiguity.__file__,annotation.AMBIGUITY_SHA256)
    wrapper_sha=session.digest(__file__)
    terminal_path=output/'result.json'
    def save(path,value):
        nonlocal terminal
        if Path(path).resolve()==terminal_path:
            validation.require(value['completed_whole_capture_authenticated'] is True
                and value['requested_start_index']==start_index and value['requested_frame_count']==frame_count
                and value['evaluated_indices']==annotation.finite_indices(value['total_capture_frames'],start_index,frame_count)
                and value['frames_evaluated']==len(value['records'])==len(value['evaluated_indices']),
                'Exact fully authenticated finite annotation chunk required')
            session.close()
            receipt=session.report()
            receipt.update(schema=SCHEMA,annotation_implementation=dict(path=str(Path(annotation.__file__).resolve()),sha256=ANNOTATION_SHA),
                wrapper=dict(path=str(Path(__file__).resolve()),sha256=wrapper_sha),
                unchanged_annotation_predicates=True,cache_scope='one_finite_annotation_chunk_of_completed_capture',
                requested_start_index=start_index,requested_frame_count=frame_count,evaluated_indices=value['evaluated_indices'],
                total_capture_frames=value['total_capture_frames'],snapshot_budget_bytes=session.snapshot_budget_bytes,native_control_performed=False)
            real_save(output/'validation_session.json',receipt)
            value=dict(value,finite_validation=dict(path=str(output/'validation_session.json'),sha256=validation._REAL_HASH(output/'validation_session.json')))
            real_save(path,value);terminal=True
        else:real_save(path,value)
    def read_json(path):return json.loads(session.bytes(path).decode('utf-8-sig'))
    with validation.Adapted(session),ExitStack() as stack:
        stack.enter_context(patch.object(annotation.ambiguity,'verify_bindings',session.verify_bindings))
        stack.enter_context(patch.object(annotation.full_coverage,'verify_bindings',session.verify_bindings))
        stack.enter_context(patch.object(annotation.full_coverage,'sha256',session.digest))
        for name,value in [('sha256',session.digest),('verify_bindings',session.verify_bindings),
                           ('read_json',read_json),('Image',validation.ImageReads(session)),
                           ('np',validation.NumpyReads(session)),('save_json',save)]:
            stack.enter_context(patch.object(annotation,name,value))
        result=annotation.run(trial,result_sha256,output,parent_metadata=parent_metadata,parent_recovery=parent_recovery,start_index=start_index,frame_count=frame_count)
    validation.require(terminal and session.closed,'Annotation result did not close authenticated source union')
    return dict(result,finite_validation=dict(path=str(output/'validation_session.json'),sha256=validation._REAL_HASH(output/'validation_session.json')))
