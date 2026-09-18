"""Hash-only finite cache around unchanged completed-scene sweep consumers.

No labels, geometry, visibility, workspace, or ambiguity predicates change.
Sources are hashed on first use and the complete union is actually rehashed
before a cache receipt and terminal result are published. Images/arrays/JSON
remain consumer-owned and are not retained as per-frame cache snapshots.
Use only in a dedicated single-threaded CPU process after native owner closure.
"""
from pathlib import Path
from contextlib import ExitStack, contextmanager
import argparse,json
from .dataset_review import require,read_json
from .depth_preview import sha256 as real_hash
from .native848_bulk_io_v1 import save_json as real_save
from . import native848_annotation_cache_v9 as cache
from . import native848_unique9mm_batch_validation_v1 as validation
from . import native848_scene_sweep_9mm_v1 as ordinary
from . import native848_scene_sweep_raw_9mm_v1 as raw
from . import native848_unique9mm_ambiguity_v3 as ambiguity
from . import native848_fully_labeled_coverage_v3 as coverage

SCHEMA='greenhouse.native848_scene_sweep_cached_annotation.v1'
MAX_FRAMES=512
MAX_INITIAL_SOURCE_FILES=50000
SNAPSHOT_BUDGET_BYTES=64*2**20
PINS={ordinary.__file__:'96adb168d8dbacb7240888e3e8e4875f0472592f0dd2004cdd0063141d567721',
      raw.__file__:'eea45d63bbbcc72d61c723fa608a4b648ab4654577b9ad39c2b393276902f0ac',
      cache.__file__:'52455e9ecb12bb2d8d62d47419be4526b8993706f3c0c68e6dcdd5f955f0d5b5',
      cache.predecessor.__file__:'b9fabd98df0671bb882d99a737e7a4b920fa50a59b4412022d3dbe5d211aa48d',
      validation.__file__:'6e54fa3c8d3c5a38b591e8dfcd7538052e32d259c00196c8a164342f5d44bbdd'}


def pin(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=real_hash(path))


def new_session():
    session=cache.Session(snapshot_budget_bytes=SNAPSHOT_BUDGET_BYTES)
    for path,h in PINS.items():session.authenticate(path,h)
    for path,h in ordinary.FROZEN.items():session.authenticate(path,h)
    session.digest(__file__)
    return session


@contextmanager
def hash_scope(session,consumer=None):
    """Only scoped hash/verification aliases change; numeric functions stay exact."""
    from unittest.mock import patch
    with validation.Adapted(session),ExitStack() as stack:
        stack.enter_context(patch.object(ambiguity.dataset,'digest',session.digest))
        stack.enter_context(patch.object(ambiguity,'verify_bindings',session.verify_bindings))
        stack.enter_context(patch.object(coverage,'verify_bindings',session.verify_bindings))
        stack.enter_context(patch.object(coverage,'sha256',session.digest))
        if consumer is not None:
            stack.enter_context(patch.object(consumer,'sha256',session.digest))
            stack.enter_context(patch.object(consumer,'verify_bindings',session.verify_bindings))
        yield session


def authenticate_completed_size(capture,session):
    """Bound the operation before consumer execution; owner authority is replayed."""
    complete_path=capture.parent/'owner_complete.json'
    session.digest(complete_path);complete=read_json(complete_path)
    result_pin=complete['result'];session.authenticate(result_pin['path'],result_pin['sha256'])
    result=read_json(result_pin['path'])
    require(Path(result_pin['path']).resolve()==capture/'result.json'
        and complete['frames']==len(result['frames']) and 1<=len(result['frames'])<=MAX_FRAMES,
        'One finite completed capture of1..512 frames required')
    context_pin=result['context'];session.authenticate(context_pin['path'],context_pin['sha256'])
    context=read_json(context_pin['path'])
    require(len(context['source_bindings'])<=MAX_INITIAL_SOURCE_FILES,'Bounded immutable source union required')
    return len(result['frames'])


def close_and_publish(session,value,output,consumer,mode,expected_count):
    """A closing source change prevents both receipt and final-result publication."""
    require(value['frames_evaluated']==expected_count and len(value['records'])==expected_count
        and value['completed_whole_capture_authenticated'] is True
        and value['all_committed_frames_evaluated'] is True,'Exact completed frame schedule required')
    session.verify_bindings(value['source_bindings'])
    session.close()
    receipt=session.report()
    require(receipt['snapshot_bytes']==0,'Hash-only wrapper must not retain per-frame snapshots')
    receipt.update(schema=SCHEMA,wrapper=pin(__file__),consumer=pin(consumer.__file__),mode=mode,
        frame_count=expected_count,maximum_frames=MAX_FRAMES,
        cache_scope='hashes_only_one_completed_owner_operation',
        numerical_predicates_unchanged=True,mutable_capture_or_label_snapshots_cached=False,
        initial_actual_hashes_and_full_closing_rehash_required=True)
    real_save(output/'hash_validation_session.json',receipt)
    result=dict(value,finite_validation=pin(output/'hash_validation_session.json'))
    real_save(output/'result.json',result)
    return result


def run(capture,output,*,mode):
    require(mode in ('ordinary','raw'),'Explicit completed consumer type required')
    capture=Path(capture).resolve();output=Path(output).resolve()
    require(capture.name=='capture' and capture.is_dir() and not output.exists()
        and not output.is_relative_to(capture.parent),'Fresh independent annotation output required')
    consumer=ordinary if mode=='ordinary' else raw
    session=new_session();expected=authenticate_completed_size(capture,session)
    terminal=None;original_save=consumer.save_json
    def save(path,value):
        nonlocal terminal
        if Path(path).resolve()==output/'result.json':
            require(terminal is None,'Only one terminal result permitted')
            terminal=close_and_publish(session,value,output,consumer,mode,expected)
        else:original_save(path,value)
    from unittest.mock import patch
    with hash_scope(session,consumer),patch.object(consumer,'save_json',save):
        consumer.evaluate_capture(capture,output)
    require(terminal is not None and session.closed,'Consumer did not close authenticated hash union')
    return terminal


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mode',choices=('ordinary','raw'),required=True)
    args=parser.parse_args();result=run(args.capture,args.output,mode=args.mode)
    print(json.dumps(dict(frames=result['frames_evaluated'],finite_validation=result['finite_validation'],training_approved=False)))
