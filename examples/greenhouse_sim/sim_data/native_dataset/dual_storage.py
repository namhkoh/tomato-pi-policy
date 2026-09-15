"""Diagnostic dual serialization of ONE callback; no data/release approval.

Write raw and compact storage independently from the same validated callback.
Compare complete RGB/native NPY bytes, every prim identity, masks, annotation,
trace and metadata. Unlike separately rendered runs, this isolates serialization
from renderer variation. Caller arrays and existing files must remain unchanged.
This expensive diagnostic is not the production capture path or a speed test.
"""
from copy import deepcopy
import json
from pathlib import Path
import numpy as np
from ..capture_contract import write_sample, jsonable
from ..capture_visibility import write_visibility
from ..dataset_review import require, write_json
from .bundle import ARRAYS, REQUIRED, SampleReader, digest
from .capture_storage import write_compact_native_sample

ARRAY_KEYS=('rgb','depth','valid','instances','components','organs','target_mask')
JSON_KEYS=('metadata','mapping','catalogue','label','trace')


def _snapshot(callback):
    result={}
    for name in ARRAY_KEYS:
        value=callback[name]
        require(isinstance(value,np.ndarray),'Callback buffer required: '+name)
        result[name]=dict(dtype=value.dtype.str,shape=list(value.shape),sha256=digest(value.tobytes()))
    for name in JSON_KEYS:
        result[name]=digest(json.dumps(jsonable(callback[name]),sort_keys=True,allow_nan=False).encode())
    return result


def write_pair(raw_directory, compact_directory, callback, *, compact_writer=write_compact_native_sample):
    """Return raw/compact result bindings after exhaustive same-callback checks.

    Destinations must be new, disjoint, non-root paths. Partial failed outputs
    are preserved but lack same_callback_storage.json; no source is removed.
    A native caller must separately bind this module and retain capture guards.
    Unit-test or replay callers do not establish a live Isaac qualification.
    """
    raw_directory,compact_directory=Path(raw_directory).resolve(),Path(compact_directory).resolve()
    for path in (raw_directory,compact_directory):
        require(not path.exists() and path != Path(path.anchor),'New non-root destination required')
    require(not raw_directory.is_relative_to(compact_directory)
            and not compact_directory.is_relative_to(raw_directory),'Disjoint destinations required')
    require(set(callback)==set(ARRAY_KEYS+JSON_KEYS),'Exact callback arguments required')
    before=_snapshot(callback)
    private={key:callback[key].copy() if key in ARRAY_KEYS else deepcopy(callback[key]) for key in callback}
    write_sample(raw_directory,private['rgb'],private['depth'],private['valid'],deepcopy(private['metadata']))
    write_visibility(raw_directory,*(private[k] for k in
        ('instances','mapping','catalogue','components','organs','target_mask')))
    write_json(raw_directory/'supervision/label.json',private['label'])
    if private['trace'] is not None:write_json(raw_directory/'supervision/query_trace.json',private['trace'])
    raw_result=dict(sample_sha256=digest((raw_directory/'sample.json').read_bytes()),
        label_sha256=digest((raw_directory/'supervision/label.json').read_bytes()),query_trace=deepcopy(private['trace']))
    raw=SampleReader(raw_directory,expected_bindings={'sample.json':raw_result['sample_sha256'],
        'supervision/label.json':raw_result['label_sha256']},expected_json=(
            {'supervision/query_trace.json':private['trace']} if private['trace'] is not None else {}))
    require(raw.verify_all(),'Raw writer integrity failed')
    compact_result=compact_writer(compact_directory,**private)
    compact=SampleReader(compact_directory,expected_bindings={'sample.json':compact_result['sample_sha256'],
        'supervision/label.json':compact_result['label_sha256']},expected_json=(
            {'supervision/query_trace.json':private['trace']} if private['trace'] is not None else {}))
    require(compact.manifest is not None,'Compact bundle format required; raw files are not a compact proof')
    require(compact.verify_all(),'Compact writer integrity failed')
    require(before==_snapshot(private)==_snapshot(callback),'Callback changed during serialization')
    exact={}
    for name in sorted(REQUIRED-{'supervision/identities.json'}):
        left,right=raw.read(name),compact.read(name)
        require(left==right,'Different same-callback observation: '+name)
        exact[name]=dict(sha256=digest(left),logical_bytes=len(left))
    names=['supervision/identities.json','supervision/label.json']
    if private['trace'] is not None:names.append('supervision/query_trace.json')
    for name in names:require(raw.json(name)==compact.json(name),'Changed identity/annotation: '+name)
    expected=deepcopy(raw.metadata)
    expected['files']={name:value for name,value in expected['files'].items() if name in REQUIRED}
    expected['files']['supervision/identities.json']['sha256']=digest(compact.read('supervision/identities.json'))
    require(expected==compact.metadata,'Changed same-callback metadata')
    require(compact_result['query_trace']==raw_result['query_trace'],'Changed trace result')
    for name,key in [('inputs/depth_m.npy','depth'),('supervision/renderer_instance_id.npy','instances'),
                     ('supervision/component_id.npy','components')]:
        require(compact.array(name).tobytes()==private[key].tobytes(),'Native callback bits changed: '+key)
    require(np.array_equal(compact.image('inputs/rgb.png'),private['rgb']),'Native callback RGB changed')
    # verify_all() validates files using cached metadata; explicitly reread each
    # externally pinned metadata file as well, including the raw sample marker.
    raw.read('sample.json');compact.read('sample.json')
    require(raw.verify_all() and compact.verify_all(),'Post-comparison storage integrity failed')
    evidence=dict(schema='greenhouse.same_callback_dual_storage.v1',
        state='same_callback_serialization_exact_not_native_qualification_by_itself',
        raw_directory=str(raw_directory),compact_directory=str(compact_directory),
        raw_result=raw_result,compact_result=compact_result,callback_buffers=before,
        exact_observations=exact,complete_prim_identity_table_equal=True,
        native_npy_bytes_exact=len(ARRAYS),rgb_png_bytes_exact=True,annotation_and_trace_equal=True,
        metadata_equal_except_review_files_and_json_newlines=True,caller_unchanged=True,
        source_removed=False,depth_recomputed=False,training_approved=False,training_diversity_increment=0,
        scope='serialization only; renderer/camera/geometry/labels remain independently validated by caller')
    write_json(raw_directory/'same_callback_storage.json',evidence)
    return evidence
