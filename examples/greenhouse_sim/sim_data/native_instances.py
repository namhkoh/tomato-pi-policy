"""Adapt native fast instance IDs without the per-subframe legacy JSON node.

The final synchronized callback is copied by the caller. Comparison mode attaches
both annotators and requires identical ID buffers and exact observed prim paths.
"""
from __future__ import annotations

import numpy as np

from .capture_visibility import decode_instances
from .dataset_review import require

LEGACY='instance_id_segmentation'
FAST='instance_id_segmentation_fast'


def canonical_fast(annotation):
    require(isinstance(annotation,dict) and {'data','info'}<=annotation.keys(), 'Missing native fast annotation')
    info=annotation['info']
    if 'idToLabels' in info:
        labels=info['idToLabels']
    else:
        require('ids' in info and 'labels' in info, 'Missing fast instance IDs/labels')
        ids=np.asarray(info['ids'])
        values=info['labels']
        require(ids.ndim==1 and np.issubdtype(ids.dtype,np.integer) and len(ids)==len(values), 'Invalid fast mapping shape')
        require(len(np.unique(ids))==len(ids), 'Duplicate fast mapping IDs')
        labels={int(i):v for i,v in zip(ids,values)}
    canonical=dict(data=annotation['data'],info={'idToLabels':labels})
    pixels,mapping=decode_instances({LEGACY:canonical})
    # Retain every observed ID exactly, not the large unused renderer table.
    # This does not drop any visible pixel identity or component catalogue entry.
    observed=set(map(int,np.unique(pixels)))
    canonical['info']['idToLabels']={i:mapping[i] for i in observed if i in mapping}
    canonical['info']['mapping_scope']='all_observed_renderer_IDs_only'
    return canonical


def normalize_payload(payload,backend):
    require(backend in ('legacy','fast','compare'), 'Unknown instance backend')
    if backend=='legacy':
        return payload
    fast=canonical_fast(payload.get(FAST))
    evidence=None
    if backend=='compare':
        a,ma=decode_instances({LEGACY:payload.get(LEGACY)})
        b,mb=decode_instances({LEGACY:fast})
        observed=set(map(int,np.unique(a)))
        require(np.array_equal(a,b), 'Fast/legacy native instance pixels differ')
        require(all(ma.get(i)==mb.get(i) for i in observed), 'Fast/legacy observed prim identities differ')
        evidence=dict(passed=True,scope='all_native_ID_pixels_and_all_observed_prim_paths',
                      compared_pixels=int(a.size),observed_ID_count=len(observed))
    result={k:v for k,v in payload.items() if k not in (LEGACY,FAST)}
    result[LEGACY]=fast
    result['native_instance_backend']=backend
    result['native_instance_mapping_scope']='all_observed_renderer_IDs_only'
    if evidence is not None: result['native_instance_equivalence']=evidence
    return result
