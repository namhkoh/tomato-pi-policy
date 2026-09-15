"""Resolution-aware fast native instance adapter, preserving every visible ID.

Legacy 848x408 adapter is unchanged. Compare mode requires exact native ID
pixels and observed prim paths from the SAME synchronized callback.
"""
from copy import deepcopy
import numpy as np
from .dataset_review import require
from .native_sensor_payload import decode_native_instances
from .native_instances import LEGACY,FAST


def canonical_fast(annotation,resolution):
    require(isinstance(annotation,dict) and {'data','info'}<=annotation.keys(),'Missing fast native annotation')
    info=annotation['info']
    if 'idToLabels' in info:labels=info['idToLabels']
    else:
        require('ids' in info and 'labels' in info,'Missing fast native mapping')
        ids=np.asarray(info['ids']);values=info['labels']
        require(ids.ndim==1 and np.issubdtype(ids.dtype,np.integer) and len(ids)==len(values),
                'Invalid fast native mapping shape')
        require(len(np.unique(ids))==len(ids),'Duplicate fast native IDs')
        labels={int(i):v for i,v in zip(ids,values)}
    result=dict(data=annotation['data'],info={'idToLabels':labels})
    pixels,mapping=decode_native_instances({LEGACY:result},resolution)
    observed=set(map(int,np.unique(pixels)))
    result['info']['idToLabels']={i:mapping[i] for i in observed if i in mapping}
    result['info']['mapping_scope']='all_observed_renderer_IDs_only'
    return result


def normalize(payload,backend,resolution):
    require(backend in ('fast','compare'),'Native fast/compare backend required')
    canonical=canonical_fast(payload.get(FAST),resolution)
    evidence=None
    if backend=='compare':
        a,ma=decode_native_instances(payload,resolution)
        b,mb=decode_native_instances({LEGACY:canonical},resolution)
        require(np.array_equal(a,b),'Fast/legacy native pixels differ')
        require(all(ma.get(int(i))==mb.get(int(i)) for i in np.unique(a)),
                'Fast/legacy observed prim paths differ')
        evidence=dict(passed=True,compared_pixels=int(a.size),resolution=list(resolution),
            observed_ID_count=len(np.unique(a)),
            scope='same_callback_all_native_ID_pixels_and_observed_prim_paths')
    result={k:v for k,v in payload.items() if k not in (LEGACY,FAST)}
    result[LEGACY]=canonical
    result['native_instance_backend']=backend
    result['native_instance_mapping_scope']='all_observed_renderer_IDs_only'
    if evidence is not None:result['native_instance_equivalence']=evidence
    return result


def make_native_writer(rep,resolution,backend):
    from .capture_pilot import make_writer
    require(backend in ('fast','compare'),'Explicit native fast backend required')
    # Reuse the established base sensor annotators/frame registration without
    # invoking its legacy-resolution instance decoder.
    base=make_writer(rep,include_instances=False)
    class NativeInstanceWriter(rep.Writer):
        def __init__(self):
            self.version='native_resolution_fast_instances.v1'
            self.annotators=list(base.annotators)
            for name in ([LEGACY,FAST] if backend=='compare' else [FAST]):
                self.annotators.append(rep.AnnotatorRegistry.get_annotator(name,
                    init_params={'colorize':False},device='cpu'))
            self.sequence,self.latest,self.request_index=0,None,0
            self.capture_error=None
        def write(self,data):
            try:
                self.latest=deepcopy(normalize(data,backend,resolution))
                self.sequence+=1
            except Exception as exc:
                self.capture_error=exc
                raise
        def write_metadata(self):self._is_metadata_written=True
    return NativeInstanceWriter()

