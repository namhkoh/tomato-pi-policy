"""Threshold-free native848 distances using the existing full/nominal-patch MAE.

No duplicate cutoff, graph admission, geometric novelty or release decision is
made here. Unsupported complete129x129 patches remain unknown. The batching
changes only integer reduction order: exact uint64 sums cannot overflow for
848x408x3 uint8 inputs, and normalization is the same mean/255 operation.
"""
from pathlib import Path
import numpy as np
from . import near_image,near_image_index

SCHEMA='greenhouse.native848_threshold_free_pair_distances.v1'
DIMENSIONS=(848,408)
PATCH_RADIUS=64


def batch_mae(left,rights):
    """Return unchanged native MAE for one image/patch against a bounded batch."""
    near_image._rgb(left)
    near_image_index._require(isinstance(rights,np.ndarray) and rights.dtype==np.uint8
        and rights.ndim==4 and rights.shape[1:]==left.shape,'Matching native uint8 batch required')
    delta=np.subtract(rights,left,dtype=np.int16)
    np.abs(delta,out=delta)
    # Sum of <=1,037,952 values in0..255 fits exactly in uint64 and float64.
    sums=delta.sum(axis=(1,2,3),dtype=np.uint64)
    return sums.astype(np.float64)/left.size/255.0


def measure_all(pins,*,batch_size=16,progress=None):
    """Measure all supported pairs; no default/optional threshold exists.

    Returned matrices retain every supplied row. NaN means the frozen metric
    cannot measure that pair, not exclusion/admission or a large distance.
    ``pins`` are the existing hash-bound near_image_index.ImagePin objects.
    Every native file is authenticated initially and again at completion.
    """
    pins=sorted(list(pins),key=lambda p:p.sample_id)
    near_image_index._require(pins and len({p.sample_id for p in pins})==len(pins)
        and all(isinstance(p,near_image_index.ImagePin) for p in pins),'Unique existing ImagePins required')
    near_image_index._require(type(batch_size)is int and 1<=batch_size<=32,'Bounded batch_size1..32 required')
    code={str(Path(p).resolve()):near_image_index._sha(p) for p in (__file__,near_image.__file__,near_image_index.__file__)}
    # At495inputs the fullRGB+local-patch cache stays below550MiB.
    native=np.empty((len(pins),408,848,3),dtype=np.uint8)
    patches=np.empty((len(pins),129,129,3),dtype=np.uint8)
    compatible=[];unknown=[]
    for i,pin in enumerate(pins):
        rgb=near_image_index._load(pin)
        near_image_index._require(rgb.shape==(408,848,3),'This path supports native848x408 only')
        native[i]=rgb
        try:
            bounds=near_image._patch_bounds(rgb,pin.nominal_uv,PATCH_RADIUS)
            patches[i]=rgb[bounds[1]:bounds[3],bounds[0]:bounds[2]]
            compatible.append(i)
        except ValueError as exc:
            if str(exc)!='Complete native junction patch required; no padding or clipping':raise
            unknown.append(dict(index=i,sample_id=pin.sample_id,reason=str(exc)))
        if progress is not None and ((i+1)%50==0 or i+1==len(pins)):
            progress(dict(phase='load',images=i+1,total=len(pins)))
    full=np.full((len(pins),len(pins)),np.nan,dtype=np.float64)
    local=np.full_like(full,np.nan);combined=np.full_like(full,np.nan)
    measured=0;reference_checks=0
    for n,i in enumerate(compatible):
        full[i,i]=local[i,i]=combined[i,i]=0.0
        for start in range(n+1,len(compatible),batch_size):
            js=np.asarray(compatible[start:start+batch_size],dtype=np.int64)
            f=batch_mae(native[i],native[js]);p=batch_mae(patches[i],patches[js]);s=np.maximum(f,p)
            # Actual-data parity guard on the first pair of every batch.
            j=int(js[0])
            near_image_index._require(float(f[0])==near_image_index._mae(native[i],native[j])
                and float(p[0])==near_image_index._mae(patches[i],patches[j]),'Existing metric parity failure')
            reference_checks+=1
            full[i,js]=full[js,i]=f;local[i,js]=local[js,i]=p;combined[i,js]=combined[js,i]=s
            measured+=len(js)
        if progress is not None and ((n+1)%10==0 or n+1==len(compatible)):
            progress(dict(phase='pairs',completed_rows=n+1,compatible_rows=len(compatible),measured_pairs=measured,total_pairs=len(compatible)*(len(compatible)-1)//2))
    expected=len(compatible)*(len(compatible)-1)//2
    near_image_index._require(measured==expected,'Incomplete supported pair accounting')
    for pin in pins:near_image_index._require(near_image_index._sha(pin.path)==pin.sha256,'Source image changed')
    for p,pin in code.items():near_image_index._require(near_image_index._sha(p)==pin,'Comparison implementation changed')
    metadata=dict(schema=SCHEMA,metric=near_image.METRIC,dimensions=list(DIMENSIONS),patch_radius=PATCH_RADIUS,
        patch_anchor='nominal_10mm_cut_point_not_anatomical_attachment',sample_ids=[p.sample_id for p in pins],
        supported_indices=compatible,unsupported=unknown,counts=dict(images=len(pins),supported_images=len(compatible),unsupported_images=len(unknown),measured_pairs=measured,unknown_pairs=len(pins)*(len(pins)-1)//2-measured,all_selection_pairs=len(pins)*(len(pins)-1)//2,actual_reference_metric_parity_checks=reference_checks),
        missing_value='NaN means frozen metric unavailable; do not impute, drop or classify',
        threshold=None,threshold_calibrated=False,near_image_graph_built=False,all_supported_pairs_measured=True,
        selection_changed=False,geometry_novelty_claimed=False,training_approved=False,implementation_bindings=code)
    return metadata,dict(full_frame_mae=full,nominal_cut_patch_mae=local,combined_score=combined)


def load_bound_table(result_path,*,result_sha256):
    """Authenticate this measurement table for a later separately bound calibration.

    This loader makes no decision and accepts no threshold. A future calibration
    consumer must bind actual848control evidence and preserve unknown rows.
    """
    import json
    path=Path(result_path).resolve()
    near_image_index._require(near_image_index._sha(path)==result_sha256,'Result receipt changed')
    result=json.loads(path.read_text(encoding='utf-8'))
    near_image_index._require(result['schema']==SCHEMA and result['state']=='all_supported_native848_pair_distances_complete_no_threshold','Unsupported distance receipt')
    near_image_index._require(result['definition']['dimensions']==[848,408] and result['definition']['patch_radius']==64
        and result['definition']['metric']==near_image.METRIC and result['threshold'] is None,'Metric contract changed')
    for filename,pin in {**result['input_bindings'],**result['implementation_bindings']}.items():
        near_image_index._require(near_image_index._sha(filename)==pin,'Bound input/implementation changed')
    arrays={}
    for name,row in result['arrays'].items():
        file=(path.parent/row['file']).resolve();near_image_index._require(file.is_relative_to(path.parent),'Array path escaped output')
        near_image_index._require(near_image_index._sha(file)==row['sha256'],'Distance array changed')
        arrays[name]=np.load(file,allow_pickle=False,mmap_mode='r')
    return result,arrays
