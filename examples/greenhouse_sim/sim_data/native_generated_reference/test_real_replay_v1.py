"""Opt-in CPU regression on existing native bytes, NOT new capture evidence.

Copies an existing original sample to pytest's temporary directory and wraps
metadata in the NEW schema only to test replay plumbing. This synthetic wrapper
must never enter an inventory, bank, qualification receipt, or native output.
"""
from copy import deepcopy
import os
from pathlib import Path
import shutil

import numpy as np
import pytest

from . import execution_v1 as ex, audit_v1 as audit

pytestmark=pytest.mark.skipif(os.environ.get('GENERATED_REFERENCE_REAL_CPU')!='1',reason='explicit existing-buffer CPU regression only')


@pytest.mark.parametrize('offset,accepted',[(5e-8,True),(2e-7,False)])
def test_actual_camera_tolerance_matches_worker_and_native_fk(offset,accepted):
    root=Path(__file__).resolve().parents[4]
    proof=ex.oc.read_json(root/'data/sim_data/diagnostics/generated_original_seed73_prepare_20260916_v1/original_anchor_evidence.json')
    pin=proof['native_observation']['sample'];meta=ex.oc.read_json(ex.oc.pin(pin['path'],pin['sha256']))
    expected=deepcopy(meta['calibration'])
    meta['calibration']['camera_to_world_usd_row_vectors'][3][0]+=offset
    actual=deepcopy(meta['calibration'])
    if accepted:audit.verify_actual_camera(meta,expected)
    else:
        with pytest.raises(ValueError):audit.verify_actual_camera(meta,expected)
    assert meta['calibration']==actual


def test_native_rendered_camera_check_not_bypassed_by_tolerant_plan_comparison():
    root=Path(__file__).resolve().parents[4]
    proof=ex.oc.read_json(root/'data/sim_data/diagnostics/generated_original_seed73_prepare_20260916_v1/original_anchor_evidence.json')
    pin=proof['native_observation']['sample'];meta=ex.oc.read_json(ex.oc.pin(pin['path'],pin['sha256']))
    meta['rendered_camera_params']['cameraViewTransform'][12]+=1
    with pytest.raises(ValueError,match='Native renderer camera'):
        audit.verify_actual_camera(meta,meta['calibration'])


def test_existing_original_bytes_new_metadata_plumbing_and_mutation(tmp_path):
    from ..collection_plan import load_plan
    from ..native_dataset import inventory as inv
    from ..native_dataset.bundle import SampleReader
    from .scene import current_scene_evidence
    root=Path(__file__).resolve().parents[4]
    plan_path=root/'data/sim_data/diagnostics/generated_original_seed73_prepare_20260916_v1/plan.json'
    ex.oc.pin(plan_path,'077bcd88bc6e5231c1d212c753184159bd4a1101f837143a2444e6f5a19a953f')
    plan=ex.oc.read_json(plan_path)
    proof=ex.oc.read_json(ex.oc.pin(plan['anchor_evidence']['path'],plan['anchor_evidence']['sha256']))
    source=Path(proof['native_observation']['sample']['path']).parent
    meta=ex.oc.read_json(source/'sample.json');mode='original_control';folder=tmp_path/mode
    for name,entry in meta['files'].items():
        src=ex.oc.pin(source/name,entry['sha256']);dest=folder/name
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
    from PIL import Image
    with Image.open(folder/'supervision/target_visible.png') as image:mask=np.asarray(image)>0
    ids=np.load(folder/'supervision/renderer_instance_id.npy',allow_pickle=False)
    identities=ex.oc.read_json(folder/'supervision/identities.json')
    old_root=plan['scene_authority']['original_variant']['plant_root']
    old_ids=[int(i) for i,p in identities['renderer_id_to_prim'].items() if p==old_root or p.startswith(old_root+'/')]
    meta.update(schema_version=ex.SAMPLE,state='fresh_native_reference_pair_pending_replay',sample_id=mode,
        native_target_pixels=int(mask.sum()),old_plant_native_pixels=int(np.isin(ids,old_ids).sum()),
        scene_evidence=current_scene_evidence(plan,lighting=meta['lighting'],counts=meta['scene_counts'],renderer=meta['renderer']))
    ex.oc.write_new(folder/'sample.json',meta)
    _,reports=load_plan(plan['scene_authority']['clear_plan']['path']);reports={r['plant_id']:r for r in reports}
    # New sample_id legitimately changes the derived label identity. Derive it
    # afresh from copied native bytes; never relabel a historical saved trace.
    from ..native_clear_labels import derive
    from ..automated_native_review import trace_review
    with Image.open(folder/'inputs/rgb.png') as im:rgb=np.asarray(im).copy()
    with Image.open(folder/'inputs/depth_valid.png') as im:valid=np.asarray(im)!=0
    depth=np.load(folder/'inputs/depth_m.npy',allow_pickle=False)
    components=np.load(folder/'supervision/component_id.npy',allow_pickle=False)
    report=reports[plan['source_family']];catalogue=identities['component_catalogue']
    label=derive(meta,report,rgb,depth,valid,components,catalogue)
    trace=trace_review(meta,report,label,rgb,depth,valid,components,catalogue)
    ex.oc.write_new(folder/'supervision/label.json',label)
    ex.oc.write_new(folder/'supervision/query_trace.json',trace)
    record=dict(sample_sha256=ex.oc.sha256(folder/'sample.json'),label_sha256=ex.oc.sha256(folder/'supervision/label.json'),
        query_trace=ex.oc.read_json(folder/'supervision/query_trace.json'),screen=meta['geometry_screen'],
        target_id=plan['source_row']['target_id'],decision=inv.STRICT)
    reviewed,_=audit.review_sample(tmp_path,plan,mode,record,reports,None,inv._Bindings())
    assert reviewed['native_ID_masks_replayed'] and reviewed['decision']==inv.STRICT
    assert not reviewed['training_approved']
    bad=deepcopy(record);bad['target_id']='seed73_full/SubStem_42'
    with pytest.raises(ValueError,match='ancestry'):
        audit.review_sample(tmp_path,plan,mode,bad,reports,None,inv._Bindings())
