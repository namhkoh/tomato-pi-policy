from copy import deepcopy

import numpy as np
import pytest

from sim_data.capture_visibility import (component_masks, decode_instances, interval_visibility,
                                         owner_for_path, view_quality, write_visibility)
from sim_data.capture_contract import write_sample


def fixture():
    target = {"prim_path":"/World/Plant/Stem/Petiole","component_index":2,"organ_type":"sub_stem"}
    leaf = {"prim_path":"/World/Plant/Stem/Petiole/Leaf","component_index":3,"organ_type":"leaf"}
    mapping = {4:target["prim_path"]+"/Mesh", 5:leaf["prim_path"]+"/Mesh"}
    ids = np.zeros((408,848),np.uint32)
    ids[200:205,400:408] = 4
    depth = np.full((408,848),np.inf,np.float32)
    depth[ids==4] = .499
    p = {"projection_status":"in_frame","pixel_xy":[401.2,201.2],"camera_optical_xyz_m":[0,0,.5]}
    return target, leaf, mapping, ids, depth, p


def test_deepest_organ_ownership_excludes_descendant_leaf_and_prefix_collision():
    target,leaf,mapping,*_ = fixture()
    assert owner_for_path(mapping[5],[target,leaf])==leaf
    assert owner_for_path(target["prim_path"]+"Other/Mesh",[target,leaf]) is None
    assert owner_for_path(mapping[4],[target,leaf])==target


@pytest.mark.parametrize("kind",["missing","dtype","shape","colorized","mapping","unmapped"])
def test_decode_refuses_incomplete_or_ambiguous_identity(kind):
    *_, ids, _, _ = fixture()
    annotation = {"data":ids,"info":{"idToLabels":{"4":"/World/Target"}}}
    if kind=="dtype": annotation["data"]=ids.astype(np.int32)
    if kind=="shape": annotation["data"]=ids[:-1]
    if kind=="colorized": annotation["data"]=np.zeros((408,848,4),np.uint8)
    if kind=="mapping": annotation["info"]["idToLabels"]={"4":{"class":"target"}}
    if kind=="unmapped": annotation["data"]=np.full(ids.shape,200,np.uint32)
    with pytest.raises(ValueError): decode_instances({} if kind=="missing" else {"instance_id_segmentation":annotation})


def test_native_identity_and_depth_are_both_required_and_masks_separate_organs():
    target,leaf,mapping,ids,depth,p=fixture()
    components,organs,owners=component_masks(ids,mapping,[target,leaf])
    v,mask=interval_visibility(p,[p,p],depth,np.isfinite(depth),ids,mapping,owners,target,.002)
    assert v["nominal"]["visible_target_evidence"]
    assert v["unique_in_frame_interval_pixels"]==1  # No oversampling inflation.
    assert v["sampled_interval_visible_pixel_fraction"]==1
    assert v["amodal_surface_visibility_fraction"] is None
    assert mask.sum()==40 and components[201,401]==2 and organs[201,401]==2
    ids[201,401]=5
    components,organs,owners=component_masks(ids,mapping,[target,leaf])
    v,_=interval_visibility(p,[p],depth,np.isfinite(depth),ids,mapping,owners,target,.002)
    assert not v["nominal"]["visible_target_evidence"]  # Same-depth leaf is not petiole.
    assert organs[201,401]==3 and components[201,401]==3
    depth[201,401]=.3
    v,_=interval_visibility(p,[p],depth,np.isfinite(depth),ids,mapping,owners,target,.002)
    assert v["nominal"]["status"]=="foreground_occluder_identified"
    assert v["identified_foreground_occluders"][0]["component"]==leaf


@pytest.mark.parametrize("depth_value",[np.inf,np.nan,0.,.3,.8])
def test_matching_instance_with_invalid_or_conflicting_depth_is_not_visible(depth_value):
    target,leaf,mapping,ids,depth,p=fixture()
    depth[201,401]=depth_value
    _,_,owners=component_masks(ids,mapping,[target,leaf])
    valid=np.isfinite(depth)&(depth>0)
    v,_=interval_visibility(p,[p],depth,valid,ids,mapping,owners,target,.002)
    assert not v["nominal"]["visible_target_evidence"]


def test_out_of_frame_unknown_and_folded_projection_are_conservative():
    target,leaf,mapping,ids,depth,p=fixture()
    _,_,owners=component_masks(ids,mapping,[target,leaf])
    outside={**p,"projection_status":"out_of_frame","pixel_xy":[-1,201]}
    v,_=interval_visibility(outside,[outside],depth,np.isfinite(depth),ids,mapping,owners,target,.002)
    assert v["sampled_interval_visible_pixel_fraction"] is None
    folded={**p,"camera_optical_xyz_m":[0,0,.8]}
    v,_=interval_visibility(p,[p,folded],depth,np.isfinite(depth),ids,mapping,owners,target,.002)
    assert v["visible_interval_pixels"]==0


def test_quality_does_not_approve_thin_dark_or_occluded_targets():
    target,leaf,mapping,ids,depth,p=fixture()
    _,_,owners=component_masks(ids,mapping,[target,leaf])
    interval=[p,{**p,"pixel_xy":[406.2,201.2]}]
    v,mask=interval_visibility(p,interval,depth,np.isfinite(depth),ids,mapping,owners,target,.002)
    cal={"intrinsics":[[470,0,424],[0,470,204],[0,0,1]]}
    q=view_quality(cal,p,interval,.002,v,np.full((408,848,3),100,np.uint8),mask)
    assert q["clear_view_gate_passed"] and not q["training_approved"]
    q=view_quality(cal,p,interval,.001,v,np.zeros((408,848,3),np.uint8),mask)
    assert not q["clear_view_gate_passed"] and len(q["clear_view_rejection_reasons"])==2


def test_supervision_does_not_change_rgb_or_observation_allowlist(tmp_path):
    import hashlib, json
    target,leaf,mapping,ids,depth,p=fixture()
    components,organs,owners=component_masks(ids,mapping,[target,leaf])
    v,mask=interval_visibility(p,[p],depth,np.isfinite(depth),ids,mapping,owners,target,.002)
    directory=tmp_path/"sample"
    meta={"input_policy":{"allowed_observation_files":["inputs/rgb.png"]},
          "supervision":{"nominal_projected":p,"projected_interval":[p],
                         "depth_evidence":{"status":"test_only"}}}
    write_sample(directory,np.full((408,848,3),100,np.uint8),depth,np.isfinite(depth),meta)
    before=(directory/"inputs/rgb.png").read_bytes()
    write_visibility(directory,ids,mapping,[target,leaf],components,organs,mask)
    assert (directory/"inputs/rgb.png").read_bytes()==before
    saved=json.loads((directory/"sample.json").read_text())
    assert saved["input_policy"]==meta["input_policy"]
    for path,entry in saved["files"].items():
        assert hashlib.sha256((directory/path).read_bytes()).hexdigest()==entry["sha256"]
        if path.startswith("supervision/"): assert entry["role"]=="ground_truth_supervision"
    with pytest.raises(FileExistsError): write_visibility(directory,ids,mapping,[target,leaf],components,organs,mask)
