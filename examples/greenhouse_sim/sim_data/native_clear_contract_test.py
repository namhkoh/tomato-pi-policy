from copy import deepcopy
import json
import numpy as np
from PIL import Image, ImageChops
import pytest
from .capture_contract import project
from .native_clear_contract import (RESOLUTION,CONTRACT,contract_hash,crop_box,model_messages,
    model_answer,canonical_answer,validate_answer,clear_screen)
from .native_query_visibility import NativeQueryVisibility
from .query_visibility import QueryVisibility
from .query_visibility_test import fixture as query_fixture
from .native_clear_labels import derive


def close_messages(messages):
    for item in messages[1]["content"]:
        if item["type"]=="image":item["image"].close()


def fixture():
    # UNIT-TEST geometry/buffers only, never capture/training observations.
    component=dict(id="Petiole",type="sub_stem",parent="Main",deleafed=False,
        translation_plant_m=[0,0,-1],attachment_plant_m=[0,0,-1],axis_plant=[1,0,0],
        capsules_local_m=[[[0,0,0,.002],[.2,0,0,.002]]])
    report=dict(plant_id="fixture",components={"Petiole":component,"Main":dict(id="Main",type="main_stem")})
    cal=dict(resolution=list(RESOLUTION),crop_resize=None,depth_convention="optical_axis_z_metres_not_ray_range",
        camera_to_world_usd_row_vectors=np.eye(4).tolist(),clipping_range_m=[.01,10.],
        intrinsics=[[3000,0,848],[0,3000,408],[0,0,1]])
    nominal=[.01,0.,-1.]
    meta=dict(sample_id="unit_test_only",calibration=cal,
        synchronization=dict(scene_unchanged_during_capture=True,dynamic_recording_supported=False),
        quality=dict(estimated_petiole_diameter_px=12.),
        supervision=dict(target_id="fixture/Petiole",split_group="fixture",
            plant_to_world_usd_row_vectors=np.eye(4).tolist(),nominal_world_m=nominal,
            nominal_projected=project([nominal],cal)[0],
            interval_world_m=[[float(x),0.,-1.] for x in np.linspace(.01,.02,11)]))
    components=np.zeros((816,1696),np.int32)
    components[402:414,852:1500]=1
    components[380:436,840:852]=2
    rgb=np.full((816,1696,3),150,np.uint8)
    rgb[components==1]=[50,100,30]
    depth=np.full((816,1696),.998,np.float32)
    valid=np.ones((816,1696),bool)
    catalogue=[dict(variant_id="fixture",component_id="Petiole",component_index=1),
               dict(variant_id="fixture",component_id="Main",component_index=2)]
    return meta,report,rgb,depth,valid,components,catalogue


@pytest.mark.parametrize("query",[[0.,0.],[1695.999,815.999],[848.,408.],[1300.1,123.4]])
def test_explicit_native_coordinates_crop_and_roundtrip(query):
    answer=dict(status="localized",cut_point_uv=query,visibility="clear",next_action="inspect_cut_region")
    decoded=canonical_answer(model_answer(answer))
    assert max(abs(a-b) for a,b in zip(query,decoded["cut_point_uv"]))<=.01696
    x0,y0,x1,y1=crop_box(query)
    assert x1-x0==y1-y0==768 and 0<=x0<x1<=1696 and 0<=y0<y1<=816


@pytest.mark.parametrize("bad",[[1696.,1.],[1.,816.],[-1.,0.],[float("nan"),0.],[True,3.],[1.,2.,3.],None])
def test_invalid_coordinates_never_guessed_or_clipped(bad):
    with pytest.raises(ValueError):crop_box(bad)


@pytest.mark.parametrize("crop",[False,True])
def test_training_inference_and_native_crop_no_resize_or_answer_leak(crop):
    with Image.new("RGB",RESOLUTION,"green") as rgb:
        rgb.putpixel((1500,450),(255,0,0))
        answer=dict(status="localized",cut_point_uv=[1300.,400.],visibility="clear",next_action="inspect_cut_region")
        train=model_messages(rgb,[1500.,450.],answer=answer,crop=crop)
        infer=model_messages(rgb,[1500.,450.],crop=crop)
        other=model_messages(rgb,[1500.,450.],answer=dict(answer,cut_point_uv=[1200.,350.]),crop=crop)
        try:
            assert [m["role"] for m in infer]==["system","user"]
            assert train[0]==infer[0] and train[1]["content"][-1]==infer[1]["content"][-1]==other[1]["content"][-1]
            assert "1300" not in infer[1]["content"][-1]["text"]
            for a,b in zip(train[1]["content"][:-1],infer[1]["content"][:-1],strict=True):
                assert ImageChops.difference(a["image"],b["image"]).getbbox() is None
            if crop:
                with rgb.crop(crop_box([1500.,450.])) as exact:
                    assert train[1]["content"][1]["image"].size==(768,768)
                    assert ImageChops.difference(exact,train[1]["content"][1]["image"]).getbbox() is None
        finally:
            for messages in (train,infer,other):close_messages(messages)
    with Image.new("RGB",(848,408)) as old:
        with pytest.raises(ValueError):model_messages(old,[100.,100.])


def test_query_legacy_equivalence_without_changing_legacy():
    rgb,mask=query_fixture()
    old=QueryVisibility(rgb,mask).inspect([135.4,104.2])
    new=NativeQueryVisibility(rgb,mask,(848,408)).inspect([135.4,104.2])
    assert {k:v for k,v in old.items() if k!="policy_version"}=={k:v for k,v in new.items() if k!="policy_version"}


def test_native_query_right_half_and_gap_not_filled():
    rgb=np.full((816,1696,3),150,np.uint8)
    mask=np.zeros((816,1696),bool)
    mask[200:212,1300:1380]=True
    mask[205,1382:1384]=True
    rgb[mask]=[50,100,30]
    before=mask.copy()
    checker=NativeQueryVisibility(rgb,mask,RESOLUTION)
    assert checker.inspect([1340.,206.])["passed"]
    assert not checker.inspect([1382.,205.])["passed"]
    assert np.array_equal(before,mask)


def test_derive_native_visible_target_keeps_sources_and_no_approval():
    args=fixture()
    original=[a.copy() for a in args[2:6]]
    result=derive(*args)
    assert result["eligible"] and not result["training_approved"]
    assert result["nominal_pixel_uv"]==[878.,408.]
    assert result["answer"]["cut_point_uv"]==[878.,408.]
    assert result["query_evidence"]["arc_m"]>=.045
    assert result["query_cut_visible_connection_verified"]
    assert not result["hidden_cut_coordinates_executable"]
    for before,after in zip(original,args[2:6]):assert np.array_equal(before,after)


@pytest.mark.parametrize("fault,reason",[
    ("mask","cut_interval_not_fully_native_visible"),
    ("foreground","cut_interval_not_fully_native_visible"),
    ("gap","no_usable_visible_query_connected_to_cut"),
    ("parent","insufficient_visible_parent_clearance"),
    ("dark","strict_native_clarity_failed")])
def test_nonclear_native_cases_remain_excluded(fault,reason):
    a=list(fixture())
    if fault=="mask":a[5][:,878:909]=0
    if fault=="foreground":a[3][408,878]=.5
    if fault=="gap":a[5][:,950:955]=0
    if fault=="parent":a[5][a[5]==2]=0
    if fault=="dark":a[2][a[5]==1]=10
    result=derive(*a)
    assert not result["eligible"] and result["reason"]==reason and "answer" not in result


@pytest.mark.parametrize("fault",["geometry","resolution","stale","organ","validity","float_component_ids"])
def test_invalid_or_stale_native_evidence_fails(fault):
    a=list(fixture())
    if fault=="geometry":a[0]["supervision"]["nominal_world_m"][0]=.011
    if fault=="resolution":a[0]["calibration"]["resolution"]=[848,408]
    if fault=="stale":a[0]["synchronization"]["scene_unchanged_during_capture"]=False
    if fault=="organ":a[1]["components"]["Petiole"]["type"]="fruit"
    if fault=="validity":a[4][0,0]=False
    if fault=="float_component_ids":a[5]=a[5].astype(float)
    with pytest.raises(ValueError):derive(*a)


def test_absent_target_answer_contract_and_digest():
    answer=dict(status="abstain",cut_point_uv=None,visibility="occluded",next_action="change_viewpoint")
    assert canonical_answer(model_answer(answer))==answer
    with pytest.raises(ValueError):validate_answer(dict(answer,cut_point_uv=[1.,2.]))
    with pytest.raises(ValueError):validate_answer(dict(answer,xyz=[1,2,3]))
    before=contract_hash()
    saved=deepcopy(CONTRACT)
    assert len(before)==64 and json.loads(json.dumps(CONTRACT))==CONTRACT and saved==CONTRACT
