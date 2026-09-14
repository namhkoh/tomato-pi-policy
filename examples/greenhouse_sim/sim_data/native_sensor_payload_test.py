from copy import deepcopy
import numpy as np
import pytest
from .capture_contract_test import fixture
from .capture_contract import validate_payload,validate_static_capture
from .capture_sensor import HIRES_RESOLUTION
from .native_sensor_payload import validate_native_payload,validate_native_static,decode_native_instances,native_dimensions


def high_fixture():
    data,cal=fixture()
    cal["resolution"]=list(HIRES_RESOLUTION)
    cal["intrinsics"]=(np.diag([2.,2.,1.])@np.asarray(cal["intrinsics"])).tolist()
    data["rp_head"]["resolution"]=list(HIRES_RESOLUTION)
    data["camera_params"]["renderProductResolution"]=list(HIRES_RESOLUTION)
    # Synthetic UNIT-TEST buffers, never used as capture/training observations.
    data["rgb"]=np.zeros((816,1696,4),np.uint8)
    data["rgb"][:,:,0]=np.arange(1696,dtype=np.uint16)%255
    data["rgb"][:,:,3]=255
    data["distance_to_image_plane"]=np.full((816,1696),2,np.float32)
    return data,cal


def test_legacy_payload_and_static_token_exact_equivalence():
    d,c=fixture();d["distance_to_image_plane"][0,:3]=[np.nan,np.inf,0]
    a=validate_payload(d,c);b=validate_native_payload(d,c)
    for x,y in zip(a[:3],b[:3]):assert np.array_equal(x,y,equal_nan=True)
    assert a[3]==b[3]
    assert validate_static_capture(d,c,"a"*64,"a"*64,1)[-1]==validate_native_static(d,c,"a"*64,"a"*64,1)[-1]


def test_high_resolution_native_buffers_and_raw_invalid_depth():
    d,c=high_fixture();d["distance_to_image_plane"][0,:3]=[np.nan,np.inf,0]
    rgb,z,valid,ref=validate_native_payload(d,c,[9,60])
    assert rgb.shape==(816,1696,3) and z.shape==(816,1696) and ref==[10,60]
    assert np.isnan(z[0,0]) and np.isinf(z[0,1]) and not valid[0,:3].any()
    assert z[1,1]==2.
    rgb[:]=0;z[:]=0
    assert d["rgb"].std()>0 and d["distance_to_image_plane"][1,1]==2.


@pytest.mark.parametrize("dtype",[np.int32,np.int64,np.uint32])
def test_native_numpy_resolution_arrays(dtype):
    d,c=high_fixture()
    d["camera_params"]["renderProductResolution"]=np.asarray(HIRES_RESOLUTION,dtype=dtype)
    d["rp_head"]["resolution"]=np.asarray(HIRES_RESOLUTION,dtype=dtype)
    assert validate_native_payload(d,c)[0].shape==(816,1696,3)


@pytest.mark.parametrize("dimensions",[[1696.,816.],[True,False],[1696,408],[[1696,816]],[1696],['1696','816']])
def test_native_dimensions_reject_ambiguous_or_unsupported_values(dimensions):
    with pytest.raises(ValueError):native_dimensions(dimensions)


@pytest.mark.parametrize("fault",["cal_size","product_size","params_size","rgb_size","depth_size",
    "camera","projection","units","stale","invalid","blank","reference","guard","sequence"])
def test_high_resolution_mismatch_stale_or_incomplete_fails(fault):
    d,c=high_fixture();before=after="a"*64;sequence=1
    if fault=="cal_size":c["resolution"]=[848,408]
    if fault=="product_size":d["rp_head"]["resolution"]=[848,408]
    if fault=="params_size":d["camera_params"]["renderProductResolution"]=[848,408]
    if fault=="rgb_size":d["rgb"]=d["rgb"][:408,:848]
    if fault=="depth_size":d["distance_to_image_plane"]=d["distance_to_image_plane"][:408,:848]
    if fault=="camera":d["rp_head"]["camera"]="/WrongCamera"
    if fault=="projection":d["camera_params"]["cameraProjection"][0]*=1.01
    if fault=="units":d["camera_params"]["metersPerSceneUnit"]=.01
    if fault=="invalid":d["distance_to_image_plane"][:]=np.inf
    if fault=="blank":d["rgb"][:]=0
    if fault=="reference":d["ReferenceTime"]["referenceTimeNumerator"]=11
    if fault=="guard":after="b"*64
    if fault=="sequence":sequence=0
    with pytest.raises(ValueError):
        if fault=="stale":validate_native_payload(d,c,[10,60])
        else:validate_native_static(d,c,before,after,sequence)


@pytest.mark.parametrize("size",[(848,408),HIRES_RESOLUTION])
def test_native_id_shape_mapping_and_copy(size):
    width,height=size
    ids=np.full((height,width),7,np.uint32)
    d={"instance_id_segmentation":{"data":ids,"info":{"idToLabels":{"7":"/World/Plant/Petiole"}}}}
    out,mapping=decode_native_instances(d,size)
    assert mapping[7]=="/World/Plant/Petiole"
    out[:]=0;assert ids[0,0]==7
    d["instance_id_segmentation"]["info"]["idToLabels"]={}
    with pytest.raises(ValueError):decode_native_instances(d,size)


def test_high_resolution_instance_buffer_cannot_masquerade_as_legacy():
    d={"instance_id_segmentation":{"data":np.zeros((816,1696),np.uint32),"info":{"idToLabels":{"0":"/World"}}}}
    with pytest.raises(ValueError):decode_native_instances(d,(848,408))
