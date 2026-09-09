from copy import deepcopy
import json

import numpy as np
import pytest

from sim_data.capture_contract import (depth_evidence, fingerprint, project, transform_points,
                                       validate_payload, validate_static_capture, write_sample)


def fixture():
    camera = np.eye(4)
    camera[3, :3] = [1, 2, 3]
    cal = {"camera_path":"/World/RBY1/head/Camera", "resolution":[848,408],
           "intrinsics":[[500,0,410],[0,520,210],[0,0,1]],
           "camera_to_world_usd_row_vectors":camera.tolist(),"clipping_range_m":[.01,100]}
    projection = np.zeros((4,4))
    projection[0,0],projection[1,1] = 1000/848,1040/408
    projection[2,0],projection[2,1] = 1-820/848,420/408-1
    projection[2,2],projection[2,3],projection[3,2] = -1.0002,-1,-.02
    rgb = np.zeros((408,848,4),dtype=np.uint8)
    rgb[:,:,0] = np.arange(848,dtype=np.uint16)%255
    rgb[:,:,3] = 255
    data = {"rgb":rgb,"distance_to_image_plane":np.full((408,848),2,np.float32),
            "ReferenceTime":{"referenceTimeNumerator":10,"referenceTimeDenominator":60},
            "reference_time":[10,60],
            "rp_head":{"camera":cal["camera_path"],"resolution":[848,408]},
            "camera_params":{"renderProductResolution":[848,408],"metersPerSceneUnit":1.,
                             "cameraViewTransform":np.linalg.inv(camera).reshape(-1),
                             "cameraProjection":projection.reshape(-1)}}
    return data,cal


def test_projection_optical_frame_and_nonzero_principal_point():
    _,cal=fixture()
    points=transform_points([[0,0,-2],[.2,.1,-2],[0,0,1],[10,0,-1]],cal["camera_to_world_usd_row_vectors"])
    result=project(points,cal)
    assert result[0]["pixel_xy"]==[410,210]
    assert np.allclose(result[1]["pixel_xy"],[460,184])
    assert result[1]["camera_optical_xyz_m"]==pytest.approx([.2,-.1,2])
    assert result[2]["projection_status"]=="behind_camera"
    assert result[3]["projection_status"]=="out_of_frame"
    # Back-projection with Z gives the original point, not a ray-range point.
    optical=np.linalg.solve(cal["intrinsics"],[*result[1]["pixel_xy"],1])*2
    restored=transform_points([optical*[1,-1,-1]],cal["camera_to_world_usd_row_vectors"])[0]
    assert np.allclose(restored,points[1])


def test_complete_payload_raw_invalid_depth_and_buffer_copies():
    data,cal=fixture()
    data["distance_to_image_plane"][0,:5]=[0,np.inf,np.nan,-1,101]
    rgb,depth,valid,reference=validate_payload(data,cal,[9,60])
    assert rgb.shape==(408,848,3) and reference==[10,60]
    assert not valid[0,:5].any() and valid[1:].all()
    assert np.isnan(depth[0,2]) and np.isinf(depth[0,1])
    rgb[:]=0
    depth[:]=0
    assert data["rgb"].std()>0 and data["distance_to_image_plane"][1,1]==2


@pytest.mark.parametrize("kind",["missing_depth","stale","mismatched_reference","camera","extra_product",
                                "rgb_shape","rgb_type","blank","depth_shape","depth_type","all_invalid",
                                "units","resolution","camera_pose","projection"])
def test_bad_payload_rejected(kind):
    data,cal=fixture()
    previous=[9,60]
    if kind=="missing_depth": del data["distance_to_image_plane"]
    if kind=="stale": previous=[10,60]
    if kind=="mismatched_reference": data["ReferenceTime"]["referenceTimeNumerator"]=11
    if kind=="camera": data["rp_head"]["camera"]="/FloatingDiagnostic"
    if kind=="extra_product": data["rp_other"]=deepcopy(data["rp_head"])
    if kind=="rgb_shape": data["rgb"]=data["rgb"][:100]
    if kind=="rgb_type": data["rgb"]=data["rgb"].astype(np.float32)
    if kind=="blank": data["rgb"][:]=0
    if kind=="depth_shape": data["distance_to_image_plane"]=data["distance_to_image_plane"][:100]
    if kind=="depth_type": data["distance_to_image_plane"]=data["distance_to_image_plane"].astype(np.float64)
    if kind=="all_invalid": data["distance_to_image_plane"][:]=np.inf
    if kind=="units": data["camera_params"]["metersPerSceneUnit"]=.01
    if kind=="resolution": data["camera_params"]["renderProductResolution"]=[1280,720]
    if kind=="camera_pose": data["camera_params"]["cameraViewTransform"][12]+=.1
    if kind=="projection": data["camera_params"]["cameraProjection"][0]*=1.01
    with pytest.raises(ValueError): validate_payload(data,cal,previous)


def test_depth_evidence_is_conservative_not_visibility_approval():
    data,cal=fixture()
    _,depth,valid,_=validate_payload(data,cal)
    point={"pixel_xy":[410.2,210.6],"camera_optical_xyz_m":[0,0,2],"projection_status":"in_frame"}
    assert depth_evidence(point,depth,valid,.002)["status"]=="depth_consistent_not_visibility_verified"
    depth[210,410]=1.8
    assert depth_evidence(point,depth,valid,.002)["status"]=="foreground_occlusion_evidence"
    depth[210,410]=2.1
    assert depth_evidence(point,depth,valid,.002)["status"]=="unknown_no_target_surface_at_pixel"
    valid[210,410]=False
    assert depth_evidence(point,depth,valid,.002)["status"]=="unknown_invalid_depth"
    assert depth_evidence({**point,"projection_status":"out_of_frame"},depth,valid,.002)["status"]=="out_of_frame"


def test_files_separate_inputs_from_overlay_and_refuse_overwrite(tmp_path):
    from PIL import Image
    data,cal=fixture()
    rgb,depth,valid,_=validate_payload(data,cal)
    point={"pixel_xy":[410.,210.],"projection_status":"in_frame","camera_optical_xyz_m":[0,0,2]}
    meta={"training_sample_approved":False,"supervision":{"nominal_projected":point,
          "projected_interval":[point,{**point,"pixel_xy":[418.,210.]}],
          "depth_evidence":depth_evidence(point,depth,valid,.002)}}
    before=rgb.copy()
    directory=tmp_path/"sample"
    result=write_sample(directory,rgb,depth,valid,meta)
    assert np.array_equal(before,rgb)
    assert np.array_equal(np.asarray(Image.open(directory/"inputs/rgb.png")),rgb)
    assert not np.array_equal(np.asarray(Image.open(directory/"review/overlay.png")),rgb)
    assert np.array_equal(np.load(directory/"inputs/depth_m.npy"),depth)
    assert result["files"]["review/overlay.png"]["role"]=="review_only"
    assert json.loads((directory/"sample.json").read_text())["training_sample_approved"] is False
    with pytest.raises(FileExistsError): write_sample(directory,rgb,depth,valid,meta)


def test_geometry_rejects_nan_and_wrong_matrix_convention():
    with pytest.raises(ValueError): transform_points([[np.nan,0,0]],np.eye(4))
    bad=np.eye(4); bad[0,3]=1
    with pytest.raises(ValueError): transform_points([[0,0,0]],bad)
    with pytest.raises(ValueError): fingerprint({"bad":float("nan")})


@pytest.mark.parametrize("fault",[None,"guard","sequence","pose","rgb","depth"])
def test_static_capture_uses_pose_and_both_buffers_not_fabric_clock(fault):
    data,cal=fixture()
    guard=fingerprint({"scene":"static"})
    *_,previous=validate_static_capture(data,cal,guard,guard,1)
    # The static clock remains the same, while both pose and rendered observations change.
    cal["camera_to_world_usd_row_vectors"][3][0]+=.1
    data["camera_params"]["cameraViewTransform"]=np.linalg.inv(cal["camera_to_world_usd_row_vectors"]).reshape(-1)
    data["rgb"][20,20,0]+=1
    data["distance_to_image_plane"][20,20]=2.4
    after,sequence=guard,2
    if fault=="guard": after="0"*64
    if fault=="sequence": sequence=1
    if fault=="pose": previous["camera_sha256"]=fingerprint(cal)
    if fault=="rgb": previous["rgb_sha256"]=__import__("hashlib").sha256(data["rgb"][:,:,:3].copy().tobytes()).hexdigest()
    if fault=="depth": previous["depth_sha256"]=__import__("hashlib").sha256(data["distance_to_image_plane"].tobytes()).hexdigest()
    if fault:
        with pytest.raises(ValueError): validate_static_capture(data,cal,guard,after,sequence,previous)
    else:
        *_,token=validate_static_capture(data,cal,guard,after,sequence,previous)
        assert token["callback_sequence"]==2
