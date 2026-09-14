"""Native resolution changes affect pixels, not optics, geometry or metric Z."""
from copy import deepcopy
import numpy as np
import pytest
from .capture_contract import project
from .capture_sensor import (LEGACY_RESOLUTION, HIRES_RESOLUTION,
    checked_resolution, sensor_profile, calibration_for_native_resolution,
    normalized_coordinates)


def example_calibration():
    focal, ax, ay = 24., 20.955, 20.955*408/848
    ox, oy = .13, -.07
    return dict(resolution=[848,408], intrinsics=[
        [848*focal/ax,0,848*(.5-ox/ax)],
        [0,408*focal/ay,408*(.5+oy/ay)],[0,0,1]],
        camera_to_world_usd_row_vectors=np.eye(4).tolist(),
        focal_length_mm=focal,apertures_mm=[ax,ay],aperture_offsets_mm=[ox,oy],
        clipping_range_m=[.01,100.],crop_resize=None,
        depth_convention="optical_axis_z_metres_not_ray_range")


@pytest.mark.parametrize("bad",[(1696,408),(1280,720),(848.,408),[True,408],
                                "1696x816",None,[],[1696,816,3]])
def test_only_declared_native_modes(bad):
    with pytest.raises(ValueError): checked_resolution(bad)


@pytest.mark.parametrize("resolution",[LEGACY_RESOLUTION,HIRES_RESOLUTION])
def test_native_profile_is_explicit_not_hardware_claim(resolution):
    p=sensor_profile(resolution)
    assert p["resolution"]==list(resolution)
    assert not p["rgb_resized"] and not p["hardware_resolution_validated"]
    assert p["depth_source"]=="native_distance_to_image_plane"


def test_native_calibration_preserves_scene_and_optics_and_input():
    low=example_calibration(); original=deepcopy(low)
    high=calibration_for_native_resolution(low,HIRES_RESOLUTION)
    assert low==original
    assert high["intrinsics"][:2]==(np.asarray(low["intrinsics"])[:2]*2).tolist()
    assert high["intrinsics"][2]==[0,0,1]
    for k in set(low)-{"intrinsics","resolution"}: assert high[k]==low[k]
    points=[[0,0,-1],[.1,.1,-1.2],[-.2,-.1,-2.]]
    for a,b in zip(project(points,low),project(points,high)):
        assert np.allclose(np.asarray(a["pixel_xy"])*2,b["pixel_xy"])
        assert a["camera_optical_xyz_m"]==b["camera_optical_xyz_m"]
        assert a["projection_status"]==b["projection_status"]
    assert calibration_for_native_resolution(high,LEGACY_RESOLUTION)==low


def test_unchanged_legacy_calibration_is_exact():
    cal=example_calibration()
    assert calibration_for_native_resolution(cal,LEGACY_RESOLUTION)==cal


@pytest.mark.parametrize("field,value",[
    ("crop_resize",{"scale":2}),("depth_convention","ray_range"),
    ("intrinsics",np.eye(3).tolist()),("focal_length_mm",0),
    ("apertures_mm",[0,1]),("aperture_offsets_mm",[0]),
    ("camera_to_world_usd_row_vectors",np.diag([-1,1,1,1]).tolist())])
def test_reject_bad_or_pre_resized_calibration(field,value):
    cal=example_calibration();cal[field]=value
    with pytest.raises(ValueError):calibration_for_native_resolution(cal,HIRES_RESOLUTION)


@pytest.mark.parametrize("resolution",[LEGACY_RESOLUTION,HIRES_RESOLUTION])
def test_coordinate_roundtrip_has_no_pixel_or_depth_resampling(resolution):
    points=np.asarray([[.5,.5],[resolution[0]-.5,resolution[1]-.5],[300.25,210.75]])
    normalized=normalized_coordinates(points,resolution)
    assert np.allclose(normalized_coordinates(normalized,resolution,inverse=True),points)
    assert np.allclose(normalized_coordinates([resolution[0]/2,resolution[1]/2],resolution),[500,500])


@pytest.mark.parametrize("points",[[1696,2],[2,816],[-1,2],[np.nan,2],[],[1,2,3]])
def test_coordinates_refuse_outside_frame(points):
    with pytest.raises(ValueError):normalized_coordinates(points,HIRES_RESOLUTION)


def test_normalized_upper_edge_is_not_a_pixel():
    with pytest.raises(ValueError):normalized_coordinates([1000,500],HIRES_RESOLUTION,inverse=True)
