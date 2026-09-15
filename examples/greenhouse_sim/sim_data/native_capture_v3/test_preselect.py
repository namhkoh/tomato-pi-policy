import copy
import numpy as np
import pytest
from sim_data.native_capture_v3.preselect import projection_screen,POLICY

def calibration():
    return dict(resolution=[1696,816],crop_resize=None,depth_convention='optical_axis_z_metres_not_ray_range',
        camera_to_world_usd_row_vectors=np.eye(4).tolist(),intrinsics=[[1000,0,848],[0,1000,408],[0,0,1]],
        clipping_range_m=[.01,20.])

def test_clear_geometry_only_is_not_visibility():
    c=calibration();p=[[0,0,-.5],[.01,0,-.5]];before=copy.deepcopy((c,p))
    r=projection_screen(c,p,.003)
    assert r['worth_rendering'] and r['projected_interval_length_px']==20 and r['estimated_nominal_diameter_px']==12
    assert not r['native_depth_computed'] and not r['native_visibility_verified'] and not r['training_approved']
    assert (c,p)==before

def test_whole_interval_not_just_nominal_is_screened():
    r=projection_screen(calibration(),[[0,0,-.5],[.5,0,-.5]],.004)
    assert not r['worth_rendering'] and not r['all_interval_samples_in_frame']

@pytest.mark.parametrize('points',[
 [[0,0,-.5],[.01,0,.1]],[[0,0,-.5],[.01,0,-21]],[[0,0,-.5],[.01,0,-.001]],
 [[0,0,.5],[.01,0,.5]]])
def test_clipped_and_behind_camera(points):
    r=projection_screen(calibration(),points,.004)
    assert not r['worth_rendering'] and 'interval_outside_native_frame_or_clipping' in r['reasons']

def test_bends_are_preserved_not_endpoint_shortcut():
    r=projection_screen(calibration(),[[0,0,-.5],[.005,.005,-.5],[0,.01,-.5]],.003)
    assert r['projected_interval_length_px']==pytest.approx(2*np.sqrt(200))
    assert len(r['projected_interval'])==3

def test_short_interval_and_small_diameter():
    r=projection_screen(calibration(),[[0,0,-2],[.01,0,-2]],.002)
    assert not r['worth_rendering'] and len(r['reasons'])==2

def test_exact_thresholds_are_inclusive():
    r=projection_screen(calibration(),[[0,0,-.5],[.006,0,-.5]],.002)
    assert r['projected_interval_length_px']==12 and r['estimated_nominal_diameter_px']==8
    assert r['worth_rendering']

@pytest.mark.parametrize('radius',[0,-1,float('nan'),float('inf'),True])
def test_invalid_radius(radius):
    with pytest.raises(ValueError):projection_screen(calibration(),[[0,0,-.5],[.01,0,-.5]],radius)

@pytest.mark.parametrize('kind',['scale','reflection','nan','bottom_row'])
def test_malformed_camera(kind):
    c=calibration();m=c['camera_to_world_usd_row_vectors']
    if kind=='scale':m[0][0]=2
    if kind=='reflection':m[0][0]=-1
    if kind=='nan':m[0][0]=float('nan')
    if kind=='bottom_row':m[0][3]=1
    with pytest.raises(ValueError):projection_screen(c,[[0,0,-.5],[.01,0,-.5]],.004)

@pytest.mark.parametrize('points',[[],[[0,0,-.5]],[[0,0,float('nan')],[0,0,-1]],[[0,1],[2,3]]])
def test_invalid_interval(points):
    with pytest.raises(ValueError):projection_screen(calibration(),points,.004)

def test_rigid_world_equivariance():
    c=calibration();p=np.array([[0,0,-.5],[.01,0,-.5]])
    expected=projection_screen(c,p,.003)
    m=np.eye(4);m[:3,:3]=[[0,1,0],[-1,0,0],[0,0,1]];m[3,:3]=[1,2,3]
    c['camera_to_world_usd_row_vectors']=m.tolist();p=(np.column_stack([p,np.ones(2)])@m)[:,:3]
    got=projection_screen(c,p,.003)
    assert got['worth_rendering']==expected['worth_rendering']
    assert got['projected_interval_length_px']==pytest.approx(expected['projected_interval_length_px'])
