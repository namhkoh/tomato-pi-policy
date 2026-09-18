"""CPU bounded-pose checks; FK helper mocked only in synthetic pose fixtures."""
from copy import deepcopy
import numpy as np
import pytest
from . import batch_pose_v2 as p


@pytest.fixture
def pose(monkeypatch):
    from ..native_original_capture import audit
    root=np.eye(4);root[3,0]=1.
    reference=dict(robot_root_to_world_usd_row_vectors=root.tolist(), joint_degrees={'head_0':0.,'head_1':0.,'arm':5.},
        camera_to_head_column_vectors=np.eye(4).tolist())
    cal=dict(camera_path=p.oc.HEAD_CAMERA,resolution=[1696,816],intrinsics=[[900,0,848],[0,900,408],[0,0,1]],
        focal_length_mm=2.,apertures_mm=[3.,2.],aperture_offsets_mm=[0.,0.],clipping_range_m=[.01,100.],
        depth_convention='optical_axis_z_metres_not_ray_range',crop_resize=None,
        camera_to_world_usd_row_vectors=np.eye(4).tolist())
    native=dict(robot_snapshot=reference,calibration=cal,geometry_screen={'passed':True,'UNIT':True},
        audit_replayed_by_this_build=True,sample={'path':'UNIT/sample.json','sha256':'1'*64},
        native_plan={'path':'UNIT/plan.json','sha256':'2'*64})
    case=dict(native_reference=native,expected_robot_snapshot=reference,expected_calibration=cal)
    spec=dict(root_x_m=1.,y_offset_m=0.,root_yaw_degrees=0.,base_heading_preserved=True,
        desired_pixel_xy=[424.,204.],framing_reference_resolution=[848,408])
    actual=deepcopy(reference)
    actual.update(desired_cut_pixel_xy=spec['desired_pixel_xy'],pose_sampling='bounded_reference_offsets_and_real_head_joints.v1',
        source_body_orientation_preserved=True,arm_and_torso_joints_preserved=True,joint_limits_checked=True,
        motion_between_snapshots_validated=False)
    meta=dict(robot_snapshot=actual,calibration=deepcopy(cal),pose_reference_evidence=p.reference_evidence(case))
    calls=[]
    monkeypatch.setattr(audit,'verify_camera',lambda m:calls.append(m))
    return case,spec,meta,calls


def test_runtime_screen_mapping_preserves_original_record_and_verifies_actual_pose(pose):
    case,spec,meta,calls=pose
    before=deepcopy(case)
    mapped=p.screened_reference(case)
    assert mapped['visual_bound_screen']==case['native_reference']['geometry_screen']
    assert 'visual_bound_screen' not in case['expected_robot_snapshot'] and case==before
    assert p.verify_pose(meta,case,spec,[0.,0.,-1.]) and calls==[meta]


@pytest.mark.parametrize('mode',['base_x','base_z','body_joint','mount','framing','resolution','provenance'])
def test_changed_pose_or_optics_cannot_pass(pose,mode):
    case,spec,meta,calls=pose
    actual=meta['robot_snapshot']
    if mode=='base_x':actual['robot_root_to_world_usd_row_vectors'][3][0]+=.01
    elif mode=='base_z':actual['robot_root_to_world_usd_row_vectors'][3][2]+=.01
    elif mode=='body_joint':actual['joint_degrees']['arm']+=1
    elif mode=='mount':actual['camera_to_head_column_vectors'][0][3]+=.001
    elif mode=='framing':meta['calibration']['camera_to_world_usd_row_vectors'][3][0]=.1
    elif mode=='resolution':meta['calibration']['resolution']=[848,408]
    elif mode=='provenance':meta['pose_reference_evidence']['geometry_screen_sha256']='0'*64
    with pytest.raises(ValueError):p.verify_pose(meta,case,spec,[0.,0.,-1.])
    assert not calls


def test_failed_native_screen_cannot_be_replaced_with_historical_flag(pose):
    case,_,_,_=pose
    case['native_reference']['geometry_screen']['passed']=False
    case['expected_robot_snapshot']['visual_bound_screen']={'passed':True}
    with pytest.raises(ValueError):p.screened_reference(case)
