"""Use actual original-native pose/screen evidence for existing bounded poses.

The old pose helper expects a visual_bound_screen field. Original-native pose
records store their actual static screen separately. Construct a runtime input
from that authenticated pair; do not rewrite or relabel historical metadata.
"""
from copy import deepcopy
import numpy as np

from ..native_original_capture import contracts as oc
from ..native_view_pose import bounded_reference_root

OPTICS = ('camera_path', 'resolution', 'intrinsics', 'focal_length_mm',
    'apertures_mm', 'aperture_offsets_mm', 'clipping_range_m', 'depth_convention', 'crop_resize')


def screened_reference(case):
    native = case['native_reference']
    oc.require(native['robot_snapshot'] == case['expected_robot_snapshot']
        and native['calibration'] == case['expected_calibration']
        and native['geometry_screen']['passed'] is True
        and native['audit_replayed_by_this_build'] is True,
        'Authenticated native pose and successful scene screen required')
    reference = deepcopy(case['expected_robot_snapshot'])
    reference['visual_bound_screen'] = deepcopy(native['geometry_screen'])
    return reference


def reference_evidence(case):
    screened_reference(case)
    native = case['native_reference']
    return dict(method='fresh_original_native_pose_and_separate_static_screen.v2',
        sample=deepcopy(native['sample']), native_plan=deepcopy(native['native_plan']),
        geometry_screen_sha256=oc.digest(oc.canonical(native['geometry_screen'])),
        historical_pose_metadata_rewritten=False, original_source_cap_reset=False)


def verify_pose(metadata, case, spec, target_world):
    """Reconstruct bounded base, unchanged joints/mount and actual native FK."""
    from ..capture_contract import project
    from ..native_original_capture.audit import verify_camera
    reference = screened_reference(case)
    expected_root = bounded_reference_root(reference, spec, target_world).T
    pose, cal = metadata['robot_snapshot'], metadata['calibration']
    oc.require(np.allclose(pose['robot_root_to_world_usd_row_vectors'], expected_root, atol=1e-7, rtol=0),
        'Captured base differs from bounded reference position, height or heading')
    oc.require(np.allclose(pose['camera_to_head_column_vectors'], reference['camera_to_head_column_vectors'],
        atol=1e-9, rtol=0), 'Captured camera mount changed')
    actual, original = pose['joint_degrees'], reference['joint_degrees']
    oc.require(set(actual) == set(original)
        and all(actual[k] == value for k, value in original.items() if k not in ('head_0','head_1')),
        'Captured arm or torso joints changed')
    oc.require(pose['desired_cut_pixel_xy'] == spec['desired_pixel_xy']
        and pose['pose_sampling'] == 'bounded_reference_offsets_and_real_head_joints.v1'
        and pose['source_body_orientation_preserved'] is True and pose['arm_and_torso_joints_preserved'] is True
        and pose['joint_limits_checked'] is True and pose['motion_between_snapshots_validated'] is False
        and metadata['pose_reference_evidence'] == reference_evidence(case), 'Captured pose provenance differs')
    oc.require(all(cal[k] == case['expected_calibration'][k] for k in OPTICS)
        and cal['resolution'] == [1696,816] and spec['framing_reference_resolution'] == [848,408],
        'Native optical calibration or proposal coordinate system changed')
    projected = project([target_world], cal)[0]
    oc.require(projected['projection_status'] == 'in_frame'
        and np.linalg.norm(np.asarray(projected['pixel_xy']) - 2*np.asarray(spec['desired_pixel_xy'])) < 4,
        'Actual native framing differs from the mounted-pose proposal')
    verify_camera(metadata)  # Includes all URDF joint limits and renderer projection.
    return True
