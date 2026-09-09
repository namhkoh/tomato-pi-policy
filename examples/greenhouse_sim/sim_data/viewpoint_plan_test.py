from copy import deepcopy
import json
import subprocess
import sys

import numpy as np
import pytest
from pxr import Usd, UsdGeom

from greenhouse_sim.robot_kinematics import Rby1Kinematics
from greenhouse_sim.robot_model import DEFAULT_ASSET
from sim_data.capture_scene import calibration, mounted_camera_to_head, set_snapshot_pose
from sim_data.capture_viewpoints import screen_bounds
from sim_data.depth_preview import sha256
from sim_data.robot_preview import add_robot_preview
from sim_data.viewpoint_plan import focus_specs, load_plan, select_planned, SCHEMA


def test_focus_grid_changes_actual_robot_pose_with_bounded_metre_offsets():
    specs = focus_specs(.8, .035)
    assert len(specs) == 90
    assert len({(r['root_x_m'], r['y_offset_m'], r['root_yaw_degrees']) for r in specs}) == 90
    assert min(r['root_x_m'] for r in specs) >= .4
    assert all(r['root_x_m']-.035 >= .35 for r in specs)
    assert {r['root_yaw_degrees'] for r in specs} == {150., 180., 210.}
    assert all(0 < r['desired_pixel_xy'][0] < 848 and 0 < r['desired_pixel_xy'][1] < 408 for r in specs)
    assert all(r['root_x_m']-.15 >= .35 for r in focus_specs(.8, .15))


def test_plan_validation_import_is_safe_before_simulation_app_startup():
    code = "import sys; import sim_data.viewpoint_plan; assert not any(k=='pxr.Usd' or k.startswith('omni.') for k in sys.modules)"
    subprocess.run([sys.executable, '-c', code], check=True, timeout=20)


@pytest.mark.parametrize('margin', [0, .01])
def test_touching_bounds_not_cleared_at_zero_margin(margin):
    robot = [{'path': 'robot', 'min': [0, 0, 0], 'max': [1, 1, 1]}]
    obstacle = [{'path': 'leaf', 'min': [1, 0, 0], 'max': [2, 1, 1]}]
    assert not screen_bounds(robot, obstacle, margin_m=margin)['passed']


def test_plan_selection_retains_base_or_yaw_diversity():
    rows = [{**s, 'predicted_diameter_px': 4., 'predicted_interval_px': 10.,
             'base_xy_m': [s['root_x_m'], s['y_offset_m']]} for s in focus_specs(.8, .03)]
    chosen = select_planned(rows)
    assert len(chosen) == 6 and len({r['candidate_id'] for r in chosen}) == 6
    assert len({r['root_yaw_degrees'] for r in chosen}) > 1
    assert select_planned([]) == []


@pytest.fixture
def saved_plan(tmp_path):
    draft = tmp_path / 'draft.json'
    draft.write_text('{}')
    spec = focus_specs(.8, .03)[0]
    plan = {'schema_version': SCHEMA, 'state': 'ready_for_rendered_visibility_check',
            'draft_sha256': sha256(draft), 'package': str(tmp_path), 'original_base_x_m': .8,
            'source_usd_sha256': {str(draft): sha256(draft)}, 'target_world_m': {'B03': [.03, .5, 1.3]},
            'selected': {'B03': [{**spec, 'state': 'geometry_screen_passed_visibility_unknown',
                                'visual_bound_screen': {'passed': True}}]}}
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(plan))
    return path, draft, tmp_path, plan


def test_valid_plan_requires_rendered_visibility_not_approval(saved_plan):
    path, draft, package, plan = saved_plan
    assert load_plan(path, draft, package) == plan


@pytest.mark.parametrize('kind', ['draft', 'yaw', 'distance', 'pixel', 'state', 'duplicate', 'unscreened', 'target'])
def test_stale_or_unbounded_plans_rejected(saved_plan, kind):
    path, draft, package, plan = saved_plan
    row = plan['selected']['B03'][0]
    if kind == 'draft': plan['draft_sha256'] = 'stale'
    if kind == 'yaw': row['root_yaw_degrees'] = 90
    if kind == 'distance': row['root_x_m'] = .01
    if kind == 'pixel': row['desired_pixel_xy'] = [424, 204]
    if kind == 'state': plan['state'] = 'planning'
    if kind == 'duplicate': plan['selected']['B03'].append(deepcopy(row))
    if kind == 'unscreened': row['visual_bound_screen']['passed'] = False
    if kind == 'target': plan['selected']['B01'] = [row]
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError): load_plan(path, draft, package)


@pytest.mark.skipif(not DEFAULT_ASSET.exists(), reason='Model A v1.2 asset needed')
def test_base_yaw_preserves_fixed_head_mount_and_arm_torso_joints(monkeypatch):
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, 'Z')
    stage.SetEditTarget(stage.GetSessionLayer())
    robot = add_robot_preview(stage, gutter_x=-.2, right_tool='gripper')
    monkeypatch.setattr('sim_data.floor_alignment.align_robot_to_floor', lambda *_: {'test': True})
    mount = mounted_camera_to_head(stage).copy()
    source = stage.GetRootLayer().ExportToString()
    pose = set_snapshot_pose(stage, robot, [.03, .4, 1.3], .2, [350, 220], root_x_m=.55, root_yaw_degrees=150)
    assert np.allclose(mounted_camera_to_head(stage), mount, rtol=0, atol=1e-9)
    assert all(pose['joint_degrees'][k] == v for k, v in robot['pose_degrees'].items() if not k.startswith('head_'))
    root = np.asarray(pose['robot_root_to_world_usd_row_vectors']).T
    expected = root @ Rby1Kinematics().all_link_transforms(pose['joint_degrees'])['link_head_2'] @ mount
    assert np.allclose(np.asarray(calibration(stage)['camera_to_world_usd_row_vectors']).T, expected, atol=1e-9)
    assert stage.GetRootLayer().ExportToString() == source
    with pytest.raises(ValueError, match='yaw'):
        set_snapshot_pose(stage, robot, [.03, .4, 1.3], 0, [350, 220], root_yaw_degrees=100)
