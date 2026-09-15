"""Shared sensor proof must not silently permit scene/optics/lineage changes."""
from copy import deepcopy
import numpy as np
import pytest
from sim_data.native_multitarget_plan import assert_compatible,merge_bindings,capture_jobs,OPTICS


def fixture():
    return dict(source_family='family',split='train',split_group='family',
        variant_directory='/variant',source_collection_plan='/source',
        expected_scene_counts={'components':400},original_variant={'plant_root':'/plant'},
        expected_calibration={k: k for k in OPTICS},
        expected_robot_snapshot={'camera_to_head_column_vectors':np.eye(4).tolist(),
                                 'visual_bound_screen':{'passed':True}},
        source_row={'component_id':'petiole'},generated_row={'component_id':'petiole'})


def test_different_reference_head_pose_is_not_a_different_sensor():
    a=fixture();b=deepcopy(a)
    b['expected_robot_snapshot']['head_joints']=[20,10]
    b['expected_calibration']['camera_to_world_usd_row_vectors']=np.eye(4).tolist()
    assert_compatible(a,b)


@pytest.mark.parametrize('key',['source_family','split','split_group','variant_directory',
    'source_collection_plan','expected_scene_counts','original_variant'])
def test_reject_changed_scene_or_lineage(key):
    a=fixture();b=deepcopy(a);b[key]='changed'
    with pytest.raises(ValueError):assert_compatible(a,b)


@pytest.mark.parametrize('key',OPTICS)
def test_reject_changed_optics(key):
    a=fixture();b=deepcopy(a);b['expected_calibration'][key]='changed'
    with pytest.raises(ValueError):assert_compatible(a,b)


def test_mount_or_unscreened_reference_rejected():
    a=fixture();b=deepcopy(a);b['expected_robot_snapshot']['camera_to_head_column_vectors'][0][3]=.001
    with pytest.raises(ValueError):assert_compatible(a,b)
    b=deepcopy(a);b['expected_robot_snapshot']['visual_bound_screen']['passed']=False
    with pytest.raises(ValueError):assert_compatible(a,b)


def test_binding_conflicts_rejected():
    data={'a':'sha'};merge_bindings(data,{'a':'sha','b':'sha2'})
    assert data=={'a':'sha','b':'sha2'}
    with pytest.raises(ValueError):merge_bindings(data,{'a':'different'})


def test_single_target_legacy_jobs_unchanged():
    base=fixture();plan={'views':[{'candidate_id':'view_001'},{'candidate_id':'view_002'}]}
    assert capture_jobs(plan,base)==[(base,plan,s) for s in plan['views']]


def test_native_preflight_does_not_replay_or_import_geometry(monkeypatch):
    from sim_data import native_multitarget_plan as module
    from sim_data import generated_capture
    anchor=fixture()
    plan=dict(schema=module.SCHEMA,source_bindings={},prerequisite_bindings={},
        implementation_bindings={},anchor_pair_plan='anchor.json',resolution=[1696,816],
        split='train',requested_render_subframes_per_view=56,source_cap_reset=False,
        training_approved=False,physical_motion_commanded=False,hidden_cut_coordinates_executable=False)
    monkeypatch.setattr(module,'read_json',lambda p:anchor)
    monkeypatch.setattr(module,'verify_bindings',lambda p:None)
    monkeypatch.setattr(generated_capture,'check_plan',lambda p:None)
    def unexpected(*args,**kwargs):raise AssertionError('USD catalogue replay before SimulationApp')
    monkeypatch.setattr(module,'build',unexpected)
    assert module.check(plan,replay_geometry=False) is anchor
