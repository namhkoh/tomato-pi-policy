"""CPU batch planning boundaries, with explicit synthetic bank/generator hooks."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from . import batch_prepare_v2 as p, batch_scene_v2 as scene


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(p.oc.canonical(value))
    return dict(path=str(path), sha256=p.oc.sha256(path))


@pytest.fixture
def unit(tmp_path, monkeypatch):
    root = tmp_path / 'sources'
    clear = put(root / 'clear.json', {'UNIT_ONLY': 'current scene'})
    old = put(root / 'old.json', {'UNIT_ONLY': 'historical pose'})
    calibration = dict(camera_path=p.oc.HEAD_CAMERA, resolution=[1696, 816], intrinsics=[[900,0,848],[0,900,408],[0,0,1]],
        focal_length_mm=2., apertures_mm=[3.,2.], aperture_offsets_mm=[0.,0.], clipping_range_m=[.01,100.],
        depth_convention='optical_axis_z', crop_resize=None, camera_to_world_usd_row_vectors=np.eye(4).tolist())
    pose = dict(robot_root_to_world_usd_row_vectors=np.eye(4).tolist(), camera_to_head_column_vectors=np.eye(4).tolist(),
        joint_degrees={'head_1':0., 'head_2':0.})
    pose['robot_root_to_world_usd_row_vectors'][3][0] = 1.
    prior = dict(manifest=old, robot_snapshot=deepcopy(pose), calibration=dict(calibration, resolution=[848,408]))
    prior['robot_snapshot']['joint_degrees']['head_2'] = -10.
    entries, source_bindings = [], {clear['path']:clear['sha256'], old['path']:old['sha256']}
    for n in (41,42):
        target = f'seed73_full/SubStem_{n}'
        source = dict(target_id=target, source_plant_id='seed73_full', split_group='seed73_full', variant_id='seed73_full',
            component_id=f'SubStem_{n}', cut_region_proposal={'nominal':{'point_plant_m':[0.,0.,-1.]}})
        authority = dict(clear_plan=clear, source_row=source, package=str(root/'package'),
            policy=deepcopy(p.oc.SCENE_POLICY), actual_lighting={'dome_intensity':6000}, actual_renderer='RealTimePathTracing',
            expected_scene_counts={'components':5})
        sample = dict(schema_version=p.oc.SAMPLE_SCHEMA, training_sample_approved=False, historical_labels_inherited=False,
            calibration=calibration, robot_snapshot=pose, geometry_screen={'passed':True},
            lighting=authority['actual_lighting'], renderer=authority['actual_renderer'], supervision=dict(target_id=target,
                plant_to_world_usd_row_vectors=np.eye(4).tolist(), nominal_world_m=[0.,0.,-1.],
                interval_world_m=[[0.,0.,-1.],[.02,0.,-1.]]))
        sample_pin = put(root/f'native/sample_{n}/sample.json', sample)
        source_bindings[sample_pin['path']] = sample_pin['sha256']
        entries.append(dict(id=f'UNIT_{n}', source_family='seed73_full', target_id=target, split='train',
            reference_status={'usable_as_original_reference':True}, native_decision=p.bank_api.inv.STRICT,
            native_observation=dict(sample=sample_pin, calibration=calibration, robot_snapshot=pose),
            compatibility_group='UNIT_GROUP', compatibility_basis={'UNIT':True},
            pose_prior=deepcopy(prior), scene_authority=authority))
    bank = dict(schema=p.bank_api.SCHEMA, entries=entries, source_bindings=source_bindings)
    bank_pin = put(root/'bank.json', bank)
    events, catalogues = [], {}
    def verified(value):
        assert value == bank, 'Changed synthetic bank'
        events.append('bank_replay')
        return deepcopy(value)
    def generate(clear_path, family, components, seed, directory):
        assert clear_path == clear['path'] and family == 'seed73_full'
        events.append(('generate', list(components)))
        rows, recipes = [], []
        for i, key in enumerate(components):
            original = next(e['scene_authority']['source_row'] for e in entries if e['scene_authority']['source_row']['component_id']==key)
            row = deepcopy(original)
            row.update(target_id=directory.name+'/'+key, variant_id=directory.name,
                conservative_view_cap_group=original['target_id'], cut_region_proposal={'nominal':{'point_plant_m':[.02,i*.03,-1.]}})
            rows.append(row)
            recipes.append(dict(seed=seed+i*104729, source_target_id=original['target_id']))
        manifest = put(directory/'manifest.json', {'UNIT_ONLY':'not generated geometry'})
        put(directory/'qualification.json', dict(version='curved_relocated_rigid_leaf_static.v2', recipes=recipes,
            source_plan_path=clear['path'], source_plan_sha256=clear['sha256'], output_hashes={'manifest.json':manifest['sha256']}))
        catalogues[str(directory)] = dict(rows=rows, variant_id=directory.name, independent_target_novelty_approved=False,
            texture_bindings={})
    monkeypatch.setattr(p.bank_api, 'check_bank', verified)
    monkeypatch.setattr(p, '_generate', generate)
    monkeypatch.setattr(p, '_catalogue', lambda directory, clear: deepcopy(catalogues[str(directory)]))
    return dict(root=root, output=tmp_path/'prepared', bank=bank, bank_pin=bank_pin, entries=entries,
        events=events, catalogues=catalogues)


def build(f, ids=None, **kwargs):
    return p.prepare(f['bank_pin']['path'], bank_sha256=f['bank_pin']['sha256'],
        entry_ids=ids if ids is not None else [e['id'] for e in f['entries']], seed=730201, output=f['output'], **kwargs)


def test_two_targets_twelve_views_one_generated_plant_with_native_pose(unit):
    result=build(unit)
    plan=p.oc.read_json(result['plan_path'])
    p.check_plan(plan)
    assert result['target_count']==2 and result['maximum_future_native_frames']==12
    assert [len(c['views']) for c in plan['target_cases']]==[6,6]
    assert len({v['candidate_id'] for c in plan['target_cases'] for v in c['views']})==12
    assert unit['events'].count(('generate',['SubStem_41','SubStem_42']))==1
    assert plan['expected_robot_snapshot']==unit['entries'][0]['native_observation']['robot_snapshot']
    assert plan['expected_robot_snapshot']!=plan['pose_prior']['robot_snapshot']
    assert plan['resolution']==[1696,816] and plan['render_subframes_per_view']==56
    assert all(plan[k] is v for k,v in p.FLAGS.items())
    assert not any(k in plan for k in ('old_manifest','prerequisite_directory','base_pair_plan'))


@pytest.mark.parametrize('mode', ['unreviewed','same_target','scene','mount','unknown','duplicate'])
def test_invalid_reference_selection_rejected_before_generation(unit, mode):
    ids=[e['id'] for e in unit['entries']]
    second=unit['entries'][1]
    if mode=='unreviewed': second['reference_status']['usable_as_original_reference']=False
    elif mode=='same_target': second['target_id']=unit['entries'][0]['target_id']
    elif mode=='scene': second['compatibility_group']='OTHER'
    elif mode=='mount':
        second['native_observation']=deepcopy(second['native_observation'])
        second['native_observation']['robot_snapshot']['camera_to_head_column_vectors'][0][3]=.001
    elif mode=='unknown': ids[1]='UNKNOWN'
    elif mode=='duplicate': ids[1]=ids[0]
    unit['bank_pin']=put(Path(unit['bank_pin']['path']),unit['bank'])
    with pytest.raises(ValueError): build(unit,ids)
    assert not unit['output'].exists() and not any(isinstance(e,tuple) for e in unit['events'])


@pytest.mark.parametrize('key,value', [('resolution',[848,408]),('render_subframes_per_view',8),
    ('maximum_native_frames',13),('source_cap_reset',True),('native_launch_supported',True),('native_instance_backend','fast')])
def test_saved_plan_cannot_change_policy_or_gain_authority(unit,key,value):
    result=build(unit)
    plan=p.oc.read_json(result['plan_path']);plan[key]=value
    with pytest.raises(ValueError):p.check_plan(plan)


def test_current_clear_plan_and_recipe_order_are_bound(unit):
    result=build(unit);plan=p.oc.read_json(result['plan_path'])
    path=Path(plan['variant_directory'])/'qualification.json'
    value=p.oc.read_json(path);value['recipes'].reverse()
    changed=put(path,value)
    plan['source_bindings'][str(path)]=changed['sha256']
    with pytest.raises(ValueError,match='recipes must match'):p.check_plan(plan)


def test_create_only_and_capture_protection(unit):
    build(unit)
    with pytest.raises(ValueError):build(unit)
    unit['output']=unit['root']/'native/new_output'
    with pytest.raises(ValueError):build(unit)
    assert not unit['output'].exists()


def test_scene_rejects_uninitialized_application_before_native_imports():
    with pytest.raises(ValueError,match='Initialized SimulationApp'):
        scene.prepare_scene(None,{})
