"""CPU seam fixtures, not native proof. Only NEW adapter hooks are stubbed.

The real original bank/proof replay has its own no-mock seed73 test. Here synthetic
qualification JSON tests orchestration/joins; no plants, images or Kit are created.
"""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import sys

import pytest

from . import prepare as p, scene
from ..native_original_capture import scene as original_scene
from .. import native_scene


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__('json').dumps(value), encoding='utf-8')
    return dict(path=str(path), sha256=p.oc.sha256(path))


@pytest.fixture
def unit(tmp_path, monkeypatch):
    root = tmp_path/'inputs'
    clear = write(root/'clear.json', {'UNIT_ONLY': 'current clear authority'})
    old = write(root/'old.json', {'UNIT_ONLY': 'historical pose authority'})
    source = dict(source_plant_id='seed73_full', split_group='seed73_full', variant_id='seed73_full',
        component_id='SubStem_41', target_id='seed73_full/SubStem_41',
        cut_region_proposal={'nominal': {'point_plant_m': [0., 0., -1.]}})
    matrix = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    cal = dict(resolution=[1696,816], intrinsics=[[1000,0,848],[0,1000,408],[0,0,1]],
        camera_to_world_usd_row_vectors=matrix, clipping_range_m=[.01,10])
    pose = dict(joint_degrees={'UNIT_ONLY': 0}, robot_root_to_world_usd_row_vectors=matrix,
                camera_to_head_column_vectors=matrix)
    light = dict(day=172, minutes=780, dome_intensity=6000)
    sample = dict(schema_version=p.oc.SAMPLE_SCHEMA, calibration=cal, robot_snapshot=pose,
        lighting=light, renderer='RealTimePathTracing', supervision=dict(target_id=source['target_id'],
        plant_to_world_usd_row_vectors=matrix, nominal_world_m=[0,0,-1], interval_world_m=[[0,0,-1],[.01,0,-1]]))
    sample_pin = write(root/'native/sample.json', sample)
    bank_pin = write(root/'bank.json', dict(source_bindings={clear['path']:clear['sha256'],
        old['path']:old['sha256'], sample_pin['path']:sample_pin['sha256']}))
    authority = dict(clear_plan=clear, source_row=source, package=str(root/'package'),
        policy=deepcopy(p.oc.SCENE_POLICY), actual_lighting=light, actual_renderer='RealTimePathTracing',
        expected_scene_counts={'components':2})
    proof = dict(schema=p.bank_api.PROOF_SCHEMA, bank=bank_pin, entry_id='UNIT_ANCHOR', target_id=source['target_id'],
        scene_authority=authority, pose_prior=dict(robot_snapshot=pose, manifest=old),
        native_observation=dict(sample=sample_pin, calibration=cal, robot_snapshot=pose),
        paired_848_1696_proof=False, training_approved=False, source_cap_reset=False)
    events, catalogues = [], {}
    def anchor(path, *, bank_sha256, entry_id):
        assert str(Path(path)) == bank_pin['path'] and bank_sha256 == bank_pin['sha256'] and entry_id == 'UNIT_ANCHOR'
        events.append('anchor'); return deepcopy(proof)
    def generate(clear_path, family, component, seed, directory):
        assert clear_path == clear['path'] and family == 'seed73_full' and component == 'SubStem_41'
        events.append('generate')
        row = deepcopy(source)
        row.update(target_id=directory.name+'/SubStem_41', variant_id=directory.name,
            conservative_view_cap_group=source['target_id'], cut_region_proposal={'nominal': {'point_plant_m':[.02,0,-1]}})
        manifest = write(directory/'manifest.json', {'UNIT_ONLY':'not generated geometry'})
        write(directory/'qualification.json', dict(version='curved_relocated_rigid_leaf_static.v2',
            recipes=[dict(seed=seed, source_target_id=source['target_id'])],
            source_plan_path=clear['path'], source_plan_sha256=clear['sha256'],
            output_hashes={'manifest.json':manifest['sha256']}))
        catalogues[str(directory)] = dict(rows=[row], variant_id=directory.name,
            independent_target_novelty_approved=False, texture_bindings={})
    def variant(directory, clear_path):
        assert clear_path == clear['path']; events.append('catalogue')
        return deepcopy(catalogues[str(directory)])
    monkeypatch.setattr(p.bank_api, 'verify_anchor', anchor)
    monkeypatch.setattr(p, '_generate', generate)
    monkeypatch.setattr(p, '_variant', variant)
    return dict(root=root, output=tmp_path/'new_prepared', proof=proof, bank=bank_pin, events=events, catalogues=catalogues)


def build(u):
    return p.prepare(u['bank']['path'], bank_sha256=u['bank']['sha256'], entry_id='UNIT_ANCHOR', seed=730001, output=u['output'])


def test_cpu_prepare_and_saved_plan_replay_keep_two_authorities(unit):
    out = build(unit); plan = p.oc.read_json(out['plan_path']); p.check_plan(plan)
    assert unit['events'][0:2] == ['anchor','generate']
    assert plan['schema'] == p.SCHEMA and plan['sample_count_limit'] == 2
    assert plan['modes'] == ['original_control','generated_variant'] and plan['render_subframes_per_view'] == 56
    assert plan['pose_prior']['manifest']['path'] != plan['scene_authority']['clear_plan']['path']
    assert plan['expected_robot_snapshot'] == unit['proof']['native_observation']['robot_snapshot']
    assert plan['resolution'] == [1696,816] and plan['native_instance_backend'] == 'legacy'
    assert plan['execution'] == p.EXECUTION and not plan['execution']['native_launch_supported']
    assert all(plan[k] == v for k,v in p.FLAGS.items())
    assert not any(k in plan for k in ('source_capture','source_sample','old_manifest','prerequisite_directory'))


def test_no_generation_or_files_before_native_anchor_validation(unit, monkeypatch):
    def rejected(*args, **kwargs): raise ValueError('UNIT rejected anchor')
    monkeypatch.setattr(p.bank_api, 'verify_anchor', rejected)
    with pytest.raises(ValueError): build(unit)
    assert not unit['output'].exists() and 'generate' not in unit['events']


def test_create_only_and_disjoint_from_inputs(unit):
    build(unit); before=list(unit['events'])
    with pytest.raises(ValueError): build(unit)
    assert unit['events'] == before
    unit['output'] = unit['root']/'native/new_output'
    with pytest.raises(ValueError): build(unit)
    assert not unit['output'].exists()


@pytest.mark.parametrize('key,value', [('render_subframes_per_view',8), ('resolution',[848,408]),
    ('sample_count_limit',12), ('native_instance_backend','fast'), ('source_cap_reset',True),
    ('pose_changes_between_modes',True), ('execution',dict(native_launch_supported=True))])
def test_bound_policy_cannot_be_mutated_into_launch_or_short_capture(unit, key, value):
    out=build(unit); plan=p.oc.read_json(out['plan_path']); plan[key]=value
    with pytest.raises(ValueError): p.check_plan(plan)


def test_qualified_variant_must_use_current_clear_plan(unit):
    out=build(unit); plan=p.oc.read_json(out['plan_path'])
    q=Path(plan['variant_directory'])/'qualification.json'; value=p.oc.read_json(q)
    value['source_plan_path']=unit['proof']['pose_prior']['manifest']['path']
    value['source_plan_sha256']=unit['proof']['pose_prior']['manifest']['sha256']
    plan['source_bindings'][str(q)]=write(q,value)['sha256']
    with pytest.raises(ValueError, match='CURRENT clear plan'): p.check_plan(plan)


def test_original_reference_sample_cannot_be_replaced_by_legacy(unit):
    build(unit)
    proof=deepcopy(unit['proof']); pin=proof['native_observation']['sample']
    value=p.oc.read_json(pin['path']); value['schema_version']='greenhouse.rgbd_pilot_sample.v2'
    proof['native_observation']['sample']=write(Path(pin['path']),value)
    directory=unit['output']/'seed73_full_cr_730001'
    with pytest.raises(ValueError): p._facts(proof,directory,unit['catalogues'][str(directory)])


def test_current_scene_metadata_uses_actual_values_and_rejects_old_light(unit):
    plan={'scene_authority':unit['proof']['scene_authority']}
    actual=deepcopy(plan['scene_authority']['actual_lighting'])
    evidence=scene.current_scene_evidence(plan,lighting=actual,counts={'components':2},renderer='RealTimePathTracing')
    assert evidence['lighting']==actual and evidence['lighting'] is not actual
    actual['dome_intensity']=1200
    with pytest.raises(ValueError): scene.current_scene_evidence(plan,lighting=actual,counts={'components':2},renderer='RealTimePathTracing')
    with pytest.raises(ValueError): scene.current_scene_evidence(plan,lighting=evidence['lighting'],counts={'components':1},renderer='RealTimePathTracing')


def test_substitution_rejects_foreign_context_before_native_imports():
    with pytest.raises(ValueError, match='another prepared plan'):
        scene.substitute_generated({'prepared_plan_sha256':'0'*64}, {'UNIT_ONLY':True})


def test_scene_reuses_exact_pose_restore_and_full_scene_budget_policy_ast():
    assert scene.restore_pose is original_scene.restore_pose
    tree=ast.parse(inspect.getsource(scene.prepare_scene))
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call)]
    def attr(n,name): return isinstance(n.func,ast.Attribute) and n.func.attr==name
    light=[n for n in calls if attr(n,'apply') and isinstance(n.func.value,ast.Name) and n.func.value.id=='daylight']
    old=ast.parse(inspect.getsource(native_scene.prepare_native_scene))
    old_light=[n for n in ast.walk(old) if isinstance(n,ast.Call) and attr(n,'apply')
        and isinstance(n.func.value,ast.Name) and n.func.value.id=='daylight']
    assert len(light)==len(old_light)==1 and ast.dump(light[0])==ast.dump(old_light[0])
    names=[n.func.id for n in calls if isinstance(n.func,ast.Name)]
    assert names.count('populate')==names.count('freeze_rigid_bodies')==names.count('restore_pose')==1
    assert not any(attr(n,'render_product') for n in calls)
    assert not any(isinstance(n,ast.Name) and n.id=='old_manifest' for n in ast.walk(tree))
    assert 'SimulationApp' not in names and 'make_writer' not in names
    assert 'omni.usd' not in sys.modules and 'isaacsim' not in sys.modules
