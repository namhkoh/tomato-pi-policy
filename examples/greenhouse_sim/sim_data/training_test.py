from copy import deepcopy
import json

import numpy as np
from PIL import Image
import pytest

from sim_data import training_plan as planning
from sim_data import training_contract as contract
from sim_data.collection_test import report
from sim_data.cut_regions import load_rule
from sim_data.capture_contract import project


def test_training_schedule_preserves_all_reserved_families():
    reports=[report(n) for n in ['seed101_full','seed103_full']+[f'new{i:02d}' for i in range(22)]]
    p=planning.schedule(reports,load_rule(),planning.configuration(12,64))
    assert len(p['jobs'])==24
    assert [list(p['family_assignments'].values()).count(s) for s in ['train','validation','test']]==[16,4,4]
    assert all(j['split']==p['family_assignments'][j['plant_family']] for j in p['jobs'])
    assert all(not r['training_label_approved'] for j in p['jobs'] for r in j['targets'])
    assert p==planning.schedule(list(reversed(reports)),load_rule(),planning.configuration(12,64))


@pytest.mark.parametrize('args',[(0,64,0),(37,64,0),(12,0,0),(12,161,0),(12,64,1),(True,64,0)])
def test_training_schedule_rejects_changed_split_seed_or_unbounded_size(args):
    with pytest.raises(ValueError): planning.configuration(*args)


def test_grounding_framing_is_deterministic_continuous_and_aisle_bounded():
    a=planning.view_specs(.8,0,'target-A',64)
    assert a==planning.view_specs(.8,0,'target-A',64)
    assert a!=planning.view_specs(.8,0,'target-B',64)
    assert len(set(tuple(s['desired_pixel_xy']) for s in a))==len(a)
    for s in a:
        assert .4<=s['root_x_m']<=.65 and -.4<=s['y_offset_m']<=.4
        assert 150<=s['root_yaw_degrees']<=210
        assert .15*848<=s['desired_pixel_xy'][0]<=.85*848
        assert .15*408<=s['desired_pixel_xy'][1]<=.85*408


@pytest.mark.parametrize('torso',[False,True])
def test_view_shards_are_disjoint_and_preserve_the_same_global_sequence(torso):
    full=planning.view_specs(.8,0,'a',16,vary_torso=torso)
    first=planning.view_specs(.8,0,'a',8,vary_torso=torso)
    second=planning.view_specs(.8,0,'a',8,vary_torso=torso,view_offset=8)
    assert first+second==full
    assert not {s['candidate_id'] for s in first}&{s['candidate_id'] for s in second}


@pytest.mark.parametrize('offset',[-1,2049,True,1.5])
def test_invalid_shard_offsets_rejected(offset):
    with pytest.raises(ValueError): planning.configuration(view_offset=offset)
    with pytest.raises(ValueError): planning.view_specs(.8,0,'a',8,view_offset=offset)


@pytest.mark.parametrize('answer',[
    dict(status='localized',cut_point_uv=[434,204],visibility='clear',next_action='inspect_cut_region'),
    dict(status='localized',cut_point_uv=[12.1,15.2],visibility='partial',next_action='inspect_cut_region'),
    dict(status='abstain',cut_point_uv=None,visibility='occluded',next_action='change_viewpoint')])
def test_answer_contract_roundtrip(answer):
    assert contract.validate_answer(json.loads(json.dumps(answer)))==answer


@pytest.mark.parametrize('kind',['extra','nan','bool','bounds','abstain_point','unsafe_action','visibility'])
def test_invalid_or_unsafe_answers_rejected(kind):
    a=dict(status='localized',cut_point_uv=[434,204],visibility='clear',next_action='inspect_cut_region')
    if kind=='extra': a['world_xyz']=[0,0,0]
    if kind=='nan': a['cut_point_uv']=[float('nan'),12]
    if kind=='bool': a['cut_point_uv']=[True,12]
    if kind=='bounds': a['cut_point_uv']=[848,204]
    if kind=='abstain_point': a['status']='abstain'
    if kind=='unsafe_action': a['next_action']='execute_cut'
    if kind=='visibility': a['visibility']='occluded'
    with pytest.raises(ValueError): contract.validate_answer(a)


@pytest.fixture
def label_fixture(tmp_path):
    r=report('fixture')
    c=r['components']['Petiole']
    c.update(id='Petiole',translation_plant_m=[0,0,-1],attachment_plant_m=[0,0,-1],
             capsules_local_m=[[[0,0,0,.004],[.20,0,0,.004]]])
    cal=dict(camera_to_world_usd_row_vectors=np.eye(4).tolist(),intrinsics=[[1000,0,424],[0,1000,204],[0,0,1]],
             resolution=[848,408],clipping_range_m=[.04,10])
    sup=dict(target_id='fixture/Petiole',variant_id='fixture',split_group='fixture',plant_to_world_usd_row_vectors=np.eye(4).tolist(),
             nominal_projected=project([[.01,0,-1]],cal)[0],nominal_world_m=[.01,0,-1],
             visibility_evidence=dict(nominal=dict(visible_target_evidence=True,status='target_identity_and_depth_consistent'),
                                      interval_fully_in_frame=True,sampled_interval_visible_pixel_fraction=1.))
    meta=dict(sample_id='sample_0001',calibration=cal,supervision=sup,
              quality=dict(estimated_petiole_diameter_px=8.,projected_interval_length_px=10.,target_mask_dark_fraction=0.))
    (tmp_path/'inputs').mkdir()
    (tmp_path/'supervision').mkdir()
    np.save(tmp_path/'inputs/depth_m.npy',np.full((408,848),.996,dtype=np.float32))
    Image.fromarray(np.full((408,848),255,dtype=np.uint8)).save(tmp_path/'inputs/depth_valid.png')
    ids=np.zeros((408,848),np.uint32)
    ids[201:209,424:625]=2
    ids[180:230,418:424]=1
    rgb=np.full((408,848,3),120,np.uint8); rgb[ids==2]=(50,100,30)
    Image.fromarray(rgb).save(tmp_path/'inputs/rgb.png')
    np.save(tmp_path/'supervision/component_id.npy',ids)
    (tmp_path/'supervision/identities.json').write_text(json.dumps(dict(component_catalogue=[
        dict(component_id='Petiole',variant_id='fixture',component_index=2),
        dict(component_id='Main',variant_id='fixture',component_index=1)])))
    return tmp_path,meta,r


def test_visible_query_and_nominal_are_distinct_and_easy_label_is_bounded(label_fixture):
    d,m,r=label_fixture
    v=contract.derive_label(d,m,r)
    assert v['eligible'] and v['difficulty']=='easy'
    assert v['answer']['cut_point_uv']==[434.,204.]
    assert np.linalg.norm(np.asarray(v['query_pixel_uv'])-[434,204])>=18
    assert not v['human_review_performed'] and not v['physical_execution_approved']
    assert '434.0' not in v['user_prompt']


def test_proved_occlusion_is_abstention_not_hidden_point_regression(label_fixture):
    d,m,r=label_fixture
    m['supervision']['visibility_evidence']['nominal']=dict(visible_target_evidence=False,
        status='foreground_occluder_identified',observed_component={'organ_type':'leaf'})
    v=contract.derive_label(d,m,r)
    assert v['eligible'] and v['difficulty']=='hard' and v['answer']['cut_point_uv'] is None


def test_two_pixel_hidden_query_is_excluded_not_called_hard(label_fixture):
    d,m,r=label_fixture
    ids=np.load(d/'supervision/component_id.npy'); ids[ids==2]=0
    ids[204,469:471]=2
    np.save(d/'supervision/component_id.npy',ids)
    m['supervision']['visibility_evidence']['nominal']=dict(visible_target_evidence=False,
        status='foreground_occluder_identified',observed_component={'organ_type':'main_stem'})
    v=contract.derive_label(d,m,r)
    assert not v['eligible'] and v['reason']=='no_usable_visible_query'
    assert v['query_usability_rejection_counts']['minimum_island_pixels']


def test_query_gate_versions_contract_and_does_not_edit_observations(label_fixture):
    d,m,r=label_fixture
    files=[*d.glob('inputs/*'),*d.glob('supervision/*')]
    before={p:p.read_bytes() for p in files}
    v=contract.derive_label(d,m,r)
    assert v['task_id'].endswith('.v3') and v['query_usability']['passed']
    assert all(p.read_bytes()==b for p,b in before.items())


def test_disconnected_distal_query_does_not_make_a_localizable_example(label_fixture):
    d,m,r=label_fixture
    ids=np.load(d/'supervision/component_id.npy')
    ids[:,455:465]=0 # Cut is visible but every >=45 mm query is behind a mask gap.
    np.save(d/'supervision/component_id.npy',ids)
    v=contract.derive_label(d,m,r)
    assert not v['eligible'] and v['reason']=='no_visible_query_connected_to_cut'


def test_connected_query_is_chosen_instead_of_isolated_far_fragment(label_fixture):
    d,m,r=label_fixture
    ids=np.load(d/'supervision/component_id.npy')
    ids[:,495:505]=0
    np.save(d/'supervision/component_id.npy',ids)
    for i in range(12):
        m['sample_id']=f'sample_{i:04d}'
        v=contract.derive_label(d,m,r)
        assert v['eligible'] and v['query_cut_visible_connection_verified']
        assert v['query_pixel_uv'][0]<495


def test_hidden_cut_does_not_falsely_require_visible_query_connection(label_fixture):
    d,m,r=label_fixture
    ids=np.load(d/'supervision/component_id.npy'); ids[:,430:465]=0
    np.save(d/'supervision/component_id.npy',ids)
    m['supervision']['visibility_evidence']['nominal']=dict(visible_target_evidence=False,
        status='foreground_occluder_identified',observed_component={'organ_type':'leaf'})
    v=contract.derive_label(d,m,r)
    assert v['eligible'] and v['difficulty']=='hard' and not v['query_cut_visible_connection_verified']


@pytest.mark.parametrize('kind',['unknown_visibility','thin','dark','no_parent','parent_blend','no_query'])
def test_ambiguity_is_excluded_not_mislabeled_as_a_negative(label_fixture,kind):
    d,m,r=label_fixture
    if kind=='unknown_visibility': m['supervision']['visibility_evidence']['nominal']=dict(visible_target_evidence=False,status='unknown_no_target_identity_at_pixel')
    if kind=='thin': m['quality']['estimated_petiole_diameter_px']=2.
    if kind=='dark': m['quality']['target_mask_dark_fraction']=.9
    ids=np.load(d/'supervision/component_id.npy')
    if kind=='no_parent': ids[ids==1]=0
    if kind=='parent_blend': ids[205:209,429:441]=1
    if kind=='no_query': ids[:,450:]=0
    np.save(d/'supervision/component_id.npy',ids)
    v=contract.derive_label(d,m,r)
    assert not v['eligible'] and 'answer' not in v
