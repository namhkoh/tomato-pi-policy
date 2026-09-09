from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from sim_data import collection_plan as planning, collection_run as runner
from sim_data.cut_regions import load_rule
from sim_data.depth_preview import sha256


def report(family):
    parent = {'id':'Main', 'type':'main_stem', 'parent':None, 'translation_plant_m':[0,0,.5],
              'capsules_local_m':[[[0,0,-.1,.003],[0,0,.1,.003]]]}
    stem = {'id':'Petiole','type':'sub_stem','parent':'Main','deleafed':False,
            'translation_plant_m':[.003,0,.5],'attachment_plant_m':[.003,0,.5],
            'axis_plant':[1,0,0],'capsules_local_m':[[[0,0,0,.002],[.06,0,0,.002]]]}
    return {'plant_id':family,'manifest_path':f'/fixture/{family}/manifest.json','manifest_sha256':'test',
            'issues':[],'components':{'Main':parent,'Petiole':stem},
            'targets':[{'target_id':family+'/Petiole','component_id':'Petiole','status':'needs_review',
                'reason_codes':[],'expected_detached_component_ids':['Petiole','Leaf'],
                'protected_descendant_ids':[],'attachment_plant_m':[.003,0,.5]}]}


@pytest.fixture
def reports():
    return [report(n) for n in ['seed101_full','seed103_full']+[f'new{i:02d}' for i in range(22)]]


def test_family_split_is_deterministic_disjoint_and_keeps_reviewed_plants_in_train(reports):
    names = [r['plant_id'] for r in reports]
    a = planning.family_splits(names, 0)
    assert a == planning.family_splits(list(reversed(names)), 0)
    assert [list(a.values()).count(s) for s in ['train','validation','test']] == [16,4,4]
    assert all(a[n]=='train' for n in planning.REVIEWED_FAMILIES)
    with pytest.raises(ValueError): planning.family_splits(names+[names[0]], 0)


def test_native_schedule_is_new_families_only_and_never_approves_labels(reports):
    before = deepcopy(reports)
    result = planning.schedule(reports, load_rule(), planning.configuration())
    assert len(result['jobs']) == 2
    assert result == planning.schedule(list(reversed(reports)), load_rule(), planning.configuration())
    for job in result['jobs']:
        assert job['plant_family'] not in planning.REVIEWED_FAMILIES and job['split']=='train'
        row = job['targets'][0]
        assert row['target_id'].endswith('/Petiole') and row['variant_id']==job['plant_family']
        assert not row['human_review_performed'] and not row['training_label_approved']
        assert row['cut_region_proposal']['accepted_centerline_interval']['arc_range_m']==[.01,.02]
    assert reports == before


@pytest.mark.parametrize('kind',['excluded','warning','protected','geometry'])
def test_bad_native_targets_are_explicitly_excluded(kind):
    r = report('plant')
    if kind=='excluded': r['targets'][0].update(status='excluded', reason_codes=['already_deleafed'])
    if kind=='warning': r['issues']=[{'severity':'warning','component_id':'Petiole','code':'attachment_gap'}]
    if kind=='protected': r['targets'][0].update(status='excluded', reason_codes=['protected_descendants'])
    if kind=='geometry': r['components']['Petiole']['capsules_local_m']=[[[0,0,0,.002],[.002,0,0,.002]]]
    rows, decisions = planning.native_rows(r, load_rule())
    assert not rows and decisions[0]['state']=='excluded' and decisions[0]['reasons']


@pytest.mark.parametrize('values',[(0,2,2,0),(5,2,2,0),(2,4,2,0),(2,2,4,0),(True,2,2,0),(2,2,2,-1)])
def test_collection_bounds_are_enforced(values):
    with pytest.raises(ValueError): planning.configuration(*values)


@pytest.fixture
def saved_plan(tmp_path, monkeypatch, reports):
    package = tmp_path/'source'
    package.mkdir()
    sentinel = package/'source.usd'
    sentinel.write_text('unchanged fixture')
    monkeypatch.setattr(planning, 'source_reports', lambda p: deepcopy(reports))
    monkeypatch.setattr(planning, 'bindings_for', lambda p,r: {str(sentinel):sha256(sentinel)})
    output = tmp_path/'plan'
    plan = planning.build_plan(package, output, planning.configuration())
    return output/'plan.json', plan, sentinel


def test_plan_is_recomputed_and_sources_are_not_modified(saved_plan):
    path, expected, source = saved_plan
    actual, _ = planning.load_plan(path)
    assert actual == expected and source.read_text()=='unchanged fixture'


@pytest.mark.parametrize('kind',['split','point','approval','config','source','missing_bindings'])
def test_changed_source_or_schedule_rejected(saved_plan, kind):
    path, plan, source = saved_plan
    if kind=='split': plan['jobs'][0]['split']='test'
    if kind=='point': plan['jobs'][0]['targets'][0]['cut_region_proposal']['nominal']['point_plant_m'][0]+=.001
    if kind=='approval': plan['training_dataset_approved']=True
    if kind=='config': plan['configuration']['target_world_height_m']=[0,10]
    if kind=='source': source.write_text('changed')
    if kind=='missing_bindings': plan['source_bindings_sha256']={}
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError): planning.load_plan(path)


def test_worker_import_does_not_load_usd_before_kit():
    code = "import sys; import sim_data.collection_worker, sim_data.collection_run; assert not any(k=='pxr.Usd' or k.startswith('omni.') for k in sys.modules)"
    subprocess.run([sys.executable,'-c',code],check=True,timeout=20)


@pytest.mark.parametrize('code,timed_out,manifest,state',[
    (1,False,{'state':'pilot_ready_for_review','samples':[1]},'stopped_worker_nonzero_exit'),
    (0,True,{},'stopped_worker_timeout'),
    (0,False,None,'stopped_missing_complete_capture'),
    (0,False,{'state':'pilot_ready_for_review','samples':[]},'stopped_missing_complete_capture'),
    (0,False,{'state':'pilot_ready_for_review','samples':[1]},'stopped_invalid_capture_contract'),
    (0,False,{'state':'pilot_ready_for_review','samples':[1],'source_assets_unchanged':True,
              'training_dataset_approved':False},'ready_for_independent_audit')])
def test_process_exit_and_capture_contract_are_both_required(code,timed_out,manifest,state):
    assert runner.worker_state(code,timed_out,manifest)==state


def test_batch_stops_on_first_failed_worker_and_does_not_overwrite(tmp_path, monkeypatch):
    plan_file=tmp_path/'plan.json'
    plan_file.write_text('{}')
    plan={'package':str(tmp_path/'source'),'jobs':[{'job_id':'job_001','plant_family':'a'},
                                                {'job_id':'job_002','plant_family':'b'}]}
    monkeypatch.setattr(runner,'load_plan',lambda p:(plan,[]))
    calls=[]
    class FailedProcess:
        pid=123
        returncode=7
        def __init__(self,*a,**kw): calls.append(a)
        def poll(self): return self.returncode
    monkeypatch.setattr(runner.subprocess,'Popen',FailedProcess)
    output=tmp_path/'batch'
    result=runner.run_jobs(plan_file,output,max_jobs=2)
    assert len(calls)==1 and result['state']=='stopped_worker_nonzero_exit'
    assert not (output/'job_002').exists()
    with pytest.raises(ValueError): runner.run_jobs(plan_file,output)
    assert json.loads((output/'manifest.json').read_text())['jobs'][0]['returncode']==7


@pytest.mark.parametrize('fails',[False,True])
def test_view_search_restores_robot_root_and_links_even_on_failure(monkeypatch,fails):
    import numpy as np
    from pxr import Gf, Usd, UsdGeom
    from sim_data import collection_worker as worker
    from greenhouse_sim.robot_hardware import _set_transform
    stage=Usd.Stage.CreateInMemory()
    stage.SetEditTarget(stage.GetSessionLayer())
    root=UsdGeom.Xform.Define(stage,'/World/RBY1').GetPrim()
    link=UsdGeom.Xform.Define(stage,'/World/RBY1/link_head_2').GetPrim()
    _set_transform(root,np.diag([-1.,-1.,1.]),np.array([.8,0,.1]))
    _set_transform(link,np.eye(3),np.array([0,0,1.4]))
    before=[np.asarray(UsdGeom.Xformable(p).GetLocalTransformation()).copy() for p in (root,link)]
    def search(*args):
        _set_transform(root,np.eye(3),np.array([.45,.3,.1]))
        _set_transform(link,np.diag([-1.,-1.,1.]),np.array([0,0,1.2]))
        if fails: raise ValueError('test planning failure')
        return {'original_base_x_m':.8}
    monkeypatch.setattr(worker,'_prepare_views',search)
    if fails:
        with pytest.raises(ValueError,match='planning failure'):
            worker.prepare_views(stage,{'root':'/World/RBY1'},[],[],2)
    else:
        assert worker.prepare_views(stage,{'root':'/World/RBY1'},[],[],2)['original_base_x_m']==.8
    for prim, matrix in zip((root,link),before):
        assert np.array_equal(np.asarray(UsdGeom.Xformable(prim).GetLocalTransformation()),matrix)
