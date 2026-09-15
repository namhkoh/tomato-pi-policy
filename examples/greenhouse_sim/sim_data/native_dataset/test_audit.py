"""Audit provenance tests; no native renderer or dataset approval."""
import json
from copy import deepcopy
import pytest
from . import audit
from .bundle import digest

def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))

@pytest.fixture
def case(tmp_path,monkeypatch):
    root=tmp_path/'capture';root.mkdir();plan_path=tmp_path/'plan.json'
    anchor=tmp_path/'anchor.json';write(anchor,dict(variant_directory='unused',source_collection_plan='unused'))
    bind=tmp_path/'binding';bind.write_text('bound');bindings={str(bind):digest(bind.read_bytes())}
    specs=[dict(candidate_id='view1'),dict(candidate_id='view2')]
    plan=dict(split='train',resolution=[1696,816],training_approved=False,source_cap_reset=False,
        physical_motion_commanded=False,hidden_cut_coordinates_executable=False,anchor_pair_plan=str(anchor),
        source_bindings=bindings,implementation_bindings=bindings,prerequisite_bindings=bindings,
        target_cases=[dict(target_id='family_cr_1/SubStem_1',views=specs)])
    write(plan_path,plan)
    request=dict(plan_path=str(plan_path),plan_sha256=digest(plan_path.read_bytes()),training_started=False)
    rows=[dict(candidate_id=s['candidate_id'],target_id='family_cr_1/SubStem_1',requested_spec=s,
        state='rejected_possible_geometry_overlap') for s in specs]
    result=dict(state='native_generated_multiview_pilot_complete_pending_review',
        source_assets_unchanged=True,training_approved=False,records=rows,
        captured_frames=0,automatically_clear_annotation_candidates=0)
    write(root/'request.json',request);write(root/'result.json',result)
    monkeypatch.setattr(audit,'load_for_inspection',lambda *args:dict(report={}))
    return root,plan_path,plan,request,result

def test_valid_zero_capture_report_is_not_training_approval(case):
    root,path,*_=case;r=audit.audit_capture(root,path)
    assert r['counts']=={} and r['records']==[] and r['training_approved'] is False

@pytest.mark.parametrize('kind',['duplicate','unplanned','wrong_target','wrong_spec','missing'])
def test_record_plan_corruption(case,kind):
    root,path,plan,request,result=case
    if kind=='duplicate':result['records'].append(deepcopy(result['records'][0]))
    if kind=='unplanned':result['records'][0]['candidate_id']='new_view'
    if kind=='wrong_target':result['records'][0]['target_id']='different/stem'
    if kind=='wrong_spec':result['records'][0]['requested_spec']['extra']='forged'
    if kind=='missing':result['records'].pop()
    write(root/'result.json',result)
    with pytest.raises(ValueError):audit.audit_capture(root,path)

@pytest.mark.parametrize('kind',['hash','path','claim'])
def test_capture_request_binding(case,kind):
    root,path,plan,request,result=case
    if kind=='hash':request['plan_sha256']='0'*64
    if kind=='path':request['plan_path']=str(path.parent/'elsewhere.json')
    if kind=='claim':request['training_started']=True
    write(root/'request.json',request)
    with pytest.raises(ValueError):audit.audit_capture(root,path)

def test_changed_source_binding(case):
    root,path,plan,*_=case
    binding=next(iter(plan['source_bindings']));write(__import__('pathlib').Path(binding),{'changed':True})
    with pytest.raises(ValueError,match='Stale'):audit.audit_capture(root,path)

def test_failed_native_result_never_accepted(case):
    root,path,*_=case;write(root/'failure.json',{'error':'failed'})
    with pytest.raises(ValueError,match='Failed native'):audit.audit_capture(root,path)

def test_duplicate_plan_candidates(case):
    root,path,plan,request,result=case
    plan['target_cases'][0]['views'].append(deepcopy(plan['target_cases'][0]['views'][0]))
    write(path,plan);request['plan_sha256']=digest(path.read_bytes());write(root/'request.json',request)
    with pytest.raises(ValueError,match='Duplicate planned'):audit.audit_capture(root,path)
