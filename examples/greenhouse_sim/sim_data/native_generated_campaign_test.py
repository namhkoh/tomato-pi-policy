"""CPU campaign tests; never launch Kit or change real data."""
from copy import deepcopy
import pytest
from . import native_generated_campaign as campaign
from .dataset_review import write_json,read_json
from .depth_preview import sha256
from .generated_capture_test import simple_plan
from .plant_variants import VERSION as GENERATOR_VERSION

def prepared(tmp_path):
    source=tmp_path/"source";source.mkdir()
    item=simple_plan(source)
    # Plan validation reads the generator discriminator. This is a deliberately
    # non-renderable unit fixture, not a qualified plant or native-capture claim.
    variant=source/"variant";variant.mkdir()
    write_json(variant/"qualification.json",dict(version=GENERATOR_VERSION))
    item["source_bindings"][str(variant/"qualification.json")]=sha256(variant/"qualification.json")
    root=tmp_path/"inspection";case=root/"case_001";case.mkdir(parents=True)
    item.update(prerequisite_directory=str(case/"camera"),source_capture=str(source/"capture"),
                source_sample="sample_0001",variant_directory=str(source/"variant"))
    write_json(case/"plan.json",item)
    plan=dict(schema=campaign.SCHEMA,training_approved=False,worker_timeout_s=1800,
              implementation_bindings=item["source_bindings"],
              cases=[dict(case="case_001",plan_sha256=sha256(case/"plan.json"))])
    return root,plan,item

def test_valid_campaign_does_not_claim_native_or_training_approval(tmp_path):
    root,plan,_=prepared(tmp_path);write_json(root/"plan.json",plan)
    result,where=campaign.validate_campaign(root/"plan.json")
    assert result==plan and where==root and not (root/"execution_started.json").exists()

@pytest.mark.parametrize("fault",["approval","timeout_bool","timeout_long","empty","identity","hash","code"])
def test_campaign_rejects_scope_or_stale_bindings(tmp_path,fault):
    root,plan,_=prepared(tmp_path)
    if fault=="approval":plan["training_approved"]=True
    if fault=="timeout_bool":plan["worker_timeout_s"]=True
    if fault=="timeout_long":plan["worker_timeout_s"]=1801
    if fault=="empty":plan["cases"]=[]
    if fault=="identity":plan["cases"][0]["case"]="../other"
    if fault=="hash":plan["cases"][0]["plan_sha256"]="0"*64
    if fault=="code":plan["implementation_bindings"]={str(root/"case_001/plan.json"):"0"*64}
    write_json(root/"plan.json",plan)
    with pytest.raises(ValueError):campaign.validate_campaign(root/"plan.json")

def test_worker_args_only_allow_two_native_modules():
    item=dict(source_capture="capture",source_sample="sample_0001")
    assert campaign.worker_args("camera",item,"case/plan.json",123)==(
        "sim_data.native_greenhouse_pair",["--source-capture","capture","--sample","sample_0001"])
    module,args=campaign.worker_args("generated",item,"case/plan.json",123)
    assert module=="sim_data.native_generated_pair" and args[-2:]==["--controller-pid","123"]
    with pytest.raises(ValueError):campaign.worker_args("training",item,"plan.json",123)
    with pytest.raises(ValueError):campaign.worker_args("generated",item,"plan.json",True)

@pytest.mark.parametrize("code,timeout,state,approval,failure,expected",[
    (0,False,campaign.EXPECTED["generated"],False,False,True),
    (1,False,campaign.EXPECTED["generated"],False,False,False),
    (0,True,campaign.EXPECTED["generated"],False,False,False),
    (0,False,"incomplete",False,False,False),
    (0,False,campaign.EXPECTED["generated"],True,False,False),
    (0,False,campaign.EXPECTED["generated"],False,True,False),
    (False,False,campaign.EXPECTED["generated"],False,False,False),
])
def test_worker_result_does_not_infer_success_from_files(code,timeout,state,approval,failure,expected):
    assert campaign.successful_worker(code,timeout,dict(state=state,training_approved=approval),"generated",failure)==expected

def test_subprocess_unknown_result_is_not_success():
    assert not campaign.successful_worker(0,False,None,"camera")
    assert not campaign.successful_worker(0,False,{},"training")

def test_repeated_original_view_and_generated_target_rejected(tmp_path):
    root,plan,item=prepared(tmp_path)
    second=root/"case_002";second.mkdir()
    item["prerequisite_directory"]=str(second/"camera")
    write_json(second/"plan.json",item)
    plan["cases"].append(dict(case="case_002",plan_sha256=sha256(second/"plan.json")))
    write_json(root/"plan.json",plan)
    with pytest.raises(ValueError,match="Repeated"):campaign.validate_campaign(root/"plan.json")


@pytest.mark.parametrize("memory_gib,disk_gib",[(17,50),(20,39)])
def test_resource_hold_records_evidence_without_starting_process(tmp_path,monkeypatch,memory_gib,disk_gib):
    from types import SimpleNamespace
    from . import native_generated_pair
    from sim_physics import host_memory
    monkeypatch.setattr(native_generated_pair,"windows_worker_admission",lambda pid:None)
    monkeypatch.setattr(host_memory,"preflight",lambda:dict(allowed=True,commit_headroom_bytes=memory_gib*2**30))
    monkeypatch.setattr(campaign.shutil,"disk_usage",lambda path:SimpleNamespace(free=disk_gib*2**30))
    def forbidden(*args,**kwargs):raise AssertionError("Process must not start")
    monkeypatch.setattr(campaign.subprocess,"Popen",forbidden)
    with pytest.raises(ValueError):campaign.run_worker(tmp_path,"camera",{},tmp_path/"plan.json",60)
    receipt=read_json(tmp_path/"camera_admission.json")
    assert receipt["renderer_started"] is False and receipt["available_disk_bytes"]==disk_gib*2**30
    assert not (tmp_path/"camera_launch.json").exists()
