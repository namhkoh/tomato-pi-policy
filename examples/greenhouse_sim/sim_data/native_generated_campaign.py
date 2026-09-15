"""Bounded serial native generated-plant inspection campaign, never a training release.

Each frozen case qualifies the same robot-mounted camera, captures an original/
generated pair and derives fresh annotations. All failures stop; no automatic
retry, resume, image acceptance or source/split mutation.
"""
from pathlib import Path
import argparse
from collections import Counter
import os
import shutil
import subprocess
import sys
import time
import traceback

from .dataset_review import read_json,write_json,require,verify_bindings
from .depth_preview import sha256
from .generated_capture import check_plan

SCHEMA="greenhouse.native_generated_inspection_campaign.v1"
EXPECTED={
    "camera":"native_greenhouse_pair_captured_pending_visual_review",
    "generated":"generated_native_pair_captured_pending_visual_review",
}

def validate_campaign(plan_path):
    plan_path=Path(plan_path).resolve()
    plan=read_json(plan_path);root=plan_path.parent
    require(plan.get("schema")==SCHEMA and plan.get("training_approved") is False,
            "Explicit non-training inspection campaign required")
    cases=plan.get("cases")
    require(isinstance(cases,list) and 1<=len(cases)<=24,"One to24 explicit cases required")
    timeout=plan.get("worker_timeout_s")
    require(type(timeout) is int and 60<=timeout<=1800,"Bounded worker timeout required")
    verify_bindings(plan["implementation_bindings"])
    seen=set();targets=[]
    for index,row in enumerate(cases,1):
        require(row["case"]==f"case_{index:03d}","Ordered contiguous case identity required")
        path=(root/row["case"]/"plan.json").resolve()
        require(path.parent.parent==root and sha256(path)==row["plan_sha256"],"Case plan binding changed")
        item=read_json(path);check_plan(item)
        require(Path(item["prerequisite_directory"]).resolve()==path.parent/"camera",
                "Camera qualification must belong to this exact new case")
        key=(str(Path(item["source_capture"]).resolve()),item["source_sample"],item["generated_row"]["target_id"])
        require(key not in seen,"Repeated source-view/generated-target pair")
        seen.add(key);targets.append(item["conservative_view_cap_group"])
        for protected in (item["source_capture"],item["variant_directory"]):
            protected=Path(protected).resolve()
            require(not root.is_relative_to(protected) and not protected.is_relative_to(root),
                    "Campaign must be disjoint from source captures/generated assets")
    require(max(Counter(targets).values())<=12,"Inspection batch exceeds donor-target view budget")
    return plan,root

def worker_args(phase,item,plan_path,controller_pid):
    require(phase in EXPECTED,"Unknown native phase")
    require(type(controller_pid) is int and controller_pid>0,"Exact parent PID required")
    if phase=="camera":
        return "sim_data.native_greenhouse_pair",["--source-capture",item["source_capture"],"--sample",item["source_sample"]]
    return "sim_data.native_generated_pair",["--plan",str(plan_path),"--controller-pid",str(controller_pid)]

def successful_worker(returncode,timed_out,result,phase,has_failure=False):
    return (phase in EXPECTED and type(returncode) is int and returncode==0 and timed_out is False
            and not has_failure and isinstance(result,dict) and result.get("state")==EXPECTED[phase]
            and result.get("training_approved") is False)

def run_worker(case,phase,item,plan_path,timeout):
    from .native_generated_pair import windows_worker_admission
    from sim_physics.host_memory import preflight
    windows_worker_admission(None)
    reserve=preflight()
    disk_free=shutil.disk_usage(case).free
    write_json(case/(phase+"_admission.json"),dict(host_memory_preflight=reserve,
        queue_commit_reserve_bytes=18*2**30,available_disk_bytes=disk_free,
        minimum_disk_bytes=40*2**30,renderer_started=False))
    require(reserve["allowed"] and reserve["commit_headroom_bytes"]>=18*2**30,"Memory reserve unavailable; no bypass")
    require(disk_free>=40*2**30,"Insufficient disk reserve")
    output=case/phase
    require(not output.exists(),"Never overwrite a prior worker attempt")
    module,args=worker_args(phase,item,plan_path,os.getpid())
    command=[sys.executable,"-u","-m",module,*args,"--output",str(output)]
    started=time.monotonic()
    with (case/(phase+"_worker.log")).open("x",encoding="utf-8") as log:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,
                                 creationflags=subprocess.CREATE_NO_WINDOW)
        timed_out=False
        try:
            write_json(case/(phase+"_launch.json"),dict(pid=process.pid,command=command,
                timeout_s=timeout,host_memory_preflight=reserve,training_approved=False))
            try:process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out=True;process.kill();process.wait(timeout=20) # Exact owned child only.
        except BaseException:
            if process.poll() is None:
                process.kill();process.wait(timeout=20)
            raise
    result=read_json(output/"result.json") if (output/"result.json").is_file() else None
    passed=successful_worker(process.returncode,timed_out,result,phase,(output/"failure.json").exists())
    write_json(case/(phase+"_exit.json"),dict(pid=process.pid,returncode=process.returncode,
        timed_out=timed_out,elapsed_s=time.monotonic()-started,passed=passed,
        result_state=result and result.get("state"),training_approved=False,
        log_sha256=sha256(case/(phase+"_worker.log"))))
    require(passed,"Native worker failed; campaign stopped without automatic retry")

def run(plan_path):
    from .native_generated_pair import windows_worker_admission
    from .native_clear_annotation import build
    plan,root=validate_campaign(plan_path)
    require(not (root/"execution_started.json").exists(),"Already attempted; create a new explicit campaign")
    windows_worker_admission(None)
    for row in plan["cases"]:
        require(all(not (root/row["case"]/p).exists() for p in ("camera","generated","annotations")),
                "Prior case output exists; no implicit resume")
    digest=sha256(plan_path);records=[]
    write_json(root/"execution_started.json",dict(pid=os.getpid(),plan_sha256=digest,
        single_native_renderer=True,automatic_retries=False,training_approved=False))
    try:
        for row in plan["cases"]:
            require(sha256(plan_path)==digest,"Campaign plan changed")
            validate_campaign(plan_path)
            case=root/row["case"];path=case/"plan.json";item=read_json(path)
            for phase in ("camera","generated"):
                verify_bindings(plan["implementation_bindings"])
                run_worker(case,phase,item,path,plan["worker_timeout_s"])
            annotations=build(path,case/"generated",case/"annotations")
            record=dict(case=row["case"],family=item["source_family"],
                target=item["source_row"]["target_id"],generated_target=item["generated_row"]["target_id"],
                eligible_annotation_candidates=annotations["eligible_candidates"],
                training_approved=False,visual_review_performed=False)
            write_json(case/"result.json",record);records.append(record)
            print("NATIVE_INSPECTION_CASE_COMPLETE",record,flush=True)
        validate_campaign(plan_path)
        write_json(root/"result.json",dict(state="native_inspection_completed_pending_individual_review",
            cases=records,training_approved=False,final_dataset_built=False))
    except BaseException:
        write_json(root/"failure.json",dict(error=traceback.format_exc(),completed=records,training_approved=False))
        raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--validate-only",action="store_true")
    a=p.parse_args()
    if a.validate_only:
        plan,_=validate_campaign(a.plan)
        print("NATIVE_INSPECTION_PLAN_VALID",len(plan["cases"]),"not executed")
    else:run(a.plan)

if __name__=="__main__":main()
