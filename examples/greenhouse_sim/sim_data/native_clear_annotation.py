"""Build a new native1696 annotation pilot from a completed frozen plant pair.

This is NOT a production release or an H200 training-loader bypass.
"""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image, ImageDraw
from .dataset_review import read_json,write_json,require,verify_bindings,safe_file
from .depth_preview import sha256
from .generated_capture import check_plan,verify_sensor_prerequisite,assert_pair_fresh
from .plant_variant_catalogue import load_for_inspection
from .audit import audit_manifest
from .native_clear_labels import derive
from .native_clear_contract import (TASK,CONTRACT,SYSTEM,contract_hash,crop_box,user_prompt,
                                    model_messages,model_answer,canonical_answer)


def build(plan_path,capture,output):
    plan_path,capture,output=map(lambda p:Path(p).resolve(),(plan_path,capture,output))
    plan=read_json(plan_path)
    check_plan(plan)
    prerequisite=verify_sensor_prerequisite(plan)
    require(not output.exists() and all(not output.is_relative_to(p) and not p.is_relative_to(output)
            for p in (capture,Path(plan["source_capture"]),Path(plan["variant_directory"]),
                      Path(plan["source_collection_plan"]).parent,Path(plan["prerequisite_directory"]))),
            "New disjoint annotation directory required")
    request,result=read_json(capture/"request.json"),read_json(capture/"result.json")
    require(not (capture/"failure.json").exists() and request["plan_sha256"]==sha256(plan_path)
            and Path(request["plan_path"]).resolve()==plan_path
            and result["state"]=="generated_native_pair_captured_pending_visual_review"
            and result["source_assets_unchanged"] is True and result["training_approved"] is False,
            "Completed hash-bound native pair required")
    assert_pair_fresh(*result["samples"])
    generated=load_for_inspection(plan["variant_directory"],plan["source_collection_plan"])
    original_plan=read_json(plan["source_collection_plan"])
    job=next(j for j in original_plan["jobs"] if j["plant_family"]==plan["source_family"])
    require(original_plan["family_assignments"][plan["source_family"]]=="train","TRAIN pilot only")
    original=audit_manifest(job["source_manifest_path"])
    bindings={str(plan_path):sha256(plan_path),str(capture/"request.json"):sha256(capture/"request.json"),
              str(capture/"result.json"):sha256(capture/"result.json"),
              **plan["source_bindings"],**prerequisite["bindings"]}
    implementation={str(p):sha256(p) for p in Path(__file__).parent.glob("native_clear_*.py")}
    implementation[str(Path(__file__).with_name("native_query_visibility.py"))]=sha256(Path(__file__).with_name("native_query_visibility.py"))
    output.mkdir(parents=True)
    write_json(output/"request.json",dict(task=TASK,contract=CONTRACT,contract_sha256=contract_hash(),
        plan=str(plan_path),capture=str(capture),source_bindings=bindings,implementation_sha256=implementation,
        training_started=False,training_approved=False,legacy_dataset_modified=False))
    records=[]
    for name,report in (("original_control",original),("generated_variant",generated["report"])):
        source=capture/name
        metadata=read_json(source/"sample.json")
        require(metadata in result["samples"] and metadata["sample_id"]==name,"Native sample mismatch")
        bindings[str(source/"sample.json")]=sha256(source/"sample.json")
        for relative,info in metadata["files"].items():
            path=safe_file(source,relative)
            require(sha256(path)==info["sha256"],"Native source artifact changed")
            bindings[str(path)]=info["sha256"]
        with Image.open(source/"inputs/rgb.png") as im: rgb=np.asarray(im).copy()
        with Image.open(source/"inputs/depth_valid.png") as im: raw_valid=np.asarray(im).copy()
        require(set(np.unique(raw_valid))<={0,255},"Nonbinary native validity")
        depth=np.load(source/"inputs/depth_m.npy",allow_pickle=False)
        components=np.load(source/"supervision/component_id.npy",allow_pickle=False)
        catalogue=read_json(source/"supervision/identities.json")["component_catalogue"]
        annotation=derive(metadata,report,rgb,depth,raw_valid==255,components,catalogue)
        folder=output/name
        folder.mkdir()
        write_json(folder/"label.json",annotation)
        record=dict(name=name,eligible=annotation["eligible"],reason=annotation["reason"],
                    label_sha256=sha256(folder/"label.json"),source_sample_sha256=sha256(source/"sample.json"),
                    training_approved=False)
        if annotation["eligible"]:
            for relative in ("inputs/rgb.png","inputs/depth_m.npy","inputs/depth_valid.png","supervision/target_visible.png"):
                target=folder/relative
                target.parent.mkdir(exist_ok=True)
                shutil.copyfile(source/relative,target)
                require(sha256(target)==metadata["files"][relative]["sha256"],"Copy differs from native source")
            query=annotation["query_pixel_uv"]
            answer=annotation["answer"]
            with Image.open(folder/"inputs/rgb.png") as full:
                full.crop(crop_box(query)).save(folder/"inputs/query_crop.png")
                variants={}
                max_error=0.
                for crop in (False,True):
                    training=model_messages(full,query,answer=answer,crop=crop)
                    inference=model_messages(full,query,crop=crop)
                    require(training[0]==inference[0]
                            and training[1]["content"][-1]==inference[1]["content"][-1],
                            "Training/inference prompt mismatch")
                    require(all(np.array_equal(np.asarray(a["image"]),np.asarray(b["image"]))
                            for a,b in zip(training[1]["content"][:-1],inference[1]["content"][:-1],strict=True)),
                            "Training/inference image mismatch")
                    decoded=canonical_answer(json.loads(training[-1]["content"][0]["text"]))
                    max_error=max(max_error,max(abs(a-b) for a,b in zip(decoded["cut_point_uv"],answer["cut_point_uv"],strict=True)))
                    key="original_rgb_plus_native_crop" if crop else "original_rgb"
                    variants[key]=dict(images=["inputs/rgb.png"]+(["inputs/query_crop.png"] if crop else []),
                        messages=[dict(role="system",content=SYSTEM),
                                  dict(role="user",content=("<image>\n"* (2 if crop else 1))+user_prompt(query,crop=crop)),
                                  dict(role="assistant",content=json.dumps(model_answer(answer),separators=(",",":")))])
                    for messages in (training,inference):
                        for content in messages[1]["content"][:-1]:content["image"].close()
                require(max_error<=.016960001,"Coordinate roundtrip exceeded one native quantization step")
                write_json(folder/"model_inputs.json",dict(task=TASK,inspection_only=True,
                    variants=variants,maximum_roundtrip_error_px=max_error,model_processor_executed=False,
                    hidden_geometry_or_depth_in_model_inputs=False,training_loader_supported=False))
                overlay=full.copy()
                draw=ImageDraw.Draw(overlay)
                line=[tuple(p) for p in annotation["accepted_interval_uv"]]
                draw.line(line,fill="magenta",width=3)
                x,y=answer["cut_point_uv"]
                draw.line([(x-8,y),(x+8,y)],fill="white",width=2)
                draw.line([(x,y-8),(x,y+8)],fill="white",width=2)
                x,y=query
                draw.ellipse((x-7,y-7,x+7,y+7),outline="cyan",width=2)
                overlay.save(folder/"annotation_review.png")
                overlay.close()
            record.update(query_pixel_uv=query,nominal_pixel_uv=answer["cut_point_uv"],
                clarity=annotation["clarity"],rgb_sha256=sha256(folder/"inputs/rgb.png"),
                native_crop_sha256=sha256(folder/"inputs/query_crop.png"),
                maximum_roundtrip_error_px=max_error)
        records.append(record)
    verify_bindings(bindings)
    verify_bindings(implementation)
    summary=dict(state="native_clear_annotation_pilot_pending_visual_review",
        task=TASK,records=records,source_bindings=bindings,implementation_sha256=implementation,
        eligible_candidates=sum(r["eligible"] for r in records),training_approved=False,
        high_resolution_training_loader_implemented=False,native_sources_unchanged=True,
        no_new_source_family_or_view_cap_approval=True)
    write_json(output/"result.json",summary)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan",type=Path,required=True)
    p.add_argument("--capture",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(build(a.plan,a.capture,a.output),indent=2))


if __name__=="__main__":
    main()
