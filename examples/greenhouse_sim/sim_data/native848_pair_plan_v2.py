"""Explicit native848 generated-pair diagnostic; no legacy1696 qualification claim.

Reuses the established full-scene reconstruction and TRAIN-fitted generator.
This plan authorizes a bounded two-frame experiment, never target novelty,
training labels, a source-cap reset, or physical robot motion.
"""
from copy import deepcopy
from pathlib import Path
import ast, importlib.util
import numpy as np
from .dataset_review import read_json, require, verify_bindings, safe_file
from .depth_preview import sha256
from .capture_sensor import LEGACY_RESOLUTION, calibration_for_native_resolution
from .native_greenhouse_pair import load_source

SCHEMA='greenhouse.generated_native848_pair_plan.v2'
SAMPLE_SCHEMA='greenhouse.generated_native848_pair_sample.v2'
RESULT_STATE='generated_native848_pair_v2_captured_pending_independent_audit'
FLAGS=dict(training_approved=False,independent_target_novelty_approved=False,source_cap_reset=False,
    physical_motion_commanded=False,paired_848_1696_proof=False,label_contract_qualified=False)

def implementation_bindings():
    root=Path(__file__).resolve().parent
    search=(root.parent,root.parents[1])
    pending=[Path(__file__).resolve(),root/'native848_pair_worker_v2.py',root/'native848_pair_audit_v2.py']
    found={}
    while pending:
        path=pending.pop().resolve()
        if str(path) in found:continue
        raw=path.read_bytes();found[str(path)]=sha256(path)
        base=next(p for p in search if path.is_relative_to(p))
        package='.'.join(path.relative_to(base).parts[:-1])
        for node in ast.walk(ast.parse(raw)):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom):
                name='.'*node.level+(node.module or '')
                name=importlib.util.resolve_name(name,package) if node.level else name
                names=[name]+[name+'.'+a.name for a in node.names if a.name!='*']
            for name in names:
                for folder in search:
                    candidate=folder.joinpath(*name.split('.'))
                    pending.extend(p for p in (candidate.with_suffix('.py'),candidate/'__init__.py') if p.is_file())
    return dict(sorted(found.items()))

def prepare(source_capture,sample_id,variant_directory,*,review_path,workspace_path):
    from .generated_capture import prepare_plan
    source_capture=Path(source_capture).resolve();variant_directory=Path(variant_directory).resolve()
    manifest,sample,_=load_source(source_capture,sample_id)
    # This helper derives scene/asset ancestry. Its1696 execution schema is discarded.
    base=prepare_plan(source_capture,sample_id,variant_directory,variant_directory.parent/'unused_native848_no_hires_prerequisite')
    fields=('source_capture','source_sample','source_collection_plan','source_row','generated_row','variant_directory',
        'original_variant','source_bindings','expected_scene_counts','expected_robot_snapshot','source_family',
        'split','split_group','conservative_view_cap_group','generator_version','generator_code_bindings')
    plan={k:deepcopy(base[k]) for k in fields}
    rp=Path(review_path).resolve();wp=Path(workspace_path).resolve()
    reviews=read_json(rp);proof=read_json(wp)
    rgb=source_capture/sample_id/'inputs/rgb.png'
    matched=[r for r in reviews if r['rgb_sha256']==sha256(rgb) and r['decision']=='accept']
    require(len(matched)==1 and proof['sample_sha256']==sha256(source_capture/sample_id/'sample.json')
        and proof['result']['workspace_passed'] is True,'Exact reviewed/workspace original reference required')
    plan.update(schema_version=SCHEMA,state='native848_pair_planned_not_executed',resolution=list(LEGACY_RESOLUTION),
        camera_path=sample['calibration']['camera_path'],expected_calibration=deepcopy(sample['calibration']),
        expected_original_world={k:deepcopy(sample['supervision'][k]) for k in ('plant_to_world_usd_row_vectors','nominal_world_m','interval_world_m')},
        scene_variants=deepcopy(manifest['variants']),modes=['original_control','generated_variant'],sample_count_limit=2,
        render_subframes=56,instance_backend='legacy',sensor_authority='existing_native848_source_plus_new_same_callback_validation',
        source_review=dict(path=str(rp),sha256=sha256(rp),record=matched[0]),
        source_workspace=dict(path=str(wp),sha256=sha256(wp),generated_target_inherits_workspace=False),
        **FLAGS)
    plan['source_bindings'].update({str(rp):sha256(rp),str(wp):sha256(wp)})
    package=Path(manifest['package']).resolve()
    plan['scene_code_bindings']={str(p):sha256(p) for p in sorted((package/'env_panel/tomato_env').glob('*.py'))}
    require(str(package/'env_panel/tomato_env/daylight.py') in plan['scene_code_bindings'],'Pinned daylight source required')
    plan['implementation_bindings']=implementation_bindings()
    check(plan,full=True)
    return plan

def check(plan,*,full=False):
    require(plan['schema_version']==SCHEMA and plan['state']=='native848_pair_planned_not_executed'
        and plan['resolution']==list(LEGACY_RESOLUTION) and all(plan[k] is v for k,v in FLAGS.items()),'Wrong native848 pair scope')
    require(plan['modes']==['original_control','generated_variant'] and plan['sample_count_limit']==2
        and plan['render_subframes']==56 and plan['instance_backend']=='legacy','Changed pair profile')
    verify_bindings(plan['source_bindings']);verify_bindings(plan['generator_code_bindings']);verify_bindings(plan['implementation_bindings']);verify_bindings(plan['scene_code_bindings'])
    require(plan['implementation_bindings']==implementation_bindings(),'Implementation closure differs')
    manifest,sample,_=load_source(plan['source_capture'],plan['source_sample'])
    require(plan['expected_calibration']==sample['calibration']==calibration_for_native_resolution(sample['calibration'],LEGACY_RESOLUTION)
        and plan['expected_robot_snapshot']==sample['robot_snapshot'],'Changed native848 reference')
    source=read_json(plan['source_collection_plan'])
    require(source['family_assignments'][plan['source_family']]=='train',
        'Frozen TRAIN source required')
    require(plan['split']=='train' and plan['split_group']==plan['source_family']
        and plan['conservative_view_cap_group']==plan['source_row']['target_id'],'Source cap/split changed')
    require(plan['source_row']['cut_region_proposal']==sample['supervision']['cut_region_proposal']
        and plan['generated_row']['cut_region_proposal']!=plan['source_row']['cut_region_proposal'],'Changed or repeated anatomy')
    require(plan['scene_variants']==manifest['variants'] and plan['expected_scene_counts']==manifest['scene_counts'],'Changed full-scene population')
    for k,v in plan['expected_original_world'].items():require(v==sample['supervision'][k],'Changed original physical geometry')
    if full:
        from .plant_variant_catalogue import load_for_inspection
        generated=load_for_inspection(plan['variant_directory'],plan['source_collection_plan'])
        require(generated['source_family']==plan['source_family'] and plan['generated_row'] in generated['rows'],'Generated geometry replay differs')
        return generated
    return None
