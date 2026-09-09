"""Prepare the same static robot viewpoints in synchronous USD, without Kit.

This produces geometry-screened proposals only, never RGB/depth or approval.
Actual capture must still reconstruct each pose and pass native rendering gates.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np

from .collection_plan import load_plan
from .dataset_review import read_json,require,verify_bindings,write_json
from .depth_preview import sha256


def same_tree(a,b):
    if isinstance(a,dict): return isinstance(b,dict) and set(a)==set(b) and all(same_tree(v,b[k]) for k,v in a.items())
    if isinstance(a,list): return isinstance(b,list) and len(a)==len(b) and all(same_tree(x,y) for x,y in zip(a,b))
    if type(a) in (float,int) and type(b) in (float,int): return bool(np.isclose(a,b,atol=1e-9,rtol=0))
    return type(a)==type(b) and a==b


def prepare(plan_path,job_id,output,*,reference=None):
    from pxr import Usd,UsdGeom
    from launch_sim_data import load_local_payloads,populate
    from .capture_scene import capture_root,freeze_rigid_bodies,scene_guard
    from .capture_pilot import source_hashes
    from .collection_worker import prepare_views
    from .floor_alignment import PACKAGE_FLOOR
    from .robot_preview import add_robot_preview
    started=time.perf_counter(); plan_path=Path(plan_path).resolve(); output=Path(output).resolve()
    plan,_=load_plan(plan_path)
    require(plan['schema_version']=='greenhouse.grounding_collection_plan.v1','Expected grounding collection plan')
    require(not output.exists() and not output.is_relative_to(Path(plan['package'])),'Choose new preplan output outside source assets')
    job=next((j for j in plan['jobs'] if j['job_id']==job_id),None)
    require(job is not None,'Unknown collection job')
    package=Path(plan['package'])
    stage=Usd.Stage.Open(capture_root(package/'house/green_house_base.usd'),load=Usd.Stage.LoadNone)
    UsdGeom.SetStageMetersPerUnit(stage,1); UsdGeom.SetStageUpAxis(stage,'Z')
    stage.SetEditTarget(stage.GetSessionLayer())
    excluded=load_local_payloads(stage); records=[]
    # USD references compose synchronously here. This no-op satisfies the
    # scene-population UI refresh hook; it is NOT a fake render or simulator tick.
    cx,counts=populate(stage,package,SimpleNamespace(update=lambda:None),records,review_plant=job['plant_family'])
    variants=[]
    for record in records:
        family=Path(record['manifest_path']).parent.name
        variants.append(dict(variant_id=family,source_plant_id=family,split_group=family,
                             plant_root=record['plant_root'],added_components={},added_component_paths={},source_geometry_modified=False))
    robot=add_robot_preview(stage,gutter_x=cx,floor_path=PACKAGE_FLOOR,right_tool='gripper')
    freeze_rigid_bodies(stage)
    sys.path.insert(0,str(package/'env_panel'))
    from tomato_env import daylight
    lighting=daylight.apply(stage,day=172,minutes=13*60,intensity=1500,dome_intensity=1200)
    baseline=scene_guard(stage); hashes=source_hashes(stage); prepared_started=time.perf_counter()
    prepared=prepare_views(stage,robot,variants,job['targets'],job['max_rendered_views_per_target'],grounding=True,
                           vary_torso=plan['configuration'].get('vary_torso',False),view_offset=plan['configuration'].get('view_offset',0))
    require(scene_guard(stage)==baseline,'Preplanning changed the restored scene')
    require(source_hashes(stage)==hashes and not any(not l.anonymous and l.dirty for l in stage.GetUsedLayers()),'Preplanning changed source layers')
    verify_bindings(plan['source_bindings_sha256'])
    comparison=None
    if reference is not None:
        comparison=dict(reference_path=str(Path(reference).resolve()),reference_sha256=sha256(reference),
                        all_plan_fields_match=same_tree(prepared,read_json(reference)),numeric_tolerance=1e-9)
        require(comparison['all_plan_fields_match'],'Offline preplan differs from native-worker reference')
    output.mkdir(parents=True)
    write_json(output/'planned_views.json',prepared)
    receipt=dict(schema_version='greenhouse.synchronous_USD_view_preplan.v1',state='geometry_proposals_not_rendered_or_approved',
                 source_plan_path=str(plan_path),source_plan_sha256=sha256(plan_path),job_id=job_id,plant_family=job['plant_family'],
                 source_usd_sha256=hashes,scene_snapshot_sha256=baseline,scene_counts=counts,lighting=lighting,
                 excluded_external_roots=excluded,planned_views_sha256=sha256(output/'planned_views.json'),
                 reference_comparison=comparison,preparation_seconds=time.perf_counter()-prepared_started,
                 total_seconds=time.perf_counter()-started,
                 implementation_sha256={p:sha256(Path(__file__).with_name(p)) for p in
                    ('training_preplan.py','collection_worker.py','capture_scene.py','capture_viewpoints.py','training_screen.py','training_plan.py','training_views.py')},
                 image_capture_performed=False,training_release_approved=False)
    write_json(output/'preplan.json',receipt)
    return receipt


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True); p.add_argument('--job',required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-planned',type=Path)
    a=p.parse_args(argv); r=prepare(a.plan,a.job,a.output,reference=a.reference_planned)
    print('USD_PREPLAN_COMPLETE',r['total_seconds'],r['preparation_seconds'],r['reference_comparison'],flush=True)


if __name__=='__main__': main()
