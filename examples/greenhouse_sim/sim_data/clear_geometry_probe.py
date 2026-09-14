"""Bounded CPU-only viewpoint diagnosis, never collection or release approval.

Run with Isaac's Python for the matching USD build. No SimulationApp, rendering,
physics, robot commands, source edits or synthetic depth construction. Native
capture must independently rerun every admission and sensing check afterward.
"""
import argparse
from collections import Counter
from pathlib import Path
import time
import shutil
from types import SimpleNamespace
import numpy as np
from .dataset_review import require,read_json,write_json,verify_bindings
from .depth_preview import sha256


MODES={'original':{},'opposite':{'opposite_aisle':True},
       'oblique':{'oblique_clear':True},'opposite_oblique':{'opposite_aisle':True,'oblique_clear':True},
       'lean':{'lean_clear':True},'lean_oblique':{'lean_clear':True,'oblique_clear':True},
       'lean_opposite':{'lean_clear':True,'opposite_aisle':True},
       'near':{'near_clear':True},
       'near_orbit':{'near_clear':True,'oblique_clear':True,'orbit_clear':True},
       'near_orbit_opposite':{'near_clear':True,'oblique_clear':True,'orbit_clear':True,'opposite_aisle':True},
       'orbit':{'oblique_clear':True,'orbit_clear':True},
       'orbit_opposite':{'oblique_clear':True,'orbit_clear':True,'opposite_aisle':True},
       'orbit_lean':{'oblique_clear':True,'orbit_clear':True,'lean_clear':True},
       'orbit_lean_opposite':{'oblique_clear':True,'orbit_clear':True,'lean_clear':True,'opposite_aisle':True}}


def memory_reserve():
    from sim_physics.host_memory import memory_snapshot
    m=memory_snapshot()
    require(m.get('read_succeeded') and
            m['commit_limit_bytes']-m['committed_bytes']>=8*2**30 and
            m['physical_available_bytes']>=4*2**30,'CPU-only probe reserve unavailable')
    return m


def compare_native(decisions,reference):
    by_id={(r['target_review_id'],r['candidate_id']):r for r in reference['decisions']}
    for row in decisions:
        old=by_id.get((row['target_review_id'],row['candidate_id']))
        require(old is not None and row['state']==old['state'],'CPU/native geometry state mismatch')
        for key in ('predicted_diameter_px','predicted_interval_px','base_xy_m'):
            require((key in row)==(key in old),'CPU/native geometry field mismatch')
            if key in row:require(np.allclose(row[key],old[key],atol=1e-7,rtol=1e-7),'CPU/native geometry value mismatch')
    return dict(compared=len(decisions),geometry_decisions_match=True,rendered_visibility_checked=False)


def run(plan_path,job_id,output,reference_path,modes=('original','opposite','oblique','opposite_oblique')):
    from pxr import Usd,UsdGeom
    from launch_sim_data import load_local_payloads,populate
    from .collection_plan import load_plan
    from .capture_scene import capture_root,freeze_rigid_bodies
    from .robot_preview import add_robot_preview
    from .floor_alignment import PACKAGE_FLOOR
    from .collection_worker import prepare_views
    from .capture_pilot import source_hashes
    plan_path,output,reference_path=map(lambda p:Path(p).resolve(),(plan_path,output,reference_path))
    plan,_=load_plan(plan_path)
    require(modes and modes[0]=='original' and len(set(modes))==len(modes) and all(m in MODES for m in modes),'Original native comparison must precede unique supported modes')
    require(plan['configuration'].get('clear_capture')=='robot_head_close_diffuse_v1' and
            not plan['configuration'].get('view_offset') and not plan['configuration'].get('opposite_aisle') and
            not plan['configuration'].get('oblique_clear') and not plan['configuration'].get('lean_clear') and
            not plan['configuration'].get('orbit_clear') and not plan['configuration'].get('near_clear'),'Unmodified original-side clear plan required')
    job=next(j for j in plan['jobs'] if j['job_id']==job_id)
    require(job['split']=='train','Probe only training families; no model/test selection')
    require(not output.exists() and not output.is_relative_to(Path(plan['package'])),'New output outside source package required')
    native=read_json(reference_path.parent/'manifest.json')
    require(native['source_collection_plan_sha256']==sha256(plan_path) and native['collection_job_id']==job_id and
            native.get('source_assets_unchanged') is True,'Native reference binding mismatch')
    reference=read_json(reference_path);reference_hash=sha256(reference_path)
    snapshot=memory_reserve();output.mkdir(parents=True)
    shutil.copyfile(__file__,output/'probe_source.py')
    write_json(output/'request.json',dict(plan=str(plan_path),plan_sha256=sha256(plan_path),job_id=job_id,
        reference_sha256=reference_hash,usd_version=Usd.GetVersion(),script_sha256=sha256(output/'probe_source.py'),
        cpu_only=True,simulation_app_started=False,rendering=False,physics=False,training_approved=False,
        views_per_target=3,proposal_modes=list(modes),memory_snapshot=snapshot))
    started=time.monotonic()
    wrapper=capture_root(Path(plan['package'])/'house/green_house_base.usd')
    stage=Usd.Stage.Open(wrapper,load=Usd.Stage.LoadNone)
    UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    stage.SetEditTarget(stage.GetSessionLayer());excluded=load_local_payloads(stage)
    records=[]
    gutter_x,counts=populate(stage,Path(plan['package']),SimpleNamespace(update=lambda:None),records,job['plant_family'])
    require(counts==native['scene_counts'],'CPU/native populated scene counts differ')
    variants=[dict(variant_id=Path(r['manifest_path']).parent.name,plant_root=r['plant_root']) for r in records]
    robot=add_robot_preview(stage,gutter_x=gutter_x,floor_path=PACKAGE_FLOOR,right_tool='gripper')
    freeze_rigid_bodies(stage);before=source_hashes(stage);summaries={}
    for mode in modes:
        options=MODES[mode]
        require(time.monotonic()-started<1800,'Bounded geometry probe exceeded 30 minutes')
        memory_reserve()
        prepared=prepare_views(stage,robot,variants,job['targets'],3,grounding=True,vary_torso=True,
            clear_capture=True,**options)
        summary=dict(selected=sum(map(len,prepared['selected'].values())),
            targets_with_candidates=sum(bool(v) for v in prepared['selected'].values()),
            states=dict(Counter(r['state'] for r in prepared['decisions'])),rendered_visibility_checked=False)
        if mode=='original':summary['native_comparison']=compare_native(prepared['decisions'],reference)
        write_json(output/(mode+'_views.json'),prepared)
        summaries[mode]=summary;print('CPU_GEOMETRY_PROBE',mode,summary,flush=True)
    require(source_hashes(stage)==before and sha256(reference_path)==reference_hash,'Source/reference changed')
    require(not any(not l.anonymous and l.dirty for l in stage.GetUsedLayers()),'Source USD dirtied')
    verify_bindings(plan['source_bindings_sha256'])
    result=dict(state='cpu_geometry_diagnosis_pending_native_capture',summaries=summaries,scene_counts=counts,
        excluded_unbundled_props=excluded,elapsed_s=time.monotonic()-started,training_approved=False)
    write_json(output/'result.json',result);return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('plan','output','native-reference'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--job',required=True)
    p.add_argument('--modes',nargs='+',choices=list(MODES),default=['original','opposite','oblique','opposite_oblique'])
    a=p.parse_args(argv)
    run(a.plan,a.job,a.output,a.native_reference,a.modes)


if __name__=='__main__':main()
