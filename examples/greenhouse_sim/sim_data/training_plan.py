"""Source-bound schedules for target-conditioned synthetic grounding, not robot actions.

The reviewed pilot and its split assignments are immutable. This separate schema
extends collection to all reserved families without promoting old review labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path

import numpy as np

from .audit import DEFAULT_PACK
from .collection_plan import source_reports, bindings_for, family_splits, native_rows
from .cut_regions import load_rule, rule_fingerprint
from .dataset_review import read_json, require, verify_bindings, write_json
from .depth_preview import sha256

SCHEMA = 'greenhouse.grounding_collection_plan.v1'


def configuration(targets=12, views=64, seed=0, *, vary_torso=False, renderer_mode=None, view_offset=0, clear_capture=False, opposite_aisle=False):
    require(type(targets) is int and 1 <= targets <= 36, 'Invalid target cap')
    require(type(views) is int and 1 <= views <= 160, 'Invalid view cap')
    require(type(seed) is int and seed == 0, 'Preserve the reviewed seed-0 family reservations')
    require(type(vary_torso) is bool,'Invalid torso sampling flag')
    require(type(clear_capture) is bool and (not clear_capture or vary_torso and views<=12),'Clear capture requires torso sampling and <=12 views per target')
    require(type(opposite_aisle) is bool and (not opposite_aisle or clear_capture),'Opposite aisle requires explicit clear capture')
    require(renderer_mode in (None,'RaytracedLighting','RealTimePathTracing'),'Unsupported native renderer')
    require(type(view_offset) is int and 0<=view_offset<=2048,'Invalid view shard offset')
    config=dict(targets_per_plant=targets, render_views_per_target=views, seed=seed,
                target_world_height_m=[1.05, 2.2], original_plant_root_z_m=.90,
                source_geometry='unmodified_native_components', collect_split='all_reserved_splits')
    if vary_torso: config['vary_torso']=True
    if renderer_mode is not None: config['renderer_mode']=renderer_mode
    if view_offset: config['view_offset']=view_offset
    if clear_capture: config['clear_capture']='robot_head_close_diffuse_v1'
    if opposite_aisle: config['opposite_aisle']=True
    return config


def schedule(reports, rule, config):
    require(config == configuration(config['targets_per_plant'], config['render_views_per_target'], config['seed'],
                                    vary_torso=config.get('vary_torso',False),renderer_mode=config.get('renderer_mode'),
                                    view_offset=config.get('view_offset',0),clear_capture=bool(config.get('clear_capture')),
                                    opposite_aisle=config.get('opposite_aisle',False)),
            'Changed grounding configuration')
    assignments = family_splits([r['plant_id'] for r in reports], config['seed'])
    jobs, decisions = [], []
    for report in sorted(reports, key=lambda r:r['plant_id']):
        rows, excluded = native_rows(report, rule)
        low, high = config['target_world_height_m']
        selected = [r for r in rows if low <= r['cut_region_proposal']['nominal']['point_plant_m'][2]+.9 <= high]
        # Stable varied heights instead of only the easiest two petioles.
        selected.sort(key=lambda r: hashlib.sha256(r['target_id'].encode()).hexdigest())
        selected = selected[:config['targets_per_plant']]
        decisions.append(dict(family=report['plant_id'], geometry_candidates=len(rows), selected=len(selected), exclusions=excluded))
        if selected:
            jobs.append(dict(job_id=f'job_{len(jobs)+1:03d}', plant_family=report['plant_id'],
                             split=assignments[report['plant_id']], source_manifest_path=report['manifest_path'],
                             targets=selected, max_rendered_views_per_target=config['render_views_per_target']))
    require(len(jobs)==len(reports), 'Every reserved family must have source targets')
    return dict(family_assignments=assignments, jobs=jobs, selection_audit=decisions)


def view_specs(original_x, target_x, target_id, count, *, vary_torso=False, view_offset=0, clear_capture=False, opposite_aisle=False):
    require(np.isfinite([original_x,target_x]).all() and original_x>target_x, 'Expected original +X aisle')
    require(type(count) is int and 1 <= count <= 160, 'Invalid sample count')
    require(type(clear_capture) is bool and (not clear_capture or vary_torso and count<=12),'Invalid clear view configuration')
    require(type(opposite_aisle) is bool and (not opposite_aisle or clear_capture),'Invalid opposite aisle selection')
    require(type(view_offset) is int and 0<=view_offset<=2048,'Invalid view shard offset')
    seed = int.from_bytes(hashlib.sha256(('grounding-v1:'+target_id).encode()).digest()[:8], 'little')
    rng = np.random.default_rng(seed)
    # Stratified continuous framing; label position is never a fixed three-point grid.
    result=[]
    proposals=24 if clear_capture else 3
    for i in range((view_offset+count)*proposals):
        approach=original_x-(target_x+float(rng.uniform(.30,.55))) if clear_capture else float(rng.uniform(.15,.40))
        root_x=original_x-approach
        if root_x-target_x<(.30-1e-9 if clear_capture else .35):
            continue
        result.append(dict(candidate_id=f'ground_{i:04d}', root_x_m=root_x,
            approach_from_original_m=approach, y_offset_m=float(rng.uniform(-.20,.20) if clear_capture else rng.uniform(-.4,.4)),
            root_yaw_degrees=float(rng.uniform(150,210)),
            desired_pixel_xy=[float(rng.uniform(.15,.85)*848),float(rng.uniform(.15,.85)*408)]))
    if vary_torso:
        from .training_views import with_postures
        result=with_postures(result,target_id)
    if opposite_aisle:
        # Separate opt-in static snapshots, not a validated route through the row.
        # Preserve original-side RNG and all joint postures exactly.
        result=[dict(s,root_x_m=2*target_x-s['root_x_m'],
                     approach_from_original_m=original_x-(2*target_x-s['root_x_m']),
                     root_yaw_degrees=s['root_yaw_degrees']-180,opposite_aisle=True) for s in result]
    # Offset counts proposal windows, not accepted labels. New shards never
    # repeat a previous window merely to fill a desired class/size quota.
    return result[view_offset*proposals:]


def build_plan(package, output, config):
    package,output=Path(package).resolve(),Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(package), 'Choose a new plan outside sources')
    reports,rule=source_reports(package),load_rule()
    result=dict(schema_version=SCHEMA, state='ready_for_synthetic_grounding_capture',
        created_utc=datetime.now(timezone.utc).isoformat(), package=str(package), configuration=config,
        cut_rule=rule, cut_rule_sha256=rule_fingerprint(rule), source_bindings_sha256=bindings_for(package,reports),
        **schedule(reports,rule,config), split_scope='target_source_families_only_shared_greenhouse_backdrop_context',
        difficulty='assigned_only_after_native_visibility_audit', training_dataset_approved=False,
        camera_requirement='mounted_RBY1_A_v1.2_head_D405_uncropped_848x408',
        implementation_sha256=sha256(Path(__file__)))
    verify_bindings(result['source_bindings_sha256'])
    output.mkdir(parents=True)
    write_json(output/'plan.json',result)
    return result


def load_plan(path):
    plan=read_json(path)
    require(plan.get('schema_version')==SCHEMA and plan.get('state')=='ready_for_synthetic_grounding_capture', 'Invalid grounding plan')
    require(plan.get('training_dataset_approved') is False, 'Plan cannot grant training approval')
    require(plan.get('split_scope')=='target_source_families_only_shared_greenhouse_backdrop_context', 'Unsupported split scope')
    require(rule_fingerprint(plan['cut_rule'])==plan['cut_rule_sha256'], 'Changed cut rule')
    reports=source_reports(plan['package'])
    require(bindings_for(plan['package'],reports)==plan['source_bindings_sha256'], 'Changed source assets')
    expected=schedule(reports,plan['cut_rule'],plan['configuration'])
    require(all(plan.get(k)==v for k,v in expected.items()), 'Changed schedule or family reservations')
    verify_bindings(plan['source_bindings_sha256'])
    return plan,reports


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,default=DEFAULT_PACK)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--targets',type=int,default=12)
    p.add_argument('--views',type=int,default=64)
    p.add_argument('--vary-torso',action='store_true')
    p.add_argument('--clear-capture',action='store_true',help='Opt-in closer real robot snapshots and uniform diffuse lighting; unchanged sensors/overlap guards')
    p.add_argument('--renderer-mode',choices=['RaytracedLighting','RealTimePathTracing'])
    p.add_argument('--view-offset',type=int,default=0)
    p.add_argument('--opposite-aisle',action='store_true',help='Separate negative-X static viewpoints; unchanged camera/joint/floor/overlap checks, not a base-motion route')
    a=p.parse_args(argv)
    r=build_plan(a.package,a.output,configuration(a.targets,a.views,vary_torso=a.vary_torso,
                                                renderer_mode=a.renderer_mode,view_offset=a.view_offset,clear_capture=a.clear_capture,opposite_aisle=a.opposite_aisle))
    print('GROUNDING_PLAN',len(r['jobs']),sum(len(j['targets'])*j['max_rendered_views_per_target'] for j in r['jobs']),flush=True)


if __name__=='__main__': main()
