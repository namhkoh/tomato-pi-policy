"""Finite far-merge recipe: keep whole newly-near plants; reuse proven far USDs.

All144 full plants remain. This never removes a source organ or changes its
material. Foreground and newly-near background plants remain wholly ordinary.
The recipe is not collision/workspace/native/annotation admission.
"""
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
import argparse
import sys
import time
import numpy as np
from pxr import Usd, UsdGeom, Gf
from . import native848_farmerge_annotation_v1 as audit
from . import native848_fully_labeled_coverage_v3 as coverage
from . import native848_far_component_merge_prototype_v1 as prototype_api
from . import native848_fully_labeled_scene_v1 as ordinary
from .native848_bulk_workspace_v1 import BulkWorkspace
from .native_greenhouse_pair import assert_same_camera
from .native848_bulk_io_v1 import save_json
from .dataset_review import require, read_json, verify_bindings
from .capture_contract import fingerprint
from .depth_preview import sha256

SCHEMA='greenhouse.native848_finite_farmerge_population.v1'
EPS=1e-6


def affected_whole_plants(selection, all_bounds):
    """Any component/9mm entering either sphere retains its entire original plant."""
    rows=selection['selected_components'];lo=np.array([r['original_world_min'] for r in rows])-EPS
    hi=np.array([r['original_world_max'] for r in rows])+EPS
    near=np.zeros(len(rows),bool);point_near=np.zeros(len(rows),bool)
    for bound in all_bounds:
        for arm in bound['arms'].values():
            center=np.asarray(arm['shoulder_world_m']);reach=arm['conservative_probe_reach_m']+bound['margin_m']
            near |= np.linalg.norm(np.maximum(np.maximum(lo-center,center-hi),0),axis=1)<=reach
            for i,row in enumerate(rows):
                if row['organ_type']=='sub_stem':
                    p=row['petiole_exclusion'];require(p is not None and p['nominal_arc_m']==.009,'Exact original9mm selection evidence missing')
                    if np.linalg.norm(np.asarray(p['nominal_world_m'])-center)<=reach+EPS:point_near[i]=True
    bad=[r for i,r in enumerate(rows) if near[i] or point_near[i]]
    roots=sorted({'/'.join(r['prim_path'].split('/')[:4]) for r in bad})
    return roots,[dict(component_index=r['component_index'],prim_path=r['prim_path']) for r in bad]


def background_match(actual, reference, primary):
    old={r['plant_root']:r for r in reference['all_plant_roots']}
    new={r['plant_root']:r for r in actual['all_plant_roots']}
    require(len(old)==len(new)==144 and set(old)==set(new),'Exact144 original slots required')
    for root,row in old.items():
        if root==primary:continue
        require(all(new[root][k]==row[k] for k in ('source_family','manifest_sha256','plant_to_world_usd_row_vectors')),
                'Background geometry/material source or placement differs: '+root)


def candidate_bounds(candidates,cpu,plan,checker):
    """Reuse exact original-scene CPU robot states, never inherit generated collision."""
    require(cpu['plan_sha256']==candidates['source_camera_plan']['sha256'] and cpu['all_selected_pose_screens_passed'] is True,
            'Pinned original camera CPU proof required')
    poses={p['sample_id']:p for p in cpu['poses']};result=[]
    for record in candidates['records']:
        require(record['source_camera_plan']==candidates['source_camera_plan'],'Mixed source camera plans')
        require(record['source_camera_sample_id'] in plan['selected_sample_ids'],'Source camera not scheduled')
        old=poses[record['source_camera_sample_id']];proof=old['proof'];snapshot=deepcopy(proof['actual_robot_snapshot'])
        require(old['passed'] is True and proof['screen']['passed'] is True,'Original source camera CPU was held')
        assert_same_camera(record['calibration'],proof['actual_calibration'])
        require(all(record[k]==snapshot[k] for k in ('joint_degrees','camera_to_head_column_vectors'))
                and np.allclose(record['robot_root_to_world_usd_row_vectors'],snapshot['robot_root_to_world_usd_row_vectors'],atol=1e-9,rtol=0),
                'Candidate camera/root/mount/joints are not the authenticated original CPU pose')
        snapshot['visual_bound_screen']=proof['screen']
        result.append(dict(sample_id=record['sample_id'],source_camera_sample_id=record['source_camera_sample_id'],
            bounds=coverage.robot_bounds(dict(calibration=record['calibration'],robot_snapshot=snapshot),checker),
            proof_scope='original_scene_robot_FK_only_generated_collision_not_inherited',generated_scene_collision_pending=True))
    return result


def prepare(source_screen_pin,candidates_pin,source_cpu_pin,output):
    started=time.perf_counter();out=Path(output).resolve()
    require(out.is_relative_to(audit.ROOT/'data/sim_data/diagnostics') and not out.exists(),'Create-only local population recipe')
    inputs=audit.Inputs();screen=inputs.read(source_screen_pin);candidates=inputs.read(candidates_pin);cpu=inputs.read(source_cpu_pin)
    require(len(screen['records'])==4 and len(candidates['records'])==4 and candidates['source_family']=='seed103_full','Only the explicit4+4 finite scope')
    camera_plan=inputs.read(candidates['source_camera_plan']);verify_bindings(cpu['source_bindings']);inputs.bindings.update(cpu['source_bindings'])
    verify_bindings(candidates['source_bindings']);inputs.bindings.update(candidates['source_bindings'])
    prototype=inputs.read(audit.api.PROTOTYPE);selection=inputs.read(prototype['selection'])
    base=inputs.read(screen['records'][-1]['logical_census']);plants={p['plant_root']:p for p in base['all_plant_roots']}
    primary=next(p['plant_root'] for p in plants.values() if p['decision']=='retained_exact_foreground')
    checker=BulkWorkspace();poses=[]
    try:
        for record in screen['records']:
            obs=inputs.read(record['native_observation']);logical=inputs.read(record['logical_census']);background_match(logical,base,primary)
            poses.append(dict(sample_id=record['frame_id'],bounds=coverage.robot_bounds(dict(calibration=obs['calibration'],robot_snapshot=obs['robot_snapshot']),checker),
                              original_completed_observation=record['native_observation']))
        background_match(inputs.read(cpu['full_scene_census']),base,primary)
        poses.extend(candidate_bounds(candidates,cpu,camera_plan,checker));inputs.bindings.update(checker.bindings)
    finally:checker.finish()
    retained_roots,causes=affected_whole_plants(selection,[r['bounds'] for r in poses])
    require(primary not in retained_roots and len(retained_roots)<=12,'Unexpected foreground/large replacement scope; stop for review')
    original_selected={r['component_index']:r for r in selection['selected_components']}
    selected=[r for r in selection['selected_components'] if not any(r['prim_path'].startswith(root+'/') for root in retained_roots)]
    catalogue=inputs.read(dict(path=str(prototype_api.B/'ordinary_catalogue.json'),sha256=prototype_api.PINS[str(prototype_api.B/'ordinary_catalogue.json')]))
    protected=sorted({r['component_index'] for r in catalogue}-{r['component_index'] for r in selected})
    finite_selection=dict(schema='greenhouse.native848_finite_far_component_selection.v1',base_selection=prototype['selection'],
        selected_components=selected,protected_component_indices=protected,whole_plant_retention_roots=retained_roots,
        newly_near_causes=causes,finite_robot_bounds=poses,world_point_error_bound_m=EPS,
        all_parent_and_descendant_components_preserved_in_retained_plants=True,visibility_used_for_selection=False,
        foreground_always_fully_retained=True,generated_geometry_or_collision_admission=False)
    for pose in poses:
        require(audit.actual_coarse_bounds(dict(selected={r['component_index']:r for r in selected}),pose['bounds'])['all_components_outside_both_actual_arm_bounds'],'Remaining packed surface enters a finite pose bound')
    out.mkdir(parents=True);save_json(out/'selection.json',finite_selection)
    # Original source USD references are restored in whole affected slots. No
    # geometry/material arrays are rewritten; unchanged packed slots are reused.
    population_path=out/'population_render_only.usda';stage=Usd.Stage.CreateNew(str(population_path))
    UsdGeom.Xform.Define(stage,'/World');UsdGeom.Xform.Define(stage,'/World/PackPlants')
    templates={};entries=[]
    old_entries={r['plant_root']:r for r in prototype['plants']}
    for root,plant in sorted(plants.items()):
        x=UsdGeom.Xform.Define(stage,root);x.AddTransformOp().Set(Gf.Matrix4d(plant['plant_to_world_usd_row_vectors']))
        if root in retained_roots:
            manifest=Path(plant['manifest_path']);inputs.bound(dict(path=str(manifest),sha256=plant['manifest_sha256']))
            family=plant['source_family']
            if family not in templates:templates[family]=ordinary._template(manifest)
            template,relative=templates[family]
            # Persist this ordinary hierarchy into its own create-only layer;
            # immutable component/material assets remain external references.
            local=out/(root.rsplit('/',1)[1]+'_complete.usda');template.GetRootLayer().Export(str(local))
            x.GetPrim().GetReferences().AddReference(local.as_posix(),'/Plant')
            entries.append(dict(plant_root=root,mode='complete_original_components',artifact=audit.pin(local),coarse_groups=0))
        else:
            old=old_entries[root];inputs.bound(old['artifact']);inputs.bound(old['provenance'])
            x.GetPrim().GetReferences().AddReference(Path(old['artifact']['path']).as_posix(),'/Plant')
            entries.append(dict(old,mode='original_foreground' if root==primary else 'unchanged_packed_background'))
    stage.GetRootLayer().Save();templates.clear();stage=None
    # Reopen actual saved overlay and compare newly-retained mesh transforms
    # to the complete ordinary oracle. Arrays/materials are original refs.
    stage=Usd.Stage.Open(str(population_path));xc=UsdGeom.XformCache()
    old_meshes=inputs.read(dict(path=str(prototype_api.B/'ordinary_meshes.json'),sha256=prototype_api.PINS[str(prototype_api.B/'ordinary_meshes.json')]))
    verified=0
    for row in old_meshes:
        if not any(row['path'].startswith(root+'/') for root in retained_roots):continue
        prim=stage.GetPrimAtPath(row['path']);require(prim and prim.IsActive() and prim.IsA(UsdGeom.Mesh),'Restored original mesh missing')
        require(np.array_equal(np.asarray(xc.GetLocalToWorldTransform(prim),float),row['world_matrix'])
                and [str(p) for p in prim.GetRelationship('material:binding').GetTargets()]==row['material_targets']
                and UsdGeom.Imageable(prim).ComputeVisibility()==row['visibility']
                and UsdGeom.Imageable(prim).ComputePurpose()==row['purpose'],'Restored source transform/material/visibility differs')
        verified+=1
    require(len(stage.GetPrimAtPath('/World/PackPlants').GetChildren())==144,'Plant population changed')
    meshes=sum(p.IsA(UsdGeom.Mesh) for p in stage.Traverse());stage=None
    inputs.bindings[str(Path(__file__).resolve())]=sha256(__file__)
    inputs.bindings[str(Path(ordinary.__file__).resolve())]=sha256(ordinary.__file__)
    verify_bindings(inputs.bindings)
    result=dict(schema=SCHEMA,source_screen=source_screen_pin,additional_candidates=candidates_pin,original_camera_CPU=source_cpu_pin,
        base_prototype=audit.api.PROTOTYPE,base_logical_census=screen['records'][-1]['logical_census'],base_catalogue=dict(path=str(prototype_api.B/'ordinary_catalogue.json'),sha256=prototype_api.PINS[str(prototype_api.B/'ordinary_catalogue.json')]),
        primary_root=primary,plant_slots=144,background_source_assignments_and_transforms_preserved=True,
        whole_original_background_roots=retained_roots,unchanged_packed_background_count=143-len(retained_roots),
        original_restored_meshes_verified=verified,render_mesh_count=meshes,original_full_render_mesh_count=prototype['original_meshes'],
        selection=audit.pin(out/'selection.json'),population=audit.pin(population_path),plants=entries,
        all_plant_shapes_and_materials_preserved=True,no_geometry_removed=True,whole_plant_dependency_closure=True,
        geometry_array_rewriting=False,foreground_runtime_geometry_must_be_kept=True,
        runtime_background_identity_check_required=True,runtime_component_index_remap_by_exact_path_required=True,
        original_collision_shadow_required=True,generated_collision_admission_pending=True,native_census_pending=True,
        render_speed_not_measured_for_this_recipe=True,source_bindings=inputs.bindings,elapsed_seconds=time.perf_counter()-started,
        native_launched=False,native_launch_ready=False,training_approved=False,accepted_training_increment=0)
    save_json(out/'result.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for key in ('source-screen','candidates','source-cpu','output'):parser.add_argument('--'+key,required=True)
    for key in ('source-screen','candidates','source-cpu'):parser.add_argument('--'+key+'-sha256',required=True)
    args=parser.parse_args();result=prepare(dict(path=args.source_screen,sha256=args.source_screen_sha256),dict(path=args.candidates,sha256=args.candidates_sha256),dict(path=args.source_cpu,sha256=args.source_cpu_sha256),args.output)
    print(dict(result=audit.pin(Path(args.output)/'result.json'),retained=result['whole_original_background_roots'],meshes=result['render_mesh_count']),flush=True)
