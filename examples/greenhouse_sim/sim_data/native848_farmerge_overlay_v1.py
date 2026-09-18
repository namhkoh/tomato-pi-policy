"""Install the finite far overlay while preserving the current generated foreground.

The collision stage is cloned before replacing any render background. All
near roots and the foreground remain untouched. No native app is launched.
"""
from pathlib import Path
from copy import deepcopy
import hashlib
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Sdf, Gf
from . import native848_farmerge_finite_population_v1 as finite
from . import native848_farmerge_pair_scene_v1 as frozen
from . import native848_farmerge_annotation_v1 as annotation
from . import native848_fully_labeled_coverage_v3 as coverage
from . import native848_far_component_merge_prototype_v1 as prototype
from .native848_far_component_merge_prototype_v1 import graph
from .capture_contract import fingerprint
from .dataset_review import require, verify_bindings
from .depth_preview import sha256

SCHEMA='greenhouse.native848_finite_farmerge_render_census.v1'
OriginalCollisionBridge=frozen.OriginalCollisionBridge
classify_mapping=frozen.classify_mapping


def remap_members(catalogue, base_catalogue, selected_rows, group_root):
    """Global indices are scene-local; only exact path and source identity transfer."""
    current={r['prim_path']:r for r in catalogue}
    require(len(current)==len(catalogue),'Duplicate current component path')
    base={r['component_index']:r for r in base_catalogue};mapping={}
    for selected in selected_rows:
        if not selected['prim_path'].startswith(group_root+'/'):continue
        old=base[selected['component_index']];new=current.get(old['prim_path'])
        require(selected['prim_path']==old['prim_path'],'Selected source index/path mismatch')
        require(new is not None and all(new[k]==old[k] for k in ('prim_path','component_id','organ_type','source_plant_id','split_group')),
                'Background component identity changed during index remap')
        require(new.get('source_geometry_modified') is not True and new.get('geometry_source_id',new['source_plant_id'])==new['source_plant_id'],
                'Generated geometry cannot enter reused background packing')
        mapping[str(old['component_index'])]=new['component_index']
    require(len(set(mapping.values()))==len(mapping),'Repeated remapped component')
    return mapping


def retained_signature(mesh):
    # Original meshes may legitimately have no UV/normals. Absence is itself
    # preserved, not fabricated to satisfy the packed-mesh ABI.
    def value(v):return None if v is None else prototype.array_pin(np.asarray(v))
    primvars={}
    for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        if not pv.HasValue():continue
        primvars[pv.GetPrimvarName()]=dict(value=value(pv.Get()),indices=value(pv.GetIndices()),
            interpolation=pv.GetInterpolation(),element_size=pv.GetElementSize())
    return dict(points=value(mesh.GetPointsAttr().Get()),face_counts=value(mesh.GetFaceVertexCountsAttr().Get()),
        face_indices=value(mesh.GetFaceVertexIndicesAttr().Get()),normals=value(mesh.GetNormalsAttr().Get()),primvars=primvars,
        subdivision=mesh.GetSubdivisionSchemeAttr().Get(),orientation=mesh.GetOrientationAttr().Get(),
        double_sided=mesh.GetDoubleSidedAttr().Get(),normals_interpolation=mesh.GetNormalsInterpolation())


def snapshot_mesh(prim,cache):
    return dict(path=str(prim.GetPath()),world_matrix=np.asarray(cache.GetLocalToWorldTransform(prim),float).tolist(),
        material_targets=[str(p) for p in prim.GetRelationship('material:binding').GetTargets()],
        visibility=str(UsdGeom.Imageable(prim).ComputeVisibility()),purpose=str(UsdGeom.Imageable(prim).ComputePurpose()),
        signature=retained_signature(UsdGeom.Mesh(prim)))


def verify_overlay(overlay):
    stage=overlay['stage'];cache=UsdGeom.XformCache();root=stage.GetPrimAtPath('/World/PackPlants')
    children=list(root.GetChildren())
    require(len(children)==144 and all(p.IsActive() and not p.IsInstanceable() for p in children),'All144 active ordinary plant slots required')
    for old in overlay['retained_meshes']:
        prim=stage.GetPrimAtPath(old['path'])
        require(prim and prim.IsActive() and prim.IsA(UsdGeom.Mesh),'Retained foreground/near mesh absent')
        require(snapshot_mesh(prim,cache)==old,'Retained foreground/near shape, material or transform changed')
    groups=[]
    for group in overlay['render_groups']:
        mesh=UsdGeom.Mesh(stage.GetPrimAtPath(group['prim_path']))
        require(mesh and frozen.mesh_signature(mesh)==group['expected_signature'],'Packed source arrays changed')
        require(np.array_equal(np.asarray(cache.GetLocalToWorldTransform(mesh.GetPrim()),float),group['plant_to_world_usd_row_vectors']),
                'Packed slot transform changed')
        materials={}
        for subset in UsdShade.MaterialBindingAPI(mesh).GetMaterialBindSubsets():
            material,_=UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()
            require(material,'Packed material absent');key=subset.GetPrim().GetName()[2:]
            require(fingerprint(graph(material,overlay['native_texture_bindings']))==key,'Packed material graph/texture differs')
            materials[str(subset.GetPath())]=dict(material_graph_sha256=key,faces=fingerprint(list(subset.GetIndicesAttr().Get())))
        require(materials==group['expected_material_subsets'],'Packed material face partition differs')
        groups.append({k:deepcopy(v) for k,v in group.items() if k not in ('expected_signature','expected_material_subsets')})
    actual=sum(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(root))
    require(actual==len(overlay['retained_meshes'])+len(groups),'Unexpected render geometry outside retained/coarse partition')
    value=dict(schema=SCHEMA,logical_census_sha256=fingerprint(overlay['logical_census']),
        logical_anatomy_components=len(overlay['logical_catalogue']),logical_plant_slots=144,
        actual_native_render_meshes=actual,actual_native_retained_original_meshes=len(overlay['retained_meshes']),
        actual_native_coarse_groups=len(groups),native_coarse_group_catalogue=groups,
        finite_population_recipe=overlay['recipe_pin'],selection=overlay['selection_pin'],
        foreground_root=overlay['primary_root'],foreground_kept_from_current_generated_population=True,
        fully_retained_background_roots=overlay['whole_roots'],background_shapes_materials_and_placements_preserved=True,
        packed_world_coordinate_roundoff_bound_m=1e-6,individual_far_component_pixel_IDs_claimed=False,
        original_collision_representation_retained=True,collision_snapshot=overlay['collision_snapshot'],
        retained_geometry_signature_sha256=fingerprint(overlay['retained_meshes']),
        complete_native_renderer_ID_mapping_still_required=True,training_approved=False,accepted_training_increment=0)
    value['deterministic_census_sha256']=fingerprint(value)
    return value


def check_coarse_bounds(overlay,bounds):
    proof=annotation.actual_coarse_bounds(overlay['coarse_source_selection'],bounds)
    require(proof['all_components_outside_both_actual_arm_bounds'],'Packed component surface enters actual robot bound')
    return proof


def clone_collision_stage(stage):
    before=stage.GetSessionLayer().ExportToString();root_before=stage.GetRootLayer().ExportToString()
    session=Sdf.Layer.CreateAnonymous('complete_generated_collision_session.usda');session.TransferContent(stage.GetSessionLayer())
    require(session.ExportToString()==before,'Collision shadow session copy differs')
    collision=Usd.Stage.Open(stage.GetRootLayer(),session,load=Usd.Stage.LoadNone);collision.SetLoadRules(stage.GetLoadRules());collision.SetEditTarget(session)
    require(collision.GetSessionLayer()!=stage.GetSessionLayer(),'Collision authoring must be independent')
    return collision,session,before,root_before


def install_overlay(scene,population,catalogue,recipe_pin):
    inputs=annotation.Inputs();recipe=inputs.read(recipe_pin)
    require(recipe['schema']==finite.SCHEMA and recipe['plant_slots']==144
            and recipe['no_geometry_removed'] is True and recipe['whole_plant_dependency_closure'] is True,
            'Explicit finite whole-plant recipe required')
    verify_bindings(recipe['source_bindings']);inputs.bindings.update(recipe['source_bindings'])
    require(recipe['source_bindings'].get(str(Path(finite.__file__).resolve()))==sha256(finite.__file__),'Finite recipe implementation changed')
    reference=inputs.read(recipe['base_logical_census']);base_catalogue=inputs.read(recipe['base_catalogue']);selection=inputs.read(recipe['selection'])
    inputs.bindings.update(population['original_population_bindings'])
    inputs.bindings.update(population.get('source_bindings',{}))
    verify_bindings(inputs.bindings)
    logical=deepcopy(population['scene_policy_evidence']);primary=recipe['primary_root']
    require(logical.get('foreground_root',primary)==primary,'Current foreground occupies another slot')
    finite.background_match(logical,reference,primary)
    current_roots={r['plant_root']:r for r in logical['all_plant_roots']}
    require(len(catalogue)==population['counts']['components'],'Logical catalogue incomplete')
    index=coverage.CatalogueIndex(catalogue);stage=scene['stage']
    require(stage.GetRootLayer().anonymous and stage.GetSessionLayer().anonymous,'Anonymous owned stage required')
    require(not stage.GetPrimAtPath(primary+'/__FarCoarseGroup_0'),'Foreground must never be packed')
    whole=set(recipe['whole_original_background_roots']);entries={r['plant_root']:r for r in recipe['plants']}
    packed={root:e for root,e in entries.items() if e['mode']=='unchanged_packed_background'}
    require(len(whole)==6 and len(entries)==144 and len(packed)==recipe['unchanged_packed_background_count']==137
            and entries[primary]['mode']=='original_foreground'
            and all(entries[r]['mode']=='complete_original_components' for r in whole)
            and set(packed)|whole|{primary}==set(entries) and not (set(packed)&whole), 'Whole/packed/foreground partition differs')
    require(set(current_roots)==set(entries),'Runtime population root set differs')
    groups=[];selected=set()
    for root,entry in sorted(packed.items()):
        artifact=inputs.bound(entry['artifact']);provenance=inputs.read(entry['provenance'])
        require(provenance['plant']['source_family']==current_roots[root]['source_family']
                and provenance['plant']['manifest_sha256']==current_roots[root]['manifest_sha256'], 'Packed artifact family/manifest differs')
        mapping=remap_members(catalogue,base_catalogue,selection['selected_components'],root)
        require(set(mapping)=={str(r['original_component_index']) for r in provenance['faces']},'Packed face set differs from finite selection')
        selected.update(mapping.values())
        expected=Usd.Stage.Open(str(artifact));mesh=UsdGeom.Mesh(expected.GetPrimAtPath('/Plant/__FarCoarseGroup_0'))
        require(mesh,'Expected packed mesh absent')
        subsets={root+str(s.GetPath())[len('/Plant'):]:dict(material_graph_sha256=s.GetPrim().GetName()[2:],faces=fingerprint(list(s.GetIndicesAttr().Get()))) for s in UsdShade.MaterialBindingAPI(mesh).GetMaterialBindSubsets()}
        groups.append(dict(group_id='far_group_'+root.rsplit('/',1)[1],plant_root=root,prim_path=root+'/__FarCoarseGroup_0',
            source_family=current_roots[root]['source_family'],member_component_indices=sorted(mapping.values()),
            original_to_current_component_indices=mapping,face_provenance=entry['provenance'],selection=recipe['selection'],
            plant_to_world_usd_row_vectors=current_roots[root]['plant_to_world_usd_row_vectors'],expected_signature=frozen.mesh_signature(mesh),
            expected_material_subsets=subsets,artifact=entry['artifact']))
    require({int(i) for g in groups for i in g['original_to_current_component_indices']}
            =={r['component_index'] for r in selection['selected_components']},'Finite coarse selection not fully installed')
    # Capture every retained mesh from the ACTUAL current generated population.
    # This includes all foreground geometry and every organ of all whole roots.
    cache=UsdGeom.XformCache();retained=[];original_mesh_count=0
    for prim in Usd.PrimRange(stage.GetPrimAtPath('/World/PackPlants')):
        if not prim.IsA(UsdGeom.Mesh):continue
        original_mesh_count+=1;owner=index.owner(str(prim.GetPath()))
        require(owner is not None,'Original plant mesh lacks logical owner')
        if owner['component_index'] not in selected:retained.append(snapshot_mesh(prim,cache))
    # Preserve complete original/generated collision geometry before ANY overlay.
    collision,session,before,root_before=clone_collision_stage(stage)
    require(sum(p.IsA(UsdGeom.Mesh) for p in Usd.PrimRange(collision.GetPrimAtPath('/World/PackPlants')))==original_mesh_count,
            'Complete original/generated collision mesh count differs')
    old_target=stage.GetEditTarget();stage.SetEditTarget(stage.GetSessionLayer())
    try:
        for group in groups:
            root=group['plant_root'];require(root!=primary and root not in whole,'Attempt to replace protected whole plant')
            require(stage.RemovePrim(root),'Owned anonymous background replacement failed')
            x=UsdGeom.Xform.Define(stage,root);x.AddTransformOp().Set(Gf.Matrix4d(group['plant_to_world_usd_row_vectors']))
            x.GetPrim().GetReferences().AddReference(Path(group['artifact']['path']).as_posix(),'/Plant')
        require(session.ExportToString()==before and stage.GetRootLayer().ExportToString()==root_before,
                'Render replacement modified original collision/root layers')
        overlay=dict(stage=stage,robot=scene['robot'],original_collision_stage=collision,original_collision_session_keepalive=session,
            logical_census=logical,logical_catalogue=catalogue,retained_meshes=retained,render_groups=groups,
            coarse_source_selection=dict(selected={r['component_index']:r for r in selection['selected_components']}),
            template_stages_keepalive=(scene.get('template_stages'),population['template_stages']),
            native_texture_bindings={},primary_root=primary,whole_roots=sorted(whole),recipe_pin=recipe_pin,selection_pin=recipe['selection'],
            source_bindings=inputs.bindings,collision_snapshot=dict(original_plant_mesh_count=original_mesh_count,
                original_session_sha256=hashlib.sha256(before.encode()).hexdigest(),root_layer_sha256=hashlib.sha256(root_before.encode()).hexdigest(),
                distinct_session_layer=True,copy_before_any_render_replacement=True))
        overlay['render_census']=verify_overlay(overlay)
        overlay['source_bindings'].update(overlay['native_texture_bindings'])
        for module in (frozen,annotation,coverage,prototype):
            overlay['source_bindings'][str(Path(module.__file__).resolve())]=sha256(module.__file__)
        overlay['source_bindings'][str(Path(__file__).resolve())]=sha256(__file__)
        verify_bindings(overlay['source_bindings'])
        return overlay
    except BaseException:
        require(stage.GetSessionLayer().ImportFromString(before),'Cannot restore original session after overlay failure')
        raise
    finally:stage.SetEditTarget(old_target)
