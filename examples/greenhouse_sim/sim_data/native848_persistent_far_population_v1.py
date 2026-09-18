"""Finite camera-bank background packing with explicit whole/packed counts.

The installer is the frozen overlay installer with only its recipe identity and
fixed6/137 partition generalized to a proven disjoint full144 partition.
"""
from .native848_farmerge_overlay_v1 import *
from .native848_bulk_io_v1 import save_json

RECIPE_SCHEMA='greenhouse.persistent_finite_farmerge_population.v1'


def install_overlay(scene,population,catalogue,recipe_pin):
    inputs=annotation.Inputs();recipe=inputs.read(recipe_pin)
    require(recipe['schema']==RECIPE_SCHEMA and recipe['plant_slots']==144
            and recipe['no_geometry_removed'] is True and recipe['whole_plant_dependency_closure'] is True,
            'Explicit finite whole-plant recipe required')
    verify_bindings(recipe['source_bindings']);inputs.bindings.update(recipe['source_bindings'])
    require(recipe['source_bindings'].get(str(Path(__file__).resolve()))==sha256(__file__),'Persistent finite recipe implementation changed')
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
    require(0<=len(whole)<=143 and len(entries)==144 and len(packed)==recipe['unchanged_packed_background_count']==143-len(whole)
            and recipe['whole_background_count']==len(whole)
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
