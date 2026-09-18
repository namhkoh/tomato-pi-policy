"""Diagnostic merged render population plus independent original collision stage."""
from pathlib import Path
from copy import deepcopy
import numpy as np
from .dataset_review import require,read_json,verify_bindings
from .depth_preview import sha256
from .capture_contract import fingerprint

CENSUS_SCHEMA='greenhouse.native848_farmerge_native_render_census.v1'


def mesh_signature(mesh):
    from .native848_far_component_merge_prototype_v1 import array_pin
    from pxr import UsdGeom
    return dict(points=array_pin(np.asarray(mesh.GetPointsAttr().Get())),face_counts=array_pin(np.asarray(mesh.GetFaceVertexCountsAttr().Get())),
        face_indices=array_pin(np.asarray(mesh.GetFaceVertexIndicesAttr().Get())),normals=array_pin(np.asarray(mesh.GetNormalsAttr().Get())),
        uv=array_pin(np.asarray(UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').Get())),subdivision=mesh.GetSubdivisionSchemeAttr().Get(),
        orientation=mesh.GetOrientationAttr().Get(),double_sided=mesh.GetDoubleSidedAttr().Get(),normals_interpolation=mesh.GetNormalsInterpolation())


def verify_native_scene(scene):
    from pxr import Usd,UsdGeom,UsdShade
    from .native848_far_component_merge_prototype_v1 import graph
    stage=scene['stage'];xc=UsdGeom.XformCache();meshes=[p for p in Usd.PrimRange(stage.GetPrimAtPath('/World/PackPlants')) if p.IsA(UsdGeom.Mesh)]
    require(len(stage.GetPrimAtPath('/World/PackPlants').GetChildren())==144 and len(meshes)==11177,'Actual merged native render population differs')
    for old in scene['retained_meshes']:
        prim=stage.GetPrimAtPath(old['path']);require(prim and prim.IsActive() and prim.IsA(UsdGeom.Mesh),'Retained original native mesh missing')
        require(np.array_equal(np.asarray(xc.GetLocalToWorldTransform(prim),float),old['world_matrix'])
            and [str(p) for p in prim.GetRelationship('material:binding').GetTargets()]==old['material_targets']
            and UsdGeom.Imageable(prim).ComputeVisibility()==old['visibility'] and UsdGeom.Imageable(prim).ComputePurpose()==old['purpose'],
            'Actual protected/foreground native matrix/material/visibility changed')
    group_rows=[]
    for group in scene['render_groups']:
        mesh=UsdGeom.Mesh(stage.GetPrimAtPath(group['prim_path']));require(mesh and mesh_signature(mesh)==group['expected_signature'],'Actual native packed geometry/normal/UV differs from CPU artifact')
        subsets=UsdShade.MaterialBindingAPI(mesh).GetMaterialBindSubsets();actual_materials={}
        for subset in subsets:
            material,_=UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial();require(material,'Native material binding absent')
            key=subset.GetPrim().GetName()[2:];signature=fingerprint(graph(material,scene['native_texture_bindings']))
            require(signature==key,'Actual native material graph or resolved texture changed')
            actual_materials[str(subset.GetPath())]=dict(material_graph_sha256=key,faces=fingerprint(list(subset.GetIndicesAttr().Get())))
        require(actual_materials==group['expected_material_subsets'],'Native material face partition differs')
        plant_matrix=np.asarray(xc.GetLocalToWorldTransform(mesh.GetPrim()),float)
        require(np.array_equal(plant_matrix,group['plant_to_world_usd_row_vectors']),'Packed native world transform changed')
        group_rows.append({k:v for k,v in group.items() if k not in ('expected_signature','expected_material_subsets')})
    require(len(group_rows)==143 and len(scene['retained_meshes'])==11034,'Exact native coarse/fine inventory required')
    value=dict(schema=CENSUS_SCHEMA,dataset_split='train',logical_original_census_sha256=scene['logical_census']['deterministic_census_sha256'],
        logical_anatomy_components=61386,logical_plant_slots=144,actual_native_render_meshes=11177,actual_native_retained_original_meshes=11034,
        actual_native_coarse_groups=143,actual_native_packed_geometry_normals_UVs_materials_match_CPU_artifacts=True,
        actual_native_retained_original_world_matrices_exact=True,packed_world_coordinate_roundoff_bound_m=1e-6,
        native_coarse_group_catalogue=group_rows,logical_anatomy_is_not_claim_of_individual_far_pixel_IDs=True,
        collision_original_representation_retained=True,ordinary_annotation_compatible=False,training_approved=False,accepted_training_increment=0)
    value['deterministic_census_sha256']=fingerprint(value);return value


def prepare_native_scene(app,anchor,checked):
    from .native848_fully_labeled_scene_v1 import prepare_native_scene as original_prepare
    return install_render_population(original_prepare(app,anchor,checked['original_scene_policy']),checked)


def install_render_population(scene,checked):
    from pxr import Usd,Sdf,UsdGeom,Gf,UsdShade
    from .native848_fully_labeled_worker_v3 import require_native_census
    from .native848_fully_labeled_coverage_v1 import component_catalogue
    from .native848_far_component_merge_prototype_v1 import array_pin
    from . import native848_farmerge_pair_plan_v1 as api
    stage=scene['stage'];prototype=checked['prototype']
    require_native_census(scene['scene_policy_evidence'],checked['original_CPU']);logical=deepcopy(scene['scene_policy_evidence'])
    logical_catalogue=component_catalogue(stage,scene['records'],scene['reports'],scene['variants'])
    require(len(logical_catalogue)==61386,'Original logical catalogue incomplete')
    # Clone the actual original stage before any render replacement. Session
    # layers are independent; immutable root/source layers and template keepalive
    # objects are retained. Original collision triangles are never packed.
    session=Sdf.Layer.CreateAnonymous('original_collision_session.usda');session.TransferContent(stage.GetSessionLayer())
    collision=Usd.Stage.Open(stage.GetRootLayer(),session,load=Usd.Stage.LoadNone);collision.SetLoadRules(stage.GetLoadRules());collision.SetEditTarget(session)
    require(collision.GetSessionLayer()!=stage.GetSessionLayer(),'Collision stage must have independent authoring')
    selection=read_json(prototype['selection']['path']);selected={r['component_index'] for r in selection['selected_components']}
    by_path={r['prim_path']:r for r in logical_catalogue};by_root={r['plant_root']:r for r in logical['all_plant_roots']}
    old_meshes=read_json(Path(api.PROTOTYPE['path']).parent.parent/'native848_instanced_full144_CPU_parity_20260917_v3/ordinary_meshes.json')
    retained=[]
    for row in old_meshes:
        path=row['path']
        while path not in by_path and path!='/':path=path.rsplit('/',1)[0] or '/'
        require(path in by_path,'Unknown original mesh identity')
        if by_path[path]['component_index'] not in selected:retained.append(row)
    groups=[];stage.SetEditTarget(stage.GetSessionLayer())
    for entry in prototype['plants']:
        root=entry['plant_root'];plant=by_root[root]
        require(sha256(entry['artifact']['path'])==entry['artifact']['sha256'] and sha256(entry['provenance']['path'])==entry['provenance']['sha256'],'Packed plant artifact changed')
        if entry['coarse_groups']==0:
            require(plant['decision']=='retained_exact_foreground','Only foreground may remain wholly original');continue
        expected_stage=Usd.Stage.Open(entry['artifact']['path']);expected=UsdGeom.Mesh(expected_stage.GetPrimAtPath('/Plant/__FarCoarseGroup_0'))
        expected_subsets={root+str(s.GetPath())[len('/Plant'):]:dict(material_graph_sha256=s.GetPrim().GetName()[2:],faces=fingerprint(list(s.GetIndicesAttr().Get())))
            for s in UsdShade.MaterialBindingAPI(expected).GetMaterialBindSubsets()}
        require(stage.RemovePrim(root),'Original anonymous render plant replacement failed');x=UsdGeom.Xform.Define(stage,root)
        x.AddTransformOp().Set(Gf.Matrix4d(plant['plant_to_world_usd_row_vectors']));x.GetPrim().GetReferences().AddReference(entry['artifact']['path'],'/Plant')
        members=[r['component_index'] for r in selection['selected_components'] if r['prim_path'].startswith(root+'/')]
        groups.append(dict(group_id='far_group_'+root.rsplit('/',1)[1],plant_root=root,prim_path=root+'/__FarCoarseGroup_0',
            source_family=plant['source_family'],member_component_indices=members,face_provenance=entry['provenance'],
            all_members_outside_both_arm_bounds_for_every_original36_pose=True,selection=prototype['selection'],
            plant_to_world_usd_row_vectors=plant['plant_to_world_usd_row_vectors'],expected_signature=mesh_signature(expected),expected_material_subsets=expected_subsets))
    scene.update(logical_census=logical,logical_catalogue=logical_catalogue,logical_counts=deepcopy(scene['counts']),
        original_collision_stage=collision,original_collision_session_keepalive=session,retained_meshes=retained,render_groups=groups,
        native_texture_bindings={},counts=dict(logical_plant_slots=144,logical_anatomy_components=61386,native_render_meshes=11177,coarse_far_groups=143,retained_original_meshes=11034))
    scene['scene_policy_evidence']=verify_native_scene(scene)
    scene['original_population_bindings'].update(checked['prototype_closure']['source_bindings'])
    return scene


class OriginalCollisionBridge:
    def __init__(self,scene):
        from .static_geometry_parts_cache_v1 import StaticGeometryPartsCache
        self.scene=scene;self.cache=StaticGeometryPartsCache(scene['original_collision_stage'],scene['robot']['root'],include_generated_plants=True)
    def check(self,record,native_pose,native_calibration):
        from .native848_original_direct_worker_v2 import apply_cached_pose
        from .native_greenhouse_pair import assert_same_camera
        pose,cal=apply_cached_pose(self.scene['original_collision_stage'],self.scene['robot'],record)
        require(pose['joint_degrees']==native_pose['joint_degrees'] and np.array_equal(pose['robot_root_to_world_usd_row_vectors'],native_pose['robot_root_to_world_usd_row_vectors']),
            'Collision shadow does not reproduce exact actual native robot pose')
        assert_same_camera(cal,native_calibration);screen=self.cache()
        return dict(screen,collision_representation='unaltered_original_component_meshes_in_independent_USD_stage',
            exact_actual_native_robot_pose_and_camera_verified=True,merged_render_bounds_used_as_collision_proxy=False)
    def diagnostics(self):return dict(self.cache.diagnostics(),independent_original_collision_stage=True)
    def close(self):self.cache.close()


def classify_mapping(scene,mapping):
    """Every observed plant ID is exact protected component or explicit coarse group."""
    from pxr import UsdGeom
    by={r['prim_path']:r for r in scene['logical_catalogue']};coarse={r['prim_path']:r for r in scene['render_groups']}
    selected={i for r in scene['render_groups'] for i in r['member_component_indices']};result={}
    for renderer_id,path in mapping.items():
        if not path.startswith('/World/PackPlants/'):continue
        prim=scene['stage'].GetPrimAtPath(path);require(prim and prim.IsActive(),'Observed plant ID has no active native prim')
        group=next((g for p,g in coarse.items() if path==p or path in g['expected_material_subsets']),None)
        if group:
            result[str(renderer_id)]=dict(kind='coarse_far_group',group_id=group['group_id'],prim_path=group['prim_path'],
                member_component_indices=group['member_component_indices'],face_provenance=group['face_provenance'],individual_component_pixel_ID_claimed=False);continue
        ancestor=path
        while ancestor not in by and ancestor!='/':ancestor=ancestor.rsplit('/',1)[0] or '/'
        require(ancestor in by and by[ancestor]['component_index'] not in selected,'Unknown or stale selected-component renderer ID')
        row=by[ancestor];result[str(renderer_id)]=dict(kind='exact_retained_component',component_index=row['component_index'],
            variant_id=row['variant_id'],component_id=row['component_id'],source_family=row['source_plant_id'],prim_path=row['prim_path'])
    return result
