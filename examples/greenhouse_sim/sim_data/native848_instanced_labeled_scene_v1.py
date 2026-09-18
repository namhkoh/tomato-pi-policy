"""Fully occupied labeled plants with native instanced background hierarchies.

Only anonymous layers are authored. The foreground remains exact; all other
slots hold full manifest-backed plants. This is an explicitly new population.
"""
from pathlib import Path
from copy import deepcopy
import numpy as np
from .dataset_review import require, read_json, verify_bindings
from .depth_preview import sha256
from .capture_contract import fingerprint
SCHEMA='greenhouse.native848_instanced_labeled_scene_policy.v1'
CENSUS_SCHEMA='greenhouse.native848_instanced_labeled_census.v1'
POPULATE_SHA256='3960bf65ca49f52b7f07ed37c83fbbfd2608f9957cb7dbd5b521b4033af71494'

def metadata(split):
    return dict(dataset_split=split,annotation_coverage='complete_manifest_backed_plant_anatomy_pending_native_census',
        no_query9mm_acceptance_pending=True,accepted_training_increment=0,training_approved=False,
        filtered_scene_annotation_export_eligible=False)

def policy(split):
    require(split in ('train','validation','test'),'Authenticated source split required')
    return dict(schema=SCHEMA,dataset_split=split,total_plant_slots=144,
        complete_manifest_backed_plant_anatomy=True,retain_all_original_slots=True,
        source_geometry_modified=False,background_geometry_replaced=True,
        background_family_rule='round_robin_sorted_same_split_pool_by_original_slot_order',
        plant_removal_allowed=False,removal_depends_on_camera_or_target=False,
        foreground_source_geometry_and_placement_preserved=True,original_merged_scene_parity_claimed=False,
        same_split_source_family_pool_required=True,clones_count_as_independent_families=False,
        native_instanceable_plants=True,foreground_native_instanceable=False,source_assets_modified=False)

class NoRenderProgress:
    def __init__(self):self.calls=0
    def update(self):self.calls+=1

def _template(manifest_path):
    """One anonymous, non-instanceable hierarchy per family; shared source USD refs."""
    from pxr import Usd,UsdGeom,Gf
    stage=Usd.Stage.CreateInMemory(); root=UsdGeom.Xform.Define(stage,'/Plant').GetPrim();stage.SetDefaultPrim(root)
    manifest=read_json(manifest_path);by={c['id']:c for c in manifest['components']};paths={};pending=set()
    def add(cid):
        if cid in paths:return paths[cid]
        require(cid not in pending,'Manifest anatomical parent cycle');pending.add(cid)
        c=by[cid];parent=c['parent'];base=add(parent) if parent else '/Plant';path=base+'/'+cid
        x=UsdGeom.Xform.Define(stage,path);asset=(Path(manifest_path).parent/c['file']).resolve()
        x.GetPrim().GetReferences().AddReference(asset.as_posix())
        offset=np.asarray(c['transform']['translate'])-(np.asarray(by[parent]['transform']['translate']) if parent else 0)
        x.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in offset]));paths[cid]=path;pending.remove(cid);return path
    for cid in by:add(cid)
    return stage,paths

def populate_labeled(stage,package,anchor,source_plan,reports,expected_policy):
    """Identical deterministic population entry point for CPU and native runtime."""
    from pxr import Usd,UsdGeom,Gf
    import launch_sim_data
    from .capture_pilot import source_hashes
    require(sha256(launch_sim_data.__file__)==POPULATE_SHA256,'Frozen original slot recipe changed')
    split=anchor['split'];require(expected_policy==policy(split),'Fully labeled scene policy changed')
    assignments=source_plan['family_assignments'];require(assignments[anchor['source_family']]==split,'Source split differs')
    require(stage.GetRootLayer().anonymous and stage.GetSessionLayer().anonymous,'Anonymous scene required')
    stage.SetEditTarget(stage.GetSessionLayer());progress=NoRenderProgress();old_records=[]
    gutter,original_counts=launch_sim_data.populate(stage,package,progress,old_records,anchor['source_family'])
    require(progress.calls==6 and original_counts==anchor['expected_scene_counts'],'Original slot recipe inventory differs')
    require(original_counts['backdrop_instances']==142 and original_counts['component_plants']==2,'Original144 required')
    original_bindings=source_hashes(stage);report_by={r['plant_id']:r for r in reports}
    pool=sorted(f for f,s in assignments.items() if s==split and f in report_by);require(pool,'Empty same-split plant pool')
    primary=anchor['original_variant']['plant_root'];old_by={r['plant_root']:r for r in old_records}
    old_variants={v['plant_root']:v for v in anchor['scene_variants']}
    roots=sorted(stage.GetPrimAtPath('/World/PackPlants').GetChildren(),key=lambda p:str(p.GetPath()))
    require(len(roots)==144 and primary in {str(p.GetPath()) for p in roots},'Original plant slots missing')
    xc=UsdGeom.XformCache();slots=[]
    for prim in roots:
        path=str(prim.GetPath());refs=[]
        for spec in prim.GetPrimStack():
            for ref in spec.referenceList.GetAddedOrExplicitItems():
                if ref.assetPath:refs.append(spec.layer.ComputeAbsolutePath(ref.assetPath))
        slots.append(dict(plant_root=path,plant_to_world_usd_row_vectors=np.asarray(xc.GetLocalToWorldTransform(prim),float).tolist(),
            original_population_kind='component_plant' if path in old_by else 'merged_backdrop',
            original_source_asset_paths=sorted(set(refs)),original_source_family=old_variants.get(path,{}).get('source_plant_id')))
    templates={};records=[];variants=[];census=[];xc=UsdGeom.XformCache()
    for index,slot in enumerate(slots):
        path=slot['plant_root'];is_primary=path==primary;family=anchor['source_family'] if is_primary else pool[index%len(pool)]
        report=report_by[family];mp=Path(report['manifest_path']).resolve()
        require(sha256(mp)==report['manifest_sha256'] and assignments[family]==split,'Manifest/pool pin differs')
        if is_primary:
            record=deepcopy(old_by[path]);variant=deepcopy(anchor['original_variant'])
        else:
            if family not in templates:templates[family]=_template(mp)
            template,relative=templates[family]
            require(stage.RemovePrim(path),'Original anonymous slot replacement failed')
            prim=UsdGeom.Xform.Define(stage,path).GetPrim()
            UsdGeom.Xformable(prim).AddTransformOp().Set(Gf.Matrix4d(*sum(slot['plant_to_world_usd_row_vectors'],[])))
            prim.GetReferences().AddReference(template.GetRootLayer().identifier,'/Plant')
            prim.SetInstanceable(True)
            paths={cid:path+relative_path[len('/Plant'):] for cid,relative_path in relative.items()}
            record=dict(plant_root=path,manifest_path=str(mp),component_paths=paths)
            variant=dict(variant_id='fullpop_'+path.rsplit('/',1)[1],plant_root=path,source_plant_id=family,
                split_group=family,source_geometry_modified=False,added_components={},added_component_paths={})
        require(set(record['component_paths'])==set(report['components']),'Full source anatomy required in every slot')
        records.append(record);variants.append(variant)
        matrix=np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(stage.GetPrimAtPath(path)),float).tolist()
        require(np.allclose(matrix,slot['plant_to_world_usd_row_vectors'],atol=1e-12,rtol=0),'Original slot transform moved')
        census.append(dict(plant_root=path,variant_id=variant['variant_id'],source_family=family,source_plant_id=family,
            source_split=split,authenticated_component_plant=True,native_instanceable=not is_primary,component_count=len(record['component_paths']),
            manifest_path=str(mp),manifest_sha256=report['manifest_sha256'],plant_to_world_usd_row_vectors=matrix,
            original_slot_mapping=slot,active_after_policy=True,decision='retained_exact_foreground' if is_primary else 'full_manifest_plant_at_original_slot',
            source_geometry_modified=False,source_asset_paths=sorted(str((mp.parent/c['file']).resolve()) for c in report['components'].values())))
    active=list(stage.GetPrimAtPath('/World/PackPlants').GetChildren())
    require(len(active)==144 and all(p.IsActive() and ((not p.IsInstanceable()) if str(p.GetPath())==primary else (p.IsInstance() and p.IsInstanceable() and bool(p.GetPrototype()))) for p in active),'Full144 population with143 native instances and ordinary foreground required')
    bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
    for entry in census:
        box=bbox.ComputeWorldBound(stage.GetPrimAtPath(entry['plant_root'])).ComputeAlignedRange()
        require(not box.IsEmpty(),'Empty full plant');entry['world_bounds_m']=dict(min=list(box.GetMin()),max=list(box.GetMax()))
    counts=dict(original_counts,backdrop_instances=0,component_plants=144,components=sum(len(r['component_paths']) for r in records))
    bindings={**original_bindings,**source_hashes(stage)}
    from . import native848_instanced_static_guard_v1 as proxy_guard
    bindings[str(Path(proxy_guard.__file__).resolve())]=sha256(proxy_guard.__file__)
    for family in set(v['source_plant_id'] for v in variants):
        report=report_by[family];bindings[str(Path(report['manifest_path']).resolve())]=report['manifest_sha256']
        for c in report['components'].values():bindings[str((Path(report['manifest_path']).parent/c['file']).resolve())]=c['asset_sha256']
    receipt=dict(schema=CENSUS_SCHEMA,policy=expected_policy,**metadata(split),
        policy_implementation=dict(path=str(Path(__file__).resolve()),sha256=sha256(__file__)),
        population_implementation=dict(path=str(Path(launch_sim_data.__file__).resolve()),sha256=POPULATE_SHA256),
        source_collection_plan=dict(path=str(Path(anchor['source_collection_plan']).resolve()),sha256=sha256(anchor['source_collection_plan'])),
        original_counts=original_counts,active_counts=counts,all_plant_roots=census,retained_roots=sorted(v['plant_root'] for v in variants),
        removed_roots=[],removed_merged_or_unknown_count=0,removed_cross_split_count=0,
        replaced_merged_background_count=142,replaced_nonprimary_component_slot_count=1,
        extra_plant_asset_roots_outside_packplants=[],complete_active_plant_anatomy=True,full_active_catalogue_required=True,
        independent_source_families=sorted(set(v['source_plant_id'] for v in variants)),source_family_pool=pool,
        progress_updates_during_population=progress.calls,progress_updates_rendered=False,
        population_applied_before_first_app_update=True,source_assets_modified=False)
    receipt['deterministic_census_sha256']=fingerprint(receipt)
    # Proxy traversal retains static-body, animation and material checks inside every instance.
    from .native848_instanced_static_guard_v1 import assert_static_plant_sources
    assert_static_plant_sources(stage)
    return dict(gutter_x=gutter,records=records,variants=variants,counts=counts,template_stages=templates,
        scene_policy_evidence=receipt,original_population_bindings=bindings)

class PilotCPUScene:
    """CPU-only full population; actual native census match remains required."""
    def __init__(self,anchor,expected_policy):
        from pxr import Usd,UsdGeom
        from .collection_plan import load_plan
        from .capture_scene import capture_root,freeze_rigid_bodies
        from .capture_pilot import source_hashes
        from .robot_preview import add_robot_preview
        from .floor_alignment import PACKAGE_FLOOR
        from .static_geometry_parts_cache_v1 import StaticGeometryPartsCache
        from launch_sim_data import load_local_payloads
        plan,reports=load_plan(anchor['source_collection_plan']);package=Path(plan['package'])
        manifest=read_json(Path(anchor['source_capture'])/'manifest.json')
        wrapper=capture_root(package/'house/green_house_base.usd')
        self.stage=Usd.Stage.CreateInMemory();self.stage.SetLoadRules(Usd.StageLoadRules.LoadNone())
        self.stage.GetRootLayer().subLayerPaths=list(wrapper.subLayerPaths)
        UsdGeom.SetStageMetersPerUnit(self.stage,1);UsdGeom.SetStageUpAxis(self.stage,'Z')
        self.stage.SetEditTarget(self.stage.GetSessionLayer())
        excluded=load_local_payloads(self.stage)
        require(excluded==manifest['unbundled_external_prop_roots_excluded'],'CPU external-prop scope differs')
        self.scene=populate_labeled(self.stage,package,anchor,plan,reports,expected_policy)
        self.reports=reports
        self.robot=add_robot_preview(self.stage,gutter_x=self.scene['gutter_x'],floor_path=PACKAGE_FLOOR,right_tool='gripper')
        freeze_rigid_bodies(self.stage)
        self.source_bindings={**self.scene['original_population_bindings'],**source_hashes(self.stage),
                              str(Path(__file__).resolve()):sha256(__file__)}
        self.screen=StaticGeometryPartsCache(self.stage,self.robot['root'],include_generated_plants=True)
    def check(self,record):
        from .native848_original_direct_worker_v2 import apply_cached_pose
        pose,calibration=apply_cached_pose(self.stage,self.robot,record)
        return dict(screen=self.screen(),actual_robot_snapshot=pose,actual_calibration=calibration)
    def finish(self):
        verify_bindings(self.source_bindings)
        require(not any(not layer.anonymous and layer.dirty for layer in self.stage.GetUsedLayers()),'CPU source layer dirtied')
        self.screen.close();self.stage=None


def prepare_native_scene(app, plan, expected_policy):
    import time
    from .native848_pilot_plan_v1 import load_source
    from .native_greenhouse_pair import assert_same_camera
    import carb
    import omni.usd
    import omni.timeline
    import omni.replicator.core as rep
    from pxr import Usd,UsdGeom
    from launch_sim_data import load_local_payloads
    from greenhouse_sim import robot_hardware,robot_kinematics
    from .capture_scene import capture_root,calibration,freeze_rigid_bodies,target_world_geometry
    from .collection_plan import load_plan
    from .floor_alignment import PACKAGE_FLOOR
    from .robot_preview import add_robot_preview

    started = time.perf_counter()
    old_manifest, old_sample, original_bindings = load_source(plan["source_capture"], plan["source_sample"])
    original_plan, reports = load_plan(plan["source_collection_plan"])
    original_rows = [r for j in original_plan["jobs"] for r in j["targets"]
                     if r["target_id"] == plan["source_row"]["target_id"]]
    require(original_rows == [plan["source_row"]], "Source target changed")
    package = Path(original_plan["package"])
    wrapper = capture_root(package/"house/green_house_base.usd")
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    stage.SetLoadRules(Usd.StageLoadRules.LoadNone())
    stage.GetRootLayer().subLayerPaths = list(wrapper.subLayerPaths)
    require(stage.GetRootLayer().anonymous and stage.GetPrimAtPath("/World/Gutters"), "Original greenhouse required")
    UsdGeom.SetStageMetersPerUnit(stage, 1)
    UsdGeom.SetStageUpAxis(stage, "Z")
    stage.SetEditTarget(stage.GetSessionLayer())
    excluded = load_local_payloads(stage)
    records = []
    filtered=populate_labeled(stage,package,plan,original_plan,reports,expected_policy)
    gutter_x,counts,records=filtered["gutter_x"],filtered["counts"],filtered["records"]
    require(filtered["scene_policy_evidence"]["original_counts"] == plan["expected_scene_counts"] == old_manifest["scene_counts"], "Original population changed before policy")
    require(excluded == old_manifest["unbundled_external_prop_roots_excluded"], "External prop scope changed")
    require(old_manifest["variants"] == plan["scene_variants"], "Source variant inventory changed")
    variants = filtered["variants"]
    robot = add_robot_preview(stage, gutter_x=gutter_x, floor_path=PACKAGE_FLOOR, right_tool="gripper")
    freeze_rigid_bodies(stage)

    import sys
    sys.path.insert(0, str(package/"env_panel"))
    from tomato_env import daylight
    require(Path(daylight.__file__).resolve() == (package/"env_panel/tomato_env/daylight.py").resolve(),
            "Unexpected dynamically imported daylight source")
    require(original_plan["configuration"].get("clear_capture") == "robot_head_close_diffuse_v1",
            "Qualified diffuse-lighting source required")
    lighting = daylight.apply(stage, day=172, minutes=13*60, intensity=1500, dome_intensity=6000)
    require(lighting == old_manifest["lighting"], "Original lighting changed")
    settings = carb.settings.get_settings()
    require(old_manifest["renderer"] == "RealTimePathTracing", "Original renderer profile required")
    settings.set("/rtx/rendermode", old_manifest["renderer"])
    app.update()
    rep.set_global_seed(original_plan["configuration"]["seed"])

    pose = plan["expected_robot_snapshot"]
    require(pose == old_sample["robot_snapshot"], "Robot reference changed")
    root = np.asarray(pose["robot_root_to_world_usd_row_vectors"]).T
    links = robot_kinematics.Rby1Kinematics().all_link_transforms(pose["joint_degrees"])
    with Usd.EditContext(stage, stage.GetSessionLayer()):
        robot_hardware._set_transform(stage.GetPrimAtPath(robot["root"]), root[:3, :3], root[:3, 3])
        for link, matrix in links.items():
            prim = stage.GetPrimAtPath(robot["root"]+"/"+link)
            require(bool(prim), "Missing robot link")
            robot_hardware._set_transform(prim, matrix[:3, :3], matrix[:3, 3])
    assert_same_camera(calibration(stage), old_sample["calibration"])
    original_variant = next(v for v in variants if v["variant_id"] == plan["source_row"]["variant_id"])
    require(original_variant == plan["original_variant"], "Foreground source placement changed")
    old_world = target_world_geometry(stage, original_variant, plan["source_row"])
    for key, value in old_world.items():
        require(np.allclose(value, old_sample["supervision"][key], atol=1e-9, rtol=0),
                "Source world geometry moved: "+key)
    context.get_selection().clear_selected_prim_paths()
    omni.timeline.get_timeline_interface().pause()
    setup_seconds = time.perf_counter()-started

    return {
        'started': started,
        'stage': stage,
        'records': records,
        'reports': reports,
        'variants': variants,
        'counts': counts,
        'scene_policy_evidence': filtered['scene_policy_evidence'],
        'original_population_bindings': filtered['original_population_bindings'],
        'template_stages': filtered['template_stages'],
        'robot': robot,
        'old_manifest': old_manifest,
        'old_sample': old_sample,
        'original_bindings': original_bindings,
        'original_variant': original_variant,
        'pose': pose,
        'settings': settings,
        'setup_seconds': setup_seconds,
    }
