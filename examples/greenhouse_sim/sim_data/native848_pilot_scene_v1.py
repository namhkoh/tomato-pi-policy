"""Shared CPU/native pilot scene: authenticated, same-split component plants only.

Uniform filtering occurs in the anonymous session layer before any app update.
No camera/target-specific removal is allowed. Legacy files are never changed.
USD/native imports are delayed until the caller owns its runtime.
"""
from pathlib import Path
import numpy as np
from .dataset_review import require, read_json, verify_bindings
from .depth_preview import sha256
from .capture_contract import fingerprint

SCHEMA='greenhouse.native848_pilot_scene_policy.v1'
POPULATE_SHA256='3960bf65ca49f52b7f07ed37c83fbbfd2608f9957cb7dbd5b521b4033af71494'


def policy(capture_split):
    require(capture_split in ('train','validation','test'),'Actual source split required')
    return dict(schema=SCHEMA,capture_split=capture_split,
                remove_all_merged_background_plants=True,remove_cross_split_component_plants=True,
                removal_depends_on_camera_or_target=False,source_assets_modified=False)


class NoRenderProgress:
    def __init__(self):self.calls=0
    def update(self):self.calls+=1


def populate_filtered(stage,package,anchor,source_plan,reports,expected_policy):
    """The ONLY population/filter function used by both CPU and native paths."""
    from pxr import Usd,UsdGeom
    import launch_sim_data
    from .capture_pilot import source_hashes
    require(sha256(launch_sim_data.__file__)==POPULATE_SHA256,'Frozen population recipe changed')
    split=anchor['split'];require(expected_policy==policy(split),'Uniform pilot policy required')
    assignments=source_plan['family_assignments']
    require(assignments[anchor['source_family']]==split,'Anchor source split differs')
    require(stage.GetRootLayer().anonymous and stage.GetSessionLayer().anonymous,'Only anonymous scene edits allowed')
    stage.SetEditTarget(stage.GetSessionLayer())
    progress=NoRenderProgress();records=[]
    gutter,original_counts=launch_sim_data.populate(stage,package,progress,records,anchor['source_family'])
    require(progress.calls==6 and original_counts==anchor['expected_scene_counts'],'Original population differs before uniform filtering')
    original_bindings=source_hashes(stage)
    variants=anchor['scene_variants'];by_root={v['plant_root']:v for v in variants}
    report_by_family={r['plant_id']:r for r in reports}
    record_by_root={r['plant_root']:r for r in records}
    require(len(by_root)==len(variants)==original_counts['component_plants']
            and set(by_root)==set(record_by_root),'Authenticated component root census differs')
    pack=stage.GetPrimAtPath('/World/PackPlants');require(bool(pack),'Missing population root')
    roots={str(p.GetPath()):p for p in pack.GetChildren()}
    require(len(roots)==original_counts['component_plants']+original_counts['backdrop_instances'],
            'Whole plant population count differs')
    # Also inventory plant-asset reference sites outside the normal PackPlants
    # namespace. Any such extra root is removed uniformly, never ignored.
    extra={}
    plants_dir=(Path(package)/'plants').resolve()
    for prim in stage.Traverse():
        path=str(prim.GetPath())
        if path=='/World/PackPlants' or path.startswith('/World/PackPlants/'):continue
        for spec in prim.GetPrimStack():
            for ref in spec.referenceList.GetAddedOrExplicitItems():
                if not ref.assetPath:continue
                resolved=Path(spec.layer.ComputeAbsolutePath(ref.assetPath)).resolve()
                if resolved.is_relative_to(plants_dir):extra[path]=prim
    extra={p:v for p,v in extra.items() if not any(p.startswith(q+'/') for q in extra if q!=p)}
    roots.update(extra)
    xc=UsdGeom.XformCache();census=[];kept_roots=[]
    for path,prim in sorted(roots.items()):
        matrix=np.asarray(xc.GetLocalToWorldTransform(prim),float).tolist()
        refs=[]
        for spec in prim.GetPrimStack():
            for ref in spec.referenceList.GetAddedOrExplicitItems():
                if ref.assetPath:refs.append(spec.layer.ComputeAbsolutePath(ref.assetPath))
        family=plant_split=None;component_count=0
        if path in by_root:
            variant=by_root[path];record=record_by_root[path];family=variant['source_plant_id']
            require(not variant['added_components'] and not variant['added_component_paths']
                    and variant['source_geometry_modified'] is False,'Unmodified source component plant required')
            require(family==variant['variant_id']==variant['split_group'] and family in assignments,
                    'Exact original family identity required')
            report=report_by_family[family]
            require(Path(record['manifest_path']).resolve()==Path(report['manifest_path']).resolve()
                    and sha256(record['manifest_path'])==report['manifest_sha256'], 'Component source manifest differs')
            require(set(record['component_paths'])==set(report['components']), 'Incomplete component source anatomy')
            plant_split=assignments[family];component_count=len(record['component_paths'])
            keep=plant_split==split
            reason='retained_authenticated_same_split' if keep else 'removed_cross_split_component_plant'
        else:
            keep=False;reason='removed_unauthenticated_merged_or_extra_plant'
        if keep:kept_roots.append(path)
        else:
            require(path not in ('/World','/World/Environment','/World/Gutters','/World/RBY1'),
                    'Refusing to deactivate non-plant scene root')
            prim.SetActive(False)
        census.append(dict(plant_root=path,source_family=family,source_split=plant_split,
                           authenticated_component_plant=path in by_root,component_count=component_count,
                           plant_to_world_usd_row_vectors=matrix,source_asset_paths=sorted(set(refs)),
                           active_after_policy=bool(keep),decision=reason))
    require(anchor['original_variant']['plant_root'] in kept_roots,'Primary authenticated source plant must remain')
    retained_records=[r for r in records if r['plant_root'] in kept_roots]
    retained_variants=[v for v in variants if v['plant_root'] in kept_roots]
    active={str(p.GetPath()) for p in pack.GetChildren()}
    require(active==set(kept_roots) and all(not stage.GetPrimAtPath(p).IsActive() for p in roots if p not in kept_roots),
            'Post-policy active plant population differs')
    counts=dict(original_counts,backdrop_instances=0,component_plants=len(kept_roots),
                components=sum(len(r['component_paths']) for r in retained_records))
    receipt=dict(schema='greenhouse.native848_pilot_full_scene_census.v1',policy=expected_policy,
        policy_implementation=dict(path=str(Path(__file__).resolve()),sha256=sha256(__file__)),
        population_implementation=dict(path=str(Path(launch_sim_data.__file__).resolve()),sha256=POPULATE_SHA256),
        source_collection_plan=dict(path=str(Path(anchor['source_collection_plan']).resolve()),sha256=sha256(anchor['source_collection_plan'])),
        original_counts=original_counts,active_counts=counts,all_plant_roots=census,
        retained_roots=sorted(kept_roots),removed_roots=sorted(set(roots)-set(kept_roots)),
        removed_merged_or_unknown_count=sum(not r['authenticated_component_plant'] for r in census),
        removed_cross_split_count=sum(r['decision']=='removed_cross_split_component_plant' for r in census),
        extra_plant_asset_roots_outside_packplants=sorted(extra),
        complete_active_plant_anatomy=True,full_active_catalogue_required=True,
        progress_updates_during_population=progress.calls,progress_updates_rendered=False,
        filtering_applied_before_first_app_update=True,source_assets_modified=False,
        training_approved=False)
    receipt['deterministic_census_sha256']=fingerprint(receipt)
    return dict(gutter_x=gutter,records=retained_records,variants=retained_variants,counts=counts,
                scene_policy_evidence=receipt,original_population_bindings=original_bindings)


class PilotCPUScene:
    """CPU-only same scene recipe; actual native parity still needs a capture."""
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
        self.scene=populate_filtered(self.stage,package,anchor,plan,reports,expected_policy)
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
    filtered=populate_filtered(stage,package,plan,original_plan,reports,expected_policy)
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
        'robot': robot,
        'old_manifest': old_manifest,
        'old_sample': old_sample,
        'original_bindings': original_bindings,
        'original_variant': original_variant,
        'pose': pose,
        'settings': settings,
        'setup_seconds': setup_seconds,
    }
