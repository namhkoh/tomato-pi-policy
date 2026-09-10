"""Session-only physics station in the supplied greenhouse, not dataset capture."""
from pathlib import Path

import numpy as np
from pxr import Gf,Usd,UsdGeom,UsdLux,UsdPhysics


def prepare(stage,package,plant):
    from launch_sim_data import load_local_payloads
    from sim_data.audit import audit_manifest
    from sim_data.geometry import assemble_plant
    from sim_data.floor_alignment import PACKAGE_FLOOR,floor_triangles,surface_height
    package=Path(package)
    excluded=load_local_payloads(stage)
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'])
    gutters=stage.GetPrimAtPath('/World/Gutters')
    if not gutters: raise ValueError('Supplied greenhouse gutters are missing')
    stations=sorted((float(cache.ComputeWorldBound(p).ComputeAlignedRange().GetMidpoint()[0]),
                     str(p.GetPath())) for p in gutters.GetChildren())
    cx,gutter=min(stations,key=lambda item:abs(item[0]))
    disabled_scenes=[];fixed_bodies=0;disabled_joints=0;lights=[]
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        for prim in list(stage.Traverse()):
            if prim.HasAPI(UsdLux.LightAPI):
                light=UsdLux.LightAPI(prim)
                exposure=float(light.GetExposureAttr().Get() or 0.)
                light.CreateExposureAttr(exposure-3.)
                lights.append(str(prim.GetPath()))
            if prim.IsA(UsdPhysics.Scene):
                disabled_scenes.append(str(prim.GetPath()));prim.SetActive(False)
            elif prim.HasAPI(UsdPhysics.RigidBodyAPI):
                # The supplied gutters/rails are static infrastructure, not
                # additional gravity-driven articulations.
                UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr(False)
                fixed_bodies+=1
            elif prim.IsA(UsdPhysics.Joint):
                UsdPhysics.Joint(prim).CreateJointEnabledAttr(False);disabled_joints+=1
        manifest=package/f'plants/components/{plant}/manifest.json'
        audit=audit_manifest(manifest)
        paths=assemble_plant(stage,'/World/Plant',audit)
        # Exact foreground station used by launch_sim_data.populate. Neither
        # the gutter height nor the original plant geometry is changed.
        position=np.array([cx+.195,0.,.90])
        UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant')).AddTranslateOp(opSuffix='greenhouseStation').Set(Gf.Vec3d(*position))
        # Distant context on the SAME gutter, with instanced supplied assets.
        # The near interaction area contains one detailed plant initially.
        backdrop=sorted((package/'plants/backdrop').glob('backdrop_*.usd'))
        if not backdrop: raise ValueError('Missing supplied backdrop plants')
        for i,y in enumerate((-6.,-4.,4.,6.)):
            prim=UsdGeom.Xform.Define(stage,f'/World/PhysicsBackdrop/Plant_{i:02d}')
            prim.AddTranslateOp().Set(Gf.Vec3d(cx+.195,y,.90))
            prim.GetPrim().GetReferences().AddReference(str(backdrop[i%len(backdrop)]))
            prim.GetPrim().SetInstanceable(True)
    stage.Load('/World/Plant');stage.Load('/World/PhysicsBackdrop')
    triangles=floor_triangles(stage,PACKAGE_FLOOR)
    def height(x,y): return surface_height(triangles,x,y)
    colliders=sum(p.HasAPI(UsdPhysics.CollisionAPI) and bool(UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get())
                  for p in Usd.PrimRange.Stage(stage,Usd.TraverseInstanceProxies()))
    record=dict(manifest_path=str(manifest),component_paths=paths,plant_root='/World/Plant')
    report=dict(scene=str(package/'house/green_house_base.usd'),selected_gutter=gutter,
        plant_position_world_m=position.tolist(),placement='same_foreground_station_as_package_preview',
        greenhouse_geometry_moved=False,gutters_in_asset=len(stations),
        detailed_plants=1,distant_instanced_plants=4,near_neighbors='not_yet_added',
        static_infrastructure_bodies=fixed_bodies,static_infrastructure_joints_disabled=disabled_joints,
        source_scenes_disabled=disabled_scenes,active_collision_prims=colliders,
        excluded_unbundled_props=len(excluded),floor_path=PACKAGE_FLOOR,
        demo_lighting_exposure_offset_stops=-3.,demo_lights_adjusted=len(lights),
        floor_reference_height_m=height(cx+.8,0.),source_layers_modified=False,training_eligible=False)
    return record,height,report
