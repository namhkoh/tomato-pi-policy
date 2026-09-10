"""Conservative fixed-workspace wire culling. Never cull by visibility."""
import numpy as np


def intersects_xy(low,high,centre,half_extent):
    low,high,centre=map(lambda v:np.asarray(v,dtype=float),(low,high,centre))
    if low.shape!=(3,) or high.shape!=(3,) or centre.shape!=(2,) or not np.isfinite([*low,*high,*centre,half_extent]).all() or np.any(high<low) or half_extent<=0:
        raise ValueError('Invalid collision bounds')
    return bool(np.all(high[:2]>=centre-half_extent) and np.all(low[:2]<=centre+half_extent))


def inside_window(positions,radii,centre,half_extent,margin=.15):
    positions=np.asarray(positions,dtype=float);radii=np.asarray(radii,dtype=float)
    centre=np.asarray(centre,dtype=float)
    if (positions.ndim!=2 or positions.shape[1]!=3 or radii.shape!=(len(positions),)
            or centre.shape!=(2,) or not np.isfinite(positions).all()
            or not np.isfinite([*radii,*centre,half_extent,margin]).all()
            or np.any(radii<0) or not 0<=margin<half_extent):
        raise ValueError('Invalid guarded body bounds')
    return bool(np.all(np.abs(positions[:,:2]-centre)+radii[:,None]+margin<=half_extent))


def body_radii(stage,body_paths,collider_paths):
    from pxr import Usd,UsdGeom
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'],False,True)
    radii=[]
    for body in body_paths:
        origin=np.array(UsdGeom.Xformable(stage.GetPrimAtPath(body)).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation())
        radius=0.
        for path in collider_paths:
            if path!=body and not path.startswith(body+'/'): continue
            bounds=cache.ComputeWorldBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
            if bounds.IsEmpty(): raise ValueError('Cannot bound collision shape '+path)
            radius=max(radius,max(np.linalg.norm(np.array(bounds.GetCorner(i))-origin) for i in range(8)))
        radii.append(radius)
    return np.array(radii)


def configure(stage,robot,half_extent=2.):
    from pxr import Usd,UsdGeom,UsdPhysics
    root=stage.GetPrimAtPath('/World/GutterWires')
    if not root: raise ValueError('Expected supplied greenhouse wires')
    centre=robot.base[:2,3].copy()
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render','proxy','guide'],False,True)
    retained=[];disabled=[];inactive_guides=[]
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI): continue
            api=UsdPhysics.CollisionAPI(prim)
            if not api.GetCollisionEnabledAttr().Get(): continue
            bounds=cache.ComputeWorldBound(prim).ComputeAlignedRange()
            if bounds.IsEmpty(): raise ValueError('Cannot bound wire '+str(prim.GetPath()))
            if intersects_xy(bounds.GetMin(),bounds.GetMax(),centre,half_extent):
                retained.append(str(prim.GetPath()))
            else:
                disabled.append(str(prim.GetPath()))
                if UsdGeom.Imageable(prim).ComputePurpose()=='guide':
                    # These are non-rendered physics proxy Cubes, not the
                    # visible greenhouse wires. Disabled APIs alone leave
                    # thousands of prims in the per-step active scene.
                    inactive_guides.append(str(prim.GetPath()));prim.SetActive(False)
                else:
                    api.CreateCollisionEnabledAttr(False)
    robot.window=dict(centre=centre,half_extent=half_extent)
    robot.window_robot_radii=body_radii(stage,robot.body_paths,robot.collider_paths)
    rig=robot.rig
    plant_colliders=[str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(rig.root))
        if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    robot.window_plant_radii=body_radii(stage,rig.body_paths,plant_colliders)
    return dict(centre_xy_m=centre.tolist(),half_extent_m=half_extent,boundary_margin_m=.15,
        retained_wire_colliders=len(retained),disabled_unreachable_wire_colliders=len(disabled),
        deactivated_unreachable_guide_proxies=len(inactive_guides),
        retained_paths=retained,disabled_paths=disabled,all_renderable_wire_visuals_retained=True,
        floor_gutters_building_and_target_collisions_unchanged=True,
        guard='all_robot_and_dynamic_plant_collision_spheres_checked_each_step',
        scope='fixed_base_only_rebuild_before_any_base_relocation')
