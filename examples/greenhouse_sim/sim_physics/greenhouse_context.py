"""Original preview planting layout with static native contacts near the robot.

Backdrop meshes have no source physics. Nearby ones get triangle-mesh static
colliders in anonymous layers; distant ones stay render instances. Only the
selected detailed petiole is compliant. No source USD or dataset is edited.
"""
from pathlib import Path
from .file_integrity import sha256_file
import numpy as np
from pxr import Gf,Usd,UsdGeom,UsdPhysics
from .collision_window import intersects_xy


def collision_prototype(asset):
    stage=Usd.Stage.CreateInMemory()
    root=UsdGeom.Xform.Define(stage,'/Plant').GetPrim()
    root.GetReferences().AddReference(str(asset))
    stage.SetDefaultPrim(root)
    count=0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
            count+=1
    if count==0: raise ValueError('Backdrop has no usable mesh')
    # Keep the owning stage alive: this binding exposes weak layer handles.
    return stage,count


def populate(stage,package,robot,rows=3,*,detailed_neighbor_manifest=None,target_row_slot=12,target_planting_side=1):
    from .planting_slots import validate,validate_side,backdrop_source_address
    validate(target_row_slot);validate_side(target_planting_side)
    if rows not in (1,3,5): raise ValueError('Select 1, 3 or 5 context gutters')
    if robot.window is None: raise ValueError('Context requires a guarded fixed collision window')
    if stage.GetPrimAtPath('/World/PhysicsBackdrop'):
        raise ValueError('Context already exists; start a new physics stage')
    package=Path(package)
    cache=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'],False,True)
    stations=sorted((float(cache.ComputeWorldBound(p).ComputeAlignedRange().GetMidpoint()[0]),str(p.GetPath()))
        for p in stage.GetPrimAtPath('/World/Gutters').GetChildren())
    center=min(range(len(stations)),key=lambda i:abs(stations[i][0]))
    selected=stations[max(0,center-rows//2):center+rows//2+1]
    assets=sorted((package/'plants/backdrop').glob('backdrop_*.usd'))
    if not assets: raise ValueError('Missing source backdrop plants')
    bounds={};layers={};near=far=0;bindings={};instances=[];detailed=0
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        UsdGeom.Xform.Define(stage,'/World/PhysicsBackdrop')
        for gi,(cx,_) in enumerate(selected):
            for side in (-1,1):
                for i in range(24):
                    address=backdrop_source_address(side,i,target_row_slot,
                        target_planting_side=target_planting_side,
                        selected_gutter=bool(cx==stations[center][0]))
                    if address is None:continue
                    source_side,source_slot=address
                    asset=assets[(source_slot+gi*7+(5 if source_side>0 else 0))%len(assets)]
                    if asset not in bounds:
                        source=Usd.Stage.Open(str(asset))
                        bound=cache.ComputeWorldBound(source.GetDefaultPrim()).ComputeAlignedRange()
                        if bound.IsEmpty(): raise ValueError('Unbounded context plant')
                        bounds[asset]=(np.array(bound.GetMin()),np.array(bound.GetMax()))
                        bindings[str(asset)]=sha256_file(asset)
                    pos=np.array([cx+side*.195,(i-12)*.5,.90])
                    low,high=bounds[asset]
                    contact=intersects_xy(low+pos,high+pos,**robot.window)
                    path=f'/World/PhysicsBackdrop/Gutter{gi}_Side{side+1}_{i:03d}'
                    prim=UsdGeom.Xform.Define(stage,path)
                    prim.AddTranslateOp().Set(Gf.Vec3d(*pos))
                    if detailed_neighbor_manifest is not None and cx==stations[center][0] and side==1 and i==13:
                        from sim_data.audit import audit_manifest
                        from sim_data.geometry import assemble_plant
                        manifest=Path(detailed_neighbor_manifest)
                        audit=audit_manifest(manifest)
                        paths=assemble_plant(stage,path,audit)
                        # Retain the preview's second complete component plant,
                        # including its source static colliders. No backdrop
                        # substitution, shape simplification or new dynamics.
                        bindings[str(manifest)]=sha256_file(manifest)
                        for c in audit['components'].values():
                            bindings[str(manifest.parent/c['file'])]=c['asset_sha256']
                        instances.append(dict(path=path,source=str(manifest),position_m=pos.tolist(),
                            static_contact=True,detailed_components=len(paths)))
                        detailed+=1;continue
                    if contact:
                        if asset not in layers: layers[asset]=collision_prototype(asset)
                        prim.GetPrim().GetReferences().AddReference(layers[asset][0].GetRootLayer().identifier)
                        near+=1
                    else:
                        prim.GetPrim().GetReferences().AddReference(str(asset));far+=1
                    prim.GetPrim().SetInstanceable(True)
                    instances.append(dict(path=path,source=str(asset),position_m=pos.tolist(),
                        static_contact=contact,original_backdrop_address=list(address)))
    # Keep anonymous referenced layers alive for the whole diagnostic.
    robot.context_layers=[v[0] for v in layers.values()]
    return dict(populated_gutters=len(selected),context_plants=near+far+detailed,
        target_row_slot=target_row_slot,target_planting_side=target_planting_side,
        planting_asset_assignment='original' if target_row_slot==12 and target_planting_side==1 else 'detailed_target_and_existing_backdrop_swapped',
        detailed_target_plants=1,detailed_neighbor_plants=detailed,static_contact_plants=near+detailed,render_only_distant_plants=far,
        source_layout='same_0.5m_spacing_and_two_sides_as_launch_sim_data.populate',
        compliance='selected_petiole_only_context_is_static',
        source_sha256=bindings,instances=instances,training_eligible=False)
