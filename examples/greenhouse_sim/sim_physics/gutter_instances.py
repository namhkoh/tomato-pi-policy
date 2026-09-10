"""Batch repeated static gutter visuals; retain every original collision proxy."""
import numpy as np
from pxr import Gf,Usd,UsdGeom,UsdPhysics


def batch(stage):
    root=stage.GetPrimAtPath('/World/Gutters')
    if not root or stage.GetPrimAtPath('/World/GutterVisualInstances'):
        raise ValueError('Expected unbatched supplied greenhouse')
    modules=[p for g in root.GetChildren() for p in g.GetChildren()
             if p.GetName().startswith('Gutter_Module_')]
    if not modules: raise ValueError('No gutter modules to batch')
    # Fail closed if the supplied asset gains per-instance physics/variants.
    prototype=modules[0].GetPrototype()
    if not prototype or any(not p.IsInstance() or p.GetPrototype()!=prototype for p in modules):
        raise ValueError('Gutter visuals must share one USD prototype')
    for p in Usd.PrimRange(prototype):
        if (p.HasAPI(UsdPhysics.CollisionAPI) or p.HasAPI(UsdPhysics.RigidBodyAPI)
                or p.IsA(UsdPhysics.Joint)):
            raise ValueError('Cannot batch visual modules containing physics')
    positions=[];orientations=[];scales=[]
    for prim in modules:
        xf=UsdGeom.Xformable(prim)
        if xf.TransformMightBeTimeVarying(): raise ValueError('Cannot batch animated infrastructure')
        matrix=xf.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        tr=Gf.Transform(matrix);scale=tr.GetScale();q=tr.GetRotation().GetQuat()
        # SRT only, no shear or nontrivial scale orientation.
        rebuilt=Gf.Matrix4d().SetScale(scale)*Gf.Matrix4d().SetRotate(q)*Gf.Matrix4d().SetTranslate(tr.GetTranslation())
        if not np.allclose(matrix,rebuilt,atol=1e-7,rtol=0):
            raise ValueError('Unsupported gutter transform')
        positions.append(Gf.Vec3f(*tr.GetTranslation()))
        orientations.append(Gf.Quath(q.GetReal(),Gf.Vec3h(*q.GetImaginary())))
        scales.append(Gf.Vec3f(*scale))
    before=[str(p.GetPath()) for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        inst=UsdGeom.PointInstancer.Define(stage,'/World/GutterVisualInstances')
        proto=UsdGeom.Xform.Define(stage,'/World/GutterVisualInstances/Prototype')
        # Reference the exact composed visual, retaining its material bindings.
        proto.GetPrim().GetReferences().AddInternalReference(modules[0].GetPath())
        proto.GetPrim().SetActive(True);proto.GetPrim().SetInstanceable(False)
        proto.ClearXformOpOrder()
        inst.CreatePrototypesRel().SetTargets([proto.GetPath()])
        inst.CreateProtoIndicesAttr([0]*len(modules))
        inst.CreatePositionsAttr(positions);inst.CreateOrientationsAttr(orientations)
        inst.CreateScalesAttr(scales)
        for prim in modules: prim.SetActive(False)
    after=[str(p.GetPath()) for p in Usd.PrimRange(root) if p.HasAPI(UsdPhysics.CollisionAPI)]
    if before!=after: raise RuntimeError('Batching changed gutter collision membership')
    return dict(module_count=len(modules),point_instancers=1,
        collision_proxies_retained=len(after),source_visual_geometry='unchanged_internal_reference',
        all_gutters_visible=True,source_layers_modified=False)
