"""Explicit local contact-compression experiment; not calibrated tissue.

Only the two original seam-adjacent shaft collision surfaces get a native
implicit force-based spring. Geometry, masses, joints, robot materials and
release criteria are untouched. This does not implement volumetric deformation
or fracture energy. Never enable it silently for a production greenhouse.
"""
import math


def apply(rig, *, diagnostic_only=False):
    from pxr import Sdf,Usd,UsdGeom,UsdPhysics,UsdShade
    from .plant import physics_schema
    if (diagnostic_only is not True or getattr(rig,'cut',None) is not False
            or getattr(rig,'stem_contact_model',None)!='flat_cylinders_v1'
            or type(rig.cut_index) is not int or not 0<rig.cut_index<len(rig.body_paths)):
        raise ValueError('Intact isolated flat-shaft diagnostic required for contact compression')
    stage=rig.stage
    if (abs(UsdGeom.GetStageMetersPerUnit(stage)-1)>1e-9
            or UsdGeom.GetStageUpAxis(stage)!='Z'):
        raise ValueError('Metres and Z-up required')
    joint=stage.GetPrimAtPath(rig.cut_joint_path)
    if (not joint or not joint.IsA(UsdPhysics.FixedJoint)
            or UsdPhysics.Joint(joint).GetJointEnabledAttr().Get() is not True):
        raise ValueError('Original intact fixed seam required')
    paths=[rig.body_paths[i]+'/StemCollider' for i in (rig.cut_index-1,rig.cut_index)]
    material_path=rig.root+'/ExperimentalSeamContactMaterial'
    if stage.GetPrimAtPath(material_path):raise ValueError('Refuse to overwrite a contact material')
    shapes=[]
    for path in paths:
        prim=stage.GetPrimAtPath(path)
        if (not prim or not prim.IsA(UsdGeom.Cylinder)
                or not prim.HasAPI(UsdPhysics.CollisionAPI)
                or UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is not True):
            raise ValueError('Two original active seam cylinders required')
        bound,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
        if bound:raise ValueError('Existing physics material must not be silently replaced')
        # Guard against unrelated transformed/offset replacement surfaces.
        for name,expected in (('physxCollision:contactOffset',.0005),('physxCollision:restOffset',0.)):
            value=prim.GetAttribute(name).Get()
            if value is None or not math.isfinite(value) or abs(value-expected)>1e-9:
                raise ValueError('Unchanged original contact/rest offsets required')
        shapes.append(prim)
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        material=UsdShade.Material.Define(stage,material_path)
        api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
        # Explicit default friction/restitution, not a lower-friction knife.
        api.CreateStaticFrictionAttr(.5);api.CreateDynamicFrictionAttr(.5);api.CreateRestitutionAttr(0.)
        physics_schema(material.GetPrim(),'PhysxMaterialAPI',[
            ('physxMaterial:compliantContactStiffness',Sdf.ValueTypeNames.Float,1000.),
            ('physxMaterial:compliantContactDamping',Sdf.ValueTypeNames.Float,2.),
            ('physxMaterial:compliantContactAccelerationSpring',Sdf.ValueTypeNames.Bool,False)])
        for prim in shapes:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,materialPurpose='physics')
        for prim in shapes:
            bound,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial(materialPurpose='physics')
            if not bound or str(bound.GetPath())!=material_path:
                raise RuntimeError('Experimental contact binding failed; do not start physics')
    return dict(model='seam_contact_linear1000_v1',collider_paths=paths,
        material_path=material_path,stiffness_n_m=1000.,damping_n_s_m=2.,force_based=True,
        static_friction=.5,dynamic_friction=.5,restitution=0.,
        parameters_origin='uncalibrated_engineering_prior',calibrated=False,
        scope='two_seam_adjacent_shaft_surfaces_isolated_diagnostic_only',
        deformation='implicit_contact_compression_proxy_not_volumetric_tissue',
        geometry_changed=False,masses_changed=False,seam_joint_changed=False,
        release_thresholds_changed=False,knife_softened=False,source_assets_edited=False,
        native_material_readback=False,training_eligible=False)
