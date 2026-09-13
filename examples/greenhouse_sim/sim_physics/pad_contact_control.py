"""Explicit rigid-pad negative control in an owned, unstarted diagnostic stage.

Changes the pad contact law, not geometry/friction or actuator/safety limits.
This CANNOT qualify the original compliant-pad model. No production default.
"""
from copy import deepcopy
import math


def apply(fixture,*,diagnostic_only=False):
    from pxr import Usd,UsdPhysics,UsdShade
    if diagnostic_only is not True or getattr(fixture,'robot',None) is not None:
        raise ValueError('Explicit owned pre-parse diagnostic pad control required')
    original=deepcopy(fixture.finger_contact_compliance)
    if (not isinstance(original,dict) or original.get('force_based') is not True
            or not all(math.isfinite(original.get(k,float('nan'))) and original[k]>0
                for k in ('stiffness_n_m','damping_n_s_m'))):
        raise ValueError('Original positive force-based compliant pad prior required')
    stage=fixture.stage;path=fixture.root+'/ProbeFingerMaterial'
    prim=stage.GetPrimAtPath(path)
    if not prim or not prim.HasAPI(UsdPhysics.MaterialAPI):
        raise ValueError('Exact authored pad material required')
    api=UsdPhysics.MaterialAPI(prim)
    friction=(api.GetStaticFrictionAttr().Get(),api.GetDynamicFrictionAttr().Get(),api.GetRestitutionAttr().Get())
    keys=('physxMaterial:compliantContactStiffness','physxMaterial:compliantContactDamping')
    expected=(original['stiffness_n_m'],original['damping_n_s_m'])
    for key,value in zip(keys,expected):
        actual=prim.GetAttribute(key).Get()
        if actual is None or not math.isclose(actual,value,rel_tol=1e-6):
            raise ValueError('Authored pad compliance differs from its original prior')
    paths=[p for p in fixture.collider_paths if p.startswith(tuple(f+'/' for f in fixture.paths[1:]))]
    if len(paths)<2:raise ValueError('Complete two-finger collider inventory required')
    for p in paths:
        collider=stage.GetPrimAtPath(p)
        material,_=UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial('physics')
        if (not UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get()
                or str(material.GetPath())!=path):
            raise ValueError('Exact enabled finger collision/material binding required')
    with Usd.EditContext(stage,stage.GetSessionLayer()):
        for key in keys:prim.GetAttribute(key).Set(0.)
    if (any(prim.GetAttribute(key).Get()!=0 for key in keys)
            or friction!=(api.GetStaticFrictionAttr().Get(),api.GetDynamicFrictionAttr().Get(),api.GetRestitutionAttr().Get())):
        raise RuntimeError('Rigid-pad control readback failed')
    record=dict(model='rigid_pad_contact_diagnostic_v1',original_compliant_prior=original,
        stiffness_n_m=0.,damping_n_s_m=0.,force_based=False,compliance_enabled=False,
        pad_material=path,collider_paths=paths,contact_model_changed=True,
        collision_geometry_changed=False,friction_changed=False,force_limits_changed=False,
        original_compliant_contact_qualified=False,training_eligible=False)
    fixture.finger_contact_compliance=deepcopy(record)
    return record
