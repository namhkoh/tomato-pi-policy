"""Distinguish a held cut branch's own leaf from loss of SHAFT contact.

Leaves never contribute grasp force. Before release, or for any other object,
the original rejection stands. Full native contact/force/penetration guards
remain untouched; this only prevents an incidental retained leaf from making
the aperture controller open a still bilaterally grasped stem.
"""
from .released_debris import ReleasedDebris


class ReleasedGraspContacts:
    def __init__(self,fixture):
        from pxr import UsdGeom,UsdPhysics
        if getattr(fixture,'cut_strategy','bimanual')!='bimanual':
            raise ValueError('Only an actually held bimanual release has retained leaf context')
        self.release=ReleasedDebris(fixture);self.fixture=fixture
        self.joint=UsdPhysics.Joint.Get(fixture.rig.stage,fixture.rig.cut_joint_path)
        self.leaves=frozenset(path for path in self.release.colliders
            if (p:=fixture.rig.stage.GetPrimAtPath(path)).IsA(UsdGeom.Mesh)
            and p.IsActive() and p.HasAPI(UsdPhysics.CollisionAPI)
            and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get() is True)

    def classify(self,result):
        f=self.fixture
        if (f.rig.cut is not True or f.cut_event is not self.release.event
                or result.get('source_target')!=f.rig.source_target
                or self.joint.GetJointEnabledAttr().Get() is not False):
            raise RuntimeError('Released grasp context no longer matches exact disabled seam/event')
        rejected=result['rejected']
        if not rejected:return result
        if not all(r['reason']=='not_connected_detachable_shaft' and r['collider'] in self.leaves
                   for r in rejected):return result
        # Preserve stem_only=False and every rejected row/force for audit.
        # The independent eligible-shaft test includes the same positive load,
        # signed opposition, current pad geometry and penetration checks.
        valid=result.get('eligible_shaft_bilateral') is True
        result['bilateral']=valid
        result['grasp_contact_geometry_valid']=valid
        result['released_leaf_contact_classification']=dict(
            model='exact_released_leaf_incidental_not_grasp_evidence_v1',
            validated_release=True,source_target=f.rig.source_target,
            incidental_colliders=sorted({r['collider'] for r in rejected}),
            eligible_shaft_bilateral=valid,leaf_force_used_for_grasp=False,
            native_guards_changed=False,collision_filters_changed=False)
        return result
