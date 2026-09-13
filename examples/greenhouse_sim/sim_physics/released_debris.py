"""User-authorized post-cut torso/body contact scoring, never contact filtering.

Only the EXACT released target colliders may use this category, and only after
the measured blade event has disabled the native seam. Attached plants and the
right tool/arm retain all original guards. No forces or geometry are changed.
"""


class ReleasedDebris:
    def __init__(self,fixture):
        from pxr import UsdPhysics
        from .cut_strategy import validate_evidence
        self.fixture=fixture;rig=fixture.rig;event=fixture.cut_event
        if (rig.cut is not True or not isinstance(event,dict)
                or event.get('event')!='blade_contact_joint_release'
                or event.get('physical_cut_verified') is not False
                or event.get('evidence',{}).get('target')!=rig.source_target):
            raise ValueError('Actual qualified blade-release event required before debris scoring')
        validate_evidence(event['evidence'],getattr(fixture,'cut_strategy','bimanual'))
        joint=UsdPhysics.Joint.Get(rig.stage,rig.cut_joint_path)
        if not joint or joint.GetJointEnabledAttr().Get() is not False:
            raise ValueError('Native composed seam must be disabled before debris scoring')
        self.event=event
        self.colliders=frozenset(path for path,i,_,_ in fixture.held_plant_screen.local
            if rig.cut_index<=i<len(rig.body_paths) and path.startswith(rig.body_paths[i]+'/'))
        if not self.colliders:raise ValueError('Exact detached target collider inventory required')
        # Passive body landing is deferred. Active blade/right arm, both hands,
        # cameras and neighboring/attached plants still use the original guards.
        self.bodies=frozenset(p for p in fixture.body_paths if p.startswith(fixture.root+'/')
            and (p[len(fixture.root)+1:] in ('base','wheel_l','wheel_r')
                 or p[len(fixture.root)+1:] in {f'link_torso_{i}' for i in range(1,7)}))
        if not self.bodies:raise ValueError('Exact passive robot body inventory required')

    def matches(self,robot,other):
        f=self.fixture
        if f.rig.cut is not True:return False
        if f.cut_event is not self.event:
            raise RuntimeError('Release identity changed; debris classification refused')
        body=f.root+'/'+robot[len(f.root)+1:].split('/')[0]
        return body in self.bodies and other in self.colliders

    def report(self):
        return dict(model='post_release_exact_target_passive_body_contact_record_only_v1',
            source_target=self.fixture.rig.source_target,
            detached_colliders=sorted(self.colliders),passive_robot_bodies=sorted(self.bodies),
            native_collision_filters_changed=False,forces_modified=False,
            attached_plant_contacts_exempted=False,right_tool_contacts_exempted=False,
            landing_or_damage_certified=False,training_eligible=False)
