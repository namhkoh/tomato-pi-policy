from copy import deepcopy
import pytest
from pxr import UsdGeom,UsdPhysics
from .released_grasp import ReleasedGraspContacts


def policy():
    from .cut_action_test import fixture,release
    f,j,_=fixture()
    for name in ('A','B'):
        UsdGeom.Cylinder.Define(f.rig.stage,'/World/'+name+'/StemCollider')
    p=UsdGeom.Mesh.Define(f.rig.stage,'/World/B/Leaf/Collision').GetPrim()
    UsdPhysics.CollisionAPI.Apply(p).CreateCollisionEnabledAttr(True)
    release(f)
    return f,j,ReleasedGraspContacts(f)


def row(f,collider='/World/B/Leaf/Collision',supported=True):
    return dict(source_target=f.rig.source_target,stem_only=False,bilateral=False,
        eligible_shaft_bilateral=supported,forces=[[.03,0,0],[-.03,0,0]],
        rejected=[dict(reason='not_connected_detachable_shaft',collider=collider)])


def test_retained_leaf_does_not_count_as_force_or_hide_rejection():
    f,_,p=policy();r=row(f);before=deepcopy(r)
    out=p.classify(r)
    assert out['bilateral'] and out['grasp_contact_geometry_valid']
    assert not out['stem_only']
    assert out['rejected']==before['rejected'] and out['forces']==before['forces']
    assert not out['released_leaf_contact_classification']['leaf_force_used_for_grasp']
    assert not out['released_leaf_contact_classification']['native_guards_changed']


@pytest.mark.parametrize('path',['/World/Neighbor/Leaf/Collision','/World/A/StemCollider',
    '/World/B/StemCollider','/World/B/Leaf/Collision_other','/Robot/Camera'])
def test_not_a_general_contact_exception(path):
    f,_,p=policy();r=row(f,path);before=deepcopy(r)
    assert p.classify(r)==before and not r['bilateral']


def test_leaf_contact_alone_cannot_create_a_grasp():
    f,_,p=policy();r=row(f,supported=False)
    assert not p.classify(r)['bilateral'] and not r['grasp_contact_geometry_valid']


def test_attached_and_reset_or_changed_event_are_not_released_context():
    from .cut_action_test import fixture
    f,_,_=fixture()
    with pytest.raises(ValueError):ReleasedGraspContacts(f)
    f,_,p=policy();f.cut_event=deepcopy(f.cut_event)
    with pytest.raises(RuntimeError):p.classify(row(f))


def test_core_separates_eligible_shaft_support_but_keeps_default_rejection():
    from .shaft_grasp_test import setup,contact,result
    e,frames,links=setup();contact(e,0);contact(e,1)
    e.add_contact(e.pads[0].collider,'/T/Branch/Segment_003/Leaf/Collision',
        [-.003,0,0],[-1,0,0],[-.001,0,0],0.)
    r=result(e,frames,links)
    assert r['eligible_shaft_bilateral'] and not r['bilateral'] and not r['stem_only']
    assert r['normal_load_upper_n']==pytest.approx([.03,.03])


def test_core_bad_shaft_geometry_cannot_be_excused_by_leaf_context():
    from .shaft_grasp_test import setup,contact,result
    e,frames,links=setup();contact(e,0,separation=-.002);contact(e,1)
    r=result(e,frames,links)
    assert not r['eligible_shaft_bilateral'] and not r['bilateral']


def test_closure_keeps_measured_shaft_support_but_requires_release_context():
    from .force_closure import ForceClosure
    from .force_closure_test import sample
    c=ForceClosure(.003,.0005);c.command(1.,step=0,dt=1/240)
    f,_,p=policy();r=row(f);r.update(sample(1,(.12,.12)));r['stem_only']=False
    p.classify(r);c.observe(r,[.13,.13],step=1,guards_passed=True)
    assert c.geometry_valid and c.support==pytest.approx([.12,.12])
    c=ForceClosure(.003,.0005);c.command(1.,step=0,dt=1/240)
    r.pop('released_leaf_contact_classification')
    with pytest.raises(RuntimeError,match='released-leaf context'):c.observe(r,[.13,.13],step=1,guards_passed=True)
