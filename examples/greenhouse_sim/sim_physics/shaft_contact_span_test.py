"""Physical identity selection tests, not native mechanical qualification."""
import pytest
from .shaft_grasp import ShaftGraspEvidence
from .shaft_grasp_test import setup, contact, result, pose


def case(**kw):
    old,frames,links=setup()
    e=ShaftGraspEvidence(old.chain,old.pads,selected_index=2,cut_index=1,
        source_target=old.source_target,physical_grasp_span=True,**kw)
    e.begin_step(1)
    return e,frames,links


def test_more_than_one_neighbor_under_actual_pads_is_not_a_different_branch():
    e,frames,links=case()
    contact(e,0,2);contact(e,1,4)
    r=result(e,frames,links)
    assert r['bilateral'] and r['stem_only']
    assert r['physical_grasp_span']['eligible_indices']==[1,2,3,4]
    assert not r['physical_grasp_span']['grasp_verified']
    assert not r['physical_grasp_span']['contact_filters_changed']
    assert r['normal_load_upper_n']==pytest.approx([.03,.03])
    old,frames,links=setup();contact(old,0,2);contact(old,1,4)
    assert not result(old,frames,links)['bilateral']


@pytest.mark.parametrize('removed',[(2,3),(3,4)])
def test_every_intervening_joint_must_still_be_connected(removed):
    e,frames,links=case();contact(e,0,2);contact(e,1,4)
    missing={e.chain[i].body for i in removed}
    r=result(e,frames,[v for v in links if set(v)!=missing])
    assert not r['bilateral'] and e.chain[4].collider not in r['eligible_colliders']


def test_cannot_skip_an_out_of_footprint_link_to_a_folded_back_segment():
    e,frames,links=case();frames[e.chain[3].body]=pose(z=.2)
    contact(e,0,2);contact(e,1,4)
    r=result(e,frames,links)
    assert not r['bilateral'] and e.chain[4].collider not in r['eligible_colliders']


def test_current_measured_pad_motion_changes_identity_without_commands():
    e,frames,links=case()
    for pad in e.pads:frames[pad.body][:3,3]+=[0,0,.2]
    contact(e,0,2);contact(e,1,4)
    r=result(e,frames,links)
    assert r['eligible_colliders']==[] and not r['bilateral']


@pytest.mark.parametrize('other',[
    '/T/Support/Segment_000/StemCollider','/T/Branch/Segment_004/LeafCollider',
    '/Other/Branch/Segment_004/StemCollider','/T/Branch/Segment_004'])
def test_support_leaves_other_branches_and_body_aggregates_still_reject(other):
    e,frames,links=case();contact(e,0,2);contact(e,1,4)
    e.add_contact(e.pads[0].collider,other,[-.003,0,.04],[-1,0,0],[-.0003,0,0],0.)
    r=result(e,frames,links)
    assert not r['bilateral'] and not r['stem_only']


@pytest.mark.parametrize('force',[0.,.01999])
def test_eligible_proximity_or_weak_contact_never_becomes_a_grasp(force):
    e,frames,links=case(allow_signed_native_normals=True)
    contact(e,0,4,force);contact(e,1,4,force)
    r=result(e,frames,links)
    assert r['stem_only'] and not r['bilateral'] and not r['compressive_support_passed']


def test_extended_identity_still_requires_actual_inner_face_contact():
    e,frames,links=case();contact(e,0,2);contact(e,1,4,point=[.003,.02,.04])
    r=result(e,frames,links)
    assert e.chain[4].collider in r['eligible_colliders']
    assert not r['bilateral'] and r['rejected'][0]['reason']=='point_off_capsule_or_inner_pad_face'


def test_stale_frames_still_rejected():
    e,frames,links=case();contact(e,0,4);contact(e,1,4)
    with pytest.raises(ValueError,match='stale'):result(e,frames,links,frames_step_id=0)


def test_native_adapter_records_mode_and_honors_live_joint_break():
    from pxr import UsdPhysics
    from .shaft_grasp_native_test import fixture, bind, add, evaluate
    f=fixture();old=bind(f);oldhash=old.binding_sha256;old.close()
    a=bind(f,physical_grasp_span=True)
    assert a.binding_sha256!=oldhash
    try:
        a.begin_step()
        for i in (0,1):add(a,f,i,4,point=[.003 if i==0 else -.003,0,.031])
        r=evaluate(a,f,frames_step_id=1)
        assert r['adapter_valid'] and r['bilateral']
        UsdPhysics.Joint.Get(f.stage,'/T/Branch/Joint_004').GetJointEnabledAttr().Set(False)
        a.begin_step()
        for i in (0,1):add(a,f,i,4,point=[.003 if i==0 else -.003,0,.031])
        r=evaluate(a,f,frames_step_id=2)
        assert r['adapter_valid'] and not r['bilateral']
        assert f.rig.body_paths[4]+'/StemCollider' not in r['eligible_colliders']
    finally:a.close()


def test_cli_requires_complete_retention_trial(tmp_path):
    from .benchmark import main, parser
    assert not parser().parse_args(['--output','unused']).physical_grasp_span
    with pytest.raises(ValueError,match='Physical grasp span'):
        main(['--output',str(tmp_path/'unused'),'--physical-grasp-span'])
    assert not (tmp_path/'unused').exists()


def test_bimanual_opt_in_reaches_new_native_binding(monkeypatch):
    from .bimanual_grasp_test import binding_robot
    robot,view,events,new,new_monitor=binding_robot(monkeypatch)
    robot.physical_grasp_span=True
    robot.bind(view)
    assert robot.grasp_observer is new and new_monitor.normal_contact_observer is new
