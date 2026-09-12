"""Portable timing/proximity tests, not native generation-pose certification."""
import numpy as np
import pytest
from sim_physics.shaft_grasp_test import setup,contact,result
from sim_physics.shaft_grasp_native_test import fixture,bind as make_adapter


def shifted(frames,delta):
    out={p:m.copy() for p,m in frames.items()}
    for m in out.values():m[:3,3]+=delta
    return out


@pytest.mark.parametrize('force',[0.,1e-10,.03])
def test_separated_persistent_zero_row_does_not_veto_actual_support(force):
    e,pre,links=setup();contact(e,0);contact(e,1)
    contact(e,0,segment=1,force=force,separation=.0012)
    r=result(e,pre,links,contact_body_frames=pre,contact_frames_step_id=0)
    assert r['bilateral']==(force==0.)
    assert len(r['inactive_manifold_rows'])==(1 if force==0. else 0)
    assert r['counts']==[2,1]  # Original zero row remains accounted for.


def test_zero_rows_alone_never_verify_grasp_and_penetration_still_rejects():
    e,pre,links=setup()
    contact(e,0,force=0.,separation=.0012);contact(e,1,force=0.,separation=.0012)
    r=result(e,pre,links,contact_body_frames=pre,contact_frames_step_id=0)
    assert not r['bilateral'] and len(r['inactive_manifold_rows'])==2
    e,pre,links=setup();contact(e,0);contact(e,1)
    contact(e,0,force=0.,separation=-.0012)
    assert not result(e,pre,links)['bilateral']


def test_zero_separated_row_still_needs_valid_surface_geometry():
    e,pre,links=setup();contact(e,0);contact(e,1)
    contact(e,0,force=0.,separation=.0012,point=[-.003,.1,0.])
    r=result(e,pre,links)
    assert not r['bilateral'] and not r['stem_only']


def test_moving_assembly_uses_contact_geometry_not_old_point_in_new_frame():
    e,pre,links=setup()
    contact(e,0);contact(e,1)
    post=shifted(pre,[0,.02,0])
    r=result(e,post,links,contact_body_frames=pre,contact_frames_step_id=0)
    assert r['bilateral']
    assert all(p['passed'] for p in r['current_contact_proximity'])
    assert r['contact_geometry_step_id']==0
    assert r['contact_geometry_basis']=='caller_pre_step_PGS'


def test_legacy_point_comparison_still_reproducible():
    e,pre,links=setup();contact(e,0);contact(e,1)
    assert not result(e,shifted(pre,[0,.02,0]),links)['bilateral']


@pytest.mark.parametrize('mode',['withdraw','penetrate','wrong_face','through_pad'])
def test_pre_step_contacts_do_not_prove_current_hold(mode):
    e,pre,links=setup();contact(e,0);contact(e,1)
    post={p:m.copy() for p,m in pre.items()}
    if mode=='withdraw':post[e.pads[0].body][0,3]-=.004
    elif mode=='penetrate':post[e.pads[0].body][0,3]+=.002
    elif mode=='through_pad':post[e.pads[0].body][0,3]+=.006
    else:post[e.pads[0].body][1,3]+=.02
    r=result(e,post,links,contact_body_frames=pre,contact_frames_step_id=0)
    assert r['stem_only']  # Valid generated contacts, not a reason to squeeze harder.
    assert not r['bilateral']
    assert not all(p['passed'] for p in r['current_contact_proximity'])


def test_bad_generation_point_and_wrong_identity_still_rejected():
    e,pre,links=setup();contact(e,0,point=[-.003,.02,0]);contact(e,1)
    r=result(e,pre,links,contact_body_frames=pre,contact_frames_step_id=0)
    assert not r['stem_only'] and not r['bilateral']


@pytest.mark.parametrize('extra_force',[0.,.001])
def test_zero_impulse_manifold_row_is_not_support_or_a_lost_loaded_contact(extra_force):
    e,pre,links=setup();contact(e,0);contact(e,1);contact(e,0,segment=1,force=extra_force)
    post={p:m.copy() for p,m in pre.items()};post[e.chain[1].body][0,3]+=.02
    r=result(e,post,links,contact_body_frames=pre,contact_frames_step_id=0)
    assert r['stem_only']
    assert r['bilateral']==(extra_force==0.)
    absent=next(p for p in r['current_contact_proximity'] if p['collider']==e.chain[1].collider)
    assert not absent['passed'] and absent['carries_nonzero_impulse']==(extra_force!=0.)


@pytest.mark.parametrize('offset',[0,1,-2])
def test_generation_step_cannot_be_current_future_or_stale(offset):
    e,pre,links=setup()
    with pytest.raises(ValueError):
        result(e,pre,links,contact_body_frames=pre,contact_frames_step_id=1+offset)


def test_native_capture_requires_exact_adjacent_ordered_owned_copy():
    f=fixture();a=make_adapter(f)
    a.begin_step()
    frames=f.frames.copy()
    fingers=f.fingerframes.copy()
    a.capture_contact_frames(frames,fingers,step_id=0)
    original=a._contact_frames[1][a.body_paths[0]].copy()
    frames[0,0,3]+=1
    assert np.array_equal(a._contact_frames[1][a.body_paths[0]],original)
    with pytest.raises(ValueError):a.capture_contact_frames(frames,fingers,step_id=0)
    a.begin_step()
    assert a._contact_frames is None
    with pytest.raises(ValueError):a.capture_contact_frames(frames,fingers,step_id=0)
    a.close()


def test_native_missing_pre_snapshot_cannot_fall_back_to_post_fetch():
    f=fixture();a=make_adapter(f);a.begin_step()
    with pytest.raises(ValueError,match='Missing qualified'):
        a.evaluate(1/240,f.frames,f.fingerframes,frames_step_id=1,require_pre_step_frames=True)
    a.close()


@pytest.mark.parametrize('extra',[[],['--bimanual-cut','--full-robot-probe','--solver','TGS'],
    ['--bimanual-cut','--full-robot-probe','--solver','PGS','--physics-hz','480']])
def test_unqualified_phase_contract_rejected_before_native_or_output(tmp_path,extra):
    from sim_physics.benchmark import main
    output=tmp_path/'unused'
    with pytest.raises(ValueError,match='Pre-step grasp frames require'):
        main(['--output',str(output),'--grasp-contact-frames','pre_solve_pgs_v1',*extra])
    assert not output.exists()
