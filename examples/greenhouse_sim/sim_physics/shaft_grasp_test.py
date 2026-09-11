"""Portable exact-geometry fixtures, not native grasp validation."""
from dataclasses import replace

import numpy as np
import pytest

from sim_physics.shaft_grasp import FingerPad, ShaftCapsule, ShaftGraspEvidence


def pose(x=0., z=0.):
    result = np.eye(4); result[:3, 3] = [x, 0., z]
    return result


def setup(*, selected=2, max_rows=256, allow_signed_native_normals=False):
    chain = [ShaftCapsule('/T/' + ('Support' if i == 0 else 'Branch') + f'/Segment_{i:03d}',
        '/T/' + ('Support' if i == 0 else 'Branch') + f'/Segment_{i:03d}/StemCollider',
        np.eye(4), .003, .007, .0005) for i in range(5)]
    pads = [FingerPad(f'/R/finger{i}', f'/R/finger{i}/pad', np.eye(4), [.002, .01, .05],
        0, 1 if i == 0 else -1, .0005) for i in range(2)]
    evidence = ShaftGraspEvidence(chain, pads, selected_index=selected, cut_index=1,
        source_target='seed/SubStem_41', max_rows=max_rows,
        allow_signed_native_normals=allow_signed_native_normals)
    frames = {s.body: pose(z=(i - 2) * .02) for i, s in enumerate(chain)}
    frames.update({p.body: pose(x=-.005 if i == 0 else .005) for i, p in enumerate(pads)})
    links = [(a.body, b.body) for a, b in zip(chain[1:], chain[2:])]
    evidence.begin_step(1)
    return evidence, frames, links


def contact(e, finger, segment=2, force=.03, *, reverse=False, point=None, normal=None, separation=0.):
    sign = -1 if finger == 0 else 1
    n = np.array([sign, 0., 0.]) if normal is None else np.asarray(normal, float)
    p = [sign * .003, 0., (segment - 2) * .02] if point is None else point
    a, b = e.pads[finger].collider, e.chain[segment].collider
    if reverse:
        a, b = b, a; n = -n
    e.add_contact(a, b, p, n, force * .01 * n, separation)


def result(e, frames, links, **kwargs):
    return e.evaluate(step_id=kwargs.pop('step_id', 1), frames_step_id=kwargs.pop('frames_step_id', 1),
        dt=kwargs.pop('dt', .01), body_frames=frames, connected_pairs=links, **kwargs)


def test_empty_step_is_not_a_grasp():
    e, frames, links = setup(); r = result(e, frames, links)
    assert not r['bilateral'] and r['counts'] == [0, 0]
    assert r['forces'] == [[0., 0., 0.], [0., 0., 0.]]
    assert r['normal_only'] and not r['friction_used_for_grasp']


@pytest.mark.parametrize('segments', [(2, 2), (1, 3), (1, 2), (2, 3)])
@pytest.mark.parametrize('reverse', [False, True])
def test_same_continuous_shaft_and_header_order(segments, reverse):
    e, frames, links = setup()
    for finger, segment in enumerate(segments):
        contact(e, finger, segment, reverse=reverse if finger == 0 else not reverse)
    r = result(e, frames, links)
    assert r['bilateral'] and r['opposition_cosine'] == pytest.approx(-1.)
    assert r['forces'] == pytest.approx(np.array([[-.03, 0, 0], [.03, 0, 0]]))
    assert {p['body'] for p in r['pairs']} == {e.chain[i].body for i in segments}


def test_selected_view_crosscheck_and_neighbor_split_are_separate():
    e, frames, links = setup()
    contact(e, 0, 1, .024); contact(e, 0, 2, .006); contact(e, 1, 3, .04)
    r = result(e, frames, links)
    assert r['bilateral']
    pair = next(p for p in r['pairs'] if p['body'] == e.chain[2].body)
    assert pair['force_n'] == pytest.approx([-.006, 0., 0.])
    assert r['normal_load_upper_n'] == pytest.approx([.03, .04])


@pytest.mark.parametrize('force', [0., .01999])
def test_contact_count_is_not_force_evidence(force):
    e, frames, links = setup(); contact(e, 0, force=force); contact(e, 1)
    r = result(e, frames, links)
    assert not r['bilateral'] and r['counts'] == [1, 1]


def test_exact_20mn_threshold_retained():
    e, frames, links = setup(); contact(e, 0, force=.02); contact(e, 1, force=.02)
    assert result(e, frames, links)['bilateral']


@pytest.mark.parametrize('other', ['/T/Branch/Segment_002/LeafCollider',
    '/T/Support/Segment_000/StemCollider', '/T/Branch/Segment_004/StemCollider',
    '/Other/Branch/Segment_002/StemCollider', '/T/Branch/Segment_002'])
def test_exact_identity_excludes_leaves_support_distant_and_other_branch(other):
    e, frames, links = setup(); contact(e, 0); contact(e, 1)
    e.add_contact(e.pads[0].collider, other, [-.003, 0., 0.], [-1., 0., 0.], [-.0003, 0., 0.], 0.)
    r = result(e, frames, links)
    assert not r['bilateral'] and not r['stem_only']
    assert r['rejected'][0]['reason'] == 'not_connected_detachable_shaft'


def test_immediate_support_neighbor_never_eligible():
    e, frames, links = setup(selected=1); contact(e, 0, 0); contact(e, 1, 1)
    r = result(e, frames, links)
    assert not r['bilateral']
    assert e.chain[0].collider not in r['eligible_colliders']


def test_missing_live_link_excludes_disconnected_neighbor():
    e, frames, links = setup(); contact(e, 0, 1); contact(e, 1, 2)
    r = result(e, frames, links[1:])
    assert not r['bilateral'] and e.chain[1].collider not in r['eligible_colliders']


def test_nonadjacent_connectivity_cannot_be_invented():
    e, frames, links = setup()
    with pytest.raises(ValueError, match='adjacent'):
        result(e, frames, links + [(e.chain[1].body, e.chain[3].body)])


def test_opposing_rows_on_one_finger_cancel_not_become_positive_load_evidence():
    e, frames, links = setup(); contact(e, 0); contact(e, 0, normal=[1., 0., 0.]); contact(e, 1)
    r = result(e, frames, links)
    assert not r['bilateral'] and r['forces'][0] == pytest.approx([0., 0., 0.])
    assert r['normal_load_upper_n'][0] == pytest.approx(.06)
    assert r['rejected'][0]['reason'] == 'normal_not_compressive_on_capsule_and_pad'


def test_weakly_opposed_mixed_normals_keep_original_cosine_gate():
    e, frames, links = setup()
    contact(e, 0, normal=[-.6, .8, 0.]); contact(e, 1, normal=[.6, .8, 0.])
    r = result(e, frames, links)
    assert r['stem_only'] and r['opposition_cosine'] == pytest.approx(.28)
    assert not r['bilateral']


@pytest.mark.parametrize('point', [[-.006, 0., 0.], [-.003, .02, 0.], [-.003, 0., .04]])
def test_point_must_match_actual_capsule_and_inner_pad(point):
    e, frames, links = setup(); contact(e, 0, point=point); contact(e, 1)
    r = result(e, frames, links)
    assert not r['bilateral'] and r['rejected'][0]['reason'] == 'point_off_capsule_or_inner_pad_face'


def test_capsule_endcap_not_infinite_cylinder():
    e, frames, links = setup()
    z = .0085; x = np.sqrt(.003 ** 2 - .0015 ** 2)
    contact(e, 0, point=[-x, 0., z]); contact(e, 1, point=[x, 0., z])
    assert result(e, frames, links)['bilateral']


@pytest.mark.parametrize('separation', [-.00101, .00101])
def test_penetration_and_contact_offset_not_relaxed(separation):
    e, frames, links = setup(); contact(e, 0, separation=separation); contact(e, 1)
    assert not result(e, frames, links)['bilateral']


def test_fresh_frames_not_authored_rest_geometry():
    e, frames, links = setup(); contact(e, 0); contact(e, 1)
    frames[e.chain[2].body] = pose(x=.01)
    assert not result(e, frames, links)['bilateral']


@pytest.mark.parametrize('signed', [False, True])
def test_rigid_world_rotation_and_local_collider_frames(signed):
    e, frames, links = setup()
    transform = np.array([[0., -1., 0., .2], [1., 0., 0., .3], [0., 0., 1., .4], [0., 0., 0., 1.]])
    # Nonidentity collider local frames must compose with native body poses.
    local = pose(z=.003)
    chain = [replace(s, local_frame=local) for s in e.chain]
    pads = [replace(p, local_frame=local) for p in e.pads]
    fresh = ShaftGraspEvidence(chain, pads, selected_index=2, cut_index=1, source_target='seed/SubStem_41',
        allow_signed_native_normals=signed)
    fresh.begin_step(1)
    for i in range(2):
        sign = -1 if i == 0 else 1
        point = transform[:3, :3] @ [sign * .003, 0., 0.] + transform[:3, 3]
        normal = transform[:3, :3] @ [sign, 0., 0.]
        fresh.add_contact(pads[i].collider, chain[2].collider, point, normal, .0003 * normal, 0.)
    frames = {body: transform @ frame @ np.linalg.inv(local) for body, frame in frames.items()}
    r = result(fresh, frames, links)
    assert r['bilateral'] and r['forces'] == pytest.approx(np.array([[0., -.03, 0.], [0., .03, 0.]]))
    assert r['compressive_support_n'] == pytest.approx([.03, .03])
    assert r['pad_reaction_axes_world'] == [[0., -1., 0.], [0., 1., 0.]]


@pytest.mark.parametrize('bad', ['nan_point', 'zero_normal', 'scaled_normal', 'tangent_impulse', 'negative_impulse', 'nan_separation'])
def test_bad_native_rows_poison_step(bad):
    e, frames, links = setup(); contact(e, 0); contact(e, 1)
    p, n, j, d = [-.003, 0., 0.], [-1., 0., 0.], [-.0003, 0., 0.], 0.
    if bad == 'nan_point': p[0] = np.nan
    if bad == 'zero_normal': n = [0., 0., 0.]
    if bad == 'scaled_normal': n = [-2., 0., 0.]
    if bad == 'tangent_impulse': j = [0., .0003, 0.]
    if bad == 'negative_impulse': j = [.0003, 0., 0.]
    if bad == 'nan_separation': d = np.nan
    with pytest.raises(ValueError):
        e.add_contact(e.pads[0].collider, e.chain[2].collider, p, n, j, d)
    with pytest.raises(ValueError, match='Faulted'):
        result(e, frames, links)


def test_full_buffer_allowed_overflow_fails_and_next_step_fully_resets():
    e, frames, links = setup(max_rows=2); contact(e, 0); contact(e, 1)
    assert result(e, frames, links)['bilateral']
    e.begin_step(2); contact(e, 0); contact(e, 1)
    with pytest.raises(ValueError, match='overflow'): contact(e, 0)
    with pytest.raises(ValueError): result(e, frames, links, step_id=2, frames_step_id=2)
    e.begin_step(3)
    r = result(e, frames, links, step_id=3, frames_step_id=3)
    assert not r['bilateral'] and r['counts'] == [0, 0] and not r['rejected'] and not r['pairs']


@pytest.mark.parametrize('kw', [dict(step_id=0), dict(frames_step_id=0), dict(dt=0), dict(dt=np.nan)])
def test_stale_or_invalid_sampling_rejected(kw):
    e, frames, links = setup(); contact(e, 0); contact(e, 1)
    with pytest.raises(ValueError): result(e, frames, links, **kw)


def test_evaluation_once_and_step_monotonic():
    e, frames, links = setup(); result(e, frames, links)
    with pytest.raises(ValueError): result(e, frames, links)
    with pytest.raises(ValueError): e.begin_step(1)
    with pytest.raises(ValueError): contact(e, 0)


def test_missing_scaled_or_nonfinite_live_frames_fail_closed():
    for bad in ('missing', 'scale', 'nan'):
        e, frames, links = setup()
        if bad == 'missing': del frames[e.chain[1].body]
        if bad == 'scale': frames[e.chain[1].body][0, 0] = 2.
        if bad == 'nan': frames[e.chain[1].body][0, 0] = np.nan
        with pytest.raises((ValueError, KeyError)): result(e, frames, links)


def test_geometry_identity_parameters_and_immutability():
    e, _, _ = setup()
    with pytest.raises(ValueError): replace(e.chain[2], collider=e.chain[2].body + '/Leaf')
    with pytest.raises(ValueError): replace(e.chain[2], radius_m=-1.)
    with pytest.raises(ValueError): replace(e.chain[2], half_height_m=-1.)
    with pytest.raises(ValueError): replace(e.chain[2], contact_offset_m=.01)
    with pytest.raises(ValueError): replace(e.pads[0], half_extents_m=[0, 1, 1])
    with pytest.raises(ValueError): replace(e.pads[0], face_axis=3)
    with pytest.raises(ValueError): e.pads[0].half_extents_m[0] = 1.
    other = replace(e.chain[3], body='/Other/Segment_003', collider='/Other/Segment_003/StemCollider')
    with pytest.raises(ValueError):
        ShaftGraspEvidence([*e.chain[:3], other, e.chain[4]], e.pads, selected_index=2, cut_index=1, source_target='seed')


def test_unrelated_pairs_do_not_fill_sensor_buffer():
    e, frames, links = setup(max_rows=2)
    for _ in range(5): e.add_contact('/R/arm/collider', '/Floor', [], [], [], 0.)
    contact(e, 0); contact(e, 1)
    assert result(e, frames, links)['bilateral']


def test_overflowing_impulse_and_geometry_reject_not_nan_comparison_pass():
    e, frames, links = setup()
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(ValueError): contact(e, 0, force=1e308)
        with pytest.raises(ValueError): result(e, frames, links)
    e, frames, links = setup(); contact(e, 0); contact(e, 1)
    frames[e.chain[2].body] = pose(x=1e308)
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(ValueError): result(e, frames, links)


def test_sensor_order_is_explicit_and_has_no_global_force_cancellation():
    e, frames, links = setup()
    swapped = ShaftGraspEvidence(e.chain, e.pads[::-1], selected_index=2, cut_index=1, source_target='seed')
    swapped.begin_step(1)
    for i in range(2):
        sign = -1 if i == 0 else 1
        swapped.add_contact(e.pads[i].collider, e.chain[2].collider,
            [sign * .003, 0., 0.], [sign, 0., 0.], [sign * .0003, 0., 0.], 0.)
    r = result(swapped, frames, links)
    assert r['bilateral']
    assert r['forces'] == pytest.approx(np.array([[.03, 0., 0.], [-.03, 0., 0.]]))


def test_impulse_conversion_is_per_native_dt_not_commanded_force():
    e, frames, links = setup(); contact(e, 0); contact(e, 1)
    r = result(e, frames, links, dt=.02)
    assert not r['bilateral']
    assert r['normal_load_upper_n'] == pytest.approx([.015, .015])


def signed_row(e, finger, scalar, *, reverse=False, tangent=0.):
    sign = -1 if finger == 0 else 1
    n = np.array([sign, 0., 0.]); j = scalar * n + [0., tangent, 0.]
    a, b = e.pads[finger].collider, e.chain[2].collider
    if reverse:
        a, b = b, a; n = -n; j = -j
    e.add_contact(a, b, [sign * .003, 0., 0.], n, j, 0.)


def test_native35_exact_signed_zero_row_preserved_and_not_grasp_force():
    e, frames, links = setup()
    j = np.array([1.5572793825400967e-21, 2.0827434432837067e-21, -9.079692588736792e-21])
    n = np.array([-.16488264501094818, -.2205180823802948, .9613456726074219])
    original_j, original_n = j.copy(), n.copy()
    a, b = e.pads[0].collider, e.chain[2].collider
    e.add_contact(a, b, [-.003, 0., 0.], n, j, 0.)
    j[:] = 1.; n[:] = 1.  # Audit must own copies of the ORIGINAL callback row.
    contact(e, 1)
    r = result(e, frames, links, dt=1/240)
    assert not r['bilateral'] and r['stem_only']
    assert r['forces'][0] == [0., 0., 0.] and r['normal_load_upper_n'][0] == 0.
    audit, = r['roundoff_normal_impulses']
    assert audit['step_id'] == 1 and audit['row'] == 0
    assert (audit['collider0'], audit['collider1']) == (a, b)
    np.testing.assert_array_equal(audit['normal'], original_n)
    np.testing.assert_array_equal(audit['impulse'], original_j)
    assert audit['signed_scalar_ns'] == pytest.approx(-9.444774048741853e-21, rel=1e-12, abs=0.)
    assert audit['grasp_scalar_ns'] == 0.


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('scalar', [-1e-15, -5e-16])
def test_fixed_signed_zero_boundary_and_header_order(scalar, reverse):
    e, frames, links = setup(); signed_row(e, 0, scalar, reverse=reverse); contact(e, 1)
    r = result(e, frames, links)
    assert not r['bilateral'] and r['forces'][0] == [0., 0., 0.]
    audit, = r['roundoff_normal_impulses']
    assert audit['signed_scalar_ns'] == scalar
    assert audit['collider0'] == (e.chain[2].collider if reverse else e.pads[0].collider)


@pytest.mark.parametrize('scalar', [np.nextafter(-1e-15, -np.inf), -1.001e-15, -1e-6])
def test_negative_just_beyond_fixed_floor_still_faults(scalar):
    e, frames, links = setup()
    with pytest.raises(ValueError, match='compressive and collinear'):
        signed_row(e, 0, scalar)
    assert e.roundoff_normal_impulses == []
    with pytest.raises(ValueError, match='Faulted'): result(e, frames, links)


@pytest.mark.parametrize('scalar', [0., 5e-16])
def test_positive_or_zero_impulses_unchanged_not_roundoff_audited(scalar):
    e, frames, links = setup(); signed_row(e, 0, scalar); contact(e, 1)
    r = result(e, frames, links)
    assert r['roundoff_normal_impulses'] == []
    assert r['forces'][0][0] == pytest.approx(-scalar/.01, rel=1e-12, abs=0.)
    assert not r['bilateral']


@pytest.mark.parametrize('tangent', [2e-12, .0003])
def test_signed_zero_cannot_hide_original_noncollinearity_or_borrow_friction(tangent):
    e, frames, links = setup(); contact(e, 0); contact(e, 1)
    with pytest.raises(ValueError, match='compressive and collinear'):
        signed_row(e, 0, -5e-16, tangent=tangent)
    assert e.roundoff_normal_impulses == []
    with pytest.raises(ValueError): result(e, frames, links)


def test_many_negative_roundoff_rows_never_accumulate_positive_grasp_evidence():
    e, frames, links = setup()
    for finger in (0, 1):
        for _ in range(16): signed_row(e, finger, -1e-15)
    r = result(e, frames, links)
    assert r['forces'] == [[0., 0., 0.], [0., 0., 0.]]
    assert r['normal_load_upper_n'] == [0., 0.] and not r['bilateral']
    assert len(r['roundoff_normal_impulses']) == 32


def test_roundoff_audit_reset_and_no_runtime_tolerance_knob():
    e, frames, links = setup(); signed_row(e, 0, -1e-15)
    first = result(e, frames, links)
    e.begin_step(2)
    second = result(e, frames, links, step_id=2, frames_step_id=2)
    assert second['roundoff_normal_impulses'] == []
    assert len(first['roundoff_normal_impulses']) == 1
    with pytest.raises(TypeError):
        ShaftGraspEvidence(e.chain, e.pads, selected_index=2, cut_index=1,
            source_target='seed', normal_signed_zero_ns=1e-3)


@pytest.mark.parametrize('value', [0, 1, None, 'true', np.bool_(True)])
def test_signed_native_mode_requires_explicit_boolean(value):
    with pytest.raises(ValueError, match='boolean'):
        setup(allow_signed_native_normals=value)


def test_signed_native_mode_is_default_strict_and_constructor_only():
    e, frames, links = setup()
    assert not e.allow_signed_native_normals
    with pytest.raises(AttributeError): e.allow_signed_native_normals = True
    with pytest.raises(ValueError, match='compressive and collinear'):
        signed_row(e, 0, -4.2010698364712225e-8)
    with pytest.raises(ValueError): result(e, frames, links)


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('negative', [-.005, -.02, -.02001, -.04, -.07])
def test_signed_rows_reduce_support_not_upper_bound_or_header_invariance(negative, reverse):
    e, frames, links = setup(allow_signed_native_normals=True)
    contact(e, 0, force=.04, reverse=reverse)
    contact(e, 0, force=negative, reverse=not reverse)
    contact(e, 1, force=.03, reverse=reverse)
    r = result(e, frames, links)
    assert r['forces'] == pytest.approx(np.array([[-(.04+negative), 0., 0.], [.03, 0., 0.]]))
    assert r['compressive_support_n'] == pytest.approx([.04+negative, .03])
    assert r['normal_load_upper_n'] == pytest.approx([.04-negative, .03])
    pair = next(p for p in r['pairs'] if p['finger'] == e.pads[0].body)
    assert pair['force_n'] == pytest.approx(r['forces'][0])
    assert pair['normal_load_upper_n'] == pytest.approx(.04-negative)
    assert r['bilateral'] == (negative >= -.02)
    assert r['stem_only'] and r['allow_signed_native_normals'] and r['compressive_support_gate_applied']
    assert r['roundoff_normal_impulses'] == []
    audit, = r['negative_normal_impulses']
    assert audit['signed_scalar_ns'] == negative * .01 == audit['grasp_scalar_ns']
    assert audit['impulse_on_finger_ns'] == pytest.approx([-negative * .01, 0., 0.])
    assert not audit['within_signed_zero_floor']


@pytest.mark.parametrize('forces', [(-.03, -.03), (-.03, .03), (.03, -.03)])
def test_signed_tensile_resultants_cannot_pass_norm_and_opposition_alone(forces):
    e, frames, links = setup(allow_signed_native_normals=True)
    for i, f in enumerate(forces): contact(e, i, force=f)
    r = result(e, frames, links)
    assert np.all(np.linalg.norm(r['forces'], axis=1) >= .02)
    assert r['compressive_support_n'] == pytest.approx(forces)
    assert not r['bilateral'] and not r['compressive_support_passed']
    if forces[0] < 0 and forces[1] < 0: assert r['opposition_cosine'] == pytest.approx(-1.)


@pytest.mark.parametrize('signed', [False, True])
def test_projection_not_unsigned_load_or_resultant_norm_and_strict_default_unchanged(signed):
    e, frames, links = setup(allow_signed_native_normals=signed)
    n = np.array([-.1, np.sqrt(.99), 0.])
    contact(e, 0, normal=n); contact(e, 1, normal=-n)
    r = result(e, frames, links)
    assert r['stem_only'] and r['opposition_cosine'] == pytest.approx(-1.)
    assert r['normal_load_upper_n'] == pytest.approx([.03, .03])
    assert r['compressive_support_n'] == pytest.approx([.003, .003])
    assert r['bilateral'] == (not signed)


def test_signed_mode_retains_opposition_gate_even_when_both_projections_pass():
    e, frames, links = setup(allow_signed_native_normals=True)
    contact(e, 0, force=.05, normal=[-.6, .8, 0.])
    contact(e, 1, force=.05, normal=[.6, .8, 0.])
    r = result(e, frames, links)
    assert r['compressive_support_passed'] and r['compressive_support_n'] == pytest.approx([.03, .03])
    assert r['opposition_cosine'] == pytest.approx(.28) and not r['bilateral']


def test_negative_neighbor_contribution_reduces_same_connected_shaft_support():
    e, frames, links = setup(allow_signed_native_normals=True)
    contact(e, 0, 1, .015); contact(e, 0, 2, .015); contact(e, 0, 3, -.011); contact(e, 1, 1)
    r = result(e, frames, links)
    assert r['stem_only'] and not r['bilateral']
    assert r['compressive_support_n'] == pytest.approx([.019, .03])
    assert r['normal_load_upper_n'] == pytest.approx([.041, .03])
    assert r['negative_normal_impulses'][0]['other_collider'] == e.chain[3].collider


@pytest.mark.parametrize('scalar', [-1e-15, -5e-16, -9.444774048741853e-21])
def test_signed_opt_in_keeps_even_tiny_negative_projection_without_increasing_floor(scalar):
    e, frames, links = setup(allow_signed_native_normals=True)
    signed_row(e, 0, scalar); signed_row(e, 1, scalar)
    r = result(e, frames, links)
    assert r['compressive_support_n'] == pytest.approx([scalar/.01]*2, rel=1e-12, abs=0.)
    assert r['normal_load_upper_n'] == pytest.approx([-scalar/.01]*2, rel=1e-12, abs=0.)
    assert not r['bilateral'] and not r['roundoff_normal_impulses']
    assert all(a['within_signed_zero_floor'] and a['grasp_scalar_ns'] == scalar
               for a in r['negative_normal_impulses'])


def test_native37_exact_row_signed_projection_and_original_audit_in_rotated_fixture():
    e, frames, links = setup(allow_signed_native_normals=True)
    j = np.array([6.927241003040763e-9, 9.263960265570859e-9, -4.0386769484257456e-8])
    n = np.array([-.164892315864563, -.22051431238651276, .9613448977470398])
    p = np.array([.04372051730751991, .5745751857757568, 1.4217487573623657])
    unit = n / np.linalg.norm(n); scalar = float(j @ unit); original_j = j.copy()
    # Rigidly place the synthetic exact pad/capsule fixture at the observed row.
    rx = -unit; ry = np.cross([0., 0., 1.], rx); ry /= np.linalg.norm(ry)
    transform = np.eye(4); transform[:3, :3] = np.column_stack([rx, ry, np.cross(rx, ry)])
    transform[:3, 3] = p - transform[:3, :3] @ [-.003, 0., 0.]
    frames = {body: transform @ frame for body, frame in frames.items()}
    e.add_contact(e.pads[0].collider, e.chain[2].collider, p, n, j, -1.2320466339588165e-5)
    j[:] = 0.  # Core/audit must not alias external callback buffers.
    dt = 1/240
    for i in (0, 1):
        normal = unit if i == 0 else -unit
        point = transform[:3, :3] @ [-.003 if i == 0 else .003, 0., .002] + transform[:3, 3]
        e.add_contact(e.pads[i].collider, e.chain[2].collider, point, normal, .03*dt*normal, 0.)
    r = result(e, frames, links, dt=dt)
    assert r['bilateral'] and r['stem_only']  # Synthetic mechanics assertion, not native qualification.
    assert scalar/dt == pytest.approx(-1.0082567607530934e-5, rel=1e-12)
    assert r['compressive_support_n'] == pytest.approx([.03+scalar/dt, .03])
    assert r['normal_load_upper_n'] == pytest.approx([.03-scalar/dt, .03])
    audit, = r['negative_normal_impulses']
    np.testing.assert_array_equal(audit['impulse'], original_j)
    np.testing.assert_array_equal(audit['normal'], n)
    np.testing.assert_array_equal(audit['point'], p)
    assert audit['step_id'] == 1 and audit['row'] == 0 and audit['grasp_scalar_ns'] == scalar


@pytest.mark.parametrize('other', ['/T/Branch/Segment_002/LeafCollider', '/T/Support/Segment_000/StemCollider',
    '/T/Branch/Segment_004/StemCollider', '/Other/Branch/Segment_002/StemCollider'])
def test_signed_mode_does_not_broaden_identity_masks(other):
    e, frames, links = setup(allow_signed_native_normals=True); contact(e, 0); contact(e, 1)
    e.add_contact(e.pads[0].collider, other, [-.003, 0., 0.], [-1., 0., 0.], [.0001, 0., 0.], 0.)
    r = result(e, frames, links)
    assert not r['bilateral'] and not r['stem_only']
    assert r['rejected'][0]['reason'] == 'not_connected_detachable_shaft'
    assert len(r['negative_normal_impulses']) == 1


@pytest.mark.parametrize('bad', ['nan_point', 'zero_normal', 'scaled_normal', 'tangent', 'nan_impulse', 'nan_separation'])
def test_signed_mode_still_poisons_nonfinite_noncollinear_and_friction_rows(bad):
    e, frames, links = setup(allow_signed_native_normals=True); contact(e, 0); contact(e, 1)
    p, n, j, d = [-.003, 0., 0.], [-1., 0., 0.], [.0003, 0., 0.], 0.
    if bad == 'nan_point': p[0] = np.nan
    if bad == 'zero_normal': n = [0., 0., 0.]
    if bad == 'scaled_normal': n = [-2., 0., 0.]
    if bad == 'tangent': j[1] = .0003
    if bad == 'nan_impulse': j[0] = np.nan
    if bad == 'nan_separation': d = np.nan
    with pytest.raises(ValueError): e.add_contact(e.pads[0].collider, e.chain[2].collider, p, n, j, d)
    with pytest.raises(ValueError): result(e, frames, links)
    assert e.negative_normal_impulses == []


@pytest.mark.parametrize('bad', ['off_surface', 'back_normal', 'penetration', 'separation', 'disconnected', 'stale_frame'])
def test_signed_mode_keeps_geometry_connectivity_and_penetration_guards(bad):
    e, frames, links = setup(allow_signed_native_normals=True); contact(e, 0); contact(e, 1)
    options = dict(force=-.005)
    if bad == 'off_surface': options['point'] = [-.006, 0., 0.]
    if bad == 'back_normal': options['normal'] = [1., 0., 0.]
    if bad == 'penetration': options['separation'] = -.00101
    if bad == 'separation': options['separation'] = .00101
    contact(e, 0, segment=1 if bad == 'disconnected' else 2, **options)
    if bad == 'disconnected': links = links[1:]
    if bad == 'stale_frame': frames[e.chain[2].body] = pose(x=.01)
    r = result(e, frames, links)
    assert not r['bilateral'] and not r['stem_only'] and r['rejected']


def test_signed_buffer_overflow_latches_and_full_step_reset_clears_audit_and_support():
    e, frames, links = setup(max_rows=2, allow_signed_native_normals=True)
    contact(e, 0, force=-.03); contact(e, 1, force=-.03)
    first = result(e, frames, links)
    assert len(first['negative_normal_impulses']) == 2
    e.begin_step(2); contact(e, 0, force=-.03); contact(e, 1, force=-.03)
    with pytest.raises(ValueError, match='overflow'): contact(e, 0)
    with pytest.raises(ValueError): result(e, frames, links, step_id=2, frames_step_id=2)
    e.begin_step(3)
    r = result(e, frames, links, step_id=3, frames_step_id=3)
    assert r['negative_normal_impulses'] == r['roundoff_normal_impulses'] == []
    assert r['compressive_support_n'] == [0., 0.] and r['normal_load_upper_n'] == [0., 0.]
    assert r['pairs'] == [] and r['counts'] == [0, 0] and not r['bilateral']
    assert len(first['negative_normal_impulses']) == 2
