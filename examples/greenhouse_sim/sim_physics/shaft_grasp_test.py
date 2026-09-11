"""Portable exact-geometry fixtures, not native grasp validation."""
from dataclasses import replace

import numpy as np
import pytest

from sim_physics.shaft_grasp import FingerPad, ShaftCapsule, ShaftGraspEvidence


def pose(x=0., z=0.):
    result = np.eye(4); result[:3, 3] = [x, 0., z]
    return result


def setup(*, selected=2, max_rows=256):
    chain = [ShaftCapsule('/T/' + ('Support' if i == 0 else 'Branch') + f'/Segment_{i:03d}',
        '/T/' + ('Support' if i == 0 else 'Branch') + f'/Segment_{i:03d}/StemCollider',
        np.eye(4), .003, .007, .0005) for i in range(5)]
    pads = [FingerPad(f'/R/finger{i}', f'/R/finger{i}/pad', np.eye(4), [.002, .01, .05],
        0, 1 if i == 0 else -1, .0005) for i in range(2)]
    evidence = ShaftGraspEvidence(chain, pads, selected_index=selected, cut_index=1,
        source_target='seed/SubStem_41', max_rows=max_rows)
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


def test_rigid_world_rotation_and_local_collider_frames():
    e, frames, links = setup()
    transform = np.array([[0., -1., 0., .2], [1., 0., 0., .3], [0., 0., 1., .4], [0., 0., 0., 1.]])
    # Nonidentity collider local frames must compose with native body poses.
    local = pose(z=.003)
    chain = [replace(s, local_frame=local) for s in e.chain]
    pads = [replace(p, local_frame=local) for p in e.pads]
    fresh = ShaftGraspEvidence(chain, pads, selected_index=2, cut_index=1, source_target='seed/SubStem_41')
    fresh.begin_step(1)
    for i in range(2):
        sign = -1 if i == 0 else 1
        point = transform[:3, :3] @ [sign * .003, 0., 0.] + transform[:3, 3]
        normal = transform[:3, :3] @ [sign, 0., 0.]
        fresh.add_contact(pads[i].collider, chain[2].collider, point, normal, .0003 * normal, 0.)
    frames = {body: transform @ frame @ np.linalg.inv(local) for body, frame in frames.items()}
    r = result(fresh, frames, links)
    assert r['bilateral'] and r['forces'] == pytest.approx(np.array([[0., -.03, 0.], [0., .03, 0.]]))


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
