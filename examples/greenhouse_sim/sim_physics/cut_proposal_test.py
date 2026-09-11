"""Portable single-proposal checks; no simulator or source scene construction."""
from dataclasses import FrozenInstanceError
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from sim_physics.cut_proposal import (MAX_JSON_BYTES, SCHEMA, load_cut_proposal,
    parse_cut_proposal, resolve_cut_proposal, world_stroke_from_reference)
from sim_physics.knife import cut_plane_normal


TARGET = "seed101_full/SubStem_41"
WIDTH = .04952825137749362


@pytest.fixture
def document():
    return dict(schema=SCHEMA, source_target=TARGET, world_direction=[1., 0., 0.],
                normal_sign=-1, wing_m=-.01778771311987213, plane_tilt_degrees=0.,
                maximum_projection_degrees=1., provenance=dict(
                    sources={name:dict(path=name+".json", sha256="a"*64) for name in ("report", "snapshot")},
                    derivation=dict(method="synthetic unit-test vectors, NOT native evidence")))


def resolve(proposal, axis=(0., 0., 1.), **kwargs):
    return resolve_cut_proposal(proposal, source_target=kwargs.pop("source_target", TARGET),
                                stem_axis_world=axis, edge_width_m=kwargs.pop("edge_width_m", WIDTH), **kwargs)


def fake_file(monkeypatch, raw):
    monkeypatch.setattr(Path, "open", lambda *a, **k: io.BytesIO(raw))


def test_optional_none_does_not_open_anything(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Absent proposal must not read a file")
    monkeypatch.setattr(Path, "open", forbidden)
    assert load_cut_proposal(None, source_target=TARGET) is None


def test_load_is_single_source_bound_immutable_and_byte_hashed(document, monkeypatch):
    raw = json.dumps(document).encode()
    fake_file(monkeypatch, raw)
    proposal = load_cut_proposal("not-opened-on-disk.json", source_target=TARGET)
    assert proposal.input_sha256 == hashlib.sha256(raw).hexdigest()
    assert proposal.to_dict() == document
    with pytest.raises(FrozenInstanceError):
        proposal.wing_m = 0.
    document["world_direction"][0] = 0.
    copy = proposal.to_dict()
    copy["provenance"]["sources"]["report"]["sha256"] = "b"*64
    assert proposal.world_direction == (1., 0., 0.)
    assert proposal.to_dict()["provenance"]["sources"]["report"]["sha256"] == "a"*64


@pytest.mark.parametrize("extra", ["target_world_m", "centre", "standoff_m", "margin_m", "skip_ik",
                                  "planning_permission", "mount", "proposals"])
def test_no_target_guard_or_authorization_overrides(document, extra):
    document[extra] = True
    with pytest.raises(ValueError):
        parse_cut_proposal(document, source_target=TARGET)


def test_exact_schema_and_single_object_required(document):
    for bad in ([document], [document, document], list(document.items()), {}, None):
        with pytest.raises(ValueError):
            parse_cut_proposal(bad, source_target=TARGET)
    document["schema"] = "unknown_v2"
    with pytest.raises(ValueError):
        parse_cut_proposal(document, source_target=TARGET)


def test_projection_limit_is_required_not_a_hidden_default(document):
    del document["maximum_projection_degrees"]
    with pytest.raises(ValueError):
        parse_cut_proposal(document, source_target=TARGET)


@pytest.mark.parametrize("field,bad", [
    ("world_direction", [0., 0., 0.]), ("world_direction", [2., 0., 0.]),
    ("world_direction", [True, 0., 0.]), ("world_direction", ["1", 0., 0.]),
    ("world_direction", [1., 0.]), ("world_direction", [float("nan"), 0., 0.]),
    ("normal_sign", True), ("normal_sign", 1.), ("normal_sign", 0),
    ("wing_m", float("inf")), ("wing_m", True),
    ("plane_tilt_degrees", 10.001), ("plane_tilt_degrees", -10.001),
    ("plane_tilt_degrees", float("nan")), ("plane_tilt_degrees", True),
    ("maximum_projection_degrees", 0.), ("maximum_projection_degrees", -1.),
    ("maximum_projection_degrees", 1.00001), ("maximum_projection_degrees", True),
    ("maximum_projection_degrees", float("inf")), ("source_target", "../SubStem_41"),
])
def test_invalid_proposals_fail_closed(document, field, bad):
    document[field] = bad
    with pytest.raises(ValueError):
        parse_cut_proposal(document, source_target=TARGET)


def test_target_match_checked_at_parse_and_resolution(document):
    with pytest.raises(ValueError):
        parse_cut_proposal(document, source_target="seed101_full/SubStem_42")
    proposal = parse_cut_proposal(document, source_target=TARGET)
    with pytest.raises(ValueError):
        resolve(proposal, source_target="seed101_full/SubStem_42")
    with pytest.raises(ValueError):
        resolve(None)


@pytest.mark.parametrize("bad", [0., -1., .01, float("nan"), True, WIDTH - .005])
def test_current_blade_width_not_historical_width_controls_wing(document, bad):
    proposal = parse_cut_proposal(document, source_target=TARGET)
    with pytest.raises(ValueError):
        resolve(proposal, edge_width_m=bad)


@pytest.mark.parametrize("sign", [-1, 1])
def test_exact_wing_boundary_retains_full_end_reserve(document, sign):
    document["wing_m"] = sign * (WIDTH/2 - .005)
    resolve(parse_cut_proposal(document, source_target=TARGET))
    document["wing_m"] += sign * 1e-8
    with pytest.raises(ValueError):
        resolve(parse_cut_proposal(document, source_target=TARGET))


def test_fresh_projection_is_small_explicit_measured_and_not_a_cached_axis(document):
    proposal = parse_cut_proposal(document, source_target=TARGET)
    assert resolve(proposal)["projection_degrees"] == 0
    angle = np.radians(.5)
    axis = np.array([np.sin(angle), 0., np.cos(angle)])
    result = resolve(proposal, axis)
    assert result["projection_degrees"] == pytest.approx(.5)
    assert np.dot(result["direction_world"], axis) == pytest.approx(0., abs=1e-15)
    assert np.linalg.norm(result["direction_world"]) == pytest.approx(1.)
    assert not result["path_screened"] and not result["planning_permission"]
    assert result["proposal_only"] and not result["physical_cut_verified"]
    assert not result["provenance_source_hashes_verified"]
    assert not {"centre", "target_world_m", "standoff_m", "margin_m"} & set(result)
    for degrees in (1.001, 30., 90.):
        axis = [np.sin(np.radians(degrees)), 0., np.cos(np.radians(degrees))]
        with pytest.raises(ValueError, match="projection"):
            resolve(proposal, axis)
    document["maximum_projection_degrees"] = result["projection_degrees"]
    # Equality fails; the declaration is a strict bound, not a relaxed tolerance.
    with pytest.raises(ValueError, match="projection"):
        resolve(parse_cut_proposal(document, source_target=TARGET), [np.sin(angle), 0., np.cos(angle)])


@pytest.mark.parametrize("axis", [[0., 0., 0.], [0., 0., 2.], [True, 0., 0.], [0., float("inf"), 1.]])
def test_invalid_fresh_axis_is_not_normalized_or_repaired(document, axis):
    with pytest.raises(ValueError):
        resolve(parse_cut_proposal(document, source_target=TARGET), axis)


@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("tilt", [-10., 0., 10.])
def test_normal_sign_and_plane_tilt_match_existing_knife_law(document, sign, tilt):
    document.update(normal_sign=sign, plane_tilt_degrees=tilt)
    result = resolve(parse_cut_proposal(document, source_target=TARGET))
    expected = cut_plane_normal(np.array([1., 0., 0.]), np.array([0., 0., sign]), tilt)
    np.testing.assert_allclose(result["plane_normal_world"], expected)
    assert np.dot(result["direction_world"], result["plane_normal_world"]) == pytest.approx(0.)


def test_reference_rotation_and_world_covariance(document):
    old = world_stroke_from_reference([1., 0., 0.], [0., 0., 1.],
                                     angle_degrees=150., maximum_projection_degrees=1.)
    np.testing.assert_allclose(old["world_direction"], [-np.sqrt(3)/2, .5, 0.], atol=1e-15)
    rotation = np.array([[.6, -.8, 0.], [0., 0., 1.], [-.8, -.6, 0.]])
    assert np.linalg.det(rotation) == pytest.approx(1.)
    other = world_stroke_from_reference(rotation @ [1., 0., 0.], rotation @ [0., 0., 1.],
                                       angle_degrees=150., maximum_projection_degrees=1.)
    np.testing.assert_allclose(other["world_direction"], rotation @ old["world_direction"], atol=1e-15)
    document["world_direction"] = list(other["world_direction"])
    result = resolve(parse_cut_proposal(document, source_target=TARGET), rotation @ [0., 0., 1.])
    np.testing.assert_allclose(result["direction_world"], other["world_direction"], atol=1e-15)


def test_native31_recorded_basis_preserves_world150_not_new_relative150():
    # Report SHA c006c3d92c98a6874582dc6ea214e05cbe0d5c0faa87d7992a4134788e600309.
    # Snapshot SHA 73e7a97b884dff2611fb3309dc81641a20fb6387793a94e9e07941280d140ea3.
    axis = np.array([.663798956482117, .6961114368749279, .2734955444338648])
    approach = np.array([-.7295447410722647, .6832002897238345, .03165177554372817])
    result = world_stroke_from_reference(approach, axis, angle_degrees=150., maximum_projection_degrees=1.)
    expected = [.5493769895582117, -.7019562507289886, .4532574824605304]
    np.testing.assert_allclose(result["world_direction"], expected, atol=1e-14, rtol=0)
    assert result["reference_projection_degrees"] == pytest.approx(.001769459435763187)
    a0 = np.array([.6637753812000651, .6961311379447973, .27350261807958975])
    next_approach = np.cos(np.radians(20))*approach + np.sin(np.radians(20))*np.cross(a0, approach)
    new_basis = world_stroke_from_reference(next_approach, axis, angle_degrees=0., maximum_projection_degrees=1.)
    b = np.array(new_basis["world_direction"])
    relative = np.degrees(np.arctan2(axis @ np.cross(b, expected), b @ expected))
    assert relative == pytest.approx(129.99999999027625, abs=1e-8)
    assert abs(relative - 150.) > 19.9  # Do not rotate both hands inadvertently.


@pytest.mark.parametrize("change", ["absent_source", "bad_hash", "extra_source_field", "empty_math", "nonfinite_math"])
def test_provenance_is_required_and_finite(document, change):
    p = document["provenance"]
    if change == "absent_source": del p["sources"]["snapshot"]
    if change == "bad_hash": p["sources"]["report"]["sha256"] = "not-a-hash"
    if change == "extra_source_field": p["sources"]["report"]["allow_cut"] = True
    if change == "empty_math": p["derivation"] = {}
    if change == "nonfinite_math": p["derivation"]["x"] = float("inf")
    with pytest.raises(ValueError):
        parse_cut_proposal(document, source_target=TARGET)


@pytest.mark.parametrize("raw", [b'{"schema":1,"schema":2}', b'{"x":NaN}', b'{"x":Infinity}',
                                 b'[]', b'null', b'not json', b'\xff', b' '* (MAX_JSON_BYTES+1)],
                         ids=["duplicate", "nan", "infinity", "array", "null", "syntax", "utf8", "oversize"])
def test_bad_or_unbounded_json_rejected(monkeypatch, raw):
    fake_file(monkeypatch, raw)
    with pytest.raises(ValueError):
        load_cut_proposal("unused.json", source_target=TARGET)
