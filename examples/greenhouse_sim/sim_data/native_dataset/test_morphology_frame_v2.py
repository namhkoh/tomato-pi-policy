"""Pure-CPU frame/version/hold tests; no asset mutation or calibration claims."""
from copy import deepcopy
import json

import numpy as np
import pytest

from . import morphology_frame_v2 as m


def geometry():
    donor = [[.01, 0, 0, .002], [.01000001, 0, .03, .0019],
             [.06, .01, .08, .0015], [.16, .04, .13, .001]]
    return dict(reference_petiole=donor, parent=[[0, 0, -.1, .006], [0, 0, .3, .005]],
        current_petiole=deepcopy(donor), leaves=[
            dict(attachment=[.06, .01, .08], centroid=[.12, .07, .09], axis=[0, 1, 0],
                 covariance=np.diag([.001, .002, .0001]).tolist()),
            dict(attachment=[.12, .03, .11], centroid=[.15, -.08, .12], axis=[0, -1, 0],
                 covariance=np.diag([.002, .001, .0002]).tolist())])


def transform(q, rotation, scale=1., translation=(0, 0, 0)):
    result = {}
    for k in ("reference_petiole", "parent", "current_petiole"):
        chain = np.array(q[k])
        result[k] = np.column_stack((chain[:, :3] @ rotation.T * scale + translation,
                                    chain[:, 3] * scale)).tolist()
    result["leaves"] = [dict(attachment=np.array(l["attachment"]) @ rotation.T * scale + translation,
        centroid=np.array(l["centroid"]) @ rotation.T * scale + translation,
        axis=np.array(l["axis"]) @ rotation.T,
        covariance=rotation @ np.array(l["covariance"]) @ rotation.T * scale ** 2) for l in q["leaves"]]
    return result


def ready(q):
    result = m.describe(q)
    assert result["state"] == "descriptor_available_not_calibrated", result
    assert not result["holds"]
    return result


def test_versioned_same_output_replay_and_source_geometry_unchanged():
    q = geometry()
    before = deepcopy(q)
    a = ready(q)
    assert a == ready(q) and q == before
    assert a["descriptor"]["schema"] == m.SCHEMA
    assert a["frame"]["version"] == m.FRAME_VERSION
    assert a["frame"]["donor_witness_index"] == 3
    assert a["frame"]["parent_segment_index"] == 0
    basis = np.array(a["frame"]["basis_columns"])
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-14)
    assert np.linalg.det(basis) == pytest.approx(1.)
    assert a["conditioning"]["formal_certificate"] is False
    assert a["conditioning"]["linear_feature_error_estimate"] <= 1e-9
    assert "eigensolver" in a["conditioning"]["scope"]
    assert a["numerical_nuisance_tolerance"] == 1e-9
    for flag in ("qualified_geometry", "training_approved", "calibration_validated",
                 "source_cap_reset", "global_inventory_complete"):
        assert a[flag] is False
    json.dumps(a, allow_nan=False)


def test_generated_geometry_cannot_select_different_donor_frame():
    q = geometry()
    a = ready(q)
    q["current_petiole"][2][0] += .025
    q["leaves"][0]["centroid"][1] += .02
    b = ready(q)
    assert a["frame"] == b["frame"]
    assert a["frame_reference_sha256"] == b["frame_reference_sha256"]
    assert a["input_geometry_sha256"] != b["input_geometry_sha256"]
    assert m.distance(a["descriptor"], b["descriptor"]) > .01


def test_proper_rigid_scale_name_and_leaf_order_nuisances_keep_1e_9():
    q = geometry()
    a = ready(q)
    rng = np.random.default_rng(284)
    for _ in range(8):
        rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        if np.linalg.det(rotation) < 0:
            rotation[:, 0] *= -1
        for scale in (.4, 1., 100.):
            bq = transform(q, rotation, scale, [1, 2, 3])
            bq["name"] = "irrelevant"
            bq["leaves"].reverse()
            for leaf in bq["leaves"]:
                leaf["name"] = "renamed"
            b = ready(bq)
            assert m.equivalent(a["descriptor"], b["descriptor"], 1e-9)
            assert m.distance(a["descriptor"], b["descriptor"]) <= 1e-9
            assert a["frame"]["donor_witness_index"] == b["frame"]["donor_witness_index"]


@pytest.mark.parametrize("case,reason", [
    ("collinear", "degenerate_or_unresolved_donor_transverse"),
    ("tied", "ambiguous_maximum_donor_transverse"),
    ("near_tied", "ambiguous_maximum_donor_transverse"),
    ("parent_tied", "ambiguous_nearest_parent_segment"),
    ("weak", "conditioning_estimate_exceeds_numerical_tolerance"),
    ("large_offset", "conditioning_estimate_exceeds_numerical_tolerance"),
    ("zero_segment", "degenerate_chain"),
])
def test_degenerate_ambiguous_and_ill_conditioned_frames_hold_without_fallback(case, reason):
    q = geometry()
    if case == "collinear":
        q["reference_petiole"] = [[.01, 0, 0, .002], [.01, 0, .1, .001]]
    elif case in ("tied", "near_tied"):
        bump = 1e-16 if case == "near_tied" else 0
        q["reference_petiole"] = [[.01, 0, 0, .002], [.11, 0, .1, .0015], [.01, .1 + bump, .2, .001]]
    elif case == "parent_tied":
        q["parent"] = [[-.09, 0, -.2, .005], [-.09, 0, .2, .005],
                       [.11, 0, .2, .005], [.11, 0, -.2, .005]]
    elif case == "weak":
        q["reference_petiole"] = [[.01, 0, 0, .002], [.0100000001, 0, .1, .001]]
    elif case == "large_offset":
        q = transform(q, np.eye(3), translation=[1e6, 1e6, 1e6])
    else:
        q["reference_petiole"][1] = q["reference_petiole"][0][:]
    result = m.describe(q)
    assert result["state"] == "held"
    assert result["descriptor"] is None and "descriptor_sha256" not in result
    assert result["holds"][0]["reason"] == reason
    assert result["numerical_nuisance_tolerance"] == 1e-9


@pytest.mark.parametrize("damage", ["missing", "shape", "nan", "negative_radius", "zero_axis", "non_psd", "asymmetric"])
def test_malformed_geometry_does_not_yield_a_descriptor(damage):
    q = geometry()
    if damage == "missing":
        del q["parent"]
    elif damage == "shape":
        q["reference_petiole"] = [[1, 2, 3]]
    elif damage == "nan":
        q["leaves"][0]["centroid"][0] = float("nan")
    elif damage == "negative_radius":
        q["reference_petiole"][0][3] = -1
    elif damage == "zero_axis":
        q["leaves"][0]["axis"] = [0, 0, 0]
    elif damage == "non_psd":
        q["leaves"][0]["covariance"][0][0] = -1.
    else:
        q["leaves"][0]["covariance"][0][1] = .1
    with pytest.raises(ValueError):
        m.describe(q)


def test_v1_or_tampered_schema_cannot_reuse_v2_comparison():
    a = ready(geometry())["descriptor"]
    for changes in (dict(schema="greenhouse.normalized_morphology_context.v1"),
                    dict(frame_version="old_frame"), dict(leaf_count=True)):
        b = {**a, **changes}
        with pytest.raises(ValueError):
            m.distance(a, b)
        with pytest.raises(ValueError):
            m.equivalent(a, b, 1e-9)


def test_unordered_leaf_matching_is_bipartite_not_greedy():
    a = ready(geometry())["descriptor"]
    a["leaves"] = [np.zeros(15).tolist(), np.zeros(15).tolist()]
    b = deepcopy(a)
    a["leaves"][0][0] = .1
    b["leaves"][1][0] = .2
    assert m.equivalent(a, b, .11)
    assert m.distance(a, b) == pytest.approx(.1)
    b["leaves"].reverse()
    assert m.distance(a, b) == pytest.approx(.1)
    b["leaves"][0][0] = .4
    assert not m.equivalent(a, b, .11)
    assert m.distance(a, b) == pytest.approx(.3)


def test_empty_and_different_leaf_cardinality_are_explicit():
    q = geometry()
    a = ready(q)["descriptor"]
    q["leaves"] = []
    b = ready(q)["descriptor"]
    assert m.distance(b, b) == 0
    assert m.equivalent(b, b, 0)
    with pytest.raises(ValueError, match="outside equivalence domain"):
        m.distance(a, b)
    with pytest.raises(ValueError, match="outside equivalence domain"):
        m.equivalent(a, b, 1e-9)


@pytest.mark.parametrize("tolerance", [True, -1, float("nan"), float("inf"), None])
def test_no_invalid_or_implicit_comparison_threshold(tolerance):
    a = ready(geometry())["descriptor"]
    with pytest.raises(ValueError):
        m.equivalent(a, a, tolerance)


def test_results_are_detached():
    q = geometry()
    a = ready(q)
    a["descriptor"]["fixed"][0] = 12345
    a["frame"]["basis_columns"][0][0] = 12345
    assert ready(q)["descriptor"]["fixed"][0] != 12345
