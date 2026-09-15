"""Geometry-preview tests; no native renderer or training data writes."""
import numpy as np
import pytest

from .plant_variant_preview import camera_basis, triangles_by_component


@pytest.mark.parametrize("azimuth,elevation", [(0, 0), (-50, 9), (120, -15), (90, 90)])
def test_orthographic_basis_preserves_metric_lengths(azimuth, elevation):
    basis = camera_basis(azimuth, elevation)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-14)
    assert np.linalg.det(basis) == pytest.approx(1.)
    np.testing.assert_allclose(np.linalg.norm(np.array([[1., 2., 3.]]) @ basis),
                               np.sqrt(14.))


def test_mesh_preview_uses_world_transforms_and_nearest_component_owner():
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, "/Plant")
    root.AddTranslateOp().Set((1., 2., 3.))
    branch = UsdGeom.Xform.Define(stage, "/Plant/Branch")
    branch.AddTranslateOp().Set((4., 0., 0.))
    mesh = UsdGeom.Mesh.Define(stage, "/Plant/Branch/Mesh")
    mesh.GetPointsAttr().Set([(0., 0., 0.), (1., 0., 0.), (0., 1., 0.)])
    mesh.GetFaceVertexCountsAttr().Set([3])
    mesh.GetFaceVertexIndicesAttr().Set([0, 1, 2])
    actual = triangles_by_component(stage, {"parent": "/Plant", "branch": "/Plant/Branch"})
    assert actual["parent"].shape == (0, 3, 3)
    np.testing.assert_array_equal(actual["branch"], [[[5., 2., 3.], [6., 2., 3.], [5., 3., 3.]]])
