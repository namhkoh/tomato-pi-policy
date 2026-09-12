"""Independent frame-based beam-strain diagnostics, not tissue damage criteria.

Child-frame joint axes match plant._joint's authored rest transforms. Only
internal beam joints are included; the separable root interface is not a beam.
Curvature, surface strain and quadratic energy are engineering small-strain
estimates, not calibrated fracture evidence or permission to execute a cut.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def _frames(value, count):
    a = np.asarray(value, dtype=float)
    if (a.shape != (count, 4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[:, 3], [0, 0, 0, 1], atol=1e-7, rtol=0)
            or not np.allclose(a[:, :3, :3].transpose(0, 2, 1)@a[:, :3, :3], np.eye(3), atol=1e-5, rtol=0)
            or not np.allclose(np.linalg.det(a[:, :3, :3]), 1, atol=1e-5, rtol=0)):
        raise ValueError('Complete ordered finite rigid body frames required')
    return a


class RodStrain:
    def __init__(self, rest_frames, lengths_m, radii_m, stiffness_nm_rad, cut_index):
        lengths = np.array(lengths_m, dtype=float, copy=True)
        n = len(lengths)
        if (lengths.shape != (n,) or not 2 <= n <= 256 or type(cut_index) is not int
                or not 0 <= cut_index < n-1):
            raise ValueError('Bounded beam chain and nonempty internal elastic joint set required')
        radii = np.array(radii_m, dtype=float, copy=True)
        stiffness = np.array(stiffness_nm_rad, dtype=float, copy=True)
        if (radii.shape != (n,) or stiffness.shape != (n, 3)
                or any(not np.isfinite(x).all() or np.any(x <= 0) for x in (lengths, radii, stiffness))):
            raise ValueError('Positive finite SI length/radius/stiffness arrays required')
        rest = _frames(rest_frames, n).copy()
        self.n = n
        self.indices = np.arange(cut_index+1, n)
        self.parent_rest_rot = rest[self.indices-1, :3, :3].transpose(0, 2, 1)@rest[self.indices, :3, :3]
        self.lengths = lengths[self.indices]
        self.radii = radii[self.indices]
        self.stiffness = stiffness[self.indices]

    def evaluate(self, frames):
        now = _frames(frames, self.n)
        parent_joint = now[self.indices-1, :3, :3]@self.parent_rest_rot
        child_joint = now[self.indices, :3, :3]
        relative = parent_joint.transpose(0, 2, 1)@child_joint
        angle = Rotation.from_matrix(relative).as_rotvec()
        curvature = angle/self.lengths[:, None]
        bending = self.radii*np.linalg.norm(curvature[:, :2], axis=1)
        torsion = self.radii*np.abs(curvature[:, 2])
        angles = np.linalg.norm(angle, axis=1)
        return dict(model='frame_based_small_strain_rod_diagnostic_v1',
                    internal_child_indices=self.indices.tolist(),
                    relative_rotation_vectors_rad=angle.tolist(),
                    bending_surface_strain_estimate=bending.tolist(),
                    torsional_surface_shear_estimate=torsion.tolist(),
                    maximum_bending_surface_strain_estimate=float(bending.max()),
                    maximum_torsional_surface_shear_estimate=float(torsion.max()),
                    maximum_relative_rotation_rad=float(angles.max()),
                    principal_rotation_branch_clear=bool(angles.max() < np.pi-1e-4),
                    quadratic_elastic_energy_estimate_j=float(.5*np.sum(self.stiffness*angle**2)),
                    native_joint_coordinate_used=False, tissue_damage_calibrated=False,
                    contact_work_or_fracture_energy=False, execution_authority=False)
