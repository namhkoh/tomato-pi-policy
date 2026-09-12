"""Isolated native-torsion / implicit-bending numerical comparison.

The full original K/C participate in the frozen-M predictor. Only bending
efforts are submitted; original torsional drives remain inside the native
contact solve. This is operator splitting, NOT a jointly converged contact
solver or a calibrated material model. No native torsional drive work is
observable through the actuation-command tensor. No root actuation is added.
"""
import re
import numpy as np

from .implicit_springs import ImplicitJointSprings


class SplitJointSprings(ImplicitJointSprings):
    def __init__(self, articulation):
        self.articulation = articulation
        if articulation.count != 1:
            raise ValueError('Exactly one source plant articulation required')
        self.names = tuple(articulation.shared_metatype.dof_names)
        n = len(self.names)
        # plant._joint authors child-frame rotX, rotY, rotZ in this exact order;
        # local Z is the cylinder/beam axis. Reject any other inventory.
        if not n or n % 3 or n > 765 or len(set(self.names)) != n:
            raise ValueError('Complete bounded plant spherical joint triples required')
        for i in range(0, n, 3):
            match = re.fullmatch(r'(Joint_[0-9]+):0', self.names[i])
            if not match or self.names[i:i+3] != tuple(match[1]+':'+str(j) for j in range(3)):
                raise ValueError('Ordered source joint rotX/rotY/rotZ triples required')
        self.fixed_base = articulation.shared_metatype.fixed_base
        if self.fixed_base:
            raise ValueError('Floating source articulation with external support required')
        self.k = self._read('get_dof_stiffnesses')
        self.c = self._read('get_dof_dampings')
        if np.any(self.k <= 0) or np.any(self.c < 0):
            raise ValueError('Original positive native K and nonnegative C required')
        self.torsion = np.arange(n) % 3 == 2
        self.native_k = np.where(self.torsion, self.k, 0.)
        self.native_c = np.where(self.torsion, self.c, 0.)
        self.work_stiffness = np.where(self.torsion, 0., self.k)
        self.indices = np.array([0], dtype=np.uint32)
        self.caps = self._read('get_dof_max_forces')
        if np.any(self.caps <= 0):
            raise ValueError('Original positive native drive caps required')
        self._targets_and_types()
        if np.any(self._read('get_dof_actuation_forces')):
            raise ValueError('No preexisting external plant actuation allowed')
        articulation.set_dof_stiffnesses(self.native_k[None].astype(np.float32), self.indices)
        articulation.set_dof_dampings(self.native_c[None].astype(np.float32), self.indices)
        self.check_contract()
        self.receipt = dict(mode='native_torsion_implicit_bending_isolated_v1',
            native_torsion_dof_names=[n for n, keep in zip(self.names, self.torsion) if keep],
            original_k_c_preserved=True, predictor_includes_full_k_c=True,
            explicit_torsional_effort_applied=False, root_actuation_applied=False,
            work_scope='explicit_bending_only_native_torsion_work_unmeasured',
            contact_spring_joint_solve_qualified=False, production_qualified=False,
            training_eligible=False)

    def _read(self, method):
        value = np.asarray(getattr(self.articulation, method)(), dtype=float)
        if value.shape != (1, len(self.names)) or not np.isfinite(value).all():
            raise ValueError('Complete finite native readback required: '+method)
        return value[0].copy()

    def _targets_and_types(self):
        for name in ('get_dof_position_targets', 'get_dof_velocity_targets'):
            if np.any(self._read(name)):
                raise RuntimeError('Zero rest targets required: '+name)
        if np.any(self._read('get_drive_types') != 1):
            raise RuntimeError('Native force drives required; acceleration drives unsupported')

    def check_contract(self):
        if tuple(self.articulation.shared_metatype.dof_names) != self.names:
            raise RuntimeError('Source joint inventory changed')
        for method, expected in (('get_dof_stiffnesses', self.native_k),
                                 ('get_dof_dampings', self.native_c),
                                 ('get_dof_max_forces', self.caps)):
            if not np.array_equal(self._read(method), expected):
                raise RuntimeError('Split spring native contract changed: '+method)
        self._targets_and_types()
        if np.any(self._read('get_dof_actuation_forces')[self.torsion]):
            raise RuntimeError('Native torsion must not be actuated twice')

    def step(self, dt, external_joint_force=None, *, root_constrained=False):
        if (isinstance(dt, (bool, np.bool_)) or not np.isfinite(dt)
                or abs(dt-1/240) > 1e-12 or type(root_constrained) is not bool
                or external_joint_force is not None):
            raise ValueError('Isolated 240 Hz no-external-force comparison required')
        self.check_contract()
        return super().step(dt, root_constrained=root_constrained)

    def _submit(self, effort):
        # Keep torsion in the predictor but never in the effort command.
        command = np.where(self.torsion, 0., effort)
        return super()._submit(command)
