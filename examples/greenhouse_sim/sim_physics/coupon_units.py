"""Dimensionally equivalent units for a NEW three-link coupon, never a live stage.

Only kg/metres versus kg/centimetres, native angular-drive comparison supported.
This is numerical conditioning evidence, not a production unit migration.
No input geometry/material/mass is tuned. SI conversion is explicit at readback.
"""
import numpy as np


def length_scale(value):
    if type(value) is not int or value not in (1, 100):
        raise ValueError('Explicit integer 1 or 100 length units per metre required')
    return value


def poses_to_si(value, scale):
    a = np.array(value, dtype=float, copy=True)
    if a.shape[-1:] != (7,) or not np.isfinite(a).all():
        raise ValueError('Finite XYZ/XYZW poses required')
    a[..., :3] /= length_scale(scale)
    return a


def velocities_to_si(value, scale):
    a = np.array(value, dtype=float, copy=True)
    if a.shape[-1:] != (6,) or not np.isfinite(a).all():
        raise ValueError('Finite linear/angular velocities required')
    a[..., :3] /= length_scale(scale)
    return a


def contacts_to_si(rows, scale):
    scale = length_scale(scale)
    result = []
    for row in rows:
        item = dict(row)
        for key in ('point_world_m', 'impulse_on_0_ns'):
            a = np.asarray(row[key], dtype=float)
            if a.shape != (3,) or not np.isfinite(a).all():
                raise ValueError('Finite original-order contact vector required')
            item[key] = (a/scale).tolist()
        if item.get('separation_m') is not None:
            if not np.isfinite(item['separation_m']):
                raise ValueError('Finite separation required')
            item['separation_m'] /= scale
        result.append(item)
    return result


class NativeSI:
    """Read-only view of a dimensionally scaled, native angular-drive coupon."""
    def __init__(self, view, scale):
        self.view = view
        self.scale = length_scale(scale)

    def __getattr__(self, name):
        if name in ('count', 'shared_metatype', 'link_paths'):
            return getattr(self.view, name)
        if name in ('get_dof_positions', 'get_dof_velocities', 'get_masses',
                    'get_drive_types', 'get_dof_velocity_targets', 'get_dof_position_targets'):
            return getattr(self.view, name)
        if name in ('get_dof_stiffnesses', 'get_dof_dampings', 'get_dof_max_forces',
                    'get_inertias', 'get_dof_projected_joint_forces', 'get_dof_actuation_forces'):
            return lambda: np.array(getattr(self.view, name)(), dtype=float, copy=True)/self.scale**2
        if name in ('get_link_transforms', 'get_coms'):
            return lambda: poses_to_si(getattr(self.view, name)(), self.scale)
        if name == 'get_link_velocities':
            return lambda: velocities_to_si(self.view.get_link_velocities(), self.scale)
        raise AttributeError('Unsupported or mutating native operation: '+name)


def check_drive_contract(view, coupon):
    """Read-only same-epoch spring/target/external-effort checks in SI units."""
    values = {}
    for method, expected in (
            ('get_drive_types', np.ones(2)),
            ('get_dof_stiffnesses', coupon.stiffness), ('get_dof_dampings', coupon.damping),
            ('get_dof_position_targets', np.zeros(2)), ('get_dof_velocity_targets', np.zeros(2)),
            ('get_dof_actuation_forces', np.zeros(2))):
        actual = np.asarray(getattr(view, method)(), dtype=float)
        if actual.shape != (1, 2) or not np.isfinite(actual).all():
            raise RuntimeError('Invalid native drive readback: '+method)
        if method in ('get_dof_stiffnesses', 'get_dof_dampings'):
            equal = np.allclose(actual[0], expected, rtol=2e-6, atol=1e-10)
        else:
            equal = np.array_equal(actual[0], expected)
        if not equal:
            raise RuntimeError('Native drive contract changed: '+method)
        values[method] = actual.tolist()
    return values


def author_scaled(stage, coupon, scale):
    """Own fresh empty stage required; caller must not have parsed/stepped it.

    Calls the existing strict coupon author before the dimensional conversion.
    Keeps FLT_MAX drive caps as the original practical-unbounded sentinel; its
    finite SI equivalent is disclosed, never used as an acceptance tolerance.
    """
    from pxr import Gf, UsdGeom, UsdPhysics
    from .contact_spring_probe import author
    scale = length_scale(scale)
    data = author(stage, coupon, model='native')
    before = []
    linear = {'physics:centerOfMass', 'physics:localPos0', 'physics:localPos1',
              'physics:velocity', 'physics:gravityMagnitude', 'radius', 'height', 'extent',
              'physxCollision:contactOffset', 'physxCollision:restOffset',
              'physxRigidBody:maxDepenetrationVelocity'}
    squared = {'physics:diagonalInertia', 'drive:angular:physics:stiffness',
               'drive:angular:physics:damping', 'physxRigidBody:sleepThreshold',
               'physxRigidBody:stabilizationThreshold', 'physxArticulation:sleepThreshold'}
    for prim in stage.Traverse():
        for attr in prim.GetAuthoredAttributes():
            name = attr.GetName()
            original = attr.Get()
            if name.startswith('xformOp:transform'):
                a = np.array(original, dtype=float)
                a[3, :3] *= scale
                if prim.IsA(UsdGeom.Cube):
                    a[:3, :3] *= scale
                attr.Set(Gf.Matrix4d(a.tolist()))
            elif name in linear or name.startswith('xformOp:translate'):
                if name == 'extent':
                    attr.Set([v*scale for v in original])
                else:
                    attr.Set(original*scale)
            elif name in squared:
                attr.Set(original*scale**2)
            else:
                continue
            before.append(dict(path=str(attr.GetPath()), original=str(original), scaled=str(attr.Get())))
    UsdGeom.SetStageMetersPerUnit(stage, 1/scale)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.)
    # No cross-layer composition/unit reconciliation: this is one owned layer.
    return data, dict(length_units_per_m=scale, kilograms_per_unit=1.,
                      changed_attributes=before, physical_parameters_tuned=False,
                      unbounded_drive_cap='original_float32_max_sentinel_in_native_units',
                      production_qualified=False)
