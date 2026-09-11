"""Pure, bounded gravity-load diagnostic; no USD, native solver or activation API.

The caller selects and orders the distal bodies explicitly. Frames, seam, axis
and gravity share ONE Cartesian world frame; frames map body-local coordinates
to world using column vectors. Local COMs are in the body frame, not principal
inertia axes. The unit seam axis points from proximal support toward distal
material, so a positive axial gravity component is a tensile load demand.

This computes external gravity loading about a seam, NOT its actual reaction.
The circular intact elastic screen uses N/A + |M_b|/Z, Z = pi*r^3/4, and a
linear normal-traction distribution across a full disk. Compression can offset
bending tension in that approximation. It is not a unilateral contact solution,
facet quadrature, mixed-mode constitutive evaluation, equilibrium solution,
damage evolution, ultimate strength prediction or permission to activate.
No material, gravity or body-selection defaults are supplied.
"""
import math
from numbers import Real

import numpy as np


MAX_BODIES = 4096  # Work/allocation bound, not a physical or material limit.
RIGID_FRAME_ATOL = 1e-6
UNIT_AXIS_ATOL = 1e-6


def _array(value, name, shape):
    try:
        # Reject oversized outer sequences before NumPy allocates their copy.
        outer_bound = MAX_BODIES if shape[0] is None else shape[0]
        if isinstance(value, (list, tuple)) and len(value) > outer_bound:
            raise ValueError(f"{name} exceeds its explicit body/shape bound")
        raw = np.asarray(value)
        if (raw.ndim != len(shape)
                or any(want is not None and got != want
                       for got, want in zip(raw.shape, shape))
                or raw.size > 16 * MAX_BODIES or raw.dtype.kind not in "iuf"):
            raise ValueError(f"{name} has invalid shape, size or real numeric dtype")
        if isinstance(value, (list, tuple)) and any(
                isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat):
            raise ValueError(f"{name} must not contain booleans")
        with np.errstate(over="raise", invalid="raise", under="raise"):
            result = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError, FloatingPointError) as error:
        raise ValueError(f"{name} must be a bounded finite real array of shape {shape}") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _positive_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive real scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{name} must be representable as float64") from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return np.float64(result)


def gravity_seam_loads(*, native_frames, masses_kg, local_coms_m, seam_world_m,
                       unit_axis_world, radius_m, normal_strength_pa,
                       gravity_world_m_s2):
    """Return a JSON-compatible advisory dict from explicit, same-snapshot inputs.

    Required shapes: frames (N,4,4), masses (N,), local COMs (N,3), other
    vectors (3,), with 1 <= N <= MAX_BODIES and strictly positive body masses.
    No broadcasting, body discovery, source reads or input mutation occurs.
    Proper rotations are checked with fixed float32-compatible tolerances;
    scale, shear, reflection and projective transforms are rejected. Rotations
    are not repaired. An axis already unit within UNIT_AXIS_ATOL is normalized
    for roundoff only; an arbitrary direction is not accepted as a unit axis.

    Finite but unrepresentable derived arithmetic raises ValueError, rather
    than returning a nonfinite JSON value or a silently underflowed screen.
    Input validity does not authenticate native provenance or common timing.
    """
    frames = _array(native_frames, "native_frames", (None, 4, 4))
    count = len(frames)
    if not 1 <= count <= MAX_BODIES:
        raise ValueError(f"Body count must be in 1..{MAX_BODIES}")
    masses = _array(masses_kg, "masses_kg", (count,))
    coms = _array(local_coms_m, "local_coms_m", (count, 3))
    seam = _array(seam_world_m, "seam_world_m", (3,))
    axis = _array(unit_axis_world, "unit_axis_world", (3,))
    gravity = _array(gravity_world_m_s2, "gravity_world_m_s2", (3,))
    radius = _positive_scalar(radius_m, "radius_m")
    strength = _positive_scalar(normal_strength_pa, "normal_strength_pa")
    if np.any(masses <= 0):
        raise ValueError("Every selected body mass must be positive")
    if not np.all(frames[:, 3, :] == np.array([0., 0., 0., 1.])):
        raise ValueError("Frames must have the homogeneous row [0,0,0,1]")
    rotation = frames[:, :3, :3]
    # Bound entries before products/determinants to reject huge invalid frames.
    if np.any(np.abs(rotation) > 1 + RIGID_FRAME_ATOL):
        raise ValueError("Frames must contain proper rigid rotations")
    if (not np.allclose(rotation.transpose(0, 2, 1) @ rotation, np.eye(3),
                        atol=RIGID_FRAME_ATOL, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1.,
                               atol=RIGID_FRAME_ATOL, rtol=0)):
        raise ValueError("Frames must contain proper rigid rotations")
    axis_norm = math.hypot(*axis)
    if not math.isfinite(axis_norm) or abs(axis_norm - 1) > UNIT_AXIS_ATOL:
        raise ValueError("unit_axis_world must already have unit length")
    axis = axis / axis_norm

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="raise"):
            total_mass = math.fsum(masses)
            # Subtract the seam first to avoid an unnecessary large-world COM
            # subtraction. Uniform gravity makes cross(sum(m*r), g) exact in
            # real arithmetic; fsum reduces cancellation in the first moment.
            levers = frames[:, :3, 3] - seam + np.einsum("nij,nj->ni", rotation, coms)
            weighted = masses[:, None] * levers
            first_moment = np.array([math.fsum(weighted[:, j]) for j in range(3)])
            com_offset = first_moment / total_mass
            combined_com = seam + com_offset
            force = total_mass * gravity
            moment = np.cross(first_moment, gravity)
            axial = np.dot(force, axis)
            transverse = force - axial * axis
            torsion = np.dot(moment, axis)
            bending = moment - torsion * axis
            bending_norm = math.hypot(*bending)
            transverse_norm = math.hypot(*transverse)
            area = math.pi * radius**2
            second_moment = math.pi * radius**4 / 4
            section_modulus = second_moment / radius
            tensile_onset_force = strength * area
            bending_onset_moment = strength * section_modulus
            axial_stress = axial / area
            bending_stress = bending_norm / section_modulus
            peak_tension = max(0., axial_stress + bending_stress)
            utilization = peak_tension / strength
    except (OverflowError, FloatingPointError, ZeroDivisionError, ValueError) as error:
        raise ValueError("Derived seam loads/stresses exceed representable float64 arithmetic") from error
    scalars = dict(total_mass_kg=total_mass, axial_force_n=axial,
                   transverse_force_n=transverse_norm, torsional_moment_nm=torsion,
                   bending_moment_nm=bending_norm)
    screen = dict(radius_m=radius, normal_strength_pa=strength, area_m2=area,
                  second_moment_m4=second_moment, section_modulus_m3=section_modulus,
                  uniform_tensile_onset_force_n=tensile_onset_force,
                  pure_bending_first_tensile_onset_moment_nm=bending_onset_moment,
                  axial_stress_pa=axial_stress, bending_stress_pa=bending_stress,
                  maximum_tensile_stress_pa=peak_tension,
                  first_tensile_damage_utilization=utilization)
    vectors = dict(seam_world_m=seam, unit_axis_world=axis,
                   gravity_world_m_s2=gravity, combined_com_world_m=combined_com,
                   seam_to_com_world_m=com_offset, gravity_force_world_n=force,
                   gravity_moment_world_nm=moment, transverse_force_world_n=transverse,
                   bending_moment_world_nm=bending)
    if (not all(math.isfinite(x) for x in (*scalars.values(), *screen.values()))
            or not all(np.isfinite(x).all() for x in vectors.values())
            or min(area, second_moment, section_modulus, tensile_onset_force, bending_onset_moment) <= 0):
        raise ValueError("Derived seam loads/stresses must remain finite and representable")
    return dict(
        schema="greenhouse.seam_gravity_loads.v1", body_count=count,
        method="uniform_gravity_wrench_about_explicit_seam_from_body_local_COMs",
        **{k: float(v) for k, v in scalars.items()},
        **{k: v.tolist() for k, v in vectors.items()},
        circular_intact_elastic_screen=dict(
            method="full_disk_linear_normal_traction_max(0,N/A+abs(M_b)/(I/r))",
            **{k: float(v) for k, v in screen.items()},
            at_or_above_first_tensile_damage=bool(utilization >= 1),
            shear_or_torsional_failure_screened=False,
            compression_contact_solution=False, ultimate_capacity_prediction=False),
        provenance=dict(
            body_selection="explicit_caller_supplied_order_no_discovery",
            body_selection_verified=False, common_snapshot_verified=False,
            native_frames_masses_COMs_verified=False, source_hashes_verified=False,
            proximal_to_distal_axis_orientation_verified=False,
            unit_axis_roundoff_normalized=bool(axis_norm != 1.)),
        scope=dict(
            gravity_only=True, actual_seam_reaction=False,
            gripper_reaction_excluded=True, other_contact_reactions_excluded=True,
            inertial_reaction_excluded=True, other_external_loads_excluded=True,
            equilibrium_verified=False, material_calibration_verified=False,
            constitutive_success=False, activation_permission=False,
            physical_cut_verified=False))
