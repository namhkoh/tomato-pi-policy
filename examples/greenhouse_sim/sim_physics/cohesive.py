"""Isolated, uncalibrated engineering cohesive interface; no native/cut API.

Small-separation material law at one persistent, initially bonded patch. For
normal opening u = max(jump . normal, 0) and tangential jump s, define

    r = sqrt(u**2 + (Kt / Kn) * (s . s))                 [m]
    r0 = normal_strength / Kn; rf = 2 * Gc / normal_strength
    kappa = max(previous_kappa, r)
    psi = (1 - damage) * Kn * r**2 / 2                  [J/m^2]

The traction is the jump derivative of psi at fixed damage. Its radial envelope
is elastic to r0, then linear to zero at rf. Damage is irreversible; unloading
and reloading below kappa use the degraded elastic stiffness. Integrating the
damage dissipation gives D = Gc * (kappa - r0) / (rf - r0), clipped to [0, Gc].
This energy-norm specialization has ONE common fracture energy in all modes;
the pure-shear strength is normal_strength * sqrt(Kt / Kn), not an independent
fit. It is not a general BK law, tissue calibration, viscosity or fatigue model.

Background: bilinear mixed-mode damage laws and their solver limitations:
https://mooseframework.inl.gov/source/materials/cohesive_zone_model/BiLinearMixedModeTraction.html
(Camanho and Davila, NASA/TM-2002-211737). The restricted potential and analytical
energy accounting above specify THIS implementation, not the full cited model.

Integration contract (not implemented here): supply actual material-anchor
jumps relative to their stress-free reference, a consistently corotated unit
normal from face A toward B, and reference quadrature area. Persist states by
material patch, not ephemeral contact IDs; commit only converged/accepted trial
states. A bond can carry traction without a rigid-contact callback. ``bonded``
is INITIAL connectivity, never a per-step contact/eligibility switch. An
unbonded patch stays unbonded and fully damaged material never heals.

Compression is reported geometrically, but its contact reaction, energy and
friction are EXCLUDED. Pure compression cannot damage this interface; actual
tangential slip can cause mode-II/III damage even under compression. Do not
double-count cohesive shear as Coulomb friction or native normal-edge evidence.
Constitutive work below is analytical accounting, NOT measured actuator/contact
work; unsampled displacement excursions and native integration error are not
observable here. There is no force clipping, timer, loading-rate input, release
decision or default material calibration.

A continuous contact/deformation patch, separate nonpenetration/friction, and
replacement of the local unbreakable seam remain necessary. Neither this law
nor area partition invariance proves spatial/solver convergence, angular-momentum
balance of a future force application scheme, or a physical/robot cut. External
0.5 N / 1 mm guards and the existing 0.3 mm diagnostic are not changed here.

For a force-based native implicit bridge, facet elastic/secant stiffnesses are
area * retained_stiffness * Kn/Kt [N/m], at zero reference jump. Do not apply the
returned force AGAIN in parallel with that drive. This frozen-damage secant is
not the softening algorithmic tangent. A bilateral normal drive would add an
unmodeled compression penalty; external contact must handle that branch. Any
native damping, solver lag or force clipping changes the realized work and must
be accounted for independently, not silently identified with this law's D.
"""

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np


ENGINEERING_STATUS = "uncalibrated_engineering_diagnostic_only"


def _scalar(value, name, *, positive=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a real scalar")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} is outside the representable numeric range") from error
    if not math.isfinite(result) or (result <= 0 if positive else result < 0):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return result


def _vector(value, name):
    raw = np.asarray(value)
    if raw.shape != (3,) or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real 3-vector")
    result = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite real 3-vector")
    return result


@dataclass(frozen=True)
class CohesiveParameters:
    """Material inputs, all REQUIRED; stiffness is traction/jump, NOT N/m."""

    normal_stiffness_pa_m: float
    shear_stiffness_pa_m: float
    normal_strength_pa: float
    fracture_energy_j_m2: float

    def __post_init__(self):
        for name, value in vars(self).items():
            object.__setattr__(self, name, _scalar(value, name, positive=True))
        derived = (self.initiation_separation_m, self.final_separation_m,
                   self.shear_weight, self.shear_strength_pa,
                   self.normal_strength_pa * self.initiation_separation_m / 2)
        if not all(math.isfinite(v) and v > 0 for v in derived):
            raise ValueError("Material scales are outside the representable numeric range")
        if self.final_separation_m <= self.initiation_separation_m:
            raise ValueError("Gc must exceed elastic onset energy: strength^2 / (2 Kn)")

    @property
    def initiation_separation_m(self):
        return self.normal_strength_pa / self.normal_stiffness_pa_m

    @property
    def final_separation_m(self):
        return 2 * (self.fracture_energy_j_m2 / self.normal_strength_pa)

    @property
    def shear_weight(self):
        return math.sqrt(self.shear_stiffness_pa_m / self.normal_stiffness_pa_m)

    @property
    def shear_strength_pa(self):
        return self.normal_strength_pa * self.shear_weight


@dataclass(frozen=True)
class CohesiveState:
    """Immutable per-material-point history; parameters cannot change mid-path.

    Construct the virgin state with ``CohesiveState(parameters)``. Keeping area
    OUT of the history permits conservative subdivision of a uniform patch:
    copy the state to children and partition its reference area, never Gc.
    """

    parameters: CohesiveParameters
    bonded: bool = True
    maximum_effective_separation_m: float = 0.0
    effective_separation_m: float = 0.0

    def __post_init__(self):
        if not isinstance(self.parameters, CohesiveParameters) or type(self.bonded) is not bool:
            raise ValueError("Explicit material parameters and boolean initial connectivity required")
        for name in ("maximum_effective_separation_m", "effective_separation_m"):
            object.__setattr__(self, name, _scalar(getattr(self, name), name))
        if self.effective_separation_m > self.maximum_effective_separation_m:
            raise ValueError("Current effective separation exceeds its history maximum")
        if not self.bonded and self.maximum_effective_separation_m != 0:
            raise ValueError("An initially unbonded patch cannot contain cohesive history")
        if not math.isfinite(self.stored_energy_j_m2):
            raise ValueError("Stored energy is outside the representable numeric range")

    @property
    def retained_stiffness(self):
        p = self.parameters
        k = self.maximum_effective_separation_m
        if not self.bonded or k >= p.final_separation_m:
            return 0.0
        if k <= p.initiation_separation_m:
            return 1.0
        # Compute the small residual directly, not by subtracting damage from 1.
        return (p.initiation_separation_m / k
                * ((p.final_separation_m - k)
                   / (p.final_separation_m - p.initiation_separation_m)))

    @property
    def damage(self):
        return 1.0 - self.retained_stiffness

    @property
    def fully_separated(self):
        """Exact constitutive endpoint, not rounded damage==1 or release authority."""
        return self.bonded and self.maximum_effective_separation_m >= self.parameters.final_separation_m

    @property
    def stored_energy_j_m2(self):
        if self.retained_stiffness == 0:
            return 0.0
        r = self.effective_separation_m
        return (0.5 * self.retained_stiffness * self.parameters.normal_stiffness_pa_m * r) * r

    @property
    def dissipated_energy_j_m2(self):
        if not self.bonded:
            return 0.0  # Absent material did not consume fracture energy.
        p = self.parameters
        k = min(max(self.maximum_effective_separation_m, p.initiation_separation_m),
                p.final_separation_m)
        return p.fracture_energy_j_m2 * ((k - p.initiation_separation_m)
                                       / (p.final_separation_m - p.initiation_separation_m))


@dataclass(frozen=True)
class CohesiveResponse:
    state: CohesiveState
    area_m2: float
    traction_a_pa: tuple[float, float, float]
    force_a_n: tuple[float, float, float]
    compression_m: float
    stored_energy_j: float
    dissipated_energy_j: float
    dissipation_increment_j: float
    constitutive_work_increment_j: float

    @property
    def traction_b_pa(self):
        return tuple(-v for v in self.traction_a_pa)

    @property
    def force_b_n(self):
        return tuple(-v for v in self.force_a_n)

    def report(self):
        return dict(status=ENGINEERING_STATUS, material_calibrated=False,
                    physical_cut_verified=False, seam_release_authorized=False,
                    compression_response_included=False, friction_included=False,
                    work_is_constitutive_not_native_measurement=True,
                    bonded=self.state.bonded, fully_separated=self.state.fully_separated,
                    damage=self.state.damage, area_m2=self.area_m2,
                    stored_energy_j=self.stored_energy_j,
                    dissipated_energy_j=self.dissipated_energy_j,
                    dissipation_increment_j=self.dissipation_increment_j,
                    constitutive_work_increment_j=self.constitutive_work_increment_j)


def cohesive_response(state, separation_m, normal, *, area_m2):
    """Return a trial state and balanced forces; no input is mutated.

    ``separation_m`` is face-B displacement minus face-A displacement, relative
    to stress-free material anchors, in the SAME frame as ``normal``. The latter
    is a unit vector from A to B. A positive opening attracts A toward B. The
    reported work uses +traction_A . d(separation), whereas power delivered by
    the internal force pair to the bodies has the opposite sign.

    Area must be positive even for unbonded material. No-contact is NOT a reason
    to recreate/reset a bonded state; use bonded=False only for initially absent
    cohesive connectivity. Caller supplies a consistently transported frame for
    large rotations and separately solves contact, friction and equilibrium.
    """
    if not isinstance(state, CohesiveState):
        raise ValueError("A CohesiveState is required")
    area = _scalar(area_m2, "area_m2", positive=True)
    jump = _vector(separation_m, "separation_m")
    n = _vector(normal, "normal")
    length = math.hypot(*n)
    if not math.isclose(length, 1.0, rel_tol=1e-10, abs_tol=1e-12):
        raise ValueError("normal must be a unit vector, not an arbitrary direction")
    n /= length
    p = state.parameters
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            signed_opening = float(jump @ n)
            tangent = jump - signed_opening * n
            opening = max(0.0, signed_opening)
            if state.bonded:
                effective = math.hypot(opening, *(p.shear_weight * tangent))
                updated = CohesiveState(p, True,
                    max(state.maximum_effective_separation_m, effective), effective)
            else:
                updated = state
            retained = updated.retained_stiffness
            # Do not evaluate huge elastic tractions after complete separation.
            traction = (retained * (p.normal_stiffness_pa_m * opening * n
                        + p.shear_stiffness_pa_m * tangent)
                        if retained else np.zeros(3))
            force = area * traction
            stored = area * updated.stored_energy_j_m2
            dissipated = area * updated.dissipated_energy_j_m2
            delta_d = area * (updated.dissipated_energy_j_m2 - state.dissipated_energy_j_m2)
            work = area * (updated.stored_energy_j_m2 - state.stored_energy_j_m2) + delta_d
            if not np.isfinite(np.r_[traction, force, stored, dissipated, delta_d, work]).all():
                raise ValueError("Response is outside the representable numeric range")
    except (FloatingPointError, OverflowError) as error:
        raise ValueError("Response is outside the representable numeric range") from error
    return CohesiveResponse(updated, area, tuple(map(float, traction)), tuple(map(float, force)),
                            max(0.0, -signed_opening), stored, dissipated, delta_d, work)
