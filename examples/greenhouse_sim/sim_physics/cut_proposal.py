"""One opt-in world-stroke proposal, never a plan or a cut authorization.

No simulator, scene mutation, default proposal, search, target position, station,
mount, standoff, contact exemption or margin override is provided here. The
caller owns the fresh measured stem axis AND centre. Only the direction is
projected onto that transverse plane, with an explicitly declared STRICT angle
limit (at most one degree); excess change fails rather than silently rotating.

Integration: load_cut_proposal(None, ...) returns None, preserving the caller's
existing grid. Otherwise resolve_cut_proposal returns ONE orientation/wing for
the existing full rigid stroke -> IK -> self/scene -> transit checks. Use the
fresh native centre, current blade size and original stroke offsets. None of
these helpers certifies any path, native grasp, contact, release or physical cut.

The schema has exactly the fields in PROPOSAL_KEYS. Provenance is inert evidence:
it must declare report/snapshot paths and SHA256s plus derivation metadata, but
loading does NOT open those referenced files or claim their hashes were checked.
The input proposal's own byte hash is recorded. Coordinates inside provenance
are never interpreted as targets or runtime transforms.

Units: world_direction and stem_axis_world are unit vectors in the same world
frame; wing_m is metres; plane_tilt_degrees is limited to the existing +/-10
degrees. The current blade's half edge width minus the existing 5 mm end reserve
bounds wing_m. Only roundoff-sized unit-vector normalization is allowed.
"""

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import re

import numpy as np

from .knife import cut_plane_normal


SCHEMA = "single_world_cut_proposal_v1"
MAX_JSON_BYTES = 65536
PROPOSAL_KEYS = frozenset(("schema", "source_target", "world_direction", "normal_sign",
    "wing_m", "plane_tilt_degrees", "maximum_projection_degrees", "provenance"))


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} is outside numeric range") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit(value, name):
    if not isinstance(value, (list, tuple, np.ndarray)) or np.shape(value) != (3,):
        raise ValueError(f"{name} must be a real unit 3-vector")
    vector = np.array([_number(v, name) for v in value])
    length = math.hypot(*vector)
    if not math.isclose(length, 1., rel_tol=0., abs_tol=1e-10):
        raise ValueError(f"{name} must be unit length; no direction repair")
    return vector / length


def _target(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_][\w.-]*/[A-Za-z0-9_][\w.-]*", value):
        raise ValueError("source_target must explicitly identify family/component")
    return value


def _limit(value):
    angle = _number(value, "maximum_projection_degrees")
    if not 0 < angle <= 1.:
        raise ValueError("Explicit projection limit must be positive and at most 1 degree")
    return angle


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key: " + key)
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError("Nonfinite JSON number: " + value)


def _json(text):
    return json.loads(text, object_pairs_hook=_unique_pairs, parse_constant=_bad_constant)


def _provenance(value):
    if (not isinstance(value, dict) or set(value) != {"sources", "derivation"}
            or not isinstance(value["sources"], dict)
            or not {"report", "snapshot"} <= set(value["sources"])
            or not isinstance(value["derivation"], dict) or not value["derivation"]):
        raise ValueError("Provenance requires source report/snapshot bindings and explicit derivation")
    for source in value["sources"].values():
        if (not isinstance(source, dict) or set(source) != {"path", "sha256"}
                or not isinstance(source["path"], str) or not source["path"].strip()
                or not isinstance(source["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", source["sha256"])):
            raise ValueError("Every provenance source needs a path and lowercase SHA256")
    # Also rejects nested Infinity produced by a JSON exponent such as 1e999.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class CutProposal:
    source_target: str
    world_direction: tuple[float, float, float]
    normal_sign: int
    wing_m: float
    plane_tilt_degrees: float
    maximum_projection_degrees: float
    provenance_json: str
    input_sha256: str | None = None

    def __post_init__(self):
        _target(self.source_target)
        object.__setattr__(self, "world_direction", tuple(map(float, _unit(self.world_direction, "world_direction"))))
        if type(self.normal_sign) is not int or self.normal_sign not in (-1, 1):
            raise ValueError("normal_sign must be integer -1 or 1, not bool")
        object.__setattr__(self, "wing_m", _number(self.wing_m, "wing_m"))
        tilt = _number(self.plane_tilt_degrees, "plane_tilt_degrees")
        if abs(tilt) > 10.:
            raise ValueError("Plane tilt must preserve the existing +/-10 degree gate")
        object.__setattr__(self, "plane_tilt_degrees", tilt)
        object.__setattr__(self, "maximum_projection_degrees", _limit(self.maximum_projection_degrees))
        if not isinstance(self.provenance_json, str):
            raise ValueError("provenance_json must be serialized evidence")
        object.__setattr__(self, "provenance_json", _provenance(_json(self.provenance_json)))
        if self.input_sha256 is not None and (not isinstance(self.input_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", self.input_sha256)):
            raise ValueError("Invalid input proposal SHA256")

    def to_dict(self):
        return dict(schema=SCHEMA, source_target=self.source_target,
                    world_direction=list(self.world_direction), normal_sign=self.normal_sign,
                    wing_m=self.wing_m, plane_tilt_degrees=self.plane_tilt_degrees,
                    maximum_projection_degrees=self.maximum_projection_degrees,
                    provenance=_json(self.provenance_json))


def parse_cut_proposal(document, *, source_target):
    """Validate one exact-schema object and its caller-selected target."""
    expected = _target(source_target)
    if not isinstance(document, dict) or set(document) != PROPOSAL_KEYS or document["schema"] != SCHEMA:
        raise ValueError("Exactly one single_world_cut_proposal_v1 object is required")
    if document["source_target"] != expected:
        raise ValueError("Proposal source_target differs from the current target")
    return CutProposal(document["source_target"], document["world_direction"], document["normal_sign"],
                       document["wing_m"], document["plane_tilt_degrees"],
                       document["maximum_projection_degrees"], _provenance(document["provenance"]))


def load_cut_proposal(path, *, source_target):
    """Optional bounded read; referenced provenance paths are NOT opened."""
    _target(source_target)
    if path is None:
        return None
    with Path(path).open("rb") as stream:
        raw = stream.read(MAX_JSON_BYTES + 1)
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Single-proposal JSON exceeds 64 KiB")
    proposal = parse_cut_proposal(_json(raw.decode("utf-8")), source_target=source_target)
    return CutProposal(proposal.source_target, proposal.world_direction, proposal.normal_sign,
                       proposal.wing_m, proposal.plane_tilt_degrees, proposal.maximum_projection_degrees,
                       proposal.provenance_json, hashlib.sha256(raw).hexdigest())


def _project(direction, axis, limit):
    tangent = direction - axis * float(direction @ axis)
    length = math.hypot(*tangent)
    change = math.degrees(math.atan2(abs(float(direction @ axis)), length))
    if change >= limit or length <= 1e-12:
        raise ValueError(f"World direction needs {change:.12g} degrees projection; required < {limit:.12g}")
    return tangent / length, change


def world_stroke_from_reference(approach_world, stem_axis_world, *, angle_degrees, maximum_projection_degrees):
    """Reproduce the planner's transverse rotation; no FK, search or scene read.

    approach_world is the ORIGINAL planner direction (-commanded left goal Z),
    not the parked right wrist/blade direction. A measured-left-FK direction
    may be supplied for a separately labelled cross-check, never silently
    substituted for the original command. Both projection and azimuth are
    explicitly supplied; the returned projection angle belongs in provenance.
    """
    approach = _unit(approach_world, "approach_world")
    axis = _unit(stem_axis_world, "stem_axis_world")
    angle = _number(angle_degrees, "angle_degrees")
    if abs(angle) > 180.:
        raise ValueError("Reference azimuth must be within +/-180 degrees")
    tangent, change = _project(approach, axis, _limit(maximum_projection_degrees))
    radians = math.radians(angle)
    direction = tangent * math.cos(radians) + np.cross(axis, tangent) * math.sin(radians)
    return dict(world_direction=tuple(map(float, _unit(direction, "derived world_direction"))),
                reference_projection_degrees=change)


def resolve_cut_proposal(proposal, *, source_target, stem_axis_world, edge_width_m):
    """Resolve ONE proposal against fresh axis/current knife, with no position."""
    if not isinstance(proposal, CutProposal):
        raise ValueError("An explicitly loaded/parsed CutProposal is required")
    if proposal.source_target != _target(source_target):
        raise ValueError("Proposal source_target differs from the current target")
    axis = _unit(stem_axis_world, "fresh stem_axis_world")
    width = _number(edge_width_m, "current edge_width_m")
    if width <= .01 or abs(proposal.wing_m) > width / 2 - .005:
        raise ValueError("Wing must retain 5 mm from the CURRENT knife edge end")
    direction, change = _project(np.array(proposal.world_direction), axis, proposal.maximum_projection_degrees)
    normal = cut_plane_normal(direction, proposal.normal_sign * axis, proposal.plane_tilt_degrees)
    return dict(source_target=proposal.source_target, direction_world=tuple(map(float, direction)),
                plane_normal_world=tuple(map(float, normal)), normal_sign=proposal.normal_sign,
                wing_m=proposal.wing_m, plane_tilt_degrees=proposal.plane_tilt_degrees,
                projection_degrees=change, maximum_projection_degrees=proposal.maximum_projection_degrees,
                input_sha256=proposal.input_sha256, proposal_only=True,
                path_screened=False, planning_permission=False, physical_cut_verified=False,
                provenance_source_hashes_verified=False)
