"""Fresh finite capsule/box model for observed PAIRS, never native feature replay.

One generated normal point and one generated friction anchor per pair. The
caller supplies the original kc/dc ONCE PER PAIR, not once per native row:
this deliberately changes the aggregate interpretation when PCM has several
points. No normal/friction impulses, native points, normals or separations
enter the geometry or prediction. This module neither solves nor applies force.

Native rows only identify pairs. Their presence does not authenticate model
features, prove reporting completeness, or discover unobserved/new pairs.
"""
from collections import defaultdict
from numbers import Real

import numpy as np

from .capsule_box_witness import closest_capsule_box
from .contact_patch_prediction import GENERALIZED_MODEL
from .grasp_contact_geometry import _array, _snapshot, MAX_ROWS
from .shaft_grasp import ShaftCapsule, FingerPad

MODEL = 'fresh_grasp_capsule_box_one_point_per_pair_v1'
COVERAGE = 'geometrically_covered_not_native_feature_observations'
MAX_PATCHES = 16


def _pair_witness(start, end, half, radius):
    """Check A's finite local witness contract; never repair its output."""
    raw = closest_capsule_box(start, end, half, radius)
    axis = _array(raw['axis_point'], (3,), 'axis witness')
    box = _array(raw['box_point'], (3,), 'box witness')
    point = _array(raw['capsule_point'], (3,), 'capsule witness')
    normal = _array(raw['normal'], (3,), 'box-to-capsule normal')
    distance = float(_array(raw['distance_m'], (), 'axis distance'))
    gap = float(_array(raw['gap_m'], (), 'surface gap'))
    parameter = float(_array(raw['parameter'], (), 'segment parameter'))
    interval = _array(raw['minimizer_interval'], (2,), 'minimizer interval')
    if distance <= 0 or not 0 <= interval[0] <= parameter <= interval[1] <= 1:
        raise ValueError('Positive axis distance and bounded segment parameter required')
    # Floating arithmetic consistency checks, NOT geometric offsets or a
    # replacement for the helper's bounded, exact source-primitive algorithm.
    scale = max(float(np.max(abs(start))), float(np.max(abs(end))),
                float(np.max(half)), radius, distance)
    tolerance = max(1e-12, 128*np.finfo(float).eps*scale)
    errors = (
        np.linalg.norm(axis-((1-parameter)*start+parameter*end)),
        np.linalg.norm(axis-box-distance*normal),
        np.linalg.norm(point-(axis-radius*normal)),
        abs(gap-(distance-radius)),
        float(np.max(np.maximum(abs(box)-half, 0.))),
        float(np.min(abs(abs(box)-half))),
    )
    if abs(np.linalg.norm(normal)-1) > 1e-10 or max(errors) > tolerance:
        raise ValueError('Inconsistent source capsule/box witness; no normal or point repair')
    return dict(axis=axis, box=box, point=point, normal=normal, gap=gap,
        distance=distance, parameter=parameter, minimizer_interval=interval.tolist(),
        capsule_feature=str(raw['feature']['capsule']),
        box_feature=str(raw['feature']['box']), witness_model=str(raw['model']))


def _tangents(normal_local, box_rotation):
    """Deterministic source-box basis; circular friction is basis-invariant."""
    direction = np.eye(3)[int(np.argmin(abs(normal_local)))]
    first = np.cross(normal_local, direction)
    first /= np.linalg.norm(first)
    second = np.cross(normal_local, first)
    return np.stack((box_rotation@first, box_rotation@second))


def compile_contacts(chain, pads, *, source_target, all_plant_colliders,
                     reference, current, rows, mu):
    """Return full-floating J, g, s, report from current authored primitives.

reference/current bind pair-discovery generation and body inventories only;
no reference contact feature is transported. All geometry, COM Jacobians and
surface velocities come from current. n/g use the capsule and box witnesses.
The friction anchor is the COMMON capsule witness, so both tangent velocities
are evaluated at that same world point; no cached friction-position bias.

Pass report['feature_covered'] to the algebra solver, NOT feature_observed.
patches.observed is the historical solver field for geometric coverage here;
its semantics are explicitly COVERAGE, never native feature observations.
Full root columns are retained. Root conditioning, source hashes/epoch, exact
material values, coverage completeness and effort caps remain caller-owned.
"""
    chain = tuple(chain); pads = tuple(pads)
    if isinstance(all_plant_colliders, (str, bytes)):
        raise ValueError('Full plant collider inventory required')
    inventory_list = tuple(all_plant_colliders)
    if (not isinstance(source_target, str) or not source_target
            or not inventory_list or len(inventory_list) > 4096
            or any(not isinstance(p, str) or not p.startswith('/') for p in inventory_list)
            or len(set(inventory_list)) != len(inventory_list)):
        raise ValueError('Unique full source plant collider inventory required')
    inventory = set(inventory_list)
    if (not chain or len(pads) != 2
            or not all(isinstance(s, ShaftCapsule) for s in chain)
            or not all(isinstance(p, FingerPad) for p in pads)
            or len({s.collider for s in chain}) != len(chain)
            or len({s.body for s in chain}) != len(chain)
            or len({p.collider for p in pads}) != 2
            or not {s.collider for s in chain} <= inventory):
        raise ValueError('Complete original shaft and two pad-box geometries required')
    if isinstance(mu, (bool, np.bool_)) or not isinstance(mu, Real) or not np.isfinite(mu) or mu < 0:
        raise ValueError('Explicit finite nonnegative uniform source friction required')
    ref = _snapshot(reference, source_target); cur = _snapshot(current, source_target)
    if (current['step_id']-reference['step_id'] not in (0, 1)
            or ref['paths'] != cur['paths'] or ref['robot'] != cur['robot'] or ref['n'] != cur['n']):
        raise ValueError('Same inventories and current/one-step pair-generation binding required')
    if not set(cur['paths']) <= {s.body for s in chain}:
        raise ValueError('Dynamic body missing from complete source shaft chain')
    shafts = {s.collider: s for s in chain}; padmap = {p.collider: p for p in pads}
    discovered = defaultdict(lambda: dict(normal=0, friction=0, row_indices=[]))
    ignored = []
    for index, row in enumerate(rows):
        if index >= MAX_ROWS:
            raise ValueError('Native pair-discovery row overflow')
        # Intentionally access ONLY pair identity and kind. Even invalid or
        # inaccessible native geometry/impulse fields cannot influence features.
        a, b, kind = row['collider0'], row['collider1'], row['kind']
        if (not isinstance(a, str) or not isinstance(b, str) or a == b
                or not ({a, b} & inventory) or kind not in ('normal', 'friction')):
            raise ValueError('Explicit attributed plant normal/friction pair required')
        dynamic = [p for p in (a, b) if any(p == body or p.startswith(body+'/') for body in cur['paths'])]
        if not dynamic:
            ignored.append(dict(row=index, colliders=[a, b], reason='outside_dynamic_plant_system'))
            continue
        if len(dynamic) != 1 or dynamic[0] not in shafts:
            raise ValueError('Unsupported dynamic leaf/internal/nonshaft contact pair')
        cap = shafts[dynamic[0]]
        other = b if a == cap.collider else a
        if other not in padmap:
            raise ValueError('Unsupported dynamic shaft contact pair (knife/scene not modelled)')
        found = discovered[(cap.collider, other)]
        found[kind] += 1; found['row_indices'].append(index)
    if len(discovered) > MAX_PATCHES:
        raise ValueError('Observed pair bound exceeded')
    J = []; gaps = []; speeds = []; T = []; st = []; features = []; pairs = []
    for (cap_path, pad_path), counts in sorted(discovered.items()):
        cap = shafts[cap_path]; pad = padmap[pad_path]
        if cap.body not in cur['frames'] or pad.body not in cur['robot_frames']:
            raise ValueError('Source pair body missing from actual native inventory')
        bi = cur['paths'].index(cap.body); ri = cur['robot'].index(pad.body)
        body = cur['frames'][cap.body]; robot = cur['robot_frames'][pad.body]
        capsule = body@cap.local_frame; box_frame = robot@pad.local_frame
        R = box_frame[:3, :3]; centre = box_frame[:3, 3]
        start = R.T@(capsule[:3, 3]-cap.half_height_m*capsule[:3, 2]-centre)
        end = R.T@(capsule[:3, 3]+cap.half_height_m*capsule[:3, 2]-centre)
        try:
            w = _pair_witness(start, end, pad.half_extents_m, cap.radius_m)
        except (ValueError, KeyError, TypeError, FloatingPointError) as exc:
            raise ValueError('Fresh pair '+cap_path+' / '+pad_path+': '+str(exc)) from exc
        normal = R@w['normal']; point = R@w['point']+centre
        box_point = R@w['box']+centre; axis_point = R@w['axis']+centre
        tangents = _tangents(w['normal'], R)
        com = body[:3, 3]+body[:3, :3]@cur['com'][bi, :3]
        robot_com = robot[:3, 3]+robot[:3, :3]@cur['rc'][ri, :3]
        def point_jacobian(directions):
            return directions@cur['jac'][bi, :3]+np.cross(point-com, directions)@cur['jac'][bi, 3:]
        def surface_velocity(where):
            return cur['rv'][ri, :3]+np.cross(cur['rv'][ri, 3:], where-robot_com)
        j = point_jacobian(normal[None])[0]
        jt = point_jacobian(tangents)
        speed = float(normal@surface_velocity(box_point))
        tangent_speed = tangents@surface_velocity(point)
        if not all(np.isfinite(x).all() for x in (j, jt, speed, tangent_speed, point, box_point, normal)):
            raise ValueError('Nonfinite current source geometry/velocity mapping')
        i = len(J); J.append(j); gaps.append(w['gap']); speeds.append(speed)
        T.append([jt.tolist()]); st.append([tangent_speed.tolist()])
        pairs.append(dict(cap=cap_path, pad=pad_path, native_pair_observed=True,
            native_normal_row_count=counts['normal'], native_friction_row_count=counts['friction'],
            native_row_indices=list(counts['row_indices']), model_feature_count=1, model_anchor_count=1))
        features.append(dict(cap=cap_path, pad=pad_path, point_world_m=point.tolist(),
            capsule_witness_world_m=point.tolist(), box_witness_world_m=box_point.tolist(),
            axis_witness_world_m=axis_point.tolist(), normal_on_plant=normal.tolist(),
            tangent_basis_world=tangents.tolist(), gap_m=w['gap'], axis_box_distance_m=w['distance'],
            parameter=w['parameter'], minimizer_interval=w['minimizer_interval'],
            capsule_feature=w['capsule_feature'],
            box_feature=w['box_feature'], witness_model=w['witness_model'],
            native_pair_observed=True, feature_covered=True, feature_observed=False,
            features_generated=True, current_step=current['step_id']))
    count = len(J); n = cur['n']
    patches = dict(model=GENERALIZED_MODEL, normal_indices=[[i] for i in range(count)],
        tangent_jacobians=T, surface_speeds_m_s=st, mu=float(mu), observed=[True]*count)
    report = dict(geometry_model=MODEL, patches=patches, features=features, observed_pairs=pairs,
        feature_covered=[True]*count, feature_observed=[False]*count,
        patch_covered=[True]*count, features_generated=True, native_features_observed=False,
        coverage_semantics=COVERAGE, patches_observed_semantics=COVERAGE,
        contact_coefficient_interpretation='one unscaled original kc/dc per observed pair, not per native row',
        native_point_count_stiffness_parity=False, model_normal_points_per_pair=1,
        model_friction_anchors_per_pair=1, normal_surface_velocity_at='box_witness',
        friction_surface_velocity_at='common_capsule_witness',
        current_geometry_basis='current_source_primitives_not_cached_native_features',
        reference_step=reference['step_id'], current_step=current['step_id'], source_target=source_target,
        ignored_outside_dynamic_system=ignored, root_columns_retained=6,
        geometry_generation_binding='caller_asserted_pair_discovery_only',
        native_contact_geometry_read=False, current_native_body_frames_used=True,
        contact_impulses_read=False, friction_position_bias=False,
        native_contact_law_parity=False, native_completeness_verified=False,
        unobserved_pairs_enumerated=False, actuation_authorized=False, training_eligible=False)
    return np.array(J).reshape(count, n), np.array(gaps), np.array(speeds), report
