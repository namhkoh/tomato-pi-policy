"""Offline contact-aware spring proposals from a recorded full-plant diagnostic.

Never imports SimulationApp, applies force, alters input evidence, or generates
training records. Proposed motion is counterfactual: the recorded native step
used a DIFFERENT spring effort and cannot be an accuracy target for this solve.
The geometry/model gates remain strict; unsupported contacts are reported.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from .contact_coupled_prediction import MaterialLaw
from .contact_patch_prediction import GENERALIZED_MODEL, solve_generalized
from .grasp_contact_geometry import compile_contacts


def predict(previous, current, binding, drive_parameters, *, geometry_model='cached_plane'):
    """Counterfactual pure solve; caller supplies source-authenticated geometry.

The support flag is an explicit recorded assertion, not inferred from velocity.
No root constraint is introduced after release. Native contact forces are not
inputs. No output of this function authorizes an actuation write.
"""
    ref, now = previous['before'], current['before']
    stream = previous['contacts']
    h = current['dt_s']
    if (type(h) not in (int, float) or not np.isfinite(h) or not 0 < h <= .02
            or previous['dt_s'] != h or previous['step_id'] != ref['step_id'] + 1
            or now['step_id'] != previous['step_id']
            or current['step_id'] != now['step_id'] + 1
            or stream['error'] is not None
            or stream['row_count'] != len(stream['rows'])
            or stream['plant_collider_paths'] != list(binding.plant_collider_paths)
            or ref['source_target'] != binding.source_target
            or now['source_target'] != binding.source_target
            or now['joint_names'] != drive_parameters['names']
            or ref['joint_names'] != now['joint_names']):
        raise ValueError('Adjacent complete source-bound native diagnostic required')
    if not np.array_equal(previous['after_generalized_velocity'], now['generalized_velocity']):
        raise ValueError('Native velocity changed between fetch and next snapshot')
    compliance = binding.finger_contact_compliance
    if not compliance or compliance.get('force_based') is not True:
        raise ValueError('Explicit original force-based finger compliance required')
    root = now['external_root_support_enabled_caller_asserted']
    if type(root) is not bool:
        raise ValueError('Explicit recorded external support state required')
    compiler = compile_contacts
    if geometry_model == 'fresh_box':
        from .grasp_contact_fresh import compile_contacts as compiler
    elif geometry_model != 'cached_plane':
        raise ValueError('Explicit supported diagnostic geometry model required')
    J, gaps, speed, geometry = compiler(binding.chain, binding.pads,
        source_target=binding.source_target, all_plant_colliders=binding.plant_collider_paths,
        reference=ref, current=now, rows=stream['rows'], mu=binding.finger_friction)
    d = len(now['joint_names']); n = d + 6
    def vector(value, length):
        a = np.asarray(value, dtype=float)
        if a.shape != (length,) or not np.isfinite(a).all():
            raise ValueError('Finite original dynamics vector required')
        return a
    K = vector(drive_parameters['stiffness'][0], d)
    C = vector(drive_parameters['damping'][0], d)
    caps = vector(drive_parameters['max_forces'][0], d)
    if np.any(caps <= 0): raise ValueError('Positive original actuator bounds required')
    q = vector(now['q_rad'], d)
    v = vector(now['generalized_velocity'], n)
    f = vector(now['native_known_noncontact_force'], n)
    M = np.asarray(now['mass_matrix'], dtype=float)
    if M.shape != (n, n): raise ValueError('Full floating mass matrix required')
    patches = geometry['patches']
    patches['model'] = GENERALIZED_MODEL
    root_momentum = np.zeros(d)
    if root:
        # Actual existing external support: conditional block, NOT a free-root
        # Schur complement or a new root weld. Native support parity unverified.
        # Preserve measured initial root momentum even though this conditional
        # model prescribes zero NEXT root velocity. This is not a new force.
        root_momentum = M[6:, :6] @ v[:6]
        f = f[6:] + root_momentum / h
        M = M[6:, 6:]; v = v[6:]; J = J[:, 6:]
        patches['tangent_jacobians'] = [np.asarray(a)[:, :, 6:].tolist()
                                      for a in patches['tangent_jacobians']]
    else:
        q = np.r_[np.zeros(6), q]
        K = np.r_[np.zeros(6), K]; C = np.r_[np.zeros(6), C]
    start = time.perf_counter()
    result = solve_generalized(M, q, v, K, C, J, gaps, speed,
        np.full(len(gaps), compliance['stiffness_n_m']),
        np.full(len(gaps), compliance['damping_n_s_m']), h, f,
        law=MaterialLaw('unilateral_kv_v1'), patches=patches,
        feature_observed=geometry['feature_covered'] if geometry_model=='fresh_box' else geometry['feature_observed'],
        root_dofs=0 if root else 6)
    elapsed = time.perf_counter() - start
    cap_pass = None
    if result['status'] == 'resolved':
        tau = vector(result['tau_joint'], d)
        cap_pass = bool(np.all(abs(tau) <= caps))
    return dict(result=result, geometry=geometry, solve_wall_s=elapsed,
        geometry_model=geometry_model,
        solver_coverage_semantics=('geometrically_covered_not_native_feature_observations'
            if geometry_model=='fresh_box' else 'validated_recorded_feature_geometry'),
        configured_force_caps_passed=cap_pass, original_finger_compliance=dict(compliance),
        initial_root_to_joint_momentum=root_momentum.tolist(),
        root_next_velocity_prescribed_zero=root,
        actual_external_support_response_verified=False,
        root_treatment='recorded_external_support_conditional_block' if root else 'all_six_free_root_coordinates',
        prediction_scope='counterfactual_not_executed_native_trajectory',
        native_velocity_prediction_error=None, native_contact_law_parity=False,
        measured_contact_forces_used=False, actuation_authorized=False,
        native_qualified=False, training_eligible=False)


def source_fixture(report):
    """Reconstruct the explicit source profile; no native simulation or motion.

This binds authored LOCAL geometry, not native cooking, contacts or current
world poses. The original greenhouse is not instantiated for offline algebra.
That must never be interpreted as qualification with surroundings removed.
"""
    from pxr import Gf, Usd, UsdGeom
    from sim_data.audit import DEFAULT_PACK, audit_manifest
    from sim_data.geometry import assemble_plant
    from .plant import build
    from .bimanual import BimanualRobot
    c = report['configuration']
    expected = dict(scene='package', plant='seed101_full', target='SubStem_41',
        constraint_mode='articulation', full_robot_probe=True, bimanual_cut=True,
        sparse_contacts=True, finger_gravity=True, compliant_fingers=True,
        torso_yaw=0., approach_side=1, force_newton=0., spring_mode='implicit_effort')
    for key, value in expected.items():
        # Legacy reports do not expose approach_side; only the fixed bimanual
        # fixture default is supported here, explicitly labelled in the result.
        if key == 'approach_side' and key not in c: continue
        if c.get(key) != value: raise ValueError('Unsupported authored source profile: ' + key)
    if report['source_assets_unchanged'] is not True:
        raise ValueError('Source report did not verify unchanged assets')
    manifest = DEFAULT_PACK/'plants/components'/c['plant']/'manifest.json'
    stage = Usd.Stage.CreateInMemory()
    UsdGeom.SetStageMetersPerUnit(stage, 1); UsdGeom.SetStageUpAxis(stage, 'Z')
    paths = assemble_plant(stage, '/World/Plant', audit_manifest(manifest))
    stage.SetEditTarget(stage.GetSessionLayer())
    UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant')).AddTranslateOp().Set(Gf.Vec3d(-.005, 0, .9))
    rig = build(stage, dict(manifest_path=str(manifest), component_paths=paths,
        plant_root='/World/Plant'), c['target'], max_segment_m=c['max_segment_m'], cut_m=c['cut_arc_m'])
    fixture = BimanualRobot(stage, rig, sparse_contacts=True, finger_gravity=True,
        compliant_fingers=True, friction=c['finger_friction'],
        finger_actuator_limit_n=c['finger_actuator_limit_n'], arc=c['grasp_arc_m'],
        grasp_roll=c['grasp_roll'], grasp_skew=c['grasp_skew'], approach_tilt=c['approach_tilt'],
        station_pose=c['station_pose'], station_offset=c['station_offset'] or (0., 0.),
        station_yaw=c['station_yaw'], approach_vector=c['approach_vector'] or (1., -1., .2),
        grasp_depth=c['grasp_depth_m'], approach_distance=c['approach_distance'],
        grasp_compression=c['grasp_compression_m'], torso_degrees=[0.]*6,
        ground_height=lambda x, y: .101, right_ik_fixed_joint=c['right_ik_fixed_joint'],
        cut_model=c['cut_model'], cut_standoff=c['cut_standoff_m'])
    if fixture.rig.source_target != report['rig']['source_target']:
        raise ValueError('Authored target differs from native report')
    return fixture


def bind_source(payload, trace, report):
    """Match exact shape/joint hash to recorded grasp binding, not native cooking.

An optional separate-run export is usable only for the IDENTICAL local binding
and recorded finger parameters. Runtime contacts, world poses and safety
receipts can never be transferred by this local-geometry check.
"""
    from .plant_contact_binding import (SCHEMA, AUTHORED_SCHEMA, deserialize,
        deserialize_authored, _native_binding, _hash)
    from .shaft_grasp_native import _sensor_contract_record, NATIVE37_SENSOR_CONTRACT
    contacts=[row['contact'] for row in trace]
    hashes={row['binding_sha256'] for row in contacts}
    target=report['rig']['source_target']
    if len(hashes)!=1 or any(row['source_target']!=target for row in contacts):
        raise ValueError('Trace source binding changed')
    expected=next(iter(hashes))
    if payload['schema']==SCHEMA:
        binding=deserialize(payload,source_target=target,binding_sha256=expected,sha256=payload['sha256'])
    elif payload['schema']==AUTHORED_SCHEMA:
        binding=deserialize_authored(payload,source_target=target,sha256=payload['sha256'])
        if any(row['allow_signed_native_normals'] is not True for row in contacts):
            raise ValueError('Authored replay supports only the recorded signed-normal binding')
        copied=dict(payload,native_binding_extras=dict(allow_signed_native_normals=True,
            sensor_contract=_sensor_contract_record(NATIVE37_SENSOR_CONTRACT)))
        if _hash(_native_binding(copied,binding.chain,binding.pads))!=expected:
            raise ValueError('Authored local shapes/joints differ from recorded native binding')
    else:raise ValueError('Unknown source geometry schema')
    if (dict(binding.finger_contact_compliance or {})!=report['robot_probe']['finger_contact_compliance']
            or binding.finger_friction!=report['configuration']['finger_friction']
            or binding.selected_body!=report['robot_probe']['grasp_body']
            or len(binding.chain)!=report['rig']['body_count']):
        raise ValueError('Export material/selected body differs from native source report')
    return binding


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binding-json', type=Path, help='Optional exact-local-binding export; never a motion receipt')
    parser.add_argument('--geometry-model', choices=('cached_plane','fresh_box'), default='cached_plane')
    parser.add_argument('--times', type=float, nargs='+', default=[2.8, 3.4, 4., 8., 11.85, 11.88, 11.89])
    args = parser.parse_args(argv)
    if not 1 <= len(args.times) <= 100 or not all(np.isfinite(t) and t >= 0 for t in args.times):
        raise ValueError('1..100 finite nonnegative sample times required')
    if args.output.exists(): raise ValueError('New diagnostic output directory required')
    report_path = args.source_run/'report.json'
    trace_path = args.source_run/'bimanual_trajectory.json'
    if trace_path.stat().st_size > 2_000_000_000: raise ValueError('Trace exceeds bounded reader size')
    report = json.loads(report_path.read_text(encoding='utf-8'))
    trace = json.loads(trace_path.read_text(encoding='utf-8'))
    if not isinstance(trace, list) or len(trace) < 2: raise ValueError('Adjacent trace samples required')
    binding_path=args.binding_json or args.source_run/'plant_contact_binding.json'
    if binding_path.exists():payload=json.loads(binding_path.read_text(encoding='utf-8'))
    elif args.binding_json is not None:raise ValueError('Requested source binding file is missing')
    else:
        from .plant_contact_binding import capture_authored
        payload=capture_authored(source_fixture(report))
    binding=bind_source(payload,trace,report)
    if any(t < trace[0]['t'] or t > trace[-2]['t'] for t in args.times):
        raise ValueError('Requested time lacks a following native snapshot')
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'source_geometry.json').write_text(json.dumps(payload, allow_nan=False, indent=2), encoding='utf-8')
    results = []
    for t in args.times:
        i = min(range(len(trace)-1), key=lambda i: abs(trace[i]['t']-t))
        row = dict(requested_t=t, generation_t=trace[i]['t'], current_t=trace[i+1]['t'])
        try:
            row.update(predict(trace[i]['contact_prediction'], trace[i+1]['contact_prediction'],
                               binding, report['native_drive_parameters'], geometry_model=args.geometry_model))
            row['status'] = row['result']['status']
            if row['status']=='resolved' and row['configured_force_caps_passed'] is not True:
                row['status']='rejected_force_cap'
        except (ValueError, KeyError, TypeError, RuntimeError) as exc:
            row.update(status='rejected', error=type(exc).__name__+': '+str(exc))
        results.append(row)
        print(json.dumps({k:row[k] for k in ('current_t', 'status', 'error', 'solve_wall_s') if k in row}), flush=True)
    def digest(path):
        with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
    summary = dict(source_run=str(args.source_run.resolve()), report_sha256=digest(report_path),
        trace_sha256=digest(trace_path), source_geometry_sha256=binding.sha256,
        source_geometry_schema=binding.schema, geometry_model=args.geometry_model,
        native_cooked_geometry_verified=False, results=results,
        actuation_authorized=False, training_eligible=False)
    (args.output/'shadow.json').write_text(json.dumps(summary, allow_nan=False, indent=2), encoding='utf-8')
    return 0 if all(r['status']=='resolved' for r in results) else 1


if __name__ == '__main__': raise SystemExit(main())
