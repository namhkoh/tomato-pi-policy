"""Owned, bounded unit-conditioning/native-CFM diagnostic; never a robot run."""
import argparse
import json
from pathlib import Path
import time
import traceback

import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--length-units-per-m', type=int, choices=(1, 100), required=True)
    parser.add_argument('--free-control', action='store_true')
    parser.add_argument('--zero-cfm', action='store_true',
                        help='Explicit numerical-regularization comparison, no material/force-limit changes')
    args = parser.parse_args(argv)
    from .contact_spring_probe import from_report, bind, sample, assess_tail
    from .coupon_units import author_scaled, NativeSI, contacts_to_si, check_drive_contract
    from .host_memory import preflight
    from .file_integrity import sha256_file
    from sim_data.audit import DEFAULT_PACK
    out = args.output.resolve()
    if out.exists() or out.is_relative_to(DEFAULT_PACK.resolve()):
        raise ValueError('New diagnostic output outside the source package required')
    coupon = from_report(args.source_report, iterations=(128, 0), held_contacts=not args.free_control)
    result = dict(state='failed_unit_conditioning_coupon', memory=preflight(),
                  configuration=coupon.report(), training_eligible=False, production_qualified=False,
                  material_parameters_changed=False, zero_cfm_requested=args.zero_cfm)
    if not result['memory']['allowed']:
        print(json.dumps(result), flush=True)
        return 2
    out.mkdir(parents=True)
    from isaacsim import SimulationApp
    app = SimulationApp({'headless': True, 'multi_gpu': False, 'sync_loads': False})
    rows = []; monitor = None; sim = None; errors = None
    try:
        import omni.usd
        import omni.physics.tensors as tensors
        from omni.physx import get_physx_simulation_interface
        from pxr import Usd, UsdUtils, UsdGeom, PhysxSchema
        from .native_errors import NativeErrors
        from .contact_spring_native_probe import ContactRows, read_state, cleanup_native
        errors = NativeErrors()
        stage = Usd.Stage.CreateInMemory()
        data, result['unit_conversion'] = author_scaled(stage, coupon, args.length_units_per_m)
        result['cfm'] = []
        for path in data['body_paths']:
            attr = PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath(path)).GetCfmScaleAttr()
            before = attr.Get()
            if args.zero_cfm:
                attr.Set(0.)
            result['cfm'].append(dict(path=path, before=before, authored=attr.Get(), native_readback_available=False))
        # Copy the entire composed authored state for equality checks after parse.
        expected = {str(a.GetPath()): a.Get() for p in stage.Traverse() for a in p.GetAuthoredAttributes()}
        stage_id = UsdUtils.StageCache.Get().Insert(stage).ToLongInt()
        context = omni.usd.get_context(); attached = []
        if not context.attach_stage_with_callback(stage_id=stage_id,
                on_finish_fn=lambda ok, message: attached.append((ok, message))):
            raise RuntimeError('Stage attachment refused')
        deadline = time.monotonic()+30
        while not attached:
            if time.monotonic() > deadline or not app.is_running():
                raise RuntimeError('Stage attachment timeout')
            app.update()
        if attached[0][0] is not True or context.get_stage() != stage:
            raise RuntimeError('Stage attachment mismatch')
        sim = get_physx_simulation_interface()
        if not sim.attach_stage(stage_id) or sim.get_attached_stage() != stage_id:
            raise RuntimeError('Native stage ownership mismatch')
        monitor = ContactRows(robot_root=coupon.root, target_root=coupon.root, fingers=[], floor_root=None)
        monitor.subscribe(); monitor.begin_step()
        sim.simulate(1e-6, 0.); sim.fetch_results(); monitor.measurements(1e-6)
        errors.check('explicit_1us_bootstrap')
        result['bootstrap'] = dict(steps=1, dt_s=1e-6, no_reset=True,
                                   contacts_si=contacts_to_si(monitor.rows, args.length_units_per_m))
        if UsdGeom.GetStageMetersPerUnit(stage) != 1/args.length_units_per_m:
            raise RuntimeError('Native context changed unit metadata')
        for path, value in expected.items():
            # Pose/velocity attributes may be published by physics; everything
            # else, including inertia, scene, material and drives must match.
            if path.split('.')[-1].startswith('xformOp') or path.endswith((
                    '.physics:velocity', '.physics:angularVelocity',
                    '.state:angular:physics:position', '.state:angular:physics:velocity')):
                continue
            if stage.GetAttributeAtPath(path).Get() != value:
                raise RuntimeError('Parsed authored property changed: '+path)
        simulation_view = tensors.create_simulation_view('numpy', stage_id=stage_id)
        simulation_view.set_subspace_roots('/')
        raw = simulation_view.create_articulation_view(data['body_paths'][0])
        view = NativeSI(raw, args.length_units_per_m)
        result['binding'], unused = bind(view, coupon, model='native')
        if unused is not None:
            raise RuntimeError('No external predictor permitted in native comparison')
        frames, velocity, q, qdot = read_state(view, maximal=False)
        np.testing.assert_allclose(frames[:, :3, 3], data['frames'][:, :3, 3], atol=1e-6, rtol=0)
        np.testing.assert_allclose(frames[:, :3, :3], data['frames'][:, :3, :3], atol=1e-6, rtol=0)
        np.testing.assert_allclose(np.asarray(view.get_inertias()).reshape(3, 3, 3), coupon.inertias, rtol=2e-6, atol=1e-12)
        result['initial_state_si'] = dict(frames=frames.tolist(), velocities=velocity.tolist(), q=q.tolist(), qdot=qdot.tolist())
        result['native_inertia_and_geometry_si_verified'] = True
        result['native_drive_contract_si'] = check_drive_contract(view, coupon)
        result['native_drive_envelope_raw'] = np.asarray(raw.get_dof_drive_model_properties(), dtype=float).tolist()
        result['native_joint_friction_raw'] = np.asarray(raw.get_dof_friction_properties(), dtype=float).tolist()
        started = time.perf_counter()
        for step in range(1, 721):
            if not app.is_running():
                raise RuntimeError('Owned app closed early')
            monitor.begin_step(); errors.check('before_step')
            check_drive_contract(view, coupon)
            sim.simulate(coupon.dt, 1e-6+(step-1)*coupon.dt); sim.fetch_results()
            errors.check('after_step'); monitor.measurements(coupon.dt)
            frames, velocity, q, qdot = read_state(view, maximal=False)
            row = sample(coupon, step_id=step, frames=frames, velocities=velocity, q=q, qdot=qdot,
                         contact_rows=contacts_to_si(monitor.rows, args.length_units_per_m),
                         full_normal_friction_stream=monitor.native_full_contact_reporting, model='native')
            rows.append(row)
            if (np.max(abs(q)) >= .05 or max(row['per_body_contact_upper_bound_n']) > 1.
                    or np.max(np.linalg.norm(velocity[:, :3], axis=1)) > 1.):
                raise RuntimeError('Original coupon angle/speed/contact guard')
        result['wall_seconds_without_startup_or_export'] = time.perf_counter()-started
        result['assessment'] = assess_tail(rows[-121:], coupon, model='native', whole_run_samples=rows)
        errors.check('acceptance')
        if sha256_file(args.source_report) != coupon.source_sha256:
            raise RuntimeError('Source report changed during comparison')
        result['source_report_unchanged'] = True
        if result['assessment']['passed']:
            result['state'] = 'passed_unit_conditioning_coupon_NOT_plant'
    except Exception:
        result['error'] = traceback.format_exc()
    finally:
        from .contact_spring_native_probe import cleanup_native
        result['cleanup'] = cleanup_native(monitor, sim, errors)
        if result['cleanup']['failures']:
            result['state'] = 'failed_unit_conditioning_coupon'
        result['sample_count'] = len(rows)
        (out/'trace.json').write_text(json.dumps(rows, allow_nan=False), encoding='utf-8')
        (out/'report.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(dict(state=result['state'], sample_count=len(rows), error=result.get('error'),
                              assessment=result.get('assessment'))), flush=True)
        from .qualification_exit import exit_code
        code = exit_code(result, passed_state='passed_unit_conditioning_coupon_NOT_plant')
        app.close(exit_code=code)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
