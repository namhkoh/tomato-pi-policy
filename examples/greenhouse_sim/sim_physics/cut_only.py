"""Fail-closed parked-left diagnostic; not a reachability or drop-safety proof."""
import numpy as np


def validate_profile(args):
    if not getattr(args,'right_only_cut_trial',False):
        return False
    if not (args.fixed_root_cut_trial and (args.branch_contact_fixture or getattr(args,'greenhouse_cut_trial',False))
            and args.native_startup_clearance and args.native_static_clearance
            and args.native_drives_after_cut and args.physics_hz==480
            and args.blade_force_feed and args.cut_style=='downward'
            and (args.cut_model,getattr(args,'knife_edge_mode','source_side_edge_v1')) in (
                ('signed_edge_load_brittle_seam_v1','source_side_edge_v1'),
                ('loaded_downward_lower_rim_seam_v1','source_lower_rim_v1'),
                ('loaded_downward_lower_rim_seam_v1','source_crossbar_edge_v1'))
            and args.bimanual_reposition_m==0 and not args.bimanual_hold_control
            and not args.cut_convergence_trial and not args.require_retention_screen
            and not args.physical_grasp_span and not args.settle_retention_preload
            and not args.effort_bounded_grasp_target and not args.preload_force_servo
            and not args.gui and not args.robot_interactive and not args.measured_withdrawal):
        raise ValueError('Cut-only requires explicit isolated 480 Hz native-release trial; no grasp/retention claims')
    return True


def parked_left(fixture, record, *, step, physics_hz):
    """Same-fetch readback, plus existing complete native guards at the caller.

    No grasp/held state is fabricated. A left contact, even well below the
    normal hard safety threshold, invalidates this intentionally unheld mode.
    """
    from .diagnostic_rate import frequency
    frequency(physics_hz)
    if (type(step) is not int or step<1 or abs(record['t']-step/physics_hz)>1e-9
            or record['contact'].get('step_id')!=step
            or record['contact'].get('adapter_valid') is not True
            or record['contact'].get('bilateral') is not False or record['slip_m'] is not None):
        raise RuntimeError('Fresh unheld cut-only observation required')
    q=np.asarray(fixture.robot.get_dof_positions(),float)
    targets=np.asarray(fixture.robot.get_dof_position_targets(),float)
    idx=fixture.left_indices
    expected=np.asarray(np.radians(fixture.initial_q),dtype=np.float32).astype(float)
    if (q.shape!=targets.shape or q.shape!=(1,len(fixture.names))
            or not np.isfinite(q).all() or not np.isfinite(targets).all()
            or not np.array_equal(targets,fixture.targets)
            or not np.allclose(targets[0,idx],expected,atol=1e-7,rtol=0)
            or np.max(np.abs(q[0,idx]-expected))>.005):
        raise RuntimeError('Left park reference changed or was not tracked; cut-only refused')
    fingers=np.asarray(q[0,fixture.finger_indices],float)*np.array([-1.,1.])
    loads=record['robot']['per_finger_contact_upper_bound_n']
    if (set(loads)!=set(fixture.paths[1:]) or not np.isfinite(list(loads.values())).all()
            or min(loads.values())<0 or max(loads.values())>.005
            or np.min(fingers)<fixture.pregrasp_half_aperture-.001):
        raise RuntimeError('Left hand is not open and unloaded; cut-only refused')
    return dict(model='native_unheld_left_park_v1',step_id=step,
        maximum_left_joint_error_rad=float(np.max(np.abs(q[0,idx]-expected))),
        minimum_finger_half_aperture_m=float(np.min(fingers)),
        maximum_left_finger_contact_n=float(max(loads.values())),
        left_grasp_verified=False,retention_expected=False,deposit_expected=False,
        dropped_material_expected=True,drop_corridor_certified=False,training_eligible=False)
