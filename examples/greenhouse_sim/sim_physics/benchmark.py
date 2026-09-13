"""Bounded real-PhysX qualification, not dataset collection or robot cutting.

Uses an isolated plant by default; --scene package retains the supplied building,
plants, v1.2 static robot and head/wrist cameras. Never takes over a running Kit.
"""
import argparse
from dataclasses import asdict
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import time
import traceback
from .file_integrity import sha256_file


def report_configuration(args,output):
    """Keep failure reports serializable when optional CLI paths are supplied."""
    result={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    result['output']=str(output)
    json.dumps(result,allow_nan=False)  # Fail before native work, not during publication.
    return result


def report_exit_code(report,args):
    """Use the requested milestone for both Python return and native Kit exit.

    A limited cut action may pass while final withdrawal fails; its separate
    execution/contact/support gates must still be present and true. Never
    upgrade a full-sequence request or accept a success label by itself.
    """
    if getattr(args,'through_stroke_trial',False):
        if (report.get('full_forward_cut_stroke_verified') is not True
                or report.get('full_sequence_qualified') is not True):return 2
    if getattr(args,'cut_action_trial',False):
        action=report.get('cut_action') or {}
        gates=action.get('gates') or {}
        required={'no_execution_fault','checked_right_plan','blade_contact_release',
                  'required_pre_cut_support','post_cut_observation','native_guards'}
        mode='right_only' if args.right_only_cut_trial else 'bimanual'
        passed=(report.get('state')=='passed_cut_action_not_complete_robot_task'
            and report.get('error') is None and report.get('source_assets_unchanged') is True
            and report.get('cut_strategy')==mode
            and report.get('left_grasp_verified') is (mode=='bimanual')
            and action.get('model')=='ground_truth_cut_action_milestone_v1'
            and action.get('strategy')==mode and action.get('passed') is True
            and set(gates)==required and all(value is True for value in gates.values())
            and all(report.get('gates',{}).get(k) is True for k in
                    ('bounded','collision_clear_right_plan','blade_contact_release')))
        return 0 if passed else 2
    if report.get('state')=='passed_cut_only_mechanism_not_robot_task':
        gates=report.get('gates') or {}
        required={'bounded','completed','collision_clear_right_plan','blade_contact_release',
                  'released_material_separates','right_withdrawal_completed','left_parked_open_unloaded'}
        passed=(getattr(args,'right_only_cut_trial',False)
            and report.get('error') is None and report.get('source_assets_unchanged') is True
            and report.get('cut_strategy')=='right_only' and report.get('left_grasp_verified') is False
            and set(gates)==required and all(value is True for value in gates.values()))
        return 0 if passed else 2
    if getattr(args,'right_only_cut_trial',False):
        return 2  # A requested unheld trial cannot inherit a grasp-mode success.
    return 0 if report.get('state') in (
        'passed_mechanism_qualification_not_robot_task','interactive_demo_not_qualification',
        'passed_gripper_mechanism_not_robot_task','passed_bimanual_mechanism_not_robot_task',
        'interactive_full_robot_diagnostic','scene_ablation_diagnostic_not_qualification') else 2


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scene',choices=('isolated','package'),default='isolated')
    p.add_argument('--isolate-station',action='store_true',
        help='Matched package station with complete source plant/floor/robot, no surroundings; NOT greenhouse qualification')
    p.add_argument('--greenhouse-cut-trial',action='store_true',help='Explicit intact greenhouse diagnostic with all original target-plant components and guarded preview context')
    p.add_argument('--plant',default='seed101_full')
    p.add_argument('--target',default='SubStem_41')
    p.add_argument('--target-row-slot',type=int,choices=(0,12,23),default=12,
        help='Swap target with an original end-row backdrop; retain all plants and original spacing')
    p.add_argument('--physics-hz',type=int,choices=(120,240,480,1920),default=240)
    p.add_argument('--cut-convergence-trial',action='store_true',
        help='Default-OFF isolated native-release comparison at480 Hz; same SI material/force limits and controller dwell durations, not production qualification')
    p.add_argument('--native-station-park-reference',action='store_true',
        help='Isolated endpoint check: evaluate unchanged right park joint angles in the actual measured station, not an absolute configured-world pose')
    p.add_argument('--solver',choices=('TGS','PGS'),default='TGS')
    p.add_argument('--uniform-solver-iterations',type=int,nargs=2,
        help='Isolated HOLD convergence comparison on every robot/plant body and articulation; default unchanged')
    p.add_argument('--solve-articulation-contact-last',action='store_true',
        help='Opt-in pre-parse solver-order comparison; physical properties and guards unchanged')
    p.add_argument('--gravity',type=float,choices=(0.,9.81),default=9.81)
    p.add_argument('--constraint-mode',choices=('articulation','maximal','fixed_articulation'),default='articulation')
    p.add_argument('--attached-only',action='store_true')
    p.add_argument('--spring-mode',choices=('native','implicit_effort'),default='native')
    p.add_argument('--native-torsion-trial',action='store_true',
        help='Isolated comparison: original native torsional drives, coupled implicit bending; not production qualified')
    p.add_argument('--force-newton',type=float,default=.02)
    p.add_argument('--physics-threads',type=int,choices=(1,2,4,8,16))
    p.add_argument('--fabric',action='store_true')
    p.add_argument('--render-hz',type=int,choices=(0,15,30,60),default=0)
    p.add_argument('--seconds',type=float,default=6.)
    p.add_argument('--max-segment-m',type=float,default=.025)
    p.add_argument('--stem-contact-model',choices=('flush_capsules_v1','continuous_internal_capsules_v1','flat_cylinders_v1'),
        default='flush_capsules_v1',help='Explicit continuity comparison; internal capsules reach joint anchors, no seam crossing')
    p.add_argument('--gui',action='store_true')
    p.add_argument('--interactive',action='store_true',help='Keep an isolated physics demo open with pull/release/reset controls')
    p.add_argument('--gripper-probe',action='store_true',help='Bounded actual left-gripper contact fixture, not full-arm IK')
    p.add_argument('--finger-friction',type=float,default=.5)
    p.add_argument('--grasp-arc-m',type=float,default=.12)
    p.add_argument('--exact-grasp-arc',action='store_true',
        help='Use the requested material arc instead of snapping to a segment centre; bimanual diagnostic only')
    p.add_argument('--cut-arc-m',type=float,default=.01,
        help='Explicit diagnostic seam within agreed 10..20 mm petiole interval; original 10 mm default unchanged')
    p.add_argument('--blade-axial-aim-offset-m',type=float,default=0.,
        help='Downward contact proposal: -1.5..+2.5 mm about SAME seam; >1.5 mm also requires source-body/stump and 3 mm section checks')
    p.add_argument('--diagnostic-detach',action='store_true')
    p.add_argument('--full-robot-probe',action='store_true',help='Full dynamic v1.2 robot with an IK-driven left arm')
    p.add_argument('--bimanual-cut',action='store_true',help='Guarded native left grasp and original right knife seam-release qualification')
    p.add_argument('--right-only-cut-trial',action='store_true',
        help='Separate isolated unheld cut diagnostic: left stays parked/open, no retention or deposit credit')
    p.add_argument('--cut-action-trial',action='store_true',
        help='Limited grasp/cut milestone; record released-target torso landings separately, keep full-task gates visible')
    p.add_argument('--watch-cut-trial',action='store_true',
        help='Visible one-shot observer for the same cut-action trial; no interactive reset/replay or physical parameter changes')
    p.add_argument('--knife-alignment',choices=('legacy','camera'),default='legacy',
        help='Camera aligns the arc to the actual wrist camera radial side; original source asset untouched')
    p.add_argument('--cut-style',choices=('legacy','downward'),default='legacy',
        help='Downward: extended arm, gravity-aligned stroke within measured angular limits and straight Cartesian approach; no detour fallback')
    p.add_argument('--right-ready-degrees',type=float,nargs=7,
        help='Explicit initial right pose for a coordinated downward fixture; screened before physics, never a runtime teleport')
    p.add_argument('--right-ready-lift-m',type=float,default=0.,
        help='Raise the initial right waiting pose 0..50 mm using IK; preserve orientation and all native/path checks')
    p.add_argument('--right-ready-retreat-m',type=float,default=0.,
        help='Withdraw the initial right waiting pose toward the wrist along its +Z axis, 0..50 mm; all checks remain')
    p.add_argument('--left-ik-seed-degrees',type=float,nargs=7,
        help='Seed the coordinated left pregrasp IK; exact target and path guards still apply')
    p.add_argument('--blade-force-feed',action='store_true',
        help='Isolated downward diagnostic: retime the screened stroke from fresh native cutting load')
    p.add_argument('--blade-dwell-feedback',action='store_true',
        help='Isolated compliant feed comparison: regulate the minimum of seven raw loads; release gates remain unsmoothed')
    p.add_argument('--compliant-blade-rate',action='store_true',
        help='Isolated 1000 N/m contact comparison: loading <=0.3 mm/s, no force/dwell/clearance changes')
    p.add_argument('--blade-friction-budget',action='store_true',
        help='Isolated feed comparison: 0.40 N full-contact backoff, unchanged 0.50 N hard guard and raw cut gates')
    p.add_argument('--seam-contact-compliance',action='store_true',
        help='Isolated blade feedback experiment: uncalibrated 1000 N/m local stem contact compression')
    p.add_argument('--seam-contact-yield',action='store_true',
        help='Experimental isolated crossbar process-zone softening from qualified measured loaded advance; uncalibrated')
    p.add_argument('--through-stroke-trial',action='store_true',
        help='Explicit measured post-release follow-through; no collision/force bypass, can fail on solid cut faces')
    p.add_argument('--postrelease-feed-m-s',type=float,default=.0003,
        help='Experimental post-release contact feed cap 0.0003..0.002 m/s; original force limits remain')
    p.add_argument('--postcut-egress-trial',action='store_true',
        help='Reobserve after unloaded reverse and screen a separate withdrawal goal; no clearance allowance')
    p.add_argument('--material-clearance-trial',action='store_true',
        help='Experimental measured sharp-edge section clearance and bounded post-release cut-face sliding')
    p.add_argument('--rectilinear-floor-contacts',action='store_true',
        help='Exact source-solid native box contacts for the package floor; original rendering preserved')
    p.add_argument('--native-spring-cut-trial',action='store_true',
        help='Explicit isolated comparison with unchanged native spring/contact drives in all phases')
    p.add_argument('--finger-target-antiwindup',action='store_true',
        help='Isolated explicit-finger trial: bound target windup using fresh native position/velocity and existing PD caps')
    p.add_argument('--retention-preload',action='store_true',
        help='Isolated comparison: 0.24 N support setpoint / 0.30 N PD cap; unchanged 0.5 N native contact and 0.8 N motor guards')
    p.add_argument('--symmetric-finger-closure',action='store_true',
        help='Isolated feedback comparison: one aperture command with a fixed target center; no physical weld or gear constraint')
    p.add_argument('--measured-withdrawal',action='store_true',
        help='Opt-in measured-start reverse path with fresh native geometry/hold checks; diagnostic only')
    p.add_argument('--native-static-clearance',action='store_true',help='Opt-in live native static-box refinement during the single synchronous bimanual plan')
    p.add_argument('--native-startup-approach-search',action='store_true',
        help='Zero-motion higher/lateral waiting-pose proposals with unchanged blade orientation; no path approval')
    p.add_argument('--native-startup-pose-search',action='store_true',
        help='Read-only frozen-scene elbow proposals; always stops before any physics motion')
    p.add_argument('--native-startup-heading-search',action='store_true',
        help='Zero-motion source knife heading proposals, preserving arc-up and the complete mounted tool')
    p.add_argument('--native-startup-clearance',action='store_true',
        help='Isolated contact diagnostic: full native collider/actor validation before FIRST step; no path/contact guard is bypassed')
    p.add_argument('--native-startup-station-search',action='store_true',
        help='Zero-motion base and both-arm proposals; same world grasp, complete native startup controls')
    p.add_argument('--station-proposal-report',type=Path,
        help='Reinitialize from a previously controlled station proposal, never replay its path or inherit its checks')
    p.add_argument('--coupled-fingers-trial',action='store_true',
        help='Experimental compliant one-aperture jaw mechanism; no plant weld or increased effort/contact limits')
    p.add_argument('--cut-priority-report',type=Path,help='Warm-start only candidate order from a prior native crossbar trial')
    p.add_argument('--native-capsule-sphere-cover',action='store_true',
        help='Optional whole-capsule conservative native sphere union after a coarse box hit; no sampled gaps')
    p.add_argument('--native-static-planning-seconds',type=float,default=8.,
        help='Bounded synchronous diagnostic planning budget, up to 60 seconds; default 8; no physics/collision guard changes')
    p.add_argument('--knife-edge-mode',choices=('source_side_edge_v1','source_lower_rim_v1','source_crossbar_edge_v1'),default='source_side_edge_v1',
        help='Source lower rim requires arc-up, measured downward load/travel; original mesh unchanged')
    p.add_argument('--source-wrist-contacts',action='store_true',help='Isolated comparison: source-complete bracket hulls shared by native and planner')
    p.add_argument('--cut-model',choices=('force_qualified_pre_authored_seam_release','signed_edge_load_brittle_seam_v1','loaded_downward_lower_rim_seam_v1'),
        default='force_qualified_pre_authored_seam_release',
        help='Opt-in engineering brittle seam strength model; neither mode is calibrated tissue fracture')
    p.add_argument('--diagnostic-grasp-dynamics',action='store_true',
        help='Opt-in same-step body/finger/contact/drive telemetry for bimanual retention debugging; not training data')
    p.add_argument('--grasp-contact-frames',choices=('post_fetch_legacy','pre_solve_pgs_v1'),default='post_fetch_legacy',
        help='Experimental discrete PGS contact-generation geometry plus independent post-fetch proximity; no force/guard changes')
    p.add_argument('--diagnostic-contact-prediction',action='store_true',
        help='Read-only complete plant-contact rows and native floating M/J snapshots; no predicted effort is applied')
    p.add_argument('--experimental-contact-springs',action='store_true',
        help='HOLD-ONLY fresh-contact springs activated after legacy grasp verification; native contacts and safety guards remain')
    p.add_argument('--finger-actuator-limit-n',type=float,choices=(.5,.8),default=.5,
        help='Engineering total motor budget including gravity, not contact force; both sparse modes enforce an independent 0.5 N all-contact cap per left finger')
    p.add_argument('--cut-proposal-json',type=Path,help='One source-bound world-direction diagnostic instead of the default orientation grid; all safety/IK checks remain')
    p.add_argument('--right-ik-fixed-joint',type=float,nargs=2,metavar=('INDEX','DEGREES'),
        help='Opt-in exact-URDF redundancy constraint through endpoint and entire stroke; all transit/scene/contact guards remain')
    p.add_argument('--cut-standoff-m',type=float,default=.025,
        help='Collision-screened precontact offset (8..25 mm), also checked against actual shaft size; bimanual only')
    p.add_argument('--bimanual-reposition-m',type=float,default=0.,
        help='Opt-in 0..10 mm held-target pull along the checked left approach, followed by native reobservation')
    p.add_argument('--grasp-compression-m',type=float,default=.0005,
        help='Diagnostic 0.25..1 mm shaft-width closure bias; no change to effort, slip or penetration guards')
    p.add_argument('--force-closure',action='store_true',
        help='Bimanual 240 Hz native-feedback finger closure with slower contact approach and five-second extra verification window; >=28 seconds')
    p.add_argument('--explicit-finger-effort',action='store_true',
        help='Isolated HOLD-only bounded explicit left-finger PD comparison; no native contact or force-guard change')
    p.add_argument('--isolated-cut-contact-trial',action='store_true',
        help='Explicit isolated flat-cylinder/effort-control cut DIAGNOSTIC; native126 hold is not tissue or greenhouse certification')
    p.add_argument('--fixed-root-contact-hold',action='store_true',
        help='Isolated fixed-root grasp HOLD comparison; original implicit or explicit native-spring comparison, no release')
    p.add_argument('--fixed-root-cut-trial',action='store_true',
        help='Isolated fixed-root cut diagnostic with no-step checked detach; not production or tissue qualification')
    p.add_argument('--diagnostic-free-root-dynamics',action='store_true',
        help='Read-only post-release native mass/Jacobian/velocity consistency; fixed-root cut diagnostic only')
    p.add_argument('--native-drives-after-cut',action='store_true',
        help='Fixed-root cut diagnostic: restore original native K/C after checked release, never tune material gains')
    p.add_argument('--rigid-pad-control',action='store_true',
        help='Explicit isolated CONTACT-LAW comparison only; cannot qualify the compliant-pad production model')
    p.add_argument('--require-retention-screen',action='store_true',
        help='Isolated fixed-root trial: require current contact-patch static gravity capacity before knife planning; not a dynamic certificate')
    p.add_argument('--pregrasp-half-aperture-m',type=float,default=.025,
        help='Isolated fixed-root trial only: commanded initial jaw half-opening, at least shaft radius +2 mm; original geometry/limits/guards unchanged')
    p.add_argument('--physical-grasp-span',action='store_true',
        help='Isolated retention trial: derive connected shaft identity from measured full pad footprint; original contact geometry/force verification remains mandatory')
    p.add_argument('--settle-retention-preload',action='store_true',
        help='Isolated retention trial: wait bounded 0.2 s measured original-preload dwell before static capacity audit; no force-limit increase')
    p.add_argument('--solver-convergence-trial',type=int,choices=(64,),default=None,
        help='Explicit numerical comparison only:64/0 vs baseline128/0, no fidelity equivalence assumed')
    p.add_argument('--joint-transit-fallback',action='store_true',
        help='Bounded whole-arm joint search after Cartesian approach failure; cut stroke remains downward')
    p.add_argument('--staged-downward-transit',action='store_true',
        help='Isolated guarded comparison: screen orient-then-straight approach before arm IK')
    p.add_argument('--preload-force-servo',action='store_true',
        help='Isolated retention trial only: narrower force-control deadband and bounded 0.5 s outer loop; unchanged physical limits')
    p.add_argument('--effort-bounded-grasp-target',action='store_true',
        help='Isolated retention trial: bound nominal position-reference bias by original PD effort/Kp; actual native penetration/force guards unchanged')
    p.add_argument('--branch-contact-fixture',action='store_true',
        help='CONTACT ONLY: keep original main stem and selected complete petiole/leaves; excludes other source branches in session; NOT intact-plant/greenhouse qualification')
    p.add_argument('--anchored-pad-damping',action='store_true',
        help='Explicit uncalibrated pad-damping prior for an anchored shaft; no mass/stiffness/force-guard change')
    p.add_argument('--bimanual-hold-control',action='store_true',
        help='Negative control: hold left grasp with right arm parked; never qualifies as cutting')
    p.add_argument('--diagnostic-grasp-contacts',action='store_true',
        help='Hold-control only: finish capturing a faulted contact step, reconcile raw signed tensors, then stop before any next command')
    p.add_argument('--robot-interactive',action='store_true',help='Keep the full-robot test window open with replay controls')
    p.add_argument('--robot-auto-run',action=argparse.BooleanOptionalAction,default=True,
        help='In robot interactive mode, run immediately; disable to inspect the mounting/target before Run')
    p.add_argument('--sparse-contacts',action='store_true',help='Native event accounting including all greenhouse/neighbor contacts')
    p.add_argument('--finger-gravity',action='store_true',help='Compensate native finger weight inside the original 0.5 N total effort budget')
    p.add_argument('--budgeted-joint-gravity',action='store_true',help='Experimental angular gravity compensation within source effort, preserving corrective reserve')
    p.add_argument('--compliant-fingers',action='store_true',help='Experimental native force-based finger-pad compliance; unchanged masses/effort/guard limits')
    p.add_argument('--approach-tilt',type=float,default=0.,help='Bounded diagnostic wrist tilt around the shaft, in degrees')
    p.add_argument('--station-offset',type=float,nargs=2,metavar=('FORWARD_M','LEFT_M'),
        help='Initial fixed-base station offset only (norm <=0.3 m); never moves a running robot')
    p.add_argument('--park-left-ready',action='store_true',help='Right-only trial with task-independent SDK left park, no grasp IK or grasp path')
    p.add_argument('--cut-station-orbit',action='store_true',help='Zero-motion station search checking raised waiting and cut-entry poses')
    p.add_argument('--station-reference-report',type=Path,help='Same-anatomy prior initial pose as zero-motion search seed; no replay authority')
    p.add_argument('--watch-auto-run',action='store_true',help='Explicitly start one watched demonstration after native startup; no reset/replay')
    p.add_argument('--station-yaw',type=float,default=0.,
        help='Initial station heading relative to palm approach, within +/-90 degrees; no live base motion')
    p.add_argument('--station-pose',type=float,nargs=3,metavar=('X_M','Y_M','YAW_DEG'),
        help='Explicit fixed station independent of grasp orientation; full robot only, no live base motion')
    p.add_argument('--approach-vector',type=float,nargs=3,
        help='Explicit initial palm approach direction for paired-layout qualification')
    p.add_argument('--grasp-roll',type=int,choices=(0,180),default=0,
        help='Initial equivalent finger orientation about palm approach axis; native grasp must be requalified')
    p.add_argument('--grasp-skew',type=float,default=0.,
        help='Bounded +/-30 degree jaw skew in the palm plane; requires native grasp requalification')
    p.add_argument('--grasp-pitch',type=float,default=0.,
        help='Isolated downward diagnostic: +/-60 degree pad-span pitch, no geometry or live pose override')
    p.add_argument('--grasp-depth-m',type=float,default=.1025,
        help='Shaft distance from the palm within the original pads: 90..125 mm; no live base or plant override')
    p.add_argument('--torso-yaw',type=float,default=0.,
        help='Fixed initial torso_5 yaw in package robot tests, bounded to +/-45 degrees')
    p.add_argument('--torso-degrees',type=float,nargs=6,
        help='Explicit prephysics torso proposal for downward bimanual test; exact URDF limits and full scene screens required')
    p.add_argument('--approach-distance',type=float,default=.08,
        help='Initial palm approach distance 0.01..0.08 m; leaves the selected fixed base unchanged')
    p.add_argument('--profile',action='store_true',help='Save diagnostic Python/native call timing alongside the non-training report')
    p.add_argument('--step-profile',action='store_true',help='Time the installed physics-only step phases without bypassing physics manager events')
    p.add_argument('--stream-trajectory',action='store_true',help='Lossless gzip JSONL full samples; bounded diagnostic memory, all per-step guards retained')
    p.add_argument('--no-physics-profiler',action='store_true',help='Disable optional native profiling instrumentation in this process only')
    p.add_argument('--local-wire-physics',action='store_true',help='Guarded fixed-base 4 m collision window; all wire visuals retained')
    p.add_argument('--physics-window-half-m',type=float,default=2.,
        help='Explicit 1..2 m fixed collision half-window; original full bounds plus 150 mm margin must fit, all visuals retained')
    p.add_argument('--context-gutters',type=int,choices=(1,3,5),help='Restore original preview planting density, with static mesh contacts near the fixed robot')
    p.add_argument('--scene-profile',action='store_true',help='Non-qualifying root-removal timing controls; never use as demo evidence')
    p.add_argument('--batch-gutter-visuals',action='store_true',help='Batch all identical static gutter visuals; retain every original gutter collider')
    p.add_argument('--capture-milestones',action=argparse.BooleanOptionalAction,default=True,
        help='Save paused diagnostic screenshots during a trial; disable for smoother interactive playback')
    return p


def validate_fixed_root_hold(args):
    """Narrow opt-in; never turns an attached-only result into cutting evidence."""
    enabled=args.fixed_root_contact_hold
    if enabled and not (args.isolated_cut_contact_trial and args.bimanual_hold_control
            and args.attached_only and args.constraint_mode=='fixed_articulation'
            and args.spring_mode==('native' if args.native_spring_cut_trial else 'implicit_effort') and args.diagnostic_grasp_dynamics
            and args.bimanual_reposition_m==0 and not args.native_torsion_trial
            and not args.measured_withdrawal):
        raise ValueError('Fixed root contact HOLD requires attached-only explicitly selected original springs, instrumented no-reposition isolated HOLD')
    return enabled


def main(argv=None):
    args=parser().parse_args(argv)
    if args.station_reference_report is not None:
        from .station_reference import validate_mode
        validate_mode(args)
    from .cut_priority import from_arguments as priority_from_arguments
    cut_priority=priority_from_arguments(args)
    proposal_receipt=None
    if args.station_proposal_report is not None:
        from .station_proposal import apply_to_arguments
        proposal_receipt=apply_to_arguments(args)
        if cut_priority is None:
            cut_priority=proposal_receipt.get('cut_frame_priority')
    fixed_hold=validate_fixed_root_hold(args)
    fixed_cut=args.fixed_root_cut_trial
    if args.coupled_fingers_trial and not (fixed_cut and args.bimanual_cut and args.symmetric_finger_closure
            and args.explicit_finger_effort and args.force_closure and args.diagnostic_grasp_dynamics
            and args.physics_hz==480 and not args.right_only_cut_trial):
        raise ValueError('Coupled fingers require the explicit 480 Hz instrumented bimanual symmetric-effort trial')
    from .greenhouse_cut import validate as validate_greenhouse
    greenhouse_trial=validate_greenhouse(args)
    if args.target_row_slot!=12 and not (args.scene=='package' and args.full_robot_probe and args.context_gutters):
        raise ValueError('End-row swap requires full package robot and preserved context planting')
    contact_scope=args.isolate_station or greenhouse_trial
    from .cut_only import validate_profile
    cut_only=validate_profile(args)
    if args.budgeted_joint_gravity and not args.full_robot_probe:
        raise ValueError('Budgeted angular gravity requires the full native robot')
    if args.park_left_ready and not cut_only:
        raise ValueError('Independent left park requires the complete right-only trial')
    if args.park_left_ready and args.native_startup_station_search and not args.cut_station_orbit:
        raise ValueError('Task-independent left park station search requires cut-station-orbit')
    from .cut_watch import validate as validate_watch
    validate_watch(args)
    if args.material_clearance_trial and not args.through_stroke_trial:
        raise ValueError('Material-clearance trial requires guarded through-stroke profile')
    if args.joint_transit_fallback and not (args.staged_downward_transit and args.native_static_clearance
            and args.native_startup_clearance and args.cut_style=='downward'):
        raise ValueError('Joint approach fallback requires the fully checked staged downward profile')
    from .postrelease_feed import validate as validate_feed
    validate_feed(args.postrelease_feed_m_s)
    if args.postrelease_feed_m_s!=.0003 and not args.material_clearance_trial:
        raise ValueError('Faster feed requires complete native material-clearance trial')
    if args.postcut_egress_trial and not (args.material_clearance_trial and args.native_station_park_reference
            and args.native_static_clearance and args.native_startup_clearance and not args.measured_withdrawal):
        raise ValueError('Post-cut egress requires the guarded material-clearance and native station profile')
    if args.rectilinear_floor_contacts and not (args.scene=='package' and args.full_robot_probe and args.bimanual_cut):
        raise ValueError('Exact floor contacts require the package full-robot cut diagnostic')
    if args.stream_trajectory and not args.bimanual_cut:
        raise ValueError('Streaming evidence currently requires the bimanual/direct-cut probe')
    if args.through_stroke_trial and not (args.seam_contact_yield and args.cut_action_trial
            and args.seconds>=80 and not args.measured_withdrawal):
        raise ValueError('Through-stroke requires complete process-zone cut-action trial and >=80 seconds')
    if args.cut_action_trial and not (fixed_cut and args.native_drives_after_cut
            and (args.branch_contact_fixture or greenhouse_trial) and args.native_startup_clearance
            and args.native_static_clearance and not args.gui and not args.robot_interactive
            and not args.measured_withdrawal):
        raise ValueError('Cut action milestone requires explicit native-release diagnostic without interactive reset')
    if args.cut_convergence_trial and not (fixed_cut and args.physics_hz==480
            and args.native_drives_after_cut and args.require_retention_screen
            and args.physical_grasp_span and args.settle_retention_preload
            and args.effort_bounded_grasp_target and args.preload_force_servo
            and args.native_startup_clearance and args.native_static_clearance
            and not args.gui and not args.robot_interactive and not args.measured_withdrawal):
        raise ValueError('480 Hz convergence requires complete fixed-root native-release retention trial without interactive reset')
    robot_rate_allowed=args.physics_hz==240 or args.cut_convergence_trial or cut_only
    if args.native_station_park_reference and not (fixed_cut and (args.require_retention_screen or cut_only)
            and args.native_static_clearance and not args.measured_withdrawal):
        raise ValueError('Native-station park requires isolated joint-reference cut/retention trial')
    if args.staged_downward_transit and not (args.cut_style=='downward' and contact_scope
            and args.fixed_root_cut_trial and (args.require_retention_screen or cut_only) and args.native_static_clearance):
        raise ValueError('Staged approach requires complete isolated downward native-retention fixture')
    if args.preload_force_servo and not args.effort_bounded_grasp_target:
        raise ValueError('Preload force servo requires the complete effort-bounded retention trial')
    if args.effort_bounded_grasp_target and not (args.settle_retention_preload and args.require_retention_screen
            and args.physical_grasp_span and args.explicit_finger_effort and args.finger_target_antiwindup):
        raise ValueError('Effort-bounded target requires the full explicit settled physical-span retention trial')
    if args.settle_retention_preload and not (args.require_retention_screen and args.retention_preload
            and args.symmetric_finger_closure and args.force_closure):
        raise ValueError('Preload settling requires original symmetric feedback and retention preflight')
    if args.physical_grasp_span and not (fixed_cut and args.require_retention_screen):
        raise ValueError('Physical grasp span requires the checked isolated fixed-root retention trial')
    if (not math.isfinite(args.pregrasp_half_aperture_m) or not 0<args.pregrasp_half_aperture_m<=.025
            or args.pregrasp_half_aperture_m!=.025 and not (fixed_cut and (args.require_retention_screen or cut_only)
                and args.force_closure and args.explicit_finger_effort and args.finger_target_antiwindup)):
        raise ValueError('Pre-grasp opening requires the complete fixed-root feedback trial with retention preflight')
    if args.require_retention_screen and not (fixed_cut and args.diagnostic_grasp_dynamics
            and args.finger_friction==.5 and args.gravity==9.81):
        raise ValueError('Retention preflight requires the original isolated fixed-root contact/dynamics trial')
    if args.rigid_pad_control and not (fixed_cut and args.compliant_fingers and args.branch_contact_fixture):
        raise ValueError('Rigid pad control requires complete isolated fixed-root branch comparison')
    if args.native_drives_after_cut and not fixed_cut:
        raise ValueError('Post-cut native drives require checked fixed-root cut diagnostic')
    if args.diagnostic_free_root_dynamics and not fixed_cut:
        raise ValueError('Free root dynamics requires isolated fixed-root cut diagnostic')
    if fixed_cut and not (args.isolated_cut_contact_trial and not args.bimanual_hold_control
            and not args.attached_only and args.constraint_mode=='fixed_articulation'
            and args.spring_mode=='implicit_effort' and args.diagnostic_grasp_dynamics
            and args.bimanual_reposition_m==0 and not args.native_torsion_trial
            and not args.native_spring_cut_trial and not fixed_hold):
        raise ValueError('Fixed root CUT requires original implicit instrumented no-reposition isolated cut trial')
    if args.blade_friction_budget and not (args.compliant_blade_rate and args.blade_dwell_feedback):
        raise ValueError('Blade friction budget requires isolated compliant rate and minimum-window feedback')
    if args.native_capsule_sphere_cover and not (args.bimanual_cut and args.native_static_clearance):
        raise ValueError('Native capsule sphere cover requires bimanual native static clearance')
    contact_trial=args.isolated_cut_contact_trial
    if args.compliant_blade_rate and not (contact_trial and args.blade_force_feed
            and args.seam_contact_compliance and not args.native_torsion_trial and not args.native_spring_cut_trial):
        raise ValueError('Compliant blade rate requires original implicit isolated compliant feed trial')
    if args.native_torsion_trial and not (contact_trial and args.spring_mode=='implicit_effort'
            and not args.native_spring_cut_trial and args.diagnostic_grasp_dynamics):
        raise ValueError('Native torsion comparison requires complete isolated implicit trial and dynamics telemetry')
    if args.blade_dwell_feedback and not (contact_trial and args.blade_force_feed and args.seam_contact_compliance):
        raise ValueError('Dwell feedback requires the complete isolated compliant blade feed')
    if args.symmetric_finger_closure and not (contact_trial and args.explicit_finger_effort
            and args.force_closure and args.finger_target_antiwindup):
        raise ValueError('Symmetric finger closure requires isolated explicit feedback and antiwindup')
    if args.retention_preload and not (contact_trial and args.explicit_finger_effort
            and args.force_closure and args.finger_target_antiwindup and args.compliant_fingers
            and args.blade_force_feed and args.seam_contact_compliance):
        raise ValueError('Retention preload requires the complete isolated compliant feedback trial')
    if args.finger_target_antiwindup and not (contact_trial and args.explicit_finger_effort and args.force_closure):
        raise ValueError('Finger antiwindup requires explicit isolated feedback cut trial')
    if args.native_spring_cut_trial and not (contact_trial and args.spring_mode=='native'):
        raise ValueError('Native spring cut comparison requires explicit isolated native-spring trial')
    if args.seam_contact_compliance and not (contact_trial and args.blade_force_feed):
        raise ValueError('Seam contact compression requires isolated blade feedback diagnostic')
    if args.seam_contact_yield and not (contact_trial and args.seam_contact_compliance
            and args.blade_friction_budget and args.physics_hz==480
            and args.knife_edge_mode=='source_crossbar_edge_v1'
            and args.cut_model=='loaded_downward_lower_rim_seam_v1'):
        raise ValueError('Seam yielding requires complete isolated 480 Hz crossbar feedback diagnostics')
    lower_rim=args.knife_edge_mode in ('source_lower_rim_v1','source_crossbar_edge_v1')
    if args.cut_station_orbit and not args.native_startup_station_search:
        raise ValueError('Cut station orbit requires explicit zero-motion station search')
    startup_search=any((args.native_startup_pose_search,args.native_startup_approach_search,args.native_startup_heading_search,args.native_startup_station_search))
    if sum((args.native_startup_pose_search,args.native_startup_approach_search,args.native_startup_heading_search,args.native_startup_station_search))>1:
        raise ValueError('Select one zero-motion startup search mode')
    if startup_search and not (args.native_startup_clearance and
            contact_trial and args.knife_edge_mode=='source_crossbar_edge_v1'):
        raise ValueError('Startup pose search requires isolated native crossbar diagnostics')
    if args.source_wrist_contacts and not (contact_trial and lower_rim):
        raise ValueError('Wrist contact comparison requires the isolated lower-rim trial')
    if (lower_rim!=(args.cut_model=='loaded_downward_lower_rim_seam_v1')
            or lower_rim and not (contact_trial and args.cut_style=='downward' and args.bimanual_cut)):
        raise ValueError('Lower rim requires the isolated measured downward cut model and planner')
    if args.blade_force_feed and not (contact_trial and args.cut_style=='downward'
            and args.cut_model in ('signed_edge_load_brittle_seam_v1','loaded_downward_lower_rim_seam_v1') and args.seconds>=40):
        raise ValueError('Blade feedback requires isolated downward signed-seam trial and >=40 seconds')
    if (not math.isfinite(args.blade_axial_aim_offset_m) or not -.0015<=args.blade_axial_aim_offset_m<=.0025
            or args.blade_axial_aim_offset_m and not (contact_trial and args.cut_style=='downward')):
        raise ValueError('Blade aim offset requires isolated downward contact trial and finite -1.5..+2.5 mm')
    if args.blade_axial_aim_offset_m>.0015 and not (args.knife_edge_mode=='source_crossbar_edge_v1'
            and args.material_clearance_trial and args.through_stroke_trial):
        raise ValueError('Distal blade aim requires source crossbar and complete measured material-section trial')
    if args.native_startup_clearance and not (contact_trial and args.native_static_clearance):
        raise ValueError('Native startup validation requires isolated cut contact and native path clearance')
    if args.branch_contact_fixture and not contact_trial:
        raise ValueError('Branch-only selection requires the isolated cut contact diagnostic')
    solver_comparison=args.solver_convergence_trial is not None
    if solver_comparison and not (args.uniform_solver_iterations==[64,0]
            and args.physics_hz==480 and args.material_clearance_trial and args.through_stroke_trial
            and fixed_cut and args.native_drives_after_cut and not args.watch_cut_trial):
        raise ValueError('Solver convergence comparison requires explicit64/0 and complete480Hz material-clearance trial')
    if contact_trial and not (args.scene=='package' and contact_scope and args.full_robot_probe
            and args.bimanual_cut and (not args.bimanual_hold_control or fixed_hold) and not args.robot_interactive
            and args.explicit_finger_effort and args.force_closure and robot_rate_allowed
            and args.solver=='PGS' and args.spring_mode==('native' if args.native_spring_cut_trial else 'implicit_effort')
            and args.constraint_mode==('fixed_articulation' if fixed_hold or fixed_cut else 'articulation')
            and args.stem_contact_model=='flat_cylinders_v1' and args.grasp_contact_frames=='pre_solve_pgs_v1'
            and (args.uniform_solver_iterations in ([128,0],[128,8]) or solver_comparison) and args.force_newton==0
            and not args.diagnostic_contact_prediction and not args.experimental_contact_springs):
        raise ValueError('Isolated cut contact trial requires the complete explicit flat/PGS/spring/128-0-or-8 fixture')
    if args.explicit_finger_effort and not (contact_scope and (args.bimanual_hold_control or contact_trial)
            and args.bimanual_cut and args.force_closure and robot_rate_allowed
            and not args.robot_interactive and not args.experimental_contact_springs):
        raise ValueError('Explicit finger effort requires isolated 240 Hz feedback HOLD ONLY')
    if args.uniform_solver_iterations is not None and not (contact_scope and not args.robot_interactive
            and (args.bimanual_hold_control or contact_trial) and args.bimanual_cut
            and tuple(args.uniform_solver_iterations) in ((32,8),(64,0),(128,0),(128,8))):
        raise ValueError('Uniform iterations require isolated HOLD and a bounded diagnostic pair')
    native_hold=(args.spring_mode=='native' and args.isolate_station and args.bimanual_cut
        and args.bimanual_hold_control and args.force_newton==0
        and not args.diagnostic_contact_prediction and not args.experimental_contact_springs
        and not args.robot_interactive)
    if args.stem_contact_model=='flat_cylinders_v1' and not (
            contact_scope and (args.bimanual_hold_control or contact_trial) and args.bimanual_cut and not args.robot_interactive
            and args.grasp_contact_frames=='pre_solve_pgs_v1'
            and not args.diagnostic_contact_prediction and not args.experimental_contact_springs):
        raise ValueError('Flat cylinder experiment requires isolated corrected-frame bimanual HOLD ONLY')
    if args.isolate_station and not (args.scene=='package' and args.full_robot_probe
            and not args.local_wire_physics and not args.context_gutters
            and not args.batch_gutter_visuals and not args.scene_profile):
        raise ValueError('Isolated station requires package full robot without greenhouse context/window/batching/profiling')
    if args.torso_degrees is not None:
        if not (args.bimanual_cut and args.cut_style=='downward' and args.scene=='package') or args.torso_yaw:
            raise ValueError('Explicit torso requires package downward bimanual fixture and no yaw override')
        from greenhouse_sim.robot_kinematics import Rby1Kinematics
        lower,upper=Rby1Kinematics().torso_limits_degrees()
        if not all(math.isfinite(v) and lo<=v<=hi for v,lo,hi in zip(args.torso_degrees,lower,upper,strict=True)):
            raise ValueError('Explicit torso must obey exact URDF limits')
    if args.anchored_pad_damping and not (args.force_closure and args.compliant_fingers):
        raise ValueError('Anchored pad damping requires compliant feedback closure')
    if (args.right_ready_degrees is not None or args.left_ik_seed_degrees is not None) and (not args.bimanual_cut or args.cut_style!='downward'):
        raise ValueError('Coordinated right ready pose requires the downward bimanual fixture')
    if (not math.isfinite(args.right_ready_lift_m) or not 0<=args.right_ready_lift_m<=.05
            or args.right_ready_lift_m and not (args.bimanual_cut and args.cut_style=='downward'
                and args.native_startup_clearance and args.native_static_clearance)):
        raise ValueError('Waiting-pose lift requires downward fixture with complete native startup/path checks')
    if (not math.isfinite(args.right_ready_retreat_m) or not 0<=args.right_ready_retreat_m<=.05
            or args.right_ready_retreat_m and not (args.bimanual_cut and args.cut_style=='downward'
                and args.knife_alignment=='camera' and args.native_startup_clearance and args.native_static_clearance)):
        raise ValueError('Waiting-pose retreat requires the fitted downward knife and complete native checks')
    if args.exact_grasp_arc and not args.bimanual_cut:
        raise ValueError('Exact grasp arc requires bimanual qualification')
    if (args.knife_alignment!='legacy' or args.cut_style!='legacy') and not args.bimanual_cut:
        raise ValueError('Knife alignment and cut style require bimanual qualification')
    if args.cut_style=='downward' and (args.cut_proposal_json is not None or args.right_ik_fixed_joint is not None):
        raise ValueError('Downward style cannot reuse legacy direction or fixed-shoulder proposals')
    if not math.isfinite(args.grasp_skew) or abs(args.grasp_skew)>30 or (args.grasp_skew and not args.full_robot_probe):
        raise ValueError('Jaw skew requires a full robot and finite +/-30 degrees')
    if (not math.isfinite(args.grasp_pitch) or abs(args.grasp_pitch)>60
            or args.grasp_pitch and not (args.full_robot_probe and args.bimanual_cut
                and args.cut_style=='downward' and args.isolated_cut_contact_trial)):
        raise ValueError('Pad-span pitch requires the isolated downward full-robot trial within +/-60 degrees')
    if not math.isfinite(args.grasp_compression_m) or not .00025<=args.grasp_compression_m<=.001 or (
            args.grasp_compression_m!=.0005 and not args.bimanual_cut):
        raise ValueError('Grasp compression requires bimanual qualification within 0.25..1 mm')
    if not math.isfinite(args.bimanual_reposition_m) or not 0<=args.bimanual_reposition_m<=.01 or (
            args.bimanual_reposition_m and (not args.bimanual_cut or args.seconds<24
                or args.approach_distance<args.bimanual_reposition_m+.008)):
        raise ValueError('Reposition requires bimanual, >=24 seconds, 0..10 mm and approach clearance for pull +8 mm retention')
    if not math.isfinite(args.cut_standoff_m) or not .008<=args.cut_standoff_m<=.025 or (
            args.cut_standoff_m!=.025 and not args.bimanual_cut):
        raise ValueError('Cut standoff requires bimanual qualification within 8..25 mm')
    if not math.isfinite(args.cut_arc_m) or not .01<=args.cut_arc_m<=.02 or (
            args.cut_arc_m!=.01 and not args.bimanual_cut):
        raise ValueError('Non-default cut arc requires bimanual qualification within 10..20 mm')
    if not math.isfinite(args.grasp_depth_m) or not .09<=args.grasp_depth_m<=.125 or (
            args.grasp_depth_m!=.1025 and not args.full_robot_probe):
        raise ValueError('Grasp depth requires a full robot and finite 90..125 mm')
    if not math.isfinite(args.station_yaw) or abs(args.station_yaw)>90 or (args.station_yaw and not args.full_robot_probe):
        raise ValueError('Station yaw requires a full robot and finite +/-90 degrees')
    if args.bimanual_hold_control and not args.bimanual_cut:
        raise ValueError('Bimanual hold control requires the guarded bimanual harness')
    if getattr(args,'diagnostic_grasp_contacts',False) and not args.bimanual_hold_control:
        raise ValueError('Raw grasp contact diagnostic requires right-parked bimanual hold control')
    if getattr(args,'native_static_clearance',False) and not args.bimanual_cut:
        raise ValueError('Native static clearance requires the bimanual harness')
    if getattr(args,'measured_withdrawal',False) and (not args.bimanual_cut
            or args.bimanual_hold_control or not getattr(args,'native_static_clearance',False)):
        raise ValueError('Measured withdrawal requires bimanual cutting and native static clearance, not hold control')
    budget=getattr(args,'native_static_planning_seconds',8.)
    if (not math.isfinite(budget) or not 0<budget<=60
            or (budget!=8 and not getattr(args,'native_static_clearance',False))):
        raise ValueError('Non-default native planning budget requires native clearance and (0,60] seconds')
    if (getattr(args,'cut_model','force_qualified_pre_authored_seam_release')!='force_qualified_pre_authored_seam_release'
            and not args.bimanual_cut):
        raise ValueError('Engineering brittle seam model requires guarded bimanual qualification')
    if getattr(args,'diagnostic_grasp_dynamics',False) and not args.bimanual_cut:
        raise ValueError('Grasp dynamics telemetry requires guarded bimanual qualification')
    if args.grasp_contact_frames=='pre_solve_pgs_v1' and not (
            args.bimanual_cut and args.full_robot_probe and args.solver=='PGS' and robot_rate_allowed
            and not args.diagnostic_grasp_contacts):
        raise ValueError('Pre-step grasp frames require synchronous 240 Hz PGS bimanual qualification')
    if getattr(args,'experimental_contact_springs',False) and not (
            args.bimanual_hold_control and args.bimanual_cut and args.full_robot_probe
            and args.diagnostic_contact_prediction and args.diagnostic_grasp_dynamics
            and args.sparse_contacts and args.finger_gravity and args.compliant_fingers
            and args.force_newton==0 and not args.bimanual_reposition_m
            and not args.robot_interactive and not args.measured_withdrawal):
        raise ValueError('Experimental contact springs require bounded stationary full-robot HOLD ONLY with contact/dynamics diagnostics and native compliant fingers')
    if getattr(args,'diagnostic_contact_prediction',False) and not (
            args.bimanual_cut and args.diagnostic_grasp_dynamics):
        raise ValueError('Contact prediction snapshots require bimanual grasp dynamics diagnostics')
    if (getattr(args,'finger_actuator_limit_n',.5)!=.5
            and not (args.bimanual_cut and args.sparse_contacts and args.finger_gravity and args.compliant_fingers)):
        raise ValueError('Experimental finger actuator budget requires guarded bimanual, sparse contacts, gravity and compliant fingers')
    if getattr(args,'cut_proposal_json',None) is not None and not args.bimanual_cut:
        raise ValueError('Single cut proposal requires the bimanual harness')
    fixed=getattr(args,'right_ik_fixed_joint',None)
    if fixed is not None and (not args.bimanual_cut or len(fixed)!=2
            or not all(math.isfinite(v) for v in fixed)
            or fixed[0]!=int(fixed[0]) or not 0<=int(fixed[0])<7):
        raise ValueError('Fixed right IK joint requires bimanual, index 0..6 and finite degrees')
    if args.station_offset is not None and (not args.full_robot_probe
            or not all(math.isfinite(x) for x in args.station_offset)
            or math.hypot(*args.station_offset)>.3):
        raise ValueError('Station offset requires a full robot and finite norm <=0.3 m')
    if args.grasp_roll and not args.full_robot_probe:
        raise ValueError('Grasp roll requires a full robot')
    if not math.isfinite(args.approach_distance) or not .01<=args.approach_distance<=.08 or (
            args.approach_distance!=.08 and not args.full_robot_probe):
        raise ValueError('Approach distance requires a full robot and finite 10..80 mm')
    if not math.isfinite(args.torso_yaw) or abs(args.torso_yaw)>45 or (
            args.torso_yaw and not (args.full_robot_probe and args.scene=='package')):
        raise ValueError('Torso yaw requires a package full robot and finite +/-45 degrees')
    if args.bimanual_cut and (not args.full_robot_probe or not args.sparse_contacts or not args.finger_gravity or args.seconds<20):
        raise ValueError('Bimanual cutting requires full robot, sparse contacts, finger gravity and >=20 seconds')
    if args.station_pose is not None and (not args.full_robot_probe or args.station_offset is not None
            or args.station_yaw or not all(math.isfinite(v) for v in args.station_pose) or abs(args.station_pose[2])>180):
        raise ValueError('Explicit station requires full robot, finite x/y/yaw and no relative offsets')
    if args.approach_vector is not None and (not args.full_robot_probe
            or not all(math.isfinite(v) for v in args.approach_vector) or sum(v*v for v in args.approach_vector)<1e-12):
        raise ValueError('Approach vector requires full robot and a finite nonzero direction')
    if not args.robot_auto_run and not args.robot_interactive:
        raise ValueError('Disabling robot auto-run requires robot interactive mode')
    if args.robot_interactive and (not args.full_robot_probe or not args.gui or not args.render_hz):
        raise ValueError('Robot interactive requires full-robot probe, GUI and rendering')
    if args.force_closure and not (args.bimanual_cut and args.compliant_fingers and args.seconds>=28):
        raise ValueError('Force closure requires bimanual compliant native fingers')
    if args.full_robot_probe and (args.gripper_probe or args.interactive or (args.scene=='package' and not args.sparse_contacts)
            or (args.constraint_mode!='articulation' and not (fixed_cut or (native_hold or fixed_hold) and args.constraint_mode=='fixed_articulation' and args.attached_only))
            or (args.spring_mode!='implicit_effort' and not native_hold and not args.native_spring_cut_trial)
            or args.solver!='PGS' or not robot_rate_allowed or args.gravity!=9.81 or args.seconds<7
            or args.diagnostic_detach or not -30<=args.approach_tilt<=30
            or not 0<=args.finger_friction<=1 or not .04<=args.grasp_arc_m<=.25):
        raise ValueError('Full robot probe requires implicit articulation, PGS 240 Hz, gravity, >=7 s, bounded grasp/friction, no diagnostic detach and sparse contacts for package scenes')
    if (args.sparse_contacts or args.finger_gravity or args.approach_tilt or args.compliant_fingers) and not args.full_robot_probe:
        raise ValueError('Robot contact/gravity/approach options require the full robot probe')
    if args.local_wire_physics and not (args.full_robot_probe and args.scene=='package'):
        raise ValueError('Local wire physics requires the fixed full robot in the supplied package')
    if (not math.isfinite(args.physics_window_half_m) or not 1<=args.physics_window_half_m<=2
            or args.physics_window_half_m!=2 and not args.local_wire_physics):
        raise ValueError('Physics window requires guarded fixed workspace and finite 1..2 m half extent')
    if args.context_gutters and not args.local_wire_physics:
        raise ValueError('Dense context requires a guarded local collision window')
    if args.batch_gutter_visuals and not (args.full_robot_probe and args.scene=='package'):
        raise ValueError('Gutter batching requires a full-robot package scene')
    if args.scene_profile and not (args.full_robot_probe and args.scene=='package' and not args.robot_interactive and not args.gui):
        raise ValueError('Scene profiling requires a bounded headless full-robot package diagnostic')
    if args.profile and (not (args.gripper_probe or args.full_robot_probe) or args.robot_interactive):
        raise ValueError('Profiling requires a bounded gripper or full-robot probe')
    if args.gripper_probe and (args.interactive or args.scene!='isolated'
            or args.constraint_mode!='articulation' or args.spring_mode!='implicit_effort'
            or args.solver!='PGS' or args.gravity!=9.81 or args.seconds<7
            or not 0<=args.finger_friction<=1 or not .04<=args.grasp_arc_m<=.25):
        raise ValueError('Gripper probe requires isolated articulation, implicit_effort, PGS, gravity, >=7 s and bounded friction/grasp location')
    if args.interactive and (not args.gui or args.scene!='isolated' or args.render_hz==0
            or args.spring_mode!='implicit_effort' or args.constraint_mode!='articulation'
            or args.gravity!=9.81 or args.solver!='PGS' or args.physics_hz!=240):
        raise ValueError('Interactive demo requires GUI, isolated scene, rendering, implicit_effort, articulation, Earth gravity, PGS and 240 Hz')
    if args.constraint_mode=='fixed_articulation' and not args.attached_only and not fixed_cut:
        raise ValueError('Fixed-base comparison is attached-only until topology transition is qualified')
    if args.spring_mode=='implicit_effort' and args.constraint_mode=='maximal':
        raise ValueError('Implicit spring diagnostic requires an articulation')
    if not 0<=args.force_newton<=.2:
        raise ValueError('Diagnostic force must be between zero and 0.2 N')
    from sim_data.audit import DEFAULT_PACK
    output=args.output.resolve()
    if output.exists() or output.is_relative_to(DEFAULT_PACK.resolve()):
        raise ValueError('Choose a NEW output outside the source package')
    if not 4<=args.seconds<=(90 if args.through_stroke_trial else 60 if args.blade_force_feed else 30) or not .005<=args.max_segment_m<=.05:
        raise ValueError('Qualification requires 4-30 seconds (feedback:40-60; through-stroke:80-90) and 5-50 mm segments')
    output.mkdir(parents=True)
    report=dict(state='initializing',started_utc=datetime.now(timezone.utc).isoformat(),
                configuration=report_configuration(args,output),training_eligible=False)
    if args.full_robot_probe and args.scene=='package':
        from .host_memory import preflight
        report['host_memory_preflight']=preflight()
        if not report['host_memory_preflight']['allowed']:
            report.update(state='blocked_host_memory',simulation_started=False,
                error='Host memory reserve unavailable; inspect report before restarting the full greenhouse')
            (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
            print('PHYSICS_HOST_MEMORY_BLOCKED '+json.dumps(report),flush=True)
            return 2
    from isaacsim import SimulationApp
    from .startup_report import start as start_application
    app=start_application(SimulationApp,{'headless':not (args.gui or args.watch_cut_trial),'width':1280 if args.interactive or args.watch_cut_trial else 848,'height':900 if args.watch_cut_trial else 720 if args.interactive else 408,'multi_gpu':False,
                       'sync_loads':False,'renderer':'RaytracedLighting'},output,report)
    import carb.settings
    process_settings=carb.settings.get_settings()
    thread_setting='/persistent/physics/numThreads'
    previous_threads=process_settings.get(thread_setting)
    profiler_setting='/physics/exposeProfilerData'
    previous_profiler=process_settings.get(profiler_setting)
    cylinder_setting=None;previous_cylinders=None
    if args.no_physics_profiler: process_settings.set_bool(profiler_setting,False)
    if args.physics_threads is not None:
        process_settings.set_int(thread_setting,args.physics_threads)
    try:
        import numpy as np
        import omni.usd
        from pxr import Gf,Usd,UsdGeom,UsdPhysics,PhysxSchema
        from isaacsim.core.api import SimulationContext
        from greenhouse_sim.physics_clock import PhysicsClock
        from sim_data.audit import audit_manifest
        from sim_data.geometry import assemble_plant
        from .plant import build
        from .runtime import PlantRuntime
        context=omni.usd.get_context()
        manifest=DEFAULT_PACK/f'plants/components/{args.plant}/manifest.json'
        audit=audit_manifest(manifest)
        source_hashes={manifest:sha256_file(manifest)}
        for component in audit['components'].values():
            path=manifest.parent/component['file'];source_hashes[path]=component['asset_sha256']
        robot_options=dict(sparse_contacts=args.sparse_contacts,finger_gravity=args.finger_gravity,
            budgeted_joint_gravity=args.budgeted_joint_gravity,
            park_left_ready=args.park_left_ready,
            exact_grasp_arc=args.exact_grasp_arc,
            right_ready_degrees=args.right_ready_degrees,
            right_ready_lift_m=args.right_ready_lift_m,
            right_ready_retreat_m=args.right_ready_retreat_m,
            left_ik_seed_degrees=args.left_ik_seed_degrees,
            anchored_pad_damping=args.anchored_pad_damping,
            grasp_skew=args.grasp_skew,
            grasp_pitch=args.grasp_pitch,
            approach_tilt=args.approach_tilt,grasp_roll=args.grasp_roll,approach_distance=args.approach_distance,
            compliant_fingers=args.compliant_fingers,station_yaw=args.station_yaw,grasp_depth=args.grasp_depth_m)
        if args.station_offset is not None: robot_options['station_offset']=args.station_offset
        if args.station_pose is not None: robot_options['station_pose']=args.station_pose
        if proposal_receipt is not None:report['startup_station_proposal']=proposal_receipt
        if args.approach_vector is not None: robot_options['approach_vector']=args.approach_vector
        if args.scene=='package' and args.full_robot_probe:
            from .greenhouse_scene import prepare
            from sim_data.floor_alignment import PACKAGE_FLOOR
            scene=DEFAULT_PACK/'house/green_house_base.usd'
            source_hashes[scene]=sha256_file(scene)
            if not context.open_stage(str(scene),load_set=omni.usd.UsdContextInitialLoadSet.LOAD_NONE):
                raise RuntimeError('Cannot open supplied greenhouse')
            stage=context.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
            record,height,scene_report=prepare(stage,DEFAULT_PACK,args.plant,sparse_backdrop=not args.context_gutters,
                target_row_slot=args.target_row_slot)
            report['greenhouse']=scene_report
            if args.rectilinear_floor_contacts:
                from .floor_contacts import apply as apply_floor_contacts
                report['floor_contact_geometry']=apply_floor_contacts(stage,PACKAGE_FLOOR)
            if args.isolate_station:
                from .isolated_station import isolate
                report['isolated_station']=isolate(stage,floor_root=PACKAGE_FLOOR)
                report['greenhouse']['full_environment_present']=False
            torso=args.torso_degrees if args.torso_degrees is not None else [0.,0.,0.,0.,0.,args.torso_yaw]
            robot_options.update(ground_height=height,torso_degrees=torso,floor_root=PACKAGE_FLOOR)
        elif args.scene=='package':
            from launch_sim_data import load_local_payloads,populate
            from sim_data.robot_preview import add_robot_preview,select_camera
            from sim_data.floor_alignment import PACKAGE_FLOOR
            scene=DEFAULT_PACK/'house/green_house_base.usd'
            source_hashes[scene]=sha256_file(scene)
            context.open_stage(str(scene),load_set=omni.usd.UsdContextInitialLoadSet.LOAD_NONE)
            stage=context.get_stage();stage.SetEditTarget(stage.GetSessionLayer())
            load_local_payloads(stage);records=[]
            cx,counts=populate(stage,DEFAULT_PACK,app,records,args.plant)
            record=records[0]
            robot=add_robot_preview(stage,gutter_x=cx,floor_path=PACKAGE_FLOOR,right_tool='knife_only')
            report.update(scene_counts=counts,robot=robot)
            from omni.kit.viewport.utility import get_active_viewport
            viewport=get_active_viewport()
            if viewport: select_camera(viewport,robot['camera_paths']['Robot head D405'])
        else:
            context.new_stage();stage=context.get_stage()
            UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
            stage.SetEditTarget(stage.GetSessionLayer())
            paths=assemble_plant(stage,'/World/Plant',audit)
            record=dict(manifest_path=str(manifest),component_paths=paths,plant_root='/World/Plant')
            if args.full_robot_probe:
                # Explicit test-station placement BEFORE building physics, never
                # a running plant pose override or source-package edit.
                UsdGeom.Xformable(stage.GetPrimAtPath('/World/Plant')).AddTranslateOp(opSuffix='testStation').Set(Gf.Vec3d(0,0,.35))
        if args.branch_contact_fixture:
            from .branch_contact_fixture import select_components
            report['branch_contact_fixture']=select_components(stage,record,args.target)
            report['isolated_station']['scope']='branch_contact_fixture_NOT_intact_source_plant'
        if greenhouse_trial:
            from .greenhouse_cut import intact_plant
            report['greenhouse_cut_scope']=intact_plant(stage,record)
            report['greenhouse']['full_environment_present']=True
        rig=build(stage,record,args.target,max_segment_m=args.max_segment_m,constraint_mode=args.constraint_mode,
                  cut_m=args.cut_arc_m,stem_contact_model=args.stem_contact_model)
        if args.seam_contact_compliance:
            from .seam_contact_compliance import apply as apply_seam_compliance
            report['seam_contact_compliance']=apply_seam_compliance(rig,diagnostic_only=True)
        if args.stem_contact_model=='flat_cylinders_v1':
            from omni.physx.bindings._physx import SETTING_COLLISION_APPROXIMATE_CYLINDERS
            cylinder_setting=SETTING_COLLISION_APPROXIMATE_CYLINDERS
            previous_cylinders=process_settings.get(cylinder_setting)
            # Owned diagnostic process only, before physics parsing. Exact
            # analytic cylinders, zero margin, never silently cooked polygons.
            process_settings.set_bool(SETTING_COLLISION_APPROXIMATE_CYLINDERS,False)
            value=process_settings.get(SETTING_COLLISION_APPROXIMATE_CYLINDERS)
            if value is not False:raise RuntimeError('Exact cylinder backend setting not applied')
            report['stem_collision_backend']=dict(approximate_cylinders=value,margin_m=0.,
                source='explicit_USD_cylinder_and_process_setting',native_cooked_shape_readback=False,
                full_greenhouse_qualified=False)
        report['rig']=rig.report()
        if args.station_reference_report is not None:
            from .station_reference import initial_options
            seed_options,receipt=initial_options(args,rig)
            robot_options.update(seed_options)
            report['station_search_reference']=receipt
        fixture=None
        if args.full_robot_probe:
            from .full_robot import FullRobotGripper
            robot_class=FullRobotGripper
            if args.bimanual_cut:
                from .bimanual import BimanualRobot
                robot_class=BimanualRobot
                robot_options['cut_standoff']=args.cut_standoff_m
                robot_options['blade_axial_aim_offset_m']=args.blade_axial_aim_offset_m
                robot_options['knife_alignment']=args.knife_alignment
                robot_options['cut_style']=args.cut_style
                robot_options['grasp_compression']=args.grasp_compression_m
                robot_options['force_closure']=args.force_closure
                if args.cut_convergence_trial or cut_only:robot_options['diagnostic_physics_hz']=args.physics_hz
                if cut_only:robot_options['cut_strategy']='right_only'
                robot_options['pregrasp_half_aperture']=args.pregrasp_half_aperture_m
                robot_options['physical_grasp_span']=args.physical_grasp_span
                robot_options['effort_bounded_grasp_target']=args.effort_bounded_grasp_target
                robot_options['preload_force_servo']=args.preload_force_servo
                robot_options['staged_downward_transit']=args.staged_downward_transit
                robot_options['explicit_finger_effort']=args.explicit_finger_effort
                robot_options['finger_target_antiwindup']=args.finger_target_antiwindup
                robot_options['retention_preload']=args.retention_preload
                robot_options['symmetric_finger_closure']=args.symmetric_finger_closure
                robot_options['native_static_clearance']=getattr(args,'native_static_clearance',False)
                robot_options['native_capsule_sphere_cover']=args.native_capsule_sphere_cover
                robot_options['native_static_planning_seconds']=getattr(args,'native_static_planning_seconds',8.)
                robot_options['cut_model']=getattr(args,'cut_model','force_qualified_pre_authored_seam_release')
                robot_options['knife_edge_mode']=args.knife_edge_mode
                robot_options['cut_priority']=cut_priority
                robot_options['source_wrist_contacts']=args.source_wrist_contacts
                robot_options['finger_actuator_limit_n']=getattr(args,'finger_actuator_limit_n',.5)
                robot_options['cut_proposal_json']=getattr(args,'cut_proposal_json',None)
                robot_options['right_ik_fixed_joint']=getattr(args,'right_ik_fixed_joint',None)
                robot_options['diagnostic_grasp_contacts']=getattr(args,'diagnostic_grasp_contacts',False)
                robot_options['grasp_contact_frames']=args.grasp_contact_frames
            fixture=robot_class(stage,rig,arc=args.grasp_arc_m,friction=args.finger_friction,**robot_options)
            fixture.joint_transit_fallback=args.joint_transit_fallback
            fixture.station_reference_search=args.station_reference_report is not None
            if args.coupled_fingers_trial:
                from .finger_coupling import author as author_coupling
                report['finger_mechanism_trial']=author_coupling(stage,fixture)
            if args.rigid_pad_control:
                from .pad_contact_control import apply as apply_pad_control
                report['pad_contact_control']=apply_pad_control(fixture,diagnostic_only=True)
            source_hashes[fixture.asset]=sha256_file(fixture.asset)
            report['robot_probe']=fixture.report()
            if args.local_wire_physics:
                from .collision_window import configure
                report['collision_window']=configure(stage,fixture,half_extent=args.physics_window_half_m)
            if args.context_gutters:
                from .greenhouse_context import populate as populate_context
                context_options=dict(target_row_slot=args.target_row_slot)
                if greenhouse_trial:
                    # Match launch_sim_data.populate's ordered second detailed
                    # plant, not a lower-detail backdrop at the adjacent station.
                    candidates=sorted((DEFAULT_PACK/'plants/components').glob('*/manifest.json'))
                    neighbors=[p for p in candidates if p.parent.name!=args.plant]
                    if not neighbors:raise ValueError('Preview detailed neighbor is missing')
                    context_options['detailed_neighbor_manifest']=neighbors[0]
                report['context_plants']=populate_context(stage,DEFAULT_PACK,fixture,args.context_gutters,**context_options)
                source_hashes.update({Path(p):h for p,h in report['context_plants']['source_sha256'].items()})
            if args.batch_gutter_visuals:
                from .gutter_instances import batch
                report['gutter_visual_batch']=batch(stage)
        if args.gripper_probe:
            from .gripper_probe import GripperFixture
            fixture=GripperFixture(stage,rig,arc=args.grasp_arc_m,friction=args.finger_friction)
            source_hashes[fixture.asset]=sha256_file(fixture.asset)
            report['gripper_fixture']=fixture.report()
        if args.bimanual_cut:
            if args.uniform_solver_iterations is not None:
                from .solver_configuration import uniform_iterations
                report['uniform_solver_iterations']=uniform_iterations(stage,
                    (rig.root,fixture.root),args.uniform_solver_iterations)
            from .startup_screen import screen
            report['startup_collision_screen']=screen(stage,fixture)
            if not report['startup_collision_screen']['passed'] and not args.native_startup_clearance:
                raise RuntimeError('Bimanual spawn has possible collision overlaps; inspect startup_collision_screen before any physics motion')
        physics=UsdPhysics.Scene.Define(stage,'/World/QualificationPhysics')
        physics.CreateGravityDirectionAttr(Gf.Vec3f(0,0,-1));physics.CreateGravityMagnitudeAttr(args.gravity)
        settings=PhysxSchema.PhysxSceneAPI.Apply(physics.GetPrim())
        settings.CreateSolverTypeAttr(args.solver);settings.CreateEnableGPUDynamicsAttr(False)
        settings.CreateBroadphaseTypeAttr('MBP')
        settings.CreateFrictionTypeAttr('patch')  # Required for native friction-anchor reports.
        from .solver_configuration import author_contact_order
        report['contact_solver_order']=author_contact_order(physics.GetPrim(),
            enabled=getattr(args,'solve_articulation_contact_last',False))
        if args.native_startup_clearance:
            from .native_startup_screen import screen as native_startup_screen
            if startup_search:fixture.startup_right_pose_search=True
            if args.native_startup_approach_search:fixture.startup_approach_search=True
            if args.native_startup_heading_search:fixture.startup_heading_search=True
            if args.native_startup_station_search:fixture.startup_station_search=True
            if args.cut_station_orbit:fixture.cut_station_orbit=True
            report['native_startup_collision_screen']=native_startup_screen(stage,fixture)
            if startup_search:
                raise RuntimeError('Read-only startup pose search completed; relaunch and revalidate before motion')
            if not report['native_startup_collision_screen']['passed']:
                raise RuntimeError('Complete native startup geometry not verified; no physics motion allowed')
        # One explicit scene, fixed dt. Rendering is independently scheduled.
        sim=SimulationContext(physics_dt=1/args.physics_hz,rendering_dt=1/60,
                              stage_units_in_meters=1,physics_prim_path='/World/QualificationPhysics',
                              set_defaults=False,backend='numpy')
        if args.fabric:
            sim.get_physics_context().enable_fabric(True)
        report['effective_scene']=dict(gravity_m_s2=float(physics.GetGravityMagnitudeAttr().Get()),
            solver=settings.GetSolverTypeAttr().Get(),physics_dt=sim.get_physics_dt())
        if (report['effective_scene']['solver']!=args.solver
                or not np.isclose(report['effective_scene']['gravity_m_s2'],args.gravity,rtol=1e-6,atol=1e-8)):
            raise RuntimeError('Simulation initialization changed explicit scene configuration')
        sim.reset()
        if 'uniform_solver_iterations' in report:
            from .solver_configuration import verify_iterations
            verify_iterations(stage,report['uniform_solver_iterations'])
        if cylinder_setting is not None and process_settings.get(cylinder_setting) is not False:
            raise RuntimeError('Physics reset changed the exact cylinder backend setting')
        report['effective_scene_after_reset']=dict(gravity_m_s2=float(physics.GetGravityMagnitudeAttr().Get()),
            solver=settings.GetSolverTypeAttr().Get(),physics_dt=sim.get_physics_dt(),
            friction_type=settings.GetFrictionTypeAttr().Get(),
            gpu_dynamics=settings.GetEnableGPUDynamicsAttr().Get(),
            update_to_usd=process_settings.get('/physics/updateToUsd'),
            physics_threads=process_settings.get(thread_setting))
        order_after=physics.GetPrim().GetAttribute('physxScene:solveArticulationContactLast').Get()
        report['contact_solver_order']['usd_value_after_reset']=order_after
        if getattr(args,'solve_articulation_contact_last',False) and order_after is not True:
            raise RuntimeError('Physics reset changed requested contact solver order')
        report['native_diagnostics_settings']={k:process_settings.get(k) for k in (
            '/physics/enableSynchronousKernelLaunches','/physics/exposeProfilerData',
            '/persistent/physics/pvdEnabled','/physics/omniPvdOutputEnabled','/physics/omniPvdIsRecording',
            '/physics/physxDispatcher','/physics/updateVelocitiesToUsd')}
        effective=report['effective_scene_after_reset']
        if (effective['solver']!=args.solver or effective['gpu_dynamics'] or effective['friction_type']!='patch'
                or not np.isclose(effective['gravity_m_s2'],args.gravity,rtol=1e-6,atol=1e-8)
                or not np.isclose(effective['physics_dt'],1/args.physics_hz,rtol=1e-6)):
            raise RuntimeError('Physics reset changed explicit configuration')
        sim.step(render=False)
        runtime=PlantRuntime(rig,sim.physics_sim_view)
        report['native_drive_parameters']=runtime.drive_diagnostics
        springs=None
        if native_hold or args.native_spring_cut_trial:
            from .native_spring_observer import NativeSpringObserver
            springs=NativeSpringObserver(runtime.articulation,allow_release=args.native_spring_cut_trial)
            report['spring_control']=dict(mode='native_drives_isolated_cut_comparison' if args.native_spring_cut_trial else 'native_drives_isolated_hold_comparison',
                external_spring_effort_applied=False,native_drive_readback_verified=True,
                drive_work_measured=False,cutting_qualified=False)
        if args.spring_mode=='implicit_effort':
            from .implicit_springs import ImplicitJointSprings,NativeBodyLoads
            if args.native_torsion_trial:
                from .split_springs import SplitJointSprings
                springs=SplitJointSprings(runtime.articulation)
                report['spring_control']=dict(springs.receipt)
            else:
                springs=ImplicitJointSprings(runtime.articulation)
            loads=NativeBodyLoads(runtime.articulation,[0,0,-args.gravity]) if args.force_newton else None
            if loads:
                report['force_mapping']=dict(reference=loads.reference,gravity_relative_errors=loads.reference_errors)
        if fixture is not None:
            if args.scene_profile:
                from .scene_profile import run as profile_scene
                report.update(profile_scene(sim,rig,fixture,output))
                (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
                return 0
            from .gripper_probe import run
            if args.bimanual_cut:
                from .bimanual_probe import run
            if args.watch_cut_trial:
                from .cut_watch import watch
                report.update(watch(app,sim,rig,runtime,springs,fixture,args,output,run))
            elif args.robot_interactive:
                from .full_robot import interactive
                report.update(interactive(app,sim,rig,fixture,args,output))
            else:
                if args.profile:
                    import cProfile
                    profile=cProfile.Profile();profile.enable()
                    try: report.update(run(app,sim,rig,runtime,springs,fixture,args,output))
                    finally:
                        profile.disable();profile.dump_stats(str(output/'profile.pstats'))
                else: report.update(run(app,sim,rig,runtime,springs,fixture,args,output))
            report['source_assets_unchanged']=all(sha256_file(path)==h for path,h in source_hashes.items())
            if not report['source_assets_unchanged']: raise RuntimeError('Source asset changed during probe')
            (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
            print('GRIPPER_PROBE_RESULT '+json.dumps({k:report.get(k) for k in ('state','gates','measurements')}),flush=True)
            return report_exit_code(report,args)
        if args.interactive:
            from .demo import run
            report['state']='interactive_demo_not_qualification'
            (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            report['demo']=run(app,sim,rig,runtime,springs,args,output)
            report['source_assets_unchanged']=all(sha256_file(path)==h for path,h in source_hashes.items())
            if not report['source_assets_unchanged']: raise RuntimeError('Source asset changed during demo')
            (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            return 0
        frames,initial_velocity=runtime.sample()
        report['initial_body_state']=dict(paths=rig.body_paths,frames=frames.tolist(),
            rest_frames=rig.rest_frames.tolist(),velocities=initial_velocity.tolist(),
            masses=[p['mass'] for p in rig.properties],
            inertia=[p['inertia'].tolist() for p in rig.properties])
        clock=PhysicsClock(sim,physics_hz=args.physics_hz,render_hz=args.render_hz)
        samples=[];events=[];start=time.perf_counter()
        release_step=int(args.seconds*(1 if args.attached_only else .75)*args.physics_hz)
        def before(stamp,dt):
            t=stamp.simulation_time_s
            if stamp.step==release_step: events.append(rig.diagnostic_release())
            force=[0,args.force_newton,0] if 1<=t<2 else [0,0,0]
            runtime.apply_tip_force(force)
            if springs is not None:
                springs.step(dt,loads.at_com(rig.body_paths[-1],force) if loads else None,
                             root_constrained=not rig.cut)
        def after(stamp,dt):
            frames,velocity=runtime.sample()
            if np.linalg.norm(velocity[:,:3],axis=1).max()>20:
                report['failed_body_state']=dict(step=stamp.step,frames=frames.tolist(),velocities=velocity.tolist())
                raise RuntimeError('Unstable plant velocity; qualification stopped')
            samples.append(dict(t=stamp.simulation_time_s,tip=runtime.tip().tolist(),
                dof_positions=runtime.articulation.get_dof_positions().tolist() if runtime.articulation is not None else None,
                root_rotation=frames[rig.cut_index,:3,:3].tolist(),
                contact_force_n=float(np.linalg.norm(runtime.contacts.get_net_contact_forces(dt),axis=1).max()),
                max_speed=float(np.linalg.norm(velocity[:,:3],axis=1).max()),
                support_error=float(np.max(np.linalg.norm(frames[:rig.cut_index,:3,3]-rig.rest_frames[:rig.cut_index,:3,3],axis=1))),
                anchor_gap=float(np.linalg.norm(
                    frames[rig.cut_index,:3,3]-frames[rig.cut_index,:3,2]*np.linalg.norm(rig.chain_world[rig.cut_index+1]-rig.chain_world[rig.cut_index])/2
                    -rig.chain_world[rig.cut_index]))))
        print('PHYSICS_QUALIFICATION_READY '+json.dumps(rig.report()),flush=True)
        for step in range(int(args.seconds*args.physics_hz)):
            if not app.is_running(): raise RuntimeError('Kit closed before bounded qualification completed')
            clock.tick(before=before,after=after,before_render=lambda stamp:runtime.sync_visuals())
            if step%args.physics_hz==0: print('PHYSICS_SECOND',step//args.physics_hz,flush=True)
        runtime.sync_visuals()
        report['timing']=clock.report();report['run_wall_seconds']=time.perf_counter()-start
        report['events']=events
        report['measurements']=dict(max_speed_m_s=max(s['max_speed'] for s in samples),
            max_native_contact_force_n=max(s['contact_force_n'] for s in samples),
            max_support_error_m=max(s['support_error'] for s in samples),
            max_attached_anchor_gap_m=max(s['anchor_gap'] for s in samples[:release_step]),
            released_anchor_gap_m=samples[-1]['anchor_gap'])
        tips=np.asarray([s['tip'] for s in samples])
        baseline=tips[int(.9*args.physics_hz)]
        report['measurements']['max_attached_tip_displacement_m']=float(np.linalg.norm(
            tips[:release_step]-rig.chain_world[-1],axis=1).max())
        report['measurements']['force_pulse_tip_change_m']=float(np.linalg.norm(
            tips[int(1.9*args.physics_hz)]-baseline))
        report['measurements']['recovery_tip_residual_m']=float(np.linalg.norm(
            tips[int(2.9*args.physics_hz)]-baseline))
        if args.attached_only:
            tail=samples[-int(.5*args.physics_hz):]
            report['measurements']['settled_tip_displacement_m']=float(np.linalg.norm(tips[-1]-rig.chain_world[-1]))
            report['measurements']['tail_tip_variation_m']=float(np.linalg.norm(np.ptp(tips[-len(tail):],axis=0)))
            report['measurements']['tail_max_body_speed_m_s']=max(s['max_speed'] for s in tail)
        # Reset is a separate parse boundary; no stale tensor handles reused.
        sim.stop();rig.restore_authored_state();sim.reset();sim.step(render=False)
        restored=PlantRuntime(rig,sim.physics_sim_view);frames,_=restored.sample()
        report['reset_max_body_error_m']=float(np.max(np.linalg.norm(frames[:,:3,3]-rig.rest_frames[:,:3,3],axis=1)))
        if springs is not None:
            report['reset_controller_parameters_restored']=bool(
                np.allclose(restored.drive_diagnostics['stiffness'],report['native_drive_parameters']['stiffness'])
                and np.allclose(restored.drive_diagnostics['damping'],report['native_drive_parameters']['damping']))
            reset_springs=ImplicitJointSprings(restored.articulation)
            replay_errors=[]
            for i in range(int(.5*args.physics_hz)):
                restored.apply_tip_force([0,0,0])
                reset_springs.step(1/args.physics_hz,root_constrained=True)
                sim.step(render=False);restored.sample()
                replay_errors.append(float(np.linalg.norm(restored.tip()-tips[i])))
            report['reset_replay_tip_error_m']=max(replay_errors)
        sim.stop()
        report['source_assets_unchanged']=all(sha256_file(path)==h for path,h in source_hashes.items())
        report['gates']=dict(finite_bounded_motion=report['measurements']['max_speed_m_s']<20,
            fixed_support=report['measurements']['max_support_error_m']<1e-5,
            connected_before_release=report['measurements']['max_attached_anchor_gap_m']<.005,
            bounded_attached_deflection=report['measurements']['max_attached_tip_displacement_m']<.08,
            reset_restored=report['reset_max_body_error_m']<.005,
            source_assets_unchanged=report['source_assets_unchanged'])
        if args.force_newton>0:
            report['gates']['responds_to_force']=report['measurements']['force_pulse_tip_change_m']>.0001
        if args.attached_only:
            report['gates']['settles_after_load']=(report['measurements']['tail_tip_variation_m']<.001
                and report['measurements']['tail_max_body_speed_m_s']<.01)
        if springs is not None:
            # A pulse is a known external load, not a contact qualification.
            report['gates']['no_unqualified_contact_loads']=report['measurements']['max_native_contact_force_n']<1e-5
            report['gates']['reset_controller_replay']=(report['reset_controller_parameters_restored']
                and report['reset_replay_tip_error_m']<.001)
        if not args.attached_only:
            report['gates']['detached_after_release']=report['measurements']['released_anchor_gap_m']>.02
        report['qualification_scope']='attached_only' if args.attached_only else 'attached_and_diagnostic_release'
        report['state']='passed_mechanism_qualification_not_robot_task' if all(report['gates'].values()) else 'failed_qualification'
        (output/'trajectory.json').write_text(json.dumps(samples,allow_nan=False),encoding='utf-8')
        (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
        print('PHYSICS_QUALIFICATION_RESULT '+json.dumps(report),flush=True)
        return 0 if all(report['gates'].values()) else 2
    except Exception:
        report['state']='error';report['error']=traceback.format_exc()
        if 'samples' in locals():
            (output/'trajectory.json').write_text(json.dumps(samples,allow_nan=False),encoding='utf-8')
        (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        raise
    finally:
        if cylinder_setting is not None:
            if previous_cylinders is None:process_settings.destroy_item(cylinder_setting)
            else:process_settings.set(cylinder_setting,previous_cylinders)
        if args.no_physics_profiler:
            if previous_profiler is None: process_settings.destroy_item(profiler_setting)
            else: process_settings.set(profiler_setting,previous_profiler)
        if args.physics_threads is not None:
            if previous_threads is None: process_settings.destroy_item(thread_setting)
            else: process_settings.set(thread_setting,previous_threads)
        # Fast Kit shutdown otherwise exits with zero even after an exception.
        app.close(exit_code=report_exit_code(report,args))


if __name__=='__main__':
    raise SystemExit(main())
