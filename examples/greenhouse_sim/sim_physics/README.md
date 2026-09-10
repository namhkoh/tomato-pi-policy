# Current-package physics qualification (experimental)

This is an opt-in engineering harness on `koh-dev/sim-vlm`, not a replacement
for the static dataset collector, and not yet a validated robot manipulation
environment. Do not collect training demonstrations from this harness.

## Latest full-robot checkpoint

The complete v1.2 RB-Y1 is now exercised through native joint drives, with
an IK side approach and force-limited left fingers. Run the persistent window:

```powershell
examples\greenhouse_sim\run_full_robot_grasp_demo.cmd --output ..\..\data\sim_physics\my_full_robot_demo
```

Choose a new output directory. A single trial starts automatically; use
**Run grasp + 10 mm movement** to replay. Full robot, grasp close-up and
mounted head/left/right D405 views are available. The source robot and plant
assets are unchanged. The right arm/knife remains parked.

Native trial `full_robot_demo_20260910_02/trial_001` passed the bounded
grasp/move/reset gates: 8.884 mm target follow for a 10 mm command,
1.542 mm maximum slip and 0.401 mm maximum penetration. The user confirmed
the visible grasp. This is one privileged fixture case, NOT a full deleafing
task, calibrated grasp reliability or approved training experience.

Important integration findings: a kinematic palm cannot report contact with
fixed neighboring foliage. Full robot dynamics exposed those overlaps, so the
approach was changed to side entry. Eight explicitly documented mounting-proxy
pair exclusions are used for this fixed-torso diagnostic; remaining self and
plant collisions stay active. Arbitrary torso/tool movements need collision
model qualification. Replay camera setup is now idempotent.

The sections below preserve earlier qualification results and failure evidence.

## Implemented

- Adapt one intact, leaf-bearing petiole from the supplied package. Keep its
  original textured geometry and leaf meshes; do not substitute a procedural vine.
- Session-layer-only capsule bodies with reduced-coordinate bending/torsion
  joints, a fixed parent support, and an external separable joint at exactly
  10 mm along the material centerline.
- Stem stiffness `EI/L`, torsional stiffness `GJ/L`, SI mass and inertia.
  Leaf mass is an uncalibrated prior distributed over actual triangle area;
  composite COM and principal inertia include the parallel-axis terms.
- Native contact-report APIs. Disable only the original colliders that the
  target adapter replaces. Visibility alone does not remove duplicate contacts.
  Other plant colliders remain active.
- Batched native body state/force access and precomputed visual skinning.
  Visual skin weights do not cross the severance seam. Rendering has its own
  cadence; control does not perform filesystem writes or full-stage traversal.
- Shared explicit physics clock, bounded timing statistics, fail-closed
  qualification gates, diagnostic seam release, and stop/reparse/reset.

Material constants are engineering priors, **not calibrated tomato tissue**.
Leaves are rigid laminas with convex contacts, not deformable shells. The parent
plant is fixed in this increment. The cut seam is uncapped. Disabling its joint
is **not force/direction-verified blade cutting or simulated tissue fracture**.
No robot grasp, slip, retention, deposit, damage or dynamic sensor synchronization
is certified by this harness.

## Run a bounded probe

For a persistent **visible isolated demo**, use a new output directory:

```powershell
examples\greenhouse_sim\run_physics_demo.cmd --output ..\..\data\sim_physics\my_visible_demo
```

The `Plant Physics - experimental demo` panel provides a six-second pull/recover
cycle, pause/resume, diagnostic seam release and reset/reattach. Pull/recover
initially repeats; uncheck the repeat box for manual inspection. Release stops
after 0.3 seconds of free fall, so the branch may leave the close-up; Reset
restores it. Only the target petiole and its leaf carriers are dynamic. This is
not the full greenhouse, a robot-camera capture, mouse grasp or blade-cut demo.
Use this panel instead of Kit's timeline controls. Native viewport snapshots
are diagnostic evidence only, never dataset input or replacement depth.

The first visible run is `data/sim_physics/demo_20260910_01`: `settled.png`,
`pulled.png`, `recovered.png`, first-cycle native tip trace and `demo_status.json`.
All four capture waits verified zero physical pose change while rendering.
Actual rendered tick throughput was approximately 0.62-0.68x real time in the
first cycles (excluding deliberate playback pacing/capture waits); the faster
headless result must not be presented as GUI performance.

From the repository root (choose a new output directory):

```powershell
examples\greenhouse_sim\run_physics_qualification.cmd --output ..\..\data\sim_physics\my_probe --spring-mode implicit_effort --solver PGS --physics-threads 1 --fabric --seconds 6 --render-hz 0
```

Default scene: the supplied `seed101_full` plant, with `SubStem_41` converted
to physics. A 20 mN diagnostic force pulse is applied from 1 to 2 seconds,
followed by an explicitly diagnostic joint release at 75% of the run.
There is no floor in the isolated probe; free fall after release is not deposit.
Gravity-free runs are diagnostic comparisons, not Earth-gravity qualifications.

`--scene package` is an integration scaffold for the greenhouse and the static
v1.2 robot/cameras; it is not yet qualified. It does not restore robot dynamics.
Do not launch a second full package instance under memory pressure.

Reports distinguish requested and effective scene settings: SimulationContext
must use `set_defaults=False` or it overwrites authored gravity and solver
choices. Controller step counters are not native sensor frame identifiers.
The existing static collector still obtains metric depth from Replicator's
`distance_to_image_plane`; this module creates no replacement depth.

## Evidence and remaining gates (2026-09-10)

- `data/sim_physics/joint_probe_20260910_02.json` and `..._03.json`:
  small D6/revolute and external-attachment probes match the analytical
  gravitational deflection within 0.001 rad, including mass/stiffness scaling.
  These validate elementary drive behavior, not the whole plant.
- `qualification_20260910_01` through `_17`: retained diagnostic failures. The original
  duplicate source colliders caused explosive startup; removing those duplicates
  resolved that particular overlap, but the longer chain still fails bounded
  deflection/stability under load. Do not claim complete physics restoration.
- Runs `_11` and `_12` have invalid requested-vs-effective solver/gravity
  comparisons because the context overwrote the requested settings. Run `_13`
  is the first gravity-free run with an effective-configuration check.
- Run `_14` completed 6 s at 480 Hz/PGS, but failed the 80 mm deflection gate
  (513 mm measured). Reset error was 0.264 mm; source hashes were unchanged.
  Its 0.60 real-time factor is physics/diagnostics only, with no rendering.
- Run `_15` at 1920 Hz still produced 515 mm attached deflection; higher
  frequency alone did not fix the loaded chain. Run `_16` tested regular
  maximal-coordinate D6 joints and still produced 480 mm deflection.
- An explicit world-fixed articulation comparison (`_17`) removes root
  attachment compliance, but the original petiole still exceeded the
  velocity limit. Use `--constraint-mode fixed_articulation --attached-only`
  only for diagnosis. Seam release is rejected in this mode until a
  state-preserving topology transition is implemented and validated.
- Joint probes `_04` through `_12` reproduce an orientation/inertia-sensitive
  error without plant geometry or collisions. Fixed-base authoring improves
  angular accuracy, but does NOT establish a settled solution: probe `_12`
  measures 0.263 rad/s residual generalized velocity despite only
  0.000083 rad angular error. Qualification now requires both angle accuracy
  and low final-half-second RMS velocity. The rotated cases still fail.
- Joint probe `_11` was requested as a GPU comparison but its recorded
  effective setting is CPU. Do not interpret it as GPU evidence. The probe
  now rejects that mismatch after reset. No valid GPU comparison is claimed.

These failures do not yet distinguish every engine/numerical effect from
model-authoring issues. No calibrated material, arbitrary mass inflation,
stiffness multiplier, or successful robot manipulation is claimed.

The failures above describe the native-drive reference, which remains the CLI
default. The explicit `implicit_effort` option in the command above is a new,
limited qualification path; see the later checkpoint below.

Next: demonstrate timestep/mesh convergence for the loaded chain;
then test full-scene rendering/contact performance, dynamic native RGB-D
synchronization, robot contact/grasp/slip and a contact-verified cut state machine.
Only after grasp, cut, retain, deposit and recovery/reset pass should dynamic
action-outcome dataset collection begin.

Native unit/constraint reference:
[Omni Physics joints](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.1/dev_guide/rigid_bodies_articulations/joints.html).
This implementation uses D6 joints with locked translation; it does not rely
on unsupported drives on a USD SphericalJoint.

## Coupled elastic effort checkpoint (2026-09-10, runs 19-28)

`implicit_springs.py` uses the native articulation mass matrix in a frozen-matrix
backward-Euler spring/damper solve, with native drive gains disabled at runtime:

```
(M + h*C + h*h*K) v_next = M*v + h*(external - K*q)
tau = -K*(q + h*v_next) - C*v_next
```

PhysX still integrates motion and constraints. This does not add mass, teleport
bodies or establish unconditional nonlinear/contact stability. Gravity and
Coriolis terms come from native readback. A known diagnostic force is applied at
the last body's COM and projected through its native COM Jacobian; the Jacobian
reference is checked independently against native gravity compensation. While
attached, the root is constrained in the solve. After diagnostic release, the
floating root is included without applying artificial root spring forces.
Unknown contact loads are not qualified: these probes fail if native contact
forces exceed 1e-5 N. A no-contact pass is NOT a contact-response pass.

- `_21`, fixed root, 240 Hz: 20 mN force pulse produces 21.65 mm tip change;
  recovery residual at the measured checkpoint is 1.08 mm. Steady gravity sag
  is 38.86 mm. `_22` at 480 Hz gives 21.99 mm pulse change and 38.86 mm sag.
- `_28`, separable external attachment: all bounded-motion, support, connection,
  force-response, diagnostic detach and reset gates pass. Pulse change is
  21.75 mm; recovery residual is 1.08 mm. Reset restores controller parameters
  and exactly reproduces the first 0.5 s of the measured tip trace. This is
  joint release followed by free fall, not blade cutting, retention or deposit.
- `_24` versus `_25`: disabling per-tick USD pose writes via Fabric preserves
  the exact trajectory-file SHA256 while reducing headless tick wall time from
  5.63 s to 3.85 s per 6 simulated seconds. `_28` measures 3.82 s (~1.57x real
  time). These are single-petiole CPU measurements, not rendered greenhouse FPS.
- Mesh refinement remains incomplete: segment limits 25 / 12.5 / 6.25 mm
  (`_21`, `_26`, `_27`) give steady sag 38.86 / 40.83 / 41.90 mm, but pulse
  change 21.65 / 24.10 / 26.86 mm and recovery residual 1.08 / 2.75 / 5.10 mm.
  Transient response is mesh-sensitive; the per-joint damping prior requires
  further investigation. No material calibration or full convergence is claimed.
- Regression checkpoint before the last reset-replay addition: 629 tests and
  47 subtests passed. Run `_28` subsequently validates native reset replay.

Evidence: `data/sim_physics/qualification_20260910_<number>/report.json` and
`trajectory.json`. All artifacts remain non-training. Original source hashes
are unchanged. Main simulator integration, robot contacts, grasp/slip, cutter
contact/direction/force checks and moving-scene native RGB-D synchronization
remain required gates. No collection or training was started.

## Left-gripper contact tests (2026-09-10)

The new `--gripper-probe` uses the **actual v1.2 left gripper USD** (palm,
fingers, fitted wrist-camera geometry and existing collision proxies). The palm
is a prescribed kinematic fixture; the two fingers have dynamic, force-limited
prismatic joints. It does NOT execute full-arm IK or claim full-robot collision
clearance. No artificial fixed grasp joint or plant-pose following is used.
The 0.5 N drive limit per finger and friction 0.5 are engineering test choices,
not calibrated hardware/tissue parameters. Leaves remain rigid convex laminas.

From repository root, with a NEW output directory:

```powershell
examples\greenhouse_sim\run_physics_qualification.cmd --output ..\..\data\sim_physics\my_grasp_probe --gripper-probe --spring-mode implicit_effort --solver PGS --physics-threads 1 --fabric --seconds 7 --force-newton 0 --grasp-arc-m .08 --render-hz 30
```

Add `--gui` for a visible **bounded** run (it closes afterward). This does not
take over the persistent pull-demo window. The requested 80 mm grasp snaps to
the nearest physical segment centre, **71.13 mm** for this target, not the
10 mm cut seam. It is privileged diagnostic placement, not a VLM-executed goal.

| Run under `data/sim_physics/` | Measured result |
|---|---|
| `gripper_20260910_01` | Approach to the 117.8 mm grasp location hits an off-shaft leaf-carrier surface first. Velocity reaches 29.58 m/s and the probe stops before grasping. General leaf-contact stability is NOT solved. |
| `gripper_20260910_02` | Clearer 71.13 mm shaft location passes the initial grasp/move gates. A 10 mm commanded palm translation gives 9.187 mm target displacement and maximum 1.207 mm slip. Maximum measured penetration is 0.465 mm. |
| `gripper_20260910_03` | Matched **zero-friction negative control**: opposing finger loads remain, but target motion is -0.006 mm and slip is 10.006 mm. Correctly fails follow/slip gates; no artificial weld hides this failure. Reset replay error is zero. |
| `gripper_20260910_04` | Adds diagnostic seam release at 4.75 s. Hold fails the proposed 3 mm slip gate: maximum slip 11.188 mm in 4.8-5.4 s; opposing load criterion met in only 82.1% of those samples. The branch remains between the fingers in the saved 5.3 s view, but this is NOT a reliable, low-slip hold. Reset replay error is zero. |
| `gripper_20260910_05` | Headless repeat of the positive shaft test passes, including reset/replay. Its full recorded physical trajectory equals rendered run `_02` exactly. This is deterministic repeatability for one case, not a grasp success-rate estimate. |

Each run saves `report.json` and `gripper_trajectory.json`; rendered runs also
save native viewport PNGs. Inspected `_02/closed.png` and
`_04/hold_after_diagnostic_release.png`. These are diagnostic views, not mounted
robot-camera data. No dynamic training episodes are approved by these tests.
The no-contact gate on the original pull demo is unchanged; this contact fixture
has separate explicit bounds and grip/slip criteria. The elastic predictor does
not yet include unknown contact loads; native PhysX resolves contact itself.

Remaining: stable off-shaft/leaf interactions, robust post-severance hold,
contact-point/force feedback, pad/tissue calibration and penetration convergence,
safe full-arm approaches, blade cutting and synchronized robot-camera evidence.
Native startup also warns about the open prismatic joint anchor offsets; the
recorded fingers retain their open spacing, but joint initialization deserves
explicit qualification before full integration. Do not suppress the warning or
interpret the fixture pass as a complete bimanual robot-task pass.
