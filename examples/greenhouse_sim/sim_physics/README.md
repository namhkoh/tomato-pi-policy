# Current-package physics qualification (experimental)

This is an opt-in engineering harness on `koh-dev/sim-vlm`, not a replacement
for the static dataset collector, and not yet a validated robot manipulation
environment. Do not collect training demonstrations from this harness.

## Experimental knife integration, September 11

`run_bimanual_cut_probe.cmd --output <new-directory>` is an **unqualified**
bounded integration test, not a working end-to-end demonstration. The existing
grasp-demo launchers remain unchanged. Add `--gui --robot-interactive` only for
diagnostic replay; a failed plan/guard never forces release.

- `knife.py` corrects the original right knife in the session layer. Its old
  blade occupied EE +Z=0..71.48 mm, overlapping the wrist's +Z=0..46.5 mm.
  The 180-degree EE-Y correction places the same blade/support along distal
  -Z, retains the flat +Y cutting direction and leaves source meshes unchanged.
  This is geometric flange alignment, not a new CAD fastener certification.
- `bimanual.py` caches source knife transforms, reads native right-wrist state
  and contact positions, and searches bounded IK/edge-wing alternatives. Arm
  capsule clearance is only one screen, not full-tool/path certification.
- `startup_screen.py` rejects possible initial robot/scene collision overlaps
  before starting dynamics. Static triangle/quad surfaces refine broad boxes;
  convex solids retain containment checks. Hidden and instanced collisions
  count. Runtime native force, speed, tracking, slip and window guards remain.
- `bimanual_probe.py` gates right motion on opposing left stem contact. Only
  the actual flat leading strip on the selected seam-adjacent stem capsules
  can authorize `PlantRig.release_from_blade`. Arc, camera, neighbors, broad
  blade-face impacts, stationary force and separate taps cannot trigger it.
- The current shear prior is 0.2 N sustained for 25 ms plus 0.3 mm measured
  relative loading travel, bounded by 0.5 N tool force and 3 mm grasp slip.
  These are **unvalidated engineering thresholds**, not tomato tissue data.
  Release disables one preauthored 10 mm joint; it is not continuum fracture,
  arbitrary-position mesh cutting, or verified physical deleafing success.
- Retention/separation and withdrawal are separate measured gates after
  release. Deposit, recovery, synchronized dynamic camera records and VLM/VLA
  execution are not implemented by this probe. No result is training-approved.

Measured current blocker: `data/sim_physics/bimanual_cut_20260911_03` passed
the spawn screen and reached the left grasp (all 120 measured hold frames had
opposing stem contact). At 3.5 s, the right IK/arm-clearance search failed and
the test stopped without a cut. An earlier opposite-side candidate in
`bimanual_cut_20260910_01` collided at spawn; it is rejected, not a demo setup.
Do not report a completed grasp-cut-retain sequence from either run.

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

## Full robot in the supplied greenhouse (2026-09-10)

The isolated, user-confirmed full-robot checkpoint is commit `ea168ab`.
The new greenhouse diagnostic uses the actual supplied
`data/sim_data/package_20260905/tomato_greenhouse_pack/house/green_house_base.usd`.
All 75 gutters, the building and floor retain their original geometry and
collisions. One detailed original plant occupies the same station as the
package preview; four distant same-gutter instances provide context.
Only the selected petiole/leaf carriers are dynamic in this increment.
This is not the full populated-greenhouse collector.

From repository root (choose a NEW output):

```powershell
examples\greenhouse_sim\run_greenhouse_physics_demo.cmd --output ..\..\data\sim_physics\my_greenhouse_grasp
```

The persistent window has Run / Stop / Reset, full-robot and grasp views,
and the existing head/left/right D405 views at 848x408. The right arm/knife
remains parked. The left arm performs an IK approach, physical finger closure,
a grasp-gated 10 mm movement, hold and opening. Targets come from privileged
fixture geometry, not observation-verified perception. No hardware commands.

- Robot base sits 1 mm above the original local floor surface (0.101 m), with
  upright torso; no artificial ground or plant-height shortcut.
- Sparse native contact events cover self, target and environment contacts;
  only the designated fingers on the target and chassis/wheels on the floor
  are permitted. Existing eight mounting-proxy exclusions remain unchanged
  and are not qualified for arbitrary torso motion.
- Finger gravity compensation shares the original 0.5 N total actuator budget.
  In this near-vertical jaw orientation, ignoring approximately 0.31 N of
  finger weight produced uneven contact. No grasp weld or force-limit increase.
- The tested 10-degree wrist-entry tilt avoids the camera/leaf contact seen
  with the original entry. It is a tested fixture pose, not a general-purpose
  collision-certified planner.
- `greenhouse_physics_20260910_06` passes all 1,680 steps and reset/replay:
  target motion **7.825 mm**, maximum slip **2.489 mm**, maximum penetration
  **0.370 mm**, maximum gripper net contact **0.233 N**. One case only.
- Failed variants `_01` through `_05` are retained. They include insufficient
  opposing contact, unsuccessful pressure/centering experiments (removed),
  camera/leaf contact, and a rejected wrist approach hitting fixed foliage.
- Native wide and close-up PNGs are diagnostic evidence, not dataset images.
  Demo lighting is reduced by three exposure stops in the session layer to
  avoid saturation; source lighting and dataset capture settings are unchanged.

Performance remains a gate: the initial rendered greenhouse trial was about
0.11x real time, with a median native step near 34 ms. Adding CPU threads and
removing zero-load floor-warning spam did not resolve this. Do not advertise
real-time performance or extrapolate an isolated headless result to this scene.
Bounded tests accept `--profile` for call timing. A dedicated PhysX dispatcher
experiment (`_07`) stalled at initialization and was stopped; that experimental
option was removed. The known-working dispatcher remains unchanged.

Next gates are wider contact/approach coverage, performance, post-severance
retention, verified blade cutting and deposit, followed by synchronized native
robot-camera RGB-D action/outcome records. Material parameters and force limits
still require calibration. No dynamic training eligibility is implied.

## Dense context and latency optimization (2026-09-10)

The greenhouse demo launcher now adds `--batch-gutter-visuals`,
`--local-wire-physics --context-gutters 3` and `--no-capture-milestones`.
The command above still works with a new output directory. Override
`--context-gutters 1` or `5` for alternate bounded layouts; only three rows
have been exercised in the native dense-context grasp test at this checkpoint.
This restores the original preview's **144 plant positions**, not plants on
all 75 greenhouse gutters. All 75 gutters remain visible.

- Batched 3,525 identical, non-physical gutter modules into one USD point
  instancer. Original geometry/materials/transforms remain; all 75 gutter
  collision proxies remain at their source coordinates. Animated, mixed-
  prototype or physical modules are rejected by the batching adapter.
- Kept 12 wire collision proxies intersecting a fixed 4x4 m XY workspace;
  deactivated 7,038 unreachable **guide-purpose** proxies, not rendered wires.
  Full collision bounds decide membership, not visibility or prim origins.
  Every robot and dynamic plant collision sphere is checked each step with a
  15 cm boundary margin. This is fixed-base only: rebuild the collision window
  before moving the base, changing stations or promoting another plant.
- Restored 143 supplied backdrop instances plus the detailed target. In the
  tested three-row layout, 71 nearby backdrop plants receive static native
  triangle-mesh collisions, while 72 distant ones are visual-only instances.
  Anonymous prototype stages keep source USDs immutable. Nearby context is
  **not compliant**; selected petiole/carriers remain the only dynamic plant.
- Preserved gravity, 240 Hz solver step, finger effort/friction, collision
  guards, IK sequence and all head/wrist camera view buttons. This does not
  qualify contact with every leaf or implement full-robot cutting.
- Interactive milestone PNG capture is off by default to avoid repeated
  diagnostic pauses. The new UI checkbox enables it for the next trial, or
  pass `--capture-milestones`. Bounded qualification still captures by default.
  Captures are native viewport images, not synchronized training RGB-D.

Measured on this workstation, one target, PGS/240 Hz, one physics worker:

| Run under `data/sim_physics/` | Median native step | Rendered tick throughput |
|---|---:|---:|
| Original `greenhouse_physics_20260910_06` | 34.20 ms | 0.108x real time |
| Batched, sparse `greenhouse_opt_20260910_06` | 5.99 ms | 0.446x |
| Batched, 144 plants `greenhouse_dense_20260910_01` | 6.82 ms | 0.411x |
| Dense without Fabric `greenhouse_dense_20260910_02` | 9.10 ms | 0.331x |
| Normal step dense repeat `greenhouse_dense_20260910_03` | 6.75 ms | 0.421x |
| Visible GUI `greenhouse_optimized_demo_20260910_01/trial_001` | 8.36 ms | 0.345x |

All three optimized runs passed the limited grasp/reset gates. Their entire
1,680-step physical records equal the original trajectory exactly. Native
wide images from the sparse/dense runs were inspected. Tick throughput excludes
startup, paused milestone snapshots and post-trial reset; it is not total
job throughput, camera FPS, an average over targets, or a real-time claim.
The existing once-per-trial IK replan still costs about 0.9 s.

Causal root-removal controls are saved separately as
`greenhouse_scene_profile_20260910_01` through `_04`. Removing only the repeated
gutter visual modules reduced native step time to about 10 ms even with wire
proxies retained. Removing disabled rigid-body APIs or profiling instrumentation
did not help. Disabling far wire CollisionAPI values alone did not help;
deactivating unreachable guide prims did. These ablations are **not qualifying
physics runs** and must never be presented as successful demos.

Opt-in diagnostics: `--step-profile` splits the installed Isaac physics-only
step path; `--scene-profile` performs non-qualifying root/module removal timing
controls in a bounded headless process and restores session state. All new
behavior is isolated from the static dataset capture/review/export pipeline.

Remaining runtime warning: Isaac Fabric reports a point-instancer prototype
mismatch on reset, even though the static gutter visuals render and physical
reset replay matches. It is not suppressed or treated as proof of sensor
synchronization. Additional native render/reset verification is required
before using this representation for dynamic dataset capture.
The native `greenhouse_dense_20260910_03/reset_replay.png` was inspected and
retains the visible robot, vines and gutters. The relaunched GUI passes the
grasp/reset gates with automatic milestone capture disabled.

For the smallest VLM + low-level-controller experiment and its explicit
unimplemented gates, see [PROOF_OF_LIFE.md](PROOF_OF_LIFE.md).
