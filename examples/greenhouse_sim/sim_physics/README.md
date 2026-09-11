# Current-package physics qualification (experimental)

This is an opt-in engineering harness on `koh-dev/sim-vlm`, not a replacement
for the static dataset collector, and not yet a validated robot manipulation
environment. Do not collect training demonstrations from this harness.

## Experimental knife integration, September 11

Latest checkpoint: the shared left-grasp corridor screen now checks open-jaw
approach and <=1 mm closure samples against cached native plant/scene geometry.
It permits only finger contact with the selected detachable shaft and its
immediate capsule neighbors, never an intervening leaf. Native bilateral
selected-shaft contact remains the authority for grasp success. This caught
the distal leaf obstruction in `bimanual_cut_20260911_29`.

`bimanual_hold_control_20260911_08` passed 65 corridor checks and held the
selected stem for 20 simulated seconds (4,800 ticks), no fault, max slip
0.005253 mm. It deliberately does not move/cut with the right arm and exits
with false cutting gates, not a successful full-sequence claim. Its headless
tick RTF was 0.5146 under concurrent offline work, not a GUI-speed result.
Vectorized mesh expansion and conservative triangle indexing avoid repeated
large-mesh narrow-phase work without removing geometry or reducing margins.
**No successful current-environment grasp-cut-retain-deposit run yet.**

Cut evidence now measures net advance since the first consecutive qualified
contact sample. Free approach motion and positive-only sums of tiny contact
jitter cannot satisfy loading travel. This fixes evidence accounting; the
rigid seam's physical ability to satisfy the force/travel criterion is still
unqualified. A force-limited native loading diagnostic is required before
calling this a validated cutting model. `RigidToolScreen` can cheaply reject
incompatible wrist-tool subsets before IK, but does not certify an arm path
and uses conservative source-derived bounds, not exact mesh collision proof.

`run_bimanual_cut_probe.cmd --output <new-directory>` is an **unqualified**
bounded integration test, not a working end-to-end demonstration. The existing
grasp-demo launchers remain unchanged. Add `--gui --robot-interactive` only for
diagnostic replay; a failed plan/guard never forces release.

- `knife.py` corrects the original right knife in the session layer. Its old
  blade occupied EE +Z=0..71.48 mm, overlapping the wrist's +Z=0..46.5 mm.
  The distal correction keeps the same blade/support along -Z. The subsequent
  user-requested 180-degree roll about wrist Z reverses the previous mounting:
  flat edge toward wrist -Y, curved support on +X. It does NOT flip back into
  the wrist. Visuals/colliders/edge rotate together; source meshes are unchanged.
  This is geometric flange alignment, not a new CAD fastener certification.
- `bimanual.py` caches source knife transforms, reads native right-wrist state
  and contact positions, and searches bounded IK/edge-wing alternatives. Arm
  capsule clearance is only one screen, not full-tool/path certification.
- `blade_contacts.py` replaces the enclosing blade box with two source-triangle
  convex hulls in the session layer. Cross-sections of the supplied long plate
  are Z=-6.5..-0.5 mm (6 mm thick), not the mounting-end box's -6.5..+6.5 mm.
  Its usable long edge is centred at Z=-3.5 mm and follows the CAD's 2-degree
  slant. The thick mount still collides. Visible source meshes are untouched.
- `held_plant_screen.py` snapshots actual native held-body poses and caches
  nearby static contacts, including hidden/instanced geometry, after scene
  population. Right-arm paths must stay within the checked 2 m shoulder-centred
  cube. `joint_path.py` adds deterministic, budgeted detours with <=1-degree
  samples. These are conservative screens, not a continuous dynamic certificate.
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

Earlier native checkpoint: `bimanual_cut_20260911_18` loads the source-derived
knife contacts, verifies opposing left contact, and rejects the blocked right
approach at 3.5 s. Zero edge contacts/cuts; source assets unchanged. The revised
station's parked-right control `bimanual_hold_control_20260911_05` completes
20 s with no guard fault and maximum post-verification slip 0.0691 mm.
This is one attached-plant fixture, not post-cut retention or general reliability.
The control deliberately has false cutting gates and a non-success exit.

New opt-in parameters: `--grasp-depth-m .125` places the shaft farther toward
the original finger tips without moving the base; default .1025 is unchanged.
`--cut-arc-m .02` selects a diagnostic seam within the existing agreed 10..20 mm
interval; default .01 and all dataset labels remain unchanged. The 20 mm option
has geometry/unit checks, not a passing native cutting qualification.
Stroke end is now shaft radius + half leading-strip width + 1 mm, instead of
fixed 12 mm overtravel. Reaching it cannot authorize a cut.

Latest orientation / nearer-ground-truth grasp checkpoint (September 11):
`bimanual_cut_20260911_22` verifies the left grasp but rejects all 180 sampled
right approaches (99 IK converge; 43 interarm and 56 other self/tool failures).
No right motion or cut occurs. `bimanual_hold_control_20260911_06` completes
20 seconds / 4,800 steps, opposing shaft contact continuously after verification,
maximum slip 0.005615 mm and maximum finger penetration 0.032025 mm. One fixture
only; no severed-material retention, deposit or calibrated tissue cutting.

The requested 50 mm grasp selects the actual shaft body centre at 46.676 mm
from the protected attachment. Original full finger bounds leave 20.676 mm
from the nominal 10 mm cut plane (required placement margin 10 mm). The target
identity, attachment/cut/grasp world points and requested/selected arcs are
reported separately. The bimanual jaw controller now stops at the known shaft
radius minus 0.5 mm instead of requesting zero aperture; native opposing
contact, the original force cap and 1 mm penetration guard still decide pass.
This is a privileged geometry-based fixture setting, not a calibrated force
controller or evidence that a VLM can find this point.

Example **unqualified** latest interactive diagnostic (choose a new output):

```powershell
examples\greenhouse_sim\run_bimanual_cut_probe.cmd --output D:/research/tomato-pi-policy/data/sim_physics/my_cut_check --station-offset .04 .2285 --station-yaw 60 --grasp-arc-m .05 --grasp-roll 180 --grasp-depth-m .125 --approach-distance .02 --compliant-fingers --gui --robot-interactive --no-robot-auto-run --no-capture-milestones
```

Inspect **Right knife mount**, then **Grasp plant-side** and **Show / hide
ground-truth points**: yellow protected attachment, white cut, cyan distal grasp.
Markers are optional session-only visual diagnostics without collisions;
positions update from native target frames on render, not physics ticks.
They are NOT training images. Head/wrist camera views remain available.
The Run button preserves the selected diagnostic view. Inspection-first mode
does not start physics automatically; the existing default still auto-runs.
The current bounded cutting planner blocks UI updates while searching and
can still return a no-clear-path result; it does not force a cut.

The remaining task is a compatible, scene-clear grasp/cutter configuration,
then native blade loading, release, withdrawal and retained-material tests.
No successful current-environment grasp-cut-retain sequence is claimed.

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

## Intact-seam blade loading diagnostic (2026-09-11)

Run from `examples/greenhouse_sim` with a **new** output directory:

```powershell
& D:/isaac-sim-6.0.1/python.bat -B -m sim_physics.blade_loading_probe --output ../../data/sim_physics/my_blade_loading
```

This headless 8-second test isolates the original knife against the two source
capsules adjacent to the 10 mm seam. A native force-limited prismatic drive
loads at <=1 mm/s. It has no robot grasp and **never releases the seam**.
`report.json` and `trace.jsonl` distinguish configured drive limits from native
contact measurements and retain source hashes. Full-plant bending/leaf loads,
arm/camera clearance and actual tissue fracture are not modeled by the coupon.

Optional `--contact-stiffness-n-m 250|500|1000|2000` authors an explicit
uncalibrated native contact material on the coupon; default `0` remains rigid.
These are model-discovery experiments, not changes to greenhouse defaults.
The full-contact 0.5 N and 1 mm penetration guards remain enabled. Native contact
normals reject broad-face contact; force, net advance and joint-anchor checks
must coexist. The 0.3 mm loading criterion is not reduced to obtain a pass.

Recorded cases under `data/sim_physics/`:

| `blade_loading_20260911_` run | Contact stiffness | Drive cap | Maximum qualified net advance |
|---|---:|---:|---:|
| `03` | Rigid | 0.35 N | 0.0000167 mm |
| `04` | 1000 N/m | 0.35 N | 0.01336 mm |
| `05` | 250 N/m | 0.35 N | 0.00242 mm |
| `06` | 250 N/m | 0.45 N | 0.03649 mm |

All four complete their bounded diagnostic without a guard fault; **none**
meets the mechanical loading window. These results motivate a separately
validated local indentation/fracture model, not more force or timed release.
They do not establish camera-clear right-arm access or bimanual cutting.

Correction to those first four reports: their `total_contact_magnitude_n`
contains contact-point impulses only; friction was not recorded. Do not treat
that field as a complete force bound. The current diagnostic subscribes to
native full contact reports, retaining friction anchors separately and adding
their magnitudes conservatively for guards. Friction never counts as normal
edge loading. PhysX documents the separate anchor stream in its
[contact reporting guide](https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/AdvancedCollisionDetection.html#contact-friction-information).

Full-report run `07` repeats k1000/cap0.35 at240 Hz: peak conservative load
0.31161 N, same0.01336 mm loaded advance as `04`. `08` at480 Hz and `09` at960 Hz,
with unchanged physical/drive parameters, reach0.04231/0.05127 mm respectively;
neither qualifies. `--physics-hz` is an isolated diagnostic option, not a
greenhouse rate change. Native and pose-derived signed velocities are both
recorded; they are not assumed equivalent under the solver's
[split-impulse handling](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Simulation.html#solver-iterations).

The full-robot planner now screens the entire rigid tool stroke before spending
endpoint IK iterations. This rejects camera/bracket/plant conflicts early;
all downstream arm/transit/native checks remain mandatory. Native full-robot
contact guards also include friction-anchor loads, with no friction used as
cutting evidence. Neither change claims a successful bimanual cut.

## Native collision-representation diagnostic (2026-09-11)

`cooked_geometry_probe.capture_current_stage(NEW_JSON_PATH)` can be awaited
in one already-open **stopped diagnostic** Kit stage containing the original
seed101 plant and full robot. It does not own SimulationApp, play the timeline,
step physics, change a collider or replace any planner screen. The caller
must supply an outer wall-time limit; pending native requests are cancelled
on the probe's own timeout or stage/timeline changes.

The installed PhysX cooking API returns convex vertices, polygon spans and
planes for exactly the right wrist bracket and MainStem26/27 collision prims.
The probe copies callback buffers, validates topology, binds source geometry,
USD transforms/units, physics settings and the native binding binary, and
checks that these remain unchanged. Returned vertices are collider-local;
full transforms (including scale) are applied before world-metre reporting.
No visual mesh or newly authored approximation substitutes for native cooking.

`data/sim_physics/cooked_geometry_20260911_01/{run,convexes}.json` captures
all three meshes successfully: **16 convex parts each**, four service updates,
32.547 s including startup/fixture construction and before shutdown. Original
source hashes/settings remain unchanged; no timeline play or physics steps.
The reproduction wrapper and full log are preserved alongside the run.
The isolated headless stage is not a full-greenhouse cut or a performance FPS
benchmark. Unit regression including this diagnostic: **315 passed**.

Important: the cooking interface supplies a prim's representation; it does
not expose an attached live actor's shape identity. The report deliberately
sets `eligible_to_replace_screen=False`. Native transform/scale/query agreement
and conservative narrow-phase checks are still required before these pieces
can safely replace enclosing-box rejection in the actual planner. Existing
hand clearance, protected-plant checks and runtime contact guards stay active.

### Native geometry agreement and current planning limits

`python -B -m sim_physics.cooked_query_probe --output NEW_DATA_SIM_PHYSICS_DIR`
creates one stopped, diagnostic robot/plant stage, parses native actors, captures
four explicitly selected colliders and compares labelled native ray queries.
It never advances physics or changes the source assets. Use Isaac's Python.

Native `cooked_query_20260911_03` passes 2,328 sampled rays with source state
unchanged; maximum distance error 1.517 micrometres. This is **sampled query
agreement**, not exhaustive shape equivalence or a collision-free robot path.
Two preceding blocked reports are preserved: `_01` used a vertex hull that
differs from native polygon planes; `_02` corrected the planes but bound source
state before native parsing changed session bookkeeping. `_03` binds after
parsing and preserves the native planes with full inverse-transpose transforms.
Do not re-hull captured vertices and assume they enclose the native solid:
some fitted polygon planes extend outside that hull. `convex_clearance.py`
therefore remains advisory, not an actor-clearance authorization.

The bimanual harness has opt-in `--native-static-clearance`. Only conservative
boxes of fitted right camera/knife attachments can refine an existing static
scene-box rejection, using native `overlap_box` on the *whole tool box plus
the unchanged margin*. Dynamic stems/leaves, arm/self checks and all later
IK/transit/native safety checks remain mandatory. Positive collider coverage,
bounded queries, owned scene/physics-change subscriptions and final validation
prevent reuse outside one synchronous planning snapshot. Missing, changed or
late query evidence fails closed. This does not prove complete simulation-shape
query coverage and cannot interrupt a stalled native call.

Pre-hardening native trial `bimanual_cut_20260911_31` re-verifies the known grasp,
clears 605 conservative pair rejections with 867 native queries and still finds
**zero** complete corridors/IK attempts across 756 proposals (560 hand/tool,
196 scene failures). Planning takes 3.178 s, not a camera-FPS measurement.
No right motion or cut occurs; the report is not a success case.

`--cut-proposal-json` replaces the orientation grid with ONE explicit,
source-target-bound world-direction proposal. It cannot supply target position,
mounting, margin, contact permission or skip IK. Projection onto the fresh
measured stem axis must stay below the declared limit (at most one degree).
This enables controlled left-posture tests without accidentally rotating the
right corridor too. It remains privileged diagnostic input, not VLM execution.

### Isolated progressive material-interface prototype

`cohesive.py` implements an explicitly parameterized, uncalibrated bilinear
mixed-mode cohesive law: persistent material-point history, irreversible
damage, elastic unload/reload and area-scaled stored/dissipated energy. It has
one common fracture energy across modes; it does not implement a general
mixed-mode toughness fit. Compression and friction are separate. No timer,
commanded displacement, native release API or production default is present.

`cohesive_native_probe.py` couples four material facets to native implicit D6
springs in an isolated, zero-gravity two-body laboratory coupon. There is no
FixedJoint or parallel weld across the failing interface. Damage uses measured
post-step material-anchor displacement, then updates next-step secant stiffness;
this is a one-step-lagged coupling, not an implicit nonlinear fracture solve.
Drive forces/work are reconstructed, **not native force sensor readbacks**.
The present bridge deliberately stops on compression rather than claiming a
validated unilateral contact law.

Example, from `examples/greenhouse_sim`, with a new output directory:

```powershell
& D:/isaac-sim-6.0.1/python.bat -B -m sim_physics.cohesive_native_probe --native-run --kn-pa-m 1e8 --kt-pa-m 2e8 --strength-pa 1e4 --gc-j-m2 2 --physics-hz 960 --case softening --output ../../data/sim_physics/my_cohesive_coupon
```

These are toy coupon parameters, not measured tomato tissue: reference area
4 mm2, two native density-derived 4 g bodies, carriage stiffness300 N/m,
damping8 N.s/m and cap0.08 N. Normal onset is0.1 mm, final separation0.4 mm,
peak0.04 N and fracture energy8 microjoules. The carriage was chosen before
native tests to avoid a known quasistatic instability of the earlier draft's
100 N/m drive; material coefficients/guards were not changed between cases.

Native `cohesive_subcritical_20260911_01`, `cohesive_disabled_20260911_01` and
`cohesive_softening_20260911_01` each complete9,600 steps without a guard fault.
The first two retain zero applied damage. Softening reaches complete separation
at3.407292 s, dissipates8 microjoules and retains zero connection stiffness on
unloading. Corresponding softening `_02`/`_03` complete at480/1920 Hz. These
are restricted material-bridge tests: **no blade, grasp, plant, camera stream,
cutting channel or training episode**. Passing all runs does not itself prove
timestep convergence; separation timing at480 Hz misses the strict proposed
timing comparison despite completing safely.

Next: a local, continuous deforming contact band on both sides of the seam,
with spatially progressive failure and measured passage of the actual6 mm
plate. Simply replacing two rounded capsules' FixedJoint cannot create that
channel. Bulk/closing-contact tests and source-volume mass checks must precede
blade loading. The production beam, original knife mount, material defaults
and legacy shear qualification thresholds are unchanged by these prototypes.

`cohesive_closing_probe.py` now isolates opening, closing-contact and reopening
with the bounded `cohesive_unilateral.py` soft-limit adapter. The active normal
limit is [-2 mm, 0], with measured motion guarded inside +/-1 mm. At complete
failure its limit schemas/properties are removed; tangential stiffness becomes
zero. This avoids a zero-stiffness limit turning into a hard stop. Remote
support is not a weld across the tested interface. `native_errors.py` passively
stops on delivered native physics errors during setup or solves.

Do not use the first closing-intact run (`_01`): native PhysX rejected its
infinite lower limit. Corrected 960 Hz runs each complete10,560 steps and pass
predeclared per-step momentum, cumulative energy, contact and hold-tail checks:

| Coupon | Evidence directory suffix | Result |
| --- | --- | --- |
| Intact | `cohesive_closing_intact_20260911_02` | Zero damage; settled25/50 micrometre opening; compression then reopening |
| Partially damaged | `cohesive_closing_damaged_20260911_01` | Damage0.799976 retained through closing/reopening |
| Fully separated | `cohesive_closing_failed_20260911_01` | Damage1; failed normal axes remain free; compression collision remains active |

`bounded_protocol_accepted` means only that individual coupon protocol passed.
Joint forces remain endpoint reconstructions, material uncalibrated, coefficient
updates one step lagged. These tests contain no blade, plant grasp or training
episode. The original6 mm-thick blade still needs a physically opened channel.
`material_band.py` authors a continuous-volume discrete rigid-cell/spring
prototype with checked mass/inertia and area-weighted fracture anchors; native
bulk mechanics, spatial refinement and blade passage are not yet qualified.

### Connected-shaft grasp evidence

The bimanual probe uses exact inner-pad and connected detached-shaft collider
identities. Physical support comes from signed native callback impulses; any
tensile contribution subtracts from grip. Each pad must show >=20 mN net
compressive reaction, with the existing opposition, geometry, dwell and slip
checks. Selected tensor rows are matched independently for sensor integrity;
their observed magnitude-only scalars are not signed force evidence.

Native `bimanual_hold_control_20260911_40` completes20 s with verified grasp,
no guard fault and maximum slip0.005253 mm. This is a right-parked control, NOT
cut/retention/deposit success. Its whole-cut state intentionally remains failed
because there is no blade sequence; the hold gates are individually true.
Trials41/42 investigate a different held posture. Trial42 verifies grasp but
rejects the first proposed tool frame for left-palm/right-camera interference.
Safety margins and force limits remain unchanged. Raw fault reproduction uses
`--diagnostic-grasp-contacts --bimanual-hold-control`; it always stops at the
captured fault and cannot enable cutting or signed-mode execution.
