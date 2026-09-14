# Current-package physics qualification (experimental)

This is an opt-in engineering harness on `koh-dev/sim-vlm`, not a replacement
for the static dataset collector, and not yet a validated robot manipulation
environment. Do not collect training demonstrations from this harness.

## September 14 current status - A/B/C NOT frozen

Latest continuation: isolated A/B retain their native passes; full-greenhouse
right-only B passes331/340/349/355. Greenhouse A and responsive C remain OPEN.
Native344 grasps but fails static post-cut retention preflight before knife
motion. Native360's expanded waiting search finds no station; its final native
controls now finish within the reserved query budget. See root `dev.md` for
per-run evidence and failures. None of the following historical successes
establish general greenhouse bimanual reliability or calibrated tissue cutting.

`--station-waiting-search` is an opt-in ZERO-motion option requiring
`--screen-station --cut-station-orbit`. Twelve bounded initial waiting offsets
leave the cut-entry pose/orientation unchanged. Fresh waiting/entry native
checks and left self/interarm path checks remain; no prior clearance or motion
authority is inherited. A fresh execution must revalidate the whole moving
path and measured grasp. The search reserves final actor controls instead of
exhausting the native-query budget; it never increases that budget.
Zero-motion searches also cache exact native query arguments within the one
guarded epoch (4096 entries). Actor positive controls always query afresh.
Native364 explores18 bases with12633 actual queries/10334 cache hits but finds
no station. This is not a dynamic-scene cache or a physics runtime speedup.

Earlier checkpoint: **native310 passes all10 isolated bimanual mechanism gates**, including
full-section traversal, retention, unloaded reverse and a fresh clear withdrawal
endpoint after a2mm upward egress. Native293 remains the isolated direct-cut pass.
This does not establish general reliability, a neutral-stance approach, greenhouse
A/B, calibrated tissue cutting or responsive performance. Native312's bounded
whole-arm approach search still times out without a path. VLM work stays paused.

Reproduce native310 in a NEW folder (headless; add `--capture` for diagnostics):

```bat
D:\isaac-sim-6.0.1\python.bat -B -m sim_physics.ground_truth_trial --output D:\research\tomato-pi-policy\data\sim_physics\my_new_egress_trial --mode bimanual --milestone cut_action --process-zone-trial --through-stroke-trial --material-clearance-trial --postcut-egress-trial --blade-aim-offset-m .0015 --rectilinear-floor-contacts --coupled-fingers-trial --physics-threads 1 --cut-priority-report D:\research\tomato-pi-policy\data\sim_physics\bimanual_downward_20260914_native293\report.json
```

`--postcut-egress-trial` uses fresh post-cut geometry. Initially close, previously
severed faces can only escape along increasing separation bounds, under0.01N
measured full-tool load; all other obstacles and the final1mm margin stay checked.
The old waiting pose is not silently redefined. An explicit screened egress joint
goal is independently checked against the final native pose. No weld/pose setter.

`--joint-transit-fallback` is an experimental bounded APPROACH search, not permission
to deviate from the required downward cutting stroke. `--postrelease-feed-m-s`
defaults to0.0003; the0.001 comparison FAILED a finger-load guard in native311 and
must not be presented as a working faster setting. Regression v6:4578 tests pass.

The current knife is the source's straight lower crossbar, with arc up and a
measured downward stroke. Historical native253-256 below used the superseded
mounting-plate contact recipe; they do **not** qualify this corrected edge.
Current native293 passes isolated right-only cut, section traversal and unloaded
withdrawal. Experimental coupled-jaw native297 retains the branch through the
same stages (max material slip1.286mm), but its final parking-clearance screen
fails. Native301's rendered repeat retains the same failure: tighter
flat-cylinder planning bounds do not clear that final endpoint. See root
`dev.md` for results; do not infer success from an active run or the limited
`cut_action.passed` field alone.

Intact-greenhouse native295/296/298 establishes the left grasp but rejects the
right tool approach before motion; native298 exhausts its60s planning budget.
Lift-before-rotation proposals and same-epoch native empty-box certificates are
under qualification. No surrounding plant visual/contact geometry is removed
by these planner changes. All native contact, force, grasp and slip limits stay.
Real-time full-scene operation, repeats/reset and realistic tissue fracture
are **not** qualified. VLM/data collection remains paused until A/B/C pass.

New diagnostic options (not frozen defaults): `--coupled-fingers-trial` models
compliant equal/opposite left-jaw motion at480Hz; it is not a plant weld and
its transmission stiffness is uncalibrated. `--cut-priority-report REPORT`
changes candidate ordering only, never imports a motion path or its clearance.
`--right-ready-lift-m METRES` proposes an IK-solved higher initial wrist pose;
3mm/10mm native300/299 proposals were rejected against the original plant.
Do not use those lifts as a verified fix. Each run needs a NEW output folder.

## Historical September 13 limited cut-action checkpoint (superseded)

Both requested actions now pass on the isolated original source branch fixture:
left grasp + right cut (native254/255), and right-only cut (native253/256). These are
50-second, 480 Hz native trials with the complete RB-Y1 A v1.2 and original
camera-aligned knife. Source geometry/material priors and force, slip and
active-contact guards are unchanged. This is NOT general greenhouse reliability,
calibrated tissue cutting, a complete robot task, or training-approved experience.

The user explicitly permits released material to land on the torso/base. The
opt-in `--milestone cut_action` records those exact post-release contacts as
`released_debris`, including normal/friction loads, without changing collision
filters or physical response. Attached plants, neighboring plants, both hands
and the active right arm/tool still use the existing guards. Release identity,
exact detached collider inventory and the disabled seam are checked first.

The limited success contract is checked approach + strategy-specific support +
measured blade-contact seam release + at least two seconds of guarded observation.
The actual trials continue to 50 seconds. Full withdrawal/retention/deposit are
reported separately. Native254 retains for 31.5 seconds after cutting with
2.21 mm maximum slip, but FAILS final blade clearance against the held branch.
Native253 is unheld throughout, separates, and passes right withdrawal. Its
maximum recorded passive-body debris load is 10.07 N; landing/damage is NOT
certified. It also emits native material-index warnings after the fall, an
unresolved engine/contact-metadata caveat. Do not suppress these or claim
calibrated post-impact material behavior from this result.

Reproduce the limited actions from `examples/greenhouse_sim`, using Isaac's
Python and a NEW output directory (headless; does not take over the open UI):

```bat
D:\isaac-sim-6.0.1\python.bat -B -m sim_physics.ground_truth_trial --mode bimanual --milestone cut_action --output D:\research\tomato-pi-policy\data\sim_physics\my_new_grasp_cut_trial
```

Use `--mode right_only` and another new output directory for the distinct unheld
test. Optional `--capture` renders at 15 Hz and saves paused native milestone
PNGs with frame receipts. Verified grasp/cut captures follow measured events,
not nominal schedule times; cut-only never produces a verified-grasp image.
These are diagnostic viewport images, NOT synchronized training RGB-D.
Native255 (bimanual,11 PNGs) and256 (right-only,14 PNGs) repeat the non-rendered
cut times and force/slip results with rendering enabled, and exit successfully.
Their `image_evidence` receipts bind each PNG to the observed timestep/state.
Right-only has no verified-grasp images. Close-up views accompany cutting in256;
the wrist/main stem partly occludes the edge, so images alone do not establish
the contact location. The native cut evidence remains authoritative for this
simulated seam-release model. Final physics regression:4,221 tests passed.

### Visible one-shot watch panel

To open this same native trial in a NEW Isaac window, add `--watch`:

```bat
D:\isaac-sim-6.0.1\python.bat -B -m sim_physics.ground_truth_trial --mode bimanual --milestone cut_action --watch --output D:\research\tomato-pi-policy\data\sim_physics\my_new_cut_watch
```

It waits for **Run once: left grasp + right cut** and offers the full-robot,
grasp close-up/plant-side, original knife, head and wrist-camera views. Use
`--mode right_only` in a separate new process to watch unheld cutting. No
physics values or native execution checks change; only 15 Hz rendering and
the visible one-shot observer are enabled. The 50 simulated seconds currently
take several minutes of wall time. Use the panel, not the timeline or object
transform controls: changed idle state/timeline is rejected, not resumed.

The result stays paused for inspection. Run can be requested only once;
reset/replay are deliberately unavailable for this cut topology. Close and
relaunch with a new output folder for another run. `live_status.json` reports
ready/progress/result; `watch_result.json` stores the completed probe result.
Original benchmark `report.json` is finalized on window closure. This watch
panel does not waive final withdrawal failure or certify a full robot task.

Omitting `--milestone cut_action` preserves the older full-sequence assessment
and passive-contact failure policy. Both Python and Kit exit codes now agree
with the explicitly requested milestone; a success label alone is insufficient.
The original full-sequence gates remain visible in `report.json` even when the
limited action passes. See `ground_truth_trial.json` for the exact fixture,
root `dev.md` for results, and `cut_action.py` for the limited contract.

Limits: one selected source petiole (seed101_full/SubStem_41), pre-positioned
task-ready right arm, 20 mm seam / 80 mm grasp arc, no reposition, and no
surroundings in the isolated contact fixture. Automatic strategy switching,
arbitrary target approach, reset/retry qualification, intact greenhouse
requalification and real-time performance remain unfinished. Retention/deposit
and the bimanual final retreat must not be inferred from a limited cut pass.

## September 12: camera-aligned knife and precise closer grasp

The tested inspection launcher is **grasp-only**, not a repaired cutting demo:

```bat
examples\greenhouse_sim\run_camera_aligned_grasp_demo.cmd --output D:\research\tomato-pi-policy\data\sim_physics\camera_grasp_review_new
```

Use a new output directory. It opens the full greenhouse paused at the test
start; press **Run left grasp (hold only)**. The right knife remains parked.
The guide toggle distinguishes the protected junction, nominal cut and grasp.

- `--knife-alignment camera` rotates the entire original knife about wrist Z
  so its arc is on the actual right-camera radial side (-90 degrees relative
  to the previous approved mount). Camera/bracket/flange translations and
  source meshes remain unchanged. **Global blade-down is a wrist-pose
  requirement, not established merely by this mount correction.**
- `--exact-grasp-arc --grasp-arc-m .060` places the grasp at 60 mm from the
  attachment, not the old nearest-segment centre at 71.126 mm. Planning,
  post-fetch slip and guide markers use the same body-local material point.
  The nominal cut stays at 10 mm. Native65 held this closer point for the
  20-second test with verified bilateral contact, no guard fault, max slip
  0.1903 mm and 34 mm initial finger-to-cut-plane clearance. This is not
  post-cut retention or general grasp reliability.
- A more aggressive 46.676 mm centre grasp (native64) reached the unchanged
  0.5 N finger-contact limit at 2.9125 seconds and is **not approved**.
- `--cut-style downward` is a separate **unqualified** planner mode. It
  projects gravity onto the fresh stem-transverse plane (at most 30 degrees
  from down), screens the complete blade stroke, requires 0.80..0.98 arm
  extension, and uses a straight Cartesian approach with no RRT detour.
  Pose/force/collision/retention gates remain. Upward measured blade
  directions cannot authorize release. Legacy world proposals/fixed-shoulder
  constraints cannot silently override this mode. No successful native
  downward sequence has been recorded; the nearer stance in native66 was
  rejected before dynamics for surrounding-plant overlaps. Native67 at the
  original station verified the closer grasp, then rejected all 50 tool
  corridors before right-arm motion (27 hand/camera and 23 blade-plate/main-
  stem conflicts), retaining native-query and conservative collision checks.
- Batched rigid-pose validation preserves all scalar tolerances and validates
  every new frame. Matched profiled native62/63 reduced median control tick
  16.19 -> 13.34 ms with four physics threads; recorded physics/contact values
  were identical, excluding wall-clock validation receipts. This is a modest
  improvement, **not real-time performance** or a GUI FPS measurement. Neither
  visuals, source geometry, contact reporting nor physics rate was reduced.

Evidence lives under `data/sim_physics/bimanual_latency_20260912_62`, `_63`,
`bimanual_downward_20260912_64`, `bimanual_exact_grasp_20260912_65`, and
`bimanual_downward_20260912_66` / `_67`. The CPU/USD regression passed 3,082
tests (`data/sim_physics/regression_20260912_downward_v2.log`); this is not
native cutting qualification. See `dev.md` for limitations.

### Later September 12 investigations (not working cutting presets)

The exact60 mm **hold-only** control remains the demonstrated baseline. Native83
repeats its20-second hold with0.1903 mm maximum slip. Nearer40..54 mm feedback
closure/damping trials still fail contact qualification. `--force-closure` and
`--anchored-pad-damping` are explicit failed/experimental comparisons, default
off; neither is a reliable-grasp or calibrated-material claim.

Downward planning now uses a gravity-bounded transverse direction fan and an
original-vertex-enclosing slanted plate bound. Optional initial right pose,
left IK seed and six torso angles support coordinated **prephysics** proposals;
exact URDF limits, full-scene spawn, sampled paths and native guards still apply.
The tested torso proposals85..87 fail camera/plant corridor checks; they are not
launch presets. No new downward cutting success is established.

With native clearance explicitly requested, the left approach can now refine
static coarse-bound rejections using the same live PhysX query contract as the
right planner. Full hand bounds and1 mm scene margin remain; dynamic leaves,
unknown geometry and genuine native hits are never removed. Query/final-epoch/
cleanup failures prevent acceptance. This does not replace opposing-contact
grasp evidence. See the native88 result in `dev.md`.

Native88 now verifies a closer54 mm underhand grasp with28 mm finger-to-cut
clearance, but rejects all350 downward tool corridors before right motion.
Native89 holds that grasp for20 s with maximum slip0.000680 mm, then fails its
final integrity check on a host MemoryError. Native92 repeats the hold after
streaming hashing, with the same maximum slip and all source hashes verified.
Its completed/bounded/grasp gates pass; full bimanual status intentionally fails
because no right cut occurred. Native91's shorter8 mm knife standoff still
rejects all350 corridors. This is not cut/withdraw/retain qualification and
does not replace the baseline launcher.

Full-package robot launches now record a read-only Windows memory preflight
before importing SimulationApp. Less than16 GiB commit headroom or4 GiB available
physical RAM produces `blocked_host_memory` and no simulation launch. This is
a conservative engineering reserve, not a measured runtime-capacity guarantee
or an Isaac minimum requirement. It never changes apps, drivers or pagefiles.
Other OSes are explicitly marked not checked. To inspect without launching:

```bat
cd examples\greenhouse_sim
python -m sim_physics.host_memory
```

Authored joint-topology caching avoids repeated unchanged relationship reads,
not fresh native-state checks. A smaller guarded collision workspace is an
opt-in timing experiment, **not an established speedup**:83/84 preserve all4800
physical trajectory rows and full visible surroundings, but84 ran slower in
this pair. The default2 m half-window and full-detail visuals are unchanged.

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

### Current cut-path and material limitations

Native44 verifies the farther selected-shaft grasp but rejects camera/main-stem
overlap. Native45 finds one endpoint before its query budget expires. Native46
isolates it: the whole tool corridor passes, then upper-arm/leaf clearance stops
the sampled stroke before execution. A timeout fallback is not a native hit.

The planner now tries the complete stroke/transit immediately after a feasible
endpoint. Invalid native queries stop with an explicit unavailable-query error.
`--right-ik-fixed-joint INDEX DEGREES` optionally selects exact-URDF redundancy
for the entire stroke; it does not move the parked robot, authorize a pose or
relax any check. Joint1=+0.5 is NOT a working preset: offline wrist-limit failure.

`seam_interface.py` is a PRE-START, inactive-only alternative to the many-cell
band:24 quadrature connections between existing bodies, no source/body edits,
explicit material, checked rounded geometry and unloaded initial states. It
does not remove existing welds, activate physics or certify a blade channel.
`seam_loads.py` reports explicit gravity-only load and a circular-section first
tensile-damage estimate; it cannot measure actual seam/gripper reaction.

Do not integrate the small coupon's10 kPa strength into this plant. Its estimated
gravity bending demand is about66 times the circular3 mm section's first-damage
moment under that toy strength. The32-cell native rest diagnostic also remains
unqualified on force balance despite tiny displacements. Neither prototype is
the production cutting backend. Detailed failed/passing evidence and limitations
are recorded in `dev.md`; no full cut/retain/deposit pass is established yet.

### Explicit brittle-seam engineering mode (2026-09-11)

`--cut-model signed_edge_load_brittle_seam_v1` is an opt-in strength-only
approximation for the rigid admissible seam. It requires actual leading-edge
normal resistance >=0.2 N for25 ms, verified left shaft contact, correct
direction/location, and unchanged force/slip/penetration guards. Original
collider order determines impulse sign; tensile contributions subtract.
Non-cancelling normal PLUS friction loads still enforce the0.5 N tool cap.
Travel is recorded, not claimed as tissue work or a prerequisite in this mode.
Legacy `force_qualified_pre_authored_seam_release` remains the default with
its0.3 mm post-qualified-contact travel criterion (now also signed-load gated).
Neither model is calibrated tissue fracture; the cohesive prototypes remain
inactive. No mesh, pose, velocity, mass or collision-filter change creates a cut.

Native49 establishes the first current-greenhouse signed blade-contact release
with left grasp: at11.8667 s,0.21477..0.30992 N signed resistance for29.17 ms,
upper load0.35757 N; designated seam at10 mm. Its full61-sample cut path and
405-check transit search pass, planning11.081 s with an explicit60 s budget.
The trial then FAILS at12.9083 s because slip reaches3.01165 mm. Bilateral contact
persists, but that is not reliable retention; incidental post-release leaf/arm
contacts are also recorded. No complete cut/retain/deposit success is claimed.
The left goal no longer pulls on a timer after release. Time-based withdrawal
has been demoted to schedule diagnostics; verified completion requires current
native endpoint and clearance evidence. `withdrawal_evidence.py` supplies a
pure same-sample helper. `withdrawal_native_check.py` now integrates final
post-fetch endpoint and current bounds checks; this is not swept-path evidence.

Native51 repeats49 with opt-in `--diagnostic-grasp-dynamics`. Native52 tests an
explicit `--finger-actuator-limit-n .8` engineering prior (legacy default .5),
with an independent >=.5 N ALL-contact rejection at each left finger in both
sparse modes. It cuts and retains longer, but fails at14.3625 s on.5935 N finger2
load. Do not call this a successful sequence or calibrated hardware setting.

The diagnostics expose excessive post-release internal rotations in the
contact-uncoupled implicit spring predictor; its no-contact qualification does
not establish correct held-branch mechanics. The nominal energy proxy reaches
~2.50 J and a spherical rotation-vector wraps near pi. Contact-coupled spring
qualification is required before reporting reliable cutting/retention. Detailed
evidence and the distinct command-timestamp correction are logged in `dev.md`.

### Fixed-root diagnostic and retention prerequisite (2026-09-13)

The complete isolated `--fixed-root-cut-trial` now supports an evidence-gated,
no-step fixed-to-free transition. It preserves native plant/robot state and
commands and refreshes plant tensor metadata without a global reset. Original
material/geometry and all contact, direction, force, slip and collision gates
remain. Repeated blade-qualified releases are recorded; **retention still
fails**, so this is not a reliable grasp/cut/deposit demo or training source.

Add `--require-retention-screen` to that existing complete diagnostic command
to require a current native contact-patch gravity-capacity check BEFORE knife
planning. It waits one measured tick after establishing the grasp reference,
rejects stale/unbound data, and uses only compressive detachable-shaft contact.
The recorded current grasp fails this prerequisite in native209, with no knife
motion or seam release. A pass would establish only approximate static gravity
capacity, not cutter-load tolerance, actuator feasibility or dynamic retention.
This is source/native-state privileged diagnostics, not VLM execution from
hidden cut coordinates. No default or GUI launch preset is changed.

`--native-drives-after-cut` and `--rigid-pad-control` are isolated comparisons,
not recommended fixes. The former restores original K/C and removes explicit
spring effort after checked release. The latter changes the pad contact law,
not geometry/friction/caps, and cannot qualify compliant-pad physics. Both
comparisons still fail retention. See `dev.md` for individual failures and
regression results; never report software test passes as physical success.

`--pregrasp-half-aperture-m .008` is an opt-in commanded initial jaw opening
for the complete isolated fixed-root feedback trial with retention preflight.
The opening must leave at least 2 mm beyond the selected shaft radius on each
side. Initial joint state, approach/closure screening, feedback scheduling and
antiwindup use the same opening. Source geometry, physical joint limits,
actuator/contact budgets, collision margins and cut gates are unchanged; the
default remains 25 mm per side. Native213 passes zero-step startup but rejects
the approach against an original target leaf. This is not a successful grasp
or evidence that narrower pre-shaping alone solves access or retention.

Grasp approach/closure now also screens every robot collider against the full
current target, separately from the arm's bounded static-context cache. This
catches torso/parked-arm contact with distal leaves without pretending that a
target-only subset covers the whole greenhouse. Native startup and execution
guards remain mandatory. The downward planner also uses source-enclosing,
edge-aligned bounds for each original knife arc partition; no native mesh,
visual, mounting or collision margin is changed. Native233/234 establish an
attached grasp and static retention prerequisite, but their tool corridors
still fail. Neither change establishes reliable cutting or post-cut retention.

Additional **default-OFF isolated diagnostics**:

- `--physical-grasp-span`: connected exact shaft identities under the current
  native finger footprint, instead of a fixed +/-one-segment list. Broken
  connections, support, leaves and other branches remain excluded. Full raw
  normal-row geometry/sign checks and selected-tensor reconciliation remain.
- `--settle-retention-preload`: require 0.2 s of measured preload dwell within
  the original13.5 s acquisition budget before static capacity assessment.
  It checks the existing symmetric controller's mean support target/deadband,
  both compressive fingers, low speed, current telemetry and bounded slip.
  Knife timing is rebuilt only after readiness; no elapsed-time stroke jump.
- `--effort-bounded-grasp-target`: distinguish nominal PD reference bias from
  actual measured penetration. The original0.30 N/200 N/m bounds the reference
  bias to1.5 mm; actual native penetration remains limited to1 mm. Original
  0.24 N desired support,0.30 N PD cap,0.8 N total actuator and0.5 N per-finger
  all-contact budgets remain. Requires explicit effort/antiwindup, physical
  span, preload settling and retention preflight; not a production preset.
- `--preload-force-servo`: requires the complete effort-bounded trial. Uses
  a +/-0.01 N control deadband INSIDE the unchanged +/-0.03 N readiness band,
  with a nominal 0.5 s outer force loop and the same 0.5 mm/s closing limit.
  Original force, actual penetration, freshness and safety-backoff checks
  remain. This is an uncalibrated controller setting, not tissue physics.
  Native226 passes measured preload dwell and the static retention prerequisite
  on the same source/pose that timed out in225; fitted tool/camera interference
  then rejects all cut corridors. No retained cut or production qualification.

Native216 exposed an identity mismatch and stopped without a grasp.217
established bilateral grasp after the span correction.220 completed measured
preload dwell but failed static retention capacity (~2.077 budget utilization),
so the knife stayed parked. Neither software test counts nor stable ATTACHED
grasp establish a retained cut, withdrawal, deposit or calibrated tissue model.

`--staged-downward-transit` is another default-OFF isolated retention diagnostic.
It screens bounded wrist staging routes before expensive full-arm IK, without
giving transit the cutting stroke's seam-contact allowance. The actual joint
transit and rebuilt cut stroke still require full collision checks. Native240
rejects the blocked approach in6.604 s versus native237's60-second planning
timeout; this is not a successful arbitrary-start approach or a frame-rate claim.

Native242/243 use a separately checked task-ready initial right pose: both
establish a native grasp and trigger blade-load-qualified joint release at
18.720833 s.242 then exceeds1 mm finger penetration;243 restores original native
spring drives but exceeds3 mm slip.244 adds velocity iterations and still loses
retention. No complete retain/withdraw/deposit sequence is qualified. The fixed
root instrumented cut profile now also records passive full-plant contact rows,
including plant/support pairs omitted by robot-only summaries. Raw recording
does not establish callback completeness, contact work or tissue calibration.

### Current guarded crossbar trials (2026-09-13)

These are experimental diagnostics, **not a frozen reliable cutting system**.
Use the actual sharpened source crossbar, never the legacy component called
`Blade` (that was the mounting plate). Native274..276 passed limited isolated
mechanisms before the full-knife load-accounting correction; those passes do
not qualify the current controller. Latest results and limitations are in
the corresponding dated section of repository-root `dev.md`.

From `examples/greenhouse_sim`, using Isaac's Python and a NEW output directory:

```powershell
$env:PYTHONPATH=(Get-Location).Path
$env:OPENBLAS_NUM_THREADS='1'
& D:\isaac-sim-6.0.1\python.bat -B -m sim_physics.ground_truth_trial --output D:\research\tomato-pi-policy\data\sim_physics\NEW_TRIAL --mode bimanual --milestone cut_action --process-zone-trial --stream-trajectory
```

- `--mode right_only`: left remains open/unloaded; no grasp/retention claim.
- `--through-stroke-trial`: 85 s bounded trial, requires measured actual full
  forward stroke before reversal; a seam release alone is not a pass. No kerf,
  collision removal, increased load limit or calibrated tissue model is added.
- `--greenhouse-trial`: retains the intact source plant, provided greenhouse,
  three-gutter preview planting and both detailed plants. Native startup and
  path checks must pass anew. The current saved pose is blocked by a neighboring
  leaf in native280, so this is NOT yet an executable greenhouse demonstration.
  Nearby context plants are static colliders; only the selected petiole flexes.
- `--screen-ready-pose`: zero-motion same-wrist elbow proposal search.
- `--screen-approach-start`: zero-motion higher/lateral waiting-pose search.
  The two searches are mutually exclusive; both preserve the stopped scene and
  stop before physics. Native281/282 found no clear candidate in their bounded
  families. An eventual proposal still needs a new full native sequence test.
- `--capture`: paused milestone PNGs, not synchronized RGB-D training frames.
- `--watch`: separate one-shot visible panel, no automatic run or reset; cannot
  combine with a zero-motion search. A blocked startup never becomes a GUI run.

`--stream-trajectory` preserves full per-step JSON records in
`bimanual_trajectory.jsonl.gz`, including failure-tick evidence, and keeps small
result summaries plus the latest full row in memory. Read it with `gzip.open`
and `json.loads` one line at a time. Default historical output remains a JSON
array. Reports include archive counts, size and serialization timing; storage
optimization does not skip native checks or certify real-time performance.

The corrected knife load signal includes normal and friction loads from ALL
knife contacts, even when their face/direction cannot authorize cutting. This
prevents rejected contact from looking like free space to the feed controller.
The original 0.5 N load and 1 mm available-contact penetration guards remain.
Native278's full stroke fails on seam-face contact; corrected-accounting
right-only native279 stops short of release. Physically consistent cut-face
deformation/separation, new A/B passes and full-greenhouse performance remain
open. Do not turn these diagnostic outputs into training-approved episodes.

### Current measured section/withdrawal trials (2026-09-14)

Native293 passes the current **isolated right-only** sequence. Bimanual289/291
lose retention after release, so the milestone is NOT complete. Full source
greenhouse startup has a clear proposal (294), not yet a complete motion pass.
See `dev.md` for exact failures/results and remaining fidelity limitations.

```powershell
& D:\isaac-sim-6.0.1\python.bat -B -m sim_physics.ground_truth_trial --output D:\research\tomato-pi-policy\data\sim_physics\NEW_TRIAL --mode right_only --milestone cut_action --process-zone-trial --through-stroke-trial --material-clearance-trial --blade-aim-offset-m .0015 --rectilinear-floor-contacts --physics-threads 1
```

- `--material-clearance-trial`: fresh measured sharp-edge clearance of the
  entire shaft/blade-plane section, then measured unloaded withdrawal. Only
  after guarded release may exact cut-face side contacts support bounded
  sliding; wrong-face contact can never authorize a cut.
- `--rectilinear-floor-contacts`: exact solid decomposition of the original
  source floor, including raised strips, with unchanged rendered mesh.
- `--grasp-arc-m .09`: explicit bimanual placement proposal; not an approved
  preset (native290 rejects this location because of an attached leaf).
- `--screen-station`: zero-step base/two-arm startup proposals. Mutually
  exclusive with other searches; always revalidate in a separate fresh trial.
- `--station-proposal-report PATH`: explicitly load an initial-pose proposal
  for `--greenhouse-trial`. Requires matching task and final-controlled report;
  full native startup/approach/grasp/cut checks run anew, no path replay.

The cut remains an uncalibrated force/direction-qualified seam-release model,
not tissue fracture/kerf or a measured clean-cut quality model. Native293
headless rate is0.26x real time; visual full-scene responsiveness remains open.

### Isolated bimanual repeat (2026-09-14, native310/313/318)

All ten isolated sequence gates pass in these three runs, including retained
material, full shaft-section traversal, unloaded reversal and freshly screened
post-cut egress.313 includes23 paused inspection PNGs. This is still one source
petiole, not broad reliability, intact-greenhouse qualification or tissue
fracture calibration. Native318 runs at about0.20x real time. A/B/C is NOT done.

```powershell
& D:\isaac-sim-6.0.1\python.bat -B -m sim_physics.ground_truth_trial --output D:\research\tomato-pi-policy\data\sim_physics\NEW_TRIAL --mode bimanual --milestone cut_action --process-zone-trial --through-stroke-trial --material-clearance-trial --postcut-egress-trial --blade-aim-offset-m .0015 --rectilinear-floor-contacts --coupled-fingers-trial --physics-threads 1 --cut-priority-report D:\research\tomato-pi-policy\data\sim_physics\bimanual_downward_20260914_native293\report.json
```

The prior report changes candidate ORDER ONLY, never imports clearance or a
joint path. Add --capture for paused diagnostic PNGs. Keep128/0 iterations and
the default0.3mm/s loaded post-release feed:64 iterations lost grasp in317 and
1mm/s exceeded a finger-contact limit in311. Neither is a qualified speedup.
`--screen-station --cut-station-orbit` is a zero-motion initial-station proposal
search with an optional cut-priority report; it cannot certify or execute a path.
See `dev.md` for rejected greenhouse approaches and current memory constraints.

Experimental `--park-left-ready` is RIGHT-ONLY: it holds the source SDK left
ready configuration instead of requiring target-grasp IK for an unused hand.
All native park/contact/self/scene checks remain. Its full-greenhouse zero-step
searches322/323 found no clear station; no execution qualification is claimed.
Do not use it for bimanual grasp or infer retention from a direct cut.

The cut-station orbit now also accepts the current anatomy's downward frame
without a prior report. `--source-station-trial seed19_full/SubStem_41` selects
that existing source and discards the historical seed101 initial-pose recipe.
With `--greenhouse-trial --target-row-slot 0`, swap the detailed target into
the original end-row slot (all144 plants/asset identities/spacing retained).
Native327 found a clear waiting/entry proposal there; fresh328 executed the
approach but rejected wrong-face blade contact and did NOT cut. No middle-row
or bimanual greenhouse success is implied. `--budgeted-joint-gravity` is an
explicit effort-bounded controller experiment, currently under native test.
It must not be treated as a qualified default until its results are recorded.

Native330 confirms that source-budgeted gravity removes the observed steady
wrist bias and enables a greenhouse right-only release, but follow-through
stalls on the attached stump. `--blade-aim-offset-m .0023` is the subsequent
geometry-checked experiment: source-body halfspace clearance and full section
inside the ORIGINAL3mm window, not increased cutting tolerance. It requires
both `--material-clearance-trial` and `--through-stroke-trial`. No completion
or tissue-cutting claim follows from the release alone.

### Intact greenhouse direct-cut reference (native331)

All nine full diagnostic gates PASS for this single end-row target: cut,
full section traversal, unloaded reversal, fresh egress and85s bounded run.
No grasp, safe deposit, broad reliability or tissue-calibration claim.
20 paused inspection PNGs are in the native331 directory. Measured0.1575x
real time is still too slow; bimanual greenhouse validation is still open.

```powershell
& D:\isaac-sim-6.0.1\python.bat -B -m sim_physics.ground_truth_trial --output D:\research\tomato-pi-policy\data\sim_physics\NEW_GREENHOUSE_TRIAL --mode right_only --milestone cut_action --process-zone-trial --through-stroke-trial --material-clearance-trial --postcut-egress-trial --blade-aim-offset-m .0023 --rectilinear-floor-contacts --physics-threads 1 --greenhouse-trial --park-left-ready --source-station-trial seed19_full/SubStem_41 --target-row-slot 0 --station-proposal-report D:\research\tomato-pi-policy\data\sim_physics\bimanual_downward_20260914_native327\report.json --budgeted-joint-gravity --capture
```

The old327 file supplies initial poses ONLY. All native startup, source-section
and full moving-path checks run again; no inherited cut authority. Bimanual
`--approach-vector X Y Z` proposes a different grasp approach side and needs
its OWN new station/path/grasp qualification.

### Watched direct-cut repeat and bimanual search (native340, 2026-09-14)

Native340 passes the same nine full greenhouse direct-cut gates in a visible
window, with the same18.718750s cut/3439 contact rows as331. Its durable result
is `data/sim_physics/bimanual_downward_20260914_native340/watch_result.json`.
This is live execution evidence, not a recorded video, calibrated tissue model
or bimanual pass. It remains slow:0.116834x physics real-time factor. UI updates
are wall-scheduled separately; do not confuse render rate with physics speed.

For the preceding native331 command, replace `--capture` with
`--watch --watch-auto-run` to start one visible run automatically. Without
`--watch-auto-run`, choose a camera then press Run once. Stop aborts the trial;
reset/replay is deliberately not enabled. Use a NEW output directory each time.
Run full greenhouse trials serially; never bypass the Windows memory reserve.

Planning refreshes the UI without advancing physical steps. Automatic timeline
advance is temporarily disabled and restored around the native query; the
existing epoch still rejects any scene, physics, play-state or time change.
Native339's first implementation correctly stopped when rendering advanced
timeline time;340 qualifies the corrected clock ownership for this case.

Bimanual diagnostic proposals may explicitly select `--torso-degrees`,
`--grasp-roll`, `--grasp-pitch`, `--station-pose`, `--left-ik-seed-degrees` and
`--right-ready-degrees`. These are INITIAL proposals, not native pose writes
during execution. Source joint limits, complete native startup/path screens
and all contact/retention/cut checks remain. Report-based and explicit initial
poses cannot be mixed. `--station-reference-report` is an alternative only for
zero-motion same-anatomy station searches; it never grants motion authority.

Current bimanual investigation tests a neutral torso: the old isolated recipe
leans sideways and places the two shoulders about182mm apart vertically.
Do not treat isolated successes or a kinematic candidate as full-greenhouse A.
VLM/data collection remains paused until A, B and acceptable C are established.

Native344 establishes an actual upright-torso bilateral grasp in the intact
source19 greenhouse, but STOPS before knife planning: the current patch's
static retention utilization is2.1423 against unchanged0.5N per-finger contact
bounds. Maximum slip17.4 micrometres alone does not establish post-cut retention.
It is a failed bimanual qualification, not a working A demo. Native345..347
are further rejected zero-motion station proposals, not executed sequences.

Contact bookkeeping now reuses only unchanged bucket sums and bounded plain
path syntax. All changed sums, original contact rows and finite-value checks
remain immediate.126 focused and4750 full tests pass; measured1.207x speedup
is for a mixed-contact helper benchmark only. Whole-simulator C is still open.

Native349 is another complete intact-greenhouse RIGHT-ONLY pass after that
optimization: all9 gates, cut18.71875s,3439 edge contacts, reverse45.177083s.
The40800 selected physical-state/guard records (including full robot and plant
joint arrays) match331 exactly. Runtime0.155116x is still too slow; no net
whole-simulator speed improvement is inferred from the helper benchmark.

A zero-motion station-reference search now preserves the successful seed's
blade-frame tilt. A winning proposal carries that family into fresh execution
as candidate ordering, never inherited path/contact authority. Native350 uses
the corrected15-degree family but still fails neighboring-foliage clearance.
Native351/352 fail self-clearance before motion. Full-greenhouse A remains open.

`--background-evidence-compression` is an explicit OFF-by-default comparison
for full-rate streamed diagnostics. Finite JSON and all native checks remain
on the control thread; only immutable bytes go to a bounded gzip worker.
FIFO/backpressure and joined close preserve every sample; errors cannot count
as a complete archive. Native355 passes all9 direct-cut gates and retains all
40800 physical records, but is NOT faster overall (RTF0.152631 vs3490.155116).
Keep the default synchronous path until a matched net improvement is measured.
Do not sum overlapping worker elapsed time with main-thread elapsed time.

Native354 fails approach IK;356's subsequent self-screened proposal intersects
the gutter with torso1. Neither is a working greenhouse bimanual demonstration.

Native367 repeats the complete right-only greenhouse sequence after lossless
JSON validation optimization: all9 gates and40800 full-rate records. Tick wall
504.274s versus349's547.978s (about8% less), RTF0.168559. All selected physical
records including plant dynamics match; one final epoch metadata timestamp
differs. This remains below real-time responsiveness: C is NOT finished.

Zero-motion station search now covers different orbit sides earlier without
removing candidates. A fresh negative native query can prune right-arm variants
only if exact collider ownership/URDF ancestry proves the blocked body cannot
move with any right joint. Final actor/epoch controls remain mandatory. Native
373/374 explore61/86 candidates but find no clear station; neither proves global
infeasibility. Native344/362's bilateral contact still does not pass post-cut
retention. Full-greenhouse A remains OPEN; no milestone freeze or VLM restart.

`--physics-dispatcher physx` (or `carb`) is an explicit process-local CPU
scheduling comparison. Omit it to preserve the current preference. It is set
before owned scene parsing, verified after reset and restored on exit. Solver
iterations, timesteps, contacts, forces and visuals remain unchanged. A native
matched sequence and physical-outcome comparison are required before adoption;
the setting's readback alone is not speed or native scheduler evidence.

Native378, with `--physics-dispatcher physx --physics-threads 4`, completes all9
greenhouse direct-cut gates. All40800 trajectory records and report measurements
exactly match367. It takes484.902s for85 simulated seconds (RTF0.175293), about
3.84% less wall time than367. This is still far below real time; default settings
are unchanged. Earlier one-worker375 stalled during reset, so do not infer that
all dispatcher/worker combinations work. Explicit selection now also occurs
before Kit boot; exclusive startup checkpoints do not certify task success.

`--station-left-seed-search` is an optional ZERO-MOTION bimanual search extension
with `--screen-station --cut-station-orbit`. It tries at most6 initial IK guesses
per station for the same left wrist frame, with all native endpoint/path checks
and the same45s/global query budget. Never used for right-only parked arms.
Native379 finds different elbows but no collision-free station for its source71
case. A remains OPEN; this is not an executed grasp, cut or retention result.

The expanded search now reuses only identical stock right-chain IK computations,
with fixed parsed-model/torso/solver binding and fresh epoch checks. Native384
checks71 candidates versus379's19 in45s, with619 exact IK hits; all native scene
and robot final controls pass. It still finds no motion proposal. Collision and
contact results are not inherited from this kinematic cache.

`--target-planting-side -1` selects the original negative-X gutter slot; `1`
remains the default. Requires `--source-station-trial` and `--greenhouse-trial`.
Together with `--target-row-slot 0|12|23`, this swaps the detailed target with
one existing backdrop, retaining that exact asset at the old target position.
No density, source geometry, spacing or gutter-height reduction. The detailed
neighbor stays in its original slot. Native385 verifies all144 positions and
asset counts match the original-side trial; its proposed grasp still collides
with a target leaf. Side selection is not proof of robot access or safe cutting.

`--screen-grasp-approach` is an explicit zero-motion bimanual alternative to
the other startup searches. It keeps the current anatomical grasp material
point, base and right arm fixed while proposing bounded left approach/pad
orientations. It checks full finger-to-cut-plane clearance, native startup,
and47 dense self/inter-arm path knots, within45s and the final actor-control
reserve. At most8 `proposed_grasps` are saved; they are NOT scene-path, contact,
retention or cut certificates. Final-control failure revokes all proposals.
Reconstruct an explicit fresh launch with a chosen proposal's approach vector,
pitch/roll, station and joint seeds; all actual moving-scene checks still run.
Do not replay `left_path_degrees` as a certified trajectory. Native394 found
only an already-tested finger-swapped source19 grasp, and395 found no source71
proposal. Full-greenhouse grasp-and-cut remains unqualified.

`--solver-convergence-trial 96` is a diagnostic-only intermediate numerical
comparison, alongside64. It is NOT a qualified faster default. Both robot and
plant receive96/0 instead of128/0, while480Hz, all original contact/visual
geometry and every force/slip/retention/cut gate stay unchanged. A full native
matched outcome and accuracy/performance comparison is required before any
adoption. Merely passing CLI/readback tests establishes no fidelity equivalence.

Native404 completes all9 direct-cut gates at96/0, but takes498.998s for85s
of simulation versus378's484.902s at128/0. No speed benefit was measured;
default128/0 remains unchanged. This does not qualify bimanual96/0 behavior.

The explicit `--screen-approach-start` search now checks a5mm conservative
rest-target margin for the whole waiting right arm/tool, without seam contact
exceptions. It returns up to8 `proposed_waiting_poses`, retaining the first
proposal's legacy fields. If coarse poses yield fewer options, bounded10/20mm
translations around native-clear coarse poses receive the same checks. The
30s deadline, original source geometry, all native actors and final validation
remain mandatory. These are INITIAL-pose proposals, never live commands.
All proposed fields are revoked on failed final controls. The margin is not
a deformation prediction: native402 records approximately40mm distal plant
movement during its initial0.9s. Moving-scene reobservation remains essential.
Native407 finds4 zero-motion alternatives; none is a physical grasp/cut pass.

`--screen-settled-waiting` runs the original guarded initialization to0.9s,
then stops before grasp/cut. It proposes at most8 new INITIAL right waiting
poses from the actual post-fetch plant geometry and bounded static scene.
Every guarded initialization sample is retained (432 at480Hz); conservative
leaf hulls and shaft boxes enclose their sampled settling motion, as well as
the original rest shape. No native shape is edited. Missing/skipped/stale
samples reject the search. The30s search makes no physics step or robot command.
These are geometric proposals, NOT native startup, continuous-time clearance,
equilibrium, left scene-path or cut certificates. Relaunch and revalidate all
native checks. Native412 finds7 proposals;413 physically verifies a left grasp
after selecting a lower waiting pose, but fails static retention before cutting.

`--native-retention-trial` is a separate opt-in, noninteractive bimanual
through-stroke experiment. Default static-retention refusal is unchanged.
The experiment can proceed to independent cut planning after a VALID native
patch's solved over-budget static balance; that failure remains in the report.
It tests whether a flexible branch can remain held while changing shape,
instead of assuming the entire original static orientation must be sustained.
Missing, stale, unbalanced or malformed evidence still rejects. All original
bilateral/dwell,3mm slip,0.5N per-finger all-contact, blade force/direction,
scene, full stroke and measured withdrawal guards remain. Native retention
qualification still requires actual retained bilateral contact and bounded
slip after cutting. Neither this option nor a static solver result proves a
successful cut or retention. NOT a production-qualified faster demo default.

The staged planner stops remaining nominal wrist templates only after the
shared endpoint IK has actually failed. Previously each template could repeat
rigid-sweep screening despite reusing that same failed solve. Ordinary full
arm/path/stroke failures still try every fallback, and blocked rigid strokes
still reject before IK. No native query/force/scene check is waived for a
candidate that can proceed. Unit checks prove reduced redundant calls, not
faster physical stepping; native planning latency remains to be measured.

With explicit `--joint-transit-fallback`, the already-required shared endpoint
solve now precedes all nominal approach previews (but follows the rigid cut
stroke screen). Native415's first pruning version still timed out after18
candidates in68.128s. Native416 completes all50 candidates in14.852s:25 rigid
tool conflicts and25 failed endpoint IK solves, no accepted path or cut.
This removes redundant planning work without changing physical stepping or
claiming that the current bimanual pose is feasible.

The rigid wrist corridor also includes a proven coaxial right-arm5 capsule.
The Model A1.2 chain has one final revolute wrist joint followed by a fixed
tool transform; a capsule on that rotation axis is independent of elbow/roll
at a fixed wrist pose. Its cached actual geometry and parsed chain must prove
this relation. Unknown/noncoaxial shapes remain for full-arm screening. A tiny
contained subset handles numerical roundoff; no native collider is modified.
This rejects impossible forearm-versus-left-camera corridors before repeated
IK/transit work, retaining all full-path/native checks. Native417 had spent
60.223s at one placement repeatedly failing2.701mm versus the3mm margin.
Native418 is the pending physical comparison; this is not an FPS or cut claim.

Native418 subsequently checks7 placements in14.970s and performs the downward
source-crossbar release at24.141667s with left contact retained at that instant.
It stops0.104s later on a0.567389N finger load (>0.5N), before completing traversal
and withdrawal. This is a partial physical result, NOT reliable full greenhouse
grasp-and-cut qualification. Full regression v39:4972 pass.

The explicit symmetric preload-force servo now uses measured jaw position
and velocity when backing off high (>0.4N) all-contact loads. Its temporary
closing-effort reference ceiling is0.12N; original0.3N PD,0.5N native contact,
aperture, bilateral and3mm slip guards remain. No physical state/contact is
edited. Native419 avoids418's overload stop but exceeds3mm slip after release,
so a load-capable grasp is still needed. Full regression v40:4989 pass. Neither
the new controller nor these partial cuts qualify full bimanual completion.

The explicit public `--grasp-arc-m` proposal range is60..180mm from attachment
(default80mm unchanged). Exact source-segment membership, detachable-side and
full finger/cut clearance, IK, native scene/contact checks and static retention
are still required by the normal bimanual recipe. Farther holding can reduce
gravity torque but may worsen foliage access: native422 hits a fruit/truss at
startup and423 rejects a settled leaf obstruction. Native424 finds a different
above-shaft INITIAL-pose proposal;425 is pending physical validation. These
are privileged engineering tests, never perception-derived execution labels.
