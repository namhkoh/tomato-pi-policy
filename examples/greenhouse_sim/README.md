# Greenhouse Deleafing Benchmark (Isaac Sim)

A tomato **deleafing** benchmark — removing orphan/lower leaves from high-wire
vines — for evaluating vision-language-action policies, built on Isaac Sim 5.1.
The intended path is: collect demonstrations in sim, finetune π0.5, then deploy
the same policy in sim and on a Rainbow Robotics RB-Y1.

## Why the deleafing task

Deleafing is ~19% of greenhouse tomato labour and still largely unautomated: the
canonical robot attempt (van Henten et al. 2006) ran ~70 s per leaf, about 35×
slower than a human. Agronomically, *where* the cut lands matters as much as
whether it happens — pruning flush to the stem produces wounds that are near
absolutely resistant to *Botrytis cinerea*, while leftover petiole stubs are
highly susceptible. That makes residual stub length a physically meaningful,
objectively measurable headline metric, which is the benchmark's main
differentiator from existing plant-manipulation suites.

## Environment

This example runs inside **Isaac Sim's bundled Python**, not the repository's uv
environment — the same pattern as `examples/libero` (Python 3.8) and
`examples/aloha_sim` (Python 3.10), which own interpreters incompatible with the
main one. Isaac deps must never be added to the root `pyproject.toml`.

Everything here depends only on what a bare Isaac Sim install already ships
(numpy, scipy, pytest, and USD), so no `pip install` is required to build assets.

### Requirements

- **Isaac Sim 5.1** or newer (developed against 5.1.0-rc.19, Kit 107.3.3).
- An RTX GPU. Rendering the scene at 15M triangles wants ~8 GB of VRAM;
  ~6 GB of system RAM is resident with one planted bed.
- A clone of this repository. No `pip install` step — the pipeline uses only
  what Isaac Sim already bundles (numpy, scipy, pytest, USD).

### Pointing at your Isaac Sim install

Every script finds Isaac Sim automatically when run through Isaac's own
interpreter. Set `ISAAC_SIM_PATH` only if you are running them another way, or
if auto-detection fails:

```bash
export ISAAC_SIM_PATH=/home/you/isaacsim          # Linux / macOS
```
```powershell
$env:ISAAC_SIM_PATH = "D:\isaac-sim"              # Windows PowerShell
```

Auto-detection tries, in order: `ISAAC_SIM_PATH`; the install containing the
running interpreter (so Isaac's own python always works); then `D:\isaac-sim`,
`~/isaacsim`, and `/isaac-sim`.

### Which interpreter each script needs

Substitute your own install path throughout — the examples below use
`$ISAACSIM` / `%ISAACSIM%` to stand for it:

| Platform | Isaac Sim's interpreter |
|---|---|
| Linux / macOS | `$ISAACSIM/python.sh` |
| Windows | `%ISAACSIM%\python.bat` |
| pip install (`pip install isaacsim`) | the `python` of that virtualenv |

Only the scripts that **start a simulator** — `launch_greenhouse.py`,
`interactive_greenhouse.py`, `build_robot.py`, `inspect_robot.py`, and
`cut_demo.py` — actually require it.

The **asset scripts** (`convert_vines_to_usd.py`, `build_scene.py`) need USD but
no simulator, and `usd_env.py` loads USD out of an Isaac Sim install directly.
So they also run under any Python 3.11 that has numpy and scipy, as long as
`ISAAC_SIM_PATH` is set:

```bash
ISAAC_SIM_PATH=/path/to/isaacsim python examples/greenhouse_sim/convert_vines_to_usd.py
```

That matters for CI and for rebuilding assets on a machine with no GPU.

Run everything **from the repository root**, since the default asset paths are
relative to it.

## Building and launching the scene

```bash
# Linux / macOS. On Windows use %ISAACSIM%\python.bat instead of $ISAACSIM/python.sh.

# 1. Segment the vine GLBs and write per-organ USD (~2 s per vine, ~360 MB total).
$ISAACSIM/python.sh examples/greenhouse_sim/convert_vines_to_usd.py

# 2. Compose vines into the greenhouse.
$ISAACSIM/python.sh examples/greenhouse_sim/build_scene.py --beds 1 --plants-per-bed 8

# 3. Open it in the Isaac Sim viewport.
$ISAACSIM/python.sh examples/greenhouse_sim/launch_greenhouse.py
```

Steps 1 and 2 are one-time; only step 3 is needed on later runs. The viewport
stays open until you close the window. Loading takes about a minute, because the
stage is ~15M triangles — the window appears well before the plants do.

Source assets stay under `greenhouse/`; everything generated lands in the
gitignored `data/greenhouse_sim/`. Step 1 re-checks each vine against its
metadata sidecar and exits non-zero if any invariant breaks, so a regenerated
asset set cannot quietly degrade the cut sites.

The benchmark scene directly sublayers the supplied `greenhouse/gh_tomato_test.usd`
with the same Z-up, metre-scale stage metadata. It does not rescale or recreate
the greenhouse. In that original asset, the target gutter spans Z=0.671–0.888 m
over a cultivation floor at Z=-0.305 m, placing its top 1.193 m above the floor.

The default `source` placement mode leaves that gutter and all bed transforms
unchanged. Each physics-ready `tomato_NNN.usd` inherits the exact root
translation and two-sided yaw of its matching `tomato_stem_NNN` prim in the
source stage. Use `--placement-mode procedural` only when a generated
bed-relative layout is intentionally required.

To render a frame without a display (this is how the scene is checked in CI or
over SSH):

```bash
$ISAACSIM/python.sh examples/greenhouse_sim/launch_greenhouse.py \
    --headless --screenshot data/greenhouse_sim/scene.png
```

Useful flags: `--beds` / `--plants-per-bed` / `--spacing` / `--seed` on
`build_scene.py`; `--eye` and `--target` to move the inspection camera on
`launch_greenhouse.py`.

## Interactive greenhouse physics

Launch the composed greenhouse with the fitted RB-Y1 and the first static vine
replaced, at the same bed transform, by the stable articulated vine. The
replacement and robot reference are authored only in the USD session layer, so
the built scene and source assets are not modified:

```bash
$ISAACSIM/python.sh examples/greenhouse_sim/interactive_greenhouse.py
```

The other seven vines remain static by default to keep viewport performance
practical. Use `--physics-vines N` to upgrade more placements, or `--no-robot`
to return to the accepted vine-only environment.

While the simulation is running:

- Hold **Shift** and left-drag any visible stem, petiole, or leaf blade.
- Press `[` / `]` to select the previous / next deleafing petiole.
- Press `G` to close the retained left tongs and `O` to open/release them. A
  valid grasp requires loaded contact from both fingers on the selected petiole
  or one of that selected branch's physical leaf proxies.
- If positive physical load reaches both fingers on a different visible branch
  while the gripper is closing, that opposed pinch automatically selects its
  owning `Vine/SubStem`. A one-finger brush never retargets. Press `T` or click
  **Target pinched leaf** only as an explicit fallback.
- Press `C` only for an explicit **DEBUG FORCE CUT**. It releases the selected
  joint but is never benchmark-valid; a physical cut triggers automatically
  from real straight-edge contact, force, direction, travel, and work.
- Press `V` to select the next dynamic vine when more than one is enabled.
- Press `1` for the inspection view, `2` for the head D405, `3` for the left
  wrist D405, or `4` for the right wrist D405. The same choices are buttons in
  the **Vine Interaction** window.

Mouse pulling raycasts the original rendered GLB mesh and maps the hit to its
supporting rigid body, so Shift-drag works across broad leaf blades and thin
stems. In `interaction` collision mode each foliage organ also owns a thin,
oriented physical contact box on that same body. Those boxes are filtered from
the plant's own rigid bodies to preserve the stable authored rest pose, but they
collide normally with the robot, fingers, and knife. A selected branch's leaf
boxes are valid left-finger grasp contacts. Neighbouring foliage contact remains
physical and benchmark-unsafe, but the default `--teleop-contact-policy monitor`
logs it without pausing mailbox updates. Use `--teleop-contact-policy rollback`
only when automatic return to the last contact-free pose is desired. The mouse
pull is a bounded spring; `--drag-stiffness`, `--drag-damping`, and
`--drag-max-force` tune it.
The live report at `data/greenhouse_sim/interactive_greenhouse.json` records
`visual_mouse_grabs`, releases, peak force, and cuts.

A 1.0 m/s deterministic airflow field is enabled by default so the foliage has
subtle stationary sway. It applies `0.5*rho*Cd*A*v^2` at each compound leaf's
measured foliage centroid; this is a physical external load, not render
animation. Use `--airflow-speed 0` for still air, or tune `--airflow-speed`,
`--airflow-frequency`, and `--airflow-direction` for the episode.

Run the automated renderer-picked GUI pull acceptance:

```bash
$ISAACSIM/python.sh examples/greenhouse_sim/interactive_greenhouse.py \
    --visual-pull-probe \
    --report data/greenhouse_sim/interactive_greenhouse_visual_pull_probe.json
```

Run the combined 10-second airflow, bounded pull/recovery, and flush-cut
acceptance without a display:

```bash
$ISAACSIM/python.sh examples/greenhouse_sim/interactive_greenhouse.py \
    --headless --settle-steps 240 --airflow-probe-steps 2400 \
    --pull-probe SubStem_00 --cut SubStem_00 \
    --report data/greenhouse_sim/interactive_greenhouse_airflow_pull_cut_10s.json
```

Real Shift-drag, stationary airflow motion, and cutting were manually accepted
in the viewport on 2026-08-06.

## RB-Y1, cameras, and deleafing knife

Build the exact RB-Y1 Model A v1.0 asset from the bundled official SDK URDF:

```bash
$ISAACSIM/python.sh examples/greenhouse_sim/build_robot.py
```

The builder keeps the mobile base dynamic, disables all importer-created
collision instance scopes, restores the 17 active custom URDF capsules exactly
once, and adds conservative base and left-gripper
contact proxies, and fits the supplied hardware under `greenhouse/robot_assets/`:

- one Intel RealSense D405 and bracket on the head;
- one D405 and bent bracket on the exact 18 mm-spaced M3 screw pair of each
  wrist force-sensor plate, using the RB-Y1 v1.1 CAD datum and the robot chain's
  native left/right mirroring;
- the supplied deleafing knife directly on the right end-effector flange.

The original right gripper body and both tongs are fully removed from rendering
and collision. Their invisible URDF links/joints remain only to preserve the
exact v1.0 articulation and controller indexing; the right tool is knife-only.

For the knife, only the outer 2 mm strip along the long local `-X` side of the
flat plate carries the cutting semantic; cutting travel is local `-X` and the
straight edge axis is local `+Y`. The full plate remains the physical collider
but its broad face cannot trigger a cut. The U-shaped arc remains non-cutting
support geometry. Its nearest bound is more than 10 mm inside the semantic
strip, so the arc cannot block centreline edge contact as it did with the former
distal local `-Y` definition. The knife remains rolled with the arc upward.

The nominal base pose is `(6.99114, 3.93000, -0.3050817)` m at -90 degrees
yaw, on the opposite side of the target gutter. By default,
`--robot-position-mode target-conditioned` resolves the settled physical
target before authoring the robot, preserves a 20 mm reach reserve, requires a
distal grasp segment, and tests deterministic 0/30/60/90 mm aisle advances with
exact RB-Y1 IK and wrist-D405 clearance. `SubStem_00` keeps the nominal pose;
`SubStem_01` advances 30 mm. Use `--robot-position-mode fixed` to preserve
`--robot-position X Y Z` exactly for calibration or regression work.

Each dynamic vine has a finite hidden catch tray so severed foliage does not
fall forever. This is a synthetic episode fixture rather than greenhouse
structure, so its collision pair with the `/World/RBY1/base` articulation is
filtered. Without that filter, a low arm pose can intersect the tray and receive
an immediate depenetration impulse even though no visible object was touched.

Inspect each fitted assembly without greenhouse occlusion:

```bash
$ISAACSIM/python.sh examples/greenhouse_sim/inspect_robot.py --view head_camera
$ISAACSIM/python.sh examples/greenhouse_sim/inspect_robot.py --view left_wrist
$ISAACSIM/python.sh examples/greenhouse_sim/inspect_robot.py --view right_tool
```

Run the integrated non-contact stability check and capture a fitted D405:

```bash
$ISAACSIM/python.sh examples/greenhouse_sim/interactive_greenhouse.py \
    --headless --settle-steps 480 --contact-diagnostics \
    --capture-camera right_wrist \
    --screenshot data/greenhouse_sim/right_wrist_d405_acceptance.png \
    --report data/greenhouse_sim/robot_wrist_screw_mount_acceptance.json
```

The wrist datum is important. A read-only audit of the supplied v1.1 FreeCAD
assembly and bent-bracket STEP places the two bolt origins at
`(+/-9, -42, 38.5)` mm, 3 mm outside the `y=-39` mm force-sensor mounting face.
The same centres in the normalized bracket mesh are
`(+/-9, 56.919538, 30.119184)` mm. The rebuilt asset therefore places each
right bracket root at `(0, -0.098919538, 0.008380816)` m with identity local
rotation and the left root at `(0, +0.098919538, 0.008380816)` m with a local
180-degree Z rotation. The explicit left mirror puts it on the operator-confirmed
outer screw face in the generated URDF/USD chain. Sensor-only rolls of 180
degrees right and 0 degrees left keep the policy observations upright without
changing either physical CAD fit.

A 480-step fixed-station soak with 127 plant contact shapes kept all 34 robot
bodies finite, recorded no robot-vine contact, and reported zero runaway vine
organs.

A complete robot-driven bi-manual acceptance was verified for the earlier
`Vine_0000/SubStem_00` aisle layout: qualifying required at least 66.3 N,
transverse forward motion, sustained work through the measured diameter, a
clear protected-contact ledger, and a prior left two-finger grasp. That run
reported one intended cut, zero unsafe contacts, and the complete `grasped ->
orphan_retained -> transported -> released -> deposited` sequence.

The current default has intentionally moved to the exact Side_1 source plant
`Vine_0002/SubStem_00`, with RB-Y1 in the wider negative-Y inter-gutter aisle.
Its startup/contact/stability gate is accepted, but its full bimanual route is
not yet accepted: the strengthened live arm-vine gate safely rejects the current
left approach before the forearm can enter the lower canopy. Use the normal
interactive launch to inspect and pull/cut manually; treat **Run Full IK** as a
re-planning diagnostic until the new-side episode passes the strict report.
See `dev.md` for the current evidence and open blocker.

When a bimanual probe is run visibly, the GUI remains open after completion or
safe rejection. The visible scheduler advances exactly one 240 Hz physics sample per
control sample and performs a render-only refresh every fourth sample. It does
not use `SimulationContext.step(render=True)`, which would execute four physics
substeps at the configured 60 Hz rendering rate before the force, grasp, and cut
monitors run. Target-conditioned runs reuse the already accepted distal grasp
selection before beginning motion, while still resolving live IK at every
approach waypoint. After a visible probe finishes, the GUI remains open on the
final state until it is closed manually.

Post-cut retract is target-conditioned as well. The planner evaluates both
knife-wing choices and multiple right-tool routes against live vine capsules.
It sweeps the full flat blade and non-cutting U-support through exact RB-Y1
kinematics, chooses the safer lateral direction family, then maximizes
separation before aisle stow. The strict PhysX contact ledger remains the
acceptance authority; geometric planning does not waive a physical contact.

For live video, launch without `--headless` and use the interaction-window
camera buttons or keys `1`-`4`; the selected D405 becomes the active viewport.
For an image without the GUI, use `--capture-camera head`, `left_wrist`, or
`right_wrist` together with `--screenshot` as shown above.


### World Bank live teleop demo

The `koh-dev/worldbank` branch provides a presentation-focused station with the
original greenhouse, fitted RB-Y1 Model A, one articulated nearby tomato vine,
live whole-body state mirroring, camera views, pulling/grasping, and direct
flat-blade traversal cutting. It does not run the unfinished autonomous full-IK
sequence.

Start the read-only physical-robot mirror in the first terminal:

```bat
examples\greenhouse_sim\run_worldbank_robot_mirror.cmd
```

Then start the visible simulator in a second terminal:

```bat
examples\greenhouse_sim\run_worldbank_demo.cmd
```

Pass `record` to the simulator launcher to save synchronized head and wrist
camera frames plus JSONL state/action/task/safety samples:

```bat
examples\greenhouse_sim\run_worldbank_demo.cmd record
```

The station uses the validated `Vine_0002/SubStem_00` fixed-base pose. Camera
buttons or keys `1`-`4` switch inspection/head/wrist views, and the UI's
10 mm forward/back controls provide bounded final positioning. For the visual
cut demo, move the exposed flat blade through a severable leaf or petiole proxy
at at least 0.01 m/s for two consecutive physics steps. The curved support and
main stem remain protected. This convenience traversal is deliberately logged
as non-benchmark; it is intended for the live presentation, not RL scoring.

The robot bridge is read-only and never commands physical hardware. It mirrors
both arms, torso, head, and?when
`http://192.168.50.243:8765/status` is available?the calibrated left gripper.
If that endpoint is unavailable, the simulator safely holds the left gripper
open while all other measured joints continue to mirror.
### Deterministic targets, repeatability, and simulator teleoperation

Select an exact physical petiole or use seeded selection:

```bash
$ISAACSIM/python.sh examples/greenhouse_sim/interactive_greenhouse.py \
    --headless --bimanual-probe full \
    --target-vine Vine_0000 --target-organ SubStem_00 --episode-seed 0 \
    --report data/greenhouse_sim/target_00.json
```

Run strict isolated-process trials with `run_bimanual_repeatability.py`. The
summary does not count partial approaches or process return codes as success;
it requires one intended physical cut, zero unsafe contacts, clear blade safety,
and a deposited orphan.

For simulator-only demonstration collection, start the GUI first:

```bash
$ISAACSIM/python.sh examples/greenhouse_sim/interactive_greenhouse.py \
    --teleop-command-file data/greenhouse_sim/teleop_command.json \
    --teleop-record-dir data/greenhouse_sim/demonstrations
```

Then publish the lab leader arms from the Python environment that has
`rby1_sdk` installed:

```bash
python examples/greenhouse_sim/rby1_leader_to_sim.py \
    --command-file data/greenhouse_sim/teleop_command.json --record
```

Hold each leader tool button to enable only that simulated arm; releasing it
holds the measured pose. The left trigger closes the simulated left gripper.
The bridge never connects to or commands the physical RB-Y1. Commands remain
watchdog-, joint-limit-, speed-, and deadman-gated. PhysX still resolves every
contact and records it in the safety ledger; default `monitor` mode does not
pause teleop after contact, while opt-in `rollback` mode returns toward the last
contact-free pose. Each recording episode contains JSONL state/action/task/safety
samples plus the selected head
and wrist D405 RGB frames. Use `--dry-run` to publish one disabled command for a
hardware-free integration check.

#### Read-only physical RB-Y1 state mirroring

This connected-hardware adapter is owned by `koh-dev/rby1`; that branch includes
the latest `koh-dev/deleaf` simulator and keeps hardware I/O out of the benchmark
foundation branch.

To mirror the connected robot rather than the leader devices, start the read-only
publisher first from the Python environment containing `rby1_sdk`:

```bash
python examples/greenhouse_sim/rby1_robot_state_to_sim.py \
    --address 192.168.12.1:50051 \
    --command-file data/greenhouse_sim/teleop_command.json
```

It reads the 24-position Model A state vector and publishes the six torso, seven
left-arm, seven right-arm, and two head joints. The simulated left gripper polls
`http://192.168.50.243:8765/status` at no more than 10 Hz and normalizes motor
ID 1 with that homing session's `gripper_min_q`/`gripper_max_q`. On the left
motor the numeric minimum is physically open and the numeric maximum is closed,
so openness is `(max_q - position) / (max_q - min_q)`. The physical right
gripper is intentionally ignored because the simulated right wrist carries the
knife. Neither state source contains a physical robot or gripper command API. Add
`--record` to the read-only publisher only after manual grasp/cut validation to
set the simulator mailbox recording flag; recording still writes only simulator
state, actions, task/safety labels, and D405 frames.

For the current `Vine_0002/SubStem_00` teleop station, the validated
collision-clear startup command is:

```bash
$ISAACSIM/python.sh examples/greenhouse_sim/interactive_greenhouse.py \
    --teleop-command-file data/greenhouse_sim/teleop_command.json \
    --target-vine Vine_0002 --target-organ SubStem_00 \
    --robot-position-mode fixed \
    --robot-position 10.639221515539253 4.25 -0.15254085567917297
```

The 4.25 m aisle coordinate includes 50 mm of measured-pose leaf clearance;
launching the same live pose at 4.30 m correctly latched on a 2.0 mm
neighbouring-leaf overlap before accepting motion. The interaction window now
provides **Robot forward +10 mm** and **Robot back -10 mm** fixed-base
preposition controls. Session travel is capped at +100/-50 mm and clipped again
to the measured gutter/chassis-safe greenhouse aisle interval. This deliberately
lets the operator advance beyond the former +30 mm cap and enter foliage contact;
the live contact monitor remains active. The action is refused while a branch
is grasped. It atomically updates the PhysX articulation root and world
fixed-joint anchor, zeros root velocity, and suppresses interaction-cut evidence
for four steps so repositioning cannot be mistaken for a knife stroke.

Teleop initialization consumes a fresh mailbox sample before PhysX starts, so
the simulator begins at the measured upper-body pose rather than the scripted
knife pre-contact pose. If no fresh sample exists, it uses the symmetric SDK
ready pose. The benchmark base remains fixed at the selected UI pose: mirroring
mobile-base odometry is deliberately excluded until a separate greenhouse
collision envelope is implemented. Malformed or stale mailbox input still
captures and holds one safe pose; the
hold target never follows a gravity-driven falling state. Contact alone does
not hold under the default `monitor` policy.

Plant capture is accepted only below 20% gripper openness and only when the
exact closest point on a petiole capsule or foliage proxy lies inside the
visible closed-jaw channel for three consecutive steps. The generated finger
contact boxes match the supplied finger CAD; they do not extend invisibly into
the channel. PhysX opposed-finger events remain preferred, with the reported
closed-jaw geometry path covering thin-shape callback misses.

For benchmark cutting, a real flat-edge PhysX impulse remains preferred. If the
thin rigid petiole misses that callback, only the exact active target already
retained by the left grasp can produce a reported compliant reaction when the
finite blade edge enters its radius. Direction, speed, transverse alignment,
66.3 N force, full-diameter work/crossing, and protected-contact gates remain
mandatory.

Live teleop also enables a separate interaction cut path so the supplied flat
blade behaves like a usable knife against the plant's broad contact proxies.
A non-zero PhysX contact from `BladeCollision` against a severable foliage,
petiole-grasp, or petiole-cut proxy must persist for two consecutive physics
steps while the commanded cutting edge moves at least 0.01 m/s. It then releases
that proxy's associated pre-authored `SubStem_XX` junction. The curved support
and protected main stem remain non-cutting. Because Isaac does not split the
render mesh at an arbitrary contact point, a leaf-blade hit releases the whole
associated branch at its authored junction. These events are recorded in
`blade_traversal_cuts` with `benchmark_valid=false`; the strict IK/benchmark
sequence temporarily disables this convenience path.

## Online RL environment

Teleoperation is not part of the RL control path. Start the simulator as a
loopback-only synchronous environment in one terminal:

```powershell
D:\isaac-sim\python.bat examples\greenhouse_sim\interactive_greenhouse.py `
  --headless --rl-server `
  --target-vine Vine_0002 --target-organ SubStem_02 `
  --robot-position-mode fixed `
  --robot-position 10.639221515539253 4.38 -0.15254085567917297
```

The server advances no unrequested policy steps. Each action advances exactly
12 physics samples at the default 20 Hz policy / 240 Hz physics rates. The
15-value normalized action is:

1. left-arm joint velocities, seven values;
2. right-arm joint velocities, seven values; and
3. left-gripper aperture velocity, where `-1` closes and `+1` opens.

The stable 56-value state observation contains normalized arm positions and
velocities, gripper openness, left-jaw-to-grasp and blade-to-cut vectors,
target/tool axes, strict task-phase one-hot state, grasp/cut/transport progress,
and protected-contact state. Rewards use distance-potential differences plus
bonuses only for strict `grasped`, `orphan_retained`, `transported`, `released`,
and `deposited` transitions. An unsafe contact, wrong sequence, or task failure
terminates the episode. The manual two-frame blade-traversal convenience path
is disabled, so it cannot become an RL shortcut.

Reset restores every robot and per-organ vine articulation root, joint state,
velocity, severance joint, fixed grasp, task state, cut work, and contact
Policy velocity is integrated from a persistent drive target, so zero action
holds position rather than following gravity-driven measured-state drift. Arm
and gripper commands are acceleration-limited, and reward penalizes consecutive
action reversals. `--rl-max-arm-acceleration` exposes the arm limit. An optional
seven-value `--rl-initial-left-arm` provides an explicit fixed curriculum start;
it does not bypass contact, grasp, cut, task-order, or safety gates.

ledger. The seed applies bounded +/-1 degree arm-start variation and a seeded
phase of the accepted foliage airflow. The target is fixed for one server
process; change the launch target or run multiple workers for a target
curriculum.

A trainer can use the small client directly or construct the optional
Gymnasium wrapper from `greenhouse_sim.rl_client.gymnasium_env`. A reference
single-environment PPO implementation is included:

```powershell
D:\isaac-sim\python.bat examples\greenhouse_sim\train_online_rl.py `
  --total-steps 1000000 `
  --checkpoint data\greenhouse_sim\rl\ppo_deleaf.pt
```

The trainer runs outside Kit, keeping policy dependencies and GPU allocations
separate from the simulator. Closing it asks the simulator server to shut down
cleanly. This is a low-dimensional online-RL baseline; synchronized D405 image
observations remain on the policy/VLA track and are not claimed by this state
environment. The included PPO smoke verifies rollout, reset, update, and
checkpoint plumbing; it is not a converged deleafing policy.

### Parallel PPO and grasp-first curriculum

For grasp curriculum collection, launch the server from the validated 100 mm
`SubStem_02` start and terminate the episode at the strict grasp event:

```powershell
D:\isaac-sim\python.bat examples\greenhouse_sim\interactive_greenhouse.py `
  --headless --rl-server --rl-terminal-phase grasped `
  --target-vine Vine_0002 --target-organ SubStem_02 `
  --robot-position-mode fixed `
  --robot-position 10.639221515539253 4.38 -0.15254085567917297 `
  --rl-initial-left-arm -29.348612 -0.999427 -38.306154 `
    -93.646180 126.029107 -56.984894 46.707186 `
  --rl-reset-joint-noise 0.25 --rl-max-arm-acceleration 60 `
  --rl-max-episode-steps 96
```

Collect accepted physical demonstrations through that same server API:

```powershell
D:\isaac-sim\python.bat examples\greenhouse_sim\collect_online_rl_grasp_demo.py `
  --episodes 8 `
  --output data\greenhouse_sim\rl\grasp_expert.npz
```

The collector velocity-controls the validated collision-clear IK waypoints and
writes an episode only after the physical task reaches `grasped` without an
unsafe contact. It does not set joint state or create a synthetic grasp.
During `seek_grasp`, the environment freezes the right arm and prevents gripper
closure outside 50 mm. PPO applies the identical mask to log probabilities and
entropy. `--rl-terminal-phase grasped` reports `objective_reached`; full-task
`success` remains reserved for safe floor deposit.

For shared-policy process-parallel collection, launch one headless server per
port, then pass those ports to the parallel trainer:

```powershell
D:\isaac-sim\python.bat examples\greenhouse_sim\train_online_rl_parallel.py `
  --ports 8766 8767 8768 8769 `
  --total-steps 8192 --rollout-steps 128 --epochs 4 `
  --learning-rate 1e-4 --target-kl 0.02 `
  --entropy-coefficient 0.001 `
  --demonstrations data\greenhouse_sim\rl\grasp_expert.npz `
  --bc-epochs 300 --bc-log-std -2.3 `
  --checkpoint data\greenhouse_sim\rl\grasp_ppo.pt `
  --report data\greenhouse_sim\rl\grasp_ppo.json
```

Each port owns a complete independent Isaac physics process. Network actions
are issued concurrently, observations are batched through one actor-critic,
and GAE is computed independently along each worker trajectory. This is not an
Isaac Lab in-stage vector environment, so GPU memory scales approximately with
the worker count. Use `nvidia-smi` to choose a safe count.

The accepted grasp run behavior-cloned 618 physical transitions, then collected
8,192 PPO actions. Its final deterministic policy reached `grasped` in 8/8
unseen-seed trials with zero unsafe contacts and a 40.25-action mean. This is a
validated grasp curriculum, not a complete deleafing policy: right-arm cut,
transport, drop, image observations, and multi-target generalization remain
gated stages.

Rendered RL ticks advance physics with `step(render=False)` and call
`context.render()` separately. Do not replace this with `step(render=True)`:
at the authored 60 Hz render / 240 Hz physics rates it advances four physics
samples and changes the learned trajectory.


## Cutting demo

Rigs one vine with compliant physics, settles it, cuts the lowest petiole, and
measures whether the organ came away without disturbing the rest of the plant:

```bash
$ISAACSIM/python.sh examples/greenhouse_sim/cut_demo.py --gui
```

Results are written to `data/greenhouse_sim/cut_report.json`.

## Troubleshooting

**Nothing is printed to the terminal.** Kit buffers and swallows Python stdout,
and `--/app/fastShutdown=True` masks non-zero exit codes, so a failed script can
look like a clean run. Every simulator script therefore writes a machine-readable
report file; read that rather than the console. This bites hard when debugging —
an apparent hang or crash is often just a result that was never written.

**`ModuleNotFoundError: No module named 'pxr'`.** You are running the asset
scripts under a plain Python rather than Isaac's. Either use Isaac's interpreter
or set `ISAAC_SIM_PATH`; `usd_env.py` will then locate the bundled USD build
without booting Kit.

**`Failed to read texture file /home/.../Desktop/...`.** The greenhouse asset's
DomeLight points at an absolute path from the machine it was authored on. The
scene layer clears it automatically, so this should only appear if you open
`greenhouse/gh_tomato_test.usd` directly rather than the built scene.

**The scene is slow or the GPU runs out of memory.** Reduce
`--plants-per-bed`; each vine is ~750k triangles.

**`KeyError: <class 'NoneType'>` from `manipulator_selector.py`, followed by an
ill-formed empty `SdfPath`.** This is an Isaac Sim 5.1 transform-gizmo selector
bug: mixed USD/Fabric selection can forward a handled selection as `None` to
the next manipulator. It is not a PhysX or vine failure. The live greenhouse
clears stale selection and unsubscribes that native selector because its own
rendered-mesh raycast owns Shift-drag. Camera navigation and custom vine pulling
remain enabled.

## How a vine becomes addressable

The vine GLBs fuse every organ into a single mesh split only by material, with
no node hierarchy — nothing in the file says which triangles form one leaf or
where its petiole meets the stem. `organs.py` reconstructs that:

1. Organs are connected components of each material batch.
2. Stem organs are rooted into a tree by shortest path over surface adjacency,
   weighted so hop count dominates and the surface gap only breaks ties.
   Otherwise a drooping petiole brushing a lower one gets picked as its parent.
3. Foliage and fruit attach to their nearest stem organ rather than joining that
   search, since leaflets of neighbouring leaves touch constantly.
4. The main stem's children are named from the generator's attach points by
   optimal one-to-one assignment, so one bad pairing cannot cascade.

`vine_usd.py` then writes each organ as an Xform whose **origin is its junction
with its parent** — simultaneously the cut site and the joint anchor — carrying
its geometry as a child mesh and its child organs as child Xforms. Severing a
petiole is therefore an operation on one subtree:

```
/World/Vines/Vine_0000/MainStem/SubStem_06/...
```

On all 20 vines this reproduces the metadata organ counts exactly, keeps
sub-stem labels monotonic in height (which the bottom-up deleafing rule
depends on), and never falls back to grafting.

## Layout

| Path | Purpose |
|---|---|
| `greenhouse_sim/glb.py` | Minimal glTF/GLB reader (geometry, materials, embedded textures) |
| `greenhouse_sim/organs.py` | Organ segmentation and plant topology reconstruction |
| `greenhouse_sim/vine_usd.py` | Per-organ USD authoring |
| `greenhouse_sim/vine_physics.py` | Stable articulated links, contact zones, trellis clips, and floor |
| `greenhouse_sim/vine_visuals.py` | Original GLB visuals attached to physics links |
| `greenhouse_sim/vine_interaction.py` | Visible-mesh pulling and foliage-area airflow forces |
| `greenhouse_sim/cutting.py` | Directional force/work cut gate, joint release, tear monitoring, and cut grading |
| `greenhouse_sim/deleaf_task.py` | Required bi-manual grasp/cut/transport/deposit state machine |
| `greenhouse_sim/episode.py` | Deterministic exact/seeded physical target selection |
| `greenhouse_sim/repeatability.py` | Strict repeated-episode acceptance aggregation |
| `greenhouse_sim/teleop.py` | Simulator-only mailbox safety gate and demonstration recorder |
| `greenhouse_sim/rl_env.py` | Framework-neutral action, observation, reward, and termination contract |
| `greenhouse_sim/isaac_rl.py` | Live Isaac physics adapter, full episode reset, and loopback server |
| `greenhouse_sim/rl_client.py` | JSON-lines client and optional Gymnasium wrapper |
| `greenhouse_sim/greenhouse_scene.py` | Vine placement over the greenhouse stage |
| `greenhouse_sim/robot_hardware.py` | D405/bracket mounts and flat-blade cut semantics |
| `greenhouse_sim/robot_kinematics.py` | Exact RB-Y1 v1.0 FK/IK, Jacobian, and effort-capacity checks |
| `greenhouse_sim/robot_scene.py` | RBY1 placement, ready pose, and fit validation |
| `greenhouse_sim/usd_env.py` | Makes USD importable without booting Kit |
| `extract_robot_hardware.py` | Reproducible supplied-CAD extraction and manifest |
| `build_robot.py` | RBY1 v1.0 import, collision restoration, and hardware build |
| `inspect_robot.py` | Isolated fitted-hardware close-up renderer |
| `convert_vines_to_usd.py` | Asset build + invariant checks |
| `build_scene.py` | Scene composition |
| `launch_greenhouse.py` | Static viewer / headless capture |
| `interactive_vine.py` | Isolated physics-vine inspection |
| `interactive_greenhouse.py` | Physics greenhouse, mouse pulling, cutting UI, and acceptance probes |
| `run_bimanual_repeatability.py` | One-Isaac-process-per-target/seed repeatability runner |
| `rby1_leader_to_sim.py` | One-way lab leader-arm input publisher; never commands the real RB-Y1 |
| `train_online_rl.py` | Reference PPO trainer for the synchronous Isaac environment |

Run the tests with Isaac's interpreter (they are hermetic and use synthetic
assets, so they need no GLB files):

```bash
$ISAACSIM/python.sh -m pytest examples/greenhouse_sim/greenhouse_sim
```

## Status

Built and verified: asset pipeline, scene composition, stable
articulated vine physics, task-directed robot contacts, visible-mesh pulling,
foliage-area airflow, greenhouse integration, pull/recovery, and flush cutting.
Manual Shift-drag, airflow-motion, and debug-cut acceptance is complete. Exact RB-Y1
v1.0 import, ready-pose stability, three D405 fits/optical frames, and right
knife-only end-effector semantics are also verified. Physical leading-edge cut
qualification, protected-contact accounting, hardware-effort-limited dual-arm
motion, and complete grasp/contact/cut/transport/floor-deposit acceptance on
`Vine_0000/SubStem_00` and the formerly blocked `SubStem_01` are verified in
| `train_online_rl_parallel.py` | Shared-policy PPO over independent live Isaac server processes |
Isaac Sim with zero unsafe contacts. A fresh `Vine_0002/SubStem_02` run also
verified a 24 N opposed grasp retained through a 15 mm pre-tension pull.
Deterministic target selection, strict repeatability aggregation, and the
simulator-only teleoperation/D405 recorder are implemented and hardware-free
validated. Zero-load contact filtering, automatic opposed-pinch target
selection, and the exposed long-side cutting semantic pass the full regression
suite. The single-environment online-RL state baseline now has bounded bimanual
control, strict rewards/termination, seeded resets, a Gymnasium client, and a
PPO trainer. Remaining work is autonomous-policy convergence, D405 observations,
multi-worker batching, the full target/seed matrix, 32.5 N tear calibration,
and benchmark metrics.
See `dev.md` at the repository root for evidence and remaining work.
