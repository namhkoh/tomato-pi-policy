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

The benchmark scene directly sublayers the supplied `greenhouse/green_house.usd`
with the same Z-up, metre-scale stage metadata. It does not rescale or recreate
the greenhouse. In that original asset, the target gutter spans Z=0.671–0.888 m
over a cultivation floor at Z=-0.305 m, placing its top 1.193 m above the floor.

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
- Press `C` (or the window's **CUT target** button) to sever the selected petiole.
- Press `V` to select the next dynamic vine when more than one is enabled.
- Press `1` for the inspection view, `2` for the head D405, `3` for the left
  wrist D405, or `4` for the right wrist D405. The same choices are buttons in
  the **Vine Interaction** window.

Mouse pulling raycasts the original rendered GLB mesh and maps the hit to its
supporting rigid body. It therefore works across broad leaf blades as well as
thin stems without enlarging the 28 robot-contact proxies. The pull is a bounded
spring; `--drag-stiffness`, `--drag-damping`, and `--drag-max-force` tune it.
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

The builder keeps the mobile base dynamic, restores all 17 active custom URDF
capsules omitted by Isaac's importer, adds conservative base and left-gripper
contact proxies, and fits the supplied hardware under `greenhouse/robot_assets/`:

- one Intel RealSense D405 and bracket on the head;
- one D405 and bent bracket on the exact 18 mm-spaced M3 screw pair of each
  end effector, using the mirrored RB-Y1 v1.1 CAD mounting datum;
- the supplied deleafing knife directly on the right end-effector flange.

The original right gripper body and both tongs are fully removed from rendering
and collision. Their invisible URDF links/joints remain only to preserve the
exact v1.0 articulation and controller indexing; the right tool is knife-only.

For the knife, only the flat straight plate is tagged as a cutting surface. The
U-shaped arc is support geometry and cannot trigger a cut. The knife is rolled
about its unchanged blade axis so that arc faces upward in the ready pose and
the flat plate is presented cleanly toward the cut.

The default base pose is `(6.99114, 3.78000, -0.3050817)` m at -90 degrees yaw,
on the opposite side of the target gutter. It starts with about 229 mm chassis
clearance to that gutter and faces world -Y toward `Vine_0000/SubStem_00`. The
official SDK ready vector is retained as a separate reference; the launcher
replaces only the right arm with a collision-aware pre-contact IK pose. The
elbow stays in the inter-row aisle. Override the stance with
`--robot-position X Y Z` when testing a different vine or posture.

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

The accepted pose is stable, both wrist brackets coincide with the v1.1 CAD's
actual screw pair, and all three optical frames remain present. The
480-step report measures the settled flat blade 118.1 mm and upward U-support
65.2 mm from the actual lower-petiole attachment. All 34 robot rigid bodies
remain finite, the base settles with 2.08 degrees tilt, and the contact trace
contains only wheel/chassis support against the greenhouse floor. A final
approach controller, blade-to-petiole contact trigger, and robot-driven cut
remain the next gate before benchmark and RL work.

For live video, launch without `--headless` and use the interaction-window
camera buttons or keys `1`-`4`; the selected D405 becomes the active viewport.
For an image without the GUI, use `--capture-camera head`, `left_wrist`, or
`right_wrist` together with `--screenshot` as shown above.

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
`greenhouse/green_house.usd` directly rather than the built scene.

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
| `greenhouse_sim/cutting.py` | Runtime joint release, tear monitoring, and cut grading |
| `greenhouse_sim/greenhouse_scene.py` | Vine placement over the greenhouse stage |
| `greenhouse_sim/robot_hardware.py` | D405/bracket mounts and flat-blade cut semantics |
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

Run the tests with Isaac's interpreter (they are hermetic and use synthetic
assets, so they need no GLB files):

```bash
$ISAACSIM/python.sh -m pytest examples/greenhouse_sim/greenhouse_sim
```

## Status

Built and verified: asset pipeline, scene composition, stable
articulated vine physics, task-directed robot contacts, visible-mesh pulling,
foliage-area airflow, greenhouse integration, pull/recovery, and flush cutting.
Manual Shift-drag, airflow-motion, and cut acceptance is complete. Exact RB-Y1
v1.0 import, ready-pose stability, three D405 fits/optical frames, and right
knife-only end-effector semantics are also verified. Next: robot-to-vine reach/contact/cut
acceptance and 32.5 N tear calibration, then task definition, benchmarking,
and RL.
See `dev.md` at the repository root for evidence and remaining work.
