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

Only the scripts that **start a simulator** — `launch_greenhouse.py` and
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

To render a frame without a display (this is how the scene is checked in CI or
over SSH):

```bash
$ISAACSIM/python.sh examples/greenhouse_sim/launch_greenhouse.py \
    --headless --screenshot data/greenhouse_sim/scene.png
```

Useful flags: `--beds` / `--plants-per-bed` / `--spacing` / `--seed` on
`build_scene.py`; `--eye` and `--target` to move the inspection camera on
`launch_greenhouse.py`.

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
| `greenhouse_sim/greenhouse_scene.py` | Vine placement over the greenhouse stage |
| `greenhouse_sim/usd_env.py` | Makes USD importable without booting Kit |
| `convert_vines_to_usd.py` | Asset build + invariant checks |
| `build_scene.py` | Scene composition |
| `launch_greenhouse.py` | Viewer / headless capture |

Run the tests with Isaac's interpreter (they are hermetic and use synthetic
assets, so they need no GLB files):

```bash
$ISAACSIM/python.sh -m pytest examples/greenhouse_sim/greenhouse_sim
```

## Status

Built: asset pipeline, scene composition, launcher.
Next: compliant vine physics, the cut/pull severance mechanism, RB-Y1
integration, then task definition and metrics. See `dev.md` at the repository
root for the design decisions and their rationale.
