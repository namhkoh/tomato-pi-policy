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

Set `ISAAC_SIM_PATH` if Isaac Sim is not at `D:\isaac-sim` or `~/isaacsim`.

## Building the scene

```bash
# 1. Segment the vine GLBs and write per-organ USD (~2 s per vine).
D:\isaac-sim\python.bat examples/greenhouse_sim/convert_vines_to_usd.py

# 2. Compose vines into the greenhouse.
D:\isaac-sim\python.bat examples/greenhouse_sim/build_scene.py --beds 1 --plants-per-bed 8

# 3. Open it.
D:\isaac-sim\python.bat examples/greenhouse_sim/launch_greenhouse.py
```

Source assets stay under `greenhouse/`; everything generated lands in the
gitignored `data/greenhouse_sim/`. Step 1 re-checks each vine against its
metadata sidecar and exits non-zero if any invariant breaks, so a regenerated
asset set cannot quietly degrade the cut sites.

To render a frame without a display:

```bash
D:\isaac-sim\python.bat examples/greenhouse_sim/launch_greenhouse.py \
    --headless --screenshot data/greenhouse_sim/scene.png
```

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
D:\isaac-sim\python.bat -m pytest examples/greenhouse_sim/greenhouse_sim
```

## Status

Built: asset pipeline, scene composition, launcher.
Next: compliant vine physics, the cut/pull severance mechanism, RB-Y1
integration, then task definition and metrics. See `dev.md` at the repository
root for the design decisions and their rationale.
