# Development Log — Greenhouse Deleafing Benchmark

Goal: an Isaac Sim benchmark for evaluating VLAs on tomato **deleafing** (removing
orphan/lower leaves from high-wire vines), targeting demo collection → π0.5
finetuning → deployment on the Rainbow Robotics **RB-Y1** (sim and real).

Environment: Isaac Sim **5.1.0-rc.19** (Kit 107.3.3, omni.physx 107.3.26, USD 0.24.5)
at `D:\isaac-sim`, Windows 11. Current integration branch `koh-dev/online-rl`
(fork of openpi).

Branch ownership: `koh-dev/deleaf` contains the simulator physics, grasp/cut
benchmark, IK, and generic simulator mailbox/recorder. The connected physical
RB-Y1 read-only state publisher is isolated on `koh-dev/rby1`, which starts from
the latest deleaf commit. Teleoperation is stopped for the online-RL phase.

## Status

- [x] Recon: assets, Isaac 5.1 APIs, RB-Y1 SDK, repo conventions, prior art
- [x] Verified: Isaac Sim launches headless; greenhouse stage opens (7667 prims,
      3433 meshes); `breakForce`/`breakTorque`/`excludeFromArticulation` author OK
- [x] Verified: fused vine GLBs decompose into per-organ connected components
- [x] Asset pipeline: GLB → organ graph → structured USD (all 20 vines)
- [x] Greenhouse composition + launch (renders headless; see below)
- [x] Physics-ready vines inherit exact `gh_tomato_test.usd` stem root/yaw
      frames while preserving the supplied gutter and bed transforms
- [x] Compliant vine physics — stable per-organ articulations (biological calibration pending)
- [x] Cut severance — verified: severed organ detaches, plant stays connected
- [x] Trellis support + stable interaction-contact physics (evidence below)
- [x] Physics-enabled greenhouse integration + automated pull/cut acceptance
- [x] Visible-mesh mouse pulling + explicit foliage-area airflow
- [x] Manual viewport acceptance: real Shift-drag grab/release and UI/keyboard cut
- [ ] Sustained pull/tear validation against the 32.5 N threshold
- [x] RB-Y1 Model A v1.0 import, greenhouse placement, and stable ready pose
- [x] Supplied D405 head/wrist brackets fitted; right tongs removed and replaced by the deleafing knife
- [x] Physical cut gate: real leading-edge contact, force/work, direction, and centre crossing
- [x] Protected-contact ledger for main stem, non-target organs, neighbouring vines, structure, and robot
- [x] Bi-manual task state/retention support: left grasp -> right cut -> transport -> floor deposit
- [x] Stable non-contact robot/vine startup after collider deduplication and wrist-envelope correction
- [x] Leaf-blade/robot physical contact with target-only left-finger grasp semantics and non-pausing teleop contact monitoring
- [x] Robot-to-vine gripper/blade contact and cut validation
- [x] Hardware-bounded dual-arm approach + complete grasp/cut/transport/deposit acceptance
- [x] Simulator-only leader-arm command bridge + synchronized D405/action recording
- [x] Deterministic target selection + strict isolated-process repeatability runner
- [x] Fresh opposed grasp acceptance on `Vine_0002/SubStem_02`: exact distal body retained through 15 mm pre-tension
- [x] Synchronous online-RL environment: bounded bimanual actions, strict state/reward/termination, full articulation reset, seeded airflow/pose variation
- [x] Loopback client, optional Gymnasium wrapper, and reference PPO rollout/update/checkpoint path
- [x] Four-worker shared-policy PPO collection with independent live Isaac physics servers
- [x] Grasp-first RL curriculum: strict expert replay, behavior cloning, guarded PPO, and 8/8 unseen-seed physical grasp evaluation
- [x] Rendered/headless RL timing parity with synchronized four-camera policy evidence
- [ ] Live lab-teleop selected-leaf pinch/grasp/cut acceptance
- [ ] Record synchronized successful trajectories and prepare the π0.5 dataset/export contract
- [ ] Lab leader-arm hardware validation + multi-target physical repeatability acceptance
- [ ] Converged deleafing policy, D405 RL observations, and benchmark-wide target curriculum
- [ ] Benchmark task definition + metrics
- [x] VLM cut-point evaluation architecture assessed on `koh-dev/vlm-eval`
- [x] Qwen3-VL OpenAI-compatible RGB/prompt/validated-JSON/overlay smoke path
- [ ] Private simulator target/cut/protected-region projection and quantitative VLM labels
- [ ] Leakage-safe RGB/depth/calibration/mask capture and public/private VLM manifests
- [ ] Provider-neutral GPT/Claude/Gemini/open-weight runner and cut-point scorer

## VLM cut-point evaluation assessment, 2026-09-03

The next perception increment has been designed on `koh-dev/vlm-eval`; the full
architecture and acceptance criteria are in
`docs/vlm_cutpoint_evaluation.md`. The recommended boundary is an offline,
provider-neutral evaluation pipeline: Isaac captures immutable synchronized
head/left-wrist/right-wrist RGB plus private depth, calibration, semantic masks,
and exact cut geometry; hosted or local VLMs receive only RGB and the natural-
language instruction; a separate scorer maps a strict image-space response back
to the simulator's existing target, stub-length, direction, protected-contact,
and physical cut semantics. Provider calls must not run in the 240 Hz physics
loop, and VLM proposals remain advisory until held-out sim and real-D405
validation passes the existing planner and safety gates.

The first implementation gate is intentionally the dataset contract rather than
provider SDK integration: create a 50-sample, plant-disjoint pilot with positive
and no-safe-cut examples, all three cameras, synchronized private labels, and
human-review overlays. Inputs must exclude depth, masks, world geometry,
`SubStem` labels, robot state, UI selections, and target highlights. The current
`tomato_glb_30` asset stays an OOD/topology-shift set until its changed organ
mapping is validated. Only after every pilot projection and leakage check passes
should the common runner add OpenAI, Claude, Gemini, and separately hosted
open-weight adapters.

### Qwen3-VL-32B endpoint smoke, 2026-09-04

A dependency-light OpenAI-compatible runner now sends an unannotated RGB frame
plus a coordinate-explicit agronomic prompt and requires the canonical
`greenhouse.vlm_cutpoint.v1` response. It records source/submitted image hashes,
raw provider output, validated prediction, latency/usage, and a human-review
overlay; credentials and base64 pixels are excluded from artifacts. The point
is the primary output for future depth back-projection, while a tight petiole
box diagnoses whether the model selected the intended structure.

The supplied Qwen3-VL-32B-Instruct Cloudflare endpoint authenticated and exposed
the expected model. Text inference passed. Live testing found two deployment-
specific limitations: vLLM returned HTTP 500 for the Isaac PNG and a larger
quality-95 JPEG, and strict `json_schema` response formatting also returned 500.
An in-memory quality-92 RGB JPEG with unchanged dimensions plus `json_object`
formatting succeeded; the immutable PNG remains the scored source and both
hashes are retained. Six focused schema/prompt/encoding/overlay tests pass.

On synchronized frame 000092, Qwen correctly returned `target_not_visible` with
0.0 confidence for the head view containing no plant. It returned schema-valid
`uncertain` decisions at 0.3 confidence for both wrist views, citing occluded
attachment and depth ambiguity instead of hallucinating a cut point. A separate
vine-visible head capture produced the same conservative abstention because the
petiole junctions were too distant for a 2-5 mm cut decision. Latency was
2.23-2.99 seconds on successful trials.

These are valid transport, schema, semantic-reasoning, and abstention results,
but not cut-localization accuracy evidence: the archived frames do not contain
private projected target labels and none resolves an attachment well enough for
a safe positive. The next gate is therefore to project the selected 3D petiole,
2-5 mm flush-cut region, broader mechanical cut zone, and protected geometry
into each camera with exact intrinsics/extrinsics, then capture deliberately
visible positive and occluded/no-safe-cut examples. Only those private labels
can score box IoU, point error, stub error, direction, hazards, and abstention.

## Findings

### Vine assets — `greenhouse/tomato_glb_20/` (2026-08-06)

20 vines, `tomato_XXX.glb` + `tomato_XXX.json` sidecar. Metadata gives units (m),
height, organ counts, and an **attachment graph**: `SubStem_XX` / `Truss_XX` /
`Fruit_XX` / `Flower_XX` with parent + 3D attach point.

The GLB itself is **one node, one mesh, no hierarchy** — organs are fused and split
only by material (`TomatoStem`, leaf `Material.0xx`, `FruitRipe_r_c`). No skins,
animations, or morph targets. ~528–557k verts / 755–791k tris per vine, of which
the stem material is ~83% of triangles.

Critically, **connected-component analysis recovers the organs** (tomato_000):

| Material class | Components | Interpretation |
|---|---|---|
| `TomatoStem` | 121 | 1 main stem (118,560 tris, full height) + 120 laterals |
| leaf materials | 113 | 109 leaf blades (~895 tris each) + 4 flowers (~5.4k) |
| `FruitRipe_*` | 8 | 5 fruits (body + calyx parts) |

Component counts match the metadata exactly (`leaves: 109`, `flowers: 4`,
`fruits: 5`). A radial profile of the main-stem component shows r99 ≈ r50 above
0.5 m, i.e. it is a clean tube with **no laterals fused into it** — so laterals are
genuinely separable. A nearest-neighbour proximity graph over components yields
~25 direct children of the main stem (vs 18 sub-stems + 3 trusses expected) and
chains up to depth 8 (petiole → petiolule → leaflet).

Coordinates: glTF is **Y-up**, metadata is Blender **Z-up**; the mapping
`(x, y, z)_meta → (x, z, −y)_glTF` was verified by landing `Fruit_00`'s attach
point on that fruit's bounding-box centre.

**Anatomy note.** 18 sub-stems and 109 leaves means `SubStem_XX` is the **petiole of
a compound leaf** and the 109 "leaves" are its leaflets. The 18 sub-stems are
therefore the deleafing targets, and each sub-stem's attach point on the main stem
is the agronomically correct (flush) cut site.

### Greenhouse stage — `greenhouse/green_house.usd`

Z-up, meters, `defaultPrim=/World`. Two cultivation zones (`Main_Cultivation_Zone`,
`..._01` offset +24.9 m in Y), each with 4 bed rows = **256 BedSet groups**. Each
BedSet payloads `objects/Bed.usd` + 4 pipes and owns a local `Strings/Cylinder_0X`
group of 6 trellis strings — the natural anchor points for vines.

The generated benchmark scene loads this source without rescaling or rebuilding
it: `data/greenhouse_sim/scenes/deleafing_bench.usd` has
`../../../greenhouse/green_house.usd` as its root sublayer, and both layers are
Z-up with `metersPerUnit=1.0`. A direct composed-vs-source bounds audit found the
same 256 BedSets and identical first target gutter bounds: Z=0.671233–0.888308 m.
The authored cultivation floor is Z=-0.305082 m, so the gutter top is **1.193390 m
above the floor**. The gutters therefore do look high, but that height comes from
the supplied original greenhouse rather than a loader transform or unit error.

Physics present: only 3 ground `Plane` prims with `PhysicsCollisionAPI`. **No
PhysicsScene, no rigid bodies, no colliders** on beds/pipes/walls. No cameras. No
tomato stems referenced into the stage yet.

Object USDs are Y-up/cm and are brought in via payload arcs with auto-inserted
`unitsResolve` xform ops (rotateX 90, scale 0.01).

### Current benchmark source and vine placement (2026-08-10)

The generated benchmark now sublayers `greenhouse/gh_tomato_test.usd` and
selects `Gutter_01` without changing any gutter or bed transform. Its embedded
legacy `Stems` scopes are disabled only in the generated layer and replaced by
the existing physics-ready vines. Default `source` placement maps each
`tomato_NNN.usd` to its matching `tomato_stem_NNN` source prim and converts the
original Y-up frame to an exact Z-up root translation plus one of the authored
+/-90 degree row facings. For example, `Vine_0000`/`tomato_000.usd` now uses
`(15.830962741, 5.580149839, 0.392986681)` and yaw `+90 degrees`, exactly
matching its supplied Side_2 stem frame. Procedural bed placement remains an
explicit opt-in mode.

The previous generated root/yaw put the canopy into the wall-side robot aisle,
then target-relative base placement still admitted arm capsules through lower
foliage. Base planning now rejects fixed torso/head overlap, both actual waiting
arm postures, each solved left grasp arm, the wrist camera envelope, and
inter-arm overlap before the robot is authored. The 240-step acceptance report
`data/greenhouse_sim/source_placement_contact_check6_20260810.json` completed
with `succeeded=true`, zero robot-vine contact pairs, 0/121 runaway organs,
68.603 mm maximum compliant vine motion, and a stable 34-body robot.

The default interactive target is now the exact Side_1 source placement
`Vine_0002/SubStem_00`, with RB-Y1 yaw `+90 degrees` in the wider negative-Y
inter-gutter aisle. The accepted base is `(10.639222, 4.799768, -0.152541)` m;
the selected petiole is 297.253 mm directly forward. The 240-step report
`data/greenhouse_sim/wide_aisle_contact_final_20260810.json` passed with zero
robot-vine contact pairs, zero runaway organs, 17.495 mm maximum compliant vine
motion, and a stable robot. Physics-vine selection now promotes an explicitly
requested non-first vine before filling the physics budget.

All bimanual Cartesian directions (grasp approach, counterpull, blade stroke,
retract, and orphan transport) are now expressed in the robot's yaw-relative
task frame. Live arm capsules are checked against vine capsules at endpoints
and across every command chord; this exposed the former missing arm-vine gate.
The first strict `Vine_0002/SubStem_00` left-approach rerun rejected the route
without motion through the canopy and recorded zero unsafe contacts, so the
new-side full bimanual episode remains an open re-planning item rather than an
accepted result. Exact vectorized capsule batches replace the equivalent scalar
distance loop to keep that stronger gate practical.

### Physical leaf contact, gripper direction, and wrist mounts (2026-08-11)

The arm-through-leaf failure was not a solver-timestep problem: foliage was only
render and mouse-raycast geometry, while the stable `interaction` collider set
contained stems and petioles. Interaction mode now authors one 3 mm-minimum,
PCA-oriented contact box per foliage organ (115 on `tomato_002`). Each box rides
the same rigid link as its rendered leaflet and is filtered against every other
plant body, avoiding self-depenetration explosions while preserving external
robot/tool collision. The boxes retain their owning `SubStem_*` identity, so an
opposed left-finger pinch on the selected branch can establish the existing
branch grasp. Neighbouring foliage contacts remain physical and benchmark-unsafe,
but default `--teleop-contact-policy monitor` no longer pauses or latches command
consumption; opt-in `rollback` returns toward the last contact-free target.

A 480-step fixed-station soak at `(10.639222, 4.30, -0.152541)` m authored 127
plant contact shapes, kept 384 links and all 34 robot bodies finite, produced
zero runaway organs and no robot-vine contact pairs. A stricter staged approach
from the old 4.799768 m base then produced seven real foliage contact pairs and
was rejected (`succeeded=false`) before grasp/cut. This is the intended evidence
that leaves no longer tunnel through the robot and that neighbouring contact is
not silently accepted. It also supersedes the old near-canopy base as an
autonomous acceptance pose now that leaf volume is physical. With the measured
lab pose, the first 4.30 m GUI launch correctly latched on one 2.014 mm
neighbouring-leaf/finger spawn overlap. Relaunching only 50 mm farther back at
`(10.639222, 4.25, -0.152541)` m reached `running`, accepted 432 fresh whole-body
commands, retained 0.998478 left-gripper openness, and had zero robot-vine pairs
or unsafe latch. Collision-aware replanning of the autonomous route through the
dense physical canopy remains open.

The live left gripper convention was reversed because motor ID 1 uses numeric
minimum=open and numeric maximum=closed. The read-only publisher now computes
`(max_q - position) / (max_q - min_q)`. Live verification read position
-4.488427 with session stops -4.502233/4.570971 and published 0.998478 openness,
exactly matching the endpoint-derived value. No physical robot or gripper
command path was added.

A later read-only audit against the operator's wrist screenshot, the supplied
`RBY1_Example_setup.FCStd`, and the standalone bent-bracket STEP corrected this
initial interpretation. The desired screw pair is on the wrist force-sensor
plate above the tool: bolt origins are `(+/-9, -42, 38.5)` mm and the mounting
face is at `y=-39` mm. The normalized bracket bolt centres are
`(+/-9, 56.919538, 30.119184)` mm, giving the exact root translation
`(0, -0.098919538, 0.008380816)` m on the right. Operator visual verification
showed that the generated left EE frame still requires an explicit outside-face
mirror: its root is `(0, +0.098919538, 0.008380816)` m with a 180-degree local Z
rotation. The earlier negative-Z transforms placed both cameras beside the
knife/gripper; the corrected positive screw-plate height is retained on both
sides. The generated robot USD/manifest were rebuilt.

The first live post-change report captured an active `Foliage_155`/left-finger
contact with `contact_policy=monitor`, `unsafe_latched=false`, and
`hold_active=false`; accepted mailbox commands then increased from 1,706 to
1,966, confirming the old permanent contact freeze was removed. Source-plant
ancestry maps that visible leaf to `Vine_0002/SubStem_02`, not the selected
`Vine_0002/SubStem_00`, which explains why collision worked while target-only
grasp attachment did not.

The follow-up live report
`data/greenhouse_sim/physical_robot_teleop_leaf_grasp_20260811.json` isolated two
additional root causes. PhysX contact-offset/CONTACT_LOST records with zero
impulse could overwrite a real pinch with a zero-force one-finger diagnostic,
and the original semantic cutting strip occupied the plate's distal local `-Y`
end where the U-support extends beyond it. The run therefore closed the gripper
to 0.00149 openness without establishing a grasp and recorded only non-cutting
arc contact; no cutting-edge target progress was possible.

The grasp manager now discards zero-impulse records before persistence scoring.
While closing, positive contact on both left fingers automatically retargets to
the physically pinched branch and records
`trigger=opposed_finger_physical_pinch`; one-finger brushes cannot retarget, and
`T` remains an explicit fallback. The knife semantic is now the unobstructed
outer 2 mm strip along the flat plate's long local `-X` side, with local `-X`
cut travel and local `+Y` edge axis. The support's closest X bound remains over
10 mm inside that strip. The fitted RB-Y1 USD was rebuilt, focused geometry and
contact-policy tests pass, and the complete suite passes 126 tests with one
PhysX-only skip. Manual live opposed-pinch and force-gated cut acceptance is
still intentionally open; the strict autonomous `Vine_0002` canopy route also
remains a separate collision-aware replanning blocker.

### Optimized vine assets — `greenhouse/tomato_glb_30/` (2026-08-11)

The collaborator-supplied directory contains 30 GLB/JSON pairs. It is materially
lighter but is not a drop-in mesh swap for the existing benchmark semantics. A
read-only `tomato_002` comparison reduced resolved triangles from 819,490 to
374,714, while connected components increased from 264 to 840, foliage labels
from 115 to 374, and `SubStem` targets from 20 to 44; the old/new organ label
sets do not match. Migration is therefore deferred until the current grasp/cut
gate is accepted and must include sidecar/topology validation, target remapping,
physics-capacity profiling, and placement regression rather than changing
`_DEFAULT_VINE_DIR` silently.

### Isaac Sim 5.1 physics capabilities

- **No topological cutting of deformables exists**, and NVIDIA's own guidance is to
  fake it by pre-splitting geometry and disabling attachments (IsaacSim issue #258
  closed without a feature). Attachments have **no break force** and can be added
  but never deleted at runtime.
- **Newton is not in 5.1 at all** — it arrives with Isaac Sim 6.0. No `newton`
  package or extension is installed.
- FEM deformables are disqualified as the benchmark substrate: no static friction
  (grippers slip), OOM around 256 trivial bodies, no break threshold, and
  undocumented GPU determinism.
- `physics:breakForce` is **silently ignored on articulation joints** — breakable
  joints must be maximal-coordinate.

## Decisions

### D1 — Vine physics: compliant articulated capsule chains, not FEM

Model each organ as a chain of rigid capsule links joined by compliant D6 joints
with angular drive stiffness from Euler–Bernoulli beam theory (`Kp = E·I/ℓ`,
`I = πr⁴/4`, `Kd = 0.1·Kp`), following OrchardBench. This delivers the bending,
droop, and pull compliance the task needs while staying deterministic and fast.

Rationale: the user's requirement is *deformable behaviour* (bend, pull, cut), not
the FEM solver specifically. FEM in Isaac 5.1 cannot cut, cannot hold a grasp
(no static friction), and caps scale at tens of environments — all fatal for a
data-collection benchmark. Compliant chains give the same observable physics.
FEM is kept only as an optional visual/validation comparison.

### D2 — Cut primitive: in-place joint release (superseded architecture)

Sever by releasing the petiole's base joint (zeroing drive gains and freeing the
constraint) when the tool's blade plane intersects the link and the jaws close;
`breakForce` on the same joint provides tensile **tear** when the policy pulls
instead of cuts, which distinguishes clean cut from tear as a metric. The whole
plant stays out of any `PhysicsArticulationRootAPI` subtree so `breakForce` is
honoured. Never delete prims at runtime — deactivate.

Stub length stays continuous (not quantised by link resolution) by measuring the
blade plane against the parent stem surface geometrically and shrinking the
parent-side capsule to the cut plane.

### D3 — Asset pipeline: hybrid proximity graph + generator metadata

Segment each GLB into organ components, build a parent graph by closest-approach,
then label the primary laterals with the generator's `SubStem`/`Truss` attach
points. Metadata is authoritative for semantics and cut sites; the graph is
authoritative for which geometry detaches together. Validation invariant: every
leaf blade maps to exactly one sub-stem, and per-vine totals match the sidecar
counts.

### D4 — Code layout: `examples/greenhouse_sim/`, per repo convention

openpi examples are self-contained clients with their own interpreter that talk to
`scripts/serve_policy.py` over websocket (libero runs Python 3.8 against a 3.11
main env). The sim therefore runs inside **Isaac Sim's bundled Python** with
`openpi-client` pip-installed into it; Isaac deps never enter the root
`pyproject.toml`. RB-Y1 SDK is vendored as a `third_party/` submodule per
`.gitmodules` convention. Generated USD goes to gitignored `data/`.

Offline asset tooling depends only on packages already present in Isaac's bundled
Python (numpy 1.26, scipy 1.15, pytest 9, plus `pxr` for USD authoring), so the
pipeline is reproducible from a bare Isaac install with nothing pip-installed.

### D5 — Headline metric: residual petiole stub length

No existing plant-manipulation benchmark measures it, and it is agronomically
decisive: flush pruning wounds are near-absolutely resistant to *Botrytis cinerea*
while petiole stubs are highly susceptible (Beyers et al. 2014), with lesions
advancing 0.3–0.5 cm/day. Bins: ≤5 mm flush / 5–20 mm marginal / >20 mm risk.

### D6 — Benchmark cuts require verified leading-edge evidence and bi-manual order

The full flat knife plate remains a physical collider, but it is not itself a
cutting semantic. Only the outer 2 mm strip along the unobstructed long local
`-X` plate side can accumulate cut work. PhysX leading-edge impulses are the
preferred evidence. Thin rigid petiole capsules can miss those callbacks even
when the exact edge segment intersects them, so the already counter-held active
target may instead produce a reported compliant reaction from exact
edge-to-centreline distance. This path is unavailable without a prior opposed
left grasp and outside the target radius plus 1.5 mm contact tolerance.

A valid severance still requires the measured petiole cut zone, transverse edge
and motion alignment, forward edge motion, at least the petiole's 66.3 N cut
force, sustained work through its diameter, and a sweep across the target
centre. Separate taps do not combine. The U support, blade face, direct `C`
debug release, tensile tear, unheld target, wrong direction, low force, and
non-target geometry cannot produce a benchmark cut.

The required task order is explicitly bi-manual: both left fingers establish a
loaded grasp on the selected petiole, the right leading edge physically severs
that same target without protected contact, the left grasp retains and moves the
orphan at least 0.15 m, then releases it into the floor drop zone. A simulator
cut still follows physics if the policy failed to grasp first, but that episode
is recorded as a benchmark failure rather than silently counted as success.

## Validation

### Organ segmentation, all 20 vines (2026-08-06)

Sub-stem, truss, and foliage counts match the metadata sidecar **exactly on every
vine**; no foliage organ fails to trace to a primary lateral; the adjacency search
reaches every stem organ, so the graft-onto-root fallback never fires. Junction
gaps are 0.3 mm median / ~5 mm p95 / 9.9 mm max against a 10 mm adjacency radius.
Runtime ≈ 1.8 s per vine.

Sub-stem labels are **monotonic in height on all 20 vines**, matching the
metadata's own ordering — which is what the bottom-up "remove the lowest leaves"
rule depends on.

The assigned junctions sit a systematic **+28 mm above** and ~7 mm lateral of the
metadata attach points. This is expected rather than an error: the generator
records attach points on the main-stem **centreline**, while segmentation computes
the **surface contact** where the two organs actually meet. The horizontal offset
matches the main-stem radius. Surface contact is the correct reference for
residual stub length (D5), so the computed junction is the one to keep.

Risk: the worst junction gap (9.9 mm) is close to the 10 mm adjacency radius. A
future asset generation with looser junctions could silently fall back to
grafting. `convert_vines_to_usd.py` now asserts every one of these invariants
and exits non-zero, so this cannot regress unnoticed.

### Scene composition (2026-08-06)

Greenhouse layout as measured, not assumed: beds are 5 m troughs whose planting
surface is at **Z = 0.888 m**, spaced 5.7 m apart, with two 5 m trellis strings
at each bed's ends (not per-plant strings). 256 BedSets across two zones.

Vines compose in at 0.25 m in-row spacing with seeded yaw jitter, and organ
prims stay addressable through the reference
(`/World/Vines/Vine_0000/MainStem/SubStem_06`), with materials bound. A headless
render confirms textured foliage, fruit trusses, and plants seated in the gutter.

**Asset defect found and worked around**: the greenhouse `DomeLight` references
`/home/jhlee/Desktop/...`, an absolute path from the machine the asset was
authored on. It fails to load on any other machine and leaves lighting to a
renderer fallback — unacceptable variation for a vision benchmark. The scene
layer now clears unresolvable light textures, leaving a uniform sky at the
authored colour and intensity. Worth fixing in the source asset too.

Open question for later: the source vines lean substantially (canopy extends
~0.75 m to one side). Real high-wire tomato is trained near-vertical up a string
and only leaned along the wire. Whether to correct this at placement time is a
fidelity decision that should be made deliberately, not by accident.

### Cut mechanism (2026-08-06)

`cut_demo.py` rigs `tomato_000` as **396 capsule links / 396 joints / 120
severable organs**, settles it under gravity, and cuts the lowest sub-stem.
Result: the severed organ detaches and falls while the rest of the plant stays
connected. The severance primitive is proven.

Engine behaviour worth remembering:

- Authoring `physics:jointEnabled` **at build time** and only setting its value
  at runtime avoids forcing a PhysX resync of the whole plant on every cut.
- Kit swallows stdout and `--/app/fastShutdown=True` masks non-zero exits, so
  simulator scripts must write a machine-readable report file. Several apparent
  "crashes" during development were simply the final result never being written.

### Trellis support, resolved (2026-08-06)

Clips implemented as compliant world anchors every 0.30 m, plus a ground plane.
Stem sag over the clipped span fell from **435 mm to under 24 mm**. Four things
had to be right, and each was found by measurement rather than guessed:

1. **Springs alone are not enough.** A clip needs a *travel limit*, because any
   sustained load walks a pure spring out indefinitely however stiff it is. A
   clip is a collar (±5 mm of play, then the wall), not a spring.
2. **Clip to the growing point.** Advancing a running counter by `spacing`
   silently drops the topmost clip whenever the remaining run is shorter than
   one interval — which is exactly the stretch whose deflection matters, since
   cantilever droop grows with the *cube* of free length. Leaving 0.3 m
   unclipped cost 0.45 m of fold-over. Now targets explicit heights up to
   `top − 0.15 m`.
3. **The fitted tip radius is a trap.** The render mesh tapers the stem to a
   point (0.5 mm fitted radius), and stiffness goes as r⁴, so the top of the
   stem came out ~10⁴× too floppy. Floored at 3.2 mm, the measured tomato apex
   radius (Gao et al. 2024). Collision geometry still uses the fitted value.
4. **Anchoring itself was never the problem.** An isolated test confirmed all
   four joint types behave: locked-D6-to-world and FixedJoint both hold exactly,
   a ±5 mm limited joint sags exactly 5 mm, a free body falls. Worth knowing
   before blaming the engine.

Residual: the top ~0.15 m of unclipped growing head still nods ~0.18 m. It sits
outside the deleafing work zone and is the floppiest part of a real plant, so it
is acceptable for now but should be revisited.

**Open bug — spurious tearing at start-up.** Authoring `breakForce` at the
measured 32.5 N detachment force makes *every* petiole snap off within the first
frames: solver transients during settling far exceed 32.5 N. Visualising the
colliders is what exposed this; every scalar metric had looked plausible while
the plant was quietly shedding its leaves. Tearing is therefore disabled by
default until it can be armed after settling, which needs care because PhysX may
not pick up a `breakForce` change at runtime.

**Lesson worth keeping:** render the physics, do not trust aggregate numbers.
Both the fold-over and the tearing were invisible in the metrics.

### Runaway organs: diagnosis (superseded by the resolution below, 2026-08-06)

Organs fly apart on start-up. Ruled out by measurement, not argument:

| Hypothesis | Test | Verdict |
|---|---|---|
| `breakForce` firing on transients | set tear force to 0 | not the cause |
| Stiff chain unintegrable at 240 Hz | — | **wrong**; see below |
| Gravity / collapse under load | run at `--gravity 0` | **not the cause** — identical |
| Self-collision | run with `--no-collision` | **the driver**: 73 → 21 runaways |

Gravity-independence is the key result: with zero gravity nothing should move at
all, so this was never collapse, load, or the stiff-chain integration limit
blamed earlier. That earlier diagnosis was wrong and is retracted.

**Fixed along the way (correct regardless):**

- *Joint frames.* Only local *positions* were authored, never local rotations.
  The angular drives therefore targeted zero relative rotation between two
  identity frames, asking every link to lie parallel to its parent and snapping
  the plant straight on frame one. Joint frames are now anchored to the child's
  rest orientation, so "zero" means the authored pose.
- *Collider size.* A capsule's true length is `height + 2·radius`, so a short,
  thick segment collided as a ball up to **4.2×** longer than its link, engulfing
  neighbours. Clamped to stay within its own segment.
- *Offline auditability.* `PhysxSchema` is imported lazily, so `vine_physics` can
  be inspected without booting Kit.

**Still open.** `UsdPhysics.CollisionGroup` self-filtering has no measurable
effect here — neither adding several hundred collider targets individually nor
including the physics scope and letting the collection expand. Since collision
is demonstrably the driver, the next step is the mechanism Isaac supports
reliably for this: `PhysxArticulationAPI` with `enabledSelfCollisions = False`.
That needs the articulation to build, which needs the chain decimated well below
~400 links — worth doing anyway for throughput, since demo collection wants many
plants per scene.

An offline rig audit (`scratchpad/rig_audit2.py` pattern) confirmed the rig
itself is sound: exactly one organ attaches to the world (the main stem), and
every joint anchor lies within 30 mm of its parent link.

### Honest verification, and a correction (2026-08-06)

An earlier claim that the vine was "stable" was wrong, and is withdrawn. It
rested on a tight-framed screenshot after **one second**, while the run's own
metric reported 72 of 121 organs past 100 mm — which was explained away rather
than investigated. A favourable picture was trusted over an unfavourable number,
which is the same failure as trusting numbers over pictures, in the other
direction.

Proper check: 10 s run, wide framing that would show anything ejected.

| t (s) | max | p95 | median | organs > 200 mm |
|---|---|---|---|---|
| 1 | 458 mm | 359 | 96 | 82 |
| 2 | 507 | 363 | 98 | 105 |
| 5 | 501 | 363 | 96 | 98 |
| 10 | 499 | 360 | 96 | 101 |

What this actually shows:

- **Not divergence.** Displacement plateaus after ~2 s and holds for the rest of
  the run, and the plant's extent *shrinks* (0.39 → 0.24 m wide, 1.89 → 1.65 m
  tall). An explosion grows; this contracts.
- **Nothing detaches.** The wide 10 s frame shows an intact plant with no debris,
  which the 64 position iterations fixed.
- **The plant goes limp.** It slumps up to 0.5 m at the extremities on start-up.
  That first-second lurch is what looks like an explosion in the viewport.

**Root cause, still unfixed: the angular drives are not applied at all.** Sweeping
`stiffness_scale` over 1e2, 1e4, 1e6 and 1e8 gives *byte-identical* results
(max 500 mm, p95 370 mm, extent 0.24/0.67/1.65). Six orders of magnitude with no
response is not "too soft", it is ignored, so the plant hangs as a free-jointed
chain instead of holding its authored shape. The calibrated `stiffness_scale`
constant is therefore meaningless and should be removed once the real cause is
found.

Caveat on method: the isolated drive harness is **not deterministic** between
runs (the same case gave −20.7 mm and +44.6 mm on two runs, and a free-translation
control flew −5.9 m then +6.9 m). Small A/B results from it cannot be trusted;
only the plant-scale sweep above is reliable, because it is identical across
runs.

Next step: drives on **articulation** joints are the well-trodden path in Isaac
(every robot uses them), whereas driven non-articulation D6 joints are not. That
points back at articulations, which need the chain decimated from ~400 links —
0.05–0.10 m segments instead of 0.02 m would bring it near 100.

### Visual meshes now ride the physics

`vine_visuals.py` parents each organ's GLB mesh to the rigid body carrying it, so
the rendered plant is the original art and the capsules stay hidden. Foliage and
fruit have no bodies of their own, so they ride the nearest ancestor that does —
which is what makes a severed leaf travel with its petiole.

Two USD traps found here:

- **Purpose is inherited by the whole subtree.** Parenting art under a
  `guide`-purpose capsule hides the art too, and authoring `default` on the child
  does *not* override it. Bodies are therefore plain Xforms with the collider and
  the art as siblings — which is the conventional structure anyway.
- `Gf.Vec3f(*p)` rejects numpy scalars; array attributes must go through
  `Vt.*Array.FromNumpy`, which is also far faster at these mesh sizes.

### Blocking problem — stiff chains are unstable in maximal coordinates (resolved below)

With physically correct stiffness the chain explodes. Beam theory at the floored
3.2 mm radius and 250 MPa gives ~1030 N·m/rad per joint, while a link masses
~1e-4 kg, so the natural frequency is enormous and 240 Hz cannot integrate it.
The stability ceiling is roughly `K < 0.25·I/dt²` ≈ 6e-4 N·m/rad — about **six
orders of magnitude** below the physical value. A real tomato stem is stiff and
light, and that combination is exactly what maximal-coordinate joints handle
worst.

Lowering stiffness to fit is not an option: it reintroduces the collapse.

**The fix is to move the plant into a PhysX articulation** (reduced coordinates),
which handles stiff serial chains stably. That was ruled out earlier because
`breakForce` is ignored on articulation joints — but that only matters if
tearing depends on `breakForce`. It does not have to: severance already works by
disabling a joint, and tearing can be implemented by monitoring joint force in
Python and disabling past the 32.5 N threshold. That yields both stability and a
tear model, and it removes the spurious-tearing bug at the same time.

This supersedes the "no articulation root" decision in D2.

### Runaway vine physics and contact: resolved (2026-08-06)

The default simulator no longer explodes. The original tomato GLB visuals and
all **396 structural links / 396 joints** are retained; the fix changes how those
links enter PhysX, not which vine is shown or which organs can bend and sever.

The failure was three coupled implementation problems:

1. **Articulation topology/API mismatch.** A single ~400-link plant articulation
   exceeds the practical Isaac 5.1 limit, while maximal-coordinate chains cannot
   carry the calibrated stiffness. The stable topology is one reduced-coordinate
   articulation per organ, rooted on an Xform, with cross-organ base joints
   explicitly excluded from both articulations. Isaac 5.1 also has no
   PhysxArticulationAPI fixBase attribute; the invalid call was removed and the
   main-stem world joint provides the fixed base.
2. **Dense rest-pose collision was physically invalid.** The artistic vine pose
   contains intentional intersections at petiole/stem junctions and among
   foliage. Giving every structural capsule CollisionAPI created hundreds of
   endpoint-touching/interpenetrating shapes and made PhysX resolve the authored
   plant itself as penetration. CollisionGroup and pairwise FilteredPairs did not
   make that mixed articulation graph reliable and were removed.
3. **Inertia depended on whether a link happened to have a collider.** PhysX
   used fallback inertia for non-contact links and shape-derived inertia for
   contact links, so adding robot contact changed the plant dynamics and could
   invalidate every transform. Each link now authors mass and diagonal inertia
   independently of contact. The minimum diagonal inertia is 1e-5 kg*m^2: the
   stable petiole-probe scale and a conservative lump for unresolved leaf/flower
   art riding the structural links. This is a numerical/effective inertia and
   must remain a domain-randomisation/calibration parameter, not be cited as a
   measured tomato material property.

The public collision modes now make the separation explicit:

| Mode | Purpose | tomato_000 result |
|---|---|---|
| interaction (default) | Ten non-overlapping main-stem zones at 0.20 m spacing plus one midpoint zone on each of the 18 real SubStem deleafing petioles | 28 contact colliders, stable |
| none | Constraint/inertia isolation | 0 contact colliders, same structural motion |
| all | Negative-control diagnostic for the raw dense capsule set | reproduces invalid transforms; do not use for episodes |

This is task-directed collision geometry, not deletion of vine physics. Every
organ still has mass, inertia, compliant joints, gravity response, visuals, and
a severable attachment. Contact is authored only where the RB-Y1 should touch
the plant for deleafing; internal foliage self-collision is intentionally absent
because the source rest pose is not collision-clean.

**Validation evidence (Isaac Sim 5.1, 240 Hz):**

| Check | Result | Report |
|---|---|---|
| Contact off vs default contact, 120 steps | identical 7.2 mm maximum stem sag; 0/121 runaway organs in both; no invalid-transform or broadphase log entries | data/greenhouse_sim/interactive_inertia_floor_none.json and interactive_inertia_floor_contact.json |
| Default contact, 10 s / 2400 steps | 7.2 mm maximum stem sag; 0/121 runaway organs; no PhysX errors | data/greenhouse_sim/interactive_final_10s.json |
| Flush cut with ground contact | SubStem_00 dropped 233.22 mm; rest of plant moved 2.44 mm; 0 mm stub/quantisation; succeeded=true; no PhysX errors | data/greenhouse_sim/cut_contact_final.json |

The cut run proves the stable mode is dynamic rather than frozen: the severed
petiole falls and contacts the ground while the clipped parent vine remains
compliant and connected.

Remaining calibration work is deliberately not hidden by the stability result:
validate gripper forces and friction on the 28 interaction zones with the RB-Y1
finger geometry; calibrate the effective inertia distribution against tracked
real-vine deflection; and validate sustained-force tearing at 32.5 N. Full
leaf-blade collision proxies may be added only after producing a collision-clean
proxy asset, never by re-enabling the raw dense structural capsules.
### Physics-enabled greenhouse and mouse interaction (2026-08-06)

`interactive_greenhouse.py` is now the acceptance launcher between isolated-vine
physics and robot integration. It opens the generated greenhouse, switches to
the USD session layer, deactivates the selected static vine references there,
and rebuilds the corresponding source GLBs as articulated rigs at the exact
same bed transforms. The generated scene and source assets remain unchanged.
One of eight placements is dynamic by default; `--physics-vines` can increase
that only when the GPU/performance cost is intentional.

Integration exposed and fixed three non-solver errors:

1. Greenhouse translation was correctly applied to geometry points but was also
   being added to mesh normals. `vine_visuals.attach_organ_visuals` now accepts a
   separate direction transform, and both launchers pass translation-free
   `TransformDir`/GLB direction conversion for normals.
2. The severed-organ catch plane was still centred at world XY = (0, 0).
   `add_ground_plane` now accepts `centre_xy`, and each plane is centred beneath
   its greenhouse vine placement.
3. The first automated pull used `apply_force_at_pos(..., "Acceleration")`, a
   mode PhysX 5.1 rejects for off-centre force application. The probe now reads
   the selected rigid body's authored mass and applies `mass × requested
   acceleration` in supported `Force` mode. This also makes the requested pull
   reproducible across link masses.

The first manual acceptance exposed that native PhysX mouse grabbing was not a
usable interaction surface. Five UI cuts were recorded correctly, but no
`POINT_GRABBED` event appeared. Each SubStem exposed only one hidden structural
capsule: typically about 20 mm long and 5 mm across, just a few pixels at the
inspection-camera distance. Leaf blades had no colliders at all. Shift-clicking
the missed/hidden target also drove Isaac 5.1's transform selector into repeated
`KeyError: <class 'NoneType'>` callbacks. Raising the native coefficient alone
would not solve the missing pick surface (though Isaac's own rope demo uses 10.0
rather than the original 1.0).

The corrected GUI backend disables that native collider-only grab. A viewport
Shift-drag now raycasts the **actual rendered GLB triangles**, walks from the hit
Visual prim to its supporting rigid body, holds the hit at its original camera
depth, and applies a damped spring force at that material point. Defaults are
10 N/m stiffness, 0.02 N*s/m damping, and a hard 1 N force cap. This permits
pulling broad leaf blades, petioles, and the visible stem without inflating the
28 robot-contact proxies or reintroducing rest-pose contact. A four-frame camera
warm-up removes a startup race where the overlay could read the previous
viewport projection. Grab/release count, selected Visual/body, peak target
offset, and peak force are persisted in the report.

The answer to "should a stationary vine move?" is conditional: once damping has
removed transients, a vine in still air should settle, not jitter forever. To
model greenhouse motion honestly, `vine_interaction.Airflow` adds an explicit
aerodynamic load. For every compound petiole with foliage, it sums the actual
one-sided triangle area, finds the foliage centroid, and applies
`F = 0.5*rho*Cd*A*v^2` there with deterministic low-frequency speed and direction
variation. The current GUI default is 1.0 m/s at 0.18 Hz over 14 populated
petioles (0.1593 m2 total measured foliage area); `--airflow-speed 0` restores
still air. The speed is an initial interactive default and remains a calibration
and domain-randomisation parameter, not a measured condition of this greenhouse.

`[` / `]` still select deleafing petioles, `V` selects a dynamic vine, and `C` or
the interaction window releases the selected base joint.

**Revised integrated acceptance (Isaac Sim 5.1, 240 Hz):**

| Check | Result | Evidence |
|---|---|---|
| Greenhouse composition | 8 source placements found; Vine_0000 replaced in-session by 396 links, 6 clips, 120 severable joints, and 28 robot-contact colliders | `data/greenhouse_sim/interactive_greenhouse_airflow_pull_cut_10s.json` |
| Visible-mesh raycast pull | renderer selected a foliage Visual and mapped it to its supporting body; 0.7551 N peak force produced 29.95 mm motion and 1.80 mm residual under airflow; grab/release counters both 1; finite=true | `data/greenhouse_sim/interactive_greenhouse_visual_pull_probe.json` |
| Airflow, 10 s / 2400 steps | 14 foliage targets; 2.343 mm peak-to-peak axis motion; 2.515 mm maximum displacement; finite=true; 0 runaway organs | `data/greenhouse_sim/interactive_greenhouse_airflow_pull_cut_10s.json` |
| Bounded body-force pull after airflow | 0.1674 N produced 3.723 mm peak deflection; after recovery residual was 0.632 mm; finite=true | same report |
| Flush cut after airflow and recovery | exact BaseJoint changed enabled=true -> false; detached organ dropped 238.27 mm and travelled 247.82 mm; finite=true | same report |
| Error scan | no Python tracebacks, PhysX errors, invalid transforms, broadphase faults, NaNs, or explosion signatures | both reports' `.log` files |
| Rendered composition | greenhouse beds/trellis and the updated vine render together from the inspection camera | `data/greenhouse_sim/interactive_greenhouse_acceptance.png` |
| Manual viewport acceptance | user accepted visible Shift-drag pull/release, stationary airflow motion, and cutting in the relaunched greenhouse session | direct acceptance, 2026-08-06 |

The automated GUI probe reaches the same renderer raycast, Visual-to-body map,
and bounded-force path as Shift-drag; the headless probe independently checks
body compliance and the cut joint. Direct viewport acceptance was completed on
2026-08-06 after a clean relaunch: visible foliage could be Shift-dragged and
released, airflow produced acceptable stationary motion, and the cut mechanism
worked. This clears the interaction gate for RB-Y1 integration. Benchmarking
and RL remain gated on stable robot/tool/camera integration and contact testing.
Regression status: 36 passed, 1 failed, and 1 skipped in the complete hermetic
greenhouse suite. The only failure is the unchanged
`test_arc_length_sampling_is_continuous`, whose 10 mm
absolute tolerance is exceeded by 0.17 mm; no skeleton code changed in this
milestone, so that numerical test issue is tracked separately rather than hidden
inside the robot integration. The skipped test requires `PhysxSchema` from a
running `SimulationApp`; the integrated live soak exercises that path.

### RB-Y1 Model A v1.0, D405, and knife integration (2026-08-06)

The fitted robot is rebuilt reproducibly by `build_robot.py` from the exact
physical-robot URDF,
`third_party/rby1-sdk/models/rby1a/urdf/model_v1.0.urdf` (SHA-256
`33cb8cd34abc0f58f0e65f8dc7b59acabf3fd62cb820b1be1f40d513578a65ae`).
The v1.2 simulator asset is intentionally not substituted. Import keeps the
mobile base dynamic, preserves fixed joints and URDF inertia, disables
self-collision, and changes only the two wheel drives to velocity mode.

Collision reconstruction corrected an earlier source audit: v1.0 contains
**17 active custom capsule elements**, not 26. The other nine matches are inside
XML comments and must not become live colliders. Isaac's URDF importer handles
the non-standard capsules inconsistently: ordinary stage traversal did not show
usable shapes, but live contact traces later proved its instance proxies still
participated in PhysX alongside the restored siblings. The builder therefore
deactivates every importer-created `collisions` scope and restores the 17 source
capsules exactly once under authorable sibling scopes. The source also omits standard collisions for the base,
wheels, gripper bodies, and fingers; conservative proxies add 3 base/wheel and
3 retained left-end-effector/finger shapes. The original right end-effector body
and both right fingers are a knife-only tool slot: their visual and collision
scopes are inactive, while their invisible rigid links and joints remain so the
exact URDF articulation and controller joint indexing are not changed. Eight
more shapes cover the three fitted camera/bracket assemblies and two knife
components, for **31 collision shapes** on the generated robot.

`extract_robot_hardware.py` makes the supplied CAD reproducible without a
runtime FreeCAD dependency. The external
`D:\research\freecad-mcp\deleaf_knife.stl` was copied byte-for-byte to
`greenhouse/robot_assets/deleaf_knife.stl` (SHA-256
`0237eb46c8980cec4cd9b09623f72d55e6f2491928e673a67172294c3a5f8dbd`)
and split into its two disconnected components:

- the 30.362 x 71.480 x 13.000 mm **flat straight plate**, the only prims
  carrying `tomato:cuttingSurface=true`;
- the 6.000 x 62.301 x 50.918 mm **U-shaped arc**, explicitly non-cutting
  support geometry.

The plate projects along the right tool's -Z axis directly from the retained
`ee_right` flange frame; the obsolete 73 mm gripper-tip offset is no longer used.
The U arc is never accepted as a cutting surface. Both pieces retain collision
geometry, so support contact remains physical without corrupting cut semantics.

The supplied bent D405 bracket STEP was decomposed into the exact 27 x 59.920 x
34.619 mm bracket and 42.090 x 23 x 42 mm camera body while preserving the
authored bracket-to-camera transform. The first wrist fit incorrectly treated
the re-origin of that extracted STL as its mounting face, so the assemblies
were only approximately placed on the outer wrist surfaces. The authoritative
RB-Y1 v1.1 assembly `RBY1_Example_setup.FCStd` fixes the actual interface at the
18 mm-spaced M3 pair: bolt origins are `(x, y, z) = (+/-9, -42, 38.5)` mm,
3 mm outside the force-sensor mounting face at `y=-39` mm, in each mirrored
end-effector frame. In the normalized bracket mesh those same centres are
`(+/-9, 56.919538, 30.119184)` mm, which gives the exact bracket-root translation
`(0, -98.919538, 8.380816)` mm with identity assembly rotation on the right and
`(0, +98.919538, 8.380816)` mm with a 180-degree local Z rotation on the left.
The explicit left mirror is required by the generated handed EE frames and puts
the camera on the operator-confirmed opposite outer wrist face.

The supplied head bracket carries the same D405 body on `link_head_2`. All three
USD cameras use the D405's 84 by 58 degree depth FOV and 40 mm near clip. Head
optical forward/up are +robot-X/+robot-Z; wrist cameras look down the tool and
outward. Sensor-only rolls of 180 degrees right and 0 degrees left normalize
policy-image orientation without rotating the physical CAD or its mounting
holes. Authored metadata records `rby1_wrist_m3_pair` and the 18 mm bolt spacing
on both wrist assemblies.

A real transform bug was caught during rendered-camera validation: the CAD and
NumPy matrices use column vectors, but `Gf.Matrix4d` transforms row vectors.
Transposing at the Gf boundary corrected the initially inverted head view,
horizontal wrist views, and wrong blade direction. The authored-stage
regression now checks the actual Gf camera forward/up axes and actual blade
projection, not only the source NumPy matrices.

`interactive_greenhouse.py` now composes the robot on the **opposite side of the
target gutter** at `(6.99114, 3.78000, -0.3050817)` m with -90 degree yaw, where
the Z value is the measured cultivation-zone collision floor. This places the
robot in the inter-row aisle facing world -Y toward dynamic `Vine_0000`, with
about 229 mm initial chassis clearance to the target gutter and 35 mm to the
neighbouring gutter. `--no-robot` retains the accepted vine-only launcher. The
official 22-joint SDK ready vector is retained as `SDK_READY_POSE_DEGREES`; the
greenhouse launcher changes only the right arm to the exact v1.0-URDF IK vector
`[-101.724, -83.623, 34.196, -135.683, -57.431, 94.832, -74.920]` degrees. This
alternate branch keeps the elbow in the aisle, preserves joint-limit margin,
points the flat blade toward the real `SubStem_00` attachment, and keeps the
upward arc clear at spawn.

The first nominal IK attempt exposed an independent fixture bug rather than a
robot balance problem. Each dynamic vine owns a 1.2 m hidden `CatchPlane` cube
at bed height to catch severed foliage. It is synthetic episode bookkeeping,
not greenhouse structure, but it was colliding with the robot. A 10-step PhysX
trace measured 76.21 mm initial penetration and up to 1123.16 N*s impulse
between the tray and right-arm capsules, tipping the robot within 60 steps even
at the previously stable base stance. `add_ground_plane` now accepts filtered
actor paths, and each interactive catch tray filters the actual
`/World/RBY1/base` articulation root. Targeting the reference root alone was
insufficient; PhysX filtering requires the authored articulation-root prim.
This preserves tray contact for detached vine organs while removing the
invisible robot obstacle. `--contact-diagnostics` records collider pairs,
separation, and maximum impulse for future task-pose acceptance.

**Automated fit and stability acceptance (Isaac Sim 5.1):**

| Check | Result | Evidence |
|---|---|---|
| Hardware semantics | 3 D405 cameras; both wrist brackets aligned to the v1.1 18 mm M3 pair; original right EE/finger visuals and collisions inactive; 1 blade pair marked cutting; U arc pair non-cutting; 31 collision shapes | `robot_hardware_test.py`, `data/greenhouse_sim/robots/rby1a_v1.0.json` |
| Authored transforms and placement | exact bracket and mirrored reference bolt centres coincide on both wrists; right mount is identity and left mount uses the explicit 180-degree outside-face mirror; right tool faces world -Y with its U arc upward; opposite-aisle clearances are regression checked | 10 passed, 1 PhysX-only test skipped outside SimulationApp |
| Visual fit | brackets seat against the actual wrist screw faces; the right end remains knife-only; opposite-aisle robot, greenhouse, and vine render together | `data/greenhouse_sim/wrist_mount_reference_left.png`, `data/greenhouse_sim/wrist_mount_reference_right.png` |
| Opposite-aisle knife-only 480-step soak | 34/34 rigid bodies finite; base settled 11.524 mm toward the target row and 8.203 mm vertically; 2.078 degree tilt; succeeded=true | `data/greenhouse_sim/robot_wrist_screw_mount_acceptance.json` |
| Vine during robot soak | 121/121 tracked organs finite; 68.710 mm maximum compliant settling; 0 runaway organs | same report |
| Live camera path | all three camera paths remain present after right-gripper removal; explicit right-wrist D405 capture is non-black and sees the greenhouse/vines; head/wrist viewport selection remains available | `data/greenhouse_sim/right_wrist_d405_acceptance.png`; `data/greenhouse_sim/robots/rby1a_v1.0.json` |
| Measured lower-petiole reach | settled flat blade 118.072 mm and upward U-support 65.175 mm from the actual `Vine_0000/SubStem_00` attachment; blade extension Y=-0.999876 and arc normal Z=+0.999845; no spawn contact | same report |
| Contact trace | only normal chassis/wheel support against the cultivation-zone floor; no catch-tray, gutter, vine, arm, camera, or knife contact | same report |
| Error scan | no Isaac `[Error]`, Python traceback, selector `NoneType`, ill-formed `SdfPath`, PhysX error, invalid transform, broadphase fault, or explosion signature | `kit_20260808_104554.log` |

This clears the **asset import, hardware fit, optical-frame, collision-clear
task reach, and pre-contact stability** gate. It does not yet claim robot
deleafing success: the measured 65-118 mm air gap is intentional and still
requires a controlled final approach. Blade-to-petiole contact, robot-driven cut
triggering, left-tool interaction if used, and sustained-force
tearing remain the next explicit gate. The existing `C` command directly
releases the selected cut joint; it is not evidence of physical blade-triggered
severance.

A follow-up visual review requested that the U arc face upward so the flat plate
is presented cleanly for cutting. Rolling the knife 90 degrees about its
unchanged blade axis moved CAD +Z from tool +Y to tool +X. The rendered ready
pose now measures the arc-facing vector at Z=+0.999845 while preserving a blade
extension of Y=-0.999876 toward the target row. The same review requested direct camera access: the interaction
window and keys `1`-`4` now switch the active viewport between inspection,
head D405, left-wrist D405, and right-wrist D405 video respectively. Headless
captures remain available through `--capture-camera`.

The reported `KeyError: <class 'NoneType'>` in
`omni.kit.manipulator.selector` was isolated from physics. Isaac Sim 5.1's
mixed USD/Fabric selector can mark a selection handled by setting it to `None`,
then pass that value to the transform manipulator, which indexes it as an Sdf
path type and emits the subsequent ill-formed empty-`SdfPath` warning. The
greenhouse does not use the native transform gizmo: visible-mesh Shift-drag is
owned by `VisualPull`. GUI startup now clears stale USD selection and destroys
only the default transform selector's event subscription, while retaining
viewport camera navigation and the custom pull path. The report records
`native_transform_selector_disabled=true` when this targeted workaround is
active.

### Physical blade and bi-manual deleafing foundation (2026-08-08)

This work is isolated on `koh-dev/deleaf`. It implements the benchmark substrate
without claiming that an autonomous or teleoperated robot trajectory has already
completed the task.

The exploding/violent-contact behaviour was traced to geometry and filtering,
not hidden with extra damping:

- Each new flush petiole cut-zone capsule begins at an artistic mesh junction
  that can overlap the main stem or a sibling petiole at rest. These zones must
  contact the robot but must not make PhysX depenetrate one part of the plant
  from another. Every cut zone is therefore filtered from all other plant
  interaction bodies while robot/tool contact remains enabled. The interaction
  set now contains 43 sparse colliders: protected main-stem zones plus one cut
  zone and one grasp zone for each of the 18 real petioles.
- Two nearest visual vines receive 56 static semantic safety proxies. They do
  not add articulated plant load, but make neighbouring-vine collision both
  physical and measurable.
- The URDF import had duplicate live instance proxies in addition to the 17
  restored capsules. All 34 importer-created collision scopes, including empty
  link scopes, are now inactive before the exact 17 source capsules are restored.
- The active vendor `link_*_arm_5` capsule was a whole-tool planning envelope:
  250 mm cylinder length with 75 mm end radii, centred 100 mm below the wrist.
  The source URDF preserves a precise right-wrist endpoint capsule from
  z=+2 mm to z=-50 mm. Both symmetric wrists now use that 52 mm cylinder centred
  at z=-24 mm for contact simulation. The left palm proxy also stops at z=-25 mm
  instead of filling the finger channel down to z=-73 mm.

A closer numerical dual-arm IK spawn was tested and explicitly rejected. Contact
traces showed the right U support touching the protected main stem and
`SubStem_01`, while the broad old arm-5 and left-palm proxies overlapped the
target area. The launcher was restored to the previously accepted non-contact
right pre-contact pose and official SDK-ready left arm. Starting closer is not a
substitute for collision-aware approach motion.

The physical cut implementation now exposes all 18 `SubStem_XX` junctions as
world-space targets with centre, axis, local biological radius, and cut force.
The local radius is the first centreline sample at the junction, not the
segment-average contact radius; for `SubStem_00` this is 4.189 mm rather than the
incorrect 14.420 mm downstream-flare average. Required work is
`66.3 N * max(2r, 3 mm)`. The default acceptance limits are 10 mm/s minimum
forward speed, at most 35 degrees axial alignment for both edge and motion,
2 mm before/25 mm after the junction, 4 mm radial tolerance, 3 mm minimum
travel, and four physics steps of contact memory. Force is capped at 3x while
integrating work so a single solver impulse cannot fake a full cut.

Live contact monitoring distinguishes the straight leading edge from the plate
face and U support, aggregates all contact points once per 240 Hz step, and logs
main-stem, non-target-organ, neighbouring-vine, greenhouse-structure, and robot
self-contact as blocking safety violations. A physical cut is benchmark-valid
only when it is the selected target, the safety ledger is clear, and the
bi-manual state machine accepts it. `C` remains an explicit debug-only joint
release and is always invalid for benchmark scoring.

The retained left tongs now have contact reporting on both fingers and one
pre-authored, initially disabled fixed grasp joint per petiole. Three consecutive
steps with both fingers and at least 1 N establish the grasp; enabling a
pre-authored joint avoids runtime prim deletion and preserves the orphan after
severance. The task state machine enforces `seek_grasp -> grasped ->
orphan_retained -> transported -> released -> deposited`, including wrong-target
cuts, protected contact, cut-before-grasp, early release, and premature orphan
loss as explicit failure reasons. Deposit requires the orphan to be within the
aisle floor zone, below the floor tolerance, and moving no faster than 0.15 m/s.

**Current acceptance (Isaac Sim 5.1):**

| Check | Result | Evidence |
|---|---|---|
| Hermetic regressions | cut force/direction/work, disconnected taps, protected-contact failure, complete bi-manual ordering, cut-zone filtering, knife semantics, and robot placement covered; 48 passed, 1 PhysX-only test skipped outside a running app | `python -m pytest examples/greenhouse_sim/greenhouse_sim -q` |
| Robot collider rebuild | 34 imported collision scopes inactive; exactly 17 URDF capsules restored; both wrist contact capsules 52 mm; 31 authored robot colliders | `data/greenhouse_sim/robots/rby1a_v1.0.json` |
| 480-step integrated soak | succeeded=true; 121/121 vine bodies finite; 0 runaway organs; 68.693 mm maximum compliant settling; 34/34 robot bodies finite; 2.078 degree base tilt | `data/greenhouse_sim/deleaf_collision_fix_stability_480.json` |
| Rest safety | no robot-vine, knife-vine, gutter, camera, or arm contact; only chassis/wheel support contacts; safety_clear=true | same report |
| Safe right pre-contact | blade 117.945 mm and U support 65.033 mm from settled `SubStem_00`; arc still faces upward | same report |
| Error scan | no Python traceback, PhysX error, invalid joint, NaN, broadphase fault, or explosion signature | `kit_20260808_152754.log` |

This cleared the stable benchmark-physics and event-accounting milestone. The
robot-to-vine execution gate described here is completed in the 2026-08-09
validation below.

### Complete physical bi-manual execution (2026-08-09)

The staged RB-Y1 controller now completes the required task in the supplied
greenhouse using the original tomato mesh and the authored physics backend:

1. the left arm approaches the live moving petiole, closes both physical
   fingers, and requires three consecutive opposed-contact steps before a grasp
   can be established;
2. the left arm counterholds the exact selected organ while the right flat blade
   approaches from a collision-checked side and executes force-limited transverse
   slicing cycles;
3. the selected petiole severs only after the leading edge satisfies contact,
   cut-zone, direction, force, work, selected-target, and protected-contact
   gates;
4. the left arm retains the orphan, clears the vine row, carries it to a
   chassis-clear side-aisle zone, opens the gripper, and lets it settle on the
   greenhouse floor.

The earlier false starts identified real constraints rather than being hidden by
weaker thresholds:

- A planned `Link_3` grasp was not the contacted physical body. The manager now
  groups eligible cut/grasp colliders by body and anchors the fixed grasp joint
  to the actual opposed-finger contact on `Link_2`; contact gaps reset the
  consecutive-step gate.
- Both arm drives use the vendor URDF effort limits `(70, 70, 70, 40, 10, 10,
  8)` N m. Exact RB-Y1 v1.0 FK/IK and a point Jacobian reject a counterhold
  posture unless it can supply at least 1.10 times the 66.3 N cut requirement.
  A tested 72.1 N posture was therefore rejected rather than accepted by
  reducing the safety margin.
- Moving the entire unheld rigid petiole can no longer accumulate cut work.
  Reverse/unload motion also contributes none. The rigid-tissue fracture proxy
  can accumulate commanded penetration only while the real leading edge is in
  an admissible physical contact, the blade is moving in the required direction,
  force is above threshold, the exact target is counterheld by the left grasp,
  and the protected-contact ledger is clear.
- One large joint-space transport interpolation arced through foliage. The
  accepted route uses short Cartesian row-clearance waypoints, preserves the
  grasp pose whenever full-pose IK passes, falls back to bounded point/axis IK
  without changing its thresholds, translates sideways only after row
  clearance, and holds the terminal pose before release.
- Dropping directly over the original base stance let the long orphan rotate
  into a wheel. The accepted base is 150 mm farther back at
  `(6.99114, 3.93000, -0.3050817)` m. Its plan footprint extends beneath the
  neighbouring elevated supplied gutter, but the full 3-D contact trace has no
  gutter, robot, tool, or neighbouring-vine contact. Smaller trial offsets that
  reduced counterhold effort margin were explicitly rejected.

**Severance-model limitation, recorded rather than hidden:** PhysX cannot make a
live internal joint disappear inside the current reduced-coordinate petiole
articulation. Disabling `Joint_002` in USD left the subtree constrained in the
running solver. The severer therefore records the nearest geometric cut joint
and 21.003 mm geometric stub, but physically disables the organ's pre-authored
maximal-coordinate `BaseJoint`, yielding a realised physical stub of zero. This
is reported as `release_mode=maximal_coordinate_organ_base`. A future topology
backend must partition/rebuild the articulation at the selected internal joint
if non-zero residual stub physics is required. Likewise, the commanded
traction-separation penetration is an explicit approximation for a rigid
capsule that cannot deform or split; it does not replace the required physical
contact, force, direction, work, counterhold, or safety evidence.

**Definitive Isaac Sim 5.1 acceptance:**

| Check | Result | Evidence |
|---|---|---|
| Complete greenhouse suite | 63 passed, 1 PhysX-only test skipped outside a running app | `python -m pytest examples/greenhouse_sim -q` |
| 480-step integrated stability | succeeded=true; finite vine; 0 runaway organs; 68.700 mm maximum compliant motion; robot stable at 2.078 degrees tilt; pre-contact valid; empty stderr | `data/greenhouse_sim/stability_480_final.json` |
| Opposed physical grasp | actual `Link_002`; 26.603 N peak grasp force; exact selected target retained | `data/greenhouse_sim/bimanual_full_acceptance_final_pass.json` |
| Hardware effort margin | vendor joint limits active; counterhold posture admitted only above 72.93 N capacity | same report, `left_static_counterhold` |
| Physical cut | one intended cut; 73.875 N peak; 0.5910 J work vs 0.5554 J required; 8.697 mm forward travel; edge alignment 0.9825; transverse-motion alignment 0.9871; benchmark_valid=true | same report |
| Safe transport/deposit | 315.0 mm maximum clearance; endpoint hold; task phase `deposited`; floor contact=true; final speed 0.1490 m/s; zero unsafe contacts | same report |
| Whole acceptance | top-level succeeded=true; bimanual probe=true; robot stability=true; pre-contact=true; no Python stderr | same report and `.stderr.log` |

This is the known-target simulator-stability gate for the lab phase, not the
completion of the benchmark. The next section records the deterministic target
and simulator-only teleoperation layer added on top of it. Alternate-target
physical acceptance, lab leader-hardware validation, tear-force calibration,
task/scene randomisation, observation/action interfaces, VLA policy adapters,
metrics, and RL batching remain downstream and must not be claimed as complete.

### Deterministic targets and simulator-only teleoperation (2026-08-09)

Episode targets are no longer hard-coded in monitor, grasp, pre-contact, and
probe code. `--target-vine`, `--target-organ`, and `--episode-seed` resolve an
exact or seeded `auto` target from stable sorted physical cut-joint candidates;
the selected key and selection mode are persisted in every report. The visible
UI starts on the same target instead of silently resetting the managers to
`SubStem_00`. `run_bimanual_repeatability.py` launches one isolated Isaac
process per target/seed and counts an episode only when all strict checks pass:
top-level and probe success, exactly one intended benchmark cut, clear blade
safety, zero unsafe contacts, and terminal `deposited` state.

The simulator now consumes an atomic `greenhouse.teleop.v1` JSON mailbox. Each
arm has an independent deadman bit; commands must have a strictly increasing
sequence and a fresh host-monotonic timestamp, pass exact URDF joint limits,
and pass a configurable joint-speed limiter. Deadman release, stale watchdog,
or invalid command actively changes the drive target to the measured pose.
Physical contacts remain collision-enforced and benchmark-scored; default
`monitor` mode continues consuming commands without pausing, and opt-in
`rollback` drives toward the last contact-free target while contact remains
active. Right-arm commands are also mapped through exact RB-Y1
FK to a commanded knife-edge velocity, so teleoperation still cannot bypass
the physical contact/force/direction/work/counterhold cut gate.

`rby1_leader_to_sim.py` reads the vendored dual leader arms and trigger tools
and publishes only this simulator mailbox. It contains no RB-Y1 address,
command stream, power, servo, or gripper connection; the physical robot is not
commanded. Leader torque is limited to vendor-style gravity compensation,
joint-limit resistance, and damping, and a communication fault disables leader
torque. This path is code-complete but remains **lab-hardware unverified** until
the leader devices are connected deliberately.

When `--teleop-record-dir` is supplied, each run creates a unique episode with
metadata, synchronized JSONL steps, measured arm positions/velocities, both EE
world transforms, active target/task/cut/safety state, the gated action, and
selected head/wrist D405 RGB frames. Replicator warmup occurs while the timeline
is paused so observation setup cannot advance unobserved physics.

**Verification:**

| Check | Result | Evidence |
|---|---|---|
| Complete greenhouse suite | 124 passed, 1 PhysX-only test skipped outside a running app | `D:\isaac-sim\python.bat -m pytest -q examples/greenhouse_sim` |
| Final known-target physical acceptance | top-level/probe true; target `Vine_0000/SubStem_00`; one benchmark-valid cut at 71.444 N; task deposited; zero unsafe contacts; blade safety clear | `data/greenhouse_sim/bimanual_full_target_teleop_final.json` |
| Final integrated stability | succeeded=true; finite vine; 0 runaway organs; 68.732 mm maximum compliant motion; robot stable at 2.078 degrees tilt; selected-target pre-contact valid | `data/greenhouse_sim/stability_target_teleop_480_final.json` |
| Hardware-free teleop/recording | one fresh disabled command accepted; neither arm enabled; no unsafe latch; one synchronized JSONL step; head RGB 320x180, range 1-244, 6,736 unique colors | `data/greenhouse_sim/teleop_camera_warmup_validation.json` and its reported episode directory |

### Multi-target reach/collision blocker resolved, 2026-08-09

The recorded fixed-base `SubStem_01` failure was real: distal Link 3/2 were
outside left-arm reach, while falling back to proximal Link 1 caused a
non-target collision. The fix does not relax either gate. Robot authoring now
waits for the settled physical target and uses exact RB-Y1 IK to test
deterministic 0/30/60/90 mm aisle advances. Planning enforces a 20 mm reach
reserve, a distal segment floor, and wrist-D405 clearance; fixed positioning
remains available explicitly through `--robot-position-mode fixed`.

The bimanual sequence now also:

- keeps the planned distal segment after settling instead of silently falling
  back proximally;
- scores multiple left-wrist IK branches using the authored D405 volume;
- selects the knife wing with the greatest live non-target clearance;
- evaluates exact segment-to-oriented-box clearance for both the flat blade and
  non-cutting U-support over sampled RB-Y1 joint interpolation;
- selects the safer right-retract direction family and then maximizes lateral/
  vertical separation before aisle stow; and
- selects a swept, payload-clear left transport route while the PhysX contact
  ledger remains the strict zero-waiver acceptance authority.

The force/direction/work/counterhold cut gate remains physical. Low-force
leading-edge contact can establish entry-side geometry but cannot contribute
cut work. A counterheld full-diameter crossing may account for rigid U-guide
displacement only after physical leading-edge contact and still must meet
force, direction, work, travel, intended-target, and protected-contact
requirements.

**Verification:**

| Check | Result | Evidence |
|---|---|---|
| Complete greenhouse suite | 85 passed, 1 PhysX-only test skipped; new base-planner modules pass Ruff | `D:\isaac-sim\python.bat -m pytest examples\greenhouse_sim\greenhouse_sim -q` |
| Baseline `SubStem_00` full episode | top-level/probe true; nominal base; distal Link 3 plan; `positive_x_extra_wide_high` retract; one valid cut at 72.559 N and 0.55564 J vs 0.55540 J required; task deposited with 257.780 mm clearance; zero unsafe contacts; blade safety clear | `data/greenhouse_sim/bimanual_full_substem00_clearance_tiebreak.json` |
| Former blocker `SubStem_01` full episode | top-level/probe true; 30 mm base advance; distal Link 2 plan; positive-X blade wing; `negative_x_then_lift` retract; one valid cut at 74.462 N and 0.45248 J vs 0.43719 J required; task deposited with 264.425 mm clearance; zero unsafe contacts; blade safety clear | `data/greenhouse_sim/bimanual_full_substem01_direction_family.json` |
| 480-step integrated stability | succeeded=true; 121/121 vine bodies finite; 0 runaway organs; 68.710 mm maximum compliant motion; 34/34 robot bodies finite; 2.078 degree base tilt; selected-target pre-contact valid; contacts limited to floor support | `data/greenhouse_sim/stability_multitarget_blocker_fix_480.json` |

This closes the known fixed-base reach/collision blocker and verifies two
distinct target geometries. It does not yet claim benchmark-wide
repeatability: the next simulator gate is the isolated-process target/seed
matrix, followed by deliberate lab leader-arm validation. Robot benchmarking,
policy/VLA integration, and RL remain downstream of those stability gates.

### Full upper-body read-only teleop and torso-collapse fix, 2026-08-11

The connected RB-Y1 path now mirrors the complete fixed-base upper body rather
than only the arms. `rby1_robot_state_to_sim.py` uses the SDK model indices for
torso `[2..7]`, right arm `[8..14]`, left arm `[15..21]`, and head `[22,23]`.
It polls the robot-PC's read-only gripper cache at
`http://192.168.50.243:8765/status` at 10 Hz and maps motor ID 1 continuously
from the current session's calibrated closed/open stops into simulated left-jaw
aperture. The right physical gripper remains excluded because the simulated
right end effector is knife-only. The bridge still contains no power, servo,
control-stream, or physical-gripper command call.

Teleop startup no longer shows or sweeps through the scripted asymmetric
right-arm knife pre-contact pose. A fresh mailbox sample seeds torso, head, and
both arm drive targets plus initial PhysX joint state before physics
initialization; a stale/missing sample falls back to the symmetric Model A SDK
ready pose. D405 assemblies remain link-parented, so head and wrist observations
follow the mirrored joints automatically.

The observed disappearing-torso failure was a simulator hold-controller defect.
After an unsafe-contact/watchdog latch, every frame reset the drive target to
the latest measured position. That target followed the gravity-driven fall,
eliminating restoring position error and allowing the six-link torso to collapse
through the base. Holds now snapshot exactly one safe joint vector and continue
driving that fixed vector. Disabled joint groups likewise retain their last
commanded target rather than chasing current state. Torso drives use the v1.0
URDF effort limits `(270,270,270,120,120,120) N m` with bounded position gains;
head drives are separately bounded. The live cutting command velocity now comes
from the measured cutting-edge transform, so torso motion is included in cut
direction/speed evidence.

Continuous left-gripper aperture drives the real simulated finger joints. Closing
still requires opposed physical finger contact before the benchmark fixed grasp
is established; opening releases the retained branch. Existing force/direction/
work/counterhold cutting, orphan transport, floor deposit, neighbour-contact,
and safety-latch semantics are unchanged.

**Verification:**

| Check | Result | Evidence |
|---|---|---|
| Complete greenhouse regressions | 111 passed, 1 PhysX-only skip | `D:\isaac-sim\python.bat -m pytest -q examples/greenhouse_sim/greenhouse_sim` |
| Fresh measured startup | mailbox age 31 ms; no asymmetric pre-contact sweep | `data/greenhouse_sim/teleop_fullbody_live.json` |
| 12 s stationary full-body soak | left/right error <0.061 deg, torso <0.096 deg, head <0.109 deg; no unsafe contact; watchdog fresh | live report above |
| Forced state-source dropout | six-second watchdog hold; torso drift <0.096 deg, head <0.117 deg; no collapse or unsafe contact | live report above |
| Gripper channel | live calibrated source/request both approximately 0.999 open during soak | live report above |
| GUI responsiveness | fixed-pose run reached `running`; Kit `Responding=True` | PID health check and live report |

The visible simulator is running at the preflight-checked fixed base
`(10.639221515539253, 4.25, -0.15254085567917297)` with
`Vine_0002/SubStem_00`. Deliberate lab motion plus physical grasp/pull/cut
acceptance remains pending and must not be inferred from the stationary and
watchdog soaks.

### Closed-jaw grasp and counter-held thin-petiole cut reliability, 2026-08-11

Live physical-robot mirroring exposed two independent interaction defects. The
left grasp manager treated any measured openness below `0.95` as a close
request, so nearly open jaws could qualify. Its finger collision boxes also
extended 5 mm inward beyond the supplied `EE_FINGER.dae` mesh, allowing an
invisible capture before the rendered tongs enclosed the plant. Grasp
eligibility now begins only at openness `<=0.20`. The rebuilt robot USD uses
the exact mesh bounds `(-3,-16,-60.5)` to `(13,16,1.5)` mm in each finger
frame. A three-consecutive-step grasp may fill a missing thin-shape PhysX
callback only when the exact closest point on a petiole capsule or foliage OBB
lies in the visible closed-jaw channel. The fixed joint anchors at that actual
point, and every contact report records whether evidence came from `physx` or
`closing_jaw_geometry`.

The knife failure had a separate cause: the rigid 3.7 mm-radius petiole and the
thin semantic edge did not emit a non-zero PhysX callback in the live run, so
the existing strict cut gate never received a sample. The monitor now computes
the exact finite distance between the exposed leading-edge segment and each
collider segment of the active target. It synthesizes a compliant tissue
reaction only when the same target is already held by the verified left grasp,
the edge is within the tissue radius plus 1.5 mm, and no real edge callback is
available that step. Reaction force ramps with penetration and is capped by the
existing `3x` force cap. The existing target, direction, transverse alignment,
minimum speed, 66.3 N force, full-diameter work/crossing, consecutive-contact,
and protected-contact gates are unchanged. Reports identify this evidence as
`counterheld_rigid_tissue_geometry`; stationary, distant, open-jaw, unheld,
wrong-direction, and low-force states cannot cut.

**Verification:**

| Check | Result | Evidence |
|---|---|---|
| Focused grasp/cut policy regressions | 9 passed | `D:\isaac-sim\python.bat -m pytest -q examples\greenhouse_sim\greenhouse_sim\interactive_policy_test.py` |
| Complete greenhouse regressions | 120 passed, 1 PhysX-only skip | `D:\isaac-sim\python.bat -m pytest -q examples\greenhouse_sim\greenhouse_sim` |
| Rebuilt finger proxy audit | both proxies are 16 x 32 x 62 mm and centered at `(5,0,-29.5)` mm in their finger frames | direct generated-USD inspection of `data/greenhouse_sim/robots/rby1a_v1.0.usd` |
| 480-step fixed-station Isaac smoke | stage done; 34 robot bodies finite; base displacement under 0.001 mm; 129 vine bodies finite; 0 runaway organs; blade safety clear; contacts limited to floor support; no stationary grasp or cut evidence | `data/greenhouse_sim/grasp_cut_runtime_smoke_20260811.json` |

Automated implementation and stability acceptance are complete. Deliberate live
lab pinch, pull, and blade traversal in the relaunched `koh-dev/rby1` simulator
remains the final manual acceptance gate before demonstration recording is
enabled.

### Direct flat-blade traversal interaction cut, 2026-08-11

The live report showed that the knife was already making real contact, but the
contact was being routed to the wrong semantic gate. `BladeCollision` produced
repeated non-zero PhysX impulses against `foliage_grasp` proxies (including 34
events on one branch, with 2.80 mm maximum penetration), while
`BladeContactMonitor._target_colliders` accepted only `petiole_cut_zone`
paths. Consequently, visible blade-through-leaf contact was recorded as
diagnostics but could never reach `Severer`.

Live teleop now has a separate direct interaction path. Every severable
`foliage_grasp`, `petiole_grasp`, and `petiole_cut_zone` collider maps back to
its owning `Vine_XXXX/SubStem_XX` cut joint. A cut requires all of:

- the physical flat `BladeCollision`, never the non-cutting U-shaped arc;
- a non-zero PhysX impulse against one of those mapped proxies;
- commanded cutting-edge speed of at least 0.01 m/s, preventing idle jitter or
  a resting overlap from cutting; and
- two consecutive contact steps, rejecting a one-frame solver spike.

The protected main stem is never entered into this map. A successful traversal
releases the contacted proxy's associated pre-authored `SubStem_XX` junction
and is recorded under `blade_traversal_cuts`. Isaac 5.1 still cannot split the
render mesh at an arbitrary contact coordinate, so a blade hit on broad foliage
severs the whole associated branch at its authored junction; this approximation
is stated in every cut record.

This path is deliberately isolated from benchmark scoring:
`interaction_valid=true`, `benchmark_valid=false`, and
`benchmark_invalid_reason=direct_interaction_not_bimanual_benchmark`. The
strict physical gate remains unchanged, and **Run Full IK Sequence** disables
interaction cutting for its duration so only target, direction, force, work,
crossing, counterhold, ordering, and safety evidence can produce a benchmark
cut. If a strict and interaction decision could coincide in one frame, the
strict decision takes precedence.

**Verification:**

| Check | Result | Evidence |
|---|---|---|
| Direct traversal gate policy | stationary contact resets; first moving contact waits; second consecutive moving contact cuts; a gap resets | `interactive_policy_test.py` |
| Focused interaction/contact regressions | 10 passed | `D:\\isaac-sim\\python.bat -m pytest -q examples\\greenhouse_sim\\greenhouse_sim\\interactive_policy_test.py` |
| Complete deleaf simulator regressions | 121 passed, 1 PhysX-only skip | `D:\\isaac-sim\\python.bat -m pytest -q examples\\greenhouse_sim\\greenhouse_sim` |

The software gate is accepted on `koh-dev/deleaf`. The remaining acceptance is
deliberate live blade traversal after this commit is merged into
`koh-dev/rby1`; that manual confirmation must not be inferred from unit tests.

### Bounded RB-Y1 fixed-base UI preposition, 2026-08-11

The source greenhouse and vine placement remain unchanged. To let the operator
close the final reach gap without editing the scene or enabling unbounded mobile
base motion, the `koh-dev/rby1` interaction window now exposes **Robot forward
+10 mm** and **Robot back -10 mm**. The base is still world-fixed after each
action.

The controller reads the live articulation-root pose, computes robot-forward
from the authored yaw, and intersects two independent intervals: a per-session
offset of -50 to +100 mm and the already measured chassis-clear greenhouse
aisle bounds. It then pauses physics, updates
`/World/RBY1/joints/benchmark_world_fixed.physics:localPos0`, teleports the
initialized articulation root to the same pose through Isaac's supported tensor
API, zeros root linear/angular velocity, and resumes. An exception restores the
previous fixed-joint anchor and is written to `robot_base_ui_errors`; successful
moves are written to `robot_base_ui_nudges`.

Movement is refused while the grasp joint is active. After any successful
teleport, measured and commanded blade-velocity history is synchronized and
interaction cutting is suppressed for four simulation steps, preventing the
10 mm chassis adjustment from appearing as a blade traversal. The expanded
allowance intentionally permits controlled foliage contact beyond the former
+30 mm cap while the measured gutter/chassis aisle bounds remain authoritative.
At the current 4.28 m station the session cap is 4.38 m.

The same update refits both bent D405 brackets to the actual wrist force-sensor
screw plates. FreeCAD and STEP bolt-centre alignment gives the right local root
`(0, -0.098919538, 0.008380816)` m with identity rotation and the left root
`(0, +0.098919538, 0.008380816)` m with a 180-degree Z rotation. The explicit
left mirror places that camera on the opposite outer wrist face. Optical-only
rolls are 180 degrees right and 0 degrees left. The D405-to-bracket CAD transform
and the right knife transform are unchanged.

**Verification:**

| Check | Result | Evidence |
|---|---|---|
| Bounded nudge policy | +10 mm applies; +100 mm session cap applies; tighter aisle bound takes precedence | `interactive_policy_test.py` |
| Focused hardware/UI regressions | 23 passed | `robot_hardware_test.py`, `interactive_policy_test.py` |
| Complete RB-Y1 greenhouse regressions | 122 passed, 1 PhysX-only skip | `D:\isaac-sim\python.bat -m pytest -q examples\greenhouse_sim\greenhouse_sim` |
| Rendered wrist fit | both D405 bodies attach through the supplied bent brackets at the force-sensor screw plates; the left uses the explicit opposite-side mirror confirmed by the operator | `data/greenhouse_sim/wrist_mount_left_outer_20260811.png`, `data/greenhouse_sim/wrist_mount_wrist_plate_right_20260811.png` |
| 240-step integrated fixed-station soak | 129/129 vine organs and 34/34 robot rigid bodies finite; 0 runaway organs; base tilt 0.000099 degrees; contacts limited to wheel/floor support; initial tool safety clear | `data/greenhouse_sim/grasp_validation_left_outer_mount_smoke_20260811.json` |
| Updated visible station launch | `stage=running`; exact `Vine_0002/SubStem_02` target; 20 graspable target segments; fresh read-only teleop mailbox; contact policy `monitor`; no unsafe latch; recording off | `data/greenhouse_sim/physical_robot_teleop_grasp_left_outer_20260811.json` |

The visible station and read-only physical-robot bridge are relaunched for
operator confirmation. The report had accepted 384 fresh whole-body commands,
a fresh 0-16 ms watchdog, zero watchdog holds, and no rate limiting at the
verification snapshot. Manual opposed-finger grasp remains the open acceptance
observation; demonstration recording stays disabled until it succeeds.

### Fresh grasp acceptance and synchronous online RL baseline, 2026-08-12

The requested clean asset baseline is commit `b58c9f8` on `koh-dev/rby1`
(`feat(greenhouse): finalize wrist mounts and add optimized vines`). It includes
the complete `greenhouse/tomato_glb_30` collaborator asset set and the previously
accepted wrist/base work. The physical-robot state bridge, leader publisher,
and interactive simulator were stopped before the following checks; the RL mode
rejects `--teleop-command-file`, so physical-robot mirroring cannot silently
remain active during training.

A fresh `Vine_0002/SubStem_02` deterministic probe found a real grasp-planning
bug. The grasp pose still required EE local `+Z` to point toward fixed world
`+Y`, which was valid for the old -90 degree station but turns the wrist backward
at the current +90 degree station. Candidate search therefore reported a camera
clearance failure even when measured D405 clearance was 67-71 mm; the actual IK
orientation error was 119-160 degrees. Candidate selection, live left-arm IK,
multistart fallback, and base planning now all use `-robot_forward`. A regression
covers both +90 and -90 degree base yaw.

The rerun physically closed on the exact selected distal body
`/World/InteractiveVines/Vine_0002/Physics/Organ_0106/Link_003`, established
a 24.0 N opposed-jaw grasp, enabled
`/World/RBY1/ee_left/BenchmarkGrasps/Vine_0002_SubStem_02`, and placed the
anchor 1.105 mm from the visible jaw centre. It then retained that same body
through the 15 mm pre-tension pull and static counterhold. The episode later
stopped at `right IK failed at side -0.100 m, servo attempt 0`; that is a
right-arm route blocker after accepted grasp retention, not a grasp failure.

The first online-RL baseline now runs in the same Isaac process and consumes the
same `BimanualDeleafTask`, `LeftGraspManager`, `BladeContactMonitor`, `Severer`,
and protected-contact ledger as manual and deterministic execution:

- One normalized 15-value action commands bounded left-arm joint velocity (7),
  right-arm joint velocity (7), and left-gripper aperture velocity (1). At the
  default 20 Hz policy rate, every action advances exactly twelve 240 Hz physics
  samples. URDF joint limits and a configurable 35 degree/s arm cap remain
  authoritative.
- The stable 56-value state contains normalized arm positions/velocities,
  gripper openness, left-jaw-to-grasp and blade-to-cut vectors, target/tool axes,
  strict task phase, grasp/cut/transport progress, and protected-contact state.
- Potential differences shape approach motion. Event bonuses come only from the
  strict `grasped -> orphan_retained -> transported -> released -> deposited`
  state sequence. Unsafe contacts and task failure terminate with penalties.
  Direct two-frame interaction cutting is forcibly disabled in RL mode and
  cannot produce benchmark reward.
- Reset removes an active grasp, re-enables every severance joint, clears cut
  work, task events, and contact ledgers, restores every robot and per-organ vine
  articulation root/joint/velocity snapshot, opens the gripper, and settles
  outside episode time. The seed applies bounded +/-1 degree arm-start variation
  and a seeded phase of the accepted 1.0 m/s foliage airflow.
- A loopback-only synchronous JSON-lines server decouples Kit/PhysX from policy
  dependencies. `rl_client.py` provides direct and optional Gymnasium APIs;
  `train_online_rl.py` provides a reference tanh-Gaussian PPO trainer and
  checkpoint writer. Closing the trainer shuts down the server cleanly.

The first policy video's high-frequency arm motion had three independent
causes. The initial Gaussian standard deviation was `exp(-0.5)=0.607`, the
randomly initialized actor emitted non-zero means before any learning, and
every 50 ms action rebuilt its position-drive target from the already moving
measured joint state. Therefore zero action followed physical drift instead of
holding the previous command, while alternating exploration samples could
reverse a joint at the full 35 degree/s speed limit.

The runtime now integrates acceleration-limited velocity from a persistent
position target and resets that target with every episode. Zero action is a
real position hold. The policy starts at exactly zero mean with standard
deviation 0.223; arm and gripper acceleration are bounded; reward includes an
action-change cost; and near-target gripper closing has dense shaping without
awarding a grasp event. A 16-action live zero-command check at the validated
SubStem_02 approach held jaw distance within 17.8-23.1 mm with zero unsafe
contact, instead of drifting from 23.2 mm to 194.4 mm under the old controller.

`train_online_rl_parallel.py` batches one actor-critic across independent
loopback Isaac workers. Socket steps execute concurrently and generalized
advantage estimation remains separated along each worker trajectory. Four
workers used 19.6-21.3 GB on the RTX 5090 and completed 8,192 physical actions
in 888.0 s. This is process-parallel physics, not an in-stage GPU-vectorized
Isaac Lab environment. A validated collision-clear 100 mm left-grasp approach
was used as the first curriculum start; the neutral right arm was preserved.

The first non-smoke PPO launch also exposed why the right arm looked abnormal
at startup. `READY_POSE_DEGREES` still replaced the official SDK right-arm
ready vector with `GREENHOUSE_PRECONTACT_RIGHT_ARM_DEGREES`, a numerical IK
seed authored for the old -90 degree robot station. At the current +90 degree
station that seed folds the knife arm backward. Generic interactive and RL
startup now uses the symmetric Model A SDK ready pose on both arms; the legacy
pre-contact tuple remains labelled as historical route data, and task planners
compute their own approaches after reset.

**Verification:**

| Check | Result | Evidence |
|---|---|---|
| Fresh opposed grasp and retention | exact `Link_003`; 24.0 N; active fixed grasp; 1.105 mm jaw-centre distance; retained through 15 mm pre-tension and counterhold | `data/greenhouse_sim/grasp_validation_yaw_axis_fix_20260812.json` |
| Focused grasp/cut/reset/RL regressions | 36 passed | focused Isaac-Python pytest run |
| Complete greenhouse regressions | 134 passed, 1 expected PhysX-only skip | `D:\isaac-sim\python.bat -m pytest -q examples\greenhouse_sim\greenhouse_sim` |
| Randomized live reset/action/reset | stage `done`; three clean resets and one 12-substep action; finite 56-value observations; same-seed delta 0.003894; different seed changed state; zero unsafe contacts/cuts; empty stderr | `data/greenhouse_sim/online_rl_randomized_reset_smoke_20260812.json` |
| PPO process-to-physics smoke | four live actions, two time-limit episodes, three resets, one CPU gradient update, 1,025,669-byte checkpoint, clear blade safety, no server error | `data/greenhouse_sim/online_rl_ppo_smoke_20260812.json` and `data/greenhouse_sim/rl/ppo_smoke_20260812.pt` |
| Neutral startup regression | both arms exactly use the official Model A SDK vectors; focused startup/kinematics/RL suite 27 passed with one expected PhysX-only skip | `robot_scene_test.py`, `robot_kinematics_test.py`, and `rl_env_test.py` |
| Simple live PPO run | 512 physical actions; four 128-step rollout updates; three safe time limits near -1.38 return and one correct unsafe-contact termination at step 510; 56-state/15-action checkpoint | `data/greenhouse_sim/rl/ppo_simple_neutral_20260812.pt` and `data/greenhouse_sim/rl/ppo_simple_neutral_20260812_sim_report.json` |
| Deterministic checkpoint trial | 128 steps; return -1.346; zero unsafe contacts; stayed in `seek_grasp`; no grasp/cut/success | `data/greenhouse_sim/rl/ppo_simple_neutral_20260812_eval.json` |
| Zero-action physical hold | 16 actions; jaw distance 17.8-23.1 mm; zero command delta and zero unsafe contacts | `data/greenhouse_sim/rl/parallel_ppo_20260812/preflight_hold_sim_report.json` |
| Four-worker PPO run | 8,192 physical actions; eight 1,024-sample PPO updates; 32 episodes; 28 safe time limits, four protected-contact terminations, zero grasps/successes; 888.0 s | `data/greenhouse_sim/rl/parallel_ppo_20260812/ppo_8192_training.json` and `ppo_8192.pt` |
| Deterministic parallel-checkpoint trial | 64 safe actions; return -0.666; action-delta RMS 0.0437; stayed in `seek_grasp`; no grasp/cut/success | `data/greenhouse_sim/rl/parallel_ppo_20260812/ppo_8192_eval.json` |
| Stabilized rendered trial | four synchronized views, 64/64 frame sets, 1280 x 720 at 20 FPS | `data/greenhouse_sim/rl/parallel_ppo_20260812/post_stability_video/ppo_8192_stabilized_4view.mp4` |

This verifies the reusable online environment, stable command semantics,
physical safety termination, and shared-policy parallel PPO; it does **not**
claim a converged deleafing policy. The 8,192-step run never left `seek_grasp`,
and its deterministic policy held approximately 0.10 m from the target. More
blind steps with the same 15-dimensional direct-joint action space are not a
justified route to success. The next training stage must shorten grasp-only
episodes and add an expert/IK-seeded or task-space grasp curriculum before
unlocking right-arm cut and transport actions. D405 observations,
multi-target randomisation, and policy convergence remain open. The current
deterministic full-IK sequence
also still needs a new collision-clear right-arm route after the accepted grasp;
the RL interface does not hide or mark that scripted route blocker as success.

### Grasp-first curriculum, stable PPO, and rendered parity, 2026-08-12

This section supersedes the baseline's proposal to add an expert-seeded grasp
curriculum. The accepted `SubStem_02` route is now replayed through the same
15-value online-RL action API; it never teleports joints, creates a grasp, or
bypasses the physical contact/task gates. Five collision-clear IK waypoints
move from the 100 mm curriculum start to the exact distal `Link_003` grasp
pose. The gripper closes only within the 50 mm grasp neighbourhood. A trace is
saved for behavior cloning only if the strict task reaches `grasped` without an
unsafe contact.

The task now supports an optional `--rl-terminal-phase`. Grasp training uses
`grasped`, while full-task runs retain the original `deposited` success
definition. Curriculum completion is reported separately as
`objective_reached`; it never claims full deleafing success and a simultaneous
unsafe contact takes precedence. During `seek_grasp`, a deterministic action
mask freezes all seven right-arm dimensions and keeps the gripper open outside
50 mm. The environment enforces the mask, and PPO excludes inactive dimensions
from action log probability and entropy so masked exploration cannot corrupt
the policy ratio.

Eight episodes with +/-0.25 degree joint reset variation and independently
seeded airflow all established strict physical grasps. They produced 618
accepted state/action transitions. Behavior cloning reduced active-action MSE
from 0.080918 to 0.001191 and MAE from 0.16150 to 0.02212. Before PPO, the
deterministic cloned actor independently reached `grasped` in 4/4 unseen-seed
trials with no expert controller active.

The first PPO attempt used eight epochs at `3e-4`; approximate KL reached 0.131
and 58.7% of samples were clipped, so it was stopped rather than allowed to
erase the cloned policy. The trainer now supports a positive `--target-kl` and
stops an update before applying a candidate minibatch above that threshold.
The accepted run used four workers, 8,192 physical actions, four epochs,
`1e-4` learning rate, 0.001 entropy coefficient, and 0.02 target KL. It
completed in 1,074.97 s. Of 152 completed stochastic episodes, 133 reached a
safe grasp, 18 were safe time limits, and one stronger protected-contact event
was rejected. The last 24 completions contained 22 safe grasps. Full-task
`success` remains zero because cutting and deposit were intentionally locked.

A fresh deterministic evaluation used eight unseen reset seeds. All 8/8
reached the strict grasp objective with zero unsafe contacts in 39-42 actions
(`40.25` mean), at 1.23-20.34 mm closest jaw-target distance. Requested action
delta RMS averaged 0.0807 over all steps and 0.0750 after the initial command;
the physical controller additionally enforces 60 degree/s^2 acceleration.
This satisfies the gate for beginning a right-arm cut curriculum, but does not
yet claim a learned cut, transport, deposit, multi-target policy, or D405-image
policy.

The first rendered replay exposed a timing defect: calling
`SimulationContext.step(render=True)` advances one 60 Hz rendering interval, or four physics samples,
instead of one 240 Hz sample. It changed the trained trajectory and timed out
at 61 mm. RL now always calls `step(render=False)` exactly once per physics
sample and separately calls `context.render()` on the requested final substep.
The corrected four-camera replay reached `grasped` in 39 actions with zero
unsafe contacts and captured 39 synchronized inspection/head/left-wrist/
right-wrist frames. The verified MP4 is 1280 x 720 at 20 FPS.

Low-force contact with flexible leaf-area proxies also needed an explicit
benchmark distinction. The accepted deterministic probe itself brushes dense
foliage. Contacts on `FoliageContact_*` up to 0.01 N s (about 2.4 N average at
240 Hz) are now incidental canopy brushing; stronger foliage contact and every
rigid stem, petiole, neighbour, and greenhouse-structure contact remain unsafe.
The expert, policy, and safety tests cover both sides of this threshold.

**Verification:**

| Check | Result | Evidence |
|---|---|---|
| Strict deterministic expert replay | 74 actions; 3.83 mm minimum distance; `curriculum_grasped`; zero unsafe contacts | `data/greenhouse_sim/rl/grasp_curriculum_20260812/expert_tolerance125.json` |
| Randomized expert set | 8/8 accepted; 618 transitions; +/-0.25 degree reset variation; seeded airflow | `data/greenhouse_sim/rl/grasp_curriculum_20260812/grasp_expert_seeded8.json` and `.npz` |
| Behavior-cloned policy | MSE 0.080918 -> 0.001191; MAE 0.16150 -> 0.02212 | `data/greenhouse_sim/rl/grasp_curriculum_20260812/grasp_bc.json` and `grasp_bc.pt` |
| BC-only unseen-seed evaluation | 4/4 strict grasp objectives; zero unsafe contacts | `data/greenhouse_sim/rl/grasp_curriculum_20260812/grasp_bc_eval4.json` |
| Stabilized grasp PPO | 8,192 actions; 133 safe grasps, 18 timeouts, one rejected unsafe contact; 1,074.97 s | `data/greenhouse_sim/rl/grasp_curriculum_20260812/ppo8192_stable/grasp_ppo8192_stable.json` and `.pt` |
| Final deterministic evaluation | 8/8 unseen-seed strict grasps; zero unsafe contacts; 39-42 actions | `data/greenhouse_sim/rl/grasp_curriculum_20260812/deterministic_eval8/policy_eval8.json` |
| Corrected rendered replay | strict grasp in 39 actions; 39 synchronized frame sets; four views; 1280 x 720 at 20 FPS | `data/greenhouse_sim/rl/grasp_curriculum_20260812/policy_video4_fixed/grasp_policy_8192_4view.mp4` |
| Focused curriculum/policy/timing/safety regressions | passing | `rl_env_test.py`, `rl_policy_test.py`, `grasp_demo_test.py`, `isaac_rl_tick_test.py`, `contact_safety_test.py` |

**Next gated stage:** use the now-independent left grasp as the initial state
for `--rl-terminal-phase orphan_retained`, solve and validate a collision-clear
right-knife approach/sweep through the strict API, collect accepted cut
demonstrations, and only then fine-tune PPO for cut. Transport/release/deposit
and target randomisation remain later curriculum stages.

### Dense-vine geometry, contact capture, and opposed-grasp checkpoint, 2026-08-13

The deterministic full-task probe has been hardened for the optimized
`tomato_glb_30` vine without weakening its safety gates. Long foliage contact
shapes are decomposed into bounded local OBB proxies, and robot/vine screening
now uses exact capsule/OBB geometry for the arm, cameras, fingers, knife, and
greenhouse. Target-conditioned base planning searches bounded grasp-yaw and
transverse alternatives, verifies the complete joint-space route, and retries
only foliage-clearance rejections after measured live-vine settling. The
required foliage clearance remains 0.5 mm plus the measured sway envelope;
rigid/payload, greenhouse, and runtime inter-arm requirements remain 5, 10,
and 5 mm respectively.

The left final approach is now receding-horizon and drive-lag aware. It accepts
only IK updates that reduce measured jaw error and freezes the command at the
first current-step structural contact, preventing the old open-finger
push/chase behavior. Capture then performs a bounded lateral disengage,
tangential Y/Z centering, independent-finger backstop closure, and at most two
measured X seating corrections. The contacted structural body is locked for
the capture attempt; foliage-only or one-finger load cannot activate the grasp
joint. The existing three-consecutive-step, opposed-finger, minimum-1-N task
gate remains authoritative.

Four strict `Vine_0002/SubStem_02` full-probe iterations isolated the remaining
blocker. The first contact-aware run reduced target displacement from roughly
59 mm and 36.4 N in the old chase to 7.3 mm and 1.69 N at capture. Subsequent
backstop runs stayed finite with zero protected contacts and zero cuts. In the
latest run, the first bounded seat reduced the opposite-jaw gap from 18.785 to
12.992 mm; closure and a 3.927 mm reseat reduced the remaining gap from 2.927
to 1.346 mm. Only `left_finger_1` carried load, so the validator correctly
stopped before pretension or right-arm cutting. This checkpoint therefore does
not claim a full optimized-vine grasp/cut/deposit pass.

**Verification:**

| Check | Result | Evidence |
|---|---|---|
| Focused planner/IK/vine/grasp/RL regressions | 93 passed | `base_planner_test.py`, `robot_kinematics_test.py`, `vine_physics_test.py`, `interactive_policy_test.py`, `rl_env_test.py`, `rl_policy_test.py` |
| Backstop-first physical probe | stable base and route; no unsafe contact; no cut; opposed grasp rejected | `data/greenhouse_sim/rl/cut_curriculum_20260812/full_substem02_backstop_capture.json` |
| Bounded seat physical probe | gap improved 19.031 -> 13.114 mm; no unsafe contact; opposed grasp rejected | `data/greenhouse_sim/rl/cut_curriculum_20260812/full_substem02_backstop_seat.json` |
| Bounded reseat physical probe | final backstop gap 1.346 mm; 6.53 N one-finger load; no unsafe contact/cut | `data/greenhouse_sim/rl/cut_curriculum_20260812/full_substem02_backstop_reseat.json` |

**Next gated stage:** close the measured 1.346 mm residual with one bounded
near-contact micro-seat, require persistent physical opposition, and only then
execute pretension, right-arm force/direction/work-qualified cutting, orphan
retention, transport, and floor deposit. Cut-stage demonstrations and PPO stay
locked until that complete strict physical replay passes.

### Natural startup to IK task-stow transition, 2026-08-26

The visible simulator now starts both RB-Y1 arms in the symmetric Model A SDK
ready pose instead of exposing the asymmetric, task-specific knife stow. The
full-IK callback reads the measured PhysX joint state and transitions to the
existing task stows only after the user requests motion.

The initial direct transition was not safe: about 29% through the right-arm
chord, `right_arm_5` intersected
`Vine_0002/Organ_0097/FoliageContact_0175`; the former route reached only 0.2 mm
clearance against a 5.6 mm sway-conditioned requirement. A deterministic
bidirectional joint-space search discovered one intermediate waypoint per arm.
Both routes were then physically replayed, promoted to the instant interactive
shortlist, and remain screened at every command against the unchanged payload,
foliage, greenhouse, and 50 mm inter-arm planning gates. Headless RRT remains a
diagnostic fallback only and never blocks Kit's interactive UI thread.

The same run exposed a separate `right_approach` control-flow error:
`selected_pull` was referenced even when that mode intentionally skipped the
left-grasp branch. It is now initialized as shared probe state, so a completed
right-arm approach returns its physical result instead of a false Python
failure.

The complete replay then exposed a separate fixed-base left-ingress blocker
before jaw closure: the original aisle chord stopped at 0.5 mm from foliage.
A deterministic one-waypoint detour was discovered, physically accepted, and
promoted ahead of the existing aisle-clearance route. The live command guard
continues to enforce the unchanged 5.6 mm foliage planning floor and 50 mm
inter-arm clearance.

The simulator now exposes a dedicated `Run Grasp IK Sequence` action. It
executes the measured startup transition, left approach, live petiole
reacquisition, opposed-finger closure, and static counterhold validation, then
returns before right-arm cut planning. Its first isolated replay exposed a
missed `target_conditioned` mode initialization; after adding `grasp` to that
existing planning branch, the exact physical rerun passed in 500 seconds.

Full mode has separately established the same active physical grasp and reached
knife precontact, but all 229 committed cut-plane candidates remain ineligible.
That cut-plane search, not gripper IK, is the current end-to-end blocker.

**Verification:**

| Check | Result | Evidence |
|---|---|---|
| Natural visible startup | symmetric SDK-ready left/right arms; task stow deferred until IK request | `data/greenhouse_sim/startup_sdk_ready_fix303.json` |
| Deterministic startup transitions | one validated waypoint per arm; both start and target states valid; no interactive RRT stall | `data/greenhouse_sim/startup_to_stow_right_approach_fix308.json` |
| Physical right IK approach | `stage=done`; probe and top-level success true; final side `-0.015 m`; zero unsafe contacts | `data/greenhouse_sim/startup_to_stow_right_approach_fix308.json` |
| Deterministic left ingress | physically accepted one-waypoint detour; old 0.5 mm foliage blocker cleared without lowering safety gates | `data/greenhouse_sim/full_ik_left_ingress_fix310.json` |
| Physical grasp-only IK | top-level/probe success true; task phase `grasped`; active joint on `Vine_0002/SubStem_07`; final stage `left_static_counterhold`; zero unsafe contacts; blade safety clear | `data/greenhouse_sim/grasp_ik_fix312.json` |
| Focused interactive policy regressions | 182 passed | `interactive_policy_test.py` |
| Complete greenhouse regressions | 351 passed, 2 expected skips; one unrelated existing skeleton arc-length assertion differs by 0.167 mm and the skeleton module is untouched by this change | `python -m pytest examples\greenhouse_sim\greenhouse_sim -q` |

This validates natural startup, both startup-to-task transitions, the full
right-arm IK approach, and the opposed left grasp through static counterhold.
The complete force/direction/work-qualified cut, post-cut orphan retention,
transport, and floor deposit are still required before a full
deleafing-policy success claim. The dedicated grasp action exists specifically
so gripper IK can be tested without conflating it with the unresolved cut grid.


### Dense-vine cut recovery and report integrity checkpoint, 2026-09-01

The strict `Vine_0002/SubStem_07` full probe now reaches committed cut segment
30 while retaining the exact modeled 24 N left counterhold. The measured-state
right-arm recovery preserves the unchanged 5.0 mm hard blade-clearance floor,
5.5 mm recovery guard, 1.5 mm target-intersection floor, 66.3 N cut-force
requirement, and RB-Y1 hardware effort limits. Residual contact-loaded motion is
braked before Cartesian feedback recovery; no protected contact is accepted.

Two long probes exposed a report-only circular reference after an accepted
stationary recovery. The parent rigid-recovery record had been inserted back
into its child stationary-stage dictionary. Recovery completion is now emitted
as a distinct sibling stage, and a JSON-serialization regression prevents that
ownership cycle from returning.

The resulting `fix371` run serialized a complete failure report rather than
crashing. It retained the grasp, recorded zero cuts and zero unsafe contacts,
and kept blade safety clear. The brake remained physically within bounds
(minimum blade clearance 5.233 mm and minimum target intersection 3.058 mm),
but contact-loaded joint speed rebounded to 2.143 degrees/s and did not satisfy
the 1.5 degrees/s non-safety completion gate. The full dense-vine cut, orphan
retention, transport, and drop therefore remain open; this checkpoint does not
claim RL readiness.

**Verification:**

| Check | Result | Evidence |
|---|---|---|
| Report ownership/serialization regression | passed; no child-to-parent backlink | `interactive_policy_test.py` |
| Focused planner/IK/hardware/vine regressions | 281 passed, 1 expected skip | focused five-file test run |
| Complete greenhouse regressions | 375 passed, 2 expected skips | `python -m pytest examples\\greenhouse_sim\\greenhouse_sim -q` |
| Strict dense-vine full probe | safe failure at residual-motion brake; grasp retained; zero unsafe contacts | `data/greenhouse_sim/full_substem07_report_cycle_fix371.json` |

### Measured-entry torso brake checkpoint, 2026-09-03

The strict dense-vine full probe no longer fails at the residual-motion brake.
Stationary rigid recovery now holds the right arm and torso at their separately
measured entry states while braking, instead of moving the torso toward an
older latched kinematic target. The one-shot torso frame used by cut planning
is preserved after braking, and every brake sample now records measured torso
position and velocity. No force, clearance, overlap, or collision threshold
was relaxed.

The deterministic Vine_0002/SubStem_07 replay passed both stationary
recoveries that followed the segment-30 near-guard stops. The first accepted
after 12 brake steps at 0.787 deg/s, with 5.281 mm minimum blade clearance and
2.316 mm minimum target intersection. The second accepted after 9 brake steps
at 0.937 deg/s, with 5.399 mm minimum blade clearance and 2.441 mm minimum
target intersection. Both remain inside the unchanged 1.5 deg/s brake gate,
5.0 mm hard blade floor, and 1.5 mm target-intersection floor. The 24 N left
grasp remained active, blade safety stayed clear, and the run recorded zero
unsafe contacts.

| Check | Result | Evidence |
|---|---|---|
| Focused residual-brake regressions | 4 passed | interactive_policy_test.py |
| Complete interactive policy regressions | 206 passed | python -m pytest examples/greenhouse_sim/greenhouse_sim/interactive_policy_test.py -q |
| Strict dense-vine full probe | former brake blocker cleared; safe failure at the next segment-30 live-IK continuation gate | data/greenhouse_sim/full_substem07_measured_torso_brake_fix372.json |

The next blocker is numerical rather than a collision: after five accepted
segment-30 replans, solve_pose returned a bounded pose with 0.000062 mm
position error and 0.000281 degrees orientation error, but marked it failed
because SciPy exhausted its optimizer termination budget. That residual-valid
result must be admitted to the existing payload, inter-arm, joint-reserve, and
live-motion screens; it must not bypass those screens.

## Research findings, 2026-08-06 (pre-implementation)

### Stiffness — the current E is 5–15× too low

`TissueProperties.youngs_modulus_pa = 2.0e7` (20 MPa) is not defensible. Three
independent lines converge on **100–300 MPa** for fresh turgid vine tissue:
fresh greenhouse cucumber cane in tension, 280/199/137 MPa base→apex
(Xu et al. 2016, the nearest high-wire analogue); petioles measured at valid
span in four species, 110–192 MPa (Langer et al. 2021); and a self-weight
cantilever check on a tomato leaf back-solving to ~148 MPa.

Adopt **150 MPa** for petioles and young upper stem, **250 MPa** for the
lignified lower stem. The reviewer-proof sentence: at 5 MPa a tomato leaf would
deflect 2.9 m on a 0.12 m petiole, so any modulus below ~50 MPa is falsified by
the observable fact that tomato leaves hold themselves up.

The only direct tomato measurement (pedicel, 2.8–7.1 MPa, Weng et al. 2024) must
be cited but with its caveat: span/depth of 1.5–2.4 means it is shear-dominated
and is an *apparent* modulus; the Timoshenko correction reconciles it to
11–60 MPa. Its own data gives it away — modulus correlates with specimen
diameter at r = −0.893, which no real material does.

**No published Young's modulus exists for the fresh tomato main stem.** Say so
rather than imply otherwise. Other gaps: no tomato stem density (use 950 kg/m³
inferred from the measured 73–79% moisture), no tomato droop data.

Geometry defaults: stem 6.4 mm apex / 9.8 mm mid / 7.5 mm base (Gao et al. 2024,
2.1 m plants); petiole 4–8 mm (Sun et al. 2024). Detachment 32.5–40.3 N and
cutting 62–66 N, now corroborated across two independent groups.

### Trellis — clips every 0.30 m, compliant not fixed

Recommended: one fixed joint at the base elbow, then **D6 joints to world anchors
at 0.30 m spacing** (randomise 0.25–0.35 m per episode), 2–10 kN/m along the
string axis, ±5 mm lateral free play then ~250 N/m. Leave the stem joints
compliant — a real tomato stem self-buckles beyond 0.6–1.1 m, and the clips are
what carry it. Fixed joints over-stiffen laterally by 2–3 orders of magnitude;
a kinematic lower stem has infinite effective mass so the robot cannot perturb
it and contact forces become unbounded.

Consequence for metrics: with correct support the stem barely moves (1–3 cm
mid-span). **Essentially all real canopy disturbance comes from leaves and
trusses**, so the disturbance metric must be defined on leaf/truss motion, not
stem displacement, or it will stop discriminating between policies.

### Severance — manual snapping is standard commercial practice

The decisive finding. Growers routinely **snap** tomato leaves off by hand, and
that produces a *stub-free, faster-healing* wound than a knife (Decognet et al.
2010). The real acceptance criterion is "flush, no stub", not "cut". So the
RB-Y1's stock parallel gripper is sufficient and no custom tool is required —
the pull track is not a shortcut, it is the human baseline.

There is also **no COTS cobot leaf cutter to buy**; requiring one would mean
every replicating lab starts with a machining job and their blade geometry
becomes an uncontrolled cross-lab confound.

### The task as specified is shortcut-solvable

As authored the grower rule yields **the same answer on 20 of 20 vines**. Fixes,
in order of cost: the `N_retain = 15` term (free, agronomically real, alone
spreads the answer distribution); randomised pre-deleaf depth (uses the existing
joint-release primitive); ripeness rebinding. Together: 611 configurations.

Fruit ripeness **is** recoverable — the `FruitRipe_r_c` material's `c` index is
the ripeness stage, validated exact across all 155 fruits in the asset set. Bake
it as a `greenhouse:ripenessStage` attribute at conversion time.

Episode parameters must live **only in the instruction string**, with a
generation-time assertion that no observation→target mapping is a function.

### Occlusion is likely the binding constraint

Predicted to fail before reachability at the current 0.25 m spacing combined with
the assets' 0.75–1.11 m one-sided lean. Real high-wire in-row spacing is
0.40–0.50 m. Both spacing and lean are one-line changes in
`greenhouse_scene.py` and far cheaper to fix now than after thousands of demos.

### RB-Y1 asset path

**Confirmed 2026-08-06: the physical robot is RB-Y1 Model A v1.0.** Target
`models/rby1a/urdf/model_v1.0.urdf` (Apache-2.0, actively maintained). The official `rby1-sim-isaac` USD is **v1.2 kinematics** — 28.7 mm
end-effector offset and 6 cm head offset versus v1.0 — and ships with no licence
grant. v1.0 is identifiable on real hardware by a discrete FT-sensor puck between
wrist and gripper flange (wrist→EE 154.8 mm vs 126.1 mm).

Import settings that matter: `fix_base=False`, `merge_fixed_joints=False`,
self-collision off, inertia from URDF, position drives with wheels overridden to
velocity. The importer drops the URDF's non-standard `<capsule>` tags. A
comment-aware implementation audit found 17 active capsules to restore; the
earlier count of 26 incorrectly included nine capsules inside XML comments.

## Novelty position

Nearest neighbour is **OrchardBench** (GPU-parallel apple orchard, compliant
branches, fruit detachment). Differentiators, all of which must be delivered:
severance/cut quality rather than fruit pull-off, bimanual whole-body mobile
manipulation on RB-Y1, language-conditioned VLA evaluation, and real-robot
deployment (which OrchardBench explicitly disclaims).

## Commit log

One logical change per commit; no AI attribution trailers.

- 2026-08-06 — Stabilised vine physics with per-organ articulations, explicit
  contact-independent inertia, task-directed interaction colliders, cut/ground
  validation, and a clean 10-second stability soak.
- 2026-08-06 — Added visible-mesh mouse pulling, explicit foliage airflow,
  integrated greenhouse pull/cut probes, and manual viewport acceptance.
- 2026-08-06 — Imported exact RB-Y1 Model A v1.0, restored active collisions,
  fitted three supplied D405 assemblies and the right flat-plate knife, corrected
  optical/tool transforms, and passed the integrated robot-plus-vine soak.
- 2026-08-06 — Rolled the knife arc upward without changing blade extension,
  aligned the ready right tool with Vine_0000 from a collision-clear closer
  stance, and added live inspection/head/wrist camera switching.
- 2026-08-06 — Removed the synthetic foliage catcher's invisible collision
  with the RB-Y1 articulation, added contact tracing and measured petiole-range
  acceptance, staged a trough-clear right-arm pre-contact pose, and disabled
  the faulty unused Isaac transform selector in the interactive GUI.
- 2026-08-08 — Verified the benchmark sublayers the original greenhouse without
  scale changes and documented its authored 1.193 m floor-to-gutter-top height;
  moved RB-Y1 to the opposite aisle, fully removed the original right tongs from
  rendering/contact, mounted the knife directly to the retained EE flange, and
  passed a 480-step robot/vine/contact acceptance soak.
- 2026-08-08 — Replaced the approximate wrist-camera offsets with the exact
  RB-Y1 v1.1 FreeCAD screw-pair datum, corrected the extracted-STL origin error,
  regression-checked both mirrored mounts, verified a live right D405 frame,
  and repeated the 480-step robot/vine/contact acceptance soak.
- 2026-08-08 — Added leading-edge force/direction/work cuts, protected-contact
  accounting, 18 petiole cut/grasp targets, and the required left-grasp/right-cut/
  retain/transport/floor-deposit state machine; removed plant self-depenetration,
  duplicate imported robot colliders, oversized wrist planning envelopes, and
  the left-palm jaw obstruction; passed 48 regressions and a clean 480-step
  integrated robot/vine/contact soak on `koh-dev/deleaf`.
- 2026-08-09 — Completed the hardware-effort-limited RB-Y1 left-grasp/right-cut/
  retain/transport/floor-deposit execution, added exact v1.0 kinematics and
  force-capacity checks, made rigid-tissue fracture/topology approximations
  explicit in reports, and passed 63 regressions plus top-level physical
  acceptance with one valid cut and zero unsafe contacts on `koh-dev/deleaf`.
- 2026-08-09 — Added deterministic physical-target episodes and strict
  isolated-process repeatability aggregation; added a simulator-only dual
  leader-arm mailbox with deadman/watchdog/URDF/speed/contact gates and
  synchronized D405/action recording; preserved the complete known-target
  physical acceptance and recorded the fixed-base `SubStem_01` reach/collision
  failure instead of relaxing criteria on `koh-dev/deleaf`.
- 2026-08-09 — Resolved the fixed-base `SubStem_01` blocker with
  target-conditioned base placement, distal-segment and D405-clear left IK,
  target-specific knife-wing selection, exact full-tool swept-volume retract
  planning, and payload-clear orphan transport; passed 85 regressions, both
  `SubStem_00`/`SubStem_01` full physical episodes with zero unsafe contacts,
  and the 480-step integrated stability soak on `koh-dev/deleaf`.
- 2026-08-09 — Removed the implicit headless override from deterministic
  bimanual probes and render probe physics steps whenever `--headless` is
  omitted, allowing the same validated IK sequence to be inspected live while
  preserving explicit headless repeatability runs.
- 2026-08-09 — Removed a redundant pre-motion multistart grasp search from
  target-conditioned probes by reusing the exact distal/D405-clear collider
  accepted during base placement; live waypoint IK and all physical task gates
  remain active. Fixed-position regressions retain the full fallback search.
- 2026-08-09 — Corrected the interactive control flow so a non-headless
  `--bimanual-probe` executes the same validated probe instead of entering the
  idle interaction loop; visible runs emit live
  waypoint progress, and hold the final state open for inspection.
- 2026-08-09: Corrected the visible probe scheduler to advance exactly one
  240 Hz physics/control sample before monitoring and issue a render-only
  refresh every fourth sample. This avoids Isaac's four physics substeps per
  `step(render=True)` at 60 Hz. The visible `SubStem_01` full episode passed:
  one intended physical cut, 68 valid contact steps, 0.47107 J work, 76.819 N
  peak force, deposited orphan, zero unsafe contacts, and clear blade safety
  (`data/greenhouse_sim/visible_ik_end_to_end_substem01_fixed.json`). The
  complete greenhouse suite remains green at 85 passed and 1 PhysX-only skip.
- 2026-08-11 — Fixed live physical-grasp attribution by excluding zero-impulse
  contact-offset/lost records and automatically selecting a differently targeted
  branch only from a positive-load opposed-finger pinch; retained `T` as an
  explicit fallback and added regression coverage.
- 2026-08-11 — Moved the knife semantic edge from the U-support-occluded distal
  local `-Y` end to the exposed long local `-X` flat-plate side, propagated the
  axis convention through monitoring, IK, force-capacity checks, inspection,
  and the generated robot USD, and passed 126 tests with one PhysX-only skip.
  Manual lab pinch/cut acceptance and the dense-canopy autonomous route remain
  open gates; `tomato_glb_30` is recorded as a non-drop-in future migration.
- 2026-08-11 — Corrected inaccurate live grasps by requiring <=20% openness,
  matching finger proxies to visible CAD, and validating the exact closest
  plant point inside the closed jaw channel; added a counter-held, finite-edge
  rigid-tissue reaction for missing thin-petiole callbacks while preserving all
  direction/force/work/crossing/safety gates; passed 120 regressions plus a
  clean 480-step Isaac stability smoke on `koh-dev/deleaf`.
- 2026-08-11 — Routed real moving flat-blade contact on severable foliage and
  petiole interaction proxies to their associated pre-authored branch joints,
  guarded by 0.01 m/s commanded speed and two consecutive PhysX contact steps;
  kept the arc/main stem protected, labelled traversal cuts non-benchmark, and
  disabled the convenience path during strict full-IK runs; passed 121
  regressions with one PhysX-only skip on `koh-dev/deleaf`.
- 2026-08-11 — Added bounded ±10 mm fixed-base preposition buttons on
  `koh-dev/rby1`, capped session travel at +30/-50 mm inside measured aisle
  bounds, synchronized the world fixed anchor and articulation pose, blocked
  movement during grasp, and suppressed teleport-induced cut evidence; passed
  122 regressions with one PhysX-only skip and relaunched the visible station.
- 2026-08-11 — Expanded fixed-base forward preposition from +30 mm to +100 mm
  while retaining 10 mm steps, the -50 mm reverse cap, measured gutter/chassis
  bounds, grasp lockout, and cut-history suppression. Refit both D405 assemblies
  to the actual wrist force-sensor screw plates using the FreeCAD/STEP bolt
  datums, explicitly mirrored the left assembly to the operator-confirmed
  opposite outer face, normalized its optical view separately, rebuilt the robot
  USD, passed 23 focused and 122 full regressions, and completed a clean
  240-step robot/vine/contact soak.
- 2026-09-05 - Created `koh-dev/sim-data` from the current VLM branch, preserving
  its four uncommitted VLM files. Imported the supplied `tomato_greenhouse_pack`
  into an isolated, ignored data directory and added `run_sim_data.cmd` /
  `launch_sim_data.py` for an Isaac Sim 6.0.1 session-layer asset preview.
  The preview assembles two detailed plants plus 142 instanced plants on three
  existing gutters, provides three inspection cameras and the supplied lighting
  panel, converts plant-frame manifest translations to parent-local offsets,
  and excludes unbundled external prop payloads without editing source assets.
  This import does not yet integrate the RB-Y1, grasp/cut physics, or data export;
  setup and scope are documented in `examples/greenhouse_sim/SIM_DATA.md`.
  Verified the 833-part detailed-plant assembly (z=0.70-3.97 m), unchanged
  source USD layer, and a responsive visible GUI with a rendered aisle capture
  at `data/sim_data/preview_20260905/preview.png`. The successful launch uses
  asynchronous material loading and records `PACKAGE_READY` in `stdout_v3.log`.
- 2026-08-13 ? Added exact dense-vine collision screening, sway-conditioned
  route retries, measured receding-horizon contact capture, independent-finger
  backstop closure, and bounded live-geometry seating on `koh-dev/online-rl`.
  Passed 93 focused regressions and four strict physical probes with zero unsafe
  contacts; recorded the remaining 1.346 mm opposed-jaw residual without
  falsely advancing to cutting.
- 2026-09-06 - Started Phase 1 of `vlm_train_data.md` on `koh-dev/sim-data`:
  added the read-only manifest/optional USD auditor, per-substem review records,
  session-only anatomy overlays, reversible isolation, and fingerprinted human
  review records. The 24-plant audit found 870 review candidates, 971 excluded
  stubs, 492 degenerate capsule chains on old stubs, and 91 attachment-to-parent
  AABB warnings over a 2 mm diagnostic threshold. A headless single-plant review
  smoke passed with a viewport capture and unchanged source fingerprints;
  commands and evidence are in `examples/greenhouse_sim/sim_data/PHASE1.md`.
  Human anatomy review and horticultural cut/grasp rules remain pending; no
  cutting coordinates, physics approval, or training labels were fabricated.
- 2026-09-07 - Replaced the active single-target Phase 1 panel with a six-image
  batch gallery and deterministic, plant-balanced mixed/exception queues over all
  24 assets. Reads prior v1 decisions unchanged, skips completed current reviews,
  flags stale/conflicting records and ambiguous old workspace notes, and adds
  scoped preset reasons with save-and-advance. Batch approvals require explicit
  selected/captured cards, reviewer identity, current asset and image fingerprints,
  and atomic all-or-nothing publication. Explicit rereviews retain superseded
  records. At most one extra detailed plant and the existing viewport are used;
  source assets and original visibility opinions are restored. Default review
  runs are uniquely named; custom history directories are supported. All review
  overlays/captures remain excluded from model inputs; no cut/grasp or physical
  approval is inferred. Passed 42 focused regression tests under Isaac Python;
  the initial multi-plant six-card headless smoke passed at
  `data/sim_data/review_smokes/20260907T022304Z_98e3757d/`. Final UI-enabled smoke
  passed at `data/sim_data/review_smokes/20260907T023900Z_46a955da/`, with six
  rendered cards, visible gallery controls and all 24 source fingerprints
  unchanged. Existing user reviews were read without modification (26 records,
  25 unique targets; seven legacy scope clarifications). Usage and evidence are
  documented in `examples/greenhouse_sim/sim_data/PHASE1.md`.
- 2026-09-07 - Migrated fitted robot/FK/default camera workflows to the official
  RB-Y1 Model A v1.2 URDF (32 links, 31 joints). Preserved vendor CAD/meshes and
  v1.0 output; isolated v1.2 importer layers and added byte-identical safe aliases
  for four mesh names whose dots otherwise cause invalid SdfPath/null-prim errors.
  Build uses installed Isaac 5.1; rendered/runtime fit checks use Isaac 6.0.1.
  Head limits now come from v1.2, and the 28.7 mm tool-offset reduction is shared
  by FK/USD. Re-solved the counterhold seed while retaining all original gates.
  Exported the supplied HeadCam_Bracket_D405.FCStd saved solid without source
  recomputation; aligned its four-hole pattern to NECK_2 and the D405 rear pair
  to its actual bracket mating face. Both wrists use the exact supplied
  D405_Wrist_Bracket_v2-Body.stl. The 18 mm bracket pattern does not match the
  stock v1.2 wrist's 34 mm pair: added an explicitly simulator-designed adapter,
  included in rendering/collision envelopes, NOT approved lab hardware or a
  direct bolt-on claim. Right tool remains knife-only. All three RGB cameras
  default to exactly 848x408 with nominal square-pixel pinhole intrinsics;
  physical camera calibration/RGB-D realism remain unvalidated. Added the robot
  and three camera buttons to the new greenhouse package's session-only preview,
  starting with the head view and a static SDK-ready pose (no preview dynamics).
  Preserved anatomy-review data and diagnostic gallery; these are not training
  inputs. Passed 311 focused tests plus 10 subtests (one standalone-USD skip),
  seven-view Isaac 6 camera/fit render and the existing isolated physics fit
  inspector. Actual greenhouse head-view capture is 848x408. Full new-model
  bimanual/plant/RL acceptance and physical adapter review remain pending.
  Commands, scope and evidence: `examples/greenhouse_sim/ROBOT_V12.md`.
- 2026-09-07 - Corrected the new package preview's robot/floor placement. The
  old root Z=0 embedded the wheels in the visible greenhouse slab, whose local
  surface is Z=0.101 m (not the distant raised-strip maximum of 0.111 m).
  The launcher now measures floor-triangle heights at wheel/footprint samples,
  seats both rendered wheel bounds, checks chassis clearance and rejects
  missing or uneven support. The complete robot/cameras move together in the
  session layer; this is geometric alignment, not a physics/contact validation.
  Robot visual payloads are explicitly loaded even in a LoadNone host stage.
  Added floor/root/wheel-clearance provenance to launch status. Passed 52 floor
  and annotation regressions plus 10 subtests, including the actual supplied
  package and generated v1.2 robot; existing anatomy reviews are not modified.
- 2026-09-07 - At the user's request, annotation previews now default to the
  original right parallel gripper. Reactivated the retained stock body and two
  finger visual scopes and deactivated the complete knife subtree in the session
  only; both wrist D405 assemblies and the head camera remain. The generated
  asset/interactive physics workflow stay unchanged. Tool choice is explicit via
  `--right-tool gripper|knife_only` and recorded in launch status. Inspection
  found that the legacy knife geometry exists but projects back into the wrist;
  that mount still needs correction if knife use resumes (not a missing-file or
  solved-fit claim). Added real full-hand render views and tool-configuration
  regressions. 55 focused tests plus 10 subtests passed; nine-view Isaac RGB
  smoke passed at `data/sim_data/robot_v12_gripper_restore_20260907_1441/`, and
  the restored right-hand render was visually inspected. Static anatomy review,
  not physical gripper operation or grasp/cut validation, remains this mode's scope.
- 2026-09-07 - Added on-demand automatic reachability diagnostics to Phase 1
  anatomy review on koh-dev/sim-data. Reads measured v1.2 FK, current target/plant
  placement and gripper/camera envelopes; performs bounded multistart position IK
  independently for both arms with fixed base/torso/other arm. Separates a sound
  conservative outer-radius rejection from numerical failure, timeout and an IK
  solution. Partial endpoint overlap screening remains explicitly non-certifying.
  Worker execution keeps the UI updating; optional orange pose outlines never
  move the robot. Target/robot/screened-geometry changes invalidate results and
  remove outlines. Snapshot/source-fingerprinted diagnostic JSON is separate from
  human reviews; no anatomy, grasp, cut, training or physical approval is inferred.
  Knife-only right tools are not evaluated as grippers. Passed 335 regressions and
  10 subtests (one skip), plus real Isaac UI smoke at
  data/sim_data/reachability_smokes/20260907T055757Z_e076afe6/ with an intentionally
  relocated attainable fixture, rendered UI, cancellation/staleness checks and
  unchanged robot/source/review data. Full pose-constrained grasp, collision-safe
  paths and bimanual physical execution remain future work. Usage and limitations
  are documented in examples/greenhouse_sim/sim_data/PHASE1.md.
- 2026-09-07 - Corrected the mismatch between floating anatomy close-ups and
  fixed-parking-pose reachability. Robot-loaded annotation/gallery views now
  default to the actual mounted head D405 at 848x408 and retain that mode across
  target changes; floating 1280x720 close-up is an explicit diagnostic option.
  Added attachment projection/frustum status (NOT occlusion), current camera,
  intrinsics, robot/plant pose context in new review/capture records, and checks
  against context/resolution changes during capture. The complete robot and
  camera mounting transforms stay unchanged. Reach UI explicitly states that
  out-of-range applies to CURRENT base/torso only and does not reject the target
  globally. All existing human annotations remain unchanged. Passed 344 focused
  regressions plus 10 subtests (one skip). Base/torso stance selection and safe
  repositioning remain separate future work, as do clean VLM dataset captures.
  Real Isaac head-POV/six-card capture smoke passed at
  data/sim_data/reachability_smokes/20260907T061610Z_785a6e97/; the actual rendered
  UI was inspected. Existing live user sessions are not force-restarted by tests.
- 2026-09-07 - Added opt-in `--extra-cut-candidates 1|2|3` to the supplied-package
  static preview. Three candidates per foreground plant yields six attached
  leaf-bearing petiole subtrees at existing lower-main-stem stub nodes. Reuses
  complete supplied donor geometry with translation-only assembly; hides only
  empty recipient stubs in the session layer. Records new variant/component IDs,
  source hashes, complete subtree parent mapping and world attachment positions
  in candidate_branches.json; keeps original-plant split groups and inherits no
  human reviews. Original meshes/manifests and saved annotations are unchanged.
  Added a lower-branch diagnostic camera/capture alongside all mounted D405 views.
  Added targets remain unreviewed; physical cutting/grasping, collision/occlusion
  validation and cut/grasp labels are NOT implemented by this augmentation.
  Rejects annotation mode with variants until its metadata adapter exists.
  Also fixed restart rejection of saved camera-only USD overrides under an
  undefined /World/RBY1 root; preserves those overrides, still rejects a defined
  duplicate robot. Focused tests passed (82 tests plus 27 subtests), including
  real donor meshes, world transforms, attachment bounds/capsules, source hashes,
  duplicate protection, reversible session edits and saved camera overrides.
  Full selected robot/sim-data suite: 351 passed, one skipped, 27 subtests passed.
  Visible Isaac 6.0.1 preview reached PACKAGE_READY and its head/branch renders
  were inspected at data/sim_data/candidate_preview_20260907_160045_809b00/.
  Six branches comprise 43 added components; the greenhouse file hash stayed
  unchanged from launch, including pre-existing saved camera overrides.
- 2026-09-07 - Added assisted draft anatomy labels for the six candidate branch
  variants (`sim_data.label_drafts`). Re-audits source geometry and exact recipe
  fingerprints, reconstructs the derived parent graph, and exports B01-B06 IDs,
  petiole/parent/leaf membership, attachment and centreline geometry plus a
  Markdown review packet. Headless offline renders show normal appearance,
  isolated role colours and an alternate junction close-up (18 images total).
  Blue=petiole, green=leaf subtree, red=parent stem, yellow=attachment NOT cut.
  This is not a live/head-camera capture or an occlusion/reachability label;
  all human decisions remain pending and cut/grasp regions remain null. Source
  plants, variant sidecar and human reviews are not changed by generation.
  Fixed Hydra material restoration by using a removable diagnostic USD layer
  rather than reimporting the entire session, and added blank-render colour QA.
  First failed render packet was explicitly marked rejected and retained.
  Passed 67 focused regressions and 31 subtests. Final review packet:
  data/sim_data/label_drafts/20260907_223123_e879b8/review.md.
- 2026-09-07 - Implemented optional versioned prototype cut-region proposals:
  10 mm nominal, accepted 10-20 mm along the petiole centreline, measured from
  the manifest attachment (not the enlarged yellow marker or a spherical radius).
  Rule JSON records pending horticultural/physical validation. Arc-length
  interpolation preserves centreline bends and supports reversed endpoint order;
  ambiguous, degenerate, non-finite, too-short or misattached geometry is flagged
  with no guessed/extrapolated cut label. Draft v2 records proposed points,
  interval samples/radii/tangents and cut-plane normal separately from still-null
  approved cut/grasp labels. Rule/configuration fingerprints are rechecked before
  publication. Added optional --cut-rule to the existing offline review exporter,
  with white nominal-point and magenta interval overlays in six extra close-ups.
  Prior anatomy-only mode, source assets, saved variants and reviews are retained.
  Per-target review, main-stem/blade-stroke clearance, executable grasp/cut paths
  and horticultural validation are NOT inferred from these proposals. The next
  collection gate is user review of the six examples, followed by a synchronized
  robot-head RGB-D exporter/20-sample pilot (not implemented in this increment).
  Final selected robot/sim-data suite: 366 passed, one skipped, 47 subtests
  passed. Final headless Isaac review packet (24 images; all six cut-region
  views visually inspected):
  data/sim_data/cut_region_drafts/20260907_235517_89790c/review.md.
  Added a nominal-point distance diagnostic against parent capsule geometry:
  B02 (~0.95 mm) and B04 (~1.91 mm) have point gaps smaller than their petiole
  radii and are flagged for potential envelope overlap. This is only a proxy,
  not an exact mesh/full-interval/blade-stroke clearance test; unflagged targets
  are NOT thereby approved. The stored attachment is not necessarily the outer
  main-stem surface, so the rule does not guarantee a 10 mm external stub.
  Cut close-up cameras look partly from the distal petiole side to expose the
  interval; hidden/partially visible yellow/white markers remain subject to
  scene depth, not forced overlay visibility. No source geometry was moved.
  DRAFT_PACKET_READY reached; Kit emitted Fabric VtValue/array warnings after
  publication and shut down normally. Source manifests/variants and existing
  reviews remain unchanged; no physical grasp/cut execution occurred.

### 2026-09-08: Full-greenhouse mounted-head RGB-D pilot

- Implemented `sim_data/capture_contract.py`, `capture_scene.py` and
  `capture_pilot.py`, plus their regression tests. Captured nine actual
  848x408 mounted RB-Y1 v1.2 head-camera samples: three views each of
  provisional B03/B05/B06. B01/B02/B04 remain held for junction checks.
  Full greenhouse context is retained: two detailed plants, 142 backdrop
  instances and all six added branches, with the complete stock-gripper robot.
  No plant isolation, recolouring, source-geometry movement or label overlays
  in RGB inputs. Unbundled external props remain excluded as in the preview.
- Each sample stores clean RGB, raw float32 optical-axis depth in metres,
  a separate validity mask, actual camera calibration, robot/plant transforms,
  provisional world/camera/pixel cut labels and file hashes. Review overlays
  are separate. Nominal 10 mm and admissible 10-20 mm centreline intervals
  remain proposals, not human-approved cuts or demonstrated blade clearance.
- Loads the source through a disposable anonymous root layer; relative assets
  remain correctly resolved. Only the capture session freezes 75 gutter-base
  rigid bodies. Head framing uses actual mounted-camera extrinsics and
  URDF-limited head joints; robot snapshots change aisle offset without moving
  source plants. This does not certify arm reachability or collision-free poses.
- Synchronization is explicitly STATIC ONLY. Installed Isaac 6.0.1 rc.7 /
  Replicator 1.13.27 reports zero ReferenceTime and NoFrameNumber for this
  paused path and emits host-buffer WriterSyncGate illegal-cycle warnings.
  Dynamic native frame-ID synchronization is not verified. Static capture
  checks one copied writer payload, unchanged scene/pose/material-binding
  fingerprints, rendered calibration, fresh callback counters, distinct camera
  poses and changes in BOTH RGB/depth buffers. Counters are not engine frame IDs.
  GPU preflight verifies known plane depths at 2.0/2.4 m, on/off-axis optical Z,
  projection and both buffers' freshness. No installed Isaac extensions edited.
- Successful evidence:
  `data/sim_data/rgbd_pilots/pilot_20260908_005526/review.md`.
  Manifest: `pilot_ready_for_review`, `source_assets_unchanged=true`.
  Independent post-publication audit passed all nine sample records, 36 image/
  array file hashes, depth shape/dtype/validity masks, nominal-point projection
  and loaded source USD hashes. Earlier failed attempts are retained and marked
  `failed_do_not_train`; none of their outputs are promoted as accepted evidence.
- Visually reviewed all nine overlays. Eight targets are depth-consistent
  WITHOUT verified instance visibility; sample_0009 has foreground-occlusion
  evidence. No automatic visible/difficulty/approval labels were inferred.
  Native-resolution petiole diameters are only about 2.2-2.6 px and projected
  admissible intervals about 2-5 px, with strong backlighting. Improve physically
  appropriate viewpoints before scaling; capture success is not proof that
  these images support precise cut-point learning.
- Verification: capture-specific tests 41 passed; final selected robot/sim-data
  suite 407 passed, one skipped, 47 subtests passed. Headless pilot completed
  and shut down. No GUI relaunch, physical robot commands, grasp/cut execution,
  training, human-review mutations, commit or push in this increment.
- Next: review the full-scene pairs, refine robot viewpoints, add organ-instance
  masks and visibility checks, and validate cut eligibility before extending
  the pilot. Moving demonstrations, grasp/trajectory labels, D405 sensor-noise
  modelling and training approval remain unsupported by this exporter.

### 2026-09-08: Screened viewpoints, native organ visibility, and live preview

- Added opt-in `--refine-views` to `sim_data.capture_pilot`; the baseline capture
  command and interactive preview behaviour are preserved. New modules:
  `capture_viewpoints.py`, `capture_visibility.py`, `capture_search.py`, with
  viewpoint/visibility regression tests. No commit or push requested/performed.
- Search samples original base X and up to 0.3 m nearer, plus three aisle offsets,
  using only real head joints and whole-robot translations. Robot yaw, arm/torso
  pose, mounted-camera extrinsics, 848x408 optics, source plants and lighting stay
  unchanged. Floor-support and joint-limit checks remain enabled. This is a
  geometry-guided static dataset pilot, not blind evaluation or robot navigation.
- Added conservative whole-visible-robot/scene bound screening, including USD
  instance proxies. Integration revealed that merged backdrop plant AABBs cover
  large empty spaces and the actual archive contains mixed triangles/quads.
  Plant surface refinement now tests triangles against enclosing robot-local
  boxes; both quad diagonal choices are retained conservatively. A 10 mm margin
  is used; rigid greenhouse shapes keep conservative AABB checks. Possible
  overlaps are not exact collision findings. Self-collision, paths, dynamics,
  hidden collision shapes and closed-plant-volume containment are not certified.
- The writer now optionally exports native uncolorized instance IDs alongside
  RGB-D. GPU preflight passed known 2.0/2.4 m optical depth and object identity,
  then correctly identified/depth-tested a foreground occluder added and removed.
  Scene/camera/static-content checks and mask freshness remain fail-closed.
  Installed native frame IDs remain unavailable in this paused path; existing
  WriterSyncGate/Fabric warnings persist. No dynamic synchronization is claimed.
- Exact renderer prim paths map to 870 active source/variant components across
  the two detailed plants. Deepest component ownership separates child leaves
  from petioles. Backdrops retain native prim identities but do not receive
  invented organ labels. Raw instance IDs, component/organ masks, exact visible
  target masks and identity tables are written under `supervision/`, never into
  the model observation allowlist. Review-only green highlights preserve inputs.
- Nominal visibility requires BOTH petiole identity and compatible depth at the
  projected pixel. The accepted 10-20 mm interval reports unique-pixel sampled
  coverage, not amodal surface visibility. Known foreground occluders and same-
  instance depth conflicts are distinguished; unknowns are not auto-approved.
  Provisional image-quality gates and explicit rejection reasons are recorded.
- Completed evidence:
  `data/sim_data/rgbd_pilots/refined_20260908_135740/review.md`.
  36 candidate poses screened, 10 rejected for possible geometry proximity,
  26 rendered, nine selected. Manifest: `pilot_ready_for_review`, unchanged source
  assets. Three selected views pass the provisional clear-view gates: B05
  sample_0004/sample_0005 and B06 sample_0007. They have estimated widths
  3.08-3.61 px and projected intervals 7.38-8.97 px. This is not training approval.
  B03 remains limited: its best identity/depth-confirmed views are 2.55-2.72 px
  wide; moving nearer introduces foreshortening/depth conflict in this search.
  The six failed-gate selections remain explicit diagnostic examples.
- Verification: 435 selected robot/teleop/sim-data tests passed, one skipped,
  47 subtests passed. Independently audited all nine samples / 90 exported-file
  hashes, RGB-D shapes/validity, reprojection, and component ownership/masks using
  parent-path lookup. Checked all 872 loaded source USD hashes, the draft hash
  and all six capture-module hashes against the final manifest. Visually
  inspected all nine overlays and representative exact-mask highlights.
- The bounded headless run shut down after about 19.6 minutes while a separate
  GUI preview remained open. This is not a throughput benchmark; profile native
  annotation/warm-up overhead before scaled collection. Existing failed and
  blocked attempts remain retained and are not promoted to dataset evidence.
- At the user's additional request, launched interactive Isaac Sim 6.0.1 with
  RB-Y1 Model A v1.2, both stock grippers, three mounted D405 views and six added
  branches. Verified the GUI window and ongoing PACKAGE_HEARTBEAT output.
  Launch evidence: `data/sim_data/candidate_preview_20260908_135338/`.
  Use `Tomato package preview` -> `Robot head D405` / `Added lower branches`.
  This archive's annotation preview does not enable grasp/cut physics or teleop.
- Next: review passing and diagnostic full-scene views, improve B03's useful
  camera geometry, validate horticultural cut eligibility and visibility rules,
  then expand the pilot. No human approvals, difficulty labels, grasp/path
  labels, training or physical robot commands were added in this increment.

### 2026-09-08: Colour heatmaps of native Isaac depth

- Added `sim_data/depth_preview.py` and tests. The tool reads saved native
  Replicator `distance_to_image_plane` float32 arrays DIRECTLY. It does not
  estimate depth from RGB, reconstruct it from geometry, or invoke another model.
- Generated 18 review PNGs for the latest nine samples: common linear 0.04-2.0 m
  near-scene colours and full camera clipping-range colours, each with a metre
  legend. Yellow is near, purple far, grey checkerboard invalid. Colours saturate
  at the stated range limits; no raw depth clipping, interpolation, resizing,
  invalid-pixel filling or changes to RGB/depth/labels are performed.
  The pixel plane remains 848x408; the legend is a separate 118 px footer.
- New companion evidence:
  `data/sim_data/rgbd_pilots/refined_20260908_135740/depth_heatmaps/review.md`.
  Linked it from the existing image review page and documented the reusable CLI
  in PHASE1.md. Existing output directories are never overwritten; native input
  hashes, validity and calibration are checked before producing new previews,
  then source observation/metadata hashes are checked again afterwards.
- Verification: selected heatmap/capture/visibility tests 61 passed, including
  common-scale colour consistency, invalid values, unchanged native pixels,
  source tamper rejection and preservation of original observations/labels.
  Visually checked the B05 near-depth heatmap and legend. No simulator restart,
  live camera alteration, physical robot command, commit or push was performed.

### 2026-09-08: Robot-POV audit, explicit reviews and versioned prototype index

- Added `sim_data/dataset_review.py`, `dataset_package.py` and regression tests.
  The offline audit reconstructs the RB-Y1 A v1.2 fixed head-camera mount in an
  in-memory USD and verifies saved camera transforms from URDF FK/base/joints.
  It checks joint limits, rigid root, unchanged mount/848x408 optics, uncropped
  inputs, native renderer view/projection, source assets/recipes, geometry,
  masks, native depth validity and identity/depth visibility. It does not move
  the live GUI camera or issue any physical robot commands.
- The latest nine input frames all passed robot-POV verification: camera path
  `/World/RBY1/link_head_2/attachments/HeadCamera/D405/DepthCamera` and camera
  translation agreement below 1e-12 m (simulation arithmetic, not physical
  calibration accuracy). These are geometry-guided simulated whole-robot/head
  snapshots, NOT cinematic cameras, live lab poses or validated navigation.
- Inspected original RGB and magnified cut/mask/native-depth panels for 0004,
  0005 and 0007. B05 0004/0005 recommended for human prototype-label review.
  B06 0007 held for weak RGB separation near stem/fruit/leaves despite passing
  numerical visibility checks. Six other frames remain failed-gate diagnostic
  holds. No automatic easy/medium/hard labels or human approvals were added.
- Created `data/sim_data/dataset_reviews/review_20260908_v2/audit.json` and nine
  review-only cards (original full-scene RGB plus labelled 4x crops). Native
  RGB/depth/labels and original capture metadata were hash-verified unchanged.
  Recorded three explicit ASSISTANT decisions; local roles are self-declared,
  not authenticated. Older human anatomy review records remain untouched.
- Added append-only UUID review records, explicit supersession, stale/tampered
  source and card rejection, role-separated human prototype confirmation and
  conflict detection. Neither assistant recommendation nor human prototype
  confirmation enables training or physical execution in this unvalidated rule.
- Published `data/sim_data/datasets/robot_head_prototype_v1/review.md` and a
  versioned JSONL reference index with calibration, robot poses, provisional
  supervision, input allowlist, provenance and review states. Two recommended
  for human review, seven holds, zero training-eligible samples. The index is
  not self-contained; retain linked source/audit folders. With two source plant
  families, splits/difficulty remain unassigned and related variants stay grouped.
- Remaining: resolve B06 visual ambiguity/B03 camera geometry using actual robot
  views; explicitly confirm prototype labels, validate horticultural eligibility
  and the accepted annotation spec, then expand independent plant/view coverage.
  Static synchronization limitations, approximate parent-capsule diagnostics,
  unvalidated blade/grasp/path physics and backdrop organ-label gaps remain.
- Verification: 480 selected robot/teleop/sim-data tests passed, one skipped,
  47 subtests passed in 27.13 s, including 34 new review/package tests and a
  fresh audit of the real nine-sample run. No commit/push was requested or made.

### 2026-09-08: Lightweight browser GUI for human prototype-label review

- Added `sim_data/review_gui.py`, local HTML/CSS/JS under `review_gui_assets/`,
  `review_gui_test.py`, and the double-clickable `run_dataset_review.cmd` launcher.
  Default URL is `http://127.0.0.1:8877`. No Isaac application, GPU workload,
  external models, cloud service or physical robot connection is started.
- Starts with B05 0004/0005; offers all-sample and pending filters, original RGB,
  cut overlay, exact native target-mask and directly coloured native-depth tabs,
  full-size images and labelled evidence cards. Crops remain diagnostic-only.
- Human enters their name, explicitly acknowledges inspection, then clicks
  Confirm/Hold/Reject. Save is immediate with pending-sample auto-advance, resume,
  optional notes and append-only revisions through the existing review API.
  Failed-gate confirmation, stale browser edits, changed sources/cards and
  conflicting review history are rejected. Engineering holds stay independent;
  no training, horticultural or physical-cut approval is inferred.
- Loopback-only HTTP, exact Host/Origin checks, per-process anti-CSRF token,
  bounded JSON bodies, allowlisted cached evidence (no arbitrary file serving),
  restrictive browser CSP and no external assets. Existing matching server is
  reopened by the launcher; a second server cannot claim the same port.
- Verified the actual nine-sample page in isolated headless Edge: source image
  1152x838 review card, correct B05 0004 starting target, 0/2 recommended and 0/9
  reviewed, confirmation disabled until explicit input, no horizontal overflow
  at 1440x1050, all image tabs loaded, zero JavaScript runtime errors. Screenshot:
  `data/sim_data/review_gui/browser_review_20260908.png`.
- Browser save/revision/reload tests used ONLY a separate synthetic three-sample
  fixture. Confirm, Hold, Reject, auto-advance, failed-gate disabling and revision
  persistence passed. Real human-review count remained zero after those tests.
  Existing capture, audit, assistant records and dataset version remain unchanged.
- Final targeted GUI/review/native-depth regressions: 73 passed in 12.39 s.
  Opened the real review page via the existing-server launcher. The synthetic
  browser-test server and isolated headless browser were stopped; the real GUI
  remains running. No simulator restart, commit or push was performed.

### 2026-09-08: Human-reviewed v2 snapshot and focused robot-head recapture

- Verified the saved human decisions and all nine source captures again without
  modifying them: B05 samples 0004/0005 confirmed; B03 sample 0001 held. No
  conflicts or changed source hashes. Exported the NEW reference index
  `data/sim_data/datasets/robot_head_prototype_v2/`; two human-confirmed prototype
  labels, seven diagnostic holds, zero training-eligible samples. v1 is unchanged.
- Added CPU-only `sim_data.viewpoint_plan` to screen tighter base-XY/yaw samples
  before expensive native rendering. Same aisle, at most 40 cm approach, at least
  35 cm base-X to nominal-target-X separation, base yaw 150-210 degrees, unchanged
  arm/torso joints, fixed head-camera mount and 848x408 optics. Real head joints
  solve framing; no independent camera repositioning, plant edits or lighting
  changes. This is not a navigation, self-collision or physical reachability proof.
- Screened 165 finite poses across B03/B06. Six B03 and three B06 candidates
  passed projected-sampling and conservative geometry checks. Plan is saved at
  `data/sim_data/viewpoint_plans/focus_20260908_v1/plan.json`, with full rejections,
  source fingerprints and selected poses. Predicted widths are not visibility
  measurements; actual RGB/native-depth/native-ID evidence is still required.
- `capture_pilot --refine-views --view-plan PLAN` rechecks all selected poses
  against the loaded scene and uses the existing synchronized static capture.
  Default capture mode is preserved. Fixed the zero-margin AABB screen so
  touching/overlapping bounds are not incorrectly cleared when margin is zero.
- The first focused run failed before any sample because the new planner's
  import path loaded standalone USD before SimulationApp. Moved USD imports
  inside the CPU-planning function, added a fresh-process import regression,
  and retried into a NEW directory. `focused_20260908_v1` remains explicitly
  `failed_do_not_train`; no source data was altered by the failure.
- Focused retry `focused_20260908_v2` passed the real-GPU depth/known-occluder
  preflight, rendered nine planned candidates and saved six selected samples.
  Independent audit `data/sim_data/dataset_reviews/focused_20260908_v1/audit.json`
  passed source fingerprints, native head-camera FK/optics, projection, metric
  depth and exact renderer-derived component/organ masks for all six. Four pass
  numerical clear-view gates; numerical success is not visual/human acceptance.
- The browser reviewer now names its source capture run to distinguish sample
  numbers that repeat across capture sessions. Existing old-run human records
  are not copied to new camera views.
- Visually inspected all six full-scene RGB/cut/mask/native-depth cards. New B03
  0001/0002/0003 are recommended for human prototype-label review: approximately
  3.97/3.65/3.68 px petiole widths and 6.97/9.08/6.30 px projected cut intervals.
  All three remain one target geometry. B06 0004 stays held for RGB ambiguity
  despite numerical visibility; 0005/0006 are obscured by Added03_Leaf_204/205,
  with native depth 62.34/89.76 mm in front of the nominal centreline pixel.
  These are occlusion diagnostics, not validated hard-difficulty labels.
- Saved six assistant-only append-only review records and built a NEW index at
  `data/sim_data/datasets/robot_head_focused_v1/`: three recommendations, three
  diagnostic holds, zero human confirmations at export, zero training-eligible
  samples. Original B05 human confirmations and B03 hold are preserved in v2;
  neither old observations nor old reviews were rewritten.
- Launched the focused lightweight reviewer on `http://127.0.0.1:8878`; original
  review server on 8877 is untouched. Default Recommended shows three new B03
  views. The GUI labels the capture-run name; no human button clicks were made
  by the assistant. Documentation includes the explicit focused-review command.
- Final report inspection caught a hard-coded original nine-image count in the
  generic package summary. It now derives image/family counts from exported rows,
  with one- and three-sample fixture regressions. Exported corrected reporting to
  `data/sim_data/datasets/robot_head_focused_v2/`; focused v1 is retained unchanged.
  Six captured samples, review records and source hashes are identical between
  the focused indices. The review GUI continues to use the same audited capture.
- Process limitation: the six-sample manifest finalized as `pilot_ready_for_review`
  with unchanged sources and all saved artifacts subsequently passed audit, but
  Kit exited with code 1 during shutdown and did not print the post-close result.
  Shutdown cause is not yet established. Do not describe this as a clean-exit
  capture service; paused/static freshness guards and WriterSyncGate warnings
  remain separate from any future dynamic recorder validation.
- Verification: 525 selected robot/teleop/sim-data tests passed, one skipped,
  47 subtests passed in 32.54 s. Includes bounded planning, lazy USD imports,
  zero-margin overlap checks, capture-run naming and report-count regressions.
  Final hash checks passed for both current review indices and the six-sample
  camera audit. No commit or
  push was requested or made. Next: review the new B03 examples, retain B06 holds,
  then expand independent plant/target coverage before scaling or training.

### 2026-09-08: Native multi-plant collection scheduler and reviewed v3 snapshots

- Verified the three new B03 human confirmations and the updated original B03
  holds. Exported NEW `robot_head_prototype_v3` and `robot_head_focused_v3` review
  indices: five confirmed images across two distinct targets, with source/card
  hashes intact. Earlier snapshots and all append-only human records remain.
- Added `sim_data.collection_plan`, `collection_worker`, `collection_run` and
  `collection_test`. Scheduling reconstructs native cut proposals from original
  manifests; it does not depend on B03/B05/B06 or create/reposition branches.
  Target status, protected descendants, relevant source warnings, centreline
  validity and parent-proxy warnings are filtered with recorded reasons.
- Audited all 24 source families and reserved 16 train / 4 validation / 4 test
  target-family groups deterministically. Previously reviewed seed101/103 stay
  in train; the initial batch selects two NEW train families (seed11/seed17),
  two native petioles each, at most two rendered views per target. Family splits
  cover target ancestry only: greenhouse/backdrop context is shared and no
  scene-disjoint generalization claim is made. No difficulty/training approval.
- Plan: `data/sim_data/collection_plans/native_20260908_v1/plan.json`. It binds all
  source component manifests/meshes and greenhouse/backdrop USDs, the prototype
  cut rule, deterministic target selections, exclusions and split assignments.
  Changed sources, target coordinates, budgets or assignments fail validation.
- Worker loads the selected plant at the existing detailed greenhouse slot,
  leaving original gutter placement, neighbouring vegetation, light settings,
  native geometry, robot mounting and 848x408 optics unchanged. It searches actual
  bounded base/head poses and reuses native RGB/Z/instance writer, static freshness
  guards, visibility gates and independent FK/depth/mask audit. No teleop, free
  camera, physics/cutting or executable trajectory is introduced.
- Scheduler uses one isolated Kit subprocess at a time with raw binary logs,
  exact return codes, bounded timeout, and independent audit before the next job.
  Failures stop the batch, never become successful data or trigger overwrite/
  automatic retry. Manifest/result output is durable BEFORE Kit fast shutdown;
  the installed close implementation can terminate before post-close prints.
- First bounded attempt `collection_batches/native_20260908_v1` stopped with
  exit 1 before any saved images: CPU pose search left the root at its last
  candidate, correctly tripping the subsequent original-base consistency check.
  Added transactional root/link restoration on success AND failure; exact-matrix
  regressions pass. The failed job and logs remain; job 2 never started.
- Retrying the unchanged two-plant schedule in NEW
  `data/sim_data/collection_batches/native_20260908_v2/`. Final rendered results,
  process exit status and explicit visual inspection will be appended below.

#### 2026-09-09 KST: native batch completed and human reviews received

- Retry v2 completed both isolated jobs with exit code 0: seed11 in 299.44 s,
  seed17 in 308.83 s (worker time, excluding offline audit). All eight saved
  views passed independent RGB/Z/mask/projection/FK/source integrity checks.
  Seven pass numerical clear-view gates; this does not imply seven clear images.
- Inspected all eight RGB/cut/mask/native-depth cards. Recommended seed11
  0001/0002 and seed17 0001/0002/0003. Held seed11 0003 (petiole/parent blend),
  seed11 0004 (occluded nominal cut) and seed17 0004 (ambiguous/obscured junction).
- Independently reconstructed all 827/824 native component paths, organ types
  and source-family mappings from source manifest parent chains; each saved
  catalogue matched exactly. Each job audit binds 10,282 source/artifact files.
- Saved eight assistant-only records and immutable native_seed11_review_v1 /
  native_seed17_review_v1 indices. The user then explicitly confirmed all five
  recommended views. New native per-plant v2 indices preserve those confirmations;
  earlier assistant-only snapshots and original/focused v3 snapshots remain.
  Confirmations are prototype image-label agreement, not training/cut approval.
- Added collection_review: one lightweight page for both jobs on port 8879,
  job-prefixed UI sample IDs, original per-job audit/records save routing, local
  same-origin/CSRF protections and stale-review checks inherited from ReviewApp.
  All 40 image tabs loaded at expected dimensions. Old 8877/8878 pages untouched.
- Evidence: data/sim_data/collection_batches/native_20260908_v2/visual_review.md.
  Both supervised native jobs demonstrated real zero exits. Earlier nonzero
  shell-reported shutdown was not retroactively reclassified; its cause remains
  unproven. Static synchronization limits and WriterSyncGate warnings remain.
- Verification before the batch-review adapter: 556 passed, one skipped,
  47 subtests in 35.97 s. Adapter plus existing GUI tests: 32 passed in 7.29 s.
  No real human review was submitted by the assistant. No commit/push performed.

### 2026-09-09: Target-conditioned VLM dataset release engineering (in progress)

- The user requested persistence to a complete first VLM training dataset.
  Scope is RGB cut-point/visibility/abstention grounding; bimanual trajectories,
  dynamic synchronization, physical execution, agronomic validation and Cosmos
  real-hard remain separate. No lab-robot command, commit or push was made.
- Added training_plan/contract/export/evaluate modules and TRAINING_DATASET.md.
  Clean mounted 848x408 RGB plus an explicit visible distal petiole query pixel
  disambiguates the target. The query is at least 45 mm along the source petiole
  and 18 pixels from the nominal cut, not the answer coordinate. Output is a
  cut pixel or an explicit occlusion abstention. Native camera-Z, calibration,
  centreline camera/world XYZ and evaluation intervals remain sidecars.
- Preserved seed-0 target-family reservations (16 train / 4 validation / 4 test).
  New collection uses continuously varied framing; fixed-position pilot frames
  are not silently promoted. The shared greenhouse/backdrop is NOT scene-disjoint.
- Conservative label derivation on the previous native batch agrees with its
  five confirmed clear examples and excludes all three held examples. Numerical
  visibility alone is insufficient: visible parent separation and proximal
  context are also required. Unknown occlusion is excluded, not a false negative.
- Added explicit job selection, streaming saved captures and streaming audit
  cards. Zero worker exit plus independent audit is required for export. Portable
  JSONL chats, clean RGB, native depth/validity, per-sample labels, hashes, exclusion
  ledger and offline loader are implemented. Source prototype approvals remain
  false; new labels explicitly describe automatic synthetic-task supervision.
- Release gates require 10,000 train / 500 validation / 500 test unique images,
  family/target/difficulty coverage, varied cut locations and portable-loader
  consistency. --allow-incomplete is a diagnostic path and never grants release
  status. Constant-pixel, query-copy, train-fitted query-offset and always-abstain
  baselines are available; they are NOT a trained VLM evaluation.
- First randomized smoke: collection_batches/grounding_smoke_20260909_v1,
  job_003 seed11, seven captured/audited frames, worker exit 0, 409.28 s worker
  time. Four accepted automatic hard/abstention labels; three undersampled
  positives excluded. Portable grounding_engineering_smoke_20260909_v1 passes
  its loader but explicitly FAILS coverage and is not a complete training release.
  Inspected its first RGB/mask/native-depth card: the nominal cut is leaf-occluded.
- Dataset-scale throughput remains under investigation. StaticSceneMonitor
  keeps an initial full static scan and USD edit notices, allowing only explicit
  between-frame robot snapshots. First optimized attempt v2 failed closed on
  Replicator /Orchestrator bookkeeping before any saved frames; failed logs remain.
  Added that observed renderer-graph exception and guard regressions. Fresh v3
  comparison is running; do not claim a speedup or successful optimized capture yet.
- Tests so far: 101 capture/contract/source/audit regressions passed; 74 selected
  export/contract/audit tests passed; 51 guard/contract/export tests passed;
  10 real-URDF posture-diversity/evaluation tests passed. These overlapping suites
  are not added together. Full combined suite and dataset-scale QA still pending.

#### Capture qualification and coverage follow-up (2026-09-09)

- Grounding smoke v3 completed with seven independently audited frames and a
  real zero worker exit in 313.94 seconds. The initial full scan plus USD notice
  guard worked; original native depth matched the same-pose long-render baseline.
- Short-render experiments v4-v8 failed closed with zero saved training frames.
  Native geometry/instance identity agreed, but RGB convergence or comparison
  against the long render failed. v8 global mean absolute RGB difference was
  3.3056/255 against the declared 3.0 limit. These are failed qualifications,
  not successful speedups. All logs remain available.
- Runtime readback identifies RealTimePathTracing, DLSS AA, exposure adaptation
  off. The source scene overrides a requested legacy RaytracedLighting mode.
  Both legacy-mode diagnostics failed before images; the temporary RT2 switch
  was restored and preference persistence disabled. No installed SDK or source
  scene was edited to force a renderer.
- Production now defaults explicitly to the established seven eight-subframe
  captures (six warmups plus final callback). Short/reference comparison is
  opt-in only. Added bounded-loop, final-callback and failed-reference tests.
- Started grounding_coverage_20260909_v1/job_003 with real torso/head/base
  snapshot diversity. First nine complete image files provisionally include one
  easy, two medium and four hard examples; two are excluded. This is a LIVE
  yield observation, not a completed audit/release. Original mounted RGB for
  sample_0003 was inspected; its label is medium with proximal visibility 0.92.
- Added training_progress: reports saved files, provisional label/exclusion
  counts and independently audited counts separately. It cannot grant release
  approval. Added opt-in first-render CPU profiling. One small profile worker
  temporarily runs alongside the coverage worker (two total isolated workers)
  after low GPU utilization and memory headroom were observed; each supervisor
  still owns at most one child. No live user simulator or robot is controlled.
- Added class-balance release gates (5,000 localized/1,000 occluded train;
  250/50 each validation and test), exact camera/scene-condition deduplication,
  and explicit reviewer hold/reject vetoes. Noise cannot inflate image counts.
- Full selected suite passed 665 tests, one skip and 47 subtests before these
  last renderer/progress additions. Latest focused suite: 42 passed. The current
  four-row engineering export remains explicitly incomplete; no training
  release or VLM fine-tuning result is being claimed.

#### First completed coverage job and native annotator qualification

- grounding_coverage_20260909_v1/job_003 completed with 63 saved frames, a real
  zero worker exit and independent audit hash
  85e08dad174ceb1b3fe3235fcbb39a19bc2e2ca462fce79c70cb41d3c1633ae7.
  Worker elapsed time was 2209.78 s. The task contract accepts 42 labels:
  29 easy, 2 medium and 11 hard. It excludes 21 (3 missing visible distal query,
  13 ambiguous junctions, 5 unknown visibility). These counts are not a full
  multi-family release. Collection of the other 23 families has started in
  grounding_coverage_remaining_20260909_v1 with the established render budget.
- Added training_qa to produce hash-bound native RGB/mask/depth cards without
  altering observation files or inventing approvals. Inspected samples 0003,
  0009, 0015, 0023; observations are in
  data/sim_data/dataset_reviews/grounding_live_20260909_v1/visual_review.md.
  No obvious wrong-organ localization in those four; the scene is strongly
  backlit and this small review is not a statistical accuracy certificate.
- CPU profiling found repeated legacy instance-label serialization expensive.
  Added native_instances plus --instance-backend fast/compare. Comparison mode
  checks the same callback's complete uint32 image and all observed prim paths;
  writer errors are fatal, never replaced by an earlier successful callback.
  The two-frame full-greenhouse comparison and GPU calibration/occluder preflight
  passed; both fast-only and comparison jobs exited zero and passed audit.
  Each comparison frame checked all 345,984 pixels (228/592 observed IDs).
- The fast path removes the legacy JSON conversion graph without changing RGB,
  native depth, full-image resolution or scene geometry. It has NOT yet shown a
  clear total wall-clock speedup under concurrent collection. CPU profile time
  is not substituted for total capture time. No installed NVIDIA files changed.
- Consolidating the requested 56 subframes into one step failed the current
  strict RGB-difference comparison, with native geometry/identity unchanged.
  The failed run saved zero training frames. Added a repeated-reference noise
  characterization mode: experimental candidate/first-reference RGB remain
  review-only, and the final established-budget reference is the observation.
  This measures whether the comparison itself is below the renderer noise floor;
  it does not automatically approve a faster renderer.
- Added deterministic --view-offset shards. Disjoint windows preserve the same
  global base/head/torso proposal sequence; later batches cannot repeat prior
  windows just to fill a quota. Existing seed-0 plans still load unchanged.
- Recent selected suites: 65 native-adapter/capture tests; 82 shard/posture/
  planning tests; 425 sim_data + vlm_eval_test + robot-verifier tests with 47
  subtests. These overlap and are not added. A complete 10k/500/500 training
  release is still pending collection, stratified QA and final export.

#### Task-qualified faster capture and non-overlapping scale collection

- The repeated-reference noise diagnostic failed at its second pose; its one
  saved reference observation is excluded because the worker exited nonzero.
  Original logs remain. Native identity comparison now resolves prim identity
  per pixel instead of treating renderer-local ID renumbering or unobserved
  mapping entries as geometry changes; the earlier failure cause is not asserted.
- Added an explicitly different capture profile, warm56_then8: initial full
  56-subframe warmup followed by fresh eight-subframe static snapshots. This is
  task-level qualification, NOT a claim that failed strict RGB comparisons passed.
  The long-budget default and all native capture/label integrity gates remain.
- First-family qualification: nine matched poses, identical native target masks
  and Z depth, eight identical accepted answers (5 easy / 1 medium / 2 hard),
  one no-query exclusion in both. Measured after-first-frame median fell from
  32.202 s to 4.870 s. Actually inspected six independent cards and the hard
  sample 0004 overlay; notes are in grounding_short_profile_20260909_v1.
- Second-family qualification: 12 matched poses, identical native masks and Z,
  10 identical easy answers and two ambiguous-junction exclusions in both.
  Measured median fell from 33.990 s to 4.788 s. Actually inspected three cards;
  notes are in grounding_short_seed101_20260909_v1. Strong backlighting remains.
- The established-budget seed101 job completed with 54 frames, zero exit and
  independent audit f3fb9531dd470b0d30895c8d089762bb2f0b714792a103057af32159d8072f3a.
  Preserved it and the 63-frame seed11 job. To switch the remaining queue,
  terminated only its verified newly started job_002 child with zero saved
  samples. Its nonzero cancellation ledger and operator note remain unchanged.
- Two new isolated fast coverage queues A/B cover the remaining 22 families.
  First four finished jobs (41, 53, 39 and 20 frames) all exited zero and passed
  independent audit. Counts of raw frames are not counts of training labels.
- Fast native mapping now stores all observed renderer IDs only, retaining
  every pixel and observed prim path and the full anatomical component catalogue.
  The compact-mapping 53-frame job passed independent audit; sample 0002 is
  6.39 MB instead of approximately 11.8 MB with the unused mapping entries.
- Created scale plan: all original 297 height-band targets, up to 160 views per
  target, deterministic view-offset 8, varied valid torso/base/head snapshots.
  Started two additional isolated workers for already-reviewed seed101/seed11
  after checking GPU headroom (maximum four workers, no user GUI/robot control).
  These are new proposal windows, not repeated images to inflate a quota.
- The 10,000 / 500 / 500 accepted-image release, final stratified visual QA,
  portable export/loader audit and baseline evaluation remain unfinished.

#### Stratified release QA and query-association correction

- Added training_release_review: hash-bound full-scene/native-evidence cards,
  review-only cyan query marker and exact answer, append-only explicit decisions,
  and separate assistant/human attribution. Complete exports now require two
  inspected examples per nonempty family/difficulty stratum (or all if only one),
  plus every capture profile. Preparing cards grants no approval. Holds/rejects,
  stale records and missing strata fail closed. Portable releases retain the
  inspection evidence and recompute its coverage instead of trusting a flag.
- Refactored validated candidate gathering for reuse by review/export. Cached
  per-audit hashes instead of rehashing a large audit for every candidate.
  Portable validation also rejects duplicate RGB within a split, not only
  train/test leakage. Relevant broad suite: 442 passed plus 47 subtests before
  the subsequent v2 association correction; latest focused suite: 65 passed.
- At 11:22 KST, nine completed coverage families supplied 408 independently
  audited frames. The then-current v1 label contract accepted 331 (246 easy,
  8 medium, 77 hard), spanning 74 targets. These are historical v1 counts,
  NOT current-v2 accepted totals or a training release. Four capture workers
  continue; large jobs spend substantial CPU time preparing screened poses.
- Actual first-wave visual QA found a visible-cut example whose input query
  was an isolated petiole fragment across foreground foliage. Native mask
  connectivity confirmed this; related v1 examples also had disconnected
  associations. Preserved all initial decisions and added explicit holds and
  a follow-up note under grounding_release_wave1_20260909_v1. No failed review
  is silently rewritten into a pass.
- Versioned the task as greenhouse.target_conditioned_cutpoint_rgb.v2.
  Localized queries must share the nominal cut's eight-connected native visible
  target region, with no gap filling, while retaining >=45 mm attachment arc
  and >=18 pixel query/cut separation. Choose among valid connected queries;
  exclude when none exists. Occluded hard examples retain abstention and do not
  pretend a visible connection. Raw camera/depth/geometry captures are unchanged.
  New tests cover isolated-fragment rejection, connected alternative selection
  and hidden-cut abstention. A separate v2 review bundle is being prepared;
  accepted counts and visual evidence must be recomputed under this rule.

#### 2026-09-09: reproducible code checkpoint (dataset collection unfinished)

- Keep the implementation, tests, review GUI assets, versioned cut-rule config
  and operating documentation in Git on `koh-dev/sim-data`. Generated captures,
  native depth/masks, review decisions, collection ledgers and exports remain
  under the already ignored `data/` tree; they are not deleted by Git cleanup.
  No credentials or model-service calls are needed for this checkpoint.
- The current task is v2 target-conditioned RGB localization/abstention, using
  the actual mounted 848x408 head camera. The nominal cut is 10 mm along the
  petiole with a 10-20 mm acceptable arc interval. This is not autonomous target
  selection, dynamic manipulation, a trajectory release, or a trained VLM.
- The older `vlm_eval` exploratory prompt retains its historical 2-5 mm visual
  heuristic and autonomous target selection; it is NOT the current training
  contract. Training exports embed their own versioned prompt and contract hash.
- Added a diagnostic synchronous-USD viewpoint preplanner, with no Kit app,
  RGB/depth capture or approval. Its seed17 coverage probe completed in 118.67 s
  (99.85 s preparation) and every planned field matched the completed native
  worker reference within absolute numeric tolerance 1e-9. Scene/source guards
  passed. Evidence: `collection_preplans/grounding_cpu_seed17_probe_20260909_v1`
  under `data/sim_data`. This single-family diagnostic is NOT integrated into
  production workers and does not establish a general collection speedup.
- Before this diagnostic's comparison tests, the broad relevant suite passed
  446 tests plus 47 subtests. Current collection and v2 stratified visual review
  remain active; 10,000 train / 500 validation / 500 test accepted examples,
  all coverage gates, complete visual QA, portable release validation and
  baseline evaluation are still required before declaring dataset completion.
- Final checkpoint regression: `python.bat -B -m pytest sim_data
  vlm_eval/vlm_eval_test.py verify_robot_v12.py -q` passed **458 tests and
  47 subtests in 52.33 s**. `git diff --check` passed. The scoped changed-file
  credential-pattern scan found no matches; generated data/caches remain ignored.
  At the subsequent audit snapshot, 16 completed coverage jobs contained 725
  independently audited raw frames. V2 eligible counts are computed separately;
  raw counts do not imply training eligibility or complete visual sign-off.

#### Post-checkpoint v2 visual QA and portable engineering verification

- Code/tests/docs checkpoint committed as `e7d9861` on `koh-dev/sim-data`;
  subsequent `git status --porcelain=v1` was empty. No push or data deletion.
- Recomputed the 16-audit snapshot under v2: 562 eligible examples from 725
  raw frames (335 train, 102 validation, 125 test). Difficulty totals were
  426 easy, 8 medium and 128 hard. Excluded 64 disconnected-query examples,
  72 ambiguous junctions, 13 absent usable queries, 13 unknown visibility and
  one insufficient visual sampling/exposure example. These are a dated
  partial snapshot, not the full release or current live-worker totals.
- Completed all 41 selected inspections in
  `dataset_reviews/grounding_release_wave1_20260909_v2`: 27 newly inspected
  cards plus 14 exact-RGB/query/answer/sample/audit matches to prior actual
  inspections. Each has individual notes and assistant attribution. The
  source-bound review verifier passed the first nine families' nonempty
  difficulty strata and both capture profiles (10 established / 31 faster).
  This is representative QA, not human confirmation or statistical accuracy;
  new families and newly populated strata still require inspection.
- Built and validated `training_releases/grounding_v2_engineering_20260909_v1`
  from those nine source audits: 317 unique examples (215 train / 102 validation,
  no test rows). It contains clean RGB/chat records, native Z/masks/calibration
  sidecars and bound QA evidence. Its state remains explicitly
  `incomplete_engineering_export_do_not_claim_release`; visual QA passing
  cannot override missing release coverage. Historical v1 exports remain intact.
- Image-free baseline check on its 102 validation rows: query-copy median error
  72.07 px; fixed training-median pixel error 179.09 px; training-median query
  offset error 73.27 px. None localized a visible cut within 5 px. Always-abstain
  status accuracy was 27.45%; always-localize falsely localized all 28 occluded
  rows. These are small engineering shortcut checks, NOT VLM evaluation/training.
- Recomputed both completed matched-render comparisons under v2 in new
  `matched_pose_comparison_v2.json` receipts. At 21 matched poses eligibility
  agrees, with 17 jointly eligible identical answers (14 easy / 1 medium /
  2 hard) and four jointly excluded. The query pixel differs in 15 pairs;
  this is not an identical-query controlled model comparison or photometric
  equivalence claim. Earlier v1 receipts and failed render experiments remain.
- Collection continues in the existing four isolated workers; no live robot
  command or GUI reset was used for review/export. The complete 10k/500/500
  dataset and final all-family QA remain unfinished.
