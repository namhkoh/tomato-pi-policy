# Development Log — Greenhouse Deleafing Benchmark

Goal: an Isaac Sim benchmark for evaluating VLAs on tomato **deleafing** (removing
orphan/lower leaves from high-wire vines), targeting demo collection → π0.5
finetuning → deployment on the Rainbow Robotics **RB-Y1** (sim and real).

Current dataset track: `koh-dev/sim-data`, Isaac Sim 6.0.1 at
`D:\isaac-sim-6.0.1`; static robot-head native RGB-D capture. The legacy runtime
and branch description below belongs to the earlier physics/RL integration.

Current physics integration: `koh-dev/sim-vlm`. See the 2026-09-10 latency/
dense-context entry at the end and
[`PROOF_OF_LIFE.md`](examples/greenhouse_sim/sim_physics/PROOF_OF_LIFE.md)
for the revised no-teleoperated-demonstrations VLM + feedback-controller plan.

## VLM source checkpoint before simulator integration, 2026-09-10

- Checkpoint the pending VLM work on `koh-dev/sim-data` before creating
  `koh-dev/sim-vlm` from that commit: task-v3 advisory review/human follow-ups,
  task-v4 static active-perception contract, pilot, reviewer and subset exporter,
  launchers, browser checks, regression tests and documentation.
- Fresh verification: Isaac Sim 6.0.1 Python ran
  `-B -m pytest sim_data -q -p no:cacheprovider` from
  `examples/greenhouse_sim`: **591 tests and 47 subtests passed** (65.02 s).
  Node syntax checks passed for both reviewer clients and both browser-smoke
  scripts; changed Python files parsed successfully. This checkpoint did not
  rerun a live browser or launch Isaac Kit.
- Preserve existing source images, native depth, frozen splits, review decisions
  and generated evidence outside the source commit. No collection, training,
  model inference or changes to existing running jobs were performed.
- Physics restoration remains future work on the new branch; this checkpoint
  adds no dynamic episodes or claim of validated grasping/cutting. The revised
  research objective is automatically generated manipulation experience without
  teleoperated demonstrations, with VLM strategy proposals checked by geometry
  and feedback controllers. Historical teleoperation/RL notes remain below.

## Active-perception review/export increment, 2026-09-10

- Continued the dataset track on `koh-dev/sim-data`; no physical robot, Kit
  scene, original RGB/depth, or task-v3 label/contract changes in this increment.
- Prepared `data/sim_data/dataset_reviews/active_perception_20260910_v3/`:
  **28 review-only task examples** = 8 visible + 8 occluded + 8 invalid-organ
  queries + 4 new leaf-covered uncertain regions, on reused native frames from
  8 source families. These are not 28 fresh independent captures. The new
  regions use exact native leaf identities and valid depth; they are unknown,
  not fabricated no-target cases. True bounded absence needs new scene evidence.
- Added the separate task-v4 GUI and `run_active_perception_review.cmd` on port
  8882. It displays original 848x408 head RGB, native identity and coloured saved
  Isaac optical-Z; preserves source holds; and binds append-only assistant and
  human records separately to the exact pilot/example. No v3 decision is edited.
- Added a reviewed-subset exporter: explicit human acceptance required; source
  holds/unreviewed cases excluded; exact native RGB/depth copies, source-family
  splits, and private-truth separation checked. Chat messages are RGB-only;
  calibration/depth/evaluator truth stay in sidecars. It is not a full release,
  trainer integration or a motion dataset. No actual v4 training export yet.
- Browser verification exposed an SVG bug: assigning an HTML-style `hidden`
  property did not hide SVG markers. Fixed with an actual attribute toggle and
  added computed-style assertions. This prevents misleading query overlays on
  depth images with legends. An earlier transient image-load timeout did not
  reproduce in the subsequent full pass; the underlying endpoint returned 200.
- Current work still needs new-task visual review, true scoped no-target
  captures, broader balanced collection and release QA. Physical occlusion
  handling additionally requires verified native dynamic frame/time capture,
  executed viewpoint/left-arm reveal and recovery, and grasp-to-cut transitions.
  There are **zero collected dynamic episodes** in this pilot.
- Usage and safeguards: `examples/greenhouse_sim/sim_data/ACTIVE_PERCEPTION.md`.

## Dataset review suggestions and active-perception prototype, 2026-09-10

- Subsequent user feedback: "I had a look at your assessment and they largely
  look good. So I will follow yours." Recorded as broad endorsement of using the
  assistant assessment as the working baseline, not fabricated per-image human
  inspection. At this check, 18 human decisions were saved. Of the seven advisory
  flags, one had a human acceptance, four had human holds, and two remained
  unreviewed. Existing individual decisions were preserved; no reviews, source
  gates or GUI counts were changed by this acknowledgment. Unresolved flags and
  current export gates remain in effect. The original assessment remains
  assistant-authored; neither the whole release nor new v4 labels are approved.
  Provenance: `data/sim_data/dataset_audits/independent_v3_20260909/endorsements/user_follow_assessment_20260910.json`.
- Implemented hash-bound, read-only assistant suggestions in the task-v3 GUI:
  79 prior findings (72 support, seven advisory holds), evidence-first inspection,
  optional editable note copying, explicit independent/agree/disagree selection,
  and an advisory-holds filter. Suggestions never create approvals or source holds.
- Preserved all 14 existing human reviews and all 29 legacy assistant records.
  The 29 assistant-reviewed cards can now receive a separate human final pass in
  `human_decisions/`, bound to the unchanged prior record. Pending now means
  awaiting a human decision (94/108 at verification), not absence of any review.
  Export verification includes both histories and fails on human holds/rejections.
- Original RGB/depth/labels/assets are unchanged. Saved human decisions remain
  read-only. Real source/task holds cannot be cleared by accepting advice.
  Updated reviewer runs on `http://127.0.0.1:8881`; the existing 8880 process was
  not interrupted. Command: `examples\greenhouse_sim\run_training_review.cmd --port 8881`.
- Added separate `greenhouse.active_perception.rgb.v4` prototype contract/prompt
  and validators for observable eligibility, hidden/unknown targets, explicit
  invalid candidates, scoped no-target and measured execution constraints.
  Task-v3 contract hash remains unchanged. Hidden XYZ stays evaluator-only;
  non-localization cutting points are null, and no output authorizes a knife motion.
- Added bounded, reproducible static pilot preparation with immutable native
  source bindings, family-preserving splits, same-anatomy visible/hidden contrasts,
  native-supported invalid-organ queries, conservative hold filtering and explicit
  human resolution of current advice. No approvals are copied to new v4 tasks.
- Latest pilot: `data/sim_data/dataset_reviews/active_perception_20260910_v2/`:
  8 visible + 8 occluded + 8 invalid-candidate draft examples across 8 families;
  the negatives are five main stems and three fruits. All remain review-only,
  with zero dynamic episodes. v1 retained as an earlier all-main-stem pilot.
- Browser checks exercised the live GUI without real-data review writes and
  loaded the pilot's original images. The assistant visually inspected all eight
  v2 invalid-candidate screenshots; this is advisory, not human or physical approval.
- Verification: full `pytest sim_data -q` passed **559 tests plus 47 subtests**
  (43.75 s); JavaScript syntax and live browser checks passed with zero review
  writes. All 253 pilot source/advice/history bindings were rechecked unchanged.
  Browser evidence: `data/sim_data/review_gui/suggestions_20260910_v1/browser_final/`.
- Remaining: v4 human review/export, true bounded no-target/unknown examples,
  native frame/time synchronized dynamic recording, collision-checked viewpoint
  movement, physically validated left-arm reveal/recovery and transition to target
  grasp, then coordinated cutting. Frozen view pairs are not action demonstrations.
  See `examples/greenhouse_sim/sim_data/ACTIVE_PERCEPTION.md` and
  `vlm_train_data.md` for scope.

The legacy simulator/RL status below is historical and is not a claim that these
physical capabilities have been integrated into the current dataset preview.

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

#### 2026-09-09 13:03 KST: complete coverage recount and scheduled scale campaign

- Recomputed all 24 successful coverage-job audits under the unchanged v2
  contract. Result: **834 eligible / 1,103 audited raw frames**: 544 train,
  141 validation, 149 test. Easy/medium/hard counts are 401/3/140 train,
  97/3/41 validation, 121/3/25 test. Exclusions: 19 unusable queries,
  88 disconnected query-to-cut regions, 135 ambiguous junctions, 19 unknown
  visibility cases, 8 insufficient sampling/exposure cases. These are not
  automatically visual-review-approved observations.
- Family coverage now passes 16/4/4; distinct targets pass at 135/30/33;
  training cut spread passes at 162 occupied bins, with query-copy median
  error 59.27 px. Volume, localized/occluded counts, medium examples and final
  visual QA remain open. The immutable count/evidence receipt is
  `data/sim_data/status/coverage_complete_v2_20260909_1250.json`.
- Added `sim_data.collection_campaign` to schedule the remaining 22 source
  families without duplicating the two running training jobs. A new held-out
  plan uses 64 candidate views/target; training retains 160. Both preserve
  target cap 16, the original geometry, real torso/head snapshots, seed-0
  family assignments, and non-overlapping view-offset 8. Partial occlusion is
  still measured, never fabricated to fill the scarce medium class.
- Frozen campaign: `data/sim_data/collection_campaigns/grounding_scale_20260909_v1`.
  Four serial lanes cover 14 remaining training + 4 validation + 4 test
  families, up to 34,016 additional raw frames before geometry/visibility
  filtering. This is a maximum, not a promised yield or complete release.
  Held-out jobs precede training jobs within each lane. Lane 01 waits for
  seed101/job_001; lane 02 waits for seed11/job_003. Both require a matching
  clean-exit completion ledger and independently hashed audit before starting.
- Launched all four lane supervisors. Verified lane 03 starts seed79/job_020
  and lane 04 starts seed61/job_016; lanes 01/02 remain waiting with no capture
  folders. Exactly four capture workers were observed including the two
  existing jobs; GPU use was 20,465/32,607 MiB and free RAM 18.4 GiB at this
  check. The concurrency guarantee covers this campaign and its named
  predecessors, not unrelated simulators started externally.
- Each lane runs one bounded worker at a time, stops on capture/audit failure,
  refuses duplicate lane launches, and performs no automatic retry or deletion.
  Before each job it requires 169 GiB free disk for this schedule: a 64 GiB
  reserve plus four worst-job 12 MiB/sample storage envelopes. Queued jobs do
  not count as saved, audited or eligible training data.
- Regression suite passed **472 tests + 47 subtests in 36.15 s** (not additive
  with the previous 458-test run). New tests cover complete/nonduplicate
  allocation, split/source/shard mismatches, failed/stale predecessors, bounded
  short polling, serial execution, duplicate launches, low disk and stop-on-error.
  No physics, teleop, model training or external inference was performed.

#### 2026-09-09 13:50 KST: all-family visual QA and first clean scale completions

- Completed actual inspection and individual assistant records for all 63 cards
  in `dataset_reviews/grounding_release_wave2_20260909_v2`, covering the remaining
  15 initial-coverage families. Combined with wave 1, this is 104 inspected
  examples across all 24 families, not 104 human confirmations. Reviewed clean
  full head-camera RGB, cut overlays, native petiole identity, camera-Z, query
  association and the exact task answer. No unviewed card was auto-approved;
  one display failure was reloaded and inspected before its decision was saved.
- The seed101 scale worker completed all 973 planned frames, exited 0 without
  timeout, and passed the independent audit. Its result elapsed time is
  7,807.109 s; audit SHA-256 is
  `e29e95806afb469d6d91784f94eb2de55876a8b9697914b4b36ee97bc6351077`.
  Current v2 derivation accepts 688: 524 easy / 7 medium / 157 hard. The other
  285 are excluded: 100 disconnected queries, 38 unusable distal queries,
  122 ambiguous junctions, 6 insufficient sampling/exposure, 19 unknown
  visibility. No failed or partial worker data was promoted.
- Inspected all six representative scale-seed101 cards and recorded individual
  notes in `dataset_reviews/grounding_scale_seed101_20260909_v2`. Its partial
  cases retain a visible query-to-cut segment despite nearby occlusion; hidden
  cuts remain null-coordinate abstentions. Leaves, fruit and the main stem can
  all occlude a cut. Strong backlighting remains a dataset limitation.
- Combined source/label/deduplication and visual-evidence verification passed:
  25 audits, 2,076 audited raw, 1,522 eligible (1,232 train / 141 validation /
  149 test), with 110 actual inspections. Difficulty counts are 925/10/297
  train, 97/3/41 validation and 121/3/25 test (easy/medium/hard). Training cut
  spread is 178 bins; query-copy median error is 58.62 px. Receipt:
  `data/sim_data/status/coverage_plus_scale_v2_20260909_1350.json`.
  Its state is explicitly `audited_visually_checked_snapshot_not_a_complete_release`.
- Confirmed the predecessor handoff: lane 01 started held-out seed13 only after
  seed101's clean audited completion. Lane 03 subsequently completed seed79:
  267/267 frames, exit 0, audit passed, worker elapsed 2,274.047 s; audit hash
  `eb076df1ddae7e9aedb22adfcca6c3e44b9cc1f2b0d9ce1d619edb42adda030e`.
  It automatically continued to training seed17. The seed79 completion is not
  included in the preceding 25-audit frozen count. Its six selected QA cards
  were actually inspected and recorded separately; a new combined recount is
  in progress. Lane requests remain initial-state provenance, not live status.
- Four capture workers remain the limit. At 13:48 KST free RAM was 11.6 GiB,
  GPU allocation 21,197/32,607 MiB and D: free disk 449.9 GiB. No unrelated
  simulator or fifth capture worker was launched. Original source images,
  failed historical runs, v1 decisions and incomplete exports were preserved.
- Updated the dataset guide to distinguish historical v1 render qualification
  (18 eligible / 3 excluded) from the already recomputed current v2 result
  (17 eligible / 4 excluded), including the 15 changed query pixels. The
  labeling, rendering, split and release gates were not changed.

#### 2026-09-09 13:58 KST: second scale yield and native-depth requirement

- The frozen 26-audit snapshot adds seed79's 199 eligible / 267 raw examples:
  146 easy / 5 medium / 48 hard. All six selected scale-seed79 cards were
  inspected and recorded as assistant QA, not human confirmation or model tuning.
  The combined result is 1,721 eligible / 2,343 raw: 1,232 train, 141 validation,
  348 test. Test difficulty is now 267 easy / 8 medium / 73 hard, meeting the
  localized/occluded test-count gates but not test volume or medium coverage.
  All 116 representative decisions pass exact source/task/hash binding checks.
  Receipt: `data/sim_data/status/coverage_plus_two_scale_v2_20260909_1355.json`,
  with a companion hash-bound row-identity index. This snapshot is not a release.
- Reaffirmed the user's native-depth-only requirement in both dataset guides.
  Traced `capture_pilot.make_writer` to the native `distance_to_image_plane`
  annotator; `validate_payload` copies that float32 array, and `write_sample`
  saves it unchanged. `capture_search` uses that same production path. Our
  geometric projections provide anatomical supervision only; they do not
  generate the depth image. `depth_preview` only colour-displays saved values.
- Checked completed seed79 `sample_0152`: native 408x848 float32 camera-Z metres,
  saved-file hash valid, and loaded-array byte hash exactly matches the native
  capture's `synchronization.freshness.depth_sha256`
  (`b22a0535eb62081ab7055488bfcbd99f8c1ced0d784496ff9bee8e752c58df7f`).
  Static capture is still explicit: native frame number is unavailable and
  dynamic sensor synchronization is not claimed.
- Extended the depth provenance check to all 2,343 raw frames in the frozen
  snapshot: saved depth-file hashes match the sample metadata, and loaded
  float32 array byte hashes match the native capture hashes; shape and camera-Z convention
  checks also pass. No raw array was modified. Evidence is preserved in
  `data/sim_data/status/native_depth_provenance_20260909_1358.json`.
- Focused contract/export/visual-QA/campaign tests: 73 passed in 4.27 s.
  Native payload/pilot/depth-display tests: 45 passed in 7.59 s. An earlier
  invocation used a nonexistent test filename and ran zero tests; it was
  corrected to `training_test.py`. These focused suites are not added to the
  historical 472-test total. No capture or training contract code was changed.

#### 2026-09-09: complete wave-2 anatomy audit and v3 quality correction

- Completed actual individual inspection of all 63 wave-2 cards, including all
  30 hidden-query closeups and the boundary positive. The 33 visible nominal
  labels are on intended petioles; hidden nominal foregrounds are 16 leaves,
  9 fruit, 5 main stems. All hidden examples already answered abstain/null.
  The 57 distinct targets are intact leaf-bearing sub-stems, not fruit trusses.
  Report: `data/sim_data/dataset_audits/wave2_anatomy_20260909/audit_report.html`,
  with Markdown, CSV, source-bound diagnostics and per-image visual findings.
  Source bundle SHA-256:
  `9ae4b0ceb88486b120d7c5877563f64ff89330acd38264dcf49a0a4d6df914c3`.
- The reported seed41/SubStem_38 nominal projects behind MainStem_25; its native
  camera-Z is 0.319352 m vs target centreline Z 0.344622 m. This 25.27 mm Z gap
  is occlusion evidence, not a blade-clearance measurement. Source mesh checks
  did not place the nominal/10-20 mm interval inside checked own-plant main
  stems. These geometric checks are supervision diagnostics, not synthesized
  depth or robot/blade-safety certification.
- Found two inadequate old queries: seed71 `603809db886dad985f06` has a two-pixel
  connected visible island; seed53 `88e335449fb97690a44b` has 49 pixels and poor
  RGB readability. Earlier representative acceptance was too permissive.
- Fixed root causes in `dataset_review.py`, `training_release_review.py` and
  `training_contract.py`: hidden-cut overlays are suppressed; visible interval
  marks require native evidence and do not fill mask gaps; each training card
  includes a separate query crop and prominent localization/abstention/exclusion
  state. New `query_visibility.py` checks native island/local segment support,
  interior radius, frame margin, local exposure and luminance contrast. It
  never dilates target connectivity or generates/replaces native depth.
- Versioned task contract v3 and visual-QA schema v2. Portable export validation
  independently rechecks query usability. New `training_rescreen.py` validates
  original audit/card/sample bindings, derives new labels and writes exclusive
  comparison/card/browser outputs. It cannot migrate prior approvals or silently
  overwrite existing review decisions. Old portable v1/v2 task releases are
  historical artifacts and are intentionally not accepted by the v3 loader.
- Re-screened all 63 prior cards and their 15 audited source batches: 54 survive
  (32 localized / 22 abstentions), 9 excluded, 45 retained queries changed.
  Twenty old query points fail the new gate; usable alternatives rescue some.
  Both original holds and the reported seed41 card are now exporter-excluded;
  the latter's local target dark fraction is 0.5612, exceeding the 0.5 gate.
  The 15 source batches contain 488 v3 candidates and 207 exclusions. These
  numerical thresholds are conservative engineering judgments, not measured
  human/VLM readability accuracy; do not relax them merely to fill quotas.
- New evidence browser:
  `data/sim_data/dataset_reviews/grounding_wave2_rescreen_20260909_v3/index.html`.
  Personally inspected the three reported/held regenerated cards and four
  retained easy/medium/hard cards. Recorded only those four retained examples
  as fresh assistant QA; remaining v3 decisions are pending. No blanket approval,
  no human/agronomic confirmation, and no claim of a complete training release.
- Full sim-data tests: **486 passed, 47 subtests passed** (47.72 s). Added
  regressions for tiny/isolated/thin/dark/low-contrast/edge queries, immutable
  native observations, hidden overlays, mask gaps, stale approvals, exclusive
  re-screen outputs and rejected-source handling. Raw package/assets, RGB,
  native camera-Z, original annotations and old decisions were preserved.
- The frozen 29-audit v3 recount completed at 15:30 KST: **3,246 candidates from
  4,623 audited raw frames**, 1,377 exclusions. Splits are 2,102 train / 368
  validation / 776 test. Easy/medium/hard counts are 1,523/22/557 train,
  260/6/102 validation and 692/9/75 test. All 24 target-family splits remain
  intact, with 130/31/35 distinct targets. This is a numerical v3 candidate
  snapshot, not a visually approved release. Fresh all-family v3 QA remains.
  Receipt: `data/sim_data/status/post_anatomy_audit_v3_20260909.json`, with
  hash-bound row index and exclusions. All 4,623 float32 native camera-Z arrays
  match the capture-time native depth byte hashes; no depth was recomputed.
- Remaining coverage failures: train volume/localization/abstention counts,
  validation volume, and medium counts in validation/test. Training medium
  count now passes 20; do not claim held-out balance passes or lower its gate.
- Follow-up process audit found that the earlier campaign supervisors have
  exited; two native renderers (seed17 and seed103) remain running without
  their original parents. Seed37 job_010 and seed29 job_008 wrote 338 and 462
  frames respectively but have no final worker-exit/result ledger. Neither
  those 800 frames nor the two running jobs enter the 29-audit snapshot. Do
  not synthesize successful exit codes from a shutdown log. Recover verifiable
  execution evidence or rerun affected jobs in new directories, preserving
  original outputs, frozen splits/view windows and deduplication. The old
  campaign queue cannot be assumed to advance automatically. The user's GUI
  was left running and no capture process was killed by this audit.
- Implementation/docs committed as `d3d5fce` on `koh-dev/sim-data`; no push was
  performed. Next: restore durable bounded capture supervision, complete fresh
  stratified v3 QA, fill remaining coverage and validate a portable complete
  release before VLM fine-tuning.

#### 2026-09-09: detached capture recovery and fresh v3 review

- Added exact Windows process-handle observation (`collection_process`) and a
  new bounded three-lane restart (`collection_resume`). The normal runner now
  saves a hash-bound worker exit receipt BEFORE independent auditing. Export
  validation checks new/recovered receipts against the original launch and
  final result. Raw RGB, native depth, optics, scene geometry, family splits
  and task v3 labels were not changed by this supervision work.
- Detached hidden supervisors are running from
  `data/sim_data/collection_campaigns/grounding_resume_20260909_v1/resume.json`.
  Five completed scale jobs were skipped; one live job is recovered and 18
  unproven/unstarted jobs get new-directory captures. Initial new jobs are
  seed37 and seed31. The cap is three capture workers, with the GUI preserved,
  RAM/disk start gates, exclusive lane outputs and no automatic retries.
- Seed17 worker PID 83752 was bound by creation time, executable and full
  launch command. Its retained handle produced real exit code 0 / no timeout
  at 16:12:47 KST. The 1,164 saved frames entered `audit_recovered`; these are
  not part of the 29-audit training-candidate count yet. Receipt:
  `data/sim_data/collection_recovery/scale_20260909_v1/seed17_observer/exit.json`.
  Seed103 exited before observer attachment and was correctly rejected for
  adoption, joining seed37/seed29 for clean recapture rather than inferred exit
  success. Failed observer logs and all old outputs remain preserved.
- Fresh v3 one-by-one inspection increased to 20 unique cards: 19 accepts and
  one hold. `seed61_full_dc14e648a1bbf0d3eb4c` has correct native target identity
  but an inadequately readable dark triangular query fragment merging into
  foliage. Inspected the lossless crop too. Added task-level hold and append-only
  source hold `job_016/audit/records/4eb890a305a54806bec5670203fc2d2c.json` under
  the fast-B coverage batch. Verified the exporter excludes sample_0023 even
  when the task-review bundle is not supplied. No source pixels were changed.
- The hold-adjusted baseline is **3,245 candidates / 4,623 audited raw**, still
  pending full v3 QA. Incremental, hash-bound receipt:
  `data/sim_data/status/v3_after_fresh_hold_20260909.json`. Training/validation
  remain 2,102/368; test becomes 775. The prior 3,246 count remains an immutable
  pre-hold numerical snapshot, not a claim that this held frame is approved.
- Prepared **108 fresh v3 cards across all 24 source families** under
  `data/sim_data/dataset_reviews/grounding_all_sources_20260909_v3`. The held card
  is absent and the selected IDs still match the hold-adjusted candidate strata.
  Fourteen already-inspected v3 entries match task identity AND exact card SHA;
  explicitly referenced those inspections in the new bundle. No v2 approvals
  migrated and no repeat inspection was counted as a new unique sample. 94
  cards remain pending; the fresh bundle does not pass full visual QA yet.
- Full sim-data regression: **494 passed, 47 subtests passed** (42.45 s).
  Includes real Windows handle/exit tests, PID identity mismatch rejection,
  pre-audit receipt persistence, partial-audit preservation, exclusive restart
  allocation and unrelated-worker rejection. GUI and capture processes were
  not broadly stopped or reset. Restart behavior and limitations are documented
  in `examples/greenhouse_sim/sim_data/COLLECTION_RECOVERY.md`.
- This is a detached local process arrangement, not a reboot-persistent service.
  Future restarts must reconcile completion ledgers from the new resume tree,
  not blindly rerun the old campaign. Dataset completion still requires clean
  exits/audits, v3 labeling, fresh QA, volume/difficulty coverage and portable
  release validation. No VLM fine-tuning has started.
- Recovery follow-up: seed17's independent audit and label checks completed.
  Its 1,164 raw frames contribute **890 v3 candidates** (877 easy / 7 medium /
  6 hard), with 274 excluded. All 1,164 saved native float32 camera-Z buffers
  match capture-time byte hashes; the merge finds no duplicate RGB or camera/
  scene views. Combined current snapshot: **4,135 candidates / 5,787 audited
  raw**, 30 audits, **2,992 train / 368 validation / 775 test**. Receipt:
  `data/sim_data/status/recovered_seed17_v3_20260909.json` and bound row index.
  Historical pre-recovery counts above remain dated checkpoints, not current totals.
- Actually inspected all six representative recovered cards (two per difficulty).
  Four nominal cut labels are on the visible target petiole, not fruit/main stem;
  two foreground-leaf cases correctly abstain with null coordinates. Saved six
  hash-bound assistant accepts and verified the bundle at
  `data/sim_data/dataset_reviews/grounding_recovered_seed17_20260909_v3`.
  Fresh v3 inspection total is **26 unique cards: 25 accepts / one enforced hold**.
  The larger all-family bundle still has 94 pending; no blanket release approval.
- The recovered lane automatically advanced to new seed103 capture. Seed37 and
  seed31 workers now save new native frames under their detached supervisors;
  the existing GUI remains running. Inspected the first resumed head-camera RGB
  and verified all source hashes/native depth bytes, with evidence in
  `data/sim_data/status/resume_first_native_frame_20260909.json`. Unfinished new
  jobs are not added to candidate totals. No custom/replacement depth was used.
- Fixed progress reporting to use the explicit hash-bound recovered audit path.
  Final regression: **495 passed, 47 subtests passed** (46.13 s). Coverage still
  fails train volume/localization/abstention, validation volume and validation/
  test medium counts. Continue native collection and fresh QA before packaging
  a complete release or starting VLM fine-tuning.
- Broader follow-up: actually inspected 15 more cards from seed61, seed79 and
  seed101 (14 accepts, one hold). The new hold is
  `seed61_full_cd2eabc75725cdcefcdc`, scale seed61 sample_0469: native identity
  identifies a petiole, but the nominal shaft blends into a tomato in RGB.
  Inspected both lossless overlay crop and untouched unmarked RGB. This is a
  nominal-cut readability hold, not an inferred anatomy error or depth occlusion.
  Saved task hold plus source record `d5980d5e6c88438aa246e985b38753d4.json` in
  `grounding_scale_20260909_v1/lane_04/capture_job_016/job_016/audit/records`.
  Verified exporter exclusion without supplying task-review bundles.
- Current post-hold snapshot: **4,134 candidates / 5,787 audited raw**, with
  **2,992 train / 368 validation / 774 test**. The prior 4,135 snapshot is an
  immutable pre-hold checkpoint. New bound snapshot and row index:
  `data/sim_data/status/v3_after_nominal_rgb_hold_20260909.json`.
  A temporary accounting assertion initially used the wrong exclusion key;
  corrected it to `source_sample`, reused the exact existing hold records,
  and reran successfully without duplicate decisions or source changes.
- Fresh v3 QA now has **41 unique inspected cards: 39 accepts / two holds**.
  The broader 108-card set contains 28 accepts (14 explicitly reused identical
  v3 inspections), one hold, and **79 pending**. Exact source/task/card bindings
  verify; the held card is absent from candidates and the original bundle is
  correctly rejected for release. Preserve it, replace the held selection in
  a new bundle and complete remaining inspection rather than deleting a hold.
  Receipt: `data/sim_data/status/visual_qa_followup_20260909_v3.json`.
- Important limitation: v3 query readability and native geometric visibility
  do not independently guarantee nominal-cut RGB readability. Continue explicit
  nominal-region inspection; any later automatic screen needs its own evidence
  and contract/re-screen validation, not thresholds tuned merely to fill quotas.
  Full test rerun remained **495 passed, 47 subtests passed** (47.10 s).

#### 2026-09-09: task-v3 human review GUI launched

- Added `sim_data.training_review_gui`, local HTML/CSS/JS and the repeatable
  `run_training_review.cmd` launcher. Reused the existing loopback-only HTTP
  server with an optional UI asset directory; legacy prototype UI remains intact.
- Opened <http://127.0.0.1:8880> for the user. It starts with 79 pending of 108
  current all-family cards, and preserves 28 prior accepts / one hold as
  assistant decisions. No real-data human decisions were created by testing.
- Original RGB and saved annotated evidence are separate tabs. Both must load;
  an entered name, substantive note and explicit inspection checkbox are required
  before Accept/Hold/Reject. Saving advances, restart resumes, existing decisions
  are read-only. The answer panel explicitly distinguishes localization from
  occluded abstention; masks alone do not prove nominal-cut RGB readability.
- Hash-checks the selected bundle, audit, sample, original RGB, saved card,
  native depth/validity and target mask. Serves images lazily to limit memory.
  No Kit app, camera/scene changes, model call or regenerated depth is involved.
- Human Hold/Reject appends an existing-schema source block BEFORE the task
  record, so exporter exclusion works even without the task bundle. Interrupted
  task writes remain fail-closed; exact retries reuse the original negative
  source record. Acceptance cannot clear an existing source hold. No source,
  old review or frozen candidate snapshot is overwritten.
- Focused reviewer regressions: **62 passed**. Full sim-data: **516 passed,
  47 subtests passed** (48.17 s). Includes fixture-only persistence/restart,
  immutable decisions, hidden-cut acceptance, changed evidence, negative-source
  blocking, interrupted saves, duplicate posts, loopback/CSRF and legacy UI tests.
- Read-only real-browser smoke loaded the actual first pending card, original
  848×408 RGB and 1152×1230 evidence, with controls disabled before explicit
  inspection, no JavaScript exceptions, and zero review writes. Screenshots,
  result and server logs: `data/sim_data/review_gui/task_v3_20260909_v1/`.
  Inspected both browser screenshots. Three capture workers remain active.
- Instructions: `examples/greenhouse_sim/sim_data/TRAINING_REVIEW_GUI.md`.
  GUI decisions are per-label review only; the full dataset and physical cut
  safety are not approved. Future source holds require a fresh candidate recount.

#### 2026-09-10: VLM checkpoint and opt-in current-package physics

- Committed VLM/active-perception tooling on `koh-dev/sim-data` as
  `68f35001fbcbad07b6bb800592300150ff53ea95`, then created `koh-dev/sim-vlm`.
  No push or dataset/review/split mutation.
- Read-only recount across 55 completed grounding audits: 19,932 raw frames,
  14,235 current-contract candidates (11,627 train / 1,317 validation / 1,291
  test). These are static candidates, not individually approved dynamic
  demonstrations. Numerical release gates still fail validation medium 7/20
  and test medium 17/20; visual review and portable release remain incomplete.
- Current 108-card v3 human review: 18 decisions (8 accept / 7 hold / 3 reject).
  The v4 pilot has 28 static contrasts (8 visible / 8 occluded / 8 invalid /
  4 uncertain), zero human v4 decisions and zero dynamic episodes. No new
  collection, training, inference or physical-robot command was started.
- Added opt-in `examples/greenhouse_sim/sim_physics`: native-source petiole
  adapter, exact 10 mm separable seam, beam joints, area-integrated leaf
  inertia, batched PhysX state/contact access, shared physics/render clock
  and bounded qualification/reset harness. Original art, source layers and
  all dataset observations remain unchanged.
- Found duplicate active colliders behind hidden replaced plant meshes.
  Disable only replaced source contacts; retain neighboring plant collisions.
  Axis-aligned elementary joint probes match analytical deflection, but
  the full loaded chain is NOT qualified. Failure artifacts are preserved
  under `data/sim_physics`; no robot grasp/cut/retention success is claimed.
- Prevent SimulationContext from overriding requested solver/gravity settings.
  Earlier `_11`/`_12` comparisons are invalid; new runs check effective
  configuration. Isaac 6 tensor views reuse its explicitly attached stage.
- Metric depth remains original Replicator `distance_to_image_plane` float32
  camera-Z metres; color heatmaps are display-only. Dynamic native frame
  synchronization remains a separate required gate.
- Regression checkpoint: 611 tests and 47 subtests passed before leaf-inertia
  additions; latest focused mechanics/USD/clock tests: 22 passed.
  See `examples/greenhouse_sim/sim_physics/README.md` for commands and limits.
  Physics work remains experimental and uncommitted.

#### 2026-09-10: continued loaded-petiole qualification and dataset clarification

- Clarified the eligible v3 training pool: 11,627 static training candidates,
  comprising 9,156 localized (9,044 easy / 112 medium) and 2,471 occluded
  abstention examples. Validation/test are separate; these counts are not a
  final approved release. The task is RGB plus candidate-pixel instruction
  to cut-point UV / visibility / inspect-or-change-viewpoint JSON, not
  executable XYZ, robot trajectories, or dynamic action-outcome supervision.
  Native camera-Z stays a sidecar, not an inferred replacement.
- Retained additional failed physical probes. Increasing full-chain frequency
  to 1920 Hz (_15) still yielded 515 mm deflection; regular D6 constraints
  (_16) yielded 480 mm. An explicitly world-fixed articulation (_17) still
  exceeded the velocity gate. The main simulator has NOT been switched to
  this experimental adapter.
- Added attached-only diagnostic scope and a fixed-base comparison. It
  explicitly rejects seam release because that mode requires a validated,
  state-preserving topology transition. Tests verify the world anchor,
  unchanged source/session behavior for rejected modes, and no output/Kit
  startup when the fixed-base release request is invalid.
- The two-body joint reproduction exposes orientation/inertia-sensitive
  errors with no plant collisions. An angular-position-only pass was too
  weak: joint_probe_20260910_12.json shows a fixed-base angle error of
  0.000083 rad but residual joint velocity RMS of 0.263 rad/s. Added the
  final-half-second RMS velocity gate; these cases correctly remain failed.
  Root authoring alone is not a complete fix; cause is not yet fully isolated.
- The attempted GPU joint probe (_11) was overridden to CPU by context
  initialization, as its effective-scene record shows. It is NOT a GPU
  comparison. New probes reject requested/effective mismatch after reset.
- All diagnostic trials are non-training and kept under data/sim_physics.
  No new collection, inference, training, review edits, split changes,
  hardware commands, main-UI restart, or commit occurred in this continuation.
- Final regression run: 617 tests and 47 subtests passed (66.80 s), covering
  sim_data, sim_physics and explicit-clock/legacy-tick tests. This is code
  regression evidence, not a pass of the failed loaded-plant physics gates.

#### 2026-09-10: bounded original-petiole mechanics pass; robot task still gated

- Added opt-in coupled implicit elastic efforts using the native articulation
  mass matrix, original SI masses/inertias and authored stiffness/damping.
  Native drives are disabled only in this experimental mode. Native PhysX still
  integrates motion; no artificial mass floor, pose teleport or root spring
  force is used. Known force projection verifies COM Jacobians against native
  gravity compensation. Unknown contact loads remain explicitly unqualified.
- `data/sim_physics/qualification_20260910_28/report.json` passes bounded
  gravity/20 mN force response, attachment, diagnostic release and reset gates
  on the original SubStem_41 geometry. Measured pulse tip change: 21.75 mm;
  recovery residual: 1.08 mm. Reset restores controller parameters and exactly
  reproduces the first 0.5 s tip trace. Diagnostic joint release/free fall is
  NOT tissue cutting, robot grasping, retention or deposit. Training stays false.
- Fixed-root 240/480 Hz comparisons (_21/_22) give 38.86 mm steady sag and
  pulse changes of 21.65/21.99 mm. Mesh refinement (_21/_26/_27, segment limits
  25/12.5/6.25 mm) gives sag 38.86/40.83/41.90 mm but pulse response
  21.65/24.10/26.86 mm: transient/damping convergence is NOT yet established.
  Material constants remain engineering priors, not measured tomato tissue.
- Added process-local physics-thread selection and opt-in Fabric transport.
  Paired _24/_25 trajectories have identical SHA256; headless tick wall time
  improves from 5.63 to 3.85 s for 6 simulated seconds. Latest _28 takes 3.82 s
  (~1.57x real time). This is an isolated single-petiole CPU result, not a
  rendered full-greenhouse performance claim. Native sensor sync is unverified.
- Regression before the final reset-replay addition: 629 tests and 47 subtests
  passed (54.57 s); _28 subsequently passes the actual native reset/replay gate.
  Failed native-drive probes remain preserved. README includes the explicit
  working experimental command and all scope limitations.
- Still on koh-dev/sim-vlm at base 68f3500, physics edits uncommitted. Main UI,
  dataset/review/splits and hardware were not changed; no collection/training
  started. Next gates: mesh/damping consistency, contact/grasp/slip validation,
  contact-verified cutting, retention/deposit, dynamic native RGB-D and then
  integrated greenhouse performance. Do not yet promote this as robot-ready.

#### 2026-09-10: visible original-petiole physics demo launched on request

- Added opt-in `run_physics_demo.cmd` / `sim_physics.demo` using the same native
  adapter and coupled elastic efforts, not a visual animation. Keeps a visible
  Isaac Sim window open with run-pull/recover, pause/resume, diagnostic seam
  release, reset/reattach, target close-up and whole-plant view controls. UI
  commands execute only at controller boundaries; reset rebuilds tensor views.
- Launched the isolated source plant with one dynamic petiole. The rest remains
  fixed; the main greenhouse/static collector was not replaced. Repeat uses
  bounded six-second trials with explicit reset; diagnostic release stops after
  0.3 s of free fall. Contact/velocity/support/deflection guards pause on failure.
- Inspected native `settled.png` and `pulled.png` under
  `data/sim_physics/demo_20260910_01`: the textured stem and original leaf meshes
  visibly follow the native motion. Initial/settled/pulled/recovered captures
  show zero body-pose advancement during render-only capture waits. These are
  diagnostic viewport views, not robot-camera or training observations.
- The visible process completed repeated pull/recovery and a diagnostic release
  with no recorded fault. Measured rendered tick real-time factor was ~0.62-0.68
  in initial cycles, before deliberate pacing/capture waits. This is slower than
  the headless qualification; no real-time full-scene claim is made.
- Focused mechanics, demo argument/force, clock and legacy-tick regression:
  52 passed (10.73 s); git diff whitespace check clean. No collection, training,
  physical-robot commands, dataset/review changes or commit. Existing review
  services were left running. Physics is still experimental and uncommitted.

#### 2026-09-10: actual left-gripper contact, movement and retention tests

- Audited the old `LeftGraspManager`: it belongs to the legacy runtime and can
  enable a fixed grasp joint. It is not integrated with the new package rig.
  Added separate `sim_physics.gripper_probe` instead of claiming that legacy
  task-state logic proves a physical grasp in the current environment.
- The fixture references current RBY1-A v1.2 left palm/finger/camera geometry,
  preserves imported finger masses and uses dynamic force-limited prismatic
  fingers. Palm motion is kinematic fixture motion, NOT full-arm IK. No physical
  robot command, artificial plant-gripper weld, plant pose following, collection
  or training is used. Dataset/review/splits and the visible pull demo are intact.
- Test `_01` at the requested 120 mm grasp position contacts off-shaft geometry
  on a leaf-bearing carrier during approach; the plant reaches 29.58 m/s and
  stops before grasping. This is a retained failure, not a corrected contact
  model. General leaf contact remains unqualified.
- At the clearer requested 80 mm location (actual segment centre 71.13 mm),
  `_02` establishes opposing native shaft contacts and follows a 10 mm palm
  translation by 9.187 mm, with 1.207 mm maximum relative slip. Maximum measured
  penetration is 0.465 mm. This passes limited fixture gates, not calibrated
  tissue grasp realism, safe neighboring-plant clearance or population reliability.
- `_03` uses the same case with zero finger friction (minimum combine mode).
  The fingers still load the shaft but target movement is -0.006 mm and slip is
  10.006 mm: correctly fails grasp-follow/slip gates. This negative control
  supports that the positive result comes from contact friction, not a weld.
- `_04` tests diagnostic seam release at 4.75 s. The branch remains between
  fingers in the saved 5.3 s view but slips up to 11.188 mm in the 4.8-5.4 s
  hold interval. Opposing load is present in 82.1% of samples, failing the
  explicit low-slip retention gate. Do not claim reliable post-cut retention.
  Joint release is still NOT blade/tissue cutting; no deposit was attempted.
- `_05` repeats the positive case headlessly and includes reset/replay gates.
  Its complete physical trace exactly matches rendered `_02`. Native reset body
  error is 1.054 mm and first-half-second replay error is zero (_03/_04/_05).
  Artifacts: `data/sim_physics/gripper_20260910_01` through `_05`.
- Inspected native `_02/closed.png` and `_04/hold_after_diagnostic_release.png`.
  RGB captures freeze physics during render waits and are non-training diagnostic
  viewpoints, not robot-camera observations. Grasp coordinates are privileged
  fixture inputs, never presented as observation-verified VLM execution.
- Added fixture/no-weld/session-isolation/argument/force tests. A new offline
  regression caught drive writes going to a weaker root-layer edit target;
  closure now explicitly writes only the session layer. Live probes already
  used the session edit target, so this repair does not change their physics.
  Open-prismatic anchor warnings remain documented; no warning was hidden.
- Next: leaf-contact stability, contact-feedback retention and calibration,
  then collision-checked full-arm integration and dynamic native RGB-D. Do not
  collect action demonstrations or promote the current fixture as robot-ready.
- Final regression after the session-layer repair: **654 tests and 47 subtests
  passed (53.99 s)**. `git diff --check` is clean. All five bounded gripper
  probes have exited; the original visible pull demo and review services remain
  running. Changes are uncommitted on `koh-dev/sim-vlm`.

### 2026-09-10 ? Full-robot grasp diagnostic and replay repair

- Added `sim_physics/full_robot.py`, its tests, benchmark full-robot options,
  and `run_full_robot_grasp_demo.cmd`. Uses the complete v1.2 robot with a
  fixed base, dynamic links and fingers, joint-drive IK, and the original plant
  raised 350 mm in an isolated test station before physics construction.
  Right arm/knife stays parked. No hardware commands, data collection or training.
- Full-robot tests exposed what the kinematic-palm fixture could not: contacts
  between the palm and fixed neighboring foliage. Replaced the top-down entry
  with side entry and moved the robot clear of the upper canopy. No plant
  collision exclusions or grasp weld were added.
- Eight explicitly listed mounting-proxy self-contact exclusions are used
  for this fixed-torso diagnostic. Other self contacts, inter-arm checks and
  plant contacts remain enabled. These conservative proxy exclusions are NOT
  a validated collision model for arbitrary torso/tool motions.
- Native evidence: `data/sim_physics/full_robot_demo_20260910_01/trial_001`.
  Completed all 1,680 steps: bilateral shaft grasp, target movement 8.884 mm
  for 10 mm command, maximum slip 1.542 mm, maximum penetration 0.401 mm,
  maximum gripper contact 0.282 N. Reset/replay gate passed. One case only;
  no claim of reliable cutting, retention/deposit or whole-greenhouse readiness.
  Target coordinates are privileged test inputs, not perception-verified.
- Full-robot rendered tick throughput was 0.45x real time in this trial
  (7 simulated seconds / 15.47 tick-wall seconds), excluding capture waits.
  Do not claim this diagnostic is yet optimized for real-time greenhouse use.
- Earlier failed startup/placement trials are retained in
  `data/sim_physics/full_robot_20260910_01` through `_06`.
- The first GUI trial passed, but a requested replay closed the app because
  camera setup appended a duplicate transform operation. Camera setup is now
  session-only and repeatable; added a three-call regression test.
  Full robot / close-up / head and wrist D405 view buttons are available.
  Changes remain uncommitted; dataset reviews/splits and source assets untouched.

### 2026-09-10 - User-confirmed full-robot checkpoint

- Relaunch `full_robot_demo_20260910_02` passed the same grasp/move gates;
  the user reports the grasp looks very good and requests greenhouse integration.
- Before the camera repair, the full regression completed: 663 tests and
  47 subtests passed. After repair, 27 focused tests passed, including the new
  repeat-camera/session-layer regression. No collection or training started.
- Commit the current isolated physics/full-robot checkpoint before beginning
  greenhouse integration. Next increment: actual supplied greenhouse geometry,
  local detailed plant physics, native environment contacts, repeatable grasp
  test and measured timing. Keep the static dataset workflow unchanged.
- Final checkpoint regression: **664 tests and 47 subtests passed (58.86 s)**.
  User-confirmed demo remains running; reviews and source assets are untouched.

### 2026-09-10 - Supplied-greenhouse physics integration (qualification in progress)

- User-confirmed isolated physics and full-robot grasp checkpoint committed as
  `ea168ab` on `koh-dev/sim-vlm`. The isolated GUI was subsequently closed
  gracefully to run bounded greenhouse tests without competing simulations.
- New `sim_physics/greenhouse_scene.py` loads the supplied package's
  `house/green_house_base.usd`, preserves all 75 gutters and their geometry,
  and uses the exact foreground-plant placement from the package preview:
  `seed101_full` at [-0.005, 0, 0.900] m. Four distant instanced plants on
  the same gutter provide context; nearby neighboring plants are not added yet.
- One detailed target petiole remains compliant; other plant structures are
  static collision geometry. Building, floor and gutter collisions stay active.
  The robot is a complete fixed-base dynamic v1.2 articulation, upright torso,
  with base support sampled from the original floor triangles (surface 0.101 m,
  base origin 0.102 m). No artificial floor or source-gutter height changes.
  This floor-placement geometry calculation is NOT camera depth generation.
- New sparse native contact-event accounting checks robot self contact,
  non-finger target contact, and finger/arm contacts with the environment or
  non-target plants. Opposing impulses cannot cancel each other. Native tensor
  bilateral contact is cross-checked against the callback stream; callback
  errors latch until reset/rebind. No collision pairs are disabled for speed.
- `sparse_contacts_20260910_01`: isolated control passes; all recorded physical
  poses/contact/slip fields exactly match the earlier rendered full-robot
  checkpoint. Native event monitoring itself did not change those dynamics.
- Greenhouse `greenhouse_physics_20260910_01`: bounded 7-second run and reset
  pass, target follows 8.747 mm with 1.591 mm maximum slip, but loaded opposing
  contact falls below the required 90% movement fraction. This is NOT a pass.
- `_02`: a bounded pressure-bias experiment weakens the grasp and correctly
  blocks movement. This controller was removed. A 0.001 N contact-report
  threshold on the fixed base/wheels eliminates zero-impulse floor proximity
  warning spam without disabling floor collisions.
- `_03`: a bounded tactile-centering experiment also fails sustained contact;
  removed rather than made the default. Four physics worker threads do not
  improve native step time in this scene.
- `_04`: identified uncompensated gravity along the nearly vertical finger
  slides. Native compensation is approximately 0.311 N per finger. New optional
  `--finger-gravity` reserves that effort INSIDE the original 0.5 N total
  budget, leaving about 0.189 N drive force. Both shaft contact loads are now
  balanced (~0.10 N each). However, pulling brings a leaf into the wrist-camera
  collision body at 4.296 s; the 0.623 N unwanted-contact guard correctly stops
  the trial. Do not claim completed greenhouse grasp/move or ignore this hit.
- Native integration is still slow: median native step approximately 34 ms at
  a requested 240 Hz; rendered runs about 0.11x real time. Warning suppression
  and additional threads did not solve the bottleneck. Optional `--profile`
  records call timing; performance is not yet qualified.
- All diagnostic artifacts are under `data/sim_physics/`, non-training and
  ignored by Git. Original asset hashes remain unchanged. No review decisions,
  dataset splits, collectors, training jobs or hardware interfaces were changed.
- Next tests: bounded wrist-entry tilt to clear the camera/leaf interaction,
  readable greenhouse views, native timing investigation and full regression.
  Cutting, post-release retention, deposit, general leaf-contact robustness and
  synchronized dynamic robot-camera RGB-D remain separate unpassed gates.
- `_05`: -15-degree wrist tilt is rejected by native contact with fixed
  neighboring foliage during approach. `_06`: +10-degree tilt passes the
  complete 7-second greenhouse grasp/move/open test and deterministic reset.
  Target movement 7.825 mm, maximum slip 2.489 mm, penetration 0.370 mm,
  maximum gripper net contact 0.233 N. This is a single-case pass, not a
  success-rate estimate, tissue-cutting result or general collision plan.
- Native images visually inspected: `_05/initial.png` (exposure corrected,
  wide framing subsequently improved) and `_06/closed_detail.png`.
  Demo-only source-light exposure is lowered by three stops in the session
  layer; no original lighting assets or dataset capture settings are changed.
- `_07`: attempted the dedicated PhysX CPU dispatcher. It failed to reach
  the probe-ready marker after roughly four minutes, with no progress beyond
  local-payload loading. Stopped only its verified process (PID 116268);
  removed the experimental option. No episode or speedup is claimed.
- Final integration regression: **674 tests and 47 subtests passed (63.28 s)**.
  The passing `_06` run has opposing shaft load in **94.19%** of movement
  samples, maximum unwanted contact **0.111 N** (below the 0.5 N stop bound),
  zero reported non-excluded self-contact load, unchanged source hashes and
  zero first-half-second reset/replay error. Rendered tick throughput remains
  **0.108x real time**, median native step **34.20 ms**. Performance is still
  an explicit blocker; passing the grasp gates does not waive it.

## Greenhouse latency, dense vines and proof-of-life scope, 2026-09-10

- Started from clean `koh-dev/sim-vlm` at `b9b83b8`. Continued the current
  full-robot greenhouse physics track; no source assets, dataset splits,
  human/advisory decisions, model credentials or collection jobs changed.
  Closed only the verified prior physics-demo window to run bounded profiling.
  Review services were left alone. No training or hardware commands started.
- Split timing of the installed native physics-only step. Stage/time/dt lookups
  are small; most time is in native `simulate`. Disabling optional native
  profiler instrumentation and removing disabled infrastructure body APIs
  produced no useful improvement. Disabling far wire collision flags alone
  also did not help. Failed startup diagnostics `greenhouse_opt_20260910_02`
  and `_03` exposed USD BBox binding/guide-purpose mistakes, fixed before
  qualification; those runs do not count as successful tests.
- Causal controls `data/sim_physics/greenhouse_scene_profile_20260910_01`
  through `_04`: full scene ~35 ms/step; without wires ~29 ms; without
  gutters ~9-10 ms; without backdrop or building ~35-38 ms. Removing only
  the repeated gutter visual modules gives ~10 ms. Removing the entire
  background gives ~3 ms but removes required functionality and is NOT the
  adopted fix. All ablation records are explicitly non-qualifying.
- Implemented session-only batching of **3,525 identical gutter visual modules**
  into one point instancer. Exact source geometry/materials are referenced;
  all **75 gutter collision proxies** remain. Reject animated, mixed-prototype
  or physical visual modules. Test transforms, preserved collisions and
  unchanged source/session ownership.
- Implemented a guarded fixed-base 4x4 m XY wire collision window. It keeps
  12 intersecting proxies, deactivates 7,038 unreachable non-rendered guide
  proxies and leaves visible wires/floor/building/gutter/target geometry.
  Full world collision bounds (including invisible/guide-purpose shapes)
  determine retention. All native robot/dynamic-plant collision spheres are
  checked per step with a 15 cm margin; incomplete body coverage is rejected.
  This is not mobile-robot streaming: changing the base requires a rebuild.
- Restored original preview planting positions on three gutters: **143 supplied
  context plants + one detailed target**. Native static triangle-mesh contact
  is added to 71 nearby context plants through anonymous prototype stages;
  72 distant plants remain visual-only instances. Only the selected petiole/
  carriers are compliant; context is not a validated flexible-plant model.
  All 75 gutters are still visible, but not all 75 are populated with vines.
- Rendered bounded results with all original grasp/contact/effort/240 Hz gates:
  sparse batched `greenhouse_opt_20260910_06` native-step median **5.99 ms**,
  tick throughput **0.446x real time**; dense
  `greenhouse_dense_20260910_01` **6.82 ms / 0.411x**, compared with original
  **34.20 ms / 0.108x**. The no-Fabric dense control `_02` is slower
  (**9.10 ms / 0.331x**), so Fabric remains enabled.
  These are per-trial measurements, excluding startup, paused screenshots and
  reset; not real-time operation, sensor FPS or average task performance.
- All optimized runs above passed the limited grasp gates. Their **entire
  1,680-step physical records equal the original baseline**, including robot
  tracking/contact, plant/finger motion and zero half-second reset replay
  error. No artificial grasp weld or increased finger force was introduced.
  Target movement is 7.825 mm, max slip 2.489 mm, penetration 0.370 mm.
- Inspected native images: sparse `_06/closed.png`, dense `_01/closed.png`,
  and normal-step repeat `greenhouse_dense_20260910_03/reset_replay.png`.
  Greenhouse/vines/robot remain visible after reset. The latter run also passes
  grasp/reset gates without the step-profile wrapper. These are diagnostic
  viewport images, not new training data or synchronized mounted RGB-D.
- Isaac Fabric still emits a point-instancer prototype-mismatch warning on
  reset. It is not suppressed; inspected renders and physical replay pass.
  Broader sensor/render/reset qualification remains required before dynamic
  dataset use. A ~0.9 s once-per-trial IK replan also remains a UI latency cost.
- The default greenhouse launcher now enables batching, the guarded collision
  window and three populated gutters. Automatic paused milestone PNGs are
  off for interactive playback; a checkbox enables them for the next trial.
  Rendering, contact feedback, camera-view buttons, Run/Stop/Reset and bounded
  evidence capture remain available. Added a post-reset evidence snapshot.
- Initial regression: **685 tests + 47 subtests passed (61.89 s)**.
  Final regression and interactive relaunch results are recorded below.
- Added `examples/greenhouse_sim/sim_physics/PROOF_OF_LIFE.md`: parallel
  VLM grounding/abstention fine-tuning and completion of the mechanical
  grasp/cut/retain/deposit controller. Proposed 8B LoRA baseline, frozen 32B
  comparison, smaller model only after measurement. Four-H200 host access/
  availability and deadline still need confirmation. No trainer was added
  or launched in this optimization increment.
- The existing task-v3 candidate data is static and not a VLA action dataset.
  Privileged fixture IK does not prove observation-driven VLM execution.
  Native synchronized reobservation, guarded flat-blade cutting, retained
  orphan/deposit and recovery must pass before auto-generated dynamic
  action/outcome records can support a learned low-level policy. Explicit
  acceptance gates, effort estimates and failure-record requirements are
  documented; none of those future results are claimed as achieved.
- Final regression: **685 tests + 47 subtests passed (62.01 s)**;
  `git diff --check` passed. Normal-step dense repeat `_03` passes at
  **6.75 ms / 0.421x** with unchanged monitored source hashes.
- Relaunched the optimized persistent GUI as Kit PID **95964** (hidden helper
  cmd 60624). Windows reports a responding `Isaac Sim Python 6.0.1` window.
  `data/sim_physics/greenhouse_optimized_demo_20260910_01/trial_001/report.json`
  confirms all grasp/reset gates pass with milestone captures disabled.
  Actual GUI throughput is **0.345x**, 7 simulated seconds in 20.28 measured
  tick-wall seconds, median native step 8.36 ms. Do not substitute the faster
  headless rendered timing for this visible-UI measurement.

### Bimanual knife foundation and bounded additional data (2026-09-10/11)

Branch remains `koh-dev/sim-vlm`, based on `b9b83b8`. User explicitly requested
additional collection, left-hand grasp/right-hand cutting, and use of the
existing deleafing knife with careful engineering. No source asset, frozen
split, review decision, approved release, or hardware state was changed.

**New static captures, not trained policies or dynamic episodes:**

- Generated immutable plan `data/sim_data/collection_plans/grounding_sunday_20260910_v1/plan.json`
  (16 targets, 32 views, new proposal offset 128). The first validation worker
  failed after 114 captures with NumPy `_ArrayMemoryError` during static
  geometry screening. Its durable exit receipt records return code 1. Output
  `collection_batches/grounding_sunday_20260910_v1` remains `failed_do_not_train`;
  it was not salvaged by overriding worker/review state. Simultaneous heavy
  collection and physics qualification exhausted Windows commit headroom.
- Changed scheduling, not data semantics: new plan
  `collection_plans/grounding_sunday_small_20260910_v2/plan.json`, two targets,
  12 views, offset 176, unchanged family reservations. Ran three serial workers
  with `--instance-backend fast --render-budget warm56_then8`, separately from
  subsequent physics runs. Batch `collection_batches/grounding_sunday_small_20260910_v2`
  completed with exit/audit receipts for every worker: **48 raw captures**,
  12 `seed101_full` train-family, 14 `seed13_full` validation-family, and
  22 `seed31_full` test-family. Automatic clear-view counts are 10/13/18;
  these are numerical gates, not human anatomy approvals or difficulty quotas.
- Within each job: `capture/sample_?/inputs/rgb.png` and `depth_m.npy`, with
  native segmentation, calibration and provenance. Depth is Isaac Replicator
  `distance_to_image_plane`, not script-reconstructed depth. Native calibration
  checks passed; full-scene mounted-camera 848x408 capture contract preserved.
  All 48 remain pending visual review. No new final task-v3 export or VLM
  training was started; none of this batch is action/outcome experience.

**Knife integration and contact mechanism (experimental):**

- Found an actual mount-frame error in the fitted v1.2 asset: blade EE Z range
  0..71.48 mm and U-support 61.47..123.77 mm lie back into the arm; wrist
  geometry occupies Z=0..46.5 mm. `sim_physics/knife.py:mount_forward` rotates
  the existing knife 180 degrees around EE Y at the same flange origin,
  retaining flat-edge +Y cutting direction and moving the unchanged geometry
  to distal -Z. Session-only, idempotent, does not restore the right tongs or
  edit the source robot/knife. Fastener-level CAD fit remains unqualified.
- `bimanual.py` adds native right-wrist feedback, cached actual knife geometry,
  bounded orientation/edge-wing IK candidates and dense inter-arm screens.
  `robot_kinematics.solve_pose` now accepts a bounded evaluation budget while
  preserving the old default. Bimanual planning uses 250 evaluations per solve.
  Whole-tool/swept-scene collision certification is still incomplete.
- Extended sparse native contact accounting with point-specific expected tool
  loads. Only `BladeCollision` contacts inside the actual leading-edge strip,
  on the designated two seam-adjacent stem capsules, qualify. Broad plate,
  arc, camera, neighboring plant and protected-structure contacts remain guarded.
- `ShearGate` requires a verified opposing left stem grasp, <3 mm slip,
  transverse geometry, native force 0.2..0.5 N for >=25 ms and >=0.3 mm measured
  relative loading travel. No commanded velocity, timer or disconnected taps
  can substitute for this evidence. Thresholds are **engineering priors**, not
  measured tomato fracture properties. `release_from_blade` validates evidence
  before disabling the preauthored 10 mm seam joint. This is a contact-triggered
  joint-failure proxy, NOT calibrated tissue cutting or arbitrary mesh fracture.
- `bimanual_probe.py` executes guarded grasp/plan/approach/load/withdraw/retain
  phases and logs failures separately from successful release. No welds, plant
  pose overrides, hardware commands, or automatic training approval. Retention,
  actual material separation and completed withdrawal must independently pass.
  Deposit/recovery and moving-scene RGB-D observation integration remain TODO.
- Added a collision-geometry startup screen before native motion, including
  hidden/instance proxies, triangle/quad surface refinement, and convex solid
  containment. Broad boxes alone wrongly flagged the empty space of merged
  foliage; the refined check clears the known safe station while retaining
  physical contact guards. No collision shapes were disabled for cutting.

**Evidence and current stopping point:**

- `data/sim_physics/bimanual_cut_20260910_01`: rejected opposite-side station;
  native collision guard stopped at spawn. No cut or valid grasp. Never present
  it as a successful bimanual demonstration.
- `bimanual_cut_20260911_02`: conservative preflight rejected broad foliage
  boxes; this exposed missing quad refinement, subsequently fixed with tests.
- `bimanual_cut_20260911_03`: original stable station + corrected knife + all
  143 contextual plants. Refined spawn screen passes (118 broad pairs refined,
  zero remaining overlaps). Native left-arm approach/grasp ran **840 steps /
  3.5 s**; every measured 3.0..3.5 s hold frame had opposing stem contact,
  last minimum separation -0.334 mm, max unintended contact 0.111 N. The right
  cutting path is not feasible under the current IK/arm-clearance candidates;
  it fails closed before right motion. **Zero blade contacts, zero cuts, no
  post-cut retention claim.** Source hashes unchanged; `grasp.png` is rendered
  diagnostic evidence, not a robot-camera training frame.
- `run_bimanual_cut_probe.cmd` is explicitly labeled experimental/unqualified;
  the established grasp demo launchers remain on the prior qualified behavior.
  Next work is a jointly planned bimanual station/pose with full-tool corridor
  checks, followed by native force-controlled stroke/retention qualification.
  Do not weaken contacts, slip/force guards, or classify timed release as a cut
  to obtain a deadline demonstration. The full requested sequence is NOT ready.
- Final corrected-mount replay `bimanual_cut_20260911_04/report.json` explicitly
  records `left_grasp_verified=true`, hold bilateral fraction 1.0, and
  `collision_clear_right_plan=false`. The knife is visible extending from the
  flange in `knife_mount.png` (visually inspected); right tongs are absent and
  the D405 assembly is retained. All monitored source hashes match. This is
  failed-sequence diagnostic evidence, not a grasp-cut demonstration.
- Regression: **730 tests + 47 subtests passed (67.27 s)**, covering the native
  harness, kinematics, fixed physics clock and static data pipeline; subsequent
  edits only restrict candidate support orientation to the upward side and
  remove unused imports. No collection/physics worker from this turn remains
  running. All data/review releases and training jobs are unchanged.

### Guarded bimanual planning and portable Qwen preparation (2026-09-11)

Started clean on `koh-dev/sim-vlm` at `985791f`. User requests continued guarded
grasp/cut work, native data collection and a clean committed worktree. User will
transfer the finished dataset to H200 servers before training. No training,
hardware commands, source-asset edits, split changes or human review edits.

- Added bounded initial station offset, equivalent gripper roll, fixed torso
  yaw and 10-80 mm pregrasp distance options. Original defaults stay unchanged;
  shortening approach does not silently reposition the base. These are initial
  test-station choices, not mobile-base/torso trajectories or qualified poses.
- Refined startup capsule/triangle screening with exact finite segment/triangle
  distance. Enclosing-box empty corners no longer falsely imply capsule contact;
  true intersections, convex solid containment and unsupported-shape fallbacks
  remain. Native contacts are never disabled to make the knife reachable.
- Added cached actual-capsule robot self screening, including arm-versus-torso,
  before native initialization and on initial/gravity-settled left paths and
  right candidate paths. Existing joint/mount filters are read, not expanded.
  Full robot coverage is 17 capsules / 118 eligible pairs; 16 non-capsule tool,
  gripper, camera and chassis shapes remain explicitly unsupported by this
  particular screen. It does not certify full tooling or whole-scene paths.
- Cut-plan diagnostics now record endpoint IK errors/evaluation counts,
  clearance rejection and failed paths, instead of an uninformative empty list.
  An experimental collision-penalty IK residual did not find a valid solution
  and was removed; no unused new optimization API is shipped.
- Preserved failed native runs: `data/sim_physics/bimanual_cut_20260911_05`
  rejects shifted-base robot/plant overlap at startup. `_06` clears the plant
  screen but native guard stops a folded left arm intersecting torso_5 on the
  first step. No grasp/cut occurs in either. The new self screen rejects the
  exact `_06` pose before physics, covered by a regression test.
- Offline diagnostics find torso-clear short-pregrasp configurations, including
  forward/left offset (0.06,0.24) m, torso yaw 20 degrees, roll 180 degrees.
  Twelve right endpoints solve IK and three clear endpoint screens, but all
  three straight joint-space approaches cross the left arm. Endpoint reach is
  not a collision-free trajectory. This configuration is not native-qualified;
  no successful full grasp-cut-retain/deposit sequence is claimed.
- Completed new serial native collection `grounding_sunday_20260911_v3`, plan
  with 2 targets / 16 views / offset 208, frozen family reservations unchanged:
  job004 seed13 validation 18 raw (205.52 s), job009 seed31 test 32 raw
  (298.38 s). Both durable worker exits are zero and independent audits complete.
  Forty task-v3 eligible rows (16 validation easy, 22 test easy / 2 test hard).
  The other 10 are excluded, not relabeled. No medium examples in this batch.
- Together with `grounding_sunday_small_20260910_v2`, 98 raw / 84 eligible:
  train11 (9 easy / 2 hard), validation29 (28 easy / 1 hard), test44 (38 easy /
  6 hard). New hash-bound review bundles `grounding_sunday_20260911_v3` and `_v4`
  contain 17 individually viewed assistant accept decisions (10 visible,
  7 occluded). The folder suffix is a review wave, not a task-v4 dynamic dataset.
  Human decisions and prior assistant holds/rejects remain untouched.
- New `training_exports/grounding_sunday_20260911_engineering` has 84 portable
  rows, original RGB and byte-preserved native Isaac Replicator optical-Z,
  validity/calibration/evaluator sidecars, frozen chats and review/hash receipts.
  Validated using ordinary non-Isaac Python with explicit incomplete mode. It
  deliberately remains `incomplete_engineering_export_do_not_claim_release`.
  Do not add its counts to the historical global recount without deduplication.
- Added `sim_data/qwen_adapter.py`: exact system/user/assistant preservation,
  original 848x408 RGB only, shared prompt conversion, verified generation-prefix
  loss masking, no truncation and no incomplete-release training override.
  Tests use a fake processor with Torch, not loaded Qwen weights/tokenizer.
  Actual model forward/backward, LoRA/distributed trainer and H200 performance
  remain untested; no inference-to-motion bridge is claimed. See new
  `sim_data/H200_HANDOFF.md` for user-transfer sequence and acceptance gates.
- Heavy Isaac jobs were serialized due Windows commit pressure; no collections
  or native physics trials from this increment are left running. Existing review
  servers are untouched. New data/logs remain ignored runtime artifacts, while
  source/tests/documentation are committed as a verified engineering checkpoint.
- Regression: **764 tests + 47 subtests passed (60.19 s)**. This validates code
  contracts/offline geometry, not physical cut reliability or Qwen performance.

### Sustained hold control and full-tool self screening (2026-09-11)

The prior checkpoint is `05f2a54`. Continued with bounded, headless current-package
trials; no hardware commands, training or dataset/review/split mutations in this
physics increment. Default established grasp launchers are unchanged.

- A deterministic outward-shoulder waypoint search now screens direct and up to
  three shoulder detours. Every vertex and <=1-degree joint sample preserves
  the 10 mm inter-arm and 3 mm self-clearance gates; finer +/-15/30-degree blade
  orientation candidates retain the original flat-edge and upward-support
  constraints. This is a bounded local search, not a complete motion planner.
- Native `bimanual_cut_20260911_07`: left grasp verified, right endpoint/path
  accepted by the then capsule-only checks; stopped at **4.5875 s** after losing
  opposing grasp contact. Zero blade contacts/cuts. The roll-0 control `_08`
  is rejected at startup because the left D405 overlaps a target leaf.
- Added explicit `--bimanual-hold-control`, keeping the right arm parked. Its
  reports intentionally do not pass cutting gates, even when holding completes.
  `bimanual_hold_control_20260911_01` loses the grasp at **4.65 s** without right
  movement. Plant trace shows growing torsional joint excursions and elastic
  energy (0.010 J at 3 s, 0.221 J at 4 s, 0.783 J at 4.3 s). Right-arm movement
  is therefore not necessary for the instability. A single successful short
  grasp does not establish sustained-contact or post-cut retention reliability.
- Added opt-in `--compliant-fingers`: native PhysX force-based compliant contact
  material on the existing left finger shapes only. Stiffness 1000 N/m;
  damping 0.7 times critical using the larger finger/stem reduced physical mass
  (1.0574 N s/m for this fixture). These are **uncalibrated engineering priors**.
  No mass/beam-stiffness inflation, contact removal, grasp weld, increased
  finger effort, relaxed slip/penetration thresholds or plant pose override.
  Native implicit pad compliance is supported by the installed NVIDIA example
  and [PhysX material schema](https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_material_a_p_i.html).
- Matched compliant control `bimanual_hold_control_20260911_02`: completes
  **4,800 steps / 20 s**, no error, 100% bilateral contact throughout 3..20 s,
  max post-verification slip **0.1132 mm**, max hold penetration **0.1026 mm**.
  Final 10..20 s plant-body speed <=0.02533 m/s and elastic energy <=0.02347 J.
  Source hashes unchanged. This is one successful sustained attached-stem hold,
  NOT a successful cut, detached retention, deposit, calibrated tissue model or
  qualified VLM action episode. The default rigid-contact demo is not silently
  promoted to this new material model.
- Full compliant trial `bimanual_cut_20260911_09` maintains grasp and advances
  right motion until **6.7583 s**. Native guard stops a right D405 body / left
  finger contact at 3.028 N; the knife support also brushes a target leaf at
  0.0123 N. Zero leading-edge contacts and zero cuts. Native feedback correctly
  exposes that arm-only planning did not cover the attached tool geometry.
- Extended `SelfCapsuleScreen` with opt-in conservative OBBs for all 16 formerly
  unsupported camera, bracket, knife, palm, finger and chassis colliders. The
  bimanual planner enables them: **33 shapes / 463 eligible pairs**. Capsule/
  capsule and capsule/box distances plus conservative box SAT and lower bounds
  check all shape bounds; this is still sampled planning, not continuous
  whole-scene certification. No collision filters are added or broadened.
  Regression explicitly detects a camera/finger intersection that arm capsules
  alone miss. Runtime cut planning snapshots the **actual held finger aperture**
  rather than using the initially open hand's frames.
- The new full-tool screen rejects previously accepted unsafe knife endpoints.
  Other grasp arcs, tilts and source-family station probes remain blocked by
  anatomy, scene, IK or tool-clearance checks; none is mislabeled successful.
  A jointly feasible grasp/tool station and whole-scene transit remain required
  before native blade loading, cutting, detached retention and deposit tests.
- Added plant joint position/velocity/elastic-energy diagnostics, robot joint
  name mapping, per-second phase/hold logs and explicit hold-only/close-up
  milestone names. Diagnostic viewport evidence is not native synchronized
  robot-head RGB-D and is not added to the VLM training set.
- Regression after the main additions: **768 tests + 47 subtests (60.73 s)**.
  Final hold repeat and final regression are recorded below before commit.

Continuation:

- Repeated the 20-second compliant negative control in
  `bimanual_hold_control_20260911_03` and `_04`. Both complete with no fault,
  unchanged source hashes and the same 0.1132 mm maximum slip. These deterministic
  repeats are not additional target diversity. Their cutting gates intentionally
  remain false. The new plant-side diagnostic view improves inspection, but
  foreground leaves still obscure part of the fingers; screenshots alone do not
  establish opposing contact. Native contact traces are the evidence.
- Added bounded initial `--station-yaw` (+/-90 degrees); this changes only
  initialization, not a live robot base. Zero preserves established launchers.
  Camera/finger/knife checks stay enabled. A changed station needs new native
  qualification; it does not inherit the earlier grasp result.
- Offline station yaw 60 degrees / requested grasp arc 100 mm finds a screened
  right approach and all 75 stroke samples with a -10-degree blade-plane tilt,
  30-degree transverse direction and -28 mm edge offset. It uses a -90-degree
  shoulder waypoint. The planner now tries the original plane first, then
  +/-10 degrees, retaining the original upward support, anatomical stem axis,
  angular/contact/force/slip limits and source tool geometry. Proposed blade
  normal is stored separately from actual stem axis. Stroke rejection records
  now identify the precise IK, inter-arm or full-tool screen failure.
- Native `bimanual_cut_20260911_10` tests that station with compliant fingers:
  left opposing-shaft verification fails at 3.5 s, before right planning/motion.
  One finger has shaft contact; the other does not. Zero cuts, source assets
  unchanged. Offline geometric success is therefore NOT a working native
  sequence. Added failed-grasp event diagnostics for the actual colliding
  shapes; no broad target contact is promoted to a valid shaft grasp.
- Regression: **780 tests + 47 subtests passed (66.48 s)** after the bounded
  blade-plane search. Earlier station-option regression: 771 + 47 (62.40 s).
- Started two serial collection jobs (004 validation seed13 and 009 test
  seed31) under `grounding_sunday_20260911_v4`, using the prebuilt 4-target /
  24-view / offset-240 plan, fast instances and warm56_then8 rendering. This
  folder suffix is a batch version, not task-v4 dynamic experience. Counts,
  independent audits and new visual QA must be checked after both workers exit;
  no approval or successful collection is implied by launch. Frozen splits,
  existing decisions and native depth provenance remain unchanged.

### 2026-09-11: collection completion and handoff-preview validation

Physics checkpoint committed as `d98b82e`; no successful cut is claimed.
Bounded offline yaw 50/55-degree variants at the previously held grasp arc
still fail right transit; yaw 65 is rejected for a torso/plant spawn overlap.
No unsafe configuration was executed or collision margin relaxed.

- `collection_batches/grounding_sunday_20260911_v4` completed serial jobs004
  and009 with durable exit code zero, unchanged source bindings and independent
  audits. Validation seed13: 35 raw in 348.72 s, 31 eligible (28 easy / 3 hard).
  Test seed31: 71 raw in 526.37 s, 48 eligible (45 easy / 3 hard). Total
  **106 raw / 79 eligible / 27 excluded**, zero medium. Logs and failed
  label reasons are preserved. This is static task-v3 data, not task-v4 action
  experience despite the batch version suffix.
- Actually inspected all eight representative cards in new
  `dataset_reviews/grounding_sunday_20260911_v5` (four localized, four occluded)
  using full head RGB and both cut/query RGB, native-mask and native-depth
  crops. Recorded individual assistant accepts with limitations, not human
  confirmation, anatomical-fracture validation or tool-safety approval.
- Also individually inspected and recorded **22** previously unreviewed cards
  in `grounding_all_sources_20260909_v3` across seed19/23/41/43/47/53/67.
  Broader queue now: **58 accept / 8 hold / 3 reject / 39 unreviewed** out of
  108. Old decisions and human history are untouched. These are representative
  inspections, not a measured dataset-wide accuracy estimate.
- Built NEW `training_exports/grounding_sunday_20260911_engineering_v2` from
  the seven completed jobs across three recent batches. **163 deduplicated
  rows / 204 raw**: train11 (9 easy / 2 hard), validation60 (56 easy / 4 hard),
  test92 (83 easy / 9 hard). Includes 25 hash-bound representative QA records.
  Prior preview is retained. No addition to the historical 14,235-candidate
  global total is claimed without a fresh global deduplicated recount.
- Ordinary `egodelta_robot` Python validates every portable RGB/depth,
  validity, label, chat and hash with explicit engineering mode. The normal
  `GroundingDataset` correctly rejects this preview with `Incomplete release`.
  No model weights/processor, training process, server connection or hardware
  command was launched. H200 transfer follows a future complete release only.
- All native workers from this increment have exited; no collection/training
  or physics worker is left running. Existing unrelated review servers remain.
  Next data work is targeted medium-visibility acquisition plus remaining
  stratified QA and global reconciliation, not blind relabeling of easy/hard
  frames. Updated `H200_HANDOFF.md` and `vlm_train_data.md` accordingly.

### 2026-09-11: cutting geometry, scene-aware approach and native failure audit

User requested improved successful cutting. **The implementation is improved,
but a successful current-environment cut is still unverified.** No guards were
weakened, no contact was relabelled to force a pass, and no timer/pose override,
grasp weld, hardware command, collection/training job or dataset change was used.

- Fixed contact diagnostics: `target_palm()` clears the native contact stream
  for the next tick. Failed-grasp diagnostics had read that cleared stream.
  Each completed native step now snapshots its collider pairs, so the failure
  event retains actual blocking contacts. Planning-time native body frames,
  actual left joints and finger aperture are saved separately for reproduction;
  these privileged diagnostics are explicitly non-training.
- Added bounded `--grasp-depth-m` (90..125 mm, original 102.5 mm default).
  Original finger collision pads span palm Z=-135.5..-73.5 mm. The 125 mm
  proposal moves the hand, not the plant/base or finger geometry. Settled-goal
  updates preserve that chosen depth. Grasp still requires opposing selected
  shaft contact, not a finger stopped by an attached leaf.
- Added cached held-plant geometry from native body poses plus active local
  static contacts, including hidden and instance-proxy shapes. Cache creation
  occurs after context population. A 2 m cube about the fixed right shoulder
  bounds the checked region; any proposed right shape leaving it is rejected.
  Trial18 includes 271 local static colliders. Source convex hulls/triangles and
  conservative tool bounds are not a continuous whole-scene certificate.
- Added deterministic bounded bidirectional joint-space detours after the
  direct/outward-shoulder options, maximum 300 iterations / 4,000 predicate
  calls with <=1-degree path samples. Full-tool/inter-arm/plant checks remain.
  Stroke is screened before spending the transit budget. Failed planning wall
  time is recorded separately: an exception can bypass PhysicsClock's completed
  tick timing, so its real-time factor is not an end-to-end planning metric.
- Replaced arbitrary +12 mm post-seam stroke extent with shaft radius + half
  leading-strip width + 1 mm, sampled at <=0.5 mm. For this fixture the endpoint
  is +4.990 mm. Actual native edge load/dwell/travel still gates release.
- Added explicit `--cut-arc-m` for a diagnostic seam within the existing agreed
  10..20 mm longitudinal interval. Default 10 mm, source assets, annotation
  rule, existing reviews and dataset labels are unchanged. A 20 mm offline
  alternative also fails the sampled full-scene approach search; it is not
  native-qualified or an automatic execution fallback.
- Found and corrected a source-geometry mismatch in the knife: actual long
  plate sections have Z=-6.5..-0.5 mm, while the inherited full bounding box
  used Z=-6.5..+6.5 mm. Its centre-Z=0 semantic edge was above that long plate.
  `blade_contacts.py` partitions all source triangles at Y=-11.9819 mm into
  plate and mount convex contact hulls, retaining both and preserving source
  surface area. The usable long leading strip follows the measured 2-degree
  slant, is centred at Z=-3.5 mm, and is 49.528 mm long / 6 mm thick. The old
  enclosing box is superseded only in the session layer. This is not a claim
  that the supplied 6 mm plate is a calibrated sharp-tissue cutting model.

Native evidence, under `data/sim_physics` (all failures preserved):

| Trial | Measured result |
|---|---|
| `bimanual_cut_20260911_11`, `_12` | Requested 100 mm grasp selects 95.576 mm body centre. Only one opposing shaft contact; the other finger is stopped by Segment005/Leaf000. Trial12's corrected pair log identifies this explicitly. No right motion/cut. |
| `_13` | 125 mm pad depth at that same farther grasp clears the palm/leaf contact but does not establish opposing shaft contact. Rejected at 3.5 s. |
| `_14` | Requested 80 mm grasp selects 71.126 mm body centre, depth125, station yaw60. Grasp verified; right forearm brushes distal leaves and grasp is lost at 5.6875 s. No cut. |
| `_15` | Native held-plant snapshot screening rejects the leaf-conflicting stroke before right motion. |
| `_16` | Geometry-sized stroke plus held-plant-aware detour keeps opposing grasp through 7 s, then native guard catches a 44.357 N forearm/tomato contact. This was a failed diagnostic, not safe execution. It motivated adding surrounding static geometry. |
| `_17` | Local static scene screen rejects the blocked approach at 3.5 s, avoiding the earlier right-arm collision. Native planning snapshot saved. |
| `_18` | New source-derived blade contacts cook/load and grasp verifies (100% bilateral during 3..3.5 s). 180 endpoint attempts: 114 IK-converged, none pass all geometry checks. No right motion, edge contacts or cuts. Source hashes unchanged; paused native inspection images saved. |
| `bimanual_hold_control_20260911_05` | Same revised station/tool, right parked: all 4,800 steps / 20 s complete without a guard fault; maximum post-verification slip 0.06903 mm. Cutting gates intentionally false and control exit remains non-success. Attached hold only, not severed retention. |

Additional read-only offline checks explored original/finer blade angles,
redundant IK seeds, six station/torso combinations and six other eligible
seed101 petioles. They did not establish a complete sampled collision-clear
sequence; alternatives failed IK, self/scene clearance or spawn checks. These
bounded searches do not prove the task physically impossible. +/-15-degree
blade proposals were inspected offline only; the implementation's +/-10-degree
proposal and original measured angular gate remain unchanged.

Regression: **799 tests + 47 subtests passed (86.63 s)** across sim_physics,
robot kinematics/clock/RL tick and sim_data. Source-surface preservation, exact
plate profile, negative cut controls, deterministic detours/budgets, static
contacts, unchecked-region rejection and unchanged default cut rule are tested.
No success rate, calibrated tissue fracture or H200 readiness is inferred.
Reset also invalidates the cached static scene and previous planning timing;
the next plan rebuilds its scene snapshot. Final focused regression, including
that reset test: **87 passed (22.02 s)**. `git diff --check` is clean.
Inspected trial18's native knife-mount and grasp-close images: the supplied
knife remains visible; foliage still partly obscures the grasp in the close-up.
These are diagnostic views, not robot-camera training inputs or visual proof
of opposing contact. The force/identity traces provide that evidence. All
workers launched for this increment have exited; no jobs were left running.
Next required work is joint grasp/cutter configuration selection against the
complete scene (including refinement of conservative tool bounds where
warranted), followed by actual native blade load/release/withdraw/retain tests.
Do not promote these diagnostic records into a VLM action dataset.

## 2026-09-11: user-requested knife roll and nearer ground-truth grasp

Branch `koh-dev/sim-vlm`, following `4415ecb`. User asked for a 180-degree knife
rotation, testing cutting, and grasp aligned with the ground-truth junction.
The protected attachment is NOT the grasp/cut itself: retain the 10 mm nominal
cut and grasp on detachable material with physical finger/tool clearance.

Implementation:

- Corrected `knife.mount_forward` with a 180-degree **wrist-Z roll relative to
  the previous distal mount**. Distal -Z is retained; another Y flip would put
  the knife back into the wrist. Flat cutting direction is now wrist -Y,
  curved support on +X. Recognizes explicit source/previous/corrected frames,
  preserves flange translation/camera pose, is idempotent, rejects unknown
  frames. Common-parent transform rotates mesh, arc, contacts and edge together.
  Source CAD/USDs and camera mounting assets are unchanged.
- Replaced the full-robot grasp's arbitrary minimum segment-start rule with
  actual complete finger collider bounds projected onto the cut-plane axis:
  all bounds must lie on the detachable side, at least 10 mm from the plane.
  It remains only a placement screen; full-tool/path/native guards stay active.
  Added explicit ground-truth target identity, attachment, cut, grasp, requested
  arc and selected physical body centre to reports. This is privileged truth,
  not perception-verified execution or an annotation/dataset change.
- Requested 50 mm selects Segment002 at **46.675573 mm** attachment arc, rather
  than the previous 71.125955 mm grasp. Original 32 mm-wide pad bounds leave
  **20.675572 mm** from the 10 mm cut plane. Station offset `.04 .2285`, yaw60,
  roll180, depth125 mm, approach20 mm keeps the base within about 0.48 mm of
  the previous station; changing grasp must not cause a spawn collision.
- Bimanual closure stops at known shaft radius minus 0.5 mm, not zero aperture.
  Nominal compression is an engineering setting, not measured tissue behavior.
  No force limit, collision exclusion, penetration limit, material stiffness,
  grasp verification or native blade gate was relaxed.
- Added optional session-only yellow attachment / white cut / cyan grasp
  visual markers, without collision or mass. Updates use native body frames
  only at render. Added `--no-robot-auto-run` for inspection before Run; default
  auto-run unchanged. Selected camera survives replay initialization. These
  diagnostic views/markers must not be used as clean model training inputs.

Native evidence under `data/sim_physics` (all failed trials retained):

| Run | Result |
|---|---|
| `bimanual_cut_20260911_19` | Closer grasp with old station offsets rejected by spawn screen (torso/foliage, forearm/shaft). No motion. |
| `_20` | Station corrected; zero-aperture closure makes opposing contact but exceeds the existing 1 mm penetration limit at 3.4875 s (1.1464 mm). No right movement. |
| `_21` | Radius-minus-0.25 mm width stop avoids penetration, but one finger's selected-shaft force is only 0.013 N, below the unchanged 0.02 N minimum. Grasp rejected, not relabelled. |
| `_22` | Radius-minus-0.5 mm closure verifies stable bilateral contact; all samples at 3..3.5 s bilateral. Right search: 180 endpoints, 99 IK converged, 43 interarm rejects and 56 other self/tool rejects; no all-checks-clear endpoint, no right movement, edge contact or cut. |
| `bimanual_hold_control_20260911_06` | Same nearer grasp/tool/closure, right parked: **4,800 steps / 20 s**, no guard fault, continuous bilateral contact after verification, maximum slip **0.005615 mm**, maximum penetration **0.032025 mm**, source hashes unchanged. Attached hold only; overall cutting qualification remains false by design. |

Tests: 804 passed + 47 subtests (82.03 s) across physics/robot/sim_data before
the final closure constant and inspection-first option; focused final-change
regression then passed 78 tests (16.87 s); the final marker-only check passed
2 tests (4.43 s). GUI `greenhouse_cut_gui_20260911_02` loaded with the full
greenhouse/robot and inspection-first controls. Two GUI replays verify the same
grasp and reject the same 180 blocked right approaches, with no cut. The GUI
remains open; native plant/robot rendering was inspected. No collection,
training, hardware command, review
decision, split or training export changed. Successful cutting/withdrawal/
post-cut retention/deposit remains **unverified**. Next work is jointly choosing
a left grasp orientation and right-tool corridor, including mounted camera/arc
geometry, not bypassing safety gates.

## 2026-09-11: committed knife, paired-layout and contact-screen diagnostics

The requested knife mounting is committed as `d0866e4` on `koh-dev/sim-vlm`.
That wrist-Z correction is retained unchanged. The user requested continued
work until cutting is correct; **the complete native cut is still unverified**.
The previous interactive GUI was closed for serial native qualification;
the earlier statement that it remains open no longer applies.

- Replaced the source arc's unknown cooked convex-decomposition planning box
  with 14 explicit source-triangle spatial convex partitions. Every original
  surface is retained, and the upper open window is no longer filled by one
  contact box. The same parts serve native contacts and conservative planning
  bounds. Visible meshes/mounting/source assets are unchanged. Missing/disabled
  replacement parts fail knife validation. No protected contact filter changed.
- Added capsule-versus-oriented-box narrow-phase startup checks: an overlap of
  world bounding boxes alone no longer rejects the empty corner beside a rotated
  finger. Exact intersections still reject; unsupported shapes stay conservative.
- Added explicit fixed initial station XY/yaw and palm approach-vector options,
  without live base/plant motion. Relative and absolute station modes cannot mix.
  A bounded left wrist-seed fallback avoids interpreting one IK branch failure
  as unreachable. Default station/grasp/solver/effort settings are preserved.
- Right wrist proposals now include both signs of the transverse blade plane;
  the fixed mounting roll is not a world-up wrist constraint. Expanded angle/
  usable-edge sampling retains the same force, direction, clearance and grasp
  gates. An opt-in 8..25 mm precontact standoff is checked against actual shaft
  radius, edge width and clearance; it is not permission to start in contact.
- Added offline `sim_physics.paired_layout` with preserved rejection evidence,
  configurable initial poses and optional fixed-shoulder redundancy proposals
  (`redundant_ik`). These use exact URDF FK, not hardware commands. An endpoint
  proposal is NOT a checked transit/stroke or native grasp. Full greenhouse,
  dynamic contact and retention qualification must follow. No training approval.

Evidence in `data/sim_physics` (failures retained):

| Run | Measured result |
|---|---|
| `paired_layout_20260911_01` .. `_23` | Offline searches, some early ones interrupted and preserved as partial. Configurations, source-derived geometry, IK and collision rejection records; no physics stepping. `_13` finds three clear cutter endpoint proposals for a 117.802 mm grasp, but this does not validate grasp or the path. |
| `bimanual_cut_20260911_23` | Full greenhouse, farther 117.802 mm grasp. Native leaf contact prevents opposing selected-shaft contact. Rejects at grasp verification; no right motion/cut. |
| `_24` | Intermediate grasp likewise blocked by the first leaf: one shaft contact and both fingers touching leaf geometry. Not relabelled as a stem grasp; no right motion/cut. |
| `_25` | Original 46.676 mm grasp again verifies (all 3..3.5 s samples bilateral). Expanded search: 756 endpoints, 300 IK converged, 122 interarm rejects, 155 self/tool rejects, 23 scene rejects; zero all-clear endpoints and no cut. Scene conflicts include right upper arm / distal leaf and right arm / neighboring SubStem43. |

Regression: **822 tests + 47 subtests passed** (102.46 s), covering sim_physics,
robot kinematics, clocks and sim_data; log
`data/sim_physics/regression_20260911_knife_planning.log`. Tests of a contact gate
or offline endpoint are not measured tissue cutting or successful manipulation.
Remaining work: establish a jointly feasible native grasp and whole right-tool
corridor, then validate edge load/release/withdraw/retain with unchanged guards.
No hardware, collection, tuning, review, split or training-export changes.

#### 2026-09-11: Agent-assisted VLM release QA and export safeguards

- User explicitly requested independent agents for uncertain images. Three
  agents inspected 51 unique original RGB/card pairs with verified hashes:
  37 pending plus 14 existing uncertain/negative. Their reports and exact
  findings are in `data/sim_data/dataset_audits/release_20260911_agents/`.
- Appended 33 assistant task accepts and four task/source holds. A fifth new
  source hold records ambiguity in `seed11_full_b64924da6591cb0a7b72`; its prior
  human acceptance is preserved, not overwritten. `applied_assessments.json`
  binds every action to the agent report. Queue: 91 accept / 14 hold / three
  reject / zero pending; one nominal accept is blocked by that source hold.
  No blanket approval of all candidate images or human-confirmation claim.
- Three newly held examples are medium. The earlier medium-coverage count
  is not valid current approval. A fresh all-source recount and small native
  seed61/offset487 collection probe are running separately; numeric coverage,
  replacement QA and final portable archive remain unfinished.
- Added `sim_data/DATASET_CARD.md` explaining query-conditioned RGB input,
  exact JSON outputs, 10 mm nominal / 10-20 mm arc convention, native metric
  depth sidecars, frozen family splits and unsupported robot-action training.
- `training_export --reconcile-reviews` retains source review provenance but
  counts only exact current-label accepts. Any negative review on a retained
  image still blocks the export. Default strict behavior and all gates stay
  unchanged. Directory snapshots now catch new review files appended during
  copying, in addition to changed existing-file hashes.
- Complete regression: **842 tests plus 47 subtests passed in 110.66 s**,
  covering sim_data, sim_physics, robot kinematics and simulation clocks.
  New tests cover excluded-review accounting, stale accepts, retained holds,
  and review races during file copying. No training or hardware command.

#### 2026-09-11: Bounded held-target and grasp-posture diagnostics

- Added opt-in held-target reposition (0..10 mm), settling and fresh native
  grasp/seam snapshot before right planning. Default no-reposition timing is
  unchanged. Added bounded diagnostic closure bias (0.25..1 mm; default0.5)
  and +/-30-degree jaw-skew proposals; force/slip/penetration gates unchanged.
- Native `_26`: 8 mm reposition lost opposing grasp contact. `_27`: 4 mm
  reposition with 1 mm closure bias stopped at the native force guard.
  `_28`: 1 mm reposition with 0.75 mm bias retained opposing contact through
  reobservation at 5 s, max measured slip 0.192789 mm, but right planning found
  no clear endpoint. This is not a successful cutting or long retention trial.
- Added bounded 7-DOF pose-family continuation to offline `paired_layout`.
  Exact URDF pose residuals/limits are maintained; each returned posture still
  requires independent clearance and native execution checks. `_33` inspected
  508 proposals at the original near-junction grasp: no clear endpoint.
  `_34` changed-grasp tilt hit spawn overlap; `_35` torso -30 deg had no IK
  endpoint; `_36` torso +30 deg failed arm clearance. None authorized motion.
- `_37` moved the diagnostic grasp beyond the first leaf attachment (requested
  arc145 mm), preserving source plant, cameras and corrected knife. Three
  clear offline endpoint proposals were found. Full greenhouse native run
  `bimanual_cut_20260911_29` failed native grasp verification: leaf000 loaded
  both fingers, but the selected shaft did not have opposing contact. No
  right-arm motion/cut was authorized. An endpoint is NOT a valid grasp,
  stroke, cut, or success claim. All artifacts remain non-training.

#### 2026-09-11: Reconciled data counts and full-greenhouse grasp screening

- `release_20260911_agents/preflight_02/summary.json`: 63 completed audits,
  20,151 raw / 14,404 deduplicated current-contract candidates (11,637 train,
  1,369 validation, 1,398 test). Includes recovered seed17 `audit_recovered`
  and the 15 new seed61 easy frames; removes seven new source holds. The
  initial 61-audit discovery omitted the nonstandard recovery directory.
  `count_scope_check.json` records the exact provenance reconciliation.
- All numerical release gates except held-out medium coverage pass in this
  diagnostic: validation 2/20, test 17/20. Original 61-audit buffers were
  checked in the first recount, not rehashed in the follow-up diagnostic;
  final export must still run normal gather and portable validation. No ZIP,
  training release, trained VLM or qualified dynamic episode is claimed.
- Agent D inspected two further original/native audit-card pairs and recorded
  conservative source holds via the main agent. Agent B inspected/accepted
  two new seed61 original/task-card pairs. All 55 unique inspected examples
  have attributed findings and hashes. No human records, input images,
  native depth arrays, family splits or release thresholds were changed.
- Export source holds now follow duplicate RGB hashes across audits. Tests
  prove a second, unreviewed copy cannot reintroduce a withheld image.
- Added a shared left-grasp approach/closure collision screen before native
  movement and expensive right IK. Only finger contact with the selected
  detachable shaft and its immediate discretization neighbors is expected;
  foliage, other branches, support, palm and wrist cameras remain checked.
  Actual bilateral contact on the selected body is still mandatory. This
  rejects the leaf-blocked distal proposal that failed native run29.
- Native hold control07 was stopped during excessively slow scene-cache
  preparation; no success or trajectory was inferred from it. Optimized mesh
  triangulation (retaining both quad diagonals), workspace clipping and
  conservative indexed triangle lookup. Same geometry/margins/narrow-phase
  tests and native contact guards; no plant/robot collision filtering added.
- Native hold control08 completed 4,800 steps / 20 s without a guard fault:
  65 approach/closure checks passed, selected-stem grasp verified, maximum
  slip **0.005253 mm**. Headless tick time 38.8635 s, RTF0.5146; native-step
  p50 5.8516 ms / p95 9.0819 ms under concurrent offline work. This is not a
  GUI FPS or hardware measurement. Deliberate no-cut control exits with
  failed cutting gates; it does not prove cut, detached retention or deposit.
- Full regression after these changes: **852 tests + 47 subtests passed in
  99.94 s**. Indexed geometry tests agree with exhaustive checks, including
  long/degenerate triangles, grazing bounds and solid-hull containment.
- Bounded alternative station/posture searches remain unsuccessful. They
  preserve original source assets and the committed knife mounting; endpoint
  proposals never bypass native grasp, contact, force, direction or slip gates.

#### 2026-09-11: Guarded training archive preparation

- Added `python -m sim_data.release_archive --release <complete-release>
  --output <new.zip>`. Always runs the normal complete-release validator,
  includes only manifest-bound payloads, checks hashes of bytes read from the
  ZIP, and publishes a SHA256 sidecar. ZIP64 supported; incomplete/interrupted
  archives cannot be mistaken for successful final files. No overwrite.
- Four packaging unit tests pass. Ordinary non-Isaac Python also correctly
  refused the actual `grounding_sunday_20260911_engineering_v2` preview with
  `Incomplete release` before creating any ZIP. No training archive, server
  transfer or training start is claimed. This command is ready for the final
  approved export, not a way around remaining coverage/review requirements.

#### 2026-09-11: Cut evidence integrity and pre-IK rigid-tool checks

- Independent code/results assessments are retained in
  `data/sim_physics/cutting_model_review_20260911.md` and
  `paired_geometry_review_20260911.md`. None of native cut25..29 or hold08
  recorded an accepted blade-edge contact; they do not validate mechanical
  feasibility of the 0.2..0.5 N / 25 ms / 0.3 mm loading condition.
- Fixed two evidence-accounting bugs in `ShearGate`: precontact approach
  travel no longer enters the first qualifying contact window, and tolerated
  submicrometre reversals subtract from net travel instead of ratcheting a
  false cut. All physical thresholds stay unchanged. New negative tests cover
  a large free-space approach followed by stationary load, a force dropout,
  and 1,000 tiny forward/back cycles. These are logic tests, not tissue tests.
- Added reusable `RigidToolScreen`: checks fixed left-hand/arm geometry versus
  proposed right-wrist hardware, and hardware versus the scene snapshot,
  before arm IK. A pass explicitly does NOT certify the omitted right arm,
  its transit, native grasp/contact, or cutting. Original caches are not mutated.
- `rigid_corridor_20260911_01.json`: 12 targeted 75 mm-grasp / 120-degree
  sector configurations, negative near-end wings, +/-10-degree plane tilt,
  10/25 mm standoffs. All are conservatively rejected at the first pose by
  left D405 body / right bracket bounds; total offline wall time 5.317 s.
  This is not proof of actual mesh collision or global unreachability. The
  bracket's enclosing bound is being compared with its source geometry.
- Focused regression: 38 tests pass (knife, rigid tool screen, archive). A
  separate native force-limited loading diagnostic is still needed to resolve
  the rigid-interface/indentation-model question. No automatic force-threshold
  relaxation, timed release, hidden collision exclusion, or successful cut.

- Follow-up assistant A source-geometry inspection (offline, not native):
  right bracket visual/collision meshes match at 2,308 triangles; enclosing
  box is 40x44x41 mm, approximately 65.1% empty relative to source mesh volume.
  Native collision uses convex decomposition; planner uses the enclosing box.
  For 120 deg / -18 mm wing / normal -1 / zero tilt at requested75 mm grasp
  (actual71.126 mm), the 25 mm staging pose's left camera OBB clears bracket
  triangles by 4.78 mm, while the 10 mm pose intersects 861 triangles including
  the mounting foot. Thus an initial source-mesh false positive does NOT make
  the straight stroke clear. Remaining ten variants and native cooked hulls
  were not measured. Refine conservativeness without shrinking hardware or
  claiming a source triangle test certifies the native decomposed collider.

## Explicit visible/occluded VLM baseline preparation, 2026-09-11

- User authorized the narrower `visible_occluded_v1` release before returning
  to cutting development. `balanced_v1` and its medium quotas remain unchanged.
  The new profile selects easy/clear and hard/occluded only; medium is excluded,
  never relabeled. Minimum counts, frozen families, target diversity, cut-pixel
  spread, query-copy check and representative visual QA remain required.
- Export/portable loader bind the profile to a distinct completion state and
  reject out-of-profile rows, mismatched difficulty/answers, arbitrary gates
  and incomplete previews. Consumer instructions are hash-bound package files.
- Independent review preflight found 115 exact current easy/hard representative
  accepts, 11 missing stratified inspections, and one retained legacy-v1 held
  RGB (`seed17_full_0f2fc164d4bdb3a8509e`). A newer connected-query accept does
  not silently erase that hold. New `training_review_history.py` preserves
  negative RGB identity across task versions and exports the historical
  evidence without changing original decisions or granting legacy approvals.
- Fresh normal source rebuild started from all 63 authoritative audit paths,
  including the recovered seed17 path. Diagnostic scope before the legacy
  exclusion: 14,273 easy/hard candidates; this is NOT a final release count.
  Fresh selected-card review is separate from source verification. Packaging
  cannot complete before normal coverage, representative QA and portable
  validation pass. No collection, training, hardware command or gate waiver.
- Focused exporter/review/history/archive/Qwen-adapter regression: **60 passed**
  in 4.48 s using Isaac Python, no SimulationApp. Existing simulator cutting
  status is unchanged: stable hold control, no qualified full cutting sequence.

- Follow-up independent code review closed a fail-open history omission: a
  complete baseline now requires a separately pinned, nonempty historical
  bundle/decision inventory *before* source gathering. Bundle omission,
  deleted decisions, changed negative cards and newly appended records fail
  checks. Twelve historical bundles / 20 negative records / 328 bound files
  are pinned for this build. Fresh task reviews may finish during source
  gathering but cannot erase those earlier negatives. No prior reviews edited.
- Expanded focused tests: **62 passed** (5.48 s). Combined simulator/kinematics/
  clock/data regression: **876 tests + 47 subtests passed** (110.58 s).
  The first portable-Python source rebuild stopped before export on missing
  `psutil` in the capture-exit validator; the checked restart uses Isaac Python.
  One own exporter was deliberately stopped before writing a release to add
  the inventory safeguard. Neither interruption affected the review GUI,
  simulation assets, source labels, splits, or any training job.

- The first new baseline review bundle has 24 individually inspected
  original/card pairs: **18 assistant accepts / 6 holds**, with no human
  confirmation. The remaining seed37/hard QA gap requires another immutable
  bundle; no old held row or bundle is overwritten to fill that gap.
- Added an explicit JSON review-list input, read and hash-bound after source
  derivation, so completed additional review bundles can join finalization
  without editing existing evidence. Original pinned history remains checked
  before and after the source scan. The list itself is also rehashed before
  publishing the manifest. Source byte hashing now uses at most four workers
  and batches of 256 futures; every file is still checked on both passes and
  progress counts are emitted. Focused regression: **64 passed** (9.83 s).
  Tests cover the final file in a multi-batch verification being modified,
  explicit unique/nonempty review lists and all previous profile/history gates.

### 2026-09-11: Baseline visual QA completed; normal rebuild still running

- Two new immutable baseline bundles contain 35 individually inspected original
  RGB/card pairs: 22 attributed assistant accepts and 13 holds. The second
  wave closes the remaining seed37/hard representative QA gap. Earlier review
  records and human decisions remain unchanged; this is not human confirmation
  or a measured corpus-wide label error rate.
- `release_20260911_agents/baseline_completed_review_preflight.json` passes
  snapshot coverage/QA with 136 exact unique representative accepts. Expected
  selected rows: 14,259 (train11,520 / validation1,358 / test1,381), comprising
  11,237 easy/clear and 3,022 hard/occluded. All medium rows, one legacy held
  RGB and the 13 new ambiguous baseline images are excluded, not relabeled.
- These are preflight counts, not yet the completed export. The normal source
  build has verified 232,171 bound source files and is rederiving all 20,151
  raw-frame labels from 63 audits. It must finish validation and archive-member
  readback before the ZIP is published. No training or hardware job was started.

### 2026-09-11: Native blade loading isolated from grasp/path planning

- Added `sim_physics/blade_loading_probe.py`: original approved knife and two
  source-derived seam-adjacent capsules, native PGS/240 Hz/gravity/Fabric,
  50 g diagnostic carriage on a force-limited implicit prismatic velocity drive.
  This is a **two-body coupon**, not the greenhouse, full plant or full robot.
  The seam remains enabled throughout; there is no release API or fabricated
  grasp. Source assets are hashed before/after and remain unchanged.
- Logs include actual body poses, both joint anchors, contact positions,
  normals, separations and impulses, projected edge force, full contact-load
  magnitudes and net advance. Later runs also gate native angular joint-frame
  residual. Configured drive limits are not reported as measured forces.
  Full load <=0.5 N, penetration <=1 mm, speed <=20 mm/s and intact-seam
  guards precede window credit. Broad-face contacts inside the leading-strip
  volume are rejected using their native normal (30-degree plane tolerance).
- `blade_loading_20260911_03`: rigid interface, 8 s / 1,920 ticks, no error,
  5.033 s contiguous qualifying force but only **0.0000167 mm** net loaded
  advance against the unchanged 0.3 mm requirement. This directly demonstrates
  a local rigid-contact stall; it does not prove every full-plant pose impossible.
- Explicit **coupon-only**, uncalibrated compliant-material variants: `_04`
  1000 N/m / 0.35 N drive cap gives 0.01336 mm; `_05` 250 N/m / 0.35 N gives
  0.00242 mm; `_06` 250 N/m / 0.45 N gives 0.03649 mm net qualified advance.
  All complete without guard faults but none satisfies the loading window.
  These are diagnostic parameter checks, not adopted benchmark coefficients or
  validated tissue fracture. Main simulator material defaults remain unchanged.
- `_01` exited before recording usable native evidence; `_02` exposed a
  float-exact gravity check (9.81 authored as float32), fixed with tolerance and
  explicit scene diagnostics. Failure reports now survive setup exceptions.
- Shared normal classification is integrated into the bimanual callback:
  absent normals/separations cannot grant a tool-contact allowance; broad-face
  contact is unwanted; accepted tool penetration is checked before seam logic.
  Existing force, dwell, net travel, grasp and collision thresholds are unchanged.
  Physics regression: **222 passed** in 61.91 s. A new full-robot hold regression
  is running; no full-robot cutting success or training eligibility is claimed.

- Follow-up limitation on coupon `_03..06`: the original callback reports
  contact-point impulses but not friction anchors, so their field named
  `total_contact_magnitude_n` is a normal-contact-only sum, not a full-load
  certificate. Raw evidence is preserved; do not reinterpret it as complete
  load measurement. `_06` changes both cap and velocity-drive damping, not
  only cap. Neither the force cap nor stationary poses measure applied effort.
- Added full native friction-anchor reporting and finite/buffer checks.
  `_07` repeats `_04` at240 Hz and preserves its physical trajectory: measured
  peak normal magnitude0.23069 N + friction0.08093 N gives a conservative
  0.31161 N bound. Friction is not credited as normal cutting-edge evidence.
  Signed native velocity and pose-derived velocity are logged separately;
  PhysX split-impulse handling can make them differ under stationary contact.
- Same1000 N/m / 0.35 N /350 Ns/m parameters, full reports: `_08` at480 Hz
  gives0.04231 mm and `_09` at960 Hz gives0.05127 mm maximum qualified advance.
  Neither passes. This tests timestep sensitivity, not material calibration or
  a converged whole-robot contact model. Full-robot simulation stays240 Hz.
  Native frictionType is explicitly patch and checked after reset; the current
  SDK emits a deprecation warning for the authored setting, which is retained
  in logs rather than suppressed.
- Full-robot sparse accounting now includes normal and friction magnitudes
  without cancellation. A mixed/unverified tool patch cannot inherit friction
  permission; buffers with missing/invalid data fail closed. Friction-only
  headers still count; empty lost-contact offsets are not dereferenced. The
  existing0.5 N allowed-tool guard therefore includes tangential load.
- `bimanual_hold_control_20260911_10` completes4800 steps with verified left
  grasp, no fault and maximum slip0.005253 mm after normal/penetration changes.
  It is a parked-right negative control, not a cut/retention/deposit pass.
  Full-friction regression `_11` is running; `_09` was rejected before launch
  because the diagnostic command omitted required PGS/implicit spring options.
- Integrated the existing `RigidToolScreen` into `_plan_cut` **before** each
  endpoint IK. Every complete <=0.5 mm tool stroke is checked against the
  current held-hand geometry and scene snapshot. Constant-orientation wrist
  poses are translated in one array, avoiding repeated matrix inversions.
  A new regression verifies756 blocked proposals cause zero IK calls. The
  remaining arm IK/transit/native guards are not removed; this necessary-subset
  screen does not certify an executable whole-scene path.

- Full-friction native hold `_11` now completes all4800 steps, no error,
  verified opposing grasp and maximum slip0.005253 mm. All records confirm
  full/patch friction reporting; palm and grasp-point native poses exactly
  match hold08. Peak summed target-contact bound0.61178 N includes0.17837 N
  friction (two fingers; not a per-finger force). Peak unwanted bound0.04346 N,
  no blade contacts or release. Headless control only; no GUI FPS or cutting
  success is inferred. Updated physics regression: **281 passed** (64.82 s).

### 2026-09-11: Normal VLM baseline packaged; Qwen3-VL H200 handoff

- Normal rebuild and portable validation complete: **14,259** records,
  train11,520 / validation1,358 / test1,381; easy11,237 / hard3,022;
  24 frozen source families16/4/4, **246** distinct target IDs168/39/39.
  136 exact representative assistant accepts support QA; not every image was
  individually reviewed. Medium/partial, ambiguous holds and dynamic episodes
  remain excluded. Existing review decisions, source images/depth and splits
  were not rewritten.
- Published `training_archives/visible_occluded_20260911_v1.zip`:
  **23,453,235,122 bytes**, SHA256
  `23e180dcb74f6daf0437414513b023d41d76f29e755aa1e2fb1638f6b268db74`.
  Normal archive validation, embedded-member SHA/CRC and publication completed.
  The external finalization driver then failed writing its receipt because
  `training_started` was specified twice; no ZIP/source bytes were lost.
  Fixed that driver and independently re-read **71,306** published members
  plus the complete ZIP checksum before writing `baseline_archive_receipt.json`.
  Original failure and recovery logs remain under
  `dataset_audits/release_20260911_agents`; no rebuild or validation waiver.
- Verified the adapter against Qwen's official fine-tuning source and grounding
  guidance. Release `images/messages` is **not a drop-in** upstream
  `image/conversations` file; the role-preserving adapter avoids converting
  system instructions into assistant supervision.
- Added explicit `qwen3_grounding_normalized_1000.v1`: both query and cut
  answer are converted to Qwen-relative coordinates, preserving continuous
  values and original RGB. Canonical dataset pixels/labels remain immutable.
  Strict inverse conversion precedes canonical evaluation. No clipping,
  coordinate-scale guessing, hidden XYZ or non-null occluded answers.
- Read-only `qwen_format_audit_v3.json` checks all14,259 frozen chat records:
  both input/output frames, train/inference prompt equality, preserved roles
  and abstention semantics; max pixel roundtrip error1.14e-13. This supplements,
  not replaces, the normal image/depth/QA validator and is not model accuracy.
- Added `sim_data/H200_RUNBOOK.md` and explicit `h200_train.py` server recipe:
  local dense8B snapshot only; actual processor/mask smoke; train-only2-step
  smoke and32-example/100-step overfit; initial4-GPU BF16 language-attention
  LoRA; original frozen validation and no test reads. One-image microbatches,
  SDPA, finite loss/gradient guards, no silent truncation or output overwrite.
  Hyperparameters and DDP behavior await real H200 verification; no throughput,
  checkpoint-reload, trained-model or simulator-action result is asserted.
- User explicitly requested no local Qwen download. An isolated dependency
  environment had been prepared, but **no Qwen model files/weights were
  downloaded and no training was started**. CLI/help and helper/adapter tests
  run without loading a processor or model. Focused data regressions53 passed.
  Independent review caught mixed pixel/normalized wording and a decoder that
  accepted an array of pairs; both now fail the corresponding regressions.

### 2026-09-11: Cutter blockage and local loading follow-up

- Native `bimanual_cut_20260911_30` re-verifies opposing left grasp, then
  rejects756 complete rigid-tool strokes before IK:556 hand/tool and200
  tool/scene conflicts, **zero IK calls**, planning1.941793 s. No right-arm
  movement, contact or cut. This is proposal-screen latency, not GUI FPS.
- Offline actual71.126 mm grasp:756 rejected (594 scene/162 intertool), no IK.
  At left tilt30 degrees, the original10 mm seam and explicit20 mm diagnostic
  seam also return zero complete corridors across756 proposals each. Both
  pass their left-only screens; this does not certify native grasp or access.
- One independently checked right-camera/left-finger pair has genuine
  2.823875 mm clearance, below the unchanged3 mm screen margin. Right bracket
  and MainStem26/27 use native convex decomposition but conservative boxes in
  the planner; their precise occupancy remains a separate geometry question.
- Native coupon `blade_loading_20260911_10` repeats k250/cap0.45/damping450
  at960 Hz with full contact/friction reports. Last valid loaded advance is
  **0.165609 mm**; at4.0375 s the full bound reaches0.500383 N and stops the
  run. Edge-normal0.303623 N, friction-magnitude0.192213 N, deepest contact
  overlap0.464915 mm. Source hashes unchanged; no seam release or grasp claim.
  This is a force-budget stop, not eligibility resets or reaching the1 mm
  overlap limit. A higher step rate alone does not solve the unchanged0.3 mm
  loading criterion against the current elastic, intact material interface.
- The existing contact/spring model has no local crack-growth law. A future
  localized deformable/cohesive interface needs actual deformation/contact,
  balanced reactions, energy/area accounting and negative controls; increasing
  force or replacing measured evidence with a timer is not a valid fix.
  Tissue coefficients remain uncalibrated. Full grasp-cut-retain-deposit and
  observation-driven VLM actions remain unqualified.

### 2026-09-11: Portable code package and native convex extraction

- Matching H200 code package from commit `127b43e`:
  `training_archives/visible_occluded_20260911_v1_code_v2.zip`,422,426 bytes,
  SHA256 `82e1e8e5e4b7d4b84a48b10bc9c2ad044ccc511adda1e9d5cb971656f7f8ff8f`.
  All135 tracked member bytes match that commit. Independently extracted
  ordinary-Python CLI/help and the full14,259-row format audit pass. This
  package includes the server recipe; older `_code.zip` lacks that recipe.
  No actual Qwen processor/model was loaded. Data ZIP and source release stay
  byte-identical; the server user owns model download and training execution.
- Added `sim_physics/cooked_geometry_probe.py`: stopped-stage, async native
  PhysX collision-representation extraction, bounded requests/cancellation,
  copied/validated convex buffers, source/settings/version/transform hashes,
  and exclusive advisory report publication. It cannot approve execution,
  change current planner geometry, or clear a protected-structure collision.
- Native `cooked_geometry_20260911_01` completes on the original47 mm-class
  fixture with full robot: right bracket/MainStem26/MainStem27 each return
  **16 native cooked convex pieces**.4 Kit service updates;32.547 s including
  startup and authoring, no play, physics step or motion. Source hashes and
  native exposed physics settings unchanged. Capture payload SHA256
  `d40c50ae2f081985faec825c5572ff723fd42af608bb813056b8ca1b63858957`.
  This confirms data extraction, not live actor equivalence or a valid path.
- The actual primary obstacle remains native-contact cutting. Cooked pieces
  permit a more precise advisory check of bracket/stem box rejections; native
  actor/query agreement is still needed before replacing planner bounds.
  No force, penetration, clearance, dwell or loaded-travel gate was relaxed.
  Full physics regression **315 passed** (54.41 s); no full-cut claim.

### 2026-09-11: Native clearance and progressive interface development

- Added explicit exact-collider capture scope (legacy three-target default
  unchanged), plus a stopped-stage native query comparison including the right
  wrist adapter. Independent inspection found that raw native polygon planes
  differ from a hull reconstructed from their vertices. The latter can miss
  native solid occupancy and must NOT approve motion. Preserve `_01`'s blocked
  result and the independent `cooked_query_planes_assessment_20260911_c.json`.
- `cooked_query_20260911_02` matches all2,328 rays using raw planes but remains
  blocked because actor parsing changes session source bookkeeping after its
  source snapshot. `_03` parses actors first, then binds/captures: all2,328 rays
  agree, maximum error1.517 micrometres, source snapshot unchanged, no play or
  physics step. Finite ray agreement is not exhaustive actor/contact equivalence.
- Added default-off `--native-static-clearance`: native overlap of the whole
  fitted right-tool box expanded by the existing1 mm margin against exact
  static collider identities. All other arm/plant/IK/transit/runtime checks
  remain. Missing actor coverage, query failure, deadlines and source/physics
  epoch changes fail closed; final validation rechecks used actor coverage.
  Native queries do not substitute a newly fitted visual/cooked vertex hull.
- Preliminary native `bimanual_cut_20260911_31`: left grasp verified,605 coarse
  pair rejections cleared,867 queries,80 static collider positive controls,
  3.178 s planning. All756 complete strokes still reject (560 hand/tool,
  196 scene), zero IK/right motion/contact/cut. Subsequent hardening adds
  invalidation subscriptions, final acceptance checks, deadline-after-call
  checks and explicit tool-only scope. Do not treat `_31` as native evidence
  of those later changes or as a successful full sequence.
- Added a single world-direction proposal adapter/CLI for controlled posture
  tests. Source target, fresh-axis projection, blade-end reserve and full path
  checks remain; the JSON cannot override a target position or safety rule.
  Native `_32` changes only left tilt10 to30 and preserves native31's longest
  right corridor. It fails opposing grasp before planning. A newly introduced
  optional-Path serialization bug prevented its report publication; its840-row
  trajectory remains intact and must not be presented as a complete report.
  Fixed central configuration serialization and added regressions. Fresh `_33`
  uses the same tilt30 with the existing allowed0.75 mm closure command; it
  publishes correctly but also fails opposing grasp at3.5 s before right motion.
- Implemented pure energy-consistent `cohesive.py` and isolated native
  `cohesive_native_probe.py`: four area-weighted facets, implicit D6 secant
  springs, measured post-step anchor separation, irreversible damage, no weld
  across failure. Compression remains separately unqualified. Material is an
  explicit uncalibrated prior, not tomato calibration. Native forces/work are
  endpoint reconstructions, not independent joint-force readbacks.
- Same material/carriage configuration at960 Hz: subcritical and damage-disabled
  controls remain undamaged; softening fully separates at3.407292 s and keeps
  zero spring stiffness on unloading. Each completes9,600 ticks without guard
  fault. Reference area4 mm2 and Gc2 J/m2 give8 microjoules; softening's
  reconstructed absorbed work is8.014777 microjoules. Subcritical displacement
  matches the300/(300+400) elastic equilibrium; maximum momentum residual
  2.87e-7 N. Softening repeats at480/1920 Hz complete without faults, but480 Hz
  fails the strict separation-time refinement comparison. No blade/plant cut
  or training eligibility follows from these coupon runs.
- Full physics regression before the single-proposal follow-up:596 passed in
  54.41 s. Later single-proposal/reporting tests run separately; final whole
  regression will be recorded after the next integration checkpoint.
- Next physical model is a separately validated local deforming material band:
  the actual6 mm plate needs an opened channel, not merely a released seam.
  Native discrete rigid cells/D6 connections retain full contact reporting;
  FEM attachment/reporting limitations do not justify faking contact forces.
  Source knife/assets, current production material/thresholds, VLM data/splits,
  reviews and H200 archives remain unchanged. No training or hardware commands.

- Checkpoint regression: **782 passed in 67.42 s**, recorded in
  `data/sim_physics/regression_20260911_physics_checkpoint2.log`. This includes
  the new source-bound proposal, report serialization, exact-identity normal
  grasp evidence and isolated material-band geometry tests; no native cut claim.
- Found a grasp identity mismatch: the planner permits the selected shaft and
  its immediate connected detached-side segments, while the tensor verifier
  filters one body only. Native trials32/33 report contacts on neighbouring
  shaft colliders that the verifier cannot count. Added pure `shaft_grasp.py`
  and a normal-only observer hook: exact inner pad/StemCollider identity,
  live-frame geometry, unchanged 20 mN opposition threshold, no leaves,
  protected support, friction-only evidence, or disconnected neighbours.
  Live adapter and independent tensor sign/load cross-check remain pending.
- Independent rate analysis is saved as
  `data/sim_physics/cohesive_rate_assessment_20260911_01.json`. At960 Hz,
  softening absorbed-work residual is14.777 nJ (0.1847% of8 microjoules),
  separation timing passes the declared comparison with1920 Hz. At480 Hz,
  timing misses it. These are restricted isolated-coupon findings, not
  full-plant spatial/material calibration.

### 2026-09-11: Native grasp measurements and unilateral interface controls

- Integrated the exact-identity `shaft_grasp_native.py` adapter with the full
  robot probe. It snapshots shaft/pad geometry and live joint identities,
  invalidates changed bindings, and compares selected contacts with the native
  tensor stream. Contact faults remain latched across steps; empty exception
  strings now also stop the controller. No weld, force-limit change or timed cut.
- Native hold controls34/35 reproduce the known left posture with the right
  knife parked. Control34 reconciles selected callback/tensor vectors through
  712 steps (maximum error1.46e-8 N), then rejects a negative point impulse.
  Control35 captures that first value as -9.44e-21 N.s. Added a fixed1e-15 N.s
  signed-zero bound with zero, never positive, grasp contribution. Control36
  reaches step719 and rejects a larger -4.20e-8 N.s impulse; it is NOT roundoff.
- Added hold-only `--diagnostic-grasp-contacts`, prohibited during cutting.
  It drains only the faulted callback step, reads both selected tensor buffers,
  then always raises before another control step. Native control37 captures all
  12 delivered rows with no truncation. At the identical point/normal/separation,
  the callback impulse is negative but the tensor scalar is positive magnitude
  (10.083 microNewtons). Thus the old signed-vector tensor comparison is not
  valid for negative point impulses. Correct signed physical support versus
  unsigned sensor reconciliation is under investigation; no grasp/cut pass.
- Added isolated `material_band.py` geometry/mass/interface authoring and
  `cohesive_unilateral.py` bounded normal soft-limit coupling. Neither is
  integrated into the production plant or a claim of calibrated tissue.
  Compression contacts remain active; full failure removes the normal limit
  APIs/properties instead of turning zero stiffness into a hard stop.
- `cohesive_closing_intact_20260911_01` is DISQUALIFIED: the first draft's
  active negative-infinite bound caused four native `limit invalid` errors.
  Replaced it with a finite -2 mm bound outside the guarded +/-1 mm domain.
  This is a bounded-domain unilateral approximation, not an infinite free axis.
  Added passive bounded `native_errors.py` observation before author/reset and
  before/after every solve; delivered native physics errors now stop the test.
- Corrected native intact control02 completes10,560 steps at960 Hz, all three
  hold tails checked, zero damage/dissipation, zero observed native errors.
  Measured settled opening passes the predeclared25/50 micrometre +/-5% bands.
  Maximum reconstructed momentum residual8.05e-7 N (limit4.1e-4 N);
  maximum cumulative energy residual0.418 nJ (intact limit25 nJ).
  `bounded_protocol_accepted=true` is only this coupon's acceptance, NOT
  native material calibration, blade passage, plant cutting or robot success.
- Focused regression after integration:392 passed in21.51 s under Isaac Python.
  Evidence directories are under `data/sim_physics/`; failed runs are preserved.
  Next: partial/final failure closing controls, timestep refinement, corrected
  signed grasp evidence, then actual-blade material-band and full-robot trials.
  VLM records, frozen splits, reviews, training archives and hardware untouched.
- The three corrected closing cases now each pass at both960 and1920 Hz:
  intact02/03, damaged01/02, failed01/02. Each finer run completes21,120 steps;
  dependency hashes remain unchanged, native error observers report no errors.
  Partial damage0.799976 is retained; fully separated cases retain zero interface
  stiffness/free normal axes while allowing native compression contact.
  Same-time failed-case displacement differs by at most0.545 micrometres,
  damage by0.004114, per-facet reconstructed normal force by34.24 microNewtons.
  Complete separation occurs at2.693750/2.692188 s respectively; both dissipate
  the specified8 microjoules. These measured refinement differences do not
  constitute continuum calibration or actual-knife cutting evidence.

### 2026-09-11: Signed grasp evidence verified in the supplied greenhouse

- Corrected the callback/tensor contract explicitly, not by widening force
  tolerances or fitting signs. `ShaftGraspEvidence` has an opt-in signed-normal
  mode; signed contributions sum before qualification. Each current native pad
  must support >=20 mN compressively, in addition to existing opposition,
  geometry, topology, dwell and slip checks. Negative-only loads cannot grasp.
- `ShaftGraspNative` independently matches every selected callback/tensor row
  by exact collider identities, point, normal, separation and magnitude.
  Unsigned magnitudes only verify sensor integrity; they never supply physical
  grip force. Missing, duplicate/ambiguous or mismatched rows fail closed.
  The measured contract is explicit and hash-bound; it is not a universal API
  claim. The strict diagnostic mode remains separately reproducible.
- Trials38/39 exposed float32 norm underflow, not mechanical overload:
  one8.77e-38 N.s vector became zero, and a9.44e-21 N.s nonzero norm lost
  relative accuracy because its squared components were subnormal. Added
  bounded zero-only/component and positive squared-subnormal arithmetic checks,
  preserving the signed physical input. Boundary, counterfeit-zero, negative
  grasp and nonfinite-diagnostic regressions cover these paths.
- Native `bimanual_hold_control_20260911_40` completes4,800 steps /20 s with
  `error=null`, bounded/completed/left_grasp_verified all true, unchanged source
  assets, and maximum measured slip5.252520e-6 m (0.005253 mm). Right-arm
  commands stay parked; zero blade contacts/cuts. Its generic whole-cut state
  remains failed because the deliberately omitted cutting gates are false;
  inspect the individual hold gates, not that state, for this negative control.
- Native41 changes left tilt10 to30 with0.5 mm commanded pad compression;
  correct row matching passes, but one pad has only6.4 mN support. It stops
  before right motion. Native42 uses the existing0.75 mm closure setting and
  verifies the left grasp. The single historical world-direction proposal then
  fails rigid clearance at its first sample: left palm versus right camera
  bracket, required margin3 mm, measured conservative clearance approximately
  zero. No IK execution, blade contact, release or full-cut success is claimed.
- Full physics regression excluding the still-isolated new native-band probe:
  **1,235 passed in88.90 s**, saved in
  `data/sim_physics/regression_20260911_grasp_measurements.log`.
  Independent new band-probe tests:24 passed. Native band rest/elastic/contact
  controls are the next stage; no production material change or training data.
