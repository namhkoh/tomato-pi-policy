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

## Clear-target overnight collection checkpoint, 2026-09-15

- User requested NEW clearer native captures, review/annotation and a training
  ZIP, not repackaging the47 old easy-image candidates. No training is started.
- Pilot `clear_capture_20260915_pilot_v3/job_002`:15 captures, exit0,
  independent native audit passed,8 strict clear-task survivors from3 targets.
  Original robot-head RGB for all8 was visually inspected; attributed assistant
  reviews are under `data/sim_data/dataset_reviews/clear_native_pilot_20260915_v1`.
  This small pilot is not a training release or an independent human evaluation.
- Serial overnight campaign `clear_capture_20260915_overnight_v1`:24 reserved
  source families,297 eligible height-band targets, at most12 captured views per
  target (3564 theoretical maximum, NOT an accepted-data count). One native
  renderer at a time; disk/memory reserve checks and real worker-exit receipts.
  Verified no-screened-viewpoint exits can be recorded and skipped; errors stop
  without automatic retry. First family exited0 with7 audited captures.
- Whole greenhouse context, camera mount/optics,848x408 original RGB and native
  optical-Z remain. Uniform diffuse scene lighting is a declared new capture
  condition; no RGB retouching or target-specific spotlight.
- Added opt-in assistant-reviewed experimental release policy, explicit GUI
  attribution and contract consistency checks. Human-heldout remains default;
  assistant mode does not bypass clarity, coverage, holds or split integrity.
- Regression:665 tests plus47 subtests passed (43.83s), log
  `data/sim_data/clear_regression_20260915_v3.log`.
- Follow-up intake reads completed audit/exit/hash receipts only and creates
  separate per-family engineering pools, strict clear drafts and static review
  pages; it cannot launch Isaac, approve labels, finalize a release or train.
  Run: `python -m sim_data.clear_collection_intake --campaign <campaign>
  --output <new-intake-directory>`. It also builds an aggregate draft only after
  successful campaign completion. Errors and incomplete receipts fail safely.
- First2 completed families:62 native captures,41 strict clear candidates from
  8 targets. Required visual sampling is complete for these batches:7/7 seed101
  frames and11/34 seed103 frames, explicit assistant records outside Git. Every
  exported frame additionally has native numerical/identity/depth checks; do
  not describe uninspected training views as manually reviewed.
- Follow-up regression:669 tests plus47 subtests passed (44.67s), log
  `data/sim_data/clear_regression_20260915_v4.log`.
- ZIP handoff now generates a root `DATASET_CARD.md` from the actual manifest:
  counts, review attribution, target-conditioned2D task, native depth sidecars,
  shared-background caveat and unsupported execution claims.21 focused tests
  passed (5.34s). No final training ZIP has been generated yet.
- Collection diagnostic: first6 completed train families yielded78 strict clear
  candidates from5 families; seed23 yielded0. All78 received explicit per-image
  assistant visual assessment. Source seed23 rejected3048 proposals for image
  scale,374 for possible robot/scene overlap and33 for framing; only1 pose passed
  geometry and its rendered frame was not clear. No threshold was weakened.
- Added separately selected `training_plan --opposite-aisle` static snapshots:
  negative-X root and yaw within30 degrees of0, with identical camera optics,
  torso range, floor, joint-limit and scene-overlap checks. This is NOT a validated
  base trajectory across the gutter. Original positive-side proposal digest is
  unchanged; native negative-side rendering still awaits a pilot. Plan:
  `data/sim_data/collection_plans/clear_capture_20260915_opposite_pilot_v1`.
- A CPU-only deep-crouch probe was NOT accepted as certification: raw-asset
  capsule self-screen flags overlaps even at its baseline. No deeper torso
  range or self-collision-filter changes were made.
- Regression after the opt-in aisle extension:671 tests plus47 subtests passed
  (43.79s), `data/sim_data/clear_regression_20260915_v5.log`.
- Pending: complete collection, per-image QA, coverage/diversity assessment,
  review finalization, portable validation and verified ZIP. Do not claim the
  final dataset exists from this checkpoint. Physics implementation unchanged.

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

### 2026-09-11: Cutting planner and material-feasibility checkpoint

- Native43 stopped on an invalid diagnostic proposal JSON provenance shape,
  before the physical trial. Corrected that input for native44; no guard or
  simulator implementation was changed to accept it. Native44 verifies the
  farther 71.126 mm centre grasp (75 mm requested), but the native expanded
  camera-body query overlaps protected MainStem26 at the first tool pose.
- Native45 tests the full bounded orientation grid at the same grasp. One
  endpoint converges (45 degrees, normal +1, wing0, plane tilt -10), but the
  native query epoch expires during the remaining search. Its later reported
  MainStem27 overlap is a conservative fallback, NOT proof of native collision.
- Native46 isolates that endpoint in a source-bound proposal. Planning takes
  2.024 s, 293 native queries, 213 coarse rejections cleared, no query errors.
  The full tool corridor and endpoint pass; the arm stroke stops at -6.506 mm
  because right upper-arm link1 approaches the held branch's Leaf_004 within
  the unchanged 1 mm margin. No right motion, blade release or cut is claimed.
- Fixed planner ordering: immediately validate each feasible endpoint's full
  sampled stroke and transit before searching more orientations. This is first
  fully screened feasibility, not shortest-path optimization. All scene, self,
  tool, inter-arm, IK and final native-epoch checks remain mandatory. Native
  query invalidation now raises "unavailable; collision not determined" instead
  of continuing to report timeout fallbacks as ordinary collision failures.
- Added opt-in `--right-ik-fixed-joint INDEX DEGREES`: exact-URDF redundant IK
  through the endpoint AND whole stroke, with unchanged parked start and
  separately checked transit. The initial +0.5 degree joint1 proposal is
  rejected offline: it clears the leaf but reaches wrist joint5's limit at
  sample26. This option is not evidence that another posture is feasible.
- Native `cohesive_mixed_20260911_01` completes9,600 steps at960 Hz with
  shear/opening ratio1, no guard fault; dissipation8 microjoules and reconstructed
  drive absorption8.021885 microjoules. This remains the original four-facet,
  zero-gravity toy coupon, not a blade, full-section, unilateral mixed-mode,
  calibrated material, current plant cut or dynamic training episode.
- The separate32-cell `material_band_native_probe` remains UNQUALIFIED.
  `material_band_rest_20260911_01` stops after one step on force-balance limits:
  max cell residual0.691 mN, global axial residual1.105 mN. Motion is small
  (0.855 nm max displacement) but does not establish correct internal forces.
  A read-only float32-storage uncertainty analysis cannot account for the whole
  global residual; no baseline subtraction, broad collision filtering or
  tolerance relaxation was applied. The native D6 force readback remains absent.
- Added inactive-only `seam_interface.py`:24 area/moment-weighted connections
  between existing bodies, with no new cells or body/source modifications.
  It requires pre-start authoring, explicit material, active bodies and unloaded
  initial states; checks float32-rounded anchor moments. Existing welds are
  inventoried but NOT removed; caller must audit all load paths before activation.
  This is an alternative reduced-order prescribed fracture plane, not a sharp
  knife continuum model or production integration.
- Crucial feasibility result: native44 distal mass is8.875602 g including six
  0.5 g leaf priors. Gravity plus reconstructed body-local COMs gives14.007 mN.m
  bending about the seam. A3 mm circular10 kPa toy interface reaches first
  tensile damage at only0.212 mN.m (roughly66 times below this demand). This is
  gravity loading, NOT measured seam reaction; gripper, contact and inertia are
  excluded. Native COMs were not logged; reconstruction matches native mass and
  inertia. `seam_loads.py` makes this explicit advisory calculation testable,
  without granting activation/constitutive-success permission.
- Do NOT port toy strength into the loaded plant, or weaken the seam only when
  a knife touches. Strength, stiffness, fracture work and actual blade bevel
  need a consistent physical basis. Published tomato-petiole bending data also
  concern specific cultivars/growth stages and cannot directly calibrate this
  fixture: [Fukushima et al., 2020](https://www.jstage.jst.go.jp/article/jsamfe/82/4/82_347/_pdf/-char/ja).
- Full CPU/USD regression: **1,412 passed in89.80 s**, logged in
  `data/sim_physics/regression_20260911_incremental_cut_planning.log`.
  No current-greenhouse cut/retention/deposit success; no training records,
  reviews, splits, model downloads or physical robot commands changed.

### 2026-09-11: Native right-stroke clearance and explicit engineering-model correction

- Native47 uses the same approved knife mount, farther left grasp and station,
  with an explicit -12 degree blade-plane proposal and right joint1=-1.5 degrees.
  The proposal-only tilt envelope is now +/-15 degrees; the default grid remains
  0,+/-10. This remains inside the unchanged measured transverse-direction gate
  (sin15=0.259 <0.3). No physical force, joint-limit or collision margin changes.
  Grasp verifies, but the near-final stroke pose is rejected against the coarse
  MainStem27 box by right forearm link5; this is not proof of native contact.
- Added conservative native refinement for the exact restored right-arm capsule
  paths only. The query box contains the WHOLE capsule, both spherical ends,
  and the existing >=1 mm margin. Only a native miss can clear a coarse static
  box rejection; dynamic foliage, source triangles and unsupported robot
  shapes keep their original checks. Positive collider coverage and single
  unchanged-physics epoch/final validation remain mandatory. Independent review
  agreed with the enclosure argument; 91 focused CPU tests passed in1.29 s.
- Native48 passes the sampled stroke with this refinement, then exhausts the
  8 s synchronous planning budget in transit: total planning8.758 s,408 queries,
  327 coarse rejections cleared,6 capsule queries. No right motion/cut occurs.
  It correctly reports unavailable queries rather than a fabricated collision.
  Explicit diagnostic --native-static-planning-seconds now permits up to60 s;
  default8 s and20,000 query cap remain. Physics must stay frozen in the same
  audited epoch, including cache/final validation. This is not a latency claim.
  Budget/configuration/planner/native-bound tests:107 passed in1.41 s.
- Contact audit found a real release-authorization bug: unsigned blade impulse
  magnitudes could turn tensile or cancelling contributions into cutting load.
  Correction preserves raw collider order and separates signed resistance
  from non-cancelling load upper bounds. A successful cut must use the corrected
  path; prior tests do not establish signed blade resistance in a native cut.
- Explicit model decision: the intact FixedJoint has no tissue-opening law.
  Requiring0.3 mm blade-centre travel while intact is not a constitutive fracture
  criterion and can reward compliance/penetration instead. Implementing opt-in
  signed_edge_load_brittle_seam_v1: unchanged >=0.2 N compressive edge resistance
  for25 ms, verified left grasp, exact admissible seam and all existing guards.
  Pre-release travel stays measured but is NOT a failure prerequisite in this
  strength-only approximation. Legacy mode remains separately named/default.
  This does NOT activate the unqualified cohesive prototypes, calibrate tomato
  tissue, invent fracture energy, or count joint release as a complete robot task.
- Mixed-mode toy coupon repeat cohesive_mixed_20260911_02 at1920 Hz completed
  19,200 steps; applied full damage at3.4390625 s,8 microjoules dissipation,
  8.010948 microjoules reconstructed drive absorption. Compared with960 Hz:
  max same-time jump difference0.952 micrometres and damage difference0.006950.
  Still a zero-gravity four-facet diagnostic, not the loaded plant/knife model.
- Pending native qualification: signed contact trigger, safe right approach,
  actual release, measured withdrawal and retained detached branch. Deposit,
  calibrated tissue cutting and dynamic training examples remain unverified.

- Follow-up native49: full sampled cut stroke and bounded bidirectional transit
  pass; planning11.0807 s,405 transit collision checks. Physical right approach
  and blade loading execute while left bilateral contact remains established.
  At11.8667 s the new explicitly named engineering model releases the10 mm seam:
  seven qualifying steps/29.1667 ms, signed resistance0.214771..0.309924 N,
  non-cancelling normal-plus-friction upper0.357566 N; left slip0.03547 mm.
  Actual pre-release loaded travel0.15142 mm is reported, not treated as tissue
  work or as a pass of the legacy0.3 mm criterion. No physical tissue-cut claim.
- Native49 is still a FAILURE overall: stopped at12.9083 s when left slip
  reaches3.01165 mm. Opposing contact persists, but retention is not qualified.
  Released leaves also contact right arm2/5 during withdrawal (not reported as
  zero-contact or safe complete task). Do not use this run as a training success.
- Removed the unconditional release+1 s left pull. Withdrawal elapsed time is
  now a diagnostic only, with completion false until measured pose/current
  clearance is integrated. New pure withdrawal_evidence helper requires
  <0.5 mm/<0.005 rad and matching episode/step clearance, no elapsed-time credit;
  68 tests and independent review pass. No path/tissue certification follows.
- Signed blade pipeline/model focused suite:474 CPU tests passed including68
  new cases; main cross-check211 passed in12.18 s. Raw collider/normal/impulse
  rows remain available, with tensile/cancellation/friction/invalid-provenance
  negative controls. Model identity survives reset and is explicit in reports.
- Full CPU/USD physics regression after these changes: **1,584 passed in93.36 s**,
  saved in data/sim_physics/regression_20260911_signed_blade_release.log.
  Current signed blade release is established only for this privileged test
  fixture; reliable post-release retention remains the next native blocker.

### 2026-09-11: Retention telemetry exposes a spring/contact coupling blocker

- Dataset handoff remains `data/sim_data/training_archives/visible_occluded_20260911_v1.zip`
  (23,453,235,122 bytes, 14,259 RGB grounding/visibility-abstention examples).
  Paired code archive: `visible_occluded_20260911_v1_code_v2.zip`. These are
  static image supervision, NOT manipulation trajectories. No records, review
  decisions, splits, collection jobs or H200 training were changed here.
- Native50 rejects tighter1 mm jaw compression before approach: finger2 would
  overlap Leaf000. The0.75 mm closure remains; no margin was relaxed.
- Added opt-in `--diagnostic-grasp-dynamics`: same-step native body/finger poses,
  signed shaft contacts and body-local coordinates, q/qdot, commanded targets,
  gravity compensation and remaining drive caps. Reconstructed PD demand is
  explicitly NOT measured actuator effort. Local contact coordinates are NOT
  tracked material anchors. Diagnostic data never grant training approval.
- Native51 repeats49 exactly: release11.8667 s and slip failure12.9083 s.
  The selected segment rotates23.12 degrees, contact points migrate toward
  opposite shaft ends, and the lower finger's reconstructed demand exceeds
  its0.18624 N residual drive cap after gravity compensation. This establishes
  a capacity hypothesis, not measured hardware saturation or calibrated force.
- Explicit0.8 N left-actuator engineering prior added (default0.5 remains).
  Gravity stays INSIDE the total budget; right fingers remain0.5 N. BOTH sparse
  modes now independently reject >=0.5 N total normal-plus-friction contact
  upper bound at EACH left finger, counting ALL objects without cancellation.
  Missing native patch-friction reporting fails closed before cutting. This
  is a post-step guard, not proof of zero overshoot or a tissue-damage threshold.
- Native52 uses0.8 N, otherwise the same physical closure/path. Release remains
 11.8667 s; held branch persists longer, but the trial FAILS at14.3625 s:
  finger2 contact jumps0.3621 to0.5935 N (normal alone0.5262 N), slip1.589 mm.
  Reconstructed PD is below its residual0.48624 N limit at this failure.
  Right-arm leaf contacts persist. Retention gate true over the observed tail
  does NOT make the full sequence successful or contact-free.
- Independent native-frame reconstruction confirms the deeper deformation:
  nominal elastic-energy proxy0.00689 J at release rises to~2.50 J just before
  failure; Joint003 crosses the shortest rotation-vector boundary near pi.
  The resulting coordinate wrap changes the linear spring term sharply.
  These are NOT measured cutting work or a verified total work balance.
  Root-world-roll contamination and joint-order errors were checked and do
  not explain the recorded angles. Whole-body frame readback agrees.
- `ImplicitJointSprings` omits unknown contacts in its coupled prediction.
  Under sustained contact this can substantially reduce effective elastic
  response; the previously qualified no-contact test does not qualify this
  regime. Lagged measured contact forces alone can recover stiffness very
  slowly and are not being presented as an adequate fix. Next: qualify native
  contact-coupled spring behavior in a small held/free-root diagnostic before
  choosing another full-robot controller change. No artificial grasp weld.
- `withdrawal_native_check.py` now integrates final post-fetch measured park
  pose and current full robot/plant bounds with a fresh validated native static
  epoch. It is an ENDPOINT check, not swept-path certification. Native52 never
  reaches it. Fixed a separate post-fetch timestamp error: reversal now latches
  the last sent stroke fraction, avoiding one extra forward command increment.
  It still does not guarantee instantaneous braking or a measured-start path.
- Focused pre52 regression273 passed; diagnostics and guard received independent
  review. Full CPU/USD regression: **1,797 passed in94.98 s**, recorded in
  `data/sim_physics/regression_20260911_retention_guards.log`. Small native
  spring/contact qualification follows; CPU tests do not qualify those physics.
  Reliable cut-retain-withdraw, deposit and calibrated tissue cutting remain
  incomplete. Existing failed evidence is preserved, not relabeled successful.

### 2026-09-11: Reduced spring/contact reproducer (not a cutting success)

- Added standalone three-link/two-revolute contact coupon using source52
  Segment001..003 masses/inertias and Joint002/003 Y-axis stiffness/damping.
  Six fixed compliant pads create internal bending; no root weld, body-force
  injection or moved runtime poses. This elementary geometry is NOT the full
  spherical-joint plant. Same-step signed normal/friction contacts are retained.
  Equilibrium requires contact moments, spring response, low velocity and net
  wrench balance; visually static output is insufficient.
- Initialization failures01..03 are preserved, not physics results: Kit adds
  render/camera prims to a new context; SimulationManager warm-up advances two
  solves; raw attachment defers articulation insertion until a first solve.
  The runner now authors an owned empty anonymous stage, attaches it explicitly,
  and runs one REPORTED1 microsecond native bootstrap for BOTH comparison modes.
  It preserves bootstrap contacts and initial q/qdot, verifies the original
  0.005 rad strain within unchanged tolerance, and then owns every simulate/fetch.
  No hidden reset, zeroing or state override. Native-error monitoring remains
  active through owned cleanup; final errors cannot be hidden by a pass flag.
- Valid3-second runs, each720 recorded steps: native PGS16/4 (`...native...04`),
  PGS32/0 (`...native...05`), old predictor PGS16/4 (`...implicit...06`), and
  native TGS16/4 (`...native...07`) all fail load/velocity qualification.
  Max static moment residuals respectively3.4604/3.8093/3.2847/3.9171 mN.m;
  native joint-velocity RMS0.3224/0.2009/0.3430/0.9048 rad/s. Net contact wrench
  balance and pad coverage pass. No plant or robot result follows from these.
- Native04..06 positions change <0.8 microradian over the last0.5 s, while
  reported joint/body velocities remain substantial. Independent projected
  body-angular velocities agree with native joint velocities. Public PhysX
  source has distinct saved position-integration and final carried velocities;
  even requested PGS0 velocity iterations has a mandatory writeback solve.
  These sources explain a possible mechanism, not the exact installed defect:
  [PGS solver](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/lowleveldynamics/src/DySolverControl.cpp),
  [articulation integration](https://github.com/NVIDIA-Omniverse/PhysX/blob/main/physx/source/lowleveldynamics/src/DyFeatherstoneForwardDynamic.cpp).
  Finite differences must not silently replace carried velocities in damping.
- Test08 investigates [PhysX issue498](https://github.com/NVIDIA-Omniverse/PhysX/issues/498):
  native legacy joint-friction readback is already0, and setting it explicitly
  to0 reproduces04's result exactly. The unintended-bearing-friction hypothesis
  is therefore NOT supported here. Finger/pad contact friction was unchanged.
- Helper/runner CPU tests75 pass; no complete native contact-spring pass yet.
  Next isolated comparison: paired off-axis native linear springs with the
  same small-angle angular stiffness/damping, no added bodies or grasp weld.
  This must pass force/motion checks before any greenhouse integration. No
  physical robot, model training, data collection or dataset approvals changed.

### 2026-09-11: Section-spring and maximal-coordinate controls (still not robot qualification)

- Added off-axis native linear D6 section springs for the SMALL planar coupon:
  two connectors per joint, each k=K/(2r^2), c=C/(2r^2), only transZ driven.
  Rest-local anchors preserve initial strain; original angular drives are zero
  before parsing. Ideal finite-angle energy is K*sin(theta)^2/2, not a globally
  linear beam or calibrated plant model. Pose-derived fiber energy includes
  central-anchor error. External native gains remain USD-verified only.
- Native09, articulated section springs with pads, fails: 3.2521 mN.m static
  moment residual and 0.3444 rad/s native joint-velocity RMS. This alternative
  was NOT integrated into the greenhouse. Both native friction representations
  are zero. Native10, the contact-FREE counterpart, grows from approximately
  7.97 microjoules initial fiber energy to 509.9 microjoules at step14; q2
  reaches +0.057865 rad while carried qdot2 is -3.3685 rad/s. The small-angle
  guard stops the run. Independent body-pose/angular-velocity checks agree
  with the respective joint readings: no joint-order explanation was found.
- Native11, FREE original angular drives, passes the existing 720-step tail
  tests: velocity RMS 2.28e-14 rad/s. This narrows the external-constraint
  hypothesis, but does not establish an exact installed PhysX implementation
  defect. That historical report predates the new whole-run energy gate.
- Added an explicit maximal-coordinate counterpart: same rigid-body masses,
  inertias, rest frames, pads and fibers; external revolutes, no articulation.
  Native rigid-body paths/masses/inertias/COM/initial frames are verified.
  Derived joint angles and projected relative angular velocities are labelled
  as body-derived, never fabricated articulation readbacks. No root weld,
  runtime pose/velocity writes, mass inflation or contact-filter workaround.
- Native12 preserves an initialization rejection: even a reported 1 us solve
  shifts the initial angles by 9--10 microradians, beyond the unchanged
  tolerance. Rigid bodies can instead be bound BEFORE any solve. Native13
  uses that explicit zero-bootstrap path and passes FREE 720-step checks,
  including whole-run modeled kinetic-plus-fiber energy. Initial reference
  7.9714 microjoules; maximum recorded energy 6.4229 microjoules; largest rise
  above a prior minimum 2.75e-14 J. This is a small free-motion result only.
- Free acceptance now requires the complete contiguous run, not merely a
  settled tail, and rejects energy growth beyond a disclosed allowance of
  1 nJ + 1e-4 times initial energy. Kinetic energy uses actual world body
  velocities and body-frame inertia, not finite-difference substitution.
  The runner also retains whole-run angle maxima and validates new native
  telemetry before strict JSON serialization.
- Native14, maximal section springs HELD by the same pads under PGS16/4,
  fails contact qualification: anchor error, 2.8622 mN.m moment residual and
  0.19435 rad/s velocity RMS. Thus free recovery alone is insufficient. A
  matched TGS maximal contact test follows; no complete cut-retain-withdraw
  or deposit result is claimed. Dataset ZIPs/labels/splits remain unchanged.
- Native15, the matched maximal HELD TGS16/4 case, also fails: 2.8349 mN.m
  residual and 0.56754 rad/s native velocity RMS, with failed anchor tolerance.
  The free success therefore does not justify production integration. Next
  diagnostic isolates compliant versus rigid contacts with the same spring,
  body and collision geometry; this is not removal of production compliance.
- Checkpoint regression: **1,955 CPU/USD tests passed in 100.75 s**, logged in
  `data/sim_physics/regression_20260911_contact_spring_coupons.log`. This code
  regression is separate from, and cannot override, native physical failures.
- Native16 raises only the maximal section numerical iterations to128/32 on
  all rigid bodies. Error improves but still fails (2.4245 mN.m residual,
  0.0960 rad/s velocity RMS, anchor gate false). No unlimited iteration sweep.
- Added a direct angular-D6 maximal control: two generic joints, rotY driven,
  other axes locked, original SI-to-USD conversion, no fibers/predictor.
  Native17 FREE passes720 steps including whole-run modeled energy. Native18
  HELD fails again: 2.8574 mN.m residual and0.19646 rad/s velocity RMS.
  No maximal-coordinate replacement is qualified for production contacts.
- A rigid-contact diagnostic option was added to the helper but NOT run:
  fixed5.50 mm pad clearance cannot admit the5.60 mm shaft without deformation.
  Its held qualification is explicitly ineligible; it cannot become a pass
  through penetration or escape. Production compliant contacts are unchanged.
- Independent full52 trace audit finds elastic-coordinate energy growth BEFORE
  right-arm contact:6.89 mJ at release,20.30 mJ at+50 ms,31.71 mJ at+70.8 ms;
  first unintended arm contact is+75 ms. Right-arm withdrawal needs fresh-path
  validation, but is not supported as the initial cause of that growth. These
  energies remain proxies, not a complete measured work/energy balance.
- User fidelity requirement: preserve supplied greenhouse surroundings, plant
  meshes/materials, robot and camera views. Simplified coupons are diagnostic
  only, never demonstration/training replacements. Existing full-scene trials
  retain all75 gutter structures and the package preview's3 planted rows
  (144 plant positions including the detailed target). Visual gutter batching
  references original geometry/materials and preserves collision proxies.
  Nearby context plants are currently static colliders, not deformable tissue;
  only the selected petiole has compliant dynamics. Do not hide that limitation
  or remove nearby context to obtain a grasp/cut success.

### 2026-09-11: Contact-aware spring prediction qualification (diagnostic only)

- Native19 splits explicit original `-K*q` stiffness from unchanged native
  damping. FREE720 steps pass energy/recovery. HELD native20 instead reaches
  |q|=0.06761 rad and 1.839 N body contact upper bound at step3; the unchanged
  guard stops it. This split is NOT integrated in the greenhouse. A purely
  mathematical passivity condition does not qualify its native realization.
- Native21 records pre/post-step floating mass matrices, world COM Jacobians,
  velocities and signed contact separations. Native `J*v` agrees with body
  velocities, COM origins are checked and mass is SPD without regularization.
  The original HELD failure is reproduced over240 steps: 3.4604 mN.m moment
  residual and0.32247 rad/s velocity RMS. No native error is reported. These
  snapshots distinguish contact-generation geometry from post-fetch poses.
- Added `contact_coupled_prediction.py` pure frozen-M/J backward-Euler solve
  including finite pad contact stiffness/damping in the prediction. It retains
  all six free-root coordinates, but proposes ONLY the original plant spring
  effort. Native contacts/friction remain authoritative; their predicted
  forces are never replayed. Finite-face manifold matching, bounded active-set
  convergence, explicit material-law choice and contact-epoch checks fail closed.
  Normal-only prediction omits friction and is not declared native-law parity.
- `contact_coupled_native.py` and the standalone runner enforce original K/C,
  zero native angular drives after the disclosed bootstrap, unchanged drive
  caps, contiguous fresh snapshots and only two joint-effort submissions. No
  root actuation, pose/velocity writes, gain fitting or clipping. All changes
  remain in the small diagnostic: no production spring replacement yet.
- Native22 FREE coupled prediction passes720 steps and whole-run energy:
  initial modeled7.97146 microjoules, maximum recorded6.01001 microjoules,
  no positive rise above a prior minimum; velocity RMS6.75e-15 rad/s. This is
  not a held-grasp or tissue-cut result. The matching HELD native23 test follows.
- Focused helper/predictor/submission/runner regression:201 tests passed in
  1.88 s. No dataset, split, review, training or hardware command was changed.
  Reliable full-greenhouse cut-retain-withdraw and deposit remain incomplete.
- Native23 HELD completes720 steps but fails equilibrium/velocity gates:
  2.6624 mN.m static residual,0.16933 rad/s velocity RMS. Native finite-contact
  geometry matching is now corrected to source-seeded persistent capsule
  segment anchors rather than freshly re-clipped pad edges. Native21 matches
  240/240 frames and2880 normal rows: maximum point19.43 nm, separation11.08 nm,
  normal-vector error1.2211e-6; original tolerances unchanged. This correction
  does not fix the mismatched native force response. Focused tests203 passed.
- Independent native23 audit: predicted versus reported normal force differs
  by up to42.87 mN in the tail; using actual native final velocities in the
  assumed contact law still leaves53.69 mN discrepancy. Generalized joint
  momentum balance using actual impulses and submitted spring effort agrees
  within4.91 nN.m. Therefore no native-law equivalence is claimed.
- Native24 halves the ORIGINAL native control timestep to480 Hz for1 s.
  It still fails:3.3932 mN.m residual and0.14590 rad/s velocity RMS. This is
  a timestep sensitivity result, not convergence or a coupled-model pass.
- Native25 opts into NVIDIA's documented `solveArticulationContactLast` scene
  ordering. USD is read back true (previous false); the static-pad result is
  identical to21. A shared stopped-scene `solver_configuration.py` helper adds
  the same opt-in to the full-robot benchmark, with after-reset USD checks.
  Defaults, material values, collision checks and force/slip limits are unchanged.
  Full53 compares52 with only this ordering option: dynamic robot-finger
  contacts differ from the static-pad coupon. Effective native flag readback
  is not independently available; authored USD is labelled accordingly.
- CPU/USD physics regression before the full-benchmark option:2089 passed in
  96.32 s (`regression_20260911_contact_prediction.log`). Option/refactor tests:
  36 passed. Added1920 Hz as a bounded diagnostic timestep choice, within the
  helper's existing dt range, for a convergence check if needed. No production
  timestep is changed. Finite contact iteration convergence and differing
  native integration/writeback phases remain hypotheses under investigation.
- Full53 retains all source assets (verified hashes unchanged). It again
  releases the seam at11.8667 s while left contact is bilateral, then stops
  at14.4333 s: per-finger all-contact upper bound0.61401 N exceeds0.5 N.
  Maximum slip1.7014 mm. Contact-last is NOT a fix and stays opt-in. Raw
  `native_retention` subset gate being true does not override failed bounded,
  completed and withdrawal gates. No full cut-retain-withdraw success.
- Native26 ORIGINAL native drives at1920 Hz still fail:2.4943 mN.m moment
  residual,0.03812 rad/s velocity RMS. Native27 contact-aware normal prediction
  at1920 Hz improves to0.36175 mN.m and0.004252 rad/s; all existing tail gates
  EXCEPT stiffness equilibrium pass. Maximum |q|0.005034 rad. Friction is still
  absent from this predictor, and native27 is explicitly failed/unqualified.
  No1920 Hz greenhouse default or production predictor integration follows.
- Public PhysX5.9 source review (commit517a0073715120e114ee055b63b26c95e00d9039)
  identifies different preparation, iterative impulse and final writeback
  states. An observed positive-gap normal impulse is not explained by merely
  switching clamped/signed Kelvin-Voigt modes. The isolated row impulse factor
  alone is NOT a coupled convergence-rate bound. Frozen native23 normal-only
  matrix analysis motivates timestep refinement; it omits friction, drives
  and native phase ordering and is not measured solver equivalence.
- Review found cleanup exceptions could prevent evidence export. The runner
  now attempts every owned cleanup, records failures, marks the run failed
  and still exports its trace/receipt. Five injected failure cases cover it.
  Full CPU/USD checkpoint:2095 passed in94.88 s, recorded in
  `regression_20260911_contact_prediction_checkpoint.log`. A separate measured-
  start withdrawal helper is next; no runtime pose/state overrides are allowed.

### 2026-09-11: Friction-aware spring diagnostic and measured withdrawal

- Native28 tests original native springs at1920 Hz with128/32 iterations,
  unchanged materials/masses, for1 s. It still fails:0.224984 mN.m static
  moment residual,0.0120112 rad/s joint-velocity RMS and global torque balance.
  More iterations alone are not established as the full-greenhouse fix.
- Added coupon-only `contact_patch_prediction.py`: two geometry-bound native
  friction anchors per patch, circular Coulomb disks sharing compressive normal
  support, all six free-root coordinates and original M/K/C/friction. Native
  contact rows provide geometry ONLY; measured normal/friction forces never
  enter this predictor. Bounded semismooth/trust-step numerical solve requires
  momentum, contact-law, friction-cone and dissipativity residuals. Its internal
  trust steps do not alter physical material coefficients. This deliberately
  omits PhysX strong-friction positional bias and its scalar friction bounds;
  native-law parity, tissue mechanics and full-plant qualification are false.
- `--contact-patch-friction` is an explicit opt-in ONLY for the standalone
  `coupled_contact_prediction` coupon and `unilateral_kv_v1` law. The native
  wrapper persists exact anchor identities and submits only the two spring
  efforts within unchanged native caps. Missing/changed geometry or unresolved
  algebra stops before submission. No root wrench, contact replay, pose/velocity
  override, production spring change or greenhouse visual simplification.
- Native29 FREE240 Hz,720 steps passes all recovery/whole-run energy gates:
  initial7.97146 microjoules, maximum recorded6.01001 microjoules; no growth
  above the unchanged energy allowance. This is the required unloaded control,
  not held grasp/cut evidence. Median prediction1.856 ms, diagnostic tick4.769 ms.
- Native30 HELD1920 Hz,1920 steps restores stiffness equilibrium:maximum
  static moment residual2.34327 microN.m (native27 normal-only361.75 microN.m).
  All tail gates EXCEPT joint velocity pass; native velocity RMS0.0169005 rad/s
  exceeds0.01 rad/s. Therefore it remains FAILED. Median predictor48.534 ms,
  p95 63.051 ms; total diagnostic ticks107.498 wall seconds for1 simulated
  second. It is far too slow for interactive production. Native31 compares
 128/32 iterations without changing this model or physical parameters.
- Focused pure geometry/patch/submission/runner/coupon regression:275 passed
  in2.34 s. Standalone timing explicitly excludes startup, export and console
  progress; it does not claim full-greenhouse performance. Dataset releases,
  reviews/splits, training and hardware commands are unchanged.
- Native31 HELD1920 Hz/PGS128/32 completes1920 steps. Stiffness residual
  7.82931 microN.m and velocity RMS0.00108002 rad/s pass; global net torque
  still fails. Tail spikes include step1042:-15.44 microN.m and1885:-13.52
  microN.m, despite near-zero values between them. Do not average these away
  or declare full equilibrium. Native32 uses TGS with the same coupon/patch
  predictor; step2 prediction fails its unchanged line-search gate, so only
  one physical sample executes. Native33 original native TGS drives also fail
  at1920 Hz/128/32:1.02552 mN.m moment residual,0.104306 rad/s velocity RMS.
- Full working-tree CPU/USD regression (including measured-withdrawal helper
  and adapter tests):2299 passed in103.06 s, logged at
  `data/sim_physics/regression_20260911_patch_withdrawal.log`. It overlaps
  native32 diagnostics in wall time; these timings are not isolated machine
  performance measurements. No coupon result yet establishes reliable
  full-greenhouse cut-retain-withdraw; all failed native traces are preserved.
- Added a source/anchor/material/timestep/contiguous-step-bound seed containing
  ONLY the previous resolved predicted forces. One all-stick KKT candidate may
  accelerate the patch solve; all original physical equations, residual gates
  and cold fallback remain. Six free-root coordinates remain present; no
  predictive contact force is submitted. Native34 matches native31 bitwise
  across all1920 joint positions/velocities, body frames/velocities, submitted
  spring efforts and net-contact torques.1849 steps accept the fast candidate.
  Median wrapper prediction13.818 to4.887 ms; total28.290 to14.895 s. This is
  a timing comparison of diagnostic runs, not isolated-machine production RTF.
  The same global-torque gate still FAILS; optimization does not qualify it.

### 2026-09-11: Measured-start withdrawal integrated as explicit diagnostic

- `measured_withdrawal.py` constructs a backward connector from the actual
  fetched wrist/joints into the existing reverse stroke/approach. It rejects
  off-path/forward-extraction connectors, stale samples, changed left targets,
  bad receipts and exhausted budgets. Waypoints advance on measured posture
  attainment, not elapsed time. It never sends native commands itself.
- `measured_withdrawal_native.py` reads complete native robot/plant inventories,
  all joint/finger positions and command targets. Raw observed finger-stop
  excursions are retained/reported, not clamped: run53's right finger1 is
  +0.207 micrometres beyond its nominal zero stop. Measured geometry is distinct
  from command approval; all submitted/planned targets retain source limits.
  Actual robot frames, exact source FK, held-leaf convex hulls and unchanged
  self3 mm/interarm10 mm/scene1 mm margins are checked. Existing native force,
  penetration, slip, support and bilateral-grasp guards stay authoritative.
- Native static actor positive controls can now be requested lazily, immediately
  BEFORE that exact actor may clear a conservative rejection. Every cached
  obstacle is retained. Every used actor is checked again before acceptance;
  missing coverage, native epoch changes or cleanup errors reject the receipt.
  Fresh per-call epochs and8 s/20,000 total-query bounds remain. This avoids
  eagerly querying80 unchanged actors on every physics tick; defaults outside
  the new withdrawal adapter remain eager. Focused coverage tests95 passed.
- `--measured-withdrawal` is opt-in to the bounded bimanual harness with native
  scene clearance enabled. `withdrawal_controller.py` submits only the right
  joint drive targets after fresh validation. Left targets are held unchanged.
  Float32 radians are rounded toward the measured state when nearest rounding
  overshoots, BEFORE path validation/hash binding; no limit clamp. Submission
  must reproduce that exact packet and its native drive-target readback.
  Final success also requires same-step helper completion AND the independent
  legacy native parked-wrist/clearance check.216 focused tests pass.
- Full54 preserves source assets (hash check true), verifies the left grasp
  and releases the seam at11.8667 s. Initial withdrawal path passes in4.406 s
  with119 native queries. Subsequent checks use8 queries each. At11.9000 s,
  the actual right upper-arm capsule versus distal Leaf004 crosses the1 mm
  planning margin:step2855 clearance1.037 mm,step2856 clearance0.772 mm.
  Left grasp remains bilateral; slip78.2 micrometres at stop. This is a
  conservative-clearance rejection, NOT measured unintended contact/damage.
  Full54 is FAILED; the guard stops before continuing an unsafe corridor.
- Independent reconstruction separates motion:release robot/release leaf
  clearance2.260 mm;last robot/release leaf2.272 mm;release robot/last leaf
  0.766 mm. Leaf deflection, not the new arm command, closes the gap. A bounded
  single-pose IK proposal fixes right joint1 at0 degrees instead of-1.5 and
  preserves the knife TCP while increasing this pair's clearance to5.407 mm.
  Right joint5 still has1.583 degrees below110-degree source limit. Full55
  tests the entire original stroke/transit and measured withdrawal; this
  single-pose calculation is not whole-path or future-motion qualification.

### 2026-09-12: Guarded withdrawal checkpoint; full-path constraint remains

- Full55 (same full greenhouse, right joint1 fixed at0 degrees) verifies the
  left grasp, then rejects the original full stroke at offset+3.490645 mm:
  `stroke_IK`. No right cutting motion or seam release is executed. The better
  endpoint clearance does NOT establish a feasible whole path. Source assets
  remain unchanged. Next work must address whole-path redundancy/clearance;
  it must not clip the stroke or relax source joint/collision limits.
- Full CPU/USD regression:2408 passed in104.51 s, recorded at
  `data/sim_physics/regression_20260911_measured_withdrawal.log`.
  Independent read-only review found no remaining material blocker in the
  exact native packet binding or same-sample completion checks. These tests
  validate software guards, NOT successful full-greenhouse physics.
- Installed `PhysxMaterialAPI` exposes no strong-friction flag or equivalent
  setter/readback. Public PhysX source makes positional friction-anchor bias
  a plausible explanation for the diagnostic predictor/native discrepancy,
  but native31/34 do not establish that cause. No invented USD attribute or
  unsupported material change is used. The optimized patch predictor stays
  coupon-only because its global torque qualification still fails.
- Cut-retain-withdraw remains unqualified; calibrated tissue fracture,
  completed forward stroke and deposit remain unverified. No training,
  collection, dataset review/split changes or hardware commands were made.

### 2026-09-12: Post-cut spring-work discrepancy isolated, not a grasp slip

- CPU replay of the original61-pose stroke confirms full55's failed sample57
  reaches joint5=109.999427 degrees and rotation error0.00507984 rad. Keeping
  right joint1=-0.75 degrees solves all61 poses with minimum joint-limit margin
  0.346804 degrees; no stroke shortening or limit change. Failed stroke IK
  diagnostics now retain the exact failing vector/residuals/sample/evaluation
  count (including null when unavailable), rather than only prior clearance.
- Full56 uses that constant shoulder setting. The full plan and left grasp
  pass; signed edge-load seam release occurs11.866667 s. At11.900000 s the
  unchanged conservative left fingerl2/Leaf000 screen rejects. Actual source
  convex shapes remain separated1.411121 mm; the box-expansion margin is
  0.997787 mm against the required1 mm. No unintended native load was observed
  at this stop; maximum grasp slip97.463 micrometres. Leaf carrier motion
  (2.6925 mm and6.0023 degrees since release), not finger motion, closes the gap.
  Changing grasp skew trades finger clearances and is not a physics repair.
- Added read-only `spring_work.py`: exact submitted float32 actuation readback,
  copied pre/post native joint coordinates and adjacent-step checks. It reports
  constant generalized-effort work plus declared quadratic-potential change;
  it is NOT contact work, native-integrator energy or whole-system balance.
  Full57 reproduces56's2856 samples exactly in plant dynamics, palm and contact
  state. Across the8 post-release intervals, declared elastic energy rises
  5.115758 mJ while commanded spring work is-0.013339 mJ: the coordinate-level
  constitutive discrepancy is+5.102420 mJ. At the final sample Joint004:1 has
  q=-0.191231 rad but only+0.000904572 N.m spring effort. This is strong evidence
  against treating the contact-omitting predictor as the intended beam law.
  Root support, collision geometry, materials and motion commands were unchanged.
- Corrected interpretation of coupon31/34 torque spikes: the old gate checks
  STATIC zero torque. Independent complete signed-contact/momentum accounting
  gives maximum tail dynamic torque residual2.97048 nN.m (RMS0.94565 nN.m).
  Spike1042's-15.439874 microN.m is explained by angular-momentum change,
  including orbital terms. Static limits still fail704/961 samples; low RMS
  velocity does not imply negligible acceleration. New `contact_momentum.py`
  keeps static and dynamic results separate and introduces NO dynamic pass gate.
  Full physics/material/settling qualification is still not established.
- Focused work/snapshot/momentum tests80 passed in11.52 s; earlier IK/planning
  tests69 passed (additional null-evaluation test subsequently added). Work is
  now on contact-aware spring prediction and source-bound native geometry,
  not further tiny posture changes to mask a constitutive error. Full57 remains
  FAILED; tissue fracture, retention/withdrawal completion and deposit remain
  unverified. Original greenhouse/plant assets remain hash-unchanged.

### 2026-09-12: Full contact capture and finite-finger geometry qualification

- Full58 records all native target contact rows plus the full45-coordinate
  plant mass matrix, COM Jacobians, actual robot/plant poses and velocities.
  The added collector is read-only. Cut release still occurs11.866667 s;
  withdrawal still fails11.900000 s. This is NOT an end-to-end success.
- `plant_contact_binding.py` exports the exact cached shaft/finger local
  geometry, original force-based compliance and complete plant collider list.
  Native-bound and CPU-authored exports have distinct schemas. Replay checks
  source shape/joint hashes against recorded grasp bindings and checks material
  parameters against the source report; a checksum is not native cooking proof.
- `grasp_contact_shadow.py` adds bounded OFFLINE counterfactual solves. Native
  impulses are never force inputs, predictions are not actuated, and recorded
  next velocities cannot certify predictions made with a different effort.
  Attached-root algebra retains measured initial root-to-joint momentum;
  detached algebra retains all six unactuated root coordinates. These remain
  approximations, not verified native support or robot-response models.
- Strict replay01/02 rejects real held contacts rather than accepting an ideal
  infinite pad plane. At native58 step811, PCM normals differ from the actual
  inner-pad plane by7.49524 and3.51940 degrees. Native raw-normal medial features
  agree with shaft geometry within0.100 micrometres, while ideal-normal
  substitution introduces9.7--17.5 micrometre errors. Some cached representative
  points project80.992 micrometres outside the current finite pad footprint.
  No tolerance enlargement, coordinate clipping, or invented contact is used.
- `capsule_box_witness.py` separately computes one exact finite capsule/box
  closest witness, including cap/face, edge and corner geometry. Axis/box
  intersection is explicitly unsupported because unsigned distance cannot
  determine penetration normal/depth.44 witness tests pass (125 related).
  A fresh-pair predictor is being evaluated separately from cached PCM replay;
  one source-coefficient spring per pair has a different aggregate convention
  from multiple full-strength springs per native contact point. It is NOT a
  calibrated/native-equivalent contact law and is not enabled for actuation.
- Native59b is a20-second right-parked HOLD CONTROL, not a cut trial. It
  completes bounded, verifies bilateral grasp and records maximum slip
  0.875076 micrometres, zero blade contacts and no right-arm motion. Source
  assets remain unchanged. Its overall cut-qualification state correctly
  remains FAILED: cut/withdrawal/retention/deposit cannot pass a no-cut control.
  Runs59/59a were CLI rejections before SimulationApp startup (incompatible
  withdrawal/hold flags, then insufficient duration); neither ran physics.
- Bound capture was verified natively in59b. Latest focused geometry/binding/
  shadow tests122 passed; additional source-binding checks bring shadow tests
  to22 passing. Dynamic knife contact prediction remains unsupported. No
  training, dataset/review/split edits, hardware commands or source-asset edits.

### 2026-09-12: Contact-aware hold experiment completed, NOT promoted

- `--experimental-contact-springs` is an explicit default-OFF, stationary
  full-robot HOLD-ONLY diagnostic. `FreshHoldSprings` writes only the39 source
  intrinsic float32 joint efforts, with exact command readback. It rejects
  changed source K/C, nonzero native drives, stale q/v, missing contacts,
  unsupported geometry, unresolved solves and predicted target-finger
  noncancelling normal-plus-friction totals >=0.5 N. Actual native all-contact,
  slip, penetration and support guards remain. Native FLT_MAX effort caps are
  labelled numerical source limits, NOT meaningful mechanical certification.
- Shadow03 resolves six of seven sampled cases (one is contact-free); the
  dynamic knife case rejects. Post-release predicted finger totals reach
  0.5938/0.7143 N: not measured native violations, but not permissible actuation
  proposals. No cut-phase adoption or custom rigid-knife force replay is made.
- Native60 applies30 experimental packets, then stops at0.129167 s because a
  Leaf005/right-forearm4 contact is outside the model. The same contact pair
  exists in the baseline. No leaf/contact is removed to get past the rejection.
- Native60a explicitly uses the original controller to establish the verified
  grasp, then attempts experimental activation at3.5 s. From that point there
  is no legacy fallback.3960 exact effort packets are verified, steps841--4800.
  The20-second HOLD control completes bounded with uninterrupted bilateral
  evidence, source assets unchanged and no native guard failures. Maximum
  slip is0.945853 mm, final slip0.744439 mm; native per-finger load upper bounds
  peak0.33674/0.29755 N. This is NOT cut, retention-after-cut, or deposit success.
- Independent comparison REJECTS60a as a physics improvement: two proximal
  coordinates end at+1.05254/-1.26771 rad; declared beam-coordinate potential
  grows from6.202 to346.096 mJ. Since activation, potential change339.894 mJ
  plus submitted-effort work0.713 mJ gives diagnostic discrepancy340.607 mJ.
  This is not native-integrator work/conservation proof. Final5-second distal
  capsule-pole position jitter RMS increases1.049 to6.451 micrometres versus
  baseline59b. A visibly held, nearly stationary tip can hide severe opposing
  internal rotations. Full-plant material-domain limits are unspecified;
  the separate coupon's0.05-rad limit cannot be presented as plant calibration.
- Instrumented median tick grows15.54 to36.52 ms; real-time factor falls0.253
  to0.121.60a's solve median is2.079 ms; capture, copies, native step and guard
  work also contribute. These are diagnostic/headless timings, not GUI FPS.
  Do not promote this model or claim a latency improvement.
- Rechecked the native-drive alternative before adding a custom knife solver.
  Native33's static moment test is NOT an internal drive-law measurement.
  NVIDIA's implicit-drive derivation describes different position/velocity
  iteration behavior (source documentation, not installed-phase readback):
  https://nvidia-omniverse.github.io/PhysX/physx/5.3.0/_downloads/6acf3afb8f69452757e0e766b5a22978/implicitDrives.pdf
  Added explicit128/0 comparison support without changing defaults or gates.
  Matched native35 (TGS1920 Hz,128/0) reduces carried joint-velocity RMS from
  0.104306 to0.007934 rad/s, but static spring residual remains0.99948 mN.m.
  Native36 (PGS1920 Hz,128/0) gives0.008399 rad/s and0.264405 mN.m. Both still
  FAIL the unchanged spring/static qualification. Dynamic momentum accounting
  must stay separate; native constitutive-law failure is not proved by these
  endpoint measurements. No finite-difference velocity replaces native state.
- Final CPU/USD regression:3059 passed in124.71 s, recorded in
  `data/sim_physics/regression_20260912_contact_geometry_v3.log`. Earlier v2:
  3038 passed; the first run's test-only NameError was corrected and rerun.
  Full grasp-cut-retain-withdraw still requires a physically qualified spring/
  contact integration; tissue fracture calibration and deposit remain absent.

### 2026-09-12: User-requested closer grasp, camera alignment and downward path

- User GUI61 reproduced the baseline signed seam release at11.866667 s,
  then withdrawal rejection at11.9 s.11.9 simulated seconds cost71.267 wall
  seconds; median control15.287 ms, native step11.013 ms, render17.494 ms.
  A planning call cost11.217 s. This is not real-time performance. Startup
  also reported severe OS commit pressure; helper/CPU imports sometimes failed
  with a paging-file error. No unrelated process or memory setting was changed.
- Added opt-in `--knife-alignment camera`: measure the original right wrist
  camera radial placement, and rotate the COMMON knife parent about wrist Z
  (-90 degrees relative to legacy) to put the arc on that same side. The flange
  translation, camera/bracket transforms, original visible meshes and all blade/
  arc collision parts stay intact. Unknown mounting fails closed; repeat is
  idempotent. Legacy/coupon mode does not require a camera. This is geometric
  alignment, NOT CAD screw certification. Global blade-down still requires a
  validated wrist pose and is not claimed for the parked SDK-ready posture.
- Added opt-in `--exact-grasp-arc`. Old requests snap to ~24.45 mm segment
  centres, making small requested changes ineffective or abruptly changing
  the contacted segment. The new hand goal is a material point inside the
  selected existing capsule. Settled replanning, post-fetch slip, report and
  guide markers use its rotating body-local offset. No plant vertex, body,
  joint, mass, gain or source asset is moved/edited by this option. All original
  full-finger cut-plane and native contact guards remain.
- Native64 used the closer centre46.675573 mm with camera-aligned knife.
  The approach/closure geometry screen passed, but native contact reached the
  unchanged0.5 N per-finger cap at2.9125 s before bilateral verification. This
  setup is REJECTED. No knife motion/release occurred and no guard was loosened.
- Native65 uses exact60 mm (old centre71.125955 mm), nominal cut10 mm and
  0.5 mm closure compression. Its20-second right-parked HOLD CONTROL completes
  all4800 ticks bounded, left grasp verified, max slip0.190292 mm and final
  slip0.000896 mm. Initial full-finger cut-plane clearance34.0 mm (required10).
  This moves the grasp11.126 mm closer and preserves source assets. The overall
  bimanual report deliberately remains FAILED because no cutting/withdrawal/
  post-cut retention/deposit was attempted. This is one qualified hold case,
  not a general reliability claim or a calibrated plant material model.
- Added opt-in `--cut-style downward`: fresh measured stem-transverse gravity
  direction, no >30-degree deviation from down, complete <=0.5 mm tool stroke,
  arm chord/length ratio0.80..0.98, and straight Cartesian transit with <=2 mm
  pose samples plus <=1-degree joint screening and <=0.5 mm line deviation.
  Failed straight transit has NO joint-space detour fallback. All IK/self/plant/
  native-force gates remain, and measured non-downward blade direction refuses
  release. Legacy proposal/fixed-shoulder flags cannot override the new mode.
  These are planning constraints, NOT proof of a completed physical sequence.
- Offline plant-only posture checks found nearer-stance downward IK candidates,
  but native66 at station(0.3,0.9,-128.5445 deg) failed the full-greenhouse
  startup screen with10 possible overlaps, including torso/target foliage and
  left-wrist/main-stem contacts. Physics did not start. No surrounding vine or
  collision was removed. A full-scene-valid coordinated stance/grasp/tool
  approach is still required; new downward cutting has NOT been demonstrated.
- Profiled native62 isolated the slow step: native simulate median10.613 ms;
  disabling optional profiler output alone did not solve it. Batched rigid-pose
  checks now copy/validate ALL current frames with unchanged scalar tolerances,
  without caching past frames or dropping contacts. Native63 uses this plus4
  physics threads: median control16.189 ->13.337 ms (~18% lower), post-step
  4.408 ->2.758 ms, native simulate10.613 ->9.502 ms. Profiled headless RTF
  0.1627 ->0.1821. All2856 trajectory rows' physical/contact/guard values match
  native62 exactly; differences are only8 wall-time receipts,8 remaining wall
  budgets and1 fault receipt time. Same cut time11.866667 s, same withdrawal
  rejection11.9 s and max slip0.097463 mm. This is NOT a GUI FPS measurement.
- `run_camera_aligned_grasp_demo.cmd` exposes only the tested closer HOLD
  inspection, with clear hold-only UI/button/result labels and no automatic
  motion. It is not advertised as a grasp-cut-retain demo. Existing legacy
  cutting defaults remain available; the unqualified downward mode is opt-in.
- No hardware commands, dataset collection/review/split changes, training,
  source-asset modifications, collision disabling or fidelity reduction.
  Full downward cut/withdrawal/retention/deposit and real-time latency remain
  unresolved.
- Native67 retests the ORIGINAL collision-clear robot station with exact60 mm
  grasp, camera alignment and the downward planner, with native static queries
  enabled. Left grasp verifies again; at3.5 s all50 proposed tool corridors are
  rejected before arm IK/motion:27 left-hand/right-camera conflicts and23 blade
  plate/protected-main-stem conflicts.104 native queries clear one coarse
  rejection; the remaining checks retain their conservative margins. This does
  not prove every possible downward approach impossible or every bound overlap
  an actual mesh collision. No knife contact or release occurred. Evidence:
  `data/sim_physics/bimanual_downward_20260912_67/report.json`.
- Additional offline exact60 mm grasp checks at the original XY station and
  three headings found no clear tested tool subset. Moving the diagnostic cut
  from10 to20 mm within the existing admissible interval did not clear those
  sampled conflicts either. These are bounded CPU geometry checks, not native
  execution or a justification to change the nominal cut. The launcher still
  uses10 mm. A jointly feasible grasp orientation, wrist-tool corridor and
  robot stance must be established before promoting downward execution.
- Final CPU/USD regression:3082 passed in161.11 s; evidence:
  `data/sim_physics/regression_20260912_downward_v2.log`. The earlier regression
  caught two legacy coupon fixtures without cameras and three synthetic
  report fixtures without the new material-point fields; backward compatibility
  was restored, then the complete suite rerun. `git diff --check` passed.

### 2026-09-12: Coordinated downward posture and closer-contact investigations

- This increment is still experimental. No new downward grasp-cut-withdraw-
  retain/deposit sequence has passed. All motion uses native joint drives;
  no weld, body-pose override, forced release, force-cap relaxation or visual
  simplification was used. The supplied plant, knife, cameras and surrounding
  greenhouse remain. Dataset/review/split/training/hardware state is untouched.
- Added source-enclosing, edge-aligned plate bounds (`tool_bounds.py`) for
  downward planning only. All original plate vertices, full thickness and an
  extra1 micrometre remain inside; this removes an empty bounding-box wedge,
  not blade geometry. It is not a native-cooking certificate. The downward
  transverse fan includes0,+/-10,+/-20,+/-25 degrees when within30 degrees of
  gravity; upward fallback remains forbidden. Native68 still rejects all350
  tested tool corridors before right motion (139 plate/main-stem,199 hand/tool,
  and12 other rejections); native refinement clears78 coarse rejections from
  297 queries. These are bounded-search results, not global unreachability.
- Added explicit, exact-URDF-bounded initial right-arm and left-IK-seed inputs,
  then `--torso-degrees` for six-joint coordinated posture proposals. The torso
  option requires the package downward fixture and cannot coexist with a yaw
  override. These set the diagnostic initial condition BEFORE physics; they
  never teleport a running robot or certify a path. `torso_limits_degrees()`
  reads the installed exact URDF, not remembered/model-family limits.
- Native69 rejects open left fingers against right forearm (2.708 mm versus
  required3 mm). Native70/71 fail full-scene spawn checks; no physical motion.
  A37 mm request (72) fails the CLI's existing40 mm minimum before Kit; do not
  describe it as a physical grasp test. Native73's exact40 mm original closure
  reaches0.765296 N at2.895833 s, failing the unchanged0.5 N finger-contact cap.
- `--force-closure` is an opt-in experimental drive-target controller, NOT a
  new grasp detector or approved default. It requires adjacent guard-accepted
  contact samples at240 Hz, slows near-contact closure to0.5 mm/s, targets
  0.12 N compressive support with a0.03 N deadband, and backs away at5 mm/s
  from excessive/unqualified contact. Non-gravity drive effort is at most
  0.15 N inside the original gravity-reserved motor budget. No support is
  invented from unsigned loads, and actual bilateral dwell/slip still gate
  right motion. Failed/unsafe contact cannot reenter the controller. Feedback
  adds5 s for closure qualification and requires >=28 s diagnostic duration.
- Native74/75/76/77/78 still fail the contact cap: slower closure, changed
  contact-solver ordering and smaller non-gravity drive effort did not fix the
  close-to-root contact instability. Native75 measured an approximately0.5 mm
  root-body displacement in one failing tick. A predictor that ignores unknown
  contact/support reactions is still a limitation; no calibrated root cause or
  physical material accuracy is claimed from this single measurement.
- At0.9 s the selected shaft in75 had rotated2.112382 degrees from its rest
  palm alignment; position-only replanning left that mismatch. Experimental
  feedback mode now uses bounded (<10-degree) fresh segment-axis alignment and
  orientation interpolation, with renewed whole-finger cut-plane clearance.
  Native79/80 encounter unqualified inner-pad geometry;81 reaches0.652793 N.
  This correction has NOT established a successful closer grasp.
- `--anchored-pad-damping` explicitly compares damping based on finger mass
  against the original free-two-body reduced-mass prior. K=1000 N/m, actual
  masses, contact friction and guards stay unchanged. Damping8.075220 versus
  1.080535 N.s/m in82 still fails at0.590560 N. Default OFF; neither prior is
  lab calibrated. Do not promote this failed comparison as a realism fix.
- Cached authored joint connectivity in `shaft_grasp_native.py` now invalidates
  on joint enabled/endpoint/removal/resync notices. Cached sets are copied;
  closed/missing notice bindings cannot reuse success. Native body frames,
  contacts, forces and per-step evidence remain fresh and uncached.
- Native83 repeats the exact60 mm, camera-aligned, right-parked geometric
  closure HOLD control:4800 ticks/20 s, verified grasp, maximum slip0.19029247 mm,
  final0.00089563 mm. Its bimanual result remains FAILED because it performs no
  cut/withdrawal. Native84's explicitly smaller1.25 m collision half-window has
  EXACTLY identical4800 trajectory rows; all143 surrounding plants plus the
  detailed target remain rendered, all source hashes unchanged. Static contact
  plants change76->50 and wire proxies14->8 only outside the guarded window.
  Complete initial collision spheres plus150 mm margin must fit BEFORE any
  exclusions, and the same boundary guard runs each tick. Default stays2 m.
  This pair is NOT a speedup: median control8.41315->9.60910 ms, native step
  6.28315->7.28655 ms, wall44.3892->49.9254 s. Regression work overlapped some84
  execution; a quiet paired comparison is required before any latency claim.
- Bounded CPU coordinated torso/arm searches found three self-screened endpoint
  candidates with underhand left approach and right extension0.92..0.96. These
  omitted full-greenhouse/native corridor qualification and are NOT presets.
  Native85 fails open-finger/right-camera clearance2.871 mm (<3 mm). Raising
  the waiting knife clears that self check, but86 fails23 full-scene possible
  overlaps, including main stem and foliage. Native87 with a retracted right
  waiting pose passes startup, then at0.9 s rejects an open-finger/main-stem
  coarse bound during the left approach. No right motion or release occurred.
- The left approach previously had no live native refinement of static coarse
  bounds, although the right planner did. With `--native-static-clearance`,
  left known palm/pad/camera boxes and restored arm capsules now use the same
  enclosing-shape-plus-margin query contract. Exact static-actor positive
  coverage, a single unchanged physics/USD epoch and final coverage validation
  are required; exceptions/timeouts/cleanup faults reject acceptance. Dynamic
  shaft/leaf geometry, unknown shapes and actual native hits stay rejected.
  Queries close before returning; this is not contact/grasp evidence. Tests
  cover whole-bound preservation, unknown-pad rejection, dynamic leaves,
  geometry failure, initialization/check/validation/cleanup faults and state
  restoration. Native88 is the matched execution test; results recorded below.
- Intermediate complete regression:3123 passed in159.58 s in
  `data/sim_physics/regression_20260912_feedback_v2.log` (before torso/native-left
  additions). Targeted torso/kinematics87 tests and native-left/collision95 tests
  pass. Native reports/logs for this increment are under
  `data/sim_physics/bimanual_downward_20260912_68` through `_88`, except72
  (CLI-rejection log only). Final validation is recorded separately below.

### 2026-09-12: Closer underhand hold, source-hash memory fix and C-drive cleanup

- Native88 passes full-scene spawn and the previously rejected left approach.
  Twelve live static queries clear ten coarse main-stem bounds; final query
  coverage/epoch validation passes. Exact54 mm underhand grasp verifies at3.5 s
  after134 bilateral-contact steps, leaving28 mm whole-finger clearance from
  the nominal10 mm cut (required10 mm). This uses geometric closure, NOT the
  unsuccessful force-closure experiment. All350 downward tool corridors still
  fail before right IK/motion:215 plate/main-stem,65 finger/right-camera,
  19 finger/right-bracket and51 other rejections. No cut; source hashes match.
  Evidence: `data/sim_physics/bimanual_downward_20260912_88/report.json`.
- Native89 is the matching right-parked HOLD control. It finishes4800 ticks/20 s
  with verified opposing contact, maximum slip0.00067964 mm and final slip
  0.00018854 mm. Final whole-source hashing then raises MemoryError, so the run
  is NOT fully qualified and must be repeated. Cut/withdrawal/retention gates
  remain false. An independent post-shutdown check of24 context-source files
  (188,349,947 bytes) matches but does not reconstruct the missing full final
  inventory. Evidence: native89 `report.json`, `trajectory.json`, sibling log.
- `file_integrity.sha256_file` now streams1 MiB blocks instead of allocating
  entire source assets. Initial/final benchmark and context SHA-256 coverage
  is unchanged. Tests cover empty/boundary/multiblock files, exact digests,
  bounded read sizes and missing files.64 targeted integrity/benchmark tests
  pass; a native repeat remains necessary.
- Windows memory pressure is a separate unresolved host issue: after our
  simulator exited, committed memory was338.97 GB of350.36 GB and kernel paged
  pool254.35 GB (decimal), with7.6 GB available RAM. These are measurements,
  not identification of the responsible driver or proof of a simulator leak.
  No processes, review apps, drivers, pagefile settings or machine restart
  were changed. Disk cleanup does not establish a memory/performance fix.
- User-authorized C-drive cleanup: moved five old crash dumps (3,198,266,382
  bytes) to `data/maintenance/c_drive_cleanup_20260912`, verified SHA-256 then
  removed originals; they remain recoverable there. Deleted708,330,546 bytes
  of completed stale VSCode installer archives/markers. Subsequently removed
  82,354,219,414 bytes of six unused text-model caches: Meta-Llama-3-8B-Instruct,
  Meta-Llama-3.1-8B-Instruct, internlm2-7b, Qwen3-8B, Qwen2.5-7B-Instruct and
  Qwen2.5-1.5B-Instruct. Cache deletions have no local backup; upstream
  redownload is needed. Exact roots, ages, internal links and exclusive file
  access were checked before deleting leaves. No broad home/cache-root delete.
  Preserved Qwen3-VL, other vision/robotics weights, credentials, active caches
  and downloads. C reported257.01 GB free afterward; do not attribute the
  entire change in free space to the86.26 GB of logical cleanup alone.
- Expanded regression v2:3551 passed,2 skipped,1 failed in170.05 s. The failure
  reproduces in isolation in the unchanged legacy skeleton implementation:
  its interpolation test assumes an extracted centreline is perfectly straight
  to1e-12, despite documented partial-ring fitting bias. The interpolation
  fixture now uses an exact polyline; a separate bent-polyline test verifies
  piecewise arc length instead of an endpoint chord. Existing mesh-fitting
  accuracy tests and implementation are unchanged. Full rerun v3 passes3553
  tests with2 skipped in179.52 s; evidence:
  `data/sim_physics/regression_20260912_left_clearance_v3.log`.
  `git diff --check` passes. This is CPU/USD validation, not native cutting
  qualification; the8 mm-standoff native90 trial is still in progress.

### 2026-09-12: Native retry, qualified closer HOLD and launch-memory preflight

- Checkpoint `8881c17` commits the preceding guarded posture/contact diagnostics,
  bounded-memory source hashing, tests and documentation. Experimental force
  closure/damping remain default OFF. No training data or source assets enter
  this commit; the tracked worktree was clean immediately afterward.
- Native90 failed during full-context construction, before physics: even the
  1 MiB streaming read could not allocate memory. Windows then reported340.62 GB
  committed of350.36 GB,254.46 GB paged pool. The streaming change removes the
  unnecessary whole-file allocation but cannot fix exhausted system commit.
  The responsible process/driver is unidentified; no restart or app termination
  was performed. Evidence: native90 `report.json` and sibling log.
- Added `sim_physics.host_memory`: read-only Windows GetPerformanceInfo counters
  converted from pages to bytes, with explicit timestamps. The full-package
  robot benchmark records them before importing SimulationApp. It refuses
  startup if counters fail or reserve is below16 GiB commit headroom /4 GiB
  available RAM, saves a `blocked_host_memory` report, exits2 and never creates
  a simulator. Those limits are conservative launch policy after observed
  failures, NOT a measured peak requirement or capacity guarantee. No package
  download, persistent setting, process or pagefile modification is involved.
  Unsupported OSes are explicitly not checked.70 targeted tests pass, including
  installed Windows counter reads and proof of no Isaac import when blocked.
  API reference: https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-performance_information
- Host commit headroom later recovered without agent intervention. Native91
  starts with20.69 GB reserve, completes full scene loading, verifies the54 mm
  grasp, then rejects all350 tested8 mm-standoff tool paths before right IK.
  Static refinement makes421 queries, clearing227 coarse bounds with no query
  error. Remaining first-rejection counts:108 plate/main-stem,156 left-hand/
  tool,72 right-camera/bracket versus neighboring fruit,4 versus leaf and10
  versus backdrop. All108 plate/main-stem conflicts are at the first waiting
  pose. This does not establish actual collision for every enclosing-tool
  overlap or prove global infeasibility. Source hashes match. Native91 report
  and log retain the complete failures; no cut or right movement occurred.
- Native92 repeats the closer underhand HOLD control for20 s/4800 ticks with
  no exception: completed=true, bounded=true, left_grasp_verified=true and
  source_assets_unchanged=true. Maximum slip0.00067964 mm; no right motion or
  cut. Overall bimanual status/exit2 intentionally remains failed, not a false
  full-sequence success. Tick wall43.104 s (real-time factor0.464); measured
  retained timing samples have median control7.930 ms / native5.800 ms. This
  is a headless hold measurement, not a visual FPS or established speedup.
  Evidence: `data/sim_physics/bimanual_downward_20260912_92/report.json`.
- A bounded offline subset search now checks the permitted10/20 mm cut arcs
  and left approach tilt. It loads source anatomy and full robot geometry in
  an anonymous USD stage without Kit, but deliberately has no static-scene,
  native contact, right-IK or grasp qualification. Its results are proposals
  only and cannot bypass full-scene/native checks. Diagnostic script/log:
  `data/sim_physics/underhand_subset_20260912.py` / `.log` (local ignored data).
  Final regression and any resulting native trial are recorded below.
- Offline subset search finishes18 cases in142.13 s:10 left-IK solutions and
  8 failed solves with the tested seed. Tilting the left approach does not
  improve the count of locally clear tool paths at either cut arc; this is a
  bounded seed/search result, not global IK infeasibility. Native93 therefore
  retains the verified tilt0 underhand posture and tests20 mm cut /8 mm
  standoff. It does NOT change the10 mm default or any annotation.
- Final CPU/USD regression after memory-preflight integration:3567 passed,
  2 skipped in173.23 s (`data/sim_physics/regression_20260912_host_memory_v1.log`).
  The offline subset search overlapped part of this CPU-only regression, not
  any native timing run. `git diff --check` passes. No training/hardware work.
- Native93 (same54 mm grasp,20 mm seam,8 mm standoff) passes setup but stops
  during closure at2.941667 s/706 ticks: finger2 all-contact load0.52796594 N
  exceeds the unchanged0.5 N guard. Grasp never verifies, so right planning and
  motion are not authorized. Full source hashes match. The cut-plane change
  also changes preauthored shaft segmentation (selected body centre46.676 ->
  52.926 mm while the exact material grasp stays54 mm). This trial therefore
  is not an isolated tool-clearance comparison or evidence that20 mm cuts are
  intrinsically unsafe. Contact/discretization sensitivity remains unqualified;
  no causal/material claim is inferred from this single failed transient.
  Evidence: `data/sim_physics/bimanual_downward_20260912_93/report.json` and log.
- Current handoff: native92 qualifies the specified20-second intact HOLD only;
  no new downward cut, withdrawal, post-cut retention or deposit pass. Next
  investigate contact/discretization stability and a jointly clear hand/tool
  corridor; do not enlarge force limits, remove surroundings or use the offline
  subset results as execution permission. All native trials launched here have
  exited; the pre-existing review Kit process130120 remains untouched.

### 2026-09-12: Contact-envelope, physical grasp-span and vertical-stroke qualification

- Still **not a reliable full cutting sequence**. This increment does not
  authorize training, hardware motion, dataset/review changes, forced release,
  welds, body-pose overrides, wider force/slip limits or removal of surroundings.
  Source knife/cameras, original greenhouse and surrounding plant visuals stay.
- Found a geometric discretization defect: USD capsule height is its cylinder
  length, excluding hemispheres. Shortening every segment by two radii makes
  neighboring capsule tips meet at zero-radius internal necks. Added explicit
  `--stem-contact-model continuous_internal_capsules_v1`: internal capsule
  spines reach their joint anchors; physical endpoints and BOTH sides of the
  separable seam keep flush rounded caps. No capsule crosses the seam. This
  is not a flat wound or tissue model. Legacy `flush_capsules_v1` remains the
  default. Correct capsule extents are authored in both modes. Reference:
  https://openusd.org/dev/api/class_usd_geom_capsule.html
- `stem_envelope.py` and tests verify internal coverage, seam-side separation,
  rejection of impossible endpoint spans, unchanged source layers, body masses,
  COM/inertia, material K/C, topology, friction and visuals. This fixes geometry;
  it is NOT proof that all native contact instability had that single cause.
- Native94: continuous envelope, nominal10 mm cut and exact54 mm underhand
  grasp; HOLD completes20 s/4800 ticks with bounded/verified grasp and unchanged
  source hashes. Maximum slip0.00093052 mm. Overall bimanual status intentionally
  remains failed because no right motion/cut/withdrawal occurred. Native95's
  20 mm cut still fails closure at0.51134269 N; native96 with the existing lower
  0.25 mm compression option fails at0.50626783 N. The0.5 N guard is unchanged.
  Evidence: `data/sim_physics/bimanual_downward_20260912_94` through `_96`.
- The downward planner formerly proposed exactly transverse strokes only.
  Added true world-vertical proposals when BOTH existing native angular gates
  remain satisfied (stroke/actual-stem and edge/actual-stem absolute dot<0.3).
  Actual anatomical axis is never replaced in ShearGate or release evidence.
  Vertical candidates precede each existing transverse tilt fan; the same full
  tool, arm, static-scene, contact and Cartesian transit checks still apply.
  Native97a (10 mm cut) and98a (15 mm cut) each verify grasp then reject all400
  tool corridors before right IK. This is a bounded-search failure, not a proof
  of global unreachability. Source hashes match. CLI typo attempts97/98 never
  launched Kit; their logs are not physics tests.
- Native99:20 mm cut with finer20 mm segments initially fails the LEFT planned
  closure against a farther segment of the same petiole. Found a second
  discretization-dependent rule: planning allowed selected +/-one segment,
  even when the same physical pad spans more links. The continuous-envelope
  mode now derives a bounded axial span from both complete finger colliders at
  the final grasp pose, then walks only contiguous source shaft capsules whose
  current extents meet that span. No skipping a gap to a folded-back distal
  part, support-side segments, leaves, other branches, palm or camera allowance.
  This changes expected planning contact identity ONLY, not collision filters
  or native grasp/contact verification. Native evidence remains stricter and
  can still reject contacts outside its independently qualified neighborhood.
- Matched native100 now passes that left corridor and verifies native grasp at
  54 mm with18 mm whole-finger clearance from the20 mm seam (required10 mm).
  Of400 cutter candidates,398 fail the full-tool corridor; TWO true vertical
  proposals (normal sign-1, center-edge contact, plane tilt10/15 degrees) pass
  the corridor but fail endpoint IK. No right motion or release. Source hashes
  unchanged. Evidence: `data/sim_physics/bimanual_downward_20260912_native99`
  and `_native100`, including reports, events and trajectory records.
- Offline source check confirms the nominal10 mm center is outside the parent
  source convex hull (11.24 mm separating-halfspace lower bound). The full
  plate corridor conflict is not evidence that the cut center is inside the
  trunk. Diagnostics remain local ignored files under `data/sim_physics/`.
- Offline48-seed-per-pose IK search does not solve the two native100 poses at
  the existing station. Coordinated base/dual-arm proposals can solve both ends
  of the vertical stroke, with5-degree right joint-limit reserve, by shifting
  the station within150 mm per horizontal axis and25 degrees yaw. Four of five
  tested proposals pass held-finger self and local rigid-tool screening; these
  omit full greenhouse/native contact qualification and are NOT presets or
  permission to move a running robot. No torso or source plant change.
- Regression at this checkpoint:4250 passed,2 skipped,47 subtests in212.31 s
  (`data/sim_physics/regression_20260912_contact_span_v1.log`); targeted physical
  grasp-span/native-clearance tests50 passed. No visual FPS/speedup claim: native
  trials here were headless, and some CPU-only diagnostics overlapped regression.
  Windows paged pool remains abnormally large (~256.7 GB at09:44 UTC); cause
  unidentified. Read-only launch preflight remains enabled. Review Kit130120
  untouched; no OS restart or unrelated process termination.

### 2026-09-12: Native contact timing and guarded acquisition reliability

- Still **no qualified complete downward grasp/cut/withdraw/retain sequence**.
  Deposit is not implemented in this harness. Do not present hold-only trials,
  IK solutions, or joint/seam release as full success or calibrated tissue cutting.
- Fixed a conservative startup false positive: the rotated rectangular bound
  of a robot capsule can intersect a scene bound while the actual capsule is
  separated from the ENTIRE scene bound. `startup_screen.py` now tests that
  separation with the original1 mm margin. Unsupported capsule geometry and
  actual overlap remain rejected; no collision mesh, filters or margins change.
  Native101's shifted-base proposal remained blocked. Fixed-base torso proposals
  native102-108 pass startup; their full-scene checks are not clearance for a
  moving torso or authorization to teleport a running robot.
- Feedback grasp acquisition is now event-driven: earliest3.5 s, unchanged
 100 ms consecutive native bilateral dwell, hard13.5 s timeout. The whole right
  sequence is scheduled from actual verification; no right-arm motion while
  waiting. Geometric closure retains its3.5 s deadline. Timers cannot create
  a grasp or cut. Native103 (old8.5 s deadline),104 and105 (bounded13.5 s) all
  fail acquisition: more waiting alone did not resolve the feedback limit cycle.
- Native105 records native body/finger poses and signed normal contact rows.
  A read-only previous/post-frame comparison finds **all194 rejected rows**
  satisfy the SAME authored capsule/pad surface tolerances in the preceding
  native pose, but fail against the moved post-fetch pose. Median preceding
  capsule surface error is19.3 nm; median post-fetch error is0.636 mm.
  This is contact geometry timing, not permission to enlarge shapes/tolerances.
  Evidence: `data/sim_physics/bimanual_downward_20260912_native105` and
  `data/sim_physics/contact_frame_diagnosis_20260912.py` / `.log`.
- Primary PhysX documentation distinguishes actor state at the end of the step
  from the pose at contact detection, exposed in native C++ via CONTACT_EVENT_POSE:
  https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/docs/Simulation.html#callback-sequence
  Our Python adapter does NOT claim to read that native extra-data field.
  The opt-in `--grasp-contact-frames pre_solve_pgs_v1` instead copies exact native
  poses before synchronous240 Hz PGS stepping. It validates adjacent step IDs,
  ordered coverage, rigid transforms and one capture per step; missing/stale
  captures fail closed. Default `post_fetch_legacy` stays reproducible.
- Generation poses validate each reported point/normal against its authored
  collider, without altering signed callback force or tensor integrity checks.
  Separate POST-fetch capsule-to-inner-pad-face proximity, side-of-pad and
  penetration checks qualify continuing contact, using existing offsets and
 1 mm penetration guard. Old contact after withdrawal/tunneling cannot verify
  a hold. Current pad reaction axes still determine compressive support.
  Force closure can distinguish a valid generated contact from a currently
  maintained grasp; only the unchanged bilateral/dwell controller permits cutting.
- Matched native106 with corrected timing reaches an actual0.668417 N finger
  contact at6.629 s and stops at the unchanged0.5 N guard. No grasp, right motion,
  cut or retention. Native107 geometric closure with the corrected timing also
  fails acquisition at3.5 s. The sensor bug is fixed experimentally; contact/
  spring/controller robustness in the new downward posture remains unresolved.
- Native108 repeats the known20 s HOLD control (original torso,10 mm cut,
 54 mm grasp, continuous internal capsules) with corrected contact timing.
 4,800 ticks complete, native grasp verifies at3.5 s, and all subsequent samples
  remain bilateral. Maximum post-verification slip0.00093052 mm; source hashes
  unchanged. This is intact hold only; no right motion, cut, withdrawal or deposit.
- No datasets, review decisions, training jobs, hardware commands or source
  plant/robot assets were changed. Trial results are diagnostic, training-ineligible.
  GUI FPS and low-latency operation are not qualified by these headless trials.
  Windows kernel paged pool is still abnormal (~258.7 GB at10:41 UTC); driver
  cause unknown. Read-only launch reserves stay enabled; unrelated review jobs
  remain untouched. No reboot or further cache deletion was performed.
- Native109 repeats the corrected-frame feedback trial with the existing
  higher-damping option; no force guard violation, but no simultaneous20 mN
  support from both fingers before the13.5 s timeout. Native110 additionally
  tries40 mN acquisition before a ramp to the old120 mN holding target. It also
  fails:0 simultaneous qualifying support samples out of3,240 ticks. Pressure
  control changes were therefore NOT promoted; their patch is preserved at
  `data/sim_physics/native110_light_acquisition_trial.diff`. The production
  `ForceClosure` pressure law remains unchanged. These tests do not justify
  increasing force limits, acquisition time, or calling a one-sided contact a hold.
- A read-only source-envelope audit of native110 finds1,573 of3,753 nonzero
  normal-load rows (>1 microN) inside an adjacent intact shaft capsule by more
  than1 micrometre. Maximum load among these rows0.327352 N; median15.23 mN.
  This is measured geometry/force overlap, NOT proof that removing these rows
  yields correct dynamics. No contacts were removed, rescaled or relabeled;
  the native contact modification API was NOT implemented. Script/report:
  `data/sim_physics/shaft_union_audit_20260912.py` / `.log`. Next investigate
  the segmented-contact surface and coupled spring/contact response with a
  matched physical-contact coupon before changing the full-scene model again.
- Current-contact proximity applies to every pair carrying a NONZERO impulse;
  exact-zero manifold rows remain checked for identity/generation geometry and
  recorded, but cannot count as support or invalidate other continuing loaded
  contacts solely because their old pair has separated. No nonzero magnitude
  cutoff, force threshold or tensor reconciliation rule changed.
- Acceptance before promoting the downward preset: repeated guarded acquisition
  in that actual posture, then full approach/cut/withdraw/intact retention,
  followed by perturbation trials. Hold-only success cannot pass that gate.
  Cut remains a force-qualified pre-authored seam release, not calibrated
  tissue fracture; deposit and a validated learned policy remain separate work.
- Final CPU/USD regression for these retained changes:4,284 passed,2 skipped,
 47 subtests passed in214.08 s (`data/sim_physics/regression_20260912_grasp_frames_v3.log`).
  Targeted contact/acquisition/adapter/benchmark tests:370 passed. The earlier
  v1 regression found five legacy mock-interface failures; compatibility was
  repaired and both v2 (4,278 passed) and final v3 passed. Failed experiments
  are not silently included in the production pressure controller.
- Native111 repeats native108 using the final nonzero-contact proximity check:
 4,800 ticks,20 s, identical recorded palm/plant joint state/grasp point/seam/
  slip and measured finger force at every tick. Grasp verifies at3.5 s; maximum
  slip0.00093052 mm. These are two identical-initial-condition hold trials,
  NOT perturbation robustness, cutting or post-release retention validation.
- Rechecking earlier base proposals AFTER the startup capsule-bound correction
  finds four of five v1 proposals clear the complete SOURCE plant startup
  screen; the fifth has arm/torso self-interference. All four otherwise-valid
  v2 proposals still collide with target Leaf004 and remain rejected. No leaf
  was removed and no native robot was moved to these proposals by this check.
  Evidence: `data/sim_physics/coordinated_spawn_check_20260912_v2.log`.

### 2026-09-12: Downward IK joint-limit reserve

- Offline v1 base proposals pass the27-pose straight-down wrist stroke, but the
  unconstrained solver ends at right wrist joint5=109.999427 degrees, only
 0.000573 degrees below its110-degree URDF limit. These solutions were not
  promoted or executed as a reliable cutting preset.
- `Rby1Kinematics.solve_pose` now accepts an explicit nonnegative joint-limit
  reserve; zero preserves the previous solver exactly. Invalid/nonfinite or
  impossible intervals reject. Downward `BimanualRobot` IK uses a3-degree reserve
  for initial/fallback solves, every stroke sample and Cartesian transit solve.
  Exact URDF limits, native drives, collision/pose tolerances and other modes
  are unchanged. This is a planning margin, not hardware validation.
- Coordinated offline proposals limited to120 mm X/150 mm Y base displacement
  and25 degrees yaw retain the original torso and solve BOTH stroke endpoints
  with3-degree right-arm reserve. Four proposals pass complete source-plant
  startup, held-finger self/tool and27-pose stroke checks. They are not full
  greenhouse transit, native grasp or cutting certification. Evidence:
  `data/sim_physics/coordinated_margin_station_20260912.py` / `.log` and
  `data/sim_physics/qualified_margin_station_20260912.py` / `.log`.
- Native112 executes only the left acquisition at the source-clear yaw-seed20
  proposal in the full greenhouse: startup passes, but finger2 reaches
 0.599550 N and the original0.5 N guard stops closure before verified grasp.
  No right motion, cut or retention; source hashes unchanged. Thus correcting
  geometric/IK feasibility is insufficient to fix contact-model robustness.
  No unsafe preset replaces the existing demonstration configuration.
- Regression:4,295 passed,2 skipped,47 subtests in212.52 s
  (`data/sim_physics/regression_20260912_joint_reserve_v1.log`);110 focused IK/
  downward planning tests passed. Full grasp-cut-withdraw-retain still unqualified.

### 2026-09-12: No-reboot diagnostics and isolated contact comparisons

- User cannot restart Windows. No reboot, unrelated process termination, driver
  unloading, security changes or pagefile changes were performed. Native113
  stopped BEFORE Isaac initialization: 10.69 GiB commit headroom was below our
  conservative 16 GiB launch reserve (not an NVIDIA minimum). Later headroom
  recovered to approximately 31 GiB without an OS mutation, allowing114 onward
  to run with the same memory gate. This is not a kernel-memory repair.
- Added read-only `python -m sim_physics.pool_tags`: bounded Windows x64 pool-tag
  query, explicit undocumented ABI, no writes/process control. Snapshot:
  `data/sim_physics/pool_tags_20260912_no_reboot.json`. CBnb accounts for about
  220.85 GiB raw paged allocations; this differs from committed pool accounting.
  A read-only driver binary search found CBnb in `cbfltfs4.sys`, signed Callback
  Filter 4.1.105.114, a running boot filesystem filter. That is an investigation
  lead, NOT verified consumer ownership or allocation-stack proof. No unload
  attempted; in-use filters can require reboot and unsafe unloading risks harm.
- `--isolate-station` builds the same source-bound package station, complete
  foreground plant, floor and full robot, then session-deactivates surroundings
  before physics. All original source files remain unchanged. The result is
  explicitly NOT full-greenhouse/training qualification. Native114 exactly
  reproduces112: at2.891667 s finger2 reaches0.5995504725 N before grasp. Thus
  the failure is reproducible locally; removed surroundings were not needed.
- Matched slow `--force-closure` native115 still fails at4.85 s,0.7990595423 N.
  A contact audit found one nonzero row inside an adjacent capsule envelope.
  Added an isolated HOLD-only `flat_cylinders_v1` comparison: same segment
  frames/mass/inertia/K/C/art, exact-length zero-margin analytic cylinders,
  no overlapping spherical caps. Native cylinder process setting is explicitly
  selected/read back; cooked geometry readback is not claimed. Pad verification
  requires actual cylindrical side proximity, not an enclosing capsule. Native
  116 geometric closure fails0.7626870197 N; this alternative is NOT promoted.
- Native117 flat/slow closure stalls on cached zero-load manifold rows whose
  positive separation exceeds contactOffset. Fixed their classification:
  retain/log those rows as inactive; do not let them veto approach or count as
  support. Every nonzero row, penetration guard, identity/frame check and
  bilateral force/dwell requirement remains unchanged. Native118 progresses
  beyond that stall but still fails0.7483432344 N. This is a readout/controller
  bug fix, not a claim that physical contact instability is resolved.
- Added isolated native-drive HOLD observer: keeps original native K/C, verifies
  zero targets/external actuation and unchanged drive inventory, never invents
  measured drive effort/work. Native119 floating-root/external-anchor and120
  fixed-articulation controls fail BEFORE grasp, at0.266667/0.116667 s, from
  unwanted plant/robot contact0.72963547/0.62748046 N. Native maxForce readback
  is float32 maximum, not a missing/zero drive budget. Fixed-root release is
  still forbidden pending a state-preserving topology implementation.
- Added HOLD-only uniform solver iteration authoring on EVERY robot/plant
  rigid body and articulation, with post-reset USD checks. Native121 native
  drives at128/0 also fails (unwanted contact0.61903849 N); effective native
  iteration readback is unavailable. No default is changed. Native122 tests
  the original implicit springs under the same uniform128/0 configuration;
  its outcome is recorded separately after completion.
- CPU/USD physics regression: **3,319 passed in134.40 s** in
  `data/sim_physics/regression_20260912_isolated_contact_v1.log`.
  No test count here represents native cutting success. Full downward
  grasp-cut-withdraw-retain remains FAILED/unqualified; deposit and calibrated
  tissue fracture remain absent. No dataset labels/splits, training jobs,
  hardware commands or source visual assets were changed.

### 2026-09-13: Explicit finger-effort comparison and long-hold failure

- Native122 (implicit springs, uniform128/0, slow closure) avoids the earlier
  force stop but times out at13.5 s: support2.23/117.33 mN, no bilateral grasp.
  The unloaded finger is at the geometry-derived minimum aperture. The other
  finger's native drive requests opening while measured contact remains loaded.
  Neither a normal static equilibrium nor an exact native defect is established.
- Added isolated HOLD-only `--explicit-finger-effort`: same200 N/m,5 N s/m PD,
  same measured gravity feed-forward, <=0.15 N nongravity effort, original total
  actuator/contact budgets. Disable ONLY the two left native drive gains; submit
  their bounded PD effort explicitly after current targets are set. Native
  gain/inventory, contiguous step, latest targets and submitted effort readback
  are checked. Other joint drives, native contact/friction, plant mass/K/C and
  visuals stay intact. Logged commands are not measured internal drive forces.
- Native123 matched122 with explicit fingers: still no grasp,13.5 s timeout,
  peak all-contact finger upper bound0.275828 N; final support3.68/112.48 mN.
- Native124 changes ONLY the existing bounded closure bias0.5->1.0 mm from123.
  Bilateral grasp is acquired, but HOLD fails at14.6125 s with a force spike
  (signed support4.034535/0.578837 N); max slip0.182961 mm. The low slip and
  several seconds of bilateral contact do NOT establish a reliable hold.
  Immediately before failure neighboring torsion coordinates approach opposite
  half-turns; quadratic coordinate energy grows to3.208767 J, then maximum
  reported joint speed reaches165.605 rad/s. This remains a spring/contact
  stability failure, not successful cutting or a calibrated tissue-energy result.
- Read-only124 pose/coordinate audit: the positive rotation-vector interpretation
  of native joint positions reconstructs relative body orientations within
  0.49 microradian across sampled initial/1/8/14.608 s frames (joint rest frames
  inferred from the initial sample). Thus the observed opposite rotations are
  consistent with native body poses, not merely a mislabeled plotted angle.
  Native125 repeats124 with velocity iterations8 instead of0; outcome pending.
- Regression before the final nonfinite explicit-PD guard: **3,329 passed in
  125.19 s**, `data/sim_physics/regression_20260913_finger_effort_v1.log`.
  Focused controller/configuration checks44+58 passed. No experimental option
  was promoted to a default, cutting permission, GUI preset or dataset approval.

### 2026-09-13: Flat-side hold and collision-aware downward endpoint search

- Native125 changes124 velocity iterations0->8: grasp is acquired, but lost
  at11.279167 s. It does not qualify the controller. Native126 instead changes
  only124's overlapping capsules to the explicit flat cylinders. It completes
  7,200 ticks/30 s: grasp verifies8.166667 s, maximum slip0.175265 mm and maximum
  per-finger all-contact upper bound0.165512 N. No cut/right motion occurs.
  The generic full-cut result remains failed because those stages were deliberately
  absent; bounded/completed/left-grasp gates are true, not full-cut qualification.
-126's final2 s have max|q|0.104411 rad, coordinate span0.00078325 rad and
  quadratic coordinate energy approximately9.53 mJ, unlike124's opposite
  half-turn growth. HOWEVER native carried joint-velocity RMS is0.769893 rad/s
  while poses barely change. Thus this is a bounded contact-hold result, NOT
  validated constitutive dynamics or realistic tissue behavior. Headless tick
  time76.3376 s for30 simulated seconds (RTF0.393); no real-time/GUI FPS claim.
- Added separate `--isolated-cut-contact-trial`, requiring the complete explicit
  flat-cylinder/PGS240 Hz/implicit-spring/128-0/feedback fixture. This authorizes
  only a guarded isolated diagnostic, not automatic seam release. Original
  signed blade load/direction/dwell, source identity, native bilateral grasp,
  slip, full robot contact and clearance guards remain. Defaults and full
  greenhouse settings are unchanged; native126 is not a tissue certificate.
- Native127 reproduces the grasp then rejects cutting BEFORE right motion:
  two IK-converged blade poses intersect the held Leaf002 near the right
  shoulder.398 other rigid tool corridors reject before IK. No leaf/camera
  collider is removed to resolve this. This is geometric rejection, not a cut.
- Downward planning now reuses bounded local seven-DOF pose-family continuation
  after a converged but blocked IK endpoint (initially16 steps in either direction;
  subsequently32, still stopping at joint bounds or failed correction).
  Added explicit3-degree reserve to predictor and corrector. Every alternative
  retains endpoint arm/self/held-plant checks, full sampled stroke and separate
  transit checks. Legacy behavior stays single-branch. Unit tests cover rejecting
  the first colliding solution, screening the alternative, and refusing all
  alternatives when the obstacle persists. This is not collision-free IK by fiat.
- Native128 examines34 converged configurations; all remain blocked by held
  foliage/stem. No right movement or cut. A bounded offline nearby-station search
  uses the recorded held-plant frames and original source assets; it finds
  endpoint-only proposals, NOT native/full-stroke/transit clearance. Evidence:
  `data/sim_physics/held_station_search_20260913.py` / `.log`. Native129 checks
  one80 mm lateral base proposal through the real startup/grasp/path guards;
  its outcome is recorded below. The plant is not moved/removed.
- Focused cut-config/shape/controller checks98 passed; planner/IK61 passed,
  then29 passed including the two new real-planner/synthetic-geometry regressions.
  Full regression is running separately. No dataset, training or hardware changes.

### 2026-09-13: Native capsule refinement and nearby-grasp checks (not a cut success)

- Native129's80 mm lateral station preserves the grasp but all34 converged
  endpoints still collide with held foliage. Native130 extends bounded local
  pose-family continuation to32 steps/direction:55 converged endpoints, no
  full path. Later family poses reach static MainStem27 near right forearm link5.
- Added opt-in `--native-capsule-sphere-cover`, requiring native static queries.
  After an enclosing-box hit, a finite sphere UNION covers the ENTIRE proposed
  arm capsule plus the unchanged >=1 mm margin. Radius >=sqrt((r+margin)^2+(h/2)^2),
  endpoint caps, float32 centre error and upward radius rounding prevent gaps.
  This is not sparse point testing. Exact-actor positive controls are required
  for both native box and sphere queries, including final epoch validation.
  Errors, stale scenes and bounded query/time-budget exhaustion fail closed.
  No native geometry, self/held-plant checks, force limits or visuals change.
- Native131 uses that refinement:67 sphere queries, one coarse static rejection
  cleared, no query errors. All55 endpoints nevertheless reject; no right motion
  or cut occurs. The original source plant and wrist cameras stay present.
- Native132 proposes grasp75 mm from the attachment with cut20 mm: startup
  rejects an open-finger/Leaf000 overlap before physics. Native133 instead uses
  grasp65 mm: native bilateral grasp verifies5.508333 s, but all83 converged
  endpoints reject before cutting. Some late poses intersect another branch's
  Leaf028 near the forearm. These are recorded failures, not successful cuts.
- Offline grasp/approach searches in `data/sim_physics/grasp_cut_clearance_search_20260913*`
  propose only source-startup and rigid wrist-tool clearance, not full robot
  paths or native grasp. Invalid165/195-degree grasp-roll proposals were rejected;
  subsequent search uses the supported roll, approach tilt and jaw-skew controls.
- Full physics regression: **3,359 passed in120.54 s**, evidence
  `data/sim_physics/regression_20260913_sphere_cover_v1.log` (includes new capsule
  cover containment/error tests, native-query controls and isolated-trial guards).
  Reliable grasp-cut-withdraw-retain remains UNQUALIFIED. Native126's30 s hold
  is the bounded positive result; deposit and calibrated tissue fracture remain
  unsupported. No labels/splits, training, hardware, source assets or OS settings
  changed. Windows was not restarted.

### 2026-09-13: Reobserved retraction corridor correction

- Native134: verified65 mm grasp, requested5 mm target pull, stopped at5.883333 s
  with `IK interpolation exceeds 0.5 mm`. Root cause: native reobservation changes
  the grasp goal translation/rotation, but the pull used palm Z instead of the
  actual refreshed start-to-goal path. Those directions are not necessarily equal.
- `checked_approach_retraction` now returns a bounded metric vector along that
  actual checked translation corridor, with an8 mm reserve inside its measured
  length. Rigid/finite inputs are required. Existing path orientation is reused;
  this is not an assertion of fixed wrist orientation or new plant clearance.
  Logs identify the chosen vector. No IK tolerance, grasp-loss window, contact
  limit, or mandatory post-motion native reobservation was relaxed.
- Native135 passes the previous interpolation failure and physically moves while
  bilateral at6.0 s, but loses the grasp at6.283333 s before finishing5 mm.
  Maximum recorded slip1.414957 mm; latest signed support0/12.847961 mN.
  Right arm never moves and no cut occurs. Thus the path-command bug is fixed,
  but5 mm retained repositioning is NOT qualified. Native136 tests2 mm separately.
- Additional offline searches:75 nearby base proposals (no clear whole-source
  endpoint),72 torso/station cases (same result), and72 wrist seed combinations
  (19 distinct converged/self-clear poses rejected by scene,17 duplicates,36
  solve failures). These conservative offline failures are not proof of physical
  impossibility. Opposite-side grasps at the tested station fail left IK; direct
  below/longer grasps have startup foliage conflicts or no new rigid tool corridor.
  Evidence under `data/sim_physics/held_scene_station_search_20260913*`,
  `grasp_cut_clearance_search_20260913*`, `wrist_branch_search_20260913*`.
- Focused98 tests pass. Full physics regression: **3,376 passed in123.48 s**,
  `data/sim_physics/regression_20260913_retraction_v1.log`. No current reliable
  downward cut, native retained deposit, or tissue-fracture calibration is claimed.

### 2026-09-13: Small retained pull and native tool-clearance diagnostics

- Native136 completes the requested2 mm pull and reobserves at7.008333 s:
  measured palm displacement1.999424 mm, grasped-point displacement1.989396 mm,
  junction displacement0.244040 mm, maximum recorded slip0.216938 mm. Bilateral
  grasp is retained at reobservation, but all83 converged cutter endpoints still
  reject; no right motion/cut. This qualifies neither5 mm repositioning nor a
  long-duration/repeated retained manipulation sequence.
- Added read-only `native_body_bounds.py`: synchronous, same-stage complete
  native property-query inventory, callback/finite/rigid guards, and the rigid
  countertransform identity. Native137/138 found that PhysX property queries
  also report the two explicitly disabled legacy knife/arc boxes. They are now
  separately audited, not accepted as active geometry; unknown responses still
  fail closed. Native139 obtains19 active wrist records plus the selected shaft
  and first leaf carrier without a physics step or USD epoch change. Property
  records alone do NOT prove active actor shape identity or authorize motion.
- `native_occupancy_clearance.py` is an unintegrated diagnostic for complete
  volume queries, NOT sparse point sampling. Given complete enclosing bounds,
  countertransform each occupied obstacle cell into the parked tool's frame.
  Actual native obstacle queries prune empty cells; actual native tool queries
  test the entire remaining cell plus margin. Gap-free subdivisions, frozen-epoch
  occupancy caching, roundoff reserve, actor controls, budgets and retained
  rejection at1 mm cell resolution prevent unresolved geometry becoming clear.
  No body/collider is repositioned, removed or filtered by this algorithm.
- Native140 checks15 pre-cut poses against the three camera/bracket actors and
  three selected target shapes. Native141 adds the full-cell refinement, with
  bounds enclosing both native property ranges and source vertices before the
  independent1 mm margin.2,305 refinement queries/1,664 node visits/859 cached
  cells resolve7 previously unresolved pairs, retain37 pair rejections; every
  pose still has at least one unresolved camera/leaf pair. These are subset
  diagnostics, NOT full-arm/stroke clearance or actual cutting. Right arm stays
  parked. Final actor/epoch validation passes. Outputs:
  `data/sim_physics/bimanual_downward_20260912_native137` through`native141`,
  `native_tool_bounds.json` within each completed probe.
- Geometry-only screening of other intact original-package petioles is underway
  (`data/sim_physics/source_tool_access_search_20260913*`) to find an unobstructed
  baseline. A source tool corridor does not establish grasp, arm IK, startup,
  native stability or a cut. No source foliage or camera hardware is removed.
- Full physics regression: **3,410 passed in119.71 s**, evidence
  `data/sim_physics/regression_20260913_native_occupancy_v1.log`;34 focused tests
  cover the new diagnostic algorithms. Main execution still uses the existing
  guarded planner; these queries are NOT silently enabled as motion permissions.

### 2026-09-13: Continuous downward approach/cut IK and intact-source search

- Fixed a disconnected planning assumption: Cartesian approach IK can reach the
  same knife pose on a different redundant-arm branch from the independently
  proposed cut endpoint. Requiring identical elbow joints unnecessarily rejected
  that approach. Downward planning now explicitly obtains the screened approach
  terminal and rebuilds the entire cut stroke from that exact joint vector.
  The first stroke sample is identical to the final approach sample: no snap,
  appended joint jump, or reuse of a stroke checked on a different branch.
- The pose-only transit mode is explicit; the helper's default still requires
  the original joint endpoint. Both modes verify terminal position/orientation.
  The rebuilt stroke retains extension, inter-arm, whole self/tool and held/static
  plant checks. Straight Cartesian approach, <=0.5 mm line error, existing joint
  reserves, contact loads and cut gates are unchanged. Legacy planning is unchanged.
- Synthetic redundant-arm continuity/rejection tests and the full physics suite
  pass: **3,419 passed in120.61 s**, evidence
  `data/sim_physics/regression_20260913_transit_v1.log`. This is code validation,
  NOT a measured native cutting success.
- Offline screening examined30 intact original-package petioles with the actual
  fitted wrist tools. Six have at least one sampled source-clear tool stroke:
  seed11/SubStem43, seed19/SubStem41, seed47/SubStem41, seed53/SubStem41,
  seed67/SubStem43 and seed73/SubStem41 (all `*_full` plant IDs). No source leaves,
  neighboring stems, camera hardware or native colliders are removed.
- Paired-hand tests and coupled base/torso/arm solves remain proposals only.
  Seed19's tested positive-normal paired-hand options require an85 mm grasp arc
  (65 mm distal to the20 mm seam), farther than the desired close-junction grasp.
  Many pose-reachable configurations fail whole-robot startup against the plant.
  Source-stem/robot-capsule avoidance now guides the offline search; unchanged
  full source/tool/path screening still decides acceptance. These candidates are
  NOT native, greenhouse, VLM or physical-cut qualification.
- Evidence: `data/sim_physics/source_tool_access_search_20260913.log`,
  `paired_hand_search_*20260913*.log`, `paired_station_solve_*20260913*.log`,
  `paired_robot_search_*20260913*.log`. No dataset, review, training, hardware,
  running user application or Windows memory configuration is changed.

### 2026-09-13: Frozen native whole-robot startup validator (diagnostic only)

- Added `sim_physics/native_startup_clearance.py` with complete source/robot
  inventories, before-and-after native actor positive controls, frozen-epoch
  guards, bounded queries, full-margin boxes and enclosing capsule sphere unions.
  Unknown actors, incomplete callbacks, expired epochs and missing actors reject.
  This helper is NOT integrated as execution authority; existing startup and
  motion guards remain. It does not load, step, reposition or edit the scene.
- Native143 distinguished real foliage contacts from five coarse bounding-box
  false positives among seven startup pairs. Native144/145 then screened five
  proposed whole-robot poses: all still reject on leaf collisions. Source stem
  `convexDecomposition` geometry explains some conservative USD-AABB rejections,
  but does not explain away the remaining actual native overlap results.
- Native145: all 475 source colliders controlled twice, 47 robot shapes in the
  inventory, 1,349 queries, 0.297 s after scene loading; zero physics steps,
  zero authored body-pose change during queries, final epoch validation passes.
  Initial PhysX loading changes a matrix element by at most 8.33e-7; this is
  recorded separately from query activity. No motion/cut qualification follows.
  Evidence: `data/sim_physics/bimanual_downward_20260912_native145/`.
- 29 fake-query contract tests pass; full physics regression **3,448 passed in
  120.19 s**, `data/sim_physics/regression_20260913_native_startup_v1.log`.
  Regression success is not evidence of native grasp/cut completion.

### 2026-09-13: Explicit branch contact fixture and native retained-grasp test

- Added opt-in `--branch-contact-fixture`, permitted only with the complete
  isolated cut-contact diagnostic protocol. It retains the original complete
  main stem, selected petiole and ALL that petiole's leaves. Other branch/fruit
  components are deactivated in the diagnostic session only, with an explicit
  inventory in the report. Source geometry, materials, transforms, target
  mechanics, force limits and production greenhouse defaults are unchanged.
  This fixture is NOT intact-plant access, greenhouse, damage or training proof.
- Native146/147 exposed a source USD `/World/RBY1` override stub: it is active
  but undefined, not a constructed robot. The preconstruction guard now rejects
  active defined rigs, not this empty source override. Both inactive namespace
  and undefined-override cases have regression tests. Neither failed launch
  entered contact simulation. Native148 passes actual package construction.
- Native148 verifies grasp at5.508333 s, completes the2 mm retained retraction
  and reobserves at7.008333 s; maximum recorded slip0.216938 mm.23 right-arm
  endpoints pass the endpoint checks, but all straight transits from the parked
  arm reject. There are ZERO blade contacts, no cut and no deposit. This result
  separates an approach-planning failure from the branch contact mechanics;
  it does not qualify a reliable cut. Original source hashes remain unchanged.
- Native149 uses the unchanged stop-before-motion probe for a prepositioned
  cutter with4 mm extra upward standoff. Four coarse main-stem overlaps clear
  through native queries; the blade/shaft and upper-arm/leaf pairs still reject.
  Zero physics steps, no motion authorized. No validation gate was bypassed.
- Other source-plant searches now bind each IK proposal to its own passing knife
  facing direction and include the existing inter-arm clearance in proposal
  costs. Seed11/67/73 offer closer hand-only candidates, but none is native
  full-sequence qualified. A prepositioned cutter test, if successful, would
  still not certify the normal parked-pose approach.
- Evidence: `data/sim_physics/bimanual_downward_20260912_native146` through
  `native149`, `paired_*20260913*.log`, `preposition_search_20260913*.log`.
 17 focused fixture tests; full physics regression **3,465 passed in124.55 s**,
 `data/sim_physics/regression_20260913_branch_fixture_v2.log`.

### 2026-09-13: Native startup integration and first prepositioned blade loading

- Downward transit failures now report the exact IK/inter-arm/self/plant/line/
  terminal rejection, without changing checks or issuing extra native queries.
  Native151 isolates the parked approach failure: right forearm versus target
  Leaf_001 at approximately13.2% of the Cartesian approach, all23 endpoints.
- Native152 checks a prepositioned cutter with4 mm extra upward standoff on a
  different redundant elbow configuration. The complete fitted-shape native
  startup check passes:75 scene colliders,47 robot shapes, before/after actor
  controls, zero physics steps and no query-time authored pose change. The old
  box warning on the blade/shaft is overly conservative for this exact pose.
- Added explicit `--native-startup-clearance`, restricted to isolated cut-contact
  diagnostics with native path clearance enabled. It always checks the complete
  current scene, not just coarse warnings: all source AND robot actors have
  positive controls before/after; self geometry, stage/epoch/time, callback
  completeness, finite transforms, loading quantization and cleanup are checked.
  Failure stops before the first step. Default startup behavior is unchanged;
  no named-plant exception, collision filter or contact-force waiver exists.
- Native153 uses that option:75 scene/47 robot collider inventory,47 robot actor
  controls twice,484 queries,1.6333 s startup including native parsing, zero
  physics steps during startup validation. It then physically verifies grasp
  at5.675 s, reobserves after2 mm pull and plans at7.179167 s, and moves the right
  arm into blade contact. It stops at13.929167 s: BladePlateContact versus the
  intended Segment_001 shaft produces0.508689 N classified as unwanted contact.
  Loaded normal rows lie outside the unchanged3 mm seam axial window;203 eligible
  contact rows carry zero load. Maximum eligible signed resistance and dwell are
  both zero. Maximum grasp slip0.326810 mm. NO seam release/cut/deposit occurred.
  A prepositioned start does not certify the default parked-pose approach.
- Evidence: `data/sim_physics/bimanual_downward_20260912_native150` through
  `native153`; native153 `bimanual_trajectory.json` retains original-order normal
  rows and eligibility reasons.48 focused startup/lifecycle tests pass; full
  physics regression **3,491 passed in124.00 s**,
  `data/sim_physics/regression_20260913_native_startup_screen_v1.log`.

### 2026-09-13: Bounded blade aim and fresh-contact path feed

- Added opt-in axial blade aim, bounded to +/-1.5 mm in the isolated downward
  contact fixture. This shifts only the commanded edge centre: actual seam,
  release joint, GT annotations, 3 mm axial band and all force/dwell/grasp gates
  stay unchanged. Both approach and stroke use the same offset; default is zero.
- Native154 uses -1 mm aim with the native153 configuration. Load-bearing rows
  now enter the valid leading strip and axial band. Grasp remains verified;
  at13.804167 s total blade load jumps from0.204147 N to2.321678 N, including
  2.113857 N against the kinematic proximal support. It stops before release.
  Last accepted signed resistance0.135525 N is below the0.2 N threshold.
- Added `--blade-force-feed`: isolated downward signed-seam diagnostic only,
  240 Hz, fresh preceding guard-accepted native load, no pose overrides. It
  retimes/reverses only the already screened stroke:2 mm/s free,0.1 mm/s near,
  <=0.05 mm/s loaded, hold at0.22..0.28 N signed resistance and backoff above
  0.32 N total or0.28 N signed. It is NOT a guaranteed actuator force cap.
  Actual original release gates are unchanged. Contact acquisition times out
  after30 s with NO release; this option permits a bounded40..60 s trial.
  Failure-tick raw normals are copied before guards without calling cut logic.
- Native155: same geometry/materials as154, feedback enabled. It delays and
  reduces the spike but does NOT solve it: at35.420833 s total blade load reaches
  0.598164 N (0.225184 N friction), above the unchanged0.5 N limit. Last accepted
  signed resistance0.166641 N; maximum qualifying dwell only4.167 ms. Loaded
  support and branch contact are both eligible. Maximum grasp slip0.293065 mm;
  no release, withdrawal, retention-after-cut or deposit is qualified. Even a
  0.078 micrometre commanded increment can precede this rigid-contact spike.
- Evidence: native154/155 `report.json`, `bimanual_trajectory.json`;37 feedback
  tests plus17 aim tests, full physics regression **3,545 passed in132.05 s**
  (`data/sim_physics/regression_20260913_blade_feed_v1.log`). Source hashes stay
  unchanged. These are isolated branch diagnostics, not greenhouse-access,
  tissue-calibration, hardware or VLM-training approval. Next: test explicitly
  labelled local stem contact compliance, not larger safety thresholds.

### 2026-09-13: Native downward seam release; post-cut grasp still fails

- Added `--seam-contact-compliance`, available ONLY with the complete isolated
  blade-feedback trial. Two original seam-adjacent shaft surfaces receive a
  native force-based implicit contact spring (1000 N/m,2 N s/m, engineering
  priors). Geometry, masses, joints, original 0.5 friction, contact offsets,
  knife/camera/other plant materials and release guards are unchanged. Existing
  material bindings are never silently overwritten. Session-only; no source
  asset edit. This is local contact compression, NOT volumetric tissue or a
  calibrated fracture-energy model. Mechanism reference:
  https://nvidia-omniverse.github.io/PhysX/physx/5.4.0/docs/RigidBodyDynamics.html
- Native156 reaches a genuine native contact-qualified engineering seam release
  at37.700 s:7 consecutive steps/29.167 ms, signed resistance0.200093..0.200946 N,
  peak normal-plus-friction bound0.304333 N, maximum axial distance2.562981 mm,
  and left slip0.229372 mm. No timed release, force-limit increase or forced
  grasp was used. Exact native fixed-joint disabling is the modeled event;
  calibrated tissue cutting remains false.
- Overall trial FAILS at37.733333 s: left contact is lost on the first fetched
  step after release; slip grows to3.446175 mm in33.3 ms. Both force and contact
  geometry guards remain active. No post-cut retention, withdrawal, deposit,
  full-greenhouse sequence or training eligibility is claimed.
- Diagnosis in progress: the contact-omitting implicit spring predictor reports
  about48.6 mJ declared quadratic coordinate energy before release and83.4 mJ
  at failure, while submitted spring work is tiny. This repeats the previously
  documented constitutive discrepancy, NOT a measured physical energy balance.
  Also the second finger's command has drifted to14.1 mm while its measured
  opening is2.27 mm and explicit opening PD effort is saturated. Saturation,
  actual contact equilibrium and post-release retention need separate checks.
- Evidence: `data/sim_physics/bimanual_downward_20260912_native156/` report and
  trajectory.14 local-material tests; full physics regression **3,559 passed
  in133.21 s** (`regression_20260913_seam_contact_compliance_v1.log`). Source
  hashes unchanged. Next is coupled contact/spring and retention validation;
  this one cut is not advertised as reliable grasp-cut-retain.

### 2026-09-13: Repeated cuts, retained-grasp capacity remains unresolved

- Added explicit `--native-spring-cut-trial` comparison: original native K/C
  throughout, zero external spring actuation, unchanged native inventory/targets
  checked before every step. The read-only observer may follow a real release
  only in this opt-in; it never authorizes one. Native157 fails BEFORE grasp at
 0.270833 s: target Leaf002/left wrist0.529517 N plus smaller unwanted contact.
  The native-drive alternative is therefore NOT promoted as a repair.
- The complete isolated protocol can now explicitly compare128 position /0 or8
  velocity iterations. Native158 repeats156 with8 velocity iterations: signed
  seam release at37.683333 s,29.167 ms dwell, resistance0.201048..0.202053 N,
  total upper0.279986 N. It still fails retention at37.716667 s, slip3.250057 mm.
  This numerical comparison leaves all contact, grasp and release guards intact.
- Added `--finger-target-antiwindup` for explicit isolated feedback only. It
  projects target state using fresh native position/velocity into the existing
 200 N/m /5 N s/m PD's effort interval, subject to the original geometry limits.
  No positions, velocities, actuator/contact caps or safety thresholds change.
  Default behavior remains available; an unachievable target is not a grasp.
- Native159 repeats156 with antiwindup. The second-finger target before release
  is3.019 mm instead of14.107 mm; measured opening2.271 mm. Cut releases again
  at37.654167 s, but the branch still slips to3.423117 mm by37.691667 s. Correcting
  target drift alone does NOT establish retention or resolve the spring model.
- Read-only static wrench audit (`grasp_wrench_audit_20260913_v2/v3.log`) uses
  native159's contact points, source COMs/masses cross-checked against native
  masses, and an8-sided mu0.5 friction cone. Detached mass8.613447 g; gravitational
  moment about the grasp approximately9.508 mN.m. The recorded contact patch
  cannot balance it under the0.5 N per-finger normal-plus-friction budget in
  this fixed-patch approximation, even including recorded zero-load contacts.
  The latter optimistic case needs about0.548/0.590 N normal load per finger.
  These are CONDITIONAL static LP results, not proof that every possible
  contact/grasp is infeasible. No force caps were increased. Blade-friction and
  dynamic effects are omitted, so the audit cannot authorize execution.
- Current blocker is not only cut triggering: select/verify a retained grasp
  with adequate force/moment capacity and repair/qualify spring/contact dynamics.
  All156/158/159 are FAILED full sequences despite valid native seam events.
  Full regression **3,586 passed in132.96 s** in
  `data/sim_physics/regression_20260913_target_antiwindup_v1.log`.

### 2026-09-13: Retention-capacity and near-junction grasp-layout work

- Added `sim_physics/grasp_wrench.py`: bounded offline six-axis static wrench
  audit using supplied points/normals, both finger identities, an inscribed
  eight-sided friction cone and the unchanged <=0.5 N per-finger contact budgets.
  It reports the minimum worst-finger budget utilization; it neither commands
  that force nor changes any controller/safety limit. Static patch assumptions,
  unknown native provenance and absent actuator/dynamic qualification are explicit.
  Solver timeout/error, nonfinite/unbalanced output and invalid geometry fail closed.
  This is a diagnostic utility, NOT an integrated retention certificate.
- Added bounded initial `--grasp-pitch` (+/-60 degrees) ONLY to the complete
  isolated downward CLI protocol. Rotation is about the jaw closing axis, to
  use more of the original pads' long dimension. The palm offset now uses the
  actual pitched axis; zero pitch preserves existing behavior. Source geometry,
  material, force limits, grasp clearance, native scene screens and release
  checks remain unchanged. This is a new pose proposal, not a running pose reset.
- Native160 repeats159 with a farther180 mm grasp as a moment-arm diagnostic.
  It is rejected BEFORE physics by torso/left-wrist self clearance (-21.165 mm
  conservative bound). It is NOT a replacement for the requested near-junction
  grasp. Other farther-grasp offline proposals meet attached-leaf obstructions.
- Ideal rest-shaft/pad search suggests a larger contact span and better normal
  orientation reduce required normal load, but existing-station near-junction
  pitch proposals fail IK/inter-arm checks. A coordinated stationary robot-pose
  search is underway. No rejected pose is executed or relabelled safe, and no
  new force cap is introduced. The predictor/contact-model issue remains open.
- Diagnostic evidence (not training data): `retention_candidate_audit_20260913_seed101.log`,
  `ideal_grasp_search_20260913.log`, `retention_layout_search_20260913*.log`,
  `retention_coordinated_ik_20260913.log` and native160/report.json under
  `data/sim_physics/`. Source assets, dataset releases/reviews/splits, training,
  hardware commands and Windows configuration are unchanged. No reboot requested.
- Native161 tests an85 mm pitched grasp from a coordinated station/torso/arm
  proposal. The complete native zero-step check confirms an actual left-wrist
  versus Leaf002 obstruction, with47 robot actor controls checked before/after.
  Physics steps=0; the candidate is rejected, not an executed grasp/cut failure.
- Full regression: **3,626 passed in120.73 s**
  (`regression_20260913_grasp_capacity_v2.log`). The first run found three mocked
  report fixtures lacking the new zero-default pitch field; those fixtures were
  updated without weakening assertions or changing the physical guard limits.

### 2026-09-13: Holding preload and common-aperture comparison

- `--retention-preload` is an explicit isolated comparison, not a new default:
  desired support0.12->0.24 N; non-gravity PD cap0.15->0.30 N and controller
  backoff0.30->0.40 N. The native hard per-finger all-contact0.5 N, total motor
 0.8 N, blade0.5 N, slip3 mm and penetration1 mm guards are UNCHANGED. This
  uses available actuator budget; it is not a calibrated safe plant load.
  Explicit effort and antiwindup share the same profile, including exact
  float32 cap representation. No hidden body/velocity/root actuation is added.
- Native162 repeats159 with preload. Cut occurs at38.320833 s with signed
  resistance0.200503..0.200988 N,29.167 ms dwell and tool upper0.319616 N.
  Bilateral contact now persists after release, but slip3.020804 mm at38.375 s
  stops the run after54.17 ms. No retention/withdrawal/deposit qualification.
  Full regression **3,638 passed in131.22 s** (`regression_20260913_retention_preload_v1.log`).
- Added `--symmetric-finger-closure`, also isolated/explicit/feedback-only.
  One aperture is regulated from mean opposing support; finger load asymmetry
  no longer commands a translating grasp center. Shared antiwindup intersects
  both measured PD intervals with the original geometric bounds. If no common
  interval exists, it refuses the command. This is joint-target control, NOT
  a native gear, object weld, pose reset or substitute grasp detector. The
  Model A v1.2 URDF has two prismatic finger joints, with no mimic relation;
  physical mechanical coupling has not been independently calibrated here.
- Native163 repeats162 with the common aperture. Cut at34.620833 s; bilateral
  contact persists to failure at34.858333 s (237.5 ms), with slip3.083448 mm.
  Final target pair remains symmetric(-1.89882,+1.89882 mm). The distal body
  speed reaches2.29164 m/s. Thus correcting commanded center drift is useful
  but does NOT fix plant spring/contact dynamics or establish retention.
  Full regression **3,648 passed in131.81 s** (`regression_20260913_symmetric_closure_v1.log`).
- All four new native comparisons160..163 remain FAILED qualifications (160/161
  reject before stepping). Evidence is retained. Native spring/root handling
  and support-capable grasp geometry remain under investigation; do not launch
  collection/training or advertise reliable physical cutting from these runs.

### 2026-09-13: Independent strain audit and bounded contact diagnostics

- Added `sim_physics/rod_strain.py`: rest-relative body-frame rotation, bending
  surface strain and torsional shear estimates, independent of native joint-q.
  These are small-strain/quadratic engineering diagnostics, NOT tissue damage
  calibration or a complete energy balance. Native163 frame energy agrees with
  the scalar estimate: 0.000672 J at3.5 s, 0.436331 J at18 s and0.212153 J just
  before cut. Maximum estimated torsional shear reaches0.202278 at18 s, while
  maximum bending strain is0.022717. Small grasp translation hides substantial
  torsional wind-up. Post-release retention remains unqualified.
- Added owned three-link unit-conditioning runner/helpers `coupon_units*.py`.
  Metre/centimetre conversions preserve SI geometry, mass, inertia, native
  force-based K/C, contact material and guards. Native drive type, coefficients,
  targets and zero external actuation are read back; source assets are hashed.
  Native168(metre),169(centimetre),170(metre/zero authored CFM) ALL fail the held
  coupon: residuals about0.00251 N.m and joint-speed RMS0.0186..0.0193 rad/s.
  Native171 centimetre FREE planar coupon passes, with decaying modeled energy
  and speed RMS2.31e-14 rad/s. This is NOT a held/full-plant qualification.
  Unit conversion and CFM changes are not promoted to production. No native CFM
  getter is available; the zero-CFM comparison is explicitly USD-authoring only.
- Root coupon164 shows an externally constrained floating root and fixed root
  can fall when their external support is disabled, but the fixed-base tensor
  metadata remains stale: NOT a qualified topology transition. Full-chain,
  contact-free native-spring coupon165 is unstable before release. Its original
  post-release comparison is INVALID (weaker-layer edit did not disable the
  composed joint); runner corrected to use session edit and bounded speed guards,
  not rerun. Neither finding authorizes changing the production root model.
- Native172 repeats163 without the optional2 mm pull. Native173 additionally
  tests new opt-in `--blade-dwell-feedback`, a seven-contact-sample minimum for
  bounded feed control ONLY. Raw signed force, direction, geometry,25 ms dwell,
  0.5 N blade/finger caps,3 mm slip and1 mm penetration gates are unchanged.
  Both timeout without a cut; native173 maximum slip0.638738 mm. New feedback
  is NOT a proven repair or a new default. No timed/filtered release is used.
- Offline compact-branch layout searches for seed19/SubStem50 and seed31/SubStem46
  reject on exact robot limits or scene/leaf clearance; no rejected path is run.
  No leaf, wrist camera, collider or source geometry was removed to gain a pass.
- Evidence under `data/sim_physics/`: `frame_strain_audit_20260913_native163.log`,
  `coupon_units_20260913_native168`..`native171`, root probes164/165,
  `bimanual_downward_20260912_native172`/`native173`, and
  `compact_branch_pose_search_20260913*.log`. Full regression **3,696 passed in
  146.12 s**, `regression_20260913_strain_dwell_v1.log`. Diagnostics are not
  training records; reviews, splits, hardware, running external jobs, Windows
  configuration and source assets are untouched. No reboot requested.

### 2026-09-13: Native torsion / implicit bending comparison rejected

- Added default-OFF `--native-torsion-trial`, restricted to the complete isolated
  diagnostic. Full original K/C remain in the predictor, but only bending
  efforts are submitted; original native torsional drives participate in the
  PhysX contact solve. Exact joint triples, force-drive type, zero rest targets,
  native coefficients/caps and zero external torsional actuation are checked.
  No root force, mass/inertia inflation or source/geometry change is introduced.
  Spring-work reporting excludes unmeasured native torsional work explicitly.
- Native174 repeats163 with this split and FAILS before grasp at0.029167 s
  (seven recorded steps): maximum body speed76.832 m/s triggers the unchanged
  guard. Zero blade contacts. This split is REJECTED as a physics repair; do not
  enable it for demos, production or training. Default springs remain unchanged.
- Evidence: `data/sim_physics/bimanual_downward_20260912_native174/`.
  Focused tests50 passed; full regression **3,718 passed in138.77 s** in
  `regression_20260913_split_springs_v1.log`. These code tests do not override
  the native failure. The installed fixed-tendon schema excludes spherical
  joints, so that alternative is not being silently substituted either.

### 2026-09-13: Contact-load estimate, process status and faster guarded feed

- Added a small-coupon-only one-step normal+friction load estimate for the
  implicit spring predictor. Original-order signed impulses map through the
  native COM Jacobian; only intrinsic spring effort is submitted, never contact
  forces or root wrench. Contiguous fixed-rate guard-accepted observations are
  required. Native175 FREE passes720 steps and whole-run energy decay. Native176
  HELD fails equilibrium (0.771570 mN.m residual) and velocity (0.062368 rad/s RMS).
  It is NOT integrated into the robot or presented as a repair. Full regression
  before subsequent changes: **3,731 passed in158.73 s**, `regression_20260913_lagged_load_v1.log`.
- Fixed two diagnostic launchers returning zero through Kit fast shutdown even
  when their JSON reported failure. `qualification_exit.py` requires the exact
  pass state, affirmative assessment and no error/cleanup fault. Native177
  repeats176: physics still fails, but now the launcher correctly exits nonzero
  (python.bat wrapper1; requested Kit code2). Existing failed evidence is retained.
- Added default-OFF `--compliant-blade-rate`, restricted to the existing isolated
  1000 N/m seam-contact fixture. Loading/near speed capped at0.3 mm/s (1.25 um
  per240 Hz tick), feedback gain0.00125 m/(N.s), original0.5 mm/s backoff and
 0.22..0.28 N hold band. All original raw force, direction, dwell, geometry,
  slip, penetration and contact guards remain. This is a tested control-rate
  comparison, NOT an analytical force-overshoot guarantee or a spring repair.
- Native178 repeats163 with only this rate profile: contact-qualified seam event
  at16.729167 s (163:34.620833 s), slip0.729458 mm at release. Still FAILS at
 16.954167 s (225 ms later), slip3.104875 mm and body speed2.221159 m/s while
  bilateral contact persists. Elastic proxy at cut0.212265 J and maximum
  torsional-shear estimate0.142492 remain nearly identical to163. Faster cutting
  does not solve the underlying constitutive/contact response or retention.
- Seed59/SubStem48 compact-branch offline proposals also reject on self/plant
  clearance; no rejected pose is executed. Physics assets and surroundings are
  not modified to hide obstruction. No collection/training or hardware control.
- Evidence: `data/sim_physics/lagged_contact_20260913_native175`..`native177`,
  `bimanual_downward_20260912_native178`, compact seed59 log. Full regression
  **3,747 passed in139.54 s**, `regression_20260913_compliant_rate_v1.log`.
  Reliable full grasp-cut-retain-withdraw/deposit remains NOT COMPLETE.

### 2026-09-13: Bundled Newton reference solver comparison (not promoted)

- Added an isolated, CPU-only Newton1.2.1/Warp1.13.0 coupon runner, using the
  Isaac6.0.1 bundled packages without enabling its auto-switching extension,
  starting Kit, changing production physics or downloading another backend.
  Original source-report mass/inertia/K/C, small-coupon geometry, zero gravity
  and240 Hz are retained and native arrays checked. AVBD Rayleigh damping is
  explicitly C/K seconds (not physical C passed into the wrong units).
- Contact law is explicitly DIFFERENT: unilateral compliant penalty with
  compression-only damping and IPC regularized friction; not PhysX patch
  friction or a calibrated plant/gripper material. Static-friction equivalence
  is unverified. No contact-history snap, body replay, root weld or external
  spring-force override is used. Original coupon acceptance gates remain.
- Native179 was an adapter startup error (None numeric seed, zero steps), fixed
  before180. Native180 FREE passes720 steps, including full-run modeled energy
  decay (initial7.971460uJ, maximum6.016374uJ). Native181 HELD at32 iterations
  FAILS equilibrium/velocity/global wrench balance: moment residual0.202076mNm,
  joint velocity RMS0.013755rad/s. Native182 at128 iterations still FAILS:
  residual0.314862mNm, velocity0.015083rad/s. All pads observed; no plant test.
- Native183 tests a more localized IPC low-speed regularizer (0.1 rather than
  10mm/s, same mu/K/C) and FAILS the unchanged1N per-body contact guard on the
  first step (maximum1.109510N). Not promoted or treated as a physical repair.
- Native contact telemetry preserves shape order and force-on-body1 sign,
  splits normal/friction exactly once, and retains both solver application
  points rather than inventing a shared point. Shared coupon oracle supports
  that optional second point; existing single-point PhysX rows are unchanged.
  Its maximal-coordinate arithmetic label is explicitly not PhysX execution
  or an independent articulation position sensor in the Newton receipts.
- Evidence: `data/sim_physics/newton_coupon_20260913_native179`..`native183`.
  Focused130 passed; full regression **3,763 passed in137.66s**, recorded in
  `regression_20260913_newton_reference_v1.log`. These are code validations,
  not full robot grasp/cut qualification. Production defaults are unchanged.
- A bounded-memory read-only audit of173/178 confirms the no-pull173 trial has
  lower but still nonzero twisting energy; at18s it is3.873mJ. This motivates
  a separate grasp-and-hold/no-reposition comparison with the already bounded
  feed profile, not weakening any release, contact, slip or retention guard.
  Cut reliability, withdrawal, retained material and deposit remain incomplete.

### 2026-09-13: No-pull cut rejection isolated to moving attachment reference

- Native184 combines the already bounded faster feed with NO optional2mm
  pull. Grasp remains bilateral; maximum recorded slip0.653260mm. Still FAILS
  at the30s loading timeout with no cut (20,939 eligible edge contact rows).
  No full retained-cut/withdrawal/deposit qualification is claimed.
- Exercised the previously unit-tested native-C/explicit-K coupon split.
  Native185 FREE passes720 steps and whole-run modeled energy decay.
  Native186 HELD PGS128/0 fails equilibrium (1.293492mNm residual) and velocity
  (0.045929rad/s RMS). Native187 TGS128/0 is worse:20.039702mNm,2.396180rad/s.
  Neither split nor solver-mode change is promoted into production.
- Added default-OFF `--blade-friction-budget`, requiring the complete isolated
  compliant feed and seven-sample minimum feedback configuration. Total force
  includes friction:0.26N normal with mu=.5 can require0.39N total, conflicting
  with the old0.32N CONTROL backoff. The comparison uses0.40N backoff, retaining
  the0.50N HARD full-contact guard,0.28N normal backoff, raw signed cut force,
  geometry/direction/dwell, bounded feed increments and failure timeout. This
  is not an analytical force-overshoot guarantee or permission to release.
- Native188 still FAILS without a cut; maximum slip0.657997mm. It is NOT a
  working cutting configuration. Feedback changes alone are insufficient.
- Added read-only `ShearGate.diagnostic` and native `gate_diagnostic`: exact
  rejected predicates, measured relative step and pending dwell/travel. The
  original release predicate/evidence is unchanged; input faults cannot leave
  a stale qualifying diagnostic. Receipts never authorize a physical cut.
- Read-only replay of178/184/188 reproduces EVERY logged signed force and
  dwell value and their exact cut/no-cut outcome (zero mismatches).184 and188
  reach at most ONE consecutive valid sample. With edge contacts,184 fails
  signed load3862 times and reverse relative motion3013 times;188 reduces load
  failures to989 but still has2975 reverse-motion failures.178 reaches seven
  valid samples and its original16.729167s event, then fails retention as above.
- Decomposing188's native transforms identifies the moving SEAM, not backward
  blade commands: at15.325s the blade advances0.129736um while the attached
  seam advances165.509443um. Other inspected steps alternate about75..170um
  seam motion while blade increments are submicrometre. The externally anchored
  floating root is not behaving like the exactly constrained root assumed by
  the current predictor. This is measured attachment/reference motion; it does
  not by itself prove a particular installed PhysX defect or a full energy law.
- Next engineering check is a genuinely fixed root and state-preserving detach
  with valid fresh tensor metadata; the existing fixed-root release prohibition
  is NOT bypassed. Native189's exploratory refresh fails with stage ID0 before
  post-release stepping, identifying a launcher-context identity error. No
  production stage, running user job, source asset or dataset is changed.
- Source-preserving seed19/50 proposals at an isolated0.35m plant lift also
  fail existing arm/path checks; none is executed or called reachable.
- Evidence under `data/sim_physics`: `bimanual_downward_20260912_native184`,
  `native188`, `native_damping_20260913_native185`..`native187`,
  `replay_shear_rejections_20260913.log`, `reverse_motion_audit_20260913.log`.
  Full regression before diagnostics:3774 passed145.98s; after diagnostics:
  **3,784 passed in134.03s**, `regression_20260913_shear_diagnostics_v1.log`.
  These are software checks; reliable cutting/retention is still NOT COMPLETE.

### 2026-09-13: Fixed attachment, continuous detach, and remaining free-root instability

- Native190's small topology check and native192's original full source branch
  transition fixed-to-free with no measured pose/velocity jump and the correct
  free-root mass dimensions. Native193 repeats the full branch WITHOUT global
  tensor reset: unchanged state, correct 54x54 free mass, 0.3049565m root fall
  over 0.25s. These are contact-free diagnostics, not grasp/cut qualification.
- Added default-OFF `--fixed-root-contact-hold`: complete isolated original
  implicit/flat/PGS240/128-0-or-8 diagnostic, attached-only, no reposition or
  right-arm motion. Native194 completes 12,000 ticks / 50s with all runtime
  guards passing, verified grasp and maximum slip 0.146941mm. Maximum attached
  root translation difference is 9.698nm (float32 readback). Its overall
  bimanual result intentionally remains failed: a hold-only control cannot
  pass cut/retention/withdrawal gates.
- Added default-OFF `--fixed-root-cut-trial`, still restricted to the complete
  isolated instrumented no-reposition fixture. `root_transition.py` flushes the
  qualified seam edit without stepping/resetting, creates a fresh explicit-stage
  tensor view, verifies original coordinate/link inventory, free mass layout,
  plant and robot poses/velocities/commands/mass/inertia/K/C/caps, then rebinds
  only plant articulation and cached spring controller. Existing robot/body
  views must also pass continuity. No fixture.bind(), warmup, weld, force replay,
  pose setter or velocity setter; partial transition failure is latched and
  stops before another tick. Direct fixed-root diagnostic release stays banned.
- Native195 produces a raw blade-qualified seam release at16.770833s; every
  recorded no-step plant/robot change is zero. All49 logged release/post-release
  grasp adapter checks agree. This is NOT successful retained cutting: at
 16.970833s, full finger loads0.525118/0.672567N exceed the unchanged0.5N limit.
  Maximum slip2.166239mm. Modeled spring potential rises from1.00851mJ at release
  to3.14015J in0.20s, corroborated by body-derived strain. Native contact cache
  preservation is not independently certified; Python controller/contact
  histories are preserved and next-step raw callback/tensor reconciliation
  remains mandatory.
- Tested original native damping / explicit stiffness as an independent
  post-release diagnostic in `native_damped_springs.py`; original K/C, masses,
  caps, no added root/contact force. Native196 fails the0.25rad diagnostic
  bound even WITHOUT contact despite C >= hK/2. That inequality is NOT a
  certificate of the installed native solver. This alternative is rejected
  and has NO benchmark/robot integration or production default.
- `--diagnostic-free-root-dynamics` is read-only and restricted to the fixed
  cut trial. Native197 repeats the same cut, then stops on disagreement between
  native COM Jacobian-mapped and native reported body velocities. Added exact
  rejected snapshot capture, never returning invalid values as a prediction.
  `kinetic_consistency.py` separately reconstructs generalized inertia/kinetic
  energy from reported body mass, COM inertia, transforms and Jacobians; it
  does not repair state or infer a new physical law. Native198 captures this
  mismatch for further diagnosis; coordinate-error vs solver-lag cause pending.
- Native198 stops before any post-cut spring step: angular mapping residual
  1.40707e-4rad/s. Independent body/Jacobian reconstruction agrees with the
  free mass matrix (maximum diagonal-scaled difference4.075e-7); shifting the
  Jacobian to/from body origin makes this much worse. This rules against a
  simple COM-origin/mass-layout correction, NOT proof that this relatively
  small velocity residual explains the large energy growth. Native199 compares
  the Jacobian/velocity residual before and after the no-step topology edit.
- Native199 finds EXACTLY the same residual before and after release:
  linear3.45956e-5m/s, angular1.40707e-4rad/s, intrinsic Jacobian change ZERO.
  Thus the no-step transition does not introduce this mismatch. Do not claim
  the residual proves a root-coordinate bug or explains the3J growth. The
  remaining physics investigation is the spring/contact split with a freely
  moving, finger-constrained branch; do not retain a hidden fixed root after
  cutting, weaken contact/direction guards or substitute computed velocities.
- Evidence: `data/sim_physics/root_transition_20260913_native190/192/193/196.json`
  (individual files with those suffixes), `bimanual_downward_20260912_native194/`,
  `native195/`, `native197/`, `native198/`, `fixed_root_audit_20260913.log`.
  First broad regression exposed ten legacy mock-call incompatibilities;
  restored the unchanged positional release call when no adapter is requested.
  Rerun: **3,857 passed in126.02s**, `regression_20260913_fixed_root_v2.log`.
  Final broad rerun: **3,861 passed in127.86s**,
  `regression_20260913_fixed_root_v3.log`; final velocity-target continuity,
  failed-attempt latching and release-evidence checks: **64 passed in3.25s**.
  Tests do not establish native retained-cut reliability. New fixed-root
  options remain opt-in diagnostics, not a GUI or production default.
- Reliable free-branch retention, withdrawal and deposit remain incomplete.
  Production scene/assets/fidelity, physical force/slip/collision limits,
  datasets/splits/reviews, training and hardware are unchanged. No reboot or
  host-memory preflight bypass. These diagnostic cuts remain simulated joint
  release, not calibrated tissue fracture or training-approved demonstrations.

### 2026-09-13: Post-cut comparisons and native retention preflight

- Native200's original implicit three-body held coupon at1920 Hz completes
  but fails static moment balance (2.23937 mN.m); velocity RMS4.50814 mrad/s
  alone is insufficient. Increasing rate is not an established physics fix.
- Explicit original-native-drive fixed-root HOLD201 fails at0.295833s before
  grasp on unwanted leaf/palm load0.586531 N.202's longer80mm approach is
  rejected by the existing3mm self-clearance screen before physics. Neither
  comparison changes the production spring or contact model.
- Added isolated `--native-drives-after-cut`: only after a successful checked
  fixed-root release, remove external implicit spring effort and restore the
  cached original native K/C. Exact native state, mass/inertia and rest/velocity
  targets must remain unchanged. No double spring actuation, new gains or
  state setters. Native drive work is explicitly unmeasured, not zero.
- Contact-free source-branch203 completes60 free steps with original native
  K/C. Extended207 stops at free step123 on the5m/s diagnostic speed bound
  (5.03500m/s, max angle0.06893rad): this is consistent with ordinary gravity
  free fall, NOT evidence of a new spring explosion. The original60-step203
  sampled total kinetic+gravity+quadratic-elastic proxy declines by0.6916mJ;
  no contact/retention qualification follows from that control.
- Full-robot204 again cuts at16.770833s, then FAILS retention after162.5ms:
  slip3.127006mm and elastic proxy1.10832J.205's128/8 iteration comparison
  also loses the grasp.206 adds an explicit rigid-pad CONTACT-LAW control
  (original geometry/friction/caps, pad compliance disabled only in the owned
  session): cut16.741667s, failure141.667ms later, slip3.196983mm and elastic
  proxy1.15472J. This does not fix retention and cannot qualify the original
  compliant-pad model. All options remain diagnostic/default-OFF.
- Read-only current-patch gravity audit195/206 finds inadequate static moment
  capacity for the8.613447g detached branch under the existing0.5N per-finger
  normal-plus-friction budgets.195 utilization is5.518 for loaded points and
  1.960 even including zero-load points optimistically. These are fixed-patch,
  inscribed-eight-sided-friction-cone calculations, NOT proof that every grasp
  or changing contact patch is infeasible. They do not explain all numerical
  energy growth or authorize higher forces.
- Implemented `retention_preflight.py` and default-OFF
  `--require-retention-screen`, scoped to the original isolated fixed-root
  diagnostic. Before knife planning, bind current post-fetch frames, timestep,
  exact body/finger identities, native masses/COMs and guard-accepted bilateral
  contact. Only compressive contact on the detachable shaft contributes;
  zero/tensile, support, leaf and unrelated contacts never support approval.
  Static gravity capacity is a prerequisite, NOT dynamic retention, cutter
  load, actuator feasibility or calibrated-tissue certification.
- Native208 exposed an integration timing issue: the just-established grasp
  reference had no subsequent measured slip yet. It stopped without knife
  motion. The new protocol now waits one actual physics tick (no fake zero
  slip).209 correctly rejects at6.8375s, step1641, utilization5.52834, seven
  compressive rows; zero blade contacts, no knife plan, no cut. This is a
  successful negative-control rejection, NOT a completed bimanual sequence.
- Offline near-junction pad-span searches retain all original geometry and
  blade clearance checks. Releasing an unnecessary fixed-world right-ready
  pose constraint yields more self-clear candidates, but tested65/75mm pitched
  candidates still meet leaf/wrist or cut-corridor obstruction. Rejected paths
  are not executed. Grasp/layout selection and spring/contact consistency both
  remain open; no qualified retain/withdraw/deposit result is claimed.
- Evidence: `data/sim_physics/implicit_contact_20260913_native200/`,
  `bimanual_downward_20260912_native201/202/204/205/206/208/209` (individual
  directories), `root_transition_20260913_native203.json` and`native207.json`,
  `current_retention_audit_20260913.py`, and`retention_free_ready_search_20260913*.log`.
  Focused138 tests pass. Broad regression: **4,939 passed,2 skipped,47 subtests
  passed in187.36s**, `regression_20260913_retention_preflight_v1.log`
  (`sim_physics`, `sim_data`, `greenhouse_sim`). Tests are not native physical
  qualification. Dataset/splits/reviews/training, hardware, source assets and
  production greenhouse fidelity are unchanged. No reboot or memory-check bypass.
  Final explicit240Hz/step-clock preflight guard:139 focused tests pass in1.32s.

### 2026-09-13: Source-preserving grasp access and consistent pre-shaping

- Tightened the static retention prerequisite to require the same9.81m/s^2
  gravity assumed by its load model; a zero-gravity diagnostic cannot reuse
  that result.140 focused tests pass after this additional scope check.
- Native210 tests original seed37_full/SubStem_41 at55mm grasp arc,20mm
  cut arc and19mm finger/cut-plane clearance. Zero-step full native startup
  passes; the approach screen rejects finger1/main-stem contact at7.5% of
  the approach.211's75mm grasp rejects self-clearance against the parked
  knife before physics.212 changes the initial right shoulder by-5 degrees;
  self-clearance passes but zero-step scene checking rejects finger1 against
  the target's original Leaf_019. No rejected approach is executed.
- The approach had used a50mm total jaw opening for a shaft only a few
  millimetres wide. Added diagnostic `--pregrasp-half-aperture-m` (default
  unchanged25mm per side): require shaft radius+2mm clearance, initialize
  native fingers and drive targets consistently, screen the entire actual
  approach/closure, and preserve this commanded opening in force feedback
  and antiwindup. This changes a proposed robot pose/command, NOT gripper
  geometry, physical joint limits, actuator limits or contact mechanics.
  CLI scope requires the complete fixed-root explicit-feedback diagnostic
  and current-contact retention preflight. Runtime pose/velocity setters,
  welds and reduced collision/force/slip guards are not introduced.
- Native213 repeats210 with8mm half-opening. Zero-step native startup passes
  (484 queries). At0.9s the planner rejects the approach at5% against the
  target's Leaf_019/Segment_005; static main-stem coarse rejections encountered
  before that are cleared by native queries. The changed opening does NOT
  solve all access constraints. No grasp, knife motion or cut is qualified.
- Alternative-angle/parent-plane and independently parked right-arm offline
  proposals remain subject to original self/leaf/main-stem/camera/cut-plane
  checks. No geometry is removed to manufacture a clear grasp. Offline
  candidates do not establish native contact or physical retention.
- Tests:220 focused pass in9.05s, including in-memory USD initial native
  joint/drive readback and unchanged joint limits/source layer, complete
  approach/closure sampling, default compatibility and bounded feedback.
  Broad regression: **4,961 passed,2 skipped,47 subtests passed in183.26s**,
  `data/sim_physics/regression_20260913_pregrasp_v1.log`.
  Evidence: individual `bimanual_downward_20260912_native210` through
  `native213` report/trajectory directories in `data/sim_physics`.
- Reliable post-cut retention, withdrawal and deposit remain incomplete.
  Native tests still model force/direction-qualified joint release, not
  calibrated tissue fracture. Production greenhouse/assets/fidelity,
  datasets/reviews/splits/training and hardware are unchanged. Host reserve
  passed before213 (~29.3GiB commit headroom); abnormal kernel paged pool
  remains. No reboot, unrelated-process stop or memory-check bypass.
- Native214 moves the grasp to50mm with the same8mm half-opening. Startup
  again passes, but the approach intersects the original main stem at2.5%
  of the path. No grasp/cut follows. Final scalar/type validation regression:
  **223 focused tests pass in9.21s**. This includes invalid text, boolean,
  complex and nonfinite opening rejection before physical commands.

### 2026-09-13: Physical shaft identity, preload and bounded reference correction

- Native215's more side-on seed37 proposal passes offline left-arm screening
  but full zero-step native startup rejects torso5 against target Leaf_026.
  No physical motion. Offline checking of one arm is not whole-robot clearance.
- Selected another ORIGINAL source target, seed13_full/SubStem_43, with its
  original six leaves/main-stem context in the existing isolated branch fixture.
 50mm grasp arc,20mm cut arc,8mm pre-grasp half-opening. Native216 passes
  startup and full left approach/closure planning, but never acquires a grasp:
  the evidence core hardcodes selected+/-one shaft segment while the collision
  planner uses the real finger span. Segment004 under the same fingers is
  incorrectly classified as outside the connected target; zero-load proximity
  from that segment keeps `stem_only=False` and prevents further closure.
- Added explicit `--physical-grasp-span`: compute candidate identities from
  CURRENT native pad/collider poses and authored contact offsets, walking
  every contiguous currently enabled adjacent shaft connection. Stop at a
  missing link or outside-span segment; do not jump to a folded-back limb.
  Protected support, leaves, body aggregates and other branches are excluded.
  Exact inner-face/material-side witnesses, impulse signs,20mN support,
  selected-tensor crosschecks, native all-contact/slip/penetration guards and
  topology invalidation remain. Mode is in the binding hash/telemetry.
  USD connectivity still cannot prove an unreflected native joint break.
- Native217 acquires a verified bilateral grasp at6.7625s after that correction.
  The immediate retention audit rejects at6.766667s (four compressive rows,
  static utilization30.504). Initial measured supports are only0.03787/0.02803N,
  not the controller's intended0.24N preload. No knife planning or cut occurs.
- Added `--settle-retention-preload`: original symmetric mean-support target
 0.24N and existing+/-0.03N deadband, both fingers compressive, <=2mm/s finger
  speed, current guard-accepted telemetry, slip<3mm,48 consecutive240Hz steps.
  Readiness does NOT certify retention. Preserve full subsequent approach/
  stroke durations; do not jump into cutting after waiting.218's initial3s
  timer expires while preload is still growing; use the existing13.5s absolute
  acquisition budget instead, without extending it.219 still times out:
  supports0.20707/0.19101N; attached grasp remains bilateral and maximum slip
  is4.109 micrometres. Both trials keep the right arm parked.
- Diagnosed a separate controller-reference inconsistency: with200N/m PD,
  the legacy1mm nominal target bias can request only about0.2N at the nominal
  shaft surface.219 reaches its minimum command gap1.997514mm while measured
  gaps are3.011462/2.955121mm; reconstructed PD is0.204586/-0.191407N.
  This is NOT the same quantity as measured physical shaft penetration.
- Added explicit `--effort-bounded-grasp-target` with original0.30N/200N/m
  reference-bias bound1.5mm, slow contact approach and existing antiwindup.
  No increase to desired support, PD/total actuator/all-contact limits, or the
  actual native1mm penetration guard. Screen the entire resulting commanded
  closure. Reports separate actual reference floor/bias from legacy geometric
  compression configuration. Native220's per-step force_closure records contain
  the actual1.5mm bias; its older robot summary minimum-field formatting was
  corrected afterward (not a retrospective edit to that run).
- Native220 passes native preload dwell at12.85s: support0.223288/0.206976N,
  bilateral grasp verified6.733333s, maximum slip4.18513 micrometres. Static
  retention then rejects with10 compressive rows, detached mass8.22748g and
  utilization2.07694 against original0.5N normal-plus-friction budgets. This
  is a stable ATTACHED grasp, not proof of free-branch retention or cutting.
- A read-only rotated-current-patch proposal predicts utilization0.90152 at
  -45degrees and0.92269 at-30degrees. This assumes the rotated patch can form;
  it is neither a measured new grasp nor dynamic/collision authorization.
  A source-preserving-45degree approach passes offline left collision checks
  and is submitted as native221 for full native validation.
- Evidence: native215 through220 individual report/trajectory directories,
  `compact_branch_free_ready_20260913_v7.log`, `v8.log`, and
  `current_patch_orientation_20260913_native220.log` in`data/sim_physics`.
  Span regression:4,980 passed,2 skipped,47 subtests in183.98s. Preload stage:
 5,001 passed,2 skipped,47 subtests in191.16s. Effort-reference broad regression:
  **5,010 passed,2 skipped,47 subtests in186.66s**
  (`regression_20260913_effort_reference_v1.log`). Final report-field and
  integration subset: **435 passed in5.48s**. Test passes are NOT physics
  qualification. All new options remain opt-in diagnostics/default-OFF.
- Reliable retained cut/withdrawal/deposit and full-greenhouse qualification
  remain open. No material/source-asset edits, production fidelity reduction,
  dataset/review/split/training or hardware changes. No reboot, unrelated
  process stop, or memory-check bypass. Every native test used the launch guard.
- Native221 tests the-45degree proposal: full native startup and grasp corridor
  pass; bilateral grasp6.695833s; preload dwell11.866667s at0.209410/0.210852N;
  current static retention prerequisite **passes**, utilization0.862586 with
  12 compressive rows. Maximum attached slip6.33772 micrometres. This improves
  measured grasp-load feasibility without increasing physical force limits.
  Knife planning then rejects the source branch: a transverse cut cannot be
  within the existing30-degree downward cone. No knife motion, cut or free
  retention is claimed. Direction/target feasibility must also pass; the
  source13 successful attached grasp is NOT a completed cutting benchmark.

### 2026-09-13: Preload convergence and coupled grasp/tool access

- Further ORIGINAL-asset diagnostics, with all physical/collision gates intact:
  native222 returns to seed101_full/SubStem_41,65 mm grasp. Bilateral contact
  6.545833 s, preload12.0125 s, but static retention utilization5.507846 fails;
  maximum attached slip0.150163 mm. Native223's rotated/pitched candidate is
  rejected at zero-step startup for torso5 against distal target leaves.
  Native224 keeps the previously tested base/torso and changes only the grasp
  proposal: native startup/approach pass, grasp6.641667 s, preload12.191667 s,
  retention utilization3.383998 fails; maximum attached slip0.248194 mm.
  None of these trials reaches knife motion or cuts.
- Selected original seed41_full/SubStem_41 (shallow enough for the unchanged
  downward cutting cone). Native225:55 mm grasp,20 mm cut,8 mm pre-grasp
  half-opening,-45 degree radial approach,roll180. Native startup and the
  left approach pass; bilateral grasp6.625 s; maximum attached slip0.027971 mm.
  Preload times out at13.5 s with support0.208739/0.215352 N and only two of
  48 required consecutive ready ticks. The old force controller stops at the
  same +/-0.03 N boundary used for readiness, and small force fluctuations
  repeatedly reset the independent dwell. No cut planning in225.
- Added default-OFF `--preload-force-servo`, requiring the complete existing
  effort-bounded explicit-finger/antiwindup/static-retention diagnostic.
  The symmetric mean-force controller uses a tighter +/-0.01 N deadband and
  nominal0.5 s force-error-to-reference-velocity time constant at the existing
  200 N/m PD gain. Closing speed stays <=0.5 mm/s. Same0.24 N desired support,
  0.30 N PD,0.8 N total actuator,0.5 N all-contact and1 mm actual penetration
  limits; same geometry/freshness rejection and5 mm/s emergency backoff.
  This is a controller hypothesis validated below, not hardware calibration.
- Native226 is the SAME source/initial pose as225 with only that servo enabled.
  Bilateral contact verified6.6125 s; preload ready8.954167 s with48 measured
  ticks and support0.212011/0.218740 N. Maximum attached slip0.028183 mm.
  Current static retention prerequisite PASSES: utilization0.902101,
  12 compressive rows,7.585276 g detached inventory. It assumes fixed patch,
  friction0.5 and force redistribution; it does NOT certify actuator feasibility,
  cutter loading or dynamic retention. No force limits were raised.
-226 then reaches cut planning, but all400 candidate tool corridors reject
  before arm IK: primarily right camera versus left fingers, or fitted knife/
  camera versus the original main stem. Tools/plant are not removed, resized
  or repositioned to hide the conflicts. No knife motion or seam release.
  Small source-preserving grasp-station/orientation proposals are being
  screened jointly with tool access, not judged by attached grasp alone.
- Evidence: `data/sim_physics/bimanual_downward_20260912_native222` through
  `native226` reports/trajectories; `current_patch_orientation_20260913_native226.log`
  and `coupled_grasp_access_20260913_v1.log` are OFFLINE proposal diagnostics,
  not measured new contacts or full-robot path certificates.
  Initial servo/controller/CLI regression:133 passed in0.99 s. Broad suite:
  **5,023 passed,2 skipped,47 subtests in203.71 s**
  (`regression_20260913_preload_servo_v1.log`). Reliable retained cut/withdrawal/
  deposit remains open; software regression is not physical qualification.
  Production presets, source assets/fidelity, datasets/reviews/splits/training
  and hardware remain unchanged. No reboot or host memory-check bypass.
- Native227 keeps that source/approach but places the grasp65 mm from the
  junction (cut remains20 mm). Full native startup/approach pass; verified
  grasp6.729167 s; preload9.033333 s at0.213629/0.218442 N; static retention
  prerequisite passes with utilization0.814660. Maximum attached slip0.067946 mm.
  All400 fitted-tool corridors still reject before IK; no right motion/cut.
  Offline60..70 mm station comparisons likewise find no coarse-clear complete
  tool stroke. Further station proposals must retain full native validation;
  do not alter the fixed cut point or remove camera/plant collision geometry.
- Further candidate diagnostics are geometry proposals, NOT additional
  successful grasp/cut episodes. The offline combined hand/tool search now
  checks the original left hand against the target leaves during approach and
  closure as well as the full sampled right-tool stroke. Full-arm IK, static
  native geometry, actual contact and retention remain separate requirements.
  Unresolved coarse static boxes are not proof of actual collision.
- Native228 tests seed67_full/SubStem_43,95 mm grasp,pitch20,20 mm cut,
  nominal zero blade axial offset. Two offline combined tool strokes clear,
  but full native startup rejects finger1 against original Leaf_032/Leaf_033.
  No physics motion or cut. The local-only screen did not certify this
  initial leaf clearance; adding that check removes all sampled survivors
  for this branch, without deleting leaves or changing collision filters.
- Native229 tests original seed19_full/SubStem_41,55 mm grasp,pitch40,
  20 mm cut and zero blade offset. Offline left approach/closure and seven
  complete tool strokes clear. Full native startup/left approach pass;
  measured preload ready8.920833 s at0.252262/0.187568 N. Attached maximum
  slip0.052075 mm. Static retention fails: detached mass12.385444 g,
  14 compressive rows, utilization1.456021, required normal-plus-friction
  bounds0.570360/0.728011 N versus unchanged0.5 N per-finger limits.
  Right planning/motion and cut are refused. More geometric access does not
  make this longer branch physically retainable by that measured patch.
- Evidence in `data/sim_physics`: native228/229 report directories;
  `compact_branch_free_ready_20260913_v13.log` through`v18.log`,
  `coupled_grasp_access_20260913_v1.log` through`v8.log`, and
  `local_hand_tool_search_20260913_v2.log` through`v6.log`. These local searches
  are privileged source-geometry diagnostics, not VLM observations or accepted
  demonstrations. No new training/collection/review/split decisions.
  Native230 is a frozen post-grasp access diagnostic on the lighter original
  seed41 target, using live static-actor query refinement and stopping before
  ANY right motion. It repeats the227 grasp and finds zero clear bare-tool
  corridors among the tested original downward proposals.699 native queries,
  379 coarse rejections cleared,316 retained rejections; final query validation
  and cleanup pass. Robot positions/velocities/targets/submitted-force arrays
  are exactly unchanged across the probe. No proposed grasp or cut is
  authorized. See `native230/native_local_hand_tool.json` under the same
  `bimanual_downward_20260912_` prefix. This is an access diagnostic, not a
  physical cutting failure or proof about every possible manipulation strategy.

### 2026-09-13 ? Coupled grasp/access investigation and complete-target preview

- Reliable cut/retain/withdraw/deposit is still NOT established. No dataset,
  review, split, collection, training, hardware or source-asset changes here.
- Native231 completes an attached source101 grasp, then the diagnostic preserves
  the actual failed static retention result (utilization5.512214). A frozen
  native-static geometry query finds two bare-tool corridors and hypothetical
  hand configurations. It unconditionally stops before right motion; these are
  not successful grasps at the proposed frames or demonstrations.
- Native232 proposes an80 mm grasp/pitch40/roll0 with a newly optimized torso.
  The arm-only offline preview passes, but the full native zero-step screen
  rejects torso5 versus original distal leaves. No motion is allowed.
  FIX: `whole_robot_target.py` now independently checks ALL supplied robot
  colliders against the complete current target snapshot during every sampled
  left approach and closure pose. It runs alongside the unchanged bounded
  static-context screen. The finite target subset has no arm-cache workspace
  restriction, but explicitly does NOT certify surrounding scene coverage.
  Only left-finger/selected connected-shaft pairs retain the existing expected
  grasp allowance; torso, parked right arm, head, palm and cameras do not.
- Native233 keeps the previously checked base/torso and tests source101/41 at
  80 mm, pitch40, roll0. Native startup and grasp approach pass; grasp6.720833 s,
  measured preload8.9875 s; static retention prerequisite passes with0.827933
  utilization,12 compressive rows and8.613447 g detached inventory. Maximum
  attached slip0.015345 mm. This is NOT dynamic retention or cutter-load proof.
  All400 original fitted-tool corridors reject before IK, notably left palm
  versus knife ArcContacts/Part_00. No cut or right motion.
- FIX: source-enclosing YZ-oriented boxes for the thin-X convex arc partitions
  replace their XYZ planning boxes in the existing downward fitted-tool screen.
  Every original source vertex/convex hull remains enclosed with1 micrometre
  padding. Native colliders, visible knife, mounting, material and all existing
  collision margins are unchanged. This reduces empty-wedge conservatism;
  it is not an exact native convex-cooking certificate.
  Native234 repeats233: identical attached slip and preload, but still no
  complete clear corridor. Tighter bounds alone do NOT fix this configuration.
- Native235 repeats the attached grasp and uses a frozen read-only native
  static query to explore nearby depths/pitches/stations. Two bare-tool paths,
  16 local hand proposals,105 combined checks,90 left-path rejections in3.375 s.
  Query errors absent; robot q/v/targets/submitted-effort arrays unchanged.
  These omit full arm IK/new native contacts and never authorize right motion.
  Native236 is the subsequent physical candidate, not yet qualified here.
- Validation:50 focused whole-target/native-cleanup/aperture tests;30 focused
  arc/source-enclosure/target tests; complete `sim_physics` suite **4,008 passed
  in151.07 s** (`data/sim_physics/regression_20260913_whole_target_arc_bounds_v1.log`).
  Test counts overlap and are not independent episodes. Original guarded
  launch policy retained; no reboot or memory preflight bypass.
- Native reports: `data/sim_physics/bimanual_downward_20260912_native231` through
  `native235`; native235 includes `native_local_hand_tool.json`. Offline proposal
  logs `compact_branch_free_ready_20260913_v19..v22` and
  `coupled_grasp_access_20260913_v9..v10` remain diagnostics, not training data.

### 2026-09-13 ? Grasp depth and checked right-wrist approach schedules

- Native236 (80 mm/pitch20/roll0/depth110 mm) passes setup but never verifies
  grasp.151 loaded rows are beyond a flat cylinder's side extent; repeated
  existing backoff keeps opposing support near16 mN, below20 mN. These are
  NONZERO impulses, not stale zero-load points. No geometry/force gate relaxed.
- Native237 changes the commanded depth to125 mm with a newly solved initial
  left posture. Grasp and static retention prerequisite PASS (utilization
  0.852996); attached maximum slip0.023368 mm. A native-screened vertical cut
  corridor at15-degree plane tilt exists. The right arm's simultaneous rotation
  and translation approach, however, sweeps the knife arc into MainStem_26.
  The planner repeats effectively identical Cartesian transits for redundant
  endpoint elbow guesses, exhausting its60-second WALL budget (549 queries,
  below20,000 query-count cap). No right motion/cut was authorized.
- FIX, default-OFF `--staged-downward-transit`: same native retention fixture
  can screen nominal complete right-tool transits BEFORE arm IK. Transit NEVER
  inherits the cutting stroke's intended-seam contact allowance. Current options
  include simultaneous, orient-then-straight, bounded60/120 mm side waypoints,
  and30/60/90 mm above-endpoint waypoints. Each uses <=2 mm/1-degree wrist samples.
  Full arm IK/inter-arm/self/plant checks remain at <=1-degree joint samples;
  interpolation must stay within0.5 mm of its own checked Cartesian segment.
  Via-waypoint approaches are NOT described as one straight approach; the actual
  cut remains straight/downward with the original extension and angular gates.
  The initial endpoint IK is only a pose seed. Staged mode solves transit from
  the actual configured parked branch, then rebuilds and screens the ENTIRE
  cut stroke from its exact terminal joint vector. It does not repeat a fixed
  wrist sweep for unrelated endpoint guesses or skip actual-arm path checks.
- Native238 repeats237 with the first two schedules: same grasp, blocked approach
  rejected in2.225 s instead of timing out.239 adds above waypoints (4.174 s),
  240 adds side waypoints (6.604 s). All tested approaches reject before arm IK:
  original main stems, held plant or left hand remain in the way. These are
  measured planning-latency improvements on a REJECTED case, not successful cuts
  or a full-greenhouse frame-rate claim. No force/collision threshold changed.
- A task-ready initial right pose is being evaluated separately to isolate
  contact/retention from the difficult neutral-pose approach. Native241 tests
  a15-degree-plane ready pose10 mm above the pre-cut endpoint. Its offline
  left self/target preview passes but native zero-step startup rejects the
  blade plate versus MainStem_27. No physics motion.242 is a subsequent candidate
  at the previously screened pre-cut endpoint; not qualified in this entry.
  Initial-pose proposals do not solve general approach planning or authorize
  runtime pose snaps. `near_ready_pose_search_20260913_v1.log` is a failed file-
  lookup diagnostic;v2 contains the actual source-preserving pose proposals.
- Regression: **4,033 physics tests passed in151.60 s** in
  `regression_20260913_staged_transit_v1.log`; **640 sim_data tests plus47 subtests
  passed in42.69 s** in`regression_20260913_sim_data_unchanged_v1.log`.
  Focused113-test wrist/planning suite also passes; counts overlap.
  Reliable physical cut/retain/withdraw/deposit remains open. No source assets,
  dataset decisions, production/GUI defaults, hardware or host OS settings changed.

### 2026-09-13 - Task-ready grasp/cut achieved; released-branch retention still fails

- Native242 uses the original source101/SubStem_41,80 mm grasp,pitch20,roll0,
  depth125 mm,20 mm cut and independently startup-checked right pre-cut pose.
  Native grasp, preload and cut planning pass. Planning takes1.763 s, without
  an arbitrary-neutral-start approach. Measured blade-load joint release occurs
  at18.720833 s. This is the signed-edge brittle-seam proxy, NOT calibrated
  tissue cutting (measured loading travel only0.359 micrometres).
- At18.904167 s,242 stops on1.378 mm finger penetration (>1 mm), with1.894 mm
  slip (<3 mm), finger contact upper bound0.363 N (<0.5 N). The frame-based
  elastic-energy estimate rises from0.001676 J at release to0.331324 J.
  It agrees with joint-coordinate strain, so this is not just a q-readout
  artifact. Explicit spring work accounting reports an apparent constitutive
  energy source~0.329 J after18.7 s; it is NOT complete native energy balance.
- Native243 changes only the existing original-K/C native-drive post-release
  comparator. Same grasp/cut time; failure at18.920833 s on3.052 mm slip.
  Frame-based energy estimate0.484580 J. Native drive torque/work is unmeasured;
  no zero-work or calibrated-energy claim. Restoring native drives alone fails.
- Native244 repeats243 with the already supported128 position/8 velocity
  solver iterations. Cut18.7375 s, maximum slip3.058 mm, retention fails again.
  These are negative controlled comparisons, not promoted physics defaults.
- Passive diagnostic improvement: instrumented fixed-root cut trials record
  original signed normal/friction rows for all observed target contacts BEFORE
  robot-only load classification, including plant/support contacts. Existing
  source bodies already have zero-threshold native reporting; no geometry,
  filtering, material, force, timing or execution gate is changed. Replayed/gap
  snapshots and callback faults fail; an empty stream does not certify absence.
  Native245 repeats243 with this instrumentation to investigate released loads.
  Focused passive recorder/contact suite:118 passed (0.40 s).
- Solver investigation follows NVIDIA's articulation stability guidance:
  https://docs.omniverse.nvidia.com/kit/docs/omni_physics/latest/dev_guide/guides/articulation_stability_guide.html
  and https://nvidia-omniverse.github.io/PhysX/physx/5.6.1/docs/Articulations.html .
  These document competing contact/drive constraints; they do not establish
  the root cause of this particular failure. Original safety thresholds remain.
- Native245 matches243's cut time and failure slip exactly with passive tracing.
  The trace includes a distal Leaf_004 / link_torso_5 contact peaking0.109313 N
  at18.766667 s. It is below the existing0.5 N unwanted-contact stop threshold,
  but is not desired target manipulation and attached startup clearance does
  not guarantee post-release clearance. No observed non-robot support impulse
  explains the failure; callback completeness remains unproven. The torso
  contact may contribute, not yet a demonstrated sole root cause.
- Full physics regression after trace integration: **4,042 passed in148.60 s**
  (`data/sim_physics/regression_20260913_release_trace_v1.log`). Offline roomier
  body-pose proposals preserve both wrist goals and original plant geometry;
  these still require native zero-step and full execution checks.

### 2026-09-13 - Native retained cut, endpoint audit, and explicit cut-only mode

- Scope remains isolated source seed101_full/SubStem_41 with original main
  stem, complete selected petiole/leaves, full RB-Y1 A v1.2 and original
  camera-aligned knife. No source assets, dataset labels/splits, hardware,
  memory policy or production/GUI defaults were changed.
- Native246 moves the torso away while preserving the two wrist goals. No
  torso contact is observed, but at240 Hz retention still fails at18.7 s
  (3.140 mm slip). The earlier torso/leaf contact was not the sole cause.
- Explicit --cut-convergence-trial propagates480 Hz through contact impulse
  conversion, native spring handoff, finger controller, blade feedback,
  retention timestamps and settling dwell. Original SI gains/materials,
  force/slip/penetration limits and dwell durations are preserved. This is
  an opt-in diagnostic, not a converged/calibrated physics claim.
- Native247/248/249 repeat the same physical trajectory through50 s. The
  force-qualified seam releases at18.495833 s; the branch remains held for
 31.504 s afterward. Maximum slip2.209229 mm, final2.190898 mm. A streamed247
  audit finds maximum finger-contact bound0.459287 N and minimum separation
 -0.349496 mm, below the unchanged0.5 N/1 mm guards. There are TWO separate
  single-tick (~2.08 ms) nonbilateral samples after release; do not claim
  perfect uninterrupted contact. All native guards pass. These deterministic
  repeats of ONE fixture do not establish perturbation/greenhouse reliability.
- Endpoint correction --native-station-park-reference evaluates the unchanged
  RIGHT JOINT goal in the measured station.247 used the configured torso frame
  and reported1.559 mm error despite nearly exact right-joint return.248's
  measured-station endpoint error is0.497 micrometres. No actual wrist is used
  as its own goal and the0.5 mm/0.005 rad tolerances are unchanged.
-249 also propagates the explicitly selected complete capsule sphere-cover
  query into the final clearance check (same1 mm margin, positive controls,
  fresh epoch, budget and cleanup). This clears the coarse arm/main-stem
  rejection, but exposes a retained-branch/BladePlateContact overlap at the
  final waiting pose. Withdrawal STILL FAILS. A new checked retreat is needed;
  returning to the pre-cut ready pose is not enough once the plant has moved.
- User revised the task to allow right-only cutting when left grasp obstructs
  the tool. Added a separate --right-only-cut-trial engineering mode. It uses
  exact signed edge contact, original load/dwell/direction/location gates and
  the same checked topology release, but NEVER fabricates held=True or zero
  grasp slip. Release independently checks matching strategy evidence.
  The left must be parked/open/unloaded with current native readback. Its
  final screen has NO intended-grasp collision allowance. Retention/deposit
  are explicitly not expected or credited; dropped material is expected.
- The pure strategy selector prefers a clear bimanual plan, permits cut-only
  only with a clear independent right-only plan and allowed drop, and inspects
  or skips unknown/blocked cases. It is NOT integrated as automatic GUI retry.
  A runtime collision/sensor fault does not authorize fallback through contact.
- Native250 executes an unheld force-qualified cut at13.272917 s, then stops
  because falling material reaches the nearby inactive left hand.251 moves
  the initial left park from20 to80 mm clearance but rejects a mismatched
  park command on the first step: the grasp-path IK's first node was not the
  exact original park reference. Added hold_left_park using original checked
  joint targets and existing gravity/native-drive code, without state setters.
- Native252 repeats the80 mm parked case with that correction. Cut occurs
  at13.254167 s without grasp; then the detached Leaf_004 contacts torso_4
  with1.144307 N normal-plus-friction, exceeding the existing0.5 N guard.
  It stops. This is NOT a successful cut-only sequence or a verified drop.
  Unheld fall clearance remains a separate physical blocker.
- Regression:4,106 passed after clock changes;4,115 after station reference;
 4,160 passed in151.67 s with cut-only mode, and4,161 passed in156.17 s after
  exact park-command correction. Full logs are
  data/sim_physics/regression_20260913_{rate_all_v2,station_park_all_v1,
  cut_only_all_v1,cut_only_all_v2}.log. Focused suites overlap these counts.
- Durable isolated replay: from examples/greenhouse_sim, Isaac Python
  -B -m sim_physics.ground_truth_trial --mode bimanual --output NEW_DIRECTORY
  (or --mode right_only). Source profile ground_truth_trial.json records the
  exact fixture/options; these are FAILED qualification reproducers, not a
  recommended successful demo preset. New output directories are mandatory.
- Current physical model remains articulated rigid beam/leaf contacts plus
  force-qualified preauthored seam release. Tissue-fracture calibration,
  continuous collision certification, full forward slicing, autonomous deposit
  and hardware safety are NOT established. No VLM action-training data is
  approved by any of these runs.

### 2026-09-13 - Freeze gate requested before returning to VLM integration

- Immediate priority: complete the accessible ground-truth bimanual
  grasp -> cut -> retain -> clear withdrawal baseline. Cut-only is a separately
  scored fallback and additionally needs unobstructed falling-material space.
- Before a manipulation freeze: (1) one complete native sequence with fresh
  final clearance; (2) a recorded repeat/perturbation matrix covering multiple
  reachable target configurations with no safety-limit breaches; (3) repeat
  in the actual visible greenhouse with original surroundings/contact fidelity;
  (4) verify reset/retry cannot reuse stale plans or cut/grasp evidence.
  Unit tests, a released joint, or one successful pose are insufficient.
- Proposed first qualification matrix:10 trials each on3 explicitly chosen
  reachable configurations, including bounded placement/initial-state
  variations. This is an engineering acceptance set, NOT proof of universal
  reliability. Inspect/skip must remain distinct from successful completion.
- A qualified grasp/cut/retention release must clearly exclude any unverified
  autonomous deposit, calibrated tissue mechanics, general VLM control and
  hardware operation. Continue the separate deposit goal later; do not call a
  limited manipulation freeze the complete deleafing task.
- Save engineering checkpoints independently of a frozen release. The user's
  today deadline does not change collision, force, contact, provenance or
  repeatability requirements. No reliable completion ETA is established yet.

### 2026-09-13 - User-scoped grasp/cut and unheld cut milestone

- The user narrowed today's milestone: support left grasp + right cut and
  right-only cut; released material may land on the torso. This supersedes the
  landing/withdrawal prerequisite in the preceding freeze plan ONLY for the
  limited cut-action checkpoint. Do not call it a complete robot task, safe
  deposit, universal reliability, calibrated tissue fracture or hardware proof.
- Added explicit --cut-action-trial, exposed as ground_truth_trial
  --milestone cut_action. It requires the existing isolated fixed-root/native
  release profile and original startup/static native checks. Default behavior
  and full-sequence reports remain unchanged. No source assets, material
  parameters, native collision filters, force/slip/penetration bounds, sensor
  cadence, dataset decisions, training or physical robot state were changed.
- released_debris.py binds to the actual strategy-verified blade event and
  composed disabled seam. Only exact detached target colliders contacting
  exact torso/base/wheel bodies get a separate record-only classification.
  Attached supports, neighboring plants, active right arm/tool and both hands
  retain their existing guards. All original raw impulses and normal/friction
  accounting remain available; reset or changed event identity fails closed.
- Limited action success requires a checked right plan, correct native support
  evidence (actual stable grasp OR explicit parked/open/unloaded cut-only),
  measured blade release, >=2 s post-cut observation and all execution guards.
  Native trials still run50 s. Full mechanism gates are preserved under
  full_sequence_state/full_sequence_qualified, not overwritten by the new score.
- Native253 (right-only): cut13.254167 s, completed50 s, all cut-action AND
  right-only mechanism gates pass, including released separation and measured
  right withdrawal. The left stays parked/open/unloaded: no fake grasp/slip.
  Maximum separately recorded debris/torso-base contact10.069878 N. This is
  accepted task scoring, NOT a safe-impact or material-damage certificate.
  Log contains317,176 lines matching native getMaterialFromInternalFaceIndex after
  the fall; no matching warning in249 or254. Root cause/physical relevance is
  unresolved. Keep logs and do not silently suppress or label these harmless.
-253 wrote a correct successful action report but returned an error exit:
  the launcher/Kit success allowlists omitted new action and cut-only states.
  Added one report_exit_code helper used by both return and app.close. It
  checks requested mode, exact complete evidence gates, source integrity and
  absence of execution fault. Full-sequence requests cannot claim limited
  action success; unverified success labels cannot pass. Legacy exits retained.
- Native254 (bimanual): process exit0; limited cut-action PASS through50 s.
  Cut18.495833 s, maximum slip2.209229 mm; native retention passes through
 31.504 s after release. All original native execution guards and source-asset
  integrity pass. Final held-branch/BladePlate clearance STILL FAILS, so the
  full_sequence_state remains failed_bimanual_qualification. This is consistent
  with247..249, not hidden by the narrower milestone. No torso debris load.
- Evidence: data/sim_physics/bimanual_downward_20260912_native{253,254}/report.json,
  corresponding .log files, and streamed bimanual_trajectory.json in each run.
  Full regression:4,182 passed161.49 s before exit fix;4,210 passed160.77 s
  after exit fix. Logs regression_20260913_cut_action_all_{v1,v2}.log.
- Diagnostic capture correction: legacy 'grasp' screenshots were scheduled
  before verification and could even be named in cut-only mode. They now
  follow measured grasp verification; severed/post_cut_2s follow actual cut
  time. Optional --capture saves paused15 Hz viewport evidence with step,
  timestamp, camera, cut/grasp state and unchanged-plant-pose receipt, plus a
  final image. They are explicitly NOT synchronized RGB-D or training data.
- Current checkpoint is one isolated original branch, complete robot and
  original knife, task-ready prepositioned right arm, cut at20 mm and grasp
  at80 mm material arc. No reposition, arbitrary target solution, automatic
  fallback, reset qualification, intact-greenhouse cut or real-time claim.
  Measured no-render real-time factor:249 ~0.192,253 ~0.240; do not present
  these instrumented runs as responsive online control or VLM demonstrations.
- Native255 repeats the bimanual action WITH15 Hz rendering and11 native PNGs:
  limited action PASS, process exit0, same18.495833 s cut and2.209229 mm peak
  slip as254. Independent bounded-memory audit of all24,000 records confirms
  zero native guard failures, two isolated post-cut nonbilateral samples, and
  final bilateral contact with2.190898 mm slip. Full final withdrawal remains
  failed. Source assets unchanged. Images/receipts are in the native255 folder.
  The plant-side grasp image visibly shows the shaft between the finger pads.
  Wide cut images are too distant to prove edge contact visually; added knife
  and plant-side detail captures for the next rendered trial. Native contact
  evidence, not an image filename, remains the basis of the release result.
- Native256 repeats right-only WITH15 Hz rendering and14 native PNGs: process
  exit0, limited action and all right-only mechanism gates PASS through50 s.
  Cut13.254167 s,6519 native edge contacts, maximum recorded debris load
 10.069878 N, same as253. No left grasp/slip is fabricated; every image receipt
  explicitly records left_grasp_verified=False. Original source assets intact.
  Plant-side images show the branch present before cutting and absent after
  detachment, with the left hand open. The knife-side view is partly occluded
  by the wrist/main stem: this view alone does not prove the precise cut line.
  Native material-index warnings repeat after the fall; unresolved caveat.
  Instrumented real-time factor ~0.235 (NOT real-time online control).
- Regression after event-bound screenshots:4,217 passed148.51 s in
  regression_20260913_cut_action_all_v3.log. Final detail-view suite:4,221
  passed150.49 s in regression_20260913_cut_action_all_v4.log. A subsequent
  exit-mode consistency assertion also passes all28 focused exit tests:
  requesting right-only cannot inherit a legacy grasp-mode success state.
- Saved engineering checkpoint, not a general frozen manipulation release:
  two requested actions reproduce with/without rendering on ONE identical
  ground-truth fixture. Outstanding: bimanual final clearance, post-fall native
  material warnings, perturbation/multiple-target and reset/retry coverage,
  arbitrary-start approach, and full-greenhouse requalification/performance.
  VLM training/data, other running applications and OS settings were untouched.

### 2026-09-13 - Visible one-shot native cut inspection

- User requested a visible launch of the current working actions. Added explicit
  ground_truth_trial --watch / benchmark --watch-cut-trial instead of opening
  the older grasp-only demo or bypassing the trial's reset restrictions.
- Separate native Isaac window,15 Hz rendering, Run-once/Stop and existing
  camera-view buttons. Same complete isolated480 Hz action profile, original
  assets, controls and guards. No automatic run, reset/replay, hardware calls,
  dataset operations, OS changes or takeover of an existing Isaac process.
- Idle robot/plant pose and timeline are checked before Run; changed timeline
  during execution aborts rather than incrementing stale physics timestamps.
  After the one trial, pause preserves the final scene for viewing. Results
  distinguish limited cut success from final withdrawal; no full-task claim.
- First native watch launch v1 exposed an observer initialization-order error:
  fixture.robot exists only after the actual probe binds its control interface.
  Corrected idle observation to create a read-only native articulation view;
  it does NOT bind/set controller gains or execute an extra physics step.
  Regression test explicitly uses a fixture with no pre-bound robot attribute.
- Visible launch cut_action_watch_20260913_v2 reached CUT_WATCH_READY with an
  Isaac Sim Python6.0.1 window and one-shot bimanual ready state. User then
  started the run from the panel; live native telemetry reports progress.
  Reports/logs: data/sim_physics/cut_action_watch_20260913_v{1,2}.
- Focused suites132 passed before the idle-reader correction; all14 watch
  tests passed after it. Full physics regression4,234 passed136.05 s in
  regression_20260913_cut_watch_all_v1.log (before the extra idle-reader test).
  Ready-state launch is not by itself a completed native cutting result.

### 2026-09-13 - Revised arc-up cutting milestone (IN PROGRESS, not frozen)

- User acceptance replaces the earlier limited side-edge success: A) approach,
  left grasp with cutter clearance, correctly oriented right approach and
  downward cut; B) independently demonstrated right-only cut; both first in
  isolation then in the supplied greenhouse; C) responsive simulation without
  deleting physics, surroundings or original visual detail. VLM dataset work
  is on hold until A/B/C are demonstrated. Tonight is the requested deadline,
  not a qualification claim. Misalignment must not yield a clean-cut label.
- Diagnosis of native255/watch: world stroke was downward but the semantic
  source -X side edge stood the plate upright and pointed the arc sideways.
  Initial right pose was effectively the cut precontact pose (no visible
  approach). Strength-only seam release credited only ~1.09 micrometres of
  loaded travel. Those prior passes do NOT satisfy the revised milestone.
- Added explicit experimental `source_lower_rim_v1` with
  `loaded_downward_lower_rim_seam_v1`. Uses the original straight outer lower
  rim; source +Z arc maps to up. All original Blade/Arc visual triangles and
  source layers remain untouched. Requires source rim contact, signed native
  load 0.2..0.5 N, 25 ms consecutive qualified contact, original shaft/edge
  alignment and axial tolerances, arc-up/downward within 5 degrees, and BOTH
  >=0.3 mm relative loaded advance and >=0.3 mm actual world-down movement.
  Moving a target into a stationary blade, pressure only, tiny jitter, wrong
  orientation, unrelated surfaces and insufficient force cannot trigger it.
  Plant release independently validates the additional motion/axis evidence.
  Old side-edge recipes remain explicit legacy diagnostics, not upgraded demos.
- This still releases a preauthored seam. It does NOT model calibrated sharpness,
  tissue fracture, fraying or partial incision. Bad alignment is rejected, not
  relabeled as a realistic partial cut. Native success of the NEW mode is not
  established. The old force-hold feed is also not yet a qualified slicing law.
- Native257 (profiled): left grasp verified, then all 50 corrected tool corridors
  rejected before right IK/contact. Native258 repeats exactly the same 4,271
  steps / 8.897917 s, grip slip and contact summary after validation optimization.
  Measured profiled tick wall time 51.264 -> 41.449 s (~19.1% lower); RTF .174 ->
  .215. Grasp evaluator cumulative time 12.713 -> 5.014 s. No solver, force,
  material, dt, collision or visual changes in this comparison. Not real-time.
- Optimization: validate owned immutable current pose batches once, then reuse
  them for each cylinder-side witness; retain every row's proximity/force checks.
  Replace general-purpose allclose calls with equivalent finite absolute-bound
  checks at identical tolerances. Native257/258 source/physics configuration is
  identical. Existing 266 shaft/adapter/contact tests passed after the change.
- Interactive clock now optionally schedules rendering by wall time (15 Hz in
  watch), without extra physics steps or skipping controls; addresses UI input
  starvation when physics is slower than real time. Tested with a fake clock;
  new native watch responsiveness is not yet measured.
- Native259: proposed arc-up start with 30-degree lower left approach failed
  complete native startup (right upper arm vs original distal leaf), zero motion.
  Native260: keeping original right start, 30-degree left approach grasped but
  failed static retention prerequisite; no knife planning/execution. Native261
  failed prelaunch due an old right-only model allowlist, corrected to accept
  only the matching lower-rim/model pair. Native262: right-only left park open
  and unloaded, but all 50 corrected corridors rejected; no cut/IK execution.
- Added opt-in `--source-wrist-contacts`: eight-cell upper-bound partitioning
  per source bracket/adapter, preserving every original triangle, replacing
  opaque convex-decomposition carriers with explicit convex pieces shared by
  planner and native physics. Original visible hardware unchanged. Twenty
  focused source-preservation/lower-rim tests pass. Native263's new source-complete
  wrist representation passes startup, but the same 50 right-only corridors are
  rejected before IK (plate/parent, camera/parent or parked finger/arc conflicts).
  The partitions alone do not resolve access; no new cutting pass is claimed.
  This is an experimental contact representation, not verified CAD fastener fit.
- Offline helper `sim_physics.lower_rim_layout` proposes arc-up starts with a
  visible approach distance, checks whole tool and left path, and labels every
  proposal native_tested=False/motion_authorized=False. Coarse box rejection
  is not proof of physical infeasibility. Native guards remain authoritative.
- Evidence: `data/sim_physics/bimanual_downward_20260912_native257..263`,
  `lower_rim_layout_20260913_v1..v3.log`, regression logs prefixed
  `regression_20260913_lower_rim`. First broad regression exposed legacy mock
  API compatibility failures; corrected without weakening new-mode checks.
  All 124 affected legacy pipeline/planning tests then passed. Full rerun:
  4,654 passed, 2 skipped in 161.96 s (`regression_20260913_lower_rim_v2.log`).
- Remaining: safe arc-up approach and joint posture, loaded-stroke control and
  uncalibrated failure-law limitations, native A/B repeated successes with
  visible evidence, full-greenhouse A/B/C requalification, post-fall material
  warnings. No source assets, datasets, training, hardware or OS settings changed.

## 2026-09-13 - Actual source cutting edge audit and correction (IN PROGRESS)

- Critical root cause found by inspecting the native knife image and three
  orthographic views of the unmodified CAD: the importer's widest-X component
  named `Blade` is the mounting plate. The actual narrow beveled straight bar
  below the curved support belongs to the component named `Arc`. Both older
  side-edge and the first lower-rim mode targeted the mounting plate, not that
  bar. Historical releases must NOT be presented as intended-blade validation.
- Added explicit `source_crossbar_edge_v1`: derive the actual sloped tip from
  source cross-sections (3 mm wide bar, ~2-degree straight lower edge), separate
  its central source triangles into `CrossbarContact`, retain every other arc
  surface in collidable convex partitions, and retain both mounting-plate
  carriers. No visual mesh, STL, mass, source plant or physical dimensions are
  changed. Only the exact sharpened strip of the crossbar can supply eligible
  cutting contact; the curved support and mounting plate remain noncutting.
  Legacy modes remain explicitly reproducible, not promoted as realistic cuts.
- Public `sim_physics.ground_truth_trial` now selects the corrected crossbar and
  loaded-downward law by default, still experimental and allowed to fail closed.
  Reproducing old mounting-plate results requires `--historical-mounting-plate`.
  The internal recipe-vector helper remains unchanged for historical comparisons;
  the source edge in each report distinguishes the two. No claim that the new
  default already passes the manipulation milestone or fixes full-greenhouse UI.
- The existing stricter downward seam law also accepts this explicitly named
  source edge: actual arc-up/downward orientation, transverse alignment, exact
  seam identity, signed measured normal load, full load cap, dwell and measured
  net world AND relative loaded travel remain mandatory. Still an uncalibrated
  joint-release approximation, not measured tissue fracture/partial incision.
- Corrected a controller mismatch: the old pressure-only feed stopped at its
  force target. The stricter model now proposes bounded 0.15 mm/s loaded
  advance with the same 0.28 N normal/0.40 N full-load backoff and 0.50 N hard
  cap. Fresh geometry/support verification gates loading; bad alignment/contact
  holds or backs off. Commanded advance never counts as measured cut evidence.
  This may still stall on physical load/penetration limits; no force increase
  or reduced 0.3 mm cut-travel requirement was used to manufacture success.
- New source area-preservation/open-window/edge mapping checks initially passed
  5/5; the existing blade/arc/gate regression selection passed 147 tests and the
  separate feed selection passed 140. Full regression passed 4,680 tests with
  two skips in 178.64 s (`regression_20260913_crossbar_v1.log`). Additional
  source-frame negative controls then passed with the expanded 11-test crossbar
  suite: zero/insufficient/excess load and sideways/upside-down blade rejected.
- Native264 (old lower rim, left park tilted 30 degrees) rejected all 50 tool
  corridors. Native265 (corrected crossbar, original seed101/SubStem_41) passed
  startup and found three locally clear full tool strokes, but rejected the
  old parked-wrist rotation/transit into them. No IK motion or cut occurred.
  Proposed higher arc-up starts failed IK at the existing stance; stance/path
  diagnosis continues with all guards. No A/B/C completion or deadline promise.
- Evidence: `data/sim_physics/knife_source_edge_audit_20260913_v1.png` and
  `knife_source_edge_audit_20260913_v2_corrected.png` (code component labels vs
  actual edge); native264/265 reports; `crossbar_ready_proposals_v1.log`;
  `lower_rim_source_candidate_survey_v1.log` and `lower_rim_other_sources_v1.log`.
  Candidate surveys are read-only source audits, not collection/review changes.
  VLM work remains paused until isolated and greenhouse A/B/C are demonstrated.
- Native266 (corrected crossbar bimanual, original stance) again verified the
  left grasp, max slip 0.01671 mm, and found one local tool corridor at the
  opposite heading; old ready-pose transit remained blocked. Native267 used a
  new arc-up start 50 mm above pre-contact: IK and full left self-path passed,
  but the native zero-step startup screen found right upper-arm contact with
  the original distal `Leaf_005`. It correctly refused all physics motion.
  Next: screen same-wrist-pose elbow alternatives against the frozen native
  scene, then re-run grasp/approach/loaded cut. All new-mode cuts remain unverified.
- Public crossbar launcher also enables the source-complete wrist partitions
  used in these native trials. Public/historical launcher and source geometry
  tests passed 23/23; watch-panel tests passed 14/14. Historical profile vectors
  remain unchanged; no existing jobs, datasets, hardware or OS settings changed.
- Added `--native-startup-pose-search`, an explicit read-only same-wrist elbow
  diagnostic. It keeps the stopped zero-step scene, checks the full dense left
  self-path and every robot collider against the complete native environment,
  retains before/after positive actor controls, and always stops before motion.
  A proposal cannot repair the currently authored bad spawn; relaunch/startup
  and the complete action must still be checked. Final control failure revokes
  any provisional proposal. Native268 checked six feasible elbow alternatives;
  all retained the right-upper-arm/Leaf_005 overlap. No actor, collision filter,
  stage pose or physics step changed; all final native controls passed.

- Expanded the frozen-scene elbow diagnostic with 16 deterministic bounded
  global IK seeds after the connected local family. Disconnected solutions are
  still proposals only: full left self-path, native scene queries and final
  positive controls are mandatory, with no motion or authored pose changes.
  The 53 startup/search/lifecycle/clearance tests pass. No search result changes
  the saved action profile automatically.
- Native269's 80 mm robot-station adjustment still failed upper-arm/Leaf_005
  startup clearance. A disconnected elbow-up branch at the original station
  fixes that particular overlap, but native270 found camera/MainStem_28 contact
  at the 50 mm raised start; native271's lateral start hit MainStem_27 instead.
  All three stopped at zero physics steps. Native272 is testing the lower,
  pre-contact elbow-up pose; completion and an extended approach remain pending.

- Native272 clears the original scene at startup, verifies the left grasp and
  completes right Cartesian/stroke planning with the corrected crossbar. It
  reaches actual eligible edge contact but times out without release. Peak
  signed resistance 0.2552 N, peak full contact bound 0.4397 N; qualifying load
  persists for 21.65 s, but net loaded travel reaches only 0.06185 mm of the
  required 0.3 mm. Max grasp slip 0.02664 mm. Arc-up/downward/alignment checks
  pass during loaded contact. This exposes a contact-model incompatibility,
  not a missing contact callback: a linear 1000 N/m elastic obstacle stalls
  inside the unchanged force envelope instead of yielding to a blade.
- Added OFF-BY-DEFAULT `--seam-contact-yield` for isolated, 480 Hz corrected
  crossbar trials only. This uncalibrated process-zone experiment starts at
  measured 0.23 N qualified edge load; source indentation sets the origin.
  Positive nonincreasing secant stiffness follows measured net world AND
  relative downward travel. No time/jitter/desired-motion credit, no healing,
  no geometry/pose/joint changes, and no release or force-threshold override.
  Wrong alignment, lost support, excessive force and foreign camera/finger
  contact prevent updates. All original native guard and release gates remain.
  Only the already-bound two seam shaft surfaces change; material updates use
  the installed Isaac compliant-contact USD backend, NOT a claimed native
  material readback. This is NOT volumetric fracture, partial mesh incision,
  a calibrated force law, or proof of clean tissue-cut quality.
- Fifteen yield negative/positive/ownership tests pass; the 12 original
  material tests also passed before two added negative cases. A broad prior
  `sim_physics` regression passes 4,296 tests in 225.19 s, before yield changes
  (`regression_20260913_crossbar_v2.log`; different scope from the earlier full
  greenhouse regression). Native273 is the first yield/material-response test;
  its result and native A/B qualification are pending.
- Native272 headless timing: 0.20975 real-time factor, median native step
  5.46 ms and post-fetch checks 3.48 ms at 480 Hz. This is NOT real-time. The
  full diagnostic trace is 1.51 GB for 20,831 steps; telemetry/memory overhead
  also needs optimization without skipping guards or reducing scene fidelity.

- Native273 is the FIRST measured release with the actual sharpened crossbar:
  t=33.5125 s, 0.30004 mm net loaded downward advance, signed resistance
  0.20014..0.23183 N, peak full load 0.41105 N. Grasp slip at release is
  0.02452 mm. It fails after 1.79 s of post-cut observation because a leaf of
  the retained branch touches a finger: the old stem-only flag commanded
  opening despite continuing opposing shaft support. This is not a success
  for the whole sequence. Native273 predates the extra yield angle sanity
  check/post-release foreign-contact exemption; no thresholds were changed.
- Fixed the retention/contact classification: after a verified exact disabled
  seam/event ONLY, incidental mesh-leaf contacts on that same detached branch
  are separated from eligible shaft support. Leaves supply NO grasp force,
  `stem_only=False` and rejected raw rows remain in the log. Opposing shaft
  loads, current pad proximity/penetration, slip and all native contact caps
  remain mandatory. Foreign leaves, parent/support, camera or stem errors are
  not exempt. No pre-cut behavior or collision filtering changes. The aperture
  controller needs the explicit released-leaf context to accept this case.
  Focused grasp/native-adapter/controller/cut/yield tests passed 308/308; the
  expanded launcher/context/yield/controller selection then passed 56/56.
- Native274 RIGHT ONLY: full bounded 50 s diagnostic passes, cut at 27.075 s,
  actual world/relative loaded advance 0.30002 mm, withdrawal verified, left
  parked/open/unloaded throughout. Passive released-target landing reaches
  4.23081 N (recorded under the user-authorized deferred landing policy).
  Native275 BIMANUAL: full bounded 50 s diagnostic passes, cut at 33.5125 s,
  opposing shaft grasp retained through the end, max slip 2.61124 mm (<3 mm),
  right withdrawal verified. Each is ONE source fixture, not general reliability.
  Native276 is the no-render repeat; result pending. Native milestone PNGs are
  in native273..275; they are paused diagnostic views, not synchronized RGB-D.
- Reproducible explicit experiment (from examples/greenhouse_sim with Isaac's
  Python): `-B -m sim_physics.ground_truth_trial --output NEW_DIRECTORY --mode
  bimanual --milestone cut_action --process-zone-trial`; use `right_only` for B,
  and optional `--capture` or `--watch`. Source station/robot/grasp are unchanged;
  the corrected elbow-up precontact pose is in `crossbar_process_zone_trial.json`.
  Public linear-contact and explicit historical recipes remain separately
  reproducible. This is not a stable/frozen production profile.
- Remaining important limitations: short precontact start (not yet the extended
  visible approach), reversal at release instead of completed through-stroke,
  prescribed seam release rather than tissue fracture/kerf or cut-surface quality,
  greenhouse/intact-plant integration and performance requalification. Native274
  also emits a large post-fall `getMaterialFromInternalFaceIndex(0xFFFFFFFF)`
  warning stream; it is NOT hidden or treated as resolved. Native274 RTF 0.27555
  differs in mode/contact/render workload from native272, not a matched speedup.
  Existing cohesive-interface/coupon modules remain separate and unintegrated:
  the new contact-compression experiment is not their material fracture law.
  No VLM/data/hardware work, source visual deletion, OS setting change or full
  greenhouse A/B/C completion is claimed.

- Native276 repeats native275 WITHOUT rendering: same 33.5125 s cut, same
  2.61123 mm max slip, all bounded cut/retention/withdrawal gates pass through
  50 simulated seconds. This is two matched bimanual passes plus one right-only
  pass on ONE isolated source branch, not arbitrary-target reliability. The
  first new full regression found four legacy test doubles missing an explicit
  intact-rig state; fixed the fixtures, not the runtime gate. All 58 affected
  integration/context/yield/launcher tests pass. Final full `sim_physics` rerun:
  4,325 passed in 209.27 s (`regression_20260913_process_zone_v2.log`).
  Save this as an experimental, reproducible checkpoint, NOT the A/B/C freeze.

### 2026-09-13 - Full-knife load correction, measured follow-through and intact greenhouse (IN PROGRESS)

- A/B/C is **not complete or frozen**. The earlier native274..276 mechanism
  passes used the earlier contact accounting and cannot qualify the corrected
  controller below. No VLM collection/training, hardware commands, source-asset
  edits, review changes, reboot or OS memory-setting changes were performed.
- Implemented explicit `--through-stroke-trial`: follows the already screened
  downward stroke after an evidence-validated native seam release. Completion
  requires actual edge position beyond the shaft, unloaded endpoint dwell,
  current support, actual arc-up/edge-down orientation and unchanged native
  force/penetration/collision checks. Time or sent joint targets cannot count
  as completion. Withdrawal reverses the last actually sent stroke fraction,
  not an inferred extra forward increment. It stops on load/corridor limits
  or bounded timeout; it creates no kerf and disables no collisions.
- Native277 exposed a new report-generation bug: an empty follow-through receipt
  masked the original fault and the diagnostic gzip lacked its closing footer.
  **Do not use native277 as qualified evidence.** Fixed failure preservation,
  unconditional trace closure and null-receipt handling. The initial new 1 mm
  lateral check was also inconsistent with the existing trial's 3 mm target
  envelope; retained that original envelope plus 0.25 mm endpoint progress
  tolerance. All original force, collision and penetration bounds remain.
- Native278: actual-edge bimanual release again occurs at 33.5125 s. Following
  down instead of immediately reversing reaches only 0.06193 mm additional
  measured travel before a 34.46875 s native guard stop. There is still
  7.32396 mm to the planned endpoint. Max grasp slip 2.67444 mm (<3 mm).
  The crossbar contacts the seam's solid support/branch faces; unwanted full
  normal-plus-friction contact rises to 0.50958 N. **No through-stroke pass.**
- This exposed a real controller accounting defect: rejected/noncutting knife
  contacts could make `allowed_tool_contact_n` zero while the knife still
  carried physical load. Added `knife_load.py`: sum every native pair involving
  the exact knife assembly, normal PLUS friction, independent of edge
  eligibility. The feed and 0.5 N knife cap now see this full bound; wrong
  faces/normals still never become cutting evidence. Available rejected tool
  normal rows also retain the 1 mm penetration bound. Self-pair separation
  is not available in that callback; its full load is nevertheless counted.
  No reclassification or removal of unwanted contacts was used to get a pass.
- Native279 RIGHT ONLY with corrected full-knife accounting stops without
  release at the unchanged 30 s load-acquisition deadline (38 s episode).
  Qualified travel reaches about 0.23267 mm, below the required 0.3 mm.
  This invalidates promotion of the old right-only pass to the new controller.
  A physically consistent cut-face/local deformation treatment and renewed
  native A/B qualification remain necessary, not a higher force limit.
- Added opt-in lossless `--stream-trajectory`: every full diagnostic sample is
  written to `bimanual_trajectory.jsonl.gz`; only the current full sample and
  small per-step result summaries stay in RAM. Current/failure samples retain
  late annotations. No physics/sensing guard is decimated. Native278 saved
  16,545 complete rows in 349,508,948 bytes, serialization 19.6951 s total,
  max 2.779 ms per row. Native279 saved 18,240 rows in 273,916,441 bytes,
  serialization 15.5396 s, max 1.734 ms. This reduces diagnostic retention,
  but is **not a measured matched FPS improvement**. RTFs 0.19805 (278) and
  0.31343 (279) differ in mode/contact work; neither is real-time.
- Added explicit `--greenhouse-trial` to the public launcher. It cannot select
  isolation or remove other target-plant components. It restores the provided
  three-gutter preview planting, including BOTH detailed component plants
  (not substituting a backdrop for the second one), original source meshes,
  full greenhouse/gutter visuals and nearby static plant collisions. Existing
  source-preserving gutter instancing and guarded distant-wire optimization
  are reused. Only the selected petiole is deformable; context is static.
- Native280 loaded the intact greenhouse: all 404 target-plant components,
  143 context plants (one detailed neighbor), 84 static-contact context plants,
  59 distant render instances. The existing bounded robot/plant window retains
  14 local wire colliders and excludes 7,036 unreachable wire proxies, with
  their renderable wires preserved. Native startup stops at ZERO physics steps:
  right forearm `link_right_arm_4` intersects original `SubStem_44/Leaf_028`.
  All 71 robot actor positive controls pass before/after; no motion authorized.
- Native281: 20 converged same-wrist elbow candidates checked (12 local family
  solutions, 16 global IK attempts), no fully clear start. Native282: expanded
  read-only 16 higher/lateral waiting-pose proposals, same blade orientation,
  no fully clear start. Both retain final native controls and zero physics
  steps. These bounded failures do not prove global unreachability. A different
  stance/tool corridor or target is needed; do not delete the neighboring leaf.
- Public diagnostic switches: `--screen-ready-pose` (same-wrist elbow search)
  and `--screen-approach-start` (higher/lateral start search). Both stop before
  physics, preserve the authored scene, and produce proposals only. Even a
  clear proposal requires a fresh launch and full approach/action checks.
- Validation: 4,377 `sim_physics` tests passed in 211.74 s
  (`regression_20260913_follow_through_v1.log`). Later approach-search/reporting
  changes passed 74 focused tests. A focused-only test-import ordering issue
  was fixed by loading its USD helper before the native callback mock, not by
  weakening runtime checks. The full final rerun is recorded below when done.
- Native283 completed the limited isolated bimanual recheck with corrected
  full-knife accounting: stable grasp, but acquisition timeout and NO cut.
  Maximum consecutive qualified travel was 0.247321 mm, below 0.3 mm.
  Remaining: physically traversable cut faces, successful full A/B strokes,
  feasible full-scene approach, measured greenhouse performance and calibrated
  damage/partial-cut behavior. Native274's material-index warning remains
  unresolved. Do not resume VLM tasks or describe this as reliable tissue cutting.

### 2026-09-14 - Actual edge traversal, post-cut load transfer and exact floor contacts (IN PROGRESS)

Milestone A/B/C is still ACTIVE, not frozen; the requested evening deadline
passed without full greenhouse qualification. VLM collection/training remains
paused. All trials below use the source knife crossbar, arc up, world-down
stroke, 0.2 N minimum signed resistance, 0.5 N ALL-knife hard load limit,
fresh native guards and the original source meshes. No timed seam release,
grasp weld, hidden collision removal or relaxed slip limit was introduced.

- Added explicit `--blade-aim-offset-m` comparison, constrained to the existing
  +/-1.5 mm proposal range. Moving the aim from -1 to +1.5 mm along the petiole
  avoids the prior proximal-face wedge at initial loading. This moves only the
  commanded contact proposal, NOT the preauthored release seam or its 3 mm
  native axial tolerance. It still needs native checks per target.
- Native284 (`bimanual_downward_20260913_native284`): right-only release and
  4.932809 mm measured post-release downward travel. The old through controller
  stalled on 0.101497 N side contact / 0.123405 mm indentation, with 2.411845 mm
  remaining to an arbitrary overshoot waypoint. It correctly failed that old
  requested qualification. This exposed a post-release controller issue, not
  permission to treat side contact as cutting evidence.
- Added `cut_section.py` and opt-in `--material-clearance-trial`. Analytically
  intersect the current blade plane with the shaft cylinder and bound the full
  elliptical cross-section. Require the measured SOURCE sharp edge to clear it
  by 0.5 mm, adequate edge-end span and the entire section inside the original
  3 mm seam envelope, plus 25 ms dwell and >0.3 mm actual downward travel.
  Geometry matches dense cylinder/plane intersections in tests. Section pose is
  bound to the freshly sampled proximal body, not a falling detached target.
- After verified release ONLY, allow bounded side-face sliding on the exact
  crossbar and the two seam shaft colliders. All other knife-part/scene pairs,
  missing contact provenance, unknown separation, over-force or lost support
  withhold advance. Broad-side contact still CANNOT authorize a new cut.
  A loaded blade face after edge traversal is not an unloaded withdrawal;
  the latter is checked separately. Original full-waypoint trial remains
  available without this explicit mode.
- Native286: first current ALL-knife guarded **right-only isolated pass**.
  Release 22.377083 s; full shaft-section clearance 42.266667 s, after
  5.962313 mm actual post-release downward travel; edge clearance 0.506796 mm,
  maximum section axial distance 2.752769 mm, full knife load 0.305440 N at
  clearance. Final fresh native withdrawal endpoint passed at 85 s. This is
  ONE case, with the older timed withdrawal and original triangle floor;
  material-index warnings persisted. Not general reliability or tissue proof.
- Native287: bimanual grasp, release at 33.766667 s and full section traversal
  at 54.583333 s passed, but the subsequent two-second reverse dragged the
  retained stem: 3.002639 mm material-point slip at 57.75625 s stopped execution.
  Therefore this is NOT a complete A pass. The trace contains a real axial
  displacement (~2.746 mm), not merely normal pad compression; the 3 mm guard
  has NOT been reclassified or loosened to make it pass.
- Added `cut_retraction.py`: same screened reverse stroke, limited to 0.3 mm/s
  while the blade is loaded, 2 mm/s when unloaded, same 0.4 N controller stop
  and 0.5 N hard limit. Requires fresh support/exact cut-face contact provenance,
  actual reverse motion and unloaded precontact endpoint dwell before reversing
  the approach path. Commands and elapsed time cannot complete retraction.
  Integrated only into explicit material-clearance trials; final qualification
  now requires this extra measured retraction gate.
- Fixed a load-acquisition boundary problem in `BladeFeed`: the setpoint was
  the SAME 0.22 N value as the loaded-state entry boundary. Native287 lingered
  at ~0.219945 N before quantization eventually crossed it. Loaded-downward
  acquisition now targets 0.25 N, INSIDE the unchanged 0.22..0.28 N control band.
  Other historical feed modes and all release/force limits are unchanged.
- Added opt-in `--rectilinear-floor-contacts`. The original floor's 44 triangles
  form a closed rectilinear solid exactly represented by THREE boxes, including
  both raised strips. `floor_contacts.py` verifies manifold winding, grid-cell
  parity, source/union volume and exposed area BEFORE editing only session
  collision opinions. Rendering, floor height, holes/steps and material are
  preserved; unsupported or previously approximated geometry is rejected.
  This addresses the suspected floor material-index path, not by suppressing
  logs. Native287 used it, but held debris did not reach the floor; a falling
  right-only repeat is still required before claiming warning resolution.
- Native285: all 24 bounded waiting-pose translations (up to 200 mm) failed
  complete greenhouse startup clearance. Native288: 20 new complete-tool
  heading/multistart proposals also failed. All final native controls passed,
  source poses were unchanged, and zero physics steps executed. The new
  `--screen-tool-heading` rotates proposed wrist/tool frames, not the knife
  mounting. These bounded searches do NOT prove the target globally unreachable;
  a different stance or target/corridor is required without deleting neighbors.
- Validation: 33 startup/CLI tests, 39 section/through tests, 27 floor/section
  tests, 120 feed/startup/CLI tests, 10 expanded startup tests and 24 new
  retraction/section tests passed in focused runs (overlapping sets).
  Full suite `regression_20260914_material_clearance_v1.log`: 4425 passed.
- Native289 tested the bimanual sequence with midpoint acquisition,
  contact-paced retraction and exact floor contacts: release at 23.570833 s,
  then 3.002063 mm grasp slip at 39.677083 s, BEFORE withdrawal started.
  Thus retraction pacing alone does not solve retention; no complete A pass.
  The full regression overlaps this diagnostic; do NOT use its elapsed timing
  as a matched performance comparison. A dedicated performance run is needed.
- Still not implemented/qualified: complete A with retention/withdrawal,
  A/B in the intact greenhouse, visibly extended right-arm approach, measured
  responsive full-scene operation, material-calibrated fracture/kerf/partial
  damage, and broader repeat/reset coverage. This remains a preauthored
  force-qualified seam-release approximation, NOT calibrated tomato tissue.

### 2026-09-14 - Fresh greenhouse stance and current right-only withdrawal (IN PROGRESS)

- Native290 tested an explicit 90 mm grasp (instead of 80 mm). An original
  attached leaf intersects the finger approach; rejected at 0.9 s before grasp.
  Native291 tested centered blade aim rather than +1.5 mm: release at
  23.564583 s, then 3.013083 mm material-point slip at 24.429167 s. Both fail;
  neither leaf collisions nor the original 3 mm slip guard were relaxed.
- Native293 is a CURRENT right-only isolated pass, including the NEW measured
  section traversal and contact-paced retraction. Release 18.135417 s;
  unloaded reverse endpoint at 47.683333 s after 10.938336 mm measured reverse
  travel, zero knife load and 25 ms unloaded dwell; final trial passes at85 s.
  All requested cut/clearance/withdrawal/open-left guards pass. Debris/torso
  landing remains deferred per user (observed peak released-debris contact
  5.779724 N); this is not a safe-deposit or calibrated tissue result.
  Exact three-box source floor enabled; ZERO material-index warning messages
  in this run. 85 simulated seconds /327.005 s tick wall time =0.259935 RTF,
  headless/no renders. NOT real-time and NOT full-greenhouse evidence.
- Added explicit public grasp-location and CPU worker-count comparisons.
  Native293 used ONE worker, unchanged480 Hz/128PGS iterations, all source
  geometry/materials/guards. It also used optimized snapshot copying and a
  different controller from historical runs; do not attribute its aggregate
  timing improvement to thread count alone. Matched repeat needed.
- Replaced recursive deepcopy of the validated flat native-contact schema
  with independently copied dictionaries/vectors. All rows, signed impulses,
  precision, validation and snapshot mutation isolation are retained. A128-row
  local microbenchmark: 0.594743 ms recursive versus0.061962 ms schema copy
  (~9.6x for this COPY ONLY, not simulator FPS). 52 overlapping contact tests
  passed. No contact sensor or physics step was decimated.
- Added bounded zero-step two-arm/base proposals with complete native startup
  controls and left-path self/interarm checks. Native292: original15 translated
  stance/near-tool proposals fail. Expanded search includes the SDK withdrawn
  right-arm pose. Native294 finds a proposal with all initial/final71 robot
  controls passing, unchanged scene and ZERO physics steps:
  station=[0.31459792940322134,0.984160534448182,-147.10477763841965],
  right=[0,-5,0,-120,0,70,0]. Full proposal and left seed are in its report.
  This is not a base driving path, grasp, right approach or cut qualification.
- Added `--station-proposal-report` for a NEW intact-greenhouse launch. It
  checks task identity and final native controls, records the report SHA256,
  imports ONLY initial base/arm proposals, and reruns native startup and all
  execution/path checks. Stored paths and prior motion authority are never
  replayed. Any final-control failure revokes all proposed base/arm fields.
  Native295 is running this fresh greenhouse recheck; result pending.
- Validation: `regression_20260914_station_contact_v2.log`:4437 passed in
  190.96 s (before the fresh-proposal adapter was added); the adapter plus
  stance/comparison focused tests:19 passed. These suites overlap. VLM tasks
  remain paused, and this is an engineering checkpoint, NOT an A/B/C freeze.

### 2026-09-14 - Coupled jaws and intact-scene planning (IN PROGRESS)

- A/B/C are NOT frozen: finish grasp/downward cut and right-only cut in the
  isolated fixture AND intact greenhouse, then qualify responsiveness without
  removing physics/visual surroundings. VLM work remains paused. Source
  checkpoint before this increment: `2ab8bfa` on `koh-dev/sim-vlm`.
- Native295/296 establish the left grasp in the intact greenhouse but right
  planning stops at its60s WALL budget, not the20,000-query cap. Query counts:
  3513/3292;296 has123 exact-cache hits. Budget diagnostics now distinguish
  time and query count. No right approach or cut occurred.
- Added exact float64 native-query memoization in one invalidation-guarded
  epoch; initial/final actor-positive controls are NEVER cached. Added
  `--cut-priority-report`: changes candidate ORDER only, not poses/path replay
  or inherited motion authority. Native298 tries the known15deg/-1/zero-wing
  frame first but all original tool transits intersect main stem/foliage.
  Stops at60.0215s,3589 queries,129 exact hits,6 proposals, ZERO arm IK attempts.
- The imported v1.2 URDF has two independent finger slides; the manufacturer's
  specification lists one opening DOF per gripper. Added explicit
  `--coupled-fingers-trial`: compliant PhysX mimic q2+q1=0 on original left
  prismatic joints, session-only, frequency480/damping1.2 engineering prior.
  Metre/Z-up, original parent/child/limits and USD values checked. No plant
  weld, extra grasp force, collision filter, mass change or pose overwrite.
  Transmission stiffness/backlash are UNCALIBRATED; native parameters have
  not been independently read back. Actual jaw relation is monitored every step.
  Manufacturer: https://www.rainbow-robotics.com/product_spec/download?id=ba9fd188-b019-4c7b-9812-d55b27c4b1c0
  Schema: https://docs.omniverse.nvidia.com/kit/docs/omni_usd_schema_physics/latest/physxschema/class_physx_schema_physx_mimic_joint_a_p_i.html
- Native297 isolated coupled-jaw result: release23.564583s, full-section
  traversal44.925s, unloaded reverse78.789583s, retained through85s. Max
  material slip1.286114mm; maximum measured |q1+q2|1.791865micrometres.
  Measured reverse10.945775mm, load0.002309N,25ms unloaded dwell.9/10 gates
  pass. FINAL PARK CLEARANCE FAILS: crossbar versus detached Segment_001.
  Final wrist position error0.000402mm: reaches its command, but not clear.
  NOT a complete A pass. Headless RTF0.2020, not real-time.
- Native299/300 proposed +10mm/+3mm world-up waiting poses with checked IK.
  Full native startup rejects camera/main-stem and arc/main-stem interference,
  respectively, ZERO physics steps. Raising the tool alone is not a solution.
  Bounded `--right-ready-lift-m` and `--right-ready-retreat-m` propose initial
  poses only; original startup/path checks run again. Wrist-axis retreat is
  being tested next. No runtime body pose is assigned by these options.
- Flat-cylinder/box separating-plane refinement removes only mathematically
  proved capsule-end false positives under the explicit analytic-cylinder
  backend. The complete original1mm margin is required. Native geometry,
  contacts and seam allowances stay unchanged. Native301 rendered repeat
  still has the SAME final park failure (zero such refinements there), so
  this improvement did NOT fix withdrawal. It repeats297's cut time and max
  slip, traversal, unloaded reverse and retention through85s without a fault.
- Native301 saves17 paused native diagnostic PNGs with timestep receipts:
  `data/sim_physics/bimanual_downward_20260914_native301/`. Inspected knife and
  plant-side views: camera hardware/main stem partly obscure the edge. Images
  alone are not proof of clean cutting and are NOT synchronized training RGB-D.
- Added lift-before-rotation/above-target wrist proposals, preserving existing
  <=2mm/1deg samples, whole-arm IK/self/plant checks and every native guard.
  Old above-waypoint modes rotated while low; they could not test this sweep.
- Added native empty-region reuse: a REAL overlap miss on an expanded box can
  certify a later box only when it is wholly contained, including original
  margin, for the EXACT same static collider and unchanged epoch. Expansion
  5mm/half extent, containment reserve0.1micrometre,128paths x16 certificates.
  An expanded hit falls back to the original query; unknown/invalidation stops
  planning. Fresh positive controls remain. Speed benefit UNMEASURED.
  Native302 is profiling the intact scene with these changes; result pending.
- Full regression `regression_20260914_coupled_clearance_v3.log`:4504 passed
  in199.82s. Subsequent waiting-retreat/coupling focused tests:24 passed.
  Overlapping focused sets:84 cylinder/transit/withdrawal,91 query/cache tests.
  No datasets/reviews/splits/model jobs or hardware changed. Existing review
  apps were not closed. No reboot or memory-reserve bypass. Original native
  force/slip limits and480Hz physics retained. Tissue fracture/kerf/partial
  damage, greenhouse A/B, responsive execution and repeat/reset remain open.

### 2026-09-14 continuation: isolated withdrawal passes; greenhouse approach open

- Native302 profile identified repeated Python geometry checks as well as native
  stepping cost. Native304 completes all 50 full-greenhouse pose proposals in
  34.7023s (2632 native queries,6631 same-epoch empty-region hits), compared with
  the earlier 60s timeout after six proposals. It still finds NO full approach.
  These are combined engineering changes, not an isolated cache speedup study.
- Reduced repeated per-shape transform checks and empty triangle queries; bounds
  come from the same actual collision points/faces as the narrow phase. Source
  visuals, native colliders, solver iterations and contact limits are unchanged.
  Full regression v4:4512 passed. v5:4536 passed. Latest v6:4578 passed in210.30s
  (`data/sim_physics/regression_20260914_egress_joint_search_v6.log`).
- Native303's 10mm waiting-wrist retreat fails native startup. Native305/306
  also reject all50 cut proposals from the SDK-ready station in ISOLATION, so
  the approach obstruction is not solely the extra greenhouse plants. New
  back-away/lift/side-entry schedules retain <=2mm/1deg sampling and full-arm
  checks. Native309's horizontal goal-side entries also find no valid path.
- Native307 zero-step cut-frame station search:28 of32 new choices fail IK;
  remaining new choices fail native geometry (including right forearm versus
  neighboring Leaf_028). It falls back to the already known native294 station;
  this is NOT a new qualified approach or a solved greenhouse deployment.
- Added opt-in `--joint-transit-fallback`: after failed Cartesian templates,
  try bounded, deterministic whole-arm joint-space approach paths and rebuild
  the exact downward stroke from their terminal state. Existing 10mm interarm,
  3mm self,1mm scene margins and native epoch checks remain. No filter change.
  Exact configurations, not rounded neighbors, key the joint search cache.
  Native312 tries five clear endpoint IK configurations but exhausts60.0422s
  (5635 native queries) without a connecting path. No right motion is executed.
- Native308 confirms that an unloaded blade can still be within1mm of a
  previously severed face; a strict free-space start therefore rejects it.
  Added opt-in `--postcut-egress-trial`: reobserve after verified unloaded
  reverse, then search fixed-orientation egress. Only the two already severed
  faces can have an initial proximity allowance; their conservative separation
  bounds must increase until clear. All other pairs remain checked normally.
  Execute under the stricter <0.01N full-tool limit, keep the left grasp and
  all original guards, and require a fresh full-margin native endpoint check.
  This neither ignores neighboring geometry nor authorizes a new release.
- **Native310: isolated bimanual mechanism passes all10 sequence gates.**
  Same source `seed101_full/SubStem_41`,20mm seam,80mm left-grasp arc, original
  full RB-Y1 and camera-aligned crossbar blade. Release23.564583s; full material
  section traversal44.925s; unloaded reverse78.789583s; then a screened2mm
  world-up egress over4s. Retained through85s and final full-margin withdrawal
  check passes. See `data/sim_physics/bimanual_downward_20260914_native310/report.json`.
  This is ONE privileged source-target mechanism trial, not broad reliability,
  a long visible approach from a neutral stance, greenhouse proof, safe deposit
  or calibrated tissue fracture. Source anatomy/knife/cameras were not edited.
- Added explicit post-release feed comparison (`--postrelease-feed-m-s`, default
  unchanged0.0003m/s, bounded up to0.002m/s). Faster rate reduces as measured
  knife load rises; original hold/backoff, contact, penetration and slip gates
  stay active. **Native311 at0.001m/s FAILS the left-finger0.5N limit** just after
  release, despite low knife load. Faster feed is NOT qualified or a new default.
  Any future acceleration must account for left-grasp load, not just knife load.
- Current A/B/C is NOT frozen or complete. Next: rendered/repeated native310,
  isolated direct-cut repeat, a genuine collision-free approach in the intact
  greenhouse for both strategies, and matched latency measurements. Do not
  claim real-time speed from these overlapping diagnostic runs. VLM collection,
  training, review decisions/splits and hardware remain untouched.

### 2026-09-14 continuation: rendered repeat and exact contact-accounting optimization

- Native313 repeats native310 with 15 Hz rendering and 23 paused, step-bound
  diagnostic PNGs, including new front/back blade-plane inspection cameras.
  All ten isolated sequence gates pass. Release23.564583s; full section
  traversal44.925s; unloaded reverse78.789583s; retained through85s after the
  fully screened2mm upward egress. Maximum slip1.286446mm. The new views do not
  replace the three D405 views. These PNGs are NOT a continuous video or
  synchronized RGB-D training observations; foliage/fingers still obscure
  parts of the edge in some views.
- Native315/316 retain the intact greenhouse. The broader redundant-arm
  endpoint search checks59 IK configurations but finds no clear endpoint;
  therefore it cannot proceed to joint-space path search. Diagnostics now
  retain endpoint rejection details and bounded RRT tree/check counts. The
  observed rejection includes right forearm versus SubStem_44/Leaf_028.
  No foliage, tool geometry or contact margin was removed to obtain a pass.
- Native317 explicitly compares64/0 solver iterations against128/0. It releases
  at23.629167s but FAILS the3mm grasp-slip guard (3.005258mm). The64-iteration
  option remains an unqualified numerical experiment, NOT the default.
- Optimized ContactEvents bookkeeping: recompute the changed pair, while
  preserving original pair order and fresh math.fsum totals for every bucket.
  No delta-subtraction drift, deferred observer or dropped contact row. Tests
  compare the old and new dictionaries/forces EXACTLY after every header,
  including friction-only input, zero rows and late eligibility revocation.
  Synthetic accounting-only benchmark medians: old0.129210s/new0.056285s
  (~2.30x for that helper, NOT whole-simulator speed).
- Native318 with this optimization PASSES all ten isolated gates with the
  exact same release time,19,079 edge contacts and1.286446mm maximum slip as
  native310/313.85s simulated takes421.807s tick wall time (~0.202x real time).
  Runs were not matched for host load; no causal whole-sim speedup is claimed.
  Full regression v7:4584 passed207.84s; v8:4587 passed216.20s
  (`data/sim_physics/regression_20260914_contact_accounting_v8.log`).
- Added explicit --screen-station --cut-station-orbit: zero-motion proposed
  base/two-arm stations must clear BOTH a30mm-raised waiting pose and the
  cut-entry pose using the full native robot/scene inventory. Existing hand
  goals, target, torso and assets are unchanged. Proposals require a fresh
  launch, native floor/startup checks and the entire grasp/cut path; endpoint
  checks confer no path authority. Native319's first48 (radius>=.45m) choices
  found no two-arm IK configuration. A local kinematic check showed the search
  omitted reachable closer stations; the revised bounded family includes
  .30/.35/.40m stations and15-degree heading variations, within the ORIGINAL
  .25..1m base-distance limit.22 focused station/proposal tests pass. Native321
  did not start: full-greenhouse memory reserve unavailable during native320.
  The expanded family has NOT yet had its native search completed.
- Native314/321 memory preflight refusals are host-capacity failures, not cut
  results. No launch-reserve bypass, reboot or closure of unrelated apps.
  Native320 is a profiling repeat, pending. A/B still lack an intact-greenhouse
  motion pass and a neutral-stance visible approach; C is NOT real-time.
  Three isolated bimanual passes remain single-source mechanism evidence,
  not arbitrary-target reliability, calibrated tissue fracture or safe deposit.
  VLM collection/training, datasets, review decisions and hardware stay paused.

- Follow-up native320 profiling repeat also PASSES all ten isolated gates.
  `profile.pstats` attributes substantial cost to native contact callbacks,
  per-row shaft-contact evidence and diagnostic serialization. Native318's
  full-rate archive alone spent45.580s serializing/compressing40,800 records.
  Profiling timings are instrumentation-affected, not a speed benchmark.
- Added opt-in `--park-left-ready` for RIGHT-ONLY trials. Previously even an
  unheld cut initialized the left hand through target-grasp IK and a unused
  grasp path. The new mode uses the source SDK left-ready joint configuration,
  no grasp IK/path, and retains full startup/self/environment checks plus the
  per-step open/unloaded/unchanged-left-reference guards. It is not available
  for bimanual trials and does not claim grasp/retention. Historical defaults
  stay unchanged. Station proposals cannot cross between the two left modes.
- Native322 uses this independent left park in the intact scene: no clear
  station within45.015s. Native323 additionally checks both shaft-normal signs
  while keeping arc-up/downward motion;125 stations examined within45.218s,
  no clear proposal. Both preserve final native positive controls and take
  zero physics steps. These are bounded failures, not infeasibility proofs.
- Arithmetic-only hot-path changes preserve the exact scalar normal tolerance,
  rigid-transform bounds and material-point witness calculations. No rate,
  iteration, contact, friction, slip or geometry checks were removed. Boundary
  and randomized equivalence tests pass; native324 repeat is pending.
- Full regression v9:4594 passed215.23s. v10:4614 passed202.45s
  (`data/sim_physics/regression_20260914_independent_left_park_v10.log`).
  Investigation identified existing outward-facing candidates in seed19/47/53/79
  among the supplied plants. No source switch or new-target physics pass yet.
  Original seed101/SubStem_41 greenhouse access and real-time speed remain open.

### 2026-09-14 continuation: intact end-row approach and gravity-bias investigation

- Native324 repeats all ten isolated bimanual gates after the arithmetic-only
  optimization, with the same23.564583s release and1.286446mm maximum slip.
 85s simulated /402.757s tick wall time is0.211x real time. Host loads were not
  matched; this does NOT establish a causal whole-simulator speedup or C.
- Added explicit `--source-station-trial PLANT/SubStem_N`. It removes the old
  seed101-specific initial base/arm poses before proposing a station for an
  existing source petiole. No source anatomy or labeling is changed. Optional
  `--target-row-slot 0` or23 swaps that detailed plant with an existing backdrop
  at an original end-row position: same144 plants, same assets and0.5m spacing,
  intact greenhouse/gutters. This is a different, easier test case, NOT a fix
  or a pass for the previously blocked middle-row target.
- Native325 exposed a startup import-order bug: importing robot_scene before
  SimulationApp loaded incompatible USD bindings. Moved unchanged SDK pose
  constants to a pure module; fresh-subprocess tests forbid pxr/omni imports
  from source-trial configuration. Startup failures now leave an explicit
  failure report;325 predates this fix and has only its preserved log.
- Native326 seed19/SubStem_41 middle-row search found no clear station within
 45s. The obstructing backdrop collision meshes are exact static triangles,
  NOT whole-plant convex hulls. No contact geometry was removed.
- Native327 seed19/SubStem_41 at original row slot0 found a native-clear raised
  waiting pose and cut-entry proposal after15 stations/8.656s. Final native
  positive controls passed. Zero physics steps: this was no motion proof.
- Fresh native328 revalidated327 and executed the right-only approach in the
  full scene, but FAILED load acquisition: leading-edge guard rejected shaft
  end-face contacts. No release or material change occurred. Steady measured
  right-wrist position error was about1.45mm, already present before contact,
  comparable to the1.5mm axial aiming offset. This motivates investigating
  the historical30%-of-effort gravity clip; causation is not yet confirmed.
- New opt-in `--budgeted-joint-gravity` shares the SOURCE total angular effort
  between native gravity feedforward and PD. Original stiffness, damping and
 70% PD ceiling remain; gravity above90% of source effort fails closed. Float32
  rounding cannot enlarge the total budget. Original finger budgets and all
  cut/contact/slip/geometry guards remain. Full-rate diagnostics report native
  required gravity, applied effort, PD allowance and hypothetical legacy clip.
  Native329 is the first fresh end-row test; no pass claimed yet.
- Regression v12:4651 passed,1 skipped in216.02s
  (`data/sim_physics/regression_20260914_end_row_v12.log`).187 focused gravity,
  full-robot, finger-budget and launcher tests pass. A/B still require intact
  greenhouse sequence passes; C is not real time. VLM/data/hardware remain
  untouched. Cutting is an uncalibrated force-qualified joint-release model,
  not tissue fracture, kerf or measured clean-cut quality.

- Gravity follow-up:329 failed controller binding before motion because
  imported continuous wheel joints have no URDF effort field. The allocator
  now selects torso/arms/head only, retaining the prior wheel/finger behavior.
 330 confirms the diagnosis: native torso_0 gravity113.45965Nm versus the old
 81Nm clip (32.45965Nm residual). New right-wrist precontact error at8s is
 0.006683mm versus about1.45mm in328. The same intact-scene approach now cuts
  at approximately18.7s, with original contact/alignment/force gates.
-330 is NOT a complete pass: it times out in follow-through. At40s the only
  loaded knife pair is CrossbarContact against the attached proximal stump,
  about0.388N and0.550mm penetration, outside the exact leading strip. The
 35s traversal deadline stops it; no force increase/collider removal/bypass.
- Added a bounded distal placement experiment, up to+2.5mm (negative bound
  remains-1.5mm). This changes a commanded AIM bound, not the3mm release or
  traversal window. Extended proposals require the actual source crossbar,
  full measured section trial, and a conservative cylinder-cap/knife-slab
  preflight. Source mesh vertices supply actual blade thickness. Require
  >=0.1mm separation from the attached stump and the entire section within
  the ORIGINAL3mm window with50um planning reserve. All native guards remain.
 331 tests+2.3mm with captured inspection PNGs; result pending. No kerf or
  tissue removal is introduced. This is not a geometry exemption.
- Regression v13:4669 passed,1 skipped in219.79s before the distal-placement
  addition (`regression_20260914_joint_gravity_v13.log`);89 focused placement,
  source knife, section and launcher tests passed after it. A/B/C remain open.

### 2026-09-14 intact-greenhouse RIGHT-ONLY sequence pass (native331)

- Native331 PASSES all nine complete diagnostic sequence gates for
  seed19_full/SubStem_41 at original end-row slot0: checked approach, native
  force/direction-qualified release18.718750s, full shaft-section traversal
 39.525000s, unloaded reversal45.177083s, freshly screened2mm upward egress,
  and bounded observation through85s. Left remains SDK-parked/open/unloaded.
  All configured144 plants, source greenhouse/gutters, knife, cameras and
  existing physics/contact/clearance limits are retained. Captured20 native
  inspection PNGs (NOT continuous video or synchronized training RGB-D).
- The source-geometry placement screen reports blade half-thickness1.500000mm,
  proximal stump clearance0.173072mm and maximum section axial distance2.897694mm
  inside the unchanged3mm window. Native follow-through is unloaded, unlike330.
  Maximum passive released-material contact2.973177N is disclosed, not a safe
  deposit claim. Tissue fracture/kerf/clean-cut quality remain uncalibrated.
- Performance is NOT satisfactory yet:85s simulated /539.616s tick wall time
  (0.1575x). Render median13.256ms, whereas480Hz control+native ticks have
  median12.068ms (native-step8.505ms, before0.948ms, after2.619ms). These are
  instrumentation timings, not a matched host-load comparison. Rendering
  capability alone does not establish real-time physics. C stays open.
- Native332 performs an independent BIMANUAL station search on the same source.
  No valid station in45.141s/280 candidates:243 initial-left IK failures; among
  right attempts106 waiting-IK failures,34 self-collision rejections,8 native
  foliage/torso rejections. Zero steps and final native positive controls pass.
  Do not reuse the right-only station as a bimanual motion certificate.
- Exposed the existing bounded `--approach-vector X Y Z` proposal through the
  bimanual trial wrapper. It changes grasp approach side, not target, grasp arc,
  aperture, force/slip criteria or source anatomy. Station reports must match
  approach vector/distance. Native333 searches the opposite side; pending.
- Regression v14:4674 passed,1 skipped,1 stale command-bound test failed. Its
  +1.5001mm-invalid assumption was updated to the new+2.5001mm-invalid bound;
  new tests still require full source-section/runtime guards above+1.5mm.
 59 focused greenhouse/placement/source-geometry tests and49 source/proposal
  tests pass. Full repeat pending. A greenhouse grasp/cut pass, user-visible
  continuous sequence and responsiveness work remain; VLM tasks stay paused.

- Full regression v15 now PASSES:4678 passed,1 skipped in195.84s
  (`data/sim_physics/regression_20260914_greenhouse_cut_v15.log`). Native333's
  opposite-side search also found no valid station:263 candidates/45.109s,
  with final native controls intact and zero steps. An offline kinematic
  screen identified below-shaft grasp approaches with separated wrists; those
  are NOT scene-clear or motion-authorized results. Native334 tests one such
  approach with fresh native scene checks. Startup search now deduplicates
  identical right-arm IK seeds and rejects cut-entry arm extension outside the
  EXISTING0.8..0.98 execution bound;16 focused station tests pass. No physical
  or contact guard is changed. B for native331's end-row case is established;
  full-greenhouse A, continuous visible evidence and acceptable C remain open.

### 2026-09-14 continuation: bimanual station diagnosis and watched execution

- Native334 (below-shaft approach),335 (opposite side at original row23),336
  (source101 at row23) and337 (source101 warm-start) found no complete native
  station proposal. Each took zero physics steps and passed final native
  positive controls.338 also found no proposal with source19 grasp arc120mm;
  the tested native rejections include torso versus Gutter_31. These bounded
  searches are NOT proofs of infeasibility and do not qualify greenhouse A.
- Added optional same-anatomy `--station-reference-report` for ZERO-MOTION
  searches only. It transfers an isolated run's initial joint seeds and moves
  the proposed station by the new authored attachment displacement, recomputing
  floor height. Current source/task/robot checks, both native endpoints and
  all later motion checks remain mandatory. No old path/clearance is inherited.
 337 uses native324 at original row23: more IK branches are found, but foliage
  still blocks the tested poses. The reference did not authorize any motion.
- Public diagnostic wrapper now exposes existing initial torso, grasp roll/
  pitch, station and arm-seed proposals. Defaults and all physical guards are
  unchanged. Explicit seeds are applied AFTER clearing source-specific legacy
  poses and cannot conflict with report-based initialization. The orbit search
  now checks the exact proposed initial station before its coarse radial grid;
  previously a valid between-grid IK proposal could be omitted entirely.
- URDF FK diagnosis: the inherited isolated-test torso posture puts the left
  shoulder at local z1.228133m and right at1.410557m, about182.4mm apart in
  height. A neutral torso places both at1.370m before base/floor translation.
  This is a kinematic measurement, not yet proof of native bimanual improvement.
  Read-only neutral-torso screening found two source19 candidates with the
  unchanged right extension0.8..0.98 and >=10mm interarm-capsule requirement;
  full tool/scene/native grasp checks are still needed. Native341 attempted
  the second candidate but its memory preflight REFUSED startup while340 was
  open:10.33GiB commit headroom versus16GiB reserve. It is not a physics failure.
  Preserve the reserve; run these full-scene tests serially.
- Diagnostic JSON encoding now optionally uses installed orjson with an
  explicit recursive finite/type/cycle check and stdlib fallback. Unlike raw
  orjson, NaN/infinity cannot silently become null. All480Hz records, float
  values, late failure-tick mutations and exclusive output creation remain.
  On102 sampled native331 records, five-repeat median encoding time was
 44.831ms stdlib versus36.302ms finite-checked fast encoding (~1.235x for this
  helper ONLY). This does NOT establish whole-simulator real-time speed.
 25 archive/encoding tests passed in Isaac's interpreter, including invalid
  values and lossless parsed-record equality. Scalar diagnostic finite checks
  also use equivalent math.isfinite after float conversion, not NumPy dispatch.
- Watched trials now have an explicit optional `--watch-auto-run`, still one
  shot with no reset/replay. Coupled-finger watch retains the full480Hz effort,
  contact, slip and diagnostic contract. Render-only heartbeat services the UI
  during synchronous native planning, bounded at15 wall-Hz. Every heartbeat
  is bracketed by the existing scene/physics/timeline epoch guard; rendering
  time counts against the original planning deadline, never extends it.
- Native339 exposed a REAL integration failure: sim.render updates Kit's
  animation timeline although physical stepping is suppressed. At3.5s its
  first planning refresh invalidated the strict time snapshot. No cut occurred;
  the fault and full1680-step archive are retained. Closed only that owned
  failed trial afterward; no unrelated apps or jobs were stopped.
- Corrected this by temporarily disabling automatic timeline advance during
  each rendered native query, committing on the main thread and restoring the
  previous setting at cleanup. Never sets/rewinds time or ignores a timeline
  change. The epoch exists BEFORE the commit, so pending user scrubs/edits
  cannot become a silently accepted new snapshot. Native physics-step and USD
  changes still revoke the plan. See NVIDIA's timeline API for the independent
  auto-update/commit operations:
  https://docs.omniverse.nvidia.com/kit/docs/omni.timeline/1.0.11/omni.timeline/omni.timeline.Timeline.html
- Native340 is the fresh watched full-greenhouse right-only repeat with this
  fix, original source19/row0/2.3mm placement and unchanged physical settings.
  It has passed initial planning, released the seam and reached post-cut
  observation; final85s result pending. No watched sequence pass claimed yet.
- Regression v16:4724 passed in226.03s before the timeline-freeze follow-up.
  The follow-up initially repeated already-latched epoch failures on cleanup;
  fixed cleanup while retaining the post-cleanup acceptance guard.102 focused
  UI/epoch/egress tests now pass, as do56 source/reference/profile tests and30
  station/orbit tests. Full v17 repeat is running. VLM/data/training/hardware
  remain untouched. A/C are open; do not freeze the entire milestone from B.

- Native340 watched repeat now PASSES all nine complete diagnostic gates in
  `data/sim_physics/bimanual_downward_20260914_native340/watch_result.json`.
 40,800 native steps/85s, cut18.718750s,3439 edge contacts and maximum passive
  released-material contact2.973177N: same measured values as headless331.
  Three render-only planning updates survive the strict epoch/final controls.
  This is a live rendered run, NOT an archived video or training observation.
  Closed only that completed owned window after the result was written, to
  free the launch reserve for342; its durable result is watch_result.json.
- C remains too slow:340 took727.527s tick wall time (0.116834x), with8272
  wall-scheduled renders (~11.37 frames/wall-second over measured ticks).
  Render median14.069ms; native/control median14.629ms. Concurrent read-only
  checks/regression and host memory pressure prevent a matched speed claim.
  UI event service is improved, but physical action playback is NOT real time.
- Full regression v17:4733 passed in252.96s. Latest explicit-pose forwarding
  and exact-initial-station additions separately pass56 source/profile and30
  station/reference tests. Native342 is the fresh serial full-scene neutral-
  torso search, using the second kinematic candidate with80mm grasp arc and
  the original20-degree pad pitch. Still a proposal, not a motion certificate.

### 2026-09-14 continuation: upright bimanual grasp and contact-loop cost

- Native342 rejected the neutral-torso candidate: left forearm versus original
  MainStem_24/25. Native343, using the alternate -20-degree grasp pitch and
  elbow solution, found a fresh native station at original source19/row0.
  Raised waiting and cut-entry endpoints clear; right extension0.874351.
  Both searches used zero physics steps. A station proposal is not a complete
  moving-path or grasp/retention certificate.
- Native344 freshly launched343's proposal and physically established bilateral
  grasp in the intact greenhouse. Maximum measured slip17.400 micrometres;
  original finger-to-cut-plane placement clearance41.374mm. It stopped BEFORE
  right-arm planning at8.922917s because static retention was not established.
  Detached mass12.385g and gravity moment about14.21mNm require a worst-finger
  contact bound1.071N under the fixed-patch audit, versus the unchanged0.5N
  limit (utilization2.1423). This is one tested grasp, not proof that the plant
  is ungraspable. No force cap, grasp weld, geometry or mass was changed.
  Evidence: `data/sim_physics/bimanual_downward_20260914_native344/report.json`,
  full compressed trajectory and native milestone images in that directory.
- Native345 tests source101, whose isolated324 previously passed retention,
  with a neutral torso at original row23. Its bounded45s native station search
  found no proposal. Read-only IK candidates were not scene certificates.
  Native346 additionally prioritizes324's original15-degree blade-plane family;
  no prior path/clearance is inherited.346 found no proposal: the exact seed
  intersects a target leaf with torso5.347 tries the opposite end-row source19
  approach with a180-degree jaw roll and40-degree pitch; its seed intersects
  Leaf_000 with the first finger and its bounded search found no proposal.
  These zero-motion failures retain full geometry and final native controls.
- Reduced repeated contact bookkeeping without deferring any guard: unchanged
  bucket sums are reused, while every changed bucket is recomputed from the
  complete original-order finite terms. Late normal rejection still revokes
  friction permission within the SAME consume call. No subtractive running
  total, row filtering, force rounding or callback batching is introduced.
- Passive plant-contact syntax validation now caches only immutable bounded
  plain collider strings (4096 entries); contact values/poses/eligibility are
  never cached. Native Python floats avoid generic numeric-ABC dispatch while
  bool, nonnumeric and nonfinite values retain the original rejection path.
-126 focused tests pass, including exact per-consume reference equivalence,
  zero-load headers, resets, overflow, cache bounds and nonfinite row rejection.
  Seven-repeat matched mixed-contact helper median77.078ms ->63.868ms (1.207x),
  with equal final contact records. This is helper-only timing, NOT whole-sim
  latency or a new greenhouse sequence pass. See contact_helper_timing_20260914_v1.log.
- Full regression v18:4750 passed in204.69s. No new physical sequence pass
  is inferred from these software tests. Further intact-source grasp/station
  proposals must still pass fresh native contact/retention and blade checks.
- A and C remain OPEN. Isolated A/B and greenhouse B evidence are not a full
  milestone freeze. VLM collection/training and hardware remain untouched.

### 2026-09-14 continuation: physical equivalence and blade-frame handoff

- Native349 repeats the complete intact-greenhouse right-only cut after the
  contact optimization (commit c0f32e9). All nine gates pass over85s/40800
  physics steps: cut18.71875s,3439 edge-contact samples, unloaded reverse
  complete45.177083s, passive-debris maximum2.973177N. Twenty paused native
  milestone PNGs are diagnostic evidence, not video or training observations.
  `data/sim_physics/bimanual_downward_20260914_native349/report.json` and
  `contact_cache_native349_comparison.log` contain the results. All40800
  selected physical-state/guard records match331 exactly; robot and plant
  joint arrays have zero maximum difference. This is not all-metadata equality.
- C is still NOT real time:349 tick wall547.978s, RTF0.155116 versus331's
  0.157519. The measured1.207x helper improvement did not establish a net
  full-run speed improvement. Geometry, rendering, physics and guards remain.
- Fixed a planning handoff: a successful native station-reference seed now
  supplies its associated blade-plane tilt to the bounded zero-motion search.
  An explicit cut-priority report still wins. Winning station proposals carry
  their exact tilt/normal-sign/wing family into a fresh launch as candidate
  ORDER only; execution still considers its original families and recomputes
  the complete path. Old proposals without this field remain compatible.
  Task identity, finite original family, source hashes, zero-motion and native
  control receipts are checked; no prior trajectory/clearance is inherited.
- Native350 exercises the implicit15-degree reference family but finds no
  complete proposal: its original left forearm intersects a neighboring plant.
  Native348's source19/SubStem_43 seed hits context foliage with torso1.
  Side-station351/352 are rejected before motion by torso5/right-arm1 self
  clearance (2.690mm and -0.765mm versus unchanged3mm). These are failures,
  not executed bimanual demonstrations or proofs of general infeasibility.
-137 focused tests passed; full regression v19 passed4765 tests in202.20s.
  Native353's balanced lower-torso proposal still intersects a target leaf
  with torso5; the zero-motion search ends without a proposal and with final
  native controls intact. Further kinematic proposals remain unqualified.
  No A/C completion, calibrated tissue cutting or VLM restart is claimed.

### 2026-09-14 continuation: bounded evidence-compression comparison

- Added explicit `--background-evidence-compression` comparison, OFF by
  default. Main-thread finite JSON validation/encoding and every native guard
  remain synchronous. Only immutable serialized bytes enter a two-packet
  queue; one worker compresses/writes in order. Backpressure blocks, never
  drops samples. Worker/disk errors remain sticky; close joins and flushes
  before publication. The latest archive report distinguishes closed resources
  from `archive_complete`, including final-record/trailer failures.
- Native355 is a complete full-greenhouse RIGHT-ONLY repeat using that option:
  all nine gates pass, cut18.71875s,3439 native edge contacts, reverse complete
  45.177083s, passive debris2.973177N. All40800 records were compressed and the
  worker joined without error. This native run precedes the report-only
  archive_complete field addition, which is covered by later tests.
- Read-back comparison against349: all40800 selected physical/guard records
  and full plant_dynamics match; robot-joint maximum difference0. Whole JSON
  records match40799/40800. The sole remaining difference is the final
  withdrawal receipt's Kit timeline epoch (28.533333 vs28.566667s), not measured
  robot/plant motion. Evidence: background_native355_comparison.log and
  background_native355_metadata_difference.log under data/sim_physics.
- Main-thread recording cost59.821s ->47.183s, but full tick wall547.978s ->
  556.899s and RTF0.155116 ->0.152631. NO net speed improvement is established;
  regression/candidate checks also ran concurrently for part of355. The
  option remains experimental and is NOT a faster default. Worker498.590s
  elapsed compression time overlaps control work and must not be added to
  main-thread timing as exclusive CPU time.
- Encoder comparison on six actual349 records, five repeats of180 encodes:
  finite-checked orjson median68.292ms versus stdlib83.866ms, equal decoded
  records. Retain the existing strict finite encoder. This is helper timing,
  not simulator performance or native-sensor synchronization evidence.
- Full v20 regression:4778 pass in221.90s. Latest focused76 tests pass under
  both Conda and installed Isaac interpreters, including bounded queue/FIFO,
  mutable final-fault tick, NaN rejection, worker failure and trailer failure.
- Native354 fails the left approach IK before motion. Full-rest-path offline
  filtering then finds additional source19/SubStem_43 candidates; these are
  only robot self/IK proposals. Native356 rejects its tested side station:
  torso1 versus original Gutter_31. Zero-motion final native controls pass;
  no station proposal and no bimanual motion. Testing stations outside the
  gutter with source-limit torso articulation; no plant/gutter relocation.
- A and C remain open, B retains its native passes. VLM/data/training remain
  paused. This commit is experimental infrastructure/evidence, not a freeze.

### 2026-09-14 continuation: bounded waiting search and final-query reservation

- Added explicit zero-motion `--station-waiting-search`: twelve bounded initial
  waiting offsets, retaining the same cut-entry pose and blade orientation.
  Both waiting and entry still need fresh native clearance, straight-arm entry,
  and the complete left self/interarm approach screen. No endpoint certifies a
  moving path, grasp, cut or base motion. Default waiting pose is unchanged.
- Native357 (source101/row0, prior324 frame family) intersects an original
  non-target leaf with right arm4; native358's outside-gutter forward-torso
  source19/SubStem43 seed intersects context foliage with torso1. Neither
  bounded zero-motion search produces a station; final actor controls pass.
- Native359's first expanded search exhausted20000 queries before final actor
  controls. It correctly withheld all motion authority. Fixed the search to
  reserve exact worst-case capsule-cover/box queries plus every final scene and
  robot positive control, strictly below the unchanged hard query ceiling.
  Stale epochs still invalidate; soft budget exhaustion is not infeasibility.
- Native360 exercises twelve waiting offsets and that reservation: nine base
  candidates, no proposal,19913 native queries,23.578s search/26.668s overall.
  All1095 scene and71 robot final positive controls complete, with zero physics
  steps and no scene/robot/filter changes. The reservation fixes invalidated
  search evidence, NOT the remaining bimanual access problem.
- Evidence: `data/sim_physics/bimanual_downward_20260914_native357` through
  `native360` report.json files. Regression v21 (before the final reservation
  tests) passes4783 tests in199.31s; latest focused reservation tests pass132.
- Alternate80/100/120mm grasp and180-degree jaw-roll offline proposals remain
  only full-left-path IK/self-screen results. Native361 tests a100mm grasp at
  the previously graspable344 station with unchanged contact/retention limits.
  Do not infer its outcome from the launch or from offline feasibility.
- A/C remain open; no milestone freeze, VLM restart or hardware action.
- Latest complete regression v23:4786 passed in208.00s. The earlier v22
  invocation had one subprocess import failure because PYTHONPATH omitted
  examples/greenhouse_sim (4785 passed); corrected the launch environment,
  not the assertion. Native361 stops at0.9s before grasp: the left wrist
  camera's conservative bound intersects a distal leaf of the current target.
  Native362's reversed80mm grasp acquires bilateral contact (max slip16.597um)
  but again fails static retention (utilization2.14217); no knife motion.

### 2026-09-14 continuation: exact frozen native-query reuse

- Zero-motion startup SEARCH can now reuse identical validated native-query
  inputs within its one live guarded epoch. Full float64 arguments and query
  provider identity are retained, without rounding/extrapolation. Both hits
  and misses are copied;4096-entry bound. Epoch/budget failures revoke results;
  close clears the cache. Initial/final positive actor controls NEVER cache.
  Execution startup without search retains uncached native queries.
- Native364 repeats360's source101/row0 waiting search:12633 actual native
  queries,10334 exact cache hits,18 base candidates instead of360's9. It reaches
  the unchanged45s search deadline (45.110s), not the query ceiling. No station
  proposal; all1095 scene and71 robot final controls pass, zero physics steps,
  unchanged assets/filters. This improves explored proposals per query budget,
  NOT runtime physics responsiveness or bimanual success.
-78 focused tests pass under Conda and installed Isaac Python. Full regression
  v24:4799 passed in209.87s. Covers blocked as well as clear queries, changed
  exact poses/margins/shapes/providers, cache bounds/copies, stale epochs,
  missing final actors and cleanup. Evidence: native364/report.json and
  regression_20260914_frozen_query_cache_v24.log under data/sim_physics.
- Native363's120mm/zero-pitch reversed-jaw proposal stops before grasp: left
  finger2 versus a target leaf at approach fraction0.075. No contact/cut pass.
  An offline40/60-degree pad-pitch sweep at344's fixed waiting stance finds no
  full IK/self-clear candidate. This is not global grasp infeasibility.
- Native365 is a fresh full-greenhouse right-only profiling run, not a VLM
  episode. Profiling overhead/concurrent CPU checks mean its elapsed timing
  must not be reported as an uninstrumented responsiveness comparison.
- A/C remain open. No contact cap, solver rate, plant mass, source geometry,
  surrounding visuals, tissue-calibration claim or training status changed.

### 2026-09-14 continuation: current profiling and lossless finite validation

- Native365 completes the intact-greenhouse RIGHT-ONLY action with profiling:
  all nine gates,85s/40800 steps, cut18.71875s,3439 native edge contacts,
  unloaded reverse45.177083s and fresh post-cut egress. Passive-debris maximum
  2.973177N; original assets unchanged. Archive40800 records complete. Native
  close-up inspected: curved support above the straight lower cutting edge.
  Twenty paused milestone PNGs remain diagnostics, not a continuous video.
- Instrumented tick wall667.909s/RTF0.127263 is NOT a normal-speed comparison.
  cProfile records596011921 calls,293.457s of profiled function time, including
  80.798s in recursive finite validation and100.433s in record append. These
  nested/instrumented costs are not disjoint wall-time shares. See native365
  report.json/profile.pstats; concurrent offline CPU searches also ran.
- Optimized strict JSON validation for builtin containers: finite builtin
  numeric leaves are checked in their parent's loop instead of recursively
  calling Python for each scalar/key. Original subclass handling, cycle
  detection, finite dictionary keys, integer precision, unsupported-type
  rejection, final-fault record and gzip-close behavior stay intact.
- Seven repeats of180 encodes across six actual365 records: old median
  61.274ms, inline-leaf28.087ms (2.182x). Every representative encoded packet
  matches byte-for-byte. This is encoding-helper timing, NOT a whole-sim speed
  result. Evidence: native365_inline_finite_timing.log under data/sim_physics.
  Forty-three focused tests pass under both Conda and Isaac; full v25 regression passes
  4803 tests in197.27s. No record sampling/dropping or physical changes.
- Further offline source19/SubStem41 steep-pad search with right SDK park can
  fit the left approach, but68 tested solutions cannot also clear cut entry;
  16 fail finger/cut clearance. No native bimanual pass is inferred. Including
  zero-normal-load rows in an OFFLINE344 capacity calculation leaves the same
  2.1423 utilization; do not relax the original compressive-row preflight.
- A source19/SubStem47 outer-gutter, forward-torso positive blade-plane scan
  finds six full-left-path/self/interarm/entry/finger-clearance proposals.
  Native366 tests the first, at original row0 and initial base x0.683841m,
  y-5.159018m/yaw180deg. It retains80mm grasp/20mm cut and all force/retention
  guards. Offline feasibility and a launch are NOT native success evidence.
- A/C remain open. VLM tasks remain paused; no milestone freeze is claimed.

### 2026-09-14 continuation: measured runtime gain and additional access checks

- Native367 repeats the intact-greenhouse right-only recipe without profiling:
  all nine gates pass,85 simulated seconds/40800 steps, cut18.71875s,3439 native
  edge contacts, reverse45.177083s and screened egress. Maximum passive-debris
  load remains2.973177N. Original assets/physics/guards and full-rate records
  are retained. Tick wall504.274s versus native349's547.978s: about8% less wall
  time, RTF0.168559 versus0.155116. This is still far below real time; C is OPEN.
- Serialization time falls from59.821s to36.462s. All40800 selected physical
  and guard records, including complete plant_dynamics, compare exactly;
  robot joint maximum difference is zero. Whole records match40799/40800.
  The sole difference is final withdrawal/native_static/epoch/simulation_time_s
  (28.533333333333 to28.55). Do not claim byte-identical whole-run metadata.
  Evidence: native367/report.json, inline_finite_native367_comparison.log and
  inline_finite_native367_metadata_difference.log under data/sim_physics.
- Native366 and368 both reject the source19/SubStem47 initial bimanual pose
  before physics: left arm4 versus original Truss02/Fruit03, then alternate
  left arm0 versus an original SubStem41 leaf. No contact filters, fruit,
  foliage, force limits or retention checks were removed to admit these poses.
- A read-only metadata survey of24 original plants ranks possible lower-lever
  leaf-only branches. Its leaf-lever score is NOT a measured mass/COM or native
  grasp certificate. See metadata_candidate_inventory_20260914.log. An offline
  seed71/SubStem42 search finds12 IK/self/interarm/entry proposals; native369
  tests one with original row0 geometry,80mm grasp,20mm cut and fresh guards.
  Neither offline feasibility nor launching it establishes bimanual success.
- A/C remain open. VLM/data/training/hardware remain untouched and paused.

### 2026-09-14 continuation: bounded search coverage, not a grasp certificate

- Native369/370 reject forward/upright source71/SubStem42 poses before motion:
  torso5 versus the original opposite-side backdrop plant. Despite the path's
  Gutter1_Side0 prefix, backdrop_007 is a PLANT triangle mesh, not infrastructure.
  Native371's other-side pose instead intersects a source SubStem40 leaf with
  left arm4. These remain genuine authored-contact obstructions; no shapes or
  source geometry were removed.
- The zero-motion station search now stops varying the right arm when a fresh
  native rejection names a uniquely bound collider whose complete URDF ancestry
  contains no right-arm joint. Base/left/other joints stay fixed for that station;
  ambiguous ownership, robot-vs-robot hits, missing native hits and stale epochs
  cannot use this negative-only shortcut. It never certifies a clear pose/path.
- Native372:23 base candidates,14 pruned redundant right-search groups,3384
  actual queries, all1077 scene and71 robot final controls,45.172s, no proposal.
  Also corrected candidate ordering: all sides near the original radius are
  considered before spending the budget on yaw/radius variants of one side.
  The same294 orbit candidates plus original station remain available.
- Native373:61 candidates/10 pruned groups in45.062s,3271 queries. Native374's
  horizontal-jaw alternative:86 candidates/17 pruned in45.016s,3232 queries.
  Neither finds native waiting clearance. All final actor controls complete;
  zero physics steps, no A success or global infeasibility claim.
- Full v26 regression (before the ordering test):4854 passed in237.48s,
  including robot_kinematics tests. Latest27 focused search tests pass under
  Conda and Isaac. Evidence: native369..374 report.json and
  regression_20260914_invariant_obstruction_v26.log under data/sim_physics.
- A/C remain open; B's latest full physical pass is367. VLM remains paused.

### 2026-09-14 continuation: explicit CPU scheduler comparison

- Added optional `--physics-dispatcher carb|physx` to the owned native trial.
  Default is unchanged. Sets only the process-local /physics/physxDispatcher
  preference before scene parsing, verifies requested readback after reset and
  restores the prior value (or absence) on exit. Restoration faults cannot be
  published as successful trials. No solver/iteration/rate/contact/visual edits.
- NVIDIA documents this scheduler setting in the [Omni Physics settings](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/110.0/dev_guide/settings.html).
  Configured readback is not a native scheduler getter or measured speed claim.
  Native367 used carb/false; a fresh physx/true comparison is still required.
-86 focused dispatcher/CLI/search/exit tests pass. Covers default no-write,
  true/false/missing prior settings, lost readback and failed setter restoration.
  This experimental switch is not a new qualified default or A/C completion.

### 2026-09-14 continuation: scheduler stall evidence and startup checkpoints

- Native375's explicit physx dispatcher/one-thread trial STALLS in the SDK's
  initial simulate inside SimulationManager.initialize_physics (line704),
  called by SimulationContext.reset. Repeated external Python/native stack
  samples agree; there is no control-loop trajectory or completed cut result.
  Stopped only owned Kit94928 after revalidating its command line. This is NOT
  a performance result; default dispatcher stays unchanged. See native375's
  termination_receipt.json and native375_*stack*_20260914.log files.
- Installed py-spy0.4.1 into ignored data/sim_physics/debug_tools_pyspy solely
  for that owned-process stack inspection (no model download/system install,
  no local-variable dump). Other apps and hardware remain untouched.
- Explicit dispatcher selection is now also passed through the installed
  SimulationApp extra_args BEFORE Kit initializes its native interfaces.
  Post-start/readback/restoration checks remain. Whether earlier configuration
  or more workers resolves the native stall is an UNTESTED hypothesis, not a fix
  claim. Fifty-five focused startup/dispatcher/exit tests pass.
- New exclusive startup_before_native_parse.json, startup_before_reset.json,
  startup_after_reset.json snapshots expose blocking phases. They retain the
  current report as diagnostic data, explicitly with no task/motion/training
  authority, never overwrite report.json or an existing checkpoint, and reject
  nonfinite serialization before file creation. No per-tick I/O is added.
- Full v27 regression before those final bootstrap/checkpoint changes passes
  4871 tests in234.39s. Latest focused tests cover the added changes.
- Native376 moves the source71 horizontal-grasp trial to existing end-row23
  while preserving all plants. Native startup rejects left arm4 versus the
  opposite-side backdrop006 leaf mesh. Native377's bounded search checks76
  candidates in45s with3549 queries, all1053 scene/71 robot final controls,
  and no clear waiting pose. No bimanual success or geometry removal.
- A/C remain OPEN; isolated A passes use the explicit main-stem/selected-branch
  contact fixture, not full intact-plant access. B's latest physical pass is367.

### 2026-09-14 continuation: identical physical repeat and alternate left elbows

- Native378 uses the explicit pre-Kit physx dispatcher and four workers. Unlike
  the earlier stalled one-worker375, it completes startup and all9 full-greenhouse
  right-only cut-action gates. This does not isolate which setting resolved375.
  All40,800 parsed trajectory records EXACTLY match367, including contacts,
  robot joints and plant dynamics; all report measurements also match. Native
  stepping, original source geometry/visuals,480Hz and128/0 PGS are unchanged.
  Evidence: bimanual_downward_20260914_native378/report.json and
  native378_vs367_20260914.log under data/sim_physics.
-378 tick wall time484.901786s for85s simulated (RTF0.175293), versus367's
  504.274315s (about3.84% less wall time). This one comparison is NOT real-time
  qualification, a general scheduler guarantee or a default change. Startup
  checkpoints now identify completed parse/reset phases. C remains OPEN.
- Static patch audit on344: adding a hypothetical central normal-force ray
  reduces utilization2.142288 to2.009449, STILL above the unchanged budget.
  A virtual rotation of that fixed patch also fails to establish an adequate
  grasp. Those were read-only mathematical proposals, not native contacts.
  No friction, force cap, tissue model, retention prerequisite or LP changed.
- Added opt-in --station-left-seed-search, restricted to zero-motion bimanual
  cut-station searches. At most6 exact-deduplicated initial guesses (original,
  SDK ready and bounded wrist seed variants) can expose different elbow
  solutions for the SAME pregrasp wrist frame. Seeds are not commanded poses.
  Every solution still receives native waiting/entry checks, left path/self/
  inter-arm screens, original45s deadline and reserved final actor controls.
  Right-only parked arms cannot use this option. Defaults are unchanged.
- Native379 exercises that search on source71/SubStem42 at existing row23:
  19 seed/station candidates in45.094s, no clear native waiting pose. Some
  alternate elbows change the blocking link from left arm4 to arm3, but both
  intersect the original opposite-side backdrop006 foliage. All final native
  controls pass; zero physical steps and no motion proposal. This is NOT an
  A pass or a proof that every target is infeasible.
- Full regression v28 before the left-seed addition:4879 passed in240.07s.
  Forty focused search/seed tests pass in both Conda and Isaac Python,
  including input immutability, bounded candidates, required fresh native
  checks, inter-arm rejection, query reserve and invalid-mode no-output tests.
  Full v29 after the addition:4892 passed in240.27s. An added public forwarding
  check is tested separately. No dataset/training/hardware changes.
- Offline source19 larger pad-span proposals found no combined kinematic
  candidate in their bounded scans. Source41/SubStem41 supplies9 arm/self/path
  proposals, not native certificates. Native380 is the fresh intact-greenhouse
  test of its80mm grasp/40deg pitch/180deg roll at original end-row23. Startup
  rejects left arm3 versus the original SubStem39/Leaf017; all1042 scene and71
  robot final controls complete, no physics motion. Native381 searches the
  alternate approach direction with left-elbow seeds; its result is pending.
  A/C are not frozen or complete; VLM work remains paused.

### 2026-09-14 continuation: reuse identical IK without reusing clearance

- Native381's alternate source41 approach/elbow search finds no complete
  native station in45.047s. Native382's horizontal pad proposal is rejected
  at the left palm by original SubStem39/Leaf013 and shaft geometry. Native383
  moves the base farther out and leans the torso30deg; left arm3 still hits
  the original Leaf017. None moves or changes foliage/force limits.
- Implemented bounded exact right-arm IK reuse only for the optional frozen
  left-elbow search. Original solver methods, complete parsed right-chain
  transforms/axes/ancestry, limits and torso are bound; any model/epoch change
  revokes reuse. Exact base/goal/seed inputs key up to128 immutable IK results,
  including failed IK attempts. No native clearance, self/inter-arm outcome,
  contact or motion authorization is cached. Every candidate retains those
  checks, the45s deadline and final native controls. Unknown/custom IK objects
  do not use this optimization; default non-expanded search stays uncached.
- A24-request helper comparison takes2.633323s uncached versus0.403211s with
  4 actual solves/20 exact hits. All result fields match exactly. This measures
  repeated IK only, NOT simulator dynamics, native collision latency or A/C.
  Evidence: data/sim_physics/frozen_right_ik_helper_20260914_v1.log.
-58 focused tests pass in both Conda and Isaac. Full regression v30:4910
  passed in237.23s. Includes model/torso/axis/limit/method mutation, stale epoch,
  mid-solve changes, nonfinite input, saturation, exact results and mode guards.
  Native384 is the original379 scene/search repeated with this reuse; pending.
- Work continues on original authored planting-slot selection. Current code
  fixes the detailed target to the positive-X gutter side; opposite-facing
  petioles can point into the other row. Selecting an existing negative-X slot
  must swap the same backdrop into the old target slot and keep all plants,
  spacing, source geometry and independent native checks. Not implemented or
  qualified by the results above. VLM/data/training remain paused.

### 2026-09-14 continuation: original planting-side selection, no foliage removal

- Native384 repeats379's frozen search with exact right-IK reuse:71 candidates
  versus19 in the same45s budget;813 requests,194 solves and619 cache hits.
  All1053 scene and71 robot final controls complete;3071 actual native queries.
  Still no collision-free station or physics motion. This is useful search
  coverage, not an A pass or faster physical simulation.
- Implemented explicit --target-planting-side -1 (negative X) or1 (default
  positive X), restricted to an intact greenhouse trial with an existing source.
  The selected detailed plant occupies the ORIGINAL cx +/-195mm slot; no plant
  rotation, gutter height, spacing or geometry changes. The exact displaced
  backdrop, including its original side-dependent asset identity, occupies the
  old (+X,row12) detailed-target slot. The detailed neighbor remains (+X,row13).
  All other planting positions remain. Old station proposals cannot cross sides;
  any accepted new proposal still requires full fresh native qualification.
- Native385 confirms source71 at existing negative-X/end-row23 root
  [-0.395,5.5,0.9]m. Its143 context plants plus detailed target have EXACTLY the
  same144 world planting positions and source-asset multiplicities as376.
  Native startup now rejects left arm5 versus the TARGET's Leaf034, rather
  than the old opposite-row backdrop. No physical step/cut is authorized.
  Native386's lower approach instead intersects target Leaf031 with the palm.
  These are access failures, not reasons to remove the leaves or force a grasp.
-79 focused side/layout/source/proposal tests pass in Conda (11.10s) and
  Isaac (13.06s), including original package placement, immutable source layer,
  all side/row permutations, exact displaced-asset identity, full context and
  mode/proposal rejection. Full v31:4932 passed in251.45s. Source USDs, datasets,
  reviews, splits, collection, training and hardware remain untouched.
- Native387 tests the above-branch approach on that same intact planting side;
  pending. Full-greenhouse A and responsiveness C remain OPEN. No full milestone
  freeze or reliable-tissue-fracture claim; current cuts remain force-qualified
  authored-joint release, not calibrated biological fracture.

### 2026-09-14 before-noon continuation: bounded grasp-orientation screening

- Native387/388/389/390 all stop before motion: original SubStem45 leaves,
  SubStem40/Leaf013, target Leaf031 at the palm, and target Leaf033 at the wrist
  camera respectively. No geometry/contact exclusions were introduced.
- Native391 checks alternate right elbows for the prior source101 grasp:
  12 local and16 global IK attempts, no complete native startup proposal.
  Native392's cut-frame/waiting/station search examines92 candidates with no
  complete proposal. Neither is an executed grasp/cut episode.
- Added explicit --screen-grasp-approach, restricted to a stopped native
  bimanual process-zone search. It keeps the anatomical material point, base,
  right arm, original camera/knife mounts, foliage and all source geometry.
  At most112 bounded wrist orientations with at most6 IK seeds each receive
  full finger/cut-plane clearance, native startup and47-knot self/inter-arm
  approach checks. At most8 proposals are returned in the same45s search
  budget, with reserved final native actor controls. No proposal grants scene
  path/contact/retention/cut authority; fresh launch and physical tests remain
  mandatory. Any final-control failure revokes EVERY proposed_* field.
- Native393's initial single-result search finds the finger-swapped variant of
  the known source19 grasp. Native394 collects alternatives over107 orientations
  in45.266s but finds only that same variant. Its similar physical grasp already
  failed retention in362, so it is NOT a solution to A. Native395 examines78
  orientations on source71/opposite planting side in45.172s with no proposal.
  Final native controls pass in all these searches; zero physical steps.
-49 focused search/owner tests pass in both Python interpreters before the
  multi-proposal addition.50 updated focused tests pass in Isaac, including
  revocation of all grasp proposals. The entire sim_physics test directory
  passes4909 tests in246.92s (v32); this invocation does not include extra
  tests outside that directory and is not a like-for-like count against v31.
- Native396's100mm source19 search checks75 orientations in45s and finds a
  rolled20-degree proposal with all final controls. Native397 is the fresh
  physical trial: native startup and the reobserved approach pass, and it has
  reached finger closure. Grasp/retention/cut outcome is still pending; no pass
  inferred from the search or from the first few moving steps.
- A read-only1200-record JSON encoder comparison takes0.584s current versus
  0.595s stdlib; all parsed records match. No encoder change is justified by
  this measurement. It is not a physical-simulator performance test. A/C remain
  OPEN; no freeze, dataset, training, hardware or source-asset changes.

### 2026-09-14 continuation: physical100mm grasp and numerical comparison option

- Native397 physically verifies the source19/100mm/rolled20deg bilateral grasp
  after fresh startup and moving-scene approach checks. Max material slip is
  16.998 micrometres. It stops at8.975s, before knife motion: static retention
  utilization2.143519 exceeds the unchanged contact budget. Detached mass
  12.385444g,13 compressive patch rows. Moving the grasp farther did not solve
  capacity; no cut/retention credit. Native398's flat-pad alternative hits the
  main stem/truss/fruit with left arm5. Native399's orientation search returns
  no proposal. Native400's above-branch alternative hits target Leaf006 with
  right arm5; zero motion. Native401 screens withdrawn right waiting poses.
- Read-only flat-pad IK proposals with the right arm parked can satisfy
  kinematic/self/inter-arm approach checks, but these do not prove native
  scene access or contact capacity. Evidence: kin19_flat_park_20260914_v2.log.
- Added96/0 as an EXPLICIT numerical-convergence comparison between the
  existing64/0 failed comparison and128/0 baseline. Default remains128/0;
  480Hz, original collision/visual assets, contact accounting, force/slip,
  retention and full-through-stroke gates are unchanged. Both plant and robot
  must receive the matching explicit count before parsing, with USD readback.
  No native96/0 result or fidelity equivalence yet; not a faster demo default.
  Fifteen focused profile/solver tests pass in both Python interpreters.
  Initial added negative test changed an overridden earlier argv occurrence;
  corrected test changes the effective last value and confirms pre-launch
  rejection. No application was started by that failing Conda test.
- Native401 finds a native-clear above-branch grasp / withdrawn right waiting
  proposal at world offset[0,-0.05,0.03]m from SDK ready, with final controls.
  Native402 starts physics but stops at0.9s reobservation: the conservative
  complete-current-target screen overlaps right ArcContacts/Part_11 with distal
  Leaf006. This is a planning rejection, NOT a measured damaging contact or
  a grasp. Native403's additional50mm wrist retreat/20mm lift fails IK before
  native startup. None grants cut authority.
- Full sim_physics directory v33 passes4913 tests in236.64s. Native404 is the
  explicit96/0 greenhouse right-only comparison against378's128/0 recipe,
  using the same physx dispatcher/four workers,480Hz and all original scene/
  contact checks. Pending result; default remains128/0. It cannot qualify
  bimanual retention or calibrated tissue fidelity even if its direct cut passes.

### 2026-09-14 continued: waiting-pose clearance and measured gravity settling

- Native404 completes all9 direct-cut gates at96/0,85 simulated seconds,
  40800 steps and1275 renders. Tick wall498.997912s,RTF0.170341 versus
  native378's484.901786s/0.175293 at128/0. No measured speed benefit; default
  stays128/0. Cut at18.71875s,3438 edge contacts; passive debris peak3.476720N.
  This is not exact numerical equivalence or bimanual qualification. Full A/B/C
  was not finished by the requested noon KST deadline; no milestone freeze.
- Native405's20mm wrist-axis retreat fails complete native startup: knife
  ArcContacts/Part_10 versus original target Leaf006. No physical step.
- Implemented improved explicit stopped waiting search: complete right-arm/tool
  conservative5mm clearance from original resting target, no seam exception;
  up to8 alternatives;10/20mm local refinement around native-clear coarse poses,
  bounded120 candidate inventory and unchanged30s/final-control reserve. Each
  keeps the original47-knot left self/inter-arm screen and whole native startup
  query. No live movement, contact exclusions, source or force changes. Final
  native-control failure revokes the entire proposed_waiting_poses list.
- Native406 checks the24 coarse poses and returns the same401 proposal.
  Native407 checks34 coarse/refined poses and finds4 waiting alternatives,
  with final native controls. This is zero-motion search evidence only.
- Read-only native402 trajectory comparison finds40.487094mm maximum body
  centre displacement from first fetch to0.9s; body22 moves[-19.113,-7.334,
  -34.930]mm and rotates. Knife edge moves only0.201789mm. The5mm rest margin
  cannot certify this gravity settling. No hidden pose compensation or relaxed
  moving-target screen was added. Native408's+10mm world-X alternative clears
  startup but still stops at0.9s on BladePlateContact versus settled Leaf006.
  Native409 tests the-20mm alternative with fresh physical checks; pending.
-34 focused waiting/owner tests pass in Conda and Isaac, including source/pose
  immutability, real shaft/leaf blocking, no waiting-seam exception, native
  query requirement, multiple alternatives, stale epoch rejection and final
  revocation. No claim that these unit tests prove physical retention or cuts.
- Installed SimulationContext/PhysicsContext source confirms physics-only
  ticks call native simulate directly (no per-tick app/render update), and
  native404 already has Fabric/USD transform updates disabled. Neither a
  presumed UI update nor enabling Fabric again is a demonstrated speed fix.
  Full-greenhouse A and responsiveness C remain open; VLM/data/training paused.

### 2026-09-14 continued: measured settling proposals and explicit retention experiment

- Committed waiting search e4b5f5e. Full directory regression v34:4917 passed
  in227.93s. Native409's-20mm alternative still stops at0.9s on the current
  target screen; no grasp or cut. No collision exclusion added.
- Added --screen-settled-waiting, an explicit noninteractive initialization
  diagnostic which stops before arm approach/grasp/cut. It binds the current
  record/step/time/native frames, original target identity and static cache.
  Up to35 wrist-translation candidates receive independent FK/source-limit,
  all left self/inter-arm knots, original-rest and observed-target checks.
  At most8 geometric restart proposals,30s limit; never motion authority.
- Native410's endpoint-only first version finds7 proposals in7.438s. Its fresh
  physical candidate411 passes startup but contacts Leaf006 at0.097917s:
  knife ArcContacts08/10, total normal-plus-friction0.925449N; the guard stops.
  This is a FAILED trial, not safe cutting. Native411 briefly included a
  contact-buffer copy experiment, with unchanged accounting/forces; that
  source change was subsequently reverted as described below.
- Strengthened the diagnostic to retain EVERY guarded settling step, reject
  missing/repeated/nonrigid/stale history and enclose the sampled motion in
  conservative swept leaf hulls/shaft boxes. No source/native collider change.
  Native412 includes432 samples,35 candidates,7 geometric proposals in7.5s.
  This does not prove continuous-time clearance or that a new simulation
  reproduces the identical settling path; fresh native guards are mandatory.
- Native413 uses proposal3: right wrist world offset[0,-0.04,-0.04]m from401's
  waiting pose. It passes original startup AND moving-scene left approach,
  achieves bilateral grasp, max material slip18.651873 micrometres. It stops
  at9.06875s before knife planning: static utilization3.217302, detached mass
  12.385444g. Evidence: native413/report.json and grasp_close.png. Full original
  surroundings, source assets, cameras, knife and native force limits remain.
- Added explicit --native-retention-trial to TEST a different retention
  assumption, not to hide this failure. The default still refuses over-budget
  static balance. In this headless full bimanual through-stroke experiment,
  only a valid, bound, solved over-budget patch may continue to independently
  checked knife planning; the failed static result and changed assessment mode
  stay recorded. Actual bilateral/dwell,3mm slip,0.5N per-finger all-contact,
  blade alignment/force/travel, scene, through-stroke and withdrawal guards are
  unchanged. Full native-retention gate still requires the physical outcome.
  Missing/unbalanced/invalid data still refuses. A flexible branch may droop
  rather than sustain the original six-axis static orientation; whether this
  succeeds under our actual contact model is UNKNOWN. Native414 tests it; pending.
-57 waiting/owner tests pass in both interpreters; full directory v35:4940
  passed in242.77s before the retention-policy addition.63 focused retention/
  preflight/settled-search tests pass in both interpreters after that addition.
- A one-pass contact-buffer copy experiment preserved test accounting but a
  6000-header Python-stub comparison showed no consistent speed benefit (16-row
  case slightly slower). Reverted both implementation and its specific test;
  original callback remains. Evidence: contact_copy_comparison_20260914_v2.log.
  v1 failed due missing PYTHONPATH and measured nothing. No native speed claim.
- Read-only host snapshot during412: Ryzen9950X,32 logical processors,23% load,
  7587MiB available RAM,6077 pages-input/s,451 page reads/s,267089309696-byte
  paged pool. Existing severe host memory pressure remains a timing confounder;
  no reboot, unrelated app shutdown or launch-reserve bypass. A/C remain open,
  milestone unfrozen; VLM/dataset/training/hardware untouched.

### 2026-09-14 continued: prune redundant transit work after failed endpoint IK

- Native414 repeats413's actual stable bilateral grasp, max slip18.651873um.
  The explicit native-retention experiment records the failed static patch
  and continues only to independent planning. It does NOT reach knife motion:
  native static query wall budget expires; plan remains null and no cut occurs.
  Planning68.387814s;18 endpoint candidates,8 IK attempts,0 converged. Nine
  rigid-stroke rejections involve the two wrist-camera envelopes. No retention
  or cut result exists for this experimental mode yet.
- Found redundant staged-planner work: after one nominal wrist template called
  the shared endpoint IK and it failed, remaining templates repeated expensive
  rigid sweeps before consulting the same failed result. Added an optional
  stop predicate to try_modes and bound it only to that known failed solve.
  It returns failure, not clearance. Other paths still try every fallback;
  rigid-stroke conflicts still reject before IK, so existing collision-order
  safeguards remain. No change to source geometry, force/slip caps, blade
  gates, templates, candidate order, IK guesses or physical timestep.
-86 focused planner/transit/fallback tests pass in Isaac12.45s and Conda8.74s.
  The new synthetic complete-planner case checks one sweep rather than all17
  modes for each of50 failed endpoints. Full directory v36 before this change:
  4957 passed in234.97s. Full v37 and physical native415 are pending.
  Native415 repeats414 to assess actual search coverage/outcome, not to claim
  native physics FPS improvement. A/C remain open; no full milestone freeze.

### 2026-09-14 continued: endpoint-first joint-fallback search, measured result

- Native415 confirms the first pruning change alone is insufficient:68.127904s,
  18 endpoint candidates,8 attempted IK,0 converged, native query wall timeout.
  Many nominal sweeps were all rejected before the shared endpoint was tried.
- For explicit joint-transit fallback only, moved that already-required IK
  solve ahead of nominal approach previews, still AFTER full rigid cut-stroke
  screening. Both the old Cartesian and joint branches required the same solve;
  a failed result could reach neither. Non-fallback mode retains lazy IK and
  the stop-after-known-failure behavior. No templates, physical checks, source
  geometry, contact limits, candidate order or IK seeds were removed/changed.
- Native416 repeats the same physical grasp (max slip18.651873um) and checks
  ALL50 candidates in14.852184s instead of timing out after18 in68.127904s.
  25 rigid-tool conflicts,25 attempted IK and0 converged, no path/cut. This is
  about78% less planning wall time with greater search coverage; it is not an
  FPS gain, safe motion certificate or global infeasibility proof. Left/right
  camera interference blocks one heading; tested opposite-heading IK fails.
-87 focused planner/transit/fallback tests pass in Isaac12.15s and Conda11.35s.
  Full directory v37 (before endpoint-first addition):4960 passed in245.53s.
  Full v38 is pending. Native417 tests the previously physically verified
  source19/100mm/rolled20-degree side grasp with the explicit native-retention
  experiment and all original cut/force/scene guards; pending, not a pass.
  A/C remain open; B's prior direct-cut evidence remains separate. VLM work
  remains paused and the simulator is not frozen as a finished milestone.

### 2026-09-14 continued: invariant forearm corridor rejection before IK

- Native417 physically repeats the source19/100mm/rolled20-degree grasp,
  maximum material slip16.998003um. In the explicit native-retention experiment
  the failed static assessment stays recorded. Cut planning times out after
  60.223246s at the FIRST blade placement: endpoint IK converges in11 evaluations,
  but rebuilt strokes repeatedly fail the unchanged3mm self-clearance margin.
  LeftWristCamera/D405/BodyCollision versus right-arm5 capsule_00 gives2.700536mm
  at stroke offset0.780798mm;30 joint-fallback members were considered,29 with
  clear endpoints. No knife motion, cut, traversal or retention qualification.
- Traced the repeated geometry through actual Model A1.2: the arm5 capsule is
  coaxial with revolute right_arm_6; tool_right is a fixed transform. At a fixed
  blade pose this volume cannot move with elbow redundancy. Added a necessary
  subset test to RigidToolScreen before expensive endpoint/transit IK. It proves
  the chain and cached native capsule alignment; unknown chains, off-axis shapes
  or boxes are NOT treated as invariant. Numerical radial error is projected
  and subtracted from radius, making a contained rejection subset. Original
  collider identity/pair filters stay intact; no stage or native shape edits.
  Full original arm/plant/native checks remain necessary for any accepted path.
-52 focused tests pass in Conda2.69s and Isaac2.77s, including80 arbitrary
  actual-model FK configurations, unknown/noncoaxial rejection and source-state
  preservation. Full v38 before this change:4961 passed in262.46s. v39 pending.
  Native418 repeats417 with this planning change only; pending, not a cut pass.
  Native FPS, A and C remain open; no safety margin, force cap or visual-fidelity
  reduction. Direct-cut B remains supported by the separate prior native runs.

- Native418 reaches the seventh placement in14.970s and executes the right
  approach while maintaining left contact. True vertical stroke, tilt0,
  normal+1, edge wing-9.158993mm; independent original joint transit passes.
  Force-qualified source-crossbar release occurs at24.141667s: peak signed
  resistance0.231081N, all-tool bound0.378111N, measured loading travel0.300249mm,
  arc-up cosine0.999390, downward cosine0.999999999, grasp slip13.862682um at
  release. This is modeled attachment release, not calibrated tissue fracture.
- The complete trial FAILS at24.245833s: per-finger all-contact bound0.567389N
  exceeds the unchanged0.5N cap. Maximum slip0.985996mm; full forward traversal,
  retention and withdrawal remain unverified. No successful-A claim. This
  materially advances beyond417's planning failure but is not a freeze point.
- Full directory regression v39:4972 passed in261.13s. Native418 was concurrent
  with this regression, so wall-step timings are not an uncontended benchmark.

### 2026-09-14 continued: load-aware measured-jaw backoff

- Committed invariant corridor fix bee432e. Inspecting418's last valid native
  records shows that release drives the jaws outward faster than the5mm/s
  reference backoff. The old position/damping controller therefore increases
  closing effort to0.294813N even as a high contact load requests backoff.
  This is a controller limitation; the static-over-budget grasp may ALSO remain
  physically unsuitable. No assumption that reducing effort proves retention.
- In the explicit symmetric preload-force-servo profile, preceding guard-
  accepted ALL-contact load above0.4N now constrains closing PD to half the
  existing0.24N preload request. The target interval is derived from current
  measured jaw q/v, including damping, while intersecting both original0.3N
  PD intervals and aperture bounds. Only targets change. Invalid/no-overlap
  intervals refuse; jaw positions/velocities, contact forces and friction are
  never set. The0.5N native all-contact guard and3mm slip limit remain intact.
  Lower-load recovery uses the existing slow closure. This engineering backoff
  is uncalibrated and needs native holding evidence; legacy profiles unchanged.
-88 focused tests pass: Conda0.86s, Isaac1.09s. Recorded418 q/v reproduction
  verifies that the new reference reduces closing effort despite moving jaws;
  stale observations still reject, and no physical-state setter is called.
  Full v40 and matched native419 are pending. No milestone freeze or VLM work.

- Native419 reproduces the same source-crossbar release at24.141667s. The
  previous finger-overload stop is avoided, but the trial fails on3.035987mm
  material slip (>3mm). Native retention, full traversal and withdrawal remain
  unqualified. Backoff is not a substitute for a load-capable grasp; the earlier
  static-over-budget warning remains relevant. Do not enlarge the slip limit.
- Full v40:4989 tests pass in250.50s. The next physical experiment shifts only
  the hold point farther along the same detachable petiole, retaining the
  original cut location and surrounding geometry, and REQUIRES the default
  static-retention prerequisite before cutter planning. No dynamic override.

### 2026-09-14 continued: distal hold geometry, static gate retained

- Native420 did NOT start: the public experimental arc range stopped at120mm.
  Expanded that proposal-only range to60..180mm; default80mm is unchanged.
  The backend's exact in-segment material point, detachable side, complete
  finger/cut clearance, IK, native contacts and retention checks still apply.
  This is not a force, collision or tissue-limit change. The larger distance
  is a deliberate tradeoff against the preference for holding near the cut:
  reducing the detached branch's gravity lever may permit safer retention.
- Native421:160mm with the previous20-degree pitch fails initial left IK;
  no native motion. A read-only source-geometry/Model A screen checks nearby
  arcs and pitches with original robot shapes, all47 self/inter-arm knots and
  finger/cut bounds. v2 finds6 kinematic proposals in13.594s, NOT native scene
  or grasp approval. v1 failed on a diagnostic configuration-key typo and
  produced no result. Logs: distal_grasp_screen_20260914_v1/v2.log.
- Native422:160mm/-20-degree pitch passes kinematic screening but native
  startup rejects the left forearm against original Fruit_00 and Truss_00.
  Native423:160mm/0-degree pitch passes startup; after0.9s settling, Leaf_000
  on Segment_007 obstructs the left finger corridor. No grasp/cut in either.
  These results show why kinematics alone is insufficient; no foliage removed.
- Native424's45.032s zero-motion native orientation search proposes one above-
  shaft approach (shaft rotation-90 degrees,pitch0,roll180), with final native
  actor controls passing. It does not certify settled approach/contact or
  retention. Native425 tests this proposal at160mm, with the original default
  static-retention prerequisite and without --native-retention-trial; pending.
-25 public-launcher tests pass in Isaac0.41s. Full v41:4993 passed in256.59s.
  Grasp-plus-release is measured in418/419, but full greenhouse A and overall
  responsiveness C remain OPEN. No freeze, dataset collection or VLM training.

### 2026-09-14 continued: settled left-wrist path, bounded elbow replanning

- Native425 rejects the above-shaft160mm proposal after0.9s settling: nominal
  left-arm1/torso5 clearance2.803051mm is below the unchanged3mm self margin.
  No arm approach or cut was executed. A clear frozen initial configuration
  does not certify the gravity-settled path.
- Added bounded local redundant-IK replanning only for a valid self-screen
  rejection after the initial knot. The measured starting joint vector and
  desired wrist path are preserved. All proposed transitions include interior
  samples, at most0.5-degree joint increments,0.5mm wrist-position error and
 0.005rad orientation error, with original self/inter-arm limits. No partial
  path is published on failure; the complete current plant/native corridor
  screen still follows before motion. Invalid starting poses and query errors
  retain their original refusal; this does not authorize plant contact.
- Native426 clears the torso conflict in3.284480s (97 path points,245 checks,
 102 alternative poses), then correctly stops at approach fraction0.35 for
  the target branch's own Leaf_000 versus left finger2. No actual grasp/cut in
  this run. Distal retention and full greenhouse A remain unverified.
-155 focused tests passed in both interpreters before the final initial-knot
  guard. Full v42 found one regression in an existing invalid-initial-pose
  refusal test; implementation corrected without changing that test. Final
  Isaac focused check:50 passed in9.59s. Full v43:5007 passed in227.12s.
  Native426 preceded that initial-knot guard correction (its collision was at
  knot37, not0). No native-FPS claim, milestone freeze or VLM work.

### 2026-09-14 continued: reject obstructed grasp hands before arm IK

- Committed settled elbow replanning as d33a155. Added a target-only rigid
  hand approach/closure screen to the frozen grasp-orientation search. Original
  palm, fingers and wrist attachments are transformed with the actual Model A
  prismatic-finger chain. Complete sampled approach (<=1mm translation/1degree
  rotation) and <=1mm closure increments use the existing native-target convex
  hulls and shaft shapes with the unchanged1mm margin. Only the existing
  contiguous detachable shaft/finger allowance applies; leaves remain obstacles.
  No stage, native state, contact filter, force threshold or visual edits.
- The screen is a necessary sampled target subset, NOT a full-robot/context,
  native grasp, retention or continuous-time certificate. All original native
  startup and later settled whole-scene checks remain mandatory. This closes
  the earlier search gap that could propose a clear initial arm configuration
  despite an obstructed subsequent hand approach.
- Native427 completes the zero-motion diagnostic:111 tested orientations at
 160mm all reject in0.328s;81 hit Leaf_000, with other leaves/support/shaft
  accounting for the rest. Final native actor controls pass. Its diagnostic
  exit is intentionally nonzero; NO grasp or cut was executed, and no global
  infeasibility is claimed. Unlike the earlier45s search, the hand rejection
  occurs before arm IK/native candidate queries. Not a native-FPS improvement.
-26 focused tests pass in Isaac0.88s, including actual-model FK equivalence,
  complete hand shapes, leaf refusal, closure/intermediate samples and invalid
  mechanisms/frames. Full v44:5017 passed in235.17s. Initial randomized FK test
  sampled outside URDF limits; corrected the test sampling to actual limits.
- Read-only offline settled-grasp diagnostic reconstructs the original source
  geometry and checks the archived guard-accepted native4260.9s snapshot.
  v1 failed from a diagnostic variable-name collision; v2 stopped on the
  existing10-degree observed-axis guard. v3 preserves that guard, skips those
  arcs, and returns5 kinematic/hand-subset proposals in36.203s. These do NOT
  certify surrounding-plant access or retention, and require fresh native tests.
  Evidence: data/sim_physics/settled_grasp_screen_20260914_v1.py and v1/v2/v3.log.
- Inspected native426 stopped_on_fault image; original full greenhouse remains
  visible. Existing native378 timing remains0.175x real time, median native
  step7.293ms versus2.083ms requested timestep; this is historical timing, not
  a new speed measurement. Full greenhouse A and responsiveness C remain open.
  No milestone freeze, new dataset collection or VLM training.

### 2026-09-14 visible replay regression, neutral-start requirement and rate trials

- Fresh visible native428 exposed a regression: the half-preload measured-jaw
  backoff added for the unqualified greenhouse419 experiment also affected the
  otherwise stable isolated recipe. Release23.564583s was followed by grasp
  loss24.072917s. This was NOT shown as a successful end-to-end replay.
- Quarantined that response behind `--measured-jaw-backoff-trial`, requiring
  an explicit non-watched native-retention experiment. Original preload servo,
  ordinary load backoff,0.3N PD,0.5N all-contact and3mm slip limits remain.
  Native429 replays the corrected default visibly: all10 sequence gates pass,
  release23.564583s, complete forward section44.925s, unloaded reverse78.789583s,
  final withdrawal and retained branch85s. Maximum slip1.286446mm,19079 native
  edge contacts. This is a privileged isolated original-branch fixture, NOT
  intact-greenhouse bimanual qualification, neutral-start qualification,
  calibrated tissue fracture, deposit or a broadly reliable robot task.
- Native429 tick timing:85sim seconds /512.770wall seconds, RTF0.16577; median
  native step4.332ms. Regression work overlapped part of this run, so this is
  not an uncontended performance comparison or a real-time claim. Full v45:
  5025 passed275.89s, before the new neutral/rate trial modules below.
- Optional `--watch-exit-after-s` closes only a completed explicitly automatic
  watched run after1..300wall seconds of inspection; default stays open.429
  verified normal report publication and owned-app teardown.428 was stopped
  by exact PID/command verification AFTER its failed result was saved; its
  ignored demo_stop_receipt.json records this. No other apps were stopped.
- Diagnostic captures now add full_forward_stroke and knife_unloaded detail
  views at measured events, plus final grasp close-ups. Receipts include
  current contact/slip and traversal/unloading/final-withdrawal state. Capture
  remains paused native viewport output, not video or synchronized RGB-D.
- Visible greenhouse direct-cut430: release18.718750s, forward section39.525s,
  unloaded reverse45.177083s. It was stopped at48.331250s (`Stopped; reset
  required`) before final withdrawal validation, so it is NOT a full pass.
  User reported the downward motion was not evident and requested neutral
  ready start plus faster cutting. Neither429 nor430 satisfies neutral start:
  their archived torso/arm recipes are prepared near-target configurations.
- Added explicit `--neutral-ready-start`: SDK arm-ready joints with an upright
  fixed torso/base, complete source-shape/current-target/static-scene sampled
  planning, effort-limited native drive transit, no intended plant contact,
  fresh endpoint check, then fresh grasp/cut sensing without native pose reset.
  Phase clocks are separately reported; no old observation authorizes a cut.
  Native431/432 reject a tilted inherited torso before launch;433 rejects a
  report-based station/torso mismatch.434 uses explicit upright initialization:
  self margin5.611491mm passes, but native startup rejects right-arm5 versus
  original target Leaf_006 on Segment_022. Zero physics steps, no approach/cut.
  This feature is experimental, NOT an executed neutral-to-cut demonstration.
- Added explicit `--support-aware-feed-trial` for faster post-release commands:
  fresh knife AND left-finger all-contact state, bilateral grasp/slip or an
  independently verified unloaded cut-only park, control-only stability history,
  bounded1mm/s^2 acceleration and immediate slowdown/stop/backoff. Independent
  unsmoothed force/contact/slip/orientation/travel guards stay unchanged. This
  addresses native311's failed naive1mm/s trial; it is not tissue calibration.
  New focused Isaac checks:36 neutral/capture/watch tests and67 support-rate/
  traversal/retraction/neutral tests pass. Full v46:5049 passed in258.01s.
  Native435 faster isolated insertion reached full forward traversal34.629167s
  (11.064583s after release versus21.360417s in429), while retaining the branch.
  Reverse completed54.0375s, but final escape failed54.066667s at0.010063504N
  against the unchanged0.01N unloaded limit. Maximum slip1.071436mm. This is
  a partial phase improvement, NOT a full-sequence pass or a qualified default.
  The experimental profile now accelerates insertion ONLY; reverse returns to
  the original0.0003m/s contact feed. Native validation of that change pending.
  Native436 places the initial base10cm farther back (no runtime base motion):
  neutral startup and the74-point/330-check current-scene path screen pass.
  Before the first controlled step, explicit open-finger PD rejected the
  inherited0.8N total motor limit instead of its normal0.3N PD allowance.
  The prelude now applies the SAME bounded PD ceiling used during closure,
  preserving smaller limits and open targets;20 focused tests pass. Native437
  retry did NOT start:14.69GiB commit headroom is below the16GiB launch reserve.
  Reserve, contact limits and assets were not bypassed. Neutral execution and
  neutral-to-cut completion remain UNVERIFIED.
  Follow-up:96 focused controller tests pass, including insertion-only reverse
  rate selection and passive previous-drive-versus-measured-start diagnostics.
  Handoff diagnostics change no commands and establish no contact causality.
  Full v47 under Isaac's launcher:5055 pass,7 child-process import failures
  (bare kit.exe lacks NumPy). The unchanged full suite rerun with installed
  standalone Python (v48):5062 passed in246.53s. No tests were skipped to hide
  those failures. Log:physics_regression_20260914_v48.log. With explicit user
  approval after saved reviews, stopped ONLY verified review servers8880/8881/
  8882 (PIDs130120/122136/106656); confirmed ports no longer listening. No data
  or annotation files deleted. Post-stop commit headroom15.00GiB remains below
  the unchanged16GiB reserve; native retry still pending, no bypass/reboot.
  Native438 subsequently launched after memory recovered. The corrected open
  fingers executed, but neutral transit stopped near1.591667s on unintended
  plant/tool contact. Last accepted wrist tracking error1.351927mm exceeded
  the old1mm planning margin. That suggests insufficient tracking reserve;
  the old failure log did not retain the exact rejected pair, so causality is
  not established from438 alone. Neutral transit now plans AFTER the initial
  native1s hold, screens complete target and both arm/context paths at5mm
  (planning reserve only), and retains the rejected step's exact normal plus
  friction pair loads. No force, penetration, collision or slip guard relaxed.
  Focused transit checks:42 pass; expanded neutral ordering tests:13 pass.
  Native439 isolated faster-insertion/original-reverse comparison PASSES all10
  sequence gates with the original full robot/knife: release23.564583s,
  full forward34.629167s, unloaded reverse67.158333s, final native withdrawal
  and retention verified85s. Maximum slip1.271148mm;10741 native edge contacts.
  Release-to-traversal11.064583s versus429's21.360417s. Measured final drive
  handoff0.002814583deg, no compensation applied or causality inferred.
  Tick wall438.096535s for85s simulation (RTF0.194021), NOT real time and not
  an uncontended controlled performance comparison. Source visuals/physics
  unchanged. One successful prepared-pose isolated trial is NOT general
  greenhouse reliability, neutral bimanual completion or tissue calibration.
  Native440 revised neutral approach physically PASSES at9.7375s:88-point
  post-hold path,492 checks,23 search iterations,5mm planning clearance; fresh
  endpoint screen passes. Last wrist tracking error0.148035mm, no native pose
  reset. Cut continuation subsequently FAILED after62.189s planning wall time:
  the native static-query budget expired.434 path rejections repeated the same
  blade-body/stump/original-cut-window predicate. No cut executed in440; this
  does not establish global target infeasibility. Preserve
  initial motion-planning evidence separately from the endpoint recheck in
  future receipts (440's initial88-point plan is retained in its native log).
  VLM/data/splits/review decisions, hardware and source visual assets untouched.

### 2026-09-14 neutral-start bimanual demonstration retry

- Defer cuRobo integration until the initial grasp/cut/performance milestone.
  Move the unchanged source-section placement predicate before expensive
  wrist sweeps/IK/transit-family searches; retain its defensive candidate
  recheck and all native force, collision, penetration and slip guards.
  Synthetic impossible sections reject before any arm solve or tool sweep.
-90 focused tests pass in8.65s (planner, neutral, whole-target, blade aim).
  This is not a new full-suite regression or native cut qualification.
- Native441 launched as an explicit visible isolated bimanual trial: original
  seed101/SubStem_41, full RBY1/knife/cameras, upright torso, SDK-ready arms,
  guarded neutral approach,1mm/s support-aware forward feed with original
  reverse feed. No native pose reset during execution.441 mistakenly used the
  older D:/isaac-sim installation and failed before motion on the missing
  PhysxMimicJointAPI.CreateNaturalFrequencyAttr API; shutdown also rejected
  SimulationApp.close(exit_code). Do not treat441 as controller evidence.
  Memory preflight allowed launch with16.97GiB commit headroom. No reserve
  bypass, other-app closure, source visual edits or VLM/data changes.

- Native442 retried with the verified D:/isaac-sim-6.0.1 installation. Memory
  preflight passed, but native startup geometry rejected upright torso5 versus
  Segment017's stem and Leaf004/Leaf005. Robot self-screen passed (5.611mm
  minimum);361 native queries and all77 final actor controls succeeded. This
  is a real initial-placement rejection, not an IK/cut execution failure or
  a demonstrated memory crash. Zero controlled physics motion; no cut.
- Read-only offline same-target left pregrasp IK with upright torso succeeds
  at10cm and15cm initial backward station offsets; sampled endpoint interarm
  capsule clearances249.17/248.32mm.20/25cm fail this bounded solve (not a
  global infeasibility result).15cm candidate x=.4108916307204727,
  y=.9273492407943318,yaw=-147.10477763841965deg. No plant or robot motion,
  collision certificate or blade-path approval follows from this offline IK.
  It still needs complete native startup and neutral-to-grasp/cut checks.
- Later launch preflight at09:11:41Z has15.05GiB commit headroom, below16GiB;
  no memory reserve bypass or unrelated process termination. Tool access had
  intermittently failed on approval-service capacity; user-provided442 error
  was corroborated by the recovered read-only native report.

### 2026-09-14 neutral staging continuation (simulator priority; VLM held)

- Native443 moved the initial base15cm back: complete native startup passed,
  but the post-settle pregrasp goal failed5mm clearance for left finger1 vs
  Segment006/Leaf000. No neutral arm transit or cut. Its failed watch result
  was saved before closing only the verified owned paused kit PID68252.
- Added explicit `ground_truth_trial --approach-distance` within the existing
 10..80mm constructor range, bimanual process-zone trials only. Source target,
  grasp/cut arc, native checks and historical default20mm stay unchanged.
 39 launcher/neutral tests pass. Native444 (12cm retreat,40mm standoff) still
  rejects initial torso/leaf overlap. Native445 (15cm retreat,-10deg initial
  heading adjustment,40mm standoff) passes startup but again rejects the
  pregrasp leaf margin. Auto-run consumed its one-shot before the user's
  button click; it did not perform a grasp/cut. Its requested60s final pause
  ended through normal teardown. No physical guard was bypassed.
- Offline larger-standoff/heading proposals use original RBY1 limits and do
  not certify motion. Native446 selects15cm initial retreat,-20deg initial
  heading adjustment,80mm standoff, upright torso and SDK-ready arms. Same
  original plant, camera-aligned knife, physics/visuals/contact limits.
  Bounded14-joint neutral path passes:162 points,1388 checks,15 iterations.
  Actual native neutral transit and fresh endpoint screen PASS at17.129167s;
  last joint error1.4782e-5rad, right wrist error3.638e-6m, no endpoint contact
  pairs. No native pose reset. Left-grasp continuation subsequently verifies
  bilateral native grasp contact (7..8s continuation clock), maximum recorded
  slip0.015156mm through8.9s. Right entry then FAILS:50 tool-frame proposals,
  five corridor-clear IK attempts,zero converged. No blade approach or cut.
  Full bimanual cut, direct-cut neutral completion, greenhouse reliability
  and performance C remain unqualified. Do not freeze on this approach pass.
- User reaffirmed stable full-motion simulator BEFORE VLM fixes. VLM work is
  on hold; no training, collection, data/split/review edits or hardware motion.

### 2026-09-14 jointly reachable neutral-start station qualification

- Reconstruct446 blade-entry goals from native wrist/edge frames and its
  settled planning snapshot, using the actual straight lower crossbar (not
  the mounting-plate geometry).24 bounded seeds per corridor-clear pose find
  zero IK solutions at446's station. This is finite-search evidence, not a
  global impossibility proof; the best attempts reach a right shoulder limit.
  Evidence: `data/sim_physics/right_entry_20260914_v1.log`.
- Joint station search checks left pregrasp/grasp and the same right blade
  entry with upright torso, original limits,3deg right reserve,10mm interarm
  capsule clearance and0.8..0.98 right extension. Nine of48 sampled stations
  pass these offline endpoint checks. No native collision or motion approval
  is inherited. Evidence: `data/sim_physics/joint_station_20260914_v1.log`.
- Native447 tests x=.4,y=.75,yaw=-157.10477763841965deg,80mm pregrasp,
  explicit left IK seed and SDK-ready waiting arm. This candidate also solves
  right entry from the SDK-ready seed offline. New isolated output only;
  startup, full neutral path, grasp, complete cut corridor and native contact
  checks still required. Launch reserve passed with22.59GiB commit headroom.
  All VLM work remains held until simulator milestone evidence is complete.

- Native447 rejected initial torso5 vs Segment017/Leaf004+005 before motion.
  Endpoint reachability alone was insufficient. Added an explicit bounded
  `--screen-station --neutral-station-candidates FILE` diagnostic: same-target
  finite proposals only; upright SDK-neutral whole-native clearance first,
  then5mm pregrasp clearance, both-arm IK/self/extension and complete right
  entry geometry with left still pregrasp. No grasp-contact exceptions,
  scene writes, physics steps, path or execution authority. Existing native
  epoch checks and final actor-control reservation remain in effect.
- Native448 checked nine candidates of a48-candidate batch, rejecting eight
  torso/leaf conflicts before IK and finding x=.52,y=.75,yaw=-147.1047776384.
  Neutral, pregrasp and entry native geometry pass; interarm capsule bound
 205.834mm, right extension0.967255. Final native controls PASS:71 robot actors
  before/after,1686 native queries,zero steps andzero authored-pose changes;
  total startup screening1.742132s. This is NOT a complete moving-path result.
  Evidence: `data/sim_physics/bimanual_downward_20260914_native448/report.json`;
  input `data/sim_physics/neutral_station_candidates_20260914_v1.json`.
- Added optional `--right-entry-seed-degrees`: source-limit endpoint IK hint
  only after the original ready-seed attempts fail. Never changes the parked
  arm or native body poses; stroke/transit IK continues from its predecessor.
  All existing endpoint, complete approach, rebuilt stroke and native contact
  checks remain.88 batch/lifecycle/neutral regressions pass;93 entry/planning/
  launcher regressions pass. Complete v49 standalone regression PASSES5108
  tests in260.25s. Subsequently added clearer one-shot UI status/blocking-pair
  messages (18 focused UI tests pass; native visible rendering still pending).
- Native449 uses that station and the endpoint hint, starts from SDK neutral,
  and revalidates from scratch. Actual neutral transit, left grasp and right
  approach PASS. Native force-qualified crossbar release at24.829167s,8221
  native edge contacts, maximum slip1.559367mm; retention and material
  separation gates pass. Full through-stroke FAILS its35s post-release timeout
  near59.829s, with roughly1.5mm remaining to the commanded endpoint and
  sharp-edge clearance still below its0.5mm requirement. Withdrawal does not
  begin. No end-to-end cut or performance qualification claimed. This is a
  pre-authored attachment release model, NOT calibrated tissue fracture.
  Diagnostic search448 intentionally ends before any motion. All native
  contact/penetration/slip limits and source visuals/geometry remain unchanged.

### 2026-09-14 neutral-start direct cut passes; bimanual traversal comparison

- Committed the guarded station/entry planning and one-shot UI checkpoint as
  `b424c5f` on `koh-dev/sim-vlm`; NOT a simulator freeze or general reliability
  claim. VLM work remains paused. No hardware, training or data decisions changed.
- Native450 direct cut from SDK neutral PASSES all10 gates in isolation:
  left parked/open/unloaded, full right approach, qualified release19.4375s,
  full measured forward stroke28.114583s, unloaded retraction38.3125s and
  final native withdrawal85s.3855 native edge contacts. Released debris peak
  contact3.785N is reported separately under the user-permitted post-release
  landing policy; it is not a knife-load exception. One run, not repeatability,
  greenhouse validation, deposit or calibrated biological fracture.
  Evidence: `data/sim_physics/bimanual_downward_20260914_native450/report.json`.
- Read-only replay of449 versus439 observations shows no support-cap stops.
  After32s,449 has401 fixed-rate reverse decisions; median full-knife load
 0.375N (peak0.428N), versus439 median0.356N andno reverse decisions in its
  remaining forward interval. Replay is diagnostic, not native action evidence.
  Last449 edge clearance0.361mm remains below required0.5mm. Blade contacts
  are proximal and distal cut faces, not a neighboring leaf or gripper.
  Evidence: `data/sim_physics/postrelease_comparison_20260914_v1.log`.
- Added explicit `--proportional-face-backoff-trial` for a matched comparison:
  retain0.4N reverse onset,0.5N hard native limit, signed-load onset0.28N,
  maximum reverse0.5mm/s, original slip/penetration/35s timeout and complete
  measured section/withdrawal requirements. Only reverse magnitude scales
  with instantaneous excess load (gain0.005m/N/s); no filtered safety evidence
  or collision/geometry edits. Default profile remains unchanged. Save the
  successfully issued through-stroke command separately from observations.
 122 focused force-feed/section/launcher tests PASS. Complete v50 standalone
  regression PASSES5129 tests in248.20s, including UI and feed changes.
- Native451 matched neutral-start bimanual comparison retains449's release
  time24.829167s,8221 edge contacts andmaximum slip1.559367mm. Gentler backoff
  improves the observed sharp-edge clearance (about0.401mm at52s vs449's
  final0.361mm), but FAILS the same35s traversal timeout. All retention and
  release gates pass; full forward/withdrawal gates do not. Keep this explicit
  comparison disabled by default. It is negative evidence against treating
  controller-rate tuning alone as the fix; do not freeze on joint release.
- The source crossbar is3mm thick (half1.500000013mm in440's original source
  section receipt). Conservative whole-blade/stump clearance and the existing
 3mm cut window must both be respected when proposing offsets; do not remove
  the stump collider or enlarge the acceptance window to force success.
  Further bimanual work must address blade/cut-face accommodation while
  retaining the branch, not rely on uncontrolled tracking error or extra time.
- Native452 begins fresh full-greenhouse direct-cut validation using450's
  neutral station/entry hint, original surroundings and default backoff.
  Startup REJECTS right wrist D405 body vs neighboring Gutter1_Side2_013
  main stem before motion. Do not reuse isolation placement without fresh
  surrounding-plant checks. Current C remains unqualified:450 measured RTF0.22044,
  median native physics4.596ms vs480Hz's2.083ms budget, plus controller work.
  No visual or physical-detail reduction, VLM restart or hardware action.

- Native453 runs the bounded zero-motion neutral/pregrasp/entry batch in the
  full greenhouse with48 local placement/heading proposals around450's
  station (`neutral_station_candidates_20260914_v2.json`). All neighbors and
  camera geometry remain. It can only propose a fresh launch, not move the
  robot or certify a cut. All48 candidates were checked in9.870488s with18670
  queries and successful final native controls. No proposal:30 neutral scene
  conflicts,6 pregrasp IK failures,12 entry scene conflicts. No physics motion.
  This finite local grid is not a global infeasibility proof. Both a fully
  clear greenhouse station/path and retained-branch blade accommodation still
  need implementation/validation before the requested A/B/C simulator freeze.

### 2026-09-14: neutral retained-cut accommodation (native454/455)

- Added the default-OFF `--retained-separation-trial`, restricted to the guarded
  480 Hz bimanual measured-section cut action. This addresses native449/451's
  post-release full-blade-passage stall without relaxing load limits, extending
  the traversal timeout, enlarging the 3 mm cut window or removing cut faces.
  `sim_physics/retained_separation.py` plans a 2 mm **palm command** along the
  current blade normal toward the detachable side after verified native release.
  Seventeen sampled left-arm/branch configurations check the whole robot, every
  original right stroke sample, protected plant and static context. The branch
  sweep is a conservative rigid-translation proposal, not a deformation model.
  Only the already released adjacent shaft pair has its existing contact
  allowance; the proposed translation must not decrease cut-face separation.
- Normal joint drives execute the left path, at at most0.5mm/s, with unchanged
  force-controlled fingers. No grasp weld or plant/robot pose assignment. Positive
  knife feed waits while overload backoff remains. Same-step native contact,
  slip, tool/finger load and tracking evidence is mandatory. Both palm and held
  material must actually move for25ms; a new frozen-native corridor check must
  pass before downward feed resumes. The original full-section traversal,
  unloaded reverse, egress and final native withdrawal checks remain mandatory.
  A separate `retained_separation_reobserved` gate is included from455 onward.
- **Native454 and native455 both PASS isolated neutral-start bimanual action**
  on the same source seed101_full/SubStem_41 and station [.52,.75,-147.1047776384].
  Exact common timings (protocol seconds after the neutral approach): native
  release24.829167; separation/reobservation28.8625; full sharp-edge
  traversal40.35625; unloaded reverse74.008333; final native withdrawal85.
  Native454 has all11 existing gates true plus verified separation;455 has all12
  including its explicit separation gate. Both record5700 native edge contacts,
  maximum grasp slip1.752025mm (<3mm), and no released-debris landing contact.
  At separation completion the actual palm moved1.978312mm and held material
  2.531387mm:2mm is the command bound, NOT a clamp/teleport of the physical plant.
  Blade clearance at completed traversal is0.500735mm, within the original3mm
  seam. This is two repetitions of ONE isolated fixture, not a population-level
  reliability result, greenhouse qualification, deposit or calibrated fracture.
- Comparison to449 is controlled: same neutral posture, source geometry, force
  limits, IK hints and pre-release timing.449 failed full traversal at35s after
  release while retaining.454/455 relieve that blockage through actual held
  target motion. Default proportional-backoff experiment stays OFF. No source
  meshes, visual fidelity, collision filtering, hardware or VLM data changed.
- Speed remains OPEN:454 RTF0.175325 with overlapping offline regressions;
  455 RTF0.181859 (85 simulated seconds/467.396 tick-wall seconds). These are
  not real-time passes. Greenhouse startup/access failures452/453 remain
  unresolved; A/B/C is NOT frozen and VLM collection/training remains paused.
- Added `examples/greenhouse_sim/run_neutral_cut_demo.cmd` with explicit
  `bimanual`/`right_only` modes and a mandatory NEW output directory. Uses the
  measured neutral station/IK hints, correct Isaac6.0.1, original memory preflight
  and a manual Run-once panel. The local native293 report is still required only
  for cut-family ordering; this launcher does not inherit an old clearance/path.
  The direct mode matches450 and does not enable retained-target movement.
- Visual audit found the old knife/plane diagnostic cameras remained aimed at
  the INITIAL neutral knife, sometimes photographing the base instead of the
  action. `knife_inspection.py` retargets ONLY those three external cameras from
  the fetched native right wrist at milestone capture. Robot D405 views,
  calibration and physics are untouched. These are paused diagnostic images,
  not synchronized training data or motion video. Native456 is the final
  recapture/qualification run with this diagnostic correction.
- Regression evidence so far: full suite5153 passed in264.35s
  (`physics_regression_20260914_v51.log`); subsequent protected-neighbor,
  launcher-mode and camera tests pass in focused runs. Final full-suite v52 and
  native456 results will be recorded below before committing the checkpoint.
- **Final native456 PASS, all12 gates**, with identical release/traversal/
  retraction timings and maximum1.752025mm slip. This is3/3 repetitions of the
  same isolated fixture, not target/randomization robustness qualification.
  Final source/camera run RTF0.178497 (476.199 tick-wall seconds for85 simulated
  seconds, overlapping offline regression load). Corrected front/back close-ups
  were inspected: they now centre the actual tool/target rather than the base;
  gripper/plant occlusion remains visible in some angles. Capture receipts prove
  no plant motion during each paused image. `full_forward_stroke_blade_back.png`
  is step19371/t40.35625 with bilateral contact,0.926825mm slip and full-forward
  verification; it does NOT claim withdrawal already complete at that step.
  Final withdrawal is separately verified at85s. Evidence directory:
  `data/sim_physics/bimanual_downward_20260914_native456` (`report.json`, full-rate
  gzip trajectory and native milestone PNGs). These are not motion videos.
- Final complete regression: **5160 passed in257.53s**,
  `data/sim_physics/physics_regression_20260914_v52.log`; `git diff --check` clean.
  The demo launcher's no-argument usage guard was tested; the new captures are
  from headless native execution, not a claimed manual UI acceptance test.
  All owned validation jobs completed. No dataset/review/training changes,
  hardware control, unrelated application shutdown, reboot or fidelity reduction.
  This checkpoint resolves the tested isolated retained-cut blockage; the full
  greenhouse neutral A/B paths and responsive C still precede any VLM restart.

### 2026-09-14 ? neutral greenhouse access search and same-fetch hand poses

- Continued A/B/C engineering on `koh-dev/sim-vlm`; VLM collection/training,
  source assets, annotations and splits remain untouched. This is NOT the
  requested greenhouse/real-time freeze. The isolated456 checkpoint remains
  the comparison case; no force cap, solver rate/iterations, cut angle, source
  collider, visual detail or retention requirement has been reduced.
- `neutral_station_search.py` now accepts bounded per-station redundant-right
  IK exploration (0..16 steps per direction, same wrist pose, original3-degree
  joint reserve). Every alternate arm configuration receives the same native
  whole-robot/scene entry check. The40s search and reserved final actor controls
  remain; search output never authorizes motion or proves global infeasibility.
- Added explicit per-candidate cut-frame families restricted to the existing
  vertical-crossbar orientations (0/+/-10/+/-15-degree tilt, normal sign +/-1,
  bounded edge offset). Current anatomical direction and source edge geometry
  still decide feasibility; all cutting/contact alignment gates are unchanged.
  `station_proposal.py` can import a final-controlled zero-step neutral proposal
  only into a fresh neutral-start launch. Entry IK seeds remain distinct from
  actual ready joints. A conflicting explicit cut priority is rejected BEFORE
  changing launch arguments; no old paths or collision approval are inherited.
- Native457:3 local stations/51 redundant entry alternatives, no clear proposal.
  Native458:30 station/frame combinations;20 entry/grasp IK rejections,5
  self/extension rejections and5 entry-scene rejections; no proposal. Native459:
  48 wider station/heading/frame combinations;44 neutral-scene collisions and4
  entry IK failures; no proposal. Final native controls passed in all three,
  with zero physics steps and no budget exhaustion. Reported screen wall times:
  native457:5.680s; native458:13.982s; native459:4.993s. These are negative bounded SEARCH results,
  not executed cuts. Local entry collisions chiefly involve the right forearm
  and SubStem_44/Leaf_028, or neighboring fruit; the plants were not removed.
  Evidence: `data/sim_physics/bimanual_downward_20260914_native457` through459.
- Extended neutral station screening to the explicit parked-left direct-cut
  strategy. In that mode the left SDK joint posture stays relative to EACH
  candidate base; it does not solve an irrelevant old-world grasp goal. This
  cannot replace a required bimanual grasp; mode and park flags are validated.
  Endpoint success remains only a proposal requiring fresh full-motion testing.
- Added `fetched_hands.py`: one local post-fetch snapshot of both finger pads
  and both wrists, sampled anew every physics step. The identical native XYZW
  conversion is batched once. Grasp, diagnostics, cutting and forward-traversal
  checks share it only with matching robot identity and step ID; wrong-step
  reuse fails closed. Arrays are owned/read-only; no persistent cache, FK pose,
  interpolation, pre-step substitution or reduced contact sampling. Independent
  native consumers outside this callback retain their existing reads.
- In-memory conversion microbenchmark (NOT native latency):3000 callbacks,
  old6 conversions158.273us/callback versus shared33.535us. This estimates only
  a small Python conversion cost, not a real-time simulator improvement.
  Native460 repeats the full456 neutral retained-cut recipe to check unchanged
  behavior. Its final report and physical-record comparison are pending below.
- Regression v53:5199 passed in258.09s
  (`data/sim_physics/physics_regression_20260914_v53.log`). Subsequent direct-cut
  neutral-search changes:75 focused tests passed. New tests cover invalid frame
  families/budgets, control-reserve exhaustion, pose ownership/stale steps,
  exact batched conversion equivalence, wrong-task imports and parked-left
  semantics. Remaining acceptance: full greenhouse neutral grasp+cut and direct
  cut, repeatability, and measured responsive performance at unchanged fidelity.
- **Native460 PASS, all12 gates**, after the new post-fetch sharing. Same
  isolated seed101/SubStem_41 neutral-start case as456: cut24.829167s,
  reobserved retained separation28.8625s, forward traversal40.35625s, unloaded
  reverse74.008333s, final native withdrawal85s. Same5700 native edge contacts,
  maximum1.752025mm slip and zero released-debris contact. Complete archive:
  40800 records,812958161 bytes. Zero-tolerance comparison against456 found
  **40799/40800 whole records identical**. The sole changed value is the final
  `withdrawal.native_static.epoch.simulation_time_s` (Kit query-clock metadata);
  every recorded physical value is identical. Evidence logs:
  `data/sim_physics/native456_native460_record_comparison_v2.log` and
  `data/sim_physics/native456_native460_last_record_diff.log`.
- **No overall latency gain claimed.** Native460 RTF0.175748,483.647 tick-wall
  seconds for85 simulated seconds;456 was0.178497/476.199s. Both overlapped
  regression work. Post-step p50 was3.664ms versus3.771ms, but native/render
  timings varied. The measured Python microbenchmark gain is not a responsive
  C pass. No solver frequency/iterations, graphics or contact limits changed.
- Final regression including direct-cut station mode: **5202 passed in251.17s**,
  `data/sim_physics/physics_regression_20260914_v54.log`.
- Native461 tries prior greenhouse-success source seed19/SubStem_41 at original
  end-row slot0 with the NOW upright neutral posture. It correctly stops BEFORE
  motion: neutral right forearm overlaps two target leaves. This is not the
  previous pre-positioned, nonzero-torso378 recipe, and378's success cannot be
  claimed for this new start. All plants remain at original planting slots;
  detailed target/backdrop assignment is swapped without removing plants.
- Native462 uses the new parked-left neutral search, finding a proposal at
  [0.3,-6.2,135 degrees] after3 candidates (27 available), in4.874s. Right
  extension0.925753; inter-arm clearance0.248386m; final native controls pass,
  no budget exhaustion and ZERO physics steps. This is an endpoint/station
  proposal ONLY. Native463 rejected duplicate explicit-ready/proposal CLI
  arguments before starting Isaac. Native464 removes that redundant argument
  and performs fresh startup/path/execution validation from462; outcome pending.

### 2026-09-14 evening: neutral approach retiming and detached-branch blocker

- Native464 passed fresh full-greenhouse neutral startup and right-path screening,
  but stopped at5.85625s: wrist tracking12.05mm exceeded the unchanged12mm
  limit, without robot contact. The four-second index ramp did not account for
  the long redundant-joint path. This is not a successful action.
- Added opt-in `--rate-limited-approach-trial` and `approach_timing.py`:
  preserves every original piecewise-linear joint-path edge; weights duration
  by joint, sampled wrist-translation and orientation travel with a smooth
  time ramp. Engineering caps:30deg/s joints/orientation,0.1m/s sampled wrist;
  duration bounded4..45s. These are not acceleration or hardware certificates.
  Only precontact time is extended; final-step withdrawal accounting follows
  that extension. Cut feed, tracking/contact thresholds and both35s postrelease
  deadlines are unchanged. Default remains off.
- Focused retiming tests94 passed; complete regression5215 passed in231.82s,
  `data/sim_physics/physics_regression_20260914_v55.log` (before velocity-test
  CLI addition below).
- Native465, same complete greenhouse seed19/SubStem_41 end-row case and
  proposal462: retimed approach15.916227s; force-qualified cut30.679167s.
  At33.73125s the detached chain became numerically unstable on the floor:
  max body speed4696.704m/s, versus about0.01m/s immediately beforehand.
  The dynamic-workspace guard stopped execution. Full through-stroke and
  withdrawal did NOT pass. This is not merely allowable branch landing or
  insufficient window size. No limits/window were widened. Evidence:
  `data/sim_physics/bimanual_downward_20260914_native465/report.json` and its
  full-rate `bimanual_trajectory.jsonl.gz`. Native floor contacts are recorded;
  the precise numerical cause is still under investigation.
- Added explicit guarded `--velocity-solver-trial` launcher option selecting
  already-supported PGS128/8 instead of default128/0. Complete cut/clearance/
  egress recipe required; incompatible with position-iteration comparison.
  No mass, material stiffness, geometry, visuals or contact limits changed.
  Native466 is the controlled comparison; outcome pending, not a qualified fix.
- Acceptance remains INCOMPLETE: isolated neutral direct and retained bimanual
  examples pass, but current greenhouse neutral end-to-end stability and
  responsive performance do not. No VLM collection/training has resumed.
- Native466 (128/8) cut at34.060417s and survived the earlier branch-floor
  instability, but hit the original35s forward-stroke deadline: knife side-face
  load approached0.40N with2.157mm of material-section clearance still missing.
  It is NOT a successful cut sequence and is not promoted to the default.
- Native465 raw floor contact: Segment_001 separation jumped from+0.868777mm
  to-54.607864m in one step; reported witness moved to[21.346,-5.477,17.157]m
  and normal impulse32.732825Ns. Source-floor box is140.200x63.140x0.100m.
  This supports investigating narrow-phase scale conditioning, not hiding the
  failure by expanding the dynamic window or deleting the fallen branch.
- Added opt-in `--local-floor-tiles-trial`: one-metre cells within a six-metre
  square around an explicit fixed station, retaining ALL exterior floor solid.
  Actual package:80 boxes instead of3; source visual meshes/material/height,
  holes, steps and union volume preserved. Positive axis-aligned metric
  transforms only;512-cell cap; unchanged defaults and per-step guards.
  Geometry equivalence is not proof of identical native contact behavior.
  Native467 changes ONLY this floor representation relative to465; pending.
- Focused retiming/solver-option/floor regression:30 passed in1.88s,
  `data/sim_physics/floor_contacts_20260914_v1.log`. Includes original source
  immutability, exact volume, nonoverlap, local cell dimensions, invalid input
  rejection and pre-write CLI validation.
- Further contact comparison: successful historical native378 used a2.3mm
  distal blade aim, whereas the current neutral greenhouse comparisons used
  1.5mm. That placement is already within the existing explicit admissible
  interval and requires fresh path/force/full-section checks; it is not a
  certificate for the new neutral station. Next comparison follows467.
- 22:30KST deadline was NOT met. A/B/C are not declared frozen; VLM stays paused.
- Native467 floor-only comparison: no explosion across31526 recorded steps;
  force-qualified release30.679167s unchanged from465, but the unchanged35s
  traversal deadline stopped the knife at0.399677N side-face load and
  -2.297468mm edge clearance. Recorded maximum plant speed6.070852m/s,
  minimum floor separation-6.097855mm, maximum floor-row impulse0.0197736Ns.
  These passive trace summaries are not native contact-completeness or tissue
  certificates. This eliminates the observed explosion in this run, not all
  numerical instability or floor penetration.
- **Native468 PASS: all10 neutral-start greenhouse direct-cut gates.** Same
  end-row seed19/SubStem_41, fixed station[0.3,-6.2,135deg], unchanged128/0
  solver, exact local floor partition; use the PRE-EXISTING2.3mm distal aim
  and its source-blade/stump half-space check, rather than1.5mm from the other
  isolated target. Release30.664583s; full-section stroke37.372917s;
  unloaded reverse43.041667s; final native withdrawal96.9s. Sharp-edge margin
  0.524664mm; full ellipse axial extent2.897626mm stays inside3mm.3462 native
  edge contacts, no released-debris robot contact, source assets unchanged.
  Evidence: `data/sim_physics/bimanual_downward_20260914_native468/report.json`
  and event-bound original PNGs. This is ONE case/run, not general reliability,
  deposition, reset/replay, calibrated tissue fracture or synchronized data.
- Native468 timing:96.9 simulated seconds /629.806 tick-wall seconds,
  RTF0.153857. It overlapped regression work. Responsiveness C remains OPEN;
  no real-time or speedup claim. Current greenhouse scope retains419 original
  target components and original infrastructure; nearby surrounding plants
  are static contacts, selected petiole dynamic; existing three-gutter context.
- Final regression v56: **5222 passed in254.15s**,
  `data/sim_physics/physics_regression_20260914_v56.log`.
- Native469:13 source19 bimanual station proposals, no clear result; final
  native controls passed, zero physics steps and no budget exhaustion. Nine
  neutral-scene rejections, one pregrasp-self, two grasp/entry IK and one
  interarm/extension rejection. This does not prove global infeasibility.
- Native470 was CLI-only rejection (missing explicit source for end-row swap).
  Native471 corrected this and passed initial native startup/neutral-path
  screening for seed101 at original positive end-row slot23, preserving the
  isolated relative robot/plant transform (base[0.52,6.25,-147.104778deg]).
  It was deliberately stopped, not qualified, when the user narrowed the task
  to showing ONE working example before VLM. No completed bimanual greenhouse
  result is claimed. Only its owned headless Kit process was stopped.
- User's revised immediate handoff: show the qualified greenhouse direct cut;
  broader A/B/C freeze is deferred, not declared achieved. Native472 repeats
  native468 with `--watch --watch-auto-run`; visible Isaac6.0.1 window verified
  responding and `CUT_WATCH_READY` logged. Live repeat outcome pending.
  Isolated neutral bimanual454/455/456/460 are four repeats of one case.
  VLM work may resume after the requested demonstration; no collection,
  dataset mutation or training was started during this continuation.

### 2026-09-15: visible cut-point dataset rebuild (overnight work)

- Checkpoint update: full data suite660 passed +47 subtests (45.20s), log
  `data/sim_data/clear_regression_20260915_v2.log`;22 focused new-source/capture
  checks passed afterward. Pilot_v1 exited1 before any frames because source
  scene overrode explicit RaytracedLighting. No renderer check was bypassed.
  Corrected pilot_v2 uses source-selected renderer; one seed101 job is running.
  No native new-capture success, trainable new dataset or ZIP claimed yet.
- Fresh-capture export can explicitly consume an integrity-validated engineering
  pool from new bound clear-capture plans; this does NOT waive the new clear
  release's coverage or visual review gates. Existing training releases remain
  untouched and no old source may masquerade as a fresh capture pool.

- User requested a simpler visible-target VLM task, H200 instructions and a ZIP,
  then explicitly chose NEW clearer native captures instead of packaging the
  old geometrically-easy images. No training or physical robot commands started.
- Implemented a separate `clear_cutpoint_v1` derivative, explicit legacy/new
  validator dispatch, query-only384px crop magnified to768px, deterministic
 12-view/target cap, strict visible-interval/width/exposure proxies, review GUI,
  immutable finalization and status-preserving ZIP transfer. Original RGB and
  native optical-Z bytes, canonical10mm labels and16/4/4 family splits remain.
- Qwen trainer supports explicit visible-only smoke/overfit and crop comparison;
  saved input mode is reused for inference. Clear scorer counts invalid/missing/
  abstaining/truncated predictions as failures, reports interval+target hits,
  target/family macro averages and latency/token cost. Test requires explicit
  access. Instructions: `examples/greenhouse_sim/sim_data/CLEAR_CUTPOINT.md`.
- Complete old-release scan: only47 strict candidates (train28/9targets/7families;
  validation15/3/2; test4/4/3). Preserved as
  `data/sim_data/training_exports/clear_cutpoint_20260914_v1`, explicitly DRAFT
  and not trainable. No ZIP of this diagnostic set is offered as training data.
  Earlier101-image train-only preliminary sample had0 passes; full scan is the
  authoritative count. No existing reviews, splits or old releases were edited.
- Added default-OFF `--clear-capture`: closer original-aisle base proposals,
  bounded real torso/head pose sampling, predicted8px petiole/12px interval,
  uniform dome fill6000 (from1200), unchanged sun1500. No per-target spotlights,
  RGB postprocessing, sensor changes or hidden scene geometry. Existing scene
  overlap rejection remains. Existing Windows memory preflight runs per job.
  This preset is NOT yet natively qualified; pilot plan creation is running.
- Tests before capture changes:652 passed +47 subtests;17 focused clear-release/
  transfer tests subsequently passed. Capture/plan focused regression69 passed.
  Actual Qwen processor/training performance on the new crop remains unmeasured.
- All selected held-out images currently require human review in the release
  policy; assistant review is explicitly attributed. User has authorized overnight
  collection/review/annotation and packaging, not fabricated human acceptance.

### 2026-09-14: faster-cut trial checkpoint and VLM branch handoff

- Added default-OFF `--faster-cut-trial` through the ground-truth launcher,
  benchmark and force-feedback blade controller. Commanded free-space feed
  increases from2 to10mm/s, near-contact cap from0.3 to1mm/s, qualified loaded
  advance from0.15 to0.75mm/s, and loading feedback gain from0.00125 to0.005.
  Requires the complete480Hz neutral-start seam/through-stroke/egress recipe,
  native startup/static clearance and existing support-aware2mm/s postrelease
  feed. Original defaults, force/geometry limits, dwell, backoff, fresh-step
  checks and deadlines are unchanged. Command-rate increases are NOT measured
  motion speedups or native qualification.
- Focused regression:84 passed in0.67s, rerun before this checkpoint; includes
  `faster_cut_test.py`, `blade_feed_test.py`, `through_stroke_test.py` and
  `support_aware_feed_test.py`. Earlier matching run is logged at
  `data/sim_physics/faster_cut_20260914_v1.log`. The prior5222-test full-suite
  result predates this faster-feed change; no new full-suite pass is claimed.
- Native472 (regular-speed visible greenhouse repeat) stopped at53.964583s
  with `Stopped; reset required`, after release, full traversal and unloaded
  reverse, before the final withdrawal qualification. Native473 (faster-feed
  visible comparison) stopped at16.860417s during approach, before blade
  contact, with the same recorded error. These interrupted runs do not qualify
  the faster profile; the cause of the stop is not established. Evidence:
  `data/sim_physics/bimanual_downward_20260914_native472/report.json` and
  `data/sim_physics/bimanual_downward_20260914_native473/report.json`.
- Existing qualified evidence remains native468 for ONE neutral greenhouse
  direct-cut case and native454/455/456/460 for four isolated bimanual repeats
  of ONE case. Greenhouse bimanual, faster native cuts and real-time performance
  remain unqualified. This is not an A/B/C freeze or a tissue-fracture claim.
- User requested checkpointing `koh-dev/sim-vlm`, syncing its history into
  `koh-dev/sim-data` and switching there for VLM work. At handoff inspection no
  repository Isaac Kit process was running. No new simulation, collection or
  training job is launched for this branch operation; datasets, splits and
  review decisions are not changed. Simulator limitations remain open while
  the user's requested VLM work resumes.

### 2026-09-15 overnight fresh VLM collection checkpoint

- User requested new, clearer native RGB data instead of packaging 47 old
  candidates. Active work is collection, source-derived annotation and visual
  review, then a verified Qwen3-VL-8B transfer ZIP; no model weights or training
  are running locally. See the new checkpoint in `vlm_train_data.md` and
  `examples/greenhouse_sim/sim_data/CLEAR_CUTPOINT.md`.
- 01:52 KST measured checkpoint: eight completed audited jobs, 208 native frames,
  127 strict clear images / 27 targets / seven training families; all127 original
  RGBs individually assistant-reviewed. Final coverage and held-out capture are
  still incomplete. No archive or research-result claim is made at this point.
- Native capture runs serially with the existing memory/floor/geometry checks;
  native depth remains Isaac optical-axis Z. Source assets, reserved splits and
  old decisions are unchanged. Opposite-aisle static capture is opt-in and has a
  bounded pilot queued behind the current campaign, not yet native-qualified.
- Source through `d291017` is committed. Latest data-code regression passed671
  tests plus47 subtests. Generated captures/reviews remain under ignored `data/`.
  Simulator contact/cutting limitations documented above remain open; this data
  task is not a claim that greenhouse bimanual physics has been qualified.

- 02:59 KST additional data checkpoint:300 fresh native frames,194 strict
  candidates,192 assistant accepts and two holds; still incomplete held-out
  capture and coverage. Details/paths in `vlm_train_data.md`. Added bounded
  CPU/native-matched viewpoint diagnosis, opt-in real torso-lean proposals and
  immutable negative-review exclusions. None bypasses native quality checks or
  claims robot dynamics/physical cutting validation. Full data regression before
  exclusion changes:682 tests +47 subtests; focused exclusion/release27 passed.

- 03:21 KST VLM-data checkpoint:all227 first-pass training candidates reviewed
  (224 accepts/3 holds), plus5 validation candidates (4 accepts/1 hold). Still
  insufficient final coverage. Target-facing static root orientation is opt-in
  and native qualification pending. Full data regression691 +47 subtests passed.
  Replaced only3 still-waiting assistant pilot queues; active native collection
  was untouched. See `vlm_train_data.md` for exact provenance and limitations.

- Subsequent collection checkpoint:472 fresh native frames /300 strict
  candidates through seed31;269 visually reviewed (265 accepts/4 holds), with
  test-family review ongoing. No final coverage/ZIP claim. Added explicit
  campaign job subsets to avoid duplicate pilot capture without changing
  source plans/splits. Full data tests696 +47 subtests passed in47.13s.
