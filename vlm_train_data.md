# VLM Training Data Plan: Greenhouse Deleafing

Date: 2026-09-06

Branch: `koh-dev/sim-data`

## Current priority: newly captured clear localization (2026-09-15)

User chose fresh clearer robot-head captures over reusing the old easy-image
pilot. The strict scan retained only47 images (28train/15validation/4test),
insufficient coverage and unreviewed; it is NOT the new training release.
The new `clear_cutpoint_v1` implementation and H200 workflow are documented in
`examples/greenhouse_sim/sim_data/CLEAR_CUTPOINT.md`. Original native Isaac
optical-Z, unchanged head optics, frozen source-family splits and10mm nominal /
10-20mm evaluation interval remain. Crop input is derived only from the query.
Overnight scope: capture clearer observations, audit/review/annotate them with
honest reviewer attribution, then package a status-accurate ZIP. No training,
physical robot actions, fabricated human reviews or relabeling hidden cuts.
Native closer-view lighting pilot is qualified: seed103 pilot-v3 exited cleanly,
independent native audit passed15 captures, and8 passed the stricter clear-task
screen. All8 have image-bound assistant visual assessments, not human approval.
Serial24-family collection is running under
`data/sim_data/collection_campaigns/clear_capture_20260915_overnight_v1`.
The first family produced7 independently audited captures; retained clear-task
counts and final coverage are not known yet. No final training ZIP exists yet.

The explicit `assistant_reviewed_experiment_v1` review mode permits overnight
assistant review of held-out label quality, while recording that independent
human validation was NOT performed. It does not change splits, optical size,
legibility thresholds, coverage gates or failed/held review decisions. Default
release review still requires human held-out review. All held-out records and
at least two training views per target require explicit hash-bound assessment.

## Current increment: reviewable active perception (2026-09-10)

Goal remains a robot-view VLM that can localize an observable petiole cut,
recognize insufficient evidence/invalid targets, and eventually request a
validated reveal action before bimanual grasp–cut–deposit execution.

The latest static pilot has **28 task examples**: eight visible cuts, eight
occluded cuts, eight invalid candidates and four leaf-covered uncertain regions.
These reuse native head-camera frames (848x408 RGB plus original Isaac depth),
not fresh captures or executed action sequences. Unknown regions intentionally
do not assert hidden target presence or absence. The older 24-example pilot is
retained. Current path: `data/sim_data/dataset_reviews/active_perception_20260910_v3/`.

The v4 reviewer at <http://127.0.0.1:8882> now supports native RGB/mask/depth tabs,
optional query/region markers, explicit per-label human review, separate
assistant records and immutable evidence bindings. Existing v3 reviews remain
separate at port 8881. `run_active_perception_review.cmd` reopens this new GUI.
Saved Isaac optical-Z is coloured only for display; no replacement depth is
estimated. SVG markers are explicitly hidden on depth images with legends.

A guarded exporter can package only explicitly human-accepted, unheld v4 rows
as a **reviewed perception subset**. It preserves family splits, copies original
RGB/depth bytes and keeps privileged XYZ/native metadata out of model messages.
It does not certify the full release, train a model, or create motion labels.
No real v4 training package or dynamic episode has been released here.

Next: complete new-label QA; capture genuinely assessed no-target regions;
expand and balance native task classes; validate and export the static release.
Then integrate actual native timestamp/frame synchronization and collect
collision-checked viewpoint/left-arm reveal successes, failures and recovery.
See [the v4 guide](examples/greenhouse_sim/sim_data/ACTIVE_PERCEPTION.md).

The checkpoint notes below are retained chronologically; they are not a claim
that every earlier pending item is still missing after this increment.

User direction (2026-09-10): broadly agrees with the assistant's assessment and
will follow its recommendations. Use it as the working review baseline while
preserving explicit individual decisions and unresolved flags. This endorsement
does not assert per-image human inspection, clear source holds, approve the entire
training release, or transfer approval onto the new v4 task. Eighteen individual
human decisions were present at this follow-up; earlier counts below describe
the prior implementation checkpoint. The statement is logged separately in
`data/sim_data/dataset_audits/independent_v3_20260909/endorsements/`.

Latest implementation (2026-09-10): the review GUI now loads separate assistant
suggestions and supports append-only human follow-ups to legacy assistant
decisions. Fourteen existing human decisions were preserved; 79 advisory findings
(72 support / 7 holds) are available. See the updated
[GUI guide](examples/greenhouse_sim/sim_data/TRAINING_REVIEW_GUI.md).

An adjacent task-v4 observability contract and bounded static pilot are implemented:
eight matched visible/occluded target pairs plus eight native-identity invalid-organ
queries, all pending new-task human review. Task-v3 labels/hash are unchanged.
No-target-in-region is distinct from hidden/unknown/invalid/execution-blocked.
The pilot contains **zero dynamic episodes**, no copied approvals and no synthesized
depth. Full no-target collection, moving-camera synchronization, physical reveal
trajectories and v4 review/export remain next steps. See
[active-perception scope and commands](examples/greenhouse_sim/sim_data/ACTIVE_PERCEPTION.md).

Current checkpoint (2026-09-09, post-anatomy audit): the 63-image wave-2 audit is
complete. Task v3 now gates query legibility and suppresses hidden-cut overlays;
54/63 sources remain candidates and 9 are excluded. Fresh v3 visual review and
larger native-camera collection are still in progress. Earlier v2 counts and
approvals below are historical, not v3 release approval. See the final audit
entry below and `examples/greenhouse_sim/sim_data/TRAINING_DATASET.md`.

Status: Phase 1 audit/review tools and prototype cut-region drafts are implemented. The bounded full-greenhouse robot-head RGB-D pilot now has opt-in viewpoint screening and renderer-identity organ masks for the two detailed plants. Synchronization remains static-only, not a dynamic recorder or approved training dataset. The user-agreed engineering rule is 10 mm nominal / 10-20 mm along the petiole centreline; per-target approval and horticultural validation remain pending. Complete-scene organ annotation, approved cut/grasp regions, dynamic synchronization, scaled collection and training are still pending. See `examples/greenhouse_sim/sim_data/PHASE1.md` for the refined-pilot command, visibility scope and provisional quality gates.

## 1. Goal and recommended approach

Latest Phase 1 increment (2026-09-08): the human-reviewed, hash-bound snapshot is
[robot_head_prototype_v3/review.md](data/sim_data/datasets/robot_head_prototype_v3/review.md).
The user confirmed original B05 samples 0004/0005 and held B03 samples 0001/0003;
the index contains two confirmed prototype labels and seven diagnostic holds.
The confirmations are two views of one B05 target, not two independent targets.
All nine original camera poses were independently reconstructed from RB-Y1 A
v1.2 joint/base transforms and the fixed head-camera mount; native projection
and uncropped 848x408 optics agree. These are actual simulated robot snapshots,
not free/cinematic cameras or live lab robot poses.

A focused base-XY/yaw and real-head recapture now supplies six additional audited
views without moving plants, changing the mount/resolution or tuning lighting:
[robot_head_focused_v3/review.md](data/sim_data/datasets/robot_head_focused_v3/review.md).
The assistant inspected RGB/overlay/native mask/native depth for all six and
recommended three clearer B03 views, now explicitly confirmed by the user.
B06 remains held: one numerically visible but visually ambiguous cut, and two
cuts obscured by identified leaves. Total human-confirmed evidence is five
images across two target geometries, not five independent targets. All training
eligibility flags remain false. The focused review GUI is on port 8878, independently of the original on
8877. The capture artifacts passed audit, but Kit shutdown returned nonzero;
clean shutdown and dynamic synchronization are not established. See PHASE1.md
for commands, numerical evidence, scope limits and remaining approval boundaries.

The next increment adds a deterministic native multi-plant scheduler. It audits
all 24 source families (870 geometry candidates), reserves 16/4/4 target-family
train/validation/test groups and selects seed11/seed17 native SubStem_41/42 for
the first bounded batch. No branches are added or moved, no leaves are hidden,
and actual mounted head views retain native RGB/Z/identity QA. Shared greenhouse
and backdrop context is not held out. Job subprocesses run serially and must
pass both process-exit checks and independent data audit; failed attempts remain
diagnostic evidence. See PHASE1.md for the schedule/run commands and dev.md for
measured batch results. Native targets receive no copied human confirmations,
difficulty assignments, physical-cut labels or training approval.

Measured native result (2026-09-09 KST): both workers exited 0 and eight views
of four petioles on two new plants passed independent audit. Assistant review
recommended five views and held three, despite seven numerical clear-view passes.
The user then confirmed all five recommendations. Across original/focused and
native pilots, there are ten confirmed images of five targets from four source
families; this is still prototype label agreement, not training readiness.
[Native evidence report](data/sim_data/collection_batches/native_20260908_v2/visual_review.md).
The combined review page on port 8879 preserves separate source/audit identities.

Given greenhouse observations and a deleafing instruction, identify an appropriate petiole, determine whether it is observable and accessible, and generate coordinated guidance for the left gripper and right knife. The system must be able to inspect, reveal, or reject a target when necessary.

The eventual interface should support a specific metric cutting point `(x, y, z)`, explanations of environmental occlusion and hazards, and left/right motion guidance for a low-level bimanual controller. The complete action is left-hand grasp, right-knife cut, left-hand retention of the detached branch, and release into a designated floor area.

Recommended architecture:

- Separate the visual planning model from the 3D-aware execution layer.
- Capture RGB-D from the beginning, even if the first VLM receives RGB only.
- Train image-plane grounding and sparse paths first; obtain validated metric coordinates through calibration, depth, and geometry.
- Build reliable static perception labels before scaling image generation.
- Generate manipulation-path supervision from validated episodes, not merely plausible drawings.
- Treat Cosmos augmentation as an optional appearance-domain transformation that must preserve label correspondence.

This document records the planning discussion. Numerical dataset sizes, difficulty thresholds, and split ratios below are initial engineering proposals, not measured sufficiency or achieved performance.

## 2. Current implementation and asset baseline

The supplied greenhouse on `koh-dev/sim-data` is a rendering foundation, not yet a synthetic manipulation-data generator.

Available:

- Greenhouse geometry, camera presets, lighting controls, and efficient background vegetation.
- 24 component-level plant assets with parent relationships, attachment points, axes, capsule geometry, and physical parameters.
- A current preview containing two detailed plants and 142 background instances.
- Existing VLM infrastructure for image submission, structured cut-point responses, request/response logging, and overlays.
- Older simulator implementations of grasping, cutting, robot cameras, and bimanual task sequencing that can be adapted.

Important limitations and findings:

- The new package has a static RB-Y1 Model A v1.2 preview with three mounted 848x408 D405 RGB views and stock grippers. The new `sim_data.capture_pilot` captures clean full-greenhouse head RGB, metric depth, validity and calibration for B03/B05/B06, with separate provisional-label review overlays. Native frame IDs/times are unavailable in this paused rendering mode; the pilot validates one writer payload against a frozen scene and camera/content freshness. It is not a moving-demo recorder. Validated grasp/cut behavior and semantic dataset annotation are not yet integrated. See [Phase 1 notes](examples/greenhouse_sim/sim_data/PHASE1.md).
- The inspected manifests contain 38-42 substems marked `deleafed` per plant. A `sub_stem` class alone cannot identify a valid cutting target.
- The plant generator is not included. Camera and lighting randomization do not turn 24 source plants into thousands of independent plant geometries.
- The older `_TeleopCameraRecorder` can reuse a previous valid RGB frame after a capture error. The dataset exporter must instead reject or retry incomplete captures so RGB, depth, and labels remain synchronized. See [interactive_greenhouse.py](examples/greenhouse_sim/interactive_greenhouse.py).
- The existing cut/grasp implementation belongs to the older composed scene; its presence in the repository does not establish that it works with the new package.

The first engineering milestone is trustworthy geometry and labels, not large-scale rendering or training.

## 3. HAMSTER inspiration and system boundary

HAMSTER separates a high-level VLM, which receives RGB and an instruction and predicts sparse image-plane paths with gripper events, from a low-level, 3D-aware policy that follows that guidance. The paper uses VILA-1.5-13B for the high-level model and evaluates RVT2 and 3D Diffuser Actor as low-level policies. Its path supervision includes trajectories projected from simulator or robot geometry. See the [HAMSTER paper](https://arxiv.org/html/2502.05485v3) and the [user-supplied PDF](https://hamster-robo.github.io/paper.pdf).

Our proposed extensions are two coordinated end effectors, a knife with a specific cutting edge and orientation, thin deformable targets, occlusion removal, re-observation after foliage movement, and explicit uncertainty or rejection outcomes.

This is a HAMSTER-inspired extension, not a direct reproduction of demonstrated greenhouse or bimanual capabilities. The [HAMSTER beta repository](https://github.com/liyi14/HAMSTER_beta) is principally a model-serving/demo starting point, not a turnkey data generator or greenhouse control stack.

```text
RGB observation + task + current phase
                 |
                 v
VLM: target, visibility, hazards, next phase, left/right 2D paths
                 |
                 v
Depth/calibration + 3D geometry + reachability/collision checks
                 |
                 v
Low-level bimanual controller
                 |
                 v
Observe the changed plant -> update the plan
```

Model inference must remain outside the simulator's physics loop. The VLM is an advisory planner; its wording or confidence must never override execution validation.

## 4. Coordinate and trajectory representations

The requested "2D trajectory map with x, y, z" needs separate, explicit representations:

| Representation | Meaning | Purpose |
|---|---|---|
| `(u, v)` | Position in an image | VLM grounding and displayed paths |
| `(u, v, z)` | Image position plus camera-frame depth | Optional 2.5D guidance |
| `(x, y, z)` plus orientation | Metric tool pose in a named 3D frame | Robot execution |

A path drawn over an image is 2D. Adding depth does not make it a complete robot trajectory. Tool orientation, approach clearance, stroke direction, speed, and synchronization are still required.

### 4.1 First model interface

The first VLM should predict:

- Target petiole region.
- Cut point in image coordinates when sufficiently observable.
- Occlusion and hazard classifications.
- Next manipulation phase.
- Separate, synchronized left-gripper and right-knife image-plane paths.
- Gripper events and re-observation requirements.

The execution layer should resolve these into validated metric coordinates.

For a rectified camera with optical-axis depth `z` and intrinsics `K`:

```text
p_camera = z * inverse(K) * [u, v, 1]^T
```

Camera extrinsics then transform the point into the robot base frame. Store units, coordinate conventions, reference frame, and timestamp explicitly.

### 4.2 Important depth and geometry constraints

1. Depth at an occluded cutting point belongs to the foreground leaf, not the hidden petiole.
2. A free-space waypoint cannot generally obtain its depth by sampling the background surface beneath its pixel.
3. A visible stem surface point is not necessarily the stem centerline or desired cutting-plane origin.
4. Camera-range depth and optical-axis depth are different quantities. Replicator exposes separate annotators; confirm the selected convention and invalid-value behavior with tests. See [NVIDIA depth documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_replicator/1.13.30/source/extensions/omni.replicator.core/docs/API.html).

These issues require target association, local geometry estimation, and 3D planning rather than blind back-projection of every predicted pixel.

Initial recommendation: RGB to the VLM; metric depth to the geometry/controller layer. Direct VLM prediction of metric XYZ is a later experimental comparison, not the initial authority for execution coordinates.

## 5. Ground-truth definition

### 5.1 Establish eligible deleafing targets

Add a task-specific annotation layer above the package's segmentation taxonomy. For each potential target, determine:

- Which petiole connects the target compound leaf to the main stem.
- Attachment geometry and local centerline.
- Which organs detach after cutting.
- Whether the branch is already deleafed.
- Whether its descendants contain structures that must be protected.
- Whether it satisfies the chosen horticultural eligibility rule.
- Where the left gripper can hold it without interfering with the blade.

The package's `candidate_node` class is useful but does not establish agronomic eligibility or physical accessibility.

Review representative examples with a tomato-cultivation expert before scaling. The permitted cut location and residual stub length must be explicit, versioned task rules rather than arbitrary offsets in a script.

### 5.2 Label valid regions as well as canonical points

Store:

- A canonical cut point for training.
- The admissible cut region or centerline interval.
- Allowed blade orientations and stroke directions.
- Main-stem and protected-organ exclusion volumes.
- Permitted grasp regions.
- Expected detached subtree.

Scoring should accept other valid cuts inside the admissible region rather than treating one selected coordinate as the only correct answer. XYZ alone is insufficient to specify knife execution.

### 5.3 Separate hidden truth from observable evidence

For an occluded target, the simulator knows the hidden location, but the model may not have enough evidence to determine it.

- Preserve hidden geometry in supervision/evaluator labels.
- Record whether localization is supported by the actual supplied observation.
- Train `inspect`, `reveal`, or `uncertain` responses when appropriate.
- Do not force precise coordinate supervision for fully unobservable targets in the executable-planning task.

An optional amodal-localization benchmark may assess hidden-point estimation separately. Such estimates must not be presented as verified cutting commands.

## 6. Two connected datasets

### 6.1 Dataset A: perception and visual reasoning

Each example contains an observation and instruction, with labels for:

- Target identification.
- Cut region and point.
- Attachment visibility.
- Occluding organ regions.
- Protected structures.
- Whether further observation or manipulation is required.

This dataset can be developed before physical manipulation is integrated into the new package.

### 6.2 Dataset B: interaction and bimanual planning

Each example is an episode containing:

- Synchronized observations over time.
- Robot joint and tool states.
- Target and occluder state changes.
- Left/right trajectories and gripper events.
- Contact, grasp, cut, retention, and release evidence.
- Success or failure outcomes.

Distinguish geometrically proposed paths from physically replayed and validated trajectories. A visually plausible arrow around a leaf is not evidence that the robot can execute the motion.

### 6.3 Common capture and provenance contract

Capture all modalities from the same simulator state:

- Lossless RGB.
- Metric depth and validity mask.
- Semantic and instance IDs.
- Camera intrinsics, extrinsics, resolution, and crop/resize transforms.
- Plant/component identities and transforms.
- Task labels and label provenance.
- Scene seed, asset hashes, simulator version, and configuration version.
- Timestamps and simulation/frame identifiers.
- Robot state and tool calibration when present.

Maintain explicitly separated exports:

- **Model inputs:** only permitted observations, the instruction, and declared state.
- **Supervision/evaluation:** hidden geometry, masks, target identities, and outcomes.

Training answers use supervision, but those answers must not leak into evaluation prompts, filenames, or input overlays. Input images must not contain target highlights, selected-prim outlines, or debugging markers.

### 6.4 Segmentation and depth storage

- Use separate perception masks and Cosmos control maps.
- Background plants are joined assets and cannot honestly be labeled leaf-by-leaf.
- Preserve the supplied class IDs and define an explicit unlabelled value. File ID `0` already means fruit in the package taxonomy; do not silently reuse it for background.
- Store exact supervision in lossless arrays or ID masks.
- Keep raw metric depth separate from colorized or normalized depth controls.
- Use videos for review and compatible augmentation inputs, not as exact metric or class-ID labels.

## 7. Difficulty curriculum and appearance domains

Preserve the user-facing categories easy, medium, hard, and real-hard, while recording geometric/observational difficulty independently from appearance domain.

### 7.1 Initial difficulty definitions

The following thresholds are proposed starting points to calibrate through the pilot, not established standards.

| Category | Proposed characterization | Expected response |
|---|---|---|
| Easy | Attachment and cut neighborhood clearly visible; approximately >=90% local visibility; sufficient resolution | Localize and propose grasp/cut guidance |
| Medium | Partial occlusion, roughly 40-90% local visibility; ambiguity or obstruction | Localize if justified; otherwise inspect or reveal |
| Hard | Heavy occlusion, roughly <40% visibility, hidden attachment, or overlapping structures preventing confident association | Reveal, change view, or abstain |
| Real-hard | Cosmos-transformed samples drawn from easy, medium, and hard | Same task, with appearance shift evaluated separately |

Local visibility concerns the attachment and admissible cut region, not the visible fraction of the whole leaf.

Estimate it using paired renders of the target region:

- Under normal scene occlusion.
- Without competing occluders.

The visible-area ratio is one measurement. Also record attachment-point visibility, projected petiole diameter, occluder identities/count, depth ambiguity, lighting/blur, reachability, and collision clearance. Out-of-view targets should be distinguished from in-view occlusion.

A distant, unoccluded petiole may still be impossible to localize accurately. A visible target may be unreachable. Keep those factors explicit rather than hiding them inside an occlusion score.

### 7.2 Real-hard stratification

Retain metadata such as:

```text
geometry_difficulty = medium
appearance_domain  = cosmos
```

For the initial release, use seeded random sampling with an approximately balanced mixture of easy, medium, and hard source samples. Record the mixture for reproducibility and report each underlying category separately.

Cosmos-generated imagery is appearance-augmented synthetic data, not real-world data. Maintain a separate, untouched real-camera test set for actual sim-to-real transfer evaluation.

### 7.3 Negative examples

Use an initial approximately 20% no-cut/inspect mixture only as a pilot sampling
hypothesis, not an optimal class ratio. Preserve distinct labels rather than
putting all non-cut outcomes into a single negative class:

- Invalid candidate: already-cut stub, main stem, fruit or protected fruit truss.
- No eligible target in an explicitly bounded, fully observed region.
- Unknown eligibility / ambiguous target selection: inspect, not assert absence.
- Occluded attachment: valid hidden geometry is evaluator-only; reveal or inspect.
- Valid anatomy but blocked grasp/blade approach: execution constraint, not invalid anatomy.
- Target outside the supplied observation: obtain another view, not a global no-target claim.

A model that always returns a cutting point must score poorly.

## 8. Coordinated bimanual trajectory generation

### 8.1 Conditional task sequence

1. Inspect and select the target.
2. Resolve visibility if necessary.
3. Establish a suitable left-hand grasp.
4. Pull or position the target within validated limits.
5. Re-observe and update the cutting pose.
6. Execute the right-knife stroke while the left hand retains the target.
7. Verify separation.
8. Withdraw the knife.
9. Transport and release the detached branch into the designated floor area.

The existing [bimanual task implementation](examples/greenhouse_sim/greenhouse_sim/deleaf_task.py) already represents grasp, cut, retention, transport, release, and deposit. Adapt that contract to the new assets rather than assuming the current preview implements it.

### 8.2 Two-arm feasibility constraint

The left hand cannot independently hold an obstructing leaf and the target petiole at the same time.

Include physically feasible strategies:

- Grasp the target itself and pull it into view.
- Change the camera viewpoint.
- Move an obstruction, release it, and verify visibility remains sufficient before grasping the target.
- Reject a sequence that requires an unavailable third contact.

Do not generate two independent arm paths and assume they form a feasible bimanual plan.

### 8.3 Trajectory labels and camera references

Keep full-resolution 3D trajectories as authoritative records. Derive sparse image-plane paths while preserving:

- Grasp/release events.
- Blade entry, crossing, and exit.
- Phase transitions.
- Synchronization dependencies.
- Re-observation points.

The left tracked point is the grasp center. The right tracked point is defined relative to the actual cutting edge, not merely the wrist origin. Knife orientation must respect the intended flat cutting edge rather than treating the arc as the blade.

Execution still requires full tool orientation, joint limits, inter-arm clearance, permitted contacts, and force/speed limits.

Attach every path to a reference camera pose and timestamp. Initially freeze the reference view within a phase. Regenerate or reproject guidance after head/wrist movement or a material change in target geometry.

### 8.4 Physical validity boundary

The older [cutting implementation](examples/greenhouse_sim/greenhouse_sim/cutting.py) releases pre-authored joints and records its geometric approximation. It is not a general-purpose mesh-slicing or tissue model. Preserve this limitation in data and benchmark claims.

Simulator force parameters are not automatically measured biological ground truth. Real cutting-force and grasp-damage limits require separate calibration.

## 9. Cosmos augmentation and label preservation

The primary risk is geometry drift: an augmented image may move or alter a small petiole while retaining its old labels.

Current [Cosmos 3 transfer examples](https://github.com/NVIDIA/Cosmos/blob/main/cookbooks/cosmos3/generator/transfer/README.md) support spatial controls including edges, depth, segmentation, and combinations of controls. This does not establish that the user's particular fine-tuned checkpoint preserves our structures accurately. Its checkpoint and conditioning interface remain TBD.

Proposed workflow:

1. Split source plant/episode families into train, validation, and test.
2. Render immutable RGB and geometry-derived labels.
3. Export appropriate depth, edge, and segmentation controls.
4. Generate augmented images with the selected checkpoint.
5. Check correspondence around the cut region, petiole centerline, grasp region, occluding leaves, knife, and gripper.
6. Accept only samples that preserve the required structural correspondence.

If geometry changes, reject the sample or re-annotate it and downgrade/remove inherited metric labels. Do not silently attach original XYZ or depth to changed imagery.

For sequences, test temporal consistency. Independent image-to-image conversion can introduce flicker, moving attachments, or inconsistent tools. Independently generated camera views can also violate multiview correspondence.

Keep each source and all its Cosmos derivatives in the same split. Real images used to customize Cosmos must also remain separate from the final real-world evaluation set.

Cosmos integration is optional until checkpoint availability, conditioning compatibility, and label-preservation quality have been verified.

## 10. Model selection, output contract, and training

### 10.1 Recommended first trainable model

Start with **Qwen3-VL-8B-Instruct**:

- It maintains continuity with the existing Qwen evaluation.
- Its documented capabilities include visual grounding and spatial/occlusion understanding.
- An official fine-tuning framework is available.
- It provides a smaller initial experiment than adapting the hosted 32B model.

These are reasons to test it, not evidence that it already solves petiole localization. References: [Qwen3-VL-8B model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) and [official fine-tuning framework](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune).

Use the existing **Qwen3-VL-32B-Instruct** deployment as a frozen comparison. It may assist with instruction paraphrases or brief descriptions, but geometry and reviewed rules must determine coordinate labels, not model guesses.

HAMSTER's VILA-based checkpoint is an optional research baseline, not a required dependency.

### 10.2 Supervised training progression

1. Grounding: target selection, cut-region localization, and negatives.
2. Observability: occluders, uncertainty, and inspect/reveal decisions.
3. Single-phase paths: grasp approach or cutting approach.
4. Coordinated phases: reveal, grasp, cut, retain, and release.
5. Domain robustness: accepted Cosmos variants mixed with raw synthetic data.

Begin with parameter-efficient tuning and profile memory at the actual image resolution. Model-weight fit alone does not establish that training batches will fit.

Preserve sufficient resolution for thin petioles. Compare full-frame input with a full-frame-plus-crop approach. Evaluation crops must come from the deployed selection process, not hidden ground-truth centering.

A conventional VLM does not become a metric RGB-D model by simply appending a fourth image channel. Later ablations may use a depth-image input or dedicated geometry encoder with explicit calibration and modality handling.

### 10.3 Structured output

Extend the existing [provider-neutral schema](examples/greenhouse_sim/vlm_eval/schema.py) with:

- `decision`: proceed / inspect / reveal / no-safe-plan / uncertain.
- Target region.
- Nullable cutting point.
- Visibility and hazard fields.
- Next phase.
- Left/right sparse paths.
- Left-gripper events.
- Re-observation requirement.
- Brief evidence-based explanation.

Use explicit schemas and coordinate conventions rather than free-form action text. The physical controller validates proposals independently of wording or reported confidence.

## 11. Implementation phases and acceptance gates

The phases below are acceptance milestones. Phase 1 tooling has started (see Section 15); Phase 1 human sign-off and subsequent milestones are not complete.

| Phase | Work | Completion gate |
|---|---|---|
| 1. Task and asset audit | Validate anatomy, attachment frames, deleafed flags, eligible targets, and cut/grasp regions | Reviewed examples agree with geometry; invalid stubs are not targets |
| 2. Synchronized exporter | RGB, metric depth, masks, calibration, provenance, and visibility scoring | Projection/depth tests pass; no stale or mismatched modalities |
| 3. Pilot dataset | About 300 observations across at least six plant assets, covering difficulty and negatives | Every pilot overlay reviewed; systematic label errors fixed |
| 4. VLM baseline | Frozen 8B/32B evaluation, geometry baseline, and first small fine-tune | Held-out results improve without increasing unsafe proposals |
| 5. Interaction adapter | Bring one detailed plant, RB-Y1 tools, and bimanual task logic into the new package | Repeatable grasp-cut-retain-release with logged physical evidence |
| 6. Path dataset | Initially 100-300 validated episodes, including reveal/re-observe cases | Replay verifies coordination, retention, and protected-object safety |
| 7. Scale and augmentation | Expand capture; add accepted Cosmos variants and real-camera evaluation | Data integrity and held-out performance justify expansion |

Phases 3-4 may advance while the physical adapter in Phase 5 is developed. Large-scale trajectory collection must wait for Phase 5's gate.

Execution checklist:

- [ ] Agree on target eligibility, acceptable cut region, protected organs, and disposal rule.
- [ ] Audit all 24 component plants and validate task-label mappings.
- [ ] Implement and test the synchronized multimodal capture contract.
- [ ] Implement visibility, difficulty, provenance, and leakage-safe splits.
- [ ] Produce and review the 300-observation pilot.
- [ ] Establish frozen-model and geometry baselines.
- [ ] Run the first small supervised fine-tuning experiment.
- [ ] Adapt and validate one-plant bimanual interaction in the new scene.
- [ ] Collect and replay-check initial interaction episodes.
- [ ] Verify the selected Cosmos checkpoint and label-preservation checks.
- [ ] Evaluate on held-out real-camera observations before transfer claims.
- [ ] Scale only after the preceding quality and performance gates pass.

## 12. Splits, scale, compute, and storage

### 12.1 Leakage-safe splits

An initial split could use 16/4/4 plant assets for training/validation/test after auditing asset relationships. Keep all views, poses, episodes, and augmented derivatives of each plant family together.

Four held-out plants provide limited generalization evidence. Broader claims require more independent geometries, ideally additional exports or access to the generator.

After the pilot, aim for roughly 5,000-10,000 controlled observations before considering much larger datasets. Measure learning curves and failure coverage rather than optimizing image count alone.

### 12.2 Resource isolation

Keep simulator rendering, VLM inference/training, and Cosmos generation in separate processes and environments. Avoid network calls in the physics loop and uncontrolled competition for the interactive simulator's GPU memory.

Benchmark capture throughput, training memory, and augmented-sample rejection rates before committing to large runs. No wall-clock completion estimate is established by this plan.

### 12.3 Storage budgeting

At 1280x720, 100,000 samples with three float32 depth views alone consume approximately 1.1 TB uncompressed. Measure actual compressed sample sizes and include RGB, masks, trajectories, and augmented derivatives in the budget.

Store reproducible code/configuration separately from large dataset artifacts. Preserve immutable sample identities, asset hashes, modality metadata, and label versions so runs can be audited and rescored.

## 13. Evaluation and evidence

Report perception/planning separately from physical execution.

### 13.1 Perception and planning metrics

- Correct target selection.
- Pixel error and metric error where depth/observation support it.
- Distance to the admissible cutting region.
- Localization error relative to projected petiole diameter.
- Occlusion and hazard classification.
- Unsafe-proposal rate and abstention coverage.
- Cross-view agreement.
- Schema validity and inference latency.

### 13.2 Physical execution metrics

- Successful left grasp and retention.
- Visibility improvement following a reveal action.
- Correct cutting-edge crossing and direction.
- Protected-organ and inter-arm contact violations.
- Correct branch separation.
- Knife withdrawal and designated floor deposit.
- End-to-end success rate with explicit failure categories.

Compare a privileged simulator oracle, a perception-based geometry baseline, frozen VLMs, and the fine-tuned model. The oracle is an integration test, not model performance.

Compare raw-synthetic training with raw-plus-Cosmos training on the same held-out real-camera data. More realistic appearance is not itself evidence of improved transfer.

## 14. Immediate next increment

Implement the task-label adapter and synchronized capture pilot on `koh-dev/sim-data` first.

The first deliverable should support review of:

```text
Input RGB -> exact cut/grasp labels -> occlusion evidence -> expected model response
```

Every sample should have complete provenance. Once these labels are trustworthy, VLM training can begin while the physical bimanual adapter is validated separately.

Before mass generation, confirm the horticultural target/cut rules, first camera/input configuration, available training resources, and the exact Cosmos checkpoint/conditioning interface. Those choices should be recorded in versioned configurations rather than silently assumed.

Related existing design: [VLM cut-point evaluation architecture](docs/vlm_cutpoint_evaluation.md).

## 15. Implementation log

### 2026-09-06: initial Phase 1 tools

- Added a standard-library manifest auditor, per-substem review records, source
  fingerprints, and optional assembled-USD placement/bounds diagnostics.
- Added session-only visual review with target navigation, attachment/subtree
  overlays, reversible isolation, and explicit reviewer/notes records.
- Audited all 24 plants: 870 intact leaf-bearing candidates need review and
  971 stubs are excluded. No structural blockers were found.
- Flagged 492 degenerate capsule chains, all on already-deleafed components,
  and 91 attachment-to-parent-AABB gaps over the 2 mm diagnostic threshold.
  These observations do not establish a safe cut region or physical validity.
- The headless single-plant UI smoke passed with a rendered review-only image
  and unchanged source-asset fingerprints.
- Human anatomy sign-off, horticultural cut/grasp rules, resolution of geometry
  warnings, and the initial multi-plant review set remain open. No targets have
  received approved cutting coordinates, and no training dataset was generated.

Commands, evidence paths, and test instructions:
[Phase 1 implementation notes](examples/greenhouse_sim/sim_data/PHASE1.md).

### 2026-09-07: batch review and exception sampling

- User feedback identified repetitive individual review as unnecessary overhead.
  Added six-card review-only viewport captures, explicit multi-selection, preset
  reasons, and deterministic plant-balanced normal/negative/exception sampling.
- Resume uses fingerprint-matching saved decisions, including original v1 files;
  stale/conflicting decisions cannot silently approve targets. Explicit rereviews
  supersede old decisions without rewriting their records.
- Anatomy, camera visibility and intended workspace are now separate reason
  scopes. No robot reachability is inferred from a user's workspace exclusion.
- The earlier roughly-50 review count is a sampling proposal, not a mandatory
  quota or achieved validation threshold. Systematic issues and annotation-rule
  coverage still need resolution before Phase 1 can be declared complete.
- Gallery images are inspection evidence only, never VLM training observations.
  Cut/grasp coordinates and physical feasibility remain unapproved.

### 2026-09-07: automatic configuration-specific reach diagnostics

- Added a review-panel check using current v1.2 robot/plant transforms, bounded
  position IK and a conservative endpoint overlap screen. Both arms are checked
  independently with fixed base and torso; there is no motion command.
- Reach results are separate snapshot-bound diagnostic JSON, not human anatomy
  decisions or dataset approvals. Relocated gallery samples are explicitly marked;
  failure to find IK is not labeled definitively unreachable. Orange outlines are
  review-only and disappear when the target or relevant geometry changes.
- The probe remains an uncalibrated finger-centre midpoint at the manifest
  attachment with unconstrained orientation. Approved cut/grasp regions, full
  collision/path checks and bimanual physical validation still need implementation.
  This increment reduces manual geometric guesswork but does not complete Phase 1.

### 2026-09-07: robot-conditioned review views

- Default robot-loaded review/gallery capture to the real mounted head D405 at
  848x408; retain explicit floating close-up inspection as a separate mode.
- Record view/robot/plant context and attachment projection in new review data.
  Projection into an image is not measured occlusion, and an annotated review
  image with overlays is not a clean training input.
- Interpret fixed-base reach failures only for that base/torso configuration,
  never as asset-global target ineligibility. Dataset stance sampling must precede
  image-dependent visibility and manipulation-feasibility labeling.
- Floor-supported, gutter-clear base/torso stance selection and pose-conditioned
  collision/path validation remain to implement; no automatic repositioning was
  added in this increment.
# 2026-09-09: first training-release implementation status

The active work is a **target-conditioned RGB cut-point/visibility/abstention
dataset**, not a bimanual trajectory dataset. The detailed versioned task,
release coverage/balance gates, provenance, export layout, unsupported features
and commands are in [TRAINING_DATASET.md](examples/greenhouse_sim/sim_data/TRAINING_DATASET.md).
The portable exporter and offline loader are implemented and tested. The
historical four-row v1 smoke package is obsolete for the current task contract.
The current v2 engineering package contains 317 validated rows and is explicitly
incomplete; no complete VLM training release exists yet. Dataset-scale
rendering/QA is still in progress.

The active contract is now `greenhouse.target_conditioned_cutpoint_rgb.v2`:
visual inspection caught queries on isolated visible petiole fragments. For a
localized answer, the visible query must connect to the cut in the same native
petiole mask component, without gap filling. Hidden-cut examples abstain;
ambiguous examples are excluded. Historical v1 counts and changed-query review
decisions are not current-v2 validation. Complete releases additionally require
hash-bound stratified visual inspection, not just sufficient image counts.

Implementation, tests, GUI/config assets and this plan are being checkpointed on
`koh-dev/sim-data`; generated datasets and review evidence remain in ignored
`data/sim_data/`. Cleanup does not remove that evidence. See `dev.md` for measured
checks, failed experiments, current limitations and pending release gates.

### 2026-09-09: full coverage recount and scale collection

All 24 source families now have successful coverage audits. The initial coverage
snapshot accepts 834 of 1,103 raw observations (544 train / 141 validation / 149 test).
Source-family coverage, target diversity and cut-location spread satisfy the
first-release checks; volume and partial-occlusion coverage do not. There are
only three medium examples in each split. No thresholds were relaxed to fill
this gap, and the complete training dataset has not been released.

The remaining 22 families are scheduled in a bounded four-lane campaign, with
the two already-running jobs retaining their slots until clean audited exits.
Training families use up to 160 candidate views per target and held-out families
64; both use new, non-overlapping viewpoint windows. Native captures, final
stratified QA and the portable release audit must finish before fine-tuning.
Operational details and the source-bound count report are documented in `dev.md`.

### 2026-09-09: all-family visual QA and first audited scale yield

Actual representative inspection now covers all 24 initial-coverage families:
41 prior plus 63 newly inspected, individually recorded v2 examples. Six more
examples from the first completed scale run cover its easy, medium and hard
strata. The combined 25-audit snapshot passes hash-bound stratified QA with
110 inspected examples and contains 1,522 eligible rows from 2,076 audited raw
frames (1,232 train / 141 validation / 149 test). These are synthetic annotation
checks, not human confirmation or a statistical label-accuracy estimate.

The first scale family contributes 688 eligible rows from 973 raw: 524 easy,
7 medium and 157 hard. Full-scene robot-head RGB, native masks, camera-Z and
query/cut association were inspected without changing the source images or
label thresholds. Strong backlighting and thin pixel-scale petioles remain
limitations. Training has 10 medium examples; validation and test each still
have three in this frozen snapshot. More collection is required before the
10,000/500/500 release, final QA/portable export, and any model fine-tuning.

The subsequent 26-audit snapshot, including the completed seed79 scale run,
contains 1,721 eligible / 2,343 raw (1,232 train / 141 validation / 348 test).
All 116 representative visual decisions verify against those exact sources.
Medium counts are now 10/3/8 by split; this remains below the release gates.

Depth-source requirement reaffirmed: acquire the native Isaac Sim Replicator
`distance_to_image_plane` output on the mounted head camera, preserve float32
metric camera-Z unchanged, and store validity separately. Our scripts may save,
validate and colour-display the sensor array, but must not generate replacement
depth from RGB, custom geometry or projected anatomical XYZ. Heatmaps are for
review only; RGB-D fine-tuning, if selected later, must read the native metric
array and calibration rather than those coloured PNGs.

### 2026-09-09: anatomy-audit correction and task v3

The user's wave-2 complaint was audited one image at a time across all 63 cards.
All 33 visible nominal labels mapped to their intended leaf-bearing petioles;
the 30 hidden nominal points were behind 16 leaves, 9 fruit and 5 main stems.
Those hidden samples had abstention answers, but the review renderer painted
white/magenta geometric projections on the foreground and was misleading.
Two old query points were inadequate (2-pixel and 49-pixel target islands).
The complete report preserves per-image anatomy, native identity/depth,
query crops and visual findings, without changing the source annotations.

Implemented task v3 query size/interior support, local exposure/contrast and
frame-context screens; alternate usable queries are selected deterministically.
Positive queries still require native visible connectivity to the nominal cut.
Hard examples still require a usable visible target query plus independently
identified cut occlusion; hidden XYZ is never an RGB localization answer.
Review cards now explicitly distinguish hidden cut evidence from a visible cut
and provide separate query RGB/native-mask/native-Z crops. The portable loader
rechecks usability independently. Native Isaac camera-Z and raw RGB remain
unchanged. No robot, optics, physics, teleop or source plant geometry was changed.

The revised policy retains 54/63 prior cards (32 localization / 22 abstention),
excludes 9 (including both visual holds and the reported seed41 image), and
changes 45 retained queries. These are conservative engineering thresholds,
not a statistically validated readability model. The original bundle, its
decisions and 63 images are preserved. A fresh hash-bound review bundle has four
actually inspected assistant accept records; other v3 reviews remain pending.
Full sim-data regression: 486 tests plus 47 subtests passed. Collection continues
under the original frozen family splits; complete-release gates are unchanged.

Open the corrected browser:
`data/sim_data/dataset_reviews/grounding_wave2_rescreen_20260909_v3/index.html`.
The underlying one-by-one anatomy report is
`data/sim_data/dataset_audits/wave2_anatomy_20260909/audit_report.html`.
Next: complete fresh stratified QA of v3 candidates across completed batches,
fill volume/medium-class deficits with native robot-head captures, and build
and validate a portable release before VLM fine-tuning. Do not use historical
v2 eligible counts as current v3 totals or migrate changed-query approvals.

The subsequent frozen 29-audit v3 recount is complete: 3,246 candidates from
4,623 audited raw frames (2,102 train / 368 validation / 776 test). All native
Isaac camera-Z arrays match their original capture byte hashes. Train medium
count is 22; validation/test have only 6/9, so held-out balance still fails.
Fresh v3 visual QA, train volume/class counts and validation volume remain
incomplete. See `data/sim_data/status/post_anatomy_audit_v3_20260909.json`.
Process inspection also found interrupted campaign supervision: two renderers
remain active, while seed37/seed29 outputs have no final worker-exit ledger.
Those unfinalized jobs are not counted. Restore bounded durable supervision
and recover provable exits or rerun into new directories before resuming the
queue; a completed frame manifest/shutdown log is not proof of clean execution.

### 2026-09-09: recovery running and fresh review progressing

Implemented durable pre-audit exit receipts, exact-handle Windows observation
and detached three-lane resume supervision. Five completed scale jobs are
preserved, one live seed17 job produced a verified zero exit and entered
independent recovery audit, and 18 unproven/unstarted jobs are assigned new
capture directories. Seed103 exited before observation and is being recaptured,
not retroactively declared successful. Native depth and RGB are unchanged.

Fresh v3 inspection now covers 20 unique wave-2 cards (19 accepts, one hold).
The new seed61 query hold is enforced at source audit level as well as in task
review. The current hold-adjusted baseline is 3,245 candidates / 4,623 audited
raw (2,102 train / 368 validation / 775 test). The 1,164 recovered seed17 frames
are not added until their independent audit and v3 label checks finish.

The all-family v3 bundle now has 108 cards. Fourteen exactly matching previously
inspected v3 task/card hashes have explicit reused evidence; the remaining 94
need actual inspection. These are not old v2 approvals or 14 new independent
samples. Complete release gates remain unmet and training has not started.
See `examples/greenhouse_sim/sim_data/COLLECTION_RECOVERY.md` for running paths,
scope and restart limitations. Regression: 494 tests plus 47 subtests passed.

Recovery follow-up: the seed17 independent audit and v3 label checks finished,
adding 890 eligible rows from 1,164 raw frames. Every native depth buffer matches
its original capture hash and no cross-snapshot RGB/view duplicates were found.
The current combined snapshot is **4,135 candidates / 5,787 audited raw frames**
(2,992 train / 368 validation / 775 test), bound in
`data/sim_data/status/recovered_seed17_v3_20260909.json`. Six representative
recovered cards were individually inspected and accepted: four visible petiole
cut points and two correct leaf-occluded abstentions. Fresh v3 visual QA now has
26 unique decisions (25 accepts, one enforced hold), with 94 broader cards still
pending. Assistant inspection is not human/agronomic or physical cut approval.

The recovered lane advanced to a new seed103 job; seed37 and seed31 are saving
new native head-camera frames under detached supervision. These unfinished jobs
do not inflate audited totals. The first resumed frame's RGB/source hashes and
unchanged native camera-Z were verified. Progress reporting now recognizes the
recovered audit path; final regression is 495 tests plus 47 subtests passed.
Train volume/class counts, validation volume, held-out medium balance, remaining
QA and portable release validation still precede actual VLM fine-tuning.

Further one-by-one QA inspected 15 broader cards (14 accepts, one hold). The
new seed61 nominal cut point blends into a tomato in untouched RGB, despite
correct native petiole identity. It is held at source/export level for RGB
readability uncertainty, not relabeled as anatomical fruit or depth-occluded.
The current snapshot is **4,134 candidates / 5,787 audited raw** (2,992 train /
368 validation / 774 test), at
`data/sim_data/status/v3_after_nominal_rgb_hold_20260909.json`.

Fresh v3 QA now totals **41 unique inspections: 39 accepts / two enforced holds**.
The broader set has 79 pending cards and its new held selection needs replacement
in a fresh QA bundle. All old sources/reviews are retained; exact reused reviews
are not additional independent inspections. See
`data/sim_data/status/visual_qa_followup_20260909_v3.json`. Query readability
alone cannot certify nominal-cut readability; explicit RGB inspection remains
necessary. Complete release gates still apply. Latest tests: 495 + 47 subtests.

The current task-v3 human review GUI is now available at
<http://127.0.0.1:8880>, with `examples/greenhouse_sim/run_training_review.cmd`
as its repeatable launcher. It starts with the remaining 79 pending cards,
displays untouched head-camera RGB and saved native evidence, and saves explicit
Accept/Hold/Reject decisions with progress. Existing assistant records remain
read-only; negative human decisions also block the source frame before the task
record is saved. No observation, native depth or frozen snapshot is overwritten.
See `examples/greenhouse_sim/sim_data/TRAINING_REVIEW_GUI.md`. Full tests:
516 passed plus 47 subtests; actual browser rendering verified without saving
any automatic human reviews. This UI does not grant whole-release or cut-safety
approval and does not replace the remaining dataset collection/QA/export work.

### 2026-09-11: portable Qwen handoff increment

The user will transfer the completed release to H200 servers before training.
No local/server training was launched. See
`examples/greenhouse_sim/sim_data/H200_HANDOFF.md` for the exact task, portable
file layout, validation command, model adapter and remaining acceptance gates.

Two additional bounded batches contain 98 raw captures / 84 eligible v3 rows
(11 train, 29 validation, 44 test), with 17 actual assistant visual QA accepts.
Depth is unchanged native Isaac optical-Z; all original review decisions and
family reservations remain. The new 84-row engineering export validates
portably but is **not a complete training release**. It has no medium examples
and does not replace the historical 14,235-candidate broader dataset recount.
No new qualified dynamic episodes or trained action policy are claimed.

Continuation: `grounding_sunday_20260911_v4` completed 106 additional native
captures (35 validation / 71 test), both clean worker exits and independent
audits. Current task labeling retains 79; 27 are excluded. Eight representative
cards were individually inspected as assistant QA. There are no medium samples
in this batch, so it does not close held-out medium coverage.

The combined increment preview is now
`data/sim_data/training_exports/grounding_sunday_20260911_engineering_v2`:
**163 rows** (11 train / 60 validation / 92 test), **25 representative reviews**,
from 204 raw captures. Portable RGB/native-depth validation passed without Isaac;
the normal training loader rejects its explicit incomplete-release status.
This is not the final H200 transfer archive and must not replace the broader
corpus. The historical global recount still requires deduplicated reconciliation.

Additionally, 22 previously unreviewed broader cards were actually inspected and
accepted with individual notes. The 108-card broader queue now has 58 accepts,
8 holds, 3 rejects and 39 pending. Existing decisions are preserved. Continue
stratified QA and targeted native partial-occlusion capture; the recent random
view increments have not yielded medium examples. Do not relax their definition
or collection/release gates to meet a deadline. No training, remote server
access, hardware action or new dynamic training episode was started.

### 2026-09-11: Requested multi-agent label reassessment and release preparation

Three independent assistant work threads visually inspected 51 full original
RGB/card pairs, with hash-bound per-image reasoning. They assessed all 37
remaining pending cards (33 support, four hold) and 14 existing uncertain cases
(11 hold including one new disagreement, three reject). These are assistant
findings, not invented human decisions or statistical dataset-wide accuracy.
The broader 108-card queue is now 91 accept / 14 hold / 3 reject / zero pending;
one existing human acceptance remains intact but has a separate assistant
source hold, so its sample is excluded pending disagreement resolution.
No source image, depth array, split or prior review was changed.

New evidence: `data/sim_data/dataset_audits/release_20260911_agents/`.
The exact model input/output and annotation limitations are documented in
`examples/greenhouse_sim/sim_data/DATASET_CARD.md`. Task v3 is still original
robot-head RGB plus explicit target query -> cut UV / visibility / abstention,
not autonomous target discovery, XYZ action generation or grasp-cut episodes.

Implemented opt-in exact-label review reconciliation for exports, preserving
negative decisions and retaining provenance for reviews that no longer count
toward the current subset. Added directory-snapshot checks to reject a new
review/hold appended while an export is being copied. Neither change weakens
the numerical/stratified release gates. Three fresh medium-label holds make
new native partial-visibility capture and its actual visual review necessary.
The reconciled diagnostic (`release_20260911_agents/preflight_02/summary.json`)
now covers 63 audits / 20,151 raw / 14,404 candidates: 11,637 train,
1,369 validation and 1,398 test. It includes the nonstandard recovered seed17
audit and 15 new native seed61 frames, all easy, and excludes seven fresh
source holds. The full release remains blocked by validation medium 2/20 and
test medium 17/20, plus current-subset representative QA/final export work.
These are not 14,404 individually approved labels. The follow-up count reuses
the first diagnostic snapshot; it is not a replacement for normal export
validation. Two more validation-medium original/audit-card pairs were held,
and two new seed61 original/task-card pairs were assistant-accepted (55 unique
new targeted inspections overall). Source holds now follow duplicate RGB
hashes across audits. No source images/depth, human records, splits or gates
were altered. A narrower visible/occluded baseline is an explicit user choice,
not an automatic gate waiver. No final archive, actual-model training or H200
job is claimed yet.

## User-authorized first baseline: visible/occluded only (2026-09-11)

The first handoff now targets the explicit `visible_occluded_v1` profile:
easy/clear localization and hard/occluded abstention, no medium/partial rows.
The balanced release's requirements remain unchanged. Normal validation still
requires the established row counts, 16/4/4 frozen families, target diversity,
localization spread and representative visual QA. A separate completion state
prevents conflating this narrower experiment with the balanced release.

Legacy task-version negative RGB decisions are preserved through bound
`negative_review_history.json`; changing a query does not quietly override an
older hold. The original labels/review decisions/splits are not edited.
An independent snapshot check found 115 exact current representative accepts
and 11 missing inspections, plus one legacy held image to exclude. The final
normal source rebuild and fresh QA are in progress; a ZIP is only published
after normal portable validation and every archived member's hash check.
This remains static target-conditioned RGB perception, not dynamic action
supervision. Four-H200 training remains a user-side step after handoff and
real-processor/loss-mask/forward-backward validation.

The first new baseline QA wave inspected 24 original/card pairs and recorded
18 attributed assistant accepts plus six holds. One seed37/hard representative
gap remains at this checkpoint; the remaining unheld candidates are prepared
in a second immutable bundle for actual inspection. No medium requirement or
existing hold was relaxed. Final export now accepts a hash-bound explicit list
of completed review bundles at finalization, retaining the earlier pinned
history. Parallel source hashing is bounded to four workers/256-future batches
and does not replace or cache away any source-byte verification.

The second immutable QA wave is now complete. Across both new bundles,
35 original/card pairs received 22 assistant accepts and 13 holds. Preflight
now has 136 exact current representative accepts and passes all narrowed-profile
QA strata. Expected baseline: 14,259 images (11,520 train / 1,358 validation /
1,381 test), 11,237 easy and 3,022 hard. The normal rebuild is still deriving
original labels; these are not a claim that the final ZIP exists. All medium
and ambiguous held rows remain excluded, frozen families unchanged. This is
static target-conditioned localization/abstention data, not bimanual episodes.

## Completed narrowed release and Qwen server handoff (2026-09-11)

Normal source rebuild, image/native-depth/label validation and archive-member
verification have now completed. **14,259** records:11,520 train /1,358 validation
/1,381 test;11,237 easy/clear localized and3,022 hard/occluded abstain.24 frozen
source families and246 targets,136 exact representative assistant QA accepts.
Medium/partial and dynamic trajectories are not included; the balanced release
remains a separate unfinished task. Existing source images, native depth,
human/assistant decisions and split assignments are unchanged.

Transfer `data/sim_data/training_archives/visible_occluded_20260911_v1.zip`
(23,453,235,122 bytes) and its `.sha256`. SHA256:
`23e180dcb74f6daf0437414513b023d41d76f29e755aa1e2fb1638f6b268db74`.
All71,306 members were independently reverified after a post-publication
receipt-writing error; see the immutable `baseline_archive_receipt.json` and
preserved logs under `dataset_audits/release_20260911_agents`.

The executable server instructions are
[H200_RUNBOOK.md](examples/greenhouse_sim/sim_data/H200_RUNBOOK.md).
Qwen's upstream container differs from our canonical chat files. Use the
role-preserving adapter, which now explicitly supports Qwen's recommended
normalized0..1000 grounding coordinates for BOTH the query and answer.
The release itself remains original848x408 pixels. The complete conversion
audit passes with max pixel roundtrip error1.14e-13; abstention remains null.
Never treat normalized coordinates as pixels or feed sidecar depth/masks/XYZ
to this RGB-only task. Preserve the adapter convention with each checkpoint.

`h200_train.py` is an explicit local-snapshot-only H200 recipe: real processor
checks, one-GPU smoke,32-example train-only overfit, then4-GPU language-attention
LoRA. It has not been run with actual Qwen weights. Server processor/mask,
finite forward/backward, checkpoint reload, memory, speed and held-out generation
must still be verified. The user requested model setup only on H200; no Qwen
model files were downloaded here and no training/server job was started.
This is a perception-training handoff, not evidence of VLM-controlled cutting.

## Fresh clear-cut-point recollection (2026-09-15, overnight in progress)

The user rejected packaging the 47 older limited candidates and requested fresh,
clearer simulator captures, review, annotation and a Qwen3-VL-8B training ZIP.
The earlier 14,259-row visible/occluded release above is historical, not this new
easy-task release. No new H200 training or local model download has been started.

Task: given unmarked full-scene robot-head RGB and a visible target-petiole query
pixel, predict the nominal cut point in the original 848x408 image. Nominal is
10 mm along the petiole from its attachment; the 10-20 mm centreline interval is
the evaluation tolerance region, not a circular neighbourhood or a cutter pose.
The optional second RGB input is a query-centred crop, never an answer-centred
crop. Native optical-axis Z, validity, calibration and target masks are copied
as audit sidecars, not supplied to the RGB model. This is static perception data,
not action-outcome experience, contact validation or safe-cut certification.

New capture preset `robot_head_close_diffuse_v1` retains the supplied greenhouse,
native plant geometry, mounted robot-head camera and 848x408 optics. It searches
closer static robot/root/torso/head poses with diffuse scene lighting. Floor,
joint-limit and visible-geometry overlap screens remain enabled. Pose snapshots
do not establish navigation or dynamic collision-free transitions. Existing
source assets, split reservations and historical review decisions are unchanged.

At 01:52 KST, eight completed family jobs produced 208 native captures. The new
strict screen retained 127 images from 27 targets in seven TRAIN families; all
127 were inspected individually in original RGB and have hash-bound, named
assistant decisions. Seed23 produced one native frame and zero strict candidates.
These are measured intermediate counts, NOT a complete/approved training ZIP.
Held-out jobs are scheduled after training-family jobs; no held-out results have
been used to select model settings. The 24 source families remain frozen at
16 train / 4 validation / 4 test; they share greenhouse scenery and do not imply
real-world or fully scene-disjoint generalization.

Release criteria stay at >=500/100/100 images, >=60/15/15 targets and 16/4/4
families (train/validation/test), <=12 diverse views per target, plus strict
native-mask/visibility, projected size, exposure and explicit visual review.
These are minimum experiment gates, not evidence that such a small release will
generalize. The explicit `assistant_reviewed_experiment_v1` policy does not claim
independent human validation. Default human-heldout review policy is unchanged.

Evidence and live outputs:

- Plan: `data/sim_data/collection_plans/clear_capture_20260915_overnight_v1/plan.json`.
- Serial native campaign: `data/sim_data/collection_campaigns/clear_capture_20260915_overnight_v1`.
- Completed-batch source pools, strict drafts and review pages:
  `data/sim_data/collection_intake/clear_capture_20260915_overnight_v1`.
- Individual review JSON: `data/sim_data/dataset_reviews/clear_capture_20260915_overnight_v1`
  and the first seven decisions in `dataset_reviews/clear_native_seed101_20260915_v1`.
- A bounded seed23 opposite-aisle pilot is queued AFTER successful completion of
  the original-side campaign. It is not yet native-qualified or scaled up. Same
  geometry, optics, floor and clearance screens; no validated motion between aisles.
- `examples/greenhouse_sim/sim_data/CLEAR_CUTPOINT.md` contains the H200 smoke,
  train-only overfit, full-FT and sealed-test instructions. Final archive creation
  must pass `clear_cutpoint_release.validate` and member-hash verification first.

Implementation through `d291017`: serial native collection with bounded exits,
non-approving completed-batch intake, review provenance, portable RGB/crop adapter,
release validator and manifest-generated ZIP dataset card. Latest CPU regression:
671 tests plus 47 subtests passed (`data/sim_data/clear_regression_20260915_v5.log`).
No new-model accuracy, throughput or successful final ZIP is claimed here.

### 2026-09-15 02:59 KST collection/review checkpoint

The original-side serial campaign has completed 15 training-family attempts:
300 native frames, 194 strict candidates / 38 targets / 11 represented families.
All194 original RGBs were individually assistant-inspected: 192 accepts and two
holds in seed83 (immediate junction uncertain behind foreground leaf despite
native proximal-visibility pass). These are interim counts, not training-ready
coverage. The last training family and eight reserved held-out families remain.
No old47 images were substituted, no model training/weights were downloaded.

New hash-bound negative-review exclusion preserves holds in a separate NEW
derivative with the original source index and image hashes. Source reviews are
not overwritten. Evidence: `job_022_reviews.json` in the overnight review folder.
The strict quality/coverage gates and family split reservations stay unchanged.
Target-conditioned full RGB plus query-centred crop remains the proposed input;
the crop may omit the cut, so full RGB is always supplied. Native metric Z is a
sidecar, not an RGB-derived estimate or current model input.

CPU-only seed47 probe (matching Isaac USD0.25.11) matched all864 original
decisions against the native run; original/opposite/oblique/opposite-oblique
each found zero geometry-admissible views. A separate opt-in forward-torso-lean
probe found one lean+oblique candidate so far. That is NOT rendered visibility
or native qualification. See `data/sim_data/geometry_probes/clear_seed47_20260915_v1`
and `clear_seed47_20260915_lean_v1`. No source geometry or camera mounting moved.

Data regression before negative-review exclusion:682 tests +47 subtests passed
(`clear_regression_20260915_v7.log`); exclusion/release focused suite:27 passed
(`clear_exclusion_tests_20260915_v2.log`). ZIP packaging already has a measured
isolated extracted-code validation/model-input smoke test, but no real release
ZIP or Qwen training performance is claimed until coverage and reviews pass.

Post-exclusion full data regression passed688 tests +47 subtests in48.21s
(`clear_regression_20260915_v8.log`). The completed lean probe found one
lean+oblique geometry candidate, zero in lean-original or lean-opposite. One
native seed47 lean+oblique pilot is queued after the existing serial pilots;
it has not run or approved any image at this checkpoint.

### 2026-09-15 03:21 KST review and scheduling update

The first pass through all16 training families completed:355 native frames,
227 strict images /43 targets /12 represented families. Every227 RGB was
assistant-inspected:224 accepts and3 holds. The first validation family added
6 raw/5 strict images across3 targets, visually4 accepts/1 hold. Holds concern
immediate junction occlusion, not a reassignment of the nominal cut point.
These totals still fail the release coverage gates; no final ZIP exists.
The remaining held-out capture is running serially.

Real-data exclusion validation produced a NEW22-image seed83 draft with
22 hash-bound accepts and both earlier holds retained outside the selection:
`collection_intake/clear_capture_20260915_overnight_v1/job_022/curation_v1`.
No source or review decision was overwritten.

Forward lean alone did not yield seed53 geometry candidates. The new explicit
target-facing root-heading/orbit proposal found one candidate in the small
seed53 CPU test; native visibility remains unknown. Original CPU decisions
matched936 native reference decisions. All changes are opt-in; original plan
behavior remains covered by hash and exact-kinematics regression tests.
Full suite691 tests +47 subtests passed in50.95s
(`clear_regression_20260915_v9.log`).

Three assistant-owned waiting pilot queues were replaced before they launched:
opposite-aisle, position-only oblique and the old chained lean queue. Their
PowerShell PIDs128668/20312/80828 were verified to have no children and stopped;
no active renderer, captures, source files or user application was stopped.
Replacement queue `clear_replacement_pilots_queue_20260915_v1.log` waits for the
original campaign's success and zero Kit processes, then tests seed53 orbit
and seed47 lean-oblique sequentially, with the existing memory/exit/audit guards.
Neither pilot nor any scaled alternate campaign is yet native-qualified.

Additional checkpoint: the first seed31 test capture completed with53 native
frames and31 strict candidates (visual review in progress). Through this job:
472 raw /300 strict candidates;269 individually reviewed so far with265 accepts
and4 holds. These are incomplete per-job counts, not a released training set.
No model was evaluated on these held-out images. Collection continues.

`clear_collection_campaign --jobs` can select explicit unique existing job IDs
without modifying the frozen source plan. It preserves train-first ordering,
records omitted versus selected jobs, and does not count omitted jobs as
collected. This avoids recapturing an already-qualified pilot when scaling an
alternate viewpoint campaign; no alternate scale-up has been launched yet.
Full data regression696 tests +47 subtests passed in47.13s
(`data/sim_data/clear_regression_20260915_v10.log`).

### 2026-09-15 03:53 KST capture-boundary and intake fixes

All31 seed31 test RGBs were individually inspected with GT-coordinate guidance
for label QA:30 accepts and1 hold (foreground leaf hides the immediate junction).
Total reviewed so far300, with295 accepts/5 holds; source labels unchanged.
Held-out QA is not model evaluation or independent human validation.

Code inspection found the native prepared-view recheck omitted lean/orbit flags.
`prepared_view_specs` now reconstructs all opt-in proposal flags exactly; existing
tamper and geometry checks remain enabled. These flags were absent in the
original running campaign, so that campaign's captures are not affected. Native
alternate-pilot qualification is still pending. Regression713 +47 subtests passed
before the separate intake fix below (`clear_regression_20260915_v11.log`).

The first intake watcher stopped on a valid seed37 batch with two native images
but no eligible query-conditioned labels. This is not a native capture failure.
The export now raises a dedicated validated-empty exception after its source and
review checks. Intake records that case without approval and continues; all other
exceptions remain fatal. The failed intake directory/receipt is preserved. A new
intake run will revalidate completed batches rather than overwrite old outputs.
Full data regression714 +47 subtests passed in46.19s
(`clear_regression_20260915_v12.log`). No coverage gate was relaxed; no final ZIP.

The new `clear_capture_20260915_overnight_v2` intake successfully recorded seed37
as zero eligible labels without failing; original native collection continued.
Seed59 added one strict image, individually accepted (301 reviewed,296 accepts,
5 holds). A bounded seed47 CPU comparison of original/orbit/opposite-orbit is
running; opposite-orbit is only a composition of already-supported flags, not a
new geometry allowance. Focused reconstruction/probe tests19 passed in0.50s.
Two additional immutable plans are prepared but NOT launched:
`clear_capture_20260915_original_shard2_v1` (new offset12 proposal windows) and
`clear_capture_20260915_orbit_opposite_v1`. Their24 family assignments exactly
match the frozen first plan. The prior alternate native pilots remain queued.

### 2026-09-15 04:19 KST review and closer-base search checkpoint

Seed61 added46 native frames /42 strict candidates; all42 originals individually
assistant-inspected and accepted. Completed review totals343,338 accepts/5 holds.
The original campaign has reached its final seed97 family. Counts are per-job
screening totals, not a completed/approved aggregate. No model training/evaluation
has run. Reviews are in `dataset_reviews/clear_capture_20260915_overnight_v1`;
new intake artifacts are under `collection_intake/clear_capture_20260915_overnight_v2`.

The seed47 CPU original/orbit/opposite-orbit probe found zero candidates in its
small72-proposal/target window;864 original decisions matched native reference.
Native process exit was explicitly0. This does not prove no valid view exists.
A separate .20-.40m closer-base opt-in is now CPU-tested against the same source,
without changing camera mount/optics/resolution or admission screens. Near-oblique
requires target-facing orbit so radius bounds are checked consistently. Default
proposals remain unchanged. Full data regression722 tests +47 subtests passed
in48.17s (`clear_regression_20260915_v13.log`), including real robot mount/FK/floor
and exact prepared-view reconstruction. No closer-base native run is qualified.

An optional user question asks about a SEPARATE higher-resolution native capture
set. No answer has been received and no resolution change has been implemented;
current collection remains848x408. No upscaled image is counted as a new capture.

### 2026-09-15 completed first-pass capture and portable checkpoint

All24 original jobs finished:20 audited native batches and4 verified zero-view
batches.523 native frames produced344 strict candidates. Every344 original RGB
was inspected by the assistant:339 accepts/5 holds. The NEW curated derivative
preserves the five negative decisions and passes all339 review/hash checks, with
no newly selected unreviewed image. Exact combined counts:

| Split | Images | Targets | Families |
|---|---:|---:|---:|
| train |224|43|12|
| validation |42|8|3|
| test |73|11|3|

Coverage is still below the declared500/100/100-image,60/15/15-target and16/4/4-family
gates. No final training release is claimed. Curated files and review page:
`data/sim_data/collection_intake/clear_capture_20260915_overnight_v2/curation_v1`.
Evidence: `clear_firstpass_curation_20260915_v1.log`.

A clearly marked INSPECTION checkpoint was created, not substituted for the
requested final training ZIP:
`data/sim_data/transfers/clear_firstpass_339_reviewed_20260915.inspection.zip`
(776812080 bytes, code52477b5).
SHA256 `26d61677c74b55b27a694ced72218fcf75ec937291f8d3709af8f9a92608f735`.
All ZIP member hashes were verified. Independently extracted REAL data validated
with bundled code in Python isolated mode, including normalized full-frame/crop
inference input construction, no answer turn, no Isaac imports, and correct draft
training refusal. No model weights/training were loaded. Evidence:
`clear_firstpass_transfer_20260915_v1.log` and
`clear_firstpass_portability_20260915_v1.log`.

The bounded seed47 closer-base CPU diagnostic completed with explicit native
exit0 in995.83s: original/near/near-orbit/near-opposite-orbit all found zero
geometry-admissible candidates in72 proposals/target;864 original decisions
matched native reference. This is negative diagnostic evidence, not a global
infeasibility claim. No near-mode native capture is qualified. The original
campaign is complete and the queued seed53 orbit native pilot is now running;
seed47 lean-oblique remains next. No larger alternate campaign has launched.

### 2026-09-15 alternate native capture qualification

The seed53 target-facing orbit pilot completed native capture with exit0 in
587.79s. Its one image passed the native depth/identity audit, strict clear
screen and individual original-RGB visual review. The exposed petiole joins its
diagonal parent near(638,320); the nominal point(650,328) is connected to the
query(728,350), away from the tomato below-left. Review attribution is assistant,
not independent human or physical cutting approval. This is one newly represented
training family, not proof that every orbit pose or family is suitable.

Evidence: `collection_batches/clear_capture_20260915_orbit_pilot_v1`,
`collection_intake/clear_orbit_pilot_20260915_v1`, and
`dataset_reviews/clear_capture_20260915_orbit_v1/job_014_reviews.json` under
`data/sim_data`. The original339-image curated draft/inspection ZIP is unchanged.
The additional image has not yet been merged into a new aggregate release.

A serial23-job orbit campaign is queued behind the still-running seed47
lean-oblique pilot, omitting only the already captured seed53 job014. It requires
the prior pilot to have a validated audited/empty outcome and no remaining Kit
process before launch. Each worker retains the1800s bound, memory reserve,
unchanged full greenhouse/mounted848x408 camera and native admission checks.
Campaign: `collection_campaigns/clear_capture_20260915_orbit_v1`; log:
`clear_capture_20260915_orbit_campaign_v1.log`. A separate non-approving intake
watcher is queued at `collection_intake/clear_capture_20260915_orbit_v1`.
There is no automatic visual acceptance, source/split mutation or training.

The lean-oblique seed47 pilot subsequently exited0 in687.08s. Both original
RGBs passed the strict screen and individual assistant review; the petiole-parent
junction is exposed in both, at nominal(302,290) and(712,147). New reviews are in
`dataset_reviews/clear_capture_20260915_lean_oblique_v1/job_013_reviews.json`.
These two images and the seed53 image remain separate from the339-image aggregate.

The first orbit campaign launch used the wrong controller interpreter (conda);
its first child stopped before rendering with `ModuleNotFoundError: isaacsim`.
No samples were accepted, and the stopped campaign/intake v1 directories and
exit receipts are preserved. The corrected launch explicitly uses
`D:/isaac-sim-6.0.1/python.bat -m sim_data.clear_collection_campaign`, because
workers inherit `sys.executable`. The active replacement is
`collection_campaigns/clear_capture_20260915_orbit_v2`, with non-approving intake
at `collection_intake/clear_capture_20260915_orbit_v2`; logs have matching v2
suffixes. No memory/geometry checks were bypassed and no source dataset changed.

### 2026-09-15 05:15 KST reviewed aggregate checkpoint

The NEW `collection_intake/clear_combined_20260915_checkpoint_v2` merges only
the completed first-pass audits and the two qualified pilots. All342 selected
RGBs have individual attributed accept records; the original five holds remain
excluded. Validation passed, but coverage still fails: train227/45/14,
validation42/8/3, test73/11/3 (images/targets/families). Review page:
`review/index.html`; evidence:`clear_combined_342_curation_20260915_v1.log`.
The earlier339-image inspection ZIP remains unchanged; no final training ZIP.

Read-only TRAIN input characterization found that218/227 query-centred crops
contain the complete accepted interval. The full RGB is always supplied for
all227. No answer-centred recropping or held-out model selection occurred.
Evidence:`diagnostics/clear_checkpoint_342_train_input_quality_20260915_v1.json`.

The corrected orbit campaign's first job exited0 in699.91s: eight native images,
four strict candidates, all four individually inspected and accepted. They add
one petiole not represented in the prior aggregate; these four have not yet
been merged/capped with the full collection. Reviews:
`dataset_reviews/clear_capture_20260915_orbit_v2/job_001_reviews.json`.
The next family is rendering. Total individual reviews so far351, with346
accepts/five holds; this is NOT a final combined release count. No local model
download, training or held-out model evaluation has started.

### 2026-09-15 05:40 KST capped selection and next coverage pilots

Orbit job002(seed103) contributed30 native frames/17 strict candidates. All17
were individually inspected:16 accepts and1 hold. The held image
`seed103_full_265d54eb11f2cedb994f` has a fruit-adjacent immediate attachment
that could not be confidently judged fully exposed, despite a visible nominal
point. Its hash-bound decision is preserved, not overridden by numerical gates.

The NEW `collection_intake/clear_combined_20260915_checkpoint_v3` combines the
first pass, both qualified pilots and orbit jobs001/002. All357 selected images
have individual accept records; six holds are excluded and per-target view caps
remove five otherwise accepted views. No newly selected image lacked review.
Counts:train242/46/14, validation42/8/3, test73/11/3 (images/targets/families).
The validator passed artifact/review checks but all coverage categories remain
incomplete. Evidence:`clear_combined_checkpoint_v3_20260915.log`. Across reviewed
candidates there are368 inspections,362 accepts/six holds; those uncapped counts
must not be reported as final release size.

Orbit job003(seed11) exited0 with14 native frames but zero strict clear
candidates. Its seven exportable labels were not easy/clear; they stay excluded.
The campaign continues without weakening visibility rules or relabeling them.

Two bounded opposite-aisle orbit pilots (training seed101/job001, seed73/job019)
are queued ONLY after the current23-job orbit campaign finishes successfully and
all Kit processes exit. They use the existing source-bound
`clear_capture_20260915_orbit_opposite_v1` plan, unchanged848x408 mounted camera,
1800s worker bounds and native memory/geometry/depth checks. Queue wait is bounded
at6h; no full opposite-side campaign or approval is automatic. Planned outputs:
`collection_batches/clear_capture_20260915_opposite_orbit_seed101_pilot_v1` and
`collection_batches/clear_capture_20260915_opposite_orbit_seed73_pilot_v1`.
Log:`clear_opposite_orbit_pilots_queue_20260915_v1.log`. Native/visual qualification
is still pending; these are static poses, not a validated aisle-crossing path.

### 2026-09-15 06:05 KST fresh-data checkpoint

Orbit job005(seed17) supplied nine strict candidates and job006(seed19) eight;
all17 originals were individually inspected and accepted, using existing review
crops where attachment detail needed closer inspection. No blade-clearance or
physical safety approval is implied by these visibility reviews. Job007(seed23)
finished as a verified zero-screened-viewpoint result and was skipped without
relaxing admission checks. Job011(seed41) is collecting next.

The NEW `collection_intake/clear_combined_20260915_checkpoint_v4` validates
369 selected images with369 attributed accept records and no unreviewed newly
selected image. Six holds remain excluded. Counts:train254/48/14,
validation42/8/3, test73/11/3 (images/targets/families). Coverage still fails;
this remains `draft_clear_cutpoint_not_for_training`. Evidence:
`clear_combined_checkpoint_v4_20260915.log`. Individual reviewed-candidate totals
are385 images,379 accepts/six holds before combined view capping; do not confuse
these with the369 selected images.

A separate second look at all six held original RGBs retained every hold,
including the borderline fruit-adjacent example; it neither changed decisions
nor increased unique review counts. This is the same assistant, not independent
validation. Hash-bound notes:
`diagnostics/clear_holds_secondary_20260915_v1.json`.

Saved an additional INSPECTION checkpoint, not a final training release:
`data/sim_data/transfers/clear_checkpoint_369_reviewed_20260915.inspection.zip`
(844377452 bytes, bundled code`cdcb5a3`). SHA256:
`0c23ebc95eab80eb7f36f3938a098003130b4a6ef1967507a63066527605c644`.
All archive members were hash-verified. A new extraction then validated all369
real rows using only bundled code in an isolated Python interpreter, constructed
two-image inference input without an answer turn, imported no Isaac/USD, and
correctly refused normal training validation for the draft. No weights or training
were loaded. Evidence:`clear_checkpoint_369_transfer_20260915_v1.log` and
`clear_checkpoint_369_portability_20260915_v1.log`. The prior339-image ZIP is
preserved. Native collection and the queued opposite-side qualification continue.

### 2026-09-15 06:30 KST continued collection and individual review

Orbit job011(seed41) exited0 after721.83s, with84 native frames and41 strict
clear candidates. All41 original RGBs were individually inspected and accepted;
existing query-centred crops resolved four fruit-adjacent attachment details.
These decisions concern visual localization only, not blade clearance or safety.
Job012(seed43) exited0 after661.52s, with seven native frames/four strict
candidates; all four were individually inspected and accepted, including a
closer check of the leaf-adjacent attachment. Decisions are hash-bound in
`dataset_reviews/clear_capture_20260915_orbit_v2/job_011_reviews.json` and
`job_012_reviews.json`. No source labels, native depth, splits or earlier holds
were changed. Unique candidate reviews now total430:424 accepts/six holds,
before combined per-target capping.

The NEW `collection_intake/clear_combined_20260915_checkpoint_v5` includes
job011 (job012 is reviewed separately, not yet aggregated). It validates393
selected images with393 individual accepts and zero missing reviews:
train278/50/14, validation42/8/3, test73/11/3 (images/targets/families).
All nine coverage categories still fail. It remains a draft, not a training
release; the latest verified transfer is still the369-image inspection ZIP.
Evidence:`clear_combined_checkpoint_v5_20260915.log`.

Training-only diagnostic `diagnostics/clear_sampling_training_diagnosis_20260915_v1.json`
records why seed23 is absent. Original/orbit searches rejected3048/2950
proposals for approximate projected size,374/470 for possible geometry overlap,
and33/36 for pose; only the original search admitted one geometry candidate,
which did not yield a strict clear image. Some rejected size estimates lie near
the thresholds, but those views have neither passed subsequent geometry checks
nor been rendered. This is not evidence to lower any gate. No new resolution,
optics, source-geometry, pre-screen or final-quality change was made. Native
collection, conservative launch reserves and the two bounded opposite-side
pilots remain in place. No model download, training or model-based held-out
selection has occurred.

### 2026-09-15 06:35 KST bounded followup and real-input integrity

Queued a distinct original-side proposal window (`view_offset=12`) for the four
lowest-coverage training families:jobs007/013/014/019(seed23/47/53/73). This uses
the existing `collection_plans/clear_capture_20260915_original_shard2_v1` plan,
not duplicate images or an altered split. It can start only after the current
orbit campaign completes, both opposite-side pilots finish with audited or
verified-empty outcomes, and every Kit process has exited. The queue expires
after6h; workers are serial and bounded at1800s, with no automatic retries.
All camera, native depth, geometry, quality and memory requirements remain.
No full opposite-side campaign is automatically authorized by those pilots.

Saved launcher/evidence:`diagnostics/clear_original_shard2_queue_20260915_v1.py`,
the matching`.ps1`, and`clear_original_shard2_train_queue_20260915_v1.log`.
Planned output:`collection_campaigns/clear_capture_20260915_original_shard2_train_v1`.
A separate non-approving intake waits at most7h for its request, then watches
for at most3h; output:`collection_intake/clear_capture_20260915_original_shard2_train_v1`,
log:`clear_original_shard2_train_intake_20260915_v1.log`. These are queued jobs,
not completed data or additional approved images. Orbit job013 independently
returned a verified zero-screened-viewpoint result (478.47s); the two previously
qualified lean-pilot images remain its only reviewed contribution so far.

On all278 actual training rows of the393-image checkpoint, the adapter constructed
the full848x408 RGB plus768x768 query-crop inputs. Perturbing the answer in memory
left both input images and the prompt identical; no assistant answer turn was
present. No held-out rows, Torch/Transformers, Isaac/USD or model were loaded by
this diagnostic. It does not waive draft validation or measure model performance.
Evidence:`diagnostics/clear_checkpoint_393_train_input_contract_20260915_v1.json`
and matching log. Bound dataset manifest SHA256:
`5bc2e57f66f00bdcfc358dd81a249d300a513917f3f9d63d65c4af2ce07c9b04`.

### 2026-09-15 07:00 KST native-resolution decision and review checkpoint

The user's latest decision explicitly allows a separate higher-resolution
capture set, superseding their preceding848x408-only answer. Planned native
pilot:1696x816 (same aspect ratio, fixed mounted head camera, focal length,
apertures and full greenhouse). This is new native rendering, not enlargement
of existing848x408 files, and not a claim that the physical D405 supports this
mode. Existing captures, plans and queued848x408 jobs remain unchanged.

Added the isolated`capture_sensor.py` contract and28 passing unit tests.
It validates supported dimensions, recomputes pixel intrinsics from unchanged
authored optics, preserves the rigid camera transform/metric optical-Z geometry,
and converts coordinates against explicitly declared dimensions. It does not
render, resample RGB/depth or grant training readiness. No high-resolution
capture has yet run. The existing collector, reviewers, export validators and
Qwen adapters still contain848x408 assumptions; they must be handled together
before a high-resolution training release, with separate native RGB/depth/
instance and calibration smoke tests. The running legacy collector has not
been changed to use this new helper.

`collection_intake/clear_combined_20260915_checkpoint_v6` validated397 images;
the NEW checkpoint_v7 includes orbit job017(seed67) and validates401 selected
images/all401 individual accepts, with zero missing reviews. Counts:
train286/50/14, validation42/8/3, test73/11/3. All coverage categories still fail.
Job017 provided10 native frames/seven strict candidates, all individually
accepted; three otherwise accepted views were removed by combined view capping.
Evidence:`clear_combined_checkpoint_v7_20260915.log`. The latest curation
workflow additionally refuses promotion if ANY selected image lacks its own
review, even if the general release minimum would otherwise pass.

Orbit job018(seed71) exited0 in689.42s with14 native frames/nine strict
candidates. All originals were inspected; eight accepts and one hold are saved
in`dataset_reviews/clear_capture_20260915_orbit_v2/job_018_reviews.json`.
Held`seed71_full_d0563270fb33b532fe82`: a folded foreground leaf makes the
immediate attachment uncertain despite a visible nominal segment. This new
hold joins the six preserved earlier holds; it is not overridden by numeric
visibility. Job018 is not yet in checkpoint_v7. Individual candidate-review
totals are446 images:439 accepts/seven holds before combined view capping.
The final training ZIP remains pending coverage, complete review and validation.

### 2026-09-15 07:40 KST higher-resolution pilot prepared;419 reviewed images

The user's latest authorization allows higher resolution. Added isolated
`native_sensor_payload.py`, `native_resolution_smoke.py` and
`native_greenhouse_pair.py`, with tests and
`examples/greenhouse_sim/sim_data/NATIVE_RESOLUTION.md`.
No existing production848x408 capture, source asset, split, review decision,
quality threshold, Qwen adapter or training/export contract was changed.

The new path explicitly supports native1696x816 RGB, optical-Z metres and
renderer instance IDs, validates authored/native projection and dimensions,
preserves raw invalid-depth values with a separate mask, and rejects changed
static state. NumPy integer dimensions from native annotators are supported
without accepting floating-point/configuration dimensions. Native startup
retains the existing16GiB commit/4GiB physical reserve. Blocked or failed runs
have failure receipts/nonzero exits, not fallback observations.

The paired diagnostic first runs a known-surface/occluder smoke test, then
reconstructs one completed TRAIN-family seed7/SubStem_42 snapshot and renders
both848x408 and1696x816 at the same real robot-head pose, optics, original
greenhouse population, daylight and RTPT renderer.877 real source bindings
were verified before queueing. Physical target coordinates are recomputed from
the loaded anatomy; calibration/pose/geometry/identity and2x pixel-projection
checks are required. Both resolutions retain the reference56-subframe budget.
This is static perception qualification, not hardware, contact, moving-scene
synchronization, visual-clarity or VLM-performance evidence.

Validation:84 targeted tests passed. Complete regression:
**806 tests plus47 subtests passed in48.61s**, recorded in
`data/sim_data/clear_regression_20260915_v16.log`.
An earlier test iteration exposed create-only JSON fixture writes; those were
corrected before the successful run. No native high-resolution rendering has
yet completed; the production exporter remains848x408-only.

A bounded paired pilot is WAITING, not running, behind the existing serial
orbit campaign, two opposite-side pilots and four-job original-side followup.
The queue leaves these jobs unchanged, waits at most6h, requires successful
predecessor completion and no other Kit process, rechecks memory, and starts
only one owned native child with a1800s limit. It does not retry, collect a
campaign, approve images or train.168 implementation/launcher files are pinned;
a code change before execution causes a refusal, not a mixed-version run.
Operational files:
`data/sim_data/diagnostics/native_hires_pair_queue_20260915_v1.ps1` and
matchingPython, `native_hires_pair_code_binding_20260915_v1.json`.
Queue log:`data/sim_data/native_hires_pair_queue_20260915_v1.log`.
Planned native result:`data/sim_data/diagnostics/native_hires_greenhouse_pair_20260915_v1`.
Qualification and a separately tested resolution-aware label/review/export/Qwen
profile are still required before higher-resolution images become training data.

Collection/review continued independently. Orbit job021(seed7) completed in
582.67s with one strict view; its original and query crop were individually
accepted. Job022(seed83) completed in711.13s with17 native frames/11 strict
candidates. All11 originals were inspected; the leaf-adjacent attachment in
candidate6 was also checked in its existing query crop. All11 received explicit
assistant accepts, not human/horticultural or physical-cutting approval.
Reviews:`dataset_reviews/clear_capture_20260915_orbit_v2/job_021_reviews.json`
and`job_022_reviews.json`. Total individual candidate assessments:
458 originals =451 accepts/seven holds before combined capping.

Immutable checkpoint_v8 validated410 selected images; NEW checkpoint_v9
validated**419** with ALL419 individually accepted and no missing reviews:
train304 images/51 targets/14 families; validation42/8/3; test73/11/3.
Seven prior uncertain images remain excluded before view capping. Two otherwise
accepted new views were removed by combined capping. All nine coverage gates
still fail; state remains`draft_clear_cutpoint_not_for_training`.
Evidence:`data/sim_data/clear_combined_checkpoint_v9_20260915.log`;
review:`data/sim_data/collection_intake/clear_combined_20260915_checkpoint_v9/review/index.html`.
No final training ZIP, training run or model download is claimed. The prior
369-image inspection-only ZIP remains separate from training readiness.

### 2026-09-15 07:50 KST confirmed20k TRAINING requirement and generator assessment

User confirmed **at least20,000 training images plus separate held-out sets**.
The exact held-out sizes were not specified;2,000 validation/2,000 test is a
working proposal. Original counts/splits/review decisions and the small
`clear_cutpoint_v1` qualification gates remain unchanged. That old qualification
release must not be presented as fulfillment of this new20k milestone.

Added read-only`clear_scale_capacity.py` and21 tests. It checks recorded target
identities/counts and computes capacity without importing Isaac, changing
collection or approving data. The confirmed-goal report is NEW
`data/sim_data/diagnostics/clear_20k_capacity_20260915_v2.json`; v1 preserves the
earlier pre-clarification scenarios. Source plan SHA256:
`0fb6409073ada123e52af73620886d622a2f63f493a04d34962b999b98957f21`.
These are recorded-plan counts, not a fresh native visibility/geometry audit.

The active schedule has297 targets across24 families, at most3,564 images under
the12-view cap. Training alone:198 scheduled targets/max2,376 images.
All recorded anatomical candidates:870 overall/max10,440 images;584 in training/
max7,008 images. The latter includes out-of-band/unverified candidates and is an
optimistic upper bound.20k training needs at least1,667 training target geometries,
i.e.1,083 beyond all recorded training-family candidates, before rejections.
Higher resolution, repeated shards, crop copies or new IDs for cloned branches
do not remove this diversity limit.

User asked whether we can create a generator because the original is unavailable.
Inspected the package README, seed101 component manifest, `candidate_branches.py`,
`audit.py`, `cut_regions.py`, collection/split code and H200 recommendations.
The assets include meshes/materials, topology, attachments, axes and capsule
centerlines/radii; manifests identify generator1.1.0 and some physical constants,
but not its full morphology/growth algorithm. Existing branch-preview code is
limited to three recipes on each of two plants. A new parametric component
generator is feasible in principle, but not already implemented or botanically
validated. The current manifest audit accepts translation only, so generated
mesh transforms and anatomical metadata must be updated together.

Design/acceptance plan:`examples/greenhouse_sim/sim_data/CLEAR_SCALE_20K.md`.
Proposed20-layout/100?200-native-view pilot before scaling. New junction geometry,
consistent rendered/label geometry, protected structures, donor lineage,
morphology deduplication, immutable original splits, and full-greenhouse native
sensing are required. Generated-layout counts are separate from original donor
families; no claim that random seeds create independent real plants. Source
distribution fitting should use training assets, with original held-outs retained
as a separate transfer check. A larger review policy and20k export profile must
be explicit; no automatic acceptance of unseen images or gate weakening.

Validation:`data/sim_data/clear_regression_20260915_v17.log` records
**827 tests plus47 subtests passed in49.52s**.
No generator assets, new collection campaign, model download or training was
started in this assessment. Existing serial collection advanced to held-out
job004; the native higher-resolution pair remains queued behind existing jobs.
The most recent individually reviewed/validated checkpoint remains419 images
(304 train/42 validation/73 test), still a draft. Code additions do not alter the
168 existing code hashes pinned by that native pilot.

### 2026-09-15 component-generator CPU pilot (not a training release)

Implemented the user's requested first generator prototype on koh-dev/sim-data:
`plant_variants.py`, `plant_variant_usd.py`, and
`plant_variant_catalogue.py`, with their regression tests.
Details/commands: `examples/greenhouse_sim/sim_data/PLANT_GENERATOR.md`.
This is a bounded, source-derived petiole similarity generator, not a recovered
original growth algorithm or a new physics implementation.

Measured envelope uses only the16 frozen training donors/584 candidate
centerlines. The engineering grid varies petiole/leaf subtree azimuth, tilt and
uniform scale within the observed training ranges. It preserves main stems,
protected organ geometry, source assets and frozen reservations. Mesh points,
normals/extents and anatomical origins/axes/centerlines/radii move together.
Nominal10mm/10?20mm labels are resampled on the new curve, never inherited
pixel annotations. Copied static assets deliberately omit donor physics
constants/APIs; new dynamics are NOT certified by this generator.

Completed copied-asset pilot:
`data/sim_data/generated_plants/petiole_similarity_pilot_20260915_v2`.
20 layouts,16 donor families,160 transformed target instances,156 exact shape
hashes,138 original donor targets;665,859,812 bytes. Four exact repeats are not
new geometry. None are new independently sourced families; no novel-target
approval, native image capture, image review or training eligibility is claimed.
The conservative12-view group remains the ORIGINAL donor target.

All160 targets passed24 sparse radial mesh rays at10/15/20mm. The generated
catalogue and existing world-label projection worked in anonymous CPU USD
stages with8,496 component instances and maximum translation error1.15e-16m.
These are measured CPU checks, not native visual or collision validation.
Actual source-export edge cases were found and tested: zero normals and
faceless already-deleafed stubs are preserved/reported, not replaced with
invented geometry. Failed intermediate attempts remain separate diagnostic
directories; none are accepted data.

The plant-wide bounds audit found75 parent-attachment warnings. Comparison
with all16 original families found ZERO newly flagged components. Two selected
targets inherit such warnings: seed71/SubStem_48 and seed89/SubStem_48.
The inspection adapter withholds them, leaving158 candidates for further
inspection. The warnings are point-to-AABB distances, not proven surface gaps.
No existing dataset review decision was changed.

Evidence:
`diagnostics/plant_variant_pilot_inspection_20260915_v1.json`,
`diagnostics/plant_variant_attachment_comparison_20260915_v1.json`, and
`diagnostics/plant_variant_pilot_inspection_20260915_v2.json`, all under
`data/sim_data`. Reproducible commands are in PLANT_GENERATOR.md.
The legacy plan binds USD/JSON but not texture files: new copied texture bytes
are independently verified against their generator-copy-time donor hashes.
We do not retroactively claim the legacy plan froze textures.

The first complete data regression after the adapter passed856 tests plus77
subtests in69.29s (`clear_regression_20260915_v19.log`). The final regression
including the inherited-attachment hold passed857 tests plus77 subtests in68.93s:
`clear_regression_20260915_v20.log`. The v2 inspection completed all20 layouts
with158 inspection rows and two held targets.

Remaining before20k: qualify native higher-resolution camera/export contracts;
add the explicit generated-catalogue full-greenhouse capture path; inspect
100?200 native pilot images; validate local attachments/intersections and
meaningful geometry novelty/near-duplicates. Then choose generator expansion,
review policy and20k release profile based on measured yield. Mere new seeds,
copied crops or repeated source targets do not satisfy20k diversity.
No model weights were downloaded, training was not started, and no final
training ZIP was produced. Existing native collection/jobs and168 pinned native
code hashes were unchanged. Last reviewed image checkpoint remains419 total
(304/42/73 train/validation/test), not20,000 training images.

### 2026-09-15 generated-plant native adapter and output preview

Added generated_capture.py / native_generated_pair.py and regression tests.
The fixed two-frame diagnostic reconstructs the full existing greenhouse and
recorded robot/head-camera pose, then substitutes one generated plant at the
same world transform. It requires completed native high-resolution sensor
qualification; no source plan, family split, review decision or production
exporter is changed. Native optical-Z remains Isaac-derived. Scene/instance/
buffer freshness, camera, geometry, provenance and memory checks are enforced;
no training approval or physical qualification is granted.

CPU rehearsal on seed7/SubStem_42 preserved421 components and measured a
2.087127mm cut-label change /4.662410px at1696x816. Both are in frame; this is
not rendered-visibility evidence. Artifact under data/sim_data/diagnostics:
generated_native_pair_cpu_rehearsal_20260915_v1.json.

The user requested to see generator output. Added plant_variant_preview.py:
actual USD triangles with shared orthographic views and diagnostic colors,
no textures or native sensor outputs. Original/generated each497,579 triangles
and421 components, with8 selected transformed subtrees. Visually inspected
the comparison and junction close-ups; this does NOT approve training labels,
physical safety or greenhouse rendering. Preview under data/sim_data/diagnostics:
generator_visual_preview_20260915_v1/plant_comparison.png.
Orange attachment, white nominal10mm point, magenta10-20mm arc interval.

Bounded generated pair queued behind existing collection and camera diagnostic.
Independent186-file code/plan pin; all168 existing native code pins unchanged.
Six-hour wait deadline,1800s owned-worker budget, no competing-renderer startup,
automatic retry, model download or training. Queue log:
data/sim_data/generated_native_pair_queue_20260915_v1.log.
At checkpoint it is WAITING, not captured or approved.

Details:examples/greenhouse_sim/sim_data/GENERATED_NATIVE_CAPTURE.md.
Data regression before preview:913 tests plus77 subtests in65.94s
(clear_regression_20260915_v21.log); preview geometry tests:5passed.
Latest individually reviewed checkpoint remains419 total304/42/73.
New generator assets/previews/queued frames are not20k training data. Existing
native collection progressed, but its new frames were not individually reviewed
during this integration step.
Final full regression including the preview:918 tests plus77 subtests passed
in67.43s (data/sim_data/clear_regression_20260915_v22.log).

### 2026-09-15 reviewed checkpoint v10 and resumed native generation

Latest immutable reviewed draft:
data/sim_data/collection_intake/clear_combined_20260915_checkpoint_v10.
449 selected images:311 train /48 validation /90 test,72 original targets and
20 donor families. All449 individually assistant-reviewed;14 negatives excluded.
This is not independent human validation. All small coverage gates remain
incomplete, and this is far below20,000 accepted TRAINING images plus held-outs.
No training ZIP or local model/training launch is claimed.

New wave3 reviewed76 original captures:69 accepts,7 holds for ambiguous branch
crossings, hidden attachments or fruit/petiole blending. Decisions are hash-bound
under data/sim_data/dataset_reviews/clear_orbit_wave3_20260915_v1.
Net checkpoint increase30, not69, after preserved negatives and the original
12-view cap. Old checkpoints/splits/decisions are untouched. Clean images remain
native848x408; depth sidecars are byte-identical native Isaac optical-Z.

Added clear_input_audit.py and8 regression cases. On all449 draft records,
actual training/inference image+prompt construction matches in full-RGB and
full-RGB-plus-query-crop modes. Normalized-coordinate roundtrip maximum error
0.00416000000007px. The checker does not run a model/HF processor, approve data,
change the training loader or feed depth/GT overlays to the RGB model.
Data regression926 tests plus77 subtests in65.83s:
data/sim_data/clear_regression_20260915_v23.log.

Native supplemental v1 failed before images on the Windows commit-memory gate.
After the user freed memory,22.38GiB headroom passed the recovery reserve and
the two frozen native diagnostics completed. Camera848/1696 pair exit0 in
150.406s; original/generated native1696 pair exit0 in145.235s. Same mounted
camera/optics, robot pose, source greenhouse, lighting and source files verified.
Native interval projection grows15.47 to30.94px. Generated interval31.54px,
all11 sampled points match native target identity/depth. The recomputed marker
remains on the visible petiole in visual review. No physical-cut/biology claim.

Native outputs under data/sim_data/diagnostics:
native_hires_greenhouse_pair_20260915_v1 and generated_native_pair_20260915_v1.
Post-capture QA, four individually attributed diagnostic visual decisions and
native-depth heatmaps: native_camera_generated_review_20260915_v1.
These four diagnostics are not extra accepted training images or new families.
Resolution-aware query labels, general sampling/export and novelty validation
remain necessary before generated high-resolution images enter training.

Resumed exactly the interrupted four TRAIN-family jobs in NEW
collection_campaigns/clear_capture_20260915_original_shard2_train_v2 under
data/sim_data; native job007/seed23 started10:23 KST. Same frozen plan,1800s
worker limits and memory/geometry/visibility gates. Only the exact stranded
assistant-owned dependency helper was stopped; no user apps or datasets removed.
The non-approving intake prepares completed native captures for later review:
data/sim_data/collection_intake/clear_original_shard2_recovery_20260915_v2.
Details and evidence:examples/greenhouse_sim/sim_data/CLEAR_CHECKPOINT_20260915.md.

### 2026-09-15 native high-resolution annotation path and checkpoint v11

Current reviewed draft: data/sim_data/collection_intake/clear_combined_20260915_checkpoint_v11.
450 selected native848 images:312 train/48 validation/90 test,73 original
targets across21 donor families. All450 have explicit individual assistant
reviews;16 negative decisions remain excluded. Small coverage and20k coverage
are still incomplete; no final training archive or local training is claimed.

The resumed shard produced3 seed23 images (newly represented training family);
one was accepted and two held for confusing query crossings after full-RGB and
lossless-crop review. Seed47/53 yielded no screened views. Memory then stopped
seed73 before launch; it was resumed separately with recovered reserve and
yielded no screened views. Failed/empty attempts remain unchanged, not counted.
The v11 builder consumes only the completed, independently hash-bound seed23
job and keeps the stopped campaign's overall status honest.

Added separate native_query_visibility.py, native_clear_contract.py,
native_clear_labels.py and native_clear_annotation.py with28 focused tests.
They support native1696 RGB, a query-only native768 square crop with no resizing,
recomputed visible distal query and anatomical10mm/10-20mm labels, exact native
petiole/parent masks and optical-Z, and paired RGB/prompt/normalized-coordinate
construction. Legacy848 task contracts, generator assets, frozen splits, prior
reviews and production training loader are untouched.

Actual pilot: data/sim_data/diagnostics/native_clear_annotation_20260915_v1.
Both original/generated records passed numeric checks and individual assistant
query/cut annotation inspection. The normalized generated example is query
(975.47,255.51), answer cut_point_uv[824.42,288.54], localized/clear/inspect_cut_region.
The model input contains clean RGB and query only, with optional native crop;
depth/geometry/GT overlays remain sidecars, not model input. The native query
and cut are recomputed from changed geometry, not copied source pixel labels.
Review: data/sim_data/dataset_reviews/native_clear_annotation_20260915_v1.
These are annotation-pilot approvals, NOT training-release accepts or new donor
families. Native-aware H200 loading/evaluation/processor testing remains undone.

Broader native pilot: diagnostics/native_generated_broader_pilot_20260915_v1
under data/sim_data. Three additional TRAIN families, fixed reviewed source
views, one renderer,1800s worker budgets, memory guards and202 bound files.
Seed101 camera test passed; generated substitution failed robot/environment
clearance and was held without weakening the check. Only unstarted seed103/17
cases continued separately; their outcomes are pending at this checkpoint.
No generated pilot frame joins the450-image dataset automatically.

Full data regression954 tests plus77 subtests passed in73.10s:
data/sim_data/clear_regression_20260915_v24.log.
The actual450-record legacy input audit also passed in both image modes:
data/sim_data/diagnostics/clear_checkpoint_v11_input_audit_20260915.json.
Detailed contracts, commands and limits:
examples/greenhouse_sim/sim_data/NATIVE_CLEAR_ANNOTATION.md.

Follow-up before annotation-code commit: seed103 also failed at the generated
clearance screen and seed17 remains unstarted. Inspection found the detailed
triangle refiner is restricted to /World/PackPlants; generated plants under
/World/GeneratedNativePilot get only conservative enclosing-box tests.
This integration gap will be fixed and tested explicitly; failed cases remain
held until fresh native evidence exists. No currently running renderer remains.

### 2026-09-15 generated foliage capture fix verified

Code992b99b fixes the generated plant root's accidental AABB-only fallback.
It explicitly applies the existing mesh screen, preserving the10mm margin,
true-intersection rejection, default legacy scope and conservative fallbacks.
Full suite968 tests +77 subtests passed in83.92s:
data/sim_data/clear_regression_20260915_v25.log.

Fresh native output:
data/sim_data/diagnostics/generated_refinement_native_20260915_v1.
Seed101/103/17 all exit0 in176.954/140.047/141.765s for the original/generated
pair workers. Generated mesh screening clears261/172/285 coarse pairs; no
remaining overlaps. Both prior generated-stage failures are resolved in NEW
attempts; failed historical runs, source assets, plans and reviews remain intact.

Six new1696x816 annotation candidates passed numeric checks and individual
assistant visual inspection. Exact native depth/identity agrees at all11 cut
interval points per frame. Reviews and hashes:
data/sim_data/dataset_reviews/generated_refinement_native_20260915_v1.
Full-frame overviews and clean lossless native crops were inspected; seed17
has more clutter/darker query context. All decisions are annotation-pilot-only,
not training-release, blind evaluation, human/botanical or physical-cut approval.

With the earlier seed7 pair:8 annotation-pilot frames from4 donors/4 generated
layouts. Repeated controls are not new target diversity. Legacy450 reviewed
images (312/48/90), frozen splits and donor view caps remain unchanged. Native
depth is unreconstructed. No model/training or final20k ZIP. All workers exited.
Broader bounded capture and native-resolution export/loader integration remain.

### 2026-09-15 reusable native inspection campaign and seven-family expansion

Added native_generated_campaign.py with20 CPU regression cases; full suite
988 tests +77 subtests passed in63.28s (clear_regression_20260915_v27.log
under data/sim_data). Serial original/native-camera/generated/annotation phases,
exact exit/failure evidence, immutable plans/code and no automatic approvals.
Admission logging records memory/disk even when no renderer starts.

Inventory matched62 reviewed reference views/23 targets/11 TRAIN donors.
Seven additional donors selected: seed19/41/43/67/71/83/89, one source view each.
No held-out donor or existing review/split/view cap was changed. These are
planned new captures, not62 new images or an unbiased sampling-yield estimate.

The first seven-family attempt stopped before any renderer/image because commit
headroom dropped below18GiB. Preserved:
data/sim_data/diagnostics/native_generated_seven_family_20260915_v1.
Fresh v2 holds the same source geometry/views, new local qualification outputs
and explicit code bindings. A bounded2h readiness queue waits for20GiB commit
and no other Kit process; it launches once and retains all worker guards.
data/sim_data/diagnostics/native_generated_seven_family_20260915_v2.
Queue: data/sim_data/native_generated_seven_family_memory_queue_20260915_v2.log.
No user applications were closed or system settings changed. Check receipts;
waiting/queued is not native completion. Data count stays450 legacy-reviewed
plus8 separate annotation-pilot frames pending release integration.
See examples/greenhouse_sim/sim_data/NATIVE_GENERATED_CAMPAIGN.md.

### 2026-09-15 15:34 KST autonomous review and native capture checkpoint

User authorized autonomous review, asking only for genuinely unresolved
decisions. Added automated_native_review.py: frozen source/implementation
bindings, actual native RGB/Z/camera fingerprints, exact target-mask checks,
recomputed anatomical cut labels and continuous query-to-cut chain verification.
Projected probes <=0.5 pixel /0.5mm, chain vertices preserved, no gap filling.
Clean native crop/prompt/answer parity checked. No computed replacement depth,
hidden-coordinate execution, source/review/split mutation or training approval.
Automatic and actually performed assistant visual reviews remain separate.
30 new adversarial tests; full suite1018 tests +77 subtests passed in65.91s:
data/sim_data/clear_regression_20260915_v29.log.

Seven-family v2 completed with queue exit0.14 native original/generated frames,
13 clear candidates. Final automatic review:4 accepts,9 visual holds,1 reject;
zero integrity holds. All14 actually visually inspected (full-scene overview
and lossless native query/junction crops).12 annotation-pilot accepts,2 excluded
from the easiest task. Seed71 generated cut was leaf-occluded; seed41 generated
query had a distracting crossing despite a visible cut. No unresolved user
decision and no prior review altered. Exact hashes/reasons/crop scopes:
data/sim_data/dataset_reviews/native_generated_seven_family_20260915_v2/assistant_visual_reviews.json.
Automatic receipts:automatic_first_six_v2.json and automatic_final_pair_v2.json.
This selected small pilot does not measure population yield or VLM accuracy.

Counts: legacy450 reviewed (312 train/48 validation/90 test), unchanged.
Separately20 native annotation-pilot accepts including the earlier8; these are
NOT20k training images or final release approvals. Original donor view caps
still apply. Similarity variants do not create independent botanical families.
Still needed: qualified larger target/morphology diversity, scalable native
production collection, native-resolution export/H200 processor qualification,
held-out release validation, and only then final20k+held-out ZIP.

Memory assistance: after explicit approval and saved reviews, stopped ONLY
legacy review server PIDs13524/22728/11076 on ports8877/8878/8879, after checking
their module/port ownership. All exited, releasing their measured2,480,340,992
private bytes (~2.31GiB); saved annotations untouched. This is process-private
commit released, not a measured equal increase in concurrent whole-system
headroom. The earlier22.741GiB queue admission occurred before these stops.
No other apps/VMs, Parsec, drivers or services were stopped; no reboot/pagefile
changes. Large kernel paged-pool allocation remains unresolved, not fixed by
closing these servers. CBnb/cbfltfs4.sys remains an investigation lead, not a
proven allocation owner. No driver-unload attempt was made.

Next plan validated:12 additional original targets from the frozen inventory,
one best existing reference per target; no previously qualified target repeated.
Native1696x816 original/generated pairs; at most24 annotation candidates plus
non-training camera/sensor controls. New root:
data/sim_data/diagnostics/native_generated_remaining_targets_20260915_v1.
Readiness queue launched15:33:47KST (PowerShell PID121832), waits at most2h for
20GiB commit/no Kit, then one serial campaign with unchanged worker guards.
Automatically reviews all completed cases after the attempt; no auto-retry or
claimed visual inspection. Log:
data/sim_data/native_generated_remaining_targets_memory_queue_20260915_v1.log.
Native log:data/sim_data/native_generated_remaining_targets_20260915_v1.log.
Check queue/result receipts; launch/queued is not completed or accepted data.
See examples/greenhouse_sim/sim_data/AUTOMATED_NATIVE_REVIEW.md.

### 2026-09-15 16:43 KST curved-plant generator checkpoint

Implemented source-derived curved/relocated petioles with common-field leaf
deformation, preserved detailed meshes/UV/materials, TRAIN-only envelope,
actual parent-surface attachment and recomputed10 mm /10?20 mm cut geometry.
Read-only catalogue replays recipes and verifies serialized points/normals,
topology, textures, metadata, source/split/code hashes and attachment bounds.
Real seed101/SubStem41 variant retains404 components and passes CPU catalogue
checks with no selected-subtree attachment warnings. Native rendering is still
pending; this is static geometry, not validated plant dynamics or botanical growth.
Details: examples/greenhouse_sim/sim_data/CURVED_PLANT_GENERATOR.md.

Full regression1057 tests +87 subtests passed in67.68s (v31 log). The previous
v30 failures were two campaign unit fixtures missing qualification.json; corrected
the fixtures, not the production validation. Command-approval service capacity
and local ACL failures interrupted work; code/data were preserved, not bypassed.

Additional old-similarity native batches: remaining_targets_v1 completed2 cases,
then held seed103/SubStem47 for generated geometry within robot-torso clearance.
unattempted_targets_v2 completed4 cases, then held seed41/SubStem44 because a
generated leaf entered head/neck clearance. Failed attempts preserved.12 new
frames from6 complete pairs have automatic reviews (2 automatic annotation
accepts,9 visual holds,1 clear-task reject); individual visual adjudication still
pending. They do not add to the previous20 reviewed native-pilot accepts.

Legacy450 reviewed (312train/48validation/90test) unchanged. No20k release orZIP.
New curvature is not a new donor family or automatic independent-target approval;
original caps remain until explicit global geometry-novelty admission is qualified.

First curved native plan ready under diagnostics/curved_native_20260915_v1.
Hidden readiness queue PID119832 launched16:42:29 KST; at most2h waiting for20GiB
commit/no other Kit, then one native original/generated pair followed by CPU
annotation/review on success. Existing mounted camera/optics, full greenhouse,
native1696x816 RGB and Isaac optical-Z preserved. Check result/exit receipts;
queued is not executed or accepted data. No user apps or settings changed.

### 2026-09-15 native curved capture and review checkpoint

Two source-derived curved donor pairs captured successfully with native1696x816
robot-head RGB/Isaac optical-Z and full greenhouse retained. First seed101 pair
visually checked; seed19 pair automatically checked, visual review pending.
Two other generated recipes rejected for folded leaf geometry; no gate relaxed.
See sim_data/CURVED_PLANT_GENERATOR.md under examples/greenhouse_sim for receipts.

Fixed curve interval sampling to preserve capture's anatomical knots; new failure
tests reject mismatched/stale/occluded intervals. Added explicit append-only code
requalification requiring exact saved-label equality and all old source bindings.
18 pairs/36 raw frames:11 automatic accepts,23 visual holds,2 rejects,0 integrity
holds,0 final train approvals. Prior review decisions remain immutable.
Actual partial visual review adds4 pilot candidates,1 excluded target and1
repeated control (not diversity). Native pilot accepted count24; legacy450 draft
(312train/48validation/90test) unchanged. Eight old frames and the newest pair
still require actual visual review. No20k release, ZIP or local model download.

Full regression1075 tests +87 subtests passed (v34,58.49 s). Added capture timing
without quality/gate changes: ~14 s geometry screen and30-34 s warmups dominate
each frame; pair137.90 s plus annotation. Production amortization and global
geometry novelty admission still required; original source-target12-view caps
remain. Next: rigid leaf transport, broader qualified shape diversity and a
measured fidelity-preserving persistent native collector.

### 2026-09-15 rigid-leaf and native performance checkpoint

Explicit V2 generator rigidly transports detailed leaf blades at their curved
attachments. All3 former donor recipes CPU-qualified, including the2 previously
folded examples. V1 receipt implementation left unchanged. Native2 pairs completed,
4frames/3clear candidates; third rejected zero visible target pixels (robot
clearance passed). Rigid-leaf pilot images still require actual visual review.
Root: data/sim_data/diagnostics/curved_rigid_leaf_native_20260915_v1.

Actually completed remaining8 previous frame reviews:7accepts/1exclusion. Second
V1 curved pair:1generated accept/1repeated control. Native pilot accepted32;
legacy450 draft unchanged, no20k final release or ZIP. Hash-bound visual receipts
listed in examples/greenhouse_sim/sim_data/AUTOMATED_NATIVE_REVIEW.md.

Exact static-obstacle cache: native cold13.55-13.66s to warm0.861-0.864s, all
screen fields exactly equal reference, invalidation verified after substitution.
~16x for repeated geometry screening only, NOT total capture. Full native scene,
1696x816 mounted-head camera, native optical-Z and56 requested subframes retained.
Opt-in profiling implemented; persistent production integration/moving-robot
native view qualification/render convergence still pending. Details:
examples/greenhouse_sim/sim_data/STATIC_GEOMETRY_CACHE.md. Full1107 tests +87
subtests passed (v36,57.81s). No source/split/review mutation or model download.

Next collection must recompute admissible actual robot viewpoints for changed
geometry, not assume an old camera pose still sees the target. Global shape
admission remains required before resetting any original source-target cap.

### 2026-09-15 native multiview checkpoint

Full-greenhouse persistent capture now reuses the exact obstacle cache, native
writer and render product across bounded actual robot-head snapshots. Source
heading/arms/torso/mount preserved; floor, joint limits, camera calibration and
geometry rechecked. Failed whole-body rotations retained, not bypassed.
Seed17:5captured/5eligible/4automatic; seed19:4/4/3. All9 actually visually
reviewed; two contrast holds resolved with visible continuous target shafts.
Native annotation-pilot candidates41 separately; legacy450 draft unchanged.
Global duplicate/cap/release admission still pending. No20k release or ZIP.
See examples/greenhouse_sim/sim_data/NATIVE_MULTIVIEW.md for receipts and scope.
Runs209.30/179.80s, additional views~32s, unchanged56 requested subframes.
Full1132 tests +87 subtests passed (v39,57.93s). Consolidated native render
qualification and larger genuinely diverse multi-target collection remain next.
Lossless storage prototypes preserve exact original native optical-Z bytes;
three-frame22k size projection83-112GiB excludes final ZIP/assets, not measured
release size. No depth reconstruction, source edits or training performed.

### 2026-09-15 multi-target and throughput qualification

Implemented shared-scene multi-target plans/capture, keeping native56-subframe
quality and all per-pose checks. First launch exposed early USD import/ABI
conflict; fixed hash-only preflight followed by mandatory geometry replay after
SimulationApp. Failed run preserved. Two new plants each have4 CPU-qualified
targets; corrected12-view/donor native batches are pending completion/review.
Fast native instance adapter now passes exact same-callback1696x816 ID/prim
equivalence. Fast-only later view32.88s vs legacy32.91s: no useful steady-state
speedup demonstrated. Consolidated56 also failed to provide a useful gain.
See examples/greenhouse_sim/sim_data/NATIVE_CAPTURE_SCALING.md for all evidence,
negative results, native short-budget qualification still required, context-aware
geometry-diversity sensitivity and lossless storage measurements.
1176tests +87subtests passed (v41,70.33s).41 accepted native annotation-pilot
candidates and legacy450 draft remain separate; diagnostic rerenders do not
increase diversity. No20k final release/ZIP, split edits or training.

Corrected multi-target native runs completed: seed17 7captured/7eligible;
seed19 7/4. Both~284s inside collector, all full56-subframe quality retained.
All7 seed17 images actually visually reviewed (including5 contrast holds);
native annotation-pilot total48, not final release approvals. Seed19 pending
visual review. Added explicitly unqualified warm56_then8_trial for native
matched-budget testing; reference56 stays default. Trial is not accepted20k data.
### 2026-09-15 evening native throughput and diversity checkpoint

Matched1696x816 short-budget pilot:14 frames/two donors. Exact poses/optics,
native depth (0.2mm), target masks, visibility eligibility and cut answers agree.
Median subsequent capture33.07->5.93s; complete batches284s->119-123s.
RGB shading and3 query pixels changed: not photometric equivalence. Five short
overviews/native crops inspected. Reference56 remains default; wider trial QA
continues, and diagnostic repeated frames do not count toward20k.

Native full-budget visual-pilot52 accepted candidates pending global admission;
legacy450 draft unchanged. New nine-donor detailed-plant campaign:
data/sim_data/collection_batches/native_diverse_multitarget_20260915_v2.
First new plant16captured/15eligible/7automatic, not yet individually reviewed.
Only completed receipts count; numbered progress snapshots are append-only.
One progress-journal failure and one code-binding-change refusal preserved.
Keep implementation frozen while native jobs execute; retry rejected job later.

Tested context-aware morphology metrics now prevent rename/rigid/uniform-scale
variants from creating novelty and include leaf/parent context.784 replayed TRAIN
recipes remain sensitivity evidence, not automatic new donor families/cap resets.
Exact native NPY codec tested9real arrays:49.8MB->7.8MB with all file/NaN bits
preserved; original native depth/RGB untouched. Capture/export storage migration
and new release novelty policy are still pending.
Full1201tests +128subtests passed(v44,64.70s).
See sim_data/NATIVE_CAPTURE_SCALING.md under examples/greenhouse_sim for reports,
failures, limits and remaining20,000 TRAIN +held-out release requirements.

### 2026-09-15 broader yield and next sampling experiment

Nine-donor campaign finished99 native captures:20 strict automatic candidates,
22 clarity holds and57 excluded. Labels and query traces were recomputed from
actual saved native depth/identity buffers and matched exactly. Review receipts:
data/sim_data/dataset_reviews/native_diverse_automatic_20260915_v1.
Four new actual full-view/native-crop spotchecks accepted; native visual-pilot56
pending caps. Those four overlap the automatic pool and must not be double-counted.

The4cm-only camera-neighbour sampler has poor yield, including a generated plant
with zero collision-screened poses. Added a separate bounded24cm-outward/8cm-lateral
actual-robot camera policy, keeping original heading, limbs, mount and all gates.
It does not change the old sampler or move a physical robot. Native validation is
queued. See examples/greenhouse_sim/sim_data/native_capture_v2/README.md.
Full1209tests+136subtests pass. Native1-subframe probe is also isolated; no default
renderer/budget change, final20k release, held-out recapture or training claim.

### 2026-09-15 late-evening native data scaling checkpoint

The retry campaign completed38 additional native frames:14 strict automatic
candidates,12 clarity holds,12 exclusions. The broader99+retry38 totals137
frames/34 strict candidates before global admission, not approved20k data.
The wider seed19 camera search recovered7frames from a zero-yield baseline:
2strict/2holds/3excluded. Four actual full-view/native-crop visual reviews keep
the same2 accepts/2 holds. Three later wider-seed103 spotchecks also accepted;
the separate native visual-pilot count is61 pending global caps. Visual counts
overlap automatic counts and must not be added to them.

Wider campaign: collection_batches/native_wider_campaign_20260915_v1 under
data/sim_data. Its first six completed plants produced209frames/37strict
candidates; the batch continues. One old seed17 frozen-plan replay refused,
preserved without capture. An18-new-plant campaign is queued after a native
compact-storage qualification, serially with20GiB launch reserve. Completion
requires result.json and no failure.json; process exit0 alone is insufficient.

One- and two-subframe trials both failed the authored/native camera-pose
synchronization check. Neither is adopted. Eight-subframe trial remains
explicit; full56 reference default, greenhouse contents and materials unchanged.

Implemented native_dataset audit, exact compact storage/direct writer and
augmentation-aware candidate accounting in examples/greenhouse_sim/sim_data.
Independent review found/fixed request binding, duplicate accounting, source
lineage, raw/compact trace parity, missing observations and bounded-read gaps.
All1346 tests+181subtests pass(v46,99.15s). Direct writer preserves21 actual
native NPYs and28 PNGs across7 existing frames byte-for-byte;126 source files
unchanged;162.8MB->38.9MB. No diagnostic copies count as new data. Native compact
renderer integration is queued separately and must pass before promotion.

Current useful yield implies multi-day brute-force collection, not an honest
overnight20k promise. Prior137frames/34strict required1550.08 collector seconds;
linear extrapolation is253.3hours before preparation, review and duplicate losses.
Prioritize a broader screened reference bank and pre-render geometry/legibility
ranking, then stage reuse. Keep native depth/identity checks authoritative.
Seven additional TRAIN families exist outside the current nine-donor references;
seed101 andseed7 already have native sensor prerequisites, four need qualification,
andseed73 lacks screened viewpoints. Do not relabel source plans to add them.

Still outstanding: calibrated global augmentation/image admission, native
held-out capture, portable high-resolution Qwen export/processor validation,
20,000 accepted TRAIN+additional held-outs and final ZIP. No local training or
model download. See sim_data/native_dataset/README.md and NATIVE_CAPTURE_SCALING.md.

### 2026-09-15 reviewed-reference collection continuation

Wider campaign completed247 captures/52 strict automatic candidates. The current
18-plant generated campaign's first15 jobs yielded174 captures/53 strict;
three geometry-invalid generation attempts contributed zero images. Native
annotation replay runs alongside collection. Four new actual full-view/native-
crop visual inspections accepted their petiole associations and cut intervals,
bringing the separate overlapping visual pilot to65 pending global admission.
Receipt: dataset_reviews/native_diverse_generated_visual_20260915_v1/assistant_visual.json.

Added native_capture_v3 reviewed-reference bank, fresh camera qualification
schedule, same-group generated-job preparation and conservative projection
preselection. Bank312 references/52 targets/15 TRAIN donors/4 original plans/
26 groups, verified against17,089 bound files. These are historical848 pose
priors, NOT new native training images; all strict-native-trace evidence is
unknown. The chosen schedule has15 groups/51 targets and15 primary+25 conditional
fallback fresh camera pairs; controls never count toward20k. Its bounded next
campaign preserves source plan/group identity, defaults to <=3 changed targets
and6 native views/target, and retains all independent native overlap/visibility/
depth/trace checks. Legacy active jobs and implementation bindings are unchanged.

Projection retrospective416 frames/99 strict:16 non-strict rejected, zero strict
rejected;0.244s filter cost and91.38s historical rejected-frame cost. This is not
an integrated speed result. Full v49 regression1536+181subtests passes in128.88s
(separately in-progress inventory tests excluded). New launch gates check
the exact reviewed anchor, target bases, native proof and hashes after resource
waits. Generated outputs cannot be written inside original asset/plan directories.

Native compact diagnostic7 frames preserves depth bits, target masks and all
annotation decisions.16,964 identity differences are sibling gutter-hanger
cylinder swaps; none on targets/parents/selected query trace. Six pixels affect
unselected query contrast contexts. All42 codec roundtrips pass, but matched-run
comparison is not independent same-callback proof. Do NOT promote compact yet;
continue raw collection. Receipt: diagnostics/native_identity_raw_compact_seed19_20260915_v1.json.

20,000 accepted TRAIN + held-out sets remains the completion criterion. Global
duplicate/augmentation admission, held-out native capture and validated portable
high-resolution Qwen export remain required. No model download/training or final
ZIP was started. See native_capture_v3/README.md for entry points and scope.

Continuation checkpoint:18-plant campaign completed212 native frames/65 strict
automatic candidates;15 native jobs and3 rejected CPU geometry attempts. Four
additional full-scene/native-crop reviews accepted annotation-pilot associations
(overlapping visual total69, not69 independent new captures). Reviewed-reference
campaign first fresh seed101 camera pair and two generated jobs passed native
capture+audit:19 frames/8 strict. All remain subject to global admission.

Added native_dataset/near_image.py (42 focused tests): native full-frame plus
129x129 nominal-cut-context measurements, no resizing or depth calculation.
Actual controls21 duplicate rerenders versus1 visually changed petiole separate
in this small measurement set (max duplicate0.039133, changed0.347364 normalized
MAE); this is NOT threshold calibration across donors. No threshold is selected,
no approval assigned. Artifact:diagnostics/native_rgb_controls_20260915_v1.json.
Next:actual saved-mesh descriptors, full observed inventory, calibrated global
admission, continued native collection, held-out capture and portable export.

2026-09-16:implemented actual saved-mesh V2 morphology adapter with28tests.
Four completed native plans produced28descriptor rows for11 original targets,
3donors; repeated originals/replays exact. These are measurements, not28new
admitted contexts. Nuisance maximum1.0005e-8 exceeds the old1e-9 numeric test in
25/84 comparisons; kept as an explicit calibration limitation. Existing V1
seed19 visual positive is unsupported by this V2 domain, so no joint calibration
claim. See native_dataset/README.md and morphology_actual_output_20260915_v1.

Storage scale:driveD observed158GiB free, insufficient for20k plus all uncompressed
native sidecars. Added dual_storage same-callback comparison with11tests; live
native diagnostic queued after current collection. It will check full IDs and
NPY/RGB bits from the SAMEcallback and independent raw/compact annotation replays.
This separates renderer nondeterminism from storage integrity. No promotion yet.
Also queued3matched original/V2 plant pairs for joint visual/morphology controls;
these6maximum diagnostic frames add zero training diversity and need review.

2026-09-16 inventory continuation: native_dataset/inventory.py authenticates
explicit completed audit receipts against original source plans, donor/target
ancestry, saved variants, callback RGB/Z/calibration and internally consistent
trace summaries. It preserves strict, held and excluded observations, reports
exact decoded-RGB duplicate groups, and never grants approval or a view-cap reset.
Independent audit execution and geometric/near-image admission remain outside
this adapter. Follow-up review closed both lineage and trace-consistency issues.
The first observed snapshot has459 rows/22 receipts:117strict,105held,237excluded;
it covers wider+18plant campaigns only and explicitly disclaims global coverage.
An expanded snapshot is being built after fresh broader/retry audit replays.
V50 regression1687tests+181subtests passed (229.69s); only the two in-progress
V4 verifier and morphology-frame V2 modules were excluded.
