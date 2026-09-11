# VLM Training Data Plan: Greenhouse Deleafing

Date: 2026-09-06

Branch: `koh-dev/sim-data`

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
