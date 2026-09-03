# VLM Cut-Point Evaluation Architecture

Status: implementation assessment for `koh-dev/vlm-eval` (2026-09-03).

## Objective

Evaluate closed and open-weight vision-language models on whether they can
identify a safe, agronomically correct tomato-petiole cut point from greenhouse
camera observations. This is a perception-and-reasoning benchmark first. A VLM
proposal must remain advisory and must not directly command robot joints.

The work should answer three questions independently:

1. Can the model see the requested plant and decide whether a safe cut exists?
2. Can it select the correct petiole and localize the cut point and direction?
3. Is that proposal geometrically and physically executable without contacting
   the main stem, neighbouring organs, greenhouse, or robot?

Keeping those questions separate prevents motion-planning failures from being
misreported as VLM failures and prevents simulator ground truth from leaking
into model inputs.

## Existing pipeline that can be reused

The simulator already has most of the private supervision needed for a strong
visual benchmark:

- `_TeleopCameraRecorder` captures synchronized RGB PNGs from the head, left
  wrist, and right wrist cameras through Isaac Replicator.
- Demonstrations already use `greenhouse.demonstration.v1` and
  `greenhouse.demonstration.step.v1`, with camera paths, robot state, actions,
  task phase, cut state, and safety state.
- `EpisodeTarget` provides deterministic task target selection.
- `Organ.attachment` and `Junction.cut_position_m` provide hidden semantic and
  geometric cut truth.
- `PhysicalCutEvaluator` already encodes cut-zone, centre-crossing, direction,
  force, work, protected-contact, and residual-stub outcomes.
- The head and both wrist D405 views already exist in the composed RB-Y1 scene.

The online-RL observation is currently low dimensional. VLM evaluation should
therefore be an offline sibling pipeline, not an API call inside the 240 Hz
physics loop and not a provider dependency inside Isaac Sim's Python runtime.

## Recommended system boundary

```text
Isaac capture
  -> immutable RGB + calibration sample
  -> public model manifest / private evaluator labels
  -> provider-neutral offline runner
  -> canonical cut-point response
  -> geometric/agronomic scorer + review overlays
  -> later: D405 back-projection and existing planner/safety gates
```

The model-facing side receives only RGB images and a natural-language task
instruction. Depth, segmentation masks, world geometry, selected `SubStem`
names, robot state, and simulator target IDs remain private evaluator data.

This boundary also keeps credentials, network latency, provider SDK versions,
and local-model GPU allocation outside the simulator. Local models should run
in a separate environment and expose an OpenAI-compatible HTTP endpoint through
vLLM or SGLang. No provider SDK should be added to the Isaac environment.

## Benchmark tasks

Use progressive levels so that failures remain interpretable:

| Level | Task | Required output |
|---|---|---|
| L0 | Visibility and abstention | `cut`, `no_safe_cut`, or `uncertain` |
| L1 | Instruction-conditioned target selection | target petiole/branch in the image |
| L2 | Cut localization | one image-space cut point |
| L3 | Direction and hazard reasoning | 2-D cut direction and hazard flags |
| L4 | Multiview/temporal consistency | consistent proposal across head/wrist views or adjacent frames |

The first benchmark should score a single requested cut. Asking for every valid
cut in one image introduces set-matching and recall issues before the single-cut
contract is known to work.

Negative examples are mandatory: no visible target, occluded attachment,
ambiguous overlapping petioles, unsafe proximity to the main stem, already-cut
branches, and neighbouring-vine distractors. A model that always emits a point
must score poorly.

## Dataset contract

Store immutable versioned datasets under:

```text
data/greenhouse_sim/vlm_eval/datasets/<dataset_id>/
  public/
    manifest.jsonl
    frames/<sample_id>/head.png
    frames/<sample_id>/left_wrist.png
    frames/<sample_id>/right_wrist.png
  private/
    labels.jsonl
    calibration/<sample_id>.json
    depth/<sample_id>/<view>.npy
    masks/<sample_id>/<view>.png
  review/
    overlays/
```

Each public row should contain a stable sample ID, dataset/schema versions,
instruction, available view names, relative RGB paths, image dimensions, plant
asset/split identifier, and capture-condition metadata that cannot reveal the
answer. Each private row should contain target organ identity, projected cut
point, valid cut-region mask, protected-organ masks, 3-D cut position and axis,
visibility/occlusion status, and label provenance.

Capture requirements:

- Capture RGB at 1280 x 720 PNG initially. This preserves camera detail and is
  within the normal image-size handling of the target hosted APIs.
- Capture RGB, metric depth, camera intrinsics, camera extrinsics, semantic or
  instance masks, target geometry, and robot/plant transforms from the same
  simulator sample.
- Advance physics with `step(render=False)` and perform explicit render updates;
  do not restore `step(render=True)` to the physics loop because that previously
  multiplied substeps and caused slow, shaky visible execution.
- Exclude UI chrome, cursor, target highlights, selected-prim outlines, internal
  labels, and debug overlays from model inputs.
- Generate overlays only after inference for human review.
- Split by source plant asset and scene configuration, never by adjacent frame,
  to prevent near-duplicate leakage.
- Keep `tomato_glb_30` as an out-of-distribution/topology-shift set until its
  changed organ semantics are validated. It is not a drop-in replacement for
  the current labeled vines.

Use hidden organ geometry to rasterize the valid cut region and protected
regions. Human review should verify projections and ambiguous visibility, but
pixel-click labels should not replace exact simulator geometry when it exists.

## Canonical model response

All provider responses should validate against one strict schema, for example:

```json
{
  "schema_version": "greenhouse.vlm_cutpoint.v1",
  "decision": "cut",
  "primary_view": "head",
  "cut_point_px": {"x": 713, "y": 402},
  "cut_direction_px": {"dx": -0.93, "dy": 0.37},
  "confidence": 0.78,
  "hazards": ["near_main_stem"],
  "reason_codes": ["target_attachment_visible"],
  "rationale": "The target petiole joins the main stem at the marked location."
}
```

`cut_point_px` uses integer pixels in the declared `primary_view`, with the
origin at the top-left of the original submitted image. The adapter must retain
the submitted and provider-resized dimensions so coordinates can be mapped
back correctly. The direction vector is normalized in image coordinates.

For `no_safe_cut` or `uncertain`, point and direction must be null. Hazards and
reason codes should be enums, not unconstrained prose. Request a concise
rationale for auditability, but do not request or store hidden chain-of-thought.

## Provider-neutral runner

Proposed package layout:

```text
examples/greenhouse_sim/vlm_eval/
  schema.py
  capture_contract.py
  prompts.py
  runner.py
  scoring.py
  report.py
  providers/
    base.py
    openai_responses.py
    anthropic_messages.py
    gemini_interactions.py
    openai_compatible.py
```

Every adapter should accept the same sample and inference configuration and
return the canonical response plus raw response, provider/model revision,
prompt revision, image hashes, seed/temperature where supported, latency,
token use, and estimated cost. Raw responses belong in run artifacts, not in
the dataset manifest.

The runner must support resume-by-sample, bounded concurrency, exponential
backoff, immutable run IDs, response caching keyed by model/prompt/image hashes,
and a dry-run validator. Use deterministic decoding where supported and repeat
a subset to measure stochastic variance.

Initial comparison panel:

- OpenAI: one high-reasoning vision model and one lower-cost model through the
  Responses API with structured JSON output.
- Anthropic: current Claude Opus and Sonnet vision models with structured
  outputs; explicitly normalize any provider-resized coordinates.
- Google: current Gemini Pro and Flash vision models with structured outputs.
  Treat Gemini Robotics-ER as a separate embodied/spatial model track.
- Open-weight: Qwen3-VL-8B-Instruct, InternVL3.5-8B-Instruct, Molmo2-O-7B, and
  Gemma 3 12B as a representative first pass. Record quantization, engine,
  precision, GPU, and exact revision because local-runtime choices affect
  results. Use “open-weight” rather than “open source” when licenses differ.

Model IDs change over time; they belong in run configuration, not source-code
defaults. Pin exact dated versions or revisions whenever a provider supports it.

## Scoring

Report perception metrics separately from execution metrics:

- Schema-valid response rate and invalid/repair rate.
- Cut/no-safe-cut/uncertain confusion matrix and abstention coverage.
- Target-organ selection accuracy.
- Pixel localization error and PCK at multiple thresholds, normalized by image
  diagonal and by projected petiole diameter.
- Depth-back-projected 3-D error when valid depth exists.
- Signed residual-stub error and existing flush/marginal/stub/invalid bins:
  flush <= 5 mm, marginal <= 20 mm, otherwise stub.
- Direction angular error relative to the feasible blade direction.
- Protected-contact risk for main stem, non-target organs, neighbouring vines,
  greenhouse structure, and robot.
- Calibration: risk-coverage curve, Brier score, and expected calibration error.
- Cross-view 3-D disagreement and temporal proposal jitter.
- Latency, input/output tokens, estimated API cost, and local VRAM/throughput.

Use at least four baselines: image centre, random eligible pixel, a simple
segmentation-plus-geometry method, and the privileged simulator oracle. The
oracle is an upper-bound integration test and must never be presented as a VLM.

For later execution, back-project a valid VLM point through D405 depth, reject
missing/edge-discontinuous depth, fuse views where available, and pass the 3-D
proposal through the existing reachability, swept-volume, inter-arm, blade,
contact, force, and work gates. VLM confidence must never override those gates.

## Implementation order and acceptance gates

### Phase 1 — capture and labels

Implement the public/private manifest writer, synchronized RGB/depth/calibration
capture, target/protected masks, and review overlays. Produce a 50-sample pilot
covering at least five plant assets, positive and negative cases, all three
cameras, pose/airflow variation, and multiple occlusion levels.

Acceptance: every sample reopens reproducibly; image hashes and dimensions
match; projected labels pass automated bounds/depth checks; a human review of
all 50 overlays finds no answer leakage or incorrect target projection.

### Phase 2 — offline inference

Implement schema validation, prompt versioning, provider base interface, one
mock adapter, then one hosted and one local adapter. Add the remaining providers
only after the canonical contract is stable.

Acceptance: interrupted runs resume without duplicate charges, malformed
responses are retained and scored, credentials never enter artifacts, and the
same cached run can be rescored without provider access.

### Phase 3 — scoring and report

Implement metrics, confidence curves, paired per-sample comparison, bootstrap
confidence intervals by plant asset, latency/cost tables, and failure overlays.

Acceptance: synthetic oracle predictions score at the expected upper bound;
known offsets and hazard points fail the appropriate metrics; train/dev/test
splits share no plant asset or adjacent trajectory.

### Phase 4 — advisory simulator integration

Show the proposal, confidence, reason codes, and safety-gate result in the UI,
with execution disabled. Log whether a human accepts, corrects, or rejects it.

Acceptance: model/network failure cannot block or slow the physics loop; stale
responses are visibly rejected; proposals cannot move the robot.

### Phase 5 — gated perception to action

Only after held-out simulated and real D405 validation, transform accepted
points into 3-D approach poses and submit them to the existing motion planner.
The planner and physical cut evaluator remain authoritative.

## Decisions required before scaling

- Define the first natural-language agronomic instruction family and target
  eligibility rule.
- Choose head-only versus head-plus-wrist model input for the first leaderboard.
- Decide whether the benchmark requests one cut or an ordered set of cuts.
- Set human-review policy for occlusion and ambiguous attachment points.
- Approve cloud data handling for rendered and, later, real greenhouse images.
- Decide when `tomato_glb_30` has sufficient semantic validation to become a
  standard split rather than an OOD test.

## Immediate engineering increment

Implement Phase 1 only: the leakage-safe capture contract, 50-sample pilot, and
review overlays. This is the highest-leverage next step because every hosted or
local model can then be evaluated against exactly the same immutable inputs and
private physical ground truth. Provider API plumbing before this contract would
create incomparable results and encourage accidental simulator-label leakage.
