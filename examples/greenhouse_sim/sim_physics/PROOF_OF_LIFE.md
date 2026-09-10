# Minimal VLM + manipulation proof of life

2026-09-10, `koh-dev/sim-vlm`, starting from `b9b83b8`.
This is an execution plan, **not a report of a trained or end-to-end policy**.
No hardware commands, collection jobs, training jobs, release approvals or split
changes are authorized by running the physics demo.

September 11 update: the user separately authorized new static collection and
knife integration. The small serial batch
`data/sim_data/collection_batches/grounding_sunday_small_20260910_v2` completed
with 48 audited raw captures (12 train-family / 14 validation-family / 22 test-family).
These are pending visual review, not additional approved/exported task-v3
records. The larger first batch failed system-memory allocation at 114 captures
and remains `failed_do_not_train`. No tuning or dynamic collection was started.

Later September 11 update supersedes the pending-review status above: 17 new
representative assistant reviews cover that batch and a second 50-frame batch.
The two batches yield an 84-row **incomplete engineering** portable export.
See `../sim_data/H200_HANDOFF.md`. The user will transfer the completed dataset
to H200 before training; no local training/server access is needed now.

An opt-in bimanual native-contact/seam-release harness now exists, including a
session-only correction for the backward knife. The corrected-knife greenhouse
run reaches left grasp but rejects the right cutting approach. Thus the core
proof-of-life blocker remains a **collision-clear, physically verified complete
sequence**. Unit-test coverage or a reachable endpoint is not that evidence.

## What the current evidence supports

- Supplied greenhouse + full native RB-Y1 A v1.2 articulation. The left arm
  approaches a privileged fixture target, closes physical fingers, verifies
  opposing contact, moves 10 mm, holds and opens. There is no grasp weld.
- `data/sim_physics/greenhouse_dense_20260910_01/report.json`: all grasp gates
  passed for one target with 143 surrounding plants at the original preview
  planting density. Target movement 7.825 mm, maximum slip 2.489 mm,
  penetration 0.370 mm, deterministic half-second reset replay.
  Its entire 1,680-step recorded physical trajectory equals the sparse
  pre-optimization `greenhouse_physics_20260910_06` trajectory.
- Only the selected branch/leaf carriers are compliant. Nearby backdrop vines
  have static native triangle-mesh contact; distant vines are render instances.
  This is not whole-greenhouse deformable-plant simulation.
- The right knife is present but parked. Verified blade cutting, reliable
  post-severance retention, deposit and observation-driven execution remain
  unfinished in this current environment. Legacy cutting code is not evidence
  that this new native-contact sequence already works.
- The documented read-only recount is 14,235 task-v3 candidates
  (11,627 train / 1,317 validation / 1,291 test), from 19,932 audited raw frames.
  These are candidates, not a fully approved portable release. See `dev.md`,
  `sim_data/TRAINING_DATASET.md` and the immutable audit/review receipts.
  The task-v4 pilot has 28 static examples, not manipulation episodes.
  No new recount, review decision or split change was made in this increment.
- Static RGB/query -> cut pixel / visibility / abstention is a useful VLM task.
  It does **not** supervise joint actions, grasps, contact forces or trajectories.
  There are no qualified dynamic training episodes from these physics probes.

## Smallest honest system

Use a small VLM with a deterministic IK/contact-feedback controller first.
This satisfies the low-level-controller option, but must not be called a
trained VLA. Automatically executed controller episodes can later supervise
a learned low-level policy without teleoperated demonstrations.

1. Capture fresh **mounted head-camera** RGB and native Isaac
   `distance_to_image_plane` optical-Z, validity, intrinsics, camera/robot pose,
   frame identity and simulation time. Keep the existing 848x408 image contract.
   Diagnostic third-person PNGs are evidence, not training camera inputs.
2. For the first restricted experiment, provide an explicit candidate query
   and ask the VLM for the existing task-v3 answer. Target discovery is a
   separate capability; do not label a query-conditioned model autonomous.
3. Reject abstention, malformed answers, invalid/ambiguous depth, stale images
   or unverified target identity. Back-project an observed pixel through native
   optical-Z and calibration; do not substitute hidden simulator cut XYZ when
   the point is occluded. Keep evaluator truth outside executable model inputs.
4. Geometry chooses a detached-side grasp with clearance from the observed
   10 mm nominal cut / 10-20 mm longitudinal admissible segment. Validate both
   arms, fingers, camera brackets, blade corridor and neighboring structures.
   A fixture-known grasp may bootstrap mechanics, but label that stage privileged.
5. Execute left grasp -> verify opposing contact and bounded slip -> reobserve
   -> right approach / cut / withdraw -> left retain / deposit -> verify outcome.
   On grasp loss, changed geometry, invalid visibility or unsafe access:
   withdraw safely, inspect/retry within a bound, or skip. No blind cut fallback.
6. Reposition the **held target** only in a second, matched experiment. Reobserve
   after the movement and recompute the cut and tool corridor. Do not have the
   same left hand hold a separate occluder and the target simultaneously.

The current probe does not yet implement steps 1-6 as one observation-driven
loop. In particular its recorded synchronization flag is false; never convert
that to true merely by adding timestamps to existing records.

## Parallel implementation tracks and acceptance criteria

Effort estimates below are engineering workdays, **not measured runtimes or
delivery promises**. They assume one engineer plus access to the training host.
Some tracks can overlap; contact failures may require additional iteration.

| Track | Proposed effort | Concrete acceptance |
|---|---:|---|
| Scene performance and repeatable grasp | 1-2 days | Preserve source assets, 240 Hz physics, contact guards and physical trace; measure wall-time throughput with dense context and all mounted views. No faster-video claim in place of a faster simulator. |
| Frozen-model baseline + reviewed prototype export + SFT integration | 2-3 days after host access | Preserve frozen family splits and all holds; version the exact subset, prompts, model revision and environment. Validate portable RGB/native-depth hashes. Run train-only overfit smoke, then held-out evaluation; no test-set tuning. |
| Native blade event and retained detached material | 3-5 days | Blade leading flat edge, direction/sweep/contact and bilateral-grasp conditions gate attachment release. Negative controls (no contact, wrong direction, no grasp, protected stem) do not cut. Left retains the orphan after withdrawal, not a hidden weld. Report slip/contact and damage; distinguish joint release from calibrated tissue fracture. |
| Mounted observation -> guarded controller integration | 2-3 days | Verify frame/time/pose alignment while moving; stale/occluded/invalid-depth inputs block execution. Execute from observed goals, reobserve after grasp/movement, verify release and deposit, recover/reset without stale handles. |
| Matched visible and reposition trials | 1-2 days after above gates | At least 20 fixed visible cases and 20 paired hold-vs-reposition cases, same controller/seeds, all failures recorded. Predeclare tolerances before examining results. |

Initial **engineering** promotion target: >=18/20 completed visible sequences,
zero unintended cuts/protected-structure violations, each successful sequence
retains the orphan until commanded deposit, and every trial resets or ends in
an explicit recoverable failure. These are proposed gates, not current results
or evidence of hardware safety. Tune slip/contact/damage limits from controlled
qualification, not after seeing favorable outcomes.

For each pair measure completion, grasp loss, native load/penetration, target
visibility, observed cut localization error, minimum cutter clearance,
unintended contact/cuts, retention, deposit location and wall/simulation time.
Use native identities/hidden geometry for **evaluation only**. Compare
hold vs reposition with identical initial states and the same controller;
record whether access actually improves after reobservation.

## VLM/model decision

Start with **Qwen3-VL-8B-Instruct + LoRA** for continuity with
`vlm_train_data.md`; keep the existing 32B evaluation setup as a frozen
reference, not a source of metric ground truth. A 4B model is a latency
candidate only after measuring the same held-out localization and abstention
metrics. There are no measured 4B/8B latency or manipulation results here.
Model reference: [official 8B card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct);
training route: [official fine-tuning framework](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune).

The four-H200 machine was previously described by the user, but its current
SSH access, available VRAM, installed stack and scheduling are not verified.
Confirm these before launching. Use a single-device smoke first, then measured
batching/distributed LoRA if needed; GPU count alone does not establish speed
or guarantee a particular batch/context length. No trainer dependencies were
installed and no training was launched in this optimization increment.

Evaluate zero-shot vs fine-tuned 8B, frozen 32B where available, and simple
geometry/query-only baselines on the same untouched held-out families.
Measure valid JSON, localization error in original pixels and verified metric
space, acceptable-segment hit rate, occluded false-localization rate,
abstention/coverage, and p50/p95 inference latency. Separate inference time
from image transfer and controller execution.

Do not wait for a full VLA before a first controlled demonstration. Conversely,
a controller using fixture truth does not prove that a VLM controls grasping.
Promote only once the observable-goal bridge works and errors can be traced
through perception, geometry, controller and physical outcome separately.

## Minimum subsequent dynamic record

Version episode/seed/source-family/split/controller and model; native before
and after RGB/Z/validity/calibration/frame-time/poses; observed target and grasp
identity; requested strategy and metric command; actual joint/finger states;
controller-reported contact/bilateral-grasp/slip; blade contact/direction and
cut-verification event; attachment state; retention/deposit/damage outcomes;
failure reason, retry/recovery/reset events; provenance and approval status.

Store private simulator truth separately. Include failed grasps, blocked cuts,
slip, ambiguous occlusion, skip and recovery examples. Static alternate views
are not action-outcome episodes. Defer learned path guidance, whole-house
deformable physics, separate-leaf clearing, photorealistic augmentation and
large-scale online RL until this smaller loop is repeatable.
