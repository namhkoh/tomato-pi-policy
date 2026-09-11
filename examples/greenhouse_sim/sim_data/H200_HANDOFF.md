# Qwen3-VL-8B dataset handoff (not yet a training release)

2026-09-11, `koh-dev/sim-vlm`. The user will transfer the completed dataset to
the H200 servers; training follows that transfer. Do not start local training,
request server credentials, or treat the current engineering export as ready.

The exact consumer task, annotation convention, file layout and limitations
are explained in [DATASET_CARD.md](DATASET_CARD.md).

### Agent-review checkpoint (2026-09-11, not a release)

Three explicitly requested agents inspected 51 original RGB/card pairs:
37 previously pending examples and 14 existing uncertain/negative cases.
Recommendations were 33 supports, 15 holds and 3 rejects; this is targeted
representative QA, not a measured error rate across the corpus. The 37 pending
task records now have 33 attributed assistant accepts and four holds.
The broader 108-card queue has **91 accepts / 14 holds / 3 rejects / 0 pending**.
One of those 91 accepts has a new assistant source hold because its RGB query
fragment remains ambiguous; the earlier human acceptance is preserved as a
documented disagreement, not overwritten or counted as clean current support.

Evidence: `data/sim_data/dataset_audits/release_20260911_agents/` contains the
three reports, exact per-image reasoning, source/card hashes and
`applied_assessments.json`. New source-level holds are enforced independently
of whether a task review bundle is supplied to the exporter. Three newly held
examples were medium; do not use their old labels as approved coverage anchors.
The review-reconciled diagnostic now includes **63 audits / 20,151 raw
frames / 14,404 candidates**: train 11,637, validation 1,369, test 1,398.
`preflight_02/summary.json` includes the recovered seed17 audit missed by the
initial `audit/audit.json` glob, the new 15-frame seed61 pilot, and seven
latest source holds. The seed61 pilot yielded 15 easy examples and no medium;
two new representative original/card pairs were inspected and assistant-accepted.
Wegener/D additionally inspected two validation-medium original/audit-card
pairs and recommended holds, which were recorded at source level without
fabricating fresh task-card review. Total new targeted inspections: 55 unique
original/card pairs, not a statistical sample of the complete corpus.

The full release fails **validation medium 2/20 and test medium 17/20**.
All other numerical gates pass in this diagnostic; replacement/current-subset
stratified QA and a normal complete export still remain. These are candidate
counts, not individually approved/exportable records. The existing 163-row
engineering preview is unchanged and remains rejected by the training loader.
No final ZIP or training-ready claim is made. A narrower visible/occluded
baseline was proposed to the user as an explicit scope choice, not silently
substituted for the balanced release. No server or training job was started.

The exporter now offers `--reconcile-reviews` for a newly rebuilt corpus:
all input review provenance is retained, but only exact current-label accepts
count toward QA. A hold on a retained image still fails; stale accepts never
approve changed queries. Numeric/visual gates and frozen splits are unchanged.
Export also detects appended review files during copying, so a new hold cannot
silently enter after the initial source scan. Default strict behavior remains.
Source-image holds also follow exact RGB hashes across duplicate audited runs;
another unreviewed copy cannot resurrect a held image.

## Task and files

Task `greenhouse.target_conditioned_cutpoint_rgb.v3` takes **unaltered mounted
head-camera RGB, 848x408**, and a query pixel on a visible target petiole. It
returns JSON `status`, `cut_point_uv`, `visibility`, `next_action`. Coordinates
remain in the original image after any model-internal resizing. The nominal
label is 10 mm along the petiole; the evaluation interval is 10-20 mm, not a
spatial sphere. A verified occlusion produces abstention and a null cut point.

This is localization/abstention supervision, not target discovery, metric XYZ
execution, bimanual action demonstrations, or a trained VLA. Native Isaac
Replicator `distance_to_image_plane` is preserved as float32 optical-Z metres
with validity and calibration. Depth, hidden anatomical XYZ, native identity
masks and review overlays must not enter the RGB model's observation.

Transfer the entire final export, not just its `images` directory:

- `images`: original RGB; `splits`: frozen train/validation/test chat JSONL.
- `depth`: byte-preserved native metric Z and validity; `labels`: evaluator
  geometry/calibration/identity sidecars, separate from model observations.
- `index.jsonl`, `contract.json`, `exclusions.json`, `visual_review.json`,
  `manifest.json`: provenance, task/prompt, exclusions, QA and content hashes.
- Matching repository commit and environment/model revision receipts.

Target-source families are split-disjoint; shared greenhouse backdrop context
is not fully scene-disjoint. Preserve every existing human/assistant hold and
rejection. Do not move examples across splits to satisfy coverage thresholds.

## Current increment, not the global dataset total

Two completed serial batches, `grounding_sunday_small_20260910_v2` and
`grounding_sunday_20260911_v3`, contain 98 raw frames. Current task-v3 screening
retains 84 (11 train, 29 validation, 44 test); 14 are excluded. None is medium
difficulty. Seventeen representative cards were individually inspected and
accepted as **assistant** QA (10 visible, 7 occluded), not fabricated human
decisions or physical cut approvals. Review folders ending `_v3` and `_v4`
here denote review waves, not active-perception task-v4 episodes.

`data/sim_data/training_exports/grounding_sunday_20260911_engineering` packages
these 84 rows. Its state is `incomplete_engineering_export_do_not_claim_release`.
The portable validator passed using ordinary non-Isaac Python. It deliberately
fails complete-release coverage and is rejected by `GroundingDataset`.
The historical broader recount is 14,235 candidates from 19,932 raw frames;
do not add this increment to that number without a new deduplicated recount.
Complete broader QA and held-out medium coverage remain pending. There are no
qualified dynamic grasp-cut training episodes in this increment.

Subsequent bounded batch `grounding_sunday_20260911_v4` completed both native
workers cleanly: 106 raw / 79 task-eligible (validation 28 easy + 3 hard; test
45 easy + 3 hard). Twenty-seven are excluded, not reclassified. Eight new
representative cards were actually inspected and accepted as assistant QA in
`dataset_reviews/grounding_sunday_20260911_v5`. No medium examples were produced.

Latest **engineering preview**, not the transfer-ready release:
`data/sim_data/training_exports/grounding_sunday_20260911_engineering_v2`.
This combines seven completed jobs from three batches: 204 raw / 163 portable
deduplicated rows (11 train, 60 validation, 92 test), 25 bound representative
reviews, original RGB and byte-preserved native depth. Validation passed under
ordinary non-Isaac Python with explicit incomplete mode. The normal training
loader correctly rejects it as incomplete. The old preview remains unchanged.
These increment counts are not a fresh global recount or a substitute for the
broader training corpus. Broader QA gained 22 individually inspected assistant
accepts; its 108-card queue now has 58 accepts, 8 holds, 3 rejects, 39 pending.
No prior/human decisions, release gates or family assignments were changed.

## Adapter implemented; real-model validation pending

`sim_data/qwen_adapter.py` shares RGB/prompt conversion for supervised encoding
and inference. It preserves system/user/assistant roles and masks prompt/image
tokens from loss using a checked generation-prefix boundary, not guessed token
IDs. Oversize examples fail explicitly; no silent truncation. Only a complete
validated release can be opened as `GroundingDataset`.

Unit tests use a fake processor with Torch tensors. **No actual Qwen processor
or weights were loaded, no forward/backward pass or training was run.** A real
processor template-prefix/image-grid smoke test is required on the server.
This adapter is not a complete distributed trainer or an executable robot bridge.

The [official Qwen fine-tuning loader](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/data/data_processor.py)
maps `human` turns to user and other conversation turns to assistant. Blindly
converting this release's system prompt to that format would mislabel its role.
Use the role-preserving adapter or explicitly validate an upstream integration.

## Acceptance and server sequence

1. Recount only clean-exit, independently audited jobs; deduplicate all selected
   sources under the current v3 contract. Fill held-out medium deficits with
   actual native captures, not threshold relaxation. Complete stratified QA.
2. Build a **new** export without `--allow-incomplete`; require all frozen
   numerical and visual gates. Validate locally, archive with a checksum, and
   have the user copy it. Current engineering artifacts are not that archive.

   Once the normal validator passes, the guarded packager can create the ZIP:

   ```bash
   python -m sim_data.release_archive --release /path/to/final_release --output /path/to/grounding_release.zip
   ```

   This has no incomplete-mode override. It packages only manifest-bound
   files, verifies the bytes inside the ZIP, and publishes a `.zip.sha256`
   sidecar. Interrupted or failed copies remain `.partial`, not final ZIPs.
   It does not start training. No archive has been published at this checkpoint.
3. From the matching checkout's `examples/greenhouse_sim`, on H200:

   ```bash
   python -m sim_data.training_export validate --output /path/to/final_release
   ```

   This uses portable files, not Windows source paths or Isaac Sim. Install
   the audited Python dependencies in an isolated server environment first.
   Record archive hash, repo/model revision and installed package versions.
4. Load [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
   Verify actual processor image and prefix/loss masks, original-coordinate
   decoding, and one forward/backward pass using train examples only. No test
   labels may choose hyperparameters or stopping criteria.
5. Run a small train-only overfit smoke, then the planned LoRA baseline. Freeze
   image augmentation initially; the image coordinate contract must not drift.
   The [official framework](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune)
   is a reference, not evidence this repository's distributed trainer runs.
   Measure memory and speed on one H200 before configuring four-GPU training.
   GPU availability, batch capacity, wall time and model latency are unmeasured.
6. Compare untuned versus tuned 8B on held-out families. Keep existing 32B
   evaluations as a separate frozen reference; smaller-model latency is an
   experiment, not a demonstrated advantage. Report JSON validity, pixel and
   admissible-segment accuracy, false localization under occlusion,
   abstention/coverage, per-family/difficulty scores and p50/p95 latency.

Only afterward connect predictions to fresh native depth/calibration and the
guarded simulator controller. Reobservation, stale-frame rejection, target
identity, valid grasp, cut corridor, retained orphan and verified deposit must
be checked independently. Hidden cut coordinates never rescue an abstention.
