# Qwen3-VL-8B dataset handoff (not yet a training release)

2026-09-11, `koh-dev/sim-vlm`. The user will transfer the completed dataset to
the H200 servers; training follows that transfer. Do not start local training,
request server credentials, or treat the current engineering export as ready.

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
