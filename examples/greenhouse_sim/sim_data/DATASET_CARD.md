# Tomato cut-point grounding dataset: consumer guide

This describes task `greenhouse.target_conditioned_cutpoint_rgb.v3`.
It is a synthetic **perception** task, not an action-demonstration dataset.
Only an export whose normal validator passes is the substantive training
release. Directory names, assistant recommendations and successful file
copying are not release approval. See `H200_HANDOFF.md` for current readiness.

## Explicit release profiles

The user authorized `visible_occluded_v1` on 2026-09-11 as the first, narrower
fine-tuning baseline. It includes **easy/clear localization and hard/occluded
abstention only**. All medium/partial examples are excluded, never relabeled.
The task-v3 prompt and coordinate convention remain unchanged. Performance on
partial occlusion is unvalidated; this is not the balanced easy/medium/hard
release. `balanced_v1` retains its original medium-coverage requirements.

A valid baseline manifest has `release_profile: visible_occluded_v1` and
`state: complete_visible_occluded_baseline_release`. Its normal validator must
pass the same split-family, target-diversity, minimum total/localized/occluded
counts, localization-spread and representative visual-QA checks. The profile
is not a way to accept an incomplete preview. Actual counts and exclusions
are in that package's manifest. No training run or model quality is implied.

## One training example

Input: one unaltered, full-scene **848 x 408 RGB** frame from the simulated
RB-Y1 A v1.2 mounted head D405, plus a textual pixel query identifying a
visible target petiole. The input query is NOT the cut point. It is currently
supplied by the dataset, not selected by the VLM or a trained detector.

Output: an assistant JSON answer with exactly four fields:

| Field | Meaning |
|---|---|
| `status` | `localized` or `abstain` |
| `cut_point_uv` | `[u, v]` in original image pixels, or `null` |
| `visibility` | `clear`, `partial`, or `occluded` |
| `next_action` | `inspect_cut_region` or `change_viewpoint` |

Coordinates have a top-left edge origin, x right and y down. Pixel centres
are `(column + 0.5, row + 0.5)`. Model-internal image resizing must NOT change
the coordinate system of the answer. There are no bounding-box outputs.

A visible label is `localized`, a finite pixel pair, `clear`/`partial`, and
`inspect_cut_region`. A proven hidden-cut example is:

```json
{"status":"abstain","cut_point_uv":null,"visibility":"occluded","next_action":"change_viewpoint"}
```

The complete, versioned system prompt is exported in `contract.json` and in
every chat row. The user prompt is:

> The target petiole passes through pixel (u, v). Locate its nominal cut point
> if the junction and cut region are visually distinguishable; otherwise
> abstain. Use the original full image.

Do not remove the system role or treat it as an assistant answer. The current
adapter in `qwen_adapter.py` keeps roles and masks prompt/image tokens from
the supervised loss. Its actual-model validation is a separate prerequisite.

## Where labels come from

1. Original plant component identities and attachment/centreline geometry
   define the candidate petiole and a nominal point **10 mm along it toward
   the detachable branch**. The evaluation interval is 10-20 mm along this
   centreline, not a 1-2 cm sphere around the junction.
2. The robot pose, head-camera mount and intrinsics project the geometric
   point into the original RGB frame. Exact visible component identity and
   native optical-Z evidence determine whether the image supports it.
3. A target query is selected on a visible distal petiole region, at least
   45 mm from attachment and 18 image pixels from the nominal point. Positive
   examples additionally require a connected visible petiole region from
   query to nominal cut, without artificial mask-gap filling.
4. Sampling, exposure, query-usability and junction-context checks exclude
   weak examples. A proven foreground occluder produces a null answer, not
   hidden cut coordinates. Unknown visibility is excluded rather than
   automatically labeled a negative.
5. Source holds/rejections and representative task-specific visual QA are
   applied before export. Assistant review is attributed as assistant review;
   it is not independent human or horticultural confirmation.

The 10 mm rule is our synthetic labeling convention. It is not calibrated
tissue-cutting physics or proof that every source attachment is agronomically
appropriate. Source geometry can be internally consistent yet visually
ambiguous; native masks do not override an ambiguous original RGB image.

## Difficulty and limitations

- **Easy:** eligible visible cut, at least 95% sampled proximal and interval
  visibility under the current contract.
- **Medium:** eligible visible cut with partial context; the contract still
  requires at least 80% proximal and interval visibility plus other checks.
- **Hard:** the nominal cut is behind an identified foreground occluder;
  the desired answer is abstention. A fruit, leaf or main stem visible in the
  diagnostic crop is not itself the proposed cutting target.
- **Real-hard:** not included; no Cosmos image-to-image augmentation is
  approved in task v3.

These are operational geometric visibility classes, not measured human or
VLM difficulty ratings. A visually crowded image can still be in Easy.
Ambiguous/unknown and genuinely invalid-target examples are not a complete
negative class here. The separate task-v4 pilot is not merged into this task.

## Portable files

```text
release/
  images/                 original RGB, the ONLY model image input
  splits/train.jsonl      image path + system/user/assistant chat rows
  splits/validation.jsonl
  splits/test.jsonl
  depth/                  native float32 optical-Z .npy + validity PNG
  labels/                 evaluator geometry/calibration/target-mask sidecars
  index.jsonl             IDs, frozen split, family, task and source identity
  contract.json           exact task definition and system prompt
  exclusions.json         excluded samples and reasons
  visual_review.json      bound representative review evidence
  negative_review_history.json  historical image holds across task versions
  manifest.json           release state, gate results, hashes and provenance
```

Depth is copied byte-for-byte from Isaac Sim Replicator's
`distance_to_image_plane` annotator on the same camera render product as RGB.
It is **408 x 848 float32 optical-axis Z in metres**, not Euclidean range,
estimated monocular depth or a colour heatmap. Invalid values remain in the
array and are described by the validity mask. It is ideal simulator depth,
not a measured D405 sensor-noise model. Heatmaps are review visualizations.

Centreline XYZ, hidden cut truth, masks, overlays and depth are sidecars for
evaluation/geometric processing, not observations for this RGB-only model.
Do not upload the diagnostic review cards as training input images.

## Splits and intended training

Original target families retain seed-0 reservations: **16 train, 4 validation,
4 test families**. Exact row counts belong to the validated release manifest,
not this general guide. Family labels and camera/scene duplicates are checked
across the export. The shared greenhouse backdrop is NOT scene-disjoint;
this is not evidence of real-world generalization.

Initial intended model: `Qwen/Qwen3-VL-8B-Instruct`, supervised fine-tuning
with a planned LoRA baseline. Existing 32B inference runs are separate
evaluation evidence, not trained checkpoints. Validate the real processor,
loss mask, original-coordinate decoding, finite forward/backward loss and
checkpoint reload before a multi-GPU run. Do not tune on test labels.

From the matching checkout's `examples/greenhouse_sim`, run:

```bash
python -m sim_data.training_export validate --output /path/to/release
```

Do not add `--allow-incomplete` to make a final release pass. Transfer the
whole validated directory and its archive checksum, not only `images/`.

## What this cannot train or authorize

No autonomous target discovery, RGB-to-metric-XYZ guarantee, left/right arm
trajectories, physical grasp/cut/retain/deposit policy, dynamic reobservation,
stale-frame safety or safe blade corridor is supervised here. An eventual
controller must verify observations, metric geometry, grasp and tool access
independently; it must not use hidden sidecar coordinates to rescue abstention.
The physics qualification runs are not automatically dynamic training data.
