# Tomato petiole cut-point VLM: results, models, and recommendations

**Date:** 2026-09-13  |  **Task:** `greenhouse.target_conditioned_cutpoint_rgb.v3`  |  **Dataset:** `visible_occluded_20260911_v1` (11,520 train / 1,358 validation / 1,381 test; test never used)  |  **Base model:** Qwen/Qwen3-VL-8B-Instruct (revision `0c351dd`)  |  **Hardware:** 4x H200

## 1. TL;DR

- Fine-tuning turned a model that failed the output contract on 100% of images into one that answers the required JSON on 100% of images and localizes visible cut points with a median error of ~18 px (about 21 mm at the target). That is 3.4x better than the best image-free prior (61 px).
- **The models cannot yet be used to command a cut.** Only ~3% of answers fall inside the 10-20 mm admissible interval, and the localize/abstain decision is close to chance on held-out plants (balanced accuracy 0.60, AUC 0.64 at best): the best model still fabricates a cut point for 52% of occluded targets.
- Root causes are mostly in the data, not the recipe: 11,520 training rows cover only 168 targets on 16 plants; occlusion correlates with target identity; the hidden-vs-visible cue is an ~8 px detail on 5 px-wide petioles.
- Next experiment already queued: RGB-D input (native depth as a second image). Data recommendations are in section 6.

## 2. Published models (private HuggingFace repos, account `namhokaist`)

| Model | Repo | Notes |
|---|---|---|
| **Full FT v2 (recommended)** | https://huggingface.co/namhokaist/qwen3-vl-8b-tomato-cutpoint-full-v2 | full-02: abstention-focused; 2-decimal normalized coordinates |
| Full FT v1 | https://huggingface.co/namhokaist/qwen3-vl-8b-tomato-cutpoint-full | full-01: plain full fine-tune, 3 epochs |
| LoRA baseline | https://huggingface.co/namhokaist/qwen3-vl-8b-tomato-cutpoint-lora | lora-01: runbook recipe, rank 16, 1 epoch |

Each repo contains weights, processor/tokenizer, `grounding_adapter.json` (coordinate convention), a model card with the validation table, and `provenance/` (run contract, trainer state, validation report, environment freeze).
Weights & Biases (project `tomato-pi`): LoRA `xcwqmy6m`, full-01 `5plv8gya`, full-02 `hmue3ag3`.

Coordinate convention: prompts and answers use normalized coordinates in [0,1000), `x=1000*u/848`, `y=1000*v/408`; v2 rounds to 2 decimals. Decode with `[848*x/1000, 408*y/1000]`. Output JSON: `status`, `cut_point_uv`, `visibility`, `next_action`.

## 3. Validation results (1,358 held-out rows, 4 unseen plants, greedy decoding, invalid answers counted as failures)

| Metric | Untuned base | LoRA (lora-01) | Full FT v1 (full-01) | Full FT v2 (full-02) |
|---|---:|---:|---:|---:|
| Valid JSON answers (of 1,358) | 0 | 1358 | 1358 | 1358 |
| Status accuracy | n/a | 0.711 | 0.637 | 0.655 |
| Abstain accuracy on occluded (n=346) | n/a | 0.113 | 0.358 | 0.483 |
| False localization on occluded | n/a | 0.887 | 0.642 | 0.517 |
| Localized coverage on visible (n=1,012) | n/a | 0.915 | 0.732 | 0.713 |
| Balanced accuracy of localize/abstain | n/a | 0.514 | 0.545 | 0.598 |
| Status-score AUC (probe) | n/a | n/a | 0.576 | 0.635 |
| Within 5 px of label | n/a | 0.043 | 0.045 | 0.047 |
| Projected 10-20 mm interval hit (2 px) | n/a | 0.024 | 0.020 | 0.027 |
| Median pixel error (answered) | n/a | 27.8 px (~33 mm) | 17.5 px (~21 mm) | 18.0 px (~21 mm) |
| Answered within 10 px / 20 px | n/a | 0.14 / 0.35 | 0.24 / 0.57 | 0.23 / 0.56 |
| Latency p50 / p95 (H200, batch 1) | 0.73 / 0.76 s | 1.62 / 1.68 s | 1.07 / 1.21 s | 0.67 / 0.73 s |

Pixel-to-mm conversion uses the median projected length of the 10 mm interval in the validation images (8.5 px, i.e. 0.85 px/mm).
Untuned base: every answer is `{"status":"abstain","cut_point_uv":null,"visibility":"occluded","next_action":"inspect_cut_region"}`, an inconsistent pair the contract rejects.

### 3.1 Training runs

| Run | Method | Key settings | Time (4x H200) | Val loss per epoch |
|---|---|---|---|---|
| lora-01 | LoRA r16/a32 on language q/k/v/o | LR 1e-4, 1 epoch, eff. batch 32, DDP | 14 min | 0.221 |
| full-01 | Full FT (8.77B params), DeepSpeed ZeRO-2 | LR 1e-5 (LM/mergers/head), 1e-6 (vision), 3 epochs | 64 min | [0.228, 0.198, 0.191] |
| full-02 | full-01 + abstention fixes | + 2-dp coordinates (57 -> 36 answer tokens), inverse-frequency class weights (abstain 2.33 / localized 0.64), 3x weight on status tokens | 66 min | [0.384, 0.33, 0.306] (not comparable: format changed) |

### 3.2 Full-01 per-epoch checkpoints (why the decision looks like a shifting prior)

| Checkpoint | Abstain acc. (occluded) | Coverage (visible) | Median px error | Balanced acc. |
|---|---:|---:|---:|---:|
| epoch 1 | 0.000 | 1.000 | 31.0 px | 0.500 |
| epoch 2 | 0.688 | 0.347 | 18.5 px | 0.517 |
| epoch 3 (published v1) | 0.358 | 0.732 | 17.5 px | 0.545 |

Pixel precision improves monotonically; the abstain rate swings from 0% to 65% to 27% while balanced accuracy stays near 0.5.

### 3.3 Image-free shortcut baselines (validation)

| Baseline | Status acc. | False loc. on occluded | Coverage | Interval hit | Median error |
|---|---:|---:|---:|---:|---:|
| constant_pixel | 0.745 | 1.000 | 1.000 | 0.001 | 180.0 px |
| copy_query | 0.745 | 1.000 | 1.000 | 0.000 | 60.9 px |
| query_plus_train_median_offset | 0.745 | 1.000 | 1.000 | 0.000 | 60.7 px |
| always_abstain | 0.255 | 0.000 | 0.000 | 0.000 | n/a |

Always-localize beats every model on raw status accuracy (0.745), which is why balanced accuracy and false-localization rate are the metrics that matter.

## 4. Can the models drive robot motion in the simulator?

No. Three independent blockers, in order of severity:
1. **Abstention is unreliable.** The best model still localizes 52% of hidden cut points. The runbook's "false localization under occlusion" gate is not met.
2. **Precision is an order of magnitude short.** Typical error ~21 mm versus a 10 mm admissible interval; only 2.7% of answers hit it.
3. **By design the model emits no motion inputs.** It outputs a 2-D pixel and a status. Metric depth, grasp, blade corridor and collision checks belong to the separate guarded execution layer (plan sections 3-4), which must never use hidden simulator coordinates to rescue an abstention.

What it is good for now: a perception module in a human-checked loop (pointing a camera or a crop window at the right region), and a baseline for the next iteration.

## 5. Diagnosis

- **Status probe** (teacher-forced log P(abstain) - log P(localized)): full-01 AUC 0.784 on training plants but 0.576 on held-out plants; full-02 raises held-out AUC to 0.635. The decision is under-learned and weakly transferable; no threshold calibration can fix a 0.58-0.64 AUC.
- **Loss composition:** original answers spent ~32 of 57 tokens on coordinate digits with 15 decimals (unpredictable noise); rounding to 2 decimals and reweighting the status token was the first fix (full-02) and helped moderately.
- **Dataset** (section 6): the remaining gap is explained by target redundancy, identity-correlated occlusion, and a cue at the RGB resolution limit.

## 6. Dataset assessment and recommendations

### 6.1 Findings

- **Effective size:** train = 168 targets on 16 plants (median 60 views/target); validation = 39 targets on 4 plants; test = 39 targets on 4 plants. No duplicate camera poses.
- **Identity leakage into labels:** 10 of 39 validation targets are abstain-only; per-family abstain rate ranges 0.01-0.43. Family/target priors do not transfer.
- **Cue legibility:** in zoomed crops the labels are geometrically consistent, but hidden-vs-visible is an ~8 px interval next to a fruit or stem edge on 4-5 px petioles, often backlit. Sub-token at the model's 32 px image-token grid.
- **Class balance:** 78/22 localized/abstain; no medium/partial class at all (excluded by the `visible_occluded_v1` profile).
- **Labels:** automatic (geometry + native masks + camera-Z), 136 assistant QA accepts, no per-image human review; shared greenhouse backdrop; 24 source plants in total.

### 6.2 Recommendations (priority order)

1. **Spend rendering budget on targets, not views.** Cap ~10-15 views per target; grow target count toward the 870 audited candidates. New plant geometries (generator access) are the single most valuable addition; held-out splits have only 4 plants each.
2. **Make occlusion view-dependent per target.** Sample viewpoints so every target appears both visible and hidden where physically possible; ensure no held-out family has an abstain rate near 0 or above 40%. Balance at collection time, not by loss reweighting.
3. **Add the medium/partial class and the plan's other negatives** (invalid candidate, no target in region, out of view). Use `visibility: partial`. The 40-90% visibility band is where the decision boundary is learned.
4. **Export a deterministic query-centred crop as a second model input** (e.g. 2x zoom, 256 px). Deployable (the query is known at inference), matches the plan's full-frame-plus-crop comparison, attacks both abstention and precision.
5. **Evaluate per target and per family** (macro-average), not per row.
6. **Human-verify held-out occlusion labels** (a few hundred validation rows via the existing review GUI) to separate "unlearnable" from "not visible in RGB".
7. **Family-wise cross-validation** if more plants are unavailable.
8. **Richer supervision fields:** expose the occluder type (leaf/fruit/stem/none); keep `proximal_visible_pixel_fraction` for auxiliary supervision.
9. **Appearance variation:** lighting/exposure to reduce backlit low-contrast petioles; vary backdrop/stance.
10. **Keep native depth in the release** regardless of the RGB-D outcome (ground truth for occlusion labels).

Top three: more targets with capped views, view-dependent occlusion per target, query-centred crop.

## 7. In progress / next

- **full-03 (RGB-D):** full-02 recipe + native depth rendered as a second image (turbo inverse-depth 0.25-1.5 m) + query depth in the prompt. Queued to launch once all four GPUs have been idle for 3 minutes (currently occupied by RoboMeter LIBERO jobs). Chain runs smoke -> overfit -> reload -> 3-epoch train -> validation eval -> probe unattended.
- **Resolution experiment (full-04 candidate):** 2x upsampled input (~4x image tokens), ~3.5 h of exclusive GPU time.
- **Repository:** the trainer, evaluator, probe, depth-input and validator changes live in `/workspace/nhkoh/tomato-vlm/greenhouse_training_code` (from the v2 code zip) and are not yet committed to `koh-dev/sim-vlm`.

## 8. Engineering notes

- Validator: 11 train rows differed from the export machine by <= 3.6e-15 in one float (`hypot` rounding); `training_export.evidence_matches` now tolerates 1e-9, everything else exact (`runs/validation-platform-note.json`).
- Full fine-tuning: DeepSpeed 0.17.6 (>= 0.18 breaks on torch 2.6), ZeRO-2, ~81 GB/GPU, 3.3 s per 32-example step. Smoke/overfit for the full method run as 4-GPU ZeRO-2 because fp32 master weights do not fit one GPU.
- Each trainer invocation revalidates the release (~9 min) before touching weights.
- Environment: `/root/venvs/deleaf-qwen` (Python 3.11, torch 2.6.0+cu124, transformers 4.57.6, peft 0.17.1, accelerate 1.11.0). Runs and reports: `/workspace/nhkoh/tomato-vlm/runs` (`RESULTS.md`, `eval-*/report.json`, `probe-*/summary.json`, `image-free-baselines.json`).
