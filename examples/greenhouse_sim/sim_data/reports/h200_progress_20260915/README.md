# Qwen3-VL-8B cut-point localization: progress report

2026-09-15. Task `greenhouse.target_conditioned_cutpoint_rgb.v3`, release `visible_occluded_20260911_v1`, base Qwen/Qwen3-VL-8B-Instruct (`0c351dd`), 4x H200. Code on `koh-dev/sim-vlm`, `examples/greenhouse_sim/sim_data/`.

## 1. Data

Synthetic Isaac Sim greenhouse, robot-head D405 camera, 848x408 RGB. One example = image + query pixel on a visible petiole + JSON answer. Labels come from plant geometry, native masks and native depth; no per-image human review.

| Split | Rows | Plants | Targets | Localized | Abstain | Median views/target |
|---|---:|---:|---:|---:|---:|---:|
| train | 11,520 | 16 | 168 | 9,051 | 2,469 | 59 |
| validation | 1,358 | 4 | 39 | 1,012 | 346 | 37 |
| test (unused) | 1,381 | 4 | 39 | 1,174 | 207 | 27 |

- Label: nominal cut = 10 mm along the petiole from the main-stem attachment; admissible interval = 10-20 mm along the centreline (yellow in figures). Query pixel >= 45 mm from the attachment and >= 18 px from the cut.
- Classes: easy-visible (answer = cut point) and hard-occluded (cut region behind a leaf, fruit or stem in the native masks/depth; answer = abstain). Medium/partial occlusion excluded at export.
- Splits are disjoint by plant (16/4/4 of 24). Backdrop shared.
- Input text uses normalized coordinates x' = 1000u/848, y' = 1000v/408. Output: `status` (localized|abstain), `cut_point_uv` ([x',y']|null), `visibility`, `next_action`.

System prompt (all examples): "You localize a synthetic tomato petiole cut point in a robot-head RGB image. The user supplies a normalized query coordinate on the target petiole, NOT the cut point. Trace that petiole to its junction with the main stem. A localized target must be traceable through visible petiole pixels from the query to the cut region. The dataset convention places the nominal cut 10 mm along the petiole toward the leaves from its attachment; 10-20 mm is the evaluation interval, not a spherical tolerance. Use visual evidence. If the cut region is occluded or cannot be distinguished, abstain instead of inventing coordinates. Coordinates (query and cut_point_uv) are normalized continuous coordinates in [0,1000), top-left edge origin, x right and y down: x_norm=1000*x/848 and y_norm=1000*y/408 for the ORIGINAL 848x408 image, independent of processor resizing. Return ONLY JSON with exactly these keys: status (localized or abstain), cut_point_uv (a two-number normalized coordinate array or null), visibility (clear, partial, or occluded), next_action (inspect_cut_region or change_viewpoint). This is perception only; never claim that a blade motion is safe or executable."

User prompt template: "The target petiole passes through normalized coordinates (x', y'). Locate its nominal cut point if the junction and cut region are visually distinguishable; otherwise abstain. Use the original full image."

Samples (left: full frame; right: zoom of the white box; white = query, magenta = label cut, yellow = 10-20 mm interval):

![train_visible_1](figures/train_visible_1.jpg)

Visible: `seed71_full_5f84a906b9019192dc6d`, family seed71_full, query pixel [221.8, 28.3]. Target: `{"status":"localized","cut_point_uv":[277.71,460.05],"visibility":"clear","next_action":"inspect_cut_region"}`

![train_visible_2](figures/train_visible_2.jpg)

Visible: `seed47_full_4e395a3c012d162d460d`, family seed47_full, query pixel [642.2, 317.1]. Target: `{"status":"localized","cut_point_uv":[782.19,846.32],"visibility":"clear","next_action":"inspect_cut_region"}`

![train_occluded_1](figures/train_occluded_1.jpg)

Occluded (cut behind foreground; proximal petiole 0% visible): `seed89_full_479f23d1e840c24160b9`, family seed89_full, query pixel [709.6, 192.9]. Target: `{"status":"abstain","cut_point_uv":null,"visibility":"occluded","next_action":"change_viewpoint"}`

![train_occluded_2](figures/train_occluded_2.jpg)

Occluded: `seed101_full_79ac1e2a98ca713e5823`, family seed101_full, query pixel [582.4, 285.7]. Target: `{"status":"abstain","cut_point_uv":null,"visibility":"occluded","next_action":"change_viewpoint"}`

## 2. Training

Supervised fine-tuning: next-token cross-entropy on the assistant answer tokens only (prompt and image masked), mean per example; 32 examples per optimizer step (1 per GPU x 8 accumulation x 4 GPUs); BF16, cosine schedule, 3% warmup, clip 1.0, seed 41.

| Run | Trainable | LR | Epochs | Answer text | Loss weighting | Time |
|---|---:|---|---:|---|---|---:|
| LoRA | 15.3 M (r16 on language q/k/v/o) | 1e-4 | 1 | full precision, 57 tokens | none | 14 min |
| Full v1 | 8.77 B (DeepSpeed ZeRO-2) | 1e-5; vision tower 1e-6 | 3 | full precision | none | 64 min |
| Full v2 | 8.77 B | same | 3 | 2 decimals, 36 tokens | abstain x2.33, localized x0.64; status tokens x3 | 66 min |
| Full v3 | 8.77 B | same | 3 | 2 decimals | same as v2 | 136 min (shared GPUs) |

Full v3 adds the native depth map as a second input image (turbo colormap of inverse depth, 0.25-1.5 m) and the query depth in the prompt.

## 3. Evaluation

Validation split only; test never read. Greedy decoding, batch 1, same prompt as training. Answers are parsed, converted back to pixels, and count as failures if not valid JSON, outside the frame, or an inconsistent status/next_action pair; nothing is repaired.

Metrics: abstain accuracy = share of the 346 occluded rows answered `abstain`; coverage = share of the 1,012 visible rows answered `localized`; balanced accuracy = mean of the two; interval hit = answer within 2 px of the projected 10-20 mm interval; median error over answered visible rows (the projected 10 mm interval is 8.5 px long here, so 0.85 px/mm); AUC = ranking quality of the model's own log P(abstain) - log P(localized) at the status token, independent of decoding.

## 4. Results

| Validation (1,358 rows, 4 unseen plants) | Untuned | LoRA | Full v1 | Full v2 | Full v3 (RGB-D) |
|---|---:|---:|---:|---:|---:|
| Valid JSON answers | 0/1358 | 1358/1358 | 1358/1358 | 1358/1358 | 1358/1358 |
| Abstains correctly on occluded (346) | n/a | 0.113 | 0.358 | 0.483 | 0.425 |
| Fabricates a point on occluded | n/a | 0.887 | 0.642 | 0.517 | 0.575 |
| Answers on visible (1,012) | n/a | 0.915 | 0.732 | 0.713 | 0.672 |
| Balanced accuracy of the decision (chance 0.5) | n/a | 0.514 | 0.545 | 0.598 | 0.548 |
| Status-score AUC (chance 0.5) | n/a | n/a | 0.576 | 0.635 | 0.572 |
| Median error of answers (0.85 px/mm) | n/a | 27.8 px = 33 mm | 17.5 px = 21 mm | 18.0 px = 21 mm | 20.7 px = 24 mm |
| Answers within 20 px | n/a | 0.35 | 0.57 | 0.56 | 0.48 |
| Inside the 10-20 mm interval (2 px) | n/a | 0.024 | 0.020 | 0.027 | 0.020 |
| Latency p50, H200 | 0.73 s | 1.62 s | 1.07 s | 0.67 s | 0.82 s |

Untuned base: every answer is `abstain` + `inspect_cut_region`, an inconsistent pair, hence invalid. Image-free baselines: always-localize scores 0.745 status accuracy (above every model); copying the query pixel gives a 61 px median error; no baseline reaches the interval.

## 5. Failure points

1. **Abstention is not learned as a visual cue.** Best AUC 0.635 (v2) on unseen plants; on training rows v1 reaches 0.784. v1's checkpoints swing the abstain rate from 0% (epoch 1) to 65% of visible targets (epoch 2) to 27% (epoch 3) while balanced accuracy stays at 0.50-0.55: the prior moves, the discrimination does not. v2's class and status-token weights raise abstain accuracy 0.358 -> 0.483; v2 still fabricates a point for 52% of occluded targets.
2. **Answers miss the petiole.** Only 10% of v2's answers on visible targets land on the target mask; the typical miss lies ~1.4 m behind the plant on the background. Median error 18 px (21 mm) against a 10 mm interval; 2.7% inside it.
3. **Depth as a model input does not help** (v3 worse on every metric). **Depth outside the model does**: gating v2's answers by native depth at the predicted pixel vs the query pixel separates occluded from visible answers with AUC 0.686. A 3 cm "occluder in front" rule cuts fabricated points 0.517 -> 0.350 at coverage 0.630; a 5 cm depth-consistency rule cuts them to 0.092 at coverage 0.152, and the kept answers reach 12.6 px median error and 16% interval hits (4x).
4. **Data limits.** 168 training targets behind 11,520 rows (median 59 views each); 10 of 39 validation targets are abstain-only and per-plant abstain rates span 0.01-0.43, so identity predicts the label; the visible/hidden difference is an ~8 px interval beside a fruit or stem edge on 4-5 px-wide petioles, below the model's 32 px image-token grid; no medium-occlusion class.

Examples from v2 (cyan = prediction):

![eval_visible_good](figures/eval_visible_good.jpg)

Correct: `seed29_full_7571bd5dfa27e36d0841`. Error 0.4 px. Output: `{"status":"localized","cut_point_uv":[445.5,750.25],"visibility":"clear","next_action":"inspect_cut_region"}`

![eval_visible_missed](figures/eval_visible_missed.jpg)

Missed: point placed on the wrong structure: `seed97_full_f84d56f31548040c5d4f`. Error 46.7 px. Output: `{"status":"localized","cut_point_uv":[770.52,330.15],"visibility":"clear","next_action":"inspect_cut_region"}`

![eval_occluded_correct_abstain](figures/eval_occluded_correct_abstain.jpg)

Correct abstention: `seed97_full_e457cbe592bc6821f137`. Output: `{"status":"abstain","cut_point_uv":null,"visibility":"occluded","next_action":"change_viewpoint"}`

![eval_occluded_false_localization](figures/eval_occluded_false_localization.jpg)

Fabricated point: cut is behind the fruit: `seed97_full_d8cb1af6a81e1bc0f615`. Output: `{"status":"localized","cut_point_uv":[140.68,330.15],"visibility":"clear","next_action":"inspect_cut_region"}`

## 6. Artifacts

Models (private): [full-v2](https://huggingface.co/namhokaist/qwen3-vl-8b-tomato-cutpoint-full-v2), [full-v1](https://huggingface.co/namhokaist/qwen3-vl-8b-tomato-cutpoint-full), [lora](https://huggingface.co/namhokaist/qwen3-vl-8b-tomato-cutpoint-lora). wandb `tomato-pi`: `xcwqmy6m`, `5plv8gya`, `hmue3ag3`. Reports and per-example predictions: `/workspace/nhkoh/tomato-vlm/runs/`.
