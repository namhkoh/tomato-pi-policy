# v2 data spec: single true cut point per image (agreed 2026-09-17)

Objective to verify first: given RGB image i and metric depth map d, how reliably can a VLM output the cut point (u, v)
(+ depth) of the single deleafable petiole in view, or abstain when its cut region is hidden. No query pixel.
Later extension: several visible cut points per image (answer becomes a list; scoring by matching).

## Requirements for the target to be well defined from the image alone
1. Exactly one leaf-bearing (deleafable) petiole is visible per frame. Every other petiole in view must be a deleafed stub or
   out of frame; the asset `deleafed` flags and the per-frame instance masks let the exporter verify this per frame.
   If a second leaf-bearing petiole is visible, the frame must be excluded (or the second one labelled, for the later variant).
2. Keep random framing (as in v1: the target's cut lands at a random pixel per sample), so position cannot stand in for identity.
3. Keep the occluded class: frames where the single target's cut region is hidden by a leaf/fruit/stem, label = abstain,
   at roughly the natural rate (v1: 22-25%), and view-dependent per target (no target that is occluded in every view).

## Per-frame contents (same layout as v1; the query fields become optional)
- `images/<id>.png`: unaltered 848x408 RGB.
- `depth/<id>.npy` float32 optical-axis Z in metres + `depth/<id>_valid.png`; camera intrinsics/extrinsics in the label.
- `labels/<id>.json`: `answer` (status, cut_point_uv, visibility, next_action), `nominal_optical_xyz_m`, `accepted_interval_uv`
  (10-20 mm polyline), `proximal_evidence`, difficulty; `labels/<id>_target.png` target petiole mask (scoring only).
- `index.jsonl`, `splits/*.jsonl` (system/user/assistant chats; the user turn no longer carries a query),
  `manifest.json` with gates and hashes, `exclusions.json`.

## Diversity (the v1 lesson)
- Targets over views: cap 10-15 views per target; aim for several hundred distinct targets across as many plants as possible.
- Family-disjoint splits (train/val/test by plant), as in v1. Report per-target macro metrics alongside per-row.
- No plant with an occlusion rate near 0 or above ~40%.

## What is ready on the training side
- `sim_data/hamster3d/` (3D HAMSTER path): `--no-query` in `hamster_train.py`; the evaluator and probe read the mode from the
  saved model contract (`query_pixel_given`). Prompt: "Exactly one leaf-bearing petiole in this view is the deleafing target..."
- Same metrics as v1: abstain accuracy / false localization / coverage / balanced accuracy / AUC; median error; within 5 px;
  10-20 mm interval hit; predicted-depth error; post-hoc native-depth gate.
- Mechanical check of the no-query path on v1 (smoke, overfit, reload): `runs/logs/chain-hamster-nq.log`.
