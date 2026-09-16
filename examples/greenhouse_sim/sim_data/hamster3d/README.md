# 3D HAMSTER bridge experiment (2026-09-16)

Fine-tunes the released `DAVIAN-Robotics/3D_HAMSTER` checkpoint (Qwen3-VL-8B + frozen LingBot-Depth encoder + geometry merger,
arXiv 2606.31329) on the task-v3 cut-point release in HAMSTER's own prompt/answer format (RGB + metric depth in, `point_3d` out,
`null` point = occluded). Requires the 3D_HAMSTER package (`pip install -e` of the upstream repo) and its checkpoint; these scripts
were run from `/workspace/nhkoh/3D_HAMSTER/agro/` with the `hamster3d` package importable.

- `cutpoint_data.py`: task adapter (prompt, answer text, 640-px preprocessing, supervised encoding, token weights, output parser).
- `hamster_train.py`: ZeRO-2 full fine-tune (encoder frozen, vision tower 0.1x LR), class weights, 3x status prefix; smoke/overfit/train.
- `hamster_evaluate.py`: sharded greedy generation, canonical-pixel scoring via `sim_data.h200_evaluate._metrics`, depth metrics, status probe.
- `launch-hamster-01-chain.sh`, `compare_hamster.py`, `real_model_check.py`, `zero_shot_cutpoint.py`: chain, comparison, checks.

Result (validation, vs full-02): balanced decision accuracy 0.598 -> 0.755, probe AUC 0.635 -> 0.884, median error 18.0 -> 13.6 px,
within 5 px 0.047 -> 0.125. See `../H200_RESULTS.md` history and `runs/RESULTS.md` on the server.
