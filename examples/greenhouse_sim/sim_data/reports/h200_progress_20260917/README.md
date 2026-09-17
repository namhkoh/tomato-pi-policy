# Progress deck, 2026-09-17

`deck.html` is a self-contained 12-slide presentation (figures embedded; prints one slide per page):
task, data collection pipeline, release statistics and exclusion funnel, training examples, training runs
(LoRA, full v1/v2/v3, 3D HAMSTER hamster-01), evaluation protocol, results across all models, RGB vs depth-aware
comparison, hamster-01 example outputs, failure points with re-collection priorities, artifacts.
All numbers are read from `runs/eval-*/report.json`, `runs/probe-*/summary.json`, `runs/image-free-baselines.json`
and the release `manifest.json`. `figures/` holds the source images.

`model_outputs.html`: the evaluation task in one paragraph, an 8-tile contact sheet of hamster-01 outputs by outcome
category (A correct, B near miss, C wrong structure, D wrongly abstained, E correct abstention, F fabricated point),
outcome frequencies for the RGB and depth-aware models, the issues in order of severity, and the same four frames
predicted by the untrained model, the RGB fine-tune and the depth-aware fine-tune. Sheets in `figures/contact_*.jpg`,
selections and frequencies in `figures/contact_*.json` and `figures/outcome_frequencies.json`.

`labels.html`: how a label is built (petiole centreline, 10 mm cut point, 10-20 mm interval, query pixel, visible vs
occluded decision), annotated training and validation frames with the target petiole mask, and what the model receives
versus what it is scored against. Figures in `figures/label_anatomy_*.jpg` and `figures/model_input_vs_supervision.jpg`.
