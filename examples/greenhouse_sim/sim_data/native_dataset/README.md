# Native high-resolution dataset support

This package is separate from the legacy 848x408 release/export path. It does
not change active capture defaults, frozen splits, existing reviews or model
training. Native captures are 1696x816 mounted-head RGB with native Isaac
optical-Z and exact rendered identity buffers.

## Implemented and tested

- `audit.py`: replay completed static annotations from raw or compact storage.
  Authenticate the recorded request/plan, source/prerequisite/code bindings,
  planned candidate membership/uniqueness, sample identity, target ancestry,
  robot/screen metadata, original callback hashes, labels and query traces.
  Receipts bind the loaded audit code. Failed/incomplete captures and duplicate
  result rows cannot inflate counts. Successful replay is NOT release approval.
- `bundle.py`: create-only compact copies and a logical reader. Original RGB
  PNG bytes and whole native NPY files, including NaN payloads/signed zeros,
  remain exact. Review-only PNGs are explicitly omitted from copies. Sources
  remain unchanged. Required observations, external sample/label/trace bindings,
  bounded reads and integrity checks are mandatory.
- `capture_storage.py`: write the same compact format directly from validated
  native callback arrays, without staging raw NPYs or drawing review overlays.
  It validates shapes/dtypes and callback fingerprints, but does not replace
  the collector's native camera/scene/visibility checks. Native integration is
  an explicit diagnostic trial, not yet a production-default change.
- `admission.py`: pure candidate accounting. Keep original donor ancestry and
  frozen splits; finalize transitive global context duplicate groups; reject
  exact decoded-RGB, repeated scene/camera, near-image and cross-split duplicates;
  apply a shared per-context view cap and deterministic donor/target balancing.
  Require 20,000 TRAIN candidates plus separately specified validation/test.
  Geometry/image calibration and inventory evidence are externally supplied,
  not verified or invented by this accounting module. All results remain
  `training_approved=False`; the actual admission adapter/calibration and final
  release validator are still required.

## Evidence, 2026-09-15 checkpoint

Full sim-data suite: **1346 tests +181 subtests**,99.15s:
`data/sim_data/clear_regression_20260915_v46.log`.

Direct compact-writer CPU qualification on seven ACTUAL prior native captures:
21 NPY and28 PNG files byte-exact,126 original source files unchanged.
162,793,157 ->38,901,025 bytes. These are storage replays, not seven new images.
`data/sim_data/diagnostics/native_direct_writer_qualification_20260915_v1/qualification.json`.

Raw-to-compact copy plus final request-bound anatomical replay:
`data/sim_data/diagnostics/native_compact_sample_qualification_20260915_v2/final_bound_audit.json`.
All seven labels replay;2 strict candidates,2 clarity holds,3 exclusions.

Native diagnostic worker:
`data/sim_data/diagnostics/native_compact_wider_worker_20260915_v1.py`.
Only storage changes from the guarded wider-camera worker; full greenhouse,
materials, actual mounted head camera, native optical-Z/IDs, memory reserve,
static synchronization and reference56/warm56_then8 profiles remain intact.
Request/result bind the worker and storage implementations. Check
`diagnostics/native_compact_wider_seed19_20260915_v1/capture/result.json`
and absence of `failure.json` before claiming native qualification.
Repeated matched views contribute ZERO additional training diversity.

Isaac Python3.12 lacked Zstandard. The tested0.23.0 cp312 wheel is isolated in
`data/sim_data/native_capture_deps_py312_v1`; only the diagnostic native worker
adds it to PYTHONPATH. Isaac's installation and system packages are unchanged.
The CPU host's Python3.13 environment must NOT import the cp312 wheel.

## Remaining release gates

### 2026-09-15 native similarity measurement checkpoint

`near_image.py` compares unchanged native RGB using full-frame MAE and a complete
129x129 patch centred on each image's nominal cut location. Patches use integer
indexing only; no resizing, interpolation, padding or brightness correction.
Hidden coordinates are offline deduplication supervision, never model inputs.
The combined score is the maximum of full-frame and local MAE, normalized to
0..1. Both measurements and percentiles remain available. This metric is not
an automatic novelty gate. Its 42 focused tests pass.

Read-only actual-image diagnostic:
`diagnostics/native_rgb_controls_20260915_v1.json`.
21 matched-render duplicate pairs have maximum score0.039133; one visually
adjudicated curved-geometry pair scores0.347364. All22 pairs were measured at
1696x816 with bound source evidence. Only two donors are represented, and only
one positive control exists: no threshold is selected or globally calibrated.
No source image, depth, review or split was changed; controls add zero diversity.

The 18-plant native campaign completed212 frames/65 strict automatic candidates
(15 native jobs,3 invalid geometry attempts). Four further actual full-view and
unscaled crop inspections were recorded, bringing the overlapping visual pilot
to69 pending global admission. Receipt:
`dataset_reviews/native_diverse_generated_visual_20260915_v2/assistant_visual.json`.

The reviewed-reference continuation is now natively exercised, not merely queued:
first fresh seed101 sensor pair passed; first two generated jobs captured19
frames/8 strict candidates with independent annotation replay. First job timing:
29.66s scene setup,132.44s total,40.66s first render,3.92-4.27s later render calls.
These are collection measurements, not simulator interaction or training speed.
Existing full environment, native sensing and quality gates are unchanged.

1. Calibrated global morphology/image novelty admission; provenance-backed
   view inventories and donor-balanced sampling. No per-name source-cap reset.
2. Better reference-bank/pose selection and bounded generation qualification;
   measure useful reviewed images/hour, not raw renders or diagnostic repeats.
3. Native compact capture qualification and portable export/loader integration.
4. Held-out donor capture after TRAIN-only calibration is frozen.
5. Native-resolution Qwen3-VL-8B export, H200 processor validation, final counts,
   leakage/duplicate checks, dataset card and transfer ZIP.

There is **no final 20k TRAIN release or ZIP** at this checkpoint. Static
cut-point localization examples are not action demonstrations, dynamic episodes,
physical cut approvals or evidence of a trained policy.
