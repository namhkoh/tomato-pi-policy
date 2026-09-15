# Native high-resolution clear annotations

This is a separate, versioned **annotation pilot**, not a production training
release. Task: `greenhouse.native_clear_cutpoint_rgb.v1`.
The old 848x408 task, releases, reviews and training loader are unchanged.

## Implemented

- `native_query_visibility.py`: explicit native dimensions, exact target
  connectivity, minimum support, brightness and local contrast. No gap filling.
  Its legacy-resolution measurements match the old query checker.
- `native_clear_contract.py`: native 1696x816 RGB, clear-image screen, query-only
  768x768 crop with **no resizing**, and paired training/inference prompts.
  Normalized coordinates are relative to the original full image, not the crop.
- `native_clear_labels.py`: recompute the 10 mm nominal and 10-20 mm arc interval
  from the actual source/generated anatomy. Verify captured world/pixel labels,
  native component identity and optical-Z, visible parent context, and a visible
  distal query connected to the cut. Occluded/ambiguous cases are excluded;
  hidden coordinates never become an executable answer.
- `native_clear_annotation.py`: hash-check a completed original/generated native
  pair, derive both labels, copy unchanged RGB/depth/validity/masks to a NEW
  diagnostic output, and save model-input descriptions and review-only overlays.
  Existing source files, frozen donor-family splits and approvals are untouched.

The physical convention remains 10 mm from the attachment along the petiole;
10-20 mm is an arc segment, **not a 1-2 cm spherical radius**.
A valid RGB label is not evidence that a gripper or knife can execute it.

The new profile uses the same *absolute native-pixel* legibility thresholds:
12 px interval, 8 px width proxy, at least 95% proximal visibility, all sampled
interval points on the native petiole, local median luma at least 40/255 and at
most 10% local darkness. Higher resolution can resolve a smaller angular target.
These thresholds are explicitly versioned; they do not retroactively change
the old profile. Native query checks retain their original absolute pixel
thresholds; distal query remains at least 45 mm along the petiole and 18 px from
the nominal. Numeric checks remain proxies requiring visual review.

## Actual first-pair evidence

Output: `data/sim_data/diagnostics/native_clear_annotation_20260915_v1`.

| Record | Distal query, native pixels | Nominal cut, native pixels | Width proxy | Interval |
|---|---|---|---:|---:|
| Original control | (1580.1, 248.9) | (1397.99, 240.11) | 16 px | 30.94 px |
| Generated plant | (1654.4, 208.5) | (1398.22, 235.45) | 14 px | 31.54 px |

Both passed numeric checks and subsequent individual assistant annotation
review. The query is visibly on the petiole and traceable to the cut region in
these two cases. This is **one target/pose, one donor and one generated layout**,
not general anatomy validation or two new independent training targets.

Review decisions:
`data/sim_data/dataset_reviews/native_clear_annotation_20260915_v1/assistant_reviews.json`.
They are `accept_annotation_pilot_only`, not production training approvals.
The original control is a repeated control; RGB render noise does not create
new source diversity. The original donor target still owns the 12-view cap.

The native depth is saved Isaac `distance_to_image_plane`, optical-axis Z in
metres. It is neither reconstructed nor supplied to the RGB model. The native
1696x816 mode is an ideal-simulator pinhole configuration, not validation of a
physical D405 sensor mode.

## Model input and output

Each record has unchanged `inputs/rgb.png`, native `inputs/query_crop.png`,
`label.json`, `model_inputs.json`, and separate `annotation_review.png`.
Cyan marks the supplied query; white is nominal cut; magenta is the evaluation
interval. **The model receives none of those colored overlays.**

`model_inputs.json` records both RGB-only and full-RGB-plus-native-crop prompt
variants. It is an inspectable contract, not yet a production H200 dataset.
System instructions specify the anatomy task, abstention, original 1696x816
frame, and normalized [0,1000) coordinates. The generated example's user prompt:

> The target petiole passes through normalized coordinates (975.47, 255.51).
> Locate its nominal cut point if the junction and cut region are visible;
> otherwise abstain.

The supervised answer is:

```json
{"status":"localized","cut_point_uv":[824.42,288.54],"visibility":"clear","next_action":"inspect_cut_region"}
```

For crop mode, the prompt also gives the full-frame pixel bounds
[928,0,1696,768] and explicitly requires output in the full image's normalized
coordinates. Crop selection depends only on the query, never the cut label.

Actual training/inference input images and text match. Maximum normalized
two-decimal answer round-trip error was 0.00584 px on the control and 0.00368 px
on the generated record. No HF processor, image-token grid, model weights,
training speed or localization accuracy has been tested by this adapter.

## Reproduce without launching Isaac or training

Run from the repository with the ordinary NumPy/Pillow/SciPy/USD environment.
Choose a NEW output directory:

```powershell
$env:PYTHONPATH='examples;examples/greenhouse_sim'
$env:OPENBLAS_NUM_THREADS='1'
python -m sim_data.native_clear_annotation --plan data/sim_data/diagnostics/generated_native_pair_plan_20260915_v1.json --capture data/sim_data/diagnostics/generated_native_pair_20260915_v1 --output data/sim_data/diagnostics/NEW_native_clear_annotation
```

This requires an already completed, matching, provenance-verified native pair.
It cannot relabel an arbitrary image as a native sensor capture.

Regression: 28 focused tests; full suite **954 tests plus 77 subtests** passed
in 73.10 s (`data/sim_data/clear_regression_20260915_v24.log`).
Tests include stale geometry, wrong dimensions, foreground depth, missing
target/parent, darkness, disconnected queries, native crop equality, answer
separation and coordinate boundaries.

## Broader pilot and remaining work

Three additional TRAIN donors (seed101, seed103, seed17) were selected from
previously reviewed original views, with generated target geometry from the
frozen library. The bounded serial native pilot is:
`data/sim_data/diagnostics/native_generated_broader_pilot_20260915_v1`.

It runs same-camera resolution qualification, original/generated capture and
new annotations per case. One native renderer, 1800 s worker deadlines,
18 GiB queue admission plus the existing native memory gate, no automatic
retry and no automatic visual acceptance. It binds 202 source/code/plan files.
Log: `data/sim_data/native_broader_pilot_20260915_v1.log`.
Execution started 11:53 KST; inspect receipts for actual completion/failures.

At the next checkpoint, case001/seed101 passed its original paired-camera test
but FAILED the robot/environment geometry screen after generated-plant
substitution. No generated image/annotation was approved for this case.
The failure and original-control image are retained; no clearance threshold was
disabled and the case is held. This is a screen failure, not proof of measured
physical damage. Source case plans and all202 pins remain unchanged.

Only the unstarted seed103/seed17 cases were explicitly continued at12:07 KST.
Continuation receipts:
data/sim_data/diagnostics/native_generated_remaining_cases_20260915_v1.
Log: data/sim_data/native_remaining_cases_20260915_v1.log.
The failed seed101 case is not retried or reclassified. This is why generated
geometry needs new scene-clearance checks and eventually variant-aware camera
sampling before mass collection.

Subsequent inspection: seed103 also stopped at the same generated-stage
clearance screen; seed17 remains unstarted. Source inspection found that the
triangle refiner currently recognizes only /World/PackPlants, not the generated
/World/GeneratedNativePilot root. Generated foliage therefore falls back to
coarse enclosing boxes. This is a confirmed integration gap; whether it caused
each failed native case still requires an explicit corrected native run.

Still required before the 20,000-training-image release:

1. Broader native anatomy/query review and sampling yield measurements.
2. Versioned portable native-resolution export, strict release validation,
   Qwen training-loader/evaluation integration and H200 processor/token checks.
3. Meaningful target-geometry novelty and near-duplicate grouping; generated
   seeds or renamed donors must not reset the original view cap.
4. Sufficient accepted TRAIN images plus separately reserved held-out sets.
   Current two-record pilot does not change the reviewed legacy image count.

No 20k ZIP, training authorization, policy actions or physical-cut qualification
is produced by this annotation pilot.
