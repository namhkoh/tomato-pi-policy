# Native mounted-head multiview pilot

This is static data capture, not robot navigation, manipulation, or a released
20k training dataset. Original source-family splits and source-target 12-view
caps remain; rendering more views does not establish independent morphology.

## Implemented and measured (2026-09-15)

`native_scene.py` shares the original full-greenhouse setup with the paired
collector. `native_generated_views.py` reuses a single stage, render product,
native writer and exact obstacle-geometry cache. All greenhouse populations,
lighting, meshes, mounted camera optics and native 1696x816 output are retained.
Depth is Isaac `distance_to_image_plane` float32; it is not reconstructed.

`native_view_plan.py` binds source/code hashes and generates up to six bounded
snapshots per target. `native_view_pose.py` preserves the already screened
body heading, all arm/torso joints, and camera mount; solves actual head joints;
and rechecks floor placement, URDF limits, camera calibration and static
robot/scene triangle clearance. No trajectory between snapshots is certified.
Continuous framing locations avoid a constant-centre cut-answer shortcut.

Early whole-body target-facing rotation put stowed arms in foliage and was
rejected. Preserving reference heading fixed this. Seed103 remained visually
occluded from the bounded local views; those four captures are not easy examples.

| Run below data/sim_data/diagnostics | Captured | Preliminary eligible | Automatic clear | Geometry rejected |
|---|---:|---:|---:|---:|
| native_multiview_seed17_20260915_v2 | 5 | 5 | 4 | 1 |
| native_multiview_seed19_20260915_v1 | 4 | 4 | 3 | 2 |

Each run has plan.json and capture/result.json with per-view decisions.
All nine were actually inspected using full-image overviews and unscaled
junction/query crops. The two local-contrast holds had exposed, continuous
foreground petioles and zero native identity/depth gaps. Append-only decisions:
`data/sim_data/dataset_reviews/native_multiview_20260915_v1/assistant_visual.json`.
These add nine annotation-pilot candidates (41 total separately from the legacy
450-image draft), NOT nine final release approvals. Global duplicates and caps
still apply, including V1/V2 geometry alternatives.

Runs took 209.30 and 179.80 seconds inside the app. Each built the obstacle
cache once with five hits. Additional captured views took about32 seconds;
all seven eight-subframe callbacks were retained. This is not adequate20k
throughput yet. SDK WriterSyncGate cycle warnings persisted; native callback,
camera-pose, changing RGB/depth and static-scene guards passed, but engine frame
IDs are explicitly NOT verified. Do not represent this as dynamic synchronization.

Full regression:1132 tests +87 subtests,57.93s;
`data/sim_data/clear_regression_20260915_v39.log`.

## Storage evidence, not a production codec

Ignored prototypes measured reversible compression of native sidecars on three
saved captures. Byteplane shuffling followed by deflate restored every original
file SHA256, dtype, shape and raw float bits (including NaNs). RGB was unchanged.
RGB + compressed sidecars project to roughly83-112GiB for22k images from these
three samples; this is not a measured full-dataset size. Final archive duplication,
assets and disk reserve must be budgeted too. No old captures were deleted.

Reports: `data/sim_data/diagnostics/native_byteplane_storage_20260915_v1/report.json`
and `native_lossless_storage_20260915_v1/report.json`.

## Remaining scaling work

Qualify consolidated native rendering against repeated full-budget references;
amortize multi-target generated plants; implement global geometry/near-duplicate
admission and lossless native storage; capture diverse TRAIN and separate held-out
families; export portable RGB/crop records and verify processor parity on H200.
No local weights or training, no source/review/split mutations, no final ZIP yet.

