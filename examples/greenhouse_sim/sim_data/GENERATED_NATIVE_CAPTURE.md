# Generated plant: native integration and output preview

## Status

Update2026-09-15: the frozen native comparison completed and its four paired
camera/generated diagnostic images were individually assistant-reviewed.
See the completed-recovery evidence below; historical queue notes are retained.

Implemented: generated_capture.py / native_generated_pair.py, a frozen
two-frame original/generated diagnostic. This is not a production collector,
training release, new family-diversity approval or physical grasp/cut test.

The worker reconstructs the existing full greenhouse, recorded RB-Y1 camera
and robot pose, source plant placement and lighting. It replaces exactly one
foreground plant in an anonymous session. One native1696x816 render product
is reused, at the existing56-subframe budget per frame. No image upscaling.

Launch requires completed native848x408/1696x816 qualification on the exact
source sample, including native known-surface optical-Z and instance-ID
checks. Source/generated assets, copy-time textures and code are hash-bound.
The existing Windows memory reserve remains required. No unrelated Kit process
is allowed at startup; the controller must be the worker's actual parent.
This admission check is not a global process lock.

Capture validates camera calibration, robot/scene/lighting identity, unchanged
plant placement, robot/environment geometry, native target identity, changed
RGB/depth buffers and increasing writer callback, static-scene changes, no
old-plant pixels after substitution and unchanged source files.
No training approval is granted. The conservative12-view cap still belongs to
the ORIGINAL donor target. The generic pilot quality report is diagnostic only,
not the stricter clear-cut-point training acceptance gate.

## Measured CPU evidence

Artifact under data/sim_data/diagnostics:
generated_native_pair_cpu_rehearsal_20260915_v1.json.

Actual seed7 assets:421 components preserved, same recorded camera and plant
placement. SubStem_42 nominal cut point moved2.087127mm and4.662410px at1696x816.
Both projections remain in frame. This rehearsal did not load the complete
greenhouse, render native images or qualify visibility.

Frozen plan:diagnostics/generated_native_pair_plan_20260915_v1.json.
Reference:collection_campaigns/clear_capture_20260915_orbit_v2/job_021/job_021/capture/sample_0001.
Generated:generated_plants/petiole_similarity_pilot_20260915_v2/seed7_full_pv_2b7747b1cdfff86d.
All three paths are under data/sim_data.

## Actual output preview

plant_variant_preview.py renders the authored USD triangles using shared
orthographic views and flat organ colors. It does not use textures, native
depth or physically based lighting. Triangle sorting approximates occlusion.
This isolated engineering illustration is NOT training input or physics proof.

Preview and provenance under data/sim_data/diagnostics:
generator_visual_preview_20260915_v1/plant_comparison.png and preview.json.

Original/generated each497,579 fan-triangulated faces and421 components.
Eight selected petiole/leaf subtrees are blue in BOTH panels for comparison.
Unchanged organs are green; fruit/flowers have separate flat diagnostic colors.
SubStem_42:scale0.90, azimuth+15deg, tilt+12deg.

Orange dot:attachment. White cross:recomputed nominal10mm point.
Magenta:recomputed10-20mm centerline interval, NOT a spherical cutting radius.
The close-up uses the same view and includes a10mm scale bar.

From the repository in PowerShell, set PYTHONPATH to
'examples;examples/greenhouse_sim' and OPENBLAS_NUM_THREADS to1, then run:

    python -m sim_data.plant_variant_preview --variant data/sim_data/generated_plants/petiole_similarity_pilot_20260915_v2/seed7_full_pv_2b7747b1cdfff86d --source-plan data/sim_data/collection_plans/clear_capture_20260915_orbit_v1/plan.json --output data/sim_data/diagnostics/NEW_generator_preview --target SubStem_42

Requires NumPy, USD and Matplotlib. New output directory only; no Isaac app or
robot connection is opened. Source layers/files remain unchanged.

## Serial native queue (2026-09-15)

Operational controller under data/sim_data/diagnostics:
generated_native_pair_queue_20260915_v1.ps1 and corresponding Python file.
Log:data/sim_data/generated_native_pair_queue_20260915_v1.log.

The queue waits for the prior camera controller's successful exit and no Kit
processes. Six-hour waiting deadline;1800-second owned-worker budget.
No automatic retry, training or unseen-image acceptance. On timeout it stops
only its own worker. Existing collection, frozen splits and reviews are intact.
Separate code/plan pin:generated_native_pair_code_binding_20260915_v1.json
(186 files); all168 previous native code pins were verified unchanged.

At checkpoint:GENERATED_PAIR_WAITING_FOR_NATIVE_CAMERA_QUALIFICATION.
Expected output, NOT YET CAPTURED:
data/sim_data/diagnostics/generated_native_pair_20260915_v1.

After capture inspect both clean RGBs, overlays, instance masks and native
depth before extending to the100-200-view pilot. No generated image currently
counts toward the20,000 accepted TRAINING-image requirement.

Final regression:918 tests plus77 subtests in67.43s, recorded in
data/sim_data/clear_regression_20260915_v22.log. CPU tests and the preview
do not substitute for the pending native generated-scene validation.

## Completed native comparison and review (2026-09-15)

The original waiting queue could not proceed after supplemental collection
failed the Windows memory gate. Only that exact stranded helper was stopped.
Its cancellation receipt is
data/sim_data/diagnostics/generated_native_pair_queue_20260915_v1.cancelled.json.
After the user freed memory, the new bounded recovery queue completed both
native diagnostics; all191 source/code/plan pins were unchanged.

Native output: data/sim_data/diagnostics/generated_native_pair_20260915_v1.
Worker exit0/no timeout in145.235s including startup/shutdown. Capture function
elapsed128.187s; setup22.800s; control/generated screen-and-capture51.409/48.896s.
These two fixed diagnostic frames are not production throughput measurements.

Same mounted camera, robot pose, lighting, original plant placement and scene
population verified. Replaced exactly one plant in the anonymous stage. New
native target identity and changed RGB/depth buffers were observed; no original
foreground-plant pixels remain after substitution. The control's native depth
hash matches the earlier high-resolution sensor pair.

Original/generated nominal points: (1397.9906,240.1116) and(1398.2220,235.4550).
Generated estimated petiole diameter14.5832px,10-20mm interval31.5432px, native
target mask4,653px; all11 sampled interval points have matching target/depth.
Sampled native optical-Z ranges0.34433-0.34990m on the generated interval.
Pilot numerical visibility is NOT the stricter clear-training acceptance gate.

The original and generated full RGBs, native-resolution junction crops,
annotations, generated target mask and four depth heatmaps were inspected.
The visible attachment is continuous and the recomputed nominal marker stays
on the petiole. Foliage orientation changes visibly; protected parent geometry
and surrounding plants remain. This single view cannot certify hidden
self-intersections, botany, physical cutting or target-diversity novelty.

Hash-bound QA, review decisions and depth heatmaps:
data/sim_data/diagnostics/native_camera_generated_review_20260915_v1.
The four decisions are pass_visual_diagnostic_only, not training accepts.
Generated training images in the current checkpoint remainZERO; these diagnostic
schemas are not fed into the hard-coded848x408 release. No query-conditioned
higher-resolution training record or20k ZIP has been published.

Next: broaden the native pilot and qualify resolution-aware query labels,
strict clarity review and export. Retain the ORIGINAL donor target's view cap
until morphology novelty and near-duplicate grouping are explicitly validated.
