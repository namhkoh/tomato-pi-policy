# Generated plant: native integration and output preview

## Status

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
