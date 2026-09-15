# Native capture scaling experiments (2026-09-15)

Scope: static, robot-mounted-head RGB/Isaac optical-Z collection in the full
original greenhouse. No physical robot commands, local VLM weights/training,
frozen-split changes, inherited reviews, or final training-release approval.

## Current implementation

- Persistent single-plant, multiple-target/multiple-view capture now shares
  stage, writer, render product, native component catalogue and obstacle cache.
  Plans are create-only and bind all original/generated source and code hashes.
- Shared sensor qualification covers the exact mount/optics/native annotators,
  **not** visibility or safety of another target. Each pose still solves real
  head joints and validates floor alignment, joint limits, static robot/plant
  geometry, rendered calibration, fresh native buffers, target identity/depth
  and query-to-cut trace.
- The first multi-target launch exposed a USD ABI collision: CPU geometry replay
  imported pip USD before SimulationApp loaded Kit USD. Hash-only native
  preflight now runs first; full geometry replay is mandatory after app startup,
  before any capture. Failure remains under
  `data/sim_data/diagnostics/native_multitarget_seed17_full_20260915_v1/capture`.
- Two create-only plant assets each contain four changed, CPU-qualified petioles:
  `data/sim_data/generated_plants/curved_multitarget_20260915_v1`.
  No source plants, component populations, UVs or textures were removed.
- Corrected native plans have12 view proposals per donor:
  `data/sim_data/diagnostics/native_multitarget_seed17_full_20260915_v2`
  and `native_multitarget_seed19_full_20260915_v2`. Check capture/result.json and
  per-view decisions for completion. Queued/running is not accepted data.

## Negative and positive performance results

One consolidated56-subframe call did NOT provide the hoped-for throughput.
On seed17 it took24.02-27.53s versus25.80-30.16s for seven8-subframe calls.
Both short/reference and repeated-reference RGB differ. Resolved identity
differences on the first pose involved3-106 pixels of distant gutter hanger
cylinders, zero cut-region pixels; the second pose had identical identities.
Native optical-Z agreed within0.2mm throughout. None of this approves a faster
profile or establishes pixel-identical RGB. Diagnostic images/metrics:
`data/sim_data/diagnostics/native_render_probe_seed17_20260915_v2/capture`.
Initial seed17/seed19 failed probes are retained, not counted as training data.
The probe now retains failed measurements instead of discarding the evidence.

A CPU profile records substantial scene setup/audit/filesystem work but does
not reliably attribute all C++/callback wall time. Do not call it proof of the
remaining renderer bottleneck. Two-frame reference wall time120.34s; later
view32.91s. Evidence: `data/sim_data/diagnostics/native_cpu_profile_20260915_v1`.
Sampled GPU load during another render interval was3-13%, not sustained
full-run utilization and not a hardware-capacity guarantee.

The high-resolution fast-instance adapter is independently implemented without
changing the legacy848 decoder. Same-callback comparison passed all1,383,936
pixels and all observed prim paths on both saved1696x816 poses, with guards on
every callback. Full anatomical catalogue and all visible identity mappings
remain; only unobserved renderer map entries are omitted. Evidence:
`data/sim_data/diagnostics/native_fast_compare_seed17_20260915_v1/capture`.

Fast-only two-frame capture completed113.96s (setup29.21s; later view32.88s).
This does NOT demonstrate a useful steady-state speedup over legacy32.91s.
It does establish native pixel-identity compatibility and more compact mappings.
Evidence: `data/sim_data/diagnostics/native_fast_only_seed17_20260915_v1/capture`.
All56 requested subframes, scene meshes, renderer and optics were unchanged.

The next relevant render-budget experiment is native-resolution
warm56_then8. It was task-qualified at848x408 only (TRAINING_DATASET.md);
that evidence must not be silently applied at1696x816. Native target/depth,
annotation, visual-detail and matched-reference validation are still required.

## Diversity experiments, not production admission

784 bounded recipes from198 source targets were explored without making images.
A normalized petiole-only descriptor collapses almost all of them at coarse
thresholds and ignores actual leaf/attachment context. It is insufficient as
the entire global duplicate key.

A source-parent-frame signature additionally includes relocated attachment,
curve/radius relative to the donor length, parent segment, and each rigid leaf's
attachment, centroid, axis and intrinsic covariance scales. Exploratory L-infinity
thresholds0.025/0.05/0.10/0.15 keep784/784/781/769 of these784 recipes.
These are sensitivity results, NOT a validated admission threshold, new biological
families, or approval to reset old source-target caps.
Reports under `data/sim_data/diagnostics/curved_diversity_inventory_20260915_v1`
and `curved_context_inventory_20260915_v1`.
Production tests must cover rename/rigid-transform/uniform-scale invariance,
replay consistency, duplicates across batches and actual image similarity.

## Storage and dataset accounting

Lossless byteplane/bitplane/integer-bit-predictor experiments all restored exact
original native file bytes, including float32 depth and NaN bits. No depth was
recomputed. Bitplanes were larger; Zstandard9 byteplanes were generally smaller
than the tested fast Zstandard3 alternatives. These remain experiments, not an
integrated production codec. RGB stayed unchanged. Three-frame size projections
are not release-size guarantees and exclude assets/final archive overhead.
See native_*storage_20260915_v1 reports under data/sim_data/diagnostics.

41 actually visually accepted native annotation-pilot candidates remain separate
from the legacy450 reviewed draft (312train/48validation/90test). Recent renderer
and annotator diagnostics repeat existing target/views and add no diversity.
Final global deduplication/caps, new native held-out capture, portable Qwen3-VL-8B
export/processor validation and20,000 accepted TRAIN records +held-out sets are
still outstanding. No final training ZIP exists for that requested release.

Full regression after startup fix/adapter:1176 tests +87 subtests passed in70.33s,
`data/sim_data/clear_regression_20260915_v41.log`.

## Completed multi-target native checkpoint

Corrected seed17 batch:7 captured/7 label-eligible/2 automatic-clear/5 geometry
rejections,284.31s inside collector, one obstacle-cache build and11 hits.
All7 actually inspected with full overviews and unscaled native crops, including
five contrast holds. Accepted as annotation-pilot candidates, not release data.
Receipt: data/sim_data/dataset_reviews/native_multitarget_20260915_v1/seed17_assistant_visual.json.
Native visual-pilot total48 separately from legacy450. Fruit behind some targets
was visually distinguished from the foreground petiole; it is not a fruit label.
Corrected seed19:7 captured/4 label-eligible/2 automatic-clear,284.27s; visual
review pending. No failed/occluded frame is counted as an accepted easy example.

Added explicit --render-budget warm56_then8_trial for matched high-resolution
experiments ONLY. Reference56 remains the default. The first actual capture
retains56 requested subframes; later captures request8, with native callback,
geometry, optics/depth and label guards unchanged. Each sample records actual
budget and orchestrator request count. No high-resolution qualification or
training approval is granted by this option. Combined render experiments are
rejected. New targeted suite78 tests passed; broader trial qualification pending.
## 2026-09-15 matched short-budget and broader-campaign checkpoint

Measured matched high-resolution experiment:
-14 actual1696x816 frames, two TRAIN donors, same exact robot/camera poses,
 optics, scene geometry, lighting configuration and target labels.
-12 frames used8 requested subframes after one56-subframe warmup per scene.
 Median later-view time33.066s reference ->5.935s trial; complete collector
 batches284.31->122.88s and284.27->119.12s, including setup and rejects.
-All native optical-Z values agree within0.2mm, validity masks and exact target
 masks agree, zero cut-ROI identity changes, eligibility and output answers match.
-Three query pixels changed because RGB-dependent query selection changed.
 RGB is NOT pixel/photometrically equivalent: later-frame global mean absolute
 differences5.09-7.44/255; target-region differences are reported, not suppressed.
-Five representative short-frame full views and native crops actually inspected
 (three seed17 targets; two seed19 targets). Detail remained usable. This is
 limited pilot evidence, not a universal profile qualification or20k approval.
-Matched rerenders add ZERO training diversity. Reference56 remains default.
 Report: data/sim_data/diagnostics/native_short_multitarget_comparison_20260915_v1/report.json.

Second donor full-budget review completed:4 actual visual accepts,3 automatic
ineligible exclusions. Total native visual-pilot52 pending global caps/releases,
separate from legacy450 draft. Receipt:
data/sim_data/dataset_reviews/native_multitarget_20260915_v1/seed19_assistant_visual.json.

Broader native campaign (explicit short-profile trials, original full greenhouse):
data/sim_data/collection_batches/native_diverse_multitarget_20260915_v2.
Nine available qualified TRAIN donors; new curved/relocated, rigid-leaf variants,
up to6 real mounted-head views/target. First new plant:16 captured,15 eligible,
7 automatic-clear;182.32s collector. These are not individually reviewed yet.
Read the highest-numbered progress_*.json; only result.json/campaign_result.json
proves completion. Sensor annotations remain native optical-Z and visible IDs.

Two failures are preserved, neither counted:
-v1 controller tried updating the create-only JSON progress path; v2 writes
 append-only numbered progress snapshots and replays the verified unused asset.
-v2 seed19 preparation detected code changes during plan freezing and refused
 to launch. Keep code frozen during campaign; retry later with fresh bindings.
 No force-bypassing of code/source/memory checks; no other applications closed.

## Tested context metrics and exact native sidecar codec

morphology_context.py now measures curve/radius, parent segment and leaf
attachment/centroid/orientation/covariance in a source-parent canonical frame.
Tests cover renaming/reordering, proper rigid transforms, uniform scale,
leaf-context changes, unordered bipartite matching and cross-batch neighbours.
A signed-zero fingerprint issue was caught and fixed. Rounded fingerprints are
only provenance aids; tolerance matching is authoritative. No training approval,
source-cap reset or new biological family is granted by the module.

784 TRAIN recipes were replayed against actual source mesh leaf summaries.
Normalized tolerances.025/.05/.10/.15 yield784/784/781/768 context candidates.
For multi-chain parents the audit uses the nearest nondegenerate chain, not an
invented unified trunk. These are sensitivity results, not a selected release
threshold. Native/image duplicates and donor-balanced admission remain needed.
Report: data/sim_data/diagnostics/replayed_context_inventory_20260915_v2/report.json.
The failed initial single-chain-assumption audit is retained separately.

native_lossless_codec.py provides an explicit create-only packer/reader using
Zstandard byte-plane storage of the COMPLETE original NPY file. No float math,
depth inference, quantization, dtype conversion, deletion or legacy-reader change.
Checks bound file/header/declared array size before allocation, prohibit pickle,
and validate compressed/original hashes, shape and dtype. Signed zeros and NaN
payload bits remain exact. Requires optional zstandard on the packing host.
Nine actual native arrays:49,822,848bytes ->7,829,008bytes, every original file and
array byte identical after decoding. RGB unchanged and raw sources preserved.
Report: data/sim_data/diagnostics/native_codec_qualification_20260915_v1/report.json.
This adapter is not yet a compact capture-writer/export migration.

Full pytest:1201 tests +128 subtests passed in64.70s, clear_regression_20260915_v44.log.
The v43 unittest invocation found only16 unittest-style cases; it was not the
full suite. No final20k release/ZIP or local model training exists.
