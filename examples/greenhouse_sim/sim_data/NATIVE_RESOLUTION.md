# Native higher-resolution qualification

The latest user decision permits higher-resolution captures. This is a separate
native **1696x816** simulator mode, not an enlargement of existing848x408 RGB.
The mounted RB-Y1 A v1.2 head-camera transform, focal length, apertures, aspect
ratio and greenhouse geometry must remain unchanged. This does **not** establish
that the physical D405 supports this resolution or has matching noise/optics.

## Current boundary

- Implemented and unit-tested: explicit resolution/intrinsics contract; native
  RGB, optical-Z and instance-ID checks; known-surface sensor diagnostic; paired
  full-greenhouse snapshot diagnostic.
- Native high-resolution execution and visual qualification: **pending**.
- The production848x408 collector, existing releases, frozen family splits,
  reviews and Qwen training/export contracts are unchanged.
- New pilot samples have their own schema and are **not training-approved**.
  Passing projection checks alone does not establish a clear anatomical junction.
- No model download, training, robot command or physics action is part of this pilot.

## Diagnostics

Only run after other native Kit workers have exited. Keep the existing
16GiB commit/4GiB available-RAM launch reserve; it is not a peak-memory guarantee.
Both commands require a fresh output directory. From the repository root:

```powershell
$env:PYTHONPATH='examples;examples/greenhouse_sim'
$env:OPENBLAS_NUM_THREADS='1'
& D:/isaac-sim-6.0.1/python.bat -u -m sim_data.native_resolution_smoke --output data/sim_data/diagnostics/native_resolution_manual_01
```

The standalone smoke renders a diagnostic cube/occluder, **not training data**.
It checks1696x816 native buffers, camera projection, exact native instance identity,
known2m/2.4m and1m optical-Z surfaces, off-axis depth, and fresh static observations
after camera/occluder changes. It preserves raw invalid depth plus a validity mask.
It does not claim moving-scene synchronization.

The paired greenhouse diagnostic includes that sensor smoke, then reconstructs
one completed training-family snapshot and renders it at BOTH resolutions:

```powershell
& D:/isaac-sim-6.0.1/python.bat -u -m sim_data.native_greenhouse_pair --source-capture data/sim_data/collection_campaigns/clear_capture_20260915_orbit_v2/job_021/job_021/capture --sample sample_0001 --output data/sim_data/diagnostics/native_greenhouse_pair_manual_01
```

This reference is seed7/SubStem_42, not a validation/test family. Its source plan,
scene files, sample and recorded inputs are hash-verified. The same source
geometry, full greenhouse population, diffuse lighting, renderer, robot joints
and camera pose are reconstructed and checked. A fresh geometry screen is
required. This is a static snapshot, not a validated navigation/manipulation path.

Both products use the reference56-subframe budget and native legacy instance
annotator. This is intentionally not a new speed claim. Native high-resolution
timing, peak memory, instance-backend equivalence and visual improvement still
need measurement.

Outputs under each `native_<width>x<height>` directory:

- `inputs/rgb.png`: clean, uncropped native RGB.
- `inputs/depth_m.npy`: unchanged Isaac `distance_to_image_plane`, optical-Z metres.
- `inputs/depth_valid.png`: validity only; no inferred surface.
- `supervision/`: native renderer IDs, component/organ identity and visible target mask.
- `sample.json`: explicit calibration, source target, reprojected physical10?20mm
  interval, robot pose, native sensor and static freshness evidence.
- `review/`: diagnostic overlays, never model inputs.

A final `result.json` means captures and geometric pairing completed, **not**
visual approval, export qualification, additional independent targets or training
readiness. A failure/nonzero exit must stop the pilot; no automatic retry.

## Before a higher-resolution training release

1. Inspect both native full images and the target attachment; verify the observed
   petiole/masks/depth and physical nominal10mm point, not just a doubled pixel label.
2. Qualify full-greenhouse sampling at the new resolution, including thin targets
   rejected at848x408, with unchanged collision/provenance/split requirements.
3. Update and test review, label validation, exporter, query-crop construction,
   Qwen prompts/coordinate conversion and packaging as one explicit new profile.
   Do not feed1696x816 images into hard-coded848x408 contracts.
4. Choose crop size and evaluation tolerances on a declared pixel/physical basis.
   A native768x768 crop can preserve the old384x384 crop's field of view without
   enlargement; this is a proposal, not the currently implemented export.
5. Measure Qwen preprocessing/token limits, full-frame-plus-crop behavior and
   throughput on the H200s. More source pixels do not automatically mean the
   model preserves every detail.
6. Reapply coverage, individual visual review and release validation. Do not
   count paired renders of the same target as new target/family diversity.

## Queued September15 pilot

Operational launcher:
`data/sim_data/diagnostics/native_hires_pair_queue_20260915_v1.ps1`,
with its matchingPython controller and a pinned implementation-file hash manifest.

It waits at most6h for the already queued original-side supplementary campaign
to complete successfully and for all Kit processes to exit. Existing collection
is not stopped or modified. The controller then permits only its own Kit process,
checks memory again, and starts one owned diagnostic child with a1800s timeout.
There are no automatic retries, training jobs or data approvals.

Queue log: `data/sim_data/native_hires_pair_queue_20260915_v1.log`.
Planned result directory:
`data/sim_data/diagnostics/native_hires_greenhouse_pair_20260915_v1`.
If pinned implementation code changes before execution, the controller refuses
to launch; inspect/requalify the new code rather than bypassing this check.
