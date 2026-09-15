# Bounded native generated-plant inspection campaigns

`native_generated_campaign.py` generalizes the previously successful one-off
original/generated capture runners. It is an inspection collector, not a20k
training exporter, a dynamics experiment or an automatic image reviewer.

## What one case does

1. Load an explicitly frozen, TRAIN-only generated comparison plan.
2. Qualify the recorded mounted RB-Y1 head camera: native known-surface
   optical-Z/identity smoke test and full-greenhouse848/1696 comparison.
3. Capture native1696 original/generated frames with unchanged robot pose,
   camera mount, optics, lighting and scene population.
4. Recompute cut/query labels from each actual anatomy and saved native
   optical-Z/component buffers. Save clean RGB, native768 query crop and
   review-only overlay if the strict clarity/identity/visibility checks pass.
5. Leave every candidate pending individual visual review.

This produces at most one generated frame and three original/control frames
per case. Repeated controls, crops, alternative prompts and depth visualizations
are not new target diversity or extra accepted training images.

## Safety and reproducibility boundaries

- One serial renderer; the Kit bootstrap controller itself has no SimulationApp.
- Explicit1-24 cases, each with a new local camera/generated/annotation output.
- Frozen code and source-plan hashes; repeated view/generated-target pairs
  rejected. A changed hash needs a new explicit qualification, not an override.
- At most12 cases per original donor target within a campaign. This local
  inspection limit is not the final cross-campaign release cap: the original
  source-target12-view grouping remains authoritative for any future release.
-18GiB queue commit reserve plus the native16GiB/4GiB physical-memory guard;
 40GiB free-disk reserve. Exact pre-launch memory/disk snapshots are recorded,
  including failed admission. No user processes are closed.
-60-1800s per worker; only its exact owned child may be killed on timeout.
  Actual process exit, result state, failure-file absence and log hash are
  checked. A nonzero exit cannot be converted to successful capture.
- Stop on any worker failure; no automatic retry or implicit resume. Completed
  and failed cases remain on disk. Annotation numeric rejection is recorded,
  not converted into an accepted cut point.
- Original assets, splits, prior reviews and captures are unchanged.
  Physics/reachability, botanical correctness and hardware sensing remain
  outside this static inspection qualification.

## Local command

Run from the repository root with a fully prepared NEW campaign directory.
Its `plan.json` uses schema
`greenhouse.native_generated_inspection_campaign.v1`, containing ordered
`case_001`... records and each immutable generated-case plan SHA256.

```powershell
$env:PYTHONPATH='examples;examples/greenhouse_sim'
$env:OPENBLAS_NUM_THREADS='1'
python -m sim_data.native_generated_campaign --plan data/sim_data/diagnostics/NEW_CAMPAIGN/plan.json --validate-only
D:/isaac-sim-6.0.1/python.bat -m sim_data.native_generated_campaign --plan data/sim_data/diagnostics/NEW_CAMPAIGN/plan.json
```

Do not rerun an already attempted campaign or invoke this while a different
Isaac renderer is active. Validation does not launch Isaac or approve data.

## September15 execution

Read-only inventory:
`data/sim_data/diagnostics/native_pilot_inventory_20260915_v1.json`.
It matched62 reviewed TRAIN reference views/23 targets/11 original donor
families with eligible targets in the existing20-layout generated library.
Original sample/RGB hashes and source plans were checked. This is a selected
inventory, not62 fresh high-resolution images or an unbiased yield estimate.

Next batch picks one large, clear reference per previously untested donor:
seed19/41/43/67/71/83/89, seven generated layouts/targets, at most14 new annotation
candidates. Geometry and native checks may reject some. No acceptance is promised.

The v1 campaign stopped **before its first renderer** on the18GiB queue reserve.
No images were created by that attempt. Failure preserved at:
`data/sim_data/diagnostics/native_generated_seven_family_20260915_v1`.

Fresh v2 preserves all seven source views/assets and changes only the local
qualification destination plus the explicitly bound admission-logging fix:
`data/sim_data/diagnostics/native_generated_seven_family_20260915_v2`.
Its bounded readiness queue waits at most2h for20GiB commit headroom and zero
other Kit processes, then launches the campaign once. It checks every30s,
does not retry failures, and does not alter the18/16GiB worker guards.
Queue log:
`data/sim_data/native_generated_seven_family_memory_queue_20260915_v2.log`.
Native log (only after launch):
`data/sim_data/native_generated_seven_family_20260915_v2.log`.
Inspect queue/result/exit receipts for the actual state; queued is not captured.

Regression:20 focused campaign tests; full suite **988 tests +77 subtests**
passed in63.28s (`data/sim_data/clear_regression_20260915_v27.log`).
Earlier three-family native requalification and its six individual visual
reviews are documented in [NATIVE_CLEAR_ANNOTATION.md](NATIVE_CLEAR_ANNOTATION.md).
That native success should not be mislabeled as completion of this new batch.

Current reviewed legacy draft remains450 images (312 train/48 validation/90
test). There are separately8 reviewed native annotation-pilot frames across4
donors/4 generated layouts. No20k training release, final ZIP, H200 processor
qualification or model training has been produced.
