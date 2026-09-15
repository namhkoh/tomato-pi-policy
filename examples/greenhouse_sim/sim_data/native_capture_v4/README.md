# Opt-in faster reference verification and preparation

V4 reuses the frozen V3 bank/schedule and generation contracts. It caches only
path resolution and parsed metadata within ONE verification: all17,089 unique
bank files are still fully hashed initially and at completion. Original path
spellings and link topology are rechecked; file size/mtime never authorizes data.
This is a fail-closed snapshot check, not an atomic filesystem lock.

Measured read-only schedule verification:73.80s(V3) versus29.55s(V4), identical
312-reference bank output. Receipt:
`data/sim_data/diagnostics/reference_verification_v4_ab_20260916_v1.json`.
This is not a60% renderer/whole-collection speedup.

Prepare a NEW job with `python -m sim_data.native_capture_v4.prepare`:

```text
--schedule ABSOLUTE_SCHEDULE_JSON --schedule-sha256 ORIGINAL_SHA256
--job-id donor_001 --attempt-id donor_001_anchor_00
--seed NEW_UINT32 --max-targets 12 --output NEW_JOB_DIRECTORY
```

Set PYTHONPATH to `examples;examples/greenhouse_sim` on Windows. Use repository
CPU Python with USD installed, not a second concurrent Isaac renderer. The exact
anchor must already have completed its native sensor qualification. Generation
keeps original TRAIN ancestry; new seeds/names do not grant novelty or approvals.

The launcher must use V4 `verify_prepared(...)` as well as V4 CLI, pin
`implementation_bindings()`, recheck immediately before launch, and require
native exit0, complete result, no failure artifact and annotation replay.
V3 preparation receipts are deliberately rejected by the V4 handoff.
No monkeypatching, original-asset edits or current-collector-default changes.

Independent combined regression:297tests passed in52.99s. A real seed101 CPU
pilot prepared two changed target petioles and a12-view native plan:
`data/sim_data/diagnostics/reference_prepare_v4_pilot_20260916_v1`.
That pilot has NOT yet rendered; preparation does not count as training images.
Keep one Kit process,20GiB system commit headroom and60GiB disk launch reserves.

## Serial continuation campaign

`python -m sim_data.native_capture_v4.campaign` waits for the named previous
campaign and all three matched V2 controls. It then requires the completed,
hash-bound SAME-callback RAW/compact storage proof before using the opt-in
compact collector. It cannot launch a second Kit process. Use an explicit new
output directory; this bounded wave is create-only, not an in-place resume.

Required arguments: `--schedule`, `--schedule-sha256`, `--prior-campaign`,
`--after-controls`, `--storage-qualification`, `--output`, `--isaac-python`,
`--native-deps`. Defaults: four rounds, seed base4000000, at most12 targets/job.
Only previously natively qualified TRAIN anchors enter the wave. Native exit0,
complete result, no failure marker and fresh annotation audit are mandatory.
Three consecutive native failures halt continuation; failed evidence is retained.
Worker-cleanup uncertainty is fatal. No other application's processes are stopped.

All counts remain candidates pending global image/morphology deduplication,
original-target budgets and review. Completed matched control captures alone do
not certify meaningful geometry novelty or grant a new view budget.
The original renderer, sensing, pose, geometry and annotation rules remain intact.

2026-09-16 verification: full sim_data regression2081tests+181subtests passed
(436.84s; optional staged-readback test file excluded). Follow-up99tests passed
for all campaign guards/cleanup and staged-readback tests (2.56s).
The compact production worker remains unqualified until actual live proof passes.
