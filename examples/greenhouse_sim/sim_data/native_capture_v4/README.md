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
