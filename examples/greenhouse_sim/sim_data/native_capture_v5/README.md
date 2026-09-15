# Bounded twelve-view throughput diagnostic

This is an opt-in experiment, not a change to the production view policy.

- Input: a hash-pinned completed V4 preparation with exactly two target petioles.
- Preserve the original six proposals exactly. Add six distinct interior base
  positions within the same 0–4cm outward / ±4cm lateral bounds.
- Keep body heading, mounted head camera,1696x816 native RGB/Z/IDs, full greenhouse,
  geometry screening, annotation and trace checks. Call the frozen collector.
- One stage/product,24 maximum proposals; RAW diagnostic storage. Render budget
  must explicitly select the existing reference56 or warm56_then8_trial profile.
- More proposals do not grant new geometry contexts, source budgets or approvals.
  Compare strict candidates retained after cumulative duplicate/cap checks, not
  raw count. Block timings include their actual cold-start costs.

Commands (repository CPU Python for plan/verify, Isaac bootstrap for capture):

```text
python -m sim_data.native_capture_v5.bounded_views plan --prepared ABS_PREPARED_JSON --prepared-sha256 SHA --schedule ABS_SCHEDULE --schedule-sha256 SHA --render-budget warm56_then8_trial --output NEW_PLAN
python -m sim_data.native_capture_v5.bounded_views verify --plan ABS_PLAN --plan-sha256 SHA
D:\isaac-sim-6.0.1\python.bat -m sim_data.native_capture_v5.bounded_views capture --plan ABS_PLAN --plan-sha256 SHA --output NEW_DIAGNOSTIC_CAPTURE
```

Real CPU-prepared probe:
`data/sim_data/diagnostics/reference_bounded12_v5_probe_20260916_v1/plan.json`
SHA256:2a0164adae0f488243003430e4ab6a40225672d12fb29320aeaf266163ef4139.
This is prepared, not natively exercised yet.

## Serial queue

`python -m sim_data.native_capture_v5.campaign_queue --request ABS_REQUEST --request-sha256 SHA`
accepts an explicit versioned request (see SCHEMA/validate_request). It waits for
the named previous collection, all three matched controls and full same-callback
storage proof. Then it runs the bounded RAW probe, independently audits it,
optionally runs exactly one fresh original-target pilot and post-exit audit,
then starts the unchanged six-view V4 scale coordinator with a NEW output.

The launch environment must provide absolute repository PYTHONPATH entries:
`examples;examples/greenhouse_sim` (resolved against this repository).
One Kit only,20GiB commit headroom and60GiB disk reserve. No unrelated process
is stopped; uncertain owned-child cleanup is fatal. A completed worker is not
training admission. The original and probe capture receipts remain diagnostic.

Independent combined validation:109 tests passed in14.60s for V5/queue and
native_original_capture. No native speedup or new source qualification claimed.

Measured baseline receipt:
`data/sim_data/diagnostics/first20_throughput_receipt_20260916_v1.json`.
55strict candidates in122.619min (26.913/h), before dedup/caps;212 source bindings.
This poor throughput motivates the experiment, not a promised completion ETA.

Final queue-review follow-up:115 combined tests passed in13.96s. CPU and native
Python paths are constructed separately. The original pilot receipt binds the
owned PID/command/launch event, submitted plan, actual capture request/result,
and independent post-exit audit. It still grants no training approval.
