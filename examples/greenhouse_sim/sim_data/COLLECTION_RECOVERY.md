# Interrupted native collection: recovery and durable restart

Current branch: `koh-dev/sim-data`. Current resume plan:
`data/sim_data/collection_campaigns/grounding_resume_20260909_v1/resume.json`.
This is static robot-head native RGB-D collection, not teleop or robot control.

Latest quality checkpoint: **4,134 candidates / 5,787 audited raw**, 41 unique
fresh v3 inspections (39 accepts / two source-enforced holds). The current
snapshot is `data/sim_data/status/v3_after_nominal_rgb_hold_20260909.json`;
the original recovered 4,135-row checkpoint remains preserved below.

## What failed and what changed

The old campaign supervisors disappeared while their child Kit renderers
continued. Several children later shut down without a final supervisor result.
Previously the supervisor only persisted the exit code AFTER a potentially long
independent audit, creating a second window in which proof could be lost.

`collection_run` now writes `worker_exit.json` immediately after observing the
child exit, before auditing. The final result binds that receipt by hash; the
training exporter verifies the receipt, original launch identity and outcome.
An audit exception cannot turn a nonzero/unknown worker exit into success.
Historical completed jobs retain their existing verified legacy ledgers.

`collection_process` can observe a still-live Windows worker using a retained
kernel process handle. It checks the launch PID, process creation time,
executable and complete argument list. It writes a binding before waiting,
then reads the actual exit code from that SAME handle. The original deadline
is preserved; timeout termination targets only that verified native worker,
never the user's GUI. Already-exited/reused PIDs are rejected. A shutdown log,
complete-looking image directory or capture manifest is not exit proof.

`collection_resume` creates an explicit new three-lane plan. It skips bound,
completed scale jobs; waits for a verified observer where one exists; otherwise
captures into NEW directories. No old capture, partial audit, result or review
decision is overwritten. Recovery auditing uses `audit_recovered` beside the
original capture. There are no automatic retries or implicit training approvals.

## Running plan (2026-09-09)

- Five previously completed scale jobs were skipped.
- One seed17 worker was observed successfully: real exit code 0, no timeout,
  at 16:12:47 KST. Independent recovery auditing and v3 label checks subsequently
  completed: 1,164 raw frames, 890 eligible candidates, 274 exclusions. All native
  depth byte hashes match capture; no cross-snapshot RGB/view duplicates were found.
- Seed103 exited before its observer could attach. That run, seed37 and seed29
  lack durable exit proof and are scheduled for new-directory recapture.
- Eighteen unproven or unstarted jobs are assigned across the three lanes.
  Seed37 (validation) and seed31 (test) are saving new native frames; lane 1
  advanced automatically from the recovered audit to a new seed103 capture.
  These unfinished new jobs are not counted as audited training candidates.
- The cap is three capture workers while the user's GUI remains open, with an
  8 GiB available-RAM start gate and the original conservative disk reserve.
  Unrelated capture processes cause a lane to stop rather than oversubscribe.
- Supervisors run as detached **hidden** processes, independently of the tool
  terminal. This is not a Windows service and does not survive a reboot.
  A failure/dead observer/resource gate stops the affected lane without retry.

## Inspect, do not duplicate, an active lane

From the repository root:

```powershell
Get-Content data\sim_data\collection_campaigns\grounding_resume_20260909_v1\logs\lane_02.stdout.log -Tail 10
Get-Content data\sim_data\collection_campaigns\grounding_resume_20260909_v1\logs\lane_02.stderr.log -Tail 10
```

Native worker progress is under that lane's
`capture_job_010/job_010/worker.log`. `launch.json` records the exact native
worker; `worker_exit.json` is the pre-audit receipt. `result.json` records the
independent audit outcome. `completed_job_010.json` is the resume-lane checkpoint.
Do not equate a sample directory appearing with a completed audited job.

For a later explicit restart, first inspect processes and receipts. Then use
`python.bat -B -m sim_data.collection_resume create --campaign SOURCE_CAMPAIGN
--observer VERIFIED_OBSERVER_DIRECTORY --output NEW_RESUME_DIRECTORY` from
`examples/greenhouse_sim`. Omit observers for workers that cannot be adopted.
The command only writes the frozen plan; it does not start captures. Run its
three named lanes with `collection_resume run --resume NEW/resume.json --lane
lane_01` (and 02/03), using hidden detached processes on Windows. Never re-run an
existing lane directory or launch extra workers outside this bounded plan.

The current restart consumes the original campaign's source-job mapping. A
future restart after resumed jobs finish must first reconcile their completion
ledgers; blindly creating another plan from the old campaign alone would not
discover completed jobs stored in this new resume directory.

## Dataset quality remains separate

Worker completion is not visual QA. Native RGB and Isaac Replicator camera-Z
are unchanged; no replacement depth, renderer shortcut, camera relocation or
anatomy modification was introduced. Frozen source-family splits, task v3 query
gates and complete-release volume/balance gates remain in force.

Fresh v3 inspection has 20 unique wave-2 decisions: 19 accepts and one hold.
The held seed61/SubStem_41 query is a dim triangular fragment merging into leaf
silhouettes. Native identity is correct, but RGB readability is not convincing.
An append-only source hold is enforced by `training_export.gather`.

The hold-adjusted baseline is 3,245 candidates / 4,623 audited raw frames;
`data/sim_data/status/v3_after_fresh_hold_20260909.json` binds the earlier full
recount, hold record and revised coverage. The newly prepared all-family set
has 108 cards in `dataset_reviews/grounding_all_sources_20260909_v3`. Fourteen
already-inspected v3 cards matched both complete task identity and saved-card
hash exactly, allowing explicit evidence reuse; they are not 14 additional
independent inspections. The other 94 cards remain pending. No v2 review was
reused and no complete release or VLM fine-tuning is claimed.

The completed recovered audit adds 890 candidates to that hold-adjusted
baseline: **4,135 candidates / 5,787 audited raw frames**, across 30 audits.
Splits are 2,992 train / 368 validation / 775 test. The source-bound receipt and
row index are `data/sim_data/status/recovered_seed17_v3_20260909.json` and its
`.rows.json` companion. The six recovered seed17 review cards were each actually
inspected: four localized cut points on their native target petioles, two
leaf-occluded cuts with abstain/null/change-viewpoint answers. All six bindings
verify under `dataset_reviews/grounding_recovered_seed17_20260909_v3`.
That recovered review brought fresh v3 inspection to 26 unique cards. Subsequent
one-by-one review of 15 broader cards added 14 accepts and one hold: 41 unique
inspections in total, 39 accepts / two enforced holds. The broader all-family
bundle has 28 accepted records (14 exact-evidence reuses), one hold and 79
pending cards. The held selection must be replaced in a new QA bundle before
release; passing checks cannot be manufactured by deleting its decision.

The new hold is `seed61_full_cd2eabc75725cdcefcdc` / scale seed61 sample_0469:
native petiole identity is correct, but the nominal shaft blends into a tomato
in untouched RGB. Lossless crop and original unmarked RGB were both inspected.
An append-only source hold is verified exporter-enforced, subtracting one
test/easy candidate: **4,134 candidates / 5,787 raw** (2,992/368/774 by split).
This is nominal-region readability uncertainty, not a reclassification as fruit
anatomy or depth occlusion. The query screen does not guarantee nominal RGB
readability. Neither native identity nor assistant QA grants biological accuracy
or physical cut approval. Bindings and pending counts are recorded in
`data/sim_data/status/visual_qa_followup_20260909_v3.json`.

The first resumed seed37 frame also passed a source-file and native depth byte
integrity check. Its original RGB was inspected; the capture is from the mounted
`/World/RBY1/link_head_2/attachments/HeadCamera/D405/DepthCamera`, not an extra
cinematic camera. Receipt: `data/sim_data/status/resume_first_native_frame_20260909.json`.
This single-frame check is not a completed-job or full-release claim.

Validation: 495 tests plus 47 subtests passed (latest rerun 47.10 s), including a real Windows child
process test observing exit code 7 through a retained handle and rejecting
wrong creation time/command identity, plus audit interruption and source-hold
regressions. Progress reporting also recognizes the explicit recovered audit
path. Live seed17 exit observation returned actual code 0 and the queue advanced.
