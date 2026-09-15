# Native high-resolution dataset support

This package is separate from the legacy 848x408 release/export path. It does
not change active capture defaults, frozen splits, existing reviews or model
training. Native captures are 1696x816 mounted-head RGB with native Isaac
optical-Z and exact rendered identity buffers.

## Implemented and tested

- `audit.py`: replay completed static annotations from raw or compact storage.
  Authenticate the recorded request/plan, source/prerequisite/code bindings,
  planned candidate membership/uniqueness, sample identity, target ancestry,
  robot/screen metadata, original callback hashes, labels and query traces.
  Receipts bind the loaded audit code. Failed/incomplete captures and duplicate
  result rows cannot inflate counts. Successful replay is NOT release approval.
- `bundle.py`: create-only compact copies and a logical reader. Original RGB
  PNG bytes and whole native NPY files, including NaN payloads/signed zeros,
  remain exact. Review-only PNGs are explicitly omitted from copies. Sources
  remain unchanged. Required observations, external sample/label/trace bindings,
  bounded reads and integrity checks are mandatory.
- `capture_storage.py`: write the same compact format directly from validated
  native callback arrays, without staging raw NPYs or drawing review overlays.
  It validates shapes/dtypes and callback fingerprints, but does not replace
  the collector's native camera/scene/visibility checks. Native integration is
  an explicit diagnostic trial, not yet a production-default change.
- `admission.py`: pure candidate accounting. Keep original donor ancestry and
  frozen splits; finalize transitive global context duplicate groups; reject
  exact decoded-RGB, repeated scene/camera, near-image and cross-split duplicates;
  apply a shared per-context view cap and deterministic donor/target balancing.
  Require 20,000 TRAIN candidates plus separately specified validation/test.
  Geometry/image calibration and inventory evidence are externally supplied,
  not verified or invented by this accounting module. All results remain
  `training_approved=False`; the actual admission adapter/calibration and final
  release validator are still required.

## Evidence, 2026-09-15 checkpoint

Full sim-data suite: **1346 tests +181 subtests**,99.15s:
`data/sim_data/clear_regression_20260915_v46.log`.

Direct compact-writer CPU qualification on seven ACTUAL prior native captures:
21 NPY and28 PNG files byte-exact,126 original source files unchanged.
162,793,157 ->38,901,025 bytes. These are storage replays, not seven new images.
`data/sim_data/diagnostics/native_direct_writer_qualification_20260915_v1/qualification.json`.

Raw-to-compact copy plus final request-bound anatomical replay:
`data/sim_data/diagnostics/native_compact_sample_qualification_20260915_v2/final_bound_audit.json`.
All seven labels replay;2 strict candidates,2 clarity holds,3 exclusions.

Native diagnostic worker:
`data/sim_data/diagnostics/native_compact_wider_worker_20260915_v1.py`.
Only storage changes from the guarded wider-camera worker; full greenhouse,
materials, actual mounted head camera, native optical-Z/IDs, memory reserve,
static synchronization and reference56/warm56_then8 profiles remain intact.
Request/result bind the worker and storage implementations. Check
`diagnostics/native_compact_wider_seed19_20260915_v1/capture/result.json`
and absence of `failure.json` before claiming native qualification.
Repeated matched views contribute ZERO additional training diversity.

Isaac Python3.12 lacked Zstandard. The tested0.23.0 cp312 wheel is isolated in
`data/sim_data/native_capture_deps_py312_v1`; only the diagnostic native worker
adds it to PYTHONPATH. Isaac's installation and system packages are unchanged.
The CPU host's Python3.13 environment must NOT import the cp312 wheel.

## Remaining release gates

### Same-callback storage qualification queued, not yet passed

`dual_storage.py` independently writes RAW and compact observations from one
caller snapshot, then checks full native NPY bytes, RGB PNG bytes, all prim
identities, masks, labels, traces and metadata. It rejects a raw-only writer
masquerading as compact storage and rereads externally pinned metadata after
comparison. Eleven synthetic-buffer tests pass; these are not live sensor proof.

The separate ignored worker
`diagnostics/native_same_callback_worker_20260915_v1.py` applies this comparison
to one live validated Isaac callback, then replays RAW and compact annotations.
Queue: `diagnostics/queue_same_callback_native_20260916_v1.py`.
It waits for the current reference-bank campaign, then requires20GiB Windows
commit headroom, existing physical-memory preflight and60GiB disk reserve.
It uses the existing seed19 wider plan:36 proposed frames across4 targets;
seven was the previous observed capture count, NOT the prospective bound.
Exit0, no failure artifact, bound evidence and post-exit audits are mandatory.
No production-default change or source removal is authorized by this diagnostic.

Three original/V2 matched plant pairs are queued AFTER storage qualification:
`diagnostics/queue_v2_visual_controls_20260916_v1.py`. This closes the missing
joint image/morphology positive-control domain; it does not assert positivity
until the actual observations are reviewed. All controls add ZERO training
diversity. Read their queue/results to distinguish scheduled from completed work.

### Actual saved-geometry adapter

`morphology.py` adds `extract_output(OutputPin(...))` for hash-pinned rigid-leaf
V2 variants. It replays the existing catalogue, reads authored meshes and saved
chains, includes original counterparts and nearest valid parent chains, and
retains original family/target/split ancestry. Unsupported domains are held;
tampered inputs fail. It does not derive novelty from recipe predictions or
grant new view budgets. Twenty-eight new focused tests pass.

Actual evidence: `diagnostics/morphology_actual_output_20260915_v1/report.json`.
Four completed native plans yield28 descriptor rows for11 original targets and
3 donors; repeated originals and same-output replay are exact. These are NOT
28 independently qualified contexts.84 numerical nuisance comparisons have
maximum residual1.0005e-8;25 exceed the previous1e-9 test tolerance. That mismatch
is preserved, not silently relaxed. Generated/original distances0.2721-1.1443
are measurements, not adjudicated novelty positives. The existing seed19 V1
visual positive is outside this V2 adapter; joint calibration remains pending.

### 2026-09-15 native similarity measurement checkpoint

`near_image.py` compares unchanged native RGB using full-frame MAE and a complete
129x129 patch centred on each image's nominal cut location. Patches use integer
indexing only; no resizing, interpolation, padding or brightness correction.
Hidden coordinates are offline deduplication supervision, never model inputs.
The combined score is the maximum of full-frame and local MAE, normalized to
0..1. Both measurements and percentiles remain available. This metric is not
an automatic novelty gate. Its 42 focused tests pass.

Read-only actual-image diagnostic:
`diagnostics/native_rgb_controls_20260915_v1.json`.
21 matched-render duplicate pairs have maximum score0.039133; one visually
adjudicated curved-geometry pair scores0.347364. All22 pairs were measured at
1696x816 with bound source evidence. Only two donors are represented, and only
one positive control exists: no threshold is selected or globally calibrated.
No source image, depth, review or split was changed; controls add zero diversity.

The 18-plant native campaign completed212 frames/65 strict automatic candidates
(15 native jobs,3 invalid geometry attempts). Four further actual full-view and
unscaled crop inspections were recorded, bringing the overlapping visual pilot
to69 pending global admission. Receipt:
`dataset_reviews/native_diverse_generated_visual_20260915_v2/assistant_visual.json`.

The reviewed-reference continuation is now natively exercised, not merely queued:
first fresh seed101 sensor pair passed; first two generated jobs captured19
frames/8 strict candidates with independent annotation replay. First job timing:
29.66s scene setup,132.44s total,40.66s first render,3.92-4.27s later render calls.
These are collection measurements, not simulator interaction or training speed.
Existing full environment, native sensing and quality gates are unchanged.

1. Calibrated global morphology/image novelty admission; provenance-backed
   view inventories and donor-balanced sampling. No per-name source-cap reset.
2. Better reference-bank/pose selection and bounded generation qualification;
   measure useful reviewed images/hour, not raw renders or diagnostic repeats.
3. Native compact capture qualification and portable export/loader integration.
4. Held-out donor capture after TRAIN-only calibration is frozen.
5. Native-resolution Qwen3-VL-8B export, H200 processor validation, final counts,
   leakage/duplicate checks, dataset card and transfer ZIP.

There is **no final 20k TRAIN release or ZIP** at this checkpoint. Static
cut-point localization examples are not action demonstrations, dynamic episodes,
physical cut approvals or evidence of a trained policy.

## 2026-09-16 opt-in scale components

- `compact_qualification.py` checks the pinned same-callback proof, all original
  payloads, complete prim tables, label/trace equality, native exit and both
  post-exit audits. `compact_views.py` changes serialization only; AST parity
  tests compare scene/pose/render/annotation logic with the frozen RAW collector.
  Actual qualification is still queued, not yet passed. No default was changed.
- `compact_image_index.py` authenticates compact identity-encoded PNG payloads
  and delegates to the unchanged native RGB duplicate index. No extraction,
  re-encoding or depth calculation. Its explicit v2 graph retains compact
  source provenance; downstream v1-only selection must not silently relabel it.
- `staged_readback.py` is an isolated experimental factory. CPU remains default.
  Its comparison mode checks same-callback CPU/GPU RGB and optical-Z bits; the
  staged timing mode requires at least three distinct-camera comparison captures.
  Installed Isaac6.0.1 APIs are pinned. Native comparison/timing has NOT run;
  speedup and qualification are unknown. Do not use this for production capture.

The live reference wave reached18 completed generated jobs; later jobs continue.
Separately, morphology_observed_contexts_20260916_v1/report.json now contains222
available descriptors (111original/111generated) from30 actual generated assets.
These are measured geometry, not222 approved independent contexts. Meaningful
joint image/geometry qualification, held-outs and final20k release remain open.

`provisional_compact.py` now accepts the compact index's v2 graph without losing
its source/label/trace bindings or pretending it is a v1 receipt. It reconstructs
and validates the internal base graph before delegating the unchanged12-view
original-target policy. Raw-only v1 output is unchanged; compact inventories
cannot use the raw-only shortcut.179 focused tests passed in65.69s. No actual
compact production wave or final dataset is approved by these software tests.

Expanded observed checkpoint (through reference job020):784frames/51receipts,
206strict/213held/365excluded across14TRAIN donor families/51original targets.
All306,936 image pairs resolved in43.15s with zero edges at the unchanged
PROVISIONAL empirical cutoff.197provisional candidates remain after9source-cap
exclusions;578nonstrict rows and the existing explicit visual hold stay excluded.
Artifacts: `diagnostics/native_completed_campaign_inventory_20260916_v3.json`,
`native_near_image_graph_20260916_v2/`, `native_provisional_selection_20260916_v2/`.
Later active jobs are not included in that fixed snapshot. Three more direct
visual reviews are in `dataset_reviews/native_reference_visual_20260916_v3/`.

Measured first20reference jobs:264proposals,188captures,55strict candidates in
122.62min from first preparation to twentieth audited completion:26.9strict/h,
NOT final-approved images/h. Mean job window309.2s:preparation123.7,scene32.5,
render/callback74.3,geometry22.5,other collector14.0,native boundaries/audit42.1.
V4's measured verifier saving alone cannot make20k practical. The current51
scheduled targets have at most612 images under the unchanged12/source cap.
Broader original targets, substantive geometry qualification and bounded extra
view yield tests are required; no cap relaxation or visual-fidelity reduction.

## Conservative observed geometry budgets

`geometry_admission_v2.py` joins pinned observed inventory, actual extracted
geometry, native image graph and reviews. It supplies budget inputs, not final
admission or proof of native execution. Without external qualified controls,
all variants retain their original12-view pool. Uncertain comparisons merge;
an explicit novelty-withheld review does not invalidate an otherwise clear label.

`morphology_set_v1.py` is a separately versioned pseudometric: fixed-feature
L-infinity plus symmetric Hausdorff distance on leaf feature sets. It supports
unequal nonempty cardinalities and ignores duplicate/coincident multiplicity,
so distinct arrangements can merge conservatively. No prior cutoff transfers.
Actual nuisance replay756/756 passes, max5.1831e-13; this is numerical stability,
not a meaningful shape novelty threshold. Joint native controls/review remain
required before any extra geometry budget. Main independent focused regression:
190tests+45subtests passed2.62s (`geometry_context_regression_20260916_v2.log`).

## Native-resolution export and server preprocessing

`native_clear_export.py` builds a create-only portable directory from external
final admission, not from raw counts or a provisional selection. It requires
at least20kTRAIN, explicit positive heldout counts, the exact original donor
reservations, native compact sources and bound review/calibration evidence.
It copies original RGB/native optical-Z/IDs/calibration bytes and adds an exact
unresized768x768query crop. Answers remain normalized1000 full-frame coordinates.
No public small-release switch exists. This is not yet an archive or code kit.

`native_clear_processor_preflight.py` validates a completed package and checks
explicit TRAIN rows with an existing server-local Qwen3-VL-8B AutoProcessor.
It forbids network/weights, requires an explicit total token budget, checks
expanded image tokens/grids and exact assistant-only supervision, and never
truncates an over-budget example. RGB/query are the model inputs; native depth
and geometry remain sidecars. Run its documented CLI only after an admitted
release and server processor snapshot exist. No model was downloaded locally.

Main independent tests:143passed,1optionalactual-sample skipped,81.57s. The
legacy H200 scripts still require a separately integrated native-profile route;
these tests are not evidence of actual server preprocessing or training.
