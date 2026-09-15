# Write-only pre-render observer / post-exit CPU processor v1

Implemented here: `shadow_observer_v1.py`, `shadow_observer_cpu_v1.py`, and two
synthetic test files. Frozen `visibility_shadow.py` and its tests are unchanged.
No native process, collector, renderer product, companion process or gate is
created by these modules. No native observation was produced by their tests.

## Maxwell / single-owner integration handoff

The one owner is `sim_data.native_generated_reference.worker_v1`, matching
`native_generated_reference/execution_v1.py:WORKER`. Its current insertion point
is the explicitly marked pre-warmup observer seam in `worker_v1.collect`, after
actual pose/calibration/geometry/framing checks and before any `step_payload` for
that mode. Also record the existing `held_pre_render` branch before its continue.
I did NOT edit the worker, owner, execution request, audit, inventory or frozen
CPU preparation/scene files. Their owner must wire and qualify these extensions.

The current bridge fixes two candidates: `original_control`, `generated_variant`.
Use those same IDs and order in the observer spec. The processor requires the
same mode count/order and target/component identity as the pinned bridge plan.
Original control is retained as `not_run_original_control`: the frozen geometry
predictor covers generated plants only. It must never receive generated geometry
for a control, nor silently claim the controls were assessed.

Public native-side API (stdlib only, including on import):

```python
from sim_data.native_capture_v5.shadow_observer_v1 import EventWriter, canonical
observer = EventWriter(new_journal_directory, canonical(bound_spec))
observer.bind_context(canonical(generated_context))  # after checked substitution
observer.record_candidate(canonical(pre_render_or_hold_or_error))
observer.seal()  # before publishing the native result; includes missing entries
```

All three public event methods return **None**, never a score/decision. Every
event is exclusively written, flushed/fsynced, then given a separately fsynced
SHA256 seal chained to the preceding seal, BEFORE the method returns. The owner
must call `record_candidate` before its first warmup payload, and bind that exact
seal plus `writer.request_index + 1` as the first actual render request index in
native completion evidence. This is source-bound software chronology, not an
independent OS/clock attestation.

Normal invalid inputs are sanitized to `observer_error`; forbidden payloads are
not written. Event I/O failures and missing calls retain per-candidate error rows.
No observer error may add a native skip/retry or change existing hard checks.
Initializer failure/persistent storage failure cannot magically publish evidence:
the owner must keep its native outcome and declare observer unavailable. It must
not advertise observer qualification when the manifest cannot be authenticated.

## Strict event schemas

Schema prefix: `greenhouse.shadow_observer.v1.`. Paths are absolute; hashes are
lowercase SHA256. Pins are exactly `{path, sha256}`. Flags are exactly
`{skip_adopted:false, ranking_adopted:false, training_approved:false}`.

`spec` fields: `schema`, `plan`, `bank`, `clear_plan`, `anchor_entry_id`,
`producer_module`, `producer_bindings`, `implementation_bindings`, `candidates`,
`flags`. Each candidate is exactly `{candidate_id, mode, target_id, component_id}`.
Use `implementation_bindings()` in CPU preflight to pin the static local Python
closure AND the two literal frozen predictor/test hashes. Producer bindings must
include the actual `native_generated_reference/worker_v1.py` and be covered by
the owned exit's source map. This is not an unbound environmental opt-in.

The owner should bind the canonical spec bytes in its versioned execution request
before launch. `journal/spec.json` is their identical canonical copy. Its expected
path and SHA256 can be named in the owned-launch event before the file exists.
The journal constructor can run after admission without importing any USD/Kit.

Generated context fields (exact whitelist):

```python
dict(kind='generated_context', context_id=..., variant_directory=...,
     geometry_bindings={absolute_path: sha256, ...},
     plant_to_world_usd_row_vectors=actual_replacement_matrix,
     scene_evidence_sha256=sha256_of_canonical_current_scene_evidence)
```

Only a checked generated substitution may publish that context. Geometry bindings
must include the generated manifest and all referenced meshes/layers, and be a
subset of the pinned plan's source map. The actual matrix must match the original
bound placement within the existing1e-9 tolerance. Use current clear scene evidence
from Maxwell's scene hook, not a fabricated historical `old_manifest`.

Pre-render fields: `kind='pre_render'`, `candidate_id`, `context_id`, `calibration`,
`robot_snapshot_sha256`, `writer_request_index_before`. Context ID is null for
original control, and names an earlier sealed generated context otherwise.
Calibration accepts only the actual1696x816 uncropped calibration fields. Hash
the canonical actual robot snapshot; do not send full sample metadata. Holds and
errors are exactly `{kind:'geometry_hold'|'observer_error', candidate_id, code}`.
The code is a bounded identifier; full existing native reasons stay in native
receipts. Context/event schemas reject extra RGB, depth, validity, masks, native
visibility, labels, decisions or other unrecognized fields, including nested
calibration fields. Hash/source pins are provenance, not predictor sensor inputs.

The1024-candidate and8MiB JSON limits are observer resource/schema bounds, not
new geometry gates or training caps. This first processor supports only the named
two-mode clear-authority bridge plan; no discovery, replacement or volume claim.

## Owned post-exit extension contract (owner must implement publication)

The processor intentionally does NOT mint exit receipts. Its externally pinned
`owned_exit` extension must be published by the actual native owner AFTER the
owned worker exits; no completion inference from `result.json` or PID existence.
Keep the owner's normal exit/audit proof as authority, and bind this extension in
that owner's provenance. These JSON chains authenticate producer declarations,
not historical OS execution. They do not rerun native annotation themselves.

Exact `owned_exit` keys:

`schema`, `state='owned_worker_exited'`, integer `exit_code`,
`worker={pid,command}`, `launch_event`, `manifest`, `native_request`,
`native_result`, `native_failure`, `native_audit`, `outcomes`, `source_bindings`,
`flags`. Pin values may be null only as allowed by exit disposition. Exit0 requires
result/audit/outcomes and no failure. Nonzero exit requires a failure pin; all
event candidates can still receive CPU diagnostics, but no native outcome join.

The pinned `owned_launch` extension is exactly `schema`,
`state='owned_worker_running'`, the same `worker`, `spec`, `native_request`,
`producer_bindings`, `flags`. Worker command starts with an absolute executable,
`-m`, and the exact `worker_v1` module. Launch/exit command and PID must agree.

Minimal additional links the single worker/auditor must publish:

| Existing owner document | Observer-bound fields |
|---|---|
| Native request | `plan_path`, `plan_sha256`, `shadow_observer_spec` pin |
| Native result | `request_sha256`, `shadow_observer_manifest` pin |
| Owned post-exit native audit | `result_sha256`, `shadow_observer_outcomes` pin |

These are integration requirements, not edits performed here. Current worker
does not yet emit all these fields. Do not rewrite an old saved request/result.
An opt-in versioned execution request/audit extension belongs to Maxwell's owner.

`native_outcomes` is a comparison sidecar produced from the independently replayed
native audit. Exact keys: `schema`, `manifest`, ordered `records`, `flags`.
Every record: `candidate_id`, `state` (`captured`, `geometry_hold`, `capture_error`),
`event_seal`, `first_render_request_index`, `sample` pin or null,
`calibration_sha256`, `robot_snapshot_sha256`, `old_decision`,
`native_foreground_components` (unique component IDs). Uncaptured rows cannot
carry a sample or label. Keep unmapped/error native evidence explicit upstream;
do not fabricate an occluder identity to fill this field. Full native provenance,
replay checks and scene validation remain the native audit's responsibility.

## CPU use: only after native-owned exit

```powershell
python -B -m sim_data.native_capture_v5.shadow_observer_cpu_v1 --exit-receipt ABS_OWNED_EXIT_EXTENSION --exit-sha256 SHA256 --output NEW_CPU_DIRECTORY
```

Python API: `process(exit_path, *, exit_sha256, output) -> report_file_pin`.
There is no native-side call to this function and no injectable predictor/audit
callback. After exit verification and full source hashes, it lazily imports the
unchanged predictor and portable USD in the standalone CPU process. No live
companion process, Kit dependency injection, native process guard exemption or
early USD import is introduced. One immutable geometry snapshot per context is
shared only inside that invocation, followed by full uncached end hashes.

Processing records all candidates, holds, original controls, unknowns and errors.
It seals `predictions.json` for the COMPLETE inventory before reading/comparing
native outcome content. Failed prediction cases are not retried or tuned.
The comparison then checks seal/first-payload/calibration/pose links, retains
native component sets and strict false-block comparisons, and rehashes all inputs.
Malformed integrity links produce no completed report; a started CPU invocation
preserves `failure.json`. Capture failures yield unavailable native comparisons.

## Tests / qualification boundary

Synthetic fixtures model the new observer/owner extension, not a real native
capture. Tests patch only PRIVATE NEW processor/I/O boundaries for controlled
fixtures; frozen predictor globals/source are not monkeypatched or edited. The
actual frozen predictor runs on immutable synthetic arrays in end-to-end tests.
Other tests cover sensor-field rejection, no early numpy/USD/native imports,
seal tampering/order, I/O gaps, missing/extra candidates, owned-exit gating,
native counter/pose mismatches, delayed outcome reads, and full final rehash.

No prospective native result exists yet. Native-hook/owner/audit integration and
a separately authorized bounded capture remain required. Even successful
prospective evidence will not itself grant skipping, ranking, visual cut approval,
source-cap reset, diversity credit or training release.
