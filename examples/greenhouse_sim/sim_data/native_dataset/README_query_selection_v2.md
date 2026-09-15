# Query selection v2: standalone, not integrated

Public API in `query_selection_v2.py`:

```python
from sim_data.native_dataset.query_selection_v2 import annotate_v2

label, trace, evidence = annotate_v2(
    metadata, report, rgb, depth, valid, components, catalogue,
    target_mask=target_visible_uint8,
    expected_label=old_label,  # optional: exact fresh legacy derivation required
)
```

Epoch: `greenhouse.native_query_selection.v2`.
Composite trace schema: `greenhouse.native_query_trace.v2`.
No separate derive/trace wrappers: the combined API enumerates once.
`select_query(...)` exposes the full selection evidence without returning a label.
`fixed_photometric_arcs(cumulative, end)` exposes the deterministic grid.

The caller supplies a report from the existing validated catalogue/manifest API,
the actual saved native target mask (uint8 0/255), and exact native arrays.
Source/plan/worker/owned-exit bindings remain the caller's responsibility.
This module performs no file writes, native launch, inventory change or adoption.

## Non-negotiable checks

- Recompute the frozen legacy label and native RGB/depth/camera fingerprints and
  exact target-mask identity. A supplied old label must match the derivation.
- Preserve upstream parent, 10–20mm interval, cut-answer, clarity and validity gates.
- Enumerate the exact original 41 linspace candidates, including recorded skips.
  Queries below45mm are rejected. Keep native visibility, same connected target
  island, at least18px query-to-cut separation, rounding and all local checks.
- For EVERY locally usable candidate run the unchanged legacy full trace on a
  fresh private label copy, AND run anchored photometry at8,13,18,...mm plus
  all in-range anatomical knots and the exact query endpoint.
- Both must pass. New fixed probes never replace the old trace. The grid remains
  sampled evidence, not continuous-visibility or visual readability proof.
- Select deterministically in original candidate order using the documented
  version/target/sample SHA256 rule. No random global state.

Candidate and fixed-grid photometry share one private invocation-local
NativeQueryVisibility over bytes-backed immutable copies of the actual arrays.
Geometry/fixed-probe reuse is confined to that context; returned records are
copies. The frozen derive/trace functions still build their own checkers.
No caller-injected context, long-lived cache, global monkeypatch or worker edit.

## Returned disposition mapping

| Outcome | Returned label | Returned trace | Future worker mapping |
| --- | --- | --- | --- |
| At least one joint pass | Fresh copy with chosen query fields and new epoch; cut answer unchanged | Both checks pass | Strict automatic candidate, not release |
| No joint pass, legacy label locally eligible | Fresh copy retaining original query and cut answer; new epoch | Failed composite trace; fallback flag | hold_visual_clarity |
| Legacy upstream/local exclusion | Preserve rejection and eligible=False; add new epoch | None | Existing exclusion |

A failed fallback is NOT a successful selection. The evidence's selected_query is
None in that case. The label and trace are returned in memory; the future worker
owns explicit new-epoch adoption. Old labels, holds and receipts stay immutable.
Do not run v2 labels through frozen legacy equality, relabel old proof pins, or
claim an old audit executed this policy. A new audit must call this same combined
API on the same source-bound arrays and compare label/trace exactly.

## Verified CPU evidence

Final selector plus frozen label/trace regressions:98 passed in41.89s.
The test command and source pins are in the ignored diagnostic test receipt.

Actual three-control diagnostic:
`data/sim_data/diagnostics/query_selection_v2_20260916_v1/result.json`,
SHA256 `83d0caf8284e84644f09a2f6e4d4e4364a638dd38ce95d926bfbfe5cbc964e28`.

| Held frame | Local candidates | Legacy passes | Fixed passes | BOTH | Selected query UV |
| --- | ---: | ---: | ---: | ---: | --- |
| Case001 original | 6 | 3 | 4 | 3 | (288.1,92.4) |
| Case002 original | 21 | 5 | 5 | 5 | (581.2,363.6) |
| Case002 generated | 16 | 11 | 8 | 8 | (562.1,428.7) |

Legacy trial outputs equal the previous independent diagnostic exactly. Fresh
combined-API repeats match label, trace and all evidence exactly. Cut answers and
old label bytes are unchanged. No replacement labels were persisted.

Both directions of the conjunction matter: case001's60.375mm query passes fixed
but fails legacy; case002 generated's127/132.125/137.25mm queries pass legacy but
fail fixed at anatomical knots86.798351 and91.287921mm. None is selected.

## Measured evidence size and CPU; packing remains unimplemented

| Frame order above | Selector wall s | Process CPU s | Full canonical evidence bytes | Selected composite trace bytes |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0.8003 | 0.796875 | 83032 | 6823 |
| 2 | 3.3425 | 3.328125 | 354879 | 6284 |
| 3 | 1.8435 | 1.843750 | 696310 | 21148 |

Selector intervals exclude source/catalogue loading and JSON serialization.
These are single-host measurements, not population yield or scaling predictions.
Canonical JSON here is sorted-key, compact UTF8 JSON plus one newline.
Full pretty diagnostic evidence is137380/611410/1202102 bytes.

Keep full diagnostic records now. A future capture receipt can retain compact
per-candidate dispositions, selected full trace/query evidence and a precisely
defined full-evidence digest ONLY if its new audit re-enumerates all41 candidates,
replays BOTH checks, reconstructs the full evidence and verifies that digest.
A digest alone is not a substitute for those checks. No packing/storage policy or
capture integration is implemented in this module.

