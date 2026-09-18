# Original phase inventory

`original_phase_inventory` reads explicitly pinned `serial_phases` receipts for
`original_batch.v1`. It verifies the exact producer, input/publication, serial39
predecessor, ordered job/checkpoint handoffs, submitted plans, launch declarations,
post-exit files and fresh saved native RGB/ID/Z/camera/label/trace replay.

```python
from sim_data.native_dataset.original_phase_inventory import PhaseCapturePin, build_inventory

packet = build_inventory([
    PhaseCapturePin(capture_path, result_sha256, receipt_path, receipt_sha256),
])
```

The result uses the common observed-inventory schema for a later disjoint
inventory join. All captured holds and exclusions survive. Pre-render holds
remain coverage only. Earlier receipts and qualification controls authenticate
the handoff but do not implicitly add observations. The phase need not have
completed later jobs. Any existing failure marker invalidates that source.

The original row projection is reused only after phase-specific authentication,
with an explicit field mapping. Provenance names the actual phase producer and
adapter; it does not claim an older launcher produced these captures. Original
ancestry, frozen TRAIN assignments, native 1696×816 and shared source caps remain.

This module never starts a renderer, queries current processes, writes captures,
approves training or grants geometry diversity. Process/resource evidence is a
checked producer declaration, not independent historical OS attestation.

Validation: 12 CPU tests passed in 117.06 seconds on 2026-09-16. Tests use
synthetic receipts and saved native-format buffers; they are not new captures
or proof that a queued phase has executed.
