# Original reference bank v2

This separate bank accepts three exact pin types through their named inventory
readers: `OriginalCapturePin` (V5 pilot), `SerialCapturePin` (serial39), and
`PhaseCapturePin` (`serial_phases` original batches). It authenticates each
producer and replays saved native evidence before building reference entries.

```python
from sim_data.native_dataset import original_reference_bank_v2 as bank

result = bank.build_bank([bank.PhaseCapturePin(
    capture_path, result_sha256, receipt_path, receipt_sha256,
)], reviews=[bank.ReviewPin(review_path, review_sha256)])
```

`write_bank`, `check_bank`, and `verify_anchor` provide create-only publication
and full replay. The bank and anchor proof have v2 schemas. Frozen v1 consumers
reject them; production integration requires an explicit versioned consumer.

Historical 848×408 evidence supplies the pose. Fresh 1696×816 observations and
the current clear-plan assets supply native and scene evidence. Existing v1
chain, scene, optics, review and source-limit checks remain in force. Every
planned case survives, including captured exclusions and pre-render holds.
Missing, negative or conflicting visual reviews withhold reference readiness.

The bank grants no additional geometry budget, training approval or new images.
It never launches a renderer or modifies sources. Process and reviewer identity
remain declared provenance, not independent historical attestation.

Validation: seven CPU tests passed in 129.06 seconds on 2026-09-16. They exercise
all three producer readers, saved-buffer replay, review conflicts, both source
chains, mixed inputs and schema rejection using explicitly synthetic fixtures.
