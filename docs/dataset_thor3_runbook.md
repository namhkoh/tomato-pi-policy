# thor3 — two RTX PRO6000 collection runbook

**Current execution mode: [independent server collection](dataset_independent_servers.md). No connection to the Windows PC or shared coordinator is required.** The independent guide supersedes coordinator setup/reservation steps below; all image, annotation and Linux qualification requirements still apply.

**Shared goal: at least 25,000 accepted distinct samples collectively by Saturday 2026-09-19, 23:59 KST.** Follow the [shared contract](dataset_multihost_runbook.md) for complete sensor outputs, populated backgrounds, annotation, duplicates and final counts.

## Exact assignment

- Config: [`configs/dataset_capture/thor3.json`](../configs/dataset_capture/thor3.json).
- Machine ID: `thor3`; GPU worker IDs: `0,1`, one RTX PRO6000 per worker.
- Foreground donors: **seed17, seed23, seed53, seed71, seed83, seed89** (`seedN_full` identifiers).
- New generator seed range: **3,000,000–3,999,999**.
- Output prefix: `captures/diverse9mm_20260918_multihost_v1/thor3/worker_<ID>/`.
- Original donor descendants remain TRAIN. Keep the full 144 labeled unpruned greenhouse and visible background vines.

thor3 reported no active production collection; its current task was a two-image Linux preview. The downloaded 131-proposal request is historical and must not become the production queue. See the [RTX PRO hardware guide](dataset_dual_rtx_pro_6000.md). The Linux capture/ownership adapter must pass qualification before these configs can launch production.

## Execute in this order

1. Use the same coordination/generator/annotation revision and pinned HF asset release as thor1. Rebuild only destination-specific bindings and additive Linux adapter code; do not rewrite frozen annotation math.
2. Select this host's config and verify the shared lock:

   ```bash
   export HOST_CONFIG=configs/dataset_capture/thor3.json
   export PYTHONPATH="$PWD/examples:$PWD/examples/greenhouse_sim"
   export OPENBLAS_NUM_THREADS=1
   python scripts/dataset_multihost_pipeline.py --config "$HOST_CONFIG" verify
   ```

3. Invoke the full-plant generator using the locally authenticated source plan. This example starts with a supported 53/41 pair:

   ```bash
   python scripts/dataset_multihost_pipeline.py --config "$HOST_CONFIG" generate \
     --source-plan "$LOCALIZED_SOURCE_PLAN" --source-plan-sha256 "$SOURCE_PLAN_SHA256" \
     --family seed53_full --primary-target SubStem_41 \
     --controlled-targets SubStem_41 SubStem_42 --seed 3000000 \
     --output data/sim_data/generated_plants/thor3_seed53_3000000
   ```

   Generation and 9mm evidence do not establish native visibility or a unique answer. Rebuild the full scene census/catalogue for generated geometry; preserve all 143 other complete plants. Rotate among the assigned donor/target combinations and 3–4 meaningful view angles, rejecting near repeats.

4. Initialize one **host-local offline ledger** for this server using the [independent collection guide](dataset_independent_servers.md#3-initialize-a-host-local-ledger). No shared URL/token or SSH connection is needed. All GPUs on this host share this ledger; do not copy the live local5090 database.
5. Reserve fresh requests through `scripts/dataset_offline_shard.py reserve-raw` or `reserve-generated`, using this host's ledger and worker IDs. Follow the exact commands in the independent guide. Keep the returned task/claim IDs, use fresh outputs, and launch only the assigned worker when `capture_request_eligible` is true.
6. Finish the existing two-image preview on GPU 0 through Isaac Sim Replicator. Validate native 848×408 RGB, aligned float32 optical-Z depth in metres, validity, instance IDs/mapping, camera calibration and actual robot/collision/workspace evidence. Check full 144 background coverage and compare annotation against the shared controls. Mark these frames as excluded controls.
7. After native owner/GPU/annotation parity passes, launch two persistent workers on verified distinct physical GPU UUIDs. Preserve the 8-subframe baseline profile until a faster setting is separately qualified. Keep the populated stage/render product/writer resident across its assigned batch.
8. Run the common locked annotation command after successful native closure, record local ledger duplicate results and mark global merge pending, perform group/flagged-case visual review and provide reviewed exports to the single publisher. Report accepted samples/hour and counts per donor/target/morphology each hour.

**Before bulk launch, the thor3 agent must record:** the tested Linux command, both physical GPU UUIDs, shared toolchain hash, local offline ledger path, successful preview/closure/annotation parity receipts, exact output directory and measured accepted-image rate. The goal is 25k collectively; duplicate or merely rendered frames never satisfy that goal.
