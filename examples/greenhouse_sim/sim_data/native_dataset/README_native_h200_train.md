# Native full-frame + query-crop server route

Entrypoint: `python -B -m sim_data.native_dataset.native_h200_train`.
Do not use the root legacy trainer/evaluator for this release contract.
This route is CPU-tested code, **not a qualified H200/processor/model run**.
No model download, generation/scoring, release creation or ZIP builder is included.

## Required handoff

Use a matching reviewed code snapshot with `examples/greenhouse_sim` on
`PYTHONPATH`, the immutable portable native export directory, and a separately
transferred local dense Qwen3-VL-8B snapshot. The native modules and their pinned
dependencies must be present; a top-level-only legacy code kit is insufficient.
Python must be >=3.11. The preflight report must match the exact server Python,
package versions, processor files, effective tokenizer, template and source pins.
Do not reuse a report from a different environment. No installation/download
commands are supplied here. The earlier 4-H200 BF16/ZeRO-2 recipe remains a
starting hypothesis for these native images, not a capacity or accuracy result.

Required flags (see `--help` for the complete interface):

- `--dataset ABS --manifest-sha256 SHA --admission-sha256 SHA`
- `--model ABS --model-files ABS_JSON --model-files-sha256 SHA`
- `--preflight-report ABS_JSON --preflight-sha256 SHA`
- `--deepspeed ABS_JSON --deepspeed-sha256 SHA`
- `--output NEW_ABS --maximum-tokens EXPLICIT_N --mode smoke`
- Repeat `--control-id TRAIN_ID` for explicit controls in the pinned preflight.
- Repeat `--model-input-key KEY` for the processor's exact accepted tensor keys.
  Core keys are `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`;
  optional token-type keys require explicit support in the installed model's
  forward signature. Do not silently drop them or assume `**kwargs` is support.

`model-files` is an externally pinned filename-to-SHA256 JSON map: one
`model.safetensors`, or the index plus every referenced safetensors shard.
Include `generation_config.json` iff present. These are byte identities, not
proof of model origin or accuracy. The DeepSpeed JSON must exactly match the
unchanged `sim_data/clear_zero2.json`; bind its actual file hash explicitly.
Admission must retain the exact original 24-donor map and >=20,000 TRAIN rows
with explicit positive heldouts. Smoke/overfit do not bypass release admission.

## Staged operation, only after separate server authorization

First invoke the native entrypoint with all pinned arguments and **without**
`--execute-training`. It validates inputs and hashes files, writes nothing, and
does not load a processor or deserialize weights. This is not token/GPU evidence.

The future single-node launch prefix is:

```bash
torchrun --standalone --nproc_per_node=4 --max_restarts=0 \
  -m sim_data.native_dataset.native_h200_train
```

Append the complete pinned arguments and `--execute-training` only when ready
to authorize an actual server run. Start with `--mode smoke` (2 optimizer steps).
Use a different NEW output for `--mode overfit` (32 deterministic TRAIN IDs,
100 steps). Both retain microbatch1/accumulation8 defaults: effective batch32
on four ranks, with sampler repetitions explicitly not independent examples.
Native checkpoint reload/generation inspection is still a separate missing
qualification route. Do not promote low loss or a smoke completion to accuracy.
Only after that review, a separate authorized `--mode train` uses three epochs
by default and validation loss only. Test is never evaluated by this CLI.

The model receives original1696x816 RGB and an exact native768 query crop;
normalized1000 coordinates always refer to the original full frame. Depth and
supervision remain sidecars, never model inputs. Parity qualifies only selected
TRAIN controls; unchecked row lengths can still fail their per-row budget guard.
No truncation, image-budget adjustment or automatic fallback is provided.

Sources must remain immutable. Each rank validates independently; preparation
errors are gathered before weights. Runner-owned row handles close on failures
and exit. Nonfinite loss/gradient checks reduce an error flag across initialized
ranks; arbitrary CUDA faults and asymmetric callbacks still need real runtime
qualification and torchrun supervision/timeouts. No distributed test ran here.

Outputs are new-only: preparation/run receipts, model checkpoints (at most two
epoch checkpoints plus the final model), and `native_grounding_contract.json`.
Interrupted outputs remain incomplete; no automatic resume or cleanup occurs.
`completed.json` explicitly leaves reload, generation accuracy, GPU qualification
and approvals false. Pinned portable ZIP/code-kit work is deferred.
