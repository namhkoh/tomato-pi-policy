# Qwen3-VL-8B: server execution runbook

2026-09-11, branch `koh-dev/sim-vlm`. Run these commands **on the H200 server**,
not the simulator PC. The user installs/downloads Qwen on that server.
`h200_train.py` accepts an existing local model directory and prohibits model
downloads. Nothing in this handoff automatically starts a remote job.

Status: normal dataset validation and the full-corpus chat/coordinate audit
passed. The new server recipe has helper/adapter tests, **not a real-Qwen,
forward/backward, checkpoint-reload or four-H200 qualification**. The mandatory
server smoke tests below establish those remaining conditions.

## What is actually ready

Canonical release: `visible_occluded_20260911_v1`, profile
`visible_occluded_v1`, state `complete_visible_occluded_baseline_release`.

Published data ZIP: `data/sim_data/training_archives/visible_occluded_20260911_v1.zip`,
23,453,235,122 bytes (23.45 GB decimal), 71,306 independently verified members.
SHA-256: `23e180dcb74f6daf0437414513b023d41d76f29e755aa1e2fb1638f6b268db74`.
Receipt: `data/sim_data/dataset_audits/release_20260911_agents/baseline_archive_receipt.json`.

| Frozen split | Records | Visible/localized | Occluded/abstain |
|---|---:|---:|---:|
| Train | 11,520 | 9,051 | 2,469 |
| Validation | 1,358 | 1,012 | 346 |
| Test | 1,381 | 1,174 | 207 |
| Total | 14,259 | 11,237 | 3,022 |

24 target-source families (16/4/4), 246 target IDs, 136 exact representative
assistant QA accepts. These are not 14,259 individually human-reviewed images.
Shared greenhouse background is not fully scene-disjoint. Medium/partial,
Cosmos real-hard and dynamic action/outcome episodes are absent.

Input: unchanged 848x408 mounted robot-head RGB plus a visible-target query.
Output: JSON cut point/visibility/abstention/next inspection action. This is
target-conditioned perception, not autonomous target discovery, metric XYZ,
grasp/cut trajectories, a VLA or physically approved blade commands.

Native Isaac `distance_to_image_plane` optical-Z metres, validity, calibration,
identity masks and evaluator geometry are sidecars. **They do not enter the
RGB model.** Occluded answers keep `cut_point_uv: null`; hidden simulator cut
XYZ cannot be substituted in training input or execution.

## Qwen instructions/cookbook compatibility

Checked against the official [fine-tuning framework](https://github.com/QwenLM/Qwen3-VL/tree/main/qwen-vl-finetune),
[data processor](https://github.com/QwenLM/Qwen3-VL/blob/main/qwen-vl-finetune/qwenvl/data/data_processor.py),
[8B model card](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct), and
[2-D grounding guidance](https://github.com/QwenLM/Qwen3-VL/issues/1616).
Upstream files are moving references; record the installed model/package revisions.

| Concern | Our explicit handling |
|---|---|
| Dataset container | Release uses `images` + `messages`. Upstream example loader uses `image` + `conversations`; do **not** point it directly at these split JSONLs. |
| Roles | Preserve system/user/assistant. The checked upstream loader maps non-`human` conversations to assistant, so a blind conversion would corrupt our system role. |
| Model input | `AutoProcessor.apply_chat_template(... tokenize=True, return_dict=True, return_tensors='pt')`, one original PIL RGB image; the processor performs its own resizing. |
| Loss | Only assistant answer/end tokens are supervised. Prompt and image tokens get `-100`; a real generation-prefix equality and image-grid check gates encoding. No hand-guessed token IDs. |
| Coordinates | Server default converts BOTH query and answer to normalized 0..1000 coordinates through `qwen_coordinates.py`. Canonical release pixels remain unchanged. |
| Output | Our four-field JSON is a custom task schema, not a default `bbox_2d` example. Custom output formats are supported; the task needs points and abstention, not boxes. |
| Fine-tuning | Official Transformers Qwen3-VL + PEFT language-attention LoRA. This small recipe is an integration, **not the upstream trainer unmodified**. |

Qwen recommends relative 0..1000 grounding coordinates. We preserve continuous
values to avoid integer rounding of small petiole locations. Normalization is
`[1000*u/848, 1000*v/408]`; decoding uses `[848*x/1000, 408*y/1000]`.
The actual conversion of all 14,259 rows passed; maximum roundtrip error was
`1.14e-13` original pixels. This is arithmetic verification, not model accuracy.
Strict coordinates exclude the far image boundary; malformed/out-of-frame
model outputs fail rather than being clipped into a valid answer.

For example, canonical point `[212,102]` becomes `[250,250]`; a canonical
query `[424,204]` becomes `[500,500]`. The normalized system prompt explicitly
declares this frame. `answer_to_pixels()` must be used before canonical scoring.
The saved `adapter/grounding_adapter.json` declares the convention. Never infer
the scale from coordinate magnitude. `--coordinates pixels` is an explicit
legacy-contract ablation, not the recommended Qwen-native run.

## 1. Get matching code and transfer the dataset

Repository: `https://github.com/namhkoh/tomato-pi-policy.git`, branch
`koh-dev/sim-vlm`. A local commit is not available through GitHub until pushed;
verify that your checkout actually contains `sim_data/h200_train.py` and this
runbook. Do not accidentally use the older code-only archive lacking the runner.

Fresh server checkout, after this revision is available on origin:

```bash
git clone --branch koh-dev/sim-vlm --single-branch https://github.com/namhkoh/tomato-pi-policy.git tomato-vlm
cd tomato-vlm
git rev-parse HEAD
cd examples/greenhouse_sim
test -f sim_data/h200_train.py
```

For an existing clean checkout, fetch and fast-forward the same branch; do
not reset a dirty checkout. Alternatively transfer the matching **v2 code ZIP**
and checksum alongside the data ZIP, extract it, and work from
`greenhouse_training_code/examples/greenhouse_sim`. That avoids copying Isaac
or the full simulator assets. Preserve its code receipt/commit.

The publication step, if not already done, is `git push origin koh-dev/sim-vlm`
from the reviewed simulator checkout. It is distinct from committing locally;
this runbook does not claim the new revision is already on GitHub.

Transfer `visible_occluded_20260911_v1.zip` and its `.zip.sha256`. A `.partial`
file is NOT the final archive. Check the published archive receipt before use.
The data is not stored in git; pulling code does not transfer the dataset.

Choose your actual server paths (examples below require replacing `/data/...`):

```bash
export VLM_TRANSFER=/data/deleaf-transfer
export VLM_DATA=/data/deleaf-data/grounding_release
export VLM_MODEL=/data/models/Qwen3-VL-8B-Instruct
export VLM_RUNS=/data/deleaf-runs
cd "$VLM_TRANSFER"
sha256sum -c visible_occluded_20260911_v1.zip.sha256
test ! -e "$VLM_DATA"
unzip visible_occluded_20260911_v1.zip -d /data/deleaf-data
mkdir -p "$VLM_RUNS"
```

Keep the entire `grounding_release`, including native depth, labels, masks,
manifest and frozen chats. Read-only mount/permissions are recommended while
training; do not edit the release after validation. Allow space for both ZIP
and extracted files, model snapshot, optimizer/checkpoints and run logs.

## 2. Server environment and model snapshot

Use a separate Python 3.11 environment. The server owner may use an existing
known-working Qwen3-VL stack. For a starting recipe, not a measured H200 lockfile:

```bash
python3.11 -m venv /data/venvs/deleaf-qwen
source /data/venvs/deleaf-qwen/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install transformers==4.57.6 accelerate==1.11.0 peft==0.17.1 numpy==1.26.4 scipy==1.15.3 Pillow==11.3.0 pytest==8.4.2
python -m pip check
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.device_count()); assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported()'
```

The Torch/CUDA wheel must be compatible with the host driver. Do not modify
system drivers as a guessed fix. SDPA is used initially; FlashAttention,
DeepSpeed, quantization and packing are deliberately not prerequisites.

Install a pinned **Qwen/Qwen3-VL-8B-Instruct** dense BF16 snapshot at
`$VLM_MODEL` using your server's model setup. Include configuration, tokenizer,
chat template, image/video processor configs, weight index and every weight
shard. Record the model repository revision and your download verification.
The runner hashes its config but that is NOT a full weight-shard integrity check.
Do not use the 32B evaluation model, Thinking, MoE, or an FP8/quantized snapshot
as an unnoticed replacement. No credentials belong in git or run arguments.

Return to the checkout's `examples/greenhouse_sim` before subsequent commands.

## 3. Validate data and adapter before weights

```bash
python -m sim_data.training_export validate --output "$VLM_DATA"
python -m sim_data.qwen_format_audit --dataset "$VLM_DATA" --output "$VLM_RUNS/qwen-format-audit.json"
python -m pytest sim_data/qwen_adapter_test.py sim_data/qwen_coordinates_test.py sim_data/h200_train_test.py -q
python -m pip freeze > "$VLM_RUNS/environment.txt"
git rev-parse HEAD > "$VLM_RUNS/code-commit.txt"
```

Use new report paths on rerun. Normal validation reads every bound artifact
and checks labels/coverage/QA; allow minutes rather than treating silence as a
crash. It needs no Isaac installation or original Windows source paths.
The training recipe revalidates on rank zero before weights, propagating
failures to all ranks. No incomplete-release bypass exists.

## 4. One-GPU smoke (two optimizer steps, two train examples)

```bash
CUDA_VISIBLE_DEVICES=0 python -m sim_data.h200_train \
  --dataset "$VLM_DATA" --model "$VLM_MODEL" \
  --output "$VLM_RUNS/smoke-01" --mode smoke
```

Requires actual processor-prefix/image-grid agreement and supervised token
decoding equal to the converted answer. Then requires finite scalar losses,
finite nonzero LoRA gradients, two optimizer steps, and saved adapter/processor.
Only language-attention projections are trainable; vision and mergers stay
frozen. `run_contract.json` records versions, exact train IDs, conventions,
hyperparameters, manifest hash, trainable names and processor checks.

Stop on any failure. Do not bypass it by masking all tokens, truncating
labels, dropping invalid records, changing frozen splits or removing the
system prompt. A saved adapter alone does not establish useful prediction.

## 5. Small train-only overfit and reload

```bash
CUDA_VISIBLE_DEVICES=0 python -m sim_data.h200_train \
  --dataset "$VLM_DATA" --model "$VLM_MODEL" \
  --output "$VLM_RUNS/overfit-01" --mode overfit
```

This uses 16 visible and 16 occluded training examples, 100 optimizer steps.
Confirm decreasing finite loss and inspect greedy predictions on those same
32 examples with the saved adapter. Do not call training loss held-out accuracy.
Reload must succeed in a fresh process, with local base weights and saved PEFT
adapter, before four-GPU training. For example (server-side interactive Python):

```python
import json, os, torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from sim_data.qwen_adapter import model_messages
from sim_data.qwen_coordinates import answer_to_pixels
from sim_data.training_export import read_jsonl

root = Path(os.environ['VLM_DATA'])
adapter = Path(os.environ['VLM_RUNS']) / 'overfit-01' / 'adapter'
contract = json.loads((adapter / 'grounding_adapter.json').read_text())
assert contract['coordinates'] == 'normalized_1000'
processor = AutoProcessor.from_pretrained(adapter, local_files_only=True)
base = Qwen3VLForConditionalGeneration.from_pretrained(
    os.environ['VLM_MODEL'], local_files_only=True,
    dtype=torch.bfloat16, attn_implementation='sdpa')
model = PeftModel.from_pretrained(base, adapter, local_files_only=True).to('cuda').eval()
row = next(read_jsonl(root / 'splits/train.jsonl'))
messages = model_messages(row, root, coordinates='normalized_1000')
assert [m['role'] for m in messages] == ['system', 'user']
inputs = processor.apply_chat_template(messages, tokenize=True,
    add_generation_prompt=True, return_dict=True, return_tensors='pt').to('cuda')
with torch.inference_mode():
    output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
raw = processor.tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
print('RAW:', raw)
print('CANONICAL PIXELS:', answer_to_pixels(json.loads(raw)))
```

Malformed JSON/coordinates must be recorded as model failures, not repaired
into a ground-truth answer. Repeat over the 32 selected IDs. The inference
message excludes the assistant answer and all evaluator-only depth/geometry.

## 6. Four-H200 initial LoRA run

Only after the previous checks pass and all four GPUs are allocated to you:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m sim_data.h200_train --dataset "$VLM_DATA" --model "$VLM_MODEL" \
  --output "$VLM_RUNS/lora-01" --mode train --epochs 1 \
  --learning-rate 0.0001 --gradient-accumulation 8
```

Starting hyperparameters, NOT measured optima: BF16, SDPA, LoRA rank16/alpha32,
dropout0.05, language attention q/k/v/o projections, microbatch1, 8 accumulated
steps per rank, effective32 examples/update, seed41, gradient checkpointing,
gradient norm1, cosine schedule with 3% warmup. DDP replicates the 8B model
on each GPU; this is not model sharding. Measure memory and throughput before
increasing batches. Loss averages individual examples' assistant-token losses;
it is not a global variable-length token-weighted objective.

First run is one epoch; use validation loss and **validation generation** to
decide later epochs. The script evaluates validation loss, saves epoch/final
adapters and never reads test examples. It does not automatically resume or
overwrite output directories. Failed output folders remain for diagnosis;
restart only with a new directory after understanding the failure.

## 7. Evaluation and return-to-simulator gate

Generate untouched validation predictions for untuned 8B and tuned 8B using
the same normalized prompt. Save raw answers, IDs, coordinate contract and
latency. Decode with `answer_to_pixels` before `training_evaluate.score`.
Report invalid JSON separately; that scorer intentionally rejects invalid
answers and is not by itself a complete raw-generation evaluation runner.

Report JSON validity, localization coverage/error, projected 10-20 mm interval
hits, false localization under occlusion, per-family/class results and p50/p95
latency. Keep original image resolution for scoring. The interval hit score
is a 2-D proxy, not permission to execute a metric cut. Evaluate test only after
locking the model/configuration; do not tune on test labels. A fall in loss,
successful overfit, or source-family split does not establish real-world safety.

No current throughput, memory, accuracy, or full-robot VLM result has been
measured. Once perception is qualified, a separate guarded bridge must verify
fresh native depth/calibration, visible target identity, grasp, updated cut
location and collision-free tool access. Physical grasp/cut/retain/deposit
remains independent simulator work; this static dataset does not train it.
