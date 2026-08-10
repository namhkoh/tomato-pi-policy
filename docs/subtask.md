# pi0.5 subtask generation

A pi0.5 subtask model predicts a low-level subtask in language before it predicts actions. The
prompt it sees is

```
[BOS] Task: {high-level task}, State: {256-bin quantized state}. Subtask: {low-level subtask} [EOS]
```

Everything up to `Subtask: ` is bidirectional, and the subtask span itself is causal and supervised
with cross-entropy. The action expert then denoises actions conditioned on the whole thing, so the
flow-matching loss and the subtask cross-entropy come out of a single forward pass.

At inference the low-level subtask does not exist yet: only the high-level prefix is tokenized, the
model generates the subtask autoregressively, and the generated tokens are spliced back into the
prompt before actions are sampled.

This is a JAX-only feature. Loading a subtask checkpoint through the PyTorch path raises, because
`PI0Pytorch` has no generation stage and would silently serve actions from a prompt that still ends
at `Subtask: `.

## Enabling it

`subtask` is opt-in, not implied by `pi05` — every existing `pi05_*` config keeps training and
serving the flat way against the released flat checkpoints.

```python
TrainConfig(
    name="pi05_subtask_mydata",
    model=pi0_config.Pi0Config(pi05=True, subtask=True),
    data=LeRobotLiberoDataConfig(
        repo_id="my/dataset",
        base_config=DataConfig(prompt_from_task=True, subtask_key="subtask"),
    ),
)
```

Two knobs matter:

- **`DataConfig.subtask_key`** — the dataset column holding the per-frame subtask label. The LeRobot
  revision this repo pins predates native subtask support (`subtask_index` +
  `meta/subtasks.parquet`), so the label comes from a plain column your conversion script writes.
  Frames whose label is empty are skipped rather than trained on, since an empty label teaches the
  model to emit nothing. The high-level task comes from `prompt_from_task` as usual.
- **`Pi0Config.loss_mode`** — `both` (default), `ce_only`, or `fm_only`. Staged training (subtask
  first, then actions) is a training choice, not a separate model. `fm_only` trains without any
  subtask supervision and therefore takes no `subtask_key`; the two are validated against each
  other at config-creation time so neither mistake is silent.

`max_token_len` defaults to 200 for pi0.5, against a typical prefix of 60-130 tokens plus 48
generated. Lengthening the State span or raising `max_tokens` eats into that margin;
`build_full_observation` warns when the splice would overflow.

## Serving

`scripts/serve_policy.py` needs no changes — `create_trained_policy` swaps the training tokenizer
for the inference one automatically. The websocket server accepts two extra fields alongside the
observation:

- `subtask_only: true` — run generation only and return `{"subtask": "..."}` without sampling
  actions. Useful for offline captioning/labeling runs.
- `force_subtask_regen: true` — ignore the cache and regenerate.

Responses from a subtask policy carry the generated caption in `subtask`, and
`policy_timing.subtask_ms` / `policy_timing.subtask_cached` alongside `infer_ms`.

### About the cache

Generation is cached per connection, keyed on the tokenized prompt. Note what that key does and
does not cover: the pi0.5 prompt embeds the 256-bin quantized state, so the cache misses whenever
the state crosses a bin — regeneration is the steady-state cost while the robot moves, not a
one-off. Conversely the key does **not** cover the images, so a client that holds the arm still
while the scene changes keeps getting the same caption until it asks for a regen.

## Testing

`src/openpi/models/pi0_subtask_test.py`. The fast tests cover prompt/splice alignment and the
config validation. The ones that need the tokenizer download or a full dummy-model forward are
marked `manual`:

```bash
uv run pytest src/openpi/models/pi0_subtask_test.py -m manual
```
