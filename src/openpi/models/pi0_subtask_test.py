"""Regression tests for the pi0.5 subtask path.

Covers the four things that are easy to break silently:
  1. the discrete proprioceptive State span is present, and identical in the training and the
     inference prompt (they must align, since inference splices into the training format);
  2. the spliced subtask span stays causal at inference, matching how it was trained;
  3. the device-side greedy decode loop emits exactly what the eager loop it replaced did,
     including PAD-after-EOS per row;
  4. subtask_key / loss_mode stay consistent, so a subtask model cannot be trained without labels.

Tests needing the PaliGemma tokenizer download or a full model forward are marked
``@pytest.mark.manual`` (network / heavy), matching models/model_test.py conventions.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.models import pi0_config as _pi0_config
from openpi.models import tokenizer as _tokenizer
from openpi.training import config as _config

# ───────────────────────── discrete state in the subtask prompt ─────────────────────────


def test_discretize_state_str_differs_by_state():
    """Pure helper: distinct normalized states map to distinct token strings."""
    s1 = np.linspace(-1, 1, 32)
    s2 = -s1
    assert _tokenizer._discretize_state_str(s1) != _tokenizer._discretize_state_str(s2)  # noqa: SLF001
    # deterministic
    assert _tokenizer._discretize_state_str(s1) == _tokenizer._discretize_state_str(s1)  # noqa: SLF001


@pytest.mark.manual  # needs the PaliGemma tokenizer download (gs://big_vision, anon)
def test_subtask_tokenizer_includes_state():
    """The subtask tokenizers must encode `State:` when a state is given, and the high-level
    (inference) and high+low (training) prefixes must carry it the same way — inference splices
    the generated subtask into the training format, so a mismatch shifts every spliced token."""
    tok = _tokenizer.PaligemmaTokenizer(200)
    s1 = np.linspace(-1, 1, 32)
    s2 = -s1

    t_no = tok.tokenize_high_low_prompt("pick up cup", "grasp handle", state=None)[0]
    t_s1 = tok.tokenize_high_low_prompt("pick up cup", "grasp handle", state=s1)[0]
    t_s2 = tok.tokenize_high_low_prompt("pick up cup", "grasp handle", state=s2)[0]

    # state present -> strictly more real (non-pad) tokens than the state-less prompt
    assert int((t_s1 != 0).sum()) > int((t_no != 0).sum())
    # different states -> different token ids
    assert not np.array_equal(t_s1, t_s2)

    # the inference high-level prefix also carries the state
    h_no = tok.tokenize_high_level_prompt("pick up cup")[0]
    h_s1, h_mask = tok.tokenize_high_level_prompt("pick up cup", state=s1)
    assert int((h_s1 != 0).sum()) > int((h_no != 0).sum())

    # ...and it is a strict prefix of the training sequence: identical tokens up to where the
    # label begins. This is the alignment build_full_observation relies on — it splices the
    # generated subtask at exactly the inference prompt's length.
    h_len = int(h_mask.sum())
    np.testing.assert_array_equal(t_s1[:h_len], h_s1[:h_len])


@pytest.mark.manual  # constructs a PaliGemma tokenizer (network)
def test_subtask_transforms_pass_discrete_state():
    """Wiring guard: both subtask tokenize transforms must receive discrete_state_input=True,
    otherwise pi0.5 silently drops proprioceptive state."""
    cfg = _pi0_config.Pi0Config(pi05=True, subtask=True, paligemma_variant="dummy", action_expert_variant="dummy")
    assert cfg.discrete_state_input
    assert cfg.model_type == _model.ModelType.PI05_SUBTASK

    train_inputs = _config.ModelTransformFactory()(cfg).inputs
    hl = [t for t in train_inputs if isinstance(t, _transforms.TokenizeHighLowPrompt)]
    assert len(hl) == 1
    assert hl[0].discrete_state_input is True

    # policy_config swaps in the inference tokenizer; the flag must survive the swap.
    from openpi.policies import policy_config as _policy_config

    infer_inputs = _policy_config._to_inference_transforms(train_inputs)  # noqa: SLF001
    hp = [t for t in infer_inputs if isinstance(t, _transforms.TokenizeHighPrompt)]
    assert len(hp) == 1
    assert hp[0].discrete_state_input is True
    assert not any(isinstance(t, _transforms.TokenizeHighLowPrompt) for t in infer_inputs)


# ───────────────────────── the spliced subtask stays causal ─────────────────────────


def _obs(tokenized_prompt: np.ndarray, mask: np.ndarray) -> _model.Observation:
    batch = tokenized_prompt.shape[0]
    return _model.Observation(
        images={},
        image_masks={},
        state=jnp.zeros((batch, 32), dtype=jnp.float32),
        tokenized_prompt=jnp.asarray(tokenized_prompt),
        tokenized_prompt_mask=jnp.asarray(mask),
    )


def test_build_full_observation_marks_subtask_causal():
    """After splicing, the subtask+EOS span must be causal (ar=1) and everything else
    bidirectional (ar=0) — the attention pattern training used. build_full_observation touches no
    model state, so it can be called unbound."""
    length = 12
    tp = np.zeros((1, length), dtype=np.int32)
    tp[0, :4] = [2, 10, 11, 12]  # real prompt length 4
    mask = np.zeros((1, length), dtype=bool)
    mask[0, :4] = True
    subtask = jnp.asarray([[20, 21, 1]], dtype=jnp.int32)  # 3 tokens incl. EOS

    out = _pi0.Pi0.build_full_observation(object(), _obs(tp, mask), subtask)
    ar = np.asarray(out.token_ar_mask)[0].tolist()
    assert ar == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    # the spliced region is marked valid, and carries the generated ids
    assert bool(np.asarray(out.tokenized_prompt_mask)[0, :7].all())
    np.testing.assert_array_equal(np.asarray(out.tokenized_prompt)[0, 4:7], [20, 21, 1])
    # the CE mask is training-only and must not leak into inference
    assert out.token_loss_mask is None


def test_build_full_observation_masks_post_eos_filler():
    """generate_subtask pads a short row with PAD(0) after its EOS. Those slots must NOT become
    attendable prompt content — token id 0 would be read as if it were a word."""
    length = 12
    tp = np.zeros((2, length), dtype=np.int32)
    tp[:, :4] = [2, 10, 11, 12]
    mask = np.zeros((2, length), dtype=bool)
    mask[:, :4] = True
    # row 0 stops after 2 tokens; row 1 uses the full width.
    subtask = jnp.asarray([[20, 1, 0, 0], [20, 21, 22, 1]], dtype=jnp.int32)

    out = _pi0.Pi0.build_full_observation(object(), _obs(tp, mask), subtask)
    new_mask = np.asarray(out.tokenized_prompt_mask)
    ar = np.asarray(out.token_ar_mask)

    # row 0: prompt(4) + subtask(2) valid, the PAD filler at 6,7 is not
    assert new_mask[0, :6].all()
    assert not new_mask[0, 6:].any()
    assert ar[0].tolist() == [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    # row 1: all four generated slots are real
    assert new_mask[1, :8].all()
    assert ar[1, 4:8].tolist() == [1, 1, 1, 1]


# ───────────────────────── subtask_key / loss_mode consistency ─────────────────────────


def _cfg(**kwargs) -> _pi0_config.Pi0Config:
    return _pi0_config.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy", **kwargs)


def test_subtask_requires_pi05():
    with pytest.raises(ValueError, match="pi05"):
        _cfg(subtask=True)


def test_subtask_is_opt_in_so_existing_pi05_configs_are_unchanged():
    """pi05 alone must stay the flat model: the released pi05 checkpoints were trained that way,
    and flipping the default would repoint every existing pi05 config at the subtask pipeline."""
    assert _cfg(pi05=True).subtask is False
    assert _cfg(pi05=True).model_type == _model.ModelType.PI05


def test_ce_training_requires_subtask_labels():
    subtask_model = _cfg(pi05=True, subtask=True)
    with pytest.raises(ValueError, match="subtask_key"):
        _config._validate_subtask_config(_config.DataConfig(), subtask_model)  # noqa: SLF001
    # ...and with the key set it passes.
    _config._validate_subtask_config(_config.DataConfig(subtask_key="subtask"), subtask_model)  # noqa: SLF001


def test_fm_only_rejects_a_pointless_subtask_key():
    fm_only = _cfg(pi05=True, subtask=True, loss_mode="fm_only")
    _config._validate_subtask_config(_config.DataConfig(), fm_only)  # noqa: SLF001
    with pytest.raises(ValueError, match="fm_only"):
        _config._validate_subtask_config(_config.DataConfig(subtask_key="subtask"), fm_only)  # noqa: SLF001


def test_subtask_key_on_a_non_subtask_model_is_rejected():
    with pytest.raises(ValueError, match="not a subtask model"):
        _config._validate_subtask_config(_config.DataConfig(subtask_key="subtask"), _cfg(pi05=True))  # noqa: SLF001


def test_subtask_from_column_skips_unlabeled_frames():
    t = _transforms.SubtaskFromColumn("subtask")
    assert t({"subtask": "grasp the handle", "state": 1})["low_prompt"] == "grasp the handle"
    assert t({"subtask": "", "state": 1}) is None
    assert t({"subtask": "   ", "state": 1}) is None
    assert t({"subtask": None, "state": 1}) is None
    with pytest.raises(ValueError, match='no "subtask" column'):
        t({"state": 1})
    # the raw string column is consumed, so it cannot reach the collated batch
    assert "subtask" not in t({"subtask": "grasp", "state": 1})
    # numpy string scalars are what a dataset column actually yields
    assert t({"subtask": np.str_("grasp"), "state": 1})["low_prompt"] == "grasp"


def test_subtask_from_column_rejects_an_index_column():
    """Pointing subtask_key at an index column (the way LeRobot stores tasks) must fail loudly.
    str()ing it would silently train the model to emit "3"."""
    t = _transforms.SubtaskFromColumn("subtask_index")
    with pytest.raises(TypeError, match="not a string"):
        t({"subtask_index": 3, "state": 1})
    with pytest.raises(TypeError, match="not a string"):
        t({"subtask_index": np.int64(3), "state": 1})


def test_composite_transform_short_circuits_on_skip():
    """A None from a skipping transform must not be fed to the next one."""

    def boom(_data):
        raise AssertionError("should not run on a skipped sample")

    composed = _transforms.compose([_transforms.SubtaskFromColumn("subtask"), boom])
    assert composed({"subtask": ""}) is None


# ───────────── subtask generation: device-side loop == eager reference ─────────────


def _eager_generate_subtask_reference(model, observation, max_tokens):
    """The pre-optimization eager decode loop, kept as the reference implementation.

    Verbatim (modulo names) the loop `Pi0._subtask_generate` replaced: a Python loop whose
    `bool(jnp.all(done))` EOS check synced device->host every token. It drives the same jitted
    `prefill`/`decode_step` wrappers the optimized path is built from, so any divergence is in the
    loop bookkeeping (token writes, positions, attention masks, cache updates) rather than in the
    model.
    """
    import jax

    observation = jax.tree.map(jnp.asarray, observation)
    observation = _model.preprocess_observation(None, observation, train=False)
    prefill, decode_step = _pi0._subtask_jit_fns(model)[:2]  # noqa: SLF001

    logits, kv_cache, prefix_mask, next_pos = prefill(observation, max_tokens)
    b = prefix_mask.shape[0]
    generated = []
    done = jnp.zeros((b,), dtype=jnp.bool_)
    for i in range(max_tokens):
        next_token = jnp.argmax(logits[:, -1, :], axis=-1).astype(jnp.int32)
        next_token = jnp.where(done, jnp.zeros_like(next_token), next_token)
        generated.append(next_token)
        done = done | (next_token == _pi0.PALIGEMMA_EOS_TOKEN)
        if bool(jnp.all(done)):
            break
        gen_valid = jnp.broadcast_to(jnp.arange(max_tokens)[None, :] <= i, (b, max_tokens))
        step_valid = jnp.concatenate([prefix_mask, gen_valid], axis=1)
        logits, kv_cache = decode_step(next_token, step_valid[:, None, :], next_pos[:, None], kv_cache)
        next_pos = next_pos + 1
    return jnp.stack(generated, axis=1)


def _dummy_subtask_model(key_seed: int = 0):
    import jax

    cfg = _pi0_config.Pi0Config(
        pi05=True,
        subtask=True,
        max_token_len=32,
        dtype="float32",
        paligemma_variant="dummy",
        action_expert_variant="dummy",
    )
    return cfg, cfg.create(jax.random.key(key_seed))


@pytest.mark.manual  # two full (dummy) subtask generations incl. SigLIP So400m — heavy on CPU
@pytest.mark.parametrize("batch_size", [1, 2])
def test_subtask_generate_matches_the_eager_reference(batch_size):
    """The device-side loop must emit byte-identical tokens to the eager loop it replaced.

    That is the safety property of the optimization: it exists only to remove per-token host
    syncs, so any change in the emitted tokens is a bug.

    Covers the no-EOS path specifically. With random dummy weights argmax lands on EOS (id 1 of a
    257k-token vocabulary) with probability ~4e-6 per token, so the loop here runs to max_tokens
    and what gets compared is the full-length bookkeeping: token writes, RoPE positions, attention
    masks and cache updates. Early exit and the PAD-after-EOS contract need EOS forced — see
    test_subtask_generate_pads_after_eos_per_row.
    """
    cfg, model = _dummy_subtask_model()
    obs = cfg.fake_obs(batch_size)
    max_tokens = 6

    reference = np.asarray(_eager_generate_subtask_reference(model, obs, max_tokens))
    got = np.asarray(model.generate_subtask(obs, max_tokens=max_tokens))

    # Contract: always (B, max_tokens), PAD(0) after each row's EOS (never trimmed).
    assert got.shape == (batch_size, max_tokens)
    n = reference.shape[1]
    np.testing.assert_array_equal(got[:, :n], reference)
    assert not got[:, n:].any(), "tail past the reference's stop must be PAD(0)"

    # Deterministic: greedy argmax, and preprocess_observation is called with rng=None.
    np.testing.assert_array_equal(got, np.asarray(model.generate_subtask(obs, max_tokens=max_tokens)))


@pytest.mark.manual  # full (dummy) subtask generation incl. SigLIP So400m — heavy on CPU
def test_subtask_generate_pads_after_eos_per_row(monkeypatch):
    """PAD-after-EOS is per row, so a short row is not dragged past its EOS by a longer one.

    Guards the invariant build_full_observation depends on: it re-derives validity as
    `cumsum(is_eos) - is_eos == 0`, which is only meaningful if nothing real follows a row's EOS.

    EOS has to be forced to reach that invariant at all. Random dummy weights hit the real EOS
    (id 1 of a 257k-token vocabulary) with probability ~4e-6 per token, so an unforced run never
    emits one and every assertion below would pass vacuously. So redefine EOS to be a token the
    model provably emits — row 0's first token, read off prefill. Row 1 gets a different image and
    state from fake_obs and so generally does not stop on it, giving the staggered per-row stop
    that is the case at issue.
    """
    import jax

    cfg, model = _dummy_subtask_model()
    obs = cfg.fake_obs(2)
    max_tokens = 6

    # Probe through `prefill` alone, which holds no EOS logic, so the constant stays untraced.
    # Order matters: `_subtask_generate` bakes PALIGEMMA_EOS_TOKEN in as a traced constant and
    # _SUBTASK_JIT_FNS caches per id(model), so patching after a generate_subtask call on this
    # model would silently compile against the old value.
    prefill = _pi0._subtask_jit_fns(model)[0]  # noqa: SLF001
    prepped = _model.preprocess_observation(None, jax.tree.map(jnp.asarray, obs), train=False)
    forced_eos = int(jnp.argmax(prefill(prepped, max_tokens)[0][0, -1, :]))
    assert forced_eos != 0, "forced EOS collided with PAD(0); reroll the dummy init key"
    monkeypatch.setattr(_pi0, "PALIGEMMA_EOS_TOKEN", forced_eos)

    tokens = np.asarray(model.generate_subtask(obs, max_tokens=max_tokens))
    assert tokens.shape == (2, max_tokens)

    # Row 0 emits the forced EOS as its very first token — same prefill logits, and `done` is
    # still all-False so nothing masks it — so its whole tail must be PAD. This is the non-vacuous
    # form of the padding contract.
    assert tokens[0, 0] == forced_eos, (
        f"row 0 should open with the forced EOS {forced_eos}, got {tokens[0]}. If the probe's "
        "prefill and the one traced inside generate disagree on a near-tied argmax, take the "
        "forced EOS from a full unforced generation instead."
    )
    assert not tokens[0, 1:].any(), f"row 0 stopped at step 0, so its tail must be PAD(0): {tokens[0]}"
    for row in tokens:
        eos = np.flatnonzero(row == forced_eos)
        if eos.size:
            assert not row[eos[0] + 1 :].any(), f"non-PAD token after EOS in {row}"

    # The device loop must still match the eager loop it replaced on this stop pattern — the early
    # exit and the post-EOS masking are exactly what the unforced test cannot reach.
    reference = np.asarray(_eager_generate_subtask_reference(model, obs, max_tokens))
    n = reference.shape[1]
    np.testing.assert_array_equal(tokens[:, :n], reference)
    assert not tokens[:, n:].any(), "tail past the reference's stop must be PAD(0)"


@pytest.mark.manual  # full (dummy) model forward incl. SigLIP So400m — heavy on CPU
def test_loss_mode_dispatch():
    """ce_only must drop the flow-matching term and fm_only the cross-entropy, from the same
    single forward pass."""
    import dataclasses

    import jax

    cfg, _ = _dummy_subtask_model()
    obs, act = cfg.fake_obs(1), cfg.fake_act(1)
    rng = jax.random.key(0)

    both = cfg.create(jax.random.key(0)).compute_loss(rng, obs, act, train=False)
    ce_only = (
        dataclasses.replace(cfg, loss_mode="ce_only").create(jax.random.key(0)).compute_loss(rng, obs, act, train=False)
    )
    fm_only = (
        dataclasses.replace(cfg, loss_mode="fm_only").create(jax.random.key(0)).compute_loss(rng, obs, act, train=False)
    )

    assert both.shape == ce_only.shape == fm_only.shape == (1, cfg.action_horizon)
    # ce_only is the CE broadcast across the horizon, so it is constant along it.
    assert np.allclose(np.asarray(ce_only), np.asarray(ce_only)[:, :1])
    np.testing.assert_allclose(np.asarray(both), np.asarray(ce_only) + np.asarray(fm_only), rtol=1e-4, atol=1e-5)
