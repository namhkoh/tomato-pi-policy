import dataclasses
import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config

# gemma_05 is gemma.py plus the two things autoregressive subtask decoding needs: a fixed-size
# (idx, k, v) KV cache that can be appended to one token at a time, and deembed() for logits.
# Its non-decoding behaviour is identical, so the flat pi0/pi0.5 paths are unchanged.
import openpi.models.gemma_05 as _gemma
import openpi.models.siglip as _siglip
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("openpi")

# PaliGemma end-of-sequence token id; subtask autoregressive generation stops here.
PALIGEMMA_EOS_TOKEN = 1

# Per-model jitted subtask-generation entry points. Keyed by id() with a strong reference to the
# model (identity check), because nnx modules make no promises about hashability/weakref support.
# Entries live for the process — serving builds one model, and module_jit's frozen state pins the
# params either way. max_tokens is a static argument, so one wrapper serves any value (XLA
# recompiles per distinct value).
#
# `generate` is what serving calls: prefill + the whole decode loop inside ONE jit. The separate
# `prefill`/`decode_step` wrappers are kept because they are the reference implementation the
# equivalence test drives the eager loop with.
_SUBTASK_JIT_FNS: dict[int, tuple] = {}


def _subtask_jit_fns(model: "Pi0"):
    entry = _SUBTASK_JIT_FNS.get(id(model))
    if entry is not None and entry[0] is model:
        return entry[1]
    fns = (
        # jitted fun signature is (state, observation, max_tokens) -> static idx 2.
        nnx_utils.module_jit(model._subtask_prefill, static_argnums=(2,)),  # noqa: SLF001
        nnx_utils.module_jit(model._subtask_decode_step),  # noqa: SLF001
        nnx_utils.module_jit(model._subtask_generate, static_argnums=(2,)),  # noqa: SLF001
    )
    _SUBTASK_JIT_FNS[id(model)] = (model, fns)
    return fns


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.subtask = config.subtask
        self.loss_mode = config.loss_mode
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # Full attention between image and language inputs. For the subtask (pi0.5) path the
            # CAUSAL subtask span is applied by the callers via observation.token_ar_mask, so the
            # high-level prompt stays bidirectional and only the subtask tokens are causal.
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    # ───────────────────────── subtask (pi0.5) machinery ─────────────────────────
    # Inference is a 2-stage process orchestrated by Policy.infer:
    #   stage 1: generate_subtask (jitted device loop) -> stage 2: build_full_observation ->
    #   sample_actions. The model only exposes the building blocks; sample_actions itself stays
    #   the flat path, reading the spliced prompt like any other prompt.
    def _subtask_prefill(self, observation: _model.Observation, max_tokens: int):
        """Jitted stage-1 prefill: SigLIP + full-prefix forward, cache room for max_tokens slots.

        Returns (last-token logits, kv_cache, prefix_mask, next logical position).
        """
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        b = prefix_tokens.shape[0]
        prefix_s = prefix_tokens.shape[1]

        # Prefill: reserve cache room for max_tokens future slots (gemma_05 sizes the fixed cache
        # to attn_mask.shape[-1]).
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        prefix_attn_mask = jnp.pad(prefix_attn_mask, ((0, 0), (0, 0), (0, max_tokens)))
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        (prefix_out, _), kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None], mask=prefix_attn_mask, positions=positions, adarms_cond=[None, None]
        )

        # Logits from the last *real* prefix token (skip right-padding).
        seq_idx = jnp.arange(prefix_s)[None, :]
        last_pos = jnp.max(jnp.where(prefix_mask, seq_idx, -1), axis=1).astype(jnp.int32)
        last_hidden = prefix_out[jnp.arange(b), last_pos, :]
        logits = self.PaliGemma.llm(last_hidden[:, None, :], method="deembed")

        num_real = jnp.sum(prefix_mask, axis=1).astype(jnp.int32)  # logical position of next token
        return logits, kv_cache, prefix_mask, num_real

    def _subtask_decode_step(self, next_token, attn_mask, positions, kv_cache):
        """Jitted stage-1 decode step: embed one token, suffix forward with KV cache, logits."""
        next_emb = self.PaliGemma.llm(next_token[:, None], method="embed")
        (new_out, _), kv_cache = self.PaliGemma.llm(
            [next_emb, None],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, None],
            kv_cache=kv_cache,
        )
        logits = self.PaliGemma.llm(new_out, method="deembed")
        return logits, kv_cache

    def _subtask_generate(self, observation: _model.Observation, max_tokens: int) -> at.Int[at.Array, "b t"]:
        """Jitted stage-1 generation: prefill + the whole greedy decode loop on device.

        The EOS stop is a lax.while_loop predicate rather than a host-side `bool(jnp.all(done))`
        check, so the whole generation is one dispatch and zero device->host syncs. That matters
        because serving regenerates whenever the discretized state in the pi0.5 prompt crosses a
        bin, i.e. continuously while the robot moves.

        Returns a FIXED-WIDTH (B, max_tokens) buffer, PAD(0) after each row's EOS, rather than a
        buffer trimmed to the last row's EOS. Trimming would need the length on the host, i.e. the
        one sync this exists to remove. It is safe because the padding is handled downstream:
        build_full_observation re-derives per-row validity from the tokens and keeps the filler out
        of tokenized_prompt_mask, and sample_actions derives RoPE positions from
        cumsum(tokenized_prompt_mask), so masked filler consumes no logical position.

        The prefill attn mask is pre-padded by ``max_tokens`` so gemma_05._init_cache reserves
        room, and each generated token is written at physical slot ``prefix_S + i`` (via
        _update_cache) while its RoPE position continues logically from the real prefix length.
        """
        logits, kv_cache, prefix_mask, next_pos = self._subtask_prefill(observation, max_tokens)
        b = prefix_mask.shape[0]
        step_idx = jnp.arange(max_tokens)

        def cond(carry):
            i, done = carry[0], carry[1]
            return jnp.logical_and(i < max_tokens, jnp.logical_not(jnp.all(done)))

        def body(carry):
            i, done, tokens, logits, kv_cache, next_pos = carry
            next_token = jnp.argmax(logits[:, -1, :], axis=-1).astype(jnp.int32)  # (B,)
            # Per-row EOS tracking: once a row emits EOS it emits PAD(0) thereafter, so in a B>1
            # batch a short row is not dragged past its EOS by longer rows.
            next_token = jnp.where(done, jnp.zeros_like(next_token), next_token)
            tokens = tokens.at[:, i].set(next_token)
            done = done | (next_token == PALIGEMMA_EOS_TOKEN)

            # Validity over the fixed cache: real prefix + generated slots 0..i
            # (physical prefix_S..prefix_S+i).
            gen_valid = jnp.broadcast_to(step_idx[None, :] <= i, (b, max_tokens))
            step_valid = jnp.concatenate([prefix_mask, gen_valid], axis=1)  # (b, prefix_s + max_tokens)
            attn_mask = step_valid[:, None, :]  # (b, 1, cache_size)
            logits, kv_cache = self._subtask_decode_step(next_token, attn_mask, next_pos[:, None], kv_cache)
            return i + 1, done, tokens, logits, kv_cache, next_pos + 1

        # The predicate is checked before the body, so when every row finishes at step i the body
        # has already run one more decode_step than an eager mid-body break would. The extra step's
        # logits are discarded, so the emitted tokens are identical; it costs one token's worth of
        # compute to avoid a host sync per token.
        init = (
            jnp.int32(0),
            jnp.zeros((b,), dtype=jnp.bool_),
            jnp.zeros((b, max_tokens), dtype=jnp.int32),
            logits,
            kv_cache,
            next_pos,
        )
        return jax.lax.while_loop(cond, body, init)[2]

    def generate_subtask(self, observation: _model.Observation, *, max_tokens: int = 48) -> at.Int[at.Array, "b t"]:
        """Stage 1 of pi0.5 subtask inference: greedily generate low-level subtask tokens.

        Thin wrapper: preprocessing stays outside the jit, generation is one jitted call — see
        _subtask_generate for why the loop lives on device and why the returned (B, max_tokens)
        buffer is padded rather than trimmed.
        """
        observation = jax.tree.map(jnp.asarray, observation)
        observation = _model.preprocess_observation(None, observation, train=False)
        generate = _subtask_jit_fns(self)[2]
        return generate(observation, max_tokens)

    def build_full_observation(
        self, observation: _model.Observation, subtask_tokens: at.Int[at.Array, "b t"]
    ) -> _model.Observation:
        """Stage 2 prep: splice the generated subtask into the padded high-level prompt so the
        action expert conditions on the full sequence.

        Inserts ``[subtask ... EOS]`` starting at each example's prompt length, via jnp.where over
        the padding region. token_loss_mask is dropped and token_ar_mask is rebuilt so the subtask
        span stays causal, as in training.

        Rows whose subtask ended early are PAD(0)-filled by generate_subtask; those slots are
        excluded from tokenized_prompt_mask rather than presented as prompt content.
        """
        observation = jax.tree.map(jnp.asarray, observation)
        insert = subtask_tokens
        b, max_len = observation.tokenized_prompt.shape
        gen_len = insert.shape[1]
        if gen_len == 0:
            return observation
        prefix_len = jnp.sum(observation.tokenized_prompt_mask, axis=1)
        # Hardening: the splice below silently drops any token whose position reaches max_len.
        # Warn (eager path, B is small) so a truncated subtask is not lost silently.
        #
        # gen_len is the FIXED generation width (generate_subtask does not trim to the last EOS),
        # so this is a physical-capacity check. Post-EOS filler is masked out of attention and
        # consumes no RoPE position, but it does occupy prompt slots. Budget: max_token_len is 200
        # for pi0.5 against a prefix of roughly 60-130 plus 48 generated, so the fixed width keeps
        # a healthy margin. Raising max_tokens or lengthening the State span eats into it.
        if bool(jnp.any(prefix_len + gen_len > max_len)):
            logger.warning(
                "build_full_observation: prompt(%s) + generated(%d) exceeds max_token_len(%d); "
                "trailing subtask tokens are silently dropped.",
                prefix_len.tolist(),
                int(gen_len),
                int(max_len),
            )

        # Per-row validity of the spliced span. generate_subtask emits PAD(0) after each row's EOS,
        # so in a B>1 batch a short row carries filler that must NOT become attendable prompt
        # content (token id 0 read as if it were a word). A slot is real up to and including the
        # first EOS.
        is_eos = subtask_tokens == PALIGEMMA_EOS_TOKEN
        insert_valid = (jnp.cumsum(is_eos, axis=1) - is_eos) == 0

        idx = jnp.arange(max_len)[None, :]
        offset = idx - prefix_len[:, None]
        in_gen = (offset >= 0) & (offset < gen_len)
        offset_clamped = jnp.clip(offset, 0, gen_len - 1).astype(jnp.int32)
        rows = jnp.arange(b)[:, None]
        gen_vals = insert[rows, offset_clamped]
        in_gen = in_gen & insert_valid[rows, offset_clamped]
        new_tokens = jnp.where(in_gen, gen_vals, observation.tokenized_prompt)
        new_mask = observation.tokenized_prompt_mask | in_gen
        # Causal ar_mask over the spliced subtask+EOS span (=1), bidirectional (=0) elsewhere
        # (high prompt / State / pad) — mirrors tokenize_high_low_prompt so the action stage
        # attends over the subtask exactly as training did.
        new_ar_mask = in_gen.astype(jnp.int32)
        return dataclasses.replace(
            observation,
            tokenized_prompt=new_tokens,
            tokenized_prompt_mask=new_mask,
            token_ar_mask=new_ar_mask,
            token_loss_mask=None,
        )

    def _compute_subtask_ce_loss(
        self,
        prefix_out: at.Float[at.Array, "b s d"],
        observation: _model.Observation,
        num_image_tokens: int,
    ) -> at.Float[at.Array, " b"]:
        """Cross-entropy over the subtask span (next-token prediction): hidden[i] predicts
        token[i+1]. Masked to the subtask+EOS region via observation.token_loss_mask."""
        num_text = self.max_token_len
        text_hidden = prefix_out[:, num_image_tokens : num_image_tokens + num_text - 1, :]
        logits = self.PaliGemma.llm(text_hidden, method="deembed")
        targets = observation.tokenized_prompt[:, 1:]
        loss_mask = observation.token_loss_mask[:, 1:].astype(jnp.float32)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        token_nll = -jnp.take_along_axis(log_probs, targets[:, :, None], axis=-1).squeeze(-1)
        masked_nll = token_nll * loss_mask
        return jnp.sum(masked_nll, axis=-1) / jnp.clip(jnp.sum(loss_mask, axis=-1), 1)

    def _compute_subtask_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool
    ) -> at.Float[at.Array, "*b ah"]:
        """pi0.5 subtask training: ONE forward over [prefix | suffix], from which both the subtask
        CE and the action flow-matching loss are read. The prefix attention is bidirectional except
        the subtask span, which is causal — taken from observation.token_ar_mask. `loss_mode`
        dispatches which terms contribute."""
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_tokens, prefix_mask, _ = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask_1d, adarms_cond = self.embed_suffix(observation, x_t, time)
        b = prefix_tokens.shape[0]
        num_text = self.max_token_len
        num_image_tokens = prefix_tokens.shape[1] - num_text
        # Prefix ar_mask (batched): images bidirectional, text from token_ar_mask (subtask causal).
        img_ar = jnp.zeros((b, num_image_tokens), dtype=jnp.bool_)
        if observation.token_ar_mask is not None:
            text_ar = observation.token_ar_mask.astype(jnp.bool_)
        else:
            text_ar = jnp.zeros((b, num_text), dtype=jnp.bool_)
        prefix_ar_mask = jnp.concatenate([img_ar, text_ar], axis=1)
        suffix_ar_mask = jnp.broadcast_to(suffix_ar_mask_1d[None, :], (b, suffix_tokens.shape[1]))

        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=1)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )

        ce = (
            self._compute_subtask_ce_loss(prefix_out, observation, num_image_tokens)
            if observation.token_loss_mask is not None
            else jnp.zeros(b)
        )
        if self.loss_mode == "ce_only":
            return jnp.broadcast_to(ce[:, None], (b, self.action_horizon))

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])
        flow_loss = jnp.mean(jnp.square(v_t - u_t), axis=-1)
        if self.loss_mode == "fm_only":
            return flow_loss
        return ce[:, None] + flow_loss

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        if self.subtask:
            return self._compute_subtask_loss(rng, observation, actions, train=train)

        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        # For pi0.5 subtask models the spliced subtask span must stay CAUSAL here (matching
        # compute_loss), so honor observation.token_ar_mask when present instead of embed_prefix's
        # all-bidirectional text mask. Otherwise the action expert would read
        # bidirectionally-attended subtask hiddens at inference but causal ones in training.
        if observation.token_ar_mask is not None:
            num_image_tokens = prefix_tokens.shape[1] - self.max_token_len
            img_ar = jnp.zeros((prefix_tokens.shape[0], num_image_tokens), dtype=jnp.bool_)
            prefix_ar_mask = jnp.concatenate([img_ar, observation.token_ar_mask.astype(jnp.bool_)], axis=1)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
