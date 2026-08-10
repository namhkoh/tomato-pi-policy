from collections.abc import Sequence
import dataclasses
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class SubtaskCache:
    """Stage-1 (generate_subtask) result cache, keyed on the tokenized prompt bytes.

    Owned by the caller — the websocket server gives each connection its own — so two clients
    whose prompts and 256-bin-quantized states happen to collide are not served each other's
    subtask, generated from the other one's images.
    """

    key: bytes | None = None
    tokens: Any = None


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

        # pi0.5 two-stage subtask inference: stage 1 generates the subtask, stage 2 is the standard
        # sample_actions over the spliced prompt. Generation is cached per high-level prompt; note
        # that the pi0.5 prompt embeds the discretized state, so the cache misses whenever the
        # state crosses a bin — regeneration is the steady-state cost while the robot moves, not a
        # one-off.
        self._is_subtask_model = getattr(model, "subtask", False) and hasattr(model, "generate_subtask")
        # Used only when a caller passes no subtask_cache (offline scripts, tests, PolicyRecorder).
        # Serving passes the per-connection cache it owns.
        self._fallback_subtask_cache = SubtaskCache()
        # Lazily built PaligemmaTokenizer, used only to render the generated subtask as text.
        self._subtask_detokenizer: Any = None

    def new_subtask_cache(self) -> SubtaskCache:
        """A fresh stage-1 cache. The websocket server calls this once per connection; its
        presence is also how the server detects that this policy understands subtask kwargs."""
        return SubtaskCache()

    @override
    def infer(  # type: ignore[misc]
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        subtask_cache: SubtaskCache | None = None,
        force_subtask_regen: bool = False,
        subtask_only: bool = False,
    ) -> dict:
        # subtask_cache: per-connection stage-1 cache; falls back to an instance-level one when a
        # caller doesn't supply it. force_subtask_regen ignores a cache hit and regenerates.
        # subtask_only runs stage 1 only and returns the caption without sampling actions.
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()

        # pi0.5 stage 1: autoregressively generate the subtask, then splice it into the prompt so
        # stage 2 (sample_actions) conditions on it. The inference tokenizer
        # (TokenizeHighPrompt) emits no token_ar_mask, which is the trigger for this path.
        generated_subtask_tokens = None
        subtask_ms = None  # set when stage-1 generation actually ran this request
        subtask_cached = None  # set (True/False) whenever the stage-1 path ran at all
        if self._is_subtask_model and observation.token_ar_mask is None:
            cache = subtask_cache if subtask_cache is not None else self._fallback_subtask_cache
            # tokenized_prompt is a torch tensor for PyTorch models (possibly on GPU); move to
            # host before hashing — np.asarray on a CUDA tensor would raise.
            _tp = observation.tokenized_prompt
            cache_key = _tp.detach().cpu().numpy().tobytes() if self._is_pytorch_model else np.asarray(_tp).tobytes()
            # The key covers only the prompt (task text + 256-bin quantized state) — NOT the
            # images. A client that holds the arm still while the scene changes therefore gets a
            # stale caption unless it asks for a regen.
            if force_subtask_regen or cache_key != cache.key or cache.tokens is None:
                # Generate FIRST, record the key only on success: recording it up front poisons
                # the cache with None when generation raises, and every later same-prompt request
                # then crashes downstream on the None.
                gen_start = time.monotonic()
                cache.tokens = self._model.generate_subtask(observation)
                subtask_ms = (time.monotonic() - gen_start) * 1000
                cache.key = cache_key
                subtask_cached = False
                logger.info(
                    "Subtask regenerated in %.0f ms (%s — the pi0.5 prompt embeds the discretized "
                    "state, so this recurs while moving)",
                    subtask_ms,
                    "forced" if force_subtask_regen else "prompt cache miss",
                )
            else:
                subtask_cached = True
            generated_subtask_tokens = cache.tokens

            if subtask_only:
                # Return before build_full_observation / sample_actions: everything past this
                # point exists only to condition stage 2. Also skips _output_transform on purpose,
                # since those transforms rebuild the dict from data["actions"].
                host_tokens = self._to_host_tokens(generated_subtask_tokens)
                subtask_text = self._detokenize_subtask(host_tokens)
                if not subtask_text:
                    # In the standard path the caption is a debug field, so an empty one is
                    # harmless. Here it IS the product — a labeling run would otherwise write
                    # hundreds of blank captions without anything noticing.
                    logger.warning(
                        "subtask_only produced an empty caption (%d generated tokens)",
                        np.size(host_tokens),
                    )
                return {
                    "subtask": subtask_text,
                    "policy_timing": {
                        "infer_ms": (time.monotonic() - start_time) * 1000,
                        **({"subtask_ms": subtask_ms} if subtask_ms is not None else {}),
                        "subtask_cached": subtask_cached,
                        "subtask_only": True,
                    },
                }
            observation = self._model.build_full_observation(observation, generated_subtask_tokens)

        if subtask_only:
            # Reaching here means the stage-1 block above didn't run, so there is no caption to
            # return. Fail loudly rather than silently answering with actions the caller discards.
            raise ValueError(
                "subtask_only requires a pi0.5 subtask model served with the inference tokenizer "
                "(no token_ar_mask); this policy produced no subtask tokens."
            )

        outputs = {
            "state": inputs["state"],
            "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
        }
        model_time = time.monotonic() - start_time
        if generated_subtask_tokens is not None:
            outputs["output_tokens"] = generated_subtask_tokens
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # Render the generated subtask as text so the client can see/debug it. Detokenizing a few
        # dozen ids is microseconds. Computed before the output transforms because those rebuild
        # the dict from the actions and would drop it.
        subtask_text = None
        if "output_tokens" in outputs:
            subtask_text = self._detokenize_subtask(outputs.pop("output_tokens"))

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
            **({"subtask_ms": subtask_ms} if subtask_ms is not None else {}),
            **({"subtask_cached": subtask_cached} if subtask_cached is not None else {}),
        }
        if subtask_text is not None:
            outputs["subtask"] = subtask_text
        return outputs

    def _to_host_tokens(self, tokens: Any) -> np.ndarray:
        """Generated subtask tokens -> host numpy, batch axis stripped."""
        if self._is_pytorch_model and isinstance(tokens, torch.Tensor):
            tokens = tokens.detach().cpu()
        return np.asarray(tokens)[0, ...]

    def _detokenize_subtask(self, tokens: np.ndarray) -> str:
        """Decode generated subtask token ids to text (pad/EOS filtered)."""
        if self._subtask_detokenizer is None:
            # Import here: only subtask serving needs it, and PaligemmaTokenizer's init loads the
            # (cached) sentencepiece model.
            from openpi.models import tokenizer as _tokenizer

            self._subtask_detokenizer = _tokenizer.PaligemmaTokenizer()
        try:
            return self._subtask_detokenizer.detokenize(np.asarray(tokens).flatten())
        except Exception:
            logger.exception("Failed to detokenize the generated subtask (debug field only)")
            return ""

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

        # Delegate, so wrapping a subtask policy in a recorder does not quietly turn the subtask
        # control fields into stripped-and-ignored keys on the server. Bound as an instance
        # attribute rather than a method so the server's hasattr() check still reports False when
        # the wrapped policy has no subtask stage.
        if hasattr(policy, "new_subtask_cache"):
            self.new_subtask_cache = policy.new_subtask_cache

    @override
    def infer(self, obs: dict, **kwargs) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs, **kwargs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
