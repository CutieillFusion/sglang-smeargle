# SPDX-License-Identifier: Apache-2.0
"""SMEARGLE draft model for speculative decoding with a Llama target.

Architecture mirrors EAGLE-3 (single-layer draft, embeds + target hidden_state
concat at the input, separate small lm_head with d2t vocab translation), but the
decoder block is a **Mamba2 SSM** instead of self-attention. This is a port of
vLLM's ``vllm/model_executor/models/llama_smeargle.py`` into SGLang idiom; the
reference checkpoint layout is identical, so weights load unchanged.

Source layout the converter produces:
  config.json    architectures=["Mamba2LlamaSmeargleForCausalLM"], model_type="llama"
                 with smeargle extras: state_size, conv_kernel, n_groups, expand,
                 chunk_size, draft_vocab_size, target_hidden_size
  model.safetensors   embed_tokens, midlayer.{hidden_norm,input_layernorm,post_attention_layernorm}
                      midlayer.mamba2.{mamba2.*, out_proj.weight}
                      midlayer.mlp.{gate_proj,up_proj,down_proj}.weight
                      fc.weight, norm.weight, lm_head.weight, d2t, t2d

Like vLLM, we re-use HuggingFace transformers' ``Mamba2Block`` at runtime: it
matches the reference exactly and lets the speculative worker manage the SSM
cache state itself (the spec worker threads ``cache_params`` / ``cache_position``
through ``forward`` rather than going through ``ModelRunner.forward``).
"""

from __future__ import annotations

import logging
import os
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from transformers import LlamaConfig
from transformers.activations import ACT2FN

from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix

logger = logging.getLogger(__name__)


def _build_hf_mamba2_block(
    config: LlamaConfig,
    target_hidden: int,
    num_attention_heads: int,
    rms_norm_eps: float,
):
    """Construct an HF ``Mamba2Block`` with the same hyperparameters the SMEARGLE
    reference uses. Returns ``(mamba2_block, hf_mamba2_config)``.
    """
    from transformers.integrations import hub_kernels
    from transformers.models.mamba2.configuration_mamba2 import Mamba2Config
    from transformers.models.mamba2.modeling_mamba2 import Mamba2Block

    hf_cfg = Mamba2Config(
        hidden_size=target_hidden * 2,  # concat(embed, hidden) input
        num_heads=num_attention_heads * 2,
        head_dim=target_hidden // num_attention_heads,
        state_size=getattr(config, "state_size", 16),
        expand=getattr(config, "expand", 1),
        conv_kernel=getattr(config, "conv_kernel", 4),
        n_groups=getattr(config, "n_groups", 1),
        chunk_size=getattr(config, "chunk_size", 256),
        num_hidden_layers=1,
        residual_in_fp32=True,
        layer_norm_epsilon=rms_norm_eps,
    )
    # Use the real `causal_conv1d` + `selective_state_update` Mamba2 kernels.
    # They are REQUIRED for usable memory: HF's pure-PyTorch fallback
    # (`torch_forward`) pads every step to `chunk_size` and materializes a
    # chunk x chunk matrix per head per batch row (~2.5 GB/request), which OOMs
    # at moderate batch — fatal on unified-memory hosts (e.g. DGX Spark). The
    # kernel path is ~constant memory (~0.5 GB regardless of batch).
    #
    # HF fetches the kernels from the Hub at construction; this needs a writable
    # HF cache (`HF_HOME`). If that fails (e.g. a root-owned ~/.cache), fall back
    # to the naive path with a loud warning rather than crashing.
    saved = hub_kernels._kernels_available
    try:
        block = Mamba2Block(hf_cfg, layer_idx=0)
    except (PermissionError, OSError) as e:
        logger.warning(
            "[smeargle] Mamba2 Hub kernels unavailable (%s); falling back to the "
            "naive torch path. This is MEMORY-EXPENSIVE (~2.5 GB/request) and will "
            "not scale past small batch. Set HF_HOME to a writable directory to "
            "enable the fast kernels.",
            e,
        )
        hub_kernels._kernels_available = False
        block = Mamba2Block(hf_cfg, layer_idx=0)
    finally:
        hub_kernels._kernels_available = saved
    return block, hf_cfg


class _LlamaMLPWithBias(nn.Module):
    """Llama-style gated MLP that optionally supports bias on the projections
    (the SMEARGLE reference allows gate/up/down biases; the shipped checkpoint
    has ``mlp_bias=false`` so bias is disabled)."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        h = config.hidden_size
        i = config.intermediate_size
        bias = getattr(config, "mlp_bias", False)
        self.gate_proj = nn.Linear(h, i, bias=bias)
        self.up_proj = nn.Linear(h, i, bias=bias)
        self.down_proj = nn.Linear(i, h, bias=bias)
        self.act_fn = ACT2FN[getattr(config, "hidden_act", "silu")]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class SmeargleDecoderLayer(nn.Module):
    """Single SMEARGLE draft layer: norm(embed) + norm(hidden) -> cat -> Mamba2 ->
    +residual -> norm + MLP -> +residual."""

    def __init__(self, config: LlamaConfig):
        super().__init__()
        h = config.hidden_size
        self.hidden_size = h

        self.mamba2_block, self._hf_mamba2_config = _build_hf_mamba2_block(
            config,
            target_hidden=h,
            num_attention_heads=config.num_attention_heads,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.out_proj = nn.Linear(self._hf_mamba2_config.hidden_size, h, bias=False)
        self.mlp = _LlamaMLPWithBias(config)
        self.input_layernorm = RMSNorm(h, eps=config.rms_norm_eps)
        self.hidden_norm = RMSNorm(h, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(h, eps=config.rms_norm_eps)

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_params=None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # The spec worker packs all tokens (across the batch / chain branches)
        # into a 2D tensor (num_tokens, hidden). HF's Mamba2Block expects
        # (batch, seq_len, hidden) where the cache batch_size matches the input.
        # For spec decode every token row is its OWN chain with its own cache
        # slot, so we want batch=num_tokens, seq_len=1.
        residual = hidden_states  # (N, H)

        h_n = self.hidden_norm(hidden_states)
        e_n = self.input_layernorm(input_emb)
        x = torch.cat((e_n, h_n), dim=-1)  # (N, 2H)

        # (N, 2H) -> (N, 1, 2H): each row is a separate cache slot.
        x = x.unsqueeze(1)

        # cuDNN produces a larger Mamba2 SSM cache magnitude than the ATen
        # fallback on Blackwell, which manifests as an accept-length collapse.
        # Force the cuDNN-disabled path; env override SMEARGLE_USE_CUDNN=1
        # restores the default kernel selection.
        force_disable_cudnn = int(os.environ.get("SMEARGLE_USE_CUDNN", "0")) != 1
        if force_disable_cudnn:
            with torch.backends.cudnn.flags(enabled=False):
                x = self.mamba2_block(
                    hidden_states=x,
                    cache_params=cache_params,
                    cache_position=cache_position,
                    attention_mask=None,
                )
        else:
            x = self.mamba2_block(
                hidden_states=x,
                cache_params=cache_params,
                cache_position=cache_position,
                attention_mask=None,
            )
        x = x.squeeze(1)  # (N, 2H)
        x = x.to(self.out_proj.weight.dtype)
        x = self.out_proj(x)  # (N, H)

        hidden_states = residual + x

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class SmeargleModel(nn.Module):
    """The draft body: embed_tokens, optional fc (aux hidden-state fusion),
    midlayer (Mamba2 + MLP), final norm."""

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        eagle_config = getattr(config, "eagle_config", None) or {}
        self.use_aux_hidden_state = bool(
            eagle_config.get("use_aux_hidden_state", True)
        )
        self.norm_before_fc = getattr(config, "norm_before_fc", False)

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            prefix=add_prefix("embed_tokens", prefix),
        )

        self.midlayer = SmeargleDecoderLayer(config)

        if self.use_aux_hidden_state:
            target_hidden = getattr(
                config, "target_hidden_size", config.hidden_size
            )
            # num_aux resolution: explicit attr > eagle_config layer_ids > default 3.
            num_aux = getattr(config, "num_aux_hidden_states", None)
            if num_aux is None:
                layer_ids = eagle_config.get("eagle_aux_hidden_state_layer_ids")
                num_aux = len(layer_ids) if layer_ids else 3
            self.num_aux_hidden_states = num_aux
            fc_input_size = target_hidden * num_aux
            self.input_norm = (
                RMSNorm(fc_input_size, eps=config.rms_norm_eps)
                if self.norm_before_fc
                else None
            )
            self.fc = nn.Linear(fc_input_size, config.hidden_size, bias=False)

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compress the stacked aux hidden states (``num_aux * H``) into ``H`` via
        fc (EAGLE3-style). No-op when the target already hands us a single
        last-layer hidden state of width ``H``."""
        if not self.use_aux_hidden_state:
            return hidden_states
        if self.norm_before_fc and self.input_norm is not None:
            hidden_states = self.input_norm(hidden_states)
        return self.fc(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,  # API parity; Mamba2 doesn't use RoPE.
        hidden_states: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
        cache_params=None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)

        if hidden_states.shape[-1] != input_embeds.shape[-1]:
            # Aux hidden-state path: target gave us a stack of (N, num_aux*H).
            hidden_states = self.combine_hidden_states(hidden_states)

        out = self.midlayer(
            input_emb=input_embeds,
            hidden_states=hidden_states,
            cache_params=cache_params,
            cache_position=cache_position,
        )
        return self.norm(out)


class Mamba2LlamaSmeargleForCausalLM(nn.Module):
    """The registered architecture. SGLang routes to it via the ``architectures``
    field in config.json (``model_type`` stays ``"llama"`` for HF auto-config)."""

    def __init__(
        self,
        config: LlamaConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.quant_config = quant_config

        if getattr(config, "draft_vocab_size", None) is None:
            config.draft_vocab_size = config.vocab_size

        self.model = SmeargleModel(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        self.lm_head = ParallelLMHead(
            config.draft_vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
        )

        # Maps a draft-vocab id to its target-vocab id: hot_token_id[draft] = target.
        # Built from the checkpoint's d2t (stored diffs) in load_weights, mirroring
        # SGLang's EAGLE3 convention.
        self.hot_token_id: Optional[torch.Tensor] = None

        # Draft decode captures the pre-logits hidden state for the next step.
        self.capture_aux_hidden_states = True

    # --- API used by the speculative worker -------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.model.combine_hidden_states(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        input_embeds: Optional[torch.Tensor] = None,
        cache_params=None,
        cache_position: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.model(
            input_ids,
            positions,
            hidden_states,
            input_embeds,
            cache_params=cache_params,
            cache_position=cache_position,
        )

    def _draft_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project the draft hidden state onto the small draft vocab. Assumes
        TP=1 for the draft head (the draft is tiny; vocab-parallel gather for
        TP>1 is a future extension)."""
        return torch.matmul(hidden_states, self.lm_head.weight.t())

    def _draft_to_target(self, draft_tokens: torch.Tensor) -> torch.Tensor:
        if self.hot_token_id is None:
            return draft_tokens
        # Move the d2t map onto the compute device once and cache it back; a
        # per-call `.to(device)` is a host->device copy that breaks CUDA-graph
        # capture of the draft chain.
        if self.hot_token_id.device != draft_tokens.device:
            self.hot_token_id = self.hot_token_id.to(draft_tokens.device)
        return self.hot_token_id[draft_tokens]

    def compute_target_argmax(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Greedy: draft lm_head -> argmax in draft vocab -> translate to target
        vocab via the d2t mapping. Returns target-vocab ids, shape ``(N,)``."""
        draft_argmax = self._draft_logits(hidden_states).argmax(dim=-1)
        return self._draft_to_target(draft_argmax)

    def compute_target_topk(
        self, hidden_states: torch.Tensor, k: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Top-k: returns ``(top_logprobs, top_target_ids)``, both ``(N, k)``."""
        draft_logits = self._draft_logits(hidden_states)
        log_probs = torch.log_softmax(draft_logits.float(), dim=-1)
        values, indices = torch.topk(log_probs, k, dim=-1)
        return values, self._draft_to_target(indices)

    def get_hot_token_id(self):
        return self.hot_token_id

    # --- Weight loading ----------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        loaded = set()
        for name, w in weights:
            if "t2d" in name:
                # The chain decoder only needs draft->target (d2t); skip t2d.
                continue
            if "d2t" in name:
                # d2t stores per-draft-id diffs; hot_token_id[draft] = target.
                self.hot_token_id = w + torch.arange(w.shape[0])
                continue

            # Remap the reference's inner-block names to our module layout.
            if name.startswith("midlayer.mamba2.mamba2."):
                name = name.replace(
                    "midlayer.mamba2.mamba2.", "midlayer.mamba2_block."
                )
            elif name.startswith("midlayer.mamba2.out_proj."):
                name = name.replace("midlayer.mamba2.out_proj.", "midlayer.out_proj.")

            # Everything except lm_head lives under the SmeargleModel (`model.`).
            if not name.startswith("lm_head"):
                name = "model." + name

            if name not in params_dict:
                logger.warning("[smeargle] skipping unknown weight key: %s", name)
                continue
            param = params_dict[name]
            loader = getattr(param, "weight_loader", default_weight_loader)
            loader(param, w)
            loaded.add(name)

        missing = set(params_dict) - loaded
        if missing:
            logger.warning("[smeargle] params not loaded from checkpoint: %s", missing)


EntryClass = [Mamba2LlamaSmeargleForCausalLM]
