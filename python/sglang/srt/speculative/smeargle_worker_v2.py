"""SMEARGLE speculative-decoding worker (chain mode).

SMEARGLE is an EAGLE-3-style draft, but its single decoder layer is a Mamba2 SSM
block instead of attention. It therefore reuses the EAGLE-3 *target* aux-hidden
capture and the EAGLE verify / rejection path, but replaces the attention-based
draft loop with a Mamba2 chain:

  * No draft attention backend, no draft KV pool, no draft CUDA graphs.
  * The draft model is called directly as an ``nn.Module`` (bypassing
    ``ModelRunner.forward``), threading a fresh HF Mamba2 ``DynamicCache`` through
    the ``num_steps`` chain.
  * Chain == degenerate topk==1 tree, so the existing ``build_tree_kernel_efficient``
    / ``verify`` path is reused verbatim.

This is the chain-only v1 (vLLM's ``SMEARGLE_TREE_VERIFY=0``). Cross-iteration
SSM-cache carry-over (vLLM's stable cache) is intentionally omitted: each
``draft()`` reseeds a fresh cache from the target's aux hidden state.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.adaptive_runtime_state import AdaptiveController
from sglang.srt.speculative.eagle_info import EagleDraftInput, EagleVerifyInput
from sglang.srt.speculative.eagle_utils import (
    TreeMaskMode,
    build_tree_kernel_efficient,
    organize_draft_results,
)
from sglang.srt.speculative.eagle_worker_v2 import (
    EagleDraftWorker,
    EAGLEWorkerV2,
    _get_plan_stream,
)
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.utils import empty_context
from sglang.srt.utils.common import fast_topk

logger = logging.getLogger(__name__)


def run_mamba_chain(
    model,
    fused_hidden: torch.Tensor,
    seed_tokens: torch.Tensor,
    num_steps: int,
    cache=None,
) -> torch.Tensor:
    """Greedily draft ``num_steps`` tokens as a linear chain.

    Args:
        model: the ``Mamba2LlamaSmeargleForCausalLM`` instance.
        fused_hidden: ``[bs, H]`` target hidden state, already fc-fused from the
            aux stack.
        seed_tokens: ``[bs]`` target-vocab token that starts the chain (the last
            verified / bonus token).
        num_steps: chain length.

    Returns:
        ``[bs, num_steps]`` target-vocab draft tokens.
    """
    if cache is None:
        from transformers.cache_utils import DynamicCache

        cache = DynamicCache(config=model.model.midlayer._hf_mamba2_config)

    token = seed_tokens
    hidden = fused_hidden
    drafts = []
    for _ in range(num_steps):
        emb = model.embed_input_ids(token)  # [bs, H]
        # hidden width == H so the model skips fc; the midlayer concats
        # norm(emb)+norm(hidden) and runs one Mamba2 step (cache advances).
        hidden = model.forward(
            input_ids=None,
            positions=None,
            hidden_states=hidden,
            input_embeds=emb,
            cache_params=cache,
        )
        token = model.compute_target_argmax(hidden)  # [bs] target-vocab ids
        drafts.append(token)
    return torch.stack(drafts, dim=1)  # [bs, num_steps]


# ---------------------------------------------------------------------------
# Tree drafting (beam search over the Mamba2 SSM state).
#
# Mirrors the SMEARGLE standalone `topK_genrate` and vLLM's `_propose_tree_dynamic`:
# per-depth beam expansion (width = topk, depth = num_steps), forking/reindexing the
# Mamba2 cache by the surviving parents. The per-step bookkeeping reproduces SGLang
# EAGLE's `_select_top_k_tokens_{first,later}` exactly, so the resulting
# (score_list, token_list, parents_list) feed `organize_draft_results` /
# `build_tree_kernel_efficient` unchanged.
# ---------------------------------------------------------------------------


def _expand_mamba_cache(cache, n: int):
    """Repeat each batch row n times (bs -> bs*n), so each of the n beams forked
    from a parent starts from the parent's SSM state. Mirrors the reference's
    ``expand_mamba_cache`` (the SGLang DynamicCache stores batch dim 0)."""
    layer = cache.layers[0]
    layer.conv_states = layer.conv_states.repeat_interleave(n, dim=0).contiguous()
    layer.recurrent_states = layer.recurrent_states.repeat_interleave(
        n, dim=0
    ).contiguous()
    return cache


def _reindex_mamba_cache(cache, index: torch.Tensor):
    """Permute the batch dim by ``index`` (one entry per surviving beam), so each
    survivor inherits its parent's SSM state. Mirrors ``reindex_mamba_cache``."""
    layer = cache.layers[0]
    layer.conv_states = layer.conv_states[index].contiguous()
    layer.recurrent_states = layer.recurrent_states[index].contiguous()
    return cache


def _beam_step(i, topk_p, topk_index, hidden, scores, topk):
    """One beam-expansion step. Reproduces ``_select_top_k_tokens_{first,later}``
    (so the emitted tree_info is byte-compatible with ``organize_draft_results``),
    and additionally returns ``cache_index`` — the parent row each surviving beam
    came from — to reindex the Mamba cache. ``cache_index`` is None at i==0 (the
    caller expand-forks the cache instead)."""
    if i == 0:
        bs = topk_p.shape[0]
        input_ids = topk_index.flatten()  # [bs*topk]
        hidden = hidden.repeat_interleave(topk, dim=0)  # [bs*topk, H]
        tree_info = (
            topk_p.unsqueeze(1),  # [bs, 1, topk]
            topk_index,  # [bs, topk]
            torch.arange(
                -1, topk, dtype=torch.long, device=input_ids.device
            ).expand(bs, -1),  # [bs, topk+1]
        )
        return input_ids, hidden, topk_p, tree_info, None

    topk_sq = topk * topk
    expand_scores = scores.unsqueeze(2) * topk_p.view(-1, topk, topk)  # [bs,topk,topk]
    topk_cs_p, topk_cs_index = fast_topk(
        expand_scores.flatten(start_dim=1), topk, dim=-1
    )  # [bs, topk]
    topk_index = topk_index.view(-1, topk_sq)  # [bs, topk*topk]
    input_ids = torch.gather(topk_index, 1, topk_cs_index).flatten()  # [bs*topk]

    flat_cs = topk_cs_index.flatten()
    batch_offsets = torch.arange(0, hidden.shape[0], step=topk, device=flat_cs.device)
    cache_index = flat_cs // topk + batch_offsets.repeat_interleave(topk)  # [bs*topk]
    hidden = hidden[cache_index]

    tree_info = (
        expand_scores,  # [bs, topk, topk]
        topk_index,  # [bs, topk*topk]
        topk_cs_index + (topk_sq * (i - 1) + topk),  # [bs, topk]
    )
    return input_ids, hidden, topk_cs_p, tree_info, cache_index


def run_mamba_tree(model, fused_hidden, seed_tokens, topk: int, num_steps: int):
    """Beam-search draft tree. Returns (score_list, token_list, parents_list) in the
    format ``organize_draft_results`` expects."""
    from transformers.cache_utils import DynamicCache

    cache = DynamicCache(config=model.model.midlayer._hf_mamba2_config)

    # Root forward (the bonus token): advances the cache to the root state and
    # predicts the root's top-k children.
    hidden = model.forward(
        input_ids=None,
        positions=None,
        hidden_states=fused_hidden,
        input_embeds=model.embed_input_ids(seed_tokens),
        cache_params=cache,
    )  # [bs, H]
    topk_p, topk_index = model.compute_target_topk_probs(hidden, topk)

    score_list, token_list, parents_list = [], [], []
    scores = None
    for i in range(num_steps):
        input_ids, hidden, scores, tree_info, cache_index = _beam_step(
            i, topk_p, topk_index, hidden, scores, topk
        )
        score_list.append(tree_info[0])
        token_list.append(tree_info[1])
        parents_list.append(tree_info[2])
        if i == num_steps - 1:
            break
        # Fork/reindex the SSM state to match the selected beams, then advance.
        if i == 0:
            cache = _expand_mamba_cache(cache, topk)  # bs -> bs*topk
        else:
            cache = _reindex_mamba_cache(cache, cache_index)  # survivors' parents
        hidden = model.forward(
            input_ids=None,
            positions=None,
            hidden_states=hidden,
            input_embeds=model.embed_input_ids(input_ids),
            cache_params=cache,
        )  # [bs*topk, H]
        topk_p, topk_index = model.compute_target_topk_probs(hidden, topk)

    return score_list, token_list, parents_list


class SmeargleDraftGraphRunner:
    """Captures the fixed-shape Mamba2 draft chain into per-batch-size CUDA
    graphs to remove the eager per-step kernel-launch overhead (the draft is
    ~24% of decode GPU time and was previously un-graphed).

    Capture works by reusing one persistent HF cache per bucket: a warmup run
    allocates the cache and flips it to the decode-kernel path (which updates
    conv/ssm state IN PLACE), then the captured graph zeroes that state and runs
    the ``num_steps`` decode steps -- so each replay reproduces a fresh chain
    (decode step 0 on zeroed state == prefill). Any capture failure falls back
    to the eager ``run_mamba_chain`` so the server never crashes.
    """

    # Only the small, launch-bound batch sizes benefit: the profile shows the
    # eager draft is GEMV-launch-bound at low batch but already GEMM-efficient by
    # bs~32 (graphs gave +26% at B=16 but nothing at B=32). Capping the buckets
    # also keeps large-batch decode on the proven eager path.
    BUCKETS = (1, 2, 4, 8, 16)

    def __init__(self, model, num_steps: int, device, max_bs: int):
        self.model = model
        self.num_steps = num_steps
        self.device = device
        self.dtype = next(model.parameters()).dtype
        self.hidden_size = model.config.hidden_size
        self.buckets = [b for b in self.BUCKETS if b <= max(max_bs, 1)]
        self.graphs = {}  # bs -> dict(graph, hidden, seed, out)
        self.failed = set()

    def _bucket(self, bs: int):
        for b in self.buckets:
            if b >= bs:
                return b
        return None

    def _capture(self, bs: int) -> None:
        from transformers.cache_utils import DynamicCache

        model = self.model
        static_hidden = torch.zeros(
            bs, self.hidden_size, device=self.device, dtype=self.dtype
        )
        static_seed = torch.zeros(bs, dtype=torch.int64, device=self.device)
        cache = DynamicCache(config=model.model.midlayer._hf_mamba2_config)

        # Warmup on a side stream: allocates the cache and sets has_previous_state
        # so every captured step uses the in-place decode kernels.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                run_mamba_chain(
                    model, static_hidden, static_seed, self.num_steps, cache=cache
                )
        torch.cuda.current_stream().wait_stream(s)

        conv = cache.layers[0].conv_states
        ssm = cache.layers[0].recurrent_states

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            # Fresh chain each replay: decode step 0 on zeroed state == prefill.
            conv.zero_()
            ssm.zero_()
            out = run_mamba_chain(
                model, static_hidden, static_seed, self.num_steps, cache=cache
            )
        self.graphs[bs] = {
            "graph": g,
            "hidden": static_hidden,
            "seed": static_seed,
            "out": out,
        }

    def run(self, fused, seed):
        """Return ``[real_bs, num_steps]`` draft tokens via a CUDA graph, or
        ``None`` to signal the caller to use the eager path."""
        real_bs = fused.shape[0]
        bs = self._bucket(real_bs)
        if bs is None or bs in self.failed:
            return None
        if bs not in self.graphs:
            try:
                self._capture(bs)
            except Exception as e:  # noqa: BLE001 - never crash serving on capture
                logger.warning(
                    "[smeargle] draft CUDA graph capture failed for bs=%d (%s); "
                    "using eager draft.",
                    bs,
                    e,
                )
                self.failed.add(bs)
                return None
        entry = self.graphs[bs]
        entry["hidden"][:real_bs].copy_(fused)
        entry["seed"][:real_bs].copy_(seed)
        entry["graph"].replay()
        return entry["out"][:real_bs].clone()


class SmeargleDraftWorker(EagleDraftWorker):
    """EAGLE draft worker specialized for the Mamba2 SMEARGLE draft."""

    # -- Backend neutralization (Mamba has no attention / KV / cuda graphs) --

    def init_attention_backend(self):
        # The Mamba2 draft uses no attention backend. Leave the runner's
        # attn_backend as initialized (None, since num_attention_layers == 0).
        self.draft_attn_backend = None
        self.draft_extend_attn_backend = None
        self.tree_mask_mode = TreeMaskMode.FULL_MASK

    def init_cuda_graphs(self):
        self.cuda_graph_runner = None
        self.cuda_graph_runner_for_draft_extend = None
        # CUDA-graph the Mamba draft chain (captured lazily per batch size).
        # OFF by default: lazy mid-serving capture races with the live allocator
        # under concurrent load and can hit cudaErrorIllegalAddress, while the
        # profiler shows the draft is GPU-compute-bound (graphs ~1.5%, not worth
        # the crash risk). Opt in with SGLANG_SMEARGLE_DRAFT_CUDA_GRAPH=1 (and
        # ideally pre-capture at init) if you want to experiment.
        self._draft_graph_runner = None
        if (
            not self.server_args.disable_cuda_graph
            and os.environ.get("SGLANG_SMEARGLE_DRAFT_CUDA_GRAPH", "0") == "1"
        ):
            self._draft_graph_runner = SmeargleDraftGraphRunner(
                self.draft_runner.model,
                self.speculative_num_steps,
                self.device,
                max_bs=self.server_args.max_running_requests or 64,
            )

    def init_lm_head(self):
        # SMEARGLE ships its own embed_tokens + lm_head; do not share the
        # target's. Grab the draft->target (d2t) mapping built during weight load.
        self.hot_token_id = getattr(self.draft_runner.model, "hot_token_id", None)

    # -- Hidden-state selection helpers --

    @staticmethod
    def _last_token_indices(batch: ScheduleBatch, device) -> torch.Tensor:
        """Row index of the last prompt token of each request in a packed
        extend batch."""
        extend_lens = batch.extend_lens
        if not torch.is_tensor(extend_lens):
            extend_lens = torch.tensor(extend_lens, dtype=torch.int64, device=device)
        else:
            extend_lens = extend_lens.to(torch.int64)
        return torch.cumsum(extend_lens, dim=0) - 1

    def _fill_topk_placeholders(self, draft_input, bs: int, device) -> None:
        """Populate ``topk_p`` / ``topk_index`` with shape-valid ``[bs, topk]``
        placeholders. The chain ``draft()`` recomputes everything from
        ``hidden_states`` + ``bonus_tokens`` and never reads these, but EAGLE's
        ``EagleDraftInput.merge_batch`` / ``filter_batch`` (exercised at bs > 1)
        require them to be real tensors."""
        draft_input.topk_p = torch.ones(
            (bs, self.topk), dtype=torch.float32, device=device
        )
        draft_input.topk_index = torch.zeros(
            (bs, self.topk), dtype=torch.int64, device=device
        )

    # -- Draft chain --

    def draft(self, batch: ScheduleBatch) -> EagleVerifyInput:
        draft_input: EagleDraftInput = batch.spec_info

        if batch.forward_mode.is_idle():
            return EagleVerifyInput.create_idle_input(
                self.topk,
                self.speculative_num_steps,
                self.speculative_num_draft_tokens,
            )

        model = self.draft_runner.model
        bs = batch.seq_lens.shape[0]

        # draft_input.hidden_states is the target aux stack [bs, num_aux*H]
        # (packaged by our _draft_extend_*). Fuse to [bs, H].
        fused = model.combine_hidden_states(draft_input.hidden_states)
        seed_tokens = draft_input.bonus_tokens.to(torch.int64).view(-1)

        if self.topk == 1:
            # Chain: a single greedy path of num_steps tokens.
            draft_tokens = None
            runner = getattr(self, "_draft_graph_runner", None)
            if runner is not None:
                draft_tokens = runner.run(fused, seed_tokens)
            if draft_tokens is None:
                draft_tokens = run_mamba_chain(
                    model, fused, seed_tokens, self.speculative_num_steps
                )
            # Degenerate tree: parent_list / top_scores_index are runtime-invariant.
            top_scores_index = self._topk1_score_indices_prealloc[:bs]
            parent_list = self._topk1_parents_prealloc[:bs]
        else:
            # Tree: beam search (width=topk, depth=num_steps) over the Mamba SSM
            # state, then prune to num_draft_tokens by cumulative score. Reuses
            # EAGLE's organize_draft_results verbatim.
            score_list, token_list, parents_list = run_mamba_tree(
                model, fused, seed_tokens, self.topk, self.speculative_num_steps
            )
            parent_list, top_scores_index, draft_tokens = organize_draft_results(
                score_list, token_list, parents_list, self.speculative_num_draft_tokens
            )

        tree_mask_buf, position_buf = (
            self.target_worker.model_runner.attn_backend.get_verify_buffers_to_fill_after_draft()
        )
        seq_lens_sum = batch.seq_lens_sum
        if seq_lens_sum is None:
            if tree_mask_buf is None:
                seq_lens_sum = (
                    bs * self.target_worker.model_runner.attn_backend.max_context_len
                )
            else:
                seq_lens_sum = 0

        (
            tree_mask,
            position,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            draft_tokens,
        ) = build_tree_kernel_efficient(
            draft_input.bonus_tokens,
            parent_list,
            top_scores_index,
            draft_tokens,
            batch.seq_lens,
            seq_lens_sum,
            self.topk,
            self.speculative_num_steps,
            self.speculative_num_draft_tokens,
            self.tree_mask_mode,
            tree_mask_buf,
            position_buf,
        )

        return EagleVerifyInput(
            draft_token=draft_tokens,
            custom_mask=tree_mask,
            positions=position,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            retrieve_cum_len=None,
            spec_steps=self.speculative_num_steps,
            topk=self.topk,
            draft_token_num=self.speculative_num_draft_tokens,
            capture_hidden_mode=None,
            seq_lens_sum=None,
            seq_lens_cpu=None,
        )

    # -- "Draft extend": no KV to fill; just package the seed for the next draft --

    def _draft_extend_for_prefill(
        self,
        batch: ScheduleBatch,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        mm_input_embeds: Optional[torch.Tensor] = None,
    ) -> EagleDraftInput:
        if batch.forward_mode.is_idle():
            hidden = target_hidden_states
        else:
            last_idx = self._last_token_indices(batch, target_hidden_states.device)
            hidden = target_hidden_states[last_idx]
        draft_input = EagleDraftInput(
            hidden_states=hidden,
            bonus_tokens=next_token_ids,
            num_tokens_per_req=1,
            num_tokens_for_logprob_per_req=1,
        )
        self._fill_topk_placeholders(draft_input, hidden.shape[0], hidden.device)
        return draft_input

    def _draft_extend_for_decode(
        self, batch: ScheduleBatch, batch_result: GenerationBatchResult
    ) -> None:
        # Pick the target hidden state at the last accepted position of each
        # request (the verify forward captured FULL hidden over the tree).
        select_index = (
            torch.arange(len(batch.seq_lens), device=self.device)
            * self.speculative_num_draft_tokens
            + batch_result.accept_lens
            - 1
        )
        hidden = batch_result.logits_output.hidden_states[select_index]

        # verify() already populated next_draft_input.bonus_tokens (the per-request
        # bonus) via fill_bonus_tokens; only the target hidden state is ours to set.
        next_draft_input: EagleDraftInput = batch_result.next_draft_input
        next_draft_input.hidden_states = hidden
        self._fill_topk_placeholders(next_draft_input, hidden.shape[0], hidden.device)


class SmeargleWorkerV2(EAGLEWorkerV2):
    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.topk = server_args.speculative_eagle_topk
        self.speculative_num_steps = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = server_args.speculative_num_draft_tokens
        self.tp_rank = tp_rank
        self.gpu_id = gpu_id
        self.device = server_args.device
        self._target_worker = target_worker
        self.page_size = server_args.page_size
        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )

        # Draft model context length matches the target.
        server_args.context_length = target_worker.model_runner.model_config.context_len

        self._draft_worker = SmeargleDraftWorker(
            server_args,
            gpu_id,
            tp_rank,
            dp_rank,
            moe_ep_rank,
            attn_cp_rank,
            moe_dp_rank,
            nccl_port,
            target_worker,
        )

        # Dummy tensors expected by the base class.
        self.num_new_pages_per_topk = torch.empty(
            (), dtype=torch.int64, device=self.device
        )
        self.extend_lens = torch.empty((), dtype=torch.int64, device=self.device)
        self.plan_stream, self.plan_stream_ctx = _get_plan_stream(self.device)

        # No adaptive speculative in chain v1.
        self.adaptive_controller: Optional[AdaptiveController] = None

    @property
    def spec_v2_attn_backends(self) -> tuple:
        # Mamba draft touches no attention backend; only the target's.
        return (self._target_worker.model_runner.attn_backend,)
