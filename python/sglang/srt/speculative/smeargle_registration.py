"""Registration of the SMEARGLE speculative-decoding algorithm.

SMEARGLE is an EAGLE-3-style draft whose single decoder layer is a Mamba2 SSM
block (no attention, no KV cache). It rides the EAGLE-3 aux-hidden-state capture
on the target model (so ``is_eagle3()`` is True), but needs its own worker that
manages the Mamba2 SSM cache and drafts a linear chain of tokens — so it is
registered as a plugin (``CustomSpecAlgo``) with a factory that returns
``SmeargleWorkerV2`` rather than being routed to ``EAGLEWorkerV2``.

This module is imported at the bottom of ``spec_info.py`` so the registration
runs as an import side effect in every process that touches
``SpeculativeAlgorithm`` (main, scheduler, and draft/target worker subprocesses),
before any ``from_string("SMEARGLE")`` call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_registry import CustomSpecAlgo

if TYPE_CHECKING:
    from sglang.srt.managers.overlap_utils import FutureMap
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class SmeargleSpecAlgo(CustomSpecAlgo):
    """Duck-types the ``SpeculativeAlgorithm`` enum for SMEARGLE.

    Returns ``is_eagle()``/``is_eagle3()`` True so the *target* reuses the
    EAGLE-3 aux-hidden-state capture plumbing. The draft side (attention
    backend, draft KV pool, CUDA graphs) is neutralized inside the Smeargle
    worker, not here.
    """

    def is_eagle(self) -> bool:
        return True

    def is_eagle3(self) -> bool:
        return True

    def need_topk(self) -> bool:
        # Chain mode runs the EAGLE topk==1 verify path.
        return True

    def has_draft_kv(self) -> bool:
        # The Mamba2 draft writes no KV chains; the scheduler must reserve no
        # per-decode draft KV pages.
        return False

    def carries_draft_hidden_states(self) -> bool:
        # No disaggregated prefill->decode hidden-state transfer in chain v1.
        return False

    def create_future_map(
        self,
        device,
        req_to_token_pool,
        needs_cpu_seq_lens: bool = True,
    ) -> "FutureMap":
        from sglang.srt.managers.overlap_utils import FutureMap

        return FutureMap(device, self, req_to_token_pool, needs_cpu_seq_lens)


def _validate_server_args(server_args: "ServerArgs") -> None:
    """Force the chain-mode invariants. Mirrors the EAGLE-family handling in
    ``speculative_hook.py``, which only runs for the literal EAGLE/EAGLE3/
    STANDALONE algorithms."""
    if server_args.speculative_draft_model_path is None:
        raise ValueError(
            "SMEARGLE speculative decoding requires --speculative-draft-model-path."
        )

    # topk == 1 -> chain; topk > 1 -> beam-search draft tree. Default to the
    # EAGLE3 Llama preset (num_steps=5, topk=4, num_draft_tokens=8) when unset.
    if server_args.speculative_eagle_topk is None:
        server_args.speculative_eagle_topk = 4
    if server_args.speculative_num_steps is None:
        server_args.speculative_num_steps = 5
    topk = server_args.speculative_eagle_topk
    num_steps = server_args.speculative_num_steps

    if topk == 1:
        # Chain: the worker's topk==1 buffers require this invariant.
        server_args.speculative_num_draft_tokens = num_steps + 1
    else:
        # Tree: prune the beam (topk*num_steps candidate nodes) to num_draft_tokens.
        max_nodes = topk * num_steps + 1
        if server_args.speculative_num_draft_tokens is None:
            server_args.speculative_num_draft_tokens = min(8, max_nodes)
        if not (2 <= server_args.speculative_num_draft_tokens <= max_nodes):
            raise ValueError(
                f"SMEARGLE tree mode requires 2 <= speculative_num_draft_tokens <= "
                f"topk*num_steps+1 ({max_nodes}); got "
                f"{server_args.speculative_num_draft_tokens} "
                f"(topk={topk}, num_steps={num_steps})."
            )

    if server_args.max_running_requests is None:
        server_args.max_running_requests = 48
        logger.warning(
            "Max running requests is reset to 48 for speculative decoding. "
            "You can override this by explicitly setting --max-running-requests."
        )

    if server_args.enable_mixed_chunk:
        server_args.enable_mixed_chunk = False
        logger.warning(
            "Mixed chunked prefill is disabled because of using SMEARGLE "
            "speculative decoding."
        )


@SpeculativeAlgorithm.register(
    "SMEARGLE",
    supports_overlap=False,
    validate_server_args=_validate_server_args,
    spec_class=SmeargleSpecAlgo,
)
def _smeargle_worker_factory(server_args: "ServerArgs"):
    from sglang.srt.speculative.smeargle_worker_v2 import SmeargleWorkerV2

    return SmeargleWorkerV2
