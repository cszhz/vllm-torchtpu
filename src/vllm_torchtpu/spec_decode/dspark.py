# SPDX-License-Identifier: Apache-2.0
"""DSpark speculative decoding for TPU.

DSpark is semi-autoregressive parallel drafting: one DFlash-style parallel
block forward (anchor + noise tokens, non-causal within the block, context
KV precomputed from the target's aux hidden states), followed by a
lightweight sequential Markov head that biases each position's draft logits
on the previously sampled token. Upstream implements DSparkSpeculator as a
small DFlashSpeculator subclass; this proposer mirrors that on top of the
TPU DFlashProposer.

Dense-checkpoint layout (``sample_from_anchor`` absent or true, as in
upstream DSparkSpeculator): the query block is K tokens — the anchor is the
FIRST prediction position and every slot predicts the NEXT token — versus
DFlash's 1 + K (anchor is bonus only). Speculators-format checkpoints set
``sample_from_anchor=False`` and keep the DFlash layout.

PoC scope: greedy draft sampling only (argmax + d2t remap); probabilistic
Gumbel-coupled sampling is not implemented.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size

from vllm_torchtpu.distributed.tp_collectives import tp_token_shard_ok
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.spec_decode.dflash import DFlashProposer

logger = init_logger(__name__)


class DSparkProposer(DFlashProposer):
    def __init__(self, runner, vllm_config: VllmConfig):
        hf_config = vllm_config.speculative_config.draft_model_config.hf_config
        hf_dict = hf_config.to_dict() if hasattr(hf_config, "to_dict") else {}

        super().__init__(runner, vllm_config)

        if not self.dflash_config.get("target_layer_ids"):
            raise ValueError(
                "DSpark requires target layer ids in the draft config (one of "
                "eagle_aux_hidden_state_layer_ids, dspark_target_layer_ids, "
                "target_layer_ids, or dflash_config.target_layer_ids)."
            )

        # Same key/default as upstream DSparkSpeculator: dense checkpoints
        # omit it (-> True); the speculators converter writes it explicitly.
        self.sample_from_anchor = bool(hf_dict.get("sample_from_anchor", True))
        block = hf_dict.get("dspark_block_size") or hf_dict.get("block_size")
        K = self.speculative_config.num_speculative_tokens
        if block is not None and int(block) < K:
            raise ValueError(
                f"num_speculative_tokens={K} exceeds the DSpark "
                f"checkpoint's block size {block}; the block/Markov-head "
                "machinery supports at most that block length."
            )
        # vLLM 0.29 drafting options this proposer does not implement yet.
        spec = self.speculative_config
        if (
            spec.draft_sample_method != "greedy"
            or spec.enable_adaptive_verification
            or spec.dspark_draft_topk is not None
        ):
            raise ValueError(
                "DSpark on TPU does not support draft_sample_method="
                f"{spec.draft_sample_method!r}, enable_adaptive_verification="
                f"{spec.enable_adaptive_verification}, dspark_draft_topk="
                f"{spec.dspark_draft_topk}; only greedy drafting of a fixed "
                "block is implemented."
            )
        # DeepSeek-V4.1 captures each target layer's entry stream, so its
        # target layer ids are used unshifted.
        if getattr(hf_config, "model_type", None) == "deepseek_v41":
            self._aux_layer_shift = 0
        logger.info("DSpark: sample_from_anchor=%s, K=%d.", self.sample_from_anchor, K)

    # A reduced-vocab DSpark checkpoint carries its own lm_head
    # ([draft_vocab, hidden] plus a d2t remap) and possibly its own
    # embed_tokens; force-sharing the target's would clobber them. Honor the
    # has_own_embed_tokens / has_own_lm_head flags the loader sets (upstream
    # load_dspark_model does the same via _should_share).
    _force_share_target_embeddings = False

    def _query_block_size(self, K: int) -> int:
        # Dense layout: K query tokens (anchor + K-1 noise), every slot
        # predicts the next token. Speculators-format checkpoints keep the
        # DFlash 1 + K layout.
        return K if self.sample_from_anchor else K + 1

    def _get_draft_token_paddings(self) -> list[int]:
        # Request-aligned buckets pad less than the generic token buckets.
        max_chunk_reqs = max(
            self.runner.num_reqs_max_model_len, self.runner.num_reqs_most_model_len or 0
        )
        if self.block_size in (3, 4, 5):
            b = self.block_size
            if max_chunk_reqs == 64:
                return [16, 8 * b, 16 * b, 32 * b, 48 * b, 64 * b]
            if max_chunk_reqs > 64:
                max_draft_tokens = max_chunk_reqs * b
                base = [
                    16,
                    8 * b,
                    16 * b,
                    32 * b,
                    48 * b,
                    64 * b,
                    80 * b,
                    128 * b,
                    256 * b,
                    512 * b,
                ]
                buckets = [p for p in base if p <= max_draft_tokens]
                if buckets and buckets[-1] < max_draft_tokens:
                    buckets.append((max_draft_tokens + 7) & ~7)
                if buckets:
                    return sorted(set(buckets))
        return super()._get_draft_token_paddings()

    def _draft_kv_num_blocks(self, use_max_model_len: bool) -> int:
        # The draft's KV group (e.g. DeepSeek-V4.1's sliding-window cache) can
        # be narrower than group 0, so size the dummies from its own builder.
        runner = self.runner
        get_names = getattr(self.draft_model, "get_draft_kv_cache_layer_names", None)
        if get_names is not None:
            first_layer = get_names()[0]
            for group_list in runner.attn_groups:
                for group in group_list:
                    if first_layer in group.layer_names:
                        ctx = SimpleNamespace(
                            use_max_model_len=use_max_model_len,
                            mamba_ckpt_window=runner._mamba_ckpt_window,
                        )
                        builder = group.metadata_builders[0]
                        return builder._block_table_geometry(ctx)[1]
        return super()._draft_kv_num_blocks(use_max_model_len)

    def load_model(self, target_model) -> None:
        super().load_model(target_model)
        for attr in (
            "compute_draft_logits",
            "markov_embed",
            "markov_bias",
            "map_draft_to_target",
            "has_own_lm_head",
            "has_own_embed_tokens",
        ):
            if not hasattr(self.draft_model, attr):
                raise RuntimeError(
                    f"DSpark draft model lacks {attr}(); expected a "
                    "Qwen3DSpark/Gemma4DSpark-family model."
                )
        if hasattr(self.draft_model, "init_tp_markov_meta"):
            self.draft_model.init_tp_markov_meta()

    def _dflash_forward_and_sample_body(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backbone forward + sequential greedy Markov sampling, in-graph.

        Dense layout: every query slot predicts the token at its position + 1,
        so all block_size slots are sampled (DFlash samples only the mask
        slots). Bonus-anchor layout: the anchor is a pure bonus token, so only
        the K mask slots (1..block_size-1) are sampled. Either way the Markov
        chain seeds from the anchor input id and runs a static K-iteration
        loop: embedding gather + rank-r bias + argmax.
        """
        sample_start = 0 if self.sample_from_anchor else 1
        num_pred = block_size - sample_start
        tp_size = get_tensor_model_parallel_world_size()
        num_tokens = positions.shape[0]
        padded_num_reqs = num_tokens // block_size
        can_skip_final_ag = (
            sample_start == 0
            and num_tokens == padded_num_reqs * block_size
            and tp_token_shard_ok(num_tokens, tp_size)
            and hasattr(self.draft_model, "can_run_fused_sharded_markov")
            and self.draft_model.can_run_fused_sharded_markov()
        )
        if can_skip_final_ag:
            sharded_hidden = self.draft_model.model(
                input_ids=input_ids,
                positions=positions,
                skip_final_all_gather=True,
            )
            prev = input_ids[: padded_num_reqs * block_size : block_size]
            draft_tokens_chunk = self.draft_model.sample_draft_tokens(
                sharded_hidden, prev, num_pred, pred_sharded=True
            )
            return draft_tokens_chunk, sharded_hidden

        hidden = self.draft_model.model(
            input_ids=input_ids,
            positions=positions,
        )

        padded_num_reqs = hidden.shape[0] // block_size
        valid_hidden = hidden[: padded_num_reqs * block_size]
        if sample_start > 0:
            pred_hidden = valid_hidden.view(
                padded_num_reqs, block_size, hidden.shape[-1]
            )[:, sample_start:].reshape(-1, hidden.shape[-1])
        else:
            pred_hidden = valid_hidden

        ids_3d = input_ids[: padded_num_reqs * block_size].view(
            padded_num_reqs, block_size
        )
        prev = ids_3d[:, 0]
        if hasattr(self.draft_model, "sample_draft_tokens"):
            draft_tokens_chunk = self.draft_model.sample_draft_tokens(
                pred_hidden, prev, num_pred
            )
            return draft_tokens_chunk, hidden

        # Hoist initial step-0 Markov embedding + projection so XLA overlaps it
        # with compute_draft_logits(pred_hidden).
        bias = self.draft_model.markov_bias(self.draft_model.markov_embed(prev))
        logits = self.draft_model.compute_draft_logits(pred_hidden)
        logits_3d = logits.view(padded_num_reqs, num_pred, logits.shape[-1])

        draft_tokens = []
        for idx in range(num_pred):
            if idx > 0:
                bias = self.draft_model.markov_bias(self.draft_model.markov_embed(prev))
            step = self.draft_model.map_draft_to_target(
                (logits_3d[:, idx] + bias).argmax(dim=-1)
            )
            draft_tokens.append(step)
            prev = step
        draft_tokens_chunk = torch.stack(draft_tokens, dim=1)

        return draft_tokens_chunk, hidden

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _dflash_forward_and_sample(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._dflash_forward_and_sample_body(input_ids, positions, block_size)
