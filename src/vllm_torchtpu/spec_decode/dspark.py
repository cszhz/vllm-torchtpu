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

Draft sampling: greedy (argmax + d2t remap), or ``probabilistic`` as upstream
draws it: per request, Gumbel-max over the Markov-biased logits divided by the
request's temperature (argmax at temperature 0). The rejection sampler here
treats a draft as one-hot, so probabilistic drafts are only accepted with
``rejection_sample_method="synthetic"``, whose accept test does not read the
draft distribution.
"""

from __future__ import annotations

import torch
from vllm.config import VllmConfig

from vllm_torchtpu import envs
from vllm_torchtpu.logger import init_logger
from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
    set_vllm_model_wrapper_context,
)
from vllm_torchtpu.spec_decode.dflash import DFlashProposer

logger = init_logger(__name__)


class DSparkProposer(DFlashProposer):
    def __init__(self, runner, vllm_config: VllmConfig):
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        hf_config = speculative_config.draft_model_config.hf_config
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
        # Like upstream DSparkSpeculator, the query block follows K, not the
        # block length the checkpoint was trained at; acceptance past that
        # length is the checkpoint's business.
        trained_block = hf_dict.get("dspark_block_size") or hf_dict.get("block_size")
        K = self.speculative_config.num_speculative_tokens
        # vLLM 0.29 drafting options this proposer does not implement yet.
        spec = self.speculative_config
        if spec.enable_adaptive_verification or spec.dspark_draft_topk is not None:
            raise ValueError(
                "DSpark on TPU does not support enable_adaptive_verification="
                f"{spec.enable_adaptive_verification}, dspark_draft_topk="
                f"{spec.dspark_draft_topk}; only a fixed block is implemented."
            )
        self._probabilistic = spec.draft_sample_method == "probabilistic"
        if self._probabilistic and spec.rejection_sample_method != "synthetic":
            raise ValueError(
                "DSpark on TPU does not support draft_sample_method="
                "'probabilistic' with rejection_sample_method="
                f"{spec.rejection_sample_method!r}: the TPU rejection sampler "
                "accepts a draft as if it were greedy (q one-hot), which is "
                "only distribution-preserving for greedy drafts. Use "
                "rejection_sample_method='synthetic' or greedy drafting."
            )
        if not self._probabilistic and spec.draft_sample_method != "greedy":
            raise ValueError(
                "DSpark on TPU does not support draft_sample_method="
                f"{spec.draft_sample_method!r}."
            )
        self._vocab_parallel_markov = envs.TPU_DSPARK_VOCAB_PARALLEL_MARKOV
        if self._vocab_parallel_markov and self._probabilistic:
            raise ValueError(
                "TPU_DSPARK_VOCAB_PARALLEL_MARKOV implements greedy drafting "
                "only; set draft_sample_method='greedy'."
            )
        # Width of compute_draft_logits: a reduced-vocab draft samples in its
        # own vocabulary and remaps after, which draws the same token as
        # upstream's scatter into target space (the other columns are -inf).
        self._draft_vocab_size = hf_dict.get("draft_vocab_size") or hf_dict.get(
            "vocab_size"
        )
        if self._probabilistic and not self._draft_vocab_size:
            raise ValueError(
                "Probabilistic DSpark drafting needs the draft vocab size."
            )
        # Keyed by padded request count: the all-greedy inputs of dummy
        # forwards.
        self._dummy_sample_inputs: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        # idle-engine draft sampling operands by (rows, width) (run_dp_dummy_draft)
        self._mk_idle_rows: dict[tuple[int, int], tuple[torch.Tensor, ...]] = {}
        logger.info(
            "DSpark: sample_from_anchor=%s, K=%d (checkpoint block %s), "
            "draft sampling %s.",
            self.sample_from_anchor,
            K,
            trained_block,
            spec.draft_sample_method,
        )

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
        if self._vocab_parallel_markov:
            if not hasattr(self.draft_model, "prepare_vocab_parallel_markov"):
                raise RuntimeError(
                    "TPU_DSPARK_VOCAB_PARALLEL_MARKOV: "
                    f"{type(self.draft_model).__name__} has no vocabulary-parallel "
                    "Markov head."
                )
            self.draft_model.prepare_vocab_parallel_markov()
        if envs.TPU_DSPARK_SHARDED_CONTEXT_KV:
            prepare = getattr(
                self.draft_model.model, "prepare_sharded_context_kv", None
            )
            if prepare is None:
                raise RuntimeError(
                    "TPU_DSPARK_SHARDED_CONTEXT_KV: "
                    f"{type(self.draft_model).__name__} has no sharded context-KV "
                    "projection."
                )
            prepare()

    # --- the draft forward on the DeepSeek-V4 megakernel ------------------

    def _megakernel_draft(self):
        """(target megakernel, its draft) when this step's target forward ran
        on the megakernel and the draft runs there too; else (None, None)."""
        mk = getattr(self.runner, "_dsv4_mk", None)
        if mk is None or mk.draft is None or mk.last_run is None:
            return None, None
        return mk, mk.draft

    @torch.no_grad()
    def run_dp_dummy_draft(self, num_chunks: int) -> None:
        mk, draft = self._megakernel_draft()
        if draft is None:
            super().run_dp_dummy_draft(num_chunks)
            return
        # a megakernel step is one chunk on every rank (decide_step)
        draft.idle(mk.last_run)
        if self._draft_replicated:
            return
        # The busy engines sample the draft rows with TP collectives (vocab-
        # parallel logits / Markov argmax); like any TP collective here they
        # pair across DP engines by emission order, so replay them at the
        # busy engines' shape (`_prepare_dflash_inputs`' lockstep bucket).
        from vllm_torchtpu.utils import synchronize_tensors

        rows = self._dp_draft_bucket()
        width = draft.spec.vocab_size if draft.logits else draft.spec.hidden_size
        ops = self._mk_idle_rows.get((rows, width))
        if ops is None:
            dev = self.runner.device
            ops = (torch.zeros((rows, width), dtype=torch.bfloat16, device=dev),
                   torch.zeros(rows, dtype=torch.int32, device=dev))  # fmt: skip
            self._mk_idle_rows[(rows, width)] = ops
        hidden, ids = ops
        synchronize_tensors(
            self._sample_megakernel_draft(
                draft, hidden, ids, self._draft_sample_inputs(rows // self.block_size)
            ),
            wait=False,
        )

    @torch.no_grad()
    def propose(
        self,
        sampled_token_ids,
        discard_sampled_tokens_req_indices,
        num_rejected_tokens_np,
        scheduler_output,
        return_device: bool = False,
        next_tokens_per_chunk: list[torch.Tensor] | None = None,
        device_seed: torch.Tensor | None = None,
    ):
        mk, draft = self._megakernel_draft()
        if draft is None:
            return super().propose(
                sampled_token_ids,
                discard_sampled_tokens_req_indices,
                num_rejected_tokens_np,
                scheduler_output,
                return_device=return_device,
                next_tokens_per_chunk=next_tokens_per_chunk,
                device_seed=device_seed,
            )
        if self.runner.input_batch.num_reqs == 0:
            return []
        chunks = self.draft_chunks
        if len(chunks) != 1:
            raise AssertionError(f"megakernel step with {len(chunks)} draft chunks")
        chunk = chunks[0]
        seed = (
            device_seed[chunk.start_index : chunk.start_index + chunk.num_reqs]
            if device_seed is not None
            else None
        )
        input_ids, positions, seq_lens = self._prepare_dflash_inputs(
            chunk,
            next_tokens_device=next_tokens_per_chunk[0]
            if next_tokens_per_chunk
            else None,
            device_seed=seed,
        )
        aux = chunk.aux_hidden_states
        aux = list(aux) if isinstance(aux, (list, tuple)) else [aux]
        ctx_positions = chunk.position_ids
        if ctx_positions.dim() > 1:
            ctx_positions = ctx_positions[0]
        md = chunk.attn_metadata
        first = md[draft.caches.group_layer_names[0]]
        hidden = draft.forward(
            mk.last_run,
            input_ids,
            positions,
            seq_lens,
            aux,
            ctx_positions,
            first.seq_lens,
            first.request_distribution[2],
            [md[n].block_tables for n in draft.caches.group_layer_names],
        )
        padded_len = input_ids.shape[0]
        sample_inputs = self._draft_sample_inputs(
            padded_len // self.block_size, chunk.start_index, chunk.num_reqs
        )
        draft_tokens = self._sample_megakernel_draft(
            draft, hidden, input_ids, sample_inputs
        )
        if return_device:
            return draft_tokens[: chunk.num_reqs]
        return draft_tokens.cpu().tolist()[: chunk.num_reqs]

    def _sample_megakernel_draft(
        self, draft, hidden: torch.Tensor, input_ids: torch.Tensor, sample_inputs
    ) -> torch.Tensor:
        """The draft tokens from the draft megakernel's output (the logits with
        `draft.logits`, else the pre-norm hidden)."""
        with set_vllm_model_wrapper_context(
            mesh=self.runner.mesh, vllm_config=self.vllm_config
        ):
            if draft.logits:  # the draft step applied norm + lm_head
                return self._sample_from_draft_logits(
                    hidden, input_ids, self.block_size, *sample_inputs
                )
            tokens, _ = self._sample_from_hidden(
                hidden, input_ids, self.block_size, *sample_inputs
            )
            return tokens

    @torch.no_grad()
    def precompile_megakernel_sampling(self, logits: bool, width: int) -> None:
        """Warm the sampling graph the megakernel draft path calls
        (`_sample_from_draft_logits` on [rows, vocab] logits, else
        `_sample_from_hidden` on [rows, hidden]) at every draft bucket
        `precompile` warms the stock draft forward at."""
        from vllm_torchtpu.utils import synchronize_tensors

        runner, block_size = self.runner, self.block_size
        max_chunk_reqs = max(
            runner.num_reqs_max_model_len, runner.num_reqs_most_model_len or 0
        )
        max_draft_tokens = self._get_padded_len(max_chunk_reqs * block_size)
        for rows in runner.num_tokens_paddings:
            if not block_size <= rows <= max_draft_tokens:
                continue
            x = torch.zeros((rows, width), dtype=torch.bfloat16).to(runner.device)
            ids = torch.zeros(rows, dtype=torch.int32).to(runner.device)
            sample_inputs = self._draft_sample_inputs(rows // block_size)
            with set_vllm_model_wrapper_context(
                mesh=runner.mesh, vllm_config=self.vllm_config
            ):
                if logits:
                    out = self._sample_from_draft_logits(
                        x, ids, block_size, *sample_inputs
                    )
                else:
                    out, _ = self._sample_from_hidden(
                        x, ids, block_size, *sample_inputs
                    )
            synchronize_tensors(out)

    def _num_sampled(self, block_size: int) -> int:
        return block_size if self.sample_from_anchor else block_size - 1

    def _draft_sample_inputs(
        self, padded_num_reqs: int, start_index: int = 0, num_reqs: int = 0
    ) -> tuple[torch.Tensor, ...]:
        """Probabilistic drafting: per-row temperatures [P] (0 = greedy) and
        uniform noise [P, num_sampled, vocab] for the Gumbel-max draw."""
        if not self._probabilistic:
            return ()
        device = self.runner.device
        if num_reqs == 0:
            cached = self._dummy_sample_inputs.get(padded_num_reqs)
            if cached is None:
                cached = (
                    torch.zeros(padded_num_reqs, dtype=torch.float32, device=device),
                    torch.full(
                        (
                            padded_num_reqs,
                            self._num_sampled(self.block_size),
                            self._draft_vocab_size,
                        ),
                        0.5,
                        dtype=torch.float32,
                        device=device,
                    ),
                )
                self._dummy_sample_inputs[padded_num_reqs] = cached
            return cached
        temps_cpu = torch.zeros(padded_num_reqs, dtype=torch.float32)
        temps_cpu[:num_reqs].copy_(
            self.runner.input_batch.temperature_cpu_tensor[
                start_index : start_index + num_reqs
            ]
        )
        temps = temps_cpu.to(device, non_blocking=True)
        noise = torch.rand(
            (
                padded_num_reqs,
                self._num_sampled(self.block_size),
                self._draft_vocab_size,
            ),
            dtype=torch.float32,
            device=device,
        )
        return temps, noise

    @staticmethod
    def _sample_draft(
        logits: torch.Tensor,
        temps: torch.Tensor | None,
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        """argmax, or the Gumbel-max draw from softmax(logits / T) per row
        (rows at T == 0 stay greedy). `logits` is [P, V], `noise` [P, V]."""
        if temps is None or noise is None:
            return logits.argmax(dim=-1)
        greedy = temps <= 0.0
        scaled = logits.float() / torch.where(greedy, 1.0, temps).unsqueeze(1)
        u = noise.clamp(min=torch.finfo(torch.float32).tiny, max=1.0 - 1e-7)
        gumbel = -torch.log(-torch.log(u))
        sampled = (scaled + gumbel).argmax(dim=-1)
        return torch.where(greedy, logits.argmax(dim=-1), sampled)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _dflash_forward_and_sample(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_size: int,
        temps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Backbone forward + sequential Markov sampling, in-graph."""
        assert self.draft_model is not None
        hidden = self.draft_model.model(
            input_ids=input_ids,
            positions=positions,
        )
        return self._logits_and_sample(hidden, input_ids, block_size, temps, noise)

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _sample_from_hidden(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        block_size: int,
        temps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`_dflash_forward_and_sample` after the backbone: for a draft hidden
        computed elsewhere (the DeepSeek-V4 megakernel's draft call)."""
        return self._logits_and_sample(hidden, input_ids, block_size, temps, noise)

    def _logits_and_sample(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        block_size: int,
        temps: torch.Tensor | None,
        noise: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draft logits + the sequential Markov sampling, traced inside a
        caller's compiled graph.

        Dense layout: every query slot predicts the token at its position + 1,
        so all block_size slots are sampled (DFlash samples only the mask
        slots). Bonus-anchor layout: the anchor is a pure bonus token, so only
        the K mask slots (1..block_size-1) are sampled. Either way the Markov
        chain seeds from the anchor input id and runs a static K-iteration
        loop: embedding gather + rank-r bias + argmax.
        """
        assert self.draft_model is not None
        padded_num_reqs = hidden.shape[0] // block_size
        valid_hidden = hidden[: padded_num_reqs * block_size]

        if self._vocab_parallel_markov:
            logits = self.draft_model.compute_draft_logits_local(valid_hidden)
            return self._markov_sample_vocab_parallel(
                logits, input_ids, block_size
            ), hidden
        # Draft-vocab logits for every slot; the Markov bias is added in
        # draft space and ids are remapped to target vocab after sampling.
        logits = self.draft_model.compute_draft_logits(valid_hidden)
        return self._markov_sample(logits, input_ids, block_size, temps, noise), hidden

    @torch.compile(backend="tpu", fullgraph=True, dynamic=False)
    def _sample_from_draft_logits(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        block_size: int,
        temps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """The Markov sampling on full draft-vocab logits computed elsewhere
        (the DeepSeek-V4 megakernel's draft call with its lm_head)."""
        padded_num_reqs = logits.shape[0] // block_size
        logits = logits[: padded_num_reqs * block_size]
        return self._markov_sample(logits, input_ids, block_size, temps, noise)

    def _markov_sample_vocab_parallel(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        """Greedy on this rank's vocabulary shard of the draft logits; one
        (max, argmax) pair per rank and row is exchanged per draft position."""
        assert self.draft_model is not None
        padded_num_reqs = logits.shape[0] // block_size
        logits_3d = logits.view(padded_num_reqs, block_size, logits.shape[-1])
        prev = input_ids[: padded_num_reqs * block_size].view(
            padded_num_reqs, block_size
        )[:, 0]
        draft_tokens = []
        sample_start = 0 if self.sample_from_anchor else 1
        for i in range(sample_start, block_size):
            bias = self.draft_model.markov_bias_local(
                self.draft_model.markov_embed(prev)
            )
            step = self.draft_model.vocab_parallel_argmax(logits_3d[:, i] + bias)
            draft_tokens.append(step)
            prev = step
        return torch.stack(draft_tokens, dim=1)

    def _markov_sample(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        block_size: int,
        temps: torch.Tensor | None,
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        """[padded_num_reqs * block_size, V] draft logits -> the draft tokens
        [padded_num_reqs, num_sampled]."""
        assert self.draft_model is not None
        padded_num_reqs = logits.shape[0] // block_size
        logits_3d = logits.view(padded_num_reqs, block_size, logits.shape[-1])
        ids_3d = input_ids[: padded_num_reqs * block_size].view(
            padded_num_reqs, block_size
        )
        prev = ids_3d[:, 0]
        draft_tokens = []
        sample_start = 0 if self.sample_from_anchor else 1
        for j, i in enumerate(range(sample_start, block_size)):
            bias = self.draft_model.markov_bias(self.draft_model.markov_embed(prev))
            step = self.draft_model.map_draft_to_target(
                self._sample_draft(
                    logits_3d[:, i] + bias,
                    temps,
                    None if noise is None else noise[:, j],
                )
            )
            draft_tokens.append(step)
            prev = step
        return torch.stack(draft_tokens, dim=1)
