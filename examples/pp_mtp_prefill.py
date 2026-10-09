# SPDX-License-Identifier: Apache-2.0
"""Offline PP prefill + final-stage MTP diagnostic; see docs/pp_mtp_prefill.md."""

import argparse
import io
import json
from pathlib import Path

import torch
from vllm import LLM, SamplingParams
from vllm.engine.arg_utils import EngineArgs

from vllm_torchtpu import envs


def zero_caches(worker):
    from vllm_torchtpu.utils import synchronize_tensors

    runner = worker.model_runner
    for tensor in runner.kv_caches:
        tensor.zero_()
    synchronize_tensors(runner.kv_caches, wait=True)


def save_snapshot(worker, directory: str):
    from vllm.distributed import get_pcp_group, get_pp_group, get_world_group

    runner = worker.model_runner
    pp = get_pp_group()
    rank = get_world_group().rank
    if not pp.is_last_rank:
        assert runner.drafter is None
        return {"rank": rank, "mtp_loaded": False}
    drafter = runner._async_drafter
    layers = runner.vllm_config.compilation_config.static_forward_context
    caches = {
        name: layers[name].kv_cache.cpu()
        for name in sorted(drafter._draft_attn_layer_names)
    }
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "chunks": drafter.prefill_snapshots,
            "mtp_kv": caches,
            "pcp_rank": get_pcp_group().rank_in_group,
            "pp_rank": pp.rank_in_group,
        },
        path / f"rank-{rank}.pt",
    )
    return {"rank": rank, "mtp_loaded": True, "mtp_layers": list(caches)}


@torch.no_grad()
def replay_mtp(worker, payload: bytes):
    """Isolate the head from backbone differences across PP/PCP configurations."""
    from dataclasses import replace
    from types import SimpleNamespace

    from vllm.distributed import get_pp_group

    if not get_pp_group().is_last_rank:
        return
    reference = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    runner = worker.model_runner
    drafter = runner._async_drafter
    assert len(drafter.draft_chunks) == 1
    chunk = drafter.draft_chunks[0]
    hidden = torch.zeros(chunk.hidden_states.shape, dtype=chunk.hidden_states.dtype)
    assert int(chunk.query_start_loc_np[-1]) == len(reference["input_ids"])
    for logical in range(len(reference["input_ids"])):
        local = chunk.sequence_layout_plan.local_index_for_request_major_token(logical)
        if local is not None:
            hidden[local] = reference["target_hidden"][logical]
    zero_caches(worker)
    drafter.draft_chunks = [replace(chunk, hidden_states=hidden.to(runner.device))]
    drafter.prefill_only(
        SimpleNamespace(), reference["input_ids"][-1:].to(runner.device)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--nnodes", type=int, default=2)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--pp", type=int, default=2)
    parser.add_argument("--pcp", type=int, default=8)
    parser.add_argument("--master-addr", default="127.0.0.1")
    parser.add_argument("--master-port", type=int, default=29510)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prompt", default="Explain why the sky is blue in one sentence."
    )
    parser.add_argument("--prompt-lengths", type=int, nargs="+")
    parser.add_argument("--replay-reference")
    parser.add_argument("--logprobs", type=int, default=20)
    args = parser.parse_args()
    if args.prompt_lengths and any(n < 1 or n >= 512 for n in args.prompt_lengths):
        parser.error("Prompt lengths must be in [1, 511] for the 512-token harness.")
    if not envs.TPU_MTP_PREFILL_ONLY:
        parser.error("Set TPU_MTP_PREFILL_ONLY=1 before starting Python.")
    kwargs = dict(
        model=args.model,
        nnodes=args.nnodes,
        node_rank=args.node_rank,
        master_addr=args.master_addr,
        master_port=args.master_port,
        pipeline_parallel_size=args.pp,
        tensor_parallel_size=1,
        prefill_context_parallel_size=args.pcp,
        enable_expert_parallel=True,
        distributed_executor_backend="mp",
        async_scheduling=False,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        language_model_only=True,
        max_model_len=512,
        max_num_batched_tokens=512,
        max_num_seqs=1,
        num_gpu_blocks_override=32,
        kv_cache_dtype="fp8",
        block_size=256,
        cp_kv_cache_interleave_size=128,
        mamba_cache_mode="align",
        attention_backend="CUSTOM",
        compilation_config={"compile_sizes": [512]},
        speculative_config={"method": "mtp", "num_speculative_tokens": 1},
    )
    if args.node_rank:
        from vllm.v1.executor.multiproc_executor import MultiprocExecutor

        config = EngineArgs(**kwargs).create_engine_config(headless=True)
        executor = MultiprocExecutor(config, monitor_workers=False)
        executor.start_worker_monitor(inline=True)
        return
    llm = LLM(**kwargs)
    prompts = [(args.prompt, args.output_dir)]
    if args.prompt_lengths:
        ids = llm.get_tokenizer().encode(args.prompt)
        prompts = [
            (
                {"prompt_token_ids": (ids * ((n + len(ids) - 1) // len(ids)))[:n]},
                str(Path(args.output_dir) / f"tokens-{n}"),
            )
            for n in args.prompt_lengths
        ]
    for prompt, directory in prompts:
        llm.llm_engine.collective_rpc(zero_caches)
        results = llm.generate(
            [prompt],
            SamplingParams(temperature=0, max_tokens=1, logprobs=args.logprobs),
        )
        result = results[0]
        output = result.outputs[0]
        assert output.logprobs is not None
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        (path / "first-token-logprobs.json").write_text(
            json.dumps(
                {
                    "pp": args.pp,
                    "pcp": args.pcp,
                    "prompt_token_ids": result.prompt_token_ids,
                    "token_id": output.token_ids[0],
                    "text": output.text,
                    "logprobs": {
                        str(token): {
                            "logprob": value.logprob,
                            "rank": value.rank,
                            "decoded_token": value.decoded_token,
                        }
                        for token, value in output.logprobs[0].items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n"
        )
        print("Snapshot directory:", directory)
        print("Target first token:", results[0].outputs[0].token_ids)
        print(llm.llm_engine.collective_rpc(save_snapshot, args=(directory,)))
        if args.replay_reference:
            from compare_mtp_prefill import read_snapshot

            reference_dir = Path(args.replay_reference)
            if args.prompt_lengths:
                reference_dir /= Path(directory).name
            reference = read_snapshot(str(reference_dir))
            payload = io.BytesIO()
            torch.save(reference, payload)
            llm.llm_engine.collective_rpc(replay_mtp, args=(payload.getvalue(),))
            print(
                llm.llm_engine.collective_rpc(
                    save_snapshot, args=(directory + "-replay",)
                )
            )


if __name__ == "__main__":
    main()
