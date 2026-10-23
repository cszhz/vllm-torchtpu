# SPDX-License-Identifier: Apache-2.0
"""Check source provenance, checkpoint contracts, and replay conservation."""

import hashlib
import json
import unittest

import replay


class ReplayTests(unittest.TestCase):
    def test_fixture_hashes(self):
        replay.verify_inputs()

    def test_recorded_batch_allocations_are_reproducible(self):
        recorded = json.loads((replay.ROOT / "results/comparison.json").read_text())[
            "results"
        ]
        for expected in recorded:
            actual = replay.run(
                expected["mode"],
                expected["global_token_budget"],
                expected["compile_sizes_per_pcp_rank"],
            )
            self.assertEqual(actual["batches"], expected["batches"])
            self.assertEqual(actual["per_request"], expected["per_request"])

    def test_upstream_matches_historical_source(self):
        digest = hashlib.sha256(
            (replay.INPUTS / "vllm_scheduler.py.txt").read_bytes()
        ).hexdigest()
        self.assertEqual(
            digest, "abca7134821e2fb5cc8572df5c4a0b570ecf702646254327bc786fc452074983"
        )

    def test_current_config_matches_control(self):
        config = json.loads((replay.INPUTS / "scheduler_config.json").read_text())
        self.assertEqual(config["engine_args"]["max_num_batched_tokens"], 8192)
        self.assertEqual(
            config["engine_args"]["compilation_config"]["compile_sizes"],
            [256, 512, 1024],
        )
        self.assertEqual(config["envs"]["TPU_PP_DYNAMIC_CHUNKS"], "0")
        self.assertEqual(config["engine_args"]["long_prefill_token_threshold"], 0)

    def test_checkpoint_no_extra_hash_backoff(self):
        scheduler = replay.make_scheduler({}, 32768)
        for prompt, start, expected in [
            (172036, 153600, 18432),
            (172032, 153600, 15360),
            (49391, 43008, 6144),
        ]:
            request = replay.new_request({"id": "r", "prompt_tokens": prompt})
            request.num_computed_tokens = start
            actual = scheduler._mamba_block_aligned_split(request, prompt - start)
            self.assertEqual(actual, expected)
            self.assertTrue(scheduler.use_eagle)
            self.assertEqual(scheduler.cache_config.block_size, 384)

    def test_one_k_budget_alignment_and_tail(self):
        scheduler = replay.make_scheduler({}, 8192)
        request = replay.new_request({"id": "r", "prompt_tokens": 93027})
        request.num_computed_tokens = 82944
        self.assertEqual(scheduler._mamba_block_aligned_split(request, 8192), 6144)
        self.assertEqual(scheduler._mamba_block_aligned_split(request, 2048), 0)
        request.num_computed_tokens = 92160
        self.assertEqual(scheduler._mamba_block_aligned_split(request, 867), 867)

    def test_all_scenarios_preserve_each_request_and_budget(self):
        workload = json.loads((replay.INPUTS / "workload.json").read_text())["requests"]
        for mode in ["closed_loop_c4", "historical_arrivals", "all_ready_16"]:
            for budget, buckets in [(8192, [256, 512, 1024]), (32768, [4096])]:
                with self.subTest(mode=mode, budget=budget):
                    result = replay.run(mode, budget, buckets)
                    self.assertEqual(result["computed_tokens"], 190749)
                    for row in workload:
                        self.assertEqual(
                            sum(result["per_request"][row["id"]]),
                            row["prompt_tokens"] - row["cached_tokens"],
                        )
                    for batch in result["batches"]:
                        self.assertLessEqual(batch["tokens"], budget)
                        self.assertEqual(
                            sum(batch["assigned"].values()), batch["tokens"]
                        )

    def test_compile_only_change_does_not_change_schedule(self):
        for mode in ["closed_loop_c4", "historical_arrivals", "all_ready_16"]:
            current = replay.run(mode, 8192, [256, 512, 1024])
            compile_only = replay.run(mode, 8192, [4096])
            self.assertEqual(current["batches"], compile_only["batches"])

    def test_mtp_checkpoint_on_off_same_for_these_partial_tails(self):
        for budget, buckets in [(8192, [1024]), (32768, [4096])]:
            enabled = replay.run("closed_loop_c4", budget, buckets, mtp=True)
            disabled = replay.run("closed_loop_c4", budget, buckets, mtp=False)
            self.assertEqual(enabled["batches"], disabled["batches"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
