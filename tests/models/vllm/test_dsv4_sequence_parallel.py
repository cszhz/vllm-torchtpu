# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DeepSeek-V4 sequence-parallel prefill matches the replicated TP path.

TP ranks are simulated by threads that exchange tensors through a barrier, so
the decoder layer and the MoE block run their real control flow on each rank's
shard, against stand-ins for the kernels that keep the TP arithmetic: a
row-parallel attention (each rank a share of the output projection), a
per-token mHC seam, routed experts that see only their own rows, and a shared
expert sliced over its intermediate dimension.
"""

import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_torchtpu.models.vllm.deepseek_v4 import model as model_mod
from vllm_torchtpu.models.vllm.deepseek_v4 import moe as moe_mod
from vllm_torchtpu.models.vllm.deepseek_v4 import sequence_parallel as sp

HIDDEN, HC, TOKENS, TP = 64, 4, 32, 4


class SimulatedTP:
    """Collectives over threads; each thread is one TP rank."""

    def __init__(self, size):
        self.size = size
        self.barrier = threading.Barrier(size)
        self.slots = [None] * size
        self.local = threading.local()

    @property
    def rank(self):
        return self.local.rank

    def exchange(self, x):
        self.barrier.wait()
        self.slots[self.rank] = x
        self.barrier.wait()
        values = list(self.slots)
        self.barrier.wait()
        return values

    def all_reduce(self, x):
        values = self.exchange(x)
        out = values[0].float()
        for v in values[1:]:
            out = out + v.float()
        return out.to(x.dtype)

    def gather(self, x, num_tokens=None):
        out = torch.cat(self.exchange(x), dim=0)
        return out if num_tokens is None else out.narrow(0, 0, num_tokens)

    def reduce_scatter(self, x):
        return self.all_reduce(x).chunk(self.size, dim=0)[self.rank].clone()

    def run(self, fn):
        results, errors = [None] * self.size, []

        def body(rank):
            self.local.rank = rank
            try:
                results[rank] = fn(rank)
            except BaseException as exc:  # pragma: no cover - surfaced below
                errors.append(exc)
                self.barrier.abort()

        threads = [threading.Thread(target=body, args=(r,)) for r in range(self.size)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if errors:
            raise errors[0]
        return results


@pytest.fixture
def tp(monkeypatch):
    sim = SimulatedTP(TP)
    monkeypatch.setattr(sp, "gather", sim.gather)
    monkeypatch.setattr(sp, "reduce_scatter", sim.reduce_scatter)
    monkeypatch.setattr(sp, "tp_world_size", lambda: sim.size)
    return sim


def _rank(r):
    return torch.tensor([r], dtype=torch.int32)


def test_shard_takes_contiguous_rows_in_rank_order(tp):
    x = torch.arange(TOKENS * 3).reshape(TOKENS, 3)
    parts = [sp.shard(x, _rank(r)) for r in range(TP)]
    assert torch.equal(torch.cat(parts), x)
    ids = torch.arange(TOKENS)
    assert torch.equal(sp.shard(ids, _rank(2)), ids[16:24])


@pytest.mark.parametrize(
    "tokens,tp_size,minimum,expected",
    [
        (512, 4, 512, True),
        (256, 4, 512, False),
        (528, 4, 512, False),  # not a multiple of tp * ROW_ALIGNMENT
        (2048, 8, 512, True),
        (64, 8, 0, False),  # below tp * ROW_ALIGNMENT
        (512, 1, 0, False),
    ],
)
def test_eligible(monkeypatch, tokens, tp_size, minimum, expected):
    monkeypatch.setattr(sp.envs, "TPU_DSV4_PREFILL_SP", True)
    monkeypatch.setattr(sp.envs, "TPU_DSV4_PREFILL_SP_MIN_TOKENS", minimum)
    monkeypatch.setattr(sp, "tp_world_size", lambda: tp_size)
    assert sp.eligible(tokens) is expected


def test_eligible_is_off_by_default(monkeypatch):
    monkeypatch.setattr(sp, "tp_world_size", lambda: 4)
    assert sp.envs.TPU_DSV4_PREFILL_SP is False
    assert sp.eligible(4096) is False


class PartialAttention(nn.Module):
    """Rank r applies its column block of a shared projection to the full rows;
    the TP sum of the shares is the full projection."""

    def __init__(self, sim, weight):
        super().__init__()
        self.sim, self.weight = sim, weight

    def forward(self, positions, x, reduce_output=True):
        assert x.shape[0] == TOKENS, "attention must see every token"
        assert positions.shape[0] == TOKENS
        block = self.weight.shape[0] // self.sim.size
        r = self.sim.rank
        rows = slice(r * block, (r + 1) * block)
        part = torch.tanh(x[:, rows]) @ self.weight[rows]
        return self.sim.all_reduce(part) if reduce_output else part


class StubMHC:
    """Per-token seam with the fused op's signature: mixes the residual
    streams with `x`, returns (residual, post_mix, res_mix, layer_input)."""

    def __init__(self, fn):
        self.fn = fn

    def pre(self, x, fn_hi, fn_mid, fn_lo, scale, base, *norm):
        post_mix = torch.sigmoid(x.mean(-1, keepdim=True))
        res_mix = torch.softmax(x[..., :HC] @ self.fn, dim=-1)
        return post_mix, res_mix, x.mean(1)

    def fused(self, x, residual, post_mix, res_mix, fn_hi, fn_mid, fn_lo, *rest):
        residual = res_mix @ residual + post_mix * x.unsqueeze(1)
        post_mix, res_mix, layer_input = self.pre(
            residual, None, None, None, None, None
        )
        return residual, post_mix, res_mix, layer_input


class StubFFN(nn.Module):
    """Row-wise FFN: the replicated path all-reduces a rank share, the
    sequence-parallel path computes its own rows."""

    def __init__(self, sim, weight):
        super().__init__()
        self.sim, self.weight = sim, weight

    def forward(self, x, input_ids):
        assert x.shape[0] == TOKENS
        return self.sim.all_reduce(self._share(x, input_ids) / self.sim.size)

    def forward_sequence_parallel(self, x, input_ids, padding_mask=None):
        assert padding_mask is None
        assert x.shape[0] == TOKENS // self.sim.size
        return self._share(x, input_ids)

    def _share(self, x, input_ids):
        return torch.relu(x @ self.weight) + input_ids[:, None].to(x.dtype) * 1e-3


def _layer(sim, weights):
    layer = object.__new__(model_mod.DeepseekV4DecoderLayer)
    nn.Module.__init__(layer)
    layer.fuse_norm = True
    layer.attn = PartialAttention(sim, weights["attn"])
    layer.ffn = StubFFN(sim, weights["ffn"])
    for name in ("hc_attn_fn", "hc_ffn_fn"):
        for suffix in ("hi", "mid", "lo"):
            setattr(layer, f"{name}_{suffix}", torch.zeros(1))
    layer.hc_attn_scale = layer.hc_ffn_scale = torch.ones(3)
    layer.hc_attn_base = layer.hc_ffn_base = torch.zeros(1)
    layer.attn_norm = layer.ffn_norm = SimpleNamespace(weight=torch.ones(HIDDEN))
    object.__setattr__(layer, "_mhc_ops_instance", StubMHC(weights["mhc"]))
    return layer


def test_decoder_layers_match_replicated_path(tp):
    torch.manual_seed(0)
    weights = dict(
        attn=torch.randn(HIDDEN, HIDDEN) / HIDDEN**0.5,
        ffn=torch.randn(HIDDEN, HIDDEN) / HIDDEN**0.5,
        mhc=torch.randn(HC, HC),
    )
    x0 = torch.randn(TOKENS, HC, HIDDEN)
    positions = torch.arange(TOKENS)
    ids = torch.arange(TOKENS)

    def run(sequence_parallel):
        def per_rank(r):
            layer = _layer(tp, weights)
            x, i = x0, ids
            if sequence_parallel:
                x, i = sp.shard(x0, _rank(r)), sp.shard(ids, _rank(r))
            state = (x, None, None, None)
            for _ in range(3):
                x, residual, post_mix, res_mix = layer(
                    state[0],
                    positions,
                    i,
                    state[2],
                    state[3],
                    state[1],
                    sequence_parallel=sequence_parallel,
                )
                state = (x, residual, post_mix, res_mix)
            out = state[1] + state[0].unsqueeze(1)
            return tp.gather(out) if sequence_parallel else out

        return tp.run(per_rank)

    replicated, sharded = run(False), run(True)
    for r in range(TP):
        torch.testing.assert_close(sharded[r], replicated[r], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(replicated[r], replicated[0])


class RowRoutedMethod:
    """Routed experts that are a per-row function: the token-sharded op sees
    only this rank's rows, so it must agree row for row with the runner."""

    def __init__(self, weight):
        self.weight = weight

    def forward_sequence_parallel(
        self, layer, x, router_logits, input_ids, order_after=None
    ):
        return self.routed(x, router_logits, input_ids)

    def routed(self, x, router_logits, input_ids):
        gate = torch.softmax(router_logits, dim=-1)[:, :1]
        return torch.tanh(x @ self.weight) * gate + input_ids[:, None].to(x.dtype)


class SlicedMLP(nn.Module):
    """A shared expert TP-sliced over its intermediate dimension (partial
    output per rank), or the whole thing when `sim` is None."""

    def __init__(self, sim, up, down):
        super().__init__()
        self.sim, self.up, self.down = sim, up, down

    def forward(self, x):
        up, down = self.up, self.down
        if self.sim is not None:
            n = up.shape[1] // self.sim.size
            cols = slice(self.sim.rank * n, (self.sim.rank + 1) * n)
            up, down = up[:, cols], down[cols]
        return torch.relu(x @ up) @ down


def _moe(sim, weights, *, full_shared, routed_scale):
    moe = object.__new__(moe_mod.DeepseekV4MoE)
    nn.Module.__init__(moe)
    gate_weight = weights["gate"]
    moe.gate = type(
        "Gate", (), {"tid2eid": None, "__call__": lambda s, x: (x @ gate_weight, None)}
    )()
    method = RowRoutedMethod(weights["routed"])
    routed_layer = SimpleNamespace(quant_method=method)
    moe.experts = SimpleNamespace(
        routed_experts=routed_layer, routed_scaling_factor=routed_scale
    )
    moe.shared_experts = SlicedMLP(sim, weights["up"], weights["down"])
    moe.shared_experts_full = (
        SlicedMLP(None, weights["up"], weights["down"]) if full_shared else None
    )
    return moe, method


@pytest.mark.parametrize("full_shared", [False, True])
def test_moe_block_matches_runner_sum(tp, full_shared):
    torch.manual_seed(1)
    weights = dict(
        gate=torch.randn(HIDDEN, 8),
        routed=torch.randn(HIDDEN, HIDDEN) / HIDDEN**0.5,
        up=torch.randn(HIDDEN, 32) / HIDDEN**0.5,
        down=torch.randn(32, HIDDEN) / 32**0.5,
    )
    x = torch.randn(TOKENS, HIDDEN)
    ids = torch.arange(TOKENS)
    scale = 2.5

    def per_rank(r):
        moe, method = _moe(tp, weights, full_shared=full_shared, routed_scale=scale)
        # What the runner forms on the replicated path: all-reduced shared
        # expert plus the scaled routed output, at every row.
        shared = tp.all_reduce(moe.shared_experts(x))
        want = shared + method.routed(x, x @ weights["gate"], ids) * scale
        got = moe.forward_sequence_parallel(
            sp.shard(x, _rank(r)), sp.shard(ids, _rank(r))
        )
        return sp.shard(want, _rank(r)), got

    for want, got in tp.run(per_rank):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_full_shared_alias_only_when_the_copy_exists():
    params = {
        "model.layers.3.ffn.shared_experts.gate_up_proj.weight": 0,
        "model.layers.3.ffn.shared_experts_full.gate_up_proj.weight": 0,
        "model.layers.4.ffn.shared_experts.down_proj.weight": 0,
    }
    alias = model_mod._with_full_shared_alias
    assert alias("model.layers.3.ffn.shared_experts.gate_up_proj.weight", params) == [
        "model.layers.3.ffn.shared_experts.gate_up_proj.weight",
        "model.layers.3.ffn.shared_experts_full.gate_up_proj.weight",
    ]
    assert alias("model.layers.4.ffn.shared_experts.down_proj.weight", params) == [
        "model.layers.4.ffn.shared_experts.down_proj.weight"
    ]
    assert alias("model.layers.3.attn.wo_b.weight", params) == [
        "model.layers.3.attn.wo_b.weight"
    ]


def test_moe_block_masks_this_ranks_padded_rows(tp):
    torch.manual_seed(2)
    weights = dict(
        gate=torch.randn(HIDDEN, 8),
        routed=torch.randn(HIDDEN, HIDDEN) / HIDDEN**0.5,
        up=torch.randn(HIDDEN, 32) / HIDDEN**0.5,
        down=torch.randn(32, HIDDEN) / 32**0.5,
    )
    rows = TOKENS // TP
    x = torch.randn(rows, HIDDEN)
    mask = torch.zeros(rows, dtype=torch.bool)
    mask[rows - 3 :] = True
    moe, method = _moe(None, weights, full_shared=True, routed_scale=1.0)
    seen = {}

    def record(layer, x, router_logits, input_ids, order_after=None):
        seen["logits"] = router_logits
        return torch.zeros_like(x)

    method.forward_sequence_parallel = record
    moe.forward_sequence_parallel(x, torch.arange(rows), mask)
    logits = seen["logits"]
    assert torch.isnan(logits[mask]).all()
    assert torch.isfinite(logits[~mask]).all()


def test_select_rows_reads_this_ranks_block_of_a_longer_buffer():
    buffer = torch.arange(100)
    assert torch.equal(sp.select_rows(buffer, _rank(2), 8), torch.arange(16, 24))
