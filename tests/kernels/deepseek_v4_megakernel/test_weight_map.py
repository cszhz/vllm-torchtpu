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
"""CPU tests: served DeepSeek-V4 tensors -> megakernel weight operands.

The fake module tree carries the served attribute paths, shapes and dtypes
(DSv4-Pro dims, 3 layers HCA/HCA/CSA, 2 local experts). Each packed operand
is checked bit-exact against `dense_weights`' packers fed natural arrays
sliced here with explicit index arithmetic, independently of weight_map's
rules.
"""

from __future__ import annotations

import types

import jax.numpy as jnp
import numpy as np
import pytest
import torch

from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_config as dc
from vllm_torchtpu.kernels.deepseek_v4_megakernel import dense_weights as dw
from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI
from vllm_torchtpu.kernels.deepseek_v4_megakernel import vllm_adapter as VA
from vllm_torchtpu.kernels.deepseek_v4_megakernel import weight_map as WM

NS = types.SimpleNamespace
RATIOS = (128, 128, 4)
R = 16
E_LOCAL = 2
HF = NS(
    num_hidden_layers=len(RATIOS),
    hidden_size=7168,
    q_lora_rank=1536,
    head_dim=512,
    num_attention_heads=128,
    o_groups=16,
    o_lora_rank=1024,
    moe_intermediate_size=3072,
    n_routed_experts=384,
    index_n_heads=64,
    index_head_dim=128,
    hc_mult=4,
    vocab_size=129280,
    compress_ratios=list(RATIOS) + [0],
    num_hash_layers=1,
    num_experts_per_tok=6,
    qk_rope_head_dim=64,
)
G = WM.Geometry.from_hf(HF)


def _rand(shape, dtype, gen):
    if dtype == torch.float8_e4m3fn:
        return (torch.randn(shape, generator=gen) * 32).to(dtype)
    if dtype == torch.int32:
        return torch.randint(0, 384, shape, generator=gen, dtype=dtype)
    return torch.randn(shape, generator=gen).to(dtype)


def _set(root, path, value):
    *parents, leaf = path.split(".")
    for p in parents:
        if not hasattr(root, p):
            setattr(root, p, NS())
        root = getattr(root, p)
    setattr(root, leaf, value)


def make_tree(nk=("wp", "gate"), seed=0, hf=HF, e_local=E_LOCAL, device="cpu"):
    """A causal LM tree with the served paths: random dense tensors, empty
    (never read) experts, embedding and lm_head. Keys in `nk` are stored in
    vLLM's [n, k] like an unquantized linear that kept its layout."""
    gen = torch.Generator().manual_seed(seed)
    g = WM.Geometry.from_hf(hf)
    lm = NS(config=hf)
    layers = []
    for i in range(len(g.ratios)):
        layer = NS()
        for key, sv in WM.served_layer_spec(g, i, e_local).items():
            dtype = getattr(torch, sv.dtype)
            shape = sv.shape[::-1] if key in nk else sv.shape
            if key in ("w13", "w2"):
                t = torch.empty(shape, dtype=dtype, device=device)
            else:
                t = _rand(shape, dtype, gen).to(device)
            _set(layer, sv.path, t)
        layers.append(layer)
    for key, sv in WM.served_global_spec(g).items():
        if sv.path.startswith("model.layers."):
            idx = int(sv.path.split(".")[2])
            rest = sv.path.split(".", 3)[3]
            _set(layers[idx], rest, _rand((4096, 64), torch.float32, gen).to(device))
            continue
        dtype = getattr(torch, sv.dtype)
        if key in ("embed", "lm_head"):
            t = torch.empty(sv.shape, dtype=dtype, device=device)
        else:
            t = _rand(sv.shape, dtype, gen).to(device)
        _set(lm, sv.path, t)
    lm.model.layers = layers
    return lm


def _spec():
    return KI.ModelSpec(
        hidden_size=7168, num_layers=len(RATIOS), compress_ratios=RATIOS,
        dtype="bfloat16", q=7, mode="dp", axis_name="ep", num_ranks=R,
        aux_layers=(), vocab_size=HF.vocab_size, num_hash_layers=1,
        n_routed_experts=384,
    )  # fmt: skip


# torch <-> jax on the host, bit for bit
_BITS = {
    torch.float8_e4m3fn: (torch.uint8, jnp.float8_e4m3fn, jnp.uint8),
    torch.bfloat16: (torch.int16, jnp.bfloat16, jnp.int16),
}


def t2j(t):
    if t.dtype in _BITS:
        tb, jd, _ = _BITS[t.dtype]
        return jnp.asarray(t.view(tb).numpy()).view(jd)
    return jnp.asarray(t.numpy())


def j2t(a):
    for td, (tb, jd, jb) in _BITS.items():
        if a.dtype == jd:
            return torch.from_numpy(np.asarray(a.view(jb))).view(td)
    return torch.from_numpy(np.asarray(a))


def host_run_jax(name, fn, *tensors):
    return j2t(fn(*[t2j(t) for t in tensors]))


@pytest.fixture(scope="module")
def tree():
    return make_tree()


@pytest.fixture(scope="module")
def built(tree):
    return {r: VA.build_weight_operands(tree, _spec(), r, host_run_jax) for r in (0, 5)}


# ---------------------------------------------------------------- reference


def ref_natural(tree, r, ratios=RATIOS, gate_nk=True):
    """The natural arrays for rank r, by index arithmetic on the fake tree."""
    q, gcol, S = 4096, 1024, 192
    per = {}

    def add(k, v):
        per.setdefault(k, []).append(v)

    for i, L in enumerate(tree.model.layers):
        a, f = L.attn, L.ffn
        add("wqa", a.fused_wqa_wkv.weight)
        add("wqa_scale", a.fused_wqa_wkv.weight_scale)
        add("wq_b", a.wq_b.weight[:, r * q : (r + 1) * q])
        add("wq_b_scale", a.wq_b.weight_scale[r * q : (r + 1) * q])
        add("wo_a", a.wo_a.weight[:, r * gcol : (r + 1) * gcol])
        add("wo_a_scale", a.wo_a.weight_scale[r * gcol : (r + 1) * gcol])
        add("wo_b", a.wo_b.weight[r * gcol : (r + 1) * gcol])
        add("wo_b_scale", a.wo_b.weight_scale)
        gu = f.shared_experts.gate_up_proj
        cols = list(range(r * S, (r + 1) * S)) + list(
            range(3072 + r * S, 3072 + (r + 1) * S)
        )
        add("w13", gu.weight[:, cols])
        add("w13_scale", gu.weight_scale[cols])
        add("w2", f.shared_experts.down_proj.weight[r * S : (r + 1) * S])
        add("w2_scale", f.shared_experts.down_proj.weight_scale)
        add("gate", f.gate.weight.t() if gate_nk else f.gate.weight)
        for k in ("q_norm", "kv_norm"):
            add(k, getattr(a, k).weight)
        add("attn_norm", L.attn_norm.weight)
        add("ffn_norm", L.ffn_norm.weight)
        for s in ("attn", "ffn"):
            for x in ("fn", "base", "scale"):
                add(f"hc_{s}_{x}", getattr(L, f"hc_{s}_{x}"))
        if ratios[i] <= 1:  # SWA: no compressor, no indexer
            continue
        if ratios[i] == 4:
            add("idx_wq_b", a.indexer.wq_b.weight)
            add("idx_wq_b_scale", a.indexer.wq_b.weight_scale)
            add("wp", a.indexer.weights_proj.weight.t())
            add("idx_comp", a.indexer.compressor.fused_wkv_wgate.weight)
            add("comp_csa", a.compressor.fused_wkv_wgate.weight)
        else:
            add("comp_hca", a.compressor.fused_wkv_wgate.weight)
    return {k: jnp.stack([t2j(v.contiguous()) for v in vs]) for k, vs in per.items()}


def _cfg():
    return dc.DenseChainConfig(ratios=RATIOS, tp=R)


# -------------------------------------------------------------------- tests


def test_served_tree_reads_and_flags_nk(tree):
    t, flags = VA.read_served(
        tree.model.layers[2], WM.served_layer_spec(G, 2, E_LOCAL), "l2"
    )
    assert flags["wp"] and flags["gate"] and not flags["wq_b"]
    assert t["wq_b"] is tree.model.layers[2].attn.wq_b.weight
    assert "idx_wq_b" in t and "tid2eid" not in t and "gate_bias" in t
    t0, _ = VA.read_served(
        tree.model.layers[0], WM.served_layer_spec(G, 0, E_LOCAL), "l0"
    )
    assert (
        "tid2eid" in t0 and "idx_wq_b" not in t0 and t0["comp_ape"].shape == (128, 512)
    )


def test_served_mismatch_names_the_path(tree):
    spec = WM.served_layer_spec(G, 1, E_LOCAL)
    bad = make_tree()
    bad.model.layers[1].attn.wq_b.weight = torch.zeros(
        1536, 65536, dtype=torch.bfloat16
    )
    with pytest.raises(VA.Unsupported, match=r"layers\.1\.attn\.wq_b\.weight"):
        VA.read_served(bad.model.layers[1], spec, "layers.1")
    # a fp8 linear may not hide in [n, k]: only nk_ok entries flip
    bad.model.layers[1].attn.wq_b.weight = torch.zeros(
        65536, 1536, dtype=torch.float8_e4m3fn
    )
    with pytest.raises(VA.Unsupported, match="wq_b"):
        VA.read_served(bad.model.layers[1], spec, "layers.1")
    # the grouped-matmul expert layout (fused EP op not armed) is refused
    bad = make_tree()
    del bad.model.layers[1].ffn.experts.routed_experts._tpu_fused_w13_scale
    with pytest.raises(VA.Unsupported, match="fused EP MoE v2 op is not armed"):
        VA.read_served(bad.model.layers[1], spec, "layers.1")


def test_operands_follow_the_manifest(tree, built):
    man = KI.weight_manifest(_spec())
    ops = built[0]
    assert len(ops) == len(man)
    for e, t in zip(man, ops):
        assert WM.entry_matches(e, t.shape, str(t.dtype).removeprefix("torch.")), e
    by = {e.name: t for e, t in zip(man, ops)}
    # served operands are the module tensors themselves (zero-copy)
    L0 = tree.model.layers[0]
    assert by["w13_0"] is L0.ffn.experts.routed_experts.w13_weight
    assert by["tid2eid_0"] is L0.ffn.gate.tid2eid
    assert by["embed"] is tree.model.embed_tokens.weight
    # rotary tables: packed two positions per 128-lane row, one copy per
    # distinct served table
    cs = L0.attn.rotary_emb.cos_sin_cache
    idx = tree.model.layers[2].attn.indexer_rotary_emb.cos_sin_cache
    assert torch.equal(by["cos_sin"], cs.reshape(-1, 128))
    assert torch.equal(by["idx_cos_sin"], idx.reshape(-1, 128))
    assert by["comp_cos_sin"] is by["cos_sin"]
    # scale tables padded to the f32 sublane tile, values intact
    s = L0.ffn.experts.routed_experts._tpu_fused_w13_scale
    assert by["w13_scale_0"].shape == (E_LOCAL, 16, 6144)
    assert torch.equal(by["w13_scale_0"][:, :14], s)
    assert torch.count_nonzero(by["w13_scale_0"][:, 14:]) == 0
    assert by["w2_scale_0"].shape == (E_LOCAL, 8, 7168)


def test_relaid_out_served_operands_are_copied(tree, monkeypatch):
    """A served operand the runner relaid out (tid2eid under
    relayout_hash_tables) goes to the kernel as a default-layout copy."""
    L0 = tree.model.layers[0]
    tid, norm = L0.ffn.gate.tid2eid, tree.model.norm.weight
    w13 = L0.ffn.experts.routed_experts.w13_weight
    # tid2eid relaid out; final_norm (a global) a view (small: copied); w13 a view
    # (large: kept, bind refuses it)
    layout = {id(tid): False, id(norm): None, id(w13): None}
    monkeypatch.setattr(VA, "default_layout", lambda t: layout.get(id(t), True))
    monkeypatch.setattr(VA, "_sync", lambda t: None)
    monkeypatch.setattr(VA, "_VIEW_COPY_MAX_BYTES", norm.numel() * norm.element_size())
    ops = VA.build_weight_operands(tree, _spec(), 0, host_run_jax)
    by = {e.name: t for e, t in zip(KI.weight_manifest(_spec()), ops)}
    assert by["tid2eid_0"] is not tid and torch.equal(by["tid2eid_0"], tid)
    assert by["final_norm"] is not norm and torch.equal(by["final_norm"], norm)
    assert by["w13_0"] is w13
    assert by["embed"] is tree.model.embed_tokens.weight


@pytest.mark.parametrize("rank", [0, 5])
def test_packed_streams_match_reference_slices(tree, built, rank):
    nat = ref_natural(tree, rank)
    cfg = _cfg()
    man = KI.weight_manifest(_spec())
    by = {e.name: t for e, t in zip(man, built[rank])}
    for name in cfg.stream_names:
        want = dw.pack_stream(name, nat, cfg)
        got = t2j(by["dense_" + name])
        assert got.dtype == want.dtype and got.shape == want.shape, name
        np.testing.assert_array_equal(
            np.asarray(got.view(jnp.uint8 if got.dtype.itemsize == 1 else jnp.int16)),
            np.asarray(want.view(jnp.uint8 if want.dtype.itemsize == 1 else jnp.int16)),
            err_msg=name,
        )
    np.testing.assert_array_equal(
        np.asarray(t2j(by["dense_smalls"])), np.asarray(dw.pack_smalls(nat, cfg))
    )


def test_ranks_get_different_slices_of_sharded_streams(built):
    man = [e.name for e in KI.weight_manifest(_spec())]
    a, b = dict(zip(man, built[0])), dict(zip(man, built[5]))
    for name in ("dense_wq_b", "dense_wo_a", "dense_wo_b", "dense_w13", "dense_w2"):
        assert not torch.equal(a[name].view(torch.uint8), b[name].view(torch.uint8)), (
            name
        )
    for name in ("dense_wqa", "dense_xproj_csa", "dense_gate", "dense_hc_attn"):
        assert torch.equal(_bits(a[name]), _bits(b[name])), name


def _bits(t):
    return t.view(torch.uint8 if t.element_size() == 1 else torch.int16)


def test_stacks(tree, built):
    man = [e.name for e in KI.weight_manifest(_spec())]
    by = dict(zip(man, built[0]))
    L = tree.model.layers
    assert torch.equal(by["attn_sink"][1], L[1].attn.attn_sink)
    assert torch.count_nonzero(by["moe_bias"][0]) == 0  # hash layer
    assert torch.equal(by["moe_bias"][2], L[2].ffn.gate.e_score_correction_bias)
    ape = by["comp_ape"]
    assert ape.shape == (3, 128, 1024)
    assert torch.equal(ape[0, :, :512], L[0].attn.compressor.ape)
    assert torch.equal(ape[2, :4], L[2].attn.compressor.ape)
    assert torch.count_nonzero(ape[2, 4:]) == 0
    assert by["idx_comp_norm"].shape == (1, 128)
    assert torch.equal(
        by["comp_norm"][2], L[2].attn.compressor.norm.weight.to(torch.float32)
    )


def test_geometry_refusals():
    with pytest.raises(WM.Unsupported, match="do not divide"):
        WM.check_shardable(G, WM.Shard(0, 12))
    assert WM.lm_head_rows(15, 16, 129280) == (121200, 8080)
    assert WM.lm_head_rows(0, 16, 129280) == (0, 8080)


def test_nk_stored_compressors_pack_identically(tree, built):
    """A load path that skips `transpose_wkv_wgate` (dummy weights) leaves
    the compressors in [n, k]; the packed operands must not change."""
    flipped = make_tree()
    for src, dst in zip(tree.model.layers, flipped.model.layers):
        mods = [(src.attn.compressor, dst.attn.compressor)]
        if hasattr(src.attn, "indexer"):
            mods.append((src.attn.indexer.compressor, dst.attn.indexer.compressor))
        for a, b in mods:
            b.fused_wkv_wgate.weight = a.fused_wkv_wgate.weight.t().contiguous()
    names = [e.name for e in KI.weight_manifest(_spec())]
    ops = VA.build_weight_operands(flipped, _spec(), 0, host_run_jax)
    want = dict(zip(names, built[0]))
    got = dict(zip(names, ops))
    for name in ("dense_xproj_csa", "dense_xproj_hca"):
        assert torch.equal(_bits(got[name]), _bits(want[name])), name


def make_draft(g=G, n_draft=3, n_aux=3, e_local=E_LOCAL, seed=1):
    """A DSpark draft model tree (`draft_model.model.*`) with the served paths."""
    gen = torch.Generator().manual_seed(seed)
    glob_spec, layer_specs = WM.served_draft_spec(g, n_draft, n_aux, e_local)
    model = NS()
    layers = []
    for spec in layer_specs:
        layer = NS()
        for key, sv in spec.items():
            dtype = getattr(torch, sv.dtype)
            t = (torch.empty(sv.shape, dtype=dtype) if key in ("w13", "w2")
                 else _rand(sv.shape, dtype, gen))  # fmt: skip
            _set(layer, sv.path, t)
        layer.attn.compress_ratio = 1
        layers.append(layer)
    model.layers = layers
    for key, sv in glob_spec.items():
        shape = tuple(4096 if d == -1 else d for d in sv.shape)
        t = _rand(shape, getattr(torch, sv.dtype), gen)
        if sv.path.startswith("layers."):
            _set(layers[0], sv.path.split(".", 2)[2], t)
        else:
            _set(model, sv.path, t)
    return NS(model=model)


def test_draft_served_spec_reads_the_draft_tree():
    draft = make_draft()
    glob, layers, flags = VA.read_draft_served(draft, G, n_aux=3)
    assert glob["main_proj"].shape == (3 * 7168, 7168)
    assert glob["main_proj"] is draft.model.main_proj.weight
    assert glob["cos_sin"] is draft.model.layers[0].attn.rotary_emb.cos_sin_cache
    assert len(layers) == 3
    # SWA-only, past the hash layers: no compressor / indexer / tid2eid
    for t in layers:
        assert "comp" not in t and "idx_comp" not in t and "tid2eid" not in t
        assert t["gate_bias"].shape == (384,)
        assert t["wqa"].shape == (7168, 1536 + 512)
    assert (
        layers[2]["w13"] is draft.model.layers[2].ffn.experts.routed_experts.w13_weight
    )
    draft.model.layers[1].attn.compress_ratio = 4
    with pytest.raises(VA.Unsupported, match="draft layer 1"):
        VA.read_draft_served(draft, G, n_aux=3)


def _draft_spec():
    return KI.ModelSpec(
        hidden_size=7168, num_layers=3, compress_ratios=(1, 1, 1), dtype="bfloat16",
        q=6, mode="dp", axis_name="ep", num_ranks=R, aux_layers=(),
        vocab_size=HF.vocab_size, num_hash_layers=0, n_routed_experts=384,
    )  # fmt: skip


@pytest.mark.parametrize("rank", [0, 5])
def test_draft_operands_follow_the_draft_step_order(rank):
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_impl as SI

    draft = make_draft()
    spec = _draft_spec()
    cfg = WM.dense_config(spec)
    ops = VA.build_draft_weight_operands(draft, spec, G, 3, rank, host_run_jax)
    names = SI.draft_weight_names(spec, cfg)
    man = KI.draft_weight_manifest(spec, cfg)
    assert [e.name for e in man] == names and len(ops) == len(names)
    for e, t in zip(man, ops):
        assert WM.entry_matches(e, t.shape, str(t.dtype).removeprefix("torch.")), e
    by = dict(zip(names, ops))  # the AttnWeights cos_sin wins, as in the step
    m = draft.model
    assert by["embed"] is m.embed_tokens.weight
    cols = slice(rank * 7168 // R, (rank + 1) * 7168 // R)  # this rank's columns, bf16
    assert torch.equal(by["main_proj"], m.main_proj.weight[:, cols].to(torch.bfloat16))
    assert torch.equal(by["main_proj_scale"], m.main_proj.weight_scale[cols])
    assert torch.equal(
        by["cos_sin"], m.layers[0].attn.rotary_emb.cos_sin_cache.reshape(-1, 128)
    )
    assert ops[names.index("cos_sin")] is by["cos_sin"]  # one packed copy, twice
    assert by["w13_1"] is m.layers[1].ffn.experts.routed_experts.w13_weight
    bias = torch.stack([lyr.ffn.gate.e_score_correction_bias for lyr in m.layers])
    assert torch.equal(by["moe_bias"], bias)
    sink = torch.stack([lyr.attn.attn_sink for lyr in m.layers])
    assert torch.equal(by["attn_sink"], sink)
    # dense ring incl. wqa_ctx (the kv columns of wqa) and the SWA smalls
    nat = ref_natural(draft, rank, ratios=(1, 1, 1), gate_nk=False)
    for name in cfg.stream_names:
        want = dw.pack_stream(name, nat, cfg)
        got = t2j(by["dense_" + name])
        assert got.shape == want.shape, name
        np.testing.assert_array_equal(
            np.asarray(got.view(jnp.uint8 if got.dtype.itemsize == 1 else jnp.int16)),
            np.asarray(want.view(jnp.uint8 if want.dtype.itemsize == 1 else jnp.int16)),
            err_msg=name,
        )
    nat["idx_wq_b_scale"] = jnp.zeros((0, cfg.smalls_layout["idx_wq_b_scale"][1]))
    np.testing.assert_array_equal(
        np.asarray(t2j(by["dense_smalls"])), np.asarray(dw.pack_smalls(nat, cfg))
    )


def test_logits_mode_appends_this_ranks_lm_head_shard():
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import vocab_head as VH

    tree = make_tree()
    ops = VA.build_weight_operands(tree, _spec(), 5, host_run_jax, logits=True)
    man = KI.weight_manifest(_spec(), logits=True)
    assert [e.name for e in man][-1] == "lm_head" and len(ops) == len(man)
    want = VH.shard_lm_head(t2j(tree.lm_head.weight), 5, R)
    np.testing.assert_array_equal(
        np.asarray(t2j(ops[-1]).view(jnp.int16)), np.asarray(want.view(jnp.int16))
    )
    assert ops[-1].shape == (VH.vocab_shard(HF.vocab_size, R), 7168)


def test_draft_logits_mode_adds_norm_and_shares_the_lm_head():
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_impl as SI

    draft, spec = make_draft(), _draft_spec()
    cfg = WM.dense_config(spec)
    lm = torch.zeros((8192, 7168), dtype=torch.bfloat16)
    ops = VA.build_draft_weight_operands(
        draft, spec, G, 3, 0, host_run_jax, logits=True, shared={"lm_head": lm}
    )
    names = SI.draft_weight_names(spec, cfg, logits=True)
    assert [e.name for e in KI.draft_weight_manifest(spec, cfg, True)] == names
    by = dict(zip(names, ops))
    assert by["norm"] is draft.model.norm.weight and by["lm_head"] is lm
    assert names.index("norm") == len(SI.DRAFT_GLOBALS)


# ------------------------------------------------- vLLM TP shards (TP8 x DP2)

TP = 8


def tp_shard_tree(full, j, tp=TP):
    """TP rank j's served tree, sliced as vLLM's parallel linears shard
    (shapes measured on a TP8 server, j-7595b531): ColumnParallel wq_b / wo_a
    columns, RowParallel wo_b / down_proj rows (output-channel scales whole),
    MergedColumnParallel gate_up = [gate_j | up_j], attn_sink per local head,
    vocab-parallel embed / lm_head rows. Everything else is the same tensor."""
    D, OL, I, V = HF.head_dim, HF.o_lora_rank, HF.moe_intermediate_size, HF.vocab_size  # noqa: E741
    nh, og, si, vs = 128 // tp * D, 16 // tp * OL, I // tp, V // tp

    def cols(t, a, n):
        return t[..., a : a + n].clone()

    lm = NS(config=HF, model=NS(**vars(full.model)))
    lm.model.embed_tokens = NS(
        weight=torch.empty((vs, 7168), dtype=torch.bfloat16),
        shard_indices=NS(org_vocab_start_index=j * vs),
    )
    lm.lm_head = NS(weight=torch.empty((vs, 7168), dtype=torch.bfloat16))
    layers = []
    for src in full.model.layers:
        a, f = src.attn, src.ffn
        attn = NS(**vars(a))
        attn.wq_b = NS(
            weight=cols(a.wq_b.weight, j * nh, nh),
            weight_scale=cols(a.wq_b.weight_scale, j * nh, nh),
        )
        attn.wo_a = NS(
            weight=cols(a.wo_a.weight, j * og, og),
            weight_scale=cols(a.wo_a.weight_scale, j * og, og),
        )
        attn.wo_b = NS(
            weight=a.wo_b.weight[j * og : (j + 1) * og].clone(),
            weight_scale=a.wo_b.weight_scale,
        )
        attn.attn_sink = a.attn_sink[j * 16 : (j + 1) * 16].clone()
        gu, dn = f.shared_experts.gate_up_proj, f.shared_experts.down_proj

        def merged(t):
            return torch.cat([cols(t, j * si, si), cols(t, I + j * si, si)], -1)

        shared = NS(
            gate_up_proj=NS(weight=merged(gu.weight),
                            weight_scale=merged(gu.weight_scale)),
            down_proj=NS(weight=dn.weight[j * si : (j + 1) * si].clone(),
                         weight_scale=dn.weight_scale),
        )  # fmt: skip
        ffn = NS(**vars(f))
        ffn.shared_experts = shared
        layer = NS(**vars(src))
        layer.attn, layer.ffn = attn, ffn
        layers.append(layer)
    lm.model.layers = layers
    return lm


def _tp_spec():
    import dataclasses

    return dataclasses.replace(_spec(), attn_tp=TP)


@pytest.mark.parametrize("g, j", [(0, 0), (1, 2)])
def test_tp_shard_operands_are_the_head_block_operands(tree, built, g, j):
    """Mesh rank g*8 + j built from TP shard j carries exactly the operands
    DP16 rank hb = j*G + g builds from the whole tensors (step_kernel's
    head_block), except attn_sink (its shard's 16 heads, -inf padded) and the
    embed (the vocab shard)."""
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_kernel as SK

    r, hb = g * TP + j, j * (R // TP) + g
    assert SK.head_block(r, R, TP) == hb and hb in built
    shard = tp_shard_tree(tree, j)
    ops = VA.build_weight_operands(shard, _tp_spec(), r, host_run_jax)
    man = KI.weight_manifest(_tp_spec())
    assert [e.name for e in man] == [e.name for e in KI.weight_manifest(_spec())]
    want = dict(zip([e.name for e in man], built[hb]))
    for e, t in zip(man, ops):
        assert WM.entry_matches(e, t.shape, str(t.dtype).removeprefix("torch.")), e
        if e.name == "attn_sink":
            for li, layer in enumerate(tree.model.layers):
                assert torch.equal(
                    t[li, :16], layer.attn.attn_sink[16 * j : 16 * j + 16]
                )
                assert bool(torch.all(t[li, 16:] == -torch.inf))
        elif e.name == "embed":
            assert t is shard.model.embed_tokens.weight
        else:
            assert torch.equal(_bits(t), _bits(want[e.name])), e.name


def test_tp_geometry_and_shard(tree):
    gl = WM.tp_local(G, TP)
    assert (gl.n_heads, gl.o_groups, gl.shared_inter, gl.moe_inter) == (
        16,
        2,
        384,
        3072,
    )
    assert WM.head_shard(13, R, TP) == WM.Shard(1, 2, mesh_rank=13)
    assert WM.head_shard(13, R, TP).col_rank == 13
    assert WM.head_shard(13, R) == WM.Shard(13, R)
    with pytest.raises(WM.Unsupported, match="does not divide"):
        WM.tp_local(G, 3)
    with pytest.raises(VA.Unsupported, match="logits mode"):
        VA.build_weight_operands(tp_shard_tree(tree, 0), _tp_spec(), 0,
                                 host_run_jax, logits=True)  # fmt: skip
