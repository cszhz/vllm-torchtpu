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
"""StepGeometry.attn_pipeline (VLLM_TPU_DSV4_MK_ATTN_PIPELINE), CPU interpret:
the phase with request r + 1's first pages prefetched during request r gives
outputs and caches bit-identical to the unpipelined phase, with 1, 2 and 4
requests, for one rank and for an 8-rank attention group (where the flag
also splits the CSA scan + top-k). The race detector (DMAs at their start)
finds no race the unpipelined phase does not have (that one: kv_writes issues
a compressor window's HBM reads together with this step's state-row writes
into the same rows, whose read values it replaces from VMEM); prefetching
into the slots request r still reads, or keeping prefetched pages the
request's own stores changed, fails the check."""

import os
import re
import sys

import jax

jax.config.update("jax_num_cpu_devices", 8)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from jax._src.pallas.mosaic.interpret import interpret_pallas_call  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
import attn_phase_harness as H  # noqa: E402

from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_phase as AP  # noqa: E402

KINDS = ("hca", "csa", "hca", "csa")
LAYER = {"hca": 2, "csa": 3}
# seq_len incl. the 7 new rows: 1030 closes an HCA row (position 1023) inside
# HCA block 0; 8000 writes indexer rows into scan block 0 with more rows than
# the top-k keeps (3000: every row kept); 9000 scans 2 blocks; 300 is a short
# window.
SEQS = {1: [9000], 2: [1030, 8000], 4: [1030, 8000, 3000, 300]}


def _same(a, b, n_active=None, heads=16):
    """Caches, and the output rows the phase writes (active requests, 7 rows of
    `heads` query heads), bit-identical."""
    ca, oa = a
    cb, ob = b
    for i, (x, y) in enumerate(zip(ca, cb)):
        assert np.array_equal(np.asarray(x), np.asarray(y)), f"cache {i} differs"
    n = slice(None) if n_active is None else slice(0, n_active)
    oa = np.asarray(oa, np.float32)[n, : 7 * heads]
    ob = np.asarray(ob, np.float32)[n, : 7 * heads]
    assert np.array_equal(oa, ob), np.abs(oa - ob).max()


def _case(kind, n_req, seed=0):
    rng = np.random.default_rng(100 + 7 * n_req + seed)
    seqs = SEQS[n_req]
    ls = H.build(rng, KINDS, seqs, max(seqs) + 1024)
    return ls, H.make_inputs(rng, ls, kind)


def _races():
    r = interpret_pallas_call.races
    return bool(r is not None and r.races_found)


RACE_SITE = re.compile(r" from \(\d+, \d+\), \S+ \(([^()]*)\)")


def _race_sites(text):
    """{(function of access 1, function of access 2)} of the detector's reports."""
    out = set()
    for block in text.split("RACE DETECTED")[1:]:
        fns = RACE_SITE.findall(block)
        out.add(tuple(sorted(fns[:2])))
    return out


@pytest.mark.parametrize("n_req", [1, 2, 4])
@pytest.mark.parametrize("kind", ["hca", "csa"])
def test_pipeline_bitexact(kind, n_req):
    ls, inp = _case(kind, n_req)
    off = H.run_phase(ls, kind, LAYER[kind], inp, attn_heads=16, h0=32)
    on = H.run_phase(ls, kind, LAYER[kind], inp, attn_heads=16, h0=32, pipeline=True)
    _same(on, off)


@pytest.mark.parametrize("kind", ["hca", "csa"])
def test_pipeline_partial_batch(kind):
    """num_req = 3 of 4 slots: the last request issues no prefetch."""
    ls, inp = _case(kind, 4)
    off = H.run_phase(ls, kind, LAYER[kind], inp, num_req=3)
    on = H.run_phase(ls, kind, LAYER[kind], inp, num_req=3, pipeline=True)
    _same(on, off, n_active=3, heads=128)


@pytest.mark.parametrize("n_req", [2, 4])
@pytest.mark.parametrize("kind", ["hca", "csa"])
def test_pipeline_no_races(kind, n_req, capsys):
    ls, inp = _case(kind, n_req)
    off = H.run_phase(ls, kind, LAYER[kind], inp, attn_heads=16, races=True)
    off_sites = _race_sites(capsys.readouterr().out)
    on = H.run_phase(
        ls, kind, LAYER[kind], inp, attn_heads=16, pipeline=True, races=True
    )
    on_sites = _race_sites(capsys.readouterr().out)
    with capsys.disabled():
        print(f"\nrace sites off={sorted(off_sites)} on={sorted(on_sites)}")
    known = ("_Phase._starts.<locals>.go_loop.<locals>.one",) * 2
    assert off_sites <= {known}, off_sites
    assert on_sites <= off_sites, on_sites - off_sites
    _same(on, off)


@pytest.mark.parametrize("fault", ["same_slot", "no_refetch"])
@pytest.mark.parametrize("kind", ["hca", "csa"])
def test_positive_controls(kind, fault, capsys):
    ls, inp = _case(kind, 4)
    off = H.run_phase(ls, kind, LAYER[kind], inp, attn_heads=16, races=True)
    off_sites = _race_sites(capsys.readouterr().out)
    AP.PF_FAULT = fault
    try:
        bad = H.run_phase(
            ls, kind, LAYER[kind], inp, attn_heads=16, pipeline=True, races=True
        )
    finally:
        AP.PF_FAULT = ""
    new_sites = _race_sites(capsys.readouterr().out) - off_sites
    races = bool(new_sites)
    differ = False
    try:
        _same(bad, off)
    except AssertionError:
        differ = True
    with capsys.disabled():
        print(
            f"\nPF_FAULT={fault} kind={kind}: new race sites={sorted(new_sites)} "
            f"outputs_or_caches_differ={differ}"
        )
    assert races or differ, f"{fault}: not caught"


@pytest.mark.parametrize("n_req", [1, 2, 4])
@pytest.mark.parametrize("kind", ["hca", "csa"])
def test_group_pipeline_bitexact(kind, n_req):
    """8-rank group (attn_tp 8, 16 heads a rank): flag on (pipeline + split
    scan / top-k) vs flag off (every rank selects alone, no prefetch)."""
    ls, inp = _case(kind, n_req, seed=1)
    if n_req == 1:  # every rank scores columns
        rng = np.random.default_rng(7)
        ls = H.build(rng, KINDS, [70000], 71680)
        inp = H.make_inputs(rng, ls, kind)
    on = H.run_group(ls, kind, LAYER[kind], inp, pipeline=True)
    for jj, res in enumerate(on):
        off = H.run_phase(ls, kind, LAYER[kind], inp, attn_heads=16, h0=16 * jj)
        _same(res, off)


@pytest.mark.parametrize("attn_tp", [1, 8])
def test_kernel_config_reads_the_env(monkeypatch, attn_tp):
    """VLLM_TPU_DSV4_MK_ATTN_PIPELINE=1: pipeline, plus the split when attn_tp > 1;
    unset or 0: neither (the program of the tree before the flag)."""
    import dataclasses

    import test_step_impl as TSI

    from vllm_torchtpu.kernels.deepseek_v4_megakernel import kernel_interface as KI
    from vllm_torchtpu.kernels.deepseek_v4_megakernel import step_impl as SI

    cfg = TSI._cfg()
    model = dataclasses.replace(cfg.model, num_ranks=16, attn_tp=attn_tp)
    cfg = dataclasses.replace(cfg, model=model)
    bucket = KI.Bucket(q=7, rows=16, reqs=1)
    for value, want in (("1", True), ("0", False)):
        monkeypatch.setenv("VLLM_TPU_DSV4_MK_ATTN_PIPELINE", value)
        geom = SI.kernel_config(cfg, bucket).geometry()
        assert geom.attn_pipeline is want and AP.dims_of(geom).pf is want
        assert geom.attn_split is (want and attn_tp > 1)
    monkeypatch.delenv("VLLM_TPU_DSV4_MK_ATTN_PIPELINE")
    geom = SI.kernel_config(cfg, bucket).geometry()
    assert geom.attn_pipeline is False and geom.attn_split is False
