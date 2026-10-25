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
"""SWA + HCA attention of DeepSeek-V4 verify windows, as a megakernel step.

Per request, one online softmax runs over the sliding window (`VerifyWindow`)
and then the HCA compressed rows, streamed in blocks of ``blk_pages`` pages
through a double-buffered VMEM ring that runs ahead across requests, and
finishes with the attention sink: the union the serving path builds from two
kernels and a partial merge.

`swa_hca_attend` works on refs and is meant to be called inside a larger
kernel; `swa_hca_attention` wraps it in a standalone `pallas_call`.
"""

import dataclasses
import functools
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_core as core
from vllm_torchtpu.kernels.deepseek_v4_megakernel import attn_layout as L


@dataclasses.dataclass(frozen=True)
class HcaConfig:
    max_req: int
    hca_pages_per_req: int  # row length of the HCA page table
    ratio: int = 128
    hca_page: int = 8  # compressed rows per HCA page
    blk_pages: int = 32  # HCA pages per KV block
    enabled: bool = True  # False: SWA-only layer

    @property
    def blk_rows(self) -> int:
        return self.blk_pages * self.hca_page


class HcaScratch(NamedTuple):
    buf: object  # int32 [2, 2 * blk_rows, 128]
    sems: object  # DMA [2]


def hca_scratch_shapes(hcfg: HcaConfig) -> HcaScratch:
    return HcaScratch(
        buf=pltpu.VMEM((2, 2 * hcfg.blk_rows, L.LANES), jnp.int32),
        sems=pltpu.SemaphoreType.DMA((2,)),
    )


def swa_hca_attend(
    wcfg: core.WindowConfig,
    hcfg: HcaConfig,
    *,
    num_req,  # scalar, <= hcfg.max_req
    hca_pt,  # SMEM int32 [max_req * hca_pages_per_req]
    hca_hbm,  # int32 [N_h, 2 * hca_page, 128]
    win: core.VerifyWindow,
    hscr: HcaScratch,
):
    def n_c(r):
        return win.seq_lens[r] // hcfg.ratio

    def hca_start(r, b, slot):
        """Issues block b of request r: always blk_pages static-size page copies,
        the tail clamped to the last live page (re-read, masked), so the burst
        is branch-free and interleaves with the compute around it."""
        last = jnp.maximum(core.cdiv(n_c(r), hcfg.hca_page) - 1, 0)
        base = r * hcfg.hca_pages_per_req
        rows = 2 * hcfg.hca_page
        for p in range(hcfg.blk_pages):
            pg = jnp.minimum(b * hcfg.blk_pages + p, last)
            pltpu.make_async_copy(
                hca_hbm.at[jnp.clip(hca_pt[base + pg], 0, hca_hbm.shape[0] - 1)],
                hscr.buf.at[slot, pl.ds(p * rows, rows)],
                hscr.sems.at[slot],
            ).start()

    def hca_wait(r, b, slot):
        blk = hscr.buf.at[slot]
        pltpu.make_async_copy(blk, blk, hscr.sems.at[slot]).wait()

    def hca_step(r, b, slot):
        # Rows past n_c hold finite cache data (the tail re-reads live pages);
        # their scores are masked.
        B = hcfg.blk_rows
        kv, kv_t = core.decode_bf16_tokens(hscr.buf.at[slot], B, wcfg.transposed)
        vis = b * B + win.key_index(B) < (win.qpos(r) + 1) // hcfg.ratio
        win.update(r, kv, kv_t, vis, first=False)

    if not hcfg.enabled:
        def req_body(r, c):
            win.swa_dma(r, 0, wait=False)
            win.swa_dma(r, 0, wait=True)
            win.swa_step(r, 0)
            win.finalize(r)
            return c

        lax.fori_loop(0, num_req, req_body, 0)
        return

    core.run_window_stream(
        win, num_req=num_req,
        n_blocks=lambda r: jnp.maximum(1, core.cdiv(n_c(r), hcfg.blk_rows)),
        start=hca_start, wait=hca_wait, step=hca_step,
    )


def _kernel(wcfg, hcfg, nreq_ref, seq_ref, swa_pt_ref, hca_pt_ref, q_ref, nkv_ref,
            sink_ref, swa_hbm, hca_hbm, out_ref, *scratch):
    if wcfg.write_swa:  # the aliased output is the cache the rows go to
        swa_hbm, scratch = scratch[0], scratch[1:]
    nw = len(core.WindowScratch._fields)
    win = core.VerifyWindow(
        wcfg, seq_lens=seq_ref, swa_pt=swa_pt_ref, q_ref=q_ref, new_kv_ref=nkv_ref,
        sink_ref=sink_ref, swa_hbm=swa_hbm, out_ref=out_ref,
        scr=core.WindowScratch(*scratch[:nw]),
    )
    swa_hca_attend(wcfg, hcfg, num_req=nreq_ref[0], hca_pt=hca_pt_ref, hca_hbm=hca_hbm,
                   win=win, hscr=HcaScratch(*scratch[nw:]))


def swa_hca_attention(
    q,  # bf16 [R, q_len, H, 512]
    new_kv,  # bf16 [R, q_len, 512]
    swa_words,  # int32 [N, 2 * swa_page, 128]
    swa_table,  # int32 [R, P_swa]
    hca_words,  # int32 [N_h, 2 * hca_page, 128]
    hca_table,  # int32 [R, P_hca]
    seq_lens,  # int32 [R]
    sinks,  # f32 [H]
    *,
    sm_scale: float,
    blk_pages: int | None = None,
    with_hca: bool = True,
    write_swa: bool = False,
    pv_dtype=jnp.float32,
    interpret=False,
    vmem_limit_bytes: int = 60 * 1024 * 1024,
):
    """Standalone wrapper: bf16 ``[R, q_len, H, 512]`` (and, with `write_swa`, the
    SWA cache with the new rows stored, aliased to `swa_words`)."""
    R, T, H, D = q.shape
    assert D == L.HEAD_DIM
    wcfg = core.WindowConfig(heads=H, sm_scale=sm_scale, q_len=T,
                             swa_page=swa_words.shape[1] // 2, pv_dtype=pv_dtype,
                             write_swa=write_swa, round_window_acc=with_hca)
    hca_page = hca_words.shape[1] // 2
    if blk_pages is None:  # one block per request up to 1024 compressed rows
        blk_pages = min(128, hca_table.shape[1])
    hcfg = HcaConfig(max_req=R, hca_pages_per_req=hca_table.shape[1],
                     hca_page=hca_page, blk_pages=blk_pages, enabled=with_hca)
    swa_pt = core.swa_window_pages(
        seq_lens, swa_table, q_len=T, window=wcfg.window, swa_page=wcfg.swa_page
    )
    nkv = jnp.pad(new_kv, ((0, 0), (0, wcfg.new_rows - T), (0, 0)))
    pad = wcfg.rows_pad - wcfg.rows
    q_rows = jnp.pad(q.reshape(R, wcfg.rows, D), ((0, 0), (0, pad), (0, 0)))
    sink_row = jnp.pad(jnp.tile(sinks.astype(jnp.float32), T), (0, pad)).reshape(1, -1)
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    vmem = pl.BlockSpec(memory_space=pltpu.VMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    out_shape = [jax.ShapeDtypeStruct((R, wcfg.rows_pad, D), q.dtype)]
    out_specs = [vmem]
    if write_swa:
        out_shape.append(jax.ShapeDtypeStruct(swa_words.shape, swa_words.dtype))
        out_specs.append(hbm)
    out = pl.pallas_call(
        functools.partial(_kernel, wcfg, hcfg),
        out_shape=out_shape,
        in_specs=[smem, smem, smem, smem, vmem, vmem, vmem, hbm, hbm],
        out_specs=out_specs,
        input_output_aliases={7: 1} if write_swa else {},
        scratch_shapes=[*core.window_scratch_shapes(wcfg), *hca_scratch_shapes(hcfg)],
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=vmem_limit_bytes,
                                             disable_bounds_checks=True),
        interpret=pltpu.InterpretParams() if interpret else False,
        name=f"mk_swa_hca_h{H}_r{R}",
    )(
        jnp.array([R], jnp.int32),
        seq_lens.astype(jnp.int32),
        swa_pt.reshape(-1),
        hca_table.astype(jnp.int32).reshape(-1),
        q_rows,
        nkv,
        sink_row,
        swa_words,
        hca_words,
    )
    att = out[0][:, : wcfg.rows].reshape(R, T, H, D)
    return (att, out[1]) if write_swa else att
