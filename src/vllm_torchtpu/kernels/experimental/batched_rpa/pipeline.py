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

"""Manual triple buffering pipeline scheduling loop for RPA."""

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

from vllm_torchtpu.kernels.experimental.batched_rpa import configs


def _rem(val: int | jax.Array, n: int) -> int | jax.Array:
    """Computes remainder modulo n for both Python ints and JAX arrays/tracers."""
    if isinstance(val, int):
        return val % n
    return jax.lax.rem(val.astype(jnp.int32), n)


def run_manual_triple_buffer(
    total_steps: int | jax.Array,
    compute_fn: Callable[..., Any],
    q_bref: Any,
    o_bref: Any,
    kv_bref: Any,
    lse_bref: Any | None = None,
) -> None:
    """Executes an explicit triple-buffered pipeline over `total_steps`.

    Inputs rotate over `configs.NUM_INPUT_SLOTS` physical slots:
    - comp_slot = step % NUM_INPUT_SLOTS       (slot being computed on)
    - comm_slot = (step + 2) % NUM_INPUT_SLOTS (slot fetched two steps ahead)

    Outputs rotate independently over `configs.NUM_OUTPUT_SLOTS` slots:
    - out_slot = step % NUM_OUTPUT_SLOTS            (slot being written)
    - wait_out_slot = (step - 2) % NUM_OUTPUT_SLOTS (slot whose writeback is
    awaited before it is reused)

    Args:
      total_steps: Total number of pipeline steps to execute.
      compute_fn: Callable of the main kernel function.
      q_bref: Fetch-only ref for the Q blocks.
      o_bref: Writeback-only ref for the attention output.
      kv_bref: Input/output ref for the KV cache.
      lse_bref: Optional writeback-only ref for the log-sum-exp values.
    """
    with jax.named_scope("ep_initialize"):
        q_bref.copy_in(0, 0, total_steps)
        kv_bref.copy_in(0, 0, total_steps)
        q_bref.copy_in(1, 1, total_steps)
        kv_bref.copy_in(1, 1, total_steps)

    @pl.loop(0, total_steps)
    def steady_state(step):
        in_slots = configs.NUM_INPUT_SLOTS
        out_slots = configs.NUM_OUTPUT_SLOTS
        comp_slot = _rem(step, in_slots)
        comm_slot = _rem(step + 2, in_slots)
        out_slot = _rem(step, out_slots)
        wait_out_slot = _rem(step - 2, out_slots)

        with jax.named_scope("wait_in_q_kv"):
            q_bref.wait_in(step, comp_slot, total_steps)
            kv_bref.wait_in(step, comp_slot, total_steps)

        with jax.named_scope("wait_out_o"):
            o_bref.wait_out(step - 2, wait_out_slot, total_steps)
            if lse_bref is not None:
                lse_bref.wait_out(step - 2, wait_out_slot, total_steps)

        with jax.named_scope("ep_run_kernel"):

            def start_next_fetch():
                q_bref.copy_in(step + 2, comm_slot, total_steps)
                kv_bref.copy_in(step + 2, comm_slot, total_steps)

            def start_kv_writeback():
                kv_bref.copy_out(step, comp_slot, total_steps)

            compute_fn(
                step,
                comp_slot,
                total_steps,
                out_slot=out_slot,
                start_next_fetch=start_next_fetch,
                start_kv_writeback=start_kv_writeback,
            )

            with jax.named_scope("wait_out_kv"):
                kv_bref.wait_out(step, comp_slot, total_steps)

            o_bref.copy_out(step, out_slot, total_steps)
            if lse_bref is not None:
                lse_bref.copy_out(step, out_slot, total_steps)

    with jax.named_scope("epilogue"):
        out_slots = configs.NUM_OUTPUT_SLOTS
        last_two = total_steps - 2
        last_one = total_steps - 1
        o_bref.wait_out(last_two, _rem(last_two, out_slots), total_steps)
        if lse_bref is not None:
            lse_bref.wait_out(last_two, _rem(last_two, out_slots), total_steps)
        o_bref.wait_out(last_one, _rem(last_one, out_slots), total_steps)
        if lse_bref is not None:
            lse_bref.wait_out(last_one, _rem(last_one, out_slots), total_steps)


def run_skewed_pipeline(
    total_steps: int | jax.Array,
    stage_a: Callable[..., Any],
    stage_b: Callable[..., Any],
    q_bref: Any,
    o_bref: Any,
    kv_bref: Any,
    *,
    in_slots: int,
) -> None:
    """Runs the step loop with the softmax and PV of step s-1 next to the QK of step s.

    Iteration s holds `stage_a(s)` (stitch, writeback issue, fetch issue, K
    loads, QK and row max, scores into scratch) and `stage_b(s-1)` (softmax,
    V loads, PV, output store) in one basic block, so the two have no data
    dependency and the scheduler overlaps them. Stage A of step 0 runs before
    the loop and stage B of the last step after it. `in_slots` must be at
    least 4: iteration s computes on slot s, reads V from slot s-1 and fetches
    step s+2.

    Output slots rotate as in `run_manual_triple_buffer`; slot (s-1) % 2 is
    written in iteration s, so the writeback awaited before it is step s-3's.
    """
    assert in_slots >= 4
    out_slots = configs.NUM_OUTPUT_SLOTS

    def _fetch(step):
        def start_next_fetch():
            comm_slot = _rem(step + 2, in_slots)
            q_bref.copy_in(step + 2, comm_slot, total_steps)
            kv_bref.copy_in(step + 2, comm_slot, total_steps)

        return start_next_fetch

    def _writeback(step):
        def start_kv_writeback():
            kv_bref.copy_out(step, _rem(step, in_slots), total_steps)

        return start_kv_writeback

    with jax.named_scope("ep_initialize"):
        q_bref.copy_in(0, 0, total_steps)
        kv_bref.copy_in(0, 0, total_steps)
        q_bref.copy_in(1, 1, total_steps)
        kv_bref.copy_in(1, 1, total_steps)

    with jax.named_scope("ep_prologue"):
        q_bref.wait_in(0, 0, total_steps)
        kv_bref.wait_in(0, 0, total_steps)
        stage_a(0, 0, start_next_fetch=_fetch(0), start_kv_writeback=_writeback(0))
        kv_bref.wait_out(0, 0, total_steps)

    @pl.loop(1, total_steps)
    def steady_state(step):
        comp_slot = _rem(step, in_slots)
        prev_slot = _rem(step - 1, in_slots)
        out_prev = _rem(step - 1, out_slots)

        with jax.named_scope("wait_in_q_kv"):
            q_bref.wait_in(step, comp_slot, total_steps)
            kv_bref.wait_in(step, comp_slot, total_steps)

        with jax.named_scope("wait_out_o"):
            o_bref.wait_out(step - 3, _rem(step - 3, out_slots), total_steps)

        with jax.named_scope("ep_run_kernel"):
            stage_a(
                step,
                comp_slot,
                start_next_fetch=_fetch(step),
                start_kv_writeback=_writeback(step),
            )
            stage_b(step - 1, prev_slot, out_slot=out_prev)

            with jax.named_scope("wait_out_kv"):
                kv_bref.wait_out(step, comp_slot, total_steps)

            o_bref.copy_out(step - 1, out_prev, total_steps)

    with jax.named_scope("epilogue"):
        last = total_steps - 1
        o_bref.wait_out(last - 2, _rem(last - 2, out_slots), total_steps)
        stage_b(last, _rem(last, in_slots), out_slot=_rem(last, out_slots))
        o_bref.copy_out(last, _rem(last, out_slots), total_steps)
        o_bref.wait_out(last - 1, _rem(last - 1, out_slots), total_steps)
        o_bref.wait_out(last, _rem(last, out_slots), total_steps)


def run_folded_pipeline(
    total_steps: int | jax.Array,
    prepare: Callable[..., Any],
    prepare_wait: Callable[..., Any],
    stage_a: Callable[..., Any],
    stage_b: Callable[..., Any],
    q_bref: Any,
    o_bref: Any,
    kv_bref: Any,
    *,
    in_slots: int,
) -> None:
    """`run_skewed_pipeline` with the stitch of step s+1 inside iteration s.

    The fetch runs three steps ahead, so the block of step s+1 has arrived when
    iteration s starts; `prepare(s+1)` stitches it, starts its writeback and
    its lane zeroing, and shares the basic block with `stage_a(s)` and
    `stage_b(s-1)`, which read other slots. Only the prologue, which stitches
    step 0 and then loads it, keeps a region boundary. `in_slots` must be at
    least 5: slots s-1 (V), s (K), s+1 (stitch), s+2 and s+3 (in flight).
    """
    assert in_slots >= 5
    out_slots = configs.NUM_OUTPUT_SLOTS
    last = total_steps - 1

    def _fetch(step):
        def start_next_fetch():
            comm_slot = _rem(step + 3, in_slots)
            q_bref.copy_in(step + 3, comm_slot, total_steps)
            kv_bref.copy_in(step + 3, comm_slot, total_steps)

        return start_next_fetch

    def _writeback(step):
        def start_kv_writeback():
            kv_bref.copy_out(step, _rem(step, in_slots), total_steps)

        return start_kv_writeback

    def _prepare(step):
        # Past the last step the stitch repeats the last step's metadata on an
        # idle slot; its writeback is a no-op (`_step_guard`).
        meta_step = jnp.minimum(step, last)
        slot = _rem(step, in_slots)
        prepare(meta_step, slot, start_kv_writeback=_writeback(step))

    def _prepare_wait(step):
        prepare_wait(jnp.minimum(step, last), _rem(step, in_slots))

    with jax.named_scope("ep_initialize"):
        for i in range(3):
            q_bref.copy_in(i, i, total_steps)
            kv_bref.copy_in(i, i, total_steps)

    with jax.named_scope("ep_prologue"):
        q_bref.wait_in(0, 0, total_steps)
        kv_bref.wait_in(0, 0, total_steps)

        @pl.when(total_steps > 0)
        def _stitch_first():
            _prepare(0)

        _prepare_wait(0)
        kv_bref.wait_out(0, 0, total_steps)
        q_bref.wait_in(1, 1, total_steps)
        kv_bref.wait_in(1, 1, total_steps)
        _prepare(1)
        stage_a(0, 0, start_next_fetch=_fetch(0), start_kv_writeback=None)
        _prepare_wait(1)
        kv_bref.wait_out(1, 1, total_steps)

    @pl.loop(1, total_steps)
    def steady_state(step):
        comp_slot = _rem(step, in_slots)
        prev_slot = _rem(step - 1, in_slots)
        next_slot = _rem(step + 1, in_slots)
        out_prev = _rem(step - 1, out_slots)

        with jax.named_scope("wait_in_q_kv"):
            q_bref.wait_in(step + 1, next_slot, total_steps)
            kv_bref.wait_in(step + 1, next_slot, total_steps)

        with jax.named_scope("wait_out_o"):
            o_bref.wait_out(step - 3, _rem(step - 3, out_slots), total_steps)

        with jax.named_scope("ep_run_kernel"):
            _prepare(step + 1)
            stage_a(
                step, comp_slot, start_next_fetch=_fetch(step), start_kv_writeback=None
            )
            stage_b(step - 1, prev_slot, out_slot=out_prev)

            with jax.named_scope("wait_out_kv"):
                _prepare_wait(step + 1)
                kv_bref.wait_out(step + 1, next_slot, total_steps)

            o_bref.copy_out(step - 1, out_prev, total_steps)

    with jax.named_scope("epilogue"):
        o_bref.wait_out(last - 2, _rem(last - 2, out_slots), total_steps)
        stage_b(last, _rem(last, in_slots), out_slot=_rem(last, out_slots))
        o_bref.copy_out(last, _rem(last, out_slots), total_steps)
        o_bref.wait_out(last - 1, _rem(last - 1, out_slots), total_steps)
        o_bref.wait_out(last, _rem(last, out_slots), total_steps)
