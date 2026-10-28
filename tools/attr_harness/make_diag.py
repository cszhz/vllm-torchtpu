# ruff: noqa: E501, E731, E741, SIM105  (diagnostic branch, not for merge)
"""Build the DIAGNOSTIC copies of qa_rope / qb_rope / compress_rope: jax.named_scope stages only.

usage: make_diag.py SRC_KERNEL_DIR DST_KERNEL_DIR
Reads the three kernel modules from SRC (the committed worktree, read-only) and writes patched copies into DST (staging).
Rules (semantic-tracing.md): no wait is added, nothing is reordered, no buffer lifetime changes. Every patch either wraps
existing statements in `with jax.named_scope(...)` or splits one statement into two in the same evaluation order
(matmul value computed first, then scale/round). Each anchor must match exactly once or the script aborts.
Scope names: attr_a/*, attr_b/*, attr_c/* (searchable; kernel-name prefix of the device event is not assumed).
"""

import sys
from pathlib import Path


def _line_bounds(text, a, b):
    s = text.rfind("\n", 0, a) + 1
    e = text.find("\n", b)
    return s, (len(text) if e < 0 else e + 1)


def wrap(text, first, last=None, scope=None, dynamic=False):
    """Wrap the whole lines from `first` through `last` (inclusive) in a named_scope; reindent by 4."""
    assert text.count(first) == 1, (
        f"anchor not unique/found ({text.count(first)}): {first[:70]!r}"
    )
    a = text.index(first)
    if last is None:
        b = a + len(first)
    else:
        assert text.count(last, a) >= 1, f"end anchor not found: {last[:70]!r}"
        b = text.index(last, a) + len(last)
    s, e = _line_bounds(text, a, b)
    lines = text[s:e].splitlines(keepends=True)
    indent = lines[0][: len(lines[0]) - len(lines[0].lstrip())]
    arg = scope if dynamic else repr(scope)
    block = (
        indent
        + f"with jax.named_scope({arg}):\n"
        + "".join(("    " + l) if l.strip() else l for l in lines)
    )
    return text[:s] + block + text[e:]


def replace_once(text, old, new):
    assert text.count(old) == 1, (
        f"replace anchor not unique/found ({text.count(old)}): {old[:70]!r}"
    )
    return text.replace(old, new)


def add_body_wrapper(text, def_line, inner_name, outer_name, scope):
    text = replace_once(text, def_line, def_line.replace(outer_name, inner_name, 1))
    return (
        text.rstrip("\n")
        + f"""


def {outer_name}(*args, **kwargs):
    with jax.named_scope({scope!r}):
        return {inner_name}(*args, **kwargs)
"""
    )


def patch_qa(t):
    t = add_body_wrapper(
        t,
        "def _qa_rope_kernel(\n",
        "_qa_rope_kernel_inner",
        "_qa_rope_kernel",
        "attr_a/body",
    )
    t = wrap(
        t,
        "        for c in range(chunks):\n            weight_copy(c).start(priority=1)",
        None,
        "attr_a/weight_start",
    )
    t = wrap(
        t,
        "    encoded, row_scale = _quantize_rows(x_ref[...])",
        None,
        "attr_a/x_quantize",
    )
    t = replace_once(
        t,
        """        for c in range(chunks):
            columns = pl.ds(c * chunk, chunk)
            weight_copy(c).wait()
            project(
                columns,
                jnp.dot(
                    encoded, weight_ref[:, columns], preferred_element_type=jnp.float32
                ),
            )
""",
        """        for c in range(chunks):
            columns = pl.ds(c * chunk, chunk)
            with jax.named_scope('attr_a/weight_wait'):
                weight_copy(c).wait()
            with jax.named_scope('attr_a/matmul'):
                product = jnp.dot(
                    encoded, weight_ref[:, columns], preferred_element_type=jnp.float32
                )
            with jax.named_scope('attr_a/scale_round'):
                project(columns, product)
""",
    )
    t = replace_once(
        t,
        """        project(
            pl.ds(0, width),
            jnp.dot(encoded, weight_ref[...], preferred_element_type=jnp.float32),
        )
""",
        """        with jax.named_scope('attr_a/matmul'):
            product = jnp.dot(encoded, weight_ref[...], preferred_element_type=jnp.float32)
        with jax.named_scope('attr_a/scale_round'):
            project(pl.ds(0, width), product)
""",
    )
    t = wrap(
        t,
        "        qr_out_ref[...] = native_rms_norm(",
        "            projected_ref[:, :rank], gamma_qr_ref[...][None, :], eps\n        )",
        "attr_a/qr_norm",
    )
    t = wrap(
        t,
        "    kv = native_rms_norm(projected_ref[:, kv_start:], gamma_kv_ref[...][None, :], eps)",
        None,
        "attr_a/kv_norm",
    )
    t = wrap(
        t,
        "    if kv_out_ref.shape[-1] > _LANE:",
        "    kv_out_ref[:, -_LANE:] = (tail * cos + partner * sin).astype(jnp.bfloat16)",
        "attr_a/kv_rope_store",
    )
    return t


def patch_qb(t):
    t = add_body_wrapper(
        t,
        "def _qb_rope_kernel(\n",
        "_qb_rope_kernel_inner",
        "_qb_rope_kernel",
        "attr_b/body",
    )
    t = wrap(
        t,
        "        jax.lax.fori_loop(0, pairs, issue, None)",
        None,
        "attr_b/weight_issue_prologue",
    )
    t = wrap(
        t,
        "        jax.lax.fori_loop(0, tb // quant_fragment, quantize_fragment, None)",
        None,
        "attr_b/prepare_quant",
    )
    t = wrap(
        t,
        "        cos_lanes, sin_lanes = rope_gather.coefficients(rope_rows_ref[...])",
        "        coeff_ref[1, pl.ds(token_start, tb), :] = sin_lanes",
        "attr_b/prepare_coeff",
    )
    t = wrap(t, "                copy(g - slots).wait()", None, "attr_b/reclaim_wait")
    t = wrap(
        t,
        "                weight_copy(window + 1, pair).start(priority=1)",
        None,
        "attr_b/weight_next_start",
    )
    t = wrap(
        t, "            weight_copy(window, pair).wait()", None, "attr_b/weight_wait"
    )
    t = wrap(
        t,
        "        for half in range(2):\n            products[parity][half] = jnp.dot(",
        "                preferred_element_type=jnp.float32,\n            )",
        "attr_b/matmul_pair",
    )
    t = wrap(
        t,
        "        for f in range(tb // post_fragment):",
        "                [low_tail, high_tail], packed_dtype=jnp.bfloat16\n            )",
        "attr_b/post_pair",
    )
    t = wrap(t, "            copy(g).start()", None, "attr_b/out_copy_start")
    t = wrap(
        t,
        "        for back in range(slots):\n            copy(last - back).wait()",
        None,
        "attr_b/finish_drain",
    )
    return t


def patch_c(t):
    t = add_body_wrapper(
        t,
        "def _compress_rope_kernel(*refs, eps, ratio, tb, chunk, banks):",
        "_compress_rope_kernel_inner",
        "_compress_rope_kernel",
        "attr_c/body",
    )
    # the inner function keeps its signature; the wrapper forwards *args/**kwargs
    t = wrap(
        t,
        "                for c in range(_RECORD_ROWS):\n                    move_piece(stage, slot, row, c, read)",
        None,
        "attr_c/ring_write_stage",
    )
    t = wrap(
        t,
        "            if banks == 1:\n                record_copy(",
        "                    record_copy(stage, slot, pools[1], base, read, sem).start()",
        '"attr_c/ring_%s_start" % ("read" if read else "write")',
        dynamic=True,
    )
    t = wrap(
        t,
        "        jax.lax.fori_loop(start, offsets[tile + 1], body, 0)",
        None,
        '"attr_c/ring_%s_loop" % ("read" if read else "write")',
        dynamic=True,
    )
    t = wrap(
        t,
        "        jax.lax.fori_loop(offsets[tile], offsets[tile + 1], body, 0)",
        None,
        '"attr_c/ring_%s_wait" % ("read" if read else "write")',
        dynamic=True,
    )
    t = wrap(
        t,
        "        jax.lax.fori_loop(start, read_offsets[tile + 1], body, 0)",
        None,
        "attr_c/ring_read_unstage",
    )
    t = wrap(
        t,
        "        for c in range(chunks):\n            weight_copy(c).start(priority=1)",
        None,
        "attr_c/weight_start",
    )
    t = replace_once(
        t,
        """        for c in range(chunks):
            weight_copy(c).wait()
            columns = pl.ds(c * chunk, chunk)
            product_ref[:, columns] = jnp.dot(
                x_ref[...], weight_ref[:, columns], preferred_element_type=jnp.float32
            )
""",
        """        for c in range(chunks):
            with jax.named_scope('attr_c/weight_wait'):
                weight_copy(c).wait()
            columns = pl.ds(c * chunk, chunk)
            with jax.named_scope('attr_c/matmul'):
                product_ref[:, columns] = jnp.dot(
                    x_ref[...], weight_ref[:, columns], preferred_element_type=jnp.float32
                )
""",
    )
    t = wrap(
        t,
        "        product_ref[...] = jnp.dot(\n            x_ref[...], weight_ref[...], preferred_element_type=jnp.float32\n        )",
        None,
        "attr_c/matmul",
    )
    t = wrap(
        t,
        "        in_chunk = shifted_ref[pl.ds(7, tb), :]",
        "            e_cur / total\n        )",
        "attr_c/pooling",
    )
    t = wrap(
        t,
        "    inv_rms = jax.lax.rsqrt(jnp.mean(values * values, axis=-1, keepdims=True) + eps)",
        "    latent_out_ref[...] = latent",
        "attr_c/rmsnorm",
    )
    t = wrap(
        t,
        "    cos, sin = rope_gather.coefficients(rope_rows_ref[...])",
        "    rotated = jnp.concatenate((latent[:, :-_LANE], rotated_tail), axis=-1).astype(\n        jnp.float32\n    )",
        "attr_c/rope",
    )
    t = wrap(
        t,
        "    words, scales = _encode_csa2(rotated)",
        "    scales_out_ref[...] = scales",
        "attr_c/encode",
    )
    return t


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for name, fn in (
        ("qa_rope.py", patch_qa),
        ("qb_rope.py", patch_qb),
        ("compress_rope.py", patch_c),
    ):
        text = (src / name).read_text()
        out = fn(text)
        compile(out, name, "exec")
        (dst / name).write_text(out)
        print(
            "patched",
            name,
            text.count("\n"),
            "->",
            out.count("\n"),
            "lines;",
            out.count("attr_"),
            "scope mentions",
        )
