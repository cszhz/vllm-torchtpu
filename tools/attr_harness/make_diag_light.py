"""Patch only outer QB semantic scopes for light attribution diagnostics.

Usage: make_diag_light.py SOURCE_QB DEST_QB
The original interleaved matmul/post order is preserved verbatim.  No scope is
placed inside the pair or fragment loops; the four scopes cover prepare_rows,
matmul, post, and finish function bodies as a whole.
"""

import sys
from pathlib import Path


def wrap_body(text: str, marker: str, end_marker: str, scope: str) -> str:
    assert text.count(marker) == 1, (marker, text.count(marker))
    start = text.index(marker)
    end = text.index(end_marker, start)
    line_start = text.rfind("\n", 0, start) + 1
    body = text[line_start:end]
    lines = body.splitlines(keepends=True)
    indent = lines[0][: len(lines[0]) - len(lines[0].lstrip())]
    wrapped = indent + f"with jax.named_scope({scope!r}):\n"
    wrapped += "".join(("    " + line) if line.strip() else line for line in lines)
    return text[:line_start] + wrapped + text[end:]


def patch(text: str) -> str:
    text = wrap_body(
        text,
        "        def quantize_fragment(fragment, _):",
        "    def copy(g):",
        "attr_b/prepare_rows",
    )
    text = wrap_body(
        text,
        "        # Later tiles of a window find its weights already resident.",
        "    def post(g, parity):",
        "attr_b/matmul_loop",
    )
    text = wrap_body(
        text,
        "        valid = g >= 0",
        "    first = step * pairs",
        "attr_b/post_loop",
    )
    text = wrap_body(
        text,
        "        reclaim(last)",
        "def vmem_bytes(",
        "attr_b/finish",
    )
    compile(text, "qb_rope.py", "exec")
    assert text.count("jax.named_scope") == 4
    return text


if __name__ == "__main__":
    src, dst = map(Path, sys.argv[1:])
    dst.write_text(patch(src.read_text()))
