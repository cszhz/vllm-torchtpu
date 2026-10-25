"""NumPy reference for the SparseCore top-k contract as `streamindex_topk` uses it."""

import numpy as np

MONO_MASK = np.int32(0x7FFFFFFF)
KEY_NEG_INF = np.int32(np.array(-np.inf, np.float32).view(np.int32)) ^ MONO_MASK


def monotone_key(bits: np.ndarray) -> np.ndarray:
    bits = bits.astype(np.int32)
    return bits ^ ((bits >> 31) & MONO_MASK)


def reference_topk(
    bits: np.ndarray, k: int, row_lengths: np.ndarray, *, ties_to_lowest: bool = True
) -> np.ndarray:
    """i32[R, k]: per row the k largest monotone keys among columns < length with
    key > key(-inf); ties to the lowest column; ascending columns; -1 suffix.
    ``ties_to_lowest=False`` is the wrong rule, for positive controls."""
    r, n = bits.shape
    out = np.full((r, k), -1, np.int32)
    for i in range(r):
        ln = int(np.clip(row_lengths[i], 0, n))
        key = monotone_key(bits[i, :ln])
        cols = np.flatnonzero(key > KEY_NEG_INF)
        order = np.lexsort((cols if ties_to_lowest else -cols, -key[cols].astype(np.int64)))
        sel = np.sort(cols[order[:k]])
        out[i, : sel.size] = sel
    return out


def f32_bits(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, np.float32).view(np.int32)


def make_case(kind: str, r: int, n: int, rng: np.random.Generator):
    """(bits i32[r, n], lengths i32[r]) for one adversarial / random case."""
    lens = np.full((r,), n, np.int32)
    if kind == "random":
        x = f32_bits(rng.standard_normal((r, n), dtype=np.float32))
    elif kind == "verify":  # 7-row verify windows: lengths step by one
        x = f32_bits(rng.standard_normal((r, n), dtype=np.float32))
        lens = (n - 7 + np.arange(r) % 7 + 1).astype(np.int32)
    elif kind == "short":  # fewer than k eligible, mixed lengths incl. 0
        x = f32_bits(rng.standard_normal((r, n), dtype=np.float32))
        lens = rng.integers(0, 1500, size=r).astype(np.int32)
        lens[0] = 0
        lens[-1] = 1024
    elif kind == "ties":  # 17 distinct values: boundary always inside a tie run
        x = f32_bits(rng.integers(0, 17, size=(r, n)).astype(np.float32))
    elif kind == "all_equal":
        x = f32_bits(np.full((r, n), 0.5, np.float32))
    elif kind == "zeros_signed":  # mostly +-0.0 with a few positives: -0.0 < +0.0
        v = np.where(rng.random((r, n)) < 0.5, -0.0, 0.0).astype(np.float32)
        v[rng.random((r, n)) < 0.01] = 1.0
        x = f32_bits(v)
    elif kind == "neg_inf":  # many -inf, a few finite: fewer than k eligible
        v = np.full((r, n), -np.inf, np.float32)
        m = rng.random((r, n)) < 0.02
        v[m] = rng.standard_normal(m.sum()).astype(np.float32)
        x = f32_bits(v)
    elif kind == "desc":  # strictly descending negatives: answer is columns 0..k-1
        x = f32_bits(np.repeat(-np.arange(n, dtype=np.float32)[None], r, 0) - 1.0)
    elif kind == "asc":  # top-k are the last k columns of each row
        x = f32_bits(np.repeat(np.arange(n, dtype=np.float32)[None], r, 0))
    elif kind == "nan":  # +NaN bits sort above +inf, -NaN below -inf (raw-bit contract)
        v = rng.standard_normal((r, n)).astype(np.float32)
        v[:, 5] = np.inf
        x = f32_bits(v)
        x[:, 3] = np.int32(0x7FC00000)
        x[:, 7] = np.int32(-0x00400000)  # 0xFFC00000
    elif kind == "strided":  # winners concentrated on one lane / segment pattern
        v = rng.standard_normal((r, n)).astype(np.float32)
        v[:, ::128] += 100.0
        x = f32_bits(v)
    elif kind == "straddle":  # a tie run at the k-boundary across column n // 8 (a slice edge)
        v = -np.abs(rng.standard_normal((r, n))).astype(np.float32) - 1.0
        for i in range(r):
            v[i, rng.choice(n, 900, replace=False)] = 2.0 + rng.random(900).astype(np.float32)
        v[:, n // 8 - 200:n // 8 + 200] = 0.5
        x = f32_bits(v)
    elif kind == "garbage_tail":  # past each row's length: -inf / NaN / -1 / +inf bits
        x = f32_bits(rng.standard_normal((r, n), dtype=np.float32))
        lens = rng.choice(np.array([750, 1024, 1500, 5000, n // 2 + 3], np.int32), size=r)
        junk = np.array([0xFF800000, 0xFFFFFFFF, 0x7FC00000, 0x7F800000], np.uint32).view(np.int32)
        cols = np.arange(n)[None, :]
        x = np.where(cols >= lens[:, None], junk[rng.integers(0, 4, size=(r, n))], x)
    else:
        raise ValueError(kind)
    return x, lens


KINDS = ("random", "verify", "short", "ties", "all_equal", "zeros_signed", "neg_inf",
         "desc", "asc", "nan", "strided", "garbage_tail", "straddle")


def reference_union(lists: np.ndarray) -> np.ndarray:
    """Ascending union of an 8-row group's top-k lists, as ``column << 8 | member bits``."""
    bits = {}
    for r, row in enumerate(lists):
        for c in row[row >= 0]:
            bits[int(c)] = bits.get(int(c), 0) | (1 << r)
    return np.array([(c << 8) | b for c, b in sorted(bits.items())], np.int64).astype(np.int32)


def split_cuts(bits: np.ndarray, k: int, row_lengths: np.ndarray, n_loc: int, ranks: int):
    """Per-row global cut for columns split contiguously over `ranks` slices of n_loc:
    G (i32[R], monotone key of the k-th largest eligible key, INT_MIN if fewer than k) and
    allow (i32[ranks, R]): ties at G each slice takes in column order (-1 = all of them),
    lower slices first, which reproduces the single-slice tie rule."""
    r = bits.shape[0]
    g = np.full((r,), np.iinfo(np.int32).min, np.int32)
    allow = np.full((ranks, r), -1, np.int32)
    for i in range(r):
        ln = int(np.clip(row_lengths[i], 0, bits.shape[1]))
        key = monotone_key(bits[i, :ln])
        key = np.where(key > KEY_NEG_INF, key, np.iinfo(np.int32).min)
        valid = np.sort(key[key > np.iinfo(np.int32).min])[::-1]
        if valid.size < k:
            continue
        g[i] = valid[k - 1]
        left = k - int((valid > g[i]).sum())
        for q in range(ranks):
            ties = int((key[q * n_loc:(q + 1) * n_loc] == g[i]).sum())
            allow[q, i] = -1 if ties <= left else left
            left = max(left - ties, 0)
    return g, allow
