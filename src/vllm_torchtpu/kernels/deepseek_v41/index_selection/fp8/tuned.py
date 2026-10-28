# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU v7 tiles and pipeline depths for the packed FP8 indexer."""

from dataclasses import astuple, dataclass

from .cache import PAGE_RECORDS, PAGE_ROWS


@dataclass(frozen=True)
class PagedMQAConfig:
    """Paged MQA tiles and pipeline depths."""

    query_tile: int = 8
    key_tile: int = 16384
    issue_depth: int = 4
    key_buffers: int = 4

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in astuple(self)):
            raise ValueError(
                "Paged MQA tiles and issue_depth must be positive integers"
            )
        if self.key_buffers < 2:
            raise ValueError("Paged MQA needs at least two K buffers to overlap DMA")


# TPU v7, H32/D128, transposed packed FP8 pages; measured at P-state 7.
PACKED_FP8_MQA_CONFIG = PagedMQAConfig(6, 65536, 8, 4)
PACKED_FP8_PREPARED_MQA_CONFIG = PagedMQAConfig(2, 32768, 4, 4)

# Dense tiles follow configured decode width, capped at one 256-row H32 MXU tile.
PACKED_FP8_MAX_QUERY_TILE = 8

# Bound short-output padding to one tile; 4K is the smallest measured choice.
PACKED_FP8_DENSE_MIN_KEY_TILE = 4096


def paged_mqa_key_tile(width: int, page_size: int, cap: int) -> int:
    """Balance tiles at v7 MXU-column granularity instead of padding to the cap."""
    alignment = max(page_size, 256)
    # Tiny storage pages use padded owner rows; keep the same K-buffer budget.
    page_rows = (PAGE_ROWS * page_size + PAGE_RECORDS - 1) // PAGE_RECORDS
    cap = cap * PAGE_ROWS * page_size // (PAGE_RECORDS * page_rows)
    cap = max(alignment, cap // alignment * alignment)
    tiles = max(1, (width + cap - 1) // cap)
    tile = (width + tiles * alignment - 1) // (tiles * alignment) * alignment
    return max(min(cap, PACKED_FP8_DENSE_MIN_KEY_TILE), tile)


# (queries, context, candidate blocks, topk, block size, cache banks).
# 96 = lcm(6 query rows, 32 SparseCore subcores). Three independent chunks
# expose TC/SC overlap without changing each query's six-row head grouping.
PACKED_FP8_INDEXER_QUERY_ROWS = {
    (288, width, candidates, 512, 8, 2): 96
    for width, candidates in ((98304, 0), (98304, 2048), (98816, 2048), (49408, 0))
}

# (queries, compact columns, topk): query rows for prepared-candidate pipelines.
PACKED_FP8_PREPARED_QUERY_ROWS = {(288, width, 512): 96 for width in (16384, 32768)}

# Exact streaming selection after dense scoring; measured at 288 x 98304, P7.
PACKED_FP8_DENSE_TOPK_THRESHOLD = 2048

# (queries, compact columns, topk): reconstruction threshold.
PACKED_FP8_CANDIDATE_TOPK_CONFIG = {
    (queries, 16384, 512): 2048 for queries in (288, 96)
}


# Rolling SparseCore gather: three buffers overlap reads and writes for 8-token blocks.
PAGED_MQA_GATHER_CONFIG = {(8, 1): (96, 3), (8, 2): (64, 3)}
PAGED_MQA_WHOLE_PAGE_GATHER_CONFIG = {(8, 1): (64, 3), (8, 2): (64, 3)}
