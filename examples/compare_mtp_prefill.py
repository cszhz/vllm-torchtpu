# SPDX-License-Identifier: Apache-2.0
"""Compare trusted PP/PCP diagnostic snapshots by logical prompt position."""

import argparse
from pathlib import Path

import torch


def read_snapshot(directory: str) -> dict[str, torch.Tensor]:
    chunks = []
    for path in sorted(Path(directory).glob("rank-*.pt")):
        snapshot = torch.load(path, map_location="cpu", weights_only=True)
        if len(snapshot["chunks"]) != 1:
            raise ValueError("Comparison requires one complete prompt per run.")
        chunks.append(snapshot["chunks"][0])
    if not chunks:
        raise ValueError(f"No rank snapshots in {directory}")
    logical = torch.cat([c["logical_token_indices"] for c in chunks]).long()
    order = logical.argsort()
    torch.testing.assert_close(logical[order], torch.arange(logical.numel()))
    return {
        key: torch.cat([c[key] for c in chunks], dim=axis).index_select(axis, order)
        for key, axis in (
            ("input_ids", 0),
            ("positions", -1),
            ("target_hidden", 0),
            ("mtp_hidden", 0),
        )
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--atol", type=float, required=True)
    parser.add_argument("--rtol", type=float, required=True)
    args = parser.parse_args()
    reference = read_snapshot(args.reference)
    candidate = read_snapshot(args.candidate)
    for key in ("input_ids", "positions"):
        torch.testing.assert_close(candidate[key], reference[key], atol=0, rtol=0)
    for key in ("target_hidden", "mtp_hidden"):
        lhs, rhs = candidate[key].float(), reference[key].float()
        if not torch.isfinite(lhs).all() or not torch.isfinite(rhs).all():
            raise ValueError(f"Non-finite {key}")
        error = (lhs - rhs).abs()
        print(f"{key}: max_abs={error.max().item()} mean_abs={error.mean().item()}")
        torch.testing.assert_close(lhs, rhs, atol=args.atol, rtol=args.rtol)
    print("Hidden-state comparison passed. KV numerical equivalence is not checked.")


if __name__ == "__main__":
    main()
