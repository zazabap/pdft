#!/usr/bin/env python3
"""Run the DIV2K-10q benchmark (m=n=10, 1024×1024) on a single GPU.

Usage: python benchmarks/run_div2k_10q.py <preset> [--gpu N] [--out DIR] [--allow-cpu]

At m=n=10 each image is 2^20 = 1M complex128 elements (~16 MB). With
batch_size=16 and forward+inverse einsum intermediates, peak GPU memory
is in the 5–10 GB range — fits comfortably on a 24 GB RTX 3090. If you
hit OOM at heavy presets, drop batch_size in `benchmarks/config.py`.
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401

import pdft

from data_loading import load_div2k
from run_quickdraw import _parse_args, run_dataset

DATASET_NAME = "div2k_10q"
M = 10
N = 10

BASIS_FACTORIES = {
    "qft": lambda: pdft.QFTBasis(m=M, n=N),
    "entangled_qft": lambda: pdft.EntangledQFTBasis(m=M, n=N),
    "tebd": lambda: pdft.TEBDBasis(m=M, n=N),
    "mera": lambda: pdft.MERABasis(m=M, n=N),  # 10+10=20 — power of 2 only via dim factoring
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_dataset(
        dataset_name=DATASET_NAME,
        m=M,
        n=N,
        basis_factories=BASIS_FACTORIES,
        loader_fn=lambda preset: load_div2k(
            preset.n_train, preset.n_test, seed=preset.seed, size=2**M,
        ),
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())
