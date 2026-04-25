#!/usr/bin/env python3
"""Run the DIV2K-8q benchmark (m=n=8, 256×256) on a single GPU.

Usage: python benchmarks/run_div2k_8q.py <preset> [--gpu N] [--out DIR] [--allow-cpu]
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401

import pdft

from data_loading import load_div2k
from run_quickdraw import _parse_args, run_dataset

DATASET_NAME = "div2k_8q"
M = 8
N = 8

BASIS_FACTORIES = {
    "qft": lambda: pdft.QFTBasis(m=M, n=N),
    "entangled_qft": lambda: pdft.EntangledQFTBasis(m=M, n=N),
    "tebd": lambda: pdft.TEBDBasis(m=M, n=N),
    "mera": lambda: pdft.MERABasis(m=M, n=N),  # 8+8=16 is power of 2 → runs
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_dataset(
        dataset_name=DATASET_NAME,
        m=M,
        n=N,
        basis_factories=BASIS_FACTORIES,
        loader_fn=lambda preset: load_div2k(
            preset.n_train,
            preset.n_test,
            seed=preset.seed,
            size=2**M,
        ),
        args=args,
    )


if __name__ == "__main__":
    sys.exit(main())
