#!/usr/bin/env python3
"""DIV2K-8q with BlockedBasis(RichBasis(3, 3), block_log_m=5, block_log_n=5).

Each 8×8 block is transformed by a SINGLE-LAYER parametric circuit whose
gate set is {Hadamard, U(4)} — the latter is a fully learnable 2-qubit
unitary (15 free real params per gate, vs 1 for plain CP). The H + U(4)
gate family is provably universal for SU(2^n), so the per-block circuit
contains 8×8 DCT as a special case.

The optimiser starts EXACTLY at the QFT trajectory (each U(4) gate is
warm-started to the diagonal CP it would replace), so any improvement
above the K=1 plain-blocked baseline (32.26 dB) is real.

Usage: python benchmarks/run_div2k_8q_blocked_rich.py <preset> [...]
"""

from __future__ import annotations

import sys

import _bootstrap  # noqa: F401

import pdft

from baselines import (
    block_dct_compress,
    block_fft_compress,
    global_dct_compress,
    global_fft_compress,
)
from data_loading import load_div2k
from run_quickdraw import _parse_args, run_dataset

DATASET_NAME = "div2k_8q_blocked"  # share preset table (256x256 outer image)
M_OUTER = 8
N_OUTER = 8
M_INNER = 3
N_INNER = 3
BLOCK_LOG_M = M_OUTER - M_INNER  # 5
BLOCK_LOG_N = N_OUTER - N_INNER  # 5


def _factories() -> dict:
    return {
        "blocked_rich": lambda: pdft.BlockedBasis(
            inner=pdft.RichBasis(m=M_INNER, n=N_INNER),
            block_log_m=BLOCK_LOG_M,
            block_log_n=BLOCK_LOG_N,
        ),
    }


def _baselines() -> dict:
    return {
        "fft": global_fft_compress,
        "dct": global_dct_compress,
        "block_fft_8": lambda img, kr: block_fft_compress(img, kr, block=8),
        "block_dct_8": lambda img, kr: block_dct_compress(img, kr, block=8),
    }


BASIS_FACTORIES = _factories()


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    return run_dataset(
        dataset_name=DATASET_NAME,
        m=M_OUTER,
        n=N_OUTER,
        basis_factories=_factories(),
        loader_fn=lambda preset: load_div2k(
            preset.n_train,
            preset.n_test,
            seed=preset.seed,
            size=2**M_OUTER,
        ),
        args=args,
        baselines=_baselines(),
    )


if __name__ == "__main__":
    sys.exit(main())
