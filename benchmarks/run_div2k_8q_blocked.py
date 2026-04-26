#!/usr/bin/env python3
"""DIV2K-8q with BlockedBasis — sweepable across block sizes.

The outer image is 2^8 × 2^8 = 256 × 256. Each block is `--block-size N`
wide (must be a power of two), so the within-block parametric circuit is
at m=n=log2(N) and the framework is compared head-to-head with BlockDCT
of the *same* size. BlockDCT 8 is also always reported as the JPEG-style
reference.

Usage:
    python benchmarks/run_div2k_8q_blocked.py <preset> [--block-size N] [...]

Defaults to block-size 8 (the original 8x8 result that matches BlockDCT 8x8).

Inner topologies trained: blocked_qft, blocked_entangled_qft, blocked_tebd.
MERA is omitted whenever m_inner + n_inner is not a power of 2.
"""

from __future__ import annotations

import argparse
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

DATASET_NAME = "div2k_8q_blocked"
M_OUTER = 8
N_OUTER = 8


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _factories_for_block_size(block_size: int, seed: int) -> dict:
    """BlockedBasis factories at a chosen block size, seeded for repro.

    block_size must be a power of 2 with 2 <= block_size <= 2^M_OUTER.
    """
    if not _is_power_of_two(block_size):
        raise ValueError(f"--block-size must be a power of 2, got {block_size}")
    log_block = block_size.bit_length() - 1
    if log_block < 1 or log_block > M_OUTER:
        raise ValueError(f"--block-size out of range [{2}, {2**M_OUTER}], got {block_size}")
    m_inner = log_block
    n_inner = log_block
    block_log_m = M_OUTER - m_inner
    block_log_n = N_OUTER - n_inner

    factories = {
        "blocked_qft": lambda: pdft.BlockedBasis(
            inner=pdft.QFTBasis(m=m_inner, n=n_inner),
            block_log_m=block_log_m,
            block_log_n=block_log_n,
        ),
        "blocked_entangled_qft": lambda: pdft.BlockedBasis(
            inner=pdft.EntangledQFTBasis(m=m_inner, n=n_inner, seed=seed),
            block_log_m=block_log_m,
            block_log_n=block_log_n,
        ),
        "blocked_tebd": lambda: pdft.BlockedBasis(
            inner=pdft.TEBDBasis(m=m_inner, n=n_inner, seed=seed),
            block_log_m=block_log_m,
            block_log_n=block_log_n,
        ),
    }
    # MERA inner only if m_inner + n_inner is a power of 2.
    if _is_power_of_two(m_inner + n_inner):
        factories["blocked_mera"] = lambda: pdft.BlockedBasis(
            inner=pdft.MERABasis(m=m_inner, n=n_inner, seed=seed),
            block_log_m=block_log_m,
            block_log_n=block_log_n,
        )
    return factories


def _baselines_for_block_size(block_size: int) -> dict:
    """Custom baseline dict: full-image FFT/DCT + BlockDCT/BlockFFT at the
    matched block size + BlockDCT8 (JPEG reference, kept for cross-size comp)."""
    bls = {
        "fft": global_fft_compress,
        "dct": global_dct_compress,
        f"block_fft_{block_size}": lambda img, kr: block_fft_compress(img, kr, block=block_size),
        f"block_dct_{block_size}": lambda img, kr: block_dct_compress(img, kr, block=block_size),
    }
    # Always include BlockDCT 8 as JPEG reference (skipping if it duplicates the matched one).
    if block_size != 8:
        bls["block_dct_8"] = lambda img, kr: block_dct_compress(img, kr, block=8)
    return bls


def _parse_blocked_args(argv: list[str] | None) -> argparse.Namespace:
    """Layer --block-size on top of the shared --preset/--gpu/etc CLI.

    _parse_args (from run_quickdraw) is strict about unknown args, so strip
    --block-size out before passing through, then attach it to the namespace.
    """
    if argv is None:
        argv = list(sys.argv[1:])
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument(
        "--block-size",
        type=int,
        default=8,
        help="block size (must be power of 2, 2 <= N <= 256). Default: 8.",
    )
    extra_args, remaining = extra.parse_known_args(argv)
    args = _parse_args(remaining)
    args.block_size = extra_args.block_size
    return args


# Module-level default factories, used by run_dataset for re-seeding tag check
# and dispatch (`seed` is filled in at run time using the active preset's seed).
BASIS_FACTORIES = _factories_for_block_size(block_size=8, seed=42)


def main(argv: list[str] | None = None) -> int:
    args = _parse_blocked_args(argv)
    factories = _factories_for_block_size(args.block_size, seed=42)
    baselines = _baselines_for_block_size(args.block_size)
    # Stamp the block size into the dataset name segment for clean dir naming.
    return run_dataset(
        dataset_name=DATASET_NAME,
        m=M_OUTER,
        n=N_OUTER,
        basis_factories=factories,
        loader_fn=lambda preset: load_div2k(
            preset.n_train,
            preset.n_test,
            seed=preset.seed,
            size=2**M_OUTER,
        ),
        args=args,
        baselines=baselines,
    )


if __name__ == "__main__":
    sys.exit(main())
