#!/usr/bin/env python3
"""DIV2K-8q with BlockedBasis(<inner>, block_log_m=5, block_log_n=5).

Each 8x8 block is transformed by a SINGLE-LAYER parametric circuit.
Inner topology selected via --basis-class:

    rich    QFT topology, H + 3 U(4) gates per dim, 54 free real params
            (fully complex). Strict 54-dim submanifold of SU(8).
            -0.30 dB vs BlockDCT 8x8.

    real    Approach A: same QFT topology with REAL-orthogonal gates.
            21 free real params per dim (3 H × 1 + 3 SO(4) × 6).
            Strict submanifold of O(8) (dim 28). Whether it contains DCT
            is empirical; trains via UnitaryManifold (real init + real
            gradient + Cayley retraction → real-staying tensors).

    dct     Approach B: 1D macro-gate per direction, init at canonical DCT.
            56 free real params total (2 × 28 = 2 × dim O(8)).
            Provably contains DCT exactly. If Adam moves AWAY from DCT
            and lands at higher PSNR, we have found a basis genuinely
            better than DCT for natural images.

CLI extras:
  --basis-class {rich,real,dct}
  --epochs N
  --lr-final F

Usage: python benchmarks/run_div2k_8q_blocked_rich.py <preset> [...]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace

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
M_INNER = 3
N_INNER = 3
BLOCK_LOG_M = M_OUTER - M_INNER  # 5
BLOCK_LOG_N = N_OUTER - N_INNER  # 5


def _make_inner(basis_class: str):
    """Build the within-block parametric basis for the chosen variant."""
    if basis_class == "rich":
        return pdft.RichBasis(m=M_INNER, n=N_INNER)
    if basis_class == "real":
        return pdft.RealRichBasis(m=M_INNER, n=N_INNER)
    if basis_class == "dct":
        return pdft.DCTBasis(m=M_INNER, n=N_INNER, init_dct=True)
    raise ValueError(f"unknown --basis-class {basis_class!r}")


def _factories(basis_class: str) -> dict:
    name = f"blocked_{basis_class}"

    def factory():
        return pdft.BlockedBasis(
            inner=_make_inner(basis_class),
            block_log_m=BLOCK_LOG_M,
            block_log_n=BLOCK_LOG_N,
        )

    return {name: factory}


def _baselines() -> dict:
    return {
        "fft": global_fft_compress,
        "dct": global_dct_compress,
        "block_fft_8": lambda img, kr: block_fft_compress(img, kr, block=8),
        "block_dct_8": lambda img, kr: block_dct_compress(img, kr, block=8),
    }


def _parse_rich_args(argv: list[str] | None) -> argparse.Namespace:
    if argv is None:
        argv = list(sys.argv[1:])
    extra = argparse.ArgumentParser(add_help=False)
    extra.add_argument(
        "--basis-class",
        choices=["rich", "real", "dct"],
        default="rich",
        help="within-block parametric circuit topology (default: rich)",
    )
    extra.add_argument("--epochs", type=int, default=None, help="override preset.epochs")
    extra.add_argument("--lr-final", type=float, default=None, help="override preset.lr_final")
    extra_args, remaining = extra.parse_known_args(argv)
    args = _parse_args(remaining)
    args.basis_class = extra_args.basis_class
    args.epochs_override = extra_args.epochs
    args.lr_final_override = extra_args.lr_final
    return args


# Module-level placeholder; main() rebuilds with active config.
BASIS_FACTORIES = _factories("rich")


def main(argv: list[str] | None = None) -> int:
    args = _parse_rich_args(argv)
    factories = _factories(args.basis_class)
    baselines = _baselines()
    return run_dataset_with_overrides(args, factories, baselines)


def run_dataset_with_overrides(args, factories, baselines) -> int:
    """Wrap run_dataset to apply --epochs/--lr-final overrides."""
    from config import _DATASETS, get_preset as _orig_get_preset

    original = _orig_get_preset(DATASET_NAME, args.preset)
    overrides = {}
    if args.epochs_override is not None:
        overrides["epochs"] = args.epochs_override
    if args.lr_final_override is not None:
        overrides["lr_final"] = args.lr_final_override
    patched = replace(original, **overrides) if overrides else original
    if overrides:
        print(f"== applying preset overrides: {overrides} ==", flush=True)
    _DATASETS[DATASET_NAME] = {**_DATASETS[DATASET_NAME], args.preset: patched}

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
