#!/usr/bin/env python3
"""DIV2K-8q with BlockedBasis(RichBasis(3, 3), 5, 5).

Each 8×8 block is transformed by a SINGLE-LAYER parametric circuit whose
gate set is {Hadamard, U(4)} — fully learnable 2-qubit unitaries that
make the per-block circuit universal for SU(8). Contains 8x8 DCT exactly.

CLI extras (over the stock --preset/--gpu/etc):
  --epochs N           override preset.epochs (default: from preset)
  --lr-final F         override preset.lr_final (default: from preset)
  --init-dct           pre-fit RichBasis tensors to DCT before training,
                       so the basis IS DCT at training step 0. Cheap
                       (~30s on GPU). Tests whether DCT is a local optimum.

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


def _factories(init_dct: bool, dense: bool) -> dict:
    if init_dct:
        # Cheap one-off optimisation that pre-fits RichBasis tensors to DCT.
        # With dense=True, this drives loss → 0 (DCT is in the family). With
        # dense=False, the loss plateaus at ~63.7 (DCT is structurally unreachable).
        print(f"== fitting RichBasis init to DCT (dense={dense}) ==", flush=True)
        dct_tensors = pdft.fit_to_dct(M_INNER, N_INNER, dense=dense)
        print("== DCT-init done ==", flush=True)

        def factory():
            return pdft.BlockedBasis(
                inner=pdft.RichBasis(
                    m=M_INNER, n=N_INNER, tensors=dct_tensors, dense=dense
                ),
                block_log_m=BLOCK_LOG_M,
                block_log_n=BLOCK_LOG_N,
            )
    else:

        def factory():
            return pdft.BlockedBasis(
                inner=pdft.RichBasis(m=M_INNER, n=N_INNER, dense=dense),
                block_log_m=BLOCK_LOG_M,
                block_log_n=BLOCK_LOG_N,
            )

    return {"blocked_rich": factory}


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
        "--epochs", type=int, default=None, help="override preset.epochs (default: from preset)"
    )
    extra.add_argument(
        "--lr-final",
        type=float,
        default=None,
        help="override preset.lr_final (default: from preset)",
    )
    extra.add_argument(
        "--init-dct", action="store_true", help="pre-fit RichBasis tensors to DCT before training"
    )
    extra.add_argument(
        "--dense",
        action="store_true",
        help="use the universal-coverage RichBasis topology (provably contains DCT)",
    )
    extra_args, remaining = extra.parse_known_args(argv)
    args = _parse_args(remaining)
    args.epochs_override = extra_args.epochs
    args.lr_final_override = extra_args.lr_final
    args.init_dct = extra_args.init_dct
    args.dense = extra_args.dense
    return args


# Module-level placeholder; main() rebuilds with active config.
BASIS_FACTORIES = {
    "blocked_rich": lambda: pdft.BlockedBasis(
        inner=pdft.RichBasis(m=M_INNER, n=N_INNER),
        block_log_m=BLOCK_LOG_M,
        block_log_n=BLOCK_LOG_N,
    )
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_rich_args(argv)
    factories = _factories(init_dct=args.init_dct, dense=args.dense)
    baselines = _baselines()
    return run_dataset_with_overrides(args, factories, baselines)


def run_dataset_with_overrides(args, factories, baselines) -> int:
    """Wrap run_dataset to apply --epochs / --lr-final overrides on the preset.

    The preset is resolved INSIDE run_dataset, so the cleanest way to inject
    overrides is to call get_preset ourselves and replace before run_dataset
    re-fetches. Simpler: monkey-patch get_preset via a wrapper.
    """
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
    # Patch the preset table for this dataset_name + preset_name.
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
