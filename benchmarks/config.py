"""Benchmark presets matching ParametricDFT-Benchmarks.jl/config.jl.

Values mirror the Julia preset table for the batched pipeline that lands in
`pdft.train_basis_batched` (issue #5). One shared basis is trained across the
whole dataset for `epochs` full passes; each pass yields `ceil(n_train / batch_size)`
optimizer steps with a cosine-with-warmup learning-rate schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Preset:
    name: str
    epochs: int
    n_train: int
    n_test: int
    optimizer: str  # "gd" | "adam"
    batch_size: int
    warmup_frac: float
    lr_peak: float
    lr_final: float
    max_grad_norm: float | None
    validation_split: float
    early_stopping_patience: int
    seed: int = 42
    keep_ratios: tuple[float, ...] = field(default_factory=lambda: (0.05, 0.10, 0.15, 0.20))
    val_every_k_epochs: int = 1


# Mirrors ParametricDFT-Benchmarks.jl/config.jl::TRAINING_PRESETS.
# Identical values for QuickDraw and DIV2K (the Julia table is dataset-agnostic);
# we keep two dicts so per-dataset overrides remain possible without breaking callers.

_BASE_PRESETS: dict[str, Preset] = {
    # Fast CPU sanity check (CI). Not a quality benchmark.
    "smoke": Preset(
        "smoke",
        epochs=2,
        n_train=5,
        n_test=2,
        optimizer="adam",
        batch_size=16,
        warmup_frac=0.05,
        lr_peak=0.01,
        lr_final=0.001,
        max_grad_norm=1.0,
        validation_split=0.2,
        early_stopping_patience=2,
    ),
    # Julia-parity preset: same hyperparameters as
    # ParametricDFT-Benchmarks.jl/config.jl::TRAINING_PRESETS[:moderate].
    # Reference number for QFT @ DIV2K-8q kr=0.20: 27.81 dB.
    "moderate": Preset(
        "moderate",
        epochs=10,
        n_train=20,
        n_test=50,
        optimizer="adam",
        batch_size=16,
        warmup_frac=0.05,
        lr_peak=0.01,
        lr_final=0.001,
        max_grad_norm=1.0,
        validation_split=0.2,
        early_stopping_patience=10,
    ),
    # Long schedule for the deepest config. Mirrors Julia's `:generalized`
    # but with epochs bumped from 40 → 60. Reference number (Julia, 40 ep)
    # for QFT @ DIV2K-8q kr=0.20: 29.36 dB.
    "generalized": Preset(
        "generalized",
        epochs=60,
        n_train=500,
        n_test=50,
        optimizer="adam",
        batch_size=50,
        warmup_frac=0.05,
        lr_peak=0.003,
        lr_final=0.0003,
        max_grad_norm=1.0,
        validation_split=0.15,
        early_stopping_patience=5,
    ),
}

PRESETS_QUICKDRAW: dict[str, Preset] = dict(_BASE_PRESETS)
PRESETS_DIV2K: dict[str, Preset] = dict(_BASE_PRESETS)


# Helper to clone a Preset with overridden batch_size and (optionally)
# val_every_k_epochs — used by the DIV2K-10q table below, where the base
# presets' bs values (16/50) overflow 24 GB at m=n=10 due to the (2,)*20
# × batch einsum intermediates, and val passes are expensive enough that
# evaluating every-other-epoch is a meaningful speedup.
def _override_bs(p: Preset, bs: int, val_every_k_epochs: int | None = None) -> Preset:
    from dataclasses import replace

    fields: dict[str, Any] = {"batch_size": bs}
    if val_every_k_epochs is not None:
        fields["val_every_k_epochs"] = val_every_k_epochs
    return replace(p, **fields)


# DIV2K-10q (m=n=10) batch_size — empirically measured on RTX 3090 (24 GB):
#   bs=2  → peak ~3.5 GB,  baseline throughput
#   bs=4  → peak ~7.0 GB,  1.7× throughput per image
#   bs=8  → peak ~14 GB,   2.9× throughput per image (the new default)
# At bs=8 the per-step einsum eats ~14 GB; train+val sets resident on GPU
# add ~10 GB more. Total expected peak ~22-24 GB on the heaviest preset
# (`generalized`, n_train=500, val_split=0.15) — within margin but tight.
# Drop to --batch-size 4 if you see OOM on `generalized`.
#
# val_every_k_epochs=2 halves the validation cost per epoch (each val pass
# at m=n=10 with 75 images is ~60s). Early-stopping patience now counts
# in evaluations, so 5 evals × 2 epochs = 10 epochs without improvement.
# MERA at m=n=10 is silently skipped (m+n=20 is not a power of 2).
PRESETS_DIV2K_10Q: dict[str, Preset] = {
    name: _override_bs(p, bs=8, val_every_k_epochs=2) for name, p in _BASE_PRESETS.items()
}


_DATASETS = {
    "quickdraw": PRESETS_QUICKDRAW,
    "div2k_8q": PRESETS_DIV2K,
    "div2k_10q": PRESETS_DIV2K_10Q,
}


def get_preset(dataset: str, preset_name: str) -> Preset:
    """Look up a preset by dataset + name. Raises KeyError on unknown values."""
    if dataset not in _DATASETS:
        raise KeyError(f"unknown dataset {dataset!r}; choices: {sorted(_DATASETS)}")
    presets = _DATASETS[dataset]
    if preset_name not in presets:
        raise KeyError(
            f"unknown preset {preset_name!r} for dataset {dataset!r}; choices: {sorted(presets)}"
        )
    return presets[preset_name]
