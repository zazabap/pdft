"""Benchmark presets matching ParametricDFT-Benchmarks.jl/config.jl.

Values mirror the Julia preset table for the batched pipeline that lands in
`pdft.train_basis_batched` (issue #5). One shared basis is trained across the
whole dataset for `epochs` full passes; each pass yields `ceil(n_train / batch_size)`
optimizer steps with a cosine-with-warmup learning-rate schedule.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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


# Mirrors ParametricDFT-Benchmarks.jl/config.jl::TRAINING_PRESETS.
# Identical values for QuickDraw and DIV2K (the Julia table is dataset-agnostic);
# we keep two dicts so per-dataset overrides remain possible without breaking callers.

_BASE_PRESETS: dict[str, Preset] = {
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
    "light": Preset(
        "light",
        epochs=5,
        n_train=10,
        n_test=20,
        optimizer="adam",
        batch_size=8,
        warmup_frac=0.05,
        lr_peak=0.01,
        lr_final=0.001,
        max_grad_norm=1.0,
        validation_split=0.2,
        early_stopping_patience=5,
    ),
    # `moderate` uses lr_peak=0.3 instead of Julia's 0.01. Empirically: at
    # the QFT-init point the topk-MSE loss surface is approximately flat
    # plateau under our PRNG-selected image draw, so Adam at lr=0.01 oscillates
    # without escaping (10-epoch loss drop ~0.18). At lr_peak=0.3, the cosine
    # schedule's first few large steps escape the plateau, and the basis
    # converges to PSNR≈35 dB at kr=0.20 in 10 epochs (vs Julia's 30.26 dB
    # on their own image draw at lr=0.01). The lr_peak that Julia uses is
    # tuned for Julia's specific PRNG draw and isn't robust across draws —
    # see issue #7. The Julia-parity preset is preserved as `julia_moderate`.
    "moderate": Preset(
        "moderate",
        epochs=10,
        n_train=20,
        n_test=50,
        optimizer="adam",
        batch_size=16,
        warmup_frac=0.05,
        lr_peak=0.3,
        lr_final=0.03,
        max_grad_norm=None,
        validation_split=0.2,
        early_stopping_patience=10,
    ),
    "julia_moderate": Preset(
        "julia_moderate",
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
    "heavy": Preset(
        "heavy",
        epochs=20,
        n_train=50,
        n_test=100,
        optimizer="adam",
        batch_size=16,
        warmup_frac=0.05,
        lr_peak=0.01,
        lr_final=0.001,
        max_grad_norm=1.0,
        validation_split=0.2,
        early_stopping_patience=10,
    ),
    "generalized": Preset(
        "generalized",
        epochs=40,
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


_DATASETS = {
    "quickdraw": PRESETS_QUICKDRAW,
    "div2k_8q": PRESETS_DIV2K,
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
