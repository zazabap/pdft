"""Benchmark presets. Plain dataclasses; values mirror Julia repo defaults."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Preset:
    name: str
    epochs: int  # passed as `steps` to pdft.train_basis
    n_train: int  # number of target images
    n_test: int  # held-out images for eval; equals n_train (P pairing)
    optimizer: str  # "gd" | "adam"
    lr: float
    seed: int = 42
    keep_ratios: tuple[float, ...] = field(default_factory=lambda: (0.05, 0.10, 0.15, 0.20))


PRESETS_QUICKDRAW: dict[str, Preset] = {
    "smoke": Preset("smoke", epochs=10, n_train=2, n_test=2, optimizer="gd", lr=0.01),
    "light": Preset("light", epochs=100, n_train=10, n_test=10, optimizer="gd", lr=0.01),
    "moderate": Preset("moderate", epochs=500, n_train=50, n_test=50, optimizer="adam", lr=0.01),
    "heavy": Preset("heavy", epochs=2000, n_train=200, n_test=200, optimizer="adam", lr=0.005),
}

PRESETS_DIV2K: dict[str, Preset] = {
    "smoke": Preset("smoke", epochs=10, n_train=2, n_test=2, optimizer="gd", lr=0.01),
    "light": Preset("light", epochs=100, n_train=5, n_test=5, optimizer="gd", lr=0.01),
    "moderate": Preset("moderate", epochs=500, n_train=20, n_test=20, optimizer="adam", lr=0.01),
    "heavy": Preset("heavy", epochs=2000, n_train=50, n_test=50, optimizer="adam", lr=0.005),
}


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
