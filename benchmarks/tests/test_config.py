"""Layer A: config.py unit tests. No GPU/datasets. Fast (<1s)."""

from __future__ import annotations

import pytest

from config import (
    PRESETS_DIV2K,
    PRESETS_QUICKDRAW,
    Preset,
    get_preset,
)


def _mk(**overrides):
    """Build a minimally-valid Preset, allowing per-test overrides."""
    base = dict(
        name="x",
        epochs=2,
        n_train=5,
        n_test=2,
        optimizer="adam",
        batch_size=4,
        warmup_frac=0.05,
        lr_peak=0.01,
        lr_final=0.001,
        max_grad_norm=1.0,
        validation_split=0.2,
        early_stopping_patience=2,
    )
    base.update(overrides)
    return Preset(**base)


def test_preset_dataclass_fields():
    p = _mk()
    assert p.name == "x"
    assert p.epochs == 2
    assert p.batch_size == 4
    assert p.lr_peak == 0.01
    assert p.lr_final == 0.001
    assert p.max_grad_norm == 1.0
    assert p.warmup_frac == 0.05
    assert p.validation_split == 0.2
    assert p.early_stopping_patience == 2
    assert p.seed == 42  # default
    assert p.keep_ratios == (0.05, 0.10, 0.15, 0.20)  # default


def test_preset_is_frozen():
    p = _mk()
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        p.epochs = 99  # type: ignore[misc]


@pytest.mark.parametrize("presets", [PRESETS_QUICKDRAW, PRESETS_DIV2K])
def test_all_presets_have_named_levels(presets):
    """Match the Julia table levels."""
    assert set(presets.keys()) == {"smoke", "light", "moderate", "heavy", "generalized"}


@pytest.mark.parametrize("presets", [PRESETS_QUICKDRAW, PRESETS_DIV2K])
def test_keep_ratios_are_valid(presets):
    for name, p in presets.items():
        assert len(p.keep_ratios) > 0, f"{name}: empty keep_ratios"
        for kr in p.keep_ratios:
            assert 0.0 < kr <= 1.0, f"{name}: invalid keep_ratio {kr}"


@pytest.mark.parametrize("presets", [PRESETS_QUICKDRAW, PRESETS_DIV2K])
def test_optimizer_strings_valid(presets):
    for name, p in presets.items():
        assert p.optimizer in {"gd", "adam"}, f"{name}: unknown optimizer {p.optimizer}"


@pytest.mark.parametrize("presets", [PRESETS_QUICKDRAW, PRESETS_DIV2K])
def test_batch_size_positive(presets):
    for name, p in presets.items():
        assert p.batch_size >= 1, f"{name}: invalid batch_size {p.batch_size}"


@pytest.mark.parametrize("presets", [PRESETS_QUICKDRAW, PRESETS_DIV2K])
def test_warmup_frac_in_unit_range(presets):
    for name, p in presets.items():
        assert 0.0 <= p.warmup_frac < 1.0, f"{name}: invalid warmup_frac {p.warmup_frac}"


@pytest.mark.parametrize("presets", [PRESETS_QUICKDRAW, PRESETS_DIV2K])
def test_validation_split_in_unit_range(presets):
    for name, p in presets.items():
        assert 0.0 <= p.validation_split < 1.0, (
            f"{name}: invalid validation_split {p.validation_split}"
        )


def test_get_preset_quickdraw():
    p = get_preset("quickdraw", "smoke")
    assert p.name == "smoke"
    assert p.n_train == 5
    assert p.batch_size == 16


def test_get_preset_div2k():
    p = get_preset("div2k_8q", "moderate")
    assert p.name == "moderate"
    assert p.batch_size == 16


def test_get_preset_generalized_matches_paper():
    p = get_preset("div2k_8q", "generalized")
    assert p.epochs == 40
    assert p.n_train == 500
    assert p.batch_size == 50
    assert p.lr_peak == 0.003
    assert p.lr_final == 0.0003
    assert p.validation_split == 0.15


def test_get_preset_unknown_dataset():
    with pytest.raises(KeyError, match="unknown dataset"):
        get_preset("unknown", "smoke")


def test_get_preset_unknown_preset():
    with pytest.raises(KeyError, match="unknown preset"):
        get_preset("quickdraw", "unknown")
