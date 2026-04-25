"""Layer A: config.py unit tests. No GPU/datasets. Fast (<1s)."""

from __future__ import annotations

import pytest

from config import (
    PRESETS_DIV2K,
    PRESETS_QUICKDRAW,
    Preset,
    get_preset,
)


def test_preset_dataclass_fields():
    p = Preset(name="x", epochs=10, n_train=2, n_test=2, optimizer="gd", lr=0.01)
    assert p.name == "x"
    assert p.epochs == 10
    assert p.n_train == 2
    assert p.n_test == 2
    assert p.optimizer == "gd"
    assert p.lr == 0.01
    assert p.seed == 42  # default
    assert p.keep_ratios == (0.05, 0.10, 0.15, 0.20)  # default


def test_preset_is_frozen():
    p = Preset(name="x", epochs=10, n_train=2, n_test=2, optimizer="gd", lr=0.01)
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        p.epochs = 20  # type: ignore[misc]


@pytest.mark.parametrize("presets", [PRESETS_QUICKDRAW, PRESETS_DIV2K])
def test_all_presets_have_four_levels(presets):
    assert set(presets.keys()) == {"smoke", "light", "moderate", "heavy"}


@pytest.mark.parametrize("presets", [PRESETS_QUICKDRAW, PRESETS_DIV2K])
def test_n_train_equals_n_test_for_p_pairing(presets):
    """P pairing: basis_i evaluated on test_i — forces n_train == n_test."""
    for name, p in presets.items():
        assert p.n_train == p.n_test, f"{name}: n_train={p.n_train}, n_test={p.n_test}"


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


def test_get_preset_quickdraw():
    p = get_preset("quickdraw", "smoke")
    assert p.name == "smoke"
    assert p.n_train == 2


def test_get_preset_div2k():
    p = get_preset("div2k_8q", "moderate")
    assert p.name == "moderate"


def test_get_preset_unknown_dataset():
    with pytest.raises(KeyError, match="unknown dataset"):
        get_preset("unknown", "smoke")


def test_get_preset_unknown_preset():
    with pytest.raises(KeyError, match="unknown preset"):
        get_preset("quickdraw", "unknown")
