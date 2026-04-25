"""Layer A: data_loading.py unit tests using fixtures (no real datasets)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_loading import load_div2k, load_quickdraw

FIXTURES = Path(__file__).parent / "fixtures"
QD_STUB = FIXTURES / "quickdraw_stub"
DV_STUB = FIXTURES / "div2k_stub"


def test_load_quickdraw_shape_and_dtype():
    train, test = load_quickdraw(n_train=2, n_test=2, seed=42, data_root=QD_STUB)
    assert train.shape == (2, 32, 32)
    assert test.shape == (2, 32, 32)
    assert train.dtype == np.float32
    assert 0.0 <= train.min() and train.max() <= 1.0


def test_load_quickdraw_seed_deterministic():
    a_train, _ = load_quickdraw(n_train=2, n_test=2, seed=42, data_root=QD_STUB)
    b_train, _ = load_quickdraw(n_train=2, n_test=2, seed=42, data_root=QD_STUB)
    np.testing.assert_array_equal(a_train, b_train)


def test_load_quickdraw_too_many_raises():
    # Stub has 2 categories × 10 = 20 images. Asking for 30 raises.
    with pytest.raises(ValueError, match="not enough"):
        load_quickdraw(n_train=20, n_test=20, seed=42, data_root=QD_STUB)


def test_load_quickdraw_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="data_root"):
        load_quickdraw(n_train=1, n_test=1, seed=42, data_root=tmp_path / "nope")


def test_load_div2k_shape_and_dtype():
    train, test = load_div2k(n_train=1, n_test=1, seed=42, size=32, data_root=DV_STUB)
    assert train.shape == (1, 32, 32)
    assert test.shape == (1, 32, 32)
    assert train.dtype == np.float32


def test_load_div2k_resize_works():
    """Stub PNGs are 64x64; loader resizes to size=128."""
    train, _ = load_div2k(n_train=1, n_test=1, seed=42, size=128, data_root=DV_STUB)
    assert train.shape == (1, 128, 128)


def test_load_div2k_too_many_raises():
    # Stub has 3 PNGs. Asking for 5 raises.
    with pytest.raises(ValueError, match="not enough"):
        load_div2k(n_train=3, n_test=3, seed=42, data_root=DV_STUB)


def test_load_div2k_missing_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="data_root"):
        load_div2k(n_train=1, n_test=1, seed=42, data_root=tmp_path / "nope")
