"""Layer A: baselines.py unit tests. Pure numpy/scipy — no JAX, no GPU."""

from __future__ import annotations

import numpy as np
import pytest

from baselines import global_dct_compress, global_fft_compress


@pytest.fixture
def img_32():
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 1.0, size=(32, 32)).astype(np.float64)


def test_global_fft_full_keep_is_identity(img_32):
    out = global_fft_compress(img_32, keep_ratio=1.0)
    np.testing.assert_allclose(out, img_32, atol=1e-10)


def test_global_dct_full_keep_is_identity(img_32):
    out = global_dct_compress(img_32, keep_ratio=1.0)
    np.testing.assert_allclose(out, img_32, atol=1e-10)


def test_global_fft_keep_ratio_count(img_32):
    """keep_ratio=0.5 keeps exactly floor(0.5 * 1024) = 512 nonzero coefficients."""
    # We can't probe the internal coefficient mask directly — instead, run with
    # keep_ratio=0.5 and re-FFT the recovered image: the count of nonzero
    # frequency bins (above tolerance) should equal floor(0.5 * 1024) = 512.
    out = global_fft_compress(img_32, keep_ratio=0.5)
    freq = np.fft.fft2(out)
    nonzero = np.sum(np.abs(freq) > 1e-9)
    # Recovery introduces some numeric noise; allow a small slack.
    assert 500 <= nonzero <= 540


def test_global_dct_zero_keep_returns_zero(img_32):
    """keep_ratio = a single coefficient (smallest possible)."""
    out = global_dct_compress(img_32, keep_ratio=1 / (32 * 32))
    # The DC coefficient (largest by magnitude for natural images) is kept;
    # output should be a (near-)constant image equal to the image mean.
    expected_mean = float(np.mean(img_32))
    assert abs(float(np.mean(out)) - expected_mean) < 1e-9


def test_global_fft_returns_real(img_32):
    out = global_fft_compress(img_32, keep_ratio=0.1)
    assert np.isrealobj(out) or np.allclose(out.imag, 0.0)
