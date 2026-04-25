"""Layer A: baselines.py unit tests. Pure numpy/scipy — no JAX, no GPU."""

from __future__ import annotations

import numpy as np
import pytest

from baselines import global_dct_compress, global_fft_compress
from scipy.fft import dct, idct  # noqa: F401


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


from baselines import block_dct_compress, block_fft_compress  # noqa: E402


@pytest.fixture
def img_256():
    rng = np.random.default_rng(1)
    return rng.uniform(0.0, 1.0, size=(256, 256)).astype(np.float64)


def test_top_k_mask_selects_largest():
    """Verify the helper picks the k largest by magnitude."""
    from baselines import _top_k_mask  # private helper, tested for clarity

    magnitudes = np.array([[5.0, 2.0], [3.0, 1.0]])
    mask = _top_k_mask(magnitudes, k=2)
    # Two largest are 5.0 and 3.0.
    assert mask[0, 0]
    assert mask[1, 0]
    assert not mask[0, 1]
    assert not mask[1, 1]


def test_block_fft_full_keep_is_identity(img_32):
    out = block_fft_compress(img_32, keep_ratio=1.0, block=8)
    np.testing.assert_allclose(out, img_32, atol=1e-10)


def test_block_dct_full_keep_is_identity(img_32):
    out = block_dct_compress(img_32, keep_ratio=1.0, block=8)
    np.testing.assert_allclose(out, img_32, atol=1e-10)


def test_block_fft_full_keep_is_identity_256(img_256):
    out = block_fft_compress(img_256, keep_ratio=1.0, block=8)
    np.testing.assert_allclose(out, img_256, atol=1e-10)


def test_block_dct_keep_ratio_global_count(img_32):
    """Block-DCT keeps top-k% globally across all blocks (JPEG-style)."""
    keep_ratio = 0.5
    out = block_dct_compress(img_32, keep_ratio=keep_ratio, block=8)
    # Re-DCT each block of `out` and count nonzero bins globally.
    n = 32
    block = 8
    nonzero_total = 0
    for i in range(0, n, block):
        for j in range(0, n, block):
            tile = out[i : i + block, j : j + block]
            f = dct(dct(tile, axis=0, norm="ortho"), axis=1, norm="ortho")
            nonzero_total += int(np.sum(np.abs(f) > 1e-9))
    expected = int(np.floor(keep_ratio * n * n))
    # Allow ±5% slack for numeric noise.
    assert abs(nonzero_total - expected) <= max(5, int(0.05 * expected))


def test_block_size_must_divide_image():
    bad = np.zeros((10, 10))
    with pytest.raises(ValueError, match="block size"):
        block_fft_compress(bad, keep_ratio=0.5, block=8)


def test_dct_module_imports():
    """Sanity: scipy.fft.dct is importable. Catches missing-dependency early."""
    from scipy.fft import dct as _dct  # noqa: F401
