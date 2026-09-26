import numpy as np
import pytest

import pdft.completion.metrics as M


def test_psnr_values():
    a = np.zeros((8, 8))
    assert M.psnr(a, a) == pytest.approx(150.0)
    assert M.psnr(a, a + 0.1) == pytest.approx(20.0)
    assert M.psnr(a, a + 5.0) == pytest.approx(0.0)  # clipped to [0, 1]


def test_ssim_family_is_one_for_identical_images():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64))
    assert M.ssim(a, a) == pytest.approx(1.0, abs=1e-12)
    assert M.ms_ssim(a, a) == pytest.approx(1.0, abs=1e-12)
    noisy = np.clip(a + 0.2 * rng.standard_normal(a.shape), 0, 1)
    assert 0 < M.ssim(a, noisy) < 1
    assert 0 < M.ms_ssim(a, noisy) < 1
    s = M.score(a, noisy)
    assert set(s) == {"psnr", "ssim", "ms_ssim"}
    assert s["ssim"] == pytest.approx(M.ssim(a, noisy))


def test_gaussian_filter_matches_scipy():
    ndimage = pytest.importorskip("scipy.ndimage")
    rng = np.random.default_rng(1)
    a = rng.random((40, 37))
    ours = M.gaussian_filter(a, 1.5)
    ref = ndimage.gaussian_filter(a, 1.5, truncate=3.5)
    assert np.abs(ours - ref).max() < 1e-12
    assert np.abs(ours.sum() - a.sum()) / a.sum() < 0.01  # blur roughly preserves mass
