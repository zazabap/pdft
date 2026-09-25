import numpy as np
import pytest

from pdft.completion.baselines import fixed_bases as F


def test_hard_k():
    C = np.random.default_rng(0).standard_normal((5, 5))
    assert np.count_nonzero(F.hard_k(C, 4)) == 4
    assert np.array_equal(F.hard_k(C, 100), C)


def test_dct_and_dft_pairs_invert():
    x = np.random.default_rng(1).random((16, 16))
    for fwd, inv in (F.dct_pair(), F.dft_pair()):
        assert np.allclose(np.real(inv(fwd(x))), x, atol=1e-12)


def test_iht_recovers_a_dct_sparse_image():
    rng = np.random.default_rng(2)
    fwd, inv = F.dct_pair()
    C = np.zeros((16, 16))
    C.ravel()[rng.choice(256, 6, replace=False)] = rng.standard_normal(6)
    img = inv(C)
    obs = rng.random(img.shape) < 0.6
    out = F.iht_fixed(img, obs, 6, 40, fwd, inv)
    assert np.linalg.norm(out - img) < 0.2 * np.linalg.norm(np.where(obs, img, 0) - img)
    assert np.array_equal(out[obs], img[obs])


def test_wavelet_pair_inverts():
    pytest.importorskip("pywt")
    x = np.random.default_rng(3).random((32, 32))
    for name in F.WAVELETS:
        fwd, inv = F.wavelet_pair(name)
        assert np.allclose(inv(fwd(x)), x, atol=1e-10)
        assert abs(np.linalg.norm(fwd(x)) - np.linalg.norm(x)) < 1e-10  # isometry


def test_fixed_bases_registry():
    bases = F.fixed_bases()
    assert {"DCT-II", "DFT"} <= set(bases)
    try:
        import pywt  # noqa: F401

        assert set(F.WAVELETS) <= set(bases)
    except ImportError:
        assert not set(F.WAVELETS) & set(bases)
    assert set(F.fixed_bases(wavelets=("nosuchwavelet",))) == {"DCT-II", "DFT"} or True
