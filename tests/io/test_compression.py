import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pdft.bases.base import QFTBasis
from pdft.io.compression import (
    compress,
    compress_with_k,
    compression_stats,
    load_compressed,
    recover,
    save_compressed,
)


def _fixed_image(m=2, n=2, seed=0):
    return np.asarray(
        jax.random.normal(jax.random.PRNGKey(seed), (2**m, 2**n)).astype(jnp.complex128).real
    ).astype(np.float64)


def test_compress_with_k_keeps_exactly_k():
    basis = QFTBasis(m=2, n=2)
    img = _fixed_image()
    c = compress_with_k(basis, img, k=3)
    assert len(c.indices) == 3
    assert len(c.values_real) == 3
    assert len(c.values_imag) == 3
    assert c.original_size == (4, 4)


def test_compress_with_k_rejects_non_positive():
    basis = QFTBasis(m=2, n=2)
    with pytest.raises(ValueError):
        compress_with_k(basis, _fixed_image(), k=0)


def test_compress_ratio_bounds():
    basis = QFTBasis(m=2, n=2)
    with pytest.raises(ValueError):
        compress(basis, _fixed_image(), ratio=-0.1)
    with pytest.raises(ValueError):
        compress(basis, _fixed_image(), ratio=1.0)


def test_roundtrip_full_k_recovers_image_exactly():
    basis = QFTBasis(m=2, n=2)
    img = _fixed_image()
    c = compress_with_k(basis, img, k=img.size)  # keep everything
    recov = recover(basis, c)
    np.testing.assert_allclose(recov, img, atol=1e-10)


def test_roundtrip_with_truncation_is_approximate():
    basis = QFTBasis(m=2, n=2)
    img = _fixed_image()
    c = compress_with_k(basis, img, k=img.size // 2)
    recov = recover(basis, c)
    # With 50% truncation, recovery should not be exact but should be bounded.
    err = np.linalg.norm(recov - img) / np.linalg.norm(img)
    assert err < 1.0


def test_recover_rejects_hash_mismatch():
    basis = QFTBasis(m=2, n=2)
    img = _fixed_image()
    c = compress_with_k(basis, img, k=4)
    bad_basis = QFTBasis(m=2, n=2, tensors=[t + 1e-6 for t in basis.tensors])
    with pytest.raises(ValueError, match="hash mismatch"):
        recover(bad_basis, c, verify_hash=True)


def test_recover_skips_hash_mismatch_when_disabled():
    basis = QFTBasis(m=2, n=2)
    img = _fixed_image()
    c = compress_with_k(basis, img, k=4)
    bad_basis = QFTBasis(m=2, n=2, tensors=[t + 1e-6 for t in basis.tensors])
    # Should not raise
    recover(bad_basis, c, verify_hash=False)


def test_file_roundtrip(tmp_path):
    basis = QFTBasis(m=2, n=2)
    img = _fixed_image()
    c = compress_with_k(basis, img, k=4)
    p = tmp_path / "compressed.json"
    save_compressed(p, c)
    loaded = load_compressed(p)
    assert loaded.indices == c.indices
    assert loaded.values_real == c.values_real
    assert loaded.original_size == c.original_size


def test_compression_stats():
    basis = QFTBasis(m=2, n=2)
    c = compress_with_k(basis, _fixed_image(), k=4)
    s = compression_stats(c)
    assert s["total_coefficients"] == 16
    assert s["kept_coefficients"] == 4
    assert abs(s["compression_ratio"] - 0.75) < 1e-12
