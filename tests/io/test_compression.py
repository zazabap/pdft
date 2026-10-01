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


def test_both_entry_points_keep_the_same_coefficients_for_the_same_count():
    """`compress` and `compress_with_k` differ only in how the count is chosen."""
    from pdft.io import compressed_to_dict

    basis = QFTBasis(m=3, n=2)
    image = np.random.default_rng(3).normal(size=(8, 4))
    # ratio 0.75 of 32 coefficients keeps 8
    by_ratio = compress(basis, image, ratio=0.75)
    by_count = compress_with_k(basis, image, k=8)
    assert compressed_to_dict(by_ratio) == compressed_to_dict(by_count)
    assert len(by_count.indices) == 8
    # a ratio that would keep none keeps one; a count beyond the size keeps all
    assert len(compress(basis, image, ratio=0.999).indices) == 1
    assert len(compress_with_k(basis, image, k=1000).indices) == 32


@pytest.mark.parametrize("entry", ["compress", "compress_with_k"])
def test_compressing_an_image_of_the_wrong_size_is_refused(entry):
    basis = QFTBasis(m=3, n=2)
    call = {
        "compress": lambda image: compress(basis, image),
        "compress_with_k": lambda image: compress_with_k(basis, image, k=3),
    }[entry]
    with pytest.raises(ValueError, match=r"image shape \(4, 4\) must match basis size \(8, 4\)"):
        call(np.zeros((4, 4)))


def test_recover_refuses_a_size_the_basis_does_not_have():
    basis = QFTBasis(m=2, n=2)
    compressed = compress_with_k(basis, np.ones((4, 4)), k=3)
    compressed.original_size = (8, 8)
    with pytest.raises(ValueError, match="does not match basis size"):
        recover(basis, compressed)
