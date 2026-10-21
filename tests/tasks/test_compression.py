import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pdft.bases.base import QFTBasis
from pdft.color import YCBCR
from pdft.tasks.compression import (
    compress,
    compress_with_k,
    compressed_to_dict,
    compression_stats,
    dict_to_compressed,
    load_compressed,
    recover,
    save_compressed,
)

from ..helpers import BASES


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
    with pytest.raises(
        ValueError, match=r"image has shape \(4, 4\), expected \(8, 4\) or \(C, 8, 4\)"
    ):
        call(np.zeros((4, 4)))
    with pytest.raises(ValueError, match=r"image has shape \(2, 3, 8, 4\), expected"):
        call(np.zeros((2, 3, 8, 4)))


def test_recover_refuses_a_size_the_basis_does_not_have():
    basis = QFTBasis(m=2, n=2)
    compressed = compress_with_k(basis, np.ones((4, 4)), k=3)
    compressed.original_size = (8, 8)
    with pytest.raises(ValueError, match="does not match basis size"):
        recover(basis, compressed)


def test_the_kept_count_is_rounded_to_the_nearest_coefficient():
    """10% of 16 coefficients is 1.6: two are kept, as upstream's `round` keeps them."""
    assert len(compress(QFTBasis(m=2, n=2), _fixed_image(), ratio=0.9).indices) == 2


def _colour_planes(basis, seed=0) -> np.ndarray:
    """A colour image of the basis's size as its three YCbCr planes, ``(3, H, W)``."""
    rgb = np.random.default_rng(seed).random(basis.image_size + (3,))
    return np.asarray(YCBCR.to_planes(rgb))


@pytest.mark.parametrize("case", list(BASES))
def test_a_stack_of_planes_is_compressed_under_one_budget(case):
    """The k kept coefficients are the k largest over all the planes together."""
    basis = BASES[case]()
    planes = _colour_planes(basis)
    k = planes.size // 4
    c = compress_with_k(basis, planes, k=k)
    assert c.original_size == planes.shape and len(c.indices) == k
    # column-major over (C, H, W), channel fastest, as upstream flattens a plane
    coefficients = np.stack([np.asarray(basis.forward_transform(jnp.asarray(p))) for p in planes])
    flat = coefficients.flatten(order="F")
    kept = np.asarray(c.indices) - 1
    rest = np.setdiff1d(np.arange(flat.size), kept)
    assert np.abs(flat[kept]).min() >= np.abs(flat[rest]).max()
    np.testing.assert_allclose(
        np.asarray(c.values_real) + 1j * np.asarray(c.values_imag), flat[kept], atol=1e-12
    )
    assert recover(basis, c).shape == planes.shape
    everything = compress_with_k(basis, planes, k=planes.size)
    np.testing.assert_allclose(recover(basis, everything), planes, atol=1e-10)


def test_the_budget_is_shared_so_a_faint_plane_gets_few_coefficients():
    basis = QFTBasis(m=3, n=3)
    rng = np.random.default_rng(4)
    planes = np.stack([rng.random((8, 8)), 0.01 * rng.random((8, 8)), 0.01 * rng.random((8, 8))])
    c = compress_with_k(basis, planes, k=planes.size // 4)
    per_plane = np.bincount((np.asarray(c.indices) - 1) % 3, minlength=3)
    assert per_plane[0] > per_plane[1] + per_plane[2]
    # the ratio counts every plane
    assert len(compress(basis, planes, ratio=0.9).indices) == round(0.1 * planes.size)


def test_a_stack_of_planes_round_trips_through_json(tmp_path):
    basis = QFTBasis(m=3, n=2)
    planes = _colour_planes(basis)
    c = compress_with_k(basis, planes, k=10)
    d = compressed_to_dict(c)
    assert d["version"] == "1.1" and d["original_channels"] == 3
    assert (d["original_height"], d["original_width"]) == basis.image_size
    loaded = load_compressed(save_compressed(tmp_path / "colour.json", c))
    assert loaded.original_size == (3, 8, 4) and loaded.indices == c.indices
    np.testing.assert_array_equal(recover(basis, loaded), recover(basis, c))
    # one plane stays in upstream's schema
    one = compressed_to_dict(compress_with_k(basis, planes[0], k=3))
    assert one["version"] == "1.0" and "original_channels" not in one


def test_a_document_whose_version_and_channels_disagree_is_refused():
    basis = QFTBasis(m=2, n=2)
    stack = compressed_to_dict(compress_with_k(basis, _colour_planes(basis), k=3))
    plane = compressed_to_dict(compress_with_k(basis, _fixed_image(), k=3))
    assert dict_to_compressed(stack).original_size == (3, 4, 4)
    assert dict_to_compressed(plane).original_size == (4, 4)
    with pytest.raises(ValueError, match="unknown compressed-image schema version '2.0'"):
        dict_to_compressed({**plane, "version": "2.0"})
    with pytest.raises(ValueError, match="schema version 1.1 needs original_channels"):
        dict_to_compressed({k: v for k, v in stack.items() if k != "original_channels"})
    with pytest.raises(ValueError, match="schema version 1.0 has no original_channels"):
        dict_to_compressed({**plane, "original_channels": 3})


def test_compression_stats_count_every_plane():
    basis = QFTBasis(m=2, n=2)
    stats = compression_stats(compress_with_k(basis, _colour_planes(basis), k=12))
    assert stats["original_size"] == (3, 4, 4) and stats["total_coefficients"] == 48
    assert stats["kept_coefficients"] == 12 and stats["compression_ratio"] == 0.75


def test_recover_refuses_planes_of_another_size():
    basis = QFTBasis(m=2, n=2)
    compressed = compress_with_k(basis, _colour_planes(basis), k=3)
    for size in [(3, 8, 8), (2, 3, 4, 4)]:
        compressed.original_size = size
        with pytest.raises(ValueError, match="does not match basis size"):
            recover(basis, compressed)
