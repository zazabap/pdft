"""Parity tests for compression against Julia goldens."""
from pathlib import Path

import numpy as np

from pdft.basis import QFTBasis
from pdft.compression import compress_with_k, recover

GOLDENS = Path(__file__).parent.parent / "reference" / "goldens"


def test_compress_with_k_values_match_julia_where_indices_overlap():
    """Compressed values agree with Julia at every co-kept index.

    The *selected set* of k indices may differ in 1-2 entries when
    multiple coefficients have near-identical magnitude — Python's
    argpartition and Julia's partialsortperm use different tie-breaking
    strategies. The pixelwise recovery test below verifies the
    downstream outcome matches regardless.
    """
    g = np.load(GOLDENS / "compression_roundtrip_4x4.npz")
    k = int(g["k"][0])
    image = np.asarray(g["image"])

    basis = QFTBasis(m=2, n=2)
    c = compress_with_k(basis, image, k=k)

    py_map = {idx: complex(re, im) for idx, re, im in zip(c.indices, c.values_real, c.values_imag)}
    jl_map = {int(idx): complex(float(re), float(im)) for idx, re, im in
              zip(g["indices"], g["values_real"], g["values_imag"])}
    overlap = set(py_map) & set(jl_map)
    assert len(overlap) >= k - 2, f"only {len(overlap)} indices overlap, expected >= {k - 2}"
    for idx in overlap:
        assert abs(py_map[idx] - jl_map[idx]) < 1e-10


def test_recover_matches_julia_pixelwise():
    g = np.load(GOLDENS / "compression_roundtrip_4x4.npz")
    k = int(g["k"][0])
    image = np.asarray(g["image"])
    basis = QFTBasis(m=2, n=2)
    c = compress_with_k(basis, image, k=k)
    recovered = recover(basis, c)
    np.testing.assert_allclose(recovered, g["recovered"], atol=1e-8)
