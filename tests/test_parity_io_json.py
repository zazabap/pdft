"""Cross-language JSON serialization parity.

Verifies that a QFTBasis saved by Julia can be loaded by Python with
an identical basis_hash and identical ft_mat output. This is the
"bidirectional JSON" exit criterion from Phase 2 of the spec.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pdft.bases.base import QFTBasis, bases_allclose
from pdft.io_json import basis_hash, basis_to_dict, load_basis
from pdft.bases.circuit.qft import ft_mat, qft_code

GOLDENS = Path(__file__).parent.parent / "reference" / "goldens"


def test_python_loads_julia_saved_basis_and_hashes_match():
    loaded = load_basis(GOLDENS / "qft_basis_trained.json")
    julia_hash = (GOLDENS / "basis_roundtrip_hash.txt").read_text().strip()
    assert basis_hash(loaded) == julia_hash, (
        "Python's basis_hash differs from Julia's. The canonical string form "
        "(float formatting, column-major iteration, field order) must match "
        "bit-for-bit."
    )


def test_julia_saved_basis_applied_in_python_reproduces_julia_ft_mat():
    g = np.load(GOLDENS / "basis_roundtrip.npz")
    loaded = load_basis(GOLDENS / "qft_basis_trained.json")

    m, n = int(g["m"][0]), int(g["n"][0])
    pic = jnp.asarray(g["pic"])
    code_fwd, _ = qft_code(m, n)
    fwd_py = ft_mat(loaded.tensors, code_fwd, m, n, pic)

    np.testing.assert_allclose(np.asarray(fwd_py), g["fwd"], atol=1e-10)


def test_roundtrip_python_to_python_preserves_tensors():
    # Sanity: even just the Python side must round-trip losslessly.
    b = QFTBasis(m=2, n=2)
    d = basis_to_dict(b)
    from pdft.io_json import dict_to_basis

    reloaded = dict_to_basis(d)
    assert bases_allclose(b, reloaded, atol=1e-14)
    assert basis_hash(b) == basis_hash(reloaded)
