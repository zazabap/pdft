"""Parity tests for QFT against Julia-generated goldens."""
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pdft.qft import ft_mat, ift_mat, qft_code

GOLDENS = Path(__file__).parent.parent / "reference" / "goldens"


def _load(name: str):
    return np.load(GOLDENS / name)


def test_ft_mat_matches_julia_on_fixed_image():
    """Round-trip match: Python-computed fwd/roundtrip equal Julia's to atol=1e-10.

    The `pic` comes from Julia's `Random.seed!(123); rand(4, 4)`; Python
    reproduces the same circuit output because the QFT gate sequence is
    deterministic given (m, n).
    """
    g = _load("ft_mat_roundtrip.npz")
    code_fwd, tensors = qft_code(2, 2)
    code_inv, _ = qft_code(2, 2, inverse=True)
    pic = jnp.asarray(g["pic"])
    fwd = ft_mat(tensors, code_fwd, 2, 2, pic)
    np.testing.assert_allclose(np.asarray(fwd), g["fwd"], atol=1e-10)
    recovered = ift_mat([jnp.conj(t) for t in tensors], code_inv, 2, 2, fwd)
    np.testing.assert_allclose(np.asarray(recovered), g["roundtrip"], atol=1e-10)
