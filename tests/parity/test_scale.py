"""Scale parity: QFT 16x16 (m=n=4) — verify parity holds at larger sizes."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pdft.bases.circuit.qft import ft_mat, qft_code

GOLDENS = Path(__file__).resolve().parent.parent.parent / "reference" / "goldens"


def test_qft_code_16x16_tensors_match_julia():
    g = np.load(GOLDENS / "qft_code_16x16.npz")
    n_tensors = int(g["n_tensors"])

    _code, tensors = qft_code(4, 4)
    assert len(tensors) == n_tensors
    for i, t in enumerate(tensors):
        np.testing.assert_allclose(np.asarray(t), g[f"tensor_{i}"], atol=1e-12)


def test_ft_mat_16x16_matches_julia():
    g = np.load(GOLDENS / "qft_code_16x16.npz")
    code, tensors = qft_code(4, 4)
    pic = jnp.asarray(g["pic"])
    out = ft_mat(tensors, code, 4, 4, pic)
    np.testing.assert_allclose(np.asarray(out), g["fwd"], atol=1e-10)
