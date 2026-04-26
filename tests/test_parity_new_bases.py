"""Parity tests for EntangledQFT / TEBD / MERA against Julia goldens."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from pdft.bases.circuit.entangled_qft import entangled_qft_code
from pdft.bases.circuit.mera import mera_code
from pdft.bases.circuit.tebd import tebd_code

GOLDENS = Path(__file__).parent.parent / "reference" / "goldens"


def _load(name):
    return np.load(GOLDENS / name)


def test_entangled_qft_ft_matches_julia():
    g = _load("entangled_qft_4x4.npz")
    entangle_phases = [float(x) for x in g["entangle_phases"]]
    m, n = 2, 2
    code_fwd, tensors, n_ent = entangled_qft_code(m, n, entangle_phases=entangle_phases)
    assert n_ent == int(g["n_entangle"][0])

    pic = jnp.asarray(g["pic"])
    out = code_fwd(*tensors, pic.reshape((2,) * (m + n)))
    fwd = np.asarray(out.reshape(2**m, 2**n))
    np.testing.assert_allclose(fwd, g["fwd"], atol=1e-10)


def test_tebd_ft_matches_julia():
    g = _load("tebd_4x4.npz")
    phases = [float(x) for x in g["phases"]]
    m, n = 2, 2
    code_fwd, tensors, _, _ = tebd_code(m, n, phases=phases)

    pic = jnp.asarray(g["pic"])
    out = code_fwd(*tensors, pic.reshape((2,) * (m + n)))
    fwd = np.asarray(out.reshape(2**m, 2**n))
    np.testing.assert_allclose(fwd, g["fwd"], atol=1e-10)


def test_mera_ft_matches_julia():
    g = _load("mera_4x4.npz")
    phases = [float(x) for x in g["phases"]]
    m, n = 2, 2
    code_fwd, tensors, _, _ = mera_code(m, n, phases=phases)

    pic = jnp.asarray(g["pic"])
    out = code_fwd(*tensors, pic.reshape((2,) * (m + n)))
    fwd = np.asarray(out.reshape(2**m, 2**n))
    np.testing.assert_allclose(fwd, g["fwd"], atol=1e-10)
