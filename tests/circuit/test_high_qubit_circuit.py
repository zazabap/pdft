"""Regression tests for circuits whose flat-einsum representation would
exceed the 52-character a-zA-Z label pool.

The package once built every circuit as a single einsum (a builder called
`build_circuit_einsum`, since removed), which allocated a fresh label per
Hadamard output and two per U4 output. RealRichBasis at inner_m=5,
inner_n=5 needs 2(m + m²) = 60 fresh labels: over the limit, a ValueError
at construction. Circuits are now applied one gate at a time, so there are
no labels to run out of.

These tests pin both:
  - construction succeeds at inner_m=5
  - the forward+inverse roundtrip returns the input image to machine
    precision (a 32×32 image at m=n=5)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from pdft import RealRichBasis, RichBasis


@pytest.mark.parametrize("BasisCls", [RichBasis, RealRichBasis])
def test_basis_constructs_at_m5(BasisCls):
    """Construction must succeed at m=n=5 (10 qubits per block, 60 fresh labels)."""
    basis = BasisCls(m=5, n=5)
    assert basis.m == 5
    assert basis.n == 5


@pytest.mark.parametrize("BasisCls", [RichBasis, RealRichBasis])
def test_basis_forward_inverse_roundtrip_at_m5(BasisCls):
    """Forward then inverse on the initial (untrained) basis must reproduce the input.

    At step 0 the basis is initialised to a known unitary (QFT-like for RichBasis,
    Walsh-Hadamard ⊗ identity for RealRichBasis), so U U^{-1} = I exactly.
    """
    rng = np.random.default_rng(0)
    pic = jnp.asarray(rng.normal(size=(32, 32)).astype(np.float64))

    basis = BasisCls(m=5, n=5)
    transformed = basis.forward_transform(pic)
    recovered = basis.inverse_transform(transformed)

    np.testing.assert_allclose(np.real(recovered), np.asarray(pic), atol=1e-9)


@pytest.mark.parametrize("BasisCls", [RichBasis, RealRichBasis])
def test_basis_forward_preserves_norm_at_m5(BasisCls):
    """Unitary forward circuits must preserve the Frobenius norm of the input."""
    rng = np.random.default_rng(1)
    pic = jnp.asarray(rng.normal(size=(32, 32)).astype(np.float64))

    basis = BasisCls(m=5, n=5)
    transformed = basis.forward_transform(pic)

    norm_in = float(jnp.linalg.norm(pic))
    norm_out = float(jnp.linalg.norm(transformed))
    np.testing.assert_allclose(norm_out, norm_in, rtol=1e-10)
