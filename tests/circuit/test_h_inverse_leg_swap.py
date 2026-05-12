"""Regression test for the H-gate inverse leg-swap bug.

The stepped-tensordot rewrite of `compile_circuit` (commit
`feat(circuit): stepped tensordot for circuits beyond the 52-label pool`)
forgot to swap the contracted leg of the Hadamard tensor when
``inverse=True``. Combined with `loss._scalar_loss`'s convention of
passing ``conj(tensors)`` to the inverse closure, this means the
inverse path applies ``conj(H)`` instead of the proper adjoint
``conj(H.T) = H†``.

For the default symmetric Hadamard (``[[1, 1], [1, -1]] / sqrt 2``),
``conj(H) = H = H†`` so the bug is invisible. For a *trained* (and
therefore non-symmetric) Hadamard the bug breaks the round-trip:
``T† T x ≠ x``. The U4 gate handler in the same function does the
leg-swap correctly already; this test pins the equivalent contract for
the H gate.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import pdft


def _non_symmetric_unitary(seed: int = 0) -> jnp.ndarray:
    """A 2x2 unitary that is NOT real-symmetric (so conj ≠ adjoint).

    Built as exp(i * theta * X) for a Hermitian X with non-zero off-diagonal
    imag parts. The result is unitary by construction and demonstrably
    non-Hermitian and non-symmetric.
    """
    rng = np.random.default_rng(seed)
    # Random Hermitian generator with complex off-diagonal
    g = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    H = (g + g.conj().T) / 2  # Hermitian
    # exp(i * H) is unitary; for a non-trivial H it is non-symmetric in general
    eigvals, eigvecs = np.linalg.eigh(H)
    U = eigvecs @ np.diag(np.exp(1j * eigvals)) @ eigvecs.conj().T
    return jnp.asarray(U, dtype=jnp.complex128)


def test_qft_inverse_round_trip_with_non_symmetric_h():
    """Round-trip ``T† T x = x`` must hold to machine precision when one or
    more Hadamard tensors are non-symmetric unitaries (the situation that
    arises after Riemannian-Adam training perturbs the gates off the
    initial real-symmetric H).
    """
    m, n = 3, 3
    basis = pdft.QFTBasis(m=m, n=n)

    # Replace the FIRST Hadamard tensor with a non-symmetric unitary.
    # QFTBasis sorts tensors Hadamards-first, so tensors[0..m+n-1] are H slots.
    perturbed = list(basis.tensors)
    U_nonsym = _non_symmetric_unitary(seed=42)
    perturbed[0] = U_nonsym

    # Sanity: the replacement is unitary but not symmetric.
    assert np.allclose(
        np.asarray(U_nonsym) @ np.asarray(U_nonsym).conj().T, np.eye(2), atol=1e-12
    ), "test setup error: replacement matrix must be unitary"
    assert not np.allclose(
        np.asarray(U_nonsym), np.asarray(U_nonsym).T, atol=1e-6
    ), "test setup error: replacement matrix must be non-symmetric"

    perturbed_basis = pdft.QFTBasis(m=m, n=n, tensors=perturbed)

    rng = np.random.default_rng(0)
    x = jnp.asarray(
        rng.standard_normal((2**m, 2**n)) + 1j * rng.standard_normal((2**m, 2**n)),
        dtype=jnp.complex128,
    )

    # Forward must preserve norm (operator is unitary).
    Tx = perturbed_basis.forward_transform(x)
    np.testing.assert_allclose(
        float(jnp.sum(jnp.abs(Tx) ** 2)),
        float(jnp.sum(jnp.abs(x) ** 2)),
        rtol=1e-10,
        err_msg="forward operator is not norm-preserving",
    )

    # Round-trip must recover x to machine precision.
    rt = perturbed_basis.inverse_transform(Tx)
    np.testing.assert_allclose(
        np.asarray(rt),
        np.asarray(x),
        atol=1e-10,
        err_msg="round-trip T† T x != x — H-gate inverse leg-swap bug",
    )


def test_blocked_inverse_round_trip_with_non_symmetric_h():
    """Same regression for BlockedBasis(QFTBasis, ...) — this is the path
    actually hit by the trained_blocked_8 cell whose val loss the bug
    inflated by ~12x.
    """
    inner = pdft.QFTBasis(m=3, n=3)
    perturbed = list(inner.tensors)
    perturbed[0] = _non_symmetric_unitary(seed=7)
    perturbed_inner = pdft.QFTBasis(m=3, n=3, tensors=perturbed)

    blocked = pdft.BlockedBasis(inner=perturbed_inner, block_log_m=2, block_log_n=2)

    rng = np.random.default_rng(1)
    side = 2 ** blocked.m
    x = jnp.asarray(
        rng.standard_normal((side, side)) + 1j * rng.standard_normal((side, side)),
        dtype=jnp.complex128,
    )

    Tx = blocked.forward_transform(x)
    np.testing.assert_allclose(
        float(jnp.sum(jnp.abs(Tx) ** 2)),
        float(jnp.sum(jnp.abs(x) ** 2)),
        rtol=1e-10,
    )

    rt = blocked.inverse_transform(Tx)
    np.testing.assert_allclose(
        np.asarray(rt), np.asarray(x), atol=1e-10,
        err_msg="BlockedBasis round-trip broken — H-gate inverse leg-swap bug",
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_qft_inverse_round_trip_with_non_symmetric_h_random_seeds(seed: int):
    """Robustness: hold across multiple random non-symmetric H replacements."""
    basis = pdft.QFTBasis(m=2, n=2)
    perturbed = list(basis.tensors)
    perturbed[0] = _non_symmetric_unitary(seed=seed)
    perturbed_basis = pdft.QFTBasis(m=2, n=2, tensors=perturbed)

    rng = np.random.default_rng(seed)
    x = jnp.asarray(
        rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4)),
        dtype=jnp.complex128,
    )
    rt = perturbed_basis.inverse_transform(perturbed_basis.forward_transform(x))
    np.testing.assert_allclose(np.asarray(rt), np.asarray(x), atol=1e-10)
