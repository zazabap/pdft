"""Coherence with the pixel basis, for operators given as closures or matrices.

``mu(U) = N max_ij |U_ij|^2`` lies in ``[1, N]``. It is 1 at the Fourier
point, maximal incoherence with the pixel basis and the most favourable case
for recovery from pointwise samples; it is ``N`` for an atom living on a single
pixel, invisible to any sample set that misses it.

:mod:`pdft.coherence` defines the same quantity for a basis object and returns
Python floats. This module is its counterpart for operators applied to the
image by a closure, and it keeps every function traceable so a trainer can add
``lam_mu * mu`` to a loss inside ``jax.jit``. The two agree to round-off.

Proposition 1 of the completion paper is why any of this matters: if every
gate is diagonal except exactly one Hadamard per wire, then
``|U_ij| = N^{-1/2}`` for every parameter value, so ``mu == 1`` identically and
``sqrt(N) U`` is a complex Hadamard matrix. ``certify_flat_modulus`` checks that
as a property of the model rather than of the point it happens to sit at.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

Array = jax.Array

__all__ = [
    "certify_flat_modulus",
    "coherence",
    "dense_operator",
    "flat_modulus_deviation",
    "is_flat_modulus",
]


def dense_operator(apply_fn: Callable, n: int, dtype=jnp.complex128) -> Array:
    """The ``2^n x 2^n`` matrix of a transform that is normally never formed.

    ``apply_fn`` maps an array to its transform along axis 0. Costs
    ``O(N^2 log N)`` and is a diagnostic only.
    """
    return apply_fn(jnp.eye(2**n, dtype=dtype))


def coherence(U: Array) -> Array:
    """``mu(U) = N max_ij |U_ij|^2`` in ``[1, N]``; 1 is maximal incoherence."""
    return U.shape[0] * jnp.max(jnp.abs(U) ** 2)


def flat_modulus_deviation(U: Array) -> Array:
    """``max_ij | |U_ij| - N^{-1/2} |``, the residual of Proposition 1.

    Reported rather than thresholded wherever a number is quoted: it says how
    exactly the guarantee holds, not merely that it does.
    """
    return jnp.max(jnp.abs(jnp.abs(U) - U.shape[0] ** -0.5))


def is_flat_modulus(U: Array, atol: float = 1e-12) -> bool:
    """True if ``|U_ij| = N^{-1/2}`` everywhere, i.e. ``sqrt(N) U`` is complex Hadamard."""
    return bool(flat_modulus_deviation(U) <= atol)


def certify_flat_modulus(
    apply_fn: Callable,
    n: int,
    sampler: Callable,
    trials: int = 8,
    atol: float = 1e-12,
    seed: int = 0,
) -> dict:
    """Check Proposition 1 as a property of the model, not of one point.

    ``sampler(rng)`` draws a parameter value and ``apply_fn(params)`` returns
    the transform closure for it. The certificate holds when every draw is
    flat-modulus, which is what distinguishes a basis that happens to be
    incoherent from a family that cannot leave the complex Hadamard set.
    Returns the verdict, the worst deviation and the worst ``mu`` seen, so a
    failure says how far it drifted.
    """
    rng = np.random.default_rng(seed)
    worst_dev, worst_mu = 0.0, 0.0
    for _ in range(trials):
        U = dense_operator(apply_fn(sampler(rng)), n)
        worst_dev = max(worst_dev, float(flat_modulus_deviation(U)))
        worst_mu = max(worst_mu, float(coherence(U)))
    return {
        "holds": worst_dev <= atol,
        "worst_deviation": worst_dev,
        "worst_mu": worst_mu,
        "trials": trials,
        "n": n,
    }
