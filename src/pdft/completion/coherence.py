"""Coherence with the pixel basis, for transforms given as closures or matrices.

``mu(U) = N max_ij |U_ij|^2`` in [1, N]. It is 1 at the Fourier point, maximal
incoherence with the pixel basis and the most favourable case for recovery from
pointwise samples; it is N for an atom living on a single pixel, invisible to
any sample set that misses it.

:mod:`pdft.coherence` defines the same quantity for a basis *object* (a tensor
list contracted by einsum) and returns Python floats. This module is its
counterpart for the representation the completion subpackage carries --- gates
applied directly to the image by a closure, and no matrix formed unless one is
asked for --- and it keeps every function traceable, because ``train`` can add
``lam_mu * mu`` to the loss inside ``jax.jit``. The two agree to round-off (a
test pins it): the operator differs between representations; mu does not.

Proposition 1 of the completion paper is why any of this matters. If every
gate is diagonal except exactly one Hadamard per wire, then ``|U_ij| = N^{-1/2}``
for every parameter value, so mu == 1 identically over the whole parameter
space and sqrt(N) U is a complex Hadamard matrix. ``certify_flat_modulus``
checks that as a property of the model rather than of the point it happens to
sit at, which is what the paper claims and what training must not be able to
break. (:func:`pdft.coherence.certify_flat_modulus` gives the same guarantee
for a basis object, structurally, from which tensors are left trainable.)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

__all__ = [
    "certify_flat_modulus",
    "coherence",
    "dense_operator",
    "flat_modulus_deviation",
    "is_flat_modulus",
]


def dense_operator(apply_fn, n: int, dtype=jnp.complex128) -> jnp.ndarray:
    """Form the 2^n x 2^n matrix of a transform that is normally never formed.

    ``apply_fn`` maps an array to its transform along axis 0. Costs O(N^2 log N)
    and is a diagnostic only: every code path in the subpackage applies gates
    directly to the image instead.
    """
    return apply_fn(jnp.eye(2**n, dtype=dtype))


def coherence(U: jnp.ndarray) -> jnp.ndarray:
    """``mu(U) = N max_ij |U_ij|^2``, in [1, N]. 1 is maximal incoherence."""
    return U.shape[0] * jnp.max(jnp.abs(U) ** 2)


def flat_modulus_deviation(U: jnp.ndarray) -> jnp.ndarray:
    """``max_ij | |U_ij| - N^{-1/2} |``, the residual of Proposition 1.

    Reported rather than thresholded where a number is quoted: it says how
    exactly the guarantee holds, not merely that it does.
    """
    return jnp.max(jnp.abs(jnp.abs(U) - U.shape[0] ** -0.5))


def is_flat_modulus(U: jnp.ndarray, atol: float = 1e-12) -> bool:
    """True if ``|U_ij| = N^{-1/2}`` everywhere, i.e. sqrt(N) U is complex Hadamard."""
    return bool(flat_modulus_deviation(U) <= atol)


def certify_flat_modulus(
    apply_fn, n: int, sampler, trials: int = 8, atol: float = 1e-12, seed: int = 0
) -> dict:
    """Check Proposition 1 as a property of the model, not of one point.

    ``sampler(rng)`` draws a parameter value; ``apply_fn(params)`` returns the
    transform-application closure for it. The certificate holds when every
    draw is flat-modulus, which is what distinguishes "this basis happens to
    be incoherent" from "this family cannot leave the complex Hadamard set".

    Returns the verdict, the worst deviation and the worst mu seen, so a
    failure says how far it drifted rather than only that it did.
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
