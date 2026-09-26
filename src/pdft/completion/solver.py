"""Completion as an unrolled, differentiable map.

Model X as k-sparse in the transform domain and alternate sparsity enforcement
with data consistency, from ``X0 = P_Omega Y``:

    X^(t+1) = P_Omega Y + (I - P_Omega) Re S( H_k( A(X^(t)) ) ).

K steps of that is a differentiable map from the transform's parameters to a
reconstructed image; ``solver_for`` builds it for any family from its per-axis
operator, so the iteration is written once.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from .transform import apply_u, separable


def kth_largest(mag, k):
    """The k-th largest entry. A Python int k (training: k fixed for the run)
    goes through top_k; a traced k (evaluation: k = budget_k differs with every
    mask) through a sort and a dynamic index, so a solver compiles once per
    shape rather than once per (image, budget). The two thresholds are identical."""
    flat = mag.reshape(-1)
    if isinstance(k, int):
        return jax.lax.top_k(flat, k)[0][-1]
    return jnp.sort(flat)[flat.size - k]


def hard_k(C, k):
    """Keep the k largest-magnitude entries; straight-through in the backward pass.

    The support mask is piecewise constant, so differentiating it kills the
    selection pathway; holding it fixed lets gradient flow through the retained
    *values*, which is the pathway that carries signal.
    """
    mag = jnp.abs(C)
    return C * jax.lax.stop_gradient((mag >= kth_largest(mag, k)).astype(C.dtype))


def soft_k(C, k):
    """Soft threshold at the k-th largest magnitude, the level held fixed in the
    backward pass so the shrinkage, not the level, carries the gradient."""
    mag = jnp.abs(C)
    lam = jax.lax.stop_gradient(kth_largest(mag, k))
    return C * (jnp.maximum(mag - lam, 0.0) / jnp.maximum(mag, 1e-12)).astype(C.dtype)


_THRESH = {"hard": hard_k, "soft": soft_k}


def iht(analysis_fn, synthesis_fn, Y, obs, k, K: int, mode: str = "hard", remat: bool = True):
    """K unrolled steps for any exact (analysis, synthesis) pair. ``remat``
    rematerialises each step in the backward pass: without it the K steps keep
    every gate output alive, tens of GB at n = 9, K = 30."""
    thresh = _THRESH[mode]
    X0 = jnp.where(obs, Y, 0.0)

    def step(X, _):
        return jnp.where(obs, Y, jnp.real(synthesis_fn(thresh(analysis_fn(X), k)))), None

    return jax.lax.scan(jax.checkpoint(step) if remat else step, X0, None, length=K)[0]


def solver_for(apply):
    """The jitted K-step solver of a per-axis operator:
    ``reconstruct(pr, pc, Y, obs, k, K, mode="hard", remat=True)`` with Y the
    zero-filled observation. ``k`` is traced (see ``kth_largest``), so a new
    budget does not recompile."""
    analysis, synthesis = separable(apply)

    @functools.partial(jax.jit, static_argnames=("K", "mode", "remat"))
    def reconstruct(pr, pc, Y, obs, k, K, mode="hard", remat=True):
        return iht(
            lambda X: analysis(X, pr, pc), lambda C: synthesis(C, pr, pc), Y, obs, k, K, mode, remat
        )

    return reconstruct


def batched(reconstruct):
    """A solver vmapped over a leading image axis. ``k`` is shared, so top_k
    never runs over the whole batch."""

    def reconstruct_batch(pr, pc, Y, obs, *args, **kwargs):
        return jax.vmap(lambda y, o: reconstruct(pr, pc, y, o, *args, **kwargs))(Y, obs)

    return reconstruct_batch


reconstruct = solver_for(apply_u)
reconstruct_batch = batched(reconstruct)
