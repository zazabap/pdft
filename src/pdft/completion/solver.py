"""Completion as an unrolled, differentiable map, for any transform family.

Model the image as ``k``-sparse in the transform domain and alternate sparsity
enforcement with data consistency from ``X0 = P_Omega Y``:

    X^(t+1) = P_Omega Y + (I - P_Omega) Re S( H_k( A(X^(t)) ) ).

``K`` steps of that is a differentiable map from the transform's parameters to
a reconstructed image. ``solver_for`` builds it for any per-axis operator, so
the iteration is written once.
"""

from __future__ import annotations

import functools
from collections.abc import Callable

import jax
import jax.numpy as jnp

from .transform import separable

Array = jax.Array


def kth_largest(mag: Array, k) -> Array:
    """The ``k``-th largest entry of ``mag``.

    A Python int ``k`` (training, where ``k`` is fixed for the run) goes
    through ``top_k``; a traced ``k`` (evaluation, where ``k = budget_k``
    differs with every mask) through a sort and a dynamic index, so a solver
    compiles once per shape rather than once per image and budget. The two
    thresholds are identical.
    """
    flat = mag.reshape(-1)
    if isinstance(k, int):
        return jax.lax.top_k(flat, k)[0][-1]
    return jnp.sort(flat)[flat.size - k]


def hard_k(C: Array, k) -> Array:
    """Keep the ``k`` largest-magnitude entries, straight-through in the backward pass.

    The support mask is piecewise constant, so differentiating it kills the
    selection pathway; holding it fixed lets the gradient flow through the
    retained values, the pathway that carries signal.
    """
    mag = jnp.abs(C)
    return C * jax.lax.stop_gradient((mag >= kth_largest(mag, k)).astype(C.dtype))


def soft_k(C: Array, k) -> Array:
    """Soft-threshold at the ``k``-th largest magnitude, the level held fixed in the backward pass."""
    mag = jnp.abs(C)
    lam = jax.lax.stop_gradient(kth_largest(mag, k))
    return C * (jnp.maximum(mag - lam, 0.0) / jnp.maximum(mag, 1e-12)).astype(C.dtype)


THRESHOLDS = {"hard": hard_k, "soft": soft_k}


def iht(
    analysis_fn: Callable,
    synthesis_fn: Callable,
    Y: Array,
    obs: Array,
    k,
    K: int,
    mode: str = "hard",
    remat: bool = True,
) -> Array:
    """``K`` unrolled steps for any exact analysis and synthesis pair.

    ``remat`` rematerialises each step in the backward pass; without it the
    ``K`` steps keep every gate output alive, tens of GB at ``n = 9, K = 30``.
    """
    thresh = THRESHOLDS[mode]
    X0 = jnp.where(obs, Y, 0.0)

    def step(X, _):
        return jnp.where(obs, Y, jnp.real(synthesis_fn(thresh(analysis_fn(X), k)))), None

    return jax.lax.scan(jax.checkpoint(step) if remat else step, X0, None, length=K)[0]


def solver_for(apply: Callable) -> Callable:
    """The jitted ``K``-step solver of a per-axis operator.

    Returns ``reconstruct(pr, pc, Y, obs, k, K, mode="hard", remat=True)``
    with ``Y`` the zero-filled observation. ``k`` is traced (see
    ``kth_largest``), so a new budget does not recompile.
    """
    analysis, synthesis = separable(apply)

    @functools.partial(jax.jit, static_argnames=("K", "mode", "remat"))
    def reconstruct(
        pr, pc, Y: Array, obs: Array, k, K: int, mode: str = "hard", remat: bool = True
    ) -> Array:
        """``K`` solver steps from the zero-filled observation ``Y`` under the mask ``obs``."""
        return iht(
            lambda X: analysis(X, pr, pc), lambda C: synthesis(C, pr, pc), Y, obs, k, K, mode, remat
        )

    return reconstruct


def batched(reconstruct: Callable) -> Callable:
    """A solver vmapped over a leading image axis; ``k`` is shared, so ``top_k`` never spans the batch."""

    def reconstruct_batch(pr, pc, Y: Array, obs: Array, *args, **kwargs) -> Array:
        """The solver over a leading image axis of ``Y`` and ``obs``."""
        return jax.vmap(lambda y, o: reconstruct(pr, pc, y, o, *args, **kwargs))(Y, obs)

    return reconstruct_batch
