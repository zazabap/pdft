"""Completion as an unrolled, differentiable map.

Model X as k-sparse in the transform domain and alternate sparsity enforcement
with data consistency, from X0 = P_Omega Y:

    X^(t+1) = P_Omega Y + (I - P_Omega) Re S_theta( H_k( A_theta(X^(t)) ) ).

K steps of that is a differentiable map from the transform's parameters to a
reconstructed image, which is what makes the task objective of
:mod:`pdft.completion.training` trainable.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from .transform import analysis, synthesis


def kth_largest(mag: jnp.ndarray, k) -> jnp.ndarray:
    """The k-th largest entry.

    A Python int k (training, where k is fixed for the run) goes through
    top_k; a traced k (evaluation, where k = budget_k differs with every mask's
    observed count) through a full sort and a dynamic index, so the K-step
    solver compiles once per shape rather than once per (image, budget). The
    two thresholds are identical and the first iterations agree to the bit;
    after hundreds of steps float32 trajectories can part by the usual
    ~0.05 dB.
    """
    flat = mag.reshape(-1)
    if isinstance(k, int):
        return jax.lax.top_k(flat, k)[0][-1]
    return jnp.sort(flat)[flat.size - k]


def hard_k(C: jnp.ndarray, k: int) -> jnp.ndarray:
    """Keep the k largest-magnitude entries; straight-through in the backward pass.

    H_k(C) = C * M(C) with M the support indicator. M is piecewise constant, so
    dM/dC = 0 almost everywhere and differentiating it naively kills the
    selection pathway entirely. Holding M fixed lets gradient flow through the
    retained *values*, which is the pathway that carries signal.
    """
    mag = jnp.abs(C)
    keep = jax.lax.stop_gradient((mag >= kth_largest(mag, k)).astype(C.dtype))
    return C * keep


def soft_k(C: jnp.ndarray, k: int) -> jnp.ndarray:
    """Soft threshold at the k-th largest magnitude: weakly differentiable.

    The alternative resolution of the same problem. lambda is held fixed in the
    backward pass so the shrinkage, not the level, carries the gradient.
    """
    mag = jnp.abs(C)
    lam = jax.lax.stop_gradient(kth_largest(mag, k))
    scale = jnp.maximum(mag - lam, 0.0) / jnp.maximum(mag, 1e-12)
    return C * scale.astype(C.dtype)


_THRESH = {"hard": hard_k, "soft": soft_k}


def iht(analysis_fn, synthesis_fn, Y, obs, k: int, K: int, mode: str = "hard", remat: bool = True):
    """K unrolled steps for any exact (analysis, synthesis) pair --- the one
    iteration every transform family runs.

    A family plugs in as two closures over its own parameters;
    :mod:`pdft.completion.unroll` carries the nested-schedule variant of this
    scan and is the one deliberate exception.
    """
    thresh = _THRESH[mode]
    X0 = jnp.where(obs, Y, 0.0)

    def step(X, _):
        C = thresh(analysis_fn(X), k)
        R = jnp.real(synthesis_fn(C))
        return jnp.where(obs, Y, R), None

    # Without rematerialisation the K unrolled steps keep every gate's output
    # alive for the backward pass, which is tens of GB at n=9, K=30.
    body = jax.checkpoint(step) if remat else step
    X, _ = jax.lax.scan(body, X0, None, length=K)
    return X


@functools.partial(jax.jit, static_argnames=("n", "K", "mode", "remat"))
def reconstruct(
    theta_r: jnp.ndarray,
    theta_c: jnp.ndarray,
    Y: jnp.ndarray,
    obs: jnp.ndarray,
    n: int,
    k,
    K: int,
    mode: str = "hard",
    remat: bool = True,
) -> jnp.ndarray:
    """K unrolled steps on one image. Y is the zero-filled observation. k is
    traced (see kth_largest), so a new budget does not recompile."""
    return iht(
        lambda X: analysis(X, theta_r, theta_c, n),
        lambda C: synthesis(C, theta_r, theta_c, n),
        Y,
        obs,
        k,
        K,
        mode,
        remat,
    )


def reconstruct_batch(theta_r, theta_c, Y, obs, n, k, K, mode="hard", remat=True):
    """vmap over a leading batch axis. k is per-image, so this cannot be
    expressed by broadcasting alone --- top_k would run over the whole batch."""
    f = functools.partial(reconstruct, n=n, k=k, K=K, mode=mode, remat=remat)
    return jax.vmap(f, in_axes=(None, None, 0, 0))(theta_r, theta_c, Y, obs)


def evaluate_theta(params, images, n, p, frac, K, seed, mode="hard"):
    """Held-out PSNR of a phase-only basis under the shared protocol."""
    from .protocol import evaluate

    return evaluate(
        lambda Y, obs, k: reconstruct(
            params["r"], params["c"], Y, obs, n, k, K, mode=mode, remat=False
        ),
        images,
        p,
        frac,
        seed,
    )
