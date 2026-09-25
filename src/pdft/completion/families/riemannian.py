"""A general unitary basis trained by Riemannian optimization on U(N).

This is the adaptive competitor the paper has to beat, and the control that
decides whether pinned coherence is load-bearing or merely decorative.

Rather than restricting the transform to the FFT-factorised family, take
U_r, U_c in U(N) as free unitary matrices and optimise them on the manifold:
project the Euclidean gradient onto the tangent space at U, then retract with
a Cayley transform, which preserves U^H U = I exactly [Wen & Yin 2013; Li,
Fuxin & Todorovic 2020].

The two models are then identical in every respect except the constraint set:

                     phase-only           free unitary
    parameters       n(n-1) = 72          2 * 2N^2 = 1,048,576 real
    transform cost   O(N log N)           O(N^2)
    optimiser        Adam, unconstrained  Cayley SGD with momentum
    coherence        mu = 1 always        free to drift

Both start at the DFT, see the same images, masks, solver, depth and budget.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
import numpy as np

from ..coherence import coherence
from ..solver import iht
from ..transform import complex_dtype

# --------------------------------------------------------------------------
# transform with explicit matrices (same convention as transform.analysis)


def analysis_mat(X, Ur, Uc):
    """A(X) = Ur^H X conj(Uc)."""
    return jnp.einsum("ia,ij,jb->ab", jnp.conj(Ur), X.astype(Ur.dtype), jnp.conj(Uc))


def synthesis_mat(C, Ur, Uc):
    """S(C) = Ur C Uc^T, the exact inverse of analysis_mat for unitary Ur, Uc."""
    return jnp.einsum("ai,ij,bj->ab", Ur, C, Uc)


def dft_matrix(N, dtype=jnp.complex128):
    """The starting point, as an explicit matrix (the QFT sign convention)."""
    j = jnp.arange(N)
    return jnp.exp(2j * jnp.pi * j[:, None] * j[None, :] / N).astype(dtype) / jnp.sqrt(N)


# --------------------------------------------------------------------------
# solver, identical to solver.reconstruct but matrix-valued


@functools.partial(jax.jit, static_argnames=("k", "K", "mode", "remat"))
def reconstruct_mat(Ur, Uc, Y, obs, k, K, mode="hard", remat=True):
    """K unrolled solver steps with explicit matrices. The matrices take the
    image's precision (protocol.EVAL_DTYPE hands a float32 Y): a complex128 U
    against a float32 carry changes the scan's carry type and jax refuses it."""
    Ur, Uc = Ur.astype(complex_dtype(Y)), Uc.astype(complex_dtype(Y))
    return iht(
        lambda X: analysis_mat(X, Ur, Uc),
        lambda C: synthesis_mat(C, Ur, Uc),
        Y,
        obs,
        k,
        K,
        mode,
        remat,
    )


def evaluate_unitary(U, images, p, frac, K, seed, mode="hard"):
    """Held-out PSNR of a free-unitary basis under the shared protocol."""
    from ..protocol import evaluate

    return evaluate(
        lambda Y, obs, k: reconstruct_mat(U["r"], U["c"], Y, obs, k, K, mode=mode, remat=False),
        images,
        p,
        frac,
        seed,
    )


# --------------------------------------------------------------------------
# Riemannian machinery


def skew(U, G):
    """Project the Euclidean gradient onto the tangent space at U in U(N).

    A = U^H G - G^H U is skew-Hermitian, so exp(-tau A) and its Cayley
    approximant are unitary and U exp(-tau A) stays on the manifold.
    """
    A = jnp.conj(U).T @ G - jnp.conj(G).T @ U
    return 0.5 * (A - jnp.conj(A).T)  # numerical symmetrisation


def cayley(U, A, tau):
    """Retract: U <- U (I + tau/2 A)^{-1} (I - tau/2 A). Exactly unitary,
    and to first order U (I - tau A), so tau > 0 descends along A."""
    N = U.shape[0]
    eye = jnp.eye(N, dtype=U.dtype)
    return U @ jnp.linalg.solve(eye + 0.5 * tau * A, eye - 0.5 * tau * A)


def train_unitary(
    images,
    k,
    K=100,
    p=0.10,
    steps=200,
    lr=0.05,
    momentum=0.9,
    mode="hard",
    batch=2,
    seed=0,
    log_every=25,
    verbose=True,
):
    """Cayley SGD with momentum on (Ur, Uc), initialised at the DFT."""
    # The Cayley solve accumulates roundoff; in complex64 the unitarity
    # residual reaches 7e-4 within ten steps, which would make "exactly on the
    # manifold" false and the comparison against the circuit (exact by
    # construction) unfair. Double precision holds it near machine epsilon.
    images = jnp.asarray(images, dtype=jnp.float64)
    N = images.shape[-1]
    U = {"r": dft_matrix(N), "c": dft_matrix(N)}
    mom = {a: jnp.zeros_like(v) for a, v in U.items()}

    def loss(U, X, obs):
        f = functools.partial(reconstruct_mat, k=k, K=K, mode=mode, remat=True)
        Xh = jax.vmap(f, in_axes=(None, None, 0, 0))(U["r"], U["c"], X * obs, obs)
        return jnp.mean((Xh - X) ** 2)

    @jax.jit
    def step(U, mom, X, obs, tau):
        v, G = jax.value_and_grad(loss)(U, X, obs)
        newU, newm = {}, {}
        for a in ("r", "c"):
            # jax.grad on a real loss of a complex input returns the conjugate
            # Wirtinger derivative; conjugating recovers the Euclidean gradient.
            A = skew(U[a], jnp.conj(G[a]))
            newm[a] = momentum * mom[a] + A
            # Descent is cayley(U, A, +tau) ~ U (I - tau A): A = U^H G - G^H U
            # is the Riemannian gradient's generator, so the step must
            # subtract it.
            newU[a] = cayley(U[a], newm[a], tau)
        return newU, newm, v

    rng = np.random.default_rng(seed)
    history = []
    for it in range(steps):
        idx = rng.choice(len(images), size=min(batch, len(images)), replace=False)
        X = images[idx]
        obs = jnp.asarray(rng.random(X.shape) < p)
        U, mom, v = step(U, mom, X, obs, lr)
        rec = {"step": it, "loss": float(v)}
        if it % log_every == 0 or it == steps - 1:
            rec["unitarity"] = float(
                jnp.abs(jnp.conj(U["r"]).T @ U["r"] - jnp.eye(N, dtype=U["r"].dtype)).max()
            )
            rec["mu_r"] = float(coherence(U["r"]))
            rec["mu_c"] = float(coherence(U["c"]))
            if verbose:
                print(
                    f"  {it:4d}  loss {rec['loss']:.6e}  "
                    f"mu {rec['mu_r']:.3f}/{rec['mu_c']:.3f}  "
                    f"|U^H U - I| {rec['unitarity']:.2e}",
                    flush=True,
                )
        history.append(rec)
    return U, history
