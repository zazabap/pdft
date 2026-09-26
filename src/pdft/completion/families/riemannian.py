"""A free unitary basis on U(N), trained by Cayley SGD --- and the Riemannian
machinery every manifold step in the subpackage uses.

The adaptive competitor of the circuit families: take ``U_r, U_c`` in U(N) as
free matrices, project the Euclidean gradient onto the tangent space at U and
retract with a Cayley transform, which keeps ``U^H U = I`` exactly (Wen & Yin
2013). Same images, masks, solver, depth and budget as the circuits; the only
differences are the constraint set (2N^2 reals per axis against n(n-1)/2), the
O(N^2) transform, and a coherence that is free to drift.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp

from ..coherence import coherence
from ..solver import solver_for
from ..training import minibatches, task_loss
from ..transform import apply_dense, separable

analysis_mat, synthesis_mat = separable(apply_dense)
reconstruct_mat = solver_for(apply_dense)


def dft_matrix(N: int, dtype=jnp.complex128) -> jnp.ndarray:
    """The starting point as an explicit matrix, in the QFT sign convention."""
    j = jnp.arange(N)
    return jnp.exp(2j * jnp.pi * j[:, None] * j[None, :] / N).astype(dtype) / jnp.sqrt(N)


def skew(U, G):
    """The Euclidean gradient ``G`` projected onto the tangent space at ``U``:
    ``A = U^H G - G^H U`` (skew-Hermitian, symmetrised numerically), so that
    ``U (I - tau A)`` descends. Batched over leading axes."""
    Uh, Gh = jnp.conj(jnp.swapaxes(U, -1, -2)), jnp.conj(jnp.swapaxes(G, -1, -2))
    A = Uh @ G - Gh @ U
    return 0.5 * (A - jnp.conj(jnp.swapaxes(A, -1, -2)))


def cayley(U, A, tau):
    """Retract: ``U <- U (I + tau/2 A)^{-1} (I - tau/2 A)``. Exactly unitary for
    skew-Hermitian ``A`` and ``U (I - tau A)`` to first order, so ``tau > 0``
    descends along ``skew(U, G)``. Batched over leading axes."""
    eye = jnp.eye(U.shape[-1], dtype=U.dtype)
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
    """Cayley SGD with momentum on ``(U_r, U_c)`` from the DFT. Double precision
    throughout: in complex64 the Cayley solve's unitarity residual reaches 7e-4
    within ten steps, which would make "exactly on the manifold" false."""
    images = jnp.asarray(images, dtype=jnp.float64)
    U = {a: dft_matrix(N) for a, N in zip(("r", "c"), images.shape[-2:])}
    mom = jax.tree.map(jnp.zeros_like, U)
    loss = functools.partial(task_loss, reconstruct_mat, k=k, K=K, mode=mode)

    @jax.jit
    def step(U, mom, X, obs):
        v, G = jax.value_and_grad(loss)(U, X, obs)
        # jax.grad of a real loss in a complex input is the conjugate Wirtinger
        # derivative; conjugating recovers the Euclidean gradient.
        mom = {a: momentum * mom[a] + skew(U[a], jnp.conj(G[a])) for a in U}
        return {a: cayley(U[a], mom[a], lr) for a in U}, mom, v

    history = []
    for it, (X, obs) in enumerate(minibatches(images, steps, batch, p, seed)):
        U, mom, v = step(U, mom, X, obs)
        rec = {"step": it, "loss": float(v)}
        if it % log_every == 0 or it == steps - 1:
            eye = jnp.eye(U["r"].shape[0], dtype=U["r"].dtype)
            rec["unitarity"] = float(jnp.abs(jnp.conj(U["r"]).T @ U["r"] - eye).max())
            rec.update({f"mu_{a}": float(coherence(U[a])) for a in U})
            if verbose:
                print(
                    f"  {it:4d}  loss {rec['loss']:.6e}  mu {rec['mu_r']:.3f}/{rec['mu_c']:.3f}  "
                    f"|U^H U - I| {rec['unitarity']:.2e}",
                    flush=True,
                )
        history.append(rec)
    return U, history
